# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental: cold experts resident on, and computed by, a peer GPU.

A sidecar process on the peer (an RTX PRO 6000 next to the GB300) owns every
layer's cold experts in its VRAM. The two processes share a pinned host buffer
(``/dev/shm``, mapped into both CUDA contexts). Per MoE layer the GB300:

1. writes the batch's activations, compacted-cold route ids and weights to a
   ring slot and publishes a sequence number (release, system scope);
2. runs its hot experts;
3. waits (acquire, system scope) until the peer reports that sequence done,
   then adds the peer's weighted cold output.

Every step is a device kernel, so CUDA graphs capture it. A batch with no cold
route skips the wait; the peer still consumes the slot in order. The wait has a
timeout: a missing peer degrades to dropping cold routes and raises a counter.
"""

from __future__ import annotations

import mmap
import os

import torch
import triton
import triton.language as tl

HEADER = 64          # int64 words in the slot header
MAX_TOKENS = 64
HIDDEN = 5120
TOPK = 6
# Global words: [0] published seq, [1] completed seq, [2] timeouts.
# One slot: header | x fp8 [T,H] | x scale u8 [T,H/32] | ids i32 [T,K] | w f32 [T,K] | out bf16 [T,H]
# One slot suffices: the GB300 waits for completion whenever a layer has cold
# routes, so it never overwrites a slot the peer still has to read.
SLOT = 4096
X_OFF = SLOT + HEADER * 8
XS_OFF = X_OFF + MAX_TOKENS * HIDDEN
IDS_OFF = XS_OFF + MAX_TOKENS * (HIDDEN // 32)
W_OFF = IDS_OFF + MAX_TOKENS * TOPK * 4
OUT_OFF = W_OFF + MAX_TOKENS * TOPK * 4
TOTAL_BYTES = (OUT_OFF + MAX_TOKENS * HIDDEN * 2 + 4095) // 4096 * 4096
PATH = os.environ.get("VLLM_EXP_PEER_SHM", "/dev/shm/vllm_peer_tier")


def open_shared(create: bool) -> mmap.mmap:
    # Open an existing buffer without O_CREAT: fs.protected_regular refuses
    # O_CREAT on another user's file in sticky /dev/shm, even for root.
    try:
        fd = os.open(PATH, os.O_RDWR)
    except FileNotFoundError:
        if not create:
            raise
        fd = os.open(PATH, os.O_RDWR | os.O_CREAT, 0o666)
    if os.fstat(fd).st_size < TOTAL_BYTES:
        os.ftruncate(fd, TOTAL_BYTES)
    buf = mmap.mmap(fd, TOTAL_BYTES)
    os.close(fd)
    return buf


def register(buf: mmap.mmap, device: torch.device) -> tuple[torch.Tensor, int]:
    """Pin the mapping in this process's CUDA context; return (host view, device ptr)."""
    from cuda.bindings import runtime as rt

    host = torch.frombuffer(buf, dtype=torch.uint8)
    pointer = host.data_ptr()
    flags = rt.cudaHostRegisterMapped | rt.cudaHostRegisterPortable
    with torch.cuda.device(device):
        (error,) = rt.cudaHostRegister(pointer, TOTAL_BYTES, flags)
        if error != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostRegister failed: {error}")
        error, dptr = rt.cudaHostGetDevicePointer(pointer, 0)
        if error != rt.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {error}")
    return host, int(dptr)


class _Bytes:
    def __init__(self, pointer: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (pointer, False), "version": 3,
        }


def device_view(dptr: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(_Bytes(dptr, TOTAL_BYTES), device=device)


@triton.jit(do_not_specialize=["LAYER", "T"])
def _publish(WORDS, HEADER_PTR, SEQ, LAYER, T, COUNT):
    """Advance the device sequence, write the header, publish with release."""
    seq = tl.load(SEQ) + 1
    tl.store(SEQ, seq)
    tl.store(HEADER_PTR + 1, LAYER.to(tl.int64))
    tl.store(HEADER_PTR + 2, T.to(tl.int64))
    tl.store(HEADER_PTR + 3, tl.load(COUNT).to(tl.int64))
    tl.store(HEADER_PTR + 0, seq)
    tl.atomic_xchg(WORDS, seq, sem="release", scope="sys")


@triton.jit
def _wait(WORDS, SEQ, COUNT, TIMEOUT: tl.constexpr):
    """Spin (acquire) until the peer completed this sequence; skip without cold routes."""
    if tl.load(COUNT) == 0:
        return
    seq = tl.load(SEQ)
    spins = 0
    done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
    while (done < seq) & (spins < TIMEOUT):
        done = tl.atomic_add(WORDS + 1, 0, sem="acquire", scope="sys")
        spins += 1
    if done < seq:
        tl.atomic_add(WORDS + 2, 1, sem="relaxed", scope="sys")


@triton.jit
def _accumulate(OUT, RESULT, COUNT, N: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(COUNT) == 0:
        return
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    value = tl.load(OUT + offsets, mask=mask).to(tl.float32)
    value += tl.load(RESULT + offsets, mask=mask).to(tl.float32)
    tl.store(OUT + offsets, value.to(OUT.dtype.element_ty), mask=mask)


class PeerTier:
    """GB300 side of the peer tier; one instance serves every layer of the process."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.buf = open_shared(create=True)
        self.host, dptr = register(self.buf, device)
        self.host.zero_()
        shared = device_view(dptr, device)
        self.words = shared[:64].view(torch.int64)
        self.header = shared[SLOT:SLOT + HEADER * 8].view(torch.int64)
        self.x = shared[X_OFF:XS_OFF]
        self.xs = shared[XS_OFF:IDS_OFF]
        self.ids = shared[IDS_OFF:W_OFF].view(torch.int32)
        self.w = shared[W_OFF:OUT_OFF].view(torch.float32)
        self.out = shared[OUT_OFF:OUT_OFF + MAX_TOKENS * HIDDEN * 2].view(torch.bfloat16)
        self.seq = torch.zeros(1, dtype=torch.int64, device=device)
        self.max_tokens = MAX_TOKENS
        self.timeout = int(os.environ.get("VLLM_EXP_PEER_SPINS", "2000000"))

    def run(self, output, x_quant, x_scale, ids, weights, cold_start: int, layer: int,
            hot_call) -> None:
        tokens, topk = ids.shape
        if tokens > MAX_TOKENS or topk != TOPK or x_scale is None:
            raise ValueError("peer tier handles up to 64 tokens of top-6 MXFP8 routes")
        count = (ids >= cold_start).sum(dtype=torch.int32).reshape(1)
        local = torch.where(ids >= cold_start, ids - cold_start, -1).to(torch.int32)
        self.x[: x_quant.numel()].copy_(x_quant.view(torch.uint8).reshape(-1))
        self.xs[: x_scale.numel()].copy_(x_scale.view(torch.uint8).reshape(-1))
        self.ids[: local.numel()].copy_(local.reshape(-1))
        self.w[: weights.numel()].copy_(weights.float().reshape(-1))
        _publish[(1,)](self.words, self.header, self.seq, layer, tokens, count)
        hot_call()
        _wait[(1,)](self.words, self.seq, count, TIMEOUT=self.timeout)
        n = tokens * HIDDEN
        _accumulate[(triton.cdiv(n, 1024),)](output, self.out, count, N=n, BLOCK=1024)
