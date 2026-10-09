"""Compressed MegaMoE weight scales: encode at load, decode the routed experts into a shared scratch before each call.

MegaMoE's SFB loader TMA-loads one 512 B chunk per (expert, k-block, 128-row N-block): 128 int32 words, each word
packing 4 UE8M0 K-group scales (byte j = K-group 4*kb+j). The DeepGEMM SF tensors are int32 [E, N, Kp] with
MN-major strides (N*Kp, 1, N), so in memory they are [E, Kp, N] and every chunk is contiguous.

Each chunk becomes one fixed 80 B slot (20 int32 words) - see phase-0 notes for the statistics:
  word 0      header: base (bits 0-7) | n_exc (bits 8-9, 0..3) | spill (bit 10) | spill index (bits 11-31)
  words 1-16  bit plane, 1 bit per scale; scale p (= 4*row + kgroup, the byte order of the chunk) is bit p%32 of
              word 1 + p//32, and its value is base + bit
  words 17-19 up to 3 exceptions: position p (bits 0-8) | raw byte (bits 16-23), overriding base + bit
A chunk with more than 3 exceptions sets spill and stores its raw 512 B in an overflow tensor [n_spill, 128].
Decoding is bit-exact; base is chosen to maximize the number of scales in {base, base+1} and is at most 254.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

SLOT_WORDS = 20
MAX_EXC = 3


@dataclass
class CompressedSF:
    slots: torch.Tensor     # int32 [E, Kp, NB, SLOT_WORDS]
    overflow: torch.Tensor  # int32 [max(1, n_spill), 128]
    shape: tuple            # (E, N, Kp) of the original SF tensor
    stride: tuple

    def nbytes(self) -> int:
        return self.slots.numel() * 4 + self.overflow.numel() * 4


def _check_layout(sf: torch.Tensor):
    assert sf.dtype == torch.int32 and sf.dim() == 3, (sf.dtype, sf.shape)
    E, N, Kp = sf.shape
    assert N % 128 == 0, N
    assert sf.stride() == (N * Kp, 1, N), f"expected MN-major SF strides, got {sf.stride()} for {tuple(sf.shape)}"
    return E, N, Kp


@torch.no_grad()
def encode(sf: torch.Tensor, chunk_batch: int = 65536) -> CompressedSF:
    """sf: DeepGEMM MegaMoE SF tensor, int32 [E, N, Kp], MN-major. Runs on sf's device."""
    E, N, Kp = _check_layout(sf)
    NB = N // 128
    dev = sf.device
    # [E, Kp, N] contiguous view -> chunks [C, 512] bytes in chunk byte order (word r = row r, byte j = kgroup j).
    chunks = sf.permute(0, 2, 1).contiguous().view(torch.uint8).view(E * Kp * NB, 512)
    C = chunks.shape[0]
    slots = torch.zeros(C, SLOT_WORDS, dtype=torch.int32, device=dev)
    spill_raw = []
    n_spill = 0
    pos = torch.arange(512, device=dev, dtype=torch.int64)
    for s in range(0, C, chunk_batch):
        x = chunks[s:s + chunk_batch].long()                                   # [B, 512]
        B = x.shape[0]
        hist = torch.zeros(B, 257, dtype=torch.int32, device=dev)
        hist.scatter_add_(1, x, torch.ones_like(x, dtype=torch.int32))
        pair = hist[:, :255] + hist[:, 1:256]                                  # count in {v, v+1}, v <= 254
        base = pair.argmax(dim=1)                                              # [B]
        d = x - base[:, None]
        bit = (d == 1)
        exc = (d != 0) & (d != 1)
        n_exc = exc.sum(dim=1)
        # bit plane: 16 words of 32 bits
        words = (bit.view(B, 16, 32).long() << torch.arange(32, device=dev)).sum(dim=2)
        words = torch.where(words >= 2**31, words - 2**32, words)              # to int32 range
        hdr = base | (n_exc.clamp(max=MAX_EXC) << 8)
        spill = n_exc > MAX_EXC
        # exceptions for non-spilled chunks: first MAX_EXC exception positions in order
        key = torch.where(exc, pos, 512 + pos)                                 # exceptions first, stable by position
        first = key.argsort(dim=1)[:, :MAX_EXC]                                # [B, 3]
        ex_pos = first
        ex_val = x.gather(1, first)
        ex_ok = torch.arange(MAX_EXC, device=dev)[None, :] < n_exc.clamp(max=MAX_EXC)[:, None]
        ex_words = torch.where(ex_ok & ~spill[:, None], ex_pos | (ex_val << 16), torch.zeros_like(ex_pos))
        if spill.any():
            idx = spill.nonzero().squeeze(1)
            spill_index = torch.zeros(B, dtype=torch.int64, device=dev)
            spill_index[idx] = torch.arange(n_spill, n_spill + idx.numel(), device=dev)
            hdr = torch.where(spill, (hdr & 0xFF) | (1 << 10) | (spill_index << 11), hdr)
            spill_raw.append(chunks[s:s + chunk_batch][idx].view(torch.int32).clone())
            n_spill += idx.numel()
        assert n_spill < 2**21, n_spill
        slots[s:s + B, 0] = hdr.to(torch.int32)
        slots[s:s + B, 1:17] = words.to(torch.int32)
        slots[s:s + B, 17:20] = ex_words.to(torch.int32)
    overflow = (torch.cat(spill_raw) if spill_raw else torch.zeros(1, 128, dtype=torch.int32, device=dev))
    return CompressedSF(slots.view(E, Kp, NB, SLOT_WORDS), overflow.contiguous(), (E, N, Kp), sf.stride())


@triton.jit
def _decode_kernel(slots, overflow, out, elist, ecount, CPE, SLOT: tl.constexpr, CH: tl.constexpr):
    # Chunks are numbered g = e * CPE + kb * NB + nb (CPE = Kp * NB chunks per expert); both the slots and the
    # output are linear in g, so program (i, t) decodes chunks [t*CH, t*CH + CH) of the i-th routed expert
    # (elist[i], i < ecount) as one [CH, 128] tile.
    i = tl.program_id(0)
    if i >= tl.load(ecount):
        return
    e = tl.load(elist + i)
    c = tl.program_id(1) * CH + tl.arange(0, CH)
    live = c < CPE
    g = e.to(tl.int64) * CPE + c
    rows = tl.arange(0, 128)
    sp = slots + g * SLOT
    hdr = tl.load(sp, mask=live, other=0)
    base = hdr & 0xFF
    # row r's 4 scales are bits 4r..4r+3 of the plane: word 1 + r//8, shift 4*(r%8). Load the 16 words once
    # per chunk and broadcast each to its 8 rows.
    w16 = tl.load(sp[:, None] + 1 + tl.arange(0, 16)[None, :], mask=live[:, None], other=0)
    w = tl.reshape(tl.broadcast_to(w16[:, :, None], (CH, 16, 8)), (CH, 128))
    nib = (w >> ((rows[None, :] % 8) * 4)) & 0xF
    val = base[:, None] * 0x01010101 + ((nib * 0x00204081) & 0x01010101)
    spill = live & (((hdr >> 10) & 1) == 1)
    si = ((hdr >> 11) & 0x1FFFFF).to(tl.int64)
    raw = tl.load(overflow + si[:, None] * 128 + rows[None, :], mask=spill[:, None], other=0)
    val = tl.where(spill[:, None], raw, val)
    tl.store(out + g[:, None] * 128 + rows[None, :], val, mask=live[:, None])
    # Exceptions (n_exc is 0 for spilled chunks): after the tile store, rewrite each excepted word. Word j is
    # rebuilt from its row's base/bits with every exception of the same row applied, so duplicates agree.
    n_exc = (hdr >> 8) & 0x3
    if tl.max(n_exc, axis=0) > 0:
        tl.debug_barrier()
        e0 = tl.load(sp + 17, mask=live & (n_exc > 0), other=0)
        e1 = tl.load(sp + 18, mask=live & (n_exc > 1), other=0)
        e2 = tl.load(sp + 19, mask=live & (n_exc > 2), other=0)
        for j in tl.static_range(3):
            ej = e0 if j == 0 else (e1 if j == 1 else e2)
            r = (ej & 0x1FF) // 4
            wr = tl.load(sp + 1 + r // 8, mask=live & (j < n_exc), other=0)
            word = base * 0x01010101 + ((((wr >> ((r % 8) * 4)) & 0xF) * 0x00204081) & 0x01010101)
            for k in tl.static_range(3):
                ek = e0 if k == 0 else (e1 if k == 1 else e2)
                sh = (ek & 0x3) * 8
                hit = (k < n_exc) & (((ek & 0x1FF) // 4) == r)
                word = tl.where(hit, (word & ~(0xFF << sh)) | (((ek >> 16) & 0xFF) << sh), word)
            tl.store(out + g * 128 + r, word, mask=live & (j < n_exc))


DECODE_CH = 32
DECODE_WARPS = 4


@dataclass
class ExpertList:
    """The routed experts of one MoE call: elist[:count] (unique, ascending), built on the GPU."""
    elist: torch.Tensor   # int32 [E]
    count: torch.Tensor   # int32 [1]
    flags: torch.Tensor   # int32 [E] scratch
    slots_max: int        # upper bound on count known on the host (min(E, number of ids)) = decode grid dim 0


def decode(c: CompressedSF, out: torch.Tensor, experts: ExpertList) -> None:
    """Decode the chunks of the experts in `experts` into out (same shape/strides as the original SF)."""
    E, N, Kp = c.shape
    assert tuple(out.shape) == (E, N, Kp) and out.stride() == c.stride, (out.shape, out.stride(), c.stride)
    cpe = Kp * (N // 128)
    _decode_kernel[(experts.slots_max, triton.cdiv(cpe, DECODE_CH))](
        c.slots, c.overflow, out, experts.elist, experts.count, cpe, SLOT=SLOT_WORDS, CH=DECODE_CH,
        num_warps=DECODE_WARPS)


@triton.jit
def _list_kernel(ids, n_ids, flags, elist, ecount, E, BLOCK_IDS: tl.constexpr, BLOCK_E: tl.constexpr):
    eo = tl.arange(0, BLOCK_E)
    tl.store(flags + eo, 0, mask=eo < E)
    tl.debug_barrier()
    offs = tl.arange(0, BLOCK_IDS)
    for s in range(0, n_ids, BLOCK_IDS):
        x = tl.load(ids + s + offs, mask=s + offs < n_ids, other=-1)
        tl.store(flags + x, 1, mask=(x >= 0) & (x < E))
    tl.debug_barrier()
    f = tl.load(flags + eo, mask=eo < E, other=0)
    pos = tl.cumsum(f, axis=0)
    tl.store(elist + pos - 1, eo, mask=f == 1)
    tl.store(ecount, tl.sum(f, axis=0))


def make_expert_list(E: int, device) -> ExpertList:
    return ExpertList(torch.zeros(E, dtype=torch.int32, device=device), torch.zeros(1, dtype=torch.int32, device=device),
                      torch.zeros(E, dtype=torch.int32, device=device), E)


def fill_expert_list(ids: torch.Tensor, experts: ExpertList) -> ExpertList:
    """Unique expert ids of `ids` (ids < 0 or >= E ignored) into experts.elist/count. Graph-safe (no host sync)."""
    E = experts.elist.numel()
    n = ids.numel()
    _list_kernel[(1,)](ids.reshape(-1), n, experts.flags, experts.elist, experts.count, E,
                       BLOCK_IDS=1024, BLOCK_E=triton.next_power_of_2(E))
    experts.slots_max = max(1, min(E, n))
    return experts


def empty_like_sf(shape, stride, device) -> torch.Tensor:
    E, N, Kp = shape
    return torch.empty_strided((E, N, Kp), stride, dtype=torch.int32, device=device)
