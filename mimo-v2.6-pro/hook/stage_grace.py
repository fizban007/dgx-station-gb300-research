"""Stage the Grace-resident experts a decode step routes to into a shared HBM buffer, then let Marlin read them there.

Marlin reading its weights straight through the UVA view gets ~258 GB/s from Grace on the GB300 at C1 (profile, 2026-09-25);
a plain copy over NVLink-C2C gets 350-390 GB/s for 18-36 MiB (microbenchmark). Per layer and step:

  _stage_map  (1 program): mark the Grace experts routed this step, give each a slot (exclusive scan over the 384 ids),
              write EMAP[e] = slot or -1 (Marlin's expert_map for the staged bank), INV[slot] = Grace-local row, NUSED.
  _stage_copy (fixed grid): copy the used rows of the four Marlin tensors (w13, w2 and their scales) into the staging
              tensors, a grid-stride loop over (slot, chunk) work items so the grid does not grow with the slot count.

Both are device-only with fixed grids, so CUDA graphs capture them. The staging tensors are shared by every layer (layers
run one at a time); capacity S bounds the distinct Grace experts a step may route to, so the staged bank serves batches
with T * top_k <= S and larger batches keep the UVA bank.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_map(IDS, NROUTES, CMAP, EMAP, INV, NUSED, E: tl.constexpr, BLOCK_E: tl.constexpr,
               BLOCK_R: tl.constexpr):
    e = tl.arange(0, BLOCK_E)
    used = tl.zeros([BLOCK_E], dtype=tl.int32)
    for start in range(0, NROUTES, BLOCK_R):
        r = start + tl.arange(0, BLOCK_R)
        ids = tl.load(IDS + r, mask=r < NROUTES, other=-1)
        used = used | tl.max((ids[:, None] == e[None, :]).to(tl.int32), axis=0)
    cm = tl.load(CMAP + e, mask=e < E, other=-1)
    used = used & (cm >= 0).to(tl.int32)
    slot = tl.cumsum(used, 0) - used
    tl.store(EMAP + e, tl.where(used > 0, slot, -1), mask=e < E)
    tl.store(INV + slot, cm, mask=used > 0)
    tl.store(NUSED, tl.sum(used, 0))


@triton.jit
def _copy_seg(SRC, DST, N, NUSED, INV, pid, NPROG, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    """Copy N int64 words per used slot: DST[s, :] = SRC[INV[s], :], work item w = (slot, chunk)."""
    nch = tl.cdiv(N, CHUNK)
    for w in range(pid, NUSED * nch, NPROG):
        s = w // nch
        j = w - s * nch
        c = tl.load(INV + s).to(tl.int64)
        for off in range(0, CHUNK, BLOCK):
            idx = j * CHUNK + off + tl.arange(0, BLOCK)
            m = idx < N
            v = tl.load(SRC + c * N + idx, mask=m)
            tl.store(DST + s.to(tl.int64) * N + idx, v, mask=m)


@triton.jit
def _stage_copy(INV, NUSED_PTR, BASE, SLOTS, S13, D13, N13, S2, D2, N2, SS13, DS13, NS13, SS2, DS2, NS2,
                CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    """Copy mapped slots [BASE, BASE + SLOTS) (those below NUSED) into staging slots [0, SLOTS)."""
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    nused = tl.minimum(tl.maximum(tl.load(NUSED_PTR) - BASE, 0), SLOTS)
    inv = INV + BASE
    _copy_seg(S13, D13, N13, nused, inv, pid, nprog, CHUNK, BLOCK)
    _copy_seg(S2, D2, N2, nused, inv, pid, nprog, CHUNK, BLOCK)
    _copy_seg(SS13, DS13, NS13, nused, inv, pid, nprog, CHUNK, BLOCK)
    _copy_seg(SS2, DS2, NS2, nused, inv, pid, nprog, CHUNK, BLOCK)


def _words(t: torch.Tensor) -> torch.Tensor:
    """Per-expert rows as int64 words ([n, words]); every Marlin tensor here has a per-expert size divisible by 8 bytes."""
    per = t[0].numel() * t.element_size()
    assert per % 8 == 0, f"per-expert bytes {per} not a multiple of 8"
    return t.contiguous().view(t.shape[0], -1).view(torch.uint8).view(torch.int64).view(t.shape[0], per // 8)


class GraceStager:
    """Shared staging tensors (capacity S experts) plus per-step scratch; one instance serves every layer."""

    def __init__(self, w13: torch.Tensor, w2: torch.Tensor, s13: torch.Tensor, s2: torch.Tensor, slots: int,
                 num_experts: int, nprog: int = 0, chunk_words: int = 8192, block_words: int = 1024):
        dev = torch.device("cuda", torch.cuda.current_device())
        self.slots, self.E = slots, num_experts
        self.w13 = torch.empty((slots,) + tuple(w13.shape[1:]), dtype=w13.dtype, device=dev)
        self.w2 = torch.empty((slots,) + tuple(w2.shape[1:]), dtype=w2.dtype, device=dev)
        self.s13 = torch.empty((slots,) + tuple(s13.shape[1:]), dtype=s13.dtype, device=dev)
        self.s2 = torch.empty((slots,) + tuple(s2.shape[1:]), dtype=s2.dtype, device=dev)
        self._dst = [_words(t) for t in (self.w13, self.w2, self.s13, self.s2)]
        self.emap = torch.full((num_experts,), -1, dtype=torch.int32, device=dev)
        self.inv = torch.zeros(num_experts, dtype=torch.int32, device=dev)  # slots <= E; a map over S never overflows
        self.nused = torch.zeros(1, dtype=torch.int32, device=dev)
        props = torch.cuda.get_device_properties(dev)
        import os
        self.nprog = nprog or int(os.environ.get("HOTSPLIT_STAGE_PROGRAMS", "0")) or 4 * props.multi_processor_count
        self.chunk, self.block = chunk_words, block_words
        self.nbytes = sum(t.numel() * t.element_size() for t in (self.w13, self.w2, self.s13, self.s2))

    def stage_rows(self, start: int, n: int, src: tuple) -> None:
        """Copy Grace-local rows [start, start + n) into slots [0, n) (the prefill slab path; eager only)."""
        self.inv[:n].copy_(torch.arange(start, start + n, dtype=torch.int32, device=self.inv.device))
        self.nused.fill_(n)
        d = self._dst
        _stage_copy[(self.nprog,)](self.inv, self.nused, 0, self.slots,
                                   src[0], d[0], d[0].shape[1], src[1], d[1], d[1].shape[1],
                                   src[2], d[2], d[2].shape[1], src[3], d[3], d[3].shape[1],
                                   CHUNK=self.chunk, BLOCK=self.block)

    def map(self, topk_ids: torch.Tensor, cmap: torch.Tensor) -> torch.Tensor:
        """Slot per routed Grace expert (EMAP) and the used count (NUSED); no copies."""
        _stage_map[(1,)](topk_ids, topk_ids.numel(), cmap, self.emap, self.inv, self.nused, E=self.E,
                         BLOCK_E=triton.next_power_of_2(self.E), BLOCK_R=16)
        return self.emap

    def stage(self, topk_ids: torch.Tensor, cmap: torch.Tensor, src: tuple, mapped: bool = False,
              base: int = 0) -> torch.Tensor:
        """src = (w13, w2, s13, s2) of one layer's Grace bank, each viewed as int64 words [n, words]. Returns EMAP.
        Copies mapped slots [base, base + slots) into the staging slots (a round); slots past NUSED are skipped."""
        if not mapped:
            self.map(topk_ids, cmap)
        d = self._dst
        _stage_copy[(self.nprog,)](self.inv, self.nused, base, self.slots,
                                   src[0], d[0], d[0].shape[1], src[1], d[1], d[1].shape[1],
                                   src[2], d[2], d[2].shape[1], src[3], d[3], d[3].shape[1],
                                   CHUNK=self.chunk, BLOCK=self.block)
        return self.emap
