"""Fused decode-size send for peer tier v2: one single-CTA kernel per MoE layer.

For T <= FUSED_MAX tokens it replaces the ~25 small PyTorch ops the hook ran between the router and MegaMoE
(valid mask, route counting, hot/cold split through the row map, has/any, prefix positions, row count, casts)
plus peer_tier2's _pack and _publish. Semantics match mega_peer_hook._forward + PeerTier2.send exactly:

    valid   = ids >= 0
    counts += 1 per valid route                     (only when counting is on)
    hot     = row_map[id] if valid and row_map[id] >= 0 else -1
    cold    = -row_map[id] - 1 if valid and row_map[id] < 0 else -1
    has[t]  = any(cold[t] >= 0);  pos = cumsum(has) - 1;  rows = sum(has)
    rows with a cold route are packed (MXFP8 activations, scales, cold ids, fp32 weights) at pos[t];
    rows [rows, PAD) get masked ids; then the header is written and the sequence published (release, sys).

PeerTier2.finish (wait + scatter-add) is unchanged and consumes has/pos/rows.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

FUSED_MAX = 64


@triton.jit
def _fence_sys(dep):
    return tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, 0;", "=r,r", [dep], dtype=tl.int32,
                                     is_pure=False, pack=1)


@triton.jit(do_not_specialize=["LAYER", "T"])
def _route_send(IDS, WTS, ROWMAP, COUNTS, HOT, HAS, POS, ROWS,
                X, XS, PX, PXS, PIDS, PW, WORDS, HEADER_PTR, SEQ, LAYER, T,
                H: tl.constexpr, HS: tl.constexpr, K: tl.constexpr, PAD: tl.constexpr,
                BLOCK_T: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr,
                COUNT: tl.constexpr):
    t = tl.arange(0, BLOCK_T)
    k = tl.arange(0, BLOCK_K)
    live = t < T
    m = live[:, None] & (k[None, :] < K)
    ids = tl.load(IDS + t[:, None] * K + k[None, :], mask=m, other=-1).to(tl.int32)
    valid = ids >= 0
    safe = tl.where(valid, ids, 0)
    if COUNT:
        tl.atomic_add(COUNTS + safe, tl.full((BLOCK_T, BLOCK_K), 1, tl.int64), mask=valid, sem="relaxed")
    rm = tl.load(ROWMAP + safe, mask=m, other=0)
    hot = tl.where(valid & (rm >= 0), rm, -1)
    cold = tl.where(valid & (rm < 0), -rm - 1, -1)
    tl.store(HOT + t[:, None] * K + k[None, :], hot.to(HOT.dtype.element_ty), mask=m)

    has = tl.max((cold >= 0).to(tl.int32), axis=1)
    has = tl.where(live, has, 0)
    pos = tl.cumsum(has, axis=0) - 1
    rows = tl.sum(has, axis=0)
    tl.store(HAS + t, has.to(tl.int8), mask=live)
    tl.store(POS + t, pos, mask=live)
    tl.store(ROWS, rows)

    packed = has > 0
    dst = tl.where(packed, pos, 0).to(tl.int64)
    pm = m & packed[:, None]
    w = tl.load(WTS + t[:, None] * K + k[None, :], mask=m, other=0.0).to(tl.float32)
    tl.store(PIDS + dst[:, None] * K + k[None, :], cold, mask=pm)
    tl.store(PW + dst[:, None] * K + k[None, :], w, mask=pm)
    src = t.to(tl.int64)
    for off in tl.static_range(0, H, BLOCK_H):
        cols = off + tl.arange(0, BLOCK_H)
        cm = packed[:, None] & (cols[None, :] < H)
        row = tl.load(X + src[:, None] * H + cols[None, :], mask=cm)
        tl.store(PX + dst[:, None] * H + cols[None, :], row, mask=cm)
    s = tl.arange(0, BLOCK_S)
    sm = packed[:, None] & (s[None, :] < HS)
    tl.store(PXS + dst[:, None] * HS + s[None, :], tl.load(XS + src[:, None] * HS + s[None, :], mask=sm), mask=sm)
    if PAD > 0:
        # The peer copies whole buckets of up to PAD rows inside its graphs: rows past `rows` carry masked routes.
        r = tl.arange(0, PAD)
        padm = (r[:, None] >= rows) & (k[None, :] < K)
        tl.store(PIDS + r[:, None] * K + k[None, :], tl.full((PAD, BLOCK_K), -1, tl.int32), mask=padm)

    # Every thread's payload stores are ordered before the release below.
    _fence_sys(rows)
    tl.debug_barrier()
    seq = tl.load(SEQ) + 1
    tl.store(SEQ, seq)
    tl.store(HEADER_PTR + 1, LAYER.to(tl.int64))
    tl.store(HEADER_PTR + 2, rows.to(tl.int64))
    tl.store(HEADER_PTR + 3, T.to(tl.int64))
    tl.store(HEADER_PTR + 0, seq)
    tl.atomic_add(WORDS + 3, rows.to(tl.int64), sem="relaxed", scope="sys")
    tl.atomic_add(WORDS + 4, T.to(tl.int64), sem="relaxed", scope="sys")
    tl.atomic_xchg(WORDS, seq, sem="release", scope="sys")


def route_send(peer, topk_ids, topk_weights, row_map, counts, x_quant, x_scale, layer: int,
               hidden: int, topk: int, pad: int):
    """Split routes, pack cold rows and publish in one launch; returns (hot_ids, has, pos, rows)."""
    tokens, k = topk_ids.shape
    if tokens > FUSED_MAX or k != topk or (pad and pad & (pad - 1)):
        raise ValueError(f"fused send handles up to {FUSED_MAX} tokens of top-{topk} routes, power-of-two pad")
    dev = topk_ids.device
    ids = topk_ids.contiguous()
    hot = torch.empty_like(ids)
    has = torch.empty(tokens, dtype=torch.int8, device=dev)
    pos = torch.empty(tokens, dtype=torch.int32, device=dev)
    rows = torch.empty(1, dtype=torch.int32, device=dev)
    block_t = max(16, triton.next_power_of_2(tokens))
    _route_send[(1,)](
        ids, topk_weights.contiguous(), row_map, counts if counts is not None else row_map, hot, has, pos, rows,
        x_quant.view(torch.uint8), x_scale.view(torch.uint8), peer.px, peer.pxs, peer.pids, peer.pw,
        peer.words, peer.header, peer.seq, layer, tokens,
        H=hidden, HS=hidden // 32, K=topk, PAD=pad, BLOCK_T=block_t, BLOCK_K=8,
        BLOCK_H=max(256, 16384 // block_t), BLOCK_S=256, COUNT=counts is not None, num_warps=4)
    return hot, has, pos, rows
