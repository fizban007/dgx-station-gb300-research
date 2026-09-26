"""hotsplit: per-expert HBM residency for UVA-offloaded MXFP4/Marlin MoE layers (experiment, not upstream).

vLLM's UVA offloader works per parameter, so a layer's 384 experts are either all in HBM or all in pinned
host memory. hotsplit runs after process_weights_after_loading and re-partitions every routed-expert layer
into two expert sets:
  hot  -> compact HBM tensors  (index_select of the kernel-format weights)
  cold -> compact pinned-host tensors viewed through UVA
then replaces the layer's quant_method.apply with two Marlin launches (one per set) driven by expert_maps
(global id -> local id, -1 = not in this set) and sums the partial outputs. Marlin + moe_sum already
honour expert_map (the EP path), so no kernel changes.

Budget: by default the hot set gets exactly the HBM bytes the stock layer-granular placement used for
routed experts, so KV-cache size is unchanged. Cells (layer, expert) are ranked by routed count from a
histogram JSON and taken greedily until the budget is spent.

Three-tier extension (research/mimo-pro, 2026-09-25): with HOTSPLIT_ROWMAP the plan comes from plan_tiers.py
({"layers": {"<layer>": {"hot": [...], "peer": [...], "cold": [...]}}}). Peer experts live only on the RTX PRO 6000
sidecar (peer_server_mimo.py); the GB300 keeps neither an HBM nor a Grace copy. apply() publishes the tokens that
route to peer experts (peer_tier_mimo.py, graph-safe device kernels), runs the hot and Grace banks, then waits for
the sidecar and adds its rows.

Env:
  HOTSPLIT_ROWMAP   three-tier plan JSON (overrides the counts-based ranking; HOTSPLIT_COUNTS is then unused)
  HOTSPLIT_PEER_CHECK=N  also keep the peer experts in a Grace Marlin bank and, for the first N eager calls with
                    T >= HOTSPLIT_PEER_CHECK_MIN_T (default 1), log the sidecar's contribution against Marlin's
  HOTSPLIT_FUSED_SEND=1  peer send as three fused kernels (peer_tier_mimo.send_fused); 0 = the original torch ops
  HOTSPLIT_STAGE_SLOTS=128  stage the Grace experts a batch with T * top_k <= slots routes to into shared HBM staging
                    (stage_grace.py) and run that bank from HBM; larger batches keep the UVA bank. 0 disables.
  HOTSPLIT_STAGE_OVERLAP=1  run the staging copy on a side stream, overlapped with the HBM bank
  HOTSPLIT_STAGE_CHECK=N  for the first N eager calls that stage, also run the UVA bank and log the difference
  HOTSPLIT_COUNTS   path to counts JSON ({"train": {"decode"|"prefill": {layer: [384]}}} or {layer: {"counts": [...]}})
  HOTSPLIT_WEIGHTS  "decode=1,prefill=0.25"  mix of phases (default)
  HOTSPLIT_HOT_GIB  override hot budget (GiB)
  HOTSPLIT_DRYRUN=1 compute + log the plan only
"""
from __future__ import annotations

import json
import os
import re
import time
import types

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.hotsplit")
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _load_counts(path: str) -> dict[int, torch.Tensor]:
    raw = json.load(open(path))
    weights = {"decode": 1.0, "prefill": 0.25}
    for kv in os.environ.get("HOTSPLIT_WEIGHTS", "").split(","):
        if "=" in kv:
            k, v = kv.split("="); weights[k.strip()] = float(v)
    out: dict[int, torch.Tensor] = {}
    if "train" in raw:
        for phase, w in weights.items():
            for li, c in raw["train"].get(phase, {}).items():
                t = torch.tensor(c, dtype=torch.float64)
                t = t / max(t.sum().item(), 1.0) * w  # normalise per layer+phase so phases mix by weight
                out[int(li)] = out.get(int(li), 0) + t
    else:
        for li, v in raw.items():
            t = torch.tensor(v["counts"], dtype=torch.float64); out[int(li)] = t / max(t.sum().item(), 1.0)
    return out


def _is_uva(p: torch.Tensor) -> bool:
    return bool(getattr(p, "_vllm_is_uva_offloaded", False))


def _pinned_view_from_gpu(src: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Exact-size pinned host buffer (cudaHostAlloc via the unpinned-input branch) holding src, as a UVA view."""
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = torch.empty(src.shape, dtype=src.dtype, device="cpu")  # untouched pages; helper allocs pinned + copies
    view = get_accelerator_view_from_cpu_tensor(host)
    del host
    for s in range(0, src.shape[0], chunk):
        view[s:s + chunk].copy_(src[s:s + chunk])
    torch.cuda.synchronize()
    view._vllm_is_uva_offloaded = True
    return view


def _gather_rows(w: torch.Tensor, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """HBM copy of w[ids] (w may be a UVA view; reads stream over C2C)."""
    out = torch.empty((ids.numel(),) + tuple(w.shape[1:]), dtype=w.dtype, device="cuda")
    for s in range(0, ids.numel(), chunk):
        out[s:s + chunk].copy_(w.index_select(0, ids[s:s + chunk]))
    return out


def _gather_rows_to_host(w: torch.Tensor, ids: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    host = torch.empty((ids.numel(),) + tuple(w.shape[1:]), dtype=w.dtype, device="cpu")
    view = get_accelerator_view_from_cpu_tensor(host)
    del host
    for s in range(0, ids.numel(), chunk):
        view[s:s + chunk].copy_(w.index_select(0, ids[s:s + chunk]))
    torch.cuda.synchronize()
    view._vllm_is_uva_offloaded = True
    return view


def _host_avail_gib() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return -1.0


def apply_hotsplit(model: torch.nn.Module) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod

    t0 = time.time()
    rowmap = None
    if os.environ.get("HOTSPLIT_ROWMAP"):
        rowmap = {int(k): v for k, v in json.load(open(os.environ["HOTSPLIT_ROWMAP"]))["layers"].items()}
        counts = None
    else:
        counts = _load_counts(os.environ["HOTSPLIT_COUNTS"])
    wanted = rowmap if rowmap is not None else counts
    layers: dict[int, torch.nn.Module] = {}
    for name, mod in model.named_modules():
        qm = getattr(mod, "quant_method", None)
        if isinstance(qm, Mxfp4MoEMethod) and hasattr(mod, "w13_weight") and qm.moe_kernel is not None:
            m = _LAYER_RE.search(name)
            if m and int(m.group(1)) in wanted:
                layers[int(m.group(1))] = mod
    if not layers:
        logger.warning("hotsplit: no eligible MXFP4 MoE layers found; nothing to do")
        return
    any_l = next(iter(layers.values()))
    E = any_l.w13_weight.shape[0]
    cell_bytes = {li: (l.w13_weight[0].numel() * l.w13_weight.element_size()
                       + l.w2_weight[0].numel() * l.w2_weight.element_size()) for li, l in layers.items()}
    hbm_now = sum(cell_bytes[li] * E for li, l in layers.items() if not _is_uva(l.w13_weight))
    budget = float(os.environ["HOTSPLIT_HOT_GIB"]) * 2**30 if os.environ.get("HOTSPLIT_HOT_GIB") else hbm_now
    peer: dict[int, list[int]] = {li: [] for li in layers}
    if rowmap is not None:
        hot = {li: [int(e) for e in rowmap[li]["hot"]] for li in layers}
        peer = {li: [int(e) for e in rowmap[li].get("peer", [])] for li in layers}
        for li in layers:
            if set(hot[li]) & set(peer[li]) or len(set(hot[li]) | set(peer[li]) | set(rowmap[li]["cold"])) != E:
                raise RuntimeError(f"hotsplit: rowmap layer {li} is not a partition of 0..{E - 1}")
        logger.info("hotsplit plan (rowmap %s): %d layers, hot cells %d (%.1f GiB HBM), peer cells %d (%.1f GiB "
                    "on the sidecar), Grace cells %d; stock HBM experts %.1f GiB",
                    os.environ["HOTSPLIT_ROWMAP"], len(layers), sum(map(len, hot.values())),
                    sum(len(hot[li]) * cell_bytes[li] for li in layers) / 2**30, sum(map(len, peer.values())),
                    sum(len(peer[li]) * cell_bytes[li] for li in layers) / 2**30,
                    sum(E - len(hot[li]) - len(peer[li]) for li in layers), hbm_now / 2**30)
    else:
        hot = _rank_hot(layers, counts, cell_bytes, E, budget, hbm_now)
    if os.environ.get("HOTSPLIT_DRYRUN") == "1":
        return
    _split_all(layers, hot, peer, cell_bytes, E, make_mxfp4_moe_kernel, make_mxfp4_moe_quant_config, t0)


def _rank_hot(layers, counts, cell_bytes, E, budget, hbm_now) -> dict[int, list[int]]:
    # rank all (layer, expert) cells by normalised routing share per byte
    cells = []
    for li in layers:
        c = counts[li]
        for e in range(E):
            cells.append((c[e].item() / cell_bytes[li], li, e))
    cells.sort(reverse=True)
    hot: dict[int, list[int]] = {li: [] for li in layers}
    used = 0
    for _, li, e in cells:
        if used + cell_bytes[li] > budget:
            continue
        hot[li].append(e); used += cell_bytes[li]
    cov = sum(counts[li][hot[li]].sum().item() for li in layers) / sum(counts[li].sum().item() for li in layers)
    logger.info("hotsplit plan: %d layers, budget %.1f GiB (stock HBM experts %.1f GiB), hot cells %d/%d, "
                "train-weighted coverage %.1f%%", len(layers), budget / 2**30, hbm_now / 2**30,
                sum(len(v) for v in hot.values()), E * len(layers), 100 * cov)
    logger.info("hotsplit per-layer hot counts: %s", {li: len(hot[li]) for li in sorted(hot)})
    return hot


def _split_all(layers, hot, peer, cell_bytes, E, make_mxfp4_moe_kernel, make_mxfp4_moe_quant_config, t0) -> None:
    # order: UVA layers (consume HBM, free host) while HBM allows, else an HBM layer (frees HBM, consumes host)
    uva_q = [li for li in sorted(layers) if _is_uva(layers[li].w13_weight)]
    hbm_q = [li for li in sorted(layers) if not _is_uva(layers[li].w13_weight)]
    margin = 6 * 2**30
    done = 0
    while uva_q or hbm_q:
        free, _ = torch.cuda.mem_get_info()
        if uva_q and (free - len(hot[uva_q[0]]) * cell_bytes[uva_q[0]] > margin or not hbm_q):
            li = uva_q.pop(0)
        else:
            li = hbm_q.pop(0)
        host_need = (E - len(hot[li]) - (0 if PEER_CHECK else len(peer[li]))) * cell_bytes[li] / 2**30
        if _host_avail_gib() - host_need < float(os.environ.get("HOTSPLIT_HOST_FLOOR_GIB", "24")):
            logger.warning("hotsplit: host floor hit (avail %.1f GiB, need %.1f); leaving layer %d and %d others stock",
                           _host_avail_gib(), host_need, li, len(uva_q) + len(hbm_q))
            break
        _split_layer(layers[li], hot[li], E, make_mxfp4_moe_kernel, make_mxfp4_moe_quant_config, li, peer[li])
        torch.cuda.empty_cache()  # release the freed originals before the next layer's copies (avoids allocator retries)
        done += 1
        if done % 10 == 0:
            torch.cuda.empty_cache()
            logger.info("hotsplit: %d/%d layers; HBM free %.1f GiB; host avail %.1f GiB",
                        done, len(layers), torch.cuda.mem_get_info()[0] / 2**30, _host_avail_gib())
    torch.cuda.empty_cache()
    if _LIVE is not None and _LIVE.layer_idx:
        mnbt = int(os.environ.get("HOTSPLIT_MAX_TOKENS", "8192"))
        _LIVE.finalize(E, mnbt, 8)
    logger.info("hotsplit done in %.0fs; HBM free %.1f GiB; host avail %.1f GiB",
                time.time() - t0, torch.cuda.mem_get_info()[0] / 2**30, _host_avail_gib())


def _split_layer(layer, hot_ids: list[int], E: int, make_kernel, make_qcfg, li: int = -1,
                 peer_ids: list[int] | None = None) -> None:
    qm = layer.quant_method
    dev = torch.device("cuda")
    hot_set = set(hot_ids)
    peer_ids = list(peer_ids or [])
    peer_set = set(peer_ids)
    h = torch.tensor(sorted(hot_set), dtype=torch.long, device=dev)
    c = torch.tensor([e for e in range(E) if e not in hot_set and e not in peer_set], dtype=torch.long, device=dev)
    p = torch.tensor(peer_ids, dtype=torch.long, device=dev)  # sidecar order: local id = position in the rowmap list

    def emap(ids: torch.Tensor) -> torch.Tensor:
        m = torch.full((E,), -1, dtype=torch.int32, device=dev)
        if ids.numel():
            m[ids] = torch.arange(ids.numel(), dtype=torch.int32, device=dev)
        return m

    w13, w2 = layer.w13_weight, layer.w2_weight
    s13, s2 = layer.w13_weight_scale, layer.w2_weight_scale
    if _is_trt(qm):
        _split_layer_trt(layer, qm, h, c, p, E, li, w13, w2, s13, s2)
        return
    parts = []
    banks = [(h, "hbm"), (c, "host")]
    if peer_ids and PEER_CHECK:
        banks.append((p, "check"))
    for ids, where in banks:
        if ids.numel() == 0:
            continue
        if where == "hbm":
            pw13, pw2 = _gather_rows(w13, ids), _gather_rows(w2, ids)
        else:  # "host" and "check" both live in Grace
            pw13, pw2 = _gather_rows_to_host(w13, ids), _gather_rows_to_host(w2, ids)
        ps13, ps2 = s13.index_select(0, ids).contiguous(), s2.index_select(0, ids).contiguous()
        qcfg = make_qcfg(mxfp4_backend=qm.mxfp4_backend, w1_scale=ps13, w2_scale=ps2,
                         swiglu_limit=getattr(layer, "swiglu_limit", None), layer=layer)
        kern = make_kernel(moe_quant_config=qcfg, moe_config=qm.moe, experts_cls=qm.experts_cls,
                           mxfp4_backend=qm.mxfp4_backend, routing_tables=layer._expert_routing_tables())
        parts.append((kern, pw13, pw2, emap(ids)))
        if where == "host" and STAGE_SLOTS:
            # The same Grace bank, staged: a kernel over the shared HBM staging tensors, driven by a per-step expert map.
            stager = _get_stager(pw13, pw2, ps13, ps2, E)
            sq = make_qcfg(mxfp4_backend=qm.mxfp4_backend, w1_scale=stager.s13, w2_scale=stager.s2,
                           swiglu_limit=getattr(layer, "swiglu_limit", None), layer=layer)
            skern = make_kernel(moe_quant_config=sq, moe_config=qm.moe, experts_cls=qm.experts_cls,
                                mxfp4_backend=qm.mxfp4_backend, routing_tables=layer._expert_routing_tables())
            from stage_grace import _words
            cmap = torch.full((E,), -1, dtype=torch.int32, device=dev)
            cmap[ids] = torch.arange(ids.numel(), dtype=torch.int32, device=dev)
            qm._hotsplit_stage = (skern, cmap, tuple(_words(t) for t in (pw13, pw2, ps13, ps2)))

    # drop the original full tensors (frees HBM or the pinned host buffer)
    placeholder = torch.nn.Parameter(torch.empty(0, dtype=w13.dtype, device=dev), requires_grad=False)
    layer.w13_weight = placeholder
    layer.w2_weight = torch.nn.Parameter(torch.empty(0, dtype=w2.dtype, device=dev), requires_grad=False)
    del w13, w2
    # free the full-size scales in place (old kernel's quant config still points at these Parameters; unused now)
    for sname in ("w13_weight_scale", "w2_weight_scale"):
        sp = getattr(layer, sname, None)
        if isinstance(sp, torch.Tensor):
            sp.data = torch.empty(0, dtype=sp.dtype, device=sp.device)
    torch.cuda.synchronize()
    if peer_ids and PEER_CHECK:
        qm._hotsplit_peer_ref = parts.pop()  # the "check" bank: never part of the served sum
    qm._hotsplit_parts = parts
    qm._hotsplit_nhot = h.numel()
    if peer_ids:
        pm = torch.full((E,), -1, dtype=torch.int32, device=dev)
        pm[p] = torch.arange(p.numel(), dtype=torch.int32, device=dev)
        qm._hotsplit_peer_map = pm
        qm._hotsplit_layer = li

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        assert shared_experts is None, "hotsplit: shared experts path not supported"
        slot = getattr(self, "_hotsplit_live_slot", None)
        if slot is not None and _LIVE is not None and _LIVE.buf is not None \
                and topk_ids.shape[0] <= _LIVE.decode_max:  # decode-sized steps only (static shape under capture)
            _LIVE.count(slot, topk_ids)
        pm = getattr(self, "_hotsplit_peer_map", None)
        if pm is not None:
            tier = _peer_tier(x.device)
            if FUSED_SEND:
                peer_state = tier.send_fused(x, topk_ids, topk_weights, pm, self._hotsplit_layer)
            else:
                xq, xs = _mxfp8(x)
                pids = torch.where(topk_ids >= 0, pm[topk_ids.clamp_min(0).long()], -1).to(torch.int32)
                peer_state = tier.send(xq, xs, pids, topk_weights, self._hotsplit_layer)
        stage = getattr(self, "_hotsplit_stage", None)
        staged = stage is not None and topk_ids.numel() <= STAGE_SLOTS
        parts = self._hotsplit_parts
        if staged:
            kern_s, cmap, src = stage
            cur = torch.cuda.current_stream()
            if STAGE_OVERLAP:  # the copy reads Grace over C2C while the HBM bank reads HBM
                side = _side_stream(x.device)
                side.wait_stream(cur)
                with torch.cuda.stream(side):
                    emap = _STAGER.stage(topk_ids, cmap, src)
            else:
                emap = _STAGER.stage(topk_ids, cmap, src)
            out = None
            for kern, pw13, pw2, m in parts[:-1]:  # the HBM bank
                o = _run_part(kern, pw13, pw2, m, layer, x, topk_weights, topk_ids)
                out = o if out is None else out.add_(o)
            if STAGE_OVERLAP:
                cur.wait_stream(side)
            o = _run_part(kern_s, _STAGER.w13, _STAGER.w2, emap, layer, x, topk_weights, topk_ids)
            _stage_check(parts[-1], layer, x, topk_weights, topk_ids, o)
            out = o if out is None else out.add_(o)
        else:
            out = None
            for kern, pw13, pw2, m in parts:
                o = _run_part(kern, pw13, pw2, m, layer, x, topk_weights, topk_ids)
                out = o if out is None else out.add_(o)
        if pm is not None:
            ref = _peer_check_ref(self, layer, x, topk_weights, topk_ids, out)
            if FUSED_SEND:
                tier.finish_fused(out, peer_state)
                rows = peer_state[1]
            else:
                has, pos, rows = peer_state
                tier.finish(out, has, pos, rows)
            if ref is not None:
                _peer_check_log(self._hotsplit_layer, x.shape[0], rows, ref[0], out, ref[1])
        return out

    qm.apply = types.MethodType(apply, qm)
    if _LIVE is not None:
        _LIVE.register(qm, li)


def _is_trt(qm) -> bool:
    return "TRTLLM" in getattr(qm.mxfp4_backend, "name", str(qm.mxfp4_backend))


_MODULAR_CLASSES: dict = {}


def _force_modular(qm) -> None:
    """The TRT backend picks the monolithic experts class (routing inside the kernel), which never calls apply().
    Give this instance a subclass whose is_monolithic is False: vLLM's runner then routes (select_experts) and calls
    our apply with topk ids/weights."""
    cls = type(qm)
    if cls not in _MODULAR_CLASSES:
        _MODULAR_CLASSES[cls] = type(cls.__name__ + "Hotsplit", (cls,),
                                     {"is_monolithic": property(lambda self: False)})
    qm.__class__ = _MODULAR_CLASSES[cls]


def _split_layer_trt(layer, qm, h, c, p, E, li, w13, w2, s13, s2) -> None:
    """TRT layout: hot rows to HBM, Grace rows to exact-size pinned host (UVA views, only ever read by the staging
    copies), scales of both in HBM; every expert GEMM runs from HBM (trt_banks.TrtLayer)."""
    import sys
    sys.path.insert(0, "/w")
    from trt_banks import TrtLayer
    assert STAGE_SLOTS > 0, "TRT hotsplit needs HOTSPLIT_STAGE_SLOTS > 0 (the Grace bank is only read by staging)"
    dev = torch.device("cuda")
    hot = grace = None
    if h.numel():
        hot = (_gather_rows(w13, h), s13.index_select(0, h).contiguous(), _gather_rows(w2, h),
               s2.index_select(0, h).contiguous())
    stager = None
    if c.numel():
        grace = (_gather_rows_to_host(w13, c), _gather_rows_to_host(w2, c), s13.index_select(0, c).contiguous(),
                 s2.index_select(0, c).contiguous())
        stager = _get_stager(*grace, E)
    inter = layer.w13_weight.shape[1] // 2
    qm._hs_trt = TrtLayer(E, inter, hot, h, grace, c, stager)
    placeholder = torch.nn.Parameter(torch.empty(0, dtype=w13.dtype, device=dev), requires_grad=False)
    layer.w13_weight = placeholder
    layer.w2_weight = torch.nn.Parameter(torch.empty(0, dtype=w2.dtype, device=dev), requires_grad=False)
    for sname in ("w13_weight_scale", "w2_weight_scale"):
        sp = getattr(layer, sname, None)
        if isinstance(sp, torch.Tensor):
            sp.data = torch.empty(0, dtype=sp.dtype, device=sp.device)
    torch.cuda.synchronize()
    if p.numel():
        pm = torch.full((E,), -1, dtype=torch.int32, device=dev)
        pm[p] = torch.arange(p.numel(), dtype=torch.int32, device=dev)
        qm._hotsplit_peer_map = pm
        qm._hotsplit_layer = li
    _force_modular(qm)

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        assert shared_experts is None, "hotsplit: shared experts path not supported"
        slot = getattr(self, "_hotsplit_live_slot", None)
        if slot is not None and _LIVE is not None and _LIVE.buf is not None \
                and topk_ids.shape[0] <= _LIVE.decode_max:
            _LIVE.count(slot, topk_ids)
        pm = getattr(self, "_hotsplit_peer_map", None)
        if pm is not None:
            tier = _peer_tier(x.device)
            peer_state = tier.send_fused(x, topk_ids, topk_weights, pm, self._hotsplit_layer)
        out = self._hs_trt.forward(x, topk_ids, topk_weights)
        _trt_stage_check(self._hs_trt, x, topk_ids, topk_weights)
        if pm is not None:
            tier.finish_fused(out, peer_state)
        return out

    qm.apply = types.MethodType(apply, qm)
    if _LIVE is not None:
        _LIVE.register(qm, li)


def _trt_stage_check(tl_, x, topk_ids, topk_weights):
    """Eager decode-size calls: the staged Grace bank against the slab path on the same inputs (both TRT from HBM)."""
    global _stage_check_left
    if (_stage_check_left <= 0 or not tl_.n_c or topk_ids.numel() > tl_.stager.slots
            or torch.cuda.is_current_stream_capturing()):
        return
    import sys
    sys.path.insert(0, "/w")
    from trt_banks import mxfp8
    xq, xs = mxfp8(x)
    w = topk_weights.to(torch.bfloat16)
    a = tl_.grace_out(xq, xs, topk_ids, w, path="staged").float()
    b = tl_.grace_out(xq, xs, topk_ids, w, path="slab").float()
    if not bool(b.abs().amax() > 0):
        return
    _stage_check_left -= 1
    logger.info("hotsplit TRT stage check T=%d: staged-vs-slab rel_diff=%.2e cos=%.6f norm ratio %.5f", x.shape[0],
                float((a - b).norm() / b.norm()), float(torch.nn.functional.cosine_similarity(a.flatten(),
                                                                                            b.flatten(), dim=0)),
                float(a.norm() / b.norm()))


FUSED_SEND = os.environ.get("HOTSPLIT_FUSED_SEND", "1") == "1"
STAGE_SLOTS = int(os.environ.get("HOTSPLIT_STAGE_SLOTS", "0"))
STAGE_OVERLAP = os.environ.get("HOTSPLIT_STAGE_OVERLAP", "1") == "1"
_stage_check_left = int(os.environ.get("HOTSPLIT_STAGE_CHECK", "0"))
_STAGER = None
_SIDE = None


def _get_stager(gw13, gw2, gs13, gs2, E):
    """One staging set (STAGE_SLOTS experts) shared by every layer; all layers have the same expert geometry."""
    global _STAGER
    if _STAGER is None:
        import sys
        sys.path.insert(0, "/w")
        from stage_grace import GraceStager
        _STAGER = GraceStager(gw13, gw2, gs13, gs2, slots=STAGE_SLOTS, num_experts=E)
        logger.info("hotsplit Grace staging: %d slots, %.2f GiB HBM, %d copy programs, overlap %s",
                    STAGE_SLOTS, _STAGER.nbytes / 2**30, _STAGER.nprog, STAGE_OVERLAP)
    for mine, theirs in ((_STAGER.w13, gw13), (_STAGER.w2, gw2), (_STAGER.s13, gs13), (_STAGER.s2, gs2)):
        assert mine.shape[1:] == theirs.shape[1:] and mine.dtype == theirs.dtype, "Grace banks differ in geometry"
    return _STAGER


def _side_stream(device):
    global _SIDE
    if _SIDE is None:
        _SIDE = torch.cuda.Stream(device)
    return _SIDE


def _stage_check(grace_part, layer, x, topk_weights, topk_ids, staged_out):
    """Eager calls only: the staged bank against the UVA bank on the same inputs (same Marlin, same weight bytes)."""
    global _stage_check_left
    if _stage_check_left <= 0 or torch.cuda.is_current_stream_capturing():
        return
    ref = _run_part(*grace_part, layer, x, topk_weights, topk_ids)
    if not bool(ref.abs().amax() > 0):
        return
    _stage_check_left -= 1
    a, b = staged_out.float().flatten(), ref.float().flatten()
    logger.info("hotsplit stage check T=%d: staged-vs-UVA rel_diff=%.2e cos=%.6f max_abs=%.2e", x.shape[0],
                float((a - b).norm() / b.norm().clamp_min(1e-9)),
                float(torch.nn.functional.cosine_similarity(a, b, dim=0)), float((a - b).abs().max()))


PEER_CHECK = int(os.environ.get("HOTSPLIT_PEER_CHECK", "0"))
_peer_check_left = PEER_CHECK
_PEER_CHECK_MIN_T = int(os.environ.get("HOTSPLIT_PEER_CHECK_MIN_T", "1"))
_peer = None


def _peer_tier(device):
    global _peer
    if _peer is None:
        import sys
        sys.path.insert(0, "/w")
        import peer_tier_mimo
        _peer = peer_tier_mimo.PeerTier2(device)
        logger.info("hotsplit peer tier on: shm %s, max_rows=%d", peer_tier_mimo.PATH, _peer.max_rows)
    return _peer


def _mxfp8(x: torch.Tensor):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_impl
    return _mxfp8_e4m3_quantize_impl(x.contiguous(), is_sf_swizzled_layout=False)


def _run_part(kern, pw13, pw2, m, layer, x, topk_weights, topk_ids):
    return kern.apply(hidden_states=x, w1=pw13, w2=pw2, topk_weights=topk_weights, topk_ids=topk_ids,
                      activation=layer.activation, global_num_experts=layer.global_num_experts,
                      apply_router_weight_on_input=layer.apply_router_weight_on_input, expert_map=m,
                      shared_experts=None, shared_experts_input=None)


def _peer_check_ref(qm, layer, x, topk_weights, topk_ids, out):
    """Eager check calls only: (GB300-only output copy, Marlin's output for the peer experts from Grace)."""
    global _peer_check_left
    ref = getattr(qm, "_hotsplit_peer_ref", None)
    if (ref is None or _peer_check_left <= 0 or x.shape[0] < _PEER_CHECK_MIN_T
            or torch.cuda.is_current_stream_capturing()):
        return None
    ref_out = _run_part(*ref, layer, x, topk_weights, topk_ids)
    if not bool(ref_out.abs().amax() > 0):  # vLLM's memory-profiling dummy run: all-zero activations; don't spend a check
        return None
    _peer_check_left -= 1
    return out.clone(), ref_out


def _peer_check_log(li, T, rows, y_gb300, y_served, y_ref_peer):
    peer = (y_served.float() - y_gb300.float()).flatten()
    ref = y_ref_peer.float().flatten()
    rel = float((peer - ref).norm() / ref.norm().clamp_min(1e-9))
    cos = float(torch.nn.functional.cosine_similarity(peer, ref, dim=0))
    logger.info("hotsplit peer check layer=%d T=%d rows=%d peer-vs-marlin rel_diff=%.4f cos=%.5f "
                "(peer share of output norm %.3f)", li, T, int(rows.item()), rel, cos,
                float(ref.norm() / (y_gb300.float() + y_ref_peer.float()).norm().clamp_min(1e-9)))


class _LiveCounter:
    """Graph-safe per-layer routed-expert counter for the serving lane.
    index_add_ into a preallocated int64 device tensor (no host sync, no allocation) so it is captured
    into the decode CUDA graph and replays with it. A daemon thread snapshots to HOTSPLIT_LIVE_COUNTS
    every HOTSPLIT_LIVE_SECS (default 600) as {"live": {"decode": {layer: [E]}}, "meta": {...}}.
    Caveat: padded slots in a captured batch are counted too (C1 has no padding)."""

    def __init__(self, path: str, secs: int):
        self.path, self.secs = path, secs
        self.decode_max = int(os.environ.get("HOTSPLIT_LIVE_DECODE_MAX", "16"))
        self.buf = None
        self.layer_idx: list[int] = []
        self.ones = None

    def register(self, qm, li: int):
        slot = len(self.layer_idx)
        self.layer_idx.append(li)
        qm._hotsplit_live_slot = slot

    def finalize(self, E: int, max_tokens: int, topk: int):
        dev = torch.device("cuda")
        self.buf = torch.zeros((len(self.layer_idx), E), dtype=torch.int64, device=dev)
        self.ones = torch.ones(max_tokens * topk, dtype=torch.int64, device=dev)
        import threading
        threading.Thread(target=self._loop, daemon=True, name="hotsplit-live").start()
        logger.info("hotsplit live counter: %d layers -> %s every %ds", len(self.layer_idx), self.path, self.secs)

    def count(self, slot: int, topk_ids: torch.Tensor):
        ids = topk_ids.reshape(-1).to(torch.int64)
        self.buf[slot].index_add_(0, ids, self.ones[: ids.numel()])

    def _loop(self):
        t0 = time.time()
        while True:
            time.sleep(self.secs)
            try:
                snap = self.buf.to("cpu", non_blocking=False)
                out = {"live": {"decode": {str(li): snap[s].tolist() for s, li in enumerate(self.layer_idx)}},
                       "meta": {"since": t0, "at": time.time(), "note": "live lane counts incl. padded slots"}}
                tmp = self.path + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(out, f)
                os.replace(tmp, self.path)
            except Exception as e:  # never take the lane down for telemetry
                logger.warning("hotsplit live snapshot failed: %s", e)


_LIVE = _LiveCounter(os.environ["HOTSPLIT_LIVE_COUNTS"], int(os.environ.get("HOTSPLIT_LIVE_SECS", "600"))) \
    if os.environ.get("HOTSPLIT_LIVE_COUNTS") else None
