"""MegaMoE hot tier + cold tier for DeepSeek-V4.1-Flash on one GB300 (+ RTX PRO 6000).

Each of the 40 target MoE layers keeps its 295 hot experts (static rowmap) in HBM as a
DeepGEMM MegaMoE module (shared expert fused). The 89 cold experts never reach HBM:
the loader stages them on the host, and finalize converts them to FlashInfer TRT-LLM
layout in pinned Grace memory (UVA views). Per forward:

  MEGA_PEER=1 and MEGA_PEER_MIN_TOKENS <= T <= 64:
      publish activations + cold routes to the 6000 sidecar, run MegaMoE (hot + shared),
      wait for the sidecar, add its cold output.            (decode / verify)
  otherwise:
      MegaMoE (hot + shared), then TRT-LLM routed on the Grace-resident cold experts,
      unfinalized, added in fp32.                            (prefill)

The DSpark drafter's MoE layers (128 experts, layers 40+) are untouched.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import re
import sys

import torch

E = 384
TOPK = 6
HIDDEN = 5120
LIMIT = 10.0
TUNE_MAX = int(os.environ.get("MEGA_TUNE_MAX", "8192"))
ROWMAP_PATH = os.environ.get("MEGA_ROWMAP", "/w/rowmap-static-v1.json")
PEER_MODE = os.environ.get("MEGA_PEER", "0")
PEER = PEER_MODE == "1"    # v1: T <= 64 to the Triton sidecar
PEER2 = PEER_MODE == "2"   # v2: compacted rows, any T, b12x sidecar
# MEGA_PEER_CHECK=N: for the first N eager calls with T > 64, also run the TRT-LLM cold path and log the difference.
_check_left = int(os.environ.get("MEGA_PEER_CHECK", "0"))
CHECK_MIN_T = int(os.environ.get("MEGA_PEER_CHECK_MIN_T", "65"))
# MEGA_COUNT=<path>: count routed tokens per (layer, expert) on device, dump JSON to <path> every 20 s.
COUNT_PATH = os.environ.get("MEGA_COUNT", "")
# MEGA_COLD_TRT=0: with peer v2, skip the Grace copy of the cold experts (62 GiB pinned). It only backs the
# TRT-LLM fallback (batches over 8,192 tokens, which max_num_batched_tokens rules out) and MEGA_PEER_CHECK.
COLD_TRT = os.environ.get("MEGA_COLD_TRT", "1") != "0"
_states: list = []
_count_host = None     # pinned int64 [40, E]; filled from the engine thread, written to disk by the dumper
_count_copied = 0.0
PEER_MIN_TOKENS = int(os.environ.get("MEGA_PEER_MIN_TOKENS", "1"))
TARGET = "vllm.models.deepseek_v4.nvidia.model"
_PREFIX = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.ffn\.experts$")

_LOG = lambda m: sys.stderr.write(f"MEGA_PEER {m}\n")

_rowmap = None
_cpu_keep: list = []
_perm_cache: dict = {}
_peer = None
_peer2 = None
_pt = None
_add_cold_kernel = None
_orig = {}


def _load_rowmap() -> dict:
    global _rowmap
    if _rowmap is None:
        with open(ROWMAP_PATH) as f:
            raw = json.load(f)
        layers = raw["layers"] if "layers" in raw else raw
        _rowmap = {int(k): {"hot": [int(x) for x in v["hot"]], "cold": [int(x) for x in v["cold"]]}
                   for k, v in layers.items()}
        for idx, v in _rowmap.items():
            if set(v["hot"]) | set(v["cold"]) != set(range(E)) or set(v["hot"]) & set(v["cold"]):
                raise RuntimeError(f"MEGA_PEER rowmap layer {idx} is not a partition of 0..{E - 1}")
        _LOG(f"rowmap {ROWMAP_PATH}: {len(_rowmap)} layers")
    return _rowmap


class ColdState:
    """Per-layer split: row[e] = hot row, or -(cold row + 1)."""

    def __init__(self, idx: int, hot: list[int], cold: list[int]):
        self.idx = idx
        self.hot = hot
        self.cold = cold
        self.n_hot = len(hot)
        self.n_cold = len(cold)
        self.row = [0] * E
        for r, e in enumerate(hot):
            self.row[e] = r
        for c, e in enumerate(cold):
            self.row[e] = -(c + 1)
        self.row_map = None   # int32 [E] on device, built at finalize
        self.raw = {}         # host staging of cold experts in checkpoint layout
        self.trt = None       # cold experts in TRT-LLM layout, UVA views of pinned host
        self.clamp = None
        self.inter = 0
        self.counts = None    # int64 [E] routed tokens per expert (MEGA_COUNT)


def _target_layer(prefix: str, num_experts: int, num_local_experts: int):
    m = _PREFIX.search(prefix)
    if m is None or num_experts != E or num_local_experts != E:
        return None
    idx = int(m.group(1))
    return idx if idx in _load_rowmap() else None


# ---------------------------------------------------------------------------
# DeepseekV4MegaMoEExperts patches
# ---------------------------------------------------------------------------


def _init(self, vllm_config, *, num_experts, num_local_experts, experts_start_idx, **kw):
    idx = _target_layer(kw.get("prefix", ""), num_experts, num_local_experts)
    if idx is None:
        _orig["init"](self, vllm_config, num_experts=num_experts, num_local_experts=num_local_experts,
                      experts_start_idx=experts_start_idx, **kw)
        self._mp = None
        return
    split = _load_rowmap()[idx]
    n_hot = len(split["hot"])
    kw["num_logical_experts"] = E
    _orig["init"](self, vllm_config, num_experts=n_hot, num_local_experts=n_hot,
                  experts_start_idx=0, **kw)
    self._mp = ColdState(idx, split["hot"], split["cold"])
    self._mp.inter = self.intermediate_size
    if idx == 0:
        _LOG(f"layer 0 ({kw.get('prefix')}): {n_hot} hot experts in MegaMoE, "
             f"{E - n_hot} cold on the host; peer mode {PEER_MODE}")


def _param_key(self, param) -> str:
    for key in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"):
        if param is getattr(self, key):
            return key
    raise ValueError("MEGA_PEER: unknown MegaMoE parameter")


def _weight_loader(self, param, loaded_weight, weight_name, shard_id, expert_id,
                   return_success=False):
    st = getattr(self, "_mp", None)
    if st is None:
        return _orig["weight_loader"](self, param, loaded_weight, weight_name, shard_id,
                                      expert_id, return_success)
    r = st.row[expert_id]
    if r >= 0:
        dst = param.data[r]
    elif not COLD_TRT:
        return True if return_success else None  # cold experts live only on the 6000
    else:
        key = _param_key(self, param)
        buf = st.raw.get(key)
        if buf is None:
            # Explicit device: the loader may run under a CUDA default-device context.
            buf = torch.zeros((st.n_cold,) + tuple(param.shape[1:]), dtype=param.dtype,
                              device="cpu")
            st.raw[key] = buf
        dst = buf[-r - 1]
    if shard_id in ("w1", "w3"):
        if "w13_" not in weight_name:
            return False if return_success else None
        offset = 0 if shard_id == "w1" else self.intermediate_size
        dst = dst.narrow(0, offset, self.intermediate_size)
    elif shard_id == "w2":
        if "w2_" not in weight_name:
            return False if return_success else None
    else:
        raise ValueError(f"Unsupported expert shard id: {shard_id}")
    if dst.shape != loaded_weight.shape:
        raise ValueError(f"MEGA_PEER shape mismatch for {weight_name}: {tuple(dst.shape)} "
                         f"vs {tuple(loaded_weight.shape)}")
    dst.copy_(loaded_weight)
    return True if return_success else None


_weight_loader.supports_moe_loading = True


def _uva(t: torch.Tensor) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    cpu = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
    cpu.copy_(t)
    _cpu_keep.append(cpu)
    return get_accelerator_view_from_cpu_tensor(cpu)


def _convert_cold(self, st: ColdState) -> None:
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format)

    dev = torch.device("cuda", torch.cuda.current_device())
    if not COLD_TRT:
        _finish_cold(st, dev)
        return
    missing = {"w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"} - set(st.raw)
    if missing:
        raise RuntimeError(f"MEGA_PEER layer {st.idx}: cold weights never loaded: {missing}")
    raw = {k: v.to(dev) for k, v in st.raw.items()}
    w13, w2, s13, s2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, layer=None,
        w13_weight=raw["w13_weight"], w2_weight=raw["w2_weight"],
        w13_weight_scale=raw["w13_weight_scale"], w2_weight_scale=raw["w2_weight_scale"],
        _cache_permute_indices=_perm_cache)
    st.trt = {"w1": _uva(w13), "s1": _uva(s13), "w2": _uva(w2), "s2": _uva(s2)}
    st.clamp = torch.full((st.n_cold,), LIMIT, dtype=torch.float32, device=dev)
    del raw, w13, w2, s13, s2
    _finish_cold(st, dev)
    if st.idx in (0, 39):
        free = torch.cuda.mem_get_info()[0] / 1024**3
        _LOG(f"layer {st.idx}: cold experts in TRT-LLM layout on Grace "
             f"({sum(t.numel() for t in st.trt.values()) / 1e9:.2f} GB); HBM free {free:.1f} GiB")


def _finish_cold(st: ColdState, dev) -> None:
    row_map = torch.tensor(st.row, dtype=torch.int32, device="cpu")
    st.row_map = row_map.to(dev)
    if COUNT_PATH:
        st.counts = torch.zeros(E, dtype=torch.int64, device=dev)
        _states.append(st)
        if st.idx == 0:
            _start_count_dumper(dev)
    st.raw = {}
    torch.cuda.empty_cache()


def _start_count_dumper(device):
    """Write the pinned host copy of the counts every 20 s. The thread makes no CUDA calls: a CUDA call from
    another thread during vLLM's graph capture invalidates the capture."""
    import threading
    import time

    global _count_host
    _count_host = torch.zeros(40, E, dtype=torch.int64, device="cpu", pin_memory=True)

    def loop():
        while True:
            time.sleep(20)
            try:
                data = {str(st.idx): _count_host[st.idx].tolist() for st in _states}
                tmp = COUNT_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"layers": data, "time": time.time()}, f)
                os.replace(tmp, COUNT_PATH)
            except Exception as exc:  # keep serving; report once per failure
                _LOG(f"count dump failed: {exc!r}")

    threading.Thread(target=loop, daemon=True, name="mega-count-dump").start()
    _LOG(f"route counting on, dumping to {COUNT_PATH}")


def _maybe_copy_counts(st) -> None:
    """From the engine thread, at most every 10 s, outside graph capture: async copy of all layers' counts."""
    global _count_copied
    import time

    if st.idx != len(_states) - 1 or _count_host is None or torch.cuda.is_current_stream_capturing():
        return
    now = time.time()
    if now - _count_copied < 10:
        return
    _count_copied = now
    for s in _states:
        _count_host[s.idx].copy_(s.counts, non_blocking=True)


def _finalize_weights(self, shared_experts=None):
    _orig["finalize_weights"](self, shared_experts)
    st = getattr(self, "_mp", None)
    if st is not None and st.trt is None:
        _convert_cold(self, st)


def _quant(x: torch.Tensor):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_impl)

    return _mxfp8_e4m3_quantize_impl(x.contiguous(), is_sf_swizzled_layout=False)


def _peer_tier(device):
    global _peer, _pt
    if _peer is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import peer_tier

        _pt = peer_tier
        _peer = peer_tier.PeerTier(device)
        _LOG(f"peer tier on: max_tokens={_peer.max_tokens} min_tokens={PEER_MIN_TOKENS}")
    return _peer


def _peer_tier2(device):
    global _peer2
    if _peer2 is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import peer_tier2

        _peer2 = peer_tier2.PeerTier2(device)
        _LOG(f"peer tier v2 on: max_rows={_peer2.max_rows}")
    return _peer2


def _log_check(st, T, rows, y_hot, y_peer, y_trt):
    cold_peer = (y_peer.float() - y_hot.float()).flatten()
    cold_trt = (y_trt.float() - y_hot.float()).flatten()
    rel = float((cold_peer - cold_trt).norm() / cold_trt.norm().clamp_min(1e-9))
    cos = float(torch.nn.functional.cosine_similarity(cold_peer, cold_trt, dim=0))
    _LOG(f"check layer={st.idx} T={T} rows={int(rows.item())} cold rel_diff={rel:.4f} cos={cos:.5f} "
         f"(cold share of output norm {float(cold_trt.norm() / y_trt.float().norm()):.3f})")


def _get_add_cold():
    """y[t] = bf16(float(y[t]) + sum_k w[t,k] * g2[e2p[t,k]]) over live cold routes."""
    global _add_cold_kernel
    if _add_cold_kernel is not None:
        return _add_cold_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _k(y_ptr, g2_ptr, e2p_ptr, ids_ptr, w_ptr, K, H, stride_g2, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(K):
            p = tl.load(e2p_ptr + t * K + k)
            live = (p >= 0) & (tl.load(ids_ptr + t * K + k) >= 0)
            w = tl.load(w_ptr + t * K + k).to(tl.float32)
            row = tl.load(g2_ptr + tl.where(live, p, 0).to(tl.int64) * stride_g2 + offs,
                          mask=mask & live, other=0.0)
            acc = tl.math.fma(w, row.to(tl.float32), acc)
        y = tl.load(y_ptr + t * H + offs, mask=mask).to(tl.float32)
        tl.store(y_ptr + t * H + offs, (y + acc).to(tl.bfloat16), mask=mask)

    def launch(y, g2, e2p, ids, w):
        T, K = ids.shape
        H = y.shape[-1]
        _k[(T, triton.cdiv(H, 1024))](y, g2, e2p.contiguous(), ids.contiguous(),
                                      w.float().contiguous(), K, H, g2.stride(0), BLOCK=1024)

    _add_cold_kernel = launch
    return launch


def _unfinalized_parts(call_out, T: int):
    if isinstance(call_out, (list, tuple)) and len(call_out) >= 3:
        g2, e2p = call_out[0], call_out[2]
    else:
        g2 = getattr(call_out, "gemm2_permuted", None)
        e2p = getattr(call_out, "expanded_idx_to_permuted_idx", None)
        if g2 is None or e2p is None:
            raise RuntimeError(f"MEGA_PEER unexpected unfinalized output {type(call_out)}")
    e2p = e2p.reshape(-1)[: T * TOPK].view(T, TOPK)
    if g2.shape[1] != HIDDEN:
        g2 = g2[:, :HIDDEN]
    return g2, e2p


def _cold_trt(st: ColdState, y, x, cold_ids, topk_weights):
    from flashinfer import trtllm_fp4_block_scale_routed_moe
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    T = x.shape[0]
    xq, xs = _quant(x)
    c = st.trt
    out = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(cold_ids, topk_weights.to(torch.bfloat16)), routing_bias=None,
        hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=c["w1"], gemm1_weights_scale=c["s1"], gemm1_bias=None, gemm1_alpha=None,
        gemm1_beta=None, gemm1_clamp_limit=st.clamp, gemm2_weights=c["w2"],
        gemm2_weights_scale=c["s2"], gemm2_bias=None, output1_scale_scalar=None,
        output1_scale_gate_scalar=None, output2_scale_scalar=None, num_experts=st.n_cold,
        top_k=TOPK, n_group=None, topk_group=None, intermediate_size=st.inter,
        local_expert_offset=0, local_num_experts=st.n_cold, routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize, do_finalize=False, enable_pdl=True,
        tune_max_num_tokens=TUNE_MAX)
    g2, e2p = _unfinalized_parts(out, T)
    _get_add_cold()(y, g2, e2p, cold_ids, topk_weights)


def _forward(self, hidden_states, topk_weights, topk_ids, *, activation_clamp, fast_math=True):
    st = getattr(self, "_mp", None)
    if st is None:
        return _orig["forward"](self, hidden_states, topk_weights, topk_ids,
                                activation_clamp=activation_clamp, fast_math=fast_math)
    T = hidden_states.shape[0]
    valid = topk_ids >= 0
    if st.counts is not None:
        st.counts.index_add_(0, topk_ids.clamp_min(0).flatten().long(), valid.flatten().long())
        _maybe_copy_counts(st)
    rm = st.row_map[topk_ids.clamp_min(0).long()]
    hot_ids = torch.where(valid & (rm >= 0), rm, -1).to(topk_ids.dtype)
    cold_ids = torch.where(valid & (rm < 0), -rm - 1, -1).to(torch.int32)

    if PEER2 and (T <= 8192 or st.trt is None):
        global _check_left
        peer2 = _peer_tier2(hidden_states.device)
        xq, xs = _quant(hidden_states)
        has, pos, rows = peer2.send(xq, xs, cold_ids, topk_weights, st.idx)
        y = _orig["forward"](self, hidden_states, topk_weights, hot_ids,
                             activation_clamp=activation_clamp, fast_math=fast_math)
        check = (_check_left > 0 and st.trt is not None and T >= CHECK_MIN_T
                 and not torch.cuda.is_current_stream_capturing())
        if check:
            _check_left -= 1
            y_hot = y.clone()
            y_trt = y.clone()
            _cold_trt(st, y_trt, hidden_states, cold_ids, topk_weights)
        peer2.finish(y, has, pos, rows)
        if check:
            _log_check(st, T, rows, y_hot, y, y_trt)
        return y

    peer = _peer_tier(hidden_states.device) if PEER else None
    if peer is not None and PEER_MIN_TOKENS <= T <= peer.max_tokens:
        pt = _pt
        xq, xs = _quant(hidden_states)
        count = (cold_ids >= 0).sum(dtype=torch.int32).reshape(1)
        peer.x[: xq.numel()].copy_(xq.view(torch.uint8).reshape(-1))
        peer.xs[: xs.numel()].copy_(xs.view(torch.uint8).reshape(-1))
        peer.ids[: cold_ids.numel()].copy_(cold_ids.reshape(-1))
        peer.w[: topk_weights.numel()].copy_(topk_weights.float().reshape(-1))
        pt._publish[(1,)](peer.words, peer.header, peer.seq, st.idx, T, count)
        y = _orig["forward"](self, hidden_states, topk_weights, hot_ids,
                             activation_clamp=activation_clamp, fast_math=fast_math)
        pt._wait[(1,)](peer.words, peer.seq, count, TIMEOUT=peer.timeout)
        n = T * HIDDEN
        pt._accumulate[((n + 1023) // 1024,)](y, peer.out, count, N=n, BLOCK=1024)
        return y

    y = _orig["forward"](self, hidden_states, topk_weights, hot_ids,
                         activation_clamp=activation_clamp, fast_math=fast_math)
    _cold_trt(st, y, hidden_states, cold_ids, topk_weights)
    return y


def _patch(module):
    cls = module.DeepseekV4MegaMoEExperts
    if getattr(cls, "_mega_peer", False):
        return
    for name, fn in (("init", _init), ("weight_loader", _weight_loader),
                     ("finalize_weights", _finalize_weights), ("forward", _forward)):
        attr = "__init__" if name == "init" else name
        _orig[name] = getattr(cls, attr)
        setattr(cls, attr, fn)
    cls._mega_peer = True
    _LOG(f"patched {TARGET}.DeepseekV4MegaMoEExperts (peer mode {PEER_MODE}, check {_check_left})")


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig_exec=orig_exec):
            _orig_exec(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


def install():
    if TARGET in sys.modules:
        _patch(sys.modules[TARGET])
    else:
        sys.meta_path.insert(0, _Finder())
