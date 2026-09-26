"""Pin-hot-experts hook for DeepSeek-V4.1-Flash MXFP4 MoE.

PIN_MODE=count — accumulate per-layer expert bincounts on device (E2a).
PIN_MODE=off   — pure pass-through (control).
PIN_MODE=split — F4 split: re-home 295 hot / 89 cold per layer, two
                 unfinalized routed-kernel calls + one fp32-FMA finalize.

Graph-safe on the hot path: no allocation / .item() / host sync inside a
CUDA-graph body. Count dumps are eager-only.
"""
from __future__ import annotations

import atexit
import gc
import json
import os
import sys
import threading
import time

import torch

E = 384
N_HOT = 295
N_COLD = 89
TOPK = 6
HIDDEN = 5120
DECODE_T = 16  # T <= 16 ≈ decode/verify when not capturing
MAX_N = int(os.environ.get("PIN_MAX_N", "49152"))  # 8192 tokens * top_k=6
DUMP_DIR = os.environ.get("PIN_DUMP_DIR", "/tmp/pin-hot-experts/e2a")
DUMP_SEC = float(os.environ.get("PIN_DUMP_SEC", "300"))
SENTINEL = os.path.join(DUMP_DIR, "DUMP")
MODE = os.environ.get("PIN_MODE", "").strip().lower()
ROWMAP_PATH = os.environ.get("PIN_ROWMAP", "/w/rowmap-static-v1.json")
MIN_HBM_FREE = 8 * (1024**3)
MIN_HOST_AVAIL = 10 * (1024**3)

_LOG = lambda m: sys.stderr.write(f"PIN_HOT {m}\n")

_lock = threading.Lock()
_layers = {}  # id(self) -> LayerState
_order = []  # first-seen order of id(self)
_idbuf = None
_wbuf = None
_ready = False
_want_dump = False
_dumped_n = 0
_orig_invoke = None
_installed = False

# split-mode
_cpu_keep: list = []  # pinned host tensors that back UVA views
_rowmap = None  # dict[int, dict]
_finalize_kernel = None
_rehomed = False
_PIN_BY_W1: dict = {}  # hot w1 data_ptr -> PinState


class LayerState:
    __slots__ = ("idx", "name", "dec", "pre", "n_dec", "n_pre")

    def __init__(self, idx: int, name: str, device):
        self.idx = idx
        self.name = name
        self.dec = torch.zeros(E, dtype=torch.int64, device=device)
        self.pre = torch.zeros(E, dtype=torch.int64, device=device)
        self.n_dec = torch.zeros((), dtype=torch.int64, device=device)
        self.n_pre = torch.zeros((), dtype=torch.int64, device=device)


class PinState:
    """Per-Modular-experts F4 partitions. All device tensors; CPU refs in _cpu_keep."""

    __slots__ = (
        "idx",
        "name",
        "row_map",
        "hot",
        "cold",
        "n_hot",
        "n_cold",
        "is_uva",
    )

    def __init__(self, idx: int, name: str):
        self.idx = idx
        self.name = name
        self.row_map = None  # int32 [384]: >=0 hot row, <0 -> cold row = -v-1
        self.hot = {}
        self.cold = {}
        self.n_hot = N_HOT
        self.n_cold = N_COLD
        self.is_uva = False


def _layer_name(mod) -> str:
    for attr in ("layer_idx", "_layer_idx", "layer_id", "layer_name"):
        v = getattr(mod, attr, None)
        if v is not None:
            return f"{attr}={v}"
    n = getattr(mod, "_pin_name", None)
    if n:
        return n
    return f"id={id(mod)}"


# ---------------------------------------------------------------------------
# Count mode (E2a) — unchanged behaviour
# ---------------------------------------------------------------------------


def _ensure_bufs(device, n: int):
    """Allocate workspaces on first non-capturing call. Illegal during capture."""
    global _idbuf, _wbuf, _ready
    if _ready:
        if n > _idbuf.numel():
            raise RuntimeError(f"PIN_HOT n={n} exceeds PIN_MAX_N={_idbuf.numel()}")
        return
    if torch.cuda.is_current_stream_capturing():
        return
    _idbuf = torch.zeros(MAX_N, dtype=torch.int64, device=device)
    _wbuf = torch.zeros(MAX_N, dtype=torch.int64, device=device)
    torch.cuda.synchronize()
    _ready = True
    _LOG(f"workspace allocated MAX_N={MAX_N} on {device}")


def _get_layer(mod, device) -> LayerState | None:
    key = id(mod)
    st = _layers.get(key)
    if st is not None:
        return st
    if torch.cuda.is_current_stream_capturing():
        return None
    idx = len(_order)
    name = _layer_name(mod)
    st = LayerState(idx, name, device)
    _layers[key] = st
    _order.append(key)
    _LOG(f"layer registered idx={idx} name={name}")
    return st


def _accumulate(mod, topk_ids: torch.Tensor) -> None:
    """Device-only scatter_add. Ignore ids < 0. No host sync."""
    if topk_ids is None:
        return
    ids = topk_ids
    if isinstance(ids, (tuple, list)):
        ids = ids[0]
    if not torch.is_tensor(ids):
        return
    T = int(ids.shape[0])
    flat = ids.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return
    device = ids.device
    _ensure_bufs(device, n)
    st = _get_layer(mod, device)
    if st is None or not _ready:
        return
    capturing = torch.cuda.is_current_stream_capturing()
    is_decode = capturing or T <= DECODE_T
    idv = _idbuf[:n]
    wv = _wbuf[:n]
    idv.copy_(flat)
    wv.copy_(idv)
    wv.add_(1)
    wv.clamp_(0, 1)
    idv.clamp_(0, E - 1)
    dst = st.dec if is_decode else st.pre
    dst.scatter_add_(0, idv, wv)
    (st.n_dec if is_decode else st.n_pre).add_(1)


def _snapshot():
    """Host copy. MUST NOT run during capture. Caller holds no graph."""
    if not _layers:
        return None
    torch.cuda.synchronize()
    L = len(_order)
    dec = torch.zeros(L, E, dtype=torch.int64)
    pre = torch.zeros(L, E, dtype=torch.int64)
    names = []
    n_dec = []
    n_pre = []
    for i, key in enumerate(_order):
        st = _layers[key]
        dec[i].copy_(st.dec)
        pre[i].copy_(st.pre)
        names.append(st.name)
        n_dec.append(int(st.n_dec.item()))
        n_pre.append(int(st.n_pre.item()))
    return {
        "dec": dec.numpy(),
        "pre": pre.numpy(),
        "all": (dec + pre).numpy(),
        "names": names,
        "n_dec": n_dec,
        "n_pre": n_pre,
    }


def dump(reason: str = "periodic") -> str | None:
    global _dumped_n, _want_dump
    with _lock:
        try:
            if torch.cuda.is_current_stream_capturing():
                _want_dump = True
                return None
            snap = _snapshot()
            if snap is None:
                return None
            os.makedirs(DUMP_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            path = os.path.join(DUMP_DIR, f"counts-{ts}.npz")
            import numpy as np

            np.savez_compressed(
                path,
                cnt_dec=snap["dec"],
                cnt_pre=snap["pre"],
                cnt_all=snap["all"],
                n_dec=np.asarray(snap["n_dec"], dtype=np.int64),
                n_pre=np.asarray(snap["n_pre"], dtype=np.int64),
                names=np.asarray(snap["names"]),
                decode_T=np.int64(DECODE_T),
                E=np.int64(E),
                L=np.int64(len(snap["names"])),
                reason=np.asarray(reason),
                ts=np.asarray(ts),
            )
            _dumped_n += 1
            _want_dump = False
            if os.path.exists(SENTINEL):
                try:
                    os.remove(SENTINEL)
                except OSError:
                    pass
            tot_dec = int(snap["dec"].sum())
            tot_pre = int(snap["pre"].sum())
            _LOG(
                f"dump {path} reason={reason} L={len(snap['names'])} "
                f"sel_dec={tot_dec} sel_pre={tot_pre}"
            )
            return path
        except Exception as e:
            _LOG(f"dump error {e!r}")
            return None


def _maybe_dump_eager():
    global _want_dump
    if torch.cuda.is_current_stream_capturing():
        return
    if _want_dump or os.path.exists(SENTINEL):
        dump("eager" if _want_dump else "sentinel")


def _bg_thread():
    global _want_dump
    while True:
        time.sleep(DUMP_SEC)
        _want_dump = True


def _invoke_count(
    self,
    output,
    x_quant,
    x_scale,
    topk_ids,
    topk_weights,
    w1,
    w2,
    activation,
    global_num_experts,
    local_num_experts,
    local_expert_offset,
    topk,
):
    try:
        _accumulate(self, topk_ids)
        _maybe_dump_eager()
    except Exception as e:
        if not torch.cuda.is_current_stream_capturing():
            _LOG(f"count error {e!r}")
    return _orig_invoke(
        self,
        output,
        x_quant,
        x_scale,
        topk_ids,
        topk_weights,
        w1,
        w2,
        activation,
        global_num_experts,
        local_num_experts,
        local_expert_offset,
        topk,
    )


def _invoke_off(
    self,
    output,
    x_quant,
    x_scale,
    topk_ids,
    topk_weights,
    w1,
    w2,
    activation,
    global_num_experts,
    local_num_experts,
    local_expert_offset,
    topk,
):
    return _orig_invoke(
        self,
        output,
        x_quant,
        x_scale,
        topk_ids,
        topk_weights,
        w1,
        w2,
        activation,
        global_num_experts,
        local_num_experts,
        local_expert_offset,
        topk,
    )


# ---------------------------------------------------------------------------
# Split mode — F4
# ---------------------------------------------------------------------------


def mem_available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def hbm_free_bytes(device=None) -> int:
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)


def _budget_ok(where: str) -> None:
    hbm = hbm_free_bytes()
    host = mem_available_bytes()
    if hbm < MIN_HBM_FREE:
        raise RuntimeError(
            f"PIN_HOT abort {where}: HBM free {hbm/1024**3:.2f} GiB < 8 GiB"
        )
    if host < MIN_HOST_AVAIL:
        raise RuntimeError(
            f"PIN_HOT abort {where}: MemAvailable {host/1024**3:.2f} GiB < 10 GiB"
        )


def load_rowmap(path: str | None = None) -> dict:
    path = path or ROWMAP_PATH
    with open(path) as f:
        raw = json.load(f)
    layers = raw["layers"] if isinstance(raw, dict) and "layers" in raw else raw
    out = {}
    for k, v in layers.items():
        idx = int(k)
        hot = [int(x) for x in v["hot"]]
        cold = [int(x) for x in v["cold"]]
        if len(hot) != N_HOT or len(cold) != N_COLD:
            raise RuntimeError(
                f"PIN_HOT rowmap layer {idx}: hot={len(hot)} cold={len(cold)} "
                f"expected {N_HOT}/{N_COLD}"
            )
        if set(hot) & set(cold):
            raise RuntimeError(f"PIN_HOT rowmap layer {idx}: hot/cold overlap")
        if set(hot) | set(cold) != set(range(E)):
            raise RuntimeError(f"PIN_HOT rowmap layer {idx}: hot∪cold != 0..383")
        out[idx] = {"hot": hot, "cold": cold}
    if len(out) != 40:
        _LOG(f"rowmap has {len(out)} layers (expected 40)")
    return out


def build_row_map_tensor(hot: list[int], cold: list[int], device) -> torch.Tensor:
    """int32 [384]: value = hot row, or -(cold_row+1)."""
    rm = torch.empty(E, dtype=torch.int32, device=device)
    for local, eid in enumerate(hot):
        rm[eid] = local
    for local, eid in enumerate(cold):
        rm[eid] = -(local + 1)
    return rm


def split_ids(ids: torch.Tensor, row_map: torch.Tensor):
    """Graph-safe remap. ids [T,K] int32, row_map [384] int32.
    Returns hot_ids, cold_ids (both [T,K] int32, -1 for skip)."""
    valid = ids >= 0
    clamped = ids.clamp_min(0).long()
    rm = row_map[clamped]
    is_hot = valid & (rm >= 0)
    is_cold = valid & (rm < 0)
    neg1 = torch.full_like(ids, -1)
    hot_ids = torch.where(is_hot, rm, neg1)
    cold_ids = torch.where(is_cold, -rm - 1, neg1)
    return hot_ids, cold_ids


def _uva_view(cpu: torch.Tensor) -> torch.Tensor:
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    if not cpu.is_pinned():
        cpu = cpu.pin_memory()
    return get_accelerator_view_from_cpu_tensor(cpu)


def partition_leading_e(src: torch.Tensor, hot_idx: torch.Tensor, cold_idx: torch.Tensor):
    """Copy dim-0 rows into HBM hot and pinned-host cold (UVA view)."""
    if src is None:
        return None, None, None
    hot = src.index_select(0, hot_idx).contiguous()
    if hot.device.type != "cuda" or getattr(src, "_vllm_is_uva_offloaded", False):
        hot_hbm = torch.empty(hot.shape, dtype=hot.dtype, device="cuda")
        hot_hbm.copy_(hot)
        hot = hot_hbm
    else:
        # Ensure we own a fresh HBM allocation, not a view of the original.
        hot = hot.clone()
    cold_src = src.index_select(0, cold_idx).contiguous()
    cpu = torch.empty(cold_src.shape, dtype=cold_src.dtype, pin_memory=True)
    cpu.copy_(cold_src)
    view = _uva_view(cpu)
    return hot, view, cpu


def nbytes(t) -> int:
    if t is None:
        return 0
    return int(t.numel() * t.element_size())


def _get_finalize():
    """Triton fmaf finalize: acc = fma(w_k, float(row_k), acc) in k order, one bf16 cast."""
    global _finalize_kernel
    if _finalize_kernel is not None:
        return _finalize_kernel
    import triton
    import triton.language as tl

    @triton.jit
    def _k(
        g2h_ptr,
        e2ph_ptr,
        g2c_ptr,
        e2pc_ptr,
        w_ptr,
        out_ptr,
        T,
        K,
        H,
        stride_g2h,
        stride_g2c,
        BLOCK: tl.constexpr,
    ):
        t = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < H
        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(K):
            eh = tl.load(e2ph_ptr + t * K + k)
            ec = tl.load(e2pc_ptr + t * K + k)
            w = tl.load(w_ptr + t * K + k).to(tl.float32)
            valid_h = eh >= 0
            valid_c = ec >= 0
            eh_i = tl.where(valid_h, eh, 0)
            ec_i = tl.where(valid_c, ec, 0)
            row_h = tl.load(
                g2h_ptr + eh_i * stride_g2h + offs,
                mask=mask & valid_h,
                other=0.0,
            )
            row_c = tl.load(
                g2c_ptr + ec_i * stride_g2c + offs,
                mask=mask & valid_c,
                other=0.0,
            )
            row = row_h.to(tl.float32) + row_c.to(tl.float32)
            acc = tl.math.fma(w, row, acc)
        tl.store(out_ptr + t * H + offs, acc.to(tl.bfloat16), mask=mask)

    def launch(g2h, e2ph, g2c, e2pc, wts, out):
        T, K = e2ph.shape
        H = out.shape[-1]
        stride_h = g2h.stride(0)
        stride_c = g2c.stride(0)
        w = wts if wts.dtype == torch.float32 else wts.float()
        BLOCK = 1
        while BLOCK < H:
            BLOCK *= 2
        _k[(T,)](
            g2h,
            e2ph.contiguous(),
            g2c,
            e2pc.contiguous(),
            w.contiguous(),
            out,
            T,
            K,
            H,
            stride_h,
            stride_c,
            BLOCK=BLOCK,
        )
        return out

    _finalize_kernel = launch
    return launch


def finalize_f4(g2h, e2ph, g2c, e2pc, wts, out):
    """Write bf16 [T,H] into out. Graph-safe (fixed T)."""
    fn = _get_finalize()
    return fn(g2h, e2ph, g2c, e2pc, wts, out)


def _unfinalized_parts(call_out, T: int, topk: int, hidden: int):
    """Normalize flashinfer do_finalize=False return to (g2, e2p[T,K])."""
    if isinstance(call_out, (list, tuple)) and len(call_out) >= 3:
        g2, _expw, e2p = call_out[0], call_out[1], call_out[2]
    else:
        g2 = getattr(call_out, "gemm2_permuted", None)
        e2p = getattr(call_out, "expanded_idx_to_permuted_idx", None)
        if g2 is None or e2p is None:
            raise RuntimeError(f"PIN_HOT unexpected unfinalized type {type(call_out)}")
    if e2p.numel() >= T * topk:
        e2p = e2p.reshape(-1)[: T * topk].view(T, topk)
    if g2.ndim != 2:
        raise RuntimeError(f"PIN_HOT gemm2 ndim={g2.ndim} shape={tuple(g2.shape)}")
    if g2.shape[1] != hidden:
        g2 = g2[:, :hidden]
    return g2, e2p


def call_routed(
    *,
    x_quant,
    x_scale,
    ids,
    wts,
    w1,
    s1,
    b1,
    a1,
    beta1,
    clamp1,
    w2,
    s2,
    b2,
    n_exp,
    topk,
    activation_type,
    tune_max,
    enable_pdl=True,
    do_finalize=False,
    output=None,
    intermediate_size=None,
):
    from flashinfer import trtllm_fp4_block_scale_routed_moe
    from vllm.model_executor.layers.fused_moe.config import RoutingMethodType

    if intermediate_size is None:
        intermediate_size = int(w1.shape[1]) // 2
    return trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, wts),
        routing_bias=None,
        hidden_states=x_quant,
        hidden_states_scale=x_scale,
        gemm1_weights=w1,
        gemm1_weights_scale=s1,
        gemm1_bias=b1,
        gemm1_alpha=a1,
        gemm1_beta=beta1,
        gemm1_clamp_limit=clamp1,
        gemm2_weights=w2,
        gemm2_weights_scale=s2,
        gemm2_bias=b2,
        output1_scale_scalar=None,
        output1_scale_gate_scalar=None,
        output2_scale_scalar=None,
        num_experts=n_exp,
        top_k=topk,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=n_exp,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        output=output,
        tune_max_num_tokens=tune_max,
    )


_PEER_TIER = None
_PEER_MIN_TOKENS = int(os.environ.get("PIN_PEER_MIN_TOKENS", "1"))


def _peer_tier(device):
    """Experimental peer tier, enabled by PIN_PEER=1 (sidecar must be serving)."""
    global _PEER_TIER
    if os.environ.get("PIN_PEER") != "1":
        return None
    if _PEER_TIER is None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from peer_tier import PeerTier
        _PEER_TIER = PeerTier(device)
        _LOG(f"peer tier on: max_tokens={_PEER_TIER.max_tokens}")
    return _PEER_TIER


def _maybe_slice_n(t, n):
    if t is None:
        return None
    if t.ndim == 1 and t.shape[0] == E:
        return t[:n]
    return t


def invoke_f4_from_state(
    st: PinState,
    output,
    x_quant,
    x_scale,
    topk_ids,
    topk_weights,
    activation_type,
    topk,
    tune_max,
    gemm1_alpha=None,
    gemm1_beta=None,
    gemm1_clamp=None,
    intermediate_size=None,
):
    ids = topk_ids
    if isinstance(ids, (tuple, list)):
        ids = ids[0]
    ids = ids.to(dtype=torch.int32)
    T = ids.shape[0]
    hidden = output.shape[-1]
    hot_ids, cold_ids = split_ids(ids, st.row_map)
    peer = _peer_tier(output.device)
    if (peer is not None and _PEER_MIN_TOKENS <= T <= peer.max_tokens
            and x_scale is not None):
        # Experimental: cold rows are resident on, and computed by, the peer GPU.
        perm = torch.where(hot_ids >= 0, hot_ids,
                           torch.where(cold_ids >= 0, cold_ids + st.n_hot, -1)).to(torch.int32)

        def hot_call():
            call_routed(
                ids=hot_ids, w1=st.hot["w1"], s1=st.hot["s1"], b1=st.hot.get("b1"),
                a1=_maybe_slice_n(gemm1_alpha, st.n_hot), beta1=_maybe_slice_n(gemm1_beta, st.n_hot),
                clamp1=_maybe_slice_n(gemm1_clamp, st.n_hot), w2=st.hot["w2"], s2=st.hot["s2"],
                b2=st.hot.get("b2"), n_exp=st.n_hot, x_quant=x_quant, x_scale=x_scale,
                wts=topk_weights, topk=topk, activation_type=activation_type, tune_max=tune_max,
                do_finalize=True, output=output, intermediate_size=intermediate_size,
            )

        peer.run(output, x_quant, x_scale, perm, topk_weights, cold_start=st.n_hot,
                 layer=st.idx, hot_call=hot_call)
        return output

    def _maybe_slice(t, n):
        if t is None:
            return None
        if t.ndim == 1 and t.shape[0] == E:
            return t[:n]
        return t

    common = dict(
        x_quant=x_quant,
        x_scale=x_scale,
        wts=topk_weights,
        topk=topk,
        activation_type=activation_type,
        tune_max=tune_max,
        do_finalize=False,
        output=None,
        intermediate_size=intermediate_size,
    )
    oh = call_routed(
        ids=hot_ids,
        w1=st.hot["w1"],
        s1=st.hot["s1"],
        b1=st.hot.get("b1"),
        a1=_maybe_slice(gemm1_alpha, st.n_hot),
        beta1=_maybe_slice(gemm1_beta, st.n_hot),
        clamp1=_maybe_slice(gemm1_clamp, st.n_hot),
        w2=st.hot["w2"],
        s2=st.hot["s2"],
        b2=st.hot.get("b2"),
        n_exp=st.n_hot,
        **common,
    )
    oc = call_routed(
        ids=cold_ids,
        w1=st.cold["w1"],
        s1=st.cold["s1"],
        b1=st.cold.get("b1"),
        a1=_maybe_slice(gemm1_alpha, st.n_cold),
        beta1=_maybe_slice(gemm1_beta, st.n_cold),
        clamp1=_maybe_slice(gemm1_clamp, st.n_cold),
        w2=st.cold["w2"],
        s2=st.cold["s2"],
        b2=st.cold.get("b2"),
        n_exp=st.n_cold,
        **common,
    )
    g2h, e2ph = _unfinalized_parts(oh, T, topk, hidden)
    g2c, e2pc = _unfinalized_parts(oc, T, topk, hidden)
    finalize_f4(g2h, e2ph, g2c, e2pc, topk_weights, output)
    return output


def _invoke_split(
    self,
    output,
    x_quant,
    x_scale,
    topk_ids,
    topk_weights,
    w1,
    w2,
    activation,
    global_num_experts,
    local_num_experts,
    local_expert_offset,
    topk,
):
    st = getattr(self, "_pin", None)
    if st is None or st.row_map is None:
        st = _PIN_BY_W1.get(w1.data_ptr() if torch.is_tensor(w1) else None)
        if st is not None:
            self._pin = st
    if st is None or st.row_map is None:
        return _orig_invoke(
            self,
            output,
            x_quant,
            x_scale,
            topk_ids,
            topk_weights,
            w1,
            w2,
            activation,
            global_num_experts,
            local_num_experts,
            local_expert_offset,
            topk,
        )
    from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket

    act = self._flashinfer_activation_type(activation)
    tune_max = fi_moe_largest_bucket(self.moe_config)
    return invoke_f4_from_state(
        st,
        output,
        x_quant,
        x_scale,
        topk_ids,
        topk_weights,
        act,
        topk,
        tune_max,
        gemm1_alpha=self.gemm1_alpha,
        gemm1_beta=self.gemm1_beta,
        gemm1_clamp=self.gemm1_clamp_limit,
        intermediate_size=self.intermediate_size_per_partition,
    )


def _unwrap_model(m):
    seen = set()
    while m is not None and id(m) not in seen:
        seen.add(id(m))
        nxt = None
        for attr in ("model", "wrapped", "module", "_model"):
            v = getattr(m, attr, None)
            if v is not None and v is not m and isinstance(v, torch.nn.Module):
                nxt = v
                break
        if nxt is None:
            return m
        m = nxt
    return m


def _iter_routed(model):
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "RoutedExperts":
            w13 = getattr(mod, "w13_weight", None)
            if w13 is None or not torch.is_tensor(w13) or w13.ndim < 2:
                continue
            if int(w13.shape[0]) != E:
                continue
            out.append((name, mod))
    return out


def _iter_modular(model):
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "TrtLlmMxfp4ExpertsModular":
            out.append((name, mod))
    return out


def _find_modular_for_routed(routed):
    for attr in ("quant_method", "experts", "moe_quant", "kernel"):
        obj = getattr(routed, attr, None)
        if obj is None:
            continue
        if type(obj).__name__ == "TrtLlmMxfp4ExpertsModular":
            return obj
        if hasattr(obj, "named_modules"):
            for _subn, sub in obj.named_modules():
                if type(sub).__name__ == "TrtLlmMxfp4ExpertsModular":
                    return sub
        for a in dir(obj):
            if a.startswith("_"):
                continue
            try:
                v = getattr(obj, a)
            except Exception:
                continue
            if type(v).__name__ == "TrtLlmMxfp4ExpertsModular":
                return v
    return None


def _leading_e_from_routed(routed) -> dict:
    names = [
        ("w1", "w13_weight"),
        ("s1", "w13_weight_scale"),
        ("b1", "w13_bias"),
        ("w2", "w2_weight"),
        ("s2", "w2_weight_scale"),
        ("b2", "w2_bias"),
    ]
    found = {}
    for key, attr in names:
        t = getattr(routed, attr, None)
        if t is None:
            continue
        if not torch.is_tensor(t):
            t = getattr(t, "data", t)
        if torch.is_tensor(t) and t.ndim >= 1 and int(t.shape[0]) == E:
            found[key] = t
    return found


def _replace_param(mod, attr: str, new: torch.Tensor) -> None:
    cur = getattr(mod, attr, None)
    if cur is None:
        return
    if isinstance(cur, torch.nn.Parameter):
        cur.data = new
        if hasattr(cur, "_vllm_is_uva_offloaded"):
            try:
                delattr(cur, "_vllm_is_uva_offloaded")
            except Exception:
                cur._vllm_is_uva_offloaded = False
    else:
        setattr(mod, attr, new)


def rehome_one_layer(
    routed,
    experts,
    layer_idx: int,
    hot_ids: list[int],
    cold_ids: list[int],
) -> PinState:
    tensors = _leading_e_from_routed(routed)
    if "w1" not in tensors or "w2" not in tensors:
        raise RuntimeError(
            f"PIN_HOT layer {layer_idx} missing w13/w2 on {type(routed).__name__} "
            f"keys={list(tensors)}"
        )
    is_uva = bool(
        getattr(tensors["w1"], "_vllm_is_uva_offloaded", False)
        or getattr(tensors["w2"], "_vllm_is_uva_offloaded", False)
    )
    # UVA→HBM needs ~5.5 GiB; keep 8 GiB after. HBM→split frees ~1.6.
    need = (6 * (1024**3)) if is_uva else (512 * 1024**2)
    gc.collect()
    torch.cuda.empty_cache()
    hbm = hbm_free_bytes()
    if hbm < need + MIN_HBM_FREE:
        gc.collect()
        torch.cuda.empty_cache()
        hbm = hbm_free_bytes()
    if hbm < need:
        raise RuntimeError(
            f"PIN_HOT abort layer {layer_idx} before: HBM free {hbm/1024**3:.2f} GiB "
            f"need {need/1024**3:.2f} GiB (uva={is_uva})"
        )
    _budget_ok(f"layer {layer_idx} before")
    device = tensors["w1"].device
    if device.type != "cuda":
        device = torch.device("cuda")
    hot_ix = torch.tensor(hot_ids, dtype=torch.int64, device=tensors["w1"].device)
    cold_ix = torch.tensor(cold_ids, dtype=torch.int64, device=tensors["w1"].device)
    is_uva = bool(
        getattr(tensors["w1"], "_vllm_is_uva_offloaded", False)
        or getattr(tensors["w2"], "_vllm_is_uva_offloaded", False)
    )
    st = PinState(layer_idx, getattr(routed, "layer_name", f"layer{layer_idx}"))
    st.is_uva = is_uva
    st.row_map = build_row_map_tensor(hot_ids, cold_ids, device)
    attr_of = {
        "w1": "w13_weight",
        "s1": "w13_weight_scale",
        "b1": "w13_bias",
        "w2": "w2_weight",
        "s2": "w2_weight_scale",
        "b2": "w2_bias",
    }
    dropped = []
    for key, src in list(tensors.items()):
        hot, cold, cpu = partition_leading_e(src, hot_ix, cold_ix)
        st.hot[key] = hot
        st.cold[key] = cold
        if cpu is not None:
            _cpu_keep.append(cpu)
        dropped.append(src)
        _replace_param(routed, attr_of[key], hot)
    for t in dropped:
        del t
    qc = getattr(experts, "quant_config", None) if experts is not None else None
    if qc is not None:
        if "s1" in st.hot:
            qc.w1_scale = st.hot["s1"]
        if "s2" in st.hot:
            qc.w2_scale = st.hot["s2"]
        if "b1" in st.hot:
            qc.w1_bias = st.hot.get("b1")
        if "b2" in st.hot:
            qc.w2_bias = st.hot.get("b2")
    if experts is not None:
        experts._pin = st
    gc.collect()
    torch.cuda.empty_cache()
    _budget_ok(f"layer {layer_idx} after")
    host = mem_available_bytes() / 1024**3
    hbm = hbm_free_bytes() / 1024**3
    _LOG(
        f"rehome layer={layer_idx} name={st.name} uva={is_uva} "
        f"hot_w1={tuple(st.hot['w1'].shape)} cold_w1={tuple(st.cold['w1'].shape)} "
        f"hbm_free={hbm:.2f}GiB host_avail={host:.2f}GiB"
    )
    return st


def rehome_model(model) -> dict:
    global _rowmap, _rehomed
    model = _unwrap_model(model)
    if _rowmap is None:
        _rowmap = load_rowmap()
    routed = _iter_routed(model)
    modular = _iter_modular(model)
    _LOG(f"rehome scan routed={len(routed)} modular={len(modular)}")
    if not routed:
        raise RuntimeError("PIN_HOT rehome: no RoutedExperts with E=384 found")
    pairs = []
    for i, (rname, rmod) in enumerate(routed):
        m = _find_modular_for_routed(rmod)
        if m is None and i < len(modular):
            m = modular[i][1]
        if m is None:
            _LOG(f"rehome {rname}: Modular experts not on module tree (will bind via w1 ptr)")
        pairs.append((i, rname, rmod, m))
    n = min(len(pairs), len(_rowmap))
    uva_idx = []
    hbm_idx = []
    for i, rname, rmod, m in pairs[:n]:
        w13 = rmod.w13_weight
        is_uva = bool(getattr(w13, "_vllm_is_uva_offloaded", False))
        (uva_idx if is_uva else hbm_idx).append(i)
    order = hbm_idx + uva_idx
    _LOG(
        f"rehome order n={n} hbm_layers={len(hbm_idx)} uva_layers={len(uva_idx)} "
        f"strategy=hbm-then-uva (interleave overflows; UVA needs +5.2GiB, HBM frees 1.6)"
    )
    states = []
    for i in order:
        _, rname, rmod, m = pairs[i]
        rm = _rowmap[i]
        st = rehome_one_layer(rmod, m, i, rm["hot"], rm["cold"])
        _PIN_BY_W1[st.hot["w1"].data_ptr()] = st
        states.append(st)
    hbm_bytes = 0
    pin_bytes = 0
    for st in states:
        for t in st.hot.values():
            hbm_bytes += nbytes(t)
        for t in st.cold.values():
            pin_bytes += nbytes(t)
    _LOG(
        f"rehome done layers={len(states)} "
        f"HBM_expert={hbm_bytes/1024**3:.2f}GiB pinned_expert={pin_bytes/1024**3:.2f}GiB "
        f"hbm_free={hbm_free_bytes()/1024**3:.2f}GiB "
        f"host_avail={mem_available_bytes()/1024**3:.2f}GiB "
        f"(expect ~206.6 / ~62.4)"
    )
    try:
        dev = states[0].hot["w1"].device
        T = 1
        dummy_g2 = torch.zeros((TOPK, HIDDEN), dtype=torch.bfloat16, device=dev)
        dummy_e2p = torch.zeros((T, TOPK), dtype=torch.int32, device=dev)
        dummy_w = torch.zeros((T, TOPK), dtype=torch.float32, device=dev)
        dummy_out = torch.zeros((T, HIDDEN), dtype=torch.bfloat16, device=dev)
        finalize_f4(dummy_g2, dummy_e2p, dummy_g2, dummy_e2p, dummy_w, dummy_out)
        torch.cuda.synchronize()
        _LOG("finalize kernel warmed")
    except Exception as e:
        _LOG(f"finalize warmup error {e!r}")
        raise
    _rehomed = True
    return {
        "layers": len(states),
        "hbm_expert_giB": hbm_bytes / 1024**3,
        "pinned_expert_giB": pin_bytes / 1024**3,
        "hbm_free_giB": hbm_free_bytes() / 1024**3,
        "host_avail_giB": mem_available_bytes() / 1024**3,
        "hook_point": "GPUModelRunner.load_model end (after weights+UVA, before KV profile)",
    }


def _do_rehome(self, where: str):
    global _rehomed
    if _rehomed:
        return
    _LOG(f"{where}: starting re-home")
    try:
        rehome_model(self.model)
    except Exception as e:
        _LOG(f"rehome FAILED {e!r}")
        raise


def _patch_load_model(module):
    cls = getattr(module, "GPUModelRunner", None)
    if cls is None:
        return False
    if getattr(getattr(cls, "load_model", None), "_pin_wrapped", False):
        return True
    orig_load = cls.load_model
    orig_profile = getattr(cls, "profile_run", None)

    def load_model(self, *a, **k):
        orig_load(self, *a, **k)
        if MODE == "split":
            _do_rehome(self, "load_model")
        return None

    load_model._pin_wrapped = True
    cls.load_model = load_model
    if orig_profile is not None and not getattr(orig_profile, "_pin_wrapped", False):

        def profile_run(self, *a, **k):
            if MODE == "split":
                _do_rehome(self, "profile_run")
            return orig_profile(self, *a, **k)

        profile_run._pin_wrapped = True
        cls.profile_run = profile_run
    _LOG(f"wrapped GPUModelRunner.load_model+profile_run in {getattr(module, '__name__', '?')}")
    return True


def _scan_and_patch_runner() -> int:
    n = 0
    for name, mod in list(sys.modules.items()):
        if not isinstance(name, str) or name.startswith("torch"):
            continue
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict) or "GPUModelRunner" not in d:
            continue
        try:
            if _patch_load_model(mod):
                n += 1
        except Exception as e:
            _LOG(f"scan patch fail {name} {e!r}")
    return n


def _runner_scanner():
    for i in range(180):
        try:
            n = _scan_and_patch_runner()
            if n:
                _LOG(f"runner scanner hit n={n} i={i}")
                return
        except Exception as e:
            _LOG(f"runner scanner {e!r}")
        time.sleep(0.5)
    _LOG("runner scanner gave up after 90s — GPUModelRunner never appeared")


def install():
    global _orig_invoke, _installed
    if _installed:
        return
    import importlib.abc
    import importlib.util

    target = "vllm.model_executor.layers.fused_moe.experts.trtllm_mxfp4_moe"
    runner = "vllm.v1.worker.gpu_model_runner"
    if MODE == "count":
        wrapper = _invoke_count
    elif MODE == "split":
        wrapper = _invoke_split
    else:
        wrapper = _invoke_off

    def _patch_experts(module):
        global _orig_invoke
        cls = module.TrtLlmMxfp4ExpertsModular
        _orig_invoke = cls._invoke_kernel
        cls._invoke_kernel = wrapper
        _LOG(
            f"installed PIN_MODE={MODE} dump_dir={DUMP_DIR} "
            f"DECODE_T={DECODE_T} MAX_N={MAX_N} target={target} "
            f"rowmap={ROWMAP_PATH if MODE == 'split' else '-'}"
        )

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self, name, patch):
            self._name = name
            self._patch = patch

        def find_spec(self, name, path, target_mod=None):
            if name != self._name:
                return None
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            spec = importlib.util.find_spec(name)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module, _orig=orig_exec, _p=self._patch):
                _orig(module)
                _p(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder(target, _patch_experts))
    mod = sys.modules.get(target)
    if mod is not None:
        _patch_experts(mod)

    if MODE == "split":
        sys.meta_path.insert(0, _Finder(runner, _patch_load_model))
        rmod = sys.modules.get(runner)
        if rmod is not None:
            _patch_load_model(rmod)
        _scan_and_patch_runner()
        tscan = threading.Thread(target=_runner_scanner, daemon=True, name="pin-hot-runner-scan")
        tscan.start()
        _LOG("runner scanner thread started")

    if MODE == "count":
        os.makedirs(DUMP_DIR, exist_ok=True)
        t = threading.Thread(target=_bg_thread, daemon=True, name="pin-hot-dump")
        t.start()
        atexit.register(lambda: dump("atexit"))

        def _term(signum, frame):
            dump("signal")
            raise SystemExit(0)

        try:
            import signal

            signal.signal(signal.SIGTERM, _term)
        except Exception:
            pass
    _installed = True
    _LOG(f"finder armed PIN_MODE={MODE}")
