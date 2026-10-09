"""Online switching between the official and abliterated DeepSeek-V4.1-Flash weights (M3 lane).

The abliterated checkpoint differs from upstream in 132 tensors: `attn.wo_b` and `ffn.shared_experts.w2`
(weight + scale) for layers 4-36, 1.65 GiB in all. Everything else, every routed expert included, is
byte-identical, so a switch is an in-place update of two dense tensors per layer. Decode runs inside CUDA
graphs that bake data pointers into kernel arguments, so in-place writes keep every graph, JIT artefact and
split-K tactic valid. See PLAN-abliterated-switch.md in this directory.

Safety model:

  * Nothing happens unless MEGA_ABLIT resolves to enabled (default `auto`: on when the delta exists).
  * Before the first switch a self-test runs: for all 132 tensors it reads the official bytes, rebuilds the
    derived resident form, and compares bytes against what is actually resident. The resident conventions
    (which parameter holds a scale, its expansion factor, swizzle, dtype) are *discovered* this way rather
    than assumed, because this image has more than one plausible convention. A mismatch disables switching
    and leaves the server serving.
  * A switch validates every planned update before writing any of them, so a half-applied model cannot happen.
  * Polling and writing happen on the engine thread between steps, never during graph capture.
  * The poll sites are the model runners (V2 then V1) and the worker, because this fork imports exactly one of
    the two runners; whichever module the image loads logs "poll installed on <module>.<class>.execute_model",
    so a future move shows up as a missing line rather than as silence.

Files (all under MEGA_ABLIT_DIR, default /ablit/data; the directory is mounted read-write into the container):

  abliterated-delta.json / .bin   input, from tools/extract_abliterated_delta.py
  mode.json                       input, written by tools/ablit.sh: {"mode": ..., "seq": N}
  state.json                      output: applied mode/seq, switch count and cost, errors
  selftest.json                   output: the verified (resident tensor, rebuild recipe) plan

Env: MEGA_ABLIT (auto|0|1), MEGA_ABLIT_DIR, MEGA_ABLIT_MODEL, MEGA_ABLIT_POLL, MEGA_ABLIT_MIN_CALLS,
MEGA_ABLIT_READY_TIMEOUT, MEGA_ABLIT_SELFTEST (full|light).
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import re
import struct
import sys
import time

import torch

DIR = os.environ.get("MEGA_ABLIT_DIR", "/ablit/data")
MEGA_ABLIT = os.environ.get("MEGA_ABLIT", "auto").strip().lower()
MODEL_ROOT = os.environ.get("MEGA_ABLIT_MODEL", "/model")
POLL_S = float(os.environ.get("MEGA_ABLIT_POLL", "1.0"))
MIN_CALLS = int(os.environ.get("MEGA_ABLIT_MIN_CALLS", "2"))
READY_TIMEOUT_S = float(os.environ.get("MEGA_ABLIT_READY_TIMEOUT", "180"))
SELFTEST = os.environ.get("MEGA_ABLIT_SELFTEST", "full").strip().lower()

DELTA_JSON = os.path.join(DIR, "abliterated-delta.json")
DELTA_BIN = os.path.join(DIR, "abliterated-delta.bin")
MODE_FILE = os.path.join(DIR, "mode.json")
STATE_FILE = os.path.join(DIR, "state.json")
SELFTEST_FILE = os.path.join(DIR, "selftest.json")

MODES = ("official", "abliterated")
BLOCK = 32  # the checkpoint's 32x32 MXFP8 block for these tensors

MODELOPT = "vllm.model_executor.layers.quantization.modelopt"
# Per-step poll sites, tried in order. This fork defaults to Model Runner V2
# (vllm.v1.worker.gpu.model_runner); gpu_worker imports *either* V2 or V1, never both, so a single target
# silently does nothing on the wrong branch — which is exactly what happened on the first boot with the
# injection in place (0 ABLIT poll lines, no selftest). The worker itself is the third site because it is one
# level above the runner and would survive both runners moving again.
_LINEAR_RE = re.compile(r"layers\.(\d+)\.(attn\.wo_b|ffn\.shared_experts\.down_proj)$")
_MOE_RE = re.compile(r"layers\.(\d+)\.ffn\.experts$")
_SHARED_RE = re.compile(r"layers\.(\d+)\.ffn\.shared_experts$")
_SCALE_ATTRS = ("weight_scale", "weight_scale_inv")
_LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _log(msg: str) -> None:
    sys.stderr.write(f"ABLIT {msg}\n")
    sys.stderr.flush()


STATE: dict = {
    "enabled": None,
    "reason": "",
    "selftest": None,
    "applied_mode": "official",
    "applied_seq": 0,
    "switches": 0,
    "last_switch_s": None,
    "last_switch_at": None,
    "errors": [],
}

_delta: dict | None = None
_model = None
_linears: dict[tuple[int, str], object] = {}
_moe: dict[int, object] = {}
_shared: dict[int, object] = {}
_targets: list[dict] = []
_calls = 0
_last_poll = 0.0
_mode_stat: tuple | None = None
_installed = False
_disabled: str | None = None
_ready_since: float | None = None
_said_incomplete = False


# --------------------------------------------------------------------------- small utilities


def _write_json(path: str, obj) -> None:
    try:
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(path + ".tmp", path)
    except OSError as e:
        _log(f"could not write {path}: {e!r}")


def _flush_state() -> None:
    _write_json(STATE_FILE, STATE)


def _error(msg: str) -> None:
    _log(f"ERROR {msg}")
    STATE["errors"].append({"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "msg": msg})
    del STATE["errors"][:-20]
    _flush_state()


def _bytes_of(t: torch.Tensor) -> torch.Tensor:
    """Raw bytes in logical order, independent of dtype and strides."""
    return t.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)


def _same_bytes(a: torch.Tensor, b: torch.Tensor) -> bool:
    ba, bb = _bytes_of(a), _bytes_of(b)
    return ba.numel() == bb.numel() and bool(torch.equal(ba, bb))


def _scale_param(module):
    for name in _SCALE_ATTRS:
        t = getattr(module, name, None)
        if isinstance(t, torch.Tensor) and t.numel():
            return name, t
    return None, None


def _swizzle():
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale

    return swizzle_mxfp8_scale


def _deep_gemm():
    """Indirection so the CPU test can exercise the fused-shared path without the CUDA extension."""
    from vllm.utils.deep_gemm import _import_deep_gemm

    return _import_deep_gemm()


def _ue8m0_to_float(sf: torch.Tensor) -> torch.Tensor:
    """e8m0 bytes -> float32 exponents, the idiom the model uses (`(x << 23).view(float32)`)."""
    return (sf.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


_UTCCP_CACHE: dict = {}


def _utccp_transpose(sf: torch.Tensor) -> torch.Tensor:
    """The 4x32 tile transpose DeepGEMM applies to UTCCP scales; vendored copy if importable.

    Faithful to vllm/third_party/deep_gemm/mega/__init__.py::_transpose_sf_for_utccp. The live resident for
    the fused shared expert's scale is packed int32 (5120, 18), i.e. 72 e8m0 bytes per row in 18 words, and
    whether the transpose is part of that packing is exactly what the self-test decides.
    """
    if "fn" not in _UTCCP_CACHE:
        try:
            from vllm.third_party.deep_gemm.mega import _transpose_sf_for_utccp

            _UTCCP_CACHE["fn"] = _transpose_sf_for_utccp
        except Exception:
            def _transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
                squeeze = sf.dim() == 2
                if squeeze:
                    sf = sf.unsqueeze(0)
                num_groups, mn, packed_sf_k = sf.shape
                if mn % 128:
                    raise ValueError(f"MN {mn} is not a multiple of 128")
                res = (sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
                       .transpose(2, 3).reshape(num_groups, mn, packed_sf_k))
                res = torch.empty_like(sf).copy_(res)
                return res.squeeze(0) if squeeze else res

            _UTCCP_CACHE["fn"] = _transpose_sf_for_utccp
    return _UTCCP_CACHE["fn"](sf)


def _ckpt_scale_2d(u8: torch.Tensor, scale_shape) -> torch.Tensor:
    """Checkpoint scale bytes as the 2-D block grid they are stored in (one e8m0 byte per 32x32 block)."""
    if scale_shape is None or len(scale_shape) != 2:
        raise ValueError(f"expected a 2-D scale shape, got {scale_shape!r}")
    rows, cols = int(scale_shape[0]), int(scale_shape[1])
    if u8.numel() != rows * cols:
        raise ValueError(f"scale has {u8.numel()} bytes, expected {rows}x{cols}")
    return u8.view(torch.uint8).reshape(rows, cols)


# --------------------------------------------------------------------------- registration


def register_moe(layer, moe, shared) -> None:
    """Secondary source (mega_peer_hook._finalize_weights). The model walk is the primary one."""
    if layer is None or moe is None:
        return
    _moe[int(layer)] = moe
    if shared is not None:
        _shared[int(layer)] = shared


def _register_from_model(model) -> None:
    """Walk the loaded model tree: names are authoritative and no quant-method knowledge is needed."""
    _linears.clear()
    _moe.clear()
    _shared.clear()
    for name, mod in model.named_modules():
        m = _LINEAR_RE.search(name)
        if m is not None:
            _linears[(int(m.group(1)), "wo_b" if m.group(2).startswith("attn.") else "shared_down")] = mod
            continue
        m = _MOE_RE.search(name)
        if m is not None:
            _moe[int(m.group(1))] = mod
            continue
        m = _SHARED_RE.search(name)
        if m is not None:
            _shared[int(m.group(1))] = mod


def _unwrap(model):
    for _ in range(8):
        if isinstance(model, torch.nn.Module):
            return model
        model = getattr(model, "model", None)
        if model is None:
            return None
    return None


def _patch_modelopt(module) -> None:
    scheme = getattr(module, "KMxfp8Static", None)
    if scheme is None or getattr(scheme, "_ablit_reg", False):
        return
    orig = scheme.create_weights

    def create_weights(self, layer, role, ctx, shapes, wl):
        m = _LINEAR_RE.search(getattr(layer, "prefix", "") or "")
        if m is not None:
            _linears[(int(m.group(1)), "wo_b" if m.group(2).startswith("attn.") else "shared_down")] = layer
        return orig(self, layer, role, ctx, shapes, wl)

    scheme.create_weights = create_weights
    scheme._ablit_reg = True


def _patch_runner(module) -> None:
    """A model runner's per-step entry point: once per step, before any layer of that step runs."""
    cls = getattr(module, "GPUModelRunner", None)
    if cls is None or getattr(cls, "_ablit_poll", False):
        return
    orig = cls.execute_model

    def execute_model(self, *args, **kwargs):
        maybe_poll(getattr(self, "model", None))
        return orig(self, *args, **kwargs)

    cls.execute_model = execute_model
    cls._ablit_poll = True
    _log(f"poll installed on {cls.__module__}.{cls.__name__}.execute_model")


def _patch_worker(module) -> None:
    """Fallback site: the worker's per-step entry point, one level above the model runner."""
    cls = getattr(module, "Worker", None)
    if cls is None or getattr(cls, "_ablit_poll", False):
        return
    orig = cls.execute_model

    def execute_model(self, *args, **kwargs):
        runner = getattr(self, "model_runner", None)
        maybe_poll(getattr(runner, "model", None))
        return orig(self, *args, **kwargs)

    cls.execute_model = execute_model
    cls._ablit_poll = True
    _log(f"poll installed on {cls.__module__}.{cls.__name__}.execute_model")


# Tried in this order; whichever module the image imports gets patched, and the log says which one it was.
SITES = (
    ("vllm.v1.worker.gpu.model_runner", _patch_runner),   # Model Runner V2, the default in this image
    ("vllm.v1.worker.gpu_model_runner", _patch_runner),   # Model Runner V1
    ("vllm.v1.worker.gpu_worker", _patch_worker),         # the worker, if both runners move again
)


def _patch_request(module) -> None:
    """Mode-scoped cache salt.

    vLLM hashes blocks from the tokens plus `cache_salt` and knows nothing about weights, so KV computed under
    one mode is happily reused under the other: measured on 2026-10-08, re-sending a prompt the official run had
    already sent moved `prefix_cache_queries_total` by 239,287 and `prefix_cache_hits_total` by 239,268 (a ~100%
    hit) after a switch. Requests that bring their own salt keep it; everything else gets one derived from the
    applied mode, so a switch starts a fresh hash namespace by itself and no dev-mode cache flush is required.
    """
    cls = getattr(module, "Request", None)
    if cls is None or getattr(cls, "_ablit_salt", False):
        return
    orig = cls.from_engine_core_request.__func__      # classmethod: patch the underlying function

    def from_engine_core_request(cls_, request, block_hasher=None, *args, **kwargs):
        # The block hashes are computed *inside* the original call — `block_hasher` is its argument — so the salt
        # must be on the input EngineCoreRequest before it runs. Setting it only on the returned Request was too
        # late, which the A/B of 2026-10-09 showed plainly: after a switch, the first official-mode request still
        # reused 16,000 blocks of abliterated-era KV (dhits 16,000 of 16,043 queries).
        salt = None
        try:
            if not getattr(request, "cache_salt", None):
                salt = f"ablit-{STATE.get('applied_mode', 'official')}"
                request.cache_salt = salt
        except Exception as e:
            _error(f"could not set the cache salt on the engine request: {e!r}")
        r = orig(cls_, request, block_hasher, *args, **kwargs)
        try:
            if not getattr(r, "cache_salt", None):
                r.cache_salt = salt or f"ablit-{STATE.get('applied_mode', 'official')}"
        except Exception:
            pass
        return r

    cls.from_engine_core_request = classmethod(from_engine_core_request)
    cls._ablit_salt = True
    _log(f"mode-scoped cache salt installed on {cls.__module__}.{cls.__name__}")


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, target, patch, required):
        self.target, self.patch, self.required = target, patch, required

    def find_spec(self, name, path, target=None):
        if name != self.target:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            if self.required:
                _log(f"WARNING {self.target} is not importable; that site stays off")
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _patch=self.patch):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


def install() -> None:
    global _installed, _delta
    if _installed:
        return
    _installed = True
    if MEGA_ABLIT in ("0", "off", "false", "no"):
        STATE.update(enabled=False, reason="MEGA_ABLIT=0")
        _log("switching off (MEGA_ABLIT=0)")
        return
    if not (os.path.exists(DELTA_JSON) and os.path.exists(DELTA_BIN)):
        STATE.update(enabled=False, reason=f"no delta at {DELTA_JSON}")
        _log(f"no delta in {DIR}; switching stays off (normal boot path)")
        return
    try:
        with open(DELTA_JSON) as f:
            _delta = json.load(f)
        if _delta.get("schema") != "dsv41-abliterated-delta/1":
            raise ValueError(f"unexpected schema {_delta.get('schema')!r}")
    except Exception as e:
        STATE.update(enabled=False, reason=f"bad delta index: {e!r}")
        _error(f"could not read {DELTA_JSON}: {e!r}")
        return
    STATE.update(enabled=True, reason="booting; no model yet")
    sys.meta_path.insert(0, _Finder(MODELOPT, _patch_modelopt, False))
    sys.meta_path.insert(0, _Finder("vllm.v1.request", _patch_request, False))
    for target, patch in SITES:
        sys.meta_path.insert(0, _Finder(target, patch, False))
    _log(f"armed poll sites: {', '.join(t for t, _ in SITES)}")
    _log(f"on: {_delta['totals']['tensors']} tensors, {_delta['totals']['bytes'] / 2**20:.1f} MiB, "
         f"layers {_delta['layers'][0]}-{_delta['layers'][1]}, dir {DIR}, selftest {SELFTEST}"
         + (" [SYNTHETIC DELTA - not the real abliteration]" if _delta.get("synthetic") else ""))


# --------------------------------------------------------------------------- sources


class _Ckpt:
    """Local checkpoint reader: header once per shard, then byte-range reads."""

    def __init__(self, root: str):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._h: dict[str, tuple[int, dict]] = {}

    def read_u8(self, name: str) -> torch.Tensor:
        shard = self.index[name]
        if shard not in self._h:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._h[shard] = (8 + n, json.loads(f.read(n)))
        base, hdr = self._h[shard]
        start, end = hdr[name]["data_offsets"]
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(base + start)
            buf = bytearray(f.read(end - start))
        if len(buf) != end - start:
            raise IOError(f"short read for {name}")
        return torch.frombuffer(buf, dtype=torch.uint8)


class _DeltaSource:
    """The abliterated bytes, straight out of the delta blob."""

    def __init__(self, index: dict, bin_path: str):
        with open(bin_path, "rb") as f:
            self.blob = bytearray(f.read())
        self.entries = {t["name"]: t for t in index["tensors"]}
        for t in index["tensors"]:
            if t["bin_offset"] + t["nbytes"] > len(self.blob):
                raise ValueError(f"delta blob is short for {t['name']}")

    def read_u8(self, name: str) -> torch.Tensor:
        t = self.entries[name]
        return torch.frombuffer(self.blob, dtype=torch.uint8, count=t["nbytes"], offset=t["bin_offset"])


# --------------------------------------------------------------------------- layout discovery


def _wo_b_scale_candidates(u8: torch.Tensor, scale_shape, resident):
    """Rebuild a dense MXFP8 scale from checkpoint bytes under every plausible convention.

    The checkpoint holds one e8m0 byte per 32x32 block. A resident copy may expand the block rows by 32, swizzle
    to F8_128x4, and/or widen each byte to a word; the self-test decides which one this image uses.
    """
    ckpt = _ckpt_scale_2d(u8, scale_shape)
    N, K = ckpt.shape[0] * BLOCK, ckpt.shape[1] * BLOCK
    out = []
    for e in dict.fromkeys((N // ckpt.shape[0], 1)):
        rows = ckpt if e == 1 else ckpt.repeat_interleave(e, dim=0)
        out.append((f"raw_e8m0/expand{e}", rows.view(torch.float8_e8m0fnu)))
        out.append((f"raw_u8/expand{e}", rows))
        out.append((f"raw_i32/expand{e}", rows.to(torch.int32)))
        if rows.shape[0] == N and rows.shape[1] == K // BLOCK:
            sw = _swizzle()(rows, M=N, K=K)
            out.append((f"swizzle_u8/expand{e}", sw))
            if sw.numel() % 4 == 0:
                out.append((f"swizzle_view4/expand{e}", sw.view(torch.int32)))
    return out


FUSED_LABELS = ("helper_expand", "helper_ckpt", "inline", "inline_utccp", "packed4", "packed4_utccp")


def _build_fused_scale(label: str, moe, down, grid_u8: torch.Tensor, hidden: int, inter: int,
                       device) -> torch.Tensor:
    """One way the fused shared expert's scale might be laid out, by label.

    Used both to discover which one the resident is and to replay that same one on every switch, so the layout
    the self-test verified is exactly the layout the switch writes.

    The live resident (observed 2026-10-08) is int32 (5120, 18): the row-expanded 1x32 grid packed 4 e8m0 bytes
    per word, with or without DeepGEMM's 4x32 UTCCP transpose.
    """
    expand = max(1, hidden // grid_u8.shape[0])
    rows = grid_u8 if expand == 1 else grid_u8.repeat_interleave(expand, dim=0)
    if label.startswith("helper"):
        helper = None
        for attr in ("_prepare_shared_expert_scale", "prepare_shared_expert_scale"):
            if callable(getattr(moe, attr, None)):
                helper = getattr(moe, attr)
                break
        if helper is None:
            raise RuntimeError("the MoE module has no shared-expert scale helper")
        arg = rows if label == "helper_expand" else grid_u8
        out = helper(_deep_gemm(), down, arg.to(device).view(torch.float8_e8m0fnu), hidden, inter)
        if out is None:
            raise RuntimeError(f"the helper refused the {label} form")
        return out
    if label.startswith("inline"):
        sf = _ue8m0_to_float(rows.to(device))
        packed = _deep_gemm().transform_sf_into_required_layout(
            sf.reshape(hidden, inter // BLOCK).unsqueeze(0), hidden, inter, (1, 32), 1).squeeze(0)
        return _utccp_transpose(packed) if label.endswith("utccp") else packed
    words = rows.contiguous().view(torch.int32)
    return _utccp_transpose(words) if label.endswith("utccp") else words


_LOGGED_CANDIDATE_FAILURES: set = set()


def _log_candidate_failure(label: str, err: Exception) -> None:
    """One line per label per boot: 33 layers would otherwise repeat the same refusal 33 times."""
    if label in _LOGGED_CANDIDATE_FAILURES:
        return
    _LOGGED_CANDIDATE_FAILURES.add(label)
    _log(f"  fused scale {label}: {err!r} (logged once per boot)")


def _shared_scale_candidates(moe, down, u8: torch.Tensor, hidden: int, inter: int, resident):
    """Every plausible rebuild of the fused shared scale; the self-test keeps the one that matches."""
    hidden, inter = int(hidden), int(inter)
    grid = _ckpt_scale_2d(u8, (hidden // BLOCK, inter // BLOCK))          # (160, 72) e8m0 bytes
    out = []
    for label in FUSED_LABELS:
        try:
            out.append((label, _build_fused_scale(label, moe, down, grid, hidden, inter, resident.device)))
        except Exception as e:
            _log_candidate_failure(label, e)
    return out


def _linear_scale_candidates(u8: torch.Tensor, scale_shape, resident):
    """The shared expert's own linear scale: the loader's row expansion.

    The live resident is flat uint8 (368640,) holding the 5120x72 grid, so this works from element counts
    rather than from ndim.
    """
    ckpt = _ckpt_scale_2d(u8, scale_shape)
    out = []
    if ckpt.numel() and resident.numel() % ckpt.numel() == 0:
        e = max(1, resident.numel() // ckpt.numel())
        rows = ckpt if e == 1 else ckpt.repeat_interleave(e, dim=0)
        out.append((f"expand{e}/e8m0", rows.view(torch.float8_e8m0fnu)))
        out.append((f"expand{e}/u8", rows))
        out.append((f"expand{e}/i32", rows.to(torch.int32)))
    return out


def _match(cands, resident):
    """Candidates that reproduce the resident tensor byte for byte.

    Every hit is verified against the resident, so more than one hit is fine (they are the same bytes); only
    zero hits is a problem.
    """
    return [(label, cand) for label, cand in cands
            if cand.dtype == resident.dtype and cand.numel() == resident.numel()
            and _same_bytes(cand, resident)]


def _dedupe(dsts):
    seen, out = {}, []
    for d in dsts:
        t = d["tensor"]
        key = (t.data_ptr(), tuple(t.shape), t.dtype)
        if key not in seen:
            seen[key] = True
            out.append(d)
    return out


def _desc(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.Tensor):
        return f"Tensor{tuple(x.shape)} {str(x.dtype).replace('torch.', '')}"
    if isinstance(x, (tuple, list)):
        return f"{type(x).__name__}(len={len(x)}: " + ", ".join(_desc(e) for e in x) + ")"
    return type(x).__name__


def _report_plan_inputs() -> None:
    """One log line describing what the plan actually sees. The single most useful diagnostic when it fails."""
    try:
        layers = sorted({int(_LAYER_RE.search(n).group(1)) for n in
                         (t["name"] for t in _delta["tensors"]) if _LAYER_RE.search(n)})
        layer = next((l for l in layers if (l, "wo_b") in _linears), layers[0] if layers else None)
        lin = _linears.get((layer, "wo_b"))
        m, sh = _moe.get(layer), _shared.get(layer)
        down = getattr(sh, "down_proj", None) if sh is not None else None
        pname, pscale = _scale_param(lin) if lin is not None else (None, None)
        dname, dscale = _scale_param(down) if down is not None else (None, None)
        tsw = getattr(m, "_transformed_shared_l2_weights", None) if m is not None else None
        _log(f"plan inputs (layer {layer}): wo_b.weight={_desc(getattr(lin, 'weight', None))} "
             f"wo_b.{pname}={_desc(pscale)} | down_proj.weight={_desc(getattr(down, 'weight', None))} "
             f"down_proj.{dname}={_desc(dscale)} | _transformed_shared_l2_weights={_desc(tsw)}")
    except Exception as e:
        _log(f"plan inputs report failed: {e!r}")


def _build_targets(ckpt: _Ckpt) -> list[dict]:
    targets = []
    shape_of = {t["name"]: tuple(t["shape"]) for t in _delta["tensors"]}
    _report_plan_inputs()
    for t in _delta["tensors"]:
        name, shape = t["name"], tuple(t["shape"])
        layer = int(_LAYER_RE.search(name).group(1))
        entry = {"name": name, "layer": layer, "shape": shape, "dtype": t["dtype"], "nbytes": t["nbytes"],
                 "dsts": [], "kind": None, "rejected": []}
        try:
            _target_for(ckpt, shape_of, entry)
        except Exception as e:  # one surprising tensor must not abort the whole plan
            entry["dsts"] = []
            entry["rejected"].append(f"exception: {e!r}")
            _log(f"plan: {name} raised {e!r}")
        targets.append(entry)
    return targets


def _target_for(ckpt: _Ckpt, shape_of: dict, entry: dict) -> None:
    """Fill in one entry's destinations and rebuild recipe, or record why it cannot be switched."""
    name, shape, layer = entry["name"], entry["shape"], entry["layer"]
    is_wo_b = ".attn.wo_b." in name
    u8 = ckpt.read_u8(name)

    if not name.endswith(".scale"):
        entry["kind"] = "weight"
        src_fp8 = u8.view(torch.float8_e4m3fn)
        dsts = []
        if is_wo_b:
            lin = _linears.get((layer, "wo_b"))
            if lin is not None:
                dsts.append(lin.weight)
        else:
            m = _moe.get(layer)
            tsw = getattr(m, "_transformed_shared_l2_weights", None) if m is not None else None
            if isinstance(tsw, (tuple, list)) and len(tsw) >= 1 and isinstance(tsw[0], torch.Tensor):
                dsts.append(tsw[0])
            sh = _shared.get(layer)
            if sh is not None and hasattr(sh, "down_proj"):
                dsts.append(sh.down_proj.weight)
        for dst in dsts:
            if dst is None:
                continue
            if tuple(dst.shape) == shape:
                orient = "NK"
            elif tuple(dst.shape) == shape[::-1]:
                orient = "KN"
            else:
                entry["rejected"].append(f"resident shape {tuple(dst.shape)} != {shape}")
                continue
            want = src_fp8.reshape(shape) if orient == "NK" else src_fp8.reshape(shape).t()
            if SELFTEST != "light" and not _same_bytes(dst, want):
                entry["rejected"].append(f"resident bytes differ from the official checkpoint ({orient})")
                continue
            entry["dsts"].append({"tensor": dst, "orient": orient, "recipe": None})
        entry["dsts"] = _dedupe(entry["dsts"])
        return

    if is_wo_b:
        entry["kind"] = "wo_b_scale"
        lin = _linears.get((layer, "wo_b"))
        pname, resident = _scale_param(lin) if lin is not None else (None, None)
        if resident is None:
            entry["rejected"].append("no wo_b scale parameter")
            return
        pick = _match(_wo_b_scale_candidates(u8, shape, resident), resident)
        if pick:
            entry["dsts"].append({"tensor": resident, "orient": None,
                                  "recipe": ("swizzle", shape, pick[0][0])})
        else:
            entry["rejected"].append(
                f"no layout matched {pname} {tuple(resident.shape)} {resident.dtype}")
        return

    entry["kind"] = "shared_scale"
    m = _moe.get(layer)
    sh = _shared.get(layer)
    tsw = getattr(m, "_transformed_shared_l2_weights", None) if m is not None else None
    down = getattr(sh, "down_proj", None) if sh is not None else None
    if down is None:
        entry["rejected"].append("no shared expert down_proj")
    # Geometry from the checkpoint, never from the fused copies:
    # layers.N.ffn.shared_experts.w2.weight is (hidden, intermediate), the scale is its 32x32 grid.
    wshape = shape_of.get(name[: -len(".scale")] + ".weight")
    if wshape is not None and len(wshape) == 2:
        hidden, inter = int(wshape[0]), int(wshape[1])
    else:
        hidden, inter = int(shape[0]) * BLOCK, int(shape[1]) * BLOCK
    if isinstance(tsw, (tuple, list)) and len(tsw) >= 2 and isinstance(tsw[1], torch.Tensor):
        resident = tsw[1]
        pick = _match(_shared_scale_candidates(m, down, u8, hidden, inter, resident), resident)
        if pick:
            entry["dsts"].append({"tensor": resident, "orient": None,
                                  "recipe": ("deepep", (hidden, inter), pick[0][0])})
        else:
            entry["rejected"].append(
                f"no rebuild matched the fused scale {tuple(resident.shape)} {resident.dtype}")
    elif tsw is not None:
        entry["rejected"].append(f"_transformed_shared_l2_weights is {_desc(tsw)}")
    if down is not None:
        pname, lin_scale = _scale_param(down)
        if lin_scale is not None:
            pick = _match(_linear_scale_candidates(u8, shape, lin_scale), lin_scale)
            if pick:
                entry["dsts"].append({"tensor": lin_scale, "orient": None,
                                      "recipe": ("linear", shape, pick[0][0])})
            else:
                entry["rejected"].append(
                    f"no expansion matched {pname} {tuple(lin_scale.shape)} {lin_scale.dtype}")
    entry["dsts"] = _dedupe(entry["dsts"])


def _ensure_ready() -> bool:
    """Build and verify the plan once the model is loaded."""
    global _targets, _disabled, _ready_since, _said_incomplete
    if _disabled is not None or _targets:
        return bool(_targets)
    need_wo_b = {int(_LAYER_RE.search(t["name"]).group(1)) for t in _delta["tensors"] if ".attn.wo_b." in t["name"]}
    need_sh = {int(_LAYER_RE.search(t["name"]).group(1)) for t in _delta["tensors"] if "shared_experts" in t["name"]}
    have_wo_b = {l for (l, k) in _linears if k == "wo_b"}
    have_sh = set(_moe) & set(_shared)
    if not need_wo_b <= have_wo_b or not need_sh <= have_sh:
        if _ready_since is None:
            _ready_since = time.time()
        elif not _said_incomplete and time.time() - _ready_since > 30:
            _said_incomplete = True
            _log(f"waiting for modules: wo_b {len(need_wo_b & have_wo_b)}/{len(need_wo_b)}, "
                 f"shared {len(need_sh & have_sh)}/{len(need_sh)}; tree walk found "
                 f"{len(_linears)} linears and {len(_moe)} MoE modules")
        if time.time() - _ready_since > READY_TIMEOUT_S:
            # Only the process that owns the model writes state.json: the API-server process has no model and
            # must not clobber what the engine core reports.
            if _model is not None:
                _disabled = f"modules never appeared within {READY_TIMEOUT_S:.0f}s"
                STATE["reason"] = _disabled
                _error(_disabled)
            elif not _said_incomplete:
                _log("this process has no model; switching stays idle here")
            return False
    t0 = time.time()
    try:
        targets = _build_targets(_Ckpt(MODEL_ROOT))
    except Exception as e:
        import traceback

        _disabled = f"plan build failed: {e!r}"
        STATE["reason"] = _disabled
        _error(f"{_disabled}; at " + " | ".join(traceback.format_exc().strip().splitlines()[-3:]))
        return False
    ok = [t for t in targets if t["dsts"]]
    bad = [t for t in targets if not t["dsts"]]
    report = {
        "checked": len(targets),
        "verified": len(ok),
        "failed": len(bad),
        "failed_tensors": [{"name": t["name"], "why": t["rejected"]} for t in bad],
        "recipes": sorted({str(d["recipe"][0]) + ":" + str(d["recipe"][-1]) for t in ok for d in t["dsts"]
                           if d["recipe"]}),
        "layers": sorted({t["layer"] for t in ok}),
        "official_root": MODEL_ROOT,
        "mode": SELFTEST,
        "delta": {"repo": _delta.get("repo"), "synthetic": bool(_delta.get("synthetic")),
                  "tensors": _delta["totals"]["tensors"]},
        "seconds": round(time.time() - t0, 2),
    }
    _write_json(SELFTEST_FILE, report)
    STATE["selftest"] = report
    _targets = targets
    if bad:
        _disabled = f"self-test failed for {len(bad)} of {len(targets)} tensors"
        STATE["reason"] = _disabled
        _error(f"{_disabled}; first: {bad[0]['name']} {bad[0]['rejected']}")
        _flush_state()
        return False
    STATE["reason"] = "verified, waiting for a mode change"
    _flush_state()
    _log(f"selftest: {len(ok)}/{len(targets)} tensors verified in {report['seconds']}s; "
         f"recipes {report['recipes']}" + (" [SYNTHETIC DELTA]" if _delta.get("synthetic") else ""))
    return True


# --------------------------------------------------------------------------- apply


def _rebuild_scale(entry, recipe, u8: torch.Tensor, resident: torch.Tensor):
    kind, shape, label = recipe
    if kind == "swizzle":
        rows = _ckpt_scale_2d(u8, shape)
        N, K = rows.shape[0] * BLOCK, rows.shape[1] * BLOCK
        e = int(label.rsplit("expand", 1)[1])
        rows = rows if e == 1 else rows.repeat_interleave(e, dim=0)
        if label.startswith("swizzle"):
            cand = _swizzle()(rows, M=N, K=K)
            return cand.view(torch.int32) if "view4" in label else cand
        if label.startswith("raw_u8"):
            return rows
        if label.startswith("raw_i32"):
            return rows.to(torch.int32)
        return rows.view(torch.float8_e8m0fnu)
    if kind == "linear":
        rows = _ckpt_scale_2d(u8, shape)
        e = int(label.split("expand", 1)[1].split("/")[0])
        rows = rows if e == 1 else rows.repeat_interleave(e, dim=0)
        if label.endswith("/u8"):
            return rows
        if label.endswith("/i32"):
            return rows.to(torch.int32)
        return rows.view(torch.float8_e8m0fnu)
    if kind == "deepep":
        hidden, inter = int(shape[0]), int(shape[1])
        grid = _ckpt_scale_2d(u8, (hidden // BLOCK, inter // BLOCK))
        # Replay the exact label the self-test verified, not "whatever the helper does today".
        return _build_fused_scale(label, _moe[entry["layer"]], _shared[entry["layer"]].down_proj,
                                  grid, hidden, inter, resident.device)
    raise RuntimeError(f"unknown scale recipe {kind}")


def _apply(mode: str, seq: int) -> None:
    """Validate every planned write, then perform them all. In place, so CUDA graphs stay valid."""
    t0 = time.time()
    src = _Ckpt(MODEL_ROOT) if mode == "official" else _DeltaSource(_delta, DELTA_BIN)
    plan = []
    for entry in _targets:
        if not entry["dsts"]:
            raise RuntimeError(f"{entry['name']}: no verified destination")
        u8 = src.read_u8(entry["name"])
        if u8.numel() != entry["nbytes"]:
            raise RuntimeError(f"{entry['name']}: source has {u8.numel()} bytes, expected {entry['nbytes']}")
        for d in entry["dsts"]:
            dst = d["tensor"]
            if entry["kind"] == "weight":
                want = u8.view(torch.float8_e4m3fn).reshape(entry["shape"])
                want = want if d["orient"] == "NK" else want.t()
                if tuple(want.shape) != tuple(dst.shape):
                    raise RuntimeError(f"{entry['name']}: resident shape changed since the self-test")
                plan.append((dst, want))
            else:
                cand = _rebuild_scale(entry, d["recipe"], u8, dst)
                if cand is None or cand.dtype != dst.dtype or cand.reshape(-1).numel() != dst.reshape(-1).numel():
                    raise RuntimeError(f"{entry['name']}: rebuilt scale no longer fits {tuple(dst.shape)}")
                plan.append((dst, cand))
    written = 0
    for dst, cand in plan:
        dst.detach().copy_(cand.to(dst.device).reshape(dst.shape))
        written += dst.numel() * dst.element_size()
    del plan
    STATE.update(applied_mode=mode, applied_seq=seq, switches=STATE["switches"] + 1,
                 last_switch_s=round(time.time() - t0, 3),
                 last_switch_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), reason=f"applied {mode}")
    _flush_state()
    _log(f"applied {mode} (seq {seq}) in {STATE['last_switch_s']}s, {written / 2**20:.1f} MiB, "
         f"switch #{STATE['switches']}")


# --------------------------------------------------------------------------- polling


def maybe_poll(model=None) -> None:
    """Engine-thread entry point: rate-limited, skipped during graph capture."""
    global _calls, _last_poll, _mode_stat, _model
    if STATE.get("enabled") is not True:
        return
    if model is not None and model is not _model:
        _model = model
        tree = _unwrap(model)
        if tree is None:
            _error(f"cannot walk the model tree of {type(model).__name__}")
        else:
            try:
                _register_from_model(tree)
                _log(f"model tree {type(tree).__name__}: {len(_linears)} target linears, "
                     f"{len(_moe)} MoE modules, {len(_shared)} shared experts")
                STATE["reason"] = "model found; self-test pending"
                _flush_state()
            except Exception as e:
                _error(f"model walk failed: {e!r}")
    _calls += 1
    if _calls < MIN_CALLS:
        return
    now = time.monotonic()
    if now - _last_poll < POLL_S:
        return
    _last_poll = now
    # Everything below must be unable to break a serving step: any failure is logged and switched off.
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
        if not _targets and not _ensure_ready():
            return
        try:
            st = os.stat(MODE_FILE)
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            return
        if key == _mode_stat:
            return
        with open(MODE_FILE) as f:
            req = json.load(f)
        mode = str(req.get("mode", "official"))
        if mode not in MODES:
            _error(f"mode.json has unknown mode {mode!r}; expected one of {MODES}")
            return
        seq = int(req.get("seq", 0))
        _mode_stat = key
        if mode == STATE["applied_mode"] and seq == STATE["applied_seq"]:
            return
        _apply(mode, seq)
    except Exception as e:  # a switch failure must never take the server down
        _error(f"switch failed: {e!r}")
