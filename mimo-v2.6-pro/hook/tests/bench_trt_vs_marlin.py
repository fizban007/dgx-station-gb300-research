"""TRT-LLM (FlashInfer routed MXFP4 x MXFP8) vs Marlin (MXFP4 x BF16) on the same MiMo-V2.6-Pro experts, weights in HBM.

A bank of N experts from layer 5, each token routing R of its 8 slots into the bank (the rest masked), T tokens.
Marlin gets global ids + an expert_map (hotsplit's HBM bank); TRT-LLM gets bank-local ids with -1 elsewhere (vLLM's
TRT wrapper ignores expert_map). TRT-LLM is autotuned per T first; small T is timed inside CUDA graphs.
Run in the vLLM image with the checkpoint at /model on the GB300.
"""
import json
import os
import struct
import sys
import types

import torch

dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
N = int(os.environ.get("BANK", "96"))
R = int(os.environ.get("ROUTES", "5"))
LAYER, H, I, K, E = 5, 6144, 2048, 8, 384

index = json.load(open("/model/model.safetensors.index.json"))["weight_map"]
headers = {}


def tensor(name):
    f = os.path.join("/model", index[name])
    if f not in headers:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            headers[f] = (8 + n, json.loads(fh.read(n)))
    base, h = headers[f]
    s, e = h[name]["data_offsets"]
    with open(f, "rb") as fh:
        fh.seek(base + s)
        return torch.frombuffer(bytearray(fh.read(e - s)), dtype=torch.uint8).view(h[name]["shape"])


parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
for e in range(N):
    p = f"model.layers.{LAYER}.mlp.experts.{e}"
    parts["w13"].append(torch.cat([tensor(f"{p}.gate_proj.weight"), tensor(f"{p}.up_proj.weight")]))
    parts["s13"].append(torch.cat([tensor(f"{p}.gate_proj.weight_scale"), tensor(f"{p}.up_proj.weight_scale")]))
    parts["w2"].append(tensor(f"{p}.down_proj.weight"))
    parts["s2"].append(tensor(f"{p}.down_proj.weight_scale"))
raw = {k: torch.stack(v).to(dev) for k, v in parts.items()}
del parts
print(f"bank: {N} experts, {sum(t.numel() for t in raw.values()) / 2**30:.2f} GiB raw", flush=True)

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (  # noqa: E402
    Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (  # noqa: E402
    prepare_moe_mxfp4_layer_for_marlin)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_impl  # noqa: E402
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import MoEActivation  # noqa: E402
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import activation_to_flashinfer_int  # noqa: E402
import flashinfer  # noqa: E402
from flashinfer.autotuner import autotune  # noqa: E402

with set_current_vllm_config(VllmConfig()):
    cl = {k: v.clone() for k, v in raw.items()}
    tw13, tw2, ts13, ts2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, layer=None, w13_weight=cl["w13"],
        w2_weight=cl["w2"], w13_weight_scale=cl["s13"], w2_weight_scale=cl["s2"], _cache_permute_indices={})
    cl = {k: v.clone() for k, v in raw.items()}
    mw13, mw2, ms13, ms2, _, _ = prepare_moe_mxfp4_layer_for_marlin(
        types.SimpleNamespace(params_dtype=torch.bfloat16), cl["w13"], cl["w2"], cl["s13"], cl["s2"], None, None)
del cl
torch.cuda.empty_cache()
act = activation_to_flashinfer_int(MoEActivation.SILU)
emap = torch.full((E,), -1, dtype=torch.int32, device=dev)
emap[:N] = torch.arange(N, dtype=torch.int32, device=dev)
g = torch.Generator(device=dev).manual_seed(0)


def inputs(T):
    x = (torch.randn(T, H, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    local = torch.stack([torch.randperm(N, device=dev, generator=g)[:R] for _ in range(T)]).int()
    other = torch.stack([N + torch.randperm(E - N, device=dev, generator=g)[: K - R] for _ in range(T)]).int()
    gids = torch.cat([local, other], 1)                      # Marlin: global ids, masked by expert_map
    lids = torch.cat([local, torch.full_like(other, -1)], 1)  # TRT: bank-local ids, -1 elsewhere
    w = torch.softmax(torch.randn(T, K, device=dev, generator=g), -1)
    return x, gids, lids, w


def run_trt(x, lids, w, out):
    xq, xs = _mxfp8_e4m3_quantize_impl(x, is_sf_swizzled_layout=False)
    flashinfer.trtllm_fp4_block_scale_routed_moe(
        topk_ids=(lids, w.to(torch.bfloat16)), routing_bias=None, hidden_states=xq, hidden_states_scale=xs,
        gemm1_weights=tw13, gemm1_weights_scale=ts13, gemm1_bias=None, gemm1_alpha=None, gemm1_beta=None,
        gemm1_clamp_limit=None, gemm2_weights=tw2, gemm2_weights_scale=ts2, gemm2_bias=None,
        output1_scale_scalar=None, output1_scale_gate_scalar=None, output2_scale_scalar=None, num_experts=N,
        top_k=K, n_group=None, topk_group=None, intermediate_size=I, local_expert_offset=0, local_num_experts=N,
        routed_scaling_factor=None, routing_method_type=1, do_finalize=True, enable_pdl=True,
        activation_type=act, output=out, tune_max_num_tokens=8192)
    return out


def run_marlin(x, gids, w):
    return fused_marlin_moe(x, mw13, mw2, None, None, ms13, ms2, w, gids, scalar_types.float4_e2m1f.id,
                            global_num_experts=E, activation=MoEActivation.SILU, expert_map=emap)


def time_it(fn, graphed, reps=30):
    fn(); torch.cuda.synchronize()
    if graphed:
        gr = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(dev)
        with torch.cuda.graph(gr, stream=s):
            fn()
        call = gr.replay
    else:
        call = fn
    call(); torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(reps):
        call()
    t1.record(); torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1e3 / reps


bytes_per_expert = sum(t[0].numel() * t.element_size() for t in (tw13, tw2, ts13, ts2))
MODE = os.environ.get("MODE", "compare")
with set_current_vllm_config(VllmConfig()):
    for T in (1, 2, 4, 8, 16, 32, 128, 512, 2048, 8192):
        x, gids, lids, w = inputs(T)
        out = torch.empty(T, H, dtype=torch.bfloat16, device=dev)
        graphed = T <= 16
        if MODE == "tuning":
            # untuned (heuristic) first, then tuned, same inputs
            tu = time_it(lambda: run_trt(x, lids, w, out), graphed)
            with autotune(True):
                run_trt(x, lids, w, out)
            torch.cuda.synchronize()
            tt = time_it(lambda: run_trt(x, lids, w, out), graphed)
            print(f"T={T:5d}: TRT untuned {tu:8.1f} us  tuned {tt:8.1f} us  untuned/tuned {tu / tt:4.2f}", flush=True)
            continue
        with autotune(True):
            run_trt(x, lids, w, out)
        torch.cuda.synchronize()
        ref = run_marlin(x, gids, w).float()
        got = run_trt(x, lids, w, out).float()
        cos = float(torch.nn.functional.cosine_similarity(ref.flatten(), got.flatten(), dim=0))
        tm = time_it(lambda: run_marlin(x, gids, w), graphed)
        tt = time_it(lambda: run_trt(x, lids, w, out), graphed)
        distinct = len(set(lids[lids >= 0].tolist()))
        gb = distinct * bytes_per_expert / 1e9
        print(f"T={T:5d} distinct={distinct:3d}: Marlin {tm:8.1f} us ({gb / tm * 1e6 / 1e3:5.2f} TB/s)  "
              f"TRT {tt:8.1f} us ({gb / tt * 1e6 / 1e3:5.2f} TB/s)  speedup {tm / tt:4.2f}x  cos {cos:.5f}", flush=True)
