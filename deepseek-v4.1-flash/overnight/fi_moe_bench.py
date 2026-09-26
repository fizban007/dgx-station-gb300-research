"""FlashInfer TRTLLM MXFP4 x MXFP8 routed MoE at DeepSeek-V4.1 geometry on one GPU.

Times one MoE layer (all experts resident) for several token counts, with the
same weight conversion and activation quantization vLLM uses for the
FLASHINFER_TRTLLM_MXFP4_MXFP8 backend. Also times a hot/cold split: two calls
over expert ranges [0, hot) and [hot, E) as in expert parallelism.
"""
import argparse
import json
import statistics
import time

import torch

p = argparse.ArgumentParser()
p.add_argument("--experts", type=int, default=384)
p.add_argument("--hidden", type=int, default=5120)
p.add_argument("--intermediate", type=int, default=2304)
p.add_argument("--top-k", type=int, default=6)
p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 256, 1024, 4096, 16384])
p.add_argument("--hot", type=int, default=0, help="also time a split at this expert index")
p.add_argument("--cold-host", action="store_true",
               help="place experts [hot, E) in cacheable mapped Grace memory for the split")
p.add_argument("--stitch", action="store_true",
               help="one VMM tensor per weight: experts [0, hot) in HBM, the rest in Grace")
p.add_argument("--hot-route-share", type=float, default=None,
               help="route this share of top-k slots to hot experts (default: uniform)")
p.add_argument("--output", default=None)
args = p.parse_args()

from flashinfer import trtllm_fp4_block_scale_routed_moe
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend, convert_weight_to_mxfp4_moe_kernel_format,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import _mxfp8_e4m3_quantize_impl

E, H, I, K = args.experts, args.hidden, args.intermediate, args.top_k
dev = torch.device("cuda", 0)
g = torch.Generator(device=dev).manual_seed(0)
w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g)
w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device=dev, generator=g)
s13 = torch.randint(121, 125, (E, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g)
s2 = torch.randint(121, 125, (E, H, I // 32), dtype=torch.uint8, device=dev, generator=g)
w13, w2, s13, s2, _, _ = convert_weight_to_mxfp4_moe_kernel_format(
    mxfp4_backend=Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8, layer=None,
    w13_weight=w13, w2_weight=w2, w13_weight_scale=s13, w2_weight_scale=s2,
    _cache_permute_indices={},
)
clamp = torch.full((E,), 10.0, dtype=torch.float32, device=dev)
torch.cuda.synchronize()
cold = None
if args.hot and args.stitch:
    import sys
    sys.path.insert(0, "/home/jasonc/vllm-karmic/vllm/model_executor/layers/fused_moe/experts")
    from vmm_stitch import stitch
    stitched = [stitch(t, args.hot, dev) for t in (w13, w2, s13, s2)]
    w13, w2, s13, s2 = (x.tensor for x in stitched)
    torch.cuda.synchronize()
if args.hot and args.cold_host:
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    owners, views = [], []
    for t in (w13, w2, s13, s2):
        part = t[args.hot:].contiguous()
        owner = MappedHostAllocation(tuple(part.shape), part.dtype, dev)
        owner.host_view.copy_(part.cpu())
        owners.append(owner); views.append(owner.device_view)
    cold = views
    torch.cuda.synchronize()


def call(x_q, x_s, ids, wts, out, offset, count):
    sl = slice(offset, offset + count)
    a, b, c, d = w13[sl], w2[sl], s13[sl], s2[sl]
    if cold is not None and offset == args.hot:
        a, b, c, d = cold
    trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, wts), routing_bias=None, hidden_states=x_q, hidden_states_scale=x_s,
        gemm1_weights=a, gemm1_weights_scale=c, gemm1_bias=None, gemm1_alpha=None,
        gemm1_beta=None, gemm1_clamp_limit=clamp[sl], gemm2_weights=b, gemm2_weights_scale=d,
        gemm2_bias=None, output1_scale_scalar=None, output1_scale_gate_scalar=None,
        output2_scale_scalar=None, num_experts=E, top_k=K, n_group=None, topk_group=None,
        intermediate_size=I, local_expert_offset=offset, local_num_experts=count,
        routed_scaling_factor=None, routing_method_type=1, do_finalize=True, enable_pdl=True,
        output=out, tune_max_num_tokens=max(args.tokens),
    )


def timed(fn, reps=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    samples = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); graph.replay(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    graph.reset()
    return statistics.median(samples)


rows = []
for m in args.tokens:
    x = torch.randn(m, H, dtype=torch.bfloat16, device=dev, generator=g) * 0.1
    x_q, x_s = _mxfp8_e4m3_quantize_impl(x, is_sf_swizzled_layout=False)
    scores = torch.rand(m, E, device=dev, generator=g)
    if args.hot_route_share is not None and args.hot:
        # Hot experts win a slot unless a Bernoulli draw sends it cold.
        bias = torch.zeros(m, E, device=dev)
        bias[:, :args.hot] = 1.0
        cold_slots = torch.rand(m, 1, device=dev, generator=g) > args.hot_route_share
        scores = scores + torch.where(cold_slots, -bias, bias)
    wts, ids = scores.topk(K, dim=-1)
    ids, wts = ids.to(torch.int32).contiguous(), torch.softmax(wts, -1).to(torch.bfloat16).contiguous()
    out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    full = timed(lambda: call(x_q, x_s, ids, wts, out, 0, E))
    row = {"tokens": m, "full_us": full}
    if args.hot:
        out2 = torch.empty_like(out)
        row["split_us"] = timed(lambda: (call(x_q, x_s, ids, wts, out, 0, args.hot),
                                         call(x_q, x_s, ids, wts, out2, args.hot, E - args.hot)))
    touched = torch.unique(ids).numel()
    row["unique_experts"] = touched
    row["weight_gbps"] = touched * (3 * H * I // 2 + 3 * H * I // 32) / (full * 1e-6) / 1e9
    row["tok_per_s"] = m / (full * 1e-6)
    rows.append(row)
    print(json.dumps(row), flush=True)
if args.output:
    json.dump(rows, open(args.output, "w"), indent=1)
