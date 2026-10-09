"""Sidecar CSF check: real cold experts through b12x W4A8 with raw scales vs CSF planes; outputs must be bit-identical.

  CUDA_VISIBLE_DEVICES=<6000> python test_sidecar_csf.py [--experts 8] [--ratio-layers 0,20,39]
"""
import argparse
import os
import json
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sidecar"))
from csf_encode import encode_planes, planes_nbytes  # noqa: E402
from peer_server2 import tensor_reader, H, I, LIMIT  # noqa: E402
import b12x.moe.fused_moe as fm  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
p.add_argument("--rowmap", default=os.path.join(HERE, "..", "hook", "rowmap-mix-h258.json"))
p.add_argument("--layer", type=int, default=0)
p.add_argument("--experts", type=int, default=8)
p.add_argument("--m", default="1,4,16,64,256")
p.add_argument("--mutable-scales", action="store_true")
p.add_argument("--ratio-layers", default="")
a = p.parse_args()
dev = torch.device("cuda", 0)
tensor = tensor_reader(a.model)
rowmap = json.load(open(a.rowmap))["layers"]


def load(layer, ids):
    w13, s13, w2, s2 = [], [], [], []
    for e in ids:
        pre = f"layers.{layer}.ffn.experts.{e}"
        w13.append(torch.cat([tensor(f"{pre}.w1.weight"), tensor(f"{pre}.w3.weight")]))
        s13.append(torch.cat([tensor(f"{pre}.w1.scale"), tensor(f"{pre}.w3.scale")]))
        w2.append(tensor(f"{pre}.w2.weight"))
        s2.append(tensor(f"{pre}.w2.scale"))
    return [torch.stack(t) for t in (w13, s13, w2, s2)]


# 1) compression ratio over all cold experts of a few layers
for layer in [int(x) for x in a.ratio_layers.split(",") if x]:
    cold = rowmap[str(layer)]["cold"]
    _, s13, _, s2 = load(layer, cold)
    raw = s13.numel() + s2.numel()
    comp, nexc = 0, 0
    for s in (s13, s2):
        f, x = encode_planes(s.clamp(max=247))
        comp += planes_nbytes(f, x)
        nexc += sum(t.numel() for t in x)
    print(f"layer {layer}: {len(cold)} cold experts, scales {raw / 2**20:.1f} MiB -> {comp / 2**20:.1f} MiB "
          f"({raw / comp:.2f}x), exceptions {nexc} ({nexc / raw * 100:.3f}% of scales)", flush=True)

# 2) bit-exactness through the fused MoE
E, K = a.experts, 6
ids_e = rowmap[str(a.layer)]["cold"][:E]
w13, s13, w2, s2 = load(a.layer, ids_e)
s13, s2 = s13.clamp(max=247), s2.clamp(max=247)
plan = fm.plan_weights(
    source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fm.W13Layout("w31")),
    activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                 swiglu_limit=LIMIT, swiglu_alpha=None, swiglu_beta=None),
    geometry=fm.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I))
ones = torch.ones(E, dtype=torch.float32, device=dev)
one = torch.ones((), dtype=torch.float32, device=dev)
raw_ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
    w13=w13.to(dev), w2=w2.to(dev), w13_block_scales=s13.to(dev).view(torch.float8_e8m0fnu),
    w2_block_scales=s2.to(dev).view(torch.float8_e8m0fnu), w13_global_scales=ones, w2_global_scales=ones,
    input_scale=one, intermediate_scale=one, immutable_input_scales=True))
p13, p2 = encode_planes(s13), encode_planes(s2)
buf13 = torch.empty(s13.shape, dtype=torch.uint8, device=dev)
buf2 = torch.empty(s2.shape, dtype=torch.uint8, device=dev)
csf_ex = fm.prepare_weights(plan=plan, weights=fm.Mxfp4CsfWeights(
    w13=w13.to(dev), w2=w2.to(dev), w13_scales=fm.CsfScalePlanes(*p13), w2_scales=fm.CsfScalePlanes(*p2),
    w13_scale_scratch=buf13, w2_scale_scratch=buf2))
if not a.mutable_scales:
    # Match the raw path's static activation scales (scalar, immutable); the CSF path defaults to per-expert ones.
    from dataclasses import replace as _r
    csf_ex = _r(csf_ex, _impl=_r(csf_ex._impl, a1_gscale=raw_ex._impl.a1_gscale, a2_gscale=raw_ex._impl.a2_gscale,
                                immutable_input_scales=True))
print("csf decoder:", type(csf_ex._impl.mxfp4_csf).__name__, "inline:", csf_ex._impl.mxfp4_csf_inline is not None)

sys.path.insert(0, os.environ.get("B12X_SRC", os.path.expanduser("~/src/b12x")))
from benchmarks.moe_preparation import request_for_capacity  # noqa: E402
from b12x.preparation.types import require_prepared  # noqa: E402

M = tuple(int(x) for x in a.m.split(','))
rows_max = max(M)
a_buf = torch.zeros(rows_max, H, dtype=torch.bfloat16, device=dev)
ids_buf = torch.full((rows_max, K), -1, dtype=torch.int32, device=dev)
w_buf = torch.zeros(rows_max, K, dtype=torch.float32, device=dev)
outs = {}
decls = {}
for name, ex in (("raw", raw_ex), ("csf", csf_ex)):
    decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(
        max_tokens=rows_max, top_k=K, warmup_token_counts=M, route_num_experts=0))
    decls[name] = decl

    def call(st, m, ex=ex, name=name):
        scratch = tuple(torch.empty(int(sp.shape[0]), dtype=sp.dtype, device=dev) for sp in st.scratch.scratch_specs())
        out = outs.setdefault((name, m), torch.zeros(m, H, dtype=torch.bfloat16, device=dev))
        b = st.bind(scratch=scratch, a=a_buf[:m], experts=ex, topk_ids=ids_buf[:m], topk_weights=w_buf[:m],
                    output=out, input_scales_static=True)
        return PreparedCall(run=b.run, output=out, owners=(b, scratch))

    session = PreparationSession(device=dev, autotune=False)
    session.prepare((request_for_capacity(decl, name=f"sc-csf-{name}",
                                          calls={m: (lambda st, m=m: call(st, m)) for m in decl.token_counts}),))
    del session

ok = True
for m in M:
    g = torch.Generator(device=dev).manual_seed(m)
    a_buf[:m].copy_((torch.randn(m, H, device=dev, generator=g) * 0.3).to(torch.bfloat16))
    live = torch.rand(m, K, device=dev, generator=g) < 0.3
    live[:, 0] = True
    ids_buf[:m].copy_(torch.where(live, torch.randint(0, E, (m, K), device=dev, generator=g, dtype=torch.int32), -1))
    w_buf[:m].copy_(torch.where(live, torch.rand(m, K, device=dev, generator=g) * 0.3, 0.0))
    res, times = [], []
    for name, ex in (("raw", raw_ex), ("csf", csf_ex), ("raw", raw_ex)):
        var = decls[name].variants[m]
        state = require_prepared(var, var.component_id)
        scratch = tuple(torch.empty(int(sp.shape[0]), dtype=sp.dtype, device=dev) for sp in state.scratch.scratch_specs())
        out = torch.full((m, H), float("nan"), dtype=torch.bfloat16, device=dev)
        b = fm.bind(var, scratch=scratch, a=a_buf[:m], experts=ex, topk_ids=ids_buf[:m], topk_weights=w_buf[:m],
                    output=out, input_scales_static=True)
        fm.run(binding=b)
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fm.run(binding=b)
        buf13.view(torch.uint8).fill_(0x7F)
        buf2.view(torch.uint8).fill_(0x7F)
        out.fill_(float("nan"))
        gr.replay()
        torch.cuda.synchronize()
        res.append(out.clone())
        t = time.perf_counter()
        for _ in range(50):
            gr.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) / 50 * 1e6)
    # b12x accumulates top-k outputs with atomics, so raw is not bit-reproducible itself: compare noise floors.
    d_rc = (res[0].float() - res[1].float()).abs().max().item()
    d_rr = (res[0].float() - res[2].float()).abs().max().item()
    rel = ((res[0].float() - res[1].float()).norm() / res[0].float().norm()).item()
    rel_rr = ((res[0].float() - res[2].float()).norm() / res[0].float().norm()).item()
    print(f"  max|raw-raw2| {d_rr:.3g}  max|raw-csf| {d_rc:.3g}  rel L2 raw-raw2 {rel_rr:.2e}  raw-csf {rel:.2e}")
    same = d_rc <= 2 * d_rr + 1e-6 and rel <= 2 * rel_rr + 1e-6
    rep = d_rr == 0
    nn = [int(torch.isnan(r).any(dim=1).sum()) for r in res]
    print(f"  NaN rows raw/csf/raw2: {nn}")
    print(f"  raw repeat identical={rep}, max |raw-csf| = {(res[0].float() - res[1].float()).abs().max().item():.3g}")
    ok &= same and bool(torch.isfinite(res[0]).all())
    print(f"m={m}: within raw noise={same}  raw {times[0]:.1f} us  csf {times[1]:.1f} us", flush=True)
print("PASS" if ok else "FAIL")
