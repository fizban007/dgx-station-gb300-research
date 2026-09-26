"""Can the RTX PRO 6000 run a prefill chunk's cold routes inside the GB300's hot-MoE window?

One layer's 89 cold experts (real checkpoint weights, rowmap order) on the 6000. Input: M compacted
tokens, each with one cold route (local id) and five masked (-1) routes, which is what an 8K prefill chunk
sends (~2.2% of 49K routes are cold -> ~1,000 tokens). Times b12x fused_moe (w4a8_mx, SM120-native)
against the sidecar's per-route Triton GEMV, and checks b12x against it.
Target: under ~1.7 ms per layer (MegaMoE hot + shared expert per 8K chunk on the GB300).
"""
import json
import os
import statistics
import struct
import sys

import torch

sys.path.insert(0, "/home/jasonc/vllm-karmic/vllm/model_executor/layers/fused_moe/experts")
from cold_moe_triton import ColdExperts  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402
from b12x.preparation import PreparationSession  # noqa: E402
from b12x.preparation.types import require_prepared  # noqa: E402

MODEL = "/home/jasonc/models/DeepSeek-V4.1-Flash"
ROWMAP = os.environ.get("ROWMAP_FILE", "/home/jasonc/research/megamoe/hook/rowmap-static-v1.json")
LAYER = int(os.environ.get("LAYER", "20"))
H, I, K, LIMIT = 5120, 2304, 6, 10.0
MS = [int(v) for v in os.environ.get("MS", "64 256 512 1024 2048").split()]
index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_headers = {}
dev = None


def tensor(name):
    file = os.path.join(MODEL, index[name])
    if file not in _headers:
        with open(file, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            _headers[file] = (8 + size, json.loads(f.read(size)))
    base, header = _headers[file]
    meta = header[name]
    start, end = meta["data_offsets"]
    with open(file, "rb") as f:
        f.seek(base + start)
        data = f.read(end - start)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).view(meta["shape"])


def mxfp8(x):
    m = x.shape[0]
    xb = x.float().view(m, H // 32, 32)
    e = torch.ceil(torch.log2(xb.abs().amax(-1).clamp_min(1e-30) / 448.0))
    q = (xb / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn)
    deq = (q.float() * torch.exp2(e)[..., None]).view(m, H).to(torch.bfloat16)
    return q.view(m, H), (e + 127).to(torch.uint8), deq


FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def dequant(w, s):
    """[N, K/2] e2m1 nibbles (low = even) and [N, K/32] E8M0 bytes -> float [N, K]."""
    lut = torch.tensor(FP4, device=w.device)
    v = torch.stack([lut[(w & 15).long()], lut[(w >> 4).long()]], dim=-1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, dim=-1)


def dense_ref(W, x, ids, wts):
    out = torch.zeros(x.shape[0], H, device=x.device)
    for t in range(x.shape[0]):
        for k in range(K):
            e = int(ids[t, k])
            if e < 0:
                continue
            h = dequant(W["w13"][e], W["s13"][e]) @ x[t].float()
            gate, up = h[:I].clamp(max=LIMIT), h[I:].clamp(-LIMIT, LIMIT)
            y = dequant(W["w2"][e], W["s2"][e]) @ (torch.nn.functional.silu(gate) * up)
            out[t] += float(wts[t, k]) * y
    return out


def main():
    global dev
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    sys.path.insert(0, "/home/jasonc/b12x-exp")
    from benchmarks.moe_preparation import prepared_call, request_for_capacity
    torch.cuda.set_device(dev)



    cold = json.load(open(ROWMAP))["layers"][str(LAYER)]["cold"]
    if os.environ.get("ALL") == "1":
        cold = list(range(384))
    E = len(cold)
    parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
    for e in cold:
        p = f"layers.{LAYER}.ffn.experts.{e}"
        parts["w13"].append(torch.cat([tensor(f"{p}.w1.weight"), tensor(f"{p}.w3.weight")]))
        parts["s13"].append(torch.cat([tensor(f"{p}.w1.scale"), tensor(f"{p}.w3.scale")]))
        parts["w2"].append(tensor(f"{p}.w2.weight"))
        parts["s2"].append(tensor(f"{p}.w2.scale"))
    W = {k: torch.stack(v).to(dev) for k, v in parts.items()}
    print(f"layer {LAYER}: {E} cold experts, {sum(t.numel() for t in W.values()) / 1e9:.2f} GB", flush=True)

    W_before = {k: v.clone() for k, v in W.items()}  # prepare_weights may rewrite its sources in place
    triton_ref = ColdExperts(W_before["w13"], W_before["s13"], W_before["w2"], W_before["s2"], cold_start=0, limit=LIMIT, gate_first=True)

    ones = torch.ones(E, dtype=torch.float32, device=dev)
    one = torch.ones((), dtype=torch.float32, device=dev)  # scalar input scales, as b12x's benchmark
    plan = fm.plan_weights(
        source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fm.W13Layout(os.environ.get("W13", "w31"))),
        activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                     swiglu_limit=LIMIT, swiglu_alpha=None, swiglu_beta=None),
        geometry=fm.MoEGeometry(num_experts=E, hidden_size=H, intermediate_size=I),
    )
    # b12x's DS-V4.1 loader: E8M0 scale dtype, bytes clamped to 247, [w1, w3] rows as layout "w31".
    s13 = W["s13"].clone().clamp_(max=247).view(torch.float8_e8m0fnu)
    s2 = W["s2"].clone().clamp_(max=247).view(torch.float8_e8m0fnu)
    experts = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
        w13=W["w13"], w2=W["w2"], w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one,
        immutable_input_scales=True))


    g = torch.Generator(device=dev).manual_seed(0)
    inputs, outputs = {}, {}
    for m in MS:
        x = torch.randn(m, H, device=dev, generator=g) * 0.3
        xq, xs, xdq = mxfp8(x)
        ids = torch.full((m, K), -1, dtype=torch.int32, device=dev)
        ids[:, 0] = torch.randint(0, E, (m,), device=dev, generator=g, dtype=torch.int32)
        wts = torch.zeros(m, K, dtype=torch.float32, device=dev)
        wts[:, 0] = torch.rand(m, device=dev, generator=g) * 0.3 + 0.05
        if os.environ.get("ALLVALID") == "1":
            ids = torch.randint(0, E, (m, K), device=dev, generator=g, dtype=torch.int32)
            wts = torch.rand(m, K, device=dev, generator=g) * 0.3 + 0.05
        inputs[m] = (xq, xs, xdq, ids, wts)
        outputs[m] = torch.empty(m, H, dtype=torch.bfloat16, device=dev)

    session = PreparationSession(device=dev)
    declaration = fm.plan_execution(experts=experts, capacity=fm.ExecutionCapacity(
        max_tokens=max(MS), top_k=K, warmup_token_counts=tuple(MS), route_num_experts=0))
    calls = {m: prepared_call(output=outputs[m], bind=lambda st, scratch, m=m: st.bind(
        scratch=scratch, a=inputs[m][2], experts=experts, topk_ids=inputs[m][3], topk_weights=inputs[m][4],
        output=outputs[m], input_scales_static=True)) for m in getattr(declaration, "token_counts", MS)}
    from dataclasses import replace

    def race_call(state, m):
        call = calls[m](state)
        source = inputs[m][2].clone()
        return replace(call, produce=lambda: inputs[m][2].copy_(source), owners=(*call.owners, source))

    session.prepare((request_for_capacity(declaration, name="cold", calls=calls,
                                          benchmark_calls={m: (lambda st, m=m: race_call(st, m)) for m in calls}),))


    for m_, v in getattr(declaration, "variants", {}).items():
        print(json.dumps({"tokens": m_, "selection": str(getattr(v.selection, "config", v.selection))[:200]}), flush=True)

    def bind(m):
        ex = getattr(declaration, "variants", {}).get(m, declaration)
        state = require_prepared(ex, ex.component_id)
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=s.device) for s in state.scratch.scratch_specs())
        return fm.bind(ex, scratch=scratch, a=inputs[m][2], experts=experts, topk_ids=inputs[m][3],
                       topk_weights=inputs[m][4], output=outputs[m], input_scales_static=True), scratch


    def timed(fn, reps=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize(); t.append(a.elapsed_time(b) * 1000)
        return statistics.median(t)


    rows = []
    for m in MS:
        xq, xs, xdq, ids, wts = inputs[m]
        binding, _scratch = bind(m)
        fm.run(binding=binding)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fm.run(binding=binding)
        b12x_us = timed(graph.replay)
        ref = torch.zeros(m, H, dtype=torch.bfloat16, device=dev)
        triton_ref.add_to(ref, xq, xs, ids, wts)
        returned = fm.run(binding=binding)
        torch.cuda.synchronize()
        print(json.dumps({"returned_is_output": returned is outputs[m] or (isinstance(returned, torch.Tensor) and returned.data_ptr() == outputs[m].data_ptr()),
                          "returned_type": type(returned).__name__}), flush=True)
        out = (returned if isinstance(returned, torch.Tensor) else outputs[m]).float()
        cos = float(torch.nn.functional.cosine_similarity(out.flatten(), ref.float().flatten(), dim=0))
        rel = float((out - ref.float()).norm() / ref.float().norm().clamp_min(1e-9))
        tri = torch.zeros(m, H, dtype=torch.bfloat16, device=dev)
        triton_us = timed(lambda: (tri.zero_(), triton_ref.add_to(tri, xq, xs, ids, wts)), reps=5) if m <= 1024 else None
        dense = dense_ref(W_before, xdq[:4], ids[:4], wts[:4])
        print(json.dumps({"weights_changed_by_prepare": {k: bool((W[k] != W_before[k]).any()) for k in W}}), flush=True)
        from b12x.moe._shared.kernels.reference import moe_reference_w4a8_mx
        for lay in ("w13", "w31"):
            orc = moe_reference_w4a8_mx(xdq[:4], W["w13"], W["s13"], None, ones, W["w2"], W["s2"], None, ones,
                                        ids[:4].long(), wts[:4], E, H, I, activation="silu", swiglu_limit=LIMIT, w13_layout=lay)
            print(json.dumps({"layout": lay, "oracle_vs_dense": round(float(torch.nn.functional.cosine_similarity(orc.float().flatten(), dense.flatten(), dim=0)), 5),
                              "oracle_vs_b12x": round(float(torch.nn.functional.cosine_similarity(orc.float().flatten(), out[:4].flatten(), dim=0)), 5)}), flush=True)
        def c(u, v):
            return round(float(torch.nn.functional.cosine_similarity(u.float().flatten(), v.float().flatten(), dim=0)), 5)
        print(json.dumps({"tokens": m, "dense_vs_b12x": c(dense, out[:4]), "dense_vs_triton": c(dense, ref[:4])}), flush=True)
        row = {"tokens": m, "b12x_us": round(b12x_us, 1), "triton_gemv_us": round(triton_us, 1) if triton_us else None,
               "cosine_vs_triton": round(cos, 5), "rel_diff": round(rel, 4)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open("/home/jasonc/research/megamoe/peer_prefill_bench.json", "w"), indent=1)
    if os.environ.get("REPLAY_TEST") == "1":
        # Capture with routing A, overwrite the bound buffers in place with routing B, replay, compare with B.
        m = MS[0]
        xq, xs, xdq, ids, wts = inputs[m]
        binding, _scratch = bind(m)
        fm.run(binding=binding); torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fm.run(binding=binding)
        g2 = torch.Generator(device=dev).manual_seed(99)
        new_ids = torch.full_like(ids, -1)
        new_ids[:, 0] = torch.randint(0, E, (m,), device=dev, generator=g2, dtype=torch.int32)
        new_w = torch.zeros_like(wts); new_w[:, 0] = torch.rand(m, device=dev, generator=g2) * 0.3 + 0.05
        new_x = (torch.randn(m, H, device=dev, generator=g2) * 0.3)
        nxq, nxs, nxdq = mxfp8(new_x)
        ids.copy_(new_ids); wts.copy_(new_w); xdq.copy_(nxdq)
        graph.replay(); torch.cuda.synchronize()
        graph_out = outputs[m].float().clone()
        fm.run(binding=binding); torch.cuda.synchronize()
        eager_out = outputs[m].float().clone()
        ref = dense_ref(W_before, nxdq[:4], new_ids[:4], new_w[:4])
        c = lambda u, v: round(float(torch.nn.functional.cosine_similarity(u.flatten(), v.flatten(), dim=0)), 5)
        print(json.dumps({"replay_after_new_routing_vs_dense": c(graph_out[:4], ref), "eager_vs_dense": c(eager_out[:4], ref)}), flush=True)
    if os.environ.get("PROFILE") == "1":
        m = max(MS)
        binding, _scratch = bind(m)
        for _ in range(3):
            fm.run(binding=binding)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                fm.run(binding=binding)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=90))


if __name__ == "__main__":
    main()
