"""How long does b12x preparation take per layer for the sidecar's bucket set? Two layers, 14 buckets."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/home/jasonc/research/megamoe")
from peer_prefill_bench import H, I, K, LIMIT, ROWMAP, tensor  # noqa: E402

import b12x.moe.fused_moe as fm  # noqa: E402
from b12x.preparation import PreparationSession  # noqa: E402

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def main():
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    sys.path.insert(0, "/home/jasonc/b12x-exp")
    from dataclasses import replace
    from benchmarks.moe_preparation import prepared_call, request_for_capacity

    rowmap = json.load(open(ROWMAP))["layers"]
    session = PreparationSession(device=dev)
    a = torch.randn(max(BUCKETS), H, device=dev, dtype=torch.bfloat16) * 0.3
    ids = torch.full((max(BUCKETS), K), -1, dtype=torch.int32, device=dev)
    ids[:, 0] = torch.randint(0, 89, (max(BUCKETS),), device=dev, dtype=torch.int32)
    w = torch.zeros(max(BUCKETS), K, device=dev)
    w[:, 0] = 0.2
    out = torch.empty(max(BUCKETS), H, device=dev, dtype=torch.bfloat16)
    for layer in (int(v) for v in os.environ.get("LAYERS", "0 1").split()):
        cold = rowmap[str(layer)]["cold"]
        parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
        for e in cold:
            p = f"layers.{layer}.ffn.experts.{e}"
            parts["w13"].append(torch.cat([tensor(f"{p}.w1.weight"), tensor(f"{p}.w3.weight")]))
            parts["s13"].append(torch.cat([tensor(f"{p}.w1.scale"), tensor(f"{p}.w3.scale")]))
            parts["w2"].append(tensor(f"{p}.w2.weight"))
            parts["s2"].append(tensor(f"{p}.w2.scale"))
        W = {k: torch.stack(v).to(dev) for k, v in parts.items()}
        t0 = time.time()
        ones = torch.ones(len(cold), dtype=torch.float32, device=dev)
        one = torch.ones((), dtype=torch.float32, device=dev)
        plan = fm.plan_weights(
            source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fm.W13Layout("w31")),
            activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16,
                                         swiglu_limit=LIMIT, swiglu_alpha=None, swiglu_beta=None),
            geometry=fm.MoEGeometry(num_experts=len(cold), hidden_size=H, intermediate_size=I))
        experts = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(
            w13=W["w13"], w2=W["w2"],
            w13_block_scales=W["s13"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w2_block_scales=W["s2"].clamp_(max=247).view(torch.float8_e8m0fnu),
            w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one,
            immutable_input_scales=True))
        t1 = time.time()
        decl = fm.plan_execution(experts=experts, capacity=fm.ExecutionCapacity(
            max_tokens=max(BUCKETS), top_k=K, warmup_token_counts=tuple(BUCKETS), route_num_experts=0))
        calls = {m: prepared_call(output=out[:m], bind=lambda st, scratch, m=m: st.bind(
            scratch=scratch, a=a[:m], experts=experts, topk_ids=ids[:m], topk_weights=w[:m], output=out[:m],
            input_scales_static=True)) for m in decl.token_counts}

        def race(st, m):
            call = calls[m](st)
            src = a[:m].clone()
            return replace(call, produce=lambda: a[:m].copy_(src), owners=(*call.owners, src))

        session.prepare((request_for_capacity(decl, name=f"cold.l{layer}", calls=calls,
                                              benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
        t2 = time.time()
        sel = {m: str(v.selection.config)[:60] if v.selection is not None else None for m, v in decl.variants.items()}
        print(json.dumps({"layer": layer, "prepare_weights_s": round(t1 - t0, 1), "plan_and_tune_s": round(t2 - t1, 1),
                          "sources": sorted({str(v.selection.source) for v in decl.variants.values() if v.selection})}), flush=True)
        print(json.dumps(sel)[:600], flush=True)


if __name__ == "__main__":
    main()
