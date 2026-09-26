import gc, sys, torch
sys.path.insert(0, "/home/jasonc/research/megamoe")
from peer_prefill_bench import H, I, LIMIT, ROWMAP, tensor
import json
import b12x.moe.fused_moe as fm

def walk(obj, seen, out, depth=0):
    if id(obj) in seen or depth > 6: return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        out.append(obj); return
    if isinstance(obj, (list, tuple)):
        for v in obj: walk(v, seen, out, depth+1); return
    if isinstance(obj, dict):
        for v in obj.values(): walk(v, seen, out, depth+1); return
    d = getattr(obj, "__dict__", None)
    if d is None and hasattr(obj, "__slots__"):
        d = {k: getattr(obj, k, None) for k in obj.__slots__}
    if d is None and hasattr(obj, "__dataclass_fields__"):
        d = {k: getattr(obj, k, None) for k in obj.__dataclass_fields__}
    if d:
        for v in d.values(): walk(v, seen, out, depth+1)

def main():
    dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
    cold = json.load(open(ROWMAP))["layers"]["0"]["cold"]
    parts = {k: [] for k in ("w13", "s13", "w2", "s2")}
    for e in cold:
        p = f"layers.0.ffn.experts.{e}"
        parts["w13"].append(torch.cat([tensor(f"{p}.w1.weight"), tensor(f"{p}.w3.weight")]))
        parts["s13"].append(torch.cat([tensor(f"{p}.w1.scale"), tensor(f"{p}.w3.scale")]))
        parts["w2"].append(tensor(f"{p}.w2.weight")); parts["s2"].append(tensor(f"{p}.w2.scale"))
    m0 = torch.cuda.memory_allocated()
    W = {k: torch.stack(v).to(dev) for k, v in parts.items()}
    m1 = torch.cuda.memory_allocated()
    ones = torch.ones(len(cold), device=dev); one = torch.ones((), device=dev)
    plan = fm.plan_weights(source=fm.PackedSource(format=fm.PackedSourceFormat("fp4_e8m0_k32"), w13_layout=fm.W13Layout("w31")),
        activation=fm.ActivationSpec(mode=fm.ActivationMode.A8, nonlinearity="silu", io_dtype=torch.bfloat16, swiglu_limit=LIMIT, swiglu_alpha=None, swiglu_beta=None),
        geometry=fm.MoEGeometry(num_experts=len(cold), hidden_size=H, intermediate_size=I))
    ex = fm.prepare_weights(plan=plan, weights=fm.PackedWeights(w13=W["w13"], w2=W["w2"],
        w13_block_scales=W["s13"].clamp_(max=247).view(torch.float8_e8m0fnu), w2_block_scales=W["s2"].clamp_(max=247).view(torch.float8_e8m0fnu),
        w13_global_scales=ones, w2_global_scales=ones, input_scale=one, intermediate_scale=one, immutable_input_scales=True))
    torch.cuda.synchronize(); m2 = torch.cuda.memory_allocated()
    ts = []; walk(ex, set(), ts)
    ptrs = {t.untyped_storage().data_ptr(): t for t in ts if t.is_cuda}
    print(f"raw {(m1-m0)/1e9:.2f} GB, after prepare +{(m2-m1)/1e9:.2f} GB; prepared tensors:")
    for t in ts:
        if t.is_cuda and t.numel() * t.element_size() > 1e6:
            print(f"  {tuple(t.shape)} {t.dtype} {t.numel()*t.element_size()/1e9:.3f} GB")
    for k, v in W.items():
        print(f"  W[{k}] storage shared with prepared: {v.untyped_storage().data_ptr() in ptrs}")
    import os
    from dataclasses import replace
    sys.path.insert(0, "/home/jasonc/b12x-exp")
    from benchmarks.moe_preparation import prepared_call, request_for_capacity
    from b12x.preparation import PreparationSession
    BUCKETS = [int(v) for v in os.environ.get("BUCKETS", "1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192").split()]
    K = 6
    a = torch.zeros(max(BUCKETS), H, dtype=torch.bfloat16, device=dev)
    ids = torch.full((max(BUCKETS), K), -1, dtype=torch.int32, device=dev); ids[:, 0] = 0
    w = torch.zeros(max(BUCKETS), K, device=dev)
    out = torch.zeros(max(BUCKETS), H, dtype=torch.bfloat16, device=dev)
    torch.cuda.synchronize(); m3 = torch.cuda.memory_allocated()
    decl = fm.plan_execution(experts=ex, capacity=fm.ExecutionCapacity(max_tokens=max(BUCKETS), top_k=K, warmup_token_counts=tuple(BUCKETS), route_num_experts=0))
    torch.cuda.synchronize(); m4 = torch.cuda.memory_allocated()
    calls = {m: prepared_call(output=out[:m], bind=lambda st, scratch, m=m: st.bind(scratch=scratch, a=a[:m], experts=ex, topk_ids=ids[:m], topk_weights=w[:m], output=out[:m], input_scales_static=True)) for m in decl.token_counts}
    def race(st, m):
        call = calls[m](st); src = a[:m].clone()
        return replace(call, produce=lambda: a[:m].copy_(src), owners=(*call.owners, src))
    session = PreparationSession(device=dev)
    session.prepare((request_for_capacity(decl, name="x", calls=calls, benchmark_calls={m: (lambda st, m=m: race(st, m)) for m in calls}),))
    torch.cuda.synchronize(); m5 = torch.cuda.memory_allocated()
    del session, calls, race; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); m6 = torch.cuda.memory_allocated()
    print(f"plan_execution +{(m4-m3)/1e9:.3f} GB, prepare +{(m5-m4)/1e9:.3f} GB, after dropping session {(m6-m3)/1e9:.3f} GB over buffers")
    fm.clear_caches(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    print(f"after clear_caches {(torch.cuda.memory_allocated()-m3)/1e9:.3f} GB over buffers")
    from b12x.preparation.types import require_prepared
    import time
    for m in (64, 8192):
        v = decl.variants[m]; st = require_prepared(v, v.component_id)
        scratch = tuple(torch.empty(x.shape, dtype=x.dtype, device=x.device) for x in st.scratch.scratch_specs())
        b = fm.bind(v, scratch=scratch, a=a[:m], experts=ex, topk_ids=ids[:m], topk_weights=w[:m], output=out[:m], input_scales_static=True)
        t0 = time.time(); fm.run(binding=b); torch.cuda.synchronize()
        print(f"  run m={m} after clear_caches ok in {time.time()-t0:.2f}s; scratch {sum(x.numel()*x.element_size() for x in scratch)/1e6:.1f} MB")
    ts = []; walk(decl, set(), ts)
    big = sorted([t for t in ts if t.is_cuda], key=lambda t: -t.numel()*t.element_size())[:8]
    for t in big: print(f"  decl holds {tuple(t.shape)} {t.dtype} {t.numel()*t.element_size()/1e6:.1f} MB")

if __name__ == "__main__":
    main()
