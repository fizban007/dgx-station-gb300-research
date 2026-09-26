"""b12x mHC boundary (run_post_pre, fused RMSNorm, default tuned config), CUDA-graph timed."""
import json, statistics, sys, torch
from b12x.norm import mhc
from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall
if __name__ == "__main__":
    H, HC, dev = 5120, 4, torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    fn = torch.randn(HC * (HC + 2), HC * H, device=dev, generator=g) * 0.01
    scale = torch.ones(3, device=dev); base = torch.randn(HC * (HC + 2), device=dev, generator=g) * 0.1
    w = torch.ones(H, device=dev, dtype=torch.bfloat16)
    inv = FrozenMapping({"operation": "post_pre", "has_norm_weight": True, "expanded_residual": False,
                         "norm_eps": 1e-20, "rms_eps": 1e-20, "hc_eps": 1e-6, "sinkhorn_iters": 20})
    rows = []
    def main():
      pass
    for M in [int(v) for v in (sys.argv[1:] or (1, 4, 8, 16, 32, 64))]:
        residual = torch.randn(M, HC, H, device=dev, generator=g).bfloat16()
        x = torch.randn(M, H, device=dev, generator=g).bfloat16()
        prev_post = torch.full((M, HC), 0.25, device=dev)
        prev_comb = torch.eye(HC, device=dev).expand(M, HC, HC).contiguous()
        plan = mhc.plan(mhc.Caps(device=dev, max_tokens=M, hidden_size=H, split_k=H // 64), invocation=inv)
        scratch = tuple(torch.empty(s.shape, dtype=s.dtype, device=dev) for s in plan.scratch_specs())
        y, out = torch.empty_like(x), torch.empty_like(residual)
        post, comb = torch.empty_like(prev_post), torch.empty_like(prev_comb)
        from b12x.norm.mhc import _impl
        def factory(state):
            sc = [torch.empty(sp.shape, dtype=sp.dtype, device=dev) for sp in state.scratch_specs()]
            b = state.bind(scratch=sc, tokens=M, out=torch.empty_like(residual), y=torch.empty_like(x),
                           post=torch.empty_like(prev_post), comb=torch.empty_like(prev_comb))
            return PreparedCall(run=lambda: _impl._b12x_mhc_post_pre_impl(
                x, residual, prev_post, prev_comb, fn, scale, base, rms_eps=1e-20, hc_eps=1e-6,
                sinkhorn_iters=20, norm_weight=w, norm_eps=1e-20, binding=b, _state=state), produce=lambda: (residual.normal_(), x.normal_()), owners=(b, *sc))
        session = PreparationSession(device=dev, autotune=True, compile_workers=16)
        session.__enter__()
        session.prepare((plan.request(name=f"mhc.m{M}", prepare_call=factory, benchmark_call=factory),))
        binding = mhc.bind(plan, scratch=scratch, y=y, out=out, post=post, comb=comb)
        def step():
            return mhc.run_post_pre(x, residual, prev_post, prev_comb, fn, scale, base, binding=binding,
                                    norm_weight=w, norm_eps=1e-20, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20)
        for _ in range(3): step()
        torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20): step()
        t = []
        for _ in range(30):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); graph.replay(); b.record(); b.synchronize(); t.append(a.elapsed_time(b) * 1000 / 20)
        rows.append({"tokens": M, "us": round(statistics.median(t), 2)}); print(json.dumps(rows[-1]), flush=True)
