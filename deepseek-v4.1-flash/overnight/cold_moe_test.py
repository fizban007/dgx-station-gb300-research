"""Check the Triton cold-route MoE against a dense PyTorch reference; time it with Grace-resident weights."""
import argparse, statistics, sys, torch
sys.path.insert(0, "/home/jasonc/vllm-karmic/vllm/model_executor/layers/fused_moe/experts")
from cold_moe_triton import ColdExperts
p = argparse.ArgumentParser(); p.add_argument("--host", action="store_true"); p.add_argument("--experts", type=int, default=8)
p.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 32]); a = p.parse_args()
dev = torch.device("cuda", 0); g = torch.Generator(device=dev).manual_seed(0)
E, H, I, K, L = a.experts, 5120, 2304, 6, 10.0
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)
def rand_mx(n, k):
    return (torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g),
            torch.randint(118, 124, (E, n, k // 32), dtype=torch.uint8, device=dev, generator=g))
def dequant(w, s):
    lo, hi = LUT[(w & 15).long()], LUT[(w >> 4).long()]
    v = torch.stack((lo, hi), -1).flatten(-2)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, -1)
w13, s13 = rand_mx(2 * I, H); w2, s2 = rand_mx(H, I)
if a.host:
    sys.path.insert(0, "/home/jasonc/b12x-exp")
    from b12x.sequence._shared.disk_table import MappedHostAllocation
    owners = []
    def to_host(t):
        o = MappedHostAllocation(tuple(t.shape), t.dtype, dev); o.host_view.copy_(t.cpu()); owners.append(o); return o.device_view
    w13, s13, w2, s2 = (to_host(t) for t in (w13, s13, w2, s2))
cold = ColdExperts(w13, s13, w2, s2, cold_start=100, limit=L, gate_first=True)
d13, d2 = dequant(w13, s13), dequant(w2, s2)
for T in a.tokens:
    x = torch.randn(T, H, device=dev, generator=g)
    xs = torch.randint(122, 128, (T, H // 32), dtype=torch.uint8, device=dev, generator=g)
    xq = (x / torch.exp2(xs.float() - 127).repeat_interleave(32, -1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    xf = xq.float() * torch.exp2(xs.float() - 127).repeat_interleave(32, -1)
    ids = torch.randint(0, 100 + E, (T, K), device=dev, generator=g, dtype=torch.int32)
    ids[ids < 100] = torch.randint(0, 100, (int((ids < 100).sum()),), device=dev, generator=g, dtype=torch.int32)
    wts = torch.rand(T, K, device=dev, generator=g).to(torch.bfloat16)
    out = torch.zeros(T, H, device=dev, dtype=torch.bfloat16)
    cold.add_to(out, xq, xs, ids, wts)
    ref = torch.zeros(T, H, device=dev)
    for t in range(T):
        for k in range(K):
            e = int(ids[t, k]) - 100
            if e < 0: continue
            h = d13[e] @ xf[t]; gate, up = h[:I].clamp(max=L), h[I:].clamp(-L, L)
            ref[t] += float(wts[t, k]) * (d2[e] @ (gate * torch.sigmoid(gate) * up))
    err = float((out.float() - ref).norm() / ref.norm().clamp_min(1e-9))
    cold_routes = int((ids >= 100).sum())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cold.add_to(out, xq, xs, ids, wts)
    times = []
    for _ in range(20):
        s, e_ = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); graph.replay(); e_.record(); e_.synchronize(); times.append(s.elapsed_time(e_) * 1000)
    us = statistics.median(times)
    uniq = int(torch.unique(ids[ids >= 100]).numel())
    print(f"T={T:3d} cold routes {cold_routes:3d} unique {uniq:3d}: rel err {err:.2e}, {us:8.1f} us, {us/max(uniq,1):6.1f} us/unique expert", flush=True)
