"""Upstream vLLM 2671fedf mHC boundary (TileLang post + pre_delayed with fused RMSNorm), CUDA-graph timed."""
import json, statistics, torch
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang, mhc_pre_delayed_tilelang
H, HC, dev = 5120, 4, torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
fn = torch.randn(HC * (HC + 2), HC * H, device=dev, generator=g) * 0.01
scale = torch.ones(3, device=dev); base = torch.randn(HC * (HC + 2), device=dev, generator=g) * 0.1
w = torch.ones(H, device=dev, dtype=torch.bfloat16)
rows = []
for M in (1, 4, 8, 16, 32, 64):
    residual = torch.randn(M, HC, H, device=dev, generator=g).bfloat16()
    x = torch.randn(M, H, device=dev, generator=g).bfloat16()
    post, res, _, pre = mhc_pre_delayed_tilelang(residual, fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20, norm_weight=w, norm_eps=1e-20)
    def step():
        r2 = mhc_post_tilelang(x, residual, post, res)
        return mhc_pre_delayed_tilelang(r2, fn, scale, base, 1e-20, 1e-6, 1e-6, 2.0, 20, pre_mix=pre, norm_weight=w, norm_eps=1e-20)
    for _ in range(3): step()
    torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20): step()
    t = []
    for _ in range(30):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize(); t.append(a.elapsed_time(b) * 1000 / 20)
    rows.append({"tokens": M, "us": round(statistics.median(t), 2)}); print(json.dumps(rows[-1]), flush=True)
