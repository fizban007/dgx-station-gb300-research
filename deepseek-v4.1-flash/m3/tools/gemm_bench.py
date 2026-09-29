"""MXFP8 dense GEMM backends at DS-V4.1-Flash decode shapes on the GB300 (CUDA-graph timed, L2-cold weights)."""
import json
import sys

import torch
from flashinfer import mm_mxfp8, mxfp8_quantize
from flashinfer.autotuner import autotune

torch.manual_seed(0)
dev = "cuda"
# (name, N, K): wq_a+wkv fused, wq_b, wo_b, DSpark fc (3 target layers -> hidden), DSpark 25600x6144
SHAPES = [("wq_a+wkv", 1792, 5120), ("wq_b", 32768, 1280), ("wo_b", 5120, 8192), ("dspark_fc", 5120, 15360)]
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "1,2,4,6,8,12,16,24,32,48,64").split(",")]
BACKENDS = (sys.argv[2] if len(sys.argv) > 2 else "cute-dsl,cutedsl_low_latency,cutlass,cudnn").split(",")
L2_BYTES = 600 << 20  # rotate enough weight copies that none stays in L2


def bench_graph(fn, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1e3  # us per graph


results = []
for name, N, K in SHAPES:
    wbytes = N * K
    copies = max(2, -(-L2_BYTES // wbytes))
    ws = []
    for _ in range(copies):
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        wq, wsf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
        ws.append((wq.contiguous().t(), wsf.contiguous()))  # [K, N] column-major, as vLLM's cute-dsl kernel stores it
        del w
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        xq, xsf = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        ref = None
        row = {"shape": name, "N": N, "K": K, "M": M}
        for be in BACKENDS:
            if be == "cutedsl_low_latency" and M > 8:
                continue
            try:
                def call(i=0, be=be):
                    b, bsf = ws[i]
                    return mm_mxfp8(xq, b, xsf, bsf, out=out, out_dtype=torch.bfloat16, backend=be)
                with autotune(True):
                    call()
                torch.cuda.synchronize()
                o = call().float().clone()
                if ref is None:
                    ref = o
                cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()

                def run_all(be=be):
                    for i in range(copies):
                        call(i, be)
                us = bench_graph(run_all) / copies
                row[be] = round(us, 2)
                row[be + "_TBps"] = round(wbytes / (us * 1e-6) / 1e12, 2)
                if cos < 0.999:
                    row[be + "_cos"] = round(cos, 5)
            except Exception as exc:  # noqa: BLE001
                row[be] = f"ERR {type(exc).__name__}: {str(exc)[:120]}"
        print(json.dumps(row), flush=True)
        results.append(row)
    del ws
    torch.cuda.empty_cache()

# HBM read reference: sum over a large fp8 buffer
buf = torch.empty(2 << 30, dtype=torch.uint8, device=dev)
v = buf.view(torch.int64)
us = bench_graph(lambda: v.sum(), iters=10)
print(json.dumps({"hbm_read_ref_TBps": round((2 << 30) / (us * 1e-6) / 1e12, 2)}), flush=True)
