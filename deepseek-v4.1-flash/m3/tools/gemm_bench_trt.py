"""TRT-LLM MXFP8 GEMM (shuffled weights, 8x4 activation scales) vs cute-dsl at DS-V4.1 decode shapes."""
import json
import sys

import torch
from flashinfer import SfLayout, mm_mxfp8, mxfp8_quantize, shuffle_matrix_a, shuffle_matrix_sf_a
from flashinfer.autotuner import autotune

torch.manual_seed(0)
dev = "cuda"
SHAPES = [("wq_a+wkv", 1792, 5120), ("wq_b", 32768, 1280), ("wo_b", 5120, 8192)]
MS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "6,12,24,32,48,64").split(",")]
L2_BYTES = 600 << 20


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
    return st.elapsed_time(en) / iters * 1e3


for name, N, K in SHAPES:
    copies = max(2, -(-L2_BYTES // (N * K)))
    cute, trt = [], []
    for _ in range(copies):
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        wq, wsf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
        cute.append((wq.contiguous().t(), wsf.contiguous()))
        wq_l, wsf_l = mxfp8_quantize(w, is_sf_swizzled_layout=False)
        wsf_l = wsf_l.view(torch.uint8).reshape(N, K // 32)
        trt.append((shuffle_matrix_a(wq_l, 128).reshape(N, K).t(),
                    shuffle_matrix_sf_a(wsf_l, 128, num_elts_per_sf=32).reshape(-1)))
        del w
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        xq, xsf = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        xq8, xsf8 = mxfp8_quantize(x, backend="cuda", sf_swizzle_layout=SfLayout.layout_8x4)
        row = {"shape": name, "M": M}
        arms = {
            "cute-dsl": lambda i: mm_mxfp8(xq, cute[i][0], xsf, cute[i][1], out_dtype=torch.bfloat16, backend="cute-dsl"),
            "trtllm": lambda i: mm_mxfp8(xq8, trt[i][0], xsf8, trt[i][1], out_dtype=torch.bfloat16, backend="trtllm",
                                         use_8x4_sf_layout=True),
        }
        ref = None
        for arm, call in arms.items():
            try:
                with autotune(True):
                    call(0)
                o = call(0).float()
                ref = o if ref is None else ref
                cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
                us = bench_graph(lambda: [call(i) for i in range(copies)]) / copies
                row[arm] = round(us, 2)
                row[arm + "_TBps"] = round(N * K / (us * 1e-6) / 1e12, 2)
                row[arm + "_cos"] = round(cos, 5)
            except Exception as exc:  # noqa: BLE001
                row[arm] = f"ERR {type(exc).__name__}: {str(exc)[:160]}"
        print(json.dumps(row), flush=True)
    del cute, trt
    torch.cuda.empty_cache()
