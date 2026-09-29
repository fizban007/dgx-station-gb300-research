"""Triton split-K MXFP8 GEMM vs FlashInfer cute-dsl / cutedsl_low_latency at DS-V4.1 decode shapes (L2-cold, graphs)."""
import itertools
import json
import sys

import torch
from flashinfer import mm_mxfp8, mxfp8_quantize
from flashinfer.autotuner import autotune

sys.path.insert(0, "/w")
import mxfp8_splitk as sk  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
SHAPES = [("wq_a+wkv", 1792, 5120), ("wq_b", 32768, 1280), ("wo_b", 5120, 8192)]
MS = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "1,6,12,24,32,48,64").split(",")]
L2_BYTES = 600 << 20


def timed(fn, iters=20):
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
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3


for name, N, K in SHAPES:
    copies = max(2, -(-L2_BYTES // (N * K)))
    ws = []
    for _ in range(copies):
        w = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02
        wq, wsf = mxfp8_quantize(w, is_sf_swizzled_layout=True)
        ws.append((wq.contiguous(), wsf.contiguous()))
        del w
    for M in MS:
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        xq, xsf = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        ref_out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        row = {"shape": name, "M": M}
        with autotune(True):
            mm_mxfp8(xq, ws[0][0].t(), xsf, ws[0][1], out=ref_out, out_dtype=torch.bfloat16, backend="cute-dsl")
        ref = mm_mxfp8(xq, ws[0][0].t(), xsf, ws[0][1], out_dtype=torch.bfloat16, backend="cute-dsl").float()
        row["cute-dsl"] = round(timed(lambda: [mm_mxfp8(xq, ws[i][0].t(), xsf, ws[i][1], out=ref_out,
                                                         out_dtype=torch.bfloat16, backend="cute-dsl")
                                                for i in range(copies)]) / copies, 2)
        if M <= 8:
            with autotune(True):
                mm_mxfp8(xq, ws[0][0].t(), xsf, ws[0][1], out=ref_out, out_dtype=torch.bfloat16,
                         backend="cutedsl_low_latency")
            row["ll"] = round(timed(lambda: [mm_mxfp8(xq, ws[i][0].t(), xsf, ws[i][1], out=ref_out,
                                                       out_dtype=torch.bfloat16, backend="cutedsl_low_latency")
                                              for i in range(copies)]) / copies, 2)
        best = None
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        for split, bk, stages, warps in itertools.product((1, 2, 4, 8, 16), (128, 256), (3, 4), (4, 8)):
            if (K // split) % bk or K // split < bk:
                continue
            try:
                sk.mxfp8_gemm(xq, xsf, ws[0][0], ws[0][1], N, K, out=out, split_k=split, block_k=bk,
                              num_stages=stages, num_warps=warps)
                torch.cuda.synchronize()
                o = out.float()
                cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
                rel = ((o - ref).abs().max() / ref.abs().max()).item()
                us = timed(lambda: [sk.mxfp8_gemm(xq, xsf, ws[i][0], ws[i][1], N, K, out=out, split_k=split,
                                                  block_k=bk, num_stages=stages, num_warps=warps)
                                    for i in range(copies)]) / copies
                if cos < 0.9999:
                    row.setdefault("bad", []).append([split, bk, stages, warps, round(cos, 5)])
                    continue
                if best is None or us < best[0]:
                    best = (us, split, bk, stages, warps, cos, rel)
            except Exception as exc:  # noqa: BLE001
                row.setdefault("err", []).append(f"{split}/{bk}/{stages}/{warps}: {type(exc).__name__}: {str(exc)[:150]}")
        if best:
            row["splitk"] = round(best[0], 2)
            row["splitk_TBps"] = round(N * K / (best[0] * 1e-6) / 1e12, 2)
            row["cfg(split,bk,stages,warps)"] = best[1:5]
            row["cos"], row["maxrel"] = round(best[5], 6), round(best[6], 5)
        if "err" in row:
            row["err"] = row["err"][:3]
        print(json.dumps(row), flush=True)
    del ws
    torch.cuda.empty_cache()
