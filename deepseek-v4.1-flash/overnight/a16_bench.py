"""SM103 decode GEMM: b12x blockscaled A16 (split-K GEMV) vs the quantized MXFP8 dense path.

V4.1 dense projections store FP8 E4M3 weights with [32,32] UE8M0 blocks; as
MXFP8 (1x32 scales) they are the same values with scales replicated per row.
"""
import json
import statistics
import sys

import torch

sys.path.insert(0, "/home/jasonc/b12x-exp")
from tests.gemm.test_blockscaled_a16 import make_weight, prepared_execution, make_workspace  # noqa: E402
from b12x.gemm import blockscaled  # noqa: E402
from b12x.gemm.blockscaled._tuning import BlockscaledConfig  # noqa: E402

SHAPES = {  # name: (N, K)
    "wq_a": (1280, 5120), "wq_b": (32768, 1280), "wkv": (512, 5120),
    "wo_b": (5120, 8192), "shared_w13": (4608, 5120), "shared_w2": (5120, 2304),
}


def timed(fn, reps=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(10):
            fn()
    samples = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); graph.replay(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b) * 100)  # us per call (10 calls)
    return statistics.median(samples)


rows = []
for name, (n, k) in SHAPES.items():
    weight, decoded, _ = make_weight("mxfp8", n, k)
    for m in (1, 4, 8):
        source = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        row = {"name": name, "n": n, "k": k, "m": m, "weight_mb": n * k / 1e6}
        configs = {"quantized": ("quantized", None)}
        for split in (2, 4, 8):
            for tile_n in (64, 128):
                configs[f"a16_n{tile_n}_s{split}"] = ("a16", BlockscaledConfig(mode="a16", tile_n=tile_n, tile_k=64, split_k=split))
        for label, (mode, config) in configs.items():
            try:
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                ws = make_workspace(source, weight, activation_mode=mode, out=out, config=config, expected_m=m)
                with prepared_execution(source, weight, activation_mode=mode, out=out, workspace=ws,
                                        expected_m=m, config=config) as (_, plan):
                    fn = lambda: blockscaled.mm(source, weight, out=out, workspace=ws, plan=plan)
                    us = timed(fn)
                    err = float((out.float() - source.float() @ decoded.T).norm() / (source.float() @ decoded.T).norm())
                row[label] = round(us, 2)
                row[label + "_err"] = round(err, 5)
            except Exception as exc:  # record unsupported configs
                row[label] = f"{type(exc).__name__}: {str(exc)[:80]}"
        best = min((v, key) for key, v in row.items() if isinstance(v, float) and key.startswith("a16") and not key.endswith("_err"))
        row["best_a16"] = best[1]
        print(json.dumps({key: row[key] for key in ("name", "m", "quantized", "best_a16")} | {"best_us": best[0], "gbps": round(row["weight_mb"] * 1e3 / best[0], 0)}), flush=True)
        rows.append(row)
json.dump(rows, open("/home/jasonc/ds41f-exp/micro/a16_bench.json", "w"), indent=1)
