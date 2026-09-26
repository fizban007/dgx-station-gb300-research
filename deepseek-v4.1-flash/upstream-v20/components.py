"""Bucket GPU kernel time from a torch-profiler trace into DeepSeek-V4.1 decode components.

Usage: components.py <trace.json.gz> [--step-regex REGEX]
Reports ms per decode step and share of kernel time for each component, plus
the top unmatched kernels so the patterns can be extended.
"""
import argparse
import collections
import gzip
import json
import re

BUCKETS = [
    ("moe", r"bmm_.*E2m1|moe::dev|fused_moe|routingIndices|finalizeKernel|trtllm_fp4|_moe_|MoE"),
    ("mhc", r"mhc|MHC|hyperconnection|HyperConnection|hc_prenorm"),
    ("indexer", r"indexer|Indexer|_dsv4_topk|PagedScore|SortPositions|deepselect|DeepSelect"),
    ("attention", r"flash_?mla|FlashMLA|flash_fwd|mla_metadata|mla_combine|sparse_?mla|SparseMLA|sharedmla|mla_warp|UnifiedDecode|SplitDecode|attention|attn"),
    ("engram", r"engram|Engram"),
    ("norm_quant_rope", r"rms|Rms|norm|Norm|quant|Quant|MXFP8|mxfp8|rotary|Rotate|rope|Rope"),
    ("dense_gemm", r"dense_blockscaled|DenseBlockscaledGemm|SplitK|splitk|deep_gemm|fp8_gemm|Fp8Gemm|gemm|Gemm|GEMM|gemv|Gemv|nvjet|cublas|sm100_|sm90_"),
    ("act", r"act_and_mul|silu"),
    ("elementwise", r"elementwise|vectorized|unrolled|Fill|copy|Copy|index|reduce|Reduce|cat|softmax"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--step-regex", default=r"^execute_")
    p.add_argument("--tokens", type=int, default=0, help="generated tokens in the window; report per token")
    a = p.parse_args()
    t = json.load(gzip.open(a.trace))
    ev = [e for e in t["traceEvents"] if e.get("ph") == "X"]
    kern = [e for e in ev if e.get("cat") == "kernel"]
    steps = [e for e in ev if re.search(a.step_regex, e.get("name", "")) and e.get("cat") != "kernel"]
    total = sum(e["dur"] for e in kern)
    span = max(e["ts"] + e["dur"] for e in kern) - min(e["ts"] for e in kern)
    by = collections.defaultdict(float)
    unmatched = collections.defaultdict(lambda: [0, 0.0])
    for e in kern:
        for name, pattern in BUCKETS:
            if re.search(pattern, e["name"]):
                by[name] += e["dur"]
                break
        else:
            by["other"] += e["dur"]
            u = unmatched[e["name"][:90]]
            u[0] += 1
            u[1] += e["dur"]
    n = a.tokens or max(1, len(steps))
    unit = 'token' if a.tokens else 'step'
    print(f"steps {len(steps)}  kernel busy {total / 1e3:.1f} ms over span {span / 1e3:.1f} ms "
          f"({100 * total / span:.0f}% busy)")
    for name, _ in BUCKETS + [("other", "")]:
        print(f"  {name:16s} {by[name] / 1e3 / n:8.3f} ms/{unit}  {100 * by[name] / total:5.1f}%")
    print("  top unmatched:")
    for name, (c, d) in sorted(unmatched.items(), key=lambda x: -x[1][1])[:8]:
        print(f"    {d / 1e3:8.2f} ms {c:5d}x  {name}")


if __name__ == "__main__":
    main()
