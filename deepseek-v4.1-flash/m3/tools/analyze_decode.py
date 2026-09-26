"""Split decode GPU time per engine step: waiting on the 6000, 6000-path bookkeeping, MegaMoE, attention, etc.

analyze_decode.py <trace.json.gz> [...]. Steps = MegaMoE (285-expert, main model) launches / 40 layers.
"""
import collections
import gzip
import json
import re
import sys

BUCKETS = [
    ("peer wait (GB300 idle on 6000)", r"^_wait$"),
    ("peer bookkeeping", r"^_pack$|^_publish$|^_scatter_add$|MXFP8Quantize|cumsum|scan_kernel|DeviceScan|reduce_kernel.*any"),
    ("route counter", r"index_add|indexFuncLargeIndex|indexFuncSmallIndex"),
    ("MegaMoE (hot experts)", r"mega_moe_impl"),
    ("router gate", r"mega_gate|topk|TopK|routing"),
    ("attention + indexer", r"flash|mla|attn|Attn|mqa_logits|indexer|sparse|topKPerRow|deepselect"),
    ("mHC", r"mhc|hc_prenorm"),
    ("dense GEMM", r"gemm|Gemm|GEMM|nvjet|cutlass|einsum"),
    ("elementwise/other", r"."),
]


def main():
    for path in sys.argv[1:]:
        t = json.load(gzip.open(path))
        k = [e for e in t["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "kernel"]
        main_stream = collections.Counter(e.get("args", {}).get("stream") for e in k).most_common(1)[0][0]
        mm = [e for e in k if "mega_moe_impl" in e["name"] and "285u" in e["name"]]
        steps = max(1, len(mm) // 40)
        span = (max(e["ts"] + e["dur"] for e in k) - min(e["ts"] for e in k)) / 1e3
        by = collections.defaultdict(float)
        for e in k:
            if e.get("args", {}).get("stream") != main_stream:
                continue
            for name, pat in BUCKETS:
                if re.search(pat, e["name"]):
                    by[name] += e["dur"] / 1e3
                    break
        busy = sum(by.values())
        print(f"{path.split('/')[-1]}: {steps} steps, {span / steps:.2f} ms/step span, main-stream busy {busy / steps:.2f} ms/step")
        for name, _ in BUCKETS:
            print(f"   {name:34s} {by[name] / steps:7.3f} ms/step  {100 * by[name] / max(busy, 1e-9):5.1f}%")


if __name__ == "__main__":
    main()
