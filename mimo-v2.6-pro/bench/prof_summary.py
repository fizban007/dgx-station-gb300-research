"""Summarise a vLLM torch-profiler trace from the three-tier MiMo-Pro lane, per generated token.

Buckets GPU kernels by name. Marlin MoE kernels are split into the HBM bank and the Grace bank by order: hotsplit runs
the hot bank first, then the Grace bank, after each layer's peer publish. The peer tier's own kernels show what the
6000 costs the GB300: _pack/_publish (send), _wait (the spin while the sidecar computes: the stall), _scatter_add.
usage: prof_summary.py <trace.json[.gz]> <generated tokens in the window> [steps]
"""
import collections
import gzip
import json
import sys

path, tokens = sys.argv[1], int(sys.argv[2])
d = json.load(gzip.open(path) if path.endswith(".gz") else open(path))
ev = d["traceEvents"] if isinstance(d, dict) else d
k = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
           key=lambda e: e["ts"])
if not k:
    sys.exit(f"no kernel events in {path}")


def bucket(name: str) -> str:
    n = name.lower()
    if "_wait" in n and "peer" not in n and "cuda" not in n:
        return "peer: wait on 6000 (stall)"
    if n.startswith("_pack") or n.startswith("_publish"):
        return "peer: pack+publish"
    if "_scatter_add" in n:
        return "peer: scatter-add"
    if "marlin" in n:
        return "marlin"
    if "mxfp8" in n or "quantize" in n and "fp8" in n:
        return "mxfp8 quant (peer input)"
    if "moe_align" in n or "moe_sum" in n or "topk" in n or "sort" in n or "count_and_sort" in n or "grouped_topk" in n:
        return "moe routing/align/sum"
    if "flash" in n or "fmha" in n or "attn" in n or "attention" in n:
        return "attention"
    if "gemm" in n or "cutlass" in n or "nvjet" in n or "matmul" in n or "sm100" in n or "sm90" in n:
        return "dense gemm"
    if "norm" in n:
        return "norms"
    if "rotary" in n or "rope" in n:
        return "rope"
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    return "other"


by, cnt = collections.Counter(), collections.Counter()
# Each Marlin MoE call launches two GEMM kernels (w13, then w2): after a layer's publish, Marlin kernels 0-1 are the
# HBM bank and 2-3 the Grace bank.
n_marlin = 0
for e in k:
    b = bucket(e["name"])
    if e["name"].lower().startswith("_publish"):
        n_marlin = 0
    if b == "marlin":
        b = "marlin: HBM bank" if n_marlin < 2 else "marlin: Grace bank"
        n_marlin += 1
    by[b] += e["dur"]
    cnt[b] += 1
span = (k[-1]["ts"] + k[-1]["dur"] - k[0]["ts"]) / 1e3
busy = sum(by.values()) / 1e3
print(f"window {span:.1f} ms, GPU busy {busy:.1f} ms ({100 * busy / span:.0f}%), {tokens} tokens -> "
      f"{span / tokens:.2f} ms/token wall, {busy / tokens:.2f} ms/token busy")
for b, us in by.most_common():
    print(f"  {us / 1e3 / tokens:7.3f} ms/token  {100 * us / 1e3 / busy:5.1f}%  {cnt[b] // max(tokens, 1):6d} calls/token  {b}")
top = collections.Counter()
for e in k:
    top[e["name"][:110]] += e["dur"]
print("top kernels:")
for name, us in top.most_common(12):
    print(f"  {us / 1e3 / tokens:7.3f} ms/token  {name}")
