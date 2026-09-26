"""Per-step GPU busy time and idle gaps from a vLLM torch-profiler trace (the prefill profile of prof_prefill.py).
Usage: prof_gaps.py <dp0_pp0_*.pt.trace.json.gz>"""
import gzip, json, sys
ev = json.load(gzip.open(sys.argv[1]))["traceEvents"]
steps = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation"
         and str(e.get("name", "")).startswith("execute_context_1") and e["dur"] > 5000]
kern = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
              key=lambda e: e["ts"])
print("step (tokens)                            GPU span ms  kernels  busy ms  busy %  idle gaps ms")
for s in steps:
    t0, t1 = s["ts"], s["ts"] + s["dur"]
    ks = [k for k in kern if t0 <= k["ts"] < t1]
    busy = sum(k["dur"] for k in ks)
    gaps = sum(max(0, b["ts"] - (a["ts"] + a["dur"])) for a, b in zip(ks, ks[1:]))
    print(f"{s['name']:40s} {s['dur'] / 1e3:11.1f}  {len(ks):7d}  {busy / 1e3:7.1f}  {100 * busy / s['dur']:5.0f}%  {gaps / 1e3:12.1f}")
