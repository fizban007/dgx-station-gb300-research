"""Profile a short C1 decode window on :30007 via /start_profile + /stop_profile, then summarise the trace.
usage: python3 prof_decode.py TAG   (server must run with --profiler-config profiler=torch, dir /prof -> host ${WORKDIR:-$PWD}/prof)"""
import json, sys, time, urllib.request, glob, gzip, os, collections
WD = os.environ.get("WORKDIR", os.getcwd())
TAG = sys.argv[1]
H = "http://127.0.0.1:30007"
def post(path, body=None):
    return urllib.request.urlopen(urllib.request.Request(H + path, data=json.dumps(body or {}).encode(),
                                  headers={"Content-Type": "application/json"}), timeout=900).read()
def gen(n):
    b = {"model": "mimo26-pro", "messages": [{"role": "user", "content": "You have a tool run_shell(cmd). List the docker containers, then show disk usage of /models. Respond with the tool calls as JSON only."}],
         "max_tokens": n, "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time(); r = json.loads(post("/v1/chat/completions", b)); return r["usage"]["completion_tokens"], time.time() - t
gen(16)
before = set(glob.glob(os.path.join(WD, "prof/**/*"), recursive=True))
post("/start_profile"); n, dt = gen(64); post("/stop_profile")
print(f"[{TAG}] profiled request: {n} tokens in {dt:.2f}s ({n/dt:.1f} tok/s incl. profiler overhead)", flush=True)
time.sleep(20)
new = [p for p in set(glob.glob(os.path.join(WD, "prof/**/*"), recursive=True)) - before if os.path.isfile(p)]
print("new files:", [os.path.basename(p) + f" {os.path.getsize(p)//1024}KiB" for p in new], flush=True)
traces = [p for p in new if p.endswith(".json.gz") or p.endswith(".json")]
if not traces:
    sys.exit("no trace")
tr = max(traces, key=os.path.getsize)
d = json.load(gzip.open(tr) if tr.endswith(".gz") else open(tr))
ev = d["traceEvents"] if isinstance(d, dict) else d
k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not k:
    sys.exit("no kernel events in " + tr)
t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
busy = sum(e["dur"] for e in k)
by = collections.Counter(); cnt = collections.Counter()
def bucket(name):
    n = name.lower()
    if "marlin" in n: return "moe_marlin (routed experts)"
    if "moe_align" in n or "moe_sum" in n or "topk" in n or "sort" in n or "count_and_sort" in n: return "moe routing/align/sum"
    if "flash" in n or "fmha" in n or "attn" in n or "attention" in n: return "attention"
    if "gemm" in n or "cutlass" in n or "sm90" in n or "sm100" in n or "nvjet" in n or "matmul" in n or "fp8" in n: return "dense gemm (qkv/o/shared/lm_head)"
    if "norm" in n: return "norms"
    if "rotary" in n or "rope" in n: return "rope"
    if "memcpy" in n or "memset" in n: return "memcpy/memset"
    return "other"
for e in k:
    b = bucket(e["name"]); by[b] += e["dur"]; cnt[b] += 1
span = t1 - t0
print(f"[{TAG}] trace {os.path.basename(tr)}: kernels {len(k)}, GPU span {span/1e3:.1f} ms, GPU busy {busy/1e3:.1f} ms ({100*busy/span:.0f}% of span)")
steps = n
for b, v in by.most_common():
    print(f"  {b:36s} {v/1e3:8.1f} ms  {100*v/busy:5.1f}% busy  {cnt[b]/steps:6.0f} launches/token  {v/steps/1e3:6.2f} ms/token")
print(f"  per token: span {span/steps/1e3:.2f} ms, busy {busy/steps/1e3:.2f} ms, idle {(span-busy)/steps/1e3:.2f} ms")
top = collections.Counter()
for e in k: top[e["name"][:90]] += e["dur"]
print("  top kernels:")
for nme, v in top.most_common(8): print(f"    {v/1e3:8.1f} ms  {nme}")
