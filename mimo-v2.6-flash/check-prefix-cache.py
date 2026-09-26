"""Prefix-cache sanity check: send one ~32K-token prompt twice (max_tokens=1) and compare latency plus the
server's vllm:prefix_cache_{queries,hits} counters. A working cache makes the second request far faster."""
import json, os, random, re, time, urllib.request
PORT = os.environ.get("PORT", "30006"); MODEL = os.environ.get("MODEL_NAME", "mimo-v26-flash")
def metrics():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    out = {}
    for k in ("prefix_cache_queries", "prefix_cache_hits"):
        m = re.findall(rf"^vllm:{k}_total\{{[^}}]*\}} ([0-9.e+]+)", txt, re.M)
        out[k] = sum(float(x) for x in m) if m else None
    return out
random.seed(time.time_ns())
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
prompt = f"[{random.random()}] " + " ".join(random.choice(words) for _ in range(30000)) + "\nSummarize in one word."
def once():
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter(); r = json.load(urllib.request.urlopen(req, timeout=600)); dt = time.perf_counter() - t
    return dt, r["usage"]["prompt_tokens"]
m0 = metrics(); a, n = once(); m1 = metrics(); b, _ = once(); m2 = metrics()
d = lambda x, y, k: None if x[k] is None else y[k] - x[k]
print(f"prompt_tokens={n} first={a*1000:.0f}ms second={b*1000:.0f}ms speedup={a/b:.1f}x")
print(f"first: queries={d(m0,m1,'prefix_cache_queries')} hits={d(m0,m1,'prefix_cache_hits')}; "
      f"second: queries={d(m1,m2,'prefix_cache_queries')} hits={d(m1,m2,'prefix_cache_hits')}")
