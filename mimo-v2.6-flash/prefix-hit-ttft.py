"""Prefix-cache hit TTFT at several prompt lengths: for each length, one cold request (unique random prompt) then
REPS streamed repeats of the same prompt; prints cold ms and hit TTFT median/min, plus hit tokens from /metrics.
Usage: prefix-hit-ttft.py [lengths_in_k=8,32,128] [reps=5]"""
import json, os, random, re, statistics, sys, time, urllib.request
PORT = os.environ.get("PORT", "30006"); MODEL = os.environ.get("MODEL_NAME", "mimo-v26-flash")
lens = [int(x) * 1000 for x in (sys.argv[1] if len(sys.argv) > 1 else "8,32,128").split(",")]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
words = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()
def hits():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    return sum(float(x) for x in re.findall(r"^vllm:prefix_cache_hits_total\{[^}]*\} ([0-9.e+]+)", txt, re.M))
def ttft(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 2, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if line.startswith(b"data: ") and b'"text"' in line:
                return (time.perf_counter() - t) * 1000
    return float("nan")
random.seed(time.time_ns())
for n in lens:
    # ~1.07 tokens per word for this vocabulary; exact length is not important here.
    prompt = f"[{random.random()}] " + " ".join(random.choice(words) for _ in range(int(n / 1.07)))
    cold = ttft(prompt)
    h0 = hits(); xs = [ttft(prompt) for _ in range(reps)]; h1 = hits()
    print(f"len~{n//1000}K cold={cold:.0f}ms hit_median={statistics.median(xs):.0f}ms hit_min={min(xs):.0f}ms "
          f"hit_tokens/req={(h1 - h0) / reps:.0f}", flush=True)
