import json, random, sys, time, urllib.request
BASE = "http://127.0.0.1:30006"
def post(path, body=None, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else b"",
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()
def run(n, seed):
    rng = random.Random(seed)
    ids = [rng.randrange(1000, 150000) for _ in range(n)]
    t = time.perf_counter()
    post("/v1/completions", {"model": "qwen38-flash-next", "prompt": ids, "max_tokens": 1, "temperature": 0})
    return time.perf_counter() - t
for n in (1000, 2000, 5000):  # warm
    run(n, n)
lens = [int(x) for x in sys.argv[1].split(",")]
post("/start_profile")
time.sleep(1)
for i, n in enumerate(lens):
    print(n, f"{run(n, 100 + i) * 1e3:.1f} ms", flush=True)
    time.sleep(0.3)
post("/stop_profile", timeout=600)
