"""Teacher-forced prompt logprobs on real text; save, or compare against a saved run (noise floor / A-B).
Every request is cold: /reset_prefix_cache when the server allows it (VLLM_SERVER_DEV_MODE=1), and a unique
cache_salt per request, so prompts that share a prefix never hit each other's cache."""
import glob, json, sys, urllib.error, urllib.request, uuid
BASE = "http://127.0.0.1:30006"
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())
text = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/home/jasonc/b12x/b12x/**/*.py", recursive=True))[:400])
ids = post("/tokenize", {"model": "qwen38-flash-next", "prompt": text})["tokens"]
res = {}
for n in (1900, 3000, 6000, 8192, 16384):
    for off in (0, 30000):
        try:
            urllib.request.urlopen(urllib.request.Request(BASE + "/reset_prefix_cache", data=b"", method="POST")).read()
        except urllib.error.HTTPError:
            pass
        p = ids[off:off + n]
        r = post("/v1/completions", {"model": "qwen38-flash-next", "prompt": p, "max_tokens": 1, "temperature": 0,
                                     "prompt_logprobs": 0, "cache_salt": uuid.uuid4().hex})["choices"][0]["prompt_logprobs"]
        res[f"{n}@{off}"] = [None if d is None else d[str(t)]["logprob"] for d, t in zip(r, p)]
mode, path = sys.argv[1], sys.argv[2]
if mode == "save":
    json.dump(res, open(path, "w")); print("saved")
else:
    base = json.load(open(path))
    for k, v in res.items():
        a = [x for x in v[1:]]; b = base[k][1:]
        d = sorted(abs(x - y) for x, y in zip(a, b))
        nll = lambda z: -sum(z) / len(z)
        print(f"{k:>11}: NLL {nll(b):.4f} -> {nll(a):.4f}  |dlogprob| mean {sum(d)/len(d):.4f} p99 {d[int(.99*len(d))]:.4f} max {d[-1]:.3f}")
