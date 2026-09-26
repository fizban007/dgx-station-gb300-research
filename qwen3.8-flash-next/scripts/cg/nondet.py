"""Run-to-run determinism: 3 cold runs of one prompt per length, first output token's top-5 logprobs.
Cold = /reset_prefix_cache when allowed (VLLM_SERVER_DEV_MODE=1) plus a unique cache_salt per request."""
import glob, json, urllib.error, urllib.request, uuid
BASE = "http://127.0.0.1:30006"
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
text = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/home/jasonc/b12x/b12x/**/*.py", recursive=True))[:400])
ids = post("/tokenize", {"model": "qwen38-flash-next", "prompt": text})["tokens"]
for n in (1900, 2040, 2100, 2500, 3000, 3150):
    runs = []
    for _ in range(3):
        try:
            urllib.request.urlopen(urllib.request.Request(BASE + "/reset_prefix_cache", data=b"", method="POST")).read()
        except urllib.error.HTTPError:
            pass
        r = post("/v1/completions", {"model": "qwen38-flash-next", "prompt": ids[5000:5000 + n], "max_tokens": 1,
                                     "temperature": 0, "logprobs": 5, "cache_salt": uuid.uuid4().hex}
                 )["choices"][0]["logprobs"]["top_logprobs"][0]
        runs.append(r)
    t1 = [max(r, key=r.get) for r in runs]
    lp1 = [round(r[t1[0]], 4) if t1[0] in r else None for r in runs]
    common = set(runs[0]).intersection(*runs[1:])
    d = max(abs(r[k] - runs[0][k]) for r in runs[1:] for k in common)
    print(f"{n:5d} tokens: top1 {t1} logprob {lp1}; top-5 overlap {len(common)}/5, max |diff| on shared {d:.4f}")
