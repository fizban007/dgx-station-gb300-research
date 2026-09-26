"""Greedy 32-token continuations with logprobs on real-text prompts of fixed token lengths.
save: python greedy_check.py save out.json ; compare: python greedy_check.py compare base.json"""
import glob, json, sys, urllib.error, urllib.request
BASE = "http://127.0.0.1:30006"
LENS = [900, 2000, 3300, 5000, 8192, 12000, 16384]
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())
text = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/home/jasonc/b12x/b12x/**/*.py", recursive=True))[:400])
ids = post("/tokenize", {"model": "qwen38-flash-next", "prompt": text})["tokens"]
assert len(ids) > max(LENS) + 1000, len(ids)
res = {}
for n in LENS:
    for off in (0, 20000):
        try:  # needs VLLM_SERVER_DEV_MODE=1; without it the first pass over new prompts is still cold
            urllib.request.urlopen(urllib.request.Request(BASE + "/reset_prefix_cache", data=b"", method="POST"), timeout=60).read()
        except urllib.error.HTTPError:
            pass
        r = post("/v1/completions", {"model": "qwen38-flash-next", "prompt": ids[off:off + n], "max_tokens": 32,
                                     "temperature": 0, "logprobs": 1})["choices"][0]
        lp = r["logprobs"]
        res[f"{n}@{off}"] = {"tokens": lp["tokens"], "lps": lp["token_logprobs"]}
mode, path = sys.argv[1], sys.argv[2]
if mode == "save":
    json.dump(res, open(path, "w"))
    print("saved", len(res), "cases")
else:
    base = json.load(open(path))
    for k, v in res.items():
        b = base[k]
        m = next((i for i, (x, y) in enumerate(zip(v["tokens"], b["tokens"])) if x != y), len(v["tokens"]))
        d = max(abs(x - y) for x, y in zip(v["lps"][:m], b["lps"][:m])) if m else float("nan")
        print(f"{k:>12}: first mismatch at {m:2d}/32, max |dlogprob| before it {d:.4f}")
