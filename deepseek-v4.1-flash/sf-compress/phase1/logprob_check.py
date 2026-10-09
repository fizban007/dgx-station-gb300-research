"""Prompt logprobs (prefill only, no speculative decoding) over fixed texts, for bit-exact A/B of MEGA_SF_COMPRESS.

  python logprob_check.py --out lp-base.json [--compare lp-other.json]
"""
import argparse, glob, json, os, urllib.request
URL = os.environ.get("SERVER", "http://localhost:8001")
def post(path, body):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                                                   headers={"Content-Type": "application/json"}), timeout=900))
p = argparse.ArgumentParser(); p.add_argument("--out", required=True); p.add_argument("--compare"); a = p.parse_args()
# Fixed local texts (any ~70 KB of source files will do; the A/B only needs both runs to use the same ones).
files = sorted(glob.glob(os.path.expanduser("~/src/b12x/b12x/moe/fused_moe/*.py")))[:6] + [os.path.expanduser("~/ai/llm/ds4-station/m3/README.md")]
texts = [open(f).read()[:12000] for f in files]
res = []
for t in texts:
    r = post("/v1/completions", {"model": "local-model", "prompt": t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
    lp = [next(iter(d.values()))["logprob"] if d else None for d in r["choices"][0]["prompt_logprobs"]]
    res.append(lp)
json.dump(res, open(a.out, "w"))
n = sum(len(x) for x in res)
print(f"{len(res)} texts, {n} prompt tokens, mean logprob {sum(v for x in res for v in x if v is not None) / n:.4f}")
if a.compare:
    base = json.load(open(a.compare))
    diff = [(i, j, x, y) for i, (X, Y) in enumerate(zip(res, base)) for j, (x, y) in enumerate(zip(X, Y)) if x != y]
    print(f"tokens differing from {a.compare}: {len(diff)} of {n}")
    if diff:
        print("max abs diff:", max(abs(x - y) for _, _, x, y in diff if x is not None and y is not None))
