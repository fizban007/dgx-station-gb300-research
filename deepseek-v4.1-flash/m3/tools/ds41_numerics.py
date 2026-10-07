"""Long-context numerics for DS41 on :30006 (copy of qwen38 qwen_numerics.py / glm kv_numerics.py; MODEL and NUMLOG env). Compare an FP8-KV and an NVFP4-KV boot of the same lane config (pinned tactics).

GLM's per-token KV lives only in its 11 DSA (MLA, 512-wide NoPE latent) layers. With --kv-cache-dtype fp8 the
FlashInfer sparse-MLA kernel also quantizes the absorbed query to FP8, and the checkpoint ships no KV/Q scales, so both
use 1.0. This measures what that costs against a BF16-KV boot of the same checkpoint and speculator:
  short: teacher-forced prompt logprobs on 48 seeded 2,048-token windows of real text (code + Markdown docs)
         (prompt logprobs materialize [tokens, vocab] logits, so windows stay short)
  long:  greedy 64-token continuations (T=0, top-5 logprobs) after 16K / 64K / 128K-token real-text prompts, 3 each
Speculative decoding is lossless at T=0 only up to kernel nondeterminism; compare boots with the same speculator.

    kv_numerics.py <label>           -> logs/kvnum-<label>.json
    kv_numerics.py --compare A B     -> NLL delta, argmax agreement, |dlogprob| percentiles; long-context agreement
"""
import glob, json, math, random, sys, time, urllib.request
import os
BASE = "http://127.0.0.1:30006"; MODEL = os.environ.get("MODEL", "dsv41-flash-uva"); LOG = os.environ.get("NUMLOG", "/home/jasonc/research/megamoe/logs/kvnum")
WIN, NWIN = 2048, 48
LONG = (16384, 65536, 131072)


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def corpus_tokens():
    files = sorted(glob.glob("/home/jasonc/research/megamoe/src/*.py")) + \
        sorted(glob.glob("/home/jasonc/b12x/docs/**/*.md", recursive=True)) + \
        sorted(glob.glob("/home/jasonc/b12x/b12x/**/*.py", recursive=True)) + \
        sorted(glob.glob("/home/jasonc/research/*.md")) + sorted(glob.glob("/home/jasonc/research/*/README.md"))
    corpus = "\n\n".join(open(f, errors="ignore").read() for f in files)
    return post("/tokenize", {"model": MODEL, "prompt": corpus})["tokens"]


def run(label):
    toks = corpus_tokens()
    assert len(toks) > max(LONG) + 1000, f"corpus too short: {len(toks)} tokens"
    rng = random.Random(20260930)
    salt = f"kvnum-{label}-{time.time()}"
    short = []
    for w in range(NWIN):
        start = rng.randrange(0, len(toks) - WIN)
        r = post("/v1/completions", {"model": MODEL, "prompt": toks[start:start + WIN], "max_tokens": 1, "temperature": 0,
                                     "prompt_logprobs": 1, "cache_salt": salt})
        lp, top = [], []
        for pos, d in enumerate(r["choices"][0]["prompt_logprobs"][1:], start=1):
            lp.append(d[str(toks[start + pos])]["logprob"])
            top.append(min(d.items(), key=lambda kv: kv[1]["rank"])[0])
        short.append({"start": start, "lp": lp, "top": top})
    allv = [x for w in short for x in w["lp"]]
    print(json.dumps({"label": label, "short_tokens": len(allv), "mean_nll": round(-sum(allv) / len(allv), 5)}), flush=True)
    long = []
    for n in LONG:
        for k in range(3):
            start = rng.randrange(0, len(toks) - n)
            t = time.perf_counter()
            r = post("/v1/completions", {"model": MODEL, "prompt": toks[start:start + n], "max_tokens": 64, "temperature": 0,
                                         "logprobs": 5, "cache_salt": salt})
            lp = r["choices"][0]["logprobs"]
            long.append({"n": n, "start": start, "tokens": lp["tokens"], "top": lp["top_logprobs"],
                         "s": round(time.perf_counter() - t, 2)})
            print(json.dumps({"n": n, "k": k, "s": long[-1]["s"], "text": r["choices"][0]["text"][:60]}), flush=True)
    json.dump({"label": label, "short": short, "long": long}, open(f"{LOG}/kvnum-{label}.json", "w"))


def compare(a, b):
    A = json.load(open(f"{LOG}/kvnum-{a}.json")); B = json.load(open(f"{LOG}/kvnum-{b}.json"))
    la = [x for w in A["short"] for x in w["lp"]]; lb = [x for w in B["short"] for x in w["lp"]]
    ta = [x for w in A["short"] for x in w["top"]]; tb = [x for w in B["short"] for x in w["top"]]
    d = sorted(abs(x - y) for x, y in zip(la, lb))
    q = lambda p: d[min(len(d) - 1, int(p * len(d)))]
    na, nb = -sum(la) / len(la), -sum(lb) / len(lb)
    print(json.dumps({"short_tokens": len(d), f"nll_{a}": round(na, 5), f"nll_{b}": round(nb, 5),
                      "delta_nll": round(nb - na, 5), "ppl_ratio": round(math.exp(nb - na), 5),
                      "argmax_agree": round(sum(x == y for x, y in zip(ta, tb)) / len(ta), 5),
                      "abs_dlogprob_p50": round(q(0.5), 5), "p90": round(q(0.9), 4), "p99": round(q(0.99), 4),
                      "max": round(d[-1], 4)}))
    for x, y in zip(A["long"], B["long"]):
        assert x["start"] == y["start"] and x["n"] == y["n"]
        same = 0
        for s, t in zip(x["tokens"], y["tokens"]):
            if s != t:
                break
            same += 1
        common = set(x["top"][0]) & set(y["top"][0])
        dl = max((abs(x["top"][0][k] - y["top"][0][k]) for k in common), default=float("nan"))
        print(json.dumps({"ctx": x["n"], "identical_prefix": f"{same}/{len(x['tokens'])}", "first_top5_overlap": len(common),
                          "first_max_dlogprob": round(dl, 4), "s": [x["s"], y["s"]]}))


if __name__ == "__main__":
    compare(sys.argv[2], sys.argv[3]) if sys.argv[1] == "--compare" else run(sys.argv[1])
