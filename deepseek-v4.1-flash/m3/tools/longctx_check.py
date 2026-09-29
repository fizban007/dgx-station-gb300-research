"""Long-prompt quality probes for DS-V4.1 on :30006, identical inputs every run.

needle: a passphrase hidden at 10/50/90% depth in ~64K and ~125K tokens of seeded filler; thinking off, T=0.
cont:   greedy 48-token continuations (top-5 logprobs) of six real-text prompts (16K and 64K tokens) from the
        realtext_prefill corpus, each with a unique numeric prefix; saved for run-to-run comparison.

    longctx_check.py <label>          -> logs/longctx-<label>.json
    longctx_check.py --compare A B    -> per-prompt agreement between two saved runs
"""
import glob
import json
import os
import random
import sys
import time
import urllib.request

PORT = os.environ.get("PORT", "30006")
BASE = f"http://127.0.0.1:{PORT}"
MODEL = "dsv41-flash-uva"
LOG = "/home/jasonc/research/megamoe/logs"
WORDS = ("river mountain harbor lantern orchard meadow granite copper willow thunder quiet ancient northern valley "
         "signal archive compass ember glacier prairie canyon beacon harvest silver distant market bridge village "
         "forest meridian cedar falcon tide marble season journal engine cable garden").split()
PASSPHRASE = "violet-anchor-5817"


SALT = None  # per-run cache_salt: identical prompts across runs must prefill from scratch, not hit the prefix cache


def post(path, body, timeout=1800):
    if SALT and path.startswith("/v1/"):
        body = {**body, "cache_salt": SALT}
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def filler(n_words, seed):
    rng = random.Random(seed)
    out, count = [], 0
    while count < n_words:
        k = rng.randint(8, 16)
        out.append(" ".join(rng.choice(WORDS) for _ in range(k)).capitalize() + ".")
        count += k
    return " ".join(out)


def needle(results):
    for target in (65536, 131072):
        words = filler(int(target / 1.35), seed=1234).split(" ")
        for depth in (0.1, 0.5, 0.9):
            cut = int(len(words) * depth)
            hay = " ".join(words[:cut] + [f"Remember this carefully: the secret passphrase is {PASSPHRASE}."] + words[cut:])
            t0 = time.time()
            r = post("/v1/chat/completions", {
                "model": MODEL, "temperature": 0, "max_tokens": 40, "chat_template_kwargs": {"thinking": False},
                "messages": [{"role": "user", "content": hay + "\n\nQuestion: What is the secret passphrase mentioned "
                              "somewhere in the text above? Reply with the passphrase only."}]})
            reply = r["choices"][0]["message"]["content"] or ""
            row = {"target": target, "depth": depth, "prompt_tokens": r["usage"]["prompt_tokens"],
                   "s": round(time.time() - t0, 2), "pass": PASSPHRASE in reply, "reply": reply.strip()[:60]}
            print(json.dumps(row), flush=True)
            results["needle"].append(row)


def cont(results):
    files = sorted(glob.glob("/home/jasonc/research/megamoe/src/*.py")) + \
        sorted(glob.glob("/home/jasonc/b12x/docs/**/*.md", recursive=True)) + \
        sorted(glob.glob("/home/jasonc/b12x/b12x/moe/fused_moe/*.py"))
    corpus = "\n\n".join(open(f, errors="ignore").read() for f in files)
    toks = post("/tokenize", {"model": MODEL, "prompt": corpus})["tokens"]
    rng = random.Random(20260928)
    for n in (16384, 65536):
        for rep in range(3):
            start = rng.randrange(0, len(toks) - n)
            nonce = post("/tokenize", {"model": MODEL, "prompt": f"Document {rng.random():.12f}.\n"})["tokens"]
            prompt = (nonce + toks[start:start + n])[:n]
            r = post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": 48, "temperature": 0,
                                         "logprobs": 5})
            c = r["choices"][0]
            row = {"n": n, "rep": rep, "text": c["text"], "tokens": c["logprobs"]["tokens"],
                   "lp": c["logprobs"]["token_logprobs"], "top": c["logprobs"]["top_logprobs"]}
            print(json.dumps({"n": n, "rep": rep, "first": row["tokens"][:6], "lp0": round(row["lp"][0], 4)}), flush=True)
            results["cont"].append(row)


def compare(a, b):
    A = json.load(open(f"{LOG}/longctx-{a}.json"))
    B = json.load(open(f"{LOG}/longctx-{b}.json"))
    print(f"needle {a}: {sum(r['pass'] for r in A['needle'])}/{len(A['needle'])}   "
          f"{b}: {sum(r['pass'] for r in B['needle'])}/{len(B['needle'])}")
    for x, y in zip(A["cont"], B["cont"]):
        same = 0
        for s, t in zip(x["tokens"], y["tokens"]):
            if s != t:
                break
            same += 1
        # first-token distribution: max |dlogprob| over tokens in both top-5 sets
        tx, ty = x["top"][0], y["top"][0]
        common = set(tx) & set(ty)
        dmax = max((abs(tx[k] - ty[k]) for k in common), default=float("nan"))
        print(f"n={x['n']:6d} rep={x['rep']}: identical prefix {same:2d}/48 tokens, first-token top-5 overlap "
              f"{len(common)}/5, max |dlogprob| {dmax:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        SALT = f"{sys.argv[1]}-{time.time()}"
        res = {"label": sys.argv[1], "needle": [], "cont": []}
        cont(res)
        needle(res)
        json.dump(res, open(f"{LOG}/longctx-{sys.argv[1]}.json", "w"))
