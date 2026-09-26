"""Collect per-(layer, expert) routing counts for three disjoint workloads (server launched with MEGA_COUNT).

  A  calibration: Python stdlib source + /usr/share/doc text (prefill), MMLU-Pro and GSM8K first halves (chat decode)
  B  held-out:    vLLM model sources + b12x docs and MoE sources (prefill), MMLU-Pro and GSM8K second halves (chat)
  C  random:      catid-style random token ids (prefill)

Counts are cumulative on the server; each phase's counts are the difference of snapshots taken after the dumper
(every 20 s) has caught up. Writes prof/counts-{A,B,C}.json.
"""
import concurrent.futures as cf
import glob
import json
import os
import random
import time
import urllib.request

PORT = os.environ.get("PORT", "30006")
BASE = f"http://127.0.0.1:{PORT}"
DUMP = "/home/jasonc/research/megamoe/prof/route-counts.json"
OUT = "/home/jasonc/research/megamoe/prof"
CHUNK = 8192
rng = random.Random(11)


def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def tokens_of(files, limit):
    toks = []
    for f in files:
        try:
            text = open(f, errors="ignore").read()
        except OSError:
            continue
        if len(text) < 200:
            continue
        toks += post("/tokenize", {"model": "dsv41-flash-uva", "prompt": text[:400000]})["tokens"]
        if len(toks) >= limit:
            break
    return toks[:limit]


def prefill(toks, label):
    t0, n = time.time(), 0
    for i in range(0, len(toks) - 256, CHUNK):
        nonce = post("/tokenize", {"model": "dsv41-flash-uva", "prompt": f"Document {rng.random():.12f}\n"})["tokens"]
        prompt = (nonce + toks[i:i + CHUNK])[:CHUNK]
        post("/v1/completions", {"model": "dsv41-flash-uva", "prompt": prompt, "max_tokens": 1, "temperature": 0})
        n += len(prompt)
    print(f"{label}: prefilled {n} tokens in {time.time() - t0:.0f}s", flush=True)


def chat(questions, label, max_tokens=384):
    def one(i_q):
        i, q = i_q
        body = {"model": "dsv41-flash-uva", "messages": [{"role": "user", "content": q}], "max_tokens": max_tokens,
                "temperature": 0.6, "chat_template_kwargs": {"thinking": i % 2 == 0}}
        return post("/v1/chat/completions", body)["usage"]["completion_tokens"]

    t0 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        out = sum(ex.map(one, enumerate(questions)))
    print(f"{label}: {len(questions)} chats, {out} generated tokens in {time.time() - t0:.0f}s", flush=True)


def mmlu(lines):
    out = []
    for line in lines:
        d = json.loads(line)
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(d["options"]))
        out.append(f"{d['question']}\n\n{opts}\n\nAnswer with the letter and a short justification.")
    return out


def snapshot(after):
    while True:
        try:
            d = json.load(open(DUMP))
            if d["time"] > after + 1:
                return {int(k): v for k, v in d["layers"].items()}
        except (OSError, ValueError):
            pass
        time.sleep(5)


def flush_snapshot():
    """The hook copies counts to host only on eager passes, at most every 10 s: wait, force one, then read."""
    time.sleep(11)
    post("/v1/completions", {"model": "dsv41-flash-uva", "prompt": [rng.randrange(1000, 120000) for _ in range(300)],
                             "max_tokens": 1, "temperature": 0})
    return snapshot(time.time())


def diff(a, b):
    return {k: [y - x for x, y in zip(a[k], b[k])] for k in b}


def main():
    mm = open("/home/jasonc/llm-inference-bench/data/mmlu_pro_1000.jsonl").read().splitlines()
    gs = [json.loads(l)["question"] for l in open("/home/jasonc/llm-inference-bench/data/gsm8k_test.jsonl")]
    rng.shuffle(mm)
    rng.shuffle(gs)

    base = flush_snapshot()
    stdlib = sorted(glob.glob("/usr/lib/python3.12/*.py")) + sorted(glob.glob("/usr/lib/python3.12/*/*.py"))
    docs = sorted(f for f in glob.glob("/usr/share/doc/**/*", recursive=True)
                  if os.path.isfile(f) and not f.endswith(".gz") and os.path.getsize(f) < 2_000_000)
    prefill(tokens_of(stdlib, 700_000), "A stdlib code")
    prefill(tokens_of(docs, 500_000), "A docs prose")
    chat(mmlu(mm[:200]) + gs[:200], "A chat")
    snap_a = flush_snapshot()
    json.dump({"layers": diff(base, snap_a)}, open(f"{OUT}/counts-A.json", "w"))

    held = sorted(glob.glob("/home/jasonc/research/megamoe/src/*.py")) + \
        sorted(glob.glob("/home/jasonc/b12x/docs/**/*.md", recursive=True)) + \
        sorted(glob.glob("/home/jasonc/b12x/b12x/moe/fused_moe/*.py"))
    prefill(tokens_of(held, 300_000), "B held-out text")
    chat(mmlu(mm[500:580]) + gs[700:780], "B chat")
    snap_b = flush_snapshot()
    json.dump({"layers": diff(snap_a, snap_b)}, open(f"{OUT}/counts-B.json", "w"))

    rand = [rng.randrange(1000, 120000) for _ in range(200_000)]
    prefill(rand, "C random ids")
    snap_c = flush_snapshot()
    json.dump({"layers": diff(snap_b, snap_c)}, open(f"{OUT}/counts-C.json", "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
