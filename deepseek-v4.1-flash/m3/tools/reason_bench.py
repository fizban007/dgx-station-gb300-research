"""Steady-state reasoning decode at fixed concurrency (closed loop), with DSpark acceptance.

C workers each keep one request in flight: a GSM8K or MMLU-Pro question, thinking on at the server's default
effort, T=1.0 / top_p=0.95, up to 4,096 tokens, natural stops. Tokens are counted from streamed usage inside a
measurement window after a warmup, so the batch stays at C the whole time. Acceptance is the /metrics delta over
the same window.

    reason_bench.py <tag> [concurrencies] [--window 60] [--warmup 15]
"""
import argparse
import os
import json
import random
import re
import statistics
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:30006"
DATA = "/home/jasonc/llm-inference-bench/data"


def prompts():
    rows = [json.loads(l)["question"] for l in open(f"{DATA}/gsm8k_test.jsonl")]
    for l in open(f"{DATA}/mmlu_pro_1000.jsonl"):
        r = json.loads(l)
        opts = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(r.get("options", [])))
        rows.append(f"{r['question']}\n\n{opts}\n\nAnswer with the letter of the correct option.")
    random.Random(20260928).shuffle(rows)
    return rows


def metrics():
    text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(vllm:spec_decode_num_\w+?)_total(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if m:
            pos = re.search(r'position="(\d+)"', m.group(2) or "")
            key = m.group(1) + (f"[{pos.group(1)}]" if pos else "")
            out[key] = out.get(key, 0.0) + float(m.group(3))
    return out


def stream(prompt, events, stop):
    body = {"model": os.environ.get("MODEL", "glm53-flash"), "temperature": 1.0, "top_p": 0.95, "max_tokens": 4096, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    seen, first, last = 0, None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            usage = json.loads(line[6:]).get("usage")
            if not usage:
                continue
            n = usage.get("completion_tokens") or 0
            if n > seen:
                now = time.time()
                events.append((now, n - seen))
                first = first or now
                last = now
                seen = n
            if stop.is_set():
                break
    return seen, first, last


def run(c, pool, window, warmup):
    events, rates, stop = [], [], threading.Event()
    lock = threading.Lock()

    def worker(i):
        j = i
        while not stop.is_set():
            n, first, last = stream(pool[j % len(pool)], events, stop)
            j += c
            if n > 32 and last and first and last > first and not stop.is_set():
                with lock:
                    rates.append((n - 1) / (last - first))

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(c)]
    for t in threads:
        t.start()
    time.sleep(warmup)
    m0, t0 = metrics(), time.time()
    time.sleep(window)
    m1, t1 = metrics(), time.time()
    stop.set()
    for t in threads:
        t.join(timeout=120)
    toks = sum(n for ts, n in events if t0 <= ts < t1)
    d = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in m1}
    drafts = d.get("vllm:spec_decode_num_drafts", 0.0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
    per_pos = [round(d[k] / drafts, 2) for k in sorted((k for k in d if "per_pos[" in k),
                                                        key=lambda k: int(k.split("[")[1][:-1]))] if drafts else []
    return {"C": c, "agg_tok_s": round(toks / (t1 - t0), 1), "per_stream_tok_s": round(toks / (t1 - t0) / c, 1),
            "median_request_decode_tok_s": round(statistics.median(rates), 1) if rates else None,
            "requests_done": len(rates), "accept_tokens_per_step": round(1 + acc / drafts, 2) if drafts else None,
            "avg_k": round(d.get("vllm:spec_decode_num_draft_tokens", 0.0) / drafts, 2) if drafts else None,
            "accept_per_position": per_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("concurrency", nargs="?", default="1,4,8,12,16,24")
    ap.add_argument("--window", type=float, default=60)
    ap.add_argument("--warmup", type=float, default=15)
    args = ap.parse_args()
    pool = prompts()
    results = []
    for c in (int(x) for x in args.concurrency.split(",")):
        r = run(c, pool, args.window, args.warmup)
        results.append(r)
        print(json.dumps(r), flush=True)
    json.dump({"tag": args.tag, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "window_s": args.window,
               "sampling": "T=1.0 top_p=0.95, thinking on at server default effort", "results": results},
              open(f"/home/jasonc/research/glm53-flash/logs/reason-{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
