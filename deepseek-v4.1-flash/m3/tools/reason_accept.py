"""Reasoning-heavy C1 decode: 8 GSM8K questions with thinking on at the server's default effort, greedy,
up to 4,096 tokens each. Reports per-request tok/s and reasoning length; acceptance comes from the ab-suite snapshot."""
import json
import time
import urllib.request

rows = [json.loads(l) for l in open("/home/jasonc/llm-inference-bench/data/gsm8k_test.jsonl")][:8]
total_tokens, total_s = 0, 0.0
for i, row in enumerate(rows):
    body = {"model": "dsv41-flash-uva", "temperature": 0, "max_tokens": 4096,
            "messages": [{"role": "user", "content": row["question"]}]}
    req = urllib.request.Request("http://127.0.0.1:30006/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    dt = time.time() - t
    msg = r["choices"][0]["message"]
    n = r["usage"]["completion_tokens"]
    total_tokens, total_s = total_tokens + n, total_s + dt
    print(json.dumps({"q": i, "tokens": n, "tok_s": round(n / dt, 1),
                      "reasoning_chars": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
                      "finish": r["choices"][0]["finish_reason"]}), flush=True)
print(f"reasoning C1: {total_tokens} tokens in {total_s:.1f} s = {total_tokens / total_s:.1f} tok/s (includes TTFT)")
