"""Separate prefill from decode on the :30007 lane. Streams responses and records TTFT and decode tok/s.
Prompt sizes ~2k/8k/32k tokens of mixed agent-shaped text; 3 reps each; prefix caching defeated by a nonce.
usage: python3 ttft_bench.py TAG"""
import json, sys, time, urllib.request, random, statistics
TAG = sys.argv[1]
B = "http://127.0.0.1:30007/v1/chat/completions"
import os
WD = os.environ.get("WORKDIR", os.getcwd())
# filler text for the prompt: two shell scripts from the lane (the measured runs used the unscrubbed originals, so token counts differ by a few tokens)
BLOCK = open(os.path.join(WD, "launch-mimo26-pro.sh")).read() + "\n" + open(os.path.join(WD, "knee.sh")).read()
def prompt(ntok, nonce):
    body = (BLOCK * (1 + ntok * 4 // len(BLOCK)))[: ntok * 4]
    return f"[{nonce}] Here are some shell scripts.\n\n{body}\n\nSummarise what the first script does in three sentences."
def run(ntok, nonce, maxtok=128):
    req = {"model": "mimo26-pro", "messages": [{"role": "user", "content": prompt(ntok, nonce)}], "max_tokens": maxtok,
           "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
           "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; usage = None
    r = urllib.request.urlopen(urllib.request.Request(B, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=1800)
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        if first is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"): first = time.time()
    t1 = time.time()
    n = usage["completion_tokens"]
    return usage["prompt_tokens"], first - t0, (n - 1) / (t1 - first)
res = {}
for ntok in (2000, 8000, 32000):
    rows = [run(ntok, f"{TAG}-{ntok}-{i}-{random.random()}") for i in range(3)]
    pt = rows[0][0]; ttft = statistics.median(r[1] for r in rows); dec = statistics.median(r[2] for r in rows)
    res[ntok] = {"prompt_tokens": pt, "ttft_s": round(ttft, 2), "prefill_tok_s": round(pt / ttft, 0), "decode_tok_s": round(dec, 1),
                 "runs": [(round(a, 2), round(b, 1)) for _, a, b in rows]}
    print(f"[{TAG}] prompt {pt:6d} tok: TTFT {ttft:6.2f}s ({pt/ttft:6.0f} tok/s prefill)  decode {dec:5.1f} tok/s  runs {res[ntok]['runs']}", flush=True)
json.dump(res, open(os.path.join(WD, f"ttft-{TAG}.json"), "w"))
