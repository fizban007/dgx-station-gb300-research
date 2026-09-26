"""Split TTFT of one cold request into frontend input processing vs engine queue + prefill, using the server's own
/metrics histogram sums (Python frontend only). Compares a text chat prompt with the same tokens sent as token ids.
Usage: ttft_split.py [sizes=32768,131072]"""
import glob, json, re, sys, time, urllib.request, uuid
BASE = "http://127.0.0.1:30006"
MODEL = "qwen38-flash-next"
def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
def metrics():
    txt = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("time_to_first_token_seconds", "request_queue_time_seconds", "request_prefill_time_seconds"):
        m = re.search(rf'^vllm:{name}_sum{{[^}}]*}} ([\d.e+-]+)', txt, re.M)
        out[name] = float(m.group(1))
    return out
files = sorted(glob.glob("/home/jasonc/b12x/**/*.py", recursive=True))
text = "".join(open(f, errors="ignore").read() for f in files)
ids = post("/tokenize", {"model": MODEL, "prompt": text})["tokens"]
sizes = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "32768,131072").split(",")]
print(f"corpus {len(ids)} tokens")
for n in sizes:
    chunk = post("/detokenize", {"model": MODEL, "tokens": ids[:n]})["prompt"]
    for kind in ("chat-text", "completion-ids"):
        m0 = metrics(); t = time.perf_counter()
        if kind == "chat-text":
            r = post("/v1/chat/completions", {"model": MODEL, "messages": [{"role": "user", "content": chunk}],
                     "max_tokens": 1, "temperature": 0, "cache_salt": uuid.uuid4().hex,
                     "chat_template_kwargs": {"enable_thinking": False}})
        else:
            r = post("/v1/completions", {"model": MODEL, "prompt": ids[:n], "max_tokens": 1, "temperature": 0,
                     "cache_salt": uuid.uuid4().hex})
        wall = time.perf_counter() - t; m1 = metrics()
        d = {k: m1[k] - m0[k] for k in m0}
        pt = r["usage"]["prompt_tokens"]
        print(f"{kind:14s} prompt {pt:7d} tok: client wall {wall*1e3:7.0f} ms | server TTFT {d['time_to_first_token_seconds']*1e3:7.0f} "
              f"= queue {d['request_queue_time_seconds']*1e3:5.0f} + prefill {d['request_prefill_time_seconds']*1e3:6.0f} "
              f"+ other {1e3*(d['time_to_first_token_seconds']-d['request_queue_time_seconds']-d['request_prefill_time_seconds']):5.0f} ms", flush=True)
