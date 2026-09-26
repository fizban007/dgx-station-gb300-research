"""Profile a fixed decode load: start_profile, C concurrent knee prompts of N tokens, stop_profile."""
import concurrent.futures as cf, json, sys, time, urllib.request
port, model, conc, ntok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
base = f"http://127.0.0.1:{port}"
def post(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=900).read()
def one(i, n=ntok):
    body = {"model": model, "messages": [{"role": "user", "content": f"Write a detailed paragraph about the number {i}, its history and uses. No lists."}],
            "max_tokens": n, "temperature": 0, "chat_template_kwargs": {"thinking": False}, "ignore_eos": True}
    return json.loads(post("/v1/chat/completions", body))["usage"]["completion_tokens"]
with cf.ThreadPoolExecutor(conc) as ex: list(ex.map(lambda i: one(i, 16), range(conc)))  # warm
post("/start_profile"); t = time.time()
with cf.ThreadPoolExecutor(conc) as ex: toks = sum(ex.map(one, range(100, 100 + conc)))
dt = time.time() - t; post("/stop_profile")
print(f"C{conc}: {toks} tokens in {dt:.2f}s = {toks / dt:.1f} tok/s (profiled)")
