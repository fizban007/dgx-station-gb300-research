"""Time the server's /tokenize for long text, as a raw prompt and as a chat message (best of 3; includes HTTP and
the JSON of the returned ids). Isolates the Python frontend's tokenization cost from engine prefill."""
import glob, json, time, urllib.request
B = "http://127.0.0.1:30006"
def post(p, b):
    r = urllib.request.Request(B + p, data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(r, timeout=600) as f:
        d = json.loads(f.read())
    return d, time.perf_counter() - t
text = "".join(open(f, errors="ignore").read() for f in sorted(glob.glob("/home/jasonc/b12x/**/*.py", recursive=True)))
ids, _ = post("/tokenize", {"model": "qwen38-flash-next", "prompt": text[:3_000_000]})
ids = ids["tokens"]
for n in (32768, 131072):
    chunk, _ = post("/detokenize", {"model": "qwen38-flash-next", "tokens": ids[:n]})
    chunk = chunk["prompt"]
    for kind, body in (("tokenize prompt", {"model": "qwen38-flash-next", "prompt": chunk}),
                       ("tokenize messages", {"model": "qwen38-flash-next", "messages": [{"role": "user", "content": chunk}]})):
        best = min(post("/tokenize", body)[1] for _ in range(3))
        print(f"{n:6d} tok  {kind:18s} best of 3: {best * 1e3:6.0f} ms (includes HTTP + JSON of the returned ids)")
