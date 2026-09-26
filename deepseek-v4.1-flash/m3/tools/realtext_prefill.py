"""Prefill on real text (source code + docs) with a unique prefix per request; reports tok/s and cold rows/token."""
import glob, json, os, random, struct, time, urllib.request
PORT = os.environ.get("PORT", "30006")
files = sorted(glob.glob("/home/jasonc/research/megamoe/src/*.py")) + sorted(glob.glob("/home/jasonc/b12x/docs/**/*.md", recursive=True)) \
    + sorted(glob.glob("/home/jasonc/b12x/b12x/moe/fused_moe/*.py"))
corpus = "\n\n".join(open(f, errors="ignore").read() for f in files)

def tokenize(text):
    body = {"model": "dsv41-flash-uva", "prompt": text}
    r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/tokenize", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})))
    return r["tokens"]

def counters():
    return struct.unpack("<qqqqq", open("/dev/shm/vllm_peer_tier2", "rb").read(40))

toks = tokenize(corpus)
print(f"corpus {len(corpus)/1e6:.1f} MB, {len(toks)} tokens", flush=True)
rng = random.Random(7)
for n in (16384, 65536):
    for rep in range(3):
        start = rng.randrange(0, len(toks) - n)
        nonce = tokenize(f"Request {rng.random():.12f}.\n")
        prompt = (nonce + toks[start:start + n])[:n]
        a = counters(); t = time.time()
        body = {"model": "dsv41-flash-uva", "prompt": prompt, "max_tokens": 1, "temperature": 0}
        urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600).read()
        dt = time.time() - t; b = counters()
        print(f"{n} tokens: {n/dt:8.0f} tok/s  ttft {dt:.3f}s  cold rows/token {(b[3]-a[3])/max(1,b[4]-a[4]):.3f}", flush=True)
