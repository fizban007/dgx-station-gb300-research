"""Long-context probe for the :30007 MiMo lane (served max_model_len 262,144).
Per target size: 3 needle-recall runs (depth 10/50/90%) + 1 decode-timing run (256 tok, ignore_eos).
Filler = varied real text (Python stdlib source), unique nonce at the head defeats prefix caching.
Results append to longctx-TAG.jsonl; summary to longctx-TAG.json.
usage: python3 longctx_bench.py TAG [sizes...]"""
import json, os, sys, time, urllib.request, random, glob, statistics
TAG = sys.argv[1]
SIZES = [int(s) for s in sys.argv[2:]] or [65536, 131072, 196608, 253952]
BASE = os.environ.get("BASE", "http://127.0.0.1:30007")
MODEL = "mimo26-pro"

files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True))
random.Random(7).shuffle(files)
corpus = []
n = 0
for f in files:
    try: t = open(f, errors="ignore").read()
    except Exception: continue
    corpus.append(f"\n# ==== {f} ====\n{t}"); n += len(t)
    if n > int(os.environ.get("CORPUS_CHARS", 8_000_000)): break
CORPUS = "".join(corpus)

def post(path, obj, stream=False):
    return urllib.request.urlopen(urllib.request.Request(BASE + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}), timeout=3600)

def ntok(text):
    return json.load(post("/tokenize", {"model": MODEL, "prompt": text}))["count"]

sample = CORPUS[:400_000]
CPT = len(sample) / ntok(sample)   # chars per token for this filler
print(f"[{TAG}] chars/token {CPT:.3f}", flush=True)

def build(target, depth, code, nonce):
    body_chars = int((target - 400) * CPT)
    off = random.Random(nonce).randrange(0, len(CORPUS) - int(body_chars * 1.6))
    body = CORPUS[off: off + body_chars]
    for _ in range(3):  # exact-size via tokenizer (filler density varies by file)
        have = ntok(body)
        if abs(have - (target - 400)) < 256: break
        body_chars = int(body_chars * (target - 400) / have)
        body = CORPUS[off: off + body_chars]
    needle = f"\n# NOTE FOR REVIEWERS: the deployment access code for project Kestrel is {code}.\n" if code else ""
    cut = int(len(body) * depth)
    cut = body.rfind("\n", 0, cut) + 1 or cut
    body = body[:cut] + needle + body[cut:]
    q = ("What is the deployment access code for project Kestrel mentioned in the source above? "
         "Reply with the code only.") if code else "Summarise the first file above in three sentences."
    return f"[{nonce}] Below is a large dump of source code.\n\n{body}\n\n{q}"

def run(prompt, maxtok, ignore_eos):
    req = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": maxtok,
           "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos,
           "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.time(); first = None; usage = None; out = []
    for line in post("/v1/chat/completions", req):
        line = line.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"): continue
        d = json.loads(line[5:])
        if d.get("usage"): usage = d["usage"]
        if d.get("choices"):
            c = d["choices"][0].get("delta", {}).get("content")
            if c:
                if first is None: first = time.time()
                out.append(c)
    t1 = time.time()
    n = usage["completion_tokens"]
    dec = (n - 1) / (t1 - first) if first and n > 1 else None
    return {"prompt_tokens": usage["prompt_tokens"], "ttft_s": round(first - t0, 2) if first else None,
            "decode_tok_s": round(dec, 1) if dec else None, "completion_tokens": n, "text": "".join(out)}

rows = []
log = open(f"longctx-{TAG}.jsonl", "a")
for size in SIZES:
    for depth in (0.1, 0.5, 0.9):
        code = f"{random.randint(100,999)}-{random.choice(['ORCA','LYNX','WREN','BISON','OTTER'])}-{random.randint(1000,9999)}"
        nonce = f"{TAG}-{size}-{depth}-{random.random()}"
        try:
            r = run(build(size, depth, code, nonce), 32, False)
            r.update(kind="needle", size=size, depth=depth, code=code, pass_=code in r["text"])
        except Exception as e:
            r = {"kind": "needle", "size": size, "depth": depth, "error": repr(e)}
        rows.append(r); log.write(json.dumps(r) + "\n"); log.flush()
        print(f"[{TAG}] needle {size:>6} d={depth}: prompt={r.get('prompt_tokens')} TTFT={r.get('ttft_s')}s "
              f"prefill={round(r['prompt_tokens']/r['ttft_s']) if r.get('ttft_s') else None} tok/s "
              f"pass={r.get('pass_')} got={r.get('text', r.get('error'))!r}", flush=True)
    nonce = f"{TAG}-{size}-dec-{random.random()}"
    try:
        r = run(build(size, 0.5, None, nonce), 256, True); r.update(kind="decode", size=size); r.pop("text")
    except Exception as e:
        r = {"kind": "decode", "size": size, "error": repr(e)}
    rows.append(r); log.write(json.dumps(r) + "\n"); log.flush()
    print(f"[{TAG}] decode {size:>6}: prompt={r.get('prompt_tokens')} TTFT={r.get('ttft_s')}s decode={r.get('decode_tok_s')} tok/s", flush=True)
json.dump(rows, open(f"longctx-{TAG}.json", "w"), indent=1)
print(f"[{TAG}] LONGCTX_DONE needles {sum(1 for r in rows if r.get('pass_'))}/{sum(1 for r in rows if r['kind']=='needle')}", flush=True)
