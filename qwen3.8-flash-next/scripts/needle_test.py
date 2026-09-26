"""Long-context retrieval check: hide a passphrase at several depths in ~TARGET tokens of filler and ask for it.

Filler is deterministic pseudo-prose built from a fixed word list (seeded), so every run and config sees the same
haystack. Thinking is off and temperature is 0; a depth passes if the reply contains the passphrase.
Usage: HOST=... PORT=... MODEL_NAME=... needle_test.py <label> [target_tokens=125000] [depths=0.1,0.5,0.9]
"""
import json
import os
import random
import sys
import time
import urllib.request

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = os.environ.get("PORT", "5000")
MODEL = os.environ.get("MODEL_NAME", "Qwen3.8-27B")
LABEL = sys.argv[1]
TARGET = int(sys.argv[2]) if len(sys.argv) > 2 else 125000
DEPTHS = [float(d) for d in (sys.argv[3] if len(sys.argv) > 3 else "0.1,0.5,0.9").split(",")]
WORDS = ("river mountain harbor lantern orchard meadow granite copper willow thunder quiet ancient northern valley "
         "signal archive compass ember glacier prairie canyon beacon harvest silver distant market bridge village "
         "forest meridian cedar falcon tide marble season journal engine cable garden").split()
PASSPHRASE = "violet-anchor-5817"


def filler(n_words, seed):
    rng = random.Random(seed)
    sentences, count = [], 0
    while count < n_words:
        k = rng.randint(8, 16)
        sentences.append(" ".join(rng.choice(WORDS) for _ in range(k)).capitalize() + ".")
        count += k
    return " ".join(sentences)


def ask(prompt):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 40,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"] or "", out["usage"]["prompt_tokens"], time.time() - t0


n_words = int(TARGET / 1.35)  # ~1.35 tokens per filler word, calibrated on the Qwen tokenizer
passed = 0
for depth in DEPTHS:
    words = filler(n_words, seed=1234).split(" ")
    cut = int(len(words) * depth)
    needle = f"Remember this carefully: the secret passphrase is {PASSPHRASE}."
    haystack = " ".join(words[:cut] + [needle] + words[cut:])
    prompt = (f"{haystack}\n\nQuestion: What is the secret passphrase mentioned somewhere in the text above? "
              "Reply with the passphrase only.")
    reply, n_tok, dt = ask(prompt)
    ok = PASSPHRASE in reply
    passed += ok
    print(f"{LABEL} depth={depth:.0%} prompt_tokens={n_tok} time={dt:.0f}s {'PASS' if ok else 'FAIL'} reply={reply.strip()[:60]!r}",
          flush=True)
print(f"{LABEL} needle {passed}/{len(DEPTHS)}")
