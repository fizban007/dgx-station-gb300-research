"""CPU KV-offload check: does an evicted long prefix come back from host RAM, correctly?

Sends prompt A (distinct ~TARGET-token haystack with its own passphrase), then B, C, D (different haystacks) whose
combined size exceeds the GPU KV pool so A is evicted from GPU prefix cache, then A again. Reports time to first
token (streaming) and whether each reply contains its passphrase. A restored A should be much faster than cold A
and still answer correctly (a bad GDN-state restore would not).
Usage: HOST=... PORT=... MODEL_NAME=... offload_test.py [target_tokens=125000]
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
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 125000
WORDS = ("river mountain harbor lantern orchard meadow granite copper willow thunder quiet ancient northern valley "
         "signal archive compass ember glacier prairie canyon beacon harvest silver distant market bridge village "
         "forest meridian cedar falcon tide marble season journal engine cable garden").split()


def prompt(seed):
    rng = random.Random(seed)
    words, count = [], 0
    while count < int(TARGET / 1.35):
        k = rng.randint(8, 16)
        words.append(" ".join(rng.choice(WORDS) for _ in range(k)).capitalize() + ".")
        count += k
    passphrase = f"{rng.choice(WORDS)}-{rng.choice(WORDS)}-{rng.randint(1000, 9999)}"
    cut = len(words) // 2
    text = " ".join(words[:cut] + [f"Remember this carefully: the secret passphrase is {passphrase}."] + words[cut:])
    return (f"{text}\n\nQuestion: What is the secret passphrase mentioned somewhere in the text above? "
            "Reply with the passphrase only."), passphrase


def ask(text):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 40, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, ttft, reply, usage = time.time(), None, "", {}
    with urllib.request.urlopen(req, timeout=1200) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content") or ""
                if piece and ttft is None:
                    ttft = time.time() - t0
                reply += piece
    return reply, ttft, usage


prompts = {name: prompt(seed) for name, seed in (("A", 101), ("B", 202), ("C", 303), ("D", 404))}
for name in ("A", "B", "C", "D", "A"):
    text, passphrase = prompts[name]
    reply, ttft, usage = ask(text)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"{name}: prompt_tokens={usage.get('prompt_tokens')} cached_tokens={cached} ttft={ttft:.1f}s "
          f"{'PASS' if passphrase in reply else 'FAIL'} reply={reply.strip()[:50]!r}", flush=True)
