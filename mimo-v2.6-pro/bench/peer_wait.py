"""How long does the GB300 stall on the 6000 per forward pass? Reads the MiMo peer-tier counters around steady decode.

Counters (hook/peer_tier_mimo.py, words 0-6 of /dev/shm/vllm_peer_mimo): [0] published seq (one per MoE layer call),
[2] timeouts, [3] rows sent, [4] tokens seen, [5] ns the GB300 spent spinning on the sidecar, [6] waits.
For each concurrency: C streams x N tokens of short prose (thinking off), then per forward pass (published / 69):
wall ms, GB300 wait ms, rows sent to the 6000, and aggregate tok/s.
usage: peer_wait.py [concurrencies=1,4,8,16] [tokens=256]
"""
import concurrent.futures as cf
import json
import struct
import sys
import time
import urllib.request

LAYERS = 69
CS = [int(c) for c in (sys.argv[1] if len(sys.argv) > 1 else "1,4,8,16").split(",")]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 256


def words():
    return struct.unpack("<7q", open("/dev/shm/vllm_peer_mimo", "rb").read(56))


def one(i):
    body = {"model": "mimo26-pro", "max_tokens": N, "temperature": 0, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": f"Write a detailed paragraph about the number {i}, its history and uses. No lists."}]}
    req = urllib.request.Request("http://127.0.0.1:30007/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))["usage"]["completion_tokens"]


for c in CS:
    one(10_000 + c)  # warm this prompt shape
    a, t = words(), time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        toks = sum(ex.map(one, range(100 * c, 100 * c + c)))
    dt, b = time.time() - t, words()
    passes = (b[0] - a[0]) / LAYERS
    waits = b[6] - a[6]
    print(json.dumps({"C": c, "tok_s": round(toks / dt, 1), "passes": int(passes),
                      "ms_per_pass": round(dt * 1e3 / max(passes, 1), 2),
                      "wait_ms_per_pass": round((b[5] - a[5]) / 1e6 / max(passes, 1), 3),
                      "wait_us_per_wait": round((b[5] - a[5]) / 1e3 / max(1, waits), 1),
                      "rows_per_layer": round((b[3] - a[3]) / max(1, b[0] - a[0]), 2),
                      "timeouts": b[2] - a[2]}), flush=True)
