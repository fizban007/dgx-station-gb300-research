"""How long does the GB300 stall on the 6000 per decode step? Reads the peer counters around steady decode runs.

For each (label, bucket override) and concurrency: runs C streams x 320 tokens (short prose prompts), then reports
forward passes (published / 40), wall ms per pass, GB300 wait ms per pass, and aggregate tok/s.
"""
import concurrent.futures as cf
import json
import struct
import sys
import time
import urllib.request

OVERRIDE = "/home/jasonc/research/megamoe/logs/peer_buckets.json"


def words():
    return struct.unpack("<7q", open("/dev/shm/vllm_peer_tier2", "rb").read(56))


def one(i):
    body = {"model": "dsv41-flash-uva", "max_tokens": 320, "temperature": 0, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": f"Write a detailed paragraph about the number {i}, its history and uses. No lists."}]}
    req = urllib.request.Request("http://127.0.0.1:30006/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["usage"]["completion_tokens"]


def run(c, seed):
    a, t = words(), time.time()
    with cf.ThreadPoolExecutor(c) as ex:
        toks = sum(ex.map(one, range(seed, seed + c)))
    dt, b = time.time() - t, words()
    passes = (b[0] - a[0]) / 40
    waits = b[6] - a[6]
    return {"tok_s": round(toks / dt, 1), "passes": int(passes), "ms_per_pass": round(dt * 1e3 / passes, 2),
            "wait_ms_per_pass": round((b[5] - a[5]) / 1e6 / passes, 2), "waits_per_pass": round(waits / passes, 1),
            "wait_us_per_wait": round((b[5] - a[5]) / 1e3 / max(1, waits), 1), "timeouts": b[2] - a[2]}


def main():
    fix = {**{str(r): 32 for r in range(9, 17)}, **{str(r): 1024 for r in range(257, 513)}}
    variants = [("buckets default", None), ("fixed buckets", fix), ("buckets default", None), ("fixed buckets", fix)]
    for label, override in variants:
        if override is None:
            open(OVERRIDE, "w").write("{}")
        else:
            json.dump(override, open(OVERRIDE, "w"))
        time.sleep(4)
        run(4, 50)  # let the sidecar pick up the override (it rereads every ~2 s of served calls)
        for c in (8, 16):
            print(json.dumps({"variant": label, "C": c, **run(c, 2000 + c)}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
