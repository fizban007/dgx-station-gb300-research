"""Boot warm-up: one prefill at every 16th batch size so DeepGEMM compiles each kernel variant before real traffic.

DeepGEMM picks a block configuration from the token count and JIT-compiles each new one (1.7-2.4 s, cached in
/root/.dj, mounted from jit-cache/dj). Single random-id prompts of L tokens give forward passes of L tokens.
  warmup.py            sweep, then check 40 unseen sizes for new compiles
  warmup.py --check    only the check
"""
import json
import os
import random
import sys
import time
import urllib.request

PORT = os.environ.get("PORT", "30006")
CACHE = "/home/jasonc/research/megamoe/jit-cache/dj/cache"
rng = random.Random()


def prefill(n):
    body = {"model": "dsv41-flash-uva", "prompt": [rng.randrange(1000, 120000) for _ in range(n)],
            "max_tokens": 1, "temperature": 0}
    t = time.time()
    urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(),
                                                  headers={"Content-Type": "application/json"}), timeout=600).read()
    return time.time() - t


def cached():
    return len(os.listdir(CACHE)) if os.path.isdir(CACHE) else 0


def main():
    if "--check" not in sys.argv:
        before, t0, slow = cached(), time.time(), []
        for n in list(range(1, 16)) + list(range(16, 8193, 16)):
            dt = prefill(n)
            if dt > 1.0 + n / 20000:
                slow.append((n, round(dt, 2)))
        print(f"sweep: {time.time() - t0:.0f}s, {cached() - before} new DeepGEMM kernels, "
              f"{len(slow)} slow sizes {slow[:12]}", flush=True)
    before = cached()
    sizes = sorted(rng.sample(range(1, 8193), 40))
    times = [(n, prefill(n)) for n in sizes]
    worst = max(times, key=lambda x: x[1] - x[0] / 20000)
    print(f"check: 40 unseen sizes, {cached() - before} new DeepGEMM kernels, "
          f"slowest {worst[0]} tokens in {worst[1]:.2f}s", flush=True)


if __name__ == "__main__":
    main()
