#!/usr/bin/env bash
# Profile M3 prefill (server launched with PROF=1). Two traces into /home/jasonc/research/megamoe/prof:
#  1. a warm 16K prefill (after a same-length warm-up): per-chunk breakdown, hot MegaMoE vs cold TRT from Grace;
#  2. the first request at an unseen length (90K): the one-time stall seen at 32K/64K/100K.
set -euo pipefail
PORT=${PORT:-30006}
OUT=/home/jasonc/research/megamoe/prof
source /home/jasonc/venvs/bench/bin/activate
req() {  # req <tokens> <seed>: one random-token prompt, one output token; prints TTFT seconds
  python3 - "$1" "$2" <<'PY'
import json, random, sys, time, urllib.request
n, seed = int(sys.argv[1]), int(sys.argv[2])
rng = random.Random(seed)
body = {"model": "dsv41-flash-uva", "prompt": [rng.randrange(1000, 120000) for _ in range(n)],
        "max_tokens": 1, "temperature": 0}
t = time.time()
urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{__import__('os').environ.get('PORT','30006')}/v1/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read()
print(f"{n} tokens: {time.time() - t:.3f} s")
PY
}
prof() { curl -s -X POST "http://127.0.0.1:$PORT/$1_profile" >/dev/null; }

req 16384 1
req 16384 2
prof start; req 16384 3; prof stop
sleep 20
ls -t "$OUT" | head -2
prof start; req 92160 4; prof stop
sleep 30
req 92160 5
ls -t "$OUT" | head -3
