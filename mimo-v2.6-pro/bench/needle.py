"""Needle recall at several prompt lengths (depth 50%), thinking off, greedy. usage: needle.py 8,60,250 (thousands)"""
import json, random, sys, time, urllib.request
B = "http://127.0.0.1:30007/v1/chat/completions"
filler = open("/home/jasonc/research/mimo-pro/launch-mimo26-pro.sh").read() + open("/home/jasonc/research/mimo-pro/knee.sh").read()
for k in [int(x) for x in sys.argv[1].split(",")]:
    random.seed(k)
    body = (filler * (k * 4 * 1000 // len(filler) + 1))[: k * 4000]
    secret = f"{random.randint(100000, 999999)}"
    doc = body[: len(body) // 2] + f"\nThe vault access code is {secret}. Remember it.\n" + body[len(body) // 2:]
    req = {"model": "mimo26-pro", "messages": [{"role": "user", "content": f"[{random.random()}]\n" + doc + "\n\nWhat is the vault access code mentioned above? Reply with just the number."}],
           "max_tokens": 20, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(B, data=json.dumps(req).encode(), headers={"Content-Type": "application/json"}), timeout=3600))
        got = r["choices"][0]["message"]["content"]
        print(f"needle @ {r['usage']['prompt_tokens']:7d} tokens: {'PASS' if secret in got else 'FAIL'} (want {secret}, got {got!r}) in {time.time() - t:.1f}s", flush=True)
    except Exception as e:
        print(f"needle @ ~{k}K: request failed: {e}", flush=True)
