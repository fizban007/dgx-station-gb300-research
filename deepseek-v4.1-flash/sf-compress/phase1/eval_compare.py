"""Greedy outputs + C1 decode speed from the running server, for A/B of MEGA_SF_COMPRESS.

  python eval_compare.py --out baseline.json        # record
  python eval_compare.py --out sfc.json --compare baseline.json
"""
import argparse
import json
import os
import time
import urllib.request

URL = os.environ.get("SERVER", "http://localhost:8001") + "/v1/chat/completions"
PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips "
    "did Natalia sell altogether in April and May? Think step by step.",
    "Write a Python function that returns the n-th Fibonacci number using matrix exponentiation, with a docstring.",
    "Explain in three paragraphs why the sky is blue, mentioning Rayleigh scattering.",
    "Translate into French: 'The quick brown fox jumps over the lazy dog while the farmer watches from the barn.'",
    "List the first 15 prime numbers, then compute their sum.",
    "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. When does it arrive? Answer briefly.",
]


def chat(content, max_tokens, effort=25, ignore_eos=False):
    body = {"model": "local-model", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"reasoning_effort": effort}}
    if ignore_eos:
        body["ignore_eos"] = True
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900))
    dt = time.time() - t
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or m.get("reasoning") or "") + "\n---\n" + (m.get("content") or ""), \
        r["usage"]["completion_tokens"], dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--compare")
    p.add_argument("--repeats", type=int, default=2)
    a = p.parse_args()
    res = {"outputs": [], "repeat_identical": [], "speed": []}
    for prompt in PROMPTS:
        runs = [chat(prompt, 400)[0] for _ in range(a.repeats)]
        res["outputs"].append(runs[0])
        res["repeat_identical"].append(all(r == runs[0] for r in runs))
    for _ in range(3):
        _, n, dt = chat("Write a long story about a lighthouse keeper.", 1024, ignore_eos=True)
        res["speed"].append(n / dt)
    print("repeat-identical per prompt:", res["repeat_identical"])
    print("C1 decode tok/s (1024 forced tokens, incl. prefill):", [f"{s:.1f}" for s in res["speed"]])
    if a.compare:
        base = json.load(open(a.compare))
        same = [x == y for x, y in zip(res["outputs"], base["outputs"])]
        print("identical to baseline per prompt:", same)
        for i, (x, y) in enumerate(zip(res["outputs"], base["outputs"])):
            if x != y:
                k = next(j for j in range(min(len(x), len(y))) if x[j] != y[j]) if x[:len(y)] != y[:len(x)] else min(len(x), len(y))
                print(f"  prompt {i}: first difference at char {k}: base {y[k:k + 60]!r} vs new {x[k:k + 60]!r}")
        print(f"speed vs baseline: {sum(res['speed']) / sum(base['speed']):.3f}x")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
