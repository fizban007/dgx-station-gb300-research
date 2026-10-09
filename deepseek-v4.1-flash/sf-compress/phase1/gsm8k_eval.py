"""GSM8K accuracy on the first N test questions against the running server (concurrent requests).

  python gsm8k_eval.py --n 200 --out gsm8k-base.json
"""
import argparse, json, os, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

URL = os.environ.get("SERVER", "http://localhost:8001") + "/v1/chat/completions"
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def ask(q, effort, max_tokens):
    body = {"model": "local-model", "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"reasoning_effort": effort},
            "messages": [{"role": "user", "content": q + "\nGive the final answer as a number on the last line, "
                                                         "after '####'."}]}
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800))
    return r["choices"][0]["message"].get("content") or "", r["usage"]["completion_tokens"]


def last_number(s):
    tail = s.split("####")[-1]
    m = NUM.findall(tail) or NUM.findall(s)
    return m[-1].replace(",", "").rstrip(".") if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--effort", type=int, default=50)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t = pq.read_table(os.environ.get("GSM8K_PARQUET", os.path.expanduser("~/data/gsm8k/main/test-00000-of-00001.parquet"))).to_pylist()[:a.n]
    gold = [r["answer"].split("####")[-1].strip().replace(",", "") for r in t]
    t0 = time.time()
    with ThreadPoolExecutor(a.concurrency) as ex:
        outs = list(ex.map(lambda r: ask(r["question"], a.effort, a.max_tokens), t))
    dt = time.time() - t0
    pred = [last_number(o) for o, _ in outs]
    correct = [p_ is not None and float(p_) == float(g) for p_, g in zip(pred, gold)]
    toks = sum(n for _, n in outs)
    print(f"GSM8K-{a.n}: {sum(correct)}/{a.n} = {100 * sum(correct) / a.n:.1f}%  ({toks} completion tokens in "
          f"{dt:.0f}s, {toks / dt:.0f} tok/s aggregate at C{a.concurrency})")
    json.dump({"correct": correct, "pred": pred, "gold": gold, "outputs": [o for o, _ in outs]}, open(a.out, "w"))


if __name__ == "__main__":
    main()
