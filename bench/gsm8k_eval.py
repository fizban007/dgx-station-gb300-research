"""Greedy GSM8K accuracy on the last N test questions (thinking off), 16 in flight. gsm8k_eval.py <label> [N]"""
import concurrent.futures as cf
import json
import os
import re
import sys
import urllib.request

PORT = os.environ.get("PORT", "30006")
HOST = os.environ.get("HOST", "127.0.0.1")
MODEL = os.environ.get("MODEL_NAME", "dsv41-flash-uva")
THINK_KEY = os.environ.get("THINK_KEY", "thinking")
label, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 200
rows = [json.loads(l) for l in open("/home/jasonc/llm-inference-bench/data/gsm8k_test.jsonl")][-n:]


def number(text):
    found = re.findall(r"-?\d[\d,]*\.?\d*", text.replace("$", ""))
    return found[-1].replace(",", "").rstrip(".") if found else None


def one(row):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 768,
            "chat_template_kwargs": {THINK_KEY: False},
            "messages": [{"role": "user", "content": row["question"] + "\n\nSolve step by step, then finish with 'The answer is <number>.'"}]}
    req = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"] or ""
    gold = row["answer"].split("####")[-1].strip().replace(",", "")
    got = number(text.split("answer is")[-1]) if "answer is" in text else number(text)
    try:
        return float(got) == float(gold)
    except (TypeError, ValueError):
        return False


with cf.ThreadPoolExecutor(16) as ex:
    results = list(ex.map(one, rows))
acc = sum(results) / len(results)
print(f"{label}: GSM8K {sum(results)}/{len(results)} = {100 * acc:.1f}%")
json.dump({"label": label, "correct": sum(results), "n": len(results), "per_question": results},
          open(f"/home/jasonc/research/megamoe/logs/gsm8k-{label}.json", "w"))
