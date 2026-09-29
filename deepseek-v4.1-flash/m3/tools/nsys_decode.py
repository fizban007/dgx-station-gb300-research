"""Nsight Systems ranges of steady reasoning decode at C1 and C8 (one report each).

Server launched with PROF=nsys (vLLM "cuda" profiler: PROF_DELAY iterations after /start_profile, then PROF_ITERS
iterations, then cudaProfilerStop). With PROF_END=repeat-shutdown:2 the server exits after the second range and
nsys writes prof/<PROF_NAME>-<n>.nsys-rep.
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30006"
rows = [json.loads(l) for l in open("/home/jasonc/llm-inference-bench/data/gsm8k_test.jsonl")][100:140]


def post(path, body=None, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def one(i):
    body = {"model": "dsv41-flash-uva", "max_tokens": 1536, "temperature": 0, "ignore_eos": True,
            "messages": [{"role": "user", "content": rows[i]["question"]}]}
    return json.loads(post("/v1/chat/completions", body))["usage"]["completion_tokens"]


for n, c in enumerate((1, 8)):
    with cf.ThreadPoolExecutor(c) as ex:
        futs = [ex.submit(one, n * 10 + i) for i in range(c)]
        time.sleep(3)  # past prefill, into steady decode
        post("/start_profile")
        print(f"C{c}: profiling started", flush=True)
        try:
            print(f"C{c}: {sum(f.result() for f in futs)} tokens", flush=True)
        except Exception as exc:  # the server shuts down after the last range
            print(f"C{c}: requests ended with {exc!r}", flush=True)
    time.sleep(5)
sys.exit(0)
