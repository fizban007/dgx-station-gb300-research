"""Phase A: torch-profile ~20 engine steps of steady decode at C1/C8/C16 and snapshot the 6000's per-bucket latency.

Server launched with PROF=1 PROF_ITERS=20. Writes traces to prof/ (container /prof) and logs/peer_stats_c<C>.json.
"""
import concurrent.futures as cf
import json
import os
import shutil
import time
import urllib.request

BASE = "http://127.0.0.1:30006"
LOGS = "/home/jasonc/research/megamoe/logs"
PROF = "/home/jasonc/research/megamoe/prof"


def post(path, body=None, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def one(i):
    body = {"model": "dsv41-flash-uva", "max_tokens": 320, "temperature": 0, "ignore_eos": True,
            "chat_template_kwargs": {"thinking": False},
            "messages": [{"role": "user", "content": f"Write a detailed paragraph about the number {i}, its history and uses. No lists."}]}
    return json.loads(post("/v1/chat/completions", body))["usage"]["completion_tokens"]


def main():
    for c in (1, 8, 16):
        before = set(os.listdir(PROF))
        with cf.ThreadPoolExecutor(c) as ex:
            futs = [ex.submit(one, 1000 + c * 100 + i) for i in range(c)]
            time.sleep(4)  # past prefill, into steady decode
            open(f"{LOGS}/peer_stats.reset", "w").close()
            time.sleep(3)
            post("/start_profile")
            time.sleep(4)
            post("/stop_profile")
            time.sleep(3)
            shutil.copy(f"{LOGS}/peer_stats.json", f"{LOGS}/peer_stats_c{c}.json")
            toks = sum(f.result() for f in futs)
        time.sleep(15)
        new = sorted(set(os.listdir(PROF)) - before)
        print(f"C{c}: {toks} tokens; new traces {new}", flush=True)


if __name__ == "__main__":
    main()
