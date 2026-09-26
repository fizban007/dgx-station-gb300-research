#!/usr/bin/env bash
# Concurrency knee sweep against vLLM dsv41 on :30006. Prints agg + per-stream tok/s for C1/2/4/8/12/16.
set -euo pipefail
TAG="${1:-off60}"
python3 - "$TAG" <<'PY'
import json,time,urllib.request,concurrent.futures as cf,sys,statistics
tag=sys.argv[1]
B="http://127.0.0.1:30007/v1/chat/completions"
def one(i,n=192):
    body={"model":"mimo26-pro","messages":[{"role":"user","content":f"Write a detailed paragraph about the number {i}, its history and uses. No lists."}],"max_tokens":n,"temperature":0,"chat_template_kwargs":{"enable_thinking":False},"ignore_eos":True}
    r=json.load(urllib.request.urlopen(urllib.request.Request(B,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=900))
    return r["usage"]["completion_tokens"]
one(999,16)  # warm
out=[]
for c in (1,2,4,8,12,16):
    runs=[]
    for rep in range(2):
        t=time.time()
        with cf.ThreadPoolExecutor(c) as ex: toks=sum(ex.map(one,range(c)))
        dt=time.time()-t; runs.append(toks/dt)
    agg=statistics.mean(runs)
    out.append((c,agg,agg/c))
    print(f"[{tag}] conc={c:2d}: {agg:7.1f} agg tok/s  {agg/c:6.2f} per-stream  (runs {runs[0]:.1f}/{runs[1]:.1f})",flush=True)
json.dump({"tag":tag,"rows":out,"ts":time.strftime("%Y-%m-%dT%H:%M:%S")},open(f"${WORKDIR:-$PWD}/knee-{tag}.json","w"))
PY
