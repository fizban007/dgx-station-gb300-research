#!/usr/bin/env bash
# NOTE: the measured runs used a box-local path in the tool_json prompt; it is scrubbed here (a few prompt tokens differ).
# Hermes-traffic fixture: agent-shaped prompts (tool JSON, code, shell, structured reasoning, short prose).
# Measures tok/s per category and DSpark acceptance via /metrics deltas. Usage: bash agent_fixture.sh <tag>
set -euo pipefail
TAG="${1:-run}"
python3 - "$TAG" <<'PY'
import json,time,urllib.request,sys,statistics
tag=sys.argv[1]; B="http://127.0.0.1:30007"
def metrics():
    t=urllib.request.urlopen(B+"/metrics",timeout=10).read().decode()
    d={}
    for l in t.splitlines():
        if l.startswith("vllm:spec_decode_num_draft_tokens_total"): d["draft"]=float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_accepted_tokens_total"): d["acc"]=float(l.split()[-1])
        elif l.startswith("vllm:spec_decode_num_drafts_total"): d["steps"]=float(l.split()[-1])
        elif l.startswith("vllm:generation_tokens_total"): d["gen"]=float(l.split()[-1])
    for k in ("draft","acc","steps","gen"):
        d.setdefault(k, 0.0)
    return d
SYS="You are Hermes, an autonomous agent. You have tools: terminal(command), read_file(path), write_file(path,content), web_search(query), patch(path,old,new). When you need a tool, emit a tool call. Be terse."
TOOLS=[{"type":"function","function":{"name":"terminal","parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
       {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
       {"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
CATS={
 "tool_json":[("List the docker containers on host <station-ip> and check free disk on /. Use tools.",120),
              ("Read /srv/lab/launch-dsv41-vllm.sh then write a copy to /tmp/l2.sh with port 30007. Use tools; call read_file first.",120)],
 "code":[("Write a Python function `knee(rows)` that takes [(conc, agg_tps)] and returns the concurrency where marginal gain per added stream first drops below 5 tok/s. Include a docstring and a 3-line example. Code only.",256),
         ("Write a bash script that loops over shards model-00001..00048-of-00048.safetensors, checks each exists in $SRC, and prints missing ones. Code only.",200)],
 "shell_ops":[("Give the exact sequence of shell commands to: stop docker container dsv41-vllm-v7, rename it with suffix -old, drop page caches with sudo, and relaunch via launch-dsv41-vllm.sh with OFFGB=70 SEQS=16. Commands only, one per line.",160)],
 "structured":[("Compare vLLM UVA expert offload vs SGLang OffloaderV2 whole-layer copy for a 384-expert top-6 MoE. Output a markdown table with columns: approach, bytes moved per token, bottleneck, expected tok/s on a ~330 GB/s link. Then one sentence of verdict.",220),
               ("Status report format: 5 bullet points, each '<component>: <state> — <one metric>'. Components: weights load, KV cache, autotune, graph capture, API bind. Invent plausible values for a vLLM boot.",160)],
 "prose":[("In one paragraph, explain to a smart non-engineer why a model that is fast at counting can be slow at writing prose when speculative decoding is on.",200)],
}
def call(msgs,n,tools=None):
    body={"model":"mimo26-pro","messages":msgs,"max_tokens":n,"temperature":0,"chat_template_kwargs":{"enable_thinking":False}}
    if tools: body["tools"]=tools; body["tool_choice"]="auto"
    t=time.time()
    r=json.load(urllib.request.urlopen(urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600))
    dt=time.time()-t; u=r["usage"]; ch=r["choices"][0]
    return u["completion_tokens"],dt,ch["finish_reason"],(ch["message"].get("content") or "")[:80].replace("\n"," "),bool(ch["message"].get("tool_calls"))
call([{"role":"user","content":"hi"}],8)
res={}
for cat,items in CATS.items():
    m0=metrics(); toks=0; secs=0; notes=[]
    for _ in range(2):
        for p,n in items:
            k,dt,fr,head,tc=call([{"role":"system","content":SYS},{"role":"user","content":p}],n,TOOLS if cat=="tool_json" else None)
            toks+=k; secs+=dt; notes.append((k,round(dt,2),fr,tc,head))
    m1=metrics()
    dr=m1["draft"]-m0["draft"]; ac=m1["acc"]-m0["acc"]; st=m1["steps"]-m0["steps"]; gen=m1["gen"]-m0["gen"]
    acc_rate=ac/dr if dr else float("nan"); tps=toks/secs
    res[cat]={"tok_s":round(tps,1),"accept":round(acc_rate,3),"acc_per_step":round(ac/st,2) if st else None,"tokens":toks,"gen_metric":gen,"samples":notes}
    print(f"[{tag}] {cat:11s} {tps:6.1f} tok/s  accept={acc_rate:5.1%}  acc/step={ac/st if st else 0:4.2f}  tokens={toks}",flush=True)
    for s in notes[:len(items)]: print("     ",s)
allacc=sum(r["accept"]*r["tokens"] for r in res.values())/sum(r["tokens"] for r in res.values())
print(f"[{tag}] weighted accept={allacc:.1%}")
json.dump({"tag":tag,"ts":time.strftime("%Y-%m-%dT%H:%M:%S"),"res":res,"weighted_accept":allacc},open(f"${WORKDIR:-$PWD}/agentfix-{tag}.json","w"),indent=1)
PY
