#!/usr/bin/env bash
# Determinism + cross-check probe: the same greedy prompt 5x, then one real 2K-token prefill with MEGA_PEER_CHECK on.
for i in 1 2 3 4 5; do curl -s http://127.0.0.1:30006/v1/completions -H 'Content-Type: application/json' -d '{"model":"dsv41-flash-uva","prompt":"The capital of France is","max_tokens":10,"temperature":0,"logprobs":2}' | python3 -c "import json,sys;r=json.load(sys.stdin)['choices'][0];print(repr(r['text']), r['logprobs']['top_logprobs'][0])"; done
python3 - <<'PY'
import json, urllib.request
text = open('/home/jasonc/research/FINDINGS.md').read()
body = {"model": "dsv41-flash-uva", "prompt": "REALPROMPT " + text[:6000], "max_tokens": 1, "temperature": 0}
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30006/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=300).read()
PY
sleep 1
docker logs m3v2 2>&1 | grep "MEGA_PEER check" | grep "T=2043" | awk '{for(i=1;i<=NF;i++) if($i ~ /^(layer|rows|cos)=/) printf "%s ", $i; print ""}' | awk 'NR%5==1'
