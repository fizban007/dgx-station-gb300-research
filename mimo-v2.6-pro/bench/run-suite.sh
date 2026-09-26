#!/usr/bin/env bash
# Post-boot suite for the Pro lane (:30007): smoke, GSM8K-200 (quality gate first), Al-ENGR ttft_bench (C1 decode +
# prefill), knee (C1-C16), peer-wait counters. Usage: run-suite.sh <tag>; logs in logs/<step>-<tag>.log
set -uo pipefail
TAG=${1:?tag}
D=/home/jasonc/research/mimo-pro
cd $D
curl -s localhost:30007/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"mimo26-pro","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":50,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('smoke:', repr(d['choices'][0]['message']['content']))"
(cd /home/jasonc/research/megamoe && PORT=30007 MODEL_NAME=mimo26-pro THINK_KEY=enable_thinking /home/jasonc/venvs/bench/bin/python gsm8k_eval.py mimo-pro-$TAG 200) 2>&1 | tail -1 | tee logs/gsm8k-$TAG.log
WORKDIR=$D /home/jasonc/venvs/bench/bin/python bench/ttft_bench.py $TAG 2>&1 | grep "^\[" | tee logs/ttft-$TAG.log
WORKDIR=$D bash bench/knee.sh $TAG 2>&1 | grep "conc=" | tee logs/knee-$TAG.log
/home/jasonc/venvs/bench/bin/python bench/peer_wait.py 1,4,8,16 256 2>&1 | tee logs/peerwait-$TAG.log
echo "suite $TAG done"
