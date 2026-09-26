#!/usr/bin/env bash
# Qwen3.8-Flash-Next baseline suite, comparable to catid's one-GB300 page:
# GSM8K-200 (thinking off), catid decode (8K in / 1K out, T=0, C warm-ups + 5xC) C1-C64, cold prefill C1 8K-128K.
# ENGINE=sglang sends prefill through SGLang's native /generate (vLLM and default: /v1/completions).
set -uo pipefail
LABEL=${1:?label}
D=/home/jasonc/research/qwen38
export PORT=30006 MODEL_NAME=qwen38-flash-next
cd /home/jasonc/research/megamoe
THINK_KEY=enable_thinking /home/jasonc/venvs/bench/bin/python gsm8k_eval.py "qwen-$LABEL" 200
/home/jasonc/ds41f-exp/bench_decode.sh "qwen-$LABEL" "1 2 4 8 16 32 64" 2>&1 | grep -E "^decode|rc="
source /home/jasonc/venvs/bench/bin/activate
python /home/jasonc/ds41f-exp/bench_prefill.py --engine ${ENGINE:-vllm} --host 127.0.0.1 --port 30006 --isl 8192,32768,65536,131072 \
  --concurrency 1 --seed $RANDOM$RANDOM --wait-ready 60 --output "$D/logs/prefill-$LABEL.jsonl" --label "$LABEL" 2>&1 \
  | grep -E "^\s+[0-9]+ +1 "
