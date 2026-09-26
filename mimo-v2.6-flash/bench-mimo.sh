#!/usr/bin/env bash
# MiMo-V2.6-Flash suite on one GB300, same shape as research/qwen38/bench-qwen.sh:
# GSM8K-200 (thinking off), catid decode (8K in / 1K out, T=0, C warm-ups + 5xC) at CS, cold prefill C1 8K-128K
# (fresh random seed, so prefix-cache hits can't inflate it).
# Usage: bench-mimo.sh <label> ["1 2 4 8 16 32 64"]
set -uo pipefail
LABEL=${1:?label}; CS=${2:-1 2 4 8 16 32 64}
D=/home/jasonc/research/mimo26
export PORT=30006 MODEL_NAME=mimo-v26-flash
cd /home/jasonc/research/megamoe
m0=$(python3 $D/spec-metrics.py 2>/dev/null || echo '{}')
THINK_KEY=enable_thinking /home/jasonc/venvs/bench/bin/python gsm8k_eval.py "mimo-$LABEL" 200
m1=$(python3 $D/spec-metrics.py 2>/dev/null || echo '{}')
python3 -c "import json,sys;a,b=json.loads(sys.argv[1]),json.loads(sys.argv[2])
d={k:b.get(k,0)-a.get(k,0) for k in b}
print('GSM8K spec: drafts=%d accepted/draft=%.2f accept_rate=%.1f%%'%(d['num_drafts'],d['num_accepted_tokens']/d['num_drafts'],100*d['num_accepted_tokens']/d['num_draft_tokens']) if d.get('num_drafts') else 'GSM8K spec: none')" "$m0" "$m1"
/home/jasonc/research/mimo26/bench_decode_seeded.sh "mimo-$LABEL" "$CS" 2>&1 | grep -E "^decode|rc=|^ *C[0-9]|tok/s"
source /home/jasonc/venvs/bench/bin/activate
python /home/jasonc/ds41f-exp/bench_prefill.py --engine vllm --host 127.0.0.1 --port 30006 --isl 8192,32768,65536,131072 \
  --concurrency 1 --seed $RANDOM$RANDOM --wait-ready 60 --output "$D/logs/prefill-$LABEL.jsonl" --label "$LABEL" 2>&1 \
  | grep -E "^\s+[0-9]+ +1 "
