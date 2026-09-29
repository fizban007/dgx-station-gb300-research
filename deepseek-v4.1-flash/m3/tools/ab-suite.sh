#!/usr/bin/env bash
# A/B suite for DS41F lane changes (e.g. image upgrades), run identically on each arm against :30006:
# GSM8K-200 gate (thinking off), Al-ENGR knee (thinking off prose), catid decode C1/4/8/16 (chat, thinking on at the
# server's default effort), GB300 wait on the 6000 at C1/2/4, reasoning acceptance at C1, catid prefill 16K C1.
# DSpark acceptance per phase comes from /metrics deltas. Pin the same numeric effort on both arms (REASONING_EFFORT).
# Usage: ab-suite.sh <tag>
set -uo pipefail
TAG=${1:?tag}
D=/home/jasonc/research/megamoe
OUT=$D/logs/ab-$TAG
mkdir -p "$OUT"
export PORT=30006 MODEL_NAME=dsv41-flash-uva
snap() { curl -s -m 10 http://127.0.0.1:30006/metrics | grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)' > "$OUT/$1.metrics"; }
phase() {
  local name=$1; shift
  echo "=== $name $(date +%H:%M:%S)"
  snap "$name.before"
  "$@" 2>&1 | tee "$OUT/$name.log"
  snap "$name.after"
  python3 "$D/accept.py" "$OUT/$name.before.metrics" "$OUT/$name.after.metrics" | tee -a "$OUT/$name.log"
}
{
  echo "tag $TAG image $(docker inspect -f '{{.Config.Image}}' dsv41-flash-megamoe) $(date -Is)"
  echo "effort: $(docker inspect dsv41-flash-megamoe --format '{{json .Args}}' | grep -o 'reasoning_effort[^}]*')"
  nvidia-smi --query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,clocks_throttle_reasons.active,power.draw --format=csv,noheader
} | tee "$OUT/env.txt"
phase gsm8k python3 "$D/gsm8k_eval.py" "ab-$TAG" 200
phase knee bash /home/jasonc/research/upstream/knee.sh "ab-$TAG"
phase catid /home/jasonc/ds41f-exp/bench_decode.sh "ab-$TAG" "1 4 8 16"
phase wait python3 "$D/measure_wait_c1.py"
phase reason python3 "$D/reason_accept.py"
phase prefill /home/jasonc/ds41f-exp/bench_prefill.sh "ab-$TAG" 16384 1
nvidia-smi --query-gpu=uuid,pstate,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader | tee -a "$OUT/env.txt"
echo "ab suite done $TAG"
