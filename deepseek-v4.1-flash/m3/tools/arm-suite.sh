#!/usr/bin/env bash
# One arm of the 2026-09-29 DS41F A/B (32 seats, pinned FlashInfer tactics, Mega-mHC on/off), against :30006.
# fid-suite (agent fixture per-class acceptance + teacher-forced flips vs the J-M-Recipes 0909 reference), GSM8K-200
# gate, Al-ENGR prose knee C1-C16, catid decode C1/4/8/16/24/32, closed-loop reasoning C1/8/16/24/32, catid 16K
# prefill, real-text 16K/64K prefill, sidecar per-bucket latency. DSpark acceptance per phase from /metrics deltas.
# Non-local requests seen during the window are counted in env.txt; a nonzero count means the timings are suspect.
# Usage: arm-suite.sh <tag>
set -uo pipefail
TAG=${1:?tag}
D=/home/jasonc/research/megamoe
OUT=$D/logs/arm-$TAG
mkdir -p "$OUT"
export PORT=30006 MODEL_NAME=dsv41-flash-uva
T0=$(date -u +%Y-%m-%dT%H:%M:%SZ)
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
  echo "tag $TAG start $T0 image $(docker inspect -f '{{.Config.Image}}' dsv41-flash-megamoe)"
  docker inspect dsv41-flash-megamoe --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^MEGA_' | tr '\n' ' '; echo
  docker inspect dsv41-flash-megamoe --format '{{json .Args}}' | grep -oE 'max-num-seqs","[0-9]+|num_speculative_tokens_per_batch_size[^]]*\]\]'
  docker logs dsv41-flash-megamoe 2>&1 | grep -E 'FlashInfer autotune pinned|Saved [0-9]+ configs|Loaded [0-9]+ configs|Mega-mHC off|GPU KV cache size' | cut -c1-260
  nvidia-smi --query-gpu=uuid,name,pstate,clocks.sm,clocks.mem,clocks_throttle_reasons.active,power.draw --format=csv,noheader
} | tee "$OUT/env.txt"
bash "$D/fid-suite.sh" "$TAG" > "$OUT/fid.log" 2>&1; grep -E "accept=|agent |heldout |prose |non-local" "$OUT/fid.log"
phase gsm8k python3 "$D/gsm8k_eval.py" "arm-$TAG" 200
phase knee bash /home/jasonc/research/upstream/knee.sh "arm-$TAG"
phase catid /home/jasonc/ds41f-exp/bench_decode.sh "arm-$TAG" "1 4 8 16 24 32"
phase reason python3 "$D/reason_bench.py" "arm-$TAG" 1,8,16,24,32
phase prefill /home/jasonc/ds41f-exp/bench_prefill.sh "arm-$TAG" 16384 1
phase realtext python3 "$D/realtext_prefill.py"
cp "$D/logs/peer_stats.json" "$OUT/peer_stats.json" 2>/dev/null
OTHER=$(docker logs --since "$T0" dsv41-flash-megamoe 2>&1 | grep -E '"(POST|GET) ' | grep -vc '127.0.0.1')
echo "non-local requests during window: $OTHER" | tee -a "$OUT/env.txt"
nvidia-smi --query-gpu=uuid,pstate,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader | tee -a "$OUT/env.txt"
echo "arm suite done $TAG"
