#!/usr/bin/env bash
# nvfp4_ds_mla investigation, queued after ds41-final-coexist.sh. Same lane config (GPU_UTIL 0.89, rowmap h265, NVFP4
# Engram, pinned FI tactics) with FP8 vs NVFP4 KV: numerics (teacher-forced + 16K/64K/128K greedy), needles to 1M,
# GSM8K-200, GPQA-Diamond at T=1 and T=0 (each paired with the FP8 run at the same temperature), reasoning decode
# C1/C8/C16, catid prefill.
# H3 is stopped during the NVFP4 boot's tests so nothing else touches the GPU, then restored to the mode it was in.
# LAN clients use this lane: restarts and timed runs wait for idle (lan-idle.sh). If NVFP4 KV fails a quality gate
# (needles 3/3 at each length, GSM8K >= 97%, GPQA within 3 points of the FP8 run), DS41 goes back to FP8 KV.
set -uo pipefail
M=/home/jasonc/research/megamoe; Q=/home/jasonc/research/qwen38; H=/home/jasonc/research/minimax-h3; L=$M/logs/kvnum; mkdir -p "$L"
export PORT=30006 MODEL_NAME=dsv41-flash-uva MODEL=dsv41-flash-uva HOST=127.0.0.1 NUMLOG=$L
while pgrep -f "[d]s41-final-coexist.sh" >/dev/null; do sleep 20; done
source "$M/lan-idle.sh"
H3MODE=$(docker inspect minimax-h3 --format '{{json .Args}}' | grep -q 'combined' && echo combined || echo fl2va-min)
echo "=== $(date +%H:%M) GPQA-Diamond T=0 (FP8 KV, current config; pairs with the 09-30 greedy baseline)"
(cd ~/evals-private/gpqa && /home/jasonc/venvs/bench/bin/python /home/jasonc/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port 30006 \
  --model dsv41-flash-uva --test-profile gpqa-diamond --reasoning-effort max --profile-concurrency 32 --completion-stats-temperature 0.0 \
  --completion-stats-save-text --display-mode plain --no-hw-monitor --output ds41-h265-nvfp4engram-t0-c32-max.json --compare-baseline dsv41-flash-max.json > ds41-h265-nvfp4engram-t0-c32-max.log 2>&1)
python3 -c "import json;d=json.load(open('/home/jasonc/evals-private/gpqa/ds41-h265-nvfp4engram-t0-c32-max.json'));a=d['accuracy'];c=d.get('comparison',{});print('gpqa ds41-h265-nvfp4engram-t0-c32-max:',a['correct'],'/',a['scored'],round(a['accuracy']*100,1),'% hit_max',a['hit_max_tokens'],'| vs baseline flips',c.get('flips_baseline_only_correct'),'/',c.get('flips_candidate_only_correct'),'p',c.get('mcnemar_exact_p'))"
wait_idle ds41 "FP8-KV reference numerics"
echo "=== $(date +%H:%M) FP8-KV reference numerics (H3 idle in $H3MODE)"
python3 "$M/ds41_numerics.py" fp8kv 2>&1 | tail -1 | cut -c1-160
echo "=== $(date +%H:%M) boot NVFP4 KV (stop H3 for clean measurements)"
wait_idle both "DS41 relaunch on NVFP4 KV (H3 stops until the end)"
docker rm -f minimax-h3 >/dev/null 2>&1
(cd "$M" && GPU_UTIL=0.89 ROWMAP=rowmap-mix-v1-h265.json DOCKER_MOUNTS="$(engram-nvfp4/lane-mounts.sh)" EXTRA="--kv-cache-dtype nvfp4_ds_mla" ./swap-to-m3v2.sh > "$L/swap-nvfp4kv.log" 2>&1)
docker logs dsv41-flash-megamoe 2>&1 | grep -E "Available KV|GPU KV cache size|kv_cache_dtype|nvfp4" | sed 's/^.*\] //' | cut -c1-140 | head -5
echo "=== $(date +%H:%M) NVFP4-KV numerics + compare"
python3 "$M/ds41_numerics.py" nvfp4kv 2>&1 | tail -1 | cut -c1-160
python3 "$M/ds41_numerics.py" --compare fp8kv nvfp4kv 2>&1 | sed 's/^/numerics: /'
echo "=== $(date +%H:%M) needles"
for t in 125000 500000 1000000; do THINK_KEY=thinking /home/jasonc/venvs/bench/bin/python "$Q/needle_test.py" "nvfp4kv-$t" "$t" 2>&1 | grep -E "needle [0-9]|FAIL" | tail -3; done | tee "$L/needles-nvfp4kv.txt"
echo "=== $(date +%H:%M) gsm8k-200"; (cd "$M" && python3 gsm8k_eval.py ds41-nvfp4kv 200 2>&1 | tail -1) | tee "$L/gsm8k-nvfp4kv.txt"
wait_idle ds41 "reasoning decode benchmark"
echo "=== $(date +%H:%M) reasoning decode"
python3 /home/jasonc/research/glm53-flash/reason_bench.py ds41-nvfp4kv 1,8,16 --window 90 --warmup 20 2>&1 | sed 's/^/speed: /'
wait_idle ds41 "prefill benchmark"
echo "=== $(date +%H:%M) catid prefill"
/home/jasonc/venvs/bench/bin/python /home/jasonc/ds41f-exp/bench_prefill.py --engine vllm --host 127.0.0.1 --port 30006 --isl 16384,65536,131072 \
  --concurrency 1 --seed $RANDOM$RANDOM --wait-ready 60 --output "$L/prefill-nvfp4kv.jsonl" --label nvfp4kv 2>&1 | grep -E "^\s+[0-9]+ +1 " | sed 's/^/prefill: /'
echo "=== $(date +%H:%M) GPQA-Diamond T=1 (NVFP4 KV)"
(cd ~/evals-private/gpqa && /home/jasonc/venvs/bench/bin/python /home/jasonc/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port 30006 \
  --model dsv41-flash-uva --test-profile gpqa-diamond --reasoning-effort max --profile-concurrency 32 --completion-stats-temperature 1.0 \
  --completion-stats-top-p 0.95 --completion-stats-seed 1101 --completion-stats-save-text --display-mode plain --no-hw-monitor \
  --output ds41-nvfp4kv-t1-c32-max.json --compare-baseline ds41-h265-nvfp4engram-t1-c32-max.json > ds41-nvfp4kv-t1-c32-max.log 2>&1)
python3 -c "import json;d=json.load(open('/home/jasonc/evals-private/gpqa/ds41-nvfp4kv-t1-c32-max.json'));a=d['accuracy'];print('gpqa nvfp4kv:',a['correct'],'/',a['scored'],round(a['accuracy']*100,1),'% hit_max',a['hit_max_tokens'])"
echo "=== $(date +%H:%M) GPQA-Diamond T=0 (NVFP4 KV)"
(cd ~/evals-private/gpqa && /home/jasonc/venvs/bench/bin/python /home/jasonc/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port 30006 \
  --model dsv41-flash-uva --test-profile gpqa-diamond --reasoning-effort max --profile-concurrency 32 --completion-stats-temperature 0.0 \
  --completion-stats-save-text --display-mode plain --no-hw-monitor --output ds41-nvfp4kv-t0-c32-max.json --compare-baseline ds41-h265-nvfp4engram-t0-c32-max.json > ds41-nvfp4kv-t0-c32-max.log 2>&1)
python3 -c "import json;d=json.load(open('/home/jasonc/evals-private/gpqa/ds41-nvfp4kv-t0-c32-max.json'));a=d['accuracy'];c=d.get('comparison',{});print('gpqa ds41-nvfp4kv-t0-c32-max:',a['correct'],'/',a['scored'],round(a['accuracy']*100,1),'% hit_max',a['hit_max_tokens'],'| vs baseline flips',c.get('flips_baseline_only_correct'),'/',c.get('flips_candidate_only_correct'),'p',c.get('mcnemar_exact_p'))"
NEEDLES_OK=$(grep -cE "needle 3/3" "$L/needles-nvfp4kv.txt")
GSM=$(grep -oE "= [0-9.]+%" "$L/gsm8k-nvfp4kv.txt" | tr -dc "0-9.")
GPQA_DROP=$(python3 -c "import json;f=lambda p:json.load(open(p))['accuracy']['accuracy'];print(round(100*(f('/home/jasonc/evals-private/gpqa/ds41-h265-nvfp4engram-t1-c32-max.json')-f('/home/jasonc/evals-private/gpqa/ds41-nvfp4kv-t1-c32-max.json')),1))" 2>/dev/null || echo 99)
echo "gates: needles ${NEEDLES_OK}/3 lengths, GSM8K ${GSM:-?}%, GPQA drop vs FP8 ${GPQA_DROP} pts"
if [ "$NEEDLES_OK" = 3 ] && python3 -c "import sys; sys.exit(0 if float('${GSM:-0}') >= 97 and float('$GPQA_DROP') <= 3 else 1)"; then
  echo "=== $(date +%H:%M) NVFP4 KV passed the gates: DS41 stays on it pending your decision"
else
  wait_idle ds41 "revert DS41 to FP8 KV"
  echo "=== $(date +%H:%M) NVFP4 KV failed a gate: relaunching DS41 on FP8 KV"
  (cd "$M" && GPU_UTIL=0.89 ROWMAP=rowmap-mix-v1-h265.json DOCKER_MOUNTS="$(engram-nvfp4/lane-mounts.sh)" ./swap-to-m3v2.sh > "$L/swap-fp8kv-revert.log" 2>&1)
  docker logs dsv41-flash-megamoe 2>&1 | grep -E "GPU KV cache size" | sed 's/^.*\] //' | cut -c1-120
fi
echo "=== $(date +%H:%M) restore H3 ($H3MODE)"
if [ "$H3MODE" = combined ]; then (cd "$H" && H3_MODE=combined OFFLOAD=legacy ./serve-h3-gracie.sh > /dev/null); else (cd "$H" && H3_MODE=fl2va-min ./serve-h3-gracie.sh > /dev/null); fi
echo "=== $(date +%H:%M) done"
