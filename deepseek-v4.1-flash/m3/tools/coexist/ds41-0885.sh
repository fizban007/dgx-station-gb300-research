#!/usr/bin/env bash
# Relaunch the coexistence lane at 0.885, warm it to steady state, restart H3 combined, measure the 15 s render margin.
set -uo pipefail
M=/home/jasonc/research/megamoe; Q=/home/jasonc/research/qwen38; H=/home/jasonc/research/minimax-h3; L=$M/logs/ds41-final
export PORT=30006 MODEL_NAME=dsv41-flash-uva MODEL=dsv41-flash-uva HOST=127.0.0.1
source "$M/lan-idle.sh"
gb() { nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits | awk -F', ' '/Engine/ {e=$2} /Omni/ {o=$2} END {printf "DS41 %.1f GiB, H3 %.1f GiB", e/1024, o/1024}'; }
h3up() { until docker logs minimax-h3 2>&1 | grep -qE "Application startup complete|Traceback"; do sleep 5; done
  docker logs minimax-h3 2>&1 | grep -q "Application startup complete" || echo "  H3 FAILED to start: $(docker logs minimax-h3 2>&1 | grep -E "Error" | tail -1 | cut -c1-140)"; }
wait_idle both "DS41 relaunch at 0.885"
echo "=== $(date +%H:%M) relaunch DS41 at 0.885 (NVFP4 KV)"
"$M/swap-to-ds41-h3.sh" > "$L/swap-0885.log" 2>&1; echo "  swap rc=$?"
docker logs dsv41-flash-megamoe 2>&1 | grep -E "registered host memory|Available KV|GPU KV cache size|nvfp4_ds_mla KV" | sed 's/^.*\] //' | cut -c1-120
echo "=== $(date +%H:%M) warm DS41 (1M needle, 128K prefill, C16 decode)"
THINK_KEY=thinking /home/jasonc/venvs/bench/bin/python "$Q/needle_test.py" ds41-0885-warm-1m 1000000 0.5 2>&1 | grep -E "needle [0-9]" | sed 's/^/  /'
/home/jasonc/venvs/bench/bin/python /home/jasonc/ds41f-exp/bench_prefill.py --engine vllm --host 127.0.0.1 --port 30006 --isl 131072 --concurrency 1 \
  --seed $RANDOM$RANDOM --wait-ready 60 --output "$L/prefill-0885.jsonl" --label warm0885 > /dev/null 2>&1
python3 /home/jasonc/research/glm53-flash/reason_bench.py ds41-0885-warm 16 --window 30 --warmup 10 > /dev/null 2>&1
wait_idle h3 "H3 restart (release cached memory)"
(cd "$H" && H3_MODE=combined OFFLOAD=legacy ./serve-h3-gracie.sh > /dev/null); h3up
echo "  after warm-up: $(gb)"
wait_idle both "15 s combined render check"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i GPU-c146511a-0326-7ddc-4346-998d61a64b34; sleep 1; done > "$L/mem-0885-B-15s.txt" ) & S=$!
(cd "$H" && ./gen.sh ds41-0885-B-15s 30 1024 576 15 "$(cat ref2va/prompt-t2va-astronaut-control.txt)" > "$L/ds41-0885-B-15s.log" 2>&1)
kill $S
echo "  15 s combined: $(grep -oE '^HTTP/1.1 [0-9]+' "$H/out/ds41-0885-B-15s.headers" | tail -1) $(grep -h wall "$L/ds41-0885-B-15s.log") | GB300 peak $(sort -n "$L/mem-0885-B-15s.txt" | tail -1 | awk '{printf "%.1f", $1/1024}') of 249.8 GiB"
echo "  DS41 errors: $(docker logs dsv41-flash-megamoe 2>&1 | grep -ciE 'out of memory|Traceback')"
echo "=== $(date +%H:%M) done"
