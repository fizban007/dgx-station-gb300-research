#!/usr/bin/env bash
# After the GPQA run: relaunch DS41 at GPU_UTIL 0.89 (rowmap h265, NVFP4 Engram, FP8 KV), warm it to its grown steady
# state (1M-token needle + 128K prefill + C16 decode), then coexistence tests at H3's worst case:
#   A: H3 fl2va-min (lean, no voice cloning): 15 s t2va render with DS41 C1 decoding
#   B: H3 combined + legacy layerwise offload (VAE staging, voice cloning): 15 s t2va + 6 s ref2va renders
# H3 is left in the best mode that passed. LAN clients use this lane (since 21:10), so every restart waits for idle.
set -uo pipefail
M=/home/jasonc/research/megamoe; Q=/home/jasonc/research/qwen38; H=/home/jasonc/research/minimax-h3; L=$M/logs/ds41-final; mkdir -p "$L"
export PORT=30006 MODEL_NAME=dsv41-flash-uva MODEL=dsv41-flash-uva HOST=127.0.0.1
while pgrep -f "[l]lm_decode_bench.py .*ds41-h265" >/dev/null || pgrep -f "[d]s41-gpqa-after-test.sh" >/dev/null; do sleep 20; done
source "$M/lan-idle.sh"
gb() { nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits | awk -F', ' '/Engine/ {e=$2} /Omni/ {o=$2} END {printf "DS41 %.1f GiB, H3 %.1f GiB", e/1024, o/1024}'; }
h3up() { until docker logs minimax-h3 2>&1 | grep -qE "Application startup complete|Traceback"; do sleep 5; done
  docker logs minimax-h3 2>&1 | grep -q "Application startup complete" || echo "  H3 FAILED to start: $(docker logs minimax-h3 2>&1 | grep -E "Error" | tail -1 | cut -c1-140)"; }
render() {  # render <tag> <seconds> <mode-label> [ref2va]
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i GPU-c146511a-0326-7ddc-4346-998d61a64b34; sleep 1; done > "$L/mem-$1.txt" & echo $! > "$L/sampler.pid" )
  if [ "${4:-}" = ref2va ]; then
    ( cd "$H" && curl -sS --max-time 3600 -D "out/$1.headers" -o "out/$1.mp4" -X POST http://127.0.0.1:8091/v1/videos/sync -F "prompt=$(cat ref2va/prompt-ref2va-astronaut-v2.txt)" \
      -F width=1024 -F height=576 -F fps=24 -F num_inference_steps=30 -F seed=1101 -F "image_reference=<ref2va/image-ref-astronaut.json" \
      -F "audio_reference=<ref2va/audio-ref-waitress-10s.json" -F "extra_params={\"task\":\"ref2va\",\"duration\":$2,\"audio_flow_shift\":3.0,\"aspect_ratio\":\"16:9\"}" > /dev/null 2>&1; \
      grep -oE "^HTTP/1.1 [0-9]+" "out/$1.headers" | tail -1 > "$L/$1.status" ) &
  else
    ( cd "$H" && ./gen.sh "$1" 30 1024 576 "$2" "$(cat ref2va/prompt-t2va-astronaut-control.txt)" > "$L/$1.log" 2>&1; grep -oE "^HTTP/1.1 [0-9]+" "out/$1.headers" | tail -1 > "$L/$1.status" ) &
  fi
  RP=$!; sleep 25
  python3 /home/jasonc/research/glm53-flash/reason_bench.py "ds41-final-during-$1" 1 --window 45 --warmup 10 2>&1 | sed "s/^/  DS41 C1 during $3: /"
  wait $RP; kill "$(cat "$L/sampler.pid")"
  echo "  $3 $1 ($2 s): $(cat "$L/$1.status" 2>/dev/null) | GB300 peak $(sort -n "$L/mem-$1.txt" | tail -1 | awk '{printf "%.1f", $1/1024}') of 249.8 GiB"
}
if [ "${RETRY:-0}" = 1 ]; then
  echo "=== $(date +%H:%M) retry: DS41 already relaunched and warm ($(gb)); start H3 lean"
  (cd "$H" && H3_MODE=fl2va-min ./serve-h3-gracie.sh > /dev/null); h3up
else
  wait_idle both "DS41 relaunch at 0.89 (H3 restarts with it)"
  echo "=== $(date +%H:%M) relaunch DS41 at GPU_UTIL 0.89"
  (cd "$M" && GPU_UTIL=0.89 ROWMAP=rowmap-mix-v1-h265.json DOCKER_MOUNTS="$(engram-nvfp4/lane-mounts.sh)" ./swap-to-m3v2.sh > "$L/swap.log" 2>&1)
  docker logs dsv41-flash-megamoe 2>&1 | grep -E "registered host memory|Available KV|GPU KV cache size" | sed 's/^.*\] //' | cut -c1-120
  echo "=== $(date +%H:%M) restart H3 lean (release cached memory); warm DS41 to steady state"
  (cd "$H" && H3_MODE=fl2va-min ./serve-h3-gracie.sh > /dev/null); h3up
  THINK_KEY=thinking /home/jasonc/venvs/bench/bin/python "$Q/needle_test.py" ds41-final-warm-1m 1000000 0.5 2>&1 | grep -E "needle [0-9]" | sed 's/^/  /'
  /home/jasonc/venvs/bench/bin/python /home/jasonc/ds41f-exp/bench_prefill.py --engine vllm --host 127.0.0.1 --port 30006 --isl 131072 --concurrency 1 \
    --seed $RANDOM$RANDOM --wait-ready 60 --output "$L/prefill-warm.jsonl" --label warm > /dev/null 2>&1
  python3 /home/jasonc/research/glm53-flash/reason_bench.py ds41-final-warm 16 --window 30 --warmup 10 > /dev/null 2>&1
  echo "  after warm-up: $(gb)"
fi
echo "=== $(date +%H:%M) A: H3 fl2va-min"
render final-A-15s 15 lean
wait_idle h3 "H3 restart into combined mode"
echo "=== $(date +%H:%M) B: H3 combined + VAE staging (voice cloning)"
(cd "$H" && H3_MODE=combined OFFLOAD=legacy ./serve-h3-gracie.sh > /dev/null); h3up
docker logs minimax-h3 2>&1 | grep -E "offload|staged|GPU memory after" | sed 's/^.*\] //' | cut -c1-120 | tail -4 | sed 's/^/  /'
echo "  idle: $(gb); host available $(free -g | awk '/Mem:/ {print $7}') GiB"
render final-B-15s 15 combined
render final-B-ref2va-6s 6 combined ref2va
B_OK=$(cat "$L/final-B-15s.status" "$L/final-B-ref2va-6s.status" 2>/dev/null | grep -c "200")
if [ "$B_OK" = 2 ]; then echo "=== $(date +%H:%M) B passed: H3 stays combined with VAE staging (voice cloning available)"; else
  echo "=== $(date +%H:%M) B failed: reverting H3 to fl2va-min"; wait_idle h3 "H3 revert"; (cd "$H" && H3_MODE=fl2va-min ./serve-h3-gracie.sh > /dev/null); h3up; fi
echo "  DS41 errors: $(docker logs dsv41-flash-megamoe 2>&1 | grep -ciE 'out of memory|Traceback')"
echo "=== $(date +%H:%M) done"
