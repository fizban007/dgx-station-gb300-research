#!/usr/bin/env bash
# NVFP4 MoE kernel smoke test on vLLM with MTP off (--moe-backend also hits the FP8 MTP layer), one server per kernel.
# Per kernel: sanity prompt, GSM8K-50, decode (minimal prompt, 1,024 forced output tokens, T=0, C warm-ups + 2xC requests)
# at C1/C16/C64/C128, and cold prefill C1 at 8K and 64K. Decode covers 1-128 MoE tokens per step; prefill covers 8K chunks.
# Usage: smoke-kernels.sh [moe backend ...]   (default: every NVFP4 MoE kernel vLLM offers on SM103)
set -uo pipefail
D=/home/jasonc/research/qwen38
KERNELS=("$@"); [ ${#KERNELS[@]} -gt 0 ] || KERNELS=(flashinfer_trtllm flashinfer_cutedsl flashinfer_cutlass cutlass)
source /home/jasonc/venvs/bench/bin/activate
export PORT=30006 MODEL_NAME=qwen38-flash-next
wait_gpu_free() {  # a removed container can hold GPU memory for minutes; launch only once it is released
  local used
  for _ in $(seq 120); do
    used=$(nvidia-smi --id=GPU-c146511a-0326-7ddc-4346-998d61a64b34 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 8000 ] && return 0
    sleep 5
  done
  echo "warning: GPU still holds ${used} MiB after 10 min"
}
for k in "${KERNELS[@]}"; do
  label=smoke-$k; out=$D/runs/$label; mkdir -p "$out"
  echo "=== $k  $(date +%H:%M:%S)"
  docker rm -f qwen-up qwen-sg >/dev/null 2>&1
  wait_gpu_free
  RUST_MP=0 MTP=0 EXTRA="--moe-backend $k" "$D/launch-qwen-upstream.sh" >/dev/null
  status=timeout
  for _ in $(seq 240); do
    sleep 5
    if curl -sf -m 2 localhost:30006/v1/models >/dev/null; then status=ready; break; fi
    if ! docker ps -q -f name=qwen-up | grep -q .; then status=exited; break; fi
  done
  docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
  grep -E "NvFp4 MoE backend" "$D/logs/server-$label.log" | sed 's/.*\] //' | cut -c1-60
  if [ $status != ready ]; then
    echo "$k: server $status"; grep -E "Error" "$D/logs/server-$label.log" | tail -3 | cut -c1-300; continue
  fi
  curl -s localhost:30006/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":50,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys;print('sanity:',json.load(sys.stdin)['choices'][0]['message']['content'])"
  (cd /home/jasonc/research/megamoe && THINK_KEY=enable_thinking python gsm8k_eval.py "qwen-$label" 50)
  for c in 1 16 64 128; do
    printf 'n\n' | python /home/jasonc/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port 30006 \
      --model qwen38-flash-next --concurrency "$c" --contexts 0 --request-count $((c * 2)) --warmup-request-count "$c" \
      --max-tokens 1024 --temperature 0 --skip-prefill --display-mode plain --no-hw-monitor --no-resume \
      --output "$out/c$c.json" > "$out/c$c.log" 2>&1
    python3 - "$out/c$c.json" "$c" <<'EOF'
import json, sys
try:
    r = json.load(open(sys.argv[1]))["results"][0]
    print(f"decode C{sys.argv[2]:>3}  {r['aggregate_tps']:9.1f} tok/s  ITL p50 {r['inter_token_latency_p50'] * 1000:6.2f} ms"
          f"  errors {r['num_errors']}")
except Exception as e:
    print(f"decode C{sys.argv[2]:>3}  failed: {e}")
EOF
  done
  python /home/jasonc/ds41f-exp/bench_prefill.py --engine vllm --host 127.0.0.1 --port 30006 --isl 8192,65536 \
    --concurrency 1 --seed $RANDOM$RANDOM --wait-ready 60 --output "$D/logs/prefill-$label.jsonl" --label "$label" 2>&1 \
    | grep -E "^\s+[0-9]+ +1 " | awk '{printf "prefill %6d  %9s tok/s\n", $1, $5}'
  docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
done
echo "=== done $(date +%H:%M:%S)"
