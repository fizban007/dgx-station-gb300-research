#!/usr/bin/env bash
# Run the catid-style suite (bench-qwen.sh) on vLLM with different NVFP4 MoE backends, one server at a time.
# Each argument is "label|moe backend|MTP tokens|extra vllm args|container env". --moe-backend also applies to the FP8 MTP layer,
# so kernels without FP8 block-128 support run with MTP=0 against a TRT-LLM MTP=0 baseline.
set -uo pipefail
D=/home/jasonc/research/qwen38
VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(
  "vllm-recipe-mtp3|flashinfer_trtllm|3|--no-enable-flashinfer-autotune --distributed-executor-backend mp -cc.cudagraph_mode full_decode_only|VLLM_USE_RUST_FRONTEND=1"
  "vllm-trtllm-mtp0|flashinfer_trtllm|0"
  "vllm-cutedsl-mtp0|flashinfer_cutedsl|0"
  "vllm-ficutlass-mtp0|flashinfer_cutlass|0"
)
wait_gpu_free() {  # a removed container can hold GPU memory for minutes; launch only once it is released
  local used
  for _ in $(seq 120); do
    used=$(nvidia-smi --id=GPU-c146511a-0326-7ddc-4346-998d61a64b34 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 8000 ] && return 0
    sleep 5
  done
  echo "warning: GPU still holds ${used} MiB after 10 min"
}
for v in "${VARIANTS[@]}"; do
  IFS="|" read -r label moe mtp extra denv <<< "$v"
  echo "=== $label (--moe-backend $moe, MTP ${mtp:-3}) $extra $denv  $(date +%H:%M:%S)"
  docker rm -f qwen-up qwen-sg >/dev/null 2>&1
  wait_gpu_free
  RUST_MP=0 DOCKER_ENV="$denv" MTP=${mtp:-3} EXTRA="--moe-backend $moe $extra" "$D/launch-qwen-upstream.sh" >/dev/null
  status=timeout
  for _ in $(seq 240); do
    sleep 5
    if curl -sf -m 2 localhost:30006/v1/models >/dev/null; then status=ready; break; fi
    if ! docker ps -q -f name=qwen-up | grep -q .; then status=exited; break; fi
  done
  docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
  grep -E "NvFp4 MoE backend|Fp8 MoE backend|GPU KV cache size|Executor|cudagraph_mode|autotune|vllm-rs|rust" "$D/logs/server-$label.log" | sed 's/.*\] //' | cut -c1-160
  if [ $status != ready ]; then
    echo "$label: server $status"; grep -E "Error|error:" "$D/logs/server-$label.log" | tail -4 | cut -c1-300; continue
  fi
  curl -s localhost:30006/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":50,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys;print('sanity:',json.load(sys.stdin)['choices'][0]['message']['content'])"
  "$D/bench-qwen.sh" "$label" 2>&1
  docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
done
echo "=== done $(date +%H:%M:%S)"
