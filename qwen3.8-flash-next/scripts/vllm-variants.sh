#!/usr/bin/env bash
# Run the catid-style suite (bench-qwen.sh) on vLLM with different NVFP4 MoE backends, one server at a time.
# Each argument is "label|moe backend"; defaults cover every non-default NVFP4 MoE kernel vLLM offers on SM103.
set -uo pipefail
D=/home/jasonc/research/qwen38
VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(
  "vllm-cutedsl-mtp3|flashinfer_cutedsl"
  "vllm-ficutlass-mtp3|flashinfer_cutlass"
  "vllm-cutlass-mtp3|cutlass"
)
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r label moe <<< "$v"
  echo "=== $label (--moe-backend $moe)  $(date +%H:%M:%S)"
  docker rm -f qwen-up qwen-sg >/dev/null 2>&1
  RUST_MP=0 MTP=3 EXTRA="--moe-backend $moe" "$D/launch-qwen-upstream.sh" >/dev/null
  status=timeout
  for _ in $(seq 240); do
    sleep 5
    if docker logs qwen-up 2>&1 | grep -q "Application startup complete"; then status=ready; break; fi
    if ! docker ps -q -f name=qwen-up | grep -q .; then status=exited; break; fi
  done
  docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
  grep -E "NvFp4 MoE backend|Fp8 MoE backend|GPU KV cache size" "$D/logs/server-$label.log" | sed 's/.*\] //' | cut -c1-160
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
