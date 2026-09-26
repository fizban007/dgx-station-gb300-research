#!/usr/bin/env bash
# Run the catid-style suite (bench-qwen.sh) on a list of SGLang variants, one server at a time.
# Each variant is "label|moe backend|extra launch args".
# Every variant keeps the 330 Mamba-slot / 0.85 memory-fraction base so concurrency limits match.
set -uo pipefail
D=/home/jasonc/research/qwen38
BASE="--max-mamba-cache-size 330 --mem-fraction-static 0.85 --enable-metrics"
VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=(
  "sglang-rs-mtp3|auto|--enable-linear-replayssm-spec"
  "sglang-rs-fipre-mtp3|auto|--enable-linear-replayssm-spec --linear-attn-prefill-backend flashinfer"
  "sglang-fi-mtp3|auto|--linear-attn-backend flashinfer"
  "sglang-mega-mtp3|flashinfer_megamoe|--moe-a2a-backend flashinfer_megamoe --enable-dp-attention --dp-size 1"
)
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r label moe args <<< "$v"
  echo "=== $label (MOE=$moe) $args  $(date +%H:%M:%S)"
  docker rm -f qwen-sg >/dev/null 2>&1
  MEGA_PATCH=$([ "$moe" = flashinfer_megamoe ] && echo 1 || echo 0) MOE=$moe MTP=3 EXTRA="$BASE $args" "$D/launch-qwen-sglang.sh" >/dev/null
  status=timeout
  for _ in $(seq 180); do
    sleep 5
    if docker logs qwen-sg 2>&1 | grep -q "fired up and ready"; then status=ready; break; fi
    if ! docker ps -q -f name=qwen-sg | grep -q .; then status=exited; break; fi
  done
  docker logs qwen-sg > "$D/logs/server-$label.log" 2>&1
  grep -E "max_running_requests=|Mamba Cache is|Linear attention kernel backend|MoE runner" "$D/logs/server-$label.log" | cut -c1-220 | sort -u
  if [ $status != ready ]; then
    echo "$label: server $status"; grep -E "Error|error:|Traceback" "$D/logs/server-$label.log" | tail -5; continue
  fi
  curl -s localhost:30006/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":50,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import json,sys;print('sanity:',json.load(sys.stdin)['choices'][0]['message']['content'])"
  ENGINE=sglang "$D/bench-qwen.sh" "$label" 2>&1
  docker logs qwen-sg > "$D/logs/server-$label.log" 2>&1
done
echo "=== done $(date +%H:%M:%S)"
