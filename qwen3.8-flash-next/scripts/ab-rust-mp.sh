#!/usr/bin/env bash
# A/B the Rust frontend + mp executor against the default vLLM config (TRT-LLM MoE, MTP3, autotune on).
# Arms alternate base, rustmp, base, rustmp so drift and warm-cache effects hit both. Each arm: sanity prompt,
# GSM8K-50 on its first pass, then the catid decode recipe (8K in / 1K out, T=0) at C1-C32 (the user's working range).
set -uo pipefail
D=/home/jasonc/research/qwen38
export PORT=30006 MODEL_NAME=qwen38-flash-next
wait_gpu_free() {
  local used
  for _ in $(seq 120); do
    used=$(nvidia-smi --id=GPU-c146511a-0326-7ddc-4346-998d61a64b34 --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -lt 8000 ] && return 0
    sleep 5
  done
  echo "warning: GPU still holds ${used} MiB after 10 min"
}
for pass in 1 2; do
  for arm in base rustmp; do
    label=ab-$arm-$pass
    echo "=== $label  $(date +%H:%M:%S)"
    docker rm -f qwen-up qwen-sg >/dev/null 2>&1
    wait_gpu_free
    if [ $arm = rustmp ]; then
      RUST_MP=1 MTP=3 "$D/launch-qwen-upstream.sh" >/dev/null
    else
      RUST_MP=0 MTP=3 "$D/launch-qwen-upstream.sh" >/dev/null
    fi
    status=timeout
    for _ in $(seq 240); do
      sleep 5
      if curl -sf -m 2 localhost:30006/v1/models >/dev/null; then status=ready; break; fi
      if ! docker ps -q -f name=qwen-up | grep -q .; then status=exited; break; fi
    done
    docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
    if [ $status != ready ]; then
      echo "$label: server $status"; grep -E "Error" "$D/logs/server-$label.log" | tail -3 | cut -c1-300; continue
    fi
    curl -s localhost:30006/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"qwen38-flash-next","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":50,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
      | python3 -c "import json,sys;print('sanity:',json.load(sys.stdin)['choices'][0]['message']['content'])"
    [ $pass = 1 ] && (cd /home/jasonc/research/megamoe && THINK_KEY=enable_thinking /home/jasonc/venvs/bench/bin/python gsm8k_eval.py "qwen-$label" 50)
    /home/jasonc/ds41f-exp/bench_decode.sh "qwen-$label" "1 2 4 8 16 32" 2>&1 | grep -E "^decode"
    docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
  done
done
echo "=== done $(date +%H:%M:%S)"
