#!/usr/bin/env bash
# Does presence_penalty (Qwen's documented anti-repetition knob) stop the 128K thinking-channel loops, and at what
# quality cost? For each penalty: launch the default config (TRT-LLM MoE, MTP3, Rust + mp) with the penalty as a
# server-side default, GSM8K-200, then REPS runs of llm_decode_bench (from stinger) at C1 and C8 with 128K context.
# Usage: pp-loop-test.sh [penalty ...]   (default: 0 0.5); REPS=8
set -uo pipefail
D=/home/jasonc/research/qwen38
REPS=${REPS:-8}
PENALTIES=("$@"); [ ${#PENALTIES[@]} -gt 0 ] || PENALTIES=(0 0.5)
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
for pp in "${PENALTIES[@]}"; do
  label=pp$pp
  echo "=== presence_penalty $pp  $(date +%H:%M:%S)"
  docker rm -f qwen-up qwen-sg >/dev/null 2>&1
  wait_gpu_free
  if [ "$pp" = 0 ]; then
    RUST_MP=1 MTP=3 "$D/launch-qwen-upstream.sh" >/dev/null
  else
    RUST_MP=1 MTP=3 EXTRA="--override-generation-config {\"presence_penalty\":$pp}" "$D/launch-qwen-upstream.sh" >/dev/null
  fi
  status=timeout
  for _ in $(seq 240); do
    sleep 5
    if curl -sf -m 2 localhost:30006/v1/models >/dev/null; then status=ready; break; fi
    if ! docker ps -q -f name=qwen-up | grep -q .; then status=exited; break; fi
  done
  docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
  grep -i -E "default sampling param|generation config" "$D/logs/server-$label.log" | sed 's/.*\] //' | cut -c1-200 | head -3
  if [ $status != ready ]; then echo "$label: server $status"; grep -E "Error" "$D/logs/server-$label.log" | tail -3; continue; fi
  (cd /home/jasonc/research/megamoe && THINK_KEY=enable_thinking /home/jasonc/venvs/bench/bin/python gsm8k_eval.py "qwen-$label" 200)
  for i in $(seq "$REPS"); do
    out=/home/jasonc/spark_vllm/qwen38-loop-$label-$i
    ssh -o BatchMode=yes stinger "cd /home/jasonc/spark_vllm/llm-inference-bench && python3 llm_decode_bench.py \
      --host http://192.168.1.99 --port 30006 --model qwen38-flash-next --concurrency 1,8 --contexts 128k --duration 30 \
      --max-tokens 8192 --skip-prefill --display-mode plain --no-hw-monitor --no-resume --output $out.json \
      < /dev/null > $out.log 2>&1; python3 /tmp/looptally.py $out.json '$label run$i'" < /dev/null
  done
  docker logs qwen-up > "$D/logs/server-$label.log" 2>&1
done
echo "=== done $(date +%H:%M:%S)"
