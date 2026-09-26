#!/usr/bin/env bash
# Launch DeepSeek-V4.1-Flash on the GB300 with b12x expert residency for one experiment.
# Usage: serve.sh <label> [extra vllm serve args...]; knobs come from the environment.
set -euo pipefail
label=${1:?label}; shift
EXP=/home/jasonc/ds41f-exp; RUN=$EXP/runs/$label; mkdir -p "$RUN"
cd "$RUN"   # never start from a directory that contains a b12x checkout
: "${PORT:=30000}" "${MAX_NUM_SEQS:=64}" "${MAX_MODEL_LEN:=262144}" "${MAX_NUM_BATCHED_TOKENS:=1024}"
: "${KV_CACHE_GB:=10}" "${HBM_RESERVE_GB:=12}" "${ENGRAM_TABLE_MEMORY:=disk}" "${NUM_SPECULATIVE_TOKENS:=0}"
: "${MOE_BACKEND:=b12x}" "${PREFIX_CACHING:=--enable-prefix-caching}" "${SHARED_WORKSPACE:=}" "${RESIDENCY_EXTRA:=}" "${CAPTURE_MAX:=64}" "${BLOCK_SIZE:=256}"
export PATH=/home/jasonc/venvs/vllm-karmic/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
export CUDA_VISIBLE_DEVICES=1 CUTE_DSL_ARCH=sm_103a VLLM_B12X_SM103=1 VLLM_USE_V2_MODEL_RUNNER=1
export B12X_COMPILE_WORKERS=${B12X_COMPILE_WORKERS:-0}
[ -n "${B12X_AUTOTUNE:-}" ] && export B12X_AUTOTUNE
export VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
kv_bytes=$(awk -v g="$KV_CACHE_GB" 'BEGIN{printf "%d", g*1073741824}')
shared=""; [ -n "$SHARED_WORKSPACE" ] && shared=",\"shared_workspace\":$SHARED_WORKSPACE"
residency="{\"hbm_reserve_gb\":$HBM_RESERVE_GB$shared${RESIDENCY_EXTRA}}"
engram="{\"cpu_offload\":false,\"table_memory\":\"$ENGRAM_TABLE_MEMORY\"}"
spec=()
if ((NUM_SPECULATIVE_TOKENS > 0)); then
  spec=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"attention_backend\":\"B12X\",\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"standard\",\"enable_adaptive_verification\":${DSPARK_ADAPTIVE:-true}${KSCHED:+,\"num_speculative_tokens_per_batch_size\":$KSCHED}}")
fi
env | grep -E "^(MOE_|PREFIX_|VLLM_|CUTE_|CUDA_VISIBLE|MAX_|KV_|HBM_|ENGRAM_|NUM_SPEC|SHARED_|RESIDENCY_)" | sort > env.txt
exec numactl --membind=0 /home/jasonc/venvs/vllm-karmic/bin/python -m vllm.entrypoints.cli.main serve /home/jasonc/models/DeepSeek-V4.1-Flash \
  --served-model-name DeepSeek-V4.1-Flash --host 127.0.0.1 --port "$PORT" --dtype bfloat16 \
  --tensor-parallel-size 1 --language-model-only --load-format safetensors --block-size "$BLOCK_SIZE" \
  --kv-cache-memory-bytes "$kv_bytes" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-cudagraph-capture-size "$CAPTURE_MAX" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --generation-config vllm --engram-config "$engram" --expert-residency-config "$residency" \
  --attention-backend B12X --linear-backend b12x --moe-backend "$MOE_BACKEND" "${spec[@]}" \
  --tokenizer-mode deepseek_v41 $PREFIX_CACHING "$@"
