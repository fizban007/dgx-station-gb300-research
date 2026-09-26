#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 on the Karmic fork (native Qwen3.8 support), one GB300, host venv.
# Adapted from the team's jovian launcher (stinger:~/spark_vllm/launchers/serve-qwen38-flash-next-jovian-tp4.sh):
# TP1 instead of TP4, safetensors loading, text only, GB300 by index with SM103 b12x enabled.
#   BACKENDS=b12x (default): attention/MoE/linear/GDN decode on b12x, as the jovian recipe.
#   BACKENDS=auto: let the fork pick every kernel.
#   MTP=3 (default) or 0.
set -euo pipefail
D=/home/jasonc/research/qwen38
LABEL=${LABEL:-fork-${BACKENDS:-b12x}-mtp${MTP:-3}}
mkdir -p "$D/logs" "$D/runs/$LABEL"
cd "$D/runs/$LABEL"   # never start from a directory that contains a b12x checkout
export PATH=/home/jasonc/venvs/vllm-karmic/bin:/usr/local/cuda-13.2/bin:$PATH CUDA_HOME=/usr/local/cuda-13.2
export CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUTE_DSL_ARCH=sm_103a VLLM_B12X_SM103=1
export VLLM_SERVER_DEV_MODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1
compilation_config='{"inductor_compile_config":{"triton.autotune_at_compile_time":false,"combo_kernels":false,"benchmark_combo_kernel":false,"enable_auto_functionalized_v2":false}}'
spec=()
[ "${MTP:-3}" != 0 ] && spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP:-3}}")
backend=()
case "${BACKENDS:-b12x}" in
  b12x) backend=(--attention-backend B12X --moe-backend b12x --linear-backend b12x --gdn-decode-kernel b12x
                 --no-enable-flashinfer-autotune) ;;
  # b12x's MXFP8 linear kernels are SM12x-only; let the fork pick dense linears on SM103.
  b12x-nolinear) backend=(--attention-backend B12X --moe-backend b12x --gdn-decode-kernel b12x) ;;
  # b12x GDN kernels are SM12x-only too; on SM103 only the fused MoE (and maybe attention) apply.
  b12x-moe) backend=(--moe-backend b12x) ;;
  auto) backend=() ;;
esac
setsid numactl --membind=0 /home/jasonc/venvs/vllm-karmic/bin/python -m vllm.entrypoints.cli.main serve \
  /home/jasonc/models/Qwen3.8-Flash-Next-NVFP4 \
  --served-model-name qwen38-flash-next --host 0.0.0.0 --port 30006 --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" --block-size "${BLOCK_SIZE:-16}" \
  --max-model-len 262144 --max-num-seqs "${SEQS:-128}" --max-num-batched-tokens "${MNBT:-8192}" \
  --load-format safetensors --kv-cache-dtype bfloat16 --mamba-cache-mode align \
  --recurrent-checkpoint-policy aligned --dtype bfloat16 --quantization modelopt_mixed \
  --limit-mm-per-prompt '{"image":0,"video":0}' --mm-processor-cache-gb 0 \
  --enable-prefix-caching --async-scheduling --enable-chunked-prefill --compilation-config "$compilation_config" \
  "${backend[@]}" "${spec[@]}" \
  --generation-config vllm --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --enable-prompt-tokens-details --enable-force-include-usage ${EXTRA:-} \
  > "$D/logs/$LABEL.log" 2>&1 < /dev/null &
echo $! > "$D/logs/qwen-fork.pid"
echo "launched $LABEL pid $(cat $D/logs/qwen-fork.pid); log $D/logs/$LABEL.log"
