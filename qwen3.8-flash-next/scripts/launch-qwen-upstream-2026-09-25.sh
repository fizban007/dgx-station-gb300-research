#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 on upstream vLLM nightly, one GB300, default kernels (TRT-LLM NVFP4 MoE).
# CKPT selects the checkpoint dir under ~/models (default: NVIDIA's NVFP4, native model_type qwen4_exp).
# /home/jasonc/models/qwen38-upstream-overlay symlinks every file and carries a config.json with those names.
# MTP=3 (default) enables the native MTP head with 3 speculative tokens; MTP=0 is autoregressive.
# RUST_MP=1 (default) runs the Rust API frontend and the mp executor. A/B on 2026-09-25 (2 runs per arm, catid recipe,
# MTP3): decode C1 +1%, C2-C16 within 2%, C32 -3%; TTFT 11% lower at C1 and 2x lower at C16. RUST_MP=0 restores vLLM's
# defaults (Python frontend, uni executor at TP1).
# DOCKER_ENV="A=1 B=2" passes extra environment variables into the container (e.g. VLLM_USE_RUST_FRONTEND=1).
set -euo pipefail
NAME=${NAME:-qwen-up}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423}
D=/home/jasonc/research/qwen38
JIT=$D/jit-cache
MTP=${MTP:-3}
RUST_MP=${RUST_MP:-1}
SEQS=${SEQS:-128}
MNBT=${MNBT:-8192}
SPEC_ARGS=()
if [ "$MTP" != 0 ]; then
  SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
fi
ENV_ARGS=()
for kv in ${DOCKER_ENV:-}; do ENV_ARGS+=(-e "$kv"); done
EXEC_ARGS=()
if [ "$RUST_MP" = 1 ]; then
  ENV_ARGS+=(-e VLLM_USE_RUST_FRONTEND=1)
  EXEC_ARGS=(--distributed-executor-backend mp)
fi
mkdir -p "$D/vllm-cache" "$JIT"/{tilelang,triton,nv,dj} "$D/logs"
docker run -d --name "$NAME" --gpus '"device=GPU-c146511a-0326-7ddc-4346-998d61a64b34"' \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v /home/jasonc/models:/models:ro -v "$D/vllm-cache":/root/.cache/vllm \
  -v "$JIT/tilelang":/root/.tilelang -v "$JIT/triton":/root/.triton -v "$JIT/nv":/root/.nv -v "$JIT/dj":/root/.dj \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  -e VLLM_LOGGING_LEVEL=INFO "${ENV_ARGS[@]}" \
  "$IMAGE" --membind=0 vllm serve \
  --model /models/${CKPT:-Qwen3.8-Flash-Next-NVFP4-nvidia} --served-model-name qwen38-flash-next --tensor-parallel-size 1 \
  --max-model-len 262144 --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" \
  --gpu-memory-utilization ${GPU_UTIL:-0.90} --limit-mm-per-prompt '{"image":0,"video":0}' \
  --reasoning-parser qwen3 \
  "${SPEC_ARGS[@]}" "${EXEC_ARGS[@]}" ${EXTRA:-} --host 0.0.0.0 --port 30006
echo "launched $NAME (MTP=$MTP RUST_MP=$RUST_MP)"
