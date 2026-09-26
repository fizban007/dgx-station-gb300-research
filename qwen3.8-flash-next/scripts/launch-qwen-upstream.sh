#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 on upstream vLLM nightly, one GB300, default kernels (TRT-LLM NVFP4 MoE).
# CKPT selects the checkpoint dir under ~/models (default: NVIDIA's NVFP4, native model_type qwen4_exp).
# /home/jasonc/models/qwen38-upstream-overlay symlinks every file and carries a config.json with those names.
# MTP=3 (default) enables the native MTP head with 3 speculative tokens; MTP=0 is autoregressive.
# RUST_MP=0 (default since 2026-09-25) is vLLM's Python frontend with the uni executor. It keeps the /metrics engine stats,
# --override-generation-config and structured outputs (tool_choice required/named), all of which vllm-rs lacks. With
# CG=8192 it beats the old Rust + mp default on TTFT (C16 175 vs 177-180 ms) and decode (C32 3,784 vs 3,109-3,122 tok/s).
# Python + mp and --api-server-count 4 bought nothing measurable (dgx-station-gb300-research qwen3.8-flash-next split
# test). RUST_MP=1 runs the Rust API frontend and the mp executor (2026-09-25 A/B, before CG=8192).
# CG=8192 (default) captures piecewise CUDA graphs for prefill steps up to 8192 tokens (vLLM stops at 1024, and
# larger steps run launch-bound at ~125 ms each). Costs ~4 GiB of KV (4.99M -> 4.85M tokens). CG=1024 restores vLLM. See cg/README.md.
# VISION=video (default) accepts up to 16 images and 1 video per prompt; VISION=image drops video; VISION=none is text-only.
# DOCKER_ENV="A=1 B=2" passes extra environment variables into the container (e.g. VLLM_USE_RUST_FRONTEND=1).
set -euo pipefail
NAME=${NAME:-qwen-up}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423}
D=/home/jasonc/research/qwen38
JIT=$D/jit-cache
MTP=${MTP:-3}
RUST_MP=${RUST_MP:-0}
SEQS=${SEQS:-128}
MNBT=${MNBT:-8192}
VISION=${VISION:-video}
case "$VISION" in
  none) MM_LIMIT='{"image":0,"video":0}' ;;
  image) MM_LIMIT='{"image":16,"video":0}' ;;
  video) MM_LIMIT='{"image":16,"video":1}' ;;
  *) echo "VISION must be none, image or video" >&2; exit 1 ;;
esac
CG=${CG:-8192}
COMP_ARGS=()
if [ "$CG" = 8192 ]; then
  COMP_ARGS=(--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256,272,288,304,320,336,352,368,384,400,416,432,448,464,480,496,512,528,544,560,576,592,608,624,640,656,672,688,704,720,736,752,768,784,800,816,832,848,864,880,896,912,928,944,960,976,992,1008,1024,1280,1600,1792,2048,2560,3200,3584,4096,4800,5120,6144,6400,7168,8000,8192],"max_cudagraph_capture_size":8192}')
fi
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
  --gpu-memory-utilization ${GPU_UTIL:-0.90} --enable-prefix-caching --limit-mm-per-prompt "$MM_LIMIT" \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  "${SPEC_ARGS[@]}" "${EXEC_ARGS[@]}" "${COMP_ARGS[@]}" ${EXTRA:-} --host 0.0.0.0 --port 30006
echo "launched $NAME (MTP=$MTP RUST_MP=$RUST_MP CG=$CG VISION=$VISION)"
