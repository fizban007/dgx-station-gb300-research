#!/usr/bin/env bash
# MiMo-V2.6-Flash-RL (309B/15B, MXFP4-stored experts, FP8 block-128 dense) on one GB300, TP1, upstream vLLM.
# Defaults (2026-09-25 A/Bs): nightly 29468dde, SPEC=dflash7, KVGROUP=pr58207. Suite results in logs/bench-*.log.
# Based on the vLLM recipe (recipes.vllm.ai XiaomiMiMo/MiMo-V2.6-Flash-RL; gb300 "verified", no GB300 overrides).
# SPEC=none | mtp<N> (native 3-layer MTP head) | dflash<N> (5-layer DFlash drafter in <ckpt>/dflash; recipe uses 7).
# Prefix caching is always passed explicitly (user rule); confirm "enable_prefix_caching=True" in the log.
# MOE_BACKEND=auto (FlashInfer TRT-LLM MXFP4 x BF16 on SM10x) or e.g. flashinfer_trtllm / deep_gemm / triton.
# RUST_MP=1 runs the Rust API frontend + mp executor (loses /metrics engine stats); default 0 keeps the Python frontend.
# KVGROUP=pr58207 mounts overlay/kv_cache_utils.py: vllm#58207's hybrid KV group-size choice (fewest worst-case padding
#   bytes), backported to the 7f1a5398 image. Main pads a full-attention layer (9 -> 10) once the DFlash drafter's
#   5-layer bucket joins; the PR keeps group size 9 with only bounded SWA padding.
# FA4 (CuTe DSL) kernels persist in jit-cache/fa-cute; without it the first prefix-cache hit after every restart JIT-compiles
#   FlashAttentionForwardSm100 (~5-6 s stall). vLLM's warmup doesn't cover that shape.
# DOCKER_ENV="A=1 B=2" passes extra environment variables into the container; EXTRA appends raw serve args.
set -euo pipefail
NAME=${NAME:-mimo}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098}
D=/home/jasonc/research/mimo26
JIT=$D/jit-cache
CKPT=${CKPT:-MiMo-V2.6-Flash-RL}
SPEC=${SPEC:-dflash7}
RUST_MP=${RUST_MP:-0}
SEQS=${SEQS:-64}
MNBT=${MNBT:-8192}
MAXLEN=${MAXLEN:-auto}
MOE_BACKEND=${MOE_BACKEND:-auto}
SPEC_ARGS=()
case "$SPEC" in
  none) ;;
  mtp*) SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC#mtp}}") ;;
  dflash*) SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/models/$CKPT/dflash\",\"num_speculative_tokens\":${SPEC#dflash}}"
                      --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}') ;;
  *) echo "bad SPEC=$SPEC" >&2; exit 2 ;;
esac
ENV_ARGS=()
for kv in ${DOCKER_ENV:-}; do ENV_ARGS+=(-e "$kv"); done
EXEC_ARGS=()
if [ "$RUST_MP" = 1 ]; then
  ENV_ARGS+=(-e VLLM_USE_RUST_FRONTEND=1)
  EXEC_ARGS=(--distributed-executor-backend mp)
fi
OVERLAY_ARGS=()
if [ "${KVGROUP:-pr58207}" = pr58207 ]; then
  # One backported file per image; kv_cache_utils.py differs between nightlies.
  case "$IMAGE" in
    *7f1a5398*) OVL=kv_cache_utils.py ;;
    *29468dde*) OVL=kv_cache_utils.29468.py ;;
    *) echo "no pr58207 overlay for $IMAGE" >&2; exit 2 ;;
  esac
  OVERLAY_ARGS=(-v "$D/overlay/$OVL:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro")
fi
MOE_ARGS=()
[ "$MOE_BACKEND" != auto ] && MOE_ARGS=(--moe-backend "$MOE_BACKEND")
docker run -d --name "$NAME" --gpus '"device=GPU-c146511a-0326-7ddc-4346-998d61a64b34"' \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v /home/jasonc/models:/models:ro -v "$D/vllm-cache":/root/.cache/vllm -v "$JIT/flashinfer":/root/.cache/flashinfer -v "$JIT/fa-cute":/root/.cache/fa-cute \
  -v "$JIT/tilelang":/root/.tilelang -v "$JIT/triton":/root/.triton -v "$JIT/nv":/root/.nv -v "$JIT/dj":/root/.dj \
  "${OVERLAY_ARGS[@]}" -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  -e VLLM_LOGGING_LEVEL=INFO -e FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 -e FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/root/.cache/fa-cute \
  "${ENV_ARGS[@]}" \
  "$IMAGE" --membind=0 vllm serve \
  --model /models/$CKPT --served-model-name mimo-v26-flash --trust-remote-code --tensor-parallel-size 1 \
  --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" \
  --gpu-memory-utilization ${GPU_UTIL:-0.93} --enable-prefix-caching --generation-config vllm \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  "${MOE_ARGS[@]}" "${SPEC_ARGS[@]}" "${EXEC_ARGS[@]}" ${EXTRA:-} --host 0.0.0.0 --port 30006
echo "launched $NAME (IMAGE=${IMAGE##*:} SPEC=$SPEC MOE=$MOE_BACKEND RUST_MP=$RUST_MP KVGROUP=${KVGROUP:-pr58207})"
