#!/usr/bin/env bash
# Qwen3.8-Flash-Next-NVFP4 (NVIDIA checkpoint) on the SGLang nightly, one GB300, same port and name as the vLLM runs.
#   MOE=auto (default) | flashinfer_megamoe | flashinfer_trtllm | flashinfer_cutlass | flashinfer_cutedsl | ...
#   MTP=3 (default) runs NEXTN with 3 steps, topk 1, 4 draft tokens; MTP=0 is autoregressive.
#   EXTRA passes anything else through (for example --chunked-prefill-size 16384).
#   MEGA_PATCH=1 mounts patches/moe_hook.py, which adds Qwen4Exp to SGLang's FlashInfer MegaMoE allowlist.
set -euo pipefail
NAME=${NAME:-qwen-sg}
IMAGE=${IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260924-ffac53d7}
D=/home/jasonc/research/qwen38
MTP=${MTP:-3}
SPEC_ARGS=()
if [ "$MTP" != 0 ]; then
  SPEC_ARGS=(--speculative-algorithm NEXTN --speculative-num-steps "$MTP" --speculative-eagle-topk 1
             --speculative-num-draft-tokens $((MTP + 1)))
fi
DOCKER_EXTRA=()
[ "${MEGA_PATCH:-0}" = 1 ] && DOCKER_EXTRA=(-v "$D/patches/moe_hook.py:/sgl-workspace/sglang/python/sglang/srt/arg_groups/moe_hook.py:ro")
mkdir -p "$D/sglang-cache" "$D/jit-cache/triton" "$D/logs"
docker run -d --name "$NAME" --gpus '"device=GPU-c146511a-0326-7ddc-4346-998d61a64b34"' \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v /home/jasonc/models:/models:ro -v "$D/sglang-cache":/root/.cache -v "$D/jit-cache/triton":/root/.triton \
  "${DOCKER_EXTRA[@]}" -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  "$IMAGE" --membind=0 python3 -m sglang.launch_server \
  --model-path /models/${CKPT:-Qwen3.8-Flash-Next-NVFP4-nvidia} --served-model-name qwen38-flash-next --tp 1 \
  --context-length 262144 --max-running-requests "${SEQS:-128}" \
  --moe-runner-backend "${MOE:-auto}" --reasoning-parser qwen3 \
  "${SPEC_ARGS[@]}" ${EXTRA:-} --host 0.0.0.0 --port 30006
echo "launched $NAME (MOE=${MOE:-auto} MTP=$MTP)"
