#!/usr/bin/env bash
# M3: DeepGEMM MegaMoE for the 295 hot experts (+ fused shared expert) on the GB300; cold experts
# on the RTX PRO 6000 sidecar for T <= 64 (MEGA_PEER=1) and TRT-LLM from Grace otherwise.
# Otherwise v20's flags on today's nightly, minus the UVA expert offload (cold experts never reach HBM).
set -euo pipefail
NAME=${NAME:-m3}
IMAGE=vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423
MODEL=/home/jasonc/models/DeepSeek-V4.1-Flash
CACHE=/home/jasonc/research/megamoe/vllm-cache
HOOK=/home/jasonc/research/megamoe/hook
SEQS=${SEQS:-24}
KSCHED=${KSCHED:-"[[1,4,5],[5,8,2],[9,$SEQS,1]]"}
EXTRA=${EXTRA:-}
# DOCKER_ENV="A=1 B=2" passes extra environment variables into the container (e.g. CUDA_LOG_FILE=stderr).
ENV_ARGS=()
for kv in ${DOCKER_ENV:-}; do ENV_ARGS+=(-e "$kv"); done
# JIT caches that live outside ~/.cache/vllm; persisting them skips TileLang/Triton/CuTe recompiles at boot.
JIT=/home/jasonc/research/megamoe/jit-cache
PROF_ARGS=()
if [ "${PROF:-0}" = 1 ]; then
  PROF_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\",\"active_iterations\":${PROF_ITERS:-8}}")
fi
mkdir -p "$CACHE" "$JIT"/{tilelang,triton,nv,dj} /home/jasonc/research/megamoe/prof
docker run -d --name "$NAME" --gpus '"device=GPU-c146511a-0326-7ddc-4346-998d61a64b34"' \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v "$MODEL":/model:ro -v "$CACHE":/root/.cache/vllm \
  -v "$JIT/tilelang":/root/.tilelang -v "$JIT/triton":/root/.triton -v "$JIT/nv":/root/.nv -v "$JIT/dj":/root/.dj \
  -v /home/jasonc/research/megamoe/prof:/prof \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  -e VLLM_LOGGING_LEVEL=INFO "${ENV_ARGS[@]}" -e MEGA_HOOK=/w/mega_peer_hook.py -e MEGA_ROWMAP=/w/${ROWMAP:-rowmap-static-v1.json} \
  -e MEGA_PEER=${MEGA_PEER:-1} -e MEGA_PEER_MIN_TOKENS=${MEGA_PEER_MIN_TOKENS:-1} -e MEGA_PEER_CHECK=${MEGA_PEER_CHECK:-0} -e MEGA_PEER_CHECK_MIN_T=${MEGA_PEER_CHECK_MIN_T:-65} -e MEGA_COUNT=${MEGA_COUNT:-} -e MEGA_COLD_TRT=${MEGA_COLD_TRT:-1} -e VLLM_EXP_PEER2_SMALL=${VLLM_EXP_PEER2_SMALL:-64} \
  "$IMAGE" --membind=0 vllm serve \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization ${GPU_UTIL:-0.97} --kv-cache-dtype fp8_ds_mla \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED${SPEC_EXTRA:-}}" \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 \
  "${PROF_ARGS[@]}" $EXTRA --host 0.0.0.0 --port 30006
echo "launched $NAME"
