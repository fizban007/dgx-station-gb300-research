#!/usr/bin/env bash
# M3: DeepGEMM MegaMoE for the 295 hot experts (+ fused shared expert) on the GB300; cold experts
# on the RTX PRO 6000 sidecar for T <= 64 (MEGA_PEER=1) and TRT-LLM from Grace otherwise.
# Otherwise v20's flags on today's nightly, minus the UVA expert offload (cold experts never reach HBM).
set -euo pipefail
NAME=${NAME:-m3}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}
MODEL=/home/jasonc/models/DeepSeek-V4.1-Flash
CACHE=/home/jasonc/research/megamoe/vllm-cache
HOOK=/home/jasonc/research/megamoe/hook
SEQS=${SEQS:-24}
# DSpark draft length per batch size: 5/3/3 with probabilistic drafting won the 2026-09-28 evening sweep
# (logs/sweep-pd-k.out; 5/3/2 before, logs/sweep-k-1.out). SPEC_EXTRA= (empty) drops probabilistic drafting.
KSCHED=${KSCHED:-"[[1,4,5],[5,8,3],[9,$SEQS,3]]"}
# Graph sizes cover C x (k+1) for that schedule with <= ~12% padding (padded tokens still route through experts).
CG_SIZES=${CG_SIZES:-"1 2 4 6 8 12 16 18 20 24 28 32 36 40 48 56 64 72 80 96 128"}
EXTRA=${EXTRA:-}
# DOCKER_ENV="A=1 B=2" passes extra environment variables into the container (e.g. CUDA_LOG_FILE=stderr).
ENV_ARGS=()
for kv in ${DOCKER_ENV:-}; do ENV_ARGS+=(-e "$kv"); done
# DOCKER_MOUNTS="host:container[:ro] ..." adds bind mounts, e.g. a Python overlay over the image's vllm package.
MOUNT_ARGS=()
for mnt in ${DOCKER_MOUNTS:-}; do MOUNT_ARGS+=(-v "$mnt"); done
# JIT caches that live outside ~/.cache/vllm; persisting them skips TileLang/Triton/CuTe recompiles at boot.
JIT=/home/jasonc/research/megamoe/jit-cache
# Server-side reasoning default: a V4.1 effort name or 1-100. Names map per image: from nightly af7f9488 (vllm#58316)
# low=50 high=75 max=100; 7f1a5398 has low=25 high=50 xhigh=75 max=100. Requests still override it with
# reasoning_effort or chat_template_kwargs. REASONING_EFFORT= keeps the model default.
REASONING_EFFORT=${REASONING_EFFORT-high}
EFFORT_ARGS=()
if [[ "$REASONING_EFFORT" =~ ^[0-9]+$ ]]; then
  EFFORT_ARGS=(--default-chat-template-kwargs "{\"reasoning_effort\":$REASONING_EFFORT}")
elif [ -n "$REASONING_EFFORT" ]; then
  EFFORT_ARGS=(--default-chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}")
fi
# Engram config JSON; e.g. '{"cpu_offload": true, "use_thp": true}' for huge-page host tables (af7f9488+).
ENGRAM_DEFAULT='{"cpu_offload": true}'
ENGRAM_CONFIG=${ENGRAM_CONFIG:-$ENGRAM_DEFAULT}
PROF_ARGS=()
NSYS_MOUNT=()
NSYS_CMD=()
if [ "${PROF:-0}" = 1 ]; then
  PROF_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\",\"active_iterations\":${PROF_ITERS:-8}}")
elif [ "${PROF:-0}" = nsys ]; then
  # Host Nsight Systems inside the container. Kernels inside CUDA graph replays are traced per node; collection runs
  # only between /start_profile and /stop_profile (or PROF_ITERS engine iterations after PROF_DELAY), into /prof.
  PROF_ARGS=(--profiler-config "{\"profiler\":\"cuda\",\"delay_iterations\":${PROF_DELAY:-10},\"max_iterations\":${PROF_ITERS:-40}}")
  NSYS_MOUNT=(-v /opt/nvidia/nsight-systems/2025.6.3:/opt/nsys:ro)
  # PROF_END=repeat-shutdown:N records N ranges (one report each) and then stops the server so the reports finalize.
  NSYS_CMD=(/opt/nsys/bin/nsys profile -o "/prof/${PROF_NAME:-nsys-decode}-%n" --force-overwrite=true --trace=cuda,nvtx
            --cuda-graph-trace=node --capture-range=cudaProfilerApi --capture-range-end="${PROF_END:-repeat-shutdown:2}"
            --sample=none --cpuctxsw=none)
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
  -e MEGA_FUSED_SEND=${MEGA_FUSED_SEND:-2} -e MEGA_MHC_OVERLAP=${MEGA_MHC_OVERLAP:-1} -e MEGA_LL_GEMM=${MEGA_LL_GEMM:-1} \
  "${NSYS_MOUNT[@]}" "${MOUNT_ARGS[@]}" "$IMAGE" --membind=0 "${NSYS_CMD[@]}" vllm serve \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --moe-backend deep_gemm_mega_moe --enable-expert-parallel \
  --engram-config "$ENGRAM_CONFIG" \
  --max-model-len 1048576 --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization ${GPU_UTIL:-0.97} --kv-cache-dtype fp8_ds_mla \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 --enable-prefix-caching \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED${SPEC_EXTRA-,\"draft_sample_method\":\"probabilistic\"}}" \
  --cudagraph-capture-sizes $CG_SIZES \
  "${EFFORT_ARGS[@]}" "${PROF_ARGS[@]}" $EXTRA --host 0.0.0.0 --port 30006
echo "launched $NAME"
