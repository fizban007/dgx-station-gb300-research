#!/usr/bin/env bash
# MiMo-V2.6-Pro-RL (1.02T / 42B active, 527 GiB) on one GB300 + the RTX PRO 6000 sidecar + Grace, vLLM nightly 29468dde.
#
# Base: Al-ENGR's verified "hotsplit" recipe (J-M-Recipes/recipes mimo-v2.6-pro-vllm-uva-hotsplit, v23): vLLM UVA offload
# of the routed experts, then per-expert residency by measured decode usage, Marlin MoE.
# Ours on top:
#   - today's nightly (29468dde) instead of his d05da62e, with his three open upstream fixes backported onto its files
#     (vllm#58142 fused fp8 qkv pairing, #58184 truncation guard, #58185 exact-size pinned UVA; hook/overlay, mimo_v2*.29468.py)
#     and our Flash-lane fixes (vllm#58207 KV grouping overlay, persistent FA4 JIT cache);
#   - a third tier: the rowmap's "peer" experts live only on the RTX PRO 6000 (hook/peer_server_mimo.py, b12x w4a8_mx),
#     fed through /dev/shm/vllm_peer_mimo (hook/peer_tier_mimo.py) by hook/hotsplit.py;
#   - 8K prefill chunks (his 2K re-read the Grace experts every 2,048 tokens), 16 sequences, prefix caching on,
#     --generation-config vllm (his "auto" applies the checkpoint's max_new_tokens=2048 default).
# Start the sidecar first (it must print "serving"); this script refuses otherwise.
# FUSED_SEND=1 (3-kernel peer send), STAGE_SLOTS=128 (Grace experts of decode-size batches staged into HBM before Marlin;
#   0 = Marlin reads Grace through UVA), STAGE_OVERLAP=0 (a side-stream copy measured no overlap), STAGE_CHECK=N.
# ROWMAP=<file in hook/>  PEER_CHECK=<N eager cross-checks vs Marlin; keeps the peer experts in Grace too>  PROF=1 enables
# the torch profiler (/start_profile, /stop_profile -> prof/).
set -euo pipefail
NAME=${NAME:-mimo-pro}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098}
D=/home/jasonc/research/mimo-pro
HOOK=$D/hook
JIT=/home/jasonc/research/mimo26/jit-cache   # shared with the Flash lane: FlashInfer/FA4/Triton builds are reusable
V=/usr/local/lib/python3.12/dist-packages/vllm
ROWMAP=${ROWMAP:-rowmap-3tier-v3.json}
CTX=${CTX:-1048576}
SEQS=${SEQS:-16}
MNBT=${MNBT:-8192}
if [ "${PEER:-1}" = 1 ] && ! grep -q "^serving" $D/logs/peer_server_mimo.log 2>/dev/null; then
  echo "sidecar not serving (logs/peer_server_mimo.log); start it first or set PEER=0"; exit 4
fi
ROWMAP_ENV=(-e HOTSPLIT_ROWMAP=/w/$ROWMAP)
[ "${PEER:-1}" = 1 ] || ROWMAP_ENV=(-e HOTSPLIT_HOT_GIB=${HOT_GIB:-152.8})
# MOE=trtllm (default): every bank runs FlashInfer TRT-LLM MXFP4 x MXFP8 from HBM (hook/trt_banks.py): 1.7-1.8x Marlin
#   at decode sizes, 2-6.4x at prefill sizes on these experts; untuned == tuned within 1%, so FlashInfer autotune is off
#   (tuning would read Grace-resident weights). MOE=marlin restores the Marlin banks.
MOE_ARGS=(--moe-backend flashinfer_trtllm --no-enable-flashinfer-autotune)
[ "${MOE:-trtllm}" = marlin ] && MOE_ARGS=(--moe-backend marlin)
PROF_ARGS=()
if [ "${PROF:-1}" = 1 ]; then
  PROF_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\",\"active_iterations\":${PROF_ITERS:-32}}")
fi
mkdir -p $D/{vllm-cache,live,prof,logs} "$JIT"/{flashinfer,fa-cute,triton,nv,dj,tilelang}
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null   # room for ~320 GiB of pinned experts
docker run -d --name "$NAME" --gpus '"device=GPU-c146511a-0326-7ddc-4346-998d61a64b34"' \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v /home/jasonc/models/MiMo-V2.6-Pro-RL:/model:ro -v $D/vllm-cache:/root/.cache/vllm -v $HOOK:/w:ro \
  -v $D/live:/live -v $D/prof:/prof \
  -v "$JIT/flashinfer":/root/.cache/flashinfer -v "$JIT/fa-cute":/root/.cache/fa-cute -v "$JIT/triton":/root/.triton \
  -v "$JIT/nv":/root/.nv -v "$JIT/dj":/root/.dj -v "$JIT/tilelang":/root/.tilelang \
  -v $HOOK/mimo_v2.29468.py:$V/model_executor/models/mimo_v2.py:ro \
  -v $HOOK/mimo_v2_mtp.29468.py:$V/model_executor/models/mimo_v2_mtp.py:ro \
  -v $HOOK/overlay/vllm/envs.py:$V/envs.py:ro \
  -v $HOOK/overlay/vllm/model_executor/model_loader/utils.py:$V/model_executor/model_loader/utils.py:ro \
  -v $HOOK/overlay/vllm/model_executor/offloader/base.py:$V/model_executor/offloader/base.py:ro \
  -v $HOOK/overlay/vllm/model_executor/offloader/uva.py:$V/model_executor/offloader/uva.py:ro \
  -v $HOOK/overlay/kv_cache_utils.29468.py:$V/v1/core/kv_cache_utils.py:ro \
  -v $HOOK/overlay/attn/flash_attn.py:$V/v1/attention/backends/flash_attn.py:ro \
  -v $HOOK/overlay/attn/flash_attn_diffkv.py:$V/v1/attention/backends/flash_attn_diffkv.py:ro \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  -e CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0} -e VLLM_LOGGING_LEVEL=INFO -e VLLM_USE_DEEP_GEMM=0 -e VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
  -e FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 -e FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/root/.cache/fa-cute \
  -e HOTSPLIT_COUNTS=/w/expert_hist_mix.json -e HOTSPLIT_WEIGHTS=decode=1,prefill=0 "${ROWMAP_ENV[@]}" \
  -e HOTSPLIT_LIVE_COUNTS=/live/counts.json -e HOTSPLIT_LIVE_SECS=600 -e HOTSPLIT_MAX_TOKENS=$MNBT \
  -e HOTSPLIT_PEER_CHECK=${PEER_CHECK:-0} -e PEER_SHM=/dev/shm/vllm_peer_mimo -e PEER_MAX_ROWS=${PEER_MAX_ROWS:-8192} \
  -e HOTSPLIT_FUSED_SEND=${FUSED_SEND:-1} -e HOTSPLIT_STAGE_SLOTS=${STAGE_SLOTS:-128} -e HOTSPLIT_STAGE_OVERLAP=${STAGE_OVERLAP:-0} \
  -e HOTSPLIT_STAGE_PROGRAMS=${STAGE_PROGRAMS:-304} -e HOTSPLIT_STAGE_CHECK=${STAGE_CHECK:-0} \
  "$IMAGE" --membind=0 vllm serve \
  --model /model --served-model-name mimo26-pro --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb 320 --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  "${MOE_ARGS[@]}" --max-model-len "$CTX" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" \
  --gpu-memory-utilization ${GPU_UTIL:-0.96} --enable-prefix-caching --generation-config vllm --kv-cache-dtype ${KVDTYPE:-fp8} \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --tool-call-parser mimo --reasoning-parser mimo --enable-auto-tool-choice \
  "${PROF_ARGS[@]}" ${EXTRA:-} --host 0.0.0.0 --port 30007
echo "launched $NAME (moe=${MOE:-trtllm} rowmap=${ROWMAP} peer=${PEER:-1} check=${PEER_CHECK:-0} stage=${STAGE_SLOTS:-128} fused=${FUSED_SEND:-1} ctx=$CTX seqs=$SEQS mnbt=$MNBT)"
