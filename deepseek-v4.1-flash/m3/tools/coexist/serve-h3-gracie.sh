#!/usr/bin/env bash
# MiniMax-H3 (FL2VA partition) on vllm-omni, on the RTX PRO 6000 (SM120, 96 GB) via CDI, isolated from the Qwen lane on the GB300.
# 2026-10-01 variant of serve-h3.sh: DiT and VAEs stay resident on the 6000 (62 + 10 GB); only the 63 GB Qwen3-VL text encoder is
# streamed layer by layer from pinned Grace RAM (it runs once per prompt). DS-V4.1 is gone, so host RAM is no longer tight.
# --- original notes (GB300 / DS-V4.1 co-location) ---
# The DS-V4.1 lane leaves ~8 GiB of HBM, so this is vllm-omni's lowest-residency path (the compatibility
# --enable-layerwise-offload flag): DiT blocks and the Qwen3-VL text encoder are streamed block by block from pinned
# Grace RAM, and the VAEs sit in pinned RAM and are staged onto the GPU only for decode. (An explicit
# --diffusion-offload-config keeps the VAEs resident instead, ~10 GiB, which does not fit.)
# The RTX PRO 6000 is not usable from this process: on driver R595 a CUDA process runs in one modality (ATS = the
# GB300, NONATS = the RTX PRO), so it never sees both GPUs (the b12x sidecar pins itself to the 6000 by UUID).
# Host RAM: DiT 62 GB + text encoder 63 GB + VAEs 10 GB pinned. The container gets a memory cap so an overshoot kills
# H3, not the DS-V4.1 EngineCore (the largest process on the box); numactl keeps it on node 0 (Grace DRAM).
#   H3_PORT (8091)  MEM (150g)  IMAGE (vllm/vllm-omni:nightly-aarch64)  EXTRA (extra serve args)
set -euo pipefail
D=/home/jasonc/research/minimax-h3
NAME=${NAME:-minimax-h3}
GPU6000=GPU-c51e3fdb-81ba-2821-7021-a4ae8a599eb7
# GPU=gb300 (default since 2026-10-01: ~1 min per 5 s 1024x576 clip, shares the GB300 with the Qwen lane under its 70 GiB KV cap)
# or GPU=6000 (CDI; ~6 min per clip, barely ahead of ripper, so not used). Serves the LAN on 0.0.0.0:$PORT like ripper's
# instance (same served name and task type), for the assistant's media service (VIDEO_API_BASE).
GPU=${GPU:-gb300}
IMAGE=${IMAGE:-vllm/vllm-omni:nightly-aarch64}
PORT=${H3_PORT:-8091}  # not PORT: the DS41 test scripts export PORT=30006 (both H3 restarts failed on it 2026-10-02)
# H3_MODE=fl2va (default): FL2VA only (t2va, fl2va); DiT + VAEs resident, text encoder streamed by layer.
# H3_MODE=combined: one server with both DiTs (FL2VA + Ref2VA) behind the shared text encoder and VAEs; each request's
#   extra_params.task (t2va / fl2va / ref2va) picks the DiT. Weights live in pinned Grace RAM and are streamed to the
#   GPU layer by layer (see below), so MEM defaults higher.
H3_MODE=${H3_MODE:-fl2va}
case "$H3_MODE" in
  fl2va) MODEL_DIR=/data/checkpoints/MiniMax-H3/FL2VA; TASK=fl2va; OFFLOAD=${OFFLOAD:-'{"mode":"layer","components":["text_encoder"]}'}; MEM=${MEM:-220g} ;;
  # FL2VA only, DiT streamed too: ~20 GiB peak HBM and only FL2VA pinned in Grace RAM (no Ref2VA), for sharing the box
  # with DS41 (which needs most of Grace RAM for Engram and cold experts).
  fl2va-lean) MODEL_DIR=/data/checkpoints/MiniMax-H3/FL2VA; TASK=fl2va; OFFLOAD=${OFFLOAD:-'{"mode":"layer","components":["dit","text_encoder"]}'}; MEM=${MEM:-220g} ;;
  # Lowest residency: vllm-omni's legacy --enable-layerwise-offload also parks the VAEs (~10 GB) in pinned RAM and
  # stages them onto the GPU only for the final decode (the new offload config keeps them resident).
  fl2va-min) MODEL_DIR=/data/checkpoints/MiniMax-H3/FL2VA; TASK=fl2va; OFFLOAD=legacy; MEM=${MEM:-220g} ;;
  # Layer streaming of both DiTs + encoder from pinned Grace RAM over NVLink-C2C: measured 2026-10-02 as fast as the
  # module-level swap (6 s/30-step clip 52.6 s vs 58 s) with H3 peaking at 20-28 GiB of HBM instead of ~75-84 GiB
  # (~251 GiB pinned host RAM). OFFLOAD='{"mode":"module","components":["dit","text_encoder"]}' restores the swap.
  combined) MODEL_DIR=/data/checkpoints/MiniMax-H3; TASK=combined; OFFLOAD=${OFFLOAD:-'{"mode":"layer","components":["dit","text_encoder"]}'}; MEM=${MEM:-320g} ;;
  *) echo "H3_MODE must be fl2va, fl2va-lean, fl2va-min or combined" >&2; exit 1 ;;
esac
MODEL=$MODEL_DIR
if [ "$OFFLOAD" = legacy ]; then OFFLOAD_ARGS=(--enable-layerwise-offload); else OFFLOAD_ARGS=(--diffusion-offload-config "$OFFLOAD"); fi
GB300=GPU-c146511a-0326-7ddc-4346-998d61a64b34
mkdir -p "$D/cache"
docker rm -f "$NAME" >/dev/null 2>&1 || true
# Do not start while something still listens on the port (the new server would die with "Address already in use").
for _ in $(seq 180); do ss -ltn "sport = :$PORT" | grep -q LISTEN || break; sleep 1; done
case "$GPU" in 6000) DEV=(--device "nvidia.com/gpu=$GPU6000") ;; gb300) DEV=(--gpus "\"device=GPU-c146511a-0326-7ddc-4346-998d61a64b34\"") ;; *) echo "GPU must be 6000 or gb300" >&2; exit 1 ;; esac
docker run -d --name "$NAME" "${DEV[@]}" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 -e HF_HUB_OFFLINE=1 \
  --ipc host --network host --memory "$MEM" --memory-swap "$MEM" --ulimit memlock=-1 --cap-add SYS_NICE \
  -v "$MODEL":"$MODEL":ro -v "$D/cache":/root/.cache -v /usr/bin/numactl:/usr/local/bin/numactl:ro \
  --entrypoint /usr/local/bin/numactl \
  "$IMAGE" --membind=0 vllm serve "$MODEL" --omni --trust-remote-code --served-model-name minimax-h3 --task-type "$TASK" --host 0.0.0.0 --port "$PORT" \
  --num-gpus 1 "${OFFLOAD_ARGS[@]}" --vae-use-tiling --enable-diffusion-pipeline-profiler \
  --enforce-eager ${EXTRA:-}
echo "started $NAME on 0.0.0.0:$PORT ($GPU, $H3_MODE) (docker logs -f $NAME)"
