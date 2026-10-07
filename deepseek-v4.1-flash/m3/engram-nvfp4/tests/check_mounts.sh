#!/usr/bin/env bash
# Mount /model exactly as launch-m3.sh would (-v $MODEL:/model:ro, then each DOCKER_MOUNTS entry)
# for both lane-mounts.sh modes, and run check_mounts.py inside (no server is started).
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}
for MODE in files merged; do
  if [ "$MODE" = files ]; then MODEL=/home/jasonc/models/DeepSeek-V4.1-Flash
  else MODEL=/home/jasonc/models/DeepSeek-V4.1-Flash-engram-nvfp4-merged; fi
  DOCKER_MOUNTS=$("$HERE/lane-mounts.sh" "$MODE")
  MOUNT_ARGS=()
  for mnt in $DOCKER_MOUNTS; do MOUNT_ARGS+=(-v "$mnt"); done
  echo "== $MODE: MODEL=$MODEL, ${#MOUNT_ARGS[@]} mount args"
  docker run --rm --device nvidia.com/gpu=GPU-c51e3fdb-81ba-2821-7021-a4ae8a599eb7 --memory 16g \
    --security-opt label=disable --user "$(id -u):$(id -g)" -e HOME=/work/tests/.cache -e MOUNT_MODE="$MODE" \
    -e VLLM_LOGGING_LEVEL=ERROR -v "$MODEL":/model:ro "${MOUNT_ARGS[@]}" -v "$HERE":/work -w /work/tests \
    --entrypoint python3 "$IMAGE" check_mounts.py 2>&1 | grep -E '^\{' | tee "$HERE/tests/results/mounts_$MODE.json"
done
