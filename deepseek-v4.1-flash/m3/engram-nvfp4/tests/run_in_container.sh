#!/usr/bin/env bash
# Run a test script on the RTX PRO 6000 in the DS41 lane image, with or without the overlay.
#   tests/run_in_container.sh overlay|stock <script.py> [args...]
# Checkpoints are mounted read-only at their host paths (the merged view's relative links
# resolve); the container is capped at 40 GiB of host memory (pinned + page cache).
set -euo pipefail
MODE=$1; shift
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}
GPU=GPU-c51e3fdb-81ba-2821-7021-a4ae8a599eb7   # RTX PRO 6000 only; never the GB300
D=/usr/local/lib/python3.12/dist-packages/vllm
MOUNTS=()
if [ "$MODE" = overlay ]; then
  for f in models/deepseek_v41/common/engram.py models/deepseek_v41/nvidia/engram.py; do
    MOUNTS+=(-v "$HERE/vllm/$f:$D/$f:ro")
  done
elif [ "$MODE" != stock ]; then
  echo "mode must be overlay or stock" >&2; exit 2
fi
mkdir -p "$HERE/tests/results" "$HERE/tests/.cache"
exec docker run --rm --device "nvidia.com/gpu=$GPU" \
  --memory "${MEM:-40g}" --memory-swap "${MEM:-40g}" \
  --ulimit memlock=-1 --cap-add IPC_LOCK --security-opt label=disable \
  --user "$(id -u):$(id -g)" -e HOME=/work/tests/.cache -e TRITON_CACHE_DIR=/work/tests/.cache/triton \
  -e VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING} -e RESULTS=/work/tests/results -e TEST_MODE="$MODE" \
  ${EXTRA_ENV:-} \
  -v /data/checkpoints:/data/checkpoints:ro -v "$HERE":/work "${MOUNTS[@]}" \
  -w /work/tests --entrypoint python3 "$IMAGE" "$@"
