#!/usr/bin/env bash
# Print the DOCKER_MOUNTS value for launch-m3.sh / swap-to-m3v2.sh with the NVFP4 Engram overlay.
#
#   DOCKER_MOUNTS="$(/home/jasonc/research/megamoe/engram-nvfp4/lane-mounts.sh)"            # launcher unchanged
#   DOCKER_MOUNTS="$(/home/jasonc/research/megamoe/engram-nvfp4/lane-mounts.sh merged)"     # needs MODEL override
#
# A non-empty DOCKER_MOUNTS makes swap-to-m3v2.sh skip its own vllm#58132 mounts, so they are
# included here (REPLAY_OVERLAY=0 leaves them out).
#
# files  (default): launch-m3.sh as is (MODEL=/home/jasonc/models/DeepSeek-V4.1-Flash -> /model).
#                   The merged view's config.json, index and the two NVFP4 shards are bind-mounted
#                   over the same names in /model, which then reads exactly like the merged view.
# merged:           for a launcher whose MODEL can point at the merged view (mounted at /model);
#                   mounts the view's link targets where its relative links resolve
#                   (/model/../DeepSeek-V4.1-Flash, /model/../DeepSeek-V4.1-Flash-engram-nvfp4).
set -euo pipefail
MODE=${1:-files}
D=/usr/local/lib/python3.12/dist-packages/vllm
ENG=/home/jasonc/research/megamoe/engram-nvfp4/vllm
R58=/home/jasonc/research/megamoe/overlay-58132/tree/vllm
MERGED=/data/checkpoints/DeepSeek-V4.1-Flash-engram-nvfp4-merged
NV=/data/checkpoints/DeepSeek-V4.1-Flash-engram-nvfp4
M=()
if [ "${REPLAY_OVERLAY:-1}" = 1 ]; then
  for f in config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py \
           models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py; do
    M+=("$R58/$f:$D/$f:ro")
  done
fi
for f in models/deepseek_v41/common/engram.py models/deepseek_v41/nvidia/engram.py; do
  M+=("$ENG/$f:$D/$f:ro")
done
case "$MODE" in
  files)
    M+=("$MERGED/config.json:/model/config.json:ro"
        "$MERGED/model.safetensors.index.json:/model/model.safetensors.index.json:ro"
        "$NV/hf/model-00047-of-00048.safetensors:/model/model-00047-of-00048.safetensors:ro"
        "$NV/hf/model-00048-of-00048.safetensors:/model/model-00048-of-00048.safetensors:ro")
    ;;
  merged)
    M+=("/data/checkpoints/DeepSeek-V4.1-Flash:/DeepSeek-V4.1-Flash:ro" "$NV:/DeepSeek-V4.1-Flash-engram-nvfp4:ro")
    ;;
  *) echo "usage: $0 [files|merged]" >&2; exit 2 ;;
esac
echo "${M[*]}"
