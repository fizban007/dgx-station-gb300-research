#!/usr/bin/env bash
# The DS41 lane that coexists with MiniMax-H3 on the GB300 (chosen 2026-10-03): rowmap h265 (more cold experts on the
# 6000), NVFP4 Engram in registered host memory, nvfp4_ds_mla KV, GPU_UTIL 0.885 (~3.4M KV tokens, ~2.4 GiB left for a
# 15 s combined H3 render). swap-to-m3v2.sh's own defaults are the pre-H3 lane and do not fit next to H3.
set -euo pipefail
M=/home/jasonc/research/megamoe
cd "$M"
GPU_UTIL=${GPU_UTIL:-0.885} ROWMAP=${ROWMAP:-rowmap-mix-v1-h265.json} DOCKER_MOUNTS="$(engram-nvfp4/lane-mounts.sh)" \
  EXTRA="${EXTRA:---kv-cache-dtype nvfp4_ds_mla}" exec ./swap-to-m3v2.sh "$@"
