#!/usr/bin/env bash
# One T2VA request against serve-h3.sh; saves mp4 + response headers (stage durations, peak memory) under out/.
#   [H3_URL=http://host:port] gen.sh <tag> [steps=50] [width=1024] [height=576] [duration=5] [prompt]
set -euo pipefail
TAG=$1; STEPS=${2:-50}; W=${3:-1024}; H=${4:-576}; DUR=${5:-5}
PROMPT=${6:-"At dusk, a small tugboat pushes through a choppy harbor, gulls wheeling overhead, spray catching the orange light; the engine chugs and waves slap the hull."}
OUT=/home/jasonc/research/minimax-h3/out; mkdir -p "$OUT"
t=$(date +%s.%N)
curl -sS --max-time 14400 -D "$OUT/$TAG.headers" -o "$OUT/$TAG.mp4" -X POST ${H3_URL:-http://127.0.0.1:8091}/v1/videos/sync \
  -F "prompt=$PROMPT" -F "width=$W" -F "height=$H" -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F "num_inference_steps=$STEPS" -F 'flow_shift=12' -F 'seed=1101' \
  -F "extra_params={\"task\":\"t2va\",\"duration\":$DUR,\"audio_flow_shift\":3.0}"
echo "wall $(echo "$(date +%s.%N) - $t" | bc) s"
grep -i -E "^HTTP|x-stage-durations|x-peak-memory|x-inference-time" "$OUT/$TAG.headers"
ls -la "$OUT/$TAG.mp4"; file "$OUT/$TAG.mp4"
