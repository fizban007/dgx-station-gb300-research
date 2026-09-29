#!/usr/bin/env bash
# Stop whatever serves :30006 (v20-peer2, m3, m3-prof, dsv41-flash-megamoe), start the b12x v2 sidecar on the 6000, boot M3 with peer v2.
set -euo pipefail
# Defaults = the configuration chosen 2026-09-24: balanced 285/99 split, no Grace copy, route counter on,
# 95% HBM target; DSpark k=5 to 4 streams, 3 above, probabilistic drafting, LL GEMM, fused send v2
# (2026-09-28 evening; all set in launch-m3.sh).
export GPU_UTIL=${GPU_UTIL:-0.95} MEGA_COLD_TRT=${MEGA_COLD_TRT:-0} MEGA_COUNT=${MEGA_COUNT-/prof/route-counts.json}
PEER_DIR=/home/jasonc/ds41f-exp/peer
LOG=/home/jasonc/research/megamoe/logs
mkdir -p "$LOG"
for c in v20-peer2 m3 m3-prof dsv41-flash-megamoe; do docker stop "$c" >/dev/null 2>&1 || true; done
if [ -f "$PEER_DIR/peer_server.pid" ]; then
  P=$(cat "$PEER_DIR/peer_server.pid")
  kill "$P" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 "$P" 2>/dev/null || break; sleep 1; done
fi
cd "$PEER_DIR"
(CUDA_VISIBLE_DEVICES=GPU-c51e3fdb-81ba-2821-7021-a4ae8a599eb7 PATH=/home/jasonc/venvs/vllm-karmic/bin:$PATH \
  setsid numactl --membind=0 /home/jasonc/venvs/vllm-karmic/bin/python peer_server2.py \
  --rowmap /home/jasonc/research/megamoe/hook/${ROWMAP:-rowmap-mix-v1.json} > "$LOG/peer_server2.log" 2>&1 < /dev/null &
  echo $! > peer_server.pid)
SP=$(cat "$PEER_DIR/peer_server.pid")
until grep -q "^serving" "$LOG/peer_server2.log"; do
  if ! kill -0 "$SP" 2>/dev/null; then echo "sidecar died; not starting M3"; tail -3 "$LOG/peer_server2.log"; exit 1; fi
  sleep 3
done
tail -1 "$LOG/peer_server2.log"
chmod 666 /dev/shm/vllm_peer_tier2
docker rm -f dsv41-flash-megamoe >/dev/null 2>&1 || true
# vllm#58132 (decoder SWA bounded replay; unmerged, 2026-09-28) as a Python overlay over the image: prefill
# +57-61% on real text, decode and quality unchanged (FINDINGS.md section 8). REPLAY_OVERLAY=0 boots the stock files.
# Drop it once the image carries the PR.
REPLAY_OVERLAY=${REPLAY_OVERLAY:-1}
if [ "$REPLAY_OVERLAY" = 1 ] && [ -z "${DOCKER_MOUNTS:-}" ]; then
  O=/home/jasonc/research/megamoe/overlay-58132/tree/vllm
  D=/usr/local/lib/python3.12/dist-packages/vllm
  DOCKER_MOUNTS=""
  for f in config/cache.py models/deepseek_v41/decoder_replay_layers.py models/deepseek_v41/nvidia/model.py \
           models/deepseek_v41/nvidia/model_state.py models/deepseek_v41/nvidia/vl_model.py; do
    DOCKER_MOUNTS="$DOCKER_MOUNTS $O/$f:$D/$f:ro"
  done
  export DOCKER_MOUNTS
fi
# CUDA_LOG_FILE=stderr: the driver names the failing call if graph capture dies again (2026-09-25 one-off
# "operation not permitted" at capture_end). It logs ~100 harmless lines at startup and nothing after.
DOCKER_ENV=${DOCKER_ENV:-CUDA_LOG_FILE=stderr} NAME=dsv41-flash-megamoe MEGA_PEER=2 ROWMAP=${ROWMAP:-rowmap-mix-v1.json} \
  /home/jasonc/research/megamoe/launch-m3.sh
until curl -s -m 3 http://127.0.0.1:30006/v1/models | grep -q dsv41; do
  sleep 10
  if ! docker ps -q -f name=^dsv41-flash-megamoe$ | grep -q .; then
    echo "M3 exited during startup; see: docker logs dsv41-flash-megamoe"; docker logs dsv41-flash-megamoe 2>&1 | grep -E "Error:|EngineCore failed" | tail -3; exit 1
  fi
done
echo "M3 v2 serving on :30006"
if [ "$REPLAY_OVERLAY" = 1 ] && ! docker logs dsv41-flash-megamoe 2>&1 | grep -q "Decoder SWA bounded replay"; then
  echo "WARNING: REPLAY_OVERLAY=1 but the boot log has no 'Decoder SWA bounded replay' line; prefill runs without it"
fi
