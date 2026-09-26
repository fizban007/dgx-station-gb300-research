#!/usr/bin/env bash
# Bring up the MiMo-V2.6-Pro lane: RTX PRO 6000 sidecar first (loads the rowmap's peer experts, self-tests, "serving"),
# then the GB300 vLLM container (launch-pro.sh defaults = the 2026-09-25 v6 config). ~5 min sidecar + ~8 min vLLM.
set -euo pipefail
D=/home/jasonc/research/mimo-pro
ROWMAP=${ROWMAP:-rowmap-3tier-v3.json}
docker rm -f mimo-pro >/dev/null 2>&1 || true
if [ -f $D/logs/peer_server.pid ] && kill -0 "$(cat $D/logs/peer_server.pid)" 2>/dev/null; then kill "$(cat $D/logs/peer_server.pid)"; sleep 5; fi
rm -f /dev/shm/vllm_peer_mimo
cd $D/hook
(CUDA_VISIBLE_DEVICES=GPU-c51e3fdb-81ba-2821-7021-a4ae8a599eb7 PYTHONPATH=/home/jasonc/b12x PYTHONUNBUFFERED=1 PEER_MAX_ROWS=8192 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/home/jasonc/venvs/vllm-karmic/bin:$PATH \
  setsid numactl --membind=0 /home/jasonc/venvs/vllm-karmic/bin/python peer_server_mimo.py --rowmap $D/hook/$ROWMAP \
  > $D/logs/peer_server_mimo.log 2>&1 < /dev/null & echo $! > $D/logs/peer_server.pid)
until grep -qE "^serving|SELFTEST FAILED|Traceback" $D/logs/peer_server_mimo.log; do
  kill -0 "$(cat $D/logs/peer_server.pid)" 2>/dev/null || { echo "sidecar died; see logs/peer_server_mimo.log"; exit 1; }
  sleep 5
done
grep -q "^serving" $D/logs/peer_server_mimo.log || { echo "sidecar self-test failed"; exit 1; }
ROWMAP=$ROWMAP $D/launch-pro.sh
until curl -s -m 3 http://127.0.0.1:30007/v1/models | grep -q mimo26-pro; do
  docker ps -q -f name=^mimo-pro$ | grep -q . || { echo "vLLM exited; see docker logs mimo-pro"; exit 1; }
  sleep 10
done
echo "MiMo-V2.6-Pro serving on :30007 (mimo26-pro)"
