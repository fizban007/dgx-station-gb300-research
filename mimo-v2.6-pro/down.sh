#!/usr/bin/env bash
# Stop the MiMo-V2.6-Pro lane (vLLM container + RTX PRO 6000 sidecar) and free both GPUs.
D=/home/jasonc/research/mimo-pro
docker stop -t 30 mimo-pro >/dev/null 2>&1; docker rm mimo-pro >/dev/null 2>&1
if [ -f $D/logs/peer_server.pid ] && kill -0 "$(cat $D/logs/peer_server.pid)" 2>/dev/null; then kill "$(cat $D/logs/peer_server.pid)"; fi
sleep 3; rm -f /dev/shm/vllm_peer_mimo 2>/dev/null
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
