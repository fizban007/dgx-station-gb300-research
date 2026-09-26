#!/usr/bin/env bash
# C8/C16 TTFT at the A/B's request count (5xC) for the three open frontend/executor configs.
D=/home/jasonc/research/qwen38
export CONCS="8 16" MULT=5 PASSES=1
LABEL=rust-uni-5x RUST_MP=0 DOCKER_ENV=VLLM_USE_RUST_FRONTEND=1 "$D/ttft-quick.sh"
LABEL=mp-only-5x RUST_MP=0 EXTRA="--distributed-executor-backend mp" "$D/ttft-quick.sh"
LABEL=api4-5x RUST_MP=0 EXTRA="--api-server-count 4" "$D/ttft-quick.sh"
echo "=== done $(date +%H:%M:%S)"
