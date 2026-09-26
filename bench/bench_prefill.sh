#!/usr/bin/env bash
# catid's prefill recipe: unique random-token prompts, one output token, cache reset before each point.
# Usage: bench_prefill.sh <run-label> [isl list] [concurrency list]
set -uo pipefail
label=${1:?label}; out=/home/jasonc/ds41f-exp/runs/$label/prefill; mkdir -p "$out"; cd "$out"
source /home/jasonc/venvs/bench/bin/activate
python /home/jasonc/ds41f-exp/bench_prefill.py --engine vllm --host 127.0.0.1 --port "${PORT:-30000}" \
  --isl "${2:-16384,32768,65536,131072}" --concurrency "${3:-1,4,16}" --timeout 1800 --wait-ready 60 \
  --output "$out/prefill.jsonl" --label "$label" 2>&1 | tee "$out/prefill.log"
