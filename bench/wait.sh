#!/usr/bin/env bash
# Wait for the server's /health; print the log tail on failure. Usage: wait.sh <label> [timeout_s]
label=${1:?label}; limit=${2:-5400}; log=/home/jasonc/ds41f-exp/runs/$label/server.log; t0=$(date +%s)
while true; do
  if curl -fsS --max-time 5 "http://127.0.0.1:${PORT:-30000}/health" >/dev/null 2>&1; then echo "ready after $(( $(date +%s) - t0 ))s"; exit 0; fi
  if ! pgrep -f "runs/$label" >/dev/null && ! pgrep -f "vllm.entrypoints.cli.main serve" >/dev/null; then echo "server exited"; tail -40 "$log"; exit 1; fi
  if (( $(date +%s) - t0 > limit )); then echo "timeout"; tail -40 "$log"; exit 2; fi
  sleep 10
done
