#!/usr/bin/env bash
# Wait until the server answers /health or its log shows a failure. Usage: waitready.sh <label> [timeout]
L=/home/jasonc/ds41f-exp/runs/${1:?label}/console.log; limit=${2:-5400}; t0=$(date +%s)
while true; do
  curl -fsS --max-time 5 http://127.0.0.1:${PORT:-30000}/health >/dev/null 2>&1 && { echo "ready after $(( $(date +%s)-t0 ))s"; exit 0; }
  grep -qE "Engine core initialization failed|EngineCore failed|^RuntimeError|Traceback \(most recent" "$L" 2>/dev/null && { echo FAILED; grep -nE "Error|Traceback" "$L" | head -8; exit 1; }
  (( $(date +%s)-t0 > limit )) && { echo TIMEOUT; tail -5 "$L"; exit 2; }
  sleep 15
done
