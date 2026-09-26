#!/usr/bin/env bash
# Stop a launched server by process group and wait until the GPU is released. Usage: stop.sh <label>
f=/home/jasonc/ds41f-exp/runs/${1:?label}/pgid; [ -f "$f" ] || exit 0
kill -TERM -- -"$(cat "$f")" 2>/dev/null; for _ in $(seq 60); do kill -0 -- -"$(cat "$f")" 2>/dev/null || break; sleep 2; done
kill -KILL -- -"$(cat "$f")" 2>/dev/null; rm -f "$f"
for _ in $(seq 60); do nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q c146511a || break; sleep 2; done; echo stopped
