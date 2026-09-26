#!/usr/bin/env bash
# Start serve.sh in its own session; record the process group. Usage: launch.sh <label> [args...]
label=${1:?label}; mkdir -p /home/jasonc/ds41f-exp/runs/$label
setsid /home/jasonc/ds41f-exp/serve.sh "$@" > /home/jasonc/ds41f-exp/runs/$label/console.log 2>&1 < /dev/null &
echo $! > /home/jasonc/ds41f-exp/runs/$label/pgid; echo "launched $label pgid $(cat /home/jasonc/ds41f-exp/runs/$label/pgid)"
