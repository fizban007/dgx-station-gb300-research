#!/usr/bin/env bash
# Boot + A/B suite for each arm in turn: "<tag> <env assignments...>" per argument. Waits for LAN idle before each boot.
set -uo pipefail
D=/home/jasonc/research/megamoe
for arm in "$@"; do
  read -r tag envs <<< "$arm"
  for _ in $(seq 1 180); do
    [ "$(ss -tn state established '( sport = :30006 )' | tail -n +2 | grep -vc 127.0.0.1)" = 0 ] && break
    sleep 10
  done
  echo "=== boot $tag ($envs) $(date +%H:%M:%S)"
  if ! env $envs IMAGE=vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b REASONING_EFFORT=high \
      "$D/swap-to-m3v2.sh" > "$D/logs/swap-$tag.log" 2>&1; then
    echo "boot $tag failed"; tail -5 "$D/logs/swap-$tag.log"; exit 1
  fi
  docker logs dsv41-flash-megamoe 2>&1 | grep -E "MEGA_PEER (fused|vllm.models.deepseek_v41)" | sort -u
  "$D/ab-suite.sh" "$tag" > "$D/logs/ab-$tag.out" 2>&1
  grep -E "GSM8K|reasoning C1|\"C\": 1" "$D/logs/ab-$tag.out"; grep -E "^decode" "$D/logs/ab-$tag/catid.log"
  grep -E "^\s+16384" "$D/logs/ab-$tag/prefill.log"
done
echo "chain done"
