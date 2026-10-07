#!/usr/bin/env bash
# Boot + arm-suite for each arm in turn: "<tag> <env assignments...>" per argument. Waits for LAN idle before each
# boot (no established non-local connection to :30006 for one check), then DeepGEMM warm-up, then the suite.
set -uo pipefail
D=/home/jasonc/research/megamoe
for arm in "$@"; do
  read -r tag envs <<< "$arm"
  for _ in $(seq 1 360); do
    [ "$(ss -tn state established '( sport = :30006 )' | tail -n +2 | grep -vc 127.0.0.1)" = 0 ] && break
    sleep 10
  done
  echo "=== boot $tag ($envs) $(date +%H:%M:%S)"
  if ! env $envs IMAGE=vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b REASONING_EFFORT=high \
      "$D/swap-to-m3v2.sh" > "$D/logs/swap-$tag.log" 2>&1; then
    echo "boot $tag failed"; tail -5 "$D/logs/swap-$tag.log"; exit 1
  fi
  echo "serving $(date +%H:%M:%S)"
  docker logs dsv41-flash-megamoe 2>&1 | grep -E "FlashInfer autotune pinned|pinned FlashInfer autotune file|Mega-mHC off|Saved [0-9]+ configs|source=config file|GPU KV cache size|Decoder SWA bounded replay" | cut -c1-240 | sort -u
  python3 "$D/warmup.py" > "$D/logs/warmup-$tag.log" 2>&1; tail -1 "$D/logs/warmup-$tag.log"
  "$D/arm-suite.sh" "$tag" > "$D/logs/arm-$tag.out" 2>&1
  grep -E "accept=|agent |heldout |non-local|GSM8K|conc= 1:|conc=16:" "$D/logs/arm-$tag.out"; grep -E "^decode" "$D/logs/arm-$tag/catid.log"
done
echo "chain done"
