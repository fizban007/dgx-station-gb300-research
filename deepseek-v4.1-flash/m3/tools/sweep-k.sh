#!/usr/bin/env bash
# DSpark k-schedule sweep: per arm "<tag> <KSCHED|current>". "current" benchmarks the running server without a reboot.
# Each arm: reasoning closed-loop C1/4/8/12/16/24, catid decode C8/C16, prose knee. Waits for LAN idle before boots.
set -uo pipefail
D=/home/jasonc/research/megamoe
for arm in "$@"; do
  read -r tag ks <<< "$arm"
  if [ "$ks" != current ]; then
    for _ in $(seq 1 180); do
      [ "$(ss -tn state established '( sport = :30006 )' | tail -n +2 | grep -vc 127.0.0.1)" = 0 ] && break
      sleep 10
    done
    echo "=== boot $tag KSCHED=$ks $(date +%H:%M:%S)"
    if ! KSCHED="$ks" "$D/swap-to-m3v2.sh" > "$D/logs/swap-$tag.log" 2>&1; then
      echo "boot $tag failed"; tail -5 "$D/logs/swap-$tag.log"; exit 1
    fi
  fi
  echo "=== bench $tag $(docker inspect dsv41-flash-megamoe --format '{{json .Args}}' | grep -o 'num_speculative_tokens_per_batch_size[^}]*') $(date +%H:%M:%S)"
  python3 "$D/reason_bench.py" "$tag"
  (export PORT=30006 MODEL_NAME=dsv41-flash-uva
   /home/jasonc/ds41f-exp/bench_decode.sh "k-$tag" "8 16" | grep -E "^decode"
   bash /home/jasonc/research/upstream/knee.sh "k-$tag" | grep -E "conc= (1|4|8|16):")
done
echo "sweep done"
