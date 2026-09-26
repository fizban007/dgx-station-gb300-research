#!/usr/bin/env bash
# Phase C: DSpark speculative-decoding sweep on M3 v2 (balanced split, 285 hot). One boot per variant.
# Usage: phase_c.sh <label> <ksched json> [spec extra json fragment]
set -uo pipefail
LABEL=$1; KS=$2; EXTRA=${3:-}
D=/home/jasonc/research/megamoe
KSCHED="$KS" SPEC_EXTRA="$EXTRA" GPU_UTIL=0.95 ROWMAP=rowmap-mix-v1.json MEGA_COLD_TRT=0 MEGA_COUNT=/prof/route-counts.json \
  $D/swap-to-m3v2.sh > $D/logs/swap-$LABEL.log 2>&1 || { echo "$LABEL: boot failed"; tail -5 $D/logs/swap-$LABEL.log; docker logs m3v2 2>&1 | grep -iE "Error|raise" | tail -5; exit 1; }
cd $D
/home/jasonc/venvs/bench/bin/python warmup.py --check 2>&1 | tail -1
/home/jasonc/venvs/bench/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "/home/jasonc/research/megamoe")
import measure_wait as m
open(m.OVERRIDE, "w").write("{}")
m.run(4, 50)
for c in (8, 16):
    d = m.run(c, 2000 + c)
    print(f"  wait check C{c}: {d['tok_s']} tok/s, {d['ms_per_pass']} ms/pass, GB300 wait {d['wait_ms_per_pass']} ms/pass")
PY
PORT=30006 MODEL_NAME=dsv41-flash-uva /home/jasonc/ds41f-exp/bench_decode.sh "phaseC-$LABEL" "1 4 8 16" 2>&1 | grep -E "^decode"
python3 -c "import json; d=json.load(open('$D/logs/peer_stats.json')); print('  6000 buckets:', {k: v['mean_us'] for k, v in d.items() if int(k) <= 64})"
