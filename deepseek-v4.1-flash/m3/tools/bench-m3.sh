#!/usr/bin/env bash
# Same suite as v20+peer: Al-ENGR knee, catid decode C1/C8/C16, catid prefill 16K at C1.
set -uo pipefail
TAG=${1:-m3}
export PORT=30006 MODEL_NAME=dsv41-flash-uva
bash /home/jasonc/research/upstream/knee.sh "$TAG"
/home/jasonc/ds41f-exp/bench_decode.sh "$TAG" "1 8 16"
/home/jasonc/ds41f-exp/bench_prefill.sh "$TAG" 16384 1 | tail -3
python3 -c "
import struct;b=open('/dev/shm/vllm_peer_tier','rb').read(24);print('peer published %d completed %d timeouts %d'%struct.unpack('<qqq',b))"
