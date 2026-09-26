#!/usr/bin/env bash
# Quick TTFT smoke for one server config: catid decode recipe (8K in / 1K out, T=0) with C warm-ups and MULTxC requests
# per cell, PASSES passes on the same server. CONCS defaults to "1 8 16", MULT to 2, PASSES to 2.
# Caveat: with MULT=2 the fixed 1,024-token outputs keep streams in synchronized waves, which inflates C8-C16 TTFT;
# use MULT=5 (the A/B's count) for C8+ comparisons. C1 is unaffected. Reference points from ab-rust-mp.sh (2 runs per arm):
#   TTFT p50 base (Python frontend, uni) C1 153 / C8 257 / C16 364 ms; Rust + mp C1 136 / C8 170 / C16 179 ms.
# Usage: LABEL=mp-only RUST_MP=0 EXTRA="--distributed-executor-backend mp" ttft-quick.sh
set -uo pipefail
D=/home/jasonc/research/qwen38
LABEL=${LABEL:?label}
docker rm -f qwen-up qwen-sg >/dev/null 2>&1
for _ in $(seq 120); do
  [ "$(nvidia-smi --id=GPU-c146511a-0326-7ddc-4346-998d61a64b34 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 8000 ] && break
  sleep 5
done
MTP=3 "$D/launch-qwen-upstream.sh"
for _ in $(seq 240); do sleep 5; curl -sf -m 2 localhost:30006/v1/models >/dev/null && break; done
docker logs qwen-up > "$D/logs/server-ttft-$LABEL.log" 2>&1
grep -E "Executor|Rust frontend|Default vLLM sampling" "$D/logs/server-ttft-$LABEL.log" | sed 's/.*\] //' | cut -c1-150 | sort -u
source /home/jasonc/venvs/bench/bin/activate
for pass in $(seq "${PASSES:-2}"); do
  for c in ${CONCS:-1 8 16}; do
    out=$D/runs/ttft-$LABEL/p$pass; mkdir -p "$out"
    printf 'n\n' | python /home/jasonc/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port 30006 \
      --model qwen38-flash-next --concurrency "$c" --contexts 8k --request-count $((c * ${MULT:-2})) --warmup-request-count "$c" \
      --max-tokens 1024 --temperature 0 --token-targeting exact --skip-prefill --display-mode plain --no-hw-monitor \
      --no-resume --output "$out/c$c.json" > "$out/c$c.log" 2>&1
    python3 -c "
import json; r = json.load(open('$out/c$c.json'))['results'][0]
print(f'$LABEL pass $pass C$c  TTFT p50 {r[\"ttft_p50\"] * 1e3:5.0f} ms  p90 {r[\"ttft_p90\"] * 1e3:5.0f} ms  {r[\"aggregate_tps\"]:8.1f} tok/s')"
  done
done
