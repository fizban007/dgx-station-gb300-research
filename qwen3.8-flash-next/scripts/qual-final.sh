#!/usr/bin/env bash
# Qualify the 2026-09-25 launcher defaults (Python frontend + uni, CG=8192, VISION=video, qwen3_xml tools, MTP3) on the
# running qwen-up: the lane suite (GSM8K-200, catid decode C1-C64, cold prefill 8K-128K), needles at ~125K and ~250K,
# and the 128K thinking-loop check (8 reps of C1 + C8 from stinger, the pp0 protocol of pp-loop-test2.sh).
set -uo pipefail
LABEL=${LABEL:-vllm-py-cg8192-mtp3}
D=/home/jasonc/research/qwen38
echo "=== suite $(date +%H:%M:%S)"
"$D/bench-qwen.sh" "$LABEL"
echo "=== needles $(date +%H:%M:%S)"
for t in 125000 250000; do
  HOST=127.0.0.1 PORT=30006 MODEL_NAME=qwen38-flash-next /home/jasonc/venvs/bench/bin/python "$D/needle_test.py" "$LABEL-$t" "$t"
done
echo "=== 128K loop check $(date +%H:%M:%S)"
for i in $(seq "${REPS:-8}"); do
  out=/home/jasonc/spark_vllm/qwen38-loop-$LABEL-$i
  ssh -o BatchMode=yes stinger "cd /home/jasonc/spark_vllm/llm-inference-bench && python3 llm_decode_bench.py \
    --host http://192.168.1.99 --port 30006 --model qwen38-flash-next --concurrency 1,8 --contexts 128k --duration 30 \
    --max-tokens 8192 --skip-prefill --display-mode plain --no-hw-monitor --no-resume --output $out.json \
    < /dev/null > $out.log 2>&1; python3 /tmp/looptally.py $out.json '$LABEL run$i'" < /dev/null
done
echo "=== done $(date +%H:%M:%S)"
