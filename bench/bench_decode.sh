#!/usr/bin/env bash
# catid's decode recipe: 8,192 exact input, 1,024 forced output, temperature 0, 5xC requests after C warm-ups.
# Usage: bench_decode.sh <run-label> "<concurrencies>"
set -uo pipefail
label=${1:?label}; read -r -a cs <<< "${2:-1 2 4 8 16 32 64}"
out=/home/jasonc/ds41f-exp/runs/$label/decode; mkdir -p "$out"; cd "$out"
source /home/jasonc/venvs/bench/bin/activate
for c in "${cs[@]}"; do
  printf 'n\n' | python /home/jasonc/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port "${PORT:-30000}" \
    --model "${MODEL_NAME:-DeepSeek-V4.1-Flash}" --concurrency "$c" --contexts 8k --request-count $((c * 5)) --warmup-request-count "$c" \
    --max-tokens 1024 --temperature 0 --token-targeting exact --skip-prefill --display-mode plain --no-hw-monitor \
    --no-resume --output "c${c}.json" > "c${c}.log" 2>&1
  echo "C$c rc=$?"
done
python3 /home/jasonc/dgx_station_benchmarks/deepseek-v4.1-flash/recipes/summarize_decode.py "$out"
