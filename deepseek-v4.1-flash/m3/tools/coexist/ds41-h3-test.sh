#!/usr/bin/env bash
# DS41 (rowmap-mix-v1-h265, NVFP4 Engram, FP8 KV, GPU_UTIL 0.92) + MiniMax-H3 (H3_MODE=fl2va-min) on gracie, 2026-10-02:
# smoke + tools, GSM8K-200 (baseline 98.0%), needles to 1M, reasoning decode C1/C8/C16 (baseline 306/1,102/1,535),
# catid prefill (baseline ~44K tok/s), then coexistence: an H3 15 s render with DS41 C1 decode running and GB300
# memory sampled.
set -uo pipefail
M=/home/jasonc/research/megamoe; Q=/home/jasonc/research/qwen38; H=/home/jasonc/research/minimax-h3; L=$M/logs/ds41-h3-test
mkdir -p "$L"; TAG=ds41h3-h265-nvfp4e
export PORT=30006 MODEL_NAME=dsv41-flash-uva MODEL=dsv41-flash-uva HOST=127.0.0.1
echo "=== $(date +%H:%M) smoke"
curl -s -m 600 localhost:30006/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"What is 17*23? Reply with just the number."}],"max_tokens":4096}' | python3 -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];print('chat:',repr((m.get('content') or '')[-40:]))"
curl -s -m 600 localhost:30006/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"dsv41-flash-uva","messages":[{"role":"user","content":"Weather in Paris and Tokyo?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":4096}' | python3 -c "import json,sys;m=json.load(sys.stdin)['choices'][0]['message'];print('tools:',[(t['function']['name'],t['function']['arguments']) for t in m.get('tool_calls') or []])"
echo "=== $(date +%H:%M) gsm8k-200"
(cd "$M" && python3 gsm8k_eval.py "$TAG" 200 2>&1 | tail -2)
echo "=== $(date +%H:%M) needles"
for t in 125000 500000 1000000; do THINK_KEY=thinking /home/jasonc/venvs/bench/bin/python "$Q/needle_test.py" "$TAG-$t" "$t" 2>&1 | grep -E "needle [0-9]|FAIL|time=" | tail -4; done
echo "=== $(date +%H:%M) reasoning decode"
python3 /home/jasonc/research/glm53-flash/reason_bench.py "$TAG" 1,8,16 --window 90 --warmup 20 2>&1 | sed 's/^/speed: /'
echo "=== $(date +%H:%M) catid prefill"
/home/jasonc/venvs/bench/bin/python /home/jasonc/ds41f-exp/bench_prefill.py --engine vllm --host 127.0.0.1 --port 30006 --isl 16384,65536,131072 \
  --concurrency 1 --seed $RANDOM$RANDOM --wait-ready 60 --output "$L/prefill-$TAG.jsonl" --label "$TAG" 2>&1 | grep -E "^\s+[0-9]+ +1 " | sed 's/^/prefill: /'
echo "=== $(date +%H:%M) coexistence: H3 15 s render + DS41 C1 decode"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i GPU-c146511a-0326-7ddc-4346-998d61a64b34; sleep 1; done > "$L/gb300-mem-coexist.txt" & echo $! > "$L/sampler.pid" )
( cd "$H" && ./gen.sh coexist-ds41-15s-s30 30 1024 576 15 "$(cat ref2va/prompt-t2va-astronaut-control.txt)" > "$L/h3-render.log" 2>&1 & echo $! > "$L/render.pid" )
sleep 25
python3 /home/jasonc/research/glm53-flash/reason_bench.py "$TAG-during-h3" 1 --window 60 --warmup 10 2>&1 | sed 's/^/speed during H3: /'
while kill -0 "$(cat "$L/render.pid")" 2>/dev/null; do sleep 5; done
kill "$(cat "$L/sampler.pid")"
grep -E "^wall|HTTP/1.1 [0-9]|x-peak" "$L/h3-render.log" | sed 's/^/h3: /'
sort -n "$L/gb300-mem-coexist.txt" | tail -1 | awk '{printf "GB300 peak during coexistence: %.1f of 250.7 GiB\n", $1/1024}'
docker logs dsv41-flash-megamoe 2>&1 | grep -ciE "out of memory|Traceback" | sed 's/^/ds41 error lines: /'
docker logs minimax-h3 2>&1 | grep -ciE "out of memory|Traceback" | sed 's/^/h3 error lines: /'
echo "=== $(date +%H:%M) done"
