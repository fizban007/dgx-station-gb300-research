#!/usr/bin/env bash
# Continuation of pp-loop-test.sh on the same running server (TRT-LLM MoE, MTP3, Rust + mp): finish the penalty-0 reps
# directly, then run penalty 0.5 through pp_proxy.py, which adds presence_penalty to every request, because vLLM has
# no server-side presence_penalty default. Same server instance for both arms; only the per-request penalty differs.
set -uo pipefail
D=/home/jasonc/research/qwen38
REPS=${REPS:-8}; FIRST_PP0=${FIRST_PP0:-3}; PP=${PP:-0.5}; PROXY_PORT=30007
rep() {  # rep <label> <index> <port>
  local out=/home/jasonc/spark_vllm/qwen38-loop-$1-$2
  ssh -o BatchMode=yes stinger "cd /home/jasonc/spark_vllm/llm-inference-bench && python3 llm_decode_bench.py \
    --host http://192.168.1.99 --port $3 --model qwen38-flash-next --concurrency 1,8 --contexts 128k --duration 30 \
    --max-tokens 8192 --skip-prefill --display-mode plain --no-hw-monitor --no-resume --output $out.json \
    < /dev/null > $out.log 2>&1; python3 /tmp/looptally.py $out.json '$1 run$2'" < /dev/null
}
for i in $(seq "$FIRST_PP0" "$REPS"); do rep pp0 "$i" 30006; done
/home/jasonc/venvs/vllm-karmic/bin/python "$D/pp_proxy.py" $PROXY_PORT http://127.0.0.1:30006 "$PP" > "$D/logs/pp_proxy.log" 2>&1 &
proxy=$!
for _ in $(seq 30); do curl -sf -m 2 localhost:$PROXY_PORT/proxy_stats >/dev/null && break; sleep 1; done
echo "=== presence_penalty $PP via proxy  $(date +%H:%M:%S)  $(ssh -o BatchMode=yes stinger "curl -s -m 5 http://192.168.1.99:$PROXY_PORT/v1/models | head -c 60")"
(cd /home/jasonc/research/megamoe && PORT=$PROXY_PORT MODEL_NAME=qwen38-flash-next THINK_KEY=enable_thinking \
  /home/jasonc/venvs/bench/bin/python gsm8k_eval.py "qwen-pp$PP" 200)
for i in $(seq "$REPS"); do rep "pp$PP" "$i" $PROXY_PORT; done
echo "proxy stats: $(curl -s localhost:$PROXY_PORT/proxy_stats)"
kill $proxy
echo "=== done $(date +%H:%M:%S)"
