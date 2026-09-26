#!/usr/bin/env bash
# Decode (catid recipe) then prefill (catid recipe) against a running server.
# Usage: suite.sh <label> "<decode Cs>" "<prefill ISLs>" "<prefill Cs>"
label=${1:?label}; decode=${2:-1 8 16}; isl=${3:-16384,32768,65536,131072}; pc=${4:-1}
cd /home/jasonc/ds41f-exp
[ -n "$decode" ] && ./bench_decode.sh "$label" "$decode" > "runs/$label/decode-suite.log" 2>&1
[ -n "$isl" ] && ./bench_prefill.sh "$label" "$isl" "$pc" > "runs/$label/prefill-suite.log" 2>&1
python3 /home/jasonc/dgx_station_benchmarks/deepseek-v4.1-flash/recipes/summarize_decode.py "runs/$label/decode" 2>/dev/null | tail -n +1
grep -E "^ +[0-9]+ +[0-9]+ " "runs/$label/prefill/prefill.log" 2>/dev/null
