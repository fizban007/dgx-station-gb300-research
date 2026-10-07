#!/usr/bin/env bash
# After ds41-h3-test.sh: GPQA-Diamond on the DS41 test boot (265-hot rowmap, NVFP4 Engram, FP8 KV) with the GLM T=1
# run's flags (T=1.0, top_p 0.95, reasoning_effort max, C32, seed 1101), paired against the old DS41 T=0 run.
while pgrep -f "[d]s41-h3-test.sh" >/dev/null; do sleep 15; done
echo "=== $(date +%H:%M) GPQA-Diamond DS41 T=1"
cd ~/evals-private/gpqa && /home/jasonc/venvs/bench/bin/python /home/jasonc/llm-inference-bench/llm_decode_bench.py --host 127.0.0.1 --port 30006 \
  --model dsv41-flash-uva --test-profile gpqa-diamond --reasoning-effort max --profile-concurrency 32 --completion-stats-temperature 1.0 \
  --completion-stats-top-p 0.95 --completion-stats-seed 1101 --completion-stats-save-text --display-mode plain --no-hw-monitor \
  --output ds41-h265-nvfp4engram-t1-c32-max.json --compare-baseline dsv41-flash-max.json > ds41-h265-nvfp4engram-t1-c32-max.log 2>&1
python3 -c "import json;d=json.load(open('ds41-h265-nvfp4engram-t1-c32-max.json'));a=d['accuracy'];print('gpqa ds41:',a['correct'],'/',a['scored'],round(a['accuracy']*100,1),'% hit_max',a['hit_max_tokens'])"
echo "=== $(date +%H:%M) GPQA done"
