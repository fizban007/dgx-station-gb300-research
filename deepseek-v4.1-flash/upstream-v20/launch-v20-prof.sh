#!/usr/bin/env bash
# Al-ENGR v20 "Clean Nightly" on gracie: their image, v15 pin-hot-experts hook + static rowmap, their flags.
# Differences from theirs: GB300 selected by UUID; host memory bound to Grace node 0 with numactl (cpuset-mems
# breaks CUDA init here, error 802); our cache dir;
# no pinned autotune set (theirs, a34f9ad4, is unpublished), so the first boot live-tunes.
set -euo pipefail
NAME=${NAME:-v20-upstream}
IMAGE=vllm/vllm-openai:nightly-2671fedfc7ae604761990603fc736c0c4f21de57
MODEL=/home/jasonc/models/DeepSeek-V4.1-Flash
CACHE=/home/jasonc/research/upstream/vllm-cache
HOOK=/home/jasonc/research/al-engr/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-17-e2b-pin-hot-experts-v15/hook
PIN_MODE=${PIN_MODE:-split}
SEQS=${SEQS:-24}
OFFGB=${OFFGB:-54}
KSCHED=${KSCHED:-"[[1,4,5],[5,$SEQS,1]]"}
EXTRA=${EXTRA:-}
docker run -d --name "$NAME" --gpus '"device=GPU-c146511a-0326-7ddc-4346-998d61a64b34"' \
  --cap-add SYS_NICE --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 --cap-add IPC_LOCK --security-opt label=disable \
  -v "$MODEL":/model:ro -v "$CACHE":/root/.cache/vllm -v /home/jasonc/research/upstream/prof:/prof \
  -v "$HOOK/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro -v "$HOOK":/w:ro \
  -v /usr/bin/numactl:/usr/local/bin/numactl:ro --entrypoint /usr/local/bin/numactl \
  -e VLLM_LOGGING_LEVEL=INFO -e PIN_MODE="$PIN_MODE" -e PIN_HOOK=/w/pin_hot_experts_hook.py \
  -e PIN_ROWMAP=/w/rowmap-static-v1.json -e PIN_DUMP_DIR=/root/.cache/vllm/pin-dump \
  "$IMAGE" --membind=0 vllm serve \
  --model /model --served-model-name dsv41-flash-uva --trust-remote-code --tensor-parallel-size 1 \
  --offload-backend uva --cpu-offload-gb "$OFFGB" \
  --cpu-offload-params routed_experts.w13_weight routed_experts.w2_weight \
  --engram-config '{"cpu_offload": true}' \
  --max-model-len 1048576 --max-num-seqs "$SEQS" --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.97 --kv-cache-dtype fp8_ds_mla \
  --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 --enable-auto-tool-choice \
  --long-prefill-token-threshold 6144 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"num_speculative_tokens_per_batch_size\":$KSCHED}" \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 18 24 32 40 48 64 96 128 \
  $EXTRA --host 0.0.0.0 --port 30006
echo "launched $NAME"
