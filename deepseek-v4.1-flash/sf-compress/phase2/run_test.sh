#!/usr/bin/env bash
# Run test_megamoe_csf.py in a temporary vLLM container on the GB300 with the patched MegaMoE header mounted.
#   ./run_test.sh --experts 96 --max-tokens 2048 --tokens 1 8 128
# The patched header is built from the image's own copy plus sm100_fp8_fp4_mega_moe.csf.diff (and ABLATION=<name>
# from ablations/, applied on top), unless HEADER=<file> names a ready one. GPU, MODEL, IMAGE and JIT override
# the defaults; JIT points at the m3 jit-cache so DeepGEMM/Triton kernels are not recompiled every run.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
GPU=${GPU:-GPU-cc2d0965-4d58-70d1-25ad-e35af3c802c9}
MODEL=${MODEL:-$HOME/models/DeepSeek-V4.1-Flash}
IMAGE=${IMAGE:-vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b}
JIT=${JIT:-$REPO/../m3/jit-cache}
D=/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls
if [ -z "${HEADER:-}" ]; then
  B=$HERE/build
  mkdir -p "$B"
  docker run --rm --entrypoint cat "$IMAGE" $D/sm100_fp8_fp4_mega_moe.cuh > "$B/sm100_fp8_fp4_mega_moe.cuh"
  patch -s "$B/sm100_fp8_fp4_mega_moe.cuh" < "$HERE/sm100_fp8_fp4_mega_moe.csf.diff"
  if [ -n "${ABLATION:-}" ]; then patch -s "$B/sm100_fp8_fp4_mega_moe.cuh" < "$HERE/ablations/$ABLATION.diff"; fi
  HEADER=$B/sm100_fp8_fp4_mega_moe.cuh
fi
mkdir -p "$JIT/dj" "$JIT/triton"
docker run --rm --name csf-p2-test --gpus "\"device=$GPU\"" --ipc host --network host --ulimit memlock=-1 \
  -v "$MODEL":/model:ro -v "$REPO/hook":/w:ro -v "$HERE":/p \
  -v "$HEADER":$D/sm100_fp8_fp4_mega_moe.cuh:ro \
  -v "$JIT/dj":/root/.dj -v "$JIT/triton":/root/.triton \
  -e DG_JIT_PTXAS_VERBOSE=${PTXAS_VERBOSE:-0} --entrypoint python3 "$IMAGE" /p/test_megamoe_csf.py "$@" 2>&1 | grep -v "^\[CUDA\]"
