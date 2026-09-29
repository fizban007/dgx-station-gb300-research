# Microbenchmarks (2026-09-28)

All on the GB300, CUDA-graph replay, in throwaway containers of the serving image (nightly-af7f9488) while the lane
was idle. `mxfp8_backends.jsonl` is the script's saved output; the other files are its console output, copied verbatim.

| file | script | what |
|---|---|---|
| `mxfp8_backends.jsonl` | `../../tools/gemm_bench.py` | FlashInfer `mm_mxfp8` backends (cute-dsl, cutedsl_low_latency, cutlass, cuDNN) at our dense shapes, L2-cold weights, µs and TB/s; last row = HBM read reference |
| `mxfp8_trtllm_vs_cutedsl.jsonl` | `../../tools/gemm_bench_trt.py` | same shapes, TRT-LLM backend (shuffled weights, 8x4 activation scales) vs cute-dsl |
| `mxfp8_triton_splitk.txt` | `../../tools/splitk_bench.py` | our Triton split-K MXFP8 GEMM (`../../hook/mxfp8_splitk.py`, dropped) vs cute-dsl / low-latency |
| `mega_attn_variants.jsonl` | `../../tools/mega_attn_bench.py` | FlashMLA mega attention: 64 vs 128 (padded) heads, fp8 vs NVFP4 compressed cache, s_q 1..96 |
| `mega_attn_flashmla227.txt` | `../../tools/mega_attn_check.py` | deepseek-ai/FlashMLA#227 built for sm_103a vs the image's kernel |
| `peer_fused2.txt` | `../../tools/test_peer_fused2.py`, `../../tools/pf2_pieces.py` | fused send v2 bit-exactness and per-layer timing |
| `drafter_argmax.jsonl` | `../../tools/argmax_bench.py` | torch.argmax over the 129,280 vocab vs a two-stage PyTorch argmax |
