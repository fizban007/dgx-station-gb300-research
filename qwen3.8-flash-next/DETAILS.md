# Qwen3.8-Flash-Next NVFP4 on one GB300: details

Companion to [README.md](README.md). Every number in the tables below is also a row in
[results.jsonl](results.jsonl), which [tools/make_results_jsonl.py](tools/make_results_jsonl.py) builds from the raw
files under [results/](results/). Throughput is output tok/s. "Aggregate" is all streams together; "per-user" is the
median single stream.

## Setup

- **Hardware:** one DGX Station GB300 (host "gracie"), GPU `GPU-c146511a-0326-7ddc-4346-998d61a64b34`, TP1. Every
  server ran under `numactl --membind=0` (Grace memory). The RTX PRO 6000 in the same box was not used.
- **Host memory mode:** runs before 09:09 EDT on 2026-09-25 used the driver's default NUMA mode. The host was then
  rebooted into driver-managed coherent memory (CDMM). The A/B, TTFT and loop runs came after the reboot, as did every
  run from the evening of 2026-09-25 (the prefill diagnostics and the `CG=8192` configs). (From session notes and the
  host's boot record; raw file not kept.)
- **Checkpoint (all measured runs):** `nvidia/Qwen3.8-Flash-Next-NVFP4` (ModelOpt NVFP4 routed experts; the MTP
  layer's experts are FP8 block-128), mounted as `/models/Qwen3.8-Flash-Next-NVFP4-nvidia`. The local Hugging Face
  download metadata records commit `fc694b54fb0174e0913e6adf86691ef85a4ead47` (file not kept). Every server log
  names this path.
- **Checkpoint (b12x fork attempts only):** `/home/jasonc/models/Qwen3.8-Flash-Next-NVFP4`, a local-inference-lab
  NVFP4 QAD export (branch `qad-step-4000`) with NVFP4 PLE embedding tables. It never loaded on the GB300 (see below).
- **Engines:**
  - vLLM `vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423`. It reports itself as
    `v0.30.1rc1.dev48+g7f1a5398e` ([vllm-rust-mp.log](results/logs/vllm-rust-mp.log)).
  - SGLang `lmsysorg/sglang:nightly-dev-cu13-20260924-ffac53d7`.
  - The fork attempts used a Karmic vLLM fork (`0.1.dev21505+ge77be2251.d20260924`) from a host venv.
- **Harness:**
  - [bench-qwen.sh](scripts/bench-qwen.sh) runs three steps: GSM8K ([gsm8k_eval.py](../bench/gsm8k_eval.py),
    last N test questions, greedy, thinking off), the catid decode recipe
    ([bench_decode.sh](../bench/bench_decode.sh), then
    [llm_decode_bench.py](https://github.com/local-inference-lab/llm-inference-bench) v0.4.29), and cold C1 prefill
    (catid's [bench_prefill.py](https://github.com/catid/dgx_station_benchmarks/blob/ff8a496e5e027bbc462f81643361ef5516072a68/deepseek-v4.1-flash/recipes/bench_prefill.py), linked rather than copied because that repo has no license; see [../bench/README.md](../bench/README.md)).
  - The catid decode recipe is 8,192 exact input tokens, 1,024 forced output tokens and temperature 0. Each cell
    runs C warm-ups, then 5×C measured requests.
  - Cold prefill uses random token ids and flushes the cache. Each point is 1 warm-up plus 4 measured requests, with
    one output token.
  - `bench_decode.sh` also calls catid's `summarize_decode.py` for its console table. That file is not in this repo
    (no license); the tables here are computed from the raw JSON instead.

## Configurations

"Gate" is the GSM8K result from the same server instance. There is no formal pass threshold. Every gated config
landed at 95.5–98.0%. Configs without their own gate are marked unqualified.

| Config | Date (EDT) | Engine | Frontend / executor | MoE kernel | MTP | Other flags | Host mode | Gate |
|---|---|---|---|---|---|---|---|---|
| `vllm-trtllm-mtp3` | 09-24 20:36 | vLLM | Python / uni | auto → FLASHINFER_TRTLLM | 3 | – | NUMA | 194/200 |
| `vllm-recipe-mtp3` | 09-25 08:04 | vLLM | Rust / mp | flashinfer_trtllm | 3 | `--no-enable-flashinfer-autotune -cc.cudagraph_mode full_decode_only` | NUMA | 191/200 |
| `vllm-rust-mp-mtp3` | 09-25 08:50 | vLLM | Rust / mp | flashinfer_trtllm | 3 | – (= launcher defaults) | NUMA | 193/200 |
| `ab-base-1`, `-2` | 09-25 09:11, 09:29 | vLLM | Python / uni | auto → FLASHINFER_TRTLLM | 3 | – | CDMM | 48/50 (run 1 only) |
| `ab-rustmp-1`, `-2` | 09-25 09:20, 09:38 | vLLM | Rust / mp | auto → FLASHINFER_TRTLLM | 3 | – (launcher defaults) | CDMM | 49/50 (run 1 only) |
| `pp0`, `pp0.5` | 09-25 10:10 | vLLM | Rust / mp | auto → FLASHINFER_TRTLLM | 3 | `pp0.5`: presence_penalty 0.5 per request via proxy | CDMM | 192/200 each |
| `ttft-rust-uni` | 09-25 11:10 | vLLM | Rust / uni | auto | 3 | env `VLLM_USE_RUST_FRONTEND=1` only | CDMM | unqualified |
| `ttft-mp-only` | 09-25 10:43 | vLLM | Python / mp | auto | 3 | `--distributed-executor-backend mp` | CDMM | unqualified |
| `ttft-api4` | 09-25 10:52 | vLLM | Python ×4 / uni | auto | 3 | `--api-server-count 4` | CDMM | unqualified |
| `diag-rust-mp-cg1024` | 09-25 21:47 | vLLM | Rust / mp | auto → FLASHINFER_TRTLLM | 3 | prefill diagnostics; launcher of that afternoon plus explicit `--enable-prefix-caching` | CDMM | unqualified |
| `diag-py-uni-cg1024` | 09-25 22:09 | vLLM | Python / uni | auto → FLASHINFER_TRTLLM | 3 | prefill diagnostics and torch profile (`--profiler-config`) | CDMM | unqualified |
| `diag-py-uni-cg8192` | 09-25 22:17 | vLLM | Python / uni | auto → FLASHINFER_TRTLLM | 3 | CUDA graphs to 8,192 via `EXTRA="--compilation-config ..."`, `--profiler-config`, `VLLM_SERVER_DEV_MODE=1` | CDMM | unqualified |
| `diag-py-uni-cg1024-dev` | 09-25 22:27 | vLLM | Python / uni | auto → FLASHINFER_TRTLLM | 3 | CUDA graphs to 1,024; correctness baseline; tool parser, `VLLM_SERVER_DEV_MODE=1` | CDMM | unqualified |
| `cg8192-py-1` | 09-25 22:32 | vLLM | Python / uni | auto → FLASHINFER_TRTLLM | 3 | `CG=8192`, tool parser, `VLLM_SERVER_DEV_MODE=1`, vision off | CDMM | 48/50 |
| `vllm-py-cg8192-mtp3` | 09-25 22:46 | vLLM | Python / uni | auto → FLASHINFER_TRTLLM | 3 | launcher defaults (`RUST_MP=0 CG=8192 VISION=video`, tool parser) | CDMM | 195/200 |
| `smoke-<kernel>` | 09-25 08:15–08:50 | vLLM | Python / uni | `--moe-backend <kernel>` | 0 | – | NUMA | 48–49/50 |
| `sglang-trtllm-mtp3` | 09-24 21:19 | SGLang | – | flashinfer_trtllm | NEXTN 3 | metrics off | NUMA | 194/200 |
| `sglang-rs-mtp3` | 09-24 22:12 | SGLang | – | auto → flashinfer_trtllm | NEXTN 3 | `--enable-linear-replayssm-spec` | NUMA | 193/200 |
| `sglang-rs-fipre-mtp3` | 09-24 22:23 | SGLang | – | auto → flashinfer_trtllm | NEXTN 3 | ReplaySSM + `--linear-attn-prefill-backend flashinfer` | NUMA | 194/200 |
| `sglang-mega-rsfp-mtp3` | 09-25 07:09 | SGLang | – | flashinfer_megamoe | NEXTN 3 | ReplaySSM + FI GDN prefill + `--moe-a2a-backend flashinfer_megamoe --speculative-moe-runner-backend flashinfer_trtllm --speculative-moe-a2a-backend none`, [patched moe_hook.py](patches/moe_hook.diff) | NUMA | 193/200 |

Common vLLM flags come from [launch-qwen-upstream-2026-09-25.sh](scripts/launch-qwen-upstream-2026-09-25.sh), the
launcher before the evening of 2026-09-25:

- `--max-model-len 262144 --max-num-seqs 128 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90`
- `--limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3`
- MTP adds `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`.
- "Rust / mp" means the env `VLLM_USE_RUST_FRONTEND=1` plus `--distributed-executor-backend mp`.
- The evening runs (`diag-*`, `cg8192-py-1`, `vllm-py-cg8192-mtp3`) add `--enable-prefix-caching`. The last two use
  the current [launch-qwen-upstream.sh](scripts/launch-qwen-upstream.sh), which adds `CG`, `VISION` and the
  `qwen3_xml` tool parser and defaults to `RUST_MP=0`. Prefix caching was already on in every earlier vLLM run (vLLM's
  default; the server logs show `enable_prefix_caching=True`).
- The 09-24 `vllm-trtllm-mtp3` run predates the `RUST_MP` switch. Its server's non-default arguments match
  `RUST_MP=0 MTP=3` (server log not kept).

Common SGLang flags come from [launch-qwen-sglang.sh](scripts/launch-qwen-sglang.sh):

- `--context-length 262144 --max-running-requests 128 --reasoning-parser qwen3`
- NEXTN runs 3 steps with top-k 1 and 4 draft tokens.
- The variants add `--max-mamba-cache-size 330 --mem-fraction-static 0.85 --enable-metrics`
  ([sglang-variants.sh](scripts/sglang-variants.sh)).
- With 330 Mamba slots, SGLang admitted at most 66 running requests
  ([sglang-variants.log](results/logs/sglang-variants.log)).
- The default `sglang-trtllm-mtp3` run had metrics off, so its MTP acceptance is missing.

## Decode, catid recipe (all columns)

Every cell completed all 5×C requests with 0 errors. The bench flagged brief queueing (queue fraction 1.1–5.7%,
mostly the admission wave) in these cells:

- `vllm-trtllm-mtp3` C32–C64
- `vllm-recipe-mtp3` C8–C64
- `vllm-rust-mp-mtp3` C8, C32, C64
- `ab-base-*` C32
- `ab-rustmp-*` C8–C32
- `vllm-py-cg8192-mtp3` C32–C64 and `cg8192-py-1` C32

One outside chat request from the bench client (192.168.1.25) reached `vllm-py-cg8192-mtp3` at 23:02:28 EDT, during
its C4 or C8 cell. Those cells match `cg8192-py-1` within 1.3%. The server's access log is
[server-vllm-py-cg8192-mtp3.log](results/logs/server-vllm-py-cg8192-mtp3.log). The same client's benchmark also
overlapped run 8 of the 128K loop check, which lowered that run's C1 decode rate; the loop status is unaffected.
- the SGLang ReplaySSM variants at C64

Raw files: `results/runs/qwen-<config>/decode/c<C>.json` (and `.log`). "MTP accept len" is the server-side mean
accepted length per MTP step.

#### `vllm-py-cg8192-mtp3` (chosen config, CDMM)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 344.0 | 356.4 | 72 | 72 | 2.81 | 2.57 |
| 2 | 605.5 | 313.7 | 95 | 133 | 3.19 | 2.57 |
| 4 | 1,001.4 | 260.5 | 120 | 208 | 3.84 | 2.57 |
| 8 | 1,670.9 | 220.4 | 104 | 289 | 4.54 | 2.58 |
| 16 | 2,592.3 | 170.8 | 165 | 472 | 5.86 | 2.57 |
| 32 | 3,785.5 | 124.5 | 197 | 823 | 8.03 | 2.57 |
| 64 | 5,247.0 | 86.2 | 285 | 1,531 | 11.60 | 2.56 |

#### `cg8192-py-1` (same graph setting, second server, CDMM)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 344.0 | 355.5 | 80 | 80 | 2.81 | 2.54 |
| 2 | 588.1 | 304.5 | 106 | 142 | 3.28 | 2.51 |
| 4 | 989.1 | 256.9 | 105 | 239 | 3.89 | 2.55 |
| 8 | 1,671.9 | 222.1 | 142 | 379 | 4.50 | 2.59 |
| 16 | 2,591.7 | 169.6 | 175 | 559 | 5.90 | 2.56 |
| 32 | 3,784.0 | 123.8 | 182 | 910 | 8.07 | 2.57 |

#### `vllm-rust-mp-mtp3` (chosen before the graph fix, NUMA mode)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 313.2 | 334.6 | 174 | 178 | 2.99 | 2.50 |
| 2 | 550.3 | 310.0 | 194 | 226 | 3.23 | 2.55 |
| 4 | 925.1 | 250.0 | 166 | 301 | 4.00 | 2.56 |
| 8 | 1,492.0 | 195.8 | 173 | 313 | 5.11 | 2.55 |
| 16 | 2,197.7 | 142.2 | 179 | 625 | 7.03 | 2.56 |
| 32 | 3,181.2 | 102.6 | 311 | 946 | 9.74 | 2.55 |
| 64 | 4,332.8 | 69.0 | 346 | 1,488 | 14.49 | 2.55 |

#### `vllm-trtllm-mtp3` (Python frontend, uni executor)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 337.9 | 351.4 | 159 | 162 | 2.85 | 2.58 |
| 2 | 579.1 | 309.4 | 187 | 306 | 3.23 | 2.60 |
| 4 | 931.7 | 250.8 | 194 | 340 | 3.99 | 2.56 |
| 8 | 1,486.6 | 197.3 | 198 | 506 | 5.07 | 2.55 |
| 16 | 2,219.5 | 148.4 | 328 | 661 | 6.74 | 2.58 |
| 32 | 3,142.5 | 102.6 | 359 | 1,021 | 9.75 | 2.54 |
| 64 | 4,630.7 | 76.0 | 477 | 1,697 | 13.15 | 2.56 |

#### `vllm-recipe-mtp3` (Rust / mp, autotune off, `full_decode_only` CUDA graphs)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 339.6 | 361.0 | 143 | 144 | 2.77 | 2.62 |
| 2 | 565.6 | 301.9 | 165 | 303 | 3.31 | 2.54 |
| 4 | 946.7 | 249.4 | 172 | 307 | 4.01 | 2.61 |
| 8 | 1,415.4 | 185.0 | 222 | 407 | 5.41 | 2.57 |
| 16 | 2,139.6 | 139.2 | 242 | 602 | 7.18 | 2.55 |
| 32 | 3,215.5 | 105.7 | 320 | 988 | 9.47 | 2.57 |
| 64 | 4,290.5 | 69.3 | 425 | 1,585 | 14.42 | 2.55 |

This run differs from `vllm-rust-mp-mtp3` in two flags at once: autotune off and `full_decode_only`. Its C8 is 5.1%
below `vllm-rust-mp-mtp3`, but the two flags' effects are not separated.

#### `sglang-trtllm-mtp3` (SGLang default, metrics off)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 324.0 | 353.3 | 189 | 411 | 2.83 | – |
| 2 | 566.8 | 303.4 | 198 | 347 | 3.30 | – |
| 4 | 858.9 | 236.8 | 340 | 690 | 4.22 | – |
| 8 | 1,294.1 | 170.8 | 359 | 373 | 5.85 | – |
| 16 | 1,781.1 | 117.6 | 356 | 1,043 | 8.51 | – |
| 32 | 2,367.5 | 75.4 | 360 | 772 | 13.27 | – |
| 64 | 3,086.0 | 49.1 | 406 | 1,065 | 20.35 | – |

#### `sglang-rs-mtp3` (ReplaySSM)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 337.0 | 369.7 | 185 | 215 | 2.71 | 2.45 |
| 2 | 585.6 | 314.6 | 197 | 341 | 3.18 | 2.74 |
| 4 | 928.5 | 251.0 | 195 | 350 | 3.98 | 2.56 |
| 8 | 1,391.8 | 184.1 | 205 | 365 | 5.43 | 2.44 |
| 16 | 1,943.6 | 125.0 | 345 | 561 | 8.00 | 2.45 |
| 32 | 2,687.2 | 86.5 | 353 | 767 | 11.56 | 2.43 |
| 64 | 3,489.7 | 55.2 | 384 | 1,023 | 18.11 | 2.53 |

#### `sglang-rs-fipre-mtp3` (ReplaySSM + FlashInfer GDN prefill)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 336.9 | 363.1 | 173 | 204 | 2.75 | 2.33 |
| 2 | 585.4 | 320.3 | 185 | 319 | 3.12 | 2.27 |
| 4 | 928.3 | 247.5 | 185 | 331 | 4.04 | 2.77 |
| 8 | 1,403.8 | 191.5 | 328 | 344 | 5.22 | 2.67 |
| 16 | 1,978.6 | 127.7 | 341 | 524 | 7.83 | 2.66 |
| 32 | 2,803.3 | 90.6 | 336 | 723 | 11.04 | 2.61 |
| 64 | 3,555.4 | 56.0 | 366 | 966 | 17.87 | 2.65 |

#### `sglang-mega-rsfp-mtp3` (FlashInfer MegaMoE at TP1, locally patched)

| C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 106.9 | 109.7 | 199 | 236 | 9.11 | 2.45 |
| 2 | 201.1 | 105.7 | 221 | 350 | 9.47 | 2.25 |
| 4 | 367.1 | 95.4 | 222 | 360 | 10.48 | 2.41 |
| 8 | 667.0 | 87.2 | 363 | 385 | 11.47 | 2.64 |
| 16 | 1,078.7 | 71.0 | 253 | 567 | 14.08 | 2.56 |
| 32 | 1,756.5 | 56.2 | 377 | 777 | 17.80 | 2.58 |
| 64 | 2,558.3 | 40.9 | 423 | 1,055 | 24.43 | 2.64 |

## Cold prefill, C1 (all configs)

Raw file: `results/logs/prefill-<config>.jsonl`. Every point: 4 measured requests after 1 warm-up, 0 errors, and
prompt token counts matching the target. SGLang used its native `/generate` endpoint; vLLM used
`/v1/completions`.

| Config | 8K tok/s | 32K tok/s | 64K tok/s | 128K tok/s | TTFT p50 8K / 32K / 64K / 128K (s) |
|---|---:|---:|---:|---:|---|
| `vllm-py-cg8192-mtp3` | 43,612.1 | 45,124.1 | 45,164.4 | 44,398.9 | 0.19 / 0.73 / 1.45 / 2.95 |
| `diag-py-uni-cg8192` | 44,061.8 | 45,453.9 | 45,544.1 | 44,512.0 | 0.19 / 0.72 / 1.44 / 2.94 |
| `vllm-rust-mp-mtp3` | 31,402.5 | 45,844.5 | 46,446.3 | 45,529.8 | 0.26 / 0.71 / 1.41 / 2.87 |
| `vllm-trtllm-mtp3` | 31,528.4 | 44,846.6 | 45,393.6 | 44,574.5 | 0.26 / 0.73 / 1.44 / 2.94 |
| `vllm-recipe-mtp3` | 30,382.3 | 44,186.3 | 44,730.0 | 43,773.5 | 0.27 / 0.74 / 1.47 / 2.99 |
| `sglang-trtllm-mtp3` | 36,742.1 | 40,012.5 | 38,663.8 | 34,134.0 | 0.22 / 0.81 / 1.70 / 3.84 |
| `sglang-rs-mtp3` | 36,848.7 | 40,366.9 | 38,726.6 | 34,157.4 | 0.22 / 0.81 / 1.69 / 3.84 |
| `sglang-rs-fipre-mtp3` | 41,571.8 | 45,703.6 | 43,623.4 | 37,912.8 | 0.20 / 0.72 / 1.50 / 3.46 |
| `sglang-mega-rsfp-mtp3` | 34,418.4 | 39,892.8 | 38,631.0 | 34,246.7 | 0.24 / 0.82 / 1.70 / 3.83 |

## MoE kernel smoke test (vLLM, MTP off)

[smoke-kernels.sh](scripts/smoke-kernels.sh) started one server per NVFP4 MoE kernel with `RUST_MP=0 MTP=0
--moe-backend <kernel>`. MTP is off because `--moe-backend` also applies to the FP8 MTP layer, which only
TRT-LLM supports among these kernels (see failed runs).

The decode here is not the catid recipe:

- a minimal prompt (context 0) and 1,024 forced output tokens at T=0
- C warm-ups, then 2×C measured requests

It ran in two passes:

- [smoke-kernels-run1.log](results/logs/smoke-kernels-run1.log): `flashinfer_cutlass` and `cutlass` failed to start
  because the previous container had not yet released GPU memory.
- [smoke-kernels-run2.log](results/logs/smoke-kernels-run2.log) reran those two.

Raw files: `results/runs/smoke-<kernel>/c<C>.json`, `results/logs/prefill-smoke-<kernel>.jsonl`, and
`results/gsm8k/gsm8k-qwen-smoke-<kernel>.json`.

| Kernel | C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms |
|---|---:|---:|---:|---:|---:|---:|
| `flashinfer_trtllm` | 1 | 203.1 | 207.5 | 47 | 48 | 4.82 |
| `flashinfer_trtllm` | 16 | 2,238.6 | 143.6 | 186 | 197 | 6.97 |
| `flashinfer_trtllm` | 64 | 6,022.4 | 98.4 | 415 | 431 | 10.16 |
| `flashinfer_trtllm` | 128 | 8,966.6 | 72.8 | 505 | 647 | 13.74 |
| `flashinfer_cutedsl` | 1 | 180.4 | 185.6 | 48 | 50 | 5.39 |
| `flashinfer_cutedsl` | 16 | 1,778.1 | 114.0 | 214 | 228 | 8.77 |
| `flashinfer_cutedsl` | 64 | 4,620.3 | 75.3 | 513 | 534 | 13.29 |
| `flashinfer_cutedsl` | 128 | 7,042.0 | 56.9 | 519 | 617 | 17.59 |
| `flashinfer_cutlass` | 1 | 147.6 | 151.0 | 44 | 44 | 6.62 |
| `flashinfer_cutlass` | 16 | 1,563.5 | 99.9 | 188 | 198 | 10.01 |
| `flashinfer_cutlass` | 64 | 5,230.8 | 84.2 | 412 | 419 | 11.87 |
| `flashinfer_cutlass` | 128 | 8,039.5 | 65.6 | 631 | 656 | 15.24 |
| `cutlass` (vLLM CUTLASS) | 1 | 104.1 | 105.3 | 45 | 45 | 9.49 |
| `cutlass` (vLLM CUTLASS) | 16 | 1,352.3 | 87.0 | 209 | 347 | 11.49 |
| `cutlass` (vLLM CUTLASS) | 64 | 4,579.5 | 73.6 | 444 | 461 | 13.58 |
| `cutlass` (vLLM CUTLASS) | 128 | 7,133.5 | 57.8 | 595 | 673 | 17.31 |

| Kernel | 8K prefill tok/s | 64K prefill tok/s | 8K / 64K TTFT p50 (s) | GSM8K-50 |
|---|---:|---:|---|---:|
| `flashinfer_trtllm` | 32,056.2 | 43,876.0 | 0.26 / 1.49 | 48/50 |
| `flashinfer_cutedsl` | 26,448.2 | 41,357.0 | 0.31 / 1.58 | 49/50 |
| `flashinfer_cutlass` | 30,573.3 | 37,536.9 | 0.27 / 1.75 | 48/50 |
| `cutlass` | 26,039.0 | 32,036.7 | 0.31 / 2.05 | 48/50 |

## Frontend and executor: TTFT

### A/B under CDMM ([ab-rust-mp.sh](scripts/ab-rust-mp.sh), [ab-rust-mp.log](results/logs/ab-rust-mp.log))

The arms alternated base, rustmp, base, rustmp, each on a fresh server. Each arm ran the catid recipe at C1–C32.
GSM8K-50 ran on pass 1 only, so the pass-2 servers are unqualified. The per-run tables are in
`results/runs/qwen-ab-*` and in [results.jsonl](results.jsonl); aggregate and TTFT are summarized in the README.

| Config | C | aggregate tok/s | per-user tok/s p50 | TTFT p50 ms | TTFT p90 ms | ITL p50 ms | MTP accept len |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ab-base-1` | 1 | 332.4 | 352.0 | 157 | 158 | 2.84 | 2.54 |
| `ab-base-1` | 2 | 579.6 | 307.3 | 181 | 299 | 3.25 | 2.61 |
| `ab-base-1` | 4 | 943.4 | 251.5 | 315 | 347 | 3.98 | 2.54 |
| `ab-base-1` | 8 | 1,529.1 | 200.2 | 323 | 480 | 4.99 | 2.59 |
| `ab-base-1` | 16 | 2,203.3 | 146.1 | 394 | 652 | 6.84 | 2.55 |
| `ab-base-1` | 32 | 3,170.8 | 104.2 | 359 | 999 | 9.60 | 2.55 |
| `ab-base-2` (unqualified) | 1 | 332.9 | 357.0 | 150 | 153 | 2.80 | 2.56 |
| `ab-base-2` (unqualified) | 2 | 571.7 | 301.8 | 181 | 314 | 3.31 | 2.57 |
| `ab-base-2` (unqualified) | 4 | 943.8 | 250.7 | 177 | 328 | 3.99 | 2.56 |
| `ab-base-2` (unqualified) | 8 | 1,488.2 | 197.7 | 190 | 471 | 5.06 | 2.57 |
| `ab-base-2` (unqualified) | 16 | 2,235.2 | 146.5 | 334 | 636 | 6.83 | 2.58 |
| `ab-base-2` (unqualified) | 32 | 3,253.5 | 105.7 | 350 | 984 | 9.46 | 2.56 |
| `ab-rustmp-1` | 1 | 333.9 | 349.5 | 140 | 141 | 2.86 | 2.54 |
| `ab-rustmp-1` | 2 | 565.7 | 302.2 | 165 | 180 | 3.31 | 2.56 |
| `ab-rustmp-1` | 4 | 936.0 | 250.4 | 167 | 307 | 3.99 | 2.56 |
| `ab-rustmp-1` | 8 | 1,479.3 | 195.2 | 171 | 313 | 5.12 | 2.55 |
| `ab-rustmp-1` | 16 | 2,227.2 | 146.8 | 180 | 478 | 6.81 | 2.56 |
| `ab-rustmp-1` | 32 | 3,122.3 | 102.2 | 311 | 950 | 9.79 | 2.56 |
| `ab-rustmp-2` (unqualified) | 1 | 339.7 | 356.2 | 132 | 133 | 2.81 | 2.59 |
| `ab-rustmp-2` (unqualified) | 2 | 565.8 | 298.3 | 163 | 182 | 3.35 | 2.52 |
| `ab-rustmp-2` (unqualified) | 4 | 925.8 | 245.3 | 161 | 292 | 4.08 | 2.54 |
| `ab-rustmp-2` (unqualified) | 8 | 1,521.6 | 198.7 | 168 | 306 | 5.03 | 2.58 |
| `ab-rustmp-2` (unqualified) | 16 | 2,243.7 | 147.2 | 177 | 475 | 6.79 | 2.55 |
| `ab-rustmp-2` (unqualified) | 32 | 3,109.4 | 98.8 | 311 | 805 | 10.12 | 2.56 |

In NUMA mode, before the reboot, the single-run comparison `vllm-trtllm-mtp3` → `vllm-rust-mp-mtp3` read:

- decode C1 −7.3%, C2 −5.0% and C64 −6.4%
- TTFT p50 at C16 328 → 179 ms
- prefill +2.1% at 128K

The 2-run CDMM A/B above did not reproduce the C1–C2 loss.

### Split test ([ttft-quick.sh](scripts/ttft-quick.sh); unqualified)

These runs split the Rust frontend from the `mp` executor. The request shape matches the catid recipe, but each
cell ran only 2×C requests, in two passes on one server. The script's own header warns that with 2×C the fixed
1,024-token outputs keep streams in synchronized waves, which inflates C8–C16 TTFT. So only C1 compares with the
A/B above.

The 5×C follow-up ([ttft-5x.sh](scripts/ttft-5x.sh)) did not complete: [ttft-5x.log](results/logs/ttft-5x.log) stops
after the first launch, and no results exist.

Raw files: `results/runs/ttft-<variant>/p<N>/c<C>.json` and `results/logs/ttft-<variant>.log`.

| Config | Pass | C | TTFT p50 ms | TTFT p90 ms | aggregate tok/s | per-user tok/s p50 | ITL p50 ms | MTP accept len |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ttft-rust-uni` | p1 | 1 | 148 | 149 | 340.0 | 373.2 | 2.68 | 2.69 |
| `ttft-rust-uni` | p1 | 8 | 327 | 462 | 1,552.3 | 215.6 | 4.64 | 2.55 |
| `ttft-rust-uni` | p1 | 16 | 402 | 789 | 2,326.7 | 162.5 | 6.15 | 2.55 |
| `ttft-rust-uni` | p2 | 1 | 141 | 143 | 339.8 | 359.2 | 2.78 | 2.58 |
| `ttft-rust-uni` | p2 | 8 | 233 | 456 | 1,551.0 | 213.6 | 4.68 | 2.59 |
| `ttft-rust-uni` | p2 | 16 | 467 | 795 | 2,285.9 | 158.2 | 6.32 | 2.54 |
| `ttft-mp-only` | p1 | 1 | 149 | 150 | 325.0 | 348.1 | 2.88 | 2.50 |
| `ttft-mp-only` | p1 | 8 | 333 | 486 | 1,510.0 | 210.6 | 4.75 | 2.57 |
| `ttft-mp-only` | p1 | 16 | 380 | 661 | 2,319.3 | 162.1 | 6.17 | 2.55 |
| `ttft-mp-only` | p2 | 1 | 148 | 149 | 325.0 | 350.5 | 2.85 | 2.53 |
| `ttft-mp-only` | p2 | 8 | 333 | 489 | 1,510.1 | 209.3 | 4.78 | 2.57 |
| `ttft-mp-only` | p2 | 16 | 345 | 664 | 2,318.5 | 158.1 | 6.33 | 2.53 |
| `ttft-api4` | p1 | 1 | 156 | 159 | 311.5 | 339.0 | 2.95 | 2.44 |
| `ttft-api4` | p1 | 8 | 331 | 479 | 1,541.4 | 212.4 | 4.71 | 2.56 |
| `ttft-api4` | p1 | 16 | 429 | 814 | 2,352.8 | 162.0 | 6.18 | 2.59 |
| `ttft-api4` | p2 | 1 | 157 | 159 | 337.7 | 366.0 | 2.73 | 2.64 |
| `ttft-api4` | p2 | 8 | 347 | 499 | 1,505.3 | 215.3 | 4.65 | 2.52 |
| `ttft-api4` | p2 | 16 | 486 | 849 | 2,268.4 | 157.7 | 6.34 | 2.55 |

C1 TTFT p50 by config:

| Config | C1 TTFT p50 (ms) | Runs |
|---|---|---|
| Rust + mp (`ab-rustmp`) | 140, 132 | A/B |
| Rust + uni | 148, 141 | split test |
| Python + mp | 149, 148 | split test |
| Python + uni (`ab-base`) | 157, 150 | A/B |
| Python ×4 API servers | 156, 157 | split test |

## Prefill step cost and CUDA graph sizes (2026-09-25, evening)

At 8K cold prefill, SGLang with ReplaySSM + FlashInfer GDN prefill (`sglang-rs-fipre-mtp3`, 41,571.8 tok/s) was ahead
of every vLLM config (30,382.3–31,528.4). A prompt-length sweep found the cause, and a CUDA graph setting fixed it.
All runs in this section used the CDMM host, MTP3 and prefix caching on (`--enable-prefix-caching`).

### Symptom: first-token time moves in ~128 ms stairs

`diag-rust-mp-cg1024` is the launcher as it stood on 2026-09-25: Rust frontend, `mp` executor, and vLLM's default
CUDA graph sizes, which stop at 1,024 tokens. Cold C1 prefill, 1 warm-up plus 4 requests per point
([prefill-sweep-rust-mp-pc.jsonl](results/logs/prefill-sweep-rust-mp-pc.jsonl)):

| Prompt tokens | 512 | 1,024 | 2,048 | 4,096 | 6,144 | 8,192 | 12,288 | 16,384 | 32,768 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTFT p50 ms | 40 | 41 | 130 | 255 | 258 | 258 | 390 | 394 | 710 |

A boundary probe on the same server ([prefill-boundary-rust-mp-pc.jsonl](results/logs/prefill-boundary-rust-mp-pc.jsonl)):

| Prompt tokens | 1,000 | 1,100 | 1,599 | 1,601 | 3,199 | 3,201 | 4,799 | 4,801 | 6,399 | 6,401 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTFT p50 ms | 41 | 132 | 131 | 131 | 130 | 257 | 260 | 261 | 261 | 264 |

- TTFT jumps at 1,024 tokens and again at 3,200. It is flat from 3,201 to 8,192 tokens, which compute alone cannot
  explain.
- The Python frontend with the uni executor (`diag-py-uni-cg1024`) shows the same stairs: 39 / 129 / 256 / 259 ms at
  1,000 / 2,000 / 5,000 / 8,192 tokens
  ([prefill-boundary-py-uni-pc.jsonl](results/logs/prefill-boundary-py-uni-pc.jsonl)). The frontend is not the cause.

### Cause

**1. Steps above 1,024 tokens run without CUDA graphs.** Each such step launches about 2,200 kernels from Python. The
torch profiler on `diag-py-uni-cg1024` ([prof_prefill.py](scripts/cg/prof_prefill.py), then
[prof_gaps.py](scripts/cg/prof_gaps.py) →
[prefill-profile-cg1024-gaps.log](results/prof/prefill-profile-cg1024-gaps.log)) measured one prefill step per row:

| Step tokens | GPU span ms | Kernels | GPU busy ms | GPU busy |
|---:|---:|---:|---:|---:|
| 1,000 (graph) | 98.6 | 2,272 | 28.9 | 29% |
| 1,800 | 248.9 | 2,261 | 44.7 | 18% |
| 2,000 | 260.7 | 2,271 | 43.1 | 17% |
| 3,200 | 251.9 | 2,174 | 64.2 | 25% |

The profiler roughly doubles CPU overhead. Unprofiled, a 1,025–8,192-token step costs about 125–130 ms (the stairs
above), and the GPU needs well under half of that. The kernel table is in
[prefill-profile-cg1024-summary.txt](results/prof/prefill-profile-cg1024-summary.txt).

**2. Prefix caching splits prompts at 1,600-token blocks.**
- The GDN layers force a 1,600-token KV block, because the attention page must be at least the Mamba state page.
- With prefix caching, vLLM uses `mamba_cache_mode="align"`, where a chunk may only end where the SSM state can be
  cached. With MTP on, the scheduler also backs off one block (`use_eagle_block_drop`).
- So a prompt runs first to the last block boundary minus one block, then the rest. The profile's step annotations
  show 5,000 tokens running as 3,200 + 1,800. By the same rule, 8,192 runs as 6,400 + 1,792 and 32,768 as five
  steps.
- We found no way around the split in this image, from reading its source. Its prefill-checkpoint path exists only
  for Kimi-K3 KDA layers, and its ReplaySSM option is Mamba2-only and excludes speculative decode.

### Fix: piecewise CUDA graphs up to 8,192 tokens

`CG=8192`, the launcher default since 2026-09-25 ([launch-qwen-upstream.sh](scripts/launch-qwen-upstream.sh)), passes
`--compilation-config` with vLLM's default capture sizes up to 1,024 plus 1,280, 1,600, 1,792, 2,048, 2,560, 3,200,
3,584, 4,096, 4,800, 5,120, 6,144, 6,400, 7,168, 8,000 and 8,192. The multiples of 1,600 match the split. Every
prefill step now replays a graph.

- **Memory:** graph capture rose from 1.55 to 5.66 GiB, and KV fell from 4,986,403 to 4,847,538 tokens (−2.8%).
  (From session notes; those server logs were not kept.)

Cold C1 prefill, TTFT p50 in ms. The CG 1,024 column is `diag-rust-mp-cg1024` or `diag-py-uni-cg1024`; the CG
8,192 column is `diag-py-uni-cg8192` ([prefill-cg8192-py-uni.jsonl](results/logs/prefill-cg8192-py-uni.jsonl)):

| Prompt tokens | 1,000 | 2,000 | 3,300 | 5,000 | 8,192 | 12,288 | 16,384 | 32,768 | 65,536 | 131,072 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CG 1,024 | 39–41 | 129–130 | 259 | 256–262 | 258–259 | 390 | 394 | 710 | – | – |
| CG 8,192 | 42 | 52 | 93 | 129 | 186 | 288 | 365 | 721 | 1,439 | 2,939 |

From 32K up, prefill is compute-bound at about 45K tok/s either way.

**Decode:** the catid recipe on `cg8192-py-1` (CG 8,192, Python + uni, GSM8K-50 48/50) against the A/B's Python +
uni arms (CG 1,024):

| C | `ab-base-1` / `-2` aggregate tok/s | `cg8192-py-1` aggregate tok/s | `ab-base-1` / `-2` TTFT p50 ms | `cg8192-py-1` TTFT p50 ms |
|---:|---:|---:|---:|---:|
| 1 | 332.4 / 332.9 | 344.0 | 157 / 150 | 80 |
| 2 | 579.6 / 571.7 | 588.1 | 181 / 181 | 106 |
| 4 | 943.4 / 943.8 | 989.1 | 315 / 177 | 105 |
| 8 | 1,529.1 / 1,488.2 | 1,671.9 | 323 / 190 | 142 |
| 16 | 2,203.3 / 2,235.2 | 2,591.7 | 394 / 334 | 175 |
| 32 | 3,170.8 / 3,253.5 | 3,784.0 | 359 / 350 | 182 |

The C8–C32 gain comes from mixed steps. New prompts' prefill chunks join the running decodes and push a step past
1,024 tokens; without graphs, that step stalled every stream for about 125 ms.

### Correctness

- **Quality gates:** GSM8K-50 48/50 (`cg8192-py-1`) and GSM8K-200 195/200 (`vllm-py-cg8192-mtp3`).
- **Teacher-forced prompt logprobs** on real text ([plp.py](scripts/cg/plp.py); [plp_offline.py](scripts/cg/plp_offline.py) →
  [plp-compare.log](results/cg/plp-compare.log)):
  - Servers: CG 1,024 is `diag-py-uni-cg1024-dev` ([plp-cg1024-a.json.gz](results/cg/plp-cg1024-a.json.gz)). CG 8,192
    is `cg8192-py-1` ([plp-cg8192-a.json.gz](results/cg/plp-cg8192-a.json.gz)) and `vllm-py-cg8192-mtp3`
    ([plp-cg8192-b.json.gz](results/cg/plp-cg8192-b.json.gz)).
  - Each prompt was a separate cold request: `/reset_prefix_cache` on the two dev-mode servers, and a unique
    `cache_salt` per request on `vllm-py-cg8192-mtp3`, which has no dev endpoints.
  - **Graphs match eager bit for bit at the same step size.** The first 1,899 positions of the 3,000-token prompts
    are identical between CG 1,024 (eager 3,000-token step) and CG 8,192 (3,200-token graph): mean |Δ logprob|
    0.0000.
  - **Changing the step size shifts the numbers, in any config.** 1,900-token prompts differ between configs by mean
    |Δ| 0.131–0.147 (NLL +0.0065 / +0.0075), because the step runs as a padded 2,048-token graph instead of an eager
    1,900-token step. Within CG 1,024 alone, the same positions move by 0.123–0.149 (NLL +0.0066 / +0.0091) when
    they are computed as part of a 3,000-token step. The NVFP4 activations amplify small GEMM-order differences.
  - **Two CG 8,192 servers agree exactly below the sparse-attention budget.** The 1,900-token prompts are identical
    between `cg8192-py-1` and `vllm-py-cg8192-mtp3` (section B: mean |Δ| 0.0000).
  - **Above 2,048 tokens,** whole-prompt differences between the configs (section A: mean |Δ| 0.039–0.102, NLL within
    ±0.008) are the same size as the differences between the two CG 8,192 servers (section B: 0.038–0.105). Within the
    CG 1,024 server, two passes differed by 0.033–0.107 (session notes).
- **Run-to-run nondeterminism above the sparse-attention budget** ([nondet.py](scripts/cg/nondet.py) →
  [nondet-vllm-py-cg8192-mtp3.log](results/cg/nondet-vllm-py-cg8192-mtp3.log)):
  - Three cold runs of the same prompt on `vllm-py-cg8192-mtp3` are identical at 1,900 and 2,040 tokens.
  - From 2,100 tokens on, the top token stays the same, but only 2–4 of the top 5 are shared across runs, and shared
    entries differ by up to 0.31–2.75.
  - Earlier servers showed the same pattern (session notes, logs not kept): identical up to 2,040 tokens, then
    differences up to 1.06–1.81 (`diag-py-uni-cg8192`) and 0.81–1.13 (`diag-py-uni-cg1024-dev`).
  - This comes from the QSA sparse attention (`indexer_budget` 2,048), not from the graphs. Prefix-cache hits on
    prompts over ~3K tokens also change greedy continuations (session notes).

## Time to first token: frontend vs engine (chosen config)

[ttft_split.py](scripts/cg/ttft_split.py) sends one cold request per size: a chat request with real text, then the
same tokens as token ids. It splits the server's own TTFT histogram delta into queue, engine prefill and the rest,
which is input processing in the Python frontend. Each request uses a unique `cache_salt`. The server was otherwise
idle: its access log shows no other client during the run
([ttft-split-quiet.log](results/logs/ttft-split-quiet.log)).

| Prompt tokens | Chat text: client TTFT ms | Engine prefill ms (text / ids) | Frontend processing ms (text / ids) | Client tok/s (text) |
|---:|---:|---:|---:|---:|
| 8,204 | 210 | 189 / 184 | 16 / 1 | 39,067 |
| 32,780 | 762 | 703 / 704 | 47 / 3 | 43,018 |
| 65,548 | 1,516 | 1,401 / 1,405 | 94 / 7 | 43,237 |
| 131,084 | 3,081 | 2,851 / 2,853 | 188 / 13 | 42,546 |

- Real text and random-looking token ids prefill at the same engine speed.
- The chat-text path adds about 1.4 µs per token in the frontend (chat template and tokenizer), about 6% at long
  prompts.
- An earlier run of this script overlapped another client's benchmark. It was discarded.

## Needle retrieval (chosen config)

[needle_test.py](scripts/needle_test.py), run by [qual-final.sh](scripts/qual-final.sh) on `vllm-py-cg8192-mtp3`. A
passphrase sits at 10%, 50% and 90% depth in deterministic filler, with thinking off and temperature 0
([qual-final.log](results/logs/qual-final.log)).

| Target tokens | Prompt tokens | 10% | 50% | 90% |
|---:|---:|---|---|---|
| 125,000 | 113,942 | PASS | PASS | PASS |
| 250,000 | 227,674 | PASS | PASS | PASS |

## 128K thinking-loop check

[pp-loop-test.sh](scripts/pp-loop-test.sh) and [pp-loop-test2.sh](scripts/pp-loop-test2.sh) ran on one server with
the launcher defaults (Rust / mp, MTP3), under CDMM:

- 8 repetitions of llm_decode_bench from a second host
- 128K context, C1 and C8, 30 s each, up to 8,192 output tokens with thinking on
- A tally script on that host (`looptally.py`, not in this repo) printed one status per cell. Its tok/s values are
  in the logs but are not reported here, because its definition isn't in the repo.
- The first script's server-side `--override-generation-config` path was abandoned. The Rust frontend ignores that
  flag, so the penalty arm ran through [pp_proxy.py](scripts/pp_proxy.py). The proxy injected presence_penalty 0.5
  into all 321 requests it saw ([pp-loop-test2.log](results/logs/pp-loop-test2.log)).

| Config | Cells | Cells flagged as looping | GSM8K-200 | Source |
|---|---:|---:|---:|---|
| `pp0` (vLLM, no penalty) | 16 | 0 | 192/200 | [pp-loop-test.log](results/logs/pp-loop-test.log), [pp-loop-test2.log](results/logs/pp-loop-test2.log) |
| `pp0.5` (vLLM, presence_penalty 0.5) | 16 | 0 | 192/200 | [pp-loop-test2.log](results/logs/pp-loop-test2.log) |
| `vllm-py-cg8192-mtp3` (chosen, no penalty) | 16 | 0 | 195/200 | [qual-final.log](results/logs/qual-final.log) |
| `sglang-rs-fipre-mtp3` | 8 runs | 3 runs | 194/200 | from session notes; raw file not kept |
| `sglang-trtllm-mtp3` (SGLang default) | not recorded | 0 | 194/200 | from session notes; raw file not kept |

## GSM8K (greedy, thinking off, last N test questions)

Raw files: `results/gsm8k/gsm8k-qwen-<config>.json` (per-question booleans included).

| Config | Correct | Accuracy |
|---|---:|---:|
| `vllm-py-cg8192-mtp3` | 195/200 | 97.5% |
| `vllm-trtllm-mtp3` | 194/200 | 97.0% |
| `vllm-rust-mp-mtp3` | 193/200 | 96.5% |
| `vllm-recipe-mtp3` | 191/200 | 95.5% |
| `pp0` | 192/200 | 96.0% |
| `pp0.5` | 192/200 | 96.0% |
| `sglang-trtllm-mtp3` | 194/200 | 97.0% |
| `sglang-rs-mtp3` | 193/200 | 96.5% |
| `sglang-rs-fipre-mtp3` | 194/200 | 97.0% |
| `sglang-mega-rsfp-mtp3` | 193/200 | 96.5% |
| `ab-base-1` | 48/50 | 96.0% |
| `ab-rustmp-1` | 49/50 | 98.0% |
| `cg8192-py-1` | 48/50 | 96.0% |
| `smoke-flashinfer_trtllm` | 48/50 | 96.0% |
| `smoke-flashinfer_cutedsl` | 49/50 | 98.0% |
| `smoke-flashinfer_cutlass` | 48/50 | 96.0% |
| `smoke-cutlass` | 48/50 | 96.0% |

## Failed, aborted and incomplete runs (no numbers reported)

| Run | Outcome | Evidence |
|---|---|---|
| `vllm-cutedsl-mtp3` | startup failed: `moe_backend='flashinfer_cutedsl' is not supported for FP8 MoE` (the MTP layer) | [vllm-variants.log](results/logs/vllm-variants.log) |
| `vllm-ficutlass-mtp3` | startup failed: FP8 MoE backend FLASHINFER_CUTLASS does not support the FP8 block-128 scheme | same |
| `vllm-cutlass-mtp3` | startup failed: vLLM CUTLASS FP8 MoE backend disabled for this configuration | same |
| `vllm-trtllm-mtp0` | aborted after the sanity prompt; no benchmark ran. The other MTP0 kernel variants in the script's default list never started. | [vllm-variants-mtp0-aborted.log](results/logs/vllm-variants-mtp0-aborted.log) |
| `smoke-flashinfer_cutlass`, `smoke-cutlass` (run 1) | startup failed: GPU memory not yet released by the previous container; rerun in run 2 | [smoke-kernels-run1.log](results/logs/smoke-kernels-run1.log) |
| `sglang-mega-mtp3` | startup failed: FLASHINFER_MEGAMOE runner needs a fused func for a2a backend `none` | [sglang-variants.log](results/logs/sglang-variants.log) |
| `sglang-fi-mtp3` (`--linear-attn-backend flashinfer`) | startup failed: `no_buffer only supports page_size=1`. The driver script then hit a bash syntax error: it was edited while running. | same |
| `ttft-*-5x` | incomplete; no results | [ttft-5x.log](results/logs/ttft-5x.log) |
| `fork-b12x-mtp3` | startup failed: no MXFP8 linear kernel (`b12x MXFP8 kernels require a Blackwell 12x device`) | [fork-b12x-mtp3.log](results/logs/fork-b12x-mtp3.log) |
| `fork-b12x-nolinear-mtp3` | startup failed: `b12x GDN prefill requires SM12x ...` | [fork-b12x-nolinear-mtp3.log](results/logs/fork-b12x-nolinear-mtp3.log) |
| `fork-auto-mtp3` | startup failed: `PLE format 'nvfp4' requires the b12x execution backend` | [fork-auto-mtp3.log](results/logs/fork-auto-mtp3.log) |
| b12x vs FlashInfer MoE layer-0 microbench | failed before timing: `SM103 NVFP4 projection requires K and N divisible by 256` (I_tp = 640) | [b12x-vs-flashinfer-layer0.log](results/shootout/b12x-vs-flashinfer-layer0.log) |

On 2026-09-24:

- The fork runs ([launch-qwen-fork.sh](scripts/launch-qwen-fork.sh)) and the microbench used the local-inference-lab
  QAD checkpoint.
- Neither produced a measurement on this model on SM103.
- The b12x kernels the fork selects (MXFP8 linear, GDN) are SM12x-only.
- The b12x SM103 MoE path rejected this model's expert shape.
- This says nothing about b12x on other models or GPUs.

## Scripts without results in this lane

- [offload_test.py](scripts/offload_test.py): a CPU KV-offload check. Its defaults (`MODEL_NAME=Qwen3.8-27B`,
  `PORT=5000`) match separate Qwen3.8-27B work on another host. No result file exists here, so no offload numbers are
  reported. [needle_test.py](scripts/needle_test.py) has the same defaults; [qual-final.sh](scripts/qual-final.sh)
  overrides them for the needle results above.
- [launch-qwen-fork.sh](scripts/launch-qwen-fork.sh): see failed runs.

## Differences from catid's Qwen3.8 page

catid's page is <https://github.com/catid/dgx_station_benchmarks/tree/main/qwen3.8-flash-next>. It uses one DGX
Station, TP1.

| | catid | this lane |
|---|---|---|
| Checkpoint | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89` | `nvidia/Qwen3.8-Flash-Next-NVFP4` |
| Engine | SGLang, custom runtime image `qwen38-4p89-sglang:runtime-v1` | stock vLLM and SGLang nightlies (tags above) |
| Memory | memory fraction 0.80 | vLLM 0.90; SGLang 0.85 |
| Decode recipe | same (8,192 in / 1,024 out, T=0, C warm-ups + 5×C) | same |
| Cold prefill | C1, 30 s per target | C1, 1 warm-up + 4 random-token requests per target, cache flushed |

The recipe numbers are comparable in method but not in checkpoint or engine build.

## Paths inside the scripts

The scripts are verbatim snapshots and keep their original absolute paths:

| Path in the scripts | Where it is here |
|---|---|
| `/home/jasonc/research/qwen38` | this lane: [scripts/](scripts/), [patches/](patches/), `results/logs`, `results/runs/{smoke,ttft}-*` |
| `/home/jasonc/research/qwen38/cg` | [scripts/cg/](scripts/cg/) (tools) and [results/cg/](results/cg/) (data; `plp-*.json` gzipped here) |
| `/home/jasonc/research/qwen38/vllm-cache/prof` | [results/prof/](results/prof/) (summary table and gap log only) |
| `/home/jasonc/spark_vllm/qwen38-loop-*` (on the bench client) | not included; the tally lines are in the driver logs |
| `/home/jasonc/ds41f-exp/bench_decode.sh` | [../bench/](../bench/) |
| `/home/jasonc/ds41f-exp/bench_prefill.py` | not included: catid's unlicensed file, [linked](https://github.com/catid/dgx_station_benchmarks/blob/ff8a496e5e027bbc462f81643361ef5516072a68/deepseek-v4.1-flash/recipes/bench_prefill.py) |
| `/home/jasonc/ds41f-exp/runs/qwen-*` | `results/runs/qwen-*` |
| `/home/jasonc/research/megamoe/gsm8k_eval.py` | [../bench/gsm8k_eval.py](../bench/gsm8k_eval.py) |
| `/home/jasonc/research/megamoe/logs/gsm8k-qwen-*.json` | `results/gsm8k/` |
| `/home/jasonc/models/*` | host checkpoint directories (not included) |
| `/home/jasonc/llm-inference-bench` | [local-inference-lab/llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) (not included) |

## Files not included

- Server logs:
  - `server-sglang-trtllm-mtp3.log` is 1.03 MB, over the 1 MB limit.
  - The other `server-*.log` files (2 KB–0.9 MB) are not cited. The driver logs above carry the lines that matter.
  - `server-sglang-rsfp-mtp3.log` (0.9 MB) appears to be the server of the SGLang 128K loop session. There are no
    client-side results for that session.
- `pp_proxy.log`: aiohttp tracebacks from client disconnects, not cited.
- The download logs.
- `qwen-fork.pid`.
- The empty `runs/fork-*` directories.
- `patches/moe_hook.py.orig`, replaced by [moe_hook.diff](patches/moe_hook.diff).
- `verify-cdmm.sh`, which belongs to the host memory-mode notes.
- Caches: `jit-cache/`, `vllm-cache/`, `sglang-cache/`.
- The torch profiler traces of `diag-py-uni-cg1024` (8.4 MB GPU trace, 0.24 MB frontend trace): over the size limit.
  The summary table and the gap log derived from them are included.
- Server logs of the `diag-*` and `cg8192-py-1` servers: the containers were removed before the logs were saved.
- `cg/README.md`, the station note the launcher's `CG` comment points to. Its content is the "Prefill step cost and
  CUDA graph sizes" section above.
