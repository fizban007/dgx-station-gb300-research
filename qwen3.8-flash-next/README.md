# Qwen3.8-Flash-Next NVFP4 on one GB300

We served Qwen3.8-Flash-Next with `nvidia/Qwen3.8-Flash-Next-NVFP4` on one DGX Station GB300, TP1, on 2026-09-24
and 2026-09-25. We compared the vLLM and SGLang nightlies, NVFP4 MoE kernels, MTP3, the vLLM Rust frontend and `mp`
executor, FlashInfer MegaMoE at TP1, and a b12x fork, and fixed a vLLM prefill slowdown by capturing larger CUDA
graphs. Full tables, every run and all columns are in
[DETAILS.md](DETAILS.md). Machine-readable rows are in [results.jsonl](results.jsonl).

## What we found

- **vLLM's slow short-prompt prefill came from a CUDA graph limit, and larger graph sizes fix it.**
  - vLLM captures CUDA graphs only up to 1,024 tokens, and prefix caching splits this model's prompts at 1,600-token
    blocks. Every prefill step over 1,024 tokens ran without graphs and took about 125–130 ms, with the GPU busy for
    less than half of it (the rest was kernel-launch overhead).
  - Capturing piecewise graphs up to 8,192 tokens (`CG=8192`) cut 8K cold-prefill TTFT from 258 to 188 ms
    (43,612.1 tok/s), ahead of the best SGLang variant (197 ms, 41,571.8 tok/s).
  - Against the best earlier vLLM run at each concurrency, catid decode rose 9–16% at C8–C64 (C8: 1,670.9 vs 1,529.1
    tok/s; C64: 5,247.0 vs 4,630.7) and 1–6% at C1–C4. TTFT p50 at C16 fell to 165 ms (177–394 ms before).
  - Outputs are bit-identical to the old path at the same step size, and GSM8K-200 is 195/200.
  - See [DETAILS.md](DETAILS.md#prefill-step-cost-and-cuda-graph-sizes-2026-09-25-evening).
- **vLLM beats SGLang on the same MoE kernel** (FlashInfer TRT-LLM NVFP4, MTP3). With the chosen config
  (`vllm-py-cg8192-mtp3`):
  - C1: 344.0 tok/s vs 324.0–336.9 on SGLang.
  - C64: 5,247.0 vs 3,086.0–3,555.4.
  - Prefill at 8K: 43,612.1 vs 34,418.4–41,571.8 tok/s. At 128K: 44,398.9 vs 34,134.0–37,912.8.
  - The one exception is 32K prefill, where SGLang with FlashInfer GDN prefill is 1.3% ahead (45,703.6 vs 45,124.1).
- **Chosen setup:** vLLM with the Python frontend, uni executor, MTP3 and `CG=8192`, plus tool calling and image and
  video input. These are the launcher defaults since the evening of 2026-09-25.
  - Earlier that day the Rust frontend + `mp` executor was chosen for its TTFT (C16 177–180 ms vs 334–394).
  - With the graph fix, the Python frontend beats those numbers (C16 165 ms and 2,592.3 vs 2,227.2–2,243.7 tok/s).
  - It also keeps what vllm-rs lacks: engine stats in `/metrics`, `--override-generation-config`, and structured
    outputs for `tool_choice: required`.
  - Rust with `CG=8192` was not tested.
- **MoE kernel:** TRT-LLM (the auto pick) was the fastest NVFP4 MoE kernel vLLM offers on SM103 at every point we
  measured. FlashInfer MegaMoE at TP1 on SGLang passed GSM8K but reached 106.9 tok/s at C1, against 336.9 without it.
- **b12x:** the fork and the b12x MoE microbench did not run on this model on SM103, so there are no b12x numbers.
- **Against catid's single-station SGLang page:** vLLM MTP3 here is higher at C16, C64 and 64K prefill than both of
  his rows. It is lower at C1 than his MTP3 + ReplaySSM row. catid used a different checkpoint.

## Chosen configuration

`vllm-py-cg8192-mtp3`: [scripts/launch-qwen-upstream.sh](scripts/launch-qwen-upstream.sh) with its defaults
(`RUST_MP=0 MTP=3 CG=8192 VISION=video`):

- **Image:** `vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423` (reports `v0.30.1rc1.dev48+g7f1a5398e`).
- **Checkpoint:** `nvidia/Qwen3.8-Flash-Next-NVFP4`, local path `/models/Qwen3.8-Flash-Next-NVFP4-nvidia`.
- **Launch:**
  - `numactl --membind=0 vllm serve --tensor-parallel-size 1 --max-model-len 262144 --max-num-seqs 128`
  - `--max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 --enable-prefix-caching`
  - `--limit-mm-per-prompt '{"image":16,"video":1}' --reasoning-parser qwen3`
  - `--enable-auto-tool-choice --tool-call-parser qwen3_xml`
  - `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`
  - `--compilation-config` with CUDA graph capture sizes up to 8,192 tokens: vLLM's defaults up to 1,024, plus 1,280,
    1,600, 1,792, 2,048, 2,560, 3,200, 3,584, 4,096, 4,800, 5,120, 6,144, 6,400, 7,168, 8,000 and 8,192.
  - Python frontend and uni executor (vLLM's defaults at TP1).
- **KV cache:** 4,802,194 tokens (BF16), 18.3 full 262K contexts
  ([server log](results/logs/server-vllm-py-cg8192-mtp3.log)).
- **Kernels (auto):** MoE on `FLASHINFER_TRTLLM` for both the NVFP4 experts and the FP8 MTP layer. FlashInfer
  autotune is on (the default).
- **Quality:** GSM8K-200, thinking off: 195/200. Needle retrieval passes at 10/50/90% depth in 114K- and 228K-token prompts
  (6/6). The 128K thinking-loop check flagged 0 of 16 cells (8 runs × C1 and C8).
- The Rust + `mp` config chosen earlier (`vllm-rust-mp-mtp3`) used
  [launch-qwen-upstream-2026-09-25.sh](scripts/launch-qwen-upstream-2026-09-25.sh) with `RUST_MP=1 MTP=3`.

## Decode, catid recipe (aggregate tok/s)

The recipe is 8,192 exact input tokens, 1,024 forced output tokens and temperature 0. Each cell runs C warm-ups, then
5×C requests. All cells completed with 0 errors, and every config below passed GSM8K-200. Per-user tok/s, TTFT, ITL
and MTP acceptance are in [DETAILS.md](DETAILS.md#decode-catid-recipe-all-columns).

| Config | C1 | C2 | C4 | C8 | C16 | C32 | C64 | GSM8K-200 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vLLM Python + uni, MTP3, CUDA graphs to 8,192 (`vllm-py-cg8192-mtp3`, chosen) | 344.0 | 605.5 | 1,001.4 | 1,670.9 | 2,592.3 | 3,785.5 | 5,247.0 | 195/200 |
| vLLM Rust + mp, MTP3 (`vllm-rust-mp-mtp3`, chosen before the graph fix) | 313.2 | 550.3 | 925.1 | 1,492.0 | 2,197.7 | 3,181.2 | 4,332.8 | 193/200 |
| vLLM Python + uni, MTP3 (`vllm-trtllm-mtp3`) | 337.9 | 579.1 | 931.7 | 1,486.6 | 2,219.5 | 3,142.5 | 4,630.7 | 194/200 |
| SGLang default, MTP3 (`sglang-trtllm-mtp3`) | 324.0 | 566.8 | 858.9 | 1,294.1 | 1,781.1 | 2,367.5 | 3,086.0 | 194/200 |
| SGLang ReplaySSM + FI GDN prefill (`sglang-rs-fipre-mtp3`) | 336.9 | 585.4 | 928.3 | 1,403.8 | 1,978.6 | 2,803.3 | 3,555.4 | 194/200 |
| SGLang FlashInfer MegaMoE TP1 (`sglang-mega-rsfp-mtp3`) | 106.9 | 201.1 | 367.1 | 667.0 | 1,078.7 | 1,756.5 | 2,558.3 | 193/200 |

Raw files: `results/runs/qwen-<config>/decode/c<C>.json` and `results/gsm8k/gsm8k-qwen-<config>.json`.

- Dates: `vllm-trtllm-mtp3`, `sglang-trtllm-mtp3` and `sglang-rs-fipre-mtp3` ran on 2026-09-24, and
  `vllm-rust-mp-mtp3` and `sglang-mega-rsfp-mtp3` on 2026-09-25. All five ran in the host's original NUMA memory
  mode.
- The A/B below and `vllm-py-cg8192-mtp3` (evening of 2026-09-25) ran after the switch to CDMM.
- A second server with the same graph setting (`cg8192-py-1`, GSM8K-50 48/50) matched `vllm-py-cg8192-mtp3` within 3%
  at C1–C32 ([DETAILS.md](DETAILS.md#prefill-step-cost-and-cuda-graph-sizes-2026-09-25-evening)).
- The single-run C1 dip of `vllm-rust-mp-mtp3` (313.2) did not reproduce in the A/B. `vllm-recipe-mtp3` and
  `sglang-rs-mtp3` are in DETAILS.

## Cold prefill, C1 (tok/s)

| Config | 8K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|
| `vllm-py-cg8192-mtp3` (chosen) | 43,612.1 | 45,124.1 | 45,164.4 | 44,398.9 |
| `vllm-rust-mp-mtp3` | 31,402.5 | 45,844.5 | 46,446.3 | 45,529.8 |
| `vllm-trtllm-mtp3` | 31,528.4 | 44,846.6 | 45,393.6 | 44,574.5 |
| `sglang-trtllm-mtp3` | 36,742.1 | 40,012.5 | 38,663.8 | 34,134.0 |
| `sglang-rs-fipre-mtp3` | 41,571.8 | 45,703.6 | 43,623.4 | 37,912.8 |
| `sglang-mega-rsfp-mtp3` | 34,418.4 | 39,892.8 | 38,631.0 | 34,246.7 |

Raw files: `results/logs/prefill-<config>.jsonl`. Each point is random tokens with the cache flushed, 1 warm-up plus 4
requests, and one output token.

## Rust frontend + mp executor A/B (vLLM, MTP3, CDMM host)

This A/B decided the earlier default. Both arms used CUDA graphs only up to 1,024 tokens; the chosen config now beats
both (see the decode table).

The arms alternated on fresh servers ([ab-rust-mp.sh](scripts/ab-rust-mp.sh)). Each cell shows run 1 / run 2. Run 2
servers were not re-gated with GSM8K (unqualified). Run 1 passed GSM8K-50 at 48/50 (base) and 49/50 (Rust + mp).

| C | Python + uni aggregate tok/s | Rust + mp aggregate tok/s | Python + uni TTFT p50 ms | Rust + mp TTFT p50 ms |
|---:|---:|---:|---:|---:|
| 1 | 332.4 / 332.9 | 333.9 / 339.7 | 157 / 150 | 140 / 132 |
| 2 | 579.6 / 571.7 | 565.7 / 565.8 | 181 / 181 | 165 / 163 |
| 4 | 943.4 / 943.8 | 936.0 / 925.8 | 315 / 177 | 167 / 161 |
| 8 | 1,529.1 / 1,488.2 | 1,479.3 / 1,521.6 | 323 / 190 | 171 / 168 |
| 16 | 2,203.3 / 2,235.2 | 2,227.2 / 2,243.7 | 394 / 334 | 180 / 177 |
| 32 | 3,170.8 / 3,253.5 | 3,122.3 / 3,109.4 | 359 / 350 | 311 / 311 |

Raw files: `results/runs/qwen-ab-{base,rustmp}-{1,2}/decode/`.

- In a split test, C1 TTFT p50 was 141–149 ms with either the Rust frontend or `mp` alone, against 132–140 ms with
  both. That test used a shorter request count and is unqualified; see
  [DETAILS.md](DETAILS.md#split-test-ttft-quicksh-unqualified).

## MoE kernel smoke test (vLLM, MTP off)

The smoke test used a minimal prompt and 1,024 forced output tokens, with 2×C requests, so it is not the catid
recipe. Numbers are aggregate tok/s.

| `--moe-backend` | C1 | C16 | C64 | C128 | 8K prefill | 64K prefill | GSM8K-50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flashinfer_trtllm` (auto pick) | 203.1 | 2,238.6 | 6,022.4 | 8,966.6 | 32,056.2 | 43,876.0 | 48/50 |
| `flashinfer_cutedsl` | 180.4 | 1,778.1 | 4,620.3 | 7,042.0 | 26,448.2 | 41,357.0 | 49/50 |
| `flashinfer_cutlass` | 147.6 | 1,563.5 | 5,230.8 | 8,039.5 | 30,573.3 | 37,536.9 | 48/50 |
| `cutlass` (vLLM) | 104.1 | 1,352.3 | 4,579.5 | 7,133.5 | 26,039.0 | 32,036.7 | 48/50 |

Raw files: `results/runs/smoke-<kernel>/`, `results/logs/prefill-smoke-<kernel>.jsonl` and
`results/gsm8k/gsm8k-qwen-smoke-<kernel>.json`.

`--moe-backend` also applies to the FP8 MTP layer. With MTP3 on, the CuteDSL, FlashInfer CUTLASS and vLLM CUTLASS
choices all failed at startup ([vllm-variants.log](results/logs/vllm-variants.log)).

## Comparison with catid's Qwen3.8 page

catid's [Qwen3.8-Flash-Next page](https://github.com/catid/dgx_station_benchmarks/tree/main/qwen3.8-flash-next)
uses one DGX Station, TP1, SGLang, and checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`. That is not
our checkpoint. He publishes an AR row and an MTP3 + ReplaySSM row, kept separate here. His numbers are quoted from
the page and were not measured by us. Decode is aggregate tok/s with the same recipe.

| Source | Config | C1 | C16 | C64 | 64K prefill |
|---|---|---:|---:|---:|---:|
| catid | TP1 / AR (SGLang) | 202.1 | 1,883.9 | 4,090.4 | 38,653 |
| catid | TP1 / MTP3 + ReplaySSM (SGLang) | 354.6 | 1,733.2 | 2,927.8 | 37,884 |
| this lane | `vllm-py-cg8192-mtp3` (chosen, MTP3) | 344.0 | 2,592.3 | 5,247.0 | 45,164.4 |
| this lane | `vllm-rust-mp-mtp3` (MTP3) | 313.2 | 2,197.7 | 4,332.8 | 46,446.3 |
| this lane | `vllm-trtllm-mtp3` (MTP3) | 337.9 | 2,219.5 | 4,630.7 | 45,393.6 |
| this lane | `sglang-rs-fipre-mtp3` (MTP3 + ReplaySSM) | 336.9 | 1,978.6 | 3,555.4 | 43,623.4 |

- We have no AR (MTP0) run with the catid recipe: the one attempt was aborted.
- The prefill methods differ slightly. See [DETAILS.md](DETAILS.md#differences-from-catids-qwen38-page).

## Known limitations

- **Rust frontend (vllm-rs in this nightly; not used by the chosen config):**
  - Its `/metrics` has no KV-cache budget. llm_decode_bench logs "KV cache budget not available", for example in
    [this run's log](results/runs/qwen-vllm-rust-mp-mtp3/decode/c1.log). MTP acceptance is still reported.
  - The "Default vLLM sampling parameters" startup line is missing, and `structured_outputs_config` is ignored
    ([ttft-rust-uni.log](results/logs/ttft-rust-uni.log) vs [ttft-mp-only.log](results/logs/ttft-mp-only.log)).
  - `--override-generation-config` is accepted but silently ignored. (From session notes; raw file not kept.)
- **presence_penalty:** vLLM applies only temperature, top_k, top_p, min_p, repetition_penalty and max_new_tokens as
  server-side defaults. presence_penalty must come per request, so [pp_proxy.py](scripts/pp_proxy.py) injects it.
  With 0.5 injected, GSM8K-200 stayed at 192/200, and no 128K loops were flagged in 16 cells
  ([pp-loop-test2.log](results/logs/pp-loop-test2.log)).
- **128K thinking loops on SGLang:** SGLang ReplaySSM + FI GDN prefill looped in 3 of 8 runs at 128K, and default
  SGLang in none. (From session notes; raw file not kept.) The chosen vLLM config flagged 0 of 16 cells.
- **Python frontend tokenization:** for text prompts, the Python frontend's tokenizer adds about 1.4 µs per prompt
  token to TTFT: 16 ms at 8K, 188 ms at 128K, about 6% of a long prompt's TTFT. It is paid again on every turn, even
  on a prefix-cache hit. The engine prefills at 46–47K tok/s from 32K up. A client sending text sees about 42–43K; for
  example, llm_decode_bench measured 42,550 / 42,714 / 41,943 tok/s at 32K / 64K / 128K
  ([DETAILS.md](DETAILS.md#time-to-first-token-frontend-vs-engine-chosen-config)). Prompts sent as token ids skip most
  of it. The Rust frontend was not measured this way.
- **Sparse-attention nondeterminism:** above 2,048 prompt tokens (the QSA indexer budget), repeated cold runs of the
  same prompt give slightly different logprobs. The top token stays the same. This happens in every vLLM config we
  tested, and prefix-cache hits also change long greedy continuations
  ([DETAILS.md](DETAILS.md#correctness)).
- **Single runs:** except for the A/B and the two `CG=8192` servers, each config ran once. Treat differences of a
  few percent as noise.
- **Not measured:** FP8 KV cache, KV offload, and a catid-recipe AR run. [offload_test.py](scripts/offload_test.py)
  is included but has no results here.

## Files

- [DETAILS.md](DETAILS.md): setup, every config's flags, all tables and columns, failed runs, the path map, and
  excluded files.
- [results.jsonl](results.jsonl): one row per number. Rebuild it with
  [tools/make_results_jsonl.py](tools/make_results_jsonl.py).
- [scripts/](scripts/): verbatim launchers, drivers and tests. The shared decode and GSM8K harness is in
  [../bench/](../bench/). [scripts/cg/](scripts/cg/) holds the CUDA graph investigation's profiling and correctness
  tools.
- [patches/](patches/): SGLang `moe_hook.py` with three local changes for MegaMoE at TP1 on this model.
  [moe_hook.diff](patches/moe_hook.diff) shows the changes. SGLang source, Apache-2.0.
- `results/`: raw llm_decode_bench JSON and logs (`runs/`), driver logs, prefill JSONL and b12x fork logs (`logs/`),
  per-question GSM8K (`gsm8k/`), the failed b12x MoE microbench log (`shootout/`), the prefill profile (`prof/`) and
  the CUDA graph correctness data (`cg/`).
