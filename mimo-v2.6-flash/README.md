# MiMo-V2.6-Flash-RL on one GB300: vLLM with DFlash speculative decoding

## What we found

- **DFlash (k=7) is worth it.** On one GB300 at TP1, it raised catid decode aggregate throughput by 33-66% over no
  speculation across C1-C64 (nightly 7f1a5398, unseeded prompts). The cost is higher TTFT (C1 p50 44 -> 83 ms) and
  1-7% slower cold prefill. GSM8K-200 did not drop (195 -> 196 correct).
- **Default config:** nightly 29468dde, DFlash k=7, and a backport of vllm#58207's KV-group sizing.
  Seeded decode reaches 383 / 1,386 / 3,741 tok/s aggregate at C1 / C8 / C32. Against 7f1a5398 on identical
  prompts, engine steps/s are within 4% at C1-C16 and 6-9% higher at C32-C64.
- **The vllm#58207 backport only changes KV-cache grouping.** The default config reports 2,259,724 tokens of
  group-aware KV capacity. Stock grouping gives about 2.12M (from session notes; raw file not kept). Decode and
  prefix-hit TTFT do not move beyond run-to-run noise.
- **Prefix-cache hits are fast.** On the default config, streamed hit TTFT is 73 / 120 / 316 ms for ~8K / 32K / 128K
  prompts, versus 204 / 846 / 5,147 ms cold.
- The long-coding generation test (PS4 Tetris prompt) run against this server is in [../longgen](../longgen/README.md).

Every config below passed GSM8K-200 (greedy, thinking off) unless it is marked "unqualified".
Full per-run tables, provenance and the DFlash k analysis are in [DETAILS.md](DETAILS.md).
Machine-readable rows for every number are in [results.jsonl](results.jsonl), built by
[tools/make_results_jsonl.py](tools/make_results_jsonl.py).

## Headline: configs compared

catid decode = 8,192 exact input tokens, 1,024 forced output tokens, temperature 0, C warm-ups then 5xC requests,
via [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py`.
Decode columns are **aggregate** tok/s. Prefill is cold, C1, 4 requests per length. All runs on 2026-09-25.

| Config | Decode prompts | C1 | C2 | C4 | C8 | C16 | C32 | C64 | TTFT p50 C1 / C32 (ms) | Prefill 8K / 128K (tok/s) | GSM8K-200 |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| nospec-7f1a | unseeded | 208.9 | 368.3 | 664.1 | 1,058.8 | 1,570.6 | 2,430.1 | 3,889.4 | 44 / 109 | 45,524.1 / 26,594.6 | 195/200 |
| dflash7-7f1a | unseeded | 302.4 | 505.9 | 880.5 | 1,445.9 | 2,277.0 | 4,039.5 | 5,671.6 | 83 / 193 | 42,514.5 / 26,211.3 | 196/200 |
| dflash7-7f1a-pr58207 | unseeded | 289.5 | 494.0 | 843.1 | 1,471.5 | 2,262.1 | 4,277.6 | 5,412.5 | 83 / 248 | 43,045.6 / 26,259.7 | 196/200 |
| dflash7-7f1a-pr58207-seeded | seeded | 355.1 | 512.4 | 890.2 | 1,445.7 | 2,360.2 | 3,716.1 | 5,443.9 | 83 / 166 | (same server as above) | (same server as above) |
| **dflash7-29468-pr58207** (default) | seeded | 382.9 | 519.8 | 853.7 | 1,385.7 | 2,297.5 | 3,740.8 | 5,685.5 | 81 / 183 | 43,940.3 / 26,651.7 | 195/200 |

Sources: `runs/<run>/decode/c<C>.json`, `logs/prefill-<label>.jsonl`, `logs/gsm8k-mimo-<label>.json` (mapping in
[DETAILS.md](DETAILS.md#configurations)).

Unseeded runs use different prompt text, and DFlash acceptance depends on the prompt. The same server config
gave C1 acceptance of 2.55 unseeded and 3.16 seeded. Compare speculative configs on the seeded rows or on engine
steps/s.

## Headline: default config, every decode column

`dflash7-29468-pr58207`, seeded prompts. Source: `runs/mimo-dflash7-29468-pr58207/decode/c<C>.json`.
Per-user = 1 / inter-token latency. Every cell completed 5xC of 5xC requests with 0 errors.

| C | agg tok/s | per-user tok/s p50 | per-user e2e tok/s p50 | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | req latency p50 s | accept len | engine steps/s |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 382.9 | 401.8 | 389.9 | 81 | 82 | 2.49 | 2.63 | 3.42 | 112.2 |
| 2 | 519.8 | 264.1 | 256.9 | 86 | 141 | 3.79 | 3.99 | 2.96 | 175.4 |
| 4 | 853.7 | 218.9 | 213.8 | 98 | 152 | 4.57 | 4.79 | 3.01 | 283.7 |
| 8 | 1,385.7 | 181.5 | 177.1 | 141 | 209 | 5.51 | 5.78 | 2.93 | 473.2 |
| 16 | 2,297.5 | 156.0 | 152.4 | 157 | 307 | 6.41 | 6.72 | 2.89 | 794.5 |
| 32 | 3,740.8 | 118.2 | 115.7 | 183 | 523 | 8.46 | 8.85 | 2.84 | 1,318.6 |
| 64 | 5,685.5 | 91.9 | 89.3 | 339 | 944 | 10.88 | 11.47 | 2.81 | 2,023.2 |

"accept len" = tokens emitted per engine step (accepted draft tokens + 1).

## Default configuration

- **Hardware:** host gracie, one GB300 (`GPU-c146511a-...`), TP1, `numactl --membind=0`.
- **Checkpoint:** `MiMo-V2.6-Flash-RL`, 166 GB on disk. Routed experts are stored MXFP4 and dense layers are FP8
  block-128. There are 48 layers (9 full attention, 39 sliding-window with window 128) and a 1,048,576-token
  context. The DFlash drafter in `dflash/` has 5 sliding-window layers (window 1024) and drafts blocks of 8.
  (These facts come from the checkpoint's `config.json` files and its size on disk; neither is included here.)
- **Engine:** `vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098` (reports
  `0.30.1rc1.dev143+g29468dde8`).
- **Launch:** [launch-mimo.sh](launch-mimo.sh) with its defaults (`SPEC=dflash7 KVGROUP=pr58207`). The serve
  arguments are:

```
vllm serve --model /models/MiMo-V2.6-Flash-RL --served-model-name mimo-v26-flash --trust-remote-code
  --tensor-parallel-size 1 --max-model-len auto --max-num-seqs 64 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.93 --enable-prefix-caching --generation-config vllm
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice
  --speculative-config '{"method":"dflash","model":"/models/MiMo-V2.6-Flash-RL/dflash","num_speculative_tokens":7}'
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --host 0.0.0.0 --port 30006
```

  The container also gets `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` with a mounted FA4 cache directory. The
  backported `overlay/kv_cache_utils.29468.py` is mounted read-only over `vllm/v1/core/kv_cache_utils.py`.

## Backends vLLM picked

These come from session notes; the server logs were not kept.

- MoE: FlashInfer TRT-LLM fused MoE with MXFP4 weights. The notes say the activations are MXFP8, but the
  launcher comment says BF16. This is unresolved.
- Dense GEMMs: DeepGEMM FP8.
- Attention: FA4 with different K/V head sizes (192/128). The startup warning "FA4 ... head_size=192 ... defaulting
  to FA version 2" is misleading. The base class logs it, then `FlashAttentionDiffKVImpl` selects FA4 again.
- KV cache: BF16.

## Gotchas

- **FA4 compile cache.** It is off by default. Without `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` and a persistent
  directory, the first prefix-cache hit after each restart JIT-compiles an FA4 kernel, which stalls it about 5-6 s.
- **Thinking is on by default.** The recipe says leaving out `enable_thinking` disables thinking, but the chat
  template thinks unless the request sends `chat_template_kwargs: {"enable_thinking": false}`.
- **`--kv-cache-dtype fp8` is ignored** for this model until vllm#58128 lands, so the KV cache stays BF16.
- **KV budget reporting.** llm-inference-bench originally printed KV capacity as blocks x 16, which is wrong for this
  hybrid layout. A local fix reads vLLM's group-aware `kv_cache_size_tokens`: branch `fix/vllm-hybrid-kv-budget`,
  commit 1b80da3, unpushed. Only the `dflash7-29468-pr58207` run used it. See
  [DETAILS.md](DETAILS.md#kv-capacity).
- **FlashInfer MoE autotune reruns all 21 profiles on every start** (~5 min), even with the cache file present. No
  upstream fix was found. (from session notes)

## Files

| Path | What |
|---|---|
| [launch-mimo.sh](launch-mimo.sh) | Server launcher (`SPEC`, `KVGROUP`, `IMAGE`, `MOE_BACKEND` knobs) |
| [bench-mimo.sh](bench-mimo.sh) | Suite: GSM8K-200, seeded catid decode, cold prefill 8K-128K |
| [bench_decode_seeded.sh](bench_decode_seeded.sh), [seeded.py](seeded.py) | catid decode with a fixed `random` seed, so every server sees the same prompts |
| [spec-metrics.py](spec-metrics.py) | Reads spec-decode counters from `/metrics` |
| [check-prefix-cache.py](check-prefix-cache.py), [prefix-hit-ttft.py](prefix-hit-ttft.py) | Prefix-cache sanity check and hit-TTFT probe |
| [overlay/](overlay/) | vllm#58207 backport for each image, plus a `diff -u` against the stock file |
| [pr58207.diff](pr58207.diff), [pr58207-kvutils.diff](pr58207-kvutils.diff) | Upstream PR diff, full and `kv_cache_utils.py`-only |
| [runs/](runs/) | Raw llm-inference-bench output (`c<C>.json` and `c<C>.log`) for the five decode runs |
| [logs/](logs/) | Suite logs, prefill jsonl, GSM8K json, prefix-hit logs |
| [DETAILS.md](DETAILS.md) | Every run and column, provenance, KV capacity, DFlash k analysis, file notes |

The scripts are verbatim snapshots and still use their original absolute paths: `/home/jasonc/research/mimo26`
is this directory.

**License note:** the files in `overlay/` are vLLM source files (Apache-2.0) carrying a backport of vllm#58207.
`pr58207.diff` and `pr58207-kvutils.diff` are taken from that vLLM PR (Apache-2.0).
