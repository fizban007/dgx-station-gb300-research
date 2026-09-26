# MiMo-V2.6-Flash-RL on one GB300: details

This file backs [README.md](README.md). It lists every run and every measured column, where each number comes
from, and what is known only from session notes. All paths are relative to this directory unless they start
with `../`.

## Configurations

All runs are from 2026-09-25 on host gracie. Each used one GB300 at TP1, the `MiMo-V2.6-Flash-RL` checkpoint and
[launch-mimo.sh](launch-mimo.sh). Times are local (EDT, UTC-4) and come from the result files.

| Config | Time | Image (vLLM nightly) | Spec | KV grouping | Decode run | Decode prompts | Qualified |
|---|---|---|---|---|---|---|---|
| nospec-7f1a | 13:26-13:36 | 7f1a5398 | none | stock | `runs/mimo-base-nospec` | unseeded | yes (GSM8K) |
| dflash7-7f1a | 13:50-13:57 | 7f1a5398 | DFlash k=7 | stock | `runs/mimo-dflash7-7f1a` | unseeded | yes (GSM8K) |
| dflash7-7f1a-pr58207 | 14:11-14:18 | 7f1a5398 | DFlash k=7 | vllm#58207 backport | `runs/mimo-dflash7-7f1a-pr58207` | unseeded | yes (GSM8K) |
| dflash7-7f1a-pr58207-seeded | 14:30-14:36 | 7f1a5398 | DFlash k=7 | vllm#58207 backport | `runs/mimo-seeded-dflash7-7f1a-pr58207` | seeded | yes (same server config as the row above) |
| dflash7-29468-pr58207 (default) | 15:00-15:07 | 29468dde | DFlash k=7 | vllm#58207 backport | `runs/mimo-dflash7-29468-pr58207` | seeded | yes (GSM8K) |
| dflash7-29468-stock | 15:15 | 29468dde | DFlash k=7 | stock (`KVGROUP=main`) | none (prefix-hit only) | - | unqualified |

- Image tags:
  - 7f1a5398 = `vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423`, which reports
    `0.30.1rc1.dev48+g7f1a5398e` and ships FlashInfer 0.6.18.post1.
  - 29468dde = `vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098`, which reports
    `0.30.1rc1.dev143+g29468dde8` and ships FlashInfer 0.7.0.
  - The engine strings come from `runs/*/decode/c1.log`. The FlashInfer versions come from session notes.
- The 7f1a runs used an earlier revision of `launch-mimo.sh` (not kept), whose default image was 7f1a5398.
  Their logs confirm the engine version, served model name and 1,048,576-token context. The other flags are
  assumed to match the kept revision.
- Every run also has prefill and GSM8K results except the seeded rerun, which had decode only, and
  `dflash7-29468-stock`, which had prefix-hit only.
  - Prefill: `logs/prefill-<label>.jsonl`
  - GSM8K: `logs/gsm8k-mimo-<label>.json`
  - Suite log: `logs/bench-<label>.log`
  - `<label>` is `base-nospec`, `dflash7-7f1a`, `dflash7-7f1a-pr58207` or `dflash7-29468-pr58207`.
- **Unseeded vs seeded decode.**
  - The three unseeded runs were written by `bench_decode.sh` from the DeepSeek-V4.1-Flash harness. An identical
    copy is at [../bench/bench_decode.sh](../bench/bench_decode.sh). They were stored under
    `/home/jasonc/ds41f-exp/runs/` and are copied into `runs/` here.
  - [bench_decode_seeded.sh](bench_decode_seeded.sh) is the same script run through [seeded.py](seeded.py), with a
    different output directory. Both seeded runs show the same padding-text run id (`udcyuascjaza`), so their
    prompts are identical.

## catid decode, every run

Recipe: 8,192 exact input tokens, 1,024 forced output tokens, temperature 0, C warm-ups then 5xC requests,
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) v0.4.29 (`llm_decode_bench.py`).
Source for each table: `runs/<run>/decode/c<C>.json`.

Column definitions:

- "agg" is aggregate tok/s.
- "user" columns are per-user: `1/ITL`, and the e2e version includes TTFT.
- "accept len" is tokens emitted per engine step (`server_spec_accept_length`).
- "engine steps/s" is aggregate tok/s divided by accept length. It measures engine speed independent of how many
  draft tokens are accepted.

### dflash7-7f1a-pr58207-seeded (seeded)

Source: `runs/mimo-seeded-dflash7-7f1a-pr58207/decode/c<C>.json`

| C | agg tok/s | user tok/s p50 | user e2e tok/s p50 | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | req latency p50 s | accept len | engine steps/s | requests | errors |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 355.1 | 368.8 | 358.4 | 83 | 83 | 2.71 | 2.86 | 3.16 | 112.7 | 5/5 | 0 |
| 2 | 512.4 | 269.0 | 259.7 | 133 | 142 | 3.72 | 3.95 | 2.92 | 175.7 | 10/10 | 0 |
| 4 | 890.2 | 227.4 | 221.9 | 133 | 151 | 4.40 | 4.61 | 3.09 | 288.6 | 20/20 | 0 |
| 8 | 1,445.7 | 201.2 | 196.1 | 142 | 223 | 4.97 | 5.22 | 3.16 | 457.3 | 40/40 | 0 |
| 16 | 2,360.2 | 156.5 | 152.9 | 156 | 310 | 6.39 | 6.70 | 2.91 | 811.2 | 80/80 | 0 |
| 32 | 3,716.1 | 124.4 | 121.9 | 166 | 518 | 8.04 | 8.40 | 2.99 | 1,241.6 | 160/160 | 0 |
| 64 | 5,443.9 | 92.4 | 88.5 | 195 | 970 | 10.82 | 11.58 | 2.94 | 1,853.5 | 320/320 | 0 |

### nospec-7f1a (unseeded)

Source: `runs/mimo-base-nospec/decode/c<C>.json`

| C | agg tok/s | user tok/s p50 | user e2e tok/s p50 | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | req latency p50 s | accept len | engine steps/s | requests | errors |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 208.9 | 212.7 | 211.0 | 44 | 47 | 4.70 | 4.85 | - | - | 5/5 | 0 |
| 2 | 368.3 | 186.4 | 184.4 | 62 | 65 | 5.37 | 5.55 | - | - | 10/10 | 0 |
| 4 | 664.1 | 168.5 | 166.5 | 81 | 100 | 5.93 | 6.15 | - | - | 20/20 | 0 |
| 8 | 1,058.8 | 134.4 | 132.7 | 104 | 152 | 7.44 | 7.72 | - | - | 40/40 | 0 |
| 16 | 1,570.6 | 100.5 | 99.5 | 110 | 267 | 9.95 | 10.29 | - | - | 80/80 | 0 |
| 32 | 2,430.1 | 75.7 | 75.2 | 109 | 484 | 13.21 | 13.61 | - | - | 160/160 | 0 |
| 64 | 3,889.4 | 61.7 | 61.1 | 134 | 932 | 16.21 | 16.76 | - | - | 320/320 | 0 |

### dflash7-7f1a (unseeded)

Source: `runs/mimo-dflash7-7f1a/decode/c<C>.json`

| C | agg tok/s | user tok/s p50 | user e2e tok/s p50 | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | req latency p50 s | accept len | engine steps/s | requests | errors |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 302.4 | 311.2 | 303.8 | 83 | 85 | 3.21 | 3.37 | 2.67 | 113.1 | 5/5 | 0 |
| 2 | 505.9 | 259.9 | 253.4 | 93 | 148 | 3.85 | 4.04 | 2.88 | 175.9 | 10/10 | 0 |
| 4 | 880.5 | 226.6 | 222.4 | 148 | 155 | 4.41 | 4.60 | 2.94 | 299.9 | 20/20 | 0 |
| 8 | 1,445.9 | 191.1 | 185.5 | 166 | 249 | 5.23 | 5.52 | 2.93 | 494.4 | 40/40 | 0 |
| 16 | 2,277.0 | 155.0 | 151.5 | 172 | 371 | 6.45 | 6.76 | 2.86 | 796.5 | 80/80 | 0 |
| 32 | 4,039.5 | 132.9 | 130.0 | 193 | 577 | 7.53 | 7.88 | 3.10 | 1,304.7 | 160/160 | 0 |
| 64 | 5,671.6 | 93.4 | 90.7 | 314 | 953 | 10.70 | 11.29 | 2.83 | 2,002.0 | 320/320 | 0 |

### dflash7-7f1a-pr58207 (unseeded)

Source: `runs/mimo-dflash7-7f1a-pr58207/decode/c<C>.json`

| C | agg tok/s | user tok/s p50 | user e2e tok/s p50 | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | req latency p50 s | accept len | engine steps/s | requests | errors |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 289.5 | 297.7 | 290.9 | 83 | 92 | 3.36 | 3.52 | 2.55 | 113.6 | 5/5 | 0 |
| 2 | 494.0 | 261.5 | 254.4 | 92 | 144 | 3.82 | 4.03 | 2.79 | 177.2 | 10/10 | 0 |
| 4 | 843.1 | 215.4 | 210.3 | 139 | 173 | 4.64 | 4.87 | 2.97 | 283.7 | 20/20 | 0 |
| 8 | 1,471.5 | 195.1 | 190.7 | 145 | 214 | 5.13 | 5.37 | 3.14 | 468.3 | 40/40 | 0 |
| 16 | 2,262.1 | 149.9 | 146.9 | 153 | 297 | 6.67 | 6.97 | 2.84 | 797.2 | 80/80 | 0 |
| 32 | 4,277.6 | 140.9 | 136.6 | 248 | 609 | 7.10 | 7.49 | 3.12 | 1,371.3 | 160/160 | 0 |
| 64 | 5,412.5 | 90.6 | 87.9 | 213 | 990 | 11.03 | 11.65 | 2.87 | 1,885.6 | 320/320 | 0 |

### dflash7-29468-pr58207, default (seeded)

Source: `runs/mimo-dflash7-29468-pr58207/decode/c<C>.json`

| C | agg tok/s | user tok/s p50 | user e2e tok/s p50 | TTFT p50 ms | TTFT p99 ms | ITL p50 ms | req latency p50 s | accept len | engine steps/s | requests | errors |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 382.9 | 401.8 | 389.9 | 81 | 82 | 2.49 | 2.63 | 3.42 | 112.2 | 5/5 | 0 |
| 2 | 519.8 | 264.1 | 256.9 | 86 | 141 | 3.79 | 3.99 | 2.96 | 175.4 | 10/10 | 0 |
| 4 | 853.7 | 218.9 | 213.8 | 98 | 152 | 4.57 | 4.79 | 3.01 | 283.7 | 20/20 | 0 |
| 8 | 1,385.7 | 181.5 | 177.1 | 141 | 209 | 5.51 | 5.78 | 2.93 | 473.2 | 40/40 | 0 |
| 16 | 2,297.5 | 156.0 | 152.4 | 157 | 307 | 6.41 | 6.72 | 2.89 | 794.5 | 80/80 | 0 |
| 32 | 3,740.8 | 118.2 | 115.7 | 183 | 523 | 8.46 | 8.85 | 2.84 | 1,318.6 | 160/160 | 0 |
| 64 | 5,685.5 | 91.9 | 89.3 | 339 | 944 | 10.88 | 11.47 | 2.81 | 2,023.2 | 320/320 | 0 |

### What the decode runs show

- **DFlash vs none (7f1a, both unseeded).** Aggregate tok/s rises at every concurrency: +45% at C1, +37% at C2,
  +33% at C4, +37% at C8, +45% at C16, +66% at C32 and +46% at C64. TTFT p50 is higher at every concurrency.
  Per-user tok/s at C1 goes from 212.7 to 311.2.
- **29468dde vs 7f1a5398 (both seeded, pr58207).** Engine steps/s change by -0.4%, -0.2%, -1.7%, +3.5%, -2.1%,
  +6.2% and +9.2% at C1-C64. The C1 tok/s gap (382.9 vs 355.1) comes from acceptance (3.42 vs 3.16), even though
  the prompts are identical. The two images do not produce identical drafts and verifications.
- **Stock vs pr58207 grouping (7f1a, both unseeded).** Steps/s moves both ways by up to about 6% (C4 -5.4%,
  C8 -5.3%, C32 +5.1%, C64 -5.8%, C1/C2/C16 within 1%). The same pr58207 config varied by up to 10% (C32: 1,371.3
  vs 1,241.6 steps/s) between its unseeded and seeded runs, so we found no consistent decode effect.
- **Seeded vs unseeded on the same config (7f1a-pr58207).** At C2-C64, acceptance differs by at most 5%. At C1 it
  differs by 24% (2.55 vs 3.16), because C1 has only 5 requests.

## Cold prefill

Method:

- catid's [`bench_prefill.py`](https://github.com/catid/dgx_station_benchmarks/blob/ff8a496e5e027bbc462f81643361ef5516072a68/deepseek-v4.1-flash/recipes/bench_prefill.py), not included because that repo has no license. See
  [../bench/README.md](../bench/README.md).
- C1, 1 warm-up and 4 requests per length, prefix cache flushed, fresh random seed per run.
- Every run reported `token_count_match: true`.

Source: `logs/prefill-<label>.jsonl`. At 8K the power figure comes from a single GPU sample.

| Config | Context | prefill tok/s | TTFT mean s | TTFT p50 s | TTFT p99 s | per-request tok/s mean | GPU power mean W | errors |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| nospec-7f1a | 8,192 | 45,524.1 | 0.180 | 0.181 | 0.183 | 45,542.6 | 395.8 | 0 |
| nospec-7f1a | 32,768 | 40,282.0 | 0.813 | 0.814 | 0.816 | 40,283.7 | 1,035.6 | 0 |
| nospec-7f1a | 65,536 | 34,261.3 | 1.913 | 1.912 | 1.915 | 34,261.8 | 1,033.0 | 0 |
| nospec-7f1a | 131,072 | 26,594.6 | 4.928 | 4.931 | 4.933 | 26,594.8 | 1,036.0 | 0 |
| dflash7-7f1a | 8,192 | 42,514.5 | 0.193 | 0.192 | 0.195 | 42,527.3 | 394.1 | 0 |
| dflash7-7f1a | 32,768 | 39,050.3 | 0.839 | 0.840 | 0.841 | 39,051.8 | 1,030.3 | 0 |
| dflash7-7f1a | 65,536 | 33,626.5 | 1.949 | 1.948 | 1.951 | 33,627.1 | 1,031.1 | 0 |
| dflash7-7f1a | 131,072 | 26,211.3 | 5.001 | 5.000 | 5.005 | 26,211.4 | 1,030.8 | 0 |
| dflash7-7f1a-pr58207 | 8,192 | 43,045.6 | 0.190 | 0.190 | 0.192 | 43,052.3 | 390.1 | 0 |
| dflash7-7f1a-pr58207 | 32,768 | 39,164.8 | 0.837 | 0.836 | 0.842 | 39,166.9 | 1,037.1 | 0 |
| dflash7-7f1a-pr58207 | 65,536 | 33,626.0 | 1.949 | 1.949 | 1.955 | 33,626.7 | 1,035.1 | 0 |
| dflash7-7f1a-pr58207 | 131,072 | 26,259.7 | 4.991 | 4.992 | 5.005 | 26,260.0 | 1,037.0 | 0 |
| dflash7-29468-pr58207 | 8,192 | 43,940.3 | 0.186 | 0.187 | 0.188 | 43,946.2 | 405.4 | 0 |
| dflash7-29468-pr58207 | 32,768 | 40,023.9 | 0.819 | 0.820 | 0.820 | 40,025.3 | 1,052.7 | 0 |
| dflash7-29468-pr58207 | 65,536 | 34,267.4 | 1.913 | 1.913 | 1.916 | 34,268.0 | 1,051.0 | 0 |
| dflash7-29468-pr58207 | 131,072 | 26,651.7 | 4.918 | 4.917 | 4.924 | 26,651.9 | 1,054.0 | 0 |

## GSM8K and DFlash acceptance

Method:

- `gsm8k_eval.py`: the last 200 GSM8K test questions, greedy (T=0), thinking off, `max_tokens` 768, 16 in flight.
  An identical copy is at [../bench/gsm8k_eval.py](../bench/gsm8k_eval.py).
- The spec counters are `/metrics` deltas across the GSM8K run, printed by [bench-mimo.sh](bench-mimo.sh).

Sources: accuracy from `logs/gsm8k-mimo-<label>.json`; drafts, accepted per draft and accept rate from
`logs/bench-<label>.log`.

| Config | Correct | Accuracy | Spec drafts | Accepted tokens per draft | Accept rate (accepted / drafted) |
|---|--:|--:|--:|--:|--:|
| nospec-7f1a | 195/200 | 97.5% | - | - | - |
| dflash7-7f1a | 196/200 | 98.0% | 8,578 | 4.26 | 60.9% |
| dflash7-7f1a-pr58207 | 196/200 | 98.0% | 8,578 | 4.26 | 60.9% |
| dflash7-29468-pr58207 | 195/200 | 97.5% | 8,472 | 4.29 | 61.3% |

- **Accepted tokens per draft** excludes the bonus token. The decode tables' "accept len" includes it
  (accepted + 1). GSM8K's 4.26-4.29 is therefore about 5.3 tokens per step, well above the 2.55-3.42 seen on catid's
  padding-text prompts.
- **Per-question results.**
  - The two 7f1a DFlash runs are identical: the same counters and the same failed questions (indices 42, 80, 99
    and 190).
  - `nospec-7f1a` and `dflash7-29468-pr58207` both fail indices 42, 57, 99, 154 and 190.

## Prefix-hit TTFT

Method: [prefix-hit-ttft.py](prefix-hit-ttft.py). For each length it sends one cold request, then 5 streamed repeats
of the same prompt. Prompt lengths are approximate.

Sources: `logs/prefix-hit-29468-pr58207.log` and `logs/prefix-hit-29468-main.log`.

| Config | Prompt | Cold TTFT ms | Hit TTFT median ms | Hit TTFT min ms | Prefix-hit tokens per request |
|---|--:|--:|--:|--:|--:|
| dflash7-29468-pr58207 | ~8K | 204 | 73 | 73 | 8,672 |
| dflash7-29468-pr58207 | ~32K | 846 | 120 | 119 | 34,624 |
| dflash7-29468-pr58207 | ~128K | 5,147 | 316 | 310 | 138,240 |
| dflash7-29468-stock (unqualified) | ~8K | 734 | 73 | 73 | 8,608 |
| dflash7-29468-stock (unqualified) | ~32K | 836 | 120 | 120 | 34,592 |
| dflash7-29468-stock (unqualified) | ~128K | 5,148 | 311 | 310 | 138,560 |

- The 734 ms cold 8K request on the stock server was the first request of that probe. It probably includes
  one-time warm-up, but we did not investigate.
- The first prefix hit after a restart is about 2x slower than the later hits, which is warm-up only.
  (from session notes; raw file not kept)

## KV capacity

llm-inference-bench prints a KV budget at startup (source: `runs/<run>/decode/c1.log`). Before the local fix, it
printed `blocks x 16`. That is not the token capacity of this hybrid full-attention/SWA layout.

| Config | Printed KV budget (tokens) | What was printed |
|---|--:|---|
| nospec-7f1a | 1,422,528 | 88,908 blocks x 16 (not group-aware) |
| dflash7-7f1a | 2,422,224 | 151,389 blocks x 16 (not group-aware) |
| dflash7-7f1a-pr58207 | 1,345,680 | 84,105 blocks x 16 (not group-aware) |
| dflash7-7f1a-pr58207-seeded | 1,345,680 | 84,105 blocks x 16 (not group-aware) |
| dflash7-29468-pr58207 | 2,259,724 | group-aware `kv_cache_size_tokens` (84,094 blocks x 16 = 1,345,504 ignored) |
| dflash7 with stock grouping | ~2,120,000 | group-aware (from session notes; raw file not kept; image not stated) |

- **Why the backport helps.** With the DFlash drafter, the layer buckets are 9 full attention, 39 SWA (window
  128) and 5 drafter SWA (window 1024).
  - Stock vLLM uses the smallest bucket as the group size. That is 5, so full attention is padded from 9 to 10
    layers.
  - vllm#58207 picks the group size with the fewest worst-case padding bytes. It keeps 9, so only SWA layers get
    bounded padding. The launcher header explains the same.
- **The block counts agree.** On 7f1a5398, 151,389 blocks x 5 layers = 756,945 = 84,105 blocks x 9 layers. The
  pool is the same size and only the page grouping differs.
- The "+6.6% KV capacity" in the session notes (2.12M -> 2.26M) compares a group-aware stock value that was not
  kept with the kept 29468dde + pr58207 value.

## DFlash k analysis

No server was restarted at k<7 for an A/B. This analysis uses the per-position acceptance counters
(`vllm:spec_decode_num_accepted_tokens_per_pos`) from one k=7 long-generation run.

- Source: [../longgen/runs/mimo-dflash7-perpos/run.json](../longgen/runs/mimo-dflash7-perpos/run.json), which
  has 79,139 tokens and 15,972 drafts.
- The config id is `dflash7-longgen`. The run files do not record the image or KV grouping, so it is
  **unqualified**.
- Tokens per step for k: `1 + sum_{i<=k} P(accept length >= i)`. This assumes the k=7 drafts are
  representative of shorter k. DFlash drafts the whole block in one pass, so a smaller k only drops the tail.

| k | Tokens per step, all | Thinking | Answer (HTML) | Projected C1 tok/s |
|--:|--:|--:|--:|--:|
| 1 | 1.84 | 1.82 | 1.96 | - |
| 2 | 2.54 | 2.51 | 2.86 | - |
| 3 | 3.15 | 3.09 | 3.72 | 415 |
| 4 | 3.69 | 3.59 | 4.53 | - |
| 5 | 4.16 | 4.03 | 5.29 | 491 |
| 6 | 4.58 | 4.43 | 5.97 | - |
| 7 | 4.96 | 4.77 | 6.56 | 545 |

- **Projected C1 tok/s** comes from session notes; no raw file was kept. It assumes each extra verified token
  costs ~0.3-0.5 ms per step at C1, anchored at the measured k=7 run (545.3 tok/s e2e).
- **k=7 is not too high.** P(accept length >= 7) is 0.373 overall, 0.349 in thinking and 0.591 in the answer.
  See [../longgen/README.md](../longgen/README.md) for the per-position table.
- **A smaller k might win only at high concurrency** (C32+), where verify cost matters more. This was not tested.

## Backends and other items known only from session notes

The server logs were not kept, so none of these have a raw file.

- **MoE:** FlashInfer TRT-LLM with MXFP4 weights. The notes say MXFP4 x MXFP8, while the `launch-mimo.sh`
  header says "MXFP4 x BF16 on SM10x". These conflict and are unresolved.
- **Dense:** DeepGEMM FP8.
- **Attention:** FA4 DiffKV (192/128), matching `head_dim` 192 and `v_head_dim` 128 in the checkpoint config.
- **KV cache:** BF16. `--kv-cache-dtype fp8` is silently ignored for MiMo until vllm#58128 lands.
- **Tool calls** parse correctly: parallel calls, with thinking on and off.
- **FlashInfer MoE autotune** reruns all 21 profiles (~5 min) on every start, even with the cache file present.
- **FA4 compile cache.** Without `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1` and a mounted cache directory, the
  first prefix hit after each restart stalled ~5-6 s while FA4 compiled `FlashAttentionForwardSm100`.
  vLLM's warm-up does not cover that shape. The launcher header documents the same.
- **llm-inference-bench KV fix:** branch `fix/vllm-hybrid-kv-budget`, commit 1b80da3 (off origin/main,
  unpushed). The pinned checkout is local branch `pinned-0b4185b-kvfix`.

## File notes

- **[launch-mimo.sh](launch-mimo.sh)** is the kept revision, from after the 29468dde switch.
  - `SPEC=none|mtp<N>|dflash<N>`.
  - `KVGROUP=pr58207` mounts `overlay/kv_cache_utils.py` on 7f1a5398 images and `overlay/kv_cache_utils.29468.py`
    on 29468dde images. Any other value runs stock vLLM.
  - `MOE_BACKEND`, `RUST_MP`, `DOCKER_ENV` and `EXTRA` are pass-through knobs.
  - It always passes `--enable-prefix-caching`.
- **[bench-mimo.sh](bench-mimo.sh)** runs GSM8K-200 with spec counters, then seeded catid decode, then cold prefill.
  - It calls `gsm8k_eval.py`, `bench_prefill.py` and catid's `summarize_decode.py` from their original paths.
  - `summarize_decode.py` is not included because catid's repo has no license. See
    [github.com/catid/dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks).
  - The four `logs/bench-*.log` files are this suite's output. Their decode tables match `runs/`.
  - The first three suite runs predate the seeded decode script and used the unseeded `bench_decode.sh`.
- **[overlay/](overlay/)**
  - `kv_cache_utils.py` (for 7f1a5398) and `kv_cache_utils.29468.py` (for 29468dde) are full vLLM files carrying the
    backport.
  - `kv_cache_utils.diff` and `kv_cache_utils.29468.diff` are `diff -u` against the unmodified vLLM file from each
    image. Those `.orig.py` files are not included.
  - Both diffs make the same change. It differs from upstream in one way: this image has no
    `--min-kv-cache-group-layers` flag, so the minimum group size is read from `VLLM_MIN_KV_CACHE_GROUP_LAYERS`
    (default 3).
  - These are vLLM files (Apache-2.0).
- **[pr58207.diff](pr58207.diff)** is the upstream PR (tests, `cache.py`, `arg_utils.py`, `kv_cache_utils.py`).
  **[pr58207-kvutils.diff](pr58207-kvutils.diff)** is its `kv_cache_utils.py` part only. Both are Apache-2.0.
- **[spec-metrics.py](spec-metrics.py)** prints the `/metrics` spec counters.
  **[check-prefix-cache.py](check-prefix-cache.py)** is a two-request prefix-cache sanity check; its output was not
  kept.
- **`runs/<run>/decode/c<C>.log`** is llm-inference-bench console output. The `.json` files hold the full
  per-request samples and server metrics.
- **Not included:**
  - `jit-cache/` and `vllm-cache/`: compile and autotune caches.
  - The overlay `.orig.py` files, replaced by diffs.
  - The `long-gen.py` symlink and `longgen/` runs, which are published in [../longgen](../longgen/README.md).
