# MiMo-V2.6-Pro on one GB300, with an RTX PRO 6000 sidecar and Grace memory

MiMo-V2.6-Pro-RL is 527 GiB: 1.02T parameters, 42B active, 384 routed experts in each of 69 MoE layers,
with MXFP4 experts. That is more than twice the GB300's HBM. We serve it from one DGX Station by
spreading its experts over three tiers:

| tier | holds | share of decode routes |
|---|---|--:|
| GB300 HBM | 8,181 hottest experts (152.8 GiB) | 59.7% |
| **RTX PRO 6000 sidecar** | next 4,829 experts (90.2 GiB) | 17.3% |
| Grace LPDDR5X, read over NVLink-C2C | remaining 13,486 experts | 23.0% |

The shares are for the qualified v2 placement ([`hook/rowmap-3tier-v2.json`](hook/rowmap-3tier-v2.json)).
They come from Al-ENGR's decode routing histogram ([`hook/expert_hist_mix.json`](hook/expert_hist_mix.json)).

## What we found

- **About 2× faster prefill, and faster decode, than the two-tier recipe we started from.** Al-ENGR's verified
  recipe keeps experts in HBM and Grace only. On the same benchmark scripts, adding the sidecar tier gave
  2.3–2.8K tok/s prefill (his receipts: about 1.1–1.4K). C1 decode rose 6–7%, and C16 aggregate went from 63 to
  102 tok/s.
- **Quality held up.** GSM8K-200 scored 98.5%. BFCL dev scored 92.83% vs his 93.83%. The 27 discordant cases
  split 17 to 10 (p≈0.25), so the difference is not significant.
- **The sidecar is cheap to wait for.** The GB300 spends under 1% of each forward pass waiting on the RTX PRO
  6000, from C1 through C16, with zero timeouts in 1,000+ passes.
- **Grace is the remaining bottleneck.** At C1 the Grace-resident experts cost 9.5 of 26 ms per token. Staging
  them into HBM before the GEMM was correct but gained only 1–2% with Marlin.
- **FlashInfer TRT-LLM banks (v4) nearly double prefill.** Every expert GEMM reads HBM. A C1 request reaches
  46 tok/s decode and 5.0–5.8K tok/s prefill, with the v2 placement and a BF16 KV cache (413K tokens).
- **v5 trades hot experts for context.** It moves 38 GiB of hot experts to Grace (rowmap v3) and switches to an
  FP8 KV cache. The KV cache grows 5.8×, from 413K to 2.41M tokens: room for two full 1M-token requests.
  Single-stream decode drops 10%, from 46 to 41 tok/s. FP8 KV needed a fix in FA4's FP8 split-KV decode, now
  a draft upstream PR ([Dao-AILab/flash-attention#2918](https://github.com/Dao-AILab/flash-attention/pull/2918)).
- **DFlash k=3 speculative decoding (v6) helps a single stream only.** C1 decode goes to 46–51 tok/s, but the
  knee does not improve: 87.6 tok/s at C16, vs 90.4 on v5. The drafter's cache costs 363K tokens of KV.
- **v4–v6 have not passed a GSM8K or BFCL gate.** v6 passed needle retrieval at 11.5K, 86K and 357K tokens
  (session notes). It ran the [long-coding test](../longgen/README.md) at 45.6 tok/s for 91K tokens, but the
  game it wrote does not run.

## Results

All measurements were taken on 2026-09-25 with Al-ENGR's scripts. The TTFT bench sends one request per prompt
size, reporting prefill (prompt / TTFT) and then decode after the first token. The knee sends 192-token prose
completions at C1–C16, and its numbers are aggregate tok/s. Each version adds to the one above it, except v4,
which replaces Marlin.

| version | change | KV cache (tokens) | prefill, 2.9K / 11.5K / 45.7K prompt | decode after TTFT, same prompts | knee C1 / C4 / C8 / C16 | quality |
|---|---|--:|--:|--:|--:|---|
| Al-ENGR v23 | HBM + Grace, Marlin | 302K (262K context) | 1,064¹ / 1,376 / 1,244¹ | 33.2¹ / 37.4 / 32.7¹ | 30.7 / 50.5 / 63.7 / 63.0² | BFCL dev 93.83% |
| **v2** | + RTX PRO 6000 sidecar | 361K⁷ | 2,335 / 2,812 / 2,775 | 39.6 / 39.4 / 38.6 | 32.3 / 61.2 / 79.7 / 102.2³ | **qualified:** GSM8K-200 98.5%, BFCL dev 92.83% |
| v3 | + Grace→HBM staging, fused send | 323K⁷ | 2,388 / 2,875 / 2,839 | 40.4 / 39.9 / 39.5 | 32.0 / 60.6 / 78.5 / 98.2 | component tests only |
| v4 | TRT-LLM banks instead of Marlin | 413,542⁷ | 2,980 / 5,032 / 5,775 | 46.3 / 46.1 / 45.1 | not run | unqualified |
| v5 | + FP8 KV cache, 115 GiB hot (rowmap v3) | 2,407,731⁷ | 2,826 / 4,961 / 5,933 | 41.7 / 41.3 / 40.9 | 23.4⁴ / 56.0 / 70.6 / 90.4 | unqualified |
| v6 | + DFlash k=3 | 2,044,736⁷ | 2,625 / 4,616 / 5,500 | 47.7 / 51.0 / 46.0⁵ | 32.2 / 48.3 / 66.4 / 87.6⁶ | unqualified; needles pass at 11.5K / 86K / 357K⁷ |

¹ His v24 receipt (1M-context variant, 110 GiB hot). He publishes v23 numbers for the 11.5K prompt only.
² His v22 receipt, which has the same budget as v23. His lane is capped at 8 sequences, so C16 does not scale.
³ This v2 knee run is from session notes; its raw log was not kept (see [DETAILS](DETAILS.md#knee-v2)).
⁴ The two C1 repetitions were 32.2 and 14.6 tok/s; the second is an outlier.
⁵ The three repetitions at each size spread from 43.7 to 55.1 tok/s. With speculative decoding, the rate depends on
how many drafted tokens are accepted.
⁶ The rerun tagged "quiet". The first v6 knee run was within 3% of it at C4–C16, and 30.2 at C1.
⁷ From session notes and the lane's own notes; the boot logs and needle output were not kept. Our context is
1M tokens; his v23 is 262K.

Sources: [`results/`](results/), [`results/logs/`](results/logs/), his
[recipe README and receipts](https://github.com/J-M-Recipes/recipes/tree/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/dgx-station-gb300/mimo-v2.6-pro-vllm-uva-hotsplit).
Every number, with its source file, is in [`results.jsonl`](results.jsonl). Config ids there: `3tier-v2`,
`3tier-v3`, `trt-v4`, `fp8-v5`, `dflash3-v6`.

Our lane and his differ in more than the sidecar. We use vLLM nightly 29468dde (his is d05da62e), 1M context
(his v23 is 262K), 16 sequences (his is 8), 8K prefill chunks (his is 2K), and prefix caching. The DETAILS
file lists every difference.

## How it works

Each MoE layer on the GB300 runs these steps:

1. Pack the tokens that route to a sidecar expert. The packed rows are MXFP8 activations, local expert ids
   and route weights. Write them to a shared pinned host buffer and publish a sequence number.
2. Run the HBM bank, then the Grace bank.
3. Wait for the sidecar to acknowledge that sequence, then add its output rows back to their tokens.

The sidecar on the RTX PRO 6000 polls the buffer and runs the rows through b12x's SM120 fused MoE
(`w4a8_mx`: MXFP4 weights, MXFP8 activations). It uses one pre-captured CUDA graph per (layer, row bucket),
and writes the rows back. Every GB300-side step is a device kernel with no host sync, so the whole path is
captured in vLLM's CUDA graphs. A missing sidecar makes the wait time out; the layer then drops that
contribution and increments a counter.

The protocol, buffer layout, and failure handling are described in
[`../docs/sidecar-peer-tier.md`](../docs/sidecar-peer-tier.md). The same design serves DeepSeek-V4.1-Flash in
[`../deepseek-v4.1-flash/m3/`](../deepseek-v4.1-flash/m3/).

## Files

| path | what |
|---|---|
| [`up.sh`](up.sh), [`down.sh`](down.sh) | start the sidecar, then vLLM, with the launcher defaults (v6); stop both |
| [`launch-pro.sh`](launch-pro.sh) | boots vLLM on the GB300. It refuses to start unless the sidecar log says `serving`. |
| [`hook/peer_server_mimo.py`](hook/peer_server_mimo.py) | the RTX PRO 6000 sidecar. It self-tests against a float32 reference before serving. |
| [`hook/peer_tier_mimo.py`](hook/peer_tier_mimo.py) | the shared-buffer protocol and the graph-safe send/wait/add kernels |
| [`hook/hotsplit.py`](hook/hotsplit.py) | per-expert placement across HBM, sidecar, and Grace. Derived from Al-ENGR's `hotsplit.py`; [diff](hook/hotsplit.from-al-engr.diff). |
| [`hook/plan_tiers.py`](hook/plan_tiers.py) | builds the three-tier rowmap from routing counts and GiB budgets |
| [`hook/stage_grace.py`](hook/stage_grace.py), [`hook/trt_banks.py`](hook/trt_banks.py) | v3 Grace→HBM staging; v4 TRT-LLM banks |
| [`hook/mimo_v2*.29468.py`](hook/), [`hook/overlay/`](hook/overlay/), [`backport/`](backport/) | vLLM and FA4 files mounted over nightly 29468dde, with a `.diff` against stock for each |
| [`hook/tests/`](hook/tests/) | standalone GPU tests (staging, fused send, TRT banks, FA4 FP8 over a full-size pool) and the TRT-LLM vs Marlin microbenchmark |
| [`bench/`](bench/) | Al-ENGR's scripts (TTFT bench, knee, BFCL gate, profile), plus our `peer_wait.py`, `prof_summary.py` and `run-suite.sh` |
| [`DETAILS.md`](DETAILS.md) | full configuration, per-version changes, decode profile, microbenchmarks, gotchas |

## Running it

The scripts are verbatim snapshots from the station. They hard-code `/home/jasonc/research/mimo-pro` (this
directory) and the GPU UUIDs; edit them for your host. Start the sidecar first:

```bash
cd hook
CUDA_VISIBLE_DEVICES=<RTX PRO 6000 UUID> PEER_MAX_ROWS=8192 PYTHONPATH=<b12x checkout> \
  python peer_server_mimo.py --rowmap $PWD/rowmap-3tier-v2.json --model <MiMo-V2.6-Pro-RL dir> \
  > ../logs/peer_server_mimo.log 2>&1 &
```

The sidecar ran against [b12x](https://github.com/local-inference-lab/b12x) `a7d7d29b`. Preparing 4,829 experts
takes about 4 minutes. Once it prints `serving`, launch the GB300 side for the qualified v2 configuration:

```bash
ROWMAP=rowmap-3tier-v2.json MOE=marlin STAGE_SLOTS=0 FUSED_SEND=0 KVDTYPE=auto SPEC=none ./launch-pro.sh
```

For an exact v2 reproduction, also mount `hook/mimo_v2.29468.nofp8.py` instead of `hook/mimo_v2.29468.py`
(see Status).

[`up.sh`](up.sh) starts the sidecar, waits for its self-test, then launches vLLM with the script defaults, which
are the unqualified v6 configuration. [`down.sh`](down.sh) stops both.

## Status

This lane was still being iterated when it was snapshotted (2026-09-25, 21:50 EDT). The launcher defaults
(`launch-pro.sh`, used by `up.sh`) are the v6 configuration:

- MOE=trtllm;
- rowmap v3 (115 GiB hot);
- KVDTYPE=fp8, with the FA4 DiffKV overlays;
- STAGE_SLOTS=256;
- SPEC=dflash3.

v5 is v6 with `SPEC=none`. v2–v4 ran with `mimo_v2.29468.nofp8.py`, before an FP8 KV cache worked for MiMo; without the v5 changes vLLM
silently keeps a BF16 KV cache for MiMo. Only v2 has passed a quality gate. The `hook-next/` work-in-progress
directory on the station is not included.
