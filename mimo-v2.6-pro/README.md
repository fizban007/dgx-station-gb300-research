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
- **FlashInfer TRT-LLM banks (v4) look like the next step, but are unqualified.** Every expert GEMM reads
  HBM, and a C1 request reaches 46 tok/s decode and 5.0–5.8K tok/s prefill. No quality gate has been run on v4.

## Results

All measurements were taken on 2026-09-25 with Al-ENGR's scripts. The TTFT bench sends one request
(prefill = prompt / TTFT, then decode after the first token). The knee sends 192-token prose completions at
C1–C16, and its numbers are aggregate tok/s.

| | Al-ENGR v23 (HBM + Grace, Marlin) | **v2** (+ 6000 sidecar) | v3 (+ Grace staging) | v4 (TRT-LLM banks) |
|---|--:|--:|--:|--:|
| Prefill, 2.9K / 11.5K / 45.7K prompt (tok/s) | 1,064¹ / 1,376 / 1,244¹ | 2,335 / 2,812 / 2,775 | 2,388 / 2,875 / 2,839 | 2,980 / 5,032 / 5,775 |
| Decode after TTFT, same prompts (tok/s) | 33.2¹ / 37.4 / 32.7¹ | 39.6 / 39.4 / 38.6 | 40.4 / 39.9 / 39.5 | 46.3 / 46.1 / 45.1 |
| Knee aggregate C1 / C4 / C8 / C16 (tok/s) | 30.7 / 50.5 / 63.7 / 63.0² | 32.3 / 61.2 / 79.7 / 102.2³ | 32.0 / 60.6 / 78.5 / 98.2 | not run |
| GSM8K-200 | — | **98.5%** (197/200) | not rerun | not run |
| BFCL dev (600 cases) | 93.83% | **92.83%** (557/600) | not rerun | not run |
| Status | verified (his protocol) | **qualified** | component tests only | **unqualified** |

¹ His v24 receipt (1M-context variant, 110 GiB hot). He publishes v23 numbers for the 11.5K prompt only.
² His v22 receipt, which has the same budget as v23. His lane is capped at 8 sequences, so C16 does not scale.
³ This v2 knee run is from session notes; its raw log was not kept (see [DETAILS](DETAILS.md#knee-v2)).

Sources: [`results/`](results/), [`results/logs/`](results/logs/), his
[recipe README and receipts](https://github.com/J-M-Recipes/recipes/tree/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/dgx-station-gb300/mimo-v2.6-pro-vllm-uva-hotsplit).
Every number, with its source file, is in [`results.jsonl`](results.jsonl).

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
| [`launch-pro.sh`](launch-pro.sh) | boots vLLM on the GB300. It refuses to start unless the sidecar log says `serving`. |
| [`hook/peer_server_mimo.py`](hook/peer_server_mimo.py) | the RTX PRO 6000 sidecar. It self-tests against a float32 reference before serving. |
| [`hook/peer_tier_mimo.py`](hook/peer_tier_mimo.py) | the shared-buffer protocol and the graph-safe send/wait/add kernels |
| [`hook/hotsplit.py`](hook/hotsplit.py) | per-expert placement across HBM, sidecar, and Grace. Derived from Al-ENGR's `hotsplit.py`; [diff](hook/hotsplit.from-al-engr.diff). |
| [`hook/plan_tiers.py`](hook/plan_tiers.py) | builds the three-tier rowmap from routing counts and GiB budgets |
| [`hook/stage_grace.py`](hook/stage_grace.py), [`hook/trt_banks.py`](hook/trt_banks.py) | v3 Grace→HBM staging; v4 TRT-LLM banks |
| [`hook/mimo_v2*.29468.py`](hook/), [`hook/overlay/`](hook/overlay/), [`backport/`](backport/) | vLLM files mounted over nightly 29468dde, with a `.diff` against stock for each |
| [`hook/tests/`](hook/tests/) | standalone GPU tests (staging, fused send, TRT banks) and the TRT-LLM vs Marlin microbenchmark |
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
ROWMAP=rowmap-3tier-v2.json MOE=marlin STAGE_SLOTS=0 FUSED_SEND=0 KVDTYPE=auto ./launch-pro.sh
```

For an exact v2 reproduction, also mount `hook/mimo_v2.29468.nofp8.py` instead of `hook/mimo_v2.29468.py`
(see Status). The script's defaults are the newer, unqualified settings.

## Status

This lane was being actively iterated when it was snapshotted (2026-09-25, 20:00 EDT). Four launcher settings
postdate every measurement above:

- MOE=trtllm;
- rowmap v3 (115 GiB hot);
- KVDTYPE=fp8;
- the model-file and FA4 DiffKV overlay changes that make an FP8 KV cache work for MiMo.

v2–v4 ran with `mimo_v2.29468.nofp8.py`, before an FP8 KV cache worked for MiMo; vLLM silently keeps BF16
KV for MiMo without these changes. The scripts are published as they were, and no result here covers the newer
combination.
