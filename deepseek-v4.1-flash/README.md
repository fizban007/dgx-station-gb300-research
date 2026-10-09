# DeepSeek-V4.1-Flash on one DGX Station GB300

Serving DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a`, 515 GB, 289 GB of it MXFP4
routed experts) on one GB300 (251 GiB HBM) plus Grace memory and, in some configurations, the RTX PRO 6000
Blackwell Max-Q in the same box. Runs: 2026-09-24, with M3 updates through 2026-10-03; host gracie. Full tables, flags and provenance: [DETAILS.md](DETAILS.md). Machine-readable numbers:
[results.jsonl](results.jsonl).

## What we found

- **Best single-station result: M3, MegaMoE hot experts plus the RTX PRO 6000 sidecar** ([m3/](m3/README.md)).
  As of 2026-09-28 it reaches 257.9–266.3 tok/s per user at C1 and 1,046.3 / 1,628.6 aggregate at C8 / C16, about
  1.8× / 2.0× v20. Prefill is 44.5K tok/s at 16K on random token ids and ~60K on real text. GSM8K-200 is 98.0%.
  On 2026-09-24 it stood at 219.1 per user at C1 and 926.4 / 1,333.4 at C8 / C16, with 24K prefill.
  Since 2026-10-03 it shares the GB300 with a video model (MiniMax-H3). It runs 265 hot experts, NVFP4 Engram
  tables and NVFP4 KV, keeps 3.42M KV tokens, and decodes reasoning traffic at 304 / 1,071 / 1,470 tok/s at
  C1 / C8 / C16. Its GPQA-Diamond is 90–92% at T=1. See [m3/README.md](m3/README.md).
- **Al-ENGR's v20 recipe reproduces on our box** (upstream vLLM nightly + a hook that keeps 295 hot experts
  per layer in HBM and streams 89 cold ones from Grace): 202.6 tok/s per user at C1, 820.5 aggregate at C16,
  15,799 tok/s cold prefill at 16K.
- **Our b12x-based stack never caught up.** It was a vLLM fork with b12x SM103 kernels, FlashInfer hot experts
  and cold experts on Grace or on the RTX PRO 6000. Its best result per column came from different runs:
  148.5 tok/s per user at C1, 464.2 aggregate at C8, 520.0 at C16 and 4,953 tok/s prefill. Even those best
  values are below v20.
- **Scope:** these b12x-on-GB300 conclusions hold for DeepSeek-V4.1-Flash's MXFP4-expert / FP8-dense / MXFP8
  mix only. Other models and quant types (NVFP4, W4A16, ...) were not tested and remain open.
- **The RTX PRO 6000 as a cold-expert tier helps upstream too:** v20 plus the 6000 reaches 757.9 at C8 and
  1,153.6 at C16 (+34% and +44% over the loaded-autotune v20 boot, `up-v20-base2`), but loses about 6% on the short-prompt knee
  at C1.
- Only M3 configurations passed a quality gate (GSM8K-200; see [m3/DETAILS.md](m3/DETAILS.md#quality-gsm8k-200-thinking-off)).
  Everything else here is **unqualified**.

## Headline comparison

| Configuration | HW | catid C1 per user | C8 agg | C16 agg | C32 agg | Prefill 16K | knee C1 | knee C8 | knee C16 | GSM8K-200 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Al-ENGR v20, reproduced, first boot (`up-v20`) | 1x GB300 | 202.6 | 583.4 | 820.5 | not run | 15,798.8 | 188.9 | 686.4 | 834.6 | not run |
| same, loaded autotune (`up-v20-base2`) | 1x GB300 | 187.8 | 567.0 | 798.5 | not run | not run | 180.8 | 657.3 | 489.0 | not run |
| v20 + 6000 cold tier (`up-v20-peer2`) | GB300 + 6000 | 216.9 | 757.9 | 1,153.6 | not run | not run | 169.4 | 736.5 | 918.4 | not run |
| b12x stack, Grace cold tier (`A-fi-tuned`) | 1x GB300 | 135.2 | 335.3 | 420.9 | not run | 4,952.7 | not run | not run | not run | not run |
| b12x stack + 6000 (`C-peer`) | GB300 + 6000 | 133.5 | 417.4 | 520.0 | not run | not run | not run | not run | not run | not run |
| b12x stack + 6000, rebased (`E-rebased`) | GB300 + 6000 | 147.4 | 464.2 | not run | not run | 1,441.3 | not run | not run | not run | not run |
| b12x stack + 6000, Al-ENGR k-schedule (`F-ksched`) | GB300 + 6000 | 148.5 | 382.4 | 502.0 | 590.5 | not run | not run | not run | not run | not run |
| M3 mix-v1 (`mix-v1`) | GB300 + 6000 | 211.6 | 776.0 | 1,133.6 | not run | 23,968.7 | 187.7 | 774.4 | 1,196.2 | 98.0% (196/200) |
| M3 final, DSpark k 5/2/1 (`phaseC-C4-k2-to-8`) | GB300 + 6000 | 219.1 | 926.4 | 1,333.4 | not run | not run | not run | not run | not run | not run |
| M3 2026-09-28, k 5/3/3, probabilistic drafts (`k-pd-ll-k533`) | GB300 + 6000 | not run | 1,046.3 | 1,603.9 | not run | not run | 208.9 | 904.4 | 1,423.2 | carries over¹ |
| M3 2026-09-28 current, + vllm#58132 (`replay58132`) | GB300 + 6000 | 257.9 | not run | 1,628.6 | not run | 44,535.2 | not run | not run | not run | 98.0% (196/200) |
| External: Al-ENGR v20, published | 1x GB300 | - | - | - | - | ~22K (v15) | 180.9 | 670 | 979 | BFCL-gated by Al-ENGR |
| External: catid vLLM PP2 + DSpark | **2x GB300** | 252.9 | 1,024.1 | 1,677.7 | 2,346.4 | 35,214.4 | - | - | - | catid "accepted" |

Units: tok/s. "catid" columns use catid's decode recipe (8,192 input / 1,024 forced output tokens,
temperature 0, C warm-ups then 5xC requests); C1 is per-user p50, C8-C32 are aggregate. Prefill is one
16K request of random token ids, aggregate prompt tok/s. Knee is Al-ENGR's short-prose sweep (192 output
tokens), aggregate, mean of two runs. Not all runs covered all columns; "not run" means not measured, not
zero. catid's DeepSeek-V4.1 numbers are from **two** DGX Stations (PP2 over RDMA); he did not run one
station, so there is no single-GPU catid bar.

Sources: `results/runs/<run>/decode/c<C>.json`, `results/runs/<run>/prefill/prefill.jsonl`,
`results/upstream-logs/knee-v20-gracie-live.json`, `knee-v20-base2.json`, `knee-v20-peer2.json`,
`knee-mix-v1.json`; M3 prefill and GSM8K: `m3/results/logs/prefill-mix.jsonl`,
`m3/results/logs/gsm8k-mix-v1.json`. External: Al-ENGR's
[v20 promotion receipt](https://github.com/J-M-Recipes/recipes/blob/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-21-v20-promotion/README.md)
(boot v20a; v20b: 180.3 / 661 / 965) and [page](https://al-engr.com/gb300-deepseek-flash-41-testing.html)
(prefill "22K tok/s from 26K to 207K tokens" on v15); catid's
[`data/throughput.csv`](https://github.com/catid/dgx_station_benchmarks/blob/ff8a496e5e027bbc462f81643361ef5516072a68/deepseek-v4.1-flash/data/throughput.csv)
and [`data/prefill.csv`](https://github.com/catid/dgx_station_benchmarks/blob/ff8a496e5e027bbc462f81643361ef5516072a68/deepseek-v4.1-flash/data/prefill.csv).
The `up-v20-base2` knee C16 mean is anomalous (C16 below C8); see DETAILS.md.
¹ `k-pd-ll-k533` differs from the GSM8K-gated `prod-20260928b` only in a bit-identical hook path; see
[m3/DETAILS.md](m3/DETAILS.md#2026-09-28-evening-probabilistic-drafting-small-m-gemms-fused-send-v2). The 2026-09-28
sources are under `results/runs/k-pd-ll-k533/`, `results/runs/replay58132/`,
`results/upstream-logs/knee-k-pd-ll-k533.json` and `m3/results/logs/gsm8k-replay58132.json`.

## What was tried

| Phase | Engine | What changed | Tables |
| --- | --- | --- | --- |
| A. Al-ENGR v20 reproduced | `vllm/vllm-openai:nightly-2671fedfc...` (vLLM 0.29.1rc1.dev9) + Al-ENGR v15 hook; flags in [`upstream-v20/launch-v20.sh`](upstream-v20/launch-v20.sh) | live vs loaded FlashInfer autotune; + RTX PRO 6000 cold tier via [`upstream-v20/hook-peer/`](upstream-v20/hook-peer/) | [DETAILS A](DETAILS.md#a-al-engr-v20-reproduced-on-gracie) |
| B. Overnight b12x / FlashInfer | local-inference-lab/vllm fork `exp/ds41f-gb300` and `exp2/ds41f-gb300` + b12x `exp/ds41f-gb300`; launcher [`overnight/serve.sh`](overnight/serve.sh) | b12x residency operator vs FlashInfer TRTLLM; positional vs calibrated hot sets; DSpark k=3/5/7; VMM-stitched tables; 6000 peer tier; A16 decode linears; rebase; k-schedule | [DETAILS B](DETAILS.md#b-overnight-b12x--flashinfer-experiments-vllm-fork--b12x-on-sm103) |
| C. Is b12x worth it on GB300? | same as A and B | per-kernel comparison, mHC microbenchmark | [findings.md](findings.md), [DETAILS C](DETAILS.md#c-is-b12x-worth-it-on-gb300-for-this-model-and-quant-mix) |

Key lessons from phase B (details and raw files in DETAILS.md and [`overnight/RESULTS.md`](overnight/RESULTS.md)):
swapping the b12x residency operator for FlashInfer's routed kernel with the same kind of static placement
raised C8 from 131.0 to 200.0 (`b12x-base` vs `fi-h290`); cold experts read from Grace inside the FlashInfer
call cost 281.0 us per layer at one token against 45.0 us with every expert in HBM
(`overnight/micro/fi_moe_coldhost.json`, `fi_moe.json`); the 6000 peer tier lifted C8 from 335.3 to 417.4 but
not C1 (135.2 vs 133.5, `A-fi-tuned` vs `C-peer`); the rebased stack regressed prefill from 4,952.7 to
1,441.3 tok/s.

## M3: MegaMoE hot tier + RTX PRO 6000 sidecar

M3 keeps the hottest experts per layer in HBM under DeepGEMM MegaMoE (285 until 2026-10-02, 265 since) and runs
the rest on the RTX PRO 6000 with b12x's SM120 fused MoE, which is the one place b12x paid off in this study. It is documented, with
its own results and quality checks, in [m3/README.md](m3/README.md).

Since 2026-10-07 M3 can keep the routed experts' MXFP4 block scales compressed (lossless) on both GPUs: 8.6 GiB
less HBM on the GB300 and 4.5 GiB less VRAM on the RTX PRO 6000, which pays for a 254 hot / 130 cold split. Code,
patches against `m3/`, tests and results: [sf-compress/README.md](sf-compress/README.md).

## Layout

| Path | Contents | Copied from (on gracie) |
| --- | --- | --- |
| `upstream-v20/` | v20 launchers, peer-tier hook, profiler bucketing (`components.py`, `profile_load.py`) | `/home/jasonc/research/upstream` |
| `overnight/` | launch/benchmark harness, calibration and placement scripts, microbenchmarks (`micro/`), routing counts and hot sets (`profiles/`), v1 peer sidecar (`peer/`), `RESULTS.md` night log | `/home/jasonc/ds41f-exp` |
| `mhc/` | mHC boundary microbenchmark scripts and results | `/home/jasonc/research/mhc` |
| `findings.md` | "Is there anything in b12x worth doing" write-up | `/home/jasonc/research/FINDINGS.md` |
| `results/runs/` | one directory per server run: `decode/c<C>.json` + logs, `prefill/`, `env.txt`, `console.log` | `/home/jasonc/ds41f-exp/runs` |
| `results/upstream-logs/` | knee JSONs and v20 container logs | `/home/jasonc/research/upstream/logs` |
| `results/upstream-prof/` | vLLM profiler summary table of the v20 profiling boot (traces not kept) | `/home/jasonc/research/upstream/prof` |
| `tools/make_results_jsonl.py` | regenerates `results.jsonl` and the DETAILS tables | - |
| `m3/` | the M3 lane | - |
| `sf-compress/` | compressed MoE weight scales for M3 (codec, patches against `m3/`, tests, results) | - |

Benchmark scripts are shared across lanes: [`../bench/`](../bench/README.md).

Not included: model weights, caches, profiler traces (`*.pt.trace.json.gz`, `prof/*.json.gz`, 2-11 MB each),
failed-launch dumps (`*.fail*`, `*.oom`), the v2 sidecar (in `m3/`), Qwen and MiMo runs (other lanes).

## Licenses and derivations

`upstream-v20/launch-v20.sh` reproduces Al-ENGR's v20 recipe and `upstream-v20/hook-peer/pin_hot_experts_hook.py`
is a modified copy of their v15 hook, both from [J-M-Recipes/recipes](https://github.com/J-M-Recipes/recipes)
at `dffd01cc` (MIT License, Copyright (c) 2026 J&M Recipes); the change is in `pin_hot_experts_hook.diff`.
`upstream-v20/hook-peer/peer_tier.py` carries vLLM's Apache-2.0 header. Nothing from catid's repository is
copied (no license); it is linked.
`sf-compress/phase2/*.diff` patch DeepGEMM's `sm100_fp8_fp4_mega_moe.cuh` as vendored in vLLM (MIT, DeepSeek);
the header itself is not copied. `sf-compress/notes/csf-codec.md` quotes short excerpts of b12x (Apache-2.0).
