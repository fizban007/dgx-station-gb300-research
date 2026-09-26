# DeepSeek-V4.1-Flash on one GB300: details

Everything behind [README.md](README.md): every configuration, every run, every measured column, and
where each file came from. The MegaMoE + RTX PRO 6000 lane ("M3") is documented in
[m3/README.md](m3/README.md); this file only indexes the M3 raw runs that were copied here for the
comparison table.

All numbers in the tables below are generated from the raw files by
[`tools/make_results_jsonl.py --tables`](tools/make_results_jsonl.py), which also writes
[`results.jsonl`](results.jsonl). Nothing in this lane is quality-gated: every configuration below is
**unqualified** (no GSM8K, BFCL or oracle check was run on it). All requests completed with 0 errors.

## Common setup

| Item | Value |
| --- | --- |
| Host | gracie: DGX Station with one GB300 (SM103, 152 SMs, 251 GiB HBM, 1,300 W limit), Grace CPU with LPDDR5X (NUMA node 0), RTX PRO 6000 Blackwell Max-Q (SM120, 96 GB, 300 W, PCIe Gen5 x16). NVIDIA driver 595.91.07 (from the `startup_diagnostics` block of every decode JSON). |
| Checkpoint | `deepseek-ai/DeepSeek-V4.1-Flash` revision `dba1be0a40aa45a94ad051997016db3960a90277` (515 GB; MXFP4 routed experts, 384 per layer, top-6, 40 layers; FP8 block-scaled dense layers; MXFP8 activations) |
| Dates | every run in this lane ran on 2026-09-24 |
| Decode | catid recipe via `bench/bench_decode.sh` (llm-inference-bench `0b4185b5`, v0.4.29) |
| Prefill | `bench/bench_prefill.sh` + catid's `bench_prefill.py` (random ids, 4 requests per point) |
| Knee | `bench/knee.sh` (Al-ENGR's short-prose sweep, 192 tokens, mean of 2 runs) |

Timestamps: llm-inference-bench JSON and vLLM log lines use the shell's local zone. The overnight runs
(sections B) were driven from a shell on US Pacific time, the later ones on US Eastern; `prefill.jsonl`
is UTC. `overnight/RESULTS.md` quotes Pacific times. Converted to UTC the overnight runs span
05:30-12:56 UTC and the v20 runs 13:40-15:30 UTC on 2026-09-24.

## A. Al-ENGR v20 reproduced on gracie

Source recipe: [J-M-Recipes/recipes @ `dffd01cc`](https://github.com/J-M-Recipes/recipes/tree/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark)
(MIT), promotion receipt `results/2026-09-21-v20-promotion/README.md`.

Launcher [`upstream-v20/launch-v20.sh`](upstream-v20/launch-v20.sh) (ours, derived from that recipe):

- Engine: image `vllm/vllm-openai:nightly-2671fedfc7ae604761990603fc736c0c4f21de57`
  (vLLM `0.29.1rc1.dev9+g2671fedfc`, FlashInfer `0.6.18.post1`, from the server logs).
- Al-ENGR's v15 pin-hot-experts hook and static rowmap (295 hot / 89 cold experts per layer, cold experts
  streamed from Grace), mounted from their repo unchanged. `PIN_MODE=split`.
- Flags: `--offload-backend uva --cpu-offload-gb 54` (routed expert weights), Engram in Grace RAM,
  `--kv-cache-dtype fp8_ds_mla`, `--max-model-len 1048576`, 24 seqs, 8,192 batched tokens,
  `--gpu-memory-utilization 0.97`, DSpark k=5 with k-schedule `[[1,4,5],[5,24,1]]` (k=5 up to 4 seqs,
  k=1 above), 15 CUDA-graph capture sizes, `--long-prefill-token-threshold 6144`. Prefix caching on
  (vLLM default).
- Differences from Al-ENGR: GB300 selected by UUID; `numactl --membind=0` inside the container
  (`--cpuset-mems` breaks CUDA init on this host, error 802); their pinned FlashInfer autotune set is
  unpublished, so the first boot live-tuned (189 configs saved) and later boots loaded that set.
- Hardware: GB300 + Grace. The `-peer` runs add the RTX PRO 6000 (below).

| Run | Boot | Server log |
| --- | --- | --- |
| `up-v20` | first boot, live FlashInfer autotune | `results/upstream-logs/v20-upstream.log` |
| `up-v20-base2` | reboot, loaded autotune set | `results/upstream-logs/v20-base2.log` |
| `up-v20-peer`, `up-v20-peer2` | reboots with `PIN_PEER=1`: [`upstream-v20/hook-peer/`](upstream-v20/hook-peer/) sends each layer's cold routes (up to 64 tokens) to a sidecar on the RTX PRO 6000 that holds the same 89 cold experts per layer in VRAM and runs them with a Triton GEMV (`overnight/peer/peer_server.py`, logs `overnight/peer/peer_server_upstream*.log`). Order: peer 1, base2, peer 2. | `v20-peer.log`, `v20-peer2.log` |

`v20-upstream-prof.log` and `v20-peer3.log` are extra boots used for profiling; no benchmark files belong
to them.

### A1. catid decode (8,192 in / 1,024 out)

| Run | C | agg tok/s | per-user tok/s p50 | TTFT p50 ms | ITL p50 ms | latency p50 s | accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| up-v20 | 1 | 193.2 | 202.6 | 100 | 4.94 | 5.15 | 2.63 |
| up-v20 | 8 | 583.4 | 74.5 | 225 | 13.43 | 13.98 | 1.71 |
| up-v20 | 16 | 820.5 | 52.6 | 387 | 19.03 | 19.86 | 1.71 |
| up-v20-base2 | 1 | 179.6 | 187.8 | 101 | 5.33 | 5.55 | 2.48 |
| up-v20-base2 | 8 | 567.0 | 72.8 | 250 | 13.74 | 14.44 | 1.70 |
| up-v20-base2 | 16 | 798.5 | 51.0 | 381 | 19.60 | 20.38 | 1.71 |
| up-v20-peer | 1 | 198.9 | 208.9 | 101 | 4.79 | 5.00 | 2.53 |
| up-v20-peer | 8 | 754.3 | 96.4 | 238 | 10.37 | 10.88 | 1.71 |
| up-v20-peer | 16 | 1,117.7 | 73.2 | 391 | 13.67 | 14.47 | 1.71 |
| up-v20-peer2 | 1 | 209.3 | 216.9 | 102 | 4.61 | 4.82 | 2.70 |
| up-v20-peer2 | 8 | 757.9 | 98.0 | 237 | 10.20 | 10.71 | 1.71 |
| up-v20-peer2 | 16 | 1,153.6 | 74.7 | 408 | 13.39 | 14.08 | 1.71 |

Source: `results/runs/<run>/decode/c<C>.json`. C32/C64 were not run (the recipe caps at 24 seqs).

### A2. Prefill and knee

| Run | ISL | C | prefill tok/s | TTFT p50 s | GPU util |
| --- | ---: | ---: | ---: | ---: | ---: |
| up-v20 | 16,384 | 1 | 15,798.8 | 1.036 | 100% |

Source: `results/runs/up-v20/prefill/prefill.jsonl`. Prefix-cache reset returned 404 on this container
(see `bench/README.md`); the four prompts were unseen by the server, so the number is a cold prefill.

| Run | C1 | C2 | C4 | C8 | C12 | C16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| up-v20 | 188.9 | 280.4 | 421.2 | 686.4 | 702.0 | 834.6 |
| up-v20-base2 | 180.8 | 268.9 | 400.5 | 657.3 | 643.3 | 489.0 |
| up-v20-peer | 171.2 | 277.9 | 463.9 | 727.8 | 961.7 | 921.6 |
| up-v20-peer2 | 169.4 | 268.3 | 447.6 | 736.5 | 961.4 | 918.4 |

Aggregate tok/s, mean of two runs. Source: `results/upstream-logs/knee-v20-gracie-live.json`,
`knee-v20-base2.json`, `knee-v20-peer.json`, `knee-v20-peer2.json`. The base2 C12/C16 means are well below
C8 and below the other boots; the per-run values were not stored, so a slow first run cannot be separated
out. The peer tier costs about 6% at knee C1 (171.2 / 169.4 vs 180.8).

## B. Overnight b12x / FlashInfer experiments (vLLM fork + b12x on SM103)

Harness: [`overnight/`](overnight/) (`serve.sh`, `launch.sh`, `stop.sh`, `suite.sh`). Log of the night:
[`overnight/RESULTS.md`](overnight/RESULTS.md) (verbatim except its last line, a link to a private report page, which was removed; see "Narrative docs" below for its known errors).

Common to every run (`overnight/serve.sh`): vLLM from the local-inference-lab/vllm fork (branches below),
b12x SM103 kernels for attention (`--attention-backend B12X`), dense linears (`--linear-backend b12x`),
mHC and Engram; TP1, `--language-model-only`, block size 256, max model length 262,144, CUDA graphs
`FULL_AND_PIECEWISE` up to 64 tokens, Engram tables read from disk (`table_memory: disk`), prefix caching
on, server bound to Grace with `numactl --membind=0`, `VLLM_USE_V2_MODEL_RUNNER=1`. Per-run settings are in
`results/runs/<run>/env.txt` (KSCHED is not captured there; F-ksched's schedule is from RESULTS.md).

| Run | MoE path | Hot experts in HBM | Cold experts | DSpark | Other settings | Source |
| --- | --- | --- | --- | --- | --- | --- |
| b12x-base | b12x residency operator | balanced static, 11,479 of 15,360 | Grace, b12x operator | off | 1,024 batched tokens, KV 10 GB | exp |
| fi-h290 | FlashInfer TRTLLM MXFP4 x MXFP8 | first 290 per layer | Grace, FlashInfer | off | 8,192 batched tokens | exp |
| fi-p12200 | FlashInfer | calibrated, 12,200 total (`profiles/hot-calib1-12200.json`) | Grace, FlashInfer | off | KV 4 GB | exp |
| fi-p12200-ds7 | FlashInfer | same | Grace | k=7, adaptive | 32 seqs | exp |
| fi-p12200-ds3 | FlashInfer | same | Grace | k=3, adaptive | 32 seqs, FI autotune off | exp |
| fi-stitch-ds3 | FlashInfer over VMM-stitched tables | same | Grace pages inside one table | k=3 | `VLLM_EXP_RESIDENCY_STITCH=1` | exp |
| A-fi-tuned | FlashInfer | 305 per layer (`profiles/hot-calib1-u305.json`) | Grace, copy-engine staging at >= 512 tokens | k=3 | FI autotune primed | exp |
| pr-validate | b12x residency operator | balanced static, 11,584 of 15,360 | Grace, b12x operator | off | 64 seqs, KV 8 GB | PR stack |
| C-peer | as A | as A | RTX PRO 6000 sidecar (`overnight/peer/peer_server.py`, Triton) | k=3 | `VLLM_EXP_COLD_KERNEL=peer` | exp |
| D-peer-a16 | as C | as A | 6000 | k=3 | + b12x A16 decode linears, 64 seqs | exp |
| E-rebased | as D | as A | 6000 | k=3 | 32 seqs; exp commits on current karmic dev | exp2 |
| F-ksched | as E | as A | 6000 | k=5 up to 4 seqs, k=1 above | Al-ENGR's k-schedule | exp2 |

`E-prof` is a profiling boot of the E configuration without the peer tier (only `prof/profiler_out_0.txt`
kept; traces excluded). Aborted launches kept for their console logs: `fi-p12200.autotune`,
`fi-p12200-ds7.autotune` (FlashInfer / b12x autotune time), `fi-p12200-ds7.serial`, `.serial2`
(`B12X_COMPILE_WORKERS=0` startup), `fi-stitch-ds3.slowtune`, `b12x-base.console` (engine init failure).

### B1. catid decode (8,192 in / 1,024 out)

| Run | C | agg tok/s | per-user tok/s p50 | TTFT p50 ms | ITL p50 ms | latency p50 s | accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b12x-base | 1 | 55.0 | 58.1 | 991 | 17.20 | 18.59 | - |
| b12x-base | 8 | 131.0 | 18.2 | 6,398 | 54.94 | 62.70 | - |
| fi-h290 | 1 | 61.9 | 63.5 | 398 | 15.74 | 16.50 | - |
| fi-h290 | 8 | 200.0 | 25.5 | 828 | 39.18 | 40.91 | - |
| fi-p12200 | 1 | 74.4 | 76.9 | 410 | 13.00 | 13.72 | - |
| fi-p12200 | 8 | 338.0 | 43.7 | 818 | 22.90 | 24.25 | - |
| fi-p12200-ds7 | 1 | 116.5 | 125.9 | 407 | 7.94 | 8.53 | 2.56 |
| fi-p12200-ds7 | 8 | 294.6 | 37.9 | 523 | 26.36 | 27.49 | 2.46 |
| fi-p12200-ds7 | 16 | 358.2 | 23.3 | 650 | 42.96 | 44.86 | 2.47 |
| fi-p12200-ds3 | 1 | 125.3 | 131.8 | 399 | 7.59 | 8.16 | 2.43 |
| fi-p12200-ds3 | 8 | 328.7 | 42.8 | 491 | 23.35 | 24.40 | 2.29 |
| fi-p12200-ds3 | 16 | 412.3 | 26.8 | 845 | 37.32 | 38.79 | 2.31 |
| fi-stitch-ds3 | 1 | 95.7 | 102.3 | 434 | 9.78 | 10.44 | 2.37 |
| fi-stitch-ds3 | 8 | 222.0 | 28.3 | 584 | 35.29 | 36.71 | 2.31 |
| fi-stitch-ds3 | 16 | 276.3 | 17.8 | 739 | 56.21 | 58.46 | 2.31 |
| A-fi-tuned | 1 | 117.9 | 135.2 | 398 | 7.40 | 7.98 | 2.46 |
| A-fi-tuned | 8 | 335.3 | 43.6 | 488 | 22.95 | 24.07 | 2.30 |
| A-fi-tuned | 16 | 420.9 | 27.1 | 551 | 36.94 | 38.35 | 2.32 |
| pr-validate | 1 | 56.5 | 59.7 | 938 | 16.76 | 18.08 | - |
| pr-validate | 8 | 136.3 | 18.9 | 6,183 | 52.76 | 60.11 | - |
| C-peer | 1 | 126.1 | 133.5 | 400 | 7.49 | 8.06 | 2.37 |
| C-peer | 8 | 417.4 | 54.8 | 466 | 18.24 | 19.14 | 2.32 |
| C-peer | 16 | 520.0 | 33.9 | 514 | 29.55 | 31.13 | 2.32 |
| D-peer-a16 | 1 | 125.9 | 138.6 | 406 | 7.22 | 7.78 | 2.41 |
| D-peer-a16 | 8 | 419.3 | 53.7 | 456 | 18.62 | 19.50 | 2.32 |
| D-peer-a16 | 16 | 517.1 | 33.0 | 529 | 30.32 | 31.62 | 2.32 |
| D-peer-a16 | 32 | 520.7 | 16.6 | 932 | 60.34 | 62.69 | 2.26 |
| D-peer-a16 | 64 | 599.6 | 9.6 | 1,294 | 104.43 | 108.29 | 2.35 |
| E-rebased | 1 | 140.0 | 147.4 | 377 | 6.79 | 7.32 | 2.40 |
| E-rebased | 8 | 464.2 | 59.9 | 396 | 16.69 | 17.59 | 2.30 |
| F-ksched | 1 | 141.0 | 148.5 | 375 | 6.74 | 7.27 | 2.61 |
| F-ksched | 8 | 382.4 | 51.3 | 722 | 19.49 | 20.50 | 1.73 |
| F-ksched | 16 | 502.0 | 33.0 | 737 | 30.28 | 31.68 | 1.73 |
| F-ksched | 32 | 590.5 | 18.9 | 893 | 52.96 | 55.09 | 1.72 |

Source: `results/runs/<run>/decode/c<C>.json`. "-" = no speculative decoding. Concurrencies not listed
were not run. No single run has the best value in every column: C1 is best in F-ksched, C8 in
E-rebased, C16 in C-peer, C32 in F-ksched, C64 was only run on D-peer-a16.

### B2. Prefill

| Run | ISL | C | prefill tok/s | TTFT p50 s | GPU util |
| --- | ---: | ---: | ---: | ---: | ---: |
| b12x-base | 16,384 | 1 | 346.5 | 47.294 | 94% |
| A-fi-tuned | 16,384 | 1 | 4,952.7 | 3.313 | 43% |
| A-fi-tuned | 32,768 | 1 | 4,844.4 | 6.752 | 43% |
| A-fi-tuned | 65,536 | 1 | 4,550.3 | 14.395 | 44% |
| A-fi-tuned | 131,072 | 1 | 4,235.7 | 30.943 | 47% |
| pr-validate | 16,384 | 1 | 370.5 | 44.241 | 100% |
| E-rebased | 16,384 | 1 | 1,441.3 | 11.367 | 100% |
| E-rebased | 65,536 | 1 | 1,414.4 | 46.333 | 100% |
| E-rebased | 131,072 | 1 | 1,382.2 | 94.829 | 100% |

Source: `results/runs/<run>/prefill/prefill.jsonl`. A-fi-tuned's GPU sits idle more than half the time
(disk-mode Engram reads). E-rebased shows 100% utilization at about 254 W mean (`gpu_power_mean_w`)
while running 3.4x slower than A: the GPU is waiting, not computing. C-peer, D, F: prefill not run.

### B3. Kernel microbenchmarks

FlashInfer TRTLLM MXFP4 x MXFP8 routed MoE, one layer at V4.1 geometry (H 5120, I 2304, 384 experts,
top-6), all experts in HBM (`overnight/fi_moe_bench.py`, output `overnight/micro/fi_moe.json`):

| Tokens | one call, all 384 experts (us) | hot 295 + cold 89 as two calls (us) |
| ---: | ---: | ---: |
| 1 | 36.8 | 45.0 |
| 8 | 149.8 | 164.1 |
| 64 | 742.2 | 762.9 |
| 256 | 1,128.8 | 1,181.6 |
| 1,024 | 1,701.3 | 1,472.5 |
| 4,096 | 2,452.9 | 1,881.1 |
| 16,384 | 5,370.7 | 5,373.3 |

Other raw microbenchmarks, not tabulated here: cold experts in cacheable Grace memory
(`micro/fi_moe_coldhost.json`), VMM-stitched tables all-hot and with Grace-backed experts
(`micro/fi_moe_stitch_allhot.json`, `micro/fi_moe_stitch_cold2.json`), b12x A16 split-K vs MXFP8 dense
decode GEMMs (`micro/a16_bench.json`), and the peer-side logs of the GB300 <-> 6000 round-trip
microbenchmark (`peer/peer_*.log`; the GB300-side timings quoted in RESULTS.md were not saved).

## C. Is b12x worth it on GB300? (for this model and quant mix)

[`findings.md`](findings.md) is the write-up (copied verbatim). Scope: DeepSeek-V4.1-Flash's MXFP4
experts / FP8 block-scaled dense / MXFP8 activations only. Its tables come from the runs above, from
the mHC microbenchmark below, from profiler traces that were not kept (each trace is 9-11 MB; the
bucketing script is [`upstream-v20/components.py`](upstream-v20/components.py) and the load generator
[`upstream-v20/profile_load.py`](upstream-v20/profile_load.py)), and from the M3 lane.

mHC sublayer boundary, identical shapes, CUDA-graph replay on the GB300 (`mhc/*.py`, `mhc/*.jsonl`):

| Tokens | upstream TileLang + DeepGEMM (us) | b12x tuned (us) | b12x default config (us) |
| ---: | ---: | ---: | ---: |
| 1 | 12.92 | 8.62 | 8.61 |
| 4 | 13.12 | 9.32 | 9.39 |
| 8 | 12.90 | 10.81 | 10.74 |
| 16 | 13.04 | 14.09 | 14.03 |
| 32 | 13.07 | 18.02 | 21.50 |
| 64 | 13.28 | 18.85 | 35.18 |

Source: `mhc/upstream.jsonl`, `mhc/b12x-tuned.jsonl`, `mhc/b12x.jsonl`.

## M3 raw runs copied here

Copied from `/home/jasonc/ds41f-exp/runs` and `/home/jasonc/research/upstream/logs` so the README
comparison table can cite them; described in [m3/README.md](m3/README.md), not in `results.jsonl`.

| Path | What |
| --- | --- |
| `results/runs/m3/` | M3 v1 (cold experts streamed from Grace): decode C1/C8/C16, prefill 16K |
| `results/runs/m3-repro/` | M3 v1 prefill 8K/32K/64K |
| `results/runs/m3v2/` | peer v2 (6000 via b12x), static map; decode and prefill 8K-128K. Before the 6000 FP8-decode fix: speed valid, quality not. The last `prefill.jsonl` line (16K, 119,827 tok/s at 0% GPU util) is a prefix-cache hit and invalid. |
| `results/runs/m3v2-c1fix/` | peer v2 after the C1 graph-copy fix, decode C1 only (before the FP8 fix) |
| `results/runs/m3v2-h285/` | peer v2 with 285 hot experts per layer (map described in m3/README.md) |
| `results/runs/mix-v1/` | peer v2, balanced 285-hot map `rowmap-mix-v1` |
| `results/runs/phaseC-*` | DSpark schedule sweep on mix-v1 after the sidecar retune; C4 (`phaseC-C4-k2-to-8`) is the chosen default |
| `results/upstream-logs/knee-m3.json`, `knee-m3-warm.json`, `knee-m3v2.json`, `knee-m3v2-h285.json`, `knee-mix-v1.json` | knee sweeps of the M3 boots |

## Provenance

| Code | Branch | Head commit | On origin? |
| --- | --- | --- | --- |
| vLLM fork, [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) (Apache-2.0) | `exp/ds41f-gb300` | `89ba1db040f72843b8a59d42c549451b8f4ed254` "exp: RTX PRO 6000 peer tier for cold experts (VLLM_EXP_COLD_KERNEL=peer)" | no (local only) |
| same | `exp2/ds41f-gb300` | `77c7770b17c18e82ae2d46dba0692fde57159b91` (same six exp commits cherry-picked onto `feat/b12x-sm103-expert-residency` `45a851dd`, which is based on karmic dev `b99d0d43`) | no (local only) |
| same | `feat/b12x-sm103-expert-residency` (was vllm#888, closed) | `45a851dda3eda329879a494773b90a98b74d5457` | yes |
| [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) | `exp/ds41f-gb300` | `8fdb5635edeef42f4a78810b9ba1b0350188923b` "exp: rank static placement by routing counts from B12X_EXP_ROUTING_COUNTS" | no (local only) |
| same | `sm103/residency` (was b12x#426, closed) | `0aba33c0029a1ef382306e0e9320eb23d418032a` | yes |
| same | `sm103/kernels` (was b12x#425, closed) | `452094fb7f6a159853fafd4f579a02860fb2a025` | yes |

The runs used the working tree of these branches while the experiments were being committed, so a run's
exact source is the nearest earlier commit plus uncommitted edits. vLLM `exp/ds41f-gb300` commits (UTC,
2026-09-24): `825dc173` 04:13, `6d50337d` 04:44, `06a8d446` 05:34, `d8db977f` 07:30, `907994ec` 08:56,
`89ba1db0` 11:39 (the peer-tier code used by C-peer at 11:02 UTC was committed afterwards). The server logs
do not record which tree was imported; E-rebased and F-ksched are attributed to `exp2` and b12x
`sm103/residency` by RESULTS.md only.

Upstream image: `vllm/vllm-openai:nightly-2671fedfc7ae604761990603fc736c0c4f21de57`. Benchmark client:
see [`../bench/README.md`](../bench/README.md).

## Derived files and licenses

- `upstream-v20/launch-v20.sh`, `launch-v20-prof.sh`: ours, reproducing the flags of Al-ENGR's v20 recipe
  (J-M-Recipes/recipes, MIT License, Copyright (c) 2026 J&M Recipes).
- `upstream-v20/hook-peer/pin_hot_experts_hook.py`: modified copy of Al-ENGR's v15
  `pin_hot_experts_hook.py` (MIT, same copyright); adds the `PIN_PEER=1` path. Full change:
  `pin_hot_experts_hook.diff`. The hook also needs Al-ENGR's unmodified `sitecustomize.py` and
  `rowmap-static-v1.json`, which are not copied: take them from
  [`results/2026-09-17-e2b-pin-hot-experts-v15/hook/`](https://github.com/J-M-Recipes/recipes/tree/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-17-e2b-pin-hot-experts-v15/hook).
- `upstream-v20/hook-peer/peer_tier.py`: ours, carries the vLLM Apache-2.0 SPDX header (written for the
  vLLM fork's peer tier).
- `bench/knee.sh`: modified copy of Al-ENGR's `knee.sh` (MIT); see `bench/README.md`.
- Everything else in this lane is ours.

## Narrative docs: known errors

`overnight/RESULTS.md` (minus one private link) and `findings.md` are kept verbatim. Checked against the raw files and the external
sources, they contain these errors:

- RESULTS.md, "Al-ENGR v20: 181 C1, 657 C8, 969 C16, 22-23K prefill". Al-ENGR's v20 receipt reports knee
  C1 180.9 / 180.3, C8 670 / 661, C16 979 / 965 (two boots). 657 is their C8 in a separate capture-size A/B
  (720 without the capture list, 657 with it); 969 does not appear; the 22K prefill is their v15 figure.
- RESULTS.md targets table: "16K 35.9K, 128K 56.0K" are catid's vLLM PP2 AR prefill numbers; the PP2 +
  DSpark row it sits in measured 35.2K and 55.2K.
- findings.md, section 5 table, "catid C1 per user 248.5": 248.5 is catid's C1 aggregate; his per-user
  C1 is 252.9.
- findings.md, "knee C12 (2nd run) 822.6 / 975.2 / 963.6" and "M3 v1 knee C12 1,032": second-run values
  that are not in the knee JSONs (which store the two-run means 643.3 / 961.7 / 961.4 and 665.3).
- findings.md, section 1, "Best b12x stack" combines four runs (C1 F-ksched, C8 E-rebased, C16 C-peer,
  prefill A-fi-tuned without the 6000). RESULTS.md's "b12x residency all-HBM" microbenchmark column and
  findings.md's per-component table have no raw file here (the traces were over 1 MB).
