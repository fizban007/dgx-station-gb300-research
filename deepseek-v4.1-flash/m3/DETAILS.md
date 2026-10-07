# M3 lane: details

This file backs the [README](README.md). Every number is in [`results.jsonl`](results.jsonl) with its source
file; regenerate that file with [`tools/make_results_jsonl.py`](tools/make_results_jsonl.py). Numbers marked
*(notes)* come from session notes; their raw output was not kept. Runs are from 2026-09-24 to 2026-10-03, in EDT.
- The 2026-09-28 work (image upgrade, hook fusion, DSpark retunes, dense-GEMM and prefill changes) starts at
  [2026-09-28: image upgrade and decode profile](#2026-09-28-image-upgrade-and-decode-profile).
- The 2026-09-29 work (32 sequences, pinned FlashInfer tactics) is at
  [2026-09-29: 32 sequences and pinned FlashInfer tactics](#2026-09-29-32-sequences-and-pinned-flashinfer-tactics).
- The 2026-10-02/03 work (making room for MiniMax-H3, NVFP4 Engram, NVFP4 KV, GPQA) is at
  [2026-10-02/03: sharing the GB300 with MiniMax-H3](#2026-10-0203-sharing-the-gb300-with-minimax-h3).

## Configuration (current, 2026-10-03)

What [`swap-to-ds41-h3.sh`](swap-to-ds41-h3.sh) boots: the 2026-09-29 configuration below, plus:
- `ROWMAP=rowmap-mix-v1-h265.json`: 265 hot experts per layer; the other 119 are on the sidecar.
- The NVFP4 Engram overlay and checkpoint mounts ([`engram-nvfp4/lane-mounts.sh`](engram-nvfp4/lane-mounts.sh)),
  which include the vllm#58132 mounts.
- `--kv-cache-dtype nvfp4_ds_mla`.
- `GPU_UTIL=0.885`.

The KV cache is 4.64 GiB, or 3,423,027 tokens (3.26 requests at the 1,048,576-token max length). The lane
shares the GB300 with MiniMax-H3 in vllm-omni, started by
[`tools/coexist/serve-h3-gracie.sh`](tools/coexist/serve-h3-gracie.sh) with `H3_MODE=combined OFFLOAD=legacy`.

## Configuration (2026-09-29)

[`swap-to-m3v2.sh`](swap-to-m3v2.sh) and [`launch-m3.sh`](launch-m3.sh) still boot this by default. It is the
2026-09-28 configuration below with 32 sequences (was 24) and FlashInfer tactics pinned by `MEGA_FI_TUNE_FILE`
([`hook/fi-tune/fiset-a964-9a5ab530.json`](hook/fi-tune/fiset-a964-9a5ab530.json)). It still uses 285 hot
experts, FP8 Engram, `fp8_ds_mla` and `--gpu-memory-utilization 0.95`: 3,527,818 KV tokens, and no room for H3.

## Configuration (2026-09-28)

Changes from the 2026-09-24 configuration below:

- Image `vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b`, plus
  [vllm#58132](https://github.com/vllm-project/vllm/pull/58132) (decoder SWA bounded replay) bind-mounted from
  [`overlay-58132/`](overlay-58132/README.md) (`REPLAY_OVERLAY=0` boots without it).
- DSpark: k=5 for ≤4 streams, 3 above (`[[1,4,5],[5,8,3],[9,24,3]]`), with
  `"draft_sample_method":"probabilistic"`. CUDA-graph sizes add 20, 28, 36, 56, 72 and 80 so `C × (k+1)` pads
  little. Server-side default reasoning effort `high` (75 on this image; requests override it).
- Hook: `MEGA_FUSED_SEND=2` ([`hook/peer_fused2.py`](hook/peer_fused2.py)) and `MEGA_LL_GEMM=1`
  ([`hook/ll_gemm.py`](hook/ll_gemm.py)); everything else as below.

## Configuration (2026-09-24)

- GB300: `vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423`, TP1,
  `--moe-backend deep_gemm_mega_moe --enable-expert-parallel`. Engram tables are offloaded to the CPU with
  `--engram-config '{"cpu_offload": true}'`. KV cache is `fp8_ds_mla`, with 24 sequences and 8,192 batched
  tokens. `--gpu-memory-utilization 0.95`.
- DSpark speculative decoding: k=5 for ≤4 streams, k=2 at 5–8 and k=1 above
  (`num_speculative_tokens_per_batch_size [[1,4,5],[5,8,2],[9,24,1]]`), without adaptive verification.
- Hook ([`hook/mega_peer_hook.py`](hook/mega_peer_hook.py)), installed by
  [`hook/sitecustomize.py`](hook/sitecustomize.py):
  - `MEGA_PEER=2` (peer v2), `MEGA_ROWMAP=rowmap-mix-v1.json`;
  - `MEGA_COLD_TRT=0`: no Grace copy of the cold experts, saving 62 GiB of pinned memory. That copy only backs
    batches over 8,192 tokens and the cross-check.
  - `MEGA_COUNT` enables the on-device route counter.
- Sidecar: [`sidecar/peer_server2.py`](sidecar/peer_server2.py) on the RTX PRO 6000 under `numactl --membind=0`.
  It uses b12x `exp/ds41f-gb300` @ `8fdb5635` with fused MoE `w4a8_mx`. Row buckets run from 1 to 8,192, with
  one CUDA graph per (layer, bucket). Buckets of 64 rows or fewer are copied whole inside the graph
  (`VLLM_EXP_PEER2_SMALL=64`).

## Every decode run (catid recipe)

Source: `../results/runs/<run>/decode/c*.json`.

| run | when | C | agg tok/s | user tok/s p50 | TTFT p50 ms | ITL p50 ms | accept len |
|---|---|--:|--:|--:|--:|--:|--:|
| `up-v20` (Al-ENGR v20 reproduced) | 09:41 | 1 | 193.2 | 202.6 | 100 | 4.94 | 2.63 |
| | | 8 | 583.4 | 74.5 | 225 | 13.43 | 1.71 |
| | | 16 | 820.5 | 52.6 | 387 | 19.03 | 1.71 |
| `up-v20-peer2` (v20 + peer v1) | 11:27 | 1 | 209.3 | 216.9 | 102 | 4.61 | 2.70 |
| | | 8 | 757.9 | 98.0 | 237 | 10.20 | 1.71 |
| | | 16 | 1,153.6 | 74.7 | 408 | 13.39 | 1.71 |
| `m3` (MegaMoE, peer v1) | 13:03 | 1 | 205.1 | 213.4 | 153 | 4.69 | 2.61 |
| | | 8 | 724.1 | 103.2 | 303 | 9.69 | 1.71 |
| | | 16 | 1,137.8 | 78.8 | 315 | 12.69 | 1.71 |
| `m3v2` (peer v2, 295 hot)¹ | 14:23 | 1 | 195.1 | 201.4 | 146 | 4.96 | 2.64 |
| | | 8 | 724.2 | 104.4 | 288 | 9.57 | 1.71 |
| | | 16 | 1,157.9 | 83.8 | 267 | 11.93 | 1.72 |
| `m3v2-c1fix` (C1 graph-copy fix)¹ | 14:44 | 1 | 215.9 | 225.6 | 153 | 4.43 | 2.67 |
| `m3v2-h285` (`rowmap-cal-v2-h285`) | 17:00 | 1 | 198.9 | 204.7 | 149 | 4.89 | 2.53 |
| | | 8 | 596.1 | 76.3 | 182 | 13.10 | 1.70 |
| | | 16 | 1,186.8 | 76.2 | 295 | 13.13 | 1.70 |
| `mix-v1` (`rowmap-mix-v1`) | 17:16 | 1 | 207.1 | 211.6 | 148 | 4.72 | 2.57 |
| | | 8 | 776.0 | 100.9 | 282 | 9.91 | 1.71 |
| | | 16 | 1,133.6 | 72.6 | 290 | 13.77 | 1.71 |

¹ These runs predate the sidecar dequant fix (next section), so their speed is valid but their outputs were
not.

## Phase C: speculative-decoding schedule sweep (after the sidecar retune)

Each variant was one boot of `mix-v1`. "Prose" is 320 forced tokens of short prose per stream
([`tools/measure_wait.py`](tools/measure_wait.py)); "wait" is the GB300's measured wait on the sidecar.
Sources: [`results/logs/phaseC-C0.txt`](results/logs/phaseC-C0.txt),
[`results/logs/phaseC-variants.txt`](results/logs/phaseC-variants.txt), and `../results/runs/phaseC-*/`.

| variant | k schedule | adaptive verify | catid C1 agg / user | C4 | C8 | C16 | prose C8 | prose C16 | wait C8 / C16, ms per pass |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|
| C0 control | 5 → 1 above 4 | no | 204.9 / 211.3 | 575.2 | 835.8 | 1,328.7 | 797.8 | 1,307.1 | 0.11 / 0.12 |
| C1 | 5 → 1 | yes | 200.9 / 210.9 | 623.8 | 840.1 | 1,022.5 | 830.0 | 1,003.9 | 0.07 / 0.11 |
| C2 | 5 → 2 | yes | 211.4 / 220.3 | 605.4 | 862.4 | 1,251.3 | 854.2 | 944.4 | 0.12 / 0.14 |
| C3 | 5 → 3 | yes | 211.4 / 220.0 | 614.4 | 890.8 | **1,453.6** | 871.0 | 1,043.5 | 0.10 / 0.12 |
| **C4 (chosen)** | 5 (≤4), 2 (5–8), 1 | no | **211.5 / 219.1** | 583.6 | **926.4** | 1,333.4 | **905.4** | **1,313.2** | 0.09 / 0.11 |

With k=1, adaptive verification collapses acceptance to 1.01 at C16. C3 wins catid C16 (random token ids)
but loses 20% on prose at C16, so it helps random-id benchmarks rather than real text. C4 was chosen for the
best balance across C1–C16.

## Prefill (cold, C1, random token ids)

Sources: `../results/runs/<run>/prefill/prefill.jsonl`, [`results/logs/prefill-mix.jsonl`](results/logs/prefill-mix.jsonl),
[`results/logs/prefill-h285.jsonl`](results/logs/prefill-h285.jsonl).

| run | 8K | 16K | 32K | 64K | 128K |
|---|--:|--:|--:|--:|--:|
| `up-v20` | — | 15,799 | — | — | — |
| `m3` (cold experts from Grace via TRT-LLM) | — | 21,501 | — | — | — |
| `m3v2` (295 hot) | 28,618 | 29,591 | 29,724 | 29,336 | 28,275 |
| `m3v2-h285` (285 hot) | — | 23,526 | — | 23,281 | 22,716 |
| `mix-v1` (285 hot) | — | 23,969 | — | — | 23,275 |

- `m3v2/prefill/prefill.jsonl` has a sixth row: 16K at 119,827 tok/s with 0% GPU utilisation. That request
  hit the prefix cache, so it is excluded.
- Real text at 64K reached 34–36K tok/s on `m3v2` *(notes)*. Random ids concentrate on fewer experts than real
  text does.

## Knee (Al-ENGR's prose knee, aggregate tok/s)

Source: `../results/upstream-logs/knee-*.json`. The knee is noisy: C12 is sometimes below C8.

| run | C1 | C2 | C4 | C8 | C12 | C16 |
|---|--:|--:|--:|--:|--:|--:|
| `m3` | 182.3 | 274.8 | 478.4 | 755.2 | 665.3 | 830.4 |
| `m3` (warm rerun) | 181.1 | 299.5 | 487.4 | 768.1 | 1,041.4 | 1,269.1 |
| `m3v2` | 173.3 | 310.5 | 437.0 | 806.2 | 652.3 | 819.2 |
| `m3v2-h285` | 186.1 | 233.3 | 400.0 | **518.1** | 836.2 | 1,190.7 |
| `mix-v1` | 187.7 | 275.5 | 469.5 | 774.4 | 1,007.1 | 1,196.2 |

## Quality (GSM8K-200, thinking off)

| config | score |
|---|--:|
| `MEGA_PEER=0`: cold experts on the GB300 via TRT-LLM from Grace, 295 hot | 197/200 (98.5%) |
| peer v2 after the dequant fix, 295 hot (`rowmap-static-v1`) | 196/200 (98.0%) |
| peer v2, `rowmap-cal-v2` | 196/200 (98.0%) |
| peer v2, `rowmap-cal-v2-h285` | 196/200 (98.0%) |
| peer v2, `rowmap-mix-v1` | 196/200 (98.0%) |

Sources: [`results/logs/gsm8k-*.json`](results/logs/). An in-server cross-check (`MEGA_PEER_CHECK`) compares the
sidecar's output with TRT-LLM on the same Grace-resident experts. It gave cosine 0.9993–0.9999 on every layer
*(notes)*. b12x is not bitwise reproducible, so greedy first-token logprobs wobble between formatting tokens.
`MEGA_PEER=0` is reproducible.

## Choosing the hot set

- Hot sets live in rowmap files: per layer, `hot` and `cold` expert lists. `rowmap-static-v1` is Al-ENGR's
  295-hot list (verbatim from his recipe).
- Decode and prefill route to largely different experts *(notes)*. `rowmap-cal-v2` was calibrated on text that
  was 92% prefill tokens. It cut prefill cold routes from 14% to 3%, but decode cold routes rose from 4.8%
  (static) to 19–22%, and knee C8 fell to 518.
- `rowmap-mix-v1` weights calibration 0.6 on decode (sets D prose and E code/QA/reasoning) and 0.4 on prefill
  (A stdlib, docs and chat; B vLLM/b12x sources and chat). Out of sample, its decode cold routes are 5.4% vs
  8.4% for static, and its prefill cold routes are about 5–7% vs 10–14% *(notes)*. Calibration inputs are in
  [`calibration/`](calibration/). The builders are [`tools/calibrate_routes.py`](tools/calibrate_routes.py) and
  [`tools/build_rowmap.py`](tools/build_rowmap.py). Decode-only scoring uses
  [`tools/decode_counts*.py`](tools/).
- Always score a candidate map on decode-only counts as well as prefill counts. Maps tuned for random token ids
  are benchmark tuning.

## Sidecar performance

Mean compute per call on the RTX PRO 6000, cumulative over the final boot (2.09 million calls). Source:
[`results/logs/peer_stats.json`](results/logs/peer_stats.json).

| row bucket | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 512 | 2,048 | 8,192 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| mean µs | 74 | 89 | 97 | 115 | 156 | 199 | 246 | 507 | 677 | 1,594 | 4,045 |
| mean rows | 1.0 | 2.0 | 3.5 | 6.3 | 11.7 | 19.0 | 48.6 | 96.5 | 353.5 | 1,204 | 5,312 |

The sidecar is fast because of b12x's fused MoE. For 512 prefill tokens on the cold experts it takes
1.08 ms. The Triton GEMV used by peer v1 takes 10.6 ms (cosine 0.99964;
[`results/peer_prefill_bench.json`](results/peer_prefill_bench.json)). That speedup is what made sending
prefill-sized batches to the 6000 viable (peer v2).

## MegaMoE vs FlashInfer TRT-LLM on the GB300 (microbenchmark)

[`tools/mega_bench.py`](tools/mega_bench.py) / [`mega_bench2.py`](tools/mega_bench2.py) time one DS-V4.1 layer
at production geometry. Results are in [`results/phase1*.json`](results/). In `phase1.json`, FlashInfer is
routed-only. In the `*-shared.json` files it includes the shared expert, and `fi_routed_only_us` is also
given. With the shared expert included, MegaMoE is up to about 25% slower at 1 token and roughly even from 4 to
256 tokens. It is 1.1–1.5× faster from 1,024 tokens up, which is consistent with M3's prefill advantage.

## Bugs worth knowing about

- **Dequant bug.** The sidecar received MXFP8 bytes in a `uint8` buffer and converted the integers instead of
  bitcasting them to `float8e4nv`. Every peer v2 output before the fix (2026-09-24, about 15:50 EDT) computed
  the cold experts on garbage. Speed numbers from that window are valid; quality numbers are not. Calibration
  counts collected then are excluded from this repo. The lesson: standalone tests must run the exact serving
  path. The dequant was the only step no test covered.
- **Autotuning on empty work.** The sidecar first tuned b12x with every route masked (`-1`), so the autotuner
  timed empty work. The 16- and 512-row buckets got configs 2–3× too slow. At C16 the GB300 then waited
  2.5 ms per step. Tuning on realistic compacted rows (1–2 live routes per row; b12x's `make_tuning_routes`
  does this) brought the wait down to 0.12 ms. catid C8 went from 776 to 836 and C16 from 1,134 to 1,329
  (`mix-v1` → Phase C control).
- **C1 graph copy.** The sidecar now copies small buckets whole, so each decode layer needs one graph replay
  (`VLLM_EXP_PEER2_SMALL=64`). Between `m3v2` and `m3v2-c1fix`, C1 rose from 195 to 216 tok/s aggregate
  (201 to 226 per user) *(notes)*.

## Gotchas

- b12x `prepare_weights` rewrites its source tensors in place. Scales must be `float8_e8m0fnu`, clamped to 247,
  with layout `"w31"`. Share tuning scratch across layers.
- The route-count dump thread must make no CUDA calls; it once crashed graph capture.
- Never edit the mounted hook while a container boots.
- Mount the DeepGEMM JIT cache (`/root/.dj`), or MegaMoE recompiles on every boot.
- Allocator "OOM" warnings during boot are recoverable cache retries, not failures.
- Docker's `--cpuset-mems` breaks CUDA init (error 802). Use host `numactl --membind` plus `--cap-add SYS_NICE`.
- Open shared `/dev/shm` files without `O_CREAT`. `fs.protected_regular` refuses `O_CREAT` on another user's
  file in sticky `/dev/shm`, even for root.
- The torch profiler shows no GPU events for graph-replayed decode here. Measure the sidecar wait with the
  `%globaltimer` counters the wait kernel keeps (words 5 and 6), via [`tools/measure_wait.py`](tools/measure_wait.py).
- The sidecar writes per-bucket latency to `logs/peer_stats.json`. A live per-bucket override can be dropped
  into `logs/peer_buckets.json`.

## 2026-09-28: image upgrade and decode profile

The A/B harness ([`tools/ab-suite.sh`](tools/ab-suite.sh), chained by [`tools/ab-chain.sh`](tools/ab-chain.sh))
runs, per boot: GSM8K-200, Al-ENGR's knee, catid decode C1/4/8/16, the C1/2/4 pass time and sidecar wait
([`tools/measure_wait_c1.py`](tools/measure_wait_c1.py)), 8 reasoning prompts at C1, and one 16K random-id prefill.
Reasoning effort was pinned to 75 on both images (the effort names map differently across them). Sources:
`results/logs/ab-*.out`, `results/logs/gsm8k-ab-*.json`, `../results/runs/ab-*/`.

| run | catid C1 agg / user | C4 | C8 | C16 | knee C1 / C8 / C16 | C1 pass ms | reasoning C1 | prefill 16K | GSM8K-200 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `ab-base-7f1a` (nightly-7f1a5398) | 213.7 / 222.2 | 592.1 | 943.0 | 1,334.2 | 179 / 837 / 1,291 | 11.7 | 304.5 | 25,777 | 97.0% |
| `ab-new-af7f` (nightly-af7f9488) | 213.5 / 225.8 | 579.5 | 931.9 | 1,328.1 | 178 / 834 / 1,289 | 11.6 | 307.4 | 26,913 | 98.0% |
| `ab-fused-af7f` (+ fused send v1) | 238.7 / 254.1 | 628.6 | 1,019.9 | 1,460.2 | 214 / 933 / 1,403 | 10.2 | 353.9 | 26,883 | 98.0% |
| `ab-fused-nomhcov-af7f` (+ `MEGA_MHC_OVERLAP=0`) | 241.6 / 256.9 | 628.7 | 1,006.9 | 1,447.1 | 228 / 909 / 1,388 | 10.2 | 351.0 | 26,956 | 97.5% |

**The image upgrade was neutral on decode** and +4% on prefill.

**Profile.** Nsight Systems with CUDA-graph node tracing (`PROF=nsys` in `launch-m3.sh`,
[`tools/nsys_decode.py`](tools/nsys_decode.py)) on `ab-new-af7f`. A C1 step (a 6-token verify) took 12.0 ms under
the profiler, and the GPU was busy 97% of it. By component: MegaMoE 3.7 ms (near HBM bandwidth for the experts it
touches); the hook's PyTorch glue between router and MegaMoE ~1.6 ms (~25 small ops per layer); mHC 1.8 ms summed,
mostly hidden on a side stream; attention 1.35 ms; dense FP8 GEMMs ~1.8 ms; sidecar publish/wait/add 0.5 ms
*(notes; the .nsys-rep files are not in this repository)*.

**Fused send v1** ([`hook/peer_fused.py`](hook/peer_fused.py), `MEGA_FUSED_SEND=1`) replaced that glue plus the
pack and publish with one single-CTA kernel for T ≤ 64: 34–47 → 5–13 µs per layer, bit-identical on 176 cases
([`tools/test_peer_fused.py`](tools/test_peer_fused.py), [`tools/bench_peer_fused.py`](tools/bench_peer_fused.py)).
It gave the jump in the `ab-fused-af7f` row: catid C1 226 → 254 per user and reasoning C1 +15%. Forcing the
single-kernel mHC path changed nothing, so the TileLang overlap stays on.

## 2026-09-28: DSpark schedule on reasoning traffic

Most generated tokens on this lane are now reasoning at effort `high`, which drafts better than catid or prose: at
k=5, DSpark accepts 3.8 tokens per step on greedy reasoning against 2.6 on catid and 2.3 on prose. The schedule
was re-swept with a closed-loop reasoning benchmark ([`tools/reason_bench.py`](tools/reason_bench.py): C workers
each keep one GSM8K or MMLU-Pro question in flight, thinking on, T=1.0, top_p 0.95, 60 s window) plus catid and
the knee ([`tools/sweep-k.sh`](tools/sweep-k.sh)). DSpark's block is 5 tokens, so k ≤ 5.
Sources: `results/logs/reason-*.json`, `results/logs/sweep-*.out`, `../results/runs/k-*/`,
`../results/upstream-logs/knee-k-*.json`.

Reasoning aggregate tok/s (accepted tokens per step):

| run | C1 | C4 | C8 | C12 | C16 | C24 |
|---|--:|--:|--:|--:|--:|--:|
| `k-k521` (5/2/1, as of 09-24) | 286 (3.03) | 680 (2.98) | 969 (2.25) | 1,059 (1.75) | 1,258 (1.76) | 1,516 (1.75) |
| `k-k532` | 270 (2.86) | 691 (3.02) | 1,023 (2.60) | 1,174 (2.29) | 1,376 (2.27) | 1,564 (2.28) |
| `k-k543` | 277 (2.96) | 692 (3.00) | 1,022 (2.83) | 1,192 (2.62) | 1,354 (2.62) | 1,591 (2.62) |
| `k-prod-k532cg` (5/3/2 + finer graphs) | 276 (2.93) | 694 (3.04) | 1,004 (2.56) | 1,187 (2.26) | 1,406 (2.27) | 1,608 (2.28) |

catid C8 / C16 and prose knee C8 / C16 on the same boots: 5/2/1 1,001 / 1,434 and 936 / 1,408; 5/3/2 1,040 /
1,551 and 883 / 1,435; 5/4/3 1,014 / 1,589 and 840 / 1,431; 5/3/2 + graphs 1,060 / 1,581 and 896 / 1,450.
5/3/2 with finer CUDA-graph sizes won (padded graph tokens still route through the experts). T=1 C1 varies
about ±6% run to run.

## 2026-09-28 evening: probabilistic drafting, small-M GEMMs, fused send v2

Each change was benchmarked as one boot with [`tools/sweep-k.sh`](tools/sweep-k.sh); `k-base-rerun` re-measured
the running 5/3/2 production server first. Sources: `results/logs/sweep-probdraft.out`,
`results/logs/sweep-pd-k.out`, `results/logs/reason-*.json`, `results/logs/passtime-*.jsonl`.

| run | reasoning C1 / C4 / C8 / C12 / C16 / C24 | catid C8 / C16 | knee C1 / C4 / C8 | C1 pass ms |
|---|--:|--:|--:|--:|
| `k-base-rerun` (5/3/2, greedy drafts) | 287 / 669 / 1,024 / 1,212 / 1,435 / 1,598 | 1,046 / 1,558 | 211 / 554 / 929 | — |
| `k-probdraft` (+ probabilistic drafts) | 308 / 752 / 1,088 / 1,215 / 1,448 / 1,694 | 1,026 / 1,543 | 225 / 556 / 910 | 10.01–10.06 |
| `k-pd-ll-k532` (+ LL GEMM) | 297 / 734 / 1,082 / 1,224 / 1,432 / 1,647 | 1,039 / 1,580 | 217 / 548 / 893 | 9.96 |
| `k-pd-ll-k533` (k 5/3/3) | 300 / 728 / 1,076 / **1,312** / **1,504** / **1,747** | 1,046 / 1,604 | 209 / 542 / 904 | — |
| `k-pd-ll-k543` (k 5/4/3) | 307 / 725 / 1,069 / 1,298 / 1,504 / 1,718 | 1,019 / 1,580 | 211 / 531 / 884 | — |
| `prod-20260928b` (5/3/3 + fused send v2) | 306 / — / 1,102 / — / 1,535 / — | catid C1 253.5 / 266.3 user, C16 1,613 | — | 9.86–9.89 |

- **Probabilistic drafting** (`"draft_sample_method":"probabilistic"`): the drafter samples from its distribution
  and rejection uses the full draft probabilities, instead of drafting its argmax. At T=1, acceptance of the
  first draft position rises from 0.73 to 0.80, and reasoning gains 6–12% at C1–C8 and C24. Greedy (T=0)
  requests draft the same as before, and catid and prose move within noise.
- **k 5/3/3.** With higher acceptance, a third draft token pays above 8 streams: reasoning C12 +7%, C16 +5%,
  C24 +6% over 5/3/2. 5/4/3 matched it on reasoning and lost 2–3% on catid and prose at C8.
- **LL GEMM** (`MEGA_LL_GEMM=1`, [`hook/ll_gemm.py`](hook/ll_gemm.py)). vLLM runs the MXFP8 dense linears on
  FlashInfer's `cute-dsl` GEMM, a persistent kernel with one CTA per 128-wide N tile: 40 CTAs for wo_b (N=5120) and
  28 for wq_a+wkv (N=1792) on 152 SMs. At decode sizes it streams weights at 2–4.5 TB/s against a 6.3 TB/s read
  reference. The hook sends M ≤ 8 (the C1 verify and the drafter) to FlashInfer's split-K `cutedsl_low_latency`
  kernel, tuned at weight-load time for M = 1..8. At M=6, L2-cold: wq_a+wkv 4.64 → 3.39 µs, wq_b 9.28 → 7.53 µs,
  wo_b 11.63 → 9.43 µs ([`results/microbench/mxfp8_backends.jsonl`](results/microbench/mxfp8_backends.jsonl)). In
  the server the C1 pass went 10.03 → 9.96 ms, about 40% of what the microbenchmark predicted.
- **Fused send v2** (`MEGA_FUSED_SEND=2`, [`hook/peer_fused2.py`](hook/peer_fused2.py)). The send runs one CTA per
  token and quantizes the rows it packs itself, so the separate MXFP8 quantize launch is gone; the sidecar wait
  and the scatter-add are one launch. It is bit-identical to v1 on 176 cases, and the in-kernel quantizer
  matches FlashInfer's byte for byte ([`tools/test_peer_fused2.py`](tools/test_peer_fused2.py)). Per layer: 9.9 →
  8.0 µs at 6 tokens, 19.3 → 9.2 µs at 48
  ([`results/microbench/peer_fused2.txt`](results/microbench/peer_fused2.txt)). The floor is the system-scope
  fence and release to the sidecar's host-mapped words, not the packing. The C1 pass went 9.96 → 9.87 ms.

GSM8K-200 of `prod-20260928b`: 98.0% (`results/logs/gsm8k-prod-20260928b.json`).

## Decoder SWA bounded replay (vllm#58132)

[vllm#58132](https://github.com/vllm-project/vllm/pull/58132), unmerged and maintainer-approved, bind-mounted over
the image ([`overlay-58132/`](overlay-58132/README.md)). DeepSeek-V4.1-Flash's layers 21–39 hold only
sliding-window KV and read long-range context from layer 20's compressed KV. In eager prefill steps the PR runs
those layers on each request's last 128 tokens. On this server every prefill chunk over 128 tokens is eager (the
largest decode CUDA graph is 128 tokens). The boot log confirms it is on, and it stays on with DSpark. Sources:
`results/logs/realtext-*.log`, `../results/runs/replay58132/`, `results/logs/longctx-*.json`,
`results/logs/reason-replay58132.json`, `results/logs/gsm8k-replay58132.json`.

| | `prod-20260928b` | `replay58132` |
|---|--:|--:|
| real-text prefill 16K, C1 (tok/s; TTFT) | 38.1–38.9K; 0.42 s | 59.8–61.3K; 0.27 s |
| real-text prefill 64K, C1 | 36.9–37.4K; 1.75–1.78 s | 59.1–60.8K; 1.08–1.11 s |
| random-id prefill 16K / 64K / 128K, C1 | 26.9K at 16K (`ab-fused-af7f`) | 44.5K / 44.8K / 44.0K |
| random-id prefill 16K / 64K / 128K, C4 aggregate | — | 46.6K / 46.2K / 44.8K |
| reasoning C1 / C8 / C16 | 306 / 1,102 / 1,535 | 312 / 1,098 / 1,519 |
| catid C1 user / C16 agg | 266.3 / 1,613 | 257.9 / 1,629 |
| C1 pass ms | 9.86–9.89 | 9.82–9.85 *(notes)* |
| GSM8K-200 | 98.0% | 98.0% |
| needle, 3 depths at 57K and at 114K tokens | 6/6 (two runs) | 6/6 |

Real text is source code and docs with a unique prefix per request ([`tools/realtext_prefill.py`](tools/realtext_prefill.py)).
The first 16K request after the overlay boot ran at 35.2K tok/s, which includes one-time JIT warm-up. catid's
two-GB300 PP2 numbers: 35.9K at 16K and 56.0K at 128K (C1).

The replay is an approximation at the window edge, not an exact rewrite. [`tools/longctx_check.py`](tools/longctx_check.py)
compares greedy 48-token continuations of six 16K/64K real-text prompts, each run with its own `cache_salt` so
nothing comes from the prefix cache. Two runs of the stock server agree on 1–48 leading tokens per prompt (the
b12x sidecar is not bitwise reproducible); stock vs replay agree on 9–48. First-token top-5 sets overlap in 4–5
of 5 in both comparisons.

## Measured and dropped (2026-09-28)

- **Attention.** FlashMLA's fused mega kernel takes 26.7 µs per layer for any s_q from 1 to 96. It runs one CTA
  per query token over 640 keys (128 SWA + 512 compressed top-k), in 10 serial blocks of 64 at ~2.7 µs each.
  Padding 64 heads to the 128-head 2-CTA kernel saves ~4 µs per layer only up to 12 tokens, and costs more at 96.
  An NVFP4 compressed cache does not change the time (it was adopted on 2026-10-02 for capacity instead; see
  [NVFP4 KV](#nvfp4-kv-ab-same-lane-util-089))
  ([`results/microbench/mega_attn_variants.jsonl`](results/microbench/mega_attn_variants.jsonl),
  [`tools/mega_attn_bench.py`](tools/mega_attn_bench.py)).
  [deepseek-ai/FlashMLA#227](https://github.com/deepseek-ai/FlashMLA/pull/227), built for sm_103a with its B200
  gate relaxed, gives nothing at decode sizes: its gains need many queries per persistent CTA
  ([`results/microbench/mega_attn_flashmla227.txt`](results/microbench/mega_attn_flashmla227.txt)). Splitting a
  token's keys across CTAs is the remaining large decode lever, about 0.6 ms per step from C1 to C24.
- **Dense GEMMs above M=8.** FlashInfer's TRT-LLM, cutlass and cuDNN MXFP8 backends are no faster than `cute-dsl`
  ([`results/microbench/mxfp8_trtllm_vs_cutedsl.jsonl`](results/microbench/mxfp8_trtllm_vs_cutedsl.jsonl)). A Triton
  split-K kernel ([`hook/mxfp8_splitk.py`](hook/mxfp8_splitk.py)) emits the native block-scaled tcgen05 MMA and is
  correct, but it is 25–60% slower because it loads with `cp.async` rather than TMA
  ([`results/microbench/mxfp8_triton_splitk.txt`](results/microbench/mxfp8_triton_splitk.txt)).
- **Drafter argmax.** With greedy drafting, DSpark's 5 sequential single-CTA argmaxes over 129,280 logits cost
  ~25 µs each; a two-stage PyTorch argmax gets 11–24 µs
  ([`results/microbench/drafter_argmax.jsonl`](results/microbench/drafter_argmax.jsonl)). Probabilistic drafting
  replaces the argmax with a Gumbel sample, so this was not pursued.

## 2026-09-29: 32 sequences and pinned FlashInfer tactics

**Tactics.** FlashInfer autotunes its dense MXFP8 GEMMs at boot. Two boots that each tuned live disagreed on 58 of
105 shared tactic keys *(notes)*. In FlashInfer 0.7.0 live-tuned entries also outrank loaded ones, and
[`hook/ll_gemm.py`](hook/ll_gemm.py) re-tuned the M ≤ 8 GEMMs on every boot. So A/Bs that changed the engine
hash, such as the 2026-09-28 k-schedule sweeps, carry ~1–4% of boot-to-boot noise.
- `MEGA_FI_TUNE_FILE` now loads the set this lane served on 2026-09-28/29
  ([`hook/fi-tune/fiset-a964-9a5ab530.json`](hook/fi-tune/fiset-a964-9a5ab530.json), 117 configs).
- [`hook/mega_peer_hook.py`](hook/mega_peer_hook.py) copies it to a working file, and `ll_gemm.py` takes its
  M ≤ 8 tactics from it instead of tuning. The boot log confirms this with `tactics from the pinned FlashInfer
  autotune file, no live tune`.
- Pinned boots agree on catid within ~1%.
- Re-pin after any image or FlashInfer upgrade: boot once with `MEGA_FI_TUNE_FILE=` and copy the saved set.
- The sidecar still races its b12x configurations on every boot.

**32 sequences.** `SEQS` went from 24 to 32 in [`launch-m3.sh`](launch-m3.sh). Each boot ran
[`tools/arm-suite.sh`](tools/arm-suite.sh): GSM8K-200, Al-ENGR's prose knee, catid decode from C1 to C32,
reasoning from C1 to C32, and catid 16K prefill. Logs are in [`results/logs/seats32/`](results/logs/seats32/),
and catid runs in `../results/runs/arm-*`. The suite also ran a fidelity check against J-M-Recipes' reference
(agent-fixture acceptance and teacher-forced top-1 flips). Its output is in the same logs and is not analysed here.

| | 24 seqs, live autotune (`arm-s24-live`) | 32 seqs, pinned (`arm-s32-pin-a`) | 32 seqs, pinned (`arm-s32-pin-b`) |
|---|--:|--:|--:|
| catid C1 agg / user | 247 / 263 | 244 / 254 | 244 / 251 |
| catid C8 / C16 / C24 agg | 1,046 / 1,645 / 1,974 | 1,059 / 1,621 / 1,950 | 1,052 / 1,604 / 1,943 |
| **catid C32 agg, TTFT p50** | **1,898, 1,911 ms** | **2,303, 319 ms** | **2,319, 311 ms** |
| reasoning C1 / C8 / C16 | 311 / 1,080 / 1,542 | 329 / 1,075 / 1,533 | 314 / 1,082 / 1,487 |
| reasoning C24 / C32 | 1,777 / 1,774 | 1,738 / 2,106 | 1,764 / 2,036 |
| prefill 16K (random ids) | 44.3K | 44.0K | 44.3K |
| GSM8K-200 | 98.0% | 98.0% | 97.5% |
| KV cache | 3,618,460 tokens | 3,527,818 | 3,527,818 |

At 24 sequences, C32 queues a quarter of its requests, which shows up as the 1.9 s TTFT. 32 sequences remove
that queue and cost 90K KV tokens. The reasoning C1 cells of the three boots span 311–329, so a single C1
reasoning run is good to about ±3%.

`MEGA_NO_MEGA_MHC=1` does not boot at 32 sequences. `cuGraphInstantiate` fails with "operation not permitted"
on a full CUDA graph ([`results/logs/seats32/arm-chain-20260929.out`](results/logs/seats32/arm-chain-20260929.out)).
That is the kernel-node resource limit, so production boots sit near it.

## 2026-10-02/03: sharing the GB300 with MiniMax-H3

MiniMax-H3 generates video with audio from text, a first frame or reference media (image, video and voice). It
runs in vllm-omni ([`tools/coexist/serve-h3-gracie.sh`](tools/coexist/serve-h3-gracie.sh)). Its DiT, text
encoder and VAEs can stream layer by layer from pinned Grace memory over NVLink-C2C. In our earlier H3 tests that
cost no speed *(notes)*, and it leaves a small HBM footprint, so H3 can share the GB300 with this lane. The
modes used here:

| H3 mode | what is pinned in Grace | GB300 idle | 15 s render peak (H3 only) |
|---|---|--:|--:|
| `fl2va-min`: FL2VA only, legacy `--enable-layerwise-offload` (DiT and text encoder streamed, VAEs staged) | ~157 GiB *(notes)* | 3.7 GiB *(notes)* | 17.5 GiB |
| `combined` + `OFFLOAD=legacy`: FL2VA and Ref2VA (voice cloning) behind one text encoder | ~268 GiB | 6.5 GiB | 20.3 GiB |

The 268 GiB is DS41 alone at 332 GiB of host memory available, then 64 GiB available with combined H3 up
([`results/logs/h3-coexist/ds41-final-coexist.log`](results/logs/h3-coexist/ds41-final-coexist.log)).

### What had to change

The 2026-09-29 lane (util 0.95) left no HBM for H3, and its Engram tables took most of Grace. Four changes:

1. **265 hot experts instead of 285** ([`hook/rowmap-mix-v1-h265.json`](hook/rowmap-mix-v1-h265.json)).
   - It uses the `mix-v1` recipe cut at 265: decode sets D and E weighted 0.3 each, prefill sets A and B 0.2
     each. [`tools/build_mix_rowmap.py`](tools/build_mix_rowmap.py) rebuilds `mix-v1` byte for byte at 285.
   - It saves 20 × 40 experts × ~18.8 MB ≈ 14 GiB of HBM. Weights load as 204.75 GiB
     ([`boot-u92.txt`](results/logs/h3-coexist/boot-u92.txt)).
   - The sidecar now holds 119 × 40 cold experts, ~89.5 GB.
   - Decode cold-route share on the calibration sets: D 3.4 → 4.9%, E 2.8 → 4.2% *(notes)*.
   - `-h270` and `-h260` were built as alternatives.
2. **NVFP4 Engram tables** ([`engram-nvfp4/`](engram-nvfp4/README.md)).
   - The source is the re-quantization `aidendle94/DeepSeek-V4.1-Flash-NVFP4-Engram` (MIT): 203 GB → 111 GB.
     Stock vLLM can't read it, so a two-file overlay adds an NVFP4 lookup kernel and an exact-size registered
     host allocation.
   - The bigger saving was a finding about the FP8 tables. torch's pinned allocator rounds each table to a
     power of two, so FP8 Engram pinned **264 GiB**, not 189. NVFP4 pins exactly **103 GiB**.
   - Both tables log `Engram NVFP4 table offloaded to registered host memory ... 51.50 GiB` at boot.
3. **`nvfp4_ds_mla` KV** (`EXTRA="--kv-cache-dtype nvfp4_ds_mla"`). The compressed per-token record shrinks from
   528 to 288 bytes *(notes)*. The image supports it and the mega attention kernel reads it. See the A/B below.
4. **GPU memory utilization 0.885**, after 0.92 and 0.89; see the coexistence tests.

### Test boot: 265 hot, NVFP4 Engram, FP8 KV, util 0.92

[`tools/coexist/ds41-h3-test.sh`](tools/coexist/ds41-h3-test.sh), log
[`ds41-h3-test.log`](results/logs/h3-coexist/ds41-h3-test.log):
- **Boot:** 6,928,817 KV tokens (13.39 GiB). Chat and parallel tool calls work.
- **Quality:**
  - GSM8K-200 98.0%;
  - needles 3/3 at 108,593, 433,938 and 867,523 prompt tokens;
  - GPQA-Diamond 92.4% at T=1 ([below](#gpqa-diamond-t1-vs-greedy)).
- **Reasoning decode:** 296 / 1,082 / 1,493 tok/s at C1 / C8 / C16. The 2026-09-29 pinned boots got 314–329 /
  1,075–1,082 / 1,487–1,533. C1 is 6–10% lower; C8 and C16 are unchanged.
- **Prefill (random ids):** 39.6K / 40.4K / 39.4K tok/s at 16K / 64K / 128K, against 44.0–44.3K at 16K
  before (−11%). The h265 rowmap and NVFP4 Engram changed together, so their shares of the loss were not
  separated. At 265 hot, more prefill rows go to the sidecar.
- **Coexistence:** a 15 s H3 render (fl2va-min) **ran out of memory** (HTTP 500 after 234 s). The GB300 peaked
  at 249.8 of 250.7 GiB ([`gb300-mem-coexist.txt`](results/logs/h3-coexist/gb300-mem-coexist.txt)).
  - DS41's allocator grows ~5.3 GiB above its post-boot size once it has served 1M-token prefills *(notes)*;
    that growth used up H3's room.
  - DS41 logged no errors and decoded at 161 tok/s (C1) during the render. Only the render failed.

### Coexistence tests at util 0.89 and 0.885

Each run first warms DS41 to its grown size: a 1M-token needle, a 128K prefill and C16 decode. Then a
30-step 1024×576 render runs while DS41 decodes reasoning traffic at C1. Peaks are `nvidia-smi memory.used`,
sampled every second. The scripts are [`tools/coexist/ds41-final-coexist.sh`](tools/coexist/ds41-final-coexist.sh)
and [`ds41-0885.sh`](tools/coexist/ds41-0885.sh); logs are in [`results/logs/h3-coexist/`](results/logs/h3-coexist/).

| DS41 lane | DS41 after warm-up | H3 render | result | wall | GB300 peak | DS41 C1 during |
|---|--:|---|---|--:|--:|--:|
| FP8 KV, 0.89 (3.05M tokens) | 226.5 GiB | fl2va-min, 15 s t2va | 200 | 257 s | 245.0 GiB | 183 tok/s |
| same | 226.5 | combined, 15 s t2va | 200 | 258 s | 247.8 | 171 |
| same | 226.5 | combined, 6 s ref2va voice clone | 200 | 93 s | 242.1 | 152 |
| NVFP4 KV, 0.89 (4.34M), after the full A/B suite | 227.3 *(notes)* | combined, 15 s t2va | 200 | 238 s | 248.6 | — |
| **NVFP4 KV, 0.885 (3.42M, current)** | **225.3** | combined, 15 s t2va | **200** | **238 s** | **246.6** | — |

With no render running, C1 is 304 tok/s on the FP8 0.89 lane and 304.5 on NVFP4 0.89. A render therefore costs
DS41 40–50% of its C1 decode while it lasts. H3's prompt encoding streams the text encoder from Grace, and in the
first 15 s render it took 31 s of the 257. At 0.885 the worst case keeps 3.2 GiB of headroom, measured against
the 249.8 GiB at which the 0.92 render failed. The 0.89 NVFP4 lane kept only 1.2 GiB.

### NVFP4 KV A/B (same lane, util 0.89)

[`tools/coexist/ds41-nvfp4kv-ab.sh`](tools/coexist/ds41-nvfp4kv-ab.sh). The FP8 reference ran on the util-0.89
FP8 lane. The NVFP4 lane was then booted with H3 stopped ([log](results/logs/h3-coexist/ds41-nvfp4kv-ab.log)).
Both boots use the same pinned tactics.

| | FP8 KV (`fp8_ds_mla`) | NVFP4 KV (`nvfp4_ds_mla`) |
|---|--:|--:|
| KV cache at util 0.89 | 5.89 GiB, 3,049,362 tokens | 5.89 GiB, 4,344,140 tokens *(notes)* (+42%) |
| reasoning C1 / C8 / C16 | 296 / 1,082 / 1,493 (util-0.92 test boot) | 304 / 1,071 / 1,470 |
| prefill 16K / 64K / 128K (random ids) | 39.6K / 40.4K / 39.4K (util-0.92 test boot) | 39.3K / 40.3K / 39.7K |
| needles at 125K / 500K / 1M targets | 3/3 each | 3/3 each |
| GSM8K-200 | 98.0% | 98.5% |
| GPQA-Diamond, T=1 | 92.4% (0 capped) | 90.4% (2 capped) |
| GPQA-Diamond, T=0 | 84.8% (18 capped) | 85.4% (18 capped) |
| 15 s combined H3 render, GB300 peak | 247.8 GiB | 248.6 GiB |

**Numerics** ([`tools/ds41_numerics.py`](tools/ds41_numerics.py)):
- **Teacher-forced over 98,256 tokens of code and docs:** NLL 0.8244 (NVFP4) against 0.8265 (FP8). That is a
  0.2% perplexity ratio below 1, which is noise. Argmax agreement 96.7%; |Δ logprob| p50 0.001, p90 0.23,
  p99 0.97.
- **Greedy continuations from 16K, 64K and 128K prompts (three each):** 7 of 9 match FP8 for all 64 tokens. One
  16K prompt splits at token 20 and one 128K prompt at token 9, both at near-ties (top-5 overlap 4/5 and 3/5).
- **No cross-boot baseline yet.** FP8 against FP8 on a second boot hasn't been measured. b12x's sidecar is not
  bitwise reproducible, so part of that 3.3% argmax disagreement is boot noise.

**Why +42% and not 1.83×:** the 528 → 288-byte saving applies only to the compressed MLA record. The rest of
each token's KV state (the SWA window and the indexer cache) keeps its size.

**Paired GPQA:** at T=1, 9 answers were right only on FP8 and 5 only on NVFP4 (McNemar p = 0.42). At T=0 the
split was 9 and 10 (p = 1.0). Neither difference is significant, and they point in opposite directions. The
script's gates kept NVFP4: needles 3/3 at every length, GSM8K ≥ 97%, and GPQA no more than 3 points under FP8.
It would have relaunched on FP8 if any gate failed. NVFP4 then ran at util 0.89, and 0.885 from 2026-10-03.

### GPQA-Diamond: T=1 vs greedy

All runs use 198 items, reasoning effort max, max_tokens 131,072, C32 and llm-inference-bench's GPQA profile.
Only aggregate scores are published here; per-item outputs are kept privately.

| lane | T=0 (greedy) | T=1, top_p 0.95 |
|---|--:|--:|
| 285 hot, FP8 Engram, FP8 KV (2026-09-30) | 87.4%, 18 answers hit the cap *(notes)* | — |
| 265 hot, NVFP4 Engram, FP8 KV | 84.8%, 18 hit the cap | 92.4%, 0 |
| 265 hot, NVFP4 Engram, NVFP4 KV | 85.4%, 18 | 90.4%, 2 |

- **Greedy runs lose 5–8 points to loops.** The answers that hit the cap are the model repeating itself to
  131K tokens, and most of them score wrong: 13 of 18 in the 285-hot run and 17 of 18 in the 265-hot FP8-KV run.
  Excluding them, those two greedy runs score 93.3% and 92.8%, the same as T=1.
- **Loops aren't tied to questions.** Only 9 of the 18 looping questions are the same in the two greedy FP8 runs:
  batching is not deterministic, even at T=0. The T=1 run answered 22 of the 27 questions that looped in either
  greedy run correctly.
- **The rowmap and Engram change didn't move quality.** The greedy pair across it (87.4 → 84.8%) is 13 against 8
  flipped answers, p = 0.38.
- **Use T=1 for comparisons.** Greedy GPQA mostly measures how often a model loops.

### Gotchas from these runs

- **`swap-to-m3v2.sh` still boots the 285-hot lane.** Start this lane with
  [`swap-to-ds41-h3.sh`](swap-to-ds41-h3.sh). The DS41 container has no restart policy, so after a host reboot
  start it by hand.
- **The H3 launcher used to read `PORT`.** The DS41 test scripts export `PORT=30006`, so every H3 restart from
  them bound DS41's port and died with "Address already in use".
  - The first test A and B attempts (21:30–21:40 in `ds41-final-coexist.log`) are void for that reason. The
    launcher now reads `H3_PORT`.
  - The 303.7 tok/s "during lean" line in that log is the no-render C1 from the void attempt.
- **Restarts and timed runs wait for an idle lane.** Clients were using the lane during these tests.
  [`tools/coexist/lan-idle.sh`](tools/coexist/lan-idle.sh) waits until the DS41 and H3 `/metrics` show no
  running or waiting request for 120 s. It treats a stopped container as idle.
- **The swap script warns falsely.** Its `no 'Decoder SWA bounded replay' line` warning fires on timing. The
  boot log has the line ([`boot-u92.txt`](results/logs/h3-coexist/boot-u92.txt)).
- **When H3 runs out of memory, only its request fails.** The render returns HTTP 500 and DS41 keeps serving.
