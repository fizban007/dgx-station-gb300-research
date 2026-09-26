# M3 lane: details

This file backs the [README](README.md). Every number is in [`results.jsonl`](results.jsonl) with its source
file; regenerate that file with [`tools/make_results_jsonl.py`](tools/make_results_jsonl.py). Numbers marked
*(notes)* come from session notes; their raw output was not kept. All runs are from 2026-09-24, in EDT.

## Configuration (final)

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
