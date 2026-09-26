# DeepSeek-V4.1-Flash on one GB300 (+ RTX PRO 6000): experiment log

Host gracie: GB300 (251 GiB HBM, SM103), 506 GB Grace LPDDR5X, RTX PRO 6000 Blackwell Max-Q (96 GB, PCIe Gen5 x16).
Model /home/jasonc/models/DeepSeek-V4.1-Flash (515 GB; routed experts 289 GB, Engram 203 GB).

## Targets

| Baseline | HW | Decode C1 per-user | Decode aggregate | Prefill |
| --- | --- | --- | --- | --- |
| catid vLLM PP2+DSpark | 2x GB300 | 252.9 tok/s | C8 1,024 / C16 1,678 / C64 3,258 | 16K 35.9K, 128K 56.0K tok/s (C1) |
| catid vLLM TP2+DSpark | 2x GB300 | 201.0 | C64 3,401.6 | ~33-35K |
| Al-ENGR v15 hot-pin | 1x GB300 | 153 prose / 250 shell | C8 705, C16 ~955 | ~17-18K (6.5K-414K) |

Decode method: llm-inference-bench 0b4185b5, 8,192 in / 1,024 out, temp 0, 5xC requests.
Prefill method: catid bench_prefill.py --engine vllm, random tokens, 1 output token, cache reset per point.

## Measured infrastructure (2026-09-23)

| Path | Bandwidth / latency |
| --- | --- |
| GB300 <-> Grace, copy engine (pinned) | 388 / 383 GB/s |
| GB300 SM reads of mapped Grace, write-combined (b12x default before) | 91 GB/s |
| GB300 SM reads of mapped Grace, cacheable | 370-388 GB/s |
| RTX PRO 6000 <-> host (PCIe Gen5 x16) | 55 / 57 GB/s; 160 KB round trip 13 us |
| Residency all-Grace route, production geometry | 223 us/route (WC) -> 60 us/route (cacheable) |

## Runs

### MoE kernel microbenchmark (one layer, all 384 experts in HBM, H5120/I2304/top-6)

| Tokens | FlashInfer TRTLLM MXFP4xMXFP8 | b12x residency all-HBM | FI split hot295/cold89 |
| ---: | ---: | ---: | ---: |
| 1 | 36.8 us (3.07 TB/s) | 93.4 us | 45.0 us |
| 8 | 149.8 us (5.65 TB/s) | 421.1 us | 164.1 us |
| 64 | 742 us (6.26 TB/s) | ~3,041 us | 763 us |
| 1,024 | 1,701 us | - | 1,472 us |
| 16,384 | 5,371 us (3.05M tok/s) | - | 5,373 us |

### Platform finding: Linux page cache on GB300 HBM

GB300 HBM is NUMA node 1. Reading the 515 GB checkpoint spilled ~37 GB of page cache onto node 1,
which CUDA then could not use (vLLM saw 213 of 249.5 GiB free; first residency placement kept only
7,653 of 15,360 experts in HBM). Fix: `numactl --membind=0` for the server (host allocations, file
cache and the pinned Grace tier stay on Grace) plus dropping caches before a run.

### Serving runs (vLLM karmic + b12x, one GB300, catid decode recipe 8K in / 1K out)

| Run | MoE kernel | Placement | C1 tok/s/user | C8 agg | Prefill 16K C1 |
| --- | --- | --- | ---: | ---: | ---: |
| b12x-base | b12x residency operator (all tiers) | balanced static, 11,479 hot | 58.1 | 131.0 | 346 tok/s (1K chunks) |
| fi-h290 | FlashInfer TRTLLM MXFP4xMXFP8, hot HBM + cold Grace | positional 290/layer | 63.5 | 200.0 | - |

Decode-step profile, fi-h290 at C1 (15 ms step, 16.5 ms kernel time):
- b12x SM103 dense block-FP8 GEMMs 4.3 ms (246 calls; one fixed 128x128 tile config, no split-K, 14-40 CTAs at M=1)
- FlashInfer cold tier read from Grace 5.1 ms (FC1 87 us + FC2 44 us per layer); hot tier 1.0 ms
- MoE finalize 0.85 ms, mHC 0.8 ms, attention decode 0.65 ms, vocab projection 0.75 ms

Routing skew (calibration workload of code/shell/math/prose/Chinese/JSON prompts, 4.7M routes):
top-192 covers 85.3% of routes, top-256 93.7%, top-290 96.7%, top-320 98.5% (mean per layer);
positional first-290 only 74.9%. A global 12,200-expert hot allocation leaves 2.2% of routes cold.

Microbenchmarks:
- FlashInfer with the cold tier in cacheable Grace memory: 1 token 281 us/layer (vs 45 us all-HBM split),
  ~170 us per touched cold expert at decode sizes; 1,024 tokens 7.6 ms (cold experts read at ~276 GB/s).
- SM103 decode GEMM, V4.1 dense shapes: b12x A16 split-K kernel 1.1-2.3x faster than the dense MXFP8 path
  (wq_a 15.0 -> 7.8 us, wq_b 24.9 -> 16.8, wkv 15.0 -> 6.6, wo_b 20.0 -> 17.9, shared w13 15.4 -> 11.2, w2 11.5 -> 8.6).
- RTX PRO 6000 VRAM streaming: 1.66 TB/s from DRAM (4-5 TB/s from its 128 MB L2); one cold expert ~11 us.
| fi-p12200 | FlashInfer hybrid | calibration profile, 12,200 hot (2.2% cold routes) | 76.9 | 338.0 | - |
| fi-p12200-ds7 | same + DSpark k=7 (adaptive, greedy), b12x autotune off | same | 125.9 (accept 2.56) | 294.6 | - |

DSpark k=7 C16: 358.2 tok/s aggregate (23.3/user).

RTX PRO 6000 peer-tier microbenchmark (two processes, pinned Grace memory, GPU-polled flags, no CPU in the loop):
- GB300 -> 6000 -> GB300 round trip with 10 KB activations each way: 15.1 us per layer.
- With 18.3 MB (one expert) of 6000 compute: 20.4 us; 36.6 MB: 21.4 us.
- Overlap: 51.5 us of GB300 HBM streaming alone -> 57.3 us with a concurrent peer round trip + one cold expert.
  At 150 MB (8 experts) the 6000 becomes the bottleneck: 229.9 us.
| fi-p12200-ds3 | same + DSpark k=3, FI autotune off, b12x autotune off | same | 131.8 (accept 2.43) | 328.7 | - |

DSpark k=3 C16: 412.3 tok/s aggregate (26.8/user).

C8 speculative-decode profile (fi-p12200-ds3): with FlashInfer autotune off the MoE runs a default
t128x8x512 tactic, 222 us FC1 + 112 us FC2 per call, 56% of kernel time. Autotune was off only because
per-layer hot counts made every layer a new shape (25 minutes of tuning per launch).

VMM-stitched expert tables (CUDA VMM: HBM pages for the hot prefix, host-NUMA Grace pages for the rest,
one address range per weight tensor): FlashInfer runs one call per layer at all-HBM speed when routes
are hot (1 token 36.6 us, 8 tokens 151.1 us, 64 tokens 656 us). Every layer has the same shape, so
autotune runs once. A route that lands on a Grace-backed expert still costs ~200 us inside the
FlashInfer kernel (latency-bound C2C streaming by few CTAs); b12x's residency operator pays ~60 us
per Grace route and the RTX PRO 6000 ~11 us per expert plus a 15 us overlappable round trip.
| fi-stitch-ds3 | FlashInfer, VMM-stitched tables, FI autotune primed, DSpark k=3 | same | 102.3 (accept 2.37) | 222.0 | - |

fi-stitch-ds3 C16: 276.3. Stitching regresses: a route that lands on a Grace-backed expert inside
the single FlashInfer call puts a C2C-latency-bound CTA on the kernel's critical path (~200 us).

Triton cold-route kernel (checkpoint-layout MXFP4 in cacheable Grace, split-K over compacted routes):
correct to 1.7e-3 relative (bf16 output rounding) against a dense reference; GB300 from Grace
83 us for one route, ~64 us per route at 4-8 tokens (b12x residency operator: ~60 us per route);
RTX PRO 6000 from its own VRAM ~20-30 us per route.

A16 decode for V4.1 block32 FP8 linears (b12x split-K BF16 x MXFP8 on SM103; [32,32] UE8M0 scales
expanded to 1x32): validated on SM120 at 1.6e-3 relative for all V4.1 shapes, rows 1-8.

PR2 qualification: the first formal run on the final head found that the launcher's residency
suites selected no tests (parametrized ids hot0..hot2 vs -k hbm/mixed/grace). Fixed by naming the
placements; all 11 residency GPU tests pass directly on the GB300; formal run 4 restarted at 02:03.

### Final runs (2026-09-24 03:26–05:45)

| Run | Config | C1 | C8 | C16 | C32 | C64 | Prefill 16K / 64K / 128K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A-fi-tuned | uniform 305 hot, primed FI tuning, staging >=512 tok, DSpark k=3 | 135.2 | 335.3 | 420.9 | - | - | 4,953 / 4,550 / 4,236 (GPU 43% busy: disk Engram) |
| pr-validate | vLLM #888 + b12x #426, b12x residency operator, 11,584 hot | 59.7 | 136.3 | - | - | - | 370.5 / - / - |
| C-peer | A + RTX PRO 6000 peer tier | 133.5 | 417.4 | 520.0 | - | - | - |
| D-peer-a16 | C + A16 decode linears, 64 seqs | 138.6 | 419.3 | 517.1 | 520.7 | 599.6 | - |
| E-rebased | D on current Karmic dev (exp2) + b12x #426 | 147.4 | 464.2 | - | - | - | 1,441 / 1,414 / 1,382 (GPU spins at 254 W) |

Peer tier: 0 timeouts over every run (421,480 layer calls in D alone); parity with local compute <= 1e-5.
Al-ENGR (J-M-Recipes/recipes) v20: 181 C1 (knee.sh short prompt, 192 tok), 657 C8, 969 C16, 22-23K prefill;
same hot/cold split idea (v15 hook: 295 HBM / 89 Grace, two do_finalize=False FlashInfer calls + fp32 finalize),
plus a DSpark k-schedule [[1,4,5],[5,24,1]], Engram in RAM, fp8_ds_mla KV, upstream SM100 kernels.
| F-ksched | E + DSpark k=5 to 4 seqs, k=1 above (Al-ENGR schedule) | 148.5 | 382.4 | 502.0 | 590.5 | - | - |

k=1 above 4 seqs: C32 improves (590.5 vs 520.7, all decode now inside the peer's 64-token window) but C8/C16 fall
(accept 1.73 vs 2.3); on this stack k=3 to 16 seqs then k=1 is the better schedule (untested).
