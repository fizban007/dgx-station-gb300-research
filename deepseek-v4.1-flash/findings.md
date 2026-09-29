# Is there anything in b12x worth doing for DeepSeek-V4.1-Flash on GB300?

Host gracie: one GB300 (SM103, 152 SMs, 251 GiB HBM), Grace (LPDDR5X), RTX PRO 6000 (SM120, 96 GB, PCIe Gen5).
Checkpoint DeepSeek-V4.1-Flash `dba1be0a` (the commit after `df42c109`; prompt-encoding scripts only, weights identical).

**Scope.** Everything here is one model and one quantization mix: DeepSeek-V4.1-Flash with MXFP4 (E2M1, E8M0 per
32) routed experts, FP8 block-scaled dense layers, and MXFP8 activations. Nothing below establishes how b12x does
on other models or formats (NVFP4, W4A16, IQ2, FP6, ...); those are open questions, not settled negatives.

## Method

The bar is upstream, measured on this machine: Al-ENGR's v20 release reproduced from
github.com/J-M-Recipes/recipes (`dffd01cc`): image `vllm/vllm-openai:nightly-2671fedfc…` (vLLM
0.29.1rc1.dev9+g2671fedfc, FlashInfer 0.6.18.post1), their v15 pin-hot-experts hook and static rowmap
(295 hot / 89 cold experts per layer), their flags (UVA offload 54 GiB, Engram in Grace RAM, fp8_ds_mla KV,
DSpark k=5 with k-schedule [[1,4,5],[5,24,1]], 24 seqs, token capture sizes). Differences: GB300 selected by
UUID; host memory bound to the Grace node with numactl (cpuset-mems breaks CUDA init on this host, error 802);
their pinned autotune set is unpublished, so the first boot live-tuned (16 min, 189 configs) and later boots
load that set.

Benchmarks: catid's decode recipe (llm-inference-bench 0b4185b5, 8,192 in / 1,024 out, T=0, 5xC requests),
catid's bench_prefill.py (uniform random token ids), a real-text prefill (source code and docs, unique prefix
per request), Al-ENGR's knee.sh (short prose prompt, thinking off, 192 tokens, 2 runs), and GSM8K (last 200
test questions, greedy, thinking off) as a quality check. catid's published DS-V4.1 numbers are all on **two**
DGX Stations (PP2/TP2 over RDMA; he did not attempt one), so they are not a single-GB300 bar.

## 1. Upstream reproduces here and beats the b12x stack on every metric

| | Upstream v20 (GB300 only) | Best b12x stack (GB300 + 6000) |
| --- | ---: | ---: |
| knee C1 / C8 | 188.9 / 686.4 (live-tuned boot) | - |
| catid C1 per user | 202.6 (live) / 187.8 (loaded) | 148.5 |
| catid C8 aggregate | 583.4 / 567.0 | 464.2 |
| catid C16 aggregate | 820.5 / 798.5 | 520.0 |
| prefill 16K | 15,799 tok/s | 4,953 tok/s |
| TTFT C1 | 100 ms | ~375 ms |

Al-ENGR publishes 181 (pinned) to 189-196 (live) at knee C1 and 657-720 at C8: reproduced.

## 2. Per component (GPU kernel ms per generated token, same load, both GB300-only, FlashInfer MoE in both)

| Component | Upstream C1 | b12x C1 | Upstream C8 | b12x C8 |
| --- | ---: | ---: | ---: | ---: |
| dense linears (FP8 block-scaled) | 1.67 | 3.52 | 0.28 | 0.99 |
| attention | 0.57 | 0.88 | 0.09 | 0.16 |
| indexer | 0.16 | 0.23 | 0.02 | 0.04 |
| Engram | 0.02 | 0.49 | 0.00 | 0.12 |
| MoE | 2.59 | 3.75 | 0.88 | 1.61 |
| mHC | 0.80 | 0.64 | 0.13 | 0.16 |
| norm / quant / RoPE | 0.45 | 0.16 | 0.07 | 0.06 |
| total | 7.36 | 10.48 | 1.66 | 3.26 |

Upstream uses FlashInfer's CuTe split-K block-scaled GEMM (~8 us/call) where b12x's SM103 dense GEMM takes
17.4 us; FlashMLA sparse FP8 decode for attention; DeepGEMM + TileLang for mHC. Speculative settings differ
(upstream k=5/k=1 schedule, b12x k=3 adaptive), so small per-token differences are not meaningful.

## 3. mHC, the one kernel where b12x might win (identical shapes, one sublayer boundary, CUDA-graph replay)

| Tokens | Upstream (TileLang + DeepGEMM) | b12x tuned |
| ---: | ---: | ---: |
| 1 | 12.9 us | 8.6 us |
| 4 | 13.1 | 9.3 |
| 8 | 12.9 | 10.8 |
| 16 | 13.0 | 14.1 |
| 32 | 13.1 | 18.0 |
| 64 | 13.3 | 18.9 |

About 80 boundaries per step: ~0.3 ms saved per step at decode sizes (~2% at C1), a small loss from ~16 tokens.

## 4. The RTX PRO 6000 peer tier on top of upstream (decode only)

Al-ENGR's hook runs cold experts with FlashInfer streaming from Grace. The patched hook
(`upstream/hook-peer/`) instead sends a layer's cold routes (batches of up to 64 tokens) to a sidecar on the
6000 that holds the same 89 cold experts per layer in VRAM (66.9 GB) and computes them with a Triton GEMV while
the GB300 runs the hot experts. Only a few KB of activations per token cross PCIe; the weights never do.

Same image, same loaded autotune set; order peer run 1 → baseline → peer run 2.

| | Baseline (loaded) | Peer run 1 | Peer run 2 | Change |
| --- | ---: | ---: | ---: | ---: |
| catid C1 per user | 187.8 (accept 2.48) | 208.9 (2.53) | 216.9 (2.70) | +11-15% (partly acceptance) |
| catid C8 aggregate | 567.0 (1.70) | 754.3 (1.71) | 757.9 (1.71) | **+33%** |
| catid C16 aggregate | 798.5 (1.71) | 1,117.7 (1.71) | 1,153.6 (1.71) | **+40-44%** |
| knee C1 | 180.8 | 171.2 | 169.4 | **-6%** |
| knee C4 | 400.5 | 463.9 | 447.6 | +12-16% |
| knee C8 | 657.3 | 727.8 | 736.5 | +11-12% |
| knee C12 (2nd run) | 822.6 | 975.2 | 963.6 | +17-19% |

## 5. MegaMoE on the GB300 + the 6000 cold tier running b12x's SM120 MoE

Image `vllm/vllm-openai:nightly-7f1a5398…`, `--moe-backend deep_gemm_mega_moe`, a hook (`megamoe/hook/`) that
keeps only the hot experts in MegaMoE (shared expert fused) and never loads the cold ones into HBM; v20's other
flags. The cold experts live on the 6000.

**Where prefill time went.** A 16K-prefill profile with the cold experts streamed from Grace (TRT-LLM over
NVLink-C2C, as upstream does): 402 of 807 ms of GPU time was the cold stream (5.0 ms per layer per 8K chunk,
~334 GB/s off Grace), against 133 ms for the hot experts. The 64-token Triton peer could not take prefill: it
rereads an expert's weights per route (21.9 ms for 1,024 rows over 89 experts).

**b12x on the 6000.** b12x's `fused_moe` (w4a8_mx, SM120-native) reads each expert once per call: 1,024
compacted rows over one layer's 89 cold experts take 1.10 ms under CUDA-graph replay (~1.5 TB/s, near the
card's peak), 53-65 us for 1-4 rows. Peer v2 (`megamoe/hook/peer_tier2.py`, `ds41f-exp/peer/peer_server2.py`)
sends every batch size: the GB300 packs only the tokens that have a cold route into pinned host memory, the
6000 replays one graph per (layer, row bucket), and the results are added back, all without host syncs.

**Correctness.** An FP8-decode bug on the 6000 (activation bytes converted as integers) made every peer-v2
result before 19:50 UTC 2026-09-24 compute the cold experts on garbage; speed numbers from that period are
valid, quality was not. After the fix, the 6000's cold contribution matches an independent TRT-LLM computation
of the same experts on every layer (cosine 0.9993-0.9999, 2,043-token real prompt), and GSM8K-200 is 98.0% with
the 6000 against 98.5% with it switched off. b12x is not bitwise reproducible (the TRT-LLM path is), so greedy
outputs vary slightly run to run, mostly between formatting tokens.

**Placement: decode and prefill route to different experts.** Share of routes landing on cold experts:

| Hot/cold split | Decode, prose (D) | Decode, code/Q&A/reasoning (E) | Prefill, real text (A / B) | Random ids |
| --- | ---: | ---: | ---: | ---: |
| Al-ENGR static (295 hot) | 4.8% | 8.4% | 14.0% / 9.9% | 16.3% |
| calibrated on prefill-heavy text (295 hot) | 19.3% | 7.0% | 3.0% / 2.7%* | 18.5% |
| **balanced, 0.6 decode + 0.4 prefill (285 hot)** | **3.4%*** | **2.8%*** | **6.6% / 4.7%*** | 21.2% |

\* in-sample. Held-out checks: the prefill calibration built from A alone scores 6.6% on B; the balanced split
built without E scores 5.4% on E (static: 8.4%).

A prefill-only calibration cut prefill cold routes to 3% but pushed prose decode to 19-22% and knee C8 from ~770
to 518. The balanced split beats the static one on both phases while keeping 10 fewer experts on the GB300, which
frees ~7 GiB of HBM (KV cache 5.3 → 7.2 GiB at 95% utilization). Uniform random token ids concentrate on a subset
of experts unlike real text; tuning placement for them would inflate random-id benchmarks only.

**Result** (balanced split, 285 hot / 99 cold, one GB300 + one RTX PRO 6000):

| | v20 + 6000 (sec. 4) | MegaMoE + Grace cold (M3 v1) | **MegaMoE + 6000 via b12x (now)** | catid, 2× GB300 |
| --- | ---: | ---: | ---: | ---: |
| catid C1 per user | 209-217 | 213 | **212** | 248.5 (PP2) |
| catid C8 / C16 aggregate | 754-758 / 1,118-1,154 | 724 / 1,138 | **776 / 1,134** | — / 1,678 |
| knee C8 / C12 | 728-736 / 961 | 755-768 / 1,032 | **774 / 1,007** | |
| prefill 16K, random ids | 15.8K (v20 alone) | 21.5K | 24.0K | 35.9K (PP2, C1) |
| prefill 128K, random ids | | | 23.3K | 56.0K (PP2, C1) |
| prefill 16K / 64K, real text | | | **38.5-39.0K / 36.5-37.2K** | |
| GSM8K-200 | | | 98.0% | |

Random-id prefill is PCIe-bound: 21% of those routes are cold, so roughly three quarters of the tokens ship a
row to the 6000 and back. Real text sends 21-30% of tokens (measured).

Not yet separated: how much of M3's gain over v20 comes from MegaMoE and how much from the newer image
(the planned control is v20's TRT-LLM hook plus the 6000 on the same image).

## Conclusions

All for DeepSeek-V4.1-Flash's MXFP4-expert / FP8-dense / MXFP8-activation mix; other model and quant types are
not covered.

1. For this mix on GB300 (SM103), b12x's serving path is not worth pursuing: upstream vLLM on the same machine
   is 26-58% faster at decode and 3.2x at prefill.
2. For this mix, b12x expert residency on the GB300 is not worth pursuing: its operator loses to FlashInfer's
   TRT-LLM routed kernel, and the hot/cold idea is served upstream by Al-ENGR's hook.
3. Kernel by kernel on SM103, b12x loses on FP8 dense linears (2-3.6x), attention (~1.6x) and the indexer
   (1.4-2x). mHC is the only win (17-33% at 1-8 tokens, ~2% end to end at C1); not a project on its own.
4. b12x is worth pursuing on SM120 as a cold-expert engine: its MXFP4/MXFP8 fused MoE on the RTX PRO 6000 is
   what makes the 6000 usable for prefill, 20x faster than the Triton GEMV at prefill batch sizes. With it,
   one GB300 plus a 6000 reaches ~39K tok/s real-text prefill with decode on par with v20 plus the 6000
   (and +33-42% over v20 alone at C8/C16).
5. Expert placement must be calibrated on decode and prefill together; either alone makes the other phase worse.
6. Before any claim or upstreaming: broader quality gates (DSpark acceptance by class, BFCL tool calling,
   long-context retrieval, greedy parity against the 6000-off path), the MegaMoE-vs-image control, and
   calibration on real traffic.

## 6. Decode profile and the hook's send path (2026-09-28)

Image nightly-7f1a5398 → nightly-af7f9488 (same flags, effort pinned to 75 on both arms, `megamoe/ab-suite.sh`):
decode within noise (catid C1 222 → 226 per user), 16K prefill 25.8K → 26.9K, GSM8K-200 97.0 → 98.0%.

Nsight Systems with CUDA-graph node tracing (`PROF=nsys`, `megamoe/nsys_decode.py`) on reasoning decode:

| C1 step (6-token verify), 12.0 ms profiled | ms/step | note |
| --- | ---: | --- |
| MegaMoE (hot experts + shared) | 3.7 | ~86 us/layer, near HBM bandwidth for the experts touched |
| hook glue between router and MegaMoE | ~1.6 | ~25 PyTorch ops/layer: masks, counts, row-map split, any/cumsum/sum, casts |
| mHC (TileLang overlap path at T <= 16) | 1.8 summed | a side stream hides most of it; forcing the single-kernel path changed nothing |
| sparse attention (FlashMLA mega kernel) | 1.35 | ~30 us/layer, one CTA per query token; vLLM measured split-KV equal at small s_q |
| dense FP8/FP4 GEMMs | ~1.8 | |
| peer publish/wait/add | 0.5 | the actual stall on the 6000 is 0.13 ms/step |

At C8 (24 tokens) MegaMoE is 52% of an 18.6 ms step, at HBM bandwidth.

`megamoe/hook/peer_fused.py` replaces the glue plus `_pack`/`_publish` with one single-CTA kernel for T <= 64
(bit-identical on 176 cases, `megamoe/test_peer_fused.py`; 34-47 → 5-13 us per layer under CUDA graphs):

| catid decode (8K in, 1,024 out) | C1 per user | C4 | C8 | C16 | reasoning C1 | GSM8K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| af7f9488 | 225.8 | 579.5 | 931.9 | 1,328.1 | 307 | 98.0% |
| af7f9488 + fused send | **254.1** | **628.6** | **1,019.9** | **1,460.2** | **354** | 98.0% |
| catid, 2× GB300 (PP2 + DSpark) | 248.5 | | | 1,677.7 | | |

DSpark acceptance on reasoning text is 3.8 tokens/step at k=5 (catid 2.6, prose 2.3); the k schedule was tuned on
catid/prose.

DSpark draft-length schedule, re-swept after the fused send on closed-loop reasoning traffic (`megamoe/reason_bench.py`,
thinking on at `high`, T=1.0/top_p 0.95; `megamoe/sweep-k.sh`). DSpark's block is 5 tokens, so k tops out at 5.

| k by batch size (≤4 / 5-8 / 9-24) | reasoning C8 / C12 / C16 / C24 | catid C8 / C16 | prose knee C8 / C16 |
| --- | ---: | ---: | ---: |
| 5 / 2 / 1 (previous) | 969 / 1,059 / 1,258 / 1,516 | 1,001 / 1,434 | 936 / 1,408 |
| 5 / 3 / 2 | 1,023 / 1,174 / 1,376 / 1,564 | 1,040 / 1,551 | 883 / 1,435 |
| 5 / 4 / 3 | 1,022 / 1,192 / 1,354 / 1,591 | 1,014 / 1,589 | 840 / 1,431 |
| **5 / 3 / 2 + graph sizes 20/28/36/56/72/80 (production)** | **1,004 / 1,187 / 1,406 / 1,608** | **1,060 / 1,581** | 896 / 1,449 |

C1/C4 use k=5 in every arm (reasoning C1 270-286 tok/s is run-to-run noise at T=1). Padded graph tokens still route
through experts, so the finer capture sizes matter once k+1 stops dividing the old sizes (C24 × 3 = 72 was padded to 96).

## 7. Drafting, dense GEMMs and the hook's send/receive path (2026-09-28 evening)

Same image (af7f9488). Arms benchmarked with `megamoe/sweep-k.sh` (reasoning closed loop at T=1, catid, prose knee); raw
output in `megamoe/logs/sweep-probdraft.out` and `megamoe/logs/sweep-pd-k.out`.

**Kept (now the `launch-m3.sh` defaults):**
- `"draft_sample_method":"probabilistic"`. The drafter samples instead of taking its argmax, and rejection uses the
  full draft distribution. At T=1 acceptance of the first draft position goes 0.73 → 0.80; reasoning C1 +7%, C4 +12%,
  C8 +6%, C24 +6%. Greedy (T=0) and prose traffic are within noise.
- DSpark k schedule 5/3/3 (k=3 above 8 streams, was 2): reasoning C12 1,224 → 1,312, C16 1,432 → 1,504, C24 1,647 →
  1,747. 5/4/3 matched it on reasoning and cost catid/prose 2–3% at C8.
- `MEGA_LL_GEMM=1` (`megamoe/hook/ll_gemm.py`): MXFP8 dense linears with M ≤ 8 (wq_a+wkv, wq_b, wo_b) run on
  FlashInfer's split-K `cutedsl_low_latency` kernel instead of `cute-dsl`, whose persistent grid launches one CTA per
  128-wide N tile (40 CTAs for wo_b). Microbench at M=6: 25.6 → 20.4 µs per layer; C1 pass 10.03 → 9.96 ms.
- `MEGA_FUSED_SEND=2` (`megamoe/hook/peer_fused2.py`): the send quantizes the rows it packs itself and runs one CTA per
  token; wait + scatter-add are one launch. Bit-identical to the previous path (`megamoe/test_peer_fused2.py`, 176
  cases; the quantizer matches FlashInfer byte for byte). Microbench: −0.075 ms/step at 6 tokens, −0.2 at 24, −0.4 at
  48–64. C1 pass 9.96 → 9.87 ms.

| | Start of evening (5/3/2, greedy drafts) | Now |
| --- | ---: | ---: |
| reasoning C1 / C8 / C16 (T=1) | 287 / 1,024 / 1,435 | **306 / 1,102 / 1,535** |
| catid C1 per user / C16 | 254 / 1,558 | **266 / 1,613** |
| GSM8K-200 | 98.0% | 98.0% |

**Measured and dropped:**
- FlashMLA mega attention is 26.7 µs per layer whether it gets 1 or 96 query tokens: one CTA per token walks 640 keys
  in blocks of 64 at ~2.7 µs per block. Padding to the 128-head 2-CTA kernel saves ~4 µs per layer only at ≤ 12 tokens;
  an NVFP4 compressed cache does not change it; deepseek-ai/FlashMLA#227 (built for sm_103a in `flashmla-227/`) gives
  nothing at decode sizes because its overlaps need many queries per CTA. Splitting a token's keys across a CTA
  cluster remains the one large decode lever (~0.6 ms/step, C1–C24).
- MXFP8 at M > 8: TRT-LLM, cutlass and cuDNN are no faster than cute-dsl; a Triton split-K kernel
  (`megamoe/hook/mxfp8_splitk.py`, native block-scaled tcgen05 MMA) is correct but 25–60% slower (cp.async loads).

## 8. Decoder SWA bounded replay (vllm#58132, as an overlay; 2026-09-28 late)

vllm#58132 (open, approved by a maintainer, not merged) lets DS-V4.1's layers 21–39, which own only sliding-window KV
and take their long-range context from layer 20's compressed KV, run on each request's last 128 tokens in eager
prefill steps. It applies cleanly to af7f9488 (which already has the encoder side, #56227, and #58586). The patched
files live in `megamoe/overlay-58132/tree` and are bind-mounted by `swap-to-m3v2.sh` (default; `REPLAY_OVERLAY=0`
boots stock). The boot log line "Decoder SWA bounded replay ... layers 21-39" confirms it is on; it stays on with DSpark.

| | Before | With #58132 |
| --- | ---: | ---: |
| real-text prefill 16K (tok/s, TTFT) | 38.6K, 0.42 s | **60.5K, 0.27 s** |
| real-text prefill 64K | 37.2K, 1.76 s | **60.0K, 1.09 s** |
| catid random-id prefill 16K / 128K, C1 | 26.9K / 23.3K | **44.5K / 44.0K** |
| catid random-id prefill 16K / 64K / 128K, C4 | | 46.6K / 46.2K / 44.8K |
| reasoning decode C1 / C8 / C16 | 306 / 1,102 / 1,535 | 312 / 1,098 / 1,519 |
| catid C16 | 1,613 | 1,629 |
| GSM8K-200 | 98.0% | 98.0% |
| needle at 57K / 114K tokens (3 depths each) | 6/6 | 6/6 |

catid's two-GB300 PP2 numbers for comparison: 35.9K at 16K and 56.0K at 128K (C1).

The replay is an approximation at the window edge, not an exact rewrite. Greedy 48-token continuations of six 16K/64K
real-text prompts (`megamoe/longctx_check.py`, per-run `cache_salt`) differ from the stock server by no more than the
stock server differs from itself (the b12x cold tier is not bitwise reproducible): identical prefixes of 9–48 tokens
vs 1–48, all first-token top-5 sets overlapping in 4–5 of 5.
