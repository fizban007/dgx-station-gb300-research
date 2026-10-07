# DeepSeek-V4.1-Flash: MegaMoE hot experts on the GB300, cold experts on an RTX PRO 6000 ("M3")

DeepSeek-V4.1-Flash has 384 routed MXFP4 experts in each of its 40 MoE layers. In this lane the GB300
keeps the most-used experts of each layer in HBM and runs them with DeepGEMM MegaMoE, with the shared expert
fused in. Since 2026-10-02 that is 265 per layer; it was 285 until then. The rest live only on the RTX PRO 6000
sidecar, which runs them with b12x: 119 per layer, 4,760 in all (3,960 at 285 hot). No routed expert is read
from Grace memory.

## What we found

- **Where it stands (2026-10-03): the lane shares the GB300 with a video model.** MiniMax-H3 (text/image to
  video with audio, on vllm-omni) now runs on the same GPU. To make room, the lane:
  - keeps 265 instead of 285 hot experts per layer;
  - reads its Engram tables in NVFP4, which pins 103 GiB of Grace memory instead of 264;
  - stores its KV cache as `nvfp4_ds_mla`;
  - runs at 0.885 GPU memory utilization ([`swap-to-ds41-h3.sh`](swap-to-ds41-h3.sh)).

  It still holds 3.42M KV tokens, against 3.53M before the change. A 15 s H3 render next to it peaks at
  246.6 GiB of the GPU's 250.7.
  - Reasoning decode on the NVFP4-KV boot: 304 / 1,071 / 1,470 tok/s at C1 / C8 / C16.
  - Random-id prefill: ~40K tok/s from 16K to 128K, about 11% under the 285-hot lane.
  - Quality: GSM8K-200 98.5%; needles 3/3 at 125K, 500K and 1M tokens; GPQA-Diamond 90.4% at T=1.
  - Details: [DETAILS](DETAILS.md#2026-10-0203-sharing-the-gb300-with-minimax-h3).
- **NVFP4 KV turned out to be free, for capacity.** At the same 5.89 GiB of KV memory it holds 4.34M tokens,
  against 3.05M for FP8 (+42%). Decode, prefill, GSM8K, needles and teacher-forced perplexity all match FP8.
  On 2026-09-28 it was dropped because it doesn't make attention faster, and that is still true.
- **Greedy decoding undersells the model on GPQA.** At T=0, 18 of 198 GPQA-Diamond answers loop until the
  131K-token cap, and the score lands at 84.8–87.4%. At T=1 none do, and the score is 92.4% (FP8 KV). Answers
  that don't loop score about 93% either way. Which questions loop changes from run to run.
- **32 sequences and pinned FlashInfer tactics (2026-09-29).** Two boots that each autotuned live disagreed on
  58 of 105 dense-GEMM tactics, which made earlier A/Bs noisy by 1–4%. With tactics pinned, boots agree within
  ~1%. Going from 24 to 32 sequences took catid C32 from 1,898 to 2,303–2,319 tok/s and its TTFT from 1.9 s to
  0.31 s; C1–C24 are unchanged. See [DETAILS](DETAILS.md#2026-09-29-32-sequences-and-pinned-flashinfer-tactics).
- **On 2026-09-28** the lane reached, on catid's decode recipe, 266 tok/s per user at C1 and 1,613–1,629
  aggregate at C16. On reasoning traffic (thinking on, T=1) it reached 306–312 tok/s at C1, ~1,100 at C8 and
  1,519–1,535 at C16. Prefill ran ~60K tok/s on real text at 16K–64K and 44–47K on random token ids.
  GSM8K-200 was 98.0%, and needles were found at 57K and 114K tokens.
- **What moved it from 2026-09-24 to 2026-09-28:**
  - fusing the hook's per-layer glue into one kernel: catid C1 226 → 254 per user;
  - probabilistic DSpark drafts plus a retuned k schedule: reasoning +6–12% at C1–C8, +5–9% at C12–C24;
  - vllm#58132, which runs layers 21–39 only on the last 128 tokens of each prefill chunk: prefill +57–61%.

  Smaller gains came from a split-K kernel for decode-sized dense GEMMs and a second fused send path. See
  [DETAILS](DETAILS.md#2026-09-28-image-upgrade-and-decode-profile).
- **The sidecar is what makes MegaMoE worthwhile on one GB300.** The 2026-09-24 configuration reaches, on catid's
  decode recipe, 211 tok/s at C1 (219 per user), 584 at C4, 926 at C8 and 1,333 at C16. That is 1.6× Al-ENGR's
  v20 recipe at C8 and C16 when we reproduce v20 on the same box. Prefill is 1.5–1.9× faster (23–30K vs 15.8K tok/s at 16K).
- **Quality is unchanged within noise.** GSM8K-200 scores 98.0% with the sidecar. With the sidecar off, the
  GB300 computes the cold experts itself from Grace, and scores 98.5%. An in-server cross-check of the sidecar
  against TRT-LLM gave cosine 0.9993–0.9999 on every layer (session notes).
- **The GB300 barely waits.** At C8 and C16 it waits 0.07–0.14 ms per forward pass (under 1%). The sidecar
  answers a 1-row call in 74 µs and a 16-row call in 155 µs.
- **Decode and prefill route to different experts.** A hot set calibrated on prefill-heavy text raised decode
  cold routes from 4.8% to 19–22% and cut C8 by a third. The final hot set (`rowmap-mix-v1`) mixes 60% decode
  and 40% prefill calibration.
- **Two bugs taught us to test the exact serving path.** One was a dequant bug that made early quality
  results garbage. The other was an autotuner run on empty inputs that picked slow kernels. Both are
  described in [DETAILS](DETAILS.md#bugs-worth-knowing-about).

## Results

The catid decode recipe uses 8,192 input tokens, 1,024 forced output tokens and temperature 0. "Agg" is
aggregate tok/s; "user" is the p50 per-user decode rate. Prefill is one cold 16K prompt (C1), with random token ids.

| configuration | C1 agg / user | C4 agg | C8 agg | C16 agg | prefill 16K | GSM8K-200 |
|---|--:|--:|--:|--:|--:|--:|
| Al-ENGR v20, reproduced here (HBM + Grace, FlashInfer TRT-LLM) | 193 / 203 | — | 583 | 821 | 15.8K | — |
| v20 + our RTX PRO 6000 tier (peer v1, cold experts for T ≤ 64) | 209 / 217 | — | 758 | 1,154 | — | — |
| M3, MegaMoE + Triton sidecar for T ≤ 64 (peer v1), cold prefill from Grace | 205 / 213 | — | 724 | 1,138 | 21.5K | — |
| M3 + b12x sidecar for every batch size (peer v2), 295 hot | 195 / 201 | — | 724 | 1,158 | 29.6K | 98.0%¹ |
| peer v2, 285 hot, `rowmap-mix-v1` | 207 / 212 | — | 776 | 1,134 | 24.0K | 98.0% |
| + sidecar retuned on realistic rows (Phase C control) | 205 / 211 | 575 | 836 | 1,329 | — | — |
| + DSpark k schedule 5/2/1 (final on 2026-09-24) | 211 / 219 | 584 | 926 | 1,333 | ~24K² | 98.0%² |
| *2026-09-28:* nightly-af7f9488, effort pinned 75 (`ab-new-af7f`) | 214 / 226 | 580 | 932 | 1,328 | 26.9K | 98.0% |
| + fused send v1 (`ab-fused-af7f`) | 239 / 254 | 629 | 1,020 | 1,460 | 26.9K | 98.0% |
| + k 5/3/2 and finer CUDA graphs (`k-prod-k532cg`) | — | — | 1,060 | 1,581 | — | — |
| + probabilistic drafts, LL GEMM, fused send v2, k 5/3/3 (`prod-20260928b`) | 254 / 266 | — | — | 1,613 | — | 98.0% |
| + vllm#58132 decoder SWA replay (`replay58132`) | 242 / 258 | — | — | 1,629 | 44.5K | 98.0% |
| *2026-09-29:* 32 sequences, pinned FlashInfer tactics (`arm-s32-pin-b`) | 244 / 251 | 644 | 1,052 | 1,604 | 44.3K | 97.5% |

¹ Measured after the dequant fix, with the same 295-hot rowmap (`gsm8k-peer2-fixed-static`). The speed
row itself predates that fix; see DETAILS.
² Only the speculative-decoding schedule changed after `mix-v1`, so prefill and GSM8K carry over from that row.
Neither was re-measured.

The 2026-09-28 changes target reasoning traffic and prefill, which catid's recipe (temperature 0, random ids)
does not show well. On the same boots:

| | reasoning C1 / C8 / C16 (T=1) | real-text prefill 16K / 64K (tok/s, TTFT) |
|---|--:|--:|
| 5/3/2, greedy drafts (`k-base-rerun`) | 287 / 1,024 / 1,435 | — |
| `prod-20260928b` | 306 / 1,102 / 1,535 | 38.6K, 0.42 s / 37.2K, 1.76 s |
| `replay58132` (current) | 312 / 1,098 / 1,519 | 60.5K, 0.27 s / 60.0K, 1.09 s |

catid C1 differs between `prod-20260928b` and `replay58132` (266 vs 258 per user) because acceptance differed
between runs (2.67 vs 2.56); their C1 pass times are the same. Reasoning uses [`tools/reason_bench.py`](tools/reason_bench.py).

Between the 295-hot and 285-hot rows, two things changed: 285 instead of 295 hot experts, and 0.95 instead of
0.97 GPU memory utilization. Together they grew the KV cache from 5.3 to 7.2 GiB (session notes). Over the
same change, prefill fell from about 29.5K to 23.5K tok/s. The effect of each change was not isolated.

**Sharing the GB300 with MiniMax-H3 (2026-10-02/03).** These columns cover what the move had to keep:
- KV capacity;
- reasoning decode;
- prefill on random token ids;
- quality;
- whether a 15 s, 30-step 1024×576 H3 render fits beside the lane.

All rows have 32 sequences and pinned tactics.

| configuration | KV tokens | reasoning C1 / C8 / C16 | prefill 16K / 128K | GSM8K-200 | GPQA-D (T=1) | 15 s H3 render, GB300 peak |
|---|--:|--:|--:|--:|--:|--|
| 285 hot, FP8 Engram, FP8 KV, util 0.95 (`arm-s32-pin-b`, 2026-09-29) | 3.53M | 314 / 1,082 / 1,487 | 44.3K / — | 97.5% | — | no room for H3 |
| 265 hot, NVFP4 Engram, FP8 KV, util 0.92 (`h265-nvfp4e-u92`) | 6.93M | 296 / 1,082 / 1,493 | 39.6K / 39.4K | 98.0% | 92.4% | 249.8 GiB, out of memory |
| the same at util 0.89 (`h265-nvfp4e-u89`) | 3.05M | 304 (C1) | — / 40.2K | — | — | 245.0 (lean H3), 247.8 (H3 with voice cloning) |
| + `nvfp4_ds_mla` KV (`h265-nvfp4e-nvfp4kv-u89`) | 4.34M | 304 / 1,071 / 1,470 | 39.3K / 39.7K | 98.5% | 90.4% | 248.6 |
| **+ util 0.885 (current, `ds41-h3-u885`)** | **3.42M** | — | — / 39.7K | — | — | **246.6** |

During a render, DS41's C1 reasoning decode drops from ~304 to 152–183 tok/s. A 15 s render takes 238–258 s next
to the lane. GPQA's T=1 difference between FP8 and NVFP4 KV is within noise (9 vs 5 flipped answers, McNemar
p = 0.42). The util-0.89 FP8 row's C1 comes from a run with no H3 render, and its 128K prefill and the 0.885 row's
come from the lanes' warm-up. Each test, with its log, is in
[DETAILS](DETAILS.md#2026-10-0203-sharing-the-gb300-with-minimax-h3).

Sources: [`../results/runs/`](../results/runs/) (`up-v20`, `up-v20-peer2`, `m3`, `m3v2`, `mix-v1`,
`phaseC-C0-control`, `phaseC-C4-k2-to-8`, for 2026-09-28 `ab-*`, `k-*`, `prod-20260928b`, `replay58132`,
and for 2026-09-29 `arm-*`), [`results/logs/`](results/logs/) (2026-09-29 in `seats32/`, 2026-10-02/03 in
`h3-coexist/`). GPQA-Diamond appears only as aggregate scores; per-item outputs are kept out of this repository.
The `replay58132` real-text 16K figure excludes the first request after boot
(35.2K tok/s with one-time JIT warm-up). Every number, with its source file,
is in [`results.jsonl`](results.jsonl); [DETAILS](DETAILS.md) has every intermediate run.

## How it works

Per MoE layer, for any batch size:

1. The GB300 routes tokens (top-6) and packs only the tokens with a cold route into a pinned shared host buffer
   (`/dev/shm`). Each packed row holds MXFP8 activations, local cold expert ids and route weights. The GB300
   then publishes a sequence number.
2. It runs MegaMoE on the hot experts plus the shared expert.
3. It waits for the sidecar's acknowledgement and adds the returned rows into their tokens.

The sidecar polls the buffer, dequantizes the rows, and replays a pre-captured CUDA graph for that
(layer, row bucket). The graph runs b12x's `w4a8_mx` fused MoE over the layer's 99 cold experts. The sidecar
then writes the results back. The GB300 side is all device kernels with no host sync, so it lives inside
vLLM's CUDA graphs. If the wait times out, the layer drops the cold contribution and increments a counter.

The protocol is described in [`../../docs/sidecar-peer-tier.md`](../../docs/sidecar-peer-tier.md). The same
sidecar design, ported, serves MiMo-V2.6-Pro in [`../../mimo-v2.6-pro/`](../../mimo-v2.6-pro/).

## Files

| path | what |
|---|---|
| [`swap-to-ds41-h3.sh`](swap-to-ds41-h3.sh) | **the current lane:** `swap-to-m3v2.sh` with rowmap h265, the NVFP4 Engram mounts, `nvfp4_ds_mla` KV and util 0.885 |
| [`swap-to-m3v2.sh`](swap-to-m3v2.sh) | starts the sidecar, waits for `serving`, then boots M3 through `launch-m3.sh` with the vllm#58132 overlay and pinned FlashInfer tactics. Its defaults are the 285-hot lane, which leaves no room for H3 |
| [`launch-m3.sh`](launch-m3.sh) | the vLLM command: nightly af7f9488, `--moe-backend deep_gemm_mega_moe`, DSpark 5/3/3 with probabilistic drafts, 32 sequences, Engram CPU offload |
| [`engram-nvfp4/`](engram-nvfp4/README.md) | the overlay that serves the Engram tables in NVFP4: diff against the image, mounts, merged checkpoint view, tests |
| [`hook/fi-tune/`](hook/fi-tune/) | the pinned FlashInfer tactic set (`MEGA_FI_TUNE_FILE`) |
| [`tools/coexist/`](tools/coexist/) | the 2026-10-02/03 test chains: H3 coexistence, NVFP4 KV A/B, the 0.885 relaunch, the idle gate, and the H3 launcher |
| [`tools/ds41_numerics.py`](tools/ds41_numerics.py), [`tools/needle_test.py`](tools/needle_test.py) | FP8 vs NVFP4 KV numerics; needles up to 1M tokens |
| [`overlay-58132/`](overlay-58132/README.md) | the vllm#58132 diff and how to rebuild the overlay the swap script mounts |
| [`hook/peer_fused2.py`](hook/peer_fused2.py), [`hook/peer_fused.py`](hook/peer_fused.py) | fused decode send/receive (v2 current, v1 before) |
| [`hook/ll_gemm.py`](hook/ll_gemm.py) | routes MXFP8 dense linears with M ≤ 8 to FlashInfer's split-K low-latency GEMM |
| [`results/microbench/`](results/microbench/README.md) | 2026-09-28 kernel microbenchmarks: dense GEMM backends, attention variants, fused send |
| [`sidecar/peer_server2.py`](sidecar/peer_server2.py) | the RTX PRO 6000 sidecar (b12x `w4a8_mx`, one CUDA graph per layer and bucket) |
| [`hook/peer_tier2.py`](hook/peer_tier2.py) | the shared-buffer protocol and graph-safe pack/wait/add kernels |
| [`hook/mega_peer_hook.py`](hook/mega_peer_hook.py), [`hook/sitecustomize.py`](hook/sitecustomize.py) | installs the hot/cold split into vLLM's DeepSeek-V4 MoE at import time |
| [`hook/rowmap-*.json`](hook/) | hot/cold expert lists per layer: `mix-v1-h265` (current, 265 hot), `mix-v1` (285 hot), `mix-v1-h270`/`-h260` (alternatives), `static-v1` (Al-ENGR's, verbatim), `cal-v2-h285` (prefill-calibrated) |
| [`calibration/`](calibration/) | route counts that the `mix-v1` rowmaps are built from ([`tools/build_mix_rowmap.py`](tools/build_mix_rowmap.py)). A, B = prefill; D, E = decode. |
| [`tools/`](tools/) | calibration, rowmap building, wait measurement, microbenchmarks, GSM8K and profiling helpers |
| [`DETAILS.md`](DETAILS.md) | every run, the Phase C DSpark sweep, microbenchmarks, bugs and gotchas |

The scripts are verbatim snapshots and hard-code station paths. `/home/jasonc/research/megamoe` is this
directory (its test chains are in `tools/coexist/`), `/home/jasonc/ds41f-exp/peer` is `sidecar/`, and the H3
launcher's `/home/jasonc/research/minimax-h3` holds the MiniMax-H3 checkpoints and prompts, which are not included.
They also hard-code the GPU UUIDs and the host
venvs. The sidecar ran against [b12x](https://github.com/local-inference-lab/b12x) branch `exp/ds41f-gb300`
at `8fdb5635`. That commit is unpublished, but the fused-MoE API the sidecar calls is on b12x master, which
the MiMo-Pro sidecar uses.
