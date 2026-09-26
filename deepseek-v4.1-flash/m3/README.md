# DeepSeek-V4.1-Flash: MegaMoE hot experts on the GB300, cold experts on an RTX PRO 6000 ("M3")

DeepSeek-V4.1-Flash has 384 routed MXFP4 experts in each of its 40 MoE layers. In this lane the GB300
keeps the 285 most-used experts of each layer in HBM and runs them with DeepGEMM MegaMoE, with the shared
expert fused in. The other 99 per layer, 3,960 experts in all, live only on the RTX PRO 6000 sidecar,
which runs them with b12x. No routed expert is read from Grace memory.

## What we found

- **The sidecar is what makes MegaMoE worthwhile on one GB300.** The final configuration reaches, on catid's
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

| configuration (2026-09-24) | C1 agg / user | C4 agg | C8 agg | C16 agg | prefill 16K | GSM8K-200 |
|---|--:|--:|--:|--:|--:|--:|
| Al-ENGR v20, reproduced here (HBM + Grace, FlashInfer TRT-LLM) | 193 / 203 | — | 583 | 821 | 15.8K | — |
| v20 + our RTX PRO 6000 tier (peer v1, cold experts for T ≤ 64) | 209 / 217 | — | 758 | 1,154 | — | — |
| M3, MegaMoE + Triton sidecar for T ≤ 64 (peer v1), cold prefill from Grace | 205 / 213 | — | 724 | 1,138 | 21.5K | — |
| M3 + b12x sidecar for every batch size (peer v2), 295 hot | 195 / 201 | — | 724 | 1,158 | 29.6K | 98.0%¹ |
| peer v2, 285 hot, `rowmap-mix-v1` | 207 / 212 | — | 776 | 1,134 | 24.0K | 98.0% |
| + sidecar retuned on realistic rows (Phase C control) | 205 / 211 | 575 | 836 | 1,329 | — | — |
| **+ DSpark k schedule 5/2/1 (final)** | **211 / 219** | **584** | **926** | **1,333** | ~24K² | 98.0%² |

¹ Measured after the dequant fix, with the same 295-hot rowmap (`gsm8k-peer2-fixed-static`). The speed
row itself predates that fix; see DETAILS.
² Only the speculative-decoding schedule changed after `mix-v1`, so prefill and GSM8K carry over from that row.
Neither was re-measured.

Between the 295-hot and 285-hot rows, two things changed: 285 instead of 295 hot experts, and 0.95 instead of
0.97 GPU memory utilization. Together they grew the KV cache from 5.3 to 7.2 GiB (session notes). Over the
same change, prefill fell from about 29.5K to 23.5K tok/s. The effect of each change was not isolated.

Sources: [`../results/runs/`](../results/runs/) (`up-v20`, `up-v20-peer2`, `m3`, `m3v2`, `mix-v1`,
`phaseC-C0-control`, `phaseC-C4-k2-to-8`), [`results/logs/`](results/logs/). Every number, with its source file,
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
| [`swap-to-m3v2.sh`](swap-to-m3v2.sh) | starts the sidecar, waits for `serving`, then boots M3 through `launch-m3.sh` |
| [`launch-m3.sh`](launch-m3.sh) | the vLLM command: nightly 7f1a5398, `--moe-backend deep_gemm_mega_moe`, DSpark, Engram CPU offload |
| [`sidecar/peer_server2.py`](sidecar/peer_server2.py) | the RTX PRO 6000 sidecar (b12x `w4a8_mx`, one CUDA graph per layer and bucket) |
| [`hook/peer_tier2.py`](hook/peer_tier2.py) | the shared-buffer protocol and graph-safe pack/wait/add kernels |
| [`hook/mega_peer_hook.py`](hook/mega_peer_hook.py), [`hook/sitecustomize.py`](hook/sitecustomize.py) | installs the hot/cold split into vLLM's DeepSeek-V4 MoE at import time |
| [`hook/rowmap-*.json`](hook/) | hot/cold expert lists per layer: `mix-v1` (final), `static-v1` (Al-ENGR's, verbatim), `cal-v2-h285` (prefill-calibrated) |
| [`calibration/`](calibration/) | route counts that `mix-v1` was built from. A, B = prefill; D, E = decode. |
| [`tools/`](tools/) | calibration, rowmap building, wait measurement, microbenchmarks, GSM8K and profiling helpers |
| [`DETAILS.md`](DETAILS.md) | every run, the Phase C DSpark sweep, microbenchmarks, bugs and gotchas |

The scripts are verbatim snapshots and hard-code station paths. `/home/jasonc/research/megamoe` is this
directory, and `/home/jasonc/ds41f-exp/peer` is `sidecar/`. They also hard-code the GPU UUIDs and the host
venvs. The sidecar ran against [b12x](https://github.com/local-inference-lab/b12x) branch `exp/ds41f-gb300`
at `8fdb5635`. That commit is unpublished, but the fused-MoE API the sidecar calls is on b12x master, which
the MiMo-Pro sidecar uses.
