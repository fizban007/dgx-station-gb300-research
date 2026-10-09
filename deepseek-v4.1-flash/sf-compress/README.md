# Compressed MoE weight scales for DeepSeek-V4.1-Flash M3

This directory keeps the routed experts' MXFP4 block scales (UE8M0, one byte per 32 weights) compressed in GPU
memory, on both GPUs of the [M3](../m3/) setup:

- **GB300 (MegaMoE hot experts):** our own fixed-slot codec (`hook/sf_compress.py`). The routed experts' scales are
  decoded into a shared scratch just before each MegaMoE call, inside the CUDA graphs.
- **RTX PRO 6000 (b12x cold-expert sidecar):** b12x's built-in MXFP4-CSF support. We encode the planes at load
  (`sidecar/csf_encode.py`), and b12x expands the routed experts per call.

Both are lossless: the decoded scales are bit-identical to the raw ones. The freed memory goes to KV cache on the
GB300 and to more cold experts on the sidecar, which in turn frees HBM on the GB300.

It runs on the official DeepSeek-V4.1-Flash checkpoint (MIT) and compresses the scales when the model loads. It
does not use LIL's lossless-CSF checkpoint or any of its files.

Date: 2026-10-07/08. Image `vllm/vllm-openai:nightly-af7f9488`, b12x master `2cc7f66a`.

## Results

| | before | after |
|---|--:|--:|
| GB300 hot-expert scales (258 hot/layer) | 10.63 GiB | 1.75 GiB + 0.27 GiB shared scratch (**-8.61 GiB**, 9.25 GB) |
| Sidecar cold-expert scales (130 cold/layer) | 5.36 GiB | 0.73 GiB + 0.13 GiB shared scratch (7.3x smaller) |
| GB300 KV cache at GPU_UTIL 0.9, 258 hot ([phase 1](DETAILS.md#phase-1-gb300-compressed-scales-with-just-in-time-decode)) | 6.99M tokens | 11.41M tokens (+63%) |
| Hot/cold split | 258 / 126 | 254 / 130 (the sidecar's savings pay for 4 more cold experts per layer) |
| GB300 KV cache at GPU_UTIL 0.87, 254 hot | 7.53M tokens (raw scales, 258 hot) | 8.98M tokens, with ~34 GiB of the GB300 left free |

Quality:
- In-server check (`MEGA_SF_CHECK=6`): 6/6 bit-exact against the raw scales.
- GSM8K-200, effort 50, C16:
  - Phase 1 alone: 97.0% uncompressed vs 97.5% compressed. Outputs aren't run-to-run deterministic on this stack
    (probabilistic DSpark drafting), so this is within noise.
  - Phases 1 + 3 with the 254/130 split: 98.5%.

Speed:
- Phase 1, C1 decode: 212 -> 198 tok/s (-6.7%).
- Phase 1, GSM8K C16: 1051 -> 1020 tok/s (-3%).
- The sidecar decode adds 1-3 us per call.
- With the 254/130 split, C1 is unchanged (194-206 tok/s). C16 GSM8K drops to 940 tok/s, because 4 more cold
  experts per layer go over the sidecar link.

The GB300 deployment now runs at GPU_UTIL 0.832 (4.07M KV tokens), leaving ~43.7 GiB of HBM for a video model.

**Phase 2** decoded the scales inside the MegaMoE kernel instead of in a separate pass. It was bit-exact but
1.5-5x slower and is paused. The patch, ablations and analysis are kept in [phase2/](phase2/).

## How it works

[DETAILS.md](DETAILS.md) has the full design. In short:

- **GB300.** Each 512 B chunk that MegaMoE's SFB loader would load (one expert, 128-row N-block, 128-wide K-block)
  is stored as a fixed 80 B slot:
  - a 4 B header
  - a 64 B bit plane (each scale is `base` or `base+1`)
  - up to 3 inline exceptions
  
  Chunks with more exceptions (~1%) spill to a raw overflow tile. Before each MegaMoE call, a one-CTA kernel lists
  the routed hot experts from `hot_ids` on the GPU. A Triton kernel then decodes only those experts into one
  scratch per shape, shared by all 40 layers. There's no host sync, so this is captured into vLLM's CUDA graphs.
- **Sidecar.** We encode b12x's `CsfScalePlanes`: per 16-row slab, 16 row-base bytes followed by selector bits,
  plus sorted `pos | value << 24` exceptions. b12x's `Mxfp4CsfWeights` path expands the routed experts into a
  scratch pair shared by all layers, inside its captured graphs. The server swaps in the raw path's scalar unit
  activation scales (`immutable_input_scales=True`), so the outputs match the raw path.

## Layout

| path | what |
|---|---|
| `hook/sf_compress.py` | GB300 codec: GPU encoder, Triton decode and expert-list kernels |
| `hook/rowmap-mix-h254.json`, `hook/rowmap-mix-h258.json` | 254- and 258-hot rowmaps (same mix-v1 weighting as `rowmap-mix-v1.json`) |
| `tools/make_mix_rowmap.py` | generates `rowmap-mix-h<N>.json` for any hot-expert count from `../m3/calibration` |
| `sidecar/csf_encode.py` | E8M0 scales -> b12x `CsfScalePlanes` encoder |
| `patches/mega_peer_hook.py.diff` | `MEGA_SF_COMPRESS` (default 1) and `MEGA_SF_CHECK=N` in the M3 hook |
| `patches/peer_server2.py.diff` | `PEER_CSF` (default 1) in the sidecar server |
| `patches/launch-m3.sh.diff` | passes `MEGA_SF_COMPRESS` into the container |
| `install.sh` | copies the new files into `../m3` and applies the patches |
| `phase0/` | compressibility study over the real checkpoint (`phase0-h258.json` = results) |
| `phase1/` | offline bit-exactness test, server A/B scripts (greedy outputs, prompt logprobs, GSM8K) and their results |
| `phase2/` | in-kernel decode: DeepGEMM header patch, ablation patches, test harness |
| `phase3/` | sidecar test (raw vs CSF through b12x on real cold experts) and results |
| `notes/megamoe-sf-path.md` | how MegaMoE's weight scales flow from checkpoint to TMEM (read-only analysis of the kernel) |
| `notes/csf-codec.md` | LIL's MXFP4-CSF format and how b12x/vLLM decode it (research that led to phase 3) |

## Install

The patches are written against this fork's `deepseek-v4.1-flash/m3` (commit `main` as of 2026-10-08). From this
directory:

```bash
./install.sh            # copies sf_compress.py, csf_encode.py and the rowmaps into ../m3, applies patches/
./install.sh --check    # only tests whether the patches apply
```

Then boot M3 as usual. Both features are on by default:
- `MEGA_SF_COMPRESS=0` keeps raw scales on the GB300.
- `PEER_CSF=0` keeps raw scales on the sidecar.

To use the 254/130 split, pass `ROWMAP=rowmap-mix-h254.json` to `swap-to-m3v2.sh`, which forwards it to both the
sidecar and M3. Check the boot logs for:

```
MEGA_PEER SF compress: 10.46 GiB raw -> 1.72 GiB compressed + 0.26 GiB shared scratch; HBM free 67.8 GiB
CSF scales: 5.36 GiB raw -> 0.73 GiB compressed + 0.13 GiB shared scratch
```

Requirements:
- `sf_compress.py` needs Triton and the vLLM image's DeepGEMM layout. It checks that the SF tensor has the
  MN-major strides MegaMoE uses.
- `csf_encode.py` needs b12x with `fm.Mxfp4CsfWeights`. That's master `2cc7f66a` (PR #481), now also in FlashInfer
  as `flashinfer/experimental/b12x`.

## Tests

| test | how | result |
|---|---|---|
| GB300 codec, offline | `phase1/test_sf_compress.py`, inside the vLLM image with `hook/` at `/w` | bit-exact on layers 0/20/37/38/39, all experts and routed subsets |
| GB300 codec, in server | boot with `MEGA_SF_CHECK=6` | 6/6 bit-exact |
| server A/B | `phase1/eval_compare.py`, `logprob_check.py`, `gsm8k_eval.py` (set `SERVER=http://host:port`) | see `phase1/*.json` |
| sidecar | `CUDA_VISIBLE_DEVICES=<6000> python phase3/test_sidecar_csf.py` (`B12X_SRC` = b12x checkout) | decoded scales and prepared weights byte-identical; fused-MoE error equals the raw-vs-raw noise floor at m = 1..256 |
| in-kernel decode | `phase2/run_test.sh --experts 96 --max-tokens 2048` | bit-exact, slower (see phase 2) |
