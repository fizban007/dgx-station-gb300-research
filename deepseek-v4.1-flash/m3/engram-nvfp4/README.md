# NVFP4 Engram tables for the M3 lane

DeepSeek-V4.1-Flash has two Engram n-gram tables, in layers 1 and 14, each with 384M rows of 256 values.
[`aidendle94/DeepSeek-V4.1-Flash-NVFP4-Engram`](https://huggingface.co/aidendle94/DeepSeek-V4.1-Flash-NVFP4-Engram)
(MIT) re-quantizes them from FP8 (E4M3 with UE8M0 block-32 scales) to NVFP4:
- e2m1 values, two per byte;
- one E4M3 scale per 16 values;
- one fp32 global scale per table.

That takes the tables from 203 GB to 111 GB; every other tensor is byte-identical to the original. Stock vLLM can't
read the format, so this directory adds a two-file overlay for the lane's image (`nightly-af7f9488`). Like
[`../overlay-58132/`](../overlay-58132/README.md), it is bind-mounted over the image, with no image build.

## What we found

- **Host memory pinned for Engram: 264 GiB → 103.0 GiB.**
  - The saving is larger than 189 → 103 because of torch's pinned allocator. torch 2.13 backs
    `torch.empty(..., pin_memory=True)` with the CachingHostAllocator, which rounds every block up to a power
    of two (`pinned_max_round_threshold_mb` is unlimited by default). A 1.10 GiB request pins 2.00 GiB.
  - The stock FP8 offload pins each table that way, so its 188.8 GiB of tables (TP1, both layers) take
    2 × (128 + 4) = 264 GiB.
  - The overlay registers one exact-size `mmap` block per table with `cudaHostRegister`: 110,595,290,400 B =
    103.0 GiB, with nothing in the caching allocator.
  - Two ways to stop the rounding for FP8, both untested at full size: `{"cpu_offload": true, "use_thp": true}`,
    or `PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:512`. The second is verified at small scale: a
    1.10 GiB request pins 1.10 GiB.
- **Lookups are exact to the format.** The GPU gather is bit-identical to a CPU reference dequantization over
  139K real rows. NVFP4 differs from the FP8 rows by 8.99% (layer 1) and 9.06% (layer 14) relative RMS,
  matching the uploader's report. Row cosine has a median of 0.996 and a minimum of 0.9945.
- **Lookups are faster on the RTX PRO 6000 over PCIe:** 1.14–1.62× at every batch shape (table below). On
  the GB300, which reads over C2C, they weren't timed separately.
- **The FP8 path is unchanged.** 15 of 15 output hashes match the stock image, and timing is within run-to-run
  noise.
- **The first GB300 boots (2026-10-02) worked.**
  - Both tables log `Engram NVFP4 table offloaded to registered host memory: ... 51.50 GiB per rank`, and no
    `pin_memory` fallback appears.
  - The lane passed GSM8K-200 (98.0%), needles at 108K / 434K / 868K tokens and GPQA-Diamond at T=1 (92.4%). See
    [`../DETAILS.md`](../DETAILS.md#2026-10-0203-sharing-the-gb300-with-minimax-h3).

## What the overlay changes

[`overlay.diff`](overlay.diff) adds 337 lines and removes 12, against the stock `af7f9488` files.

| file | change |
|---|---|
| `vllm/models/deepseek_v41/common/engram.py` | `parse_engram_quant()`. `EngramLayout` gains `quant`, `quant_block_size` and `global_scales`, from the config keys `engram_quant`, `engram_quant_block_size` and `engram_quant_global_scale`. NVFP4 parameters, a weight loader with a dtype guard, and `_engram_nvfp4_lookup_kernel`; `lookup()` dispatches on the format. |
| `vllm/models/deepseek_v41/nvidia/engram.py` | NVFP4 host storage: one exact-size registered block per table, values then scales. It is THP-backed with `use_thp` and falls back to `pin_memory`. Format arguments pass through `Engram._create_embedding`, and `can_share_engram_tables` counts bytes per format. NVFP4 with `dp_shared_memory` raises NotImplementedError. |

- **Detection:** `config.engram_quant == "nvfp4"` selects NVFP4. Without the key, FP8 construction is the stock
  call, argument for argument.
- **Storage and loading:** the parameter names don't change, so the lane's weights mapper routes the tensors as
  before.
  - The values are U8 `[rows, 128]` (low nibble = even element) and the scales are F8_E4M3 `[rows, 16]`.
  - The loader rejects FP8 shards under an NVFP4 config, and the reverse, with a clear message.
- **Lookup:** the kernel keeps the FP8 kernel's grid, row walk, head masking and invalid-id handling.
  - Per row it does one 128 B load and one 16 B load, and decodes with vLLM's `_e2m1_inline`.
  - The arithmetic is `(e2m1 × e4m3) × global` in fp32. The first product is exact, so there is one rounding,
    then a cast to bf16.

## Booting with it

[`../swap-to-ds41-h3.sh`](../swap-to-ds41-h3.sh) does this. By hand:

```bash
DOCKER_MOUNTS="$(engram-nvfp4/lane-mounts.sh)" ./swap-to-m3v2.sh
```

- [`lane-mounts.sh`](lane-mounts.sh) prints the mounts.
  - It mounts the two overlay files and the merged view's `config.json` and index. It also mounts the two NVFP4
    shards (47 and 48) over the same names in the launcher's `/model`.
  - It also includes the vllm#58132 mounts, because a non-empty `DOCKER_MOUNTS` makes the swap script skip its own.
  - `merged` mode is for a launcher whose model path is the merged view itself.
- [`make_merged_view.py`](make_merged_view.py) builds and checks the merged checkpoint view.
  - Shards 1–46 and the tokenizer are relative symlinks to the original checkpoint; shards 47–48 point to the NVFP4
    repo.
  - The config gains `engram_quant*`, and the index keeps the original `weight_map` with `total_size` recomputed.
  - Original shards 47 and 48 each hold the six tensors of one Engram layer. Only `embed.weight` and `embed.scale`
    changed format; the other four are byte-identical, which was checked.
- **Rollback:** boot without these mounts. A partial mount (an NVFP4 config with FP8 shards, or the reverse)
  fails at load with a clear error rather than serving garbage.

## Tests

The tests ran on the RTX PRO 6000 (SM120) through CDI, in the `af7f9488` image with a 40 GiB container cap. They
are in [`tests/`](tests/); results and logs are in [`tests/results/`](tests/results/).

| test | what | result |
|---|---|---|
| `test_dequant.py` | 65,536 random rows, edge rows and a 4,096-row run per table, read from the shards and loaded through the overlay loader into registered memory. Gathered over UVA (both grids), plus an HBM-resident table, against a CPU reference | bit-identical (69,625 + 69,631 rows); invalid ids give zero rows; rel RMS vs FP8 8.987% / 9.064% |
| `test_lookup_perf.py` | 16.8M real layer-1 rows, fresh random ids, L2 flushed, CUDA-event kernel time | table below |
| `test_fp8_regress.py`, `compare_fp8_regress.py` | FP8 rows on pinned / THP / HBM storage × T 1–8192 × both grids, stock image vs overlay | 15/15 output hashes identical |
| `test_load_path.py` | vLLM config parse and file discovery on the merged view; a full-size dry run of both real tables (host blocks on the meta device); 32M real rows loaded and looked up through `Engram.prepare_embeddings` | all pass; registered = exact size, `pin_memory` = rounded |
| `check_mounts.sh` | a container mounted exactly like the launcher | `/model` parses as NVFP4 with both global scales; 48 shards, 96,085 tensors; both overlays import together |

Lookup kernel time on the RTX PRO 6000 (µs, median; T tokens × 24 heads; fg = full grid):

| T | FP8 (stock, pinned) | NVFP4 (registered) | speedup |
|--:|--:|--:|--:|
| 1 | 7.2 | 6.1 | 1.17× |
| 16 | 13.0 | 10.0 | 1.31× |
| 64 | 23.3 | 14.3 | 1.62× |
| 512 | 117.4 | 79.6 | 1.48× |
| 8192 | 1735.7 | 1140.5 | 1.52× |

Every shape, background-grid and THP / `pin_memory` storage variants are in
[`tests/results/lookup_perf.log`](tests/results/lookup_perf.log).

## Caveats

- The overlay targets `af7f9488` exactly. It replaces two whole files and imports the private `_e2m1_inline`, so
  re-diff it against any new image.
- `cudaHostRegister` of 2 × 51.5 GiB needs `--ulimit memlock=-1`, which the launcher sets. If registration
  fails, the code falls back to rounded `pin_memory` (144 GiB) and logs a warning.
- `dp_shared_memory` is not implemented for NVFP4. That doesn't matter on this TP1 / DP1 lane.
- The scripts hard-code station paths (`/home/jasonc/research/megamoe/engram-nvfp4` is this directory) and
  checkpoint locations under `/data/checkpoints`.
