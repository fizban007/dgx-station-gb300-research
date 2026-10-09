# Compressed MoE weight scales: details

DeepSeek-V4.1-Flash routed experts are MXFP4: FP4 weights with one UE8M0 scale byte per 32 weights. On the
M3 setup the scales of 258 hot experts × 40 layers take 10.6 GiB of GB300 HBM. Those scales are highly
redundant. Within a small block, almost every scale is one of two adjacent values, so the block compresses to
a base byte plus one bit per scale. This directory exploits that losslessly on both GPUs.

- [Phase 0: how compressible are they?](#phase-0-compressibility)
- [Phase 1: GB300, compressed in HBM with a just-in-time decode pass](#phase-1-gb300-compressed-scales-with-just-in-time-decode) (deployed)
- [Phase 2: GB300, decode inside the MegaMoE kernel](#phase-2-decode-inside-the-megamoe-kernel-paused) (paused)
- [Phase 3: RTX PRO 6000 sidecar, b12x MXFP4-CSF](#phase-3-sidecar-scales-through-b12x-mxfp4-csf) (deployed)

## Where the scales live in MegaMoE

Full analysis: [notes/megamoe-sf-path.md](notes/megamoe-sf-path.md).

The loader parameters are `uint8 [E, N, K/32]`. `finalize_weights` converts them in three steps:

1. To FP32 powers of two.
2. `deep_gemm.transform_sf_into_required_layout` packs 4 consecutive K-groups into one int32 (byte j = K-group
   4·kb + j), giving `int32 [E, N, Kp]` with **MN-major** strides `(N·Kp, 1, N)`.
3. `transform_weights_for_mega_moe` interleaves the L1 gate/up rows in groups of 8 and transposes every 128 rows
   4×32 for UTCCP.

In memory that is `[E, Kp, N]`. The kernel's SFB producer warp TMA-loads one contiguous 512 B chunk per
(expert, K-block, 128-row N-block): 128 int32 words, word r = row r's 4 K-group scales. Warp 6 then moves each
chunk from SMEM into TMEM with `tcgen05.cp` (UTCCP) for the MMA.

| | GEMM1 (w13, L1) | GEMM2 (w2, L2) |
|---|---|---|
| SF tensor | `int32 [E, 4608, 40]` | `int32 [E, 5120, 18]` |
| chunks per expert | 40 × 36 = 1440 | 18 × 40 = 720 |

## Phase 0: compressibility

Script: `phase0/sf_compress_stats.py`. Results: `phase0/phase0-h258.json`.

Input: the official checkpoint, the 258 hot experts per layer of `rowmap-mix-h258.json`, all 40 layers. The
chunks are exactly the ones MegaMoE loads, rebuilt after the interleave and the UTCCP transpose.

| | raw | b1 (chunk base + 1 bit + 4 B exceptions) | ratio |
|---|--:|--:|--:|
| L1 (w13) | 7.09 GiB | 0.92 GiB | 13.0% |
| L2 (w2) | 3.54 GiB | 0.46 GiB | 13.0% |
| total | 10.63 GiB | 1.38 GiB | 13.0% |

- **A per-chunk base works as well as a per-row base.** It gives 13.0% vs 13.3/14.2% for LIL CSF's per-row base.
  So every chunk can be self-contained and decode independently, with no row-base table.
- **Exceptions are rare.** These are values outside {base, base+1}:
  - Mean per 512-scale chunk: 0.38 (L1) and 0.42 (L2).
  - Chunks with none: 77% (L1) and 82% (L2). p99 is 3-4.
  - The late layers are worst: layer 39 L2 averages 5.9 exceptions per chunk.
- **2 bits per scale isn't worth it.** It almost never needs exceptions, but costs 25%.

A simple GPU decoder wants a fixed-stride slot, so the address of every chunk is O(1). The candidates were a b1
bit plane plus a 4 B header plus N inline exceptions. Chunks with more than N exceptions spill to a raw 512 B
tile:

| slot | slot bytes | spill (L1 / L2) | total | ratio |
|---|--:|--:|--:|--:|
| **slot3** (chosen) | 80 | 0.76% / 1.06% | 1.75 GiB | 16.5% |
| slot7 | 96 | 0.29% / 0.56% | 2.03 GiB | 19.1% |
| slot15 | 128 | 0.14% / 0.37% | 2.68 GiB | 25.2% |

A variable-size b1 would save ~0.4 GiB more, but it needs an offset table per chunk.

## Phase 1: GB300, compressed scales with just-in-time decode

Code: `hook/sf_compress.py` and the `MEGA_SF_COMPRESS` part of the hook (`patches/mega_peer_hook.py.diff`).

### Slot format (one per 512 B chunk, 20 int32 words)

```
word 0      base (bits 0-7) | n_exc (bits 8-9, 0..3) | spill (bit 10) | spill index (bits 11-31)
words 1-16  bit plane: scale p (= 4*row + kgroup, the chunk's byte order) is bit p%32 of word 1 + p//32;
            its value is base + bit
words 17-19 up to 3 exceptions: position p (bits 0-8) | raw byte (bits 16-23), overriding base + bit
```

- **Spills.** A chunk with more than 3 exceptions sets `spill` and stores its raw 512 B in `overflow [n_spill, 128]`.
- **Base.** The encoder picks the base that maximizes the number of scales in {base, base+1}. It is at most 254.
- **Layout.** Slots are laid out `[E, Kp, NB, 20]`, so chunk g = (e·Kp + kb)·NB + nb is linear in the same order as
  the original SF memory.

### Runtime

- **At load** (`finalize_weights`, after DeepGEMM's transforms): each layer's L1/L2 SF is encoded on the GPU. The
  raw tensor is then replaced by a scratch tensor of the same shape and strides, one per shape, shared by all 40
  layers. The kernel never notices: it still reads a normal SF tensor.
- **Before each MegaMoE call:**
  - `_list_kernel` (one CTA) builds the unique routed hot experts from `hot_ids` on the GPU.
  - `_decode_kernel` writes only those experts' chunks into the scratch. Each program is a `[32 chunks × 128 rows]`
    tile with 4 warps.

  Row r's 4 scales are a nibble of the plane, expanded with `base·0x01010101 + ((nib·0x00204081) & 0x01010101)`.
  Spilled chunks copy the raw tile. Exceptions are applied after the tile store.
- **CUDA graphs:** there is no host sync, so the whole thing is captured into vLLM's CUDA graphs. The grid is
  sized by `min(E, number of ids)`, which the host knows.
- **The drafter** (DSpark, a separate 128-expert MegaMoE instance) is untouched.

### Correctness
- **Offline** (`phase1/test_sf_compress.py`): real checkpoint scales through DeepGEMM's own transforms on layers
  0/20/37/38/39. Decode is bit-exact for all experts and for routed subsets.
- **In server** (`MEGA_SF_CHECK=6`): keeps raw SF for layers 0 and 39, and compares the decoded routed experts on
  the first 6 eager calls. 6/6 were bit-exact.
- **GSM8K-200** (effort 50, T=0, C16): 97.0% uncompressed vs 97.5% compressed. That is noise, since outputs are
  not run-to-run deterministic on this stack even uncompressed.

### Memory (GPU_UTIL 0.9, 258 hot / 126 cold, 64 GiB native KV offload)
| | uncompressed | compressed |
|---|--:|--:|
| hot-expert scales | 10.63 GiB | 1.75 GiB + 0.27 GiB scratch |
| KV cache | 13.5 GiB, 6.99M tokens | 22.05 GiB, 11.41M tokens (+63%) |

### Speed
- **C1 decode** (1024 forced tokens, incl. prefill): 212 -> 198 tok/s (-6.7%).
- **GSM8K C16 aggregate:** 1051 -> 1020 tok/s (-3%).
- **Decode kernel** (GB300, graph replay):
  - All 258 experts: ~50 us (L1) + ~29 us (L2) per layer.
  - A C1 step's 36 routed ids: ~10.5 + 8.4 us per layer, including the list kernel.
  - The kernel reaches ~3.5-3.8 TB/s. Plain writes reach 6.5 TB/s.

The decode pass is what phase 2 tried to remove.

## Phase 2: decode inside the MegaMoE kernel (paused)

Code: `phase2/sm100_fp8_fp4_mega_moe.csf.diff`, against the header vendored in vLLM nightly af7f9488,
`vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`, md5 `dec0e85d...`. Ablations
are in `phase2/ablations/`, each against the main patch. Harness: `phase2/run_test.sh` -> `test_megamoe_csf.py`.
It builds one real layer in a temporary container, runs raw vs CSF with the raw SF poisoned in the CSF runs,
and requires bit-identical outputs.

### Mechanism

- **Table.** A non-null `cumulative_local_expert_recv_stats` carries a 4-pointer table: {L1 slots, L1 overflow, L2
  slots, L2 overflow}. That argument is otherwise only used by a `red_add`, and vLLM passes None. A null table
  keeps the stock path, which the drafter uses.
- **Decode.** The B producer warp decodes each 80 B slot into `smem_sfb[stage]` instead of issuing the SFB TMA.
- **Shared expert.** It keeps the TMA.

### Result
- **Correct:** bit-identical to the stock kernel at T = 1..2048 (254 experts, layer 0).
- **Slow** (254 experts):

  | T | stock | phase 2 |
  |--:|--:|--:|
  | 1 | 92 us | 136-143 us |
  | 8 | 158 us | 770-800 us |
  | 128 | 725 us | 4000-4170 us |

### Ablations (96-expert subset, T=8 / T=128, stock 137 / 312 us)
| variant | T=8 | T=128 |
|---|--:|--:|
| main patch: generic st.shared + fence + release.cluster arrive, slot loads 4 ahead | 661 | 1629 |
| `relaxed`: relaxed arrives (unsafe) | ~420 | ~2179 (254 exp) |
| `relaxed_nofence`: relaxed, no proxy fence | 356 | 860 |
| `noload`: constant slot instead of loads, decode math kept | 182 | 417 |
| `noload_waits`: no loads, stock waits | 182 | 416 |
| `noload_waits_nodecode`: no loads, no decode math | 130 | 296 |

What the ablations showed:
- **Cluster-scope waits and the proxy fence cost almost nothing.**
- **A release-ordered mbarrier arrive waits for all the warp's outstanding global loads,** including the prefetch
  loads for later stages. So the producer serializes on load latency.
- **The decode math alone costs ~33%,** even without any loads, because it sits on the producer's critical path.
- **`st.async` deadlocks** (`st_async_deadlock.diff`). An async store with complete_tx would avoid the release.
  But it can only signal a barrier in the destination CTA. In the 2-CTA cluster the non-leader CTA has to signal
  the leader's barrier, so the variant hangs.

### Why it is hard
- **The stock producer never waits on data:** TMA is fire-and-forget. A decoder has to have the compressed bytes
  in registers first.
- **Tasks are short** (40 / 18 K-blocks) and handed out one at a time from global atomic counters, through a
  2-entry scheduler ring. So per-task prefetch pays a full memory latency at every task start.

### How b12x avoids this on SM120
b12x decodes MXFP4-CSF inline on the RTX PRO 6000 without this problem:

- Its MMA is `mma.sync`, which takes scales from **registers**.
- The compressed tile travels through the normal async pipeline (cp.async into SMEM).
- After the stage barrier, each consumer thread rebuilds its own scale words in registers. Nobody waits on a
  global load at a barrier.

On SM100, `tcgen05.mma` reads scales from **TMEM**. The only way in is `tcgen05.cp` from SMEM, so the decoded
bytes have to be in SMEM before the MMA warp's copy.

### Options not yet tried
1. **Bulk-copy the compressed slot, decode where it lands.** `cp.async.bulk` the 80 B slot into `smem_sfb`
   (fire-and-forget, like the TMA), then expand it in place after the full barrier, before `tcgen05.cp`:
   - leader CTA: the MMA warp expands it.
   - non-leader CTA: the idle warp 6 waits on its own barrier, expands, and does a release arrive to the leader.
2. **Prefetch the next task's slots.** Peek at the scheduler's next task and prefetch its slots during the current
   task.
3. **Give up one pipeline stage** and reuse its SMEM as a TMA-filled ring of compressed slots.
4. **Decode into a small L2-resident per-SM global ring** and keep the stock SFB TMA.

The whole effort is capped by phase 1's cost: ~7% at C1 and ~3% at C16.

## Phase 3: sidecar scales through b12x MXFP4-CSF

Code: `sidecar/csf_encode.py` and the `PEER_CSF` part of `patches/peer_server2.py.diff`. Background on the
format: [notes/csf-codec.md](notes/csf-codec.md).

b12x already supports MXFP4-CSF for W4A8. Its API:
- `fm.Mxfp4CsfWeights`
- `fm.CsfScalePlanes`
- `Mxfp4CsfDecoder`

The scale planes stay compressed in VRAM. Inside `fm.run`, b12x expands the routed experts into a caller-owned
scratch pair. The expansion is graph-capturable and skips negative ids. So no kernel work was needed, only an
encoder.

`csf_encode.encode_planes` runs on the CPU, in batches of 8 experts:
- The input is the checkpoint's E8M0 scales, clamped to ≤ 247 as the raw path already does.
- Per expert, rows go in 16-row slabs. Each slab is 16 row-base bytes followed by the slab's selector bits (one
  bit per scale, little-endian per row).
- Values outside {base, base+1} go to a sorted uint32 exception stream: `pos | value << 24`.

Inline mode (decoding inside the GEMM) only applies to compact shapes (I % 128 == 64). DS-V4.1 has I = 2304, so
b12x uses its expansion path.

`peer_server2.py` makes three changes:
1. It keeps the raw scales on the host and encodes them per layer.
2. It shares one scratch pair across all layers with the same cold-expert count.
3. It wraps the weights in `fm.Mxfp4CsfWeights`. The CSF path defaults to per-expert mutable activation scales.
   The server swaps in the raw path's scalar unit scales (`a1_gscale = a2_gscale = 1`,
   `immutable_input_scales=True`), so CSF and raw run the same arithmetic.

### Correctness (`phase3/test_sidecar_csf.py`, 8 real layer-0 cold experts)
- **Scales and weights:** the decoded native scales and the prepared weights are byte-identical to the raw path.
- **Fused MoE output:** b12x isn't bit-reproducible run to run, because of atomic top-k accumulation. Its
  deterministic routing gave NaNs in this harness on both paths. So the test compares noise floors instead:
  raw-vs-CSF max/relative error equals raw-vs-raw at m = 1..256, with scales poisoned before every graph
  replay.
- **Decode cost:** +1-3 us per call. Calls take ~40-220 us.

### Memory
- **Measured, 126 cold/layer:**
  - Scales: 5.19 GiB raw -> 0.71 GiB compressed + 0.13 GiB scratch (7.3x). Exceptions are 0.05-0.26% of scales.
  - nvidia-smi on the 6000: 92,760 -> 89,350 MiB, -3.3 GiB. That is less than on paper because of allocator
    overhead.
- **Cost of a cold expert:** one more per layer costs ~0.67 GiB on the sidecar (17.7 MB of weights plus
  compressed scales, × 40 layers).

### Rebalanced: 254 hot / 130 cold (`hook/rowmap-mix-h254.json`)
- **Sidecar:** 92,082 MiB, vs 92,760 for raw scales with 126 cold. CSF: 5.36 GiB -> 0.73 + 0.13 GiB.
- **GB300 at GPU_UTIL 0.87, 64 GiB offload:**
  - KV cache: 17.35 GiB = 8.98M tokens, up from 14.56 GiB = 7.53M.
  - ~34 GiB left free.
- **GPU_UTIL 0.832** (the current default): KV cache 7.86 GiB = 4.07M tokens, ~43.7 GiB left free.
- **Quality and speed:**
  - GSM8K-200: 98.5%.
  - C16: 940 tok/s, vs 1020 before. Four more cold experts per layer go over the sidecar link.
  - C1 decode: 194-206 tok/s, 1.02x of phase 1.
