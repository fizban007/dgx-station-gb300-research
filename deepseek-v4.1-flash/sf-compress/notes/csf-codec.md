# MXFP4-CSF (`row-base-offset1-u24-exceptions/1`): codec spec and how runtimes decode it

> Research notes kept as written. Paths such as `csf-port/...`, `m3/hook/...` and `~/...` refer to the
> original workspace (`csf-port/` held the reference copies listed below); they are not part of this directory.

Research date: 2026-10-07. Read-only. Goal: let DeepGEMM MegaMoE (SM100/GB300) read
DeepSeek-V4.1-Flash routed-expert UE8M0 block scales straight from LIL's lossless CSF
checkpoint, so the scales stay compressed in HBM.

## Sources (pinned)

| Source | Ref | Local copy |
|---|---|---|
| HF checkpoint `local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF` | repo sha `9c60bcbf1670e06f154b5d61080e50c9b3b7b4a3` | `csf-port/ext/hf/` (README, manifest.json, build-contract.json, verification.json, LICENSE, NOTICE, all 48 `receipts/*.json`) |
| LIL vLLM fork, branch `integration/karmic-kraken-beta` | `89f1ceecd713a820def8df032df8c109dc02b9f4` | `csf-port/ext/vllm-lil/` (sparse checkout of the CSF files) |
| vLLM PR refs | #956 `a5430f5f`, #964 `b8e6bbeb`, #971 `2338b9cb`, #973 `33617d4d`, #982 `3b488b14` | local branches `pr956` ... `pr982` in `ext/vllm-lil` |
| b12x master (now archived, moved to FlashInfer `flashinfer/experimental/b12x`, FlashInfer #5767 commit `fc8fdc17`) | `2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88` | `~/src/b12x` |
| b12x PR #481 (squash-merged as `b64cb9c5566718fe1b4ae3ccf75f51871af06016`, 2026-10-06), twin of #479 | PR head `0bc6bca1` | local branches `pr481`, `pr479` in `~/src/b12x` |
| trellis-quant (`trellis_quant.lossless_scale_checkpoint`, commit `60feca330087`) | **not public**: github.com/local-inference-lab/trellis-quant gives 404, and web search finds nothing | n/a |

Key files:
- vLLM: `vllm/model_executor/model_loader/mxfp4_csf_loader.py` (the reference CPU slicer/validator for the codec),
  `vllm/model_executor/model_loader/csf_utils.py`, `vllm/models/deepseek_v4_1/mxfp4_csf.py`,
  `vllm/models/deepseek_v4/mxfp4_csf.py`, `vllm/model_executor/layers/quantization/mxfp4_csf.py`,
  `docs/features/quantization/fp4_csf.md`, `tests/quantization/test_mxfp4_csf.py`.
  (There is **no** `vllm/models/deepseek_v4/mxfp4_csf.py` codec logic. That file only holds the DS-V4-Flash
  config and the EP rejection. The codec is in the loader and in b12x.)
- b12x: `b12x/_lib/quant/x4t_scales.py` (canonical format, its runtime batch, and the two-kernel decoder),
  `b12x/_lib/quant/mxfp4_csf.py` (repack into native W4A8 order, plus the fused single-launch decoder into scratch),
  `b12x/_lib/quant/mxfp4_csf_inline.py` (#481: re-encoded tile format and the in-register decode),
  `b12x/moe/_shared/kernels/w4a8_phase1.py` / `w4a8_phase2.py` / `w4a8_compact_*` (kernel call sites),
  `tests/quantization/test_mxfp4_csf.py` (a synthetic encoder fixture that documents the layout).

"X4T" is the predecessor name of the same format. The docs say "Predecessor X4T/LSC names, schemas and
suffixes are not accepted. ... Migration preserves compressed payload values". b12x still uses the
X4T class names for the canonical CSF planes.

---

## 1. Codec: exact byte layout

### 1.1 Container level

- Safetensors shards under `tensors/`, using the source shard names. The shard header `__metadata__` is
  `{"format":"pt","schema":"lil-mxfp4-csf-checkpoint/1","codec":"row-base-offset1-u24-exceptions/1"}`.
- Each routed-expert scale tensor `<name>` (for example `layers.L.ffn.experts.E.w{1,2,3}.scale`, source dtype
  `F8_E8M0`) is replaced by two tensors:
  - `<name>.mxfp4_csf_fixed`: **U8, 1-D**, `[ (R/16) * 16 * (1 + ceil(C/8)) ]` bytes
  - `<name>.mxfp4_csf_exceptions`: **U32, 1-D**, `[n_exceptions]` (variable length per matrix)
- The logical shape `(R, C)` is **not** stored in the tensor header. It comes from the model geometry,
  or from `receipts/<shard>.json`, which lists `shape`, `fixed_bytes`, `exception_bytes`, `source_bytes`,
  and `source_sha256` for each scale. For DS-V4.1-Flash:
  - w1, w3 (gate, up): `R=2304` (moe intermediate), `C=5120/32=160` → fixed = 144 slabs × 336 B = **48,384 B** (source 368,640 B)
  - w2 (down): `R=5120` (hidden), `C=2304/32=72` → fixed = 320 slabs × 160 B = **51,200 B** (source 368,640 B)
- There is no other header and no per-matrix metadata. All exceptions of a matrix sit in one sorted list
  with **no index**.
- Example (verified by an HTTP range read of `tensors/model-00003-of-00048.safetensors`):
  `layers.0.ffn.experts.0.w2.scale.mxfp4_csf_fixed` U8 [51200], `...mxfp4_csf_exceptions` U32 [202];
  `w1...fixed` U8 [48384], `w1...exceptions` U32 [389].

### 1.2 Fixed stream (`.mxfp4_csf_fixed`)

The fixed stream is a sequence of `R/16` independent **16-row slabs**. Each slab is
`16 * (1 + SB)` bytes, with `SB = ceil(C/8)` selector bytes per row:

```
slab s (rows 16s .. 16s+15), byte offset s * 16*(1+SB):
  [0 .. 15]                  base[16s + i]           one uint8 per row ("palette base")
  [16 + i*SB .. 16+(i+1)*SB) selector bits of row 16s+i, LSB-first:
                             bit for column c is (byte[c>>3] >> (c&7)) & 1
                             bits for c >= C (padding) MUST be 0
```

- "row-base": each row has one base byte. The encoder picks the base `b` that maximizes
  `count(b) + count(b+1)` over the row (the same rule appears in b12x's re-encoder:
  `base = (histogram[..., :255] + histogram[..., 1:]).argmax(-1)`).
- "offset1": each scale is `base + bit`, with bit in {0,1}. So the fixed stream uses
  **1 bit per scale + 8 bits per row** (8/C bits per scale of amortized overhead): 1.05 bit/scale for C=160,
  1.11 bit/scale for C=72.
- Constraint checked by the loader: `base <= 254` ("palette bases must be in 0..254"), so `base+1` never overflows.
- C=160 gives SB=20 (no padding bits). C=72 gives SB=9 (no padding bits). The TP slices made by the vLLM
  slicer can have padding.

### 1.3 Exception stream (`.mxfp4_csf_exceptions`)

Each entry is one uint32 word:

```
word = (value << 24) | position          # "u24": position is a 24-bit field
position = row * C + col                 # row-major flat index into the logical (R, C) plane
value    = the original UE8M0 byte       # any 0..255 that is neither base nor base+1 for that row
```

- The list is **strictly increasing in position**: sorted, unique, `< R*C`. Both the vLLM slicer and b12x's
  `make_x4t_scale_batch` reject anything else. Because positions are row-major, all exceptions of a row,
  and of a 16-row slab, are contiguous.
- The selector bit under an exception position is whatever the encoder wrote. It is ignored, because the
  exception overrides it. Real data shows exception values are never `base` or `base+1`.
- `R*C < 2^24` is required. A DS4.1 matrix has 368,640 entries.
- There is no alignment or padding beyond safetensors' own tensor packing (the data offsets are not
  4-byte-aligned in general, e.g. `8063960`).

### 1.4 Decode algorithm (reference)

```python
def csf_decode(fixed: bytes, exc: list[int], R: int, C: int) -> bytearray:   # returns R*C UE8M0 bytes
    SB = (C + 7) // 8
    slab = 16 * (1 + SB)
    out = bytearray(R * C)
    for r in range(R):
        s, i = divmod(r, 16)
        base = fixed[s * slab + i]
        sel  = s * slab + 16 + i * SB
        for c in range(C):
            out[r * C + c] = base + ((fixed[sel + (c >> 3)] >> (c & 7)) & 1)
    for w in exc:                       # sorted ascending
        out[w & 0xFFFFFF] = w >> 24
    return out
```

Point query for `(r, c)`:
```
base  = fixed[(r>>4)*slab + (r&15)]
bit   = (fixed[(r>>4)*slab + 16 + (r&15)*SB + (c>>3)] >> (c&7)) & 1
v     = base + bit
p     = r*C + c
if p is in exc (binary search over the sorted positions, or a slab/row-offset index): v = exc_word >> 24
```

The vLLM reference slicer (`mxfp4_csf_loader.py::_slice_scale_plane`, kkb `89f1ceec`) shows the exact
layout and validation. TP slicing uses whole 16-row slabs and arbitrary column ranges:

```python
selectors = (columns + 7) // 8
stream = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
bases = stream[:, :16]
if (bases > 254).any():
    raise ValueError("MXFP4-CSF palette bases must be in 0..254")
bits = np.unpackbits(
    stream[:, 16:].reshape(rows, selectors), axis=1, bitorder="little"
)
if bits[:, columns:].any():
    raise ValueError("MXFP4-CSF unused selector bits must be zero")
selected = np.packbits(bits[r0:r1, c0:c1], axis=1, bitorder="little")
result = np.concatenate(
    (bases[r0 // 16 : r1 // 16], selected.reshape((r1 - r0) // 16, -1)), 1
)
words = exceptions.numpy().reshape(-1)
positions = words & POSITION_MASK          # (1 << 24) - 1
...
rr, cc = positions // columns, positions % columns
keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
positions = (rr[keep] - r0) * (c1 - c0) + cc[keep] - c0
words = (words[keep] & np.uint32(0xFF000000)) | positions
```

The b12x synthetic encoder (`tests/quantization/test_mxfp4_csf.py::fixture`) builds the same layout:
```python
selectors = np.packbits(bits, axis=1, bitorder="little")
fixed = np.concatenate((bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1)
exceptions = positions | (values << 24)
```

Encoder (trellis-quant, not public), as inferred from b12x/vLLM and the data:
1. per row, `base = argmax_b(count(b)+count(b+1))` with `b <= 254`;
2. `bit = (v == base+1)`, otherwise 0;
3. emit an exception for every `v ∉ {base, base+1}`, sorted by position.

On real data (layer 0, expert 0), every row base is 120 or 121, and exception deltas from base are
only −1 and +2.

---

## 2. Random access

### 2.1 What the on-disk format gives you

- **Fixed part: O(1) random access.** For any tile, the base is one byte per row, and the selector bits of
  row `r`, columns `[c0, c0+k)` are a contiguous bit range inside row `r`'s SB bytes. For a
  128-row × 4-col tile (one K128 block of 4 K32 groups × 128 N rows), each row needs 1 base byte and
  4 bits. Those bits lie in 1 selector byte if `c0 % 8 ∈ {0,4}`, which holds for K-blocks aligned to 4.
  Rows are **not** contiguous: the stride between rows is SB bytes, and each slab of 16 rows is 16 + 16·SB
  bytes. So a 128×4 tile touches 8 slabs: 8×16 base bytes plus 128 scattered selector bytes. That is
  uncoalesced, though it is cached in L2/L1 if the K sweep walks columns of the same rows.
- **Exception part: no random access as stored.** It is a single sorted per-matrix list. To find the
  exceptions of a tile you need either:
  - binary search on positions: lower_bound(r0*C + c0) per row, `O(log n)` with n ≈ 200–400 typical
    and up to 100,793 worst case; or
  - **a load-time index** (what every runtime builds): per-slab or per-row offsets (prefix sums of the
    exception count by `row // 16`). Positions are row-major, so the exceptions of slab s are the range
    `[off[s], off[s+1])`. For a tile covering 128 rows × 4 cols you scan the 8 slab ranges and keep
    `col ∈ [c0, c0+4)`. Typical count is 0–3 per slab, and the per-slab mean is 0.6–2.7 on layer 0.
    In the worst matrix it is 315 per slab, mean, so scanning the whole slab for each K block is costly
    there.
  The format needs **no scan of the fixed stream**. Only the exception list needs an index.
- Per-scale cost in the common path: about 1 byte load (selector, shared by 8 scales) + 1/C base load
  + shift/and/add. With a word-level SWAR trick (from b12x), 4 scales of a row come out of one nibble with
  ~4 integer ops:
  `word = base*0x01010101 + ((nibble*0x00204081) & 0x01010101)`
  This gives the 4 UE8M0 bytes packed little-endian, the same packing DeepGEMM uses for UE8M0 sf
  (4 K32 groups per int32).

### 2.2 How the runtimes actually do it

**Neither b12x nor vLLM decodes the on-disk layout inside a GEMM kernel.** All paths repack at load time.

(a) **b12x `X4TScaleBatch` / `make_x4t_scale_batch`** (`x4t_scales.py`) uploads the canonical fixed
streams (`[E, R/16, 16*(1+SB)]`) and the concatenated exceptions, with `exception_offsets[E+1]`. It can
optionally regroup exceptions per *task* of `exception_task_rows` rows, using
`task_exception_offsets[E, tasks+1]`. Decoding is two kernels into a scratch plane:
(1) a fixed-stride base/selector decode, then (2) a sparse exception scatter. From the docstring: "No
allocation, host readback, prefix scan, or variable-length tile parsing occurs during graph replay."

(b) **b12x `repack_mxfp4_csf_batch` + `Mxfp4CsfDecoder`** (`mxfp4_csf.py`) is the per-call "expansion"
path. At load it permutes the selector bits into the *native W4A8 scale order* (for compact N64:
128-row × 4-col chunks, row-major inside a chunk; for N256 padded: `[R/256, C/4, 256, 4]`). It keeps
one base byte per row, and clamps base and values to **247** to match native W4A8 E8M0 prep.
Exceptions are re-keyed to native positions (`loc | value<<24`), sorted, and partitioned into 8 KiB
tasks via `searchsorted`, which gives `task_offsets[E, tasks+1]`. At run time, one CTA of 256 threads
owns one contiguous 8 KiB output region: it writes 8 bytes per thread-iteration with SWAR, runs
`sync_threads`, then applies that region's exceptions. Only routed/active experts are decoded (route ids,
an active-expert mask when ≥64 ids, or all experts when ids ≥ 16·E). Output goes into a
**caller-owned scratch pair** (`w13_scale_scratch`, `w2_scale_scratch`) that every layer shares.

Key decode loop (b12x `mxfp4_csf.py`, `_Plane.decode`):
```python
index = Int64(expert) * Int64(self.size // 8) + Int64(local // Int32(8))
spread = bits[index].to(Uint64)                       # 8 selector bits -> 8 bytes
spread = (spread | (spread << Uint64(28))) & Uint64(0x0000000F0000000F)
spread = (spread | (spread << Uint64(14))) & Uint64(0x0003000300030003)
spread = (spread | (spread << Uint64(7))) & Uint64(0x0101010101010101)
out64[index] = packed_base + spread
...
cute.arch.sync_threads()
part = Int64(expert) * Int64(self.tasks + 1) + Int64(task)
entry = offsets[part] + Int64(tid)
while entry < offsets[part + Int64(1)]:
    word = exceptions[entry]
    destination = Int64(expert) * Int64(self.size) + (word & Uint32(0xFFFFFF)).to(Int64)
    out[destination] = (word >> Uint32(24)).to(cutlass.Uint8)
    entry += Int64(256)
```

(c) **b12x #481 inline path** (`mxfp4_csf_inline.py`, commit `b64cb9c5`). The kernels read a
**different, re-encoded format** built at load time from the expanded native compact plane, not the
checkpoint layout. From the docstring:

> Storage of one projection, in bytes, for E experts, B row blocks (the 128-row tiles of every row group in
> native order) and K K128 tiles per row block (T = B K):
> - E x T tile blocks of TILE_BYTES(=96), expert, row block and K tile major: 64 bytes of selector nibbles
>   in slot order (slot s in byte s // 2, low nibble first; bit c selects base + 1 over base in column c),
>   then a header word and RECORDS(=7) exception records. The header holds the number of exceptions
>   (0 to RECORDS), or HEAVY(0x80000000) plus the index of the tile's raw copy. A record holds the slot
>   (bits 0-6), the column (bits 8-9) and the native byte (bits 16-23).
> - E x B base blocks of 128 bytes: the base of each slot's row; padding slots hold zero.
> - raw tiles of 512 bytes, one per tile with more than RECORDS exceptions: the native word of every slot.
> ... Bases are chosen per row from the native bytes, independently of the checkpoint's own (unsliced) row bases.

Slot order follows the MMA consumer: `slot s = 32w + 4q + nt` ↔ `row = 32*(s>>5) + 8*(s&3) + ((s>>2)&7)`.
Each thread (warp w, quad q) owns 4 consecutive slots, so it reads **one u32 of 4 bases** from global
memory (once per task, since bases are constant over K) and **one u32 of selector nibbles** from shared
memory per stage.

Per pipeline stage, the kernel cp.asyncs the 96-byte tile block (6 × 16 B) into shared memory. Each
thread then rebuilds its 4 scale words (one per `nt`, each = 4 K32 UE8M0 bytes) **in registers**:

```python
# b12x/_lib/quant/mxfp4_csf_inline.py @ b64cb9c5
@cute.jit
def inline_scale_words(tile: Int32, bases: Uint32, slot0: Int32, raw: Int64):
    shift = Uint32(slot0 & Int32(4)) << Uint32(2)
    selectors = ld_shared_u32(tile + ((slot0 >> Int32(3)) << Int32(2))) >> shift
    spread = Uint32(0x00204081)
    ones = Uint32(0x01010101)
    w0 = (bases & Uint32(0xFF)) * ones + ((selectors & Uint32(0xF)) * spread & ones)
    w1 = ((bases >> Uint32(8)) & Uint32(0xFF)) * ones + (
        ((selectors >> Uint32(4)) & Uint32(0xF)) * spread & ones
    )
    w2 = ((bases >> Uint32(16)) & Uint32(0xFF)) * ones + (
        ((selectors >> Uint32(8)) & Uint32(0xF)) * spread & ones
    )
    w3 = (bases >> Uint32(24)) * ones + (
        ((selectors >> Uint32(12)) & Uint32(0xF)) * spread & ones
    )
    header = ld_shared_u32(tile + Int32(SELECTOR_BYTES))
    if header != Uint32(0):
        if (header & Uint32(HEAVY)) != Uint32(0):
            w0, w1, w2, w3 = ld_global_nc_v4_u32(
                raw
                + Int64(header & Uint32(HEAVY - 1)) * Int64(RAW_TILE_BYTES)
                + Int64(slot0 * Int32(4))
            )
        else:
            for index in cutlass.range_constexpr(RECORDS):
                if Uint32(index) < header:
                    record = ld_shared_u32(tile + Int32(SELECTOR_BYTES + 4 + 4 * index))
                    slot = Int32(record & Uint32(127)) - slot0
                    byte_shift = ((record >> Uint32(8)) & Uint32(3)) << Uint32(3)
                    value = ((record >> Uint32(16)) & Uint32(0xFF)) << byte_shift
                    keep = (Uint32(0xFF) << byte_shift) ^ Uint32(0xFFFFFFFF)
                    w0 = Uint32(cutlass.select_(slot == Int32(0), (w0 & keep) | value, w0))
                    w1 = Uint32(cutlass.select_(slot == Int32(1), (w1 & keep) | value, w1))
                    w2 = Uint32(cutlass.select_(slot == Int32(2), (w2 & keep) | value, w2))
                    w3 = Uint32(cutlass.select_(slot == Int32(3), (w3 & keep) | value, w3))
    return w0, w1, w2, w3
```

Kernel call site (`w4a8_phase1.py` diff in `b64cb9c5`):
```python
if cutlass.const_expr(self.csf_inline):
    csf_storage = get_ptr_as_int64(w13_sfb_rp, Int32(0))
    csf_block = Int64(expert_idx) * Int64(intermediate_tiles * Int32(2)) + Int64(output_tile)
    stage_inline_tile(csf_storage,
        (csf_block + Int64(intermediate_tiles)) * Int64(input_k128_tiles) + Int64(k128_slice),
        gate_sfb_base, tid, 0)
    stage_inline_tile(csf_storage,
        csf_block * Int64(input_k128_tiles) + Int64(k128_slice), up_sfb_base, tid, 32)
...
# once per task: row bases are constant over the K sweep
csf_slot = warp_idx * Int32(32) + q * Int32(4)
up_bases = inline_row_bases(csf_storage, csf_tiles_bytes, csf_block, csf_slot)
...
# per stage, after the stage barrier
gate_scales = inline_scale_words(gate_sfb_base, gate_bases, csf_slot, csf_raw)
up_scales   = inline_scale_words(up_sfb_base,  up_bases,  csf_slot, csf_raw)
...
gate_sfb = gate_scales[nt] >> scale_shift        # replaces ld_shared_u32 of native SFB
```

Exceptions are located with **no search at all**. Every 128×4 tile carries ≤7 inline records, and
heavier tiles fall back to a 512 B raw copy, found through `header & 0x7FFFFFFF`. Cost per thread per
K128 stage: 1 shared u32 load (selectors) + 1 shared u32 load (header) + 4×(mul, and, add) + extras.
Cost per CTA: 96 B of cp.async per 128×4 tile, versus 512 B for native scales (**5.33× less scale
traffic**), plus 128 B of bases per row block per task.

Size: 96 B per 128×4 tile = 1.5 bits/scale, plus 1 B/row, plus 512 B per heavy tile. In the PR's synthetic
benchmark: "Scale storage is 23.4 MiB versus 101.2 MiB native". In DS4.1 TP4 serving, "inline storage
0.33 GiB per GPU".

Gating (b12x master): compact W4A8 only (intermediate size ≡ 64 mod 128 per rank, i.e. N64 tails, e.g.
DS4.1 TP4 I=576). `B12X_W4A8_CSF_INLINE_MAX_TOKENS` defaults to 1536. Above that, a Triton kernel
(`_expand_tiles`) expands the inline storage into the native scratch, and the native kernels run.
`B12X_W4A8_CSF_INLINE=0` disables the inline path. Other geometries (e.g. TP2 I=1152, N256) use path (b),
the per-call expansion into scratch.

Measured results (#479/#481): DS4.1 TP4 + DSpark, 4× RTX PRO 6000 Max-Q, decode tok/s, per-call
expansion vs inline: C8 787→816, C16 1,069→1,108, C32 1,581→1,645. The uncompressed checkpoint gets
807 / 1,105 / 1,612. Layer microbench: 8/64/128 tokens 136.6/765.8/1061.6 µs → 132.0/730.6/992.2 µs.

### 2.3 Recommendation for a DeepGEMM MegaMoE port (derived)

DeepGEMM SM100 UE8M0 SF packs 4 K32 groups per int32 per row, which is exactly b12x's "native word".
The b12x inline scheme ports directly:
- Re-encode at load time per (expert, 128-row N block, K128 block). Choose the tile shape to match
  MegaMoE's BLOCK_N and its SF TMA box. Store the selector nibbles (64 B), a header, k inline records,
  and a raw fallback. Store per-row bases (128 B per N block). Pick a slot order that matches the thread
  → row mapping of the epilogue/UMMA SF consumer. On SM100 the SF goes to TMEM via `tcgen05.cp` from smem,
  so decode into the smem SF tile **after** the TMA/cp.async and before `tcgen05.cp`, and keep the
  canonical smem SF layout. The register-only trick is specific to SM120's mma.sync operand path.
- Alternatively, read the checkpoint layout directly. Fixed part: gather 1 base + 4 bits per row.
  Exceptions: precompute `slab_offsets[E, R/16+1]` (or per-(slab, K128) offsets) at load time. In the
  common case this is a short scan. In heavy cases (layer 37–39 w2), 300+ per slab makes it expensive.
  A raw-fallback flag per tile is strongly advised.
- Clamping: b12x clamps E8M0 to ≤247 for its W4A8 path. DeepGEMM must keep whatever its native path does
  (probably no clamp).

---

## 3. Statistics (DeepSeek-V4.1-Flash; computed from all 48 `receipts/*.json`)

- Matrices: 46,080 = 40 layers × 384 experts × 3 projections (MTP experts are not compressed).
- Source scale bytes 16,986,931,200 → fixed 2,272,788,480 + exceptions 40,542,344 = **2,313,330,824 (13.62%)**.
  Exceptions are only **1.75%** of the compressed bytes. In total 10,135,586 exceptions over
  16.99 G scales gives a **0.060% exception rate**.
- Per projection (shape, fixed bytes; exceptions per matrix mean / min / max; exception rate mean / p50 / p99 / max):

| proj | shape | fixed B | exc mean | min | max | rate mean | p50 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| w1 | 2304×160 | 48,384 | 235.6 | 59 | 7,240 | 0.064% | 0.044% | 0.31% | 1.96% |
| w3 | 2304×160 | 48,384 | 180.6 | 59 | 5,293 | 0.049% | 0.037% | 0.28% | 1.44% |
| w2 | 5120×72 | 51,200 | 243.6 | 62 | 100,793 | 0.066% | 0.032% | 0.59% | 27.3% |

- Typical compressed size per matrix: ~49.3 KB (w1/w3), ~52.2 KB (w2), versus 368,640 B raw (7.1–7.5×).
- Worst cases are all w2 in late layers: `layers.39.experts.7.w2` has 100,793 exceptions (27.3%) and totals
  **454,372 B, larger than the raw 368,640 B**. After that: L39 e188 (66,578), L39 e87 (56,856),
  L38 e250 (53,489), L37 e123 (43,793). The exception count per layer is flat at 145k–300k until it rises
  in layers 35–39 (359k, 286k, 400k, 624k, **1.49M** in L39).
- Inspection of the worst matrix (range-read from shard 42): bases 119–122, exception deltas from base
  +2 (49,974), −1 (29,684), +3 (19,033), +4 (2,073). Per 16-row slab: mean 315, max 381. Per row: max 40.
  Per 128×4 tile: mean 140, max 262, so **720/720 tiles exceed 7 records** (all heavy under b12x's scheme).
- Typical matrix (L0 e0): w1 389 exceptions, w2 202. Per 128×4 tile: max 6 / mean 0.54 (w1),
  max 4 / mean 0.28 (w2). No tile exceeds 7. 60–74% of tiles have no exceptions. Per slab: max 11.
- Real decoded value range per row is narrow: ≤3–4 distinct values per row in L0.

---

## 4. LIL vLLM loader behaviour

- Flags: `--quantization mxfp4_csf --load-format mxfp4_csf`. The serving dir holds metadata only.
  `quantization_config = {..., "quant_method":"mxfp4_csf", "format_version":1, "checkpoint_root": <abs path>}`.
  The loader validates `manifest.json`/`build-contract.json` schema+codec+family (`deepseek_v41`:
  E=384, H=5120, I=2304, layers 0..39).
- `Mxfp4CsfModelLoader.get_all_weights` skips routed-expert tensors of target layers. They are read in
  `Mxfp4CsfMoEMethod.process_weights_after_loading` → `read_mxfp4_csf_layer`. That function slices the
  FP4 nibbles and the **compressed CPU scale planes** per TP rank (`_slice_scale_plane`, whole 16-row
  slabs; w13 = w1 rows ++ w3 rows, with w3 exception positions offset by `local*columns`; w2 sliced by
  columns), and builds `b12x.moe.fused_moe.Mxfp4CsfWeights(w13, w2, CsfScalePlanes(...), ..., w13_scale_scratch, w2_scale_scratch)`.
- **Scratch:** one model-wide pair `owner.scale_scratch = (uint8[E, H/32, 2n], uint8[E, n/32, H])`,
  shared by all layers. From the code comment: "every layer consumes these scale grids before its
  successor may overwrite them on the same stream". This requires PP1 and no ubatching. Docs: "The B12X
  backend keeps scales compressed between layers and reconstructs the needed scales into shared GPU
  scratch before expert computation" and "Separate concurrent execution streams need separate
  decoded-scale scratch."
- **Decode location:** inside b12x. vLLM never builds GPU batches ("vLLM does not construct kernel-specific
  GPU scale batches", #956 change note). b12x `prepare_weights` uploads, repacks, and either (i) decodes
  routed experts per call into the scratch (`experts.mxfp4_csf.decode(topk_ids, w1_blockscale, w2_blockscale)`
  in `_impl.py`), or (ii) for compact W4A8 with ≤1536 planned tokens, has the kernels read the inline
  storage directly (#481). The scratch is then used once at prep to build the inline storage, and is
  used again for expansion above the limit.
- **MoE backend:** b12x only (`B12xExperts` + `MoEPrepareAndFinalizeNoDPEPModular`). Activations are
  MXFP8 (W4A8) by default, or BF16 with `VLLM_B12X_MOE_FP4_FORCE_A16=1`. Accepted activations: bias-free
  BF16 SwiGLU or SiTU(4,25).
- **EP/MegaMoE rejected.** `Mxfp4CsfMoEMethod.__init__`: `raise NotImplementedError("MXFP4-CSF experts support TP without EP/DP")`
  if `use_ep / ep_size!=1 / dp_size!=1 / all2all / eplb`. DS-V4-Flash config
  (`vllm/models/deepseek_v4/mxfp4_csf.py`):
  ```python
  if current is not None and current.parallel_config.enable_expert_parallel:
      # MegaMoE experts bypass quantization methods and would never read
      # the compressed experts.
      raise NotImplementedError(
          "MXFP4-CSF DeepSeek-V4-Flash supports TP without expert parallelism")
  ```
  Release note (#971): "Expert-parallel execution, including MegaMoE, is rejected because those experts
  would bypass the compressed-scale reader." In other words, MegaMoE was rejected for **plumbing**
  reasons (the MegaMoE path loads experts outside the quant method), not for technical reasons.
- **SM100/GB300:** **none.** `get_min_capability()` returns **120** for every MXFP4-CSF config (SM120/121:
  RTX PRO 6000, GB10/DGX Spark). GB300 is SM10x, so it would be rejected. b12x targets SM120. All
  published measurements are on RTX PRO 6000 Max-Q or GB10.
- TP sizes for DS4.1: 1, 2, 3 (#982), 4, 8. DS-V4-Flash: 1, 2, 4, 8.
- Reported effect (#956, DS4.1 TP4): KV capacity 9.73M → 13.11M tokens. With #479 inline: 12.79M
  (inline storage 0.33 GiB/GPU), with decode as fast as uncompressed.

---

## 5. Licensing

| Item | License | Notes |
|---|---|---|
| LIL vLLM fork (all CSF files carry `SPDX-License-Identifier: Apache-2.0`) | **Apache-2.0** | Standard upstream vLLM LICENSE |
| b12x (`pyproject: license = "Apache-2.0"`, LICENSE = plain Apache 2.0 since commit 5f572715) | **Apache-2.0** | Repo archived 2026-10-06. Code now lives in FlashInfer (`flashinfer/experimental/b12x`, Apache-2.0) |
| trellis-quant (encoder) | **unknown / not public** | 404 on GitHub. You cannot rely on it, but you do not need it: the decoder is fully specified above and the restore path is trivial |
| Checkpoint `DeepSeek-V4.1-Flash-lossless-CSF` | **LIL License 1.0** (`LicenseRef-LIL-1.0`) = Apache-2.0 text + restrictive Part 1 | Upstream DeepSeek weights are MIT (`LICENSES/MIT.txt`). Revisions ≤ `c5c41fe3` were MIT |

LIL-1.0 points that matter for private serving (from `ext/hf/LICENSE`):
- §2.2(a)(b)(d) **explicitly allowed**: downloading, "using, running, evaluating, studying, modifying, and
  making Derivative Works"; "keeping copies on devices and in storage that only You control"; "running the
  Licensed Work or a Derivative Work to provide a service, such as an inference service, as long as the
  files ... are not made available to users of the service". **Private serving and load-time re-encoding
  of scales are fine.**
- §2.1 **no Reupload** of the work or any "Substantially Similar Copy" (renamed, re-sharded, re-packaged,
  converted, or dequantized copies). A re-encoded on-disk cache of the checkpoint must not be shared outside
  your org. §1.8: sharing with "Your own employees and contractors" is not a Reupload.
- §3.1 Attribution Notice required at the **top of any Covered Page** (README / model card / main landing
  page) of a project that "contains, is derived from, or runs" the work. If you publish a repo or README
  about the DeepGEMM port that runs this checkpoint, its first paragraph must carry the notice. The code
  itself can stay under its own license.
- §3.6 keep copyright notices, license notices, Identifying Marks ("Ні пуху, ні пера",
  LIL-CANARY-4A08-5BF0-5CDE-25E4), and embedded metadata (safetensors `__metadata__`, configs) in any
  distributed derivative.
- §5.1 breach of §2/§3 terminates immediately, with no cure period.
- The *format* (row base + 1-bit offset + sorted u24 exceptions) is a simple, documented codec, and the
  Apache-2.0 b12x/vLLM code fully specifies it. Writing a decoder is not restricted by the checkpoint
  license, which applies to the checkpoint files. This is not legal advice.
