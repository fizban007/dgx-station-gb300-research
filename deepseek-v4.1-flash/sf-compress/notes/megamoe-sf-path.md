# MegaMoE (SM100 FP8 x MXFP4) weight-scale (SFB) path — read-only analysis

> Research notes kept as written. Paths such as `csf-port/...`, `m3/hook/...` and `~/...` refer to the
> original workspace (`csf-port/` held the reference copies listed below); they are not part of this directory.

Date: 2026-10-07. Nothing was modified.

Path abbreviations:
- `DG` = `~/ai/llm/ds4-station/csf-port/ref/vllm_third_party_deep_gemm` (identical to the container copy:
  the md5 of `sm100_fp8_fp4_mega_moe.cuh` matches `/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/...`)
- `K` = `DG/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
- `M` = `~/ai/llm/ds4-station/csf-port/ref/dsv4_nvidia_model.py`
- `UP` = upstream DeepGEMM `main` + DeepJIT `main` (downloaded to the scratchpad, used only for the C++ host
  code, which vLLM ships only as the compiled `_C.so`). The vendored build is a fork: it has `situ`, alpha/beta,
  a single `STORE_BLOCK_M`, and no `sm_locality_domains`. So treat UP host code as **close but not exact**. The
  stage counts it predicts match the JIT cache for 5 of the 6 configs.
- Hook = `~/ai/llm/ds4-station/m3/hook/mega_peer_hook.py` (the M3 deployment monkeypatch)

Deployment facts (from the JIT cache `m3/jit-cache/dj/cache/sm100_fp8_fp4_mega_moe.*/kernel.cu` and `/model/config.json`):
hidden=5120, moe_intermediate=2304, 40 MoE layers, 384 routed experts. The hook keeps the **258 hot experts**
(rowmap h258) in MegaMoE, so `kNumExperts = 258` and `kNumRanks = 1`. There is 1 fused shared expert, topk=6,
kNumSMs=152, weight dtype is `float_e2m1_unpacksmem_t` (FP4) and activation is swiglu. The separate instance with
128 experts and topk 3 is the DSpark drafter.

---------------------------------------------------------------------------------------------------

## 1. Data flow of the routed-expert weight scales

### 1a. Checkpoint to loader parameters (M)
- `w13_weight_scale` Parameter has shape `uint8 [E_local, 2*I, H/32]` = `[258, 4608, 160]` (M:259-268).
  `w2_weight_scale` has shape `uint8 [E_local, H, I/32]` = `[258, 5120, 72]` (M:284-293).
- `weight_loader` (M:320-362) copies the checkpoint `w1.scale` into rows `[0, I)` and `w3.scale` into rows
  `[I, 2I)` of the w13 slot (`narrow(0, shard_offset, I)`, M:340-344). `w2.scale` is copied as is. Each tensor
  is the raw E8M0 byte of shape `[N, K/32]` (row = output channel, column = 32-wide K group). The expert to local
  slot mapping is in `_map_global_expert_id` (M:310-318). The hook overrides `_weight_loader` with its own static
  rowmap (hook:153+).

### 1b. finalize_weights (M:542-583)
1. `_ue8m0_uint8_to_float` (M:364-366): `(u8.int32 << 23).view(float32)`, an exact power of two. This is lossless.
2. `deep_gemm.transform_sf_into_required_layout(sf_fp32, mn=N, k=K, (1,32), num_groups=E)` (M:549-561).
   This is in `_C`. In UP `csrc/apis/layout.hpp:46-52` the branch "(FP32, x, 32) on SM100" calls
   `get_mn_major_tma_aligned_packed_ue8m0_tensor` (UP `csrc/jit_kernels/impls/smxx_layout.hpp:133-184`). That
   JIT kernel `transpose_and_pack_fp32_into_ue8m0` (it is in the dj cache) takes the exponent byte back out and
   **packs 4 consecutive K-groups into one int32**: byte j holds K-group 4*kp+j, little-endian.
   - Output is `int32 [E, N, ceil(K/128)]` with strides `(Kp*N_al, 1, N_al)`, which is **MN-major**. Here
     `N_al = get_tma_aligned_size(N, 4)`, i.e. N rounded up to 4 elements (16 B). Since 4608 and 5120 are
     already aligned, **there is no padding**.
3. `deep_gemm.transform_weights_for_mega_moe((w13_u8, w13_sf), (w2_u8, w2_sf))` (M:563-567, DG/mega/__init__.py:154-172):
   - L1 SF: `_transpose_sf_for_utccp(_interleave_weights(sf))` (mega/__init__.py:164).
     - `_interleave_weights` (mega/__init__.py:120-134, gran=8): rows go from `[gate | up]` to
       `[gate0..7, up0..7, gate8..15, up8..15, ...]`. The weights get the same interleave.
     - `_transpose_sf_for_utccp` (mega/__init__.py:137-151): within every 128 rows,
       `reshape(-1,4,32,Kp).transpose(2,3)`. The value at old row `j*32+l` (j<4, l<32) moves to row `l*4+j`.
   - L2 SF: only `_transpose_sf_for_utccp` (mega/__init__.py:167).
   - Both use `torch.empty_like(x).copy_(...)`, which keeps the MN-major strides.
     `get_expert_weights`/EPLB (M:658-667) confirms "shape (E, M, N) with memory layout (E, N, M)".
4. The loader Parameters are dropped (M:575-578). Only `_transformed_l{1,2}_weights = (weight, sf)` remain.

### 1c. What the kernel receives (SFB)
| | GEMM1 (w13, L1) | GEMM2 (w2, L2) |
|---|---|---|
| tensor | `int32 [258, 4608, 40]` | `int32 [258, 5120, 18]` |
| strides (elements) | `(4608*40, 1, 4608)` | `(5120*18, 1, 5120)` |
| row order | gate/up interleaved by 8, then UTCCP 4x32 permutation per 128 rows | UTCCP permutation per 128 rows |
| bytes / expert | 737,280 | 368,640 |
| bytes / layer (258 experts) | 190 MB | 95 MB |

That totals **285 MB per layer and ~11.4 GB over 40 layers**, about 6.25% of the FP4 expert weight bytes.

The host-side checks are UP `csrc/apis/mega_moe.hpp:198-203`, which calls `check_sf_layout(..., tma_stride_check=true, kInt)`
(UP `csrc/utils/layout.hpp:101-140`). They require: dtype int32; `size == [E, N, K/128]`; `stride(-2)==1`;
`stride(-1) >= N_al` and `% 4 == 0`; `stride(-3) == stride(-1)*size(-1)`.

TMA descriptor (UP `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`, `make_tma_sf_desc` at UP
`runtime_utils.hpp:384-408`): it is 2D, `dims = {N_al, (K/128)*E}`, outer stride `= stride(-1)*4` bytes,
`box = {BLOCK_N=128, 1}`, swizzle none. The outer dimension folds experts and K-blocks together.

**One kernel load = one contiguous 512-byte chunk.** Chunk `(e, n_block nb, k_block kb)` starts at
`sf_base + 4*(e*N*Kp + kb*N + nb*128)`. Inside the chunk, int32 slot `p` (0..127) holds post-interleave row
`r = nb*128 + (p%4)*32 + p/4`. Byte `j` of that slot is K-group `4*kb + j`. For L1, physical row r maps back to
a checkpoint row like this: let `g=r/16` and `w=r%16`. If `w<8` it is `w1` row `g*8+w`, otherwise it is `w3`
row `g*8+w-8`.

Note: `prepare_megamoe.py` does **not** touch weight scales. It is the Triton input-staging kernel: it quantizes
activations to FP8, packs the activation UE8M0 into int32 (`x_sf`), and writes the shared-expert activation SF
with the same 4x32 UTCCP row permutation (prepare_megamoe.py:105-120). Weight prep is entirely
`finalize_weights` plus `transform_*`.

---------------------------------------------------------------------------------------------------

## 2. Inside the kernel

Template params baked in (K:21-57, from the JIT `kernel.cu`): `BLOCK_N=128`, `BLOCK_K=128`, `SF_BLOCK_N=128`,
`SF_BLOCK_M=align(BLOCK_M,128)`, 128 dispatch threads, 128 non-epilogue threads, 256 epilogue threads.
`BLOCK_M` and `kNumStages` are picked per call by a heuristic on the token count (UP heuristics `mega_moe.hpp`):

| BLOCK_M | stages (JIT cache) | per-stage SMEM (A+B+SFA+SFB) | approx. free SMEM |
|---|---|---|---|
| 16 | 11 | 1024+16384+512+512 = 18.4 KB | ~2.3 KB |
| 32 | 10 | 2048+16384+512+512 = 19.5 KB | ~6.1 KB |
| 64 | 9 | 21.5 KB | ~7.2 KB |
| 128 | 7 | 25.6 KB | ~12.8 KB |
| 192 | 6 | 30.2 KB (SFA 1 KB) | ~10.8 KB |
| 240 | 5 | 33.3 KB | ~15-20 KB |

The capacity is 232,448 B (UP `heuristics/sm100.hpp:16`). Stages are "max that fits" (UP
`get_pipeline_config_for_mega_moe`: `num_stages = (cap - fixed)/per_stage`). **Any new SMEM directly costs
pipeline stages.** The free column is my recomputation of the UP formula, so treat it as ±1 KB.

### SMEM layout (K:189-211)
`smem_sfb[kNumStages][SF_BLOCK_N * (BLOCK_K/128)]` is `uint32[stages][128]`, i.e. **512 B per stage per CTA**
(K:199). Each uint32 packs 4 UE8M0 values (the 4 K-groups of the 128-wide k-block) for one of the 128 N rows,
in UTCCP-permuted row order. The kernel runs a 2-CTA cluster, and **each CTA loads its own N-block**:
`n_block_idx = n_cluster_idx*2 + (leader?0:1)` (K:767).

### Producer: warp `kNumDispatchWarps+1` (= warp 5), "GEMM TMA load warp for weights with SF" (K:748-808)
- It picks the descriptor by phase (K:759-762): `tensor_map_l1_weights_sf` / `l2` / `shared_l1` / `shared_l2`.
- `shape_sfb_k = ceil(K/128)` (K:766). Per k-block (K:770-807):
  - wait `empty_barriers[stage]` (K:772)
  - `sfb_n_idx = n_block_idx*128`. **Expert selection is via the outer coordinate:**
    `sfb_k_idx = local_expert_idx*shape_sfb_k + k_block_idx` (K:777-778). There is no pointer arithmetic: the
    expert offset is folded into the TMA row coordinate.
  - one elected lane issues the weight TMA plus `tma::copy<BLOCK_N,1,0>(sfb_desc, full_barrier, smem_sfb[stage], sfb_n_idx, sfb_k_idx, 2)`
    (K:793-796). With `num_tma_multicast=2` this is `SM100_TMA_2SM_LOAD_2D` (DG/include/deep_gemm/common/tma_copy.cuh:44-53):
    each CTA loads its own data but **signals the leader CTA's full barrier**.
  - leader: `arrive_and_expect_tx(weight_bytes*2/2(FP4) + 512*2)` (K:797-800). Non-leader: `arrive(0u)`, a remote
    arrive on the leader's barrier (K:802).
- `full_barriers` are initialized with 4 arrivals: 2 CTAs x (A producer + B producer) (K:255).
- The A/SFA producer is warp 4 (K:682-747) and has the same structure.

### Consumer: warp 6 = MMA issuer, leader CTA only (K:809-928)
- wait `full_barriers[stage]`, then `tcgen05_after_thread_sync` (K:876-877).
- **UTCCP** (K:885-897): `SM100_UTCCP_4x32dp128bit_2cta` = `tcgen05.cp.cta_group::2.32x128b.warpx4`
  (DG/include/cute/arch/copy_sm100.hpp:501-517). Source is a 32-row x 16-byte (512 B) SMEM matrix described by
  `make_sf_desc` (no swizzle, SBO = 8x16 B; DG/include/deep_gemm/mma/sm100.cuh:43-55). It is multicast to all 4
  TMEM sub-partitions, in **both CTAs** (cta_group::2). Per k-block there is 1 copy for SFA (`SF_BLOCK_M/128` copies)
  and **1 copy for SFB** into TMEM columns `kTmemStartColOfSFB .. +4` (K:222-227). Result: TMEM lane l, column c =
  slot `4l+c`, which is N-row `32c+l`. The 4x32 transpose on the host exists to produce exactly this.
- 4 x `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale` (UMMA M=256 (2x128 weight rows, swap-AB), N=BLOCK_M,
  K=32). Each uses `sf_id = k` (0..3), which picks byte k of each 32-bit SF cell (K:899-912;
  DG/include/deep_gemm/ptx/tcgen05.cuh:67-86; sm100.cuh:144-148).
- `umma_arrive_multicast_2x1SM` on `empty_barriers[stage]` releases the stage in both CTAs (K:859-870, 919).

### Task granularity (DG/include/deep_gemm/scheduler/mega_moe.cuh:300-380)
A task is `(expert, m_block, n_cluster)`, taken dynamically from global atomic counters.
`kNumL1Clusters = 4608/256 = 18` and `kNumL2Clusters = 5120/256 = 20`. **Within a task the B producer walks
k_block 0..K/128-1 sequentially for one fixed (expert, 128-row N block).** That is 40 k-blocks (20 KB raw SF)
for L1 and 18 k-blocks (9 KB) for L2. The same (expert, n_block) SF is reloaded for every m_block of that expert.

---------------------------------------------------------------------------------------------------

## 3. Warp roles and resource headroom

There are 512 threads = 16 warps, one CTA per SM (`__launch_bounds__(kNumThreads,1)`, K:58), in 2-CTA clusters
with persistent grid = 152 SMs:

| warps | role | regs/thread (E/rank=258 > 64 means `kUseMoreEpilogueRegisters=false`, K:321-328) |
|---|---|---|
| 0-3 | dispatch (token pull over NVLink/TMA, then workspace cleanup) | 96 |
| 4 | A + SFA TMA producer (1 elected lane) | 88 |
| 5 | **B + SFB TMA producer (1 elected lane, 31 lanes idle)** | 88 |
| 6 | MMA issuer (**leader CTA only; idle on the non-leader**) | 88 |
| 7 | scheduler mainloop (**leader only; idle on the non-leader**) | 88 |
| 8-15 | epilogue (2 warpgroups: SwiGLU, FP8 quant, TMA store, combine) | 160 |

The register budget is **exactly** at the static-assert limit: 96*128 + 88*128 + 160*256 = 64,512 (K:325-328).
`setmaxnreg` works per warpgroup, so adding a warp means adding a whole warpgroup (or a 4x128 rebalancing) and
taking registers from someone. That is **not recommended**.

Practical headroom:
- The **SFB producer warp has 31 idle lanes and 88 regs**. It already loops k sequentially per (expert, n_block).
  This is the natural decoder.
- Warps 6 and 7 on the **non-leader** CTA are idle. They are usable, but would need extra cross-CTA sync and
  give no symmetric partner on the leader.
- SMEM: 2-6 KB free at decode-size BLOCK_M (16/32). Up to ~13 KB is free at BLOCK_M=128. One stage is ~18-19 KB.
  **`smem_sfb[stage]` (512 B) can itself hold a compressed chunk of up to 512 B**, so in-place staging needs no
  new SMEM.
- Bandwidth budget per k-block at decode (memory-bound): each CTA streams 8 KB of FP4 weights per k-block.
  At about 8 TB/s / 152 SMs that is ~150 ns, roughly 300 cycles. To stay hidden, a decoder must produce 512 B of SF
  per ~300 cycles per SM (16 B per lane). It must also keep global-load latency (~0.7-1 us) off the critical path
  by prefetching several k-blocks ahead. The ring only buffers `kNumStages` (5-11) k-blocks.

---------------------------------------------------------------------------------------------------

## 4. JIT

- `_C.init(dirname(deep_gemm/__init__.py))` (DG/__init__.py:111) leads to `init_jit(library_root)` (UP
  `csrc/runtime/jit.hpp:14-32`). The **include dir is hard-wired** to `<library_root>/include`, with
  include_prefixes `{"deep_gemm/"}`, env prefix `DG`, and extra signature `cutlass-<ver>` (meta.json shows
  `cutlass-421`). **There is no env var to override the include dir.**
- Kernel source = the `kernel.cu` shown in the cache. It `#include`s `deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
  and instantiates the template with all of: tokens/rank 9600, hidden, inter, experts, shared, topk, BLOCK_M/N/K,
  STORE_BLOCK_M, SF_BLOCK_M/N, ring tokens, SF ring tokens, stages, bytes/pull, thread counts, SMs, ranks,
  clamp/alpha/beta, fast_math, weight dtype.
- nvcc command (meta.json): `nvcc kernel.cu --cubin --gpu-architecture=sm_103f -O3 -std=c++20 --ptxas-options=--register-usage-level=10 ... --include-path <pkg>/include`.
  It runs under CUDA 13.0.
- **Cache key** (DeepJIT `runtime.hpp: cache_key`, `utils/parser.hpp:68-120`) = FNV1a over: compiler
  version/flags, plus the kernel source, plus **the recursive content of every `#include <deep_gemm/...>` header**.
  cute/cutlass headers are not hashed (they are covered by the `cutlass-421` signature).
  - So **editing any `deep_gemm/` header automatically changes the key**, and a fresh compile lands in a new
    `cache/sm100_fp8_fp4_mega_moe.<hash>/` directory. **No cache clearing is needed.** The header hash is memoized
    per process, so you must restart the process. Old entries are harmless.
  - Cache dir: `DG_JIT_CACHE_DIR` or `DJ_JIT_CACHE_DIR` (`PATH1:PATH2`, where the first is writable and the rest
    are read-only), else `$HOME/.dj` (DeepJIT `cache/disk.hpp:91-106`). The container mounts host
    `m3/jit-cache/dj` at `/root/.dj`. Other knobs (strings in `_C.so`): `DG_JIT_PRINT_COMPILER_COMMAND`,
    `DG_JIT_DEBUG`, `DG_JIT_DUMP_SASS/PTX`, `DG_JIT_PTXAS_VERBOSE`, `DG_JIT_NVCC_COMPILER`, `DG_PRINT_CONFIGS`.
- Ways to ship a patched header:
  1. **Bind-mount one file**:
     `-v patched.cuh:/usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:ro`.
     `launch-m3.sh:26-28` already supports this via `DOCKER_MOUNTS="host:container[:ro] ..."`.
     New helper headers can be mounted the same way into `include/deep_gemm/...`.
  2. Bind-mount a whole patched `include/deep_gemm/` directory.
  3. Call `vllm.third_party.deep_gemm._C.init('/w/dg_root')` from the hook/sitecustomize **before the first JIT
     compile**, pointing at a copied tree. That root must contain the full `include/` (cute/cutlass too). I have
     not verified that re-init is side-effect-free, so options 1 and 2 are safer.
- **Kernel-signature limit:** the launch arguments come from the compiled host code in `_C.so`
  (`y, cumulative_local_expert_recv_stats, num_tokens, SymBuffer, 18 TMA descriptors`). **A header-only patch
  cannot add kernel parameters.** Usable side channels without rebuilding `_C`:
  - `cumulative_local_expert_recv_stats` (`int*`) is passed as `None` by vLLM (M:763-783; Python default at
    mega/__init__.py:182). The kernel only uses it at K:653 (`red_add` stats). A patched kernel can reinterpret it
    as a pointer to a per-layer CSF descriptor table, passed from Python as any int32 CUDA tensor.
  - The SFB TMA descriptors still have to be built by the host, so a tensor passing `check_sf_layout`
    (`[E,N,K/128]` int32 MN-major) must still be passed. A CSF kernel never issues TMA on it, so **all layers
    can share one dummy/scratch tensor** (285 MB at E=258, or a VA-aliased tiny physical allocation via
    cuMemMap). Do not keep a full one per layer.
  - Rebuilding `_C` from UP-like source is the alternative. vLLM's exact fork source
    (`/workspace/.deps/deepgemm-src`) is not in the image.

---------------------------------------------------------------------------------------------------

## 5. The "situ" variant

`sm100_fp8_fp4_mega_moe_situ.cuh` is a copy of the main kernel. The only differences are the L1 epilogue
activation (SiTU: sigmoid/tanh with alpha/beta, `kActivationClamp` template removed, static asserts on
alpha/beta) and the kernel name. The SF/TMA/UTCCP/MMA path is identical. It is selected only when
`activation='situ'` (mega/__init__.py:18, 184). vLLM's DeepseekV4MegaMoEExperts never passes `activation`, so it
gets `'swiglu'` (M:763-783). The JIT cache contains **only** `sm100_fp8_fp4_mega_moe.*` entries (7 configs), no
situ. **Not relevant.** Mirror changes there only if you want parity.

---------------------------------------------------------------------------------------------------

## 6. Insertion points for CSF decoding

The natural CSF block is the 512-B kernel chunk `(expert, n_block[128 rows], k_block[4 K-groups])`. In it, lane L
of a warp owns slots `4L..4L+3` = rows `{L, 32+L, 64+L, 96+L}` of the n_block. That is **one 16-B `st.shared.v4`
per lane per k-block, bank-conflict-free**, and each int32 is 4 consecutive K-groups of one row. Because the
producer walks k sequentially per task, a **per-row sequential (along K) codec works**: each lane carries 4 rows'
decoder state in registers across k-blocks. It needs random access only at the (expert, n_block) level, i.e. an
offset table of `E * N/128` entries per GEMM. If CSF is defined on checkpoint rows, the encoder must apply the L1
gate/up interleave and the 4x32 permutation, or the decoder must apply the inverse maps in §1c. Compressing the
**final transformed int32 tensor** (after `transform_weights_for_mega_moe`, e.g. in the hook's
`_finalize_weights`, hook:281) avoids all of that.

### Option A — decode in the SFB producer warp, straight into `smem_sfb[stage]` (recommended in-kernel design)
Change K:793-803 for the routed (non-shared) path:
- wait empty
- all 32 lanes read the next compressed piece (prefetched) and write the 512 B into `smem_sfb[stage]`
- `fence.proxy.async.shared::cta` (UTCCP reads through the async proxy)
- `__syncwarp`
- elected lane issues the weight TMA
- leader: `arrive_and_expect_tx(weight bytes only)`, i.e. drop `+ sizeof(smem_sfb)*2` at K:800

The non-leader's `arrive(0u)` now also publishes its own SMEM writes to the leader's `tcgen05.cp` (cta_group::2
reads both CTAs' SMEM). That arrive must be **release at cluster scope** (`mbarrier.arrive.release.cluster`); the
cutlass `ClusterBarrier::arrive(cta_id)` default needs checking. Keep the shared-expert path on TMA.
- Pros: zero extra HBM traffic (SF bytes read drop by the compression ratio). No scratch. Both CTAs are
  symmetric. No new warps or registers. No SMEM if compressed bytes go to registers.
- Cons: decode latency and throughput sit on the B-producer path. Global loads need software prefetch several
  k-blocks deep (~1 us latency vs ~150 ns per k-block), or a `cp.async.bulk` of the compressed chunk into a small
  SMEM ring (only 2-6 KB free at BLOCK_M 16/32). It requires a side-channel pointer (stats arg) plus one shared
  dummy SF tensor, or a `_C` rebuild. The codec must be cheap: ~16 B output per lane per ~300 cycles.
- Complexity: medium (~150-300 lines in the .cuh, plus Python packing, plus a correctness harness against the
  plain path).

### Option B — decode just-in-time into a shared scratch SF buffer (no MegaMoE change)
A small decode kernel runs before each `fp8_fp4_mega_moe` call (inside the custom op, so it is CUDA-graph
capturable). It expands CSF into **one scratch `[E,N,K/128]` int32 tensor per GEMM shared by all 40 layers**
(layers run sequentially; 285 MB, or 570 MB double-buffered). MegaMoE gets the scratch as `l{1,2}_weights_sf`.
It can decode only experts with routed tokens (presence mask from `topk_ids`, e.g. fused into
`prepare_megamoe_inputs`).
- Pros: kernel untouched, no JIT/ABI risk, easy to validate bit-exactly, any codec works (even serial
  entropy coding, parallel over chunks). Saves ~11.1 GB of the 11.4 GB.
- Cons: extra traffic. Per layer it writes the decoded SF and MegaMoE reads it back (up to 285 MB per layer
  when most hot experts are touched, which is the case at batch ~100+ tokens x topk6). That is roughly
  +5-12% over the MegaMoE weight stream, about 40-70 us per layer, ~2 ms per step at full occupancy. It is also an
  extra launch per layer. Overlap is limited because MegaMoE is persistent on all SMs. EPLB `get_expert_weights`
  (M:640-667) would need adapting (the hook uses a static rowmap anyway).
- Complexity: low (a decode kernel plus Python wiring in the hook).

### Option C — decode per task in the SFB producer into a task-level SMEM buffer, or with idle non-leader warps
Decode the whole (expert, n_block) SF slice once per task (20 KB L1 / 9 KB L2) into SMEM, then memcpy or point
UTCCP at it per k-block. Alternatively, use the idle warps 6/7 on the non-leader CTA as decoders. **This is not
recommended.** 20 KB costs one pipeline stage. The non-leader-only warps break CTA symmetry: the leader's SFB
would still need a decoder, and cross-CTA barriers get added. It only makes sense if the codec cannot decode
sequentially within one k-block's budget.

**Suggested path:** prototype Option B first. It gives a bit-exact correctness and memory-saving baseline with
no kernel risk, and measures the real traffic overhead. Then move to Option A, using the same chunked/per-row
format and the stats-pointer side channel, if B's ~2 ms/step overhead matters. Design the CSF format from the
start so that a 512-B chunk = (e, nb, kb), and each lane's 4 rows decode sequentially along K from an
(e, nb) offset. Then both options use the same encoder.
