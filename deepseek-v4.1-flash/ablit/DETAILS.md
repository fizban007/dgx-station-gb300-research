# ablit: design, measurements and failures

Long-form companion to [README.md](README.md). Everything here was measured on the M3 lane on the GB300 station
(image `vllm/vllm-openai:nightly-af7f9488`, checkpoint repo revision `6ee13e7a`) on 2026-10-08/09, unless a date is
given otherwise.

## The diff

`distributedcognition/DeepSeek-V4.1-Flash-abliterated` is gated; its model card states a norm-preserving
biprojected abliteration applied to `attn.wo_b` and `ffn.shared_experts.w2`, with the FP8 32x32 ue8m0 blocks
dequantized, ablated and requantized at power-of-two scales, and routed FP4 experts, Engram and vision
byte-identical. The measurements below confirm that from three directions.

**Shard hashes.** Per-shard LFS `sha256` from the HF API against the upstream `source_sha256` recorded in
`csf-port/ext/hf/manifest.json`: 15 of 48 shards byte-identical, 33 differ. Those 33 are exactly the shards that
hold `wo_b` / `shared_experts` tensors for layers 4-36; layers 0-3 and 37-39 hold those tensors and are unchanged.
The local checkpoint was re-hashed (shards 1, 7) and matches the receipts, so it is upstream.

**Payload.** From the local safetensors headers (one layer per shard, shard 3 = layer 0):

| tensor | dtype | shape | per layer | x 33 layers |
|---|---|---|---:|---:|
| `layers.N.attn.wo_b.weight` | F8_E4M3 | [5120, 8192] | 40 MiB | 1320.0 MiB |
| `layers.N.attn.wo_b.scale` | F8_E8M0 | [160, 256] | 40 KiB | 1.3 MiB |
| `layers.N.ffn.shared_experts.w2.weight` | F8_E4M3 | [5120, 2304] | 11.25 MiB | 371.2 MiB |
| `layers.N.ffn.shared_experts.w2.scale` | F8_E8M0 | [160, 72] | 11.25 KiB | 0.4 MiB |

1,692.9 MiB total, 0.348% of the 475.3 GiB checkpoint, and 0.0% of it in the routed experts — which is why no
sidecar, rowmap or CSF work is involved in a switch.

**Range-fetched bytes.** `tools/extract_abliterated_delta.py` requests exactly those 132 byte ranges and compares
each against the local checkpoint: 132 changed, 0 identical, 10/10 spot checks identical (routed experts in layers
0, 4 and 37, the DSpark drafter's router bias, shared gate_up, dense attention, vision, final norm, an Engram key
table), 0 shard-header offset mismatches, 1,692.9 MiB in 201 s. Every range request is verified to be a `206` with
an exact `Content-Range`, so a CDN that ignores ranges cannot turn this into a 475 GiB download, and every
spot-check tensor is size-capped (an Engram table here is 93 GiB; embed/head are 1.26 GiB).

## Where the bytes live while serving

The serving process does not hold these tensors in checkpoint layout. Read off the live model by the hook's own
`ABLIT plan inputs (layer 4)` line, which prints the type, shape and dtype of every tensor the switch touches:

```
wo_b.weight=Tensor(8192, 5120) float8_e4m3fn        wo_b.weight_scale=Tensor(1310720,) uint8
down_proj.weight=Tensor(2304, 5120) float8_e4m3fn   down_proj.weight_scale=Tensor(368640,) uint8
_transformed_shared_l2_weights=tuple(len=2: Tensor(5120, 2304) float8_e4m3fn, Tensor(5120, 18) int32)
```

| checkpoint tensor | resident form | how the switch rebuilds it |
|---|---|---|
| `wo_b.weight` | `[K, N]` transposed view of the checkpoint bytes | direct copy, or the transpose of it |
| `wo_b.scale` | flat uint8 (1,310,720 = 5120 x 256), F8_128x4 swizzle of the 32x row-expanded grid | `swizzle_mxfp8_scale(expand32(ckpt), M, K)` |
| `shared.w2.weight` | two storages: the fused copy `[5120, 2304]` in checkpoint order and `down_proj.weight` `[2304, 5120]` | both written, direct or transposed |
| `shared.w2.scale` (fused) | int32 `(5120, 18)`: the row-expanded grid packed four e8m0 bytes per word, **plus** DeepGEMM's 4x32 UTCCP transpose | `transform_sf_into_required_layout(...)` then `_transpose_sf_for_utccp` (candidate label `deepep:inline_utccp`) |
| `shared.w2.scale` (linear) | flat uint8 `(368640,)` holding the 5120x72 grid | `expand32(ckpt)` |

Two lessons are baked into the hook because of this table:

- **Enumerate and verify, do not assume.** The fused scale has a transpose applied on top of what the model's own
  `_prepare_shared_expert_scale` produces — that helper was reached and its output did *not* byte-match the
  resident (`helper_ckpt` was refused by its shape gate, `helper_expand` simply differed). Six labelled candidates
  are generated and the one that reproduces the resident exactly wins; the switch then **replays that label**, so
  what gets written is what was verified.
- **A verified hit is enough.** Every hit is byte-equal to the resident, so several equivalent candidates are not
  ambiguity. An earlier version required exactly one match and would have rejected a tie.

## The switch

- **In place.** `dst.copy_(candidate)` on the resident storages, both existing by construction, so every CUDA-graph
  pointer, JIT artefact and split-K tactic stays valid. Measured 0.41-1.96 s per switch, writing 2,115.4 MiB: the
  132 tensors, with the shared expert's weight written twice because it exists twice.
- **Validated before the first write.** Shapes, dtypes, numels and the presence of every destination are checked
  for all 132 tensors first; a failure aborts with nothing written, so a half-applied model is impossible. Any
  exception is caught and logged — a switch bug cannot take the server down (and one did not: see below).
- **On the engine thread, between steps.** The poll is a wrapper on `GPUModelRunner.execute_model`, which runs
  before any layer of a step and never inside graph capture. Three sites are armed because this fork imports
  *either* Model Runner V2 (`vllm.v1.worker.gpu.model_runner`, the default) or V1
  (`vllm.v1.worker.gpu_model_runner`); the first boot below shows what happens with one site too few.
- **Order of operations** (`tools/ablit.sh set <mode>`): drain (`vllm:num_requests_running` and `_waiting` to 0),
  write `data/mode.json` with `seq+1`, poke a 1-token request if the engine is idle, wait for the hook to record
  that seq in `state.json`, then attempt `POST /reset_prefix_cache` (404 without dev mode; the cache salt is what
  actually isolates the modes).

## What the self-test proves, and the failures it caught

The self-test runs at the end of loading and reports per tensor: the official bytes are read from the checkpoint,
the derived resident form is rebuilt, and the bytes are compared. All 132 must match; a mismatch disables
switching and leaves the server serving. It runs on the engine thread after the model is ready, outside graph
capture (`MEGA_ABLIT_MIN_CALLS` steps, default 2), so it needs no traffic.

Errors found in this order, each now covered:

1. **No `ABLIT` lines at all.** The launcher mounted the injection `.pth` at `$SITE/vllm/ablit.pth`, i.e. inside
   the `vllm` package, where `site.py` never reads `.pth` files. The throwaway-container test had used the correct
   path by hand, so it passed while the launcher was wrong; the test now feeds a throwaway container the
   launcher's own `DOCKER_MOUNTS`, and the launcher refuses to boot if the injection is not landing in `$SITE`.
2. **`ABLIT on:` present, nothing else.** The poll site was armed on the V1 model-runner module only, and
   `gpu_worker` imports exactly one of V2/V1. Fixed with three sites plus an `armed poll sites` line and one
   `poll installed on <module>.<class>` line per module that loads, so a future move shows up as a missing line.
3. **`plan build failed: IndexError`.** One surprising tensor aborted the whole plan. Now each tensor is isolated
   (`exception: ...` is recorded for it), the failure logs three traceback frames, and a
   `ABLIT plan inputs (layer N)` line prints the shapes the plan sees before it starts.
4. **33 of 132 failed: the two scale families.** `no rebuild matched the fused scale (5120, 18) int32` and
   `no expansion matched weight_scale (368640,) uint8` — the plan inputs line above is what resolved it.
5. **A scorer bug, not a hook bug.** The first official-mode refusal run scored 2/100 by markers while its own
   printed examples read "I can't help with that": the marker list was ASCII and the model emits U+2019. Text is
   normalised before matching now, and `--rescore FILE` re-classifies a stored report without re-generating
   (2% -> 99% for official, 0% -> 2% for abliterated).

## KV cache isolation

vLLM hashes blocks from tokens plus `cache_salt` and knows nothing about weights, so KV computed under one mode is
reused under the other. Measured with an ~8k-token shared prefix, sending the same prompt, switching, sending
again, and reading `vllm:prefix_cache_{queries,hits}_total` around each send:

| step | ratio (hits / queries) | before the fix | after the fix |
|---|---:|---:|---:|
| in-mode, second send | 1.00 | 1.00 | 1.00 |
| first send after switching mode | 0.00 (fresh) | **1.00 (stale)** | **0.00** |
| in-mode in the new mode | 1.00 | 1.00 | 1.00 |
| send after switching back | 1.00 | 1.00 | 1.00 |

The fix is a patch on `vllm.v1.request.Request.from_engine_core_request` that defaults `cache_salt` to
`ablit-<applied mode>` for requests that bring none, so each mode has its own namespace; a client that brings its
own salt keeps it. The first version set the salt on the *returned* `Request` and had no effect at all — the block
hashes are computed *inside* that function (`block_hasher` is one of its arguments), which the A/B showed plainly
as 16,000 stale hits. It now sets the salt on the incoming `EngineCoreRequest` before calling the original.

Two further findings, in [notes/kv-cache-mode-blindness.md](notes/kv-cache-mode-blindness.md):

- **Short prompts never hit; long shared prefixes always do.** A 53-block probe repeated three times — even with a
  fixed client salt — scored 0 hits, while an 8k-token shared prefix scored 1.00 immediately.
- **The offload connector is one-way.** `vllm:kv_offload_total_bytes_total` shows 8.49 GB moved `GPU_to_CPU` and
  **0.0 `CPU_to_GPU`**, so a block that leaves the GPU is never restored, and only hot, GPU-resident prefixes are
  reused. Boot-level hit rate is otherwise healthy: 95.1% (94.8M / 90.1M) in this boot, 95.0% in the previous one.

## Host memory and the Engram

Enabling the switch turns on an Engram layout change in the same boot (`ENGRAM_CONFIG` default in
`lane-mounts.sh`), because the two are otherwise easy to conflate when reading memory:

| mapping (engine PID, `smaps`) | chunked layout | packed THP layout |
|---|---|---:|
| `/dev/zero (deleted)`, Engram host buffers | 2 x 128 GiB + 2 x 4 GiB | 190.04 GiB anon (one region) |
| `/dev/shm/vllm_offload_*.mmap` (KV offload arena) | 64.00 GiB | 64.00 GiB |
| engine RSS / host `Shmem` | 334.0 / 341.0 GiB | 258.5 / 80.6 GiB |
| `AnonHugePages` / `Mlocked` | - / 256 kB | 190.0 GiB / 256 kB |

The Engram tables are 384,006,168 and 384,016,682 rows of 256 values (the model config's
`engram_num_embeddings`), i.e. 94.42 GiB of fp8 weights plus e8m0 scales **per layer**, allocated with
`torch.empty(..., pin_memory=True)`. The default path lands each allocation in a power-of-two chunk of that
allocator (a 94.42 GiB request pins 128 GiB), so 188.83 GiB of tables occupy 264 GiB; the table's own shape is
correct in both layouts. `use_thp: true` takes the packed `mmap` path instead, which is exactly
`rows x 264 B` per layer. **This is the option `m3/engram-nvfp4/README.md` lists as "untested at full size" —
it is tested here, at full size, on the FP8 tables: 264.0 -> 190.0 GiB resident with nothing left in the caching
allocator.** The finding and its arithmetic are in [notes/engram-host-memory.md](notes/engram-host-memory.md).

Note the "pinned" wording: vLLM logs `Engram table offloaded to pinned host memory`, but `VmLck` is 64 kB per
process and host-wide `Mlocked` is 256 kB. Nothing is mlock'd, and the box has no swap, so those resident pages
stay regardless. `du /dev/shm` also under-reports: the offload arena's file is unlinked but mapped, which is why
`/proc/meminfo`'s `Shmem` (or `free`'s shared column) is the number to read.

## Relationship to sf-compress

Both run in the same process on this lane — the boot this was written against has `MEGA_SF_COMPRESS=1` and logs
`SF compress: 10.46 GiB raw -> 1.72 GiB compressed + 0.26 GiB shared scratch` — and they do not interfere,
which is worth stating because the two touch the same MoE module.

- **Disjoint tensors.** sf-compress rewrites the *routed* experts' scale planes:
  `st.sf_raw = (self._transformed_l1_weights[1], self._transformed_l2_weights[1])` and then swaps `(w, csf[i])`
  in for those two. ablit's 132 tensors are `attn.wo_b` and `ffn.shared_experts.w2`, i.e. the dense residual
  writers, held in `_transformed_shared_l1_weights` / `_transformed_shared_l2_weights` and the linear modules.
  No attribute is written by both, so there is nothing to re-encode: the routed experts are byte-identical
  between the two checkpoints, and a switch never touches a CSF plane.
- **The proof is the self-test.** `ABLIT selftest: 132/132 tensors verified` runs in exactly this configuration,
  comparing every tensor ablit writes against the official checkpoint while the routed scales are compressed in
  memory. A collision would show up there as a byte mismatch.
- **Shared vocabulary, not shared state.** The fused shared-expert scale ablit has to rebuild is UTCCP-packed —
  the same `_transpose_sf_for_utccp` idea the routed scales use, which is why `hook/ablit_switch.py`'s
  `_utccp_transpose` is a faithful copy of `vllm/third_party/deep_gemm/mega/__init__.py::_transpose_sf_for_utccp`
  (imported from there when available). sf-compress's `hook/sf_compress.py` packs UE8M0 planes for the routed
  path; ablit reads the packed form for the shared path. Same layout idea, different tensors.
- **Different install paths, on purpose.** sf-compress patches `m3/` (its `install.sh`, hook copies, and the
  `MEGA_SF_COMPRESS` / `PEER_CSF` knobs). ablit adds bind mounts and a `.pth` and leaves `m3/` alone — a grep for
  `ablit` under `m3/` returns nothing. Two hooks, one process: sf-compress installs through `sitecustomize`
  (`MEGA_HOOK=/w/mega_peer_hook.py`) and ablit through `ablit.pth`, which `site.py` processes first, so ablit's
  import finders are armed before the lane's own hook is installed. ablit's plan is built much later (after
  loading), so it discovers whatever the lane's other hooks left resident rather than assuming checkpoint layout —
  that is why it finds the F8_128x4 swizzle at all.
- **Memory works in one direction.** sf-compress frees HBM (10.46 -> 1.72 GiB of hot-expert scales here, plus the
  sidecar's saving, which pays for the 254/130 split). ablit spends no extra HBM: the delta lives on the host,
  and a switch writes 2,115.4 MiB through it without keeping a second copy resident. The optional
  "keep both variants in HBM" variant (+1.65 GiB, ~0.3 ms switches) is off by default, and would be the one
  choice that takes back part of what sf-compress saved.

## Boots and reboots

- **Full boot** (`start-server.sh`): the sidecar starts first and waits for its `serving` line — after a host
  reboot it re-prepared 40 layers in 162 s (94.5 GB allocated) — then the container starts with the extra mounts.
- **Container-only restart** (`restart-container.sh`): reads the settings of the container it replaces
  (`docker inspect` env minus the image's env, `--gpu-memory-utilization` and the KV-offload flags from its
  `Cmd`, plus `MEGA_ROWMAP`), requires the live sidecar to be on the same rowmap, stops the container gracefully,
  waits 10 s for the driver to release the old context, and retries the launch once if the container exits.
  Verified against a live container: 24/24 vLLM flags identical, every run-time env var reproduced, 17/17 mounts
  with the sole difference being an injection destination that needed fixing.
- **The failure the retry exists for.** A container-only restart once died at EngineCore init with
  `torch.AcceleratorError: CUDA error: operation not permitted`, immediately after `docker rm -f` — the same
  script had worked three times before, so it is a race with the dying context. The script now stops gracefully,
  settles, and retries once, printing the engine error if it happens again.
- **Reboot behaviour.** `data/mode.json` persists, so the hook re-applies the remembered mode during boot (the
  2026-10-09 14:30 boot logged `ABLIT applied abliterated (seq 7) ... switch #1` right after
  `ABLIT selftest: 132/132 tensors verified in 5.95s`), and it re-applies the correct weights because the mode is
  read after the self-test, not instead of it.

## Numbers appendix

```
self-test                 132/132 verified, 4.92 s (first boots 5.95 s)
switch                    applied abliterated (seq 3) in 1.941 s, 2115.4 MiB, switch #1
                          applied official (seq 8) / abliterated (seq 9) in ~0.4 s each
boot prefix-cache hit rate 94.8M queries / 90.1M hits = 95.1%   (previous boot 35.5M / 33.7M = 95.0%)
KV A/B, cross-mode         dhits 16,000 of 16,043 queries -> after the fix 0 of 16,043
KV offload                GPU_to_CPU 8.49 GB, CPU_to_GPU 0.0
engram host, chunked      334.0 GiB RSS, 341.0 GiB Shmem, 2 x 128 GiB + 2 x 4 GiB /dev/zero
engram host, packed THP   258.5 GiB RSS, 80.6 GiB Shmem, 190.04 GiB anon, 190.0 GiB AnonHugePages
refusal, markers          99/100 official, 2/98 abliterated
refusal, self-judge       100/100 official, 8/100 abliterated
refusal, wall time        95 s official, 337 s abliterated (same 100 prompts)
extraction                132/132 changed, 0 identical, 1692.9 MiB in 201 s, revision 6ee13e7a
```
