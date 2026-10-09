# The Engram host footprint, and how to make it exact

Measured on the M3 lane 2026-10-08/09, engine PID read from `/proc/<pid>/smaps`, host totals from
`/proc/meminfo`. Related: [`../../m3/engram-nvfp4/README.md`](../../m3/engram-nvfp4/README.md), which documents the
same rounding for the NVFP4 Engram overlay and lists the two ways to stop it — one of them is verified at full
size here.

## What the tables are, and what they cost

The two Engram tables (layers 1 and 14) are `engram_num_embeddings = [384006168, 384016682]` rows of 256 values.
Offloaded to the host with `{"cpu_offload": true}` they are allocated as fp8 weights plus e8m0 scales:

```python
torch.empty(part_num_embeddings, 256, dtype=torch.float8_e4m3fn, device="cpu", pin_memory=True)   # weights
torch.empty(part_num_embeddings, 8,   dtype=torch.uint8,          device="cpu", pin_memory=True)   # scales
```

Per layer: 91.55 GiB + 2.86 GiB = **94.42 GiB**, which is exactly what vLLM logs
(`Engram table offloaded to pinned host memory: 384006168 rows x 256, 94.42 GiB per rank`). Two layers:
188.83 GiB of table. The model asks for exactly what it needs.

What it *occupies* is a different number, and that is where 75 GiB goes missing:

| per layer | requested | resident region | slack |
|---|---:|---:|---:|
| weights | 91.55 GiB | 128.00 GiB | 36.45 GiB |
| scales | 2.86 GiB | 4.00 GiB | 1.14 GiB |

Both regions are exact powers of two. torch 2.13 backs `pin_memory=True` with the CachingHostAllocator, which
rounds every block up to a power of two (`pinned_max_round_threshold_mb` is unlimited by default). So 2 x 132 GiB
= 264.0 GiB of `/dev/zero (deleted)` mappings are resident for 188.83 GiB of tables, i.e. **75.17 GiB held for
nothing** — and a memory tool reports that as if it were Engram data.

The pages are resident on purpose: the loader fills the whole buffer with dummy values (1.0 for weights, 127 for
scales) so an empty hash slot cannot return garbage on lookup.

## Verified fix: `use_thp: true`

```
--engram-config '{"cpu_offload": true, "use_thp": true}'
```

takes the packed path (`_allocate_huge_page_storage(weight_bytes + weight_bytes // block_size)`, an `mmap` with
`MADV_HUGEPAGE`) instead of two `torch.empty(pin_memory=True)` buffers, so there is no power-of-two chunk and the
region is exactly `rows x 264 B` per layer. Measured, same lane, same checkpoint:

| | `use_thp: false` | `use_thp: true` |
|---|--:|--:|
| Engram mappings | 2 x 128 GiB + 2 x 4 GiB `/dev/zero` | one 190.04 GiB anonymous region |
| engine RSS | 334.0 GiB | **258.5 GiB** (-75.6) |
| host `Shmem` | 341.0 GiB | 80.6 GiB |
| `AnonHugePages` | - | 190.0 GiB (the `madvise` took) |
| `Mlocked` | 256 kB | 256 kB |

The 64 GiB `/dev/shm/vllm_offload_*.mmap` KV-offload arena is separate and unchanged by this.

The other option in `m3/engram-nvfp4/README.md`, `PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:512`, was
verified there at small scale only; `use_thp` is now verified at full size on the FP8 tables. Both avoid the
rounding; `use_thp` additionally gives huge-page backing, which is why the resident region ends up at 190 GiB
rather than the 189 GiB of NVFP4 tables over there (different quantization, not different rounding).

## Reading host memory on this lane

- `VmLck` per process is 64 kB and host `Mlocked` is 256 kB: **nothing here is actually mlock'd**, despite vLLM's
  "pinned host memory" log line. With no swap on the box the pages stay resident anyway, so the distinction
  matters only for what can be reclaimed.
- `du /dev/shm` shows 123 MB because the offload arena's file is unlinked but still mapped. Read
  `/proc/meminfo`'s `Shmem` (or `free`'s shared column), which reads 80.6 GiB with `use_thp` and 341.0 GiB without.
- The per-mapping breakdown is the only way to attribute the memory honestly: group
  `/proc/<engine-pid>/smaps` by path, summing `Rss` per region. Indented `Rss:`/`Private_Dirty:` lines belong to
  the header above them, and `/proc/<pid>/mem` reads fail inside the container (no `CAP_SYS_PTRACE` by default).
