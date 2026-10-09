# Results

Evidence from the runs described in [../README.md](../README.md) and [../DETAILS.md](../DETAILS.md). Paths recorded inside these snapshots are home-relative (`~/...`).
The 1.65 GiB
delta binary itself is not committed (`data/` is gitignored); `tools/extract_abliterated_delta.py` rebuilds it into
`data/`, and `abliterated-delta-index.json` here is the index it writes.

| file | what |
|---|---|
| `selftest.json` | what the serving process reported at boot: 132/132 tensors verified, the recipes it chose per scale family, and the wall time |
| `state.json` | the hook's own state file: applied mode and sequence, switch count, last switch time, errors |
| `abliterated-delta-index.json` | the delta's per-tensor index: 132 tensors with shape, dtype, sha256, source shard and offset, and the repo revision pinned |
| `delta-report.json` | the extractor's findings: changed/identical counts and the spot checks of tensors that must be unchanged |
| `refusal-abliterated-*-rescored.json` | 100 harmful-instruction prompts in abliterated mode: every prompt, every answer, the refusal marker that fired and the self-judge grade |
| `refusal-official-*-rescored.json` | the same harness and prompt set in official mode |

## Cache-salt A/B (2026-10-09)

An ~8k-token shared prefix, sent twice per mode with a mode switch between, reading
`vllm:prefix_cache_{queries,hits}_total` around each send. Deltas per send:

```
mode              send                        dqueries   dhits   ratio
abliterated       #1 cold                        16,043       0    0.00
abliterated       #2 same mode                   16,043  16,000    1.00
  switch -> official
official          #1 after the switch            16,043       0    0.00    <- was 16,000 (ratio 1.00) before the fix
official          #2 same mode                   16,043  16,000    1.00
  switch -> abliterated
abliterated       #3 after the round trip       16,043  16,000    1.00    <- its own cache survived
```

The pre-fix run differed in exactly one row: `official #1` scored `dhits=16,000`, i.e. an official-mode request
served from abliterated-era KV. Cause and fix: [../notes/kv-cache-mode-blindness.md](../notes/kv-cache-mode-blindness.md).

## Host memory (2026-10-08/09)

Engine process, `/proc/<pid>/smaps` grouped by mapping, plus `/proc/meminfo`:

```
chunked Engram (use_thp false)   engine RSS 334.0 GiB   2 x 128 GiB + 2 x 4 GiB /dev/zero, 64.00 GiB KV-offload shm
                                 host Shmem 341.0 GiB   Mlocked 256 kB   AnonHugePages -
packed THP Engram (use_thp true) engine RSS 258.5 GiB   190.04 GiB anon (one region), 64.00 GiB KV-offload shm
                                 host Shmem  80.6 GiB   Mlocked 256 kB   AnonHugePages 190.0 GiB
```

The Engram tables themselves are 188.83 GiB; the 75.17 GiB difference is power-of-two rounding by torch's pinned
host allocator. Details and arithmetic: [../notes/engram-host-memory.md](../notes/engram-host-memory.md).

## Boot log snapshot

The lines a boot of this lane emits, in order (see [../README.md](../README.md#install) for what each one means):

```
ABLIT on: 132 tensors, 1692.9 MiB, layers 4-36, dir /ablit/data, selftest full
ABLIT mode-scoped cache salt installed on vllm.v1.request.Request
ABLIT poll installed on vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model
ABLIT model tree DeepseekV41ForCausalLM: 80 target linears, 40 MoE modules, 40 shared experts
ABLIT selftest: 132/132 tensors verified in 5.95s; recipes ['deepep:inline_utccp', 'swizzle:swizzle_u8/expand32']
ABLIT applied abliterated (seq 7) in 1.955s, 2115.4 MiB, switch #1
```

To regenerate any of this: `tools/ablit.sh status`, `python tools/refusal_eval.py --limit 100 --max-tokens 8000
--judge` per mode, and `python tools/refusal_eval.py --rescore <report.json>` after a marker-list change.
