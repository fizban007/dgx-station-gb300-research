# vLLM's prefix cache is blind to weights

Measured on the M3 lane 2026-10-09. Relevant to anything that changes weights under a live server (this
directory's switch, or a future one), and to reading `prefix_cache_*` counters as a health signal.

## The mechanism

vLLM hashes prefix-cache blocks from the tokens plus `cache_salt`
(`vllm/v1/core/kv_cache_utils.py`, where the salt is folded into the first block's hash and the rest of the chain
follows from it). Nothing in the hash identifies the weights that produced the KV. Two consequences:

- A block cached before a weight change is indistinguishable from one cached after it, so a request can be served
  from KV computed under the other weights.
- The failure is invisible in aggregate: the cache looks healthy (95% hit rate) while a fraction of those hits are
  stale, and it is *concentrated* exactly where reuse is highest — long, shared prefixes.

In `ablit/` the fix is a default salt per mode (`cache_salt = ablit-<applied mode>`), which gives each mode its own
namespace and leaves a client-supplied salt alone. The same idea works for any global weight change: derive the
salt from something that changes with the weights (a model revision, a delta hash, a mode), not from the request.

## Measuring it

Send the same prompt, read `vllm:prefix_cache_queries_total` and `prefix_cache_hits_total`, change the weights,
send it again, read again. Compare the hits/queries *delta* per step, not the totals. With an ~8k-token shared
prefix the signal is unambiguous:

| step | before the fix | after the fix |
|---|---:|---:|
| in-mode, second send | 1.00 | 1.00 |
| first send after the switch | **1.00** (16,000 stale blocks) | **0.00** |
| in-mode in the new mode | 1.00 | 1.00 |
| send after switching back | 1.00 | 1.00 |

Use a distinctive prefix: a short, cold prompt can miss for unrelated reasons and tell you nothing.

## Two traps found while doing this

- **A short prompt never hits.** A 53-block probe repeated three times, with and without a fixed client salt,
  scored 0 hits every time, while an 8k-token shared prefix hit at 1.00 immediately. So the counter can read 0 for
  a perfectly healthy cache.
- **The offload connector on this lane is one-way.** `vllm:kv_offload_total_bytes_total` shows 8.49 GB moved
  `GPU_to_CPU` and **0.0 `CPU_to_GPU`**: blocks leave the GPU and are never restored, so only hot,
  GPU-resident prefixes are reusable. Boot-level hit rate is otherwise fine (95.1% here, 95.0% the boot before).
  Worth knowing before blaming the prefix cache for a latency or quality change.

## Applying the salt

The placement matters, and getting it wrong fails silently:

```python
# wrong: the block hashes are computed *inside* the original call, so this is too late
r = orig(cls, request, block_hasher)
r.cache_salt = f"ablit-{mode}"

# right: mutate the incoming engine request first, then let the original hash it
if not request.cache_salt:
    request.cache_salt = f"ablit-{mode}"
r = orig(cls, request, block_hasher)
```

`block_hasher` being an argument of that function is the hint: the hashing happens inside it. The first version
above produced a `cache_salt` that was set, logged and ignored — the A/B above is what caught it. Cross-check any
such patch by measuring, because a patch that installs cleanly and logs convincingly can still do nothing.
