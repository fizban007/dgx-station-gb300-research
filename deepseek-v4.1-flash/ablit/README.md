# Online switching between the official and abliterated DeepSeek-V4.1-Flash weights

`distributedcognition/DeepSeek-V4.1-Flash-abliterated` (gated, MIT) is the official checkpoint with a
norm-preserving biprojected abliteration applied to two tensor families, in layers 4-36:

- `attn.wo_b` — the attention output projection, the weight that writes attention into the residual stream;
- `ffn.shared_experts.w2` — the shared expert's down projection, the second writer into the residual stream.

That is 132 tensors, **1.65 GiB of a 475.3 GiB checkpoint (0.35%)**. Everything else is byte-identical: all
routed MXFP4 experts, the Engram tables, the vision tower, the DSpark drafter (15 of the 48 shards hash the same
as upstream; the other 33 differ only because they contain the two tensors of the layers in range).

Because the diff is that small, this directory serves the **official checkpoint** and swaps those 132 tensors in
place at runtime:

- **No restart.** One switch writes 2,115.4 MiB and takes 0.4-2.0 s. It is an in-place copy, so every CUDA-graph
  pointer, JIT artefact and split-K tactic stays valid — nothing is recaptured or retuned.
- **No second checkpoint.** The abliterated bytes live in a 1.65 GiB delta next to the official weights
  (`tools/extract_abliterated_delta.py` range-fetches just those 132 tensors from the gated repo).
- **No sidecar work.** The routed experts are unchanged, so the RTX PRO 6000 cold-expert tier, its CSF planes and
  the hot/cold rowmap are untouched.
- **m3 is not edited.** The hook is injected with a one-line `.pth` plus two bind mounts, and it verifies all 132
  resident tensors against the official checkpoint before it will switch anything.

The mode survives reboots: the hook re-reads `data/mode.json` and re-applies the abliterated weights during boot,
after re-running its self-test.

Date: 2026-10-08/09. Image `vllm/vllm-openai:nightly-af7f9488`, checkpoint repo revision `6ee13e7a`.

## Results

Refusal rate, `tools/refusal_eval.py`, 100 prompts from `mlabonne/harmful_behaviors`, greedy, max 8000 tokens,
the same harness run in both modes (report JSONs in [`results/`](results/)):

| metric | official | abliterated |
|---|--:|--:|
| explicit refusals (strong markers) | 99 / 100 | 2 / 98 |
| graded refusals (self-judge, the model in the mode under test) | 100 / 100 | 8 / 100 |
| answers mentioning legality/authorisation | 72 / 100 | 65 / 98 |
| wall time for the same 100 prompts | 95 s | 337 s |
| answers that hit max_tokens | 0 | 5 |

The switch itself:

| | |
|---|--:|
| tensors switched | 132 (33 layers x 2 tensors x weight+scale) |
| bytes written per switch | 2,115.4 MiB |
| switch time, in place | 0.41-1.96 s |
| self-test before the first switch | 132/132 tensors verified in 4.9-6.0 s |
| CUDA graphs recaptured | 0 |

The 2,115.4 MiB is larger than the 1,692.9 MiB delta because the shared expert's weight exists twice in the
serving process — the MegaMoE fused copy and `down_proj.weight` — and both are written.

Host memory, as a side effect of the boot path ([`notes/engram-host-memory.md`](notes/engram-host-memory.md)):

| | before | after |
|---|--:|--:|
| engine RSS | 334.0 GiB | 258.5 GiB (**-75.6**) |
| host `Shmem` (KV-offload arena + Engram) | 341.0 GiB | 80.6 GiB |
| Engram host buffers | 2 x 128 GiB + 2 x 4 GiB power-of-two chunks | 190.0 GiB packed, 190.0 GiB of it `AnonHugePages` |

Quality and safety:

- The self-test is exact, not sampled: every one of the 132 tensors is rebuilt from the official checkpoint and
  compared byte for byte against what is resident, and a switch replays the layout that was verified.
- Refusal rate separates the modes completely in the marker column (99/100 vs 2/98); the two abliterated refusals
  are the most severe prompts in the set (a suicide method and bomb-making). GSM8K-200 in both modes and a
  `MEGA_PEER=0` logprob A/B are not run yet.
- KV cache isolation across a switch is verified (see below). Before the fix, an official-mode request after a
  switch still reused 16,000 blocks of abliterated-era KV.
- The mode is remembered across reboots, including a host reboot: the 2026-10-09 14:30 boot logged
  `ABLIT applied abliterated (seq 7) ... switch #1` after `ABLIT selftest: 132/132`.

## How it works

[DETAILS.md](DETAILS.md) has the full design, the measured resident layouts, the failures found on the way and the
verification. In short:

- **What a switch is.** Two in-place `copy_`-style writes per layer, one for `wo_b` and up to two for the shared
  expert, plus a rebuild of the small scale tensors. Nothing is reallocated, so the pointers baked into the CUDA
  graphs stay valid.
- **Layout discovery instead of assumptions.** The serving process holds these tensors in layouts that are not the
  checkpoint's: `wo_b` and `down_proj` weights are transposed views for the kernel's `[K, N]` operand, the
  `wo_b` scale is an F8_128x4 swizzle of the 32x row-expanded grid, and the fused shared scale is that grid
  packed four e8m0 bytes per word **plus** DeepGEMM's 4x32 UTCCP transpose. The hook enumerates labelled
  candidates, keeps the one that reproduces the resident bytes exactly, and replays that same label later.
- **Mode-scoped cache salt.** vLLM hashes blocks from tokens plus `cache_salt` and knows nothing about weights, so
  KV computed in one mode would be reused in the other. The hook defaults `cache_salt` to `ablit-<applied mode>`
  on requests that bring none (`v1/core/kv_cache_utils.py` folds it into the first block's hash), which gives each
  mode its own cache namespace: a switch cannot read the other mode's KV, and toggling back does not force a
  re-prefill.
- **Control.** `tools/ablit.sh` drains the server (`vllm:num_requests_running` and `_waiting` to 0), writes
  `data/mode.json` with an incremented sequence, waits for the hook to apply it, and then attempts a
  `POST /reset_prefix_cache` (dev-mode only; the salt is what does the work). All switching happens on the engine
  thread between steps and never inside graph capture.
- **Boot injection.** `lane-mounts.sh` prints (or exports) the mounts for `launch-m3.sh` / `swap-to-m3v2.sh`:
  a one-line `ablit.pth` and the hook module, both into the image's site directory, plus the data directory at
  `/ablit/data`. The `.pth` must live in a site directory itself — a copy inside the `vllm` package is never read.
- **It coexists with sf-compress.** That lane compresses the *routed* experts' scale planes (`MEGA_SF_COMPRESS=1`)
  while ablit's 132 tensors are the dense residual writers, so no attribute is written by both and a switch never
  re-encodes a CSF plane; the 132/132 self-test above passes in exactly that configuration. sf-compress patches
  `m3/`, ablit does not, and the two hooks install through different seams (`sitecustomize` vs `ablit.pth`).
  Details, including the UTCCP packing the two share, in [DETAILS.md](DETAILS.md#relationship-to-sf-compress).

## Layout

| path | what |
|---|---|
| `hook/ablit_switch.py` | the hook: registry from the loaded model tree, the self-test, the apply path, the `mode.json` poll, the cache-salt patch |
| `ablit.pth` | one-line injection, mounted into the image's site directory |
| `lane-mounts.sh` | prints/exports `DOCKER_MOUNTS` and `DOCKER_ENV` for the lane launchers; also sets the Engram default |
| `start-server.sh` | full boot (sidecar + container) with the injection: `m3/start-server.sh`, or `m3/swap-to-m3v2.sh` |
| `restart-container.sh` | container-only restart that leaves a live sidecar alone; `--dry-run` prints the docker command |
| `tools/extract_abliterated_delta.py` | range-fetch the 132 changed tensors into `data/` (~1.7 GiB, needs an HF token) |
| `tools/make_test_delta.py` | synthetic stand-in delta (rank-1 projection + MXFP8 requantization) for testing without a token |
| `tools/ablit.sh` | control tool: `status`, `selftest`, `drain`, `set official\|abliterated` |
| `tools/test_ablit_switch.py` | CPU test of discovery, orientation, both apply directions and three failure modes |
| `tools/refusal_eval.py` | post-switch refusal rate; `--judge`, `--rescore FILE`, `--retry-from FILE` |
| `results/` | selftest/state snapshots, the refusal reports (rescored) and the cache-salt A/B |
| `notes/kv-cache-mode-blindness.md` | why a weight switch needs a cache namespace, and the one-way offload finding |
| `notes/engram-host-memory.md` | the Engram host footprint, torch's pinned-allocator rounding, and the verified fix |

## Install

Nothing edits `m3` or the image; the lane is handed extra mounts and environment.

```bash
# full boot with switching on (also the right command after a host reboot: the sidecar is restarted)
./start-server.sh

# container-only restart, sidecar kept alive; inspect the exact docker command first
./restart-container.sh --dry-run
./restart-container.sh

# or inject into your own launcher, without these wrappers (print mode, as in m3/engram-nvfp4/)
DOCKER_MOUNTS="$(lane-mounts.sh)" \
DOCKER_ENV="MEGA_ABLIT=auto MEGA_ABLIT_DIR=/ablit/data" \
  ./swap-to-m3v2.sh
```

Knobs:

| variable | default | meaning |
|---|---|---|
| `MEGA_ABLIT` | `auto` | `auto` = on when `data/abliterated-delta.json` exists, `0` = off, `1` = on and complain if the delta is missing |
| `MEGA_ABLIT_DIR` | `/ablit/data` | where the delta, `mode.json`, `state.json` and `selftest.json` live in the container |
| `MEGA_ABLIT_SELFTEST` | `full` | `light` skips the byte comparison of the 132 weights (shape-only) |
| `REPLAY_OVERLAY` | `1` | `0` also drops the vllm#58132 overlay mounts (see `m3/engram-nvfp4/lane-mounts.sh` for the same pattern) |
| `ENGRAM_CONFIG` | `{"cpu_offload": true, "use_thp": true}` | set here so a reboot does not silently restore the chunked Engram layout |
| `VLLM_SERVER_DEV_MODE` | `0` | `1` also registers `POST /reset_prefix_cache`, which `tools/ablit.sh` calls as a fallback |
| `HOST`, `PORT` | `127.0.0.1`, `8001` | the address the lane serves on and the tools talk to. The station this was written on binds vLLM with `--host <tailnet addr>`, where localhost does not answer, so there `HOST` had to be set; a checkout whose launcher uses the default `--host` needs nothing. `tools/refusal_eval.py` takes the same thing as `--base` |

Check the boot log for (in this order):

```
ABLIT on: 132 tensors, 1692.9 MiB, layers 4-36, dir /ablit/data, selftest full          (once per Python process)
ABLIT armed poll sites: vllm.v1.worker.gpu.model_runner, vllm.v1.worker.gpu_model_runner, vllm.v1.worker.gpu_worker
ABLIT mode-scoped cache salt installed on vllm.v1.request.Request
ABLIT poll installed on vllm.v1.worker.gpu.model_runner.GPUModelRunner.execute_model
ABLIT model tree DeepseekV41ForCausalLM: 80 target linears, 40 MoE modules, 40 shared experts
ABLIT selftest: 132/132 tensors verified in 5.95s; recipes ['deepep:inline_utccp', 'swizzle:swizzle_u8/expand32']
ABLIT applied abliterated (seq 7) in 1.955s, 2115.4 MiB, switch #1
```

A missing `ABLIT on:` line means the injection is not mounted; `ABLIT on:` present with no `poll installed on ...`
line means the poll site does not exist in that image (this fork imports Model Runner **V2**, which is why three
sites are armed). Either way the server boots normally and switching stays off.

## Tests

| test | how | result |
|---|---|---|
| hook logic, CPU | `CUDA_VISIBLE_DEVICES= python tools/test_ablit_switch.py` (needs torch; a synthetic delta from `make_test_delta.py`) | 43 checks pass: both scale conventions and both weight orientations discovered, apply/restore exact for the normal and transposed resident and for the fused shared pair, corrupted-resident and size-mismatch cases rejected |
| extractor, against the ungated upstream repo | `python tools/extract_abliterated_delta.py --repo deepseek-ai/DeepSeek-V4.1-Flash --out-dir /tmp/x --limit 2` | identity mode: headers, offsets and byte comparison all reproduce the local shards; 10/10 spot checks identical; the 93 GiB Engram and 1.26 GiB embed/head tensors skipped by the size caps |
| extractor, against the gated abliterated repo | `python tools/extract_abliterated_delta.py` (needs an HF token) | 132/132 tensors differ from upstream, 0 identical, 10/10 spot checks identical, 0 shard-header offset mismatches, 1,692.9 MiB in 201 s, revision `6ee13e7a`; `--verify-only` re-checks clean |
| self-test in the serving process | boot and read the log | 132/132 verified (4.9-6.0 s), twice across reboots |
| the switch | `tools/ablit.sh set abliterated` | applied in 0.41-1.96 s, 2,115.4 MiB, no graph recapture; `set official` restores |
| refusal rate | `python tools/refusal_eval.py --limit 100 --max-tokens 8000 --judge` in each mode | see [Results](#results); `--rescore` re-classifies a stored report after a marker-list change |
| KV isolation | 8k-token shared prefix: send, `tools/ablit.sh set official`, send again, read `vllm:prefix_cache_{queries,hits}_total` | cross-mode ratio 1.00 -> **0.00** after the salt fix, in-mode 1.00, and the abliterated cache still hits at 1.00 after the round trip |
| reboot | host reboot, then `start-server.sh` | sidecar re-prepared 40 layers in 162 s (94.5 GB); self-test 132/132; mode re-applied automatically |

## Limits

- **A global toggle at a request boundary.** Per-request or mixed-batch switching is out of scope: there is one
  CUDA graph per batch size with pointers baked in, so a mixed batch would need two graph sets and batch
  splitting. `tools/ablit.sh` drains in-flight requests first.
- **The delta needs the gated repo.** An HF token with access to `distributedcognition/...` is required once;
  `make_test_delta.py` builds a stand-in (marked `synthetic` in its index and in every log line) for testing.
- **Abliteration strength is not interpolated.** `refusal_directions.pt` (822 KB, in the repo) allows re-ablating
  at any scale in seconds, which is the natural follow-up: a delta per strength, or a lerp of the deltas.
- **The 2.1 GiB write is host-to-device traffic**, so a switch costs ~30 ms of PCIe time at best; keeping both
  variants resident in HBM would make it a device-to-device copy (~0.3 ms) for 1.65 GiB of HBM.
