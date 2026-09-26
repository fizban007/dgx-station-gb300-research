# MiMo-V2.6-Pro three-tier lane: details

This file backs the [README](README.md). Every number also appears in [`results.jsonl`](results.jsonl), with its
source file. Numbers marked *(notes)* come from session notes taken while the lane ran; their raw output was
not kept.

## Hardware and software

- One DGX Station GB300 (SM 10.3, 152 SMs, about 250 GiB of HBM visible to CUDA) and Grace with LPDDR5X over
  NVLink-C2C. See [`../docs/station.md`](../docs/station.md).
- One RTX PRO 6000 Blackwell Max-Q (SM 12.0, 96 GB) in the same host. It is reached only through pinned host
  memory: no P2P or NVLink to the GB300.
- Checkpoint: `XiaomiMiMo/MiMo-V2.6-Pro-RL` (527 GiB, MXFP4 routed experts, FP8 dense, BF16 `o_proj`).
- GB300 side: `vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098` plus the bind-mounted files in
  [`hook/`](hook/). TP1, UVA offload of the routed experts (`--cpu-offload-gb 320`), CUDA graphs `FULL_DECODE_ONLY`.
- Sidecar: [b12x](https://github.com/local-inference-lab/b12x) `a7d7d29b2ef8869086e0ceaa787321f17544e3c9` in a
  host venv (torch 2.13, Triton). Fused MoE mode `w4a8_mx`.

## What changed relative to Al-ENGR's v23

His recipe is [J-M-Recipes/recipes `mimo-v2.6-pro-vllm-uva-hotsplit`](https://github.com/J-M-Recipes/recipes/tree/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/dgx-station-gb300/mimo-v2.6-pro-vllm-uva-hotsplit)
(MIT, commit `dffd01cc`).

| | Al-ENGR v23 | this lane |
|---|---|---|
| vLLM image | nightly d05da62e | nightly 29468dde |
| his upstream fixes (vllm#58142 fused FP8 qkv pairing, #58184 truncation guard, #58185 exact-size UVA pinning) | in his image and model file | backported onto 29468dde ([`backport/`](backport/), [`hook/overlay/vllm/`](hook/overlay/vllm/)) |
| KV-cache group padding | stock | vllm#58207 backport ([`hook/overlay/kv_cache_utils.29468.diff`](hook/overlay/kv_cache_utils.29468.diff)) |
| expert placement | two tiers: HBM, Grace | three tiers: HBM, RTX PRO 6000, Grace ([`hook/plan_tiers.py`](hook/plan_tiers.py)) |
| `--max-num-batched-tokens` | 2,048 | 8,192. Each prefill chunk re-reads every Grace expert, so larger chunks mean fewer sweeps. |
| `--max-num-seqs` | 8 | 16 |
| `--max-model-len` | 262,144 (v23) | 1,048,576 |
| prefix caching | off | on |
| `--generation-config` | auto (applies the checkpoint's `max_new_tokens=2048`) | vllm |
| FA4 compile cache | not persisted | persisted (`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`) |

## Versions measured

| version | when (EDT) | expert GEMMs | Grace experts | peer send | rowmap |
|---|---|---|---|---|---|
| v2 | 2026-09-25 ~16:50 | Marlin, two banks | Marlin reads them through UVA | torch ops | [`rowmap-3tier-v2.json`](hook/rowmap-3tier-v2.json): 152.8 GiB hot, 90.2 GiB peer |
| v3 | ~17:50 | Marlin | staged into HBM for batches with T·top_k ≤ 128 ([`stage_grace.py`](hook/stage_grace.py)) | fused, 3 kernels | v2 |
| v4 | ~19:30 | FlashInfer TRT-LLM MXFP4×MXFP8, bank-local ids ([`trt_banks.py`](hook/trt_banks.py)) | staged (decode) or slab-copied (prefill) | fused | not recorded; see below |
| v5 | ~20:26 | TRT-LLM, as v4, plus an FP8 KV cache | as v4, `STAGE_SLOTS=256` | fused | v3 |
| v6 | ~21:01 | as v5, plus DFlash speculative decoding, k=3 (the checkpoint's 5-layer drafter) | as v4, `STAGE_SLOTS=256` | fused | v3 |

v6 is the launcher's default configuration, as [`up.sh`](up.sh) records it. The lane's session reported the
other two: v5 is v6 with `SPEC=none`, and v4 used `MOE=trtllm`, `KVDTYPE=auto` (BF16) and
[`rowmap-3tier-v2.json`](hook/rowmap-3tier-v2.json) (152.8 GiB hot). [`rowmap-3tier-v3.json`](hook/rowmap-3tier-v3.json)
(115 GiB hot, 90.2 GiB peer) was written at 19:36, after the v4 run.

KV-cache capacity, from the lane's session notes (boot logs not kept): 413,542 tokens for v4 (BF16), 2,407,731
for v5 (FP8) and 2,044,736 for v6 (FP8 plus the DFlash drafter's cache). With a 1M-token context, v5 holds 2.3
full-length requests.
[`stage_grace.py`](hook/stage_grace.py) and [`trt_banks.py`](hook/trt_banks.py) are published as they were at
20:52: after v5 and before v6.

### FP8 KV cache for MiMo (v5)

MiMo mixes 192-dim K with 128-dim V ("DiffKV"). Three changes make an FP8 KV cache work:

- The model file gives global-attention layers a cache config without the sliding window
  ([`hook/mimo_v2.29468.diff`](hook/mimo_v2.29468.diff)).
- The vLLM FlashAttention backend checks FA4 support with the real V head size
  ([`hook/overlay/attn/flash_attn.diff`](hook/overlay/attn/flash_attn.diff)).
- FA4's interface keeps `tile_n=128` for FP8 DiffKV split-KV decode
  ([`hook/overlay/fa4/interface.diff`](hook/overlay/fa4/interface.diff)). According to the comment in that
  change, the `tile_n=64` split-KV variant produced wrong results from 8K context, and illegal memory accesses
  by about 60K. The same change is open as a draft upstream PR,
  [Dao-AILab/flash-attention#2918](https://github.com/Dao-AILab/flash-attention/pull/2918).

[`hook/tests/test_fa4_fp8_bigpool.py`](hook/tests/test_fa4_fp8_bigpool.py) checks FP8 decode against a BF16-cache
reference. It covers both low block ids and blocks at the tail of a full-size pool, which is the repository's
big-page-id rule. [`hook/tests/time_fa4_decode.py`](hook/tests/time_fa4_decode.py) times FP8 single-split
against BF16 auto-split decode. Neither test's output was saved. The KV capacity gained by FP8 was not recorded
either.

## Placement

[`hook/plan_tiers.py`](hook/plan_tiers.py) ranks every (layer, expert) cell by its normalised decode routing share
in Al-ENGR's histogram. It fills HBM first, then the sidecar, and leaves the rest in Grace. All cells have
the same size (20,054,016 bytes, including scales), so ranking by share equals ranking by share per byte.

| rowmap | HBM cells | peer cells | Grace cells | decode share HBM / peer / Grace | peer experts per layer |
|---|--:|--:|--:|---|---|
| v2 | 8,181 | 4,829 | 13,486 | 59.7% / 17.3% / 23.0% | 38–126 |
| v3 | 6,157 | 4,829 | 15,510 | 50.6% / 19.9% / 29.6% | 36–129 |

Our plan budgets HBM with checkpoint cell bytes (19.1 MiB). hotsplit's own units are 18.0 MiB, so "152.8 GiB"
gives 8,181 hot cells here, versus 8,692 in his plan. That leaves about 8 GiB more for KV cache: 361K tokens
at 1M context *(notes)*.

## Sidecar

- The 96 GB RTX PRO 6000 is full. A 90.2 GiB peer budget (4,829 experts) prepares in 243 s. After capturing
  966 CUDA graphs it has 93.1 GiB allocated and 0.7 GiB free
  ([boot log](results/logs/peer_server_mimo-trt-v4-boot.log)).
- Before it prints `serving`, the sidecar self-tests the exact serving path against a float32 reference built
  from freshly read checkpoint bytes. Cosine is 0.9997–1.0001 on layers 1, 35 and 69 (same log).
- Per-bucket compute time on the sidecar during the v4 boot ([`peer_stats-trt-v4.json`](results/logs/peer_stats-trt-v4.json)):

| row bucket | 1 | 2 | 4 | 8 | 16 | 32 | 4,096 | 8,192 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| mean µs | 91 | 110 | 124 | 128 | 147 | 287 | 2,783 | 5,460 |

- The same table for the v6 boot, over 2.4 million calls
  ([`peer_stats-dflash3-v6.json`](results/logs/peer_stats-dflash3-v6.json)). Most calls land in the 4-row
  bucket, which fits DFlash k=3: each decode step verifies 4 tokens per stream.

| row bucket | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 4,096 | 8,192 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| calls | 202,077 | 308,494 | 1,707,639 | 52,570 | 78,445 | 31,902 | 46,598 | 1,433 | 8,302 |
| mean µs | 87 | 107 | 137 | 187 | 283 | 367 | 512 | 2,730 | 5,281 |

### How long the GB300 waits (v3)

Source: [`results/logs/peerwait-3tier-v3.log`](results/logs/peerwait-3tier-v3.log) ([`bench/peer_wait.py`](bench/peer_wait.py)).

| C | tok/s | ms per pass | GB300 wait, ms per pass | µs per wait | rows per layer | timeouts |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 32.3 | 30.92 | 0.176 (0.6%) | 3.2 | 0.89 | 0 |
| 4 | 61.0 | 65.32 | 0.300 (0.5%) | 4.4 | 3.58 | 0 |
| 8 | 79.6 | 100.15 | 0.363 (0.4%) | 5.3 | 7.18 | 0 |
| 16 | 100.3 | 158.84 | 0.806 (0.5%) | 11.7 | 14.18 | 0 |

## Where C1 decode time goes (v2, *notes*)

The table is from a torch-profiler trace summarised with [`bench/prof_summary.py`](bench/prof_summary.py);
the trace itself is not published (tens of MB). Total: 26 ms per token.

| ms/token | component |
|--:|---|
| 9.47 | Grace bank (Marlin over UVA, ~258 GB/s effective) |
| 5.78 | dense layers (BF16 `o_proj` about 2.5) |
| 3.34 | HBM bank |
| ~3.3 | ~1,900 small glue kernels per token |
| 1.53 | MoE align/sum, done twice (two banks) |
| 1.14 | attention |
| 0.94 | peer-tier kernels on the GB300 (pack, publish, wait, add) |

A plain copy over NVLink-C2C reads Grace at 350–390 GB/s for 18–36 MiB, with SM copy and DMA equal *(notes)*.
v3 staging brings Marlin's Grace reads close to that. It improved C1 decode by 1–2%. The knee at C8/C16 did not
improve: 78.5/98.2 vs 79.7/102.2. Staging also cost 2.4 GiB of KV (361K → 323K tokens, *notes*). Marlin
over UVA already approaches link bandwidth at larger batch sizes. Staging still matters because it makes every
decode-time expert read an HBM read, and TRT-LLM needs that (v4).

## TRT-LLM vs Marlin on MiMo-V2.6-Pro experts (HBM, microbenchmark)

Source: [`results/logs/trt-vs-marlin.log`](results/logs/trt-vs-marlin.log) ([`hook/tests/bench_trt_vs_marlin.py`](hook/tests/bench_trt_vs_marlin.py)).
The bank has 96 experts from layer 5, each token routes into the bank, and small T is timed inside CUDA graphs.

| T tokens | 1 | 2 | 4 | 8 | 16 | 32 | 128 | 512 | 2,048 | 8,192 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Marlin µs | 61.5 | 88.7 | 134.1 | 220.5 | 351.6 | 479.0 | 691.6 | 1,324.6 | 3,362.7 | 10,306.0 |
| TRT-LLM µs | 36.4 | 53.5 | 78.4 | 122.5 | 194.9 | 288.5 | 338.9 | 376.2 | 611.2 | 1,610.5 |
| speedup | 1.69× | 1.66× | 1.71× | 1.80× | 1.80× | 1.66× | 2.04× | 3.52× | 5.50× | 6.40× |
| cosine TRT vs Marlin | 0.99955 | 0.99971 | 0.99955 | 0.99973 | 0.99936 | 0.99965 | 0.99944 | 0.99952 | 0.99961 | 0.99936 |

Untuned TRT-LLM heuristics were within 1% of autotuned tactics, so the lane runs with FlashInfer autotune off.
Autotuning would read the Grace-resident weights.

vLLM's modular TRT-LLM wrapper ignores `expert_map` (it assumes contiguous expert-parallel ranges). So each
bank calls FlashInfer's routed kernel directly, with bank-local ids and `-1` for routes outside the bank.

## Quality

- GSM8K-200, v2: 197/200 = 98.5% ([`results/logs/gsm8k-3tier-v2.log`](results/logs/gsm8k-3tier-v2.log)).
- BFCL dev (simple_python 400 + multiple 200), v2, with Al-ENGR's grader: 557/600 = 92.83%. By suite:
  simple_python 93.5%, multiple 91.5%, 0 errors ([summary](results/logs/bfcl-3tier-v2-dev-summary.json),
  [per-case](results/logs/bfcl-3tier-v2-dev.jsonl)). His v23 scored 93.83% and his control 93.67%. We found
  27 discordant cases, 17 against us and 10 for us: p≈0.25, not significant *(notes)*. The gap is not
  attributed: vLLM 0.30 vs 0.29, and the W4A8 sidecar tier both differ.
- v3 components *(notes)*: staged vs UVA bank cosine 1.000000; the fused send is byte-exact against the torch-op
  send. Tests: [`hook/tests/test_stage_and_send.py`](hook/tests/test_stage_and_send.py),
  [`test_stage_overlap.py`](hook/tests/test_stage_overlap.py).
- v4: only the microbenchmark cosines above, plus the TRT-bank vs one-call check in
  [`hook/tests/test_trt_banks.py`](hook/tests/test_trt_banks.py).
- v5 and v6: the FA4 FP8 tests above; their output was not saved.
- v6: needle retrieval passes at 11.5K, 86K and 357K prompt tokens ([`bench/needle.py`](bench/needle.py), depth 50%,
  greedy, thinking off). This is from session notes; the output was not kept.
- v6 long-coding run: 45.6 tok/s end to end for 91K tokens, but the game it wrote does not run
  ([`../longgen/`](../longgen/README.md)).
- No end-to-end quality gate (GSM8K or BFCL) has been run on v4, v5 or v6. `logs/gsm8k-trt-v4.log` on the
  station is empty.

<a id="knee-v2"></a>
## Knee v2 raw data

`bench/knee.sh` (Al-ENGR's, verbatim) writes its JSON through a Python f-string containing `${WORKDIR:-$PWD}`.
Inside a quoted heredoc, Python reads that as a variable named `WORKDIR` and raises `NameError`. The v3 log
ends with that traceback. The v2 knee printout was read off the terminal and was not saved, so its row in
`results.jsonl` has `source: null`.

## Gotchas

- b12x leaks preparation scratch when the expert count varies across layers. Prepare the largest layer first.
- b12x allocates a 256 MiB L2-flush buffer per `PreparationSession`. Share one session across layers, or a
  full-card prepare runs out of memory at the end ([`results/logs/memprobe.log`](results/logs/memprobe.log)).
- The sidecar's first cross-check was spent on vLLM's all-zero memory-profiling dummy run. The sidecar now
  skips all-zero references.
- `pkill -f <pattern>` and `pgrep -f <pattern>` match the invoking shell when the pattern is on its command
  line. Kill by PID.
- Every boot re-reads 527 GiB from disk. Batch fixes into one boot and profile, rather than
  running many restart A/Bs.
- `launch-pro.sh` drops the page cache (`sudo`) to make room for about 320 GiB of pinned experts.
