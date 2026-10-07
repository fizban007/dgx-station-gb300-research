# DGX Station GB300 serving research

These are measured results from serving large MoE models on one NVIDIA DGX Station GB300, mostly from
2026-09-23 to 25, with DeepSeek-V4.1-Flash M3 updates through 2026-10-03. The station also holds an RTX PRO 6000 Blackwell Max-Q, and most of this work puts that second
GPU to use. For models whose experts don't fit in the GB300's HBM, the RTX PRO 6000 becomes an **expert
sidecar**: it holds experts in its own 96 GB and computes them there.

Every number below links to a raw result file. Numbers from a configuration that never passed a quality gate
are marked as such.

## Highlights

### The expert sidecar

| model | without the sidecar | with the sidecar | quality |
|---|---|---|---|
| **DeepSeek-V4.1-Flash** ([M3](deepseek-v4.1-flash/m3/)): MegaMoE hot experts on the GB300, 3,960 cold experts on the RTX PRO 6000 | Al-ENGR's v20 recipe, reproduced here: 583 / 821 tok/s at C8 / C16, 15.8K tok/s prefill | **1,046 / 1,629 tok/s** at C8 / C16 (1.8× / 2.0×), 258–266 tok/s per user at C1, 44.5K prefill (60K on real text); 2026-09-28 | GSM8K-200 98.0%; needle at 114K |
| **MiMo-V2.6-Pro** (527 GiB, [three tiers](mimo-v2.6-pro/)): 152.8 GiB in HBM, 90.2 GiB on the RTX PRO 6000, the rest in Grace | Al-ENGR's v23 recipe (HBM + Grace): 1,376 tok/s prefill and 37.4 tok/s decode at 11.5K; 63 tok/s at C16 | **2,812** prefill (2×), **39.4** decode; **102** tok/s at C16 | GSM8K-200 98.5%; BFCL dev 92.83% vs his 93.83% (not significant) |
| DeepSeek-V4.1-Flash, Al-ENGR's v20 plus our sidecar ([details](deepseek-v4.1-flash/)) | 583 / 821 at C8 / C16 | 758 / 1,154 | not gated |

The GB300 spends under 1% of each forward pass waiting for the sidecar. The design, protocol and failure
behaviour are in [docs/sidecar-peer-tier.md](docs/sidecar-peer-tier.md).

Since 2026-10-03 the DeepSeek M3 lane also shares its GB300 with a video model (MiniMax-H3) and keeps 3.42M KV
tokens. It uses 265 hot experts, NVFP4 Engram tables and NVFP4 KV
([how](deepseek-v4.1-flash/m3/DETAILS.md#2026-10-0203-sharing-the-gb300-with-minimax-h3)). The table above shows its
peak results from 2026-09-28.

The later MiMo-V2.6-Pro versions are **unqualified**: no GSM8K or BFCL gate has been run on them yet. They add
FlashInfer TRT-LLM expert banks that read from HBM, an FP8 KV cache, and DFlash k=3. They reach 4.6–5.0K tok/s
prefill at 11.5K, up to 51 tok/s single-stream decode, and 2.4M tokens of KV cache (from 413K).

### Single-GPU serving

| model | best configuration | decode (aggregate tok/s) | prefill | quality |
|---|---|---|---|---|
| **MiMo-V2.6-Flash** ([lane](mimo-v2.6-flash/)) | vLLM nightly 29468dde, DFlash k=7, vllm#58207 KV grouping | 383 at C1, 1,386 at C8, 3,741 at C32, 5,686 at C64 | 43.9K at 8K, 26.7K at 128K | GSM8K-200 97.5% |
| **Qwen3.8-Flash-Next NVFP4** ([lane](qwen3.8-flash-next/)) | vLLM nightly 7f1a5398, MTP3, FlashInfer TRT-LLM MoE, CUDA graphs to 8,192 tokens | 344 at C1, 2,592 at C16, 5,247 at C64 | 43.6K at 8K, 44.4K at 128K | GSM8K-200 97.5% |

- **Qwen3.8:** vLLM's prefill steps over 1,024 tokens ran without CUDA graphs. Each took about 125 ms, less than
  half of it GPU work; the rest was launching ~2,200 kernels from Python. Capturing graphs up to 8,192 tokens cut 8K TTFT from 258 to 188 ms and raised decode at C8–C64. vLLM now
  beats SGLang with the same MoE kernel at every concurrency (C64: 5,247 vs 3,086–3,555).
- **Long coding session** ([longgen](longgen/)): MiMo-V2.6-Flash ran 544 tok/s end to end over 62K generated
  tokens, peaking at 854 tok/s. MiMo-V2.6-Pro on the sidecar lane ran 45.6 tok/s over 91K tokens. Neither game
  runs as written.

### b12x on the GB300

For DeepSeek-V4.1-Flash's MXFP4-expert / FP8-dense mix, upstream kernels beat our b12x SM103 stack in every
column ([findings](deepseek-v4.1-flash/findings.md)). b12x's SM120 fused MoE is what makes the RTX PRO 6000
sidecar fast: for 512 prefill tokens it takes 1.1 ms, where a Triton GEMV takes 10.6 ms. This conclusion covers
only that model and quantization mix.

## Repository map

| path | what |
|---|---|
| [`deepseek-v4.1-flash/`](deepseek-v4.1-flash/) | DeepSeek-V4.1-Flash: v20 reproduced, overnight b12x experiments, and the b12x-on-GB300 findings |
| [`deepseek-v4.1-flash/m3/`](deepseek-v4.1-flash/m3/) | MegaMoE + RTX PRO 6000 sidecar lane |
| [`mimo-v2.6-pro/`](mimo-v2.6-pro/) | MiMo-V2.6-Pro three-tier lane (HBM, sidecar, Grace) |
| [`mimo-v2.6-flash/`](mimo-v2.6-flash/) | MiMo-V2.6-Flash with DFlash |
| [`qwen3.8-flash-next/`](qwen3.8-flash-next/) | Qwen3.8-Flash-Next NVFP4: engines, kernels, MTP, frontends, CUDA graph sizes |
| [`longgen/`](longgen/) | long-coding generation test (PS4 Tetris prompt) |
| [`docs/`](docs/) | [sidecar design](docs/sidecar-peer-tier.md), [the station and its memory mode](docs/station.md) |
| [`bench/`](bench/) | shared benchmark harness |
| [`data/`](data/) | every number in one table: [`results.jsonl`](data/results.jsonl), [`results.csv`](data/results.csv), [schema](data/README.md) |

Each lane has a short `README.md` for people, a `DETAILS.md` with every run and column, a `results.jsonl` for
machines, and the raw files under `results/`, `runs/` or `logs/`.

## For agents and scripts

Start at [`llms.txt`](llms.txt). All quoted numbers are rows in [`data/results.jsonl`](data/results.jsonl)
(schema in [`data/README.md`](data/README.md)), and each row cites its raw source file.
`python3 data/build.py` rebuilds that table from the lanes and checks that every cited file exists.

## Reading the numbers

- **Decode** mostly uses catid's recipe from
  [dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks), run with
  [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench). Each request has 8,192
  input tokens and 1,024 forced output tokens at temperature 0. A run sends C warm-up requests, then measures
  5×C requests. "Aggregate" counts all streams; "per user" is the median single-stream decode rate.
- **Prefill** is one cold request of random token ids with the prefix cache flushed, unless stated otherwise.
  Real text can differ; for DeepSeek it was faster.
- **Quality gates** are GSM8K-200 (greedy, thinking off), BFCL for MiMo-V2.6-Pro, and in-server cross-checks
  for the sidecar paths.
- **Comparisons with others.** Al-ENGR's recipes
  ([J-M-Recipes/recipes](https://github.com/J-M-Recipes/recipes)) are single-station and were our starting
  points. catid's DeepSeek-V4.1 numbers come from **two** stations, so we don't treat them as a single-GPU
  bar. His Qwen3.8 numbers are single-station, with a different checkpoint.
- Most configurations ran once. Differences of a few percent are within run-to-run noise.

## Caveats

- Scripts are verbatim snapshots from the station. They contain its absolute paths
  (`/home/jasonc/research/<lane>` corresponds to the lane directory here), GPU UUIDs, and venv locations. Each
  lane README gives the mapping.
- Several lanes used unmerged vLLM PRs or local overlays. Each file mounted over a vLLM image is included,
  with a diff against stock.
- The MiMo-V2.6-Pro lane was mid-iteration when snapshotted. Its launcher defaults are newer than its last
  measurement; its README says which.
- Where a number comes from notes taken during the session and its raw output was not kept, the lane says
  so, and its `results.jsonl` row has `source: null`.

## License

Original material: Apache License 2.0 ([LICENSE](LICENSE)). Some files come from vLLM (Apache-2.0),
J&M Recipes (MIT), FlashAttention (BSD-3-Clause) and SGLang (Apache-2.0); see [NOTICE](NOTICE).
