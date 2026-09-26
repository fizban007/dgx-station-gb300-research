# Machine-readable results

[`results.jsonl`](results.jsonl) and [`results.csv`](results.csv) hold every number quoted in this
repository's READMEs, one measurement per row, with the raw file it came from. They are merged from each lane's
own `results.jsonl` by [`build.py`](build.py). `build.py` also checks that every row has the schema keys and
that every cited source file exists. Rebuild after changing a lane:

```bash
python3 data/build.py
```

Each lane's `tools/make_results_jsonl.py` regenerates that lane's rows from its raw files.

## Schema

| key | type | meaning |
|---|---|---|
| `lane` | string | directory the row belongs to, e.g. `mimo-v2.6-pro`, `deepseek-v4.1-flash/m3` |
| `model` | string | model / checkpoint name |
| `config` | string | configuration id used in that lane's README and DETAILS; `external:<who>-<what>` for published third-party numbers |
| `date` | `YYYY-MM-DD` | measurement date (EDT) |
| `engine` | string | serving engine and image tag / commit, plus local changes |
| `benchmark` | enum | `catid-decode`, `prefill`, `knee`, `gsm8k`, `bfcl`, `ttft-bench`, `prefix-hit-ttft`, `longgen`, `kernel-microbench`, `other` |
| `metric` | string | snake_case metric name, e.g. `aggregate_tok_s`, `per_user_tok_s_p50`, `ttft_p50_ms`, `accept_len`, `prefill_tok_s`, `accuracy` |
| `concurrency` | int or null | concurrent streams (C) |
| `context_tokens` | int or null | prompt length in tokens |
| `value` | number | the measurement |
| `unit` | string | `tok/s`, `ms`, `s`, `us`, `fraction`, `tokens`, `x`, … |
| `qualified` | bool or null | true if this configuration passed a quality gate (GSM8K / BFCL / oracle cross-check); null for external rows and kernel microbenchmarks where noted |
| `source` | string or null | repo-relative path of the raw file, an https URL for external numbers, or null when the value comes from session notes and the raw output was not kept |
| `notes` | string | anything needed to interpret the value |

## Benchmarks

- **catid-decode**: 8,192 exact input tokens, 1,024 forced output tokens, temperature 0. The run starts with C
  warm-up requests, then measures 5×C requests. It uses
  [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) `llm_decode_bench.py`, the
  recipe from [catid/dgx_station_benchmarks](https://github.com/catid/dgx_station_benchmarks). `aggregate_tok_s`
  counts all completion tokens over wall time, including TTFT. `per_user_tok_s_p50` is the median
  per-request decode rate.
- **prefill**: cold prompt of random token ids with the prefix cache flushed, C1. `prefill_tok_s` = prompt tokens / TTFT.
- **knee**: Al-ENGR's prose knee, 192 forced tokens of prose per stream at temperature 0, two repetitions.
- **ttft-bench**: Al-ENGR's `ttft_bench.py`: one request per prompt size, reporting prefill tok/s and decode tok/s
  after the first token.
- **gsm8k**: the last 200 GSM8K test questions, greedy, thinking off ([`../bench/`](../bench/)).

See [`../bench/README.md`](../bench/README.md) for the harness.
