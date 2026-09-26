#!/usr/bin/env python3
"""Build qwen3.8-flash-next/results.jsonl from the raw result files in this lane.

One JSON object per measured number. Sources are repo-relative paths. Run from anywhere:
    python3 qwen3.8-flash-next/tools/make_results_jsonl.py
Reads:  results/runs/*/decode/c*.json, results/runs/smoke-*/c*.json, results/runs/ttft-*/p*/c*.json,
        results/logs/prefill-*.jsonl, results/gsm8k/gsm8k-qwen-*.json, results/logs/qual-final.log,
        results/prof/prefill-profile-cg1024-gaps.log, results/cg/plp-compare.log, results/cg/nondet-*.log
Also emits catid's published headline numbers as external baseline rows (config "external:...",
qualified null, source = the page URL).
"""
import datetime
import glob
import json
import os

LANE = "qwen3.8-flash-next"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the lane dir
MODEL = "Qwen3.8-Flash-Next NVFP4 (nvidia/Qwen3.8-Flash-Next-NVFP4)"
VLLM = "vLLM vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423 (v0.30.1rc1.dev48+g7f1a5398e)"
SGLANG = "SGLang lmsysorg/sglang:nightly-dev-cu13-20260924-ffac53d7"

# Configs whose own server instance passed a GSM8K check (thinking off). See DETAILS.md.
QUALIFIED = {
    "vllm-trtllm-mtp3", "vllm-rust-mp-mtp3", "vllm-recipe-mtp3",
    "sglang-trtllm-mtp3", "sglang-rs-mtp3", "sglang-rs-fipre-mtp3", "sglang-mega-rsfp-mtp3",
    "ab-base-1", "ab-rustmp-1", "pp0", "pp0.5",
    "smoke-flashinfer_trtllm", "smoke-flashinfer_cutedsl", "smoke-flashinfer_cutlass", "smoke-cutlass",
    "vllm-py-cg8192-mtp3", "cg8192-py-1",
}
# Prefill-diagnostic servers from 2026-09-25 (DETAILS.md, "Prefill step cost and CUDA graph sizes"). The raw files
# keep the labels the bench wrote; these map them to config ids. None of these server instances ran a GSM8K gate.
PREFILL_LABELS = {
    "sweep-rust-mp-pc": ("diag-rust-mp-cg1024", "Rust + mp, CUDA graphs to 1024 (then-default launcher); /reset_prefix_cache "
                         "unavailable (404) on this frontend, prompts are fresh random token ids"),
    "boundary": ("diag-rust-mp-cg1024", "Rust + mp, CUDA graphs to 1024; boundary probe, no cache flush, fresh random "
                 "token ids per point"),
    "py-uni": ("diag-py-uni-cg1024", "Python + uni, CUDA graphs to 1024, torch profiler configured (idle during this "
               "run); /reset_prefix_cache unavailable (404), fresh random token ids"),
    "cg8192-py-uni": ("diag-py-uni-cg8192", "Python + uni, CUDA graphs to 8192, torch profiler configured (idle), "
                      "VLLM_SERVER_DEV_MODE=1 so the cache flush worked"),
}
NOTE_UNGATED = {
    "ab-base-2": "same launch config as ab-base-1 (GSM8K-50 48/50); this server instance not re-gated",
    "ab-rustmp-2": "same launch config as ab-rustmp-1 (GSM8K-50 49/50); this server instance not re-gated",
}

rows = []


def rel(path):
    return f"{LANE}/{os.path.relpath(path, ROOT)}"


def engine_for(config):
    return SGLANG if config.startswith("sglang") else VLLM


def add(config, date, benchmark, metric, conc, ctx, value, unit, source, notes=None, model=MODEL, engine=None,
        qualified=None):
    if value is None:
        return
    if qualified is None:
        qualified = config in QUALIFIED
    if not qualified and config in NOTE_UNGATED:
        notes = "; ".join(x for x in (notes, NOTE_UNGATED[config]) if x)
    rows.append({"lane": LANE, "model": model, "config": config, "date": date,
                 "engine": engine or engine_for(config), "benchmark": benchmark, "metric": metric,
                 "concurrency": conc, "context_tokens": ctx, "value": round(value, 4) if isinstance(value, float) else value,
                 "unit": unit, "qualified": qualified, "source": source, "notes": notes})


def decode_rows(path, config, benchmark, extra_note):
    j = json.load(open(path))
    date = j["metadata"]["timestamp"][:10]  # client-local time (EDT)
    for r in j.get("results") or []:
        c, ctx = r["concurrency"], r.get("context_tokens")
        note = (f"{extra_note}; completed {r['num_completed']}/{r['request_count']}, errors {r['num_errors']}"
                + (f", queue_fraction {r['queue_fraction']}" if r.get("capacity_limited") else ""))
        src = rel(path)
        add(config, date, benchmark, "aggregate_tok_s", c, ctx, r["aggregate_tps"], "tok/s", src, note)
        add(config, date, benchmark, "per_user_tok_s_p50", c, ctx, r.get("output_tps_per_user_p50"), "tok/s", src, note)
        add(config, date, benchmark, "ttft_p50_ms", c, ctx, r["ttft_p50"] * 1e3, "ms", src, note)
        add(config, date, benchmark, "ttft_p90_ms", c, ctx, r["ttft_p90"] * 1e3, "ms", src, note)
        add(config, date, benchmark, "itl_p50_ms", c, ctx, r["inter_token_latency_p50"] * 1e3, "ms", src, note)
        acc = r.get("server_spec_accept_length")
        if acc:  # 0.0 when MTP is off or the server exposes no spec metrics
            add(config, date, benchmark, "accept_len", c, ctx, acc, "x", src,
                note + "; server-side mean accepted length per MTP step")


# catid decode recipe runs written by the shared harness (8,192 in / 1,024 out, T=0, C warm-ups + 5xC)
for d in sorted(glob.glob(os.path.join(ROOT, "results/runs/qwen-*/decode"))):
    config = os.path.basename(os.path.dirname(d))[len("qwen-"):]
    for f in sorted(glob.glob(os.path.join(d, "c*.json")), key=lambda p: int(os.path.basename(p)[1:-5])):
        decode_rows(f, config, "catid-decode", "catid recipe: 8192 in / 1024 out, T=0, C warm-ups + 5xC")

# MoE kernel smoke test: minimal prompt, 1,024 forced output, C warm-ups + 2xC requests, MTP off
for d in sorted(glob.glob(os.path.join(ROOT, "results/runs/smoke-*"))):
    config = os.path.basename(d)
    for f in sorted(glob.glob(os.path.join(d, "c*.json")), key=lambda p: int(os.path.basename(p)[1:-5])):
        decode_rows(f, config, "other", "smoke decode: minimal prompt, 1024 out, T=0, C warm-ups + 2xC, MTP off")

# TTFT smoke (ttft-quick.sh): catid request shape but 2xC requests, two passes on one server
for d in sorted(glob.glob(os.path.join(ROOT, "results/runs/ttft-*/p*"))):
    config = os.path.basename(os.path.dirname(d))
    p = os.path.basename(d)
    for f in sorted(glob.glob(os.path.join(d, "c*.json")), key=lambda p_: int(os.path.basename(p_)[1:-5])):
        decode_rows(f, config, "ttft-bench", f"ttft-quick {p}: 8192 in / 1024 out, T=0, C warm-ups + 2xC")

# Cold prefill, C1, random tokens, cache flushed, 1 warm-up + 4 measured requests per point
for f in sorted(glob.glob(os.path.join(ROOT, "results/logs/prefill-*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        ts = datetime.datetime.fromisoformat(r["ts"]).astimezone(datetime.timezone(datetime.timedelta(hours=-4)))
        note = (f"cold prefill, {r['requests']} requests + {r['warmup']} warm-up, random tokens, "
                f"token_count_match {r['token_count_match']}, errors {r['errors']}")
        config, extra = PREFILL_LABELS.get(r["label"], (r["label"], None))
        if extra:
            note = f"{note}; {extra}"
        add(config, ts.date().isoformat(), "prefill", "prefill_tok_s", r["concurrency"], r["isl"],
            r["agg_prompt_tok_s"], "tok/s", rel(f), note)
        add(config, ts.date().isoformat(), "prefill", "ttft_p50_s", r["concurrency"], r["isl"],
            r["ttft_p50_s"], "s", rel(f), note)

# GSM8K (last N test questions, greedy, thinking off)
decode_dates = {}
for row in rows:
    decode_dates.setdefault(row["config"], row["date"])
for f in sorted(glob.glob(os.path.join(ROOT, "results/gsm8k/gsm8k-qwen-*.json"))):
    j = json.load(open(f))
    config = j["label"][len("qwen-"):]
    date = decode_dates.get(config) or datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat()
    note = f"GSM8K-{j['n']} (last {j['n']} test questions), thinking off, T=0: {j['correct']}/{j['n']}"
    if config.startswith("pp0.5"):
        note += "; presence_penalty 0.5 injected per request by pp_proxy.py"
    add(config, date, "gsm8k", "accuracy", None, None, j["correct"] / j["n"], "fraction", rel(f), note,
        qualified=True)

# 128K thinking-loop check (pp-loop-test*.sh): llm_decode_bench on the bench client, 128K context, C1 and C8,
# 30 s, max 8,192 output tokens; a tally script on the client printed one status per cell ("ok" = no loop found).
import re  # noqa: E402
cells = {}
LOOP_DATES = {"pp0": "2026-09-25", "pp0.5": "2026-09-25", "vllm-py-cg8192-mtp3": "2026-09-25"}
for f in sorted(glob.glob(os.path.join(ROOT, "results/logs/pp-loop-test*.log"))
                + [os.path.join(ROOT, "results/logs/qual-final.log")]):
    for line in open(f):
        m = re.match(r"^(pp[0-9.]+|vllm-py-cg8192-mtp3) run(\d+) (C\d+)/128K (\S+)", line)
        if m:
            cells.setdefault(m.group(1), []).append((m.group(4), rel(f)))
for label, items in sorted(cells.items()):
    looped = sum(1 for status, _ in items if status != "ok")
    srcs = sorted({s for _, s in items})
    partial = "; partial: the check was still running when the log was copied" if len(items) < 16 else ""
    add(label, LOOP_DATES[label], "other", "loop_fraction", None, 131072, looped / len(items), "fraction", srcs[-1],
        f"{looped}/{len(items)} cells flagged ({len(items) // 2} runs x C1,C8; 128K context, max 8192 output){partial}; "
        f"all sources: {', '.join(srcs)}" + ("; presence_penalty 0.5 via pp_proxy.py" if label == "pp0.5" else ""))
for config, frac, note in (("sglang-rs-fipre-mtp3", 3 / 8, "3 of 8 runs looped in the thinking channel at 128K"),
                           ("sglang-trtllm-mtp3", 0.0, "0 loops; number of runs not recorded in the notes")):
    add(config, None, "other", "loop_fraction", None, 131072, frac, "fraction", None,
        f"from session notes; raw file not kept: {note}")

# Needle retrieval (needle_test.py via qual-final.sh): passphrase at 10/50/90% depth, thinking off, T=0
QUAL = os.path.join(ROOT, "results/logs/qual-final.log")
for line in open(QUAL):
    m = re.match(r"^(vllm-py-cg8192-mtp3)-(\d+) depth=(\d+)% prompt_tokens=(\d+) time=(\d+)s (PASS|FAIL)", line)
    if m:
        add(m.group(1), "2026-09-25", "other", "needle_pass", 1, int(m.group(4)), 1.0 if m.group(6) == "PASS" else 0.0,
            "fraction", rel(QUAL), f"needle_test.py target {m.group(2)} tokens, depth {m.group(3)}%, "
            f"{m.group(5)} s, thinking off, T=0; 1 = passphrase returned")

# Prefill profile (prof_prefill.py + prof_gaps.py) on diag-py-uni-cg1024: GPU busy share of each prefill step
PROF = os.path.join(ROOT, "results/prof/prefill-profile-cg1024-gaps.log")
for line in open(PROF):
    m = re.match(r"^execute_context_1\((\d+)\)\S*\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+(\d+)%", line)
    if m:
        note = (f"one {m.group(1)}-token prefill step under the torch profiler (CPU overhead inflated): "
                f"GPU span {m.group(2)} ms, {m.group(3)} kernels, busy {m.group(4)} ms")
        add("diag-py-uni-cg1024", "2026-09-25", "other", "gpu_busy_pct", 1, int(m.group(1)), float(m.group(5)), "%",
            rel(PROF), note)

# Correctness of CUDA graphs to 8192 (plp_offline.py, nondet.py). Teacher-forced prompt logprobs on real text.
PLP = os.path.join(ROOT, "results/cg/plp-compare.log")
section = None
for line in open(PLP):
    if line.startswith("== "):
        section = line[3:].strip()
        continue
    m = re.search(r"^\s+(.*?):?\s+NLL ([\d.]+) vs ([\d.]+)\s+mean\|dlogprob\| ([\d.]+)\s+p99 ([\d.]+)", line)
    if m and section:
        what = m.group(1).strip().rstrip(":")
        add("cg8192-vs-cg1024", "2026-09-25", "other", "mean_abs_dlogprob", 1, None, float(m.group(4)), "nats",
            rel(PLP), f"{section} | {what}: NLL {m.group(2)} vs {m.group(3)}, p99 {m.group(5)}", qualified=False)
for f in sorted(glob.glob(os.path.join(ROOT, "results/cg/nondet-*.log"))):
    cfg = os.path.basename(f)[len("nondet-"):-len(".log")]
    for line in open(f):
        m = re.match(r"^\s*(\d+) tokens: .*top-5 overlap (\d)/5, max \|diff\| on shared ([\d.]+)", line)
        if m:
            add(cfg, "2026-09-25", "other", "rerun_max_abs_dlogprob", 1, int(m.group(1)),
                float(m.group(3)), "nats", rel(f), f"3 cold runs of one prompt, first output token's top-5 logprobs; "
                f"top-5 overlap {m.group(2)}/5; indexer_budget is 2048")

# KV capacity and CUDA graph memory of the 2026-09-25 evening servers. Only the chosen config's server log was kept.
SLOG = os.path.join(ROOT, "results/logs/server-vllm-py-cg8192-mtp3.log")
captures = 0
for line in open(SLOG):
    m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
    if m:
        add("vllm-py-cg8192-mtp3", "2026-09-25", "other", "kv_cache_tokens", None, None, int(m.group(1).replace(",", "")),
            "tokens", rel(SLOG), "BF16 KV, --gpu-memory-utilization 0.90, vision and video enabled")
    m = re.search(r"Graph capturing finished in \d+ secs, took ([\d.]+) GiB", line)
    if m:
        captures += 1
        add("vllm-py-cg8192-mtp3", "2026-09-25", "other", "cuda_graph_memory_gib", None, None, float(m.group(1)), "GiB",
            rel(SLOG), "first capture pass (memory profiling, before KV allocation)" if captures == 1
            else "second capture pass (after KV allocation)")
for config, kv, graph in (("diag-rust-mp-cg1024", 4986403, 1.55), ("diag-py-uni-cg8192", 4847538, 5.66)):
    add(config, "2026-09-25", "other", "kv_cache_tokens", None, None, kv, "tokens", None,
        "from session notes (server log not kept); BF16 KV, --gpu-memory-utilization 0.90, vision off")
    add(config, "2026-09-25", "other", "cuda_graph_memory_gib", None, None, graph, "GiB", None,
        "from session notes (server log not kept); second capture pass, after KV allocation")

# Session-note correctness numbers whose raw output was printed but not saved (DETAILS.md, Correctness)
for config, vals in (("diag-py-uni-cg8192", (0.0, 0.0, 1.6208, 1.2500, 1.8122, 1.0626)),
                     ("diag-py-uni-cg1024-dev", (0.0, 0.0, 0.8737, 1.1250, 0.8124, 1.0000))):
    for n, v in zip((1900, 2040, 2100, 2500, 3000, 3150), vals):
        add(config, "2026-09-25", "other", "rerun_max_abs_dlogprob", 1, n, v, "nats", None,
            "from session notes (nondet.py output not kept): 3 cold runs, first output token's top-5 logprobs, max "
            "|diff| on shared entries; indexer_budget is 2048")
for config, vals in (("diag-py-uni-cg1024-dev", (0.0330, 0.0618, 0.0739, 0.0696, 0.0779, 0.0764, 0.0856, 0.1070)),
                     ("cg8192-py-1", (0.0370, 0.0551, 0.0774, 0.0718, 0.0815, 0.0738, 0.0849, 0.1056))):
    for (n, off), v in zip(((3000, 0), (3000, 30000), (6000, 0), (6000, 30000), (8192, 0), (8192, 30000),
                            (16384, 0), (16384, 30000)), vals):
        add(config, "2026-09-25", "other", "mean_abs_dlogprob", 1, n, v, "nats", None,
            f"from session notes (plp.py compare output not kept): two cold passes of the {n}-token prompt at text "
            f"offset {off} on the same server; run-to-run noise floor")

# catid's published Qwen3.8-Flash-Next page (1x DGX Station, TP1, SGLang). Not our measurement.
CATID_URL = "https://github.com/catid/dgx_station_benchmarks/tree/main/qwen3.8-flash-next"
CATID_MODEL = "Qwen3.8-Flash-Next NVFP4 (local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89)"
CATID_ENGINE = "SGLang qwen38-4p89-sglang:runtime-v1 (catid)"
CATID_NOTE = "external baseline, 1x DGX Station TP1; not measured by us; different checkpoint from ours"
for config, vals in (("external:catid-sglang-tp1-ar", (202.1, 1883.9, 4090.4, 38653)),
                     ("external:catid-sglang-tp1-mtp3-replayssm", (354.6, 1733.2, 2927.8, 37884))):
    for c, v in zip((1, 16, 64), vals[:3]):
        rows.append({"lane": LANE, "model": CATID_MODEL, "config": config, "date": None, "engine": CATID_ENGINE,
                     "benchmark": "catid-decode", "metric": "aggregate_tok_s", "concurrency": c,
                     "context_tokens": 8192, "value": v, "unit": "tok/s", "qualified": None,
                     "source": CATID_URL, "notes": CATID_NOTE})
    rows.append({"lane": LANE, "model": CATID_MODEL, "config": config, "date": None, "engine": CATID_ENGINE,
                 "benchmark": "prefill", "metric": "prefill_tok_s", "concurrency": 1, "context_tokens": 65536,
                 "value": vals[3], "unit": "tok/s", "qualified": None, "source": CATID_URL,
                 "notes": CATID_NOTE + "; 64K cold prefill"})

out = os.path.join(ROOT, "results.jsonl")
with open(out, "w") as fh:
    for row in rows:
        fh.write(json.dumps(row) + "\n")
print(f"wrote {len(rows)} rows to {out}")
