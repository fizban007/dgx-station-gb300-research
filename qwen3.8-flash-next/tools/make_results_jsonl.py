#!/usr/bin/env python3
"""Build qwen3.8-flash-next/results.jsonl from the raw result files in this lane.

One JSON object per measured number. Sources are repo-relative paths. Run from anywhere:
    python3 qwen3.8-flash-next/tools/make_results_jsonl.py
Reads:  results/runs/*/decode/c*.json, results/runs/smoke-*/c*.json, results/runs/ttft-*/p*/c*.json,
        results/logs/prefill-*.jsonl, results/gsm8k/gsm8k-qwen-*.json
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
        add(r["label"], ts.date().isoformat(), "prefill", "prefill_tok_s", r["concurrency"], r["isl"],
            r["agg_prompt_tok_s"], "tok/s", rel(f), note)
        add(r["label"], ts.date().isoformat(), "prefill", "ttft_p50_s", r["concurrency"], r["isl"],
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
for f in sorted(glob.glob(os.path.join(ROOT, "results/logs/pp-loop-test*.log"))):
    for line in open(f):
        m = re.match(r"^(pp[0-9.]+) run(\d+) (C\d+)/128K (\S+)", line)
        if m:
            cells.setdefault(m.group(1), []).append((m.group(4), rel(f)))
for label, items in sorted(cells.items()):
    looped = sum(1 for status, _ in items if status != "ok")
    srcs = sorted({s for _, s in items})
    add(label, "2026-09-25", "other", "loop_fraction", None, 131072, looped / len(items), "fraction", srcs[-1],
        f"{looped}/{len(items)} cells flagged (8 runs x C1,C8; 128K context, max 8192 output); "
        f"all sources: {', '.join(srcs)}" + ("; presence_penalty 0.5 via pp_proxy.py" if label == "pp0.5" else ""))
for config, frac, note in (("sglang-rs-fipre-mtp3", 3 / 8, "3 of 8 runs looped in the thinking channel at 128K"),
                           ("sglang-trtllm-mtp3", 0.0, "0 loops; number of runs not recorded in the notes")):
    add(config, None, "other", "loop_fraction", None, 131072, frac, "fraction", None,
        f"from session notes; raw file not kept: {note}")

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
