#!/usr/bin/env python3
"""Build mimo-v2.6-flash/results.jsonl from the raw files in this lane (plus a few session-note values, marked
with source=null). Run from anywhere: python3 mimo-v2.6-flash/tools/make_results_jsonl.py"""
import glob, json, os, re

LANE = "mimo-v2.6-flash"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../mimo-v2.6-flash
ROOT = os.path.dirname(HERE)
MODEL = "MiMo-V2.6-Flash-RL"
DATE = "2026-09-25"
E7F1A = "vLLM vllm/vllm-openai:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423 (0.30.1rc1.dev48+g7f1a5398e)"
E29468 = "vLLM vllm/vllm-openai:nightly-29468dde8b515031dc6d4d9d06bf0a2fa0442098 (0.30.1rc1.dev143+g29468dde8)"

# config id -> (engine, qualified, note)
CONFIGS = {
    "nospec-7f1a": (E7F1A, True, "no speculative decoding; stock KV grouping"),
    "dflash7-7f1a": (E7F1A, True, "DFlash k=7; stock KV grouping"),
    "dflash7-7f1a-pr58207": (E7F1A, True, "DFlash k=7; vllm#58207 KV-group backport"),
    "dflash7-7f1a-pr58207-seeded": (E7F1A, True, "same server config as dflash7-7f1a-pr58207; seeded decode prompts"),
    "dflash7-29468-pr58207": (E29468, True, "launcher default; seeded decode prompts"),
    "dflash7-29468-stock": (E29468, False, "stock KV grouping on 29468dde; only prefix-hit measured, no GSM8K"),
    "dflash7-longgen": ("vLLM, DFlash k=7 via launch-mimo.sh; image and KV grouping not recorded in the run files "
                        "(most likely nightly 29468dde)", False, "Tetris long-generation run with per-position counters"),
}
RUNS = {  # decode run dir -> config id, seeded?
    "mimo-base-nospec": ("nospec-7f1a", False),
    "mimo-dflash7-7f1a": ("dflash7-7f1a", False),
    "mimo-dflash7-7f1a-pr58207": ("dflash7-7f1a-pr58207", False),
    "mimo-seeded-dflash7-7f1a-pr58207": ("dflash7-7f1a-pr58207-seeded", True),
    "mimo-dflash7-29468-pr58207": ("dflash7-29468-pr58207", True),
}
PREFILL = {"base-nospec": "nospec-7f1a", "dflash7-7f1a": "dflash7-7f1a",
           "dflash7-7f1a-pr58207": "dflash7-7f1a-pr58207", "dflash7-29468-pr58207": "dflash7-29468-pr58207"}
GSM = {"mimo-base-nospec": "nospec-7f1a", "mimo-dflash7-7f1a": "dflash7-7f1a",
       "mimo-dflash7-7f1a-pr58207": "dflash7-7f1a-pr58207", "mimo-dflash7-29468-pr58207": "dflash7-29468-pr58207"}
BENCHLOG = {"base-nospec": "nospec-7f1a", "dflash7-7f1a": "dflash7-7f1a",
            "dflash7-7f1a-pr58207": "dflash7-7f1a-pr58207", "dflash7-29468-pr58207": "dflash7-29468-pr58207"}
PREFIX = {"29468-pr58207": "dflash7-29468-pr58207", "29468-main": "dflash7-29468-stock"}

rows = []


def row(config, benchmark, metric, value, unit, source, concurrency=None, context=None, notes=None):
    eng, qual, _ = CONFIGS[config]
    rows.append({"lane": LANE, "model": MODEL, "config": config, "date": DATE, "engine": eng,
                 "benchmark": benchmark, "metric": metric, "concurrency": concurrency, "context_tokens": context,
                 "value": value, "unit": unit, "qualified": qual,
                 "source": source and os.path.relpath(source, ROOT), "notes": notes})


# catid decode: 8,192 exact input, 1,024 forced output, T=0, C warm-ups then 5xC requests
for run, (cfg, seeded) in RUNS.items():
    for f in sorted(glob.glob(f"{HERE}/runs/{run}/decode/c*.json"), key=lambda p: int(os.path.basename(p)[1:-5])):
        r = json.load(open(f))["results"][0]
        c, ctx = r["concurrency"], r["context_tokens"]
        tag = "seeded prompts" if seeded else "unseeded prompts"
        for metric, val, unit, note in [
            ("aggregate_tok_s", r["aggregate_tps"], "tok/s", "aggregate"),
            ("per_user_tok_s_p50", r["output_tps_per_user_p50"], "tok/s", "per user, 1/ITL"),
            ("per_user_e2e_tok_s_p50", r["e2e_output_tps_per_user_p50"], "tok/s", "per user, incl. TTFT"),
            ("ttft_p50_ms", r["ttft_p50"] * 1000, "ms", None),
            ("ttft_p99_ms", r["ttft_p99"] * 1000, "ms", None),
            ("itl_p50_ms", r["inter_token_latency_p50"] * 1000, "ms", None),
            ("request_latency_p50_s", r["request_latency_p50"], "s", None),
            ("accept_len", r.get("server_spec_accept_length"), "tokens", "tokens emitted per engine step"),
            ("engine_steps_per_s", r.get("server_steps_per_s"), "steps/s", "tok/s divided by accept length"),
            ("completed_requests", r["completed_request_count"], "count", f"of {r['request_count']}"),
            ("errors", r["num_errors"], "count", None),
        ]:
            if val is None or (metric in ("accept_len", "engine_steps_per_s") and not val):
                continue
            row(cfg, "catid-decode", metric, round(val, 6), unit, f, c, ctx, "; ".join(x for x in (note, tag) if x))
        if c == 1:  # KV budget as the bench reported it at startup
            log = open(f[:-5] + ".log").read()
            m = re.search(r"KV cache budget \(vLLM metrics\): ([\d,]+) tokens \(([^)]*)\)", log, re.S)
            if m:
                how = " ".join(m.group(2).split())
                metric = "kv_capacity_tokens" if "group-aware" in how else "kv_budget_blocks_x16_tokens"
                row(cfg, "other", metric, int(m.group(1).replace(",", "")), "tokens", f[:-5] + ".log",
                    notes=f"llm-inference-bench startup line: {how}")

# cold prefill, C1, 4 requests + 1 warm-up, fresh random seed
for label, cfg in PREFILL.items():
    f = f"{HERE}/logs/prefill-{label}.jsonl"
    for line in open(f):
        r = json.loads(line)
        isl = r["isl"]
        for metric, val, unit in [("prefill_tok_s", r["agg_prompt_tok_s"], "tok/s"),
                                  ("ttft_mean_s", r["ttft_mean_s"], "s"), ("ttft_p50_s", r["ttft_p50_s"], "s"),
                                  ("ttft_p99_s", r["ttft_p99_s"], "s"),
                                  ("prefill_tok_s_per_request_mean", r["req_tok_s_mean"], "tok/s"),
                                  ("gpu_power_mean_w", r["gpu_power_mean_w"], "W"),
                                  ("errors", r["errors"], "count")]:
            row(cfg, "prefill", metric, val, unit, f, 1, isl, "cold, C1, 4 requests")

# GSM8K-200 (greedy, thinking off) + spec counters printed by bench-mimo.sh
for label, cfg in GSM.items():
    f = f"{HERE}/logs/gsm8k-{label}.json"
    d = json.load(open(f))
    row(cfg, "gsm8k", "accuracy", round(d["correct"] / d["n"], 4), "fraction", f,
        notes=f"{d['correct']}/{d['n']}, last 200 test questions, T=0, thinking off")
for label, cfg in BENCHLOG.items():
    f = f"{HERE}/logs/bench-{label}.log"
    m = re.search(r"GSM8K spec: drafts=(\d+) accepted/draft=([\d.]+) accept_rate=([\d.]+)%", open(f).read())
    if m:
        row(cfg, "gsm8k", "spec_drafts", int(m.group(1)), "count", f, notes="during the GSM8K run")
        row(cfg, "gsm8k", "accepted_tokens_per_draft", float(m.group(2)), "tokens", f,
            notes="accepted draft tokens per draft (excludes the bonus token)")
        row(cfg, "gsm8k", "accept_rate", round(float(m.group(3)) / 100, 4), "fraction", f,
            notes="accepted / drafted tokens")

# streamed prefix-hit TTFT
for label, cfg in PREFIX.items():
    f = f"{HERE}/logs/prefix-hit-{label}.log"
    for line in open(f):
        m = re.match(r"len~(\d+)K cold=(\d+)ms hit_median=(\d+)ms hit_min=(\d+)ms hit_tokens/req=(\d+)", line)
        if not m:
            continue
        n = int(m.group(1)) * 1000
        for metric, val, unit in [("cold_ttft_ms", int(m.group(2)), "ms"), ("hit_ttft_median_ms", int(m.group(3)), "ms"),
                                  ("hit_ttft_min_ms", int(m.group(4)), "ms"), ("hit_tokens_per_request", int(m.group(5)), "tokens")]:
            row(cfg, "prefix-hit-ttft", metric, val, unit, f, 1, n, "approximate prompt length; 5 hit repeats")

# DFlash k analysis from the long-generation run with per-position acceptance counters (longgen lane)
f = f"{ROOT}/longgen/runs/mimo-dflash7-perpos/run.json"
summ = json.load(open(f))["summary"]
for phase in ("spec_all", "spec_thinking", "spec_answer"):
    for k, v in summ[phase]["tokens_per_step_if_k"].items():
        row("dflash7-longgen", "longgen", "tokens_per_step_if_k", v, "tokens", f, 1, None,
            f"k={k}; {phase.split('_')[1]}; 1 + sum of P(accept length >= i) for i <= k")

# session-note values without a kept raw file
row("dflash7-7f1a", "other", "kv_capacity_tokens", 2120000, "tokens", None,
    notes="group-aware capacity with stock grouping, ~2.12M (from session notes; raw file not kept)")
for k, v in ((3, 415), (5, 491), (7, 545)):
    row("dflash7-longgen", "longgen", "projected_c1_tok_s_if_k", v, "tok/s", None, 1, None,
        f"k={k}; projection from per-position acceptance plus an assumed ~0.3-0.5 ms verify cost per extra token "
        "(from session notes; raw file not kept)")

with open(f"{HERE}/results.jsonl", "w") as out:
    for r in rows:
        out.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows to {os.path.relpath(HERE, ROOT)}/results.jsonl")
