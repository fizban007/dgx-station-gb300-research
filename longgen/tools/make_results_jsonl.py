#!/usr/bin/env python3
"""Build longgen/results.jsonl from runs/*/run.json. Thinking/answer token splits are derived from the per-chunk
timeline: thinking = cumulative completion tokens at the last usage chunk before the first answer (content) token.
Run from anywhere: python3 longgen/tools/make_results_jsonl.py"""
import bisect, json, os

LANE = "longgen"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../longgen
ROOT = os.path.dirname(HERE)
MODEL = "MiMo-V2.6-Flash-RL"
DATE = "2026-09-25"
ENGINE = ("vLLM via mimo-v2.6-flash/launch-mimo.sh, DFlash k=7, one GB300 TP1; image and KV grouping not recorded "
          "in the run files (most likely nightly 29468dde)")
RUNS = {"mimo-dflash7-run1": "cap_on=all (earlier long-gen.py revision)",
        "mimo-dflash7-run2": "cap_on=content (earlier long-gen.py revision); baseline",
        "mimo-dflash7-perpos": "cap_on=content; current long-gen.py with per-position counters",
        "mimo-pro-dflash3": "cap_on=content; current long-gen.py; MiMo-V2.6-Pro v6 lane"}
# Per-run model and engine where they differ from the MiMo-V2.6-Flash defaults above.
RUN_MODEL = {"mimo-pro-dflash3": ("MiMo-V2.6-Pro-RL",
             "vLLM nightly-29468dde + mimo-v2.6-pro hook, v6: HBM + RTX PRO 6000 sidecar + Grace, TRT-LLM banks, "
             "FP8 KV, DFlash k=3")}
rows = []


def row(config, metric, value, unit, source, notes=None):
    model, engine = RUN_MODEL.get(config, (MODEL, ENGINE))
    rows.append({"lane": LANE, "model": model, "config": config, "date": DATE, "engine": engine,
                 "benchmark": "longgen", "metric": metric, "concurrency": 1, "context_tokens": None,
                 "value": value, "unit": unit, "qualified": False,
                 "source": os.path.relpath(source, ROOT), "notes": notes})


for run, desc in RUNS.items():
    f = f"{HERE}/runs/{run}/run.json"
    d = json.load(open(f))
    s, tl = d["summary"], d["timeline"]
    n = lambda note=None: "; ".join(x for x in (desc, note) if x)
    for metric, key, unit, scale in [("completion_tokens", "completion_tokens", "tokens", 1),
                                     ("wall_s", "wall_s", "s", 1), ("ttft_ms", "ttft_s", "ms", 1000),
                                     ("e2e_tok_s", "e2e_tok_s", "tok/s", 1), ("decode_tok_s", "decode_tok_s", "tok/s", 1),
                                     ("peak_1s_tok_s", "peak_1s_tok_s", "tok/s", 1),
                                     ("peak_5s_tok_s", "peak_5s_tok_s", "tok/s", 1),
                                     ("reasoning_chars", "chars_reasoning", "count", 1),
                                     ("answer_chars", "chars_content", "count", 1),
                                     ("spec_drafts", "spec_drafts", "count", 1),
                                     ("accepted_tokens_per_draft", "accepted_per_draft", "tokens", 1)]:
        row(run, metric, round(s[key] * scale, 3), unit, f, n())
    row(run, "accept_rate", round(s["accept_rate_pct"] / 100, 4), "fraction", f, n("accepted / drafted tokens"))
    a = s.get("answer_starts_s")
    if a is None:
        row(run, "thinking_tokens", s["completion_tokens"], "tokens", f, n("no answer produced"))
        row(run, "thinking_tok_s", round(s["completion_tokens"] / s["wall_s"], 1), "tok/s", f, n("no answer produced"))
        row(run, "answer_tokens", 0, "tokens", f, n("no answer produced"))
    else:
        ts = [t for t, _ in tl]
        i = bisect.bisect_left(ts, a)
        t_think, think = tl[i - 1]
        ans = s["completion_tokens"] - think
        row(run, "answer_start_s", a, "s", f, n())
        row(run, "thinking_tokens", think, "tokens", f, n("derived from timeline"))
        row(run, "thinking_tok_s", round(think / t_think, 1), "tok/s", f, n("derived; thinking tokens / time to answer start"))
        row(run, "answer_tokens", ans, "tokens", f, n("derived from timeline"))
        row(run, "answer_tok_s", round(ans / (tl[-1][0] - t_think), 1), "tok/s", f, n("derived; answer tokens / answer time"))
    for phase in ("spec_all", "spec_thinking", "spec_answer"):
        if phase not in s:
            continue
        p = s[phase]
        name = phase.split("_")[1]
        row(run, "spec_drafts", p["drafts"], "count", f, n(f"phase={name}"))
        row(run, "accepted_tokens_per_draft", p["accepted_per_draft"], "tokens", f, n(f"phase={name}"))
        for pos, v in enumerate(p["p_len_ge"], 1):
            row(run, "p_accept_len_ge", v, "fraction", f, n(f"phase={name}; position>={pos}"))

# Headless-browser smoke test (tools/smoke_tetris.cjs), where it was run
for run in RUNS:
    f = f"{HERE}/runs/{run}/smoke.json"
    if os.path.exists(f):
        d = json.load(open(f))
        row(run, "smoke_runtime_errors", len(d["errors"]), "count", f, "; ".join(d["errors"])[:300])
        row(run, "smoke_canvas_changed", int(d["canvas_changed"]), "count", f,
            "1 if the first canvas changed after START + 7 s of key presses")

with open(f"{HERE}/results.jsonl", "w") as out:
    for r in rows:
        out.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows to {os.path.relpath(HERE, ROOT)}/results.jsonl")
