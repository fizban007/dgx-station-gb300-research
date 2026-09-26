"""Build mimo-v2.6-pro/results.jsonl from the raw result files in this lane (run from the repo root or anywhere).

Rows whose raw output was not kept carry source=null; external receipts carry their URL.
"""
import json
import os
import re

LANE = "mimo-v2.6-pro"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "MiMo-V2.6-Pro-RL"
ENGINE = "vllm nightly-29468dde + hook/ overlays"
AL = ("https://github.com/J-M-Recipes/recipes/tree/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/"
      "dgx-station-gb300/mimo-v2.6-pro-vllm-uva-hotsplit")
QUAL = {"3tier-v2": True, "3tier-v3": False, "trt-v4": False}
rows = []


def row(config, benchmark, metric, value, unit, source, concurrency=None, context=None, qualified=None, notes="",
        engine=ENGINE, date="2026-09-25"):
    rows.append({"lane": LANE, "model": MODEL, "config": config, "date": date, "engine": engine,
                 "benchmark": benchmark, "metric": metric, "concurrency": concurrency, "context_tokens": context,
                 "value": value, "unit": unit,
                 "qualified": QUAL.get(config, qualified) if not config.startswith("external:") else None,
                 "source": source, "notes": notes})


def rel(p):
    return f"{LANE}/{p}"


# TTFT bench (Al-ENGR ttft_bench.py): one request per prompt size, 3 runs, medians
for tag in ("3tier-v2", "3tier-v3", "trt-v4"):
    p = f"results/ttft-{tag}.json"
    for _, r in json.load(open(os.path.join(ROOT, p))).items():
        n = r["prompt_tokens"]
        row(tag, "ttft-bench", "prefill_tok_s", r["prefill_tok_s"], "tok/s", rel(p), 1, n)
        row(tag, "ttft-bench", "ttft_s", r["ttft_s"], "s", rel(p), 1, n)
        row(tag, "ttft-bench", "decode_after_ttft_tok_s", r["decode_tok_s"], "tok/s", rel(p), 1, n)

# Knee (Al-ENGR knee.sh): aggregate and per-stream tok/s, 192-token prose, 2 reps
p = "results/logs/knee-3tier-v3.log"
for m in re.finditer(r"conc=\s*(\d+):\s+([\d.]+) agg tok/s\s+([\d.]+) per-stream", open(os.path.join(ROOT, p)).read()):
    c = int(m.group(1))
    row("3tier-v3", "knee", "aggregate_tok_s", float(m.group(2)), "tok/s", rel(p), c)
    row("3tier-v3", "knee", "per_user_tok_s", float(m.group(3)), "tok/s", rel(p), c)
for c, v in ((1, 32.3), (4, 61.2), (8, 79.7), (16, 102.2)):
    row("3tier-v2", "knee", "aggregate_tok_s", v, "tok/s", None, c, notes="session notes; raw log not kept")

# Peer wait (bench/peer_wait.py)
p = "results/logs/peerwait-3tier-v3.log"
for line in open(os.path.join(ROOT, p)):
    d = json.loads(line)
    for k, unit in (("tok_s", "tok/s"), ("ms_per_pass", "ms"), ("wait_ms_per_pass", "ms"), ("wait_us_per_wait", "us"),
                    ("rows_per_layer", "rows"), ("timeouts", "count")):
        row("3tier-v3", "other", f"peer_{k}", d[k], unit, rel(p), d["C"], notes="GB300 wait on the RTX PRO 6000 sidecar")

# Quality
p = "results/logs/gsm8k-3tier-v2.log"
m = re.search(r"(\d+)/(\d+)", open(os.path.join(ROOT, p)).read())
row("3tier-v2", "gsm8k", "accuracy", int(m.group(1)) / int(m.group(2)), "fraction", rel(p), notes="GSM8K-200, thinking off")
p = "results/logs/bfcl-3tier-v2-dev-summary.json"
b = json.load(open(os.path.join(ROOT, p)))
for suite in ("all", "simple_python", "multiple"):
    row("3tier-v2", "bfcl", f"accuracy_{suite}", b[suite]["acc"], "fraction", rel(p), 8,
        notes=f"BFCL dev, {b[suite]['ok']}/{b[suite]['n']}, Al-ENGR grader")

# Sidecar per-bucket latency (v4 boot)
p = "results/logs/peer_stats-trt-v4.json"
for bucket, s in json.load(open(os.path.join(ROOT, p))).items():
    row("trt-v4", "other", "sidecar_mean_us", s["mean_us"], "us", rel(p),
        notes=f"RTX PRO 6000 b12x w4a8_mx, row bucket {bucket}, {s['calls']} calls, mean rows {s['mean_rows']}")

# TRT-LLM vs Marlin microbenchmark (HBM, 96-expert bank, layer 5)
p = "results/logs/trt-vs-marlin.log"
for m in re.finditer(r"T=\s*(\d+) distinct=\s*\d+: Marlin\s+([\d.]+) us.*?TRT\s+([\d.]+) us.*?speedup ([\d.]+)x\s+cos ([\d.]+)",
                     open(os.path.join(ROOT, p)).read()):
    t = int(m.group(1))
    for name, v, unit in (("marlin_us", m.group(2), "us"), ("trtllm_us", m.group(3), "us"),
                          ("trtllm_speedup", m.group(4), "x"), ("cosine_trtllm_vs_marlin", m.group(5), "fraction")):
        row("microbench", "kernel-microbench", name, float(v), unit, rel(p), notes=f"T={t} tokens, 96 experts in HBM",
            qualified=None)

# C1 decode profile (v2), session notes
for part, ms in (("grace_bank", 9.47), ("dense", 5.78), ("hbm_bank", 3.34), ("glue_kernels", 3.3),
                 ("moe_align_sum", 1.53), ("attention", 1.14), ("peer_tier_gb300_kernels", 0.94)):
    row("3tier-v2", "other", f"c1_decode_ms_per_token_{part}", ms, "ms", None, 1,
        notes="torch profiler, session notes; trace not published")

# Al-ENGR receipts (external)
for n, pf, dec in ((2952, 1064.0, 33.2), (11524, 1235.0, 33.0), (45828, 1244.0, 32.7)):
    row("external:al-engr-v24-1m", "ttft-bench", "prefill_tok_s", pf, "tok/s", AL + "/results/2026-09-22-hotsplit/receipts/ttft-v24-1m.json", 1, n, date="2026-09-22", engine="vllm nightly-d05da62e + hotsplit")
    row("external:al-engr-v24-1m", "ttft-bench", "decode_after_ttft_tok_s", dec, "tok/s", AL + "/results/2026-09-22-hotsplit/receipts/ttft-v24-1m.json", 1, n, date="2026-09-22", engine="vllm nightly-d05da62e + hotsplit")
row("external:al-engr-v23", "ttft-bench", "prefill_tok_s", 1376, "tok/s", AL + "/README.md", 1, 11524, date="2026-09-24", engine="vllm nightly-d05da62e + hotsplit")
row("external:al-engr-v23", "ttft-bench", "decode_after_ttft_tok_s", 37.4, "tok/s", AL + "/README.md", 1, 11524, date="2026-09-24", engine="vllm nightly-d05da62e + hotsplit")
row("external:al-engr-v23", "bfcl", "accuracy_all", 0.9383, "fraction", AL + "/README.md", date="2026-09-24", engine="vllm nightly-d05da62e + hotsplit")
for c, agg in ((1, 30.666), (2, 40.223), (4, 50.530), (8, 63.682), (12, 57.922), (16, 62.994)):
    row("external:al-engr-v22-hot153", "knee", "aggregate_tok_s", agg, "tok/s", AL + "/results/2026-09-22-hotsplit/receipts/knee-v22-hot153.json", c, date="2026-09-22", engine="vllm nightly-d05da62e + hotsplit", notes="max-num-seqs 8")

with open(os.path.join(ROOT, "results.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows")
