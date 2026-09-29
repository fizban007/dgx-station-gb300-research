"""Build deepseek-v4.1-flash/m3/results.jsonl from raw result files in this repo (run from anywhere).

Decode runs are llm-inference-bench JSONs (catid recipe); prefill runs are bench/bench_prefill.py JSONL; knee runs are
Al-ENGR's prose knee. Rows whose raw output was not kept carry source=null.
"""
import glob
import json
import os
import re

LANE = "deepseek-v4.1-flash/m3"
M3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(M3))
MODEL = "DeepSeek-V4.1-Flash"
E_M3 = "vllm nightly-7f1a5398 + m3 hook (deep_gemm_mega_moe)"
E_V20 = "vllm nightly-2671fedf (Al-ENGR v20 flags)"
# config id -> (engine, qualified). Speed rows of m3v2/m3v2-c1fix predate the sidecar FP8-dequant fix (quality invalid).
CONFIGS = {
    "up-v20": (E_V20, False), "up-v20-peer2": (E_V20 + " + peer v1 hook", False),
    "m3": (E_M3 + ", peer v1", False), "m3v2": (E_M3 + ", peer v2 (pre dequant fix)", False),
    "m3v2-c1fix": (E_M3 + ", peer v2 (pre dequant fix)", False), "m3v2-h285": (E_M3 + ", peer v2", True),
    "mix-v1": (E_M3 + ", peer v2", True), "phaseC-C0-control": (E_M3 + ", peer v2 retuned", True),
    "phaseC-C1-adaptive": (E_M3 + ", peer v2 retuned", False), "phaseC-C2-adaptive-k2": (E_M3 + ", peer v2 retuned", False),
    "phaseC-C3-adaptive-k3": (E_M3 + ", peer v2 retuned", False), "phaseC-C4-k2-to-8": (E_M3 + ", peer v2 retuned", True),
}
# 2026-09-28: image upgrade, hook send fusion, DSpark retunes, probabilistic drafting, LL GEMM, fused send v2, vllm#58132.
# qualified: the config passed GSM8K-200 itself, or differs from one that did only in the DSpark k schedule / CUDA-graph
# sizes / a bit-identical hook path (see DETAILS).
E_AF = "vllm nightly-af7f9488 + m3 hook (deep_gemm_mega_moe)"
E_PROD = E_AF + ", fused send v2, LL GEMM, probabilistic DSpark drafts"
CONFIGS_0928 = {
    "ab-base-7f1a": (E_M3 + ", peer v2 retuned, effort 75", True),
    "ab-new-af7f": (E_AF + ", peer v2 retuned, effort 75", True),
    "ab-fused-af7f": (E_AF + ", fused send v1, effort 75", True),
    "ab-fused-nomhcov-af7f": (E_AF + ", fused send v1, MEGA_MHC_OVERLAP=0, effort 75", True),
    "k-k521": (E_AF + ", fused send v1, k 5/2/1", True), "k-k532": (E_AF + ", fused send v1, k 5/3/2", True),
    "k-k543": (E_AF + ", fused send v1, k 5/4/3", True),
    "k-prod-k532cg": (E_AF + ", fused send v1, k 5/3/2, finer CUDA graphs", True),
    "k-base-rerun": (E_AF + ", fused send v1, k 5/3/2, finer CUDA graphs", True),
    "k-probdraft": (E_AF + ", fused send v1, k 5/3/2, probabilistic DSpark drafts", False),
    "k-pd-ll-k532": (E_AF + ", fused send v1, LL GEMM, probabilistic drafts, k 5/3/2", True),
    "k-pd-ll-k533": (E_AF + ", fused send v1, LL GEMM, probabilistic drafts, k 5/3/3", True),
    "k-pd-ll-k543": (E_AF + ", fused send v1, LL GEMM, probabilistic drafts, k 5/4/3", True),
    "prod-20260928b": (E_PROD + ", k 5/3/3", True),
    "replay58132": (E_PROD + ", k 5/3/3, vllm#58132 overlay", True),
}
DATES = {c: "2026-09-28" for c in CONFIGS_0928}
CONFIGS.update(CONFIGS_0928)
rows = []


def rel(path):
    return os.path.relpath(path, REPO)


def row(config, benchmark, metric, value, unit, source, concurrency=None, context=None, notes="", qualified=None, engine=None):
    eng, q = CONFIGS.get(config, (engine or E_M3, qualified))
    rows.append({"lane": LANE, "model": MODEL, "config": config, "date": DATES.get(config, "2026-09-24"),
                 "engine": engine or eng,
                 "benchmark": benchmark, "metric": metric, "concurrency": concurrency, "context_tokens": context,
                 "value": value, "unit": unit, "qualified": q if qualified is None else qualified,
                 "source": source, "notes": notes})


RUNS = os.path.join(REPO, "deepseek-v4.1-flash/results/runs")
for cfg in CONFIGS:
    for f in sorted(glob.glob(os.path.join(RUNS, cfg, "decode", "c*.json"))):
        for r in json.load(open(f)).get("results", []):
            c, n = r["concurrency"], r["context_tokens"]
            for metric, key, scale, unit in (("aggregate_tok_s", "aggregate_tps", 1, "tok/s"),
                                             ("per_user_tok_s_p50", "output_tps_per_user_p50", 1, "tok/s"),
                                             ("ttft_p50_ms", "ttft_p50", 1000, "ms"),
                                             ("itl_p50_ms", "inter_token_latency_p50", 1000, "ms"),
                                             ("accept_len", "server_spec_accept_length", 1, "tokens")):
                if r.get(key) is not None:
                    row(cfg, "catid-decode", metric, round(r[key] * scale, 3), unit, rel(f), c, n)
    for f in glob.glob(os.path.join(RUNS, cfg, "prefill", "prefill.jsonl")):
        seen = set()
        for line in open(f):
            d = json.loads(line)
            if (d["isl"], d["concurrency"]) in seen:
                continue  # a repeated 16K row at 119.8K tok/s / 0% GPU util is a prefix-cache hit, not a cold prefill
            seen.add((d["isl"], d["concurrency"]))
            row(cfg, "prefill", "prefill_tok_s", d["agg_prompt_tok_s"], "tok/s", rel(f), d["concurrency"], d["isl"],
                "cold, random token ids")

for f, cfg in (("results/logs/prefill-mix.jsonl", "mix-v1"), ("results/logs/prefill-h285.jsonl", "m3v2-h285")):
    for line in open(os.path.join(M3, f)):
        d = json.loads(line)
        row(cfg, "prefill", "prefill_tok_s", d["agg_prompt_tok_s"], "tok/s", f"{LANE}/{f}", d["concurrency"], d["isl"],
            "cold, random token ids")

KNEE = os.path.join(REPO, "deepseek-v4.1-flash/results/upstream-logs")
for name, cfg in (("knee-m3.json", "m3"), ("knee-m3-warm.json", "m3"), ("knee-m3v2.json", "m3v2"),
                  ("knee-m3v2-h285.json", "m3v2-h285"), ("knee-mix-v1.json", "mix-v1"),
                  *((f"knee-{cfg}.json", cfg) for cfg in CONFIGS_0928)):
    f = os.path.join(KNEE, name)
    if not os.path.exists(f):
        continue
    for c, agg, per in json.load(open(f))["rows"]:
        row(cfg, "knee", "aggregate_tok_s", round(agg, 1), "tok/s", rel(f), c, notes=name)
        row(cfg, "knee", "per_user_tok_s", round(per, 2), "tok/s", rel(f), c, notes=name)

# Phase C DSpark sweep: prose "wait check" lines (aggregate tok/s, ms/pass, GB300 wait ms/pass)
for f, default in (("results/logs/phaseC-C0.txt", "phaseC-C0-control"), ("results/logs/phaseC-variants.txt", None)):
    cfg = default
    for line in open(os.path.join(M3, f)):
        m = re.match(r"=== (C\d)-(\S+)", line)
        if m:
            cfg = {"C1": "phaseC-C1-adaptive", "C2": "phaseC-C2-adaptive-k2", "C3": "phaseC-C3-adaptive-k3",
                   "C4": "phaseC-C4-k2-to-8"}[m.group(1)]
        m = re.search(r"wait check C(\d+): ([\d.]+) tok/s, ([\d.]+) ms/pass, GB300 wait ([\d.]+) ms/pass", line)
        if m:
            c = int(m.group(1))
            row(cfg, "other", "prose_aggregate_tok_s", float(m.group(2)), "tok/s", f"{LANE}/{f}", c,
                notes="short prose, 320 forced tokens per stream, T=0 (tools/measure_wait.py)")
            row(cfg, "other", "peer_wait_ms_per_pass", float(m.group(4)), "ms", f"{LANE}/{f}", c,
                notes=f"GB300 wait on the sidecar; {m.group(3)} ms per pass")

GSM = {"gsm8k-nopeer-static.json": ("m3-nopeer-static", "MEGA_PEER=0: cold experts via TRT-LLM from Grace, 295 hot"),
       "gsm8k-peer2-fixed-static.json": ("m3v2-fixed", "peer v2 after the dequant fix, 295 hot (rowmap-static-v1)"),
       "gsm8k-peer2-cal2.json": ("m3v2-cal2", "peer v2, rowmap-cal-v2 (295 hot, prefill-calibrated)"),
       "gsm8k-peer2-h285.json": ("m3v2-h285", "peer v2, rowmap-cal-v2-h285"),
       "gsm8k-mix-v1.json": ("mix-v1", "peer v2, rowmap-mix-v1"),
       "gsm8k-ab-base-7f1a.json": ("ab-base-7f1a", "nightly-7f1a5398, effort 75"),
       "gsm8k-ab-new-af7f.json": ("ab-new-af7f", "nightly-af7f9488, effort 75"),
       "gsm8k-ab-fused-af7f.json": ("ab-fused-af7f", "fused send v1"),
       "gsm8k-ab-fused-nomhcov-af7f.json": ("ab-fused-nomhcov-af7f", "fused send v1, mHC overlap off"),
       "gsm8k-prod-20260928b.json": ("prod-20260928b", "production 2026-09-28 evening"),
       "gsm8k-replay58132.json": ("replay58132", "production + vllm#58132")}
for name, (cfg, notes) in GSM.items():
    d = json.load(open(os.path.join(M3, "results/logs", name)))
    row(cfg, "gsm8k", "accuracy", d["correct"] / d["n"], "fraction", f"{LANE}/results/logs/{name}",
        notes=f"GSM8K-200 {d['correct']}/{d['n']}; {notes}", qualified=True)

f = "results/logs/peer_stats.json"
for bucket, s in json.load(open(os.path.join(M3, f))).items():
    row("phaseC-C4-k2-to-8", "other", "sidecar_mean_us", s["mean_us"], "us", f"{LANE}/{f}",
        notes=f"RTX PRO 6000 b12x w4a8_mx, row bucket {bucket}, {s['calls']} calls, mean rows {s['mean_rows']}")

for f in ("results/phase1.json", "results/phase1-nightly-shared.json", "results/phase1-v030-shared.json"):
    for r in json.load(open(os.path.join(M3, f))):
        note = f"{r['case']} ({r['experts']} experts), T={r['tokens']}"
        row("microbench", "kernel-microbench", "megamoe_us", r["mega_us"], "us", f"{LANE}/{f}", notes=note, qualified=False)
        row("microbench", "kernel-microbench", "flashinfer_trtllm_us", r["fi_us"], "us", f"{LANE}/{f}",
            notes=note + (f", shared expert via {r['fi_shared_backend']}" if "fi_shared_backend" in r else ", routed only"),
            qualified=False)
f = "results/peer_prefill_bench.json"
for r in json.load(open(os.path.join(M3, f))):
    row("microbench", "kernel-microbench", "b12x_w4a8_mx_us", r["b12x_us"], "us", f"{LANE}/{f}",
        notes=f"RTX PRO 6000, {r['tokens']} tokens, cold experts; cosine vs Triton {r['cosine_vs_triton']}", qualified=False)
    row("microbench", "kernel-microbench", "triton_gemv_us", r["triton_gemv_us"], "us", f"{LANE}/{f}",
        notes=f"RTX PRO 6000, {r['tokens']} tokens, cold experts", qualified=False)

for metric, v, unit, notes in (("real_text_prefill_tok_s", 35000, "tok/s", "64K real text, 34-36K range; m3v2"),
                               ("decode_cold_route_share_static", 0.048, "fraction", "rowmap-static-v1"),
                               ("decode_cold_route_share_cal_v2", 0.20, "fraction", "rowmap-cal-v2, 19-22% range"),
                               ("in_server_cosine_min_vs_trtllm", 0.9993, "fraction", "MEGA_PEER_CHECK, all layers")):
    row("m3v2", "other", metric, v, unit, None, notes=notes + "; session notes, raw output not kept")

LOGS = os.path.join(M3, "results/logs")
REASON_NOTE = "closed-loop reasoning (GSM8K/MMLU-Pro prompts, thinking on at 'high', T=1.0, top_p 0.95), 60 s window"
for f in sorted(glob.glob(os.path.join(LOGS, "reason-*.json"))):
    tag = os.path.basename(f)[len("reason-"):-len(".json")]
    cfg = tag if tag in CONFIGS_0928 else f"k-{tag}"
    if cfg not in CONFIGS_0928:
        continue
    for r in json.load(open(f))["results"]:
        src = rel(f)
        row(cfg, "other", "reasoning_aggregate_tok_s", r["agg_tok_s"], "tok/s", src, r["C"], notes=REASON_NOTE)
        row(cfg, "other", "reasoning_accept_tokens_per_step", r["accept_tokens_per_step"], "tokens", src, r["C"],
            notes=f"DSpark k={r['avg_k']}; per position {r['accept_per_position']}")

WAIT_NOTE = "C1/2/4 short decode (tools/measure_wait_c1.py); GB300 forward-pass time and wait on the sidecar"


def wait_rows(cfg, f):
    for line in open(f):
        if line.startswith("{") and '"ms_per_pass"' in line:
            d = json.loads(line)
            row(cfg, "other", "ms_per_pass", d["ms_per_pass"], "ms", rel(f), d["C"], notes=f"{WAIT_NOTE}; rep {d['rep']}")
            row(cfg, "other", "peer_wait_ms_per_pass", d["wait_ms_per_pass"], "ms", rel(f), d["C"],
                notes=f"{WAIT_NOTE}; rep {d['rep']}, timeouts {d['timeouts']}")


for cfg in ("ab-base-7f1a", "ab-new-af7f", "ab-fused-af7f", "ab-fused-nomhcov-af7f"):
    f = os.path.join(LOGS, f"{cfg}.out")
    wait_rows(cfg, f)
    m = re.search(r"reasoning C1: \d+ tokens in [\d.]+ s = ([\d.]+) tok/s", open(f).read())
    if m:
        row(cfg, "other", "reasoning_c1_tok_s", float(m.group(1)), "tok/s", rel(f), 1,
            notes="8 GSM8K prompts, thinking on, greedy, sequential; includes TTFT")
for name, cfg in (("passtime-probdraft.jsonl", "k-probdraft"), ("passtime-probdraft-llgemm.jsonl", "k-pd-ll-k532"),
                  ("passtime-prod-20260928b.jsonl", "prod-20260928b")):
    wait_rows(cfg, os.path.join(LOGS, name))

for name, cfg in (("realtext-base-20260928b.log", "prod-20260928b"), ("realtext-replay58132.log", "replay58132")):
    f = os.path.join(LOGS, name)
    first = True
    for line in open(f):
        m = re.match(r"(\d+) tokens:\s+([\d.]+) tok/s\s+ttft ([\d.]+)s\s+cold rows/token ([\d.]+)", line)
        if m:
            n = int(m.group(1))
            warm = "; first request after boot, includes one-time JIT warm-up" if first and cfg == "replay58132" else ""
            first = False
            row(cfg, "prefill", "real_text_prefill_tok_s", float(m.group(2)), "tok/s", rel(f), 1, n,
                f"real text (source code + docs), unique prefix, C1 (tools/realtext_prefill.py); cold rows/token {m.group(4)}{warm}")
            row(cfg, "prefill", "real_text_ttft_s", float(m.group(3)), "s", rel(f), 1, n, "real text, C1" + warm)

for name, cfg in (("longctx-base1.json", "prod-20260928b"), ("longctx-base2.json", "prod-20260928b"),
                  ("longctx-replay1.json", "replay58132")):
    f = os.path.join(LOGS, name)
    needles = json.load(open(f))["needle"]
    for target in sorted({r["target"] for r in needles}):
        rs = [r for r in needles if r["target"] == target]
        row(cfg, "other", "needle_pass_fraction", sum(r["pass"] for r in rs) / len(rs), "fraction", rel(f), 1,
            rs[0]["prompt_tokens"], f"passphrase at depths {[r['depth'] for r in rs]}, thinking off, T=0 "
            "(tools/longctx_check.py)")

MB = os.path.join(M3, "results/microbench")
for line in open(os.path.join(MB, "mxfp8_backends.jsonl")):
    d = json.loads(line)
    if "hbm_read_ref_TBps" in d:
        row("microbench-0928", "kernel-microbench", "hbm_read_TBps", d["hbm_read_ref_TBps"], "TB/s",
            rel(os.path.join(MB, "mxfp8_backends.jsonl")), notes="int64 sum over 2 GiB, CUDA graph", qualified=False,
            engine=E_AF)
        continue
    for be in ("cute-dsl", "cutedsl_low_latency", "cutlass", "cudnn"):
        if isinstance(d.get(be), (int, float)):
            row("microbench-0928", "kernel-microbench", f"mxfp8_gemm_us_{be.replace('-', '_')}", d[be], "us",
                rel(os.path.join(MB, "mxfp8_backends.jsonl")), d["M"],
                notes=f"{d['shape']} N={d['N']} K={d['K']}, M={d['M']} tokens, L2-cold, FlashInfer mm_mxfp8",
                qualified=False, engine=E_AF)
for line in open(os.path.join(MB, "mega_attn_variants.jsonl")):
    d = json.loads(line)
    row("microbench-0928", "kernel-microbench", "mega_attn_kernel_us", d["kernel_us"], "us",
        rel(os.path.join(MB, "mega_attn_variants.jsonl")), d["s_q"],
        notes=f"FlashMLA fused_norm_rope_attn_rope_cast_decode, {d['padded_heads']} padded heads (64 live), "
        f"{d['extra']} compressed cache, topk 128+512, s_q = query tokens", qualified=False, engine=E_AF)
for line in open(os.path.join(MB, "peer_fused2.txt")):
    m = re.match(r"T=\s*(\d+): old .*?([\d.]+) us/layer, new\s+([\d.]+) us/layer", line)
    if m:
        for metric, v in (("send_receive_us_per_layer_v1", float(m.group(2))), ("send_receive_us_per_layer_v2", float(m.group(3)))):
            row("microbench-0928", "kernel-microbench", metric, v, "us", rel(os.path.join(MB, "peer_fused2.txt")),
                int(m.group(1)), notes="hook send + receive per MoE layer (v1: quant + route_send + wait + scatter; "
                "v2: route_send2 + finish2), 40 layers per CUDA graph, T = tokens", qualified=False, engine=E_AF)

with open(os.path.join(M3, "results.jsonl"), "w") as out:
    for r in rows:
        out.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows")
