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
# 2026-09-29: 32 sequences and pinned FlashInfer tactics (both arms on the replay58132 lane; arm-s24-live = that lane
# rebooted with 24 sequences and live autotuning, as served on 2026-09-28).
E_REPLAY = E_PROD + ", k 5/3/3, vllm#58132 overlay"
CONFIGS_0929 = {
    "arm-s24-live": (E_REPLAY + ", 24 sequences, live FlashInfer autotune", True),
    "arm-s32-pin-a": (E_REPLAY + ", 32 sequences, pinned FlashInfer tactics", True),
    "arm-s32-pin-b": (E_REPLAY + ", 32 sequences, pinned FlashInfer tactics", True),
}
DATES.update({c: "2026-09-29" for c in CONFIGS_0929})
CONFIGS.update(CONFIGS_0929)
# 2026-10-02/03: sharing the GB300 with MiniMax-H3 (rowmap h265 = 265 hot experts, NVFP4 Engram tables, then
# nvfp4_ds_mla KV); every config here also has 32 sequences and pinned tactics.
E_H265 = E_REPLAY + ", 32 sequences, pinned tactics, rowmap-mix-v1-h265, NVFP4 Engram"
CONFIGS_1002 = {
    "h265-nvfp4e-u92": (E_H265 + ", fp8_ds_mla KV, GPU util 0.92", True),
    "h265-nvfp4e-u89": (E_H265 + ", fp8_ds_mla KV, GPU util 0.89", True),
    "h265-nvfp4e-nvfp4kv-u89": (E_H265 + ", nvfp4_ds_mla KV, GPU util 0.89", True),
    "ds41-h3-u885": (E_H265 + ", nvfp4_ds_mla KV, GPU util 0.885 (swap-to-ds41-h3.sh)", True),
}
DATES.update({c: "2026-10-02" for c in CONFIGS_1002})
DATES["ds41-h3-u885"] = "2026-10-03"
CONFIGS.update(CONFIGS_1002)
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
       "gsm8k-replay58132.json": ("replay58132", "production + vllm#58132"),
       "gsm8k-arm-s24-live.json": ("arm-s24-live", "24 sequences, live autotune"),
       "gsm8k-arm-s32-pin-a.json": ("arm-s32-pin-a", "32 sequences, pinned tactics"),
       "gsm8k-arm-s32-pin-b.json": ("arm-s32-pin-b", "32 sequences, pinned tactics"),
       "gsm8k-ds41h3-h265-nvfp4e.json": ("h265-nvfp4e-u92", "265 hot, NVFP4 Engram, fp8_ds_mla KV"),
       "gsm8k-ds41-nvfp4kv.json": ("h265-nvfp4e-nvfp4kv-u89", "265 hot, NVFP4 Engram, nvfp4_ds_mla KV")}
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
REASON_NOTE = "closed-loop reasoning (GSM8K/MMLU-Pro prompts, thinking on at 'high', T=1.0, top_p 0.95), {:g} s window"
REASON_TAGS = {"arm-s24-live": "arm-s24-live", "arm-s32-pin-a": "arm-s32-pin-a", "arm-s32-pin-b": "arm-s32-pin-b",
               "ds41h3-h265-nvfp4e": "h265-nvfp4e-u92", "ds41-nvfp4kv": "h265-nvfp4e-nvfp4kv-u89"}
for f in sorted(glob.glob(os.path.join(LOGS, "reason-*.json"))):
    tag = os.path.basename(f)[len("reason-"):-len(".json")]
    cfg = REASON_TAGS.get(tag) or (tag if tag in CONFIGS_0928 else f"k-{tag}")
    if cfg not in CONFIGS_0928 and cfg not in REASON_TAGS.values():
        continue  # reason-*-during-*.json (C1 while H3 renders) are read below
    d = json.load(open(f))
    for r in d["results"]:
        src = rel(f)
        row(cfg, "other", "reasoning_aggregate_tok_s", r["agg_tok_s"], "tok/s", src, r["C"],
            notes=REASON_NOTE.format(d.get("window_s", 60)))
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

# 2026-09-29: Al-ENGR prose knee from the arm-suite console logs (catid decode, prefill, reasoning and GSM8K are above)
for cfg in CONFIGS_0929:
    f = os.path.join(LOGS, "seats32", f"{cfg}.out")
    kv = re.search(r"GPU KV cache size: ([\d,]+) tokens", open(f).read())
    if kv:
        row(cfg, "other", "kv_cache_tokens", int(kv.group(1).replace(",", "")), "tokens", rel(f),
            notes="GPU KV cache size from the boot log (GPU util 0.95, 285 hot, fp8_ds_mla, max model length 1,048,576)")
    for m in re.finditer(r"\[" + re.escape(cfg) + r"\] conc=\s*(\d+):\s+([\d.]+) agg tok/s\s+([\d.]+) per-stream", open(f).read()):
        row(cfg, "knee", "aggregate_tok_s", float(m.group(2)), "tok/s", rel(f), int(m.group(1)), notes="Al-ENGR prose knee")
        row(cfg, "knee", "per_user_tok_s", float(m.group(3)), "tok/s", rel(f), int(m.group(1)), notes="Al-ENGR prose knee")

# 2026-10-02/03: sharing the GB300 with MiniMax-H3
HC = os.path.join(LOGS, "h3-coexist")


def hc(name):
    return os.path.join(HC, name)


for name, cfg, note in (("prefill-ds41h3-h265-nvfp4e.jsonl", "h265-nvfp4e-u92", "cold, random token ids"),
                        ("prefill-nvfp4kv.jsonl", "h265-nvfp4e-nvfp4kv-u89", "cold, random token ids"),
                        ("prefill-warm.jsonl", "h265-nvfp4e-u89", "cold, random token ids; the lane's warm-up run"),
                        ("prefill-0885.jsonl", "ds41-h3-u885", "cold, random token ids; the lane's warm-up run")):
    for line in open(hc(name)):
        d = json.loads(line)
        row(cfg, "prefill", "prefill_tok_s", d["agg_prompt_tok_s"], "tok/s", rel(hc(name)), d["concurrency"], d["isl"], note)
        row(cfg, "prefill", "ttft_p50_s", d["ttft_p50_s"], "s", rel(hc(name)), d["concurrency"], d["isl"], note)

KV_NOTE = "GPU KV cache size from the boot log (max model length 1,048,576)"
for cfg, f, pat in (("h265-nvfp4e-u92", "boot-u92.txt", None), ("h265-nvfp4e-u89", "ds41-final-coexist.log", None),
                    ("ds41-h3-u885", "ds41-0885.log", None)):
    text = open(hc(f)).read()
    tok = int(re.search(r"GPU KV cache size: ([\d,]+) tokens", text).group(1).replace(",", ""))
    gib = float(re.search(r"Available KV cache memory: ([\d.]+) GiB", text).group(1))
    row(cfg, "other", "kv_cache_tokens", tok, "tokens", rel(hc(f)), notes=KV_NOTE)
    row(cfg, "other", "kv_cache_gib", gib, "GiB", rel(hc(f)), notes="available KV cache memory from the boot log")
row("h265-nvfp4e-nvfp4kv-u89", "other", "kv_cache_tokens", 4344140, "tokens", None,
    notes=KV_NOTE + "; session notes (the line was read from the container log, which is gone)")
row("h265-nvfp4e-nvfp4kv-u89", "other", "kv_cache_gib", 5.89, "GiB", rel(hc("ds41-nvfp4kv-ab.log")),
    notes="available KV cache memory from the boot log")

NEEDLE_NOTE = "passphrase needle at depths 10/50/90%, thinking off (tools/needle_test.py)"
for line in open(hc("ds41-h3-test.log")):
    m = re.match(r"ds41h3-h265-nvfp4e-(\d+) needle (\d+)/(\d+)", line)
    if m:
        n = {"125000": 108593, "500000": 433938, "1000000": 867523}[m.group(1)]
        row("h265-nvfp4e-u92", "other", "needle_pass_fraction", int(m.group(2)) / int(m.group(3)), "fraction",
            rel(hc("ds41-h3-test.log")), 1, n, NEEDLE_NOTE + f"; target {m.group(1)}")
for line in open(hc("needles-nvfp4kv.txt")):
    m = re.match(r"nvfp4kv-(\d+) needle (\d+)/(\d+)", line)
    if m:
        row("h265-nvfp4e-nvfp4kv-u89", "other", "needle_pass_fraction", int(m.group(2)) / int(m.group(3)), "fraction",
            rel(hc("needles-nvfp4kv.txt")), 1, int(m.group(1)), NEEDLE_NOTE + f"; context_tokens = target {m.group(1)}")
for cfg, f in (("h265-nvfp4e-u89", "ds41-final-coexist.log"), ("ds41-h3-u885", "ds41-0885.log")):
    m = re.search(r"warm-1m needle (\d+)/(\d+)", open(hc(f)).read())
    row(cfg, "other", "needle_pass_fraction", int(m.group(1)) / int(m.group(2)), "fraction", rel(hc(f)), 1, 1000000,
        "one needle at 50% depth in a ~1M-token target prompt (the lane's warm-up)")

# GPQA-Diamond: aggregate scores only. Per-item outputs stay out of this repository.
GPQA_NOTE = "GPQA-Diamond, 198 items, reasoning effort max, max_tokens 131,072, C32, llm-inference-bench"
for cfg, f, pat, sampling in (
        ("h265-nvfp4e-u92", "ds41-h3-test.log", r"gpqa ds41: (\d+) / (\d+) [\d.]+ % hit_max (\d+)", "T=1.0 top_p 0.95"),
        ("h265-nvfp4e-u89", "ds41-nvfp4kv-ab.log", r"gpqa ds41-h265-nvfp4engram-t0-c32-max: (\d+) / (\d+) [\d.]+ % hit_max (\d+)", "T=0"),
        ("h265-nvfp4e-nvfp4kv-u89", "ds41-nvfp4kv-ab.log", r"gpqa nvfp4kv: (\d+) / (\d+) [\d.]+ % hit_max (\d+)", "T=1.0 top_p 0.95"),
        ("h265-nvfp4e-nvfp4kv-u89", "ds41-nvfp4kv-ab.log", r"gpqa ds41-nvfp4kv-t0-c32-max: (\d+) / (\d+) [\d.]+ % hit_max (\d+)", "T=0")):
    m = re.search(pat, open(hc(f)).read())
    row(cfg, "other", "gpqa_diamond_accuracy", round(int(m.group(1)) / int(m.group(2)), 4), "fraction", rel(hc(f)),
        32, notes=f"{GPQA_NOTE}, {sampling}; {m.group(1)}/{m.group(2)}, {m.group(3)} answers hit max_tokens")
row("lane-20260930", "other", "gpqa_diamond_accuracy", round(173 / 198, 4), "fraction", None, 32,
    notes=f"{GPQA_NOTE}, T=0; 173/198, 18 answers hit max_tokens; the 285-hot lane as served on 2026-09-30; "
    "session notes (per-item output kept privately)", engine=E_REPLAY + ", 32 sequences, pinned tactics, 285 hot",
    qualified=True)

# FP8 vs NVFP4 KV numerics (tools/ds41_numerics.py) on the same lane config
f = hc("ds41-nvfp4kv-ab.log")
for line in open(f):
    if not line.startswith("numerics: "):
        continue
    d = json.loads(line[len("numerics: "):])
    if "short_tokens" in d:
        for k, unit in (("delta_nll", "nats"), ("ppl_ratio", "x"), ("argmax_agree", "fraction"),
                        ("abs_dlogprob_p50", "nats"), ("p90", "nats"), ("p99", "nats")):
            metric = {"p90": "abs_dlogprob_p90", "p99": "abs_dlogprob_p99"}.get(k, k)
            row("h265-nvfp4e-nvfp4kv-u89", "other", f"kv_numerics_{metric}", d[k], unit, rel(f),
                context=d["short_tokens"], notes="teacher-forced, nvfp4_ds_mla vs fp8_ds_mla KV, same lane config; "
                f"NLL {d['nll_fp8kv']} (FP8) vs {d['nll_nvfp4kv']} (NVFP4)")
    else:
        same, total = (int(x) for x in d["identical_prefix"].split("/"))
        row("h265-nvfp4e-nvfp4kv-u89", "other", "kv_numerics_greedy_identical_tokens", same, "tokens", rel(f),
            context=d["ctx"], notes=f"greedy continuation, first {total} tokens identical to FP8 KV; top-5 overlap at "
            f"the first token {d['first_top5_overlap']}/5")

# Coexistence: one 30-step 1024x576 H3 render while DS41 decodes reasoning traffic at C1
CO_NOTE = "MiniMax-H3 30-step 1024x576 render on the same GB300 while DS41 decodes reasoning traffic at C1"
for cfg, mode, tag, mem, log, during in (
        ("h265-nvfp4e-u92", "fl2va-min, 15 s t2va", "coexist-ds41-15s-s30", "gb300-mem-coexist.txt", "h3-render.log",
         "reason-ds41h3-h265-nvfp4e-during-h3.json"),
        ("h265-nvfp4e-u89", "fl2va-min, 15 s t2va", "final-A-15s", "mem-final-A-15s.txt", "final-A-15s.log",
         "reason-ds41-final-during-final-A-15s.json"),
        ("h265-nvfp4e-u89", "combined, 15 s t2va", "final-B-15s", "mem-final-B-15s.txt", "final-B-15s.log",
         "reason-ds41-final-during-final-B-15s.json"),
        ("h265-nvfp4e-u89", "combined, 6 s ref2va voice clone", "final-B-ref2va-6s", "mem-final-B-ref2va-6s.txt",
         "final-B-ref2va-6s.headers", "reason-ds41-final-during-final-B-ref2va-6s.json"),
        ("h265-nvfp4e-nvfp4kv-u89", "combined, 15 s t2va", "nvfp4-B-15s", "mem-nvfp4-B-15s.txt", "nvfp4-B-15s.log", None),
        ("ds41-h3-u885", "combined, 15 s t2va", "ds41-0885-B-15s", "mem-0885-B-15s.txt", "ds41-0885-B-15s.log", None)):
    text = open(hc(log)).read()
    ok = re.search(r"HTTP/1.1 (\d+)", text).group(1)
    wall = re.search(r"^wall ([\d.]+) s", text, re.M) or re.search(r"x-inference-time-s: ([\d.]+)", text)
    peak = max(int(x) for x in open(hc(mem)).read().split()) / 1024
    note = f"{CO_NOTE}; H3 {mode}; HTTP {ok}"
    row(cfg, "other", "h3_render_wall_s", round(float(wall.group(1)), 1), "s", rel(hc(log)), notes=note)
    row(cfg, "other", "gb300_peak_used_gib", round(peak, 1), "GiB", rel(hc(mem)),
        notes=note + "; nvidia-smi memory.used sampled every 1 s (total 250.7 GiB)")
    hdr = hc(f"{tag}.headers") if os.path.exists(hc(f"{tag}.headers")) else None
    if hdr:
        pm = re.search(r"x-peak-memory-mb: ([\d.]+)", open(hdr).read())
        row(cfg, "other", "h3_peak_memory_gib", round(float(pm.group(1)) / 1024, 1), "GiB", rel(hdr), notes=note)
    if during:
        r = json.load(open(os.path.join(LOGS, during)))["results"][0]
        row(cfg, "other", "reasoning_aggregate_tok_s_during_h3_render", r["agg_tok_s"], "tok/s",
            rel(os.path.join(LOGS, during)), 1, notes=note)
m = re.search(r'DS41 C1 during lean: \{"C": 1, "agg_tok_s": ([\d.]+)', open(hc("ds41-final-coexist.log")).read())
row("h265-nvfp4e-u89", "other", "reasoning_aggregate_tok_s", float(m.group(1)), "tok/s",
    rel(hc("ds41-final-coexist.log")), 1, notes="C1, no H3 render running (the first test A, whose H3 failed to start); "
    + REASON_NOTE.format(45))
for cfg, f, pat in (("h265-nvfp4e-u89", "ds41-final-coexist.log", r"after warm-up: DS41 ([\d.]+) GiB"),
                    ("ds41-h3-u885", "ds41-0885.log", r"after warm-up: DS41 ([\d.]+) GiB")):
    row(cfg, "other", "ds41_gpu_used_gib_after_warmup", float(re.search(pat, open(hc(f)).read()).group(1)), "GiB",
        rel(hc(f)), notes="EngineCore memory after a 1M-token needle, a 128K prefill and C16 decode")

with open(os.path.join(M3, "results.jsonl"), "w") as out:
    for r in rows:
        out.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows")
