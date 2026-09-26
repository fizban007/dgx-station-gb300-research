#!/usr/bin/env python3
"""Build deepseek-v4.1-flash/results.jsonl from the raw files in this lane.

Usage: python3 tools/make_results_jsonl.py            (writes ../results.jsonl)
       python3 tools/make_results_jsonl.py --tables   (also prints the markdown tables used in DETAILS.md)

Covers every configuration in this lane except M3 (MegaMoE + RTX PRO 6000 sidecar), whose rows live in
m3/results.jsonl. External baselines (catid, Al-ENGR) are added by hand with their https source.
"""
import glob
import json
import os
import sys

LANE = "deepseek-v4.1-flash"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the lane directory
MODEL = "DeepSeek-V4.1-Flash"

V20_IMAGE = "vLLM 0.29.1rc1.dev9+g2671fedfc (vllm/vllm-openai:nightly-2671fedfc7ae604761990603fc736c0c4f21de57)"
V20 = V20_IMAGE + " + Al-ENGR v15 pin-hot-experts hook"
V20_PEER = V20_IMAGE + " + hook-peer (v15 hook + Triton cold experts on RTX PRO 6000)"
EXP = "vLLM fork exp/ds41f-gb300 (working tree; head 89ba1db0) + b12x exp/ds41f-gb300 (8fdb5635)"
EXP2 = "vLLM fork exp2/ds41f-gb300 (77c7770b) + b12x sm103/residency (per RESULTS.md)"
PR = "vLLM fork feat/b12x-sm103-expert-residency (vllm#888) + b12x sm103/residency (b12x#426)"

# config id (= run dir under results/runs) -> (engine, knee file or None, short note)
CONFIGS = {
    "up-v20": (V20, "knee-v20-gracie-live.json", "Al-ENGR v20 reproduced; first boot, live FlashInfer autotune"),
    "up-v20-base2": (V20, "knee-v20-base2.json", "Al-ENGR v20 reproduced; loaded autotune set"),
    "up-v20-peer": (V20_PEER, "knee-v20-peer.json", "v20 + RTX PRO 6000 peer tier, run 1"),
    "up-v20-peer2": (V20_PEER, "knee-v20-peer2.json", "v20 + RTX PRO 6000 peer tier, run 2"),
    "b12x-base": (EXP, None, "b12x residency operator, all tiers, no speculation"),
    "fi-h290": (EXP, None, "FlashInfer TRTLLM hot HBM / cold Grace, positional 290 hot"),
    "fi-p12200": (EXP, None, "FlashInfer, calibrated 12,200 hot"),
    "fi-p12200-ds7": (EXP, None, "fi-p12200 + DSpark k=7"),
    "fi-p12200-ds3": (EXP, None, "fi-p12200 + DSpark k=3"),
    "fi-stitch-ds3": (EXP, None, "VMM-stitched expert tables + DSpark k=3"),
    "A-fi-tuned": (EXP, None, "uniform 305 hot, primed FI autotune, staging, DSpark k=3"),
    "pr-validate": (PR, None, "PR stack: b12x residency operator, 11,584 hot"),
    "C-peer": (EXP, None, "A + RTX PRO 6000 peer tier"),
    "D-peer-a16": (EXP, None, "C + A16 decode linears, 64 seqs"),
    "E-rebased": (EXP2, None, "D on exp2 (current karmic dev)"),
    "F-ksched": (EXP2, None, "E + Al-ENGR DSpark k-schedule (k=5 to 4 seqs, k=1 above)"),
}
DATE = "2026-09-24"  # every run in this lane ran on 2026-09-24 (UTC and US Eastern)

DECODE_METRICS = [  # (metric, json key, scale, unit)
    ("aggregate_tok_s", "aggregate_tps", 1, "tok/s"),
    ("per_user_tok_s_p50", "output_tps_per_user_p50", 1, "tok/s"),
    ("ttft_p50_ms", "ttft_p50", 1000, "ms"),
    ("itl_p50_ms", "inter_token_latency_p50", 1000, "ms"),
    ("request_latency_p50_s", "request_latency_p50", 1, "s"),
    ("accept_len", "server_spec_accept_length", 1, "tokens"),
]


def rel(path):
    return os.path.join(LANE, os.path.relpath(path, HERE))


def row(config, engine, benchmark, metric, conc, ctx, value, unit, source, notes=None, qualified=False):
    return {"lane": LANE, "model": MODEL, "config": config, "date": DATE, "engine": engine,
            "benchmark": benchmark, "metric": metric, "concurrency": conc, "context_tokens": ctx,
            "value": value, "unit": unit, "qualified": qualified, "source": source, "notes": notes}


def decode_rows(config, engine):
    out = []
    files = glob.glob(os.path.join(HERE, "results/runs", config, "decode/c*.json"))
    for f in sorted(files, key=lambda p: int(os.path.basename(p)[1:-5])):
        j = json.load(open(f))
        for r in j["results"]:
            assert r["num_errors"] == 0 and r["num_completed"] == r["request_count_target"], f
            for metric, key, scale, unit in DECODE_METRICS:
                v = r.get(key)
                if v is None or (metric == "accept_len" and not v):
                    continue  # no speculative decoding in this config
                out.append(row(config, engine, "catid-decode", metric, r["concurrency"], r["context_tokens"],
                               round(v * scale, 4), unit, rel(f), "8,192 in / 1,024 out, T=0, 5xC requests"))
    return out


def prefill_rows(config, engine):
    out = []
    f = os.path.join(HERE, "results/runs", config, "prefill/prefill.jsonl")
    if not os.path.exists(f):
        return out
    for line in open(f):
        d = json.loads(line)
        note = "random token ids, 1 output token, cache reset per point"
        out.append(row(config, engine, "prefill", "prefill_tok_s", d["concurrency"], d["isl"],
                       d["agg_prompt_tok_s"], "tok/s", rel(f), note))
        out.append(row(config, engine, "prefill", "ttft_p50_s", d["concurrency"], d["isl"],
                       d["ttft_p50_s"], "s", rel(f), note))
        out.append(row(config, engine, "prefill", "gpu_util_mean", d["concurrency"], d["isl"],
                       round(d["gpu_util_mean_pct"] / 100, 3), "fraction", rel(f), "nvidia-smi 1 Hz samples"))
    return out


def knee_rows(config, engine, knee):
    out = []
    if not knee:
        return out
    f = os.path.join(HERE, "results/upstream-logs", knee)
    for c, agg, per in json.load(open(f))["rows"]:
        note = "knee.sh: short prose prompt, 192 output tokens, thinking off, mean of 2 runs"
        out.append(row(config, engine, "knee", "aggregate_tok_s", c, None, round(agg, 4), "tok/s", rel(f), note))
        out.append(row(config, engine, "knee", "per_user_tok_s", c, None, round(per, 4), "tok/s", rel(f), note))
    return out


def micro_rows():
    out = []
    eng = "FlashInfer 0.6.18.post1 TRTLLM MXFP4 x MXFP8 routed MoE, one layer, all 384 experts in HBM"
    f = os.path.join(HERE, "overnight/micro/fi_moe.json")
    for d in json.load(open(f)):
        for metric, key in (("moe_layer_all_hot_us", "full_us"), ("moe_layer_split_295_89_us", "split_us")):
            out.append(row("fi-moe-micro", eng, "kernel-microbench", metric, None, None, round(d[key], 2), "us",
                           rel(f), f"{d['tokens']} tokens"))
    f = os.path.join(HERE, "overnight/micro/fi_moe_coldhost.json")
    for d in json.load(open(f)):
        out.append(row("fi-moe-micro-coldhost", eng.replace("all 384 experts in HBM", "89 cold experts in cacheable Grace memory"),
                       "kernel-microbench", "moe_layer_split_295_hbm_89_grace_us", None, None, round(d["split_us"], 2),
                       "us", rel(f), f"{d['tokens']} tokens"))
    for name, label in (("upstream.jsonl", "mhc-upstream"), ("b12x-tuned.jsonl", "mhc-b12x-tuned"),
                        ("b12x.jsonl", "mhc-b12x-default")):
        f = os.path.join(HERE, "mhc", name)
        for line in open(f):
            d = json.loads(line)
            out.append(row(label, "one mHC sublayer boundary, CUDA-graph replay, GB300", "kernel-microbench",
                           "mhc_boundary_us", None, None, d["us"], "us", rel(f), f"{d['tokens']} tokens"))
    return out


CATID = "https://github.com/catid/dgx_station_benchmarks/blob/ff8a496e5e027bbc462f81643361ef5516072a68/deepseek-v4.1-flash/data/"
ALENGR = ("https://github.com/J-M-Recipes/recipes/blob/dffd01cc29fb8dfed9c2a52192ee7e02e753ba26/recipes/"
          "dgx-station-gb300/deepseek-v4.1-flash-vllm-uva-dspark/results/2026-09-21-v20-promotion/README.md")
ALENGR_PAGE = "https://al-engr.com/gb300-deepseek-flash-41-testing.html"


def external_rows():
    def ext(config, date, engine, bench, metric, conc, ctx, value, unit, src, notes):
        r = row(config, engine, bench, metric, conc, ctx, value, unit, src, notes, qualified=None)
        r["date"] = date
        return r
    cat = "external:catid-vllm-pp2-dspark-2xgb300"
    eng = "vLLM PP2 + DSpark on TWO DGX Stations (2x GB300), per catid"
    out = [
        ext(cat, "2026-09-10", eng, "catid-decode", "per_user_tok_s_p50", 1, 8192, 252.9, "tok/s", CATID + "throughput.csv", "2 stations"),
        ext(cat, "2026-09-10", eng, "catid-decode", "aggregate_tok_s", 1, 8192, 248.5, "tok/s", CATID + "throughput.csv", "2 stations"),
        ext(cat, "2026-09-10", eng, "catid-decode", "aggregate_tok_s", 8, 8192, 1024.1, "tok/s", CATID + "throughput.csv", "2 stations"),
        ext(cat, "2026-09-10", eng, "catid-decode", "aggregate_tok_s", 16, 8192, 1677.7, "tok/s", CATID + "throughput.csv", "2 stations"),
        ext(cat, "2026-09-10", eng, "catid-decode", "aggregate_tok_s", 32, 8192, 2346.4, "tok/s", CATID + "throughput.csv", "2 stations"),
        ext(cat, "2026-09-10", eng, "prefill", "prefill_tok_s", 1, 16384, 35214.4, "tok/s", CATID + "prefill.csv", "2 stations"),
    ]
    al = "external:alengr-v20"
    eng = "Al-ENGR v20 'Clean Nightly': nightly 2671fedf + v15 hook, one DGX Station GB300"
    for boot, c1, c8, c16 in (("v20a", 180.9, 670, 979), ("v20b", 180.3, 661, 965)):
        for c, v in ((1, c1), (8, c8), (16, c16)):
            out.append(ext(al, "2026-09-21", eng, "knee", "aggregate_tok_s", c, None, v, "tok/s", ALENGR,
                           f"boot {boot}, knee x2, loaded autotune set"))
    out.append(ext("external:alengr-v15", None, "Al-ENGR v15 hook, one DGX Station GB300", "prefill",
                   "prefill_tok_s", 1, None, 22000, "tok/s", ALENGR_PAGE, "about 22K tok/s from 26K to 207K tokens"))
    return out


def main():
    rows = []
    for config, (engine, knee, _) in CONFIGS.items():
        rows += decode_rows(config, engine) + prefill_rows(config, engine) + knee_rows(config, engine, knee)
    rows += micro_rows() + external_rows()
    with open(os.path.join(HERE, "results.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {LANE}/results.jsonl", file=sys.stderr)
    if "--tables" in sys.argv:
        tables(rows)


def tables(rows):
    idx = {(r["config"], r["benchmark"], r["metric"], r["concurrency"], r["context_tokens"]): r["value"] for r in rows}
    print("\n| Run | C | agg tok/s | per-user tok/s p50 | TTFT p50 ms | ITL p50 ms | latency p50 s | accept |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for config in CONFIGS:
        for c in (1, 2, 4, 8, 16, 32, 64):
            k = lambda m: idx.get((config, "catid-decode", m, c, 8192))
            if k("aggregate_tok_s") is None:
                continue
            acc = k("accept_len")
            print(f"| {config} | {c} | {k('aggregate_tok_s'):,.1f} | {k('per_user_tok_s_p50'):,.1f} | "
                  f"{k('ttft_p50_ms'):,.0f} | {k('itl_p50_ms'):.2f} | {k('request_latency_p50_s'):.2f} | "
                  f"{'-' if acc is None else f'{acc:.2f}'} |")
    print("\n| Run | ISL | C | prefill tok/s | TTFT p50 s | GPU util |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        if r["benchmark"] == "prefill" and r["metric"] == "prefill_tok_s" and not r["config"].startswith("external"):
            key = (r["config"], "prefill")
            t = idx[key + ("ttft_p50_s", r["concurrency"], r["context_tokens"])]
            u = idx[key + ("gpu_util_mean", r["concurrency"], r["context_tokens"])]
            print(f"| {r['config']} | {r['context_tokens']:,} | {r['concurrency']} | {r['value']:,.1f} | {t:.3f} | {u:.0%} |")
    print("\n| Run | C1 | C2 | C4 | C8 | C12 | C16 |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for config in CONFIGS:
        vals = [idx.get((config, "knee", "aggregate_tok_s", c, None)) for c in (1, 2, 4, 8, 12, 16)]
        if vals[0] is not None:
            print(f"| {config} | " + " | ".join(f"{v:,.1f}" for v in vals) + " |")


if __name__ == "__main__":
    main()
