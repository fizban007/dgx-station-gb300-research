# SPDX-License-Identifier: Apache-2.0
"""Lookup time, NVFP4 vs FP8, at decode- and prefill-like batch shapes (RTX PRO 6000, UVA).

The host table holds PERF_ROWS real rows of layer 1 (CHUNKS contiguous runs spread across
the table), in both formats, loaded through the production loaders into the production
host storage. Each timed lookup uses fresh random hash ids ([T, 24] heads, each head inside
its bucket range, strided like the model's view), the GPU L2 is flushed first, and the GPU is
held busy while the launch is queued, so the events time the kernel alone.

Variants: fp8 pinned (stock FP8 path: pin_memory=True), fp8 THP (use_thp), nvfp4 registered
(overlay default: exact-size cudaHostRegister), nvfp4 THP (use_thp), nvfp4 pinned (fallback).
"""
import os
import sys

import numpy as np
import torch

import vllm.models.deepseek_v41.nvidia.engram as nv_engram
from engram_test_utils import (
    Timer, bits_equal, fp8_tensors, global_scales, init_vllm, load_rows, make_embedding,
    make_ids, nvfp4_tensors, ref_fp8, ref_nvfp4, rss_gib, write_result,
)
from vllm.triton_utils import triton

PERF_ROWS = int(os.environ.get("PERF_ROWS", 1 << 24))
CHUNKS = 8
SHAPES = [int(x) for x in os.environ.get("SHAPES", "1,2,4,8,16,32,64,128,512,2048,8192").split(",")]
LAYER = 1


def iters_for(t: int) -> int:
    return 200 if t <= 128 else 50 if t <= 2048 else 20


class Bench:
    def __init__(self) -> None:
        self.flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")

    def time(self, fn, ids_list) -> dict:
        for ids in ids_list[:3]:
            fn(ids)
        torch.cuda.synchronize()
        events = []
        for ids in ids_list:
            self.flush.zero_()
            torch.cuda._sleep(50_000)  # queue the launch behind a busy GPU
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn(ids)
            e.record()
            events.append((s, e))
        torch.cuda.synchronize()
        us = np.array([s.elapsed_time(e) * 1e3 for s, e in events])
        return {"median_us": round(float(np.median(us)), 2), "p10_us": round(float(np.percentile(us, 10)), 2),
                "p90_us": round(float(np.percentile(us, 90)), 2), "iters": len(us)}


def read_table(fmt: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    w, s = (nvfp4_tensors if fmt == "nvfp4" else fp8_tensors)(LAYER)
    total = w.shape[0]
    per = PERF_ROWS // CHUNKS
    starts = [int(i * (total - per) / (CHUNKS - 1)) for i in range(CHUNKS)]
    vals = np.concatenate([w.read_range(st, per) for st in starts])
    scales = np.concatenate([s.read_range(st, per) for st in starts])
    return vals, scales, starts


def build(fmt: str, alloc: str, vals, scales, g):
    saved = nv_engram._allocate_huge_page_storage
    if alloc == "pinned" and fmt == "nvfp4":
        nv_engram._allocate_huge_page_storage = lambda *a, **k: None  # force the fallback
    try:
        r0 = rss_gib()
        with Timer() as t:
            emb = make_embedding(vals.shape[0], fmt, use_thp=(alloc == "thp"),
                                 global_scale=g if fmt == "nvfp4" else None)
        with Timer() as tl:
            load_rows(emb, vals, scales, fmt)
        info = {"alloc_s": round(t.s, 2), "load_s": round(tl.s, 2),
                "table_bytes": emb.weight.nbytes + emb.weight_scale_inv.nbytes,
                "rss_delta_gib": round(rss_gib() - r0, 3)}
    finally:
        nv_engram._allocate_huge_page_storage = saved
    return emb, info


def bench_variant(bench, emb, fmt, vals, scales, g, gen_seed) -> dict:
    out = {}
    gen = torch.Generator().manual_seed(gen_seed)
    # correctness spot check of this storage variant
    ids = make_ids(64, emb, gen)
    idx = ids.cpu().long().reshape(-1).numpy()
    ref = ref_nvfp4(vals[idx], scales[idx], g) if fmt == "nvfp4" else ref_fp8(vals[idx], scales[idx])
    out["spot_check_bit_identical"] = bits_equal(emb(ids).cpu(), ref.view(64, 24, 256))
    for t in SHAPES:
        gen = torch.Generator().manual_seed(gen_seed + t)
        ids_list = [make_ids(t, emb, gen) for _ in range(iters_for(t) + 3)]
        buf = torch.empty((t, 24, 256), dtype=torch.bfloat16, device="cuda")
        row = {}
        for bg in (False, True):
            row["bg" if bg else "fg"] = bench.time(lambda ids: emb.lookup(ids, buf, background=bg), ids_list)
        rows = t * 24
        row["rows"] = rows
        row["row_bytes"] = (128 + 16) if fmt == "nvfp4" else (256 + 8)
        row["fg_GBps"] = round(rows * row["row_bytes"] / (row["fg"]["median_us"] * 1e3), 2)
        out[t] = row
    # host-side launch cost of one decode-sized lookup (eager, as in production)
    ids = make_ids(1, emb, torch.Generator().manual_seed(7))
    buf = torch.empty((1, 24, 256), dtype=torch.bfloat16, device="cuda")
    for _ in range(50):
        emb.lookup(ids, buf, background=True)
    torch.cuda.synchronize()
    with Timer() as tt:
        for _ in range(2000):
            emb.lookup(ids, buf, background=True)
        torch.cuda.synchronize()
    out["cpu_us_per_lookup_T1"] = round(tt.s / 2000 * 1e6, 2)
    return out


def sweep_nvfp4(bench, emb) -> dict:
    """Raw-kernel config sweep (BLOCK_R x num_warps x grid cap) for the NVFP4 gather."""
    from vllm.models.deepseek_v41.common.engram import _engram_nvfp4_lookup_kernel

    weight, scales = emb._storage()
    sms = emb._num_sms
    res = {}
    for t in (1, 32, 128, 2048, 8192):
        gen = torch.Generator().manual_seed(99 + t)
        ids_list = [make_ids(t, emb, gen) for _ in range(min(iters_for(t), 50) + 3)]
        buf = torch.empty((t, 24, 256), dtype=torch.bfloat16, device="cuda")
        rows = t * 24
        for block_r in (8, 16, 32):
            for warps in (2, 4, 8):
                for cap in (1, 2, 4):
                    grid = min(triton.cdiv(rows, block_r), sms * cap)

                    def fn(ids, grid=grid, block_r=block_r, warps=warps):
                        _engram_nvfp4_lookup_kernel[(grid,)](
                            weight, scales, ids, buf, emb.global_scale, emb.vocab_start_idx,
                            emb.vocab_end_idx, rows, ids.stride(0), ids.stride(1),
                            HEAD_START=0, LOCAL_HEADS=24, TOTAL_HEADS=24, DIM=256,
                            QUANT_BLOCK=16, BLOCK_R=block_r, GRID=grid, num_warps=warps)

                    res[f"T{t}/R{block_r}/w{warps}/x{cap}"] = bench.time(fn, ids_list)["median_us"]
    return res


def main() -> int:
    init_vllm()
    g = global_scales()[LAYER]
    bench = Bench()
    report = {"perf_rows": PERF_ROWS, "chunks": CHUNKS, "layer": LAYER, "gpu": torch.cuda.get_device_name(0),
              "sms": torch.cuda.get_device_properties(0).multi_processor_count, "variants": {}}
    for fmt, allocs in (("nvfp4", ("registered", "thp", "pinned")), ("fp8", ("pinned", "thp"))):
        with Timer() as tr:
            vals, scales, starts = read_table(fmt)
        report[f"{fmt}_read_s"] = round(tr.s, 1)
        report["chunk_starts"] = starts
        for alloc in allocs:
            emb, info = build(fmt, alloc, vals, scales, g)
            if fmt == "nvfp4":
                info["storage"] = ("thp" if emb._packed is not None else
                                   "registered" if emb._registered is not None else "pinned")
            res = bench_variant(bench, emb, fmt, vals, scales, g, 1234)
            if fmt == "nvfp4" and alloc == "registered" and os.environ.get("SWEEP", "1") == "1":
                report["nvfp4_kernel_sweep"] = sweep_nvfp4(bench, emb)
            report["variants"][f"{fmt}:{alloc}"] = {"build": info, "results": res}
            print(f"{fmt}:{alloc} build {info}; T1 fg {res[SHAPES[0]]['fg']['median_us']}us "
                  f"T{SHAPES[-1]} fg {res[SHAPES[-1]]['fg']['median_us']}us", flush=True)
            del emb
            torch.cuda.empty_cache()
        del vals, scales
    # side-by-side summary (production-relevant: fp8 pinned vs nvfp4 registered)
    a, b = report["variants"]["fp8:pinned"]["results"], report["variants"]["nvfp4:registered"]["results"]
    report["summary_fp8pinned_vs_nvfp4registered"] = {
        t: {"fp8_fg_us": a[t]["fg"]["median_us"], "nvfp4_fg_us": b[t]["fg"]["median_us"],
            "fp8_bg_us": a[t]["bg"]["median_us"], "nvfp4_bg_us": b[t]["bg"]["median_us"],
            "speedup_fg": round(a[t]["fg"]["median_us"] / b[t]["fg"]["median_us"], 3),
            "speedup_bg": round(a[t]["bg"]["median_us"] / b[t]["bg"]["median_us"], 3)}
        for t in SHAPES
    }
    ok = all(v["results"]["spot_check_bit_identical"] for v in report["variants"].values())
    report["pass"] = ok
    print("wrote", write_result(os.environ.get("PERF_OUT", "lookup_perf.json"), report))
    for t, row in report["summary_fp8pinned_vs_nvfp4registered"].items():
        print(t, row)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
