# SPDX-License-Identifier: Apache-2.0
"""Load-path checks for the NVFP4 Engram overlay (overlay container, RTX PRO 6000).

1. vLLM's own config parsing of the merged view: hf_config carries engram_quant*, the
   EngramLayout reads nvfp4 + per-table global scales; the original dir still reads fp8.
2. vLLM's weight-file discovery on the merged view: 48 shards, 47/48 resolve to the NVFP4 files.
3. Real-size dry run: both real Engram modules (layers 1, 14; full 384M-row tables) are built
   with their host tables on the meta device, and the real shards 47/48 of the merged view are
   streamed through vLLM's safetensors iterator + the lane's weights mapper into the real
   parameter weight loaders (zero-copy mmap tensors; meta copies read no data). Checks names,
   dtypes, shapes, TP narrowing and the format guard (FP8 tensors into an NVFP4 config fail).
4. Real subset load: a 2-head-sized slice of the real layer-1 table (first ~32M rows,
   ~4.6 GB) is loaded by the production loader from the full checkpoint tensor into
   registered host memory, then looked up through Engram.prepare_embeddings/embed (side
   stream prefetch + event wait, as in serving) and compared with O_DIRECT reference rows.
5. Host footprint: exact registered vs pin_memory (power-of-two rounded) allocation, measured,
   and projected to the real table sizes.
"""
import glob
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

import vllm.models.deepseek_v41.nvidia.engram as nv_engram
from engram_test_utils import (
    FP8_DIR, MERGED_DIR, NVFP4_DIR, SHARD, Timer, bits_equal, init_vllm, make_embedding,
    nvfp4_tensors, ref_nvfp4, rss_gib, write_result,
)
from vllm.models.deepseek_v41.common.engram import EngramLayout


def pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def check_config(report: dict):
    from vllm.transformers_utils.config import get_config

    hf = get_config(MERGED_DIR, trust_remote_code=True)
    layout = EngramLayout(hf)
    hf_fp8 = get_config(FP8_DIR, trust_remote_code=True)
    layout_fp8 = EngramLayout(hf_fp8)
    report["config"] = {
        "hf_config_class": type(hf).__name__,
        "engram_quant": getattr(hf, "engram_quant", None),
        "engram_quant_block_size": getattr(hf, "engram_quant_block_size", None),
        "engram_quant_global_scale": getattr(hf, "engram_quant_global_scale", None),
        "layout": {"quant": layout.quant, "block": layout.quant_block_size,
                   "global_scales": layout.global_scales, "layer_ids": layout.layer_ids,
                   "num_embeddings": layout.num_embeddings, "row_bytes": layout.table_row_bytes(),
                   "kwargs_l1": layout.embedding_quant_kwargs(0)},
        "fp8_dir_layout": {"quant": layout_fp8.quant, "block": layout_fp8.quant_block_size,
                           "kwargs_l1": layout_fp8.embedding_quant_kwargs(0),
                           "row_bytes": layout_fp8.table_row_bytes()},
    }
    ok = (layout.quant == "nvfp4" and layout.quant_block_size == 16 and layout_fp8.quant == "fp8"
          and layout_fp8.embedding_quant_kwargs(0) == {} and layout.table_row_bytes() == 144)
    return hf, layout, ok


def check_discovery(report: dict) -> bool:
    from vllm.model_executor.model_loader.weight_utils import filter_duplicate_safetensors_files

    files = sorted(glob.glob(os.path.join(MERGED_DIR, "*.safetensors")))
    kept = filter_duplicate_safetensors_files(files, MERGED_DIR, "model.safetensors.index.json")
    real = {os.path.basename(f): os.path.realpath(f) for f in kept}
    report["discovery"] = {
        "globbed": len(files), "kept_after_index_filter": len(kept),
        "shard47": real.get(SHARD[1]), "shard48": real.get(SHARD[14]),
        "others_in_fp8_dir": all(os.path.dirname(p) == FP8_DIR for n, p in real.items()
                                 if n not in SHARD.values()),
    }
    d = report["discovery"]
    return (len(kept) == 48 and d["others_in_fp8_dir"]
            and d["shard47"] == f"{NVFP4_DIR}/{SHARD[1]}" and d["shard48"] == f"{NVFP4_DIR}/{SHARD[14]}")


def meta_storage(num_bytes, huge_pages=True):
    return torch.empty(num_bytes, dtype=torch.uint8, device="meta")


def build_engrams(hf, layout):
    from vllm.models.deepseek_v41.nvidia.engram import Engram

    engrams = {}
    with torch.device("cuda"):
        for i, layer in enumerate(layout.layer_ids):
            engrams[layer] = Engram(hf, None, layout, i, use_sequence_parallel=False,
                                    prefix=f"model.layers.{layer}.engram")
    return engrams


def check_dry_run(report: dict, hf, layout) -> tuple[bool, dict]:
    from vllm.model_executor.model_loader.weight_utils import safetensors_weights_iterator
    from vllm.models.deepseek_v41.nvidia.model import _make_deepseek_v4_weights_mapper

    saved = nv_engram._allocate_huge_page_storage
    nv_engram._allocate_huge_page_storage = meta_storage
    try:
        engrams = build_engrams(hf, layout)
    finally:
        nv_engram._allocate_huge_page_storage = saved
    params = {f"model.layers.{layer}.engram.{n}": p
              for layer, eng in engrams.items() for n, p in eng.named_parameters()}
    out = {"modules": {}, "loaded": {}, "unmapped_or_skipped": []}
    for layer, eng in engrams.items():
        e = eng.embed_tokens
        out["modules"][layer] = {
            "quant": e.quant, "global_scale": e.global_scale, "block_size": e.block_size,
            "weight": [list(e.weight.shape), str(e.weight.dtype), str(e.weight.device)],
            "scale": [list(e.weight_scale_inv.shape), str(e.weight_scale_inv.dtype)],
            "vocab": [e.vocab_start_idx, e.vocab_end_idx], "num_embeddings": e.num_embeddings,
            "host_bytes": e.weight.nbytes + e.weight_scale_inv.nbytes,
            "staged_rows": list(eng.staged_rows.shape), "prefetch_stream": eng._prefetch_stream is not None,
        }
    ok = True
    for scale_name in ("weight_scale", "weight_scale_inv"):  # MXFP8 lane / block-FP8 linears
        mapper = _make_deepseek_v4_weights_mapper("fp4", scale_name)
        files = [f"{MERGED_DIR}/{SHARD[1]}", f"{MERGED_DIR}/{SHARD[14]}"]
        for name, tensor in mapper.apply(safetensors_weights_iterator(files, False)):
            if ".engram.embed_tokens." not in name:
                out["unmapped_or_skipped"].append(name) if name not in params else None
                continue
            param = params[name]
            with Timer() as t:
                param.weight_loader(param, tensor)  # meta destination: checks only
            out["loaded"][f"{scale_name}:{name}"] = {
                "ckpt": [list(tensor.shape), str(tensor.dtype)],
                "param": [list(param.shape), str(param.dtype)], "loader": param.weight_loader.__name__,
                "s": round(t.s, 3)}
        ok &= sum(k.startswith(scale_name + ":") for k in out["loaded"]) == 4
    out["unmapped_or_skipped"] = sorted(set(out["unmapped_or_skipped"]))
    # Format guard: the original FP8 tables must not load into an NVFP4 config.
    guard = {}
    with safe_open(f"{FP8_DIR}/{SHARD[1]}", framework="pt") as f:
        e = engrams[1].embed_tokens
        for ck, param in (("weight", e.weight), ("scale", e.weight_scale_inv)):
            try:
                param.weight_loader(param, f.get_tensor(f"layers.1.engram.embed.{ck}"))
                guard[ck] = "LOADED (bad)"
                ok = False
            except (ValueError, AssertionError) as exc:
                guard[ck] = f"rejected: {type(exc).__name__}: {str(exc)[:160]}"
    out["fp8_into_nvfp4_guard"] = guard
    report["dry_run"] = out
    ok &= all(m["quant"] == "nvfp4" and m["weight"][1] == "torch.uint8" for m in out["modules"].values())
    return ok, engrams


def check_subset(report: dict, engrams, layout, g1: float) -> bool:
    """Real rows of layer 1 via the production loader into registered memory; lookups
    through Engram.prepare_embeddings/embed (side-stream prefetch)."""
    eng = engrams[1]
    heads = [p for per in layout.primes[0] for p in per]
    rows = heads[0] + heads[1]
    emb = make_embedding(rows, "nvfp4", global_scale=g1)  # 24 heads over the first `rows` rows
    r0 = rss_gib()
    with safe_open(f"{MERGED_DIR}/{SHARD[1]}", framework="pt") as f:
        w = f.get_tensor("layers.1.engram.embed.weight")  # full [384M, 128] mmap view
        s = f.get_tensor("layers.1.engram.embed.scale")
        with Timer() as t:
            emb.weight.weight_loader(emb.weight, w)
            emb.weight_scale_inv.weight_loader(emb.weight_scale_inv, s)
    nbytes = emb.weight.nbytes + emb.weight_scale_inv.nbytes
    out = {"rows": rows, "bytes": nbytes, "load_s": round(t.s, 2),
           "load_GBps": round(nbytes / t.s / 1e9, 2), "rss_delta_gib": round(rss_gib() - r0, 2),
           "storage": "registered" if emb._registered is not None else "pinned"}
    eng.embed_tokens = emb  # swap the meta table for the real subset
    wq, sq = nvfp4_tensors(1)
    ok = True
    gen = torch.Generator().manual_seed(11)
    for t_tokens in (4, 8192):
        ids = (torch.rand((t_tokens, 24), generator=gen, dtype=torch.float64) * rows).long()
        ids[0, :3] = torch.tensor([0, rows - 1, rows - 2])
        ids_gpu = ids.to(torch.int32).cuda()
        eng.prepare_embeddings(ids_gpu)  # async lookup on the prefetch stream
        got = eng.embed(ids_gpu).cpu()  # waits on the prefetch event
        idx = ids.reshape(-1).numpy()
        uniq, inv = np.unique(idx, return_inverse=True)
        ref = ref_nvfp4(wq.read_rows(uniq), sq.read_rows(uniq), g1)[torch.from_numpy(inv)]
        same = bits_equal(got, ref.view(t_tokens, 24, 256))
        out[f"T{t_tokens}_prefetch_embed_bit_identical"] = same
        ok &= same
    report["subset_load"] = out
    del emb
    eng.embed_tokens = None
    return ok


def check_footprint(report: dict, layout) -> bool:
    rows = 3 << 21  # 6,291,456 rows: 0.75 x a power of two
    out = {"rows": rows}
    saved = nv_engram._allocate_huge_page_storage
    for name, fmt, kw in (("nvfp4_registered", "nvfp4", {}), ("nvfp4_thp", "nvfp4", {"use_thp": True}),
                          ("nvfp4_pin_memory_fallback", "nvfp4", {"force_pinned": True}),
                          ("fp8_stock_pin_memory", "fp8", {})):
        force = kw.pop("force_pinned", False)
        if force:
            nv_engram._allocate_huge_page_storage = lambda *a, **k: None
        try:
            torch.cuda.synchronize()
            h0 = torch.cuda.memory.host_memory_stats().get("allocated_bytes.current", 0)
            r0 = rss_gib()
            emb = make_embedding(rows, fmt, global_scale=0.004 if fmt == "nvfp4" else None, **kw)
            out[name] = {"tensor_bytes": emb.weight.nbytes + emb.weight_scale_inv.nbytes,
                         "rss_delta_bytes": int((rss_gib() - r0) * 2**30),
                         "caching_host_allocator_bytes":
                             torch.cuda.memory.host_memory_stats().get("allocated_bytes.current", 0) - h0,
                         "is_pinned": bool(emb.weight.is_pinned())}
            del emb
        finally:
            nv_engram._allocate_huge_page_storage = saved
    # Projection to the real tables (TP=1: one full table per layer).
    proj = {}
    for fmt, (vw, sw) in (("nvfp4", (128, 16)), ("fp8", (256, 8))):
        exact = sum(n * (vw + sw) for n in layout.num_embeddings)
        rounded = sum(pow2(n * vw) + pow2(n * sw) for n in layout.num_embeddings)
        proj[fmt] = {"exact_bytes": exact, "exact_GiB": round(exact / 2**30, 2),
                     "pin_memory_rounded_bytes": rounded, "pin_memory_rounded_GiB": round(rounded / 2**30, 2)}
    out["projection_real_tables"] = proj
    report["footprint"] = out
    reg, pin = out["nvfp4_registered"], out["nvfp4_pin_memory_fallback"]
    return (reg["is_pinned"] and reg["caching_host_allocator_bytes"] == 0
            and pin["caching_host_allocator_bytes"] > pin["tensor_bytes"])


def main() -> int:
    init_vllm()
    report = {}
    checks = {}
    hf, layout, checks["config"] = check_config(report)
    checks["discovery"] = check_discovery(report)
    checks["dry_run"], engrams = check_dry_run(report, hf, layout)
    checks["subset_load"] = check_subset(report, engrams, layout, layout.global_scales[0])
    checks["footprint"] = check_footprint(report, layout)
    report["checks"] = checks
    report["pass"] = all(checks.values())
    print("wrote", write_result("load_path.json", report), checks)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
