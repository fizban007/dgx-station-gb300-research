# SPDX-License-Identifier: Apache-2.0
"""FP8-path regression: run once in the stock image and once with the overlay mounted
(TEST_MODE=stock|overlay); compare_fp8_regress.py checks the two result files.

Same real FP8 rows (layer 14), same hash ids, every FP8 storage mode the lane can use
(pinned UVA = lane default, THP UVA, HBM-resident), forward and background grids, several
batch shapes: the output bytes (sha256) must match between runs and equal the reference
dequant. Also records the real-size FP8 Engram parameters (built on the meta device) and
FP8 lookup timings with and without the overlay.
"""
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch

from engram_test_utils import (
    FP8_DIR, bits_equal, fp8_tensors, init_vllm, load_rows, make_embedding, make_ids,
    ref_fp8, write_result,
)

MODE = os.environ.get("TEST_MODE", "unknown")
ROWS = int(os.environ.get("REGRESS_ROWS", 1 << 20))
SHAPES = (1, 3, 64, 1000, 8192)


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def time_fg(emb, t: int, iters: int = 30) -> float:
    gen = torch.Generator().manual_seed(500 + t)
    ids_list = [make_ids(t, emb, gen) for _ in range(iters + 3)]
    buf = torch.empty((t, 24, 256), dtype=torch.bfloat16, device="cuda")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for ids in ids_list[:3]:
        emb.lookup(ids, buf)
    ev = []
    for ids in ids_list[3:]:
        flush.zero_()
        torch.cuda._sleep(50_000)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        emb.lookup(ids, buf)
        e.record()
        ev.append((s, e))
    torch.cuda.synchronize()
    return round(float(np.median([s.elapsed_time(e) * 1e3 for s, e in ev])), 2)


def real_size_params() -> dict:
    """Real-size FP8 Engram modules (layers 1, 14) on the meta device: param signatures."""
    from vllm.models.deepseek_v41.common.engram import EngramLayout
    from vllm.models.deepseek_v41.nvidia.engram import Engram

    with open(f"{FP8_DIR}/config.json") as f:
        cfg = types.SimpleNamespace(**json.load(f)["text_config"])
    layout = EngramLayout(cfg)
    init_vllm(cpu_offload=False)
    out = {"layout_quant": getattr(layout, "quant", "<absent: stock>")}
    with torch.device("meta"):
        for i, layer in enumerate(layout.layer_ids):
            eng = Engram(cfg, None, layout, i, use_sequence_parallel=False,
                         prefix=f"model.layers.{layer}.engram")
            out[layer] = {n: [list(p.shape), str(p.dtype)] for n, p in eng.named_parameters()}
            out[f"{layer}_embedding"] = {
                "vocab": [eng.embed_tokens.vocab_start_idx, eng.embed_tokens.vocab_end_idx],
                "block_size": eng.embed_tokens.block_size,
                "weight_loader": eng.embed_tokens.weight.weight_loader.__name__,
            }
    init_vllm(cpu_offload=True)
    return out


def main() -> int:
    init_vllm()
    w, s = fp8_tensors(14)
    start = w.shape[0] // 3
    vals, scales = w.read_range(start, ROWS), s.read_range(start, ROWS)
    report = {"mode": MODE, "rows": ROWS, "row_start": start, "cases": {}, "ref_bit_identical": {},
              "timing_fg_us": {}}
    ok = True
    for storage in ("pinned", "thp", "hbm"):
        emb = make_embedding(ROWS, "fp8", cpu_offload=storage != "hbm", use_thp=storage == "thp")
        load_rows(emb, vals, scales, "fp8")
        for t in SHAPES:
            gen = torch.Generator().manual_seed(77 + t)
            ids = make_ids(t, emb, gen)
            if t == 1000:  # a few invalid ids (zero rows)
                ids[::97, 5] = -1
                ids[1::89, 7] = ROWS + 3
            fg = emb(ids)
            bg = torch.empty((t, 24, 256), dtype=torch.bfloat16, device="cuda")
            emb.lookup(ids, bg, background=True)
            torch.cuda.synchronize()
            flat = ids.cpu().long().reshape(-1)
            valid = (flat >= 0) & (flat < ROWS)
            want = torch.zeros((flat.numel(), 256), dtype=torch.bfloat16)
            want[valid] = ref_fp8(vals[flat[valid].numpy()], scales[flat[valid].numpy()])
            want = want.view(t, 24, 256)
            key = f"{storage}/T{t}"
            report["cases"][key] = {"fg": sha(fg), "bg": sha(bg)}
            eq = bits_equal(fg.cpu(), want) and bits_equal(bg.cpu(), want)
            report["ref_bit_identical"][key] = eq
            ok &= eq
        if storage == "pinned":
            for t in (1, 64, 8192):
                report["timing_fg_us"][t] = time_fg(emb, t)
        del emb
    report["real_size_params"] = real_size_params()
    report["pass_ref"] = bool(ok)
    print("wrote", write_result(f"fp8_regress_{MODE}.json", report), "ref PASS" if ok else "ref FAIL")
    print("timing", report["timing_fg_us"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
