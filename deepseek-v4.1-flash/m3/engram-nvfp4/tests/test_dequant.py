# SPDX-License-Identifier: Apache-2.0
"""Dequant correctness of the NVFP4 Engram gather (overlay container, RTX PRO 6000).

For each real table (layers 1 and 14): N random rows (+ edge rows + one contiguous run) are
read straight from the NVFP4 shard (O_DIRECT, no whole-table load), loaded through the
overlay's weight loader into a pinned/registered host table, gathered on the GPU over UVA
with the production lookup (forward + background grid), and compared bit-for-bit with a
CPU torch reference dequant (16-entry e2m1 LUT, e4m3 scale, fp32 global, one bf16 rounding).
The same rows of the original FP8 table give the NVFP4-vs-FP8 relative error.
"""
import os
import sys

import numpy as np
import torch

from engram_test_utils import (
    Timer, bits_equal, fp8_tensors, global_scales, init_vllm, load_rows, make_embedding,
    nvfp4_tensors, ref_fp8, ref_fp8_fp32, ref_nvfp4, ref_nvfp4_fp32, write_result,
)

N_RANDOM = int(os.environ.get("N_RANDOM", 65536))
RUN = 4096


def expected_rows(ref: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    flat = ids.cpu().reshape(-1).long()
    valid = (flat >= 0) & (flat < ref.shape[0])
    out = torch.zeros((flat.numel(), ref.shape[1]), dtype=ref.dtype)
    out[valid] = ref[flat[valid]]
    return out.view(*ids.shape, ref.shape[1])


def rel_rms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def main() -> int:
    init_vllm()
    scales_g = global_scales()
    report = {"n_random": N_RANDOM, "layers": {}}
    ok = True
    for layer in (1, 14):
        wq, sq = nvfp4_tensors(layer)
        wf, sf = fp8_tensors(layer)
        total = wq.shape[0]
        assert wf.shape[0] == total and sq.shape == [total, 16] and wq.shape == [total, 128]
        assert (wq.dtype, sq.dtype, wf.dtype, sf.dtype) == ("U8", "F8_E4M3", "F8_E4M3", "F8_E8M0")
        rng = np.random.default_rng(1000 + layer)
        run0 = int(rng.integers(0, total - RUN))
        sample = np.unique(np.concatenate([
            rng.integers(0, total, N_RANDOM),
            [0, 1, 2, total - 2, total - 1],
            np.arange(run0, run0 + RUN),
        ]))
        n = sample.size
        with Timer() as t_read:
            packed, scales = wq.read_rows(sample), sq.read_rows(sample)
            fvals, fscales = wf.read_rows(sample), sf.read_rows(sample)
        g = scales_g[layer]

        emb = make_embedding(n, "nvfp4", global_scale=g)
        load_rows(emb, packed, scales, "nvfp4")
        femb = make_embedding(n, "fp8")
        load_rows(femb, fvals, fscales, "fp8")

        # Every sampled row once, shuffled over [T, 24] heads; tail slots -1, plus some
        # out-of-range ids, which must come back as zero rows.
        gen = torch.Generator().manual_seed(layer)
        heads = emb.n_hash_cols
        tokens = -(-n // heads) + 3
        flat = torch.full((tokens * heads,), -1, dtype=torch.int32)
        flat[:n] = torch.randperm(n, generator=gen).to(torch.int32)
        flat[n + 1 : n + 6] = torch.tensor([n, n + 1, 2**31 - 1, n + 7, -5], dtype=torch.int32)
        buf = torch.full((tokens, 2, heads), -1, dtype=torch.int32)
        buf[:, 1] = flat.view(tokens, heads)
        ids = buf.cuda()[:, 1]  # strided like gathered_hashes[:, layer_hash_index]

        ref = ref_nvfp4(packed, scales, g)
        want = expected_rows(ref, ids)
        got = emb(ids).cpu()
        got_bg = torch.empty((tokens, heads, 256), dtype=torch.bfloat16, device="cuda")
        emb.lookup(ids, got_bg, background=True)
        got_bg = got_bg.cpu()
        torch.cuda.synchronize()
        diff = (got.float() - want.float()).abs()
        nv_bits = bits_equal(got, want)
        nv_bits_bg = bits_equal(got_bg, want)
        nv_equal = torch.equal(got, want)

        # The HBM-resident NVFP4 table (cpu_offload=False) must decode identically.
        hemb = make_embedding(n, "nvfp4", cpu_offload=False, global_scale=g)
        load_rows(hemb, packed, scales, "nvfp4")
        hbm_bits = bits_equal(hemb(ids).cpu(), want) and hemb.weight.is_cuda
        del hemb

        fref = ref_fp8(fvals, fscales)
        fwant = expected_rows(fref, ids)
        fgot = femb(ids).cpu()
        fp8_bits = bits_equal(fgot, fwant)

        # NVFP4 vs FP8 on the same rows (README metric: ||deq - fp8|| / ||fp8||).
        a32 = ref_nvfp4_fp32(packed, scales, g)
        b32 = ref_fp8_fp32(fvals, fscales)
        row_err = (torch.linalg.vector_norm((a32 - b32).double(), dim=1)
                   / torch.linalg.vector_norm(b32.double(), dim=1).clamp_min(1e-30))
        cos = torch.nn.functional.cosine_similarity(a32.double(), b32.double(), dim=1)
        # Nibble-order and global-scale sanity: the swapped decode / missing global scale
        # must be far worse than the README's ~9%.
        swapped = ((packed & 0x0F) << 4) | (packed >> 4)
        err_swapped = rel_rms(ref_nvfp4_fp32(swapped, scales, g), b32)
        err_no_global = rel_rms(ref_nvfp4_fp32(packed, scales, 1.0), b32)
        # Same metric through the bf16 outputs the model consumes.
        err_bf16 = rel_rms(ref.float(), fref.float())

        lay = {
            "rows_in_table": total,
            "sampled_rows": n,
            "read_seconds": round(t_read.s, 2),
            "global_scale": g,
            "nvfp4_gather_vs_reference": {
                "elements": int(want.numel()),
                "max_abs_diff": float(diff.max()),
                "bit_identical_forward": nv_bits,
                "bit_identical_background_grid": nv_bits_bg,
                "bit_identical_hbm_resident_table": hbm_bits,
                "torch_equal": nv_equal,
                "zero_rows_for_invalid_ids": bool((got[ids.cpu() < 0] == 0).all()),
                "negative_zero_outputs": int((got.view(torch.int16) == -32768).sum()),
            },
            "fp8_gather_vs_reference": {"bit_identical": fp8_bits},
            "nvfp4_vs_fp8": {
                "rel_rms_fp32": rel_rms(a32, b32),
                "rel_rms_bf16_outputs": err_bf16,
                "row_rel_err_median": float(row_err.median()),
                "row_rel_err_p99": float(row_err.quantile(0.99)),
                "row_cosine_median": float(cos.median()),
                "row_cosine_min": float(cos.min()),
                "rel_rms_if_nibbles_swapped": err_swapped,
                "rel_rms_without_global_scale": err_no_global,
            },
            "host_table_bytes": {
                "nvfp4": emb.weight.nbytes + emb.weight_scale_inv.nbytes,
                "fp8": femb.weight.nbytes + femb.weight_scale_inv.nbytes,
                "nvfp4_storage": "registered" if emb._registered is not None else "pinned",
            },
        }
        report["layers"][layer] = lay
        ok &= nv_bits and nv_bits_bg and hbm_bits and fp8_bits
        ok &= lay["nvfp4_gather_vs_reference"]["zero_rows_for_invalid_ids"]
        ok &= 0.07 < lay["nvfp4_vs_fp8"]["rel_rms_fp32"] < 0.11
        print(f"layer {layer}: {n} rows, read {t_read.s:.1f}s; NVFP4 bit-identical {nv_bits}/{nv_bits_bg} "
              f"(max abs diff {float(diff.max()):.3g}); FP8 bit-identical {fp8_bits}; "
              f"rel RMS vs FP8 {lay['nvfp4_vs_fp8']['rel_rms_fp32']:.4%} "
              f"(swapped nibbles {err_swapped:.1%}, no global {err_no_global:.1%})", flush=True)
        del emb, femb
    # load_format=dummy fills host tables with their dummy_weight_value: NVFP4 rows must
    # then decode to e2m1 1.0 x scale 1.0 x global.
    from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight

    demb = make_embedding(4096, "nvfp4", global_scale=0.5)
    for param in demb.parameters():
        initialize_single_dummy_weight(param)
    ids = torch.arange(24, dtype=torch.int32, device="cuda").view(1, 24)
    dummy = demb(ids)
    report["dummy_weights_decode_to_global_scale"] = bool((dummy == 0.5).all())
    ok &= report["dummy_weights_decode_to_global_scale"]
    report["pass"] = bool(ok)
    print("wrote", write_result("dequant.json", report), "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
