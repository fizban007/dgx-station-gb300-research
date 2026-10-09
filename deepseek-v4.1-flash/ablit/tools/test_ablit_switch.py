#!/usr/bin/env python3
"""CPU test for ../ablit_switch.py: layout discovery, orientation handling and both apply directions.

No GPU and no vLLM needed. The candidate-discovery and apply code is exercised against resident tensors built
the way the real loader builds them, using the official checkpoint bytes plus the synthetic delta produced by
tools/make_test_delta.py. The one recipe this cannot cover is the fused shared expert's DeepGEMM scale, which
needs the live model; the hook's boot self-test covers that.

  cd <this directory>
  CUDA_VISIBLE_DEVICES= ~/venvs/sidecar/bin/python tools/test_ablit_switch.py [--data-dir ...] [--vllm]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SWITCH = os.path.join(HERE, "..", "ablit_switch.py")
DATA = os.path.normpath(os.path.join(HERE, "..", "data"))
FAILURES: list[str] = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int, block: int = 32) -> torch.Tensor:
    """Faithful copy of vllm.model_executor.layers.quantization.utils.mxfp8_utils.swizzle_mxfp8_scale."""
    factor = block * 4
    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor
    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4
    scale_cols = K // block
    sf_padded = torch.zeros((m_padded, k_scale_padded), dtype=sf.dtype)
    sf_padded[:M, :scale_cols] = sf
    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    return sf_reshaped.transpose(1, 3).contiguous().view(-1)


def load_switch(path):
    spec = importlib.util.spec_from_file_location("ablit_switch", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["ablit_switch"] = m
    spec.loader.exec_module(m)
    return m


class Ckpt:
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._h = {}

    def read(self, name):
        shard = self.index[name]
        if shard not in self._h:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._h[shard] = (8 + n, json.loads(f.read(n)))
        base, hdr = self._h[shard]
        start, end = hdr[name]["data_offsets"]
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(base + start)
            return f.read(end - start)


class FakeLinear(torch.nn.Module):
    # The loader sets this for block-quantized linears; the model's scale helper gates on it.
    weight_block_size = [1, 32]

    def __init__(self, weight, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--delta-dir", default=DATA, help="where the synthetic delta lives (make_test_delta.py)")
    p.add_argument("--official-root", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    p.add_argument("--vllm", action="store_true", help="use vllm's own swizzle instead of the embedded copy")
    a = p.parse_args()

    ablit = load_switch(SWITCH)
    ablit.MODEL_ROOT = a.official_root
    ablit.DELTA_BIN = os.path.join(a.delta_dir, "abliterated-delta.bin")
    with open(os.path.join(a.delta_dir, "abliterated-delta.json")) as f:
        ablit._delta = json.load(f)
    if a.vllm:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import swizzle_mxfp8_scale as real

        ablit._swizzle = lambda: real
        print("using vllm's swizzle_mxfp8_scale")
    else:
        ablit._swizzle = lambda: swizzle_mxfp8_scale
    ablit._deep_gemm = lambda: None   # the fake fused-shared helper below ignores it

    ckpt = Ckpt(a.official_root)
    with open(os.path.join(a.delta_dir, "abliterated-delta.json")) as f:
        FULL = json.load(f)

    def use(pred):
        """Point the module at the subset of the delta under test (offsets stay valid)."""
        ablit._delta = dict(FULL, tensors=[t for t in FULL["tensors"] if pred(t["name"])])

    entries = {t["name"]: t for t in FULL["tensors"]}
    wo_b = [n for n in entries if ".attn.wo_b." in n]
    shared = [n for n in entries if "shared_experts" in n]
    check(len(wo_b) >= 2 and len(shared) >= 2,
          f"delta has wo_b and shared tensors ({len(entries)} total, {len(wo_b)} wo_b, {len(shared)} shared)")
    if len(wo_b) < 2 or len(shared) < 2:
        return 1
    # One pair of each family is enough: the rest of the delta is verified by the extractor and at boot.
    layer = int(wo_b[0].split(".")[1])
    wname = f"layers.{layer}.attn.wo_b.weight"
    sname = f"layers.{layer}.attn.wo_b.scale"
    sw_name = [n for n in shared if n.endswith(".weight")][0]
    ss_name = sw_name.rsplit(".", 1)[0] + ".scale"
    check(wname in entries and sname in entries and sw_name in entries and ss_name in entries,
          f"testing layers.{layer} wo_b and shared pairs")
    N, K = entries[wname]["shape"]

    def build_wo_b(orient="NK", scale_form="swizzle"):
        w_u8 = torch.frombuffer(bytearray(ckpt.read(wname)), dtype=torch.uint8)
        w = w_u8.view(torch.float8_e4m3fn).reshape(N, K)
        if orient == "KN":
            w = w.t()
        s_u8 = torch.frombuffer(bytearray(ckpt.read(sname)), dtype=torch.uint8).reshape(N // 32, K // 32)
        if scale_form == "swizzle":
            scale = swizzle_mxfp8_scale(s_u8.repeat_interleave(32, dim=0), M=N, K=K)
        elif scale_form == "raw_i32":
            scale = s_u8.repeat_interleave(32, dim=0).to(torch.int32)
        else:
            raise ValueError(scale_form)
        return FakeLinear(w, scale), w_u8

    print("layout discovery")
    use(lambda n: n in (wname, sname))
    lin, w_u8 = build_wo_b()
    ablit._linears = {(layer, "wo_b"): lin}
    ablit._moe, ablit._shared = {}, {}
    targets = ablit._build_targets(ablit._Ckpt(a.official_root))
    by_name = {t["name"]: t for t in targets}
    check(len(targets) == 2 and all(t["dsts"] for t in targets), "both wo_b tensors verified against the checkpoint")
    check("swizzle_u8/expand32" in (by_name[sname]["dsts"][0]["recipe"][2] if by_name[sname]["dsts"] else ""),
          f"scale convention discovered: {by_name[sname]['dsts'][0]['recipe'][2] if by_name[sname]['dsts'] else by_name[sname]['rejected']}")
    check(by_name[wname]["dsts"][0]["orient"] == "NK", "weight orientation NK discovered")

    print("alternate convention (block rows expanded and widened to int32)")
    use(lambda n: n in (wname, sname))
    lin2, _ = build_wo_b(scale_form="raw_i32")
    ablit._linears = {(layer, "wo_b"): lin2}
    t2 = {t["name"]: t for t in ablit._build_targets(ablit._Ckpt(a.official_root))}
    label = t2[sname]["dsts"][0]["recipe"][2] if t2[sname]["dsts"] else str(t2[sname]["rejected"])
    check(t2[sname]["dsts"] and "raw_i32" in label, f"int32 convention discovered instead: {label}")

    print("orientation KN")
    use(lambda n: n in (wname, sname))
    lin3, _ = build_wo_b(orient="KN")
    ablit._linears = {(layer, "wo_b"): lin3}
    t3 = {t["name"]: t for t in ablit._build_targets(ablit._Ckpt(a.official_root))}
    check(t3[wname]["dsts"] and t3[wname]["dsts"][0]["orient"] == "KN", "transposed resident weight discovered")

    print("apply abliterated, then official (NK resident)")
    use(lambda n: n in (wname, sname))
    lin, w_u8 = build_wo_b()
    ablit._linears = {(layer, "wo_b"): lin}
    ablit._targets = ablit._build_targets(ablit._Ckpt(a.official_root))
    ablit._apply("abliterated", 1)
    d_u8 = torch.frombuffer(bytearray(open(ablit.DELTA_BIN, "rb").read()), dtype=torch.uint8)
    want_w = d_u8[entries[wname]["bin_offset"]:entries[wname]["bin_offset"] + entries[wname]["nbytes"]]
    check(torch.equal(ablit._bytes_of(lin.weight), want_w), "resident weight now holds the delta bytes")
    want_s = d_u8[entries[sname]["bin_offset"]:entries[sname]["bin_offset"] + entries[sname]["nbytes"]]
    rebuilt = swizzle_mxfp8_scale(want_s.reshape(N // 32, K // 32).repeat_interleave(32, dim=0), M=N, K=K)
    check(torch.equal(ablit._bytes_of(lin.weight_scale), rebuilt), "resident scale rebuilt from the delta")
    check(ablit.STATE["applied_mode"] == "abliterated" and ablit.STATE["applied_seq"] == 1, "state records the switch")
    ablit._apply("official", 2)
    official_w = torch.frombuffer(bytearray(ckpt.read(wname)), dtype=torch.uint8)
    check(torch.equal(ablit._bytes_of(lin.weight), official_w), "switching back restores the official weight bytes")
    official_s = swizzle_mxfp8_scale(
        torch.frombuffer(bytearray(ckpt.read(sname)), dtype=torch.uint8).reshape(N // 32, K // 32)
        .repeat_interleave(32, dim=0), M=N, K=K)
    check(torch.equal(ablit._bytes_of(lin.weight_scale), official_s), "switching back restores the official scale")

    print("apply with a transposed resident (KN)")
    use(lambda n: n in (wname, sname))
    lin3, _ = build_wo_b(orient="KN")
    ablit._linears = {(layer, "wo_b"): lin3}
    ablit._targets = ablit._build_targets(ablit._Ckpt(a.official_root))
    ablit._apply("abliterated", 3)
    check(torch.equal(ablit._bytes_of(lin3.weight), want_w.view(N, K).t().contiguous().view(-1)),
          "transposed resident receives the delta bytes in its own order")
    ablit._apply("official", 4)
    check(torch.equal(ablit._bytes_of(lin3.weight), official_w.view(N, K).t().contiguous().view(-1)),
          "transposed resident restored")

    print("negative: a resident that does not match the checkpoint is rejected")
    use(lambda n: n in (wname, sname))
    lin4, _ = build_wo_b()
    bad = lin4.weight.detach().clone()
    bad.view(-1)[12345] = (int(bad.view(-1)[12345]) ^ 0xFF)
    ablit._linears = {(layer, "wo_b"): FakeLinear(bad, lin4.weight_scale)}
    t4 = {t["name"]: t for t in ablit._build_targets(ablit._Ckpt(a.official_root))}
    check(not t4[wname]["dsts"], f"corrupted resident rejected: {t4[wname]['rejected']}")
    check(t4[sname]["dsts"], "the scale is still verified when only the weight is corrupt")

    print("negative: a source/target size mismatch aborts the apply without writing")
    use(lambda n: n in (wname, sname))
    lin5, _ = build_wo_b()
    ablit._linears = {(layer, "wo_b"): lin5}
    ablit._targets = ablit._build_targets(ablit._Ckpt(a.official_root))
    before = ablit._bytes_of(lin5.weight).clone()
    index = next(t for t in ablit._targets if t["name"] == wname)
    index["nbytes"] += 1
    try:
        ablit._apply("abliterated", 5)
        check(False, "apply should have refused")
    except Exception as e:
        check("expected" in str(e), f"apply refused: {e}")
    check(torch.equal(ablit._bytes_of(lin5.weight), before), "nothing was written")

    print("shared expert (linear scale convention; the fused DeepGEMM scale needs the live model)")
    use(lambda n: n in (sw_name, ss_name))
    sN, sK = entries[sw_name]["shape"]
    s_w = torch.frombuffer(bytearray(ckpt.read(sw_name)), dtype=torch.uint8).view(
        torch.float8_e4m3fn).reshape(sN, sK)
    s_scale = torch.frombuffer(bytearray(ckpt.read(ss_name)), dtype=torch.uint8).reshape(
        sN // 32, sK // 32).repeat_interleave(32, dim=0).view(torch.float8_e8m0fnu)
    down = FakeLinear(s_w, s_scale)
    shared_mlp = torch.nn.Module()
    shared_mlp.down_proj = down
    ablit._linears = {(layer, "wo_b"): lin}
    ablit._moe = {layer: torch.nn.Module()}          # no _transformed_shared_l2_weights: fusion off
    ablit._shared = {layer: shared_mlp}
    t5 = {t["name"]: t for t in ablit._build_targets(ablit._Ckpt(a.official_root))}
    check(t5[sw_name]["dsts"] and t5[sw_name]["dsts"][0]["orient"] == "NK", "shared weight verified (NK)")
    check(t5[ss_name]["dsts"] and "expand32" in t5[ss_name]["dsts"][0]["recipe"][2],
          f"linear scale convention discovered: "
          f"{t5[ss_name]['dsts'][0]['recipe'][2] if t5[ss_name]['dsts'] else t5[ss_name]['rejected']}")
    ablit._targets = ablit._build_targets(ablit._Ckpt(a.official_root))
    ablit._apply("abliterated", 6)
    check(torch.equal(ablit._bytes_of(down.weight),
                      d_u8[entries[sw_name]["bin_offset"]:entries[sw_name]["bin_offset"] + entries[sw_name]["nbytes"]]),
          "shared weight holds the delta bytes")
    want_scale = (d_u8[entries[ss_name]["bin_offset"]:entries[ss_name]["bin_offset"] + entries[ss_name]["nbytes"]]
                  .view(torch.float8_e8m0fnu).reshape(sN // 32, sK // 32).repeat_interleave(32, dim=0))
    check(torch.equal(ablit._bytes_of(down.weight_scale), ablit._bytes_of(want_scale)),
          "shared linear scale holds the delta bytes (expanded x32)")
    ablit._apply("official", 7)
    check(torch.equal(ablit._bytes_of(down.weight), _bytes_of_src(ckpt, sw_name)), "shared weight restored")

    print("fused shared expert (live shapes: fused scale int32 (5120,18) packed, linear scale flat uint8)")
    use(lambda n: n in (sw_name, ss_name))
    import types

    def fake_transform(utccp):
        """Stands in for deep_gemm.transform_sf_into_required_layout: exponents -> packed bytes -> int32 words."""

        def fn(sf, mn, k, group_shape, num_groups):
            words = ((sf.view(torch.int32) >> 23) & 0xFF).to(torch.uint8).view(torch.int32)
            return ablit._utccp_transpose(words) if utccp else words

        return fn

    class FakeMoe(torch.nn.Module):
        """Stands in for DeepseekV4MegaMoEExperts: the fused pair plus the model's own scale helper."""

        def __init__(self, tsw):
            super().__init__()
            self._transformed_shared_l2_weights = tsw

        def _prepare_shared_expert_scale(self, deep_gemm, linear, scale, mn, k):
            # The model helper, including its shape gate and its transform call.
            block_m, block_k = linear.weight_block_size
            expected = ((mn + block_m - 1) // block_m, (k + block_k - 1) // block_k)
            if scale.dim() != 2 or tuple(scale.shape) != expected:
                return None
            sf = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
            return deep_gemm.transform_sf_into_required_layout(
                sf.unsqueeze(0), mn, k, (1, 32), 1).squeeze(0)

    ck_grid_u8 = torch.frombuffer(bytearray(ckpt.read(ss_name)), dtype=torch.uint8).reshape(sN // 32, sK // 32)
    flat_u8 = ck_grid_u8.repeat_interleave(32, dim=0).reshape(-1).clone()     # (368640,) uint8, as live
    for utccp in (False, True):
        ablit._deep_gemm = lambda utccp=utccp: types.SimpleNamespace(
            transform_sf_into_required_layout=fake_transform(utccp))
        fused_w = torch.frombuffer(bytearray(ckpt.read(sw_name)), dtype=torch.uint8).view(
            torch.float8_e4m3fn).reshape(sN, sK).clone()
        down2 = FakeLinear(torch.frombuffer(bytearray(ckpt.read(sw_name)), dtype=torch.uint8).view(
            torch.float8_e4m3fn).reshape(sN, sK), flat_u8.clone())
        fused_scale = FakeMoe(None)._prepare_shared_expert_scale(
            ablit._deep_gemm(), down2, ck_grid_u8.repeat_interleave(32, dim=0), sN, sK)
        check(tuple(fused_scale.shape) == (sN, sK // 32 // 4) and fused_scale.dtype == torch.int32,
              f"fused scale built as int32 {tuple(fused_scale.shape)} {fused_scale.dtype} (utccp={utccp})")
        moe = FakeMoe((fused_w, fused_scale))
        mlp2 = torch.nn.Module()
        mlp2.down_proj = down2
        ablit._linears = {(layer, "wo_b"): lin}
        ablit._moe = {layer: moe}
        ablit._shared = {layer: mlp2}
        t6 = {t["name"]: t for t in ablit._build_targets(ablit._Ckpt(a.official_root))}
        check(t6[sw_name]["dsts"] and not t6[sw_name]["rejected"],
              f"fused shared weight verified on both storages (utccp={utccp}): "
              f"{t6[sw_name]['rejected'] or 'no rejections'}")
        recipes = [d["recipe"][:1] + d["recipe"][-1:] for d in t6[ss_name]["dsts"] if d["recipe"]]
        check(any(r[0] == "deepep" for r in recipes) and any(r[0] == "linear" for r in recipes),
              f"both scale destinations verified (utccp={utccp}): {recipes}")
        check(not t6[ss_name]["rejected"], f"no failure (utccp={utccp}): {t6[ss_name]['rejected'] or 'clean'}")
        ablit._targets = ablit._build_targets(ablit._Ckpt(a.official_root))
        ablit._apply("abliterated", 8)
        want_fw = d_u8[entries[sw_name]["bin_offset"]:entries[sw_name]["bin_offset"] + entries[sw_name]["nbytes"]]
        want_grid = (d_u8[entries[ss_name]["bin_offset"]:entries[ss_name]["bin_offset"]
                          + entries[ss_name]["nbytes"]].view(torch.uint8).reshape(sN // 32, sK // 32))
        want_packed = fake_transform(utccp)(
            (want_grid.repeat_interleave(32, dim=0).view(torch.uint8).to(torch.int32) << 23)
            .view(torch.float32).unsqueeze(0), sN, sK, (1, 32), 1).squeeze(0)
        check(torch.equal(ablit._bytes_of(fused_w), want_fw), f"fused weight holds the delta (utccp={utccp})")
        check(torch.equal(ablit._bytes_of(down2.weight), want_fw), f"linear weight copy too (utccp={utccp})")
        check(torch.equal(ablit._bytes_of(fused_scale), ablit._bytes_of(want_packed)),
              f"fused scale rebuilt from the delta (utccp={utccp})")
        check(torch.equal(ablit._bytes_of(down2.weight_scale),
                          ablit._bytes_of(want_grid.repeat_interleave(32, dim=0).reshape(-1))),
              f"flat linear scale rebuilt (utccp={utccp})")
        ablit._apply("official", 9)
        check(torch.equal(ablit._bytes_of(fused_w), _bytes_of_src(ckpt, sw_name)),
              f"fused weight restored (utccp={utccp})")
        check(torch.equal(ablit._bytes_of(down2.weight_scale), ablit._bytes_of(flat_u8)),
              f"flat linear scale restored (utccp={utccp})")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


def _bytes_of_src(ckpt, name):
    return torch.frombuffer(bytearray(ckpt.read(name)), dtype=torch.uint8)


if __name__ == "__main__":
    sys.exit(main())
