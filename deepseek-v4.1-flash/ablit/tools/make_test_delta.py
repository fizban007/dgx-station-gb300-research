#!/usr/bin/env python3
"""Synthesize a stand-in "abliterated" delta so the switch path can be tested without the gated repo.

This is NOT the real abliteration. It applies a deterministic rank-1 projection (norm-preserving, per row) to
`attn.wo_b` and `ffn.shared_experts.w2` and requantizes to MXFP8 32x32 blocks, which gives a delta with the same
shapes, the same two-tensor-per-layer structure and a realistic mix of changed and unchanged scale bytes.
`abliterated-delta.json` is marked `"synthetic": true`, and the hook says so in its log lines.

Needs torch but no GPU (run it under the sidecar venv, which has torch):

  CUDA_VISIBLE_DEVICES= ~/venvs/sidecar/bin/python tools/make_test_delta.py --limit 2
  CUDA_VISIBLE_DEVICES= ~/venvs/sidecar/bin/python tools/make_test_delta.py   # all 33 layers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

TENSOR_SUFFIXES = (
    "attn.wo_b.weight",
    "attn.wo_b.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.scale",
)
SCHEMA = "dsv41-abliterated-delta/1"
BLOCK = 32
E4M3_MAX = 448.0
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.normpath(os.path.join(HERE, "..", "data"))


def log(m: str) -> None:
    print(m, flush=True)


class CkptBytes:
    def __init__(self, root: str):
        import struct

        self.root = root
        self._struct = struct
        with open(os.path.join(root, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._h: dict[str, tuple[int, dict]] = {}

    def read(self, name: str) -> bytes:
        shard = self.index[name]
        if shard not in self._h:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = self._struct.unpack("<Q", f.read(8))[0]
                self._h[shard] = (8 + n, json.loads(f.read(n)))
        base, hdr = self._h[shard]
        start, end = hdr[name]["data_offsets"]
        with open(os.path.join(self.root, shard), "rb") as f:
            f.seek(base + start)
            return f.read(end - start)

    def meta(self, name: str):
        shard = self.index[name]
        if shard not in self._h:
            with open(os.path.join(self.root, shard), "rb") as f:
                n = self._struct.unpack("<Q", f.read(8))[0]
                self._h[shard] = (8 + n, json.loads(f.read(n)))
        return self._h[shard][1][name]


def dequant(w_fp8: torch.Tensor, scale_u8: torch.Tensor) -> torch.Tensor:
    """MXFP8 32x32 -> fp32. Each scale byte is an e8m0 exponent: value = 2**(b-127)."""
    N, K = w_fp8.shape
    vals = torch.pow(2.0, scale_u8.to(torch.float32) - 127.0)
    big = vals.repeat_interleave(BLOCK, dim=0).repeat_interleave(BLOCK, dim=1)
    return w_fp8.to(torch.float32) * big[:N, :K]


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 -> MXFP8 32x32 with power-of-two (e8m0) block scales. Round-to-nearest, saturating."""
    N, K = w.shape
    assert N % BLOCK == 0 and K % BLOCK == 0, (N, K)
    blocks = w.reshape(N // BLOCK, BLOCK, K // BLOCK, BLOCK).permute(0, 2, 1, 3)
    amax = blocks.abs().amax(dim=(2, 3)).clamp_min(1e-30)
    exp = torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(-127, 127)
    scale_u8 = (exp + 127).to(torch.uint8)
    big = torch.pow(2.0, exp).repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[:N, :K]
    q = (w / big).to(torch.float8_e4m3fn)
    return q, scale_u8


def project(w: torch.Tensor, r: torch.Tensor, strength: float) -> torch.Tensor:
    """Directional ablation, then per-row norm preservation (the card's 'norm-preserving biprojected' idea)."""
    norms = w.norm(dim=1, keepdim=True)
    out = w - strength * torch.outer(r, r @ w)
    return out * (norms / out.norm(dim=1, keepdim=True).clamp_min(1e-12))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--official-root", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    p.add_argument("--layers", default="4-36")
    p.add_argument("--limit", type=int, default=0, help="only the first N tensors (2 = one layer's wo_b pair)")
    p.add_argument("--strength", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=20261008)
    a = p.parse_args()

    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    os.makedirs(a.out_dir, exist_ok=True)
    ckpt = CkptBytes(a.official_root)
    names = [f"layers.{i}.{s}" for i in layers for s in TENSOR_SUFFIXES]
    if a.limit:
        names = names[: a.limit]

    weights = {}   # logical weight name -> (new fp8 bytes, new scale bytes)
    order = []
    for i, name in enumerate(names):
        if not name.endswith(".weight"):
            continue
        layer = int(name.split(".")[1])
        m = ckpt.meta(name)
        shape = tuple(m["shape"])
        w = torch.frombuffer(bytearray(ckpt.read(name)), dtype=torch.uint8).view(
            torch.float8_e4m3fn).reshape(shape)
        s_name = name[: -len(".weight")] + ".scale"
        sm = ckpt.meta(s_name)
        scale = torch.frombuffer(bytearray(ckpt.read(s_name)), dtype=torch.uint8).reshape(tuple(sm["shape"]))
        gen = torch.Generator().manual_seed(a.seed + layer)
        r = torch.randn(shape[0], generator=gen)
        r = (r / r.norm()).to(torch.float32)
        w32 = dequant(w, scale)
        new = project(w32, r, a.strength)
        q, new_scale = quantize(new.contiguous())
        err = (dequant(q, new_scale) - new).abs().max().item()
        wb = q.view(torch.uint8).numpy().tobytes()
        sb = new_scale.numpy().tobytes()
        weights[name] = (wb, sb)
        order.extend([name, s_name])
        log(f"  {name}: {shape} changed scale bytes "
            f"{int((new_scale != scale).sum())}/{scale.numel()}, requant max err {err:.4f}")

    tensors = []
    changed = 0
    with open(os.path.join(a.out_dir, "abliterated-delta.bin"), "wb") as fb:
        for name in order:
            base, suffix = name.rsplit(".", 1)
            data = weights[base + ".weight"][0 if suffix == "weight" else 1]
            m = ckpt.meta(name)
            same = data == ckpt.read(name)
            changed += 0 if same else 1
            tensors.append({
                "name": name,
                "dtype": m["dtype"],
                "shape": m["shape"],
                "nbytes": len(data),
                "bin_offset": fb.tell(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "changed": not same,
                "remote_shard": None,
                "remote_offset": None,
            })
            fb.write(data)

    delta = {
        "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "synthetic": True,
        "method": "rank-1 projection with per-row norm preservation, then MXFP8 32x32 requantization",
        "strength": a.strength,
        "seed": a.seed,
        "repo": None,
        "repo_sha": None,
        "official_root": a.official_root,
        "official_repo": "deepseek-ai/DeepSeek-V4.1-Flash",
        "layers": [lo, hi],
        "tensor_suffixes": list(TENSOR_SUFFIXES),
        "totals": {"tensors": len(tensors), "bytes": sum(t["nbytes"] for t in tensors),
                   "changed": changed, "identical": len(tensors) - changed,
                   "offset_mismatches_vs_local": 0},
        "tensors": tensors,
    }
    with open(os.path.join(a.out_dir, "abliterated-delta.json"), "w") as f:
        json.dump(delta, f, indent=1)
    report = {"schema": SCHEMA + "+report", "synthetic": True, "changed": changed,
              "identical": len(tensors) - changed, "spot_checks": [], "spot_skipped": [],
              "all_spot_checks_identical": None}
    with open(os.path.join(a.out_dir, "delta-report.json"), "w") as f:
        json.dump(report, f, indent=1)
    log(f"synthetic delta: {len(tensors)} tensors, {delta['totals']['bytes'] / 2**20:.1f} MiB, "
        f"{changed} changed -> {a.out_dir}")
    log("this is a test fixture, not the real abliteration; the hook marks it SYNTHETIC in its logs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
