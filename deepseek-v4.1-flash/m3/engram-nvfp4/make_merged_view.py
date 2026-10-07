#!/usr/bin/env python3
"""Build (or re-check) the merged DS41 checkpoint view with NVFP4 Engram tables.

<models>/DeepSeek-V4.1-Flash-engram-nvfp4-merged/ holds
  * relative symlinks to ../DeepSeek-V4.1-Flash/<x> for every original file except the
    two Engram shards, config.json and the index;
  * model-0004{7,8}-of-00048.safetensors -> ../DeepSeek-V4.1-Flash-engram-nvfp4/hf/<same name>
    (the NVFP4 shards replace the originals one-for-one: each holds the same six
    tensor names as the original shard; only embed.weight/embed.scale changed format);
  * config.json = original + text_config.engram_quant* (checked equal to the NVFP4 repo's);
  * model.safetensors.index.json = original weight_map, total_size recomputed.

vLLM loads every tensor of every shard the index lists, so the check below reads all 48
headers through the links and requires every tensor name exactly once and the union to
equal the weight_map. Run with --check to only verify an existing view.
"""
import argparse
import hashlib
import json
import os
import struct
import sys

MODELS = "/home/jasonc/models"  # -> /data/checkpoints
ORIG = "DeepSeek-V4.1-Flash"
NVFP4 = "DeepSeek-V4.1-Flash-engram-nvfp4/hf"
MERGED = "DeepSeek-V4.1-Flash-engram-nvfp4-merged"
ENGRAM_SHARDS = ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")
SKIP = {".cache", ".gitattributes", "config.json", "model.safetensors.index.json"} | set(
    ENGRAM_SHARDS
)
QUANT_KEYS = (
    "engram_quant",
    "engram_quant_block_size",
    "engram_quant_global_scale",
    "engram_quant_method",
)


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return n, h


def tensor_bytes(path, n, info):
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + n + start)
        return f.read(end - start)


def build(root):
    merged = os.path.join(root, MERGED)
    os.makedirs(merged, exist_ok=True)
    orig_dir = os.path.join(root, ORIG)
    for name in sorted(os.listdir(orig_dir)):
        if name in SKIP:
            continue
        link = os.path.join(merged, name)
        target = os.path.join("..", ORIG, name)
        if os.path.islink(link) and os.readlink(link) == target:
            continue
        os.symlink(target, link)
    for name in ENGRAM_SHARDS:
        link = os.path.join(merged, name)
        target = os.path.join("..", NVFP4, name)
        if not (os.path.islink(link) and os.readlink(link) == target):
            os.symlink(target, link)
    for src, dst in (
        ("engram-nvfp4.json", "engram-nvfp4.json"),
        ("README.md", "README-engram-nvfp4.md"),
    ):
        link = os.path.join(merged, dst)
        target = os.path.join("..", NVFP4, src)
        if not (os.path.islink(link) and os.readlink(link) == target):
            os.symlink(target, link)

    # config.json: original + engram_quant* from the NVFP4 repo; must equal the NVFP4 repo's.
    with open(os.path.join(orig_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(root, NVFP4, "config.json")) as f:
        nv_cfg = json.load(f)
    for key in QUANT_KEYS:
        cfg["text_config"][key] = nv_cfg["text_config"][key]
    assert cfg == nv_cfg, "NVFP4 config.json differs from original beyond engram_quant*"
    with open(os.path.join(merged, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    with open(os.path.join(orig_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    total = 0
    for shard in sorted(set(index["weight_map"].values())):
        _, h = header(os.path.join(merged, shard))
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for v in h.values())
    index["metadata"] = dict(index.get("metadata", {}), total_size=total)
    with open(os.path.join(merged, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")


def check(root):
    merged = os.path.join(root, MERGED)
    out = {}
    with open(os.path.join(merged, "model.safetensors.index.json")) as f:
        index = json.load(f)
    wm = index["weight_map"]
    shards = sorted(set(wm.values()))
    seen = {}
    dtypes = {}
    total = 0
    for shard in shards:
        path = os.path.join(merged, shard)
        assert os.path.isfile(path), f"dangling link {path}"
        _, h = header(path)
        for name, info in h.items():
            assert name not in seen, f"{name} in both {seen[name]} and {shard}"
            seen[name] = shard
            total += info["data_offsets"][1] - info["data_offsets"][0]
            if ".engram.embed." in name:
                dtypes[name] = (info["dtype"], info["shape"])
    assert set(seen) == set(wm), (
        f"index/shards mismatch: {len(set(seen) - set(wm))} unindexed, "
        f"{len(set(wm) - set(seen))} missing"
    )
    assert all(seen[k] == v for k, v in wm.items()), "tensor indexed under the wrong shard"
    assert index["metadata"]["total_size"] == total
    # Every *.safetensors file in the dir is indexed (vLLM globs then filters by index).
    files = sorted(n for n in os.listdir(merged) if n.endswith(".safetensors"))
    assert files == shards, (files, shards)
    out["shards"] = len(shards)
    out["tensors"] = len(seen)
    out["total_size"] = total
    out["engram_tables"] = dtypes
    out["links"] = {
        n: os.readlink(os.path.join(merged, n))
        for n in sorted(os.listdir(merged))
        if os.path.islink(os.path.join(merged, n))
    }
    with open(os.path.join(merged, "config.json")) as f:
        tc = json.load(f)["text_config"]
    out["config_quant"] = {k: tc[k] for k in QUANT_KEYS}

    # The NVFP4 shards copy the other engram tensors verbatim: compare bytes.
    same = {}
    for shard in ENGRAM_SHARDS:
        po = os.path.join(root, ORIG, shard)
        pn = os.path.join(merged, shard)
        no, ho = header(po)
        nn_, hn = header(pn)
        assert set(ho) == set(hn), (shard, set(ho) ^ set(hn))
        for name in sorted(ho):
            if ".engram.embed." in name:
                continue
            a = tensor_bytes(po, no, ho[name])
            b = tensor_bytes(pn, nn_, hn[name])
            same[name] = {
                "dtype": hn[name]["dtype"],
                "shape": hn[name]["shape"],
                "identical": a == b,
                "sha256": hashlib.sha256(b).hexdigest()[:16],
            }
            assert a == b, f"{name} differs between original and NVFP4 shard"
    out["non_table_tensors_in_engram_shards"] = same
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=MODELS)
    ap.add_argument("--check", action="store_true", help="verify only")
    ap.add_argument("--out", help="write the check report (JSON) here")
    args = ap.parse_args()
    if not args.check:
        build(args.root)
    report = check(args.root)
    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "links"}, indent=1))
    print(f"{len(report['links'])} symlinks; OK")


if __name__ == "__main__":
    sys.exit(main())
