"""Phase 0: how well do MegaMoE's weight scales compress, in the exact chunks the kernel loads?

MegaMoE's SFB loader TMA-loads one 512 B chunk per (expert, 128-row N-block, 128-wide k-block): 128 rows x 4
UE8M0 K-groups, after gate/up interleave (L1, groups of 8) and the 4x32 UTCCP row transpose. This script rebuilds
those chunks from the checkpoint for the hot experts in a rowmap and scores candidate codecs per chunk.

  python sf_compress_stats.py --model ~/models/DeepSeek-V4.1-Flash --rowmap ../hook/rowmap-mix-h258.json

Codecs (bytes per 512 B chunk; exceptions cost 4 B each = 9-bit slot + 8-bit value, padded):
  b1     chunk base (1 B) + 1 bit/scale (64 B) + exceptions for values outside {base, base+1}
  b2     chunk base (1 B) + 2 bits/scale (128 B) + exceptions outside [base, base+3]
  rowb1  per-row base shared across all k-blocks (amortized) + 1 bit/scale + exceptions (LIL CSF style)
Each variable-size codec also gets a raw fallback (512 B) when it would exceed it. "slotN" is the fixed-stride
variant a GPU decoder wants: a 64 B bit plane + base + up to N inline exceptions in one fixed slot, and chunks with
more than N exceptions spill to a raw 512 B tile in an overflow area.
"""
import argparse
import json
import os
import struct
from concurrent.futures import ProcessPoolExecutor

import numpy as np

I, H = 2304, 5120


def reader(model):
    index = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    headers = {}

    def get(name):
        path = os.path.join(model, index[name])
        if path not in headers:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                headers[path] = (8 + n, json.loads(f.read(n)))
        base, hdr = headers[path]
        meta = hdr[name]
        s, e = meta["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + s)
            return np.frombuffer(f.read(e - s), dtype=np.uint8).reshape(meta["shape"])

    return get


def chunks(sf):
    """[E, N, K] uint8 scales -> [E, N/128, K/4, 128, 4] chunks in the kernel's row order (4x32 transpose)."""
    E, N, K = sf.shape
    assert N % 128 == 0 and K % 4 == 0
    t = sf.reshape(E, N // 128, 4, 32, K // 4, 4).transpose(0, 1, 4, 3, 2, 5)  # rows j*32+l -> l*4+j
    return t.reshape(E, N // 128, K // 4, 128, 4)


def interleave(w13):
    """[E, 2I, K]: [gate | up] -> gate 0..7, up 0..7, gate 8..15, ... (MegaMoE L1 layout)."""
    E, n, K = w13.shape
    g = w13[:, : n // 2].reshape(E, n // 16, 8, K)
    u = w13[:, n // 2 :].reshape(E, n // 16, 8, K)
    return np.stack([g, u], axis=2).reshape(E, n, K)


def score(c, row_base_global):
    """c: [..., 128, 4] uint8 chunks. Returns dict of per-chunk byte costs and exception counts."""
    flat = c.reshape(-1, 512).astype(np.int16)
    n = flat.shape[0]
    out = {}
    # b1/b2: pick the base that minimizes exceptions (try min and the most common value and value-1).
    lo = flat.min(axis=1)
    best1 = np.full(n, 513)
    best2 = np.full(n, 513)
    cands = [lo]
    mode = np.array([np.bincount(r).argmax() for r in flat])
    cands += [mode, mode - 1, mode - 2]
    for b in cands:
        d = flat - b[:, None]
        best1 = np.minimum(best1, ((d < 0) | (d > 1)).sum(axis=1))
        best2 = np.minimum(best2, ((d < 0) | (d > 3)).sum(axis=1))
    out["exc_b1"] = best1
    out["exc_b2"] = best2
    out["b1"] = np.minimum(65 + 4 * best1, 512)
    out["b2"] = np.minimum(129 + 4 * best2, 512)
    # rowb1: base per row fixed across all K (rows of this matrix), passed in broadcast to chunk rows.
    rb = row_base_global.reshape(-1, 128)  # aligned with flat chunk order (row-major per chunk)
    d = c.reshape(-1, 128, 4).astype(np.int16) - rb[:, :, None]
    exc_r = ((d < 0) | (d > 1)).reshape(n, -1).sum(axis=1)
    out["exc_rowb1"] = exc_r
    out["rowb1"] = np.minimum(64 + 4 * exc_r, 512)  # row bases accounted separately (amortized)
    return out


def layer_stats(args):
    model, layer, hot = args
    get = reader(model)
    w13 = np.stack([np.concatenate([get(f"layers.{layer}.ffn.experts.{e}.w1.scale"),
                                    get(f"layers.{layer}.ffn.experts.{e}.w3.scale")]) for e in hot])
    w2 = np.stack([get(f"layers.{layer}.ffn.experts.{e}.w2.scale") for e in hot])
    res = {}
    for tag, sf in (("L1", interleave(w13)), ("L2", w2)):
        E, N, K = sf.shape
        # LIL-style per-row base: the most common value v such that v/v+1 covers most of the row.
        s16 = sf.astype(np.int16)
        best_cnt = np.full((E, N), -1)
        row_base = np.zeros((E, N), np.int16)
        for off in range(-3, 1):
            b = np.median(s16, axis=2).astype(np.int16) + off
            cnt = ((s16 == b[..., None]) | (s16 == b[..., None] + 1)).sum(axis=2)
            better = cnt > best_cnt
            row_base = np.where(better, b, row_base)
            best_cnt = np.maximum(best_cnt, cnt)
        c = chunks(sf)  # [E, NB, KB, 128, 4]
        rb = chunks(np.repeat(row_base[..., None], K, axis=2).astype(np.uint8))[..., 0].astype(np.int16)
        sc = score(c, rb)
        res[tag] = {k: v.astype(np.int32) for k, v in sc.items()}
        res[tag]["n_rows"] = E * N
    return layer, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.path.expanduser("~/models/DeepSeek-V4.1-Flash"))
    p.add_argument("--rowmap", required=True)
    p.add_argument("--layers", type=int, default=40)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--out", default="phase0-results.json")
    a = p.parse_args()
    rowmap = json.load(open(a.rowmap))["layers"]
    jobs = [(a.model, l, rowmap[str(l)]["hot"]) for l in range(a.layers)]
    agg = {t: {} for t in ("L1", "L2")}
    per_layer = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for layer, res in ex.map(layer_stats, jobs):
            per_layer[layer] = {}
            for t, d in res.items():
                for k, v in d.items():
                    if k == "n_rows":
                        agg[t]["n_rows"] = agg[t].get("n_rows", 0) + v
                    else:
                        agg[t].setdefault(k, []).append(v)
                per_layer[layer][t] = {k: float(np.mean(d[k] < 512)) if k in ("b1", "b2", "rowb1") else
                                       float(np.mean(d[k])) for k in d if k != "n_rows"}
                per_layer[layer][t]["b1_bytes"] = int(d["b1"].sum())
            print(f"layer {layer:2d}: L1 b1 {res['L1']['b1'].sum() / 2**20:7.1f} MiB  L2 b1 "
                  f"{res['L2']['b1'].sum() / 2**20:6.1f} MiB  L2 raw-fallback {np.mean(res['L2']['b1'] >= 512):.1%}",
                  flush=True)

    summary = {}
    raw_total = 0
    for t in ("L1", "L2"):
        d = {k: np.concatenate(v) for k, v in agg[t].items() if k != "n_rows"}
        n_chunks = d["b1"].size
        raw = n_chunks * 512
        raw_total += raw
        s = {"chunks": int(n_chunks), "raw_bytes": int(raw)}
        for codec in ("b1", "b2", "rowb1"):
            extra = agg[t]["n_rows"] if codec == "rowb1" else 0  # one base byte per row, stored once
            s[codec] = {"bytes": int(d[codec].sum() + extra), "ratio": float((d[codec].sum() + extra) / raw),
                        "raw_fallback_chunks": float(np.mean(d[codec] >= 512))}
        for codec, key in (("b1", "exc_b1"), ("b2", "exc_b2"), ("rowb1", "exc_rowb1")):
            e = d[key]
            s[codec]["exceptions_mean"] = float(e.mean())
            s[codec]["exceptions_p99"] = float(np.percentile(e, 99))
            s[codec]["chunks_zero_exc"] = float(np.mean(e == 0))
            # fixed-slot variant: 64 B bits + 4 B header + N inline 4 B exceptions; spill -> +512 B raw tile
            for N in (3, 7, 15):
                slot = 64 + 4 + 4 * N if codec != "b2" else 128 + 4 + 4 * N
                spill = e > N
                s[codec][f"slot{N}"] = {"slot_bytes": slot, "spill_frac": float(spill.mean()),
                                        "bytes": int(n_chunks * slot + spill.sum() * 512),
                                        "ratio": float((n_chunks * slot + spill.sum() * 512) / raw)}
        summary[t] = s
    summary["raw_total_bytes"] = raw_total
    json.dump({"summary": summary, "per_layer": per_layer}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
