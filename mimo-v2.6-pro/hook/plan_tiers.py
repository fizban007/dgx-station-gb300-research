"""Three-tier MiMo-V2.6-Pro expert placement: HBM (GB300) > peer (RTX PRO 6000 sidecar) > Grace (UVA).

Ranks every (layer, expert) cell by its normalised decode routing share (Al-ENGR hotsplit's ranking: per-layer
normalised counts, mixed by phase weight) and fills HBM first, then the 6000, the rest stays in Grace. All cells are
the same size (19.1 MiB), so ranking by share == ranking by share per byte. Writes one rowmap both sides read:
  {"meta": {...}, "layers": {"<layer>": {"hot": [...], "peer": [...], "cold": [...]}}}
Usage: plan_tiers.py <counts.json> <out.json> --hbm-gib 152.8 --peer-gib 85 [--weights decode=1,prefill=0]
"""
import argparse, json

CELL = 3 * (2048 * 3072 + 2048 * 192)  # bytes per expert: gate+up+down, packed FP4 + E8M0 scales = 20,054,016
E, LAYERS = 384, range(1, 70)

def load(path, weights):
    raw = json.load(open(path)); out = {}
    for phase, w in weights.items():
        for li, c in raw["train"].get(phase, {}).items():
            s = sum(c) or 1.0
            out[int(li)] = [a + w * x / s for a, x in zip(out.get(int(li), [0.0] * E), c)]
    return out

p = argparse.ArgumentParser()
p.add_argument("counts"); p.add_argument("out")
p.add_argument("--hbm-gib", type=float, required=True); p.add_argument("--peer-gib", type=float, required=True)
p.add_argument("--weights", default="decode=1,prefill=0")
a = p.parse_args()
weights = {k: float(v) for k, v in (kv.split("=") for kv in a.weights.split(","))}
counts = load(a.counts, weights)
cells = sorted(((counts[li][e], li, e) for li in LAYERS for e in range(E)), reverse=True)
n_hbm = int(a.hbm_gib * 2**30 // CELL); n_peer = int(a.peer_gib * 2**30 // CELL)
tier = {}
for i, (_, li, e) in enumerate(cells):
    tier[(li, e)] = "hot" if i < n_hbm else "peer" if i < n_hbm + n_peer else "cold"
total = sum(sum(counts[li]) for li in LAYERS)
share = {t: sum(counts[li][e] for (li, e), tt in tier.items() if tt == t) / total for t in ("hot", "peer", "cold")}
layers = {str(li): {t: [e for e in range(E) if tier[(li, e)] == t] for t in ("hot", "peer", "cold")} for li in LAYERS}
peer_per_layer = [len(layers[str(li)]["peer"]) for li in LAYERS]
meta = {"counts": a.counts, "weights": weights, "hbm_gib": a.hbm_gib, "peer_gib": a.peer_gib, "cell_bytes": CELL,
        "cells": {"hot": n_hbm, "peer": n_peer, "cold": E * len(LAYERS) - n_hbm - n_peer},
        "decode_share": {k: round(v, 4) for k, v in share.items()},
        "peer_per_layer": {"min": min(peer_per_layer), "max": max(peer_per_layer)}}
json.dump({"meta": meta, "layers": layers}, open(a.out, "w"))
print(json.dumps(meta, indent=1))
