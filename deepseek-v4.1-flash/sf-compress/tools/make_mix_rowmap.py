"""Rebuild M3's mix-v1 weighting (0.6 decode D,E + 0.4 prefill A,B) for any hot-expert count.

  make_mix_rowmap.py <n_hot> <out.json> [--calib DIR]
"""
import argparse
import json
import os

E = 384
WEIGHTS = {"D": 0.3, "E": 0.3, "A": 0.2, "B": 0.2}

p = argparse.ArgumentParser()
p.add_argument("n_hot", type=int)
p.add_argument("out")
p.add_argument("--calib", default=os.path.join(os.path.dirname(__file__), "..", "..", "m3", "calibration"))
a = p.parse_args()

counts = {n: {int(k): v for k, v in json.load(open(os.path.join(a.calib, f"counts-{n}.json")))["layers"].items()}
          for n in WEIGHTS}
layers, cold_share = {}, {n: [0, 0] for n in counts}
for l in range(40):
    score = [sum(w * counts[n][l][e] / max(1, sum(counts[n][l])) for n, w in WEIGHTS.items()) for e in range(E)]
    order = sorted(range(E), key=lambda e: (-score[e], e))
    hot, cold = sorted(order[:a.n_hot]), sorted(order[a.n_hot:])
    layers[str(l)] = {"hot": hot, "cold": cold}
    for n, c in counts.items():
        cold_share[n][0] += sum(c[l][e] for e in cold)
        cold_share[n][1] += sum(c[l])
json.dump({"layers": layers, "meta": {"calibration": "mix-v1 weighting 0.3D+0.3E+0.2A+0.2B", "n_hot": a.n_hot}},
          open(a.out, "w"))
print(a.out, " ".join(f"{n}={100 * s / t:.2f}%cold" for n, (s, t) in cold_share.items()))
