"""Build calibrated rowmaps from equal-weight per-set counts. build_cal_rowmaps.py <n_hot> <out> <sets...> """
import json, sys
sys.path.insert(0, "/home/jasonc/research/megamoe")
from build_rowmap import load_counts, cold_share, E

n_hot, out, names = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
sets = {n: load_counts(f"/home/jasonc/research/megamoe/prof/counts-{n}.json") for n in "ABC"}
norm = {n: {l: [x / max(1, sum(c[l])) for x in c[l]] for l in c} for n, c in sets.items()}
layers = {}
for l in range(40):
    comb = [sum(norm[n][l][e] for n in names) for e in range(E)]
    order = sorted(range(E), key=lambda e: (-comb[e], e))
    layers[str(l)] = {"hot": sorted(order[:n_hot]), "cold": sorted(order[n_hot:])}
json.dump({"layers": layers, "meta": {"calibration_sets": names, "n_hot": n_hot}}, open(out, "w"))
m = {int(k): v for k, v in layers.items()}
print(out, " ".join(f"{n}={100 * cold_share(m, s)[0]:.2f}%" for n, s in sets.items()))
