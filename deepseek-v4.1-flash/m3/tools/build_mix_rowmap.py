"""Build a decode/prefill-mixed rowmap, rowmap-mix-v1's recipe at any hot count: per-layer route shares from
calibration sets D (prose) and E (code/QA/reasoning) weigh 0.6 for decode, A (stdlib+docs+chat) and B (vLLM/b12x
sources+chat) weigh 0.4 for prefill, split evenly within each phase; the n_hot highest-weight experts per layer are hot.
    build_mix_rowmap.py <n_hot> <out>   (prints the cold-route share on every calibration set, incl. held-out C)"""
import json, sys
sys.path.insert(0, "/home/jasonc/research/megamoe")
from build_rowmap import load_counts, cold_share, E
n_hot, out = int(sys.argv[1]), sys.argv[2]
sets = {n: load_counts(f"/home/jasonc/research/megamoe/prof/counts-{n}.json") for n in "ABCDE"}
norm = {n: {l: [x / max(1, sum(c[l])) for x in c[l]] for l in c} for n, c in sets.items()}
w = {"D": 0.3, "E": 0.3, "A": 0.2, "B": 0.2}
layers = {}
for l in range(40):
    comb = [sum(w[n] * norm[n][l][e] for n in w) for e in range(E)]
    order = sorted(range(E), key=lambda e: (-comb[e], e))
    layers[str(l)] = {"hot": sorted(order[:n_hot]), "cold": sorted(order[n_hot:])}
json.dump({"layers": layers, "meta": {"calibration": "0.6 decode (D: prose, E: code/QA/reasoning) + 0.4 prefill (A: stdlib+docs+chat, B: vLLM/b12x sources+chat)", "n_hot": n_hot}}, open(out, "w"))
m = {int(k): v for k, v in layers.items()}
print(out, " ".join(f"{n}={100 * cold_share(m, s)[0]:.2f}%" for n, s in sets.items()))
