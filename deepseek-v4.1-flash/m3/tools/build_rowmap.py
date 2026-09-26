"""Build a hot/cold rowmap from calibration counts and score it against the current one.

  build_rowmap.py counts-A.json [--out hook/rowmap-cal-v1.json] [--eval counts-B.json counts-C.json]

Per layer, hot = the 295 experts with the most routed tokens in the calibration counts (ties keep the current
rowmap's hot experts first). Scores are the share of routed tokens that land on cold experts.
"""
import argparse
import json

E, N_HOT = 384, 295
CURRENT = "/home/jasonc/research/megamoe/hook/rowmap-static-v1.json"


def load_counts(path):
    return {int(k): v for k, v in json.load(open(path))["layers"].items()}


def cold_share(rowmap, counts):
    cold = total = 0
    worst = 0.0
    for layer, c in counts.items():
        cs = sum(c[e] for e in rowmap[layer]["cold"])
        t = sum(c)
        cold += cs
        total += t
        worst = max(worst, cs / max(1, t))
    return cold / max(1, total), worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("calibration")
    p.add_argument("--out", default="/home/jasonc/research/megamoe/hook/rowmap-cal-v1.json")
    p.add_argument("--eval", nargs="*", default=[])
    a = p.parse_args()

    current = {int(k): v for k, v in json.load(open(CURRENT))["layers"].items()}
    counts = load_counts(a.calibration)
    new = {}
    for layer in range(40):
        c = counts[layer]
        was_hot = set(current[layer]["hot"])
        order = sorted(range(E), key=lambda e: (-c[e], e not in was_hot, e))
        hot = sorted(order[:N_HOT])
        cold = sorted(order[N_HOT:])
        new[layer] = {"hot": hot, "cold": cold}
    json.dump({"layers": {str(k): v for k, v in new.items()},
               "meta": {"calibration": a.calibration, "n_hot": N_HOT}}, open(a.out, "w"))
    moved = sum(len(set(new[l]["hot"]) - set(current[l]["hot"])) for l in range(40))
    print(f"wrote {a.out}: {moved} of {40 * N_HOT} hot slots changed vs {CURRENT.split('/')[-1]}")
    for path in [a.calibration] + a.eval:
        cs = load_counts(path)
        cur, cur_w = cold_share(current, cs)
        nw, nw_w = cold_share(new, cs)
        print(f"{path.split('/')[-1]:>16}: cold routes {100 * cur:5.2f}% -> {100 * nw:5.2f}% "
              f"(worst layer {100 * cur_w:5.2f}% -> {100 * nw_w:5.2f}%)")


if __name__ == "__main__":
    main()
