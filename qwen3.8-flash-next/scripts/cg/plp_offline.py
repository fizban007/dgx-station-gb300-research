"""Offline comparisons of saved plp.py runs (teacher-forced prompt logprobs) and greedy_check.py runs.
Usage: plp_offline.py <plp_cg1024.json[.gz]> <plp_cg8192_a.json[.gz]> <plp_cg8192_b.json[.gz] or -> <greedy_cg1024.json> <greedy_cg8192.json>
Positions are 1..n-1 of each prompt (position 0 has no logprob). Pass - for the second CG=8192 run to skip section B."""
import gzip, json, sys
load = lambda p: json.load(gzip.open(p) if p.endswith(".gz") else open(p))
base, cga, cgb, g1, g2 = (None if p == "-" else load(p) for p in sys.argv[1:6])


def cmp(a, b, n=None):
    a, b = a[1:n + 1 if n else None], b[1:n + 1 if n else None]
    d = sorted(abs(x - y) for x, y in zip(a, b))
    k = len(d)
    return (f"NLL {-sum(b) / k:.4f} vs {-sum(a) / k:.4f}  mean|dlogprob| {sum(d) / k:.4f}  "
            f"p99 {d[int(.99 * k)]:.3f}  max {d[-1]:.3f}")


print("== A: CG=1024 run vs CG=8192 run (whole prompt; each prompt its own cold request)")
for k in base:
    print(f"  {k:>11}: {cmp(cga[k], base[k])}")
if cgb is not None:
    print("== B: two CG=8192 runs on separate server instances (run-to-run noise floor)")
    for k in cga:
        print(f"  {k:>11}: {cmp(cgb[k], cga[k])}")
print("== C: first 1,899 positions, same text, different step size")
for off in (0, 30000):
    print(f"  offset {off}:")
    print(f"    CG=1024  1900-token step (eager) vs 3000-token step (eager):     {cmp(base[f'3000@{off}'], base[f'1900@{off}'], 1899)}")
    print(f"    CG=8192  1900-token step (graph 2048) vs 3000-token step (graph 3200): {cmp(cga[f'3000@{off}'], cga[f'1900@{off}'], 1899)}")
    print(f"    3000-token step, CG=1024 (eager) vs CG=8192 (graph 3200):        {cmp(cga[f'3000@{off}'], base[f'3000@{off}'], 1899)}")
    print(f"    1900-token step, CG=1024 (eager) vs CG=8192 (graph 2048):        {cmp(cga[f'1900@{off}'], base[f'1900@{off}'], 1899)}")
print("== D: greedy 32-token continuations, CG=1024 vs CG=8192 (cold)")
for k, v in g2.items():
    b = g1[k]
    m = next((i for i, (x, y) in enumerate(zip(v["tokens"], b["tokens"])) if x != y), len(v["tokens"]))
    d = max((abs(x - y) for x, y in zip(v["lps"][:m], b["lps"][:m])), default=float("nan"))
    print(f"  {k:>12}: identical tokens {m:2d}/32, max |dlogprob| before divergence {d:.4f}")
