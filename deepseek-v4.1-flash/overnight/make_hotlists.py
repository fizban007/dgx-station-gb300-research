"""Global hot-expert allocation from routing counts: pick the densest (layer, expert) pairs.

Usage: make_hotlists.py counts.json total_hot min_per_layer out.json
Writes {layer: {"hot": [expert ids]}} and prints the expected cold-route share.
"""
import json, sys
counts = json.load(open(sys.argv[1])); total = int(sys.argv[2]); floor = int(sys.argv[3])
pairs = sorted(((c, layer, e) for layer, row in counts.items() for e, c in enumerate(row)), reverse=True)
hot = {layer: set(sorted(range(len(row)), key=lambda e: -row[e])[:floor]) for layer, row in counts.items()}
budget = total - sum(len(v) for v in hot.values())
for c, layer, e in pairs:
    if budget <= 0:
        break
    if e not in hot[layer]:
        hot[layer].add(e); budget -= 1
routes = sum(sum(r) for r in counts.values())
cold = sum(sum(c for e, c in enumerate(row) if e not in hot[layer]) for layer, row in counts.items())
sizes = sorted(len(v) for v in hot.values())
print(f"hot {sum(sizes)} (per layer min {sizes[0]} max {sizes[-1]}), cold route share {cold/routes:.4f}")
json.dump({layer: {"hot": sorted(v)} for layer, v in hot.items()}, open(sys.argv[4], "w"))
