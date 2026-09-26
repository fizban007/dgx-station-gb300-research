"""Turn routing counts {layer: [count per expert]} into a hot-first ranking {layer: [expert ids]}."""
import json, sys
counts = json.load(open(sys.argv[1]))
ranking = {layer: sorted(range(len(row)), key=lambda e: (-row[e], e)) for layer, row in counts.items()}
json.dump(ranking, open(sys.argv[2], "w"))
total = {layer: sum(row) for layer, row in counts.items()}
for hot in (256, 288, 320):
    covered = [sum(sorted(row, reverse=True)[:hot]) / max(1, total[layer]) for layer, row in counts.items()]
    print(f"top-{hot}: mean route coverage {sum(covered)/len(covered):.3f}, min {min(covered):.3f}")
