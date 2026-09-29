"""DSpark acceptance between two /metrics snapshots: accept.py <before> <after>."""
import re
import sys


def load(path):
    totals = {}
    for line in open(path):
        m = re.match(r"^(vllm:spec_decode_num_\w+?)_total(\{[^}]*\})?\s+([0-9.eE+-]+)$", line.strip())
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        pos = re.search(r'position="(\d+)"', labels)
        key = name + (f"[{pos.group(1)}]" if pos else "")
        totals[key] = totals.get(key, 0.0) + value
    return totals


a, b = load(sys.argv[1]), load(sys.argv[2])
d = {k: b.get(k, 0.0) - a.get(k, 0.0) for k in b}
drafts = d.get("vllm:spec_decode_num_drafts", 0.0)
if drafts <= 0:
    print("acceptance: no drafts in this phase")
    sys.exit()
accepted = d.get("vllm:spec_decode_num_accepted_tokens", 0.0)
draft_tokens = d.get("vllm:spec_decode_num_draft_tokens", 0.0)
per_pos = [d[k] / drafts for k in sorted((k for k in d if "per_pos[" in k), key=lambda k: int(k.split("[")[1][:-1]))]
print(f"acceptance: {1 + accepted / drafts:.2f} tokens/step, {100 * accepted / max(draft_tokens, 1):.1f}% of drafted, "
      f"{drafts:.0f} steps, avg k {draft_tokens / drafts:.2f}, per position {[round(p, 2) for p in per_pos]}")
