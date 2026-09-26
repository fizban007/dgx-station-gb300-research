"""Merge every lane's results.jsonl into data/results.jsonl and data/results.csv, and check them.

Checks: every row has exactly the schema keys, and every repo-relative source file exists.
Usage: python3 data/build.py
"""
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ["lane", "model", "config", "date", "engine", "benchmark", "metric", "concurrency", "context_tokens",
        "value", "unit", "qualified", "source", "notes"]
rows, errors = [], []
for path in sorted(glob.glob(os.path.join(ROOT, "**", "results.jsonl"), recursive=True)):
    if os.path.dirname(path) == os.path.join(ROOT, "data"):
        continue
    for n, line in enumerate(open(path), 1):
        r = json.loads(line)
        where = f"{os.path.relpath(path, ROOT)}:{n}"
        if sorted(r) != sorted(KEYS):
            errors.append(f"{where}: keys {sorted(set(r) ^ set(KEYS))}")
        src = r.get("source")
        if src and not src.startswith("http"):
            if not os.path.exists(os.path.join(ROOT, src)):
                errors.append(f"{where}: missing source {src}")
        rows.append({k: r.get(k) for k in KEYS})
with open(os.path.join(ROOT, "data", "results.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(ROOT, "data", "results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=KEYS)
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} rows from {len({r['lane'] for r in rows})} lanes; {len(errors)} problems")
for e in errors[:50]:
    print("  " + e)
sys.exit(1 if errors else 0)
