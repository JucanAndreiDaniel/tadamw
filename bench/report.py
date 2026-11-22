#!/usr/bin/env python3
"""Summarize bench/sweep.tsv as a per-cell archive-size table."""
import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "bench/sweep.tsv"
rows = list(csv.reader(open(path), delimiter="\t"))
data = {}
for name, env, corpus, level, size, arc in rows[1:]:
    data.setdefault(name, {})[(corpus, "L" + level)] = int(size) if size.isdigit() else None
cells = [("text1m", "L1"), ("text1m", "L5"), ("libc1m", "L1"), ("libc1m", "L5")]
base = data.get("base", {})
b = sum(base[c] for c in cells)
print(f"{'config':18} " + " ".join(f"{c[0][4:]:>4}-{c[1]}" for c in cells) +
      "     sum   delta")
for n, d in sorted(data.items(), key=lambda kv: sum(v or 0 for v in kv[1].values())):
    vals = [d.get(c) for c in cells]
    if any(v is None for v in vals):
        print(f"{n:18} INCOMPLETE {vals}")
        continue
    s = sum(vals)
    print(f"{n:18} " + " ".join(f"{v:8}" for v in vals) + f" {s:8} {s-b:+7}")
print(f"baseline sum {b}")
