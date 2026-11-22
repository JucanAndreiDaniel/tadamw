#!/usr/bin/env python3
"""Sweep AdamW fit hyperparameters for tadamw; archive bytes are the metric.

Usage: python3 sweep.py PHASE [PHASE...]

Each phase builds a list of named configurations (env-var overrides for the
fit in src/fit.c). Every configuration is run on both corpora at the chosen
levels with `-p mix`; the resulting archive size is appended to results.tsv.
Existing archives are reused so interrupted sweeps can resume.
"""
import concurrent.futures as cf
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAW = ROOT / "build/tools/taw"
DATA = Path("/tmp/opencode/tadamw-bench")
ARCS = ROOT / "arcs_v2"       # v2 (e64cc83) archives; arcs/ holds stale v1
RESULTS = ROOT / "results_v2.tsv"
CORPORA = ("text1m", "libc1m")
BASE = {"text1m": {1: 181557, 5: 179166},
        "libc1m": {1: 379197, 5: 375291}}


def nameify(s):
    return s.replace(".", "p").replace("-", "m").replace("/", "_")


def configs(phase):
    """Return list of (name, env-overrides dict, levels)."""
    out = []

    if phase == "A":  # coarse LR sweep, lr1 = lr0/20
        for lr in ("0.02", "0.01", "0.005", "0.003", "0.002", "0.001"):
            out.append((f"A_lr{nameify(lr)}", {"TAW_LR0": lr}, (1, 5)))

    if phase == "B":  # refine LR around coarse optimum and explore the edges
        for lr in ("0.05", "0.03", "0.015", "0.008", "0.004", "0.0035",
                   "0.0025"):
            out.append((f"B_lr{nameify(lr)}", {"TAW_LR0": lr}, (1, 5)))

    if phase == "C":  # LR schedule shape at the two best starting LRs
        for lr0 in ("0.002", "0.003"):
            variants = (("r10", str(round(float(lr0) / 10, 6))),
                        ("r20", str(round(float(lr0) / 20, 6))),
                        ("r50", str(round(float(lr0) / 50, 6))),
                        ("flat", "flat"),
                        ("abs0p001", "0.001"),
                        ("abs0p002", "0.002"))
            for tag, lr1 in variants:
                out.append((f"C_lr0{nameify(lr0)}_{tag}",
                            {"TAW_LR0": lr0, "TAW_LR1": lr1}, (1, 5)))

    if phase == "F":  # v2: refine the high-LR optimum
        for lr in ("0.025", "0.035", "0.04", "0.045", "0.06", "0.07"):
            out.append((f"F_lr{nameify(lr)}", {"TAW_LR0": lr}, (1, 5)))

    if phase == "G":  # v2: LR1 schedule at best LR0 (env BEST_LR)
        best = os.environ.get("BEST_LR", "0.03")
        variants = (("r10", str(round(float(best) / 10, 6))),
                    ("r20", str(round(float(best) / 20, 6))),
                    ("r50", str(round(float(best) / 50, 6))),
                    ("r100", str(round(float(best) / 100, 6))),
                    ("flat", "flat"),
                    ("abs0p003", "0.003"),
                    ("abs0p01", "0.01"))
        for tag, lr1 in variants:
            out.append((f"G_lr0{nameify(best)}_{tag}",
                        {"TAW_LR0": best, "TAW_LR1": lr1}, (1, 5)))

    if phase == "K":  # v2: local refinement of best b1=0.95 / wd=1e-3 config
        cases = (("0.0525", "0.000525", "0.95", "1e-3"),
                 ("0.0575", "0.000575", "0.95", "1e-3"),
                 ("0.055", "0.000275", "0.95", "1e-3"),
                 ("0.055", "0.00011", "0.95", "1e-3"),
                 ("0.055", "0.00055", "0.95", "2e-3"),
                 ("0.055", "0.00055", "0.95", "7e-4"),
                 ("0.055", "0.00055", "0.94", "1e-3"))
        for lr0, lr1, b1, wd in cases:
            out.append((f"K_lr0{nameify(lr0)}_lr1{nameify(lr1)}"
                        f"_b1{nameify(b1)}_wd{nameify(wd)}",
                        {"TAW_LR0": lr0, "TAW_LR1": lr1, "TAW_B1": b1,
                         "TAW_B2": "0.999", "TAW_WD": wd}, (1, 5)))

    if phase == "J":  # v2: cross LR x b1=0.95 x WD
        for lr0, lr1, wd in (("0.04", "0.0004", "1e-3"),
                             ("0.04", "0.0004", "1e-4"),
                             ("0.055", "0.00055", "1e-3"),
                             ("0.06", "0.0006", "5e-4"),
                             ("0.06", "0.0006", "3e-4"),
                             ("0.03", "0.0003", "1e-3")):
            out.append((f"J_lr0{nameify(lr0)}_wd{nameify(wd)}",
                        {"TAW_LR0": lr0, "TAW_LR1": lr1,
                         "TAW_B1": "0.95", "TAW_B2": "0.999",
                         "TAW_WD": wd}, (1, 5)))

    if phase == "I":  # v2: final refinement around lr0=0.06, ratio 1/100
        for lr0, lr1 in (("0.055", "0.00055"), ("0.065", "0.00065"),
                         ("0.07", "0.0007"), ("0.08", "0.0008"),
                         ("0.06", "0.0003"), ("0.06", "0.0002"),
                         ("0.06", "0.001"), ("0.04", "0.0006")):
            out.append((f"I_lr0{nameify(lr0)}_lr1{nameify(lr1)}",
                        {"TAW_LR0": lr0, "TAW_LR1": lr1}, (1, 5)))

    if phase == "H":  # v2: very low final LR, and r100 at neighbouring LR0
        for lr0, lr1 in (("0.04", "0.0002"), ("0.04", "0.0001"),
                         ("0.04", "0.00004"), ("0.05", "0.0005"),
                         ("0.05", "0.0004"), ("0.03", "0.0003"),
                         ("0.06", "0.0006"), ("0.035", "0.00035")):
            out.append((f"H_lr0{nameify(lr0)}_lr1{nameify(lr1)}",
                        {"TAW_LR0": lr0, "TAW_LR1": lr1}, (1, 5)))

    if phase == "C2":  # 2-D grid: decouple initial and final LR
        for lr0, lr1 in (("0.003", "0.00003"), ("0.002", "0.00002"),
                         ("0.005", "0.0001"), ("0.005", "0.00025"),
                         ("0.01", "0.0001"), ("0.01", "0.0002"),
                         ("0.01", "0.001"), ("0.02", "0.0001"),
                         ("0.02", "0.0004"), ("0.02", "0.002"),
                         ("0.03", "0.0001"), ("0.03", "0.0003")):
            out.append((f"C2_lr0{nameify(lr0)}_lr1{nameify(lr1)}",
                        {"TAW_LR0": lr0, "TAW_LR1": lr1}, (1, 5)))

    if phase == "D":  # betas and weight decay at best LR (env BEST_LR / BEST_LR1)
        lr0 = os.environ.get("BEST_LR", "0.003")
        env0 = {"TAW_LR0": lr0}
        if os.environ.get("BEST_LR1"):
            env0["TAW_LR1"] = os.environ["BEST_LR1"]
        for b1, b2 in (("0.9", "0.999"), ("0.9", "0.99"),
                       ("0.95", "0.999"), ("0.95", "0.99")):
            for wd in ("0", "1e-5", "1e-4", "1e-3"):
                e = dict(env0, TAW_B1=b1, TAW_B2=b2, TAW_WD=wd)
                n = f"D_b1{nameify(b1)}_b2{nameify(b2)}_wd{nameify(wd)}"
                out.append((n, e, (1, 5)))

    if phase == "E":  # final confirmation at level 9
        out.append(("E_tuned_L9",
                    {"TAW_LR0": "0.003", "TAW_LR1": "0.00006"}, (9,)))
        out.append(("E_baseline_L9", {}, (9,)))

    return out


def job(spec):
    name, env_over, corpus, level = spec
    arc = ARCS / f"{name}_{corpus}_L{level}.aw"
    key = (name, corpus, level)
    if arc.exists():
        return key, arc.stat().st_size, "cache"
    env = os.environ.copy()
    for k in ("TAW_LR0", "TAW_LR1", "TAW_B1", "TAW_B2", "TAW_WD"):
        env.pop(k, None)
    env.update(env_over)
    t0 = time.monotonic()
    p = subprocess.run([str(TAW), "c", "-l", str(level), "-p", "mix",
                        str(DATA / corpus), str(arc)],
                       env=env, capture_output=True, text=True)
    secs = time.monotonic() - t0
    if p.returncode:
        return key, None, f"FAIL: {p.stderr.strip()}"
    return key, arc.stat().st_size, f"{secs:.2f}s"


def main():
    phases = sys.argv[1:] or ["A"]
    ARCS.mkdir(exist_ok=True)
    rows = {}
    if RESULTS.exists():
        with RESULTS.open() as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                rows[(p[0], p[1], int(p[2]))] = p[3:]
    specs = []
    for phase in phases:
        for name, env, levels in configs(phase):
            for corpus in CORPORA:
                for level in levels:
                    specs.append((name, env, corpus, level))
    print(f"{len(specs)} jobs", flush=True)
    with cf.ThreadPoolExecutor(max_workers=int(os.environ.get("JOBS", "3"))) as ex:
        futs = {ex.submit(job, s): s for s in specs}
        for fut in cf.as_completed(futs):
            key, size, info = fut.result()
            rows[key] = [str(size) if size else "FAIL", info]
            print(f"{key[0]:30s} {key[1]:7s} L{key[2]} "
                  f"{str(size):>8s}  {info}", flush=True)
    with RESULTS.open("w") as f:
        f.write("# name\tcorpus\tlevel\tsize\tinfo\tdelta_vs_baseline\n")
        for key in sorted(rows):
            size, info = rows[key][:2]
            delta = ""
            if size and size.isdigit():
                delta = str(int(size) - BASE[key[1]][key[2]])
            f.write("\t".join((key[0], key[1], str(key[2]),
                               size, info, delta)) + "\n")


if __name__ == "__main__":
    main()
