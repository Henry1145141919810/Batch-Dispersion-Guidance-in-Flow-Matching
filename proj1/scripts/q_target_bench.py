"""Choose a steering target per property WITHOUT reference to our own method.

    python proj1/scripts/q_target_bench.py            # run the grid (~50 min, 5080)
    python proj1/scripts/q_target_bench.py --report   # apply the rule to what ran

WHY THIS EXISTS. q50 barely tests guidance: the unguided generator already sits
0.05-0.27 sd from it on every property, so "hitting q50" is mostly concentration.
We want a target that actually tests steering -- chosen by a property of the
TASK, not by where our method happens to look good.

THE ANTI-CHERRY-PICKING CONSTRAINT. `btvg` is deliberately ABSENT from this
grid. The target is chosen from `unguided` and the two prior-art reference arms
only (`plug` = DPS, `tmpd` = PiGDM). BTVG enters afterwards, evaluated at
whatever target this picks. Choosing the q where BTVG beats the field most
would be selecting the result, not the task.

THE SELECTION STEP IS WITHDRAWN (22 Sep, before the grid finished). This script
now reports a DIAGNOSTIC TABLE only -- in-band gain versus quantile -- and
chooses nothing. An adversarial review showed the choice could not be
informative:

  * NO POWER. The score's se is 0.008-0.018 at n=512, against a measured
    cross-target spread under 0.008. Every cross-target difference in the
    existing cells is < 0.5 se, and gap/q50 flips sign between seeds
    (+0.0156 at 20260921, -0.0127 at 20260922).
  * IT RETURNS THE MEDIAN REGARDLESS. Monte Carlo: under the null it picks a
    near-median q 76 % of the time; against a true 2.3-sigma tail effect it
    recovers the right tail only 30 % of the time. It would re-derive the
    concentration task it was written to escape.
  * THE CHEMISTRY FLOOR DELETES THE TAILS. 4 of 6 already-measured
    (prop, target) pairs are ineligible, including BOTH gap targets.
  * THE CODE DID NOT IMPLEMENT THE STATED RULE. It compared each candidate to
    the RUNNING best, which the tie-break itself demotes, so it could ratchet
    downhill -- reproduced choosing a q 1.06 se below the true argmax, and
    diverging from the stated rule in 9.1 % of Monte Carlo runs.
  * EXCLUDING `btvg` WAS COSMETIC. `plug` IS BTVG's mean term (bit-identical,
    9e-8); their in-band gain correlates at r = 0.80-0.88 across strengths. The
    plug/tmpd surface is ~0.8 correlated with BTVG's own.

The headline protocol is `dist` (FR6): per-molecule targets from held-out test
molecules. It involves no choice by us, so it cannot be cherry-picked, and it is
measurably harder than q50 (0.80-0.85 sd of steering against 0.05-0.27).

WHAT THIS TABLE IS STILL GOOD FOR: showing how in-band coverage and chemistry
vary across the property range for unguided and two reference arms. That is a
real figure. It is not a decision procedure.

Fixed strength w = 1 (the pre-registered default) at window 0.5 for every
target, so targets are compared under one setting rather than each tuned.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "proj1", "scripts"), os.path.join(ROOT, "proj1", "src")]

NL = chr(10)          # literal: shell heredocs on this box mangle backslashes
QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
PROPS = ("mu", "alpha", "gap")
ARMS = ("unguided", "plug", "tmpd")          # NO btvg -- see docstring
REFS = ("plug", "tmpd")
OUT = os.path.join(ROOT, "results", "qbench")


def qname(q):
    return "q%02d" % int(round(q * 100))


def run():
    import torch
    import guidance_sweep as gs
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    tr = d["split"]["train_a"]
    targets = {}
    for p in PROPS:
        v = d["y"][tr, d["props"].index(p)].double()
        targets[p] = {qname(q): float(torch.quantile(v, q)) for q in QS}
    gs.TARGETS = targets
    cells = [(p, a, qname(q), 1.0, 0.5, "qb") for p in PROPS for q in QS for a in ARMS]
    gs.plan_cells = lambda props, arms: cells
    os.makedirs(OUT, exist_ok=True)
    json.dump(targets, open(os.path.join(OUT, "_targets.json"), "w"), indent=1)
    sys.argv = ["q_target_bench", "--props", ",".join(PROPS), "--n", "512",
                "--batch", "128", "--steps", "100", "--device", "cuda",
                "--fm", os.path.join(ROOT, "betty_pull", "fm_last.pt"),
                "--out-dir", OUT]
    return gs.main()


def se(p, n):
    return math.sqrt(max(p * (1 - p), 0) / n)


def report():
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(OUT, "*.json"))
            if not os.path.basename(f).startswith("_")]
    if not rows:
        print("no cells in %s -- run without --report first" % OUT)
        return 2
    cell = {(r["prop"], r["arm"], r["target_name"]): r for r in rows}
    targets = json.load(open(os.path.join(OUT, "_targets.json")))
    print("DIAGNOSTIC ONLY -- this table chooses nothing. See the docstring.")
    for p in PROPS:
        print(NL + "### %s" % p)
        print("%-5s %9s %8s %8s %8s %9s %8s %9s" % (
            "q", "target", "ungd_ib", "plug_ib", "tmpd_ib", "gain", "+-se", "chem_ok"))
        for q in QS:
            n = qname(q)
            u = cell.get((p, "unguided", n))
            rs = [cell.get((p, a, n)) for a in REFS]
            if not u or not all(rs):
                print("%-5s  (not run)" % n)
                continue
            gain = sum(r["in_band_fraction"] for r in rs) / len(rs) - u["in_band_fraction"]
            s_ = math.sqrt(se(u["in_band_fraction"], u["n"]) ** 2 +
                           sum(se(r["in_band_fraction"], r["n"]) ** 2 for r in rs) / len(rs) ** 2)
            ok = all(r["mol_stability"] >= 0.9 * u["mol_stability"] for r in rs)
            print("%-5s %9.4f %8.3f %8.3f %8.3f %+9.3f %8.3f %9s" % (
                n, targets[p][n], u["in_band_fraction"], rs[0]["in_band_fraction"],
                rs[1]["in_band_fraction"], gain, s_, "yes" if ok else "NO"))
    print(NL + "Read the 'gain' column against its se: differences smaller than one")
    print("se carry no information. The full run uses `dist` (FR6), not a q from")
    print("this table.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    sys.exit(report() if a.report else run())
