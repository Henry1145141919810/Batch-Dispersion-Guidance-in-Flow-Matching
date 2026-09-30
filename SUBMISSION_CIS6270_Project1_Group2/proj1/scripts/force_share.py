"""Measure how hard each arm actually pushes, at each nominal strength.

    python proj1/scripts/force_share.py            # mu, 5 arms x 4 strengths

Writes results/force_share/<prop>__<arm>__w<w>.json, one velocity_share_diag
run per (arm, strength), for results_tables.py section 7.

THE QUESTION. `w` is the knob every table sorts by, and `strength_scale`
normalises some arms so that w is meant to be comparable across them. Is it?
The quantity that decides what guidance does to a trajectory is not w but the
correction actually applied after the score-to-velocity factor and the clip,
relative to the base flow-matching velocity:

    r_t = |C(G)| / |V|

If two arms at the same w have very different r_t, then "arm X at w vs arm Y
at w" compares two different forces under one label -- and FR3a, which froze
each arm at its strongest floor-clearing w, froze them at different forces.

Measured with the update rule `euler` (the shipped rule, bit-identical to the
pre-change sampler), so r_t is the RAW correction's share. Same settings as
the compare stage: q50 target, t_min 0.5, 100 Euler steps, seed 20260921.

`tfg` and `dflow` replace the sampler rather than adding a field, so there is
no C(G) to measure and they are not in this table.

Resumable: a (prop, arm, w) with an output file is skipped. Runs one process
at a time -- each loads the generator and the dataset, and two at once would
break the leave-5-GB-free rule on this machine.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "force_share")
ARMS = ("plug", "tmpd", "lgd_mc", "btvg", "btvg_var")
STRENGTHS = (0.05, 0.25, 1.0, 4.0)


def out_path(prop, arm, w):
    return os.path.join(OUT, "%s__%s__w%g.json" % (prop, arm, w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", default="mu")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--strengths", default=",".join("%g" % w for w in STRENGTHS))
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(p, a, float(w)) for p in args.props.split(",") if p
            for a in args.arms.split(",") if a
            for w in args.strengths.split(",") if w]
    todo = [j for j in jobs if not os.path.exists(out_path(*j))]
    print("force share: %d runs, %d already done" % (len(jobs), len(jobs) - len(todo)))
    for i, (p, a, w) in enumerate(todo, 1):
        cmd = [sys.executable, os.path.join(HERE, "velocity_share_diag.py"),
               "--prop", p, "--arm", a, "--w", "%g" % w, "--rules", "euler",
               "--n", str(args.n), "--out", out_path(p, a, w)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        line = next((ln for ln in r.stdout.splitlines()
                     if ln.strip().startswith("r_t ") and "mean" in ln), "")
        print("  [%2d/%2d] %-6s %-9s w=%-5g %s%s" % (
            i, len(todo), p, a, w, line.strip(),
            "" if r.returncode == 0 else "  FAILED: " + r.stderr.strip()[-200:]))
        if r.returncode != 0:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
