"""Apply pre-registered rule FR1 to the compare stage, and freeze strengths for FR3.

NAMED FR1-FR5 ("full-run rules"), NOT G1-G5: those names already belong to
the experiment plan's gates (G1 guide resolution ... G5 soft-vs-decoded) in
docs/protocol/GUIDANCE_EXPERIMENT_PLAN.md, and g1_guide_resolution.py
implements that G1.

    python proj1/scripts/check_fullrun_go.py

Written 22 Sep, BEFORE any compare cell existed, so the decision is mechanical
and cannot be tuned after seeing the numbers. The rules it implements are in
docs/status/SCOPE_FM_GUIDANCE_STATUS.md, "PRE-REGISTRATION".

FR1 BTVG proceeds to the full-scale run UNLESS, on 2 or more of the 3
    properties, it is beaten by the best competitor by more than 3 combined
    standard errors on MAE, each arm at its best strength.

FR3 Each arm's best-MAE strength here is FROZEN for the full run, which then
    uses new seeds so the choice is not also the evaluation.

Costs seconds of CPU and no GPU: it only reads the cell JSONs.
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

PROPS = ("mu", "alpha", "gap")
COMPETITORS = ("plug", "tmpd", "lgd_mc", "dflow")
OURS = "btvg"
ALL_ARMS = ("unguided",) + COMPETITORS + (OURS, "btvg_var")
SIGMA = 3.0          # FR1 threshold, fixed in the pre-registration
BEATEN_ON = 2        # "on 2 or more of the 3 properties"


def se_mae(r):
    """se of MAE from its first two moments: var(|e|) = RMSE^2 - MAE^2."""
    v = max(r["prop_rmse_eval"] ** 2 - r["prop_mae_eval"] ** 2, 0.0)
    return math.sqrt(v / r["n"])


def load(sweep, stage, target):
    rows = []
    for fn in glob.glob(os.path.join(sweep, "*.json")):
        try:
            r = json.load(open(fn))
        except Exception:
            continue
        if r.get("stage") != stage or r.get("target_name") != target:
            continue
        if r.get("n_nonfinite", 0) > 0:
            continue        # a divergent cell may not be anyone's best
        rows.append(r)
    return rows


def best(rows, prop, arm):
    c = [r for r in rows if r["prop"] == prop and r["arm"] == arm]
    return min(c, key=lambda r: r["prop_mae_eval"]) if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=os.path.join(ROOT, "results", "sweep"))
    ap.add_argument("--stage", default="compare",
                    help="'compare' is the real test; any other value is a dry run")
    ap.add_argument("--target", default="q50")
    ap.add_argument("--json-out", default="",
                    help="write the frozen G3 strengths here for the full run")
    args = ap.parse_args()

    rows = load(args.sweep_dir, args.stage, args.target)
    if not rows:
        print("no '%s'-stage cells at target %s in %s -- has the compare run "
              "finished?" % (args.stage, args.target, args.sweep_dir))
        return 2

    # completeness: G1 is only valid on the full grid
    missing = [(p, a) for p in PROPS for a in ALL_ARMS if not best(rows, p, a)]
    if missing:
        print("INCOMPLETE -- no clean cell for: %s" % ", ".join(
            "%s/%s" % m for m in missing))
        print("FR1 is not evaluated on a partial grid. Wait for the run.")
        return 2

    print("FR1: is %s beaten by the best competitor by > %g sigma on MAE?\n"
          % (OURS, SIGMA))
    print("%-6s %-10s %10s %10s %10s %8s  %s"
          % ("prop", "best comp", "comp MAE", "btvg MAE", "diff", "sigma", "beaten?"))
    print("-" * 72)
    beaten = 0
    for p in PROPS:
        b = best(rows, p, OURS)
        comp = min((best(rows, p, a) for a in COMPETITORS),
                   key=lambda r: r["prop_mae_eval"])
        d = b["prop_mae_eval"] - comp["prop_mae_eval"]         # >0: btvg worse
        z = d / math.sqrt(se_mae(b) ** 2 + se_mae(comp) ** 2)
        hit = z > SIGMA
        beaten += hit
        print("%-6s %-10s %10.4f %10.4f %+10.4f %+8.2f  %s"
              % (p, comp["arm"], comp["prop_mae_eval"], b["prop_mae_eval"],
                 d, z, "YES" if hit else "no"))
    print("-" * 72)
    ok = beaten < BEATEN_ON
    print("beaten on %d of 3 properties (fails at %d)" % (beaten, BEATEN_ON))
    print("FR1: %s" % ("PASS -- btvg proceeds to the full run" if ok else
                      "FAIL -- btvg is reported as a negative result, NOT re-tuned"))

    # FR3: freeze each arm's best strength
    print("\nG3: strengths FROZEN for the full run (best MAE, seed %s)"
          % rows[0].get("seed"))
    print("%-10s %8s %8s %8s" % ("arm", *PROPS))
    frozen = {}
    for a in ALL_ARMS:
        ws = []
        for p in PROPS:
            r = best(rows, p, a)
            ws.append(r["w"])
            frozen.setdefault(a, {})[p] = r["w"]
        print("%-10s %8g %8g %8g" % (a, *ws))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"fr1_pass": ok, "beaten_on": beaten, "frozen_w": frozen,
                       "source_stage": args.stage, "target": args.target,
                       "source_seed": rows[0].get("seed")}, fh, indent=1)
        print("\nwrote %s" % args.json_out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
