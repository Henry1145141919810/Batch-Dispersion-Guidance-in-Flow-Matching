"""Analyse the guidance sweep, including the reward-hacking diagnostic.

    python results/bench/analyse_sweep.py
    python results/bench/analyse_sweep.py --prop mu --window 0.05

Every headline number is scored by f_B, the HELD-OUT evaluator trained on
train_b, which neither the generator nor the guide has ever seen:

    prop_mae_eval    = mean |f_B(x) - y*|          <- the number that counts
    in_band_fraction = fraction with |f_B(x) - y*| <= delta = 2 x f_B's own MAE

f_A appears in exactly one place, and it is the point of having two networks:

    guide_eval_gap   = mean |f_A(x) - f_B(x)|

Guidance is an optimiser pointed at f_A. Given enough strength it will find
inputs that make f_A report the target WITHOUT the molecule actually having the
property -- reward hacking. The signature is f_A agreeing with the target while
f_B disagrees, i.e. a gap that GROWS with guidance strength. An arm that
targets well and keeps the gap near its unguided value is steering honestly;
one whose gap balloons is exploiting the guide.

The gap is reported in units of delta, so it is comparable across properties.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SWEEP = os.path.join(ROOT, "results", "sweep")


def load():
    rows = []
    for f in glob.glob(os.path.join(SWEEP, "*.json")):
        d = json.load(open(f))
        if "prop_mae_eval" in d:
            rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", default=None)
    ap.add_argument("--window", type=float, default=None)
    ap.add_argument("--target", default="q50")
    args = ap.parse_args()

    rows = load()
    if not rows:
        raise SystemExit("no cells in %s" % SWEEP)
    print("%d cells loaded from %s\n" % (len(rows), SWEEP))

    props = [args.prop] if args.prop else ["mu", "alpha", "gap"]
    for prop in props:
        pr = [r for r in rows if r["prop"] == prop and r["target_name"] == args.target]
        if not pr:
            continue
        wins = sorted({r["t_min_guide"] for r in pr}) if args.window is None else [args.window]
        for win in wins:
            sub = [r for r in pr if r["t_min_guide"] == win]
            if not sub:
                continue
            delta = sub[0]["delta"]
            tgt = sub[0]["target"]
            # each arm at its own best strength, chosen on the f_B metric
            best = {}
            for r in sub:
                a = r["arm"]
                if a not in best or r["prop_mae_eval"] < best[a]["prop_mae_eval"]:
                    best[a] = r
            print("=== %s   target=%.4g   delta=%.4g   t_min=%g" % (prop, tgt, delta, win))
            print("  %-11s %5s | %9s %7s | %9s %9s %8s | %8s %8s"
                  % ("arm", "w", "MAE(f_B)", "in_band", "f_A mean", "f_B mean",
                     "|A-B|/d", "mol_st", "valid"))
            print("  " + "-" * 96)
            for a, r in sorted(best.items(), key=lambda kv: kv[1]["prop_mae_eval"]):
                gap_d = r["guide_eval_gap_mean"] / delta
                # does f_A believe it hit the target while f_B does not?
                a_err = abs(r["f_A_mean"] - tgt)
                b_err = abs(r["f_B_mean"] - tgt)
                flag = ""
                if a_err < b_err * 0.5 and b_err > delta:
                    flag = "  <- f_A says hit, f_B says miss"
                print("  %-11s %5g | %9.4f %7.3f | %9.4f %9.4f %8.1f | %8.3f %8.3f%s"
                      % (a, r["w"], r["prop_mae_eval"], r["in_band_fraction"],
                         r["f_A_mean"], r["f_B_mean"], gap_d,
                         r["mol_stability"], r["validity"], flag))
            print()

        # how the gap moves with guidance strength -- the reward-hacking signature
        print("--- %s: does the f_A/f_B gap grow with strength? (t_min=0.05, units of delta)" % prop)
        w05 = [r for r in pr if r["t_min_guide"] == 0.05]
        arms = sorted({r["arm"] for r in w05})
        ws = sorted({r["w"] for r in w05})
        print("  %-11s %s" % ("arm", " ".join("%8g" % w for w in ws)))
        for a in arms:
            cells = {r["w"]: r for r in w05 if r["arm"] == a}
            line = " ".join(("%8.1f" % (cells[w]["guide_eval_gap_mean"] / cells[w]["delta"]))
                            if w in cells else "%8s" % "-" for w in ws)
            print("  %-11s %s" % (a, line))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
