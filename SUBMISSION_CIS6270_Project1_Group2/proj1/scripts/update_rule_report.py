"""Read the update-rule cells and print the comparison.

    python proj1/scripts/update_rule_report.py

Three tables, in the order the question should be answered:

  1. THE FRONTIER. MAE and molecule stability for every (arm, rule, w). A rule
     only beats euler if it is better at comparable stability -- MAE alone is
     reachable by pushing harder, and every cell here reports both.
  2. STALENESS. The cosine between the momentum buffer and the current
     direction, split early/late in the guided window, beside the cosine
     between consecutive raw directions. If consecutive directions already
     disagree, an average over them is averaging a moving target.
  3. RUNTIME. Measured ms/molecule per rule.

Standard errors: MAE's se is sd/sqrt(n) with sd estimated from the reported
RMSE (|e| has mean MAE and second moment RMSE^2 by definition), and stability
is a proportion, so se = sqrt(p(1-p)/n). Same convention as select_arms.py.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
OUT = os.path.join(ROOT, "results", "update_rule")
RULE_ORDER = ["euler", "momentum", "adam", "adam_eq", "muon"]


def load(path=OUT):
    rows = []
    for f in sorted(glob.glob(os.path.join(path, "*.json"))):
        with open(f) as fh:
            rows.append(json.load(fh))
    return rows


def mae_se(r):
    n = r.get("n") or 1
    mae, rmse = r.get("prop_mae_eval"), r.get("prop_rmse_eval")
    if mae is None or rmse is None:
        return float("nan")
    var = max(rmse ** 2 - mae ** 2, 0.0)
    return math.sqrt(var / n)


def prop_se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def main():
    rows = load(sys.argv[1] if len(sys.argv) > 1 else OUT)
    if not rows:
        print("no cells in %s" % OUT)
        return 1
    props = sorted({r["prop"] for r in rows})
    arms = sorted({r["arm"] for r in rows})

    print("=" * 96)
    print("1. THE FRONTIER -- MAE and molecule stability at each strength")
    print("   A rule wins only by being better at comparable stability.")
    print("=" * 96)
    for prop in props:
        for arm in arms:
            sub = [r for r in rows if r["prop"] == prop and r["arm"] == arm]
            if not sub:
                continue
            ws = sorted({r["w"] for r in sub})
            print("\n%s / %s   (delta=%.4f, n=%d)"
                  % (prop, arm, sub[0]["delta"], sub[0]["n"]))
            print("  %-9s %-4s %10s %8s  %10s %8s  %7s %6s"
                  % ("rule", "eq", "MAE", "+-se", "mol_stab", "+-se",
                     "in_band", "valid"))
            for rule in RULE_ORDER:
                for w in ws:
                    c = [r for r in sub if r["rule"] == rule and r["w"] == w]
                    if not c:
                        continue
                    r = c[0]
                    print("  %-9s %-4s %10.4f %8.4f  %10.3f %8.3f  %7.3f %6.3f"
                          % (rule + "@" + ("%g" % w),
                             "y" if r["equivariant"] else "N",
                             r["prop_mae_eval"], mae_se(r),
                             r["mol_stability"],
                             prop_se(r["mol_stability"], r["n"]),
                             r["in_band_fraction"], r["validity"]))

    print("\n" + "=" * 96)
    print("2. STALENESS -- does an accumulated direction still point anywhere")
    print("   useful?  cos_prev: consecutive RAW directions.  cos_mom_g: the")
    print("   buffer against the direction the guide asks for NOW.")
    print("=" * 96)
    print("  %-6s %-6s %-9s %-5s %8s %8s %8s   %8s %8s %8s"
          % ("prop", "arm", "rule", "w", "cosprev", "early", "late",
             "cos_mom", "early", "late"))
    for r in sorted(rows, key=lambda r: (r["prop"], r["arm"],
                                         RULE_ORDER.index(r["rule"]), r["w"])):
        def fmt(v):
            return "%8.3f" % v if isinstance(v, float) else "%8s" % "-"
        print("  %-6s %-6s %-9s %-5g %s %s %s   %s %s %s"
              % (r["prop"], r["arm"], r["rule"], r["w"],
                 fmt(r.get("cos_prev_mean")), fmt(r.get("cos_prev_early")),
                 fmt(r.get("cos_prev_late")), fmt(r.get("cos_mom_g_mean")),
                 fmt(r.get("cos_mom_g_early")), fmt(r.get("cos_mom_g_late"))))

    print("\n" + "=" * 96)
    print("3. RUNTIME and the magnitude the preconditioner would have applied")
    print("   (pre_scale = |D|/|G| before the norm rescale; 1.0 = no change)")
    print("=" * 96)
    print("  %-9s %10s %12s %12s" % ("rule", "ms/mol", "vs euler", "pre_scale"))
    base = None
    for rule in RULE_ORDER:
        sub = [r for r in rows if r["rule"] == rule]
        if not sub:
            continue
        ms = sum(r["ms_per_mol"] for r in sub) / len(sub)
        ps = [r["pre_scale_mean"] for r in sub
              if isinstance(r.get("pre_scale_mean"), float)]
        if rule == "euler":
            base = ms
        print("  %-9s %10.1f %12s %12s"
              % (rule, ms,
                 ("%.2fx" % (ms / base)) if base else "-",
                 ("%.3g" % (sum(ps) / len(ps))) if ps else "-"))

    print("\n" + "=" * 96)
    print("4. THE DECIDING TABLE -- each rule against euler AT MATCHED")
    print("   MOLECULE STABILITY. Raising w lowers MAE and costs chemistry, so")
    print("   a lower MAE at the same w is a position on that trade-off, not a")
    print("   win. euler's own (stability, MAE) points are interpolated to the")
    print("   rule's stability and the two MAEs compared there.")
    print("=" * 96)
    for prop in props:
        for arm in arms:
            sub = [r for r in rows if r["prop"] == prop and r["arm"] == arm]
            eu = sorted([r for r in sub if r["rule"] == "euler"],
                        key=lambda r: r["mol_stability"])
            if len(eu) < 2:
                continue
            lo, hi = eu[0]["mol_stability"], eu[-1]["mol_stability"]
            print("\n%s / %s   euler frontier: %s"
                  % (prop, arm, "  ".join("%.3f->%.4g" % (r["mol_stability"],
                                                          r["prop_mae_eval"])
                                          for r in eu)))
            print("  %-9s %-5s %8s %10s %12s %10s %6s"
                  % ("rule", "w", "stab", "MAE", "euler@stab", "delta", "se"))
            for rule in RULE_ORDER:
                if rule == "euler":
                    continue
                for r in sorted([x for x in sub if x["rule"] == rule],
                                key=lambda x: x["w"]):
                    st = r["mol_stability"]
                    if not lo <= st <= hi:
                        print("  %-9s %-5g %8.3f %10.4g %12s %10s %6s"
                              % (rule, r["w"], st, r["prop_mae_eval"],
                                 "out of range", "-", "-"))
                        continue
                    e = None
                    for a, b in zip(eu, eu[1:]):
                        if a["mol_stability"] <= st <= b["mol_stability"]:
                            span = b["mol_stability"] - a["mol_stability"]
                            fr = (st - a["mol_stability"]) / span if span else 0.0
                            e = (a["prop_mae_eval"]
                                 + fr * (b["prop_mae_eval"] - a["prop_mae_eval"]))
                            break
                    if e is None:
                        continue
                    d = e - r["prop_mae_eval"]          # positive = rule better
                    se = mae_se(r)
                    print("  %-9s %-5g %8.3f %10.4g %12.4g %10.4g %6.1f %s"
                          % (rule, r["w"], st, r["prop_mae_eval"], e, d,
                             (d / se if se else 0.0),
                             "BETTER" if d > 3 * se else
                             ("worse" if d < -3 * se else "tie")))

    # The headline the user asked for, stated rather than left to the reader.
    print("\n" + "=" * 96)
    print("VERDICT per property/arm: best rule by MAE, and whether it holds up")
    print("once molecule stability is read beside it.")
    print("=" * 96)
    for prop in props:
        for arm in arms:
            sub = [r for r in rows if r["prop"] == prop and r["arm"] == arm]
            if not sub:
                continue
            eu = {r["w"]: r for r in sub if r["rule"] == "euler"}
            for w in sorted(eu):
                cands = [r for r in sub if r["w"] == w and r["rule"] != "euler"]
                if not cands:
                    continue
                best = min(cands, key=lambda r: r["prop_mae_eval"])
                b, e = best, eu[w]
                d = e["prop_mae_eval"] - b["prop_mae_eval"]
                se = math.sqrt(mae_se(b) ** 2 + mae_se(e) ** 2)
                ds = b["mol_stability"] - e["mol_stability"]
                sig = "yes" if d > 3 * se else "no"
                print("  %-6s %-6s w=%-5g best=%-9s dMAE=%+.4f (%.1f se, "
                      "sig=%s)  dStab=%+.3f"
                      % (prop, arm, w, b["rule"], -d,
                         (d / se if se else 0.0), sig, ds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
