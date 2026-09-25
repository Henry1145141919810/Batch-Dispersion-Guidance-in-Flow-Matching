"""Which target did the handoff's plug w=4 control run at?

The handoff never states it. Its plug w=4 control reports
    in_band   mu .1309  alpha .0684  gap .1602
    mol_stab  mu .2930  alpha .3438  gap .2812
Compare against plug w=4 run HERE at q50 and q90 (same generator file, same
100 steps / t_min_guide 0.5, but n=256 on one GPU batch instead of n=512 on
CPU). L1 distance over the six numbers decides nothing on its own -- it is
reported with the caveat that device, batch size and n all differ from the
handoff's runs AND from results/sweep, so this is a third setting, not a
replication of either.
"""
import glob
import json
import os

CELLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "cells")
REF_IB = {"mu": 0.1309, "alpha": 0.0684, "gap": 0.1602}
REF_MS = {"mu": 0.2930, "alpha": 0.3438, "gap": 0.2812}


def main():
    got = {}
    for p in glob.glob(os.path.join(CELLS, "*__plug__*__w4__*tgt.json")):
        c = json.load(open(p))
        got[(c["prop"], c["target_name"])] = c
    print("plug w=4 here (n=256, one GPU batch, seed 20260925, fm_ema.pt)")
    print("  %-6s %-18s %-18s %-18s" % ("prop", "handoff", "q50 here", "q90 here"))
    tot = {"q50": 0.0, "q90": 0.0}
    for prop in ("mu", "alpha", "gap"):
        row = ["%.4f/%.4f" % (REF_IB[prop], REF_MS[prop])]
        for tgt in ("q50", "q90"):
            c = got.get((prop, tgt))
            if c is None:
                row.append("-")
                continue
            ib, ms = c["in_band_fraction"], c["mol_stability"]
            row.append("%.4f/%.4f" % (ib, ms))
            tot[tgt] += abs(ib - REF_IB[prop]) + abs(ms - REF_MS[prop])
        print("  %-6s %-18s %-18s %-18s" % (prop, row[0], row[1], row[2]))
    print("\n  L1 distance to the handoff's six numbers:  q50 %.4f   q90 %.4f"
          % (tot["q50"], tot["q90"]))
    print("  closer: %s" % ("q50" if tot["q50"] < tot["q90"] else "q90"))
    print("  margin: %.4f -- treat anything under ~0.05 as no discrimination,"
          % abs(tot["q50"] - tot["q90"]))
    print("  because n=256 gives se(in_band) ~ 0.02 on each of three numbers.")


if __name__ == "__main__":
    main()
