"""Questions (iii), (iv), (v) answered from the 48-cell local grid.

(iii) at q50 vs q90, does WIDENING raise or lower in_band?
      The handoff (section 1) says widening lowers coverage "by definition".
      results/btvg3_widening_sim.json predicts the opposite at q90, where the
      batch sits 7-19 delta below the target, so the band is far out in a tail
      and a wider law puts MORE mass in it.

(iv)  does TIGHTENING cost chemistry?
      Tested against the alternative that chemistry tracks |w_eff| -- how hard
      the guidance pushes -- rather than the direction of the spread request.
      Under BDG's own reduction those are different predictions: w_eff passes
      through ZERO inside the tau ladder, so |w_eff| is non-monotone in
      tau_mult while "tightening" is monotone.

(v)   does the dispersion term move the batch MEAN?
      It is mean-zero in F by construction, so the naive answer is no. But
      w_eff multiplies the CENTRING term too, so the answer should be yes.
"""
import glob
import json
import math
import os

import torch

CELLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "cells")
ORDER = ["e4t0.5", "e4t0.75", "e4t1", "e4t1.25", "e4t1.5"]


def load():
    rows = {}
    for p in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
        c = json.load(open(p))
        pm = torch.load(p[:-5] + ".permol.pt", weights_only=False)
        rows[(c["prop"], c["target_name"], c["arm"], c["variant"])] = (c, pm)
    return rows


def bias_d(c, pm):
    fin = pm["finite"]
    return float((pm["f_B"][fin] - pm["y"][fin]).mean()) / c["delta"]


def sd_d(c, pm):
    fin = pm["finite"]
    return float(pm["f_B"][fin].std(unbiased=True)) / c["delta"]


def main():
    R = load()
    props, tgts = ("mu", "gap", "alpha"), ("q50", "q90")
    n = 256
    print("(iii) WIDENING (tau_mult 1.5) vs the eta=0 control (== plug w=1)")
    print("      se(in_band) at n=256 is ~0.02, so single rows are NOT "
          "resolvable; the SIGN PATTERN across 3 properties is.")
    print("  %-6s %-5s %-9s %-9s %-9s %-9s %s"
          % ("prop", "tgt", "|bias|/d", "sd/d", "in_b base", "in_b wide", "delta_in_b"))
    agg = {"q50": [], "q90": []}
    for tgt in tgts:
        for prop in props:
            b = R[(prop, tgt, "bdg", "e0t1")]
            w = R[(prop, tgt, "bdg", "e4t1.5")]
            d = w[0]["in_band_fraction"] - b[0]["in_band_fraction"]
            agg[tgt].append(d)
            print("  %-6s %-5s %+9.2f %9.2f %9.4f %9.4f %+9.4f"
                  % (prop, tgt, bias_d(*b), sd_d(*b),
                     b[0]["in_band_fraction"], w[0]["in_band_fraction"], d))
    for tgt in tgts:
        a = agg[tgt]
        print("  %s: mean change %+0.4f, %d of 3 negative"
              % (tgt, sum(a) / len(a), sum(1 for x in a if x < 0)))
    print("  -> 'widening lowers coverage by definition' holds at q50 "
          "(%d/3 down, mean %+0.4f) and NOT at q90 (%d/3 down, mean %+0.4f)."
          % (sum(1 for x in agg["q50"] if x < 0), sum(agg["q50"]) / 3,
             sum(1 for x in agg["q90"] if x < 0), sum(agg["q90"]) / 3))

    print("\n(iv) CHEMISTRY vs the spread request, and vs |w_eff|")
    xs_t, xs_w, ys = [], [], []
    for tgt in tgts:
        print("  %s:" % tgt)
        for prop in props:
            cells = [(v, R[(prop, tgt, "bdg", v)]) for v in ORDER]
            print("    %-6s %s" % (prop, "  ".join(
                "%s w_eff%+6.2f ms%.4f" % (v.replace("e4t", "t"),
                                           c[0]["diag"]["bdg_w_eff"],
                                           c[0]["mol_stability"])
                for v, c in cells)))
            best = max(cells, key=lambda kv: kv[1][0]["mol_stability"])
            print("      best chemistry at %-8s (|w_eff| %.3f = smallest: %s)"
                  % (best[0], abs(best[1][0]["diag"]["bdg_w_eff"]),
                     abs(best[1][0]["diag"]["bdg_w_eff"])
                     == min(abs(c[0]["diag"]["bdg_w_eff"]) for _, c in cells)))
            for v, c in cells:
                xs_t.append(float(v.split("t")[1]))
                xs_w.append(abs(c[0]["diag"]["bdg_w_eff"]))
                ys.append(c[0]["mol_stability"])

    def corr(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        da = math.sqrt(sum((x - ma) ** 2 for x in a))
        db = math.sqrt(sum((y - mb) ** 2 for y in b))
        return num / (da * db)

    print("  over all %d bdg cells:" % len(ys))
    print("    corr(tau_mult,  mol_stability) = %+0.3f" % corr(xs_t, ys))
    print("    corr(|w_eff|,   mol_stability) = %+0.3f" % corr(xs_w, ys))

    print("\n(v) DOES THE DISPERSION TERM MOVE THE BATCH MEAN? "
          "(bias/delta across the tau ladder, w fixed at 1)")
    print("  %-6s %-5s %s" % ("prop", "tgt", "  ".join(
        "%-8s" % v.replace("e4t", "t") for v in ["e0t1"] + ORDER)))
    for tgt in tgts:
        for prop in props:
            vals = [bias_d(*R[(prop, tgt, "bdg", v)]) for v in ["e0t1"] + ORDER]
            print("  %-6s %-5s %s   swing %.2f delta"
                  % (prop, tgt, "  ".join("%+8.3f" % v for v in vals),
                     max(vals) - min(vals)))
    print("  the dispersion term is mean-zero in F by construction, so a "
          "swing of this size can only come through w_eff multiplying the "
          "CENTRING term -- i.e. through the section-3 reduction.")


if __name__ == "__main__":
    main()
