"""Read the local BDG cells and answer the five questions the task asks.

Every number here is recomputed from the per-molecule sidecars, not copied
from the cell json, so a disagreement between the two is visible.

  sd ratio   sd(f_A) and sd(f_B) over the FINITE rows, divided by the SAME
             quantity on the unguided cell at the SAME target. BDG servos the
             batch variance of f_A at the endpoint estimate, so sd(f_A) is
             what the controller actually acts on and sd(f_B) is whether that
             control survives to the held-out evaluator. The handoff reports
             one "sd/unguided" without saying which; both are printed.
  bias       mean(f_B) - mean(y), in units of delta.
  se         sqrt(p(1-p)/n) on in_band. This is the INDEPENDENT-sample se and
             it is UNDERSTATED for the bdg rows: all 256 trajectories are
             coupled through F_bar and V_b (handoff open item 5).
"""
import glob
import json
import math
import os
import sys

import torch

CELLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cells")
ORDER = ["e0t1", "e4t0.5", "e4t0.75", "e4t1", "e4t1.25", "e4t1.5"]


def load():
    rows = {}
    for p in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
        with open(p) as fh:
            c = json.load(fh)
        pm_path = p[:-5] + ".permol.pt"
        pm = torch.load(pm_path, weights_only=False) if os.path.exists(pm_path) else None
        rows[(c["prop"], c["target_name"], c["arm"], c["variant"])] = (c, pm)
    return rows


def stats(c, pm):
    """Recomputed metrics. `fin` is the finite mask the evaluator also used."""
    fin = pm["finite"]
    fa, fb, y = pm["f_A"][fin], pm["f_B"][fin], pm["y"][fin]
    fbd = pm["f_B_dec"][fin]
    d = c["delta"]
    n = len(pm["finite"])
    ib = c["in_band_fraction"]
    return {
        "in_band": ib,
        "in_band_dec": c["in_band_fraction_dec"],
        "se": math.sqrt(max(ib * (1 - ib), 0) / n),
        "mae_over_delta": c["prop_mae_eval"] / d,
        "mae_dec_over_delta": c["prop_mae_eval_dec"] / d,
        "bias_over_delta": float((fb - y).mean()) / d,
        "bias_dec_over_delta": float((fbd - y).mean()) / d,
        "sd_fa": float(fa.std(unbiased=True)),
        "sd_fb": float(fb.std(unbiased=True)),
        "sd_fbd": float(fbd.std(unbiased=True)),
        "mol_stab": c["mol_stability"],
        "valid": c["validity"],
        "uniq": c["uniqueness_of_valid"],
        "nonfinite": c["n_nonfinite"],
        "clipped": c.get("clipped_sample_steps", 0),
        "delta": d,
    }


def main():
    rows = load()
    props = sorted({k[0] for k in rows})
    tgts = sorted({k[1] for k in rows})
    # FR3a-style floor, recomputed LOCALLY: 0.9 x this run's own unguided
    # mol_stability at the same target. The handoff's 0.362109375 is 0.9 x an
    # unguided screen from a DIFFERENT device, batch size and n, so it cannot
    # be used here.
    out = []
    for tgt in tgts:
        for prop in props:
            ug = rows.get((prop, tgt, "unguided", "bdgctl"))
            pl = rows.get((prop, tgt, "plug", "bdgctl"))
            if ug is None:
                continue
            us = stats(*ug)
            floor = 0.9 * us["mol_stab"]
            print("\n=== %s  target %s   delta %.5f   local floor (0.9 x unguided "
                  "mol_stab) %.4f" % (prop, tgt, us["delta"], floor))
            print("  %-9s %-7s %-7s %-7s %-7s %-7s %-7s %-7s %-6s %-6s %-6s %-5s %s"
                  % ("cell", "in_bnd", "in_b_dc", "MAE/d", "bias/d", "sdA/ug",
                     "sdB/ug", "sdBd/ug", "molst", "valid", "uniq", "nonf", "floor"))

            def line(name, st, diag=None):
                print("  %-9s %-7.4f %-7.4f %-7.3f %+-7.3f %-7.3f %-7.3f %-7.3f "
                      "%-6.4f %-6.4f %-6.4f %-5d %s"
                      % (name, st["in_band"], st["in_band_dec"],
                         st["mae_over_delta"], st["bias_over_delta"],
                         st["sd_fa"] / us["sd_fa"], st["sd_fb"] / us["sd_fb"],
                         st["sd_fbd"] / us["sd_fbd"], st["mol_stab"],
                         st["valid"], st["uniq"], st["nonfinite"],
                         "PASS" if st["mol_stab"] >= floor else "FAIL"))

            line("unguided", us)
            if pl:
                ps = stats(*pl)
                line("plug w1", ps)
            for v in ORDER:
                r = rows.get((prop, tgt, "bdg", v))
                if r is None:
                    continue
                line(v, stats(*r))
            # controller state
            print("  controller (run-mean over guided steps):")
            for v in ORDER:
                r = rows.get((prop, tgt, "bdg", v))
                if r is None:
                    continue
                dg = r[0].get("diag", {})
                print("    %-9s tau %.4f  V_b/tau^2 %9.4f  e %+10.4f  "
                      "w_eff %+8.3f  disp_rms %8.4f  dev_rms %8.4f  B %d"
                      % (v, r[0].get("bdg_tau", float("nan")),
                         dg.get("bdg_V_over_tau2", float("nan")),
                         dg.get("bdg_e", float("nan")),
                         dg.get("bdg_w_eff", float("nan")),
                         dg.get("bdg_disp_rms", float("nan")),
                         dg.get("bdg_dev_rms", float("nan")),
                         int(dg.get("bdg_batch", 0))))
            # Q(i) monotonicity of requested -> achieved
            ach = [(float(v.split("t")[1]),
                    stats(*rows[(prop, tgt, "bdg", v)])["sd_fa"] / us["sd_fa"])
                   for v in ORDER if v.startswith("e4") and (prop, tgt, "bdg", v) in rows]
            ach.sort()
            mono = all(b >= a for (_, a), (_, b) in zip(ach, ach[1:]))
            print("  Q(i) requested->achieved sd(f_A)/unguided: %s  MONOTONE=%s"
                  % (", ".join("t%g:%.3f" % x for x in ach), mono))
            out.append((prop, tgt, mono, ach))
    # Q(ii) eta=0 == plug at cell level
    print("\n=== Q(ii) bdg eta=0 vs plug w=1, cell level (same seed, same batch)")
    for tgt in tgts:
        for prop in props:
            a = rows.get((prop, tgt, "bdg", "e0t1"))
            b = rows.get((prop, tgt, "plug", "bdgctl"))
            if not a or not b:
                continue
            sa, sb = stats(*a), stats(*b)
            dfa = float((a[1]["f_A"] - b[1]["f_A"]).abs().max())
            dfb = float((a[1]["f_B"] - b[1]["f_B"]).abs().max())
            print("  %-6s %-4s in_band %.4f vs %.4f | max|df_A| %.3e  "
                  "max|df_B| %.3e  identical=%s"
                  % (prop, tgt, sa["in_band"], sb["in_band"], dfa, dfb,
                     dfa == 0.0 and dfb == 0.0))
    print("\nn = 256, one seed. se(in_band) ~ %.3f at p=0.1 and ~%.3f at p=0.5; "
          "differences below ~2 points are NOT resolvable, and the bdg rows'"
          % (math.sqrt(0.1 * 0.9 / 256), math.sqrt(0.25 / 256)))
    print("se is understated because the batch is coupled through F_bar/V_b.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
