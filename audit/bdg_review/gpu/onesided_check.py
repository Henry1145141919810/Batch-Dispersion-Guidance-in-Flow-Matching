"""The one-sided ablation at CELL level, with the prediction stated first.

Handoff section 7 claims `e4t1.5o` "reproduces plug exactly -- clamped, asked
to widen, does nothing". That is only true if e stays NEGATIVE at every guided
step. The w_eff trajectory recorded on this generator says e at tau_mult 1.5
reaches +0.0306 on mu/q90 but stays in [-0.910, -0.314] on gap/q50, so the
prediction here is:

  gap q50   e < 0 at every step  ->  bit-identical to plug
  mu  q90   e > 0 at some steps  ->  NOT identical to plug

A single "reproduces plug exactly" claim cannot be true for both, so this
distinguishes "the clamp is inert at tau_mult 1.5" (what the handoff says)
from "the clamp is inert only while the batch is under the setpoint" (what
the mechanism says).
"""
import json
import os

import torch

CELLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "cells")


def get(prop, tgt, arm, variant):
    base = "%s__%s__%s__w%s__tmin0.5__%s" % (
        prop, arm, tgt, "0" if arm == "unguided" else "1", variant)
    c = json.load(open(os.path.join(CELLS, base + ".json")))
    pm = torch.load(os.path.join(CELLS, base + ".permol.pt"), weights_only=False)
    return c, pm


def main():
    print("  %-6s %-5s %-10s %-10s %-12s %-12s %s"
          % ("prop", "tgt", "in_b 1.5o", "in_b plug", "max|df_A|",
             "max|df_B|", "identical"))
    for tgt in ("q50", "q90"):
        for prop in ("mu", "gap"):
            a, pa = get(prop, tgt, "bdg", "e4t1.5o")
            b, pb = get(prop, tgt, "plug", "bdgctl")
            da = float((pa["f_A"] - pb["f_A"]).abs().max())
            db = float((pa["f_B"] - pb["f_B"]).abs().max())
            print("  %-6s %-5s %-10.4f %-10.4f %-12.3e %-12.3e %s"
                  % (prop, tgt, a["in_band_fraction"], b["in_band_fraction"],
                     da, db, da == 0.0 and db == 0.0))
            print("      run-mean e used %+.6f (raw %+.6f), w_eff %+.6f"
                  % (a["diag"]["bdg_e"], a["diag"]["bdg_e_raw"],
                     a["diag"]["bdg_w_eff"]))


if __name__ == "__main__":
    main()
