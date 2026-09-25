"""BLUE TEAM: does the project's PRIMARY ranking metric reward spread control?

docs/results/ARM_RANKINGS.md: "Ordered by MAE/delta, lowest first, per property.
MAE is the primary metric ... band coverage is the acceptance criterion but a
binomial with se ~0.013 at n=512, too weak to rank on."  Its columns include
  |bias| = (mean f_B - target)/delta   "is the distribution centred?"
  spread = sqrt(RMSE^2 - bias^2)/delta "is it concentrated?"

So measure MAE/delta under a PURE spread intervention, at both targets.
"""
import glob
import os

import numpy as np
import torch

REPO = "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
TARGETS = {"mu": {"q50": 2.4932, "q90": 4.6627},
           "alpha": {"q50": 75.54, "q90": 85.05},
           "gap": {"q50": 0.2496, "q90": 0.3162}}
DELTA = {"mu": 0.16799, "alpha": 0.48135, "gap": 0.00760}
SS = [0.65, 0.70, 0.80, 0.90, 1.00, 1.10, 1.17]

F = {}
for prop in ("mu", "alpha", "gap"):
    acc = []
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/full/v2/n5000/seed*/%s__unguided__q90__w1__tmin0.5__full.permol.pt" % prop))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        fin = d["finite"].numpy().astype(bool)
        acc.append(d["f_B_dec"].numpy()[fin])
    F[prop] = np.concatenate(acc)

print("MAE/delta under a pure spread intervention r -> b + s(r-b)   (n=15000)")
print("%-6s %-5s " % ("prop", "tgt") + " ".join("s=%.2f" % s for s in SS)
      + "   monotone down in s?")
for prop in ("mu", "alpha", "gap"):
    for tname, y in TARGETS[prop].items():
        r = F[prop] - y
        b, delta = r.mean(), DELTA[prop]
        vals = [float(np.mean(np.abs(b + s * (r - b)))) / delta for s in SS]
        mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
        print("%-6s %-5s " % (prop, tname) + " ".join("%5.2f" % v for v in vals)
              + "   %s   (|b|/delta %.2f)" % (mono, abs(b) / delta))

print()
print("Relative MAE reduction from s=1.00 to BDG's achieved tightest:")
ACH = {"mu": 0.739, "alpha": 0.681, "gap": 0.787}
for prop in ("mu", "alpha", "gap"):
    for tname, y in TARGETS[prop].items():
        r = F[prop] - y
        b, delta = r.mean(), DELTA[prop]
        m1 = float(np.mean(np.abs(r))) / delta
        ms = float(np.mean(np.abs(b + ACH[prop] * (r - b)))) / delta
        print("  %-6s %-5s MAE/delta %5.2f -> %5.2f at s=%.3f   (%+.1f%%)"
              % (prop, tname, m1, ms, ACH[prop], 100 * (ms - m1) / m1))
