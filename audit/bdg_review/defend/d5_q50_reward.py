"""BLUE TEAM, objection "no project metric rewards batch-spread control".

The objection's cited evidence is all about `dist` (per-molecule targets) and
about q90.  BDG ran at q50 (data#1: chi2 1.4, p=0.96 vs q50; excluded from q90
at p~4e-4), and q50 is the project's own majority screen target
(759 q50 cells vs 381 q90 cells in results/sweep).

So: at q50, on the project's OWN primary metric (in_band), how much does the
spread lever pay AT THE SPREAD RATIOS BDG ACTUALLY REACHED (not the s=0.3
headline the red team quoted), against what the centring lever pays?

Read-only rescoring of real unguided v2 residuals (3 x n=5000 sidecars).
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
# handoff 6.1, seed1 / seed2 achieved sd/unguided
ACHIEVED = {"mu": {"tight": (0.739, 0.730), "wide": (1.168, 1.094)},
            "alpha": {"tight": (0.681, 0.654), "wide": (1.034, 1.042)},
            "gap": {"tight": (0.787, 0.742), "wide": (1.096, 1.117)}}

F = {}
for prop in ("mu", "alpha", "gap"):
    acc = []
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/full/v2/n5000/seed*/%s__unguided__q90__w1__tmin0.5__full.permol.pt" % prop))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        fin = d["finite"].numpy().astype(bool)
        acc.append(d["f_B_dec"].numpy()[fin])
    F[prop] = np.concatenate(acc)


def inband(r, delta):
    return float(np.mean(np.abs(r) <= delta))


def scale(r, b, s):
    return b + s * (r - b)


print("=" * 100)
print("D5  IN_BAND PAYOFF OF THE SPREAD LEVER AT BDG'S OWN ACHIEVED RATIOS")
print("    (pure spread intervention on real unguided residuals, n=15000 each)")
print("=" * 100)
print("%-6s %-5s %8s %8s | %9s %9s %9s | %9s %9s | %9s"
      % ("prop", "tgt", "|b|/sd", "in_band", "s=tight1", "s=tight2", "s=wide1",
         "d(tight1)", "d(wide1)", "d(debias)"))
rows = []
for prop in ("mu", "alpha", "gap"):
    for tname, y in TARGETS[prop].items():
        r = F[prop] - y
        b, sd, delta = r.mean(), r.std(ddof=1), DELTA[prop]
        base = inband(r, delta)
        t1, t2 = ACHIEVED[prop]["tight"]
        w1, _ = ACHIEVED[prop]["wide"]
        p_t1, p_t2, p_w1 = (inband(scale(r, b, s), delta) for s in (t1, t2, w1))
        p_db = inband(r - b, delta)
        rows.append((prop, tname, base, p_t1, p_db))
        print("%-6s %-5s %8.3f %8.4f | %9.4f %9.4f %9.4f | %+9.4f %+9.4f | %+9.4f"
              % (prop, tname, abs(b) / sd, base, p_t1, p_t2, p_w1,
                 p_t1 - base, p_w1 - base, p_db - base))

print()
print("RATIO of the two levers at BDG's own tightest achieved spread:")
for prop, tname, base, p_t1, p_db in rows:
    gs, gd = p_t1 - base, p_db - base
    rat = "inf (centring lever <= 0)" if gd <= 0 else "%.1fx" % (gs / gd)
    print("  %-6s %-5s spread %+0.4f  centring %+0.4f   spread/centring = %s"
          % (prop, tname, gs, gd, rat))

print()
print("Significance of the q50 spread gain at BDG's own n=512, one cell:")
for prop, tname, base, p_t1, p_db in rows:
    if tname != "q50":
        continue
    se = np.sqrt(base * (1 - base) / 512 + p_t1 * (1 - p_t1) / 512)
    print("  %-6s unguided %.4f -> tightened %.4f  diff %+0.4f  se(2-prop,n=512) %.4f"
          "  z = %+.2f" % (prop, base, p_t1, p_t1 - base, se, (p_t1 - base) / se))

print()
print("=" * 100)
print("D5b  IS THE SPREAD LEVER MONOTONE AT q50 OVER BDG'S WHOLE REACHABLE RANGE?")
print("=" * 100)
for prop in ("mu", "alpha", "gap"):
    y = TARGETS[prop]["q50"]
    r = F[prop] - y
    b, delta = r.mean(), DELTA[prop]
    ss = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.17]
    vals = [inband(scale(r, b, s), delta) for s in ss]
    mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
    print("  %-6s " % prop + " ".join("%.4f" % v for v in vals)
          + "   strictly decreasing in s: %s" % mono)
print("  s grid: " + " ".join("%6.2f" % s for s in ss))

print()
print("=" * 100)
print("D5c  BDG's MEASURED gap e4t0.5 vs the pure-spread prediction")
print("=" * 100)
r = F["gap"] - TARGETS["gap"]["q50"]
b, delta = r.mean(), DELTA["gap"]
print("  unguided in_band (v2 sidecars rescored at q50): %.4f" % inband(r, delta))
print("  pure-spread counterfactual at s=0.787         : %.4f"
      % inband(scale(r, b, 0.787), delta))
print("  BDG measured gap e4t0.5 (handoff 6.2)         : 0.1523")
print("  plug w=4 control (handoff 6.2)                : 0.1602")
print("  NOTE BDG's cells also move the centre (gpu#11: bias swings up to 1.8")
print("  delta along the ladder), so this is a consistency check, not a fit.")
