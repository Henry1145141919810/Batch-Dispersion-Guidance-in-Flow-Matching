"""Does controlling the batch SPREAD of f_B buy in-band coverage?

Read-only counterfactual on the v2 n=5000 unguided sidecars. The unguided
generator does not depend on the target, so the SAME samples can be scored at
q50 and at q90.  Intervention on the real residuals r_i = f_B,i - y:

    tighten/widen(s) : r -> b + s*(r - b)     (b = mean residual; s<1 tightens)
    debias(c)        : r -> c*b + (r - b)     (c=0 = perfectly centred)

This is a re-scoring of an observed property law, not a sampler run.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

REPO = "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
TARGETS = {"mu": {"q50": 2.4932, "q90": 4.6627},
           "alpha": {"q50": 75.54, "q90": 85.05},
           "gap": {"q50": 0.2496, "q90": 0.3162}}
DELTA = {"mu": 0.16799, "alpha": 0.48135, "gap": 0.00760}
SCALES = np.round(np.concatenate([np.arange(0.10, 1.00, 0.05),
                                  np.arange(1.00, 3.01, 0.10)]), 4)

rows = []
for prop in ("mu", "alpha", "gap"):
    F = []
    for f in sorted(glob.glob(os.path.join(
            REPO, "results/full/v2/n5000/seed*/%s__unguided__q90__w1__tmin0.5__full.permol.pt" % prop))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        fin = d["finite"].numpy().astype(bool)
        F.append(d["f_B_dec"].numpy()[fin])
    F = np.concatenate(F)
    delta = DELTA[prop]
    for tname, y in TARGETS[prop].items():
        r = F - y
        b = r.mean()
        sd = r.std(ddof=1)
        base = float(np.mean(np.abs(r) <= delta))
        curve = [(float(s), float(np.mean(np.abs(b + s * (r - b)) <= delta)))
                 for s in SCALES]
        s_opt, p_opt = max(curve, key=lambda t: t[1])
        # perfect centring, spread untouched
        p_debias = float(np.mean(np.abs(r - b) <= delta))
        rows.append(dict(prop=prop, target=tname, n=int(F.size), y=y,
                         bias=float(b), sd=float(sd),
                         bias_over_delta=float(b / delta),
                         sd_over_delta=float(sd / delta),
                         delta_over_sd=float(delta / sd),
                         in_band=base, s_opt=s_opt, in_band_at_s_opt=p_opt,
                         gain_spread=p_opt - base,
                         in_band_debiased=p_debias,
                         gain_debias=p_debias - base,
                         tighten_best=float(max(p for s, p in curve if s < 1.0)),
                         widen_best=float(max(p for s, p in curve if s > 1.0)),
                         in_band_s0p65=float(np.mean(np.abs(b + 0.65 * (r - b)) <= delta)),
                         in_band_s1p17=float(np.mean(np.abs(b + 1.17 * (r - b)) <= delta))))

hdr = ("prop  target  n      bias/delta  sd/delta  delta/sd  in_band  "
       "s*      inb@s*   d(spread)  inb(debias)  d(debias)  "
       "inb@s=0.65  inb@s=1.17")
print(hdr)
for r in rows:
    print("%-5s %-6s %6d %10.2f %9.2f %9.3f %8.4f %7.2f %8.4f %+10.4f %12.4f %+10.4f %11.4f %11.4f"
          % (r["prop"], r["target"], r["n"], r["bias_over_delta"], r["sd_over_delta"],
             r["delta_over_sd"], r["in_band"], r["s_opt"], r["in_band_at_s_opt"],
             r["gain_spread"], r["in_band_debiased"], r["gain_debias"],
             r["in_band_s0p65"], r["in_band_s1p17"]))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spread_lever.json")
json.dump(rows, open(out, "w"), indent=1)
print("\nwrote", out)
