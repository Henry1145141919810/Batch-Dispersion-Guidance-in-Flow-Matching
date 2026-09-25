"""Under the v1 `dist` protocol (per-molecule targets), is the BATCH spread of
f_B the right thing to control?

BDG servos V_b = Var_i(F_i) to a setpoint tau^2. Under `dist` the targets y_i
vary, so a method that matched the conditional law would need Var(F) ~ Var(y)
PLUS the per-molecule error -- V_b is not a residual statistic at all.

Counterfactual on the real v1 n=5000 sidecars (unguided and plug w=4):
    F -> Fbar + s (F - Fbar)      in_band(s) = mean(|F(s) - y| <= delta)
Also reports Var(y), Var(F), corr(F, y) -- the quantity that actually decides
conditional coverage, which no batch-variance setpoint can control.
"""
import glob
import json
import os

import numpy as np
import torch

REPO = "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
DELTA = {"mu": 0.16799, "alpha": 0.48135, "gap": 0.00760}
SCALES = np.round(np.concatenate([np.arange(0.10, 1.00, 0.05),
                                  np.arange(1.00, 3.01, 0.10)]), 4)

rows = []
for pat, tag in (("*__unguided__dist__*full.permol.pt", "unguided"),
                 ("*__plug__dist__w4__*full.permol.pt", "plug w=4")):
    for f in sorted(glob.glob(os.path.join(REPO, "results/full/n5000/seed*", pat))):
        prop = os.path.basename(f).split("__")[0]
        d = torch.load(f, map_location="cpu", weights_only=False)
        fin = d["finite"].numpy().astype(bool)
        F = d["f_B"].numpy()[fin].astype(np.float64)
        y = d["y"].numpy()[fin].astype(np.float64)
        delta = DELTA[prop]
        Fb = F.mean()
        base = float(np.mean(np.abs(F - y) <= delta))
        curve = [(float(s), float(np.mean(np.abs(Fb + s * (F - Fb) - y) <= delta)))
                 for s in SCALES]
        s_opt, p_opt = max(curve, key=lambda t: t[1])
        rows.append(dict(arm=tag, prop=prop, n=int(F.size),
                         sd_F=float(F.std(ddof=1)), sd_y=float(y.std(ddof=1)),
                         corr=float(np.corrcoef(F, y)[0, 1]),
                         sd_resid=float((F - y).std(ddof=1)),
                         bias_over_delta=float((F - y).mean() / delta),
                         sd_resid_over_delta=float((F - y).std(ddof=1) / delta),
                         in_band=base, s_opt=s_opt, in_band_at_s_opt=p_opt,
                         gain=p_opt - base,
                         in_band_perfect_corr=float(
                             np.mean(np.abs((F - Fb) - (y - y.mean())) <= delta))))

print("arm        prop   n     sd_F    sd_y   corr(F,y)  sd_res/delta  bias/delta  "
      "in_band   s*    inb@s*   gain    inb if F-Fbar == y-ybar")
for r in rows:
    print("%-10s %-5s %5d %7.3f %7.3f %9.3f %13.2f %11.2f %8.4f %5.2f %8.4f %+7.4f %12.4f"
          % (r["arm"], r["prop"], r["n"], r["sd_F"], r["sd_y"], r["corr"],
             r["sd_resid_over_delta"], r["bias_over_delta"], r["in_band"],
             r["s_opt"], r["in_band_at_s_opt"], r["gain"], r["in_band_perfect_corr"]))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist_lever.json")
json.dump(rows, open(out, "w"), indent=1)
print("\nwrote", out)
