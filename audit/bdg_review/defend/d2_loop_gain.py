"""BLUE-TEAM check 2: what per-step loop gain do the MEASURED trajectories imply?

The objection leans on math#18: 'a fitted loop gain predicts the frozen control
within ~7%, not a 6x log-widening gap'. That fit (h_calibrate.py) found
kappa_peak = 0.001-0.003 by matching terminal sd ratios with a model that starts
the batch at its terminal unguided spread and has no flow drift.

The GPU lens logged V_b/tau^2 and w_eff at every guided step on the real
generator (weff_traj.json, n=256, seed 20260925). Under the loop model
    V_{n+1} = V_n * (1 - k_n * w_n)^2 * D_n           (D_n = flow's own drift)
    dlogV_n = 2*log|1 - k_n w_n| + log D_n  ~=  -2 k w_n + d   (locally)
so an OLS fit of dlogV on w over a short early window estimates k with the
drift absorbed in the intercept. A period-2 alternation of the sign of w with V
swinging 0.45 <-> 0.85 is impossible at k = 0.002 (max per-step |dlogV| ~ 0.6%).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = json.load(open(os.path.join(os.path.dirname(HERE), "weff_traj.json")))

print("%-4s %-4s %-5s | %-10s %-10s %-8s | %-12s | %s"
      % ("prop", "tgt", "tau", "k_hat(1-20)", "k_hat(21-49)", "R2 early", "max|dlogV|", "vs k=0.002 model max"))
out = []
for r in T:
    w = np.array([x["bdg_w_eff"] for x in r["traj"]])
    V = np.array([x["bdg_V_over_tau2"] for x in r["traj"]])
    dl = np.diff(np.log(V))
    res = []
    for lo, hi in ((0, 20), (20, 49)):
        X = np.c_[w[lo:hi], np.ones(hi - lo)]
        coef, *_ = np.linalg.lstsq(X, dl[lo:hi], rcond=None)
        pred = X @ coef
        ss = ((dl[lo:hi] - dl[lo:hi].mean()) ** 2).sum()
        r2 = 1 - ((dl[lo:hi] - pred) ** 2).sum() / ss if ss > 0 else float("nan")
        res.append((-coef[0] / 2.0, r2))
    mx = np.abs(dl[:20]).max()
    model_mx = 2 * 0.002 * np.abs(w[:20]).max()
    out.append(dict(prop=r["prop"], tgt=r["target"], tau=r["tau_mult"],
                    k_early=res[0][0], r2_early=res[0][1], k_late=res[1][0],
                    max_dlogV=mx, model_max=model_mx))
    print("%-4s %-4s %-5g | %+9.3f  %+9.3f   %6.2f   | %8.3f     | %.4f  (ratio %.0fx)"
          % (r["prop"], r["target"], r["tau_mult"], res[0][0], res[1][0], res[0][1],
             mx, model_mx, mx / model_mx))
json.dump(out, open(os.path.join(HERE, "d2_loop_gain.json"), "w"), indent=1)
print("\nk_hat is the per-step fractional change in sd per unit w_eff (drift in the"
      "\nintercept). The math-lens fit that predicts '~7%' uses k = 0.001-0.003.")
