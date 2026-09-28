"""BLUE-TEAM check 3: is 'feedback matters in the widening branch, not in the
tightening branch' a PREDICTION of the loop model, or a post-hoc split of 8 cells?

Toy calibrated to what the GPU trajectories show for gap (the monotone case):
  * early per-step gain k ~ 0.10, decaying to ~0.005 late (d2_loop_gain.py);
  * the flow's own across-batch spread GROWS over the window (V_b/s^2 ~0.2 at
    t=0.5 -> ~1 at t=1; weff_traj.json gap V/tau^2 starts at 0.09 at tau=1.5 s),
    modelled as idiosyncratic per-molecule increments;
  * B = 256, heavy-ish tails (student-t, 6 dof) for the initial spread.
Deviation dynamics (the only part that sets the spread; section 3 reduction):
    d_i <- d_i * (1 - k_n * w_n) + sig * xi_i
closed loop: w_n = 1 + eta*(V_n/tau^2 - 1); replay: w_n taken from batch A's log.
For many (A, B) batch pairs report
    closed-loop seed-to-seed   |sd_C(B) - sd_C(A)| / sd
    open-loop   seed-to-seed   |sd_R(B) - sd_C(A)| / sd     (same schedule, new noise)
    paired                     |sd_R(B) - sd_C(B)| / sd
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ETA, B, N, NPAIR = 4.0, 256, 50, 2000
n = np.arange(N)


def run(k0, tau_mult, seed_pairs=NPAIR, sig=0.126, sd0=0.45, decay=8.0):
    rng = np.random.default_rng(12345)
    k = k0 * np.exp(-n / decay) + 0.004
    tau2 = tau_mult ** 2
    out = {"ca": [], "ra": [], "rc": [], "sdC": [], "w0": []}
    for _ in range(seed_pairs):
        dA = sd0 * rng.standard_t(6, size=B) / np.sqrt(6 / 4)
        dB = sd0 * rng.standard_t(6, size=B) / np.sqrt(6 / 4)
        xiA = rng.normal(size=(N, B)); xiB = rng.normal(size=(N, B))

        def closed(d, xi):
            d = d - d.mean(); ws = []
            for j in range(N):
                V = d.var(ddof=1)
                w = 1 + ETA * (V / tau2 - 1)
                ws.append(w)
                d = d * (1 - k[j] * w) + sig * xi[j]
                d = d - d.mean()
            return d.std(ddof=1), np.array(ws)

        def replay(d, xi, ws):
            d = d - d.mean()
            for j in range(N):
                d = d * (1 - k[j] * ws[j]) + sig * xi[j]
                d = d - d.mean()
            return d.std(ddof=1)

        sCA, wA = closed(dA, xiA)
        sCB, _ = closed(dB, xiB)
        sRB = replay(dB, xiB, wA)
        out["ca"].append(abs(sCB - sCA) / sCA)
        out["ra"].append(abs(sRB - sCA) / sCA)
        out["rc"].append(abs(sRB - sCB) / sCB)
        out["sdC"].append(sCA); out["w0"].append(wA.mean())
    return {kk: float(np.mean(v)) for kk, v in out.items()}


res = {}
print("%-6s %-5s | %-8s | %-9s %-9s %-9s | %s"
      % ("k0", "tau", "mean w", "closed", "open", "paired", "open/closed"))
for k0 in (0.10, 0.30):
    for tm in (0.5, 1.0, 1.5):
        r = run(k0, tm, seed_pairs=600)
        res["k%.2f_t%.1f" % (k0, tm)] = r
        print("%-6.2f %-5.1f | %+7.3f  | %.4f    %.4f    %.4f    | %.2fx"
              % (k0, tm, r["w0"], r["ca"], r["ra"], r["rc"], r["ra"] / r["ca"]))
json.dump(res, open(os.path.join(HERE, "d3_branch_toy.json"), "w"), indent=1)
print("\nclosed = mean |sd_C(B)-sd_C(A)|/sd ; open = mean |sd_R(B)-sd_C(A)|/sd ;"
      " paired = mean |sd_R(B)-sd_C(B)|/sd")
