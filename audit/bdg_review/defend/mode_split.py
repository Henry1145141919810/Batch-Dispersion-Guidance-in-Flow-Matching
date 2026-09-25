"""Blue-team check: is BDG's widening 'negative-weight plug'?

The objection reads the section-3 factorisation num_i = w_eff*(y_eff - F_i) and
concludes that a cell with w_eff < 0 IS plug at a negative weight. That reading
drops y_eff. Rewrite the same numerator in batch-mean / deviation coordinates:

    num_i = (y - F_bar) - w_eff * (F_i - F_bar)          [BDG]
    num_i = w*(y - F_bar) - w * (F_i - F_bar)            [plug at weight w]

BDG's batch-MEAN coefficient is +1 for every tau and eta; only the DEVIATION
coefficient is w_eff. Plug locks the two at the same w, so plug at w < 0 pushes
the batch mean AWAY from y. BDG at w_eff < 0 still pulls it toward y at unit
weight. The two are only equal in the deviation mode.

Part A proves the identity numerically. Part B tests the prediction on the GPU
lens's already-sampled cells (n=256, seed 20260925, one batch, paired noise):
no new sampling. If BDG widening were negative-weight plug, its bias would sit
BEYOND unguided (away from y); if the mode split holds, it sits near plug w=1.
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(os.path.dirname(HERE), "cells")
TRAJ = os.path.join(os.path.dirname(HERE), "weff_traj.json")
rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Part A
print("=" * 78)
print("PART A  mean/deviation split of the BDG numerator (algebra, numpy float64)")
worst_split = worst_mean = worst_yeff = 0.0
for _ in range(5000):
    B = int(rng.integers(2, 600))
    F = rng.normal(rng.normal(), abs(rng.normal()) + 1e-3, B)
    y = rng.normal() * 3
    eta = rng.uniform(0, 16)
    tau2 = rng.uniform(0.05, 5) * F.var(ddof=1)
    Fb = F.mean()
    V = F.var(ddof=1)
    e = (V - tau2) / tau2
    w_eff = 1 + eta * e
    num = (y - F) - eta * e * (F - Fb)
    split = (y - Fb) - w_eff * (F - Fb)
    s = np.abs(num).max()
    worst_split = max(worst_split, np.abs(num - split).max() / s)
    worst_mean = max(worst_mean, abs(num.mean() - (y - Fb)) / s)
    if abs(w_eff) > 1e-3:
        y_eff = (y + eta * e * Fb) / w_eff
        worst_yeff = max(worst_yeff, abs(w_eff * (y_eff - Fb) - (y - Fb)) / max(abs(y - Fb), 1e-12))
print("  max rel |num - [(y-Fbar) - w_eff (F-Fbar)]|        = %.2e" % worst_split)
print("  max rel |mean(num) - (y - Fbar)|  (any eta, tau)   = %.2e" % worst_mean)
print("  max rel |w_eff (y_eff - Fbar) - (y - Fbar)|        = %.2e" % worst_yeff)
print("  => mean coefficient of BDG is +1 always; plug at w has mean coefficient w.")

# ---------------------------------------------------------------- Part B
print("=" * 78)
print("PART B  existing GPU-lens cells, paired by initial noise (n=256, seed 20260925)")

traj = {(r["prop"], r["target"], r["tau_mult"]): r for r in json.load(open(TRAJ))}


def load(prop, tgt, arm, w, var):
    stem = "%s__%s__%s__w%s__tmin0.5__%s" % (prop, arm, tgt, w, var)
    j = json.load(open(os.path.join(CELLS, stem + ".json")))
    p = torch.load(os.path.join(CELLS, stem + ".permol.pt"), weights_only=False)
    return j, p


def boot_diff_se(a, b, n_boot=4000):
    """Paired bootstrap se of mean(a) - mean(b) (same molecule index = same noise)."""
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    return float(np.std(a[idx].mean(1) - b[idx].mean(1)))


out = {}
for tgt in ("q90", "q50"):
    for prop in ("gap", "mu", "alpha"):
        ju, pu = load(prop, tgt, "unguided", 0, "bdgctl")
        j1, p1 = load(prop, tgt, "plug", 1, "bdgctl")
        j4, p4 = load(prop, tgt, "plug", 4, "tgt")
        d = ju["delta"]
        y = float(pu["y"][0])
        fin = lambda p: p["finite"].numpy()
        FAu, FA1, FA4 = pu["f_A"].numpy(), p1["f_A"].numpy(), p4["f_A"].numpy()
        sdu = FAu.std(ddof=1)
        bu, b1, b4 = (FAu.mean() - y) / d, (FA1.mean() - y) / d, (FA4.mean() - y) / d
        # centring step of plug per unit weight, measured two ways
        step01 = b1 - bu
        step14 = (b4 - b1) / 3.0
        print("\n--- %s %s   delta %.5g   y %.4g   (bias of f_A, in delta units)" % (prop, tgt, d, y))
        print("  unguided  bias %+8.3f  sd/ug 1.000  molst %.4f" % (bu, ju["mol_stability"]))
        print("  plug w1   bias %+8.3f  sd/ug %.3f  molst %.4f   centring step/unit w (0->1) %+7.3f" % (
            b1, FA1.std(ddof=1) / sdu, j1["mol_stability"], step01))
        print("  plug w4   bias %+8.3f  sd/ug %.3f  molst %.4f   centring step/unit w (1->4) %+7.3f" % (
            b4, FA4.std(ddof=1) / sdu, j4["mol_stability"], step14))
        print("  %-7s %7s %8s %8s %8s %9s %9s %7s %6s" % (
            "cell", "w_eff", "bias", "retain", "plugLck", "obs-lock", "z(o-lock)", "sd/ug", "molst"))
        rows = []
        for tm in (0.5, 0.75, 1, 1.25, 1.5):
            var = "e4t%s" % tm
            jb, pb = load(prop, tgt, "bdg", 1, var)
            FA = pb["f_A"].numpy()
            weff = jb["diag"]["bdg_w_eff"]
            bb = (FA.mean() - y) / d
            retain = (bb - bu) / step01 if abs(step01) > 1e-9 else float("nan")
            # plug-locked prediction: mean coefficient = w_eff (linear in w through unguided)
            lock = bu + weff * step01
            se = boot_diff_se(FA / d, FAu / d)  # paired se of a bias difference
            z = (bb - lock) / se
            tr = traj.get((prop, tgt, float(tm)))
            wr = ("[%+.2f,%+.2f]" % (tr["w_eff_min"], tr["w_eff_max"])) if tr else ""
            print("  %-7s %+7.3f %+8.3f %8.2f %+8.3f %+9.3f %9.1f %7.3f %6.4f %s" % (
                var, weff, bb, retain, lock, bb - lock, z, FA.std(ddof=1) / sdu, jb["mol_stability"], wr))
            rows.append(dict(cell=var, w_eff=weff, bias=bb, retain=retain, lock=lock, se=se, z=z,
                             sd_ratio=float(FA.std(ddof=1) / sdu), molst=jb["mol_stability"],
                             w_eff_range=wr))
        # slopes: bias vs w_eff along the BDG ladder, vs plug's locked slope
        W = np.array([r["w_eff"] for r in rows] + [1.0])
        Bv = np.array([r["bias"] for r in rows] + [b1])
        S = np.array([r["sd_ratio"] for r in rows] + [FA1.std(ddof=1) / sdu])
        slope_b = np.polyfit(W, Bv, 1)[0]
        slope_s = np.polyfit(W, S, 1)[0]
        plug_sd_slope = (FA4.std(ddof=1) / sdu - 1.0) / 4.0
        print("  ladder OLS: d(bias)/d(w_eff) = %+.3f  vs plug centring step %+.3f  -> leakage %.2f of plug" % (
            slope_b, step01, slope_b / step01 if abs(step01) > 1e-9 else float("nan")))
        print("              d(sd/ug)/d(w_eff) = %+.3f  vs plug d(sd/ug)/dw (0->4) %+.3f" % (slope_s, plug_sd_slope))
        span_w = W.max() - W.min()
        print("  bias swing along ladder %.2f delta over w_eff span %.2f; plug-locked would swing %.2f delta" % (
            Bv.max() - Bv.min(), span_w, abs(step01) * span_w))
        out["%s_%s" % (prop, tgt)] = dict(bias_ug=bu, bias_plug1=b1, bias_plug4=b4, step01=step01,
                                          step14=step14, slope_bias=slope_b, slope_sd=slope_s,
                                          rows=rows, delta=d)

json.dump(out, open(os.path.join(HERE, "mode_split.json"), "w"), indent=1, default=float)
print("\nwrote", os.path.join(HERE, "mode_split.json"))
