"""BDG item 5: how noisy is V_b, and what does BDG's batch coupling do to the
standard error of in_band / sd computed ON a BDG batch (handoff open item 5)?"""
import glob
import os

import numpy as np
import torch

rng = np.random.default_rng(11)
REPO = "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"

# ------------------------------------------------ 5a: relative se of V_b
print("[5a] relative se of the n-1 variance estimator, MC with 200000 draws")
print("     Gaussian F: theory sqrt(2/(B-1)); general: sqrt((kurt-1)/B) approx")
for B in (2, 8, 64, 128, 512, 4096):
    X = rng.normal(size=(200000 if B <= 512 else 20000, B))
    v = X.var(1, ddof=1)
    print("     B=%5d  theory %.4f   MC %.4f" % (B, np.sqrt(2.0 / (B - 1)),
                                                 v.std() / v.mean()))
print("     B=1 : V_b is 0/0 -- UNDEFINED. BDG has no single-molecule mode at all.")

# ------------------------------------------------ 5b: real F is not Gaussian
print("\n[5b] but F_i is the property of GENERATED molecules, which is not Gaussian.")
print("     Real terminal distributions from results/full/v2/n5000 (decoded f_B):")
rows = []
for prop in ("mu", "alpha", "gap"):
    for arm in ("unguided", "plug", "tmpd"):
        pat = os.path.join(REPO, "results/full/v2/n5000/seed20261001",
                           "%s__%s__q90__*__full.permol.pt" % (prop, arm))
        fs = sorted(glob.glob(pat))
        if not fs:
            continue
        d = torch.load(fs[0], map_location="cpu")
        x = d["f_B_dec"][d["finite"]].double().numpy()
        x = x[np.isfinite(x)]
        m, sd = x.mean(), x.std(ddof=1)
        z = (x - m) / sd
        kurt = (z ** 4).mean()
        # se of s^2 for iid non-normal: sqrt((kurt - 1)/n)
        infl = np.sqrt((kurt - 1) / 2.0)
        rows.append((prop, arm, len(x), kurt, infl))
        print("     %-6s %-9s n=%4d  kurtosis %6.2f  se(V) inflation vs Gaussian "
              "x%.2f  -> rel.se at B=512 = %.4f"
              % (prop, arm, len(x), kurt, infl, np.sqrt((kurt - 1) / 512)))

# ------------------------------------------------ 5c: coupling / design effect
print("\n[5c] COUPLING. BDG's B samples share F_bar and V_b, so they are not iid.")
print("     Toy closed loop, homogeneous gain, 50 guided steps, replicated seeds.")


def simulate(B, eta, tau2, kappa0, nrep, coupled=True, delta=0.3, y=0.0,
             sd0=1.0, seed=0):
    g = np.random.default_rng(seed)
    ts = np.linspace(0.0, 1.0, 101)[:-1]
    ts = ts[ts >= 0.5]
    inb, sds, vpath = [], [], []
    for _ in range(nrep):
        F = g.normal(0.6, sd0, size=B)
        for t in ts:
            kap = kappa0 * (1 - t) / t
            Fb = F.mean()
            V = F.var(ddof=1)
            e = (V - tau2) / tau2 if coupled else 0.0
            F = F + kap * ((y - F) - eta * e * (F - Fb))
        inb.append(np.mean(np.abs(F - y) < delta))
        sds.append(F.std(ddof=1))
        vpath.append(F.var(ddof=1))
    return np.array(inb), np.array(sds), np.array(vpath)


for B in (128, 512):
    for lab, coupled in (("plug  (eta=0, iid given the batch)", False),
                         ("BDG   (eta=4, coupled)", True)):
        inb, sds, _ = simulate(B, 4.0, 0.45, 0.02, 400, coupled=coupled, seed=7)
        p = inb.mean()
        naive = np.sqrt(p * (1 - p) / B)
        true_se = inb.std(ddof=1)
        naive_sd = sds.mean() / np.sqrt(2.0 * (B - 1))
        print("     B=%4d %-36s in_band %.4f  naive se %.5f  TRUE se %.5f  "
              "design effect %.2fx" % (B, lab, p, naive, true_se, true_se / naive))
        print("           %-36s sd      %.4f  naive se %.5f  TRUE se %.5f  "
              "design effect %.2fx" % ("", sds.mean(), naive_sd, sds.std(ddof=1),
                                       sds.std(ddof=1) / naive_sd))

print("\n[5d] the two statistics move in OPPOSITE directions:")
print("     - sd  : the controller servos V_b, so replicate-to-replicate spread of")
print("             the achieved sd is SUPPRESSED -> the iid se OVERSTATES it, and")
print("             a z-score on 'sd' built from the iid formula is CONSERVATIVE.")
print("     - in_band : every molecule shares one realised (F_bar, V_b) path, a")
print("             common random effect -> the iid binomial se UNDERSTATES it.")
print("     handoff open item 5 asserts the z-scores are understated without")
print("     splitting these; that is right for in_band, wrong for sd.")

# ------------------------------------------------ 5e: noise -> bias in e
print("\n[5e] NOISE RECTIFICATION: e is used linearly, but the loop's response to")
print("     e is not symmetric once w_eff crosses 0 (the contraction factor is")
print("     rho = 1-kappa*w_eff and V is multiplied by rho^2, which is convex).")
for B in (64, 128, 512):
    relse = np.sqrt(2.0 / (B - 1))
    u = 0.75
    kap = 0.005
    eta = 4.0
    draws = u * (1.0 + relse * rng.normal(size=400000))
    rho = 1.0 - kap * (1.0 + eta * (draws - 1.0))
    print("     B=%4d  E[rho^2]-rho(E)^2 = %.3e  -> a spurious WIDENING drift of "
          "%.4f%% per step, %.2f%% over 50 steps"
          % (B, (rho ** 2).mean() - (1 - kap * (1 + eta * (u - 1))) ** 2,
             100 * ((rho ** 2).mean() / (1 - kap * (1 + eta * (u - 1))) ** 2 - 1),
             100 * (((rho ** 2).mean() / (1 - kap * (1 + eta * (u - 1))) ** 2) ** 50 - 1)))
print("     The rectification is positive (Jensen on rho^2), so an under-resolved")
print("     V_b makes the batch widen even at the setpoint. It scales as 1/B.")
