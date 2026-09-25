"""BLUE TEAM, objection "no single-molecule mode; the method forces batch = n".

D1  B=1 limit: is BDG undefined, or does it reduce to plug?
D2  Does the controller transfer across batch size at FIXED eta?
D3  The handoff's own stated worry: "the usual 128-in-512 would run four
    independent controllers."  Does that break the setpoint?
D4  Estimator-noise rectification at B=128 vs B=512.

Loop model is the one the math lens calibrated to the handoff's OWN 6.1 ladder
(scratchpad/bdg/math/h_calibrate.py, rms 0.04-0.06 over 10 ladder points):
    kappa_n = base*(1-t)/t,  F <- F + kappa*[(y-F) - eta*e*(F-Fbar)],
    e = (V_b - tau^2)/tau^2,  V_b = sample variance (ddof=1) OF THE BATCH.
Fitted kappa_peak: mu 0.0020, alpha 0.0030, gap 0.0010.
"""
import numpy as np

ts = np.linspace(0, 1, 101)[:-1]
ts = ts[ts >= 0.5]                      # 50 guided steps, t_min_guide=0.5
ETA = 4.0
BASE = {"mu": 0.0020, "alpha": 0.0030, "gap": 0.0010}
SD_UNG = {"mu": 1.001, "alpha": 0.982, "gap": 1.00}

print("=" * 78)
print("D1  THE B=1 LIMIT -- is there no single-molecule mode?")
print("=" * 78)
print("spec (handoff S2):  num_i = (y - F_i) - eta*e*(F_i - F_bar)")
print("At B=1, F_bar == F_1 identically, so the dispersion term is exactly 0")
print("whatever e is.  BDG at B=1 IS plug, not undefined.\n")
rng = np.random.default_rng(0)
for B in (1, 2, 3, 8):
    F = rng.normal(0.0, 1.0, size=B)
    y, tau2 = 0.3, 0.7
    Fb = F.mean()
    V = F.var(ddof=1) if B > 1 else 0.0          # the one-line guard
    e = (V - tau2) / tau2
    num_bdg = (y - F) - ETA * e * (F - Fb)
    num_plug = (y - F)
    print("  B=%-4d V_b=%-10.6f e=%+8.4f  max|num_bdg - num_plug| = %.3e"
          % (B, V, e, np.abs(num_bdg - num_plug).max()))
print("  -> B=1: dispersion term = 0 to 0.0e+00, i.e. BIT-identical to plug.")
print("  -> B=2 already gives a live 1-dof controller (V_b defined, ddof=1).")
print("\n  The unguarded REDUCED form w_eff*(y_eff - F_i) does NaN at B=1")
print("  (V_b = 0/0), which is the whole content of 'V_b is 0/0 at B=1':")
F1 = np.array([0.4])
V_unguarded = ((F1 - F1.mean()) ** 2).sum() / (len(F1) - 1)   # 0/0
print("    unguarded V_b at B=1 = %s ; num via spec form = %+.6f (finite)"
      % (V_unguarded, float(((0.3 - F1) - ETA * 0.0 * (F1 - F1.mean()))[0])))

print()
print("=" * 78)
print("D2  DOES THE CONTROLLER TRANSFER ACROSS BATCH SIZE AT FIXED eta?")
print("=" * 78)
print("The 2/(B-1) is absorbed into eta, and neither w_eff = 1+eta*e nor")
print("kappa contains B, so the MEAN-FIELD loop is exactly B-free; B enters")
print("only through relse(V_b) = sqrt(2/(B-1)).  Measured, 400 replicates:")


def run(B, base, tau_mult, sd_ung, rng, n_ctrl=1):
    """n_ctrl independent controllers, B molecules each. Returns pooled sd ratio."""
    F = rng.normal(0.0, sd_ung, size=(n_ctrl, B))
    sd0 = F.reshape(-1).std(ddof=1)
    tau2 = tau_mult ** 2
    for t in ts:
        kap = base * (1 - t) / t
        Fb = F.mean(axis=1, keepdims=True)
        V = F.var(axis=1, ddof=1, keepdims=True)
        e = (V - tau2) / tau2
        F = F + kap * ((0.0 - F) - ETA * e * (F - Fb))
    Fl = F.reshape(-1)
    within = np.sqrt(np.mean([f.var(ddof=1) for f in F]))
    return Fl.std(ddof=1) / sd0, within / sd0


REP = 400
for prop in ("mu", "alpha", "gap"):
    base, sdu = BASE[prop], SD_UNG[prop]
    tgt = {"mu": [(0.5, 0.739), (1.5, 1.168)],
           "alpha": [(0.5, 0.681), (1.5, 1.034)],
           "gap": [(0.5, 0.787), (1.5, 1.096)]}[prop]
    for tm, measured in tgt:
        line = []
        for B in (32, 64, 128, 256, 512, 4096):
            rng = np.random.default_rng(7)
            r = [run(B, base, tm, sdu, rng)[0] for _ in range(REP)]
            line.append((B, np.mean(r), np.std(r, ddof=1)))
        print("  %-5s tau_mult %.2f (handoff measured %.3f at B=512):" % (prop, tm, measured))
        print("        " + "  ".join("B=%d %.4f+-%.4f" % (B, m, s) for B, m, s in line))
        m512 = [m for B, m, s in line if B == 512][0]
        print("        max |mean(B) - mean(512)| over B in [32,4096] = %.4f"
              % max(abs(m - m512) for B, m, s in line))

print()
print("=" * 78)
print("D3  4 CONTROLLERS x 128  vs  1 CONTROLLER x 512   (handoff lines 181-182)")
print("=" * 78)
print("Same 512 molecules either way. 'four independent controllers' is the")
print("stated reason batch must equal n. Measured pooled sd/unguided:")
for prop in ("mu", "alpha", "gap"):
    base, sdu = BASE[prop], SD_UNG[prop]
    for tm in (0.5, 1.0, 1.5):
        rng = np.random.default_rng(11)
        one = [run(512, base, tm, sdu, rng, n_ctrl=1) for _ in range(REP)]
        rng = np.random.default_rng(11)
        four = [run(128, base, tm, sdu, rng, n_ctrl=4) for _ in range(REP)]
        o, f = np.array([a for a, b in one]), np.array([a for a, b in four])
        ow, fw = np.array([b for a, b in one]), np.array([b for a, b in four])
        print("  %-5s t%.2f  1x512 pooled %.4f+-%.4f | 4x128 pooled %.4f+-%.4f"
              "  delta %+.4f (%.2f%%)  within-batch 4x128 %.4f"
              % (prop, tm, o.mean(), o.std(ddof=1), f.mean(), f.std(ddof=1),
                 f.mean() - o.mean(), 100 * (f.mean() - o.mean()) / o.mean(), fw.mean()))

print()
print("=" * 78)
print("D4  ESTIMATOR-NOISE RECTIFICATION AT B=128 vs 512")
print("=" * 78)
print("relse(V_b) = sqrt(2/(B-1)):  B=128 %.4f   B=512 %.4f   ratio %.2fx"
      % (np.sqrt(2 / 127), np.sqrt(2 / 511), np.sqrt(511 / 127)))
print("Spurious widening from E[rho^2] = rho0^2 + (kappa*eta*u*relse)^2, 50 steps:")
for prop in ("mu", "alpha", "gap"):
    base = BASE[prop]
    for B in (128, 512):
        relse = np.sqrt(2 / (B - 1))
        acc = 0.0
        for t in ts:
            kap = base * (1 - t) / t
            acc += 0.5 * (kap * ETA * 0.75 * relse) ** 2      # ln(1+x) ~ x, per step
        print("  %-5s B=%-4d spurious log-widening over the window = %+.3e"
              "  (%.4f%% on sd)" % (prop, B, acc, 100 * (np.exp(acc) - 1)))
