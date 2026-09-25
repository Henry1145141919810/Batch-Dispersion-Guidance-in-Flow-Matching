"""What exactly separates BDG from the repo's own Three_New sec 5.2 controller?

Three_New 5.2 (equal mobility r_i): nu_i = b_mu + (b_V / (2V)) c_i,
  b_mu = eta_mu (y - mu), b_V = -eta_V (V - tau^2)_+   (one-sided, as written)
Delete (.)_+ -> "5.2-two-sided".  BDG: num_i = (y - F_i) - eta e (F_i - Fbar), e = V/tau^2 - 1.

Claims tested (CPU, numpy, no generator):
 [S1] 5.2's heterogeneous-r allocation divided by r_i is affine in F_i with batch-shared coefs.
 [S2] BDG == 5.2-two-sided with (a) the (V - tau'^2) error divided by a CONSTANT (tau^2) instead of
      the MEASURED V, and (b) the setpoint relabelled tau'^2 = tau^2 (1 - 1/eta).  Exact identity.
 [S3] 5.2-two-sided's dispersion gain has btvg's functional form (V - tau^2)/V (guidance.py:1494
      b = -0.5(1/tau^2 - 1/V)); BDG's is bounded: widening gain <= eta - 1.
 [S4] closed-loop fixed points: 5.2-two-sided -> V = tau^2; BDG -> tau^2 (1 - 1/eta).
 [S5] per-sample edit size near batch collapse: 5.2 ~ 1/sqrt(V) (diverges), BDG ~ sqrt(V) (vanishes).
 [S6] heterogeneous mobility correlated with deviation: 5.2's 2x2 solve hits b_mu exactly (no drag),
      BDG's mean is dragged (math#12).  A point in 5.2's favour.
"""
import numpy as np, json
rng = np.random.default_rng(20260925)
out = {}

# ---- S1 -----------------------------------------------------------------------------------
worst = 0.0
for _ in range(500):
    B = 64
    z = rng.normal(0, 1, B); mu = z.mean(); c = z - mu; V = np.mean(c**2)
    r = rng.uniform(0.2, 3.0, B); R = np.diag(r)
    C = np.vstack([np.ones(B) / B, 2 * c / B])
    b = np.array([rng.normal(), rng.normal()])
    nu = R @ C.T @ np.linalg.solve(C @ R @ C.T, b)
    coef = nu / r                       # displacement coefficient on M^-1 a_i
    A = np.vstack([np.ones(B), z]).T
    sol, *_ = np.linalg.lstsq(A, coef, rcond=None)
    worst = max(worst, np.max(np.abs(A @ sol - coef)) / np.max(np.abs(coef)))
out["S1_52_alloc_affine_rel_err"] = worst
print("[S1] 5.2 allocation / r_i affine in F_i (shared coefs): max rel err %.2e" % worst)

# ---- S2 -----------------------------------------------------------------------------------
worst = 0.0
for _ in range(2000):
    B = int(rng.integers(3, 600))
    F = rng.normal(rng.normal(), rng.uniform(.1, 3), B)
    y = rng.normal() * 3; eta = rng.uniform(1.01, 16); tau2 = rng.uniform(.05, 5)
    Fb = F.mean(); c = F - Fb; V = F.var(ddof=1); e = V / tau2 - 1
    bdg = (y - F) - eta * e * c
    tau2p = tau2 * (1 - 1 / eta)
    # 5.2-two-sided form: mean command + k(V) c_i with k = -g (V - tau'^2), g a gain
    g_const = eta / tau2                       # BDG's choice: constant normaliser
    s52_form = (y - Fb) - g_const * (V - tau2p) * c
    worst = max(worst, np.max(np.abs(bdg - s52_form)) / (1 + np.max(np.abs(bdg))))
out["S2_bdg_eq_52_const_gain_rel_err"] = worst
print("[S2] BDG == (y-Fbar) - (eta/tau^2)(V - tau^2(1-1/eta)) c_i : max rel err %.2e" % worst)
print("     5.2-two-sided is (y-Fbar)*eta_mu - (eta_V/(2V))(V - tau^2) c_i : the ONLY differences are")
print("     the normaliser (constant tau^2 vs measured V) and the setpoint label tau'^2 = tau^2(1-1/eta).")

# ---- S3 -----------------------------------------------------------------------------------
eta = 4.0; tau2 = 1.0; etaV = 2.0          # etaV=2 so both gains are 1 per unit relative error at V=tau^2... scale-free compare
rows = []
for u in [1e-8, 1e-4, 1e-2, 0.1, 0.444, 0.75, 1.0, 2.0, 4.0, 100.0]:
    V = u * tau2
    k52 = -(etaV / (2 * V)) * (V - tau2)          # coefficient on c_i, >0 means widen
    kbtvg = -0.5 * (1 / tau2 - 1 / V)             # guidance.py:1494 before clamp (tau2=1)
    kbdg = -(1 + eta * (V / tau2 - 1))            # = -w_eff
    rows.append((u, k52, kbtvg, kbdg))
    print("[S3] V/tau^2=%-8g  5.2-two-sided %+12.4g   btvg(raw) %+12.4g   BDG %+8.4f" % (u, k52, kbtvg, kbdg))
out["S3_gain_table"] = rows
print("     5.2-two-sided / btvg ratio is constant (same (V-tau^2)/V form):",
      set(round(r[1] / r[2], 9) for r in rows if abs(r[2]) > 1e-12))

# ---- S4 + S5 ------------------------------------------------------------------------------
def loop(mode, V0, tau2, eta=4.0, etaV=1.0, h=0.01, steps=40000, B=2000, seed=0):
    r = np.random.default_rng(seed)
    F = r.normal(0, np.sqrt(V0), B); y = 0.5
    maxedit = 0.0
    for _ in range(steps):
        Fb = F.mean(); c = F - Fb; V = np.mean(c**2)
        if mode == "bdg":
            nu = (y - F) - eta * (V / tau2 - 1) * c
        else:
            nu = etaV * (y - Fb) - (etaV / (2 * V)) * (V - tau2) * c
        maxedit = max(maxedit, np.max(np.abs(nu - nu.mean())))
        F = F + h * nu
    return np.mean((F - F.mean())**2) / tau2, maxedit

s4 = {}
for tau2, V0 in [(2.0, 1.0), (0.5, 1.0)]:
    v52, _ = loop("s52", V0, tau2); vb, _ = loop("bdg", V0, tau2)
    s4[f"tau2={tau2},V0={V0}"] = {"s52_two_sided_V_over_tau2": v52, "bdg_V_over_tau2": vb,
                                   "bdg_predicted": 1 - 1 / 4.0}
    print("[S4] tau^2=%.1f V0=%.1f: 5.2-two-sided V*/tau^2 = %.5f   BDG V*/tau^2 = %.5f (pred 0.75)"
          % (tau2, V0, v52, vb))
out["S4_fixed_points"] = s4

s5 = {}
for V0 in [1e-2, 1e-4, 1e-6]:
    r = np.random.default_rng(1); F = r.normal(0, np.sqrt(V0), 2000); Fb = F.mean(); c = F - Fb
    V = np.mean(c**2); tau2 = 1.0
    e52 = np.max(np.abs((1.0 / (2 * V)) * (V - tau2) * c))
    ebdg = np.max(np.abs((1 + 4.0 * (V / tau2 - 1)) * c))
    s5[str(V0)] = {"s52_max_dispersion_edit": e52, "bdg_max_dispersion_edit": ebdg}
    print("[S5] V/tau^2=%g  max |dispersion edit|: 5.2-two-sided %.3g   BDG %.3g" % (V0, e52, ebdg))
out["S5_edit_near_collapse"] = s5

# ---- S6 -----------------------------------------------------------------------------------
B = 512; r = np.random.default_rng(7)
z = r.normal(0, 1, B); mu = z.mean(); c = z - mu; V = np.mean(c**2)
a = np.exp(0.6 * c); a /= a.mean()             # mobility correlated with deviation (alpha-like)
y = mu + 0.3; tau2 = 0.5
bmu = 1.0 * (y - mu); bV = -1.0 * (V - tau2)
R = np.diag(a); C = np.vstack([np.ones(B) / B, 2 * c / B])
nu52 = R @ C.T @ np.linalg.solve(C @ R @ C.T, np.array([bmu, bV]))
num_bdg = (y - z) - 4.0 * (V / tau2 - 1) * c
nu_bdg = a * num_bdg * (np.mean(a * num_bdg) and 1.0)
out["S6"] = {"corr_a_c": float(np.corrcoef(a, c)[0, 1]),
             "s52_mean_move_vs_request": [float(nu52.mean()), bmu],
             "bdg_mean_move_vs_plug_request": [float(nu_bdg.mean()), float(np.mean(y - z))]}
print("[S6] corr(a,c)=%.2f  5.2 mean move %.4f vs request %.4f (exact) ;  BDG mean move %.4f vs (y-Fbar) %.4f"
      % (np.corrcoef(a, c)[0, 1], nu52.mean(), bmu, nu_bdg.mean(), np.mean(y - z)))

json.dump(out, open(__file__.replace(".py", ".json"), "w"), indent=1, default=float)
