"""
Adversarial test of BDG_HANDOFF.md claims, novelty lens.

C1. The Sec.3 affine factorisation  (y-F_i) - eta*e*(F_i-Fbar) = (1+eta*e)*(y_eff - F_i).
C2. Does the repo's OWN prior spec (Three_New_Guidance §5.2, nu* = R C^T (C R C^T)^-1 b)
    also produce a per-sample coefficient AFFINE in F_i?  If yes, the "reduction" in §3 does
    not distinguish BDG from the one-sided proposal it says it improves on.
C3. Novelty claim (d): "the loop does not reduce to any fixed weight schedule".
    Handoff evidence = freeze w_eff at its TIME AVERAGE, run plain plug at that constant weight.
    Test instead:
      (a) replay the realised TIME-VARYING (w_eff(t), y_eff(t)) open-loop, SAME batch/seed
      (b) replay constant (mean w_eff, mean y_eff)
      (c) replay constant mean w_eff but ORIGINAL y   <-- what the handoff actually ran
      (d) replay realised time-varying schedule on a DIFFERENT batch/seed
    (a) exact  => "no fixed schedule reproduces it" is FALSE as literally written for a
    deterministic sampler; the defensible claim is (d): the schedule does not TRANSFER.
    (c) vs (b) separates "feedback" from "the target shift y_eff", which the handoff's
    control confounds.
"""
import numpy as np

rng_global = np.random.default_rng(0)
D = 6                      # toy state dim
B = 64                     # batch
NSTEP = 60
ETA = 4.0
S2 = 1.0

Wf = rng_global.normal(size=D) / np.sqrt(D)   # nonlinear property f(x) = sum(Wf*tanh(x))
A = rng_global.normal(size=(D, D)) / np.sqrt(D)


def f_and_grad(m):
    th = np.tanh(m)
    F = th @ Wf
    g = (1.0 - th ** 2) * Wf          # (B,D)
    return F, g


def vfield(x, t):
    return -0.8 * x + 0.5 * np.tanh(x @ A)


def roll(x0, y, tau2, mode, sched=None):
    """mode: 'bdg' closed loop; 'open' replay sched=(w_eff[t], y_eff[t])."""
    x = x0.copy()
    rec_w, rec_y, rec_e = [], [], []
    for k in range(NSTEP):
        t = 1.0 - k / NSTEP
        v = vfield(x, t)
        m = x + (1.0 - t) * v
        F, g = f_and_grad(m)
        if mode == 'bdg':
            Fbar = F.mean()
            Vb = F.var(ddof=1)
            e = (Vb - tau2) / tau2
            num = (y - F) - ETA * e * (F - Fbar)
            w_eff = 1.0 + ETA * e
            y_eff = (y + ETA * e * Fbar) / w_eff
            rec_w.append(w_eff); rec_y.append(y_eff); rec_e.append(e)
            # consistency of the factorisation, this step
            assert np.max(np.abs(num - w_eff * (y_eff - F))) < 1e-12
        else:
            w_eff, y_eff = sched[0][k], sched[1][k]
            num = w_eff * (y_eff - F)
        push = (num / S2)[:, None] * g
        x = x + (1.0 / NSTEP) * (v + push)
    F, _ = f_and_grad(x + 0.0)
    return x, F, (np.array(rec_w), np.array(rec_y), np.array(rec_e))


def stats(F):
    return dict(mean=float(F.mean()), sd=float(F.std(ddof=1)))


print("=" * 78)
print("C1 + C2: affine factorisation")
print("=" * 78)
# C1 brute force over random draws
mx = 0.0
rg = np.random.default_rng(7)
for _ in range(2000):
    Fv = rg.normal(size=B); yv = rg.normal(); ev = rg.normal()
    lhs = (yv - Fv) - ETA * ev * (Fv - Fv.mean())
    w = 1.0 + ETA * ev
    ye = (yv + ETA * ev * Fv.mean()) / w
    mx = max(mx, float(np.max(np.abs(lhs - w * (ye - Fv)))))
print(f"C1 max |LHS - w_eff*(y_eff-F)| over 2000 random draws = {mx:.3e}")

# C2 the §5.2 least-squares allocation
rg = np.random.default_rng(11)
maxdev = 0.0
for _ in range(500):
    Fv = rg.normal(size=B)
    c = Fv - Fv.mean()
    r = rg.uniform(0.2, 2.0, size=B)          # per-trajectory mobility r_i
    b = rg.normal(size=2)                      # (b_mu, b_V)
    C = np.vstack([np.ones(B) / B, 2 * c / B])
    R = np.diag(r)
    nu = R @ C.T @ np.linalg.solve(C @ R @ C.T, b)
    # claim: nu_i / r_i  is affine in F_i, i.e. = lam1/B + lam2*2*c_i/B
    lam = np.linalg.solve(C @ R @ C.T, b)
    pred = r * (lam[0] / B + lam[1] * 2 * c / B)
    maxdev = max(maxdev, float(np.max(np.abs(nu - pred))))
print(f"C2 max |nu* - r_i*(affine in F_i)| over 500 draws     = {maxdev:.3e}")
print("    => the repo's own §5.2 allocation is ALSO affine in F_i "
      "(mobility-weighted), so the §3 reduction applies to it too.")

print()
print("=" * 78)
print("C3: open-loop replay of the realised schedule")
print("=" * 78)
rg = np.random.default_rng(42)
x0 = rg.normal(size=(B, D))
x0b = np.random.default_rng(43).normal(size=(B, D))   # different batch

# pick tau2 below the free-running spread so the controller has to TIGHTEN,
# then a second setpoint above it so it has to WIDEN.
for label, tau_mult in (("TIGHTEN", 0.6), ("WIDEN", 1.5)):
    _, F_un, _ = roll(x0, 0.0, 1e9, 'open', sched=(np.zeros(NSTEP), np.zeros(NSTEP)))
    sd_un = F_un.std(ddof=1)
    tau2 = (tau_mult * sd_un) ** 2
    y = float(np.median(F_un)) + 0.5 * sd_un

    xb, Fb, (w, ye, ev) = roll(x0, y, tau2, 'bdg')
    s_bdg = stats(Fb)

    # (a) exact realised schedule, same batch
    _, Fa, _ = roll(x0, y, tau2, 'open', sched=(w, ye))
    # (b) constant mean w_eff AND mean y_eff
    _, Fc, _ = roll(x0, y, tau2, 'open',
                    sched=(np.full(NSTEP, w.mean()), np.full(NSTEP, ye.mean())))
    # (c) constant mean w_eff, ORIGINAL y  (= the handoff's control)
    _, Fd, _ = roll(x0, y, tau2, 'open',
                    sched=(np.full(NSTEP, w.mean()), np.full(NSTEP, y)))
    # (d) realised schedule replayed on a DIFFERENT batch, vs closed loop on that batch
    _, Fe_open, _ = roll(x0b, y, tau2, 'open', sched=(w, ye))
    _, Fe_cl, _ = roll(x0b, y, tau2, 'bdg')

    print(f"\n--- {label}  (tau = {tau_mult} x unguided sd; e range "
          f"[{ev.min():+.3f},{ev.max():+.3f}], w_eff range [{w.min():.3f},{w.max():.3f}]) ---")
    print(f"unguided                       sd {sd_un:.6f}")
    print(f"BDG closed loop                sd {s_bdg['sd']:.6f}  mean {s_bdg['mean']:+.6f}"
          f"   sd/unguided {s_bdg['sd']/sd_un:.4f}")
    print(f"(a) replay realised w(t),y(t)  sd {Fa.std(ddof=1):.6f}  "
          f"max|F_open-F_bdg| = {np.max(np.abs(Fa-Fb)):.3e}   <== reproduces exactly")
    print(f"(b) const mean w, mean y_eff   sd {Fc.std(ddof=1):.6f}  "
          f"sd/unguided {Fc.std(ddof=1)/sd_un:.4f}")
    print(f"(c) const mean w, ORIGINAL y   sd {Fd.std(ddof=1):.6f}  "
          f"sd/unguided {Fd.std(ddof=1)/sd_un:.4f}   <== handoff's control")
    print(f"(d) replay on DIFFERENT batch  sd {Fe_open.std(ddof=1):.6f} vs closed-loop "
          f"{Fe_cl.std(ddof=1):.6f}   transfer err {abs(Fe_open.std(ddof=1)-Fe_cl.std(ddof=1)):.3e}")
