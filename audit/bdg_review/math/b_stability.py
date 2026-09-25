"""BDG item 3: the closed loop. Where is the fixed point, is it stable, and
what gain range makes the real 100-step Euler sampler overshoot / diverge?

Sampler facts used (proj1/src/sampling.py):
  euler:  x <- x + h*(v + G),  h = 0.01, 100 steps, guidance for t >= t_min=0.5
  G_i    = mult * (num_i/s^2) * J_i^T g_i,  mult = w*(1-t)/t   (plug family)
  F_i    = f_A(m_i(x_t))  ==>  dF_i/dx_t = J_i^T g_i
  ==> dF_i = h * mult * (a_i/s^2) * num_i   with a_i = |J_i^T g_i|^2 >= 0
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)


# ---------------------------------------------------------------- closed form
def fixed_point(eta):
    """dV/dt = -2 c (1 + eta e) V (+drift). Zero at 1+eta*e = 0."""
    return 1.0 - 1.0 / eta          # V*/tau^2


print("[3a] CONTINUOUS TIME, no drift.  d_i' = -c(1+eta*e) d_i  =>  V' = -2c(1+eta*e)V")
print("     Interior equilibrium needs 1+eta*e = 0, i.e. w_eff = 0, NOT e = 0.")
for eta in (0.5, 1.0, 2.0, 4.0, 8.0):
    u = fixed_point(eta)
    print("     eta=%4.1f : V*/tau^2 = %+.4f   sd*/tau = %s"
          % (eta, u, "%.4f" % np.sqrt(u) if u > 0 else "collapse to 0"))
print("     -> at eta=4 the loop settles at sd = sqrt(0.75) tau = 0.8660 tau,")
print("        a systematic 13.4%% undershoot of the REQUESTED spread.")
print("     -> at eta<=1 there is no interior equilibrium: V -> 0.")

# ---------------------------------------------------------------- discrete map
print("\n[3b] DISCRETE MAP with constant per-step gain kappa (homogeneous a_i):")
print("     u_{n+1} = u_n * (1 - kappa*(1 + eta*(u_n - 1)))^2,  u = V/tau^2")


def Gmap(u, kappa, eta):
    return u * (1.0 - kappa * (1.0 + eta * (u - 1.0))) ** 2


def dG(u, kappa, eta):
    ph = 1.0 - kappa * (1.0 + eta * (u - 1.0))
    return ph ** 2 + u * 2.0 * ph * (-kappa * eta)


for eta in (2.0, 4.0, 8.0):
    us = fixed_point(eta)
    k_osc = 1.0 / (2.0 * eta * us)      # G' = 0
    k_div = 1.0 / (eta * us)            # G' = -1
    print("     eta=%4.1f  u*=%.4f   monotone kappa<%.4f | oscillatory %.4f-%.4f | "
          "divergent kappa>%.4f" % (eta, us, k_osc, k_osc, k_div, k_div))
    for k in (k_osc * 0.5, k_osc * 1.01, k_div * 0.999, k_div * 1.02):
        print("        kappa=%.4f  G'(u*)=%+.4f  %s" % (k, dG(us, k, eta),
              "monotone" if dG(us, k, eta) > 0 else
              ("oscillatory-stable" if dG(us, k, eta) > -1 else "UNSTABLE")))

print("\n[3c] u=0 (total collapse) is a fixed point too. G'(0) = (1+kappa(eta-1))^2")
for eta in (0.5, 1.0, 2.0, 4.0):
    for k in (0.05,):
        print("     eta=%4.1f kappa=%.2f  G'(0)=%.4f  -> %s"
              % (eta, k, (1 + k * (eta - 1)) ** 2,
                 "REPELLING (good)" if (1 + k * (eta - 1)) ** 2 > 1
                 else "ATTRACTING: the batch collapses"))

# ------------------------------------------------- real sampler, real schedule
print("\n[3d] REAL SCHEDULE: 100 euler steps, h=0.01, guided t>=0.5 (50 steps),")
print("     kappa_n = h * w * (1-t_n)/t_n * a/s^2   (the (1-t)/t factor decays 99x)")
ts = np.linspace(0.0, 1.0, 101)[:-1]
guided = ts[ts >= 0.5]
sched = (1.0 - guided) / guided
print("     (1-t)/t at t=0.50 / 0.75 / 0.99 : %.4f / %.4f / %.4f"
      % (sched[0], (1 - 0.75) / 0.75, (1 - 0.99) / 0.99))
print("     sum_n h*(1-t_n)/t_n over the window = %.4f  (integral %.4f)"
      % (0.01 * sched.sum(), np.log(1.0) - 1.0 - (np.log(0.5) - 0.5)))


def run(u0, eta, base, drift=None, nstep_gain=None):
    """base = w*a/s^2. Returns u path (V/tau^2) and log spread ratio."""
    u = u0
    logr = 0.0
    path = [u]
    for j, t in enumerate(guided):
        kap = 0.01 * base * (1 - t) / t
        if nstep_gain is not None:
            kap = nstep_gain
        rho = 1.0 - kap * (1.0 + eta * (u - 1.0))
        u = u * rho ** 2
        if drift is not None:
            u *= drift[j]
        logr += np.log(abs(rho))
        path.append(u)
    return np.array(path), logr


# Calibrate base = w*a/s^2 from the handoff's own measured sd/unguided numbers.
print("\n[3e] CALIBRATE the real gain from handoff 6.1 (mu, eta=4, 50 guided steps)")
S_OVER_DELTA = {"mu": 1.53039 / 0.16799, "alpha": 8.18776 / 0.48135,
                "gap": 0.04752 / 0.00760}
print("     s/delta from QM9 csv: mu %.2f  alpha %.2f  gap %.2f"
      % tuple(S_OVER_DELTA[p] for p in ("mu", "alpha", "gap")))
ksum = 0.01 * sched.sum()
for tau_mult, sd_ratio, lab in ((0.5, 0.739, "e4t0.5 tighten"),
                                (1.5, 1.168, "e4t1.5 widen")):
    # mu: sd_unguided ~= s (measured 1.532 vs s 1.530), so V_unguided/tau^2 ~ 1/tau_mult^2
    u_typ = 1.0 / tau_mult ** 2
    weff = 1.0 + 4.0 * (u_typ - 1.0)
    base = np.log(sd_ratio) / (-ksum * weff)
    kap_peak = 0.01 * base * sched[0]
    print("     %-15s u~%.3f  w_eff~%+.2f  => w*a/s^2 ~ %.4f  kappa_peak ~ %.2e"
          % (lab, u_typ, weff, base, kap_peak))
    print("                     stability bound kappa < 1/(eta-1) = %.4f -> margin %.0fx"
          % (1 / 3, (1 / 3) / kap_peak))

print("\n[3f] what strength w would destabilise?  kappa_peak = 0.01*w*(a/s^2)*1.0")
for base_over_w, lab in ((0.120 / 4, "from the tightening cell"),
                         (0.454 / 4, "from the widening cell")):
    w_crit = (1 / 3) / (0.01 * base_over_w)
    print("     a/s^2 ~ %.4f (%s) -> w_crit ~ %.0f  (runs used w=4)"
          % (base_over_w, lab, w_crit))

# ---------------------------------------------- drift from the real flow
print("\n[3g] ADD THE FLOW'S OWN DRIFT (docs/results/VARIANCE_DECOMPOSITION.md,")
print("     across(t) for alpha unguided, in delta^2; it is NON-MONOTONE):")
t_dd = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])
across_alpha = np.array([283, 204, 239, 270, 285, 284, 284, 279, 283, 283, 280.0])
across_mu = np.array([107, 59.3, 59.6, 57.5, 62.2, 77.5, 83.6, 86.3, 85, 87.2, 88.0])
for nm, arr, sd_ in (("alpha", across_alpha, S_OVER_DELTA["alpha"]),
                     ("mu", across_mu, S_OVER_DELTA["mu"])):
    tau2_1 = sd_ ** 2                       # tau^2 in delta^2 at tau_mult = 1
    print("     %-5s tau^2(tau_mult=1) = %.0f delta^2 ; across(t)/tau^2 min %.3f "
          "max %.3f" % (nm, tau2_1, (arr / tau2_1).min(), (arr / tau2_1).max()))
    for tm in (0.5, 1.0, 1.5):
        u = arr / (tau2_1 * tm ** 2)
        weff = 1 + 4 * (u - 1)
        frac = float((weff < 0).mean())
        print("        tau_mult=%.1f : V/tau^2 %.3f-%.3f  w_eff %+.2f..%+.2f  "
              "fraction of logged t with w_eff<0 = %.0f%%"
              % (tm, u.min(), u.max(), weff.min(), weff.max(), 100 * frac))

print("\n[3h] WIDENING IMPLIES NEGATIVE w_eff (algebraic, not an estimate):")
print("     rho = 1 - kappa*w_eff with kappa>0; sd grows iff |rho|>1 iff w_eff<0.")
print("     handoff 6.1 reports sd/unguided 1.168/1.034/1.096 -> those cells ran")
print("     at w_eff<0 for a net-positive part of the window, i.e. as PLUG AT A")
print("     NEGATIVE WEIGHT. That is handoff open item 2's competitor, not a rival.")

# ---------------------------------------------- estimator noise at the setpoint
print("\n[3i] ESTIMATOR NOISE AT THE FIXED POINT (why 'stable' is not enough):")
for Bn in (64, 128, 512, 4096):
    rse = np.sqrt(2.0 / (Bn - 1))
    for eta in (4.0,):
        us = fixed_point(eta)
        sd_weff = eta * rse * us      # d w_eff = eta * d(V/tau^2)
        print("     B=%5d  rel.se(V_b)=%.4f  at u*=%.3f  sd(w_eff)=%.3f  "
              "-> w_eff sign flips ~every step (mean 0)" % (Bn, rse, us, sd_weff))
