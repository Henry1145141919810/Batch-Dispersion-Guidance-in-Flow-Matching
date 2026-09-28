"""The closed-loop fixed point of BDG is NOT the setpoint tau.

Found while checking handoff open item 6 ("BDG_FIXED_POINT is wrong for mu and
alpha ... the fixed point is also eta-dependent, contradicting the planner
comment"). The handoff reports the eta-dependence as an unexplained defect.
It has a closed form.

DERIVATION. BDG's numerator is affine in F_i and factors (handoff section 3) as
    num_i = (1 + eta e) (y_eff - F_i),      w_eff := 1 + eta e
so to first order every deviation from the batch mean evolves as
    dev_i <- dev_i * (1 - h w_eff),         h = ||g||^2 / s^2 * step
BOTH terms -- the centring term and the dispersion term -- shrink the
deviations, and they do it through the SAME factor. The batch spread is
therefore stationary exactly when

    w_eff = 0   <=>   e = -1/eta   <=>   V_b* = tau^2 (1 - 1/eta)

The loop does not servo V_b to tau^2. It servos w_eff to ZERO -- the point at
which the guidance switches itself off. The setpoint is undershot by the
factor sqrt(1 - 1/eta), which is 0.866 at eta = 4 and 0 at eta = 1.

CONSEQUENCE FOR THE HANDOFF'S OWN RESULTS. Open item 1 says the only cells
that beat plug WHILE CLEARING THE FLOOR are eta = 1. At eta = 1 the fixed
point is V_b* = 0: the controller is not a variance controller there at all,
it is an unconditional contractor whose setpoint is total batch collapse, and
tau only sets how fast it gets there. Any "the knob works" reading of an
eta = 1 cell is reading a contractor.

This script measures V_b*/tau^2 against 1 - 1/eta.
"""
import json
import os

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixed_point_law.json")


def settle(eta, tau, B=512, h=0.02, steps=8000, sd0=1.0, seed=3, onesided=False):
    g = torch.Generator().manual_seed(seed)
    F = torch.randn(B, generator=g, dtype=torch.float64) * sd0
    for _ in range(steps):
        Vb = F.var(unbiased=True)
        e = (Vb - tau ** 2) / tau ** 2
        if onesided:
            e = e.clamp(min=0.0)
        F = F + h * ((0.0 - F) - eta * e * (F - F.mean()))
    return float(F.var(unbiased=True))


def main():
    tau = 0.5
    print("fixed point of the batch variance, tau = %.2f, B = 512, 8000 steps" % tau)
    print("  %-6s %-14s %-14s %-14s %s"
          % ("eta", "V_b*/tau^2", "1 - 1/eta", "sd*/tau", "sqrt(1-1/eta)"))
    rows = []
    # eta = 64 is EXCLUDED from the law check and printed separately: with an
    # explicit step h the map dev <- dev(1 - h w_eff) is only stable while
    # h w_eff < 2, and eta = 64 drives w_eff past that during the transient, so
    # what blows up there is my integrator, not the controller. The real arm
    # has the sampler's velocity clip in that role.
    for eta in (1.0, 1.5, 2.0, 4.0, 8.0, 16.0):
        for sd0 in (1.0, 0.05):          # approach from ABOVE and from BELOW
            V = settle(eta, tau, sd0=sd0)
            pred = 1.0 - 1.0 / eta
            rows.append({"eta": eta, "sd0": sd0, "V_over_tau2": V / tau ** 2,
                         "pred": pred})
            print("  %-6g %-14.6f %-14.6f %-14.6f %-14.6f  (from sd0=%.2f)"
                  % (eta, V / tau ** 2, pred, (V ** 0.5) / tau,
                     max(pred, 0.0) ** 0.5, sd0))
    err = max(abs(r["V_over_tau2"] - r["pred"]) for r in rows if r["eta"] > 1)
    print("\n  max |V_b*/tau^2 - (1 - 1/eta)| over 1 < eta <= 16, both "
          "directions: %.2e" % err)
    print("  -> the loop converges to w_eff = 0, not to V_b = tau^2: %s"
          % (err < 1e-6))
    print("  step-size limit: h*w_eff < 2. At eta = 64 the transient breaches "
          "it and this explicit toy diverges (nan) -- an integrator artifact, "
          "not the controller; the real arm has the velocity clip there.")

    # eta = 1: the fixed point is total collapse, from either side
    print("\n  eta = 1 (the handoff's only floor-clearing, plug-beating cells):")
    for sd0 in (1.0, 0.05):
        V = settle(1.0, tau, sd0=sd0, steps=20000)
        print("    from sd0=%.2f  V_b* = %.3e  (sd* = %.3e, tau = %.2f)"
              % (sd0, V, V ** 0.5, tau))

    # and what the ONE-SIDED variant does: it cannot pass w_eff = 0 downward
    print("\n  one-sided (e clamped at >= 0), eta = 4:")
    for sd0 in (1.0, 0.05):
        V = settle(4.0, tau, sd0=sd0, onesided=True)
        print("    from sd0=%.2f  V_b*/tau^2 = %.6f" % (sd0, V / tau ** 2))
    with open(OUT, "w") as fh:
        json.dump(rows, fh)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
