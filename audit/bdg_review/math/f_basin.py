"""Transient / basin bound: the local stability bound kappa<1/(eta-1) is NOT the
binding one when the run STARTS far from the setpoint, which is exactly the
tau_mult=0.5 tightening case (V/tau^2 ~ 3-5 at t=0.5)."""
import numpy as np
eta = 4.0
def run(kappa, u0, n=2000):
    u = u0
    mx = u
    for _ in range(n):
        u = u*(1.0 - kappa*(1.0 + eta*(u-1.0)))**2
        mx = max(mx, u)
        if not np.isfinite(u) or u > 1e12: return np.inf, mx
    return u, mx
print("[F1] one-step amplification bound. rho = 1 - kappa*w_eff(u0);")
print("     the first step already WIDENS if kappa > 2/w_eff(u0).")
for u0 in (5.16, 4.0, 3.0, 2.0, 1.0, 0.75):
    w = 1+eta*(u0-1)
    print("     u0=%.2f  w_eff=%+6.2f  transient bound kappa < %s  (asymptotic bound 0.3333)"
          % (u0, w, "%.4f" % (2.0/w) if w>0 else "n/a (w_eff<=0)"))
print("\n[F2] simulate: where does the map actually end up?")
for kappa in (0.02, 0.10, 0.1538, 0.17, 0.20, 0.25, 0.30):
    row=[]
    for u0 in (5.16, 2.0, 0.9):
        uf, mx = run(kappa, u0)
        row.append("u0=%.2f->%s(peak %.1f)" % (u0, "DIV" if not np.isfinite(uf) else "%.4f"%uf, mx))
    print("     kappa=%.4f  " % kappa + " | ".join(row))
print("\n     u=0 is an absorbing collapse: once a step makes rho=0 exactly the")
print("     batch has zero spread and e=-1 forever (w_eff=1-eta, constant).")
print("\n[F3] the SAME bound in run units: kappa_peak = h*w*(a/s^2). With the")
print("     calibrated a/s^2 (0.030-0.114) the runs sat at kappa_peak 1.2e-3-6.4e-3,")
print("     i.e. 24x-128x below even the strict transient bound 2/w_eff(u0=5.16)=0.116.")
for a_s2, lab in ((0.030,"tightening-calibrated"), (0.114,"widening-calibrated")):
    for w in (4.0, 20.0, 100.0):
        kp = 0.01*w*a_s2
        print("     a/s^2=%.3f (%s) w=%5.1f -> kappa_peak=%.4f  %s"
              % (a_s2, lab, w, kp, "SAFE" if kp < 0.116 else "TRANSIENT-UNSTABLE"))
