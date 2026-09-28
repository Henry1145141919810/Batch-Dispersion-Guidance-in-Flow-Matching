"""Fit ONE gain base = w*a/s^2 to the handoff's own 6.1 sd/unguided ladder.
If a single gain reproduces the whole ladder, the loop model is the right model
and the fixed point V*=tau^2(1-1/eta) is the thing the ladder is converging to."""
import numpy as np
ts = np.linspace(0,1,101)[:-1]; ts = ts[ts>=0.5]
B, eta = 20000, 4.0
def sd_ratio(base, tau_mult, sd_ung_over_s=1.0, seed=1):
    g = np.random.default_rng(seed)
    F = g.normal(0.0, sd_ung_over_s, size=B)     # units of s; unguided sd
    sd0 = F.std(ddof=1); tau2 = tau_mult**2
    for t in ts:
        kap = base*(1-t)/t
        Fb, V = F.mean(), F.var(ddof=1)
        e = (V-tau2)/tau2
        F = F + kap*((0.0-F) - eta*e*(F-Fb))
    return F.std(ddof=1)/sd0

# handoff 6.1, seed1. mu sd_unguided/s ~ 1.00 (QM9 s=1.530, measured 1.532)
TARGET = {"mu":  [(0.5,0.739),(1.0,0.951),(1.21,1.016),(1.5,1.168)],
          "alpha":[(0.5,0.681),(1.0,0.905),(1.5,1.034)],
          "gap":  [(0.5,0.787),(1.0,0.984),(1.5,1.096)]}
SD_UNG = {"mu":1.001, "alpha":0.982, "gap":1.00}   # sd_unguided / s
print("fit one gain per property to the WHOLE tau ladder (eta=4, 50 guided steps)")
for prop, rows in TARGET.items():
    best=None
    for base in np.concatenate([np.linspace(0.001,0.2,200), np.linspace(0.2,2.0,180)]):
        err=sum((sd_ratio(base,tm,SD_UNG[prop])-tgt)**2 for tm,tgt in rows)
        if best is None or err<best[0]: best=(err,base)
    err, base = best
    pred=[sd_ratio(base,tm,SD_UNG[prop]) for tm,_ in rows]
    print("  %-6s best w*a/s^2 = %.4f  kappa_peak = %.5f  rms err %.4f"
          % (prop, base/0.01, base, np.sqrt(err/len(rows))))
    print("         tau_mult %s" % "  ".join("%.2f"%tm for tm,_ in rows))
    print("         measured %s" % "  ".join("%.3f"%t for _,t in rows))
    print("         model    %s" % "  ".join("%.3f"%p for p in pred))
    print("         fixed-pt %s  (= sqrt(0.75)*tau_mult/sd_ung)"
          % "  ".join("%.3f"%(np.sqrt(0.75)*tm/SD_UNG[prop]) for tm,_ in rows))
print("\nstability bound at eta=4: kappa < 1/(eta-1) = 0.3333 asymptotically,")
print("kappa < 2/w_eff(u0) transiently (0.113 at the tau_mult=0.5 start point).")

print("\nNOTE kappa_n = h*w*(1-t)/t*(a/s^2) with h=0.01; the fit's 'base' IS kappa_peak.")
for prop, kp in (("mu",0.0020),("alpha",0.0030),("gap",0.0010)):
    a_s2 = kp/(0.01*4.0)                      # runs used w = 4
    print("  %-6s kappa_peak=%.4f -> a/s^2=%.4f ; w to reach the TRANSIENT bound"
          " 0.113 = %.0f ; the ASYMPTOTIC bound 0.333 = %.0f  (runs used w=4)"
          % (prop, kp, a_s2, 0.113/(0.01*a_s2), 0.3333/(0.01*a_s2)))
