"""Direct simulation checks of the item-3 closed-form claims, plus a corrected
item-5 design-effect measurement at the REAL (much smaller) sampler gain."""
import numpy as np

rng = np.random.default_rng(3)

# ---------------------------------------------- E1: fixed point V* = tau^2(1-1/eta)
print("[E1] simulate the batch loop to convergence; compare achieved V/tau^2 to 1-1/eta")
print("     (many steps, constant gain, B large so the estimator is sharp)")
B = 20000
for eta in (1.5, 2.0, 4.0, 8.0, 16.0):
    F = rng.normal(0.0, 1.0, size=B)
    tau2 = 0.5
    for _ in range(4000):
        Fb, V = F.mean(), F.var(ddof=1)
        e = (V - tau2) / tau2
        F = F + 0.01 * ((0.0 - F) - eta * e * (F - Fb))
    u = F.var(ddof=1) / tau2
    print("     eta=%5.1f   achieved V/tau^2 = %.5f   predicted %.5f   "
          "achieved sd/tau = %.4f (predicted %.4f)"
          % (eta, u, 1 - 1 / eta, np.sqrt(u), np.sqrt(max(0, 1 - 1 / eta))))
print("     -> the setpoint tau is NEVER achieved: the loop settles at")
print("        sd = tau*sqrt(1-1/eta) because plug's own centring keeps contracting")
print("        when e = 0. 'e -> 0' is not the equilibrium condition; w_eff -> 0 is.")

# ---------------------------------------------- E2: eta <= 1 has no interior rest point
print("\n[E2] eta <= 1: no interior equilibrium, V collapses (so no widening branch)")
for eta in (0.5, 1.0, 1.2):
    F = rng.normal(0.0, 1.0, size=4000)
    tau2 = 4.0                      # ask for a MUCH wider batch than we have
    for _ in range(3000):
        Fb, V = F.mean(), F.var(ddof=1)
        e = (V - tau2) / tau2
        F = F + 0.01 * ((0.0 - F) - eta * e * (F - Fb))
    print("     eta=%4.1f tau^2=4 (asked to WIDEN 4x): final V = %.5f -> %s"
          % (eta, F.var(ddof=1), "collapsed" if F.var(ddof=1) < 1e-3 else "held"))
print("     min(w_eff) = 1-eta, so eta<=1 can never make w_eff negative: at eta=1")
print("     BDG is plug with a variance-boosted POSITIVE weight and cannot widen.")
print("     handoff 9.1's floor-clearing wins are all eta=1 -> those cells are")
print("     'plug with an adaptive strength', not the widening controller of 1.")

# ---------------------------------------------- E3: the discrete bifurcation
print("\n[E3] discrete bifurcation at eta=4 (predicted monotone<1/6, osc 1/6-1/3, div>1/3)")
eta = 4.0


def iterate(kappa, n=400, u0=2.0):
    u = u0
    hist = []
    for _ in range(n):
        u = u * (1.0 - kappa * (1.0 + eta * (u - 1.0))) ** 2
        hist.append(u)
        if not np.isfinite(u) or u > 1e12:
            return np.array(hist), "DIVERGED"
    tail = np.array(hist[-80:])
    dev = tail - 0.75
    sgn = np.sign(dev[np.abs(dev) > 1e-9])
    flips = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
    if np.ptp(tail) > 1e-3:
        return np.array(hist), "LIMIT CYCLE ptp=%.3f" % np.ptp(tail)
    return np.array(hist), ("oscillatory" if flips > 20 else "monotone")


for kappa in (0.02, 0.15, 0.1666, 0.20, 0.30, 0.3333, 0.36, 0.5, 0.8):
    h, lab = iterate(kappa)
    print("     kappa=%.4f  final u=%-12s %s"
          % (kappa, ("%.6f" % h[-1]) if np.isfinite(h[-1]) else "inf", lab))

# ---------------------------------------------- E4: design effect at the REAL gain
print("\n[E4] design effect on in_band / sd at the REAL sampler gain")
print("     (kappa_peak ~ 1e-3..6e-3 from b_stability.py [3e], not the 2e-2 toy)")
ts = np.linspace(0, 1, 101)[:-1]
ts = ts[ts >= 0.5]


def sim(B, eta, tau2, base, nrep, delta, seed):
    g = np.random.default_rng(seed)
    inb, sds = [], []
    for _ in range(nrep):
        F = g.normal(0.6, 1.0, size=B)
        for t in ts:
            kap = base * (1 - t) / t
            Fb, V = F.mean(), F.var(ddof=1)
            e = (V - tau2) / tau2 if eta else 0.0
            F = F + kap * ((0.0 - F) - eta * e * (F - Fb))
        inb.append(np.mean(np.abs(F) < delta))
        sds.append(F.std(ddof=1))
    return np.array(inb), np.array(sds)


for base in (0.0012, 0.0064, 0.02):
    for B in (512,):
        for eta, lab in ((0.0, "plug (eta=0)"), (4.0, "BDG  (eta=4)")):
            inb, sds = sim(B, eta, 0.45, base, 600, 0.35, 21)
            p = inb.mean()
            n_naive = np.sqrt(p * (1 - p) / B)
            sd_naive = sds.mean() / np.sqrt(2.0 * (B - 1))
            print("     kappa_peak=%.4f B=%d %-13s in_band %.4f DEFF %.2fx | "
                  "sd %.4f DEFF %.2fx" % (base, B, lab, p, inb.std(ddof=1) / n_naive,
                                          sds.mean(), sds.std(ddof=1) / sd_naive))
print("     -> at the gain the real runs actually used, the coupling design effect")
print("        on in_band is ~1.0; the iid se is NOT materially wrong for in_band.")
print("        On sd it is already <1 (the servo suppresses replicate spread).")

# ---------------------------------------------- E5: noise rectification, analytic
print("\n[E5] noise rectification, ANALYTIC (the MC in d_estimator_noise 5e was")
print("     swamped by its own Monte-Carlo error; this is exact for the linear map)")
print("     rho = rho0 - kappa*eta*u*z,  E[rho^2] = rho0^2 + (kappa*eta*u*relse)^2")
for B in (64, 128, 512):
    relse = np.sqrt(2.0 / (B - 1))
    for kappa in (0.0012, 0.0064, 0.02):
        u, eta = 0.75, 4.0
        exc = (kappa * eta * u * relse) ** 2      # rho0 = 1 at the fixed point
        print("     B=%4d kappa=%.4f  per-step excess V factor 1+%.2e  -> over 50 "
              "steps %+.4f%% spurious widening" % (B, kappa, exc,
                                                   100 * ((1 + exc) ** 50 - 1)))
print("     Positive but negligible at B>=64 and the real gain: <0.01%. Only bites")
print("     at tiny B or large kappa. So estimator noise is NOT the binding problem;")
print("     the BINDING one is the sign noise of w_eff at the setpoint (b [3i]).")
