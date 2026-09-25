import numpy as np
rng = np.random.default_rng(0)

# [1] BDG numerator splits into a unit-weight mean part and a w_eff-weight deviation part
errs = []
for _ in range(2000):
    B = rng.integers(3, 600); F = rng.normal(rng.normal(), rng.uniform(.1, 3), B)
    y = rng.normal() * 3; eta = rng.uniform(0, 16); tau2 = rng.uniform(.05, 5)
    Fb = F.mean(); V = F.var(ddof=1); e = V / tau2 - 1; w = 1 + eta * e
    num = (y - F) - eta * e * (F - Fb)
    split = (y - Fb) + w * (Fb - F)
    errs.append(np.max(np.abs(num - split)) / (1 + np.max(np.abs(num))))
print("[1] BDG num = (y-Fbar) + w_eff*(Fbar-F_i): max rel err %.2e" % max(errs))

# [2] MGD-style quadratic-moment corrector on a scalar feature f: phi=(f,f^2), grad phi=(g,2 f g)
#     drift_i = -(th1 + 2 th2 F_i) g_i = w'(y' - F_i) g_i with w' = 2 th2, y' = -th1/(2 th2)
#     theta from an MGD-like 2x2 Gram solve so the ensemble moments hit targets (m1, m2)
worst = 0; agree = []
for _ in range(500):
    B = 256; F = rng.normal(0, 1, B); a = rng.uniform(.5, 2, B)
    m1 = F.mean() + rng.normal() * .1; vt = rng.uniform(.3, 3); m2 = vt + m1**2
    h = 1e-3
    G = np.array([[np.mean(a), np.mean(2*F*a)], [np.mean(2*F*a), np.mean(4*F*F*a)]])
    r = np.array([F.mean() - m1, np.mean(F*F) - m2]) / h
    th = np.linalg.solve(G, r)
    coef = -(th[0] + 2*th[1]*F)
    wp = 2*th[1]; yp = -th[0]/(2*th[1])
    worst = max(worst, np.max(np.abs(coef - wp*(yp - F)))/np.max(np.abs(coef)))
    agree.append((wp < 0) == (F.var() < vt))
print("[2] MGD quadratic corrector on scalar f == w'(y'-F_i) g_i: max rel err %.2e" % worst)
print("[2] w' < 0 (widening) exactly when current var < target var: %.3f of draws" % np.mean(agree))

# [3] linear toy: BDG widening vs constant negative-weight plug matched on terminal spread
def run(mode, w=1.0, eta=4.0, tau_mult=1.5, steps=50, B=512, y=3.0, seed=1):
    r = np.random.default_rng(seed); F = r.normal(0, 1, B); s2 = 1.0; tau2 = (tau_mult**2) * s2
    a = r.uniform(0.5, 1.5, B)  # heterogeneous mobility, uncorrelated with F
    k = 0.02
    for _ in range(steps):
        Fb = F.mean()
        if mode == 'bdg':
            e = F.var(ddof=1) / tau2 - 1; num = (y - F) - eta * e * (F - Fb)
        else:
            num = w * (y - F)
        F = F + k * a * num / s2
    return F.mean() - y, F.std(ddof=1)

b0, s0 = run('plug', w=0.0)
bb, sb = run('bdg')
# find constant negative plug weight that matches BDG terminal sd
ws = np.linspace(-3, 1, 4001); best = min(ws, key=lambda w: abs(run('plug', w=w)[1] - sb))
bp, sp = run('plug', w=best); b1, s1 = run('plug', w=1.0)
print("[3] unguided       bias %+.3f  sd %.3f" % (b0, s0))
print("[3] plug w=1       bias %+.3f  sd %.3f" % (b1, s1))
print("[3] BDG e4 t1.5    bias %+.3f  sd %.3f" % (bb, sb))
print("[3] plug w=%.3f   bias %+.3f  sd %.3f   (matched sd)" % (best, bp, sp))
