"""Experiment 3: (a) d=16 Gaussian prior, f=|x|^2, exact radial reference; (b) stochastic sampler (eta=1)
on the 2-D mixture settings. lambda = 1 for every method (no tuning) unless stated."""
import numpy as np, json, sys, time
from scipy import stats
from lib import *

def build(tg, y, s, v0):
    iso = lambda t: (lambda al, sg: sg**2 * v0 / (al**2 * v0 + sg**2))(*alpha_sigma(t))
    return {
     "plug-in x0.01":          lambda m,S,x,t,rng: 0.01 * est_plugin(tg,y,s,m,S),
     "var-iso (PiGDM-like)":   lambda m,S,x,t,rng: est_var_iso(tg,y,s,m,S,r2=iso(t)),
     "SMG var-only":           lambda m,S,x,t,rng: est_smg(tg,y,s,m,S,mean=False),
     "SMG (as stated)":        lambda m,S,x,t,rng: est_smg(tg,y,s,m,S),
     "SMG-2":                  lambda m,S,x,t,rng: est_smg(tg,y,s,m,S,second_var=True),
     "LGD-MC n=4":             lambda m,S,x,t,rng: est_lgd_mc(tg,y,s,m,S,n=4,rng=rng),
     "LGD-MC n=16":            lambda m,S,x,t,rng: est_lgd_mc(tg,y,s,m,S,n=16,rng=rng),
     "OSC0 n=4":               lambda m,S,x,t,rng: est_osc(tg,y,s,m,S,n=4,rng=rng,cov_term=False),
     "OSC0 n=16":              lambda m,S,x,t,rng: est_osc(tg,y,s,m,S,n=16,rng=rng,cov_term=False),
    }

out = {}
# ---------------- (a) d = 16
d, y, s = 16, 22.0, 0.5
prior = GMM(np.zeros((1, d)), [1.0], [1.0]); tg = target_sqnorm()
fg = np.linspace(1e-3, 80, 400001)
pf = stats.chi2.pdf(fg, d) * np.exp(-0.5 * (fg - y) ** 2 / s ** 2); pf /= pf.sum()
rng = np.random.default_rng(3)
f_ref = np.sort(rng.choice(fg, size=1500, p=pf))
print(f"(a) d=16 Gaussian prior, f=|x|^2, y={y}, s={s}. exact: mean (f-y)/s = {((fg-y)/s*pf).sum():.3f}, sd = {np.sqrt((((fg-y)/s)**2*pf).sum()-(((fg-y)/s*pf).sum())**2):.3f}")
print(f"  {'method':22s} {'eta':>4s} {'W1(f)/s':>14s} {'mean(f-y)/s':>12s} {'sd(f-y)/s':>10s} {'clip':>6s}")
M = build(tg, y, s, 1.0)
for eta in [0.0, 1.0]:
    for k, fn in M.items():
        w1s, mus, sds, cl = [], [], [], []
        for sd_ in [21, 22, 23]:
            r = np.random.default_rng(sd_)
            x, clip = ddim_sample(prior, lambda m,S,xx,t: fn(m,S,xx,t,r), 1500, r, steps=100, eta=eta, clip=50.0)
            f = np.sort(tg.f(x)); w1s.append(np.abs(f - f_ref).mean() / s); mus.append(((f - y)/s).mean()); sds.append(((f - y)/s).std()); cl.append(clip)
        out[f"d16|{k}|eta={eta}"] = [float(np.mean(w1s)), float(np.std(w1s)), float(np.mean(mus)), float(np.mean(sds))]
        print(f"  {k:22s} {eta:4.1f} {np.mean(w1s):8.3f}+-{np.std(w1s):.3f} {np.mean(mus):12.3f} {np.mean(sds):10.3f} {np.mean(cl):6.3f}")
        sys.stdout.flush()

# ---------------- (b) eta = 1 on the 2-D mixture
gmm = GMM([[-1.5, 0.0], [1.5, 0.5], [0.0, 2.0]], [0.4, 0.5, 0.35], [0.5, 0.3, 0.2])
for name, tg, y, s in [("GMM, f=|x|^2, y=2.6, s=0.15", target_sqnorm(), 2.6, 0.15),
                       ("GMM, f=|x|_1, y=1.8, s=0.15", target_l1(), 1.8, 0.15),
                       ("GMM, f=x1*x2, y=0.6, s=0.15", target_product(), 0.6, 0.15)]:
    rng = np.random.default_rng(5)
    v0 = gmm.sample(20000, rng).var(0).mean()
    ref = GridRef(gmm, tg, y, s, lim=4.5, n=640); ref_s = ref.sample_py(4000, rng); props = ref.mode_props()
    print(f"\n(b) {name}, eta=1, lambda=1")
    print(f"  {'method':22s} {'ED':>16s} {'|f-y|/s':>8s} {'modeTV':>7s} {'clip':>6s}")
    M = build(tg, y, s, v0)
    for k, fn in M.items():
        eds, te, tv, cl = [], [], [], []
        for sd_ in [31, 32, 33]:
            r = np.random.default_rng(sd_)
            x, clip = ddim_sample(gmm, lambda m,S,xx,t: fn(m,S,xx,t,r), 1500, r, steps=100, eta=1.0)
            eds.append(energy_distance(x, ref_s, r)); te.append(np.abs(tg.f(x) - y).mean() / s)
            tv.append(0.5 * np.abs(gmm.resp(x).mean(0) - props).sum()); cl.append(clip)
        out[f"{name}|{k}|eta=1"] = [float(np.mean(eds)), float(np.std(eds)), float(np.mean(te)), float(np.mean(tv))]
        print(f"  {k:22s} {np.mean(eds):8.4f}+-{np.std(eds):.4f} {np.mean(te):8.2f} {np.mean(tv):7.3f} {np.mean(cl):6.3f}")
        sys.stdout.flush()
json.dump(out, open("exp3_results.json", "w"), indent=1)
