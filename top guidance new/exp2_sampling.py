"""Experiment 2: terminal-distribution error of guided DDIM sampling (100 steps, ideal denoiser).
Every method gets the same tuning opportunity: one global scale lambda chosen on seed 0 by energy
distance to the exact tilted target, then evaluated on fresh seeds. lambda = 1 is also reported."""
import numpy as np, json, sys, time
from lib import *

LAMS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]
N, STEPS, SEEDS = 1500, 100, [11, 12, 13]

def build(tg, y, s, v0):
    iso = lambda t: (lambda al, sg: sg**2 * v0 / (al**2 * v0 + sg**2))(*alpha_sigma(t))
    return {
     "unguided":               lambda m,S,x,t,rng: np.zeros_like(m),
     "plug-in":                lambda m,S,x,t,rng: est_plugin(tg,y,s,m,S),
     "var-iso (PiGDM-like)":   lambda m,S,x,t,rng: est_var_iso(tg,y,s,m,S,r2=iso(t)),
     "SMG var-only":           lambda m,S,x,t,rng: est_smg(tg,y,s,m,S,mean=False),
     "SMG mean-only":          lambda m,S,x,t,rng: est_smg(tg,y,s,m,S,var=False),
     "SMG (as stated)":        lambda m,S,x,t,rng: est_smg(tg,y,s,m,S),
     "SMG-2":                  lambda m,S,x,t,rng: est_smg(tg,y,s,m,S,second_var=True),
     "LGD-MC n=4":             lambda m,S,x,t,rng: est_lgd_mc(tg,y,s,m,S,n=4,rng=rng),
     "LGD-MC n=16":            lambda m,S,x,t,rng: est_lgd_mc(tg,y,s,m,S,n=16,rng=rng),
     "OSC0 n=4":               lambda m,S,x,t,rng: est_osc(tg,y,s,m,S,n=4,rng=rng,cov_term=False),
     "OSC0 n=16":              lambda m,S,x,t,rng: est_osc(tg,y,s,m,S,n=16,rng=rng,cov_term=False),
    }

def evaluate(x, prior, tg, y, s, ref, ref_s, props, lp_floor, rng):
    ok = np.isfinite(x).all(1)
    x = x[ok]
    ed = energy_distance(x, ref_s, rng)
    terr = np.abs(tg.f(x) - y).mean() / s
    tv = 0.5 * np.abs(prior.resp(x).mean(0) - props).sum()
    off = (prior.logpdf(x) < lp_floor).mean()
    return dict(ED=float(ed), terr=float(terr), modeTV=float(tv), off=float(off), nan=float(1 - ok.mean()))

gmm = GMM([[-1.5, 0.0], [1.5, 0.5], [0.0, 2.0]], [0.4, 0.5, 0.35], [0.5, 0.3, 0.2])
gauss = GMM([[0.0, 0.0]], [1.0], [1.0])
settings = [
  ("Gaussian prior, f=|x|^2, y=2.6, s=0.15", gauss, target_sqnorm(), 2.6, 0.15),
  ("GMM prior, f=|x|^2, y=2.6, s=0.15",      gmm,   target_sqnorm(), 2.6, 0.15),
  ("GMM prior, f=|x|_1, y=1.8, s=0.15",      gmm,   target_l1(),     1.8, 0.15),
  ("GMM prior, f=x1*x2, y=0.6, s=0.15",      gmm,   target_product(),0.6, 0.15),
]
res = {}
t0 = time.time()
for name, prior, tg, y, s in settings:
    rng = np.random.default_rng(5)
    v0 = prior.sample(20000, rng).var(0).mean()
    ref = GridRef(prior, tg, y, s, lim=4.5, n=640)
    ref_s = ref.sample_py(4000, rng)
    props = ref.mode_props()
    lp_floor = np.quantile(prior.logpdf(ref_s), 0.01)
    floor = [energy_distance(ref.sample_py(N, rng), ref_s, rng) for _ in range(5)]
    ref_terr = np.abs(tg.f(ref_s) - y).mean() / s
    M = build(tg, y, s, v0)
    res[name] = {"_noise_floor_ED": [float(np.mean(floor)), float(np.std(floor))], "_ref_terr": float(ref_terr)}
    print(f"\n== {name}   ED noise floor {np.mean(floor):.4f}+-{np.std(floor):.4f}; exact-target |f-y|/s = {ref_terr:.2f}   [{time.time()-t0:.0f}s]")
    print(f"  {'method':22s} {'lam*':>6s} | {'ED (lam=1)':>12s} | {'ED (lam*)':>16s} {'|f-y|/s':>8s} {'modeTV':>7s} {'off-supp':>8s} {'clip':>6s}")
    for k, fn in M.items():
        lams = [1.0] if k == "unguided" else LAMS
        tune = {}
        for lam in lams:
            r = np.random.default_rng(100)
            x, _ = ddim_sample(prior, lambda m,S,xx,t: lam * fn(m,S,xx,t,r), N, r, steps=STEPS)
            tune[lam] = evaluate(x, prior, tg, y, s, ref, ref_s, props, lp_floor, r)["ED"]
        best = min(tune, key=lambda l: tune[l] if np.isfinite(tune[l]) else 1e9)
        runs, runs1 = [], []
        for sd in SEEDS:
            r = np.random.default_rng(sd)
            x, clip = ddim_sample(prior, lambda m,S,xx,t: best * fn(m,S,xx,t,r), N, r, steps=STEPS)
            e = evaluate(x, prior, tg, y, s, ref, ref_s, props, lp_floor, r); e["clip"] = clip; runs.append(e)
            if best != 1.0:
                r = np.random.default_rng(sd)
                x, _ = ddim_sample(prior, lambda m,S,xx,t: fn(m,S,xx,t,r), N, r, steps=STEPS)
                runs1.append(evaluate(x, prior, tg, y, s, ref, ref_s, props, lp_floor, r)["ED"])
            else:
                runs1.append(e["ED"])
        agg = {q: [float(np.mean([u[q] for u in runs])), float(np.std([u[q] for u in runs]))] for q in runs[0]}
        agg["lam"] = best; agg["ED_lam1"] = [float(np.mean(runs1)), float(np.std(runs1))]; agg["tune"] = tune
        res[name][k] = agg
        print(f"  {k:22s} {best:6.3f} | {agg['ED_lam1'][0]:7.4f}+-{agg['ED_lam1'][1]:.3f} | {agg['ED'][0]:8.4f}+-{agg['ED'][1]:.4f} {agg['terr'][0]:8.2f} {agg['modeTV'][0]:7.3f} {agg['off'][0]:8.3f} {agg['clip'][0]:6.3f}")
        sys.stdout.flush()
json.dump(res, open("exp2_results.json", "w"), indent=1)
