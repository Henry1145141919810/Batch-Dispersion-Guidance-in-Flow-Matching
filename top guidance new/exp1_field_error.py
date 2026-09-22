"""Experiment 1: error of each guidance estimator against the exact tilted-mean shift.
x_t is drawn from the guided marginal (X ~ p_y, then noised). Error = relative RMS of Delta - Delta*."""
import numpy as np, json, sys, time
from lib import *

def methods(v0):
    iso = lambda t: (lambda al, sg: sg**2 * v0 / (al**2 * v0 + sg**2))(*alpha_sigma(t))
    M = {
     "plug-in":               (False, lambda tg,y,s,m,S,t,rng: est_plugin(tg,y,s,m,S)),
     "var-iso (PiGDM-like)":  (False, lambda tg,y,s,m,S,t,rng: est_var_iso(tg,y,s,m,S,r2=iso(t))),
     "SMG var-only":          (False, lambda tg,y,s,m,S,t,rng: est_smg(tg,y,s,m,S,mean=False)),
     "SMG mean-only":         (False, lambda tg,y,s,m,S,t,rng: est_smg(tg,y,s,m,S,var=False)),
     "SMG (as stated)":       (False, lambda tg,y,s,m,S,t,rng: est_smg(tg,y,s,m,S)),
     "SMG-2 (+2nd-order var)":(False, lambda tg,y,s,m,S,t,rng: est_smg(tg,y,s,m,S,second_var=True)),
     "SMG-2K (+curv. dir.)":  (False, lambda tg,y,s,m,S,t,rng: est_smg(tg,y,s,m,S,second_var=True,curv=True)),
     "LGD-MC n=4":            (True,  lambda tg,y,s,m,S,t,rng: est_lgd_mc(tg,y,s,m,S,n=4,rng=rng)),
     "LGD-MC n=4 antithetic": (True,  lambda tg,y,s,m,S,t,rng: est_lgd_mc(tg,y,s,m,S,n=4,rng=rng,antithetic=True)),
     "LGD-MC n=16":           (True,  lambda tg,y,s,m,S,t,rng: est_lgd_mc(tg,y,s,m,S,n=16,rng=rng)),
     "LGD-MC n=64":           (True,  lambda tg,y,s,m,S,t,rng: est_lgd_mc(tg,y,s,m,S,n=64,rng=rng)),
     "OSC0 n=4 (no cov term)":(True,  lambda tg,y,s,m,S,t,rng: est_osc(tg,y,s,m,S,n=4,rng=rng,cov_term=False)),
     "OSC n=4":               (True,  lambda tg,y,s,m,S,t,rng: est_osc(tg,y,s,m,S,n=4,rng=rng)),
     "OSC n=16":              (True,  lambda tg,y,s,m,S,t,rng: est_osc(tg,y,s,m,S,n=16,rng=rng)),
     "LGD-MC-iso n=4":        (True,  lambda tg,y,s,m,S,t,rng: est_lgd_mc(tg,y,s,m,S,n=4,rng=rng,iso_r2=iso(t))),
     "OSC-iso n=4":           (True,  lambda tg,y,s,m,S,t,rng: est_osc(tg,y,s,m,S,n=4,rng=rng,iso_r2=iso(t))),
    }
    return M

gmm = GMM([[-1.5, 0.0], [1.5, 0.5], [0.0, 2.0]], [0.4, 0.5, 0.35], [0.5, 0.3, 0.2])
gauss = GMM([[0.0, 0.0]], [1.0], [1.0])
settings = [
  ("Gaussian prior, f=|x|^2, y=2.6", gauss, target_sqnorm(), 2.6, [0.5, 0.15, 0.05]),
  ("GMM prior, f=|x|^2, y=2.6",      gmm,   target_sqnorm(), 2.6, [0.5, 0.15]),
  ("GMM prior, f=|x|_1, y=1.8",      gmm,   target_l1(),     1.8, [0.5, 0.15]),
  ("GMM prior, f=x1*x2, y=0.6",      gmm,   target_product(),0.6, [0.5, 0.15]),
]
ts = [0.9, 0.7, 0.5, 0.3, 0.1]
B, R = 240, 8
out = {}
t0 = time.time()
for name, prior, tg, y, ss in settings:
    rng = np.random.default_rng(1)
    v0 = prior.sample(20000, rng).var(0).mean()
    M = methods(v0)
    for s in ss:
        ref = GridRef(prior, tg, y, s, lim=4.5, n=640)
        key = f"{name}, s={s}"
        out[key] = {k: [] for k in M}
        for t in ts:
            al, sg = alpha_sigma(t)
            xt = al * ref.sample_py(B, rng) + sg * rng.standard_normal((B, prior.d))
            Dstar = ref.shift(xt, t, chunk=32)
            m, S = prior.posterior_moments(xt, t)
            den = (Dstar ** 2).sum(1).mean()
            for k, (stoch, fn) in M.items():
                errs = []
                for r in range(R if stoch else 1):
                    D = fn(tg, y, s, m, S, t, rng)
                    errs.append(((D - Dstar) ** 2).sum(1).mean())
                out[key][k].append(float(np.sqrt(np.mean(errs) / den)))
        print(f"\n== {key}   (relative RMS error of the shift; columns t={ts})   [{time.time()-t0:.0f}s]")
        for k in M:
            print(f"  {k:26s} " + "  ".join(f"{e:8.3f}" if e < 1e3 else f"{e:8.1e}" for e in out[key][k]))
        sys.stdout.flush()
json.dump({"ts": ts, "results": out}, open("exp1_results.json", "w"), indent=1)
