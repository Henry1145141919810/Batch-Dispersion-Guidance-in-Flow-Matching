"""Experiment 5: (a) oracle floor - the sampler driven by the exact tilted-mean shift (validates signs,
conversions and the solver error); (b) SMG-2 with isotropic, schedule-only trace terms (guide HVPs only)."""
import numpy as np, json, sys, time
from lib import *

gmm = GMM([[-1.5, 0.0], [1.5, 0.5], [0.0, 2.0]], [0.4, 0.5, 0.35], [0.5, 0.3, 0.2])
rng = np.random.default_rng(5)
v0 = gmm.sample(20000, rng).var(0).mean()
iso = lambda t: (lambda al, sg: sg**2 * v0 / (al**2 * v0 + sg**2))(*alpha_sigma(t))

def smg2_isotrace(tg, y, s, m, S, t):
    g = tg.grad(m); H = tg.hess(m); r2 = iso(t)
    c = 0.5 * r2 * np.trace(H, axis1=1, axis2=2)
    v2 = 0.5 * r2 ** 2 * np.einsum("nij,nji->n", H, H)
    vf = np.einsum("ni,nij,nj->n", g, S, g)          # free: reuses the VJP of the plug-in
    return np.einsum("nij,nj->ni", S, ((y - tg.f(m) - c) / (s ** 2 + vf + v2))[:, None] * g)

out = {}
for name, tg, y, s in [("GMM, f=|x|^2, y=2.6, s=0.15", target_sqnorm(), 2.6, 0.15),
                       ("GMM, f=x1*x2, y=0.6, s=0.15", target_product(), 0.6, 0.15)]:
    ref = GridRef(gmm, tg, y, s, lim=4.5, n=640); ref_s = ref.sample_py(4000, rng); props = ref.mode_props()
    coarse = GridRef(gmm, tg, y, s, lim=4.5, n=320)
    print(f"\n{name}")
    print(f"  {'method':30s} {'eta':>4s} {'ED':>16s} {'|f-y|/s':>8s} {'modeTV':>7s}")
    for eta in [0.0, 1.0]:
        t0 = time.time()
        r = np.random.default_rng(77)
        x, _ = ddim_sample(gmm, lambda m,S,xx,t: coarse.shift(xx, t, chunk=100), 400, r, steps=100, eta=eta)
        ed = energy_distance(x, ref_s, r, k=400); te = np.abs(tg.f(x) - y).mean() / s
        tv = 0.5 * np.abs(gmm.resp(x).mean(0) - props).sum()
        fl = np.mean([energy_distance(ref.sample_py(400, r), ref_s, r, k=400) for _ in range(5)])
        print(f"  {'ORACLE exact shift (N=400)':30s} {eta:4.1f} {ed:8.4f} (floor {fl:.4f}) {te:8.2f} {tv:7.3f}   [{time.time()-t0:.0f}s]")
        out[f"{name}|oracle|eta={eta}"] = [float(ed), float(fl), float(te), float(tv)]
        for k, fn in {"SMG-2 (model traces)": lambda m,S,xx,t: est_smg(tg,y,s,m,S,second_var=True),
                      "SMG-2 (isotropic traces)": lambda m,S,xx,t: smg2_isotrace(tg,y,s,m,S,t)}.items():
            E, T, V = [], [], []
            for sd in [51, 52, 53]:
                r = np.random.default_rng(sd)
                x, _ = ddim_sample(gmm, fn, 1500, r, steps=100, eta=eta)
                E.append(energy_distance(x, ref_s, r)); T.append(np.abs(tg.f(x) - y).mean() / s)
                V.append(0.5 * np.abs(gmm.resp(x).mean(0) - props).sum())
            print(f"  {k:30s} {eta:4.1f} {np.mean(E):8.4f}+-{np.std(E):.4f} {np.mean(T):8.2f} {np.mean(V):7.3f}")
            out[f"{name}|{k}|eta={eta}"] = [float(np.mean(E)), float(np.std(E)), float(np.mean(T)), float(np.mean(V))]
        sys.stdout.flush()
json.dump(out, open("exp5_results.json", "w"), indent=1)
