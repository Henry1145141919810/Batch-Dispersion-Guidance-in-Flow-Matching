"""Experiment 7: is the measurable standardised skew of the observable a usable TRUST SIGNAL?
gamma = K[g,g,g] / (g'Sg)^{3/2}, computable from one extra Jacobian-derivative probe.
Question: does |gamma| predict where the two-moment closure (SMG-2) is inaccurate?
Baselines it must beat: the schedule itself (t), and the posterior spread g'Sg."""
import numpy as np
from scipy import stats
from lib import *
from exp6_skew import post_K

gmm = GMM([[-1.5,0.],[1.5,0.5],[0.,2.]], [0.4,0.5,0.35], [0.5,0.3,0.2])
rows = []
for tg, y in [(target_sqnorm(), 2.6), (target_l1(), 1.8), (target_product(), 0.6)]:
    for s in [0.5, 0.15]:
        ref = GridRef(gmm, tg, y, s, lim=4.6, n=900); rr = np.random.default_rng(4)
        for t in [0.9, 0.7, 0.5, 0.3, 0.15]:
            al, sg = alpha_sigma(t)
            xt = al*ref.sample_py(200, rr) + sg*rr.standard_normal((200,2))
            D = ref.shift(xt, t, chunk=25); mm, SS = gmm.posterior_moments(xt, t); _, KK = post_K(gmm, xt, t)
            E2 = est_smg(tg, y, s, mm, SS, second_var=True)
            g = tg.grad(mm)
            V = np.einsum("ni,nij,nj->n", g, SS, g)
            M3 = np.einsum("ni,nj,nl,nijl->n", g, g, g, KK)
            gam = np.abs(M3)/np.maximum(V, 1e-12)**1.5
            err = np.sqrt(((E2-D)**2).sum(1))/max(np.sqrt((D**2).sum(1)).mean(), 1e-12)
            for i in range(len(xt)):
                rows.append((float(gam[i]), float(V[i]), t, float(err[i])))
A = np.array(rows)
gam, V, tt, err = A[:,0], A[:,1], A[:,2], A[:,3]
le = np.log10(np.maximum(err, 1e-8))
print(f"n = {len(A)} states pooled over 3 targets x 2 widths x 5 times")
print("\nrank correlation with the two-moment closure's error (Spearman):")
for nm, v in [("|skew gamma|  (1 extra probe)", gam), ("posterior spread g'Sg", V), ("time t (schedule only)", tt)]:
    r = stats.spearmanr(v, err)
    print(f"  {nm:30s} rho = {r.statistic:+.3f}   (p = {r.pvalue:.1e})")
print("\ndecile of |gamma|  ->  median relative error of the two-moment closure")
q = np.quantile(gam, np.linspace(0,1,11))
for i in range(10):
    sel = (gam >= q[i]) & (gam <= q[i+1])
    print(f"  |gamma| in [{q[i]:6.3f},{q[i+1]:6.3f}]  n={sel.sum():4d}   median err = {np.median(err[sel]):7.3f}   90th pct = {np.quantile(err[sel],0.9):8.3f}")
print("\ngate test: abstain-or-escalate on the worst 20% by each signal; what error mass is caught?")
tot = err.sum()
for nm, v in [("|skew gamma|", gam), ("spread g'Sg", V), ("time t", tt)]:
    thr = np.quantile(v, 0.8); sel = v >= thr
    print(f"  {nm:15s} flags {sel.mean()*100:4.1f}% of states, catching {err[sel].sum()/tot*100:5.1f}% of total error mass")
