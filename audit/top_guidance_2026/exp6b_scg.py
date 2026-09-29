"""Corrected SCG: the SAME second-order property expansion as SMG-2, PLUS third-cumulant terms.
   mu_F = f(m) + 0.5 tr(HS)
   V_F  = g'Sg + 0.5 tr((HS)^2) + g'K[H]
   M3_F = K[g,g,g] + 3 g'SHSg + tr((HS)^3)
   C1   = Sg + 0.5 K[H]          (= Cov(u,F))
   C2   = K[g,g] + 2 SHSg        (= E[u (F-mu)^2])
   Delta = c1 C1 + c2 C2,  [c1,c2] from the 2x2 moment system, scalar tilted moments by 1-D quadrature
   against a shifted-gamma law matching (mu_F, V_F, M3_F).
   Setting K = 0 and c2 = 0 with a Gaussian F reproduces SMG-2 exactly -- verified below."""
import numpy as np
from scipy import stats
from lib import *
from lib import _mv
from exp6_skew import post_K, pearson3

def scg2(tg, y, s, m, Sig, K, use_K=True, use_c2=True, quad=301, cap=1.5):
    g = tg.grad(m); H = tg.hess(m)
    HS = np.einsum("nij,njk->nik", H, Sig)
    trHS = np.trace(HS, axis1=1, axis2=2)
    HS2 = np.einsum("nij,njk->nik", HS, HS)
    KH  = np.einsum("nijl,njl->ni", K, H) if use_K else np.zeros_like(m)
    Kgg = np.einsum("ni,nj,nijl->nl", g, g, K) if use_K else np.zeros_like(m)
    Kggg= np.einsum("ni,nijl->njl", g, K) if use_K else None
    m3K = np.einsum("ni,nj,nl,nijl->n", g, g, g, K) if use_K else np.zeros(len(m))
    SHSg = _mv(Sig, _mv(H, _mv(Sig, g)))
    mu = tg.f(m) + 0.5*trHS
    V  = np.einsum("ni,nij,nj->n", g, Sig, g) + 0.5*np.trace(HS2, axis1=1, axis2=2) + (g*KH).sum(1)
    M3 = m3K + 3*(g*SHSg).sum(1) + np.trace(np.einsum("nij,njk->nik", HS2, HS), axis1=1, axis2=2)
    C1 = _mv(Sig, g) + 0.5*KH
    C2 = Kgg + 2*SHSg
    out = np.zeros_like(m)
    for n in range(len(m)):
        Vn = max(V[n], 1e-30); gam = float(np.clip(M3[n]/Vn**1.5, -cap, cap))
        F, dens, m4 = pearson3(mu[n], Vn, gam, quad)
        w = dens*np.exp(-0.5*(F-y)**2/s**2); Z = np.trapezoid(w, F)
        if not np.isfinite(Z) or Z <= 1e-300:
            out[n] = C1[n]*((y-mu[n])/(s**2+Vn)); continue
        M1 = np.trapezoid(w*(F-mu[n]), F)/Z; M2 = np.trapezoid(w*(F-mu[n])**2, F)/Z
        if not use_c2:
            out[n] = C1[n]*M1/Vn; continue
        A = np.array([[Vn, M3[n]], [M3[n], m4 - Vn**2]])
        try:
            c = np.linalg.solve(A, np.array([M1, M2 - Vn])); out[n] = c[0]*C1[n] + c[1]*C2[n]
        except np.linalg.LinAlgError:
            out[n] = C1[n]*M1/Vn
    return out

gmm = GMM([[-1.5,0.],[1.5,0.5],[0.,2.]], [0.4,0.5,0.35], [0.5,0.3,0.2])
gauss = GMM([[0.,0.]], [1.0], [1.0])
print("reduction check: SCG with K=0, c2 off, Gaussian F  vs  SMG-2")
xt = np.random.default_rng(0).standard_normal((6,2))
mm, SS = gmm.posterior_moments(xt, 0.5); _, KK = post_K(gmm, xt, 0.5)
tg = target_sqnorm()
a = scg2(tg, 2.6, 0.15, mm, SS, np.zeros_like(KK), use_K=False, use_c2=False)
b = est_smg(tg, 2.6, 0.15, mm, SS, second_var=True)
print(f"  max relative difference = {np.abs(a-b).max()/np.abs(b).max():.2e}\n")

for pname, prior in [("GMM prior (non-Gaussian posterior)", gmm), ("Gaussian prior", gauss)]:
    print(f"=== {pname}, f=|x|^2, y=2.6 : relative error of the exact shift ===")
    print(f"  {'setting':16s} {'SMG':>8s} {'SMG-2':>8s} {'SCG(K,no c2)':>13s} {'SCG(full)':>10s} {'LGD16':>8s} {'LGD64':>8s}")
    for s2 in [0.5, 0.15]:
        ref = GridRef(prior, tg, 2.6, s2, lim=4.6, n=1000); rr = np.random.default_rng(9)
        for t in [0.7, 0.5, 0.3]:
            al, sg = alpha_sigma(t)
            xt = al*ref.sample_py(80, rr) + sg*rr.standard_normal((80,2))
            D = ref.shift(xt, t, chunk=20); mm, SS = prior.posterior_moments(xt, t); _, KK = post_K(prior, xt, t)
            den = np.sqrt((D**2).sum(1).mean()); e = lambda V: float(np.sqrt(((V-D)**2).sum(1).mean())/den)
            print(f"  s={s2:<5} t={t:<4} {e(est_smg(tg,2.6,s2,mm,SS)):8.4f} {e(est_smg(tg,2.6,s2,mm,SS,second_var=True)):8.4f} "
                  f"{e(scg2(tg,2.6,s2,mm,SS,KK,use_c2=False)):13.4f} {e(scg2(tg,2.6,s2,mm,SS,KK)):10.4f} "
                  f"{e(est_lgd_mc(tg,2.6,s2,mm,SS,n=16,rng=rr)):8.2f} {e(est_lgd_mc(tg,2.6,s2,mm,SS,n=64,rng=rr)):8.2f}")
    print()
