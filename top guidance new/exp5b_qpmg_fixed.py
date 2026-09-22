"""QPMG, corrected Fourier normalisation + convergence control.
Finding under test: the doc's Sec 2.3 numerator ALREADY equals grad_m Z_Q, so dividing it by s^2 Z_Q
(the Sec 2.2 convention) double-counts 1/s^2."""
import numpy as np, json
from numpy.polynomial.hermite_e import hermegauss
from lib import *

def qpmg_gh(fm, g, H, Sig, y, s, n=140):
    L = np.linalg.cholesky(Sig + 1e-14*np.eye(len(g)))
    x, w = hermegauss(n); w = w/w.sum()
    X1, X2 = np.meshgrid(x, x, indexing="ij"); W = np.outer(w, w).ravel()
    U = np.stack([X1.ravel(), X2.ravel()], -1) @ L.T
    q = fm + U@g + 0.5*np.einsum("ni,ij,nj->n", U, H, U)
    l = np.exp(-0.5*(y-q)**2/s**2); Z = (W*l).sum()
    num = ((W*(y-q)*l)[:, None]*(g[None]+U@H)).sum(0)/s**2
    return Z, num/Z

def qpmg_fourier(fm, g, H, Sig, y, s, nsig=12.0, n=400001, s2div=False):
    L = np.linalg.cholesky(Sig + 1e-14*np.eye(len(g)))
    a = L.T@g; B = L.T@H@L
    b, U = np.linalg.eigh(B); at = U.T@a; HLU = H@L@U
    Om = nsig/s
    w = np.linspace(-Om, Om, n); dw = w[1]-w[0]
    C = 1 - 1j*w[:, None]*b[None]
    phi = np.exp(-0.5*np.log(C).sum(1) + 1j*w*fm - 0.5*w**2*(at[None]**2/C).sum(1))
    base = np.exp(-s**2*w**2/2 - 1j*w*y)*phi
    Z = (s/np.sqrt(2*np.pi))*np.trapezoid(base, dx=dw).real
    vec = 1j*w[:, None]*g[None] - (w**2)[:, None]*((at[None]/C)[:, None, :]*HLU[None]).sum(-1)
    num = (s/np.sqrt(2*np.pi))*np.trapezoid(base[:, None]*vec, dx=dw, axis=0).real
    return Z, num/(Z*(s**2 if s2div else 1.0))

print("=== normalisation check on the doc's own counterexample ===")
Sig = np.diag([0.4, 0.7]); y, s = 1.0, 0.8
g = np.array([1.0, 0.0]); H = np.array([[0.,1.],[1.,0.]]); fm = 0.0
Zg, Gg = qpmg_gh(fm, g, H, Sig, y, s)
Zf, Gf = qpmg_fourier(fm, g, H, Sig, y, s)
Zb, Gb = qpmg_fourier(fm, g, H, Sig, y, s, s2div=True)
print(f"  Gauss-Hermite      {Gg}")
print(f"  Fourier (num/Z)    {Gf}   max|diff| = {np.abs(Gg-Gf).max():.2e}   <-- agrees")
print(f"  Fourier (num/s^2Z) {Gb}   ratio to correct = {Gb[0]/Gf[0]:.6f}  (1/s^2 = {1/s**2:.6f})")
print(f"  likelihood Z: GH {Zg:.12f} vs Fourier {Zf:.12f}, diff {abs(Zg-Zf):.2e}")

print("\n=== Closure ladder vs EXACT guidance, Gaussian prior, f=|x|^2 (QPMG surrogate is exact) ===")
gauss = GMM([[0.,0.]], [1.0], [1.0]); tg = target_sqnorm(); y2 = 2.6
print(f"  {'setting':14s} {'plug-in':>9s} {'SMG':>8s} {'SMG-2':>8s} {'QPMG(GH120)':>12s} {'QPMG(Fourier)':>14s} {'LGD16':>9s}")
out = {}
for s2 in [0.5, 0.15, 0.05]:
    ref = GridRef(gauss, tg, y2, s2, lim=5.0, n=1000)
    rng = np.random.default_rng(9)
    for t in [0.7, 0.5, 0.3]:
        al, sg = alpha_sigma(t)
        xt = al*ref.sample_py(40, rng) + sg*rng.standard_normal((40, 2))
        D = ref.shift(xt, t, chunk=20); mm, SS = gauss.posterior_moments(xt, t)
        den = np.sqrt((D**2).sum(1).mean())
        QH = np.zeros_like(mm); QF = np.zeros_like(mm)
        for i in range(len(xt)):
            fmi, gi, Hi = tg.f(mm[i:i+1])[0], tg.grad(mm[i:i+1])[0], tg.hess(mm[i:i+1])[0]
            QH[i] = SS[i] @ qpmg_gh(fmi, gi, Hi, SS[i], y2, s2, n=120)[1]
            QF[i] = SS[i] @ qpmg_fourier(fmi, gi, Hi, SS[i], y2, s2)[1]
        e = lambda V: float(np.sqrt(((V-D)**2).sum(1).mean())/den)
        row = dict(plugin=e(est_plugin(tg,y2,s2,mm,SS)), SMG=e(est_smg(tg,y2,s2,mm,SS)),
                   SMG2=e(est_smg(tg,y2,s2,mm,SS,second_var=True)), QPMG_GH=e(QH), QPMG_F=e(QF),
                   LGD16=e(est_lgd_mc(tg,y2,s2,mm,SS,n=16,rng=rng)))
        out[f"gauss s={s2} t={t}"] = row
        print(f"  s={s2:<5} t={t:<4} {row['plugin']:9.3f} {row['SMG']:8.4f} {row['SMG2']:8.4f} {row['QPMG_GH']:12.4f} {row['QPMG_F']:14.2e} {row['LGD16']:9.2f}")

print("\n=== Same ladder, GMM prior: posterior is NOT Gaussian, so QPMG's model is wrong ===")
gmm = GMM([[-1.5,0.],[1.5,0.5],[0.,2.]], [0.4,0.5,0.35], [0.5,0.3,0.2])
print(f"  {'setting':14s} {'SMG':>8s} {'SMG-2':>8s} {'QPMG(Fourier)':>14s} {'LGD16':>8s}")
for s2 in [0.5, 0.15]:
    ref = GridRef(gmm, tg, y2, s2, lim=4.5, n=1000); rng = np.random.default_rng(9)
    for t in [0.7, 0.5, 0.3]:
        al, sg = alpha_sigma(t)
        xt = al*ref.sample_py(40, rng) + sg*rng.standard_normal((40, 2))
        D = ref.shift(xt, t, chunk=20); mm, SS = gmm.posterior_moments(xt, t)
        den = np.sqrt((D**2).sum(1).mean())
        QF = np.zeros_like(mm)
        for i in range(len(xt)):
            fmi, gi, Hi = tg.f(mm[i:i+1])[0], tg.grad(mm[i:i+1])[0], tg.hess(mm[i:i+1])[0]
            QF[i] = SS[i] @ qpmg_fourier(fmi, gi, Hi, SS[i], y2, s2)[1]
        e = lambda V: float(np.sqrt(((V-D)**2).sum(1).mean())/den)
        row = dict(SMG=e(est_smg(tg,y2,s2,mm,SS)), SMG2=e(est_smg(tg,y2,s2,mm,SS,second_var=True)),
                   QPMG_F=e(QF), LGD16=e(est_lgd_mc(tg,y2,s2,mm,SS,n=16,rng=rng)))
        out[f"gmm s={s2} t={t}"] = row
        print(f"  s={s2:<5} t={t:<4} {row['SMG']:8.4f} {row['SMG2']:8.4f} {row['QPMG_F']:14.4f} {row['LGD16']:8.2f}")
json.dump(out, open("exp5b_results.json","w"), indent=1)
