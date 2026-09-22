"""Experiment 5 (QPMG): reproduce the proposal's counterexample, then place QPMG on the closure ladder.

Ladder, all under a Gaussian posterior N(m,Sigma) with q(u) = f(m) + g'u + 0.5 u'Hu:
  plug-in : delta(q - f(m))                   -> no spread
  SMG     : mean of q correct, variance g'Sg  (MISSING 0.5 tr((HS)^2))
  SMG-2   : exact mean AND exact variance of q, Gaussian closure
  QPMG    : exact law of q (no Gaussian closure)
For quadratic f + Gaussian posterior, QPMG must equal the exact Stein-form guidance.
"""
import numpy as np, json
from numpy.polynomial.hermite_e import hermegauss
from lib import *

# ---------- QPMG by tensor Gauss-Hermite (d=2, reference) and by Fourier/Imhof (general)
def qpmg_gh(fm, g, H, Sig, y, s, n=140):
    L = np.linalg.cholesky(Sig + 1e-14 * np.eye(len(g)))
    x, w = hermegauss(n); w = w / w.sum()
    X1, X2 = np.meshgrid(x, x, indexing="ij"); W = np.outer(w, w).ravel()
    Xi = np.stack([X1.ravel(), X2.ravel()], -1)
    U = Xi @ L.T
    q = fm + U @ g + 0.5 * np.einsum("ni,ij,nj->n", U, H, U)
    l = np.exp(-0.5 * (y - q) ** 2 / s ** 2)
    Z = (W * l).sum()
    num = (W * (y - q) * l)[:, None] * (g[None] + U @ H)
    return Z, num.sum(0) / (s ** 2 * Z)

def qpmg_fourier(fm, g, H, Sig, y, s, Om=None, n=200001):
    """1-D Fourier inversion, eigen-decomposed as in the proposal."""
    L = np.linalg.cholesky(Sig + 1e-14 * np.eye(len(g)))
    a = L.T @ g; B = L.T @ H @ L
    b, U = np.linalg.eigh(B); at = U.T @ a; HL = H @ L @ U   # columns aligned with eigvecs
    Om = Om if Om else 60.0 / s
    w = np.linspace(-Om, Om, n); dw = w[1] - w[0]
    C = 1 - 1j * w[:, None] * b[None]                        # (n,r) diagonal of C
    logdet = -0.5 * np.log(C).sum(1)                          # continuous branch: log of each factor
    quad = -0.5 * w ** 2 * (at[None] ** 2 / C).sum(1)
    phi = np.exp(logdet + 1j * w * fm + quad)
    base = np.exp(-s ** 2 * w ** 2 / 2 - 1j * w * y) * phi
    Z = (s / np.sqrt(2 * np.pi)) * np.trapezoid(base, dx=dw).real
    # vector factor: i w g - w^2 H L C^{-1} a  (assembled in the eigenbasis)
    vec = 1j * w[:, None] * g[None] - (w ** 2)[:, None] * ((at[None] / C)[:, None, :] * HL[None]).sum(-1)
    num = (s / np.sqrt(2 * np.pi)) * np.trapezoid(base[:, None] * vec, dx=dw, axis=0).real
    return Z, num / (s ** 2 * Z)

print("=== 5a. The proposal's directional counterexample ===")
Sig = np.diag([0.4, 0.7]); m = np.zeros(2); y, s = 1.0, 0.8
f = lambda X: X[..., 0] * (1 + X[..., 1])
g = np.array([1.0, 0.0]); H = np.array([[0.0, 1.0], [1.0, 0.0]]); fm = 0.0
c = 0.5 * np.trace(H @ Sig); vf = g @ Sig @ g; v2 = 0.5 * np.trace((H @ Sig) @ (H @ Sig))
G_smg = (y - fm - c) / (s ** 2 + vf) * g
G_smg2 = (y - fm - c) / (s ** 2 + vf + v2) * g
Zg, G_gh = qpmg_gh(fm, g, H, Sig, y, s)
Zf, G_fo = qpmg_fourier(fm, g, H, Sig, y, s)
# finite differences of log of the true Gaussian-smoothed likelihood (exact f, not surrogate)
def logZ_true(mm, n=160):
    x, w = hermegauss(n); w = w / w.sum()
    X1, X2 = np.meshgrid(x, x, indexing="ij"); W = np.outer(w, w).ravel()
    U = np.stack([X1.ravel(), X2.ravel()], -1) @ np.linalg.cholesky(Sig).T + mm
    return np.log((W * np.exp(-0.5 * (y - f(U)) ** 2 / s ** 2)).sum())
h = 1e-5
G_fd = np.array([(logZ_true(m + e) - logZ_true(m - e)) / (2 * h) for e in (np.array([h,0]), np.array([0,h]))])
print(f"  SMG (as stated)   = [{G_smg[0]:.8f}, {G_smg[1]:.8f}]   (proposal says 0.96153846)")
print(f"  SMG-2 (exact var) = [{G_smg2[0]:.8f}, {G_smg2[1]:.8f}]")
print(f"  QPMG Gauss-Hermite= [{G_gh[0]:.8f}, {G_gh[1]:.8f}]   (proposal says 0.66267363, -0.04850552)")
print(f"  QPMG Fourier      = [{G_fo[0]:.8f}, {G_fo[1]:.8f}]   |GH-Fourier| = {np.abs(G_gh-G_fo).max():.2e}")
print(f"  finite diff (true)= [{G_fd[0]:.8f}, {G_fd[1]:.8f}]   |QPMG-FD| = {np.abs(G_gh-G_fd).max():.2e}  (f IS globally quadratic here)")
print(f"  cos(SMG, QPMG) = {G_smg@G_gh/np.linalg.norm(G_smg)/np.linalg.norm(G_gh):.6f}; angle = {np.degrees(np.arccos(G_smg@G_gh/np.linalg.norm(G_smg)/np.linalg.norm(G_gh))):.2f} deg")
print(f"  ||QPMG-SMG||/||QPMG|| = {np.linalg.norm(G_gh-G_smg)/np.linalg.norm(G_gh):.4f} ; ||QPMG-SMG2||/||QPMG|| = {np.linalg.norm(G_gh-G_smg2)/np.linalg.norm(G_gh):.4f}")
print(f"  best scalar rescale of SMG: residual = {np.linalg.norm(G_gh - (G_gh@G_smg/(G_smg@G_smg))*G_smg)/np.linalg.norm(G_gh):.4f} of ||QPMG||")

print("\n=== 5b. Closure ladder vs the EXACT guidance, Gaussian prior (posterior is exactly Gaussian) ===")
print("    relative error of the clean-shift Delta; f=|x|^2 so QPMG's quadratic surrogate is exact")
gauss = GMM([[0.0, 0.0]], [1.0], [1.0]); tg = target_sqnorm(); y2 = 2.6
rows = {}
for s2 in [0.5, 0.15, 0.05]:
    ref = GridRef(gauss, tg, y2, s2, lim=5.0, n=900)
    rng = np.random.default_rng(9)
    for t in [0.7, 0.5, 0.3]:
        al, sg = alpha_sigma(t)
        xt = al * ref.sample_py(60, rng) + sg * rng.standard_normal((60, 2))
        Dstar = ref.shift(xt, t, chunk=20); mm, SS = gauss.posterior_moments(xt, t)
        den = np.sqrt((Dstar ** 2).sum(1).mean())
        est = {"plug-in": est_plugin(tg,y2,s2,mm,SS), "SMG": est_smg(tg,y2,s2,mm,SS),
               "SMG-2": est_smg(tg,y2,s2,mm,SS,second_var=True)}
        Q = np.zeros_like(mm)
        for i in range(len(xt)):
            _, Gm = qpmg_gh(tg.f(mm[i:i+1])[0], tg.grad(mm[i:i+1])[0], tg.hess(mm[i:i+1])[0], SS[i], y2, s2, n=120)
            Q[i] = SS[i] @ Gm
        est["QPMG"] = Q
        est["LGD-MC n=16"] = est_lgd_mc(tg,y2,s2,mm,SS,n=16,rng=rng)
        key = f"s={s2}, t={t}"
        rows[key] = {k: float(np.sqrt(((v - Dstar) ** 2).sum(1).mean()) / den) for k, v in est.items()}
        print(f"  {key:16s} " + "  ".join(f"{k}={v:8.4f}" for k, v in rows[key].items()))

print("\n=== 5c. Same ladder, GMM prior (posterior NOT Gaussian) -- QPMG's assumption now fails ===")
gmm = GMM([[-1.5, 0.0], [1.5, 0.5], [0.0, 2.0]], [0.4, 0.5, 0.35], [0.5, 0.3, 0.2])
for s2 in [0.5, 0.15]:
    ref = GridRef(gmm, tg, y2, s2, lim=4.5, n=900); rng = np.random.default_rng(9)
    for t in [0.7, 0.5, 0.3]:
        al, sg = alpha_sigma(t)
        xt = al * ref.sample_py(60, rng) + sg * rng.standard_normal((60, 2))
        Dstar = ref.shift(xt, t, chunk=20); mm, SS = gmm.posterior_moments(xt, t)
        den = np.sqrt((Dstar ** 2).sum(1).mean())
        Q = np.zeros_like(mm)
        for i in range(len(xt)):
            _, Gm = qpmg_gh(tg.f(mm[i:i+1])[0], tg.grad(mm[i:i+1])[0], tg.hess(mm[i:i+1])[0], SS[i], y2, s2, n=120)
            Q[i] = SS[i] @ Gm
        est = {"SMG": est_smg(tg,y2,s2,mm,SS), "SMG-2": est_smg(tg,y2,s2,mm,SS,second_var=True), "QPMG": Q,
               "LGD-MC n=16": est_lgd_mc(tg,y2,s2,mm,SS,n=16,rng=rng)}
        print(f"  s={s2}, t={t}   " + "  ".join(f"{k}={float(np.sqrt(((v-Dstar)**2).sum(1).mean())/den):8.4f}" for k, v in est.items()))
json.dump(rows, open("exp5_results.json", "w"), indent=1)
