"""Experiment 8 (DBFG): facet/boundary estimator vs the score-function estimator the proposal itself
names as its control, at MATCHED decoded-guide-call budget.

Setup: A atoms x C type channels, correlated Gaussian proposal N(m, Sigma). Reward depends on the
argmax-decoded type vector only, so grad_m log Z_D is a pure boundary quantity. Ground truth by 4e6 draws.
Estimators:
  SF   : score-function / REINFORCE, grad = Sigma^{-1} E[(X-m) l] / E[l]          (n guide calls)
  SF-a : the same with antithetic pairs
  FACET: in-cell term is zero here, so grad = sum over facets of the surface term (paired A/B contrasts)
"""
import numpy as np
from scipy import stats
from lib import energy_distance

rng = np.random.default_rng(11)
A, C = 4, 3
d = A * C
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
Sig = (Q * np.linspace(1.0, 0.25, d)) @ Q.T * 0.55        # correlated, anisotropic
m = rng.standard_normal(d) * 0.35
L = np.linalg.cholesky(Sig); Si = np.linalg.inv(Sig)
rew = rng.random(C ** A)                                   # reward per decoded type vector
def code(T): return (T * (C ** np.arange(A))).sum(-1)
def ell(X):
    T = X.reshape(X.shape[:-1] + (A, C)).argmax(-1)
    return rew[code(T)]


# ---- facet enumeration: A * C(C,2) = 4*3 = 12 facets
facets = []
for i in range(A):
    for a in range(C):
        for b in range(a + 1, C):
            n = np.zeros(d); n[i*C+a] = 1/np.sqrt(2); n[i*C+b] = -1/np.sqrt(2)
            facets.append((i, a, b, n))

def facet_grad(nf, npf, rg):
    """Sample nf facets w.p. proportional to their boundary density, npf draws each. Returns (grad, calls)."""
    dens = np.array([stats.norm.pdf((f[3] @ m) / np.sqrt(f[3] @ Sig @ f[3])) / np.sqrt(f[3] @ Sig @ f[3]) for f in facets])
    p = dens / dens.sum(); calls = 0; G = np.zeros(d)
    idx = rg.choice(len(facets), size=nf, p=p)
    for j in idx:
        i, a, b, n = facets[j]
        v = n @ Sig @ n; bb = n @ m
        mF = m - Sig @ n * bb / v
        SF = Sig - np.outer(Sig @ n, Sig @ n) / v
        w, U = np.linalg.eigh(SF); w = np.maximum(w, 0)
        Xs = mF + (rg.standard_normal((npf, d)) * np.sqrt(w)) @ U.T
        lg = Xs.reshape(npf, A, C)
        other = np.delete(np.arange(C), [a, b])
        tie = 0.5 * (lg[:, i, a] + lg[:, i, b])
        ok = (tie[:, None] >= lg[:, i, other]).all(1) if len(other) else np.ones(npf, bool)
        XA = Xs.copy(); XA[:, i*C+a] = 1e3
        XB = Xs.copy(); XB[:, i*C+b] = 1e3
        diff = (ell(XA) - ell(XB)) * ok
        calls += 2 * npf
        G += n * (dens[j] / p[j]) * diff.mean() / nf
    return G, calls

def sf_grad(n, rg, anti=False):
    k = n // 2 if anti else n
    z = rg.standard_normal((k, d))
    if anti: z = np.concatenate([z, -z], 0)
    X = z @ L.T + m; l = ell(X)
    return Si @ ((X - m) * l[:, None]).mean(0) / max(l.mean(), 1e-12), len(X)

