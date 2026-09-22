"""Synthetic test bed for inference-time guidance estimators.

Everything here is exact or Monte Carlo on a Gaussian-mixture prior with isotropic
components, so the generator is ideal and only the guidance estimator is approximate.

Conventions (match Section 3 of the memo):
  x_t = alpha_t X + sigma_t eps,  alpha = cos(pi t/2), sigma = sin(pi t/2)  (VP)
  m = E[X|x_t], Sigma = Cov[X|x_t], J = (alpha/sigma^2) Sigma
  G = score increment.  We work with the clean-prediction shift
      Delta = m_y - m = (sigma^2/alpha) G,
  which is the quantity the sampler applies and is finite at all noise levels.
"""
import numpy as np


def alpha_sigma(t):
    return np.cos(np.pi * t / 2), np.sin(np.pi * t / 2)


# ----------------------------------------------------------------------------- prior
class GMM:
    """Isotropic Gaussian mixture in d dimensions."""

    def __init__(self, means, stds, weights):
        self.a = np.atleast_2d(np.asarray(means, float))          # (K,d)
        self.p = np.asarray(stds, float) ** 2                       # (K,) variances
        self.rho = np.asarray(weights, float) / np.sum(weights)     # (K,)
        self.K, self.d = self.a.shape

    def logpdf(self, x):
        d2 = ((x[:, None, :] - self.a[None]) ** 2).sum(-1)          # (N,K)
        lp = np.log(self.rho)[None] - 0.5 * d2 / self.p[None] - 0.5 * self.d * np.log(2 * np.pi * self.p)[None]
        mx = lp.max(1, keepdims=True)
        return (mx + np.log(np.exp(lp - mx).sum(1, keepdims=True)))[:, 0]

    def sample(self, n, rng):
        k = rng.choice(self.K, size=n, p=self.rho)
        return self.a[k] + np.sqrt(self.p[k])[:, None] * rng.standard_normal((n, self.d))

    def resp(self, x):
        d2 = ((x[:, None, :] - self.a[None]) ** 2).sum(-1)
        lp = np.log(self.rho)[None] - 0.5 * d2 / self.p[None] - 0.5 * self.d * np.log(self.p)[None]
        lp -= lp.max(1, keepdims=True)
        r = np.exp(lp)
        return r / r.sum(1, keepdims=True)

    def posterior_moments(self, xt, t):
        """Exact m (N,d) and Sigma (N,d,d) of p(X | x_t)."""
        al, sg = alpha_sigma(t)
        c = 1.0 / (1.0 / self.p + al ** 2 / sg ** 2)                # (K,) comp. posterior var
        mu = c[None, :, None] * (self.a[None] / self.p[None, :, None] + al * xt[:, None, :] / sg ** 2)  # (N,K,d)
        tv = al ** 2 * self.p + sg ** 2
        d2 = ((xt[:, None, :] - al * self.a[None]) ** 2).sum(-1)
        lw = np.log(self.rho)[None] - 0.5 * d2 / tv[None] - 0.5 * self.d * np.log(tv)[None]
        lw -= lw.max(1, keepdims=True)
        pi = np.exp(lw)
        pi /= pi.sum(1, keepdims=True)                              # (N,K)
        m = (pi[:, :, None] * mu).sum(1)
        dm = mu - m[:, None, :]
        Sig = (pi[:, :, None, None] * dm[:, :, :, None] * dm[:, :, None, :]).sum(1)
        Sig += (pi * c[None]).sum(1)[:, None, None] * np.eye(self.d)[None]
        return m, Sig


# ----------------------------------------------------------------------------- targets
class Target:
    def __init__(self, name, f, grad, hess):
        self.name, self.f, self.grad, self.hess = name, f, grad, hess


def target_sqnorm():
    return Target("sqnorm",
                  lambda x: (x ** 2).sum(-1),
                  lambda x: 2 * x,
                  lambda x: np.broadcast_to(2 * np.eye(x.shape[-1]), x.shape[:-1] + (x.shape[-1],) * 2))


def target_l1():
    # piecewise linear: Hessian is zero almost everywhere (stand-in for a ReLU guide)
    return Target("l1",
                  lambda x: np.abs(x).sum(-1),
                  lambda x: np.sign(x),
                  lambda x: np.zeros(x.shape[:-1] + (x.shape[-1],) * 2))


def target_product():
    # indefinite Hessian, d = 2 only
    H = np.array([[0.0, 1.0], [1.0, 0.0]])
    return Target("product",
                  lambda x: x[..., 0] * x[..., 1],
                  lambda x: x[..., ::-1].copy(),
                  lambda x: np.broadcast_to(H, x.shape[:-1] + (2, 2)))


# ----------------------------------------------------------------------------- exact reference
class GridRef:
    """Exact tilted quantities by quadrature on a uniform grid (d = 1 or 2)."""

    def __init__(self, prior, target, y, s, lim=5.0, n=600, dtype=np.float64):
        ax = np.linspace(-lim, lim, n)
        if prior.d == 1:
            X = ax[:, None]
        else:
            X = np.stack(np.meshgrid(ax, ax, indexing="ij"), -1).reshape(-1, 2)
        self.X = X.astype(dtype)
        self.lp = prior.logpdf(X).astype(dtype)
        self.lw = (-0.5 * (target.f(X) - y) ** 2 / s ** 2).astype(dtype)
        self.prior = prior
        w = np.exp(self.lp + self.lw)
        self.py = w / w.sum()

    def sample_py(self, n, rng):
        idx = rng.choice(len(self.py), size=n, p=self.py)
        h = self.X[1, -1] - self.X[0, -1] if self.prior.d == 1 else (self.X[1, 1] - self.X[0, 1])
        return self.X[idx] + (rng.random((n, self.prior.d)) - 0.5) * abs(h)

    def mode_props(self):
        return (self.py[:, None] * self.prior.resp(self.X)).sum(0)

    def shift(self, xt, t, chunk=64):
        """Exact Delta* = m_y - m, with both means from the same grid."""
        al, sg = alpha_sigma(t)
        out = np.empty_like(xt)
        X2 = (self.X ** 2).sum(-1)
        for i in range(0, len(xt), chunk):
            xb = xt[i:i + chunk].astype(self.X.dtype)
            lg = self.lp[None] - 0.5 * (al ** 2 * X2[None] - 2 * al * xb @ self.X.T) / sg ** 2
            lg -= lg.max(1, keepdims=True)
            p0 = np.exp(lg)
            p1 = p0 * np.exp(self.lw)[None]
            m0 = (p0 @ self.X) / p0.sum(1, keepdims=True)
            m1 = (p1 @ self.X) / np.maximum(p1.sum(1, keepdims=True), 1e-300)
            out[i:i + chunk] = m1 - m0
        return out


# ----------------------------------------------------------------------------- estimators
def _mv(S, v):
    return np.einsum("nij,nj->ni", S, v)


def _chol(S):
    d = S.shape[-1]
    return np.linalg.cholesky(S + 1e-12 * np.eye(d)[None])


def est_plugin(tg, y, s, m, Sig, scale=1.0, **kw):
    g = tg.grad(m)
    return scale * _mv(Sig, g) * ((y - tg.f(m)) / s ** 2)[:, None]


def est_var_iso(tg, y, s, m, Sig, r2=None, **kw):
    """Plug-in direction with an isotropic, schedule-only variance in the denominator (Pi-GDM-like)."""
    g = tg.grad(m)
    V = r2 * (g ** 2).sum(-1)
    return _mv(Sig, g) * ((y - tg.f(m)) / (s ** 2 + V))[:, None]


def _moments(tg, m, Sig):
    g = tg.grad(m)
    H = tg.hess(m)
    HS = np.einsum("nij,njk->nik", H, Sig)
    c = 0.5 * np.trace(HS, axis1=1, axis2=2)
    vf = np.einsum("ni,nij,nj->n", g, Sig, g)
    v2 = 0.5 * np.trace(np.einsum("nij,njk->nik", HS, HS), axis1=1, axis2=2)
    return g, HS, c, vf, v2


def est_smg(tg, y, s, m, Sig, mean=True, var=True, second_var=False, curv=False, **kw):
    """Second-Moment Guidance family (stopped derivative).
    mean/var toggle the two handoff ablations; second_var adds 0.5 tr((H Sigma)^2);
    curv adds the first Stein-series direction ((r^2-S)/S^2) Sigma H Sigma g."""
    g, HS, c, vf, v2 = _moments(tg, m, Sig)
    r = y - tg.f(m) - (c if mean else 0.0)
    S = s ** 2 + (vf if var else 0.0) + (v2 if second_var else 0.0)
    vec = (r / S)[:, None] * g
    if curv:
        vec = vec + ((r ** 2 - S) / S ** 2)[:, None] * _mv(HS, g)
    return _mv(Sig, vec)


def _draw(m, Sig, n, rng, iso_r2=None, antithetic=False):
    N, d = m.shape
    k = n // 2 if antithetic else n
    z = rng.standard_normal((N, k, d))
    if iso_r2 is None:
        dl = np.einsum("nij,nkj->nki", _chol(Sig), z)
    else:
        dl = np.sqrt(iso_r2) * z
    if antithetic:
        dl = np.concatenate([dl, -dl], 1)
    return m[:, None, :] + dl                                     # (N,n,d)


def est_lgd_mc(tg, y, s, m, Sig, n=8, rng=None, iso_r2=None, antithetic=False, **kw):
    """Monte Carlo likelihood marginalisation: grad of log mean_i w(X_i), X_i = m + delta_i."""
    X = _draw(m, Sig, n, rng, iso_r2, antithetic)
    fi = tg.f(X)
    lw = -0.5 * (fi - y) ** 2 / s ** 2
    lw -= lw.max(1, keepdims=True)
    wt = np.exp(lw)
    wt /= wt.sum(1, keepdims=True)
    vec = (wt[:, :, None] * ((y - fi) / s ** 2)[:, :, None] * tg.grad(X)).sum(1)
    return _mv(Sig, vec)


def est_osc(tg, y, s, m, Sig, n=8, rng=None, iso_r2=None, cov_term=True, **kw):
    """Observable-space closure: same n guide gradients as LGD-MC (antithetic pairs), but the
    likelihood is integrated analytically over a Gaussian fitted to the scalar F = f(X)."""
    X = _draw(m, Sig, n, rng, iso_r2, antithetic=True)
    fi = tg.f(X)
    gi = tg.grad(X)
    mu = fi.mean(1)
    V = fi.var(1, ddof=1)
    gbar = gi.mean(1)
    S = s ** 2 + V
    r = y - mu
    vec = (r / S)[:, None] * gbar
    if cov_term:
        cfg = ((fi - mu[:, None])[:, :, None] * (gi - gbar[:, None, :])).sum(1) / (fi.shape[1] - 1)
        vec = vec + ((r ** 2 - S) / S ** 2)[:, None] * cfg
    return _mv(Sig, vec)


# ----------------------------------------------------------------------------- sampler
def ddim_sample(prior, shift_fn, n, rng, steps=100, t_max=0.995, t_min=0.005, clip=25.0, eta=0.0):
    """DDIM in clean-prediction form. eta=0: deterministic (probability-flow-like); eta=1: ancestral."""
    ts = np.linspace(t_max, t_min, steps + 1)
    x = rng.standard_normal((n, prior.d))
    n_clip = 0
    for i in range(steps):
        t, t2 = ts[i], ts[i + 1]
        al, sg = alpha_sigma(t)
        al2, sg2 = alpha_sigma(t2)
        m, Sig = prior.posterior_moments(x, t)
        D = shift_fn(m, Sig, x, t)
        nr = np.linalg.norm(D, axis=1)
        big = nr > clip
        n_clip += int(big.sum())
        D[big] *= (clip / nr[big])[:, None]
        my = m + D
        eps = (x - al * my) / sg
        sd = eta * np.sqrt(max((sg2 ** 2 / sg ** 2) * (1 - al ** 2 / al2 ** 2), 0.0))
        x = al2 * my + np.sqrt(max(sg2 ** 2 - sd ** 2, 0.0)) * eps + sd * rng.standard_normal(x.shape)
    return x / alpha_sigma(t_min)[0], n_clip / (n * steps)


def energy_distance(A, B, rng, k=1500):
    a = A[rng.choice(len(A), min(k, len(A)), replace=False)]
    b = B[rng.choice(len(B), min(k, len(B)), replace=False)]
    dist = lambda u, v: np.sqrt(((u[:, None, :] - v[None]) ** 2).sum(-1)).mean()
    return 2 * dist(a, b) - dist(a, a) - dist(b, b)
