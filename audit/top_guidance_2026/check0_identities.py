"""Check 0: (a) Identity 1 vs finite differences of log E[w|x_t];
(b) behaviour of SMG near a critical point of f (1-D, Gaussian prior, f = x^2)."""
import numpy as np
from lib import *

rng = np.random.default_rng(0)

# (a) 2-D GMM, f = |x|^2
prior = GMM([[-1.5, 0.0], [1.5, 0.5], [0.0, 2.0]], [0.4, 0.5, 0.35], [0.5, 0.3, 0.2])
tg, y, s = target_sqnorm(), 2.6, 0.15
ref = GridRef(prior, tg, y, s, lim=5.0, n=600)

def log_h(xt, t):
    al, sg = alpha_sigma(t)
    X2 = (ref.X ** 2).sum(-1)
    lg = ref.lp[None] - 0.5 * (al ** 2 * X2[None] - 2 * al * xt @ ref.X.T) / sg ** 2
    mx = lg.max(1, keepdims=True)
    num = np.log((np.exp(lg - mx) * np.exp(ref.lw)[None]).sum(1))
    den = np.log(np.exp(lg - mx).sum(1))
    return num - den

print("(a) Identity 1: max |G_fd - (alpha/sigma^2)(m_y - m)| / max|G|")
for t in [0.8, 0.5, 0.2]:
    al, sg = alpha_sigma(t)
    xt = al * ref.sample_py(6, rng) + sg * rng.standard_normal((6, 2))
    G_id = (al / sg ** 2) * ref.shift(xt, t)
    h = 1e-4
    G_fd = np.zeros_like(xt)
    for j in range(2):
        e = np.zeros(2); e[j] = h
        G_fd[:, j] = (log_h(xt + e, t) - log_h(xt - e, t)) / (2 * h)
    print(f"  t={t}: rel. max error = {np.abs(G_fd - G_id).max() / np.abs(G_id).max():.2e}")
    m_an, _ = prior.posterior_moments(xt, t)
    # grid mean vs analytic mean
    lg = ref.lp[None] - 0.5 * (al ** 2 * (ref.X ** 2).sum(-1)[None] - 2 * al * xt @ ref.X.T) / sg ** 2
    p0 = np.exp(lg - lg.max(1, keepdims=True)); m_gr = (p0 @ ref.X) / p0.sum(1, keepdims=True)
    print(f"         grid-vs-analytic posterior mean, max abs diff = {np.abs(m_gr - m_an).max():.2e}")

# (b) 1-D critical point
print("\n(b) 1-D, X~N(0,1), f=x^2, y=1: slope of the shift Delta at x_t -> 0  (exact s->0 limit: alpha^3/sigma^2)")
p1 = GMM([[0.0]], [1.0], [1.0]); tq = target_sqnorm()
print("   t     s    exact   plug-in   SMG(as stated)   SMG-2(+0.5tr(HS)^2)   limit")
for t in [0.7, 0.5, 0.3]:
    al, sg = alpha_sigma(t)
    for s1 in [0.3, 0.1, 0.03]:
        r1 = GridRef(p1, tq, 1.0, s1, lim=6.0, n=24001)
        xt = np.array([[0.02]])
        m, S = p1.posterior_moments(xt, t)
        ex = r1.shift(xt, t)[0, 0] / 0.02
        pl = est_plugin(tq, 1.0, s1, m, S)[0, 0] / 0.02
        sm = est_smg(tq, 1.0, s1, m, S)[0, 0] / 0.02
        s2 = est_smg(tq, 1.0, s1, m, S, second_var=True)[0, 0] / 0.02
        print(f"  {t:.1f}  {s1:5.2f}  {ex:7.3f}  {pl:8.2f}  {sm:12.2f}  {s2:16.3f}  {al**3/sg**2:10.3f}")
