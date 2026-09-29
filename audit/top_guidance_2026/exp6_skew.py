"""Experiment 6: the third cumulant of the clean posterior is available at inference time.

Identity (exponential-family consequence of Tweedie):  d Sigma_ij / d x_k = (alpha/sigma^2) K_ijk,
K = third central moment of p(X|x_t).  So K[g,g] = (sigma^2/alpha) grad_x ( g' Sigma(x) g ), i.e. ONE
extra directional derivative of the Jacobian -- no training, no posterior samples.

Closure under test (SCG, skew-corrected guidance).  With F = f(X), u = X - m:
   Delta = m_y - m = E[u w(F)]/E[w(F)] = c1 * M1 + c2 * (M2 - V)
where E[u|F] ~= c1 (F-mu) + c2 ((F-mu)^2 - V) and M1, M2 are tilted moments of the SCALAR F under a
3-cumulant (shifted-gamma) law.  Dropping c2 and using a Gaussian F recovers SMG-2 exactly.
"""
import numpy as np
from scipy import stats
from lib import *
from lib import _mv

# ---------------- exact third central moment of the GMM posterior
def post_K(prior, xt, t):
    al, sg = alpha_sigma(t)
    c = 1.0/(1.0/prior.p + al**2/sg**2)
    mu = c[None,:,None]*(prior.a[None]/prior.p[None,:,None] + al*xt[:,None,:]/sg**2)
    tv = al**2*prior.p + sg**2
    d2 = ((xt[:,None,:] - al*prior.a[None])**2).sum(-1)
    lw = np.log(prior.rho)[None] - 0.5*d2/tv[None] - 0.5*prior.d*np.log(tv)[None]
    lw -= lw.max(1, keepdims=True); pi = np.exp(lw); pi /= pi.sum(1, keepdims=True)
    m = (pi[:,:,None]*mu).sum(1); dd = mu - m[:,None,:]
    K = np.einsum("nk,nki,nkj,nkl->nijl", pi, dd, dd, dd)
    I = np.eye(prior.d)
    cc = (pi*c[None]).sum(1)
    for (i,j,l) in [(0,1,2),(1,0,2),(2,0,1)]:
        pass
    T = np.einsum("nk,nki,k->ni", pi, dd, c)
    K = K + np.einsum("ni,jl->nijl", T, I) + np.einsum("nj,il->nijl", T, I) + np.einsum("nl,ij->nijl", T, I)
    return m, K

print("=== 6a. Identity  dSigma/dx = (alpha/sigma^2) K   (finite differences on the GMM) ===")
gmm = GMM([[-1.5,0.],[1.5,0.5],[0.,2.]], [0.4,0.5,0.35], [0.5,0.3,0.2])
rng = np.random.default_rng(2)
for t in [0.7, 0.5, 0.3]:
    al, sg = alpha_sigma(t); xt = rng.standard_normal((5,2))*1.2
    _, K = post_K(gmm, xt, t)
    h = 1e-4; dS = np.zeros((5,2,2,2))
    for k in range(2):
        e = np.zeros(2); e[k] = h
        dS[:,:,:,k] = (gmm.posterior_moments(xt+e,t)[1] - gmm.posterior_moments(xt-e,t)[1])/(2*h)
    print(f"  t={t}: max |dSigma/dx - (a/s^2)K| / max|dSigma/dx| = {np.abs(dS - (al/sg**2)*K).max()/np.abs(dS).max():.2e}")

# ---------------- the SCG closure
def pearson3(mu, V, gam, quad=301):
    """Shifted-gamma law matching (mean, variance, skew). Returns grid and density."""
    sd = np.sqrt(V)
    if abs(gam) < 1e-3:
        z = np.linspace(-9, 9, quad)
        return mu + sd*z, stats.norm.pdf(z)/sd, 3*V**2
    k = 4.0/gam**2; th = sd*abs(gam)/2.0; loc = mu - 2*sd/abs(gam)*np.sign(gam)*np.sign(gam)
    loc = mu - 2*sd/gam if gam > 0 else mu + 2*sd/abs(gam)
    q = stats.gamma.ppf([1e-9, 1-1e-9], k, scale=th)
    if gam > 0:
        F = np.linspace(loc+q[0], loc+q[1], quad); dens = stats.gamma.pdf(F-loc, k, scale=th)
    else:
        F = np.linspace(loc-q[1], loc-q[0], quad); dens = stats.gamma.pdf(loc-F, k, scale=th)
    return F, dens, V**2*(3 + 6/k)


def scg(tg, y, s, m, Sig, K, use_c2=True, quad=301, skew_cap=1.5):
    g = tg.grad(m); mu = tg.f(m)
    V = np.einsum("ni,nij,nj->n", g, Sig, g)
    M3 = np.einsum("ni,nj,nl,nijl->n", g, g, g, K)
    C1 = _mv(Sig, g)
    C2 = np.einsum("ni,nj,nijl->nl", g, g, K)
    out = np.zeros_like(m)
    for n in range(len(m)):
        Vn = max(V[n], 1e-30); gam = float(np.clip(M3[n]/Vn**1.5, -skew_cap, skew_cap))
        F, dens, m4 = pearson3(mu[n], Vn, gam, quad)
        w = dens*np.exp(-0.5*(F-y)**2/s**2); Z = np.trapezoid(w, F)
        if not np.isfinite(Z) or Z <= 1e-300:
            out[n] = C1[n]*((y-mu[n])/(s**2+Vn)); continue
        M1 = np.trapezoid(w*(F-mu[n]), F)/Z; M2 = np.trapezoid(w*(F-mu[n])**2, F)/Z
        if not use_c2:
            out[n] = C1[n]*M1/Vn; continue
        A = np.array([[Vn, M3[n]], [M3[n], m4 - Vn**2]])
        try:
            c = np.linalg.solve(A, np.array([M1, M2 - Vn]))
            out[n] = c[0]*C1[n] + c[1]*C2[n]
        except np.linalg.LinAlgError:
            out[n] = C1[n]*M1/Vn
    return out


print("\n=== 6b. MAG's counterexample: g = 0, two posteriors with identical mean and variance ===")
print("    P+ : P(X=-0.5)=0.8, P(X=2)=0.2 ;  P- mirrored.  f=X^2, y=4, s=1.  mean 0, var 1.")
for sgn, name in [(1, "P+"), (-1, "P-")]:
    supp = sgn*np.array([-0.5, 2.0]); pr = np.array([0.8, 0.2])
    w = np.exp(-0.5*(supp**2 - 4.0)**2/1.0)
    tilt = (pr*w*supp).sum()/(pr*w).sum()
    V = (pr*supp**2).sum(); M3 = (pr*supp**3).sum()
    print(f"  {name}: exact tilted mean = {tilt:+.10f}   var = {V:.4f}   third cumulant = {M3:+.4f}")
print("  every mean+covariance-only field (plug-in, SMG, SMG-2, QPMG, PiGDM) is proportional to grad f(m) = 2m = 0")
print("  -> all return exactly 0 for BOTH, so no scalar or matrix rescaling can recover the sign.")
print("  the two cases differ ONLY in the third cumulant, which the identity above makes observable.")

print("\n=== 6c. Does the skew term help where it should? 2-D GMM, relative error of the shift ===")
tg, y2 = target_sqnorm(), 2.6
print(f"  {'setting':16s} {'plug-in':>9s} {'SMG':>8s} {'SMG-2':>8s} {'SCG(no c2)':>11s} {'SCG':>8s} {'LGD16':>8s}")
for s2 in [0.5, 0.15]:
    ref = GridRef(gmm, tg, y2, s2, lim=4.5, n=1000); rr = np.random.default_rng(9)
    for t in [0.7, 0.5, 0.3]:
        al, sg = alpha_sigma(t)
        xt = al*ref.sample_py(80, rr) + sg*rr.standard_normal((80,2))
        D = ref.shift(xt, t, chunk=20); mm, SS = gmm.posterior_moments(xt, t); _, KK = post_K(gmm, xt, t)
        den = np.sqrt((D**2).sum(1).mean()); e = lambda V: float(np.sqrt(((V-D)**2).sum(1).mean())/den)
        print(f"  s={s2:<5} t={t:<4} {e(est_plugin(tg,y2,s2,mm,SS)):9.3f} {e(est_smg(tg,y2,s2,mm,SS)):8.4f} "
              f"{e(est_smg(tg,y2,s2,mm,SS,second_var=True)):8.4f} {e(scg(tg,y2,s2,mm,SS,KK,use_c2=False)):11.4f} "
              f"{e(scg(tg,y2,s2,mm,SS,KK)):8.4f} {e(est_lgd_mc(tg,y2,s2,mm,SS,n=16,rng=rr)):8.2f}")
