"""BDG item 4: does the dispersion SIGN survive J^T, and does the batch MEAN move?

F_i = f(m_i(x_t)); the push in x_t is  u_i = c_i * (J_i^T g_i).
To first order  dF_i = (J_i^T g_i) . u_i = c_i * |J_i^T g_i|^2 = c_i * a_i,  a_i >= 0.
Contrast BTVG-2, whose push was along h_i != g_i: dF_i = c_i * g_i^T J_i J_i^T h_i,
an indefinite bilinear form (guidance.py:1541-1549 records the same failure).
"""
import numpy as np

rng = np.random.default_rng(4)
B, n = 256, 12
eta, s2 = 4.0, 1.0

# heterogeneous, NON-symmetric J_i (the real J = dm/dx is not symmetric)
J = rng.normal(size=(B, n, n)) / np.sqrt(n)
scale = np.exp(rng.normal(0, 0.6, size=B))          # heterogeneous gain
J *= scale[:, None, None]
g = rng.normal(size=(B, n))
Jtg = np.einsum('bij,bi->bj', J, g)                 # J^T g
a = (Jtg ** 2).sum(1)
print("[4a] a_i = |J_i^T g_i|^2 : min %.4f  median %.4f  max %.4f  max/min %.1f"
      % (a.min(), np.median(a), a.max(), a.max() / a.min()))
print("     a_i >= 0 always? %s  (it is a squared norm, so PSD is automatic)"
      % bool((a >= 0).all()))

F = rng.normal(0, 1.0, size=B)
Fb, Vb = F.mean(), F.var(ddof=1)
d = F - Fb


def push(tau2, y=2.0):
    e = (Vb - tau2) / tau2
    c_cen = (y - F) / s2
    c_dis = -eta * e * d / s2
    return e, c_cen, c_dis


for tau2, lab in ((0.5 * Vb, "tighten e>0"), (2.0 * Vb, "widen e<0")):
    e, c_cen, c_dis = push(tau2)
    dF_dis = c_dis * a                       # exact 1st-order effect of the disp term
    ok = np.all(np.sign(dF_dis[d != 0]) == np.sign(c_dis[d != 0]))
    print("\n[4b] %s (e=%+.3f)" % (lab, e))
    print("     sign(dF_i from dispersion) == sign(c_dis_i) for every molecule: %s" % ok)
    print("     -> the per-molecule dispersion SIGN survives the pullback exactly.")
    # variance effect
    dV_dis = 2.0 / (B - 1) * np.sum(d * (dF_dis - dF_dis.mean()))
    raw = -2.0 * eta * e / s2 / (B - 1) * np.sum(d ** 2 * a)
    print("     dV from the dispersion term = %+.5f ; closed form -2*eta*e/s^2 * "
          "sum(d^2 a)/(B-1) = %+.5f" % (dV_dis, raw))
    print("     sum(d_i^2 a_i) = %.4f >= 0 => the sign of dV is ALWAYS -sign(e), "
          "however heterogeneous a is." % np.sum(d ** 2 * a))
    # mean drag
    drag = dF_dis.mean()
    print("     MEAN DRAG from the dispersion term = %+.5e  (it is NOT zero)" % drag)
    print("     centring term's own mean move      = %+.5e" % (c_cen * a).mean())

print("\n[4c] the drag is exactly -eta*e/s^2 * mean_i(d_i a_i); it vanishes iff "
      "a_i is uncorrelated with d_i")
for r in (-0.6, -0.3, 0.0, 0.3, 0.6, 0.9):
    z = rng.normal(size=B)
    dd = z - z.mean()
    w = rng.normal(size=B)
    aa = np.exp(0.6 * (r * z + np.sqrt(max(0, 1 - r * r)) * w))
    e = 1.0
    drag = -eta * e / s2 * np.mean(dd * aa)
    # plug's OWN drag from the same correlation, for scale
    plug_drag = np.mean((-dd) * aa) / s2
    print("     corr(a,d)=%+.2f (realised %+.2f)  disp drag %+.4f  plug drag %+.4f  "
          "ratio = eta*e = %.1f" % (r, np.corrcoef(aa, dd)[0, 1], drag, plug_drag,
                                    drag / plug_drag if plug_drag else np.nan))

print("\n[4d] identity check: total mean move = -w_eff * mean(d a)/s^2 + "
      "(y-Fbar)*mean(a)/s^2  (i.e. the drag is just the reduction again)")
for tau2 in (0.5 * Vb, 2.0 * Vb):
    e, c_cen, c_dis = push(tau2)
    tot = ((c_cen + c_dis) * a).mean()
    weff = 1 + eta * e
    pred = (2.0 - Fb) * a.mean() / s2 - weff * np.mean(d * a) / s2
    print("     e=%+.3f  measured %+.6f   predicted %+.6f   err %.2e"
          % (e, tot, pred, abs(tot - pred)))

print("\n[4e] CONTRAST: BTVG-2's direction h != g makes the sign indefinite")
h = rng.normal(size=(B, n))
Jth = np.einsum('bij,bi->bj', J, h)
bil = (Jtg * Jth).sum(1)
print("     g^T J J^T h : %.1f%% of molecules have the OPPOSITE sign to the "
      "intended one; min %.3f max %.3f" % (100 * (bil < 0).mean(), bil.min(), bil.max()))
print("     BDG has h = g, so the form is ||J^T g||^2 and can never flip. This is "
      "the one place BDG is structurally safer than BTVG-2.")

print("\n[4f] second-order check (finite difference through a nonlinear f and m)")
A = rng.normal(size=(n, n)); A = A + A.T
bvec = rng.normal(size=n)


def f(M):
    return np.tanh(M @ bvec) + 0.2 * np.einsum('bi,ij,bj->b', M, A, M)


X = rng.normal(size=(B, n)) * 0.3
M0 = np.einsum('bij,bj->bi', J, X)          # m = J x (linear surrogate)
F0 = f(M0)
G = np.einsum('bi,bij->bj', 2 * 0.2 * (M0 @ A) + (1 - np.tanh(M0 @ bvec) ** 2)[:, None] * bvec, J)
c = rng.normal(size=B)
for eps in (1e-3, 1e-4, 1e-5):
    X1 = X + eps * c[:, None] * G
    dF = f(np.einsum('bij,bj->bi', J, X1)) - F0
    pred = eps * c * (G ** 2).sum(1)
    print("     eps=%.0e  max rel err vs c*|J^T g|^2 : %.3e ; sign agreement %.1f%%"
          % (eps, np.max(np.abs(dF - pred) / np.maximum(np.abs(pred), 1e-12)),
             100 * np.mean(np.sign(dF) == np.sign(pred))))
