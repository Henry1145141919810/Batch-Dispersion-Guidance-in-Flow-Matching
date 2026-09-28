"""Independent check of the two load-bearing mathematical claims in BDG_HANDOFF.

(1) SEC 3 REDUCTION.  num_i = (y - F_i) - eta*e*(F_i - F_bar) is affine in F_i,
    so it equals (1+eta*e) * (y_eff - F_i) with y_eff = (y + eta*e*F_bar)/(1+eta*e).
    Claimed exact to 1e-14.  Also checks the generalisation: ANY per-sample
    coefficient affine in F_i is plug at a rescaled weight and shifted target,
    and a NONLINEAR function of F_i is not.

(2) SEC 2 GRADIENT.  dV_b/dF_i = 2/(B-1) (F_i - F_bar), checked against autograd
    (finite differences here: numpy only), so the dispersion term really is the
    gradient of the batch variance.

(3) SEC 3.6 TRANSFER CLAIM.  The dispersion term is scalar * g_i, the same
    direction plug already uses, so on any state space where plug's pullback is
    valid (e.g. a sum-zero tangent projection on the probability simplex) BDG's
    step is too.  Checked numerically on a simplex toy.

No model, no data, no sampling.
"""
import numpy as np

rng = np.random.default_rng(20260925)
B = 512

# ---------------------------------------------------------------- (1) reduction
F = rng.normal(3.0, 1.4, B)
y, eta, tau = 4.6627, 4.0, 1.1
F_bar = F.mean()
V_b = F.var(ddof=1)
e = (V_b - tau ** 2) / tau ** 2

num = (y - F) - eta * e * (F - F_bar)
w_eff = 1.0 + eta * e
y_eff = (y + eta * e * F_bar) / w_eff
num_red = w_eff * (y_eff - F)
print("(1) reduction  max|num - w_eff(y_eff - F)| = %.3e   (w_eff=%.6f, y_eff=%.6f, e=%+.4f)"
      % (np.abs(num - num_red).max(), w_eff, y_eff, e))

# generalisation: an arbitrary affine-in-F coefficient a + b*F_i
a, b = 0.37, -0.82
num2 = (a + b * F) * 1.0          # any arm whose numerator is affine in F
w2, y2 = -b, -a / b               # num2 = w2 (y2 - F) requires b != 0
print("    any affine numerator -> plug reweighted: max err = %.3e"
      % np.abs(num2 - w2 * (y2 - F)).max())

# a nonlinear-in-F numerator cannot be written that way: best least-squares fit
num3 = np.sign(y - F) * np.sqrt(np.abs(y - F))      # e.g. a robust/L1-ish arm
A = np.stack([np.ones(B), F], 1)
coef, *_ = np.linalg.lstsq(A, num3, rcond=None)
resid = np.abs(num3 - A @ coef).max()
print("    nonlinear numerator -> residual after best affine fit = %.3e (NOT a reweighting)"
      % resid)

# ---------------------------------------------------------------- (2) gradient
h = 1e-6
grad_fd = np.empty(B)
for i in range(8):                      # 8 coordinates is enough to establish it
    Fp = F.copy(); Fp[i] += h
    Fm = F.copy(); Fm[i] -= h
    grad_fd[i] = (Fp.var(ddof=1) - Fm.var(ddof=1)) / (2 * h)
grad_an = 2.0 / (B - 1) * (F - F_bar)
print("(2) dV_b/dF_i  max|fd - analytic| over 8 coords = %.3e  (values ~%.2e)"
      % (np.abs(grad_fd[:8] - grad_an[:8]).max(), np.abs(grad_an[:8]).mean()))

# ---------------------------------------------------------------- (3) simplex
K, L = 8, 40                            # L positions, K categories
p = rng.dirichlet(np.ones(K), size=(B, L))          # a batch on the simplex
wts = rng.normal(size=(K, L))
f = lambda P: np.einsum("blk,kl->b", P, wts)        # a linear property head
g = np.broadcast_to(wts.T, (B, L, K)).copy()        # grad f, per sample


def tangent(v):                                      # sum-zero per position
    return v - v.mean(axis=2, keepdims=True)


Fs = f(p)
Fs_bar = Fs.mean()
es = (Fs.var(ddof=1) - tau ** 2) / tau ** 2
coef_plug = (y - Fs) / 1.0
coef_bdg = ((y - Fs) - eta * es * (Fs - Fs_bar)) / 1.0
step_plug = tangent(coef_plug[:, None, None] * g)
step_bdg = tangent(coef_bdg[:, None, None] * g)
print("(3) simplex    plug step row-sums max|.| = %.3e ; BDG step row-sums max|.| = %.3e"
      % (np.abs(step_plug.sum(2)).max(), np.abs(step_bdg.sum(2)).max()))
ratio = step_bdg / np.where(np.abs(step_plug) > 1e-12, step_plug, np.nan)
print("    BDG step is a per-sample SCALAR multiple of the plug step: "
      "max spread within a sample = %.3e"
      % np.nanmax(np.nanmax(ratio, axis=(1, 2)) - np.nanmin(ratio, axis=(1, 2))))
print("    => wherever plug's pullback/projection is valid, BDG's is too "
      "(no Sigma, no matrix inverse, unlike BTVG's V_F = g' Sigma g).")
