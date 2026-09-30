"""Exact closed-form gates for the guidance arms.

WHY THIS FILE EXISTS. An earlier gate file (`test_new_arms.py`) tested only
affine nulls and scale/sign-invariant relations. An audit injected SEVEN
deliberate defects -- dropping the 1/2 in each Hutchinson estimator, flipping
their signs, multiplying one by 1000, flipping the sign inside the HVP -- and
ALL SEVEN still printed ALL PASS. A test that a broken implementation passes is
not a test.

The fix is to compare against values computed in closed form, on a setting
chosen so that every quantity is exact:

  posterior   m = D * x   elementwise, so J = diag(D) and Sigma = k diag(D)
  property    f(m) = 1/2 sum_i a_i m_i^2 + sum_i b_i m_i,  so H = diag(a)

With H and Sigma both DIAGONAL, a Rademacher Hutchinson estimator is exact for
any probe count (the off-diagonal contributions cancel identically), so these
are equalities to machine precision, not statistical checks:

  1/2 tr(H Sigma)      = 1/2 k sum_i a_i D_i
  1/2 tr((H Sigma)^2)  = 1/2 k^2 sum_i (a_i D_i)^2
  g                    = a * m + b
  v_f = g' Sigma g     = k sum_i D_i g_i^2
  field                = (y - f - c)/(s^2 + v_f + v2) * D * g       [smg2]
                       + ((r^2-S)/S^2) * D * (a * D * k * g)        [smg2_curv]

Any dropped factor, wrong sign or wrong scale moves these numbers, so each is a
real constraint. Run: python proj1/tests/test_arms_exact.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from guidance import (Posterior, hutchinson_tr_H_Sigma,  # noqa: E402
                      hutchinson_tr_HS_squared, guidance_field, trace_scale,
                      sigma_times_vector)

torch.set_default_dtype(torch.float64)
R = {}
B, N, K = 3, 4, 2
DIMS = N * 3 + N * K


class Quad(torch.nn.Module):
    """f(m) = 1/2 sum a_i m_i^2 + sum b_i m_i over the flattened (coords, feats)."""

    def __init__(self, a_c, a_f, b_c, b_f):
        super().__init__()
        self.a_c, self.a_f, self.b_c, self.b_f = a_c, a_f, b_c, b_f

    def forward(self, coords, feats, mask):
        return (0.5 * (self.a_c * coords ** 2).sum(dim=(1, 2))
                + 0.5 * (self.a_f * feats ** 2).sum(dim=(1, 2))
                + (self.b_c * coords).sum(dim=(1, 2))
                + (self.b_f * feats).sum(dim=(1, 2)))


def main():
    torch.manual_seed(20260920)
    mask = torch.ones(B, N)
    coords = torch.randn(B, N, 3)
    feats = torch.randn(B, N, K)
    D_c = torch.rand(B, N, 3) + 0.5          # elementwise Jacobian, positive
    D_f = torch.rand(B, N, K) + 0.5
    a_c, a_f = torch.rand(B, N, 3) + 0.2, torch.rand(B, N, K) + 0.2
    b_c, b_f = torch.randn(B, N, 3), torch.randn(B, N, K)
    k = 0.75
    f_net = Quad(a_c, a_f, b_c, b_f)

    def post_fn(c, f):
        return Posterior(D_c * c, D_f * f, k)

    m_c, m_f = D_c * coords, D_f * feats

    # ---------------- exact references
    exact_c = 0.5 * k * ((a_c * D_c).sum(dim=(1, 2)) + (a_f * D_f).sum(dim=(1, 2)))
    exact_v2 = 0.5 * k ** 2 * (((a_c * D_c) ** 2).sum(dim=(1, 2))
                               + ((a_f * D_f) ** 2).sum(dim=(1, 2)))
    g_c, g_f = a_c * m_c + b_c, a_f * m_f + b_f
    exact_vf = k * ((D_c * g_c ** 2).sum(dim=(1, 2)) + (D_f * g_f ** 2).sum(dim=(1, 2)))
    fval = f_net(m_c, m_f, mask)

    gen = torch.Generator().manual_seed(1)
    est_c = hutchinson_tr_H_Sigma(f_net, post_fn, coords, feats, mask, m_c, m_f,
                                  k, 2, gen)
    gen = torch.Generator().manual_seed(1)
    est_v2 = hutchinson_tr_HS_squared(f_net, post_fn, coords, feats, mask, m_c,
                                      m_f, k, 2, gen)
    R["half_tr_H_Sigma_exact"] = (est_c - exact_c).abs().max().item()
    R["half_tr_HS_squared_exact"] = (est_v2 - exact_v2).abs().max().item()

    sg_c, sg_f = sigma_times_vector(post_fn, coords, feats, mask, g_c, g_f, k)
    got_vf = (g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2))
    R["v_f_exact"] = (got_vf - exact_vf).abs().max().item()

    # ---------------- the fields, in closed form
    y, s = 2.0, 0.7
    for mode, use_v2, use_curv in (("smg", False, False),
                                   ("smg2", True, False),
                                   ("smg2_curv", True, True)):
        num = y - fval - exact_c
        den = s ** 2 + exact_vf + (exact_v2 if use_v2 else 0.0)
        sc = (num / den).view(-1, 1, 1)
        w_c, w_f = sc * g_c, sc * g_f
        if use_curv:
            # H (Sigma g) = a * (k D g), and the outer Sigma comes from J^T
            b = ((num ** 2 - den) / den ** 2).view(-1, 1, 1)
            w_c = w_c + b * (a_c * sg_c)
            w_f = w_f + b * (a_f * sg_f)
        ref_c, ref_f = D_c * w_c, D_f * w_f          # J^T w with J = diag(D)

        gen = torch.Generator().manual_seed(3)
        G_c, G_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                     mode=mode, n_probe=2, generator=gen)
        R["field_%s_exact" % mode] = max((G_c - ref_c).abs().max().item(),
                                         (G_f - ref_f).abs().max().item())

    # ---------------- plug and tmpd, also closed form
    gen = torch.Generator().manual_seed(3)
    G_c, G_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                 mode="plug", generator=gen)
    sc = ((y - fval) / s ** 2).view(-1, 1, 1)
    R["field_plug_exact"] = max((G_c - D_c * sc * g_c).abs().max().item(),
                                (G_f - D_f * sc * g_f).abs().max().item())

    gen = torch.Generator().manual_seed(3)
    G_c, G_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                 mode="smg_var", generator=gen)
    sc = ((y - fval) / (s ** 2 + exact_vf)).view(-1, 1, 1)
    R["field_tmpd_exact"] = max((G_c - D_c * sc * g_c).abs().max().item(),
                                (G_f - D_f * sc * g_f).abs().max().item())

    # ---------------- the MC draws must have covariance Sigma, not Sigma^2
    # trace-matched isotropic: r^2 = tr(Sigma)/d, exact here since Sigma is diagonal
    exact_r2 = k * ((D_c.sum(dim=(1, 2)) + D_f.sum(dim=(1, 2)))) / DIMS
    gen = torch.Generator().manual_seed(9)
    # this fixture's posterior is elementwise, NOT zero-CoM projected, so
    # Sigma is full rank and the ambient dimension is the right divisor
    got_r2 = trace_scale(post_fn, coords, feats, mask, k, 4, gen, com_free=False)
    R["trace_scale_exact"] = (got_r2 - exact_r2).abs().max().item()

    # a Sigma-squared draw would have scale k^2 D^2, so this separates them
    wrong_r2 = (k ** 2) * ((D_c ** 2).sum(dim=(1, 2)) + (D_f ** 2).sum(dim=(1, 2))) / DIMS
    R["trace_scale_is_not_sigma_squared"] = (
        0.0 if (got_r2 - wrong_r2).abs().max().item() > 1e-6 else 1.0)

    # ---------------- ordering sanity: v2 > 0 with curvature, so smg2 den > smg den
    R["v2_positive_with_curvature"] = 0.0 if (exact_v2 > 0).all() else 1.0

    tol = 1e-9
    print("%-36s %13s   pass" % ("check", "max abs error"))
    print("-" * 60)
    ok = True
    for kk, v in R.items():
        p = v < tol
        ok = ok and p
        print("%-36s %13.3e   %s" % (kk, v, "yes" if p else "NO"))
    print("-" * 60)
    print("dtype=%s tol=%.0e  (H and Sigma diagonal => Hutchinson is exact)"
          % (torch.get_default_dtype(), tol))
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
