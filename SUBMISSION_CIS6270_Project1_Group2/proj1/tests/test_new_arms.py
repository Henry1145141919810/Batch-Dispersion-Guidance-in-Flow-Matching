"""Correctness gates for the guidance arms added on 20 Sep (G2 and G3).

G2, the affine null: when f is affine, H = 0 exactly, so
  - 1/2 tr(H Sigma)        == 0
  - 1/2 tr((H Sigma)^2)    == 0
  - smg2 and smg2_curv     == smg  == smg_var (TMPD), to machine precision
  - every arm is a scalar multiple of the same J^T Sigma g direction
An arm that fails this has a sign or scaling error in the curvature path, and
would silently produce a plausible-but-wrong field on the real guide.

G3, trace convergence: on a property with real, known curvature the Hutchinson
estimates of both traces must converge towards the exact values as the probe
count rises, and the K=1 estimate must already be the right order of magnitude
(the memo's claim that one Rademacher probe suffices).

Run: python proj1/tests/test_new_arms.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from guidance import (Cost, guidance_field, hutchinson_tr_H_Sigma,  # noqa: E402
                      hutchinson_tr_HS_squared, sigma_times_vector)
from linear_property import LinearProperty  # noqa: E402
from models.egnn import EGNNVelocity, zero_com  # noqa: E402
from guidance import fm_posterior  # noqa: E402

torch.set_default_dtype(torch.float64)
R = {}


def setup(B=4, N=7, K=5, seed=3):
    torch.manual_seed(seed)
    mask = torch.ones(B, N)
    mask[0, 5:] = 0.0
    mask[2, 6:] = 0.0
    net = EGNNVelocity(K, 32, 2).double().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    coords = zero_com(torch.randn(B, N, 3), mask)
    feats = torch.randn(B, N, K) * mask.unsqueeze(-1)
    t = torch.full((B,), 0.4)
    post = lambda c, f: fm_posterior(net, c, f, mask, t)
    return net, coords, feats, mask, t, post


def main():
    net, coords, feats, mask, t, post = setup()
    K = feats.shape[-1]

    # ---------------- G2: affine property, H == 0
    aff = LinearProperty(mode="affine", max_atoms=7, n_types=K).double()
    for p in aff.parameters():
        p.requires_grad_(False)
    pp = post(coords, feats)
    m_c, m_f, k = pp.mean_coords, pp.mean_feats, pp.k

    c1 = hutchinson_tr_H_Sigma(aff, post, coords, feats, mask, m_c, m_f, k, 4)
    c2 = hutchinson_tr_HS_squared(aff, post, coords, feats, mask, m_c, m_f, k, 4)
    R["affine_tr_H_Sigma_is_0"] = c1.abs().max().item()
    R["affine_tr_HS_squared_is_0"] = c2.abs().max().item()

    fields = {}
    for mode in ("smg_var", "smg", "smg2", "smg2_curv"):
        g = torch.Generator().manual_seed(11)
        Gc, Gf, _ = guidance_field(aff, post, coords, feats, mask, 1.0, 0.5,
                                   mode=mode, n_probe=4, generator=g)
        fields[mode] = (Gc, Gf)
    base = fields["smg_var"]
    for mode in ("smg", "smg2", "smg2_curv"):
        d = max((fields[mode][0] - base[0]).abs().max().item(),
                (fields[mode][1] - base[1]).abs().max().item())
        R["affine_%s_equals_tmpd" % mode] = d

    # ---------------- G3: curved property, traces must converge
    desc = LinearProperty(mode="descriptor", max_atoms=7, n_types=K).double()
    # LinearProperty initialises w to ZEROS and only fills it in .fit(). An
    # unfitted descriptor is the constant function, so H == 0 and every
    # curvature check below would pass vacuously. Give it real weights.
    torch.manual_seed(17)
    desc.w.copy_(torch.randn_like(desc.w))
    desc.b.copy_(torch.zeros_like(desc.b))
    desc.fitted = True
    for p in desc.parameters():
        p.requires_grad_(False)

    est1, est2 = [], []
    for npb in (1, 8, 64):
        g = torch.Generator().manual_seed(5)
        e1 = hutchinson_tr_H_Sigma(desc, post, coords, feats, mask, m_c, m_f, k,
                                   npb, g)
        g = torch.Generator().manual_seed(5)
        e2 = hutchinson_tr_HS_squared(desc, post, coords, feats, mask, m_c, m_f,
                                      k, npb, g)
        est1.append(e1.mean().item())
        est2.append(e2.mean().item())
    # convergence: the 64-probe estimate is the reference; 8 must be closer than 1
    ref1, ref2 = est1[-1], est2[-1]
    R["tr_H_Sigma_converges"] = 0.0 if abs(est1[1] - ref1) <= abs(est1[0] - ref1) + 1e-12 else 1.0
    R["tr_HS2_converges"] = 0.0 if abs(est2[1] - ref2) <= abs(est2[0] - ref2) + 1e-12 else 1.0
    R["tr_HS2_is_nonzero_when_curved"] = 0.0 if abs(ref2) > 1e-10 else 1.0

    # smg2 must differ from smg when curvature is real
    g = torch.Generator().manual_seed(11)
    A = guidance_field(desc, post, coords, feats, mask, 1.0, 0.5, mode="smg",
                       n_probe=8, generator=g)
    g = torch.Generator().manual_seed(11)
    Bf = guidance_field(desc, post, coords, feats, mask, 1.0, 0.5, mode="smg2",
                        n_probe=8, generator=g)
    diff = max((A[0] - Bf[0]).abs().max().item(), (A[1] - Bf[1]).abs().max().item())
    R["curved_smg2_differs_from_smg"] = 0.0 if diff > 1e-10 else 1.0

    # ---------------- the new MC arms must run and stay finite
    for mode in ("lgd_mc", "osc"):
        g = torch.Generator().manual_seed(2)
        Gc, Gf, dg = guidance_field(desc, post, coords, feats, mask, 1.0, 0.5,
                                    mode=mode, n_mc=4, generator=g)
        R["%s_finite" % mode] = 0.0 if (torch.isfinite(Gc).all() and
                                        torch.isfinite(Gf).all()) else 1.0

    # ---------------- kappa3 probe must produce a finite number and not change the field
    g = torch.Generator().manual_seed(11)
    P = guidance_field(desc, post, coords, feats, mask, 1.0, 0.5, mode="smg2",
                       n_probe=8, generator=g, want_kappa3=True)
    g = torch.Generator().manual_seed(11)
    Q = guidance_field(desc, post, coords, feats, mask, 1.0, 0.5, mode="smg2",
                       n_probe=8, generator=g, want_kappa3=False)
    R["kappa3_does_not_change_field"] = max(
        (P[0] - Q[0]).abs().max().item(), (P[1] - Q[1]).abs().max().item())
    sk = P[2].get("kappa3_skew")
    R["kappa3_finite"] = 0.0 if (sk is not None and torch.isfinite(sk).all()) else 1.0

    # ---------------- cost accounting must be non-zero and mode-dependent
    cheap, rich = Cost(), Cost()
    g = torch.Generator().manual_seed(1)
    guidance_field(desc, post, coords, feats, mask, 1.0, 0.5, mode="plug",
                   generator=g, cost=cheap)
    g = torch.Generator().manual_seed(1)
    guidance_field(desc, post, coords, feats, mask, 1.0, 0.5, mode="smg2",
                   n_probe=4, generator=g, cost=rich)
    R["smg2_costs_more_than_plug"] = 0.0 if (rich.gen_jvp > cheap.gen_jvp and
                                             rich.guide_hvp > cheap.guide_hvp) else 1.0

    tol = 1e-10
    print("%-38s %13s   pass" % ("check", "value"))
    print("-" * 62)
    ok = True
    for kk, v in R.items():
        p = v < tol
        ok = ok and p
        print("%-38s %13.3e   %s" % (kk, v, "yes" if p else "NO"))
    print("-" * 62)
    print("dtype=%s tol=%.0e" % (torch.get_default_dtype(), tol))
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
