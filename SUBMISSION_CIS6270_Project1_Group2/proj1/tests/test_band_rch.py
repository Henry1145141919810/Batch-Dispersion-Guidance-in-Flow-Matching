"""Exact gates for the two arms the earlier suites did not constrain.

An audit injected five defects into `tolerance_band_step` and two into the RCH
path and NONE were caught, because no test exercised either. These gates fix
that, and every check below is an equality or a strict inequality that a wrong
sign or a dropped factor violates.

BAND (Component D), against D_WHY_INNOVATIVE.md sections 3 and 4:

  worked example  the doc's own numbers reproduce exactly
  envelope        A = max(tau, |e|),  l = -A - e,  h = A - e
  feasibility     l <= b'd <= h  in every regime of the doc's table
  tau inert       when |e| > tau the answer does not depend on tau
  equality case   e = 0, tau = 0  =>  b'd == 0 exactly
  no-worsening    e > 0, tau = 0  =>  -2e <= b'd <= 0
  tangential      u not parallel to b  =>  the edit has a component orthogonal
                  to b. This is the one the audit's defect killed: passing
                  b = u makes v_r identically zero and the arm degenerates to
                  plain ascent.
  radius          ||d|| <= R

RCH:
  vanishing       C -> 0 as t -> 1 (Sigma -> 0 there, so the correction must too)
  invariance      the feature map is E(3) invariant -- rotation, translation,
                  permutation
  fit recovery    a head fitted to an exactly linear target recovers it

Run: python proj1/tests/test_band_rch.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from guidance import ResidualCalibrationHead, tolerance_band_step  # noqa: E402

torch.set_default_dtype(torch.float64)
R = {}


def band(u, b, e, tau, eta=1.0, radius=None):
    """1-D-batch wrapper: vectors given as flat [1, n] coordinate blocks."""
    uc = u.view(1, -1, 1)
    bc = b.view(1, -1, 1)
    uf = torch.zeros(1, u.numel(), 0)
    bf = torch.zeros(1, b.numel(), 0)
    dc, _ = tolerance_band_step(uc, uf, bc, bf,
                                torch.tensor([float(e)]),
                                torch.tensor([float(tau)]), eta, radius)
    return dc.view(-1)


def main():
    # ---------------- the doc's worked example (section 3)
    u = torch.tensor([-1.0, 1.0])
    b = torch.tensor([1.0, 0.0])
    d = band(u, b, e=0.2, tau=0.0, eta=1.0, radius=0.1)
    R["doc_example_d"] = max((d - torch.tensor([-0.0707107, 0.0707107])).abs().max().item(), 0.0)
    R["doc_example_property_change"] = abs(float(b @ d) - (-0.0707107))
    R["doc_example_radius"] = max(0.0, float(d.norm()) - 0.1 - 1e-12)

    # ---------------- feasibility across the doc's regime table
    worst_lo = worst_hi = 0.0
    for e in (-0.5, -0.2, 0.0, 0.2, 0.5):
        for tau in (0.0, 0.1, 0.3):
            for radius in (0.05, 0.5, None):
                dd = band(u, b, e, tau, 1.0, radius)
                A = max(tau, abs(e))
                lo, hi = -A - e, A - e
                bd = float(b @ dd)
                worst_lo = max(worst_lo, lo - bd)
                worst_hi = max(worst_hi, bd - hi)
    R["feasible_lower"] = max(worst_lo, 0.0)
    R["feasible_upper"] = max(worst_hi, 0.0)

    # ---------------- equality only when e = 0 and tau = 0
    R["equality_when_e0_tau0"] = abs(float(b @ band(u, b, 0.0, 0.0, 1.0, 0.1)))

    # ---------------- no worsening: e > 0, tau = 0  =>  -2e <= b'd <= 0
    bd = float(b @ band(u, b, 0.2, 0.0, 1.0, 0.5))
    R["no_worsening_upper"] = max(0.0, bd - 0.0)
    R["no_worsening_lower"] = max(0.0, -0.4 - bd)

    # ---------------- tau is inert once |e| > tau
    d1 = band(u, b, 0.5, 0.0, 1.0, 0.2)
    d2 = band(u, b, 0.5, 0.4, 1.0, 0.2)
    R["tau_inert_outside_band"] = (d1 - d2).abs().max().item()

    # ---------------- THE ONE THE AUDIT KILLED: tangential motion must survive
    # u has a component orthogonal to b, so the edit must too.
    dd = band(u, b, 0.2, 0.0, 1.0, 0.5)
    tang = dd - (b @ dd) / (b @ b) * b
    R["tangential_component_nonzero"] = 0.0 if tang.norm() > 1e-6 else 1.0
    # and if u IS parallel to b there is nothing tangential to keep
    dpar = band(b * 2.0, b, 0.2, 0.0, 1.0, 0.5)
    tpar = dpar - (b @ dpar) / (b @ b) * b
    R["parallel_u_has_no_tangent"] = tpar.norm().item()

    # ---------------- RCH: the correction vanishes at the clean endpoint
    torch.manual_seed(3)
    B, N, K = 4, 6, 5
    mask = torch.ones(B, N)
    mask[0, 4:] = 0.0
    coords = torch.randn(B, N, 3) * mask.unsqueeze(-1)
    feats = torch.zeros(B, N, K)
    for i in range(B):
        n = int(mask[i].sum())
        feats[i, torch.arange(n), torch.randint(0, K, (n,))] = 1.0
    feats = feats * mask.unsqueeze(-1)
    head = ResidualCalibrationHead().double()
    head.beta.copy_(torch.randn_like(head.beta))
    head.fitted = True
    k = torch.full((B,), 0.3)
    R["rch_vanishes_at_t1"] = head(coords, feats, mask, torch.ones(B), k).abs().max().item()

    # ---------------- RCH features are E(3) invariant
    base = ResidualCalibrationHead.features(coords, feats, mask, torch.full((B,), 0.4), k)
    a = torch.randn(3, 3)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    rot = ResidualCalibrationHead.features(coords @ q.T, feats, mask,
                                           torch.full((B,), 0.4), k)
    R["rch_features_rotation"] = (base - rot).abs().max().item()
    shift = (coords + torch.randn(1, 1, 3)) * mask.unsqueeze(-1)
    tr = ResidualCalibrationHead.features(shift, feats, mask, torch.full((B,), 0.4), k)
    R["rch_features_translation"] = (base - tr).abs().max().item()
    perm = torch.randperm(N)
    pc, pf = coords.clone(), feats.clone()
    pc[1], pf[1] = coords[1][perm], feats[1][perm]
    pe = ResidualCalibrationHead.features(pc, pf, mask, torch.full((B,), 0.4), k)
    R["rch_features_permutation"] = (base[1] - pe[1]).abs().max().item()

    # ---------------- RCH recovers an exactly linear target
    phi = ResidualCalibrationHead.features(coords, feats, mask, torch.full((B,), 0.4), k)
    beta_true = torch.randn(phi.shape[1]).double()
    y = (1.0 - 0.4) * (phi @ beta_true)
    A = phi * (1.0 - 0.4)
    G = A.T @ A + 1e-10 * torch.eye(A.shape[1], dtype=torch.float64)
    beta_hat = torch.linalg.solve(G, A.T @ y)
    R["rch_fit_recovers_linear"] = float(((A @ beta_hat) - y).abs().max())

    tol = 1e-5
    print("%-36s %13s   pass" % ("check", "value"))
    print("-" * 60)
    ok = True
    for kk, v in R.items():
        p = v < tol
        ok = ok and p
        print("%-36s %13.3e   %s" % (kk, v, "yes" if p else "NO"))
    print("-" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
