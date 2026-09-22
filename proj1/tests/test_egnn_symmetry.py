"""S1 symmetry checks for the dense EGNN.

If any of these fail, nothing downstream is interpretable, so they run before
any training. Checks:

  1. EGNNScalar is INVARIANT to rotation, translation and atom permutation.
  2. EGNNVelocity is EQUIVARIANT: rotating the input rotates the velocity;
     translating it leaves the velocity unchanged (zero-CoM subspace).
  3. Padding is inert: padded atoms change nothing and receive zero gradient.
  4. Velocity coordinates stay in the zero-CoM subspace.

Run: .venv/Scripts/python.exe proj1/tests/test_egnn_symmetry.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.egnn import EGNNScalar, EGNNVelocity, zero_com  # noqa: E402

torch.manual_seed(20260917)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64          # float64 so tolerances mean something


def random_batch(B=3, N=9, K=5, n_real=(5, 7, 9)):
    coords = torch.randn(B, N, 3, dtype=DT, device=DEV)
    feats = torch.zeros(B, N, K, dtype=DT, device=DEV)
    mask = torch.zeros(B, N, dtype=DT, device=DEV)
    for b, n in enumerate(n_real):
        mask[b, :n] = 1.0
        feats[b, torch.arange(n), torch.randint(0, K, (n,))] = 1.0
    coords = zero_com(coords, mask)
    feats = feats * mask.unsqueeze(-1)
    t = torch.rand(B, dtype=DT, device=DEV)
    return coords, feats, mask, t


def random_rotation():
    a = torch.randn(3, 3, dtype=DT, device=DEV)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:                      # keep it a proper rotation
        q[:, 0] = -q[:, 0]
    return q


def main():
    K, H, L = 5, 32, 3
    scalar = EGNNScalar(K, H, L).to(DEV).to(DT).eval()
    vel = EGNNVelocity(K, H, L).to(DEV).to(DT).eval()
    coords, feats, mask, t = random_batch()
    results = {}

    with torch.no_grad():
        # ---------- 1. invariance of the scalar head
        y0 = scalar(coords, feats, mask, t)
        R = random_rotation()
        y_rot = scalar(coords @ R.T, feats, mask, t)
        results["scalar_rotation"] = (y_rot - y0).abs().max().item()

        shift = torch.randn(1, 1, 3, dtype=DT, device=DEV)
        y_tr = scalar(coords + shift * mask.unsqueeze(-1), feats, mask, t)
        results["scalar_translation"] = (y_tr - y0).abs().max().item()

        # permute the real atoms of sample 0
        n0 = int(mask[0].sum().item())
        perm = torch.randperm(n0, device=DEV)
        c_p, f_p = coords.clone(), feats.clone()
        c_p[0, :n0] = coords[0, perm]
        f_p[0, :n0] = feats[0, perm]
        y_pm = scalar(c_p, f_p, mask, t)
        results["scalar_permutation"] = (y_pm - y0).abs().max().item()

        # ---------- 2. equivariance of the velocity head
        vc0, vf0 = vel(coords, feats, mask, t)
        vc_r, vf_r = vel(coords @ R.T, feats, mask, t)
        results["velocity_rotation"] = (vc_r - vc0 @ R.T).abs().max().item()
        results["velocity_feats_rotation"] = (vf_r - vf0).abs().max().item()

        vc_t, _ = vel(coords + shift * mask.unsqueeze(-1), feats, mask, t)
        results["velocity_translation"] = (vc_t - vc0).abs().max().item()

        # ---------- 4. zero-CoM of the velocity
        n = mask.sum(1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        com = (vc0 * mask.unsqueeze(-1)).sum(1, keepdim=True) / n
        results["velocity_zero_com"] = com.abs().max().item()
        results["velocity_padding_zero"] = (
            vc0 * (1 - mask).unsqueeze(-1)).abs().max().item()

        # ---------- 3a. padding is inert: garbage in padded slots changes nothing
        c_g, f_g = coords.clone(), feats.clone()
        pad = (1 - mask).unsqueeze(-1)
        c_g = c_g + torch.randn_like(c_g) * 50.0 * pad
        f_g = f_g + torch.randn_like(f_g) * 50.0 * pad
        results["padding_ignored_scalar"] = (
            scalar(c_g, f_g, mask, t) - y0).abs().max().item()
        results["padding_ignored_velocity"] = (
            vel(c_g, f_g, mask, t)[0] - vc0).abs().max().item()

    # ---------- 3b. padded atoms receive exactly zero gradient
    c_req = coords.clone().requires_grad_(True)
    scalar(c_req, feats, mask, t).sum().backward()
    g = c_req.grad
    results["padding_grad_zero"] = (g * (1 - mask).unsqueeze(-1)).abs().max().item()

    # ---------- 3c. gradients through the COORDINATE-UPDATING path are finite
    # Regression test. This path was untested: d2.sqrt() in EGNNLayer has
    # infinite gradient where d2 == 0 -- the masked diagonal, and every pair of
    # padded atoms, which all sit at the origin -- and inf * 0 = NaN. Flow
    # matching produced NaN on its first batch while all the checks above still
    # passed, because EGNNScalar is built with update_coords=False and so never
    # exercises it.
    c_v = coords.clone().requires_grad_(True)
    f_v = feats.clone().requires_grad_(True)
    vc_g, vf_g = vel(c_v, f_v, mask, t)
    (vc_g.pow(2).sum() + vf_g.pow(2).sum()).backward()
    n_bad = ((~torch.isfinite(c_v.grad)).sum() + (~torch.isfinite(f_v.grad)).sum())
    results["velocity_grad_finite"] = float(n_bad.item())

    tol = 1e-9
    print("check                          max abs error      pass")
    print("-" * 58)
    ok = True
    for k, v in results.items():
        p = v < tol
        ok = ok and p
        print("%-30s %13.3e      %s" % (k, v, "yes" if p else "NO"))
    print("-" * 58)
    print("device=%s dtype=%s tol=%.0e" % (DEV, DT, tol))
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
