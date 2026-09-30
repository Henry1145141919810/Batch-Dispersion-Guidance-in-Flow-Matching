"""E(3) invariance and differentiability gates for every predictor architecture.

A property predictor that is not invariant will appear to work and will quietly
make the guidance comparison meaningless: two identical molecules in different
orientations would get different guidance. And a predictor that is not TWICE
differentiable in coordinates and atom features cannot be f_A at all, because
SMG takes a Hessian-vector product through it.

Checks, per architecture:
  rotation      f(Rx) == f(x)            for random R in SO(3)
  reflection    f(Qx) == f(x)            for det(Q) = -1  (E(3), not just SE(3))
  translation   f(x + a) == f(x)
  permutation   f(P x, P h) == f(x, h)
  padding       padded slots change nothing
  grad          d f / d(coords, feats) exists and is finite
  hessian       H v exists and is finite  (required of any guide)

Run: python proj1/tests/test_predictor_invariance.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from models.egnn import EGNNScalar  # noqa: E402
from models.predictors import InvariantTransformer, RidgeDescriptor  # noqa: E402

torch.set_default_dtype(torch.float64)
B, N, K = 3, 7, 5


def rand_rotation():
    a = torch.randn(3, 3)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def build():
    torch.manual_seed(11)
    mask = torch.ones(B, N)
    mask[0, 5:] = 0.0
    mask[2, 6:] = 0.0
    coords = torch.randn(B, N, 3) * mask.unsqueeze(-1)
    feats = torch.zeros(B, N, K)
    for b in range(B):
        n = int(mask[b].sum())
        feats[b, torch.arange(n), torch.randint(0, K, (n,))] = 1.0
    feats = feats * mask.unsqueeze(-1)
    return coords, feats, mask


def main():
    coords, feats, mask = build()
    models = {
        "EGNNScalar": EGNNScalar(K, 64, 3).double().eval(),
        "InvariantTransformer": InvariantTransformer(K, 64, 3, n_heads=4).double().eval(),
        "RidgeDescriptor": RidgeDescriptor(K).double().eval(),
    }
    # the ridge model is all-zero until fitted, which would pass every test
    # vacuously -- give it real weights, as test_arms_exact does.
    torch.manual_seed(4)
    models["RidgeDescriptor"].w.copy_(torch.randn(models["RidgeDescriptor"].dim))
    models["RidgeDescriptor"].b.copy_(torch.randn(()))

    R = {}
    for name, net in models.items():
        for p in net.parameters():
            p.requires_grad_(False)
        base = net(coords, feats, mask)

        Q = rand_rotation()
        R["%s/rotation" % name] = (net(coords @ Q.T, feats, mask) - base).abs().max().item()

        Qr = rand_rotation().clone()
        Qr[:, 0] = -Qr[:, 0]                      # det = -1
        R["%s/reflection" % name] = (net(coords @ Qr.T, feats, mask) - base).abs().max().item()

        a = torch.randn(1, 1, 3)
        shifted = (coords + a) * mask.unsqueeze(-1)
        R["%s/translation" % name] = (net(shifted, feats, mask) - base).abs().max().item()

        # permute only the REAL atoms of row 1 (full mask), so the mask is unchanged
        perm = torch.randperm(N)
        pc = coords.clone(); pf = feats.clone()
        pc[1] = coords[1][perm]; pf[1] = feats[1][perm]
        R["%s/permutation" % name] = (net(pc, pf, mask)[1] - base[1]).abs().max().item()

        pad = (1 - mask).unsqueeze(-1)
        gc = coords + torch.randn_like(coords) * 1e3 * pad
        gf = feats + torch.randn_like(feats) * 1e3 * pad
        R["%s/padding" % name] = (net(gc, gf, mask) - base).abs().max().item()

        c = coords.clone().requires_grad_(True)
        f = feats.clone().requires_grad_(True)
        val = net(c, f, mask)
        g_c, g_f = torch.autograd.grad(val.sum(), (c, f), create_graph=True)
        R["%s/grad_finite" % name] = 0.0 if (torch.isfinite(g_c).all() and
                                             torch.isfinite(g_f).all()) else 1.0
        v_c, v_f = torch.randn_like(c), torch.randn_like(f)
        s = (g_c * v_c).sum() + (g_f * v_f).sum()
        if s.requires_grad:
            h_c, h_f = torch.autograd.grad(s, (c, f), allow_unused=True)
            ok = all(x is None or torch.isfinite(x).all() for x in (h_c, h_f))
        else:
            ok = True                              # constant Hessian (ridge): H v = 0
        R["%s/hessian_finite" % name] = 0.0 if ok else 1.0

    tol = 1e-8
    print("%-42s %13s   pass" % ("check", "max abs error"))
    print("-" * 66)
    ok_all = True
    for k, v in R.items():
        p = v < tol
        ok_all = ok_all and p
        print("%-42s %13.3e   %s" % (k, v, "yes" if p else "NO"))
    print("-" * 66)
    print("dtype=%s tol=%.0e" % (torch.get_default_dtype(), tol))
    print("ALL PASS" if ok_all else "FAILURES PRESENT")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
