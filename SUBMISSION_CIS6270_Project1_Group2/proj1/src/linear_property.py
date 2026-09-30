"""A deliberately LINEAR property predictor -- the null control for SMG.

Not a competitor. Its purpose is a correctness test that no accuracy metric can
provide.

SMG's correction is 1/2 tr(H Sigma_t) with H the property Hessian. A property
that is exactly affine in its inputs has H = 0 identically, so SMG must collapse
onto plug-in guidance EXACTLY -- same trajectories, same samples, bit for bit up
to floating point. If it does not, something in the implementation is wrong, and
the failure would be invisible to every other check we run.

The test suite already verifies this analytically on a synthetic affine f
(4.4e-16). Running the same claim end to end, through the real sampler on real
molecules with a FITTED linear model, exercises the whole path: posterior,
Hutchinson probes, the pullback, the clip, and the integrator.

Construction. Invariant descriptors, then a least-squares readout:

    per element e          count_e                          (5 features)
    per element pair (e,f) sum of exp(-||r_i - r_j||^2 / 2 l^2) over i<e>, j<f>
                           at several length scales l        (15 * n_scales)
    plus an intercept

Every descriptor is a sum over atoms or ordered atom pairs of a function of
squared distance, so the whole thing is invariant to rotation, translation and
permutation by construction, exactly like the EGNN. The readout is linear in
the descriptors -- but NOT in the raw inputs, so H is not zero for the honest
version.

That distinction matters, so both are provided:

  LinearProperty(mode="descriptor")  fits QM9 to ~0.5-1.0 D. Invariant, smooth,
                                     nonzero curvature. A weak but real guide.
  LinearProperty(mode="affine")      f(x) = a . flat(x) + b, fitted the same way
                                     on the masked raw inputs. H = 0 EXACTLY.
                                     This is the null control. It predicts the
                                     property badly -- that is fine and expected,
                                     because it is testing an identity, not
                                     accuracy.

Fitting is a least-squares solve, about a minute, no gradient descent.
"""
from __future__ import annotations

import torch

SCALES = (1.0, 1.5, 2.5)


def _pair_descriptors(coords, feats, mask, scales=SCALES):
    """[B, 5 + 15*len(scales)] invariant descriptors."""
    B, N, K = feats.shape
    m = mask.unsqueeze(-1)
    f = feats * m
    counts = f.sum(1)                                     # [B, K]

    d2 = ((coords.unsqueeze(2) - coords.unsqueeze(1)) ** 2).sum(-1)     # [B,N,N]
    pm = mask.unsqueeze(2) * mask.unsqueeze(1)
    eye = torch.eye(N, device=coords.device, dtype=coords.dtype)
    pm = pm * (1.0 - eye)

    out = [counts]
    iu = torch.triu_indices(K, K, offset=0)
    for s in scales:
        w = torch.exp(-d2 / (2.0 * s * s)) * pm                          # [B,N,N]
        # element-pair resolved sum: f^T W f is [B, K, K], symmetric
        M = torch.einsum('bnk,bnm,bml->bkl', f, w, f)
        out.append(M[:, iu[0], iu[1]])                                   # [B, 15]
    return torch.cat(out, dim=1)


class LinearProperty(torch.nn.Module):
    """Least-squares property model. See module docstring for the two modes."""

    def __init__(self, mode="affine", max_atoms=29, n_types=5, scales=SCALES):
        super().__init__()
        assert mode in ("affine", "descriptor")
        self.mode = mode
        self.scales = scales
        dim = (max_atoms * 3 + max_atoms * n_types) if mode == "affine" \
            else (n_types + len(scales) * (n_types * (n_types + 1)) // 2)
        self.register_buffer("w", torch.zeros(dim))
        self.register_buffer("b", torch.zeros(()))
        self.fitted = False

    def features(self, coords, feats, mask):
        m = mask.unsqueeze(-1)
        if self.mode == "affine":
            return torch.cat([(coords * m).reshape(coords.shape[0], -1),
                              (feats * m).reshape(feats.shape[0], -1)], dim=1)
        return _pair_descriptors(coords, feats, mask, self.scales)

    @torch.no_grad()
    def fit(self, coords, feats, mask, y, ridge=1e-4, batch=512):
        """Ridge least squares. Accumulates normal equations so the design
        matrix is never materialised."""
        D = self.w.numel()
        XtX = torch.zeros(D + 1, D + 1, dtype=torch.float64)
        Xty = torch.zeros(D + 1, dtype=torch.float64)
        for i in range(0, coords.shape[0], batch):
            X = self.features(coords[i:i + batch], feats[i:i + batch],
                              mask[i:i + batch]).double()
            X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], 1)
            XtX += X.T @ X
            Xty += X.T @ y[i:i + batch].double()
        XtX += ridge * torch.eye(D + 1, dtype=torch.float64)
        sol = torch.linalg.solve(XtX, Xty)
        self.w.copy_(sol[:D].to(self.w.dtype))
        self.b.copy_(sol[D].to(self.b.dtype))
        self.fitted = True
        return self

    def forward(self, coords, feats, mask):
        return self.features(coords, feats, mask) @ self.w + self.b
