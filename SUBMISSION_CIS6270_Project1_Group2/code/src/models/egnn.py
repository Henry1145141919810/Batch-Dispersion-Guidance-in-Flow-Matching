"""Dense E(n)-equivariant graph network (Satorras et al., 2021).

Dense N x N tensors with a padding mask, deliberately: QM9 molecules have at most
29 atoms, so the sparse machinery of PyTorch Geometric buys nothing here and
costs a source build of torch-scatter against a brand-new CUDA. Everything below
is plain PyTorch.

Two heads share this backbone:

  EGNNVelocity  - equivariant, for the flow-matching / diffusion velocity
  EGNNScalar    - invariant, for the property predictors f_A and f_B

Conventions, fixed here and used everywhere downstream:

  coords  [B, N, 3]   zero centre-of-mass over VALID atoms only
  feats   [B, N, K]   one-hot atom type, treated as continuous
  mask    [B, N]      1.0 for a real atom, 0.0 for padding

Padding never contributes: messages are masked on both endpoints, the diagonal
is removed, and pooling divides by the true atom count.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def zero_com(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract the mean coordinate over valid atoms. Padding stays at zero.

    The generative model's distribution lives on the zero-CoM subspace, so this
    is applied to data, to noise, and to any guidance direction.
    """
    m = mask.unsqueeze(-1)                                  # [B, N, 1]
    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)           # [B, 1, 1]
    centre = (coords * m).sum(dim=1, keepdim=True) / n      # [B, 1, 3]
    return (coords - centre) * m


def pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """[B, N, N] mask that is 1 only for ordered pairs of two distinct real atoms."""
    pm = mask.unsqueeze(2) * mask.unsqueeze(1)              # [B, N, N]
    eye = torch.eye(mask.shape[1], device=mask.device, dtype=pm.dtype)
    return pm * (1.0 - eye)


def _mlp(sizes: list[int], out_act: nn.Module | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class EGNNLayer(nn.Module):
    """One equivariant layer.

        m_ij = phi_e(h_i, h_j, d_ij^2)
        h_i <- h_i + phi_h(h_i, sum_j m_ij)
        x_i <- x_i + sum_j (x_i - x_j) * phi_x(m_ij)        (only if update_coords)

    Messages depend on the squared distance and coordinate updates on the
    difference vectors, so nothing references a fixed frame: rotate or translate
    the input and the output follows.
    """

    def __init__(self, hidden: int, update_coords: bool = True,
                 coord_scale: float = 1.0):
        super().__init__()
        self.update_coords = update_coords
        self.coord_scale = coord_scale
        self.edge_mlp = _mlp([2 * hidden + 1, hidden, hidden], nn.SiLU())
        self.node_mlp = _mlp([2 * hidden, hidden, hidden])
        if update_coords:
            self.coord_mlp = _mlp([hidden, hidden, 1])
            # small init so early layers barely move atoms
            nn.init.zeros_(self.coord_mlp[-1].bias)
            nn.init.normal_(self.coord_mlp[-1].weight, std=1e-3)

    def forward(self, h, x, mask):
        B, N, _ = h.shape
        pm = pair_mask(mask).unsqueeze(-1)                  # [B, N, N, 1]

        diff = x.unsqueeze(2) - x.unsqueeze(1)              # [B, N, N, 3]
        d2 = (diff ** 2).sum(-1, keepdim=True)              # [B, N, N, 1]

        hi = h.unsqueeze(2).expand(B, N, N, h.shape[-1])
        hj = h.unsqueeze(1).expand(B, N, N, h.shape[-1])
        m = self.edge_mlp(torch.cat([hi, hj, d2], dim=-1)) * pm

        h = h + self.node_mlp(torch.cat([h, m.sum(dim=2)], dim=-1))
        h = h * mask.unsqueeze(-1)

        if self.update_coords:
            # Normalising by (d + 1) keeps the update bounded for close atoms.
            # The epsilon is NOT cosmetic: d2 is exactly 0 on the diagonal and
            # between padded atoms, sqrt has infinite gradient there, and
            # inf * 0 (from the mask) is NaN. Without it every backward pass
            # through a coordinate-updating layer produces NaN immediately.
            w = self.coord_mlp(m) * pm / (torch.sqrt(d2 + 1e-8) + 1.0)
            x = x + self.coord_scale * (diff * w).sum(dim=2)
            x = x * mask.unsqueeze(-1)
        return h, x


class _Trunk(nn.Module):
    """Shared embedding + stack of layers."""

    def __init__(self, n_types: int, hidden: int, n_layers: int,
                 update_coords: bool):
        super().__init__()
        # +1 input channel for the diffusion/flow time t
        self.embed = nn.Linear(n_types + 1, hidden)
        self.layers = nn.ModuleList(
            EGNNLayer(hidden, update_coords=update_coords)
            for _ in range(n_layers)
        )

    def forward(self, coords, feats, mask, t):
        # t arrives as [B]; broadcast to a per-atom channel
        tc = t.view(-1, 1, 1).expand(-1, feats.shape[1], 1)
        h = self.embed(torch.cat([feats, tc], dim=-1)) * mask.unsqueeze(-1)
        x = coords * mask.unsqueeze(-1)
        for layer in self.layers:
            h, x = layer(h, x, mask)
        return h, x


class EGNNVelocity(nn.Module):
    """Equivariant velocity field for flow matching, or eps for diffusion.

    Returns (v_coords, v_feats). The coordinate part is projected back onto the
    zero-CoM subspace, so integrating it never introduces a net translation.
    """

    def __init__(self, n_types: int = 5, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        self.trunk = _Trunk(n_types, hidden, n_layers, update_coords=True)
        self.feat_out = _mlp([hidden, hidden, n_types])

    def forward(self, coords, feats, mask, t):
        coords = zero_com(coords, mask)
        h, x = self.trunk(coords, feats, mask, t)
        v_coords = zero_com(x - coords, mask)
        v_feats = self.feat_out(h) * mask.unsqueeze(-1)
        return v_coords, v_feats


class EGNNScalar(nn.Module):
    """Invariant scalar property predictor.

    A SINGLE linear head on the pooled embedding, on purpose: it keeps the
    readout weight vector a constant we can inspect, and keeps the property's
    level sets exactly planar in embedding space. Exposes the pooled embedding
    so a diversity term can act in the same coordinates.
    """

    def __init__(self, n_types: int = 5, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        self.trunk = _Trunk(n_types, hidden, n_layers, update_coords=False)
        self.pool_mlp = _mlp([hidden, hidden, hidden], nn.SiLU())
        self.head = nn.Linear(hidden, 1, bias=True)

    def embed(self, coords, feats, mask, t=None):
        """Pooled invariant embedding h. [B, hidden]"""
        if t is None:
            t = torch.ones(coords.shape[0], device=coords.device)
        coords = zero_com(coords, mask)
        h, _ = self.trunk(coords, feats, mask, t)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / n
        return self.pool_mlp(pooled)

    def forward(self, coords, feats, mask, t=None):
        return self.head(self.embed(coords, feats, mask, t)).squeeze(-1)
