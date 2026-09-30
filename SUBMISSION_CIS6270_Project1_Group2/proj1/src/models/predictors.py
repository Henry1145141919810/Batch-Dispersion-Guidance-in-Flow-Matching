"""Alternative property-predictor architectures, for architecture diversity.

Every class here exposes the SAME interface as `models.egnn.EGNNScalar`:

    forward(coords, feats, mask, t=None) -> [B]      property in NORMALISED units
    embed(coords, feats, mask, t=None)   -> [B, H]   pooled invariant embedding

so they drop into `train_predictor.py`, into `PhysicalProperty`, and into the
guidance code with no changes. That matters for f_A in particular: guidance
differentiates the predictor with respect to BOTH coordinates and the
continuous atom features, and takes a Hessian-vector product through it, so an
architecture that is not twice differentiable in those inputs cannot be a guide
at all.

WHY MORE THAN ONE ARCHITECTURE. f_A steers and f_B scores. If they share an
architecture they share inductive-bias blind spots, and the f_A - f_B gap goes
quiet exactly where guidance has found one. The split protocol makes their DATA
disjoint; using different architectures makes their FAILURE MODES disjoint too.
See FA_FB_ARCHITECTURE_DECISION.md.

E(3) INVARIANCE. Both models below are invariant to rotation, translation and
permutation BY CONSTRUCTION, not by augmentation:

  * translation -- coordinates are centred (zero centre of mass) on entry;
  * rotation    -- coordinates enter ONLY through pairwise distances;
  * permutation -- the readout is a masked mean over atoms, and attention is
                   permutation-equivariant.

`tests/test_predictor_invariance.py` checks all three numerically.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .egnn import zero_com


def _rbf(d, n_rbf=32, cutoff=10.0):
    """Gaussian radial basis expansion of distances, as SchNet and DimeNet use.

    A raw distance is a poor input to an MLP (one number, huge dynamic range);
    a smooth basis gives the network a resolution it can actually learn on.
    """
    centres = torch.linspace(0.0, cutoff, n_rbf, device=d.device, dtype=d.dtype)
    gamma = (n_rbf / cutoff) ** 2
    return torch.exp(-gamma * (d.unsqueeze(-1) - centres) ** 2)


class InvariantTransformer(nn.Module):
    """Graph transformer over atoms, with distances entering as an attention bias.

    Tokens are atoms. Attention logits get an additive, learned function of the
    pairwise distance:

        logit_ij = (q_i . k_j) / sqrt(dh) + b(rbf(d_ij))

    which is the standard distance-bias construction (Graphormer / Uni-Mol).
    Because the only geometric input is d_ij, the whole model is invariant to
    rotation and translation; because attention treats tokens as a set, it is
    permutation-equivariant, and the masked-mean readout makes it invariant.

    This is a genuinely different inductive bias from EGNN: EGNN aggregates over
    a local neighbourhood with equal weights, this attends globally with learned
    weights. That is the point -- different blind spots.
    """

    def __init__(self, n_types: int = 5, hidden: int = 128, n_layers: int = 4,
                 n_heads: int = 8, n_rbf: int = 32, cutoff: float = 10.0,
                 dropout: float = 0.0):
        super().__init__()
        assert hidden % n_heads == 0, "hidden must divide n_heads"
        self.n_heads, self.dh = n_heads, hidden // n_heads
        self.n_rbf, self.cutoff = n_rbf, cutoff
        self.embed_in = nn.Linear(n_types + 1, hidden)
        self.dist_bias = nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU(),
                                       nn.Linear(hidden, n_heads))
        self.qkv = nn.ModuleList(nn.Linear(hidden, 3 * hidden) for _ in range(n_layers))
        self.proj = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.n1 = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.n2 = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.ff = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                          nn.Dropout(dropout), nn.Linear(2 * hidden, hidden))
            for _ in range(n_layers))
        self.pool_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 1, bias=True)

    def embed(self, coords, feats, mask, t=None):
        B, N, _ = coords.shape
        if t is None:
            t = torch.ones(B, device=coords.device, dtype=coords.dtype)
        m = mask.unsqueeze(-1)
        coords = zero_com(coords, mask)

        h = self.embed_in(torch.cat([feats, t.view(-1, 1, 1).expand(B, N, 1)], -1)) * m

        # pairwise distances -> per-head additive bias. Padded pairs are masked
        # out of the softmax, so their distance value is irrelevant.
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        d = (diff ** 2).sum(-1).clamp(min=1e-12).sqrt()
        bias = self.dist_bias(_rbf(d, self.n_rbf, self.cutoff))     # [B,N,N,heads]
        bias = bias.permute(0, 3, 1, 2)                              # [B,heads,N,N]
        pair = (mask.unsqueeze(1) * mask.unsqueeze(2)).unsqueeze(1)  # [B,1,N,N]
        neg = torch.finfo(h.dtype).min
        bias = bias.masked_fill(pair == 0, neg)

        for qkv, proj, n1, n2, ff in zip(self.qkv, self.proj, self.n1, self.n2, self.ff):
            x = n1(h)
            q, k, v = qkv(x).chunk(3, dim=-1)
            q = q.view(B, N, self.n_heads, self.dh).transpose(1, 2)
            k = k.view(B, N, self.n_heads, self.dh).transpose(1, 2)
            v = v.view(B, N, self.n_heads, self.dh).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) + bias
            att = att.softmax(-1)
            out = (att @ v).transpose(1, 2).reshape(B, N, -1)
            h = h + proj(out) * m
            h = h + ff(n2(h)) * m

        n = mask.sum(1, keepdim=True).clamp(min=1.0)
        pooled = (h * m).sum(1) / n
        return self.pool_mlp(pooled)

    def forward(self, coords, feats, mask, t=None):
        return self.head(self.embed(coords, feats, mask, t)).squeeze(-1)


class RidgeDescriptor(nn.Module):
    """Ridge regression on fixed E(3)-invariant descriptors. The floor.

    No learned representation at all: a fixed feature map, then one linear
    layer. It exists to answer "how much of the property does a linear model on
    hand-made features already get?", which is the honest baseline every
    learned predictor should be quoted against. It is also the ONLY predictor
    here whose Hessian is constant, which makes it a useful guidance control.

    Features, all invariant: type histogram, per-type-pair distance sums at
    several Gaussian scales, radius of gyration, atom count.

    Trained by closed-form ridge (`fit`), not by SGD, so "training time" is
    seconds. Exposed through the same nn.Module interface anyway.
    """

    def __init__(self, n_types: int = 5, scales=(1.0, 2.0, 4.0), hidden: int = 0):
        super().__init__()
        self.n_types, self.scales = n_types, tuple(scales)
        n_pair = n_types * (n_types + 1) // 2
        self.dim = n_types + len(self.scales) * n_pair + 2
        self.register_buffer("w", torch.zeros(self.dim))
        self.register_buffer("b", torch.zeros(()))
        self.fitted = False

    def embed(self, coords, feats, mask, t=None):
        m = mask.unsqueeze(-1)
        coords = zero_com(coords, mask)
        n = mask.sum(1, keepdim=True).clamp(min=1.0)
        hist = (feats * m).sum(1)                                   # [B,K]
        diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        d2 = (diff ** 2).sum(-1)
        pair = mask.unsqueeze(1) * mask.unsqueeze(2)
        cols = [hist]
        for s in self.scales:
            g = torch.exp(-d2 / (2.0 * s ** 2)) * pair
            # type-pair resolved sums, upper triangle only (symmetric)
            tp = torch.einsum("bij,bik,bjl->bkl", g, feats * m, feats * m)
            iu = torch.triu_indices(self.n_types, self.n_types, device=coords.device)
            cols.append(tp[:, iu[0], iu[1]])
        rg = ((coords ** 2).sum(-1) * mask).sum(1, keepdim=True) / n
        cols += [rg, n]
        return torch.cat(cols, dim=1)

    def forward(self, coords, feats, mask, t=None):
        return self.embed(coords, feats, mask, t) @ self.w + self.b

    @torch.no_grad()
    def fit(self, coords, feats, mask, y, ridge=1e-3, batch=512, device="cpu"):
        D = self.dim
        XtX = torch.zeros(D + 1, D + 1, dtype=torch.float64, device=device)
        Xty = torch.zeros(D + 1, dtype=torch.float64, device=device)
        for i in range(0, coords.shape[0], batch):
            X = self.embed(coords[i:i + batch].to(device), feats[i:i + batch].to(device),
                           mask[i:i + batch].to(device)).double()
            X = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64, device=device)], 1)
            XtX += X.T @ X
            Xty += X.T @ y[i:i + batch].to(device).double()
        XtX += ridge * torch.eye(D + 1, dtype=torch.float64, device=device)
        sol = torch.linalg.solve(XtX, Xty)
        self.w.copy_(sol[:D].to(self.w.dtype))
        self.b.copy_(sol[D].to(self.b.dtype))
        self.fitted = True
        return self


ARCHITECTURES = {
    "egnn": None,                 # models.egnn.EGNNScalar, resolved by the trainer
    "transformer": InvariantTransformer,
    "ridge": RidgeDescriptor,
}
