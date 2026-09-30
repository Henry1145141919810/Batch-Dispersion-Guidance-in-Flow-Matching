"""Inference-time update RULES for the guidance direction.

The sampler already computes a guidance direction G (score units) at every
step. Until now that direction was applied first-order: the step added
`mult * G` to the velocity and the ODE solver integrated it. That is Euler on
the guidance signal.

This module replaces THAT rule, and only that rule. Nothing here touches the
generator, the guide, the sampler's time grid, the clip, or the objective.
Read the trajectory t_0 ... t_N as an optimisation run and the per-step G as
its gradient, and the standard optimiser preconditioners apply:

  euler      D = G                                  (the existing behaviour)
  momentum   D = m_hat,  m = b1 m + (1-b1) G        (heavy ball / Adam's first
                                                     moment, bias-corrected)
  adam       D = m_hat / (sqrt(v_hat) + eps)        elementwise second moment
  adam_eq    same, but the coordinate block's second moment is PER ATOM
             (a scalar on each atom's 3-vector) instead of per element
  muon       D = NewtonSchulz(m_hat), the orthogonalised (U V^T) direction

THE EQUIVARIANCE PROBLEM, which decides which of these is even admissible.

The coordinate block G_c is [B, N, 3] and the model is E(3)-equivariant: a
rotation R acts on the RIGHT, G_c -> G_c R^T. An update rule is admissible only
if it commutes with that action, otherwise the sampler stops being equivariant
and a rotated copy of the same noise generates a different molecule.

  momentum   ADMISSIBLE. A sum of rotated vectors is the rotated sum.
  adam       NOT ADMISSIBLE on the coordinate block. v holds per-axis squared
             magnitudes, so dividing by sqrt(v) scales x, y and z differently:
             an axis-aligned rescaling, which does not commute with R. It is
             implemented here anyway, and reported, because "the obvious thing
             breaks equivariance" is worth measuring rather than asserting --
             but it is NOT a candidate method.
  adam_eq    ADMISSIBLE. One scalar per atom, v_i = EMA of |g_i|^2 / 3, makes
             the preconditioner a multiple of the identity on each atom's
             3-vector, and scalars commute with R. This is the honest Adam
             analogue for this modality.
  muon       ADMISSIBLE, and exactly so. Newton-Schulz iterates
             p(X) = aX + b(XX^T)X + c(XX^T)^2 X. Under X -> X R^T the Gram
             matrix XX^T is invariant, so p(X R^T) = p(X) R^T at every order.
             Orthogonalisation commutes with rotation because rotation acts on
             the side the iteration does not touch. The same argument with a
             permutation P on the left gives p(PX) = P p(X), so atom ordering
             is safe too, and a zero row (a padded atom) stays zero because
             every term carries that row as a factor.

  The atom-type block [B, N, K] has no rotational symmetry -- the K types are a
  labelled basis -- so elementwise Adam is fine there, and Muon's N x K matrix
  (atoms x types) is a genuine matrix rather than a reshape of something that
  was not one. Nothing is flattened into a matrix it was not already.

MAGNITUDE. Preconditioning changes the direction AND the scale: Adam's output
is ~unit per element and Muon's has unit singular values, neither comparable to
the raw G that the strength w and the velocity-relative clip were tuned
against. By default the preconditioned direction is rescaled to the raw
direction's per-sample norm (`rescale="norm"`), so these rules change WHERE the
guidance points and not HOW FAR it pushes, and w keeps its meaning across arms.
`rescale="none"` measures the alternative; it is not the default because it
silently changes the strength the comparison claims to hold fixed.

DIAGNOSTICS. Every state-updating call records, per step: the raw direction's
norm, the cosine between consecutive raw directions (does the guidance signal
stay put as the trajectory moves?), the cosine between the momentum buffer and
the current raw direction (IS the buffer stale?), and the cosine between the
applied direction and the raw one (how far the rule turned the step).
Staleness is the stated hypothesis for why momentum might hurt here, so it is
measured rather than argued.
"""
from __future__ import annotations

import torch

RULES = ("euler", "momentum", "adam", "adam_eq", "muon")

# Equivariance status per rule, carried into the results so a table can never
# quietly present the inadmissible variant as a candidate method.
EQUIVARIANT = {"euler": True, "momentum": True, "adam": False,
               "adam_eq": True, "muon": True}

# Muon's quintic Newton-Schulz coefficients (Jordan et al. 2024).
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def _per_sample_norm(*blocks):
    """||.|| over every non-batch axis, one number per sample. [B]"""
    tot = None
    for x in blocks:
        s = (x.reshape(x.shape[0], -1) ** 2).sum(1)
        tot = s if tot is None else tot + s
    return torch.sqrt(tot + 1e-30)


def _cosine(a_blocks, b_blocks):
    """Per-sample cosine between two (coords, feats) pairs. [B]"""
    dot = None
    for a, b in zip(a_blocks, b_blocks):
        d = (a.reshape(a.shape[0], -1) * b.reshape(b.shape[0], -1)).sum(1)
        dot = d if dot is None else dot + d
    return dot / (_per_sample_norm(*a_blocks) * _per_sample_norm(*b_blocks))


def newton_schulz(X, steps=8, eps=1e-7):
    """Approximate the orthogonal factor U V^T of X, batched over samples.

    X is [B, R, C]. The iteration runs on whichever orientation has the smaller
    Gram matrix -- p(X)^T = p(X^T) exactly, because (X X^T) X = X (X^T X), so
    the transpose is a free optimisation and not an approximation.

    Rows that are entirely zero (padded atoms) come out zero: every term of the
    polynomial carries that row as a factor.

    THE OUTPUT IS NOT EXACTLY ORTHOGONAL. Muon's quintic coefficients are tuned
    for speed, not convergence: the singular values settle into [0.68, 1.13]
    rather than at 1, and iterating further does not close that gap because the
    fixed point of this polynomial is a band, not a point. That is Muon as
    published and is kept deliberately -- the direction is what is under test,
    and `rescale="norm"` sets the magnitude afterwards anyway. The gate in
    test_update_rules.py asserts the band, not exactness.

    STEPS DEFAULTS TO 8, NOT MUON'S 5. Measured here: a well-conditioned block
    reaches the band in 5 steps, but at condition number ~900 -- which a 9x3
    coordinate gradient near rank 2 reaches easily -- 5 steps leaves the
    smallest singular value at 0.24, i.e. the direction is still stretched along
    its dominant axis and has not been orthogonalised at all. 6 steps is enough
    and 8 is settled; the extra iterations are 3x3 matmuls and cost nothing.

    Precision follows the input: the iteration is a polynomial in X, so casting
    a float64 input down to float32 would make the equivariance that makes this
    rule admissible hold only to 1e-6 instead of machine precision.
    """
    dtype = X.dtype if X.dtype in (torch.float32, torch.float64) else torch.float32
    A = X.to(dtype)
    transposed = A.shape[-2] > A.shape[-1]
    if transposed:
        A = A.transpose(-2, -1)
    # Scale into the iteration's basin of convergence. A zero block stays zero.
    nrm = A.reshape(A.shape[0], -1).norm(dim=1).clamp(min=eps).view(-1, 1, 1)
    A = A / nrm
    for _ in range(steps):
        G = A @ A.transpose(-2, -1)
        A = _NS_A * A + _NS_B * (G @ A) + _NS_C * (G @ G @ A)
    if transposed:
        A = A.transpose(-2, -1)
    return A.to(X.dtype)


class GuidanceUpdater:
    """Carries the moment buffers along one sampling trajectory.

    One instance per sampler, and the sampler is rebuilt per batch, so the
    buffers never leak across batches. `reset()` is called by `integrate`.
    """

    def __init__(self, rule="euler", beta1=0.9, beta2=0.999, eps=1e-8,
                 rescale="norm", ns_steps=8, log=True):
        if rule not in RULES:
            raise ValueError("unknown update rule %r; known: %s"
                             % (rule, ", ".join(RULES)))
        if rescale not in ("norm", "none"):
            raise ValueError("rescale must be 'norm' or 'none'")
        self.rule, self.beta1, self.beta2, self.eps = rule, beta1, beta2, eps
        self.rescale, self.ns_steps, self.log = rescale, ns_steps, log
        self.reset()

    def reset(self):
        self.m_c = self.m_f = self.v_c = self.v_f = None
        self.prev_c = self.prev_f = None
        self.k = 0                      # state updates so far (bias correction)
        self.rows = []                  # per-step diagnostics

    # -- diagnostics -------------------------------------------------------
    def _record(self, t_scalar, G_c, G_f, D_c, D_f, pre_scale):
        if not self.log:
            return
        row = {"t": (float(t_scalar) if t_scalar is not None else None),
               "step": self.k,
               "g_norm": float(_per_sample_norm(G_c, G_f).mean()),
               "pre_scale": float(pre_scale)}
        if self.prev_c is not None:
            row["cos_prev"] = float(_cosine((G_c, G_f),
                                            (self.prev_c, self.prev_f)).mean())
        if self.m_c is not None and self.rule != "euler":
            # the staleness number: how well the accumulated buffer still
            # agrees with the direction the guide is asking for NOW
            row["cos_mom_g"] = float(_cosine((self.m_c, self.m_f),
                                             (G_c, G_f)).mean())
        row["cos_applied_g"] = float(_cosine((D_c, D_f), (G_c, G_f)).mean())
        self.rows.append(row)

    def summary(self):
        """Means over the trajectory, plus an early/late split.

        The staleness hypothesis is specifically that old directions decay as
        the trajectory moves, so a single mean over t would hide it.
        """
        if not self.rows:
            return {}
        out = {}
        for k in ("g_norm", "cos_prev", "cos_mom_g", "cos_applied_g",
                  "pre_scale"):
            vals = [r[k] for r in self.rows if k in r]
            if vals:
                out[k] = sum(vals) / len(vals)
        half = len(self.rows) // 2
        for k in ("cos_prev", "cos_mom_g"):
            early = [r[k] for r in self.rows[:half] if k in r]
            late = [r[k] for r in self.rows[half:] if k in r]
            if early:
                out[k + "_early"] = sum(early) / len(early)
            if late:
                out[k + "_late"] = sum(late) / len(late)
        out["n_steps_logged"] = len(self.rows)
        return out

    # -- the rule ----------------------------------------------------------
    def apply(self, G_c, G_f, mask, t_scalar=None, update_state=True):
        """Map the raw guidance direction to the direction actually applied.

        update_state=False is the Heun second stage: it reads the buffers built
        by the first stage but does not write to them, so one solver step
        advances the optimiser state exactly once. Writing on both stages would
        make Heun take two optimiser steps per ODE step and stop being a
        controlled comparison against Euler.
        """
        if self.rule == "euler":
            # Untouched, bit for bit. The existing sweep must not move.
            if update_state and self.log:
                self._record(t_scalar, G_c, G_f, G_c, G_f, 1.0)
                self.prev_c, self.prev_f = G_c.detach(), G_f.detach()
                self.k += 1
            return G_c, G_f

        b1, b2 = self.beta1, self.beta2
        if self.m_c is None:
            m_c = torch.zeros_like(G_c)
            m_f = torch.zeros_like(G_f)
            v_c = (torch.zeros(G_c.shape[0], G_c.shape[1], 1, device=G_c.device,
                               dtype=G_c.dtype) if self.rule == "adam_eq"
                   else torch.zeros_like(G_c))
            v_f = torch.zeros_like(G_f)
        else:
            m_c, m_f, v_c, v_f = self.m_c, self.m_f, self.v_c, self.v_f

        k = self.k + 1
        m_c = b1 * m_c + (1.0 - b1) * G_c
        m_f = b1 * m_f + (1.0 - b1) * G_f
        bc1 = 1.0 - b1 ** k
        mh_c, mh_f = m_c / bc1, m_f / bc1

        if self.rule == "momentum":
            D_c, D_f = mh_c, mh_f
        elif self.rule in ("adam", "adam_eq"):
            if self.rule == "adam_eq":
                # one scalar per atom: a multiple of the identity on each
                # 3-vector, which is what keeps this rotation-equivariant
                sq_c = (G_c ** 2).mean(dim=-1, keepdim=True)
            else:
                sq_c = G_c ** 2
            v_c = b2 * v_c + (1.0 - b2) * sq_c
            v_f = b2 * v_f + (1.0 - b2) * G_f ** 2
            bc2 = 1.0 - b2 ** k
            D_c = mh_c / (torch.sqrt(v_c / bc2) + self.eps)
            D_f = mh_f / (torch.sqrt(v_f / bc2) + self.eps)
        elif self.rule == "muon":
            D_c = newton_schulz(mh_c, self.ns_steps)
            D_f = newton_schulz(mh_f, self.ns_steps)
        else:                                    # unreachable; RULES is closed
            raise AssertionError(self.rule)

        # Keep padded atoms at exactly zero. Adam divides 0 by eps, which is 0,
        # and Newton-Schulz preserves zero rows -- but masking is cheap and
        # makes that a guarantee rather than a property of the arithmetic.
        m3 = mask.unsqueeze(-1)
        D_c, D_f = D_c * m3, D_f * m3

        g_n = _per_sample_norm(G_c, G_f)
        d_n = _per_sample_norm(D_c, D_f)
        pre_scale = float((d_n / g_n.clamp(min=1e-30)).mean())
        if self.rescale == "norm":
            f = (g_n / d_n.clamp(min=1e-30)).view(-1, 1, 1)
            D_c, D_f = D_c * f, D_f * f

        if update_state:
            self._record(t_scalar, G_c, G_f, D_c, D_f, pre_scale)
            self.m_c, self.m_f, self.v_c, self.v_f = m_c, m_f, v_c, v_f
            self.prev_c, self.prev_f = G_c.detach(), G_f.detach()
            self.k = k
        return D_c, D_f
