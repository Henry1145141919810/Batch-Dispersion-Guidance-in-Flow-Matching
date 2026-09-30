"""D-Flow: property control by optimising the SOURCE NOISE of a frozen ODE.

Ben-Hamu et al., "D-Flow: Differentiating through Flows for Controlled
Generation", ICML 2024. Training-free: the generator is frozen and only the
initial condition moves.

WHY THIS ARM EXISTS, and why it is not just another `mode`.

Every other arm in this project belongs to ONE family. `plug`, `tmpd`,
`lgd_mc`, `smg`, `btvg` all form a one-step estimate of the clean sample,
m = E[x_1 | x_t], evaluate the property there, and push a gradient back through
that single step. They differ only in how they treat the uncertainty of that
estimate.

D-Flow belongs to the OTHER family. It never forms a posterior mean at all. It
treats the whole sampler as a differentiable map x_1 = ODE(x_0) and descends

    L(x_0) = mean_i ( f(ODE(x_0)_i) - y_i )^2

in x_0 by gradient descent. The property is evaluated on an ACTUAL ROLLOUT, not
on a one-step proxy, so the off-distribution error that
FINDING_QUADRATIC_CLOSURE_VALIDITY.md measures for the closure does not apply
to it. That is precisely why it is worth having: it is the comparison that our
whole one-step construction could be wrong against.

Substituting the one-step estimate for the rollout would turn it back into
`plug` and destroy the point of including it.

COST, AND WHY "MATCHED COMPUTE" NEEDS A FOOTNOTE. One D-Flow cell is
`n_iter` optimisation steps, each a full forward ODE solve plus a backward pass
through it. At n_iter=8 and 100 steps that is ~16x the network evaluations of a
single guided sample, before counting the backward. It CANNOT be compared to
the guided arms at equal wall clock; report the measured `cost` alongside.

MEMORY. A naive autograd tape over 100 EGNN steps stores every activation.
`checkpoint=True` (the default) recomputes each step's forward during the
backward instead, making memory O(1) in the number of steps at the price of one
extra forward. This is what makes the arm runnable at batch 128 on a 45 GB
slice at all.
"""
from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint as _ckpt

from guidance import Cost
from models.egnn import zero_com


def _euler_rollout(net, mask, c, f, n_steps, cost=None, checkpoint=True):
    """x_1 = ODE(x_0) by explicit Euler, DIFFERENTIABLE in (c, f).

    Deliberately a separate function from `sampling.integrate`: that one runs
    the generator inside `torch.no_grad()`, which is right for every guided arm
    and fatal here, since the whole method is a gradient through this rollout.
    Same grid (`linspace(0, 1, n+1)`) and same step rule, so the two agree
    exactly when no guidance is applied -- `test_dflow.py` gates that.
    """
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=c.device, dtype=c.dtype)
    m3 = mask.unsqueeze(-1)

    def step(cc, ff, t0v, h):
        tb = t0v.expand(cc.shape[0])
        v_c, v_f = net(cc, ff, mask, tb)
        return cc + h * v_c, ff + h * v_f

    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        h = (t1 - t0)
        if checkpoint and torch.is_grad_enabled():
            c, f = _ckpt(step, c, f, t0, h, use_reentrant=False)
        else:
            c, f = step(c, f, t0, h)
        # the generator is equivariant and zero-CoM; keeping the state on that
        # subspace every step stops a slow drift that the optimiser would
        # otherwise happily exploit to lower the loss off-manifold
        c = zero_com(c, mask)
        f = f * m3
        if cost is not None:
            cost.gen_fwd += 1
    return c, f


def dflow_optimise(net, f_net, mask, c0, f0, y, n_steps=100, n_iter=8,
                   lr=0.05, checkpoint=True, cost=None, trust=None,
                   log=None):
    """Optimise the source noise so the ROLLED-OUT sample hits the target.

    Returns (coords, feats) at t=1 for the optimised x_0.

    `lr` is this arm's strength knob: it is what the sweep's `w` scales, the
    way `w` scales the field for every guided arm. `n_iter` is held fixed so a
    strength sweep does not silently become a compute sweep.

    `trust` (optional) bounds ||x_0 - x_0_init|| per sample. D-Flow's own paper
    relies on the implicit regularisation of a small number of steps from a
    Gaussian start; an explicit ball is available because our property head is
    an EGNN evaluated far off its training distribution when the optimiser is
    free to wander, and an unbounded optimiser will find that rather than a
    molecule.
    """
    cost = cost if cost is not None else Cost()
    c_init, f_init = c0.detach().clone(), f0.detach().clone()
    c = c0.detach().clone().requires_grad_(True)
    f = f0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([c, f], lr=lr)
    m3 = mask.unsqueeze(-1)

    for it in range(n_iter):
        opt.zero_grad(set_to_none=True)
        x_c, x_f = _euler_rollout(net, mask, c, f, n_steps, cost, checkpoint)
        pred = f_net(x_c, x_f, mask)
        cost.guide_fwd += 1
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        cost.gen_vjp += 1
        cost.guide_bwd += 1
        opt.step()
        with torch.no_grad():
            # keep x_0 a legal initial condition: zero-CoM coordinates, masked
            # features. Adam does not know about either.
            c.data = zero_com(c.data, mask)
            f.data = f.data * m3
            if trust is not None:
                dc, df = c.data - c_init, f.data - f_init
                n = torch.sqrt((dc ** 2).sum((1, 2)) + (df ** 2).sum((1, 2)) + 1e-12)
                s = torch.clamp(trust / n, max=1.0).view(-1, 1, 1)
                c.data = c_init + dc * s
                f.data = f_init + df * s
        if log is not None:
            log.append(float(loss.detach()))

    with torch.no_grad():
        out_c, out_f = _euler_rollout(net, mask, c.detach(), f.detach(),
                                      n_steps, cost, checkpoint=False)
    return out_c.detach(), out_f.detach()
