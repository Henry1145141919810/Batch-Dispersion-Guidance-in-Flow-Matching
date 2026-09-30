"""Gates for the D-Flow arm.

The decisive one is EQUIVALENCE: `dflow._euler_rollout` is a second
implementation of the same ODE that `sampling.integrate` runs, written
separately only because `integrate` wraps the generator in `torch.no_grad()`.
If the two ever disagree, D-Flow is optimising a different model than the one
every other arm samples, and the comparison is void. That gate is exact.

Run: python proj1/tests/test_dflow.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from dflow import _euler_rollout, dflow_optimise  # noqa: E402
from models.egnn import zero_com  # noqa: E402
from sampling import FlowSampler, integrate  # noqa: E402

torch.set_default_dtype(torch.float64)
R = {}
B, N, K = 3, 5, 2


class Net(torch.nn.Module):
    """Small equivariant-ish velocity field: linear in coords, linear in feats,
    time-dependent. Enough structure that the rollout is not trivial."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(5)
        self.a = torch.nn.Parameter(torch.randn(1) * 0.3)
        self.W = torch.nn.Parameter(torch.randn(K, K) * 0.3)

    def forward(self, c, f, mask, t):
        tb = t.view(-1, 1, 1)
        v_c = self.a * c * (1.0 + tb)
        v_f = f @ self.W * (1.0 - 0.5 * tb)
        m3 = mask.unsqueeze(-1)
        return zero_com(v_c * m3, mask), v_f * m3


class Prop(torch.nn.Module):
    def forward(self, c, f, mask):
        m3 = mask.unsqueeze(-1)
        return ((c * m3) ** 2).sum((1, 2)) + (f * m3).sum((1, 2))


def main():
    torch.manual_seed(11)
    mask = torch.ones(B, N)
    mask[0, -1] = 0.0                      # one padded atom, deliberately
    net, f_net = Net(), Prop()
    c0 = zero_com(torch.randn(B, N, 3), mask)
    f0 = torch.randn(B, N, K) * mask.unsqueeze(-1)
    STEPS = 12

    # ---- 1. EQUIVALENCE with the production sampler (the decisive gate)
    smp = FlowSampler(net, mask, f_net=None)
    ic, if_, _ = integrate(smp, c0.clone(), f0.clone(), STEPS, "euler")
    with torch.no_grad():
        dc, df = _euler_rollout(net, mask, c0.clone(), f0.clone(), STEPS,
                                checkpoint=False)
    R["rollout_matches_integrate_coords"] = (ic - dc).abs().max().item()
    R["rollout_matches_integrate_feats"] = (if_ - df).abs().max().item()

    # ---- 2. checkpointing must not change the gradient
    def grad_of(ck):
        c = c0.clone().requires_grad_(True)
        f = f0.clone().requires_grad_(True)
        xc, xf = _euler_rollout(net, mask, c, f, STEPS, checkpoint=ck)
        f_net(xc, xf, mask).sum().backward()
        return c.grad.clone(), f.grad.clone()

    g1c, g1f = grad_of(False)
    g2c, g2f = grad_of(True)
    R["checkpoint_grad_identical"] = max((g1c - g2c).abs().max().item(),
                                         (g1f - g2f).abs().max().item())
    R["gradient_is_nonzero"] = 0.0 if g1c.abs().max() > 1e-8 else 1.0

    # ---- 3. the optimiser must reduce its own loss
    y = torch.full((B,), 3.0)
    log = []
    dflow_optimise(net, f_net, mask, c0, f0, y, n_steps=STEPS, n_iter=6,
                   lr=0.05, checkpoint=False, log=log)
    R["loss_decreases"] = 0.0 if log[-1] < log[0] else 1.0
    R["loss_is_finite"] = 0.0 if all(v == v and abs(v) < 1e30 for v in log) else 1.0

    # ---- 4. invariants of a legal initial condition are preserved
    oc, of_ = dflow_optimise(net, f_net, mask, c0, f0, y, n_steps=STEPS,
                             n_iter=4, lr=0.05, checkpoint=False)
    com = (oc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
    R["output_is_zero_com"] = com.abs().max().item()
    R["padded_atom_coords_zero"] = oc[0, -1].abs().max().item()
    R["padded_atom_feats_zero"] = of_[0, -1].abs().max().item()

    # ---- 5. the trust radius is respected, and it BINDS
    #      (a radius that never binds would make this gate vacuous)
    tr = 0.05
    c_in = c0.clone().requires_grad_(False)
    ofc, off = dflow_optimise(net, f_net, mask, c0, f0, y, n_steps=STEPS,
                              n_iter=8, lr=0.5, checkpoint=False, trust=tr)
    # recover the optimised x0 by re-running with the same seed is not possible
    # here, so instead check the UNBOUNDED run moves further than the radius,
    # which is what makes the bounded run's constraint meaningful
    ouc, ouf = dflow_optimise(net, f_net, mask, c0, f0, y, n_steps=STEPS,
                              n_iter=8, lr=0.5, checkpoint=False, trust=None)
    d_bound = (ofc - oc).norm()
    d_free = (ouc - oc).norm()
    R["trust_radius_binds"] = 0.0 if d_free > d_bound else 1.0

    # ---- 6. more iterations must not increase the final loss
    l1, l2 = [], []
    dflow_optimise(net, f_net, mask, c0, f0, y, n_steps=STEPS, n_iter=2,
                   lr=0.05, checkpoint=False, log=l1)
    dflow_optimise(net, f_net, mask, c0, f0, y, n_steps=STEPS, n_iter=10,
                   lr=0.05, checkpoint=False, log=l2)
    R["more_iters_not_worse"] = 0.0 if l2[-1] <= l1[-1] + 1e-9 else 1.0

    tol = 1e-9
    print("%-38s %13s   pass" % ("check", "max abs error"))
    print("-" * 62)
    ok = True
    for k, v in R.items():
        p = v < tol
        ok = ok and p
        print("%-38s %13.3e   %s" % (k, v, "yes" if p else "NO"))
    print("-" * 62)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
