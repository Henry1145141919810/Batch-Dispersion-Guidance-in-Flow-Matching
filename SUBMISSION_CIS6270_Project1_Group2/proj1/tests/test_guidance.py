"""Correctness checks for guidance.py and sampling.py. CPU, float64, tiny nets.

  1. Call accounting: Euler = N field evaluations, Heun = 2N.
  2. Affine property => c = 0 exactly, and SMG's direction equals plug-in's
     (the H = 0 limit that collapses SMG onto TMPD's formula).
  3. Quadratic property, tiny system: Hutchinson 1/2 tr(H Sigma) and v_f agree
     with the EXACT values from full Jacobian / Hessian.
  4. Same-field convergence: on the actual guided field, Euler error scales ~h
     and Heun error ~h^2 against a refined reference. This isolates solver
     error; it says nothing about whether the field targets the right law.

Run: .venv/Scripts/python.exe proj1/tests/test_guidance.py
"""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models.egnn import EGNNScalar, EGNNVelocity, zero_com  # noqa: E402
from guidance import (fm_posterior, guidance_field,  # noqa: E402
                      hutchinson_tr_H_Sigma, sigma_times_vector)
from sampling import FlowSampler, VPSampler, integrate, initial_noise  # noqa: E402

torch.manual_seed(20260918)
DT = torch.float64
torch.set_default_dtype(DT)
DEV = "cpu"


class AffineProperty(nn.Module):
    """f(x) = a . flat(x) + b. H = 0 identically."""
    def __init__(self, N, K):
        super().__init__()
        self.a_c = nn.Parameter(torch.randn(N, 3))
        self.a_f = nn.Parameter(torch.randn(N, K))
        self.b = nn.Parameter(torch.tensor(0.3))

    def forward(self, c, f, mask):
        m = mask.unsqueeze(-1)
        return ((c * self.a_c * m).sum((1, 2)) + (f * self.a_f * m).sum((1, 2))
                + self.b)


class QuadraticProperty(nn.Module):
    """f(x) = 1/2 flat(x)^T Q flat(x) with Q symmetric: H = Q exactly."""
    def __init__(self, N, K):
        super().__init__()
        D = N * 3 + N * K
        A = torch.randn(D, D)
        self.register_buffer("Q", 0.5 * (A + A.T) / math.sqrt(D))

    def forward(self, c, f, mask):
        m = mask.unsqueeze(-1)
        x = torch.cat([(c * m).reshape(c.shape[0], -1),
                       (f * m).reshape(f.shape[0], -1)], dim=1)
        return 0.5 * (x @ self.Q * x).sum(1)


def make_batch(B=2, N=5, K=3):
    mask = torch.ones(B, N)
    mask[1, 4] = 0.0                       # one padded atom in sample 1
    c, f = initial_noise(mask, K)
    return c, f, mask


def main():
    results = {}
    B, N, K = 2, 5, 3
    net = EGNNVelocity(K, 16, 2).to(DEV).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    c0, f0, mask = make_batch(B, N, K)

    # ---------- 1. call accounting
    fs = FlowSampler(net, mask)
    _, _, n_e = integrate(fs, c0, f0, 8, "euler")
    _, _, n_h = integrate(fs, c0, f0, 8, "heun")
    results["euler_calls_eq_N"] = abs(n_e - 8)
    results["heun_calls_eq_2N"] = abs(n_h - 16)
    vs = VPSampler(net, mask)
    _, _, n_ve = integrate(vs, c0, f0, 8, "euler")
    results["vp_euler_calls_eq_N"] = abs(n_ve - 8)

    # ---------- 2. affine property: c == 0, SMG direction == plug-in direction
    aff = AffineProperty(N, K).to(DEV)
    t = torch.full((B,), 0.5)
    post_fn = lambda c, f: fm_posterior(net, c, f, mask, t)  # noqa: E731
    y = torch.tensor([1.0, -0.5])
    Gp_c, Gp_f, dp = guidance_field(aff, post_fn, c0, f0, mask, y, 0.7, "plug")
    Gs_c, Gs_f, ds = guidance_field(aff, post_fn, c0, f0, mask, y, 0.7, "smg",
                                    n_probe=4, generator=torch.Generator().manual_seed(1))
    results["affine_c_is_zero"] = ds["c"].abs().max().item()
    # directions parallel: SMG = plug * s^2 / (s^2 + v_f) per sample
    ratio = (0.7 ** 2 / (0.7 ** 2 + ds["v_f"])).view(-1, 1, 1)
    results["affine_smg_eq_scaled_plug"] = torch.cat([
        (Gs_c - ratio * Gp_c).flatten(), (Gs_f - ratio * Gp_f).flatten()]).abs().max().item()

    # ---------- 3. quadratic property: Hutchinson vs exact
    quad = QuadraticProperty(N, K).to(DEV)
    post = post_fn(c0, f0)
    # k became a per-sample tensor when the scalar-k bug was fixed; this block
    # works on the B=1 slice, so take that sample's value.
    k = post.k[:1]
    # exact J on the flattened valid coordinates of sample 0 only (B=1 slice)
    c1, f1, m1 = c0[:1], f0[:1], mask[:1]
    pf1 = lambda c, f: fm_posterior(net, c, f, m1, t[:1])  # noqa: E731

    def flat_m(xflat):
        c = xflat[: N * 3].reshape(1, N, 3)
        f = xflat[N * 3:].reshape(1, N, K)
        p = pf1(c, f)
        return torch.cat([p.mean_coords.reshape(-1), p.mean_feats.reshape(-1)])

    x_flat = torch.cat([c1.reshape(-1), f1.reshape(-1)])
    J = torch.autograd.functional.jacobian(flat_m, x_flat)          # [D, D]
    Sigma = k.item() * J
    H = quad.Q                                                        # exact
    exact_c = 0.5 * torch.trace(H @ Sigma)
    m_flat = flat_m(x_flat)
    g = H @ m_flat
    exact_vf = g @ Sigma @ g

    p1 = pf1(c1, f1)
    # tr(H Sigma) for random Q is a sum of signed terms that can nearly cancel,
    # so relative error is the wrong yardstick. Test UNBIASEDNESS instead:
    # single-probe estimates, then |mean - exact| in units of the standard error.
    samples = []
    for i in range(1500):
        h1 = hutchinson_tr_H_Sigma(quad, pf1, c1, f1, m1, p1.mean_coords, p1.mean_feats,
                                   k, n_probe=1, generator=torch.Generator().manual_seed(100 + i))
        samples.append(h1[0].item())
    st = torch.tensor(samples)
    se = st.std() / math.sqrt(len(samples))
    results["hutchinson_zscore"] = ((st.mean() - exact_c).abs() / se).item()
    print("  hutchinson: exact %.4f  mean %.4f  se %.4f  (n=%d)"
          % (exact_c, st.mean(), se, len(samples)))

    # v_f uses ONE exact JVP, so it must match to round-off
    g_c = g[: N * 3].reshape(1, N, 3)
    g_f = g[N * 3:].reshape(1, N, K)
    sg_c, sg_f = sigma_times_vector(pf1, c1, f1, m1, g_c, g_f, k)
    vf = (g_c * sg_c).sum() + (g_f * sg_f).sum()
    results["v_f_exact"] = ((vf - exact_vf) / exact_vf.abs().clamp(min=1e-9)).abs().item()

    # ---------- 4. same-field convergence order (guided flow field)
    # Guidance kept mild on purpose: the (1 - t)/t multiplier reaches 19 at
    # t_min_guide, and a random affine property on 30 dims makes strong
    # guidance stiff enough that a 16-step Euler run overflows. This test is
    # about solver ORDER on a smooth field, not about guidance strength.
    guided = FlowSampler(net, mask, f_net=aff, y=y, s=3.0, mode="plug", w=0.02,
                         t_min_guide=0.05, clip=None)  # clip is a nonlinearity;
                                                      # order check must be unclipped
    # Interior span [0.2, 0.8], guidance on throughout, no endpoint policy:
    # the field is then smooth in t and the convergence order is well defined.
    # Over the full [0, 1] the switch-on at t_min_guide is a discontinuity and
    # error does not decrease monotonically.
    SPAN = (0.2, 0.8)
    ref_c, ref_f, _ = integrate(guided, c0, f0, 2048, "heun", span=SPAN, terminal=False)
    errs = {}
    for solver in ("euler", "heun"):
        e = []
        for n in (16, 32, 64):
            cc, ff, _ = integrate(guided, c0, f0, n, solver, span=SPAN, terminal=False)
            e.append(torch.cat([(cc - ref_c).flatten(), (ff - ref_f).flatten()]).abs().max().item())
        orders = [math.log(e[i] / e[i + 1], 2) for i in range(len(e) - 1)]
        errs[solver] = (e, orders)
        results["order_" + solver] = sum(orders) / len(orders)

    # ---------- report
    tol = {"euler_calls_eq_N": 0, "heun_calls_eq_2N": 0, "vp_euler_calls_eq_N": 0,
           "affine_c_is_zero": 1e-12, "affine_smg_eq_scaled_plug": 1e-10,
           "hutchinson_zscore": 4.0, "v_f_exact": 1e-9}
    print("check                              value          pass")
    print("-" * 60)
    ok = True
    for kname, v in results.items():
        if kname.startswith("order_"):
            want = 1.0 if kname.endswith("euler") else 2.0
            p = abs(v - want) < 0.3
            print("%-34s %10.3f       %s   (expect ~%.0f)" % (kname, v, "yes" if p else "NO", want))
        else:
            p = v <= tol[kname]
            print("%-34s %10.3e       %s" % (kname, v, "yes" if p else "NO"))
        ok = ok and p
    print("-" * 60)
    for s, (e, o) in errs.items():
        print("%-6s errors %s  orders %s" % (s, ["%.2e" % x for x in e], ["%.2f" % x for x in o]))
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
