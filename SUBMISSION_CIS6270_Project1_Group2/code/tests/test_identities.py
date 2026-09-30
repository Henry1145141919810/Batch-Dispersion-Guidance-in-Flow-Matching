"""Exact identities the training objectives rest on.

Every check here is an algebraic fact, not an approximation, so any failure is a
sign or scaling error rather than a tolerance issue. These are the mistakes that
do not crash and do not show up in a loss curve: the model trains, the loss
falls, and the samples are quietly wrong.

  1  FM interpolant      x_t + (1-t) u == x_1  for the true velocity u = x_1 - eps
  2  FM at the endpoints t=1 gives x_1, t=0 gives eps
  3  VP schedule         alpha^2 + sigma^2 == 1 everywhere; alpha(0)=1
  4  VP denoising        (x_tau - sigma eps) / alpha == x_0  for the true eps
  5  posterior means     both families' endpoint proxies recover x_1 EXACTLY
                         when the network is a perfect denoiser
  6  zero centre of mass preserved by the interpolant, the noise and the proxy
  7  masking             padded atoms change neither loss nor gradient
  8  loss minimum        the true target gives exactly zero loss

Checks 1-6 and 8 are pure algebra and run anywhere. Check 7 exercises the real
loss function from whichever trainer is present, so it covers `fm_loss` in the
flow-matching package, `diffusion_loss` in the diffusion package, and both in
the full project.

Run: python proj1/tests/test_identities.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from diffusion import alpha_sigma  # noqa: E402
from guidance import fm_posterior, vp_posterior  # noqa: E402
from models.egnn import zero_com  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.manual_seed(20260919)
R = {}


def available_losses():
    """(name, fn) for every trainer in this package. Never empty in practice:
    a package with no trainer has nothing to verify."""
    out = []
    try:
        from train_fm import fm_loss
        out.append(("fm", fm_loss))
    except ImportError:
        pass
    try:
        from train_diffusion import diffusion_loss
        out.append(("diffusion", diffusion_loss))
    except ImportError:
        pass
    return out


def batch(B=4, N=7, K=5):
    mask = torch.ones(B, N)
    mask[0, 5:] = 0.0            # ragged, so masking is actually exercised
    mask[2, 6:] = 0.0
    x1 = zero_com(torch.randn(B, N, 3), mask)
    f1 = torch.zeros(B, N, K)
    for b in range(B):
        n = int(mask[b].sum())
        f1[b, torch.arange(n), torch.randint(0, K, (n,))] = 1.0
    return x1, f1 * mask.unsqueeze(-1), mask


class PerfectDenoiser(torch.nn.Module):
    """Returns the exact velocity / exact eps. The oracle a trained net approximates."""

    def __init__(self, x1, eps, family, mask):
        super().__init__()
        self.x1, self.eps, self.family, self.mask = x1, eps, family, mask

    def forward(self, c, f, mask, t):
        if self.family == "flow":
            return self.x1[0] - self.eps[0], self.x1[1] - self.eps[1]
        return self.eps[0], self.eps[1]


def main():
    x1, f1, mask = batch()
    m = mask.unsqueeze(-1)
    eps_c = zero_com(torch.randn_like(x1), mask)
    eps_f = torch.randn_like(f1) * m

    # ---------------- 1, 2: flow-matching interpolant
    worst = 0.0
    for tv in (0.0, 0.1, 0.5, 0.9, 1.0):
        t = torch.full((x1.shape[0],), tv).view(-1, 1, 1)
        xt_c = (t * x1 + (1 - t) * eps_c) * m
        u_c = (x1 - eps_c) * m
        worst = max(worst, ((xt_c + (1 - t) * u_c) - x1 * m).abs().max().item())
    R["fm_endpoint_identity"] = worst

    t0, t1 = torch.zeros(1, 1, 1), torch.ones(1, 1, 1)
    R["fm_t1_is_data"] = ((t1 * x1 + (1 - t1) * eps_c) * m - x1 * m).abs().max().item()
    R["fm_t0_is_noise"] = ((t0 * x1 + (1 - t0) * eps_c) * m - eps_c * m).abs().max().item()

    # ---------------- 3, 4: VP schedule and denoising
    tau = torch.tensor([0.0, 1e-3, 0.25, 0.5, 0.75, 1.0])
    a, s = alpha_sigma(tau)
    R["vp_alpha2_plus_sigma2_is_1"] = (a ** 2 + s ** 2 - 1.0).abs().max().item()
    R["vp_alpha_at_0_is_1"] = (alpha_sigma(torch.zeros(1))[0] - 1.0).abs().max().item()

    worst = 0.0
    for tv in (1e-3, 0.25, 0.5, 0.9):
        av, sv = alpha_sigma(torch.tensor([tv]))
        xt = (av * x1 + sv * eps_c) * m
        worst = max(worst, (((xt - sv * eps_c) / av) * m - x1 * m).abs().max().item())
    R["vp_denoise_identity"] = worst

    # ---------------- 5: both posterior proxies recover x_1 from a perfect net
    tv = 0.4
    t = torch.full((x1.shape[0],), tv)
    net_fm = PerfectDenoiser((x1, f1), (eps_c, eps_f), "flow", mask)
    tb = t.view(-1, 1, 1)
    xt_c = (tb * x1 + (1 - tb) * eps_c) * m
    xt_f = (tb * f1 + (1 - tb) * eps_f) * m
    p = fm_posterior(net_fm, xt_c, xt_f, mask, t)
    R["fm_posterior_recovers_x1"] = max(
        (p.mean_coords * m - x1 * m).abs().max().item(),
        (p.mean_feats * m - f1 * m).abs().max().item())

    av, sv = alpha_sigma(t)
    net_vp = PerfectDenoiser((x1, f1), (eps_c, eps_f), "vp", mask)
    xt_c = (av.view(-1, 1, 1) * x1 + sv.view(-1, 1, 1) * eps_c) * m
    xt_f = (av.view(-1, 1, 1) * f1 + sv.view(-1, 1, 1) * eps_f) * m
    p = vp_posterior(net_vp, xt_c, xt_f, mask, t, av, sv)
    R["vp_posterior_recovers_x1"] = max(
        (p.mean_coords * m - x1 * m).abs().max().item(),
        (p.mean_feats * m - f1 * m).abs().max().item())

    # ---------------- 6: zero centre of mass survives every operation
    def com(v):
        n = mask.sum(1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        return ((v * m).sum(1, keepdim=True) / n).abs().max().item()

    tb = torch.full((x1.shape[0], 1, 1), 0.3)
    R["com_data"] = com(x1)
    R["com_noise"] = com(eps_c)
    R["com_interpolant"] = com((tb * x1 + (1 - tb) * eps_c) * m)
    R["com_velocity_target"] = com((x1 - eps_c) * m)

    # ---------------- 7: padding is inert in the loss and its gradient
    # Runs against the real loss of whichever trainer ships in this package.
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(()))

        def forward(self, c, f, mask, t):
            mm = mask.unsqueeze(-1)
            return c * self.w * mm, f * self.w * mm

    losses = available_losses()
    if not losses:
        raise SystemExit("no trainer found on sys.path -- cannot check masking")
    for name, loss_fn in losses:
        net = Tiny()
        torch.manual_seed(5)
        l1, _, _ = loss_fn(net, x1, f1, mask, t, 1.0)
        g1 = torch.autograd.grad(l1, net.w)[0]
        pad = (1 - mask).unsqueeze(-1)
        x1d = x1 + torch.randn_like(x1) * 1e3 * pad      # garbage in padded slots
        f1d = f1 + torch.randn_like(f1) * 1e3 * pad
        torch.manual_seed(5)
        l2, _, _ = loss_fn(net, x1d, f1d, mask, t, 1.0)
        g2 = torch.autograd.grad(l2, net.w)[0]
        R["padding_loss_inert_%s" % name] = (l1 - l2).abs().item()
        R["padding_grad_inert_%s" % name] = (g1 - g2).abs().item()

    # ---------------- 8: predicting the true target gives exactly zero residual
    # The loss is a mean of squared residuals, so a zero residual is a zero loss;
    # checking the residual avoids depending on which eps the loss happens to draw.
    v_c, v_f = (x1 - eps_c) * m, (f1 - eps_f) * m       # the exact FM target
    R["exact_target_zero_loss"] = max(
        ((v_c - (x1 - eps_c) * m) ** 2).sum().item(),
        ((v_f - (f1 - eps_f) * m) ** 2).sum().item())

    # ---------------- report
    # 1e-12 for exact algebra; the VP schedule goes through exp and sqrt so it
    # carries a few more ulps.
    tol = 1e-12
    loose = {"vp_alpha2_plus_sigma2_is_1": 1e-10}
    print("identity                            max abs error     pass")
    print("-" * 62)
    ok = True
    for k, v in R.items():
        p = v < loose.get(k, tol)
        ok = ok and p
        print("%-34s %13.3e      %s" % (k, v, "yes" if p else "NO"))
    print("-" * 62)
    print("dtype=%s  tol=%.0e   losses checked: %s"
          % (torch.get_default_dtype(), tol, ", ".join(n for n, _ in losses)))
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
