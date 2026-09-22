"""G3, run early: is the Hutchinson estimate of 1/2 tr(H Sigma) usable at all?

The RCH fit reported a target standard deviation of 786 for c = 1/2 tr(H Sigma)
on the real f_A, against a dipole scale of ~1.5 D. If that is genuine, SMG's
numerator y - f(m) - c is dominated by c and SMG is not a small correction to
plug-in -- it is a different method. If instead it is ESTIMATOR noise, the
probe count is too low and every SMG arm is running on a random number.

This separates the two: the same states, estimated at increasing probe counts.
If the spread across independent estimates at fixed state collapses as probes
rise, it is estimator noise. If the spread ACROSS states stays large while the
per-state estimate stabilises, the correction is genuinely large.

    python results/bench/probe_noise.py
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from guidance import fm_posterior, hutchinson_tr_H_Sigma, sigma_times_vector  # noqa: E402
from models.egnn import EGNNVelocity, zero_com  # noqa: E402
from m1_signed_bias import PhysicalProperty  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    ck = torch.load(os.path.join(ROOT, "betty_pull", "fm_last.pt"),
                    map_location="cpu", weights_only=False)
    ca = ck["args"]
    net = EGNNVelocity(len(types), ca["hidden"], ca["layers"]).to(DEV).eval()
    net.load_state_dict(ck["ema"])
    for p in net.parameters():
        p.requires_grad_(False)
    f_A = PhysicalProperty(os.path.join(ROOT, "proj1", "checkpoints", "f_A_mu.pt"),
                           len(types), DEV)

    g = torch.Generator().manual_seed(5)
    sel = d["split"]["train_a"][torch.randperm(51527, generator=g)[:32]]
    x1_c, x1_f = d["coords"][sel].to(DEV), d["feats"][sel].to(DEV)
    mask = d["mask"][sel].to(DEV)
    m = mask.unsqueeze(-1)

    print("%6s %10s %12s %12s %12s %12s" %
          ("t", "probes", "mean c", "sd ACROSS st", "sd WITHIN st", "|y-f(m)| ~"))
    print("-" * 70)
    for tv in (0.3, 0.6, 0.9):
        t = torch.full((len(sel),), tv, device=DEV)
        tb = t.view(-1, 1, 1)
        eps_c = zero_com(torch.randn(x1_c.shape, generator=torch.Generator(device=DEV).manual_seed(7), device=DEV), mask)
        eps_f = torch.randn(x1_f.shape, generator=torch.Generator(device=DEV).manual_seed(8), device=DEV) * m
        xt_c = (tb * x1_c + (1 - tb) * eps_c) * m
        xt_f = (tb * x1_f + (1 - tb) * eps_f) * m
        post_fn = lambda c, f: fm_posterior(net, c, f, mask, t)
        with torch.no_grad():
            post = post_fn(xt_c, xt_f)
            fm_val = f_A(post.mean_coords, post.mean_feats, mask)
        m_c, m_f, k = post.mean_coords, post.mean_feats, post.k

        for npb in (1, 4, 32):
            reps = []
            for r in range(5):
                gg = torch.Generator(device=DEV).manual_seed(100 + r)
                reps.append(hutchinson_tr_H_Sigma(f_A, post_fn, xt_c, xt_f, mask,
                                                  m_c, m_f, k, npb, gg))
            Rst = torch.stack(reps)                       # [5, B]
            within = Rst.std(0).mean().item()             # noise at fixed state
            across = Rst.mean(0).std().item()             # real state variation
            print("%6.2f %10d %12.3g %12.3g %12.3g %12.3g"
                  % (tv, npb, Rst.mean().item(), across, within,
                     (fm_val - 2.5).abs().mean().item()))
        # sanity: how big is Sigma itself here?
        gc = torch.ones_like(m_c) * m
        gf = torch.zeros_like(m_f)
        sg_c, sg_f = sigma_times_vector(post_fn, xt_c, xt_f, mask, gc, gf, k)
        print("       k=%.4g  |Sigma 1| mean=%.4g" % (k, sg_c.abs().mean().item()))
    print("\nIf 'sd WITHIN st' >> 'sd ACROSS st', the estimate is noise and the")
    print("probe count must rise (or SMG is unusable on this guide).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
