"""Is SMG's problem VARIANCE (fixable by a better estimator) or VALIDITY
(not fixable by any estimator)?

SMG replaces E[f(X) | x_t] by the two-term Taylor expansion

    E[f(X)] ~= f(m) + 1/2 tr(H Sigma)          [ = f(m) + c ]

If that is a good approximation and c is merely hard to ESTIMATE, then
Hutch++, control variates or more probes fix it -- it is a compute problem.
If the expansion itself is wrong over the posterior spread, then estimating c
precisely buys nothing: you would be computing an accurate value for a term
whose use is invalid. No estimator can fix that.

This measures both, directly, on the real guide:

  empirical    E[f(m + delta)] over many draws       <- the truth
  quadratic    f(m) + c                              <- what SMG assumes
  linear       f(m)                                  <- what plug-in assumes

and reports which is closer. It also reports the spread of f over the
posterior, which is what the MC arms (lgd_mc, osc, tfg_mc) integrate directly
WITHOUT needing H at all -- so if the expansion fails, those arms are the
principled answer rather than a cheaper approximation.

    python results/bench/quad_validity.py
"""
from __future__ import annotations

import os
import sys

import argparse

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from guidance import (fm_posterior, hutchinson_tr_H_Sigma,  # noqa: E402
                      trace_scale)
from models.egnn import EGNNVelocity, zero_com  # noqa: E402
from m1_signed_bias import PhysicalProperty  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_DRAW = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guide", default="f_A_mu.pt",
                    help="which f_A this result is ABOUT. The finding is a "
                         "statement about the guide's behaviour off its training "
                         "distribution, so it must be re-measured per guide.")
    args = ap.parse_args()
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    ck = torch.load(os.path.join(ROOT, "betty_pull", "fm_last.pt"),
                    map_location="cpu", weights_only=False)
    ca = ck["args"]
    net = EGNNVelocity(len(types), ca["hidden"], ca["layers"]).to(DEV).eval()
    net.load_state_dict(ck["ema"])
    for p in net.parameters():
        p.requires_grad_(False)
    gpath = os.path.join(ROOT, "proj1", "checkpoints", args.guide)
    gck = torch.load(gpath, map_location="cpu", weights_only=False)
    print("guide: %s  arch=%s  val MAE=%.4f\n"
          % (args.guide, gck.get("arch", "egnn"), gck["val_mae"]))
    f_A = PhysicalProperty(gpath, len(types), DEV)

    g = torch.Generator().manual_seed(5)
    sel = d["split"]["train_a"][torch.randperm(51527, generator=g)[:16]]
    x1_c, x1_f = d["coords"][sel].to(DEV), d["feats"][sel].to(DEV)
    mask = d["mask"][sel].to(DEV)
    m = mask.unsqueeze(-1)

    print("%5s %10s %10s %10s %10s %10s %10s   %s" %
          ("t", "f(m)", "E[f(m+d)]", "f(m)+c", "|lin err|", "|quad err|", "sd f", "verdict"))
    print("-" * 96)
    for tv in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98):
        t = torch.full((len(sel),), tv, device=DEV)
        tb = t.view(-1, 1, 1)
        gd = torch.Generator(device=DEV).manual_seed(7)
        eps_c = zero_com(torch.randn(x1_c.shape, generator=gd, device=DEV), mask)
        eps_f = torch.randn(x1_f.shape, generator=gd, device=DEV) * m
        xt_c = (tb * x1_c + (1 - tb) * eps_c) * m
        xt_f = (tb * x1_f + (1 - tb) * eps_f) * m
        post_fn = lambda c, f: fm_posterior(net, c, f, mask, t)
        with torch.no_grad():
            post = post_fn(xt_c, xt_f)
            fm_val = f_A(post.mean_coords, post.mean_feats, mask)
        m_c, m_f, k = post.mean_coords, post.mean_feats, post.k

        # the correction, well estimated (many probes, so variance is not the issue)
        c = hutchinson_tr_H_Sigma(f_A, post_fn, xt_c, xt_f, mask, m_c, m_f, k,
                                  64, torch.Generator(device=DEV).manual_seed(3))

        # the truth: average f over the posterior spread, trace-matched isotropic
        r2 = trace_scale(post_fn, xt_c, xt_f, mask, k, 8,
                         torch.Generator(device=DEV).manual_seed(11))
        r = r2.clamp(min=0).sqrt().view(-1, 1, 1)
        vals = []
        gd2 = torch.Generator(device=DEV).manual_seed(21)
        with torch.no_grad():
            for _ in range(N_DRAW):
                dc = torch.randn(m_c.shape, generator=gd2, device=DEV) * r * m
                df = torch.randn(m_f.shape, generator=gd2, device=DEV) * r * m
                vals.append(f_A(m_c + dc, m_f + df, mask))
        V = torch.stack(vals)
        emp = V.mean(0)
        sd = V.std(0)

        lin_err = (fm_val - emp).abs().mean().item()
        quad_err = (fm_val + c - emp).abs().mean().item()
        ratio = quad_err / max(lin_err, 1e-30)
        print("%5.2f %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g   %s"
              % (tv, fm_val.mean().item(), emp.mean().item(),
                 (fm_val + c).mean().item(), lin_err, quad_err, sd.mean().item(),
                 ("QUAD HELPS x%.2f" % (1/ratio)) if ratio < 1 else "quad worse x%.1f" % ratio))

    print("\nIf |quad err| > |lin err|, the second-order term makes the estimate WORSE:")
    print("the expansion is outside its validity region and no estimator of c can")
    print("rescue it. The MC arms integrate f over the spread directly and need no H.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
