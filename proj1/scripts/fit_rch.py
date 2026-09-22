"""Fit the Residual Calibration Head (Haimo v1) -- the only arm that trains.

    python proj1/scripts/fit_rch.py --fm betty_pull/fm_last.pt \
        --guide proj1/checkpoints/f_A_mu.pt --prop mu --n 2000

RCH learns SMG's correction c = 1/2 tr(H Sigma) instead of deriving it, so that
it can be applied at plug-in cost (one VJP, no JVP, no HVP). The fit:

  1. take real molecules from train_a (the guide's own split -- fitting on
     train_b would leak the evaluator's data into an inference-time method);
  2. noise each to a random t along the SAME interpolant the generator uses;
  3. compute the EXACT target at that state. Two targets are cached, matching
     Haimo v1's own control:
       c_exact     = 1/2 tr(H Sigma), the quantity SMG computes  -> "residual"
       f1_minus_fm = f_A(x_1) - f_A(m), the realised gap         -> "direct"
     Haimo's result is that the residual parameterisation is inert, so both are
     fitted and both reported;
  4. solve the ridge system in closed form on the cached (phi, target) pairs.

The head is deliberately cheap and deliberately weak: if it reproduces c, SMG's
per-state JVP/HVP is doing the work of a lookup table, which deflates the SMG
claim. That is the point of the arm.

Writes proj1/checkpoints/rch_<prop>.pt with both fits and their held-out R^2.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from guidance import (ResidualCalibrationHead, fm_posterior,  # noqa: E402
                      hutchinson_tr_H_Sigma)
from models.egnn import EGNNVelocity, zero_com  # noqa: E402
from m1_signed_bias import PhysicalProperty  # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT = os.path.join(ROOT, "proj1", "checkpoints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=os.path.join(ROOT, "betty_pull", "fm_last.pt"))
    ap.add_argument("--guide", required=True)
    ap.add_argument("--prop", required=True, choices=["mu", "alpha", "gap"])
    ap.add_argument("--n", type=int, default=2000, help="molecules to cache")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-probe", type=int, default=8,
                    help="Hutchinson probes for the EXACT target; higher than "
                         "inference because this is a one-off fit")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    d = torch.load(DATA, weights_only=False)
    types = d["types"]

    ck = torch.load(args.fm, map_location="cpu", weights_only=False)
    ca = ck["args"]
    sd = ck.get("ema") or ck.get("ema_state_dict")
    net = EGNNVelocity(len(types), ca["hidden"], ca["layers"]).to(dev).eval()
    net.load_state_dict(sd)
    for p in net.parameters():
        p.requires_grad_(False)
    f_A = PhysicalProperty(args.guide, len(types), dev)

    # train_a only: this is an inference-time method for the GUIDE, so it may
    # see what the guide saw and must not see train_b.
    g = torch.Generator().manual_seed(args.seed)
    tr = d["split"]["train_a"]
    pick = tr[torch.randperm(len(tr), generator=g)[: args.n]]

    PHI, C_EX, GAP = [], [], []
    t0 = time.time()
    for i in range(0, len(pick), args.batch):
        sel = pick[i:i + args.batch]
        x1_c = d["coords"][sel].to(dev)
        x1_f = d["feats"][sel].to(dev)
        mask = d["mask"][sel].to(dev)
        B = len(sel)
        t = torch.rand(B, generator=g).to(dev) * 0.98 + 0.01
        tb = t.view(-1, 1, 1)
        m = mask.unsqueeze(-1)
        eps_c = zero_com(torch.randn_like(x1_c), mask)
        eps_f = torch.randn_like(x1_f) * m
        xt_c = (tb * x1_c + (1 - tb) * eps_c) * m
        xt_f = (tb * x1_f + (1 - tb) * eps_f) * m

        post_fn = lambda c, f: fm_posterior(net, c, f, mask, t)
        with torch.no_grad():
            post = post_fn(xt_c, xt_f)
        m_c, m_f, k = post.mean_coords, post.mean_feats, post.k

        c_ex = hutchinson_tr_H_Sigma(f_A, post_fn, xt_c, xt_f, mask, m_c, m_f,
                                     k, args.n_probe, None)
        with torch.no_grad():
            gap = f_A(x1_c, x1_f, mask) - f_A(m_c, m_f, mask)
            phi = ResidualCalibrationHead.features(xt_c, xt_f, mask, t, k)
        PHI.append(phi.cpu()); C_EX.append(c_ex.cpu()); GAP.append(gap.cpu())
        if i == 0:
            print("  first batch %.1fs -> ~%.0fs total"
                  % (time.time() - t0, (time.time() - t0) * len(pick) / args.batch))

    PHI = torch.cat(PHI).double()
    targets = {"residual": torch.cat(C_EX).double(), "direct": torch.cat(GAP).double()}
    n = len(PHI)
    ntr = int(0.8 * n)
    out = {"prop": args.prop, "guide": os.path.basename(args.guide),
           "fm": os.path.basename(args.fm), "n": n, "args": vars(args)}

    print("\n%-10s %10s %10s %10s" % ("target", "train R2", "held-out R2", "target std"))
    print("-" * 44)
    for name, y in targets.items():
        Xtr, ytr = PHI[:ntr], y[:ntr]
        Xte, yte = PHI[ntr:], y[ntr:]
        # (1-t) is folded into the head at inference, so fit on the scaled design
        one_minus_t = (1.0 - PHI[:, 1]).double()
        Atr = Xtr * one_minus_t[:ntr].unsqueeze(1)
        Ate = Xte * one_minus_t[ntr:].unsqueeze(1)
        G = Atr.T @ Atr + args.ridge * torch.eye(Atr.shape[1], dtype=torch.float64)
        beta = torch.linalg.solve(G, Atr.T @ ytr)
        def r2(A, yy):
            pred = A @ beta
            ss = ((yy - pred) ** 2).sum()
            tt = ((yy - yy.mean()) ** 2).sum().clamp(min=1e-30)
            return float(1 - ss / tt)
        out[name] = {"beta": beta.float(), "r2_train": r2(Atr, ytr),
                     "r2_test": r2(Ate, yte), "target_std": float(y.std())}
        print("%-10s %10.4f %10.4f %10.4g"
              % (name, out[name]["r2_train"], out[name]["r2_test"], out[name]["target_std"]))

    path = os.path.join(CKPT, "rch_%s.pt" % args.prop)
    torch.save(out, path)
    print("\nwrote %s" % path)
    print("Read the held-out R2 as the deflation test: if 'residual' R2 is high, a\n"
          "lookup table reproduces SMG's correction and SMG's per-state JVP/HVP is\n"
          "not earning its cost. If it is low, the state dependence is real.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
