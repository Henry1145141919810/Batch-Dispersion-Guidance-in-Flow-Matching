"""M-1: does the predicted Jensen bias actually occur in our checkpoint?

The whole SMG case rests on one claim: plug-in guidance evaluates f at the
posterior mean m = E[x_1 | x_t], but the likelihood it should be using involves
E[f(x_1) | x_t], and for a nonlinear f these differ by a gap whose leading term
is c = 1/2 tr(H Sigma). This script measures that gap on real data and asks
whether c predicts it.

How E[f(x_1) | x_t] is obtained without posterior sampling: noise REAL
molecules. Each x_t then arrives with its own true x_1, and by the tower
property

    E_{x_t}[ f(m(x_t)) - E[f(x_1) | x_t] ]  =  E[ f(m) - f(x_1) ],

so the population-average gap is a plain sample mean over (x_1, x_t) pairs.
Per pair, f(m) - f(x_1) is a one-draw noisy estimate of the conditional gap,
so we also report the regression of that quantity on c: slope ~1 with c
carrying the signal is the strongest available evidence for the correction.

Decision table (FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md M-1):

    gap signed as predicted, |gap| comparable to delta      -> proceed
    gap real but << delta                                   -> stop: irrelevant
    c does not track the gap                                -> curvature too large
    sign does not follow tr(H Sigma)                        -> investigate posterior

Usage (on Betty, inside a job):
  python proj1/scripts/m1_signed_bias.py --fm proj1/checkpoints/fm.pt \
      --guide proj1/checkpoints/f_A_mu.pt --n 512 --delta 0.3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from models.egnn import EGNNScalar, EGNNVelocity, zero_com  # noqa: E402
from guidance import (fm_posterior, hutchinson_tr_H_Sigma,  # noqa: E402
                      property_derivs, sigma_times_vector)

DATA = os.path.join(ROOT, "data", "qm9.pt")


class PhysicalProperty(nn.Module):
    """Wrap a trained EGNNScalar so it returns the property in physical units
    and matches the f_net(coords, feats, mask) signature guidance.py expects."""

    def __init__(self, ckpt_path, n_types, dev):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        a = ck["args"]
        # Dispatch on the architecture recorded in the checkpoint. Guessing
        # EGNN here would silently load a transformer's weights into the wrong
        # module (or fail obscurely), and every cross-architecture experiment
        # depends on this being right.
        arch = ck.get("arch", "egnn")
        if arch == "transformer":
            from models.predictors import InvariantTransformer
            self.net = InvariantTransformer(n_types, a["hidden"], a["layers"],
                                            n_heads=a.get("heads", 8))
        elif arch == "ridge":
            from models.predictors import RidgeDescriptor
            self.net = RidgeDescriptor(n_types)
        else:
            self.net = EGNNScalar(n_types, a["hidden"], a["layers"])
        self.arch = arch
        self.net.load_state_dict(ck["state_dict"])
        self.net.to(dev).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.y_mean, self.y_std = float(ck["y_mean"]), float(ck["y_std"])
        self.prop = ck["prop"]

    def forward(self, coords, feats, mask):
        return self.net(coords, feats, mask) * self.y_std + self.y_mean

    def embed(self, coords, feats, mask):
        """Pooled invariant embedding, forwarded from the wrapped network.

        Without this, `guidance.py::_diversity_direction` finds no `embed` and
        returns the zero direction, which silently degenerates the `band` arm
        to no edit at all -- it then reports the same number at every guidance
        strength, which is how the degeneracy was caught."""
        return self.net.embed(coords, feats, mask)


def load_fm(ckpt_path, n_types, dev, use_ema=True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    net = EGNNVelocity(n_types, a["hidden"], a["layers"])
    # Two checkpoint layouts: the deliverable <tag>.pt uses
    # state_dict/ema_state_dict, the resume state <tag>_last.pt uses
    # model/ema. fm_v1 ships as the epoch-1500 resume state, so both must work.
    keys = (("ema_state_dict", "ema") if use_ema else ("state_dict", "model"))
    sd = next((ck[k] for k in keys if k in ck), None)
    if sd is None:
        raise SystemExit("%s has none of %s -- not a generator checkpoint"
                         % (os.path.basename(ckpt_path), keys))
    net.load_state_dict(sd)
    net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    ap.add_argument("--guide", required=True)
    ap.add_argument("--n", type=int, default=512, help="validation molecules")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ts", default="0.3,0.5,0.7,0.85,0.95")
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--delta", type=float, default=None,
                    help="evaluator tolerance in physical units. Default: read from "
                         "the S2 result file, which derives it from f_B's MAE. Pass "
                         "a number only to override, and say so in the write-up.")
    ap.add_argument("--s2", default=os.path.join(ROOT, "proj1", "results", "s2_harness.json"))
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=os.path.join(ROOT, "proj1", "results", "m1_signed_bias.json"))
    args = ap.parse_args()

    if args.device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = args.device
    if args.delta is None:
        if not os.path.exists(args.s2):
            raise SystemExit("no --delta and no S2 result at %s: run s2_harness.py first, "
                             "the gate is meaningless against a guessed tolerance" % args.s2)
        with open(args.s2) as fh:
            s2 = json.load(fh)
        args.delta = float(s2["delta"])
        print("delta          : %.4f from S2 (k=%.1f x f_B MAE %.4f)"
              % (args.delta, s2["k_delta"], s2["mae_B"]))
    else:
        print("delta          : %.4f (OVERRIDE on the command line)" % args.delta)
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    d = torch.load(DATA, weights_only=False)
    n_types = len(d["types"])
    net, fm_ck = load_fm(args.fm, n_types, dev)
    f_net = PhysicalProperty(args.guide, n_types, dev)
    print("fm checkpoint  : %s (val_loss %.4f)" % (args.fm, fm_ck.get("val_loss", float("nan"))))
    print("guide          : %s  property=%s  y_std=%.4f" % (args.guide, f_net.prop, f_net.y_std))

    va = d["split"]["val"][: args.n]
    coords_all, feats_all, mask_all = d["coords"][va], d["feats"][va], d["mask"][va]
    ts = [float(x) for x in args.ts.split(",")]
    print("molecules      : %d   t grid: %s   probes: %d" % (len(va), ts, args.n_probe))

    out = {"t": {}, "delta": args.delta, "y_std": f_net.y_std, "n": len(va),
           "n_probe": args.n_probe, "fm": args.fm, "guide": args.guide}
    t0 = time.time()
    for tval in ts:
        gaps, cs, vfs, f_true_all, f_m_all = [], [], [], [], []
        for i in range(0, len(va), args.batch):
            c1 = coords_all[i:i + args.batch].to(dev)
            f1 = feats_all[i:i + args.batch].to(dev)
            m = mask_all[i:i + args.batch].to(dev)
            B = c1.shape[0]
            t = torch.full((B,), tval, device=dev)

            # x_t from the TRUE x_1, exactly the training corruption
            eps_c = zero_com(torch.randn(c1.shape, generator=gen, device=dev), m)
            eps_f = torch.randn(f1.shape, generator=gen, device=dev) * m.unsqueeze(-1)
            tb = t.view(-1, 1, 1)
            xt_c = (tb * c1 + (1 - tb) * eps_c) * m.unsqueeze(-1)
            xt_f = (tb * f1 + (1 - tb) * eps_f) * m.unsqueeze(-1)

            post_fn = lambda c, f: fm_posterior(net, c, f, m, t)  # noqa: E731
            post = post_fn(xt_c, xt_f)
            m_c, m_f, k = post.mean_coords, post.mean_feats, post.k

            with torch.no_grad():
                f_true = f_net(c1, f1, m)
            f_m, g_c, g_f = property_derivs(f_net, m_c, m_f, m)
            c_term = hutchinson_tr_H_Sigma(f_net, post_fn, xt_c, xt_f, m, m_c, m_f, k,
                                           args.n_probe, gen)
            sg_c, sg_f = sigma_times_vector(post_fn, xt_c, xt_f, m, g_c, g_f, k)
            v_f = (g_c * sg_c).sum((1, 2)) + (g_f * sg_f).sum((1, 2))

            gaps.append((f_m - f_true).cpu())
            cs.append(c_term.cpu())
            vfs.append(v_f.cpu())
            f_true_all.append(f_true.cpu())
            f_m_all.append(f_m.cpu())

        gap = torch.cat(gaps)
        c = torch.cat(cs)
        vf = torch.cat(vfs)
        # gap = f(m) - f(x_1). Jensen: E[f(x_1)|x_t] ~ f(m) + c, so E[gap] ~ +c
        # in expectation... careful with sign: f(m) - E[f(x1)|xt] = -c.
        # gap_i = f(m) - f(x1_i) has expectation f(m) - E[f(x1)|xt] = -c_i.
        mean_gap, mean_c = gap.mean().item(), c.mean().item()
        se_gap = (gap.std() / gap.numel() ** 0.5).item()
        # regression of gap on (-c): slope ~ 1 if the correction is right
        x = -c
        xc, gc = x - x.mean(), gap - gap.mean()
        slope = ((xc * gc).sum() / (xc * xc).sum().clamp(min=1e-12)).item()
        corr = ((xc * gc).sum() / (xc.norm() * gc.norm()).clamp(min=1e-12)).item()
        sign_agree = ((gap.sign() == x.sign()) | (x.abs() < 1e-9)).float().mean().item()

        rec = {"mean_gap": mean_gap, "se_gap": se_gap, "mean_c": mean_c,
               "mean_minus_c": -mean_c, "mean_v_f": vf.mean().item(),
               "rms_gap": gap.pow(2).mean().sqrt().item(),
               "rms_c": c.pow(2).mean().sqrt().item(),
               "slope_gap_on_minus_c": slope, "corr": corr,
               "sign_agreement": sign_agree,
               "gap_over_delta": abs(mean_gap) / args.delta,
               "gap_over_ystd": abs(mean_gap) / f_net.y_std,
               "k": k}
        out["t"][str(tval)] = rec
        print("t=%.2f  gap %+.4f +- %.4f   -c %+.4f   slope %.2f  corr %.2f  "
              "sign %.0f%%   |gap|/delta %.2f   v_f %.3f   (%.0fs)"
              % (tval, mean_gap, se_gap, -mean_c, slope, corr, 100 * sign_agree,
                 abs(mean_gap) / args.delta, vf.mean().item(), time.time() - t0))

    # ---- the gate, stated mechanically; the human reads the table above too
    worst_ratio = max(r["gap_over_delta"] for r in out["t"].values())
    tracks = all(r["corr"] > 0.3 for r in out["t"].values())
    signed = all(r["sign_agreement"] > 0.6 for r in out["t"].values())
    if worst_ratio < 0.1:
        verdict = "STOP: gap is real but far below delta - correct arithmetic, irrelevant effect"
    elif not tracks:
        verdict = "INVESTIGATE: c does not track the measured gap - leading term insufficient"
    elif not signed:
        verdict = "INVESTIGATE: sign of gap does not follow tr(H Sigma)"
    else:
        verdict = "PROCEED: gap is signed as predicted and comparable to delta"
    out["verdict"] = verdict
    print("\nVERDICT: " + verdict)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
