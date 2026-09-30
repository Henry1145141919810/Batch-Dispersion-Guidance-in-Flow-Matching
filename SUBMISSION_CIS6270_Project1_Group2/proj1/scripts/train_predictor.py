"""S0: train a property predictor on one split.

Two predictors, trained on DISJOINT halves, because guidance is an optimiser
aimed at whatever network it can see:

  f_A  <- train_a   the GUIDE.      Steers sampling. Frozen afterwards.
  f_B  <- train_b   the EVALUATOR.  Scores results. Never touches generation.

Disjoint training data makes their blind spots less correlated, so the f_A - f_B
gap is a usable reward-hacking diagnostic.

Single linear head on the pooled invariant embedding, on purpose (see
INNOVATION_IDEAS_INDEX.md, component B): it keeps the readout weight vector a
constant we can inspect, and keeps the property's level sets planar in embedding
space.

Usage:
  python proj1/scripts/train_predictor.py --split train_a --prop mu --epochs 60
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from models.egnn import EGNNScalar  # noqa: E402
from models.predictors import InvariantTransformer, RidgeDescriptor  # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT_DIR = os.path.join(ROOT, "proj1", "checkpoints")


def batches(idx, bs, shuffle, gen=None):
    order = idx[torch.randperm(len(idx), generator=gen)] if shuffle else idx
    for i in range(0, len(order), bs):
        yield order[i:i + bs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train_a", "train_b"])
    ap.add_argument("--arch", default="egnn",
                    choices=["egnn", "transformer", "ridge"],
                    help="predictor family. f_A (the guide) and f_B (the evaluator) "
                         "should NOT share one: the split makes their data disjoint, "
                         "a different architecture makes their blind spots disjoint "
                         "too. See FA_FB_ARCHITECTURE_DECISION.md. 'ridge' is fitted "
                         "in closed form and ignores --epochs/--lr.")
    ap.add_argument("--heads", type=int, default=8, help="transformer only")
    ap.add_argument("--prop", required=True, choices=["mu", "alpha", "gap"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore an existing checkpoint and start best at inf")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CKPT_DIR, exist_ok=True)

    d = torch.load(DATA, weights_only=False)
    pj = d["props"].index(args.prop)
    coords, feats, mask = d["coords"], d["feats"], d["mask"]
    y = d["y"][:, pj]
    y_mean, y_std = d["y_mean"][pj].item(), d["y_std"][pj].item()

    tr_idx, va_idx = d["split"][args.split], d["split"]["val"]
    name = "f_A" if args.split == "train_a" else "f_B"
    tag = "%s_%s" % (name, args.prop) if args.arch == "egnn" \
        else "%s_%s_%s" % (name, args.prop, args.arch)
    print("%s | %s on %s: %d train, %d val | target mean=%.4f std=%.4f"
          % (tag, args.prop, args.split, len(tr_idx), len(va_idx),
             y_mean, y_std))

    K = len(d["types"])
    if args.arch == "egnn":
        model = EGNNScalar(K, args.hidden, args.layers).to(dev)
    elif args.arch == "transformer":
        model = InvariantTransformer(K, args.hidden, args.layers,
                                     n_heads=args.heads).to(dev)
    else:
        model = RidgeDescriptor(K).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    # RidgeDescriptor holds buffers, not parameters: it is solved in closed
    # form, so it has no optimiser at all.
    if args.arch != "ridge":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print("  parameters: %d" % n_par)

    def evaluate(idx, bs=256):
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in batches(idx, bs, shuffle=False):
                c = coords[b].to(dev, non_blocking=True)
                f = feats[b].to(dev, non_blocking=True)
                m = mask[b].to(dev, non_blocking=True)
                t = torch.ones(len(b), device=dev)
                pred = model(c, f, m, t) * y_std + y_mean
                tot += (pred - y[b].to(dev)).abs().sum().item()
                n += len(b)
        model.train()
        return tot / n                       # MAE in PHYSICAL units

    if args.arch == "ridge":
        # Closed-form ridge. No epochs, no LR; "training" is one linear solve,
        # which is the point of having it as the floor.
        t0 = time.time()
        yz = ((y[tr_idx] - y_mean) / y_std)
        model.fit(coords[tr_idx], feats[tr_idx], mask[tr_idx], yz, device=dev)
        mae = evaluate(va_idx)
        el = time.time() - t0
        print("  ridge fitted in %.1fs | val MAE = %.5f (%s)" % (el, mae, args.prop))
        atomic = {"state_dict": model.state_dict(), "args": vars(args),
                  "y_mean": y_mean, "y_std": y_std, "prop": args.prop,
                  "split": args.split, "val_mae": mae, "arch": args.arch,
                  "seconds": el}
        torch.save(atomic, os.path.join(CKPT_DIR, tag + ".pt"))
        json.dump({"best_val_mae": mae, "history": [], "args": vars(args),
                   "params": sum(p.numel() for p in model.parameters()),
                   "seconds": el},
                  open(os.path.join(CKPT_DIR, tag + "_history.json"), "w"), indent=2)
        print("%s done: best val MAE = %.5f (%s)  in %.0fs" % (tag, mae, args.prop, el))
        return 0

    gen = torch.Generator().manual_seed(args.seed)
    best, hist, t0 = float("inf"), [], time.time()

    # A relaunch used to reset `best` to inf and clobber a better run on its
    # first eval. Seed `best` from whatever is already on disk instead.
    ckpt_path = os.path.join(CKPT_DIR, tag + ".pt")
    if os.path.exists(ckpt_path) and not args.fresh:
        prev = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        best = float(prev.get("val_mae", float("inf")))
        print("  existing %s.pt has val_MAE %.5f; only improvements overwrite it"
              % (tag, best))
    for ep in range(1, args.epochs + 1):
        run, nb = 0.0, 0
        for b in batches(tr_idx, args.batch, shuffle=True, gen=gen):
            c = coords[b].to(dev, non_blocking=True)
            f = feats[b].to(dev, non_blocking=True)
            m = mask[b].to(dev, non_blocking=True)
            t = torch.ones(len(b), device=dev)
            target = ((y[b].to(dev) - y_mean) / y_std)
            loss = torch.nn.functional.mse_loss(model(c, f, m, t), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item()
            nb += 1
        sched.step()

        if ep % 5 == 0 or ep == args.epochs or ep == 1:
            mae = evaluate(va_idx)
            hist.append({"epoch": ep, "train_mse": run / nb, "val_mae": mae})
            star = ""
            if mae < best:
                best = mae
                star = "  <- best"
                torch.save({"state_dict": model.state_dict(), "args": vars(args),
                            "arch": args.arch,
                            "val_mae": mae, "y_mean": y_mean, "y_std": y_std,
                            "prop": args.prop, "split": args.split},
                           ckpt_path)
            print("  ep %3d  train_mse %.5f  val_MAE %.5f%s  (%.0fs)"
                  % (ep, run / nb, mae, star, time.time() - t0))

    with open(os.path.join(CKPT_DIR, tag + "_history.json"), "w") as fh:
        json.dump({"history": hist, "best_val_mae": best,
                   "params": n_par, "args": vars(args)}, fh, indent=2)
    print("%s done: best val MAE = %.5f (%s)  in %.0fs"
          % (tag, best, args.prop, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
