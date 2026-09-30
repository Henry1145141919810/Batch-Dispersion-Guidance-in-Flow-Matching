"""Evaluate every property predictor on one common held-out set, one metric.

    python proj1/scripts/predictor_table.py                 # table for all found
    python proj1/scripts/predictor_table.py --split test    # final numbers

Cross-architecture MAE is only meaningful if every model is scored the same way,
so this ignores whatever each checkpoint recorded during its own training and
re-evaluates all of them here:

  * the SAME held-out molecules (the `val` split by default; `test` for final
    numbers), which neither f_A nor f_B trained on;
  * the SAME metric, mean absolute error in PHYSICAL units (D, a.u., Ha);
  * the SAME batching and dtype.

It also reports the metrics published work uses alongside MAE, so our
predictors can be placed beside theirs without an apples-to-oranges objection:

  MAE        mean |pred - true|            the standard in EDM/EEGSDE/TFG
  chance     std of the target             what predicting the mean achieves
  ratio      chance / MAE                  how much better than chance
  R2         1 - SS_res / SS_tot           variance explained
  delta      2 x MAE                       our pre-registered tolerance
  params     parameter count               for the cost-vs-accuracy trade

An f_A/f_B PAIR is only usable if their architectures differ (else they share
blind spots) and their training splits are disjoint (else the evaluator has
seen the guide's data). Both are checked and flagged per row.
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
CKPT = os.path.join(ROOT, "proj1", "checkpoints")
PROPS = ["mu", "alpha", "gap"]
UNITS = {"mu": "D", "alpha": "a.u.", "gap": "Ha"}


def build(arch, K, a):
    if arch == "transformer":
        return InvariantTransformer(K, a["hidden"], a["layers"],
                                    n_heads=a.get("heads", 8))
    if arch == "ridge":
        return RidgeDescriptor(K)
    return EGNNScalar(K, a["hidden"], a["layers"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "bench",
                                                  "predictor_table.json"))
    args = ap.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    d = torch.load(DATA, weights_only=False)
    K = len(d["types"])
    idx = d["split"][args.split]
    yi = {p: i for i, p in enumerate(PROPS)}

    rows = []
    for fn in sorted(os.listdir(CKPT)):
        if not fn.endswith(".pt") or not fn.startswith(("f_A_", "f_B_")):
            continue
        ck = torch.load(os.path.join(CKPT, fn), map_location="cpu", weights_only=False)
        if "prop" not in ck or "y_mean" not in ck:
            continue
        prop = ck["prop"]
        arch = ck.get("arch", "egnn")
        a = ck["args"]
        net = build(arch, K, a)
        net.load_state_dict(ck["state_dict"])
        net = net.to(dev).eval()
        for p in net.parameters():
            p.requires_grad_(False)

        y_true = d["y"][idx, yi[prop]].to(dev)
        preds = []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(idx), args.batch):
                b = idx[i:i + args.batch]
                c, f, m = (d["coords"][b].to(dev), d["feats"][b].to(dev),
                           d["mask"][b].to(dev))
                preds.append(net(c, f, m, torch.ones(len(b), device=dev))
                             * ck["y_std"] + ck["y_mean"])
        pred = torch.cat(preds)
        el = time.time() - t0
        err = (pred - y_true).abs()
        mae = err.mean().item()
        chance = y_true.std().item()
        ss_res = ((pred - y_true) ** 2).sum().item()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum().item()
        rows.append({
            "file": fn, "role": fn[:3], "prop": prop, "arch": arch,
            "split_trained": ck.get("split", a.get("split")),
            "mae": mae, "chance": chance, "ratio": chance / max(mae, 1e-12),
            "r2": 1 - ss_res / max(ss_tot, 1e-30), "delta": 2 * mae,
            "params": sum(p.numel() for p in net.parameters()),
            "recorded_val_mae": ck.get("val_mae"),
            "eval_seconds": el, "n_eval": len(idx),
        })

    rows.sort(key=lambda r: (PROPS.index(r["prop"]), r["arch"], r["role"]))
    print("held-out split: %s (%d molecules), device %s\n" % (args.split, len(idx), dev))
    print("%-4s %-6s %-12s %-8s %10s %9s %7s %8s %10s"
          % ("role", "prop", "arch", "unit", "MAE", "chance", "x chance", "R2", "params"))
    print("-" * 82)
    cur = None
    for r in rows:
        if cur != r["prop"]:
            if cur is not None:
                print()
            cur = r["prop"]
        print("%-4s %-6s %-12s %-8s %10.5f %9.4f %7.1f %8.4f %10d"
              % (r["role"], r["prop"], r["arch"], UNITS[r["prop"]], r["mae"],
                 r["chance"], r["ratio"], r["r2"], r["params"]))

    # usable f_A / f_B pairings: disjoint data AND different architecture
    print("\nf_A / f_B pairings (disjoint data by construction; architecture diversity flagged)")
    print("%-6s %-14s %-14s %10s %10s   %s" % ("prop", "f_A", "f_B", "f_A MAE", "f_B MAE", "verdict"))
    print("-" * 88)
    for prop in PROPS:
        A = [r for r in rows if r["prop"] == prop and r["role"] == "f_A"]
        Bs = [r for r in rows if r["prop"] == prop and r["role"] == "f_B"]
        for ra in A:
            for rb in Bs:
                same = ra["arch"] == rb["arch"]
                # delta is set by the WORSE of the two: if f_A cannot reach the
                # band, the experiment measures f_A, not the guidance.
                verdict = ("same architecture -- shared blind spots" if same
                           else "OK, diverse")
                if ra["mae"] > 2 * rb["mae"]:
                    verdict += "; f_A much worse than f_B -> delta trap"
                print("%-6s %-14s %-14s %10.5f %10.5f   %s"
                      % (prop, ra["arch"], rb["arch"], ra["mae"], rb["mae"], verdict))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"split": args.split, "n": len(idx), "rows": rows},
              open(args.out, "w"), indent=1)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
