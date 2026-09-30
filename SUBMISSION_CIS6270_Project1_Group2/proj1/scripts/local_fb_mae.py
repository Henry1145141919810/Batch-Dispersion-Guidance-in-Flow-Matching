"""f_B's error near the q90 target, and the local delta = 2 x that error.

    python proj1/scripts/local_fb_mae.py            # writes results/local_fb_mae.json

v2's pre-registered delta is 2 x f_B's MAE over all val molecules. The
alternative computed here keeps the factor 2 but takes the MAE only over val
molecules whose TRUE property lies near the target, since that is where the
run is scored. "Near" is a window in train_a rank space (the target is
train_a's q90): [Q(0.90 - h), Q(0.90 + h)], h = 2.5 / 5 / 7.5 / 10 percentile
points, primary h = 5. At h = 10 the upper edge is Q(1.00), the train_a
maximum, so that window is one-sided (q80 to the max) for every property; it
is flagged `one_sided` in the output.

f_B trained on train_b, so val is not in its training data -- but val is the
set its best checkpoint was SELECTED on (train_predictor.py keeps the best val
MAE), so this MAE, like the pre-registered one, is slightly optimistic.

THE LIMIT THAT MATTERS MORE THAN THE WINDOW. Both deltas are f_B's error on
REAL QM9 molecules. in_band scores GENERATED ones, of which only ~35-40 % are
molecule-stable; f_B's error on that population is unmeasured and may well be
larger. Moving delta by a few per cent here does not address that.

Also reported per window: the MAE when molecules are selected by f_B's own
prediction instead of the truth (a check -- regression to the mean makes the
two differ in the tail), and the coverage, i.e. the fraction of molecules
truly near the target whose f_B lands within delta (and within 1 x MAE) of
their true value.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

PROPS = ("mu", "alpha", "gap")
Q = 0.90
WINDOWS = (0.025, 0.05, 0.075, 0.10)
PRIMARY_H = 0.05
K = 2.0
N_BOOT = 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "local_fb_mae.json"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    import torch
    from predictor_table import build
    from guidance_sweep import TARGETS

    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else args.device)
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    idx = d["split"]["val"]
    import datetime
    import hashlib
    res = {"rule": "delta = %g x MAE of f_B over val molecules whose true property "
                   "is in [Q_train_a(%.2f - h), Q_train_a(%.2f + h)]" % (K, Q, Q),
           "k": K, "primary_h": PRIMARY_H, "n_val": len(idx),
           "prov": {"torch": torch.__version__, "device": dev,
                    "date": datetime.date.today().isoformat(),
                    "n_boot": N_BOOT, "boot_seed": 20260924, "f_B_md5": {}},
           "delta": {}, "global_mae": {}, "delta_preregistered": {}, "windows": {}}
    g = torch.Generator().manual_seed(20260924)

    for prop in PROPS:
        pi = d["props"].index(prop)
        ck_path = os.path.join(ROOT, "proj1", "checkpoints", "f_B_%s.pt" % prop)
        res["prov"]["f_B_md5"][prop] = hashlib.md5(open(ck_path, "rb").read()).hexdigest()
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        net = build(ck.get("arch", "egnn"), len(d["types"]), ck["args"])
        net.load_state_dict(ck["state_dict"])
        net = net.to(dev).eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                preds.append((net(d["coords"][b].to(dev), d["feats"][b].to(dev),
                                  d["mask"][b].to(dev), torch.ones(len(b), device=dev))
                              * ck["y_std"] + ck["y_mean"]).cpu())
        pred = torch.cat(preds).double()
        y = d["y"][idx, pi].double()
        ya = d["y"][d["split"]["train_a"], pi].double()
        err = (pred - y).abs()
        gmae = float(err.mean())
        if abs(gmae - float(ck["val_mae"])) > 1e-6 * max(1.0, abs(gmae)):
            raise SystemExit("%s: recomputed val MAE %.6g != checkpoint %.6g"
                             % (prop, gmae, float(ck["val_mae"])))
        t = float(TARGETS[prop]["q90"])
        if abs(float(torch.quantile(ya, Q)) - t) > 1e-3 * abs(t):
            raise SystemExit("%s: target %.5g is not train_a's q90 %.5g"
                             % (prop, t, float(torch.quantile(ya, Q))))
        rows = []
        for h in WINDOWS:
            lo, hi = float(torch.quantile(ya, Q - h)), float(torch.quantile(ya, Q + h))
            sel = (y >= lo) & (y <= hi)
            e = err[sel]
            bi = torch.randint(0, e.numel(), (N_BOOT, e.numel()), generator=g)
            sel_p = (pred >= lo) & (pred <= hi)
            m = float(e.mean())
            rows.append({
                "h": h, "lo": lo, "hi": hi, "n": int(sel.sum()),
                "one_sided": bool(hi >= float(ya.max()) - 1e-12),
                "mae": m, "mae_se": float(e[bi].mean(1).std()),
                "signed_err": float((pred - y)[sel].mean()),
                "mae_predsel": float(err[sel_p].mean()), "n_predsel": int(sel_p.sum()),
                "delta": K * m,
                "coverage_local": float((e <= K * m).double().mean()),
                "coverage_1x_local": float((e <= m).double().mean()),
                "coverage_global": float((e <= K * gmae).double().mean()),
            })
        prim = [r for r in rows if r["h"] == PRIMARY_H][0]
        res["delta"][prop] = prim["delta"]
        res["global_mae"][prop] = gmae
        res["delta_preregistered"][prop] = K * gmae
        res["windows"][prop] = rows

        print("%s  target %.4f  global MAE %.5f  -> pre-registered delta %.5f"
              % (prop, t, gmae, K * gmae))
        print("   h      window              n     local MAE (se)      x global  "
              "signed err  MAE sel-by-pred  cover@2xlocal  cover@2xglobal")
        for r in rows:
            print("   %.3f  [%9.4f,%9.4f]  %5d  %.5f (%.5f)  %6.2fx   %+.5f     %.5f (n=%d)   "
                  "%.3f          %.3f%s"
                  % (r["h"], r["lo"], r["hi"], r["n"], r["mae"], r["mae_se"],
                     r["mae"] / gmae, r["signed_err"], r["mae_predsel"], r["n_predsel"],
                     r["coverage_local"], r["coverage_global"],
                     ("   <- primary" if r["h"] == PRIMARY_H else "")
                     + ("   (one-sided: upper edge is the train_a max)" if r["one_sided"] else "")))
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
