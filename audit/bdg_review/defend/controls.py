"""Blue-team check 2: the controls the objection says are missing.

(1) Matched-spread chemistry: BDG's tightest cell (e4t0.5) vs the plug weight
    that reaches the same spread (plug w=4), paired by initial noise, GPU-lens
    cells (n=256, seed 20260925). If BDG's tightening were 'plug at w_eff 2-4',
    chemistry at matched spread would be the same.
(2) MAE along the ladder: how much of the MAE rise is spread alone? Recentre
    each BDG cell's f_B residual to plug w=1's bias and recompute MAE.
(3) The plug row section 6.3 lacks, from results/sweep (fm_last, B200, batch 128,
    n=512, seed 20260921): spread range AND bias swing over plug's w sweep,
    against btvg_var's tau/w sweep. Spread range alone cannot discriminate; the
    discriminating statistic is spread moved per delta of bias moved.
"""
import glob
import json
import math
import os
import re

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(os.path.dirname(HERE), "cells")
REPO = r"C:\Users\mooooonesy\Downloads\pennstuff\cis 6270\Project 1"
rng = np.random.default_rng(1)


def load(prop, tgt, arm, w, var):
    stem = "%s__%s__%s__w%s__tmin0.5__%s" % (prop, arm, tgt, w, var)
    j = json.load(open(os.path.join(CELLS, stem + ".json")))
    p = torch.load(os.path.join(CELLS, stem + ".permol.pt"), weights_only=False)
    return j, p


def paired_boot(fn, a, b, n_boot=4000):
    n = len(a)
    idx = rng.integers(0, n, (n_boot, n))
    return float(np.std([fn(a[i]) - fn(b[i]) for i in idx]))


print("=" * 78)
print("(1) matched-spread chemistry: BDG e4t0.5 vs plug w4 (paired, same noise)")
print("  %-10s %8s %8s %8s %8s %8s %7s %8s %8s" % (
    "block", "sd BDG", "sd plug4", "ms BDG", "ms plug4", "d(ms)", "z", "bias BDG", "bias p4"))
for tgt in ("q90", "q50"):
    for prop in ("gap", "mu", "alpha"):
        ju, pu = load(prop, tgt, "unguided", 0, "bdgctl")
        jb, pb = load(prop, tgt, "bdg", 1, "e4t0.5")
        j4, p4 = load(prop, tgt, "plug", 4, "tgt")
        d, y = ju["delta"], float(pu["y"][0])
        sdu = pu["f_A"].numpy().std(ddof=1)
        msb = pb["mol_stable"].numpy().astype(float)
        ms4 = p4["mol_stable"].numpy().astype(float)
        se = paired_boot(np.mean, msb, ms4)
        print("  %-10s %8.3f %8.3f %8.4f %8.4f %+8.4f %7.1f %+8.2f %+8.2f" % (
            prop + " " + tgt, pb["f_A"].numpy().std(ddof=1) / sdu, p4["f_A"].numpy().std(ddof=1) / sdu,
            msb.mean(), ms4.mean(), msb.mean() - ms4.mean(), (msb.mean() - ms4.mean()) / se,
            (pb["f_A"].numpy().mean() - y) / d, (p4["f_A"].numpy().mean() - y) / d))

print("=" * 78)
print("(2) MAE/delta along the ladder (f_B): observed vs recentred to plug w1's bias")
for tgt in ("q90", "q50"):
    for prop in ("gap", "mu", "alpha"):
        j1, p1 = load(prop, tgt, "plug", 1, "bdgctl")
        d, y = j1["delta"], float(p1["y"][0])
        r1 = (p1["f_B"].numpy() - y) / d
        b1 = r1.mean()
        line = []
        for tm in (0.5, 1, 1.5):
            jb, pb = load(prop, tgt, "bdg", 1, "e4t%s" % tm)
            r = (pb["f_B"].numpy() - y) / d
            mae = np.abs(r).mean()
            mae_rc = np.abs(r - r.mean() + b1).mean()
            line.append("t%-4s MAE %6.2f  at plug-w1 bias %6.2f" % (tm, mae, mae_rc))
        print("  %-10s plug w1 MAE %6.2f | %s" % (prop + " " + tgt, np.abs(r1).mean(), " | ".join(line)))

print("=" * 78)
print("(3) section 6.3's missing plug row, results/sweep (n=512, seed 20260921, B200)")
SW = os.path.join(REPO, "results", "sweep")


def stats(path):
    c = json.load(open(path))
    d = c["delta"]
    bias = c["f_B_mean"] - c["target_mean"]
    sd = math.sqrt(max(c["prop_rmse_eval"] ** 2 - bias ** 2, 0.0))
    return bias / d, sd, c["mol_stability"], c.get("w"), c


for tgt in ("q50", "q90"):
    for prop in ("mu", "alpha", "gap"):
        bu, sdu, msu, _, _ = stats(os.path.join(SW, "%s__unguided__%s__w1__tmin0.5__cmp.json" % (prop, tgt)))
        for arm, pat in (("plug", "%s__plug__%s__w*__tmin0.5__cmp.json"),
                         ("btvg_var", "%s__btvg_var__%s__w*__tmin0.5*.json")):
            rows = []
            for f in sorted(glob.glob(os.path.join(SW, pat % (prop, tgt)))):
                if "__tgt" in f:
                    continue
                b, sd, ms, w, c = stats(f)
                rows.append((float(w), os.path.basename(f), b, sd / sdu, ms))
            if not rows:
                continue
            R = np.array([r[3] for r in rows])
            Bv = np.array([r[2] for r in rows])
            span_sd = (R.max() - R.min()) * 100
            span_b = Bv.max() - Bv.min()
            print("  %-5s %-4s %-8s n=%2d  sd/ug %.3f..%.3f (range %5.1f%%)  bias %+7.2f..%+7.2f (swing %5.2f d)"
                  "  unguided bias %+7.2f  sd-range per delta of bias %5.1f%%" % (
                      prop, tgt, arm, len(rows), R.min(), R.max(), span_sd, Bv.min(), Bv.max(), span_b, bu,
                      span_sd / max(span_b, 1e-9)))

# BDG ladder, same statistic, from mode_split.json (GPU lens cells, f_A)
ms = json.load(open(os.path.join(HERE, "mode_split.json")))
print("  --- BDG tau ladder (e4t0.5..e4t1.5, GPU-lens cells, f_A, n=256) ---")
for k, v in ms.items():
    R = np.array([r["sd_ratio"] for r in v["rows"]])
    Bv = np.array([r["bias"] for r in v["rows"]])
    span_sd = (R.max() - R.min()) * 100
    span_b = Bv.max() - Bv.min()
    print("  %-10s bdg     sd/ug %.3f..%.3f (range %5.1f%%)  bias swing %5.2f d  sd-range per delta of bias %5.1f%%" % (
        k, R.min(), R.max(), span_sd, span_b, span_sd / span_b))
