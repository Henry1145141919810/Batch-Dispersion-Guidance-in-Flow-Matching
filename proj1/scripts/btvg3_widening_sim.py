"""BTVG-3: does WIDENING the property law raise in-band coverage on v2?

BTVG concentrates: its KL coefficient b = -1/2(1/tau^2 - 1/V_F) is clamped to
<= 0 (guidance.py:1494), so the arm may narrow the observable's law but never
widen it. This script asks whether that clamp is costing us in-band on the v2
full run, where the target is a FIXED q90 and the generator's mean sits far
below it.

THE ANALYTIC CLAIM UNDER TEST. For f ~ N(mu, sigma^2), bias b = mu - y and
band half-width delta,

    P  = Phi((delta-b)/sigma) - Phi((-delta-b)/sigma)
    dP/dsigma = -(1/sigma^2)[(delta-b) phi((delta-b)/sigma)
                             + (delta+b) phi((delta+b)/sigma)]

so with g(x) = x phi(x), widening helps iff g((b-delta)/sigma) > g((b+delta)/sigma),
i.e. iff

    sigma^2 < 2 b delta / ln((b+delta)/(b-delta))          [needs b > delta]

and the optimum is sigma* = b, giving P_max ~ 0.4839 delta / b.

WHAT IS SIMULATED, AND WHAT IS NOT. The intervention is applied to the REAL
per-molecule residuals r_i = f_B,i - y from the v2 sidecars, as

    widen(s)   : r -> b + s (r - b)        spread scaled about its own mean
    debias(c)  : r -> c b + (r - b)        mean pulled toward the target

No Gaussian shape is assumed anywhere in the empirical sweep; the closed form
above is reported beside it only as a check. This is a REWEIGHTING of the
observed property law, not a sampler run: it says what in-band WOULD be if a
guidance change produced that law, and it cannot say whether any guidance
change can produce it, nor what it would cost in chemistry. Both caveats are
printed with the table and must travel with any number taken from here.

Cheap by construction: ~5k floats per cell, one pass per scale point. No GPU.

Usage:  python proj1/scripts/btvg3_widening_sim.py [--json-out PATH] [--md-out PATH]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

V2_GLOB = "results/full/v2/n5000/seed*/*.permol.pt"
SCALES = np.round(np.concatenate([np.arange(0.50, 1.00, 0.05),
                                  np.arange(1.00, 4.01, 0.10)]), 4)
DEBIAS = np.round(np.arange(0.0, 1.01, 0.05), 4)


def widen_threshold(b, delta):
    """sigma^2 below which widening raises coverage; nan when b <= delta."""
    b = abs(b)
    if b <= delta:
        return float("nan")
    return 2.0 * b * delta / math.log((b + delta) / (b - delta))


def gauss_in_band(b, sigma, delta):
    z = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return z((delta - b) / sigma) - z((-delta - b) / sigma)


def cell_rows(paths):
    """(prop, arm, seed) -> residuals r = f_B - y, and delta."""
    out = []
    for p in sorted(paths):
        J = json.load(open(p.replace(".permol.pt", ".json")))
        d = torch.load(p, map_location="cpu", weights_only=False)
        fin = d["finite"].numpy()
        r = (d["f_B"].numpy() - d["y"].numpy())[fin].astype(np.float64)
        out.append({"prop": J["prop"], "arm": J["arm"], "seed": J["seed"],
                    "w": J["w"], "delta": float(J["delta"]), "r": r,
                    "n": int(fin.sum()), "n_targets": int(np.unique(d["y"].numpy()).size),
                    "fm_md5": (J.get("prov") or {}).get("fm_md5")})
    return out


def analyse(cell):
    r, delta = cell["r"], cell["delta"]
    b, sd = float(r.mean()), float(r.std(ddof=1))
    # --- empirical sweeps on the real residuals -------------------------
    dev = r - b
    inb_widen = [(float(s), float((np.abs(b + s * dev) <= delta).mean()))
                 for s in SCALES]
    inb_debias = [(float(c), float((np.abs(c * b + dev) <= delta).mean()))
                  for c in DEBIAS]
    s_star, p_star = max(inb_widen, key=lambda t: t[1])
    base = float((np.abs(r) <= delta).mean())
    # --- the closed form, for comparison only ---------------------------
    thr = widen_threshold(b, delta)
    return {
        "prop": cell["prop"], "arm": cell["arm"], "seed": cell["seed"],
        "w": cell["w"], "n": cell["n"], "delta": delta,
        "n_targets": cell["n_targets"], "fm_md5": cell["fm_md5"],
        "bias_over_delta": b / delta, "sd_over_delta": sd / delta,
        "sd_over_bias": sd / abs(b) if b else float("nan"),
        "in_band": base,
        "in_band_ceiling": gauss_in_band(0.0, sd, delta),
        "widen_helps_analytic": bool(sd ** 2 < thr) if thr == thr else False,
        "sigma2_over_threshold": (sd ** 2 / thr) if thr == thr else float("nan"),
        "s_opt_gauss": abs(b) / sd if sd else float("nan"),
        "p_max_gauss": gauss_in_band(b, abs(b), delta) if b else float("nan"),
        "s_opt_emp": s_star, "in_band_at_s_opt": p_star,
        "gain_widen": p_star - base,
        "in_band_debiased": inb_debias[0][1],
        "gain_debias": inb_debias[0][1] - base,
        "curve_widen": inb_widen, "curve_debias": inb_debias,
    }


PROBES = [(9.0, 7.0, 1.0), (9.0, 12.0, 1.0), (3.0, 1.0, 1.0), (0.5, 2.0, 1.0),
          (20.0, 15.0, 1.0), (2.0, 4.0, 1.0), (9.142, 7.61, 1.0),
          (17.30, 14.22, 1.0), (5.94, 5.14, 1.0), (1.5, 0.2, 1.0)]


def exact_check():
    """Primary check, deterministic: the closed form's sign against a central
    difference of the exact Gaussian coverage. No sampling, so no noise."""
    bad = []
    for b, sd, delta in PROBES:
        thr = widen_threshold(b, delta)
        pred = (sd ** 2 < thr) if thr == thr else False
        h = 1e-5 * sd
        d = (gauss_in_band(b, sd + h, delta) - gauss_in_band(b, sd - h, delta)) / (2 * h)
        if abs(d) < 1e-12:            # sitting on the turning point
            continue
        if pred != (d > 0):
            bad.append((b, sd, delta, pred, d > 0, d))
    return bad


def mc_check(rng, n=2000000):
    """Secondary check, paired: the SAME normal draws are rescaled for both
    arms, so the comparison is a paired difference and the O(1/sqrt(n)) noise
    on the level cancels. An unpaired version needs ~10^9 draws to resolve the
    smallest probe here and will report spurious failures."""
    bad = []
    z = rng.standard_normal(n)
    for b, sd, delta in PROBES:
        thr = widen_threshold(b, delta)
        pred = (sd ** 2 < thr) if thr == thr else False
        h = 0.02 * sd
        lo = (np.abs(b + (sd - h) * z) <= delta).mean()
        hi = (np.abs(b + (sd + h) * z) <= delta).mean()
        if abs(hi - lo) < 3e-5:
            continue
        if pred != (hi > lo):
            bad.append((b, sd, delta, pred, hi > lo, lo, hi))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="results/btvg3_widening_sim.json")
    ap.add_argument("--md-out", default="docs/results/BTVG3_WIDENING_SIM.md")
    ap.add_argument("--glob", default=V2_GLOB)
    a = ap.parse_args()

    rng = np.random.default_rng(20260925)
    bad = exact_check()
    if bad:
        print("CLOSED FORM FAILED THE EXACT CHECK:")
        for x in bad:
            print("   b=%g sd=%g delta=%g predicted=%s measured=%s dP/dsigma=%.3e" % x)
        sys.exit(2)
    bad = mc_check(rng)
    if bad:
        print("CLOSED FORM FAILED THE PAIRED MONTE-CARLO CHECK:")
        for x in bad:
            print("   b=%g sd=%g delta=%g predicted=%s measured=%s (%.6f -> %.6f)" % x)
        sys.exit(2)
    print("closed form agrees with the exact derivative and with a paired MC "
          "on %d/%d probes\n" % (len(PROBES), len(PROBES)))

    paths = glob.glob(a.glob)
    if not paths:
        sys.exit("no sidecars matched %r" % a.glob)
    rows = [analyse(c) for c in cell_rows(paths)]

    # pooled over seeds
    keys = sorted({(r["prop"], r["arm"]) for r in rows})
    pooled = []
    for prop, arm in keys:
        g = [r for r in rows if r["prop"] == prop and r["arm"] == arm]
        m = lambda k: float(np.mean([x[k] for x in g]))
        pooled.append({
            "prop": prop, "arm": arm, "seeds": len(g),
            "bias_over_delta": m("bias_over_delta"), "sd_over_delta": m("sd_over_delta"),
            "sd_over_bias": m("sd_over_bias"),
            "in_band": m("in_band"), "in_band_ceiling": m("in_band_ceiling"),
            "widen_helps": all(x["widen_helps_analytic"] for x in g),
            "s_opt_emp": m("s_opt_emp"), "in_band_at_s_opt": m("in_band_at_s_opt"),
            "gain_widen": m("gain_widen"),
            "in_band_debiased": m("in_band_debiased"), "gain_debias": m("gain_debias"),
        })

    hdr = ("%-6s %-9s %9s %8s %8s | %6s %7s %8s | %8s %8s"
           % ("prop", "arm", "bias/d", "sd/d", "sd/|b|", "in_bd", "s_opt",
              "widened", "debiased", "ceiling"))
    print(hdr)
    print("-" * len(hdr))
    for r in pooled:
        print("%-6s %-9s %9.2f %8.2f %8.3f | %6.4f %7.2f %8.4f | %8.4f %8.4f"
              % (r["prop"], r["arm"], r["bias_over_delta"], r["sd_over_delta"],
                 r["sd_over_bias"], r["in_band"], r["s_opt_emp"],
                 r["in_band_at_s_opt"], r["in_band_debiased"], r["in_band_ceiling"]))
    print()
    gw = float(np.mean([r["gain_widen"] for r in pooled]))
    gd = float(np.mean([r["gain_debias"] for r in pooled]))
    print("mean gain from optimal widening : %+.4f  (%.0f%% of current in_band)"
          % (gw, 100 * gw / float(np.mean([r["in_band"] for r in pooled]))))
    print("mean gain from removing the bias: %+.4f  (%.0f%% of current in_band)"
          % (gd, 100 * gd / float(np.mean([r["in_band"] for r in pooled]))))
    print("\nthe widening sweep is a reweighting of the OBSERVED property law;")
    print("it models no sampler run and charges nothing for chemistry.")

    prov = {
        "script": os.path.relpath(__file__).replace("\\", "/"),
        "script_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
        "glob": a.glob, "n_cells": len(rows),
        "cell_fm_md5": sorted({r["fm_md5"] for r in rows if r["fm_md5"]}),
        "numpy": np.__version__, "torch": torch.__version__,
        "python": sys.version.split()[0],
    }
    try:
        prov["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        prov["git_head"] = None

    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    json.dump({"prov": prov, "pooled": pooled, "cells": rows},
              open(a.json_out, "w"), indent=1, default=float)
    print("\nwrote %s" % a.json_out)


if __name__ == "__main__":
    main()
