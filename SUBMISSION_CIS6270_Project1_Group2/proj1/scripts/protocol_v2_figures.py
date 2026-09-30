"""Figures for the v2 full-run protocol: target choice and strength choice.

    python proj1/scripts/protocol_v2_figures.py

Writes docs/results/figs/
  targets_<prop>.png      QM9 distribution, the chosen target, the +-delta band,
                          and what the UNGUIDED generator actually produces
  strength_<prop>.png     per arm: in_band and mol_stability against strength,
                          the chemistry floor, and the strength the rule picks
  pareto_<prop>.png       in_band against mol_stability, every arm x strength

Sources (screening cells at the target itself, so the plots are the decision):
  results/sweep/*__cmp.json        n=512, seed 20260921, target q90
  results/full/n5000/seed*/...permol.pt   the unguided generator's own output
  data/qm9.pt                      the QM9 distribution and delta

Nothing here reads the v2 full run; these are the plots that CHOOSE its
settings, so they must be produced (and the choices frozen) before it runs.
"""
from __future__ import annotations

import glob
import json
import os

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import torch                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGS = os.path.join(ROOT, "docs", "results", "figs")
PROPS = ("mu", "alpha", "gap")
LABEL = {"mu": r"$\mu$  (Debye)", "alpha": r"$\alpha$  (Bohr$^3$)",
         "gap": r"$\Delta\varepsilon$  (Hartree)"}
NICE = {"mu": "mu", "alpha": "alpha", "gap": "gap"}
# The q90 of train_a: the value the existing screening cells already used, so
# every one of those cells is screening data for this protocol.
TARGET = {"mu": 4.6627, "alpha": 85.05, "gap": 0.3162}
ARMS = ("plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var")
COL = {"plug": "#1f77b4", "tmpd": "#ff7f0e", "lgd_mc": "#2ca02c",
       "tfg": "#d62728", "btvg": "#9467bd", "btvg_var": "#8c564b"}
FLOOR = 0.9


def delta_of(prop):
    """The pre-registered band, 2 x the evaluator's validation MAE."""
    fn = glob.glob(os.path.join(ROOT, "results", "full", "n5000", "seed20261001",
                                "%s__unguided__dist__w1__tmin0.5__full.json" % prop))
    return json.load(open(fn[0]))["delta"]


WIN = 0.5            # the screen's guidance window; see freeze_v2.load


def screen_cells():
    """{(prop, arm, w): row} for the q90 screen, EXACTLY the cells freeze_v2
    reads: any stage (so the `__cmp` screen and the `__tgt` grid extension are
    one grid) but only the 0.5 window, because results/sweep also holds q90
    cells at 0.05 under the same keys."""
    out = {}
    for fn in sorted(glob.glob(os.path.join(ROOT, "results", "sweep", "*.json"))):
        try:
            r = json.load(open(fn, encoding="utf-8"))
        except Exception:
            continue
        if r.get("target_name") != "q90":
            continue
        if abs(float(r.get("t_min_guide", -1)) - WIN) > 1e-12:
            continue
        out[(r["prop"], r["arm"], float(r["w"]))] = r
    return out


def fig_targets(d, prop, dl):
    i = d["props"].index(prop)
    y = d["y"][d["split"]["train_a"], i].double().numpy()
    t = TARGET[prop]
    gen = None
    fn = glob.glob(os.path.join(ROOT, "results", "full", "n5000", "seed20261001",
                                "%s__unguided__dist__w1__tmin0.5__full.permol.pt" % prop))
    if fn:
        pm = torch.load(fn[0], weights_only=False)
        gen = pm["f_B"][pm["finite"]].double().numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    lo, hi = float(min(y.min(), t - 6 * dl)), float(y.max())
    hi = min(hi, float(torch.tensor(y).quantile(0.999)))
    ax.hist(y, bins=120, range=(lo, hi), density=True, color="#bbbbbb",
            label="QM9 (train_a), %d molecules" % len(y))
    if gen is not None:
        ax.hist(gen, bins=120, range=(lo, hi), density=True, histtype="step",
                lw=1.6, color="#2ca02c", label="unguided generator output")
    ax.axvspan(t - dl, t + dl, color="#d62728", alpha=0.20, lw=0,
               label=r"acceptance band $\pm\delta$ = %.4g" % dl)
    ax.axvline(t, color="#d62728", lw=2.0)
    frac = float(((y >= t - dl) & (y <= t + dl)).mean())
    ax.annotate("target = q90 = %.4g\n%.1f%% of QM9 in band" % (t, 100 * frac),
                xy=(t, ax.get_ylim()[1] * 0.92), xytext=(8, 0),
                textcoords="offset points", va="top", fontsize=9, color="#d62728")
    ax.set_xlabel(LABEL[prop])
    ax.set_ylabel("density")
    ax.set_title("%s: the fixed target is the 90th percentile of QM9" % NICE[prop])
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = os.path.join(FIGS, "targets_%s.png" % prop)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, frac


sys.path.insert(0, HERE)
from freeze_v2 import pick as _v2_pick        # noqa: E402  the ONE rule


def pick(rows, floor):
    """Rule V2, imported from freeze_v2 so a figure can never show a different
    pick from the freeze that the run actually uses."""
    if not rows:
        return None
    bad = {(r["prop"], r["arm"], float(r["w"])) for r in rows
           if r.get("n_nonfinite", 0) > 0}
    r, _status = _v2_pick(rows, floor, bad)
    return r


def fig_strength(cells, prop):
    u = cells[(prop, "unguided", 1.0)]
    floor = FLOOR * u["mol_stability"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    edge = []
    for a in ARMS:
        rows = sorted([r for (p, aa, w), r in cells.items() if p == prop and aa == a],
                      key=lambda r: r["w"])
        if not rows:
            continue
        ws = [r["w"] for r in rows]
        a1.plot(ws, [r["in_band_fraction"] for r in rows], "o-", ms=3.5,
                color=COL[a], label=a, lw=1.4)
        a2.plot(ws, [r["mol_stability"] for r in rows], "o-", ms=3.5,
                color=COL[a], label=a, lw=1.4)
        ch = pick(rows, floor)
        if ch:
            a1.plot([ch["w"]], [ch["in_band_fraction"]], "*", ms=16,
                    color=COL[a], mec="k", mew=0.6, zorder=5)
            a2.plot([ch["w"]], [ch["mol_stability"]], "*", ms=16,
                    color=COL[a], mec="k", mew=0.6, zorder=5)
        if rows[-1]["mol_stability"] >= floor - 1e-12:
            edge.append(a)
    a1.axhline(u["in_band_fraction"], color="k", ls=":", lw=1.2, label="unguided")
    a2.axhline(u["mol_stability"], color="k", ls=":", lw=1.2, label="unguided")
    a2.axhline(floor, color="r", ls="--", lw=1.4,
               label="chemistry floor (0.9x)")
    for ax, t in ((a1, "in-band fraction"), (a2, "molecule stability")):
        ax.set_xscale("log")
        ax.set_xlabel("strength w  (log scale)")
        ax.set_ylabel(t)
        ax.legend(fontsize=7.5, ncol=2)
    a1.set_title("%s: what each strength buys  (star = rule's pick)" % NICE[prop])
    a2.set_title("%s: what each strength costs" % NICE[prop])
    fig.tight_layout()
    p = os.path.join(FIGS, "strength_%s.png" % prop)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p, edge, floor


def fig_pareto(cells, prop, floor):
    u = cells[(prop, "unguided", 1.0)]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for a in ARMS:
        rows = sorted([r for (p, aa, w), r in cells.items() if p == prop and aa == a],
                      key=lambda r: r["w"])
        if not rows:
            continue
        ax.plot([r["mol_stability"] for r in rows],
                [r["in_band_fraction"] for r in rows], "o-", ms=4,
                color=COL[a], label=a, lw=1.3, alpha=0.85)
    ax.plot([u["mol_stability"]], [u["in_band_fraction"]], "ks", ms=9,
            label="unguided", zorder=5)
    ax.axvline(floor, color="r", ls="--", lw=1.4, label="chemistry floor")
    ax.set_xlabel("molecule stability  (chemistry kept)")
    ax.set_ylabel("in-band fraction  (targeting)")
    ax.set_title("%s: the trade-off every arm is on (n=512 screen)" % NICE[prop])
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGS, "pareto_%s.png" % prop)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    os.makedirs(FIGS, exist_ok=True)
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    cells = screen_cells()
    print("target and band:")
    for p in PROPS:
        dl = delta_of(p)
        _, frac = fig_targets(d, p, dl)
        print("  %-6s target %-9.4g delta %-9.4g  QM9 in band %.3f" % (p, TARGET[p], dl, frac))
    print("\nstrength rule (best in_band clearing the floor):")
    for p in PROPS:
        _, edge, floor = fig_strength(cells, p)
        fig_pareto(cells, p, floor)
        u = cells[(p, "unguided", 1.0)]
        chosen = []
        for a in ARMS:
            rows = sorted([r for (pp, aa, w), r in cells.items() if pp == p and aa == a],
                          key=lambda r: r["w"])
            ch = pick(rows, floor)
            chosen.append("%s w=%g (inb %.3f)" % (a, ch["w"], ch["in_band_fraction"])
                          if ch else "%s NONE" % a)
        print("  %-6s floor %.3f | %s" % (p, floor, "; ".join(chosen)))
        if edge:
            print("         GRID-LIMITED (still clears at w=4, extend): %s" % ", ".join(edge))
    print("\nwrote %s" % FIGS)


if __name__ == "__main__":
    main()
