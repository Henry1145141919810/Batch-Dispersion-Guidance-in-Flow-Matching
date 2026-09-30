"""Where the v2 full run's molecules sit relative to the fixed target.

    python proj1/scripts/v2_bias_figures.py

Writes docs/results/figs/bias_<prop>.png, one per property. It shows what the
`bias/d` and `resid sd/d` columns of FULL_RUN_V2_RESULTS.md mean on the
property axis itself:

  top     the distribution of f_B over all 15,000 molecules (3 seeds x 5,000)
          for `unguided` and for the guided arm that moves the mean furthest,
          against QM9 train_a (where the q90 target comes from), the target
          and its +-delta band. The arrow is unguided's bias.
  bottom  every arm's mean (dot) and +-1 sd (whisker) on the same axis, with
          bias/delta, sd/delta and in-band written beside it.

bias = mean(f_B - target), so bias/d is that mean in units of delta; with one
fixed target, resid sd is simply the sd of f_B across molecules.

Sources: results/full/v2/n5000/seed*/<prop>__<arm>__q90__w*__full.{json,permol.pt}
(continuous f_B, the table's non-decoded columns) and data/qm9.pt.
"""
from __future__ import annotations

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(ROOT, "results", "full", "v2", "n5000")
FIGS = os.path.join(ROOT, "docs", "results", "figs")
PROPS = ("mu", "alpha", "gap")
ARMS = ("unguided", "plug", "tmpd", "tfg", "lgd_mc")
UNIT = {"mu": "D", "alpha": "Bohr$^3$", "gap": "Ha"}
LABEL = {"mu": r"$\mu$  (Debye)", "alpha": r"$\alpha$  (Bohr$^3$)",
         "gap": r"$\Delta\varepsilon$  (Hartree)"}
FLOOR = 0.9

# chart chrome and the two series (validated: dataviz validate_palette, light)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
REF_FILL, GRID = "#e1e0d9", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(prop, arm):
    """Pool one (prop, arm) over the three seeds."""
    fs = sorted(glob.glob(os.path.join(RUN, "seed*", "%s__%s__q90__w*__tmin0.5__full.permol.pt"
                                       % (prop, arm))))
    if len(fs) != 3:
        raise SystemExit("%s/%s: expected 3 seeds, found %d" % (prop, arm, len(fs)))
    pm = [torch.load(f, weights_only=False) for f in fs]
    js = [json.load(open(f.replace(".permol.pt", ".json"))) for f in fs]
    ws = {re.search(r"__w([0-9.]+)__", f).group(1) for f in fs}
    fB = torch.cat([p["f_B"].double() for p in pm]).numpy()
    y = torch.cat([p["y"].double() for p in pm]).numpy()
    ok = torch.cat([p["finite"].bool() for p in pm]).numpy()
    dl = js[0]["delta"]
    r = (fB - y)[ok]
    return {"fB": fB[ok], "target": js[0]["target"], "delta": dl, "w": ws.pop(),
            "bias": r.mean(), "sd": r.std(ddof=1),
            "in_band": float((np.abs(r) <= dl).mean()),
            "stab": float(torch.cat([p["mol_stable"].double() for p in pm]).mean())}


def figure(prop, qm9):
    C = {a: load(prop, a) for a in ARMS}
    u = C["unguided"]
    t, dl = u["target"], u["delta"]
    floor = FLOOR * u["stab"]
    guided = [a for a in ARMS if a != "unguided"]
    star = min(guided, key=lambda a: abs(C[a]["bias"]))     # moves the mean furthest
    below = C[star]["stab"] < floor

    lo, hi = np.percentile(np.concatenate([qm9, u["fB"]]), [0.5, 99.5])
    hi = max(hi, t + 3 * dl)
    bins = np.linspace(lo, hi, 111)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.4, 7.0), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2.2], "hspace": 0.08})
    for ax in (a1, a2):
        ax.axvspan(t - dl, t + dl, color=INK, alpha=0.13, lw=0, zorder=0)
        ax.axvline(t, color=INK, lw=1.4, zorder=1)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=INK2, labelsize=9)

    # ---- top: the distributions
    a1.hist(qm9, bins=bins, density=True, color=REF_FILL, lw=0, zorder=0.5,
            label="QM9 train_a, real molecules\n(the target is its 90th percentile)")
    a1.hist(u["fB"], bins=bins, density=True, histtype="step", color=BLUE, lw=2,
            label="unguided, 15,000 generated")
    a1.hist(C[star]["fB"], bins=bins, density=True, histtype="step", color=ORANGE, lw=2,
            label="%s w=%s, 15,000 generated\n(the arm that moves the mean most%s)" % (
                star, C[star]["w"], ";\nbelow the chemistry floor" if below else ""))
    top = a1.get_ylim()[1]
    a1.set_ylim(0, top * 1.28)
    ya = top * 1.08
    a1.annotate("", xy=(t, ya), xytext=(u["bias"] + t, ya),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    a1.plot([u["bias"] + t], [ya], "o", color=BLUE, ms=6, mec="#fcfcfb", mew=1.5, zorder=5)
    a1.text(t - 1.5 * dl, ya + top * 0.035,
            "unguided bias = %+.3g %s = %+.1f $\\delta$" % (u["bias"], UNIT[prop], u["bias"] / dl),
            ha="right", va="bottom", fontsize=9.5, color=INK)
    a1.text(t + dl * 1.6, top * 0.62,
            "target %.4g %s\nband $\\pm\\delta$ = %.3g %s\n%.1f%% of unguided\nmolecules land in it"
            % (t, UNIT[prop], dl, UNIT[prop], 100 * u["in_band"]),
            ha="left", va="center", fontsize=9, color=INK, zorder=6,
            bbox=dict(fc="#fcfcfb", ec="none", alpha=0.9, pad=2))
    a1.set_ylabel("density", color=INK2, fontsize=9.5)
    a1.legend(loc="upper left", fontsize=8.5, frameon=False, bbox_to_anchor=(1.01, 1.0),
              labelspacing=1.0)
    a1.set_title("%s: every arm's molecules sit well below the fixed target "
                 "(v2 full run, 3 seeds x 5,000)" % prop, fontsize=11, color=INK, loc="left")

    # ---- bottom: mean +- 1 sd per arm
    rows = [("QM9 train_a", float(np.mean(qm9)), float(np.std(qm9, ddof=1)), MUTED, None)]
    rows += [("%s  w=%s" % (a, C[a]["w"]) if a != "unguided" else "unguided",
              C[a]["bias"] + t, C[a]["sd"],
              BLUE if a == "unguided" else ORANGE if a == star else INK2, a) for a in ARMS]
    for i, (lab, m, s, col, a) in enumerate(rows):
        yv = len(rows) - 1 - i
        a2.plot([m - s, m + s], [yv, yv], color=col, lw=2, solid_capstyle="round")
        a2.plot([m], [yv], "o", color=col, ms=8, mec="#fcfcfb", mew=1.5, zorder=5)
        if a is not None:
            c = C[a]
            note = "bias %+.1f$\\delta$  sd %.1f$\\delta$  in band %.1f%%%s" % (
                c["bias"] / dl, c["sd"] / dl, 100 * c["in_band"],
                "\nbelow the chemistry floor" if c["stab"] < floor else "")
        else:
            note = "mean %+.1f$\\delta$ from target  sd %.1f$\\delta$" % ((m - t) / dl, s / dl)
        a2.text(1.01, yv, note, transform=a2.get_yaxis_transform(), va="center",
                fontsize=8.5, color=INK)
    a2.set_yticks(range(len(rows)))
    a2.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=INK)
    a2.set_ylim(-0.6, len(rows) - 0.4)
    a2.set_xlim(lo, hi)
    a2.set_xlabel(LABEL[prop] + "   (dot = mean, whisker = $\\pm$1 sd; shaded = the band)",
                  color=INK2, fontsize=9.5)
    a2.grid(axis="x", color=GRID, lw=0.6)
    a2.set_axisbelow(True)

    fig.subplots_adjust(left=0.12, right=0.66, top=0.94, bottom=0.09)
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "bias_%s.png" % prop)
    fig.savefig(out, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    return out


def main():
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    for p in PROPS:
        qm9 = d["y"][d["split"]["train_a"], d["props"].index(p)].double().numpy()
        print("wrote", figure(p, qm9))


if __name__ == "__main__":
    main()
