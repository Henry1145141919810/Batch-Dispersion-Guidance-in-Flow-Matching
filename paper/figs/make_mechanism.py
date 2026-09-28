"""Figure 2: how BDG's setpoint acts, and what the deviation gain does.

Three panels:
  (a) the sign mechanism, drawn (original schematic, no data)
  (b) requested setpoint -> achieved spread, measured, q50, three properties
  (c) the measured run-mean deviation gain w_eff, which passes through zero

Panels (b) and (c) are parsed from the real port cells, so the figure cannot
drift from the numbers in the paper:
    results/bdg_port/table.txt   (64 cells, n=256, seed 20260925, fm_ema.pt)

Run from the repository root:
    .venv/Scripts/python.exe paper/figs/make_mechanism.py
Writes paper/figs/mechanism.pdf (vector, for LaTeX) and a PNG preview.

Colour: categorical slots 1-3 of the validated default palette
(blue / orange / aqua).  Validated all-pairs in light mode: worst CVD dE 9.2,
worst normal-vision dE 24.0.  The aqua slot sits below 3:1 contrast on a light
surface, so every series carries a direct label as well as its hue, which is the
documented relief for that warning.
"""
from __future__ import annotations

import pathlib
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = pathlib.Path(__file__).resolve().parents[2]
TABLE = ROOT / "results" / "bdg_port" / "table.txt"
OUT = pathlib.Path(__file__).resolve().parent / "mechanism.pdf"

# --- palette ---------------------------------------------------------------
SERIES = {"mu": "#2a78d6", "gap": "#eb6834", "alpha": "#1baf7a"}
LABEL = {"mu": r"$\mu$", "gap": "gap", "alpha": r"$\alpha$"}
MARK = {"mu": "o", "gap": "s", "alpha": "^"}          # secondary encoding
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e5e1"
ACCENT = "#B3261E"
ETA = 4.0                                             # the gain these cells ran at


def parse(path: pathlib.Path):
    """Pull the q50 ladder and the measured run-mean w_eff for each property."""
    txt = path.read_text(encoding="utf-8")
    ladders, weffs = {}, {}
    for block in re.split(r"=== ", txt)[1:]:
        head = block.splitlines()[0]
        target = re.search(r"target (q\d+)", head)
        if not target or target.group(1) != "q50":
            continue
        prop = head.split()[0]
        lad = re.search(
            r"requested->achieved sd\(f_A\)/unguided: (.+?)\s+MONOTONE=(\w+)", block)
        if lad:
            pts = dict(
                (float(k[1:]), float(v))
                for k, v in (p.strip().split(":") for p in lad.group(1).split(","))
            )
            ladders[prop] = (pts, lad.group(2) == "True")
        we = re.findall(
            r"(e4t[\d.]+)\s+tau\s+[\d.]+\s+V_b/tau\^2\s+[\d.]+\s+e\s+[-+][\d.]+"
            r"\s+w_eff\s+([-+][\d.]+)", block)
        if we:
            weffs[prop] = {float(k.split("t")[1]): float(v) for k, v in we}
    return ladders, weffs


def panel_mechanism(ax):
    """(a) The three regimes, drawn.  Original schematic."""
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-0.35, 3.05)
    ax.axis("off")

    rows = [
        (2.45, [-1.02, -0.58, -0.2, 0.22, 0.6, 1.04], "in",
         r"$V_b>\tau^2\;\Rightarrow\;e>0$", "contract the batch"),
        (1.35, [-0.62, -0.36, -0.12, 0.14, 0.38, 0.64], None,
         r"$V_b=\tau^2\;\Rightarrow\;e=0$", "term vanishes: exactly plug-in"),
        (0.25, [-0.5, -0.29, -0.1, 0.11, 0.3, 0.52], "out",
         r"$V_b<\tau^2\;\Rightarrow\;e<0$", "expand the batch"),
    ]
    for y, xs, direction, cond, what in rows:
        ax.axvline(0, ymin=(y - 0.22 + 0.35) / 3.4, ymax=(y + 0.22 + 0.35) / 3.4,
                   color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.plot(xs, [y] * len(xs), MARK["mu"], ms=4.6, color=INK2,
                mec="white", mew=0.6, ls="none", zorder=3)
        if direction:
            for x in xs:
                d = -0.19 if direction == "in" else 0.19
                d = d if x > 0 else -d
                ax.add_patch(FancyArrowPatch(
                    (x, y - 0.3), (x + d, y - 0.3),
                    arrowstyle="-|>", mutation_scale=6.5,
                    lw=1.0, color=ACCENT, zorder=4))
        ax.text(-1.3, y + 0.42, cond, fontsize=6.6, color=INK, ha="left")
        ax.text(-1.3, y + 0.18, what, fontsize=6.3, color=INK2, ha="left")

    ax.text(0.07, 2.90, r"$\bar{F}$", fontsize=7, color=MUTED, ha="left")
    ax.text(0, -0.10,
            r"each sample moves along its own $g_i$,"
            "\n"
            r"in proportion to $F_i-\bar{F}$",
            fontsize=6.0, color=INK2, ha="center", va="top")


def panel_ladder(ax, ladders):
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(0.435, 1.012, "unguided spread", fontsize=6.0, color=MUTED,
            ha="left", va="bottom")
    for prop in ("mu", "gap", "alpha"):
        pts, mono = ladders[prop]
        xs = sorted(pts)
        ys = [pts[x] for x in xs]
        ax.plot(xs, ys, MARK[prop] + "-", color=SERIES[prop], lw=1.7, ms=4.4,
                mec="white", mew=0.7, zorder=3, label=LABEL[prop])
        dy = {"mu": 4, "gap": -7, "alpha": 0}[prop]
        ax.annotate(LABEL[prop], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(5, dy), fontsize=7, color=SERIES[prop], weight="bold")
    ax.set_xlabel(r"requested setpoint  $\tau_{\mathrm{mult}}$", fontsize=7.4)
    ax.set_ylabel("achieved spread / unguided", fontsize=7.4)
    ax.set_xlim(0.40, 1.72)
    ax.set_ylim(0.60, 1.30)
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.legend(fontsize=6.2, frameon=False, loc="lower right",
              handlelength=1.3, borderpad=0.1, labelspacing=0.18)
    ax.text(0.435, 1.285,
            "monotone on 3/3 properties;\nexpansion reached on all three",
            fontsize=6.0, color=INK2, va="top", ha="left")


def panel_weff(ax, weffs):
    ax.axhline(0.0, color=INK, lw=0.9, zorder=2)
    ax.text(1.69, 0.10, r"$w_{\mathrm{eff}}=0$", fontsize=6.2, color=INK,
            ha="right", va="bottom")
    bound = 1.0 - ETA
    ax.axhline(bound, color=MUTED, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.text(1.69, bound + 0.10, r"bound $w_{\mathrm{eff}}\geq 1-\eta$",
            fontsize=6.1, color=MUTED, va="bottom", ha="right")
    for prop in ("mu", "gap", "alpha"):
        d = weffs[prop]
        xs = sorted(d)
        ys = [d[x] for x in xs]
        ax.plot(xs, ys, MARK[prop] + "-", color=SERIES[prop], lw=1.7, ms=4.4,
                mec="white", mew=0.7, zorder=3)
        dy = {"mu": 6, "alpha": -3, "gap": -10}[prop]
        ax.annotate(LABEL[prop], (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(5, dy), fontsize=7, color=SERIES[prop], weight="bold")
    ax.set_xlabel(r"requested setpoint  $\tau_{\mathrm{mult}}$", fontsize=7.4)
    ax.set_ylabel(r"measured run-mean $w_{\mathrm{eff}}$", fontsize=7.4)
    ax.set_xlim(0.40, 1.72)
    ax.set_ylim(-3.35, 3.6)
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5])
    ax.text(0.435, -1.35,
            "the gain passes through zero;\n"
            "a single strength dial cannot,\n"
            "without reversing centring too",
            fontsize=6.0, color=INK2, va="top", ha="left")


def style(ax):
    ax.tick_params(labelsize=6.8, colors=INK2, length=2.4, width=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def main() -> int:
    if not TABLE.exists():
        sys.exit(f"missing {TABLE}")
    ladders, weffs = parse(TABLE)
    missing = {"mu", "gap", "alpha"} - set(ladders) - set()
    if missing or not weffs:
        sys.exit(f"parse failed; ladders={list(ladders)} weffs={list(weffs)}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelcolor": INK,
        "text.color": INK,
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.45))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.885, bottom=0.185, wspace=0.40)

    panel_mechanism(axes[0])
    panel_ladder(axes[1], ladders)
    panel_weff(axes[2], weffs)
    for ax in axes[1:]:
        style(ax)

    for ax, tag, title in zip(
            axes, "abc",
            ["the sign mechanism",
             "setpoint controls achieved spread",
             "the deviation gain changes sign"]):
        ax.set_title(f"({tag}) {title}", fontsize=7.6, color=INK,
                     loc="left", pad=5)

    fig.savefig(OUT, format="pdf")
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
    print("wrote", OUT.name, "and", OUT.with_suffix(".png").name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
