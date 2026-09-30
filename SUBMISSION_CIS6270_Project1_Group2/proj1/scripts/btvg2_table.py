"""Every BTVG-2 cell on disk, with its full metric block and the right comparator.

    python proj1/scripts/btvg2_table.py
    python proj1/scripts/btvg2_table.py --md-out docs/results/BTVG2_FULL_METRICS.md

WHAT IS ON DISK. Three BTVG-2 variants, all n = 2,048, seed 20261001, the
`dist` target, the same generator, window, clip, batch and MC settings, so
every cell pairs at the molecule level (same mol_idx, per-molecule target y
and sizes -- verified below; the initial noise is shared by construction, from
the seeded per-cell generator, not verified here):

  btvg2        lgd_mc's mean term plus BTVG's variance term, gated by
               plausibility exp(-(y - mu)^2 / 2V)            results/pilot_btvg2
  btvg2_band   the same with a different gate: a Gaussian of width 1.96 tau
               (the band half-width) around the target, so the variance term
               is weighted 0.61 at the band edge and 0.14 at twice it, rather
               than being centred on |y - mu| ~ sqrt V        results/pilot_btvg2
  btvg2_xproj  btvg2 with the invariants imposed in x_t instead of m-space
               (the 23 Sep geometry repair)                   results/xproj

plus `lgd_mc` at the matching strengths and `unguided` (results/pilot_chem).

WHAT `variant - lgd_mc` MEASURES, AND WHAT IT DOES NOT. The variants read the
same K draws as lgd_mc and their mean term is the same arithmetic, so at the
first guided step the mean term is bit-identical. The difference therefore
measures **adding the variance term as implemented** -- the incremental effect
of the implemented correction. It does NOT isolate the variance term, and the
repo already says so (guidance.py, the btvg2_xproj comment; status doc, the
btvg2 "What" paragraph): (1) after the first step the two trajectories are at
different states, so the mean terms are evaluated at different points; (2) the
velocity clip scales the COMBINED field, and the variants clip more often, so
their mean step is shrunk more often; (3) for btvg2 and btvg2_band the
pulled-back variance step itself moves the mean (|cos| 0.45-0.51 to the mean
direction). btvg2_xproj removes (3), to first order only.

THE COMPARATOR SHOULD COME FROM THE SAME GPU. The pilot directories ran on the
local RTX 5080; xproj ran on a Betty B200 (a MIG 2g.45gb slice). This script
MEASURES the cross-GPU difference instead of asserting it: unguided at w = 1
and lgd_mc at w = 4 from the B200 v1 full run (its first 2,048 rows pair
exactly with these cells -- verified), and lgd_mc at w = 8 and 16 from xproj.
Unguided agrees across machines; guided cells do not, already at w = 4, and a
few molecules diverge chaotically. Aggregates stay within noise. B200
run-to-run determinism is untested, so on that side hardware and
nondeterminism cannot be separated; a code change is ruled out (the lgd_mc path
is unchanged across the commits these cells came from).

WHAT IS MISSING. The btvg2 and btvg2_band cells predate decoded scoring: their
JSONs have no `_dec` fields and their sidecars hold neither f_B_dec nor
coordinates, and no other copy on disk or in the root tarballs does. They can
be judged on the continuous metric only -- the one v2's V4 names as
flattering. A decoded view needs a re-run.

EVERY CELL IS ONE SEED AT n = 2,048 (se of an in_band proportion ~0.007).
Pilots, not a verdict, and none is scored under the v2 rubric.

COST is reported as generator passes and guide passes separately. Summing the
harness counters would weight a guide forward the same as a generator VJP.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

NL = chr(10)
PROPS = ("mu", "alpha", "gap")
SIGMA = 3.0
SEED = 20261001
# GPU label -> directories, and the substring its prov.device must contain
SRC = {
    "5080": (("results/pilot_chem/seed20261001_n2048",
              "results/pilot_btvg2/seed20261001_n2048"), "RTX 5080"),
    "B200": (("results/xproj/seed20261001_n2048",), "B200"),
}
CELLS = (
    ("5080", "unguided", 1.0),
    ("5080", "lgd_mc", 4.0), ("5080", "btvg2", 4.0),
    ("5080", "lgd_mc", 8.0), ("5080", "btvg2", 8.0), ("5080", "btvg2_band", 8.0),
    ("B200", "lgd_mc", 8.0), ("B200", "btvg2_xproj", 8.0),
    ("5080", "lgd_mc", 16.0),
    ("B200", "lgd_mc", 16.0), ("B200", "btvg2_xproj", 16.0),
)
PAIRS = (
    ("5080", "btvg2", 4.0), ("5080", "btvg2", 8.0), ("5080", "btvg2_band", 8.0),
    ("B200", "btvg2_xproj", 8.0), ("B200", "btvg2_xproj", 16.0),
)
# cross-GPU pairs: (arm, w, where the B200 copy lives). The v1 full run is
# n = 5,000 on the B200; its first 2,048 rows are these molecules.
XGPU = (
    ("unguided", 1.0, "results/full/n5000/seed20261001/%s__unguided__dist__w1__tmin0.5__full.json"),
    ("lgd_mc", 4.0, "results/full/n5000/seed20261001/%s__lgd_mc__dist__w4__tmin0.5__full.json"),
    ("lgd_mc", 8.0, None),
    ("lgd_mc", 16.0, None),
)
GEN_KEYS = ("gen_fwd", "gen_vjp", "gen_jvp")
GUIDE_KEYS = ("guide_fwd", "guide_bwd", "guide_hvp")


def _load_side(path):
    import torch
    return torch.load(path, weights_only=False)


def load():
    """{(gpu, prop, arm, w): (row, sidecar_path)}. A cell found in two dirs on
    the same GPU must agree molecule by molecule; the copy carrying decoded
    scores is kept."""
    import torch
    out, problems = {}, []
    want = {(c[0], c[1], c[2]) for c in CELLS}
    for gpu, (dirs, dev) in SRC.items():
        for d in dirs:
            for fn in sorted(glob.glob(os.path.join(ROOT, d, "*__tgt.json"))):
                r = json.load(open(fn))
                if (gpu, r["arm"], float(r["w"])) not in want:
                    continue
                if dev not in (r.get("prov") or {}).get("device", ""):
                    problems.append("%s says device %r, but its directory is labelled %s"
                                    % (os.path.basename(fn), (r.get("prov") or {}).get("device"), gpu))
                side = fn[:-5] + ".permol.pt"
                side = side if os.path.exists(side) else None
                key = (gpu, r["prop"], r["arm"], float(r["w"]))
                if key in out and side and out[key][1]:
                    a, b = _load_side(out[key][1]), _load_side(side)
                    same = (torch.equal(a["f_B"], b["f_B"])
                            and torch.equal(a["mol_stable"], b["mol_stable"])
                            and a["smiles"] == b["smiles"])
                    if not same:
                        problems.append("%s %s/%s@w%g: two copies on the same GPU disagree"
                                        % key)
                        continue
                    if "f_B_dec" in b and "f_B_dec" not in a:
                        out[key] = (r, side)       # keep the richer copy
                    continue
                if key not in out:
                    out[key] = (r, side)
    return out, problems


def check(cells, problems):
    for p in PROPS:
        for gpu, a, w in CELLS:
            if (gpu, p, a, w) not in cells:
                problems.append("missing %s %s/%s@w%g" % (gpu, p, a, w))
            elif cells[(gpu, p, a, w)][1] is None:
                problems.append("no sidecar for %s %s/%s@w%g" % (gpu, p, a, w))
    rows = [r for (r, _) in cells.values()]
    for key in ("n", "seed", "steps", "solver", "t_min_guide", "clip", "batch",
                "target_name"):
        vals = {r.get(key) for r in rows}
        if len(vals) > 1:
            problems.append("mixed %s: %s" % (key, sorted(map(str, vals))))
    if {r.get("seed") for r in rows} != {SEED}:
        problems.append("not every cell is seed %d" % SEED)
    vals = {(r.get("prov") or {}).get("fm_md5") for r in rows}
    if len(vals) > 1:
        problems.append("mixed generator: %s" % sorted(map(str, vals)))
    for key in ("n_mc", "sigma_mc", "k_delta", "n_probe"):
        vals = {r.get(key) for r in rows if r["arm"] != "unguided"}
        if len(vals) > 1:
            problems.append("guided arms disagree on %s: %s" % (key, sorted(map(str, vals))))
    for p in PROPS:
        vals = {round(float(r["delta"]), 12) for r in rows if r["prop"] == p}
        if len(vals) > 1:
            problems.append("%s: mixed delta %s" % (p, sorted(vals)))
    return problems


def check_pairing(cells):
    import torch
    problems = []
    for p in PROPS:
        ref = None
        for gpu, a, w in CELLS:
            d = _load_side(cells[(gpu, p, a, w)][1])
            sig = (d["mol_idx"], d["y"], d["n_atoms"])
            if ref is None:
                ref = sig
            elif not all(torch.equal(x, y) for x, y in zip(sig, ref)):
                problems.append("%s: %s %s@w%g does not pair" % (p, gpu, a, w))
    return problems


def block(r, d, delta):
    e = (d["f_B"] - d["y"])[d["finite"]].double()
    ea = (d["f_A"] - d["y"])[d["finite"]].double()
    guided = int(round(r["steps"] * (1.0 - r["t_min_guide"])))
    denom = r["n"] * guided
    cost = r.get("cost") or {}
    return {
        "in_band": r["in_band_fraction"],
        "in_band_dec": r.get("in_band_fraction_dec"),
        "mae": r["prop_mae_eval"] / delta,
        "mae_dec": (r["prop_mae_eval_dec"] / delta
                    if r.get("prop_mae_eval_dec") is not None else None),
        "bias": float(e.mean()) / delta,
        "bias_A": float(ea.mean()) / delta,
        "resid": float(e.std(unbiased=False)) / delta,
        "rmse": r["prop_rmse_eval"] / delta,
        "mol_stab": r["mol_stability"], "atom_stab": r["atom_stability"],
        "valid": r["validity"], "uniq": r["uniqueness_of_valid"],
        "yield": r["unique_valid_per_sample"],
        "div": r["diversity_mean_pairwise"],
        "gap": r["guide_eval_gap_mean"] / delta,
        "clip": (r.get("clipped_sample_steps", 0) / denom) if denom else float("nan"),
        "gen": sum(cost.get(k, 0) for k in GEN_KEYS),
        "guide": sum(cost.get(k, 0) for k in GUIDE_KEYS),
        "nonfinite": r["n_nonfinite"],
        "diag": r.get("diag") or {},
    }


def paired(a, b, delta, what, dec=False, n=None):
    """(a - b, paired z) over the same molecules; n truncates both."""
    key = "f_B_dec" if dec else "f_B"
    if key not in a or key not in b:
        return None, None
    s = slice(0, n)
    if what == "ib":
        xa = (((a[key][s] - a["y"][s]).abs() <= delta) & a["finite"][s]).double()
        xb = (((b[key][s] - b["y"][s]).abs() <= delta) & b["finite"][s]).double()
        diff = xa - xb
    elif what == "mae":
        ok = a["finite"][s] & b["finite"][s]
        diff = ((a[key][s] - a["y"][s]).abs() - (b[key][s] - b["y"][s]).abs())[ok].double() / delta
    else:
        diff = a["mol_stable"][s].double() - b["mol_stable"][s].double()
    se = diff.std(unbiased=True).item() / math.sqrt(diff.numel())
    m = diff.mean().item()
    return m, (m / se if se > 0 else float("nan"))


def f(v, fmt="%.4f"):
    return "n/a" if v is None or v != v else fmt % v


def zf(v):
    return "n/a" if v is None or v != v else ("**%+.2f**" % v if abs(v) >= SIGMA else "%+.2f" % v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    cells, problems = load()
    problems = check(cells, problems)
    if problems:
        print("REFUSING: the BTVG-2 set is not complete or not consistent.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)
    import torch
    problems = check_pairing(cells)
    if problems:
        print("REFUSING: the cells do not pair.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)

    S = lambda g, p, a, w: _load_side(cells[(g, p, a, w)][1])
    DELTA = {p: cells[("5080", p, "unguided", 1.0)][0]["delta"] for p in PROPS}
    any_r = cells[("5080", PROPS[0], "unguided", 1.0)][0]
    prov = any_r.get("prov") or {}

    B = {}
    for p in PROPS:
        for gpu, a, w in CELLS:
            B[(gpu, p, a, w)] = block(cells[(gpu, p, a, w)][0], S(gpu, p, a, w), DELTA[p])
    U = B[("5080", PROPS[0], "unguided", 1.0)]

    L = []
    L.append("# BTVG-2: every cell, the full metric block, and the right comparator")
    L.append("")
    L.append("Generated by `proj1/scripts/btvg2_table.py`. Do not hand-edit; re-run "
             "the script.")
    L.append("")
    L.append("**Exploratory pilots, one seed, n = 2,048, `dist` target.** Not part of "
             "the v2 full run and not scored under its rubric. The se of an in_band "
             "proportion here is ~0.007.")
    L.append("")
    L.append("| variant | what it changes | cells from |")
    L.append("|---|---|---|")
    L.append("| `btvg2` | `lgd_mc`'s mean term (same draws, same arithmetic) **plus** "
             "BTVG's variance term, gated by plausibility exp(-(y - mu)^2 / 2V) | "
             "`results/pilot_btvg2`, RTX 5080 |")
    L.append("| `btvg2_band` | as `btvg2` with a different gate: a Gaussian of width "
             "1.96 tau around the target, weight 0.61 at the band edge and 0.14 at "
             "twice it | `results/pilot_btvg2`, RTX 5080 |")
    L.append("| `btvg2_xproj` | as `btvg2`, with the invariants imposed in x_t (the "
             "23 Sep geometry repair) | `results/xproj`, B200 (MIG 2g.45gb) |")
    L.append("| `lgd_mc`, `unguided` | comparators | the same dirs plus "
             "`results/pilot_chem` (5080) |")
    L.append("")
    L.append("| provenance | value |")
    L.append("|---|---|")
    L.append("| n, seed | %d, %d (every cell) |" % (any_r["n"], any_r["seed"]))
    L.append("| generator | `%s` md5 `%s` |" % (any_r.get("fm", "?"),
                                              (prov.get("fm_md5") or "?")[:8]))
    L.append("| sampler | %d-step %s, window t >= %g, clip %g, batch %d |"
             % (any_r["steps"], any_r["solver"], any_r["t_min_guide"], any_r["clip"],
                any_r["batch"]))
    g0 = cells[("5080", PROPS[0], "lgd_mc", 4.0)][0]
    L.append("| MC draws | n_mc = %s, sigma_mc = %s, k_delta = %s for every guided cell |"
             % (g0.get("n_mc"), g0.get("sigma_mc"), g0.get("k_delta")))
    L.append("| pairing | mol_idx, per-molecule y and sizes identical across every "
             "cell (verified); initial noise shared by construction |")
    L.append("| device check | every cell's `prov.device` matches its GPU label |")
    L.append("")
    L.append("**What `variant - lgd_mc` measures.** The variants share `lgd_mc`'s "
             "draws and mean-term arithmetic, so the difference measures **adding the "
             "variance term as implemented**, including its effect on the clipped "
             "mean step -- not the variance term in isolation. After the first step "
             "the trajectories differ; the clip scales the combined field and the "
             "variants clip more often; and for `btvg2`/`btvg2_band` the pulled-back "
             "variance step also moves the mean. `btvg2_xproj` removes the last of "
             "these to first order. This is the wording the repo already uses "
             "(status doc, the `btvg2` \"What\" paragraph).")
    L.append("")
    L.append("**`btvg2` and `btvg2_band` have no decoded view.** Their JSONs have no "
             "`_dec` fields and their sidecars hold no coordinates; no other copy on "
             "disk or in the root tarballs does. They are judged on the continuous "
             "metric only, which v2's V4 names as flattering. A decoded view needs "
             "a re-run.")
    L.append("")

    # ---- full block -------------------------------------------------------
    L.append("## Full metric block")
    L.append("")
    L.append("Error columns in units of delta (the band half-width). bias = "
             "mean(f_B - y) and residual sd over finite rows; `bias (f_A)` is the "
             "same with the guide's own prediction, so a bias that f_A shares is "
             "the guide's view, not a guide-evaluator disagreement. On the `dist` "
             "target, unguided's bias is the generator's own offset (mu %+.2f, "
             "alpha %+.2f, gap %+.2f); guidance changes it."
             % tuple(B[("5080", p, "unguided", 1.0)]["bias"] for p in PROPS))
    L.append("")
    for p in PROPS:
        L.append("### %s (delta %.5f)" % (p, DELTA[p]))
        L.append("")
        L.append("| GPU | arm | w | in_band cont / dec | MAE/d cont / dec | bias/d | "
                 "bias (f_A)/d | resid sd/d | RMSE/d | mol_stab | atom_stab | valid | "
                 "uniq valid | useful yield |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for gpu, a, w in CELLS:
            b = B[(gpu, p, a, w)]
            L.append("| %s | `%s` | %g | %s / %s | %s / %s | %+.3f | %+.3f | %.3f | %.3f | "
                     "%.4f | %.4f | %.4f | %.4f | %.4f |"
                     % (gpu, a, w, f(b["in_band"]), f(b["in_band_dec"]),
                        f(b["mae"], "%.3f"), f(b["mae_dec"], "%.3f"), b["bias"],
                        b["bias_A"], b["resid"], b["rmse"], b["mol_stab"],
                        b["atom_stab"], b["valid"], b["uniq"], b["yield"]))
        L.append("")
        L.append("| GPU | arm | w | diversity | guide-eval gap/d | clipped guided "
                 "steps | generator passes vs unguided | guide passes | var share | "
                 "gate | capped | non-finite |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for gpu, a, w in CELLS:
            b = B[(gpu, p, a, w)]
            dg = b["diag"]
            L.append("| %s | `%s` | %g | %.4f | %.3f | %.1f %% | %.1fx | %d | %s | %s | %s | %d |"
                     % (gpu, a, w, b["div"], b["gap"], 100 * b["clip"],
                        b["gen"] / U["gen"] if U["gen"] else float("nan"), b["guide"],
                        f(dg.get("btvg2_var_share"), "%.3f"),
                        f(dg.get("btvg2_gate"), "%.3f"),
                        f(dg.get("btvg2_capped"), "%.3f"), b["nonfinite"]))
        L.append("")

    # ---- cross-GPU --------------------------------------------------------
    L.append("## How much changing GPU moves a cell (measured)")
    L.append("")
    L.append("The same configuration on the RTX 5080 and on the B200, same molecules, "
             "targets and seed. The w = 1 and w = 4 B200 copies are the first 2,048 "
             "rows of the v1 full-run cells, which pair exactly with these (checked "
             "row by row). `flips` count molecules whose in-band or stability verdict "
             "differs; `SMILES` counts molecules decoded to a different graph.")
    L.append("")
    L.append("| prop | arm | w | in-band flips | stab flips | SMILES differ | "
             "off by > 0.1 d | max abs diff / d | f_B corr | d(in_band) | z | "
             "d(mol_stab) | z |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    xrows = []
    for p in PROPS:
        dl = DELTA[p]
        for arm, w, v1 in XGPU:
            loc = S("5080", p, arm, w)
            n = loc["f_B"].numel()
            if v1:
                fn = os.path.join(ROOT, v1 % p)
                rj = json.load(open(fn))
                if "B200" not in (rj.get("prov") or {}).get("device", ""):
                    raise SystemExit("%s is not a B200 cell" % fn)
                far = _load_side(fn[:-5] + ".permol.pt")
                if not all(torch.equal(far[k][:n], loc[k]) for k in ("mol_idx", "y", "n_atoms")):
                    raise SystemExit("%s: the v1 rows do not pair with the pilot" % fn)
            else:
                far = S("B200", p, arm, w)
            s = slice(0, n)
            ib_f = ((far["f_B"][s] - far["y"][s]).abs() <= dl) & far["finite"][s]
            ib_l = ((loc["f_B"] - loc["y"]).abs() <= dl) & loc["finite"]
            diff = (far["f_B"][s] - loc["f_B"]).abs()
            smi = sum(1 for x, y in zip(far["smiles"][:n], loc["smiles"]) if x != y)
            corr = float(torch.corrcoef(torch.stack([far["f_B"][s], loc["f_B"]]))[0, 1])
            dib, zib = paired(far, loc, dl, "ib", n=n)
            dms, zms = paired(far, loc, dl, "stab", n=n)
            xrows.append((p, arm, w, int((ib_f != ib_l).sum()), zib, zms, corr))
            L.append("| %s | `%s` | %g | %d | %d | %d | %d | %.3f | %.5f | %+.4f | %s | %+.4f | %s |"
                     % (p, arm, w, int((ib_f != ib_l).sum()),
                        int((far["mol_stable"][s] != loc["mol_stable"]).sum()), smi,
                        int((diff > 0.1 * dl).sum()), float(diff.max()) / dl, corr,
                        dib, zf(zib), dms, zf(zms)))
    L.append("")
    ung = [r for r in xrows if r[1] == "unguided"]
    gd = [r for r in xrows if r[1] != "unguided"]
    zmax = max(max(abs(r[4]) for r in gd if r[4] == r[4]), max(abs(r[5]) for r in gd if r[5] == r[5]))
    # what choosing the other GPU's comparator would have done to xproj's tests
    dz = []
    for p in PROPS:
        for w in (8.0, 16.0):
            x = S("B200", p, "btvg2_xproj", w)
            for what in ("ib", "stab"):
                _, z_same = paired(x, S("B200", p, "lgd_mc", w), DELTA[p], what)
                _, z_other = paired(x, S("5080", p, "lgd_mc", w), DELTA[p], what)
                dz.append((abs(z_same - z_other), p, w, what, z_same, z_other))
    worst = max(dz)
    crossed = sum(1 for d in dz if (abs(d[4]) >= SIGMA) != (abs(d[5]) >= SIGMA))
    L.append("**Unguided agrees across machines** (%d in-band flips over three "
             "properties). **Guided cells do not, already at w = 4:** individual "
             "molecules diverge, a few of them chaotically, and on mu and gap the "
             "divergence grows with strength (on alpha it stays small). **Aggregates "
             "stay within noise:** the largest cross-GPU |z| on in-band or stability "
             "is %.2f. So pooling machines is safe for aggregate metrics *of these "
             "cells*; a paired, molecule-level comparison of a guided arm should use a "
             "comparator from the same GPU. How much that matters, measured: pairing "
             "`btvg2_xproj` with the 5080 `lgd_mc` instead of the B200 one moves a "
             "paired z by at most %.2f (%s w = %g, %s: %+.2f -> %+.2f), and %d of "
             "%d tests change which side of %g sigma they fall on. This is one arm "
             "at four strengths, not a "
             "general statement about every arm. B200 run-to-run determinism is "
             "untested, so on that side hardware and nondeterminism are not "
             "separated."
             % (sum(r[3] for r in ung), zmax, worst[0], worst[1], worst[2],
                "in-band" if worst[3] == "ib" else "stability", worst[4], worst[5],
                crossed, len(dz), SIGMA))
    L.append("")

    # ---- comparisons --------------------------------------------------------
    L.append("## Each variant against `lgd_mc` at the same w, same GPU")
    L.append("")
    L.append("Paired over the same molecules. Each row measures **adding the variance "
             "term as implemented** (see above for why that is not the variance term "
             "in isolation). Positive = the variant is higher; for MAE, negative is "
             "better. The last two columns split the MAE change into its parts: "
             "change in |bias| and change in residual sd (descriptive, not tested). "
             "Bold = |z| >= %g." % SIGMA)
    L.append("")
    L.append("| prop | variant | w | GPU | d(in_band) cont | z | d(in_band) dec | z | "
             "d(MAE/d) cont | z | d(MAE/d) dec | z | d(mol_stab) | z | d abs(bias)/d | "
             "d resid sd/d |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    T = []
    for p in PROPS:
        dl = DELTA[p]
        for gpu, a, w in PAIRS:
            x, y = S(gpu, p, a, w), S(gpu, p, "lgd_mc", w)
            bx, by = B[(gpu, p, a, w)], B[(gpu, p, "lgd_mc", w)]
            dib, zib = paired(x, y, dl, "ib")
            did, zid = paired(x, y, dl, "ib", dec=True)
            dm, zm = paired(x, y, dl, "mae")
            dmd, zmd = paired(x, y, dl, "mae", dec=True)
            ds, zs = paired(x, y, dl, "stab")
            db = abs(bx["bias"]) - abs(by["bias"])
            dr = bx["resid"] - by["resid"]
            T.append(dict(p=p, a=a, w=w, dib=dib, zib=zib, did=did, zid=zid, dm=dm,
                          zm=zm, ds=ds, zs=zs, db=db, dr=dr))
            L.append("| %s | `%s` | %g | %s | %+.4f | %s | %s | %s | %+.3f | %s | %s | %s | "
                     "%+.4f | %s | %+.3f | %+.3f |"
                     % (p, a, w, gpu, dib, zf(zib), f(did, "%+.4f"), zf(zid), dm, zf(zm),
                        f(dmd, "%+.3f"), zf(zmd), ds, zf(zs), db, dr))
    L.append("")
    n_sig = sum(1 for t in T if abs(t["zib"]) >= SIGMA or abs(t["zs"]) >= SIGMA
                or (t["zid"] is not None and abs(t["zid"]) >= SIGMA))
    neg_c = sum(1 for t in T if t["dib"] < 0)
    pos_d = [t for t in T if t["did"] is not None and t["did"] > 0]
    n_d = sum(1 for t in T if t["did"] is not None)
    mae_lo = [t for t in T if t["p"] in ("mu", "alpha") and t["dm"] < 0]
    mae_hi = [t for t in T if t["p"] == "gap" and t["dm"] > 0]
    st_up = sum(1 for t in T if t["ds"] > 0)
    # the alpha rows where MAE falls by a visible amount (>= 0.1 delta)
    alpha_bias = [t for t in T if t["p"] == "alpha" and t["dm"] <= -0.1]
    alpha_rest = [t for t in T if t["p"] == "alpha" and t["dm"] > -0.1]
    L.append("**Reading, computed from the table above.**")
    L.append("")
    L.append("- **No comparison reaches |z| >= %g** on in-band (either metric) or "
             "stability, in either direction: %d of %d." % (SIGMA, n_sig, len(T)))
    L.append("- **In-band leans negative on the continuous metric** (%d of %d point "
             "estimates below `lgd_mc`), **but not on the decoded one** (%d of %d "
             "xproj comparisons above it, largest mu w = 16, %+.4f, z = %+.2f). The "
             "comparisons share molecules and comparators, so neither count is a "
             "test. The decoded metric is the one V4 requires, and it exists only for "
             "xproj."
             % (neg_c, len(T), len(pos_d), n_d,
                max(t["did"] for t in pos_d) if pos_d else float("nan"),
                max(pos_d, key=lambda t: t["did"])["zid"] if pos_d else float("nan")))
    L.append("- **MAE falls on mu and alpha in %d of 10 comparisons and rises on gap "
             "in %d of 5**, none at %g sigma. On alpha, in the %d rows where MAE "
             "falls by at least 0.1 d (%s), the gain is **smaller bias, not tighter "
             "spread**: |bias| drops by %.2f-%.2f d while residual sd moves by %+.2f to "
             "%+.2f d. In the other %d alpha rows (%s) MAE barely moves (%+.3f to "
             "%+.3f d). The variants shrink `lgd_mc`'s alpha undershoot; they do not "
             "concentrate the output."
             % (len(mae_lo), len(mae_hi), SIGMA, len(alpha_bias),
                ", ".join("`%s` w = %g" % (t["a"], t["w"]) for t in alpha_bias),
                min(-t["db"] for t in alpha_bias), max(-t["db"] for t in alpha_bias),
                min(t["dr"] for t in alpha_bias), max(t["dr"] for t in alpha_bias),
                len(alpha_rest),
                ", ".join("`%s` w = %g" % (t["a"], t["w"]) for t in alpha_rest),
                min(t["dm"] for t in alpha_rest), max(t["dm"] for t in alpha_rest)))
    L.append("- **Stability is a wash:** %d up, %d down." % (st_up, len(T) - st_up))
    vs = {(t["a"], t["w"], t["p"]): t for t in T}
    band = [B[("5080", p, "btvg2_band", 8.0)]["diag"].get("btvg2_var_share", 0) for p in PROPS]
    plain = [B[g][
        "diag"].get("btvg2_var_share", 0) for g in B if g[2] in ("btvg2", "btvg2_xproj")]
    L.append("- **`btvg2_band` engages less, not negligibly:** its variance share is "
             "%.1f-%.1f %% against %.0f-%.0f %% for the other variants, yet on gap it "
             "is the second-worst row (in-band %+.4f, z = %+.2f; MAE %+.3f d, "
             "z = %+.2f)."
             % (100 * min(band), 100 * max(band), 100 * min(plain), 100 * max(plain),
                vs[("btvg2_band", 8.0, "gap")]["dib"], vs[("btvg2_band", 8.0, "gap")]["zib"],
                vs[("btvg2_band", 8.0, "gap")]["dm"], vs[("btvg2_band", 8.0, "gap")]["zm"]))
    L.append("")

    # ---- guidance's own alpha bias ------------------------------------------
    lg = [(w, B[(g, "alpha", "lgd_mc", w)]) for g, a, w in CELLS if a == "lgd_mc" and g == "5080"]
    ua = B[("5080", "alpha", "unguided", 1.0)]
    L.append("**Guidance itself creates a growing bias on alpha.** Unguided's alpha "
             "bias is %+.2f d; `lgd_mc` (5080) goes %s as w rises. The guide's own "
             "prediction shows the same (`bias (f_A)`: %s), so this is the guide's "
             "view of the molecules, not a guide-evaluator disagreement. The cause is "
             "not established here."
             % (ua["bias"], ", ".join("%+.2f at w = %g" % (b["bias"], w) for w, b in lg),
                ", ".join("%+.2f" % b["bias_A"] for w, b in lg)))
    L.append("")

    # ---- vs unguided --------------------------------------------------------
    L.append("## Each cell against `unguided`")
    L.append("")
    L.append("Does guidance of any kind help at these strengths? `unguided` ran on the "
             "5080; the B200 rows cross machines, which the measurement above shows "
             "is harmless for unguided (no flips).")
    L.append("")
    L.append("| prop | arm | w | GPU | d(in_band) cont | z | d(in_band) dec | z | "
             "d(mol_stab) | z |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    VU = []
    for p in PROPS:
        dl = DELTA[p]
        u = S("5080", p, "unguided", 1.0)
        for gpu, a, w in CELLS:
            if a == "unguided":
                continue
            x = S(gpu, p, a, w)
            dib, zib = paired(x, u, dl, "ib")
            did, zid = paired(x, u, dl, "ib", dec=True)
            ds, zs = paired(x, u, dl, "stab")
            VU.append((p, a, w, gpu, dib, zib, zs))
            L.append("| %s | `%s` | %g | %s | %+.4f | %s | %s | %s | %+.4f | %s |"
                     % (p, a, w, gpu, dib, zf(zib), f(did, "%+.4f"), zf(zid), ds, zf(zs)))
    L.append("")
    clear = [v for v in VU if v[5] >= SIGMA]
    miss = [v for v in VU if v[5] < SIGMA]
    stab3 = [v for v in VU if v[6] <= -SIGMA]
    L.append("**%d of %d guided cells beat unguided on continuous in-band at z >= %g** "
             "(gain %+.4f to %+.4f, z up to %+.2f)%s. **The stability cost reaches "
             "%g sigma in %d cells:** %s."
             % (len(clear), len(VU), SIGMA, min(v[4] for v in VU), max(v[4] for v in VU),
                max(v[5] for v in VU),
                ("; the exception is " + ", ".join("%s `%s` w = %g (z = %+.2f)"
                                                   % (v[0], v[1], v[2], v[5]) for v in miss))
                if miss else "",
                SIGMA, len(stab3),
                ", ".join("%s `%s` w = %g %s (z = %+.2f)" % (v[0], v[1], v[2], v[3], v[6])
                          for v in stab3)))
    L.append("")

    out = NL.join(L) + NL
    print(out)
    if args.md_out:
        dst = args.md_out if os.path.isabs(args.md_out) else os.path.join(ROOT, args.md_out)
        open(dst, "w", encoding="utf-8").write(out)
        print("wrote %s" % dst, file=sys.stderr)


if __name__ == "__main__":
    main()
