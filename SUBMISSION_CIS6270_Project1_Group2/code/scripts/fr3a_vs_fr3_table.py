"""What the chemistry floor cost: FR3a against FR3, every arm, every metric.

    python proj1/scripts/fr3a_vs_fr3_table.py > docs/results/FR3A_VS_FR3_COMPARISON.md

FR3  (as registered) freezes each arm at its best-MAE compare-stage strength.
FR3a (amended 23 Sep) freezes it at the best-MAE strength among those clearing
the chemistry floor, mol_stability >= 0.9 x unguided.

Both sets ran at full scale. FR3a has three seeds; FR3 has ONE (20261001), so
every comparison here is restricted to seed 20261001, where both sets are
complete for all 7 arms x 3 properties. That makes the contrast PAIRED: same
seed, same targets, same generator, same evaluator, same delta -- the only
thing that differs is the strength. The three-seed FR3a numbers are in
FULL_RUN_RESULTS.md section 1 and are NOT mixed in here.

Sources: results/full/n5000/seed20261001/*.json (cells),
         results/full/n5000/frozen_q90_tfg.json (which strength each set froze).
The tfg entries are post hoc: tfg replaced dflow on 23 Sep after the run was
read, frozen by the same FR3a rule on its own compare cells.
"""
from __future__ import annotations

import glob
import math
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FULL = os.path.join(ROOT, "results", "full", "n5000")
SEED = 20261001
FLOOR = 0.9
ARMS = ["unguided", "plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var"]
HA_MEV = 27211.386245988   # dist_metrics.HARTREE_TO_MEV; cells hold gap in Hartree
PROPS = [("mu", "D", 1.0), ("alpha", "Bohr^3", 1.0), ("gap", "meV", HA_MEV)]
# guided sample-steps per cell: n x steps x (1 - t_min_guide)
GUIDED_STEPS = 5000 * 100 * 0.5


def load_cells():
    out = {}
    for f in glob.glob(os.path.join(FULL, "seed%d" % SEED, "*.json")):
        d = json.load(open(f))
        out[(d["prop"], d["arm"], float(d["w"]))] = d
    return out


def get(cells, prop, arm, w):
    r = cells.get((prop, arm, float(w)))
    if r is None:
        raise SystemExit("missing cell: %s/%s/w%g at seed %d" % (prop, arm, w, SEED))
    return r


def pct(x):
    return "%.0f%%" % (100.0 * x)


def compare_curve():
    """The whole strength curve each arm was frozen FROM: compare stage, q90,
    n = 512, the cells check_fullrun_go reads. Diverged cells are dropped, as
    there (a divergent cell can never be anyone's best)."""
    by = {}
    for f in glob.glob(os.path.join(ROOT, "results", "sweep", "*.json")):
        d = json.load(open(f))
        if (d.get("stage") != "compare" or d.get("target_name") != "q90"
                or d.get("n_nonfinite", 0) > 0):
            continue
        by.setdefault((d["prop"], d["arm"]), {})[float(d["w"])] = d
    return by


GUIDED = ["plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var"]


def frontier_sections():
    """Why lgd_mc keeps w = 4 and nobody else does -- read off the curve FR3a
    selected on, not off the frozen points."""
    by = compare_curve()
    if not by:
        return ["## Selection curve unavailable", "",
                "No compare/q90 cells under results/sweep/.", ""]
    L = []
    L.append("## Why lgd_mc keeps w = 4 and no other arm does")
    L.append("")
    L.append("FR3a freezes a **grid point**, so it can only pick a strength "
             "the sweep actually ran. The table below reads the selection "
             "curve itself (compare stage, q90, n = 512 -- the cells "
             "`check_fullrun_go.py` reads) and locates where each arm's "
             "chemistry crosses the floor.")
    L.append("")
    L.append("| property | arm | last w clearing the floor | chem there | first w failing | chem there | what FR3a could see |")
    L.append("|---|---|---|---|---|---|---|")
    notes = []
    for prop, _u, _s in PROPS:
        u = by[(prop, "unguided")][1.0]["mol_stability"]
        for arm in GUIDED:
            d = by.get((prop, arm), {})
            if not d:
                continue
            ok = [w for w in sorted(d) if d[w]["mol_stability"] >= FLOOR * u]
            bad = [w for w in sorted(d) if d[w]["mol_stability"] < FLOOR * u]
            last, first = (max(ok) if ok else None), (min(bad) if bad else None)
            if last is None:
                note = "never clears the floor"
            elif first is None:
                note = "**never fails -- frozen at the grid ceiling w = 4; its own optimum is ABOVE the grid**"
                notes.append((prop, arm))
            elif first < last:
                note = "non-monotone: fails at %g, clears again at %g" % (first, last)
            else:
                note = "crossing hidden inside (%g, %g), a **%.0fx** interval" % (
                    last, first, first / last)
            L.append("| %s | %s | %s | %s | %s | %s | %s |"
                     % (prop, arm,
                        ("w = %g" % last) if last is not None else "-",
                        pct(d[last]["mol_stability"] / u) if last is not None else "-",
                        ("w = %g" % first) if first is not None else "none",
                        pct(d[first]["mol_stability"] / u) if first is not None else "-",
                        note))
    L.append("")
    L.append("**What this means.** `lgd_mc` is the only arm whose chemistry "
             "never drops below the floor anywhere on the grid, so FR3a hands "
             "it the grid's maximum strength. Every other arm crosses the "
             "floor *inside* a 2x-5x grid interval, so FR3a must fall back to "
             "the last point before the crossing -- which can be far below "
             "that arm's true floor-clearing optimum. The comparison is "
             "therefore not 'each arm at its best chemistry-matched "
             "strength'; it is 'lgd_mc at the grid ceiling against everyone "
             "else at the bottom of their crossing interval'.")
    L.append("")
    L.append("Two consequences, in opposite directions: the competitors are "
             "**understated** (their floor-optimum was never run), and "
             "`lgd_mc` is also understated (it is capped by the grid, not by "
             "the floor). Refining the grid by bisecting each crossing -- 2-3 "
             "extra n = 512 cells per (arm, property) -- would settle both.")
    L.append("")

    L.append("## The same curve as an equal-chemistry frontier")
    L.append("")
    L.append("Each cell is `chem% / MAE`, chemistry as a percentage of "
             "unguided `mol_stability`. Compare arms **along equal chemistry**, "
             "not along equal w: that is the comparison the floor was "
             "introduced to make. Units: mu in D, alpha in Bohr^3, gap in "
             "Hartree (this is the n = 512 selection stage, so the numbers are "
             "noisier and larger than the full run's).")
    L.append("")
    for prop, _u, _s in PROPS:
        u = by[(prop, "unguided")][1.0]
        L.append("**%s** -- unguided: chem 100%%, MAE %.4g, in-band %.3f"
                 % (prop, u["prop_mae_eval"], u["in_band_fraction"]))
        L.append("")
        ws = sorted({w for a in GUIDED for w in by.get((prop, a), {})})
        L.append("| arm | " + " | ".join("w = %g" % w for w in ws) + " |")
        L.append("|---" * (len(ws) + 1) + "|")
        for arm in GUIDED:
            d = by.get((prop, arm), {})
            cs = []
            for w in ws:
                if w not in d:
                    cs.append("-")
                    continue
                ch = d[w]["mol_stability"] / u["mol_stability"]
                cell = "%s / %.3g" % (pct(ch), d[w]["prop_mae_eval"])
                cs.append("**%s**" % cell if ch >= FLOOR else cell)
            L.append("| %s | %s |" % (arm, " | ".join(cs)))
        L.append("")
    L.append("Bold = clears the chemistry floor. Reading down a column "
             "compares arms at equal w (what the grid does); reading for equal "
             "chem%% across rows compares them at equal chemistry (what the "
             "claim needs).")
    L.append("")
    return L


def se_mae(r):
    """se of MAE from its first two moments, as check_fullrun_go.se_mae."""
    v = max(r["prop_rmse_eval"] ** 2 - r["prop_mae_eval"] ** 2, 0.0)
    return math.sqrt(v / r["n"])


def q50_curve():
    """The q50, n=512, t_min=0.5 cells -- the one protocol where lgd_mc, osc
    and tfg_mc were all run, so the smoothed-gradient family can be compared."""
    by = {}
    for f in glob.glob(os.path.join(ROOT, "results", "sweep", "*.json")):
        d = json.load(open(f))
        if (d.get("target_name") != "q50" or d.get("n") != 512
                or d.get("t_min_guide") != 0.5 or d.get("n_nonfinite", 0) > 0):
            continue
        by.setdefault((d["prop"], d["arm"]), {})[float(d["w"])] = d
    return by


def audit_sections(cells, W3a):
    """Three checks that bear on whether the headline survives. Each is
    recomputed here from the cells, not quoted."""
    L = ["## Three checks on the headline", ""]

    # --- (a) does the freeze survive the scale it is applied at?
    L.append("### a. Re-applying the floor at n = 5,000 changes who wins alpha")
    L.append("")
    L.append("FR3a was decided on n = 512 / q90 and then applied at "
             "n = 5,000 / `dist`. Chemistry behaves differently there, so the "
             "arms the floor excluded at selection are not the ones it would "
             "exclude at evaluation. Below: the **w = 4** cell of every arm "
             "scored against **this seed's own** floor (0.9 x unguided, seed "
             "%d), which is the comparison FR3a would have made had it been "
             "run here." % SEED)
    L.append("")
    for prop, unit, scale in PROPS:
        u = get(cells, prop, "unguided", 1.0)
        fl = FLOOR * u["mol_stability"]
        have = [a for a in GUIDED if (prop, a, 4.0) in cells]
        if not have:
            continue
        L.append("**%s** (floor = %.4f; MAE in %s)" % (prop, fl, unit))
        L.append("")
        L.append("| arm at w = 4 | mol stab | clears floor here? | MAE | in-band |")
        L.append("|---|---|---|---|---|")
        for a in have:
            r = cells[(prop, a, 4.0)]
            L.append("| %s | %.4f | %s | %.4g | %.4f |"
                     % (a, r["mol_stability"],
                        "**yes**" if r["mol_stability"] >= fl else "no",
                        r["prop_mae_eval"] * scale, r["in_band_fraction"]))
        lg = cells.get((prop, "lgd_mc", 4.0))
        if lg is not None:
            beat = [a for a in have
                    if a != "lgd_mc"
                    and cells[(prop, a, 4.0)]["mol_stability"] >= fl
                    and cells[(prop, a, 4.0)]["in_band_fraction"]
                    > lg["in_band_fraction"]]
            n = lg["n"]
            det = []
            for a in beat:
                r = cells[(prop, a, 4.0)]
                sd = math.sqrt(
                    lg["in_band_fraction"] * (1 - lg["in_band_fraction"]) / n
                    + r["in_band_fraction"] * (1 - r["in_band_fraction"]) / n)
                det.append("`%s` %+.4f (%.1f se)"
                           % (a, r["in_band_fraction"] - lg["in_band_fraction"],
                              (r["in_band_fraction"] - lg["in_band_fraction"]) / sd))
            L.append("")
            if det:
                L.append("Arms that clear the floor **here** and are nominally "
                         "ahead of `lgd_mc` on in-band: %s. Margins this small "
                         "are NOT wins -- the honest reading is that on this "
                         "property the arms become indistinguishable once the "
                         "floor is applied where the run happens, so "
                         "`lgd_mc`'s advantage does not survive the move."
                         % "; ".join(det))
            else:
                L.append("No arm clearing the floor here is ahead of "
                         "`lgd_mc` on in-band: its lead holds on this property.")
        L.append("")

    # --- (b) is the frozen point stable across seeds?
    L.append("### b. `lgd_mc` at w = 4 fails its own rubric on one seed")
    L.append("")
    L.append("The floor is defined per (property, target, seed). Checking the "
             "headline arm against each seed's own unguided reference:")
    L.append("")
    L.append("| property | seed | lgd_mc mol stab | that seed's floor | verdict |")
    L.append("|---|---|---|---|---|")
    fails = 0
    for prop, _u, _s in PROPS:
        for s in (20261001, 20261002, 20261003):
            try:
                u = json.load(open(glob.glob(os.path.join(
                    FULL, "seed%d" % s, "%s__unguided__*.json" % prop))[0]))
                g = json.load(open(glob.glob(os.path.join(
                    FULL, "seed%d" % s, "%s__lgd_mc__*w4__*.json" % prop))[0]))
            except (IndexError, IOError):
                continue
            fl = FLOOR * u["mol_stability"]
            ok = g["mol_stability"] >= fl
            fails += (not ok)
            L.append("| %s | %d | %.4f | %.4f | %s |"
                     % (prop, s, g["mol_stability"], fl,
                        "clears" if ok else "**FAILS**"))
    L.append("")
    L.append("%d of the headline arm's full-run cells violate%s the chemistry "
             "floor that FR5 requires of any arm before it can win or be "
             "beaten. The three-seed mean hides it, because the mean clears "
             "even when a constituent seed does not."
             % (fails, "s" if fails == 1 else ""))
    L.append("")

    # --- (c) the smoothed-gradient family
    L.append("### c. `lgd_mc` is not separable from its own smoothing ingredient")
    L.append("")
    L.append("`osc` and `tfg_mc` are on disk (66 cells each) but are not in "
             "`COMPARE_SET` (`guidance_sweep.py:171-172`); the source excludes "
             "`tfg_mc` on the stated ground that it is TFG's smoothing "
             "ingredient rather than a published method. That is defensible "
             "for 'three external methods', but it means the headline 'best "
             "arm' was never tested against the rest of its own family. The "
             "one protocol where all three ran is q50 / n = 512 / t_min = 0.5, "
             "one seed, FR3a applied within it:")
    L.append("")
    by = q50_curve()
    for prop, unit, scale in PROPS:
        u = by.get((prop, "unguided"))
        if not u:
            continue
        uu = u[sorted(u)[0]]
        fl = FLOOR * uu["mol_stability"]
        L.append("**%s** (floor = %.4f; MAE in %s)" % (prop, fl, unit))
        L.append("")
        L.append("| arm | FR3a w here | mol stab | MAE | in-band |")
        L.append("|---|---|---|---|---|")
        best = {}
        for arm in ("lgd_mc", "osc", "tfg_mc"):
            d = by.get((prop, arm), {})
            ok = [w for w in sorted(d) if d[w]["mol_stability"] >= fl]
            if not ok:
                continue
            b = min((d[w] for w in ok),
                    key=lambda r: (r["prop_mae_eval"], float(r["w"])))
            best[arm] = b
            L.append("| %s | %g | %.4f | %.4g | %.4f |"
                     % (arm, b["w"], b["mol_stability"],
                        b["prop_mae_eval"] * scale, b["in_band_fraction"]))
        if "lgd_mc" in best and "tfg_mc" in best:
            a, b = best["lgd_mc"], best["tfg_mc"]
            n = a["n"]
            sib = math.sqrt(
                a["in_band_fraction"] * (1 - a["in_band_fraction"]) / n
                + b["in_band_fraction"] * (1 - b["in_band_fraction"]) / n)
            sm = math.sqrt(se_mae(a) ** 2 + se_mae(b) ** 2)
            L.append("")
            L.append("tfg_mc - lgd_mc: in-band %+.4f (**%.1f se**), "
                     "MAE %+.4g (**%.1f se**)."
                     % (b["in_band_fraction"] - a["in_band_fraction"],
                        (b["in_band_fraction"] - a["in_band_fraction"]) / sib,
                        (b["prop_mae_eval"] - a["prop_mae_eval"]) * scale,
                        (b["prop_mae_eval"] - a["prop_mae_eval"]) / sm))
        L.append("")
    L.append("**Read this carefully in both directions.** `tfg_mc` is ahead of "
             "`lgd_mc` on every cell, but on in-band -- the metric FR5 decides "
             "on -- the margins are 0.1-1.0 se, far under FR5's own |sigma| >= 3 "
             "bar. So this is **not** 'tfg_mc beats lgd_mc'. It is the weaker "
             "and more awkward statement: at the protocol where they were both "
             "measured, `lgd_mc` is **statistically indistinguishable from TFG's "
             "smoothing ingredient alone**. Any claim that attributes lgd_mc's "
             "result to likelihood marginalisation, rather than to gradient "
             "smoothing that `tfg_mc` also has, is unsupported by these cells.")
    L.append("")
    return L


def main():
    fz = json.load(open(os.path.join(FULL, "frozen_q90_tfg.json")))
    W3a, W3 = fz["frozen_w"], fz["frozen_w_mae"]
    cells = load_cells()

    L = []
    L.append("# What the chemistry floor cost: FR3a against FR3")
    L.append("")
    L.append("**Generated by `python proj1/scripts/fr3a_vs_fr3_table.py`. "
             "Do not edit by hand; re-run the script.** Sources and the "
             "pairing argument are in its docstring.")
    L.append("")
    L.append("Every number below is **seed %d, n = 5,000, target `dist`, "
             "window t_min = 0.5**, the one seed where both strength sets ran "
             "in full. FR3a's headline three-seed numbers are in "
             "[FULL_RUN_RESULTS.md](FULL_RUN_RESULTS.md) section 1; they are "
             "not averaged in here, because FR3 has only this seed and a "
             "mixed-seed contrast would not be paired." % SEED)
    L.append("")
    L.append("- **FR3a** = best-MAE strength clearing the chemistry floor "
             "(mol_stability >= %.1f x unguided). The headline set." % FLOOR)
    L.append("- **FR3** = best-MAE strength, unconstrained. As pre-registered.")
    L.append("- `tfg` is POST HOC (replaced dflow 23 Sep, same rule, own compare cells).")
    L.append("")

    for prop, unit, scale in PROPS:
        u3a = get(cells, prop, "unguided", W3a["unguided"][prop])
        L.append("## %s (MAE in %s)" % (prop, unit))
        L.append("")
        L.append("| arm | w FR3a / FR3 | in-band FR3a / FR3 | in-band x unguided FR3a / FR3 "
                 "| MAE FR3a / FR3 | mol stab FR3a / FR3 | chem % of unguided FR3a / FR3 "
                 "| validity FR3a / FR3 | uniq-valid/sample FR3a / FR3 "
                 "| clipped steps FR3a / FR3 | floor FR3a / FR3 |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in ARMS:
            wa, wm = W3a[arm][prop], W3[arm][prop]
            a, m = get(cells, prop, arm, wa), get(cells, prop, arm, wm)
            same = " (same)" if wa == wm else ""
            fa = a["mol_stability"] >= FLOOR * u3a["mol_stability"] - 1e-12
            fm = m["mol_stability"] >= FLOOR * u3a["mol_stability"] - 1e-12
            L.append("| %s%s | %g / %g%s | %.3f / %.3f | %.2fx / %.2fx | %.3f / %.3f "
                     "| %.3f / %.3f | %s / %s | %.3f / %.3f | %.3f / %.3f "
                     "| %s / %s | %s / %s |"
                     % ("**%s**" % arm if arm == "lgd_mc" else arm,
                        "" if arm != "unguided" else " (ref)",
                        wa, wm, same,
                        a["in_band_fraction"], m["in_band_fraction"],
                        a["in_band_fraction"] / u3a["in_band_fraction"],
                        m["in_band_fraction"] / u3a["in_band_fraction"],
                        a["prop_mae_eval"] * scale, m["prop_mae_eval"] * scale,
                        a["mol_stability"], m["mol_stability"],
                        pct(a["mol_stability"] / u3a["mol_stability"]),
                        pct(m["mol_stability"] / u3a["mol_stability"]),
                        a["validity"], m["validity"],
                        a["unique_valid_per_sample"], m["unique_valid_per_sample"],
                        "{:,}".format(a.get("clipped_sample_steps", 0)),
                        "{:,}".format(m.get("clipped_sample_steps", 0)),
                        "ok" if fa else "**NO**", "ok" if fm else "**NO**"))
        L.append("")

    # ---- the trade, in one place
    L.append("## The trade in one table")
    L.append("")
    L.append("For each arm: what unconstrained strength bought in in-band, and "
             "what it cost in chemistry. `in-band gained` is FR3 minus FR3a in "
             "percentage points; `chemistry lost` is the same subtraction on "
             "mol_stability as a percentage of unguided. An arm whose two "
             "columns are both ~0 was never held back by the floor.")
    L.append("")
    L.append("| arm | property | w FR3a -> FR3 | in-band gained (pp) | chemistry lost (pp of unguided) | held back by the floor? |")
    L.append("|---|---|---|---|---|---|")
    for arm in ARMS:
        if arm == "unguided":
            continue
        for prop, _u, _s in PROPS:
            wa, wm = W3a[arm][prop], W3[arm][prop]
            u3a = get(cells, prop, "unguided", W3a["unguided"][prop])
            a, m = get(cells, prop, arm, wa), get(cells, prop, arm, wm)
            dib = 100.0 * (m["in_band_fraction"] - a["in_band_fraction"])
            dch = 100.0 * (m["mol_stability"] - a["mol_stability"]) / u3a["mol_stability"]
            held = "no -- same strength" if wa == wm else "**yes**"
            L.append("| %s | %s | %g -> %g | %+.1f | %+.1f | %s |"
                     % (arm, prop, wa, wm, dib, dch, held))
    L.append("")

    # ---- clip pressure, the reason lgd_mc is not like the others
    L.append("## Clip pressure at every frozen strength")
    L.append("")
    L.append("`clipped_sample_steps` out of %s guided sample-steps per cell "
             "(5,000 samples x 100 steps x the half of the trajectory that is "
             "guided at t_min = 0.5). A high fraction means the arm's raw "
             "field is being cut down to the clip norm rather than applied as "
             "the method defines it, so the strength dial is no longer what "
             "sets the step." % "{:,}".format(int(GUIDED_STEPS)))
    L.append("")
    L.append("| arm | mu FR3a / FR3 | alpha FR3a / FR3 | gap FR3a / FR3 |")
    L.append("|---|---|---|---|")
    for arm in ARMS:
        cs = []
        for prop, _u, _s in PROPS:
            a = get(cells, prop, arm, W3a[arm][prop])
            m = get(cells, prop, arm, W3[arm][prop])
            cs.append("%s / %s"
                      % (pct(a.get("clipped_sample_steps", 0) / GUIDED_STEPS),
                         pct(m.get("clipped_sample_steps", 0) / GUIDED_STEPS)))
        L.append("| %s | %s |" % (arm, " | ".join(cs)))
    L.append("")

    # ---- nominal vs applied strength
    L.append("## Nominal w is not a common scale")
    L.append("")
    L.append("`w_applied` is what the sampler actually multiplied the field "
             "by. Only btvg and btvg_var carry a rescale (`w_scale`, the "
             "(tau/s)^2 normalisation in `guidance_sweep.strength_scale`); "
             "every other arm applies w as written. This does NOT make the "
             "arms' fields equal in magnitude -- each arm's field has its own "
             "natural size -- so a shared w is a shared dial, not a shared "
             "step. Read the floor as selecting each arm's own "
             "chemistry-matched operating point.")
    L.append("")
    L.append("| arm | property | w FR3a | w_applied FR3a | w_scale | w FR3 | w_applied FR3 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        for prop, _u, _s in PROPS:
            a = get(cells, prop, arm, W3a[arm][prop])
            m = get(cells, prop, arm, W3[arm][prop])
            L.append("| %s | %s | %g | %.6g | %.6g | %g | %.6g |"
                     % (arm, prop, a["w"], a.get("w_applied", a["w"]),
                        a.get("w_scale", 1.0), m["w"],
                        m.get("w_applied", m["w"])))
    L.append("")

    L.extend(frontier_sections())
    L.extend(audit_sections(cells, W3a))

    prov = get(cells, "mu", "plug", W3a["plug"]["mu"]).get("prov", {})
    L.append("---")
    L.append("")
    L.append("Provenance: generator `%s` md5 `%s`, torch %s, cuda %s. "
             "delta = %g x f_B validation MAE. Frozen-strength file "
             "`frozen_q90_tfg.json` (%s stage, target %s, source seed %s)."
             % (os.path.basename(str(prov.get("fm_path", "?"))),
                str(prov.get("fm_md5", "?"))[:12], prov.get("torch", "?"),
                prov.get("cuda", "?"),
                get(cells, "mu", "plug", W3a["plug"]["mu"]).get("k_delta", "?"),
                fz.get("source_stage"), fz.get("target"), fz.get("source_seed")))
    print("\n".join(L))


if __name__ == "__main__":
    main()
