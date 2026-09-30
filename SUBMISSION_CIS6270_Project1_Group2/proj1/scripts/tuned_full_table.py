"""Read the tuned full run and print its table. CPU only, ~1 min.

    python proj1/scripts/tuned_full_table.py \
        --root results/full/v2_tuned/n4000 \
        --frozen results/tune/n1000/frozen_tune.json \
        --md-out docs/results/TUNED_FULL_RUN_RESULTS.md

WHY THIS IS NOT full_run_v2_table.py. That script cannot read this tree, by
design, and adapting it would have meant weakening four of its refusals:
it globs `*__full.json` and requires `stage == "full"`; it loads
`frozen_v2.json`; it refuses MORE THAN ONE cell per (property, arm) -- which
is exactly what two operating points produce; and it refuses a mixed
`t_min_guide` -- which is exactly what per-arm start times produce. Those
refusals are right for v2 and wrong here, so this run gets its own reader.

WHAT IT REPORTS, per property:
  * each arm at BOTH operating points -- floor-clearing and unconstrained --
    with in-band CONTINUOUS and DECODED, MAE, bias, residual sd, the full
    chemistry block, distinctness, and the cost counters.
  * the chemistry floor re-measured at full scale (0.9 x this run's unguided
    molecule stability), not inherited from the n = 1000 screen. v2 showed the
    screen can promote a strength the run cannot afford.
  * z against unguided, iid AND cluster-robust (all seeds reuse the same `val`
    sizes, so rows are clustered by molecule), on BOTH metrics.
  * the winner-vs-runner-up comparison, because a verdict that does not show
    it reads as a win when it is a tie.
  * what the chemistry floor COST: floor pick against open pick, per arm. That
    comparison is the reason the run sampled two points.

WHAT IT REFUSES: a missing cell, a cell whose (w, t_min) is not the one the
freeze chose, mixed n/seed/generator/delta, or a frozen file whose delta is
not the run's.
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
sys.path.insert(0, HERE)

from full_run_v2_table import (_z, cluster_se, holm, ib_vec,  # noqa: E402
                               paired_z, prop_se, pval)

NL = chr(10)
PROPS = ("mu", "alpha", "gap")
SEEDS = ("20261001", "20261002", "20261003")
FLOOR = 0.9
SIGMA = 3.0
UNIQ_MIN = 0.95
SETS = ("floor", "open")


def load(root, frozen):
    """{(set, prop, arm): [rows]} plus sidecar paths, keyed by operating point."""
    want = {}
    for p in PROPS:
        for a in frozen["picks"]:
            for s in SETS:
                pk = frozen["picks"][a].get(p, {}).get(s)
                if pk:
                    want[(s, p, a)] = (float(pk["w"]), float(pk["t_min_guide"]))
    got, problems = {}, []
    for seed in SEEDS:
        d = os.path.join(root, "seed%s" % seed)
        if not os.path.isdir(d):
            problems.append("no directory for seed %s" % seed)
            continue
        for fn in sorted(glob.glob(os.path.join(d, "*__tuned.json"))):
            r = json.load(open(fn))
            side = fn[:-5] + ".permol.pt"
            r["_side"] = side if os.path.exists(side) else None
            r["_seed"] = seed
            got[(r["prop"], r["arm"], float(r["w"]), float(r["t_min_guide"]), seed)] = r
    cells = {}
    for (s, p, a), (w, t) in want.items():
        rows = [got.get((p, a, w, t, sd)) for sd in SEEDS]
        if any(r is None for r in rows):
            miss = [sd for sd, r in zip(SEEDS, rows) if r is None]
            problems.append("%s/%s/%s at w=%g t=%g: no cell for seed(s) %s"
                            % (s, p, a, w, t, ", ".join(miss)))
            continue
        if any(r["_side"] is None for r in rows):
            problems.append("%s/%s/%s: a per-molecule sidecar is missing" % (s, p, a))
            continue
        cells[(s, p, a)] = rows
    return cells, problems


def check(cells, problems, frozen):
    rows = [r for rs in cells.values() for r in rs]
    if not rows:
        return problems + ["no cells found at all"]
    for key in ("n", "steps", "solver", "clip", "batch", "target_name"):
        vals = {r.get(key) for r in rows}
        if len(vals) > 1:
            problems.append("cells disagree on %s: %s" % (key, sorted(map(str, vals))))
    for key in ("fm_md5", "frozen_md5"):
        vals = {(r.get("prov") or {}).get(key) for r in rows}
        if len(vals) > 1:
            problems.append("cells disagree on %s: %s" % (key, sorted(map(str, vals))))
    for p in PROPS:
        ds = {round(float(r["delta"]), 12) for r in rows if r["prop"] == p}
        if len(ds) > 1:
            problems.append("%s: cells sampled under different deltas %s" % (p, sorted(ds)))
        fd = (frozen.get("delta") or {}).get(p)
        if ds and fd is not None and abs(list(ds)[0] - fd) > 1e-12:
            problems.append("%s: the run's delta %r is not the freeze's %r"
                            % (p, list(ds)[0], fd))
    for r in rows:
        if str(r.get("seed")) != r["_seed"]:
            problems.append("%s says seed=%s but sits in seed%s"
                            % (r["_file"] if "_file" in r else r["arm"],
                               r.get("seed"), r["_seed"]))
    return problems


def pm(rows):
    import torch
    parts = [torch.load(r["_side"], weights_only=False) for r in rows]
    cat = lambda k: torch.cat([p[k] for p in parts])
    keys = ("f_B", "y", "finite", "mol_idx", "n_atoms", "mol_stable", "valid")
    out = {k: cat(k) for k in keys}
    if all("f_B_dec" in p for p in parts):
        out["f_B_dec"] = cat("f_B_dec")
    return out


def agg(rows, side, delta):
    import torch
    N = sum(r["n"] for r in rows)
    wm = lambda k: sum(r[k] * r["n"] for r in rows) / N
    e = (side["f_B"] - side["y"])[side["finite"]].double()
    cse, deff = cluster_se(side, delta)
    csed, _ = cluster_se(side, delta, dec=True)
    c = rows[0].get("cost") or {}
    out = {
        "N": N, "w": rows[0]["w"], "t": rows[0]["t_min_guide"],
        "in_band": wm("in_band_fraction"), "in_band_dec": wm("in_band_fraction_dec"),
        "mae": wm("prop_mae_eval"), "mae_dec": wm("prop_mae_eval_dec"),
        "bias": float(e.mean()), "resid": float(e.std(unbiased=False)),
        "mol_stab": wm("mol_stability"), "atom_stab": wm("atom_stability"),
        "validity": wm("validity"), "uniq": wm("uniqueness_of_valid"),
        "uniq_min": min(r["uniqueness_of_valid"] for r in rows),
        "uvps": wm("unique_valid_per_sample"),
        "guide_gap": wm("guide_eval_gap_mean"),
        "nonfinite": sum(r["n_nonfinite"] for r in rows),
        "nfe": sum(v for v in c.values() if isinstance(v, (int, float))),
        "clip": sum(r.get("clipped_sample_steps", 0) for r in rows),
    }
    out["se"] = prop_se(out["in_band"], N)
    out["se_dec"] = prop_se(out["in_band_dec"], N)
    out["cse"], out["deff"], out["cse_dec"] = cse, deff, csed
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "full", "v2_tuned", "n4000"))
    ap.add_argument("--frozen", default=os.path.join(ROOT, "results", "tune", "n1000", "frozen_tune.json"))
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    frozen = json.load(open(args.frozen))
    if frozen.get("schema") != "frozen_tune/1":
        raise SystemExit("%s is not a frozen_tune file" % args.frozen)
    cells, problems = load(args.root, frozen)
    problems = check(cells, problems, frozen)
    if problems:
        print("REFUSING: the tuned full run is not complete or not consistent.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)
    import torch  # noqa: F401

    ARMS = [a for a in frozen["picks"]]
    ARMS = ["unguided"] + [a for a in ARMS if a != "unguided"]
    DELTA = {p: float(cells[("floor", p, "unguided")][0]["delta"]) for p in PROPS}
    PMs = {k: pm(v) for k, v in cells.items()}
    A = {k: agg(cells[k], PMs[k], DELTA[k[1]]) for k in cells}
    any_row = cells[("floor", PROPS[0], "unguided")][0]
    prov = any_row.get("prov") or {}

    L = []
    L.append("# Tuned full run — strength AND start-time frozen jointly, both operating points")
    L.append("")
    L.append("Generated by `proj1/scripts/tuned_full_table.py`. Do not hand-edit; "
             "re-run the script.")
    L.append("")
    L.append("Plan: [TUNE_SWEEP_PLAN.md](../protocol/TUNE_SWEEP_PLAN.md). Screen and "
             "picks: [TUNE_SWEEP_TABLE.md](TUNE_SWEEP_TABLE.md).")
    L.append("")
    L.append("| provenance | value |")
    L.append("|---|---|")
    L.append("| n per cell, seeds | %d, %s |" % (any_row["n"], ", ".join(SEEDS)))
    L.append("| target | `%s` |" % any_row["target_name"])
    L.append("| in-band delta | %s |" % any_row.get("delta_source"))
    L.append("| | " + ", ".join("%s %.5f" % (p, DELTA[p]) for p in PROPS) + " |")
    L.append("| generator | `%s` md5 `%s` |" % (any_row.get("fm"), str(prov.get("fm_md5"))[:8]))
    L.append("| frozen by | `%s` md5 `%s` (screen n = %s, seed %s) |"
             % (os.path.basename(str(prov.get("frozen_path"))),
                str(prov.get("frozen_md5"))[:8], prov.get("frozen_screen_n"),
                prov.get("frozen_screen_seed")))
    L.append("| selection metric | %s |" % prov.get("frozen_selection_metric"))
    L.append("| sampler | %d-step %s, clip %g, batch %d |"
             % (any_row["steps"], any_row["solver"], any_row["clip"], any_row["batch"]))
    L.append("| non-finite | %d across all cells |"
             % sum(A[k]["nonfinite"] for k in A))
    L.append("")
    L.append("**Each arm runs at its OWN start time**, not a shared one — that is the "
             "point of the screen. `floor` is the operating point that keeps molecule "
             "stability at or above %g x unguided; `open` is the best in-band with no "
             "chemistry constraint and is **not a recommendation**, it is the ceiling "
             "used to measure what the floor costs. Where an arm's two picks coincide "
             "the cell was sampled once and appears in both." % FLOOR)
    L.append("")

    # ---- per property ----------------------------------------------------
    for p in PROPS:
        u = A[("floor", p, "unguided")]
        fl = FLOOR * u["mol_stab"]
        L.append("## %s (target %.4f, delta %.5f)"
                 % (p, cells[("floor", p, "unguided")][0]["target"], DELTA[p]))
        L.append("")
        L.append("Chemistry floor re-measured at full scale: **%.4f** = 0.9 x this "
                 "run's unguided %.4f. The screen's floor was %.4f — a pick chosen "
                 "there is not guaranteed to clear here."
                 % (fl, u["mol_stab"], (frozen.get("floor") or {}).get(p, float("nan"))))
        L.append("")
        L.append("| set | arm | w | t_min | in_band | ±se | dec | MAE/d | bias/d | "
                 "resid sd/d | mol_stab | vs floor | valid | uniq (min cell) | yield |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        d = DELTA[p]
        for s in SETS:
            for a in ARMS:
                k = (s, p, a)
                if k not in A:
                    continue
                r = A[k]
                ok = r["mol_stab"] >= fl - 1e-12
                L.append("| %s | `%s` | %g | %g | %.4f | %.4f | %.4f | %.3f | %+.3f | "
                         "%.3f | %.4f | %s | %.4f | %.4f (%.4f) | %.4f |"
                         % (s, a, r["w"], r["t"], r["in_band"], r["se"], r["in_band_dec"],
                            r["mae"] / d, r["bias"] / d, r["resid"] / d, r["mol_stab"],
                            "ok" if ok else "**FAILS**", r["validity"], r["uniq"],
                            r["uniq_min"], r["uvps"]))
        L.append("")

        # verdict among floor-clearing arms, on both metrics, iid and clustered
        elig = [a for a in ARMS
                if ("floor", p, a) in A and A[("floor", p, a)]["mol_stab"] >= fl - 1e-12]
        L.append("| arm (floor set) | in_band | z vs unguided | clustered | "
                 "dec z | dec clustered | z_pair | chem |")
        L.append("|---|---|---|---|---|---|---|---|")
        for a in sorted([x for x in ARMS if ("floor", p, x) in A],
                        key=lambda x: -A[("floor", p, x)]["in_band"]):
            r, uu = A[("floor", p, a)], u
            f = lambda v: "--" if v != v else "%+.2f" % v
            z1 = float("nan") if a == "unguided" else _z(
                r["in_band"] - uu["in_band"], math.hypot(r["se"], uu["se"]))
            z2 = float("nan") if a == "unguided" else _z(
                r["in_band"] - uu["in_band"], math.hypot(r["cse"], uu["cse"]))
            z3 = float("nan") if a == "unguided" else _z(
                r["in_band_dec"] - uu["in_band_dec"], math.hypot(r["se_dec"], uu["se_dec"]))
            z4 = float("nan") if a == "unguided" else _z(
                r["in_band_dec"] - uu["in_band_dec"], math.hypot(r["cse_dec"], uu["cse_dec"]))
            zp = (float("nan") if a == "unguided"
                  else paired_z(PMs[("floor", p, a)], PMs[("floor", p, "unguided")], d))
            L.append("| `%s` | %.4f | %s | %s | %s | %s | %s | %s |"
                     % (a, r["in_band"], f(z1), f(z2), f(z3), f(z4), f(zp),
                        "ok" if a in elig else "**floor-limited**"))
        L.append("")
        if elig:
            best = max(elig, key=lambda a: A[("floor", p, a)]["in_band"])
            rest = sorted([a for a in elig if a != best],
                          key=lambda a: -A[("floor", p, a)]["in_band"])
            ties = []
            L.append("| top floor-clearing arm vs | gap | z | separated at %g s? |" % SIGMA)
            L.append("|---|---|---|---|")
            for a in rest:
                g = A[("floor", p, best)]["in_band"] - A[("floor", p, a)]["in_band"]
                z = _z(g, math.hypot(A[("floor", p, best)]["se"], A[("floor", p, a)]["se"]))
                if abs(z) < SIGMA:
                    ties.append(a)
                L.append("| `%s` | %+.4f | %+.2f | %s |"
                         % (a, g, z, "yes" if abs(z) >= SIGMA else "**no - tie**"))
            L.append("")
            msg = ("**%s: the highest point estimate among floor-clearing arms is "
                   "`%s`** at w = %g, t_min = %g."
                   % (p, best, A[("floor", p, best)]["w"], A[("floor", p, best)]["t"]))
            if ties:
                msg += (" It is **not separated** from %s, so under the rubric these "
                        "are tied, not a win." % ", ".join("`%s`" % a for a in ties))
            L.append(msg)
            L.append("")

        # what the floor cost
        L.append("**What the chemistry floor cost on %s.** `open` minus `floor`, per "
                 "arm — the in-band given up to stay above %.4f stability, and the "
                 "stability that buying it would have cost." % (p, fl))
        L.append("")
        L.append("| arm | floor in_band | open in_band | in_band given up | "
                 "open mol_stab | stability given up | same pick? |")
        L.append("|---|---|---|---|---|---|---|")
        for a in ARMS:
            if ("floor", p, a) not in A or ("open", p, a) not in A:
                continue
            fr, orr = A[("floor", p, a)], A[("open", p, a)]
            same = (fr["w"] == orr["w"] and fr["t"] == orr["t"])
            L.append("| `%s` | %.4f | %.4f | %+.4f | %.4f | %+.4f | %s |"
                     % (a, fr["in_band"], orr["in_band"], orr["in_band"] - fr["in_band"],
                        orr["mol_stab"], orr["mol_stab"] - fr["mol_stab"],
                        "yes" if same else "no"))
        L.append("")

    # ---- distinctness, cost, resolving power ------------------------------
    worst = min(A[k]["uniq_min"] for k in A)
    L.append("## Distinctness, cost, and what the sigmas are conditional on")
    L.append("")
    L.append("Lowest uniqueness of valid in any single cell: **%.5f** against the "
             "%.2f threshold. %s" % (worst, UNIQ_MIN,
             "No cell is collapse-contaminated." if worst >= UNIQ_MIN
             else "**At least one cell is collapse-contaminated and its in-band "
                  "number cannot be read as coverage.**"))
    L.append("")
    L.append("| property | design effect (floor set) | iid se | clustered se |")
    L.append("|---|---|---|---|")
    for p in PROPS:
        ds = [A[("floor", p, a)]["deff"] for a in ARMS if ("floor", p, a) in A]
        L.append("| %s | %.2f - %.2f | %.4f | %.4f |"
                 % (p, min(ds), max(ds), A[("floor", p, "unguided")]["se"],
                    A[("floor", p, "unguided")]["cse"]))
    L.append("")
    L.append("All three seeds reuse the same `val` sizes, so rows are clustered by "
             "molecule and the iid se understates a population claim. Both are given "
             "in the verdict tables above.")
    L.append("")
    L.append("| set | arm | total NFE (per cell) | clipped guided steps |")
    L.append("|---|---|---|---|")
    for s in SETS:
        for a in ARMS:
            k = (s, PROPS[0], a)
            if k in A:
                L.append("| %s | `%s` | %d | %d |" % (s, a, A[k]["nfe"], A[k]["clip"]))
    L.append("")

    out = NL.join(L) + NL
    print(out)
    if args.md_out:
        dst = args.md_out if os.path.isabs(args.md_out) else os.path.join(ROOT, args.md_out)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8").write(out)
        print("wrote %s" % dst, file=sys.stderr)


if __name__ == "__main__":
    main()
