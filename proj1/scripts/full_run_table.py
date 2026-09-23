"""Read the full-scale run and print the headline table. CPU only, seconds.

    python proj1/scripts/full_run_table.py                  # results/full/n5000
    python proj1/scripts/full_run_table.py --n 10000
    python proj1/scripts/full_run_table.py --md-out docs/results/FULL_RUN_TABLE.md

WHAT IT CHECKS BEFORE PRINTING ANYTHING, and refuses on:
  * every expected seed directory exists and has every (property, arm) cell
  * each cell's own `seed` matches its directory, and its target is `dist`
  * every seed used the SAME strength for a given (property, arm) -- FR3 froze
    it once; a mismatch means something re-froze from a different grid
  * one n, one generator (fm_md5) and one frozen-strength file (frozen_md5)
    across every cell

WHAT IT PRINTS, per property, arms pooled over seeds:
  in_band, MAE/delta, |bias|/delta, residual sd/delta, mol_stability,
  validity, and the rubric verdict (chemistry floor = mol_stability >= 0.9 x
  unguided, from the saved rubric). Bias and residual sd come from the
  per-molecule sidecars over FINITE rows only, so an exploded sample cannot
  contaminate them.

FR5 -- the verdict is decided on in_band (the saved rubric's metric), never
on MAE, and only between arms that BOTH clear the chemistry floor:
  sigma_ind   independent-samples z. PRIMARY -- declared 22 Sep before any
              full-run cell existed, because it is the conservative one.
  sigma_pair  paired z over the SAME molecules and the SAME initial noise
              (from the .permol.pt sidecars). Supplementary: more powerful,
              and valid only because every arm in a seed shares both.
  "btvg beats" needs sigma_ind >= 3 on in_band. Less is a tie. MAE is shown in
  its own columns; when it significantly contradicts in_band the verdict says
  "mixed" rather than letting either metric win.

WHAT THE sigmas COVER. All seeds reuse the same test molecules, so every se
here is conditional on this target set. For a claim about the target
POPULATION, a molecule-clustered se can be up to sqrt(3) larger -- say so
beside any population-level claim.

STEERING BREAKDOWN: in_band by how far each molecule's target sits from what
the unguided generator produces AT THAT MOLECULE'S SIZE (the mean unguided
f_B over all rows with the same atom count), in units of the targets' sd. Size
explains much of alpha, so distance from the global mean overstates how much
steering `dist` asks for (reviewer, 22 Sep: alpha 0.76 sd global vs 0.48 sd
size-conditional).
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
NL = chr(10)          # literal: shell heredocs on this box mangle backslashes

PROPS = ("mu", "alpha", "gap")
# dflow is not in the full run (23 Sep decision; to be replaced by another
# competitor later). Its compare-stage cells are kept and reported separately.
ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "btvg", "btvg_var")
OURS = "btvg"
SEEDS = ("20261001", "20261002", "20261003")     # full_run.slurm's SEEDS
FLOOR = 0.9           # saved rubric: mol_stability >= 0.9 x unguided
SIGMA = 3.0           # FR5
BINS = (0.0, 0.5, 1.0, 1.5, float("inf"))     # target distance, in sd
MIN_SIZE_GROUP = 10   # fewer rows at an atom count -> use the global mean


def load_cells(root):
    """{seed_dir_name: {(prop, arm, w): (row, sidecar_path)}}. Keyed by w: the
    run holds two strength sets, so one (prop, arm) can have two cells."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, "seed*"))):
        seed = os.path.basename(d)[4:]
        out.setdefault(seed, {})
        for fn in glob.glob(os.path.join(d, "*__full.json")):
            r = json.load(open(fn))
            if r.get("stage") != "full":
                continue
            side = fn[:-5] + ".permol.pt"
            out[seed][(r["prop"], r["arm"], float(r["w"]))] = (
                r, side if os.path.exists(side) else None)
    return out


def check(cells, props, want_seeds, W):
    """Refuse a partial or inconsistent PRIMARY set. W = frozen_w (FR3a)."""
    problems = []
    for s in want_seeds:
        if s not in cells:
            problems.append("seed %s: no directory" % s)
    for seed in want_seeds:
        c = cells.get(seed, {})
        miss = ["%s/%s@w%g" % (p, a, W[a][p]) for p in props for a in ARMS
                if (p, a, float(W[a][p])) not in c]
        if miss:
            problems.append("seed %s missing %s" % (seed, ", ".join(miss)))
        nos = ["%s/%s" % (p, a) for p in props for a in ARMS
               if (p, a, float(W[a][p])) in c and c[(p, a, float(W[a][p]))][1] is None]
        if nos:
            problems.append("seed %s: no .permol.pt sidecar for %s" % (seed, ", ".join(nos)))
    for seed, c in cells.items():
        for (p, a, _w), (r, _) in c.items():
            if str(r.get("seed")) != seed:
                problems.append("%s/%s in seed%s says seed=%s" % (p, a, seed, r.get("seed")))
            if r.get("target_name") != "dist":
                problems.append("%s/%s seed %s has target %r, not dist"
                                % (p, a, seed, r.get("target_name")))
    rows = [r for c in cells.values() for (r, _) in c.values()]
    for key, what in (("n", "n"), ("fm_md5", "generator"), ("frozen_md5", "frozen file")):
        vals = {(r.get(key) if key == "n" else (r.get("prov") or {}).get(key))
                for r in rows}
        if len(vals) > 1:
            problems.append("mixed %s across cells: %s" % (what, sorted(map(str, vals))))
    return problems


def _z(d, se):
    """d / se, or NaN when se is 0 (e.g. both arms at in_band 0 or 1). NaN
    compares false against every threshold, so verdict() reads it as a tie."""
    return d / se if se > 0 else float("nan")


def _sd(xs):
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def per_mol(cells, prop, arm, seeds, w):
    """One arm's sidecars over seeds, concatenated in the FIXED seed order so
    rows line up across arms. None if any sidecar is missing."""
    import torch
    parts = []
    for seed in seeds:
        side = cells[seed][(prop, arm, float(w))][1]
        if side is None:
            return None
        parts.append(torch.load(side, weights_only=False))
    cat = lambda k: torch.cat([p[k] for p in parts])
    out = {k: cat(k) for k in ("f_B", "y", "finite", "mol_idx", "n_atoms")}
    out["n_seeds"] = len(parts)
    return out


def pooled(rows, pm=None):
    """Pool seeds. Proportions and means are n-weighted; MAE's se comes from
    its first two moments, var(|e|) = RMSE^2 - MAE^2 (exact)."""
    N = sum(r["n"] for r in rows)
    wmean = lambda k: sum(r[k] * r["n"] for r in rows) / N
    ib, mae = wmean("in_band_fraction"), wmean("prop_mae_eval")
    ms = math.sqrt(sum(r["prop_rmse_eval"] ** 2 * r["n"] for r in rows) / N)
    if pm is not None and bool(pm["finite"].any()):
        e = (pm["f_B"] - pm["y"])[pm["finite"]].double()
        bias, resid = e.mean().item(), e.std(unbiased=False).item()
    else:                      # no sidecars: the json approximation
        bias = wmean("f_B_mean") - wmean("target_mean")
        resid = math.sqrt(max(ms ** 2 - bias ** 2, 0))
    return {
        "N": N, "in_band": ib, "mae": mae, "rmse": ms,
        "se_ib": math.sqrt(max(ib * (1 - ib), 0) / N),
        "se_mae": math.sqrt(max(ms ** 2 - mae ** 2, 0) / N),
        "bias": bias, "resid_sd": resid,
        "mol_stab": wmean("mol_stability"), "validity": wmean("validity"),
        "seed_sd_ib": _sd([r["in_band_fraction"] for r in rows]),
        "seed_sd_mae": _sd([r["prop_mae_eval"] for r in rows]),
        "delta": rows[0]["delta"], "w": rows[0]["w"],
    }


def paired_z(a, b, delta, what):
    """z of (a - b) over the same molecules. `what` is 'ib' or 'mae'."""
    import torch
    if not torch.equal(a["mol_idx"], b["mol_idx"]):
        raise SystemExit("sidecars do not pair: molecule order differs")
    if what == "ib":
        xa = ((a["f_B"] - a["y"]).abs() <= delta) & a["finite"]
        xb = ((b["f_B"] - b["y"]).abs() <= delta) & b["finite"]
        d = xa.double() - xb.double()
    else:
        ok = a["finite"] & b["finite"]
        d = ((a["f_B"] - a["y"]).abs() - (b["f_B"] - b["y"]).abs())[ok].double()
    se = d.std(unbiased=True).item() / math.sqrt(d.numel())
    return d.mean().item() / se if se > 0 else float("nan")


def verdict(z_ib, z_mae, floor_ours, floor_other, other):
    """FR5 under the saved rubric. in_band decides; MAE can only make it
    'mixed'; an arm below the chemistry floor cannot win or be beaten."""
    if not floor_ours and not floor_other:
        return "both fail floor"
    if not floor_ours:
        return "%s fails floor" % OURS
    if not floor_other:
        return "%s fails floor" % other
    if (z_ib >= SIGMA and z_mae >= SIGMA) or (z_ib <= -SIGMA and z_mae <= -SIGMA):
        return "mixed (in_band vs MAE)"
    if z_ib >= SIGMA:
        return "%s beats" % OURS
    if z_ib <= -SIGMA:
        return "%s beats %s" % (other, OURS)
    return "tie"


def size_reference(pm_u):
    """Per row: the mean unguided f_B over finite rows of the same atom count
    (global mean where a size has fewer than MIN_SIZE_GROUP rows)."""
    import torch
    fb, ok, na = pm_u["f_B"].double(), pm_u["finite"], pm_u["n_atoms"]
    glob_m = fb[ok].mean()
    ref = torch.full_like(fb, glob_m.item())
    for k in na.unique().tolist():
        sel = (na == k) & ok
        if int(sel.sum()) >= MIN_SIZE_GROUP:
            ref[na == k] = fb[sel].mean()
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--root", default="", help="default results/full/n<N>")
    ap.add_argument("--props", default=",".join(PROPS))
    ap.add_argument("--seeds", default=",".join(SEEDS))
    ap.add_argument("--md-out", default="")
    ap.add_argument("--frozen", default="", help="default <root>/frozen_q90.json")
    args = ap.parse_args()
    root = args.root or os.path.join(ROOT, "results", "full", "n%d" % args.n)
    props = [p for p in args.props.split(",") if p]
    want = [s for s in args.seeds.split(",") if s]

    fzp = args.frozen or os.path.join(root, "frozen_q90.json")
    if not os.path.exists(fzp):
        print("no frozen-strength file at %s -- the run has not started" % fzp)
        return 2
    fz = json.load(open(fzp))
    W, WM = fz["frozen_w"], fz.get("frozen_w_mae")
    global ARMS
    if "tfg" in W:
        # tfg replaced dflow AFTER this run was read; its frozen strengths
        # live in a separate file (tfg_run.slurm), so the table includes it
        # only when that file is passed with --frozen
        ARMS = ARMS[:ARMS.index("lgd_mc") + 1] + ("tfg",) + ARMS[ARMS.index("lgd_mc") + 1:]
    cells = load_cells(root)
    problems = check(cells, props, want, W)
    if problems:
        print("NOT READING A PARTIAL OR INCONSISTENT RUN (%s):" % root)
        for p in problems:
            print("  - " + p)
        return 2
    seeds = list(want)

    L = []
    L.append("# Full-scale run: dist target, n=%d x %d seeds (%s)"
             % (args.n, len(seeds), ", ".join(seeds)))
    L.append("")
    L.append("PRIMARY set (FR3a): each arm at its best-MAE q90 compare-stage "
             "strength among those with mol_stability >= %.1f x unguided. MAE, "
             "bias and spread are in units of delta. Rubric: in_band gain over "
             "unguided, subject to the same floor measured here at full scale. "
             "Every se is conditional on these test molecules (see the "
             "docstring)." % FLOOR)
    if "tfg" in ARMS:
        L.append("")
        L.append("**tfg is POST HOC**: it replaced dflow on 23 Sep, after this "
                 "run was read. Same seeds, targets and settings, strength "
                 "frozen by the same FR3a rule on its own compare cells.")
    for p in props:
        PM = {a: per_mol(cells, p, a, seeds, W[a][p]) for a in ARMS}
        P = {a: pooled([cells[s][(p, a, float(W[a][p]))][0] for s in seeds], PM[a])
             for a in ARMS}
        u = P["unguided"]
        dl = u["delta"]
        floor = {a: P[a]["mol_stab"] >= FLOOR * u["mol_stab"] - 1e-12 for a in ARMS}
        L.append("")
        L.append("## %s  (delta = %.4g)" % (p, dl))
        L.append("")
        L.append("| arm | w | in_band | +-se | seed sd | MAE/d | +-se | |bias|/d "
                 "| resid sd/d | mol_stab | valid | chem floor | rubric |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in ARMS:
            r = P[a]
            gain = r["in_band"] - u["in_band"]
            rub = ("reference" if a == "unguided" else
                   ("gain %+.3f" % gain) if floor[a] else "FAILS floor")
            L.append("| %s | %g | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f "
                     "| %.3f | %.3f | %s | %s |"
                     % (a, r["w"], r["in_band"], r["se_ib"], r["seed_sd_ib"],
                        r["mae"] / dl, r["se_mae"] / dl, abs(r["bias"]) / dl,
                        r["resid_sd"] / dl, r["mol_stab"], r["validity"],
                        "ok" if floor[a] else "NO", rub))

        L.append("")
        L.append("FR5 -- %s minus each arm. d in_band: + favours %s. d MAE: - "
                 "favours %s. Verdict on in_band sigma_ind >= %g, both arms "
                 "above the floor." % (OURS, OURS, OURS, SIGMA))
        L.append("")
        L.append("| vs | d in_band | sigma_ind | sigma_pair | d MAE/d | sigma_ind "
                 "| sigma_pair | verdict |")
        L.append("|---|---|---|---|---|---|---|---|")
        b = P[OURS]
        for a in ARMS:
            if a == OURS:
                continue
            c = P[a]
            dib = b["in_band"] - c["in_band"]
            zib = _z(dib, math.hypot(b["se_ib"], c["se_ib"]))
            dmae = b["mae"] - c["mae"]
            zmae = _z(dmae, math.hypot(b["se_mae"], c["se_mae"]))
            if PM[OURS] is not None and PM[a] is not None:
                pib = "%+.2f" % paired_z(PM[OURS], PM[a], dl, "ib")
                pmae = "%+.2f" % paired_z(PM[OURS], PM[a], dl, "mae")
            else:
                pib = pmae = "n/a"
            L.append("| %s | %+.4f | %+.2f | %s | %+.3f | %+.2f | %s | %s |"
                     % (a, dib, zib, pib, dmae / dl, zmae, pmae,
                        verdict(zib, zmae, floor[OURS], floor[a], a)))

        if PM["unguided"] is not None and all(PM[a] is not None for a in ARMS):
            ref = size_reference(PM["unguided"])
            y = PM["unguided"]["y"].double()
            sd_y = y.std().item()
            dist = (y - ref).abs() / sd_y
            share = [(dist >= lo) & (dist < hi) for lo, hi in zip(BINS, BINS[1:])]
            L.append("")
            L.append("Steering breakdown: in_band by |target - unguided mean at "
                     "that atom count| / sd(targets) (sd %.4g). Mean distance "
                     "%.3f sd." % (sd_y, dist.mean().item()))
            L.append("")
            hdr = ["[%g, %g) sd" % (lo, hi) for lo, hi in zip(BINS, BINS[1:])]
            L.append("| arm | " + " | ".join(hdr) + " |")
            L.append("|---|" + "---|" * len(hdr))
            L.append("| share of targets | " + " | ".join(
                "%.2f" % s.double().mean().item() for s in share) + " |")
            for a in ARMS:
                pm = PM[a]
                ib = ((pm["f_B"] - pm["y"]).abs() <= dl) & pm["finite"]
                L.append("| %s | " % a + " | ".join(
                    ("%.3f" % ib[s].double().mean().item()) if s.any() else "-"
                    for s in share) + " |")

    # SECONDARY: FR3 as registered (unconstrained best MAE), seed 1 only
    if WM:
        s1 = want[0]
        L.append("")
        L.append("## SECONDARY -- FR3 as registered (unconstrained best-MAE "
                 "strength), seed %s only" % s1)
        L.append("")
        L.append("The pre-registered choice, kept so the registered analysis is "
                 "reported. NOT the headline: at these strengths several arms "
                 "fail the chemistry floor. FR5 is not applied here.")
        for p in props:
            got = {a: cells.get(s1, {}).get((p, a, float(WM[a][p]))) for a in ARMS}
            if not all(got.values()):
                L.append("")
                L.append("%s: incomplete -- missing %s" % (p, ", ".join(
                    "%s@w%g" % (a, WM[a][p]) for a in ARMS if not got[a])))
                continue
            u = got["unguided"][0]
            dl = u["delta"]
            L.append("")
            L.append("| %s | w | in_band | MAE/d | mol_stab | chem floor |" % p)
            L.append("|---|---|---|---|---|---|")
            for a in ARMS:
                r = got[a][0]
                L.append("| %s | %g | %.3f | %.3f | %.3f | %s |" % (
                    a, r["w"], r["in_band_fraction"], r["prop_mae_eval"] / dl,
                    r["mol_stability"],
                    "ok" if r["mol_stability"] >= FLOOR * u["mol_stability"] - 1e-12 else "NO"))

    text = NL.join(L)
    print(text)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + NL)
        print(NL + "wrote %s" % args.md_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
