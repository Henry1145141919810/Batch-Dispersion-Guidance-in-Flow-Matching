"""Tabulate the EquiFM transfer tree (results/transfer_equifm/).

Why this exists, and what it does NOT do. The two scripts the runbook names
stay the authority:

  * `full_run_table.py` is the pre-registered FR5 readout (btvg against every
    arm, independent z >= 3). It refuses a partial run, which is correct, and
    its `--seeds` flag is the supported way to read the one complete seed.
  * `dist_report.py` is the literature ladder and the Holm-adjusted paired
    family (each arm vs unguided and vs plug).

Neither prints four things this run's write-up needs, so they are here and
nowhere else:

  1. atom_stability and the two uniqueness columns (on disk in every cell,
     never tabulated);
  2. the SECOND oracle (OC-Flow, the `oracle2` block) in-band, as a
     robustness column beside f_B's;
  3. a head-to-head of EVERY guided arm against unguided AND against plug
     with an explicit difference and z -- independent and paired -- on the
     continuous AND the decoded view. `full_run_table.py`'s family is
     btvg-vs-all; `dist_report.py`'s gives Holm p-values, not z;
  4. cost: the `seconds` field of every cell, summed per stage.

It also prints the compare-stage strength screen with the edge/floor status
of each pick, and the eqtune coverage. It NEVER re-decides FR5.

    python proj1/scripts/transfer_equifm_table.py --stage full  --seeds 20261001
    python proj1/scripts/transfer_equifm_table.py --stage compare --target q90
    python proj1/scripts/transfer_equifm_table.py --stage eqtune
    python proj1/scripts/transfer_equifm_table.py --stage cost

PARTIAL BY CONSTRUCTION. This tree is incomplete (the cluster cancelled the
run mid-way). Every stage section begins with a COVERAGE line stating cells
on disk against cells planned, and names what is missing. A table built on
fewer seeds than planned says so in its own caption.

Paired tests are computed only after `mol_idx` AND `n_atoms` are checked to
be identical between the two arms' sidecars; a mismatch prints `n/a` rather
than a number.
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

TREE = os.path.join(ROOT, "results", "transfer_equifm")
PROPS = ("mu", "alpha", "gap")
# the transfer's arm order: all seven, tfg included (frozen before any cell
# ran -- frozen_q90.json carries tfg_post_hoc = false)
ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var")
GUIDED = tuple(a for a in ARMS if a != "unguided")
FLOOR = 0.9            # v2 rubric: mol_stability >= 0.9 x unguided
BASE_GRID = (0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0)
EQ_GRID = (0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


# ----------------------------------------------------------------- loading
def _cells(pattern):
    """[(row, sidecar_path_or_None)] for every cell json matching `pattern`."""
    out = []
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as fh:
            r = json.load(fh)
        side = fn[:-5] + ".permol.pt"
        out.append((r, side if os.path.exists(side) else None))
    return out


def full_cells(root, seeds):
    """{(seed, prop, arm, w): (row, side)} over the seed dirs asked for."""
    out = {}
    for s in seeds:
        for r, side in _cells(os.path.join(root, "seed" + s, "*__full.json")):
            out[(s, r["prop"], r["arm"], float(r["w"]))] = (r, side)
    return out


def _load_side(path):
    import torch
    return torch.load(path, weights_only=False)


# ------------------------------------------------------------------- stats
def _z(d, se):
    return d / se if se and se > 0 else float("nan")


def _fmt_z(z):
    return "n/a" if z != z else "%+.2f" % z


def prop_se(p, n):
    return math.sqrt(max(p * (1.0 - p), 0.0) / n) if n else float("nan")


def mae_se(mae, rmse, n):
    return math.sqrt(max(rmse ** 2 - mae ** 2, 0.0) / n) if n else float("nan")


def pool(rows, key):
    """n-weighted mean of `key` over cells (one per seed)."""
    N = sum(r["n"] for r in rows)
    return sum(r[key] * r["n"] for r in rows) / N, N


def pool_rmse(rows):
    N = sum(r["n"] for r in rows)
    return math.sqrt(sum(r["prop_rmse_eval"] ** 2 * r["n"] for r in rows) / N)


def cat_sides(paths):
    """Concatenate sidecars in the given (fixed) order, or None if any is
    missing. Loaded one at a time and stacked, so peak memory is one run's
    worth of 5000-row float vectors, not the whole tree's."""
    import torch
    if any(p is None for p in paths):
        return None
    keys = ("f_B", "f_B_dec", "f_B2", "f_B2_dec", "y", "finite",
            "mol_idx", "n_atoms", "mol_stable", "valid")
    acc = {k: [] for k in keys}
    for p in paths:
        d = _load_side(p)
        for k in keys:
            acc[k].append(d[k])
        del d
    return {k: torch.cat(v) for k, v in acc.items()}


def pairable(a, b):
    """True only when the two sidecars describe the SAME molecules in the
    SAME order -- mol_idx and n_atoms both. Anything else is not a pair."""
    import torch
    return (a is not None and b is not None
            and a["mol_idx"].shape == b["mol_idx"].shape
            and bool(torch.equal(a["mol_idx"], b["mol_idx"]))
            and bool(torch.equal(a["n_atoms"], b["n_atoms"])))


def ib_vec(pm, delta, view):
    f = pm["f_B_dec"] if view == "dec" else pm["f_B"]
    if view == "o2":
        f = pm["f_B2"]
    if view == "o2dec":
        f = pm["f_B2_dec"]
    return ((f - pm["y"]).abs() <= delta) & pm["finite"]


def paired_ib_z(a, b, delta, view):
    d = ib_vec(a, delta, view).double() - ib_vec(b, delta, view).double()
    se = d.std(unbiased=True).item() / math.sqrt(d.numel())
    return _z(d.mean().item(), se)


def paired_mae_z(a, b, view):
    f = "f_B_dec" if view == "dec" else "f_B"
    ok = a["finite"] & b["finite"]
    d = ((a[f] - a["y"]).abs() - (b[f] - b["y"]).abs())[ok].double()
    se = d.std(unbiased=True).item() / math.sqrt(d.numel())
    return _z(d.mean().item(), se)


# -------------------------------------------------------------- full stage
def stage_full(L, args):
    root = os.path.join(TREE, "full", "n%d" % args.n)
    fzp = args.frozen or os.path.join(root, "frozen_q90.json")
    if not os.path.exists(fzp):
        L.append("no frozen-strength file at %s" % fzp)
        return
    with open(fzp) as fh:
        fz = json.load(fh)
    W = fz["frozen_w"]
    seeds = [s for s in args.seeds.split(",") if s]
    planned = [s for s in args.planned_seeds.split(",") if s]
    C = full_cells(root, planned)

    want = [(s, p, a) for s in planned for p in PROPS for a in ARMS]
    have = [k for k in want if (k[0], k[1], k[2], float(W[k[2]][k[1]])) in C]
    miss = [k for k in want if k not in have]
    L.append("COVERAGE: %d of %d planned primary cells on disk "
             "(%d seeds x %d properties x %d arms)."
             % (len(have), len(want), len(planned), len(PROPS), len(ARMS)))
    if miss:
        by_seed = {}
        for s, p, a in miss:
            by_seed.setdefault(s, []).append("%s/%s" % (p, a))
        for s in sorted(by_seed):
            L.append("  MISSING seed %s (%d): %s"
                     % (s, len(by_seed[s]), ", ".join(by_seed[s])))
    L.append("")
    L.append("Tables below use seed(s) **%s** only -- the complete one(s). "
             "Every arm in a table therefore shares the same molecules and "
             "the same starting noise." % ", ".join(seeds))

    for prop in PROPS:
        rows, sides, gone = {}, {}, []
        for a in ARMS:
            w = float(W[a][prop])
            got = [C.get((s, prop, a, w)) for s in seeds]
            if any(g is None for g in got):
                gone.append("%s@w%g" % (a, w))
                continue
            rows[a] = [g[0] for g in got]
            sides[a] = [g[1] for g in got]
        if "unguided" not in rows:
            L.append("")
            L.append("## %s -- NOT REPORTABLE: no unguided cell at these seeds"
                     % prop)
            continue
        delta = rows["unguided"][0]["delta"]
        d2 = (rows["unguided"][0].get("oracle2") or {}).get("delta")
        L.append("")
        L.append("## %s  (delta = %.5g; oracle-2 delta = %s)"
                 % (prop, delta, ("%.5g" % d2) if d2 else "n/a"))
        if gone:
            L.append("")
            L.append("Arms absent at these seeds: %s." % ", ".join(gone))
        u_stab, _ = pool(rows["unguided"], "mol_stability")

        L.append("")
        L.append("| arm | w | n | in_band | +-se | in_band_dec | O2 in_band | "
                 "O2 in_band_dec | MAE/d | +-se | MAE_dec/d | bias/d | "
                 "resid sd/d | mol_stab | atom_stab | validity | uniq(valid) | "
                 "uniq/sample | chem floor |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        P = {}
        for a in ARMS:
            if a not in rows:
                continue
            rs = rows[a]
            ib, N = pool(rs, "in_band_fraction")
            ibd, _ = pool(rs, "in_band_fraction_dec")
            mae, _ = pool(rs, "prop_mae_eval")
            maed, _ = pool(rs, "prop_mae_eval_dec")
            rmse = pool_rmse(rs)
            o2 = [r.get("oracle2") or {} for r in rs]
            o2ib = (sum(o.get("in_band", float("nan")) * r["n"]
                        for o, r in zip(o2, rs)) / N) if o2 and "in_band" in o2[0] else float("nan")
            o2ibd = (sum(o.get("in_band_dec", float("nan")) * r["n"]
                         for o, r in zip(o2, rs)) / N) if o2 and "in_band_dec" in o2[0] else float("nan")
            pm = cat_sides(sides[a])
            if pm is not None and bool(pm["finite"].any()):
                e = (pm["f_B"] - pm["y"])[pm["finite"]].double()
                bias, resid = e.mean().item(), e.std(unbiased=False).item()
            else:
                bias = resid = float("nan")
            ms, _ = pool(rs, "mol_stability")
            ats, _ = pool(rs, "atom_stability")
            val, _ = pool(rs, "validity")
            uv, _ = pool(rs, "uniqueness_of_valid")
            uvs, _ = pool(rs, "unique_valid_per_sample")
            ok = ms >= FLOOR * u_stab - 1e-12
            P[a] = {"ib": ib, "ibd": ibd, "N": N, "mae": mae, "maed": maed,
                    "rmse": rmse, "pm": pm, "floor": ok, "w": rs[0]["w"]}
            L.append("| %s | %g | %d | %.4f | %.4f | %.4f | %.4f | %.4f | "
                     "%.3f | %.3f | %.3f | %+.3f | %.3f | %.4f | %.4f | "
                     "%.4f | %.4f | %.4f | %s |"
                     % (a, rs[0]["w"], N, ib, prop_se(ib, N), ibd, o2ib, o2ibd,
                        mae / delta, mae_se(mae, rmse, N) / delta, maed / delta,
                        bias / delta, resid / delta, ms, ats, val, uv, uvs,
                        "ok" if ok else "**NO**"))
        L.append("")
        L.append("Chemistry floor: mol_stability >= %.2f x unguided = %.4f."
                 % (FLOOR, FLOOR * u_stab))

        for ref in ("unguided", "plug"):
            if ref not in P:
                continue
            L.append("")
            L.append("Head-to-head vs **%s** (arm minus %s). + d in_band "
                     "favours the arm; - d MAE favours the arm. An arm below "
                     "the chemistry floor carries no verdict." % (ref, ref))
            L.append("")
            L.append("| arm | d in_band | z_ind | z_pair | d in_band_dec | "
                     "z_ind_dec | z_pair_dec | d MAE/d | z_ind | z_pair | "
                     "floor |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            b = P[ref]
            for a in ARMS:
                if a == ref or a not in P:
                    continue
                c = P[a]
                dib = c["ib"] - b["ib"]
                zib = _z(dib, math.hypot(prop_se(c["ib"], c["N"]),
                                         prop_se(b["ib"], b["N"])))
                dibd = c["ibd"] - b["ibd"]
                zibd = _z(dibd, math.hypot(prop_se(c["ibd"], c["N"]),
                                           prop_se(b["ibd"], b["N"])))
                dmae = c["mae"] - b["mae"]
                zmae = _z(dmae, math.hypot(mae_se(c["mae"], c["rmse"], c["N"]),
                                           mae_se(b["mae"], b["rmse"], b["N"])))
                if pairable(c["pm"], b["pm"]):
                    zp = _fmt_z(paired_ib_z(c["pm"], b["pm"], delta, "soft"))
                    zpd = _fmt_z(paired_ib_z(c["pm"], b["pm"], delta, "dec"))
                    zpm = _fmt_z(paired_mae_z(c["pm"], b["pm"], "soft"))
                else:
                    zp = zpd = zpm = "n/a"
                L.append("| %s | %+.4f | %s | %s | %+.4f | %s | %s | %+.3f | "
                         "%s | %s | %s |"
                         % (a, dib, _fmt_z(zib), zp, dibd, _fmt_z(zibd), zpd,
                            dmae / delta, _fmt_z(zmae), zpm,
                            "ok" if c["floor"] else "**NO**"))


# ----------------------------------------------------------- compare stage
def stage_compare(L, args):
    root = os.path.join(TREE, "compare")
    cells = _cells(os.path.join(root, "tr__*.json"))
    tgt = args.target
    sel = [r for r, _ in cells if r.get("target_name") == tgt]
    L.append("COVERAGE: %d compare cells on disk in all; %d at target %s "
             "(planned base grid: 6 guided arms x %d strengths + unguided, "
             "per property per target = %d; plus edge-rule extensions)."
             % (len(cells), len(sel), tgt, len(BASE_GRID),
                (6 * len(BASE_GRID) + 1) * len(PROPS)))
    for prop in PROPS:
        rs = [r for r in sel if r["prop"] == prop]
        if not rs:
            continue
        ung = [r for r in rs if r["arm"] == "unguided"]
        if not ung:
            L.append("")
            L.append("## %s (%s): no unguided cell" % (prop, tgt))
            continue
        u = ung[0]
        floor_v = FLOOR * u["mol_stability"]
        delta = u["delta"]
        L.append("")
        L.append("## %s, target %s  (delta = %.5g; floor = %.4f = %.2f x "
                 "unguided %.4f)"
                 % (prop, tgt, delta, floor_v, FLOOR, u["mol_stability"]))
        L.append("")
        L.append("| arm | strengths on disk | FR3a pick (floored) | "
                 "MAE_dec/d at pick | in_band_dec at pick | mol_stab at pick "
                 "| unconstrained best-MAE_dec w | status |")
        L.append("|---|---|---|---|---|---|---|---|")
        for a in GUIDED:
            ar = sorted([r for r in rs if r["arm"] == a],
                        key=lambda r: float(r["w"]))
            if not ar:
                continue
            ws = [float(r["w"]) for r in ar]
            keep = [r for r in ar
                    if r["mol_stability"] >= floor_v - 1e-12]
            best_uncon = min(ar, key=lambda r: r["prop_mae_eval_dec"])
            if not keep:
                L.append("| %s | %s | none clears the floor | - | - | - | %g "
                         "| `floor_excluded` |"
                         % (a, " ".join("%g" % w for w in ws),
                            float(best_uncon["w"])))
                continue
            pick = min(keep, key=lambda r: r["prop_mae_eval_dec"])
            wp = float(pick["w"])
            status = []
            if wp == max(ws) or wp == min(ws):
                status.append("`grid_edge`")
            if float(best_uncon["w"]) != wp:
                status.append("`floor_limited`")
            if pick.get("extension"):
                status.append("`on_extension`")
            L.append("| %s | %s | **%g** | %.3f | %.4f | %.4f | %g | %s |"
                     % (a, " ".join("%g" % w for w in ws), wp,
                        pick["prop_mae_eval_dec"] / delta,
                        pick["in_band_fraction_dec"], pick["mol_stability"],
                        float(best_uncon["w"]),
                        " ".join(status) if status else "interior"))


# ------------------------------------------------------------ eqtune stage
def stage_eqtune(L, args):
    root = os.path.join(TREE, "eqchem", "tune")
    cells = [r for r, _ in _cells(os.path.join(root, "tr__*.json"))]
    planned = (6 * len(EQ_GRID) + 1) * len(PROPS)
    L.append("COVERAGE: %d of %d planned eqtune cells on disk."
             % (len(cells), planned))
    frozen = os.path.join(TREE, "eqchem", "frozen_eqtune.json")
    L.append("`frozen_eqtune.json`: **%s**. eqextend, eqfreeze and eqconfirm "
             "left no output, so the equal-chemistry frontier this stage "
             "exists to produce was never frozen."
             % ("present" if os.path.exists(frozen) else "ABSENT"))
    for prop in PROPS:
        rs = [r for r in cells if r["prop"] == prop]
        got = {}
        for r in rs:
            got.setdefault(r["arm"], []).append(float(r["w"]))
        missing = []
        for a in GUIDED:
            have = set(got.get(a, []))
            gap = [w for w in EQ_GRID if w not in have]
            if gap:
                missing.append("%s(%s)" % (a, ",".join("%g" % w for w in gap)))
        if "unguided" not in got:
            missing.append("unguided")
        n_have = sum(len(v) for v in got.values())
        L.append("")
        L.append("- **%s**: %d of %d cells. %s"
                 % (prop, n_have, 6 * len(EQ_GRID) + 1,
                    "complete." if not missing
                    else "missing " + "; ".join(missing)))
        if missing:
            continue
        u = [r for r in rs if r["arm"] == "unguided"][0]
        floor_v = FLOOR * u["mol_stability"]
        delta = u["delta"]
        L.append("")
        L.append("  POST HOC (eqfreeze never ran): the stability/in-band "
                 "frontier this grid supports, floor = %.4f, delta = %.5g."
                 % (floor_v, delta))
        L.append("")
        L.append("| arm | w at best floored MAE_dec | in_band_dec there | "
                 "mol_stab there | highest w still above floor | "
                 "in_band_dec there |")
        L.append("|---|---|---|---|---|---|")
        for a in GUIDED:
            ar = sorted([r for r in rs if r["arm"] == a],
                        key=lambda r: float(r["w"]))
            keep = [r for r in ar if r["mol_stability"] >= floor_v - 1e-12]
            if not keep:
                L.append("| %s | none clears the floor | - | - | - | - |" % a)
                continue
            pick = min(keep, key=lambda r: r["prop_mae_eval_dec"])
            top = max(keep, key=lambda r: float(r["w"]))
            L.append("| %s | %g | %.4f | %.4f | %g | %.4f |"
                     % (a, float(pick["w"]), pick["in_band_fraction_dec"],
                        pick["mol_stability"], float(top["w"]),
                        top["in_band_fraction_dec"]))


# -------------------------------------------------------------------- cost
def stage_cost(L, args):
    groups = (
        ("compare + extend", os.path.join(TREE, "compare", "tr__*.json")),
        ("eqtune", os.path.join(TREE, "eqchem", "tune", "tr__*.json")),
        ("full (n=5000)", os.path.join(TREE, "full", "n%d" % args.n,
                                       "seed*", "*__full.json")),
    )
    L.append("| stage | cells on disk | GPU-seconds | GPU-hours | "
             "mean s/cell | max s/cell |")
    L.append("|---|---|---|---|---|---|")
    tot = 0.0
    ncell = 0
    for name, pat in groups:
        secs = []
        for fn in sorted(glob.glob(pat)):
            with open(fn) as fh:
                r = json.load(fh)
            if "seconds" in r:
                secs.append(float(r["seconds"]))
        if not secs:
            L.append("| %s | 0 | - | - | - | - |" % name)
            continue
        s = sum(secs)
        tot += s
        ncell += len(secs)
        L.append("| %s | %d | %.0f | **%.1f** | %.0f | %.0f |"
                 % (name, len(secs), s, s / 3600.0, s / len(secs), max(secs)))
    L.append("| **total on disk** | %d | %.0f | **%.1f** | | |"
             % (ncell, tot, tot / 3600.0))
    L.append("")
    L.append("`seconds` is the cell's own wall clock on the Betty slice, so "
             "this is GPU-hours of work that LANDED, not hours the queue "
             "charged: the cancelled cells burned time and wrote nothing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("full", "compare", "eqtune", "cost"))
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seeds", default="20261001",
                    help="seeds to TABULATE (must be complete)")
    ap.add_argument("--planned-seeds", default="20261001,20261002,20261003",
                    help="seeds the protocol planned, for the coverage line")
    ap.add_argument("--target", default="q90", choices=("q50", "q90"))
    ap.add_argument("--frozen", default="")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    L = []
    {"full": stage_full, "compare": stage_compare,
     "eqtune": stage_eqtune, "cost": stage_cost}[args.stage](L, args)
    text = NL.join(L)
    print(text)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + NL)
        print(NL + "wrote %s" % args.md_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
