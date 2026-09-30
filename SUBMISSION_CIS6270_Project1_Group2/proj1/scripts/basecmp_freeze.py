"""Pick each arm's (strength, start-time) on one base model, two ways, and
store the whole screen as a table.

    # after the screen: what to refine
    python proj1/scripts/basecmp_freeze.py --backend equifm \
        --emit-refine results/basecmp/equifm/refine.json

    # after the refine: the strengths the full run uses, plus the table
    python proj1/scripts/basecmp_freeze.py --backend equifm \
        --json-out results/basecmp/equifm/frozen_basecmp.json \
        --table-out results/basecmp/equifm/screen_table.csv \
        --md-out docs/results/BASECMP_SCREEN_equifm.md

WHAT IT DECIDES. For every (property, arm) it reports TWO picks over the joint
(w, t_start) grid, because Henry asked for both and because they answer
different questions:

  floor   the best in-band among cells whose molecule stability is at least
          0.9 x the unguided generator's. This is v2's rule V2 and the only
          pick a full run may be headlined on: it is guidance bought at a
          chemistry price the reader has agreed to.
  free    the best in-band with the floor removed. This is NOT a result to
          headline -- at high strength these cells are often chemical rubble
          that happens to satisfy a property oracle -- but it is the honest
          answer to "how far can this method be pushed", and without it a
          method that is merely throttled by the floor is indistinguishable
          from one that cannot steer at all.

THE SELECTION METRIC IS IN-BAND, DECODED. v2's rule V2 exists because v1
selected on MAE and scored on in-band, which is a mismatch with no upside.
Decoded (`in_band_fraction_dec`), because an arm can push the continuous atom
type channels without moving the molecule that actually exists; the soft number
is reported beside it in the table, never selected on.

TIE-BREAKING, fixed here before any cell was read: smaller strength first, then
LARGER t_start. Both directions mean "less intervention" -- a later switch-on
guides fewer steps -- so a tie is always resolved toward the cheaper, gentler
cell. v2's rule already sends strength ties to the smaller strength; this
extends the same principle to the new axis rather than inventing a second one.

SELECTION IS NOT EVALUATION. These picks are made on the screen (n = 1000, one
seed). The full run scores fresh seeds at larger n, so a pick cannot inflate
the number it is later reported at.

WHAT IS *NOT* CLAIMED. The grid is coarse in strength by design (five
log-spaced points spanning 0.05 to 16, refined around each pick), so a pick is
"the best cell on this grid, refined once", not a continuous optimum. The
`grid_edge` status flags any pick that sits at the top or bottom of the grid,
because there the true optimum may be outside it -- that is exactly how v1's
w = 4 ceiling capped lgd_mc, and it is reported rather than hidden.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from transfer_sweep import (BACKENDS, BASECMP_FLOOR, BASECMP_ROOT,  # noqa: E402
                            BASECMP_SELECT, BASECMP_SETS, BASECMP_STRENGTHS,
                            BASECMP_T_STARTS, BASECMP_TARGET, EXT_FACTORS,
                            backend_arms)

RC_OK, RC_INCOMPLETE, RC_ERROR = 0, 2, 3
# v2's V6 threshold: below this, in-band may be collapse rather than steering
UNIQ_MIN = 0.95
# delta is recomputed inside every array task (`local_delta`, float32 on the
# GPU), so one property's cells legitimately differ in about the 7th
# significant figure -- measured on the 25 Sep screen, relative spread <= 9.0e-7
# on both bases. Exact equality at 12 decimals therefore refused every real
# screen (and the chain's STAGE=plan with it). A real change of rule is orders
# larger (local vs global delta differ by 5-60 %), so 1e-5 separates the two.
DELTA_RTOL = 1e-5


def delta_consistent(ds, rtol=DELTA_RTOL):
    """True when every delta in `ds` agrees within float noise."""
    ds = [float(x) for x in ds]
    if len(ds) < 2:
        return True
    lo, hi = min(ds), max(ds)
    return (hi - lo) <= rtol * max(abs(lo), abs(hi))


def se_prop(p, n):
    """Binomial se of a fraction. in-band is a fraction of n samples."""
    p = min(max(float(p), 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / max(int(n), 1))


def load_screen(out_dir, props, arms, backend, require_grid=True):
    """(rows, meta) or an int exit code. Reads screen AND refine cells: both
    carry stage == "basecmp", deliberately, so one glob is the whole grid.

    `require_grid` refuses a PARTIAL screen. This is not optional hygiene: the
    picks are an argmax, so a missing cell cannot be detected from the result --
    it just silently is not chosen, and the reported pick is the best of whatever
    finished first. A job killed by the QOS limit would therefore produce a
    plausible-looking frozen file describing a grid that was never run. Extra
    cells beyond the base grid (the refine stage's) are always welcome; holes in
    it are not."""
    if not os.path.isdir(out_dir):
        print("no cells: %s does not exist" % out_dir)
        return RC_INCOMPLETE
    rows, meta, bad = [], set(), []
    for fn in sorted(glob.glob(os.path.join(out_dir, "tr__*.json"))):
        r = json.load(open(fn))
        if r.get("stage") != "basecmp" or r.get("prop") not in props:
            continue
        if r.get("backend") != BACKENDS[backend]:
            print("REFUSING: %s is a %r cell; this is --backend %s"
                  % (os.path.basename(fn), r.get("backend"), backend))
            return RC_INCOMPLETE
        if r.get("target_name") != BASECMP_TARGET:
            print("REFUSING: %s is at target %r, not %r"
                  % (os.path.basename(fn), r.get("target_name"), BASECMP_TARGET))
            return RC_INCOMPLETE
        if r.get("arm") not in arms:
            continue
        # Everything that must be constant across the grid. n and seed are in
        # here for the reason cell_name carries them: a cell at another n is a
        # different measurement, and silently mixing them would make the pick
        # depend on which jobs happened to finish.
        meta.add((r["n"], r["seed"], (r.get("prov") or {}).get("gen_md5"),
                  r.get("batch"), r.get("steps"), r.get("solver"),
                  r.get("clip"),
                  (r.get("calibration") or {}).get("delta_mode"),
                  round(float(r.get("delta", float("nan"))), 12)))
        r["_file"] = os.path.basename(fn)
        if r.get("n_nonfinite", 0) > 0 or not math.isfinite(
                float(r.get(BASECMP_SELECT, float("nan")))):
            bad.append(r)
            continue
        rows.append(r)
    if not rows:
        print("no usable basecmp cells in %s" % out_dir)
        return RC_INCOMPLETE
    # delta is per property, so the tuple above cannot be globally unique;
    # check the non-delta part is, and check delta is unique PER property.
    core = {m[:8] for m in meta}
    if len(core) != 1:
        print("INCONSISTENT basecmp cells in %s: (n, seed, gen_md5, batch, "
              "steps, solver, clip, delta_mode) = %s"
              % (out_dir, sorted(map(str, core))))
        return RC_INCOMPLETE
    for p in props:
        ds = {round(float(r["delta"]), 12) for r in rows if r["prop"] == p}
        if not delta_consistent(ds):
            print("INCONSISTENT delta for %s: %s -- in-band means a different "
                  "thing in each cell" % (p, sorted(ds)))
            return RC_INCOMPLETE
    if bad:
        print("%d cell(s) ran but diverged; excluded from every pick:" % len(bad))
        for r in bad[:8]:
            print("   %s" % r["_file"])
    if require_grid:
        # a diverged cell counts as PRESENT: it ran, it is just not eligible to
        # be anyone's best. A missing cell is a hole.
        have = {(r["prop"], r["arm"], float(r["w"]), float(r["t_start"]))
                for r in rows + bad}
        want = {(p, a, float(w), float(t)) for p in props
                for a in arms if a != "unguided"
                for w in BASECMP_STRENGTHS for t in BASECMP_T_STARTS}
        want |= {(p, "unguided", 1.0, 0.5) for p in props if "unguided" in arms}
        holes = sorted(want - have)
        if holes:
            print("INCOMPLETE -- %d of %d base-grid cells missing, e.g. %s"
                  % (len(holes), len(want),
                     ", ".join("%s/%s/w%g/t%g" % h for h in holes[:6])))
            print("The picks are an argmax: a hole cannot be seen in the result, "
                  "it just never wins. Finish the screen (the insurance sweeper "
                  "does), or pass --allow-partial and say so in the write-up.")
            return RC_INCOMPLETE
    return rows, core.pop()


def scale_sanity(rows, props, data_path=None, k_mad=1.0):
    """Warn if the unguided generator's f_B mean is nowhere near QM9's.

    THE HOLE THIS FILLS. `build_pair` guards the feature scale with a
    calibration-slope check (0.9-1.1 x QM9's MAD), but that calibration is fitted
    on REAL molecules read straight from the data file. It therefore cannot see a
    wrong `sampler_scale`, which only affects what is fed to f_A and f_B at
    SCORING time, on generated molecules. A wrong divisor there would leave every
    slope perfectly in range and every number downstream quietly wrong.

    The detectable signature is the one that caught the same bug before (this
    file's header, caveat on `build_pair`): the oracle's mean on unguided samples
    drifts far from QM9's own mean -- measured at 4.15 D on mu against QM9's 2.49
    when the oracle was reading 2 x one-hot. So: compare the unguided cell's
    `f_B_mean` against QM9's train_a mean, in units of that property's MAD.

    THE THRESHOLD IS MEASURED, NOT GUESSED. Across the 80 unguided cells this
    project has already run -- on two different generators, ours and EDMsecond --
    the oracle's mean sits within **0.62 MAD** of QM9's train_a mean, median 0.17.
    The historical scale bug sat at 1.39 MAD (4.15 D against QM9's 2.49 on mu),
    and a clean factor-of-2 error on mu lands near 2.25. So 1.0 MAD separates
    "this generator is a bit off QM9", which is a result, from "the features are
    on the wrong scale", which is a defect, with margin on both sides.

    Advisory, not a refusal: an unguided generator legitimately differs from QM9,
    and how much is exactly what the unguided row is there to report. EquiFM has
    never been measured here, so it could in principle sit further out than any
    generator we have seen -- which is why this prints a line to read rather than
    stopping the run."""
    import math as _m
    path = data_path or os.path.join(ROOT, "data", "qm9.pt")
    if not os.path.exists(path):
        return ["(skipped: %s not present, so QM9's own means are unknown)" % path]
    try:
        import torch
        d = torch.load(path, weights_only=False)
    except Exception as exc:                                   # noqa: BLE001
        return ["(skipped: could not read %s: %r)" % (path, exc)]
    tr = d["split"]["train_a"]
    out = []
    for p in props:
        u = [r for r in rows if r["prop"] == p and r["arm"] == "unguided"]
        if not u or u[0].get("f_B_mean") is None:
            continue
        i = d["props"].index(p)
        y = d["y"][tr, i].double()
        mean, mad = float(y.mean()), float((y - y.mean()).abs().mean())
        got = float(u[0]["f_B_mean"])
        off = abs(got - mean) / max(mad, 1e-12)
        flag = "  <-- CHECK THE FEATURE SCALE" if off > k_mad else ""
        line = ("%-6s unguided f_B mean %9.4f vs QM9 train_a %9.4f  "
                "(%.2f MAD)%s" % (p, got, mean, off, flag))
        if _m.isfinite(off):
            out.append(line)
    return out


def edge_flags(best, cands):
    """Is this pick at the edge of the grid -- on EITHER axis?

    TWO THINGS THIS GETS RIGHT that the obvious version does not.

    1. THE START-TIME AXIS COUNTS. A pick at t_start = 0.05 or 0.75 sits at a
       boundary of the screened window grid just as much as one at w = 0.05 or 16,
       and the protocol calls the window the project's largest measured single
       effect. Reporting only the strength edge would re-create the v1 w = 4
       ceiling failure on the axis most likely to have its optimum outside the
       grid. The t grid is never extended (t is bounded by the trajectory itself),
       so a t edge is a permanent caveat, not a to-do.

    2. THE STRENGTH GRID IS READ AT THE PICK'S OWN WINDOW. The refine stage adds
       strengths at the window each pick landed at, so after a refine the grid is
       wider at some windows than others. Pooling every window would then hide an
       edge exactly where it matters: if the screen picked (w=16, t=0.5), the
       refine added w=64 at t=0.5 only, and the re-freeze now picks (w=16,
       t=0.75), then 16 is interior to the POOLED strengths but is still the top
       of everything ever run at t = 0.75. Comparing within the window reports
       that honestly.
    """
    if best is None or best.get("arm") == "unguided":
        return {"w_edge": False, "t_edge": False, "w_grid_at_t": []}
    t0 = float(best["t_start"])
    same_t = [r for r in cands
              if abs(float(r["t_start"]) - t0) < 1e-12]
    ws = sorted({float(r["w"]) for r in same_t})
    tss = sorted({float(r["t_start"]) for r in cands})
    return {"w_edge": len(ws) > 1 and float(best["w"]) in (ws[0], ws[-1]),
            "t_edge": len(tss) > 1 and t0 in (tss[0], tss[-1]),
            "w_grid_at_t": ws}


def pick(cands, floor, constrained):
    """(row, status, flags) -- the joint (w, t_start) argmax under the rule.

    `constrained` applies the chemistry floor. Status is "ok", "floor_limited"
    (nothing on the grid clears the floor, so the most stable cell is reported and
    must never be headlined as a win) or "grid_edge" (the pick sits at an extreme
    of the screened grid on either axis, so the optimum may lie outside it).

    `flags` carries the two edges separately and is populated for EVERY pick,
    including a floor-limited one -- an arm can be both out of chemistry budget
    and at the edge of the grid, and an earlier version returned before checking,
    so those picks carried neither warning."""
    if not cands:
        return None, "no_cell", edge_flags(None, cands)
    pool, fell = cands, False
    if constrained:
        ok = [r for r in cands if r["mol_stability"] >= floor - 1e-12]
        if ok:
            pool = ok
        else:
            fell = True
    if fell:
        best = max(cands, key=lambda r: (r["mol_stability"], -float(r["w"]),
                                         float(r["t_start"])))
    else:
        # maximise in-band; ties -> smaller w, then LARGER t_start
        best = max(pool, key=lambda r: (float(r[BASECMP_SELECT]), -float(r["w"]),
                                        float(r["t_start"])))
    fl = edge_flags(best, cands)
    if fell:
        return best, "floor_limited", fl
    return best, ("grid_edge" if (fl["w_edge"] or fl["t_edge"]) else "ok"), fl


# The refine stage will not go outside this range. The top matters: the screen
# already reaches w = 16, and extending an edge pick by x16 would ask for
# w = 256, where the sampler produces chemical rubble no oracle reading makes
# interesting. A pick that still sits at the edge after the refine is reported
# as `grid_edge` instead, which is the honest answer.
W_REFINE_MIN, W_REFINE_MAX = 1e-3, 64.0


def neighbours(w, grid, factors=EXT_FACTORS):
    """Strengths to add around a pick.

    Interior: the geometric midpoints to the neighbours on the log grid, which
    is where a coarse log grid hides its optimum. At an edge: the grid is
    EXTENDED outward by `factors` instead, because there is no neighbour to
    bisect and the optimum may simply be further out -- the failure that capped
    lgd_mc at w = 4 in v1. Clamped to [W_REFINE_MIN, W_REFINE_MAX]."""
    ws = sorted(set(float(x) for x in grid))
    if w not in ws:
        return []
    i = ws.index(w)
    out = []
    if i == 0:
        out += [w / f for f in factors]
    else:
        out.append(math.sqrt(w * ws[i - 1]))
    if i == len(ws) - 1:
        out += [w * f for f in factors]
    else:
        out.append(math.sqrt(w * ws[i + 1]))
    return [float("%.6g" % x) for x in out
            if W_REFINE_MIN <= x <= W_REFINE_MAX]


def t_neighbours(t, grid):
    """Arithmetic midpoints to the adjacent start-times. t is a time, not a
    scale, so midpoints are arithmetic; edges are not extended because t is
    bounded by the trajectory itself (0 noise, 1 data)."""
    ts = sorted(set(float(x) for x in grid))
    if t not in ts:
        return []
    i = ts.index(t)
    out = []
    if i > 0:
        out.append(0.5 * (t + ts[i - 1]))
    if i < len(ts) - 1:
        out.append(0.5 * (t + ts[i + 1]))
    return [float("%.6g" % x) for x in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", required=True, choices=["fm", "equifm"])
    ap.add_argument("--screen-dir", default="",
                    help="default results/basecmp/<backend>/screen")
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--arms", default="",
                    help="default the 7-arm comparison set")
    ap.add_argument("--json-out", default="", help="the frozen (w, t) picks")
    ap.add_argument("--table-out", default="", help="the whole screen, as csv")
    ap.add_argument("--md-out", default="", help="the whole screen, as markdown")
    ap.add_argument("--emit-refine", default="",
                    help="write the cells to add around each pick and stop; "
                         "no frozen file is produced by this mode")
    ap.add_argument("--refine-t", action="store_true",
                    help="also refine the start-time axis (adds the midpoints "
                         "to each pick's adjacent start-times). Off by default "
                         "because it roughly doubles the refine stage; the "
                         "4-point t grid is already a reported result.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="pick from an incomplete screen. Off by default: a hole "
                         "in an argmax is invisible in its result.")
    ap.add_argument("--allow-floor-limited", action="store_true",
                    help="freeze even if an arm never clears the chemistry "
                         "floor. Its `floor` pick is then its most stable cell "
                         "and is labelled floor_limited everywhere. No effect in "
                         "--emit-refine mode, which never refuses: a floor-limited "
                         "arm is exactly one whose neighbourhood is worth refining.")
    args = ap.parse_args()

    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a] or backend_arms(args.backend)
    screen = args.screen_dir or os.path.join(BASECMP_ROOT, args.backend, "screen")

    loaded = load_screen(screen, props, arms, args.backend,
                         require_grid=not args.allow_partial)
    if isinstance(loaded, int):
        return loaded
    rows, core = loaded
    n, seed, gen_md5, batch, steps, solver, clip, delta_mode = core
    print("%s: %d usable cells in %s  (n=%d seed=%d gen %s delta_mode=%s)"
          % (BACKENDS[args.backend], len(rows), screen, n, seed,
             str(gen_md5)[:12], delta_mode))

    # ---- the chemistry floor: from the ONE unguided cell per property ----
    floor, unguided = {}, {}
    for p in props:
        u = [r for r in rows if r["prop"] == p and r["arm"] == "unguided"]
        if not u:
            print("INCOMPLETE: no unguided cell for %s; the floor is undefined" % p)
            return RC_INCOMPLETE
        stabs = {round(float(r["mol_stability"]), 12) for r in u}
        if len(stabs) > 1:
            print("INCONSISTENT: %d unguided cells for %s disagree on stability "
                  "(%s)" % (len(u), p, sorted(stabs)))
            return RC_INCOMPLETE
        unguided[p] = float(u[0]["mol_stability"])
        floor[p] = BASECMP_FLOOR * unguided[p]
        print("  %-6s unguided mol-stability %.4f -> floor %.4f  (delta %.5f)"
              % (p, unguided[p], floor[p], float(u[0]["delta"])))

    # ---- the feature-scale sanity check, which nothing else covers ----
    sanity = scale_sanity(rows, props)
    if sanity:
        print("\nfeature-scale check (the calibration guard cannot see this):")
        for line in sanity:
            print("  %s" % line)

    # ---- the picks ----
    frozen = {s: {} for s in BASECMP_SETS}
    status = {s: {} for s in BASECMP_SETS}
    guided = [a for a in arms if a != "unguided"]
    for s in BASECMP_SETS:
        for a in guided:
            frozen[s][a], status[s][a] = {}, {}
            for p in props:
                cands = [r for r in rows if r["prop"] == p and r["arm"] == a]
                row, st, fl = pick(cands, floor[p], constrained=(s == "floor"))
                if row is None:
                    print("INCOMPLETE: no cell for %s/%s" % (a, p))
                    return RC_INCOMPLETE
                frozen[s][a][p] = {
                    "w": float(row["w"]), "t_start": float(row["t_start"]),
                    "w_edge": fl["w_edge"], "t_edge": fl["t_edge"],
                    "w_grid_at_this_t": fl["w_grid_at_t"],
                    "in_band_dec": float(row[BASECMP_SELECT]),
                    "in_band_soft": float(row["in_band_fraction"]),
                    "mol_stability": float(row["mol_stability"]),
                    "mae_dec": float(row.get("prop_mae_eval_dec", float("nan"))),
                    "unique_valid_per_sample": float(
                        row.get("unique_valid_per_sample", float("nan"))),
                    "n_grid_cells": len(cands), "cell": row["_file"]}
                status[s][a][p] = st

    limited = ["%s/%s" % (a, p) for a in guided for p in props
               if status["floor"][a][p] == "floor_limited"]
    # reported per axis, because they mean different things: a strength edge is
    # something the refine stage can chase, a start-time edge is a permanent
    # caveat (t is bounded by the trajectory, so there is nowhere to extend to)
    edges = ["%s/%s/%s%s" % (s, a, p, "".join(
                 [" w" if frozen[s][a][p]["w_edge"] else "",
                  " t" if frozen[s][a][p]["t_edge"] else ""]))
             for s in BASECMP_SETS for a in guided for p in props
             if frozen[s][a][p]["w_edge"] or frozen[s][a][p]["t_edge"]]
    collapsed = ["%s/%s/%s" % (s, a, p) for s in BASECMP_SETS for a in guided
                 for p in props
                 if frozen[s][a][p]["unique_valid_per_sample"] < UNIQ_MIN]

    # ---- the refine plan ----
    if args.emit_refine:
        want = set()
        for s in BASECMP_SETS:
            for a in guided:
                for p in props:
                    pk = frozen[s][a][p]
                    w0, t0 = pk["w"], pk["t_start"]
                    # the strength grid AT THIS WINDOW, not the base grid: on a
                    # second refine the base grid is no longer what exists there,
                    # and an edge decided against the wrong grid extends the wrong
                    # way (or not at all)
                    grid_here = pk.get("w_grid_at_this_t") or BASECMP_STRENGTHS
                    for w in neighbours(w0, grid_here):
                        want.add((p, a, w, t0))
                    if args.refine_t:
                        for t in t_neighbours(t0, BASECMP_T_STARTS):
                            want.add((p, a, w0, t))
        have = {(r["prop"], r["arm"], float(r["w"]), float(r["t_start"]))
                for r in rows}
        cells = sorted(want - have)
        plan = {"study": "basecmp", "backend": BACKENDS[args.backend],
                "stage": "basecmprefine", "source_screen": os.path.abspath(screen),
                "rule": "geometric midpoints around each pick on the strength "
                        "grid; an edge pick is extended outward by %s instead"
                        % (list(EXT_FACTORS),),
                "refine_t": bool(args.refine_t),
                "n_already_present": len(want & have),
                "cells": [{"prop": p, "arm": a, "w": w, "t_start": t}
                          for (p, a, w, t) in cells],
                "date": datetime.date.today().isoformat()}
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_refine)) or ".",
                    exist_ok=True)
        tmp = args.emit_refine + ".tmp"
        json.dump(plan, open(tmp, "w"), indent=1)
        os.replace(tmp, args.emit_refine)
        print("\nrefine plan: %d new cells (%d of the wanted set already exist)"
              % (len(cells), plan["n_already_present"]))
        for c in plan["cells"][:12]:
            print("   %-6s %-9s w=%-8g t=%g" % (c["prop"], c["arm"], c["w"],
                                                c["t_start"]))
        if len(cells) > 12:
            print("   ... %d more" % (len(cells) - 12))
        print("wrote %s" % args.emit_refine)
        return RC_OK

    # ---- report ----
    print("\nfrozen (w, t_start) per arm -- %s, ties to smaller w then larger t"
          % BASECMP_SELECT)
    print("  %-9s %-6s | %-22s | %-22s" % ("arm", "prop",
                                           "floor (>= 0.9x unguided)", "free (no floor)"))
    for a in guided:
        for p in props:
            f, g = frozen["floor"][a][p], frozen["free"][a][p]
            print("  %-9s %-6s | w=%-7g t=%-5g ib=%.3f%s | w=%-7g t=%-5g ib=%.3f%s"
                  % (a, p, f["w"], f["t_start"], f["in_band_dec"],
                     "*" if status["floor"][a][p] != "ok" else " ",
                     g["w"], g["t_start"], g["in_band_dec"],
                     "*" if status["free"][a][p] != "ok" else " "))
    if limited:
        print("\nFLOOR-LIMITED (no grid cell clears the floor; not a win): %s"
              % ", ".join(limited))
    if edges:
        print("GRID-EDGE picks (optimum may be outside the grid): %s"
              % ", ".join(edges))
    if collapsed:
        print("COLLAPSE-CONTAMINATED (unique valid per sample < %.2f): %s"
              % (UNIQ_MIN, ", ".join(collapsed)))

    # ---- the table: every cell, which is itself the result (v2's V2b) ----
    cols = ["backend", "prop", "arm", "w", "t_start", "n", "in_band_dec",
            "in_band_dec_se", "in_band_soft", "mol_stability", "clears_floor",
            "mae_dec", "mae_soft", "delta", "atom_stability", "validity",
            "uniqueness_of_valid", "unique_valid_per_sample",
            "guide_eval_gap_mean", "n_nonfinite", "clipped_sample_steps",
            "guided_steps", "seconds", "picked_floor", "picked_free", "cell"]
    pick_f = {(frozen["floor"][a][p]["cell"]) for a in guided for p in props}
    pick_g = {(frozen["free"][a][p]["cell"]) for a in guided for p in props}
    table = []
    for r in sorted(rows, key=lambda r: (r["prop"], r["arm"], float(r["t_start"]),
                                         float(r["w"]))):
        ib = float(r.get(BASECMP_SELECT, float("nan")))
        table.append({
            "backend": r["backend"], "prop": r["prop"], "arm": r["arm"],
            "w": float(r["w"]), "t_start": float(r["t_start"]), "n": int(r["n"]),
            "in_band_dec": ib, "in_band_dec_se": se_prop(ib, r["n"]),
            "in_band_soft": float(r.get("in_band_fraction", float("nan"))),
            "mol_stability": float(r["mol_stability"]),
            "clears_floor": bool(r["mol_stability"] >= floor[r["prop"]] - 1e-12),
            "mae_dec": float(r.get("prop_mae_eval_dec", float("nan"))),
            "mae_soft": float(r.get("prop_mae_eval", float("nan"))),
            "delta": float(r.get("delta", float("nan"))),
            "atom_stability": float(r.get("atom_stability", float("nan"))),
            "validity": float(r.get("validity", float("nan"))),
            "uniqueness_of_valid": float(r.get("uniqueness_of_valid", float("nan"))),
            "unique_valid_per_sample": float(
                r.get("unique_valid_per_sample", float("nan"))),
            "guide_eval_gap_mean": float(r.get("guide_eval_gap_mean", float("nan"))),
            "n_nonfinite": int(r.get("n_nonfinite", 0)),
            "clipped_sample_steps": int(r.get("clipped_sample_steps", 0)),
            "guided_steps": int(r.get("guided_steps", 0)),
            "seconds": float(r.get("seconds", float("nan"))),
            "picked_floor": r["_file"] in pick_f,
            "picked_free": r["_file"] in pick_g, "cell": r["_file"]})

    if args.table_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.table_out)) or ".",
                    exist_ok=True)
        tmp = args.table_out + ".tmp"
        with open(tmp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols)
            wr.writeheader()
            for t in table:
                wr.writerow(t)
        os.replace(tmp, args.table_out)
        print("\nwrote %s (%d rows)" % (args.table_out, len(table)))

    prov = {"generated_by": "proj1/scripts/basecmp_freeze.py",
            "date": datetime.date.today().isoformat(),
            "screen_dir": os.path.abspath(screen), "gen_md5": gen_md5,
            "n": n, "seed": seed, "batch": batch, "steps": steps,
            "solver": solver, "clip": clip, "delta_mode": delta_mode,
            # guarded: a property with no USABLE row is reachable with
            # --allow-partial, or if every one of its cells diverged
            "delta": {p: next((float(r["delta"]) for r in rows
                               if r["prop"] == p and r.get("delta") is not None),
                              None)
                      for p in props},
            "guide": next((r.get("guide") for r in rows if r.get("guide")), None),
            "oracle": next((r.get("oracle") for r in rows if r.get("oracle")), None),
            "feature_scale_check": sanity}

    if args.md_out:
        write_md(args.md_out, args.backend, props, guided, frozen, status,
                 table, floor, unguided, prov, limited, edges, collapsed)
        print("wrote %s" % args.md_out)

    if limited and not args.allow_floor_limited:
        print("\nREFUSING to freeze: %d arm/property pair(s) never clear the "
              "chemistry floor on this grid (%s). Their `floor` pick would be a "
              "fallback, not a selection. Pass --allow-floor-limited to freeze "
              "anyway -- every table then labels them floor_limited."
              % (len(limited), ", ".join(limited)))
        return RC_INCOMPLETE

    if args.json_out:
        out = {"study": "basecmp", "backend": BACKENDS[args.backend],
               "select_metric": BASECMP_SELECT,
               "rule": "per (property, arm), the joint (w, t_start) cell with "
                       "the highest %s; `floor` restricted to mol_stability >= "
                       "%g x unguided, `free` unrestricted; ties to smaller w "
                       "then larger t_start" % (BASECMP_SELECT, BASECMP_FLOOR),
               "sets": list(BASECMP_SETS), "target": BASECMP_TARGET,
               "floor_multiplier": BASECMP_FLOOR,
               "floor": floor, "unguided_mol_stability": unguided,
               "frozen": frozen, "status": status,
               "floor_limited": limited, "grid_edge": edges,
               "collapse_contaminated": collapsed,
               "strength_grid": sorted({t["w"] for t in table}),
               "t_start_grid": sorted({t["t_start"] for t in table}),
               "source_stage": "basecmp", "source_n": n, "source_seed": seed,
               "n_cells": len(table), "prov": prov, "table": table}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".",
                    exist_ok=True)
        tmp = args.json_out + ".tmp"
        json.dump(out, open(tmp, "w"), indent=1)
        os.replace(tmp, args.json_out)
        print("wrote %s" % args.json_out)
    return RC_OK


def write_md(path, backend, props, guided, frozen, status, table, floor,
             unguided, prov, limited, edges, collapsed):
    """The screen as a document, because a csv is not something a reader
    checks. Script-generated, with its provenance in it."""
    L = []
    L.append("# Base-model comparison: the (strength, start-time) screen on %s"
             % BACKENDS[backend])
    L.append("")
    L.append("**Generated by `proj1/scripts/basecmp_freeze.py` on %s. Do not "
             "edit by hand.** Source: `%s`, %d cells, n = %s, seed %s, "
             "generator md5 `%s`, delta mode `%s`."
             % (prov["date"], prov["screen_dir"], len(table), prov["n"],
                prov["seed"], str(prov["gen_md5"])[:12], prov["delta_mode"]))
    L.append("")
    L.append("Guide `%s`, oracle `%s` -- both external to this project and "
             "identical on both base models. delta: %s."
             % (prov.get("guide"), prov.get("oracle"),
                ", ".join("%s %.5f" % (p, prov["delta"][p]) for p in props)))
    L.append("")
    L.append("## The picks")
    L.append("")
    L.append("`floor` is v2's rule V2: the highest decoded in-band among cells "
             "whose molecule stability is at least 0.9x the unguided "
             "generator's. `free` drops the floor and is reported for reach, "
             "never as a headline. Ties go to the smaller strength, then the "
             "later start-time.")
    L.append("")
    L.append("| arm | property | floor: w | floor: t | floor: in-band | floor: stab | free: w | free: t | free: in-band | free: stab |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in guided:
        for p in props:
            f, g = frozen["floor"][a][p], frozen["free"][a][p]
            fl = " ⚠" if status["floor"][a][p] != "ok" else ""
            gl = " ⚠" if status["free"][a][p] != "ok" else ""
            L.append("| %s | %s | %g | %g | %.3f%s | %.3f | %g | %g | %.3f%s | %.3f |"
                     % (a, p, f["w"], f["t_start"], f["in_band_dec"], fl,
                        f["mol_stability"], g["w"], g["t_start"],
                        g["in_band_dec"], gl, g["mol_stability"]))
    L.append("")
    L.append("Unguided molecule stability, and the floor it sets: %s."
             % ", ".join("%s %.4f -> %.4f" % (p, unguided[p], floor[p])
                         for p in props))
    L.append("")
    if prov.get("feature_scale_check"):
        L.append("Feature-scale check — the oracle's mean on unguided samples "
                 "against QM9's own. `build_pair`'s calibration guard is fitted "
                 "on real molecules and cannot see a wrong sampler divisor, which "
                 "only affects scoring; this can. Legitimate gap across 80 earlier "
                 "unguided cells: max 0.62 MAD.")
        L.append("")
        for line in prov["feature_scale_check"]:
            L.append("- `%s`" % line)
        L.append("")
    if limited:
        L.append("**Floor-limited (no cell on the grid clears the floor; the "
                 "`floor` pick is a fallback and is not a win): %s.**"
                 % ", ".join(limited))
        L.append("")
    if edges:
        L.append("**Grid-edge picks — the optimum may lie outside the screened "
                 "grid: %s.**" % ", ".join(edges))
        L.append("")
    if collapsed:
        L.append("**Collapse-contaminated (v2 rule V6, unique valid per sample "
                 "< %.2f): %s.**" % (UNIQ_MIN, ", ".join(collapsed)))
        L.append("")
    L.append("## Every cell")
    L.append("")
    L.append("The single chosen cell is one point on a surface that is itself "
             "the result (v2 rule V2b). `se` is the binomial standard error of "
             "**one cell's own** in-band fraction at this n. It is NOT the error "
             "of a difference between two cells, and must not be read as one: "
             "every cell here shares the same molecule sizes, the same target and "
             "(at fixed batch) the same initial noise, so two arms are PAIRED and "
             "the se of their difference is smaller than these columns suggest. "
             "Comparing two cells by eye through these numbers errs toward calling "
             "a real difference noise. The paired comparison is what the "
             "`.permol.pt` sidecars are for, and v2's own verdict rule (V7) is a "
             "paired z.")
    L.append("")
    L.append("| property | arm | w | t_start | in-band (dec) | se | in-band (soft) | mol stab | clears floor | MAE (dec) | uniq/sample | pick |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in table:
        tag = ",".join(([" floor"] if t["picked_floor"] else [])
                       + ([" free"] if t["picked_free"] else [])).strip()
        L.append("| %s | %s | %g | %g | %.3f | %.3f | %.3f | %.3f | %s | %.4f | %.3f | %s |"
                 % (t["prop"], t["arm"], t["w"], t["t_start"], t["in_band_dec"],
                    t["in_band_dec_se"], t["in_band_soft"], t["mol_stability"],
                    "yes" if t["clears_floor"] else "no", t["mae_dec"],
                    t["unique_valid_per_sample"], tag or "-"))
    L.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
