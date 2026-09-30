"""Freeze the v2 strengths (rule V2 of FULL_RUN_V2_PROTOCOL.md). CPU, seconds.

    python proj1/scripts/freeze_v2.py --json-out results/full/v2/frozen_v2.json

THE RULE. Each arm runs at the strength with the best in_band on the n = 512
q90 screen, among strengths whose mol_stability is at least 0.9 x unguided's.
Ties go to the SMALLER strength. If nothing clears the floor, the arm takes its
most stable strength and is recorded as floor-limited.

WHY THIS IS A SEPARATE SCRIPT and not a flag on check_fullrun_go.py: that
script implements v1's pre-registered FR1/FR3/FR3a and its output is the
record of a finished experiment. v2 selects on a different metric (in_band,
not MAE) over a longer grid, so it gets its own script and its own file rather
than changing the meaning of a committed one.

THE GUARD THAT MATTERS (V2a). An arm that still clears the chemistry floor at
the TOP of its measured grid is grid-limited, not chemistry-limited: the rule
would cap it at wherever we happened to stop looking. `lgd_mc` did exactly
that at w = 4 on all three properties. This script REFUSES to freeze such an
arm (exit 2) and names the strengths to run, unless --allow-grid-limited.

Reads every q90 cell in --sweep-dir whatever stage wrote it, so the
`__cmp` screen and the `__tgt` grid extension are one grid. All cells must
share n and seed, which is checked.

Exit codes: 0 frozen, 2 incomplete / grid-limited / inconsistent, 3 error.
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

PROPS = ("mu", "alpha", "gap")
FLOOR = 0.9                      # the saved rubric: x unguided mol_stability
TARGET = "q90"
RC_OK, RC_INCOMPLETE, RC_ERROR = 0, 2, 3


def load(sweep_dir, target, n, seed, win):
    """{(prop, arm, w): row} over the q90 cells at ONE guidance window, plus
    the (n, seed) seen and the keys that diverged.

    THE WINDOW FILTER IS NOT OPTIONAL. results/sweep holds q90 cells at
    t_min_guide 0.05 (the v1 `main` grid, 171 cells) as well as at 0.5 (the
    compare screen this protocol uses, 204). They share (prop, arm, w), so
    without this filter the dict is last-write-wins over `glob.glob`, whose
    order is the filesystem's -- arbitrary on Linux. Measured: reversing the
    listing moves plug's mu pick from w=0.05 to w=0.01 and lgd_mc's from 4
    to 1. The run would then use strengths chosen under a different window
    from the one it samples at.

    n and seed are NOT filtered here on purpose: they are collected and the
    caller refuses a mixed set, so a cell run at the wrong scale is a loud
    failure rather than a silent omission.
    """
    rows, seen_n, seen_seed, bad, clash = {}, set(), set(), set(), []
    for fn in sorted(glob.glob(os.path.join(sweep_dir, "*.json"))):
        try:
            with open(fn, encoding="utf-8") as fh:
                r = json.load(fh)
        except Exception:
            continue
        if r.get("target_name") != target:
            continue
        if abs(float(r.get("t_min_guide", -1)) - win) > 1e-12:
            continue
        seen_n.add(r.get("n"))
        seen_seed.add(r.get("seed"))
        key = (r["prop"], r["arm"], float(r["w"]))
        if r.get("n_nonfinite", 0) > 0 or not math.isfinite(
                float(r.get("prop_mae_eval", float("nan")))):
            bad.add(key)
        prev = rows.get(key)
        if prev is not None and (prev["in_band_fraction"] != r["in_band_fraction"]
                                 or prev["mol_stability"] != r["mol_stability"]):
            clash.append("%s/%s@w%g" % key)
        rows[key] = r
    return rows, seen_n, seen_seed, bad, clash


NO_STRENGTH = ("unguided",)     # run once per property; w is not a knob


def pick(cands, floor, bad):
    """(row, status). status: 'ok', 'grid_limited', 'floor_limited'.

    `unguided` has no strength axis -- the sweep runs it once per property and
    ignores w -- so it can never be grid-limited, however stable it is.

    GRID-LIMITED means the strength grid ran out before chemistry did, so the
    rule would cap the arm at wherever we stopped looking rather than at its
    real limit. Two things end the grid legitimately: the top strength drops
    below the floor, or it DIVERGES (non-finite samples). Judging `top` over
    clean cells only would make the guard unsatisfiable for an arm that blows
    up before it destabilises -- it would keep demanding a stronger cell that
    can only diverge harder.
    """
    all_w = [float(r["w"]) for r in cands]
    usable = [r for r in cands if (r["prop"], r["arm"], float(r["w"])) not in bad]
    if not usable:
        return None, "no_clean_cell"
    ok = [r for r in usable if r["mol_stability"] >= floor - 1e-12]
    if not ok:
        return max(usable, key=lambda r: (r["mol_stability"], -float(r["w"]))), "floor_limited"
    best = max(ok, key=lambda r: (r["in_band_fraction"], -float(r["w"])))
    if best["arm"] in NO_STRENGTH:
        return best, "ok"
    top_w = max(all_w)
    top = next(r for r in cands if float(r["w"]) == top_w)
    top_diverged = (top["prop"], top["arm"], top_w) in bad
    if not top_diverged and top["mol_stability"] >= floor - 1e-12:
        return best, "grid_limited"
    return best, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=os.path.join(ROOT, "results", "sweep"))
    ap.add_argument("--arms", default="unguided,plug,tmpd,lgd_mc,tfg,btvg,btvg_var")
    ap.add_argument("--props", default=",".join(PROPS))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--win", type=float, default=0.5,
                    help="t_min_guide of the screen to read. NOT cosmetic: "
                         "results/sweep holds q90 cells at 0.05 and at 0.5 "
                         "with the same (prop, arm, w) keys.")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--allow-grid-limited", action="store_true",
                    help="freeze anyway, recording which arms were capped by "
                         "the grid rather than by chemistry")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    props = [p for p in args.props.split(",") if p]
    rows, seen_n, seen_seed, bad, clash = load(args.sweep_dir, TARGET, args.n,
                                               args.seed, args.win)
    if not rows:
        print("no %s cells at t_min_guide=%g in %s"
              % (TARGET, args.win, args.sweep_dir))
        return RC_INCOMPLETE
    if len(seen_n) > 1 or len(seen_seed) > 1:
        print("INCONSISTENT: the %s cells at window %g mix n=%s and seed=%s. "
              "A strength cannot be chosen across sample sizes or seeds; "
              "remove or re-run the odd cells." 
              % (TARGET, args.win, sorted(seen_n), sorted(seen_seed)))
        return RC_INCOMPLETE
    if sorted(seen_n)[0] != args.n or sorted(seen_seed)[0] != args.seed:
        print("INCONSISTENT: cells are n=%s seed=%s but --n %d --seed %d were asked for"
              % (sorted(seen_n), sorted(seen_seed), args.n, args.seed))
        return RC_INCOMPLETE
    if clash:
        print("DUPLICATE cells disagree for: %s" % ", ".join(sorted(set(clash))))
        return RC_INCOMPLETE

    missing = [(p, a) for p in props for a in arms
               if not [r for (pp, aa, _w), r in rows.items() if pp == p and aa == a]]
    if missing:
        print("INCOMPLETE: no cell at all for %s"
              % ", ".join("%s/%s" % m for m in missing))
        return RC_INCOMPLETE

    frozen, status, stab, grid = {}, {}, {}, {}
    floors = {}
    for p in props:
        u = [r for (pp, aa, _w), r in rows.items() if pp == p and aa == "unguided"]
        if not u:
            print("INCOMPLETE: no unguided cell for %s" % p)
            return RC_INCOMPLETE
        # the floor is defined by unguided, so duplicates must agree or the
        # floor would depend on which one was read first
        stabs = {round(x["mol_stability"], 12) for x in u}
        if len(stabs) > 1:
            print("INCONSISTENT: %d unguided cells for %s disagree on "
                  "mol_stability (%s); the chemistry floor is ambiguous"
                  % (len(u), p, sorted(stabs)))
            return RC_INCOMPLETE
        floors[p] = FLOOR * u[0]["mol_stability"]

    for a in arms:
        for p in props:
            cands = [r for (pp, aa, _w), r in rows.items() if pp == p and aa == a]
            r, st = pick(cands, floors[p], bad)
            if r is None:
                print("NO CLEAN CELL for %s/%s at any strength" % (p, a))
                return RC_INCOMPLETE
            frozen.setdefault(a, {})[p] = float(r["w"])
            status.setdefault(a, {})[p] = st
            stab.setdefault(a, {})[p] = r["mol_stability"]
            grid.setdefault(a, {})[p] = sorted(float(w) for (pp, aa, w) in rows
                                               if pp == p and aa == a)

    print("v2 freeze -- best in_band with mol_stability >= %.1fx unguided "
          "(n=%d, seed=%d, target=%s)" % (FLOOR, args.n, args.seed, TARGET))
    print("floors: %s\n" % ", ".join("%s %.3f" % (p, floors[p]) for p in props))
    print("%-10s %s" % ("arm", "  ".join("%18s" % p for p in props)))
    for a in arms:
        cells = []
        for p in props:
            flag = {"ok": "", "grid_limited": " GRID", "floor_limited": " FLOOR"}[status[a][p]]
            cells.append("%18s" % ("w=%g (stab %.3f)%s" % (frozen[a][p], stab[a][p], flag)))
        print("%-10s %s" % (a, "  ".join(cells)))

    capped = [(a, p, max(w for w in grid[a][p]
                         if (p, a, w) not in bad))
              for a in arms for p in props if status[a][p] == "grid_limited"]
    if capped:
        print("\nGRID-LIMITED -- these still clear the floor at the top of their "
              "grid, so the rule is capping them, not chemistry:")
        for a, p, top in capped:
            print("   %s/%s  grid stops at w=%g -- run w=%g and up until the floor bites"
                  % (a, p, top, top * 2))
        if not args.allow_grid_limited:
            print("\nREFUSING to freeze (rule V2a). Extend the grid, or pass "
                  "--allow-grid-limited to record it as measured.")
            return RC_INCOMPLETE

    floor_limited = ["%s/%s" % (a, p) for a in arms for p in props
                     if status[a][p] == "floor_limited"]
    if floor_limited:
        print("\nfloor-limited (nothing clears; most stable strength used): %s"
              % ", ".join(floor_limited))

    if args.json_out:
        out = {"rule": "v2 (FULL_RUN_V2_PROTOCOL.md V2): best in_band among "
                       "strengths with mol_stability >= %.1f x unguided; ties to "
                       "the smaller w" % FLOOR,
               "frozen_w": frozen, "status": status, "mol_stability": stab,
               "floor": floors, "grid": grid,
               "source_stage": "compare", "target": TARGET,
               "t_min_guide": args.win,
               "source_seed": sorted(seen_seed)[0], "source_n": sorted(seen_n)[0],
               "grid_limited": ["%s/%s" % (a, p) for a, p, _ in capped],
               "floor_limited": floor_limited,
               "note": "cells read from every stage that wrote a q90 cell at "
                       "this n and seed, so the __cmp screen and the __tgt grid "
                       "extension form one grid"}
        tmp = args.json_out + ".tmp"
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, args.json_out)
        print("\nwrote %s" % args.json_out)
    return RC_OK


def _entry():
    try:
        return main()
    except SystemExit as ex:
        return 0 if ex.code in (0, None) else RC_ERROR
    except Exception:
        import traceback
        traceback.print_exc()
        print("ERROR -- freeze_v2 crashed; nothing was written")
        return RC_ERROR


if __name__ == "__main__":
    sys.exit(_entry())
