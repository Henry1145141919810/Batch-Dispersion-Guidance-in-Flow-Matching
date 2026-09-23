"""Apply pre-registered rule FR1 to the compare stage, and freeze strengths for FR3.

NAMED FR1-FR5 ("full-run rules"), NOT G1-G5: those names already belong to
the experiment plan's gates (G1 guide resolution ... G5 soft-vs-decoded) in
docs/protocol/GUIDANCE_EXPERIMENT_PLAN.md, and g1_guide_resolution.py
implements that G1.

    python proj1/scripts/check_fullrun_go.py

Written 22 Sep, BEFORE any compare cell existed, so the decision is mechanical
and cannot be tuned after seeing the numbers. The rules it implements are in
docs/status/SCOPE_FM_GUIDANCE_STATUS.md, "PRE-REGISTRATION".

FR1 BTVG proceeds to the full-scale run UNLESS, on 2 or more of the 3
    properties, it is beaten by the best competitor by more than 3 combined
    standard errors on MAE, each arm at its best strength.

FR3 Each arm's best-MAE strength here is FROZEN for the full run, which then
    uses new seeds so the choice is not also the evaluation.

FR3a (amended 23 Sep, after the screen, before any full-run cell). At q90 the
    FR3 strengths (w=4 for plug/tmpd/btvg) fail the chemistry floor FR5 applies
    (mol_stability 0.21-0.35 vs 0.362), so FR3 alone would make the headline
    table unjudgeable. FR3a = best MAE among strengths with
    mol_stability >= 0.9 x unguided at the same (prop, target); if none clears
    it, the most stable strength. Same rule for every arm.
    JSON: "frozen_w" = FR3a (the HEADLINE set the full run uses, and what
    every downstream reader pairs at); "frozen_w_mae" = FR3 as registered
    (run as a secondary, one seed).

Costs seconds of CPU and no GPU: it only reads the cell JSONs. It does not
import torch: the grid it checks against is read out of guidance_sweep.py's
source, so the two cannot drift apart and this stays safe on a login node.

COMPLETENESS IS THE WHOLE STRENGTH GRID, not "one cell per arm". An earlier
version accepted any grid with a single clean cell per (prop, arm), so run on a
half-finished compare stage it would freeze a best-of-partial strength and
report it as FR3's. A cell that ran but diverged counts as PRESENT (it is a
real outcome at that strength) and is excluded only from being anyone's best.

Exit codes -- the full-run job branches on these:
   0  FR1 pass
  10  FR1 FAIL. Deliberately NOT 1: an uncaught Python exception also exits
      1, and a crash must never read as a pre-registered negative result.
   2  incomplete or inconsistent compare grid
   3  error (any exception, unreadable source)
Ties on MAE (bit-identical clip-saturated cells) break toward the SMALLER w,
so the frozen strength never depends on directory listing order.
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

PROPS = ("mu", "alpha", "gap")
# FR1 as REGISTERED (22 Sep) judged btvg against plug, tmpd, lgd_mc and
# dflow; that verdict (PASS on q50, beaten 0/3) is recorded in
# results/full/n5000/fr1_q50.json and is not re-litigated. On 23 Sep, after
# the full run was read, tfg replaced dflow in the compare set, so any FR1
# printed with tfg in it is POST HOC and report-only -- tfg_run.slurm never
# stops on it.
REGISTERED_COMPETITORS = ("plug", "tmpd", "lgd_mc", "dflow")
COMPETITORS = ("plug", "tmpd", "lgd_mc", "tfg")
OURS = "btvg"
ALL_ARMS = ("unguided",) + COMPETITORS + (OURS, "btvg_var")
SIGMA = 3.0          # FR1 threshold, fixed in the pre-registration
BEATEN_ON = 2        # "on 2 or more of the 3 properties"
RC_PASS, RC_FAIL, RC_INCOMPLETE, RC_ERROR = 0, 10, 2, 3


def sweep_constants(path=os.path.join(HERE, "guidance_sweep.py")):
    """STRENGTHS, DEFAULT_W and COMPARE_SET, read from the sweep's source."""
    want = {"STRENGTHS", "COMPARE_SET"}
    got = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    got[t.id] = ast.literal_eval(node.value)
                if (isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple)
                        and [getattr(e, "id", None) for e in t.elts]
                        == ["DEFAULT_W", "DEFAULT_WIN"]):
                    got["DEFAULT_W"] = ast.literal_eval(node.value.elts[0])
    missing = (want | {"DEFAULT_W"}) - set(got)
    if missing:
        raise RuntimeError("could not read %s from %s" % (sorted(missing), path))
    return got


def expected_grid(consts):
    """Every (arm, w) the compare stage runs at one target. Mirrors
    plan_compare_cells: unguided once at DEFAULT_W, the rest over STRENGTHS."""
    g = {("unguided", float(consts["DEFAULT_W"]))}
    for a in consts["COMPARE_SET"]:
        if a != "unguided":
            g |= {(a, float(w)) for w in consts["STRENGTHS"]}
    return g


def se_mae(r):
    """se of MAE from its first two moments: var(|e|) = RMSE^2 - MAE^2."""
    v = max(r["prop_rmse_eval"] ** 2 - r["prop_mae_eval"] ** 2, 0.0)
    return math.sqrt(v / r["n"])


def load(sweep, stage, target):
    """(clean rows, every (prop, arm, w) present). Diverged cells are in the
    second but not the first: present for completeness, never anyone's best."""
    rows, present = [], set()
    for fn in glob.glob(os.path.join(sweep, "*.json")):
        try:
            r = json.load(open(fn))
        except Exception:
            continue
        if r.get("stage") != stage or r.get("target_name") != target:
            continue
        present.add((r["prop"], r["arm"], float(r["w"])))
        if r.get("n_nonfinite", 0) > 0:
            continue        # a divergent cell may not be anyone's best
        if not math.isfinite(float(r.get("prop_mae_eval", float("nan")))):
            continue        # coordinates can blow up while staying "finite"
        rows.append(r)
    return rows, present


def best(rows, prop, arm):
    c = [r for r in rows if r["prop"] == prop and r["arm"] == arm]
    # (MAE, w): a tie goes to the smaller strength, never to listing order
    return min(c, key=lambda r: (r["prop_mae_eval"], float(r["w"]))) if c else None


FLOOR = 0.9           # the saved rubric's chemistry floor: x unguided


def best_floor(rows, prop, arm, floor):
    """FR3a: best MAE among strengths that clear the chemistry floor; if none
    does, the most stable strength (tie -> smaller w). Returns (row, fell_back)."""
    c = [r for r in rows if r["prop"] == prop and r["arm"] == arm]
    ok = [r for r in c if r["mol_stability"] >= floor - 1e-12]  # float-safe
    if ok:
        return min(ok, key=lambda r: (r["prop_mae_eval"], float(r["w"]))), False
    return max(c, key=lambda r: (r["mol_stability"], -float(r["w"]))), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default=os.path.join(ROOT, "results", "sweep"))
    ap.add_argument("--stage", default="compare",
                    help="'compare' is the real test; any other value is a dry run")
    ap.add_argument("--target", default="q50")
    ap.add_argument("--json-out", default="",
                    help="write the frozen FR3 strengths here for the full run")
    ap.add_argument("--must-match", default="",
                    help="an earlier frozen file (e.g. the full run's "
                         "frozen_q90.json): every arm it shares with this "
                         "freeze must get the SAME strengths and the same "
                         "floor, or nothing is written (exit 2). Used when an "
                         "arm is added after a run, so the existing arms are "
                         "provably untouched.")
    args = ap.parse_args()

    consts = sweep_constants()
    if tuple(consts["COMPARE_SET"]) != ALL_ARMS:
        print("COMPARE_SET in guidance_sweep.py is %s but this script judges %s "
              "-- update one of them before trusting FR1."
              % (consts["COMPARE_SET"], list(ALL_ARMS)))
        return RC_INCOMPLETE
    rows, present = load(args.sweep_dir, args.stage, args.target)
    if not rows:
        print("no '%s'-stage cells at target %s in %s -- has the compare run "
              "finished?" % (args.stage, args.target, args.sweep_dir))
        return RC_INCOMPLETE
    # one screen, not a blend: a best-of taken across sample sizes or seeds
    # is not the pre-registered choice
    ns = sorted({r["n"] for r in rows})
    seeds = sorted({r.get("seed") for r in rows}, key=str)
    if len(ns) > 1 or len(seeds) > 1:
        print("INCONSISTENT -- cells at target %s mix n=%s and seed=%s"
              % (args.target, ns, seeds))
        return RC_INCOMPLETE

    # completeness: FR1 and FR3 are only valid on the FULL strength grid
    grid = expected_grid(consts)
    holes = sorted((p, a, w) for p in PROPS for (a, w) in grid
                   if (p, a, w) not in present)
    if holes:
        print("INCOMPLETE -- %d of %d cells missing at target %s, e.g. %s"
              % (len(holes), len(grid) * len(PROPS), args.target,
                 ", ".join("%s/%s/w%g" % h for h in holes[:6])))
        print("FR1/FR3 are not evaluated on a partial grid. Wait for the run.")
        return RC_INCOMPLETE
    missing = [(p, a) for p in PROPS for a in ALL_ARMS if not best(rows, p, a)]
    if missing:
        print("NO CLEAN CELL at any strength for: %s (every one diverged)"
              % ", ".join("%s/%s" % m for m in missing))
        return RC_INCOMPLETE

    print("FR1: is %s beaten by the best competitor by > %g sigma on MAE?\n"
          % (OURS, SIGMA))
    print("%-6s %-10s %10s %10s %10s %8s  %s"
          % ("prop", "best comp", "comp MAE", "btvg MAE", "diff", "sigma", "beaten?"))
    print("-" * 72)
    beaten = 0
    for p in PROPS:
        b = best(rows, p, OURS)
        comp = min((best(rows, p, a) for a in COMPETITORS),
                   key=lambda r: r["prop_mae_eval"])
        d = b["prop_mae_eval"] - comp["prop_mae_eval"]         # >0: btvg worse
        z = d / math.sqrt(se_mae(b) ** 2 + se_mae(comp) ** 2)
        hit = z > SIGMA
        beaten += hit
        print("%-6s %-10s %10.4f %10.4f %+10.4f %+8.2f  %s"
              % (p, comp["arm"], comp["prop_mae_eval"], b["prop_mae_eval"],
                 d, z, "YES" if hit else "no"))
    print("-" * 72)
    ok = beaten < BEATEN_ON
    print("beaten on %d of 3 properties (fails at %d)" % (beaten, BEATEN_ON))
    print("FR1: %s" % ("PASS -- btvg proceeds to the full run" if ok else
                      "FAIL -- btvg is reported as a negative result, NOT re-tuned"))

    # FR3 (as registered): each arm's best-MAE strength, unconstrained
    frozen_mae = {a: {p: best(rows, p, a)["w"] for p in PROPS} for a in ALL_ARMS}
    # FR3a (headline): best MAE among strengths clearing the chemistry floor
    floor = {}
    for p in PROPS:
        u = [r for r in rows if r["prop"] == p and r["arm"] == "unguided"]
        floor[p] = FLOOR * u[0]["mol_stability"]
    frozen, fell_back, stab = {}, [], {}
    for a in ALL_ARMS:
        for p in PROPS:
            r, fb = best_floor(rows, p, a, floor[p])
            frozen.setdefault(a, {})[p] = r["w"]
            stab.setdefault(a, {})[p] = r["mol_stability"]
            if fb:
                fell_back.append("%s/%s" % (a, p))
    print("\nFR3a HEADLINE strengths (best MAE with mol_stability >= %.1f x "
          "unguided; floor %s), seed %s" % (FLOOR, ", ".join(
              "%s %.3f" % (p, floor[p]) for p in PROPS), ",".join(map(str, seeds))))
    print("%-10s %16s %16s %16s   | FR3 as registered (best MAE)"
          % ("arm", *("%s w (stab)" % p for p in PROPS)))
    for a in ALL_ARMS:
        print("%-10s %16s %16s %16s   | %6g %6g %6g" % (
            a, *("%g (%.3f)" % (frozen[a][p], stab[a][p]) for p in PROPS),
            *(frozen_mae[a][p] for p in PROPS)))
    if fell_back:
        print("no strength clears the floor for: %s -> most stable strength used"
              % ", ".join(fell_back))
    if args.must_match:
        # An arm added after a run (tfg) must not move anyone else: same
        # cells in, same strengths out. A difference means the compare cells
        # on disk are not the ones the earlier freeze read.
        base = json.load(open(args.must_match))
        diffs = []
        for key, new in (("frozen_w", frozen), ("frozen_w_mae", frozen_mae)):
            for a, per in (base.get(key) or {}).items():
                if a not in new:
                    continue
                for p, w in per.items():
                    if p in new[a] and float(new[a][p]) != float(w):
                        diffs.append("%s %s/%s: was %g, now %g"
                                     % (key, a, p, w, new[a][p]))
        for p, f0 in (base.get("floor") or {}).items():
            if p in floor and abs(float(floor[p]) - float(f0)) > 1e-12:
                diffs.append("floor %s: was %.6f, now %.6f" % (p, f0, floor[p]))
        if diffs:
            print("MISMATCH with %s -- nothing written:\n  %s"
                  % (args.must_match, "\n  ".join(diffs)))
            return RC_INCOMPLETE
        print("consistent with %s: every shared arm's strengths and the floor "
              "are unchanged" % args.must_match)
    if args.json_out:
        # atomic: the full-run job treats this file's EXISTENCE as "frozen"
        tmp = args.json_out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"fr1_pass": ok, "beaten_on": beaten,
                       "frozen_w": frozen, "frozen_w_mae": frozen_mae,
                       "rule": "frozen_w = FR3a: best MAE among strengths with "
                               "mol_stability >= %.1f x unguided; frozen_w_mae = "
                               "FR3 as registered" % FLOOR,
                       "floor": floor, "fell_back": fell_back,
                       "source_stage": args.stage, "target": args.target,
                       "source_seed": seeds[0] if len(seeds) == 1 else seeds},
                      fh, indent=1)
        os.replace(tmp, args.json_out)
        print("\nwrote %s" % args.json_out)
    return RC_PASS if ok else RC_FAIL


def _entry():
    """main(), with every failure mapped to RC_ERROR -- never to 1 or 10."""
    try:
        return main()
    except SystemExit as ex:              # argparse (--help is 0), or a raise
        return 0 if ex.code in (0, None) else RC_ERROR
    except Exception:
        import traceback
        traceback.print_exc()
        print("ERROR -- check_fullrun_go crashed; this is NOT an FR1 verdict")
        return RC_ERROR


if __name__ == "__main__":
    sys.exit(_entry())
