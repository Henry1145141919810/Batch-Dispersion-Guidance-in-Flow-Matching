"""Measure what a cell costs on each base model, then choose the run's n.

    python proj1/scripts/basecmp_size.py --budget-gpuh 100 \
        --out results/basecmp/plan.json

WHY THIS EXISTS. EquiFM has never had a single cell run on it -- the harness was
built and gated on 23 Sep and nothing was ever sampled (status doc: "EquiFM:
harness ready, 0 cells"). So the cost of this experiment was, until the probe
ran, a guess extrapolated from EDMsecond, a different network. Sizing a
100-GPU-hour queue on a guess is how a run gets killed by the 4-hour QOS limit
half way through and has to be resubmitted, which is the one thing Henry asked
not to happen.

So: the probe runs a handful of REAL cells at small n on both bases, this script
reads their recorded `seconds`, and the screen and full-run jobs read their n
from the plan file it writes. Everything is submitted in one go with SLURM
dependencies; the sizing happens inside the chain, not before it.

THE COST MODEL, and its one assumption. Cell time is taken as proportional to n
at fixed batch, per (backend, arm). That is right for the sampling loop, which
is the overwhelming majority of the work: the model is evaluated once per step
per batch, and the number of batches is n / batch. It is NOT exactly right for
`evaluate_samples`, which builds an n x n Gram matrix for `diversity_logdet`;
that term is quadratic, but it is seconds against minutes of sampling even at
n = 5000. A margin (--margin, default 1.25) covers it along with the usual
cluster variance. If the measured cost still overruns, the per-cell time guard
(`--max-minutes`) stops each job cleanly and the array's insurance sweeper picks
up the remaining cells; no cell is ever half-written, because the json is the
completion marker.

THE LADDERS. n for the screen is preferred at 1000 -- Henry's instruction -- and
only steps down if even the smallest useful full run will not fit beside it. n
for the full run steps down from 5000. Both ladders are fixed here, before any
timing was read, so the choice is a rule and not a judgement made to fit a
number that had already been seen.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from transfer_sweep import (BACKENDS, BASECMP_FULL_SEEDS, BASECMP_ROOT,  # noqa: E402
                            BASECMP_SETS, BASECMP_STRENGTHS, BASECMP_TARGET,
                            BASECMP_T_STARTS, backend_arms,
                            plan_basecmp_cells)

BACKS = ("fm", "equifm")
SCREEN_LADDER = (1000, 750, 500, 250)
FULL_LADDER = (5000, 2000, 1000, 500)
# Below this the full run cannot resolve what v2 asks it to resolve: v2 measured
# the se of an in-band difference at n = 5000 x 3 seeds as ~0.0025, against
# best-vs-worst spreads of 5-13 sigma. At n = 500 x 3 that se is ~0.008 and the
# same spreads fall to 1.6-4 sigma, i.e. mostly below v2's own z >= 3 verdict
# bar. A run smaller than this would produce numbers that cannot answer the
# question, which is worse than a run that is honestly smaller in scope.
FULL_MIN_USEFUL = 1000


def probe_costs(root, backend):
    """{arm: (a, b)} -- seconds per molecule modelled as `a + b * (1 - t_start)`.

    WHY NOT ONE NUMBER PER ARM. Guidance cost scales with the GUIDED FRACTION of
    the trajectory: at t_start = 0.05 about 95 of the 100 Euler steps are guided,
    against 50 at 0.5 and 25 at 0.75. A single measurement cannot separate the
    fixed sampling cost `a` from the per-guided-step cost `b`, so a flat model
    fitted at the reference window under-prices every earlier window -- which is
    how an array task ends up past the 4-hour wall having finished nothing. The
    probe therefore measures three start-times and this fits the line through
    them by least squares. `1 - t_start` is the guided fraction by construction
    (guidance runs on [t_start, 1)), so the fit extrapolates in exactly the
    variable the screen varies.

    An arm measured at fewer than two distinct start-times gets a flat model
    (b = 0) at its median cost, which is the old behaviour and is flagged."""
    d = os.path.join(root, backend, "probe")
    obs = {}
    for fn in sorted(glob.glob(os.path.join(d, "tr__*.json"))):
        r = json.load(open(fn))
        if r.get("backend") != BACKENDS[backend]:
            raise SystemExit("%s is a %r cell, not %s"
                             % (fn, r.get("backend"), BACKENDS[backend]))
        if not r.get("seconds") or not r.get("n"):
            continue
        t = float(r.get("t_start", 0.5))
        obs.setdefault(r["arm"], []).append((1.0 - t,
                                             float(r["seconds"]) / float(r["n"])))
    fit = {}
    for arm, pts in obs.items():
        xs = sorted({p[0] for p in pts})
        if len(xs) < 2:
            fit[arm] = (statistics.median([p[1] for p in pts]), 0.0)
            continue
        # least squares on (guided fraction -> seconds per molecule)
        n = len(pts)
        sx = sum(p[0] for p in pts)
        sy = sum(p[1] for p in pts)
        sxx = sum(p[0] * p[0] for p in pts)
        sxy = sum(p[0] * p[1] for p in pts)
        den = n * sxx - sx * sx
        if abs(den) < 1e-18:
            fit[arm] = (sy / n, 0.0)
            continue
        b = (n * sxy - sx * sy) / den
        a = (sy - b * sx) / n
        # a negative slope is noise, not a model: guiding more steps cannot be
        # cheaper. Fall back to the most expensive observation, so the estimate
        # never comes in under what was actually measured.
        if b < 0.0:
            fit[arm] = (max(p[1] for p in pts), 0.0)
        else:
            fit[arm] = (max(a, 0.0), b)
    return fit


def sec_per_mol(fit, arm, t_start, fallback):
    a, b = fit.get(arm, fallback)
    return max(a + b * (1.0 - float(t_start)), 0.0)


def cost_of(cells, fit, n, fallback):
    """GPU-seconds for a cell list at sample size n, window by window."""
    tot = 0.0
    for c in cells:
        arm = c[1]
        t = float(c[4]) if len(c) > 4 else 0.5
        tot += sec_per_mol(fit, arm, t, fallback) * n
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=BASECMP_ROOT)
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--budget-gpuh", type=float, default=100.0,
                    help="total GPU-hours the whole chain may cost (screen + "
                         "refine + full run)")
    ap.add_argument("--margin", type=float, default=1.25,
                    help="safety factor on every estimate")
    ap.add_argument("--refine-cells-per-arm", type=float, default=2.0,
                    help="how many refine cells each (arm, property, backend) "
                         "is assumed to add; the real number comes from the "
                         "pick positions and is not known yet")
    ap.add_argument("--full-backend", default="equifm", choices=list(BACKS),
                    help="which base model gets the v2 full run")
    ap.add_argument("--out", default="", help="where to write the plan json")
    args = ap.parse_args()

    props = [p for p in args.props.split(",") if p]
    per_mol, missing = {}, []
    for b in BACKS:
        per_mol[b] = probe_costs(args.root, b)
        if not per_mol[b]:
            missing.append(b)
    if missing:
        raise SystemExit("no probe cells for %s under %s -- the probe stage has "
                         "not run (or wrote elsewhere). Nothing can be sized "
                         "from a guess." % (", ".join(missing), args.root))

    # the fallback for an arm the probe never priced: the most expensive arm
    # measured on that base, at full guidance. Never cheaper than what was seen.
    fb_of = {b: max(per_mol[b].values(), key=lambda ab: ab[0] + ab[1])
             for b in BACKS}
    print("measured cost per molecule (probe): a + b x guided_fraction")
    for b in BACKS:
        print("  %-8s %s" % (b, "  ".join(
            "%s %.4f+%.4fg" % (a, ab[0], ab[1])
            for a, ab in sorted(per_mol[b].items()))))
        flat = [a for a, ab in per_mol[b].items() if ab[1] == 0.0]
        if flat:
            print("           FLAT MODEL (one window only, or a negative slope) "
                  "for: %s -- those windows are priced at the most expensive "
                  "observation" % ", ".join(sorted(flat)))
        print("           t=0.05 costs %.2fx what t=0.75 does, on the most "
              "expensive arm"
              % (sec_per_mol(per_mol[b], max(per_mol[b], key=lambda a:
                             per_mol[b][a][0] + per_mol[b][a][1]), 0.05, fb_of[b])
                 / max(sec_per_mol(per_mol[b], max(per_mol[b], key=lambda a:
                       per_mol[b][a][0] + per_mol[b][a][1]), 0.75, fb_of[b]), 1e-9)))

    # ---- the work, as cell lists (prop, arm, tgt, w, t_start) ---------------
    screen = {b: plan_basecmp_cells(props, backend_arms(b), BASECMP_T_STARTS)
              for b in BACKS}
    guided = [a for a in backend_arms(args.full_backend) if a != "unguided"]
    # The refine stage: assumed cells per (arm, property), both backends. Their
    # window is unknown -- it is whichever window the pick landed at -- so they
    # are charged at the most expensive one, 0.05.
    refine = {b: [(p, a, BASECMP_TARGET, 1.0, 0.05) for p in props for a in guided
                  for _ in range(int(round(args.refine_cells_per_arm)))]
              for b in BACKS}
    # The v2 full run: two strength sets, one shared unguided, three seeds. Each
    # cell sits at its arm's frozen window, which is not known yet, so they are
    # charged at 0.05 too. De-duplication between the two sets is NOT assumed:
    # assuming it would under-budget exactly when the picks differ, which is the
    # interesting case.
    full = [(p, a, BASECMP_TARGET, 1.0, 0.05)
            for _s in BASECMP_SETS for p in props for a in guided]
    full += [(p, "unguided", BASECMP_TARGET, 1.0, 0.5) for p in props]
    full = full * len(BASECMP_FULL_SEEDS)

    def hours(n_screen, n_full):
        tot = 0.0
        for b in BACKS:
            tot += cost_of(screen[b], per_mol[b], n_screen, fb_of[b])
            tot += cost_of(refine[b], per_mol[b], n_screen, fb_of[b])
        tot += cost_of(full, per_mol[args.full_backend], n_full,
                       fb_of[args.full_backend])
        return tot * args.margin / 3600.0

    def worst_task_minutes(n_screen, n_full):
        """The most expensive single array task, against the 225-min guard.

        The screen is split by (backend, arm group, property, start-time) and the
        full run by (arm group, property, seed); the groups are the ones
        basecmp_run.slurm uses. If this comes out near 225 the array will leave
        work for the sweepers, and if it comes out over it the sweepers may not
        catch up -- so it is printed rather than assumed."""
        groups = [["plug", "tmpd"], ["lgd_mc", "tfg"], ["btvg"], ["btvg_var"]]
        worst = ("", 0.0)
        for b in BACKS:
            for grp in groups:
                for t in BASECMP_T_STARTS:
                    cells = [(p, a, BASECMP_TARGET, w, t) for a in grp
                             for w in BASECMP_STRENGTHS for p in [props[0]]]
                    m = cost_of(cells, per_mol[b], n_screen,
                                fb_of[b]) * args.margin / 60.0
                    if m > worst[1]:
                        worst = ("screen %s %s t=%g" % (b, "+".join(grp), t), m)
        fbk = args.full_backend
        for grp in groups:
            cells = [(props[0], a, BASECMP_TARGET, 1.0, 0.05)
                     for _s in BASECMP_SETS for a in grp]
            m = cost_of(cells, per_mol[fbk], n_full,
                        fb_of[fbk]) * args.margin / 60.0
            if m > worst[1]:
                worst = ("full %s %s" % (fbk, "+".join(grp)), m)
        # THE REFINE STAGE was the largest task in the chain until it was split.
        # It is charged at t = 0.05 throughout, because the cells it adds by
        # extending an edge are disproportionately at the expensive early windows.
        # Split by (backend, property, arm group), matching basecmp_run.slurm.
        for b in BACKS:
            for grp in groups:
                cells = [(props[0], a, BASECMP_TARGET, 1.0, 0.05)
                         for _s in BASECMP_SETS for a in grp
                         for _ in range(int(round(args.refine_cells_per_arm)))]
                m = cost_of(cells, per_mol[b], n_screen,
                            fb_of[b]) * args.margin / 60.0
                if m > worst[1]:
                    worst = ("refine %s %s" % (b, "+".join(grp)), m)
        return worst

    # ---- choose, by the fixed ladder rule ----------------------------------
    choice = None
    for ns in SCREEN_LADDER:
        for nf in FULL_LADDER:
            if nf < FULL_MIN_USEFUL:
                continue
            if hours(ns, nf) <= args.budget_gpuh:
                choice = (ns, nf)
                break
        if choice:
            break
    print("\ncost surface (GPU-hours, margin %.2f):" % args.margin)
    print("      n_full ->  " + "  ".join("%7d" % nf for nf in FULL_LADDER))
    for ns in SCREEN_LADDER:
        print("  n_screen %5d  " % ns
              + "  ".join("%7.1f" % hours(ns, nf) for nf in FULL_LADDER))

    if choice is None:
        ns, nf = SCREEN_LADDER[-1], FULL_MIN_USEFUL
        print("\nNOTHING ON THE LADDER FITS %.0f GPU-hours." % args.budget_gpuh)
        print("The smallest screen (n=%d) beside the smallest full run that can "
              "still resolve v2's verdict (n=%d) costs %.1f GPU-hours."
              % (ns, nf, hours(ns, nf)))
        print("Shrinking n further is not an option worth taking: at n < %d x 3 "
              "seeds the in-band differences v2 calls verdicts sit inside their "
              "own standard error, so the run would cost real GPU time and "
              "answer nothing. Raise --budget-gpuh, or cut the grid (fewer "
              "start-times, or the screen on one base model), and re-run this."
              % FULL_MIN_USEFUL)
        return 2

    n_screen, n_full = choice
    h_screen = sum(cost_of(screen[b], per_mol[b], n_screen, fb_of[b])
                   for b in BACKS) * args.margin / 3600.0
    h_refine = sum(cost_of(refine[b], per_mol[b], n_screen, fb_of[b])
                   for b in BACKS) * args.margin / 3600.0
    h_full = cost_of(full, per_mol[args.full_backend], n_full,
                     fb_of[args.full_backend]) * args.margin / 3600.0
    worst_name, worst_min = worst_task_minutes(n_screen, n_full)

    plan = {"study": "basecmp",
            "date": datetime.date.today().isoformat(),
            "n_screen": n_screen, "n_full": n_full,
            "full_backend": args.full_backend,
            "budget_gpuh": args.budget_gpuh, "margin": args.margin,
            "est_gpuh": {"screen": h_screen, "refine": h_refine,
                         "full": h_full, "total": h_screen + h_refine + h_full},
            "worst_task": {"which": worst_name, "minutes": worst_min,
                           "guard_minutes": 225},
            "cost_model": "seconds per molecule = a + b x (1 - t_start), fitted "
                          "per (backend, arm) over the probe's three start-times",
            "cells": {"screen": {b: len(screen[b]) for b in BACKS},
                      "refine_assumed": {b: len(refine[b]) for b in BACKS},
                      "full": len(full)},
            "sec_per_molecule": per_mol,
            "screen_ladder": list(SCREEN_LADDER),
            "full_ladder": list(FULL_LADDER),
            "full_min_useful": FULL_MIN_USEFUL,
            "strength_grid": list(BASECMP_STRENGTHS),
            "t_start_grid": list(BASECMP_T_STARTS),
            "rule": "the largest (n_screen, n_full) on the fixed ladders whose "
                    "estimated cost fits --budget-gpuh, screen size preferred "
                    "first",
            "note": "cell time is modelled as proportional to n at fixed batch; "
                    "diversity_logdet is quadratic in n but seconds against "
                    "minutes of sampling, and --margin covers it"}
    print("\nCHOSEN: screen n=%d, full-run n=%d on %s"
          % (n_screen, n_full, BACKENDS[args.full_backend]))
    print("  screen %.1f h + refine %.1f h (assumed) + full %.1f h = %.1f GPU-h "
          "of a %.0f h budget"
          % (h_screen, h_refine, h_full, h_screen + h_refine + h_full,
             args.budget_gpuh))
    print("  worst single array task: %s, %.0f min against the 225-min guard%s"
          % (worst_name, worst_min,
             "" if worst_min <= 225 else
             "  <-- OVER: that task will stop on the guard and leave cells for "
             "the sweepers"))
    if worst_min > 225:
        print("  If the sweepers cannot absorb it the screen stalls and needs a "
              "manual resubmit, which is what the array split exists to avoid. "
              "Consider a smaller n, or splitting basecmp_run.slurm's arm groups "
              "further (ARM_GROUPS / N_GROUPS -- the task count follows them).")
    if n_screen != SCREEN_LADDER[0]:
        print("  NOTE: the screen is n=%d, not the requested %d -- the budget "
              "would not hold both it and a full run that can resolve anything."
              % (n_screen, SCREEN_LADDER[0]))
    if n_full != FULL_LADDER[0]:
        print("  NOTE: the full run is n=%d, not v2's registered %d. Its se on "
              "an in-band difference is ~%.4f against v2's ~0.0025, so report "
              "it as a reduced-n v2 run and never quote it beside v2's own "
              "numbers as though the resolution matched."
              % (n_full, FULL_LADDER[0],
                 (2 * 0.05 * 0.95 / (n_full * len(BASECMP_FULL_SEEDS))) ** 0.5))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        tmp = args.out + ".tmp"
        json.dump(plan, open(tmp, "w"), indent=1)
        os.replace(tmp, args.out)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
