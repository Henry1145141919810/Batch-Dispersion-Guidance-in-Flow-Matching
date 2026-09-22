"""Screening -> selection: decide which arms go to the full-scale final run.

    python proj1/scripts/select_arms.py --stage v2
    python proj1/scripts/select_arms.py --stage v2 --emit-arms      # for the slurm job

THE DECISION RULE IS DELIBERATELY ASYMMETRIC. The screening sample size is 512,
so a difference of a couple of points in band coverage is inside the noise and
carries no information. The instruction this implements is "prefer to run more
but not miss -- only filter out obvious non-competitors", so the DEFAULT IS TO
KEEP and an arm is dropped only when the evidence against it is unambiguous.

An arm is DROPPED only if, on EVERY property, both of:

  (a) it is separated from the best arm by more than `--sigma` combined
      standard errors ON MAE, AND
  (b) it is no better than doing nothing on EITHER metric -- neither its band
      coverage nor its MAE beats the unguided baseline by `--sigma` combined
      standard errors

...or it collapses chemistry on every property AND wins nothing. Chemistry
alone never drops an arm: an arm that dominates the property metric and costs
stability is a trade-off to report and re-tune at lower strength, not a
non-competitor. Anything else is kept, including anything the screening could
not measure.

UNCERTAINTY IS COMPUTED, NOT ASSUMED.

  band coverage   a binomial proportion:  se = sqrt(p(1-p)/n)
  MAE             |e| has mean MAE and second moment RMSE^2 by definition, so
                  var(|e|) = RMSE^2 - MAE^2 exactly and se = sqrt(var/n). No
                  distributional assumption is involved.

WHY CLAUSE (a) USES MAE AND NOT COVERAGE -- a post-hoc change, recorded.
The first real stage-v2 screen dropped 0 of 14 arms, because clause (a) was
measured on band coverage: a binomial with se ~ 0.012 at n=512, which cannot
separate anything. The same comparisons run at 6-9 sigma on MAE. Coverage is
the acceptance criterion and stays in the report; it is a poor test statistic
because it discards every sample's magnitude. This rule was changed AFTER
seeing that screen, which is why it is written down here. The three arms it
drops (band, rch, spbc) fail clause (b) on BOTH metrics independently, so the
change moved only clause (a), and by margins of 6-9 sigma against under 1.3.

DIVERGENCE IS NOT ALLOWED TO WIN. Taking each arm's best cell silently rewards
an arm that is excellent at one strength and catastrophic at another -- exactly
how a 43,599,656 MAE with 47/512 nonfinite samples stayed hidden behind a
"best strength" table earlier in this project. So:

  * cells with nonfinite samples, or with MAE above `--blowup` x the unguided
    MAE, are EXCLUDED from an arm's best cell, and
  * the arm is reported as FRAGILE with the count, which is a property of the
    arm that belongs in the writeup, not a footnote.

An arm that is fragile is still kept if it competes -- fragility changes what
must be reported about it, not whether it is measured.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SWEEP = os.path.join(ROOT, "results", "sweep")

# Arms that are references, not candidates: they are always carried into the
# final run because the comparison is meaningless without them.
ALWAYS_KEEP = ["unguided", "plug", "tmpd", "smg"]


def se_prop(p, n):
    """Standard error of a binomial proportion."""
    if not n:
        return float("inf")
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def se_mae(mae, rmse, n):
    """se of the mean absolute error, from its first two moments."""
    if not n:
        return float("inf")
    var = max(rmse ** 2 - mae ** 2, 0.0)
    return math.sqrt(var / n)


def load(stage, props):
    rows = []
    for fn in sorted(glob.glob(os.path.join(SWEEP, "*.json"))):
        try:
            d = json.load(open(fn))
        except Exception:
            continue
        if "prop_mae_eval" not in d:
            continue
        if props and d.get("prop") not in props:
            continue
        if stage != "all" and d.get("stage", "main") != stage:
            # stage-main cells have no "stage" key; keep them when asked for
            # main, and also keep them as references for a v2 comparison
            if not (stage == "v2" and d.get("arm") in ALWAYS_KEEP):
                continue
        d["_file"] = os.path.basename(fn)
        rows.append(d)
    return rows


def stratum(r):
    """Cells are only comparable within the same target, window and n.

    Mixing them let an arm's q90 cell be scored against a q50 baseline, and an
    n=16 cell share a ceiling with n=512 cells."""
    return (r.get("target_name", "q50"), r.get("t_min_guide"), r["n"])


def summarise(rows, blowup):
    """Per (prop, arm): the best clean cell WITHIN a stratum, plus divergences.

    The baseline is the unguided cell from the SAME stratum, so the primary
    stratum is whichever one has both the most arms and a baseline."""
    unguided = {}
    for r in rows:
        if r["arm"] == "unguided":
            key = (r["prop"], stratum(r))
            u = unguided.setdefault(key, r)
            if r["n"] > u["n"]:
                unguided[key] = r

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["prop"], r["arm"])].append(r)

    out = {}
    for (prop, arm), cells in sorted(by.items()):
        # score each arm in the stratum where its baseline exists and the most
        # cells were run; never pool across strata
        cells = [c for c in cells if (prop, stratum(c)) in unguided] or cells
        ref = None
        if cells:
            best_str = collections.Counter(stratum(c) for c in cells).most_common(1)[0][0]
            cells = [c for c in cells if stratum(c) == best_str]
            ref = unguided.get((prop, best_str))
        limit = blowup * ref["prop_mae_eval"] if ref else float("inf")
        clean, bad = [], []
        for c in cells:
            if c.get("n_nonfinite", 0) > 0 or c["prop_mae_eval"] > limit:
                bad.append(c)
            else:
                clean.append(c)
        if not clean:
            out[(prop, arm)] = {"best": None, "n_cells": len(cells),
                                "n_bad": len(bad), "ref": ref, "stratum": None,
                                "worst": max(
                                    (c["prop_mae_eval"] for c in bad), default=0.0)}
            continue
        # best by band coverage; MAE breaks ties, because two cells can share a
        # coverage value at this sample size
        best = max(clean, key=lambda c: (c["in_band_fraction"],
                                         -c["prop_mae_eval"]))
        out[(prop, arm)] = {
            "best": best, "n_cells": len(cells), "n_bad": len(bad),
            "ref": ref, "stratum": stratum(best),
            "worst": max((c["prop_mae_eval"] for c in bad), default=0.0)}
    return out, unguided


def decide(summary, unguided, sigma, chem_floor):
    """PROCEED / DROP / REVIEW per arm, aggregated across properties."""
    props = sorted({p for (p, _a) in summary})
    arms = sorted({a for (_p, a) in summary})

    # the best clean band coverage achieved by ANY arm on each property
    # the ceiling is a max over arms and so is biased upward by however many
    # cells each arm had; it is only used as one half of a two-part test whose
    # other half is "beats unguided", which has no such bias. Restricted to the
    # same stratum so a q90 cell cannot set the bar for q50 arms.
    ceiling = {}          # best band coverage -- reported, not tested on
    floor = {}            # best (lowest) MAE and its se -- what clause (a) tests
    for p in props:
        for (pp, _a), v in summary.items():
            if pp != p or not v["best"]:
                continue
            b = v["best"]
            key = (p, v["stratum"])
            ceiling[key] = max(ceiling.get(key, 0.0), b["in_band_fraction"])
            cand = (b["prop_mae_eval"],
                    se_mae(b["prop_mae_eval"], b["prop_rmse_eval"], b["n"]))
            if key not in floor or cand[0] < floor[key][0]:
                floor[key] = cand

    verdicts = {}
    for a in arms:
        beaten_everywhere = True
        useless_everywhere = True
        chem_dead_everywhere = True
        measured_anywhere = False
        any_win = False
        tradeoff = []
        notes = []
        for p in props:
            v = summary.get((p, a))
            if not v or not v["best"]:
                # never measured cleanly on this property -- cannot be used as
                # evidence against the arm
                beaten_everywhere = False
                useless_everywhere = False
                chem_dead_everywhere = False
                notes.append("%s: no clean cell" % p)
                continue
            measured_anywhere = True
            # PER PROPERTY. This used to be initialised outside the loop, so a
            # win on `alpha` silently excused a chemistry collapse on `mu`
            # while the mirror-image arm got a bare PROCEED -- the verdict
            # depended on alphabetical order.
            wins_property = False
            b = v["best"]
            n = b["n"]
            ib = b["in_band_fraction"]
            s_ib = se_prop(ib, n)
            # clause (a), on MAE: is this arm clearly worse than the best?
            bm, bse = floor.get((p, v["stratum"]), (float("-inf"), 0.0))
            m_here = b["prop_mae_eval"]
            s_here = se_mae(m_here, b["prop_rmse_eval"], n)
            comb = math.sqrt(s_here ** 2 + bse ** 2)
            if m_here - sigma * comb <= bm:
                beaten_everywhere = False

            ref = v.get("ref")
            if ref:
                # (b) "no better than doing nothing" must fail on BOTH metrics.
                # Band coverage is a binomial and has little power at screening
                # n; the MAE uses every sample's magnitude and has much more.
                # Requiring both makes a drop strictly harder to justify.
                ru = ref["in_band_fraction"]
                s_u = math.sqrt(s_ib ** 2 + se_prop(ru, ref["n"]) ** 2)
                better_band = ib > ru + sigma * s_u
                m_a = b["prop_mae_eval"]
                m_u = ref["prop_mae_eval"]
                s_ma = se_mae(m_a, b["prop_rmse_eval"], n)
                s_mu = se_mae(m_u, ref["prop_rmse_eval"], ref["n"])
                better_mae = m_a < m_u - sigma * math.sqrt(s_ma ** 2 + s_mu ** 2)
                if better_band or better_mae:
                    useless_everywhere = False
                if better_band or better_mae:
                    wins_property = True
                    any_win = True

                # chemistry, as a noise-aware proportion rather than a bare
                # ratio: at screening n, 0.094 vs 0.125 is not a difference
                cs, cu = b["mol_stability"], ref["mol_stability"]
                s_c = math.sqrt(se_prop(cs, n) ** 2 + se_prop(cu, ref["n"]) ** 2)
                if cs >= chem_floor * cu - sigma * s_c:
                    chem_dead_everywhere = False
                elif wins_property:
                    tradeoff.append("%s: mol_stab %.3f vs unguided %.3f while "
                                    "winning the property" % (p, cs, cu))
            else:
                useless_everywhere = False
                chem_dead_everywhere = False

        notes.extend(tradeoff)
        if a in ALWAYS_KEEP:
            verdicts[a] = ("KEEP (reference)", notes)
        elif not measured_anywhere:
            verdicts[a] = ("REVIEW (no clean cell anywhere)", notes)
        elif chem_dead_everywhere and not any_win:
            # chemistry collapse is disqualifying only when the arm is not
            # buying anything with it
            verdicts[a] = ("DROP (chemistry collapse, wins nothing)", notes)
        elif beaten_everywhere and useless_everywhere:
            verdicts[a] = ("DROP (beaten on MAE everywhere and no better "
                           "than unguided on either metric)", notes)
        elif tradeoff:
            verdicts[a] = ("PROCEED (chemistry trade-off -- re-tune strength "
                           "at full scale)", notes)
        else:
            verdicts[a] = ("PROCEED", notes)
    return verdicts, props


def main():
    global SWEEP
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="v2", choices=["main", "v2", "all"])
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--sigma", type=float, default=3.0,
                    help="how many combined standard errors count as a clear "
                         "separation; higher keeps more arms")
    ap.add_argument("--blowup", type=float, default=10.0,
                    help="a cell whose MAE exceeds this multiple of the "
                         "unguided MAE is treated as a divergence")
    ap.add_argument("--chem-floor", type=float, default=0.5,
                    help="molecular stability below this multiple of unguided "
                         "counts as chemistry collapse")
    ap.add_argument("--sweep-dir", default=SWEEP,
                    help="where the cell jsons live; the smoke directory can "
                         "be pointed at here without touching the real one")
    ap.add_argument("--emit-arms", action="store_true",
                    help="print only the comma-separated surviving arm list")
    args = ap.parse_args()

    SWEEP = args.sweep_dir
    props = [p for p in args.props.split(",") if p]
    rows = load(args.stage, props)
    if not rows:
        print("no cells found for stage %r -- run the sweep first" % args.stage)
        return 1
    summary, unguided = summarise(rows, args.blowup)

    # WITHOUT A BASELINE THIS SCRIPT IS A NO-OP AND SAYS NOTHING ABOUT IT.
    # With no unguided cell: `limit` is inf so the divergence filter is off,
    # and condition (b) can never be satisfied so no arm can ever be dropped.
    # Run against the real sweep directory -- which has 141 cells and zero
    # unguided cells -- it printed "dropped (0): none" and looked like a
    # working screen. It must refuse instead.
    missing = [p for p in props
               if not any(pp == p for (pp, _st) in unguided)]
    if missing:
        print("NO UNGUIDED BASELINE for: %s" % ",".join(missing))
        print()
        print("Without it the divergence filter is disabled and no arm can be")
        print("dropped, so any verdict this script printed would be vacuous.")
        print("Run the unguided cells first:")
        print("    python proj1/scripts/guidance_sweep.py --stage %s "
              "--arms unguided --props %s" % (args.stage, ",".join(missing)))
        return 2
    verdicts, props_found = decide(summary, unguided, args.sigma, args.chem_floor)

    survivors = [a for a, (v, _n) in sorted(verdicts.items())
                 if not v.startswith("DROP")]
    if args.emit_arms:
        print(",".join(survivors))
        return 0

    print("screening summary -- stage %s, %d cells, sigma=%g, blowup=%gx"
          % (args.stage, len(rows), args.sigma, args.blowup))
    print()
    hdr = ("%-16s %-6s %-14s %7s %8s %9s %8s %8s %7s %6s"
           % ("arm", "prop", "target/win/n", "in_band", "+-se", "MAE/delta",
              "+-se", "mol_stab", "cells", "bad"))
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(verdicts):
        for p in props_found:
            v = summary.get((p, a))
            if not v:
                continue
            if not v["best"]:
                print("%-16s %-6s %-14s %7s %8s %9s %8s %8s %7d %6d"
                      % (a, p, "-", "-", "-", "-", "-", "-",
                         v["n_cells"], v["n_bad"]))
                continue
            b = v["best"]
            d = b["delta"]
            mae_d = b["prop_mae_eval"] / d
            s_m = se_mae(b["prop_mae_eval"], b["prop_rmse_eval"], b["n"]) / d
            st = "%s/%.2f/%d" % (b.get("target_name", "?"),
                                 b.get("t_min_guide", -1), b["n"])
            print("%-16s %-6s %-14s %7.3f %8.3f %9.2f %8.2f %8.3f %7d %6d"
                  % (a, p, st, b["in_band_fraction"],
                     se_prop(b["in_band_fraction"], b["n"]),
                     mae_d, s_m, b["mol_stability"], v["n_cells"], v["n_bad"]))
        print()

    print("verdicts")
    print("-" * 78)
    for a in sorted(verdicts):
        v, notes = verdicts[a]
        print("  %-16s %s" % (a, v))
        for nt in notes:
            print("  %-16s   note: %s" % ("", nt))
    print()
    print("survivors (%d): %s" % (len(survivors), ",".join(survivors)))
    dropped = [a for a in sorted(verdicts) if verdicts[a][0].startswith("DROP")]
    print("dropped   (%d): %s" % (len(dropped), ",".join(dropped) or "none"))
    print()
    print("Arms marked FRAGILE below produced a divergent or nonfinite cell. They")
    print("are still carried forward if they compete; the fragility is a result.")
    for (p, a), v in sorted(summary.items()):
        if v["n_bad"]:
            print("  FRAGILE %-16s %-6s %d/%d cells diverged, worst MAE %.3g"
                  % (a, p, v["n_bad"], v["n_cells"], v["worst"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
