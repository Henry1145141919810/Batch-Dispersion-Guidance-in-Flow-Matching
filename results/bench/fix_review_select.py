"""Adversarial-review fixes to select_arms.py: D1, D8, D9. Run once.

D1 [CRITICAL] There is not a single `unguided` cell in results/sweep -- 141
   cells, arms lgd_mc/osc/plug/smg/smg2/smg2_curv/tfg_mc only. With no
   baseline, `limit` became inf (divergence filter OFF), condition (b) could
   never be satisfied (NO ARM COULD EVER BE DROPPED), and nothing said so.
   Run against the real directory it reported "dropped (0): none" and looked
   like a working screen. It now refuses to emit a verdict without a baseline.

D8 "The best cell" was a max over ALL cells for that arm regardless of target,
   window or sample size. An arm scoring 0.05 at q50 and 0.45 at q90 was
   reported as 0.450 and treated as beating the q50 baseline; an n=16 cell sat
   in the same ceiling as n=512 cells. Comparisons are now made WITHIN a
   (target, window, n) stratum, against the baseline from that same stratum.

D9 `wins_property` was set inside the per-property loop and read after it, so
   it leaked across properties in sorted order. Two arms with mirror-image
   evidence got different verdicts purely because "alpha" sorts before "mu".
   It is now per-property.
"""
import io

NL = chr(10)
p = "proj1/scripts/select_arms.py"
s = io.open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- D8: strata
OLD = '''def summarise(rows, blowup):
    """Per (prop, arm): the best clean cell, plus the divergence record."""
    unguided = {}
    for r in rows:
        if r["arm"] == "unguided":
            u = unguided.setdefault(r["prop"], r)
            if r["n"] > u["n"]:
                unguided[r["prop"]] = r

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["prop"], r["arm"])].append(r)

    out = {}
    for (prop, arm), cells in sorted(by.items()):
        ref = unguided.get(prop)
        limit = blowup * ref["prop_mae_eval"] if ref else float("inf")'''
NEW = '''def stratum(r):
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
        limit = blowup * ref["prop_mae_eval"] if ref else float("inf")'''
assert s.count(OLD) == 1, "summarise anchor"
s = s.replace(OLD, NEW)

OLD_R = '''        out[(prop, arm)] = {
            "best": best, "n_cells": len(cells), "n_bad": len(bad),
            "worst": max((c["prop_mae_eval"] for c in bad), default=0.0)}
    return out, unguided'''
NEW_R = '''        out[(prop, arm)] = {
            "best": best, "n_cells": len(cells), "n_bad": len(bad),
            "ref": ref, "stratum": stratum(best),
            "worst": max((c["prop_mae_eval"] for c in bad), default=0.0)}
    return out, unguided'''
assert s.count(OLD_R) == 1, "summarise return anchor"
s = s.replace(OLD_R, NEW_R)

OLD_N = '''        if not clean:
            out[(prop, arm)] = {"best": None, "n_cells": len(cells),
                                "n_bad": len(bad), "worst": max(
                                    (c["prop_mae_eval"] for c in bad), default=0.0)}
            continue'''
NEW_N = '''        if not clean:
            out[(prop, arm)] = {"best": None, "n_cells": len(cells),
                                "n_bad": len(bad), "ref": ref, "stratum": None,
                                "worst": max(
                                    (c["prop_mae_eval"] for c in bad), default=0.0)}
            continue'''
assert s.count(OLD_N) == 1, "no-clean anchor"
s = s.replace(OLD_N, NEW_N)

# ---------------------------------------------------------------- D8: ceiling
OLD_C = '''    ceiling = {}
    for p in props:
        vals = [v["best"]["in_band_fraction"] for (pp, _a), v in summary.items()
                if pp == p and v["best"]]
        ceiling[p] = max(vals) if vals else 0.0'''
NEW_C = '''    # the ceiling is a max over arms and so is biased upward by however many
    # cells each arm had; it is only used as one half of a two-part test whose
    # other half is "beats unguided", which has no such bias. Restricted to the
    # same stratum so a q90 cell cannot set the bar for q50 arms.
    ceiling = {}
    for p in props:
        for (pp, _a), v in summary.items():
            if pp != p or not v["best"]:
                continue
            key = (p, v["stratum"])
            ceiling[key] = max(ceiling.get(key, 0.0),
                               v["best"]["in_band_fraction"])'''
assert s.count(OLD_C) == 1, "ceiling anchor"
s = s.replace(OLD_C, NEW_C)

OLD_T = '''            top = ceiling[p]'''
NEW_T = '''            top = ceiling.get((p, v["stratum"]), 0.0)'''
assert s.count(OLD_T) == 1, "top anchor"
s = s.replace(OLD_T, NEW_T)

# the baseline now comes from the cell's own stratum
OLD_U = '''            ref = unguided.get(p)
            if ref:'''
NEW_U = '''            ref = v.get("ref")
            if ref:'''
assert s.count(OLD_U) == 1, "ref lookup anchor"
s = s.replace(OLD_U, NEW_U)

# ---------------------------------------------------------------- D9
OLD_W = '''        wins_property = False
        tradeoff = []
        notes = []'''
NEW_W = '''        tradeoff = []
        notes = []'''
assert s.count(OLD_W) == 1, "flags anchor"
s = s.replace(OLD_W, NEW_W)

OLD_L = '''            measured_anywhere = True
            b = v["best"]'''
NEW_L = '''            measured_anywhere = True
            # PER PROPERTY. This used to be initialised outside the loop, so a
            # win on `alpha` silently excused a chemistry collapse on `mu`
            # while the mirror-image arm got a bare PROCEED -- the verdict
            # depended on alphabetical order.
            wins_property = False
            b = v["best"]'''
assert s.count(OLD_L) == 1, "loop anchor"
s = s.replace(OLD_L, NEW_L)

OLD_A = '''        elif chem_dead_everywhere and not wins_property:'''
NEW_A = '''        elif chem_dead_everywhere and not any_win:'''
assert s.count(OLD_A) == 1, "chem verdict anchor"
s = s.replace(OLD_A, NEW_A)

OLD_AW = '''                if better_band or better_mae:
                    wins_property = True'''
NEW_AW = '''                if better_band or better_mae:
                    wins_property = True
                    any_win = True'''
assert s.count(OLD_AW) == 1, "any_win anchor"
s = s.replace(OLD_AW, NEW_AW)

OLD_AI = '''        beaten_everywhere = True
        useless_everywhere = True
        chem_dead_everywhere = True
        measured_anywhere = False'''
NEW_AI = '''        beaten_everywhere = True
        useless_everywhere = True
        chem_dead_everywhere = True
        measured_anywhere = False
        any_win = False'''
assert s.count(OLD_AI) == 1, "any_win init anchor"
s = s.replace(OLD_AI, NEW_AI)

# ---------------------------------------------------------------- D1
OLD_M = '''    summary, unguided = summarise(rows, args.blowup)'''
NEW_M = '''    summary, unguided = summarise(rows, args.blowup)

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
        return 2'''
assert s.count(OLD_M) == 1, "main summarise anchor"
s = s.replace(OLD_M, NEW_M)

# the table should say which stratum a row came from
OLD_H = '''    hdr = ("%-16s %-6s %7s %8s %9s %8s %8s %7s %6s"
           % ("arm", "prop", "in_band", "+-se", "MAE/delta", "+-se", "mol_stab",
              "cells", "bad"))'''
NEW_H = '''    hdr = ("%-16s %-6s %-14s %7s %8s %9s %8s %8s %7s %6s"
           % ("arm", "prop", "target/win/n", "in_band", "+-se", "MAE/delta",
              "+-se", "mol_stab", "cells", "bad"))'''
assert s.count(OLD_H) == 1, "header anchor"
s = s.replace(OLD_H, NEW_H)

OLD_RW = '''            print("%-16s %-6s %7.3f %8.3f %9.2f %8.2f %8.3f %7d %6d"
                  % (a, p, b["in_band_fraction"],
                     se_prop(b["in_band_fraction"], b["n"]),
                     mae_d, s_m, b["mol_stability"], v["n_cells"], v["n_bad"]))'''
NEW_RW = '''            st = "%s/%.2f/%d" % (b.get("target_name", "?"),
                                 b.get("t_min_guide", -1), b["n"])
            print("%-16s %-6s %-14s %7.3f %8.3f %9.2f %8.2f %8.3f %7d %6d"
                  % (a, p, st, b["in_band_fraction"],
                     se_prop(b["in_band_fraction"], b["n"]),
                     mae_d, s_m, b["mol_stability"], v["n_cells"], v["n_bad"]))'''
assert s.count(OLD_RW) == 1, "row anchor"
s = s.replace(OLD_RW, NEW_RW)

OLD_NR = '''                print("%-16s %-6s %7s %8s %9s %8s %8s %7d %6d"
                      % (a, p, "-", "-", "-", "-", "-", v["n_cells"], v["n_bad"]))'''
NEW_NR = '''                print("%-16s %-6s %-14s %7s %8s %9s %8s %8s %7d %6d"
                      % (a, p, "-", "-", "-", "-", "-", "-",
                         v["n_cells"], v["n_bad"]))'''
assert s.count(OLD_NR) == 1, "empty row anchor"
s = s.replace(OLD_NR, NEW_NR)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("select_arms.py: D1, D8, D9 applied")
