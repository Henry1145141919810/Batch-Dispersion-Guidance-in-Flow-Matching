"""Gates for the joint strength x start-time sweep and its freeze.

    python proj1/tests/test_tune.py

Closed-form only: no model, no dataset, seconds on a login node. Every gate
here is a property the cluster chain depends on, written so a failure names
the thing that broke rather than the assertion that noticed.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

import guidance_sweep as g            # noqa: E402
import freeze_tune as ft              # noqa: E402

PROPS = ["mu", "alpha", "gap"]
R = {}


def gate(name, ok, detail=""):
    R[name] = bool(ok)
    print("%-44s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


# ---------------------------------------------------------------- the plan
cells = g.plan_tune_cells(PROPS)
want = 3 + 6 * 3 * len(g.TUNE_STRENGTHS) * len(g.TUNE_WINDOWS)
gate("plan_tune_cell_count", len(cells) == want, "%d (want %d)" % (len(cells), want))
gate("plan_tune_unique", len(set(cells)) == len(cells))
gate("plan_tune_one_target", {c[2] for c in cells} == {g.TUNE_TARGET})
gate("plan_tune_variant", {c[5] for c in cells} == {g.TUNE_VARIANT})
gate("plan_tune_unguided_once",
     sum(1 for c in cells if c[1] == "unguided") == 3,
     "unguided applies no guidance, so w and t_min cannot change it")
gate("plan_tune_all_seven_arms",
     {c[1] for c in cells} == set(g.TUNE_ARMS), str(sorted({c[1] for c in cells})))
# pass 1 must come first, or a truncated job leaves no chemistry floor
gate("plan_tune_unguided_first",
     all(c[1] == "unguided" for c in cells[:3]),
     "the floor's reference must be written before anything else")
# the first complete slice must be the window v2 used
first_win = [c for c in cells if c[1] != "unguided"][0][4]
gate("plan_tune_v2_window_first", first_win == g.V2_WIN, "got %g" % first_win)
# names must be distinct per cell, and must not collide with any other stage
names = {g.cell_name(*c) for c in cells}
gate("plan_tune_names_unique", len(names) == len(cells))
gate("plan_tune_names_tagged", all(n.endswith("__tune.json") for n in names))

# THE CLUSTER CALLS THIS ONE WINDOW AT A TIME, which no gate exercised before.
# A pass that ran V2_WIN unconditionally made every non-0.5 array task re-plan
# the whole 0.5 slice: 819 cell-runs for 489 unique cells.
for _w in g.TUNE_WINDOWS:
    _c = g.plan_tune_cells(PROPS, windows=[_w])
    _wins = {x[4] for x in _c if x[1] != "unguided"}
    gate("plan_tune_single_window_%g" % _w, _wins == {_w},
         "asked for %g, planned %s" % (_w, sorted(_wins)))
    _want = 3 + 6 * 3 * len(g.TUNE_STRENGTHS)
    gate("plan_tune_single_window_count_%g" % _w, len(_c) == _want,
         "%d (want %d)" % (len(_c), _want))
# the union of the three single-window plans must equal the full plan exactly,
# or the array as a whole samples something different from the plan
_union = set()
for _w in g.TUNE_WINDOWS:
    _union |= set(g.plan_tune_cells(PROPS, windows=[_w]))
gate("plan_tune_array_union_is_the_plan", _union == set(cells),
     "%d vs %d" % (len(_union), len(cells)))

# the variant string must not accidentally look like a btvg tau override
kw = g.arm_kwargs("btvg", g.TUNE_VARIANT, 0.5)
gate("tune_variant_not_a_tau_override", abs(kw.get("tau", 0) - 0.5 / 1.96) < 1e-12,
     "arm_kwargs saw variant=%r and returned tau=%r" % (g.TUNE_VARIANT, kw.get("tau")))
kw2 = g.arm_kwargs("btvg", "tuned", 0.5)
gate("tuned_variant_not_a_tau_override", abs(kw2.get("tau", 0) - 0.5 / 1.96) < 1e-12)

# the grid must reach past where the chemistry floor can still bind. v2
# measured lgd_mc BELOW the floor at w = 8, so a grid topping out at 4 would
# cap the strongest arm with the grid rather than with chemistry.
gate("tune_grid_passes_floor_binding", max(g.TUNE_STRENGTHS) >= 8.0,
     "max strength %g" % max(g.TUNE_STRENGTHS))
gate("tune_windows_span", len(g.TUNE_WINDOWS) >= 3
     and min(g.TUNE_WINDOWS) <= 0.05 and max(g.TUNE_WINDOWS) >= 0.75)


# ------------------------------------------------------------- the freeze
def cell(prop, arm, w, win, ib, stab, ib_dec=None, uniq=1.0, delta=0.16992):
    return {
        "prop": prop, "arm": arm, "w": w, "t_min_guide": win,
        "in_band_fraction": ib, "in_band_fraction_dec": ib if ib_dec is None else ib_dec,
        "mol_stability": stab, "validity": 0.75, "uniqueness_of_valid": uniq,
        "unique_valid_per_sample": 0.74, "prop_mae_eval": 1.0,
        "delta": delta, "delta_source": "local", "target_name": "q90",
        "n": 1000, "seed": 20260925, "steps": 100, "solver": "euler",
        "clip": 1.0, "batch": 128, "k_delta": 2.0, "n_mc": 4, "sigma_mc": 0.1,
        "prov": {"fm_md5": "a190ac83"},
    }


def write(tmp, rows):
    for r in rows:
        n = "%s__%s__q90__w%g__tmin%g__tune.json" % (
            r["prop"], r["arm"], r["w"], r["t_min_guide"])
        json.dump(r, open(os.path.join(tmp, n), "w"))


def freeze(rows, props=("mu",), arms=("unguided", "plug"), partial=True):
    tmp = tempfile.mkdtemp()
    try:
        write(tmp, rows)
        sys.argv = ["freeze_tune.py", "--root", tmp, "--props", ",".join(props),
                    "--arms", ",".join(arms)] + (["--allow-partial"] if partial else [])
        return ft.main()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# floor vs open must differ when the best in-band is bought with chemistry
rows = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40)]
rows += [cell("mu", "plug", 1.0, 0.5, 0.05, 0.38),     # clears floor (0.36)
         cell("mu", "plug", 4.0, 0.5, 0.09, 0.20)]     # best, fails floor
out = freeze(rows)
pk = out["picks"]["plug"]["mu"]
gate("freeze_floor_pick_respects_floor", pk["floor"]["w"] == 1.0 and pk["floor"]["in_band"] == 0.05)
gate("freeze_open_pick_ignores_floor", pk["open"]["w"] == 4.0 and pk["open"]["in_band"] == 0.09)
gate("freeze_same_flag_false", pk["same"] is False)
gate("freeze_floor_value", abs(out["floor"]["mu"] - 0.36) < 1e-12,
     "0.9 x unguided 0.40 = %.4f" % out["floor"]["mu"])

# when the best in-band already clears the floor the two picks coincide and
# the full run must sample the cell ONCE
rows = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
        cell("mu", "plug", 1.0, 0.5, 0.05, 0.38),
        cell("mu", "plug", 4.0, 0.5, 0.09, 0.39)]
out = freeze(rows)
pk = out["picks"]["plug"]["mu"]
gate("freeze_same_flag_true", pk["same"] is True and pk["floor"]["w"] == 4.0)
n_plug = sum(1 for c in out["full_run_cells"] if c["arm"] == "plug")
gate("freeze_dedupes_full_run_cells", n_plug == 1, "%d cells for plug" % n_plug)

# tie-breaks: equal in-band -> smaller w, then LARGER t_min
rows = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
        cell("mu", "plug", 2.0, 0.5, 0.07, 0.38),
        cell("mu", "plug", 1.0, 0.05, 0.07, 0.38),
        cell("mu", "plug", 1.0, 0.75, 0.07, 0.38)]
out = freeze(rows)
f = out["picks"]["plug"]["mu"]["floor"]
gate("freeze_tie_smaller_w_then_later_start",
     f["w"] == 1.0 and f["t_min_guide"] == 0.75,
     "picked w=%g t=%g" % (f["w"], f["t_min_guide"]))

# an arm that clears nowhere must be flagged, not silently frozen
rows = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
        cell("mu", "plug", 4.0, 0.5, 0.09, 0.10)]
out = freeze(rows)
gate("freeze_flags_no_floor_clearing",
     out["picks"]["plug"]["mu"]["floor"] is None
     and "mu/plug" in out["flags"]["no_floor_clearing_cell"])

# grid-limited: the floor pick sits at the top of the grid and still clears
rows = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40)]
rows += [cell("mu", "plug", w, 0.5, 0.01 * w, 0.38) for w in g.TUNE_STRENGTHS]
out = freeze(rows)
gate("freeze_flags_grid_limited", "mu/plug floor" in out["flags"]["grid_limited"],
     "floor pick at w=%g" % out["picks"]["plug"]["mu"]["floor"]["w"])

# collapse must be flagged at the pick
rows = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
        cell("mu", "plug", 1.0, 0.5, 0.09, 0.38, uniq=0.5)]
out = freeze(rows)
gate("freeze_flags_collapse", any("mu/plug" in x for x in out["flags"]["collapse"]))

# the continuous / decoded disagreement must be surfaced, not resolved
rows = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
        cell("mu", "plug", 1.0, 0.5, 0.05, 0.38, ib_dec=0.09),
        cell("mu", "plug", 2.0, 0.5, 0.07, 0.38, ib_dec=0.04)]
out = freeze(rows)
gate("freeze_flags_metric_disagreement", len(out["flags"]["metric_disagrees"]) > 0,
     out["flags"]["metric_disagrees"][:1])
gate("freeze_selects_on_continuous",
     out["picks"]["plug"]["mu"]["floor"]["w"] == 2.0,
     "v2's rule is the continuous metric; the decoded pick is only flagged")


# older cells carry no decoded metric; the freeze must name that, not crash
_nodec = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
          cell("mu", "plug", 1.0, 0.5, 0.05, 0.38)]
for _r in _nodec:
    _r.pop("in_band_fraction_dec")
try:
    freeze(_nodec)
    gate("freeze_handles_missing_decoded", False, "no refusal")
except SystemExit as _e:
    gate("freeze_handles_missing_decoded", _e.code == 2, "exit %r" % _e.code)
except Exception as _e:                                   # noqa: BLE001
    gate("freeze_handles_missing_decoded", False, "crashed: %r" % _e)


def refuses(rows, **kw):
    try:
        freeze(rows, **kw)
        return False
    except SystemExit as e:
        return e.code == 2


# two deltas in one tree is two experiments: btvg's tau is delta/1.96
bad = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
       cell("mu", "plug", 1.0, 0.5, 0.05, 0.38, delta=0.16799)]
gate("freeze_refuses_mixed_delta", refuses(bad))

# no unguided cell -> the floor is undefined
gate("freeze_refuses_without_unguided",
     refuses([cell("mu", "plug", 1.0, 0.5, 0.05, 0.38)]))

# a degenerate floor (unguided stability ~0) must refuse: every cell would
# clear it and the two picks would coincide by construction, not by result
gate("freeze_refuses_degenerate_floor",
     refuses([cell("mu", "unguided", 1.0, 0.5, 0.0, 0.0),
              cell("mu", "plug", 1.0, 0.5, 0.05, 0.0)]))

# a missing planned cell must refuse unless --allow-partial
gate("freeze_refuses_partial_by_default",
     refuses([cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
              cell("mu", "plug", 1.0, 0.5, 0.05, 0.38)], partial=False))

# mixed sampler settings must refuse
bad = [cell("mu", "unguided", 1.0, 0.5, 0.03, 0.40),
       cell("mu", "plug", 1.0, 0.5, 0.05, 0.38)]
bad[1]["steps"] = 50
gate("freeze_refuses_mixed_sampler", refuses(bad))


# ------------------------------------------------- the full-run plan
frozen = {
    "schema": "frozen_tune/1", "target": "q90",
    "picks": {
        "unguided": {p: {"floor": {"w": 1.0, "t_min_guide": 0.5},
                         "open": {"w": 1.0, "t_min_guide": 0.5}, "same": True}
                     for p in PROPS},
        "plug": {p: {"floor": {"w": 1.0, "t_min_guide": 0.5},
                     "open": {"w": 4.0, "t_min_guide": 0.05}, "same": False}
                 for p in PROPS},
        "tfg": {p: {"floor": {"w": 0.05, "t_min_guide": 0.75},
                    "open": {"w": 0.05, "t_min_guide": 0.75}, "same": True}
                for p in PROPS},
    },
}
fc = g.plan_tuned_full_cells(PROPS, frozen)
gate("tuned_full_dedupes", len(fc) == 3 * (1 + 2 + 1),
     "%d cells (unguided 1 + plug 2 + tfg 1 per property)" % len(fc))
gate("tuned_full_unique", len(set(fc)) == len(fc))
gate("tuned_full_variant", {c[5] for c in fc} == {"tuned"})
gate("tuned_full_carries_window",
     any(abs(c[4] - 0.75) < 1e-12 for c in fc),
     "each arm runs at ITS OWN start time, not a shared one")
gate("tuned_full_names_distinct", len({g.cell_name(*c) for c in fc}) == len(fc))
# unguided is the floor's reference and every z-test's denominator; the
# per-task budget can cut a group short, so it must never be planned last.
# frozen_tune.json is written with sort_keys, which would put it last.
_pos = [i for i, c in enumerate(fc) if c[1] == "unguided"]
_per = len(fc) // len(PROPS)
gate("tuned_full_unguided_first_in_each_group",
     all(i % _per == 0 for i in _pos), "unguided at positions %s of %d" % (_pos, _per))
# names must not collide with the v2 run's cells
v2ish = "mu__plug__q90__w1__tmin0.5__full.json"
gate("tuned_full_no_v2_name_collision",
     all(g.cell_name(*c) != v2ish for c in fc))
try:
    g.plan_tuned_full_cells(PROPS, frozen, arms=["btvg"])
    gate("tuned_full_rejects_unknown_arm", False)
except SystemExit:
    gate("tuned_full_rejects_unknown_arm", True)

# the freeze writes full_run_cells; the planner re-derives them. If those two
# ever disagree the full run samples a different table from the one frozen.
rows = [cell(p, "unguided", 1.0, 0.5, 0.03, 0.40) for p in PROPS]
for p in PROPS:
    rows += [cell(p, "plug", 1.0, 0.5, 0.05, 0.38),
             cell(p, "plug", 4.0, 0.05, 0.09, 0.20),
             cell(p, "tfg", 0.05, 0.75, 0.04, 0.39)]
out = freeze(rows, props=PROPS, arms=("unguided", "plug", "tfg"))
planned = {(c[0], c[1], c[3], c[4]) for c in g.plan_tuned_full_cells(PROPS, out)}
frozen_set = {(c["prop"], c["arm"], c["w"], c["t_min_guide"])
              for c in out["full_run_cells"]}
gate("freeze_cell_list_matches_planner", planned == frozen_set,
     "%d planned vs %d frozen" % (len(planned), len(frozen_set)))

bad = sorted(k for k, v in R.items() if not v)
print("")
print("ALL PASS (%d gates)" % len(R) if not bad else "FAILED: %s" % ", ".join(bad))
sys.exit(1 if bad else 0)
