"""Act on the adversarial review of the `dist` protocol. Run once.

S1 BLOCKER -- `dist` was unreachable. `run_cell` handled it but no planner ever
   emitted it and there was no CLI flag, so FR6 ("the full run uses target
   dist") was a promise the code could not keep. Adds `--targets` and a
   `plan_target_cells` planner, reachable as `--stage targets`.

S13 -- `dist` targets now come from **test**, not `val`. `val` already does five
   jobs: generator validation, model selection for BOTH predictors, delta
   (= 2 x f_B val MAE), and the molecule sizes the sweep samples. Drawing the
   targets from it too would make the headline number -- the one FR6 exists to
   make comparable to published results -- rest on a split that every component
   has already seen. `test` (13,083 molecules) is untouched; verified zero
   index overlap with train_a, train_b and val.

S7 -- the same `y_t[: m.shape[0]]` slice bug is ported-fixed in
   transfer_sweep.py and update_rule_bench.py. Harmless there today (both build
   a constant target), but update_rule_bench.py reads GS.TARGETS[prop][...], so
   the first person to pass "dist" would reintroduce it silently, with no error
   and plausible numbers.

S8 -- SPBC is degenerate under per-molecule targets by construction: it centres
   the BATCH MEAN on the MEAN target and ignores each molecule's own. Under
   `dist` the mean target ~ the data mean ~ the unguided mean, so its correction
   is ~0 before any measurement. Recorded in the docstring rather than silently
   producing a null.
"""
import io

NL = chr(10)

# ---------------------------------------------------------------- S1
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

OLD_T = '''TARGETS = {"mu":    {"q50": 2.4932, "q90": 4.6627},
           "alpha": {"q50": 75.54, "q90": 85.05},
           "gap":   {"q50": 0.2496, "q90": 0.3162}}'''
NEW_T = '''TARGETS = {"mu":    {"q50": 2.4932, "q90": 4.6627},
           "alpha": {"q50": 75.54, "q90": 85.05},
           "gap":   {"q50": 0.2496, "q90": 0.3162}}

# "dist" is NOT a fixed value and so is not in TARGETS: each molecule's target
# is the real property of the held-out molecule whose size it takes, resolved
# per cell in run_cell. This is the EDM/EEGSDE/TFG conditional protocol, and it
# is the only target that involves no choice by us -- which is why FR6 makes it
# the headline. Measured steering distance from the unguided generator's own
# mean: dist 0.80-0.85 sd, against q50 0.05-0.27 and q90 1.1-1.6.
DIST_TARGET = "dist"'''
assert s.count(OLD_T) == 1, "TARGETS anchor"
s = s.replace(OLD_T, NEW_T)

OLD_P = '''def main():
    global OUT'''
NEW_P = '''def plan_target_cells(props, arms, targets, win=None, strengths=None):
    """One target-protocol grid: every arm x every strength, at one window.

    Exists so `dist` is schedulable at all (it was not), and so a target grid
    can be run without touching the q50/q90 plans already on disk. Ordered so a
    partial run is still a complete experiment: pass 1 is the whole arm set at
    the default strength.
    """
    win = V2_WIN if win is None else win
    strengths = STRENGTHS if strengths is None else strengths
    seen, cells = set(), []

    def add(prop, arm, tgt, w):
        k = (prop, arm, tgt, w, win, "tgt")
        if k not in seen:
            seen.add(k)
            cells.append(k)

    for tgt in targets:
        for prop in props:                   # pass 1: the arm comparison
            for arm in arms:
                add(prop, arm, tgt, DEFAULT_W)
        for prop in props:                   # pass 2: strength
            for arm in arms:
                if arm == "unguided":
                    continue
                for w in strengths:
                    add(prop, arm, tgt, w)
    return cells


def main():
    global OUT'''
assert s.count(OLD_P) == 1, "main anchor"
s = s.replace(OLD_P, NEW_P)

OLD_C = '''    ap.add_argument("--stage", default="main",
                    choices=["main", "v2", "all", "compare"],'''
NEW_C = '''    ap.add_argument("--targets", default="dist",
                    help="comma-separated target protocols for --stage targets: "
                         "'dist' (per-molecule, the published protocol), or any "
                         "TARGETS key such as q50,q90")
    ap.add_argument("--stage", default="main",
                    choices=["main", "v2", "all", "compare", "targets"],'''
assert s.count(OLD_C) == 1, "stage choices anchor"
s = s.replace(OLD_C, NEW_C)

OLD_A = '''    elif args.stage == "compare":
        arms = list(COMPARE_SET)'''
NEW_A = '''    elif args.stage == "compare":
        arms = list(COMPARE_SET)
    elif args.stage == "targets":
        arms = list(COMPARE_SET)'''
assert s.count(OLD_A) == 1, "arms anchor"
s = s.replace(OLD_A, NEW_A)

OLD_B = '''    if args.stage == "compare":
        cells += plan_compare_cells(props)'''
NEW_B = '''    if args.stage == "compare":
        cells += plan_compare_cells(props)
    if args.stage == "targets":
        cells += plan_target_cells(
            props, arms, [t for t in args.targets.split(",") if t])'''
assert s.count(OLD_B) == 1, "cells anchor"
s = s.replace(OLD_B, NEW_B)

# ---------------------------------------------------------------- S13
OLD_D = '''        if tgt == "dist":
            # the field's protocol: each molecule's target is the real property
            # of the SAME validation molecule whose size it already takes
            y_real = d["y"][va, d["props"].index(prop)].to(dev).float()
            target = float(y_real.mean())'''
NEW_D = '''        if tgt == DIST_TARGET:
            # The field's protocol: each molecule gets its OWN target, the real
            # property of a held-out molecule.
            #
            # Drawn from TEST, not val. `val` already carries the generator's
            # validation, BOTH predictors' model selection, delta (= 2 x f_B
            # val MAE) and the molecule sizes sampled below. Taking targets
            # from it as well would make the headline number rest on a split
            # every component has already seen. `test` is untouched: verified
            # zero index overlap with train_a, train_b and val.
            te = d["split"]["test"][: args.n]
            y_real = d["y"][te, d["props"].index(prop)].to(dev).float()
            if y_real.shape[0] < args.n:      # pad by cycling, never silently short
                reps = -(-args.n // y_real.shape[0])
                y_real = y_real.repeat(reps)[: args.n]
            target = float(y_real.mean())'''
assert s.count(OLD_D) == 1, "dist anchor"
s = s.replace(OLD_D, NEW_D)

OLD_Y = '''        y_t = (y_real if tgt == "dist"
               else torch.full((args.n,), target, device=dev))'''
NEW_Y = '''        y_t = (y_real if tgt == DIST_TARGET
               else torch.full((args.n,), target, device=dev))'''
assert s.count(OLD_Y) == 1, "y_t anchor"
s = s.replace(OLD_Y, NEW_Y)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---------------------------------------------------------------- S7
for path, anchor in (("proj1/scripts/transfer_sweep.py", "y=y_t[: m.shape[0]]"),
                     ("proj1/scripts/update_rule_bench.py", "y=y_t[: m.shape[0]]")):
    t = io.open(path, encoding="utf-8").read()
    if anchor in t:
        t = t.replace(anchor, "y=y_t[i:i + m.shape[0]]")
        io.open(path, "w", encoding="utf-8", newline="\n").write(t)
        print("slice fixed: %s" % path)

# ---------------------------------------------------------------- S8
p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()
OLD_S = '''    UNITS: this is a displacement in x_t, NOT a score. The sampler must add it
    without the (1-t)/t score-to-velocity conversion.'''
NEW_S = '''    UNITS: this is a displacement in x_t, NOT a score. The sampler must add it
    without the (1-t)/t score-to-velocity conversion.

    DEGENERATE UNDER PER-MOLECULE TARGETS. `d_common` centres the BATCH MEAN of
    f(m) on the MEAN of y, so with the `dist` protocol -- where every molecule
    has its own target -- SPBC ignores each molecule's target entirely and
    corrects only the aggregate. Under `dist` the mean target ~ the data mean ~
    the unguided generator's mean, so the correction is ~0 BY CONSTRUCTION, not
    by measurement. Any `dist` result for spbc (or for the SHG schedules that
    contain an spbc phase) is a null of the method's definition, not evidence
    about it. Correcting this needs a per-molecule formulation, which is a
    different arm.'''
assert s.count(OLD_S) == 1, "spbc units anchor"
s = s.replace(OLD_S, NEW_S)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

print("S1 (--targets + planner), S13 (dist from test), S7 (slices), S8 (spbc note) applied")
