"""Add `--stage compare`: the finalised comparison set, one controlled table.

THE SET (fixed 2026-09-22, after the audit that found `tfg_mc` mislabelled):

  unguided   reference
  plug       DPS, Chung et al. ICLR 2023          -- posterior-mean, no uncertainty
  tmpd       PiGDM / TMPD, ICLR'23 / TMLR'24      -- posterior-mean, analytic uncertainty
  lgd_mc     LGD, Song et al. ICML 2023           -- posterior-mean, Monte Carlo
  dflow      D-Flow, Ben-Hamu et al. ICML 2024    -- TRAJECTORY OPTIMISATION

DROPPED FROM THE SET, with reasons that must survive into the paper:
  tfg_mc  our implementation is TFG's MC-smoothing component only; it lacks
          TFG's clean-state refinement, recurrence and discrete update, and
          measures 1.13 sigma (median) from `lgd_mc`. QM9_INFERENCE_TIME_
          GUIDANCE_BASELINES.md:15 says explicitly not to count one function
          under two names. It stays runnable, it is not a fourth method.
  osc     no citation exists anywhere in this repo.
  rch     prior art, measured, harmful (alpha bias 10.42 delta). Reported as
          measured-and-dropped, not silently omitted.

D-Flow is the only member from the trajectory-optimisation family. Everything
else here, and both of our contributions, forms a one-step posterior mean.
Without it the "comparison" is a within-family ablation.

CONTROLLED VARIABLES: every cell in this stage shares n, steps, solver, window,
target set, seed, clip and predictors. Only the arm and its strength change.
"""
import io

NL = chr(10)
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''V2_ARMS = ["spbc", "btvg", "btvg_var"]'''
NEW = '''V2_ARMS = ["spbc", "btvg", "btvg_var"]

# ---- the finalised comparison set -----------------------------------------
# See this file's header in results/bench/add_compare_stage.py for why each
# member is in and why tfg_mc / osc / rch are out.
COMPARE_SET = ["unguided", "plug", "tmpd", "lgd_mc", "dflow"]

# D-Flow is not a `mode`: it replaces the sampler rather than adding a field.
# `w` scales its learning rate, the way `w` scales the field elsewhere, so the
# strength axis means the same thing. n_iter is FIXED so a strength sweep does
# not quietly become a compute sweep.
DFLOW_LR = 0.05
DFLOW_ITERS = 8'''
assert s.count(OLD) == 1, "v2_arms anchor"
s = s.replace(OLD, NEW)

# ---- planner
OLD_P = '''def main():
    global OUT'''
NEW_P = '''def plan_compare_cells(props):
    """The controlled comparison table: every arm, every strength, one window.

    Ordered so a partial run is still a complete experiment -- pass 1 is the
    whole set at the default strength, so even one link gives a full arm
    comparison before any tuning is explored.
    """
    seen, cells = set(), []

    def add(prop, arm, tgt, w, win):
        k = (prop, arm, tgt, w, win, "cmp")
        if k not in seen:
            seen.add(k)
            cells.append(k)

    for prop in props:                       # pass 1: the comparison itself
        for arm in COMPARE_SET:
            add(prop, arm, "q50", DEFAULT_W, V2_WIN)
    for prop in props:                       # pass 2: strength, per arm
        for arm in COMPARE_SET:
            if arm == "unguided":
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, V2_WIN)
    for prop in props:                       # pass 3: the q90 target
        for arm in COMPARE_SET:
            add(prop, arm, "q90", DEFAULT_W, V2_WIN)
            if arm == "unguided":
                continue
            for w in STRENGTHS:
                add(prop, arm, "q90", w, V2_WIN)
    return cells


def main():
    global OUT'''
assert s.count(OLD_P) == 1, "main anchor"
s = s.replace(OLD_P, NEW_P)

OLD_S = '''    ap.add_argument("--stage", default="main", choices=["main", "v2", "all"],'''
NEW_S = '''    ap.add_argument("--stage", default="main",
                    choices=["main", "v2", "all", "compare"],'''
assert s.count(OLD_S) == 1, "stage choices"
s = s.replace(OLD_S, NEW_S)

OLD_A = '''    elif args.stage == "v2":
        arms = V1_MISSING + V2_ARMS + SHG_ARMS'''
NEW_A = '''    elif args.stage == "v2":
        arms = V1_MISSING + V2_ARMS + SHG_ARMS
    elif args.stage == "compare":
        arms = list(COMPARE_SET)'''
assert s.count(OLD_A) == 1, "arms select"
s = s.replace(OLD_A, NEW_A)

OLD_C = '''    if args.stage in ("v2", "all"):'''
NEW_C = '''    if args.stage == "compare":
        cells += plan_compare_cells(props)
    if args.stage in ("v2", "all"):'''
assert s.count(OLD_C) == 1, "cells build"
s = s.replace(OLD_C, NEW_C)

# ---- run_cell: the dflow branch
OLD_R = '''            c0, f0 = initial_noise(m, len(types), gen)
            kw = dict('''
NEW_R = '''            c0, f0 = initial_noise(m, len(types), gen)
            if arm == "dflow":
                # Trajectory optimisation, not a guidance field: there is no
                # `mode` and no posterior. The sampler is replaced wholesale.
                from dflow import dflow_optimise
                cst = Cost()
                c, f = dflow_optimise(
                    net, f_A, m, c0, f0, y_t[: m.shape[0]],
                    n_steps=args.steps, n_iter=DFLOW_ITERS,
                    lr=DFLOW_LR * w, cost=cst, trust=None)
                cs.append(c); fs.append(f)
                calls += cst.gen_fwd
                for k in cost:
                    cost[k] += getattr(cst, k)
                continue
            kw = dict('''
assert s.count(OLD_R) == 1, "run_cell noise anchor"
s = s.replace(OLD_R, NEW_R)

# Cost must be importable in the sweep
OLD_I = '''from guidance import ResidualCalibrationHead  # noqa: E402'''
NEW_I = '''from guidance import Cost, ResidualCalibrationHead  # noqa: E402'''
assert s.count(OLD_I) == 1, "import anchor"
s = s.replace(OLD_I, NEW_I)

# preflight must not blow up on dflow's cost
OLD_PF = '''        args.n, args.batch, args.steps = 8, 8, 10'''
NEW_PF = '''        args.n, args.batch, args.steps = 8, 8, 10
        globals()["DFLOW_ITERS"] = 2      # preflight exercises the path, not the optimum'''
assert s.count(OLD_PF) == 1, "preflight anchor"
s = s.replace(OLD_PF, NEW_PF)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("--stage compare added; dflow wired into run_cell")
