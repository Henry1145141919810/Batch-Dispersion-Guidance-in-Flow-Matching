"""Run our guidance arms on a BORROWED base model and BORROWED property pair.

THE QUESTION THIS ANSWERS. `guidance_sweep.py` measures every arm on our
flow-matching generator, steered by our `f_A` and scored by our `f_B`. The
split protocol makes f_A and f_B disjoint, so an arm cannot win by exploiting
the scorer -- but generator, guide and evaluator still come from one pipeline,
built by one group, on one preprocessing of QM9. A reviewer is entitled to ask
whether an arm's advantage is a property of the METHOD or a property of our
STACK. Nothing in our own sweep can answer that.

So: take TFG's released `EDMsecond` generator, TFG's released `tf_predict_<p>`
guide and TFG's released `evaluate_<p>` oracle, for the three properties we
already study, and re-run the arms unchanged. Every component except the
guidance field is then external. If our arms keep their ranking here, the
ranking is about the arms. If it inverts, we have learned something more
important than a win.

WHAT IS HELD FIXED ACROSS ARMS, so that differences are attributable:
  base model      EDMsecond (one checkpoint, one md5, stamped per cell)
  guide  f_A      tf_predict_<p>, evaluated at time channel 0
  oracle f_B      evaluate_<p>, the network TFG reports its own MAE with
  sampler         100-step Euler on the probability-flow ODE under EDM's
                  polynomial_2 schedule
  targets         q50 / q90 of the QM9 property distribution
  delta           k x the oracle's measured MAE, k = 2 -- the same
                  pre-registered rule, via the same `choose_delta` helper
  window          guidance off while tau > tau_max_guide (default 0.5)
  clip            velocity-relative trust region, clip = 1.0

WHAT DIFFERS FROM THE MAIN SWEEP, and must be disclosed with every number:

  1. GUIDE/ORACLE DISJOINTNESS IS INFERRED, NOT CONSTRUCTED. Both TFG arg
     files record `dataset: qm9_second_half`, and their training script
     mutates `args.dataset` after building the training loader, so the string
     is unreliable. `audit/fa_fb_search/disjointness_test.py` finds the error
     structure of a disjoint pair and says "DISJOINT CONSISTENT", which is
     strong but circumstantial. Our own pair is disjoint by construction. This
     is the transfer's main weakness and it cannot be fixed from the released
     artifacts.
  2. DELTA IS OPTIMISTIC. It is 2 x the oracle's MAE measured on QM9
     molecules of which an unknown ~50% are in the oracle's own training half,
     so the measured MAE is lower than a clean held-out MAE and delta is
     TIGHTER than it should be. A tighter band lowers in-band coverage for
     every arm equally, so it does not favour any arm -- it makes the absolute
     coverage numbers pessimistic, and they should be read that way.
  3. THE SAMPLER IS OURS, NOT EDM'S. EDM samples a 1000-step ancestral SDE and
     TFG samples 100 DDIM steps at eta = 1. We integrate the probability-flow
     ODE at 100 Euler steps, because that is what our guidance fields are
     defined on and because holding the sampler fixed across arms is what
     makes the comparison internal. Consequence: the UNGUIDED row here is not
     EDM's published unconditional row, and must not be quoted as one.
  4. THE GUIDE IS ASKED AT t = 0, AND ONLY AT t = 0. TFG's own method
     evaluates its time-dependent guide at the noisy state x_tau with that
     tau. Our arms evaluate at the posterior mean m, an estimate of a clean
     molecule, so they ask for t = 0. The alternative is NOT measured here and
     there is no flag for it: doing it honestly needs a calibration refitted
     per t, since `calibrate` below fits once at t = 0. An earlier version of
     this file carried a `--guide-time current` switch that was wired to
     nothing and would have reported "the choice does not matter".
  5. THE GUIDANCE WINDOW IS SET, NOT INHERITED. `--tau-max-guide` defaults to
     0.5, the VP image of the main sweep's `t_min_guide = 0.5` (flow time runs
     0 -> 1 toward data, VP time 1 -> 0, so the windows mirror). This is the
     largest single effect measured anywhere in the project -- alpha MAE
     10.02 -> 5.32 as `t_min_guide` goes 0.05 -> 0.5 -- so leaving `VPSampler`
     at its historical "guide everywhere" default would have run every arm at
     the bad end of that curve, and any ranking inversion would have been
     unattributable. It also matters mechanically: `vp_posterior` divides by
     alpha, and alpha(1.0) = 0.0032 under EDM's schedule, so guiding near
     tau = 1 amplifies epsilon error by ~300x. Run `--tau-max-guide 1.0` as a
     labelled comparison if the window's effect on this backend is wanted.
  6. WHICH f_A / f_B IS PER BACKEND, and was not always. Until 26 Sep this
     file used TFG's pair for every backend -- guide `tf_predict_<p>`, oracle
     `evaluate_<p>`, both from `audit/fa_fb_search/TFG/` via `build_pair` --
     and the docstring said "never ours". Henry changed that: `fm` and the
     QM9 diffusion base now score with OUR pair (`weights/f_{A,B}_<p>.pt`,
     via `build_pair_ours`), `equifm` keeps TFG's. The table is V3_BACKENDS;
     `--pair` overrides it; every cell records `pair`, `guide` and `oracle`,
     so which was used can be checked after the fact.

     WHAT THAT COSTS. delta = k x MAE(f_B), so the pair sets the BAND WIDTH.
     Our f_B is the less accurate of the two on all three properties, so an
     ours-pair backend is scored in a wider band and its in_band is higher
     for that reason alone. IN_BAND IS THEREFORE NOT COMPARABLE ACROSS
     BACKENDS THAT USE DIFFERENT PAIRS. Within one backend every arm shares
     one delta, so the arm-vs-arm comparison -- the question this project
     asks -- is untouched. What it buys is that our pair is disjoint by
     construction, where TFG's `evaluate_<p>` saw an unknown part of QM9 and
     its delta is optimistically tight.

     Both pairs are calibrated (or, for ours, scale-checked) on train_a +
     train_b only, so the `test` split -- where the `dist` targets come from
     -- is touched by no fit, the same claim the main sweep makes.

THE PROTOCOL, the main sweep's, stage for stage (amended 23 Sep, before any
transfer cell existed -- TRANSFER_EXPERIMENT_PLAN.md section 9):

  --stage compare   n=512, one seed. TRANSFER_SET x guidance_sweep.STRENGTHS
                    (7 points) x {q50, q90}; unguided once per target. The
                    q50/q90 values are guidance_sweep.TARGETS, the same numbers
                    the main compare stage used, so the two backends are asked
                    for the same thing.
  --stage freeze    reads the compare cells, refuses a partial grid, and writes
                    the frozen-strength json: FR3a (best q90 MAE among strengths
                    with mol_stability >= 0.9 x unguided) as `frozen_w`, FR3 as
                    registered as `frozen_w_mae`. FR1 is printed for the record
                    and is NOT a stop here: the transfer's own pre-registered
                    rules (R1-R5) are reporting rules, and a negative transfer
                    result is a result.
  --stage full      the `dist` target (sizes AND targets of held-out test
                    molecules), one cell per (prop, arm) at its frozen strength,
                    new seeds, --per-mol sidecars. Cells are named `*__full.json`
                    under <out>/seed<seed>/ so full_run_table.py and
                    dist_report.py (--backend tfg) read them unchanged.

FAIR TUNING ON EquiFM (23 Sep, before any EquiFM cell; plan section 11):
  * all seven arms, tfg included (ported to EquiFM's two clocks);
  * the screen grid is the main sweep's 7 strengths plus 0.1 and 0.15, for
    EVERY arm (TRANSFER_STRENGTHS);
  * --stage extend: an arm whose FR3a pick on that grid is at an edge gets two
    more q90 strengths beyond it (x4, x16 or /4, /16); freeze refuses until
    they exist. Same rule for every arm;
  * strengths are chosen on the DECODED molecule's MAE (SELECT_METRIC).

THE EQUAL-CHEMISTRY-COST COMPARISON (the "PROPOSED" section of
SCOPE_FM_GUIDANCE_STATUS.md, on EquiFM):
  --stage eqtune     every arm on `dist` over EQ_STRENGTHS (0.05 ... 16) on the
                     fresh block test[10000:12000], n = 2000, seed 20261001
  --stage eqextend   the same edge rule on the tune grid (an edge pick gets x4,
                     x16 or /4, /16); eqfreeze waits for those cells
  --stage eqfreeze   FR3a on that block (decoded MAE, floor 0.9 x unguided) ->
                     frozen json, plus the per-arm frontier (in_band against
                     mol_stability across w) and in_band AT the floor
  --stage eqconfirm  the frozen strengths on test[5000:10000], n = 5000, seeds
                     20261004-6, per-molecule sidecars (full_run_table reads it)
All three test blocks (full run 0:5000, confirm 5000:10000, tune 10000:12000)
are disjoint.

Resumable: one JSON per cell, written atomically, completed cells are skipped;
a cell that raises leaves a `.failed` note and is retried by the next run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from evaluation import choose_delta, evaluate_samples        # noqa: E402
from checkpoint_paths import default_generator, require_predictor  # noqa: E402
from m1_signed_bias import PhysicalProperty  # noqa: E402
from external.tfg_assets import (Calibrated, EDMGenerator, PROP_INDEX,  # noqa: E402
                                 PROP_UNITS, QM9_MAD, TFGGuide, TFGOracle,
                                 fit_calibration, metadata)
from sampling import FlowSampler, VPSampler, initial_noise, integrate  # noqa: E402
from external.equifm_backend import (EQUIFM_ARGS, EQUIFM_WEIGHTS,  # noqa: E402
                                     EquiFMGenerator, EquiFMSampler,
                                     OCFlowOracle)

sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
# The grid, targets and frozen-file reader are IMPORTED, not copied, so the
# transfer cannot drift from the main sweep's protocol.
from guidance_sweep import (DEFAULT_W, DIST_TARGET, FROZEN_SETS,  # noqa: E402
                            SHG_SCHEDULES, STRENGTHS, TARGETS, load_frozen,
                            scale_schedule, tfg_config)
# `load_fm` builds OUR flow-matching generator -- the `fm` backend below runs
# it through this file's EXTERNAL property pair, which is the only way our
# base model and EquiFM can be compared without the pair changing underneath.
from m1_signed_bias import load_fm                             # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")
TFG_ROOT = os.path.join(ROOT, "audit", "fa_fb_search", "TFG")
OUT = os.path.join(ROOT, "results", "transfer")
BACKEND = "TFG/EDMsecond"
# --backend: which borrowed generator. The property pair is TFG's for both.
# EDMsecond (diffusion) was the first transfer target and was dropped on
# 23 Sep when FM was chosen as the project's base model; EquiFM is the
# borrowed FM model (docs/protocol/EQUIFM_USABILITY_AUDIT.md,
# proj1/src/external/equifm_backend.py).
# `fm` is OUR OWN generator driven through this file's external pair. It is not
# a "transfer" -- nothing is borrowed but the guide and the oracle -- and it
# exists for one reason: the base-model comparison (BASECMP below) has to hold
# f_A, f_B, delta, targets, sampler, window and clip identical across the two
# generators, and `guidance_sweep.py` cannot do that because it hardcodes our
# own predictors. Running our base HERE is what makes the two bases comparable.
BACKENDS = {"equifm": "EquiFM", "edm": "TFG/EDMsecond", "fm": "FM (ours)"}
OUT_ROOTS = {"equifm": os.path.join(ROOT, "results", "transfer_equifm"),
             "edm": OUT,
             "fm": os.path.join(ROOT, "results", "transfer_fm")}
# OUR GENERATOR, resolved rather than hardcoded.
#
# It used to be the literal proj1/checkpoints/fm_last.pt, which is the file on
# Henry's Betty tree and is NOT in the repository: it is 57.5 MB of weights
# plus optimiser and scheduler state, and .gitignore excludes it twice. A
# teammate cloning this repo got "needs our generator at
# proj1/checkpoints/fm_last.pt" and could not run `--backend fm` at all.
#
# weights/fm_ema.pt IS in the repository and carries the SAME EMA tensors --
# weights/README.md records `max |w_slim - w_full| = 0.0` across every
# parameter, and the slim file stores `source_md5 = <fm_last.pt's md5>` to say
# so. checkpoint_paths.default_generator() prefers the full file where it
# exists (Betty, then a local checkpoints dir) and falls back to the slim one,
# which is exactly the right order: identical weights either way, and the
# cluster keeps using the file it already has.
FM_CKPT = default_generator()

# --------------------------------------------------------------------------
# BASECMP: the base-model comparison (Henry, 25 Sep)
# --------------------------------------------------------------------------
# One question: for each guidance method, does its behaviour survive a change
# of BASE MODEL, when nothing else changes? So both generators are run with
#   * TFG's tf_predict_<p> as f_A and TFG's evaluate_<p> as f_B (the best
#     available pair, and external to us -- see BASECMP_PROTOCOL.md section 2);
#   * the local delta (delta_local below), which depends on f_B and QM9 only
#     and is therefore IDENTICAL for both bases by construction;
#   * v2's fixed q90 target per property, v2's chemistry floor, v2's metrics.
#
# Two axes are screened jointly, because the project has measured the window to
# be its largest single effect (alpha MAE 10.02 -> 5.32 as t_min_guide goes
# 0.05 -> 0.5) and has never measured any window between 0.05 and 0.5:
#   strength w     log-spaced, and it REACHES PAST w = 4 -- the v1 grid stopped
#                  at 4 and capped lgd_mc, which cleared the floor there; that
#                  failure is designed out rather than discovered again
#   t_start        the flow-time instant guidance switches ON. Guidance is
#                  active on t in [t_start, 1). For a VP/EquiFM backend the
#                  same instant is tau_max_guide = 1 - t_start, so ONE number
#                  describes the window on both families.
BASECMP_ROOT = os.path.join(ROOT, "results", "basecmp")
BASECMP_STRENGTHS = [0.05, 0.25, 1.0, 4.0, 16.0]
BASECMP_T_STARTS = [0.05, 0.25, 0.5, 0.75]
BASECMP_TARGET = "q90"
BASECMP_SEED = 20260925
BASECMP_N = 1000
BASECMP_FULL_SEEDS = (20261001, 20261002, 20261003)
BASECMP_FLOOR = 0.9
# v2's rule V2 selects on IN-BAND, not on MAE: "select on the metric that
# decides the verdict". transfer_sweep's older FR3a/SELECT_METRIC selects on
# decoded MAE, which is v1's rule; the basecmp stages deliberately do not use
# it. Decoded, because soft type channels can move without moving the molecule.
BASECMP_SELECT = "in_band_fraction_dec"
BASECMP_SETS = ("floor", "free")     # floor = chemistry-constrained, free = not

# --------------------------------------------------------------------------
# the arms under test
# --------------------------------------------------------------------------
# The main sweep's full-run set (full_run.slurm FULL_ARMS), so the two
# backends run the same comparison:
#   plug      DPS-style plug-in (Chung et al. 2023) -- also btvg's mean-only rung
#   tmpd      TMPD / PiGDM uncertainty denominator (Boys et al. 2024)
#   lgd_mc    loss-guided diffusion (Song et al. 2023)
#   tfg       TFG, the full update (Ye et al. 2024) -- replaced dflow in the
#             main compare set on 23 Sep; added here BEFORE any transfer cell
#             ran. On this backend it is TFG's own method on TFG's own
#             generator, guide and oracle, with TFG's own energy (QM9_MAD).
#   btvg      band-targeted variance guidance -- ours
#   btvg_var  btvg's variance-only rung, so the 2x2 ablation transfers too
# `dflow` is NOT here: it was dropped from the main full run (23 Sep), and its
# rollout (dflow._euler_rollout) integrates a flow VELOCITY on t in [0, 1] --
# EDM's network returns epsilon, so it would need a VP port before it could
# run here at all.
BASELINE = ["unguided"]
COMPARE_ARMS = ["plug", "tmpd", "lgd_mc", "tfg"]
OUR_ARMS = ["btvg", "btvg_var"]
TRANSFER_SET = BASELINE + COMPARE_ARMS + OUR_ARMS

# --------------------------------------------------------------------------
# PROTOCOL v3 (26 Sep 2026): docs/protocol/FULL_RUN_V3_PROTOCOL.md
# --------------------------------------------------------------------------
# Henry's decisions, folded in verbatim:
#   target      q50 for EVERY property -- not v2's q90
#   strength    w = 1 for EVERY arm. NOT each arm's best, and NOT equal force
#   seeds       three, the same three v2 used, so noise is comparable
#   n           2000 per cell in batches of 500 -- 4 controllers per cell
#   floor       NONE. There is no chemistry gate. Stability, validity and
#               uniqueness are REPORTED beside in-band, never used to exclude
#   arms        the full comparison set, btvg dropped, BDG added as the
#               innovation target
#   BDG         eta = 4, tau_mult in {0.5, 1.0}; guidance start stays 0.5.
#               tau_mult is the SETPOINT knob, not the guidance window
#   bases       both -- ours (`fm`) and EquiFM, identical code, one flag
#
# WHY NO FLOOR CHANGES HOW THIS IS READ. At q50, w = 1 is far above some arms'
# floor-clearing strength: tfg froze at 0.01-0.05 under v2, so w = 1 is 20-100x
# it. Measured on our base at q50, n = 512, tfg at w = 1 reaches the HIGHEST
# in-band of any arm on all three properties (mu 0.3828 against plug's 0.0840)
# at molecule stability 0.2285 and validity 0.62, against unguided's 0.4023 and
# 0.75. So an in-band ranking on its own would name tfg the winner for wrecking
# chemistry. That is exactly what v2's floor existed to prevent, and it is why
# `v3_table.py` refuses to rank on in-band alone and prints the chemistry block
# in the same row. The run is a Pareto picture, not a leaderboard.
#
# BDG'S KNOB IS NOT delta. tau = tau_mult * f_A.y_std, so delta never enters
# BDG's sampling -- only its scoring. v3 therefore scores with the
# PRE-REGISTERED delta (2 x f_B's val MAE, target-independent). The 25-Sep
# local delta was measured in q90's neighbourhood and does not apply at q50.
V3_ROOT = os.path.join(ROOT, "results", "v3")
V3_TARGET = "q50"
V3_W = 1.0
V3_T_START = 0.5                     # guidance on for t in [0.5, 1)
V3_SEEDS = (20261001, 20261002, 20261003)
# THREE SEEDS FOR BOTH STAGES (Henry, 26 Sep, final). The ablation ran one seed
# for two days; it does not any more. Every v3 row, headline or ablation,
# carries an across-seed spread.
#
# N IS DELIBERATELY UNSET. Henry left it to the operator: bobo picks the cell
# size on the cluster he runs on, because it is a memory-and-budget decision
# and nothing about the protocol depends on its value. `None` here is not a
# placeholder to fill in later -- it is the pre-registration saying "this run
# does not fix n", and --n is REQUIRED on the command line for both v3 stages.
#
# What n DOES decide is power, and that is the one thing the operator has to
# read before choosing. FULL_RUN_V3_PROTOCOL.md section 6.1 gives the binomial
# se as a function of n, with a table, so a value can be picked against the
# effect size it needs to resolve rather than against the budget alone. The
# short version: pooled se on in_band at p ~ 0.09 is sqrt(.09*.91/(3n)), so
# resolving a ~1pp contrast under a multiplicity-corrected threshold needs
# n in the low thousands per cell, not hundreds.
#
# It must be divisible by the batch. The batch is BDG's ESTIMATOR, not a speed
# knob (see section 2.4), so a remainder batch is a second, far noisier
# controller pooled into the cell's diagnostics as an equal. Both the slurm and
# guidance_sweep refuse an indivisible pair.
V3_N = None                          # required at the command line; see above
V3_BDG_ETA = 4.0
# tau_mult {0.5, 1.0} -- 0.75 dropped (Henry, 26 Sep). This is BDG's SETPOINT
# knob, tau = tau_mult * f_A.y_std; it is NOT the t >= 0.5 guidance window,
# which is unchanged. Dropping the MIDDLE rung costs no w_eff span (measured
# 1019 at 0.5 and 253 at 1.0, a 4.03x span set by the endpoints) but it does
# reduce the headline ladder to two points, which cannot show monotonicity or
# curvature. The third point survives in the ablation, which keeps all four
# tau_mults -- read it there, never off the headline's two rungs.
V3_BDG_TAU_MULTS = (0.5, 1.0)
# The ablation's two axes.
#
# eta = 0 is in the grid on purpose: it is BDG's identity gate, bit-identical
# to plug, so it appears as a row rather than only as a unit test. But at
# eta = 0 the dispersion term is multiplied by zero, so tau_mult cannot change
# the samples: all four tau_mults would be ONE computation under four names,
# and averaging them would look like four independent measurements of the same
# thing. The planner emits eta = 0 exactly once, at the reference tau_mult.
V3_ABL_ETAS = (0.0, 1.0, 2.0, 4.0, 8.0)
V3_ABL_TAU_MULTS = (0.5, 0.75, 1.0, 1.5)
V3_ABL_ETA0_REF_TAU = 1.0
# The ablation now matches the headline on n and on seeds. It is the same
# protocol with a different arm set, so an ablation row and a headline row at
# the same n ARE comparable -- which is the point of the change, and the
# opposite of what the 26 Sep morning draft said.
V3_ABL_N = None                      # follows --n, like the headline
V3_ABL_SEEDS = V3_SEEDS
# THE TWO STAGES ARE SEPARATED BY THE TREE, NOT BY n. They write
#   results/v3/<backend>/<stage>/n<N>/seed<S>/
# with <stage> in {v3, v3abl}. n used to do this job, which is why the ablation
# briefly ran at a different size; at equal n and equal seeds that separation
# is gone, and without the stage directory v3_table.py would glob the
# ablation's 15 extra arms into the headline's table. The stage level is what
# lets both stages run at whatever n bobo picks.
V3_STAGE_DIRS = {"v3": "v3", "v3abl": "v3abl"}
V3_COMPARE_ARMS = ["unguided", "plug", "tmpd", "lgd_mc", "tfg"]
# --------------------------------------------------------------------------
# WHICH BACKEND GETS WHICH PROPERTY PAIR, AND WHICH ARMS (Henry, 26 Sep)
# --------------------------------------------------------------------------
# One declaration instead of scattered `if args.backend ==` branches. Adding a
# base model should be a row here plus a loader, not an archaeology exercise.
#
#   pair   "ours" -> weights/f_A_<p>.pt + f_B_<p>.pt  (build_pair_ours)
#          "tfg"  -> TFG's tf_predict_<p> + evaluate_<p>  (build_pair)
#   arms   None   -> the stage's full arm set
#          tuple  -> only these, whatever the stage plans
#
# READ THIS BEFORE COMPARING in_band ACROSS BACKENDS. delta = k x MAE(f_B), so
# the pair sets the BAND WIDTH. Our f_B is less accurate than TFG's on all
# three properties -- measured in BASECMP_PROTOCOL.md, roughly 1.2x on mu, 3x
# on alpha, 2x on gap -- so an ours-pair backend is scored in a WIDER band and
# its in_band is higher for that reason alone, before any base model or any
# arm is considered.
#
# That is a deliberate trade, not an oversight. Our pair is disjoint by
# CONSTRUCTION (f_A and f_B trained on disjoint halves), where TFG's is
# disjoint only by inference and its evaluate_<p> saw an unknown part of QM9,
# which makes its delta optimistically tight. The cost is that
#
#     ACROSS BACKENDS WITH DIFFERENT PAIRS, in_band IS NOT COMPARABLE.
#
# Within one backend every arm shares one delta, so the arm comparison -- which
# is what this project is actually about -- is unaffected. v3_table prints the
# pair and delta on every row and refuses to mix pairs inside one table.
V3_BACKENDS = {
    "fm":     {"pair": "ours", "arms": None,
               "label": "FM (ours)"},
    "equifm": {"pair": "tfg",  "arms": None,
               "label": "EquiFM"},
    # QM9 diffusion. TFG's released EDMsecond, driven through our sampler --
    # the diffusion base model the comparison needs. UNGUIDED AND PLUG ONLY
    # (Henry, 26 Sep): it is here as a base-model reference point, not as a
    # seventh arm-by-arm column, and the full set on a third base would cost
    # more than the question is worth.
    "edm":    {"pair": "ours", "arms": ("unguided", "plug"),
               "label": "QM9 diffusion (TFG/EDMsecond)"},
}
# Modality 2 is NOT here, and cannot simply be added. Its state is a [B, L, 4]
# simplex rather than coords+feats+mask over an EGNN, its properties are
# analytic (gc/cpg) rather than learned nets, and it has its own driver in
# proj1/m2/. See docs/protocol/MODALITY2_V3_PLAN.md for what bridging it
# requires; until that is done it runs through proj1/m2/run_sweep.py.


def backend_pair(backend):
    """Which property pair this backend scores with. Default: TFG's."""
    return V3_BACKENDS.get(backend, {}).get("pair", "tfg")


def v3_backend_arms(backend, planned):
    """`planned` restricted to what this backend runs, order preserved.

    Named v3_* because `backend_arms` already exists further down for the
    TRANSFER stage's portability filter -- a different question entirely, and
    defining a second function under that name silently shadowed the first.
    """
    allow = V3_BACKENDS.get(backend, {}).get("arms")
    if allow is None:
        return list(planned)
    keep = [a for a in planned if a in allow]
    missing = [a for a in allow if a not in planned]
    # NONE of them planned -> this backend does not take part in this stage.
    # That is the QM9 diffusion base against the BDG ablation: it runs
    # unguided+plug, and the ablation plans neither. Return empty and let the
    # caller exit 0.
    #
    # SOME but not all -> the registry names an arm the stage does not have,
    # which is a typo or a stale declaration, and silently narrowing it away
    # would run a smaller experiment than the one declared. Refuse.
    if missing and keep:
        raise SystemExit(
            "--backend %s is declared to run %s, but the stage did not plan "
            "%s. The registry and the planner disagree."
            % (backend, ", ".join(allow), ", ".join(missing)))
    return keep



def bdg_arm(eta, tau_mult):
    """The arm name that carries BDG's two settings.

    They ride in the NAME rather than in the cell tuple because every planner,
    cell_name and resume check in this file is built on 5-tuples
    (prop, arm, target, w, t_start), and the previous session verified that
    every pre-existing cell name is byte-identical under that shape. Widening
    the tuple to carry a variant would have re-derived all of those names and
    orphaned 216 finished EquiFM cells. `%g` keeps 4.0 -> "4" and 0.75 ->
    "0.75", so the names match guidance_sweep's own e<eta>t<mult> convention.
    """
    return "bdg_e%gt%g" % (float(eta), float(tau_mult))


def parse_bdg_arm(arm):
    """(eta, tau_mult) for a bdg arm name, or None for any other arm."""
    if not arm.startswith("bdg_e"):
        return None
    body = arm[len("bdg_e"):]
    if "t" not in body:
        raise SystemExit("bdg arm %r must look like bdg_e<eta>t<tau_mult>" % arm)
    a, b = body.split("t", 1)
    try:
        return float(a), float(b)
    except ValueError:
        raise SystemExit("bdg arm %r has a non-numeric eta or tau_mult" % arm)


def base_mode(arm):
    """The sampler `mode=` string for an arm name.

    Only bdg carries a suffix; every other arm's name IS its mode. Without this
    the sampler would be handed mode="bdg_e4t0.5", which is not in KNOWN_MODES
    and would raise -- after loading the generator and the dataset.
    """
    return "bdg" if arm.startswith("bdg_e") else arm


def v3_arms(etas=None, tau_mults=None, compare=True):
    """v3's arm list: the comparison set, then one arm per (eta, tau_mult).

    eta = 0 collapses to a single arm whatever tau_mults is, because the
    dispersion term is multiplied by zero and tau_mult then changes nothing
    about the samples (see V3_ABL_ETA0_REF_TAU).
    """
    etas = V3_BDG_ETA if etas is None else etas
    etas = (etas,) if isinstance(etas, (int, float)) else tuple(etas)
    tms = V3_BDG_TAU_MULTS if tau_mults is None else tuple(tau_mults)
    out = list(V3_COMPARE_ARMS) if compare else []
    for e in etas:
        ts = (V3_ABL_ETA0_REF_TAU,) if float(e) == 0.0 else tms
        for tm in ts:
            a = bdg_arm(e, tm)
            if a not in out:
                out.append(a)
    return out


def plan_v3_cells(props, arms, target=None, w=None, t_start=None):
    """One cell per (property, arm). No strength sweep, no window sweep.

    v3 fixes w and t_start, so there is exactly one cell per (property, arm)
    per seed -- unlike the basecmp stages, which screen a grid and then freeze.
    `unguided` is planned first, because it is every comparison's reference and
    a job cut short must not be the one that lacks it.
    """
    tgt = V3_TARGET if target is None else target
    ww = V3_W if w is None else float(w)
    ts = V3_T_START if t_start is None else float(t_start)
    seen, cells = set(), []

    def add(c):
        if c not in seen:
            seen.add(c)
            cells.append(c)
    for prop in props:
        if "unguided" in arms:
            add((prop, "unguided", tgt, ww, ts))
    for prop in props:
        for arm in arms:
            if arm == "unguided":
                continue
            add((prop, arm, tgt, ww, ts))
    return cells
# Arms a backend cannot run. Empty for both: `tfg`, a sampler-level arm, was
# ported to EquiFM's two clocks on 23 Sep (equifm_backend.EquiFMSampler).
# `fm` runs every arm by construction: these arms were written against
# FlowSampler, which is our generator's own sampler.
NOT_PORTED = {"equifm": (), "edm": (), "fm": ()}


def backend_arms(backend):
    return [a for a in TRANSFER_SET if a not in NOT_PORTED[backend]]
# The 22-Sep plan's other arms stay runnable through --arms, as a labelled
# post-hoc analysis (R5). `tfg_mc` is the one R3 (the home-turf test) needs.
LEGACY_COMPARE = ["tfg_mc", "osc"]
LEGACY_OURS = ["smg", "smg2", "shg_plug_btvg"]
ARM_CLASS = {**{a: "BASELINE" for a in BASELINE},
             **{a: "COMPARE" for a in COMPARE_ARMS + LEGACY_COMPARE},
             **{a: "OURS" for a in OUR_ARMS + LEGACY_OURS}}

SCREEN_TARGETS = ("q50", "q90")
FREEZE_TARGET = "q90"               # FR3-corrected: dist freezes at q90
FLOOR = 0.9                         # the saved rubric: x unguided mol_stability
FR1_SIGMA, FR1_BEATEN_ON = 3.0, 2

COMPARE_SEED = 20260922             # the transfer's pre-registered screen seed
FULL_SEEDS = (20261001, 20261002, 20261003)
CALIB_SEED, N_CALIB = 4242, 3000

# ---- fair tuning (see the module docstring) ------------------------------
# The fill-in is where TFG's floor-clearing strengths sat on our own model
# (0.05-0.25). Giving it to one arm only would bias the best-of-grid choice
# toward that arm, so every arm gets it.
FILL_IN = (0.1, 0.15)
TRANSFER_STRENGTHS = sorted({float(w) for w in STRENGTHS} | set(FILL_IN))
# The main run's lgd_mc best sat at w = 4 and still cleared the floor at
# w = 8: the grid, not the method, set its strength. One round, same factors
# for every arm.
EXT_FACTORS = (4.0, 16.0)
# Soft type channels can move without moving the molecule; choosing strengths
# on them would reward an arm for that (measured for tfg on our own model).
SELECT_METRIC, SELECT_RMSE = "prop_mae_eval_dec", "prop_rmse_eval_dec"

# ---- the equal-chemistry-cost comparison ---------------------------------
EQ_STRENGTHS = [0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
EQ_TUNE_START, EQ_TUNE_N, EQ_TUNE_SEED = 10000, 2000, 20261001
EQ_CONFIRM_START, EQ_CONFIRM_N = 5000, 5000
EQ_CONFIRM_SEEDS = (20261004, 20261005, 20261006)


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def cell_name(prop, arm, tgt, w, cfg="", stage="compare"):
    """One filename per cell, carrying the configuration that produced it.

    Resumability skips any cell whose file exists, so a name that omits part of
    the configuration silently merges incompatible runs: an `--n 64` smoke test
    followed by an `--n 512` sweep would leave the small cells in place and the
    result set would mix sample sizes without saying so. `cfg` is built once in
    `main` from every argument that changes a cell's meaning. Full-stage cells
    end in `__full.json`, which is what full_run_table.py globs for.
    """
    return "tr__%s__%s__%s__w%g%s%s.json" % (
        prop, arm, tgt, w, cfg,
        "__full" if stage in ("full", "basecmpfull") else "")


def config_tag(args, t_start=None):
    """The run-configuration suffix baked into every cell name.

    The batch size is in it because it changes the SAMPLES, not just the
    speed: `initial_noise` draws one batch at a time from one generator, so
    batch 64 and batch 128 give every molecule different starting noise. Two
    arms are paired (same noise, same targets) only at the same batch.

    `t_start` makes the WINDOW a per-cell property, which the basecmp stages
    need: there the window is a screened axis, and at the full-run stage each
    arm carries its own frozen window, so one job writes cells at several. It
    is written into the name through the existing `win` slot as its VP image,
    1 - t_start, so a cell named by the old global path and a cell named by the
    new per-cell path at the same instant get BYTE-IDENTICAL names. Passing
    None reproduces the old behaviour exactly.
    """
    win = args.tau_max_guide if t_start is None else 1.0 - float(t_start)
    return "__n%d_s%d_%s_%s_win%g_b%d_seed%d%s" % (
        args.n, args.steps, args.solver, args.grid, win,
        args.batch, args.seed,
        ("_blk%d" % args.block_start) if args.block_start else "")


def flow_to_vp_schedule(sched):
    """Mirror an SHG schedule from flow time into VP time.

    THE TWO FAMILIES RUN TIME IN OPPOSITE DIRECTIONS. Flow time goes 0 (noise)
    -> 1 (data); VP time goes 1 (noise) -> 0 (data). `SHG_SCHEDULES` is written
    in flow time, and `_Base.active()` looks the schedule up with whatever
    scalar the sampler hands it -- so handing a flow-time schedule to a
    VPSampler runs every phase in the WRONG HALF of the trajectory: a band
    meant for "the last 20%, where the molecule is nearly formed" would fire in
    pure noise instead. Nothing would raise; the arm would simply be a
    different, worse method wearing its name.

    So each interval [lo, hi) becomes [1 - hi, 1 - lo), which maps the same
    physical stage of generation onto VP's clock. Order does not matter --
    `active()` scans for the first interval containing t and the images are
    still disjoint -- but they are re-sorted so a printed schedule reads in
    integration order.
    """
    return sorted(((1.0 - hi, 1.0 - lo, mode, mult)
                   for (lo, hi, mode, mult) in sched),
                  key=lambda iv: iv[0])


def arm_kwargs(arm, delta):
    """Per-arm extra state, mirroring guidance_sweep.arm_kwargs.

    BTVG's tau is the target sd that makes the band a 95% interval, not delta
    itself: a centred Gaussian with sd = delta covers only 68.3% of the band.
    """
    kw = {}
    bdg = parse_bdg_arm(arm)
    if bdg is not None:
        # BDG's setpoint is tau = tau_mult * f_A.y_std, NOT delta: it is a
        # fraction of the guide's own property scale, so it is resolved in
        # run_cell once `s` is known. Stashed under a private key that
        # guidance_field never sees.
        eta, tau_mult = bdg
        kw["bdg_eta"] = eta
        kw["_bdg_tau_mult"] = tau_mult
        return kw
    if arm in ("btvg", "btvg_var", "btvg_mean") or arm in SHG_SCHEDULES:
        # An SHG schedule containing a btvg phase needs tau for the same
        # reason a standalone btvg arm does.
        kw["tau"] = delta / 1.96
    if arm == "band":
        kw["band_tau"] = delta
    if arm in SHG_SCHEDULES:
        kw["schedule"] = flow_to_vp_schedule(SHG_SCHEDULES[arm])
    return kw


def strength_scale(arm, kw, s):
    """Put every arm's strength knob on one scale, as the main sweep does.

    BTVG's coefficient carries 1/tau^2 where the likelihood arms carry 1/s^2,
    so w = 1 means something different for it unless the ratio is divided out.
    Without this, a BTVG cell at w = 1 ran thousands of times stronger than a
    plug cell at w = 1 and was clip-saturated at every strength -- which is
    finding S1 in SCOPE_FM_GUIDANCE_STATUS.md section 10b.
    """
    if arm.startswith("btvg") and kw.get("tau"):
        return (kw["tau"] / s) ** 2
    return 1.0


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def plan_compare_cells(props, arms, targets=SCREEN_TARGETS):
    """q50 then q90; within each, the whole arm set at DEFAULT_W first, so a
    partial run is still an arm comparison. unguided has no strength axis.
    `targets` only splits the work across jobs; freeze needs both."""
    seen, cells = set(), []

    def add(c):
        if c not in seen:
            seen.add(c)
            cells.append(c)
    for tgt in targets:
        for prop in props:
            for arm in arms:
                add((prop, arm, tgt, float(DEFAULT_W)))
        for prop in props:
            for arm in arms:
                if arm != "unguided":
                    for w in TRANSFER_STRENGTHS:
                        add((prop, arm, tgt, float(w)))
    return cells


def plan_eqtune_cells(props, arms, strengths=None):
    """Every arm on `dist` over one grid (unguided once), on the tune block."""
    ws = EQ_STRENGTHS if strengths is None else strengths
    cells = []
    for prop in props:
        for arm in arms:
            if arm == "unguided":
                cells.append((prop, arm, DIST_TARGET, float(DEFAULT_W)))
            else:
                cells += [(prop, arm, DIST_TARGET, float(w)) for w in ws]
    return cells


def plan_full_cells(props, arms, frozen, sets=("primary",), exclude=()):
    """Each arm once per property per strength set, at its frozen strength,
    minus cells that also belong to an `exclude` set (so the secondary-set
    tasks never write a cell a primary task also writes)."""
    def cells_of(names):
        out = []
        for s in names:
            w = frozen[FROZEN_SETS[s]]
            out += [(prop, arm, DIST_TARGET, float(w[arm][prop]))
                    for prop in props for arm in arms]
        return out
    drop = set(cells_of(exclude))
    seen, cells = set(), []
    for c in cells_of(sets):
        if c not in seen and c not in drop:
            seen.add(c)
            cells.append(c)
    return cells


def with_t(cells, t_start):
    """Give every 4-tuple cell the same window, so one code path downstream.

    Every stage's plan becomes (prop, arm, tgt, w, t_start). For the pre-basecmp
    stages t_start is the global window's flow-time image, so their cell names
    and their behaviour are unchanged."""
    return [(p, a, tg, w, float(t_start)) for (p, a, tg, w) in cells]


def plan_basecmp_cells(props, arms, t_starts, strengths=None, target=None):
    """The joint (strength x start-time) screen, at v2's fixed target.

    UNGUIDED IS PLANNED ONCE PER PROPERTY, not once per window: with no guidance
    the window is not read by anything, so a second unguided cell at another
    window would be the same computation under another name and would then be
    averaged into the chemistry floor as if it were independent evidence. It is
    planned at the reference window (0.5), and `basecmp_freeze.py` uses that one
    cell as the floor for every window -- which is correct precisely because the
    floor is a property of the generator, not of the guidance.

    Guided arms are ordered so a partial run is still a complete comparison at
    the reference window: every (arm, strength) at t_start = 0.5 first, then the
    other windows."""
    ws = BASECMP_STRENGTHS if strengths is None else list(strengths)
    tgt = BASECMP_TARGET if target is None else target
    ts = sorted({float(t) for t in t_starts})
    # The reference window is FIXED at 0.5, not "the middle of whatever was
    # asked for". The screen is split across jobs by window, so a job given
    # --t-starts 0.05 would otherwise plan its own unguided cell at 0.05, and
    # the freeze would then find several unguided cells that disagree about the
    # chemistry floor -- or, worse, agree by luck and set the floor from the
    # wrong one. One unguided cell exists, at 0.5, and exactly one job writes it.
    ref = 0.5
    if "unguided" in arms and ref not in ts:
        raise SystemExit(
            "unguided is planned once, at t_start = %g, but --t-starts is %s. "
            "Drop unguided from --arms for this job, or include %g."
            % (ref, ",".join("%g" % t for t in ts), ref))
    seen, cells = set(), []

    def add(c):
        if c not in seen:
            seen.add(c)
            cells.append(c)
    for prop in props:
        if "unguided" in arms:
            add((prop, "unguided", tgt, float(DEFAULT_W), float(ref)))
    # The reference window runs FIRST when it was asked for, so a job that is cut
    # short still leaves a complete comparison at one window. It is NOT added
    # when it was not asked for: the screen is split across jobs by window, and
    # prepending it unconditionally would make every job also run the reference
    # window's cells -- 180 duplicated cells across the array, each computed
    # several times, and the same filename written by several jobs at once.
    order = ([ref] if ref in ts else []) + [t for t in ts if t != ref]
    for t in order:
        for prop in props:
            for arm in arms:
                if arm == "unguided":
                    continue
                for w in ws:
                    add((prop, arm, tgt, float(w), float(t)))
    return cells


def plan_basecmp_full_cells(props, arms, frozen, sets=BASECMP_SETS):
    """One cell per (property, arm, set) at that arm's frozen (w, t_start).

    The two sets -- `floor` (best in-band among windows/strengths clearing the
    chemistry floor) and `free` (best in-band, floor ignored) -- routinely pick
    the SAME (w, t) for a weak arm that never threatens the floor. Those cells
    are de-duplicated, so the run does not pay twice for one number, and the
    table reports the same cell under both sets. `unguided` has no strength or
    window, so it is one cell per property whatever the sets are."""
    fz = frozen["frozen"]
    seen, cells = set(), []
    for s in sets:
        if s not in fz:
            raise SystemExit("frozen file has no set %r (has %s)"
                             % (s, ", ".join(sorted(fz))))
        for prop in props:
            for arm in arms:
                if arm == "unguided":
                    c = (prop, arm, BASECMP_TARGET, float(DEFAULT_W), 0.5)
                else:
                    pick = fz[s].get(arm, {}).get(prop)
                    if pick is None:
                        raise SystemExit("frozen file has no %s/%s in set %r"
                                         % (arm, prop, s))
                    c = (prop, arm, BASECMP_TARGET, float(pick["w"]),
                         float(pick["t_start"]))
                if c not in seen:
                    seen.add(c)
                    cells.append(c)
    return cells


def load_basecmp_frozen(path, props, arms, backend):
    """Read basecmp_freeze.py's json and refuse one that does not belong here."""
    fz = json.load(open(path))
    if fz.get("study") != "basecmp":
        raise SystemExit("%s is not a basecmp frozen file (study=%r)"
                         % (path, fz.get("study")))
    if fz.get("backend") != BACKENDS[backend]:
        raise SystemExit("%s was frozen on %r; this run is --backend %s (%r). "
                         "A window and a strength chosen on one generator do "
                         "not transfer to another -- that is the whole point "
                         "of the comparison." % (path, fz.get("backend"),
                                                 backend, BACKENDS[backend]))
    if fz.get("select_metric") != BASECMP_SELECT:
        raise SystemExit("%s selected on %r, this run expects %r"
                         % (path, fz.get("select_metric"), BASECMP_SELECT))
    missing = [(s, a, p) for s in BASECMP_SETS for a in arms if a != "unguided"
               for p in props if fz.get("frozen", {}).get(s, {}).get(a, {}).get(p) is None]
    if missing:
        raise SystemExit("%s is missing %d (set, arm, prop) picks, e.g. %s"
                         % (path, len(missing),
                            ", ".join("%s/%s/%s" % m for m in missing[:6])))
    return fz


# --------------------------------------------------------------------------
# the backend: TFG's generator, TFG's guide, TFG's oracle
# --------------------------------------------------------------------------

def load_backend(edm_dir, dev):
    gen_w = os.path.join(edm_dir, "generative_model_ema.npy")
    gen_a = os.path.join(edm_dir, "args.pickle")
    for p in (gen_w, gen_a):
        if not os.path.exists(p):
            raise SystemExit(
                "missing %s.\nRun:  python proj1/scripts/fetch_tfg_assets.py\n"
                "The EDMsecond checkpoint is not in the repository; it is "
                "TFG's release and must be downloaded." % p)
    net = EDMGenerator(gen_w, gen_a, n_types=5, device=dev)
    return net, {"edm_md5": file_md5(gen_w), "edm_args": gen_a}


def calibration_indices(d, n_calib=N_CALIB, seed=CALIB_SEED):
    """Molecules the two-parameter calibration is fitted on: train_a + train_b.

    NOT val (it supplies the q50/q90 molecule sizes) and NOT test (it supplies
    the `dist` sizes AND targets). An earlier version drew from all of QM9, so
    ~10% of the fit set were test molecules whose labels are then used as
    targets -- a 2-parameter fit, so negligible in size, but it broke the
    "test is touched by nothing" claim the main sweep makes and this one
    should too.
    """
    pool = torch.cat([d["split"]["train_a"], d["split"]["train_b"]])
    g = torch.Generator().manual_seed(seed)
    return pool[torch.randperm(pool.numel(), generator=g)[:n_calib]]


def generator_feat_scale(edm_dir):
    """The generator's type-channel divisor: its sampler state holds
    onehot / this. Read from the checkpoint's own args, never assumed."""
    return float(metadata(os.path.join(edm_dir, "args.pickle"))
                 ["normalize_factors"][1])


def local_delta(prop, d, raw_oracle, slope, intercept, dev, k=2.0,
                q=0.90, h=0.05, batch=128):
    """(delta, report): k x the oracle's MAE NEAR THE TARGET, on real molecules.

    The rule settled on 24 Sep (`proj1/scripts/local_fb_mae.py`, status doc
    section 6 "How delta is set"), applied here to the EXTERNAL oracle instead
    of ours. The recipe transfers; the numbers do not, so they are recomputed.

    delta = k x mean |f_B(x) - y| over `val` molecules whose TRUE property y
    lies in [Q_train_a(q - h), Q_train_a(q + h)] -- the q90 target's own
    neighbourhood, +-5 percentile points. Selection is by the TRUE property and
    not by f_B's prediction: regression to the mean makes the two differ in the
    tail, and selecting on the prediction would measure a different quantity.

    WHY NEAR THE TARGET AT ALL. delta is the width of the band in-band counts
    hits in, and every hit is near the target by definition. f_B's error
    averaged over all of QM9 is dominated by the bulk, where the run never
    scores anything.

    WHY THIS IS IDENTICAL FOR BOTH BASE MODELS, and why that matters more than
    its value. Nothing in this function touches a generator: it is f_B's error
    on real QM9 molecules. So the two bases are scored against the same bar. A
    delta recomputed per base -- e.g. from each generator's own samples -- would
    make the bar easier for one generator than the other and would destroy the
    comparison while looking more careful. `--delta-mode local` is asserted to
    agree across backends by test_basecmp.py.

    THE LIMIT THAT MATTERS MORE THAN THE WINDOW, carried from the original note
    and made worse here. (a) This is f_B's error on REAL molecules; in-band
    scores GENERATED ones, only ~35-40% of which are molecule-stable, and f_B's
    error on that population is unmeasured and probably larger. (b) For OUR f_B
    the split protocol guarantees `val` is not training data. For TFG's oracle
    it guarantees nothing: `evaluate_<p>` trained on an unknown ~50% of QM9, so
    some of these val molecules are very likely in its training set and this MAE
    is optimistic -- delta is therefore TIGHTER than it should be. A tighter
    band lowers in-band for every arm on both bases equally, so it does not
    favour any arm or any generator; it makes the absolute coverage numbers
    pessimistic and they must be read that way. This is caveat 2 of this file's
    header, unchanged and unfixable from the released artifacts.
    """
    pi = PROP_INDEX[prop]
    ya = d["y"][d["split"]["train_a"], pi].double()
    lo = float(torch.quantile(ya, q - h))
    hi = float(torch.quantile(ya, q + h))
    one_sided = bool(hi >= float(ya.max()) - 1e-12)
    va = d["split"]["val"]
    y = d["y"][va, pi].double()
    keep = ((y >= lo) & (y <= hi)).nonzero(as_tuple=True)[0]
    if keep.numel() < 200:
        raise SystemExit("%s: only %d val molecules in [%.4g, %.4g]; delta would "
                         "be noise" % (prop, keep.numel(), lo, hi))
    idx = va[keep]
    truth = y[keep].to(dev)
    preds = []
    with torch.no_grad():
        for i in range(0, idx.numel(), batch):
            b = idx[i:i + batch]
            preds.append(raw_oracle(d["coords"][b].to(dev).contiguous(),
                                    d["feats"][b].to(dev).contiguous(),
                                    d["mask"][b].to(dev).contiguous()).reshape(-1))
    pred = torch.cat(preds).double() * slope + intercept
    err = (pred - truth).abs()
    mae = float(err.mean())
    rep = {"rule": "delta = %g x MAE(f_B) over val molecules with true property "
                   "in [Q_train_a(%.2f), Q_train_a(%.2f)]" % (k, q - h, q + h),
           "k": k, "q": q, "h": h, "window": [lo, hi], "one_sided": one_sided,
           "n": int(idx.numel()), "local_mae": mae,
           "signed_err": float((pred - truth).mean()),
           # k = 2 is justified by measured coverage, not convention: report it
           "coverage_at_delta": float((err <= k * mae).double().mean()),
           "coverage_at_1x": float((err <= mae).double().mean()),
           "selected_by": "true property",
           "split": "val", "generator_independent": True}
    return k * mae, rep


# OUR OWN PROPERTY PAIR, as an alternative to TFG's borrowed one.
#
# Henry, 26 Sep: `fm` and the QM9-diffusion backend score with OUR f_A/f_B;
# `equifm` keeps TFG's. Which pair a backend uses is declared in V3_BACKENDS
# below, not decided here.
#
# WHY THIS IS NOT JUST "LOAD OUR NETS". Two things have to be right or the
# numbers are quietly wrong rather than loudly broken:
#
#   1. FEATURE SCALE. `PhysicalProperty` was trained on the data file's raw
#      one-hot and has no feat_scale of its own, so it is only correct when
#      the sampler also works in raw one-hot. That is true for `fm`
#      (norm_values (1,1,1)) and FALSE for every diffusion backend, whose
#      state is one-hot/4. Handed those features unconverted it would read
#      types 4x too small, return finite numbers, and nothing downstream could
#      see it: stability and validity go through argmax. So our nets are
#      wrapped in `Calibrated` too -- slope 1, intercept 0, because they are
#      already in physical units -- purely to get `feat_scale`.
#
#   2. A GUARD THAT CAN ACTUALLY FIRE. build_pair's defence against exactly
#      that bug is `slope_over_mad`, and it cannot work here: our nets have no
#      fitted slope to compare against QM9's MAD. So a calibration IS fitted,
#      on the same molecules and with the same least squares -- and then
#      DISCARDED -- and its slope is asserted to be ~1. A well-calibrated
#      predictor fed the right features fits slope 1; one fed features 4x too
#      small does not. The fit is a thermometer, not a correction.
#
# The delta rule is the SAME rule (k x MAE of f_B), measured here on the same
# 3000 calibration molecules rather than read off the checkpoint's `val_mae`,
# so `delta_mode` keeps its two documented values and a cell from either pair
# is scored the same way. `val_mae` is recorded beside it for comparison.
def build_pair_ours(prop, d, sel, dev, k_delta, sampler_scale,
                    delta_mode="local"):
    """(f_A, f_B, delta, report) using weights/f_A_<p>.pt and f_B_<p>.pt.

    Same signature and same return shape as `build_pair`, so the caller does
    not care which pair it asked for.
    """
    idx = PROP_INDEX[prop]
    if d["props"].index(prop) != idx:
        raise SystemExit("PROP_INDEX[%r]=%d but the data file puts it at %d"
                         % (prop, idx, d["props"].index(prop)))
    c = d["coords"][sel].to(dev)
    f = d["feats"][sel].to(dev)
    m = d["mask"][sel].to(dev)
    truth = d["y"][sel, idx].to(dev)
    n_types = len(d["types"])

    ga_path = require_predictor("f_A_%s.pt" % prop)
    gb_path = require_predictor("f_B_%s.pt" % prop)
    raw_guide = PhysicalProperty(ga_path, n_types, dev)
    raw_oracle = PhysicalProperty(gb_path, n_types, dev)

    # Fit ONLY to check the scale (see the note above); the fit is not applied.
    ag, bg, mae_g = fit_calibration(raw_guide, c, f, m, truth)
    ao, bo, mae_o = fit_calibration(raw_oracle, c, f, m, truth)

    y_std = float(truth.double().std())
    # slope 1, intercept 0: already physical. feat_scale converts the SAMPLER's
    # space to raw one-hot, which is the space both nets were trained in.
    f_A = Calibrated(raw_guide, 1.0, 0.0, prop, y_std,
                     feats_are_normalised=False, feat_scale=sampler_scale)
    f_B = Calibrated(raw_oracle, 1.0, 0.0, prop, y_std,
                     feats_are_normalised=False, feat_scale=sampler_scale)

    delta_global = choose_delta(mae_o, k_delta)
    if delta_mode == "local":
        # local_delta takes (raw_oracle, slope, intercept) positionally and
        # applies them; identity is correct here for the same reason as above.
        delta, drep = local_delta(prop, d, raw_oracle, 1.0, 0.0, dev, k=k_delta)
    elif delta_mode == "global":
        delta, drep = delta_global, {"rule": "delta = %g x MAE(f_B) over the "
                                             "calibration molecules" % k_delta,
                                     "k": k_delta, "local_mae": None,
                                     "generator_independent": True}
    else:
        raise SystemExit("unknown --delta-mode %r" % delta_mode)

    def _val_mae(path):
        try:
            return float(torch.load(path, map_location="cpu",
                                    weights_only=False)["val_mae"])
        except Exception:                          # noqa: BLE001
            return None

    report = {"pair": "ours",
              "guide": "ours %s (arch %s)" % (os.path.relpath(ga_path, ROOT),
                                              raw_guide.arch),
              "oracle": "ours %s (arch %s)" % (os.path.relpath(gb_path, ROOT),
                                               raw_oracle.arch),
              # fitted and DISCARDED -- the scale thermometer, not a correction
              "fitted_slope_guide": ag, "fitted_intercept_guide": bg,
              "fitted_slope_oracle": ao, "fitted_intercept_oracle": bo,
              "guide_slope": 1.0, "guide_intercept": 0.0, "guide_mae": mae_g,
              "oracle_slope": 1.0, "oracle_intercept": 0.0, "oracle_mae": mae_o,
              "qm9_mad": QM9_MAD[prop], "y_std": y_std,
              "guide_val_mae": _val_mae(ga_path),
              "oracle_val_mae": _val_mae(gb_path),
              "n_calibration": int(len(sel)),
              "calibration_set": "train_a+train_b, seed %d" % CALIB_SEED,
              "sampler_feat_scale": float(sampler_scale),
              "guide_input_multiplier": f_A.feat_scale,
              "oracle_input_multiplier": f_B.feat_scale,
              "delta_mode": delta_mode, "delta": delta,
              "delta_global": delta_global, "delta_detail": drep}
    # THE SCALE GUARD. A predictor already in physical units, fed the features
    # it was trained on, fits slope ~1. Fed them on the wrong scale it does
    # not -- which is the one failure mode that is otherwise invisible.
    for who, sl in (("guide", ag), ("oracle", ao)):
        if not 0.85 < sl < 1.15:
            raise SystemExit(
                "%s our %s fits calibration slope %.3f; expected ~1. Our "
                "predictors are already in physical units, so a slope far "
                "from 1 means the sampler's features are reaching them on the "
                "wrong scale (sampler_feat_scale=%g). Every number downstream "
                "would look fine and be wrong."
                % (prop, who, sl, sampler_scale))
    return f_A, f_B, delta, report


def build_pair(prop, d, sel, dev, k_delta, sampler_scale, delta_mode="local"):
    """(f_A, f_B, delta, report): TFG's guide and TFG's oracle, in physical
    units, calibrated on the SAME molecules with the SAME two-parameter least
    squares, so nothing about the comparison between them depends on it.
    Raises SystemExit if either slope is not ~QM9's MAD (wrong feature scale).

    `sampler_scale` is the GENERATOR's type divisor (generator_feat_scale):
    the sampler's features are onehot / sampler_scale, and each wrapper
    converts from that space to its own network's. It is required, not
    defaulted, because the three networks do NOT share one: EDMsecond's
    normalize_factors are [1, 4, 10] and TFG's guides' are [1, 8, 1]. An
    earlier version assumed the sampler's space WAS the guide's (onehot / 8),
    so on generated molecules the guide read one-hot 2x too large and the
    oracle read 2 x one-hot. Stability and validity use argmax and could not
    see it; on unguided samples it put the guide-oracle gap at 2.50 D on mu
    (0.20 D once decoded) and the oracle's median mu at 4.15 D (QM9: 2.49).
    """
    idx = PROP_INDEX[prop]
    if d["props"].index(prop) != idx:
        raise SystemExit("PROP_INDEX[%r]=%d but the data file puts it at %d"
                         % (prop, idx, d["props"].index(prop)))
    c = d["coords"][sel].to(dev)
    f = d["feats"][sel].to(dev)
    m = d["mask"][sel].to(dev)
    truth = d["y"][sel, idx].to(dev)

    raw_guide = TFGGuide(TFG_ROOT, prop, device=dev)
    raw_oracle = TFGOracle(TFG_ROOT, prop, device=dev)

    # Three feature spaces, each read off its own checkpoint's args:
    #   sampler  onehot / sampler_scale     (EDMsecond: / 4)
    #   guide    onehot / guide_scale       (tf_predict: / 8)
    #   oracle   raw onehot                 (evaluate_*: EDM's main_qm9_prop)
    # Fit each network in the space it wants -- `f` here is raw one-hot from
    # the data file -- then give each wrapper the multiplier from the SAMPLER's
    # space to its own, which is the space every caller hands it.
    guide_scale = raw_guide.norm_values[1]
    ag, bg, mae_g = fit_calibration(
        lambda C, F, M: raw_guide(C, F / guide_scale, M), c, f, m, truth)
    ao, bo, mae_o = fit_calibration(raw_oracle, c, f, m, truth)

    y_std = float(truth.double().std())
    f_A = Calibrated(raw_guide, ag, bg, prop, y_std, feats_are_normalised=False,
                     feat_scale=sampler_scale / guide_scale)
    f_B = Calibrated(raw_oracle, ao, bo, prop, y_std, feats_are_normalised=False,
                     feat_scale=sampler_scale)
    # The project's pre-registered rule, via the same helper the main sweep
    # uses -- NOT a hardcoded 2.0.
    delta_global = choose_delta(mae_o, k_delta)
    if delta_mode == "local":
        delta, drep = local_delta(prop, d, raw_oracle, ao, bo, dev, k=k_delta)
    elif delta_mode == "global":
        delta, drep = delta_global, {"rule": "delta = %g x MAE(f_B) over the "
                                            "calibration molecules" % k_delta,
                                    "k": k_delta, "local_mae": None,
                                    "generator_independent": True}
    else:
        raise SystemExit("unknown --delta-mode %r" % delta_mode)
    report = {"guide": "TFG tf_predict_%s/model_ema_2000.npy" % prop,
              "oracle": "TFG evaluate_%s/best_checkpoint.npy" % prop,
              "guide_slope": ag, "guide_intercept": bg, "guide_mae": mae_g,
              "oracle_slope": ao, "oracle_intercept": bo, "oracle_mae": mae_o,
              "qm9_mad": QM9_MAD[prop], "y_std": y_std,
              "slope_over_mad_guide": ag / QM9_MAD[prop],
              "slope_over_mad_oracle": ao / QM9_MAD[prop],
              "n_calibration": int(len(sel)),
              "calibration_set": "train_a+train_b, seed %d" % CALIB_SEED,
              "sampler_feat_scale": float(sampler_scale),
              "guide_feat_scale": float(guide_scale),
              "guide_input_multiplier": f_A.feat_scale,
              "oracle_input_multiplier": f_B.feat_scale,
              # how delta was set travels with every cell, because the whole
              # in-band column means something different if this changes
              "delta_mode": delta_mode, "delta": delta,
              "delta_global": delta_global, "delta_detail": drep}
    # A slope far from the published MAD means the adapter is feeding the
    # network the wrong units, and every number downstream would be
    # meaningless while looking fine. Fail loudly here instead.
    for who in ("guide", "oracle"):
        r = report["slope_over_mad_%s" % who]
        if not 0.9 < r < 1.1:
            raise SystemExit(
                "%s %s calibration slope is %.3f x QM9's MAD; expected ~1. "
                "The adapter is almost certainly feeding it features on the "
                "wrong scale." % (prop, who, r))
    return f_A, f_B, delta, report


# --------------------------------------------------------------------------
# freeze: FR3a / FR3 on the transfer's own q90 compare cells
# --------------------------------------------------------------------------

def _se_mae(r):
    v = max(r[SELECT_RMSE] ** 2 - r[SELECT_METRIC] ** 2, 0.0)
    return math.sqrt(v / r["n"])


def _load_stage(args, props, stage_label, targets, grid):
    """(rows, present, meta) for this backend's `stage_label` cells, or an int
    exit code (2) with the reason printed. `grid` is the set of (arm, w) that
    must ALL be present at every target -- a best-of-partial-grid is never
    taken. A cell that ran but diverged counts as present and is excluded
    only from being anyone's best."""
    rows = {t: [] for t in targets}
    present = {t: set() for t in targets}
    meta = set()
    if not os.path.isdir(args.out_dir):
        print("no cells: %s does not exist" % args.out_dir)
        return 2
    for fn in os.listdir(args.out_dir):
        if not (fn.startswith("tr__") and fn.endswith(".json")):
            continue
        r = json.load(open(os.path.join(args.out_dir, fn)))
        if (r.get("stage") != stage_label or r.get("target_name") not in rows
                or r.get("prop") not in props):
            continue
        if r.get("backend") != BACKENDS[args.backend]:
            print("REFUSING: %s is a %r cell; this is --backend %s"
                  % (fn, r.get("backend"), args.backend))
            return 2
        t = r["target_name"]
        present[t].add((r["prop"], r["arm"], float(r["w"])))
        # `delta_mode` is in here because delta is NOT in a cell's filename: two
        # cells can differ only in what their in-band column means, and a freeze
        # that averaged across them would pick a frontier off a mixed metric with
        # nothing on the page to show it. (delta itself is per property, so it is
        # checked per property below, not here.)
        meta.add((r["n"], r["seed"], (r.get("prov") or {}).get("fm_md5"),
                  r.get("grid"), r.get("tau_max_guide"), r.get("batch"),
                  tuple(r.get("dist_block") or ()),
                  (r.get("calibration") or {}).get("delta_mode")))
        if r.get("n_nonfinite", 0) > 0 or not math.isfinite(
                float(r.get(SELECT_METRIC, float("nan")))):
            continue
        rows[t].append(r)
    if len(meta) != 1:
        print("INCONSISTENT %s cells in %s: (n, seed, fm_md5, grid, window, "
              "batch, block, delta_mode) = %s" % (stage_label, args.out_dir,
                                                  sorted(map(str, meta))))
        return 2
    # delta is per property, so it cannot live in the tuple above. Cells with no
    # recorded delta are skipped rather than compared: a missing value is not
    # evidence of disagreement, and NaN != NaN would make every such set look
    # inconsistent.
    for p in props:
        ds = {round(float(r["delta"]), 12)
              for rr in rows.values() for r in rr
              if r["prop"] == p and math.isfinite(float(r.get("delta") or
                                                        float("nan")))}
        if len(ds) > 1:
            print("INCONSISTENT delta for %s: %s -- in-band means a different "
                  "thing in each cell" % (p, sorted(ds)))
            return 2
    for t in targets:
        holes = sorted((p, a, w) for p in props for (a, w) in grid
                       if (p, a, w) not in present[t])
        if holes:
            print("INCOMPLETE -- %d of %d %s cells missing at %s, e.g. %s"
                  % (len(holes), len(grid) * len(props), stage_label, t,
                     ", ".join("%s/%s/w%g" % h for h in holes[:6])))
            return 2
    return rows, present, meta


def _fr3a(rows, p, a, floor):
    """(row, fell_back): best SELECT_METRIC among strengths with mol_stability
    >= floor (tie -> smaller w); none clears it -> the most stable (tie ->
    smaller w). check_fullrun_go.py's rule, on the decoded MAE."""
    c = [r for r in rows if r["prop"] == p and r["arm"] == a]
    if not c:
        return None, False
    ok = [r for r in c if r["mol_stability"] >= floor - 1e-12]
    if ok:
        return min(ok, key=lambda r: (r[SELECT_METRIC], float(r["w"]))), False
    return max(c, key=lambda r: (r["mol_stability"], -float(r["w"]))), True


def _base_grid(arms, strengths=None):
    ws = TRANSFER_STRENGTHS if strengths is None else strengths
    return {("unguided", float(DEFAULT_W))} | {
        (a, float(w)) for a in arms if a != "unguided" for w in ws}


def extension_plan(rows, props, arms, strengths=None, target=FREEZE_TARGET):
    """The edge rule's cells: decided on the BASE grid only, so the plan is the
    same before and after the extension cells exist. Used for the compare
    screen (q90, TRANSFER_STRENGTHS) and the equal-chemistry tune block
    (dist, EQ_STRENGTHS) alike."""
    ws = TRANSFER_STRENGTHS if strengths is None else strengths
    base = [r for r in rows if (r["arm"], float(r["w"])) in _base_grid(arms, ws)]
    lo, hi = min(ws), max(ws)
    plan = []
    for p in props:
        u = [r for r in base if r["prop"] == p and r["arm"] == "unguided"]
        if not u:
            continue
        floor = FLOOR * u[0]["mol_stability"]
        for a in arms:
            if a == "unguided":
                continue
            r, _ = _fr3a(base, p, a, floor)
            if r is None:
                continue
            w = float(r["w"])
            if w == hi:
                plan += [(p, a, target, hi * f) for f in EXT_FACTORS]
            elif w == lo:
                plan += [(p, a, target, lo / f) for f in EXT_FACTORS]
    return plan


def in_band_at_floor(points, floor):
    """Equal-chemistry-cost in_band for one arm. `points` are (w, mol_stab,
    in_band) sorted by w. Where the stability curve FIRST drops below the
    floor, in_band is interpolated linearly in stability between the two
    strengths that bracket the crossing. If it never drops below on the grid
    the value is the last strength's in_band, a lower bound ("not_reached");
    if it is below from the smallest strength there is no value."""
    if not points:
        return {"value": None, "kind": "no_points"}
    if points[0][1] < floor:
        return {"value": None, "kind": "below_floor"}
    for (w0, s0, b0), (w1, s1, b1) in zip(points, points[1:]):
        if s1 < floor:
            f = (s0 - floor) / (s0 - s1) if s0 != s1 else 0.0
            return {"value": b0 + f * (b1 - b0), "kind": "interpolated",
                    "between_w": [w0, w1]}
    return {"value": points[-1][2], "kind": "not_reached",
            "at_w": points[-1][0]}


def eqfreeze(args, props, arms):
    """FR3a on the tune block, plus the equal-chemistry frontier. 0 / 2."""
    if sorted(arms) != sorted(backend_arms(args.backend)):
        print("REFUSING to freeze arms %s: must be the whole arm set %s"
              % (arms, backend_arms(args.backend)))
        return 2
    grid = {("unguided", float(DEFAULT_W))} | {
        (a, float(w)) for a in arms if a != "unguided" for w in EQ_STRENGTHS}
    loaded = _load_stage(args, props, "eqtune", (DIST_TARGET,), grid)
    if isinstance(loaded, int):
        return loaded
    rows, present, meta = loaded
    rd = rows[DIST_TARGET]
    ext = extension_plan(rd, props, arms, EQ_STRENGTHS, DIST_TARGET)
    missing = [c for c in ext
               if (c[0], c[1], float(c[3])) not in present[DIST_TARGET]]
    if missing:
        print("EXTENSION INCOMPLETE -- %d edge-rule tune cell(s) missing, e.g. %s; "
              "run --stage eqextend" % (len(missing), ", ".join(
                  "%s/%s/w%g" % (c[0], c[1], c[3]) for c in missing[:6])))
        return 2
    frozen, stab, floor, fell_back, frontier = {}, {}, {}, [], {}
    for p in props:
        u = [r for r in rd if r["prop"] == p and r["arm"] == "unguided"]
        if not u:
            print("no clean unguided cell for %s" % p)
            return 2
        floor[p] = FLOOR * u[0]["mol_stability"]
        frontier[p] = {}
        for a in arms:
            r, fb = _fr3a(rd, p, a, floor[p])
            if r is None:
                print("NO CLEAN CELL at any strength for %s/%s" % (a, p))
                return 2
            frozen.setdefault(a, {})[p] = r["w"]
            stab.setdefault(a, {})[p] = r["mol_stability"]
            if fb:
                fell_back.append("%s/%s" % (a, p))
            c = sorted((x for x in rd if x["prop"] == p and x["arm"] == a),
                       key=lambda x: float(x["w"]))
            pts = [(float(x["w"]), x["mol_stability"], x["in_band_fraction_dec"])
                   for x in c]
            frontier[p][a] = {
                "points": [{"w": float(x["w"]), "mol_stability": x["mol_stability"],
                            "in_band_dec": x["in_band_fraction_dec"],
                            "in_band_soft": x["in_band_fraction"],
                            "mae_dec": x[SELECT_METRIC]} for x in c],
                "in_band_at_floor": in_band_at_floor(pts, floor[p])}
    print("EQUAL-CHEMISTRY FRONTIER on test[%d:%d] (decoded in_band at the "
          "floor, mol_stability >= %.1f x unguided)" % (
              args.block_start, args.block_start + args.n, FLOOR))
    for p in props:
        print("\n  %s  floor %.3f" % (p, floor[p]))
        print("  %-10s %9s %14s   frozen w (stab)" % ("arm", "at floor", "kind"))
        for a in arms:
            ib = frontier[p][a]["in_band_at_floor"]
            v = ib["value"]
            print("  %-10s %9s %14s   %g (%.3f)" % (
                a, "-" if v is None else "%.4f" % v, ib["kind"],
                frozen[a][p], stab[a][p]))
    if fell_back:
        print("no strength clears the floor for: %s -> most stable used"
              % ", ".join(fell_back))
    (n, seed, md5, grid_, win, batch, blk, dmode), = meta
    # delta_mode travels with the frozen strengths: a strength chosen under one
    # definition of in-band does not carry to a run scored under another, and
    # --stage full / basecmpfull refuse the mismatch. It comes off the meta tuple,
    # which _load_stage has already proved unique across the whole stage.
    out = {"delta_mode": dmode or "global",
           "backend": BACKENDS[args.backend], "gen_md5": md5, "arms": list(arms),
           "frozen_w": frozen,
           "rule": "FR3a on the tune block: best %s among strengths with "
                   "mol_stability >= %.1f x unguided" % (SELECT_METRIC, FLOOR),
           "select_metric": SELECT_METRIC, "floor": floor,
           "fell_back": fell_back, "frontier": frontier,
           "eq_strengths": EQ_STRENGTHS, "extension": ext,
           "tfg_post_hoc": False,
           "source_stage": "eqtune", "target": DIST_TARGET,
           "source_seed": seed, "source_n": n, "source_batch": batch,
           "source_block": list(blk), "source_dir": args.out_dir,
           "grid": grid_, "tau_max_guide": win}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        tmp = args.json_out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, args.json_out)
        print("\nwrote %s" % args.json_out)
    return 0


def load_eq_frozen(path, props, arms, backend):
    with open(path) as fh:
        fz = json.load(fh)
    if (fz.get("source_stage") != "eqtune" or fz.get("target") != DIST_TARGET
            or fz.get("backend") != BACKENDS[backend]):
        raise SystemExit("%s is not an eqtune freeze on %s (stage=%r target=%r "
                         "backend=%r)" % (path, BACKENDS[backend],
                                          fz.get("source_stage"), fz.get("target"),
                                          fz.get("backend")))
    miss = ["%s/%s" % (a, p) for a in arms for p in props
            if p not in (fz.get("frozen_w") or {}).get(a, {})]
    if miss:
        raise SystemExit("%s has no frozen strength for %s" % (path, ", ".join(miss)))
    return fz


def freeze(args, props, arms):
    """Apply FR3a / FR3 to the finished compare stage; exit 0 ok, 2 incomplete.

    check_fullrun_go.py's rule, tie-breaks and json keys -- so
    guidance_sweep.load_frozen and full_run_table.py read the result unchanged
    -- with ONE deliberate difference: strengths are chosen on the DECODED
    MAE (SELECT_METRIC), where the main sweep chose on the soft one. The two
    sweeps therefore select on different metrics; say so wherever they are
    compared. Also: the edge rule's cells must exist (extension_plan).
    """
    if sorted(arms) != sorted(backend_arms(args.backend)):
        # the frozen file IS the full run's arm set (load_frozen only checks
        # the arms it is asked about), so a reduced --arms here would produce
        # a reduced headline table without a word of protest
        print("REFUSING to freeze arms %s: the frozen file must cover the whole "
              "arm set for --backend %s, %s. A post-hoc arm is a separate "
              "analysis (R5)." % (arms, args.backend, backend_arms(args.backend)))
        return 2
    loaded = _load_stage(args, props, "compare", SCREEN_TARGETS,
                         _base_grid(arms))
    if isinstance(loaded, int):
        return loaded
    rows, present, meta = loaded
    # the edge rule's cells must exist before anything is frozen
    ext = extension_plan(rows[FREEZE_TARGET], props, arms)
    missing = [c for c in ext
               if (c[0], c[1], float(c[3])) not in present[FREEZE_TARGET]]
    if missing:
        print("EXTENSION INCOMPLETE -- %d edge-rule cell(s) missing, e.g. %s; "
              "run --stage extend" % (len(missing), ", ".join(
                  "%s/%s/w%g" % (c[0], c[1], c[3]) for c in missing[:6])))
        return 2
    if ext:
        print("edge rule: %d extra q90 cell(s) included: %s" % (len(ext), ", ".join(
            "%s/%s/w%g" % (c[0], c[1], c[3]) for c in ext)))

    def best(rs, p, a):
        c = [r for r in rs if r["prop"] == p and r["arm"] == a]
        return min(c, key=lambda r: (r[SELECT_METRIC], float(r["w"]))) if c else None

    # FR1, for the record only (see the module docstring)
    comps = [a for a in arms if ARM_CLASS.get(a) == "COMPARE"]
    fr1 = {}
    for t in SCREEN_TARGETS:
        beaten, lines = 0, []
        for p in props:
            b = best(rows[t], p, "btvg")
            cs = [x for x in (best(rows[t], p, a) for a in comps) if x]
            if b is None or not cs:
                continue
            c = min(cs, key=lambda r: r[SELECT_METRIC])
            z = (b[SELECT_METRIC] - c[SELECT_METRIC]) / math.sqrt(
                _se_mae(b) ** 2 + _se_mae(c) ** 2)
            beaten += z > FR1_SIGMA
            lines.append("  %-6s best comp %-7s %.4f  btvg %.4f  z %+.2f%s"
                         % (p, c["arm"], c[SELECT_METRIC], b[SELECT_METRIC],
                            z, "  BEATEN" if z > FR1_SIGMA else ""))
        fr1[t] = {"beaten_on": beaten, "pass": beaten < FR1_BEATEN_ON}
        print("FR1 (record only) at %s: btvg beaten on %d of %d -> %s"
              % (t, beaten, len(props), "pass" if fr1[t]["pass"] else "FAIL"))
        print("\n".join(lines))

    # FR3 as registered, and FR3a (the headline set), on q90
    rq = rows[FREEZE_TARGET]
    frozen_mae, frozen, stab, floor, fell_back = {}, {}, {}, {}, []
    for p in props:
        u = [r for r in rq if r["prop"] == p and r["arm"] == "unguided"]
        if not u:
            print("no clean unguided cell for %s at %s" % (p, FREEZE_TARGET))
            return 2
        floor[p] = FLOOR * u[0]["mol_stability"]
    for a in arms:
        for p in props:
            c = [r for r in rq if r["prop"] == p and r["arm"] == a]
            if not c:
                print("NO CLEAN CELL at any strength for %s/%s" % (a, p))
                return 2
            frozen_mae.setdefault(a, {})[p] = best(rq, p, a)["w"]
            r, fb = _fr3a(rq, p, a, floor[p])
            if fb:
                fell_back.append("%s/%s" % (a, p))
            frozen.setdefault(a, {})[p] = r["w"]
            stab.setdefault(a, {})[p] = r["mol_stability"]
    print("\nFR3a strengths (best q90 MAE with mol_stability >= %.1f x unguided;"
          " floor %s)" % (FLOOR, ", ".join("%s %.3f" % (p, floor[p]) for p in props)))
    print("%-10s %s   | FR3 as registered" % ("arm", " ".join(
        "%16s" % ("%s w (stab)" % p) for p in props)))
    for a in arms:
        print("%-10s %s   | %s" % (a, " ".join(
            "%16s" % ("%g (%.3f)" % (frozen[a][p], stab[a][p])) for p in props),
            " ".join("%6g" % frozen_mae[a][p] for p in props)))
    if fell_back:
        print("no strength clears the floor for: %s -> most stable used"
              % ", ".join(fell_back))
    (n, seed, md5, grid_, win, batch, _blk, dmode), = meta
    # delta_mode travels with the frozen strengths: a strength chosen under one
    # definition of in-band does not carry to a run scored under another, and
    # --stage full / basecmpfull refuse the mismatch. It comes off the meta tuple,
    # which _load_stage has already proved unique across the whole stage.
    out = {"delta_mode": dmode or "global",
           "backend": BACKENDS[args.backend], "gen_md5": md5, "arms": list(arms),
           "fr1_record_only": fr1,
           "frozen_w": frozen, "frozen_w_mae": frozen_mae,
           "rule": "frozen_w = FR3a: best MAE among strengths with "
                   "mol_stability >= %.1f x unguided; frozen_w_mae = FR3 as "
                   "registered" % FLOOR,
           "floor": floor, "fell_back": fell_back,
           "source_stage": "compare", "target": FREEZE_TARGET,
           "source_seed": seed, "source_n": n, "source_batch": batch,
           "source_dir": args.out_dir,
           "grid": grid_, "tau_max_guide": win,
           "grid_strengths": TRANSFER_STRENGTHS, "extension": ext,
           "select_metric": SELECT_METRIC,
           # every arm, tfg included, was fixed before any EquiFM cell ran
           "tfg_post_hoc": False}
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        tmp = args.json_out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(tmp, args.json_out)
        print("\nwrote %s" % args.json_out)
    return 0


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", default="compare",
                    choices=["compare", "extend", "freeze", "full",
                             "eqtune", "eqextend", "eqfreeze", "eqconfirm",
                             "basecmp", "basecmprefine", "basecmpfull",
                             "v3", "v3abl"])
    ap.add_argument("--backend", default="equifm", choices=sorted(BACKENDS),
                    help="which generator: equifm (the borrowed FM base, "
                         "scored with TFG's pair), fm (OURS, scored with OUR "
                         "pair), or edm (TFG's EDMsecond as the QM9 diffusion "
                         "base, our pair, unguided+plug only). See "
                         "V3_BACKENDS.")
    ap.add_argument("--edm-dir", default=os.path.join(ROOT, "weights", "EDMsecond"),
                    help="--backend edm: directory holding generative_model_ema.npy + args.pickle")
    ap.add_argument("--fm-ckpt", default=FM_CKPT,
                    help="--backend fm: our generator checkpoint")
    ap.add_argument("--t-starts", default="",
                    help="--stage basecmp: comma list of flow-time instants at "
                         "which guidance switches on (default %s). On a VP or "
                         "EquiFM backend each is applied as tau_max_guide = "
                         "1 - t_start, so one number means the same physical "
                         "instant on both families."
                         % ",".join("%g" % t for t in BASECMP_T_STARTS))
    ap.add_argument("--delta-mode", default=None, choices=["local", "global"],
                    help="how the in-band half-width is set. 'local' (the rule "
                         "settled 24 Sep) is k x f_B's MAE over val molecules "
                         "whose true property is near the q90 target; 'global' "
                         "is k x its MAE over all calibration molecules. Both "
                         "depend on f_B and QM9 only, never on the generator, so "
                         "both are identical across backends -- which is what "
                         "makes the base-model comparison a comparison. "
                         "DEFAULTS BY STAGE: 'local' for the basecmp stages, "
                         "which pre-register it; 'global' for every stage that "
                         "existed before it, whose cells are already on disk "
                         "under that definition.")
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--targets", default=",".join(SCREEN_TARGETS),
                    help="--stage compare: which screen targets to run (to "
                         "split the grid across jobs); freeze needs both")
    ap.add_argument("--arms", default="",
                    help="comma list; default TRANSFER_SET (the main full-run set)")
    ap.add_argument("--n", type=int, default=None,
                    help="default 512 for compare, 5000 for full")
    ap.add_argument("--batch", type=int, default=128,
                    help="part of the cell name: it changes the initial noise")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--solver", default="euler", choices=["euler", "heun"])
    ap.add_argument("--grid", default="uniform", choices=["uniform", "gamma"],
                    help="tau grid. 'uniform' is what the hard gate measured "
                         "best at 100 steps (96.4 / 67.3 vs gamma 95.9 / 63.8)")
    ap.add_argument("--tau-min", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--k-delta", type=float, default=2.0)
    ap.add_argument("--n-probe", type=int, default=1)
    ap.add_argument("--n-mc", type=int, default=4)
    ap.add_argument("--sigma-mc", type=float, default=0.1)
    ap.add_argument("--tau-max-guide", type=float, default=0.5,
                    help="skip guidance while tau > this, i.e. in the noisy "
                         "part of the trajectory. 0.5 is the VP image of the "
                         "main sweep's t_min_guide = 0.5 (V2_WIN); 1.0 guides "
                         "everywhere. NOT a free parameter -- see the module "
                         "docstring, item 5.")
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    ap.add_argument("--seed", type=int, default=None,
                    help="default %d for compare; REQUIRED for full" % COMPARE_SEED)
    ap.add_argument("--block-start", type=int, default=None,
                    help="`dist` molecules are test[start : start + n]; default "
                         "0 (full), %d (eqtune), %d (eqconfirm)"
                         % (EQ_TUNE_START, EQ_CONFIRM_START))
    ap.add_argument("--strengths", default="",
                    help="run only these strengths (comma list) of the planned "
                         "cells -- splits a stage across jobs; unguided is kept")
    ap.add_argument("--frozen", default="",
                    help="--stage full: the json --stage freeze wrote; "
                         "--stage basecmpfull: the json basecmp_freeze.py wrote")
    ap.add_argument("--refine", default="",
                    help="--stage basecmprefine: the json "
                         "`basecmp_freeze.py --emit-refine` wrote")
    ap.add_argument("--sets", default="primary",
                    help="--stage full: strength sets (primary = FR3a, mae = FR3)")
    ap.add_argument("--exclude-sets", default="")
    ap.add_argument("--json-out", default="",
                    help="--stage freeze: where to write the frozen strengths")
    ap.add_argument("--per-mol", action="store_true",
                    help="also save <cell>.permol.pt (per-molecule f_A, f_B, "
                         "target, stability, smiles, mol_idx)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out-dir", default="",
                    help="default results/transfer/compare, or "
                         "results/transfer/full/n<N>/seed<seed> for full")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pair", default="", choices=["", "ours", "tfg"],
                    help="which property pair scores this run. Default: the "
                         "backend's own, declared in V3_BACKENDS -- `ours` "
                         "for fm and the QM9 diffusion base, `tfg` for "
                         "equifm. delta = k x MAE(f_B), so THE PAIR SETS THE "
                         "BAND WIDTH and in_band is not comparable across "
                         "backends that use different pairs.")
    ap.add_argument("--preflight", action="store_true",
                    help="one throwaway cell per arm at minimum cost, writes "
                         "nothing, exits nonzero if any arm raises")
    args = ap.parse_args()

    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a] or backend_arms(args.backend)
    # bdg arms carry (eta, tau_mult) in the name, so they cannot be listed in
    # ARM_CLASS. parse_bdg_arm validates the suffix and raises on a typo, which
    # is what would otherwise reach the sampler as an unknown mode.
    for a in arms:
        parse_bdg_arm(a)
    unknown = [a for a in arms if a not in ARM_CLASS and not a.startswith("bdg_e")]
    if unknown:
        raise SystemExit("unknown arm(s) %s; known: %s"
                         % (unknown, ", ".join(ARM_CLASS)))
    if args.n is None:
        # v3 PRE-REGISTERS NO n (Henry, 26 Sep): the operator picks the cell
        # size for the cluster they are on. V3_N is None on purpose, so refuse
        # here rather than silently falling through to the 512 default -- a v3
        # tree written at 512 would look like a real run and be badly
        # underpowered. FULL_RUN_V3_PROTOCOL.md section 6.1 has the power table
        # to choose against.
        if args.stage in ("v3", "v3abl") and not args.preflight:
            # --preflight overrides n to 8 further down (it runs one throwaway
            # cell per arm and writes nothing), so requiring --n here would
            # break the one call that legitimately has no cell size.
            raise SystemExit(
                "--stage %s needs an explicit --n. The v3 protocol does not "
                "fix the cell size; pick it against the power table in "
                "docs/protocol/FULL_RUN_V3_PROTOCOL.md section 6.1, and make "
                "it divisible by --batch (the batch is BDG's estimator)."
                % args.stage)
        args.n = {"full": 5000, "eqconfirm": EQ_CONFIRM_N, "eqtune": EQ_TUNE_N,
                  "eqextend": EQ_TUNE_N, "eqfreeze": EQ_TUNE_N,
                  "basecmp": BASECMP_N, "basecmprefine": BASECMP_N,
                  "basecmpfull": 2000}.get(args.stage, 512)
    if args.seed is None:
        if args.stage in ("full", "eqconfirm", "basecmpfull", "v3", "v3abl"):
            raise SystemExit("--stage %s needs --seed (one of %s)" % (
                args.stage, ", ".join(map(str, {
                    "full": FULL_SEEDS, "eqconfirm": EQ_CONFIRM_SEEDS,
                    "basecmpfull": BASECMP_FULL_SEEDS,
                    "v3": V3_SEEDS, "v3abl": V3_SEEDS}[args.stage]))))
        args.seed = (EQ_TUNE_SEED if args.stage in ("eqtune", "eqextend", "eqfreeze")
                     else BASECMP_SEED if args.stage in ("basecmp", "basecmprefine")
                     else COMPARE_SEED)
    basecmp = args.stage in ("basecmp", "basecmprefine", "basecmpfull")
    # DELTA'S DEFAULT IS PER STAGE, and this is not a convenience. The local rule
    # is pre-registered for the basecmp stages only. Every stage that existed
    # before it has cells ON DISK computed with the global rule -- 216 EDMsecond
    # compare cells among them -- and `delta` is not in a cell's filename, so
    # defaulting `local` everywhere would let a resumed stage write cells whose
    # in-band column means something different from its neighbours', with nothing
    # to show it. `_load_stage` and the frozen-file gate below now also refuse a
    # mixed set, but the default is what stops it arising.
    if args.delta_mode is None:
        args.delta_mode = "local" if basecmp else "global"
    if args.delta_mode == "local" and not basecmp:
        print("NOTE: --delta-mode local on --stage %s. The local window is fixed "
              "at the q90 target's neighbourhood, so it is the WRONG "
              "neighbourhood for a q50 or `dist` cell, and cells already on disk "
              "for this stage used the global rule. Write them somewhere new."
              % args.stage)
    if basecmp and args.backend == "edm":
        raise SystemExit("the base-model comparison is our FM base against "
                         "EquiFM; --backend edm was dropped on 23 Sep")
    t_starts = ([float(t) for t in args.t_starts.split(",") if t]
                or list(BASECMP_T_STARTS))
    for t in t_starts:
        if not 0.0 < t < 1.0:
            raise SystemExit("--t-starts must lie strictly in (0, 1); got %g. "
                             "t is flow time: 0 is pure noise, 1 is data, and "
                             "guidance runs on [t_start, 1)." % t)
    if args.block_start is None:
        args.block_start = {"eqtune": EQ_TUNE_START, "eqextend": EQ_TUNE_START,
                            "eqfreeze": EQ_TUNE_START,
                            "eqconfirm": EQ_CONFIRM_START}.get(args.stage, 0)
    if args.backend == "equifm":
        # EquiFM integrates its own native tau grid (1 -> 0, uniform); the
        # --grid choice belongs to the EDM schedule and is recorded as such
        args.grid = "native"
    elif args.backend == "fm":
        # our generator integrates flow time 0 -> 1 on a uniform grid; the
        # EDM schedule choice does not apply and must not be recorded as if it did
        args.grid = "flow"
    v3stage = args.stage in ("v3", "v3abl")
    root_out = (V3_ROOT if v3stage else
                BASECMP_ROOT if basecmp else OUT_ROOTS[args.backend])
    if not args.out_dir:
        if v3stage:
            # v3 keeps its own tree, split by backend AND BY STAGE, so it can
            # never be read into v2's freeze, the transfer's freeze or basecmp's
            # table -- all of which assume a chemistry floor v3 does not use.
            #
            # THE STAGE LEVEL IS LOAD-BEARING. The headline and the ablation now
            # run at the SAME n and the SAME three seeds, so n can no longer
            # tell their cells apart; without this directory v3_table.py would
            # glob the ablation's 15 extra arms into the headline's table. n
            # used to do this job, which is the only reason the ablation was
            # ever run at a different size.
            #
            # Consequence: the two stages share nothing, so the ablation
            # recomputes its own eta = 4 rungs rather than reusing the
            # headline's. That duplication is deliberate and it is what makes
            # each stage's tree readable on its own.
            args.out_dir = os.path.join(root_out, args.backend,
                                        V3_STAGE_DIRS[args.stage],
                                        "n%d" % args.n, "seed%d" % args.seed)
        elif basecmp:
            # the comparison keeps its OWN tree, per backend, so it can never be
            # read into the transfer's freeze or the main sweep's
            args.out_dir = (
                os.path.join(root_out, args.backend, "full", "n%d" % args.n,
                             "seed%d" % args.seed)
                if args.stage == "basecmpfull"
                else os.path.join(root_out, args.backend, "screen"))
        elif args.stage == "full":
            args.out_dir = os.path.join(root_out, "full", "n%d" % args.n,
                                        "seed%d" % args.seed)
        elif args.stage == "eqconfirm":
            args.out_dir = os.path.join(root_out, "eqchem", "confirm",
                                        "n%d" % args.n, "seed%d" % args.seed)
        elif args.stage in ("eqtune", "eqextend", "eqfreeze"):
            args.out_dir = os.path.join(root_out, "eqchem", "tune")
        else:
            args.out_dir = os.path.join(root_out, "compare")

    if args.stage == "freeze":
        return freeze(args, props, arms)
    if args.stage == "eqfreeze":
        return eqfreeze(args, props, arms)
    # the label a cell carries: extension cells ARE compare cells (at more
    # strengths), eqconfirm cells are full-scale `dist` cells
    cell_stage = {"compare": "compare", "extend": "compare", "full": "full",
                  "eqtune": "eqtune", "eqextend": "eqtune",
                  "eqconfirm": "full",
                  # refine cells ARE screen cells, at more strengths -- the same
                  # convention `extend` already follows, so one glob reads both
                  "basecmp": "basecmp", "basecmprefine": "basecmp",
                  "basecmpfull": "basecmpfull",
                  # The two v3 stages write DIFFERENT labels, and must. They
                  # once shared "v3" on the argument that the headline's bdg
                  # arms are a subset of the ablation's grid, so one cell could
                  # be written by either job. That stopped being true when the
                  # stages got their own directories: a cell now belongs to
                  # exactly one stage, and a shared label left v3_table.py
                  # unable to say which plan a tree was supposed to satisfy.
                  # The directory and the label agree; a mismatch is a bug.
                  "v3": "v3", "v3abl": "v3abl"}[args.stage]

    frozen = None
    if args.stage in ("v3", "v3abl"):
        # v3 fixes w and t_start, so there is no screen and no freeze: one cell
        # per (property, arm) per seed. --arms overrides the default arm set,
        # which is how the cluster splits the run across array tasks.
        if args.arms:
            v3set = arms
        elif args.stage == "v3":
            v3set = v3_arms()
        else:
            v3set = v3_arms(etas=V3_ABL_ETAS, tau_mults=V3_ABL_TAU_MULTS,
                            compare=False)
        # THE BACKEND MAY RUN FEWER ARMS THAN THE STAGE PLANS. The QM9
        # diffusion base is declared unguided+plug only (V3_BACKENDS), so the
        # filter is applied to whatever was planned OR passed with --arms. It
        # is applied last, so a --arms typo still fails loudly rather than
        # being silently narrowed away.
        v3set = v3_backend_arms(args.backend, v3set)
        if not v3set:
            # NOT an error. The QM9 diffusion base runs unguided+plug, which
            # the BDG ablation does not plan, so its v3abl tasks have nothing
            # to do. The array grid is uniform (every backend x property x
            # seed) because a ragged one is how stride bugs happen, so those
            # tasks exist and must succeed having done nothing -- loudly.
            print("nothing to run: --backend %s is declared to run %s, and "
                  "--stage %s plans none of them. Exiting 0."
                  % (args.backend,
                     ", ".join(V3_BACKENDS.get(args.backend, {}).get("arms")
                               or ["all arms"]),
                     args.stage))
            return 0
        arms = v3set
        cells = plan_v3_cells(props, v3set)
    elif args.stage == "basecmp":
        cells = plan_basecmp_cells(props, arms, t_starts)
    elif args.stage == "basecmprefine":
        # The cells to add are decided by basecmp_freeze.py --emit-refine, so
        # the neighbour rule and the pick rule live in ONE place and cannot
        # disagree about which cell is at a grid edge.
        if not args.refine:
            raise SystemExit("--stage basecmprefine needs --refine (the json "
                             "`basecmp_freeze.py --emit-refine` wrote)")
        plan = json.load(open(args.refine))
        if plan.get("backend") != BACKENDS[args.backend]:
            raise SystemExit("%s is a %r refine plan; this is --backend %s"
                             % (args.refine, plan.get("backend"), args.backend))
        cells = [(c["prop"], c["arm"], BASECMP_TARGET, float(c["w"]),
                  float(c["t_start"])) for c in plan["cells"]
                 if c["prop"] in props and c["arm"] in arms]
    elif args.stage == "basecmpfull":
        if not args.frozen:
            raise SystemExit("--stage basecmpfull needs --frozen (the json "
                             "`basecmp_freeze.py --json-out` wrote)")
        frozen = load_basecmp_frozen(args.frozen, props, arms, args.backend)
        # This stage's sets are ("floor", "free"), fixed by the protocol.
        # --sets/--exclude-sets name FR3a/FR3, which do not exist here, so a
        # caller passing them means something the plan cannot honour: refuse
        # rather than ignore, and never record "primary" on a cell that is not.
        if args.sets != "primary" or args.exclude_sets:
            raise SystemExit("--stage basecmpfull does not take --sets / "
                             "--exclude-sets (those name FR3a/FR3). Its sets are "
                             "%s and every cell records which it belongs to."
                             % ", ".join(BASECMP_SETS))
        cells = plan_basecmp_full_cells(props, arms, frozen)
    elif args.stage == "full":
        if not args.frozen:
            raise SystemExit("--stage full needs --frozen (the json "
                             "`--stage freeze --json-out` wrote)")
        sets = [s for s in args.sets.split(",") if s]
        excl = [s for s in args.exclude_sets.split(",") if s]
        frozen = load_frozen(args.frozen, props, arms, sets + excl)
        # load_frozen checks stage/target; it cannot tell the MAIN sweep's
        # frozen file from this one, and those strengths were chosen on a
        # different generator. Refuse it.
        if frozen.get("backend") != BACKENDS[args.backend]:
            raise SystemExit("%s was not frozen on %s (backend=%r) -- the main "
                             "sweep's strengths do not transfer; run "
                             "--stage freeze on the transfer compare cells"
                             % (args.frozen, BACKENDS[args.backend],
                                frozen.get("backend")))
        for k in ("grid", "tau_max_guide"):
            if frozen.get(k) != getattr(args, k):
                raise SystemExit("frozen strengths were chosen at %s=%r, this "
                                 "run uses %r" % (k, frozen.get(k),
                                                  getattr(args, k)))
        # a strength chosen under one definition of in-band does not carry to a
        # run scored under another
        fmode = frozen.get("delta_mode", "global")
        if fmode != args.delta_mode:
            raise SystemExit("frozen strengths were chosen with delta_mode=%r, "
                             "this run uses %r -- in-band means a different thing "
                             "in each, so the strengths do not transfer"
                             % (fmode, args.delta_mode))
        cells = plan_full_cells(props, arms, frozen, sets, excl)
    elif args.stage == "eqconfirm":
        if not args.frozen:
            raise SystemExit("--stage eqconfirm needs --frozen (the json "
                             "`--stage eqfreeze --json-out` wrote)")
        frozen = load_eq_frozen(args.frozen, props, arms, args.backend)
        for k in ("grid", "tau_max_guide"):
            if frozen.get(k) != getattr(args, k):
                raise SystemExit("frozen strengths were chosen at %s=%r, this "
                                 "run uses %r" % (k, frozen.get(k), getattr(args, k)))
        cells = [(p_, a, DIST_TARGET, float(frozen["frozen_w"][a][p_]))
                 for p_ in props for a in arms]
    elif args.stage == "extend":
        loaded = _load_stage(args, props, "compare", SCREEN_TARGETS,
                             _base_grid(backend_arms(args.backend)))
        if isinstance(loaded, int):
            print("extend needs the finished compare stage")
            return loaded
        cells = [c for c in extension_plan(loaded[0][FREEZE_TARGET], props,
                                           backend_arms(args.backend))
                 if c[1] in arms]
    elif args.stage == "eqextend":
        eq_grid = {("unguided", float(DEFAULT_W))} | {
            (a, float(w)) for a in backend_arms(args.backend) if a != "unguided"
            for w in EQ_STRENGTHS}
        loaded = _load_stage(args, props, "eqtune", (DIST_TARGET,), eq_grid)
        if isinstance(loaded, int):
            print("eqextend needs the finished eqtune stage")
            return loaded
        cells = [c for c in extension_plan(loaded[0][DIST_TARGET], props,
                                           backend_arms(args.backend),
                                           EQ_STRENGTHS, DIST_TARGET)
                 if c[1] in arms]
    elif args.stage == "eqtune":
        cells = plan_eqtune_cells(props, arms)
    else:
        tg = [t for t in args.targets.split(",") if t]
        bad = [t for t in tg if t not in SCREEN_TARGETS]
        if bad:
            raise SystemExit("unknown --targets %s; the screen is %s"
                             % (bad, ",".join(SCREEN_TARGETS)))
        cells = plan_compare_cells(props, arms, tg)

    # Every stage's cells are (prop, arm, tgt, w, t_start) from here down. The
    # pre-basecmp stages take the single global window, so their names and their
    # behaviour are byte-for-byte what they were.
    # v3 is excluded for the same reason basecmp is: plan_v3_cells already
    # returns 5-tuples, carrying v3's own fixed t_start rather than the global
    # --tau-max-guide window.
    if not basecmp and not v3stage:
        cells = with_t(cells, 1.0 - args.tau_max_guide)
    # VALIDATE EVERY PATH, not just --t-starts. t_start also arrives as
    # 1 - tau_max_guide (any pre-basecmp stage), from a refine plan, and from a
    # frozen file. On the flow backend t_start = 0 makes FlowSampler's guidance
    # factor (1 - t)/t divide by zero at the first grid point -- and this file's
    # own header recommends --tau-max-guide 1.0 as an ablation, which is exactly
    # that. FlowSampler defaults t_min_guide to 0.05 for this reason.
    bad_t = sorted({float(c[4]) for c in cells if not 0.0 < float(c[4]) < 1.0})
    if bad_t:
        raise SystemExit(
            "these cells would start guidance at t_start = %s, outside (0, 1): "
            "t is flow time, 0 is pure noise and 1 is data, and guidance runs on "
            "[t_start, 1). On --backend fm, t_start = 0 divides by zero in "
            "FlowSampler's score-to-velocity factor. If this came from "
            "--tau-max-guide, note t_start = 1 - tau_max_guide, so "
            "--tau-max-guide 1.0 is t_start = 0; use 0.99 for a guide-everywhere "
            "ablation." % ", ".join("%g" % t for t in bad_t))

    if args.strengths:
        keep = {float(x) for x in args.strengths.split(",") if x}
        cells = [c for c in cells if c[1] == "unguided" or float(c[3]) in keep]
    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else args.device)
    out = args.out_dir

    if args.preflight:
        args.n, args.batch, args.steps = 8, 8, 10
        seen, first = set(), []
        for c in cells:
            if c[0] != props[0] or c[1] in seen:
                continue
            seen.add(c[1])
            first.append(c)
        cells = first
    def name(c):
        # the window is per cell, so the tag is built per cell: at the basecmp
        # full stage one job writes cells at several windows, because each arm
        # carries its own frozen one
        return cell_name(c[0], c[1], c[2], c[3],
                         cfg=config_tag(args, c[4]), stage=cell_stage)

    todo = cells if args.preflight else [
        c for c in cells if not os.path.exists(os.path.join(out, name(c)))]
    print("cells total %d | done %d | to run %d"
          % (len(cells), len(cells) - len(todo), len(todo)))
    if args.dry_run:
        for c in todo[:20]:
            print("   ", name(c))
        return 0
    if not todo:
        print("nothing to do -- this stage is complete")
        return 0
    os.makedirs(out, exist_ok=True)

    d = torch.load(DATA, weights_only=True)
    types = d["types"]
    if args.backend == "equifm":
        net = EquiFMGenerator(device=dev)
        prov = {"gen": "EquiFM (Song et al. 2023), released EMA",
                "gen_md5": file_md5(str(EQUIFM_WEIGHTS)),
                "gen_args": str(EQUIFM_ARGS),
                "path": "HB_path: coords linear+aligned, types VP (beta 0.1-20)"}
        norm_values, nf, n_layers = net.norm_values, net.args["nf"], net.args["n_layers"]
    elif args.backend == "fm":
        if not os.path.exists(args.fm_ckpt):
            raise SystemExit("--backend fm needs our generator at %s"
                             % args.fm_ckpt)
        net, ck = load_fm(args.fm_ckpt, len(types), dev)
        # OUR sampler works in RAW one-hot: `evaluate_samples`' hot_value
        # defaults to 1.0 in the main sweep and nothing divides the type
        # channels. So the sampler's type divisor is 1, and `build_pair` gets
        # the guide's multiplier (1/8) and the oracle's (1) from that.
        norm_values = (1.0, 1.0, 1.0)
        nf, n_layers = int(ck["args"]["hidden"]), int(ck["args"]["layers"])
        prov = {"gen": "our flow-matching EGNN (this project), EMA",
                "gen_md5": file_md5(args.fm_ckpt),
                "gen_args": os.path.abspath(args.fm_ckpt),
                "epoch": ck.get("epoch"),
                "path": "linear flow path, independent Gaussian noise"}
    else:
        net, prov = load_backend(args.edm_dir, dev)
        prov["gen_md5"] = prov["edm_md5"]
        prov.update({"noise_schedule": net.args["diffusion_noise_schedule"],
                     "diffusion_steps": net.args["diffusion_steps"]})
        norm_values, nf, n_layers = net.norm_values, net.args["nf"], net.args["n_layers"]
    prov.update({"fm_md5": prov["gen_md5"],   # the generator key full_run_table checks
                 "torch": torch.__version__, "cuda": torch.version.cuda,
                 "device": (torch.cuda.get_device_name(0)
                            if dev == "cuda" and torch.cuda.is_available()
                            else "cpu"),
                 "norm_values": list(norm_values)})
    if frozen is not None:
        prov.update({"frozen_path": os.path.abspath(args.frozen),
                     "frozen_md5": file_md5(args.frozen),
                     "frozen_source_stage": frozen.get("source_stage"),
                     "frozen_source_target": frozen.get("target"),
                     "frozen_source_seed": frozen.get("source_seed"),
                     # basecmpfull's sets are its own two, not FR3a/FR3
                     "frozen_sets": (list(BASECMP_SETS)
                                     if args.stage == "basecmpfull" else args.sets),
                     "frozen_exclude_sets": args.exclude_sets})
    print("base model: %s  md5 %s  nf=%d layers=%d  grid=%s"
          % (BACKENDS[args.backend], prov["gen_md5"][:12], nf, n_layers,
             args.grid))

    calib_sel = calibration_indices(d, args.n_calib)
    sampler_scale = float(norm_values[1])
    if args.backend == "edm":
        assert sampler_scale == generator_feat_scale(args.edm_dir)
    # WHICH PAIR: declared per backend in V3_BACKENDS, not decided here.
    # `--pair` overrides it for a one-off, and is recorded on every cell.
    pair = args.pair or backend_pair(args.backend)
    if pair not in ("ours", "tfg"):
        raise SystemExit("--pair must be 'ours' or 'tfg', got %r" % pair)
    _build = build_pair_ours if pair == "ours" else build_pair
    print("property pair: %s (%s)"
          % (pair, "weights/f_{A,B}_<p>.pt" if pair == "ours"
             else "TFG tf_predict_<p> + evaluate_<p>"))
    guides, evals, deltas, reports = {}, {}, {}, {}
    for prop in props:
        f_A, f_B, delta, rep = _build(prop, d, calib_sel, dev, args.k_delta,
                                      sampler_scale,
                                      delta_mode=args.delta_mode)
        rep.setdefault("pair", pair)
        guides[prop], evals[prop] = f_A, f_B
        deltas[prop], reports[prop] = delta, rep
        print("  %-6s guide MAE %.5f %s | oracle MAE %.5f | delta %.5f | "
              "scale check %.3f (guide) %.3f (oracle)"
              % (prop, rep["guide_mae"], PROP_UNITS[prop], rep["oracle_mae"],
                 delta,
                 rep.get("slope_over_mad_guide",
                         rep.get("fitted_slope_guide", float("nan"))),
                 rep.get("slope_over_mad_oracle",
                         rep.get("fitted_slope_oracle", float("nan")))))
    oracle2 = {}
    # THE SECOND ORACLE runs for every backend in the v3 registry, not for a
    # hardcoded pair of names. A second opinion present for one generator and
    # absent for another would be a difference between those columns that has
    # nothing to do with the generators -- which is the whole reason the
    # comment below gives for running it on both.
    if args.backend in V3_BACKENDS:
        # A SECOND oracle: OC-Flow's clean EGNN, different weights, same
        # (first) half as evaluate_<p>, disjoint from the guide. Calibrated
        # exactly like the first, on the same molecules. Supplementary: it
        # tells whether a gain is a quirk of one network.
        # It is on for BOTH comparison backends, not just EquiFM: a second
        # opinion that existed for one generator and not the other would be a
        # difference between the two columns that has nothing to do with them.
        idx_cal = calib_sel
        for prop in props:
            pi = PROP_INDEX[prop]
            c = d["coords"][idx_cal].to(dev)
            f = d["feats"][idx_cal].to(dev)
            m = d["mask"][idx_cal].to(dev)
            raw2 = OCFlowOracle(prop, device=dev)
            a2, b2, mae2 = fit_calibration(raw2, c, f, m, d["y"][idx_cal, pi].to(dev))
            if not 0.9 < a2 / QM9_MAD[prop] < 1.1:
                raise SystemExit("%s OC-Flow oracle slope/MAD %.3f" % (prop, a2 / QM9_MAD[prop]))
            f_B2 = Calibrated(raw2, a2, b2, prop, guides[prop].y_std,
                              feats_are_normalised=False, feat_scale=sampler_scale)
            # The second oracle's band is set the SAME way as the first's. It is a
            # second opinion on the in-band number, so scoring it against a
            # differently-defined band would make the two incomparable by
            # construction -- the one thing a second opinion must not be.
            if args.delta_mode == "local":
                delta2, drep2 = local_delta(prop, d, raw2, a2, b2, dev,
                                            k=args.k_delta)
            else:
                delta2, drep2 = choose_delta(mae2, args.k_delta), {"rule": "global"}
            oracle2[prop] = (f_B2, delta2,
                             {"oracle2": "OC-Flow exp_class_%s/best_checkpoint.npy" % prop,
                              "slope": a2, "intercept": b2, "mae": mae2,
                              "delta_mode": args.delta_mode,
                              "delta_detail": drep2})
            print("  %-6s oracle2 (OC-Flow) MAE %.5f | delta2 %.5f (%s)"
                  % (prop, mae2, delta2, args.delta_mode))

    # q50/q90: sizes from val[:n], exactly as the main sweep's compare stage.
    # dist: sizes AND targets from the SAME test molecules test[:n], exactly
    # as the main sweep's full stage (see guidance_sweep.run_cell for why the
    # two must come from one molecule).
    va = d["split"]["val"][: args.n]
    te = d["split"]["test"][args.block_start: args.block_start + args.n]
    if any(c[2] == DIST_TARGET for c in todo) and te.numel() < args.n:
        raise SystemExit("test[%d:%d] has only %d molecules" % (
            args.block_start, args.block_start + args.n, te.numel()))
    mask_v, mask_t = d["mask"][va].to(dev), d["mask"][te].to(dev)
    clip = None if args.clip < 0 else args.clip

    def basecmp_sets_of(arm, prop, w, t_start):
        """Which of ("floor", "free") this (w, t_start) is the pick for, or None
        outside the basecmpfull stage."""
        if args.stage != "basecmpfull" or frozen is None:
            return None
        hit = []
        for s in BASECMP_SETS:
            pk = frozen["frozen"].get(s, {}).get(arm, {}).get(prop)
            if arm == "unguided":
                hit.append(s)
            elif pk and abs(float(pk["w"]) - float(w)) < 1e-12 \
                    and abs(float(pk["t_start"]) - float(t_start)) < 1e-12:
                hit.append(s)
        return hit

    def run_cell(prop, arm, tgt, w, t_start):
        f_A, f_B = guides[prop], evals[prop]
        delta = deltas[prop]
        extra = arm_kwargs(arm, delta)
        if arm == "tfg":
            # TFG's published QM9 configuration; its energy normaliser is
            # TFG's own MAD, which build_pair checks the guide's calibration
            # slope against (0.9-1.1x), so this is TFG's energy exactly
            extra["tfg"] = tfg_config(prop, QM9_MAD[prop])
        s = f_A.y_std
        w_scale = strength_scale(arm, extra, s)
        w_applied = w * w_scale
        if arm in SHG_SCHEDULES:
            # per-phase btvg normalisation (finding S1)
            extra["schedule"] = scale_schedule(extra["schedule"], s,
                                               extra.get("tau"))
        if tgt == DIST_TARGET:
            mask_c, idx_c = mask_t, te
            y_t = d["y"][te, PROP_INDEX[prop]].to(dev).float()
            target = float(y_t.mean())
        else:
            mask_c, idx_c = mask_v, va
            target = TARGETS[prop][tgt]
            y_t = torch.full((args.n,), target, device=dev)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        # BDG's setpoint in property units, now that `s` is known. tau is a
        # fraction of the GUIDE's output scale, so it is the same physical
        # request on either backend even though the two generators have
        # different state scales. Popped before `extra` reaches the sampler,
        # which knows `bdg_tau` and not `_bdg_tau_mult`.
        if "_bdg_tau_mult" in extra:
            extra["bdg_tau"] = float(extra.pop("_bdg_tau_mult")) * float(s)

        cs, fs, calls = [], [], 0
        cost = {k: 0 for k in ("gen_fwd", "gen_vjp", "gen_jvp",
                               "guide_fwd", "guide_bwd", "guide_hvp")}
        clipped, guided, diag_log, sched_log = 0, 0, {}, {}
        for i in range(0, args.n, args.batch):
            m = mask_c[i:i + args.batch]
            skw = dict(f_net=(None if arm == "unguided" else f_A),
                       y=y_t[i:i + m.shape[0]], s=s, mode=base_mode(arm),
                       w=w_applied,
                       clip=clip, n_probe=args.n_probe, n_mc=args.n_mc,
                       sigma_mc=args.sigma_mc, **extra)
            # ONE window, expressed on each family's own clock. Flow time runs
            # 0 (noise) -> 1 (data) and guidance is on for t >= t_min_guide; VP
            # and EquiFM time runs 1 (noise) -> 0 (data) and guidance is on for
            # tau <= tau_max_guide. The same physical instant is therefore
            # t_start on one clock and 1 - t_start on the other, which is the
            # mirror `FlowSampler`/`VPSampler` document. Getting this backwards
            # would guide the wrong half of the trajectory on one base and
            # nothing would raise -- the arm would just be a worse method
            # wearing its name, and the base-model comparison would read it as
            # a property of the generator.
            if args.backend == "equifm":
                c0, f0 = initial_noise(m, 6, gen)     # 5 one-hot + charge
                smp = EquiFMSampler(net, m, tau_max_guide=1.0 - t_start, **skw)
            elif args.backend == "fm":
                c0, f0 = initial_noise(m, len(types), gen)
                smp = FlowSampler(net, m, t_min_guide=t_start, **skw)
            else:
                c0, f0 = initial_noise(m, len(types), gen)
                smp = VPSampler(
                    net, m, tau_min=args.tau_min, noise_schedule=net.schedule,
                    tau_max_guide=1.0 - t_start, grid=args.grid, **skw)
            c, f, nc = integrate(smp, c0, f0, args.steps, args.solver)
            cs.append(c)
            fs.append(f)
            calls += nc
            for k in cost:
                cost[k] += getattr(smp.cost, k)
            clipped += smp.n_clipped
            guided += smp.n_guided
            for mk, mv in smp.schedule_log.items():
                sched_log[mk] = sched_log.get(mk, 0) + mv
            for dk, dv in smp.diag_summary().items():
                tot, cnt = diag_log.get(dk, (0.0, 0))
                diag_log[dk] = (tot + dv, cnt + 1)

        # Scored in the SAMPLER's space, which is EDM's normalised space, and
        # that is correct for all three consumers:
        #   * coordinates -- norm_values[0] is 1 for EDM's QM9 recipe, asserted
        #     at load time, so normalised coordinates ARE angstroms and the
        #     bond-distance tables apply unchanged;
        #   * atom types -- `stability` and `to_smiles` read the element by
        #     argmax over the feature channels, which a common positive scale
        #     cannot change;
        #   * f_A and f_B -- `Calibrated` takes sampler-space features by
        #     contract and each rescales to what its own network wants.
        # Denormalising here instead would silently feed f_A features 8x too
        # large, and its predictions would be wrong while nothing raised.
        C, F = torch.cat(cs), torch.cat(fs)
        if args.backend == "equifm":
            # the charge channel is not an atom type: every consumer below
            # (stability, SMILES, f_A, f_B, the decoded view) reads types only
            F = F[..., :5]
        # hot_value: the decoded-type view must be in the SAMPLER's space too
        # (one-hot / sampler_scale), because that is what f_A/f_B take
        r = evaluate_samples(C, F, mask_c, types, f_A, f_B, y_t, delta,
                             per_mol=args.per_mol,
                             hot_value=1.0 / sampler_scale)
        r.pop("delta", None)
        if args.per_mol:
            # which real molecule each row took its size (and, for dist, its
            # target) from -- the key that pairs rows across arms and seeds
            r["_per_mol"]["mol_idx"] = idx_c.clone().cpu()
        if prop in oracle2:
            from evaluation import decode_types, _chunked
            f_B2, delta2, rep2 = oracle2[prop]
            ok = torch.isfinite(C).all((1, 2)) & torch.isfinite(F).all((1, 2))
            with torch.no_grad():
                b2 = _chunked(f_B2, C, F, mask_c)
                b2d = _chunked(f_B2, C, decode_types(F, mask_c, 1.0 / sampler_scale),
                               mask_c)
            e2, e2d = (b2 - y_t).abs(), (b2d - y_t).abs()
            r["oracle2"] = dict(rep2, delta=delta2,
                                prop_mae=float(e2[ok].mean()),
                                prop_mae_dec=float(e2d[ok].mean()),
                                in_band=float(((e2 <= delta2) & ok).float().mean()),
                                in_band_dec=float(((e2d <= delta2) & ok).float().mean()))
            if args.per_mol:
                r["_per_mol"]["f_B2"] = b2.detach().float().cpu()
                r["_per_mol"]["f_B2_dec"] = b2d.detach().float().cpu()
        r.update({
            "prop": prop, "arm": arm, "target_name": tgt, "target": target,
            "stage": cell_stage, "variant": args.stage,
            "study": ("eqchem" if args.stage in ("eqtune", "eqextend", "eqconfirm")
                      else "basecmp" if basecmp else "headline"),
            "extension": args.stage in ("extend", "eqextend", "basecmprefine"),
            "dist_block": ([args.block_start, args.block_start + args.n]
                           if tgt == DIST_TARGET else None),
            "arm_class": ARM_CLASS.get(arm, "?"),
            "backend": BACKENDS[args.backend], "prov": prov,
            "guide": reports[prop]["guide"], "oracle": reports[prop]["oracle"],
            "calibration": reports[prop],
            "w": w, "w_applied": w_applied, "w_scale": w_scale,
            "n": args.n, "steps": args.steps,
            "solver": args.solver, "grid": args.grid,
            "delta": delta, "mae_B": reports[prop]["oracle_mae"],
            "n_probe": args.n_probe, "n_mc": args.n_mc, "sigma_mc": args.sigma_mc,
            "clip": args.clip, "k_delta": args.k_delta, "seed": args.seed,
            "guide_time": "zero", "tau_min": args.tau_min,
            # the window, per cell, on BOTH clocks and as the screened axis.
            # `t_start` is the primary record: it is the one number that means
            # the same instant on a flow base and a VP/EquiFM base.
            "t_start": float(t_start),
            "tau_max_guide": 1.0 - float(t_start),
            "t_min_guide": float(t_start),
            # WHICH STRENGTH SET(S) this cell is. A cell can be both: the two
            # picks coincide whenever an arm never threatens the chemistry floor,
            # and those cells are computed once. Without this the mapping from a
            # cell back to its set lives only in the frozen json, and any reader
            # would have to re-derive it by matching floats.
            "basecmp_set": basecmp_sets_of(arm, prop, w, t_start),
            # WHICH PROPERTY PAIR SCORED THIS CELL. delta = k x MAE(f_B), so
            # the pair sets the band width: two cells with different `pair`
            # values have different in_band definitions and may not be
            # compared on that column. v3_table refuses to mix them.
            "pair": pair,
            # the generator's human label. It used to default to "EDMsecond"
            # for anything not in a two-name dict, so ANY new backend stamped
            # the wrong generator onto every cell it wrote. BACKENDS is the
            # one table that must know all of them.
            "fm": BACKENDS[args.backend],
            "cost": cost, "field_evals": calls,
            "guided_steps": guided, "clipped_sample_steps": clipped,
            "schedule_used": sched_log,
            "diag": {k: tot / cnt for k, (tot, cnt) in diag_log.items()},
            "batch": args.batch})
        if arm == "tfg":
            r["tfg_config"] = extra["tfg"]
        return r

    t0, done, failed = time.time(), 0, []
    for c in todo:
        if args.max_minutes and done:
            used = (time.time() - t0) / 60.0
            if used + used / done > args.max_minutes:
                print("  time guard: %.1f min used, stopping with %d cells left"
                      % (used, len(todo) - done))
                break
        cs_t = time.time()
        nm = name(c)
        try:
            r = run_cell(*c)
        except Exception as exc:                       # noqa: BLE001
            failed.append((nm, repr(exc)))
            print("  FAILED %s: %r" % (nm, exc))
            if not args.preflight:
                with open(os.path.join(out, nm + ".failed"), "w") as fh:
                    json.dump({"cell": nm, "error": repr(exc)}, fh)
            continue
        per_mol = r.pop("_per_mol", None)
        r["seconds"] = time.time() - cs_t
        if not args.preflight:
            # The temporary paths carry the PID. Two array tasks are never
            # planned the same cell (the basecmp tiling is gated), but if one
            # ever were -- a stage resubmitted with overlapping --arms, or a
            # sweeper started beside a live array -- a shared tmp path lets two
            # writers interleave and then atomically publish a cell with
            # embedded NULs. Resume is `os.path.exists` alone, so nothing would
            # ever retry it and the corrupt cell would be read as data.
            # (Found and fixed on the guidance_sweep side by project 1 main 2,
            # 25 Sep; the same hazard is here.)
            pid = os.getpid()
            if per_mol is not None:
                # the sidecar lands BEFORE the json: the json is the
                # completion marker, so a cell is never "done" without it
                ptmp = os.path.join(out, "%s.permol.%d.tmp" % (nm, pid))
                torch.save(per_mol, ptmp)
                os.replace(ptmp, os.path.join(out, nm[:-5] + ".permol.pt"))
            tmp = os.path.join(out, "%s.%d.tmp" % (nm, pid))
            with open(tmp, "w") as fh:
                json.dump(r, fh, indent=1)
            os.replace(tmp, os.path.join(out, nm))
            stale = os.path.join(out, nm + ".failed")
            if os.path.exists(stale):
                os.remove(stale)
        done += 1
        print("  [%3d/%3d] %-6s %-9s %-4s w=%-5g t=%-4g MAE %9.4f (%5.2f d)  "
              "in-band %.3f (dec %.3f)  mol-stab %.3f  clipped %d  %.1fs"
              % (done, len(todo), c[0], c[1], c[2], c[3], c[4],
                 r["prop_mae_eval"], r["prop_mae_eval"] / r["delta"],
                 r["in_band_fraction"], r.get("in_band_fraction_dec", float("nan")),
                 r["mol_stability"], r["clipped_sample_steps"], r["seconds"]))

    if failed:
        print("\n%d cell(s) failed:" % len(failed))
        for nm, exc in failed:
            print("   %s  %s" % (nm, exc))
        return 1
    if args.preflight:
        print("\npreflight OK: %d arms ran end to end, nothing written" % done)
        return 0
    left = [c for c in cells if not os.path.exists(os.path.join(out, name(c)))]
    print("\n%d cells this run, %.1f min; %s"
          % (done, (time.time() - t0) / 60.0,
             "STAGE COMPLETE" if not left else "%d cells remain" % len(left)))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
