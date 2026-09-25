"""Stage 1 guidance screening sweep -- resumable, one JSON per cell.

    python proj1/scripts/guidance_sweep.py --n 512 --max-minutes 225

Runs the (arm x property x target x strength x guidance-window) grid at the
screening sample size and writes ONE json per cell to results/sweep/. A cell
that already has a json is skipped, so the script is safe to run under a
wall-clock cap and re-submit: it picks up exactly where it stopped. That is the
whole design requirement, because the cluster kills jobs at 4 hours.

CELL ORDER IS DELIBERATE. Cells are emitted so that a PARTIAL run is still a
complete experiment:

  pass 1   every arm, every property, at the median target, default strength
           and the widest window   -> a full arm comparison
  pass 2   the strength sweep at the median target
  pass 3   the guidance-window sweep
  pass 4   the remaining targets

So if the queue dies a third of the way through there is still a publishable
arm comparison, rather than three properties finished and eight arms missing.

WHAT EACH CELL MEASURES. Property MAE against f_B (never f_A -- that is the
guide, and scoring with it is circular), fraction within delta = 2 x f_B MAE,
atom/molecule stability, validity, uniqueness, and the measured generator and
guide call counts so "at matched compute" is measured rather than asserted.

The unguided arm is run once per property and reused: it does not depend on
strength, window or arm, and re-running it 135 times would waste an hour.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from evaluation import choose_delta, evaluate_samples  # noqa: E402
from checkpoint_paths import (default_generator, describe,  # noqa: E402
                              find_predictor, require_predictor)
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from guidance import Cost, ResidualCalibrationHead  # noqa: E402
from sampling import FlowSampler, VPSampler, initial_noise, integrate  # noqa: E402


def file_md5(path, chunk=1 << 20):
    """Hash of the generator checkpoint, recorded in every cell.

    Cells produced on different machines are pooled into one table, and the
    only provenance previously recorded was `os.path.basename(args.fm)` --
    which is "fm_last.pt" on both the cluster and locally, so a checkpoint
    mismatch would have been invisible. One hash per run, not per cell.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_rch(path, dev, target="residual"):
    """Haimo v1's fitted head, restored exactly as fit_rch.py saved it.

    fit_rch.py stores TWO fits, not a state_dict:

      residual  beta fitted to c = 1/2 tr(H Sigma), the quantity SMG computes.
                This is Haimo v1 as specified, and the arm that can deflate the
                SMG claim: if a ridge lookup reproduces c, SMG's per-state
                JVP/HVP is not earning its cost.
      direct    beta fitted to f_A(x_1) - f_A(m), the realised endpoint gap.
                Measured held-out R^2 0.755 against residual's 0.017, so this
                is the variant that can actually work -- and it is a different
                claim, which is why both are run as separate cells rather than
                one being quietly substituted for the other.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if target not in ck:
        raise RuntimeError("%s has no %r fit (has %s)"
                           % (path, target, sorted(k for k in ck if isinstance(ck[k], dict))))
    beta = ck[target]["beta"]
    head = ResidualCalibrationHead(n_feat=beta.numel())
    head.beta.copy_(beta.to(head.beta.dtype))
    head.fitted = True
    head.r2_test = ck[target]["r2_test"]
    return head.to(dev).eval()

class MissingState(RuntimeError):
    """An arm's prerequisite file does not exist yet.

    Distinct from every other error on purpose: this one means ONE arm cannot
    run, not that the job is broken. Preflight skips it; a real run records a
    .failed sidecar so the cell is retried when the file appears.
    """


DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT = os.path.join(ROOT, "proj1", "checkpoints")   # legacy; see checkpoint_paths
OUT = os.path.join(ROOT, "results", "sweep")

# measured on the target distribution of train_a; see GUIDANCE_EXPERIMENT_PLAN.md
TARGETS = {"mu":    {"q50": 2.4932, "q90": 4.6627},
           "alpha": {"q50": 75.54, "q90": 85.05},
           "gap":   {"q50": 0.2496, "q90": 0.3162}}

# "dist" is NOT a fixed value and so is not in TARGETS: each molecule's target
# is the real property of the held-out molecule whose size it takes, resolved
# per cell in run_cell. This is the EDM/EEGSDE/TFG conditional protocol, and it
# is the only target that involves no choice by us -- which is why FR6 makes it
# the headline. Measured steering distance from the unguided generator's own
# mean: dist 0.80-0.85 sd, against q50 0.05-0.27 and q90 1.1-1.6.
DIST_TARGET = "dist"

# every arm that can currently run end to end. `band` and `rch` need extra
# state and are included only when that state exists.
# The nine arms verified end to end. `band` and `rch` are implemented and
# unit-tested but NOT in the queued sweep: band's diversity direction was
# degenerate until PhysicalProperty.embed was added (it reported an identical
# number at every strength, which is how that was caught) and rch needs a
# fitted head per property. Add them once each has been shown to vary with its
# own hyperparameters.
ARMS = ["unguided", "plug", "tmpd", "tfg_mc", "lgd_mc", "osc",
        "smg", "smg2", "smg2_curv"]

# ---- stage v2 -------------------------------------------------------------
# Arms that were implemented and gated but never queued, plus the three new
# ones. Kept in a separate list so `--stage main` reproduces the v1 sweep
# byte-for-byte and the finished cells are never invalidated.
#
#   smg_mean  completes the ablation ladder: mean-only / var-only / both
#   rch       Haimo v1 -- the residual calibration head. Needs a fitted head
#             per property (checkpoints/rch_<prop>.pt); skipped where absent.
#   band      the tolerance-band step
#   spbc      shape-preserving bias correction   (v2)
#   btvg*     band-targeted variance guidance    (v2)
V1_MISSING = ["smg_mean", "rch", "band"]
# btvg_mean is NOT here: after the (tau/s)^2 normalisation its coefficient is
# w(y-f)/s^2, which is `plug` -- bit-identical at 9.0e-8 relative. That is a
# fact worth stating in the paper (BTVG's novelty is entirely in the variance
# term) and 30 cells not worth spending. It remains implemented and gated.
V2_ARMS = ["spbc", "btvg", "btvg_var"]

# ---- the finalised comparison set -----------------------------------------
# See this file's header in results/bench/add_compare_stage.py for why each
# member is in and why tfg_mc / osc / rch are out.
# OURS is IN this list on purpose. The stage-v2 cells cannot be pooled with a
# compare table: btvg has 9 strengths at q50 and only ONE at q90, against the
# 7-point grid here -- a best-of-9 advantage at one target and a badly
# under-tuned arm at the other, biased in opposite directions. The only way to
# read "how does btvg compare" off one table is to run it in that table.
#
# btvg_var is here too because it is btvg's variance-only rung, and `plug` is
# already its mean-only rung (bit-identical, 9e-8). So this single stage
# carries the full 2x2 ablation AND the external comparison, on one grid.
#
# `tfg` REPLACED `dflow` (Henry, 23 Sep, AFTER the full run was read -- so
# every tfg number is post hoc and is labelled so). dflow was dropped for
# cost (104.5 min per n=5000 cell against btvg's 31) and because it cannot
# run on a diffusion model; its 42 compare cells stay on disk and reportable,
# and `--arms dflow` still runs it. The assignment needs three recent
# external methods at the same protocol: tmpd, lgd_mc and now tfg. `plug` is
# btvg's own mean term, an ablation rung, not one of the three.
COMPARE_SET = ["unguided", "plug", "tmpd", "lgd_mc", "tfg",
               "btvg", "btvg_var"]

# TFG (Ye et al., NeurIPS 2024), full update at N_recur = 1 -- NOT `tfg_mc`,
# which is its smoothing ingredient alone. (rho_bar, mu_bar, gamma_bar) are
# TFG's OWN published QM9 values (App. E.3, Table 11, the TFG columns; gap is
# the "Delta" row). The sweep's `w` multiplies (rho_bar, mu_bar), so w = 1 is
# TFG's configuration and the shared 7-point grid brackets it 100x below and
# 4x above; gamma_bar, the iterations and the schedules are fixed.
#   n_iter 4, eps_bsz 1   paper section 5.1; eps_bsz > 1 crashes TFG's
#                         molecule code (issue #10: "should always be 1")
#   rho/mu "increase"     paper section 5.1 and utils/configs.py. The public
#                         QM9 script says mu "decrease", which puts ~90% of mu
#                         in the steps our t_min = 0.5 window switches off.
#   sigma "decrease"      std = gamma_bar sqrt(1 - abar_t), script + configs
#   clip_scale 100        utils/configs.py default (rescale_grad)
# Energy: TFG's -((f - y)/mad)^2 with mad over the guide's own training split.
TFG_QM9 = {"alpha": (0.016, 0.001, 0.0001),
           "mu":    (0.001, 0.002, 0.1),
           "gap":   (0.032, 0.001, 0.001)}
TFG_FIXED = {"n_iter": 4, "eps_bsz": 1, "rho_schedule": "increase",
             "mu_schedule": "increase", "sigma_schedule": "decrease",
             "clip_scale": 100.0}


def tfg_config(prop, mad):
    """The sampler's `tfg=` dict for one property: Table 11 plus the fixed
    settings, and the MAD that puts TFG's energy in this guide's units."""
    if prop not in TFG_QM9:
        raise KeyError("no TFG QM9 configuration for property %r" % prop)
    rho, mu, gamma = TFG_QM9[prop]
    return dict(rho=rho, mu=mu, gamma=gamma, mad=float(mad), **TFG_FIXED)


# D-Flow is not a `mode`: it replaces the sampler rather than adding a field.
# `w` scales its learning rate, the way `w` scales the field elsewhere, so the
# strength axis means the same thing. n_iter is FIXED so a strength sweep does
# not quietly become a compute sweep.
DFLOW_LR = 0.05
DFLOW_ITERS = 8

# REFERENCES. Every v2 comparison re-runs these in the same cells so the
# contrast is measured under identical seeds, targets and windows rather than
# read off a table from another job.
V2_REFERENCES = ["unguided", "plug", "tmpd", "smg"]

# BTVG's tolerance. tau is NOT delta: a centred Gaussian with sd = delta gives
# only 68.3% band coverage, and 95% needs sd ~ delta/1.96. The multiplier
# brackets that choice so the pre-registered value can be checked rather than
# assumed.
# Retain the historical frozen value. After (tau/s)^2 normalization the
# coefficient is -0.5w/s^2 * (1 - tau^2/V)_+ for positive V. Early screen
# measurements were nearly saturated, but do not establish tau-invariance
# over a trajectory: audit_btvg_tolerance.py records threshold activation
# late in sampling. A future tau intervention needs its own outcome protocol;
# it must not silently change this historical grid.
TAU_MULT = [1.0]                    # x (delta / 1.96)

# Arms whose strength curve reads the CLIP rather than the arm on the standard
# grid. Measured clip fraction of guided sample-steps at t_min=0.5, w=0.01..4:
#   band 0.00 0.22 0.71 0.89 0.97 1.00 1.00
#   btvg 0.08 0.27 0.46 0.52 0.62 0.74 0.78
#   spbc 0.00 ...................... 0.00 (through w=2)
# Only the two lowest points are informative for band/btvg, so they get two
# extra points below the shared grid.
CLIP_BOUND = ("band", "btvg", "btvg_var")
STRENGTHS_LOW = [0.002, 0.005]

# SPBC's trust radius. eta is deliberately NOT swept: it multiplies the edit
# linearly and so does w, which would make the grid redundant -- the same
# mistake the plug-in s-sweep made (memo section 3.1).
SPBC_RADIUS = [-1.0, 0.5]           # -1 = unbounded

# SHG schedules: [(t_lo, t_hi, mode, w_multiplier), ...]. The first interval
# containing t wins; outside every interval guidance is OFF.
#
# The bands come from two measurements, not from taste:
#   - the window effect is U-shaped and large (alpha MAE 10.02 -> 5.32 -> 6.59
#     as t_min_guide goes 0.05 -> 0.5 -> 0.75), so nothing starts below 0.5
#   - the quadratic closure is invalid below t ~ 0.75
#     (FINDING_QUADRATIC_CLOSURE_VALIDITY.md), so the curvature-aware arms are
#     only ever scheduled ABOVE that boundary
SHG_SCHEDULES = {
    # cheap and well-posed early, shape-preserving centring at the end
    "shg_plug_spbc": [(0.50, 0.85, "plug", 1.0), (0.85, 1.0, "spbc", 1.0)],
    # curvature-aware where the closure holds, then centre
    "shg_smg_spbc": [(0.50, 0.85, "smg", 1.0), (0.85, 1.0, "spbc", 1.0)],
    # get the mean close, then concentrate into the band
    "shg_plug_btvg": [(0.50, 0.80, "plug", 1.0), (0.80, 1.0, "btvg", 1.0)],
    # three-phase: approach, correct curvature, then centre
    "shg_three": [(0.50, 0.75, "plug", 1.0), (0.75, 0.90, "smg", 1.0),
                  (0.90, 1.00, "spbc", 1.0)],
}
SHG_ARMS = sorted(SHG_SCHEDULES)
# THE UNION OF BOTH GRIDS RUN SO FAR, deliberately.
# The cluster ran {0.25, 0.5, 1, 2, 4}; this list was later narrowed to reach
# the low end plug-in needs (~0.003-0.01; at 0.25 it is already heavily
# clip-saturated). Neither grid contains the other, which orphaned 42 finished
# cells and meant any newly-run arm could not be compared against the eight
# already on disk at the same strengths. Keeping the union means the orphans
# stay in the plan and every arm converges to the same 7-point curve.
STRENGTHS = [0.01, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0]
WINDOWS = [0.05, 0.5, 0.75]          # t_min_guide: guidance is skipped below this
DEFAULT_W, DEFAULT_WIN = 1.0, 0.05

# The window stage v2 screens at. Chosen because it won a three-point sweep
# {0.05, 0.5, 0.75} ON THE V1 ARMS -- an empirical choice on different arms,
# not a derived optimum. Theory (k = (1-t)^2/t = 18.05 at t=0.05 and 0.083 at
# t=0.75) says the optimum is interior, and no window between 0.05 and 0.5 has
# ever been measured.
V2_WIN = 0.5


def cell_name(prop, arm, tgt, w, win, variant="-"):
    """Filename for one cell.

    variant == "-" reproduces the v1 name exactly, so the cells already on disk
    stay valid and are never recomputed.
    """
    base = "%s__%s__%s__w%g__tmin%g" % (prop, arm, tgt, w, win)
    if variant and variant != "-":
        base += "__" + variant
    return base + ".json"


def arm_kwargs(arm, variant, delta):
    """Sampler kwargs implied by (arm, variant). One place, so the sweep and
    the final benchmark cannot drift apart.

    A variant of "-" means "the pre-registered default". It used to raise
    IndexError here, which `--stage main --arms spbc` could reach and which the
    cell-level try/except turned into a .failed cell retried forever.
    """
    kw = {}
    v = variant or "-"
    if arm == "rch":
        pass                       # handled in run_cell, which has the device
    elif arm.startswith("btvg"):
        mult = float(v.split("tau")[1]) if "tau" in v else 1.0
        kw["tau"] = mult * delta / 1.96
    elif arm == "spbc":
        r = float(v.split("r")[1]) if v.startswith("r") else -1.0
        kw["spbc_radius"] = None if r < 0 else r
    elif arm in SHG_SCHEDULES:
        kw["schedule"] = SHG_SCHEDULES[arm]
        # any mode the schedule can select must have its own hyperparameters
        # available, because `active()` can hand control to it at any step
        kw["tau"] = delta / 1.96
    return kw


def scale_schedule(sched, s, tau):
    """Apply the btvg strength normalisation PER PHASE inside an SHG schedule.

    `strength_scale` below normalises a whole arm by (tau/s)^2 because btvg's
    mean coefficient carries 1/tau^2 where every plug-family arm carries 1/s^2.
    An SHG schedule mixes both kinds of phase, so one scalar cannot serve both:
    scaling the whole arm would cripple its plug phase, and not scaling it ran
    the btvg phase 3139x too strong (measured on alpha). The per-phase
    multiplier `active()` already applies is the right place for it.
    """
    out = []
    for (lo, hi, mode, mult) in sched:
        if mode.startswith("btvg") and tau:
            mult = mult * (float(tau) / float(s)) ** 2
        out.append((lo, hi, mode, mult))
    return out


def strength_scale(arm, kw, s):
    """Make `w` mean the same thing across arms.

    BTVG's mean term carries 1/tau^2 where every plug-family arm carries 1/s^2,
    and s is the guide's output scale while tau is a band half-width: the
    measured ratio s^2/tau^2 is 322 (mu), 1115 (alpha), 150 (gap). A shared
    strength grid therefore places btvg two to three orders of magnitude above
    plug at the same nominal w, and the whole grid lands clip-saturated -- the
    measured field-to-velocity ratio ran 30-660 against a clip of 1.0. Dividing
    by that ratio restores the comparison; the raw and applied strengths are
    both recorded so nothing is hidden.
    """
    # btvg2 is excluded: its mean term IS lgd_mc's and its variance
    # coefficient is already written in plug units (1/s^2), so w means the
    # same thing for btvg2 as for lgd_mc with no rescaling.
    if arm.startswith("btvg") and not arm.startswith("btvg2") and kw.get("tau"):
        return (kw["tau"] / float(s)) ** 2
    return 1.0


def plan_cells(props, arms):
    """Ordered so a partial run is still a complete experiment. See docstring."""
    seen, cells = set(), []

    def add(prop, arm, tgt, w, win, variant="-"):
        key = (prop, arm, tgt, w, win, variant)
        if key not in seen:
            seen.add(key)
            cells.append(key)

    for prop in props:                       # pass 1: the arm comparison
        for arm in arms:
            add(prop, arm, "q50", DEFAULT_W, DEFAULT_WIN)

    def expand(arm):
        """unguided has no strength or window, so it gets one cell per target."""
        return arm != "unguided"
    for prop in props:                       # pass 2: strength
        for arm in arms:
            if not expand(arm):
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, DEFAULT_WIN)
    for prop in props:                       # pass 3: guidance window
        for arm in arms:
            if not expand(arm):
                continue
            for win in WINDOWS:
                add(prop, arm, "q50", DEFAULT_W, win)
    for prop in props:                       # pass 4: the other targets
        for arm in arms:
            for tgt in TARGETS[prop]:
                if tgt == "q50":
                    continue
                if not expand(arm):
                    add(prop, arm, tgt, DEFAULT_W, DEFAULT_WIN)
                    continue
                for w in STRENGTHS:
                    add(prop, arm, tgt, w, DEFAULT_WIN)

    # pass 5: THE MISSING CROSS. Passes 2 and 3 form a cross, not a grid --
    # strengths are swept at DEFAULT_WIN (0.05) and windows at DEFAULT_W (1.0).
    # So at t_min = 0.5, where every arm performs best on alpha, the v1 arms
    # have exactly ONE cell (w=1) while the stage-v2 arms got a full 7-point
    # strength sweep there. Any "best strength" table built on that compares
    # tuned arms against untuned ones, in our own favour.
    #
    # Only the arms that stage v2 does NOT already sweep at V2_WIN need this;
    # plug/tmpd/smg get it as v2 references.
    for prop in props:
        for arm in arms:
            if arm in V2_REFERENCES or not expand(arm):
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, V2_WIN)
    return cells


def plan_v2_cells(props, arms):
    """Stage-2 plan: the new arms against re-run references, same ordering
    discipline -- a partial run is still a complete comparison.

    pass 1  every v2 arm + every reference, median target, default strength
    pass 2  the v2 hyperparameter (tau multiple / trust radius)
    pass 3  the strength sweep
    pass 4  the q90 target
    """
    seen, cells = set(), []

    def add(prop, arm, tgt, w, win, variant="-"):
        key = (prop, arm, tgt, w, win, variant)
        if key not in seen:
            seen.add(key)
            cells.append(key)

    REF_TAG = "v2ref"          # keeps reference cells off the stage-main names

    def variants(arm, single=False):
        """Hyperparameter settings for an arm; `single` returns only the
        pre-registered default so pass 1 stays one cell per arm."""
        if arm in V2_REFERENCES:
            # A reference cell must not collide with a stage-main cell: they
            # share (prop, arm, target, w) and cell_name has no stage field, so
            # 5 of them already exist on disk and would be skipped as done --
            # comparing the v2 arms against numbers from a different window.
            return [REF_TAG]
        if arm.startswith("btvg"):
            v = [1.0] if single else TAU_MULT
            return ["tau%g" % m for m in v]
        if arm == "spbc":
            v = [-1.0] if single else SPBC_RADIUS
            return ["r%g" % r for r in v]
        if arm == "rch":
            # Haimo v1's two parameterisations are different claims, so both
            # are measured rather than one standing in for the other.
            return ["residual"] if single else ["residual", "direct"]
        return ["-"]

    sel = [a for a in arms if a not in ("unguided",)]
    # References are ALWAYS scheduled, whatever `arms` says: the point of the
    # stage is to measure the new arms against them under the same seed, window
    # and target, and filtering them by `arms` left only `unguided`.
    refs = list(V2_REFERENCES)

    for prop in props:                       # pass 1: the comparison
        for arm in refs:
            add(prop, arm, "q50", DEFAULT_W, V2_WIN, REF_TAG)
        for arm in sel:
            for v in variants(arm, single=True):
                add(prop, arm, "q50", DEFAULT_W, V2_WIN, v)
    for prop in props:                       # pass 2: the v2 hyperparameter
        for arm in sel:
            for v in variants(arm):
                add(prop, arm, "q50", DEFAULT_W, V2_WIN, v)
    for prop in props:                       # pass 3: strength
        for arm in sel:
            grid = STRENGTHS + (STRENGTHS_LOW if arm in CLIP_BOUND else [])
            for w in sorted(grid):
                for v in variants(arm, single=True):
                    add(prop, arm, "q50", w, V2_WIN, v)
        for arm in refs:
            if arm == "unguided":
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, V2_WIN, REF_TAG)
    for prop in props:                       # pass 4: the q90 target
        for arm in refs:
            add(prop, arm, "q90", DEFAULT_W, V2_WIN, REF_TAG)
        for arm in sel:
            for v in variants(arm, single=True):
                add(prop, arm, "q90", DEFAULT_W, V2_WIN, v)
    return cells


def plan_compare_cells(props, arms=None):
    """The controlled comparison table: every arm, every strength, one window.

    Ordered so a partial run is still a complete experiment -- pass 1 is the
    whole set at the default strength, so even one link gives a full arm
    comparison before any tuning is explored.

    `arms` (default COMPARE_SET) restricts the plan, e.g. `--arms tfg` for an
    arm added after the rest of the stage ran. The cells, names, seed and
    settings are identical either way, so they pair with the existing ones.
    """
    arms = list(COMPARE_SET) if arms is None else list(arms)
    seen, cells = set(), []

    def add(prop, arm, tgt, w, win):
        k = (prop, arm, tgt, w, win, "cmp")
        if k not in seen:
            seen.add(k)
            cells.append(k)

    for prop in props:                       # pass 1: the comparison itself
        for arm in arms:
            add(prop, arm, "q50", DEFAULT_W, V2_WIN)
    for prop in props:                       # pass 2: strength, per arm
        for arm in arms:
            if arm == "unguided":
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, V2_WIN)
    for prop in props:                       # pass 3: the q90 target
        for arm in arms:
            add(prop, arm, "q90", DEFAULT_W, V2_WIN)
            if arm == "unguided":
                continue
            for w in STRENGTHS:
                add(prop, arm, "q90", w, V2_WIN)
    return cells


# ---------------------------------------------------------------------------
# stage `tune`: the JOINT (strength x start-time) screen, 25 Sep
# ---------------------------------------------------------------------------
# Every earlier stage swept strength at ONE window and window at ONE strength,
# so it could never see an interaction between them -- and the project's
# best-supported claim is that the window matters more than the method. This
# stage sweeps the two together, on one fixed target (q90), and is the input to
# freeze_tune.py, which picks TWO operating points per (property, arm): the
# best in-band that clears the chemistry floor, and the best in-band with no
# chemistry constraint at all.
TUNE_ARMS = ["unguided", "plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var"]
# 9 points, dense where every strength ever frozen has landed (0.01-4) and
# running to 16. v2's rule V2a found lgd_mc still clearing the floor at the top
# of a 7-point grid that stopped at 4 -- the grid, not chemistry, was capping
# the strongest competitor -- and then measured that lgd_mc falls BELOW the
# floor at w = 8. So 16 is comfortably past where the floor can still bind, and
# an arm whose pick lands at 16 is reported as grid-limited rather than quietly
# frozen at the edge. Trimmed from 10 points (a 32 was dropped) to fit the
# 15-hour cluster budget; see TUNE_SWEEP_PLAN.md for the arithmetic.
TUNE_STRENGTHS = [0.01, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
# The same three windows the v1 arms were screened on, so the two screens are
# comparable. t_min_guide is the time guidance STARTS: 0.05 guides almost the
# whole trajectory, 0.75 only the last quarter.
TUNE_WINDOWS = [0.05, 0.5, 0.75]
TUNE_TARGET = "q90"
TUNE_VARIANT = "tune"


def plan_tune_cells(props, arms=None, strengths=None, windows=None):
    """The joint strength x window screen at one fixed target.

    Ordered so that a partial run is still a usable experiment:
      pass 1  unguided, once per property -- the chemistry floor's reference,
              without which nothing can be frozen at all
      pass 2  every arm's strength curve at V2_WIN, the window v2 used, so the
              first complete slice reproduces the comparison we already know
      pass 3  the remaining windows

    `unguided` gets ONE cell per property, not one per (w, window): it applies
    no guidance, so neither knob can change its output, and running it 30 times
    per property would buy nothing but noise in the floor.
    """
    arms = list(TUNE_ARMS) if arms is None else list(arms)
    strengths = TUNE_STRENGTHS if strengths is None else list(strengths)
    windows = TUNE_WINDOWS if windows is None else list(windows)
    seen, cells = set(), []

    def add(prop, arm, w, win):
        k = (prop, arm, TUNE_TARGET, w, win, TUNE_VARIANT)
        if k not in seen:
            seen.add(k)
            cells.append(k)

    for prop in props:                                   # pass 1
        if "unguided" in arms:
            add(prop, "unguided", DEFAULT_W, V2_WIN)
    guided = [a for a in arms if a != "unguided"]
    # V2_WIN first WHEN IT IS ASKED FOR. The cluster hands this function one
    # window per array task, so a pass that ran V2_WIN unconditionally would
    # make every non-0.5 task re-plan the whole 0.5 slice: 819 cell-runs for
    # 489 unique cells, and the task's budget spent before it reached the
    # window it was submitted to run. Gated by test_tune.py's
    # plan_tune_single_window_* cases, which call this the way the cluster does.
    ordered = ([V2_WIN] if V2_WIN in windows else []) + \
              [w for w in windows if w != V2_WIN]
    for win in ordered:
        for prop in props:
            for arm in guided:
                for w in strengths:
                    add(prop, arm, w, win)
    return cells


def plan_tuned_full_cells(props, frozen, arms=None):
    """The full run at the operating points freeze_tune.py chose.

    `frozen` is frozen_tune.json. Each (property, arm) contributes up to TWO
    cells -- its floor-constrained pick and its unconstrained pick -- but the
    two are DEDUPLICATED by (w, t_min): when an arm's best in-band already
    clears the chemistry floor the two picks coincide, and the cell is sampled
    once. frozen_tune.json records which file each set points at, so the
    reporting never has to guess.

    Cells are named by their parameters alone (variant `tuned`), never by which
    set chose them, which is what makes that deduplication safe.
    """
    seen, cells = set(), []
    want = set(frozen["picks"]) if arms is None else set(arms)
    unknown = sorted(want - set(frozen["picks"]))
    if unknown:
        raise SystemExit("frozen_tune.json has no picks for arm(s): %s"
                         % ", ".join(unknown))
    # unguided FIRST. It is the chemistry floor's reference and the denominator
    # of every z-test, and the per-task budget can cut a group short, so it
    # must never be the cell that gets dropped. frozen_tune.json is written
    # with sort_keys, which would otherwise put it last.
    order = ([a for a in frozen["picks"] if a == "unguided" and a in want]
             + [a for a in frozen["picks"] if a != "unguided" and a in want])
    for prop in props:
        for arm in order:
            for which in ("floor", "open"):
                pick = frozen["picks"][arm].get(prop, {}).get(which)
                if not pick:
                    continue
                k = (prop, arm, frozen.get("target", TUNE_TARGET),
                     float(pick["w"]), float(pick["t_min_guide"]), "tuned")
                if k not in seen:
                    seen.add(k)
                    cells.append(k)
    return cells


def plan_target_cells(props, arms, targets, win=None, strengths=None):
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


# The frozen-strength sets check_fullrun_go.py writes. "primary" is FR3a, the
# headline; "mae" is FR3 as registered (unconstrained best MAE).
FROZEN_SETS = {"primary": "frozen_w", "mae": "frozen_w_mae"}


def load_frozen(path, props, arms, sets=("primary",)):
    """Read the frozen strengths, and refuse anything incomplete.

    `path` is what check_fullrun_go.py --json-out wrote from the finished
    compare stage. A missing (set, arm, prop) raises here, before any GPU time
    is spent, rather than surfacing as a KeyError after the first cell.
    """
    with open(path) as fh:
        fz = json.load(fh)
    # FR3-corrected: dist freezes at the q90 compare-stage strength. A file
    # from any other screen is the wrong decision, however complete it is.
    if fz.get("source_stage") != "compare" or fz.get("target") != "q90":
        raise SystemExit("frozen-strength file %s came from stage=%r target=%r; "
                         "the full run needs stage='compare', target='q90' (FR3)"
                         % (path, fz.get("source_stage"), fz.get("target")))
    missing = []
    for s in sets:
        if s not in FROZEN_SETS:
            raise SystemExit("unknown strength set %r (have %s)" % (s, sorted(FROZEN_SETS)))
        w = fz.get(FROZEN_SETS[s]) or {}
        missing += ["%s:%s/%s" % (s, a, p) for a in arms for p in props
                    if p not in w.get(a, {})]
    if missing:
        raise SystemExit("frozen-strength file %s has no entry for: %s"
                         % (path, ", ".join(missing)))
    return fz


def plan_full_cells(props, arms, frozen, sets=("primary",), exclude=(),
                    target=DIST_TARGET, win=None):
    """The full-scale run: each arm once per property PER STRENGTH SET, at the
    frozen strength, minus any cell that also belongs to an `exclude` set.

    No strength axis on purpose (FR3): the strength was chosen on the n=512
    compare stage, and this run uses new seeds so the choice is not also the
    evaluation. Two sets can share a cell (same w) -- it is then one cell, run
    once. `exclude` lets the secondary-set tasks run ONLY the cells the primary
    tasks do not, so no two concurrent tasks ever write the same cell. The
    "full" variant keeps these names disjoint from every screening cell.
    """
    win = V2_WIN if win is None else win

    def cells_of(names):
        out = []
        for s in names:
            w = frozen[FROZEN_SETS[s]]
            out += [(prop, arm, target, float(w[arm][prop]), win, "full")
                    for prop in props for arm in arms]
        return out
    drop = set(cells_of(exclude))
    seen, cells = set(), []
    for c in cells_of(sets):
        if c not in seen and c not in drop:
            seen.add(c)
            cells.append(c)
    return cells


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=None,
                    help="generator checkpoint. Default: the first of "
                         "betty_pull/fm_last.pt, proj1/checkpoints/fm_last.pt, "
                         "weights/fm_ema.pt that exists -- so a fresh clone "
                         "uses the published weights without being told to.")
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--arms", default="")
    ap.add_argument("--targets", default="dist",
                    help="comma-separated target protocols for --stage targets: "
                         "'dist' (per-molecule, the published protocol), or any "
                         "TARGETS key such as q50,q90")
    ap.add_argument("--stage", default="main",
                    choices=["main", "v2", "all", "compare", "targets", "full",
                             "tune", "tuned_full"],
                    help="main = the v1 grid (default arms); v2 = the new arms "
                         "against re-run references; all = both, v1 first; "
                         "full = every compare-set arm once at its --frozen "
                         "strength on the `dist` target; tune = the joint "
                         "strength x t_min_guide screen at q90; tuned_full = "
                         "the full run at the two operating points "
                         "freeze_tune.py chose (needs --frozen)")
    ap.add_argument("--frozen", default="",
                    help="--stage full only: the json check_fullrun_go.py "
                         "--json-out wrote (FR3's frozen strengths)")
    ap.add_argument("--sets", default="primary",
                    help="--stage full: comma-separated strength sets to run "
                         "(primary = FR3a headline, mae = FR3 as registered)")
    ap.add_argument("--full-target", default=DIST_TARGET,
                    help="--stage full: which target the full run uses. "
                         "'dist' (default) is v1's per-molecule protocol; a "
                         "TARGETS key such as 'q90' is v2's fixed target "
                         "(FULL_RUN_V2_PROTOCOL.md). The target is part of "
                         "every cell name, so the two never collide.")
    ap.add_argument("--exclude-sets", default="",
                    help="--stage full: drop cells that also belong to these "
                         "sets (so secondary tasks never duplicate primary ones)")
    ap.add_argument("--save-coords", action="store_true",
                    help="with --per-mol: also store the final coordinates and "
                         "atom types in the sidecar (pilot use; ~1 MB per 2k)")
    ap.add_argument("--per-mol", action="store_true",
                    help="also save each cell's per-molecule f_A/f_B/target/"
                         "stability as <cell>.permol.pt, so arms can be "
                         "compared PAIRED (same noise, same targets) and "
                         "re-scored without re-sampling")
    ap.add_argument("--only-w", default="",
                    help="comma list: keep only planned cells at these "
                         "strengths (a pilot filter; planning is unchanged)")
    ap.add_argument("--windows", default="",
                    help="stage tune: comma-separated t_min_guide values, "
                         "default TUNE_WINDOWS")
    ap.add_argument("--delta-json", default="",
                    help="take the in-band delta per property from this file "
                         "(results/local_fb_mae.json) instead of "
                         "k_delta x f_B's val MAE. It also sets BTVG's tau "
                         "(= delta / 1.96), so it changes sampling, not only "
                         "scoring -- see the note in main()")
    ap.add_argument("--strengths", default="",   # stages `targets` and `tune`
                    help="--stage targets only: comma list replacing the "
                         "registered strength grid (default: STRENGTHS)")
    ap.add_argument("--dist-offset", type=int, default=0,
                    help="dist targets from test[offset : offset+n] (default 0 "
                         "= the full run's block). Use a disjoint block to tune "
                         "strengths without touching the scored targets.")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--solver", default="euler", choices=["euler", "heun"])
    ap.add_argument("--n-probe", type=int, default=1)
    ap.add_argument("--n-mc", type=int, default=4)
    ap.add_argument("--sigma-mc", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--k-delta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="stop cleanly before the scheduler kills the job; the "
                         "next submission resumes at the first unfinished cell")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out-dir", default=OUT,
                    help="where cell jsons are written and resumed from. The "
                         "final full-scale run uses its own directory per seed "
                         "so its cells never collide with the screening cells, "
                         "which were produced at a different sample size.")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the cells and the projected cost, run nothing")
    ap.add_argument("--kappa3", action="store_true",
                    help="log the standardised skew gamma = k3[g,g,g]/(g'Sg)^1.5 "
                         "of the property law on every guided step. This is the "
                         "programme's stated go/no-go -- it measures whether the "
                         "Gaussian closure every SMG-family arm assumes is valid "
                         "-- and it has never been switched on. Costs one extra "
                         "third-cumulant evaluation per step.")
    ap.add_argument("--preflight", action="store_true",
                    help="run ONE throwaway cell per arm at minimum cost, "
                         "write nothing, and exit nonzero if any arm raises. "
                         "Catches missing checkpoints and wiring errors in a "
                         "minute instead of after a four-hour queue window.")
    args = ap.parse_args()
    if args.fm is None:
        args.fm = default_generator()

    OUT = args.out_dir
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(OUT, exist_ok=True)
    props = [p for p in args.props.split(",") if p]
    if args.arms:
        arms = [a for a in args.arms.split(",") if a]
    elif args.stage == "main":
        arms = list(ARMS)
    elif args.stage == "v2":
        arms = V1_MISSING + V2_ARMS + SHG_ARMS
    elif args.stage == "compare":
        arms = list(COMPARE_SET)
    elif args.stage in ("tune", "tuned_full"):
        arms = list(TUNE_ARMS)
    elif args.stage in ("targets", "full"):
        arms = list(COMPARE_SET)
    else:
        arms = list(ARMS)

    if args.preflight:
        # one cell per arm, first property only, smallest run that still
        # exercises every code path: load the guide, build the sampler, take
        # real steps, score the samples.
        # steps=10, not 4. At steps=4 the time grid is {0, .25, .5, .75}, so
        # with t_min_guide=0.5 the SHG handoffs at 0.80/0.85/0.90 were NEVER
        # entered and shg_plug_btvg's preflight MAE came out exactly equal to
        # plug's. steps=10 puts a node in every scheduled interval.
        args.n, args.batch, args.steps = 8, 8, 10
        globals()["DFLOW_ITERS"] = 2      # preflight exercises the path, not the optimum

    cells = []
    if args.stage in ("main", "all"):
        cells += plan_cells(props, list(ARMS) if args.stage == "all" else arms)
    if args.stage == "compare":
        # --arms restricts the compare plan (an arm added later, e.g. tfg);
        # without it, the whole COMPARE_SET as before
        cells += plan_compare_cells(props, arms if args.arms else None)
    if args.stage == "tune":
        cells += plan_tune_cells(
            props, arms if args.arms else None,
            strengths=([float(x) for x in args.strengths.split(",") if x]
                       if args.strengths else None),
            windows=([float(x) for x in args.windows.split(",") if x]
                     if args.windows else None))
    if args.stage == "tuned_full":
        if not args.frozen:
            raise SystemExit("--stage tuned_full needs --frozen "
                             "(frozen_tune.json from freeze_tune.py)")
        ft = json.load(open(args.frozen))
        if ft.get("schema") != "frozen_tune/1":
            raise SystemExit("%s is not a frozen_tune file (schema %r)"
                             % (args.frozen, ft.get("schema")))
        missing = [p for p in props
                   if any(p not in ft["picks"][a] for a in ft["picks"])]
        if missing:
            raise SystemExit("frozen_tune.json has no picks for %s"
                             % ", ".join(missing))
        cells += plan_tuned_full_cells(props, ft, arms if args.arms else None)
    if args.stage == "targets":
        cells += plan_target_cells(
            props, arms, [t for t in args.targets.split(",") if t],
            strengths=([float(x) for x in args.strengths.split(",") if x]
                       if args.strengths else None))
    frozen = None
    if args.stage == "full":
        if not args.frozen:
            raise SystemExit("--stage full needs --frozen (the json written by "
                             "check_fullrun_go.py --json-out)")
        sets = [s for s in args.sets.split(",") if s]
        excl = [s for s in args.exclude_sets.split(",") if s]
        frozen = load_frozen(args.frozen, props, arms, sets + excl)
        if args.full_target != DIST_TARGET and args.full_target not in TARGETS[props[0]]:
            raise SystemExit("--full-target %r is neither %r nor a TARGETS key (%s)"
                             % (args.full_target, DIST_TARGET,
                                ", ".join(sorted(TARGETS[props[0]]))))
        cells += plan_full_cells(props, arms, frozen, sets, excl,
                                 target=args.full_target)
    if args.stage in ("v2", "all"):
        v2 = V1_MISSING + V2_ARMS + SHG_ARMS if args.stage == "all" else arms
        cells += plan_v2_cells(props, v2)
    if args.preflight:
        seen_arm, first = set(), []
        for c in cells:
            if c[0] != props[0] or c[1] in seen_arm:
                continue
            seen_arm.add(c[1])
            first.append(c)
        cells = first
    if args.only_w:
        keep = {float(x) for x in args.only_w.split(",") if x}
        cells = [c for c in cells if float(c[3]) in keep]
    todo = [c for c in cells if not os.path.exists(os.path.join(OUT, cell_name(*c)))]
    if args.preflight:
        todo = list(cells)          # never skip: nothing was written
    print("cells total %d | already done %d | to run %d"
          % (len(cells), len(cells) - len(todo), len(todo)))
    if args.dry_run:
        for c in todo[:20]:
            print("   ", cell_name(*c))
        print("   ... (%d more)" % max(0, len(todo) - 20))
        return 0
    if not todo:
        print("nothing to do -- the sweep is complete")
        return 0

    d = torch.load(DATA, weights_only=False)
    types = d["types"]
    net, ck = load_fm(args.fm, len(types), dev)
    family = ck.get("family", "flow")
    print("generator: %s  [%s]  family=%s  epoch=%s"
          % (os.path.basename(args.fm), describe(args.fm), family,
             ck.get("epoch")))

    prov = {"fm_path": os.path.abspath(args.fm),
            "fm_md5": file_md5(args.fm),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": (torch.cuda.get_device_name(0)
                       if dev == "cuda" and torch.cuda.is_available() else "cpu")}
    if args.stage == "tuned_full" and args.frozen:
        # same idea as below, for the tune chain: every tuned cell must be
        # traceable to the freeze that chose its (w, t_min), and the freeze's
        # delta must be the one this run is sampling under -- btvg's tau comes
        # from it, so a mismatch would mean the cell was sampled at a strength
        # the freeze never screened.
        _ft = json.load(open(args.frozen))
        for _p in props:
            _fd = (_ft.get("delta") or {}).get(_p)
            if _fd is not None and abs(_fd - deltas[_p][0]) > 1e-12:
                raise SystemExit(
                    "%s froze %s at delta %r, but this run is sampling at %r. "
                    "btvg's tau is delta/1.96, so the frozen strengths were "
                    "screened under a different guidance field."
                    % (args.frozen, _p, _fd, deltas[_p][0]))
        prov.update({"frozen_path": os.path.abspath(args.frozen),
                     "frozen_md5": file_md5(args.frozen),
                     "frozen_schema": _ft.get("schema"),
                     "frozen_rule": _ft.get("rule"),
                     "frozen_selection_metric": _ft.get("selection_metric"),
                     "frozen_screen_n": _ft.get("n"),
                     "frozen_screen_seed": _ft.get("seed")})
    if frozen is not None:
        # which screen the strengths came from travels with every cell, so a
        # full-run number can always be traced to the decision that set it
        prov.update({"frozen_path": os.path.abspath(args.frozen),
                     "frozen_md5": file_md5(args.frozen),
                     "frozen_source_stage": frozen.get("source_stage"),
                     "frozen_source_target": frozen.get("target"),
                     "frozen_source_seed": frozen.get("source_seed"),
                     "frozen_sets": args.sets,
                     "frozen_exclude_sets": args.exclude_sets})
    print("provenance: %s  md5 %s  torch %s  cuda %s  on %s"
          % (os.path.basename(prov["fm_path"]), prov["fm_md5"][:12],
             prov["torch"], prov["cuda"], prov["device"]))

    guides, evals, deltas, tfg_mads = {}, {}, {}, {}
    for prop in props:
        ga = require_predictor("f_A_%s.pt" % prop,
                               "f_A steers guidance for property %r." % prop)
        gb = require_predictor("f_B_%s.pt" % prop,
                               "f_B is the held-out evaluator for %r." % prop)
        guides[prop] = PhysicalProperty(ga, len(types), dev)
        evals[prop] = PhysicalProperty(gb, len(types), dev)
        # TFG's energy is MAD-normalised (energy.py:369; TFG computes the MAD
        # on the guide's training data). Ours comes from the split f_A itself
        # records, so the normaliser and the guide saw the same molecules.
        ck_a = torch.load(ga, map_location="cpu", weights_only=False)
        split_a = ck_a.get("split") or ck_a.get("args", {}).get("split")
        if split_a not in d["split"]:
            raise SystemExit("f_A_%s.pt records split %r, not one of %s"
                             % (prop, split_a, sorted(d["split"])))
        ya = d["y"][d["split"][split_a], d["props"].index(prop)].double()
        tfg_mads[prop] = float((ya - ya.mean()).abs().mean())
        mae_b = float(torch.load(gb, map_location="cpu", weights_only=False)["val_mae"])
        deltas[prop] = (choose_delta(mae_b, args.k_delta), mae_b)
        print("  %-6s f_A %.5f  f_B %.5f  delta %.5f"
              % (prop, float(torch.load(ga, map_location="cpu",
                                        weights_only=False)["val_mae"]),
                 mae_b, deltas[prop][0]))

    # --delta-json overrides the tolerance for every property. This is NOT a
    # scoring-only switch: delta sets BTVG's tau (arm_kwargs: tau = delta/1.96)
    # and the `band` arm's width, so btvg/btvg_var/band cells SAMPLED under one
    # delta are not comparable to cells sampled under another. Every cell
    # records `delta`, `mae_B`, `k_delta` and `delta_source`, so a mixed tree
    # is detectable after the fact; freeze_tune.py refuses one.
    delta_src = "k_delta x f_B val MAE"
    if args.delta_json:
        dj = json.load(open(args.delta_json))
        miss = [p for p in props if p not in dj.get("delta", {})]
        if miss:
            raise SystemExit("--delta-json %s has no delta for %s"
                             % (args.delta_json, ", ".join(miss)))
        for prop in props:
            gm = dj.get("global_mae", {}).get(prop)
            mae_b = deltas[prop][1]
            if gm is None:
                raise SystemExit(
                    "--delta-json %s records no global_mae for %s. Without it "
                    "there is no way to tell whether this delta was computed "
                    "against the f_B this run uses, and a delta from another "
                    "f_B would silently move the in-band bar (and btvg's tau)."
                    % (args.delta_json, prop))
            if abs(gm - mae_b) > 1e-6 * max(1.0, abs(gm)):
                raise SystemExit(
                    "--delta-json %s was computed against a different f_B for "
                    "%s: its global val MAE %r is not this run's %r"
                    % (args.delta_json, prop, gm, mae_b))
            deltas[prop] = (float(dj["delta"][prop]), mae_b)
        delta_src = dj.get("rule", os.path.basename(args.delta_json))
        print("  delta overridden from %s:" % args.delta_json)
        for prop in props:
            print("    %-6s delta %.5f  (was %.5f)"
                  % (prop, deltas[prop][0], choose_delta(deltas[prop][1], args.k_delta)))

    va = d["split"]["val"][: args.n]
    mask_v = d["mask"][va].to(dev)
    # the `dist` protocol's own index set: sizes AND targets from the same
    # held-out test molecules, so the size-property coupling survives
    # --dist-offset takes the targets from a DIFFERENT block of test, so a
    # strength can be tuned on one block and scored on another (0 = the
    # full run's test[:n], unchanged)
    n_test = len(d["split"]["test"])
    if args.dist_offset < 0 or args.dist_offset + args.n > n_test:
        raise SystemExit("--dist-offset %d + --n %d exceeds the %d test molecules"
                         % (args.dist_offset, args.n, n_test))
    te_idx = d["split"]["test"][args.dist_offset: args.dist_offset + args.n]
    mask_t = d["mask"][te_idx].to(dev)
    coords_v, feats_v = d["coords"][va].to(dev), d["feats"][va].to(dev)
    clip = None if args.clip < 0 else args.clip
    t0 = time.time()
    done = 0
    failed = []
    skipped = []

    def run_cell(prop, arm, tgt, w, win, variant="-"):
        f_A, f_B = guides[prop], evals[prop]
        delta, mae_b = deltas[prop]
        extra = arm_kwargs(arm, variant, delta)
        if arm == "rch":
            # Haimo v1 needs a head fitted for THIS property. Running it
            # without one would silently fall back to an untrained head and
            # report a number that looks like a result.
            rp = find_predictor("rch_%s.pt" % prop) or os.path.join(
                CKPT, "rch_%s.pt" % prop)
            if not os.path.exists(rp):
                raise MissingState(
                    "rch needs %s -- run proj1/cluster/fit_rch.slurm. The rest "
                    "of this job is unaffected; these cells retry on the next "
                    "link once the head exists." % rp)
            # NOT `tgt`: that parameter holds the q50/q90 target name, and
            # shadowing it here sent "residual" into TARGETS[prop][...].
            rch_target = (variant if variant in ("residual", "direct")
                          else "residual")
            extra["rch"] = load_rch(rp, dev, rch_target)
        if arm == "band":
            extra["band_tau"] = delta
        if arm == "tfg":
            # TFG's own published QM9 configuration for this property; the
            # strength w multiplies its (rho, mu) inside the sampler
            extra["tfg"] = tfg_config(prop, tfg_mads[prop])
        # `dist` draws BOTH the molecule sizes and the targets from the SAME
        # held-out molecules. Taking sizes from one split and targets from
        # another destroys the size-property coupling that is physics: in the
        # real data corr(size, alpha) = +0.755, but pairing val sizes with test
        # targets gives -0.033, i.e. a 10-atom molecule asked to hit a 25-atom
        # molecule's polarisability. Every arm would fail that for reasons
        # having nothing to do with guidance.
        mask_c = mask_t if tgt == DIST_TARGET else mask_v
        if tgt == DIST_TARGET:
            # The field's protocol: each molecule gets its OWN target, the real
            # property of a held-out molecule.
            #
            # Drawn from TEST, not val. `val` already carries the generator's
            # validation, BOTH predictors' model selection, delta (= 2 x f_B
            # val MAE) and the molecule sizes sampled below. Taking targets
            # from it as well would make the headline number rest on a split
            # every component has already seen. `test` is untouched: verified
            # zero index overlap with train_a, train_b and val.
            y_real = d["y"][te_idx, d["props"].index(prop)].to(dev).float()
            target = float(y_real.mean())
        else:
            target = TARGETS[prop][tgt]
        s = f_A.y_std
        w_scale = strength_scale(arm, extra, s)
        w_applied = w * w_scale
        if arm in SHG_SCHEDULES:
            # normalise the btvg phases individually; the arm-level scale stays
            # 1.0 so the plug/smg/spbc phases are untouched
            extra["schedule"] = scale_schedule(extra["schedule"], s,
                                               extra.get("tau"))
        y_t = (y_real if tgt == DIST_TARGET
               else torch.full((args.n,), target, device=dev))
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        cs, fs, calls = [], [], 0
        cost = {k: 0 for k in ("gen_fwd", "gen_vjp", "gen_jvp",
                               "guide_fwd", "guide_bwd", "guide_hvp")}
        clipped = 0
        sched_log, diag_log, k3_acc = {}, {}, []
        for i in range(0, args.n, args.batch):
            m = mask_c[i:i + args.batch]
            c0, f0 = initial_noise(m, len(types), gen)
            if arm == "dflow":
                # Trajectory optimisation, not a guidance field: there is no
                # `mode` and no posterior. The sampler is replaced wholesale.
                from dflow import dflow_optimise
                cst = Cost()
                c, f = dflow_optimise(
                    net, f_A, m, c0, f0, y_t[i:i + m.shape[0]],
                    n_steps=args.steps, n_iter=DFLOW_ITERS,
                    lr=DFLOW_LR * w, cost=cst, trust=None)
                cs.append(c); fs.append(f)
                calls += cst.gen_fwd
                for k in cost:
                    cost[k] += getattr(cst, k)
                continue
            kw = dict(f_net=(None if arm == "unguided" else f_A),
                      y=y_t[i:i + m.shape[0]], s=s, mode=arm, w=w_applied,
                      clip=clip, n_probe=args.n_probe, n_mc=args.n_mc,
                      sigma_mc=args.sigma_mc,
                      want_kappa3=args.kappa3, **extra)
            if family == "vp_diffusion":
                smp = VPSampler(net, m, **kw)
            else:
                smp = FlowSampler(net, m, t_min_guide=win, **kw)
            c, f, nc = integrate(smp, c0, f0, args.steps, args.solver)
            cs.append(c); fs.append(f); calls += nc
            for k in cost:
                cost[k] += getattr(smp.cost, k)
            clipped += smp.n_clipped
            # accumulate across batches: a fresh sampler is built per batch, so
            # reading these off the last one described 128 of 512 samples
            for mk, mv in smp.schedule_log.items():
                sched_log[mk] = sched_log.get(mk, 0) + mv
            if args.kappa3 and smp.kappa3_log:
                import statistics
                allk = [float(v) for t in smp.kappa3_log
                        for v in t.reshape(-1).tolist()
                        if v == v and abs(v) < 1e6]
                if allk:
                    k3_acc.append((statistics.fmean(allk),
                                   statistics.fmean([abs(x) for x in allk])))
            for dk, dv in smp.diag_summary().items():
                tot, cnt = diag_log.get(dk, (0.0, 0))
                diag_log[dk] = (tot + dv, cnt + 1)
        C, F = torch.cat(cs), torch.cat(fs)
        r = evaluate_samples(C, F, mask_c, types, f_A, f_B, y_t, delta,
                             per_mol=args.per_mol)
        r.pop("delta", None)
        if args.per_mol:
            # which real molecule each row took its size (and, for dist, its
            # target) from -- the key that pairs rows across arms and seeds
            r["_per_mol"]["mol_idx"] = (te_idx if tgt == DIST_TARGET
                                        else va).clone().cpu()
            if args.save_coords:
                # the final molecules themselves, so chemistry can be re-scored
                # by an independent rule (scripts/independent_chem.py)
                r["_per_mol"]["coords"] = C.detach().float().cpu()
                r["_per_mol"]["types"] = F.argmax(-1).to(torch.uint8).cpu()
                r["_per_mol"]["mask"] = mask_c.detach().bool().cpu()
        r.update({"prop": prop, "arm": arm, "target_name": tgt, "target": target,
                  "variant": variant, "stage": args.stage, "prov": prov,
                  "schedule_used": sched_log,
                  "w_applied": w_applied, "w_scale": w_scale,
                  "kappa3": args.kappa3,
                  "kappa3_mean": (sum(a for a, _ in k3_acc) / len(k3_acc)
                                  if k3_acc else None),
                  "kappa3_abs_mean": (sum(b for _, b in k3_acc) / len(k3_acc)
                                      if k3_acc else None),
                  "diag": {dk: tot / cnt for dk, (tot, cnt) in diag_log.items()},
                  "w": w, "t_min_guide": win, "n": args.n, "steps": args.steps,
                  "solver": args.solver, "n_probe": args.n_probe,
                  "n_mc": args.n_mc, "delta": delta, "mae_B": mae_b,
                  "cost": cost, "field_evals": calls,
                  "clipped_sample_steps": clipped, "seed": args.seed,
                  "dist_offset": args.dist_offset,
                  "fm": os.path.basename(args.fm),
                  "clip": args.clip, "k_delta": args.k_delta,
                  "delta_source": delta_src,
                  "sigma_mc": args.sigma_mc, "batch": args.batch})
        if arm == "tfg":
            # n_mc / sigma_mc above are lgd_mc's and tfg_mc's settings; tfg
            # uses none of them. Its real configuration travels with the cell.
            r["tfg_config"] = extra["tfg"]
        return r

    for (prop, arm, tgt, w, win, variant) in todo:
        if args.max_minutes:
            used = (time.time() - t0) / 60.0
            per = used / max(done, 1)
            if done and used + per > args.max_minutes:
                print("  time guard: %.1f min used, %.2f min/cell, stopping with "
                      "%d cells left; re-submit to continue"
                      % (used, per, len(todo) - done))
                break
        cs = time.time()
        name = cell_name(prop, arm, tgt, w, win, variant)
        try:
            r = run_cell(prop, arm, tgt, w, win, variant)
        except Exception as e:
            # A failure must NOT be written under the completed-cell name, or
            # no later link ever retries it and the sweep reports success with
            # a hole in it. Record it beside the cell and move on.
            if args.preflight and isinstance(e, MissingState):
                # not a failure: one arm's prerequisite is absent, the job is
                # fine. Skipping it here is what stops a two-minute fitting job
                # from gating a fourteen-hour sweep.
                skipped.append(arm)
                print("  PREFLIGHT SKIP   %-22s %s" % (arm, e))
                continue
            failed.append(name)
            if args.preflight:
                print("  PREFLIGHT FAILED %-22s %s: %s"
                      % (arm, type(e).__name__, e))
                continue
            with open(os.path.join(OUT, name + ".failed"), "w") as fh:
                json.dump({"cell": name, "error": "%s: %s" % (type(e).__name__, e)}, fh)
            print("  CELL FAILED (will be retried on the next link) %s: %s" % (name, e))
            continue
        per_mol = r.pop("_per_mol", None)
        if not args.preflight:
            if per_mol is not None:
                # the sidecar lands BEFORE the json: the json is the
                # completion marker, so a cell is never "done" without it
                ptmp = os.path.join(OUT, "%s.permol.%d.tmp" % (name, os.getpid()))
                torch.save(per_mol, ptmp)
                os.replace(ptmp, os.path.join(OUT, name[:-5] + ".permol.pt"))
            tmp = os.path.join(OUT, "%s.%d.tmp" % (name, os.getpid()))
            with open(tmp, "w") as fh:
                json.dump(r, fh)
            os.replace(tmp, os.path.join(OUT, name))
        done += 1
        print("  [%3d/%3d] %-6s %-14s %-4s %-7s w=%-4g tmin=%-5g  mae=%-8s in_band=%-6s  %.0fs"
              % (done, len(todo), prop, arm, tgt, variant, w, win,
                 ("%.4f" % r["prop_mae_eval"]) if "prop_mae_eval" in r else "FAIL",
                 ("%.3f" % r["in_band_fraction"]) if "in_band_fraction" in r else "-",
                 time.time() - cs))

    if args.preflight:
        print("\npreflight: %d arms ran, %d failed, %d skipped"
              % (done, len(failed), len(skipped)))
        if skipped:
            print("SKIPPED (prerequisite missing, job proceeds): %s"
                  % ",".join(sorted(set(skipped))))
        if failed:
            print("PREFLIGHT FAILED -- fix these before submitting the sweep")
            return 1
        print("PREFLIGHT OK -- every runnable arm works end to end")
        return 0

    print("\n%d cells this job, %.1f min total" % (done, (time.time() - t0) / 60.0))
    if failed:
        print("%d cells FAILED this job and will be retried:" % len(failed))
        for nm in failed[:10]:
            print("    " + nm)
    left = [c for c in cells if not os.path.exists(os.path.join(OUT, cell_name(*c)))]
    if not left:
        print("SWEEP COMPLETE")
    else:
        print("%d cells remain -- re-submit (%d of them previously failed)"
              % (len(left), len(failed)))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
