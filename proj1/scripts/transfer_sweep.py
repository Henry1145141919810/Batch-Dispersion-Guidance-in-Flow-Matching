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
  6. f_A AND f_B ARE TFG'S, NEVER OURS. The guide is `tf_predict_<p>` and the
     oracle is `evaluate_<p>` (EDM's own `main_qm9_prop` classifier), both
     loaded from `audit/fa_fb_search/TFG/` by `build_pair`. Nothing in this
     file reads `weights/f_*.pt` or `proj1/checkpoints/f_*.pt`; every cell
     records `guide` / `oracle` so that can be checked after the fact.
     Both are calibrated to physical units on train_a + train_b only, so the
     `test` split -- where the `dist` targets come from -- is touched by no
     fit, the same claim the main sweep makes for its own pair.

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
from external.tfg_assets import (Calibrated, EDMGenerator, PROP_INDEX,  # noqa: E402
                                 PROP_UNITS, QM9_MAD, TFGGuide, TFGOracle,
                                 fit_calibration, metadata)
from sampling import VPSampler, initial_noise, integrate     # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
# The grid, targets and frozen-file reader are IMPORTED, not copied, so the
# transfer cannot drift from the main sweep's protocol.
from guidance_sweep import (DEFAULT_W, DIST_TARGET, FROZEN_SETS,  # noqa: E402
                            SHG_SCHEDULES, STRENGTHS, TARGETS, load_frozen,
                            scale_schedule, tfg_config)

DATA = os.path.join(ROOT, "data", "qm9.pt")
TFG_ROOT = os.path.join(ROOT, "audit", "fa_fb_search", "TFG")
OUT = os.path.join(ROOT, "results", "transfer")
BACKEND = "TFG/EDMsecond"

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
        prop, arm, tgt, w, cfg, "__full" if stage == "full" else "")


def config_tag(args):
    """The run-configuration suffix baked into every cell name.

    The batch size is in it because it changes the SAMPLES, not just the
    speed: `initial_noise` draws one batch at a time from one generator, so
    batch 64 and batch 128 give every molecule different starting noise. Two
    arms are paired (same noise, same targets) only at the same batch.
    """
    return "__n%d_s%d_%s_%s_win%g_b%d_seed%d" % (
        args.n, args.steps, args.solver, args.grid, args.tau_max_guide,
        args.batch, args.seed)


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
                    for w in STRENGTHS:
                        add((prop, arm, tgt, float(w)))
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


def build_pair(prop, d, sel, dev, k_delta, sampler_scale):
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
    delta = choose_delta(mae_o, k_delta)
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
              "oracle_input_multiplier": f_B.feat_scale}
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
    v = max(r["prop_rmse_eval"] ** 2 - r["prop_mae_eval"] ** 2, 0.0)
    return math.sqrt(v / r["n"])


def freeze(args, props, arms):
    """Apply FR3a / FR3 to the finished compare stage; exit 0 ok, 2 incomplete.

    Same rule, same tie-breaks and same json keys as check_fullrun_go.py, so
    guidance_sweep.load_frozen and full_run_table.py read the result
    unchanged. It is re-implemented rather than imported because that script
    judges the main sweep's COMPARE_SET, which contains dflow.
    """
    if sorted(arms) != sorted(TRANSFER_SET):
        # the frozen file IS the full run's arm set (load_frozen only checks
        # the arms it is asked about), so a reduced --arms here would produce
        # a reduced headline table without a word of protest
        print("REFUSING to freeze arms %s: the frozen file must cover the whole "
              "TRANSFER_SET %s. A post-hoc arm is a separate analysis (R5)."
              % (arms, TRANSFER_SET))
        return 2
    rows = {t: [] for t in SCREEN_TARGETS}
    present = {t: set() for t in SCREEN_TARGETS}
    meta = set()
    for fn in os.listdir(args.out_dir):
        if not (fn.startswith("tr__") and fn.endswith(".json")):
            continue
        r = json.load(open(os.path.join(args.out_dir, fn)))
        if r.get("stage") != "compare" or r.get("target_name") not in rows:
            continue
        t = r["target_name"]
        present[t].add((r["prop"], r["arm"], float(r["w"])))
        meta.add((r["n"], r["seed"], (r.get("prov") or {}).get("edm_md5"),
                  r.get("grid"), r.get("tau_max_guide"), r.get("batch")))
        if r.get("n_nonfinite", 0) > 0 or not math.isfinite(
                float(r.get("prop_mae_eval", float("nan")))):
            continue                # a diverged cell may not be anyone's best
        rows[t].append(r)
    if len(meta) != 1:
        print("INCONSISTENT compare cells in %s: (n, seed, edm_md5, grid, "
              "window, batch) = %s" % (args.out_dir, sorted(map(str, meta))))
        return 2
    grid = {("unguided", float(DEFAULT_W))} | {
        (a, float(w)) for a in arms if a != "unguided" for w in STRENGTHS}
    for t in SCREEN_TARGETS:
        holes = sorted((p, a, w) for p in props for (a, w) in grid
                       if (p, a, w) not in present[t])
        if holes:
            print("INCOMPLETE -- %d of %d cells missing at %s, e.g. %s"
                  % (len(holes), len(grid) * len(props), t,
                     ", ".join("%s/%s/w%g" % h for h in holes[:6])))
            return 2

    def best(rs, p, a):
        c = [r for r in rs if r["prop"] == p and r["arm"] == a]
        return min(c, key=lambda r: (r["prop_mae_eval"], float(r["w"]))) if c else None

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
            c = min(cs, key=lambda r: r["prop_mae_eval"])
            z = (b["prop_mae_eval"] - c["prop_mae_eval"]) / math.sqrt(
                _se_mae(b) ** 2 + _se_mae(c) ** 2)
            beaten += z > FR1_SIGMA
            lines.append("  %-6s best comp %-7s %.4f  btvg %.4f  z %+.2f%s"
                         % (p, c["arm"], c["prop_mae_eval"], b["prop_mae_eval"],
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
            ok = [r for r in c if r["mol_stability"] >= floor[p] - 1e-12]
            if ok:
                r = min(ok, key=lambda r: (r["prop_mae_eval"], float(r["w"])))
            else:
                r = max(c, key=lambda r: (r["mol_stability"], -float(r["w"])))
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
    (n, seed, md5, grid_, win, batch), = meta
    out = {"backend": BACKEND, "edm_md5": md5, "arms": list(arms),
           "fr1_record_only": fr1,
           "frozen_w": frozen, "frozen_w_mae": frozen_mae,
           "rule": "frozen_w = FR3a: best MAE among strengths with "
                   "mol_stability >= %.1f x unguided; frozen_w_mae = FR3 as "
                   "registered" % FLOOR,
           "floor": floor, "fell_back": fell_back,
           "source_stage": "compare", "target": FREEZE_TARGET,
           "source_seed": seed, "source_n": n, "source_batch": batch,
           "source_dir": args.out_dir,
           "grid": grid_, "tau_max_guide": win}
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
                    choices=["compare", "freeze", "full"])
    ap.add_argument("--edm-dir", default=os.path.join(ROOT, "weights", "EDMsecond"),
                    help="directory holding generative_model_ema.npy + args.pickle")
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
    ap.add_argument("--frozen", default="",
                    help="--stage full: the json --stage freeze wrote")
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
    ap.add_argument("--preflight", action="store_true",
                    help="one throwaway cell per arm at minimum cost, writes "
                         "nothing, exits nonzero if any arm raises")
    args = ap.parse_args()

    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a] or list(TRANSFER_SET)
    unknown = [a for a in arms if a not in ARM_CLASS]
    if unknown:
        raise SystemExit("unknown arm(s) %s; known: %s"
                         % (unknown, ", ".join(ARM_CLASS)))
    if args.n is None:
        args.n = 5000 if args.stage == "full" else 512
    if args.seed is None:
        if args.stage == "full":
            raise SystemExit("--stage full needs --seed (one of %s)"
                             % ", ".join(map(str, FULL_SEEDS)))
        args.seed = COMPARE_SEED
    if not args.out_dir:
        args.out_dir = (os.path.join(OUT, "full", "n%d" % args.n,
                                     "seed%d" % args.seed)
                        if args.stage == "full" else os.path.join(OUT, "compare"))

    if args.stage == "freeze":
        return freeze(args, props, arms)

    frozen = None
    if args.stage == "full":
        if not args.frozen:
            raise SystemExit("--stage full needs --frozen (the json "
                             "`--stage freeze --json-out` wrote)")
        sets = [s for s in args.sets.split(",") if s]
        excl = [s for s in args.exclude_sets.split(",") if s]
        frozen = load_frozen(args.frozen, props, arms, sets + excl)
        # load_frozen checks stage/target; it cannot tell the MAIN sweep's
        # frozen file from this one, and those strengths were chosen on a
        # different generator. Refuse it.
        if frozen.get("backend") != BACKEND:
            raise SystemExit("%s was not frozen on %s (backend=%r) -- the main "
                             "sweep's strengths do not transfer; run "
                             "--stage freeze on the transfer compare cells"
                             % (args.frozen, BACKEND, frozen.get("backend")))
        for k in ("grid", "tau_max_guide"):
            if frozen.get(k) != getattr(args, k):
                raise SystemExit("frozen strengths were chosen at %s=%r, this "
                                 "run uses %r" % (k, frozen.get(k),
                                                  getattr(args, k)))
        cells = plan_full_cells(props, arms, frozen, sets, excl)
    else:
        tg = [t for t in args.targets.split(",") if t]
        bad = [t for t in tg if t not in SCREEN_TARGETS]
        if bad:
            raise SystemExit("unknown --targets %s; the screen is %s"
                             % (bad, ",".join(SCREEN_TARGETS)))
        cells = plan_compare_cells(props, arms, tg)

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
    cfg = config_tag(args)

    def name(c):
        return cell_name(*c, cfg=cfg, stage=args.stage)

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
    net, prov = load_backend(args.edm_dir, dev)
    prov.update({"fm_md5": prov["edm_md5"],   # the generator key full_run_table checks
                 "torch": torch.__version__, "cuda": torch.version.cuda,
                 "device": (torch.cuda.get_device_name(0)
                            if dev == "cuda" and torch.cuda.is_available()
                            else "cpu"),
                 "norm_values": net.norm_values,
                 "noise_schedule": net.args["diffusion_noise_schedule"],
                 "diffusion_steps": net.args["diffusion_steps"]})
    if frozen is not None:
        prov.update({"frozen_path": os.path.abspath(args.frozen),
                     "frozen_md5": file_md5(args.frozen),
                     "frozen_source_stage": frozen.get("source_stage"),
                     "frozen_source_target": frozen.get("target"),
                     "frozen_source_seed": frozen.get("source_seed"),
                     "frozen_sets": args.sets,
                     "frozen_exclude_sets": args.exclude_sets})
    print("base model: EDMsecond  md5 %s  nf=%d layers=%d  schedule=%s  grid=%s"
          % (prov["edm_md5"][:12], net.args["nf"], net.args["n_layers"],
             prov["noise_schedule"], args.grid))

    calib_sel = calibration_indices(d, args.n_calib)
    sampler_scale = float(net.norm_values[1])
    assert sampler_scale == generator_feat_scale(args.edm_dir)
    guides, evals, deltas, reports = {}, {}, {}, {}
    for prop in props:
        f_A, f_B, delta, rep = build_pair(prop, d, calib_sel, dev, args.k_delta,
                                          sampler_scale)
        guides[prop], evals[prop] = f_A, f_B
        deltas[prop], reports[prop] = delta, rep
        print("  %-6s guide MAE %.5f %s | oracle MAE %.5f | delta %.5f | "
              "slope/MAD %.3f (guide) %.3f (oracle)"
              % (prop, rep["guide_mae"], PROP_UNITS[prop], rep["oracle_mae"],
                 delta, rep["slope_over_mad_guide"], rep["slope_over_mad_oracle"]))

    # q50/q90: sizes from val[:n], exactly as the main sweep's compare stage.
    # dist: sizes AND targets from the SAME test molecules test[:n], exactly
    # as the main sweep's full stage (see guidance_sweep.run_cell for why the
    # two must come from one molecule).
    va = d["split"]["val"][: args.n]
    te = d["split"]["test"][: args.n]
    if args.stage == "full" and te.numel() < args.n:
        raise SystemExit("--n %d exceeds the test split (%d)" % (args.n, te.numel()))
    mask_v, mask_t = d["mask"][va].to(dev), d["mask"][te].to(dev)
    clip = None if args.clip < 0 else args.clip

    def run_cell(prop, arm, tgt, w):
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

        cs, fs, calls = [], [], 0
        cost = {k: 0 for k in ("gen_fwd", "gen_vjp", "gen_jvp",
                               "guide_fwd", "guide_bwd", "guide_hvp")}
        clipped, guided, diag_log, sched_log = 0, 0, {}, {}
        for i in range(0, args.n, args.batch):
            m = mask_c[i:i + args.batch]
            c0, f0 = initial_noise(m, len(types), gen)
            smp = VPSampler(
                net, m, tau_min=args.tau_min, noise_schedule=net.schedule,
                tau_max_guide=args.tau_max_guide, grid=args.grid,
                f_net=(None if arm == "unguided" else f_A),
                y=y_t[i:i + m.shape[0]], s=s, mode=arm, w=w_applied, clip=clip,
                n_probe=args.n_probe, n_mc=args.n_mc, sigma_mc=args.sigma_mc,
                **extra)
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
        r.update({
            "prop": prop, "arm": arm, "target_name": tgt, "target": target,
            "stage": args.stage, "variant": args.stage,
            "arm_class": ARM_CLASS.get(arm, "?"),
            "backend": BACKEND, "prov": prov,
            "guide": reports[prop]["guide"], "oracle": reports[prop]["oracle"],
            "calibration": reports[prop],
            "w": w, "w_applied": w_applied, "w_scale": w_scale,
            "n": args.n, "steps": args.steps,
            "solver": args.solver, "grid": args.grid,
            "delta": delta, "mae_B": reports[prop]["oracle_mae"],
            "n_probe": args.n_probe, "n_mc": args.n_mc, "sigma_mc": args.sigma_mc,
            "clip": args.clip, "k_delta": args.k_delta, "seed": args.seed,
            "guide_time": "zero", "tau_min": args.tau_min,
            "tau_max_guide": args.tau_max_guide,
            "t_min_guide": 1.0 - args.tau_max_guide,   # the flow-time image
            "fm": "EDMsecond",
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
            if per_mol is not None:
                # the sidecar lands BEFORE the json: the json is the
                # completion marker, so a cell is never "done" without it
                ptmp = os.path.join(out, nm + ".permol.tmp")
                torch.save(per_mol, ptmp)
                os.replace(ptmp, os.path.join(out, nm[:-5] + ".permol.pt"))
            tmp = os.path.join(out, nm + ".tmp")
            with open(tmp, "w") as fh:
                json.dump(r, fh, indent=1)
            os.replace(tmp, os.path.join(out, nm))
            stale = os.path.join(out, nm + ".failed")
            if os.path.exists(stale):
                os.remove(stale)
        done += 1
        print("  [%3d/%3d] %-6s %-9s %-4s w=%-5g MAE %9.4f (%5.2f d)  in-band %.3f"
              "  mol-stab %.3f  clipped %d  %.1fs"
              % (done, len(todo), c[0], c[1], c[2], c[3], r["prop_mae_eval"],
                 r["prop_mae_eval"] / r["delta"], r["in_band_fraction"],
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
