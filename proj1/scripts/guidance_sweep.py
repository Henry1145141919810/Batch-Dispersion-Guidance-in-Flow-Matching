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
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from guidance import ResidualCalibrationHead  # noqa: E402
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

DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT = os.path.join(ROOT, "proj1", "checkpoints")
OUT = os.path.join(ROOT, "results", "sweep")

# measured on the target distribution of train_a; see GUIDANCE_EXPERIMENT_PLAN.md
TARGETS = {"mu":    {"q50": 2.4932, "q90": 4.6627},
           "alpha": {"q50": 75.54, "q90": 85.05},
           "gap":   {"q50": 0.2496, "q90": 0.3162}}

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
V2_ARMS = ["spbc", "btvg", "btvg_mean", "btvg_var"]

# REFERENCES. Every v2 comparison re-runs these in the same cells so the
# contrast is measured under identical seeds, targets and windows rather than
# read off a table from another job.
V2_REFERENCES = ["unguided", "plug", "tmpd", "smg"]

# BTVG's tolerance. tau is NOT delta: a centred Gaussian with sd = delta gives
# only 68.3% band coverage, and 95% needs sd ~ delta/1.96. The multiplier
# brackets that choice so the pre-registered value can be checked rather than
# assumed.
TAU_MULT = [0.5, 1.0, 2.0]          # x (delta / 1.96)

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
    if arm.startswith("btvg") and kw.get("tau"):
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
    return cells


# the window the v2 arms are screened at: the measured optimum of the U-shaped
# window effect, not the v1 default of 0.05
V2_WIN = 0.5


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
            for w in STRENGTHS:
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


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=os.path.join(ROOT, "betty_pull", "fm_last.pt"))
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--arms", default="")
    ap.add_argument("--stage", default="main", choices=["main", "v2", "all"],
                    help="main = the v1 grid (default arms); v2 = the new arms "
                         "against re-run references; all = both, v1 first")
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
    else:
        arms = list(ARMS)

    if args.preflight:
        # one cell per arm, first property only, smallest run that still
        # exercises every code path: load the guide, build the sampler, take
        # real steps, score the samples.
        args.n, args.batch, args.steps = 8, 8, 4

    cells = []
    if args.stage in ("main", "all"):
        cells += plan_cells(props, list(ARMS) if args.stage == "all" else arms)
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
    print("generator: %s  family=%s  epoch=%s" % (os.path.basename(args.fm),
                                                  family, ck.get("epoch")))

    prov = {"fm_path": os.path.abspath(args.fm),
            "fm_md5": file_md5(args.fm),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": (torch.cuda.get_device_name(0)
                       if dev == "cuda" and torch.cuda.is_available() else "cpu")}
    print("provenance: %s  md5 %s  torch %s  cuda %s  on %s"
          % (os.path.basename(prov["fm_path"]), prov["fm_md5"][:12],
             prov["torch"], prov["cuda"], prov["device"]))

    guides, evals, deltas = {}, {}, {}
    for prop in props:
        ga = os.path.join(CKPT, "f_A_%s.pt" % prop)
        gb = os.path.join(CKPT, "f_B_%s.pt" % prop)
        for p_ in (ga, gb):
            if not os.path.exists(p_):
                raise SystemExit("missing predictor %s -- pull it from the cluster" % p_)
        guides[prop] = PhysicalProperty(ga, len(types), dev)
        evals[prop] = PhysicalProperty(gb, len(types), dev)
        mae_b = float(torch.load(gb, map_location="cpu", weights_only=False)["val_mae"])
        deltas[prop] = (choose_delta(mae_b, args.k_delta), mae_b)
        print("  %-6s f_A %.5f  f_B %.5f  delta %.5f"
              % (prop, float(torch.load(ga, map_location="cpu",
                                        weights_only=False)["val_mae"]),
                 mae_b, deltas[prop][0]))

    va = d["split"]["val"][: args.n]
    mask_v = d["mask"][va].to(dev)
    coords_v, feats_v = d["coords"][va].to(dev), d["feats"][va].to(dev)
    clip = None if args.clip < 0 else args.clip
    t0 = time.time()
    done = 0
    failed = []

    def run_cell(prop, arm, tgt, w, win, variant="-"):
        f_A, f_B = guides[prop], evals[prop]
        delta, mae_b = deltas[prop]
        extra = arm_kwargs(arm, variant, delta)
        if arm == "rch":
            # Haimo v1 needs a head fitted for THIS property. Running it
            # without one would silently fall back to an untrained head and
            # report a number that looks like a result.
            rp = os.path.join(CKPT, "rch_%s.pt" % prop)
            if not os.path.exists(rp):
                raise RuntimeError("rch needs %s -- fit it with "
                                   "proj1/scripts/fit_rch.py first" % rp)
            # NOT `tgt`: that parameter holds the q50/q90 target name, and
            # shadowing it here sent "residual" into TARGETS[prop][...].
            rch_target = (variant if variant in ("residual", "direct")
                          else "residual")
            extra["rch"] = load_rch(rp, dev, rch_target)
        if arm == "band":
            extra["band_tau"] = delta
        target = TARGETS[prop][tgt]
        s = f_A.y_std
        w_scale = strength_scale(arm, extra, s)
        w_applied = w * w_scale
        if arm in SHG_SCHEDULES:
            # normalise the btvg phases individually; the arm-level scale stays
            # 1.0 so the plug/smg/spbc phases are untouched
            extra["schedule"] = scale_schedule(extra["schedule"], s,
                                               extra.get("tau"))
        y_t = torch.full((args.n,), target, device=dev)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        cs, fs, calls = [], [], 0
        cost = {k: 0 for k in ("gen_fwd", "gen_vjp", "gen_jvp",
                               "guide_fwd", "guide_bwd", "guide_hvp")}
        clipped = 0
        sched_log, diag_log, k3_acc = {}, {}, []
        for i in range(0, args.n, args.batch):
            m = mask_v[i:i + args.batch]
            c0, f0 = initial_noise(m, len(types), gen)
            kw = dict(f_net=(None if arm == "unguided" else f_A),
                      y=y_t[: m.shape[0]], s=s, mode=arm, w=w_applied,
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
        r = evaluate_samples(C, F, mask_v, types, f_A, f_B, y_t, delta)
        r.pop("delta", None)
        r.update({"prop": prop, "arm": arm, "target_name": tgt, "target": target,
                  "variant": variant, "stage": args.stage, "prov": prov,
                  "schedule_used": sched_log,
                  "w_applied": w_applied, "w_scale": w_scale,
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
                  "fm": os.path.basename(args.fm),
                  "clip": args.clip, "k_delta": args.k_delta,
                  "sigma_mc": args.sigma_mc, "batch": args.batch})
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
            failed.append(name)
            if args.preflight:
                print("  PREFLIGHT FAILED %-22s %s: %s"
                      % (arm, type(e).__name__, e))
                continue
            with open(os.path.join(OUT, name + ".failed"), "w") as fh:
                json.dump({"cell": name, "error": "%s: %s" % (type(e).__name__, e)}, fh)
            print("  CELL FAILED (will be retried on the next link) %s: %s" % (name, e))
            continue
        if not args.preflight:
            tmp = os.path.join(OUT, name + ".tmp")
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
        print("\npreflight: %d arms ran, %d failed" % (done, len(failed)))
        if failed:
            print("PREFLIGHT FAILED -- fix these before submitting the sweep")
            return 1
        print("PREFLIGHT OK -- every arm runs end to end")
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
