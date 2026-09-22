"""Wire the v2 arms (SPBC, BTVG, SHG) + the missing v1 arms into the sweep.

Run once. Two things this must NOT break:

  1. The 141 already-finished cells. Their filenames are preserved exactly:
     a cell only gets a variant suffix when it actually has v2 hyperparameters.
  2. The resumable contract. New cells are appended after the existing plan, so
     a chained job finishes the v1 sweep before starting v2.

ALSO ADDS the v1 arms that were implemented and unit-tested but never queued:
  smg_mean   completes the SMG ablation ladder (mean-only <-> var-only <-> both)
  rch        Haimo v1, the residual calibration head
  band       the tolerance-band step
"""
import io

p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------
OLD_ARMS = '''ARMS = ["unguided", "plug", "tmpd", "tfg_mc", "lgd_mc", "osc",
        "smg", "smg2", "smg2_curv"]'''
NEW_ARMS = '''ARMS = ["unguided", "plug", "tmpd", "tfg_mc", "lgd_mc", "osc",
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
SHG_ARMS = sorted(SHG_SCHEDULES)'''
assert s.count(OLD_ARMS) == 1, "ARMS anchor"
s = s.replace(OLD_ARMS, NEW_ARMS)

# --------------------------------------------------------------------------
# cell naming: a variant suffix ONLY when there is something to record
# --------------------------------------------------------------------------
OLD_NAME = '''def cell_name(prop, arm, tgt, w, win):
    return "%s__%s__%s__w%g__tmin%g.json" % (prop, arm, tgt, w, win)'''
NEW_NAME = '''def cell_name(prop, arm, tgt, w, win, variant="-"):
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
    the final benchmark cannot drift apart."""
    kw = {}
    if arm.startswith("btvg"):
        mult = float(variant.split("tau")[1])
        kw["tau"] = mult * delta / 1.96
    elif arm == "spbc":
        r = float(variant.split("r")[1])
        kw["spbc_radius"] = None if r < 0 else r
    elif arm in SHG_SCHEDULES:
        kw["schedule"] = SHG_SCHEDULES[arm]
        # any mode the schedule can select must have its own hyperparameters
        # available, because `active()` can hand control to it at any step
        kw["tau"] = delta / 1.96
    return kw'''
assert s.count(OLD_NAME) == 1, "cell_name anchor"
s = s.replace(OLD_NAME, NEW_NAME)

# --------------------------------------------------------------------------
# the v2 plan
# --------------------------------------------------------------------------
OLD_PLAN_SIG = '''def plan_cells(props, arms):
    """Ordered so a partial run is still a complete experiment. See docstring."""
    seen, cells = set(), []

    def add(prop, arm, tgt, w, win):
        key = (prop, arm, tgt, w, win)
        if key not in seen:
            seen.add(key)
            cells.append(key)
'''
NEW_PLAN_SIG = '''def plan_cells(props, arms):
    """Ordered so a partial run is still a complete experiment. See docstring."""
    seen, cells = set(), []

    def add(prop, arm, tgt, w, win, variant="-"):
        key = (prop, arm, tgt, w, win, variant)
        if key not in seen:
            seen.add(key)
            cells.append(key)
'''
assert s.count(OLD_PLAN_SIG) == 1, "plan_cells signature anchor"
s = s.replace(OLD_PLAN_SIG, NEW_PLAN_SIG)

OLD_PLAN_END = '''                for w in STRENGTHS:
                    add(prop, arm, tgt, w, DEFAULT_WIN)
    return cells'''
NEW_PLAN_END = '''                for w in STRENGTHS:
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

    def variants(arm, single=False):
        """Hyperparameter settings for an arm; `single` returns only the
        pre-registered default so pass 1 stays one cell per arm."""
        if arm.startswith("btvg"):
            v = [1.0] if single else TAU_MULT
            return ["tau%g" % m for m in v]
        if arm == "spbc":
            v = [-1.0] if single else SPBC_RADIUS
            return ["r%g" % r for r in v]
        return ["-"]

    sel = [a for a in arms if a not in ("unguided",)]
    refs = [a for a in V2_REFERENCES if a in arms or a == "unguided"]

    for prop in props:                       # pass 1: the comparison
        for arm in refs:
            add(prop, arm, "q50", DEFAULT_W, V2_WIN)
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
                add(prop, arm, "q50", w, V2_WIN)
    for prop in props:                       # pass 4: the q90 target
        for arm in refs:
            add(prop, arm, "q90", DEFAULT_W, V2_WIN)
        for arm in sel:
            for v in variants(arm, single=True):
                add(prop, arm, "q90", DEFAULT_W, V2_WIN, v)
    return cells'''
assert s.count(OLD_PLAN_END) == 1, "plan_cells end anchor"
s = s.replace(OLD_PLAN_END, NEW_PLAN_END)

# --------------------------------------------------------------------------
# CLI: --stage
# --------------------------------------------------------------------------
OLD_CLI = '''    ap.add_argument("--arms", default=",".join(ARMS))'''
NEW_CLI = '''    ap.add_argument("--arms", default="")
    ap.add_argument("--stage", default="main", choices=["main", "v2", "all"],
                    help="main = the v1 grid (default arms); v2 = the new arms "
                         "against re-run references; all = both, v1 first")'''
assert s.count(OLD_CLI) == 1, "cli anchor"
s = s.replace(OLD_CLI, NEW_CLI)

OLD_SEL = '''    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a]
    cells = plan_cells(props, arms)'''
NEW_SEL = '''    props = [p for p in args.props.split(",") if p]
    if args.arms:
        arms = [a for a in args.arms.split(",") if a]
    elif args.stage == "main":
        arms = list(ARMS)
    elif args.stage == "v2":
        arms = V1_MISSING + V2_ARMS + SHG_ARMS
    else:
        arms = list(ARMS)

    cells = []
    if args.stage in ("main", "all"):
        cells += plan_cells(props, list(ARMS) if args.stage == "all" else arms)
    if args.stage in ("v2", "all"):
        v2 = V1_MISSING + V2_ARMS + SHG_ARMS if args.stage == "all" else arms
        cells += plan_v2_cells(props, v2)'''
assert s.count(OLD_SEL) == 1, "arm selection anchor"
s = s.replace(OLD_SEL, NEW_SEL)

# --------------------------------------------------------------------------
# run_cell: pass the v2 kwargs through, and skip arms whose state is missing
# --------------------------------------------------------------------------
OLD_RUN = '''    def run_cell(prop, arm, tgt, w, win):
        f_A, f_B = guides[prop], evals[prop]
        delta, mae_b = deltas[prop]'''
NEW_RUN = '''    def run_cell(prop, arm, tgt, w, win, variant="-"):
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
            extra["rch"] = load_rch(rp, dev)
        if arm == "band":
            extra["band_tau"] = delta'''
assert s.count(OLD_RUN) == 1, "run_cell anchor"
s = s.replace(OLD_RUN, NEW_RUN)

OLD_KW = '''            kw = dict(f_net=(None if arm == "unguided" else f_A),
                      y=y_t[: m.shape[0]], s=s, mode=arm, w=w, clip=clip,
                      n_probe=args.n_probe, n_mc=args.n_mc, sigma_mc=args.sigma_mc)'''
NEW_KW = '''            kw = dict(f_net=(None if arm == "unguided" else f_A),
                      y=y_t[: m.shape[0]], s=s, mode=arm, w=w, clip=clip,
                      n_probe=args.n_probe, n_mc=args.n_mc,
                      sigma_mc=args.sigma_mc, **extra)'''
assert s.count(OLD_KW) == 1, "sampler kwargs anchor"
s = s.replace(OLD_KW, NEW_KW)

OLD_REC = '''        r.update({"prop": prop, "arm": arm, "target_name": tgt, "target": target,
                  "w": w, "t_min_guide": win, "n": args.n, "steps": args.steps,'''
NEW_REC = '''        r.update({"prop": prop, "arm": arm, "target_name": tgt, "target": target,
                  "variant": variant, "stage": args.stage,
                  "schedule_used": getattr(smp, "schedule_log", {}),
                  "w": w, "t_min_guide": win, "n": args.n, "steps": args.steps,'''
assert s.count(OLD_REC) == 1, "record anchor"
s = s.replace(OLD_REC, NEW_REC)

# --------------------------------------------------------------------------
# the driver loop unpacks six-tuples now
# --------------------------------------------------------------------------
OLD_LOOP = '''    for (prop, arm, tgt, w, win) in todo:'''
NEW_LOOP = '''    for (prop, arm, tgt, w, win, variant) in todo:'''
assert s.count(OLD_LOOP) == 1, "loop anchor"
s = s.replace(OLD_LOOP, NEW_LOOP)

OLD_CALL = '''        name = cell_name(prop, arm, tgt, w, win)
        try:
            r = run_cell(prop, arm, tgt, w, win)'''
NEW_CALL = '''        name = cell_name(prop, arm, tgt, w, win, variant)
        try:
            r = run_cell(prop, arm, tgt, w, win, variant)'''
assert s.count(OLD_CALL) == 1, "call anchor"
s = s.replace(OLD_CALL, NEW_CALL)

OLD_PRINT = '''        print("  [%3d/%3d] %-6s %-10s %-4s w=%-4g tmin=%-5g  mae=%-8s in_band=%-6s  %.0fs"
              % (done, len(todo), prop, arm, tgt, w, win,'''
NEW_PRINT = '''        print("  [%3d/%3d] %-6s %-14s %-4s %-7s w=%-4g tmin=%-5g  mae=%-8s in_band=%-6s  %.0fs"
              % (done, len(todo), prop, arm, tgt, variant, w, win,'''
assert s.count(OLD_PRINT) == 1, "print anchor"
s = s.replace(OLD_PRINT, NEW_PRINT)

# rch loader
OLD_IMPORT = '''from sampling import FlowSampler, VPSampler, initial_noise, integrate  # noqa: E402'''
NEW_IMPORT = '''from guidance import ResidualCalibrationHead  # noqa: E402
from sampling import FlowSampler, VPSampler, initial_noise, integrate  # noqa: E402


def load_rch(path, dev):
    """Haimo v1's fitted residual head, restored exactly as fit_rch.py saved it."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = ResidualCalibrationHead(**ck.get("kwargs", {}))
    head.load_state_dict(ck["state_dict"])
    return head.to(dev).eval()'''
assert s.count(OLD_IMPORT) == 1, "import anchor"
s = s.replace(OLD_IMPORT, NEW_IMPORT)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance_sweep.py: stage v2 wired in")
