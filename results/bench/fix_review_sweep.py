"""Adversarial-review fixes to guidance_sweep.py: D6, D3, D13, D14. Run once.

D6 [the worst of these] Stage v2 scheduled NO plug / tmpd / smg cells. The
   filter was `[a for a in V2_REFERENCES if a in arms or a == "unguided"]` and
   for --stage v2 `arms` is V1_MISSING + V2_ARMS + SHG_ARMS, which contains
   none of them -- so the only reference that survived was `unguided`. The
   docstring promised the opposite ("re-runs these in the same cells ... rather
   than read off a table from another job").
   Worse, fixing it naively would have collided with stage-main: `cell_name`
   has no stage field, so 9 v2 reference names matched stage-main names and 5
   ALREADY EXIST on disk -- they would have been skipped as "already done" and
   silently compared at the wrong window. Reference cells now carry an explicit
   `v2ref` variant so they are genuinely re-run under the v2 window and seed.

D3 BTVG's mean coefficient is 1/tau^2 where plug's is 1/s^2, and the measured
   ratio s^2/tau^2 is 322 (mu), 1115 (alpha), 150 (gap) at TAU_MULT=1 -- up to
   4460 at TAU_MULT=0.5. With STRENGTHS bottoming out at 0.01 the weakest btvg
   cell was still 1.5-11x plug at w=1, and the measured field-to-velocity ratio
   ran 30-660. Every btvg cell in the grid was clip-saturated, so the ablation
   would have compared clipped DIRECTIONS, not the objective. The strength is
   now normalised by (tau/s)^2 so w means the same thing for btvg as for plug,
   and both the nominal and applied values are recorded.

D13 The accumulated diagnostics now reach the cell JSON.

D14 arm_kwargs("btvg", "-", d) and arm_kwargs("spbc", "-", d) both raised
    IndexError, reachable via `--stage main --arms spbc`, and the try/except
    turned it into a .failed cell retried forever.
"""
import io

NL = chr(10)
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- D14 + D3
OLD = '''def arm_kwargs(arm, variant, delta):
    """Sampler kwargs implied by (arm, variant). One place, so the sweep and
    the final benchmark cannot drift apart."""
    kw = {}
    if arm == "rch":
        pass                       # handled in run_cell, which has the device
    elif arm.startswith("btvg"):
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
NEW = '''def arm_kwargs(arm, variant, delta):
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
    return 1.0'''
assert s.count(OLD) == 1, "arm_kwargs anchor"
s = s.replace(OLD, NEW)

# ---------------------------------------------------------------- D6
OLD_R = '''    sel = [a for a in arms if a not in ("unguided",)]
    refs = [a for a in V2_REFERENCES if a in arms or a == "unguided"]'''
NEW_R = '''    sel = [a for a in arms if a not in ("unguided",)]
    # References are ALWAYS scheduled, whatever `arms` says: the point of the
    # stage is to measure the new arms against them under the same seed, window
    # and target, and filtering them by `arms` left only `unguided`.
    refs = list(V2_REFERENCES)'''
assert s.count(OLD_R) == 1, "refs anchor"
s = s.replace(OLD_R, NEW_R)

OLD_V = '''    def variants(arm, single=False):
        """Hyperparameter settings for an arm; `single` returns only the
        pre-registered default so pass 1 stays one cell per arm."""'''
NEW_V = '''    REF_TAG = "v2ref"          # keeps reference cells off the stage-main names

    def variants(arm, single=False):
        """Hyperparameter settings for an arm; `single` returns only the
        pre-registered default so pass 1 stays one cell per arm."""
        if arm in V2_REFERENCES:
            # A reference cell must not collide with a stage-main cell: they
            # share (prop, arm, target, w) and cell_name has no stage field, so
            # 5 of them already exist on disk and would be skipped as done --
            # comparing the v2 arms against numbers from a different window.
            return [REF_TAG]'''
assert s.count(OLD_V) == 1, "variants anchor"
s = s.replace(OLD_V, NEW_V)

OLD_P1 = '''    for prop in props:                       # pass 1: the comparison
        for arm in refs:
            add(prop, arm, "q50", DEFAULT_W, V2_WIN)'''
NEW_P1 = '''    for prop in props:                       # pass 1: the comparison
        for arm in refs:
            add(prop, arm, "q50", DEFAULT_W, V2_WIN, REF_TAG)'''
assert s.count(OLD_P1) == 1, "pass1 anchor"
s = s.replace(OLD_P1, NEW_P1)

OLD_P3 = '''        for arm in refs:
            if arm == "unguided":
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, V2_WIN)'''
NEW_P3 = '''        for arm in refs:
            if arm == "unguided":
                continue
            for w in STRENGTHS:
                add(prop, arm, "q50", w, V2_WIN, REF_TAG)'''
assert s.count(OLD_P3) == 1, "pass3 anchor"
s = s.replace(OLD_P3, NEW_P3)

OLD_P4 = '''    for prop in props:                       # pass 4: the q90 target
        for arm in refs:
            add(prop, arm, "q90", DEFAULT_W, V2_WIN)'''
NEW_P4 = '''    for prop in props:                       # pass 4: the q90 target
        for arm in refs:
            add(prop, arm, "q90", DEFAULT_W, V2_WIN, REF_TAG)'''
assert s.count(OLD_P4) == 1, "pass4 anchor"
s = s.replace(OLD_P4, NEW_P4)

# ---------------------------------------------------------------- D3 applied
OLD_W = '''            kw = dict(f_net=(None if arm == "unguided" else f_A),
                      y=y_t[: m.shape[0]], s=s, mode=arm, w=w, clip=clip,
                      n_probe=args.n_probe, n_mc=args.n_mc,
                      sigma_mc=args.sigma_mc, **extra)'''
NEW_W = '''            kw = dict(f_net=(None if arm == "unguided" else f_A),
                      y=y_t[: m.shape[0]], s=s, mode=arm, w=w_applied,
                      clip=clip, n_probe=args.n_probe, n_mc=args.n_mc,
                      sigma_mc=args.sigma_mc, **extra)'''
assert s.count(OLD_W) == 1, "kw anchor"
s = s.replace(OLD_W, NEW_W)

OLD_T = '''        s = f_A.y_std'''
NEW_T = '''        s = f_A.y_std
        w_scale = strength_scale(arm, extra, s)
        w_applied = w * w_scale'''
assert s.count(OLD_T) == 1, "y_std anchor"
s = s.replace(OLD_T, NEW_T)

# ---------------------------------------------------------------- D12 + D13
OLD_ACC = '''            for k in cost:
                cost[k] += getattr(smp.cost, k)
            clipped += smp.n_clipped'''
NEW_ACC = '''            for k in cost:
                cost[k] += getattr(smp.cost, k)
            clipped += smp.n_clipped
            # accumulate across batches: a fresh sampler is built per batch, so
            # reading these off the last one described 128 of 512 samples
            for mk, mv in smp.schedule_log.items():
                sched_log[mk] = sched_log.get(mk, 0) + mv
            for dk, dv in smp.diag_summary().items():
                tot, cnt = diag_log.get(dk, (0.0, 0))
                diag_log[dk] = (tot + dv, cnt + 1)'''
assert s.count(OLD_ACC) == 1, "accumulate anchor"
s = s.replace(OLD_ACC, NEW_ACC)

OLD_INIT = '''        clipped = 0'''
NEW_INIT = '''        clipped = 0
        sched_log, diag_log = {}, {}'''
assert s.count(OLD_INIT) == 1, "clipped init anchor"
s = s.replace(OLD_INIT, NEW_INIT)

OLD_REC = '''                  "variant": variant, "stage": args.stage,
                  "schedule_used": getattr(smp, "schedule_log", {}),'''
NEW_REC = '''                  "variant": variant, "stage": args.stage,
                  "schedule_used": sched_log,
                  "w_applied": w_applied, "w_scale": w_scale,
                  "diag": {dk: tot / cnt for dk, (tot, cnt) in diag_log.items()},'''
assert s.count(OLD_REC) == 1, "record anchor"
s = s.replace(OLD_REC, NEW_REC)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance_sweep.py: D3, D6, D13, D14 applied")
