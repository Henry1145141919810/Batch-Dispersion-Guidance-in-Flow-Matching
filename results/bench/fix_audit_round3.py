"""Three fixes from the completeness audit. Run once.

S1 [SHIPPED BUG] `strength_scale` keys off `arm.startswith("btvg")`, so the SHG
   schedules that CONTAIN a btvg phase (`shg_plug_btvg`, `shg_three`) never got
   the normalisation. Measured: the identical BTVG field runs **3139x stronger**
   inside the schedule than in the standalone arm at the same nominal w --
   guaranteed clip-saturated at every point of STRENGTHS, which is exactly what
   strength_scale exists to prevent.
   A single scalar `w` cannot scale two phases differently, so the fix belongs
   in the schedule's own per-phase multiplier, which `active()` already applies.

D1 [SHIPPED BUG] `band` returns a STATE DISPLACEMENT -- `tolerance_band_step`
   constrains ||d|| <= R and treats b'd as the property change, exactly like
   spbc -- but `band` was not in DISPLACEMENT_MODES, so FlowSampler multiplied
   it by w(1-t)/t: x19 at t=0.05, x0.053 at t=0.95. That is the most plausible
   mechanical cause of the recorded "band diverges above w=0.05".

K1 The kappa3 skew probe is implemented (`kappa3_skew`), unit-tested, and
   ranked #1 in the TOP6 memo -- and `guidance_sweep.py` never switches it on,
   so it has never produced a number. It measures whether the property law is
   skewed enough to invalidate the Gaussian closure every SMG-family arm
   assumes, which is the programme's stated go/no-go. Behind a flag because it
   costs an extra third-cumulant evaluation per step.
"""
import io

NL = chr(10)

# ---------------------------------------------------------------- D1
p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()
OLD = '''DISPLACEMENT_MODES = {"spbc"}'''
NEW = '''# Arms whose field is already a STATE DISPLACEMENT in x_t, not a score. The
# sampler must NOT apply the (1-t)/t score-to-velocity conversion to these.
# `band` belongs here for the same reason `spbc` does: tolerance_band_step
# solves ||d|| <= R subject to a property-change constraint on b'd, so d is a
# displacement. Omitting it multiplied band by 19 at t=0.05 and 0.053 at
# t=0.95 -- the likely mechanical cause of its recorded divergence above
# w=0.05.
DISPLACEMENT_MODES = {"spbc", "band"}'''
assert s.count(OLD) == 1, "displacement anchor"
s = s.replace(OLD, NEW)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---------------------------------------------------------------- S1 + K1
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

OLD_S = '''def strength_scale(arm, kw, s):'''
NEW_S = '''def scale_schedule(sched, s, tau):
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


def strength_scale(arm, kw, s):'''
assert s.count(OLD_S) == 1, "strength_scale anchor"
s = s.replace(OLD_S, NEW_S)

OLD_W = '''        s = f_A.y_std
        w_scale = strength_scale(arm, extra, s)
        w_applied = w * w_scale'''
NEW_W = '''        s = f_A.y_std
        w_scale = strength_scale(arm, extra, s)
        w_applied = w * w_scale
        if arm in SHG_SCHEDULES:
            # normalise the btvg phases individually; the arm-level scale stays
            # 1.0 so the plug/smg/spbc phases are untouched
            extra["schedule"] = scale_schedule(extra["schedule"], s,
                                               extra.get("tau"))'''
assert s.count(OLD_W) == 1, "w_scale anchor"
s = s.replace(OLD_W, NEW_W)

OLD_K = '''    ap.add_argument("--preflight", action="store_true",'''
NEW_K = '''    ap.add_argument("--kappa3", action="store_true",
                    help="log the standardised skew gamma = k3[g,g,g]/(g'Sg)^1.5 "
                         "of the property law on every guided step. This is the "
                         "programme's stated go/no-go -- it measures whether the "
                         "Gaussian closure every SMG-family arm assumes is valid "
                         "-- and it has never been switched on. Costs one extra "
                         "third-cumulant evaluation per step.")
    ap.add_argument("--preflight", action="store_true",'''
assert s.count(OLD_K) == 1, "kappa3 flag anchor"
s = s.replace(OLD_K, NEW_K)

OLD_KW = '''                      sigma_mc=args.sigma_mc, **extra)'''
NEW_KW = '''                      sigma_mc=args.sigma_mc,
                      want_kappa3=args.kappa3, **extra)'''
assert s.count(OLD_KW) == 1, "sampler kwargs anchor"
s = s.replace(OLD_KW, NEW_KW)

OLD_ACC = '''            for dk, dv in smp.diag_summary().items():'''
NEW_ACC = '''            if args.kappa3 and smp.kappa3_log:
                import statistics
                allk = [float(v) for t in smp.kappa3_log
                        for v in t.reshape(-1).tolist()
                        if v == v and abs(v) < 1e6]
                if allk:
                    k3_acc.append((statistics.fmean(allk),
                                   statistics.fmean([abs(x) for x in allk])))
            for dk, dv in smp.diag_summary().items():'''
assert s.count(OLD_ACC) == 1, "diag accumulate anchor"
s = s.replace(OLD_ACC, NEW_ACC)

OLD_INIT = '''        sched_log, diag_log = {}, {}'''
NEW_INIT = '''        sched_log, diag_log, k3_acc = {}, {}, []'''
assert s.count(OLD_INIT) == 1, "init anchor"
s = s.replace(OLD_INIT, NEW_INIT)

OLD_REC = '''                  "w_applied": w_applied, "w_scale": w_scale,'''
NEW_REC = '''                  "w_applied": w_applied, "w_scale": w_scale,
                  "kappa3_mean": (sum(a for a, _ in k3_acc) / len(k3_acc)
                                  if k3_acc else None),
                  "kappa3_abs_mean": (sum(b for _, b in k3_acc) / len(k3_acc)
                                      if k3_acc else None),'''
assert s.count(OLD_REC) == 1, "record anchor"
s = s.replace(OLD_REC, NEW_REC)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("S1 (SHG btvg phase scale), D1 (band displacement), K1 (--kappa3) applied")
