"""Fixes from the fix-verification audit. Run once.

A [REGRESSION I CAUSED] Adding `band` to DISPLACEMENT_MODES rewrote the exact
  line that was a mutation anchor, so that mutation reported ANCHOR MISSING and
  the suite dropped to 28/29. Anchor updated.

B [MY RATIONALE WAS FALSE] The comment claimed `band` is a displacement "for
  the same reason spbc does". Measured: band's `d` is eta*u orthogonally
  PROJECTED onto the feasible set (||d||/||u|| = 0.3833 = cos(d,u) to 4 digits,
  and scaling eta 1->10 scales ||d|| exactly 1->10). spbc's edit is
  (nu/r)*a with nu a physical property increment -- that IS a length. band's
  is one only if eta carries units of [x]^2. The classification still stands,
  but for a different reason: u is grad D, not grad log p(y|x), so the
  score-to-velocity factor derived for the Gaussian path has no justification
  here either way.
  The causal claim is also REFUTED: with clip=None band goes non-finite at
  w>=0.05 both before and after the fix, and with the sweep's clip=1.0 neither
  version diverges. Divergence is a magnitude problem, not a (1-t)/t problem.

C [PREFLIGHT DID NOT EXERCISE THE SCHEDULES] steps=4 gives the time grid
  {0, .25, .5, .75}; with t_min_guide=0.5 the btvg phase (starts 0.80) and the
  spbc phases (start 0.85/0.90) were NEVER entered. Proof: shg_plug_btvg's
  preflight MAE equalled plug's exactly. steps=10 hits every interval.

D kappa3 reporting: record the flag so `null` is unambiguous, and clear
  kappa3_log in reset_counts for consistency with every other counter.

E [1.76 GPU-h OF PROVABLY IDENTICAL CELLS] After the (tau/s)^2 normalisation,
  btvg_mean's coefficient is w(y-f)/s^2 -- bit-identical to `plug` at 9.0e-8
  relative. And full btvg is nearly tau-invariant (1.5e-5 spread over a 16x
  range in tau^2) because tau^2/V ~ 1e-5, so the TAU_MULT sweep cannot answer
  its own question as parameterised. btvg_mean is dropped from the sweep (the
  fact is recorded in the methods index instead) and TAU_MULT collapses to the
  pre-registered value.

F [btvg AND band READ THE CLIP, NOT THE ARM] At t_min=0.5 the clip fraction of
  guided sample-steps is btvg 0.08..0.78 and band 0.00..1.00 across w=0.01..4,
  against spbc 0.00 through w=2. Only 2 of 7 grid points are informative for
  those two. A low-strength extension is added for them alone.
"""
import io

NL = chr(10)

# ---------------------------------------------------------------- A
p = "results/bench/mutation_test.py"
s = io.open(p, encoding="utf-8").read()
OLD = '''    ("SPBC removed from DISPLACEMENT_MODES (double (1-t)/t scaling)",
     'DISPLACEMENT_MODES = {"spbc"}',
     "DISPLACEMENT_MODES = set()"),'''
NEW = '''    ("SPBC removed from DISPLACEMENT_MODES (double (1-t)/t scaling)",
     'DISPLACEMENT_MODES = {"spbc", "band"}',
     "DISPLACEMENT_MODES = set()"),'''
assert s.count(OLD) == 1, "mutation anchor"
s = s.replace(OLD, NEW)
# the env var that does not work on this box
s = s.replace('CUDA_VISIBLE_DEVICES=""', 'CUDA_VISIBLE_DEVICES="-1"')
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---------------------------------------------------------------- B
p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()
OLD_C = '''# Arms whose field is already a STATE DISPLACEMENT in x_t, not a score. The
# sampler must NOT apply the (1-t)/t score-to-velocity conversion to these.
# `band` belongs here for the same reason `spbc` does: tolerance_band_step
# solves ||d|| <= R subject to a property-change constraint on b'd, so d is a
# displacement. Omitting it multiplied band by 19 at t=0.05 and 0.053 at
# t=0.95 -- the likely mechanical cause of its recorded divergence above
# w=0.05.
DISPLACEMENT_MODES = {"spbc", "band"}'''
NEW_C = '''# Arms the sampler must NOT push through the (1-t)/t score-to-velocity factor.
#
# `spbc` because its edit IS a length: d_i = (nu/r_i) a_i with nu a physical
# property increment and r_i the measured response.
#
# `band` for a different reason, and NOT because its output is a length -- it
# is measured to be eta*u orthogonally projected onto the feasible set
# (||d||/||u|| = cos(d,u) = 0.3833 to four digits), so it is a length only if
# eta carries units of [x]^2. It belongs here because its input `u` is
# grad(diversity), not grad log p(y|x_t): the (1-t)/t factor is derived for the
# Gaussian path's SCORE and has no justification for a non-score direction
# either way, so not applying it is the choice consistent with this module's
# convention.
#
# NOTE: this does NOT explain band's recorded divergence. Measured: with
# clip=None band goes non-finite at w >= 0.05 both with and without the factor,
# and with the sweep's clip=1.0 neither version diverges at any strength.
# Divergence is a magnitude problem. The factor does change the magnitude a
# lot (x19 at t=0.05, x0.053 at t=0.95) and it makes band MORE clip-bound at
# t_min=0.5, which is why the low-strength grid extension exists.
DISPLACEMENT_MODES = {"spbc", "band"}'''
assert s.count(OLD_C) == 1, "displacement comment anchor"
s = s.replace(OLD_C, NEW_C)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)

# ---------------------------------------------------------------- C, D, E, F
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

OLD_PF = '''        args.n, args.batch, args.steps = 8, 8, 4'''
NEW_PF = '''        # steps=10, not 4. At steps=4 the time grid is {0, .25, .5, .75}, so
        # with t_min_guide=0.5 the SHG handoffs at 0.80/0.85/0.90 were NEVER
        # entered and shg_plug_btvg's preflight MAE came out exactly equal to
        # plug's. steps=10 puts a node in every scheduled interval.
        args.n, args.batch, args.steps = 8, 8, 10'''
assert s.count(OLD_PF) == 1, "preflight anchor"
s = s.replace(OLD_PF, NEW_PF)

OLD_K = '''                  "kappa3_mean": (sum(a for a, _ in k3_acc) / len(k3_acc)'''
NEW_K = '''                  "kappa3": args.kappa3,
                  "kappa3_mean": (sum(a for a, _ in k3_acc) / len(k3_acc)'''
assert s.count(OLD_K) == 1, "kappa3 record anchor"
s = s.replace(OLD_K, NEW_K)

OLD_TM = '''TAU_MULT = [0.5, 1.0, 2.0]          # x (delta / 1.96)'''
NEW_TM = '''# COLLAPSED TO THE PRE-REGISTERED VALUE. The tau sweep cannot answer its own
# question as parameterised: after the (tau/s)^2 strength normalisation the
# btvg coefficient is w*b = -0.5w/s^2 * (1 - tau^2/V), and the measured
# tau^2/V ~ 1e-5, so a 16x range in tau^2 moves the field by 1.5e-5 relative.
# Twelve cells were reproducing their own siblings. Restore the list only if
# the normalisation changes.
TAU_MULT = [1.0]                    # x (delta / 1.96)

# Arms whose strength curve reads the CLIP rather than the arm on the standard
# grid. Measured clip fraction of guided sample-steps at t_min=0.5, w=0.01..4:
#   band 0.00 0.22 0.71 0.89 0.97 1.00 1.00
#   btvg 0.08 0.27 0.46 0.52 0.62 0.74 0.78
#   spbc 0.00 ...................... 0.00 (through w=2)
# Only the two lowest points are informative for band/btvg, so they get two
# extra points below the shared grid.
CLIP_BOUND = ("band", "btvg", "btvg_var")
STRENGTHS_LOW = [0.002, 0.005]'''
assert s.count(OLD_TM) == 1, "tau_mult anchor"
s = s.replace(OLD_TM, NEW_TM)

OLD_V2 = '''V2_ARMS = ["spbc", "btvg", "btvg_mean", "btvg_var"]'''
NEW_V2 = '''# btvg_mean is NOT here: after the (tau/s)^2 normalisation its coefficient is
# w(y-f)/s^2, which is `plug` -- bit-identical at 9.0e-8 relative. That is a
# fact worth stating in the paper (BTVG's novelty is entirely in the variance
# term) and 30 cells not worth spending. It remains implemented and gated.
V2_ARMS = ["spbc", "btvg", "btvg_var"]'''
assert s.count(OLD_V2) == 1, "v2_arms anchor"
s = s.replace(OLD_V2, NEW_V2)

OLD_ST = '''    for prop in props:                       # pass 3: strength
        for arm in sel:
            for w in STRENGTHS:
                for v in variants(arm, single=True):
                    add(prop, arm, "q50", w, V2_WIN, v)'''
NEW_ST = '''    for prop in props:                       # pass 3: strength
        for arm in sel:
            grid = STRENGTHS + (STRENGTHS_LOW if arm in CLIP_BOUND else [])
            for w in sorted(grid):
                for v in variants(arm, single=True):
                    add(prop, arm, "q50", w, V2_WIN, v)'''
assert s.count(OLD_ST) == 1, "pass3 anchor"
s = s.replace(OLD_ST, NEW_ST)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("A(mutation anchor) B(band rationale) C(preflight steps) D(kappa3 flag) "
      "E(btvg_mean+TAU_MULT) F(low strengths) applied")
