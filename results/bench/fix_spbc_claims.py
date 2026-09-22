"""D2 + D7: correct SPBC's docstring to what was measured. Run once.

The docstring claimed the edit "leaves every centered quantile and every
pairwise difference unchanged" and that nu = -eta(mean f - y*) moves the mean
to y*. Measured end to end on the real generator (alpha, B=16, 40 steps,
t_min_guide 0.5, w = eta = 1):

  max |change in centred f_A|   3.868 = 8.03 delta   (batch sd of f_A = 13.76)
  max change in a pairwise diff 5.932 = 12.3 delta
  mean moved                    +0.918 of the +4.27 needed

So the absolute claim is false. It is NOT caused by clipping (0/320 sample-
steps clipped; clip=1.0 and clip=None gave bit-identical output) nor by the
zero-COM projection (retention 1.0000). It is accumulated linearisation error:
a_i . d_i = nu holds exactly at every step, but f(x_1) does not follow the
first-order forecast over 20 guided steps.

The RELATIVE claim survives and is the one worth making: distortion per unit
of mean shift is 4.2 (spbc w=1) and 2.9 (spbc w=4) against 14.5 / 21.0 / 20.7 /
8.8 for plug at w = 0.002 / 0.01 / 0.05 / 0.25 -- roughly 3x less shape
distortion for the same movement of the mean.

D7 also: the increment is applied as a RATE (the integrator multiplies the
field by dt), so the realised closure is 1 - exp(-w*eta*(1-t_min)), measured at
49.0 / 43.6 / 53.9 / 55.2 % for 10 / 20 / 40 / 80 steps -- step-count invariant,
which is the property that matters, but about half of what the docstring said.
"""
import io

NL = chr(10)
p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''    THE MECHANISM. For any scalar outcome law, subtracting the bias
    b = E[Y] - y* leaves the mean at y* and leaves **every centered quantile
    and every pairwise difference unchanged** (memo section 3.4). So the ideal
    edit gives EVERY trajectory the SAME property increment

        nu_i = d = -eta * (mean_i f(m_i) - y*)'''
NEW = '''    THE MECHANISM. For any scalar outcome law, subtracting the bias
    b = E[Y] - y* leaves the mean at y* and leaves **every centered quantile
    and every pairwise difference unchanged** (memo section 3.4). So the ideal
    edit gives EVERY trajectory the SAME property increment

        nu_i = d = -eta * (mean_i f(m_i) - y*)

    WHAT ACTUALLY HAPPENS, MEASURED -- the paragraph above is the ideal, not
    the result. End to end on the real generator (alpha, B=16, 40 steps,
    t_min 0.5, w = eta = 1): max change in a centred f_A value is 3.868
    = 8.03 delta, and max change in a pairwise difference is 5.932 = 12.3
    delta, against a batch sd of 13.76. The shape is NOT preserved in absolute
    terms.

    The cause is neither clipping (0/320 sample-steps clipped at the swept
    strengths; clip=1.0 and clip=None give bit-identical output) nor the
    zero-COM projection (retention 1.0000). a_i . d_i = nu holds exactly at
    every step -- that is what the gate measures -- but f(x_1) does not follow
    the first-order forecast once 20 such steps are composed.

    THE DEFENSIBLE CLAIM IS THE RELATIVE ONE: distortion per unit of mean
    shift is 4.2 (w=1) and 2.9 (w=4) against 14.5 / 21.0 / 20.7 / 8.8 for
    `plug` at w = 0.002 / 0.01 / 0.05 / 0.25 -- roughly 3x less shape
    distortion for the same movement of the mean. Write it that way.

    AND THE INCREMENT IS A RATE, NOT A ONE-SHOT EDIT. The integrator multiplies
    the returned field by dt, so nu is dnu/dt and the realised closure of the
    bias is 1 - exp(-eta * w * (1 - t_min)), i.e. about half at the
    pre-registered eta = w = 1, t_min = 0.5. Measured: 49.0 / 43.6 / 53.9 /
    55.2 % at 10 / 20 / 40 / 80 steps -- step-count invariant, which is the
    property that matters, but NOT the full correction. w = 4 gives 147%
    (overshoot). Tune w, do not assume one pass centres the batch.

    FINALLY, `mu` IS A MINIBATCH MEAN. The sweep runs batch=128 within n=512,
    so four different increments are applied while evaluation pools all 512 --
    the common-increment property holds within a batch, not across the sample.
    At sd(f_A) = 13.76 the standard error of that batch mean is ~1.22 alpha
    units, about 2.5 delta. Raise the batch, or read the spread accordingly.'''
assert s.count(OLD) == 1, "spbc mechanism anchor"
s = s.replace(OLD, NEW)

OLD2 = '''    HONEST LIMITS. The forecast is the cheap one, f_A(m_t), not a rollout
    through the sampler suffix; it inherits the off-distribution error measured
    in FINDING_QUADRATIC_CLOSURE_VALIDITY.md. Preserving scalar property
    spacings does NOT imply preserving structural diversity. And centering is
    not guaranteed to raise band coverage -- a skewed law can lose coverage
    when centred (memo section 3.5), so it needs an acceptance check.'''
NEW2 = '''    HONEST LIMITS. The forecast is the cheap one, f_A(m_t), not a rollout
    through the sampler suffix; it inherits the off-distribution error measured
    in FINDING_QUADRATIC_CLOSURE_VALIDITY.md -- and the 8-delta shape drift
    above is that error, compounded. Preserving scalar property spacings does
    NOT imply preserving structural diversity. And centering is not guaranteed
    to raise band coverage -- a skewed law can lose coverage when centred
    (memo section 3.5), so it needs an acceptance check.'''
assert s.count(OLD2) == 1, "spbc limits anchor"
s = s.replace(OLD2, NEW2)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("spbc_displacement: claims corrected to what was measured")
