# BTVG after the negative results: what to repair, what to test, what to stop claiming

**23 September 2026.** Evidence audit and research decision, not a claim that a new
method has won. BVTG in discussion refers to the repository's `btvg` / BTVG.

**Later update:** [Evolving after BTVG2 and the chemistry guard](BTVG_EVOLUTION_AFTER_CHEM_GUARD.md)
incorporates the completed guard failures, corrected decoded scoring, new
discrete-feasibility measurements and an executed one-atom repair control.
It supersedes the next-step priority below; the historical audit remains valid.

**Subsequent claim audit:** [new statistical and trajectory checks](../results/WORKSHOP_CLAIM_AUDIT.md)
retain the LGD-MC deficit, distinguish a registered "tie" from equivalence, and
show that tau's threshold does engage late in sampling. The actual-state
projection is now implemented and gated; molecular outcome testing remains
pending. Use that audit for current paper wording and completion requirements.

## Decision

Keep the frozen generator, split protocol, independent evaluator, LGD-MC draws,
and paired evaluation infrastructure. **Do not launch another broad sweep of the
existing variance penalty.** BTVG's independent contribution is unproven, and
BTVG2 does not test all the protections its interpretation assumes.

There are two separate tasks:

1. Repair the actual-state geometry and measure whether the variance direction
   has any useful effect beyond LGD-MC. This is a correctness ablation.
2. Investigate **terminal band success under a chemistry and compute budget**.
   A prospective method must demonstrate a benefit attributable to its variance
   component beyond a band-loss baseline and a chemistry-constrained baseline.

No amount of derivation can ensure an empirical win or global novelty. A useful
outcome of this investigation may be a well-supported negative result. Calling
LGD-MC plus familiar constraints “BTVG3” would not resolve that.

## 1. What the current outcomes actually establish

The registered full run has 5,000 targets and three seeds. These are the primary
FR3a strengths, not an equal-strength comparison:

| property | BTVG in-band | LGD-MC in-band | registered comparison |
|---|---:|---:|---|
| mu | 0.081 | 0.104 | LGD-MC ahead, 6.83 independent standard errors |
| alpha | 0.065 | 0.080 | LGD-MC ahead, 5.25 |
| gap | 0.116 | 0.152 | LGD-MC ahead, 9.16 |

BTVG ties unguided on the registered coverage test. Its alpha MAE signal at
strength 4 is a secondary, one-seed observation with a marginal chemistry pass;
it does not overturn the primary result. See [full table](../../full_run_table.md)
and the latest sections of [the living status](../status/SCOPE_FM_GUIDANCE_STATUS.md).

The BTVG2 pilot already compares against locally rerun LGD-MC using the same
targets and noise. The useful question is the incremental effect of the added
term, not its large chemistry improvement over the original BTVG estimator.
Recomputing intersections from the saved per-molecule records gives:

| property, w=8 | LGD-MC in-band | BTVG2 in-band | LGD-MC valid AND in-band | BTVG2 valid AND in-band |
|---|---:|---:|---:|---:|
| mu | 12.35% | 11.18% | 9.47% | 8.59% |
| alpha | 10.21% | 8.84% | 7.37% | 6.54% |
| gap | 18.60% | 17.48% | 13.09% | 12.30% |

At w=8, BTVG2 rescues/loses respectively **177/201** mu hits, **118/146**
alpha hits, and **159/182** gap hits relative to LGD-MC. This is evidence of
turnover, not uniformly helpful concentration. The nine saved comparisons
(two BTVG2 strengths and one band-gated strength, three properties) all have a
negative point difference in valid-and-in-band yield. Their dependence and
single-seed selection prevent treating that as nine independent tests.

Gap at w=4 has a small positive raw coverage difference, 3/2048. Thus “no gain
anywhere” should mean **no demonstrated improvement**, not literally no
positive point estimate. LGD-MC at w=8 fails the pilot's gap chemistry floor;
the table above is an incremental same-strength diagnostic, not a declaration
that this operating point is acceptable.

These are reanalyses of existing overlapping test data, not new confirmation.
Also, “valid AND in-band” currently intersects decoded validity with a property
prediction on **continuous features**. Section 4 explains why that distinction
matters.

## 2. The central conceptual error: three different variances

Keep these quantities separate:

1. `g^T Sigma g`: a local linearized property variance under an approximate
   interpolant posterior. It is not the exact variance of a nonlinear property.
2. BTVG2's `Var_k f_A(m + r z_k)`: local predictor variation under isotropic
   probes with a detached radius. These are not actual sampler completions.
3. `Var_i[f_B(X_1^i) - y_i]`: terminal residual spread across generated samples,
   which is relevant to the `dist` task.

Under an appropriate true joint law, the law of total variance decomposes
terminal residual variance into conditional variance plus variance of conditional
mean residuals. Reducing the first component alone does not constrain the second.
Changing the controller also changes the state distribution, so a local derivative
is not a comparison of final population variances.

An even sharper distinction applies to our deterministic sampler: **given its
state, history and fixed future draws, its continuation has one endpoint**.
The training interpolant's conditional distribution of clean data is a different
coupling. Nonzero uncertainty of that posterior is not random future branching
of this ODE. At t=1 the local probe/posterior variance can vanish while the
terminal error distribution remains arbitrarily broad.

For `dist`, controlling raw property variance is especially misleading because
each molecule has a different target. Correctly following a broad target
distribution requires broad generated properties and narrow **residuals**.
Population signed bias can also cancel: the unguided alpha full-run bias is only
0.015 delta while MAE is 11.824 delta. An accurately centered population is not
evidence that individual targets have been reached.

With a fixed probe radius and an affine property, BTVG2's variance is independent
of m and its variance gradient is zero. Yet ordinary mean guidance can still
eliminate arbitrary target error. For nonlinear properties this term primarily
seeks lower local predictor sensitivity. That may help, but can also seek flat
off-distribution regions; it does not acquire a terminal-success interpretation
merely by being called uncertainty reduction.

### Concentration can be the wrong action even with a perfect Gaussian model

Let the property residual follow N(e,V), with allowed band [-delta,delta]. Then

```
p_band(e,V) = Phi((delta-e)/sqrt(V)) - Phi((-delta-e)/sqrt(V)).

d p_band/dV = [(e-delta) phi((delta-e)/sqrt(V))
              - (e+delta) phi((-delta-e)/sqrt(V))] / (2 V^(3/2)).
```

For e=0, reducing V helps. For e>delta it can hurt. In particular e=3 delta:
shrinking standard deviation from delta to delta/2 reduces band probability
from **0.0227185 to 0.0000316712**, about **717 times**. For e>delta the
variance derivative changes sign at

```
V* = 2 e delta / log((e+delta)/(e-delta)).
```

Below V*, widening increases band probability at fixed mean. Therefore the old
argument “likelihood sometimes widens, so it is wrong for design” is not valid
for a band-success objective. This does not recommend indiscriminate widening;
it rejects indiscriminate shrinking. A Gaussian band score is itself a closure,
not a cure for an inaccurate predictive distribution.

K=4 is also a weak basis for a state-dependent variance decision. Even for
independent Gaussian draws, the unbiased sample variance has relative standard
deviation sqrt(2/(K-1)) = **0.816**. Ratios, inverse variance and gates inherit
additional nonlinear noise. Repeating fixed-state probes is more informative
than immediately increasing K in every production step.

## 3. A newly verified failure in BTVG2's geometry

Write g=grad_m mu_hat, h=grad_m V_hat, J=dm/dx. Ignoring its nonnegative
gate/cap multiplier, BTVG2 constructs

```
S_m = -(I - g g^T/(g^T g)) h,       S_x = J^T S_m.
```

The first expression satisfies g^T S_m=0. But applying S_x to x changes
the mean at rate

```
(J^T g)^T S_x = g^T J J^T S_m,
```

which need not vanish. Similarly h^T S_m<=0 does not imply
(J^T h)^T S_x<=0. With state masking/centering Q, replace J J^T by J Q J^T.

**Exact counterexample.** Take J=diag(3,1), g=(1,1), h=(1,2).
Then S_m=(0.5,-0.5) is mean-orthogonal and decreases V in m-space.
After pullback S_x=(1.5,-0.5), the actual mean derivative is **+4** and
variance derivative is **+3.5**. Both intended protections fail. This uses a
symmetric positive-definite J; network asymmetry is not needed.

The existing BTVG2 test suite still passes all ten checks. G3/G6 check the
m-space properties; G7 separately checks that the pullback is implemented.
Their combination never checks preservation/descent in the space actually
updated. Passing those tests does not contradict this counterexample.

### Real-checkpoint diagnostic, not a performance benchmark

I ran fresh unguided trajectories for **32 validation sizes**, seed 20260923,
and examined t=0.50,0.65,0.80,0.90 with the current f_A predictors. At t=0.80:

| property | active corrections | median abs cosine to mean gradient before pullback | after pullback | cases increasing the fixed-draw V locally |
|---|---:|---:|---:|---:|
| mu | 28 | 5.3e-8 | 0.445 | 1/28 |
| alpha | 25 | 1.1e-7 | 0.493 | 0/25 |
| gap | 25 | 5.8e-8 | 0.513 | 2/25 |

These cosines measure angular leakage, not a percentage change in property.
States are unguided; this does not estimate the failure frequency of the full
BTVG2 trajectory. It establishes that the proposed safeguard is not preserved
on the real network. All numbers and the exact counterexample are in
[mechanism.json](../../results/btvg_audit/mechanism.json).

### Correctness repair: project in the state space

For the frozen-draw, frozen-radius surrogate, set

```
a = Q J^T g,       b = Q J^T h,
S = -eta [b - a (a^T b)/(a^T a)],       eta >= 0.
```

Then a^T S=0 and b^T S=-eta ||P_a b||^2<=0. Degenerate a=0 needs an explicit
branch. The offline diagnostic's repaired directions satisfy these properties
to floating-point accuracy (maximum abs cosine below 1.8e-6).

This costs additional pullbacks; it is not free. It protects only this local
surrogate. Recomputing the probe radius adds a derivative absent from the
definition above; finite updates and the remaining sampler introduce further
changes. A non-Euclidean update rule also changes the required projection metric.
This repair is standard constrained-gradient algebra, not a novelty claim.

### The second trap: clipping the sum spends the baseline's progress

Even correct orthogonality before clipping is insufficient. If baseline
d0=(1,0), added correction S=(0,1), and radius R=1, clipping the sum gives
(1,1)/sqrt(2). Its baseline-direction progress drops from 1 to **0.7071**.

A protected correction must account for the **joint applied update**. A concrete
local control is

```
min_d  0.5 ||d-d0||^2 + lambda b^T(d-d0)
 s.t.  a^T(d-d0) = 0,
       ||d|| <= R,
       chemistry constraints with declared slack.
```

Use d0 after its baseline clip. If chemistry makes that anchor infeasible,
report infeasibility and use an explicitly defined fallback; feasibility is not
automatic. When d0 saturates the ball and is parallel to a, **no nonzero
mean-preserving addition is feasible**. The controller should return d0 rather
than silently buy variance reduction by reducing target progress. For a band
objective, replace exact mean preservation by the appropriate band envelope.
The existing ball/slab machinery in `tolerance_band_step` is a relevant starting
point. Adding a second clip afterward would invalidate the guarantee again.

## 4. Ensure that improvement belongs to a decoded molecule

`evaluation.py::evaluate_samples` determines chemistry via argmax atom types,
but calls f_A and f_B on the original continuous `feats`. Thus the same result
mixes two molecular representations. This is not evidence of cheating, and does
not retroactively erase the registered comparison. It is a scientific limitation
that a claim about useful molecules must address.

The new audit separately scores the identical endpoint with original features
and masked one-hot(argmax(features)). It uses fresh validation sizes/targets,
128 samples per cell, and unguided/LGD-MC at w=4; f_B is only called afterward.
See [decoding.json](../../results/btvg_audit/decoding.json). This small check is
for representation sensitivity, not ranking guidance arms.

| LGD-MC w=4 | soft-feature hits | those lost after decoding | new hard-feature hits | median / p90 prediction shift, in delta |
|---|---:|---:|---:|---:|
| mu | 14/128 | 4 | 1 | 0.014 / 1.000 |
| alpha | 12/128 | 3 | 2 | 0.023 / 0.485 |
| gap | 15/128 | 3 | 1 | 0.007 / 0.278 |

Most samples have small shifts; the tail and band-boundary crossings matter.
This does not establish that decoding explains BTVG2's loss: no BTVG2 endpoint
comparison was run in this representation audit. The geometry failure and the
representation issue are distinct findings.

Future candidate evaluation should retain the original protocol for continuity
and add: decoded-feature f_B error, valid AND decoded-in-band, distinct valid
decoded-in-band molecules per attempt, disconnected-fragment handling, and a
second chemistry rule. Predictor agreement is not a physical property assay;
a stronger chemistry/design claim needs an independent physical calculation on
a disclosed subset when feasible. Do not feed f_B into online correction or
proposal acceptance.

## 5. The redesign worth investigating

The scientific question should be:

> Can an inexpensive correction identify and reduce **avoidable terminal band
> failures**, while preserving the useful hits and chemistry of a strong baseline,
> enough to improve distinct usable outputs per unit of computation?

This differs from “minimize local variance,” and admits three cases: move an
outside forecast toward the band, protect an inside forecast, and decline a
correction whose response is unreliable or consumes too much chemistry. It
must be evaluated per target; avoid batch centering of mixed `dist` targets.

### 5.1 First build the cheap control that could explain the whole benefit

Keep LGD-MC's draws and pullback. Replace its point-target energy by a band loss,
for example

```
ell_delta(F-y) = 0.5 [max(0, |F-y|-delta_A)]^2 / s_A^2,
G_band = J^T sum_k softmax_k(-ell_delta) [-ell_delta'(F_k-y) g_k].
```

This encourages entry into the interval without continually pulling successful
draws toward its center. delta_A, smoothing and s_A must be fixed or tuned on
development data under a shared budget; s_A is not automatically equal to the
scoring tolerance. It is a direct use of general loss guidance, hence a **control,
not our innovation**. A smooth interval-probability objective is another control;
its tail saturation and numerical stability need checking.

A variance correction has no independent value if this alone matches it. Nor
does a chemistry constraint become a variance contribution because it is attached
to a method named BTVG.

### 5.2 Give the variance component a task it can actually justify

Use it as a candidate **residual-risk correction after targeting**, not an
unconditional force toward flat predictor regions. At sparse, prespecified
anchors, define the reference continuation R_t with the full sampler state and
future random draws fixed. Its guide forecast is F_t=f_A(R_t(x)). For actual
decoded forecasts, the hard decoder is discontinuous: use explicit finite
interventions or label a continuous relaxation as such.

Before building an adaptive controller, conduct a paired intervention audit:

1. Save states from the tuned LGD-MC baseline.
2. Continue (a) unchanged; (b) with the current correction; (c) with the
   state-space and joint-budget correction. Use identical future draws.
3. Measure local V change, terminal guide change, decoded evaluator residual,
   rescued/lost hits, per-sample chemistry transitions and total cost.
4. Repeat probes independently at saved states. Determine whether a larger local
   variance predicts terminal failures and, separately, whether **reducing it
   causes fewer failures**. Correlation alone is insufficient.

Do not estimate “continuation uncertainty” by rerunning the same deterministic
suffix. If perturbations are used, state their distribution and interpretation:
numerical sensitivity, guide uncertainty, and clean-data posterior uncertainty
are different quantities. Independent probe folds reduce same-draw overfitting
but do not make the probes actual posterior samples.

If these interventions reveal a reproducible useful regime, fit/freeze a simple
selection rule on development data. Allow a correction only where the rule
predicts net terminal benefit within its cost/chemistry budget; otherwise use
the baseline. Calibration is training of a controller, even if the generator
is frozen, and must be disclosed rather than described as entirely training-free.

**Candidate incremental contribution:** a tested relation between local
variance interventions, their transported terminal effects and available
chemical/clip slack, used to retain useful concentration while rejecting harmful
edits. This is a hypothesis. A property-error gate, a time schedule, a norm
shrinkage baseline, and the same selector applied to a non-variance direction
must fail to explain its benefit before the variance mechanism earns credit.

This is deliberately an extension of the repo's earlier terminal-response
ideas, not a newly named invention. Full suffix evaluation belongs first in the
diagnostic; it is too expensive to assume in every sampling step. If practical
control requires that cost, it must beat spending the same compute on additional
LGD-MC attempts or best-of-N selection using f_A and chemistry only.

### 5.3 Chemistry protection is useful, but must be independently ablated

The current working tree already includes `soft_valence_violation` and
`chem_safe_project`. Treat these as a comparator. A half-space constraint on
the sum of squared soft valence errors gives a first-order statement about that
sum, not a guarantee of decoded validity. It can trade a large improvement at
one atom for a damaging change at another; type and bond decoding can jump.

Compare the same shield on LGD-MC/band-LGD and on the candidate. Record how much
property progress it removes. A rise above unguided stability is not inherently
a red flag: removing harmful components changes the subsequent trajectory and
can improve outcomes. Conversely a local nonincrease cannot guarantee an
unchanged terminal molecule. Test both assertions rather than infer them from
the sign of a dot product.

## 6. Novelty audit: what cannot be claimed

Primary sources were checked on 23 September 2026. This is a targeted search,
not proof that no equivalent algorithm exists.

| ingredient | closest verified precedent | consequence |
|---|---|---|
| MC guidance for a custom loss | [LGD-MC, ICML 2023](https://proceedings.mlr.press/v202/song23k.html) | a band-loss replacement alone is not a new principle |
| constrained extra gradients | [Harmless Diversity, ICML 2022](https://proceedings.mlr.press/v162/gong22b.html) | orthogonal/protected correction algebra is established |
| trust-limited guidance | [Trust Sampling, NeurIPS 2024](https://arxiv.org/abs/2411.10932) | adding a trust region is not novelty |
| terminal differentiation | [D-Flow, ICML 2024](https://arxiv.org/abs/2402.14017) | differentiating through a continuation is prior art |
| terminal geometry and constraints | [TOCFlow, 2026 preprint](https://arxiv.org/abs/2601.09474) | a terminal-frame correction/QP is not automatically original |
| molecular geometric constraints | [PCFM molecular extension, ICLR AI4Mat 2026 workshop](https://openreview.net/pdf?id=QVQjDmIFg0) | “chemistry-aware frozen flow” is too broad a claim |
| particle variance control | [Variance-Tilted Diffusion, 2026 preprint](https://arxiv.org/abs/2606.22239) | distinguish within-probe variance from terminal batch variance |
| target marginal shaping | [Jeffrey guidance, 2026 preprint](https://arxiv.org/abs/2606.13240) | changing the property marginal is already an explicit framework |
| future-reward lookahead | [LiDAR, ICML 2026](https://arxiv.org/abs/2602.03211) and [DSearch, 2025](https://arxiv.org/abs/2503.02039) | lookahead and branch selection are not sufficient distinctions |
| state-dependent correction allocation | [Adaptive Correction Scheduling, 2026 preprint](https://arxiv.org/abs/2605.11214) | a per-state timing gate is not automatically new |

Within this repository, [RATV](Three_New_Guidance_Ideas_Variance_and_Switching.md),
[Component D](D_WHY_INNOVATIVE.md), and
[the TRE review](../../archive/D_AND_TRE_REVIEW_FEEDBACK.md) already discuss
terminal responses, band protection and low-dimensional intervention estimates.
Reintroducing those under a new acronym would repeat earlier proposals. The new
evidence here is the **actual BTVG2 geometry failure**, its measured effect on
the current model, and the resulting more discriminating experimental design.

## 7. The smallest credible experimental program

**First, repair interpretation and the mechanism check.** Preserve historical
results. Add state-space and post-clip invariants before describing the method
as mean-preserving. Record probe reliability, clipping, intervention acceptance,
and continuous-versus-decoded disagreement.

**Second, a bounded development experiment.** Use a declared development set;
the old pilot's test[:2048] is already consumed. Audit all saved mol_idx values
before reserving a genuinely unused confirmation block. New seeds alone do not
make previously inspected targets untouched. Reusing test slices for tuning
must be disclosed as a redesignation, not called an untouched test.

Start with three comparisons, using the same estimator, strengths/windows,
target protocol and tuning budget:

| comparison | what it can establish |
|---|---|
| tuned LGD-MC vs band-loss LGD-MC | whether the objective mismatch alone explains the opportunity |
| baseline vs baseline + repaired variance | whether BTVG's component has value after its correctness repair |
| best baseline + chemistry protection vs the same + variance | whether any improvement is independently due to variance |

Tune to the declared **coverage/useful-yield objective with a chemistry floor**
for the new experiment. Do not freeze by q90 MAE and then claim optimization of
dist coverage. Keep the registered FR5 result unchanged and clearly separate
the exploratory revised objective. Use measured points on a chemistry frontier;
two-point interpolation is descriptive, not evidence of equality at a chosen cost.

**Third, confirm only a surviving candidate and its strongest controls.** Include
all three properties and disclose failures, with multiple seeds and paired IDs.
Add fixed-count feasible q50/q90 targets to reveal concentration versus reach.
Compare to cost-matched D-Flow/TOCFlow or another close terminal method if the
candidate uses terminal forecasts. A competitor previously dropped for cost
becomes relevant again if the new candidate incurs comparable cost.

Recommended new practical threshold, **to freeze before looking at new cells**:
at least **+1 percentage point** valid decoded in-band yield or **+10%** relative
distinct useful yield at matched wall time, with no violation of the declared
chemistry/diversity limits. Report confidence intervals; a practical threshold
is not a power calculation. Select sample size from development-set discordance.
For paired binary outcomes D in {-1,0,1},

```
SE(mean D) = sqrt((P(D != 0) - E[D]^2) / n).
```

Account for target reuse across seeds through target-cluster resampling or a
clearly conditional-on-target analysis, and adjust the prespecified comparison
family. Report the existing FR5 analysis separately, not silently replaced.

**Stop the variance-method claim** if repaired variance is null, the useful
region cannot be predicted without f_B, the benefit vanishes against band loss
or simple attenuation, decoded gains disappear, or extra compute yields fewer
useful molecules. Keeping the strongest baseline plus a rigorous negative
analysis is preferable to manufacturing novelty through additional gates.

## 8. What was executed in this audit

- Recomputed all 15 saved pilot cells and nine paired contrasts, including
  valid-and-in-band intersections and rescued/lost counts.
- Ran the existing ten BTVG2 gates: all passed.
- Verified exact pullback and coverage counterexamples and a shared-clip
  counterexample.
- Ran the 32-state real-checkpoint geometry diagnostic at four times on three
  properties; verified the corrected local geometry.
- Ran a separate 128-state representation audit on unguided/LGD-MC endpoints.

Reproduction: [audit_btvg_mechanism.py](../../proj1/scripts/audit_btvg_mechanism.py).
Use `--geometry --n 32` for the first artifact and
`--decode --n 128 --out results/btvg_audit/decoding.json` for the second.
The script writes separate audit artifacts and does not mutate production arms,
checkpoints, registered results or frozen strengths. No redesigned guidance arm
was claimed to have won a molecular benchmark in this investigation.
