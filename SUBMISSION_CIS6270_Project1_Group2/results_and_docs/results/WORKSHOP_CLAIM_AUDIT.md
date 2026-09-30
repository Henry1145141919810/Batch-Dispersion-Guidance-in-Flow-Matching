# What can be defended after BTVG and BTVG2?

23 September 2026. Executed claim audit and submission decision. This supplements
the [mechanism audit](../methods/BTVG_FAILURE_AUDIT_AND_REDESIGN.md) and the
[discrete-repair investigation](../methods/BTVG_EVOLUTION_AFTER_CHEM_GUARD.md).
It does not relabel either exploratory pilot as confirmation.

## Decision

BTVG is defensible as a concrete methodological hypothesis subjected to a
controlled, negative evaluation. It is not established as a useful improvement
over LGD-MC. A workshop contribution is plausible if the paper explains a
failure that matters beyond this particular unsuccessful proposal, with
interventions that distinguish competing explanations. Reframing alone is
insufficient; neither the code correction nor ordinary repair establishes a
novel algorithm.

The most focused question is: **When does improving a local continuous
guidance surrogate produce more chemically admissible, decoded molecules in
the requested property band?** BTVG is the motivating test case. The proposed
contribution is the measured separation between the surrogate, the actual
sampling update, and the decoded endpoint. The currently supported scope is
the evaluated QM9 pipeline, not all molecular guidance.

This is a legitimate type of scientific contribution. NeurIPS's current
[reviewing guidance](https://neurips.cc/Conferences/2026/ReviewerGuidelines)
recognizes insights from evaluating existing methods, and asks negative-result
papers to provide deeper analysis and community-relevant understanding.
It does not require a successful mitigation. That is evidence for the kind of
contribution, not an acceptance prediction or a specific workshop's policy.

## 1. New work executed for this audit

### Full-run comparison with repeated targets treated correctly

`audit_workshop_claims.py` reloaded all **63** frozen headline cells: seven
arms, three properties, three seeds, 5,000 targets per cell. It aligned
molecule IDs, rejected duplicate IDs, checked targets, tolerances and generator
identity, and reproduced every saved in-band score. Each target contributes
one cluster after averaging its paired differences across the three seeds.
This avoids counting three repeats of one target as three independent targets.

The following are BTVG minus LGD-MC in percentage points. Intervals are
approximate normal, simultaneous 95% intervals using Bonferroni over all 36
contrasts in this supplementary analysis: six controls, three properties,
and two metrics.

| Property | Historical in-band difference | Simultaneous 95% CI | Historical stable-and-in-band difference | Simultaneous 95% CI |
|---|---:|---:|---:|---:|
| mu | -2.29 pp | [-3.27, -1.30] | -0.74 pp | [-1.37, -0.11] |
| alpha | -1.57 pp | [-2.43, -0.72] | -0.55 pp | [-1.08, -0.03] |
| gap | -3.60 pp | [-4.62, -2.58] | -1.35 pp | [-1.98, -0.71] |

The negative comparison survives this analysis. BTVG also falls below plug-in
and TMPD on gap in this simultaneous family. Its positive differences from
unguided do not exclude zero in the same family. Pointwise intervals do exclude
zero for mu and alpha; neither observation establishes equivalence. Reporting
only the original "tie" labels conceals this distinction.

**Limits:** fixed checkpoint and three fixed sampling seeds; this is not
training-seed generalization. This family adjustment does not cover every
historical experiment or adaptive revision. The analysis is post hoc and
does not replace FR5. These are the historical continuous-feature f_B scores.
Most original sidecars contain no coordinates/features, so decoded full-run
scores cannot be recovered by rescoring those files.

Artifacts: [full intervals](../../results/workshop_claim_audit/cluster_intervals.md),
[machine-readable results and source hashes](../../results/workshop_claim_audit/cluster_intervals.json),
[figure](../../results/workshop_claim_audit/btvg_cluster_intervals.png),
[vector figure](../../results/workshop_claim_audit/btvg_cluster_intervals.svg).

### The claim that tau never engages is false

`audit_btvg_tolerance.py` ran original BTVG on its own trajectories for the
first 32 validation targets per property, seed 20260923, normalized strength
w=4, 100 Euler steps, guidance from t=0.5, and clip=1. Every guided step's
local variance was saved. No f_B outcome score was used; its checkpoint's
existing validation MAE defines the fixed band width.

For positive V, the normalized coefficient magnitude, relative to 1/(2s²), is
`q_tau(V) = max(1 - tau²/V, 0)`. Nonpositive V is a separate estimator failure,
not evidence that a calibrated tolerance has been attained.

| Diagnostic | mu | alpha | gap |
|---|---:|---:|---:|
| Guided sample-steps with 0 < V <= tau² | 9.50% | 14.50% | 14.19% |
| Same event, t < 0.8 | 0.21% | 1.46% | 0.42% |
| Same event, t >= 0.9 | 40.63% | 52.81% | 55.94% |
| Nonpositive V, all guided steps | 8.69% | 6.00% | 3.19% |
| Doubling tau changes active/inactive status on the same states | 9.50% | 8.69% | 10.94% |

These are descriptive fractions across dependent sample-steps, not independent
observations for significance testing. Doubling tau holds V and state fixed;
it is not an intervention on the whole trajectory. Threshold activation late
does not imply an appreciable change in the clipped sampling velocity, nor
better endpoint coverage. The local posterior variance already shrinks toward
zero by construction. The correct conclusion is **early saturation, late
threshold activity, unproven terminal benefit**.

Artifacts: [measurements and provenance](../../results/workshop_claim_audit/tolerance_activity.json),
plus the three `tolerance_*.pt` files in that directory. A controlled coefficient
example also verifies that the implementation responds across its threshold.

### The revised projection is implemented, but its outcome is not established

The five existing `test_btvg2_xproj.py` gates passed during this audit. They
include the mapping counterexample, actual-state local protections with an
asymmetric Jacobian, reduction to LGD-MC when the variance term switches off,
and the cost difference: three pullbacks instead of one.

`btvg2_xproj` protects the frozen local mean/variance surrogate after mapping
to the state the sampler updates. This is narrower than protecting final
property predictions. Joint clipping can still shrink LGD-MC's mean step;
orthogonality alone does not preserve the baseline's applied progress.
The 12-cell molecular retest in `xproj_run.slurm` is prepared, but no completed
outcome cells were available locally. Gate success is not a performance win.

## 2. Claims that must change before submission

| Draft claim | Defensible replacement |
|---|---|
| "BTVG ties unguided, plug, TMPD and TFG." | "FR5 did not establish superiority in these contrasts." Report effect sizes and intervals; this is not an equivalence test. Supplementary paired results include alpha versus unguided +3.05 standard errors and gap versus plug/TMPD -3.32/-3.61. |
| "Every test was preregistered." | Distinguish the pre-full-run FR5 rule from FR3a, amended after screening; TFG was added after the original full-run analysis; BTVG2, CSG, repair and this audit are exploratory. |
| "The variance term has no effect." | "The tested corrections have not demonstrated robust incremental terminal benefit under these settings." A negative result is scoped to these implementations, estimators, objectives and checkpoints. |
| "BTVG2 minus LGD-MC isolates mean-preserving variance reduction." | It isolates the implemented added correction. The m-space guarantee fails after pullback; the intended corrected intervention still needs an outcome test. |
| "Tau never engages." | Its threshold engages late in the new diagnostic. Do not infer terminal band calibration from that activity. |
| "Concentration is always preferable to likelihood-driven widening." | Even an exact Gaussian with a mean outside the band can lose coverage when variance shrinks. The earlier audit derives the sign change and a numerical counterexample. |
| "Chemistry per push is the binding constraint." | Chemistry restricts the usable guidance strengths in the tested grids. This does not establish a unique universal cause; baseline chemistry, estimator behavior, decoding and feasibility also matter. |
| "TFG gives 8.8x more useful alpha molecules." | 8.77x concerns historical soft-feature coverage at an unconstrained, one-seed point. On the same TFG outputs alpha coverage changes from 48.7% soft-scored to 18.6% decoded; stability is 22.0%. These are not all chemically useful hits. |
| "Timing matters more than method." | Within the screened settings, changing the guidance window produced a large effect. A universal ordering of the importance of design choices is unsupported. |
| "No arm beats the EDM #Atoms ceiling." | Report the published reference and any locally reevaluated checkpoint separately. Different training data and protocols prevent treating a published row as a matched ceiling. |
| "CSG is the positive innovation awaiting confirmation." | The specified mu-and-alpha pilot test failed; the remaining conditional interpretation is exploratory. No positive method claim is currently established. |

The apparent alpha MAE benefit is consistent with bias cancellation; aggregate
diagnostics do not identify an exclusive causal explanation. Local surrogate
variance, terminal residual variance, and variation of molecular properties
across different targets must remain distinct throughout the paper.

## 3. The contribution that could matter outside this project

The paper should establish a useful diagnostic result, not require the reader
to care about the name BTVG. Three connected pieces could support that result:

1. **A local guarantee can disappear before sampling.** The BTVG2 counterexample
   and checkpoint measurements show why a correction must be checked after the
   generator mapping, masking, centering and clipping. This is a concrete
   failure and reproducible test; projection mathematics itself is not novel.
2. **Continuous success need not survive decoding or chemical checks.** Report
   transitions from guide hits to independent decoded hits to chemically
   admissible hits on the same attempts. This supplies an actionable evaluation
   protocol, not a new metric invention. Define the primary utility as
   `U = mean[chemistry(decoded output) AND independent_band_hit(decoded output)]`.
   Count failed molecules and failed repair attempts in the denominator.
3. **Some chemical failure is impossible to fix by coordinates alone.** Under
   the neutral, closed-shell HCNOF valences 1/4/3/2/1, the required valence sum
   must be even because every integer bond contributes twice. About 88% of
   outputs failing both chemistry rules in the audited LGD-MC pilot have an
   odd sum; all 133,885 real QM9 records in this dataset have even sums.
   This is a fixed-composition obstruction, not a discovery about all chemical
   species. The graph identity is standard. Its measured prevalence and its
   consequence for choosing an intervention are the empirical finding.

The last point has a positive intervention, not only a correlation. With fixed
coordinates, atom count and attempts, one-atom repair adds **64/41/115** decoded
useful outputs per 2,048 attempts (mu/alpha/gap), while preserving every output
that passed either protected chemistry rule. However, target-blind repair
already adds **54/30/98**. Thus **73-85%** of the apparent full gain is ordinary
repair; the target-aware increment is only **10/11/17**, or 0.49/0.54/0.83 pp.
These are consumed, one-seed pilots, not confirmed novel guidance gains.

This also tells us where *not* to spend effort: an evaluation-oracle selector
over the same candidate pool could add only another 7/3/7 hits. The larger
remaining limitation is candidate reachability. A future solver must improve
feasible candidate coverage per unit cost against ordinary larger searches,
not just rename the current ranking rule.

The novelty boundary matters. [MolDiff](https://arxiv.org/abs/2305.07508)
already identifies atom-bond inconsistency and models atoms and bonds jointly.
[ChemGuide](https://arxiv.org/abs/2410.06502) already combines chemical stability
and property optimization. Neither "chemistry matters" nor "add a chemistry
constraint" is a sufficient new contribution. Our candidate insight is the
quantified failure decomposition and intervention evidence in frozen guidance;
its broader importance still needs replication on a stronger borrowed model.

## 4. Minimum experiments that distinguish a paper from a reframing

Complete these in order; do not initiate another unrestricted method search.

| Question | Minimal comparison | Decision |
|---|---|---|
| Was the BTVG2 failure substantially an implementation error? | LGD-MC and `btvg2_xproj` on identical targets/noise at the two already specified strengths; decoded coverage, useful yield, local mean/variance response and clipping loss. | Finish the prepared exploratory retest. A null narrows the corrected method's claim; a positive licenses fresh confirmation, not retrospective victory. |
| Does the phenomenon survive a stronger generator? | On the borrowed EquiFM checkpoint, compare unguided, LGD-MC, corrected BTVG2, and LGD-MC plus the same ordinary repair; use the native sampler and validated guide/evaluators. | If the chemical obstruction or method ordering disappears, report a base-model-dependent result and narrow the title. EquiFM backend gates alone are not replication. |
| Is the useful improvement reproducible and attributable? | Freeze choices before using a genuinely unexamined target block; use multiple seeds and compare no repair, target-blind repair, and target-aware repair on the same candidate set. Include the corrected variance component only if its prior gate warrants it. | Count only the incremental benefit beyond the strongest ordinary control. Record all proposal, guide, chemistry and runtime costs, and useful outputs per second. |

For the last step, the novelty claim requires a practically meaningful,
uncertainty-supported incremental effect; an improvement against unrepaired
LGD-MC alone does not qualify. Choose the effect margin and comparison family
before new outcomes. Audit all saved molecule IDs before declaring a split
untouched: another task may already have consumed it.

If the paper retains a strong claim specifically about *variance* rather than
the whole corrected algorithm, add a matched-estimator, matched-applied-budget
comparison against a direct smooth band-loss control. Log whether the
variance direction changes the local surrogate and whether that change predicts
terminal decoded success. Current correction-versus-LGD alone cannot separate
objective mismatch, estimator noise and clipping competition. This is an
additional requirement for that stronger mechanism claim, not permission to
assert it from a null outcome.

Finally, independently check a disclosed subset for physical geometry/property
quality before calling repaired outputs physically useful molecules. Two bond
construction screens and a neural property evaluator support computational
admissibility, not energetic stability or quantum-accurate target attainment.
Report diversity and duplicate handling alongside yield; preserving old useful
outputs guarantees neither new diversity nor higher throughput.

## 5. Paper text and scope

Suggested working title: **From Guidance Surrogates to Useful Molecules:
Diagnosing Variance Control in a Frozen QM9 Generator**. Retain the narrow
generator scope unless the borrowed-checkpoint experiment supports expansion.

Paper-ready contribution paragraph using only current evidence:

> We evaluate whether adding local variance control to training-free property
> guidance improves target-band coverage in a frozen QM9 flow model. Across
> three properties and three sampling seeds, the tested BTVG method does not
> establish superiority under the prespecified decision rule and underperforms
> LGD-MC. We diagnose mismatches between the local surrogate, its transported
> sampling update, and decoded chemical outcomes. An analytic counterexample
> and checkpoint measurements invalidate a proposed mean-preservation guarantee,
> while trajectory logging shows that tolerance activation alone does not
> establish terminal band control. An exploratory feasibility audit identifies
> a discrete valence obstruction and demonstrates recovery through atom-type
> repair. Most recovery is explained by a target-blind control. Together these
> results motivate evaluating guidance by independently scored, chemically
> admissible band hits per generation attempt and by attributable benefit over
> strong ordinary controls.

The main paper should show one paired outcome figure, one intervention figure
linking the suspected mechanism to endpoints, and the useful-yield attribution
table. Put the full arm inventory, failed revisions and extended grids in the
appendix. Do not turn the paper into a chronological account of every idea.

**The course and a workshop impose different completion conditions.** The
[assignment](../../course/Project1_Paper_Instructions.tex) permits honest
underperformance (around line 294) but still requires a substantive methodological
idea, its component controls, and transfer of the *same* idea to a genuinely
different modality (around lines 219-249). Another molecular generator is not
a second modality. The repository's DNA/simplex transfer remains incomplete.
A credible narrowly scoped workshop study does not, by itself, satisfy the
assignment. Nor does the permission to report negative results certify novelty.

## Reproduction

With the project's Python dependencies available, run:

```text
python proj1/scripts/audit_workshop_claims.py
python proj1/scripts/audit_btvg_tolerance.py --n 32 --seed 20260923 --w 4
python proj1/tests/test_btvg2_xproj.py
python proj1/scripts/results_tables.py
```

The scripts do not submit cluster jobs or alter production guidance behavior.
The historical tau grid and registered verdict remain unchanged. The audit
corrected the living status and table generator's interpretation text; results
are reproduced from saved outputs or explicitly labeled new diagnostics.
