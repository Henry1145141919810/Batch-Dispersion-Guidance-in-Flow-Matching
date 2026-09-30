# Evolving after BTVG2 and the chemistry guard

23 September 2026. Exploratory research memo; supersedes the *next-step*
recommendations, not the historical results, in BTVG_FAILURE_AUDIT_AND_REDESIGN.md.

## Locked diagnostic before scoring any repair proposals

The w16 paired chemistry-guard audit shows substantial useful-output turnover
and almost no net gain. A new failure decomposition also finds that most
outputs failing both chemistry rules have an **odd sum of allowed valences**.
With H/C/N/O/F valences 1/4/3/2/1, no integer bond graph can satisfy every atom
in such a composition: the sum of bond degrees must be twice the sum of bond
orders. This concerns the current neutral closed-shell QM9 task; it is not a
claim that radicals or ions are universally invalid chemistry.

Before inventing another guidance correction, run a bounded **control**, not a
newly branded method:

- Reuse the already consumed w16 LGD-MC endpoints, all 2048 per property.
- Leave every endpoint passing either table stability or independent neutral
  RDKit validity exactly unchanged.
- For the remaining endpoints, enumerate all single-atom relabelings among
  H/C/N/O/F. Keep atom count and all coordinates fixed. No continuous optimizer,
  added atoms, extra generator calls, or target changes.
- Keep candidates passing the exact table, independent neutral RDKit
  sanitization, and connectedness. These are chemistry *screening* rules, not
  an energy or synthesizability assay.
- Compare a target-blind chemistry choice (lowest smooth valence penalty;
  deterministic ordering for ties) to a choice minimizing decoded f_A target
  error from the **same candidate set**. Also report candidate multiplicity.
- Use f_B only after both output choices are final. Do not select proposals,
  strengths or hyperparameters by f_B. Report useful rescues and losses under
  both chemistry rules, all candidate failures and runtime, with denominator
  2048 rather than only accepted outputs.

Every baseline useful molecule under either protected rule is unchanged, so
its useful-yield indicator is preserved for **any** evaluator. This is an
output-level nonloss statement at fixed attempt count. It says nothing about
new successes, unseen chemistry tests, diversity, or useful outputs per second.
The latter can worsen through repair cost. Atom substitution changes molecular
identity and must be explicitly reported as post-generation refinement.

This is a cheap falsification test for a discrete-feasibility opportunity. It
is not confirmation, not a licensed change to the registered comparison, and
not a claim that constrained decoding or one-edit molecular search is novel.
If target-blind repair explains the entire gain, there is no evidence for a
new target-guidance contribution. If few candidates exist, one-edit decoding
is insufficient and a larger search must justify its added cost.

## 1. Decision after executing the diagnostic

**Stop treating scalar variance reduction as the project's presumptive
innovation. Investigate decoded chemical feasibility, with simple repair as
the new baseline that an original method must beat.**

There is now a concrete positive intervention: one-atom repair adds useful
molecules beyond the same LGD-MC endpoints. However, 73-85% of its useful-yield
gain is already obtained by target-blind repair. Calling that entire gain a new
guidance contribution would repeat the attribution mistake made with BTVG2.
This investigation establishes a useful engineering direction and a previously
unmeasured bottleneck in this generator. It does **not** establish a novel
algorithm or an independently confirmed performance result.

The nearest credible research question is:

> Can a frozen molecular generator expose and cheaply correct the discrete
> chemical decisions that prevent a target from having a usable completion,
> beyond what ordinary endpoint repair and equal-cost sampling already do?

This is materially different from making a noisy continuous property estimate
more concentrated. A narrower property distribution over impossible molecular
compositions is still unusable.

## 2. What the recent history rules out

The audit reads the committed history through `44a244b`, the newer working-tree
status, the BTVG2 pilot, both chemistry guards, decoded scoring and the current
transfer plan. The EquiFM transfer backend is gated but has no real guidance
cells yet in the latest status. It is a future generalization test, not evidence
supporting this proposal.

1. **Original BTVG:** its mean term reduces to the plug-in baseline under the
   project's normalization. Its distinguishing variance term has no consistent
   coverage benefit; the registered full run loses to LGD-MC on all properties.
2. **BTVG2:** the main target-seeking term is exactly LGD-MC. Its added probe
   variance penalty has not earned incremental credit. It also has a real
   geometric flaw: orthogonality before the generator pullback need not
   survive in the state that the sampler actually changes. Fixing that is a
   correctness repair, not evidence that variance is the missing signal.
3. **Variance interpretation:** variance among local isotropic probes is not
   uncertainty in the actual guided terminal result. K=4 is a noisy estimate.
   Shrinking variance away from a target can reduce band probability. There
   is no causal evidence that shrinking this surrogate repairs the failures
   that currently dominate useful yield.
4. **Chemistry guards:** both improve chemistry marginally, but their preset
   useful-yield gate failed. The w32 prediction also failed literally on alpha.
   The conditional headroom interpretation is plausible but exploratory; it
   must not retroactively turn the failed criterion into a pass.
5. **Decoded evaluation:** the recent additive `_dec` fields have already
   corrected the representation mismatch for new cells. Do not describe it as
   an unfixed evaluation bug. Historical soft-feature results remain historical.
6. **Earlier alternatives:** generic band losses, trust regions, terminal
   lookahead, half-space projections and decoded boundary gradients have
   already been discussed internally or published. A new name cannot make
   those ingredients original.

## 3. New paired outcome decomposition

Data: `results/pilot_chem/seed20261001_n2048`, w16, the same IDs/targets for every
arm; already consumed test data. Here H means decoded f_B in-band and C means
the indicated chemistry check. Useful yield is E[H C], not the product of the
two marginal success rates.

For the norm-preserving guard versus LGD-MC:

| property | table-stable useful outputs rescued | existing useful outputs lost | net |
|---|---:|---:|---:|
| mu | 74 | 67 | +7 |
| alpha | 37 | 36 | +1 |
| gap | 78 | 76 | +2 |

This is terminal turnover despite a correct local non-increase guarantee.
It is not evidence that the local calculation is numerically broken. The
constraint protects a surrogate at one state, not the completed baseline
molecule. The independent neutral RDKit useful-yield net changes are +1, +11,
and -4; changing the chemistry rule does not reveal a strong uniform win.

The same audit also finds guide/evaluator band agreement is limited: among
LGD-MC's decoded f_A hits, only 58.9%, 73.5%, and 56.7% are f_B hits. A rule
protecting f_A alone cannot promise preservation of f_B hits. These are
generated-sample agreement rates, not a physical accuracy measurement.

## 4. The discrete obstruction

Let a_i be an atom type, v(a_i) its required valence, and b_ij an integer
undirected bond order. A necessary condition for satisfying all valences is

    sum_i v(a_i) = sum_i sum_j b_ij = 2 sum_{i<j} b_ij.

Hence the left side must be even. This is the elementary degree-sum identity,
not a new theorem. Its prevalence in these actual outputs is the new finding.

| property | LGD-MC odd compositions / 2048 | failing both table and independent chemistry | odd among those failures | f_B hits failing both / odd subset |
|---|---:|---:|---:|---:|
| mu | 848 | 967 | 848 (87.7%) | 118 / 98 |
| alpha | 875 | 989 | 875 (88.5%) | 110 / 95 |
| gap | 884 | 1007 | 884 (87.8%) | 179 / 165 |

All 133,885 real molecules in the local QM9 tensor have an even sum. Unguided
generation has 860/2048 odd compositions (42.0%). The norm-preserving guard
still has 838/850/874 odd compositions. This problem is largely inherited from
the generator/decoding pipeline; it is not caused only by strong property
guidance. It must also be checked on the borrowed generator before claiming
a broadly useful method.

Consequences:

- Pure coordinate relaxation cannot repair an odd composition under these
  fixed valence rules. At least an atom identity, atom count, charge/radical
  assumption, or the allowed-valence model must change. The probe changes only
  one atom identity and stays within the existing task definition.
- Smooth chemistry guidance *can* change type features; this is not a proof
  that continuous guidance is mathematically incapable of fixing parity.
  The measured guards simply leave most of this obstruction present.
- Even parity is necessary, not sufficient for a chemically valid graph.
  A parity-only decoder must be a control, not the proposed full solution.
- RDKit sanitization using the table's bonds can accept radicals/implicit
  hydrogen conventions that differ from the independent neutral assignment.
  The guarantees below refer specifically to table stability and independent
  neutral validity, not every possible meaning of "valid molecule."

Simply adding a smooth parity penalty is also unconvincing. Under an
independent categorical type approximation, write

    q_i = p_i(C) + p_i(O) - p_i(H) - p_i(N) - p_i(F).
    P(even) = (1 + product_i q_i) / 2.

The derivative contains products over the other atoms, which can be tiny
when several types are uncertain. Near deterministic type logits, the softmax
derivative can instead be tiny. This does not forbid a useful estimator, but
it shows why "add another differentiable penalty" is not a sufficient design.

## 5. What the executed repair control found

The locked protocol above was executed without modifying the generator,
production guidance arms, source sidecars, registered results, or frozen
strengths. It inspected approximately 40,000 parity-compatible one-atom
alternatives per property. Only 728/760/748 passed the exact table, independent
neutral RDKit sanitization and connectedness. Those alternatives cover
528/542/534 original failed endpoints; only 179/192/195 endpoints had multiple
accepted alternatives.

Useful yield below uses table stability AND decoded f_B in-band; all counts
use denominator 2048, including failed repair attempts.

| property | LGD-MC | + target-blind repair | + target-aware repair | full repair gain | target-aware increment over blind |
|---|---:|---:|---:|---:|---:|
| mu | 95 (4.64%) | 149 (7.28%) | 159 (7.76%) | +3.13 pp | +0.49 pp |
| alpha | 79 (3.86%) | 109 (5.32%) | 120 (5.86%) | +2.00 pp | +0.54 pp |
| gap | 135 (6.59%) | 233 (11.38%) | 250 (12.21%) | +5.62 pp | +0.83 pp |

For independent neutral RDKit useful yield, the baseline counts are 137/100/199,
and the target-aware repaired counts are 201/141/314. The increments are the
same 64/41/115 because every new accepted candidate passes both checks.
Connectedness gives baseline 134/98/193 and repaired 198/139/308. All new
successful repairs have distinct independent canonical SMILES in this pilot,
with no overlap with the corresponding baseline connected useful set. This
checks duplicates, not structural or conformational diversity.

No baseline useful output was lost under the protected rules. Raw decoded
property coverage alone changes by -1/-13/+7 hits: a useful-yield improvement
is **not** automatically an improvement to the historical in-band headline.

Attribution:

- Target-blind repair contributes 54/30/98 of the 64/41/115 additional hits.
- The target-aware choice contributes only 10/11/17 net additional hits over
  that control. Its paired rescues/losses are 14/4, 12/1 and 21/4. The apparent
  target-choice gains are exploratory, not independent confirmation.
- Approximately 52-63% of target-aware accepted repairs are C->N substitutions (the
  exact counts are in `verification.json`). Composition changes require
  explicit reporting and physical follow-up; they are not small moves in
  chemical identity merely because coordinates were unchanged.
- Search, screening and guide scoring took about 1.3 seconds/property on this
  machine, including a checkpoint/reproduction check. That excludes the
  baseline independent chemistry audit and final evaluator scoring. This is
  a microbenchmark, not an end-to-end throughput result.

The repair accepts chemistry-feasible alternatives even when f_A cannot put
them in band; those attempts remain in the denominator. It is not filtering
out failures or changing targets to make the score look better.

### A useful ceiling that rules out another tempting detour

After both methods were frozen, the saved f_B candidate scores were used only
to calculate an **unattainable evaluation oracle ceiling**, never to select an
actual output. If an oracle could pick the best member of each existing
candidate set, at most 71/44/122 repaired endpoints would hit. The f_A selector
already gets 64/41/115.

Thus perfect reranking of this same set could add only 7/3/7 more hits, or
0.34/0.15/0.34 pp. A complex new ranker cannot deliver a large advance on this
pool. **Candidate reachability, not ranking sophistication, is the next
bottleneck.** This is a retrospective bound for these endpoints, not a
distribution-free bound or a license to train on f_B.

## 6. What should evolve, precisely

Keep the practical one-edit repair as a reference pipeline. A research
candidate should create affordable, chemically feasible alternatives that the
one-edit set cannot express, while retaining the original successful outputs.
The most defensible next object is a **small joint discrete/geometry repair**:

    Find (a', c') close to the failed completion (a, c),
    subject to a bounded number of type edits, bounded coordinate movement,
    explicit valence/connectivity feasibility, and decoded f_A band entry.

Use the existing frozen generator's type preferences to order alternatives,
if available; saved pilot sidecars contain hard types, not those continuous
features, so this preference was not assumed in the executed control. Query
f_A on actual edited molecules. A local soft-feature gradient may prioritize
queries, but it must not stand in for a measured discrete property change.
Allow fallback when the edit budget cannot reach the band.

This formulation is **constrained local search**, which is not itself novel.
The possible algorithmic contribution would have to be a concrete solver or
proposal estimator exploiting coupled type/bond decisions to reach more
feasible band targets per query than ordinary enumeration, local search,
score-function sampling or the earlier decoded-facet proposal. Demonstrate
that distinction before integrating it into the ODE or naming a new BTVG.
In particular, merely running the same endpoint repair earlier in the sampler
adds transport error and cost; it is not an automatic innovation.

The output-preservation rule is principled but deliberately conservative.
For two protected chemistry indicators C1/C2, change only outputs with
C1=C2=0. Then, for **any** property evaluator h,

    1[Cj(x') and |h(x')-y|<=delta]
       >= 1[Cj(x) and |h(x)-y|<=delta], j in {1,2}.

Proof: if the right side is one, x is protected and x'=x; otherwise the right
side is zero. This holds for finite decoded outputs and requires no f_A=f_B
assumption. It guarantees no lost baseline successes, not that a repair will
succeed. When editing a chemistry-valid molecule instead, an f_A-only rule
cannot generally promise no f_B loss without an additional validated relation
between the predictors. That is why a local orthogonality claim was too weak.

There is also no automatic cost guarantee. If baseline useful yield is u with
cost T, an added repair costs r and yields delta_u, then improved useful
outputs per time requires delta_u/u > r/T. Fixed-count improvement alone does
not establish this. Selection time, unsuccessful proposals and chemistry
checks all belong in r.

## 7. Novelty boundary and the decisive next comparison

Primary literature checked during this investigation:

- [LGD-MC](https://proceedings.mlr.press/v202/song23k.html) already supports
  general losses. Replacing squared error by a band/feasibility reward does
  not create a new inference principle.
- [MolDiff](https://arxiv.org/abs/2305.07508) explicitly studies atom-bond
  inconsistency and jointly models atoms and bonds. Our measured parity
  obstruction is a particular instance, not the first recognition that atom
  decisions need chemical consistency.
- [MUDiff](https://arxiv.org/abs/2304.14621) jointly represents discrete graph
  and continuous geometry information. Mixed discrete/continuous generation
  is not a new idea.
- [ChemGuide](https://arxiv.org/abs/2410.06502) combines chemistry-oracle and
  property guidance, including bilevel optimization. "Add chemistry to
  property guidance" cannot be the central novelty claim.
- [Pro-GAT](https://doi.org/10.1021/acs.jcim.6c01049) repairs diffusion-generated
  PROTAC structures using bounded coordinate and type changes. Its setting
  and learned repair differ from this frozen QM9 control, but endpoint repair
  or geometry preservation alone cannot establish originality.

The existing audit additionally covers PCFM, TOCFlow, trust constraints and
terminal lookahead. This is a targeted novelty check, not an exhaustive proof
that a specific future solver is new.

For a new component X, the contribution to establish is

    U(LGD-MC + ordinary repair + X) - U(LGD-MC + ordinary repair),

at matched total cost. Comparing only against unrepaired LGD-MC would credit
X with gains already explained by repair. Apply the same repair policy to
other surviving baselines as well; otherwise a generic decoder improvement
is being unfairly assigned to BTVG.

Before a broad run, one bounded development comparison should contain:

| arm | purpose |
|---|---|
| LGD-MC + the fixed one-edit repair | strongest cheap comparator established here |
| same + straightforward larger local search | tests whether extra candidates alone explain improvement |
| same + proposed structured candidate generator | isolates the proposed algorithm |
| extra LGD-MC attempts within the same time budget | checks practical compute value |

Use the same query/repair budgets and f_A-only selection, retain every attempted
target in the denominator, report candidate reachability and rescue/loss
counts, and include all three properties. Develop only on declared development
data. Audit used IDs before reserving confirmation data; these 2048 are already
consumed. A new seed alone does not make them untouched.

Keep the earlier proposed practical threshold: at least +1 pp useful yield or
+10% relative distinct useful yield at matched time over the strongest
appropriate comparator, with frozen chemistry/diversity limits and uncertainty
reported. Fix the exact comparison family and sample size before confirmation.
The one-edit prototype's full gain clears this magnitude against *unrepaired*
LGD-MC; it has not cleared a novelty test against repaired baselines. Verify
generalization on EquiFM when its real cells are available. Add an independent
physical geometry/property check on a disclosed subset before claiming actual
molecular design success.

If a structured solver cannot beat plain repair/search, retain the practical
repair and write up the failure mechanism honestly. If a variance claim is
retained, its state-space-corrected term must beat these same controls; the
new repair results provide no evidence in its favor. The currently justified
evolution is **from a presumed variance mechanism to measured, attributable
repair of discrete feasibility failures**.

## 8. Reproduction and limits

- `proj1/scripts/audit_joint_failures.py`: raw paired transitions, independent
  and connected chemistry, guide agreement and parity counts.
- `proj1/scripts/probe_discrete_repair.py`: locked one-edit candidate pool,
  target-blind/aware choices frozen before evaluator use, exact table check
  against all source endpoints, checkpoint prediction reproduction, no-loss
  assertions, source/checkpoint/script hashes.
- `results/btvg_joint_audit/joint_failures.json` and
  `composition_controls.json`: failure decomposition, including unguided.
- `results/btvg_discrete_probe/summary.json`, `*.proposals.pt`,
  `*.evaluation.pt`: outcomes and complete candidate/selection records.
- `verification.json`: separate literal-evaluator checks on 32 edited outputs
  per arm/property, one-edit checks, substitutions and independent SMILES
  deduplication. `ranking_ceiling.json`: retrospective oracle bound only.

All protected baseline outputs remain exactly their original coordinates and
types; accepted repairs retain their original coordinates and change exactly
one type. Scoring uses decoded features. No production sampler change, new
generator training, benchmark replacement or independent confirmation was
performed. Passing two bond heuristics and a learned evaluator is not proof
of a physically stable, synthesizable molecule with the desired true property.
