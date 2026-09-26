# Full run v2: fixed-target protocol

**24 September 2026. Pre-registration. Nothing here has been run.** Henry's
decisions of 24 Sep are folded in; the one remaining choice is §6.

**v2 REPLACES v1 as the headline** (Henry, 24 Sep). v1's `dist` run is
finished and reported, and its write-up stays where it is
([../results/FULL_RUN_RESULTS.md](../results/FULL_RUN_RESULTS.md),
[../status/SCOPE_FM_GUIDANCE_STATUS.md](../status/SCOPE_FM_GUIDANCE_STATUS.md)).
**It is not deleted and it is not hidden**: it was pre-registered and it ran,
so the paper keeps at least a paragraph reporting it, otherwise choosing to
present only the later protocol is selective reporting of exactly the kind
this project has been careful to avoid. v2 is the headline; v1 is the
companion result that used the literature's own conditional protocol.

**Rule numbering.** v2 rules are **V1–V8** to keep them distinct from v1's
FR1–FR7 and the experiment plan's G1–G5.

---

## 1. What changes, and why

v1 used `dist`: every molecule got its own target, the real property of a
held-out test molecule. That is the EDM/EEGSDE conditional protocol and it
involves no choice by us, which is why it was the headline.

v2 fixes **one target value per property**. The reasons:

- **One number per property is what a reader can hold.** "Generate molecules
  with mu = 4.66 D" is a task; "every molecule gets a different target drawn
  from the test set" is a protocol that needs a paragraph.
- **It exposes the whole strength curve.** With one target, in-band against
  strength is a single curve per arm, so the trade-off each method is on can
  be plotted rather than asserted.
- **It removes v1's strength confound by construction** (§3): with a fixed
  target the chemistry cost of a strength is comparable across arms, so the
  arms can be put at equal chemistry cost.

**What it costs, stated up front.** A fixed target makes mode collapse a
winning strategy in principle: emit the same in-band molecule 5,000 times.
v1's per-molecule targets made that impossible. v2 therefore carries an
explicit distinctness guard (V6) -- which the existing data says is not
currently binding, but which must be reported so the reader can see it was
checked.

---

## 2. V1 — the target values

**One rule, three numbers: the 90th percentile of the QM9 training
distribution.**

| property | target | unit | QM9 fraction in band | delta (= 2 x f_B val MAE) |
|---|---|---|---|---|
| mu | **4.6627** | Debye | 3.2 % | 0.16799 |
| alpha | **85.05** | Bohr^3 | 2.3 % | 0.48135 |
| gap | **0.3162** | Hartree (8604 meV) | 8.0 % | 0.00760 |

Figures: `../results/figs/targets_{mu,alpha,gap}.png` — the QM9 distribution,
the target, the +-delta band, and what the unguided generator actually
produces.

**Why the 90th percentile, and not something else.**

1. **One rule, applied blind to all three properties.** No per-property
   tuning, so there is nothing to cherry-pick. This is the whole "story":
   *we ask each method to steer into the upper decile of the property.*
2. **There is headroom to demonstrate guidance.** The unguided generator
   lands in band 2.7–6.4 % of the time (measured, n = 512 screen). A target at
   the median leaves almost nothing to show: unguided is already at 7–11 %
   there, and the measured ceiling any arm reached on q50 was ~0.15.
3. **It is attainable, so a failure is the method's fault.** At these targets
   97 % of the generator's size distribution contains real QM9 molecules
   inside the band (measured: mu 0.977, alpha 0.974, gap 0.967). The target is
   not physically blocked for the sizes we sample.
4. **The guide is well trained there.** The upper decile of QM9 is ~13,000
   molecules, not an extrapolation. f_A's validation MAE is measured on this
   distribution.
5. **It is a genuine steering task**: 1.1–1.6 sd from the generator's own
   output mean, against 0.05–0.27 sd at the median.
6. **Practical, and this is a real reason: the existing screen already used
   these exact values.** `guidance_sweep.TARGETS[<p>]["q90"]` is these
   numbers, so the finished q90 cells are already v2's screen -- 150 from the
   compare stage plus the rest of the 205 cells at this target and window,
   all at n = 512, seed 20260921. Only the grid extension in §3 is new.
   **The window matters as much as the target:** `results/sweep` also holds
   171 q90 cells at `t_min_guide = 0.05`, sharing (property, arm, strength)
   keys with these. `freeze_v2.py` filters on the window; without that the
   freeze would depend on directory-listing order (measured: it moves plug's
   mu pick from w = 0.05 to w = 0.01 and lgd_mc's from 4 to 1).

**Honest caveats to carry.** (a) The upper decile is a choice; a different
decile would give different absolute numbers, though the *ranking* question
is unaffected. (b) alpha correlates +0.755 with molecule size, so a fixed
high alpha is much easier at 23 atoms (61 % of real molecules in band) than
at 15 (1.5 %). In-band **must** therefore be reported by size stratum, which
the harness already does. (c) We do not claim any paper uses these exact
values; the justification is the five reasons above, measured on our own
data.

---

## 3. V2 — the strength rule

**Each arm runs at the strength with the best in-band on the n = 512 screen,
among strengths whose molecule stability is at least 0.9 x unguided. Ties go
to the smaller strength. If no strength clears the floor, the arm runs at its
most stable strength and is reported as floor-limited.**

Figures: `../results/figs/strength_{mu,alpha,gap}.png` (what each strength
buys and costs, with the pick starred) and `pareto_{mu,alpha,gap}.png` (the
trade-off curve every arm is on).

**Why this rule.**
- **Select on the metric that decides the verdict.** v1's FR3a selected on
  MAE but scored on in-band — a mismatch that is simply avoidable.
- **The chemistry floor is the binding resource**, measured repeatedly across
  this project. Holding it fixed is what makes "which method steers best"
  answerable.
- **Selection is not evaluation.** The strength is picked on n = 512 at seed
  20260921; the run scores n = 5,000 at three disjoint seeds. So the pick
  cannot inflate the reported number.
- **It is the same rule for every arm**, including ours.

**V2a — the grid must be extended until the floor actually binds.** The
registered v1 grid stopped at w = 4, and `lgd_mc` still clears the floor at
w = 4 on all three properties (and `btvg_var` on alpha). Those arms are
**grid-limited, not chemistry-limited**: the rule is capping the strongest
competitor. The screen therefore extends to **w = 8 and 16** for any arm
still clearing the floor at the top of its grid, repeating until it fails.
Roughly 8–12 new n = 512 cells; ~15 GPU-minutes.

**V2b — the Pareto curve is reported whatever the pick.** The single chosen
strength is one point on a curve that is itself the result. No claim may rest
on the pick alone.

**The frozen strengths**, after the V2a extension (w = 8, 16, 32 added for
every arm that still cleared the floor at 4). No arm is grid-limited any more:

| arm | mu | alpha | gap |
|---|---|---|---|
| unguided | 1 | 1 | 1 |
| plug | 0.05 | 0.5 | 0.25 |
| tmpd | 0.25 | 1 | 1 |
| lgd_mc | 4 | 2 | 4 |
| tfg | 0.05 | 0.05 | 0.01 |
| btvg | 0.05 | 1 | 0.05 |
| btvg_var | 0.05 | 0.01 | 0.01 |

**What the extension showed.** `lgd_mc` falls below the chemistry floor at
w = 8 on all three properties (mol_stability 0.324 mu, 0.348 alpha, 0.285
gap), so w = 4 really was close to its chemistry limit and the old grid was
not capping it after all. On alpha the rule picks w = 2 over w = 4 because
w = 2 has the higher in-band (0.045 against 0.039) while both clear.
`btvg_var` clears at w = 8 on alpha (0.365) but with lower in-band than
w = 0.01, so its pick is unchanged. This is recorded because the *reason* a
strength is chosen is part of the protocol, and "the grid ran out" would
have been a different reason from "chemistry ran out".

---

## 4. V3 — the run

| setting | value |
|---|---|
| target | fixed, per property (§2) |
| n | 5,000 per cell |
| seeds | 20261001, 20261002, 20261003 (as v1, so noise is comparable) |
| molecule sizes | the sizes of `val[:5000]`, identical for every arm and seed |
| sampler | 100-step Euler, guidance window t >= 0.5, velocity clip 1.0 |
| generator | `fm_last.pt`, md5 a190ac83 (as every other cell) |
| arms, first pass | unguided, plug, tmpd, lgd_mc, tfg (45 cells, ~9 GPU-h) |
| arms, later pass | btvg, btvg_var, added with `ARMS=` (18 cells, ~7 GPU-h) |
| per-molecule sidecars | on, with `--save-coords` |

**The comparison methods are queued first** (Henry, 24 Sep): 45 cells,
~9 GPU-h on a 2g slice from v1's measured per-cell times. `btvg`/`btvg_var`
follow as 18 more cells in the same tree; nothing already run is repeated,
because the target, seeds and every other arm's strength are already fixed.

**Why these arms.** Three recent external methods (tmpd, lgd_mc, tfg), the
internal reference (unguided), our method (btvg), and the two ablation rungs
(plug = btvg's mean term exactly; btvg_var = its variance term alone).
`btvg2`/`btvg2_xproj` are **out for now** (Henry, 24 Sep: wait for their own
data first). They can be added later as extra cells without disturbing these,
since every other arm's strength and seeds are already fixed.

**Pairing.** Every arm and seed sees the same sizes and the same fixed
target, so all comparisons are paired at the molecule level.

**Sizes come from `val`, and that is deliberate** (it is also what the code
does for any fixed target: `guidance_sweep.run_cell` uses `mask_v` unless the
target is `dist`). v1 had to take sizes from `test` because the target WAS a
test molecule's property and the two have to come from the same molecule. v2's
target is a fixed number that belongs to no molecule, so there is no such
constraint, and drawing sizes from `val` leaves the test split untouched by
this experiment entirely. The same 5,000 sizes are used by every arm and seed.

---

## 5. V4–V6 — what is reported

**V4: the full metric block, per cell**, as `evaluate_samples` already
produces: MAE and RMSE against f_B, in-band fraction, atom and molecule
stability, RDKit validity, uniqueness of valid, distinct-valid-per-attempt,
embedding diversity, the guide-evaluator gap, non-finite count, and the
measured cost counters. Every property metric also in its **decoded** form
(argmax one-hot atom types), because arms that push the continuous type
features are otherwise flattered.

**V5: error-ranked buckets.** Sort the molecules of a cell by **|f_B - y|**
(absolute error against the evaluator) and report the full metric block within
the best **10 %**, **50 %** and **100 %**. Henry's decision, 24 Sep: this
sorting only.

**It is descriptive, never a yield.** It selects molecules using the same
oracle it then scores them with, so the top-10 % row is not something a user
could reproduce -- they do not have f_B at generation time. Its purpose is to
show the SHAPE of an arm's error distribution: whether a gain comes from a few
excellent molecules or from a broad shift. Every table carrying it says so, and
its numbers are never compared against another method's achievable yield.

**V6: the distinctness guard.** With one fixed target, an arm could in
principle win in-band by emitting the same molecule 5,000 times, which v1's
per-molecule targets made impossible. So in-band is always reported beside
**distinct valid in-band molecules per attempt**.

**How distinctness is measured, and why it is exact.** Not by a similarity
threshold on continuous coordinates. Each generated sample is decoded to a
molecular graph and written as a canonical SMILES string, and two samples are
the same molecule iff those strings are equal. That is a discrete, exact test,
and it is the same convention EDM and EEGSDE report uniqueness under. Two
different 3D conformers of one graph count as one molecule, which is the
intended meaning here. The harness already computes it
(`uniqueness_of_valid`, `unique_valid_per_sample`), so it costs nothing.

**Measured: collapse is not currently happening.** Uniqueness of valid
molecules is 0.995-1.000 for every arm in the v1 full run, and 1.000 for every
arm at w = 4 on the fixed q90 screen -- including the strengths where chemistry
falls apart (validity 0.48-0.69). So this is cheap insurance against a failure
mode we have not yet seen, not a burden, and not a metric expected to separate
the arms. **Threshold, fixed now:** an arm whose uniqueness of valid drops
below 0.95 has its in-band result reported as collapse-contaminated.

## 6. Budget dependence, and the one open decision

**Measured on the screen, and it matters: the ranking depends on the chemistry
budget.** Applying the §3 rule at three budgets gives three different winners:

| property | at 0.9x unguided (the rubric) | at 0.7x | at 0.5x |
|---|---|---|---|
| mu | **tfg** 0.053 | **tmpd** 0.061 | **tfg** 0.109 |
| alpha | **lgd_mc** 0.045 | **tmpd** 0.066 | **tmpd** 0.066 |
| gap | **lgd_mc** 0.113 | **lgd_mc** 0.113 | **tfg** 0.150 |

So "which method is best" has no budget-free answer, and a run at one budget
will read as if it does. That is itself a result worth reporting.

**Resolving power is fine either way.** At n = 5,000 x 3 seeds the se of an
in-band difference is ~0.0025, so at the 0.9x floor the best-vs-worst spread
is 5.3 sigma (mu), 7.3 (alpha), 13.3 (gap). One budget will separate the arms;
it just will not tell the whole story.

**DECIDED (Henry, 24 Sep): one budget, 0.9x**, the saved rubric's floor, as
`freeze_v2.FLOOR`. The run therefore reports a verdict *at this chemistry
budget*, and the Pareto figures carry the rest of the curve. The two-budget
option below is kept as the note of what was given up.

**The options that were weighed:**
- **One budget (0.9x, the saved rubric).** 63 cells, ~16 GPU-h. The
  pre-registered rubric's answer, and it resolves.
- **Two budgets (0.9x and 0.7x).** 117 cells, ~31 GPU-h. Gives the budget
  dependence above at full scale, which is the more honest headline and the
  more interesting paper claim. If time is tight, dropping `btvg_var` from the
  second budget saves 9 cells.

Because only one budget is run, **every headline sentence must carry "at a
chemistry budget of 0.9x unguided stability"**, and the budget-dependence
table above must appear beside it. Reporting the single-budget ranking as if
it were budget-free would be wrong on our own screen data.

## 7. V7–V8 — analysis rules, fixed before data

**V7 — the verdict.** As v1's FR5: decided on in-band, between arms that both
clear the chemistry floor, at an independent-samples z >= 3; paired z
reported alongside as supplementary. Ties are reported as ties, with sigma
shown. Because v2's strength rule selects on in-band, the primary comparison
is **between arms at their own selected strengths** — each method at its best
under a common chemistry budget.

**V8 — multiplicity.** The comparison family is fixed here: each arm against
`unguided` and against `lgd_mc` (the v1 winner), on in-band, per property.
That is 2 x 6 x 3 = 36 tests; Holm-adjust within each property. Anything
outside this family is exploratory and labelled so.

---

## 8. Regenerating the figures

```
python proj1/scripts/protocol_v2_figures.py
```

Reads the q90 screen at window 0.5 (every stage, so the `__cmp` cells and the
`__tgt` grid extension are one grid), `data/qm9.pt` and the v1 unguided
sidecars. It imports `freeze_v2.pick`, so a figure can never show a different
strength from the freeze the run uses. The strengths themselves come from

```
python proj1/scripts/freeze_v2.py --json-out results/full/v2/n5000/frozen_v2.json
```

which `v2_run.slurm` runs once at the start of the run; §3's table is that
command's output.
