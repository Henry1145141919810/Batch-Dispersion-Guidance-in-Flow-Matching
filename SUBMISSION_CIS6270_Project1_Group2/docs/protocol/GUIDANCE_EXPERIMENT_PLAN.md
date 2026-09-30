# Guidance experiment plan: arms, properties, protocol, cost

**20 September 2026.** The base model is frozen and named **`fm_v1`** everywhere from here on
(epoch 1500 EMA weights, `fm_last.pt` key `ema`, atom 0.9366 +- 0.0012, molecule 0.3993 +- 0.0085,
validity 0.7625 +- 0.0031; see `BASE_MODEL_BENCHMARK.md`). Every arm below is evaluated on
`fm_v1`, so differences between arms are attributable to guidance and nothing else.

This document is the pre-registration for the guidance comparison. Numbers quoted as "measured"
were measured; everything else is labelled as an estimate.

---

## 1. The arms

`OWN` = a candidate for the methodological innovation the project requires.
`COMPARE` = published method, reproduced on `fm_v1` for comparison.
`CONTROL` = not a method; a floor or a null test that catches implementation bugs.

All arms share: the same frozen `fm_v1`, the same `f_A` as guide, the same `f_B` as evaluator,
the same sampler, the same NFE, the same seeds, and the shared `J^T` pull-back
(`guidance.py::_pullback`). They differ only in the scalar/vector weight applied before that
pull-back, which is what makes the comparison a comparison.

| # | arm | class | what it computes | **measured** ms/mol @ NFE 100 |
|---|---|---|---|---|
| 1 | `unguided` | CONTROL | no guidance; the floor for property targeting | **34.5** |
| 2 | `plug` | COMPARE | `(y - f(m)) / s^2 * J^T g`. DPS-style. Needs strength tuning or it is catastrophic | **95.0** |
| 3 | `tmpd` | COMPARE | plug-in with the uncertainty denominator `s^2 + g'Sg` -- TMPD / PiGDM's published scalar form, and our `smg_var` ablation | **241.8** |
| 4 | `tfg_mc` | COMPARE | TFG's Monte Carlo smoothing: K isotropic perturbations at a **tuned** sigma, softmax-weighted (Ye et al. 2024) | **125.5** (K=4) |
| 5 | `lgd_mc` | COMPARE | likelihood marginalisation, isotropic draws at a scale **read off the model** (`r^2 = tr(Sigma)/d`) rather than tuned | **271.2** (K=4) |
| 6 | `osc` | COMPARE | observable-space closure: same guide budget as `lgd_mc`, scalar observable's law integrated analytically | **273.3** (K=4) |
| 7 | `smg` | OWN | **SMG as shipped.** Mean correction `c = 1/2 tr(H Sigma)`, denominator `s^2 + g'Sg` | **421.5** (1 probe), **959.0** (4) |
| 8 | `smg2` | OWN | **Completed-moment closure.** Denominator gains `1/2 tr((H Sigma)^2)` | **802.7** (1 probe), **2538.6** (4) |
| 9 | `smg2_curv` | OWN | `smg2` plus the Stein direction `((r^2-S)/S^2) Sigma H Sigma g` | **836.4** (1 probe) |
| 10 | `escalate` | OWN | closure-residual escalation. **Not yet implemented** -- design only | >= 1074 (smg2 + lgd_mc, plus escalated cells) |
| 11 | `kappa3` | OWN (probe) | logs `gamma = k3[g,g,g] / (g'Sg)^{3/2}`. A **column on another arm**, not a sampler | +370 on `smg2` (**1173.1** total) |

All figures measured on the RTX 5080 by `results/bench/time_guided.py`, which also prints the
generator and guide call counts behind each. **An earlier version of this table was estimated
from call counting and was wrong by up to 2x** (`smg2` was given as ~450, `lgd_mc` as ~130);
those estimates are replaced by the measurements above.

**`smg2` costs ~1.9x `smg` at the same probe count**, because the second trace needs two JVPs and
two HVPs per probe against one each for the first. At 4 probes it is 2.5 s/molecule, which is why
the probe count is settled by gate G3 before Stage 2 rather than assumed.

### Why both `smg` and `smg2` are kept

The `TOP6_GUIDANCE_RECOMMENDATIONS.md` memo (section 2) shows SMG's denominator is not the
second moment of the quadratic surrogate: for `q(u) = f(m) + g'u + 1/2 u'Hu` with `u ~ N(0,Sigma)`,
`Var[q] = g'Sigma g + 1/2 tr((H Sigma)^2)`, and SMG has only the first term. On synthetic problems
with an exact reference this is the difference between a 1000x error and a 1% error near a
critical point of `f`.

Keeping both arms is a deliberate choice: the results table then *shows* the defect and its
repair rather than quietly replacing one with the other. It also means any earlier comparison
made against "SMG" is re-runnable against the corrected version.

**Honesty requirement.** The memo rates `smg2`'s novelty 2/5 -- it is the obvious extension of
TMPD/MMPS moment projection from an affine observable (where the extra trace vanishes) to a
quadratic one. It must be presented as **the control that every other arm has to beat**, not as
the contribution. The contribution candidates are `kappa3` (as an instrument) and `escalate`.

### Deferred, with reasons

| candidate | why deferred |
|---|---|
| **QPMG** (memo section 3) | The memo's own go/no-go: run it only if property-law error is >= 30% of total error at our operating `s`, which is exactly what `kappa3` measures. Implementing a 1-D Fourier closure before that measurement is work that the measurement may discard. Revisit after the decomposition. |
| **RCH** (Haimo v1 residual calibration head) | Needs a *trained* 28-coefficient head, so it is not inference-time-only like every other arm. Prior work by a group member, useful as a comparator, but it changes the cost accounting. Add only if time permits after the main grid. |
| **DBFG** (memo section 5) | The memo's gate: first measure the soft-versus-decoded guide gap on already-generated `fm_v1` samples. One cheap pass. If that gap is small, neither DBFG nor its score-function rival matters. |
| **MAG as a method** | The memo recommends keeping only its counterexample (two posteriors with identical mean and variance whose exact guidance is equal and opposite) as a motivating result. Not an arm. |
| **Tolerance Band (D)** | Rejected in the memo's section 7. Not an arm. |

---

## 2. The property functions

### THE PAIR IS FIXED: EGNN f_A, EGNN f_B, one pair per property

There is exactly one guide and one evaluator per property for the whole
comparison, and they do not vary across arms, targets, strengths or windows:

    f_A = proj1/checkpoints/f_A_<prop>.pt      EGNNScalar 128x4, trained on train_a
    f_B = proj1/checkpoints/f_B_<prop>.pt      EGNNScalar 128x4, trained on train_b

`guidance_sweep.py` loads exactly these and nothing else. The guide/evaluator is
a nuisance variable: varying it would multiply the grid and make every number
conditional on a choice the reader has to track.

**Why EGNN on both sides, when we also have a transformer.** Three reasons, in
order:

1. **It is the most accurate on both sides for every property** (table below),
   and f_B's accuracy sets `delta = 2 x MAE`. Using the transformer as evaluator
   would widen the mu band from 0.168 to 0.243 D -- 45% looser -- which flatters
   every arm and compresses exactly the differences the experiment measures.
2. **It is what published work does.** EDM uses an EGNN classifier on both
   sides; EEGSDE uses Satorras' EGNN for the oracle and an EGNN energy for the
   guide; TFG reuses EEGSDE's oracle and trains its own EGNN guide. Same
   architecture on both sides is standard; the **disjoint data split** is the
   protection against reward hacking, and `SPLIT_PROTOCOL.md` guarantees it.
3. Architecture diversity between f_A and f_B is our own extra-rigour idea, not
   a requirement, and it costs measurement sensitivity.

**The cost, disclosed.** f_A and f_B share architectural blind spots, so
guidance that exploits a quirk of the EGNN inductive bias could fool both. The
disjoint split makes that unlikely, not impossible.

**The mitigation, and it is cheap.** Once the sweep names a winner, re-score
**only that arm's samples** with the transformer f_B
(`f_B_<prop>_transformer.pt`, a genuinely different inductive bias). One extra
run. If the advantage survives a different evaluator architecture, the claim is
much stronger; if it does not, that is the finding. This is a post-hoc
robustness check on the winner, never a second primary result.

### External predictors: usable as a CHECK, never as the primary pair

Published predictors are more accurate than ours on every property, but all of
them trained on someone else's split, and ours is a private random split.

| predictor | mu (D) | alpha (Bohr^3) | gap (Ha) | overlap with our `train_a` |
|---|---|---|---|---|
| **our EGNN f_B** | **0.0840** | **0.2407** | **0.00380** | **0%, by construction** |
| TFG oracle (EGNN) | 0.0728 | 0.1529 | 0.0027 | ~37% (see below) |
| TFG guide (EGNN) | 0.0659 | 0.1551 | 0.0029 | ~37% |
| SchNet (mu only) | **0.0210** | -- | -- | ~all of QM9 |
| EDM published L-bound | 0.043 | 0.10 | 0.00235 | ~37% |

**The ~37% figure, and how it was obtained.** It is an EXPECTATION, not a
measurement. EDM (and TFG, which vendors EDM's machinery) builds its halves
with `np.random.seed(42)`, permutes its 100,000-molecule train partition and
takes the first 50,000 (`tasks/networks/qm9/data/utils.py`). A molecule drawn
from OUR 133,885 therefore has probability 50,000/133,885 = **37.3%** of
sitting in their first half, i.e. ~19,240 of our 51,527 `train_a` molecules.

Two caveats to carry into any write-up:

- It assumes their half is a uniform draw independent of ours. Both splits are
  seeded permutations, so this is reasonable but unverified.
- An **earlier draft said 38%**, using their post-exclusion pool (130,831) as
  the denominator. That was wrong: the draw is from OUR set, so OUR total is
  the denominator. **Quote 37%, not 38%.**

The exact overlap is computable -- both splits are deterministic -- but it
needs their full download-and-process pipeline reproduced, and no decision
turns on the precise value: at 30% or 45% the conclusion is the same.

**Why this rules them out as primary f_B.** The generator trained on `train_a`.
An evaluator that memorised those molecules scores their near-copies
optimistically -- exactly the leak `SPLIT_PROTOCOL.md` exists to prevent.

**Why they are still worth running.** The leak is identical across arms, since
every arm samples from the same generator. So absolute MAE is optimistic but
the RANKING is informative. Protocol, on the winning arm only, re-scoring saved
samples with no re-sampling:

| property | check evaluator | what it tests |
|---|---|---|
| mu | SchNet, 0.0210 D | 4x better AND a different architecture -- tests both accuracy and shared blind spots |
| alpha, gap | TFG oracle | the only external option; same EGNN family, so it tests the data-leak question only |

If the winner's advantage survives a 4x-better evaluator of a different
architecture, the claim is far stronger. If it does not, that is the finding.

**Not to be used as f_A, ever:** `ridge`. At 0.80 D on mu it is 9x worse than
the EGNN, and when f_A is much worse than f_B guidance cannot reach the band no
matter how good the method is -- the experiment would measure f_A's weakness
instead of the guidance. `proj1/scripts/predictor_table.py` flags this pairing
automatically as a "delta trap". Ridge exists only as the accuracy floor.


Six predictors, all trained on Betty, **all disjoint by construction**: `f_A` on `train_a`
(which `fm_v1` also saw), `f_B` on `train_b` alone (which neither the generator nor the guide has
seen). This is the protocol in `SPLIT_PROTOCOL.md`.

| property | unit | `f_A` val MAE | `f_B` val MAE | chance (std of data) | tolerance `delta = 2 x f_B MAE` |
|---|---|---|---|---|---|
| **mu** (dipole moment) | D | 0.0897 | **0.0840** | 1.539 | **0.168** |
| **alpha** (polarizability) | a.u. | 0.2447 | **0.2407** | 8.200 | **0.481** |
| **gap** (HOMO-LUMO) | Ha | 0.00390 | **0.00380** | 0.0475 | **0.00760** |

All six: EGNNScalar, hidden 128, 4 layers, 120 epochs, ~19 min each on Betty (measured).
`delta` follows `evaluation.py::choose_delta(mae, k=2)`, the project's pre-registered rule.

### Target values

Quantiles of the **`train_a`** property distribution, so no arm is asked to extrapolate outside
the data (extrapolation would confound the comparison by penalising every arm unequally). This
follows EDM's and EEGSDE's convention of drawing the condition from the training distribution,
but fixes the values so arms are directly comparable.

| property | q10 | q30 | q50 | q70 | q90 |
|---|---|---|---|---|---|
| **mu** (D) | 1.0286 | 1.7345 | 2.4932 | 3.3783 | 4.6627 |
| **alpha** (a.u.) | 65.14 | 71.57 | 75.54 | 79.41 | 85.05 |
| **gap** (Ha) | 0.1912 | 0.2215 | 0.2496 | 0.2790 | 0.3162 |

### Two synthetic properties, as controls

These are not reportable results; they are the tests that catch an implementation bug before it
reaches the real grid. Both are in `proj1/src/linear_property.py`.

- **affine mode** -- `H = 0` exactly. Every SMG-family arm must reduce *exactly* to `tmpd`, and
  `tr(H Sigma)` must be 0 to machine precision. Already verified: `tr(H Sigma) = 0.000e+00`,
  `SMG - scaled plug-in = 5.55e-17`.
- **descriptor mode** -- real, known curvature (measured 3.437). The Hutchinson estimate of
  `tr(H Sigma)` must converge to the exact trace as probes increase.

### Architecture-diverse oracle, optional

`proj1/src/schnet_ref.py` loads a pretrained SchNet at **0.0210 D** on mu -- 4x better than our
`f_B` and a different architecture, so it does not share our EGNN's blind spots. Worth reporting
as a secondary evaluator on the mu column: if an arm's advantage survives being scored by a
different architecture, that is a much stronger claim. Not the primary evaluator, because it was
not trained on our split and its training data is unknown to us.

---

## 3. Protocol

### Stage 0 -- gates (~2 h total, cheap, run first)

Not to save compute, but because each can invalidate work downstream.

| gate | question | cost | what a failure means |
|---|---|---|---|
| **G1 guide resolution** | Are the differences between closures larger than `f_A`'s own error (0.0897 D)? Compute each arm's field on the same states and compare the spread between arms to the guide's error. | ~1 h | If guide error dominates, the whole grid measures noise. Memo section 11 lists this as unresolved. **This is the single most important check in the plan.** |
| **G2 affine null** | Does every SMG-family arm reduce exactly to `tmpd` when `H = 0`? | minutes | An implementation bug in the curvature terms |
| **G3 trace convergence** | Does Hutchinson `tr(H Sigma)` converge to the exact trace on `descriptor` mode as probes increase? | minutes | The probe count is too low to be meaningful |
| **G4 kappa3 stability** | Is the second derivative `dSigma/dx` stable on a *learned* denoiser? Finite differences at three step sizes. | ~30 min | The memo's explicit stop condition for arm 11 |
| **G5 soft-vs-decoded** | How large is the guide's soft-versus-decoded type gap on existing `fm_v1` samples? | ~15 min | Decides whether DBFG is worth implementing at all |

### Stage 1 -- screening and strength tuning (n = 512, ~5-6 h)

Every arm, every property, a reduced target set (q10, q50, q90), and 5 guidance strengths per
arm. Purpose: fix each arm's strength, and catch divergence. **Not reportable** -- at n = 512 the
molecule-stability standard error is 0.022, twice the effect size we care about.

**Elimination rule, deliberately conservative.** An arm is dropped from Stage 2 **only** if it is
an obvious non-competitor, meaning it fails at its *own best strength* on one of:

1. **> 5% non-finite trajectories** -- it diverges;
2. **molecule stability < 0.10** at its best strength, against `fm_v1`'s unguided 0.399 -- it has
   destroyed the model rather than steered it;
3. **property MAE not better than `unguided` by at least 2 standard errors** -- it provides no
   targeting at all. The SE band matters: at n = 512 the MAE standard error is ~0.031 D, so a
   bare point-estimate comparison would eliminate an arm that is genuinely better by 0.03 D
   roughly half the time.

Criteria 1 and 2 are evaluated at **every** strength (an arm that diverges at one strength but is
healthy at another is kept, at the healthy strength). Only criterion 3 is evaluated at the arm's
best strength.

Everything else advances, **including arms that merely tie**. Where two arms are within 2
standard errors, both advance. Where an arm's best strength is within 1 standard error of a
neighbouring strength, **both strengths advance**. The bias is explicitly towards running more,
because a false elimination at n = 512 is unrecoverable and the marginal cost of an extra Stage-2
cell is about an hour.

Every eliminated arm, and the number it was eliminated on, is recorded in the results file.

### Stage 2 -- the reportable grid (n = 4,096)

Survivors x 3 properties x 5 targets, at the strength(s) fixed in Stage 1, 1 seed.
Then **3 seeds** on the mu column only, for error bars on the headline table.

**Why n = 4,096, concretely:**

- It is what **TFG** (NeurIPS 2024) and **TFG-Flow** (ICLR 2025) use per property -- our two
  closest published comparators. Matching them exactly removes sample size as an objection.
- Property-MAE standard error is `sigma/sqrt(n)`, where sigma is the spread of the per-molecule
  absolute error. **`sigma ~ 0.7 D` is an assumption, not a measurement** -- it is taken from the
  order of published MAEs on mu and must be replaced by the measured spread from Stage 1 before
  the Stage 2 sample size is final. With `sigma = 0.7 D`: **0.011 D** at
  n = 4,096 versus 0.031 D at n = 512. Published between-method gaps are ~0.08 D (TFG-Flow 0.88
  vs Cond-flow 0.96), i.e. **7 standard errors at 4,096** but only 2.6 at 512.
- Molecule-stability standard error is `sqrt(p(1-p)/n)`: **0.008** at 4,096 versus 0.022 at 512.
  We measured a seed-to-seed spread of 0.011 at n = 10,000, so 4,096 is the point where the
  error bar drops below the effect.
- EDM, GeoLDM, EEGSDE and PropMolFlow all use 10,000. If Stage 2 comes in under budget, raising
  the mu column to 10,000 is the obvious way to spend the remainder.

### Metrics per cell

Primary: **property MAE against `f_B`** (never `f_A` -- that is the guide, and scoring with it
would be circular), and **fraction within `delta`**. Secondary, all from
`evaluation.py::evaluate_samples`: atom stability, molecule stability, validity, uniqueness,
connectivity, and the measured `Cost` (generator and guide calls, so "at matched compute" is
measured rather than asserted).

### A matched-compute control, because the arms differ by 24x

`unguided` is 34.5 ms/molecule and `smg2_curv` is 836.4 -- a 24x spread. Comparing them at
equal *sample count* silently hands the expensive arms 24x the compute, and every expensive arm
here is an OWN arm. That would make a win uninterpretable.

So every table reports **two columns**: at matched samples, and at **matched compute**. The
matched-compute column gives each cheap arm the extra budget it is owed, spent the way its own
authors would spend it:

| arm | matched-compute variant at the `smg2_curv` budget (836 ms/mol) |
|---|---|
| `plug` | NFE 100 -> 880, or 8 restarts with best-of selection on `f_A` |
| `tfg_mc` | K = 4 -> K = 30 |
| `lgd_mc` / `osc` | K = 4 -> K = 12 |
| `tmpd` | NFE 100 -> 345 |

**If an OWN arm's advantage disappears in the matched-compute column, that is the result**, and
it goes in the paper. The memo's section 2 already names the stop condition: "stop the whole
programme if a strength-tuned plug-in arm matches it at equal cost on terminal metrics."

### Two protocol rules from the memo, adopted

- **Report both samplers.** With a stochastic sampler the reverse SDE absorbs local field error
  and methods an order of magnitude apart in field accuracy land within noise of each other; with
  the deterministic sampler the ordering survives. Reporting only one makes the result an
  artefact of the solver choice. **We do not currently have a stochastic sampler**
  (`proj1/src/sampling.py` is deterministic Euler/Heun). Either add an `eta > 0` variant before
  Stage 2, or state in the write-up that all results are deterministic-sampler results and that
  the ordering is therefore the favourable case for field-accuracy claims. **This is an open
  item, not a solved one.**
- **Equal tuning budget per arm, counted.** Plug-in needs strength ~0.003-0.01 to avoid
  catastrophe and is respectable once tuned. Any comparison at fixed strength overstates the
  winner. Stage 1 gives every arm the same 5 strengths.

---

## 4. Cost and timetable

All per-molecule figures for arms 1, 2, 4, 7 are **measured** on the 5080 at NFE 100
(`results/bench/time_guided.py`); the rest are estimates from call counting and are marked.

**Betty is not faster per job than the laptop** -- 27.4 s/epoch versus 29 s/epoch on the same
training workload, measured. Its advantage is **parallelism**: the grid is embarrassingly
parallel, so a SLURM array job divides wall time by the number of concurrent slots. Each array
task must fit the 4-hour QOS cap; at n = 4,096 the 4-probe SMG arm is ~63 min per cell, so 3
cells per task is safe. **Check `sacctmgr show qos` for the concurrent-job limit before sizing
the array** -- the wall-clock figures below assume 4 concurrent slots and scale inversely.

| stage | what | cells (all arms) | **GPU-hours, serial** | Betty, 4 slots |
|---|---|---|---|---|
| 0 | gates G1-G5 | -- | ~2 | ~2 h |
| 1 | screening + tuning, n = 512, 3 props x 3 targets x 5 strengths | 45 | **19.8** | ~5 h |
| 2 | reportable grid, n = 4,096, 3 props x 5 targets, 1 seed | 15 | **52.9** | ~13 h |
| 2b | mu at the median target, 2 extra seeds, n = 4,096 | 2 | **7.1** | ~2 h |
| -- | **total** | | **~82** | **~22 h** |

Arithmetic: summing the measured ms/molecule over the nine sampling arms gives 3,101 ms, so one
cell across all arms costs `3.101 s x n`. At n = 4,096 that is 3.53 GPU-hours per cell; at
n = 512, 0.44.

**The previous version of this table said ~58 GPU-hours and was wrong**: it costed Stages 1 and 2b
as though every arm were as cheap as `plug`. The corrected figure is ~82, and it assumes
`--n-probe 1`. **At 4 probes Stage 2 alone rises to ~150 GPU-hours**, which is the single largest
budget risk in the plan and is why G3 runs first.

**4-hour QOS cap.** At n = 4,096 and 1 probe the most expensive arm (`smg2_curv`) is 0.95 h per
cell, so 3-4 cells per array task is safe. At 4 probes `smg2` alone is 2.9 h per cell and only one
cell fits per task -- size the array accordingly if G3 forces a higher probe count.

Predictor training is already done (6 x ~19 min, complete). No retraining is required for any
arm except `RCH`, which is deferred.

**The `--n-probe` setting is the single largest cost lever.** Measured: SMG at 1 probe is 437
ms/mol, at 4 probes 921 ms/mol. Stage 0's G3 gate decides the value; if 1 probe suffices, Stage 2
drops from ~41 to ~25 GPU-hours.

---

## 5. Review of the cross-product plan

The plan of "properties x methods" is right, with four adjustments:

1. **Targets are a third axis, not an afterthought.** An arm can win at the median and lose in
   the tails, and that is a real finding about guidance strength rather than noise. 5 targets per
   property, reported per target, not averaged away.
2. **Strength is a nuisance axis that must be collapsed *before* Stage 2**, per arm, per
   property. Collapsing it by picking the best-of-5 inside Stage 2 would be selection on the
   reported metric -- the same error as the checkpoint selection caught in
   `BASE_MODEL_BENCHMARK.md` section 3.4.
3. **The two synthetic properties are not part of the cross product.** They are Stage 0 gates.
   Putting a null control in a results table invites averaging it with real results.
4. **Multiplicity.** The grid is 9 arms x 3 properties x 5 targets = 135 comparisons. At the
   5% level, roughly 7 will look significant by chance. The headline claim must therefore rest
   on a **pre-specified primary cell** -- mu at the median target q50, matched compute,
   deterministic sampler -- with everything else reported as secondary and uncorrected, or on a
   correction stated in advance. Choosing the best cell after the fact is the same error the
   checkpoint selection made in `BASE_MODEL_BENCHMARK.md` section 3.4.
5. **`kappa3` is a column, not a row.** It attaches to whichever arm is running and is logged
   per state. Treating it as an arm would imply it generates samples, which it does not.

---

## 6. Ordering, and what blocks what

```
fm_v1 frozen  --->  G1 guide resolution  --->  Stage 1 screening  --->  Stage 2 grid
   (done)              |                           |
                       +-- G2/G3 (null + trace) ---+   must pass before any SMG arm is trusted
                       +-- G4 (kappa3 stability) ------> gates arm 11 only
                       +-- G5 (soft vs decoded) -------> gates whether DBFG is built at all
```

Nothing in Stage 1 or 2 requires retraining `fm_v1` or any predictor. The optional
feature-scaling retrain (`BASE_MODEL_BENCHMARK.md` section 6) is independent and can run on
Betty in parallel with the whole programme; if it produces a better base, Stage 2 is re-run on
`fm_v2`, which is hours rather than days.
