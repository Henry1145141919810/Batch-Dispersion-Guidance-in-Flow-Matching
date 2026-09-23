# How to benchmark a guidance method

**23 September 2026.** The end-to-end procedure this project uses to take a guidance idea
from "implemented" to "a number in the paper". Written so a new arm can be run by someone
who did not build the harness, and so every arm is measured the same way.

This is the *method* protocol. Two neighbours own pieces of it and are not repeated here:

| document | owns |
|---|---|
| [`DIST_PROTOCOL.md`](DIST_PROTOCOL.md) | the headline target protocol, its reference ladder and its literature backing |
| [`BETTY_RUNBOOK.md`](BETTY_RUNBOOK.md) | how to actually run anything on the cluster, in five steps |
| [`SPLIT_PROTOCOL.md`](SPLIT_PROTOCOL.md) | which model trains on which half, and why |
| [`GUIDANCE_EXPERIMENT_PLAN.md`](GUIDANCE_EXPERIMENT_PLAN.md) | the G1–G5 pre-flight gates and cost model |
| [`SCOPE_FM_GUIDANCE_STATUS.md`](../status/SCOPE_FM_GUIDANCE_STATUS.md) | the FR1–FR7 pre-registration, live status |

---

## 0. The one-paragraph version

Implement the arm behind the existing `guidance_field` interface. Prove it with unit gates
on CPU. Run **preflight** (8 molecules) to prove it does not crash. Run the **compare
stage** (n = 512, every strength, q50 + q90) to find out whether it does anything and at
what strength. Freeze that strength with a **mechanical gate script**, never by eye. Run
the **full stage** (n = 5,000 × 3 seeds, `dist` target, new seeds) to produce the headline
number. Read it with `full_run_table.py` for the pre-registered verdict and
`dist_report.py` for the literature-facing table. Report the chemistry cost in the same
row as the property gain, always.

**Total cost of one arm through this pipeline: ~1 h of your attention, ~30 GPU-hours.**

---

## 1. Before you write any code: what counts as an arm

An arm is a function that returns a guidance **direction** in score units, given the state,
the posterior and the guide. It plugs into `guidance_field()` in
[`proj1/src/guidance.py`](../../proj1/src/guidance.py) as one more `mode`, and it inherits
the shared plumbing: the `Jᵀ` pull-back, the `(1−t)/t` score-to-velocity conversion, the
strength `w`, the velocity-relative clip, the solver and the endpoint policy.

**Everything downstream of the direction is shared and must stay shared.** That is what
makes a comparison a comparison. Two exceptions exist and both are declared:

- `DISPLACEMENT_MODES` (`spbc`, `band`) return a state **displacement**, not a score, and
  skip the `(1−t)/t` factor. Applying it twice would make the edit scale wrongly in `t`.
- `dflow`/`tfg` replace the sampler wholesale (trajectory optimisation, a different model).
  They are not `mode`s and take their own branch in `run_cell`.

If your idea cannot be expressed as "a different direction with the same downstream", say
so explicitly in the design note. It is still runnable, but it is no longer controlled by
the shared plumbing and needs its own cost accounting.

### Strength normalisation — read this before comparing anything

`w` is **not** a shared axis across arms unless you make it one. `strength_scale()`
normalises arms whose natural coefficient lives on a different scale. Measured example:
`btvg` at nominal `w = 4` is applied as **0.0124**, against `plug`'s **4.0** — a factor of
320. Consequences:

- Never compare two arms "at the same `w`". Compare them on a **frontier** (§5) or at
  **frozen strengths** chosen by the gate script (§6).
- An SHG-style schedule must normalise **per phase**. Normalising the whole arm by one
  factor once made a `btvg` phase run 3,139× too strong.

---

## 2. Metrics: what is measured, and which ones decide

Every cell is scored by `evaluate_samples()` in
[`proj1/src/evaluation.py`](../../proj1/src/evaluation.py), on the **held-out evaluator
`f_B`**, never on the guide `f_A`. The full key set written to each cell JSON:

```
n  n_nonfinite  atom_stability  mol_stability  validity  uniqueness_of_valid
unique_valid_per_sample  prop_mae_eval  prop_rmse_eval  in_band_fraction  delta
guide_eval_gap_mean  guide_eval_gap_max  f_A_mean  f_B_mean  target_mean
diversity_mean_pairwise  diversity_logdet  smiles_sample
```

### 2.1 The decision metric

> **Primary:** `in_band_fraction` gain over `unguided`, **subject to
> `mol_stability` ≥ 0.9 × unguided**, at matched compute.
> **Secondary, in the same row:** `|bias|` and `spread`, both in δ units.

`δ = 2 × f_B's validation MAE` — the project's pre-registered acceptance tolerance
(0.168 D for μ, 0.481 Bohr³ for α, 0.0076 Ha for gap).

**Never rank on MAE alone.** Measured consequence: `plug` ranks **1st on MAE and 11th once
the chemistry floor is required**. MAE is a fine statistic — in-band is a binomial with
se ≈ 0.013 at n = 512 and separates little — but ignoring chemistry inverts the answer.
Always show the chemistry-constrained column.

### 2.2 Why chemistry must be in the same row

Raising `w` lowers MAE *and* wrecks molecules, monotonically. Measured on μ/`plug`:
MAE 1.22 → 0.78 while molecule stability falls 0.416 → 0.305. So "lower MAE" is a
**position on a known trade-off**, not an achievement. Any claim of the form "arm X beats
arm Y" must be read at matched chemistry or it is measuring strength, not method.

### 2.3 Diagnostics that catch specific failure modes

| metric | catches |
|---|---|
| `n_nonfinite` | the arm blew the trajectory up; those samples count against every metric and are excluded from property averages |
| `clipped_sample_steps` | the arm is reading the **clip**, not the arm. `btvg` at w=4 hits it on 29 % of guided steps against `plug`'s 2 % |
| `guide_eval_gap` | `f_A` and `f_B` disagree — the reward-hacking diagnostic |
| `hack_rate` (dist) | guide says in-band, evaluator says no |
| `diversity_*`, `uniqueness_of_valid` | concentration bought by mode collapse |
| `schedule_used` | a scheduled arm whose `active()` never fired, i.e. it silently ran unguided |
| `cost` / `field_evals` | "at matched compute" is measured, not assumed |

**Every one of these exists because it caught a real bug.** An arm that reports an
identical number at every strength is the signature of a dead-code path — that is how the
`band` and SHG strength bugs were both found.

### 2.4 For the `dist` target, add the reference ladder

`dist` hands the arm the atom count, which alone carries much of the property. Raw MAE is
uninterpretable until placed against references that each know a different amount — see
[`DIST_PROTOCOL.md`](DIST_PROTOCOL.md) §3. `dist_report.py` adds `gap_closure`,
`beats #Atoms`, `partial_corr` (tracking *beyond* atom count) and the joint
`distinct valid in-band per attempt`.

---

## 3. Stage 0 — gates on CPU, before any GPU time

Cheap, and they fail loudly. Run from the repo root:

```
python proj1/tests/test_identities.py
python proj1/tests/test_guidance.py
python proj1/tests/test_v2_arms.py
python proj1/tests/test_full_run.py
```

Expect `ALL PASS` from each. A new arm should add its own gate file. What a good gate
asserts — all of these have caught shipped defects:

1. **A null case reduces to a known arm exactly.** `smg` with `H = 0` must equal `tmpd`;
   BTVG's mean term is bit-identical to `plug` at 9e-8 relative. A reduction that holds to
   float precision proves the wiring.
2. **The arm varies with its own hyperparameters.** If `τ` does nothing, say so *now*
   rather than after 30 cells. (It did nothing: `τ²/V_F ≈ 0.001–0.004`.)
3. **Equivariance and masking.** Padded atoms stay exactly zero; rotating the input rotates
   the output. Measured error should be ~1e-15, not ~1e-6.
4. **Units.** A displacement arm must not be pushed through the score-to-velocity factor.
5. **The estimator is unbiased** against an exact computation where one exists.

Consider also the two structural pre-flight checks in
[`GUIDANCE_EXPERIMENT_PLAN.md`](GUIDANCE_EXPERIMENT_PLAN.md):

- **G1, guide resolution** (`g1_guide_resolution.py`, ~3 min): do arms differ by more than
  `f_A`'s own noise? Partial result: they collapse onto each other late in the window
  (between-arm cosine 0.98 at t = 0.95 against a between-guide 0.76).
- **Velocity share** (`velocity_share_diag.py`, ~1 min): the guidance correction is ~7 % of
  the sampling velocity, so **any intervention that only re-points the direction cannot
  move the trajectory.** If your idea only changes direction, this predicts a null before
  you spend anything.

---

## 4. Stage 1 — preflight: prove it runs

```
python proj1/scripts/guidance_sweep.py --stage compare --arms <yourarm> --preflight
```

One throwaway cell per arm at **n = 8, 10 steps**, writing nothing, exiting non-zero on any
exception. ~1 minute on a GPU.

**Two details that are not arbitrary.** `steps = 10`, not 4: at 4 the time grid is
{0, .25, .5, .75}, so scheduled handoffs at 0.80–0.90 are never entered and a scheduled
arm's preflight silently equals `plug`'s. And preflight belongs **inside the SLURM job**,
not on the login node — the cluster scripts run it under `set -e` before the sweep, so a
broken arm kills the job in two minutes having written nothing.

This exists because an arm with a shadowed variable (`rch`) crashed on **every** cell, and
the sweep's per-cell error handling turned that into a four-hour window of `.failed` files.

---

## 5. Stage 2 — the compare stage: does it do anything, and at what strength

```
sbatch --export=ALL,SWEEP_PROPS=<prop>,STAGE=compare proj1/cluster/guidance_sweep.slurm
```

**258 cells** for the seven-arm compare set. Per property, in three passes, ordered so a
partial run is still a complete experiment:

| pass | what | why in this order |
|---|---|---|
| 1 | every arm at the default strength, q50 | one link gives a full arm comparison before any tuning |
| 2 | every arm × 7 strengths `{0.01, 0.05, 0.25, 0.5, 1, 2, 4}`, q50 | the strength curve |
| 3 | every arm at default strength, q90 | the steering task |

Fixed: **n = 512**, 100 Euler steps, `t_min_guide = 0.5`, seed 20260921, clip 1.0, one cell
per JSON, resumable (a cell with a JSON is skipped).

### 5.1 Why two targets, never pooled

| task | tests | distance from the generator's own mean |
|---|---|---|
| **q50** | concentration | 0.05–0.27 sd — almost no steering needed |
| **q90** | steering | 1.1–1.6 sd into the tail |
| **dist** | the published protocol | 0.80–0.85 sd, per molecule |

They answer different questions and are reported separately (FR7). An arm can win one and
lose the other, and that *is* the result.

### 5.2 Reading a strength curve honestly

Comparing arms at their **best** strength is a max over a grid: upward-biased, and biased
toward whichever arm has more grid points. Either say so, or use a fixed strength.

If the best cell sits on the **grid edge** (w = 4), the optimum is not bracketed and the
curve is not a curve — extend the grid before quoting it. This actually happened: FR3's
freeze landed on w = 4 for three arms.

### 5.3 Screening for which arms continue

```
python proj1/scripts/select_arms.py --stage compare
```

**The default is KEEP.** An arm is dropped only if, on *every* property, it is both (a) more
than σ combined standard errors worse than the best arm on MAE **and** (b) no better than
`unguided` on *either* metric. Chemistry alone never drops an arm — an arm that wins the
property and costs stability is a trade-off to re-tune at lower strength, not a
non-competitor.

Asymmetric on purpose: at n = 512 a false elimination is unrecoverable and an extra Stage-2
cell costs about an hour.

---

## 6. Freezing the strength — mechanically, never by eye

```
python proj1/scripts/check_fullrun_go.py --stage compare --target q90 --json-out frozen.json
```

Written **before any compare cell existed**, so the decision cannot be tuned after seeing
numbers. It reads only cell JSONs, costs seconds, and imports no torch (safe on a login
node). It refuses a partial grid — an earlier version accepted one clean cell per arm and
would have frozen a best-of-half-a-sweep.

Two frozen sets are written, and the distinction matters:

| key | rule | role |
|---|---|---|
| `frozen_w` | **FR3a**: best MAE among strengths clearing the chemistry floor; most stable if none do | **the headline set** |
| `frozen_w_mae` | **FR3** as registered: best MAE, floor ignored | secondary, one seed |

FR3a is an amendment made *after* the screen and is labelled as such, because FR3's own
choice (w = 4) failed the floor it would then be judged against, which would have made the
headline table unjudgeable.

**Why freeze at q90's strength for a `dist` run:** what matters is not where targets sit
but how far each is from the *generator's* mean, which is a mean-absolute-deviation:
dist demands 0.80–0.85 sd against q50's 0.05–0.27. Using q50's strength was wrong by
3–18× and was corrected before any dist cell ran.

---

## 7. Stage 3 — the full run: the headline number

```
FULL=$(sbatch --parsable --array=0-8 --export=ALL proj1/cluster/full_run.slurm)
sbatch --dependency=afterany:$FULL --export=ALL proj1/cluster/full_run.slurm
```

Nine array tasks (3 properties × 3 seeds), ~2.5 h each, plus one no-array **insurance
sweeper** that finishes whatever the array left and exits in minutes if nothing remains.

Fixed: **n = 5,000**, 3 seeds, `dist` target, every arm once at its frozen strength, new
seeds so the strength choice is not also the evaluation, `--per-mol` so every cell writes a
`.permol.pt` sidecar.

**Seeds vary the initial noise only.** All seeds and all arms see the *same* 5,000 test
molecules, sizes and targets — that is what makes every comparison paired.

The job re-runs the gate itself at start, so it is safe to submit before the compare stage
finishes: it stops with `STOP: the compare stage is incomplete` rather than guessing.

### Reading it

```
python proj1/scripts/full_run_table.py              # the pre-registered verdict
python proj1/scripts/dist_report.py --dir "results/full/n5000/seed*"
```

- **`full_run_table.py` is the verdict.** FR5: an arm beats another only at an
  **independent-samples z ≥ 3**, with paired z supplementary. It refuses to print a partial
  or inconsistent run and says what is missing.
- **`dist_report.py` is the literature-facing view**: the EDM/EEGSDE ladder, gap closure,
  tracking beyond atom count, published units, strata by size and steering demand. Its
  paired tests are supplementary. Where the two disagree, FR5 is the answer.

---

## 8. Statistics: the rules that keep a result real

1. **Pair everything you can.** Same molecules, same targets, same initial noise across
   arms. Match on `(seed, mol_idx)`, **never on row position**, and require matched rows to
   have equal targets. That guard exists because a batch loop once sliced every batch's
   targets from index 0 — which would have guided 75 % of molecules toward another
   molecule's target while scoring them against their own.
2. **Pool seeds; do not test them separately.** Seeds are replicates of one set, so a
   comparison is one test on pooled rows. Pairing buys variance, not the point estimate.
3. **Uncertainty is computed, not assumed.** In-band is binomial (`se = √(p(1−p)/n)`,
   Wilson CI); MAE's sd comes from the reported RMSE (`|e|` has mean MAE and second moment
   RMSE² by definition); bootstrap the mean where the shape is unknown.
4. **Differences under ~1.4 σ are ties.** Report them as ties, never as an ordering.
5. **Correct for multiplicity within a pre-declared family** (Holm). Declare the family
   *before* looking. A family of 49 arbitrary strength pairs has no power and no meaning.
6. **Regress a frontier; never interpolate between neighbours.** With n = 512, stability's
   own se is 0.021 and the MAE-vs-stability curve is visibly non-monotonic. Pointwise
   interpolation converts that noise into apparent significance — it produced fake gains of
   "10 se" once. Fit the whole curve.
7. **Divergence never wins.** Cells with non-finite samples, or MAE above 10× unguided, are
   excluded from an arm's best cell and the arm is reported as divergent at that strength.

---

## 9. Reporting: the minimum an arm's result must carry

- Property gain **and** chemistry cost, in the same row.
- The strength, and whether it was frozen, best-of-grid, or on the grid edge.
- Cost (`cost`, `field_evals`) so "matched compute" is measured. Arms differ by ~9×.
- Which target protocol (q50 / q90 / dist) — never pooled.
- n, seeds, and whether the comparison is paired.
- Anything decided **after** seeing data, labelled post hoc. FR3a and the epoch-1500
  checkpoint amendment are both labelled; so is every `tfg` number.
- The ablation ladder where the arm is a sum of terms: each term alone, neither, both. For
  BTVG that is `unguided / plug / btvg_var / btvg`, and it is clean precisely because
  `plug` **is** the mean term.

### Failure modes this protocol is built to prevent

| failure | the check that catches it |
|---|---|
| ranking on MAE and inverting the answer | chemistry floor in the rubric (§2.1) |
| "better" that is really "stronger" | frontier / frozen strengths (§5.2, §6) |
| an arm silently running unguided | `schedule_used`, strength-varies gate (§2.3, §3) |
| reading the clip instead of the arm | `clipped_sample_steps` (§2.3) |
| the guide's blind spots scored as success | `f_B` disjoint; `guide_eval_gap`, `hack_rate` |
| noise promoted to significance | regress the curve; pre-declared family; 1.4σ tie rule (§8) |
| tuning the decision after seeing data | gate scripts written before cells exist (§6) |
| a four-hour window spent writing `.failed` | preflight inside the job (§4) |

---

## 10. Cost model

Measured ms/molecule at NFE 100 on the 5080: `unguided` 34.5, `plug` 95, `tfg_mc` 126,
`tmpd` 242, `lgd_mc` 271, `smg` 422 (1 probe), `smg2` 803, `smg2_curv` 836. `btvg` ≈ 3.5×
`plug`; `dflow` ≈ 9× `plug` (578 s/cell at n = 512 on a B200).

| stage | cells | scale | GPU-hours |
|---|---|---|---|
| gates + preflight | — | CPU + n=8 | ~0 |
| compare | 258 | n = 512 | ~8 |
| full | 63 | n = 5,000 × 3 seeds | ~24 |

**The single largest cost lever is `--n-probe`** for the SMG family: 1 probe vs 4 is
437 vs 921 ms/mol. Decide it with the G3 trace-convergence gate, not by default.

Betty is **not faster per job** than the laptop (27.4 vs 29 s/epoch measured); its
advantage is parallelism. Every job must fit the **4-hour QOS cap**, so long runs are
chains of resumable links.

---

## 11. Checklist

```
[ ] arm implemented behind guidance_field; downstream plumbing untouched
[ ] strength normalisation declared; w comparable or explicitly not
[ ] unit gates: null reduction, hyperparameter sensitivity, equivariance, masking, units
[ ] G1 / velocity-share run if the idea is direction-only
[ ] preflight passes (n=8, 10 steps), inside the job
[ ] compare stage: 258 cells, q50 + q90, 7 strengths, n=512
[ ] optimum bracketed, not on the grid edge
[ ] select_arms.py run; drops justified
[ ] strength frozen by check_fullrun_go.py; FR3a vs FR3 recorded
[ ] full run: n=5000 x 3 seeds, dist, --per-mol, new seeds
[ ] full_run_table.py verdict (FR5, independent z >= 3)
[ ] dist_report.py ladder for the literature-facing table
[ ] chemistry cost in the same row as every property gain
[ ] post-hoc decisions labelled
[ ] ablation ladder for multi-term arms
```
