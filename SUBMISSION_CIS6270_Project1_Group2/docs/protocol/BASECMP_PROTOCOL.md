# The base-model comparison: one external pair, two generators

**25 September 2026. Pre-registration. Nothing here has been run.** Every rule
below is fixed before the first cell exists, including the sizing rule, so no
choice in it can have been made to fit a number that had already been seen.

Henry's instruction, 25 Sep: use the best f_A / f_B (not ours); compare our base
model against EquiFM across all seven arms; screen strength **and** the
guidance start-time jointly; report the best under the chemistry floor **and**
the best without it, each with its own start-time; store the sweep table; then
run full-protocol-v2 on **EquiFM** at both picks; queue it all at once.

---

## 1. The question

For each guidance method, how much of its measured behaviour is a property of the
**method**, and how much is a property of the **generator** it was developed on?

Nothing in `guidance_sweep.py` can answer this. There, generator, guide and
evaluator all come from one pipeline built by one group on one preprocessing of
QM9. The transfer experiment (`TRANSFER_EXPERIMENT_PLAN.md`) asked the adjacent
question — does a method survive a borrowed generator *and* a borrowed property
pair — and answered it only on EDMsecond, a diffusion model dropped on 23 Sep
when this project chose flow matching.

This protocol isolates the generator. **Both base models are run through the
same external property pair, the same delta, the same targets, the same sampler
settings, the same window grid and the same strength grid.** The only thing that
differs between the two columns is the generator. That is what makes a difference
between them attributable to it.

---

## 2. f_A and f_B: which pair, and why it is the best available

**Guide f_A = TFG `tf_predict_<p>` (evaluated at time channel 0). Oracle f_B =
EDM's classifier, TFG `evaluate_<p>`. Second oracle = OC-Flow's clean EGNN.**
Neither is ours, and this is the pair `transfer_sweep.build_pair` already builds.

Henry asked for "the best, definitely not ours". It is worth writing down that
this is not a courtesy to the reviewer — the external networks are simply
better, on every property, in both roles:

| role | network | mu (D) | alpha (Bohr³) | gap (Ha) |
|---|---|---|---|---|
| guide | **TFG `tf_predict`** | **0.044** | **0.092** | **0.0025** |
| guide | ours, `f_A_<p>.pt` | 0.0897 | 0.2447 | 0.0039 |
| oracle | **TFG `evaluate_`** (EDM classifier) | **~0.07** | **0.080** | **0.0019** |
| oracle | ours, `f_B_<p>.pt` | 0.0840 | 0.2407 | 0.0038 |
| oracle 2 | OC-Flow clean EGNN | 0.052 | 0.096 | 0.0024 |

The external guide is 1.6–2.7x more accurate than ours and the external oracle
1.2–3.0x. **Read the margin, not the digits:** ours are validation MAEs read off
our own checkpoints, the external ones are the numbers recorded in
`TRANSFER_EXPERIMENT_PLAN.md` §10.2 measured on our test split, so the two
columns are not measured identically. The gap is 2–3x on alpha and gap, far
larger than any plausible val/test discrepancy, so the ranking does not turn on
that. There is no third option: EquiFM ships no usable property pair of its own
(`EQUIFM_USABILITY_AUDIT.md` §3), which is why the audit recommended TFG's.

**Guide and oracle are disjoint, but by inference, not by construction.** Both
TFG arg files record `dataset: qm9_second_half` and their training script mutates
`args.dataset` after building the loader, so the string is unreliable;
`audit/fa_fb_search/disjointness_test.py` finds the error structure of a disjoint
pair. That is strong and circumstantial. Our own pair is disjoint by
construction. **This is the main weakness of every number produced here and it
cannot be fixed from the released artifacts.** It is, however, a weakness shared
equally by both columns, so it does not bear on the comparison between them —
only on the absolute values.

---

## 3. delta: the local rule, and why it is the same number for both bases

**delta = 2 x f_B's MAE over `val` molecules whose TRUE property lies in
[Q_train_a(0.85), Q_train_a(0.95)].** The rule settled on 24 Sep
(`proj1/scripts/local_fb_mae.py`; status doc §6 "How delta is set"), applied here
to the external oracle. Implemented as `transfer_sweep.local_delta`, selected
with `--delta-mode local` (the default).

The recipe transfers; **the three numbers do not** and are recomputed against
TFG's oracle at run time. For reference, on *our* f_B the same recipe gives
0.16992 / 0.46754 / 0.00739 against the pre-registered 0.16799 / 0.48135 /
0.00760.

**Pre-registered as THE delta for this run.** For the v2 full run the local delta
is post-hoc and sits beside the pre-registered one. Here it is chosen before any
cell exists, which makes it a legitimate pre-registration — main-2's point, and
this document is the "say so explicitly" part of it.

Selection is by the **true** property, not by f_B's prediction: regression to the
mean makes the two differ in the tail, so selecting on the prediction would
measure a different quantity.

**Why it is identical for both generators, which matters more than its value.**
Nothing in `local_delta` touches a generator — it is f_B's error on real QM9
molecules. A delta recomputed per base (say, from each generator's own samples)
would make the in-band bar easier for one generator than the other and would
destroy the comparison while looking more careful. The guarantee is structural
and gated: `test_basecmp.py::delta_is_generator_free` asserts the function has no
parameter through which a generator could reach it.

**Two limits to carry, in order of size.**

1. Both deltas are f_B's error on **real** QM9 molecules. In-band scores
   **generated** ones, of which only ~35–40 % are molecule-stable, and f_B's
   error on that population is unmeasured and probably larger. This dwarfs the
   1–3 % the local window moves delta.
2. **delta is optimistic here in a way it is not for our own pair.** Our split
   protocol guarantees `val` is not in f_B's training data. TFG's `evaluate_<p>`
   trained on an unknown ~50 % of QM9, so some of these val molecules are very
   likely in its training set and the measured MAE is low — delta is *tighter*
   than it should be. A tighter band lowers in-band for every arm on both bases
   equally, so it favours no arm and no generator; it makes absolute coverage
   numbers pessimistic and they must be read that way.

k = 2 is justified by measured coverage, not convention: 88–90 % of molecules
truly near the target have f_B within 2 x MAE of the truth, against 61.6–67.9 %
within 1 x. A one-MAE band would miss a third of genuine hits.

---

## 4. The targets: v2's, unchanged

The fixed q90 target per property, exactly the numbers `FULL_RUN_V2_PROTOCOL.md`
§2 registers and `guidance_sweep.TARGETS[<p>]["q90"]` holds:

| property | target | unit |
|---|---|---|
| mu | 4.6627 | Debye |
| alpha | 85.05 | Bohr³ |
| gap | 0.3162 | Hartree |

Sizes come from `val[:n]`, as v2 does for any fixed target, so the test split is
untouched by this experiment. Every arm, seed and base model sees the same sizes
and the same target, so all comparisons are paired at the molecule level.

---

## 5. The grid: strength and start-time, jointly

| axis | values | why |
|---|---|---|
| strength `w` | 0.05, 0.25, 1, 4, 16 | log-spaced, five points. **It reaches past w = 4 on purpose.** v1's grid stopped at 4 and capped `lgd_mc`, which still cleared the floor there; that failure is designed out rather than discovered again. |
| start-time `t_start` | 0.05, 0.25, 0.5, 0.75 | the flow-time instant guidance switches on; guidance runs on `[t_start, 1)`. This is the project's **largest measured single effect** (alpha MAE 10.02 → 5.32 → 6.59 as `t_min_guide` goes 0.05 → 0.5 → 0.75) and **nothing between 0.05 and 0.5 has ever been measured** (`guidance_sweep.py:276-281`). 0.25 closes that gap. |

**One number, both clocks.** Flow time runs 0 (noise) → 1 (data) and guidance is
on for `t >= t_min_guide`; EquiFM's and VP's time runs 1 → 0 and guidance is on
for `tau <= tau_max_guide`. The same physical instant is therefore `t_start` on
one clock and `1 - t_start` on the other. `t_start` is the primary record in
every cell. Getting this mirror backwards would guide the wrong half of the
trajectory on one base, nothing would raise, and the comparison would report it
as a property of the generator — so it is gated
(`test_basecmp.py::window_mirror`).

**Cells.** 6 guided arms x 5 strengths x 4 start-times x 3 properties = 360, plus
**one** unguided cell per property = **363 per base model**, 726 in total. Split
across 96 array tasks by (backend, arm group, property, start-time), with four arm
groups — `plug+tmpd`, `lgd_mc+tfg`, `btvg`, `btvg_var` — because the two costly
arms at the earliest window would otherwise run past the 225-minute guard.

**Unguided is planned once per property, not once per window.** With no guidance
the window is not read by anything, so a second unguided cell would be the same
computation under another name — and it would then be averaged into the chemistry
floor as though it were independent evidence, or set the floor from the wrong
cell. Exactly one array task writes it. Gated
(`plan_unguided_once`, `slurm_screen_tiles_grid`).

**Refine.** After the screen, each pick gets the geometric midpoints to its
neighbours on the log grid; a pick at an edge is extended outward by x4 and x16
instead (capped at w = 64 — extending w = 16 by x16 would ask for 256, where the
sampler produces rubble no oracle reading makes interesting). A pick still at an
edge after the refine is reported as `grid_edge`, not hidden. The start-time axis
is **not** refined by default (`--refine-t` enables it); the 4-point grid is
itself a reported result, and refining it roughly doubles the stage.

---

## 6. The two picks

For every (property, arm), over the **joint** (w, t_start) grid:

- **`floor`** — the highest decoded in-band among cells whose molecule stability
  is at least **0.9 x the unguided generator's**. This is v2's rule V2 and the
  **only** pick a headline may rest on.
- **`free`** — the same with the floor removed. **Not a headline.** At high
  strength these cells are frequently chemical rubble that happens to satisfy a
  property oracle. It is reported because without it a method that is merely
  *throttled* by the floor cannot be distinguished from one that cannot steer at
  all, which is a distinction this project has cared about since the `lgd_mc`
  grid-ceiling finding.

**The metric is in-band, decoded** (`in_band_fraction_dec`). v2's rule V2 exists
because v1 selected on MAE and scored on in-band — a mismatch with no upside.
Decoded, because an arm can push the continuous atom-type channels without moving
the molecule that exists; the soft number is reported beside it and never
selected on. Note this differs from `transfer_sweep`'s older FR3a, which selects
on decoded **MAE**; that is v1's rule and is deliberately not used here.

**Tie-breaking, fixed before any cell was read: smaller strength first, then
LARGER `t_start`.** Both directions mean less intervention — a later switch-on
guides fewer steps — so a tie always resolves toward the cheaper, gentler cell.
v2 already sends strength ties to the smaller strength; this extends that
principle rather than inventing a second one.

**Statuses that travel with every pick.** `floor_limited` (no cell on the grid
clears the floor — the arm cannot steer this base within the chemistry budget,
which is a result, not a win), `grid_edge`, and `collapse_contaminated` (v2's
rule V6: distinct valid molecules per attempt below 0.95).

**Selection is not evaluation.** Picks are made on the screen, one seed. The full
run scores fresh seeds, so a pick cannot inflate the number it is reported at.

---

## 7. The full run: v2 on EquiFM, at both picks

Per Henry's instruction the v2 full run goes on **EquiFM**. Settings are v2's
(`FULL_RUN_V2_PROTOCOL.md` §4) except where noted:

| setting | value |
|---|---|
| target | fixed q90 per property |
| n | chosen by the sizer (§8); v2 registers 5000 |
| seeds | 20261001, 20261002, 20261003 — v2's, so noise is comparable |
| sizes | `val[:n]`, identical for every arm, seed and set |
| sampler | EquiFM's native 100-step Euler grid, window per arm, velocity clip 1.0 |
| arms | all seven |
| strength sets | **both** `floor` and `free`, each at its own frozen `(w, t_start)` |
| sidecars | `--per-mol` on |

Cells where the two sets picked the same `(w, t)` are computed **once** and
reported under both, so the run never pays twice for one number
(`full_plan_dedup`).

**Reported per v2, by `proj1/scripts/basecmp_table.py`** (the `table` stage):
the full metric block including every `_dec` twin (V4); error-ranked buckets at
10 / 50 / 100 % from the per-molecule sidecars, always labelled descriptive and
never as an achievable yield, since they select molecules with the same oracle
that then scores them (V5); distinct valid in-band molecules per attempt beside
every in-band number (V6); verdicts on in-band between arms that both clear the
floor at z >= 3, Holm-adjusted within each property (V7–V8). It refuses a partial
run, because pooling a different number of seeds per arm makes the standard errors
incomparable and the verdict rests on exactly those.

**It needs its own reader, and that is worth knowing.** `full_run_v2_table.py` and
`full_run_table.py` cannot read these cells, by three independent mechanisms: they
skip any cell whose `stage` is not `"full"` (these carry `"basecmpfull"`), they key
results by (property, arm) so an arm's `floor` and `free` cells would collide, and
their consistency block refuses a directory with mixed `t_min_guide` — which every
basecmp full directory has by construction, because each arm carries its own frozen
window. The upside of the same fact: a basecmp cell can never silently contaminate
the real v2 table.

**Two things that must travel with every number from this run.**

1. **`--save-coords` does not exist on this path.** `transfer_sweep` writes
   per-molecule sidecars but not coordinates, so the V5 buckets (which need only
   per-molecule f_B and y) are available and any later analysis needing geometry
   is not. Said here rather than discovered later.
2. **It is not comparable to the existing v2 full run.** That run used *our*
   f_A/f_B on *our* base; this one uses the external pair on EquiFM. Different
   guide, different oracle, different delta, different generator. Numbers from
   the two may not be put in one table.

---

## 8. Cost, and how `n` is chosen

**The probe exists because EquiFM has never had a single cell run on it** — the
backend was built and gated on 23 Sep and nothing was ever sampled (status doc:
"EquiFM: harness ready, 0 cells"). Every cost estimate before the probe is an
extrapolation from a different network, and sizing a large queue on one is how a
run gets killed half way through and has to be resubmitted.

So `STAGE=probe` runs 19 real cells per base at n = 256 (7 arms at 3 start-times, unguided once), `STAGE=size` reads their
recorded `seconds`, and every later job reads its `n` from
`results/basecmp/plan.json` at run time. The whole chain is submitted in one go;
the sizing happens *inside* it.

**The rule, fixed here:** the largest `(n_screen, n_full)` on the ladders
`n_screen ∈ (1000, 750, 500, 250)` and `n_full ∈ (5000, 2000, 1000, 500)` whose
estimated cost fits `--budget-gpuh`, screen size preferred first. Cell time is
modelled as proportional to `n` at fixed batch, with a 1.25x margin.

**`n_full` is floored at 1000 and the chain refuses rather than going below it.**
v2 measured the standard error of an in-band difference at n = 5000 x 3 seeds as
~0.0025, against best-vs-worst spreads of 5–13 sigma. At n = 500 x 3 that se is
~0.008 and the same spreads fall to 1.6–4 sigma, mostly below v2's own z >= 3
bar. A run smaller than that would cost real GPU time and answer nothing, which
is worse than a run that is honestly smaller in scope.

**The cost model is per guided step, not per cell.** Guidance cost scales with
the guided fraction of the trajectory: at `t_start = 0.05` about 95 of the 100
Euler steps are guided, against 50 at 0.5 and 25 at 0.75. Seconds per molecule is
therefore fitted as `a + b x (1 - t_start)` per (backend, arm) across the probe's
three start-times — a flat model fitted at one window under-prices every earlier
window, which is how an array task ends up past the wall having finished nothing.
On the current estimate the earliest window costs **3.2x** the latest.

Estimated surface, from the EDMsecond cells that actually ran (13.17 GPU-h for 216
cells at n = 512 — 13.17 to be exact) decomposed into sampling and guidance cost and scaled to our net
by v2's measured throughput — **to be replaced by the probe's own numbers**:

| | n_full 5000 | 2000 | 1000 |
|---|---|---|---|
| n_screen 1000 | 240 | 157 | 130 |
| n_screen 500 | 189 | 106 | **79** |
| n_screen 250 | 163 | 81 | 53 |

GPU-hours, margin included. Wallclock is this divided by the number of mig45
slices running at once. The default `BUDGET_GPUH=96` selects the bolded cell;
`BUDGET_GPUH=130` buys the screen at the full n = 1000.

The sizer also reports the most expensive single array task against the 225-minute
guard. On the estimate above the worst is the EquiFM `lgd_mc+tfg` task at
`t_start = 0.05`, at 68 minutes — comfortably inside it, which is what the
four-way arm split is for.

---

## 9. What is held fixed, and what differs

**Fixed across both base models, so a difference is attributable to the
generator:** f_A, f_B, the second oracle, all three calibrations, delta, the
targets, the size distribution, the strength grid, the start-time grid, 100 Euler
steps, velocity clip 1.0, the chemistry-floor multiplier, the selection metric,
the tie-break, the seeds.

**Differs, unavoidably, and must be disclosed with every number:**

1. **The path geometry.** Our base is a linear flow path with independent
   Gaussian noise; EquiFM's is a hybrid, geometry-aligned path (coordinates
   near-linear with Kabsch/Hungarian-aligned noise, types on a VP schedule
   β 0.1→20). `plug` uses no covariance and is exact on both; `tmpd`, `lgd_mc`,
   `btvg` and `btvg_var` inherit a coordinate-block **approximation** on EquiFM
   (`TRANSFER_EXPERIMENT_PLAN.md` §10.3). Every released QM9 flow-matching
   checkpoint aligns its coordinate noise, so this is not "exact vs approximate"
   — it is which approximation, stated.
2. **Type-channel scale.** Our sampler works in raw one-hot (divisor 1),
   EquiFM's in its own normalised space. `build_pair` reads each network's
   divisor off its own checkpoint and converts; assuming they coincide is how an
   earlier version fed both networks features 2x wrong while stability and
   validity — which use argmax — could not see it.
3. **EquiFM's charge channel** is advanced by the generator and never guided or
   scored; its noise floor means its native endpoint is data + O(1e-4),
   reproduced faithfully rather than replaced.

   **A guard that did not exist, and now does.** `build_pair`'s calibration-slope
   check (0.9–1.1x QM9's MAD) is fitted on molecules read from the data file, so it
   **cannot** see a wrong sampler feature divisor — that only affects what f_A and
   f_B are fed at *scoring* time, on generated molecules, and a wrong divisor
   leaves every slope perfectly in range with every number downstream quietly
   wrong. `basecmp_freeze.scale_sanity` adds the check the earlier bug was actually
   caught by: the oracle's mean on unguided samples against QM9's own train_a mean.
   Threshold measured, not guessed — across the 80 unguided cells this project has
   run on two generators the gap is at most **0.62 MAD** (median 0.17), the
   historical bug sat at **1.39**, and a clean factor-of-2 error on mu lands near
   **2.25**, so the line is drawn at 1.0 MAD. It prints rather than refuses,
   because EquiFM has never been measured here and could legitimately sit further
   out than any generator we have seen.
4. **Neither unguided row is a published number.** EquiFM's paper sampled
   dopri5; ours is our own model. Both are re-measured here under one sampler.
5. **`tfg` on EquiFM** runs TFG's schedules on the coordinate clock, a stated
   choice (§11.1 of the transfer plan), gated by 10 of the backend's 46 tests.

---

## 10. What is not claimed

- Not a continuous optimum in `(w, t)`. It is the best cell on a five-by-four
  grid, refined once on the strength axis. `grid_edge` flags where the optimum
  may lie outside it.
- Not a statement about the *order* of methods on any third base model.
- Not comparable to the existing v2 full run (§7), nor to EDMsecond's stranded
  216-cell compare stage, which used a different generator and no `tfg` arm.
- The `free` picks are not achievable-quality results and are never headlined.
- The error-ranked buckets are descriptive shape, never a yield.

---

## 11. Gates and provenance

`proj1/tests/test_basecmp.py` — **30 gates**, CPU, seconds, on synthetic cells
whose right answer is fixed by construction: the cell-name compatibility
guarantee, the window mirror, the plan's shape, all four pick rules, both
tie-breaks, decoded-over-soft, the refine rules, five refusals (mixed n, two
deltas for one property, a hole in the grid, no unguided cell, a cell from the
other base), the frozen file's backend guard, the full plan's de-duplication, and
that the 96 screen tasks and 36 full-run tasks **tile their grids exactly** — the
bash arithmetic re-derived in Python and checked against the whole plan. That
last gate caught a real defect during development: the plan prepended the
reference window unconditionally, so every array task would also have run the
reference window's cells — 180 duplicated cells with several jobs writing one
filename at once.

Also still passing, unchanged by this work: `test_equifm_backend.py` (46) and
`test_transfer_protocol.py` (22).

Every cell records the generator's md5, both clocks' window, `t_start`, the
delta and its mode, the full calibration report for guide and both oracles, the
cost counters and the wall time. Every asset is verified against a pinned hash
before anything is sampled, our generator included
(`a190ac8394902027d4a951f8d30e8c5c`).

**Files.** `proj1/scripts/transfer_sweep.py` (the `fm` backend, the `basecmp*`
stages, `local_delta`), `proj1/scripts/basecmp_freeze.py` (the picks, the table),
`proj1/scripts/basecmp_size.py` (the sizer), `proj1/scripts/basecmp_table.py` (the v2 reporting for this stage's cells),
`proj1/cluster/basecmp_run.slurm`, `proj1/cluster/submit_basecmp.sh`,
`proj1/tests/test_basecmp.py`.
