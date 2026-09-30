# Full run v3: one strength, one target, four base models

**26 September 2026. Pre-registration. Nothing here has been run.** Henry's
decisions of 26 Sep are folded in verbatim; where this document and an older
protocol disagree, v3 governs the v3 tree and nothing else.

**v3 REPLACES v2 AS THE HEADLINE.** v2's cells are not deleted and not hidden —
they were pre-registered and they ran, and [their write-up](../results/FULL_RUN_V2_RESULTS.md)
stays where it is. The base-model-comparison (`basecmp`) chain queued on 25 Sep
was cancelled on 26 Sep: its protocol differs from v3 on target, strength and
arm set, so its cells could not be pooled with v3's anyway.

---

## 1. What v3 is, in one table

| setting | v3 | v2, for contrast |
|---|---|---|
| target | **q50** (the median) for every property | q90 |
| strength | **w = 1 for every arm** — not normalised, see §2.2 | per-arm, frozen under a chemistry floor |
| guidance window | **t ≥ 0.5** for every arm | t ≥ 0.5 |
| seeds | 20261001 / 20261002 / 20261003, **for both stages** | the same three |
| n | **not pre-registered — the operator picks it** (§6.1) | 5000, batch 128 |
| batch | **500**, measured; must divide n | 128 |
| chemistry floor | **none — nothing is excluded** | 0.9 × unguided, a hard gate |
| headline arms | unguided, plug, tmpd, lgd_mc, tfg, **bdg_e4t0.5, bdg_e4t1** — 7 | + btvg, btvg_var; no BDG |
| base models | **three**, each with its own property pair (§1.2) | ours only |
| δ | 2 × f_B's calibration MAE — **and f_B now differs by backend** | the same |

**Arms.** The full comparison set with `btvg` **dropped** and **BDG** added as
the innovation target. BDG's headline ladder is η = 4 with **τ_mult ∈ {0.5,
1.0}**; §4 extends it over both knobs.

**`τ_mult` is BDG's setpoint knob**, τ = τ_mult × `f_A.y_std`. It is *not* the
`t ≥ 0.5` guidance window, which did not change. The two are easy to confuse
because both get called "tau" in conversation; everywhere in this document
τ_mult is the setpoint and `t` is time.

### 1.1 What changed on 26 September, and what it costs

| | was | now | what it costs |
|---|---|---|---|
| headline τ_mult | {0.5, 0.75, 1.0} | **{0.5, 1.0}** | 7 arms, not 8. No `w_eff` span lost — 0.75 was the middle rung, and the measured span 1019/253 = 4.03× is set by the endpoints — but the headline ladder is now **two points**, which can show a difference and not a shape (§4.1) |
| ablation seeds | one | **three, the same three** | every v3 row now carries an across-seed spread, and an ablation row is comparable to a headline row again |
| n | 5000, then 2000 | **unset** | the operator chooses against §6.1's power table. Nothing in the protocol depends on its value; the run's **power** does |
| tree | `…/n<N>/seed<S>/` | `…/<stage>/n<N>/seed<S>/` | n and the seed set used to be the only things telling the two stages apart, which is why the ablation was briefly run smaller. The stage directory does that job now |
| property pair | TFG's, for every backend | **per backend** (§1.2) | the big one — see §1.2 |
| base models | two | **three** | + QM9 diffusion, unguided and plug only |

### 1.2 Which base model is scored with which property pair

Declared once, in `transfer_sweep.V3_BACKENDS`. Henry, 26 Sep.

| backend | generator | f_A / f_B | arms |
|---|---|---|---|
| `fm` | our flow-matching EGNN | **ours** — `weights/f_{A,B}_<p>.pt` | all 7 |
| `equifm` | EquiFM (Song et al.) | **TFG's** — `tf_predict_<p>` / `evaluate_<p>` | all 7 |
| `edm` | **QM9 diffusion** (TFG's released EDMsecond) | **ours** | **`unguided`, `plug` only** |

**Modality 2 is not in this table.** Its state is a `[B, L, 4]` simplex rather
than coords + feats + mask over an EGNN, its properties are analytic rather
than learned networks, and it runs through `proj1/m2/`. What bridging it would
take is written down in [MODALITY2_V3_PLAN.md](MODALITY2_V3_PLAN.md); until
that is done it is a separate run and its numbers do not belong in a v3 table.

> ### ⚠️ The pair sets the band, so in-band does not cross backends
>
> δ = k × MAE(f_B). The pair therefore sets the **width of the acceptance
> band**, and the two oracles do not have the same accuracy. **Measured** on
> the same 3000 calibration molecules, through the same code path both pairs
> use ([V3_PAIR_DELTA.md](../results/V3_PAIR_DELTA.md)):
>
> | property | δ ours | δ TFG | ours / TFG | |
> |---|---|---|---|---|
> | mu | 0.175407 | 0.156754 | **1.12×** | ours **wider** |
> | alpha | 0.506184 | 0.164797 | **3.07×** | ours **wider** |
> | gap | 0.007480 | 0.003736 | **2.00×** | ours **wider** |
>
> **Our band is wider on all three**, so `fm` and `edm` accept molecules
> `equifm` would reject, before any base model or arm is considered.
>
> ⚠️ Each of our nets is scored on the half of the calibration pool it did
> NOT train on — f_A (trained on `train_a`) on the `train_b` half, and vice
> versa. Scoring them on the whole pool, which is right for TFG's external
> pair, measures each partly on its own training data and flatters it by
> ~19 %. A draft of this section did that and reported our mu band as
> *narrower*; **withdrawn**. The held-out figures above land close to
> `2 × val_mae` from the checkpoints, which is the independent check.
>
> **Never compare `in_band` between `fm` and `equifm`.** It was already true
> that nothing is paired across backends (§3); this is stronger — the two
> columns do not even measure the same event.
>
> **Within one backend, every arm shares one δ**, so the arm-against-arm
> comparison — the question this project actually asks — is untouched. That is
> why the change is acceptable.
>
> What it buys: our pair is disjoint **by construction** (f_A and f_B trained
> on disjoint halves), where TFG's is disjoint only by inference and its
> `evaluate_<p>` saw an unknown part of QM9, making its δ optimistically
> tight. Every cell records `pair`, `guide` and `oracle`, and `v3_table.py`
> refuses to pool cells whose `pair` disagrees.

### 1.3 Amendment, 28 September 2026 — a fourth backend, `vp`

**This section is an amendment, not a revision.** Section 1.2 above is the
26 September pre-registration and is left exactly as it was written. Nothing
here changes any arm, target, strength, seed or property of the three
backends declared there, and no cell already on disk is affected.

**What is added.** A fourth backend, `vp`: **our own** QM9 VP diffusion
model, trained here. `edm` and `vp` are both diffusion and **only `vp` is
ours** — `edm` is TFG's released EDMsecond, borrowed and frozen.

| backend | generator | f_A / f_B | arms |
|---|---|---|---|
| `vp` | **QM9 diffusion, OURS** — `diff.pt` (the selected checkpoint) | **ours** — `weights/f_{A,B}_<p>.pt` | **`unguided`, `plug` only** |

**Two arms only (Henry, 28 Sep).** `vp` mirrors `edm` exactly, which is what
makes `vp`-vs-EDMsecond a legal head-to-head. The 7-arm set answers a
different question — *does guidance work on VP?* — at 3.4× the cost
(63 cells / ≈5 GPU-h against 18 / ≈1.5), and it still could not be compared
to `edm`, which has no BDG arm. **`vp` sits out the ablation**, as `edm` does.

**Why it is worth running.** `vp` and `fm` share the EGNNVelocity backbone,
3,753,229 parameters, 1500 epochs, batch 256, EMA 0.9999, the `train_a`
split and seed 20260918. `fm`-vs-`vp` therefore isolates the generator
**family** and nothing else — the matched cell `tab:fmvd` had never had until
28 Sep, and the one `main.tex:175` confessed was missing. `edm` cannot stand in
for it at any n: it differs in data, architecture and budget at once.

**The band.** `vp` scores with **our** pair, so it joins `fm` and `edm` on
the wider band of the box above and is comparable to both. It is **not**
comparable to `equifm` on `in_band`, for the same reason they are not.

**Grid.** 2 arms × 3 properties × 3 seeds × w = 1 = **18 cells**, n = 2000 in
4 batches of 500, q50, no chemistry floor — every other axis identical to
§1.2's.

**Cost ≈ 1.5 GPU-h, and it is an extrapolation, not a measurement.** It is
`edm`'s measured per-molecule rates on a B200 MIG 2g.45gb (unguided 0.0788 s,
plug 0.2130 s, from the 216 cells in `results/transfer/`) applied to 18,000
molecules per arm: 5,252 s. Our VP is *smaller* than EDMsecond (3,753,229
against 5.34 M parameters), so it should come in under. The longest single
task is ≈10 min, against a 4-hour wall.

**It therefore fits inside one window, and runs as nine standalone `sbatch`
tasks — not through `submit_v3.sh`.** `vp` is registered in `V3_BACKENDS` but
deliberately **not** in `V3_CHAIN_BACKENDS` or the job file's `BACKENDS`: the
chain's preflight loops every listed backend and is its one `afterok` link, so
listing a backend whose checkpoint is absent would stop `fm`, `equifm` and
`edm` from running at all. **This is a standing design decision, not a waiting
state**: `vp` runs as its own 9-task array via `submit_vp_bench.sh`, and the
chain's headline array stays at 27 tasks (0-26).

**Status, 28 September: DONE.** Trained by Bobo to epoch 1500, selected epoch
1475, published as `weights/diff_ema.pt` (md5 `8a3390a6`). 18 cells, 0.81 GPU-h
on a B200 MIG 2g.45gb, 0 non-finite, `v3_sanity` clean. **`fm` beats `vp`** on
molecule stability (0.3970 vs 0.2883), validity (0.7562 vs 0.6617) and atom
stability (0.9356 vs 0.9070), every gap 7-13x the seed sd -- the evidence the
base-model choice previously lacked. Results:
[V3_RESULTS_vp.md](../results/V3_RESULTS_vp.md).

⚠️ One earlier attempt produced a **full set of 18 wrong cells**: the resolver
picked up the epoch-75 remains of the crashed job 8590174 (md5 `93ab4f72`),
which has the same family, split, width and depth, so every field check passed
while atom stability came out at 0.763 instead of 0.905. `vp_bench.slurm` now
pins the checkpoint md5. A field check is not an identity check.


## 2. The three choices that shape how v3 must be read

### 2.1 No chemistry floor, so no arm is excluded — and no in-band ranking

v2 gated on molecule stability ≥ 0.9 × unguided. v3 removes the gate
(Henry, 26 Sep). Nothing is disqualified; chemistry is **reported** instead.

That has a consequence which must travel with every v3 number. `w = 1` is far
above some arms' best strength: under v2, `tfg` froze at **0.01–0.05**, so
w = 1 is 20–100× it. Measured on our base at q50, mu, n = 512:

| arm at w = 1 | in_band | mol_stab | validity |
|---|---|---|---|
| unguided | 0.072266 | 0.402344 | 0.785156 |
| plug | 0.083984 | 0.373047 | 0.730469 |
| tmpd | 0.095703 | 0.390625 | 0.746094 |
| lgd_mc | 0.085938 | 0.376953 | 0.765625 |
| **tfg** | **0.382812** | **0.228516** | **0.619141** |

(On gap, tfg reaches 0.316406 at validity 0.466797.) These are **indicative,
not the run**: single-seed at n = 512, where v3 is the operator's n × 3 seeds.

**Sourced from the cells, not from a docstring.** Every row is read from
`results/sweep/mu__<arm>__q50__w1__tmin0.5__cmp.json` (and the `gap` cell for
the last line), which is where this project actually records them.

> **Two corrections, in opposite directions, both recorded here so the next
> reader does not undo them again.**
>
> An early draft gave unguided validity as **0.75**, quoting a docstring. A
> later draft corrected it to **0.78516** but cited "the constant at
> `transfer_sweep.py:262`", which does not exist — so an anti-fabrication
> pass withdrew *the whole correction* and reinstated 0.75.
>
> **The correction was right; only its citation was invented.** The cell says
> `validity = 0.78515625` (402 of 512). 0.75 is withdrawn, 0.78516 restored,
> and the table now cites the cells. Two further claims from that pass are
> also withdrawn: the docstrings are **not** "the only place this project
> records them" (the cells hold all of it), and they do **not** disagree about
> n — both say 512, as do the cells.

v3's own unguided cells supersede all of it in three properties × three seeds
at the run's own n — that is what any published number must come from.

So **an in-band leaderboard would crown the arm that destroyed the most
chemistry.** `v3_table.py` therefore refuses to rank on in-band alone: every
in-band figure prints beside stability, validity, uniqueness and
distinct-valid-per-attempt, the summary line attaches the chemistry its best
arm spent, and a second line names the best arm that did **not** lose chemistry
against unguided. No verdict is computed; that is deliberate.

### 2.2 w = 1 is equal *nominal* strength, not equal force

Equal `w` is not equal push. The project measured the applied correction share —
`mean |C(G)| / |V|` over guided steps, after the score-to-velocity factor and the
clip — at w = 1 on mu ([FULL_RUN_RESULTS.md §7](../results/FULL_RUN_RESULTS.md)):

| arm | plug | tmpd | lgd_mc | btvg (dropped in v3) |
|---|---|---|---|---|
| share at w = 1 | 0.055 | 0.024 | 0.138 | 0.244 |

**Across all arms measured there the span is 10.0×, and across the three v3 arms
it covers it is 5.8×** (lgd_mc 0.138 / tmpd 0.024). `tfg` is absent from that
table on purpose — it replaces the sampler rather than adding a correction, so it
has no share to measure, and its force at w = 1 is **unknown**, not small.

So v3 answers "what does each method do when handed the same dial setting", which
is a legitimate and simple question, but it is **not** "which method is best" and
it is **not** each arm at its own optimum. Any sentence quoting a v3 number
carries "at w = 1".

### 2.3 δ is the pre-registered rule, because v3 is a q50 run

δ = 2 × f_B's **calibration** MAE. Target-independent, so it is the same at q50 as
at q90.

**But f_B is no longer the same network on every backend** (§1.2), so δ is no
longer the same number on every backend either. The *rule* is pre-registered
and identical; the *value* it produces depends on which oracle it is measured
on, and our oracle is the less accurate one. That is the whole content of the
warning in §1.2 and it is why `pair` travels with every cell.

The calibration set is **3000 molecules drawn from `train_a` + `train_b`**
(`calibration_indices` in `transfer_sweep.py`), which documents why: *not* `val`,
because `val` supplies the q50/q90 molecule sizes, and *not* `test`, because `test`
supplies the `dist` sizes and targets. An earlier draft of this section said "f_B's
validation MAE over all `val` molecules" — wrong on both the split and the count.
It also pre-registered the values 0.16799 / 0.48135 / 0.00760 as constants; **no
gate asserts them**, so they are not repeated here as a commitment. Each cell
records its own δ, and `v3_table.py` refuses if cells disagree.

The 25 Sep **local** δ is *not* used here. It was measured in q90's
neighbourhood, and v3's target is q50, so that window is the wrong one. The
basecmp stages keep the local rule; v3's stages default to the global rule and
every cell records `delta_mode`.

**This does not touch BDG's knob.** BDG's setpoint is τ = τ_mult × `f_A.y_std`,
a fraction of the guide's own property scale, so δ never enters BDG's sampling —
only its scoring. That is why the δ question is settled by one line here rather
than by a re-measurement.

### 2.4 Several controllers per cell, not one — a measured deviation

For every other arm the batch is a performance knob. For BDG it is **the
estimator**: `V_b` is the variance of the guide's predictions over whatever tensor
the sampler is handed, so the batch size *is* the controller's sample size.
`guidance.py` states the requirement — "The sweep must use one batch per cell."

**v3 does not meet it at any n worth running.** Measured with
[`batch_memprobe.py`](../../proj1/scripts/batch_memprobe.py) over batches
32/64/128, peak allocation is linear in the batch and flat in the step count,
and both backends reproduced exactly
([V3_BATCH_MEMORY.md](../results/V3_BATCH_MEMORY.md)):

| backend | GiB reserved / molecule | one batch of 1000 | of 2000 | of 5000 |
|---|---|---|---|---|
| ours (`fm`) | 0.0401 | 40 GiB | 80 GiB | 201 GiB |
| EquiFM | 0.0560 (1.85× ours) | **56 GiB** | 112 GiB | 280 GiB |

Against a `b200-mig45` slice, which is 45 GB = **41.9 GiB**, with an 80 %
working budget of **33.5 GiB**. So one batch of n does not fit for any n above
about 600 on the tighter backend, whatever n the operator picks.

- **`--batch 500`** is the setting, and it is **independent of n**: it is the
  largest round batch that fits EquiFM inside the budget (≈ 28 GiB
  extrapolated; 625 needs 35.0 and 1000 needs 56). Choosing a different n does
  not change it — it only changes how many controllers a cell runs.
- A cell therefore runs **n ÷ 500 independent controllers**, each estimating
  `V_b` from 500 samples: a **6.3 %** standard error on the variance,
  regardless of n. The cell's reported controller state (`e`, `V_b/τ²`,
  `w_eff`) is the **mean over them**, so the diagnostics get *less* noisy as n
  grows while each controller stays equally good. **n must be divisible by
  500**; the job, the submit script and `transfer_sweep` all refuse otherwise.
- Every arm uses the same batch, so arms stay paired on the same initial noise.

Two settings are recorded so they are not re-proposed. The default batch 128
leaves a **ragged last controller** whenever 128 does not divide n — at n =
5000 that was 40 controllers with the last over **8 molecules**, a 53 %
standard error, pooled in as an equal. And `--batch = n` would OOM every task
on its first cell. `proj1/tests/test_v3.py` ties the slurm's batch to the
probe's recorded recommendation, so changing one without re-measuring fails a
gate.

> **These figures are a laptop extrapolation, not a cluster measurement.**
> They come from an RTX 5080 at batches ≤ 128, fitted and extended. An earlier
> draft of this section reported a "Confirmed on Betty, 26 Sep" table —
> `fm` 17.57 GiB and EquiFM 26.65 GiB reserved, attributed to a specific job
> id — and **that measurement does not exist in this repository**:
> `results/v3_batch_memory.json` has an empty `confirm` block and neither
> `results/v3_batch_memory_betty.json` nor `V3_BATCH_MEMORY_BETTY.md` was ever
> written. The numbers were the laptop slopes × 500 presented as observations.
> They are withdrawn.
>
> The check itself is real and still runs: the chain's `preflight` link calls
> `batch_memprobe.py --confirm 500` on the actual slice before the array
> starts, and writes those two files. **Read them before trusting the table
> above** — and if the operator picks a large n, confirm on their own card,
> because nothing here was measured on it.

---

## 3. The headline run

**Cells**, from `V3_BACKENDS`:

| backend | arms | cells |
|---|---|---|
| `fm` | 7 | 7 × 3 × 3 = **63** |
| `equifm` | 7 | **63** |
| `edm` (QM9 diffusion, TFG's **borrowed** EDMsecond) | 2 | 2 × 3 × 3 = **18** |
| `vp` (QM9 diffusion, **OURS**) | 2 | 2 × 3 × 3 = **18** |
| | | **162 headline cells** |

`vp` is not part of the chain's array (§1.3), so the chain still submits 27
tasks for `fm`, `equifm` and `edm`; `vp`'s 18 cells come from its own 9-task
array.

The ablation adds 17 arms × **2 strengths** × 3 × 3 × 2 = **612** (`edm` sits
it out, and so does `vp`), for **774** in all — it is much the larger stage, roughly 3× the
headline's cost. The array is a uniform backend × property × seed grid — 27 tasks per
stage — because a ragged one is how stride bugs happen; `edm`'s nine ablation
tasks exit 0 immediately having planned nothing, and say so.

| arm | what it is |
|---|---|
| `unguided` | the reference. One cell per property per seed; no strength, no window |
| `plug` | DPS-style plug-in (Chung et al. 2023) — also BDG at η = 0 |
| `tmpd` | TMPD / ΠGDM uncertainty denominator (Boys et al. 2024) |
| `lgd_mc` | loss-guided diffusion (Song et al. 2023) — v1's best arm. **Not v2's**: under v2 it was floor-limited to −1.1/−1.6 σ and tied |
| `tfg` | TFG, the full update (Ye et al. 2024) |
| `bdg_e4t0.5` | BDG, η = 4, τ_mult = 0.5 — asks for half the natural spread |
| `bdg_e4t1` | η = 4, τ_mult = 1.0 — asks for the natural spread |

**τ_mult = 0.75 is not here.** Dropped from the headline on 26 Sep; it runs in
the ablation (§4), which runs the same n and the same three seeds. Dropping the
*middle* rung costs no `w_eff` span — the measured endpoints 1019 and 253 are
both kept — but it leaves the headline with **two points**, which is enough for
a difference and not enough for a shape. See §4.1.

**Every metric is recorded per cell**, as `evaluate_samples` already produces:
in-band (continuous and decoded), MAE and RMSE, bias and residual sd, atom and
molecule stability, RDKit validity, uniqueness of valid,
distinct-valid-per-attempt, embedding diversity, the guide–evaluator gap, the
non-finite count, and the measured cost counters. BDG cells also record the
controller's state: raw pre-clamp `e`, `V_b/τ²`, `w_eff`, and the batch RMS of
the dispersion term.

**Pairing.** Within one base model every arm sees the same seed, the same batch
size, the same molecule sizes, the same fixed target **and the same δ**, so
arms are paired and a paired test is valid.

**Across base models nothing is paired** — different architectures, different
noise schedules, different state scales — so cross-backend rows are printed
side by side and their differences are not tested. And where the *pair* also
differs (§1.2), the two columns do not measure the same event at all, because
δ differs: `fm` and `edm` are scored in our band, `equifm` in TFG's narrower
one. **A cross-backend `in_band` difference is not a result.**

---

## 4. The ablation

> **The ablation has its own protocol: [ABLATION_V3_PROTOCOL.md](ABLATION_V3_PROTOCOL.md).**
> As of 26 Sep (final) it runs the **same n and the same three seeds** as the
> headline, so an ablation row and a headline row are directly comparable. It
> carries all **17** grid arms and **no comparison arm**, so the grid is also
> readable on its own. `edm` sits it out.
>
> It is the **larger** of the two stages — 17 arms against 7, on two backends
> — so `submit_v3.sh` defaults to the headline and leaves this for a second
> window once the `[timing]` lines have settled the real per-pass rate.
>
> The paragraph below described an earlier plan and is kept to show what
> changed. Two figures from that period are withdrawn: the ablation briefly
> ran **one seed at a smaller n**, purely because n was then the only thing
> separating the two stages' trees — the stage directory does that now — and
> the **"51.3 GPU-h"** some drafts contrasted it against was never the
> headline, but the ablation's *own* first plan (14 grid arms in a shared
> n5000 tree).

~~The same protocol, sweeping BDG's two knobs, **at the same n and the same three
seeds** so an ablation row and a headline row are directly comparable.~~

| axis | values |
|---|---|
| η (gain) | 0, 1, 2, 4, 8 |
| τ_mult (setpoint) | 0.5, 0.75, 1.0, 1.5 |

**η = 0 is planned once, not four times.** At η = 0 the dispersion term is
multiplied by zero, so τ_mult cannot change the samples; four τ_mults would be
one computation under four names, and pooling them would look like four
independent measurements of one thing. It is planned at τ_mult = 1.0 and is
BDG's **identity gate**.

**That gate is field-level, and only field-level.** `test_bdg.py` A9 and B-a
prove the guidance field at η = 0 is **bit-identical** to `plug` — max err
0.00e+00, on a real `x_t` on CUDA, not just on synthetic tensors. A **cell-level**
identity is *not* asserted, because it would be testing the GPU rather than the
method ([V3_REPRO_ENVELOPE.md](../results/V3_REPRO_ENVELOPE.md)):

> Run `plug` three times at one seed, changing nothing. **16 of 43 numeric
> fields move** — and the ones that do are all *continuous*: `f_A_mean`,
> `f_B_mean`, the MAE/RMSE pair, the diversity measures, the guide–evaluator gap,
> `atom_stability` (5.3e-3) and `mol_stability` (0.015625 = 1 molecule in 64).
>
> **The counting metrics did not move at all.** `in_band_fraction` = 0,
> `in_band_fraction_dec` = 0, `validity` = 0, `uniqueness_of_valid` = 0,
> `clipped_sample_steps` = 0, `n_nonfinite` = 0, over 3 runs at n = 64.

The seed fixes the initial noise; it does not fix the trajectory. The EGNN's
scatter kernels use atomics, and the velocity-relative clip is a **threshold** —
once it fires on a different step, that molecule differs from then on. Measured
against a 3-run envelope, `bdg_e0t1` sits **inside** it on every field that
carries a claim; the three that sit outside are `delta` and `mae_B` (both ~1e-7
relative, and δ is a fixed constant of `f_B`, not a sample statistic) and
`guide_eval_gap_max`, a maximum over 64 molecules. With only two runs the envelope
was too tight and `f_B_mean` looked 700× outside it; that was the envelope, not
the method.

**Consequence for every v3 comparison.** Pairing is unharmed — arms within a cell
are drawn from the same generator seed, so they share identical *initial noise*.
And the measurement above is **reassuring for the headline metric specifically**:
`in_band_fraction`, the number v3 turns on, was bit-stable over 3 identical runs,
as were validity and uniqueness. So no run-to-run correction is applied to them,
and **significance is the binomial standard error alone** (`v3_table.py`), which is
what the tables report.

An earlier draft of this section claimed instead that "validity moved by 1.6
points at n = 64" and pre-registered a tie rule discarding differences below that
scale. **Both are withdrawn: the figure is not in the record** (validity's measured
envelope is 0), and no such rule is implemented anywhere. What the record *does*
show is that the **continuous** fields move at the third decimal, so a bias, MAE or
diversity difference at that scale is noise — while a counting metric's difference
is not attributable to nondeterminism at all. Larger n only tightens this.

So the ablation runs 1 + 4 × 4 = **17 arms**, of which 2 (η = 4 at τ_mult 0.5
and 1.0) are also the headline's. **The two stages do not share a tree** —
`…/v3/` and `…/v3abl/` are separate directories — so those two arms are
computed twice, once per stage, at the same n and the same seeds. That
duplication is deliberate: it is what makes each stage's tree readable on its
own, and it also gives a free consistency check, since the two copies should
agree within seed noise.

Union of the two arm sets: **22 arms.** Cells: **162** headline + **612**
ablation = **774**. The ablation's factor of two over its arm count is the
strength sweep, w ∈ {1, 4} — the headline has no strength axis
([ABLATION_V3_PROTOCOL.md §1.1](ABLATION_V3_PROTOCOL.md)).

### 4.1 What the two knobs actually do — **measured on real trajectories**

BDG's coefficient acts on the mean/deviation split
`num_i = w·[(y − F̄) − w_eff·(F_i − F̄)]`, with `w_eff = 1 + η·e` and
`e = (V_b − τ²)/τ²`. So `w_eff` weights **only the deviation term**; the mean term
is untouched.

**An earlier draft of this section predicted `w_eff` from a closed form and was
wrong by two orders of magnitude.** It read `V_b ≈ y_std²` off `test_bdg.py`'s
part B — which evaluates the field on one *synthetic* `x_t` — and concluded
`w_eff = 1 + η(1/τ_mult² − 1)` = 13.00 / 4.11 / 1.00 / −1.22. During actual
sampling `V_b/τ²` run-means in the **tens to hundreds**, not ~1. Measured by
[`bdg_ladder_probe.py`](../../proj1/scripts/bdg_ladder_probe.py) at this run's
own configuration — 100 steps, guidance on for t ≥ 0.5, one controller
([BDG_LADDER_MEASURED.md](../results/BDG_LADDER_MEASURED.md)):

| arm | V_b/τ² | `w_eff` mean | `w_eff` RMS | frac steps `w_eff` < 0 | closed form said | clipped steps |
|---|---|---|---|---|---|---|
| `plug` | – | – | – | – | – | 884 |
| `bdg_e0t1` | 67.5 | **1.0** | 1.0 | 0.00 | 1.00 | 894 |
| `bdg_e4t0.5` | 256.6 | **1019** | 4815 | 0.00 | 13.00 | **2516** |
| `bdg_e4t0.75` | 114.1 | 456 | 2144 | 0.08 | 4.11 | 2270 |
| `bdg_e4t1` | 64.0 | **253** | 1204 | 0.16 | **1.00** | 1853 |
| `bdg_e4t1.5` | 28.6 | 111 | 534 | **0.62** | −1.22 | 1752 |

Five things follow, and they govern how the ablation may be read:

1. **`bdg_e4t1` is NOT plug.** `w_eff` = 253, not 1.00. The identity would need
   `V_b = y_std²`; it is 64× that. The earlier draft called this arm "a second
   identity check" — it is an ordinary rung and must be treated as one.
2. **`bdg_e0t1` *is* plug**, and for a reason that does not depend on `V_b`: at
   η = 0 the dispersion term is multiplied by zero, so `w_eff` ≡ 1 whatever the
   controller measures. Measured `w_eff` = 1.0, 0 % negative steps, 894 clipped
   steps against plug's 884. This is the identity gate, and it is field-level
   (§2.4's envelope applies to the cell-level metrics).
3. **The rung ORDER survives; the magnitudes do not.** `w_eff` falls monotonically
   with τ_mult (1019 → 456 → 253 → 111), so ordering rows by `w_eff` is sound —
   but every quoted number must be the **measured** one, never the closed form.
   **The headline now holds only the endpoints** (τ_mult 0.5 → `w_eff` 1019 and
   1.0 → 253, a **4.03×** span). Dropping 0.75 therefore costs the headline
   *no span*, but a two-point ladder can exhibit only a difference — **not
   monotonicity and not curvature.** Those must be read off the ablation's four
   points, which keep 0.75 and 1.5.
4. **The sign flip is real and ordered.** The share of guided steps on which the
   deviation term *reverses* — spreading the batch instead of concentrating it —
   runs 0.00 → 0.08 → 0.16 → **0.62**. At τ_mult 1.5 nearly two thirds of guided
   steps push the wrong way. A signed run-mean alone hides this, which is why the
   cells now record `bdg_w_eff_sq` (→ RMS) and `bdg_w_eff_neg` (→ this fraction)
   as well. RMS exceeds the signed mean by ~4.7× on every BDG rung.
5. **The clip is a live confound, possibly the dominant one.** `plug` clips on 884
   sample steps; `bdg_e4t0.5` on **2516**. At `w_eff` of order 10³ the dispersion
   term is far outside the velocity-relative trust region, so what reaches the
   sample is the clip's truncation of it, not the controller's request. **A rung
   can be clip-limited rather than controller-limited, and nothing in v3
   distinguishes the two.** The clip is known to be load-bearing (removing it
   produced 222 non-finite samples against 0), so it cannot simply be widened.

**On the equilibrium.** `V* = τ²(1 − 1/η)` is where `w_eff` = 0, verified
analytically to 5 decimals for η ∈ {1.5, 2, 4, 8}. **It is the law's asymptote and
the loop does not reach it inside the window** — [BDG_REVIEW.md](../methods/BDG_REVIEW.md)
measured run-mean `V_b/τ²` of 1.27–1.54 at τ_mult 0.5 against a setpoint of 0.75,
with `w_eff` at 0 ± 0.19 near the setpoint and changing sign up to 34 times in 50
steps. **The paper must never say the controller "holds its setpoint", and this
protocol must not describe `w_eff` as a schedule that decays to 0.** An earlier
draft did both.

**What the ablation can settle.** Whether the deviation weight has any effect on
coverage at q50; whether that effect is monotone in **measured** `w_eff`; and
whether the η = 0 identity holds.

**What it cannot:** anything about a chemistry-matched comparison, because every
cell is at w = 1; and it cannot separate "controlling spread helped" from "a larger
deviation weight helped", because at q50 those are the same dial (§6).

---

## 5. Budget, and why this is a GPU run

**The budget is a function of n, and n is not pre-registered**, so it is not a
table in this document — it is generated:

```
python proj1/scripts/v3_power.py --md-out docs/results/V3_POWER.md
```

[V3_POWER.md](../results/V3_POWER.md) prints, for a range of n, the standard
error on `in_band`, the minimum difference the run could resolve, and the
GPU-hours for each stage. **Choose n from the minimum-detectable-difference
column, not from the GPU-hour column** — §6.1 says why.

The cost model, so the generated numbers can be argued with:

- **One measured rate**: `v2_run.slurm`'s 58 min for a 23.5-unit task at
  n = 5000, on our base, i.e. **2.468 min/unit**. Everything else is derived.
- Units are **defined at n = 5000** and assumed **linear in n**. Per
  (property, seed, backend): **23.5** for the five comparison arms as a bundle,
  plus **3.5** per BDG arm. A backend running only some comparison arms — the
  QM9 diffusion base runs two of five — is charged pro rata by arm count,
  which **overstates**, since `unguided` does no guidance at all and `plug` is
  the cheapest guided arm.
- `fm` is charged at the measured rate; **every other backend at 1.83×**,
  which is **borrowed** from the repo's one timed external backend
  (TFG/EDMsecond) and is *not* measured for EquiFM. Its memory slope is 1.85×
  ours, which is consistent but is not a timing.

⚠️ **Treat the generated GPU-hours as a floor.** Per-task **fixed** cost —
checkpoint load, `calibration_indices` + `build_pair` over 3000 molecules, the
second oracle's calibration — does not scale with n, and at small n it is a
visible fraction of a task. What actually protects the wall is `--max-minutes`
plus the `[timing]` line every task prints, so the true rate is visible in the
log before the bulk of the array lands.

**Wall time depends on concurrency.** With 27 tasks per stage, the whole run
fits a 24-hour window at modest concurrency for any n in the low thousands;
check your own limit first with
`sacctmgr show assoc user=$USER format=User,QOS,MaxJobs,GrpTRES%30`.

`submit_v3.sh` defaults to `headline` rather than `all` so that the headline's
`[timing]` lines settle the real per-pass ratio — the one assumed factor in
the whole model — before the ablation, which is the larger of the two,
commits.

**GPU, not CPU.** Measured on one identical cell (mu, plug, n = 128, batch 128,
100 steps) on the laptop: **0.1 min on a GPU against 2.6 min on CPU**, a 26× gap.
A v3 guided cell is 1400 batched passes — 4 batches × a **recorded** 350 — each
carrying 500 molecules. The BDG author's CPU choice was right for their cells
(n = 512 in one batch, 350 passes, tensors too small to use a GPU, parallelism
from running many cheap cells at once) and does not transfer here.

**The two bases run the same number of passes.** Read from the cells' own cost
counters at n = 64, 100 steps, `bdg_e4t0.5`: **both** backends record nfe = 350
(gen_fwd 200, gen_vjp 50, guide_fwd 50, guide_bwd 50). So the bases are matched
in function evaluations and EquiFM's extra cost is **per pass** only — it is the
bigger net (nf = 256, 9 layers).

**That per-pass ratio is NOT measured, and the 1.83× above is borrowed.** The
laptop was contended by another job while this was being checked: the same fm cell
timed 17.9 s and then 12.2 s, a 46 % swing, so no wall-clock number from it is
usable and none is quoted. 1.83× is the repo's one timed external backend
(TFG/EDMsecond); EquiFM's *memory* slope is 1.85× ours, which is consistent but is
not a timing. The task split is sized for **2.5×** as margin, and what actually
protects the wall is `--max-minutes` plus the `[timing]` line every task prints —
so the true ratio is visible in the log before the bulk of the array lands.
`V3_N` on the submit line lowers n further if it turns out to be worse than
2.5× — though at 75 min against a 225 min budget there is far more headroom than
there was at n = 5000.

---

## 6. What v3 can and cannot settle

### 6.1 Choosing n is choosing the run's power — read this before picking

**v3 pre-registers no n** (Henry, 26 Sep). That makes n the operator's
decision and this section the thing to make it against. It is written *before*
the run so that a null is read as "this run could not resolve it" where that
is true, and not as "the method does nothing".

The numbers live in [V3_POWER.md](../results/V3_POWER.md), regenerated by
`proj1/scripts/v3_power.py`. The reasoning behind them:

- `in_band_fraction` is a **proportion over n samples**, so its standard error
  is the binomial √(p(1−p)/n). §4 pre-registers "significance is the binomial
  standard error alone" — no run-to-run correction — because the counting
  metrics were measured **bit-stable** over three identical runs
  ([V3_REPRO_ENVELOPE.md](../results/V3_REPRO_ENVELOPE.md)).
- Three seeds pool to **3n** per (arm, property). That is the number every
  published figure rests on.
- A **contrast** between two arms is what carries a claim, and it is reported
  **unpaired**: se = √(se_a² + se_b²). Arms within a cell do share initial
  noise, so a paired test would be tighter — roughly by √2 — which makes every
  figure here **conservative**. It can miss a real difference; it cannot
  manufacture one.
- **Multiplicity**: 6 guided arms × 3 properties = 18 contrasts against
  `unguided`, so the threshold is Bonferroni at α/18, two-sided — **z ≈ 2.99**.
  The ablation's grid is a **max-over-17 selection** and needs a wider one.

**The shape of the answer**, at the indicative p ≈ 0.09 most arms sat at
under w = 1:

| n per cell | pooled se | min. detectable difference |
|---|---|---|
| 500 | 0.74 pp | 3.13 pp |
| 1000 | 0.52 pp | 2.21 pp |
| 2000 | 0.37 pp | 1.56 pp |
| 5000 | 0.23 pp | 0.99 pp |
| 10000 | 0.17 pp | 0.70 pp |

**What to weigh it against.** [BDG_REVIEW.md](../methods/BDG_REVIEW.md) found
**no** floor-clearing BDG cell beating `plug` over 72 paired tests, and §6.2
pre-registers a null as the likely outcome. A null is evidence against BDG
only if the run could have seen the effect. A BDG-vs-plug difference of about
**1 pp** is the scale the earlier work suggests; resolving that under a
multiplicity-corrected threshold needs n in the **low thousands per cell**.
Below roughly n = 1000 the run cannot resolve anything smaller than ~2 pp, and
a null there says almost nothing.

**So: pick n so the minimum detectable difference sits below the difference
you would care about. If that is not affordable, say so in the write-up** —
report the MDD beside the null rather than reporting "no difference" flat.

Two further consequences of a small n, neither of which the table shows:

- BDG's **controller diagnostics** (`e`, `V_b/τ²`, `w_eff`, `bdg_w_eff_sq`,
  `bdg_w_eff_neg`) are means over n ÷ 500 controllers, so a small n makes them
  noisier — though each individual controller is equally good (§2.4).
- The **ablation** carries a 17-way selection on top, so its own resolvable
  difference is larger than the table's at the same n.

What is *not* affected: **pairing**. Arms within a cell share initial noise at
any n, so the paired structure — and §4's finding that counting metrics were
bit-stable over three identical runs — is untouched. The loss is sample size
only.

### 6.2 The rest

**Can.** What each method does at one dial setting, on the median target, on two
different base models, with every metric reported and nothing excluded; whether
a method's behaviour survives a change of base model when nothing else changes;
and whether BDG's spread knob moves coverage at the target where its own premise
holds.

**Cannot, and must not be claimed:**

- **Which method is best.** w = 1 is not any arm's optimum and equal w is not
  equal force (§2.2).
- **That the best in-band arm is the best method.** At w = 1, in-band and
  chemistry trade off, and the arm furthest above its own best strength will
  look strongest on in-band alone (§2.1).
- **Anything at q90.** v3 is a q50 run. The project has measured that the two
  targets invert which lever matters: at q90 the batch sits 10–20 δ from target
  and bias dominates, while at q50 spread does.
- **A cross-base-model statistical difference.** Not paired (§3).
- **That BDG's spread control is what did anything — FROM THE HEADLINE ALONE.**
  v3 has no variance-only rung: `btvg_var` went with `btvg` (§1). Within the
  headline, η = 0 is exactly `plug`, so `bdg − plug` bounds the whole
  dispersion term's effect, but nothing there separates "controlling spread
  helped" from "pushing harder helped" — **at one strength those are the same
  dial**, because the target sits near the batch mean, so the deviation term
  dominates and `w_eff` acts as an effective strength spanning **4.03×** across
  the headline's **two** BDG arms — measured 1019 at τ_mult 0.5 and 253 at 1.0
  (§4.1). (An earlier draft said "13× across three arms"; 13× was the withdrawn
  closed form, and there are two arms now.) The headline's **`w_eff` ladder is a
  dose-response argument, not an ablation of the mechanism.**

  **The ablation now carries the control this caveat asks for.** It sweeps
  **w ∈ {1, 4}** at every (η, τ_mult), and `w` and `w_eff` are *not* the same
  knob: `w` scales the mean and deviation terms together, `w_eff` reweights the
  deviation term alone —
  `correction ∝ w·[(y − c − F̄) − w_eff·(F_i − F̄)]`, read off
  [`sampling.py:595-597`](../../proj1/src/sampling.py#L595-L597). So the
  separation is available **in the ablation, not here**, and it carries its own
  hazard: the clip is applied *after* `w`, so a w = 4 row may be clip-limited
  rather than controller-limited. See
  [ABLATION_V3_PROTOCOL.md §1.1](ABLATION_V3_PROTOCOL.md). A headline BDG gain
  still has to be reported with the confound stated.
- **A monotone reading of the τ_mult ladder.** On 62 % of guided steps at
  τ_mult 1.5 the deviation term reverses (§4.1). Order rows by **measured**
  `w_eff`, never by τ_mult and never by the withdrawn closed form.
- **That a BDG rung's result came from the controller rather than the clip.**
  Measured: `plug` clips on 884 sample steps, `bdg_e4t0.5` on **2516** (§4.1). At
  `w_eff` of order 10³ the clip truncates most of the dispersion term, so a rung
  may be **clip-limited**. v3 records the clipped-step count per cell but has no
  arm that separates the two, and the clip cannot be removed (222 non-finite
  samples without it). This is the single most likely explanation for a BDG null
  and must be offered as such rather than as "spread control does not help".
- **That any BDG setting beat `plug`, without a multiplicity correction.** The
  ablation runs **17 BDG settings** on 3 properties × 2 bases — and carries no
  comparison arm of its own, so the `plug` it would be read against is the
  **headline's**, at the same n and the same seeds but in the other stage
  tree. The obvious question ("did *any* rung win?") is a max-over-17
  selection, and
  `BDG_REVIEW.md` already named "the max-over-13 selection" as the larger threat
  to the port's apparent result. `v3_table.py` prints the best arm's z against
  unguided; that z is **not** selection-adjusted. Any win claimed from the grid
  needs a max-T or Bonferroni threshold stated beside it.
- **BDG's controller state per batch.** Each cell pools **n ÷ 500**
  controllers (§2.4), and the recorded `e`, `V_b/τ²` and `w_eff` are means
  over them. A claim about the controller's *trajectory* — that it converged,
  that it sat at its equilibrium — cannot be made from a cell-level mean.
- **That BDG works.** The 25 Sep review found that on the 64 existing port
  cells, no floor-clearing BDG cell beat `plug` at either target, over 72 paired
  tests. v3 is a fair, larger test at the target where BDG's premise holds, and
  its most likely outcome is another null. That is a legitimate result the rubric
  rewards investigating — but it should be pre-registered as the expectation, not
  discovered as a disappointment. Read
  [BDG_REVIEW.md](../methods/BDG_REVIEW.md) before writing any BDG claim.

---

## 7. Running it

```
V3_N=<n> bash proj1/cluster/submit_v3.sh            # headline (the default)
V3_N=<n> bash proj1/cluster/submit_v3.sh ablation   # the grid, afterwards
```

**`V3_N` is required** — v3 fixes no cell size, and every entry point refuses
without it rather than falling back to a default nobody chose. Pick it against
[V3_POWER.md](../results/V3_POWER.md) (§6.1), and make it divisible by 500.

From a checkout that is not Henry's Betty tree, set `CGM_PROJ` (and `CGM_VENV`
if the virtualenv is elsewhere); `submit_v3.sh` forwards both.

| what | where |
|---|---|
| the two stages | `transfer_sweep.py --stage v3` / `--stage v3abl` |
| which backend gets which pair and arms | `transfer_sweep.V3_BACKENDS` |
| gates (83, closed-form) | `proj1/tests/test_v3.py` |
| the power table | `proj1/scripts/v3_power.py` → [V3_POWER.md](../results/V3_POWER.md) |
| the batch measurement | `proj1/scripts/batch_memprobe.py` → [V3_BATCH_MEMORY.md](../results/V3_BATCH_MEMORY.md) |
| the generator check | `proj1/scripts/verify_generator.py` |
| the job / the chain | `proj1/cluster/v3_run.slurm` / `submit_v3.sh` |
| cells | `results/v3/<backend>/<stage>/n<N>/seed<S>/tr__*.json` |
| the tables | `v3_table.py --stage v3\|v3abl --n <n>` → `docs/results/V3_RESULTS*.md` |
| Modality 2 | **not this protocol** — [MODALITY2_V3_PLAN.md](MODALITY2_V3_PLAN.md) |

**The generator is pinned by MODEL, not by file.** `weights/fm_ema.pt` (in
every clone) and `proj1/checkpoints/fm_last.pt` (Betty only) carry the same EMA
tensors; `verify_generator.py` accepts either and rejects anything else.

Nothing above touches `results/sweep`, `results/full`, `results/basecmp` or the
transfer tree.
