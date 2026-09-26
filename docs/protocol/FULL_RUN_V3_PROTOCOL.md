# Full run v3: one strength, one target, two base models

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
| strength | **w = 1 for every arm** | per-arm, frozen under a chemistry floor |
| guidance window | **t ≥ 0.5** for every arm | t ≥ 0.5 |
| seeds | 20261001 / 20261002 / 20261003 — **unchanged** | the same three |
| n | **2000** per cell in batches of **500**, 6,000 per arm pooled | 5000, batch 128 |
| chemistry floor | **none — nothing is excluded** | 0.9 × unguided, a hard gate |
| arms | unguided, plug, tmpd, lgd_mc, tfg, **BDG** — 7 in all | + btvg, btvg_var; no BDG |
| base models | **both** — ours and EquiFM | ours only |
| δ | pre-registered, 2 × f_B's val MAE | the same, plus a post-hoc local variant |

**Arms.** The full comparison set with `btvg` **dropped** and **BDG** added as
the innovation target. BDG's headline ladder is η = 4 with **τ_mult ∈ {0.5,
1.0}**; §4 extends it over both knobs.

**Revision of 26 Sep (Henry), folded in throughout.** Three lines changed, and
nothing else:

| | was | now | what it costs |
|---|---|---|---|
| n per cell | 5000 | **2000** | 4 controllers per cell, not 10; every binomial se ×1.58 (§6.1) |
| headline τ_mult | {0.5, 0.75, 1.0} | **{0.5, 1.0}** | 7 arms, not 8. No `w_eff` span lost — 0.75 was the middle rung — but the headline ladder is now **two points** and cannot show monotonicity or curvature (§4.1) |
| seeds | three | **three** | nothing; unchanged |

**`τ_mult` is BDG's setpoint knob**, τ = τ_mult × `f_A.y_std`. It is *not* the
`t ≥ 0.5` guidance window, which did not change. The two are easy to confuse
because both get called "tau" in conversation; everywhere in this document
τ_mult is the setpoint and `t` is time.

Cost fell from ≈ 35.6 GPU-h to **≈ 12.8** (§5). The ablation moved with it, from
n = 2000 to n = 1000, for a reason that is **not** budget — see §5.1.

**Why `btvg` is dropped.** Not because it lost — because it is finished as a line
of work and its slot buys BDG's ladder. Its mean term is plug to within 9e-8, so
it was never an independent arm; its variance term was diagnosed as *harmful* (it
drags the mean, widens the spread and consumes the clip), and at w = 4 it moves
mu and alpha the **wrong way**; the BTVG-2 estimator fix raised chemistry but
never raised in-band on a single pair, and the `xproj` variant was 0/6. Dropping
it also drops `btvg_var`, which costs v3 something specific — see §6.

**Both base models run identical code**, one flag apart
(`--backend fm` / `--backend equifm`). Ours is the flow-matching EGNN this
project trained; EquiFM is the borrowed base (Song et al. 2023).

---

## 2. The three choices that shape how v3 must be read

### 2.1 No chemistry floor, so no arm is excluded — and no in-band ranking

v2 gated on molecule stability ≥ 0.9 × unguided. v3 removes the gate
(Henry, 26 Sep). Nothing is disqualified; chemistry is **reported** instead.

That has a consequence which must travel with every v3 number. `w = 1` is far
above some arms' best strength: under v2, `tfg` froze at **0.01–0.05**, so
w = 1 is 20–100× it. Measured on our base at q50, mu, n = 512:

| arm at w = 1 | in_band | mol_stab | validity |
|---|---|---|---|
| unguided | 0.0723 | 0.4023 | 0.75 |
| plug | 0.0840 | 0.3730 | 0.731 |
| tmpd | 0.0957 | 0.3906 | 0.746 |
| lgd_mc | 0.0859 | 0.3770 | 0.766 |
| **tfg** | **0.3828** | **0.2285** | **0.619** |

(On gap, tfg reaches 0.3164 at validity 0.4668.) These are **indicative, not the
run**: single-seed at small n, where v3 is n = 2000 × 3 seeds. They come from
`transfer_sweep.py`'s and `v3_table.py`'s docstrings, which is the only place this
project records them, and they are quoted here exactly as recorded — including
unguided validity 0.75.

**Do not build anything on the precision of this table.** An earlier draft of this
section "corrected" 0.750 to 0.78516 and cited "the constant at
`transfer_sweep.py:262`". That string does not occur anywhere in that file; the
citation was fabricated and the correction is withdrawn. The docstrings themselves
disagree about the n these figures were taken at, and nothing in the repo resolves
it. v3's own unguided cells supersede all of it in three properties × three seeds
at n = 2000 — that is what any published number must come from.

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

The calibration set is **3000 molecules drawn from `train_a` + `train_b`**
(`calibration_indices`, `transfer_sweep.py:736`), which documents why: *not* `val`,
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

### 2.4 Ten controllers per cell, not one — a measured deviation

For every other arm the batch is a performance knob. For BDG it is **the
estimator**: `V_b` is the variance of the guide's predictions over whatever tensor
the sampler is handed, so the batch size *is* the controller's sample size.
`guidance.py` states the requirement — "The sweep must use one batch per cell."

**v3 does not meet it, because one batch of 5000 does not fit.** Measured with
[`batch_memprobe.py`](../../proj1/scripts/batch_memprobe.py) over batches
32/64/128, peak allocation linear in the batch and flat in the step count, both
backends reproduced exactly ([V3_BATCH_MEMORY.md](../results/V3_BATCH_MEMORY.md)):

| backend | GiB reserved / molecule | one batch of 2000 would reserve |
|---|---|---|
| ours (`fm`) | 0.0401 | **80 GiB** |
| EquiFM | 0.0560 (1.85× ours) | **112 GiB** |

(At the n = 5000 this run was first planned at these were 201 and 280 GiB. Both
still exceed the slice at 2000, so the case for batching is unchanged by the
smaller n — only its margin is.)

Against a `b200-mig45` slice, which is 45 GB = **41.9 GiB**. So:

- **`--batch 500`**, which fits the tighter backend (EquiFM reserves ≈ 28 GiB,
  inside an 80 % budget of 33.5 GiB). 1000 needs 56 GiB. **The batch did not
  move when n did** — 500 is still the largest divisor of n that fits — so the
  memory measurement below stands unchanged and no re-probe was needed.
- A cell therefore runs **4 independent controllers of 500** (2000 ÷ 500). Each
  estimates `V_b` from 500 samples — a **6.3 %** standard error on the variance —
  and the cell's reported controller state (`e`, `V_b/τ²`, `w_eff`) is the **mean
  over the four**. The n cut economised controller **count**, not controller
  **quality**; holding the batch at 500 is what bought that. It does make those
  cell-level diagnostics **1.58× noisier** than at n = 5000, on top of §6.1.
- Every arm uses the same batch, so arms stay paired on the same initial noise.

Two earlier settings were worse and are recorded so they are not re-proposed. The
default batch 128 would have left a **ragged last controller** — at n = 5000, 40
of them with the last over **8 molecules** (a 53 % standard error on its
variance) pooled in as an equal, and at n = 2000, 15 whole plus a remainder of 80;
`--batch = n` would have OOMed every task on its first cell. `proj1/tests/test_v3.py`
now ties the slurm's batch to the probe's recorded recommendation, so changing one
without re-measuring fails a gate.

**Confirmed on Betty, 26 Sep.** The table above extrapolates from a 16 GB laptop
card, so the chain's `preflight` link now measures the real slice before the array
runs (`batch_memprobe.py --confirm 500`, job 8723902's predecessor). Measured on
`b200-mig45`, whose 45 GB the SLURM CLI filter confirms:

| batch 500, real slice | peak allocated | peak reserved | budget |
|---|---|---|---|
| ours (`fm`) | 10.71 GiB | **17.57 GiB** | 33.5 GiB |
| EquiFM | 19.63 GiB | **26.65 GiB** | 33.5 GiB |

Both fit, and the laptop extrapolation (≈20 and ≈28 GiB reserved) was correct and
slightly conservative. **`--batch 500` is measured, not assumed.** The probe writes
`results/v3_batch_memory_betty.json` and
`docs/results/V3_BATCH_MEMORY_BETTY.md` on every run, so the figure travels with
the cells.

---

## 3. The headline run

**Cells.** 7 arms × 3 properties × 3 seeds × 2 base models = **126 cells.**

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
the ablation (§4), at that stage's own n = 1000 and one seed. Dropping the
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
size, the same molecule sizes and the same fixed target, so arms are paired and
a paired test is valid. **Across base models nothing is paired** — different
architectures, different noise schedules, different state scales — so
cross-backend rows are printed side by side and their differences are not
tested.

---

## 4. The ablation

> **The ablation has its own protocol now: [ABLATION_V3_PROTOCOL.md](ABLATION_V3_PROTOCOL.md).**
> Henry revised it on 26 Sep to **one seed** at **n = 1000**, which costs
> **4.2 GPU-h** against the headline's 12.8. It therefore runs all **17** grid
> arms in its own tree (`n1000/`, so the headline's BDG cells cannot be reused),
> with **2 controllers per cell against the headline's 4**, and **no comparison
> arm** — so the grid is read only against itself and **an ablation row is not
> comparable to a headline row.** The paragraph below described the earlier
> equal-n plan and is kept only to show what changed.
>
> **Its n moved 2000 → 1000 when the headline moved 5000 → 2000**, and not for
> budget: `n` is the only thing separating the two stages' trees, so equal n
> would make `v3_table.py` refuse. ABLATION_V3_PROTOCOL.md §1.1 has the detail.
>
> ⚠️ The **"51.3 GPU-h"** that earlier drafts of this line contrasted the
> ablation against was never the headline — it was the *ablation's own* first
> plan (14 grid arms, shared n5000 tree, three seeds). The headline was 35.6
> then and is 12.8 now.

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
and 1.0) are also the headline's. **The two stages no longer share a tree** —
n = 2000 and n = 1000 are separate directories — so those two are computed twice,
once per stage, at each stage's own n. That duplication is real and it is priced
in: the whole ablation is 4.2 GPU-h.

Union of the two arm sets: **22 arms.** Cells: **126** headline + **102**
ablation = **228**.

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

Costed from per-arm NFE from the v1 run and `v2_run.slurm`'s measured 58 min for
a 23.5-unit task at n = 5000 — **on our base**.

The rate is `v2_run.slurm`'s measured 58 min for a 23.5-unit task at n = 5000,
i.e. **2.468 min/unit on our base**. Units are **defined at n = 5000** and are
taken to scale linearly in n, so a task's units are multiplied by n/5000 — 0.4
for the headline's 2000 and 0.2 for the ablation's 1000. Per (property, seed)
the n = 5000-equivalent budget is 23.5 for the 5 comparison arms plus 3.5 per
BDG arm.

| | units (at its own n) | our base | EquiFM at 1.83× | at 2.5× |
|---|---|---|---|---|
| headline task (7 arms, n = 2000) | 30.5 × 0.4 = **12.2** | **30.1 min** | 55.1 min | **75.3 min** |
| ablation task (17 arms, n = 1000) | 59.5 × 0.2 = **11.9** | 29.4 min | 53.7 min | 73.4 min |

| stage | tasks | our base | EquiFM 1.83× | **total 1.83×** | total 2.5× |
|---|---|---|---|---|---|
| headline | 18 (9 per base) | 4.5 GPU-h | 8.3 | **12.8 GPU-h** | 15.8 |
| ablation | 6 (3 per base) | 1.5 | 2.7 | **4.2 GPU-h** | 5.1 |
| **both** | 24 | 6.0 | 11.0 | **17.0 GPU-h** | 20.9 |

**The longest task is 75 min** at the 2.5× margin, against `BUDGET_MIN` = 225 min
(3.75 h) and the 4 h wall — a 3× margin, where at n = 5000 it was 210 against
225.

⚠️ **12.2 units is a FLOOR, not a prediction.** The only *measured* datum is v2's
58 min for 23.5 units at n = 5000; the linear-in-n scaling is assumed, and it is
wrong in one direction. Per-task **fixed** cost does not scale with n —
checkpoint load, `calibration_indices` + `build_pair` over 3000 molecules, and
the second oracle's calibration are paid once per task regardless. At an 84-min
task that was noise; at 30 min it is a visible fraction. **Expect the realized
saving to be less than the nominal 2.8×**, and read the real number off the
`[timing]` line rather than off this table.

**Wall time by concurrency.** 24 h is no longer the binding constraint — the
whole run fits it even serialized.

| array tasks at once | headline (12.8 GPU-h) | both stages (17.0) |
|---|---|---|
| 1 (serial) | 12.8 h | 17.0 h |
| 2 | 6.4 h | 8.5 h |
| 3 | 4.3 h | 5.7 h |
| 4 | 3.2 h | 4.3 h |
| 6 | 2.1 h | 2.8 h |
| all | 0.9 h (longest task) | 1.2 h |

These assume **perfect packing**, which the array does not give: task ids run
all-`fm` (30 min) then all-EquiFM (55 min), so at 2-way the realistic figure is
nearer 8.1–8.5 h for the headline. At 3-way and above the difference is small.

`submit_v3.sh` still defaults to `headline` rather than `all`, but the reason has
changed: it is no longer the wall, it is that the headline's `[timing]` lines
settle EquiFM's real per-pass ratio — the one assumed factor in this table —
before the ablation commits. Check your limit with
`sacctmgr show assoc user=$USER format=User,QOS,MaxJobs,GrpTRES%30`.

### 5.1 Why the ablation moved to n = 1000

Not budget. Both stages write one stage label into
`results/v3/<backend>/n<N>/seed<S>/`, so **`n` is the only thing separating their
trees.** While the headline was 5000 that was free; at the headline's new 2000 it
would have put both stages in `n2000/`, and `v3_table.py` — which requires every
(property, arm) at all three seeds — would have globbed the ablation's 15
single-seed arms into the headline's table and refused, 45 problems per backend.

Moving the ablation's n fixes that and also preserves §4's doctrine that an
ablation row is not comparable to a headline row, which rests on n, seed count
and controller count all differing. `n` must stay divisible by the batch, 500;
1000 gives 2 whole controllers per cell.

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

### 6.1 The power n = 2000 costs — pre-registered, not discovered

**This is a limitation of the design, accepted on 26 Sep with the n cut, and it
must be stated beside any v3 null.** It is written here *before* the run so that
a null is read as "this run could not resolve it" where that is true, and not as
"the method does nothing".

Pooled n per (arm, property) falls **15,000 → 6,000**. Every binomial standard
error rises by **√2.5 = 1.58×**:

| | n = 5000/cell, 15,000 pooled | n = 2000/cell, 6,000 pooled |
|---|---|---|
| se on in_band at p ≈ 0.09, per cell | 0.405 pp | **0.640 pp** |
| se on in_band at p ≈ 0.09, pooled | 0.234 pp | **0.369 pp** |
| se on a two-arm **contrast**, pooled | 0.310 pp | **0.490 pp** |

§4 pre-registers that "significance is the binomial standard error alone".
Applying that to the indicative q50 figures in §2.1:

| contrast | gap | z at 15,000 | z at 6,000 |
|---|---|---|---|
| plug − unguided | 1.17 pp | 3.8 | **2.4** |
| tmpd − unguided | 2.34 pp | 7.3 | 4.6 |

A Bonferroni threshold for the headline's 7 arms × 3 properties is z = 3.02
(α = 0.05/21). **`plug − unguided` crosses it at n = 5000 and does not at
n = 2000.** `tmpd` survives comfortably.

So, stated plainly:

- **v3 at n = 2000 cannot resolve an in-band difference of about 1 pp** under a
  multiplicity-corrected threshold. It resolves roughly **1.5 pp and up**.
- **That is the size a BDG-vs-plug effect is expected to be.** §6's "cannot" list
  already pre-registers a null as the most likely outcome; this section adds that
  **a null at n = 2000 is weaker evidence against BDG than a null at n = 5000
  would have been**, and the write-up must say so rather than reporting "no
  difference" flat.
- The **ablation is weaker still** — one seed at n = 1000, se ≈ 0.9 pp per cell,
  so only ~2.5 pp differences are resolvable there before the 17-way selection is
  accounted for.
- BDG's **controller diagnostics** (`e`, `V_b/τ²`, `w_eff`, `bdg_w_eff_sq`,
  `bdg_w_eff_neg`) are means over **4** controllers instead of 10, a further
  1.58× on their own noise (§2.4).

What is *not* affected: **pairing**. Arms within a cell still share initial noise
at any n, so the paired structure — and the §4 finding that counting metrics were
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
- **That BDG's spread control is what did anything.** v3 has no variance-only
  rung: `btvg_var` went with `btvg` (§1). Within v3, η = 0 is exactly `plug`, so
  `bdg − plug` bounds the whole dispersion term's effect, but nothing separates
  "controlling spread helped" from "pushing harder helped" — **at q50 those are
  the same dial**, because the target sits near the batch mean, so the deviation
  term dominates and `w_eff` acts as an effective strength spanning **4.03×**
  across the headline's **two** BDG arms — measured 1019 at τ_mult 0.5 and 253 at
  1.0 (§4.1). (An earlier draft said "13× across three arms"; 13× was the
  withdrawn closed form, and there are two arms now.) The **`w_eff` ladder is a dose-response
  argument, not an ablation of the mechanism**, and any BDG gain must be reported
  with that confound stated.
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
  ablation puts **17 BDG settings** against `plug` on 3 properties × 2 bases. The
  obvious question ("did *any* rung win?") is a max-over-17 selection, and
  `BDG_REVIEW.md` already named "the max-over-13 selection" as the larger threat
  to the port's apparent result. `v3_table.py` prints the best arm's z against
  unguided; that z is **not** selection-adjusted. Any win claimed from the grid
  needs a max-T or Bonferroni threshold stated beside it.
- **BDG's controller state per batch.** Each cell pools **10 controllers** of 500
  (§2.4), and the recorded `e`, `V_b/τ²` and `w_eff` are means over them. A claim
  about the controller's *trajectory* — that it converged, that it sat at its
  equilibrium — cannot be made from a cell-level mean of ten runs.
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
bash proj1/cluster/submit_v3.sh
```

One paste, five chained jobs, nothing to resubmit unless the table refuses.
Details, the recovery path and the watch commands are in that script's header
and in [BETTY_RUNBOOK.md](BETTY_RUNBOOK.md).

| what | where |
|---|---|
| the two stages | `proj1/scripts/transfer_sweep.py --stage v3` / `--stage v3abl` |
| gates (54, closed-form) | `proj1/tests/test_v3.py` |
| the batch measurement | `proj1/scripts/batch_memprobe.py` → `docs/results/V3_BATCH_MEMORY.md` |
| the job | `proj1/cluster/v3_run.slurm` |
| the chain | `proj1/cluster/submit_v3.sh` |
| cells | headline `results/v3/<backend>/n2000/seed<S>/tr__*.json`; ablation `…/n1000/seed20261001/` |
| the tables | `proj1/scripts/v3_table.py` → `docs/results/V3_RESULTS*.md` |

Nothing above touches `results/sweep`, `results/full`, `results/basecmp` or the
transfer tree.
