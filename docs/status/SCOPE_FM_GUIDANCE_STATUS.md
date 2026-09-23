# Scope and status — FM base model + inference-time guidance

**Written for:** the project team (Bobo, Idea, Haimo), as the single reference for
what exists, what it measured, and what is still missing. Everything here is
restricted to **Modality 1 (QM9) with the flow-matching generator**. The
diffusion generator and Modality 2 (DNA simplex) are explicitly out of scope of
this document and are listed as not-started in §10.

Status as of **2026-09-23**. Submission **29 Sep 08:30**, defence **30 Sep**.

**Latest BTVG decision (23 Sep):** the registered full run is negative against
LGD-MC; BTVG2 has no demonstrated incremental coverage benefit. A fresh audit
also found that BTVG2's mean-orthogonal correction does not remain orthogonal
after the generator pullback, confirmed on 32 validation trajectories. Property
evaluation currently uses continuous atom features while chemistry uses decoded
types; a separate 128-sample audit found band-membership changes on decoding.
See [the failure audit and bounded redesign](../methods/BTVG_FAILURE_AUDIT_AND_REDESIGN.md)
for proofs, measurements, novelty limits and the next experiments. Earlier
screen-era claims and pending labels below are historical; the full-run and
post-full-run sections take precedence. No improved production arm has yet
been established by this audit.

| mark | meaning |
|---|---|
| ✅ **DONE** | run, measured, numbers in hand |
| 🟡 **RUNNING** | executing right now |
| ⏳ **PENDING** | implemented and verified, waiting on compute |
| ❌ **NOT DONE** | not implemented, or implemented but never measured |

---

## 0. Scope

One sentence: **take a frozen flow-matching generator on QM9 and steer it to a
target property value at sampling time, without retraining, and show that our
guidance method beats the published ones.**

In scope: the FM base model; the three property predictors; the guidance arms
(ours and published); the screening sweep; the selection rule.

Out of scope here: the VP-diffusion generator (trained, not swept), Modality 2,
and any training-time guidance.

---

## 1. The pipeline, in execution order

Each stage consumes the one above. This is the order things must run in, not
the order they were built.

| # | stage | produces | status |
|---|---|---|---|
| 1 | `prepare_qm9.py` | `data/qm9.pt` — 133,885 molecules, 4-way split | ✅ DONE |
| 2 | `train_fm.py --split train_a` | `fm_last.pt` — the frozen generator | ✅ DONE |
| 3 | `benchmark_base.py` | EDM-protocol base numbers + published comparison | ✅ DONE |
| 4 | `train_predictor.py` ×6 | `f_A_*`, `f_B_*` — guide and evaluator per property | ✅ DONE |
| 5 | `predictor_table.py` | cross-architecture predictor comparison | ✅ DONE |
| 6 | `guidance_sweep.py --stage main` | screening grid, 9 v1 arms | ✅ DONE (348 cells) |
| 7 | `fit_rch.py` ×3 | Haimo v1 heads | ✅ DONE (all three, 21 Sep 23:38) |
| 8 | `guidance_sweep.py --stage v2` | screening grid, the 3 new arms + 3 unqueued v1 arms | ✅ DONE (342 cells, ×2 seeds) |
| 9 | `select_arms.py` | go/drop verdict per arm | ✅ DONE — 11 survivors, 3 dropped |
| 10 | `final_benchmark.slurm` | full-scale numbers, survivors only, 3 seeds | ❌ NOT DONE |
| 11 | robustness check (SchNet / TFG oracle) | independent scoring of the winner | ❌ NOT DONE |
| 12 | `benchmark_transfer_base.py` | **TFG's released EDM run through OUR evaluator** — the apples-to-apples base-model row, and the hard gate before any transfer cell | ✅ GATE PASSED 23 Sep (n=2000, 1 seed, 100 steps): atom **0.964** / mol **0.673** at `--grid uniform`, 0.959 / 0.638 at `gamma`. The 10k × 3 table row is still to run. [../results/BASE_MODEL_BENCHMARK.md](../results/BASE_MODEL_BENCHMARK.md) §9.4 |
| 13 | `transfer_sweep.py` | **every arm re-run on a BORROWED base model + borrowed guide/oracle (TFG's `EDMsecond` + `tf_predict_*` + `evaluate_*`)** | ⏳ READY, 0 cells — re-cut 23 Sep to the main sweep's compare → freeze → full protocol (6 arms, 216 compare cells + 18/prop-seed full), 74/74 gates. A **silent feature-scale defect** was found and fixed first (EDMsecond is one-hot/4, TFG's guides one-hot/8). See [../protocol/TRANSFER_EXPERIMENT_PLAN.md](../protocol/TRANSFER_EXPERIMENT_PLAN.md) §9 |

---

## 2. Data and split — how we got it ✅ DONE

`data/qm9.pt`, built by `proj1/scripts/prepare_qm9.py`.

- **133,885 molecules**, atom types `['H','C','N','O','F']`, max 29 atoms, padded with a mask.
- Coordinates centred to zero centre-of-mass; features are **unscaled one-hot** (see §4 — this is the leading suspect for our stability gap).
- **Four-way split, seed 20260917**, deliberately not the standard EDM split:

| split | n | used for |
|---|---|---|
| `train_a` | 51,527 | the generator **and** `f_A` (the guide) |
| `train_b` | 51,527 | `f_B` (the evaluator) **only** |
| `val` | 17,748 | early stopping; the sweep's starting masks |
| `test` | 13,083 | held out, untouched |

**Why this split.** The guide `f_A` and the evaluator `f_B` are trained on
**disjoint halves**. If we scored guidance with the same network that produced
it, a high score could just mean guidance found that network's blind spots.
`f_B` never sees `train_a`, so `prop_mae_eval` is an honest measurement.
This costs us: halving the data weakens the generator (§4).

---

## 3. The FM base model ✅ DONE

**How we got it.** `proj1/scripts/train_fm.py`, recorded in the checkpoint's own
`args`:

| setting | value |
|---|---|
| architecture | EGNN, **hidden 256, 8 layers** |
| objective | flow matching, linear interpolant `x_t = t·x₁ + (1−t)·ε`, target `u = x₁ − ε` |
| trained on | `train_a` (51,527 molecules) |
| epochs / batch | **1500** / 256 |
| optimiser | Adam, lr 2e-4, cosine to 5% |
| EMA | 0.9999 (sampling uses the EMA weights) |
| seed | 20260918 |
| sampling | 100 Euler steps |
| md5 | `a190ac8394902027d4a951f8d30e8c5c` |

### Measured, under the EDM protocol (3 seeds × 10,000 samples) ✅

*(Section 9 of the benchmark doc sets up re-measuring the published row below
through our own evaluator; until that runs, it is a quoted number.)*

| metric | ours | like-for-like published |
|---|---|---|
| atom stability | **0.937 ± 0.001** | 0.9837 |
| molecule stability | **0.40 ± 0.01** | 0.8174 |
| validity | **0.76 ± 0.003** | — |

The comparison is an **unconditional EDM trained on one 50K half**, scored with
EDM's own code (EEGSDE Table 6). We are **4.9 atom points** behind.

**What we ruled out** — this is the part worth defending:
- ❌ Not the half-data split: halving costs EDM only 0.33 atom points at equal compute.
- ❌ Not sampler steps: 1000 Euler steps buys nothing over 100.
- ✅ Leading suspect: **feature scaling.** EDM's own ablation shows one-hot scaling is worth ~2 atom points and ~35 molecule points. We use **unscaled** one-hot; the half-data EDM that beats us scales by 1/8.

**Consequence for the whole project:** every guidance number below is measured
on a generator whose unconditional molecule stability is 0.40. Guidance is
evaluated on *relative* improvement, never on absolute chemistry.

---

## 4. Property selection ✅ DONE

Three properties, chosen to span different difficulty regimes rather than to
flatter the method:

| property | meaning | units | data mean ± sd | why chosen |
|---|---|---|---|---|
| `mu` | dipole moment | D | 2.704 ± 1.539 | the standard EDM/EEGSDE conditioning target; directly comparable to published work |
| `alpha` | isotropic polarisability | Bohr³ | 75.22 ± 8.20 | strongly size-dependent, so guidance must change the molecule, not just polish it — the hard case |
| `gap` | HOMO–LUMO gap | Ha | 0.2513 ± 0.0475 | electronic, weakly correlated with size — the case where a local edit should work |

### Targets — how the numbers were derived

Targets are **quantiles of the `train_a` property distribution**, so they are
values the generator could plausibly produce:

| property | `q50` (median) | `q90` |
|---|---|---|
| `mu` | 2.4932 | 4.6627 |
| `alpha` | 75.54 | 85.05 |
| `gap` | 0.2496 | 0.3162 |

`q50` is the easy interpolation case; `q90` asks for the tail.

---

## 5. The property predictors `f_A` / `f_B` ✅ DONE

**How we got them.** `proj1/scripts/train_predictor.py`, one pair per property.

| setting | value |
|---|---|
| architecture | **EGNN, hidden 128, 4 layers** (smaller than the generator) |
| epochs | 120 |
| `f_A` trained on | `train_a` — same half as the generator |
| `f_B` trained on | `train_b` — disjoint |
| normalisation | targets standardised by `train_a` mean/sd, stored in the checkpoint |

### Measured validation MAE (physical units) ✅

| property | `f_A` (guide) | `f_B` (evaluator) | **δ = 2 × f_B MAE** |
|---|---|---|---|
| `mu` | 0.08971 D | 0.08399 D | **0.16799 D** |
| `alpha` | 0.24474 Bohr³ | 0.24068 Bohr³ | **0.48135 Bohr³** |
| `gap` | 0.00390 Ha | 0.00380 Ha | **0.00760 Ha** |

**δ is pre-registered**, fixed before any guidance ran: a sample counts as
on-target if `|f_B(x) − y*| ≤ δ`, and δ is twice the evaluator's own error. We
cannot claim accuracy finer than the instrument measuring it.

### Also built ✅ — cross-architecture predictors

`predictor_table.py` compares three architectures per property (**12 predictors
total**): EGNN, `InvariantTransformer` (distance-bias attention), and
`RidgeDescriptor` (closed-form, no parameters). Architecture diversity between
`f_A` and `f_B` is **our own extra-rigour idea**, not standard practice — the
usual setup reuses one architecture. EGNN/EGNN was selected as the operating
pair on measured MAE.

❌ **NOT DONE:** the robustness check that re-scores the winning arm with an
*independent* oracle (SchNet for `mu`, TFG's released oracle for `alpha`/`gap`).
Until that runs, all property claims rest on our own `f_B`.

---

## 6. Metrics — exact definitions ✅ DONE

All from `proj1/src/evaluation.py::evaluate_samples`, computed per cell on
n = 512 samples.

### Property metrics — always scored with `f_B`, never `f_A`

| metric | definition | note |
|---|---|---|
| `prop_mae_eval` | `mean|f_B(x) − y*|` | **the primary metric.** Uses every sample's magnitude, so it has far more power than band coverage |
| `prop_rmse_eval` | `sqrt(mean(f_B(x) − y*)²)` | with MAE gives `Var(|e|) = RMSE² − MAE²` **exactly**, so se(MAE) needs no distributional assumption |
| `in_band_fraction` | `fraction(|f_B(x) − y*| ≤ δ)` | **the acceptance criterion.** A binomial, so se = √(p(1−p)/n) ≈ 0.013 at n=512 — low power |
| `f_B_mean` | mean predicted property | with the target gives the **bias** |
| `guide_eval_gap_mean/max` | `|f_A(x) − f_B(x)|` | **reward-hacking detector.** If guidance exploits `f_A`, this grows with strength. Measured flat ⇒ no hacking |

### Chemistry metrics

| metric | definition |
|---|---|
| `atom_stability` | fraction of atoms with correct valence, Hoogeboom bond tables |
| `mol_stability` | fraction of molecules where **every** atom is stable |
| `validity` | RDKit-parseable, via the largest connected fragment |
| `uniqueness_of_valid` | distinct SMILES among valid |
| `n_nonfinite` | exploded trajectories — counted as failures, excluded from property means |

### Diversity metrics

`diversity_mean_pairwise` and `diversity_logdet`, computed in **`f_B`'s
embedding** — needed because guidance can hit the target by collapsing to one
molecule.

### Cost metrics

`cost.{gen_fwd, gen_vjp, gen_jvp, guide_fwd, guide_bwd, guide_hvp}` and
`field_evals` are recorded per cell, so "at matched compute" is **measured, not
asserted**.

---

## 7. The arms — ours vs published

`COMPARE` = published method reproduced on our generator.
`OWN` = candidate for the methodological innovation the project requires.

### v1 arms — in the running sweep

| arm | class | what it is | status |
|---|---|---|---|
| `unguided` | baseline | no guidance — the reference | ✅ DONE (q50), 🟡 q90 |
| `plug` | **COMPARE** | DPS-style `(y − f(m))/s² · Jᵀg` | ✅ DONE |
| `tmpd` | **COMPARE** | TMPD / ΠGDM — adds the uncertainty denominator `s² + gᵀΣg` | 🟡 RUNNING |
| `tfg_mc` | **COMPARE** | TFG (Ye et al. 2024) — K isotropic perturbations at a **tuned** σ, softmax-weighted | ✅ DONE |
| `lgd_mc` | **COMPARE** | likelihood marginalisation, scale **read off the model** (`r² = tr(Σ)/d`) | ✅ DONE |
| `osc` | **COMPARE** | observable-space closure, scalar law integrated analytically | ✅ DONE |
| `smg` | **OURS** | **SMG as shipped** — mean correction `c = ½tr(HΣ)` plus the TMPD denominator | ✅ DONE |
| `smg2` | **OURS** | completed-moment closure — denominator gains `½tr((HΣ)²)` | ✅ DONE |
| `smg2_curv` | **OURS** | `smg2` + the Stein direction `((r²−S)/S²)·ΣHΣg` | ✅ DONE |

### v2 arms — implemented, verified, never measured

| arm | class | what it attacks | status |
|---|---|---|---|
| `spbc` | **OURS (new)** | **bias**, measured up to 13.99 δ. Gives every trajectory the *same* property increment | ⏳ PENDING |
| `btvg` | **OURS (new)** | **spread**, 4–18 δ. Descends `KL(N(μ_F,V_F) ‖ N(y*,τ²))` instead of maximising a likelihood | ⏳ PENDING |
| `shg_*` ×4 | **OURS (new)** | the **window effect**, the largest measured signal. Different arms own different t-bands | ⏳ PENDING |
| `btvg_mean` / `btvg_var` | **OURS** | BTVG ablation halves | ⏳ PENDING |
| `smg_mean` | **OURS** | completes the SMG ablation ladder | ⏳ PENDING |
| `band` | **OURS** | tolerance-band step | ⏳ PENDING |
| `rch` | **COMPARE** | **Haimo v1** — residual calibration head. Prior work (Residual grad-DB, VGG-Flow), run as a *deflation test* for SMG | ⏳ PENDING (needs §1.7) |
| `escalate` | **OURS** | closure-residual escalation | ❌ NOT IMPLEMENTED — design only |

**Why the new three are innovations, briefly.** SPBC inverts what every
published arm does — they apply a *shared coefficient to a per-sample gradient*;
SPBC applies a *per-sample coefficient* so the realised increment is shared.
BTVG changes the *objective*, not the estimator: the Gaussian likelihood
score provably **widens** the property distribution when the target is far,
which is correct inference and wrong for design. SHG exploits a measured
U-shaped window effect that no fixed arm can.

---

## 8. Results so far ✅ — and the headline is not what we expected

Screening config: **n = 512, 100 Euler steps, batch 128, seed 20260921, q50
target, `t_min_guide` = 0.05, strength w = 1.** σ is the two-sample
significance of the MAE improvement over `unguided` (conservative — the cells
are paired, so true σ is larger).

### `mu` — δ = 0.168 D, unguided MAE 1.2204 (7.27 δ), in-band 0.072

| arm | class | MAE | in δ | σ vs unguided | in-band | mol-stab |
|---|---|---|---|---|---|---|
| `tfg_mc` | COMPARE | 0.9634 | 5.73 | **+4.97** | 0.107 | 0.377 |
| `plug` | COMPARE | 0.9753 | 5.81 | **+4.66** | 0.104 | 0.324 |
| `smg` | **OURS** | 1.0049 | 5.98 | **+4.14** | 0.092 | 0.324 |
| `lgd_mc` | COMPARE | 1.0336 | 6.15 | +3.60 | 0.090 | 0.422 |
| `osc` | COMPARE | 1.0340 | 6.16 | +3.55 | 0.098 | 0.418 |
| `tmpd` | COMPARE | 1.0656 | 6.34 | +2.87 | 0.092 | 0.365 |
| `smg2` | **OURS** | 1.1040 | 6.57 | +2.15 | 0.080 | 0.375 |
| `smg2_curv` | **OURS** | 1.1814 | 7.03 | +0.70 | 0.084 | 0.387 |

### `alpha` — δ = 0.481 Bohr³, unguided MAE 6.7052 (13.93 δ), in-band 0.043

| arm | class | MAE | in δ | σ vs unguided | in-band | mol-stab |
|---|---|---|---|---|---|---|
| `osc` | COMPARE | 5.6358 | 11.71 | **+3.33** | 0.066 | 0.400 |
| `smg2_curv` | **OURS** | 6.1123 | 12.70 | +1.82 | 0.059 | 0.375 |
| `smg2` | **OURS** | 6.1417 | 12.76 | +1.70 | 0.066 | 0.389 |
| `tmpd` | COMPARE | 6.2252 | 12.93 | +1.46 | 0.045 | 0.400 |
| `lgd_mc` | COMPARE | 6.7585 | 14.04 | −0.16 | 0.041 | 0.377 |
| `smg` | **OURS** | 9.5204 | 19.78 | **−6.22** | 0.043 | 0.270 |
| `tfg_mc` | COMPARE | 9.9865 | 20.75 | **−8.20** | 0.023 | 0.246 |
| `plug` | COMPARE | 10.0239 | 20.82 | **−8.22** | 0.029 | 0.264 |

### `gap` — δ = 0.0076 Ha, unguided MAE 0.0382 (5.03 δ), in-band 0.105

| arm | class | MAE | in δ | σ vs unguided | in-band | mol-stab |
|---|---|---|---|---|---|---|
| `plug` | COMPARE | 0.0343 | 4.52 | +2.64 | 0.115 | 0.285 |
| `tfg_mc` | COMPARE | 0.0348 | 4.59 | +2.27 | 0.113 | 0.369 |
| `tmpd` | COMPARE | 0.0355 | 4.68 | +1.78 | 0.125 | 0.367 |
| `smg` | **OURS** | 0.0364 | 4.79 | +1.18 | 0.117 | 0.324 |
| `lgd_mc` | COMPARE | 0.0364 | 4.79 | +1.16 | 0.123 | 0.412 |
| `smg2` | **OURS** | 0.0373 | 4.91 | +0.57 | 0.094 | 0.396 |
| `osc` | COMPARE | 0.0376 | 4.95 | +0.38 | 0.119 | 0.404 |
| `smg2_curv` | **OURS** | 0.0385 | 5.06 | −0.16 | 0.102 | 0.404 |

### What these say — read this before the write-up

1. **The ranking inverts completely across properties.** `plug` and `tfg_mc`
   are the two best arms on `mu` (+4.66, +4.97) and are **catastrophic** on
   `alpha` (−8.22, −8.20 — materially *worse* than not guiding at all). No arm
   wins everywhere. Any claim of the form "method X is best" is false as stated.
2. **Nothing clears 3σ on band coverage anywhere.** Best is `tfg_mc` on `mu` at
   1.97σ. So the honest claim today is *guidance reduces property error*, not
   *guidance gets samples into the band*. **That gap is exactly what SPBC and
   BTVG were built for** — and it is unmeasured until stage v2 runs.
3. **Our SMG family does not currently win.** `smg` is 3rd on `mu`, 6th on
   `alpha`, 4th on `gap`; `smg2_curv` is last or near-last on all three and is
   statistically indistinguishable from doing nothing on `mu` (+0.70σ). This is
   a real result and we should report it, not bury it.
4. **`alpha` is the hard property.** Everything sits at 11–21 δ; the best
   in-band is 0.066. Whatever we claim about the innovation will be most
   convincing there.
5. **Guidance costs chemistry.** Unguided mol-stability is 0.402 (`mu`);
   `plug` and `smg` both drop to 0.324, and on `alpha` `plug` falls to 0.264.

---

## 9. Infrastructure built ✅ DONE

| item | what it does |
|---|---|
| `guidance_sweep.py` | resumable cell-per-JSON sweep driver; ordered so a partial run is still a complete experiment |
| `--preflight` | one throwaway cell per arm, 15 s, fails fast instead of burning a 4-hour queue window |
| `select_arms.py` | screening→selection with a **conservative, pre-registered** drop rule (§ below) |
| `test_arms_exact.py` + `test_v2_arms.py` | **43 closed-form gates**, all passing on both the 5080 and the cluster |
| `mutation_test.py` | **29/29** injected defects caught; runs against a sandbox copy, never the working tree |
| provenance stamping | every cell now records checkpoint md5, torch/CUDA version, GPU name |

**The drop rule** (`select_arms.py`): default is **KEEP**. An arm is dropped
only if, on *every* property, it is (a) >3 combined se below the best arm on
band coverage **and** (b) beats unguided on **neither** coverage nor MAE by 3
se — or it collapses chemistry everywhere *and wins nothing*. Divergent cells
(non-finite, or MAE >10× unguided) cannot win a "best strength" slot.

---

## 10. What is NOT done

### Blocking for M1
| item | status |
|---|---|
| **borrowed-base-model benchmark** | ⏳ built and gated, 0 runs. Turns one row of the published comparison column from *quoted* into *measured*, and is the hard gate for the line below |
| **transfer experiment — nothing in it is ours except the guidance field** | ⏳ 0 cells; harness ready and gated (plan §9), run on Betty with `transfer_run.slurm`. The professor's ask: the innovation is the guidance, so the guidance is what must be portable |
| stage-main sweep completion | 🟡 195 / 390 cells |
| RCH heads for `alpha`, `gap` | ⏳ only `rch_mu.pt` exists; **hard dependency** of stage v2 |
| **stage-v2 sweep — the three new arms have never been measured** | ⏳ 0 / 366 cells |
| `select_arms.py` verdict | ⏳ needs the above |
| full-scale final run (n = 4096, 3 seeds) | ❌ |
| independent-oracle robustness check | ❌ |
| `escalate` arm | ❌ design only |

### Out of scope here, but on the critical path
| item | status |
|---|---|
| **Modality 2 — DNA on the simplex** | ❌ **zero lines of code.** Timetable says 23–24 Sep |
| VP-diffusion arm of M1 (trained, never swept) | ❌ |
| paper + slides | ❌ |

**The simplex transfer is the largest open risk.** [PLAN_AND_TIMETABLE.md](PLAN_AND_TIMETABLE.md)
already flags it red. Concretely: SPBC's edit must stay in the tangent space
*and* keep coordinates non-negative (an inequality constraint, not a linear
one), and BTVG's `V_F = gᵀΣg` uses `Σ = diag(p) − ppᵀ`, which is **singular by
construction** — we already found `V_F` going negative on QM9 from a merely
*non-symmetric* Σ.

---

## 10b. Corrections found by the completeness audit (2026-09-21)

A full docs-vs-implementation audit found **9 designed methods never
implemented**, **3 implemented but unreachable from the sweep**, and several
doc claims that no longer match the code. The complete inventory, with the
proof document for every method, is in
[../methods/GUIDANCE_METHODS_INDEX.md](../methods/GUIDANCE_METHODS_INDEX.md).
The ones that changed behaviour:

| # | finding | action |
|---|---|---|
| **S1** | `strength_scale` keyed off `arm.startswith("btvg")`, so SHG schedules *containing* a btvg phase were unnormalised — the identical field ran **3139× stronger** inside `shg_plug_btvg` than in standalone `btvg`, guaranteed clip-saturated | ✅ fixed: `scale_schedule()` normalises **per phase** |
| **D1** | `band` returns a state displacement but was **not** in `DISPLACEMENT_MODES`, so the sampler multiplied it by `(1−t)/t` — **×19 at t=0.05**. The most plausible mechanical cause of the recorded "band diverges above w=0.05" | ✅ fixed: `band` added |
| **K1** | the **κ₃ skew probe** — TOP6's #1-ranked idea, implemented and unit-tested — was **never switched on** by the sweep, so the programme's stated go/no-go has never been measured | ✅ fixed: `--kappa3` flag |
| **B2** | after the `(τ/s)²` normalisation, `btvg_mean` is **provably identical to `plug`** (verified to 1.4e-17) | 📝 not a bug: **BTVG's novelty is entirely in the variance term.** Say so in the paper |
| **⚠️ osc** | labelled prior art in every doc, but **no citation for it exists anywhere in this repo** — and it is the **best arm on `alpha`** | ❌ **unresolved.** Find the citation or relabel it as ours |
| **⚠️ bias claim** | "alpha/plug 13.99 δ, alpha/smg −5.69 δ" mixes **three different strengths in one sentence**. At a consistent w=1 it is plug **+13.99** vs smg **−15.06** — SMG is *more* biased than plug | ❌ **fix before the paper.** This sentence is SPBC's whole motivation |
| **⚠️ citations** | `proj1_tex/citation.bib` has 28 entries and contains **none** of TFG, DPS, TMPD, ΠGDM, LGD, EEGSDE, OC-Flow, D-Flow or FlowGrad | ❌ every method we compare against is uncited |
| **⚠️ external baselines** | the assignment wants ≥3 relevant methods published within 5 years; we reproduce five **locally**, and whether that satisfies the requirement is argued nowhere | ❌ grading risk. `FlowGrad` is the cheapest to add |

### The audit's headline

> The project has spent its entire measurement budget on arms that lose, and has
> zero measurements on every arm built to fix the reason they lose.

`smg` is beaten by prior art on **all three properties at every protocol**
(alpha: osc 4.356, lgd_mc 4.681 vs smg 5.657 · mu: tmpd 0.796, plug 0.801 vs
smg 0.966 · gap: tfg_mc 0.031, plug 0.032 vs smg 0.035, all at best strength).
Meanwhile the **largest measured effect in the project is the guidance window —
47 %** (alpha/plug MAE 10.02 → 5.32 as `t_min_guide` goes 0.05 → 0.5), which is
bigger than every arm difference combined, and it is a **scheduling** result.

**"When you guide matters more than how" is currently our strongest defensible
claim, and the arm built to exploit it (`shg_*`) has never run.**

---

## 10c. Stage-v2 results (22 Sep) — the three new arms, measured

690 cells at seed 20260921 plus a full replication at seed 20260922 (338 cells).
Screening config: n=512, 100 Euler steps, **t_min_guide = 0.5**, q50 target.
σ is against `unguided` on MAE.

| arm | class | mu | alpha | gap | verdict |
|---|---|---|---|---|---|
| `btvg` | **OURS (new)** | **+8.9σ** | **+6.7σ** | **+7.0σ** | PROCEED — tied for best on all three |
| `shg_plug_btvg` | **OURS (new)** | +9.2σ | +4.6σ | +7.3σ | PROCEED — highest band coverage on mu (0.131) |
| `shg_plug_spbc` | **OURS (new)** | +7.7σ | +4.5σ | +6.6σ | PROCEED |
| `shg_three` | **OURS (new)** | +6.3σ | +4.5σ | +2.8σ | PROCEED |
| `shg_smg_spbc` | **OURS (new)** | +2.6σ | +6.4σ | +0.8σ | PROCEED |
| `tmpd` | PRIOR | +8.1σ | **+8.1σ** | +5.2σ | reference — strongest single arm |
| `plug` | PRIOR | +9.2σ | +4.5σ | +7.8σ | reference |
| `smg_mean` | ABLATION | +2.6σ | +7.3σ | +2.4σ | PROCEED |
| `smg` | **OURS** | +4.2σ | +6.8σ | +0.9σ | reference |
| `btvg_var` | ABLATION | +0.8σ | +0.4σ | +3.4σ | PROCEED (seed 1) / DROP (seed 2) — **marginal** |
| `spbc` | **OURS (new)** | +0.2σ | +1.3σ | +0.1σ | **DROP — does nothing** |
| `band` | **OURS** | +0.5σ | +0.0σ | +0.6σ | **DROP — bit-identical to unguided on alpha** |
| `rch` | PRIOR (Haimo v1) | +1.2σ | **−1.7σ** | +1.2σ | **DROP — worse than unguided on alpha** |

### Replication across seeds — the project's first error bars

338 cells matched between seeds. `|MAE₁ − MAE₂|` divided by the claimed combined
standard error: **median 0.43, mean 0.54, 1 % above 2σ, none above 3σ.**

For two independent draws the expected median is 0.67, so the observed spread is
**smaller** than the reported se — the error bars are conservative by ~1.5×,
because both seeds reuse the same 512 validation masks and differ only in noise.
**Reported σ understate significance; they do not overstate it.**

Verdict stability: 3 of 3 drops replicate exactly (`band`, `rch`, `spbc`).
`btvg_var` drops in seed 2 only — its single win (gap, +3.4σ in seed 1) falls
below threshold in seed 2. Treat it as **marginal, not established**.

### Full metrics change the conclusion — read this before quoting MAE

The σ table above ranks on MAE alone. On the **full** metric block, at the only
fair slice (w = 1, `t_min` = 0.5 — see below), the picture is different:

| arm | mu MAE / \|bias\| / spread | alpha | gap | mol_stab |
|---|---|---|---|---|
| **`btvg`** (OURS) | **5.38 / 0.02 / 6.52** | 11.38 / **0.66** / 14.79 | **4.25 / 0.32 / 5.18** | 0.31–0.36 |
| `plug` (prior) | 5.85 / 0.97 / 7.05 | **11.04** / 2.13 / **14.21** | 4.49 / 1.26 / 5.25 | 0.36–0.37 |
| `unguided` | 7.27 / 2.48 / 8.75 | 13.93 / 0.78 / 17.95 | 5.03 / 0.99 / 5.92 | 0.402 |

- **`btvg` has the lowest bias of any arm measured** — mu **0.02 δ** against `plug`'s
  0.97 and unguided's 2.48. It centres the property distribution almost exactly.
- It also has the **lowest spread** on mu and gap. It was built to cut spread; it
  cuts spread *and* bias.
- **The irony worth putting in the paper:** `spbc` was designed to correct bias and
  does nothing (mu bias 2.23 vs unguided 2.48). `btvg`, designed for spread, is the
  best bias corrector in the table.
- **Cost:** `btvg` mol_stability 0.31–0.36 against unguided 0.402, validity 0.686 vs
  0.785 on mu. Report in the same row as the win, per R4.
- **`rch` (Haimo v1) on alpha is destructive**, not merely inert: bias **10.42 δ**,
  spread 29.08, mol_stab 0.301.

**Are we failing? No.** `btvg` beats the strongest prior-art arm on MAE, bias and
spread on two of three properties, and is the best-centred arm on all three.

### ⚠️ The best-strength comparison is biased toward us, and must be fixed

The stage-main grid is a **cross, not a full grid**: the strength sweep runs at
`t_min = DEFAULT_WIN = 0.05` and the window sweep at `w = DEFAULT_W = 1.0`. So at
the good window (0.5) the v1-only arms — `tfg_mc`, `lgd_mc`, `osc`, `smg2`,
`smg2_curv` — have exactly **one** cell, at w = 1, the pre-registered default.
Stage v2 ran its **full 7-point strength sweep at window 0.5**.

Any best-strength table therefore compares our *tuned* arms against their *untuned*
ones. **Do not quote best-strength numbers until this is closed.**

Missing: 5 arms × 6 strengths × 3 properties = **90 cells, ~2 GPU-h.**

```bash
sbatch --export=ALL,CMD="python -u proj1/scripts/guidance_sweep.py --arms tfg_mc,lgd_mc,osc,smg2,smg2_curv --props mu,alpha,gap --fm proj1/checkpoints/fm_last.pt --n 512 --batch 128 --steps 100 --max-minutes 215" proj1/cluster/run.slurm
```

Full per-property tables with every metric:
[../results/ARM_RANKINGS.md](../results/ARM_RANKINGS.md)

### THE COMPARISON GROUP — fixed 22 Sep, not to be changed after seeing results

Four published methods plus the baseline. They span the design space on two
axes: analytic-vs-Monte-Carlo, and uncertainty-aware-vs-not.

| arm | published as | what it does | role |
|---|---|---|---|
| `plug` | **DPS**, Chung et al., ICLR 2023 | `(y − f(m))/s² · Jᵀg` — treats the endpoint estimate as exact | the no-uncertainty baseline |
| `tmpd` | **ΠGDM / TMPD**, Boys et al., TMLR 2024 | adds the `s² + gᵀΣg` denominator — analytic first-order uncertainty | closest competitor to our SMG family |
| `tfg_mc` | **TFG**, Ye et al., NeurIPS 2024 | K perturbations at a **tuned** σ, softmax-weighted | the Monte-Carlo alternative to linearising |
| `lgd_mc` | **LGD**, Song et al., ICML 2023 | same MC, scale **read off the model** (`r² = tr(Σ)/d`) | isolates tuned-vs-derived scale |
| `unguided` | — | no guidance | reference |

**Excluded:** `osc` (labelled prior art, **no citation exists in this repo**);
`rch` / Haimo v1 (prior art, but dropped — worse than unguided on alpha).

**How these must be described in the paper.** All four are *our ports* onto our
generator and our `f_A`/`f_B`. ΠGDM in particular was designed for **linear**
inverse problems with a closed-form uncertainty denominator; a nonlinear EGNN
property head on molecules is outside its regime, and its claimed contribution
is calibration and strength-insensitivity, not raw MAE. **We must never write
"TMPD is worse than DPS."** We write: *on our stack, our port of TMPD scores
below our port of DPS.*

### THE RANKING RUBRIC — stated, because it is a choice

Score = mean over {mu, alpha, gap} of
`(MAE_unguided − MAE_arm) / sqrt(se_arm² + se_unguided²)`,
at each arm's **best strength**, slice `t_min = 0.5`, q50, n = 512,
non-finite cells excluded. `se(MAE) = sqrt(RMSE² − MAE²)/sqrt(n)`, exact.

Five choices are baked in, and each changes the answer:

| choice | why | what it hides |
|---|---|---|
| MAE, not `in_band` | in_band is a binomial, se ≈ 0.013 — cannot separate arms at n=512 | in_band *is* the acceptance criterion |
| unweighted mean over properties | no principled weighting exists | treats easy `gap` and hard `alpha` alike |
| best strength (min over grid) | each arm at its own optimum | winner's curse; favours finer grids |
| σ against `unguided` | a fixed reference | rewards *moving*, not band accuracy |
| **chemistry excluded** | it is a cost, not the objective | `btvg` loses 7–9 mol-stability points |

**Under a bias rubric the ranking inverts**: `btvg` wins outright (alpha 1.12 δ
vs `plug` 3.23 δ). Under a chemistry rubric `btvg` is among the worst. **No
single ordering exists, and that is a result, not an inconvenience.**

### `plug` and `btvg` are TIED — do not report an ordering

| property | `plug` MAE | `btvg` MAE | σ |
|---|---|---|---|
| mu | 0.7846 | 0.8078 | −0.67 |
| alpha | 4.4819 | **4.1872** | **+1.33** |
| gap | 0.0277 | 0.0284 | −0.59 |

Mean σ vs unguided: `plug` **8.149**, `btvg` **8.093** — a gap of **0.056**.
No pairwise comparison reaches 1.4 σ. Sorting a table on this number
manufactures a ranking out of noise. Report "statistically tied", with the
per-property σ visible.

### SHG is a NEGATIVE result — report it as one

Every schedule is **≤ its best component** at best strength:
`shg_plug_btvg` +8.0 vs `plug` +8.1 and `btvg` +8.1 · `shg_plug_spbc` +7.3 vs
`plug` +8.1 · `shg_three` +6.1 · `shg_smg_spbc` +4.0 vs `smg` +4.6.
Handing off between arms buys nothing over running the better arm throughout.

### τ is INERT — BTVG is not band-targeted at our operating point

Measured `τ²/V_F` = **0.0011 (mu), 0.0024 (alpha), 0.0038 (gap)**. The variance
coefficient `b = −½·w/s²·(1 − τ²/V)` is therefore τ-independent to **0.2 %**.

So BTVG is **DPS + monotone variance descent**, not band targeting. The real
mechanism is the **sign clamp**: the likelihood's variance coefficient is
`+½[(y−μ)²/S² − 1/S]`, *positive* when the target is far, so standard guidance
**widens** the property distribution exactly when it should concentrate. BTVG's
is always ≤ 0. That is the contribution, and it is what the ablation supports —
the variance term cuts `plug`'s alpha overshoot from 3.23 δ to 1.12 δ.

Either rename the method, or re-sweep τ at ~20× larger values where `τ² ~ V_F`
and band targeting could actually engage.

### The BTVG ablation ladder (all four measured)

| arm | what it is | mu | alpha | gap | \|bias\| alpha |
|---|---|---|---|---|---|
| `unguided` | neither term | 7.27 | 13.93 | 5.03 | 0.78 |
| `plug` | **mean term only** — bit-identical to BTVG's mean half (9e-8) | **4.67** | 9.31 | **3.64** | 3.23 |
| `btvg_var` | **variance term only** | 6.03 | 13.68 | 4.38 | 1.79 |
| `btvg` | both | 4.81 | **8.70** | 3.74 | **1.12** |

### PRE-REGISTRATION — the go/no-go for the full-scale run

Rules are named **FR1–FR5** (full-run rules). **Not G1–G5** — those already
name the experiment plan's gates in `GUIDANCE_EXPERIMENT_PLAN.md`.
Checked mechanically by `proj1/scripts/check_fullrun_go.py`.

**Written 22 Sep, before any `--stage compare` cell has run.** These rules are
fixed now so they cannot be chosen after seeing the numbers.

**What the n = 512 compare stage can and cannot decide.** At n = 512, band
coverage has se ≈ 0.013; the BTVG-vs-`plug` gap (≈ 0.011) is 0.6 σ. The screen
therefore **cannot show BTVG wins**. It decides only whether BTVG is worth the
expensive run.

**FR1 — go/no-go.** BTVG proceeds to the full run unless, on **2 or more of the 3
properties**, it is beaten by the best competitor by **> 3 combined standard
errors on MAE** at best strength. This is deliberately a low bar: it removes
"clearly worse", nothing else. Failing FR1 means BTVG is reported as a negative
result, not re-tuned until it passes.

**FR2 — the full run includes EVERY competitor, not just survivors.** `unguided`,
`plug`, `tmpd`, `lgd_mc`, `dflow`, `btvg`, `btvg_var`. Dropping a competitor
after seeing the screen is selection bias.

**FR3 — strength is chosen on the screen and FROZEN.** Each arm runs at its
best-MAE strength from the n = 512 compare stage (seed 20260921). The full run
uses **new seeds**, so the strength choice is not also the evaluation — the
full-run numbers are an unbiased estimate at the chosen strength.

**FR4 — scale.** n = **10,000** per run (the EDM protocol, and what the base-model
benchmark already uses), **3 seeds**. Reported with seed-to-seed spread, as the
base benchmark is.

**FR5 — the claim.** Scored under the saved rubric: `in_band` gain subject to
`mol_stability ≥ 0.9×` unguided, with `|bias|` and `spread` in the same row.
BTVG "beats" a competitor only at **≥ 3 σ**. Anything less is reported as a tie,
with the σ shown.

### PRE-REGISTRATION ADDENDUM — which target the full run uses (22 Sep, before data)

**FR6 — the full run uses a per-molecule target, `dist`.** Each generated
molecule gets **its own** target: the real property of a held-out **test**
molecule. (Not `val` — `val` already carries generator validation, both
predictors' model selection, `delta`, and the sampled molecule sizes.)

**What the local evidence supports, exactly.** The vendored TFG evaluator
consumes a per-sample target *vector* in physical units and reports per-molecule
MAE (`audit/fa_fb_search/TFG/evaluations/molecule.py:60,76,100`). That
establishes **per-sample scoring** and nothing more: `MoleculeSampler`, which
produces those targets, is **not vendored**, and in TFG's own configs `target`
names the *property* (`target: str`, `utils/configs.py:58`), so the file is
consistent with either protocol.

An earlier draft of this section claimed "no published QM9 guidance result uses
a fixed quantile". **That is an overclaim and is withdrawn** — it is a universal
negative over a literature that cannot be checked from this repo. Before the
write-up, cite the EDM/EEGSDE sampler code or paper text for the conditioning
protocol, or state the protocol as our choice rather than as the field's.

**How `dist` is constructed, exactly.** Take the first `n` molecules of the
**test** split. Molecule *i* of the batch is generated at **that molecule's atom
count**, and its target is **that same molecule's** real property value. Sizes
and targets come from the SAME molecule, which matters: corr(size, alpha) is
**+0.755** in the data, and an earlier version that took sizes from `val` and
targets from `test` destroyed it (**-0.033**) -- a 10-atom molecule asked to hit
a 25-atom molecule's polarisability. Verified end to end: the sampler sees
+0.777, identical to the test data.

The targets are neither above nor below the mean -- they ARE the property
distribution (mu: target mean 2.68 vs data 2.70, sd 1.61 vs 1.54). Roughly half
sit above the generator's output mean and half below, which is why the *mean*
offset is near zero while the *per-molecule* steering demand averages 0.80 sd.

**What is independently true, and is the real argument for `dist`:** it involves
**no choice by us**, so it cannot be cherry-picked, and it is measurably harder
than q50 (0.80-0.85 sd of steering against 0.05-0.27).

**Why not "the q that best shows our work".** Choosing the target after seeing
which one flatters BTVG is cherry-picking. Adopting the literature's own
protocol means we do not choose the target at all, so there is nothing to pick,
and the headline number is directly comparable to published ones.

**FR7 — q50 and q90 are reported as two DIAGNOSTIC tasks, never pooled.**

| task | what it tests | distance from the unguided generator's mean |
|---|---|---|
| q50 | **concentration** | 0.05–0.27 sd — almost no steering needed |
| q90 | **steering** | 1.1–1.6 sd into the tail |

They come from the n = 512 compare stage and say *where* BTVG helps. Every
result reported before 22 Sep is a q50 result, i.e. a concentration result.

**FR1 amended.** The go/no-go is evaluated on q50 and on q90 **separately**, and
both verdicts are reported. BTVG proceeds to the full run if it passes on
**q50**, whose strengths FR3 freezes. A q90 failure is reported, not hidden.

**FR3 for `dist` — CORRECTED 22 Sep, before any dist cell ran.** The first
version said each arm runs at its **q50** best strength, because "`dist`
targets concentrate near the median". **That reasoning was backwards by 3-18x.**
What matters is not where the targets sit but how far each is from the
*generator's own* mean, and that is a mean-absolute-deviation, not ~0:

| prop | dist | q50 | q90 |
|---|---|---|---|
| mu | **0.807 sd** | 0.271 | 1.142 |
| alpha | **0.803 sd** | 0.045 | 1.197 |
| gap | **0.854 sd** | 0.158 | 1.557 |

23-33 % of `dist` targets sit beyond 1.1 sd, i.e. at or past the *entire* q90
task. Compounding it, the q50 reference optimum is at **w = 4, the grid edge**,
so it is not even bracketed.

**So: `dist` freezes at the q90 best strength**, or screens `dist` at three
strengths if compute allows. q50's strength is the wrong approximation.

**Implementation note.** Enabling `dist` exposed a latent bug: the batch loop
passed `y_t[: m.shape[0]]`, slicing from 0 for every batch. Harmless for every
cell run before 22 Sep (q50/q90 targets are constant, every slice identical),
but with per-molecule targets it would have guided 75 % of molecules toward
another molecule's target while scoring them against their own. Fixed to
`y_t[i : i + m.shape[0]]`, and gated: each of 4 batches verified to receive
its own molecules' targets.

### PRE-REGISTRATION ADDENDUM 2 — the full run as queued (22 Sep, before any full-run cell)

**FR4 AMENDED: n = 5,000 per run, 3 seeds (was 10,000).** Compute-bound, not
result-driven: no full-run cell exists yet. Cost is measured from the compare
stage's own Betty logs (see FR3a below): ≈ 218 min per (property, seed) on a
1g slice at n = 5,000, ≈ 36 GPU-h for the whole run as queued. At n = 10,000
it would be ≈ 70 GPU-h, far past the 12 h ceiling set for this run even on
many slices. n = 5,000 gives 15,000 molecules per arm per property.

**What halving costs, stated now.** Independent-samples se of an in-band
*difference* at p ≈ 0.3 is ≈ 0.0053 at n = 5,000 × 3, against ≈ 0.0037 at
10,000 × 3. An in-band gap the size of the q50 screen's BTVG–`plug` gap
(≈ 0.011) would read ≈ 2.1 σ here against ≈ 2.9 σ at the original scale. A
further seed costs ≈ 8 GPU-h and can be appended without re-running anything.

**FR5, which test.** Primary: the **independent-samples** z (conservative).
Supplementary: the **paired** z over the same molecules and the same initial
noise — valid because every arm in a seed shares both, and reported beside the
primary, never instead of it.

**FR5, which metric.** The verdict is decided on **in_band**, the saved
rubric's metric, and only between arms that **both** clear the chemistry floor
(mol_stability ≥ 0.9 × unguided). A win on MAE alone is **not** a win — that
is the ranking the rubric rejected. MAE is reported in its own columns; if it
significantly contradicts in_band (both |z| ≥ 3, opposite directions of
merit) the verdict is "mixed", not whichever metric flatters BTVG.

**Scope of the σ.** All seeds reuse the same 5,000 test molecules, so every se
is conditional on this target set. A claim about the target *population*
needs a molecule-clustered se, up to √3 larger; state which one a claim uses.

**Caveat on FR3-corrected's "0.80 sd", found in review (22 Sep, before data).**
That steering distance was measured from the generator's *global* mean. But
each `dist` molecule is generated at its own size, and size explains much of
alpha: measured against the unguided mean **at the same atom count**, the
demand is alpha 0.48 sd (not 0.76), mu 0.68, gap 0.68, and only 8 % of alpha
targets sit beyond 1.1 sd (not 24 %). The q90 freeze is **not** changed —
FR3-corrected is pre-registered — but the write-up must state that for alpha
`dist` sits nearer the middle of q50 and q90 than the table above suggests.
`full_run_table.py` reports the size-conditional breakdown.

**Seeds** 20261001, 20261002, 20261003 — disjoint from every screening seed.
Seeds vary the initial noise only; all seeds and arms see the **same** 5,000
test molecules (sizes and targets), so every comparison is paired.

**FR3's optional dist screen** ("or screens dist at three strengths if compute
allows") is **not run**: compute does not allow. `dist` runs at q90-chosen
strengths, as FR3-corrected says — see FR3a below for which ones.

### AMENDMENT FR3a — strengths must clear the chemistry floor (23 Sep, after the screen, before any full-run cell)

**What the finished compare stage showed** (258/258 cells, n = 512, seed
20260921). FR1 passes on q50 (beaten on 0/3) and on q90 (1/3 — gap, 3.8 σ).
But FR3's q90 choice — unconstrained best MAE — is **w = 4, the grid edge, for
plug, tmpd, lgd_mc and btvg**, and at w = 4 plug, tmpd and btvg **fail the
chemistry floor on all three properties** (mol_stability 0.21–0.35 against
0.362). FR3 selected on MAE while FR5 judges in_band *under* the floor, so
run as registered, the headline table would have read "fails floor" for btvg
and most competitors: 24+ GPU-h to adjudicate nothing.

**The rule.** FR3a = each arm's **best-MAE q90 strength among those with
mol_stability ≥ 0.9 × unguided**; if none clears it, the most stable strength.
FR3's own metric plus the floor FR5 already imposes; identical for every arm.
It does **not** favour btvg: under it btvg barely clears unguided at q90
(in_band 0.045 vs 0.041 on mu), and `dflow` is the arm that gains — it
*improves* stability (0.43–0.57) while raising in_band.

| arm | mu | alpha | gap | FR3 as registered |
|---|---|---|---|---|
| unguided | 1 | 1 | 1 | 1 / 1 / 1 |
| plug | 0.05 | 0.5 | 0.25 | 4 / 4 / 4 |
| tmpd | 0.5 | 1 | 1 | 4 / 4 / 4 |
| lgd_mc | 4 | 4 | 4 | 4 / 4 / 4 |
| dflow | 0.05 | 2 | 1 | same |
| btvg | 0.05 | 1 | 0.05 | 4 / 4 / 4 |
| btvg_var | 0.01 | 0.01 | 0.05 | 0.01 / 0.01 / 4 |

**Both are run.** FR3a is the **primary** set (3 seeds, the headline, FR5
applies). FR3 as registered is run as a **secondary** set on seed 20261001
only — the 10 cells whose strength differs — so the registered analysis is
reported, not replaced, and an unconstrained-MAE comparison (what the
literature reports) exists at full scale.

**Limitations to carry into the write-up.** (i) lgd_mc's best is at the grid
edge (w = 4) under both rules, so it may be under-tuned — a bias *against* a
competitor. (ii) Several FR3a strengths sit within ~1 se of the floor at
n = 512 (plug/alpha 0.369, tmpd/alpha 0.371, lgd_mc 0.371–0.375); at full
scale some may land just under it, which the table will report as measured.
(iii) Choosing on q90 is conservative for `dist`, whose steering demand is
smaller, so every guided arm may be somewhat under-guided there.

**Measured Betty cost** (compare-stage logs, not an estimate), minutes per
cell at n = 5,000 on B200 MIG 1g.45gb / 2g.45gb: unguided 4.9/4.1, plug
11.6/9.9, tmpd 18.2/15.8, lgd_mc 20.5/17.7, **dflow 104.5/88.9**, btvg
31.1/27.0, btvg_var 27.0/23.6. One (property, seed) is ≈ 218 min on a 1g
slice — too close to the 225-min guard — so each is split into two arm groups.
All 21 tasks ≈ 36 GPU-h; wall ≈ 4 h with ≥ 9 slices, ≈ 12 h with 3.

**dflow is dropped from the full run** (Henry, 23 Sep, before any full-run
cell): it will be replaced by another trajectory-optimisation competitor. This
overrides FR2 for dflow only. Its code, gates and 42 compare-stage cells are
kept and remain reportable at n = 512. Without it one (property, seed) is
≈ 113 min on a 1g slice; the run is ≈ 20 GPU-h, 64 cells.

**Implementation.** `proj1/cluster/full_run.slurm` (12-task array: 0–8 the
primary set, one per property × seed; 9–11 the secondary set; plus two
chained insurance sweepers) →
`guidance_sweep.py --stage full`. At job start it re-applies FR1 on q50 (stop
on fail), FR1 on q90 (report), and freezes the q90 strengths **once for the
whole run** (`results/full/n5000/frozen_q90.json`), reused by every task and
link. `check_fullrun_go.py` now refuses a compare grid missing **any** strength
(the previous version accepted one cell per arm, which would have frozen a
best-of-partial strength), refuses mixed n or seeds, excludes NaN-MAE cells
from "best", breaks MAE ties toward the smaller w, and exits **10** on FR1
FAIL — never 1, which is also Python's crash code. Each cell writes a
per-molecule sidecar (`*.permol.pt`). Read the result with
`proj1/scripts/full_run_table.py`.

### FULL-RUN RESULT — `dist`, n = 5,000 × 3 seeds (run 23 Sep 02:40–07:10, read 23 Sep)

**Integrity, all verified locally.** 64/64 cells (54 primary + 10
secondary) and 64 sidecars; 0 failed, 0 non-finite; all on B200 MIG 2g.45gb;
every setting identical to the compare stage; generator md5 `a190ac83…` =
the compare stage's; one frozen file, byte-identical to a local re-computation;
sidecars hold exactly `test[:5000]` with their real sizes and targets
(corr(size, α) 0.755); every reported in_band / stability / MAE reproduced
from the sidecars. Both sweepers found nothing left. Local table == Betty's.
Regenerate: `python proj1/scripts/full_run_table.py` and
`python proj1/scripts/dist_report.py --dir "results/full/n5000/seed*" --frozen results/full/n5000/frozen_q90.json`.

**Headline (FR5, pre-registered: in_band, independent z ≥ 3, floor).** Every
arm clears the chemistry floor at its FR3a strength.

| in_band (gain vs unguided) | mu | alpha | gap |
|---|---|---|---|
| unguided | 0.076 | 0.058 | 0.114 |
| plug | +0.004 | +0.004 | +0.011 |
| tmpd | +0.007 | +0.006 | +0.012 |
| **lgd_mc** (w = 4) | **+0.029** | **+0.022** | **+0.038** |
| btvg | +0.006 | +0.006 | +0.002 |
| btvg_var | +0.002 | +0.000 | +0.001 |

- **`lgd_mc` beats btvg on all three properties** (in_band 6.8 / 5.3 / 9.2 σ,
  and on MAE). **BTVG does not beat prior art under the pre-registered test.**
  It survives Bonferroni over the 15 FR5 tests. Scope of the σ: it is
  conditional on these 5,000 targets (the seeds share them). Counting the
  targets as the only independent draws inflates the se by up to √3, which
  takes alpha to ≈ 3.0 σ; mu and gap stay above 3.
- **btvg vs unguided on in_band: a tie** (1.9 / 2.3 / 0.6 σ). On MAE it is
  better than unguided on mu and alpha.
- **Other MAE results, all against btvg, all "tie" under FR5 because in_band
  decides:** on gap, plug and tmpd beat btvg on MAE (σ_ind +6.8 / +8.6) and
  are ahead on paired in_band (−3.3 / −3.6 σ_pair; −2.3 / −2.6 σ_ind). On mu,
  tmpd beats btvg on MAE (σ_ind +5.7). btvg's MAE wins (σ_ind ≤ −3) at FR3a
  strengths: over unguided and btvg_var on mu and alpha, and over plug and
  tmpd on alpha only.
- **Ablation (plug = the mean term, btvg = mean + variance).** In the primary
  set, the only matched strength is mu at w = 0.05: dMAE −0.012 D, paired CI
  [−0.024, +0.001] (independent [−0.039, +0.016]), in_band +0.002, so nothing.
  btvg_var alone ≈ unguided. In the secondary set all three properties are
  matched at w = 4 (seed 1, paired), and **the variance term's sign flips by
  property**: alpha helps (MAE −0.50 δ, z −6.0); mu hurts (MAE +0.38 δ,
  z +5.3); gap hurts (in_band −0.039, z −5.7; MAE z +12.2). On mu and gap
  both arms fail the floor at w = 4, so this is mechanism evidence, not an
  FR5 verdict.

**The strength confound, which must travel with the headline.** FR3a forced
plug/tmpd/btvg to weak strengths (their q90 chemistry breaks above them) while
lgd_mc kept its chemistry at w = 4 — so the headline partly measures *which
arm was allowed to guide hard*. On `dist` at w = 4, **alpha** is the one
property where all four clear the floor, and then only against seed 1's own
unguided floor (0.9 × 0.394 = 0.355, same noise seeds). Against the frozen
compare floor (0.362) or the pooled full-run floor (0.361), plug (0.358) and
btvg (0.355) fail. btvg clears seed 1's floor by 2 molecules (1777 stable vs
1774.8). Secondary set, seed 1, paired, supplementary:

| alpha, w = 4 | in_band | MAE/δ | mol_stab |
|---|---|---|---|
| plug | 0.078 | 8.16 | 0.358 |
| tmpd | 0.080 | 8.43 | 0.376 |
| lgd_mc | 0.080 | 8.41 | 0.383 |
| **btvg** | **0.083** | **7.65** | 0.355 (exactly at the floor) |

btvg's alpha MAE is better than every competitor at equal strength (paired
z −6.0 / −9.6 / −7.9), in_band a tie (≤ 1.0 σ). **That is BTVG's one positive
signal: alpha MAE at w = 4.** It is **partly replicated only**. It matches the
q50 screen (8.70 δ vs plug 9.31, tmpd 9.12). On the q90 compare cells, the
target the freeze used, btvg at w = 4 does not win (13.44 δ vs plug 13.33).
The full-scale result is one seed, and it sits on the chemistry floor. It does
**not** extend to mu or gap: on gap at w = 4, plug is ahead of btvg (in_band
−0.039, z −5.7), but both fail the floor there (0.258 / 0.267), so under FR5
neither wins. On mu at w = 4, plug, tmpd and btvg all fail.

**Against the literature (reference ladder, `dist_report.py`).** Our protocol
reproduces EDM's own reference rungs: U-bound 1.63 D / 8.95 / 1482 meV vs EDM
1.616 / 9.01 / 1470; #Atoms 1.04 D / 3.93 / 868 meV vs EDM 1.053 / 3.86 / 866.
**No guided arm beats the #Atoms baseline**, EDM's test for using property
information beyond molecule size (best: lgd_mc 1.088 D / 4.03 / 912 meV).
That bar is stricter than it looks for a sampler. #Atoms is a point guess
(the median at that size), so a generator that ignores the target also pays
for its own spread (≈ √2 for a Gaussian). The fair "ignores the target"
reference is size-shuffle (`dist_metrics.py`), and unguided sits on it.
Against it, gap closure (0 = ignores the target, 1 = the real molecule) is:

| arm | mu | alpha | gap |
|---|---|---|---|
| lgd_mc | 0.28 | 0.29 | 0.26 |
| tmpd | 0.08 | 0.12 | 0.10 |
| plug | 0.01 | 0.09 | 0.08 |
| btvg | 0.02 | 0.18 | 0.02 |
| unguided | −0.03 | −0.02 | −0.01 |

So guidance does steer within size. It just does not clear EDM's published
bar, which EDM's own retrained model also misses on mu (1.111 vs 1.053).
Unguided mu (1.51 D) sits near our size-shuffle rung (1.48 D), below the
U-bound (1.63), because `dist` already gives it the right sizes (alpha:
unguided 5.69 vs size-shuffle 5.61, U-bound 8.95). For scale, EDM's *trained* conditional
model is 1.111 D / 2.76 / 655 meV and EEGSDE's best 0.777 / 2.5 / 542 — our
evaluator's own L-bound (0.086 D) is ~2× EDM's, so cross-paper MAEs are
indicative only.

**What is not in this run.** `dflow` (dropped 23 Sep by decision) was the
strongest arm under the rubric at the screen; its n = 512 result must be
reported, or its replacement run under the same protocol.

### AMENDMENT FR2a — `tfg` replaces `dflow` (23 Sep, AFTER the full run was read: POST HOC)

**Decision (Henry, 23 Sep).** The compare set's third recent external method
is now **TFG** (Ye et al., NeurIPS 2024), replacing dflow. The assignment asks
for at least three recent external methods under the same protocol
(`proj1_tex/Project1_Paper_Instructions.tex:321,394`): `tmpd`, `lgd_mc`,
`tfg`. `plug` is btvg's own mean term, an ablation rung, not one of the three.
dflow's 42 compare cells stay on disk and are reported at n = 512.

**Why TFG, and what it does not fix.** Cheap (≈ plug + 4 guide passes per
guided step), runs on both backends, and has published QM9 numbers on the
transfer's own EDMsecond stack. It is NOT a trajectory-optimisation method:
with dflow gone, every arm forms a one-step posterior mean. That is a
limitation to state, not a requirement. OC-Flow, the only real candidate for
that slot, costs about what D-Flow does (its own Table 10: 103.7 vs 102.8 s).

**It is not `tfg_mc`.** `tfg_mc` is TFG's MC-smoothing ingredient alone, at
settings TFG does not use on QM9. TFG's own QM9 search (App. E.3, Table 11)
turned the smoothing almost off and the clean-space mean guidance ON for all
six properties. `tfg` = one gradient through the denoiser (TFG's "variance
guidance" -- which, with smoothing off, is the plug direction; gated) plus
4 clean-space gradient steps on the predicted molecule ("mean guidance"),
applied as TFG applies them: an O(1) shift per step, not a velocity edit.
**Collapse gate, fixed before any real-model tfg cell ran:** the
diagnostic `tfg_d0_frac` = |mean-guidance displacement| / (|variance
displacement| + |mean-guidance displacement|), pre-clip, averaged over
guided steps, must be **>= 0.10** in the frozen full-run cell of a property.
Below that, tfg is reported on that property as mechanistically `plug` (a
rescaled plug-in step), not as a separate method.

**Configuration, fixed before any tfg cell ran** (`guidance_sweep.TFG_QM9`):
TFG's published (rho, mu, gamma) per property -- alpha (0.016, 0.001, 1e-4),
mu (0.001, 0.002, 0.1), gap (0.032, 0.001, 0.001) -- N_recur 1, N_iter 4,
one MC sample, rho/mu schedule "increase" (paper section 5.1; the public
script's mu "decrease" would put ~90% of mu in the steps our window
switches off), sigma "decrease", rescale_grad clip 100, TFG's energy
-((f - y)/MAD)^2 with the MAD of f_A's own training split. The strength `w`
multiplies (rho, mu): **w = 1 is TFG's published configuration** and the
shared 7-point grid brackets it. Shared with every arm: the t_min = 0.5
window and the velocity-relative clip (so this is "clipped TFG"; the clip's
bite is logged as `tfg_corr_over_v` and `clipped_sample_steps`).

**Port, not reproduction.** Our samplers are deterministic ODEs, where TFG
used DDIM with eta = 1. The flow model gets TFG's VP update through the exact
rescaling x_vp = x / sqrt(t^2 + (1-t)^2), verified to 4e-16 against TFG's
formula (`proj1/tests/test_tfg.py`, 62 gates). The flow model's one-hot
features are unscaled where EDMsecond divides them by 4, so the clean-space
step moves features about 16x less relative to coordinates than on EDM: that
is each model's native space, as for every other ported arm. **Never quote
a tfg number against TFG's published table.**

**Protocol.** The main protocol for one arm, via `proj1/cluster/tfg_run.slurm`.
Compare stage: 42 cells, identical settings to the other 216 (seed 20260921,
n 512, generator md5 a190ac83). Freeze: FR3a on its own q90 cells, written to
`results/full/n5000/frozen_q90_tfg.json` with `--must-match frozen_q90.json`
(every shared arm's strength and the floor must come out identical, or
nothing is written). The registered FR1 (with dflow) PASSED and stands; FR1
with tfg is printed as a post-hoc note and never stops the job. Full run:
the same three seeds and the same 5000 test molecules, so every tfg
comparison is paired. Read with
`python proj1/scripts/full_run_table.py --frozen results/full/n5000/frozen_q90_tfg.json`.

### BTVG-2: the post-full-run revision (EXPLORATORY PILOT, 23 Sep, not pre-registered)

**Why.** The full run's binding constraint was chemistry per unit of push:
lgd_mc's smoothed estimator is the only one that held the floor at w = 4.
Diagnosed from the q90 compare cells and the dist w = 4 cells, old BTVG's
variance term did four things. Items (b) and (c) compare `btvg` with `plug` at
the same w. After the (τ/s)² normalisation, btvg's mean term IS plug exactly,
so the difference between them is the variance term.
- (a) It dragged the property mean away from far targets: in the q90
  `btvg_var` cells, |bias| goes 10.7 → 12.7 δ on mu and 20.6 → 25.4 on alpha
  as w goes 0.01 → 4.
- (b) It widened the dist error spread at w = 4: 7.18 vs 6.73 δ on mu, and
  5.66 vs 4.84 on gap.
- (c) It was clipped more at w = 4 on q90: 32 % / 28 % of steps on mu / gap,
  against plug's 9 % / 4 %.
- (d) V_F ≤ 0 on about 5 % of steps. Its alpha "win" is mostly bias cancellation:
plug overshoots (+2.17 δ), and the drag offsets it (+1.21).

**What.** `btvg2` (`guidance.py::btvg2_weighted_grad`) takes lgd_mc's mean
term unchanged and adds BTVG's forward-KL variance term, both read off the
same K = 4 smoothed draws, so `btvg2 − lgd_mc` isolates the variance term.
The variance step is:
- gated by exp(−(y−μ̂)²/2V̂);
- made orthogonal to ∇μ̂;
- capped at max(|y−μ̂|, √V̂)·|ḡ|/s²;
- never allowed to widen V̂.

`btvg2_band` gates by the band instead (width δ). Its cost counters
(generator and guide passes) are identical to lgd_mc's, and in the pilot its
wall time matched too (736 vs 737 s per cell, both under the same GPU load).
The gates are `proj1/tests/test_btvg2.py` (10, ALL PASS). An independent review
ran twice. It found that a first cap at |M| crushed the variance step 17× on
target; that was fixed and re-verified. All 11 mutations it injected were caught.

**Pilot.** Run on the local RTX 5080 (sampling only), on `dist`, test[:2048],
seed 20261001. The 5080 reproduces Betty's cells: unguided max |Δf_B| 2.4e-4;
guided median 5e-6, with about 1 % of molecules diverging chaotically.
lgd_mc was re-run locally, so every comparison is paired on the same targets
and the same noise. **Exploratory only:** test[:2048] overlaps the scored
block. It is one seed at n = 2048, and two variants were tried on it. Any
confirmation must use fresh targets. Old btvg rows are Betty's, near-paired.
The reproduction check cells are in the scratchpad's `pilot_repro/`.

| mu | w | in_band | MAE/δ | mol_stab |
|---|---|---|---|---|
| lgd_mc | 4 / 8 | 0.102 / 0.124 | 6.39 / 5.69 | 0.400 / 0.369 |
| btvg2 | 4 / 8 | 0.101 / 0.112 | 6.36 / 5.56 | 0.390 / 0.373 |
| btvg2_band | 8 | 0.119 | 5.70 | 0.370 |
| old btvg | 4 | 0.119 | 5.56 | 0.277 ✗ |

| alpha | w | in_band | MAE/δ | mol_stab |
|---|---|---|---|---|
| lgd_mc | 4 / 8 | 0.084 / 0.102 | 8.48 / 7.54 | 0.376 / 0.381 |
| btvg2 | 4 / 8 | 0.074 / 0.088 | 8.49 / 7.38 | 0.375 / 0.379 |
| btvg2_band | 8 | 0.097 | 7.53 | 0.376 |
| old btvg | 4 | 0.075 | 7.83 | 0.345 ✗ |

| gap | w | in_band | MAE/δ | mol_stab |
|---|---|---|---|---|
| lgd_mc | 4 / 8 | 0.161 / 0.186 | 4.41 / 3.87 | 0.371 / 0.339 ✗ |
| btvg2 | 4 / 8 | 0.163 / 0.175 | 4.47 / 3.97 | 0.362 / 0.353 |
| btvg2_band | 8 | 0.173 | 3.96 | 0.342 ✗ |
| old btvg | 4 | 0.142 | 4.52 | 0.255 ✗ |

The floor on this subset is 0.351 (✗ = below it).

1. **The estimator fixes BTVG's chemistry.** btvg2 minus old btvg at w = 4:
   +0.113 (z +8.1) on mu, +0.031 (z +2.3) on alpha, +0.107 (z +7.8) on gap. Old
   btvg fails the floor on all three; btvg2 passes on all three at w = 4 and 8.
2. **The variance term adds no in_band on top of it.** btvg2 − lgd_mc on
   in_band over 6 (property, w) pairs: no pair gains, the largest |z| is 1.72,
   and all three w = 8 pairs lose a little. btvg2_band over 3 pairs gives the
   same picture. MAE and chemistry are n.s. **The pairs are not independent.**
   All six use the same 2048 molecules and the same noise. The Stouffer
   combined z (−2.3 for each variant) is therefore fragile: the w = 4 pairs
   alone give −0.86 and the w = 8 pairs alone −2.42. The robust reading is "no
   gain anywhere", not "a significant loss". On mu and alpha the loss shrinks as the term's share shrinks
   (btvg2: share 0.20–0.27, −0.011 to −0.014 at w = 8; band: share 0.04–0.05,
   −0.005). On gap even the band variant loses −0.013.
3. **At equal chemistry, btvg2 is on or just below lgd_mc's frontier.**
   Interpolated between w = 4 and 8: mu 0.112 vs 0.120 at 0.373; gap 0.175
   vs 0.175 at 0.353. This is a line through two points per arm, so its
   linearity is untested. Alpha's chemistry is flat in w, so there is no chemistry
   axis to match on (at w = 8, 0.088 vs 0.102).
4. **The registered grid capped the winner too.** lgd_mc at w = 8 is still
   above the floor on mu (0.369) and alpha (0.381), and its in_band rises
   0.102 → 0.124 and 0.084 → 0.102. On gap it falls below at w = 8.

**Consequence.** BTVG's own component, variance targeting, has now been
tested three ways (analytic; MC plausibility-gated; MC band-gated). It has
never raised in_band. What the pilot supports is: the estimator is the lever;
btvg2 reaches about parity with the strongest baseline; variance targeting is
a measured negative.

### PROPOSED: the equal-chemistry-cost comparison (not yet run)

This removes the strength confound. It does not choose strengths on q90 (a
reach problem) and apply them to `dist` (a spread problem). It does not
cap the grid at 4.
- **Tune** on test[10000:12000] (n = 2000, seed 20261001; test has 13,083), with the arms
  unguided, plug, tmpd, lgd_mc, btvg and btvg2, and w ∈ {0.25, 0.5, 1, 2, 4, 8,
  16}. Freeze each arm's w by FR3a on `dist`: best MAE among strengths with
  mol_stab ≥ 0.9 × unguided, measured on the tune block.
- **Confirm** on test[5000:10000] × seeds 20261004–6 (n = 5000). Apply FR5
  (in_band, independent z ≥ 3, both arms above the floor). The main figure is
  the tune-block frontier: in_band against mol_stab across w, per arm.
- **Cost** (Betty 1g minutes per n = 5000 cell; btvg2 ≈ lgd_mc ≈ 20.5):
  - tune ≈ 14 GPU-h: 21 tasks of ~41 min, i.e. 2 waves on 12 slices, ≈ 1.4 h;
  - confirm ≈ 16 GPU-h: 9 tasks of ~107 min, i.e. 1 wave, ≈ 1.8 h;
  - so about 3–4 h of wall time in two phases, plus queueing and the freeze
    step in between.
- **Code still needed:** a freeze script over the tune cells, and
  `load_frozen` accepting that source (it asserts compare/q90 today).
  `--dist-offset`, `--strengths` and `--only-w` exist already.
- **Prediction, written before it runs:** lgd_mc ≥ btvg2 > plug, tmpd, old
  btvg. It fixes the evaluation; on the pilot's evidence it will not make BTVG
  win.

### What this means

1. **BTVG does not beat prior art at full scale** (see FULL-RUN RESULT
   above): under the pre-registered test `lgd_mc` beats it on all three
   properties, and on in_band it ties unguided. Its one positive signal is
   **alpha MAE at w = 4**: best of any arm at equal strength (one seed), as it
   was on the q50 screen (8.70 δ vs `plug` 9.31) but not on q90 (13.44 vs
   13.33). It clears the chemistry floor by 2 molecules. The variance term's
   effect at w = 4 flips sign by property (helps alpha, hurts mu and gap).
   *(Screen-era wording, superseded: "the contribution that survived … tied
   for first on all three properties".)*
2. **SHG has not demonstrated an incremental win.** Its early mu screen had
   the highest observed coverage, but the controlled comparison above calls
   SHG a negative result. The early maximum is not evidence of superiority.
3. **SPBC is a clean negative result.** Built to correct a bias measured at up
   to 13.99 δ; moves nothing (0.2/1.3/0.1σ). Its best alpha cell is
   bit-identical to no guidance.
4. **Haimo v1 fails its own deflation test**, and worse than predicted — it is
   *harmful* on alpha, not merely inert.
5. **The window dominates.** Same arms, same data: `plug` is −8.2σ on alpha at
   `t_min=0.05` and +4.5σ at 0.5, and the SMG family overtakes it. **"When you
   guide matters more than how" is the best-supported claim in the project** —
   it replicates across both windows, all three properties and both seeds.

---

## 11. Known defects and corrections — carry these into the write-up

Things measured wrong at some point and since fixed. They belong in the
methods section, not hidden.

| defect | consequence | status |
|---|---|---|
| Bond table missing C≡O | inflated generated stability by 0.2–1.1 points; invisible on real QM9 | ✅ fixed, brute-force verified against EDM over 37,500 combinations |
| `tmpd` silently aliased to `plug` | **every earlier "tmpd" number was `plug`.** The `tmpd` row in §8 is the first legitimate one | ✅ fixed |
| MC arms drew from Σ² not Σ | wrong noise scale in `tfg_mc`/`lgd_mc`/`osc` | ✅ fixed |
| `Posterior.k` was a scalar from `t[0]` | wrong whenever t varied in a batch; invalidated the RCH fit | ✅ fixed, RCH refitted |
| `grad_property_variance` missing its Hessian term | BTVG's variance half would have been a **silent no-op** | ✅ fixed, gated |
| `V_F` clamped at 1e-12 when negative | produced a **+5e11** widening coefficient exactly where the model had broken down | ✅ fixed, gated |
| BTVG widened over the last ~15% of every trajectory | `V_F → 0` as `t → 1` by construction, so the sign always flipped | ✅ coefficient clamped ≤ 0 |
| SHG ignored the strength knob | 5 identical cells per arm | ✅ fixed |
| `rch` crashed on every cell (shadowed variable) | Haimo v1 unmeasurable | ✅ fixed |
| No `unguided` baseline existed | `select_arms.py` **could not drop any arm**, silently | ✅ baselines now run; script refuses without one |
| Strength grid split `{0.01,0.05,0.25,1,4}` vs `{0.25,0.5,1,2,4}` | 42 orphaned cells; `plug`/`tfg_mc` got best-of-7 draws against others' best-of-5 — biasing exactly the reported winners | ✅ unioned to 7 points; 0 orphans |

**Verified sound:** local (RTX 5080) and cluster (B200) cells are **genuinely
paired** — three cluster cells re-run locally matched to 0.15 se on MAE and
*exactly* on in-band, stability and SMILES. Pooling the two is safe.

**One cell to exclude by hand:** `alpha__smg__q50__w2__tmin0.05.json` —
MAE 4.36e7, 47/512 non-finite. `select_arms` already filters it.
