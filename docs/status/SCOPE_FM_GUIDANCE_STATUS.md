# Scope and status — FM base model + inference-time guidance

**Written for:** the project team (Bobo, Idea, Haimo), as the single reference for
what exists, what it measured, and what is still missing. Everything here is
restricted to **Modality 1 (QM9) with the flow-matching generator**. The
diffusion generator and Modality 2 (DNA simplex) are explicitly out of scope of
this document and are listed as not-started in §10.

Status as of **2026-09-21**. Submission **29 Sep 08:30**, defence **30 Sep**.

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

### What this means

1. **BTVG is the contribution that survived.** Best MAE on alpha of any arm
   (8.70 δ vs `plug` 9.31, `tmpd` 9.12), and statistically tied for first on
   all three properties. It does **not** clearly beat prior art — the alpha
   edge is ~0.8σ.
2. **SHG works.** `shg_plug_btvg` has the highest band coverage on mu of any
   arm measured.
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
