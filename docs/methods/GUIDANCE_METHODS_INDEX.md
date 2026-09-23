# Guidance methods — the complete index

**Written for:** the project team (Bobo, Idea, Haimo). This is the single place
that lists **every** guidance method this project has proposed, what class it is
(**ours** / **prior art** / **ablation**), whether it is implemented, whether it
can actually produce a number, and **which document contains its proof**. Cite
this file's row when you need to look a method up.

Last verified against the code **2026-09-21** by a full docs-vs-implementation
audit. Paths are relative to the repository root.

**23 Sep outcome correction:** the section-2 “zero measurements” inventory below
is historical. BTVG has now lost to LGD-MC in the registered full run; BTVG2's
pilot does not establish added benefit from its variance term. Its local
mean-preservation argument also fails after pullback. Read
[BTVG failure audit and redesign](BTVG_FAILURE_AUDIT_AND_REDESIGN.md) and the
latest [living status](../status/SCOPE_FM_GUIDANCE_STATUS.md) before interpreting
“OURS/new” as a supported novelty or performance claim.

## How to read the columns

| column | meaning |
|---|---|
| **class** | **OURS** = candidate for the innovation the project must contribute · **PRIOR** = published method reproduced for comparison · **ABLATION** = a piece of one of ours, not a claim |
| **mode** | the exact `mode=` string in `proj1/src/guidance.py`. `—` means not implemented |
| **cells** | production cells in `results/sweep/`. **0 means it has never produced a number** |
| **proof** | the document with its derivation. Cite this in the paper |

---

## 1. Arms with measured results ✅

These have production cells at n = 512. Results in
[SCOPE_FM_GUIDANCE_STATUS.md §8](../status/SCOPE_FM_GUIDANCE_STATUS.md).

| method | class | mode | cells | proof doc | notes |
|---|---|---|---|---|---|
| unguided | baseline | `f_net=None` | ✅ 5 | — | the control. `gap__unguided__q90` still missing |
| **plug-in / DPS** | **PRIOR** — Chung et al., DPS, ICLR 2023 | `plug` (`dps` alias) | ✅ 23 | [SMG_PRIOR_WORK_AUDIT §2](SMG_PRIOR_WORK_AUDIT.md) | `(y−f(m))/s² · Jᵀg` |
| **TMPD / ΠGDM** | **PRIOR** — Boys et al., TMLR 2024 | `tmpd` → `smg_var` | ✅ 24 | [SMG_PRIOR_WORK_AUDIT §2.2–2.5](SMG_PRIOR_WORK_AUDIT.md) | adds the `s²+gᵀΣg` denominator |
| **TFG** | **PRIOR** — Ye et al., NeurIPS 2024 | `tfg_mc` | ✅ 23 | [SMG_PRIOR_WORK_AUDIT §2.55](SMG_PRIOR_WORK_AUDIT.md) | K perturbations at a **tuned** σ |
| **LGD-MC** | **PRIOR** — Song et al., ICML 2023 eq. 9 | `lgd_mc` | ✅ 21 | [QM9_INFERENCE_TIME_GUIDANCE_BASELINES.md](../results/QM9_INFERENCE_TIME_GUIDANCE_BASELINES.md) | scale read off the model, not tuned |
| **OSC** | ⚠️ **labelled PRIOR, NO CITATION EXISTS** | `osc` | ✅ 21 | ⚠️ **none** | see the warning below — **this arm wins on `alpha`** |
| **SMG** | **OURS** (primary) | `smg` | ✅ 19 | [SMG_FULL_MATH_AND_PROOFS §3.3](SMG_FULL_MATH_AND_PROOFS.md), [SMG_WHY_INNOVATIVE](SMG_WHY_INNOVATIVE.md) | `(y−f(m)−c)/(s²+v_f)·Jᵀg`, `c = ½tr(HΣ)` |
| **SMG-2** | **OURS**, self-demoted | `smg2` | ✅ 19 | TOP6 §2 (in [INNOVATION_IDEAS_INDEX](INNOVATION_IDEAS_INDEX.md)) | its own doc: *"not novel… a correction to your own prior work"*. Use as a **control** |
| **SMG-2 curvature** | **OURS** | `smg2_curv` | ✅ 19 | [GUIDANCE_EXPERIMENT_PLAN](../protocol/GUIDANCE_EXPERIMENT_PLAN.md) | adds the Stein direction `((r²−S)/S²)ΣHΣg` |

> ### ⚠️ `osc` needs resolving before the paper
> `osc` is tagged **COMPARE** (prior art) in the experiment plan and in the
> status doc, but **no citation for it exists in any document in this repo**. It
> is defined only in code. It is also the **best arm on `alpha`** (+3.33σ).
> Claiming "published prior art beats our method" with no paper behind the
> label is the worst way round to be wrong. Either find the citation or relabel
> it **OURS** — and if it is ours, it is currently our strongest result.

---

## 2. Arms implemented and verified, with ZERO measurements ⏳

All gated and preflighted; none has produced a production cell. **This is where
the project's untested value sits.**

| method | class | mode | proof doc | what it attacks |
|---|---|---|---|---|
| **SPBC** | **OURS (new)** | `spbc` | [V2_ARMS_SELECTION_AND_TEST_PLAN §1](V2_ARMS_SELECTION_AND_TEST_PLAN.md), [Three_New §5.4](Three_New_Guidance_Ideas_Variance_and_Switching.md) | **bias.** Gives every trajectory the *same* property increment, `d_i = (ν/r_i)a_i` |
| **BTVG** | **OURS (new)** | `btvg` | [PROPOSALS §3.1–3.2](PROPOSALS_VARIANCE_TARGETED_GUIDANCE.md) | **spread.** Descends `KL(N(μ_F,V_F)‖N(y*,τ²))` instead of maximising a likelihood |
| **SHG** ×4 | **OURS (new)** | *scheduler*, not a mode — `sampling.py::active` | ⚠️ **schedules exist only in code** (`guidance_sweep.py:SHG_SCHEDULES`) | the **window effect**, the largest measured signal in the project |
| BTVG mean-half | ABLATION | `btvg_mean` | [PROPOSALS §3.2](PROPOSALS_VARIANCE_TARGETED_GUIDANCE.md) | ⚠️ **provably identical to `plug`** — see below |
| BTVG var-half | ABLATION | `btvg_var` | [PROPOSALS §3.2](PROPOSALS_VARIANCE_TARGETED_GUIDANCE.md) | the decisive half. *Stop if this is null* |
| SMG mean-only | ABLATION | `smg_mean` | [SMG_FULL §3.3](SMG_FULL_MATH_AND_PROOFS.md) | completes the SMG ladder |
| **band** (Component D) | **OURS** | `band` | [COMPONENTS_ABCD_MATH_AND_PROOFS §3.6](COMPONENTS_ABCD_MATH_AND_PROOFS.md), [D_WHY_INNOVATIVE](D_WHY_INNOVATIVE.md) | ball∩slab QP: largest diversifying edit inside the tolerance band |
| **RCH** (Haimo v1) | **PRIOR, twice published** — Residual ∇-DB (Liu, ICLR 2025 eq. 15); VGG-Flow (NeurIPS 2025 eq. 16) | `rch` | [INNOVATION_IDEAS_INDEX](INNOVATION_IDEAS_INDEX.md) | a **deflation test** for SMG: if a ridge lookup reproduces `c`, SMG's per-state JVP/HVP is not earning its cost. *Run, never claim* |
| **κ₃ skew probe** | **OURS** (the *use* is ours; the identity is standard) | `kappa3_skew` + `--kappa3` | TOP6 §1 (rank **#1**) | measures whether the Gaussian closure every SMG arm assumes is valid. **The programme's stated go/no-go** |

> **BTVG's mean half is exactly `plug`.** After the `(τ/s)²` strength
> normalisation, `btvg_mean`'s coefficient is `w(y−f)/s²`, which is `plug` at
> strength `w` — verified to 1.4e-17. This is not a bug; it means **BTVG's
> entire novelty lives in the variance term.** Say that in the paper rather
> than presenting `btvg_mean` as an independent arm.

---

## 3. Designed, deliberately NOT implemented ❌

Recorded so nobody re-derives them. Each row's verdict is from the design doc
itself or from measured evidence.

| method | class | proof doc | why not |
|---|---|---|---|
| **escalate** | OURS, contribution candidate | TOP6 §4 | Evidence is strong (the region where the closure is valid is where guidance has no effect) but costed at **1460 ms/mol**, ≈2.8× over budget. Its own precondition (κ₃) has never been measured |
| **QPMG** | OURS | TOP6 §3 | Its own prototype wins 3000× on a *Gaussian* prior and **loses to free SMG-2 on a mixture**. Real posteriors are not Gaussian |
| **DBFG** | OURS | TOP6 §5 | Gated on a soft-vs-decoded measurement (~15 min) that was never run |
| **MAG** | OURS (theorem) | TOP6 §6 | *"Not an arm."* Keep the counterexample as a figure |
| **SCG** | OURS | TOP6 §7 | *"My own idea, killed by its own decisive test"* |
| **RATV** | OURS, narrow | [Three_New §5](Three_New_Guidance_Ideas_Variance_and_Switching.md) | SPBC is its degenerate case; the doc concedes the reduction *"cannot be sold as a new mechanism"* |
| **OSHG** | OURS, narrow | [Three_New §6](Three_New_Guidance_Ideas_Variance_and_Switching.md) | SHG is its fixed-schedule reduction at a fraction of the cost |
| **RBAG** | **PRIOR too close** — HIG | [Three_New §7](Three_New_Guidance_Ideas_Variance_and_Switching.md) | prior art too close to claim. Do not reopen |
| **RWG** | PRIOR (TFG schedules) | [PROPOSALS §4](PROPOSALS_VARIANCE_TARGETED_GUIDANCE.md) | ~20 lines and reuses `grad_property_variance`. **The only cheap one left** — reconsider *only* if SHG lands and shows the window matters per-state |
| **BPS** | OURS, narrow | [archive/BPS_BALANCED_PREVIEW_SAMPLING.md](../../archive/BPS_BALANCED_PREVIEW_SAMPLING.md) | *"no tested configuration is both novel and compute-positive"* |
| **PCG** | **PRIOR on all three legs** | [archive/PCG_BRAINSTORM_DROPPED.md](../../archive/PCG_BRAINSTORM_DROPPED.md) | its own source document declines to claim it |
| **TRE** | OURS, thin | [archive/D_AND_TRE_REVIEW_FEEDBACK.md](../../archive/D_AND_TRE_REVIEW_FEEDBACK.md) | ~300 lines, gated on a failure never shown to occur |
| Component A (trust region) | **PRIOR** — OSCAR, ICML 2026 | [COMPONENTS §2.2](COMPONENTS_ABCD_MATH_AND_PROOFS.md) | already shipped as `clip` in the sampler. Nothing to test standalone |
| Component B (level-set projection) | **PRIOR ×4** (O-SVGD, PCGrad, …) | [COMPONENTS §2.3](COMPONENTS_ABCD_MATH_AND_PROOFS.md) | mechanism is prior work; and `band` already does the slab projection |
| Component C (endpoint gate) | OURS, `[DESIGN]` | [COMPONENTS §2.4](COMPONENTS_ABCD_MATH_AND_PROOFS.md) | ablation row only, never claimed |
| `G_full`, `v_quad`, 3rd/4th moments | OURS | [SMG_WHY §6.2, §8](SMG_WHY_INNOVATIVE.md) | *"need not be smaller in covariance order than the retained correction"*; expensive and sensitive to network derivative error |
| CPGO / GAPL | OURS — **GAPL is the strongest unclaimed idea in the corpus** | [INNOVATION_V8](INNOVATION_V8_TRAINING_TIME_GUIDANCE.md) Prop. 1–5 | training-time, out of scope for this deadline. **Record as future work with the Prop. 5 bound** |
| Centred-residual "orthogonal guidance" | **PRIOR** — Tilt Matching eq. 24 | [INNOVATION_V8](INNOVATION_V8_TRAINING_TIME_GUIDANCE.md) | *"renaming it would not create a new algorithm"* |
| **Exact simplex marginalisation** | **OURS, un-indexed** | [SMG_FULL §5](SMG_FULL_MATH_AND_PROOFS.md) | **Flagged: the only *exact* method in the corpus** — no Tweedie, no Jacobian, no Hutchinson — ~60 lines, and it has no ID in any ranking table. Needs **Modality 2**, which has not started |

---

## 4. External baselines the grading may require ⚠️

[QM9_INFERENCE_TIME_GUIDANCE_BASELINES.md](../results/QM9_INFERENCE_TIME_GUIDANCE_BASELINES.md)
states the assignment needs **≥3 relevant methods published within 5 years**.
We reproduce `plug`/`tmpd`/`tfg_mc`/`lgd_mc`/`osc` **locally**, on our own
generator. Whether local re-implementations satisfy that requirement is argued
**nowhere in this repo.**

| method | status |
|---|---|
| D-Flow, OC-Flow, FlowGrad, full TFG, FreeDoM, real DPS | ❌ none implemented |

**This is a grading risk, not a research one.** `FlowGrad` is the cheapest —
it reuses the existing VJP path. Also: `proj1_tex/citation.bib` has 28 entries
and contains **none** of TFG, DPS, TMPD, ΠGDM, LGD, EEGSDE, OC-Flow, D-Flow or
FlowGrad. Every method we compare against is currently uncited.

---

## 5. Corrections to statements in the older method docs

The design docs were written before the measurements. Do **not** quote these
without checking here first.

| doc claim | reality |
|---|---|
| SPBC "leaves every centred quantile and pairwise difference unchanged" | **False end-to-end**: 8.03 δ drift in centred values, 12.3 δ in pairwise differences. The defensible claim is **~3× less shape distortion per unit mean shift** than a shared multiplier |
| SPBC `ν = −η(mean f − y*)` centres the batch | It is applied as a **rate**, so realised closure is ~**50 %** at η = w = 1, not 100 % |
| "alpha/plug 13.99 δ, alpha/smg −5.69 δ" | ⚠️ **three different strengths in one sentence.** At a consistent w = 1 it is plug **+13.99** vs smg **−15.06** — i.e. **SMG is *more* biased than plug**, not less |
| BTVG "stops, and gently re-widens" | It **widens** — measured coefficients +5.3, +110.8, +78.6. Now clamped ≤ 0 in code so it stops instead |
| BTVG τ defaults to δ | τ = **δ/1.96**. A centred Gaussian with sd = δ gives only 68.3 % coverage |
| SMG-2 is "7× better (1.340 → 0.189)" | True **at η = 0 only**; at η = 1 it reverses (SMG 0.163 vs SMG-2 0.178). State the η |
| "the mean-map curvature term is not covered by any gate" | **Closed** — nonlinear-posterior gate added, 1.14e-13 |
| "47 % improvement from the guidance window" | ✅ **Verified**: alpha/plug w=1, MAE 10.02 → 5.32 as `t_min_guide` 0.05 → 0.5. **The largest measured effect in the project** |

---

## 6. The honest summary

1. **Our SMG family is beaten by prior art on all three properties at every
   protocol.** That is measured, not suspected.
2. **The largest effect anyone has measured is a scheduling result** — 47 %
   from moving `t_min_guide` — and it is bigger than every arm difference
   combined. "*When* you guide matters more than *how*" is currently the
   project's strongest defensible claim.
3. **Every arm built to fix why SMG loses has zero measurements.** SPBC, BTVG
   and SHG attack bias, spread and scheduling respectively; none has produced a
   production cell.
4. The one arm that beats everything on the hard property (`osc`) carries a
   prior-art label with no paper behind it.
