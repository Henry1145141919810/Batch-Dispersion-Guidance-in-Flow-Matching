# BDG (Batch-Dispersion Guidance): adversarial review

**Written for:** the team, and the BDG author in particular. It assumes you have
read [BDG_HANDOFF.md](BDG_HANDOFF.md), which is kept unchanged as the author's
record; where they differ, this review takes precedence.

**25 September 2026.** How it was done:
- **Checked five independent ways**, and **re-implemented from the handoff's prose alone** and run on the local GPU.
- **Attacked** by three red-team reviewers.
- **Defended** objection by objection.
- **Ruled on** by an independent judge per objection and a three-judge verdict panel (38 agents in all).
- **Fact-checked** against the evidence by a separate reviewer, which caught several places where a first draft stated an attack as fact after its judge had narrowed it. Those are corrected here.

Every section in §4 carries its judge's ruling. The scripts are in
[`audit/bdg_review/`](../../audit/bdg_review/), the port's cells in
[`results/bdg_port/`](../../results/bdg_port/), and the full agent record in
`audit/bdg_review/workflow_result.json`.

---

## 1. Verdict

| question | judge 1 | judge 2 | judge 3 | answer |
|---|---|---|---|---|
| **Valid?** | 4 | 5 | 4 | **Sound mechanics, overstated interpretation.** The algebra, the gates and the knob all reproduce. Several of the handoff's explanations and two of its data claims do not. |
| **Innovative?** | 4 | 4 | 4 | **Incremental.** Every ingredient is published, and the team's own §5.2 spec has most of it. One narrow composition is unclaimed. |
| **Reasonable?** | 4 | 4 | 3 | **Misaimed as a metric lever.** Spread governs coverage only near a centred target (q50). At the project's q90 headline the batch sits far from the target and bias governs. |
| **Doable?** | 6 | 7 | 6 | **Yes as a scoped secondary result, no as a headline.** The port fast-forwards onto `main`, and the essential fixes need no new runs. |

**Recommendation (unanimous).**
- Do **not** make BDG the §3.5 headline innovation, and put no BDG number in a q90 headline table. The reason is statistical power, not bookkeeping: at q90 the best the spread lever can do is +0.0001 to +0.007 in-band, at or below the z = 3 resolution even at n = 15,000 per arm (+0.0053 to +0.0082).
- Spend no Betty/PARCC time on it.
- Start writing BDG up **only after the headline sections are drafted**, capped at about 4–6 hours.
- Describe it as **plug with a fixed centring gain and a feedback-servoed deviation gain**, never as "variance control" or as "a loop no fixed schedule can reproduce".
- Any §4.4 row needs two controls first (§5, item 12).

**Novelty.** The novelty lens puts the chance of clearing a "slightly novel" course bar at **≈ 0.85**, provided the §3 reduction and the precedents are stated in the paper's own voice. That comes from a bounded search, about eight queries, abstracts only, 25 Sep 2026. The per-objection judge ruled that workshop- or conference-level probabilities are not supported by a search of that depth, so none is given here.

---

## 2. How it was checked

| lens | what it did |
|---|---|
| **math** | Eight numpy/torch scripts. Checked the gradient, the §3 reduction, the closed-loop equilibrium and stability in discrete time, the effect of the J-pullback on sign and mean, estimator noise against batch size, and the design effect of batch coupling. |
| **GPU port** | Implemented BDG from §2's prose into an isolated branch, with no code shared. Gated it, then ran **64 cells** on the local RTX 5080 (`fm_ema.pt`, n = 256, one batch of 256, seed 20260925, mu/alpha/gap × q50/q90). It also built the open-loop control the reduction argument needs. |
| **data** | Tested the §1 premise on the 15,000 real v2 unguided residuals at both targets, recomputed the "1,766 cells / 1.010×" claim from every `results/sweep*` cell, identified BDG's target statistically, and audited the floor. |
| **novelty** | Resolved all four §8 citations, searched for closer prior art, and re-read the team's own specs and code comments. |
| **doability** | Read the assignment rubric and paper template, and costed the port and the open items against 29 Sep 08:30. |

**What could not be verified:** BDG's own Betty cells, code and sidecars (they live
in another member's tree). The handoff's §6 tables are taken on trust, except where
the independent port reproduced them.

---

## 3. What holds

| claim | evidence |
|---|---|
| **Gradient.** `dV_b/dF_i = 2/(B−1)(F_i − F̄)`, and the chain rule to `m_i` | autograd 3.5e-18 and central differences 3.8e-11, including a nonlinear, non-separable f |
| **§3 reduction.** `(y−F_i) − ηe(F_i−F̄) = (1+ηe)(y_eff − F_i)` | exact to 3.0e-15 over 2,000 random batches, and to 6e-7 on the real field in the port |
| **η = 0 ≡ plug** | bit-identical **at field level**: `test_bdg.py` A9 and B-a, max err 0.00e+00, on a real `x_t` on CUDA. **The "and cell level" this row used to claim is withdrawn** — not because it failed, but because it is not testable. Run `plug` three times at one seed and 16 of 43 numeric fields move (scatter atomics, plus the clip's threshold turning 1e-8 into a different clip decision); `validity` moved by one molecule in 64. Against a 3-run envelope `bdg_e0t1` is inside on every field carrying a claim. [V3_REPRO_ENVELOPE.md](../results/V3_REPRO_ENVELOPE.md) |
| **Same cost as plug** | both terms are a scalar times `g_i`, and the cost dict is byte-identical to plug. That is 4,000 guide passes per n = 5,000 cell against 20,000 for lgd_mc and tfg. *Not* the cheapest arm overall: in generator passes plug and BDG use 10,000, lgd_mc 12,000 and tfg 8,000. |
| **One-sided variant asked to widen ≡ plug** | bit-identical in the **field** (`test_bdg.py` A6, B-c: max abs diff 0.000e+00, raw e −0.98422 clamped to +0.00000). The port agent predicted a failure on mu/q90 and was wrong: the clamp is self-consistently inert. Read "on all four cells tested" as *within the run-to-run envelope*, for the same reason as the η = 0 row above — a cell cannot be bit-identical to another cell when it is not bit-identical to a rerun of itself. |
| **The knob moves spread monotonically** | the port reproduces a monotone requested→achieved ladder on 6/6 curves (sd/unguided 0.67–1.22) on a different device, batch, n and seed |
| **Dispersion sign survives the pullback** | the dispersion term's contribution reaches each molecule's property as `c_i·‖J_iᵀg_i‖²`, a squared norm, so its sign is `−sign(e)` through Jᵀ. This is algebra, first order in the step and per-sample Jacobian. In the same toy BTVG-2's `gᵀJJᵀh` flipped sign on 45–57 % of molecules. **It is a correctness property, not an explanation of performance**: BTVG-2's repaired projection (xproj) still returned 0/6. |
| **Bounded on the widening side** | `V_b ≥ 0 ⇒ e ≥ −1 ⇒ w_eff ≥ 1 − η` for any batch. BTVG's coefficient `−½(1/τ² − 1/V_F)` has a pole (guidance.py:1494, measured runaway +5.3 / +110.8 / +78.6). **This, not "V_b does not vanish", is the correct licence for removing the clamp.** w_eff is unbounded above (+10.35 per step measured). |
| **V_b survives where V_F vanishes** | independently supported by [VARIANCE_DECOMPOSITION.md](../results/VARIANCE_DECOMPOSITION.md): BTVG's within-trajectory V_F ends at 0.0–0.2 % of terminal variance, while the across-molecule variance (BDG's V_b) carries it. That doc also records that the across variance concentrates then re-widens on unguided runs, i.e. the flow has its own drift that a setpoint on V_b fights. |
| **Transfers to the simplex (§3.6)** | BDG's step is a per-sample scalar times plug's step, so it inherits plug's tangency. No Σ or inverse is needed, unlike BTVG's singular `diag(p)−ppᵀ`. A simplex toy passes. |
| **The η = 1 "wins" are noise** | the handoff's own warning is right. z = 1.38 (alpha) and 1.04 (gap), Šidák-adjusted over 13 cells p = 0.68 and 0.88. Against the correlated null of 13 cells sharing noise, p = 0.40–0.44 and 0.56–0.61. |

---

## 4. What does not hold, with the judge's ruling

**Notation.**
- **κ** is the per-step contraction coefficient: step size × w × ‖Jᵀg‖² / s².
- **u₀** is V_b/τ² when guidance starts.
- **R*n*** is the per-objection ruling in the §8 ledger.

### 4.1 The loop's equilibrium is `V* = τ²(1 − 1/η)`, not τ²

*Ruling R4: the objection stands in full (0.90).*

Rewrite §3's identity in mean/deviation form:

```
num_i = w·[ (y − F̄)  −  w_eff·(F_i − F̄) ],        w_eff = 1 + η·e
            \______/     \____________________/
            centring       deviation, gain w_eff
```

- The centring gain on the batch mean is the nominal w at every η and τ.
- **Only the deviation gain is servoed.** Plug ties both to w.
- Plug's own term also contracts deviations. So `V_b = τ²` (e = 0) is where the *dispersion* term vanishes, not where the spread stops moving.
- The law's equilibrium is where **w_eff = 0**, i.e. `V* = τ²(1 − 1/η)` (sd = 0.866 τ at η = 4). Three independent reproductions agree to 2e-15 and to 5 decimals, from both directions, for η ∈ 1.5–16.
- **This is the law's asymptote, not a measured undershoot.** In the 50-step guidance window the loop never gets there: run-mean V_b/τ² is 1.27–1.54 at τ_mult 0.5 and 0.54–0.72 at 1.5, against 0.75. Near the setpoint w_eff is 0 ± 0.19 at B = 512 and flips sign almost every step. **The paper must never say the controller "holds its setpoint".**
- The handoff's measured alpha fixed point (0.84–0.87, §9 item 6) is close to √0.75 = 0.866. That is **suggestive only**: mu (0.574–0.678) and gap (0.673) do not match, and the units of that estimate are undefined.

Stability. The handoff's "V_b does not vanish, so the fixed point is stable and the clamp comes off" runs three separate questions together:
- **boundedness** — algebraic, §3 above;
- **trackability** — measured: V_F vanishes by construction, V_b does not;
- **stability** — its own conditions:
  - **η > 1**, so that V = 0 is repelling;
  - an asymptotic gain bound κ < 1/(η−1);
  - a stricter transient bound κ < 2/w_eff(u₀).

Non-vanishing is necessary, but not sufficient. The fitted real gain sits 38–113× below the transient bound, so the loop is stable in practice. It would break at w ≈ 150–450, so gate any cell above w = 4 on κ·w_eff < 2.

**At η ≤ 1 the controller's own field is never repulsive** (w_eff ≥ 0). Any widening there comes from heterogeneous gains with a distant target, or from the flow's drift. The η = 1 cells in §9 item 1, the only ones that beat plug while clearing the floor, are therefore *contractors with a variance-modulated positive weight*, not setpoint controllers. The cells that widen and the cells that "win" are disjoint in η.

> **Writer's suggestion, not from the review panel and untested on molecules.**
> If the setpoint is meant literally, drop plug's own deviation contraction:
> `num_i = w·[(y − F̄) − η·e·(F_i − F̄)]`. In a toy this equilibrates at exactly
> V = τ² from both sides.
>
> It has costs:
> - It is stable in discrete time only for 0 < κη < 1, and diverges from a wide start once κη(u₀ − 1) > 2.
> - It breaks the η = 0 ≡ plug gate: at η = 0 every molecule receives the same common shift, which is SPBC-like.
> - Joint mean/variance control of this kind is already occupied by MGD and Three_New §5.2.

### 4.2 Widening runs the deviation mode at negative gain; it is *not* negative-weight plug

*Ruling R3: rebutted as stated (0.80). The first draft of this review repeated the error.*

"BDG's widening branch is plug at a negative weight" is **false**:
- Plug at a negative w puts the negative weight on the centring term too, and pushes the batch mean *away* from y.
- BDG keeps the centring gain at +w, so `w_eff·(y_eff − F̄) = y − F̄`.
- Measured on the port's paired cells at q90, the widening cells (τ_mult 1.5) move the batch mean *toward* the target relative to unguided: +2.61 / +2.88 / +5.43 δ (gap / mu / alpha), paired z 7.7 / 4.7 / 8.5.

So the handoff's §9 item 2 control (plug at a constant negative w, 1.153× on alpha at mol_stab 0.397) is **a different field**. It is informative about how cheaply spread can be bought, but it is not the same mechanism.

What is true:
- On most widening cells the run-mean deviation gain is negative (−0.035 to −0.84, single steps to −2.64).
- On tightening cells it is +2.1 to +3.7 on average, +10.4 at peak.
- Widening also occurs at a *positive* run-mean w_eff: mu q90 τ_mult 1.25 widens 1.129× at +0.066. Run-means average over sign-flipping steps, and the gains are heterogeneous.
- **The chemistry cost tracks |w_eff|**, not the direction of the knob.

The leak from the deviation mode into the mean is a **covariance drag** through `cov(‖Jᵀg‖², F − F̄)`. It is worth 0.4–1.9 δ over the ladder, so do not call the two gains "decoupled".

### 4.3 "The loop does not reduce to any fixed weight schedule" is refuted as stated

*Ruling R1: partially defended (0.80).*

§3, §7 and §8(d) rest this on freezing w_eff at its **time-average** and running plug at that constant. That control changes three things at once:
- it removes time variation;
- it drops the y_eff shift;
- it removes feedback.

It also could not succeed: the port measured w_eff changing sign up to 34 times in the 50-step window and spanning [−2.64, +10.35]. And for any deterministic sampler, the realised schedule replayed on the *same* noise reproduces the loop exactly (4e-17 in the toy).

The informative control replays the recorded e-schedule **open-loop onto a different seed's noise**, and compares it with the closed loop on that same noise:

| | gap to the closed loop on achieved spread |
|---|---|
| open-loop replay of a schedule from another seed | **0.3–6.6 %** (8/8 configurations) |
| plain plug | 8.4–30.5 % |

- The replay recovers **69.5–96 %** of the closed loop's effect relative to plug.
- Replay and closed loop share the noise and the pipeline is deterministic, so the remaining 0.3–6.6 % is a **real, small effect of feedback**, not sampling noise.
- **What survives:** the schedule largely *transfers* across seeds, and feedback adds a small measurable remainder.
- **Scope:** one seed pair, mu and gap only, τ_mult 0.5 and 1.5, n = 256, local port. A second seed pair is required before this goes in the paper.
- **The cheapest test of whether feedback earns more:** replay a recorded schedule onto a batch whose starting spread or property differs.

### 4.4 The target is never stated, and the §1 premise flips at the project's headline

*Ruling R2: partially defended (0.86). The premise is right at q50 and wrong at q90.*

- **The handoff never says which target BDG ran at.** Its plug w = 4 control matches `results/sweep` at **q50** (χ² = 1.4, p = 0.96) and is excluded at q90 (χ² = 24.7, p ≈ 4e-4).
- It is also consistent with the v1 `dist` run (|z| ≤ 1.11), so **q50 versus `dist` cannot be separated from the control alone.** The author must read `target_name` from a Betty cell.
- It is not the q90 target the v2 headline uses.

The premise "coverage is governed by how tightly the batch clusters" depends on
|bias|/σ. The quoted δ/σ = 0.056–0.169 is correct but is the same at every target, so
it cannot support a premise that flips between them. The table rescores the 15,000
real v2 unguided residuals at both targets. It is an **idealised bound at fixed
centring**, not a prediction of what BDG would do:

| | q50 | q90 (the v2 headline) |
|---|---|---|
| \|bias\| / σ | 0.23 / 0.05 / 0.14 | ≈ 1.2 / 1.2 / 1.6 |
| spread scale that maximises in-band | tighter is better | **1.20 / 1.40 / 1.10 — widening** |
| tightening to BDG's 0.65 | raises in-band | **lowers it** (mu .0355 → .0232, alpha .0247 → .0113, gap .0580 → .0041) |
| the whole spread lever | large | **≤ +0.005 unguided; up to +0.007 on guided arms** |
| removing the bias instead | ≈ 0 | **+0.025 to +0.047** — 11 / 18 / 40× the spread lever |

Values are mu / alpha / gap. Corroborated by `results/btvg3_widening_sim.json`, whose 15 pooled cells were recomputed to 5 decimals.

**So "we can prove neither direction can help" and "widening lowers coverage by
definition" hold at q50 and are false at q90**, where widening slightly *raises*
coverage and centring is what matters.

Two further points:
- **Even at q50, plug strength already collects the contraction reward.** On gap, plug w = 4 scores 0.1602 against BDG e4t0.5's 0.1523.
- **Under `dist`, batch spread is the wrong lever too.** Coverage there is limited by corr(F, y), and the spread lever buys ≤ +0.013 unguided and ~0 at plug w = 4.

### 4.5 "No existing arm here can widen reproducibly" (1.010×) is false

*Ruling R5: partially defended (0.80).*

Recomputed from every fixed-target cell in `results/sweep*`, with sd = √(rmse² − bias²) validated against per-molecule sidecars to 2.5e-6, and re-verified directly for this review:

| arm | seed 20260921 | seed 20260922 | floor | mean shift vs unguided |
|---|---|---|---|---|
| `rch` alpha q50, w = 0.01 | 1.097× | 1.133× | clears on both | 1.3–1.5 δ |
| `rch` alpha q50, w = 0.05 | 1.423× | 1.455× | clears on both | 5.1 δ |
| `dflow` alpha q50, w = 4 | 1.132× | — | clears | 12.2 δ |

- **The 1.010× yardstick is inside noise.** At n = 512, P(ratio > 1.010 | no effect) ≈ 0.4, since the se of an sd ratio is 0.068 / 0.053 / 0.037 for mu / alpha / gap.
- **No uniqueness claim survives.** `rch` at w = 0.01 widens by as much as BDG while moving the mean 1.3–1.5 δ, comparable to BDG's own 0.4–1.9 δ drag. Claim nothing about BDG being the only arm that widens with centring held until the signed-w plug control (§5, item 12) is run.
- **BDG's own alpha widening (1.034 / 1.042) is 0.6 and 0.8 se from none**, but that uses an unpaired se. BDG and its control share noise, and pairing roughly halves the se (to ~1.7 and 2.1 se), so it is borderline rather than null. It must be recomputed paired.
- Gap widens clearly (2.6 and 3.1 se unpaired); mu on one seed.

### 4.6 The chemistry floor is borrowed, and the verdict it decides is knife-edge

*Ruling R6: partially defended (0.86).*

- The handoff's "Floor FR3a = 0.362109375" is 0.9 × 0.40234375. That is the unguided stability of all 33 B200 batch-128 n = 512 cells at seed 20260921 in `results/sweep`, the very source §5 forbids comparing against.
- **It cannot transfer to CPU batch-512 cells.** `initial_noise` draws coordinates then types per batch, so one 512-batch assigns different noise to every molecule than four 128-batches do.
- For the same reason the repo's seed-2 B200 value (0.3691 → 0.3322) is not BDG's seed-2 control either, and must not be used.
- The §6.2 gap e4t0.5 row is a **seed-1** cell: 178/512 = 0.3477, se 0.021. It is −0.69 se below 0.3621 and −0.45 se below the pooled 0.3571 (0.9 × 0.3968, the v2 3 × 5,000 unguided).
- A fresh own-control floor flips its verdict with probability ≈ 0.31. **Report it as knife-edge, not as "FAILS".**
- The only correct floor is **0.9 × each job's own CPU unguided control**, which the handoff says it re-ran, with 0.3621 and 0.3571 shown as sensitivity checks.
- §5's blanket ban should become: *no paired, molecule-level comparison with `results/sweep`*. Aggregate rates may be compared, because the weights are identical and BDG's own plug control agrees with `results/sweep` q50 to |z| ≤ 0.67.

### 4.7 §6.3's "scientific core" has no plug control

*Folded into ruling R3.*

- The contrast is `bdg` sweeping sd/unguided by 31–43 % against `btvg_var`'s 0.4–8.6 %.
- But plug alone, with no variance term, sweeps it by **19–39 %** across its strength grid on this repo's own cells.
- Since BDG is plug with a servoed deviation gain, a plug-like range is expected.
- The contrast needs three rows (plug's w-sweep, `btvg_var`, BDG's τ ladder) and a bias/δ column before it can be called the core.

### 4.8 Novelty: one citation mis-described, the closest literature missing, two claims already in the repo

*Rulings R8 (0.85), R9 (0.85), R10 (objection stands, 0.86).*

**All four §8 citations exist; none is fabricated.**
- Particle Guidance (arXiv 2310.13102) is described correctly.
- MMD Guidance (arXiv 2601.08379) exists.
- **Variance-Tilted Diffusion** (arXiv 2606.22239) is described correctly as widen-only on a fixed linear feature. But its interaction term — repelling the batch's denoised means — *matches BDG's e < 0 dispersion term*. The handoff does not say so, and it should.
- **MGD** (arXiv 2602.17211) is **mis-described.** It is not "mean-only". Its corrector sets the drift from the residual between the *empirical ensemble moment* and a *target moment*, including quadratic moments, so it is two-sided. It occupies §8(a), "one controller both narrows and widens".

§8 omits the literature that occupies claims (b) and (d):
- *Feedback Guidance of Diffusion Models* (arXiv 2506.06085) frames fixed-scale guidance as open-loop and derives a state-dependent, self-regulating coefficient.
- *Adaptive Diffusion Guidance via Stochastic Optimal Control* (arXiv 2505.19367).
- CFG-Ctrl (arXiv 2603.03281) shows CFG is a proportional state-feedback controller with the scale as gain.
- *Controlling Ensemble Variance in Diffusion Models* (arXiv 2501.14822) aligns ensemble variance in both directions.

§8 also omits items this repo had already logged:
- Jeffrey guidance (arXiv 2606.13240);
- TOCFlow (arXiv 2601.09474);
- Histogram-constrained generation (arXiv 2606.31683);
- Multilevel/SMC guidance (arXiv 2601.21104).

Two of the four "what is ours" claims are already in the tree:
- **(b)**, the V_F-vanishes argument, is the comment guarding BTVG's clamp in `proj1/src/guidance.py` (≈1478–1494), including the runaway figures. The handoff's `:1471` pointer is stale; the clamp is at `:1494`.
- **(c)**, the setpoint-relative error V/τ², is already the `btvg2_V_over_tau2` diagnostic at `guidance.py:785`.

The team's own [Three_New §5.2](Three_New_Guidance_Ideas_Variance_and_Switching.md)
already uses the batch forecast variance (BDG's V_b), the setpoint τ and the
signed-deviation gradient. Its allocation is also affine in F_i (to 1.8e-15), so under
§3's own generalisation it too is "plug reweighted".

**What is narrow and unclaimed** (not found verbatim in about eight searches,
abstracts only; *not* a priority claim):
- the composition itself: a scalar guidance weight set by the batch variance of a *learned property predictor* against a *two-sided setpoint*;
- the widening-side bound;
- the pullback-sign property.

The handoff's "two independent checks rated this yes / yes / marginal" has no artifact in the repo and cannot be reproduced.

### 4.9 Statistics and the rubric

- **Pairing is the real statistical gap.** BDG cells and their controls share the seed's noise, so they should be compared paired (McNemar or a paired bootstrap), with max-T correction over the grid. All the handoff's z-scores are unpaired.
- **§9 item 5 is half right.** At the fitted gain the coupling design effect on in-band is only 0.97–1.04×, so the iid se is not materially wrong there. On *sd* the servo suppresses replicate spread (design effect 0.72 at the fitted gain, lower at higher gains), so the iid se *overstates* it and those z-scores are conservative. The larger threat is the max-over-13 selection in §9 item 1.
- **The rubric reward is narrower than §10 says.**
  - The §3.5 (0.625 pt) and §4.4 (0.675 pt, the largest Results item) mappings are correct.
  - "The rubric explicitly rewards investigating a non-improvement" quotes §4.3, which is about *guidance vs unconditional*, not a failed innovation. Credit for an innovation null sits in §4.7 (0.2), the Discussion (0.1) and defence 3d (0.1).
  - The template asks for "at least two informative controls" for §4.4. η = 0 ≡ plug and the one-sided cell are *gates*, not controls.
  - The assignment excludes "ordinary hyperparameter tuning" from counting as innovation, which is exactly where "a feedback-set guidance weight" will be probed.

---

## 5. Required changes before any BDG claim enters the paper

Items 1–10 need no new compute.

1. **State the target and the in-band variant.** Read `target_name` from a Betty cell. Label every BDG number "q50 (or dist) / n = 512 / CPU batch 512 / its own floor". Never place it beside the q90 v2 tables.
2. **Rewrite §1.**
   - "Spread governs in-band" holds only where |bias| < σ (q50). At q90, debiasing is worth 11–40× the whole spread lever, and the optimum is on the widening side.
   - Delete "we can prove" and "by definition".
   - Replace "Spread becomes a first-class control" with the mean/deviation form of §4.1.
3. **Fix the §2/§3 math.**
   - Give the law's equilibrium V* = τ²(1 − 1/η), as an asymptote the window does not reach.
   - Give w_eff ∈ [1 − η, ∞), the identity w_eff(y_eff − F̄) = y − F̄, and the y_eff singularity at w_eff = 0.
   - Split "stable" into boundedness / trackability / stability with their conditions.
   - Add a **"Design constraint: η > 1"** paragraph and an η column on every table.
4. **Withdraw "does not reduce to any fixed weight schedule"** from §3, §7 and §8(d). Replace it with the transfer result and its scope (§4.3), and label the time-average control as confounded.
5. **Remove "no existing arm widens reproducibly" and the 1.010× yardstick.** Cite `rch`/`dflow` with their mean shifts, and make no uniqueness claim until the signed-w plug control runs.
6. **Replace the borrowed floor** with 0.9 × each job's own CPU unguided control, shown beside 0.3621 and 0.3571. Mark knife-edge rows. Narrow §5's ban to paired, molecule-level comparisons.
7. **Rewrite §9 item 1** with the z, Šidák and correlated-max p values. State that η = 1 cells are contractors, not spread control.
8. **Rebuild §6.3** as a three-row table (plug w-sweep / `btvg_var` / BDG τ ladder) with a bias/δ column. Cite VARIANCE_DECOMPOSITION.md, including its re-widening caveat. Change "tightening costs chemistry" to "chemistry cost tracks |w_eff|".
9. **Rewrite §8.**
   - Describe MGD correctly, and note VTD's interaction-term match.
   - Add Feedback Guidance, SOC guidance, CFG-Ctrl, 2501.14822, Jeffrey guidance, TOCFlow and histogram-constrained generation.
   - Credit guidance.py's clamp comment for (b) and the btvg2 diagnostic for (c), and credit Three_New §5.2 as the direct precedent.
   - Split "what is ours" (the composition, the widening-side bound, the pullback sign) from "properties inherited from plug".
   - Attach the novelty search's scope, and drop the untraceable "two independent checks".
10. **Housekeeping.**
    - Delete the `*.bak_bdg` files in the author's tree.
    - Fix the stale `:1471` pointer.
    - Keep the `bdg` mode name clear of the `startswith("btvg")` branches in `arm_kwargs` and `strength_scale`.
11. **Diagnostics** *(compute)*. Persist raw pre-clamp e, w_eff per step, and the batch RMS (not mean) of the dispersion term. Add η, run-mean and range of w_eff, and bias/δ (continuous and decoded) to every row.
12. **Controls and tests for any §4.4 row** *(compute, about an hour locally)*.
    - Run the fresh-noise e-schedule replay on a second seed pair.
    - Run a matched signed-w plug control.
    - Use paired tests with max-T correction.
    - Pre-register a drop rule: if signed-w plug reaches BDG's spread at equal chemistry, drop the controller claim and present the decomposition instead.

---

## 6. What BDG is good for in this paper

| where | what | cost |
|---|---|---|
| §4.3 / §4.7 / §5 | **The bias-versus-spread decomposition by target.** Spread governs coverage at q50; bias governs it at q90, where removing the bias is worth 11–40× any spread control. The most useful result in the handoff, and it explains the whole v2 picture. | zero — the v2 sidecars already hold it |
| the BTVG failure story | **V_b vs V_F:** the across-molecule variance is the one that survives, and BTVG servoed the one that vanishes | needs the §5 item 8 table |
| §3.6 | **Transfer:** a scalar pullback with no Σ, the sign-preserving dispersion term, and a passing simplex toy. Earnable by argument; §4.6 needs Modality 2 code, which does not exist. | zero |
| §4.5 | **Same cost as plug**: one fifth of lgd_mc's and tfg's guide passes | zero |
| appendix | one BDG table labelled q50 (or dist) / n = 512 / CPU batch 512 / its own floor | from the author's cells |

It may appear in §3.5 only as a variant with the reduction stated in the paper's own
voice, never as "variance control" or an irreducible loop.

---

## 7. Doability

**The engineering is easy.** The port agent implemented BDG from §2's prose in one session:

| item | result |
|---|---|
| size | 610 inserted lines across 4 files |
| gates | 22/22 pass (re-run for this review); 11 existing regression suites still pass |
| runs | 64 cells at about 21 s each on the local 5080 |
| stability | 0 non-finite samples, no OOM, within the 5 GB RAM rule |

The whole handoff grid would take about 30 GPU-minutes locally.

**The port is on branch `worktree-wf_bc7c0f18-844-2`** (commits `7cfc69c`, `f62b1b6`).
It is **not merged**. Its merge-base is `main`'s HEAD (`902aafa`), so it would
fast-forward cleanly. Merging is a decision for the team, and is not needed for any
recommended paper use. The author's own Betty fork is a different matter: it is stale
(its clamp pointer is 23 lines off) and would conflict with the tune chain in
`b00f11d`.

**The constraint is the calendar.**
- There is no paper draft in the repo yet.
- Modality 2 (DNA on the simplex) has no code.
- The 35–62 GPU-h tune chain is what produces the paper's §4.4/§4.5 tables.
- §5 items 1–10 are about three hours of writing. Items 11–12 are about an hour of local compute.

---

## 8. Attack, defence and ruling, per objection

| # | objection | severity | defence | ruling |
|---|---|---|---|---|
| R1 | "The loop does not reduce to any fixed schedule" rests on a confounded control and fails the clean one | fatal | partial | partially defended (0.80) |
| R2 | The unstated target is q50; the §1 premise and the impossibility result reverse at q90 | fatal | revise | partially defended (0.86) |
| R3 | Widening is negative-weight plug and the τ ladder is a strength ladder; §6.3 has no plug control | major | partial | partially defended (0.80) — *the negative-weight-plug inference rebutted* |
| R4 | The equilibrium is τ²(1 − 1/η), not τ², and η = 1 cannot widen | major | partial | **objection stands** (0.90) — *severity narrowed: an asymptote, not a measured undershoot* |
| R5 | "No existing arm widens reproducibly" is false; 1.010× is noise; alpha widening is inside noise | major | partial | partially defended (0.80) — *alpha borderline once paired* |
| R6 | The floor is borrowed, is wrong for seed 2, and decides §6.2's load-bearing row | major | partial | partially defended (0.86) — *seed-2 point rebutted; the row is knife-edge, not decisive* |
| R7 | The only floor-clearing wins (η = 1) are selection noise and cannot reflect spread control | major | partial | partially defended (0.80) — *correlated null used* |
| R8 | §8(a) is occupied by MGD, which the handoff mis-describes | major | partial | partially defended (0.85) |
| R9 | §8(b) and (c) are already in the repo's code, and (b)'s argument is the wrong one | major | partial | partially defended (0.85) — *non-vanishing is necessary, not sufficient* |
| R10 | §8 omits the closed-loop guidance literature and understates §5.2; the 0.85/0.1 has no artifact | major | partial | **objection stands** (0.86) |
| R11 | No project metric rewards batch-spread control, and there is no single-molecule mode | major | partial | partially defended (0.80) |
| R12 | Not worth the remaining days as a headline | major | partial | partially defended (0.80) — *the reason is power, not bookkeeping* |
| R13 | The handoff *under*-sells the defensible properties it never claims | minor | partial | partially defended (0.82) |

"Partially defended" means the defence salvaged a narrower claim, or won specific
points. The defences won outright on the negative-weight-plug inference (R3), the
seed-2 floor (R6), the noise comparison in the replay (R1), the null benchmark (R7)
and the unpaired se (R5). §4 is written to the rulings, not to the original attacks.

---

## 9. Provenance

| what | where |
|---|---|
| math checks (gradient, reduction, stability, pullback, estimator noise, basin, open-loop, calibration) | `audit/bdg_review/math/` |
| premise, widening scan, target identification, floor, selection, and the defences' checks | `audit/bdg_review/data/`, `audit/bdg_review/defend/` |
| GPU questions, open-loop replay on fresh noise, one-sided check, target triangulation | `audit/bdg_review/gpu/` |
| equilibrium law, closed-loop toy, w_eff trajectories | `audit/bdg_review/fixed_point_law.*`, `closed_loop_toy.*`, `weff_traj.json` |
| **the port's 64 cells** with per-molecule sidecars, and the table they produce | `results/bdg_port/cells/`, `results/bdg_port/table.txt` — regenerate with `python results/bdg_port/bdg_table.py` (verified byte-identical) |
| the full agent record (every finding, objection, defence and ruling) | `audit/bdg_review/workflow_result.json` |
| the port itself | branch `worktree-wf_bc7c0f18-844-2` |

The port's cells were sampled with `weights/fm_ema.pt` (file md5 `e19ccc06`), whose
weights are byte-identical to `fm_last.pt` (md5 `a190ac83`; see `weights/README.md`),
on the local RTX 5080 with torch 2.11.0+cu128.
