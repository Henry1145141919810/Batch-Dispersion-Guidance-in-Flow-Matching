# Part 1 content: slides 1a to 1f

Every number here is measured and may be used verbatim. Citation numbers refer to
`05_BIBLIOGRAPHY.md`; renumber per slide if you prefer, but keep the source list at the
bottom-right consistent with whatever numbering you use.

---

## 1a. Frame the study  (1 slide)

**Title:** Frame the study

**Problem.** Generate 3D molecules whose property lands inside a tolerance band around
a target, without retraining the generator. A design campaign wants many usable
candidates, so the quantity that matters is **in-band**, the fraction of samples inside
the band.

**Central question.** Guidance concentrates a batch toward the target, and that
dispersion is a by-product of its strength. Does making dispersion a quantity the user
*requests* raise band coverage?

**Hypothesis.** At a centred target, coverage is limited by batch spread rather than
centring error, so a calibrated spread setpoint should raise it.

**Two continuous modalities.**
- **Modality 1: QM9 3D molecular coordinates.** State space R^(N×3) with a relaxed
  atom-type simplex, N ≤ 29. Invariant to translation, rotation, atom permutation.
- **Modality 2: DNA enhancers on the probability simplex.** State space (Δ³)^L, one
  simplex per position over {A,C,G,T}. No rotation group; the constraint is that each
  position stays on the simplex.

**Takeaway line.** These are different state spaces, not two datasets of the same kind:
one is a Euclidean coordinate space with a geometric group acting on it, the other a
product of probability simplices.

**Sources:** [1] QM9 dataset. [2] EDM. [3] MOG-DFM enhancer data.

---

## 1b. Data and representations  (2 slides)

### Slide 1b-i: datasets, splits, leakage

**Modality 1: QM9** [1], pinned by SHA-256. 133,885 molecules, explicit hydrogens.

| Split | Used for | n |
|---|---|---|
| `train_a` | generator **and** guide f_A | 51,527 |
| `train_b` | evaluator f_B only | 51,527 |
| `val` | checkpoints, atom counts, δ | 17,748 |
| `test` | held out | 13,083 |

**How leakage is prevented.** f_A and f_B are trained on **disjoint halves**, so the
network that scores us never saw the data that trained the network that steers us.
Normalisation constants come from `train_a` alone. The calibration set that fixes δ is
disjoint from the sets supplying targets and atom counts.

**Modality 2: Kenyon-cell enhancers** [3]. 6,126 *Drosophila* sequences. The evaluator
is an exact count, so no train/validation split is needed and no leakage is possible.

**We do not merge the two species** distributed together. Their GC distributions
differ, so steering spread on the mixture would partly steer the species mixing
proportion, which is the quantity under study.

**Preprocessing.** Coordinates zero-centred per molecule, in Å; dense [B,29,·] padding
with a validity mask; DFT properties in D / Bohr³ / Ha.

**Sources:** [1], [3].

### Slide 1b-ii: representations, conditioning, modality-specific handling

**Representations, and why.**
- **M1:** coordinates plus a *relaxed* one-hot over {H,C,N,O,F}. The continuous
  relaxation is what lets one continuous flow carry both geometry and discrete types.
- **M2:** a point of (Δ³)^L per sequence. The simplex counts as a continuous state
  space by the assignment's definition, and it is the native geometry of the sequence
  literature [4,5].

**Conditioning.** Base generation is **unconditional**. The property enters only at
sampling time through guidance. This is deliberate: a conditional model would give the
controller nothing to steer and would break the M1/M2 parallel we are testing.

**Modality-specific handling.**
- **M1 invariance:** translation by projection onto the zero-centre-of-mass subspace;
  rotation by messages depending only on squared pairwise distances, with coordinate
  updates along difference vectors; permutation by symmetric pooling. Verified
  numerically: **12/12 gates pass to ~1e-17**.
- **M2 simplex mapping and decoding:** sequences are relaxed onto the simplex and
  decoded by per-position argmax; the property is scored by an **exact** GC count on the
  decoded sequence.

**Sources:** [4] Dirichlet FM. [5] Fisher Flow.

---

## 1c. Flow-matching baseline  (2 slides)

### Slide 1c-i: mathematical formulation

Data x₁ ~ p_data, source x₀, time t: 0 → 1 from noise to data.

**Probability path (linear interpolant)** [6]:
```
x_t = t·x₁ + (1-t)·x₀       u_t(x_t | x₀,x₁) = x₁ - x₀   (constant along the path)
```

**The source is not isotropic.** Coordinates are x₀ᶜ = Π₀·ε with Π₀ the projection onto
the zero-centre-of-mass subspace of the valid atoms, dimension 3(N−1); type channels are
a masked Gaussian.

**Loss, exactly as implemented** (masked mean over valid entries, two heads normalised
separately, λ = 1):
```
L_FM(θ) = E[ (1/(3·ΣM))·||M ⊙ (v^c_θ − u^c)||²  +  λ·(1/(K·ΣM))·||M ⊙ (v^f_θ − u^f)||² ]
```
**New symbols.** M is the atom mask, K the number of type channels, v_θ the learned
velocity. Separate normalisation stops padding diluting the loss.

**Endpoint estimate** that every guidance arm differentiates, available in closed form
at every t:
```
x̂₁ = x_t + (1−t)·v_θ(x_t, t)
```

**Sources:** [6] Flow Matching. [7] Rectified Flow. [8] Stochastic Interpolants.

### Slide 1c-ii: architecture, training, selection, sampling

**Architecture.** Dense E(n)-equivariant graph network [9]: width 256, 8 layers, SiLU,
**3,753,229 parameters**. Time enters at the input embedding. No attention.

**Optimiser and schedule.** Adam, lr 2e-4, cosine annealing to 5% of it, batch 256,
gradient-norm clip 5.0, EMA decay 0.9999 with warm-up. Sampling always uses the EMA copy.

**Budget.** 1500 epochs = 303k steps, 11.4 h on one B200 MIG slice, seed 20260918.

**Model selection (pre-registered).** Highest *validation* atom stability at NFE 100,
ties broken on validation loss, evaluated every 25 epochs.

**Disclosed amendment.** The rule selected epoch 1300; we ship **epoch 1500**, because
the 512-sample selection signal could not resolve the last 200 epochs. The
3-seed × 10,000 benchmark confirms it on every metric: +0.0018 atom, +0.0097 molecule,
+0.0110 validity, same sign on all three seeds.

**Sampling.** Probability-flow ODE, **explicit Euler**, uniform grid, **100 steps ⇒
NFE 100**. Quality saturates by 500 steps (100→500 buys 0.65 atom and 2.8 molecule
points), so 100 is a cheap operating point.

**Takeaway line.** The closed-form endpoint estimate at every t is why this family is
convenient for guidance, and it is the reason we carry it forward in 2a.

**Sources:** [9] EGNN. [6] Flow Matching.

---

## 1d. Diffusion baseline  (2 slides)

### Slide 1d-i: mathematical formulation

Continuous-time **variance-preserving** process [10,11], τ: 0 → 1 from clean to noise,
**linear β**:
```
β(τ) = β_min + τ(β_max − β_min),        β_min = 0.1,  β_max = 20
x_τ  = α(τ)·x₀ + σ(τ)·ε
log α(τ) = −½( β_min·τ + ½(β_max − β_min)·τ² ),        α² + σ² = 1
```

**Prediction target: ε**, under the *same* two-term masked-mean loss as the flow model,
with coordinate noise projected onto the zero-centre-of-mass subspace so the target is
equivariant. Times τ ~ U(1e-3, 1) keep σ away from zero.

**Reverse sampler: probability-flow ODE**
```
dx/dτ = −½·β(τ)·( x + s_θ(x,τ) ),        s_θ = −ε_θ / σ
```
integrated backwards on a decreasing 100-step grid, then a jump to the Tweedie
posterior mean.

**Derived quantities guidance consumes:** x̂₀ = (x_τ − σ·ε_θ)/α and k = σ²/α.
Verified: α²+σ² = 1 to 6 decimal places, α(0) = 1, α(1) = 0.0066.

**Sources:** [10] Score SDE. [11] EDM. [2] EDM molecules.

### Slide 1d-ii: architecture, training, selection, sampling

**Matched by construction.** The diffusion baseline shares the representation, the
split, the **same EGNN backbone**, the optimiser, the learning-rate schedule, the batch
size, the EMA, the gradient clipping, the seed and the **same pre-registered selection
rule** with the flow model. It differs **only** in the corruption process. That is what
makes 2a a comparison of families rather than of engineering.

**Architecture and training.** EGNN 256×8, 3.75M parameters, ε-prediction head. Adam
2e-4, cosine to 5%, batch 256, clip 5.0, EMA 0.9999, 1500 epochs.

**Model selection.** Highest validation atom stability at NFE 100, ties on loss: the
identical rule.

**Sampling.** 100-step decreasing grid, then the Tweedie endpoint jump, so the unguided
budget is **NFE 101** against the flow model's 100. We report that asymmetry rather
than hiding it.

**Status, stated plainly.** Our own VP diffusion is **trained and benchmarked** (28
Sep): epoch 1500, selected 1475, `weights/diff_ema.pt` md5 `8a3390a6`, 18 v3 cells.
There are now **two** measured diffusion comparators -- `vp` (ours) and `edm` (TFG's
borrowed EDMsecond, labelled as borrowed on 2a). Flow matching beats our own VP at
matched everything (mol stab 0.3970 vs 0.2883, validity 0.7562 vs 0.6617, atom stab 0.9356 vs 0.9070; every gap 7-14x the sd of the difference).

**Takeaway line.** We will not claim a family winner from a borrowed checkpoint. 2a
states what the evidence does and does not support.

**Sources:** [9] EGNN. [2] EDM. [11] EDM design space.

---

## 1e. Comparison plan: metrics and fairness  (2 slides)

### Slide 1e-i: metrics, and what each cannot tell us

**Control metric.**
- **in-band** = fraction of samples with |f_B(x) − y| ≤ δ, scored by the **held-out**
  evaluator.
- **δ = 2 × f_B's calibration MAE** on 3,000 molecules disjoint from the target and
  atom-count sets. **Pre-registered**, so it cannot be chosen after seeing results.
- Reported both continuous and decoded.

**Decomposition.** RMSE² = bias² + spread², both in units of δ, plus spread relative to
the unguided control. This is the quantity the innovation targets.

**Quality and diversity.** Atom stability, molecule stability, validity, uniqueness,
novelty, pairwise diversity, under EDM conventions [2], with bond tables verified
**entry-for-entry** against the reference implementation.

**Sample counts.** Base models 3 seeds × 10,000. Guidance cells n per cell × 3 seeds
`[TODO: v3 n]`. Screening cells n = 256–512, labelled wherever used.

**Limitation we state up front.** in-band rewards *any* arm that contracts the batch,
including onto chemically implausible structures. So every in-band number in this deck
appears beside the chemistry it spent.

**Takeaway line.** No arm is ranked on in-band alone. A method that wins coverage by
destroying molecules has not won.

**Sources:** [2] EDM conventions.

### Slide 1e-ii: fairness, what is matched and what is not

| | Matched | Disclosed mismatch |
|---|---|---|
| Data, split, preprocessing | identical | none |
| Conditioning | unconditional both | none |
| Atom-count distribution | drawn from `train_a` both | none |
| Evaluator, metric code, δ | identical | none |
| Starting noise, seeds | shared, so arms are *paired* | nothing paired *across* backends |
| Backbone | same EGNN 256×8 | borrowed comparator is 256×9 + attention |
| Parameters | 3.75M vs 3.75M | borrowed comparator 5.34M |
| Sampling budget | 100 steps both | diffusion needs NFE 101 for its endpoint jump |
| Training compute | identical recipe | `[TODO: v3 Job A]`; borrowed checkpoint budget unknown |
| Feature scaling | ours unscaled one-hot | borrowed comparator scales by 1/4 |

**Guidance-strength fairness.** The headline fixes **w = 1** for every arm. This asks
*what each rule does at one dial setting*, not which is best at its own optimum.
**Equal w is not equal force**: the measured applied-correction share spans **5.8×**
across the arms where it can be measured, so every sentence quoting a headline number
carries "at w = 1".

**Statistics.** Binomial standard errors; paired comparisons within a backend;
multiplicity-corrected significance reported beside the single-test value, because each
arm is selected over a grid. **in-band is never compared across backends**, because δ
is set by that backend's evaluator.

**Sources:** original table; conventions from [2].

---

## 1f. Study overview  (1 slide)

**Title:** Study overview

Full-width: `figures/fig1_overview.png` (or the PDF).

**Caption / spoken content.** (a) Modality 1: a shared representation feeds a
flow-matching and a diffusion model, and a matched protocol selects the family carried
forward. Guidance differentiates the frozen f_A at the endpoint estimate. The plug-in
baseline applies one gain to (y − F_i); BDG holds the centring gain at w and servoes
only w_eff = 1 + η·e. Scoring uses the frozen f_B, trained on the disjoint half.
(b) Modality 2: the controller transfers unchanged; state space, generator, property
map and decoder are modality specific. Colour marks trained, frozen, external
pretrained, and changed-by-the-innovation components.

**Source line:** Original figure (this work). Components after [6], [2], [4].
