# Top 6 inference-time guidance recommendations

**Ranked by innovation × rigour × expected payoff. 20 September 2026.**

Every ranking below is backed by checks run for this memo on synthetic problems with an exact
reference and an *ideal* denoiser, so the only approximation under test is the guidance estimator
itself. Code and raw outputs: `guidance_checks/`. Nothing here has been run on molecules, and no
claim below is a measured molecular result.

Scoring is on three axes, each 1–5:

- **Innovation** — would a reviewer who knows TMPD, LGD, TFG and the Jan-2026 SMC work call this new?
- **Rigour** — is it proved, is it verified numerically, and does it survive its own strongest control?
- **Payoff** — how likely is it to change a terminal metric at matched compute?

| # | Recommendation | Innov. | Rigour | Payoff | Net |
|---|---|---|---|---|---|
| 1 | κ₃ as an inference-time posterior-shape instrument | 5 | 4 | 4 | **13** |
| 2 | Completed-moment closure (repair SMG) | 2 | 5 | 5 | **12** |
| 3 | QPMG, re-scoped and re-normalised | 3 | 5 | 2 | **10** |
| 4 | Closure-residual escalation | 4 | 2 | 4 | **10** |
| 5 | DBFG, re-scoped to the objective | 3 | 4 | 2 | **9** |
| 6 | MAG's insufficiency theorem, kept as a result | 2 | 5 | 1 | **8** |

**Rejected outright:** Tolerance-Band Guidance (D), balanced preview selection, MAG as a method,
the third-cumulant *closure* built and tested for this memo. Reasons in §7.

---

## 1. κ₃ as an inference-time posterior-shape instrument

**Innovation 5 · Rigour 4 · Payoff 4**

### The claim

The third cumulant of the clean posterior is *free information* that every guidance method currently
discards, and it is the exact quantity that separates cases where a moment closure is trustworthy
from cases where it is not.

Because `p(X | xₜ)` is an exponential family in `η = αₜxₜ/σₜ²`, differentiating Tweedie's covariance
identity once more gives

```
∂Σ/∂xₜ = (αₜ/σₜ²) κ₃          [verified to 3e-9 … 3e-8]
```

so, since `Σ = (σₜ²/αₜ) J`,

```
κ₃[g,g] = (σₜ²/αₜ) ∇_{xₜ} ( gᵀ Σ(xₜ) g )      γ = κ₃[g,g,g] / (gᵀΣg)^{3/2}
```

`gᵀΣg` is a scalar built from one JVP and a dot product. Its gradient is one more reverse pass. So
the standardised skew of the *observable* costs two to three extra network passes — no training, no
posterior samples, no reference sampler.

### Why this is the strongest direction

Every closure in this space — DPS, TMPD, MMPS, SMG, QPMG, ΠGDM — assumes the clean posterior is
adequately described by two moments. **Nobody measures whether that is true on their checkpoint.**
They cannot: with a learned denoiser there is no exact reference. The κ₃ probe gives one, per state
and per target, and it is computable exactly where you need it.

This reframes the contribution from "our guidance field is better" — a crowded and now
hard-to-defend claim — to "here is how much of guidance error is posterior shape, measured, and here
is the budget rule that follows." That question is answerable rigorously and the answer changes what
anyone should build next.

### What is already established vs what is new

| Established | This proposal |
|---|---|
| Tweedie first order, `m = E[X\|xₜ]` | — |
| Tweedie second order, `Σ = (σₜ²/α)J` (TMPD, MMPS) | — |
| Third-cumulant identity as exponential-family algebra | Not novel mathematically |
| — | Computing `∂Σ/∂xₜ` **at inference, as a guidance-trust probe**, contracted into a scalar skew |

No opened source computes this. **Novelty status: provisional, unresolved for exhaustiveness.** The
identity is textbook; the use is the claim, and that is a narrow claim by construction.

### Honest limits, already measured

Two obvious uses of the probe were built and **both failed**:

- As a **closure** (skew-corrected guidance, shifted-gamma observable law): improves on the
  two-moment closure where the posterior is Gaussian (0.19 vs 0.37 relative field error) and
  **breaks where it is multimodal** — the truncated variance goes negative and a unimodal skewed law
  cannot represent a bimodal observable.
- As an **abstention gate**: pooled over 6000 states, Spearman ρ with closure error is 0.611 for
  |γ|, 0.483 for `gᵀΣg`, and **0.726 for the time index alone**. Flagging the worst 20% catches 24.9%
  of the error mass by skew vs 33.7% by time and 37.0% by spread. **The free signal wins.**

These are reported as negative results, not buried. They do not damage the instrument claim — they
constrain what may be built on it.

### Go / no-go

**Go** if |γ| predicts closure error with partial Spearman ρ ≥ 0.5 **after partialling out t and
gᵀΣg**. Raw correlation is explicitly not sufficient; the check above already shows why.
**Stop** if the learned denoiser's Jacobian is too noisy for the second derivative to be stable —
test with finite differences at three step sizes before anything else.

### Cost

2–3 extra network passes per evaluation. No training, no extra guide calls.

---

## 2. Completed-moment closure — repair SMG

**Innovation 2 · Rigour 5 · Payoff 5**

### The defect

SMG's denominator is `s² + gᵀΣg`. For the quadratic surrogate
`q(u) = f(m) + gᵀu + ½uᵀHu` with `u ~ N(0,Σ)`, the exact second moment is

```
E[q]   = f(m) + ½ tr(HΣ)          ← SMG has this
Var[q] = gᵀΣg + ½ tr((HΣ)²)       ← SMG is missing the second term
```

**A method named "second-moment guidance" does not use the second moment.** The omitted term is
exactly the variance contributed by curvature, and it is the term that keeps the field bounded where
`g` is small.

### The repair and its size

```
Δ = (y − μ_F) / (s² + V_F) · Σg,    μ_F = f(m) + ½tr(HΣ),   V_F = gᵀΣg + ½tr((HΣ)²)
```

One extra trace. No new network forwards. Reduces to SMG for affine `f`.

Near a critical point of `f` (1-D, `X~N(0,1)`, `f = x²`, `y = 1`), slope of the shift at the origin:

| t | s | exact | plug-in | SMG | **SMG-2** |
|---|---|---|---|---|---|
| 0.7 | 0.30 | 0.051 | 8.01 | 1.65 | **0.110** |
| 0.7 | 0.10 | 0.111 | 72.1 | 14.5 | **0.117** |
| 0.7 | 0.03 | 0.117 | 800.9 | 127.8 | **0.118** |
| 0.5 | 0.03 | 0.705 | 785.5 | 271.9 | **0.705** |

A **1000×** error becomes a 1% error. Field error on a Gaussian prior with `f = |x|²`:

| s | SMG | **SMG-2** |
|---|---|---|
| 0.50 | 2.62 | **0.294** |
| 0.15 | 4.90 | **0.318** |
| 0.05 | 6.47 | **0.350** |

In d = 16, Wasserstein-1 of the property in units of s: SMG 1.340 → **SMG-2 0.189**. And the traces
survive stochastic estimation — a *single* Rademacher probe gives W1 0.132 against 0.150 with exact
traces, so the cost is genuinely negligible.

### Why it ranks second despite the highest payoff

**It is not novel.** It is the obvious extension of TMPD/MMPS moment projection from an affine
observable (where `½tr((HΣ)²) ≡ 0`) to a quadratic one. It is a correction to your own prior work,
not a contribution.

**Recommendation: ship it as the control, and say so.** Every claim in this project must now beat
it at matched cost. It also changes the story of any comparison already run against "SMG" — those
comparisons were against a defective baseline.

### Go / no-go

**Go** if it beats the current implementation by ≥ 2× in field error at equal cost on the real
checkpoint. Predicted: yes, comfortably.
**Stop the whole programme** if a strength-tuned plug-in arm matches it at equal cost on terminal
metrics — the checks show tuned plug-in (λ ≈ 0.003–0.01) is far better than its untuned reputation.

---

## 3. QPMG, re-scoped and re-normalised

**Innovation 3 · Rigour 5 · Payoff 2**

### What is right

The proposition is **correct**, and the counterexample reproduces to 12 digits:

```
SMG   = (0.96153846,  0.00000000)
QPMG  = (0.66267363, -0.04850552)    matches finite differences of the true
                                      smoothed likelihood to 7e-12
```

QPMG's mean-derivative is the exact Gaussian-posterior Stein form specialised to a quadratic
property. For a *globally* quadratic `f` it is not an approximation at all.

### Three corrections required

**(a) A factor-of-s² double count.** The Sec 2.3 Fourier numerator with vector factor
`iωg − ω²HLC⁻¹a` **already equals ∇ₘZ_Q** — the `1/s²` from differentiating the Gaussian likelihood
is absorbed when the whole transform is differentiated. Applying the Sec 2.2 convention
`G_Q = num/(s²Z_Q)` to it double-counts. In your own example that is a factor of **1.5625**:

```
corrected   Fourier:  (0.66267363, -0.04850552)   ← matches quadrature to 4.8e-12
as written (num/s²Z): (1.03542754, -0.07578988)
```

Use **`G_Q = ∇ₘZ_Q / Z_Q`**, with no further `1/s²`.

**(b) The cross-check does not certify the operating regime.** The reported Gauss-Hermite / Fourier
agreement (5.4e-11) was obtained at `s = 0.8`. At small `s` the tensor GH rule fails badly while
Fourier stays exact:

| s | QPMG via GH (120 nodes/axis) | QPMG via Fourier |
|---|---|---|
| 0.50 | 0.034 | **5.5e-5** |
| 0.15 | 11.55 | **1.0e-4** |
| 0.05 | 85.62 | **1.1e-4** |

Good news for the Fourier route; a warning about the validation protocol. Re-test at the smallest
`s` you intend to run.

**(c) The "new direction" claim is overstated.** The angle between SMG and QPMG in your example is
**4.19°**. The second component is genuinely nonzero, so strictly no scalar rescaling reproduces
QPMG — but the best scalar rescaling of SMG leaves a residual of only **7.3%** of ‖QPMG‖, and the
free SMG-2 fix closes it to 16%. *"No scalar multiple reproduces it"* and *"the direction matters"*
are different claims; only the first is established.

### The decisive limitation

QPMG assumes a Gaussian clean posterior. When that fails, it is **no better than the free fix**:

| s | t | SMG | SMG-2 | QPMG |
|---|---|---|---|---|
| 0.15 | 0.7 | 0.869 | **0.815** | 1.163 |
| 0.15 | 0.5 | 0.911 | **0.488** | 0.609 |
| 0.50 | 0.7 | 0.816 | **0.782** | 1.128 |

It integrates the property law exactly under a posterior model that is wrong, and the posterior
error dominates. **This is the ranking's whole argument:** QPMG perfects the smaller term.

### The narrow claim that survives

> For a Gaussian clean posterior, the Gaussian-observable closure is not exact for a curved
> property, and the exact closure is computable by a one-dimensional Fourier integral at any ambient
> dimension. The measured size of that gap is 0.29–0.35 relative field error, closed to 1e-4.

Whether it changes terminal samples is unresolved.

### Go / no-go

**Go** only if property-law error is ≥ 30% of total error at your operating `s`. If posterior-shape
error dominates by more than 3:1 — which recommendation #1 measures directly — **drop QPMG**.

---

## 4. Closure-residual escalation

**Innovation 4 · Rigour 2 · Payoff 4 — untested, highest-variance bet**

### The idea

The failed gate in #1 tried to predict closure error from a *property of the posterior* (skew). The
untested alternative is to predict it from **disagreement between two closures you are already
computing**:

```
r(xₜ) = ‖ Δ_exact-law − Δ_two-moment ‖ / ‖ Δ_two-moment ‖
```

Spend the cheap closure everywhere; escalate to a sampling estimator (LGD-MC at high n, or SMC) only
where `r` is large. This is the standard embedded-error-estimator idea from adaptive quadrature and
adaptive ODE solvers, applied to guidance-field estimation rather than to step size.

### Why it is plausible where the skew gate failed

The skew gate failed because it measured a property of the posterior that correlates with, but is
not, the error. A residual between two estimators of the *same* quantity is a direct error proxy — 
it is what every adaptive solver uses, and it requires no new probe because both closures are on the
critical path already.

### Why it ranks fourth

**It has not been tested.** Rigour score 2 reflects exactly that. It is listed because it is the
most promising unexplored lead the checks surfaced, and because it is cheap to falsify.

Prior art to clear before claiming anything: adaptive solver tolerances in diffusion samplers, and
the TFG hyperparameter-search literature, which already adapts guidance strength over time.

### Go / no-go

**Go** if `r` beats *all three* of {t, `gᵀΣg`, |γ|} at catching error mass in the same pooled
protocol used in #1. **Stop** if the escalation budget needed to move a terminal metric exceeds
simply running the sampling estimator everywhere — that is the control, and it is a strong one.

---

## 5. DBFG, re-scoped to the objective

**Innovation 3 · Rigour 4 · Payoff 2**

### What is right

The binary formula verifies exactly (0.7419117444, finite-difference error 8e-12). The facet
decomposition is correctly stated. And the structural observation is right and worth saying plainly:
**the type-channel component of the guidance comes entirely from facet terms**, because in-cell type
derivatives vanish identically.

### What the decisive test shows

Your own document names the right control — a score-function estimator on the same decoded
likelihood. At matched decoded-guide-call budget (4 atoms × 3 types, correlated Gaussian proposal,
reference from 3–4e6 draws), relative RMS error of `∇ₘ log Z_D`:

| proposal scale | SF n=32 (32 calls) | SF n=128 (128) | facet 4×4 (48) | facet 12×8 (208) |
|---|---|---|---|---|
| 0.55 | 3.65 | 1.87 | 3.53 | 1.67 |
| 0.25 | 3.22 | 1.68 | 2.84 | 1.39 |
| 0.10 | 2.97 | 1.51 | 2.54 | 1.10 |
| 0.04 | 3.19 | 1.63 | 2.31 | **1.02** |

At loose proposals it is a wash. The facet estimator's edge appears only as the proposal tightens:
at scale 0.04 it reaches 1.02 against roughly 1.28 for a budget-matched score-function estimator —
about **1.5–1.7× in effective samples** in the favourable regime. Real, but modest against the
facet bookkeeping, the ratio bias, and the equivariance constraints it costs.

### The untested advantage that may matter more

When the reward has a **smooth coordinate part**, DBFG computes that part's gradient exactly in-cell
while the score-function estimator must estimate all of it stochastically. That was not modelled in
the test above (reward depended on decoded types only). Test it before discarding the estimator —
it is plausibly where the real gain is.

### The re-scope

The defensible contribution is the one §3.8 reaches and then under-weights: **the objective change**
from soft type mixtures to decoded types, not the estimator that computes its gradient. A guide
evaluated on fractional type mixtures may reward states that stop looking good after `argmax`. That
is a genuine, modality-specific failure and prior work on it is thinner.

### Go / no-go

**Measure the soft-versus-decoded guide gap on already-generated molecules first.** One cheap pass
over existing samples. If that gap is smaller than the improvement any arm in #1–#3 achieves,
**drop DBFG entirely** — neither estimator matters. Only if the gap is large does the estimator
question arise, and then the score-function estimator is the thing to beat.

---

## 6. MAG's insufficiency theorem — keep the result, drop the method

**Innovation 2 · Rigour 5 · Payoff 1**

### The result is correct and sharper than claimed

Two posteriors with **identical mean and variance** whose exact guidance is equal and opposite:

```
P₊:  P(X = −0.5) = 0.8,  P(X = 2) = 0.2     mean 0, var 1, κ₃ = +1.5
P₋:  mirrored                                mean 0, var 1, κ₃ = −1.5
f = X², y = 4, s = 1

exact tilted means:  ±1.9911928728          [reproduced exactly]
```

It is sharper than the document states. Because `g = ∇f(m) = 2m = 0` at the anchor, **every** field
of the form `Σ · (scalar) · g` returns exactly zero — plug-in, SMG, SMG-2, QPMG and ΠGDM alike. No
rescaling, preconditioning, or covariance choice recovers the sign. This is a clean, citable
motivating result and it should open any write-up in this space.

### Why the method must be dropped

**The conclusion drawn is too strong.** The two posteriors differ only in their third cumulant, and
the third cumulant is *not hidden from a frozen generator* — that is recommendation #1, verified to
1e-8. The honest framing is **"the two moments you computed are insufficient"**, not **"the model
cannot tell you."** MAG builds an arbitrary hypothesis family to stand in for information it could
have measured.

And the remedy fails its own controls: skew-based abstention loses to the time index (ρ 0.61 vs
0.73). Anything MAG's gate would flag, a schedule flags better and for free. The minimum-norm
common-ascent machinery is unmodified MGDA, and the document says so.

### Recommendation

Keep §4.2 of the MAG proposal as a **motivating counterexample in the introduction**. Delete the
hypothesis family, the convex-hull selection, and the abstention rule. The insufficiency result is
worth a paragraph in a paper; the method is not worth a pilot.

---

## 7. Rejected, with reasons

| Candidate | Reason |
|---|---|
| **Tolerance-Band Guidance (D)** | A dead-zone objective on a scalar residual plus a norm-constrained QP with one linear inequality. Both halves standard; the tolerance band *is* the zero-loss set of the dead-zone loss. A local acceptance test is not a terminal guarantee. Nothing here needs a new name. |
| **Balanced preview selection** | Self-defeating by construction. A marginal-preserving reordering cannot change the expected number of single-sample successes at fixed output count. It changes batch dependence only. |
| **MAG as a method** | Gate beaten by the free time signal; hypothesis family arbitrary; optimisation is unmodified MGDA. Keep the counterexample only. |
| **Third-cumulant closure** (built and tested here) | Improves on the two-moment closure for Gaussian posteriors (0.19 vs 0.37) and **breaks on multimodal ones** — truncated variance goes negative; a unimodal skewed law cannot represent a bimodal observable. My own idea, killed by its own decisive test. |

---

## 8. Two cross-cutting warnings

**Field accuracy is not terminal accuracy.** With a stochastic sampler (η = 1) and no strength
tuning, methods separated by an **order of magnitude** in field error repeatedly landed within noise
of each other in terminal energy distance — the reverse SDE absorbs local error. With the
deterministic sampler the ordering survives. **Report both**, or the claim is an artefact of the
solver choice.

**Tuning hides defects.** Plug-in guidance needs a strength of 0.003–0.01 to avoid catastrophe, and
once tuned it is respectable. Any comparison at fixed strength overstates the winner. Equal tuning
budget per arm, counted and reported.

---

## 9. The competitive landscape has moved

[Gleich and Schmidler (Jan 2026)](https://arxiv.org/abs/2601.21104) now occupy the posterior-shape
problem directly, with an unbiased SMC estimator of `p(y|xₜ)` and multilevel variance reduction —
95.6% accuracy on CIFAR-10 label guidance at 3× lower cost-per-success than baselines, and a 1.5×
cost-per-success advantage on ImageNet. They identify the same root cause this memo measures:
existing methods rely on point approximations that fail to capture posterior multimodality, and the
bias accumulates along the trajectory.

**Consequence for positioning.** "Better closure" is no longer a winning frame — a better closure of
a Gaussian posterior cannot beat an asymptotically exact sampler on the axis that matters. The
frame that survives is:

> **Measure when a closure suffices, and escalate only where it does not.**

That is recommendation #1 as the instrument, #2 as the cheap default, and #4 as the escalation rule.
It is also the only frame in which a cheap method can legitimately beat an exact one — on
cost-per-success, not on accuracy.

---

## 10. Suggested sequence

1. **Week 1** — Ship the completed-moment correction (#2). Re-run every comparison previously made
   against "SMG"; those used a defective baseline.
2. **Week 1** — Measure the soft-versus-decoded guide gap (#5 gate). One pass over existing samples.
   Decides whether DBFG stays alive at all.
3. **Weeks 2–3** — Build the κ₃ probe (#1) and run the decomposition experiment: how much guidance
   error is property-law vs posterior-shape, on the real checkpoint, split by measured |γ|. **This
   is the go/no-go for the whole programme.**
4. **Week 3** — QPMG (#3) enters only if step 3 says property-law error is ≥ 30% of the total.
5. **Week 4** — Closure-residual escalation (#4) tested against all three cheap signals.

Fix the ceiling on total configurations and GPU hours before step 3. Freeze hyperparameters before
test evaluation. Every fallback, numerical failure and discarded candidate stays in the accounting.

---

## 11. Claim ledger for this document

**Proved.** The tilted-mean form of the exact guidance and its conversion to diffusion and flow
updates. `Σ = (σ²/α)J` and `∂Σ/∂x = (α/σ²)κ₃` as exponential-family consequences (established, not
ours). That `Var[q] = gᵀΣg + ½tr((HΣ)²)`, so SMG's denominator is not the second moment. That QPMG's
mean-derivative is the exact Gaussian-posterior Stein form for quadratic `f`.

**Numerically checked** (synthetic, exact reference, ideal denoiser). Both identities to 1e-8 or
better. The SMG deficiency and its repair across t and s. QPMG's counterexample to 12 digits, the s²
double-count, and the Gauss-Hermite failure at small s. QPMG ≈ or worse than the completed-moment
closure on mixture posteriors. Hutchinson K = 1 sufficiency in d = 16. MAG's tilted means and the
exact-zero collapse of all covariance-only fields.

**Negative findings, reported as such.** The third-cumulant closure fails on multimodal posteriors.
Skew as a gate loses to the time index. DBFG's facet estimator gives ≈1.2–1.3× RMS over a
budget-matched score-function estimator, only at tight proposals.

**Hypothesised, untested.** That the decomposition holds on a learned denoiser. That |γ| survives
partialling out t and spread. That closure-residual escalation works. That DBFG's in-cell analytic
coordinate term is where its advantage lies.

**Unresolved.** Whether any of this changes terminal molecular samples. Whether novelty claims
survive full citation tracing. All wall-clock and memory figures. Whether the frozen guide is
accurate enough for closure differences to be measurable at all — if guide error exceeds closure
error, none of this matters.

**Explicitly not claimed.** No molecular result. No method here is first of its kind. Field accuracy
does not predict sample quality — the checks show it often does not.
