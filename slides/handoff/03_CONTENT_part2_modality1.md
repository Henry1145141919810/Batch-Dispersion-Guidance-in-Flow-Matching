# Part 2 content: slides 2a to 2e

This is the scoring centre of the deck: 2.25 of the 4.5 presentation points. Every
number is measured. Read `07_DO_NOT_SAY.md` before editing any wording in 2c or 2d.

---

## 2a. Choose the base model  (1 slide)

**Title:** Choose the base model

**Table (§4.2).** Base models on QM9, unconditional, 100-step Euler probability-flow
ODE, our evaluator under EDM conventions.

| Model | Source | Atom stab. ↑ | Mol. stab. ↑ | Validity ↑ | Uniq. ↑ | NFE ↓ | Params |
|---|---|---|---|---|---|---|---|
| Flow matching (ours) | reproduced | **0.9366** ±.0012 | **0.3993** ±.0085 | **0.7625** ±.0031 | 0.9939 | 100 | 3.75M |
| VP diffusion (ours) | reproduced | `[TODO: v3 Job A]` | `[TODO]` | `[TODO]` | `[TODO]` | 101 | 3.75M |
| `EDMsecond` [2] | borrowed ckpt. | 0.9644 | 0.6725 | 0.8490 | `[TODO]` | 100 | 5.34M |
| EDM, published [2] | quoted | 0.9870 | 0.8200 | 0.9190 | — | 1000 | — |
| EDM, unscaled one-hot [2] | quoted | 0.9570 | 0.4690 | — | — | 1000 | — |
| Real QM9 (ceiling) | measured | 0.994 | 0.956 | 0.982 | — | — | — |

> ⚠️ **The VP row's numbers are available (28 Sep) but are NOT pasted here,
> because this table's FM row does not match the canonical run.** From
> `results/v3/fm/v3/n2000/` the unguided pooled values are atom **0.9356**,
> mol **0.3970**, validity **0.7562**; this table says 0.9366 / 0.3993 / 0.7625.
> Reconcile the FM row first — otherwise a VP row from the canonical cells would
> sit beside an FM row from somewhere else and the comparison would be invalid.
>
> The canonical VP values, for when that is settled: atom **0.9070** ±.0013,
> mol **0.2883** ±.0134, validity **0.6617** ±.0070, uniqueness **0.9995**,
> NFE 101, 3.75 M params. Source:
> [V3_RESULTS_vp.md](../../docs/results/V3_RESULTS_vp.md) and
> [DATA_INDEX.md](../../docs/results/DATA_INDEX.md) §3.

*Our flow row is three seeds of 10,000 samples. `EDMsecond` is a single-seed gate run
at n = 2,000, so it is not directly comparable.*

**Which model is carried forward, and why.** **Flow matching** -- and as of 28 Sep
the reason includes a **measured fidelity win over the matched diffusion model**. Our
own VP diffusion ran under the identical protocol and lost on all three chemistry
metrics (mol stab 0.3970 vs 0.2883, validity 0.7562 vs 0.6617, atom stab 0.9356 vs 0.9070; every gap 7-14x the sd of the difference). The borrowed `edm` checkpoint still cannot decide the question (it differs
in architecture, training half and feature scaling); `vp` can, and does. We also carry
flow matching because its linear path gives a **closed-form endpoint estimate at every
t**,
which is the object every guidance rule differentiates, at NFE 100 against a 1000-step
published protocol.

**Our stability gap, attributed in rank order from our own measurements:**
unscaled one-hot leads (the published ablation removing exactly that scaling costs 3
atom and 35 molecule points); then training budget (303k steps vs ~1.56M); the
half-data split is **ruled out** (0.33 atom points); the evaluator is **ruled out**
(bond tables match entry-for-entry).

**Sources:** [2] EDM, Tables 1 and 10.

---

## 2b. Guidance strategy and result  (2 slides)

### Slide 2b-i: the guidance strategy

**Rule and target.** A frozen property predictor f_A is differentiated at the endpoint
estimate m_i, and the gradient is pulled back to the sampling variable:
```
G_i = ( num_i / s² ) · J_iᵀ g_i ,     num_i^plug = y − F_i ,
F_i = f_A(m_i),   g_i = ∇_m f_A(m_i)
```
Target y is the **q50** value of the property, one fixed number per property. This is a
DPS-style plug-in Gaussian-energy field [12].

**Trained / pretrained / frozen.**
- Generator v_θ: trained by us, **frozen** at sampling.
- Guide f_A: trained by us on `train_a`, **frozen**, `eval()`, gradients disabled.
- Evaluator f_B: trained on `train_b`, **frozen**, and it *never participates in
  sampling*.
- **Nothing is trained at guidance time.**

**Strength schedule, normalisation, clipping.**
- Correction converted by w·(1−t)/t.
- Denominator s² with s = f_A's property standard deviation.
- **Velocity-relative trust region**: G_i rescaled so ‖G_i‖ ≤ ‖v_i‖, per sample, after
  w. Prior practice [13], shared by every arm.
- **Window:** guidance on for t ≥ 0.5 only.

**Takeaway line.** The trust region is load-bearing, not cosmetic: removing it produces
**222 non-finite samples** where the clipped run produces **0**.

**Sources:** [12] DPS. [13] trust-region practice. [14] CFG.

### Slide 2b-ii: guidance works, and what it costs

**Table (§4.3).** Guided against unguided at q50, otherwise matched. One generator, one
seed, n = 256, best strength per arm over w ∈ {0.05,…,32}, no chemistry floor. Each
cell is **μ / gap / α**.

| Arm | in-band ↑ | Spread / unguided | Mol. stab. ↑ |
|---|---|---|---|
| unguided | .090 / .121 / .051 | 1.00 / 1.00 / 1.00 | .434 / .434 / .434 |
| plug-in (best w) | .184 / .219 / **.133** | 0.52 / 0.60 / 0.47 | .246 / .199 / .320 |
| TFG (w = 4, μ only) [15] | **.535** / – / – | 0.22 / – / – | .172 / – / – |
| BDG (w = 32, τ_mult = 0.5) | .180 / **.227** / .106 | 0.49 / 0.68 / 0.53 | **.348** / .262 / .340 |

*Three-seed confirmation: `[TODO: v3]`.*

**The target property improves.** 2.0× on μ, 1.9× on the gap, 2.6× on α for the plug-in
rule; TFG reaches 6.0× on μ.

**The cost is real and we report it.** Molecule stability falls from 0.434 unguided to
0.246 (plug-in, μ) and to 0.172 (TFG). Validity falls with it.

**The mechanism is visible.** Spread contracts from 1.00 to 0.22–0.60. Guidance works
*by concentrating the batch*, and the arm that contracts most wins in-band and loses the
most chemistry.

**Takeaway line.** That trade-off is inherited, not chosen. 2c asks whether the user
can choose it instead.

**Sources:** [15] TFG. [12] DPS.

---

## 2c. Introduce the innovation  (3 to 4 slides)

### Slide 2c-i: motivate the innovation

**The weakness.** Coverage is a *dispersion* problem at a centred target. Measured:
guided batches sit 1–2 band widths from the target while their spread is 5–9 band
widths. Yet every guidance rule optimises a *centring* objective and exposes *one*
dial, w, which moves centring and spread together in a ratio the user never chooses.

**The hypothesis.** If the sampler *measures* its own batch dispersion and servoes
toward a *requested* value, the user trades spread against centring deliberately, in
the units of the property rather than in units of push.

**How this differs from prior work.** Each ingredient separately exists, and we claim
none of them:

| Prior art | What it already owns |
|---|---|
| CFG-Ctrl [16], Feedback Guidance [17] | the guidance scale read as a control gain |
| MGD [18] | empirical ensemble moments driven to a target, two-sided |
| Variance-Tilted Diffusion [19] | batch spread targeting, widen-only, fixed linear feature |
| Particle Guidance [20] | batch-coupled repulsion, no setpoint |
| Negative-guidance window [21] | negative weight widens the distribution |

**What we did not find** in a bounded search (about eight queries, abstracts only,
25 Sep 2026): a scalar guidance weight set by the batch variance of a *learned property
predictor's own endpoint prediction*, against a *two-sided user-chosen* setpoint,
*signed so the weight passes through zero*. That bounds our claim; it is not a priority
claim.

**Sources:** [16]–[21] as listed in `05_BIBLIOGRAPHY.md`.

### Slide 2c-ii: specify the innovation, part 1 (mechanism)

**Two batch statistics and a scale-free error** (per guided step, batch B):
```
F̄  = (1/B)·Σ_i F_i        V_b = (1/(B−1))·Σ_i (F_i − F̄)²        e = (V_b − τ²)/τ²
```

**The rule** (box this; it is the centre of the talk):
```
num_i = (y − F_i) − η·e·(F_i − F̄)
      = (y − F̄) − w_eff·(F_i − F̄) ,        w_eff = 1 + η·e
        └ centring, gain w ┘   └ deviation, gain w_eff ┘
```

**Why it is a descent direction on the thing that binds.**
∂V_b/∂F_i = (2/(B−1))(F_i − F̄), and the chain rule carries it to m_i along g_i. The
2/(B−1) is absorbed into η so the gain does not scale with B.

**Signs are the whole mechanism.** V_b > τ² ⇒ contract; V_b < τ² ⇒ expand; V_b = τ² ⇒
the term vanishes and BDG is exactly the plug-in rule.

**Cost: zero extra NFE.** Both terms multiply the same g_i, so BDG shares the one
backward pass the plug-in rule already performs. Cost accounting is byte-identical.

**Trainable: nothing.** Generator and f_A stay frozen. **New hyperparameters: two**,
the gain η and the setpoint τ = τ_mult · s, a fraction of the guide's own property
standard deviation.

**Source line:** Original formulation (this work). Plug-in base after [12].

### Slide 2c-iii: specify the innovation, part 2 (the reduction and its bounds)

**We state the reduction ourselves.** num_i is affine in F_i, so it factors exactly:
```
(y − F_i) − η·e·(F_i − F̄) = (1 + η·e)·( y_eff − F_i ) ,   y_eff = (y + η·e·F̄)/(1 + η·e)
```
verified to **3.0e-15** analytically and **6.5e-7** on the real field.

**What that means.** At any instant BDG is the *plug-in field at a rescaled weight
aiming at a shifted target*. It adds no new direction, and any guidance whose
per-sample coefficient is affine in F_i is the plug-in rule reweighted.

**What it does not remove.** w_eff is set *online* from a measured quantity and may
*change sign*, which w cannot do without also reversing the centring term.

**Bound.** V_b ≥ 0 ⇒ e ≥ −1 ⇒ **w_eff ≥ 1 − η**, for any batch. That bound is what
licenses running two-sided with no clamp.

**Design constraint: η > 1**, or w_eff never reaches negative values and the controller
cannot expand at all.

**Equilibrium, stated honestly.** The law settles where w_eff = 0, that is
V* = τ²(1 − 1/η). That is an **asymptote the 50-step window never reaches**, so we never
say the controller holds its setpoint, and we order every table by *measured* w_eff.

**Source line:** Original formulation (this work).

### Slide 2c-iv: the controller, measured  (figure slide)

**Title:** The controller, measured

Full-width: `figures/fig2_mechanism.png` (or the PDF).

**Spoken content.** (a) The sign of e is the whole mechanism, and e = 0 recovers the
plug-in rule exactly. (b) The requested setpoint maps monotonically onto achieved
spread on **3/3** properties, and crosses *above* the unguided spread on all three,
which the strength dial does not reach. (c) The measured deviation gain **passes
through zero**, staying inside its bound w_eff ≥ 1 − η. A single strength dial cannot
do that without reversing the centring term as well.

**Source line:** Original figure (this work), generated by
`paper/figs/make_mechanism.py` from `results/bdg_port/table.txt`
(64 cells, n = 256, seed 20260925).

---

## 2d. Test the innovation  (2 slides)

### Slide 2d-i: the ablation table

**Table (§4.4).** BDG ablations at q50, ordered by *measured* w_eff, never by τ_mult.
Port cells: n = 256, one seed, w = 1, own unguided control.

| Variant | Role | η | τ_mult | Spread / ung. | in-band ↑ | Mol. stab. ↑ |
|---|---|---|---|---|---|---|
| plug-in, w = 1 | base | – | – | 0.714 | 0.1094 | 0.3633 |
| BDG η = 0 | **gate** | 0 | 1.0 | 0.714 | 0.1094 | 0.3633 |
| BDG one-sided, asked to expand | **gate** | 4 | 1.5 | 0.714 | 0.1094 | 0.3633 |
| BDG contract | ladder | 4 | 0.5 | 0.670 | 0.1406 | 0.3359 |
| BDG neutral | ladder | 4 | 1.0 | 0.900 | 0.0820 | 0.4258 |
| BDG expand | ladder | 4 | 1.5 | 1.093 | 0.0703 | 0.4141 |
| plug-in, w = 4 | **control** | – | – | 0.584 | 0.1406 | 0.2656 |
| open-loop e replay, fresh noise | **control** | 4 | 0.5 | recovers 69.5–96% of the closed loop | | |
| signed-w plug-in | **control** | – | – | `[TODO: not run; drop rule applies]` | | |
| three-seed v3 grid | all | `[TODO: η ∈ {0,1,2,4,8} × τ_mult ∈ {0.5,0.75,1,1.5}, w ∈ {1,4}]` | | | | |

**Gates prove the implementation; they are not controls.** η = 0 is bit-identical to
the plug-in rule, so the dispersion term is the only addition. The one-sided variant
asked to expand is also bit-identical (raw e = −0.984 clamped to 0), so the sign branch
is real and the clamp is self-consistently inert.

**Source line:** Original table (this work).

### Slide 2d-ii: conclusion from the ablations

**What each comparison isolates.**

| Comparison | What it rules out |
|---|---|
| η = 0 vs plug-in | that anything other than the dispersion term changed |
| one-sided vs two-sided | that the expansion branch is decorative |
| the τ ladder | that the setpoint does not reach achieved spread |
| plug-in at w = 4 | that the coverage came from *controlling* rather than *pushing* |
| open-loop e replay | that the effect is feedback rather than a time-varying schedule |
| signed-w plug-in `[TODO]` | that the same spread is not available more cheaply |

**What the evidence supports.**
- **The setpoint controls spread**, monotonically, on **3 of 3** q50 curves: 0.670 → 1.187
  on the guide's scale, 0.650 → 1.197 on the evaluator's, reaching *expansion above 1.0
  on all six*, which the strength dial does not reach. w_eff changes sign up to **34
  times** in the 50-step window and spans [−2.64, +10.35] against the bound w_eff ≥ −3.
- **Coverage does *not* improve.** No rung beats plug-in at |z| ≥ 3: over **72 paired
  tests** the largest is 3.07, which a Šidák correction turns into **p = 0.14**.
  Removing the chemistry floor does not change this.
- **The trade-off, not the coverage, is what moves.** At *matched* in-band of 0.1406,
  BDG keeps molecule stability **0.3359** against plug-in's **0.2656**.

**Takeaway line.** Honest reading: the schedule largely *transfers* across seeds
(69.5–96% recovered open-loop) and feedback adds a small measurable remainder
(0.3–6.6%). We do **not** claim the loop is irreducible to a schedule.

**Source line:** Original analysis (this work).

---

## 2e. Position against recent work  (2 slides)

### Slide 2e-i: the benchmark table

**Table (§4.5).** Five inference-time guidance methods from the last five years, all
reproduced inside our pipeline on our frozen checkpoint, guide and evaluator, at q50
and w = 1. Passes are per cell at n = 5,000.

| Method | Status | in-band ↑ | Mol. stab. ↑ | Diversity ↑ | Gen. passes | Guide passes |
|---|---|---|---|---|---|---|
| DPS-style plug-in [12] | reproduced | `[TODO: v3]` | `[TODO]` | `[TODO]` | 10,000 | 4,000 |
| TMPD-inspired [22] | reproduced | `[TODO: v3]` | `[TODO]` | `[TODO]` | 10,000 | 4,000 |
| LGD-MC [23] | reproduced | `[TODO: v3]` | `[TODO]` | `[TODO]` | 12,000 | 20,000 |
| TFG [15] | reproduced | `[TODO: v3]` | `[TODO]` | `[TODO]` | 8,000 | 20,000 |
| D-Flow [24] | reproduced | `[TODO: v3]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |
| Flow matching, unguided | internal ref. | `[TODO: v3]` | `[TODO]` | `[TODO]` | 10,000 | 0 |
| **BDG (ours)** | reproduced | `[TODO: v3]` | `[TODO]` | `[TODO]` | 10,000 | 4,000 |

**Why these five.** All are inference-time guidance published in the last five years,
all steer a frozen generator with a frozen predictor, and all run inside our pipeline on
*our* checkpoint. That isolates the **guidance rule** from the generator, which is what
our contribution is about.

**Efficiency, measured not inferred.** BDG's two terms share one backward pass:
**4,000** guide passes per cell against **20,000** for LGD-MC and TFG, a fifth of their
guidance cost at the same generator budget. BDG is *not* the cheapest overall: TFG uses
fewer generator passes.

**Sources:** [12], [22], [23], [15], [24].

### Slide 2e-ii: fair study, and fair discussion

**Reproduced versus quoted.** Every row in the 2e table is **reproduced by us** on one
frozen checkpoint, one guide, one evaluator, one target, one sample count. The only
*quoted* numbers anywhere in this study are the published base-model rows in 2a, and
they are labelled as quoted and never pooled with ours.

**Three port caveats we disclose rather than bury.**
- Our plug-in arm is a *DPS-style* Gaussian-energy field; the original DPS
  differentiates the residual L2 norm [12].
- Our TMPD arm is the *local-linear scalar limit* of the published method, included as
  a related-work comparison, not a faithful nonlinear port [22].
- TFG runs at its published QM9 hyperparameters but through *our* deterministic ODE
  sampler, where the original used a stochastic DDIM update; our unscaled one-hot
  features also move about 16× less relative to coordinates than the released
  checkpoint's. **These numbers must never be read against TFG's published table** [15].

**Where we are better, comparable, and worse.**
- **Better:** same guide-call pattern as the plug-in rule, a fifth of LGD-MC's. Do NOT
  claim chemistry at matched coverage: that difference is z = 1.73 on one seed, and both
  cells fall below our own chemistry floor
  and TFG's guide passes.
- **Comparable:** in-band against the plug-in rule, which is the honest reading of the
  ablation null.
- **Worse:** TFG reaches far higher in-band (0.535 vs 0.180 on μ) at a chemistry cost we
  judge unacceptable (0.172 vs 0.348), and it uses fewer generator passes.
  `[TODO: v3 completes this comparison]`

**Sources:** [12], [22], [15].
