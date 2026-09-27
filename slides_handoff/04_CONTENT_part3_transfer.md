# Part 3 content: slides 3a to 3e

Closes the story. 0.95 presentation points, and 3d and 3e are where honest limitation
analysis is rewarded.

---

## 3a. Explain the transferred method  (1 slide)

**Title:** The transferred method

**What stays the same.** The *whole controller*: F̄, V_b, the error e, the setpoint τ,
the gain η, the bound w_eff ≥ 1 − η, and the reduction. **Transferred with zero code
change.**

Why it can: BDG's step is a *per-sample scalar* times the plug-in step, so it inherits
that step's tangency and needs **no covariance matrix and no inverse**. That matters
here, because the natural simplex covariance diag(p) − ppᵀ is **singular**.

**What must change.**
- State space: R^(N×3) → (Δ³)^L.
- Generator: EGNN → a linear-path flow model on the simplex.
- Property: a trained equivariant network → an *analytic* differentiable relaxation of
  GC content.
- Evaluator: f_B → an *exact* GC count.
- Decoder: none → per-position argmax.

**Training procedure (M2).** Linear-path flow matching on (Δ³)^L, trained on all 6,126
Kenyon-cell sequences, validation-based early stopping, checkpoint selected on
validation loss. Architecture and budget: `[TODO: v3 Job B]`.

**Sampling procedure (M2).** Same solver family and step count as M1, guidance applied
on the same window, same velocity-relative trust region, decode by per-position argmax.
The clip fired on **0–3%** of steps, and the expanding cell clipped **once in 25,600**.

**Takeaway line.** The transfer tests the *controller*, not the property network. That
is the point of changing the property from a learned net to an exact function.

**Sources:** [3] MOG-DFM data. [4] Dirichlet FM. [5] Fisher Flow.

---

## 3b. Present the transfer results  (1 slide)

**Title:** Transfer results

**Table (§4.6).** Transfer to DNA enhancers on the probability simplex. Signatures are
exact statements about the controller.

| Structural signature | QM9 | DNA simplex |
|---|---|---|
| η = 0 reproduces the plug-in rule | bit-identical | **bit-identical** |
| one-sided at an expanding setpoint | exact | **exact**, w_eff = +1.000 |
| setpoint ladder monotone in spread | 0.67 → 1.19 | 0.81 → 1.01 |
| expansion reached | 1.168 | 1.006 |
| cost identical to the plug-in rule | yes | **yes**, byte-identical |
| contraction breaks the fidelity floor | yes | **yes** |
| in-band improves | **no** | **no** |
| *Against recent simplex sequence models* | | |
| Dirichlet FM [4] | `[TODO: v3 Job B]` | |
| Fisher Flow [5] | `[TODO: v3 Job B]` | |
| MOG-DFM [3] | `[TODO: v3 Job B]` | |
| **BDG, transferred (ours)** | `[TODO: v3 Job B]` | |

**Fair study: this is not the published enhancer benchmark, and we say so.** The
standard protocol is class-conditional generation at **500 bp** on 81-class / 47-class
sets of about 90,000 sequences, scored by FBD over 10,000 samples. We use **GC-band
coverage**, **6,126** sequences and **200 bp**. Our base generator *is* the linear-FM
baseline these papers report, so it is a named baseline, but the numbers are not
comparable and are never pooled.

**Three further disclosures.**
- Gumbel-Softmax FM [25] does not run the enhancer benchmark at all, so only **two** of
  the three closest named methods apply.
- Fisher Flow [5] disputes the standard evaluator, reporting the cell-type classifiers
  behind it score **11.5% / 11.2%** test accuracy.
- MOG-DFM's own guided demo runs at length 100 while reporting base FBD at 500 bp.

**Sources:** [3], [4], [5], [25].

---

## 3c. Does the innovation transfer?  (1 slide)

**Title:** Does the innovation transfer?

**Direction: yes, on every signature we can check.** The gates hold exactly, the ladder
is monotone, expansion is reachable, the cost is unchanged, and the *null* transfers
too: spread control does not raise band coverage in either state space, and in both it
is the *contraction* rung that breaks the fidelity floor.

**Magnitude: 2.6× smaller**, a span of 0.20 against 0.52 in achieved spread. We
attribute that to the *property map*, not the controller, and we measured all three
causes rather than assuming them:
- **GC is affine** in the simplex coordinates, so its gradient is identical at every
  position and carries *zero curvature*; the simplex renormalisation then partly undoes
  the edit.
- **GC is a 200-position average**, which suppresses the very variance the controller
  reads.
- **The band spans only 4.3 attainable GC values** at granularity 1/200. That is a
  ceiling on what *any* method could move.

We ruled out the trust region: it fired on 0–3% of steps, and the expanding cell
clipped once in 25,600.

**Takeaway line.** The evidence supports transfer of the *mechanism and its limits*,
and locates the magnitude difference in a measurable property of the task. The lever
for a larger effect is a **richer property**, for example a trained cell-type classifier
logit, used as classifier *guidance* rather than conditioning.

**Source line:** Original analysis (this work).

---

## 3d. Limitations and failure modes  (1 slide)

**Title:** Limitations and failure modes

**1. The trust-region clip is a live confound at large |w_eff|.** The plug-in rule clips
on **701** sample steps where BDG clips **4,457**. At w_eff of order
10³ the dispersion term sits far outside the trust region, so what reaches the sample is
the *clip's truncation* of it. A rung can be **clip-limited** rather than
**controller-limited**, and no metric in this study separates the two.
*Fix:* a run at matched clip activity.

**2. The trade-off is re-parameterised, not removed.** The chemistry cost tracks
|w_eff|, not the *direction* of the knob, so strongly contracting and strongly expanding
rungs both cost chemistry. This is a descriptive correlation over coupled cells, not a
test.

**3. The batch is the estimator.** V_b is computed over whatever tensor the sampler is
handed, so running n samples in several batches runs *several independent controllers*.
Our headline batch of 500 is a memory constraint, not a modelling choice, and we
disclose it wherever n exceeds it.

**4. The flow has a dispersion drift of its own.** The across-molecule variance
*contracts and then re-expands* on *unguided* runs, so a setpoint on V_b is fighting a
moving quantity.

**Takeaway line.** Two of these are measured confounds, not speculation. They are the
reason the controller claim is stated as provisional in 3e.

**Source line:** Original analysis (this work).

---

## 3e. Conclude and identify future directions  (1 slide)

**Title:** Conclusion and future directions

**Back to the opening hypothesis.** We asked whether making batch dispersion a
*requested* quantity raises band coverage. **It does not**, and we can say why rather
than only that.

**The strongest conclusion the two-modality evidence supports.** Servoing the
*deviation* gain of a property-gradient guidance term against a batch-dispersion
setpoint gives **monotone, two-sided, modality-portable** control of the generated
batch's spread at **no additional sampling cost**, and converts the
coverage-against-fidelity trade-off into a quantity the user *requests* rather than
removing it.

**Mechanism, and the trade-off.** The per-sample coefficient is *affine* in F_i, so BDG
is the plug-in field at a rescaled weight aiming at a shifted target: it cannot reach
outside that field's reachable set. What it changes is *which point* of that set the
sampler lands on, and *how* it gets there. Because the centring gain stays at w while
only the deviation gain is servoed, BDG buys contraction from the deviation mode rather
than by scaling the whole field, which is why it retains more chemistry at matched
coverage.

**Specific next experiments.**
- **The signed-w plug-in control**, with our pre-registered drop rule: if it reaches
  BDG's spread at equal chemistry, we withdraw the controller claim and report the
  decomposition instead.
- **A matched-clip-activity run**, to separate clip-limited from controller-limited
  rungs.
- **A second seed pair** for the open-loop replay.
- **A richer M2 property** (cell-type classifier logit, nonlinear and motif-driven) to
  test whether the magnitude gap is the property map.

**Source line:** Original conclusions (this work).
