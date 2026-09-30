# Plan and timetable — what we test, what runs when, and who is blocked on what

> **Historical (18 Sep plan).** The innovation became BDG and the protocol became v3 on
> 26 Sep; v3 finished on blade 27-28 Sep. Current state and results:
> [SCOPE_FM_GUIDANCE_STATUS.md](SCOPE_FM_GUIDANCE_STATUS.md) and
> [V3_FINAL_SUMMARY.md](../results/V3_FINAL_SUMMARY.md). Only the two stage rows
> updated 28 Sep below are current.

**18 September 2026.** Written for the team. Deadline **29 Sep 08:30**, defence **30 Sep** —
11 days. This is the schedule and the shape of the work; the mathematics lives in
[SMG_WHY_INNOVATIVE.md](../methods/SMG_WHY_INNOVATIVE.md) and [D_WHY_INNOVATIVE.md](../methods/D_WHY_INNOVATIVE.md),
and the live state of every run is in [STATUS_LIVE_RUN_LOG.md](STATUS_LIVE_RUN_LOG.md).

---

## 1. The shape of the project

The brief asks for two continuous modalities; for Modality 1, one flow-matching **and** one
diffusion model, a demonstration that guidance works, a **methodological innovation** with an
ablation, and a comparison against at least three recent methods; then the innovation transferred
to Modality 2.

We are doing that with:

| | Choice |
|---|---|
| **Modality 1** | QM9 — 3D molecules, property-targeted generation |
| **Modality 2** | DNA sequences on the probability simplex |
| **Base models** | flow matching (linear interpolant) and VP diffusion, both on a shared E(n)-equivariant GNN |
| **Primary innovation** | **SMG** — Second-Moment Guidance |
| **Safety net** | **D** — the Tolerance Band |
| **Comparators** | plug-in guidance, ABMS Monte-Carlo marginalisation, matched-energy GGDOpt |

**We run one primary and one safety net, not five ideas.** Everything else we considered — A, B,
C, TRE, BPS, PCG — is catalogued in [INNOVATION_IDEAS_INDEX.md](../methods/INNOVATION_IDEAS_INDEX.md) with
its novelty assessment, and deliberately not being run. There is not enough time to do three
things properly, and two things done properly is the whole deliverable.

### Why one primary *and* a safety net

SMG and D fail for different reasons, which is the point. SMG asks **is there a Jensen gap worth
correcting?** D asks **is there unused tolerance worth spending?** A negative answer to one says
nothing about the other. If SMG's gate comes back negative on 19 Sep, D is already scoped, proved
and coded, and the project still has an innovation to ablate rather than a hole.

---

## 2. Where we are — stage 2 of 8

| # | Stage | State |
|---|---|---|
| 1 | Data, property predictors, symmetry checks | **done** |
| 2 | Two base models | **DONE 28 Sep.** FM and our own VP diffusion both run under v3, matched on backbone, parameters, epochs, batch, EMA, split and seed. FM wins on all three chemistry metrics (mol stab 0.3970 vs 0.2883, validity 0.7562 vs 0.6617, atom stab 0.9356 vs 0.9070). v3 has **two** diffusion backends: `vp` (ours) and `edm` (TFG's borrowed EDMsecond) |
| 3 | Evaluation harness (S2) | **written and calibrated**, one run queued |
| 4 | SMG gates (M-0, M-1) | code ready, runs after S2 |
| 5 | SMG arms + 3 comparators | ablations coded; comparators to write |
| 6 | Euler/Heun solver control (M-4) | code ready |
| 7 | D (D-1 … D-3) | proved and specified; controller to wire in |
| 8 | Modality 2 + write-up | base model **done**; guidance sweep **run on blade 27-28 Sep** under [MODALITY2_V3_PROTOCOL.md](../protocol/MODALITY2_V3_PROTOCOL.md), results in [M2_V3_RESULTS.md](../results/M2_V3_RESULTS.md); write-up is paper v4 |

**Done and checkable now:** 133,885 QM9 molecules processed with a four-way split; guide $f_A$ and
evaluator $f_B$ trained on **disjoint halves** at 0.0897 and 0.0840 D validation MAE against a
1.53 D chance baseline; 12/12 symmetry checks; 9/9 guidance and solver correctness checks; the
flow model trained 150 epochs in 2 h 40 m.

---

## 3. What gets tested

Five gates. A gate is a cheap run whose result decides whether a component continues — not a
result for the paper.

| Gate | The question it answers | Kills what, if it fails |
|---|---|---|
| **S2** | do our metrics mean anything, and what is the tolerance $\delta$? | everything downstream — no metric, no comparison |
| **M-0** | are SMG and plug-in consistent under a change of property units? | an implementation bug, not the idea |
| **M-1** 🔴 | is the Jensen gap real, signed as predicted, and bigger than $\delta$? | **SMG** |
| **D-1** 🔴 | does the tolerance band ever activate, and are its edits accepted? | **D** |
| **Simplex** 🔴 | can a covariance-style step live on the simplex without going negative? | **Modality 2 transfer** |

The three marked 🔴 are real decisions. The other two are correctness.

After the gates, the experiment proper: M-2 (ablation arms and the three comparators), M-3 (a
correct conditional reference), M-4 (Euler vs Heun, so a solver artefact cannot be mistaken for
an estimator effect), and D-2/D-3.

---

## 4. Timetable

**Machine** is unattended cluster time. **Focus** is someone at a keyboard. The cluster caps every
job at 4 hours, so long runs are chains of resumable jobs — the trainers already support this.

| Date | Runs on the cluster | Focus needed |
|---|---|---|
| **18 Sep** tonight | diffusion (2 h 40 m); S2 harness; guidance tests | 10 m submitting |
| **19 Sep** am | M-0 + M-1 (2.5 h) ∥ D-1 (2 h) | 10 m submitting |
| **19 Sep** pm | simplex feasibility check (2 h) | **30 m — M-1 verdict** 🔴 · **15 m — D-1 and simplex** 🔴 |
| **20 Sep** | M-2 arms (3 h), M-3 reference (1.5 h) | 45 m reviewing |
| **21–22 Sep** | M-4 crossed design (6 h, chained); D-2 | 45 m reviewing |
| **23–24 Sep** | D-3; Modality 2 runs | 1 h reviewing |
| **25–26 Sep** | reruns and buffer | scope calls |
| **26 Sep** | — | **RESULTS FREEZE, end of day** |
| **27–28 Sep** | — | **write-up, ~5 h** |
| **29 Sep 08:30** | submit | |
| **30 Sep** | defence | 2 h prep |

**Totals: ~35 machine-hours (~18 h wall-clock at two jobs in flight), ~12 focus-hours.** Nearly
half the focus time is the write-up, which cannot be parallelised or automated.

The two days of slack at 25–26 Sep are not spare capacity. Queue waits are unpredictable and at
least one gate will probably send us back a step.

---

## 5. How the decisions get made

Each gate has its outcome table written **before** the run, so the result cannot be rationalised
after the fact.

**M-1 — the SMG gate.** We measure the gap between the property of the predicted mean and the mean
property, and ask whether the predicted correction $\tfrac12\operatorname{tr}(H\Sigma_t)$ accounts
for it.

| Outcome | Decision |
|---|---|
| Gap signed as predicted and comparable to $\delta$ | **proceed** — this is the whole case |
| Gap real but far below $\delta$ | **stop.** Correct arithmetic, irrelevant effect. Report the measurement, promote D |
| The correction does not track the measured gap | curvature too large for the leading term — add a term or stop |
| Sign does not follow the curvature | investigate the covariance estimate before believing anything |

**D-1 — the D gate.** We log band activity, active constraints, accepted edits and rejection rate.

| Outcome | Decision |
|---|---|
| Band active, edits accepted | **proceed to D-2** |
| Band never activates | **not a null result** — re-choose $\tau$ against the measured error distribution and retest |
| Nearly every edit rejected | the controller is inert; report the curvature finding and stop |
| Edits accepted, diversity unchanged | the objective or embedding is wrong, not the band |

**If SMG fails and D survives**, D becomes the primary, M-2/M-4 are rewritten around it, and we
lose roughly a day. **If both fail**, we report two honest negative results with the measurements
that produced them, plus two working base models and a calibrated harness — which is a weaker
paper but an entirely defensible one. That is why both gates are on the 19th and not the 25th.

---

## 6. Standing rules

- **Every number that judges a result comes from $f_B$**, the evaluator trained on the half of the
  data the guide never saw. $f_A$ is what guidance optimises; using it to grade would be marking
  our own homework. The $f_A - f_B$ gap is reported everywhere as the reward-hacking signature.
- **$\delta$ is derived, not chosen.** It comes from $f_B$'s own validation error (currently
  **0.168 D** = 2 × 0.0840). Any override is stated in the write-up.
- **No compute on cluster login nodes.** Everything through the batch scheduler.
- **Negative results get reported**, with the measurement that produced them. Three of the five
  gates exist specifically to produce cheap negative results early.

---

## 7. What I most want challenged

1. **Is the SMG claim narrow enough?** The prior-work audit found that TMPD and MMPS already
   contain SMG's formula for *linear* properties, and that covariance-aware guidance, the
   Jacobian identity and Heun are all established. What survives is the nonlinear mean correction
   — which vanishes when the property is affine. If that is too thin to carry the report, better
   to hear it now than on the 28th.
2. **Is D's delta real?** The band **is** the optimum set of a dead-zone loss when reachable, so
   the distinction rests entirely on the baseline-relative envelope $A=\max(\tau,\lvert e\rvert)$
   when the guide is outside tolerance. Section 7.3 of [D_WHY_INNOVATIVE.md](../methods/D_WHY_INNOVATIVE.md)
   is the argument; it is the part most likely to be wrong.
3. **Is Modality 2 scoped too small or too large?** Current plan is one small simplex model, SMG
   transferred, one ablation. Not a second full study.
