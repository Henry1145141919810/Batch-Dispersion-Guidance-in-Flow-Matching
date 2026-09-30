# Group 2 — what we're building, end to end

**Henry · 16 Sep 2026 (v2).** Read time ~10 minutes. The whole project in one pass. Everything here is a draft I'd rather have torn apart than politely accepted.

**Deadline:** code + paper + slides **Tue 29 Sep, 8:30 am**; defence Wed 30 Sep.

---

## 1. The task, in two sentences

Generate 3D molecules that hit a **target property value** — say a dipole moment of 2.5 D — without retraining the generator. The assignment wants two different continuous state spaces, so we do molecules (QM9) and DNA enhancer sequences on the probability simplex.

The standard approach is **guidance**: at each generation step, a frozen property predictor says "nudge this way to get closer to 2.5," and you add that nudge to whatever the generative model wanted to do.

## 2. The problem we attack

Ask for 64 molecules with dipole 2.5 and guidance gives you **64 near-copies**. But thousands of genuinely different molecules have that dipole. You wanted a portfolio of candidates; you got one candidate photocopied.

Worth knowing *why*, and being careful about it. The distribution guidance aims at leaves the "molecules with dipole exactly 2.5" slice **untouched** no matter how hard you guide — provable, and verified. But it *does* shift weight between slices, and molecules at different dipoles differ structurally, so even perfect sampling can lose variety overall.

So the honest version is narrower than I first wrote it: **guidance as practised may lose more within-condition variety than conditioning alone implies, and the experiment tests whether we can recover it.** That has to be measured, not assumed.

## 3. The idea

Two observations that turn out to be one method.

**(a) The target is a *surface*, not a point.** Every molecule with dipole 2.5 sits on the same "level set." So split the work: move **across** level sets to reach 2.5, and move **along** the surface to become different. Moving along it is free — the property doesn't change.

**(b) You don't need 2.5 exactly.** You need it within a tolerance the application sets, say ±0.1. Strict preservation throws that away. Sitting at 5.00 with [4.9, 5.1] acceptable, why block a move to 5.04 that might buy real structural variety?

So: **spread the samples apart, but only in directions that keep the property inside its accepted band.**

### One primary knob, three reference arms

τ is the main axis. A review caught me claiming it interpolates between two published endpoints — **it doesn't**, so we build the comparators explicitly:

| Arm | What it is |
|---|---|
| **Equality** | strict level-set preservation — the prior-work comparator |
| **τ = 0 band** | "don't make the error worse" — which is *not* the same as "don't change it" |
| **Unconstrained** | plain repulsion, built separately |
| **τ > 0 band** | the candidate |

It still **measures its own contribution**: if τ > 0 doesn't beat those three, our contribution is zero and we say so. Most projects can only succeed; this one can also produce a clean, publishable *no*.

**One trap worth knowing:** the constraint only bites when the current error is *inside* τ. If every sample is further off-target than every τ we try, all the arms are identical and a null result means nothing happened. So we log how often the tolerance is actually active.

## 4. The four components

| | Component | What it does | Is it new? |
|---|---|---|---|
| **A** | trust region | stops the nudge from overwhelming the model | **no** — OSCAR publishes this exact form |
| **B** | tangent projection | spread without changing the property | **no** — the maths is right and provable, but four papers own the general form. What's ours is two conditions on *using* it correctly |
| **C** | confidence gate | don't guide while the model hasn't decided what it's building | **no** — crowded category. Ablated, not claimed |
| **D** | **tolerance band** | spend the allowed error to buy diversity | **an adaptation** — the only one with a claim, and §9 says how big |

**Be clear-eyed about this:** A and B are prior work, C is engineering. Only D carries anything, and what's *proved* about D is local — the solver is exact and can't do worse than strict preservation at the same budget. **Nothing proves it makes better molecules.** That's what F4 is for.

**D is not just B with a bigger allowance** — at τ = 0 it becomes "don't get worse," not "don't change," so B has to be built as its own arm.

### How a step works

1. Take the **normal guided step** — unchanged, same as the baseline.
2. On some steps, add a small **diversity edit** on top:
   - which direction spreads this batch out,
   - which direction changes the property,
   - solve *exactly* for the best spreading direction that keeps the property in the band,
   - **actually apply it and re-check** — revert if it broke the band.

Step 4 is the safety net: we don't trust the linear approximation. A rejected edit reproduces the baseline exactly, so comparisons stay honest. The solve has a closed form (no optimiser in the loop) and is built so **doing nothing is always allowed** — it can never become unsolvable mid-generation.

## 5. What we can already prove

One result worth knowing, because it tells us in advance whether this can work at all.

Let **ρ** be the fraction of the "spread out" direction that points along the property direction. Then the benefit of projecting is

> **R · ‖b‖ · √(ρ(1−ρ))**

which is **zero at both extremes and largest at ρ = ½**:

| ρ | What happens |
|---|---|
| near 0 | nothing to remove — the projection does nothing |
| near 1 | the property direction carries *all* the diversity — projection removes everything |
| near ½ | **the working regime** |

Four lines of algebra, checked on 200,000 random cases. **And ρ is measurable in a couple of hours**, before any real experiment.

**But be careful what it means — a review corrected me on four points here, and it matters:**

- It measures the *size* of the property change avoided, **not** whether avoiding it was good. Sometimes the unprojected edit happens to move the property *toward* target. So we measure a signed quantity, not a magnitude.
- ρ near 1 does **not** condemn the method — that's exactly where having a tolerance rescues a direction strict preservation would block.
- ρ alone can't rank the three properties; the drift also scales with gradient size and units.
- So this is a **diagnostic**, not a prediction of the result.

## 6. Why it might fail — the four I actually worry about

Better listed up front than found in the defence.

| Risk | How we test it |
|---|---|
| **Simply loosening the property loss does the same thing**, with no geometry at all | a "dead-zone loss" control, no repulsion, from day one. **The most likely way this dies** |
| **Just generating more candidates and filtering wins** on wall-clock | generate-and-select at *equal wall-clock*, as a real baseline not a straw man |
| **ρ isn't in the working regime** on real molecules | F2 measures it directly, in hours |
| **We have no working generator yet**, and 14 days | the first gate is just "get the assets working." Nothing else means anything without it |

## 7. How we'll know — six gates

Cheapest first. Each can stop the project. `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md` has the detail.

| Gate | Question | Cost |
|---|---|---|
| **F0** | Do we have a working generator, guide and independent evaluator — and is the evaluator accurate enough to resolve the tolerance we're studying? | ~6 h machine |
| **F1** | Are our gradients actually correct? (finite differences, rotation, masks) | ~1 h |
| **F2** | **What is ρ?** Is there interference to remove, and structure left after removing it? | ~2 h |
| **F2b** | Does it matter which space we project in? | 30 min |
| **F3** | Does one edit work on one real batch? | ~2 h |
| **F4** | Does the τ sweep beat the controls, per attempt *and* per GPU-hour? | ~5 h |

Roughly **16 h of machine time**, nearly all unattended, and about **4 h of anyone's actual attention**.

## 8. What we measure

The headline number is deliberately practical:

> **Distinct valid molecules that land inside the property tolerance, per fixed number of attempts** — and the same count per GPU-hour.

Two things that are easy to get wrong and that we've fixed:

- **Diversity is measured *within* a single target condition**, then averaged. A global score across thousands of targets can look wonderful while every single condition has collapsed to one molecule.
- **The denominator is samples *attempted*, not *surviving*.** Otherwise a method with better validity gets compared on a different, self-selected population.

Alongside: property error under a **held-out** predictor that never touches generation, chemical validity, and — for a subsample — **actual quantum chemistry** (DFT at QM9's own level of theory). Worth having: it's the column nobody can argue with. *Not* a novelty, though — I claimed nobody does this and that was wrong; MolGuidance reports DFT on 500 generated molecules at the same level.

## 9. Honest novelty

**The general idea is not new.** "Diversity inside a constraint" is published — most directly [Harmless Diversity (ICML 2022)](https://proceedings.mlr.press/v162/gong22b.html), which maximises diversity inside the optimum set of a loss and even runs on molecular conformations. There's also a constrained-sampling literature (O-SVGD, Particle Guidance, OSCAR, Stein Diffusion Guidance).

**Two things I claimed and had to withdraw**, both caught on review:

- "Nobody verifies against physics" — false (see §8).
- "A tolerance band is different in kind from an optimum set" — false. The band **is** the optimum set of the dead-zone loss max(0, |F−y*|−τ)². So the framing I was leaning on collapses.

**What actually remains as differences** — differences to investigate, not priority claims:

| | Harmless Diversity | Component D |
|---|---|---|
| Setting | gradient descent on an objective | an edit **inside a frozen sampler** at inference |
| Constraint | analytic, on the main loss | a **per-anchor envelope** that adapts to the current error |
| Checking | analytic | a **real acceptance test** against the guide, with rollback |
| Domain | images, meshes, conformers | 3D property-targeted molecules; simplex sequences |

Plus the ρ analysis (§5) and some correctness results. One narrow gap survives a bounded search: nobody appears to plot diversity against property error for this task.

**I'd rather you judged this as thin-but-honest than let me oversell it.** I also removed an earlier line saying "only one of four contributions is at risk" — that wasn't supportable.

The brief allows combining and adapting prior work **as long as the paper explains what's new in our formulation**. I believe we clear that. I don't think we clear a workshop-novelty bar on the mechanism alone; a workshop paper would rest on the evidence.

**I'd like your read on whether you agree.** `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §5 has the full comparison.

### One thing about how we got here

**Two** reviews now. The first (15 Sep) found a correction term with the wrong sign that would have made the property error *worse*, and a "proof" that didn't hold in the space where the update is applied. The second (16 Sep) found twelve more problems, including the τ = 0 issue in §3, the ρ over-reading in §5, both withdrawn literature claims, and a pseudocode bug that would have skipped most sampler steps. All of it caught before any code was written.

The current documents contain only the corrected version. The errors, their counterexamples and the runnable checks are kept separately in `PROJECT_CLAIMS_VERIFICATION.md` and `audit/`. Worth a skim — partly so nobody re-implements something I got wrong, partly because that's the standard I'd like us to hold.

## 10. Who does what

Paired, not siloed. Everyone **presents something they didn't build**, because defence questions go to individuals.

| | Owns | Presents |
|---|---|---|
| **P1** | QM9 diffusion baseline; the two property predictors; quantum-chemistry verification | framing, data, flow-matching baseline |
| **P2** | QM9 flow-matching baseline; shared network; verifying every quoted number | diffusion baseline, metrics, model choice |
| **P3** | DNA modality end to end | guidance strategy, the innovation, ablations |
| **P4** | the guidance/edit code, evaluation harness, all figures | external comparison, transfer, limitations |

**P4 is on the critical path** — merged and tested by **Fri 19 Sep**.

## 11. Timeline

| Phase | Dates | Gate |
|---|---|---|
| **0** Assets + infra | 16–17 Sep | F0/F1 pass; metric code reproduces a published number |
| **1** Baselines + feasibility | 18–20 Sep | **F2/F3 decision**; guidance module merged Fri 19 |
| **2** The experiment | 21–23 Sep | **F4 result exists Tue 23** |
| **3** Transfer + freeze | 24–25 Sep | **RESULTS FREEZE Thu 25, end of day** |
| **4** Write | 26–28 Sep | full draft with every table on the 26th |
| **5** Submit + rehearse | 29–30 Sep | two timed run-throughs, no notes |

**Three hard rules.** No new experiments after 25 Sep — a missing ablation costs less than an unwritten paper. Every result writes to `results/raw/<experiment>/<config-hash>/` with its config and seed. No number is typed into a table by hand.

## 12. What I need from you this week

1. **Accept the repo invite:** https://github.com/Henry1145141919810/cis6270-project1-group2
2. **Join Slack:** [SLACK LINK]
3. **Tell me how many GPU-hours you can get** over the next two weeks.
4. **Pick a role** from §10 — first come, first served.
5. **Push back on §9.** Is the novelty position defensible? That's the question I'm least certain about and the one I most want a second opinion on.

## 13. Where to read more

| Document | What's in it |
|---|---|
| **`FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md`** | **start here** — the algorithm to implement, and the six gates |
| `COMPONENTS_ABCD_MATH_AND_PROOFS.md` | the technical reference, v5.0. §1 the idea · §2 components A–D · §3 the proofs (incl. §3.8, ρ) · §5 novelty · §6 open reading |
| `COMPONENTS_ABCD_V4_REVIEW.md` | the second review, with counterexamples in `audit/review_v4_checks.py`. Worth a skim for the standard |
| `PROJECT_GUIDE.md` | the long-form plan: modalities, evaluation, compute, slides |
| `PROJECT_CLAIMS_VERIFICATION.md` | the audit, with runnable checks in `audit/` |

If you read only one thing, read `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md`. If you read two, add §1 of `COMPONENTS_ABCD_MATH_AND_PROOFS.md`.

Call any time — easier than email for most of this.

**— Henry**
