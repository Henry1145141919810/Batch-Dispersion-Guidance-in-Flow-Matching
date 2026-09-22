# Project working guide — Group 2

**v0.8 · 16 Sep 2026** · Bobo Li · Henry Huang · Idea Idehpour · Haimo Fang

> **v0.8.** Part 2 states the current formulation; the mathematics is in `COMPONENTS_ABCD_MATH_AND_PROOFS.md` v6.0, and the error record for earlier formulations is in `PROJECT_CLAIMS_VERIFICATION.md` and `COMPONENTS_ABCD_V4_REVIEW.md`.

> **New here? Read `TEAM_BRIEF.md` first** (10 minutes, the whole project end to end), then `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md` for the algorithm we actually implement. This document is the long-form reference.

**How to read this.** Part 0 is background — skip it if you already know QM9 and equivariant networks. **Parts 2 and 3 are the heart of the project.** Part 2 states the method (components **A–D**) and the honest novelty position; all mathematics is in `COMPONENTS_ABCD_MATH_AND_PROOFS.md`, referenced by section. Parts 4–7 are the build plan. Anything marked **(verify)** came from memory or a search summary and must be checked against the source before it enters the paper; the running list is in §8.2.

**Locked decisions**

| | |
|---|---|
| Modality 1 | 3D molecular coordinates — **QM9**, de novo generation |
| Modality 2 | **Probability simplex** — DNA enhancer sequences, class-targeted generation |
| Base models (M1) | Continuous flow matching **and** Gaussian diffusion on one identical EGNN backbone |
| Guidance (M1) | Gradient guidance from a frozen property regressor |
| Innovation | **Components A–D; D is the contribution.** A property-preserving diversity edit with an application tolerance $\tau$ — $\tau=0$ recovers strict projection, $\tau\to\infty$ recovers unconstrained repulsion, both prior work. A is adopted from OSCAR; C is engineering. **Method: §2.3. Mathematics: `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §2–§3. Tests: `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md`** |
| Compute | ~1–2 A100-equivalents, shared, plus CPU for quantum chemistry |
| Due | Code + paper + slides **Tue 29 Sep, 8:30 am**; defense Wed 30 Sep |

---

## Contents

- [Part 0 — Background](#part-0--background)
- [Part 1 — What we are building](#part-1--what-we-are-building)
- [Part 2 — The innovation](#part-2--the-innovation)
- [Part 3 — Measuring diversity](#part-3--measuring-diversity)
- [Part 4 — Modality 1: QM9](#part-4--modality-1-qm9)
- [Part 5 — Modality 2: DNA on the simplex](#part-5--modality-2-dna-on-the-simplex)
- [Part 6 — Evaluation protocol](#part-6--evaluation-protocol)
- [Part 7 — Execution](#part-7--execution)
- [Part 8 — Risks and open questions](#part-8--risks-and-open-questions)
- [References](#references)

---

# Part 0 — Background

Skip if you already know this material.

## 0.1 Molecules and QM9

A small molecule is a set of atoms; each has an element and a position in 3D. We store coordinates ($N\times3$ numbers) and an element type per atom. **Bonds are not stored** — they are inferred afterwards from interatomic distances.

**QM9**: ~134k small organic molecules, each with a quantum-chemistry-computed 3D geometry and ~12 physical properties. At most 9 heavy (non-hydrogen) atoms, at most 29 atoms including hydrogens; elements are H, C, N, O, F. It is the standard benchmark for 3D molecule generation, so there are many published numbers to compare against.

**Valence** decides validity: each element forms a fixed number of bonds (H 1, C 4, N 3, O 2, F 1). *Atom stability* = the inferred bond count matches the valence. *Molecule stability* = every atom is stable. *Validity* = RDKit can parse the inferred bond graph into a legal molecule. These are rules of chemistry applied to whatever we generated — they need no reference molecule.

**Properties.** We target the **dipole moment $\mu$** (in Debye), which measures charge separation and depends on where the atoms are. Polarizability $\alpha$ grows mostly with molecule size, which makes it a poor target for us (see §4.5).

## 0.2 DNA enhancers and the probability simplex

DNA is a string over {A, C, G, T}. An **enhancer** is a ~500-letter stretch that switches a nearby gene on, but only in cell types whose transcription factors bind to short patterns inside it. Our dataset labels each enhancer with the cell type where it is active, determined by an accessibility experiment (ATAC-seq).

**Simplex representation.** A letter is discrete, but we represent each position as a probability vector over the four bases — four non-negative numbers summing to 1. That set is the simplex $\Delta^3$. A real sequence sits at a corner. Generation moves probability mass smoothly across the simplex; at the end we take the most probable letter at each position. The assignment states explicitly that this counts as a continuous modality.

## 0.3 EGNN — the network inside everything

**E(n)-Equivariant Graph Neural Network** [Satorras et al. 2021]. Input: $N$ atoms with positions and feature vectors. Each layer computes a message for every atom pair from their two feature vectors and their **distance**, updates each atom's features from its summed messages, and moves each atom's position along a learned weighted sum of the **direction vectors** to the other atoms.

Because messages use only distances and position updates only atom-to-atom directions, nothing references a fixed coordinate frame. Rotate the input and the output rotates identically (**equivariance**); a predictor that sums over atoms gives a number that does not change at all (**invariance**). Reordering atoms reorders the output, and any $N$ works.

Two roles here: the **backbone of both generative models** (identical network, only the training recipe differs), and the **property predictors** (same network with a sum over atoms at the end).

The alternative — forcing molecules into a canonical orientation and using a plain network — breaks down for near-symmetric molecules and wastes capacity learning a symmetry we can build in.

## 0.4 Naming trap: two different papers are called "EDM"

- **E(3) Equivariant Diffusion Model** [Hoogeboom et al., ICML 2022] — diffusion for 3D molecules on an EGNN backbone. **This is the EDM meant everywhere below**: its data split, metric code, released checkpoints and published numbers.
- **Elucidating the Design Space of Diffusion Models** [Karras et al., NeurIPS 2022] — an *image* diffusion paper, cited in the course template.

Write **"E(3)-EDM"** in the paper and slides.

## 0.5 How success is measured when there is no ground truth per sample

For a generated molecule, nobody has the "right answer." Three kinds of evidence:

1. **Rules.** Valence and RDKit sanitization are chemistry, not prediction. No learned model involved.
2. **Proxies.** A neural predictor estimates the property; a neural classifier estimates the cell type. These are what the whole field uses, and they are what our comparisons rest on.
3. **Physics.** Quantum chemistry computes the property directly (§4.7). Slow, so we run it on subsamples. This is real ground truth, and no baseline paper in our line reports it.

A proxy is trustworthy **for comparing two methods scored by the same proxy**, even if the proxy is biased — the bias shifts both. It is *not* trustworthy for an absolute claim like "this molecule really has dipole 2.5." The paper must say so.

## 0.6 Jargon

| Term | Meaning |
|---|---|
| Euler step | new $x$ = old $x$ + $\Delta t$ × velocity; the simplest ODE solver |
| $K$ / **NFE** | number of solver steps / network evaluations to produce one sample. A *sampling* cost; unrelated to training time |
| DDPM / DDIM | diffusion samplers: DDPM takes one stochastic step per training noise level (1000); DDIM is deterministic and can skip levels (10–100) |
| **EMA** | exponential moving average of weights: after each optimizer step, $\theta_{\text{EMA}} \leftarrow 0.999\,\theta_{\text{EMA}} + 0.001\,\theta$; sample from $\theta_{\text{EMA}}$. Averages ~the last 1000 steps → smoother model, better samples |
| training seed / sampling seed | randomness of init + data order / randomness of the starting noise. Training is expensive (1 seed), sampling is cheap (3 seeds for error bars) |
| pre-registered | decided and written down before results exist |
| **Pareto frontier** | the curve a method traces as guidance strength varies, in (property error, quality-or-diversity) space. One method **dominates** another when its curve is better everywhere |
| MAE | mean absolute error, average of \|prediction − target\| |
| FBD | Fréchet Biological Distance — the DNA analogue of FID, a distance between classifier embeddings of generated and real sequences |

---

# Part 1 — What we are building

## 1.1 The claim

> **Property guidance collapses diversity. Adding repulsion recovers diversity but gives back property control, because it scatters in every direction including the one that changes the property. Confining repulsion to directions that leave the property unchanged should recover diversity at a lower cost in control — moving the trade-off curve rather than sliding along it. We test this on two continuous state spaces with different geometry.**

The central question for the defence, one sentence: *does confining the repulsion to the reward's level set move the diversity-versus-control frontier, in both an SE(3)-Euclidean space and on the simplex?*

Note the shape of the claim. It is not "we are more diverse" — that is free, and meaningless (§3.2). It is that the **exchange rate** between diversity and control improves, measured against baselines that also have a diversity mechanism.

## 1.2 Why these two modalities

| | Modality 1 | Modality 2 |
|---|---|---|
| State space | $\mathbb{R}^{N\times3}$ + type channels; rotation/translation symmetry; variable $N$ | $(\Delta^3)^{500}$ — bounded, constrained, fixed length, no rotational symmetry |
| Dataset | QM9 | Melanoma enhancers, 89k × 500 bp **(verify)**; fly brain (104k) as alternate |
| Task | de novo generation; property-targeted generation | class-targeted generation |

The geometries are unambiguously different — unbounded Euclidean with symmetry versus a bounded manifold without one. That is the point: if the same mechanism helps in both, the idea is about guidance, not about molecules.

QM9 is Modality 1 because flow matching and diffusion can share an **identical EGNN backbone**, which makes the base-model comparison as fair as it can be, and because property-targeted generation has an established protocol with several recent methods to compare against.

## 1.3 The five experiments

1. **Flow matching vs. diffusion** on QM9, matched everything → decides which family we carry forward.
2. **Guidance works** — property error drops as strength rises, and we document what it costs in quality and diversity. That cost is the problem the innovation addresses.
3. **Ablations** — remove each component one at a time, plus matched controls, and compare full Pareto frontiers.
4. **Comparison with recent methods** on the same task and metric.
5. **Transfer** — the same mechanism on the simplex, against recent methods there.

## 1.4 Why the innovation lives at sampling time

A training-time change (new path, new coupling, new loss) needs a retrain for every ablation cell: roughly 12–24 training runs. A sampling-time change runs the **entire ablation grid from two trained checkpoints**. With ~1–2 GPUs and two weeks, that is the difference between a finished project and an unfinished one. Total generative-model training runs for the whole paper: **three**.

---

# Part 2 — The innovation

**The mathematics lives in `COMPONENTS_ABCD_MATH_AND_PROOFS.md`.** This Part states what we build, what we claim, and how it is ablated. Every derivation, proof and numerical check is referenced there rather than repeated.

## 2.1 The problem

Property-targeted generation should hand you a **portfolio**: many structurally different molecules that all hit dipole 2.5 D. Guided samplers hand you a handful of near-copies.

Why is subtler than it looks. The tilt *multiplier* depends only on $f(x)$, so it leaves every supported conditional $p(x\mid f(x)=c)$ **unchanged** at any strength (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.9.4) — guidance does not destroy *within-level* variety in the ideal limit.

But it **does** reweight the mixture over levels, and structural diversity can depend on the level. A two-line counterexample has exact sampling reduce structural variance from 1.01 to 0.11 as guidance strengthens (§1.1 of `COMPONENTS_ABCD_MATH_AND_PROOFS.md`). So the honest claim is narrow:

> **Approximate guidance may lose within-condition diversity beyond what conditioning implies. The experiment tests whether constrained diversification recovers useful coverage.**

Establish any excess loss empirically against a conditional reference. Repulsion is not automatically a repair, and can itself bias the distribution.

Two further failure modes of strong guidance shape the design: **off-manifold drift** (the predictor's gradient points partly away from the space of real molecules, and the error compounds) and **schedule mismatch** (early on the endpoint prediction is a blur, so guiding hard on it is guiding on a hallucination).

## 2.2 The baseline we modify

A DPS-style training-free guidance step: predict the endpoint, score it with a frozen property regressor, pull the gradient back to the state we can actually move.

```python
E = x + (1-t)*v_theta(x, t)               # endpoint proxy
F = f_A(E)
b = Q @ grad(F, x)                        # PROPERTY gradient, modality-projected
a = -(F - y_star)/s**2 * b                # reward direction, free from b
x = x + dt*(v_theta(x, t) + w_g*a)
```

> **Do not call this a TFG-Flow reproduction.** TFG-Flow includes discrete guidance and iterative continuous updates ([§3.3–3.4, Algorithm 1](https://arxiv.org/html/2501.14216v2)). Ours is a *related* additive baseline until those differences are reconciled — open item in §8.2.

## 2.3 The method — components A to D

| | Component | Role | Correct? | New? |
|---|---|---|---|---|
| **A** | velocity-relative trust region | keep the baseline update from overwhelming the model | it is a clip; nothing to prove | **No** — OSCAR publishes this form. Cite in Methods |
| **B** | level-set tangent projection | spread without changing the property | **yes**, proved: the exact nearest property-neutral direction | **No** — O-SVGD, Harmless Diversity, OSCAR, gradient surgery. Ours are two *correctness conditions* on using it, not the projection |
| **C** | confidence gate | damp guidance while the endpoint prediction is unstable | a heuristic; no theorem | **No** — time-varying guidance is crowded. Ablated, not claimed |
| **D** | **tolerance band** | spend the allowed error to buy diversity | **yes, locally** — feasibility, exact solve, local ascent, local dominance over equality | **An adaptation.** Inequality-constrained diverse particles exist; §2.7 states the delta |

> **So A and B are prior work, C is engineering, and only D carries a claim.** What is *not* proved for any of them: that they improve decoded diversity, accuracy, validity or yield. That is what F4 decides.

> ⚠ **D does *not* contain B as a sweep endpoint.** Since $A=\max(\tau,\lvert e\rvert)$, at $\tau=0$ with $e>0$ the constraint is $-2e\le b^\top d\le0$ — *no worsening*, inward movement allowed. Only at $e=0$ is it equality. Verified: $e=0.2$, $b=(1,0)$, $u=(-1,1)$, $R=0.1$ gives band edit $(-0.071,0.071)$ versus equality $(0,0.1)$.
>
> **Build three reference arms explicitly** — equality ($\ell=h=0$), the $\tau=0$ progress envelope, and unconstrained. The $\tau$ sweep reaches neither endpoint on its own.

### The idea in two observations

**The target is a set, not a point.** $\{x:f(x)=y^\ast\}$ is a surface, so move *across* level sets to reach the target and *along* the surface to diversify. The second is free at first order.

**You do not need the property exactly.** Nobody needs a dipole of 2.500; they need it within a tolerance the application sets. Strict preservation throws that away — at 5.00 with an accepted band of [4.9, 5.1], an equality-tangent edit blocks a move to 5.04 that might buy real structural variety.

### One primary axis, three reference arms

$\tau$ is the primary scientific axis — but **not** the only tunable parameter, and it does not interpolate between two prior-work endpoints.

| Arm | Constraint | What it is |
|---|---|---|
| **Equality** | $\ell=h=0$ at every anchor | the prior-work comparator |
| **$\tau=0$ band** | $-2\lvert e\rvert\le b^\top d\le0$ for $e>0$ | "no worsening of anchor error" |
| **Unconstrained** | guard only, no slab | plain repulsion; build explicitly |
| **$\tau>0$ band** | band active | the candidate |

> **A null sweep may mean nothing happened.** Whenever $\lvert e\rvert$ exceeds every $\tau$, $A=\lvert e\rvert$ and the constraint is **identical** for all of them. **Log the fraction of anchors with $\lvert e\rvert<\tau$** and the pairwise differences between proposed edits, or the sweep is uninterpretable.

### Component D, concretely

With $\tau\ge0$ the allowed guide error and $e$ the current signed error, at each anchor:

$$A=\max(\tau,\lvert e\rvert),\qquad \ell=-A-e,\qquad h=A-e \qquad\Longrightarrow\qquad \ell\le b^\top d\le h$$

Since $\ell\le0\le h$ **always**, the zero edit is always feasible — the problem can never become infeasible mid-sampling. Inside the band the edit stays inside; outside it, the edit may not *worsen* the anchor's error, and the baseline step keeps doing the job of reaching the target.

The local problem is then a nearest-point projection onto {ball ∩ slab ∩ subspace}, solved in **closed form** and proved optimal (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.6), followed by a **finite acceptance guard**: apply the edit, re-evaluate the real guide, and revert if the band is broken. Rejected edits reproduce the baseline anchor exactly.

### Three implementation rules that are easy to get wrong

| Rule | Why | Reference |
|---|---|---|
| Build the tangent space from $\nabla F$, never $\nabla r$ | $\nabla r$ vanishes at the target while the level-set normal does not, so a reward-gradient projector loses its constraint exactly at convergence | §3.9.1 |
| Project in **state space**, after the pullback | orthogonality is not preserved by $J^\top$: $\Delta r = a^\top JJ^\top u^\perp\neq0$ | §3.4 |
| Cap the **baseline only** | a shared cap on (reward + diversity) cuts the reward step when it binds — progress falls $1\to1/\sqrt2$ for orthogonal unit vectors | §3.9.5 |

## 2.4 The algorithm

```text
for each sampler step x at t -> z at t':

    # ---- baseline step, identical in every arm
    v = v_theta(x, t);  E = x + (1-t)*v;  F = f_A(E)
    b = Q @ grad(F, x)                    # PROPERTY gradient
    a = -(F - y_star)/s**2 * b
    g = w_g * c * a                       # c = gate                      (C)
    g = g * min(1, rho_A*|v| / |g|)       # cap the BASELINE only         (A)
    z = x + dt*(v + g)

    x = z                                 # <-- COMMIT, always
    if not a scheduled diversity step: continue

    # ---- diversity edit, anchored at z, fresh graph
    z = z.detach().requires_grad_(True)   # re-enable input grads
    E2 = z + (1-t2)*v_theta(z, t2);  F2 = f_A(E2);  e = F2 - y_star
    b2 = Q @ grad(F2, z)                  # VJP #1
    sigma = median_pairwise(h(E2)).detach()   # cache at the anchor
    u  = Q @ grad(D(h(E2); sigma), z)     # VJP #2, batch-coupled
    A_ = max(tau, abs(e))                 # tolerance band                (D)
    d  = project_ball_slab(eta*u, b2, -A_-e, A_-e, R)   # exact           (B,D)
    for alpha in (1, 1/2, 1/4):           # finite guard, sigma fixed    (D)
        if accept(z + alpha*d, sigma): x = z + alpha*d; break

decode terminal states; evaluate with independent f_B
```

**Schedule:** one edit every five steps over the final half of generation — a starting configuration, not an optimum. Untouched steps use exactly the baseline.

**Batches must share one $(y^\ast, N)$ condition.** The diversity claim only means anything within a condition; with per-sample targets a batch never collapses, it spreads across the target distribution (§3.9.3). This constrains the sampler interface — build it in from the start.

The commit precedes the scheduling branch, so unedited steps still advance the state.

**Cost:** roughly 1.2–1.4× forwards and 1.2× backwards at a 10% edit rate. Operation counts, not wall-clock — measure and report the real numbers.

## 2.5 Hyperparameters

| Symbol | Component | Meaning | Default | Sweep |
|---|---|---|---|---|
| $w_g$ | baseline | guidance strength | — | 0, 0.5, 1, 2, 4, 8 |
| **$\tau$** | **D** | **allowed property error — the knob** | from the application | **0, ¼δ, ½δ, δ** |
| $\lambda_D$ | D | diversity step scale | 1.0 | 0.5, 1, 2 |
| $\sigma$ | D | kernel bandwidth | **median pairwise distance, detached** | — |
| $\rho_A$ | A | max guidance-to-velocity ratio | 0.5 | 0.25, 0.5, 1 |
| $\tau_C$ | C | gate half-point | **median $\Delta$, unguided run** | 0.5×, 1×, 2× |

Two are **measured, not tuned** ($\sigma$, $\tau_C$) — the difference between "we added five knobs" and "we added one."

## 2.6 What predicts whether this works — the $\rho$ proposition

Let $\rho$ be the fraction of the diversity gradient's energy lying along the property direction:

$$\rho=\frac{(b^\top u)^2}{\lVert b\rVert^2\lVert u\rVert^2}\in[0,1]$$

At matched diversity, the projection avoids first-order property drift of

$$\mathcal{D}(\rho)=R\lVert b\rVert\sqrt{\rho(1-\rho)}$$

— zero at both extremes, maximal at $\rho=\tfrac12$ (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8; derived, and checked on 200,000 instances).

| Regime | What happens |
|---|---|
| $\rho\to0$ | nothing to remove — the projection is a no-op |
| $\rho\to1$ | the property direction carries all the diversity signal — projection removes everything |
| $\rho\approx\tfrac12$ | **the working regime** |

> ⚠ **Four things this identity does not say** — each with a counterexample in `audit/review_v4_checks.py`:
>
> 1. Avoided *change* is not avoided **harm**. At $e=0.2$ the unprojected edit *improves* the error $0.2\to0.15$ while the tangent edit holds it at $0.2$. **Measure the signed $e\,b^\top u$.**
> 2. There is no benefit-per-unit-diversity peak — $\mathcal D/G^\ast$ is **monotone** in $\rho$. The peak is in absolute avoided change at the equality arm's ceiling, which is a choice of comparison point.
> 3. $\rho$ cannot rank properties alone; $\mathcal D$ scales with $R\lVert b\rVert$ and units. Cross-property prediction is a **hypothesis**.
> 4. **$\rho\to1$ does not condemn component D** — it is exactly where available tolerance rescues a direction equality blocks. Never stop the band experiment on $\rho$ alone.
>
> Keep $\rho$ as a diagnostic for the **equality** arm. It is undefined at $b=0$ or $u=0$ — log that separately rather than as zero.

## 2.7 Is this new? — the honest assessment

**Read this and form your own view.** Three rounds of literature search, 13–16 Sep 2026. "Not found" means nothing surfaced, not that nothing exists; open reading is in §8.2.

### The four contributions

| # | Item | Status |
|---|---|---|
| **1** | **The $\rho$ identity** (§2.6) — magnitude of avoided first-order property change at the equality ceiling | proved as **algebra**; not a performance theorem |
| **2** | **Component D** — always-feasible per-anchor envelope, exact local solve, finite guard | local dominance over equality proved; **outcome is a hypothesis** |
| **3** | **Correctness results** for this class of method (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.9) | proved; external value depends on §8.2 reading |
| **4** | **The evaluation protocol** | good practice, **not novelty** |

> **No molecular improvement is established by any proof or numerical check.** The strongest valid theorem is local: the band solver attains a local objective at least as good as equality under the same budget, and accepted edits pass the stated checks at that anchor. Better decoded diversity, accuracy, validity and yield are all `[HYPOTHESIS]` for F4.

### What is taken

| Piece | Owner |
|---|---|
| Gradient through the clean prediction | DPS; TFG; TFG-Flow — **this is our baseline** |
| **Diversity inside a loss's optimum set**, "in a fashion that does not hurt the optimization of the main loss", **with molecular conformation experiments** | **[Harmless Diversity, ICML 2022](https://proceedings.mlr.press/v162/gong22b.html)** — the closest prior work |
| Normal/tangent decomposition for constrained particle sampling | [O-SVGD, NeurIPS 2022](https://arxiv.org/abs/2210.06447) |
| Repulsive joint-particle sampling; molecular conformers | [Particle Guidance, ICLR 2024](https://arxiv.org/abs/2310.13102) |
| Endpoint feature diversity, state-space pullback, **velocity-relative cap (component A)**, preservation proof | [OSCAR, ICML 2026](https://arxiv.org/abs/2510.09060) — against $v_\theta$, no reward model, images only |
| Reward-guided sampling **with** repulsive particle correction, on molecules | [Stein Diffusion Guidance, ICML 2026](https://arxiv.org/abs/2507.05482) |
| Diverse particles under inequality constraints | [Constrained Stein Variational Trajectory Optimization](https://arxiv.org/abs/2308.12110) |
| Diversity beside MAE on 3D QM9, **and DFT on 500 generated molecules at B3LYP/6-31G(2df,p)** | [MolGuidance](https://arxiv.org/html/2512.12198v1) — nearest neighbour on evaluation |

### What the delta is

**Two boundaries that do *not* separate us from prior work:**

1. **Physics validation is occupied.** MolGuidance reports DFT on 500 generated molecules at QM9's own level of theory.
2. **"Tolerance band" is not a distinction from "optimum set."** `[PROVED]` The band *is* the optimum set of the dead-zone loss $\max(0,\lvert F-y^\ast\rvert-\tau)^2$ when reachable. Nor is "through a network" a distinction from "on the state" — $F=f_A(E_t(x))$ is a function of the state.

**What remains as differences to investigate** — differences, not priority:

| | Harmless Diversity (2022) | Component D |
|---|---|---|
| Setting | population gradient descent on an objective | an edit **inside a frozen sampler** at inference |
| Constraint | analytically derived, on the main loss (eqs 4–6, Algs 1–2) | a **per-anchor progress envelope** $A=\max(\tau,\lvert e\rvert)$ |
| Verification | analytic | a **finite acceptance check** against the actual guide, with rollback |
| Domain | images, meshes, conformers, ensembles | 3D property-targeted de novo molecules; simplex sequences |

One narrow gap survives a **bounded** search: we found no diversity-versus-property-error frontier for property-targeted 3D molecular generation. MolGuidance reports both axes separately. This is what a bounded search found, not an absolute absence claim.

### Bottom line

The brief permits combining and adapting prior work **provided the paper explains what is new in our formulation**, excluding only hyperparameter tuning and unchanged reuse. This is neither, and §2.7 states the delta precisely. It does **not** clear a workshop-novelty bar on mechanism alone — a workshop paper would rest on the evidence.

**Attribute clearly and early** — Related Work *and* Methods. Being straightforward about what is borrowed reads as competence; being caught overclaiming does not.

## 2.8 Ablations and controls

Every row: same checkpoint, same conditions, same seeds, full sweep, 3 sampling seeds, **same-condition batches**.

The table asks one question: **does spending the tolerance beat both of its endpoints and the cheap alternatives?**

| Arm | What it isolates |
|---|---|
| Unguided | reference |
| Standard property guidance | the baseline |
| **Unconstrained** diversity, same guard (no slab, built explicitly) | does the constraint add anything beyond guard-and-backtrack |
| **Equality** arm, $\ell=h=0$ at every anchor | the prior-work comparator |
| **$\tau=0$ progress envelope** | separates "no worsening" from "no change" |
| **Component D, $\tau>0$ swept** | **the candidate** |
| **Dead-zone loss, no repulsion** | **does simple relaxation explain the whole gain** |
| **Generate-and-select at equal wall-clock** | **the practical baseline that must be beaten** |
| Full − A | does the method need the trust region |
| Full − C | does the method need the gate |

> **The last two arms in bold are the ones that can kill the idea, so run them from the start.** Beating plain guidance proves nothing.

**Matched controls:**

| Control | Replaces | Question |
|---|---|---|
| **C1** fixed-norm clipping at matched average magnitude | A | is it the *relative* bound that matters, or any clipping? |
| **C2** best fixed schedule at matched total guidance mass | C | does a measured gate beat a hand-tuned curve? |
| **C3** centroid repulsion at matched strength | D's energy | how much does $O(B^3)$ log-det buy over $O(B)$ centroid? |
| **C4** diversity in a *separately trained* embedding | D's embedding | is the guide's own embedding load-bearing? |
| **C5** project against $v_\theta$ instead of $\nabla F$ | B's target | are realism-preservation and property-preservation complementary? |
| **C6** band activity log | — | did $\tau$ ever engage? A null sweep with zero activity is not a result about the method |

C2 is the control we might lose, and that is fine — C is not claimed as a contribution. C5 turns OSCAR from a threat into a row.

---

# Part 3 — Measuring diversity

## 3.1 Why this part gets its own section

Diversity is the axis our innovation claims to improve. If the measurement is weak, the paper has nothing. And measuring it here is harder than usual for one specific reason: **our method contains a term whose explicit purpose is to push samples apart.** Any naive diversity metric is therefore suspect — we would be measuring the thing we optimised for.

Three failure modes to design against.

## 3.2 Failure mode 1 — circularity

> *"Your method has a term whose entire job is to push samples apart. Of course diversity went up. You optimised the metric."*

This is the objection that decides whether the paper is worth anything, so work through it carefully.

### What does not solve it

**Changing the measurement space does not solve it.** The spread term repels in $f_A$'s invariant embedding, so an obvious move is to score diversity somewhere else — Morgan fingerprints, scaffolds, coverage. It is worth doing, and we do it, but be clear about how little it buys: $f_A$ was trained on molecules and its embedding encodes molecular structure, so spreading out there is *correlated* with spreading out in any other sensible molecular representation. The spaces are not independent. Treat "we measured elsewhere" as hygiene, not as an answer.

### What does solve it: make the claim frontier-shaped

The objection is really about a *level* claim — "we are more diverse." A level claim is unfalsifiable and uninteresting, because diversity is trivially maximised by turning guidance off, or by emitting noise. Nobody needs a method for that.

**Our claim is about the trade-off, not the level.** Vanilla guidance already has a diversity knob: turn $w$ down and diversity rises while property error rises with it. So does ours. Both methods trace a *curve* in (property error, diversity) space. State the null hypothesis explicitly:

> **Null hypothesis: our curve is nothing but a reparametrisation of vanilla's.** We found another way to slide along the same trade-off, and at any matched property error we are no more diverse.

That is exactly what a skeptic means by "of course diversity went up," and it is a real, testable hypothesis. Adding *any* diversity term slides you along the existing curve — more diversity, more error. Only a mechanism that buys diversity **without paying error** moves the curve. So the finding we are after is not "diversity went up" but "the curve moved," and nothing in our method optimises the curve's position. We optimise one axis and *pay* on the other; the exchange rate is what we measure.

### The decisive experiment: match the diversity, compare the cost

Run the frontier in **both** directions.

| Direction | Setup | Interpretation |
|---|---|---|
| Match error, compare diversity | pick $w$ for each method so property error is equal | the usual reading |
| **Match diversity, compare error** | tune $\gamma$ on the baseline until its diversity equals ours, then compare property error | **the direct answer to circularity** |

The second one is the important one, because the baseline in that comparison is **also a repulsion method**. Specifically:

- **vanilla + unprojected repulsion** (our "+B, projection OFF" row) — identical repulsion, identical strength, only the projection removed
- **vanilla + pairwise kernel repulsion** (control C3, Particle-Guidance style)
- **repulsion in a separately trained embedding** (control C4) — the one that tests whether `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8's alignment argument is actually what drives the effect

If both baselines have a diversity term and we still reach the same diversity at lower property error, "you added a diversity term" explains nothing — *they added one too.* What is left to explain the gap is the projection, which is precisely the thing we claim.

This is why the projection-OFF row is described in §2.8 as the key ablation. It is not one row among several; it is the row that answers this objection.

### Three supporting safeguards

1. **The cost axis is verifiable by physics.** Even if the diversity axis is somewhat self-serving, the axis we claim *not* to sacrifice is checked by quantum chemistry (§4.7), which no method can fool. "Equal DFT-verified property error, more distinct scaffolds" is a much harder claim to wave away than one resting entirely on learned metrics.
2. **Scaffold count is a discrete chemical fact.** A Bemis–Murcko scaffold comes from a deterministic graph algorithm; two molecules either share a core or they do not. Harder to slide than any continuous distance, though not immune.
3. **Pre-register the metrics** (they are fixed in §3.5, before any results exist) and **report the gameable one too**. If pairwise Tanimoto, scaffold count and coverage all move together, that agreement is itself evidence; if only the gameable one moves, we have learned something and must say so.

### Still hygiene, still required

> **Circularity rule: never report diversity measured in the space the repulsion acts in.** Embedding-space spread is never evidence — it is the objective.

Same discipline as the guide/oracle split for the property metric (§4.6): what you optimise is never what you score.

## 3.3 Failure mode 2 — diversity by garbage

Repulsion pushes toward *different*, and broken molecules are different. A method could raise pairwise distance simply by producing junk.

Defences: compute ratio metrics over **valid molecules only**; use metrics that are undefined for invalid molecules (scaffolds); use metrics referenced to **real** data (coverage — garbage is far from everything, so it *lowers* coverage); and cross-check with quantum chemistry (§4.7), which catches junk the neural oracle scores well.

## 3.4 Failure mode 3 — selection bias from the validity filter

This one is subtle and easy to get wrong.

If vanilla at $w=8$ yields 40% validity and ours yields 80%, then "diversity among valid molecules" compares **4,000 survivors against 8,000 survivors**. Different sample sizes, and the survivors are a *selected* subpopulation. The bias can run either way: vanilla's survivors might be the simple, similar molecules (flattering us), or the ones that escaped the collapse (hurting us). Either way it is not a clean comparison.

**The fix: normalise by samples *attempted*, not samples *surviving*.**

> distinct scaffolds **per 1,000 generated** · distinct valid molecules **per 1,000 generated**

A fixed denominator removes the selection effect completely. It also measures what a practitioner actually wants — how many distinct usable candidates per 1,000 attempts — and it cannot be gamed in either direction: garbage produces no scaffolds, and collapse produces few distinct ones.

## 3.5 The metric design

| Tier | Metric | What it is | Immune to |
|---|---|---|---|
| **Primary** | **Conditional coverage / density** | For target $y^\ast$, take the real test molecules whose property is within ±0.1 D of $y^\ast$ — ground truth for "molecules with this property." Coverage = fraction of those real molecules that have a generated neighbour within radius $r$; density = concentration of generated samples near real ones. Computed on Morgan fingerprints [Naeem et al. 2020] | circularity (different space), garbage (garbage is near nothing real), selection bias (fewer valid ⇒ less coverage) |
| **Secondary** | **Distinct scaffolds per 1,000 generated** | count of distinct Bemis–Murcko scaffolds ÷ attempts. Highly legible: *"vanilla at $w{=}8$: 12 distinct cores per 1,000 attempts; ours: 47"* | garbage, selection bias |
| **Secondary** | **Distinct valid molecules per 1,000 generated** | unique canonical SMILES ÷ attempts | garbage, selection bias |
| **Secondary** | **Vendi Score** | the effective number of distinct molecules: $\exp$ of the Shannon entropy of the eigenvalues of the normalised Tanimoto similarity matrix. Reports "these 1,000 samples are worth ~47 distinct molecules" — a single interpretable number that degrades gracefully as samples become near-duplicates, unlike a hard uniqueness count | collapse-by-near-duplicate (which uniqueness misses entirely) |
| **Tertiary** | Mean pairwise Tanimoto distance | Morgan fingerprints, radius 2, 2048 bits, over valid molecules, subsampled to matched counts | nothing — report it, flag it as the gameable one |

**Why Vendi and not just uniqueness.** Uniqueness is a hard count: 1,000 samples that are all *slightly* different score 100%, even under total collapse. Vendi asks how many *effectively distinct* things are present, so near-duplicates are penalised continuously. It is the standard diversity measure in recent generative-chemistry work and reviewers will expect it. Compute on the same Morgan fingerprints, so it costs nothing beyond the similarity matrix we already build.

Coverage is primary precisely because it is referenced to real data. Pairwise spread rewards *scattering*; coverage rewards *reaching real molecules that have the target property*. Reporting both makes the story unfakeable in either direction: a method that scatters into junk shows high pairwise distance and low coverage, and we would catch it.

## 3.6 The matched-control principle

**No diversity number is ever reported as a standalone comparison.** Unguided generation is trivially the most diverse and the least controlled, so "more diverse" means nothing on its own (§3.2).

Every diversity number is read off a Pareto frontier, and the frontier is read in **both** directions:

| Direction | How | What it answers |
|---|---|---|
| Match property error → compare diversity | pick $w$ per method so MAE is equal | "at equal control, how much variety do we get?" |
| **Match diversity → compare property error** | tune the baseline's $\gamma$ until its diversity equals ours | **"is the gain explained by 'you added repulsion'?"** — no, if the baseline also has repulsion |

Both directions come from the same sweep; they are two readings of one plot. The second is the one that answers the circularity objection, and it is only meaningful when the comparison baselines **also have a diversity mechanism** — the projection-OFF row and control C3. Report it against those, not only against plain vanilla guidance.

The frontier plot — not a table row — is the paper's headline result. State the null hypothesis beside it: *our curve is a reparametrisation of the baseline's.* The figure either rejects that or it does not.

### 3.6b Summarising a frontier with one number, defensibly

"The curve moved" has to become a number a reader can check, or the claim rests on eyeballing a plot. Use the three standard multi-objective indicators, all computed in (property MAE ↓, diversity ↑) space with a **pre-registered reference point** fixed from the unguided run:

| Indicator | What it says | Why include it |
|---|---|---|
| **Hypervolume (HV)** | area dominated by the frontier relative to the reference point | the headline single number; a strictly better curve has strictly larger HV |
| **Distance to ideal point (DIP)** | how close the best achievable compromise gets to the (0 error, max diversity) corner | robust to one method having a longer tail |
| **R2 indicator** | average best achievable value over a spread of scalarisation weights | insensitive to the reference-point choice, so it guards against HV being an artefact of where we put the reference |

> **Report bootstrap confidence intervals on every indicator.** Resample the generated molecules within each cell (1,000 resamples), recompute the frontier, recompute HV/DIP/R2. A frontier drawn from 3 sampling seeds looks decisive and often is not; an HV gap whose CI straddles zero is **not a result**, and we would rather find that ourselves than be told. Pre-register the reference point before looking at any frontier.

**The mechanism plot to put beside the frontier.** Sweep $\gamma$ at fixed $w_g$ and plot property MAE. `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8 predicts the unprojected arm rises and diverges while ours stays flat. That figure shows *why* the frontier moved, and it is stronger evidence than the frontier itself because it tests the derivation rather than the outcome.

## 3.7 Practical notes

- **Uniqueness depends on sample count** (more samples, more chance of a duplicate), so it is comparable only across cells with identical counts. One more reason counts are fixed.
- **Pairwise Tanimoto on 10k samples is 50M comparisons.** Subsample 2,000 (2M pairs, seconds) using RDKit's `BulkTanimotoSimilarity`.
- **Morgan fingerprints see the molecular graph, not 3D geometry**, so two conformers of one molecule score as identical. Acceptable — chemical variety is what we claim — but state it as a limitation.
- **Report mean ± std over 3 sampling seeds** for every diversity number.

## 3.8 Modality 2 equivalents

Same logic: repulsion acts in the DNA classifier's embedding, so diversity is measured elsewhere.

- normalised **pairwise Hamming distance** — compare against the *real-data* value, not against 1.0; random 500-mers already sit near 0.75
- **4-mer frequency JS divergence** to real target-class sequences
- **Vendi Score** over a 4-mer or 6-mer similarity kernel — the effective number of distinct sequences
- **unique fraction** and **distinct sequences per 1,000 generated**
- **coverage/density** against real test sequences of the target class

Same frontier discipline as M1: HV / DIP / R2 with bootstrap CIs (§3.6b), read in both directions, against a baseline that also has a spread term.

---

# Part 4 — Modality 1: QM9

## 4.1 Data and the four-way split

Standard E(3)-EDM / Cormorant split: **100,000 train / 17,748 val / 13,083 test (verify)** — use their split files. Coordinates centred at zero centre of mass; atom types as one-hot channels treated as continuous; hydrogens explicit; $N$ sampled from the training histogram before generation.

| Partition | Size | Used by | Never used by |
|---|---|---|---|
| Train half 1 | 50k | FM base, diffusion base, **guide $f_A$** | $f_B$ |
| Train half 2 | 50k | **oracle $f_B$** only | generative models, $f_A$ |
| Validation | 17,748 | checkpoint selection; defaults for $\rho,\gamma,\tau$; $w$ calibration | — |
| Test | 13,083 | the $(N, y^\ast)$ evaluation targets; novelty and coverage reference | any training |

**Why half 2 is not wasted on the oracle.** Its accuracy is the floor on every MAE we can measure. And if the generative model trained on all 100k, it would have seen the oracle's training molecules — generated samples would resemble them and the oracle would score them optimistically. That is a leak. The cost, to disclose: our base models see 50k where E(3)-EDM sees 100k. TFG-Flow and PropMolFlow use the same 50/50, so our numbers stay comparable to theirs.

## 4.2 Shared backbone

EGNN in E(3)-EDM's QM9 configuration: 9 equivariant layers, 256 hidden **(verify)**. Both families use this backbone with identical hyperparameters. The **only** differences are the interpolant/forward process, the regression target, and the sampler. Report the exact parameter count from the code.

## 4.3 Flow-matching baseline

Source $x_0 \sim \mathcal{N}(0,I)$ in the zero-CoM subspace.

$$x_t = (1-t)x_0 + t\,x_1, \qquad u_t = x_1 - x_0, \qquad \mathcal{L} = \mathbb{E}\big[\|v_\theta(x_t,t) - (x_1-x_0)\|^2\big]$$

*Draw a straight line from noise to data; the network learns to output that constant direction from any point on the line.*

Clean prediction: $\hat{x}_1 = x_t + (1-t)v_\theta$ — where you land if you keep going at the predicted velocity.

**Sampling:** Euler, $K \in \{10,20,50,100\}$, NFE $=K$; carry $K=100$ into all guidance work. Heun (2nd order, NFE $=2K$) optional. Decode types by argmax at $t=1$.

## 4.4 Diffusion baseline

$$x_t = \alpha_t x_1 + \sigma_t\epsilon,\quad \alpha_t^2+\sigma_t^2=1,\qquad \mathcal{L} = \mathbb{E}\big[\|\epsilon - \epsilon_\phi(x_t,t)\|^2\big]$$

E(3)-EDM's polynomial noise schedule, $T=1000$ **(verify)**. Clean prediction $\hat{x}_1 = (x_t - \sigma_t\epsilon_\phi)/\alpha_t$. Samplers: ancestral DDPM at 1000 steps (its native setting, and how published numbers were produced) plus DDIM at $K\in\{10,20,50,100\}$ for an NFE-matched comparison with FM.

## 4.5 Training settings, model selection, and the family decision

Identical for both families: Adam, lr $10^{-4}$, batch 128, EMA 0.999, no weight decay, **fixed budget of 16 GPU-hours per model** (record the resulting epochs). Validate every ~5 epochs on validation loss plus atom stability over 1k samples at NFE 100.

**Pre-registered checkpoint rule:** highest validation atom stability at NFE 100, ties broken by validation loss.

**Pre-registered family rule:** carry forward whichever has higher molecule stability at NFE 100; if within one standard deviation, carry forward the one that reaches 95% of its own best atom stability at the smaller NFE. Our method is family-agnostic, so this choice does not threaten it.

One training seed per family (disclosed); three sampling seeds everywhere.

## 4.6 Guidance setup

**Two predictors, trained on disjoint halves.**

- **$f_A$ — the guide.** Its gradient steers sampling. Frozen. Exposes both the reward and its invariant embedding $h$.
- **$f_B$ — the oracle.** Scores results. Never touches generation.

The separation exists because guidance is literally an optimiser aimed at $f_A$, and given enough strength it will find inputs that fool $f_A$ rather than molecules that have the property (*reward hacking*). $f_B$ catches that. Disjoint halves make their blind spots less correlated. Report the **$f_A - f_B$ gap** as a hacking diagnostic; it should grow with $w$.

**Reward:** $r(x) = -\dfrac{(f_A(x)-y^\ast)^2}{2s^2}$, with $s$ the training standard deviation of the property, so $w$ is unitless and comparable across properties.

**Where it enters:** $g = \nabla_{x_t}\,r(\hat{x}_1(x_t,t))$, added to the velocity. Project the coordinate part into the zero-CoM subspace so guidance never introduces a net translation. $g$ is automatically equivariant because $f_A$ is invariant.

### Three properties, at three depths (pre-registered)

The generative model is **unconditional and property-agnostic** — it never sees a property value. Only the predictors are property-specific. So adding a property costs two small regressors and some sampling; it does **not** cost another base model. That makes breadth cheap, and the baselines we compare against report six properties, so reporting one would look thin.

| Property | Depth | Why |
|---|---|---|
| **$\mu$** dipole moment | **Primary — full development.** Ablation grid, all matched controls, both frontier directions, quantum-chemistry validation | Depends on geometry rather than atom count, so guidance can actually move it. The property everything else is calibrated on. |
| **gap** HOMO–LUMO | **Secondary — headline comparison only**: vanilla / projection-OFF / full, across the $w$ sweep | Tests whether the mechanism generalises across properties or was lucky on one. A different physical quantity with different level-set geometry. |
| **$\alpha$** polarisability | **Stress test — headline comparison only** | Expected to *fail*. $\alpha$ scales mostly with molecule size, and $N$ is fixed before sampling starts, so guidance can barely move it. Running it turns an asserted limitation into a **measured** one (§7.7). |

**What this buys.** Generality evidence rather than a single data point; a table shaped like the baselines'; a failure mode we demonstrate instead of assert; and protection against the effect being an artefact of one property.

**What it costs.** Six small regressors instead of two (~1 h each), and the secondary/stress sweeps — three arms × six $w$ values × three seeds ≈ 54 cells each. The full nine-row ablation grid runs for $\mu$ **only**; gap and $\alpha$ get the three-arm comparison. Roughly +14 GPU-h total.

**Report per property, never pooled.** Each property gets its own frontier and its own row. If the projection helps $\mu$ and gap but not $\alpha$, that is the result — say so, and explain the mechanism.

**Also report the alignment diagnostic per property** (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8): the *signed* $\langle \tilde u, \tilde a
angle$, its normalised $\cos$, and the **fraction of samples carrying the predicted sign**. We expect the alignment to differ between properties, and if it predicts where the projection helps, that is the mechanism evidence the Discussion needs. Also report the $\gamma$-sweep of property MAE at fixed $w_g$ — flat for us, rising for the unprojected arm, is the fixed-point prediction of `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8 made visible.

**Protocol:** for each of 10k test molecules take its $(N, y^\ast)$, generate with that $N$ guided toward $y^\ast$, and report MAE under $f_B$. Sweep $w \in \{0,0.5,1,2,4,8\}$, reporting quality and diversity at every value.

**Note for the paper:** our vanilla row *is* TFG-Flow's published rule, so it is a reproduction of a recent baseline, not a straw man.

## 4.7 Quality metrics and ground-truth validation

| Metric | What it is |
|---|---|
| Atom / molecule stability | inferred bond counts match valence, per atom / for all atoms — E(3)-EDM's `bond_analyze` tables |
| Validity | RDKit sanitizes the inferred bond graph |
| MAE under $f_B$ | the target metric |
| $f_A - f_B$ gap | reward-hacking diagnostic |
| NFE, backward passes, wall-clock | efficiency; report our extra passes honestly |
| Diversity | Part 3 |

**How validity is actually computed.** The model emits a cloud of atoms; nothing guarantees it is a molecule. (1) argmax the type channels; (2) for each atom pair, compare the distance against reference bond lengths (C–C single ≈ 1.54 Å, double ≈ 1.34, triple ≈ 1.20, with tolerances) and assign the highest matching order; (3) sum bond orders per atom against the allowed valence; (4) build an RDKit molecule and sanitize.

Two things to settle before running anything:

- **Disconnected fragments.** RDKit parses them as one molecule with a `.` in the SMILES, so junk fragments can register as "unique" and inflate diversity. **Decide the rule now and apply it everywhere: count multi-fragment outputs as invalid** (recommended), or take the largest fragment.
- **The ceiling is not 100%.** Distance-based bond inference is a heuristic and does not score *real* QM9 molecules perfectly — around 97–99% atom stability **(verify by running it)**. Run the pipeline on real test molecules first; that number is the ceiling and every generated result is read against it.

### Ground-truth validation with quantum chemistry

The guidance methods we compare against — TFG-Flow, OC-Flow, EEGSDE, PropMolFlow — score properties with a *neural* oracle. We add a physics check, because guidance is an optimiser aimed at a neural network and the only way to show it did not merely fool one is to ask physics. **CPU only**, so it never competes with GPU training.

> **Not a novelty claim.** [MolGuidance](https://arxiv.org/html/2512.12198v1) reports DFT on 500 generated molecules at B3LYP/6-31G(2df,p) — QM9's own level of theory (verified 16 Sep). Physics validation is good evaluation practice and it is already occupied. Do it; claim nothing for it.

| Tier | Method | Cost | Coverage |
|---|---|---|---|
| 1 | **GFN2-xTB** (semi-empirical) | ~1–5 s/molecule | 2,000 per condition |
| 2 | **DFT at B3LYP/6-31G(2df,p)** — QM9's own level of theory, single-point | ~1–5 min/molecule **(measure day 1)** | 400 per condition, stratified |

Tier 2 is the real answer: the same functional and basis that produced QM9's labels, so the number is directly comparable to the target. Five conditions × 400 × ~3 min ≈ 100 core-hours ≈ 6 h on a 16-core machine. Software: PySCF or Psi4 for DFT, `xtb` for tier 1.

Rules that keep it honest:

1. **Single-point on the geometry as generated.** Do not optimise first — that measures a different, relaxed molecule. A second column after relaxation is optional and informative (how far did it move?).
2. **Never drop failures.** Non-converged molecules are usually the broken ones. Report the failure rate per condition beside the MAE; never a bare MAE over survivors.
3. **Calibrate first** on 500 real QM9 molecules against their stored labels. DFT should be ≈ 0; xTB will show a real bias. Publishing that separates tool error from method error.
4. Assume neutral closed-shell singlet, as QM9 is.
5. Stratify by target value so conditions are compared on matched targets.

If our DFT-MAE tracks our oracle-MAE while vanilla's diverges at high $w$, that is direct evidence of reward hacking and a result worth reporting on its own.

## 4.8 External comparisons

**All quoted numbers are labelled "reported by original paper" and must be re-checked (§8.2).** Predictors and protocols differ between papers, so cross-paper numbers are indicative; the fair comparison is the rows we run ourselves.

| Method | Year | Mechanism | Training-free | MAE $\mu$ (D) | Diversity reported? |
|---|---|---|---|---|---|
| Conditional E(3)-EDM | 2022 | conditionally trained diffusion | no | 1.111 | uniqueness only |
| EEGSDE | 2023 | energy-guided SDE | no (trains an energy model) | 0.777 | uniqueness only |
| **TFG-Flow** | 2025 | gradient through $\hat{x}_1$, no scheduling | **yes** | 0.817 | **no** |
| **OC-Flow** | 2024/25 | optimal-control trajectory guidance, iterative | yes, expensive | 0.314 | uniqueness only |
| JODO / PropMolFlow | 2023/25 | conditionally trained | no | 0.620 / 0.620 | — |
| Ours: vanilla (reproduced) | — | TFG-Flow rule | yes | | **yes** |
| Ours: full method | — | + reward-orthogonal repulsion (+ A, C) | yes | | **yes** |

Reference points (PropMolFlow Table 1): QM9 lower bound 0.043, random 1.616; "#atoms" ≈ 1.05 **(verify)**. Guidance "works" when MAE sits well below #atoms.

**Why these baselines.** TFG-Flow and OC-Flow solve the identical problem — training-free property guidance of a flow model on QM9 — and our vanilla row is TFG-Flow's rule. EEGSDE is the diffusion-side gradient-guidance reference. Conditionally trained methods are a different regime, reported for context.

**The gap this table exposes:** none of them report a diversity–control trade-off. Our diversity columns are where the contribution shows.

---

# Part 5 — Modality 2: DNA on the simplex

## 5.1 Data and split

Melanoma enhancer set from the Dirichlet FM benchmark: 89k sequences, length 500, cell-type labels **(verify class count — 47)**. Fly brain (104k) is the alternate. Use their split files.

Same four-way structure as M1: train half 1 → FM base (class-conditional, 15% label dropout) + guide classifier $p_A$; train half 2 → oracle classifier $p_B$; validation → selection; test → target classes and the real-sequence reference for FBD.

Three cautions that do not arise with molecules:

- **Homology leakage.** Genomic sequences can be near-duplicates across a split (paralogs, repeats, overlapping loci). A random split leaks; the standard fix is splitting by chromosome. **Verify what Dirichlet FM does** and state it — this is the one place in the project where leakage is a real risk.
- **FBD comparability.** FBD lives in a classifier's embedding, so it is only comparable across papers with an identical embedding model. Use their released classifier if it exists; otherwise every FBD number is internal-only and labelled as such. The reference sequences come from **test**.
- **Class imbalance.** Fix the evaluated classes in advance — the 10 most frequent plus 3 rare ones reported separately — and use the same list for every method.

## 5.2 What transfers unchanged

The whole mechanism, line for line: the property gradient through $\hat{x}_1$, **components A–D**, the finite acceptance guard, the ablation rows, the controls, the frontier protocol. The code path is literally shared — the guidance module takes a model, a guide network, and a modality projection, and **only line 12 of the algorithm (§2.6) differs between modalities.**

The embedding argument of `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8 transfers too, and this is what makes M2 a real test rather than a demo: the DNA guide classifier also ends in a linear head over a pooled invariant embedding, so $\langle w_c, h_i - \bar h\rangle$ is again "this sequence's class logit minus the batch's." **Run T0's alignment diagnostic on M2 as well, before the sweep.** If the alignment is there on molecules and absent on DNA, that is a finding about when the mechanism applies, and it is more interesting than a second win.

## 5.3 What must change

| | Modality 1 | Modality 2 |
|---|---|---|
| Base model | EGNN FM on $\mathbb{R}^{N\times3}$ | 1D dilated-CNN (or small transformer) FM on $(\Delta^3)^{500}$, class-conditional |
| Source | zero-CoM Gaussian | uniform on the simplex, Dirichlet(1,1,1,1) per position |
| Path | linear | linear — a convex combination stays on the simplex, so the unguided flow needs no projection |
| Parameterisation | velocity | predict clean-sequence probabilities (softmax per position), then $v = (\hat{x}_1 - x_t)/(1-t)$ — the standard simplex-FM form, which gives $\hat{x}_1$ for free |
| Reward | $-(f_A - y^\ast)^2/2s^2$ | $\log p_A(y^\ast \mid \hat{x}_1)$ |
| Guidance projection | zero centre of mass | **tangent space** — subtract the per-position mean so each 4-vector sums to zero |
| State projection | none | clip $\ge 0$, renormalise each position |
| Quality | stability, validity | FBD |
| Target | MAE under $f_B$ | target-class accuracy under $p_B$ |

Training: Adam, lr $3\times10^{-4}$, batch 256, EMA 0.999, ~4–6 GPU-hours, select on validation loss with an FBD sanity check. NFE sweep {20, 50, 100}; carry 100.

**Base-model decision (by 16 Sep):** if Dirichlet FM's official code trains on this dataset within a day and its sampler is easy to hook, use it as the base and cite the code. Otherwise implement linear simplex FM ourselves — it is their own "linear FM" baseline and simpler.

## 5.4 Two things M2 can do that M1 cannot

**Raw-space vs. embedding-space repulsion.** On M1 the repulsion *must* live in the guide's embedding — molecules have different atom counts and orientations, so raw-space repulsion is undefined. On M2 every sequence is the same shape, so both are possible. Run both. If embedding-space wins here too, the design choice is validated rather than merely excused. One extra sampling config.

**Applying the method to classifier-free guidance.** Train the base with 15% label dropout (free), then apply the same update to the CFG vector $g = v_\theta(x,t,c) - v_\theta(x,t,\varnothing)$ instead of a classifier gradient. Two payoffs: a **directly comparable row** against Dirichlet FM's published CFG numbers, and evidence the mechanism generalises across *guidance types*, not only state spaces. One extra sampling config on a model we train anyway.

## 5.5 External comparisons

| Method | Year | Family |
|---|---|---|
| DDSM | 2023 | Dirichlet diffusion on the simplex |
| **Dirichlet FM** | 2024 | FM with a Dirichlet path; closest comparison; reports class-conditional CFG with FBD and accuracy |
| Fisher Flow | 2024 | FM on the sphere via the Fisher–Rao map |
| Gumbel-Softmax FM | 2025 | FM on a Gumbel-softmax simplex path |

Plus our base unguided, base + vanilla classifier guidance, and base + full method. Their guidance is CFG and ours is classifier-gradient — disclose it, and use the CFG variant above for a direct row.

**Honest risk:** whether diversity collapse is as severe on DNA is unknown. It could be milder (many valid sequences per class) or worse (CFG on DNA is known to converge on a few motifs). Either outcome is reportable, and the assignment says a weak transfer result is still useful if we explain what it reveals.

---

# Part 6 — Evaluation protocol

Written down before results exist.

| Item | Choice |
|---|---|
| Sample count | M1 10k per cell (5k for the ablation grid only if sampling is slower than estimated, disclosed); M2 5k |
| Seeds | 3 sampling seeds for every number; 1 training seed per base model, disclosed as a limitation |
| Uncertainty | mean ± std over sampling seeds in every table; error bars on every frontier |
| Matching | same checkpoint, $N$ list, target list, NFE and seeds across all guidance rows; same split and metric code across the base comparison |
| Quoted vs. reproduced | every external number carries a label; never mixed within a row |
| Compute reporting | parameters, training hours and epochs, NFE, backward passes per step, seconds per 1k samples, GPU model, seeds |
| Metric validation | before any model trains, run E(3)-EDM's released checkpoint through **our** metric code and reproduce its published numbers. If we cannot, our metrics are wrong |
| Pre-registered rules | family selection, checkpoint selection, property choice, table operating point — all fixed in Part 4 |

---

# Part 7 — Execution

## 7.1 Timeline

| Phase | Dates | Work | Gate |
|---|---|---|---|
| **0 — Lock + infra** | Fri 12 – Sun 14 Sep | **request Betty access on day 1** (§7.4 — access requests take time); repo, data loaders with official split files, backbones, **all metric code**, E(3)-EDM checkpoint through our metrics, sampling-speed measurement | metrics reproduce published numbers; 5-minute smoke run of train → sample → eval on both modalities |
| **1 — Baselines + feasibility test** | Mon 15 – Thu 18 Sep | train QM9 FM + diffusion in parallel; train $f_A$, $f_B$; train DNA base + classifiers; **run `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md`** | **feasibility decision Tue 16 Sep**; base-model comparison table Thu 18; guidance module merged Fri 19 |
| **2 — Innovation** | Fri 19 – Mon 22 Sep | vanilla $w$ sweep; ablation grid + controls; Pareto frontiers; decide the framing from the results | **frontier figure exists Mon 22 Sep** |
| **3 — Transfer + freeze** | Tue 23 – Thu 25 Sep | M2 sweep and comparison; quantum-chemistry runs; verify all quoted numbers; overview figure; failure-mode samples | **RESULTS FREEZE Thu 25 Sep, end of day** |
| **4 — Write** | Fri 26 – Sun 28 Sep | 26th full draft with every table; 27th rotate-review; 27–28th slides; 28th README and code cleanup | draft compiles to ≤ 5 pages on the 26th |
| **5 — Submit + rehearse** | Mon 29 – Tue 30 Sep | submit before 8:30 am; **two full timed run-throughs, no notes**; mock Q&A where each person answers on parts they did not build | — |

**Three hard rules.** No new experiments after 25 Sep — a missing ablation costs less than an unwritten paper. Every result writes to `results/raw/<experiment>/<config-hash>/` with its config and seed. No number is typed into a table by hand; tables are generated from `results/`.

## 7.2 Roles

Paired, not siloed. Everyone **presents something they did not build**, because defence questions go to individuals and cannot be redirected.

| | Owns | Backs up | Writes | Presents |
|---|---|---|---|---|
| **P1** | QM9 diffusion base; $f_A$/$f_B$; **quantum-chemistry validation** | M1 sampling grid | diffusion methods, base comparison, compute table | study framing, data, FM baseline |
| **P2** | QM9 FM base; shared EGNN; verified external quotes | guidance code review | FM methods, external comparison, related work (M1) | diffusion baseline, metrics, overview figure, base-model choice |
| **P3** | M2 end to end — data, simplex FM, classifiers, FBD, transfer results | overview figure | modalities/data, transfer methods, transfer results, related work (M2) | guidance strategy, the innovation, ablations |
| **P4** | **guidance module + eval harness + all figures**; paper editor | M2 sampling | innovation, guidance results, ablations, cross-modality, abstract, intro, discussion | external comparison, transfer, limitations, conclusions |

Assign names by 13 Sep. **P4's module is on the critical path — merged and unit-tested by Fri 19 Sep.**

## 7.3 Repository layout

```
proj1/
├── README.md                     # structure, setup, data prep, how to run everything,
│                                 # and a table mapping scripts/configs → paper tables
├── environment.yml
├── configs/
│   ├── qm9/       fm.yaml  diffusion.yaml  guide.yaml  oracle.yaml
│   ├── dna/       fm.yaml  guide.yaml  oracle.yaml
│   └── sampling/  base_comparison.yaml  vanilla_sweep.yaml
│                  ablation_grid.yaml  controls.yaml  dna_sweep.yaml
├── src/
│   ├── data/        qm9.py  enhancer.py
│   ├── models/      egnn.py  cnn1d.py  heads.py       # guide nets expose (reward, embedding)
│   ├── flows/       fm.py  diffusion.py
│   ├── guidance/    vanilla.py  ours.py  controls.py  # the innovation; comment heavily
│   ├── modality/    euclid_com.py  simplex.py         # projections
│   ├── sampling/    samplers.py                       # Euler / Heun / DDIM with guidance hooks
│   └── eval/        qm9_metrics.py  dna_metrics.py  fbd.py  quantum.py  pareto.py
├── scripts/         train.py  sample.py  evaluate.py  sweep.py  make_figures.py
├── results/         raw/  tables/  figures/
├── paper/
└── slides/
```

One YAML per experiment; every run logs its config, git hash and seed; `sweep.py` expands a grid; `make_figures.py` regenerates every table and figure from `results/raw`. The three guidance components are three functions whose docstrings quote the equations in Part 2.

## 7.4 Compute and environment

### Known hardware

**Henry's machine:** NVIDIA **RTX 5080, 16 GB VRAM** · AMD Ryzen 7 9800X3D, **8 cores / 16 threads** · 31 GB RAM.

This is enough to run Modality 1 end to end without cluster access. 16 GB is ample for EGNN on QM9 (small graphs, ≤29 atoms). The 8-core CPU handles the quantum-chemistry validation. **Collect everyone else's GPU situation** — a second GPU roughly halves wall-clock in Phase 1, which is the tightest week.

### Two environment traps — resolve these on day 1, before anything else

1. **The RTX 5080 is Blackwell (compute capability sm_120).** A default `pip install torch` may fetch a build without sm_120 kernels, which fails at runtime with *"no kernel image is available for execution on the device."* Install a CUDA 12.8-or-newer wheel explicitly:
   `pip install torch --index-url https://download.pytorch.org/whl/cu128`
   Verify immediately: `torch.cuda.is_available()` and a real matmul on the GPU, not just the import.
2. **Python 3.14 is on PATH and is probably too new** for the PyTorch / RDKit / PyG stack. **Use Python 3.11 or 3.12** in a dedicated environment. `conda` is not installed; either install Miniconda or use `py -3.12 -m venv`. Pin the version in `environment.yml`.

Budget half a day for this. Getting it wrong costs more.

### Betty / PARCC — the production resource

Penn's university-wide cluster: NVIDIA DGX SuperPOD, **31 eight-way GPU nodes**, SLURM, **$1 per GPU-hour and $0.01 per CPU-hour**. Docs at `parcc.upenn.edu`, including a "Zero to MNIST" getting-started guide and a multi-node training page.

At those rates our entire budget is roughly **$80 of GPU time and about $1 of CPU time**. The blocker is not cost, it is *an account to charge it to* — normally a faculty PI allocation. **Ask the instructors on day 1 whether the course provides one.** Access requests take time; this is the single most time-sensitive item in the plan.

**What Betty changes**

| | Single RTX 5080 | With Betty |
|---|---|---|
| Two base models | 32 h sequential — the schedule bottleneck | run concurrently, ~16 h, plus DNA and the predictors alongside |
| Ablation grid (~160 sampling cells) | most of a day | a SLURM job array — about an hour |
| Quantum chemistry (100 core-hours) | ~13 h wall-clock on 8 cores | minutes, at ~$1 |
| PyTorch setup | Blackwell/sm_120 trap (see above) | standard datacentre GPUs, stock wheels work |

**What it does not change:** the 29 Sep deadline, and the fact that writing becomes the bottleneck once compute stops being one.

### How to split the two

**Local RTX 5080 = development.** Writing code, debugging, the 2-D toy, smoke tests, quick sampling checks. The submit-wait-fail loop on a cluster is far slower than local iteration, and most of Phase 0 is iteration.

**Betty = production.** The base-model training runs, the full ablation grid as a job array, the M2 runs, and the quantum-chemistry validation.

Two cluster gotchas to check before relying on it: **compute nodes often have no internet**, so QM9, the enhancer data and any released checkpoints must be downloaded on a login node first; and confirm the **maximum walltime per job** — if it is under 16 h, base-model training needs checkpoint/resume built in from the start.

### Spend the surplus on rigour, not on more experiments

With Betty available the temptation is to add experiments. Resist it — every extra experiment adds a way to run out of time. Put the surplus into making the planned results stronger:

1. **Train the base models longer.** This directly retires the biggest risk in §8.1 (weak absolute numbers versus published baselines).
2. **2–3 training seeds** for the selected family, not one. Turns "one training seed, disclosed as a limitation" into a real error bar.
3. **Scale the DFT validation up** — 2,000 molecules per condition instead of 400. At $0.01 per CPU-hour this is nearly free, and it is the part of the paper no baseline has.

### Budget

| Job | Est. GPU-h |
|---|---|
| QM9 FM base / diffusion base | 16 each |
| $f_A$, $f_B$ — three properties ($\mu$, gap, $\alpha$) | 1.5 each × 6 = 9 |
| gap and $\alpha$ sweeps (three arms only, no ablation grid) | ~10 |
| DNA base + classifiers | ~6.5 |
| Base comparison sampling | 3 |
| Guidance + ablation grid (M1) — **×1.5 for the second VJP** (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.4) | 18–30 |
| M2 sampling | 4 |
| Debug margin | 15 |
| **Total** | **≈ 80 GPU-h (≈ $80 on Betty)** |
| Quantum chemistry | ≈ 100 CPU-core-hours (≈ $1 on Betty; ~13 h on 8 local cores) |

Measure actual sampling speed on day 1 and update this table. **Measure the guided step cost separately** — vanilla (1 backward) versus ours (2 backwards, shared forward). That ratio goes in the paper's compute table; a method that wins on the frontier while costing 1.5× per step must say so.

## 7.5 What the paper must contain

Five pages before references; appendix unlimited but the only evidence for a central claim never lives there.

| Section | Pages | Content |
|---|---|---|
| Abstract | 0.25 | 200–250 words, written last, no citations or equations |
| Introduction | 0.6 | three paragraphs then contribution bullets; state the hypothesis |
| Related work | 0.45 | frameworks · guidance and our honest positioning (§2.7) · M1 baselines · M2 baselines |
| Methods | 1.5 | modalities and data · FM · diffusion · guidance · **the innovation (largest subsection: the projection as a constrained optimum, the alignment argument, the pullback)** · transfer · overview figure |
| Results | 1.9 | protocol · base comparison · guidance works · **ablations + frontier figure** · external comparison · transfer · cross-modality and failure modes |
| Discussion | 0.3 | findings · why it worked or did not · limitations and the narrowest supported claim |

Two figures in the body: the overview (two panels — M1 development, M2 transfer, with trainable/frozen components labelled) and the Pareto frontiers with HV/DIP/R2 and bootstrap CIs (§3.6b).

**Two small mechanism plots go in the body if they fit, appendix otherwise** — they are what separate this from a tuning paper: the signed alignment against $t$ per property (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8), and the $\gamma$-sweep of MAE showing the unprojected arm's error floor rising while ours stays flat.

## 7.6 What the presentation must contain

Hard 15 minutes, ~3.5 min each. Slides must be labelled **1a → 3e** in the top-right corner, follow that order, carry numbered on-slide citations with a small source list bottom-right, end with a bibliography slide, and be delivered **without notes**. No slide changes after submission.

| Label | Content |
|---|---|
| 1a | problem, central question, both state spaces |
| 1b | datasets, splits, representations, conditioning, leakage |
| 1c | FM: path, target, loss, architecture, selection, sampler |
| 1d | diffusion: forward process, schedule, target, loss, sampler, selection |
| 1e | metrics and fairness (parameters, budget, NFE, seeds, hardware) |
| 1f | overview figure |
| 2a | base comparison table; which family and why |
| 2b | guidance rule, where it enters, frozen components; guided vs. unguided |
| 2c | the innovation: hypothesis, **the mechanism** with its two equations (projection + why it is not a no-op), the two adopted/engineering stabilisers named as such, new hyperparameters and which are measured rather than tuned |
| 2d | ablation table and frontier figure; what each row isolates |
| 2e | external comparison; quoted vs. reproduced; better / same / worse |
| 3a | what stays and what changes in transfer; M2 training and sampling |
| 3b | transfer results; fair-study disclosures |
| 3c | direction and magnitude across modalities — does it transfer? |
| 3d | at least two failure modes with samples |
| 3e | back to the hypothesis; mechanism, trade-offs, next experiments |
| — | Bibliography |

## 7.7 Failure modes to collect deliberately

Plan these; do not hope to stumble on them.

1. **The trust region caps achievable control.** At large $w$, vanilla reaches lower MAE than we can, at terrible validity. Show the frontier tails and report it as the trade-off it is.
2. **$N$ is fixed, so size-dependent properties are unreachable.** $\alpha$ is in the experiment grid for exactly this reason (§4.6): guidance should saturate far above the floor while $\mu$ reaches it. A measured failure, not an asserted one — plot MAE versus target for both and show the saturation.
3. **Reward hacking.** The $f_A - f_B$ gap grows with $w$; quantum chemistry confirms it. Show examples.
4. **Simplex decoding failures at high $w$** — mass collapsing onto a vertex too early. Plot one position's 4-vector over time.
5. **Repulsion toward outliers** with B on and A off — validity drops. The "Full − A" row is the evidence.
6. **Gate saturation.** If the prediction stabilises after ~15% of steps, C reduces to "skip the beginning" and control C2 will match it. Report that honestly; it is still a measured explanation of why limited-interval guidance works. C is not claimed as a contribution (§2.5), so this costs nothing.
7. **The alignment vanishes late in sampling.** `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8 predicts the benefit concentrates mid-trajectory: the overlap is large while the batch still has property spread and fades as guidance drives everything to $y^\ast$. Plot the signed alignment against $t$. If the benefit is confined to a time window, say so — and consider it as a cheap efficiency result (apply the projection only where it pays).
8. **The orthogonal subspace carries no chemistry.** If $f_A$ compressed too hard, spreading in $h^{\perp}$ moves the embedding without changing the molecule meaningfully (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §2.5). Symptom: embedding spread rises, scaffold count does not. This is exactly why scaffold count and coverage are the reported metrics and embedding spread never is.

---

# Part 8 — Risks and open questions

## 8.1 Risks

| Risk | Mitigation |
|---|---|
| Base models undertrained → weak absolute numbers | Fixed budget for every internal row; disclose both the budget and the 50k split; quote external numbers in clearly labelled rows. Our causal evidence is the ablation grid, which is internally matched and unaffected. |
| Metric bugs invalidate everything | Phase 0 gate: reproduce E(3)-EDM's released checkpoint numbers with our metric code *before* any model trains. |
| The innovation shows no gain | Run `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md` first. **We now carry one claim, not three, so a null is a real possibility** — which is exactly why T0 exists and why the fallbacks in the feasibility doc are ranked in advance. A diagnosed negative with the alignment measurement to explain it is still a complete project. Decide framing 22 Sep, not 27 Sep. |
| The projection turns out to be a no-op | T0's signed-alignment diagnostic tells us within hours, before anything is committed (`COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.8). |
| Someone published this already | §2.7 lists what we found and what remains unread. Finish those readings this week. |
| FBD not comparable | Verify Dirichlet FM released its classifier; otherwise label all FBD as internal. |
| GPU access thinner than expected | Cut the NFE sweep to {20, 100}; grid to 5k samples; drop optional ablation rows first, never the frontier. |
| Scope creep | Nothing outside Part 4/5 happens; nothing new after 25 Sep. |
| Q&A exposure on unfamiliar parts | Rotating presentation assignment (§7.2); mock Q&A 29–30 Sep. |

## 8.2 Verify before use

Everything here came from memory or a search summary. Tick each off against the source.

**Datasets and protocols**
- [ ] QM9 split sizes (100k / 17,748 / 13,083) and official split files
- [ ] E(3)-EDM config (9 layers, 256 hidden), parameter count, noise schedule, normalisation constants
- [ ] E(3)-EDM released QM9 checkpoint and property classifiers — available? usable as our oracle?
- [ ] Real-QM9 atom stability under our metric code (the ceiling, ~97–99%?)
- [ ] Melanoma / fly-brain sizes, length, class counts, split files; **how Dirichlet FM prevents homology leakage**
- [ ] Dirichlet FM: released FBD classifier? checkpoints? CFG guided-generation numbers, guidance scales, sample counts
- [ ] Fisher Flow, Gumbel-Softmax FM, DDSM: which enhancer numbers exist, under which protocol

**Numbers**
- [x] Verified 13 Sep via PropMolFlow Table 1: cond-E(3)-EDM $\mu$ 1.111, EEGSDE 0.777, JODO 0.620, PropMolFlow 0.620, lower bound 0.043, random 1.616; $\alpha$ lower bound 0.10, random 9.01
- [ ] "#atoms" baseline (≈1.05 for $\mu$), and $\alpha$ values for cond-EDM (2.76) and EEGSDE (2.50)
- [ ] TFG-Flow Table 1 ($\mu$ 0.817, $\alpha$ 2.32), its protocol and 4,096-sample count, code availability
- [ ] OC-Flow ($\mu$ 0.314, $\alpha$ 1.383): which predictor, and its per-sample cost

**Novelty — finish these before claiming anything in §2.7. Highest priority in the document.**
- [ ] **OSCAR [2510.09060]** — read the method section **in full**. This is the closest work and it owns the trust region verbatim. Confirm: projects against $v_	heta$ not $
abla r$; no reward model; images only; what exactly its preservation proposition states. **If any of that is wrong, §2.8 changes materially and so does the project.**
- [ ] **O-SVGD (NeurIPS 2022)** — confirm the orthogonal decomposition is for constrained *sampling* and does not involve a learned reward embedding
- [ ] **Particle Guidance [2310.13102]** — read the method section. Does its feature-space kernel project anything?
- [ ] **EDDY [2605.06553]** — read in full; confirm the marginal-vs-reward distinction holds
- [ ] **APG [2410.02416]** — confirm its rescaling is not already a velocity-relative cap
- [ ] **Don't Settle at the Mode [2606.27371]** — confirm images-only, no reward-model embedding
- [ ] **MolGuidance [2512.12198]** — confirm it reports diversity and property error but never crosses them into a frontier. This is the "nobody has done the plot" claim.
- [ ] Search once more for reward-orthogonal / null-space diversity guidance, and for anyone projecting a sampling-time update **after** a clean-prediction pullback (the `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.4 correctness claim)

**Citations to add to `citation.bib`** — all of the above plus DPS, Universal Guidance, FreeDoM, MPGD, DSG, PCGrad, Perp-Neg, **OSCAR**, **O-SVGD**, **Vendi Score**, EEGSDE, Guided Flows, TFG, TFG-Flow, OC-Flow, VAGS, STRIDE, PropMolFlow, JODO, MolGuidance, EGNN, E(3)-EDM, Naeem coverage/density.

---

# References

Verify all arXiv IDs and years before they enter the bibliography.

| Short name | Identifier |
|---|---|
| EGNN — Satorras et al. 2021 | arXiv 2102.09844 **(verify)** |
| E(3)-EDM — Hoogeboom et al., ICML 2022 | arXiv 2203.17003 **(verify)** |
| EDM (images) — Karras et al., NeurIPS 2022 | in `citation.bib` as `karras2022elucidating` |
| EEGSDE — Bao et al., ICLR 2023 | arXiv 2209.15408 |
| Particle Guidance — Corso et al. 2023 | arXiv 2310.13102 |
| PCGrad — Yu et al. 2020 | arXiv 2001.06782 |
| Limited-interval guidance — Kynkäänniemi et al., NeurIPS 2024 | arXiv 2404.07724 |
| TFG — Ye et al., NeurIPS 2024 | arXiv 2409.15761 |
| APG — Sadat et al. 2024 | arXiv 2410.02416 |
| OC-Flow | arXiv 2410.18070 |
| TFG-Flow — ICLR 2025 | arXiv 2501.14216 |
| Dirichlet FM — Stark et al. 2024 | arXiv 2402.05841 |
| PropMolFlow 2025 | arXiv 2505.21469 |
| STRIDE 2026 | arXiv 2605.11494 |
| VAGS 2026 | arXiv 2605.15661 |
| EDDY 2026 | arXiv 2605.06553 |
| **OSCAR** (closest related work) | arXiv 2510.09060 **(verify — read in full, §8.2)** |
| **O-SVGD** — orthogonal-space SVGD, NeurIPS 2022 | **(verify identifier)** |
| **Vendi Score** — Friedman & Dieng 2022 | arXiv 2210.02410 **(verify)** |
| Don't Settle at the Mode 2026 | arXiv 2606.27371 |
| MolGuidance 2026 | arXiv 2512.12198 |
| Density/coverage metrics — Naeem et al. 2020 | arXiv 2002.09797 **(verify)** |

Already in `proj1_tex/citation.bib`: Lipman flow matching, Karras EDM, stochastic interpolants, Rectified Flow, OT-CFM, Multisample FM, SiT, CFG, Guided Flows, Dirichlet FM, Gumbel-Softmax FM, Fisher Flow, GeoDiff, E(3)-EDM, SemlaFlow, and the lab's MOG-DFM / AReUReDi / PepTune / TR2-D2 / moPPIt.
