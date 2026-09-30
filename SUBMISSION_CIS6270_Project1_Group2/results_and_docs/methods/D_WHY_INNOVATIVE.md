# D (Tolerance Band) — why innovative?

## The safety net: explained, proved where possible, and compared with prior work

**Prepared 18 September 2026.** Companion to [SMG_WHY_INNOVATIVE.md](SMG_WHY_INNOVATIVE.md),
written to the same standard: exact identities, controlled approximations, design choices and
untested hypotheses are kept apart, and the title is a question rather than a claim.

**The answer we can currently defend.** D is an **adaptation**, and a narrower one than SMG.
Its geometry is a solved convex problem and we have a complete proof plus two independent
numerical verifications of the solver. What is ours is the *constraint we put into that
geometry* — a per-anchor, baseline-relative progress envelope — together with a finite
acceptance check against the real guide. Whether that improves decoded diversity, accuracy,
validity or yield is **unmeasured**. Nothing below claims priority.

**Why we carry it anyway.** D is the safety net. It is gated on a different question from SMG,
so the two cannot fail for the same reason. SMG asks *is there a Jensen gap worth correcting?*
D asks *is there unused tolerance worth spending?* If M-1 kills SMG, D is still a live
methodological contribution and the project still has an innovation to ablate.

**Reading map**

1. [The idea in one page](#1-the-idea-in-one-page)
2. [Notation](#2-notation)
3. [The band, and what each regime means](#3-the-band-and-what-each-regime-means)
4. [The local problem and its exact solution](#4-the-local-problem-and-its-exact-solution)
5. [Proofs](#5-proofs)
6. [Two traps that shape every experiment](#6-two-traps-that-shape-every-experiment)
7. [What is taken, and what is ours](#7-what-is-taken-and-what-is-ours)
8. [What is not proved](#8-what-is-not-proved)
9. [How it would be tested and decided](#9-how-it-would-be-tested-and-decided)

---

## 1. The idea in one page

Property-targeted generation is usually written as *hit this value*. Applications almost never
want that. A chemist wants a dipole moment **in a usable range**; a target of 5.00 D with
[4.90, 5.10] acceptable is a band, not a point.

Guidance that preserves the property exactly throws that room away. If you are already at 5.00
and you hold the property fixed while doing anything else — diversifying, repelling from the
batch, exploring — you forbid a move to 5.04 even though 5.04 was always acceptable. The
forbidden move may be the only one that carries real structural change.

**D spends the band instead of protecting a point.** At each step it solves a small convex
program: move as far as you can in a useful direction (here, increasing batch diversity),
subject to (i) staying in the admissible subspace, (ii) not pushing the property outside its
allowed envelope, and (iii) a step-size cap. Then — because the constraint was linearised — it
*checks the finite edit against the actual guide* and rolls back if the check fails.

The three parts, and their honest status:

| Part | Status |
|---|---|
| Convex ball ∩ slab ∩ subspace projection | solved problem; our proof and solver are exact and verified |
| The envelope $A=\max(\tau,\lvert e\rvert)$ put into the slab | **the candidate contribution** |
| Finite acceptance check with rollback | ours in this setting; ordinary line-search practice elsewhere |
| Any claim about better molecules | **nothing. Unmeasured hypothesis** |

### 1.1 Where D runs

Inference time, inside a frozen sampler. No retraining, no change to the generator or the
property guide. That is the same operating regime as SMG, which is why the two compose: SMG
changes *where a trajectory is steered*, D changes *how much freedom the trajectory has while
being steered*.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| $x$ | sampler state at time $t$ — the only thing we can move |
| $E_t(x)=x+(1-t)v_\theta(x,t)$ | endpoint proxy: a conditional-mean estimate, **not** the ODE endpoint |
| $F_t(x)=f_A(E_t(x))$ | the guide's predicted property at the current state |
| $e=F_t-y^\ast$ | signed property error |
| $b=Q\nabla_x F_t$ | property gradient, projected into the admissible subspace: the level-set normal |
| $u=Q\nabla_x \mathcal D$ | diversity gradient, projected: batch-coupled |
| $Q$ | symmetric orthogonal projector onto admissible state directions (zero-CoM, padding, simplex) |
| $\tau$ | the allowed guide error — the tolerance, and the primary scientific axis |
| $\delta$ | the **independent evaluator's** acceptance tolerance, fixed across all arms |
| $\eta,\ R$ | step scale and edit radius |

$\tau$ and $\delta$ are different on purpose. $\tau$ is what we *let* the guide drift; $\delta$
is what the held-out evaluator will *accept*. Conflating them would let the method grade its own
homework.

---

## 3. The band, and what each regime means

At each anchor, with $\tau\ge0$:

$$A=\max(\tau,\lvert e\rvert),\qquad \ell=-A-e,\qquad h=A-e
\qquad\Longrightarrow\qquad \ell\ \le\ b^\top d\ \le\ h$$

Read $b^\top d$ as *the first-order change in the property caused by the edit $d$*.

| Regime | Interval | What it permits |
|---|---|---|
| $e=0,\ \tau=0$ | $[0,0]$ | equality — **only** here |
| $e>0,\ \tau=0$ | $[-2e,\ 0]$ | no worsening; inward movement allowed |
| $e<0,\ \tau=0$ | $[0,\ -2e]$ | mirror image |
| $\lvert e\rvert\le\tau$ | $[-\tau-e,\ \tau-e]$ | free movement inside the band |
| $\lvert e\rvert>\tau$ | $A=\lvert e\rvert$ | **$\tau$ is inert**; identical for every such $\tau$ |

The envelope is *baseline-relative*: when the guide is already outside the tolerance, $A$ tracks
the current error rather than $\tau$, so the rule becomes "do not make it worse" instead of
"get inside the band immediately." This is the part that is not the optimum set of any fixed
loss, and §7 is where that matters.

**Worked example.** `[CHECKED]` $e=0.2$, $b=(1,0)$, $u=(-1,1)$, $\eta=1$, $R=0.1$. The $\tau=0$
band returns $d=(-0.0707,\,0.0707)$ with property change $-0.0707$ — it *improves* the error while
diversifying. Strict equality returns $d=(0,\,0.1)$ with property change $0$. Same anchor, same
budget, different answers.

---

## 4. The local problem and its exact solution

$$d^\star=\arg\max_d\Big[u^\top d-\tfrac{1}{2\eta}\lVert d\rVert^2\Big]
\quad\text{s.t.}\quad Qd=d,\quad \ell\le b^\top d\le h,\quad \lVert d\rVert\le R$$

This is a nearest-point projection onto {ball ∩ slab ∩ subspace}, which is convex, so the
solution is unique and has a closed form. With $r=\eta u$, $n=b/\lVert b\rVert$, and $w$ the
projection of $r$ onto the ball inside the subspace: return $w$ if it already satisfies the slab,
otherwise

$$s=\mathrm{clip}\!\left(n^\top w,\ \tfrac{\ell}{\lVert b\rVert},\ \tfrac{h}{\lVert b\rVert}\right),
\qquad
d=s\,n+\min\!\left(1,\ \tfrac{\sqrt{\max(0,\,R^2-s^2)}}{\lVert v_r\rVert}\right)v_r,
\qquad v_r=r-(n^\top r)\,n$$

No division by the active bound, so the degenerate cases need no special handling beyond
$\lVert b\rVert\approx0$, which returns $d=0$ and is logged.

### 4.1 The diversity objective

$$\mathcal D(Z)=\log\det\big(K(Z)+\lambda_K I\big),\qquad
K_{ij}=\exp\!\left[-\frac{\lVert h(E(z_i))-h(E(z_j))\rVert^2}{2\sigma^2}\right]$$

Via Cholesky, never a raw determinant. Cost $O(B^2k)$ for the distances, $O(B^3)$ for the
factorisation. $h$ is an **invariant embedding**, not raw coordinates: two configurations far
apart in $\mathbb{R}^{3N}$ may be one structure rotated or relabelled, and a Euclidean kernel
would score that as diversity.

$\sigma$ is cached at the anchor and reused for every acceptance trial. Otherwise the guard
compares two different objectives and its decision means nothing.

### 4.2 The finite acceptance guard

The band constrains $b^\top d$, which is only the **first-order** property change. The real
change is $F_t(z+d)-F_t(z)=b^\top d+O(\lVert d\rVert^2)$, so a linearly-feasible edit can still
leave the band. The guard is what makes the method honest: try $\alpha\in\{1,\tfrac12,\tfrac14\}$,
re-evaluate the *actual* endpoint map and guide at $Z+\alpha d$, and accept only if for every
sample

$$\lvert F_{t'}(z_i+\alpha d_i)-y^\ast\rvert\ \le\ A_i+\epsilon_{\text{num}}$$

and the diversity actually improved by an Armijo-style margin. Otherwise return $Z$ unchanged.
A rejected edit reproduces the baseline anchor **exactly**, which keeps the arms comparable.

**Rejecting every trial can be correct.** `[PROVED]` For $F(x,y)=x^2+y^2$ at anchor $(1,0)$ with
target $0$ and tolerance $1$, the tangent direction $(0,1)$ satisfies the linearised band, but
every positive step gives $F=1+\alpha^2>1$. So a high rejection rate is a *finding* about
curvature, not a bug — and near-total rejection means the controller is inert and D is dead.
That distinction is why rejection rate is a logged metric and not a diagnostic afterthought.

---

## 5. Proofs

| Claim | Status |
|---|---|
| $\ell\le0\le h$ always, so $d=0$ is feasible and the program is never infeasible | `[PROVED]` |
| The closed-form ball/slab solve is exact; the clipped normal coordinate is optimal | `[PROVED]` `[CHECKED]` twice, independently |
| Local ascent: $u^\top d\ \ge\ \lVert d\rVert^2/\eta\ \ge\ 0$ | `[PROVED]` |
| The band's optimised **local** objective is $\ge$ equality's at the same $\eta,R$ | `[PROVED]`, scoped |
| Projection order: gradients must enter the subspace before the tangent step | `[PROVED]` `[CHECKED]` |
| Better decoded diversity, accuracy, validity or yield | **nothing.** `[HYPOTHESIS]` |

### 5.1 Feasibility

$A=\max(\tau,\lvert e\rvert)\ge\lvert e\rvert$, so $\ell=-A-e\le0$ and $h=A-e\ge0$. Hence
$\ell\le0\le h$, so $b^\top 0=0$ satisfies the slab, and $d=0$ satisfies the ball and the
subspace trivially. The feasible set always contains the zero edit. $\square$

This is why D can never stall a sampler: the worst case is that it does nothing.

### 5.2 Exactness of the solve

1. $w$ is the unique ball projection of $r$ within the subspace. If $w$ lies in the slab, it
   solves the intersection problem directly.
2. Suppose $n^\top w>U$ (the upper bound). An intersection optimum $d$ with $n^\top d<U$ is
   impossible: by convexity and uniqueness of the ball optimum, a sufficiently small move from
   $d$ toward $w$ stays in both the ball and the slab and *strictly* decreases
   $\lVert\cdot-r\rVert^2$, contradicting optimality.
3. So the upper face is active. Fix the normal coordinate at $U$ and project the tangent
   component of $r$ onto the remaining tangent ball of radius $\sqrt{R^2-U^2}$. The lower face
   is symmetric. $\square$

**Verification, twice and independently:**

- `audit/test_tolerance_controller.py` — 1,000 random 12-dimensional problems, 100,000 feasible
  comparisons, 120 dense reduced-coordinate references. Max feasibility residual
  $\approx4.6\times10^{-15}$.
- `audit/ball_slab_bruteforce_check.py` — a separate pure-Python brute force, 4,000 problems in
  2–8 dimensions against a dense scan plus random feasible search. Zero infeasible results,
  worst constraint excess $1.8\times10^{-12}$.

The second exists because agreeing with your own implementation proves nothing.

### 5.3 Local ascent

From the projection inequality with $v=0$ and $r=\eta u$: $u^\top d\ge\lVert d\rVert^2/\eta\ge0$.
So the edit never *decreases* the local diversity objective. $\square$

### 5.4 Dominance over equality, and exactly how weak that is

The equality-constrained set is a subset of the band set, so the optimised **local regularised
objective** under the band is at least as large. $\square$

That sentence is doing much less work than it appears to. The sets need not differ — at
$e=\tau=0$ they coincide. It is a statement about a *linearised local objective at one step*, not
about final diversity, property error, validity, or wall-clock. And it is only a fair comparison
at identical $\eta$ and $R$. **This proof is not evidence that D produces better molecules.**

### 5.5 Projection order

Project both gradients into the subspace *first*: $b=Q\nabla F_t$, $z=Q\nabla\mathcal D$, then
$d=z-\frac{b^\top z}{b^\top b}b$ satisfies both $Qd=d$ and $\nabla F_t^\top d=0$. The joint
projector is $Q-bb^\top/(b^\top b)$. Reversing the order leaves a residual — $-0.333$ in our
reference example `[CHECKED]` — meaning the edit silently violates either the subspace or
property neutrality.

On the simplex this becomes load-bearing rather than tidy: sum-zero is **necessary but not
sufficient**, because feasibility at the boundary also needs inward-or-tangent directions at
zero coordinates. That is the open derivation for Modality 2, and it is gated separately.

---

## 6. Two traps that shape every experiment

Both are proved, both have burned us already, and either one silently invalidates a $\tau$ sweep.

**Trap 1: $\tau=0$ is not equality.** `[PROVED]` `[CHECKED]` At $e>0$ the constraint is
$-2e\le b^\top d\le0$ — *no worsening*, with inward movement allowed. Equality holds only at
$e=0$ exactly. So "set $\tau=0$ to recover the strict-preservation baseline" is **wrong**, and
the equality arm has to be built as its own code path.

**Trap 2: $\tau$ is inert whenever $\lvert e\rvert>\tau$.** `[PROVED]` Because $A=\lvert e\rvert$
for every such $\tau$, a whole sweep over $\tau$ can produce identical results — not because
tolerance does not help, but because the tolerance never activated. **A null result is
uninterpretable without band-activity logging.** So we log, per step: slack, which constraints
are active, band activity, accepted nonzero edits, and rejection rate.

### 6.1 A related trap in the $\rho$ diagnostic

$\rho=(b^\top u)^2/(\lVert b\rVert^2\lVert u\rVert^2)\in[0,1]$ measures how much of the diversity
gradient points along the property gradient. It is a useful diagnostic for the *equality* arm and
we verified its identity to $1.3\times10^{-11}$ over 200,000 instances. Four limits, each with a
counterexample in `audit/review_v4_checks.py`:

1. **Avoided change is not avoided harm.** The bound concerns the *magnitude* of avoided property
   change, not benefit. An unprojected edit can *improve* the error while the tangent edit holds
   it fixed. Interference is harmful only when $e$ and $b^\top u$ share a sign.
2. **No benefit-per-unit-diversity peak.** $\mathcal D/G^\ast$ is monotone in $\rho$; the apparent
   peak at $\rho=\tfrac12$ is an artefact of the comparison point.
3. **$\rho$ alone cannot rank properties** — it depends on units, anchor error, curvature and
   guide calibration.
4. **$\rho\to1$ does not condemn D.** `[CHECKED]` At $e=0$, $\tau=0.2$, $b=u=(1,0)$, $R=0.1$:
   equality gives $d=0$ and zero gain, while the band permits $d=(0.1,0)$ with gain $0.1$. That
   is precisely the case where tolerance rescues a direction equality blocks.

We over-read $\rho$ once already. It is a diagnostic for one arm, not a decision rule for D.

---

## 7. What is taken, and what is ours

### 7.1 Taken, cited, claimed for nothing

| Ingredient | Owner |
|---|---|
| Baseline descent step then a diversity correction under analytic constraints — including diversity *inside a loss's optimum set*, with molecular conformation experiments | [Harmless Diversity, ICML 2022](https://proceedings.mlr.press/v162/gong22b/gong22b.pdf), eqs 4–6, Algs 1–2 — **the closest prior work** |
| Normal/tangent decomposition for constrained particle sampling | [O-SVGD, NeurIPS 2022](https://arxiv.org/abs/2210.06447) |
| Repulsive joint-particle sampling on molecular conformers | [Particle Guidance, ICLR 2024](https://arxiv.org/abs/2310.13102) |
| Endpoint feature diversity, state-space pullback, velocity-relative cap | [OSCAR, ICML 2026](https://arxiv.org/abs/2510.09060) |
| Reward-guided sampling with a repulsive correction, on molecules | [Stein Diffusion Guidance, ICML 2026](https://arxiv.org/abs/2507.05482) |
| Diverse particles under inequality constraints | [Constrained Stein Variational Trajectory Optimization](https://arxiv.org/abs/2308.12110) |
| Training-free constrained diffusion; trust-region guidance | [Trust Sampling, NeurIPS 2024](https://arxiv.org/abs/2411.10932); [Constrained Diffusers](https://arxiv.org/abs/2506.12544) |
| Inference-time gradient surgery | Perp-Neg, APG, PDGrad |

Components **A** (velocity-relative trust region) and **B** (level-set tangent projection) are
prior work outright. **C** (endpoint-stability gate) is engineering in a crowded category. Only
D carries a claim.

### 7.2 Two boundaries that do NOT separate us

Stated plainly, because both look like distinctions and neither is:

1. **`[PROVED]`** The tolerance band **is exactly** the optimum set of the dead-zone loss
   $\max(0,\lvert F-y^\ast\rvert-\tau)^2$ *when the band is reachable*. So "we use a tolerance
   band rather than a loss's optimum set" is not a distinction. Harmless Diversity already
   diversifies inside an optimum set.
2. $F=f_A(E_t(x))$ is a function of the state. "Through a network" rather than "on the state" is
   not a distinction either.

### 7.3 What actually survives

| | Harmless Diversity | Component D |
|---|---|---|
| Setting | population gradient descent on an objective | an edit inside a **frozen generative sampler at inference** |
| Constraint | analytic, on the main loss | a **per-anchor progress envelope** $A=\max(\tau,\lvert e\rvert)$ — baseline-relative, so **not** the optimum set of a fixed loss when $\lvert e\rvert>\tau$ |
| Verification | analytic | a **finite acceptance check** against the actual guide, with rollback |
| Domain | images, meshes, conformers, ensembles | 3D property-targeted de novo molecules; simplex sequences |

The one clause that carries the claim is the second row's second half. When the guide is outside
tolerance, $A$ tracks the current error, and that envelope is not any fixed loss's optimum set —
which is exactly where boundary (1) stops applying.

One further gap survived a bounded literature search: we found **no diversity-versus-property-error
frontier for property-targeted 3D molecular generation**. MolGuidance reports both axes but
separately. That states what a bounded search found; it is not a claim of absence.

### 7.4 Honest assessment

For the course, the brief permits combining and adapting prior work provided the report explains
what is new in the formulation — and §7.3 is that explanation. For a publication the mechanism
alone would be insufficient; it would need empirical results, a demonstrated failure of the
alternatives, or an efficiency finding. **This document certifies priority for nothing.**

---

## 8. What is not proved

- **No molecular improvement.** Not diversity, not accuracy, not validity, not yield.
- The band constrains a **linearisation**. The guard catches violations but does not prevent them,
  and the rejection rate is unknown until measured.
- The norm ball is **not** evidence the edit stays on the data manifold.
- The guard does **not** guarantee chemical validity — decoding is discontinuous.
- The centroid-repulsion motivation assumes the batch prediction mean sits near the target. That
  is an empirical condition, not a consequence of sharing a target.
- The simplex formulation for Modality 2 is **open**: a covariance-style step preserves the
  coordinate sum but can drive coordinates negative, so boundary feasibility needs its own
  derivation.
- A globally valid Hessian bound is unavailable, so the second-order remainder is controlled only
  by the edit radius and ultimately by the finite guard.

---

## 9. How it would be tested and decided

Full protocol, metrics and decision tables in
[FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md](../../archive/FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md), sections S0–S2 and D-1
to D-3; the schedule is in [PLAN_AND_TIMETABLE.md](../status/PLAN_AND_TIMETABLE.md).

**D-1, the gate — does the controller do anything at all?** One batch, one condition. Log band
activity, active constraints, slack, accepted nonzero edits and rejection rate.

| Outcome | Decision |
|---|---|
| Band active on a meaningful fraction of steps, edits accepted | **proceed to D-2** |
| Band never activates ($\lvert e\rvert>\tau$ throughout) | **not a null result** — re-choose $\tau$ against the measured error distribution, then retest |
| Nearly every edit rejected | controller is inert; report the curvature finding and **stop** |
| Edits accepted but diversity unchanged | the objective or the embedding is wrong, not the band |

**The arms that can falsify D**, and which we will not drop:

1. **The dead-zone loss control.** Since the band *is* that loss's optimum set when reachable,
   an explicit dead-zone-loss baseline is the sharpest comparator we have.
2. **Generate-and-select at equal wall-clock.** Sample more, keep the in-band ones, and see
   whether the controller beats simply doing more of the cheap thing. If it does not, D has no
   case regardless of its geometry.
3. **The equality arm, built separately** — not $\tau=0$ (Trap 1).

**Metrics** are the shared S2 set, so D and SMG are judged identically: molecule/atom stability
and validity against the real-data ceiling (96.0% / 98%), property MAE and in-band fraction under
the **held-out evaluator** $f_B$ at $\delta=0.168$ D, the $f_A-f_B$ gap as the reward-hacking
signature, and diversity as both mean pairwise embedding distance and Gram log-determinant.

**What would make us drop D:** near-total guard rejection, or generate-and-select matching it at
equal wall-clock. Either is a reportable finding rather than a wasted week.
