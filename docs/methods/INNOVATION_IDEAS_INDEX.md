# Innovation catalogue

**Proposed training-time branch (18 Sep):** [Innovation v8](INNOVATION_V8_TRAINING_TIME_GUIDANCE.md)
develops a coherent posterior guidance operator and guidance-aware partition learning, with
proofs, a prior-work audit, and [separate feasibility gates](TRAINING_GUIDANCE_FEASIBILITY.md).
These are provisional candidates, not novelty-cleared replacements for the current plan.

**v2.1 · updated 18 Sep 2026.** Every innovation idea currently live in the project, named, with its proof or logic, and a novelty verdict against four criteria with citations.

**SMG update:** primary innovation under evaluation; research novelty is not established.
[SMG, Heun, and prior work](SMG_PRIOR_WORK_AUDIT.md) records the formula-level comparison.
Heun is adopted prior work and a required solver control in `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md` M-4.
Earlier rankings reflect the initial brainstorm, not verified molecular results or measured costs.

This is the index. Detail lives elsewhere: **`COMPONENTS_ABCD_MATH_AND_PROOFS.md`** (A–D mathematics), **`BPS_BALANCED_PREVIEW_SAMPLING.md`** (BPS), **`SMG_FULL_MATH_AND_PROOFS.md`** (SMG), **`PCG_BRAINSTORM_DROPPED.md`** (PCG), **`D_AND_TRE_REVIEW_FEEDBACK.md`** (TRE origin), **`FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md`** (what to run, per point).

## Naming

Existing letters are preserved. New items get names rather than letters so they are visually distinct.

| ID | Name | Kind | First appeared |
|---|---|---|---|
| **A** | Velocity-Relative Trust Region | method component | v0.1 |
| **B** | Level-Set Tangent Projection | method component | v0.1 |
| **C** | Endpoint-Stability Gate | method component | v0.1 |
| **D** | Tolerance Band | method component | 15 Sep |
| **TRE** | **Terminal-Response Edits** | **method candidate — new** | **17 Sep** |
| **BPS** | **Balanced Preview Sampling** | **method candidate — new** | **17 Sep** |
| **SMG** | **Second-Moment Guidance** | **primary method candidate; novelty unconfirmed** | **17 Sep** |
| **PCG** | **Covariance-Preconditioned CFG** | **method candidate — new** | **17 Sep** |
| **RCH** | **Residual Calibration Head** (Haimo v1) | **comparator arm for SMG — prior work, never a claim** | **18 Sep** |
| **RHO** | The ρ removed-energy identity | analysis result | 16 Sep |
| **CX-1 … CX-7** | Correctness results | analysis results | 15–16 Sep |
| **YIELD** | Within-condition useful-yield protocol | evaluation protocol | 15 Sep |

## Status labels

`[PROVED]` derivation given · `[CHECKED]` verified numerically, script named · `[ASSUMPTION]` required, not established · `[HYPOTHESIS]` an experiment decides · `[DESIGN]` a choice · `[TAKEN]` prior work

---

# Part 1 — The method components

## A — Velocity-Relative Trust Region

**What it does.** Caps the guidance term at a fixed fraction of the model's own velocity: $g \leftarrow g\cdot\min(1,\ \rho_A\lVert v\rVert/\lVert g\rVert)$ when $\lVert g\rVert>0$.

**Logic.** $\lVert v\rVert$ varies by orders of magnitude across $t$, so an absolute clip is wrong nearly everywhere. A relative cap keeps the update from overwhelming the model's own dynamics at every time.

**Proof status.** Nothing to prove — it is a clip. Two derived rules do carry proofs: cap the baseline **only** (CX-5), and the cap is what makes B's first-order argument valid over a finite step.

**Status:** `[TAKEN]`. Not a contribution. Cite in Methods, not only Related Work.

## B — Level-Set Tangent Projection

**What it does.** Removes the component of the diversity direction that lies along the property normal: $d = u - \frac{b^\top u}{b^\top b}b$, so $b^\top d=0$.

**Logic.** The target is a surface, not a point. Travel *across* level sets to reach the target; travel *along* the surface to diversify. A tangent move changes the property by zero to first order.

**Proof.** `[PROVED]` For $b\neq0$, $d$ is the **unique** nearest vector to $u$ satisfying $b^\top d = 0$ — projection onto a closed convex set is always unique, including when $d=0$.

> **Correction to `COMPONENTS_ABCD_MATH_AND_PROOFS.md` §3.1.** That section says the projection is "non-unique when $d=0$." That is wrong. What is non-unique is the *different* problem $\max\langle u,\delta\rangle$ s.t. $b^\top\delta=0$, $\lVert\delta\rVert\le1$: when $u_\perp=0$ every feasible unit $\delta$ scores zero. State the two separately.

**Status:** `[TAKEN]` as a mechanism. Its general form is owned four times over. Ours are two *conditions on using it correctly* (CX-1, CX-3), not the projection.

## C — Endpoint-Stability Gate

**What it does.** Damps guidance in proportion to how much the endpoint prediction moved since the previous step: $c_i = 1/(1+\Delta_i/\tau_C)$.

**Logic.** Early in sampling the endpoint estimate is a smeared average, so guiding hard on it guides on a hallucination — the main route to off-manifold structures. $\tau_C$ is set from the median $\Delta$ of one unguided run, so it is measured rather than tuned.

**Proof status.** None. It is a heuristic formula, and endpoint stability is not calibrated accuracy — the proxy can be stably wrong.

**Status:** `[DESIGN]`. Ablation row only; never claimed.

## D — Tolerance Band

**What it does.** Replaces B's equality constraint with a band whose width comes from the application. At each anchor, with $A=\max(\tau,\lvert e\rvert)$:

$$-A-e\ \le\ b^\top d\ \le\ A-e$$

then solves $\max_d\,[u^\top d-\lVert d\rVert^2/2\eta]$ over {ball ∩ slab ∩ subspace}, and verifies the finite edit against the actual guide with rollback.

**Logic.** Nobody needs the property exactly; they need it inside a tolerance. Strict preservation discards that room.

**Proofs.**

| Claim | Status |
|---|---|
| $\ell\le0\le h$ always, so the zero edit is always feasible and the problem is never infeasible | `[PROVED]` |
| The closed-form ball/slab solve is exact; the clipped normal coordinate is optimal | `[PROVED]` `[CHECKED]` — complete convexity argument; 1,000 12-dim problems + 4,000 2–8-dim brute force, worst excess $1.8\times10^{-12}$ |
| Local ascent: $u^\top d\ge\lVert d\rVert^2/\eta\ge0$ | `[PROVED]` |
| The band's optimised **local** objective is ≥ equality's at the same $\eta,R$ | `[PROVED]`, scoped — sets coincide at $e=\tau=0$ |
| Better decoded diversity, accuracy, validity or yield | **nothing.** `[HYPOTHESIS]` |

**Two facts that shape any experiment.**

`[PROVED]` `[CHECKED]` **$\tau=0$ is not equality.** At $e>0$ the constraint is $-2e\le b^\top d\le0$ — *no worsening*, inward movement allowed. Equality holds only at $e=0$. Build the equality arm separately.

`[PROVED]` **$\tau$ is inert whenever $\lvert e\rvert>\tau$**, since $A=\lvert e\rvert$ for all such $\tau$. A null sweep may mean the tolerance never activated. Log band activity.

**Status:** adaptation. The band **is** the optimum set of the dead-zone loss $[\max(0,\lvert F-y^\ast\rvert-\tau)]^2$ `[PROVED]`, so "tolerance rather than optimum set" is not a distinction.

## TRE — Terminal-Response Edits **(new)**

**What it does.** Estimates both the property cost and the diversity benefit of a proposed edit **after the remaining sampler**, rather than at the current step, then feeds those estimates into D's constrained solve.

1. At one late editing opportunity, let $R_t(z)$ be the remaining baseline sampler to the terminal continuous output.
2. Choose an admissible subspace $S_i$ (orthonormal, $QS_i=S_i$), 2–4 directions: the diversity direction, the property normal, one admissible random direction.
3. For each column, run **paired suffix rollouts** from $z_i\pm\epsilon s_{ij}$ and finite-difference the terminal property and terminal features:
   $$\hat b_{ij}=\tfrac{P_i(z_i+\epsilon s_{ij})-P_i(z_i-\epsilon s_{ij})}{2\epsilon},\qquad \hat u_{ij}=(\nabla_{H_i}D)^\top\hat J_{H,ij}$$
4. Solve D's ball/slab problem in the **coefficient** space $a_i$ (valid because $S_i$ is orthonormal, so $\lVert d_i\rVert=\lVert a_i\rVert$).
5. Verify by running the actual continuations; fall back to the cached baseline continuation if every trial fails.

**Logic — the failure it targets.** A local guard need not protect the terminal property, and the locally best spread direction need not be the terminally best one.

`[PROVED]` **Local guard ≠ terminal guard.** $F_{\text{local}}(x)=x_1$, target 0, remaining map $R(x)=(x_1+2x_2,\ x_2)$. At the origin, $d=(0,0.1)$ leaves the local property exactly unchanged but produces terminal error $0.2$, exceeding a tolerance of $0.05$.

`[PROVED]` **Local optimum ≠ terminal optimum.** In a 2-D property-neutral subspace with local objective gradient $(2,1)$ and remaining contraction $\mathrm{diag}(0.01,1)$, the terminal gradient is $(0.02,1)$. At unit budget the locally optimal direction gains $0.4651$ terminally; the terminal-optimal direction gains $1.0002$.

These are counterexamples to a local-to-terminal implication. **They do not show the failure occurs in our checkpoint** — that is the first thing to measure.

**Cost.** For $N$ total steps, $K$ remaining, $r$ paired directions, $J$ trials: suffix overhead $\approx(2r+J)K/N$. At $r{=}2$, $K{=}5$, $N{=}50$, $J{=}1$ that is ~50% extra sampler-step work.

**Status:** `[HYPOTHESIS]`, and see Part 3 — both halves have close prior work.

## BPS — Balanced Preview Sampling **(new)**

**What it does.** Changes *which* partial trajectories get finished. Draw $L=mB$ candidates, run $k$ early steps on all of them, compute structural previews, partition into **B groups of exactly $m$** by preview similarity, draw **one uniformly at random from each group**, shuffle, and finish those $B$ with the unchanged sampler.

**Logic.** A stratified sampling design over partial trajectories: the batch is spread across preview space while each output keeps its own baseline distribution.

**Proof.**

> `[PROVED]` **Every randomly ordered BPS output has the baseline distribution $P_c$, for any previews and any equal-size partition.** Each candidate lies in one group of size $m$, so it is selected with probability $1/m$; the shuffle places it in a given slot with probability $1/B$; hence $\Pr(I_j=i\mid\text{pool, partition})=1/(mB)=1/L$, and $\mathbb E[\psi(X_{I_j})]=\mathbb E_{P_c}[\psi(X)]$.

`[CHECKED]` Exhaustive enumeration, four configurations including two non-power-of-two: $\max_i\lvert\Pr(\text{slot}=i)-1/L\rvert = 0.00$ exactly.

`[CHECKED]` **Both conditions are necessary**, measured on mean *and* variance (targets 0 and 1):

| Scheme | mean | variance |
|---|---|---|
| uniform within equal groups | +0.0029 (1.8 s.e.) | **0.9993** |
| unequal group sizes | **+0.0393** (24 s.e.) | 0.9178 |
| pick best in group | **+0.3148** (197 s.e.) | 0.9806 |
| pick medoid in group | −0.0007 | **0.8361** |

The medoid row matters: it looks unbiased on the mean by symmetry while the distribution is visibly wrong. **That is the rule the closest prior implementation uses.**

**What it cannot do.** `[PROVED]` It cannot raise the expected number of valid or on-target outputs at fixed $B$, and cannot reach a zero-probability mode. Its only opportunity is reducing redundancy.

**The measured risk.** `[CHECKED]` At $k\le4$ the network preview is statistically indistinguishable from grouping on the raw initial noise (difference +0.047, +0.028, −0.011) — in that regime BPS **is** a Gaussian noise coupling, which is prior work. At $k\ge8$ it adds real information (+0.238, +0.578) but the 48% overhead loses at matched compute. **In the toy, no configuration is both novel and compute-positive.**

**Status:** `[HYPOTHESIS]`. Guarantee exact; benefit unmeasured on molecules.

## SMG — Second-Moment Guidance **(primary candidate)**

**18 Sep audit:** TMPD is a shared linear-property special case. The covariance/Jacobian
identity and covariance-aware flow guidance also predate this project. GGDOpt's equation (20)
uses a different quadratic-energy formula, but a different formula does not establish priority.
See [the explicit comparisons and counterexample](SMG_PRIOR_WORK_AUDIT.md#2-what-the-cited-methods-actually-share-with-smg).

**What it does.** Corrects the quantity training-free guidance actually computes.

Guidance needs $\nabla_{x_t}\log p(y^\ast\mid x_t)$, where
$p(y^\ast\mid x_t)=\mathbb E_{x_1\sim p(x_1\mid x_t)}[p(y^\ast\mid x_1)]$. DPS, FreeDoM, MPGD, TFG,
TFG-Flow and the molecular methods built on them all substitute the integrand at the mean:

$$\mathbb E[f(x_1)\mid x_t]\ \approx\ f\big(\hat x_1(x_t)\big)$$

They compute **the property of the average molecule** where they need **the average property**. SMG
computes the leading term of that gap from the model itself and subtracts it.

**Logic.** The gap is the Jensen gap. Its leading term needs the posterior covariance, which the
denoiser already encodes.

**Proofs.**

| Claim | Status |
|---|---|
| **Lemma 1.** $\Sigma_t:=\operatorname{Cov}(x_1\mid x_t)=\frac{\sigma_t^2}{\alpha_t}\frac{\partial\hat x_1}{\partial x_t}$ | `[PROVED]` `[CHECKED]` — second-order Tweedie; standard, independently re-derived here |
| **Lemma 2.** the two moments of the property | `[PROVED]` `[CHECKED]` |
| **Prop 3.** practical corrected guidance | `[DESIGN]` — Gaussian moment closure and omitted derivatives; measure full derivative cost |
| **Cor 4.** the mean correction **vanishes identically for linear operators** | `[PROVED]` |
| **Prop 6.** reparameterization consistency | `[PROVED]` `[CHECKED]` |
| **Prop 7.** leading local mean bias is signed by $\operatorname{tr}(H\Sigma)$ | `[PROVED]` for the Taylor term; final-sampler sign is not a universal guarantee |
| **Prop 8.** no guidance weight can imitate the mean correction | `[PROVED]` `[CHECKED]` |
| Molecular improvement | **nothing.** `[HYPOTHESIS]` |

**Corollary 4 establishes a linear-property reduction.** TMPD's linear-observation formula
needs no nonlinear-property mean correction. This does not describe the entire guidance
literature: covariance-aware flow guidance and quadratic-energy guidance already exist.
The specific nonlinear-property estimator must be distinguished from those constructions.

**Evidence, and it is the strongest of any candidate here.** `[CHECKED]`
`audit/marginalized_guidance_checks.py`, three-mode toy against the model's own exact conditional by
rejection sampling:

| arm | bias | mode TV | mode fractions |
|---|---|---|---|
| **exact conditional (reference)** | — | — | 0.357 / 0.249 / 0.394 |
| unguided | −0.124 | 0.077 | 0.341 / 0.326 / 0.333 |
| **plug-in (standard guidance)** | +0.053 | **0.282** | **0.075** / 0.379 / 0.546 |
| **SMG (marginalized)** | **−0.005** | **0.076** | 0.281 / 0.319 / 0.400 |

Standard guidance drives the first mode from 0.357 to **0.075** — it deletes a mode the model's own
exact conditional keeps. And **no tuned weight recovers it**: sweeping $w\in[0.125,4]$ gives mode TV
from 0.19 to 0.44, never approaching SMG's 0.076. The signed prediction also holds — bias $+0.053$
for a convex property, $-0.086$ for a concave one, flipping with $\operatorname{tr}H$. Decomposed,
the **mean** correction does more work (TV 0.138) than the **variance** correction (0.258).

**Status:** `[HYPOTHESIS]` on molecules; full cost and novelty remain to be established.
The historical toy uses a hard-band reference for a soft Gaussian likelihood. Retain those
numbers as historical observations, then run the matched-target reference and Euler/Heun
controls in M-3/M-4 before interpreting them as conditional-sampling accuracy.

## PCG — Covariance-Preconditioned CFG **(new)**

**What it does.** Multiplies the classifier-free guidance chord by the normalized conditional
denoiser Jacobian:

$$B_c(x_t,t):=\alpha_t\nabla_{x_t}D_c=\frac{\alpha_t^2}{\sigma_t^2}\Sigma_c,\qquad
D^{\rm PCG}_w=D_c+wB_c\,(D_c-D_u)$$

**Logic.** The CFG chord is a noisy condition-evidence gradient; preconditioning it by the posterior
covariance rescales it per-direction rather than by one scalar weight.

**Proof status.** The two supporting identities — the Gaussian-channel information identity and
covariance = denoiser Jacobian — are `[PROVED]` under stated assumptions. **They do not establish**
an optimal guidance window, a transferable information budget, manifold preservation, or improved
generation. A learned Jacobian is a proxy and need not be symmetric or PSD.

**Status:** `[HYPOTHESIS]`, and the weakest-positioned of the five candidates. Its own source
document concludes: *"plausible as a focused course-project contribution; not established as a new
research method."* Posterior covariance, Jacobian-based guidance and information-based scheduling all
have substantial precedent. On the simplex, a covariance step preserves the coordinate sum but can
make probabilities negative.

---

## RCH — Residual Calibration Head **(comparator for SMG, from Haimo v1)**

**What it does.** Learns SMG's gap instead of deriving it. Noise real training molecules, record
$f_A(x_1)-f_A(m)$, fit a ridge head $C=(1-t)\,\phi^\top\beta$ on ~30 E(3)-invariant summary
statistics of the state, and guide with $y-f(m)-C$ in place of $y-f(m)-c$. A six-number
time-only offset $C(t)$ and a *direct* head predicting $f_A(x_1)$ outright come from the same
cached features. Inference-time only; costs exactly plug-in's derivative work (one VJP, no JVP/HVP).

**Logic.** SMG's $c=\tfrac12\operatorname{tr}(H\Sigma)$ is, for the properties Haimo tested,
mostly a function of $t$: a per-time offset removed 80% of the plug-in error on CIFAR variance and
the learned head the rest. If that holds for dipole on our checkpoint, SMG's JVP-priced state
dependence is doing the work of a lookup table. This is the cheapest test that can deflate the SMG
claim, so it runs first.

**Proof status.** None needed; it is a regression. Haimo v1's own control: residual head vs direct
head is null on images (0.0049 vs 0.0052 MAE) and reversed on text (0.0610 vs 0.0534), so the
residual parameterisation is inert.

**Status:** `[TAKEN]`. Plug-in + learned residual vanishing at the clean endpoint is
*Residual ∇-DB* ([Liu et al., ICLR 2025](https://arxiv.org/abs/2412.07775), Eq. 15) and
*VGG-Flow* ([Liu et al., NeurIPS 2025](https://arxiv.org/abs/2512.05116), Eq. 16, same
$\hat x_1=x_t+(1-t)v$); [Reward Score Matching (2026)](https://arxiv.org/abs/2604.17415)
App. G.1 finds those residuals "effectively negligible". Ablation row only. Gates RC-0/RC-1 in
[Feasibility tests](FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md#rch--residual-calibration-head-from-haimo-v1--comparator-for-smg-not-a-claim).

---

# Part 2 — Analysis results

## RHO — the ρ removed-energy identity

`[PROVED]` `[CHECKED]` With $\rho=(b^\top u)^2/(\lVert b\rVert^2\lVert u\rVert^2)$, comparing an unprojected step $\alpha u$ with a tangent step $\alpha u_\perp$ at norm $\le R$:

$$G^\ast=R\lVert u\rVert\sqrt{1-\rho},\qquad \mathcal D=R\lVert b\rVert\sqrt{\rho(1-\rho)}$$

$\mathcal D$ is the **magnitude of avoided first-order property change** at the tangent arm's ceiling. 200,000 instances, max relative error $1.3\times10^{-11}$.

**Four scope limits, each with a counterexample** (`audit/review_v4_checks.py`): avoided change is not avoided *harm* (the unprojected edit can improve the error); $\mathcal D/G^\ast$ is **monotone** in $\rho$ so there is no benefit-per-unit-diversity peak; $\rho$ cannot rank properties because $\mathcal D$ scales with $R\lVert b\rVert$; and $\rho\to1$ does **not** condemn D — that is exactly where tolerance rescues a blocked direction.

**Use:** a diagnostic for the equality arm. Undefined at $b=0$ or $u=0$ — log separately.

## CX-1 … CX-7 — correctness results

Each `[PROVED]`, most `[CHECKED]`. These are conditions for implementing this class of method correctly.

| ID | Result |
|---|---|
| **CX-1** | For $r=-(F-y^\ast)^2/2s^2$, $\nabla r=-\frac{e}{s^2}\nabla F$, so $\nabla r$ **vanishes at the target** while the level-set normal does not. A reward-gradient projector loses its constraint exactly at convergence. Build the tangent space from $\nabla F$ |
| **CX-2** | A regularised projector leaks exactly $\frac{\varepsilon^2}{\lVert b\rVert^2+\varepsilon^2}b^\top u$. Exact neutrality needs $\varepsilon=0$ plus an explicit small-gradient branch |
| **CX-3** | Endpoint orthogonality is **not generally** preserved by the pullback: $\Delta r=a^\top JJ^\top u^\perp$. Preserving *every* orthogonal pair requires $JJ^\top\propto I$; a particular pair can survive a non-similarity map ($J=\mathrm{diag}(2,1)$, $a=(1,0)$, $u=(0,1)$ gives 0). Project in state space and **measure** the residual |
| **CX-4** | State-space interference is $a^\top u=-\frac{e}{s^2}w^\top Md_h$, $M=LL^\top$ (or $LQL^\top$). Conditioning of $M$ is neither necessary nor sufficient for the embedding sign to transfer ($M=\mathrm{diag}(1,1.02)$, $\kappa=1.02$, flips $+0.01\to-0.0098$). **Sufficient condition:** $M=cI+E$ with $\lvert w^\top Ed_h\rvert<c\lvert w^\top d_h\rvert$ |
| **CX-5** | A shared norm cap multiplies property progress by $\min(1,C/\sqrt{A^2+T^2})/\min(1,C/A)$ — an angle cosine only under saturation. Reserve the progress budget first; cap the baseline separately |
| **CX-6** | A tilt multiplier depending only on $f$ preserves every conditional $p(x\mid f=c)$, but **reweights the mixture over levels**: $X\sim\mathcal N(0,1)$, $Z\mid X\sim\mathcal N(0,\epsilon+X^2)$ gives $\operatorname{Var}_w(Z)=\epsilon+\frac1{1+w}$, falling $1.01\to0.11$ under *exact* sampling |
| **CX-7** | A damped Newton retraction gives $e_{\text{new}}=(1-\lambda)e+O(e^2)$; the fourth-order improvement needs $\lambda=1$ (or $1-\lambda=O(e)$). Measured orders: **3.99** at $\lambda=1$, **2.00** at $\lambda=\tfrac12$ |

## YIELD — within-condition useful-yield protocol

`[DESIGN]` Primary number, per condition $c$ with attempt budget $M_c$:

$$U_\delta(c)=\frac{\#\{\text{distinct valid decoded graphs with }\lvert f_B(x)-y_c^\ast\rvert\le\delta\}}{M_c}$$

Reported per fixed attempt count **and** per GPU-hour, within condition then macro-averaged, with attempts (not survivors) as the denominator, independent batches as the statistical units, and $\delta$ frozen and identical across arms.

---

# Part 3 — Novelty assessment

## 3.1 Method description table

| ID | Architecture it attaches to | Key method change | Claimed innovation point |
|---|---|---|---|
| **A** | any flow/diffusion sampler with an additive guidance term | clip guidance to $\rho_A\lVert v_\theta\rVert$ | none — adopted |
| **B** | property-guided sampler with a differentiable guide | project the diversity term onto $\{b^\top d=0\}$ in state space | none as a mechanism; CX-1 and CX-3 as usage conditions |
| **C** | any guided sampler with an endpoint estimate | scale guidance by endpoint-prediction stability | none — engineering |
| **D** | frozen flow-matching EGNN + frozen property guide, per-sample dynamics | replace equality with a per-anchor dead-zone envelope $A=\max(\tau,\lvert e\rvert)$; exact ball/slab solve; finite guard with rollback | the per-anchor **baseline-relative** envelope, the exact local solve, and the finite acceptance check, inside a frozen sampler |
| **TRE** | same, plus paired suffix rollouts of the remaining sampler | estimate **terminal** property and diversity responses by finite differences in a 2–4-dim admissible subspace; solve D's problem in coefficient space | combining terminal-response estimation for **both** property *and* batch-coupled diversity into one tolerance-constrained solve |
| **BPS** | any sampler with sample-wise continuation dynamics; source-agnostic | oversample $L=mB$, preview $k$ steps, partition into equal-size groups by preview, **draw one uniformly per group** | equal-inclusion thinning of a previewed candidate pool, giving **exact** individual-output marginal preservation |
| **SMG** | any diffusion or flow sampler with a differentiable nonlinear property guide; works for both required model families | replace $f(\hat x_1)$ with $f(\hat x_1)+\tfrac12\operatorname{tr}(\nabla^2 f\,\Sigma_t)$, $\Sigma_t=\frac{\sigma_t^2}{\alpha_t}\partial\hat x_1/\partial x_t$; one JVP | the **deterministic second-moment estimator with the curvature term for nonlinear scalar properties**, plus Cor 4 explaining the literature gap |
| **PCG** | conditional + unconditional denoiser (CFG), images then centred 3D coordinates | precondition the CFG chord by $B_c=\alpha_t^2\Sigma_c/\sigma_t^2$ | a per-direction covariance rescaling of the chord in place of a scalar weight |
| **RHO** | analysis of B | — | quantifies avoided first-order property change as $R\lVert b\rVert\sqrt{\rho(1-\rho)}$ |
| **CX-1…7** | analysis of this method class | — | seven implementation-correctness conditions |
| **YIELD** | evaluation | within-condition distinct-valid-on-target per attempt and per GPU-hour | none — good practice |

## 3.2 The four criteria, with citations

**Criteria.** (1) *Exact*: has anyone done exactly this? (2) *Formula*: is the formula changed? (3) *Modality*: is the modality application the same? (4) *Base model*: is the base model the same?

| ID | (1) Exact match? | (2) Formula changed? | (3) Modality differs? | (4) Base model differs? | Verdict |
|---|---|---|---|---|---|
| **A** | **Yes** — [OSCAR, ICML 2026](https://arxiv.org/abs/2510.09060) publishes $g\cdot\min(1,\lVert v\rVert/\lVert g\rVert)$ | no | yes (images → molecules) | yes | **not new** |
| **B** | **Yes, four times** — [O-SVGD, NeurIPS 2022](https://arxiv.org/abs/2210.06447); [Harmless Diversity, ICML 2022](https://proceedings.mlr.press/v162/gong22b/gong22b.pdf); OSCAR; gradient surgery (PCGrad, Perp-Neg, APG) | no | yes | yes | **not new** |
| **C** | **Category occupied** — [limited-interval guidance](https://arxiv.org/abs/2404.07724), VAGS, adaptive CFG. Exact trigger not found | trigger differs | yes | yes | **not claimed** |
| **D** | **No exact match.** Closest: Harmless Diversity (baseline step → diversity correction under analytic constraints); [Constrained Stein Variational Trajectory Optimization](https://arxiv.org/abs/2308.12110) (diverse particles under inequalities); [Trust Sampling, NeurIPS 2024](https://arxiv.org/abs/2411.10932); [Constrained Diffusers](https://arxiv.org/abs/2506.12544) | **yes** — per-anchor $A=\max(\tau,\lvert e\rvert)$ is baseline-relative, not a fixed loss's optimum set when $\lvert e\rvert>\tau$ | **yes** — 3D de novo molecules + simplex sequences; prior work is images/meshes/conformers/trajectories | **yes** — frozen FM EGNN with a frozen property guide | **adaptation** |
| **TRE** | **Both halves occupied.** Property half: [LiDAR, ICML 2026](https://arxiv.org/pdf/2602.03211) rewrites expected future reward with lookahead samples, backprop-free. Diversity half: [Variance-Tilted Diffusion](https://arxiv.org/html/2606.22239v1) estimates its curvature term by **finite differences**; [EDDY](https://arxiv.org/html/2605.06553v1) uses finite-difference/Hutchinson estimation for marginal-preserving drift. Subspace finite differences: [2503.14868](https://arxiv.org/abs/2503.14868) | **yes** — a *constrained* solve on jointly estimated terminal property **and** batch-coupled diversity responses | yes | yes | **thin — investigate, do not claim** |
| **BPS** | **No exact match.** Closest two, both read in full: [Couple to Control, 2026](https://arxiv.org/html/2605.11311v1) — inference-time marginal-preserving noise coupling, **no previews, no clustering, no pool thinning**; [Wildfire §3.2.1, 2026](https://arxiv.org/html/2603.20188v1) — previews a pool, clusters, keeps **k-medians** representatives, makes no marginal claim and reports the resulting distributional failure. Device itself is textbook stratified sampling | **yes** — equal-capacity groups + uniform within-group draw replaces the k-medians rule, which is what makes the marginal exact | **yes** — molecules and simplex sequences; prior work is images and wildfire fields | **yes** | **narrow but verified** |
| **SMG** | **Novelty unresolved.** Shares TMPD's linear-property formula and the earlier flow-guidance covariance identity; differs from the displayed GGDOpt quadratic-energy formula. [Formula audit](SMG_PRIOR_WORK_AUDIT.md) | nonlinear-property mean correction is the candidate; measure total cost | molecular application alone is insufficient; MC competitors already include molecules | covariance-aware guidance already covers diffusion and flow matching | **primary hypothesis; not novelty-cleared** |
| **PCG** | **Substantial precedent** on all three legs: posterior covariance, Jacobian-based guidance, and information-based scheduling. Closest: [Covariance-aware sampling](https://arxiv.org/html/2605.13910), [Divergence is Uncertainty](https://arxiv.org/abs/2605.00941), Free Hunch | the $\alpha_t$ normalization is a change; its own source calls the rest a collection of standard identities | yes | yes | **weakest — not established as a research method by its own assessment** |
| **RHO** | not found as stated. Removed-energy quantities exist in gradient-surgery analysis | new identity, elementary | — | — | **analysis, not an innovation** |
| **CX-1…7** | not found as a collected set; each is elementary | — | — | — | **analysis; external value depends on whether published work falls into these traps — unverified** |
| **YIELD** | **Occupied** — [MolGuidance](https://arxiv.org/html/2512.12198v1) reports DFT on 500 generated molecules at B3LYP/6-31G(2df,p) and diversity beside MAE (though **not** as a frontier) | reporting choice | — | — | **good practice, not novelty** |
| **BDG** | **No verbatim match; every ingredient occupied.** Two-sided ensemble-moment control against a target: [MGD, 2602.17211](https://arxiv.org/abs/2602.17211) (incl. quadratic moments); [Controlling Ensemble Variance, 2501.14822](https://arxiv.org/abs/2501.14822). Closed-loop, state-dependent guidance scale: [Feedback Guidance, 2506.06085](https://arxiv.org/abs/2506.06085), [SOC guidance, 2505.19367](https://arxiv.org/abs/2505.19367), [CFG-Ctrl, 2603.03281](https://arxiv.org/abs/2603.03281). Batch spread: [VTD, 2606.22239](https://arxiv.org/abs/2606.22239) — widen-only, but its interaction term matches BDG's e < 0 dispersion term. Prescribed marginal: [Jeffrey guidance, 2606.13240](https://arxiv.org/abs/2606.13240). In-repo: [Three_New §5.2](Three_New_Guidance_Ideas_Variance_and_Switching.md) already has V_b, τ and the signed gradient | **partly** — by its own reduction it is plug with a feedback-set weight and a shifted target; what changes is that the deviation gain is servoed while the centring gain stays fixed | yes | yes | **incremental; narrow unclaimed composition.** ≈0.85 for the course bar if the reduction and precedents are stated. No higher-bar estimate: the search was ~8 queries, abstracts only, 25 Sep 2026 (not covered by the 13–17 Sep scope below). [Review](BDG_REVIEW.md) |

### Search scope and confidence

**Read in full:** Couple to Control, Wildfire §3.2.1, MolGuidance, OSCAR (v2).
**Abstract or method-summary level only:** Harmless Diversity, Stein Diffusion Guidance, O-SVGD, Trust Sampling, Constrained Diffusers, Constrained Stein Variational Trajectory Optimization.
**Search-summary level only — must be read before any claim rests on them:** **LiDAR (2602.03211)**, **Variance-Tilted Diffusion (2606.22239)**, **EDDY's finite-difference estimator (2605.06553)**.

Searches run 13–17 Sep 2026. "Not found" means this bounded search surfaced nothing; it is not a priority claim.

## 3.3 Bottom line

| | Status |
|---|---|
| **A, B** | prior work. Cite, claim nothing |
| **C** | engineering. Ablation row |
| **D** | adaptation with a changed formula, new modality, new base model. Clears the course bar; not a publishable mechanism |
| **TRE** | **thinnest of the three candidates.** Both halves have close 2026 prior work; the delta is the combination. Needs the §Part 4 mismatch measurement *before* any build |
| **BPS** | **narrow delta, well verified.** Both closest papers read; the device demonstrably matters (medoid variance 0.836); the paper using the alternative rule reports the failure this fixes. Exact guarantee. Measured risk: novel-and-affordable window may be empty |
| **SMG** | **primary under evaluation.** A motivated nonlinear-property mean correction with toy checks; novelty, full cost and molecular benefit remain open. Matched-target references, prior-formula comparators and Euler/Heun controls are required |
| **PCG** | **drop.** Its own source document declines to claim it as a research method, and all three legs have precedent |
| **RCH** | **run, never claim.** Published twice (ICLR 2025, NeurIPS 2025) and reported negligible (RSM 2026); its value here is as the strongest cheap control on SMG — "could a schedule or a 1-VJP fit have done this?" |
| **RHO, CX-1…7, YIELD** | supporting analysis and protocol. Not innovations |

**Against the assignment.** The brief excludes only *"ordinary hyperparameter tuning, more epochs, more compute, a different random seed, or simply replacing a standard backbone with a larger standard backbone."* **D, TRE and BPS are each none of those.** Any one of them satisfies §3.5, provided the paper states the delta precisely — which Part 3.2 does — and ablates it honestly.

**Recommendation, revised now that five candidates exist.** Carry **one**.

| Rank | Candidate | Why |
|---|---|---|
| **1** | **SMG** | selected primary hypothesis: a testable property-estimation correction applicable to both specified Gaussian-path model families; numerical-solver effects, runtime and matched-target accuracy remain to be tested |
| 2 | **BPS** | exact guarantee, cleanest atomic ablations, source-agnostic Modality-2 transfer — but a measured squeeze between novelty and cost |
| 3 | **D** | adaptation with local theorems; the natural diversity comparator |
| — | **TRE** | defer: most expensive, both halves occupied, gated on an unmeasured failure |
| — | **PCG** | drop |

**SMG and BPS are complementary, not competing** — SMG corrects *where each trajectory goes*, BPS chooses *which trajectories to finish*. If time allows one candidate plus one comparator, SMG primary with D as the diversity comparator is the lowest-risk pair.
