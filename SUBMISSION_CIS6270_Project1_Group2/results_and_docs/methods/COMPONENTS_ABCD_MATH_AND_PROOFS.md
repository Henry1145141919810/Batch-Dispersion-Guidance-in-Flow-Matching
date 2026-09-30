# Innovation — the method and its mathematics

**v6.0 · 16 Sep 2026.** States the current formulation and what is established about it. Nothing here is a history of earlier drafts; that record is in `PROJECT_CLAIMS_VERIFICATION.md`, `COMPONENTS_ABCD_V4_REVIEW.md`, and the scripts in `audit/`.

| | |
|---|---|
| Test plan | `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md` |
| Plan, compute, roles, slides | `PROJECT_GUIDE.md` |
| Teammate summary | `TEAM_BRIEF.md` |

**Naming.** Components are lettered **A–D**; sections of this document are numbered **§1–§6**.

### Status labels

| Label | Meaning |
|---|---|
| `[PROVED]` | derivation given; holds under its stated conditions |
| `[CHECKED]` | confirmed numerically; the script is named |
| `[ASSUMPTION]` | required for the method to work; not established |
| `[HYPOTHESIS]` | what an experiment decides |
| `[DESIGN]` | a choice; no claim of optimality or novelty |
| `[TAKEN]` | prior work; cite, claim nothing |

### Notation

| Symbol | Meaning |
|---|---|
| $x$ | sampler state at time $t$; the only variable we can move |
| $E_t(x)=x+(1-t)v_\theta(x,t)$ | endpoint proxy; a conditional-mean estimate, not the ODE endpoint `[ASSUMPTION]` |
| $F_t(x)=f_A(E_t(x))$ | the guide's predicted property |
| $b=Q\nabla_x F_t$ | property gradient, modality-projected; the level-set normal |
| $e=F_t-y^\ast$ | signed property error |
| $u=Q\nabla_x D$ | spread gradient, modality-projected; batch-coupled |
| $Q$ | symmetric orthogonal projector onto admissible state directions |
| $\tau$ | allowed guide error; the primary scientific axis |
| $\delta$ | the independent evaluator's acceptance tolerance, fixed across arms |
| $\rho$ | fraction of $u$'s energy along $b$ (§3.8) |
| $L=\partial h/\partial x$, $J=\partial E_t/\partial x$ | guide-embedding and flow Jacobians |

---

# §1 The idea

## 1.1 The problem

Property-targeted generation should return a **portfolio**: many structurally different molecules that all hit dipole 2.5 D. Guided samplers return near-copies.

Two facts bound how this can be framed.

`[PROVED]` A tilt multiplier depending only on $f(x)$ leaves every supported conditional $p(x\mid f(x)=c)$ unchanged at any strength (§3.9.4). Guidance does not destroy within-level variety in the ideal limit.

`[PROVED]` `[CHECKED]` The same tilt reweights the mixture **over** levels, and structural diversity can depend on the level. With $X\sim\mathcal N(0,1)$, $Z\mid X{=}x\sim\mathcal N(0,\epsilon+x^2)$, $f=X$, and tilt $e^{-wX^2/2}$: every conditional is untouched while $\operatorname{Var}_w(Z)=\epsilon+\frac1{1+w}$, falling from $1.01$ to $0.11$ as $w:0\to9$ under exact sampling.

So the claim available to us is narrow:

> `[HYPOTHESIS]` Approximate guidance may lose within-condition diversity beyond what conditioning implies. The experiment tests whether constrained diversification recovers useful coverage.

Any excess loss must be established empirically against a conditional reference. Repulsion is not automatically a repair of sampling fidelity, and can itself bias the distribution.

Two further failure modes of strong guidance shape the design: off-manifold drift, and schedule mismatch — early in sampling the endpoint prediction is a blur, so guiding hard on it guides on a hallucination.

## 1.2 The geometry

$\{x: f(x)=y^\ast\}$ is a surface. Move **across** level sets to reach the target; move **along** the surface to diversify. A tangent move changes the property by zero to first order (§3.3).

## 1.3 The tolerance

The application accepts a band, not a value. Strict preservation discards that: at a prediction of 5.00 with [4.9, 5.1] acceptable, an equality-tangent edit blocks a move to 5.04 that may carry real structural change. Component D spends the band instead.

## 1.4 One primary axis, three reference arms

$\tau$ is the primary scientific axis. It does not interpolate between two prior-work endpoints, so each comparator is constructed explicitly.

| Arm | Constraint | What it is |
|---|---|---|
| **Equality** | $\ell=h=0$ at every anchor | strict level-set projection; the prior-work comparator |
| **$\tau=0$ band** | $-2\lvert e\rvert\le b^\top d\le0$ for $e>0$ | no worsening of anchor error; inward movement allowed |
| **Unconstrained** | guard only, no slab | plain repulsion |
| **$\tau>0$ band** | band active | the candidate |

> **Band activity is a prerequisite for interpreting the sweep.** `[PROVED]` `[CHECKED]` Since $A=\max(\tau,\lvert e\rvert)$, whenever $\lvert e\rvert$ exceeds every $\tau$ in the sweep the constraint is identical for all of them. Log the fraction of anchors with $\lvert e\rvert<\tau$ and the pairwise differences between proposed edits; a null sweep with zero activity says nothing about the method.

## 1.5 What is established

| Item | Status |
|---|---|
| The $\rho$ identity (§3.8) | `[PROVED]` `[CHECKED]` as algebra. It bounds the magnitude of avoided first-order property change, not benefit |
| Component D's feasibility, exact solve, local ascent, local dominance over equality (§3.5–3.6) | `[PROVED]` `[CHECKED]`, each within its stated scope |
| Correctness conditions for this class of method (§3.9) | `[PROVED]`; external value depends on §6 items 1–2 |
| Improvement in decoded diversity, accuracy, validity, or yield | **Nothing.** `[HYPOTHESIS]` for F4 |

> The strongest valid theorem is local: the band solver attains a local objective at least as good as equality under the same budget, and accepted edits pass the stated checks at that anchor. Nothing here establishes better molecules.

## 1.6 What we claim

> We study whether an application-defined property tolerance can be spent to improve within-condition structural diversity in frozen continuous molecular generators. We formulate a baseline-relative constrained diversity edit with an exact local solve and a finite acceptance check, and test it against separately constructed equality, $\tau=0$-envelope, unconstrained, dead-zone-relaxation, and generate-and-select controls at matched attempts and matched wall-clock.

An adaptation with evidence, not a new sampling principle. §5 states what is taken.

## 1.7 Why every outcome is reportable

| Result | What we write |
|---|---|
| $\tau>0$ beats equality and the dead-zone control | a method contribution |
| $\tau>0$ ties equality | tolerance exploitation does not help here, with band-activity diagnostics to say why |
| the dead-zone control wins | simple relaxation is sufficient — a useful negative result |
| generate-and-select wins at equal wall-clock | an efficiency result |

## 1.8 The evaluation protocol `[DESIGN]`

For condition $c$ with attempt budget $M_c$:

$$U_\delta(c)=\frac{\#\{\text{distinct valid decoded graphs with } \lvert f_B(x)-y_c^\ast\rvert\le\delta\}}{M_c}$$

Distinct valid molecules meeting the evaluator tolerance per fixed attempt count, and per GPU-hour. $\delta$ is frozen in advance and identical across arms; it is not $\tau$.

Companions: Vendi Score on a fixed fingerprint kernel within each condition at matched accepted counts; conditional coverage against real data; MAE under a held-out $f_B$ that never touches generation; validity and 3D geometry.

Two rules:

- **Diversity is measured within one condition, then macro-averaged.** A global score across thousands of targets can be high while every condition has collapsed.
- **The denominator is attempts, not survivors.** Otherwise a method with better validity is scored on a self-selected population.

**Quantum chemistry.** DFT at QM9's own level of theory on a stratified subsample, CPU only. Good practice, and occupied — MolGuidance reports DFT on 500 generated molecules at B3LYP/6-31G(2df,p). Claim nothing for it.

**Experimental units are independent batches and runs**, because repulsion couples particles within a batch. Paired cluster bootstrap over matched seed/batch IDs and condition strata.

---

# §2 The method — components A to D

## 2.1 The four components

| | Component | Role | Status | Prior art |
|---|---|---|---|---|
| **A** | velocity-relative trust region | keep the baseline update from overwhelming the model | `[TAKEN]` | OSCAR publishes this form |
| **B** | level-set tangent projection | spread without changing the property | `[PROVED]` correct (§3.1); `[TAKEN]` as a mechanism | O-SVGD; Harmless Diversity; OSCAR; gradient surgery. Ours are the two correctness conditions in §3.4 and §3.9.1, not the projection |
| **C** | confidence gate | damp guidance while the endpoint prediction is unstable | `[DESIGN]` | time-varying guidance is a crowded category; not claimed |
| **D** | tolerance band | spend the allowed error to buy diversity | `[PROVED]` locally (§3.5–3.6); outcome `[HYPOTHESIS]` | inequality-constrained diverse particles exist (§5.1); the delta is §5.2 |

**D is not B with a wider allowance.** At $\tau=0$ the band is a progress envelope, not an equality constraint (§2.5). Build the equality arm separately.

## 2.2 Component A — velocity-relative trust region `[TAKEN]`

$$g \leftarrow g\cdot\min\!\left(1,\ \frac{\rho_A\lVert v\rVert}{\lVert g\rVert}\right)\quad\text{if }\lVert g\rVert>0;\qquad g\leftarrow g\ \text{ otherwise}$$

The zero branch is explicit because $w_g=0$ or $e=0$ gives $\lVert g\rVert=0$.

**Cap the baseline only.** A cap applied jointly to reward and diversity reduces the reward component when it binds (§3.9.5). The diversity edit is bounded separately by its own radius $R$, which is also what controls its curvature error.

## 2.3 Component B — level-set tangent projection

$$d = u-\frac{b^\top u}{b^\top b}\,b \qquad\Longrightarrow\qquad b^\top d=0 \quad (b\neq0)$$

Four conditions on implementing it:

| Condition | Reason | Result |
|---|---|---|
| Build the tangent space from $\nabla F_t$, not $\nabla r$ | $\nabla r$ vanishes at the target while the level-set normal does not | §3.9.1 |
| Project in state space, after the pullback | endpoint orthogonality is not generally preserved by $J^\top$ | §3.4 |
| Use the exact projector with a logged small-$\lVert b\rVert$ branch | a regularised projector leaks a computable amount | §3.9.2 |
| Project both gradients into the modality subspace first | reversing the order destroys neutrality | §3.7 |

## 2.4 Component C — confidence gate `[DESIGN]`

$$\Delta_i(t)=\frac{\lVert E_t(x_i)-E_{t_{\text{prev}}}(x_i)\rVert}{\lVert E_t(x_i)\rVert+\varepsilon},\qquad c_i(t)=\frac{1}{1+\Delta_i(t)/\tau_C}$$

At the first step set $c=1$; if the unguided median $\Delta$ is zero, fall back to $c\equiv1$. Record both choices. $\tau_C$ comes from the median $\Delta$ of one unguided run.

Endpoint stability is not calibrated accuracy — the proxy can be stably wrong, and a median from an unguided run is a heuristic. C carries an ablation row and no claim; control C2 in `PROJECT_GUIDE.md` is built to beat it.

## 2.5 Component D — the tolerance band

With $\tau\ge0$, at each anchor:

$$A=\max(\tau,\lvert e\rvert),\qquad \ell=-A-e,\qquad h=A-e \qquad\Longrightarrow\qquad \ell\le b^\top d\le h$$

$\ell\le0\le h$ always, so the zero edit is always feasible and the local problem is never infeasible (§3.5).

### What each regime gives

| Regime | Interval | Meaning |
|---|---|---|
| $e=0,\ \tau=0$ | $[0,0]$ | equality — only here |
| $e>0,\ \tau=0$ | $[-2e,\,0]$ | no worsening; inward movement allowed |
| $e<0,\ \tau=0$ | $[0,\,-2e]$ | mirror image |
| $\lvert e\rvert\le\tau$ | $[-\tau-e,\ \tau-e]$ | stay inside the band |
| $\lvert e\rvert>\tau$ | $A=\lvert e\rvert$ | $\tau$ inert; identical for every such $\tau$ |

`[CHECKED]` With $e=0.2$, $b=(1,0)$, $u=(-1,1)$, $\eta=1$, $R=0.1$: the $\tau=0$ band returns $(-0.0707,0.0707)$ with property change $-0.0707$; equality returns $(0,0.1)$ with property change $0$.

### The local problem

$$d^\star=\arg\max_d\Big[u^\top d-\tfrac{1}{2\eta}\lVert d\rVert^2\Big]\quad\text{s.t.}\quad Qd=d,\ \ \ell\le b^\top d\le h,\ \ \lVert d\rVert\le R$$

A nearest-point projection onto {ball ∩ slab ∩ subspace}; exact closed form in §3.6.

### The diversity objective `[DESIGN]`

$$D(Z)=\log\det\big(K(Z)+\lambda_K I\big),\qquad K_{ij}=\exp\!\left[-\frac{\lVert h(E(z_i))-h(E(z_j))\rVert^2}{2\sigma^2}\right]$$

Cholesky log-determinant, never a raw determinant. Cost $O(B^2k)$ for distances and $O(B^3)$ for the factorisation.

**Cache $\sigma$ at the anchor and reuse it for every acceptance trial**, or the guard compares different objectives. Give $\sigma$ a positive floor for coincident features; jitter protects the factorisation but not the bandwidth.

### Scale `[DESIGN]`

State displacements, not velocity corrections: $\eta=\lambda_D\Delta t$ and $R_i=\rho_D\Delta t\sqrt{m_i}$ with $m_i$ the count of free scaled coordinates, so more steps does not mean more total editing distance. The norm ball is not evidence the edit stays on the data manifold.

### The finite acceptance guard `[DESIGN]`

Try $\alpha\in\{1,\tfrac12,\tfrac14\}$, re-evaluating the actual endpoint map and guide at $Z+\alpha d$ with $\sigma$ held at its anchor value. Accept only if for every sample

$$\lvert F_{t'}(z_i+\alpha d_i)-y^\ast\rvert\le A_i+\epsilon_{\text{num}}$$

and $D(Z+\alpha d)\ge D(Z)+c\,\alpha\sum_i u_i^\top d_i-\epsilon_{\text{num}}$ with $c=10^{-4}$. Otherwise return $Z$ unchanged. Rejected edits reproduce the baseline anchor exactly, keeping arms comparable.

The guard does not guarantee chemical validity `[ASSUMPTION]`; decoding is discontinuous.

**Rejecting every trial can be correct.** `[PROVED]` For $F(x,y)=x^2+y^2$ at anchor $(1,0)$, target $0$, tolerance $1$: the tangent direction $(0,1)$ satisfies the linearised band, but every positive step gives $F=1+\alpha^2>1$. A high rejection rate is a finding; near-total rejection means the controller is inert.

## 2.6 The full algorithm

```text
for each sampler step, state x at time t, t2 = t + dt:

    # ---- baseline step, identical in every arm
    v = v_theta(x, t);  E = x + (1-t)*v;  F = f_A(E)
    b = Q @ grad(F, x)                        # property gradient      (2.3)
    a = -(F - y_star)/s**2 * b
    g = w_g * c * a                           # c = gate               (C)
    if norm(g) > 0:
        g = g * min(1, rho_A*norm(v)/norm(g)) # cap the baseline only  (A)
    z = x + dt*(v + g)

    x = z                                     # commit, every step
    if not a scheduled diversity step: continue

    # ---- diversity edit, anchored at z, fresh graph
    z = z.detach().requires_grad_(True)
    E2 = z + (1-t2)*v_theta(z, t2);  F2 = f_A(E2);  e = F2 - y_star
    b2 = Q @ grad(F2, z)                      # VJP #1
    sigma = median_pairwise(h(E2)).detach()    # cached at the anchor   (2.5)
    u  = Q @ grad(D(h(E2); sigma), z)          # VJP #2, batch-coupled
    A_ = max(tau, abs(e))                      # band                   (D)
    d  = project_ball_slab(eta*u, b2, -A_-e, A_-e, R)    # exact        (3.6)
    for alpha in (1, 1/2, 1/4):                # guard, sigma fixed     (2.5)
        if accept(z + alpha*d, sigma): x = z + alpha*d; break

decode terminal states; evaluate with independent f_B
```

The commit precedes the scheduling branch, so unedited steps still advance the state.

**Schedule `[DESIGN]`:** one edit every five steps over the final half of generation. Untouched steps use exactly the baseline; every arm shares the time convention, schedule, features and conditions.

## 2.7 Hyperparameters

| Symbol | Component | Meaning | Default | Sweep |
|---|---|---|---|---|
| $\tau$ | D | allowed guide error; primary axis | from the application | 0, ¼δ, ½δ, δ |
| $w_g$ | baseline | guidance strength | — | 0, 0.5, 1, 2, 4, 8 |
| $\lambda_D$ | D | diversity step scale | 1.0 | 0.5, 1, 2 |
| $\rho_D$ | D | edit radius scale | 0.5 | 0.25, 0.5, 1 |
| $\sigma$ | D | kernel bandwidth | median pairwise, cached at anchor | — |
| $\rho_A$ | A | max baseline guidance-to-velocity ratio | 0.5 | 0.25, 0.5, 1 |
| $\tau_C$ | C | gate half-point | median $\Delta$, unguided run | 0.5×, 1×, 2× |

$\tau$ is the primary axis, not the only tunable parameter. Report the full tuning budget per arm.

## 2.8 Assumptions, each with its gate

| # | Assumption | Checked by |
|---|---|---|
| 1 | Time runs noise→data as 0→1; endpoint map matches the checkpoint; $t'=t+\Delta t$ | F0 |
| 2 | $E_t$ is informative about the decoded molecule | F3, proxy-to-final error by time |
| 3 | Generator frozen but differentiable w.r.t. its input | F1 |
| 4 | $F_t(x_i)$ depends only on sample $i$ — eval mode, no dropout, no cross-sample layers | F1 |
| 5 | $Q$ symmetric idempotent in consistently scaled coordinates; padded atoms masked | F1 |
| 6 | Gradients flow through both $h$ and $E_t$ into the state | F1 |
| 7 | A batch shares one $(y^\ast,N)$ condition | protocol |
| 8 | The batch prediction mean sits near the target, so interference is systematically harmful | F2: measure $e\,b^\top u$ |
| 9 | $\tau$ is active at some anchors | F2/F3: band-activity fraction (§1.4) |

## 2.9 Cost

Per edited step, against one forward and one backward for a simple guided baseline:

| Stage | Cost |
|---|---|
| Baseline | 1 forward, 1 backward |
| Fresh anchor at $t'$ | 1 forward, 2 backward ($F$ and $D$, shared graph) |
| Acceptance | $q\in\{1,2,3\}$ forward-only evaluations |

Over $S$ steps with edited fraction $p$: about $S[1+p(1+q)]$ forwards and $S(1+2p)$ backwards. At $p\approx0.1$, roughly 1.2–1.4× forwards and 1.2× backwards.

Operation counts, not wall-clock multipliers. Training-free does not mean cheap inference; disabling parameter gradients does not remove the activation memory needed for input gradients.

## 2.10 DNA — a separate formulation `[DESIGN]`

For linear logits $\ell=Wh+c$, $\nabla_h\log p(c\mid h)=w_c-\sum_j p(j\mid h)w_j$. What fails for DNA is the planar single-class-weight simplification, not the existence of scalar Newton methods: a smooth scalar log-probability admits a Newton correction under regularity.

Use a lower bound on $s(x)=\log p_A(c\mid E_t(x))$ rather than a symmetric band. With anchor score $s_0$ and floor $s_{\min}$:

$$\nabla s(z)^\top d\ \ge\ \min(s_{\min},s_0)-s_0$$

Same zero-always-feasible structure as §3.5. Sum-zero directions still need positivity and domain checks before acceptance. Check the actual finite log-probability at acceptance, using autograd through the full log-softmax.

## 2.11 The objections the experiment must answer

1. **It may merely relax accuracy.** Dead-zone control $L_\tau(F)=\max(0,\lvert e\rvert-\tau)^2/2s^2$, no repulsion. The most likely way this fails, and note §5.2: the band *is* this loss's optimum set.
2. **It may only spread property values** inside the band. Independent graph and fingerprint diversity; finer evaluator bins.
3. **The guide can be wrong.** $F_A(x,y)=x$ with $F_B(x,y)=x+10y$ permits large evaluator error along the guide's exact tangent. Independent evaluation; never feed $f_B$ back.
4. **The endpoint proxy can be wrong.** Late edits first; measure proxy-to-final error by time.
5. **The next baseline step can undo the edit.** Track retained change across steps.
6. **The boundary can stall, or $\tau$ may never activate.** Accepted-edit rate, rejection reasons, band activity. Exact duplicates have zero repulsive gradient under a smooth symmetric kernel.
7. **Validity can fall.** Decoded validity and 3D geometry, not the continuous state.
8. **It may lose on speed.** Generate-and-select at equal wall-clock is mandatory.
9. **It does not preserve a known sampling law.** Claim no exact conditional sampling, marginal preservation, or unbiased posterior.
10. **One checkpoint is not robustness.** Paired seeds and independent batches; a second checkpoint before any large grid.

---

# §3 Results

## 3.1 The projection is the exact nearest neutral direction `[PROVED]`

For $b\neq0$, $d=u-\frac{b^\top u}{b^\top b}b$ gives $b^\top d=0$ and $u^\top d=\lVert d\rVert^2\ge0$: the nearest vector to $u$ with first-order property neutrality. Non-unique when $d=0$.

## 3.2 A local constrained-program form `[PROVED]`

For $b\neq0$,

$$\max_m\ \langle m,u\rangle \quad\text{s.t.}\quad \langle m,b\rangle=c,\quad \lVert m\rVert\le\eta$$

is feasible iff $\lvert c\rvert\le\eta\lVert b\rVert$, with optimum

$$m^\star=\frac{c}{\lVert b\rVert^2}b+\sqrt{\eta^2-\frac{c^2}{\lVert b\rVert^2}}\;\frac{d}{\lVert d\rVert}$$

and $d=0$ handled separately, where every feasible point has the same tangent contribution.

**Scope.** The constraint $b^\top m=c$ fixes signed **property** change, not reward progress, unless the reward-gradient scale is folded in. This local program is not the split sampler we run, so its optimality does not make the sampler optimal. The implementation consequence that does carry: weights followed by a shared cap is not this optimiser — reserve the property-progress budget first.

## 3.3 Order of the property change `[PROVED]`

$$F_t(z+d)=F_t(z)+b^\top d+O(\lVert d\rVert^2)$$

A tangent direction has no linear property term; a non-tangent direction has one iff its directional derivative is nonzero.

For the reward $r=-e^2/2s^2$, use the expansion rather than an order claim:

$$\Delta r=-\frac{e}{s^2}\Delta F-\frac{(\Delta F)^2}{2s^2}$$

At $e=0$ the linear reward term vanishes for every direction: unprojected reward change is $O(h^2)$ and tangent reward change $O(h^4)$ — `[CHECKED]` $-5.0\times10^{-3}$ versus $-5.0\times10^{-5}$ at $h=0.1$. Away from the target the linear term exists, and its sign may indicate improvement rather than cost. State which quantity a finite-difference experiment measures.

A globally valid Hessian bound is unavailable, so the remainder is controlled by the edit radius $R$ and ultimately by the finite guard.

## 3.4 The pullback must happen in state space `[PROVED]` `[CHECKED]`

With $J=\partial E_t/\partial x$,

$$\langle J^\top a_1,\ J^\top u_1^\perp\rangle=a_1^\top JJ^\top u_1^\perp$$

which is not generally zero. Preserving **every** orthogonal pair requires the scalar-metric condition $JJ^\top\propto I$. A particular pair can survive a non-similarity map: with $J=\mathrm{diag}(2,1)$, $a=(1,0)$, $u=(0,1)$ the pulled-back inner product is $0$ `[CHECKED]`.

So project in state space because neutrality is not otherwise guaranteed, and **measure** the residual rather than assuming it is large. `[CHECKED]` naive state dot $3.0$ versus corrected $0$ in the reference example. Writing $J=I+(1-t)\,\partial v/\partial x$ does not by itself establish anisotropy; at $t=1$ it equals $I$. The displayed inner product is a directional derivative, not the full finite $\Delta r$.

## 3.5 The band is always feasible `[PROVED]`

$\ell\le0\le h$ always, so $d=0$ is feasible and the local problem never becomes infeasible. Regimes in §2.5.

## 3.6 The local solve is exact `[PROVED]` `[CHECKED]`

Conditions: $b\neq0$, $\eta>0$, $R\ge0$, zero-feasible bounds, gradients already projected into the subspace.

With $r=\eta u$, $n=b/\lVert b\rVert$, and $w$ the projection of $r$ onto the ball within the subspace: return $w$ if it satisfies the slab, else

$$s=\mathrm{clip}\!\left(n^\top w,\ \tfrac{\ell}{\lVert b\rVert},\ \tfrac{h}{\lVert b\rVert}\right),\qquad
d=s\,n+\min\!\left(1,\tfrac{\sqrt{\max(0,R^2-s^2)}}{\lVert v_r\rVert}\right)v_r,\qquad v_r=r-(n^\top r)n$$

### Proof

1. $w$ is the unique ball projection of $r$ in the subspace. If $w$ lies in the slab it solves the intersection problem.
2. Suppose $n^\top w>U$. An intersection optimum $d$ with $n^\top d<U$ is impossible: by convexity and uniqueness of the ball optimum, a sufficiently small move from $d$ toward $w$ remains in both ball and slab and strictly reduces $\lVert\cdot-r\rVert^2$.
3. The upper face is therefore active. With the normal coordinate fixed at $U$, project the tangent component of $r$ onto the remaining tangent ball of radius $\sqrt{R^2-U^2}$. The lower face is symmetric. $\square$

This covers the zero and lower-face cases and the inactive-tangent-ball case, and requires no division by $U$.

`[CHECKED]` `audit/test_tolerance_controller.py`: 1,000 random 12-dimensional problems, 100,000 feasible comparisons, 120 dense reduced-coordinate references; max feasibility residual $\approx4.6\times10^{-15}$. `audit/ball_slab_bruteforce_check.py`: independent pure-Python brute force, 4,000 problems in 2–8 dimensions against a dense scan and random feasible search; 0 infeasible, worst excess $1.8\times10^{-12}$.

The small-normal branch returns $d=0$. That is a conservative exception, not the optimiser of the non-degenerate program; log how often it fires.

**Local ascent.** `[PROVED]` From the projection inequality with $v=0$ and $r=\eta u$: $u^\top d\ge\lVert d\rVert^2/\eta\ge0$.

**Dominance over equality.** `[PROVED]` The equality set is a subset of the band set, so the optimised **local regularised objective** is at least as large. The sets need not be strictly different — at $e=\tau=0$ they coincide, and inactive constraints give other qualifications. This says nothing about final diversity, MAE, yield or wall-clock, and the comparison is fair only at the same $\eta$ and $R$.

## 3.7 Constraint composition order `[PROVED]` `[CHECKED]`

Project both gradients into the subspace first: $b=Q\nabla F_t$, $z=Q\nabla D$, then $d=z-\frac{b^\top z}{b^\top b}b$ satisfies $Qd=d$ and $\nabla F_t^\top d=0$. The joint projector is $Q-bb^\top/(b^\top b)$. Reversing the order leaves residual $-0.333$ in the reference example.

The simplex requires the joint treatment, and there sum-zero is necessary but not sufficient: boundary feasibility needs inward-or-tangent directions at zero coordinates `[ASSUMPTION]`.

## 3.8 The $\rho$ identity `[PROVED]` `[CHECKED]`

With $n=b/\lVert b\rVert$, $u=u_\parallel+u_\perp$, $u_\parallel=(n^\top u)n$:

$$\rho=\frac{\lVert u_\parallel\rVert^2}{\lVert u\rVert^2}=\frac{(b^\top u)^2}{\lVert b\rVert^2\lVert u\rVert^2}\in[0,1]$$

Comparing an unprojected step $\alpha u$ with a tangent step $\alpha u_\perp$, each of norm at most $R$:

$$G^\ast=R\lVert u\rVert\sqrt{1-\rho},\qquad \mathcal D=R\lVert b\rVert\sqrt{\rho(1-\rho)}$$

$G^\ast$ is the tangent arm's maximum first-order diversity gain; $\mathcal D$ is the **magnitude of avoided first-order property change** at that gain. `[CHECKED]` `audit/rho_proposition_check.py`, 200,000 instances in 2–20 dimensions, max relative error $1.3\times10^{-11}$.

### Scope — four limits, each with a counterexample

All in `audit/review_v4_checks.py`.

**1. Avoided change is not avoided harm.** `[CHECKED]` With $F(x,y)=x$, target $0$, $e=0.2$, $u=(-1,1)$, matched gain $0.1$: the unprojected edit moves the error $0.2\to0.15$ — an improvement — while the tangent edit holds it at $0.2$. Both have $\rho=\tfrac12$.

> Measure the signed $e\,b^\top u$, or the signed change in squared error. Interference is harmful only when $e$ and $b^\top u$ share a sign. For centroid repulsion that requires the batch prediction mean to sit near the target — an empirical condition, not a consequence of sharing a target (§3.9.3). Assumption 8 in §2.8.

**2. No benefit-per-unit-diversity peak.** $\mathcal D/G^\ast=\lVert b\rVert\sqrt\rho/\lVert u\rVert$ is monotone in $\rho$ at fixed norms. The peak at $\rho=\tfrac12$ exists only in absolute $\mathcal D$, and only because $G^\ast$ shrinks with $\rho$ — a choice of comparison point, not a theorem about all operating points.

**3. $\rho$ alone cannot rank properties.** $\mathcal D$ scales with $R\lVert b\rVert$ and depends on units, anchor error, curvature and guide calibration. `[CHECKED]` $R=1$, $\lVert b\rVert=100$, $\rho=0.99$ gives $9.95$, above $0.5$ for $R=1$, $\lVert b\rVert=1$, $\rho=0.5$. Any cross-property ordering is `[HYPOTHESIS]`, needs normalisation, and still says nothing about final yield.

**4. $\rho\to1$ is not a failure condition for component D.** `[CHECKED]` At $e=0$, $\tau=0.2$, $b=u=(1,0)$, $R=0.1$: equality gives $d=0$ and gain $0$, while the band permits $d=(0.1,0)$ with gain $0.1$. This is the case where available tolerance rescues a direction equality blocks. Do not stop the band experiment on $\rho$ alone.

### Use

Keep $\rho$ as a diagnostic for the equality arm. At $b=0$ or $u=0$ it is undefined — log a separate category, not zero. For component D, measure slack, active constraints, band activity, accepted nonzero edits, structural change, and independent outcomes directly.

## 3.9 Further results

**3.9.1 Which gradient defines the tangent space.** `[PROVED]` `[CHECKED]` For $r=-(F-y^\ast)^2/2s^2$, $\nabla r=-\frac{e}{s^2}\nabla F$. Away from the target both give the same projector; at the target $\nabla r=0$ while $\nabla F\neq0$, so a reward-gradient projector loses its constraint exactly at convergence. Build from $\nabla F$ and recover the reward direction by scalar multiplication — no extra backward pass.

**3.9.2 Regularised projectors leak a computable amount.** `[PROVED]` `[CHECKED]`

$$b^\top\!\left(u-\frac{b^\top u}{\lVert b\rVert^2+\varepsilon^2}b\right)=\frac{\varepsilon^2}{\lVert b\rVert^2+\varepsilon^2}\,b^\top u$$

So $\varepsilon$ can be set against a stated error budget. Exact neutrality requires $\varepsilon=0$ plus an explicit small-gradient branch.

**3.9.3 When embedding-space sign intuition transfers.** `[PROVED]` The state-space interference is $a^\top u=-\frac{e}{s^2}w^\top M d_h$ with $M=LL^\top$, or $LQL^\top$ if $L$ has not already been restricted to the allowed subspace.

Conditioning of $M$ is neither necessary nor sufficient for the sign to transfer. `[CHECKED]` $M=\mathrm{diag}(1,1.02)$, $w=(1,1)$, $d_h=(1,-0.99)$: $\kappa(M)=1.02$ while $w^\top d_h=+0.01$ and $w^\top M d_h=-0.0098$.

**Sufficient condition.** Write $M=cI+E$ with $c>0$. The sign is preserved if

$$\lvert w^\top E\,d_h\rvert < c\,\lvert w^\top d_h\rvert$$

with a conservative version replacing the left side by $\lVert E\rVert_2\lVert w\rVert\lVert d_h\rVert$. An angular margin is essential: nearly orthogonal vectors flip under small anisotropy.

The centroid expression $-\frac{2}{s^2}(p_i-y_i^\ast)(p_i-\bar p)$ is an embedding-space calculation, not the exact state-space interference. Sharing a target does not imply $\bar p=y^\ast$. Within-condition evaluation is justified on its own grounds (§1.8), not by this argument.

**3.9.4 The tilt multiplier preserves conditionals.** `[PROVED]` `[CHECKED]` For a positive multiplier depending only on $f$, $p_w(x\mid f(x)=c)=p(x\mid f(x)=c)$ at every supported $c$ and every $w$. Verified with independent Gaussians: tangential variance exactly $1.0$ out to $w=1000$.

Three scope limits: it is the multiplier, not the whole tilted density, that is a function of $f$ alone; this holds for the fixed-variance squared-error likelihood, not arbitrary heteroscedastic ones; and it does not imply the ideal target retains the structural diversity we want — §1.1 gives a case where exact sampling reduces structural variance from $1.01$ to $0.11$.

**3.9.5 What a shared norm cap costs.** `[PROVED]` `[CHECKED]` For normal magnitude $A$, orthogonal magnitude $T$ and cap $C$, the progress ratio relative to a separately capped baseline is

$$\frac{\min(1,\,C/\sqrt{A^2+T^2})}{\min(1,\,C/A)}$$

This equals an angle cosine only under additional saturation conditions: in the reference example the ratio is $0.485$ where the cosine form gives $0.243$. Either way, reserve the property-progress budget first and cap the baseline separately.

**3.9.6 Centroid repulsion diverges in embedding space.** `[PROVED]` Under $\dot h_i=2(h_i-\bar h)$ deviations grow, with an unstable coincident equilibrium. This concerns unconstrained embedding-space dynamics and does not imply the same behaviour after state-space pullback. Log-det behaviour near duplicates and outliers is a design motivation, not established robustness of the full generator.

**3.9.7 A correct retraction.** `[PROVED]` `[CHECKED]` For residual $e$, smooth $F$, and projected gradient bounded away from zero:

$$z\leftarrow z-\lambda\frac{F_{t'}(z)-y^\ast}{\lVert Q\nabla F_{t'}(z)\rVert^2}Q\nabla F_{t'}(z) \qquad\Longrightarrow\qquad e_{\text{new}}=(1-\lambda)e+O(e^2)$$

The fourth-order improvement requires $\lambda=1$, or more generally $1-\lambda=O(e)$. `[CHECKED]` measured convergence orders: $3.99$ at $\lambda=1$, $2.00$ at $\lambda=\tfrac12$. Exact for $F=x^2+y^2-1$ after a tangent trial $(1,h)$: $e_{\text{new}}=(1-\lambda)h^2+\frac{\lambda^2h^4}{4(1+h^2)}$.

Neither variant is globally guaranteed to reduce error without a locality or acceptance condition. Not in the first experiment; the finite guard covers curvature more cheaply.

---

# §4 Motivation, not proof

**4.1 The embedding identity.** $\langle w,h_i-\bar h\rangle=p_i-\bar p$ is exact in embedding space and explains why interference might be expected. It does not transfer to state space unconditionally (§3.9.3). Use it to motivate the measurement.

**4.2 The relaxation heuristic.** `[HYPOTHESIS]` Under $de=-ke\,dt+\sigma\,dW$ with $k=w_g\kappa-\gamma c$, stationary spread is $\sigma^2/2k$, diverging as $\gamma c\to w_g\kappa$. Correct as a model; whether it applies is what the sweep tests.

---

# §5 Prior work and position

## 5.1 What is taken `[TAKEN]`

| Component | Owner |
|---|---|
| Baseline descent step followed by a diversity correction under analytically derived constraints, for diversity inside a loss's optimum set, including molecular conformation experiments | [Harmless Diversity, ICML 2022](https://proceedings.mlr.press/v162/gong22b/gong22b.pdf), eqs 4–6 and Algs 1–2 — the closest prior work |
| Normal/tangent decomposition for constrained particle sampling | [O-SVGD, NeurIPS 2022](https://arxiv.org/abs/2210.06447) |
| Repulsive joint-particle sampling; molecular conformers | [Particle Guidance, ICLR 2024](https://arxiv.org/abs/2310.13102) |
| Endpoint feature diversity, state-space pullback, velocity-relative cap (component A), preservation proof | [OSCAR, ICML 2026](https://arxiv.org/abs/2510.09060) |
| Reward-guided sampling with a repulsive particle correction, on molecules | [Stein Diffusion Guidance, ICML 2026](https://arxiv.org/abs/2507.05482) |
| Diverse particles under inequality constraints | [Constrained Stein Variational Trajectory Optimization](https://arxiv.org/abs/2308.12110) |
| Training-free constrained diffusion; trust-region guidance | [Trust Sampling, NeurIPS 2024](https://arxiv.org/abs/2411.10932); [Constrained Diffusers](https://arxiv.org/abs/2506.12544) |
| Gradient surgery at inference | Perp-Neg, APG, PDGrad |
| DFT evaluation of generated molecules at QM9's level of theory; diversity metrics reported beside MAE on 3D QM9 | [MolGuidance](https://arxiv.org/html/2512.12198v1) — 500 molecules, B3LYP/6-31G(2df,p) |

So: **A and B are prior work. C is engineering in a crowded category. Only D has a claim, and §5.2 states its size.**

## 5.2 The delta

Two boundaries that do **not** separate us from prior work:

1. `[PROVED]` The tolerance band is exactly the optimum set of the dead-zone loss $\max(0,\lvert F-y^\ast\rvert-\tau)^2$ when the band is reachable. "Tolerance band" rather than "optimum set" is therefore not a distinction.
2. $F=f_A(E_t(x))$ is a function of the state. "Through a network" rather than "on the state" is not a distinction either.

What remains, as differences to investigate rather than priority claims:

| | Harmless Diversity | Component D |
|---|---|---|
| Setting | population gradient descent on an objective | an edit inside a frozen generative sampler at inference |
| Constraint | analytic, on the main loss | a per-anchor progress envelope $A=\max(\tau,\lvert e\rvert)$, baseline-relative, so not the optimum set of a fixed loss when $\lvert e\rvert>\tau$ |
| Verification | analytic | a finite acceptance check against the actual guide, with rollback |
| Domain | images, meshes, conformers, ensembles | 3D property-targeted de novo molecules; simplex sequences |

One narrow gap survives a bounded search: we found no diversity-versus-property-error frontier for property-targeted 3D molecular generation. MolGuidance reports both axes separately. This states what a bounded search found, not an absolute absence.

**Assessment.** For the course, the brief permits combining and adapting prior work provided the paper explains what is new in the formulation; the table above is that explanation. For a publication, the mechanism alone is insufficient — a workshop paper would rest on empirical results, a demonstrated failure of the alternatives, or an efficiency finding. This document certifies priority for nothing.

## 5.3 Coupling: not pursued `[DESIGN]`

Every coupling idea is training-time, so the ablation grid costs 4–5 retrains with no cheap kill test and no recovery inside the deadline. Equivariant OT is taken since 2023 and [contested](https://arxiv.org/abs/2502.12456); pseudo-labelled noise coupling is C²OT (arXiv 2503.10636); property–noise coupling is [Reward Transport](https://arxiv.org/abs/2607.08781). Permutation-marginalised soft-Sinkhorn coupling is the most interesting unclaimed idea found, and is not certified unoccupied here.

The Reward Transport result on $\varepsilon$-prediction attenuating coupling-level signal is worth citing in the FM-versus-diffusion discussion. It does not establish that endpoint guidance is unaffected by parameterisation or guide-Jacobian conditioning; that needs its own measurement.

Euclidean diversity kernels can overestimate molecular diversity, since two configurations far apart in $\mathbb{R}^{3N}$ may be one structure rotated or relabelled. An invariant embedding and graph fingerprints avoid that specific failure.

---

# §6 Open items, ranked

1. **Read Harmless Diversity's eqs 4–6 and Algorithms 1–2.** It uses a baseline step followed by a diversity correction under analytic constraints, and secondary sources describe a threshold rather than strict equality. If that threshold is a tolerance, §5.2's remaining delta narrows to the frozen-sampler setting, the per-anchor envelope, and the finite guard. Highest priority.
2. **Read OSCAR's method section.** It owns component A and adjacent machinery.
3. **Reconcile the baseline.** TFG-Flow includes discrete guidance and iterative continuous updates ([§3.3–3.4, Alg 1](https://arxiv.org/html/2501.14216v2)). Ours is a related additive baseline, not a reproduction.
4. **Settle the diversity objective's remaining freedoms:** sign, feature normalisation, bandwidth caching and floor, jitter (§2.5).
5. **Fix $\delta$** from application need and evaluator calibration, before test-set experiments, and hold it across arms.
6. **Calibration is a diagnostic, not a bound.** A residual quantile on real validation molecules is not a distribution-free error bound for adaptively generated molecules under generator-induced shift.
7. **Cite the constrained-MCMC literature.** Riemannian MCMC on level sets is mature.

---

**Status.** Every claim carries a label; every `[CHECKED]` claim names its script in `audit/`. No molecular training, sampling, decoding, quantum-chemistry calculation or runtime benchmark has been performed. `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md` F0 is blocking.
