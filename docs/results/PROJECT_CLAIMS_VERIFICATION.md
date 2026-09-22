# Project verification: mathematics, evidence, novelty, and compute

Audit date: 15 September 2026 (America/New_York).

> **Role of this file, as of 16 Sep.** This is the **preserved error record**. The corrections it recommended have been adopted, and `COMPONENTS_ABCD_MATH_AND_PROOFS.md` v4.0 now contains only the current, correct formulation — it does not repeat the errors. Keep this file for the counterexamples and the runnable checks in `audit/`, which are what make the corrections verifiable.

Scope: the proposed method and evaluation in `PROJECT_GUIDE.md`, `COMPONENTS_ABCD_MATH_AND_PROOFS.md`, and `FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md`, checked against the local paper instructions and primary research sources. The repository is a scaffold; there are no implemented generators, trained checkpoints, or experimental results here to validate. This is a mathematical and research-design audit, not evidence that the method improves generated molecules.

**Recommendation: keep a small, corrected property-preserving diversity experiment. Do not use the current algorithm or theoretical/novelty claims unchanged.** The projection has a valid local guarantee. Several surrounding proofs are incorrect, the retraction can move in the wrong direction, and a missing prior paper contradicts the broadest novelty claim. A careful adaptation can still fit the course's stated innovation requirement.

The existing planning documents have been left unchanged. This report records corrections separately.

Follow-up: `COMPONENTS_ABCD_MATH_AND_PROOFS.md` sections 2.5 and 3.5-3.6 (component D) specify one consistent tolerance-aware candidate, its exact local solve, finite checks, limits, and feasibility sequence. Its CPU reference checks pass; molecular integration and performance are untested. This does not make all of the original planning documents correct.

## 1. What is mathematically correct?

### 1.1 The central projection and pullback correction are correct, with conditions

At a fixed time, define the *predicted property*

$$
F_t(x)=f_A(\hat x_1(x,t)),\qquad b=\nabla_x F_t(x),\qquad u=\nabla_x D.
$$

For nonzero $b$, the exact Euclidean tangent projection is

$$
d=u-\frac{b^\top u}{b^\top b}b.
$$

Then $b^\top d=0$, and $u^\top d=\|d\|^2\geq0$. It is the nearest vector to $u$ satisfying first-order property neutrality. Its normalized version maximizes linearized diversity gain over the unit tangent ball, unless the projected vector vanishes, in which case the optimum is nonunique.

If the Hessian of $F_t$ is bounded by $L$ along the step, then

$$
\left|F_t(x+h d)-F_t(x)\right|\leq \tfrac12 Lh^2\|d\|^2.
$$

This is a local, finite-step bound on the guide's predicted property. It is not a guarantee about the final decoded molecule, an independent evaluator, chemical validity, or the final sampling distribution.

The documents correctly identify that projection before a general pullback is insufficient. If $J=\partial\hat x_1/\partial x$, endpoint orthogonality $a_1^\top u_1=0$ does not imply $(J^\top a_1)^\top(J^\top u_1)=0$. The latter inner product contains $JJ^\top$. Project in the actual state coordinates, using the derivatives of the composed functions.

The global statement about preserving *all* inner products up to scale requires a similarity map; a particular pair can remain orthogonal under many other maps. Thus a small measured residual for one pair does not show that the whole Jacobian is close to a similarity transform.

### 1.2 Critical correction: use the property gradient, not the squared-reward gradient

The specified reward is

$$
r=-\frac{(F_t-y^*)^2}{2s^2},\qquad
a=\nabla r=-\frac{F_t-y^*}{s^2}b.
$$

Away from the target, the *exact* projectors based on $a$ and $b$ agree. At the target, $a=0$ even when $b\neq0$: the reward gradient has lost the normal direction but the property surface still has one.

The guide calls switching projection off at this point "correct behavior." It is the opposite of what property preservation requires. At the minimum of a squared loss, every direction is first-order reward-neutral; many change the property to first order.

Use $b=\nabla F_t$ to define the tangent space, and use $a=-(F_t-y^*)b/s^2$ only to move toward the target. This does not require an additional generator backward pass: recover $a$ from $b$ by scalar multiplication. If $b$ itself is nearly zero, the regular-level-set assumption fails; use a conservative fallback and record how often this happens.

Also, with regularization,

$$
b^\top\left(u-\frac{b^\top u}{\|b\|^2+\epsilon^2}b\right)
=\frac{\epsilon^2}{\|b\|^2+\epsilon^2}b^\top u.
$$

This is generally nonzero. The regularized operator is a soft projection, not an exact orthogonal projector. The claims of exact first-order neutrality in both documents need this qualification.

### 1.3 Critical correction: the proposed retraction is wrong

`PROJECT_GUIDE.md` §2.3.7 and algorithm line 10 use

$$
g\mathrel{+}=\frac{y^*-F_t(x)}{\|a\|^2+\epsilon^2}a.
$$

That is not Newton correction of $F_t(x)=y^*$, because $a$ is the reward gradient, not $b$. For the simplest case $F(x)=x$, $y^*=0$, $s=1$, $x=1$, we get $a=-1$ and the proposed correction is **+1**: it increases the property error.

After taking a trial step to $z$, a proper damped scalar-constraint correction is

$$
z_{\rm new}=z-\lambda\frac{F_{t'}(z)-y^*}{\|Q\nabla F_{t'}(z)\|^2}
Q\nabla F_{t'}(z),\qquad 0<\lambda\leq1,
$$

where $t'$ is the time of the trial state and $Q$ enforces linear modality constraints. Use safeguards for small gradients and check the actual residual.

Three consequences:

- Evaluate the property at the *trial point*. The old residual cannot reveal curvature drift created by a new tangent step. Reusing the old Jacobian can be an approximation, but a new residual evaluation is still needed.
- This is a state displacement. Adding it to a velocity that is subsequently multiplied by $\Delta t$ changes the algorithm.
- Correcting a noisy intermediate prediction all the way to the final target may be too aggressive. For a clean ablation, retract the extra diversity move toward the property value achieved by the corresponding baseline step, or apply target retraction only when close enough to the target.

Under regularity, an exact tangent trial from a level set has $O(h^2)$ residual; one correctly evaluated Newton step can reduce this to $O(h^4)$. The documented stale-residual update does not obtain that result. **Remove line 10 from the first experiment; add a genuine, costed corrector only later.**

### 1.4 The systematic-opposition proof does not hold in state space

Let $L=\partial h(\hat x_1(x,t))/\partial x$, $f_A=w^\top h+c$, and $d_h=\nabla_hD$. The actual interference is

$$
a^\top u=-\frac{F_t-y^*}{s^2}\,w^\top LL^\top d_h.
$$

The documents analyze $w^\top d_h$, omitting $LL^\top$. That metric factor can change both magnitude and sign. Sharing an embedding does not remove it.

Even in embedding space, centroid repulsion only yields

$$
\nabla_{h_i}r_i^\top\nabla_{h_i}D
=-\frac{2}{s^2}(p_i-y_i^*)(p_i-\bar p).
$$

This is always nonpositive only under additional conditions, such as every sample sharing target $y^*=\bar p$. If the batch is below the target, repulsion can push above-average samples *toward* the target. The proposed protocol uses different targets per sample, making the universal assertion particularly unsuitable.

For positive but unequal pairwise weights, $\sum_j c_{ij}(p_i-p_j)$ points away from a *weighted local mean*, not necessarily the ordinary batch mean. Log-det weights can also have mixed signs. A bandwidth differentiated through the batch introduces further terms.

The claimed straight-line scatter plot is also wrong for the signed *reward–spread* inner product. In the ideal centroid case with $\bar p=y^*$, it is proportional to $-(p_i-y^*)^2$: a negative parabola. Property-gradient alignment and reward-gradient alignment are different diagnostics.

**Replace T0's hard sign-fraction gate.** Measure the state-space interference magnitude and sign, normalized removed energy $(b^\top u)^2/(\|b\|^2\|u\|^2)$, retained diversity gain, and actual finite-step drift. A sign fraction near 0.5 alone neither proves nor disproves usefulness. Randomly signed interference can increase absolute error or error variance instead of canceling harmlessly.

### 1.5 The stationary-variance argument is not a proof of improvement

The displayed equation is a deterministic relaxation:

$$
\dot e=-ke,\qquad k=w_g\kappa-\gamma c.
$$

It gives $\operatorname{Var}(e_t)=e^{-2kt}\operatorname{Var}(e_0)$, with zero stationary variance when $k>0$. To obtain an Ornstein–Uhlenbeck stationary variance, one must add stochastic forcing:

$$
de=-ke\,dt+\sigma\,dW_t,
\qquad \operatorname{Var}_{\rm stationary}(e)=\frac{\sigma^2}{2k}.
$$

Even then, variance is not MAE; for a zero-mean Gaussian, $\mathbb E|e|=\sqrt{2\operatorname{Var}(e)/\pi}$. A finite nonstationary sampler, changing Hessians, clipping, model drift, and decoding are not described by that stationary formula.

Retain the intuition that repulsion can weaken restoring forces, but remove the claims that frontier dominance and flat MAE versus $\gamma$ have been derived. They are empirical hypotheses.

### 1.6 Constraint composition and trust-region clipping need changes

Projecting onto a reward tangent and *then* onto the simplex tangent can destroy reward orthogonality. With a symmetric orthogonal modality projector $Q$, compute

$$
b=Q\nabla F_t,\quad z=Q\nabla D,\quad
d=z-\frac{b^\top z}{b^\top b}b.
$$

Then $Qd=d$ and $\nabla F_t^\top d=0$. Equivalently, the joint projector is $Q-bb^\top/(b^\top b)$. For molecules with exactly translation-invariant composed functions, gradients already satisfy zero-CoM constraints; the extra projection is redundant. The simplex case generally requires the joint treatment.

On the simplex, sum-zero is necessary but insufficient: boundary feasibility requires inward or tangent directions at zero coordinates. Clipping and renormalizing afterward can reintroduce first-order property drift. Use a feasible step bound/tangent-cone method or a defined interior parameterization, and measure the effect of decoding. The terminal state has boundary complications even if earlier states are interior.

The shared cap on $g_r+\gamma d$ causes a second problem: if adding $d$ triggers clipping, the cap reduces the *reward* component too. For orthogonal unit vectors and a unit cap, reward progress falls from 1 to $1/\sqrt2$. Thus exact tangency alone does not make the *full algorithm's* property trajectory independent of diversity strength.

For the stated constrained optimization problem, feasibility requires $|c|\leq\eta\|a\|$. When feasible and $u_\perp\neq0$, its optimizer is

$$
m^*=\frac{c}{\|a\|^2}a+
\sqrt{\eta^2-\frac{c^2}{\|a\|^2}}\frac{u_\perp}{\|u_\perp\|}.
$$

Arbitrary weights followed by a shared cap are not generally this optimizer. To preserve a fixed reward step, reserve its budget first and spend only the remaining orthogonal budget on diversity. A velocity-relative cap also does not certify chemical-manifold membership, and $\|v\|$ need not stay nonzero.

### 1.7 The distributional motivation overclaims

Even if a sampler exactly produced

$$
p_w(x)\propto p(x)\exp[-w(f(x)-y^*)^2/(2s^2)],
$$

increasing $w$ need not collapse diversity *along* the target level set. For independent standard-normal coordinates $(x,z)$, $f(x,z)=x$, and $y^*=0$, the variance of $x$ shrinks to $1/(1+w/s^2)$, while the variance of $z$ remains exactly 1. This directly disproves the document's universal single-mode claim.

At regular levels, conditioning on a scalar equality has surface density proportional to $p(x)/\|\nabla f(x)\|$, by the coarea formula. Tangent motion and retraction alone do not reproduce this weighting. An arbitrary FM velocity is not a score; adding projected Brownian noise without a derived drift does not establish constrained Langevin sampling from the desired conditional distribution.

Moreover, ordinary guided samplers do not automatically realize the claimed tilted terminal density. Related misconceptions for CFG are explicitly analyzed in [Classifier-Free Guidance is a Predictor-Corrector](https://arxiv.org/abs/2408.09000). Frame the project as **improving useful conditional sample sets**, unless you derive the actual sampling law.

### 1.8 Other corrections worth making

- **DNA is not a line-for-line proof transfer.** For linear logits $\ell=Wh+c$, $\nabla_h\log p(c|h)=w_c-\sum_jp(j|h)w_j$, not just $w_c$. Multiclass log-probability level sets are not generally planes. DNA class targeting has no scalar property-equality Newton correction as currently written. Protect a log-probability or margin explicitly. Orthogonality to a CFG velocity difference is not automatically orthogonality to a property gradient.
- **Embedding dimensions do not imply structural freedom.** The reachable rank is at most $\min(d,256)$, may be much smaller, and a property predictor can encode mainly the property. In the extreme $h(x)=q(f(x))$, exact property projection removes all diversity signal. The claim that 255 useful structural directions remain is unsupported. Invariance prevents counting pure rigid transforms as new molecules; it does not prevent arbitrary invalid structures from spreading in the embedding.
- **Centroid repulsion does not inherently converge to a shell.** Under $\dot h_i=2(h_i-\bar h)$, deviations grow exponentially. Shell equilibria require other forces. Centroid energy is weak against duplicates and outliers, but the claimed universal shell theorem is false.
- **Log-det definitions conflict.** The guide maximizes an RBF-kernel log determinant; the math file also gives a negative linear-Gram log determinant that must be minimized. These are different objectives. Choose one, specify the sign, feature normalization, fixed/detached bandwidth, and jitter. Dense kernel construction costs $O(B^2k)$; exact Cholesky/log-det generally costs $O(B^3)$, not merely $O(B^2)$.
- **An endpoint readout is a proxy.** For ideal linear FM, $x_t+(1-t)v(x_t,t)$ is a conditional-mean endpoint estimate, not necessarily the endpoint of the remaining ODE trajectory. Properties of that mean are not generally expected endpoint properties.
- **NFE is not total work.** Report generator forwards, backwards, predictor calls, wall-clock, peak memory, and samples per GPU-hour. A second VJP does not universally cost exactly 1.5 times the method. Batched VJPs can also be implemented; separate derivatives are needed, not necessarily two separately launched calls.
- **The baseline is not automatically a TFG-Flow reproduction.** The published method includes discrete guidance and iterative continuous-state updates before computing its guided velocity. Your single additive all-continuous update is a related baseline unless those differences are reconciled. See [TFG-Flow, §3.3–3.4 and Algorithm 1](https://arxiv.org/html/2501.14216v2).
- **QMC is not invalid above three dimensions.** Effective dimension and integrand structure matter; a 75-dimensional state alone does not justify rejecting Sobol sampling. It remains a cheap control, not an established novelty. See [Wang and Fang, effective dimension and QMC](https://doi.org/10.1016/S0885-064X(03)00003-7).

## 2. Will the innovation make the project better?

**Plausible, conditional, and testable; not guaranteed.** Correct projection increases the selected diversity surrogate locally without first-order drift in the selected predicted property. That is a useful mechanism if:

1. Unprojected diversity creates meaningful property interference at useful sampling times.
2. The projected direction retains enough structurally meaningful motion.
3. The guide is locally trustworthy and its gradient predicts changes that the independent evaluator also sees.
4. Finite steps, boundaries, clipping, and discretization do not erase the advantage.
5. Extra inference cost does not outweigh the added useful samples.

It can fail because the tangent direction only changes geometry within the same graph, leaves the data distribution, exploits the predictor, cannot cross disconnected chemical modes, or consumes the reward-step budget. A weak base model can also hide the mechanism: "relative comparison" does not make a low-quality model a reliable final kill test.

**Minimal experiment:** use one frozen checkpoint, one property, fixed target/size groups, and common seeds. Compare vanilla guidance, unprojected diversity, flow-velocity-orthogonal diversity (an OSCAR-style adaptation), and corrected property-orthogonal diversity. Keep the reward step, gate, norm rules, kernel, batch size, and sweep budget matched. Include random tangent noise as a cheap control if feasible. Compare both matched steps and matched wall-clock. First use the simplest method without the faulty corrector or additional gates; add components only after the core comparison is informative.

Before generation, test the property derivative with central differences, tangent drift versus step size, symmetry handling, simplex boundaries, and the difference between guide and evaluator gradients. Then inspect the *independent* useful-output frontier. These experiments, not T0 sign fractions or a toy with prescribed outcomes, should determine whether to continue.

The local paper instructions explicitly allow combinations/adaptations with an explained methodological change. A careful formulation and causal ablations can satisfy that standard without claiming a new general sampling principle.

## 3. What should the metrics be?

### 3.1 Primary scientific result

**Within-target structural diversity at matched independent property accuracy and comparable validity, shown under a fixed compute budget.** Report each target group, then macro-average groups. A single global diversity score over thousands of different target values can be high even if every individual target collapses to one molecule.

Use two separate protocols:

- A conventional target-list protocol to compare property MAE with relevant published methods.
- A repeated-condition protocol for the innovation: select a modest set of validation-defined targets and atom counts, then generate many samples per condition. For example, start with 6–10 feasible conditions and 512 attempts per condition, with identical groups/batches across arms; increase only for finalists. Match $(N,y^*)$ where that is the task, rather than silently counting different sizes as diversity.

### 3.2 Primary operational number

For condition $c$, let

$$
A_{c,\delta}=\{x_i:\operatorname{valid}_{2D+3D}(x_i),\ |f_B(x_i)-y_c^*|\leq\delta\}.
$$

Report

$$
U_\delta(c)=1000\,
\frac{|\{\operatorname{canonicalGraph}(x):x\in A_{c,\delta}\}|}
{M_c},
$$

where $M_c$ is the number of attempted samples. In words: **distinct valid molecules meeting the property tolerance per 1,000 attempts**. Also report this count per GPU-hour. This operational metric combines success, uniqueness, and validity transparently; it is a reporting choice, not a novelty claim.

Fix attempt counts: unique-count yield is nonlinear in sample count, so "per 1,000" scaling does not make arbitrary sample sizes comparable. Freeze the primary tolerance from application needs and evaluator calibration, and show a small tolerance curve rather than optimizing the threshold on test results. Do not claim accuracy finer than the evaluator can support.

### 3.3 Metric panel

| Role | Metric | Guardrail |
|---|---|---|
| Target control | Independent $f_B$ MAE in physical units, plus tolerance success rate | Report validity and failure rates beside MAE; validate evaluator error on held-out real data |
| Diversity | Morgan/Tanimoto Vendi Score within each condition | Fixed sample counts and kernel; retain duplicates; report accepted count, not just a high score on a few survivors |
\left| Useful output \right| $U_\delta$, plus distinct successful scaffolds per fixed attempts | Define graph identity and acyclic/empty-scaffold handling |
| Quality | Connected RDKit validity, atom/molecule stability, 3D geometry checks | RDKit validity alone does not establish plausible coordinates |
| Reference coverage | Conditional fingerprint coverage and density against real data | Fixed real reference size, neighborhood rule, target/size bins; apply property acceptance for useful coverage |
| Robustness | Independent quantum-chemistry check on a matched random/stratified subset | Same protocols/units; report convergence failures and calibration errors |
| Efficiency | Useful candidates/GPU-hour, seconds per fixed attempts, peak memory, forward/backward counts | Same hardware, batch sizes, and total tuning budget |

The [Vendi Score](https://arxiv.org/abs/2210.02410) is $\exp[-\sum_j\lambda_j\log\lambda_j]$ for eigenvalues of a trace-normalized positive-semidefinite similarity matrix. It complements uniqueness, but is not immune to invalid or out-of-distribution outputs. Use repeated equal-size subsamples when comparing diversity among accepted samples; keep yield as the complementary all-attempts metric.

[Density and coverage](https://proceedings.mlr.press/v119/naeem20a.html) are useful reference-based diagnostics. Their usual neighborhood radii are based on real-data nearest neighbors; a fixed-radius implementation is a defined adaptation, not automatically the published metric. Fingerprints ignore 3D geometry, so malformed conformers can still look close to real graphs. Neither coverage nor quantum chemistry makes the whole evaluation "unfakeable."

Scaffolds are secondary on small molecules: many acyclic molecules share an empty Murcko scaffold. Use full-graph uniqueness and soft fingerprint diversity too. Dividing by attempts measures a useful yield but does not remove every statistical effect of validity filtering.

### 3.4 Frontiers and uncertainty

- Use MAE–diversity frontiers with validity/success annotations, and useful-yield versus compute as the practical comparison. Compare matched-error points chosen on validation data, then evaluate held-out samples.
- Hypervolume can summarize the frontier with fixed axis scaling and a reference worse than the acceptable operating region. The unguided point is often very diverse, so it is not automatically a suitable worst reference. DIP and R2 are optional; R2 also depends on normalization, ideal point, and weights.
- **Resample independent generated batches/runs, not individual molecules as if independent.** Repulsion couples molecules within a batch. Use paired cluster bootstrap across matched seed/batch IDs and target strata, recomputing both methods' metrics/frontiers. For nonadditive richness/Vendi metrics, use repeated independent runs or appropriate subsampling as a check against bootstrap-induced duplicates. More within-cell molecule resamples cannot replace independent runs.
- One training seed supports a result conditional on one checkpoint. If extra calendar time permits a few sequential runs, prioritize a second independent checkpoint for the central result over a large secondary ablation grid.

### 3.5 DNA counterpart

Use independent target-class success, distinct successful sequences per fixed attempts, class-conditional Vendi on a fixed k-mer kernel, class-conditional coverage, and FBD using an independent encoder. Report k-mer distribution agreement, motif composition, and decoded-sequence results. Random DNA can have high Hamming distance and uniqueness, so those alone are insufficient. Keep genomic splits/homology controls and class-specific reporting.

## 4. Is it truly innovative relative to current research?

**The general mechanism is established. The exact adaptation may be a useful incremental contribution. The documents' strongest novelty statements are not supportable.** This is a targeted search of relevant primary sources through the audit date, not a proof that every current paper has been excluded.

| Prior work | Established overlap | Consequence |
|---|---|---|
| [O-SVGD, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f092c84221d73387a6a5dd7517c500a5-Abstract-Conference.html) | Constraint-gradient normal/tangent decomposition, including interacting-particle sampling | Neither level-set exploration nor projected repulsion is new as a general principle |
| [Particle Guidance, ICLR 2024](https://arxiv.org/abs/2310.13102) | Repulsive joint-particle sampling, including molecular conformers | Do not claim the first diversity intervention for molecules |
| [OSCAR / Letting Trajectories Spread, v2 May 2026](https://arxiv.org/html/2510.09060v2) | Endpoint feature-volume guidance, state-space pullback, velocity-orthogonal control/noise, relative norm safeguard | Much of the proposed architecture is already present; the changed protected quantity is the meaningful delta |
| [Stein Diffusion Guidance, ICML 2026, v3](https://arxiv.org/html/2507.05482v3) | Reward-guided diffusion with repulsive particle correction; molecular-graph experiments | **Directly contradicts "no repulsion method has ever been composed with a property/reward gradient."** It does not establish that your exact 3D property-tangent algorithm is already published |
| [MolGuidance, December 2025](https://arxiv.org/html/2512.12198v1) | 3D property-guided generation with validity, uniqueness, scaffold diversity, and computational evaluation | Measuring diversity beside property control is established; the causal projection comparison is a more specific contribution |
| [EDDY, May 2026](https://arxiv.org/html/2605.06553v1) | Marginal-preserving diversification under its assumptions | First-order predictor neutrality is a different and weaker type of guarantee |
| [SGRPO, May 2026](https://arxiv.org/html/2605.08659v1) | Biomolecular utility–diversity frontiers, HV/DIP/R2, independent-run confidence intervals | Frontiers and uncertainty estimates are good evaluation practice, not a stand-alone novelty claim |

A defensible provisional claim is:

> We adapt constrained particle diversification to property-guided continuous molecular generation, using the property derivative in the sampler's feasible state space. We test whether preserving the local property constraint improves independently evaluated conditional diversity at matched accuracy and computation, and examine transfer to sequence simplexes.

This is a hypothesis and description of the proposed adaptation, not a claim of verified priority or demonstrated improvement. The chain rule alone is standard mathematics; presenting the pullback order as a new general discovery overstates it. A useful contribution would be a correct formulation, a demonstrated failure of relevant alternatives, and a reproducible gain.

For a course project: potentially sufficient under the provided brief. For a research publication: incremental unless the experiments reveal a strong mechanism, a genuinely distinct constraint-aware formulation, or an important efficiency result. Additional ingredients do not automatically make the method new.

## 5. Alternatives suited to limited GPUs

The wording of the request suggests calendar time is available but GPUs are scarce. The options below distinguish zero generator retraining from small auxiliary-model training. None is certified unpublished; each needs targeted positioning before being claimed as an innovation.

### 5.1 Best first candidate: tolerance-aware diversification

The user needs molecules inside a property tolerance, not an exact floating-point equality. Replace always removing one direction with a feasible-band update. For an actual state increment $\delta$, one local subproblem is

$$
\max_d\;u^\top d-\frac{1}{2\eta}\|d\|^2
\quad\text{s.t.}\quad
\left|e+b^\top(\delta_{\rm base}+d)\right|\leq\Delta,
\quad Qd=d,\quad \|d\|\leq R,
$$

where $e=F_t(x)-y^*$. In a time-stepping implementation, include the time-change term or formulate/check the constraint at the baseline trial point at $t+\Delta t$. Use slack or a progress constraint when the linearized band is unreachable. Check the finite trial property and backtrack when needed; count those calls.

Deep inside the band, more diversity directions are available; at a boundary, suppress outward motion while retaining helpful directions. This directly tests a weakness of equality projection, which discards useful normal movement even when the application can tolerate it. Transfer to DNA through a target-class probability or margin inequality.

**Cost:** no generator retraining; a small constraint solve, with optional additional forward evaluations. **Controls:** exact projection, conflict-only projection, same-budget unprojected repulsion, and post-generation filtering/selection. **Claim limit:** constrained guidance already exists, including [Constrained Diffusers](https://arxiv.org/abs/2506.12544); the proposed contribution would be the condition-specific diversity/feasibility controller and its evidence, not inequality constraints themselves.

### 5.2 Best use of a little training: protect several guide estimates

The current tangent plane protects one learned predictor and may expose its blind spots. Train a small ensemble of guide heads or modest predictors; keep the evaluation oracle entirely separate. With projected property gradients as rows of $B$, the exact joint tangent operator is

$$
P=Q-B^\top(BB^\top)^\dagger B.
$$

Here $BQ=B$, and $\dagger$ is a pseudoinverse. With several guides this requires only a small Gram-system solve, although obtaining the gradients has a real inference cost. Alternatively constrain predicted property changes to an uncertainty-calibrated band instead of enforcing equality for every guide.

**Hypothesis:** directions that fool one guide will more often disturb another, reducing disagreement and independent-evaluator error at comparable useful diversity. **Cost:** a few small sequential trainings; generator unchanged. **Risks:** shared heads may have highly correlated blind spots; too many constraints can eliminate useful motion. **Controls:** single guide, ensemble-mean guide, ensemble constraints, equal total compute. Ensemble uncertainty and multi-constraint projection are established ideas; novelty must lie in a measured solution to the guidance failure, not merely using an ensemble.

### 5.3 Cheap representation experiment: structural features conditional on the target

Learn or choose an invariant structural embedding with useful information beyond the scalar property, for example through a small structural auxiliary objective. Continue to enforce the property constraint in state space.

**Correction to the first version of this audit:** subtracting the same conditional mean $\mu(y,N)$ from every feature in a repeated-condition group leaves every pairwise distance unchanged: $(h_i-\mu)-(h_j-\mu)=h_i-h_j$. It therefore cannot change the proposed RBF diversity objective. That earlier suggestion was a no-op and should not be implemented as an innovation. A different encoder or a nontrivial property-subspace transformation can change the geometry, but neither proves statistical independence.

**Cost:** no generator retraining; potentially a small encoder/head fit. **Controls:** original guide features, independent structural features, and real-data conditional coverage. **Risk:** the auxiliary model may capture irrelevant variability. **Novelty:** uncertain; representation learning and disentanglement are established, so this is a hypothesis to test rather than a claim of invention.

### 5.4 If sequential retraining is genuinely affordable: distill the expensive controller

After a teacher controller demonstrates a gain, collect its states and control vectors and train a small equivariant residual controller on frozen generator features. For a batch-coupled teacher, the student must receive a permutation-invariant summary of the current sample set; an independent per-molecule student cannot in general reproduce the teacher's joint dependence. Retain an inexpensive feasibility check where necessary.

**Hypothesis:** preserve useful-output performance while reducing generator backward passes. **Cost:** one teacher data-collection pass and one small supervised training, not a complete model-family retraining grid. **Controls:** teacher, student, unguided/same-budget baseline; compare total amortized cost. Guidance distillation is already established, e.g. [Meng et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Meng_On_Distillation_of_Guided_Diffusion_Models_CVPR_2023_paper.pdf). The research question is whether set-aware property-preserving control can be distilled efficiently across these geometries.

### 5.5 Where to spend the limited compute

1. Establish valid base generation and a calibrated independent evaluator.
2. Correct the projection and run the four-arm repeated-condition experiment.
3. Advance **one** alternative, preferably tolerance-aware control. Do not add all four.
4. Spend additional sequential training on replication and a useful guide/structural model before speculative full-generator changes.
5. Evaluate finalists at both equal attempts and equal wall-clock. Include generate-and-select as a practical baseline; count rejected candidates and selection costs.

Property-aware OT already overlaps [Reward Transport](https://arxiv.org/abs/2607.08781). Simple soft-Sinkhorn alignment does not automatically equal marginalization over discrete atom permutations: it can average assignments into geometrically unrealizable configurations. Neither that option nor repulsion plus Feynman–Kac steering should be called "genuinely unoccupied" without a more specific derivation and comparison. Resampling changes genealogies, and deterministic clones need an explicit diversity-restoring move; the combination also changes the target law unless properly accounted for.

## 6. Verification performed and remaining evidence

Executed `audit/math_checks.py` successfully. The saved results in `audit/math_checks_results.json` reproduce the pullback counterexample, the nonzero regularized-projection residual, the incorrect retraction sign, the shared-Jacobian sign reversal, the target/batch-mean mismatch, the quadratic alignment scatter, the simplex projection-order failure, and the reward-progress loss under shared clipping. They also check a correctly evaluated Newton correction and the elementary relaxation/Gaussian calculations.

These are algebraic checks only. No molecular training, quantum-chemistry calculation, runtime benchmark, or empirical novelty-performance claim was performed. No search can certify absence from all current research. The strongest actionable conclusion is that **the project has a testable core, but its current proofs, full algorithm, feasibility gates, and broad priority claims require correction before implementation or submission.**
