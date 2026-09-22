# SMG, second-order Heun, and the closest prior formulas

For a full teaching document covering SMG's purpose, proofs, approximation orders, and contribution limits, see [Why innovative?](SMG_WHY_INNOVATIVE.md).

**18 Sep 2026.** Companion to [Innovation v7](SMG_FULL_MATH_AND_PROOFS.md).
SMG is the primary innovation under evaluation. Heun is an adopted numerical solver.
This note supersedes broader novelty claims in the original 17 Sep brainstorm.

**Status:** Heun's prior use is verified; the formula comparisons below are scoped to the
specified published equations. SMG's research novelty and molecular benefit remain unestablished.
The solver experiment is specified in [Feasibility tests, M-4](FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md#m-4-eulerheun--guidance-comparison--required-solver-control).

## 1. Three different meanings of second order

| Object | Meaning | Status |
|---|---|---|
| Covariance identity | Relates a posterior's second central moment to a denoiser Jacobian | Established identity under independent Gaussian corruption and an exact posterior mean |
| SMG mean correction | Retains the quadratic term in the Taylor expansion of the property | Established moment expansion; its particular use in guidance is the candidate |
| Heun integration | A second-order numerical approximation to a specified ODE | Established numerical method, already used for diffusion and rectified-flow sampling |

SMG's mean correction is second order in the clean-sample deviation and first order in its
covariance. Its variance estimate uses a first-order property expansion. Neither description
specifies the order of the ODE solver. Euler and Heun can integrate the same SMG field.

## 2. What the cited methods actually share with SMG

Use common notation throughout this note:

$$m=\mathbb E[X\mid x_t],\quad J=\partial m/\partial x_t,\quad
\Sigma=\operatorname{Cov}(X\mid x_t),\quad g=\nabla f(m),\quad H=\nabla^2 f(m).$$

Here $f$ is the scalar property, $y$ its target, and $s>0$ the Gaussian target width.
The practical v7 field in score units is

$$c=\tfrac12\operatorname{tr}(H\Sigma),\qquad v_f=g^\top\Sigma g,\qquad
\boxed{G_{\rm SMG}=\frac{y-f(m)-c}{s^2+v_f}J^\top g.}$$

The comparator is $G_{\rm plug}=(y-f(m))J^\top g/s^2$. These are guidance fields,
before conversion to the sampler's velocity. SMG holds $c$ and $v_f$ fixed during the
local derivative, but recomputes them at each field evaluation.

### 2.1 Heun: prior use is definite

[Karras et al., NeurIPS 2022](https://papers.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf),
Algorithm 1, PDF page 5, explicitly uses Heun's second-order predictor/corrector for
deterministic diffusion sampling. That settles prior use of this solver. Applying it to
our guided field is an implementation choice, not a new integration method.
This is the image-generation **Elucidating the Design Space** paper, not the molecular E(3)-EDM paper.

Flow-model applications are also verified:

- [Lee, Lin and Fanti, *Improving the Training of Rectified Flows*, NeurIPS 2024](https://arxiv.org/html/2405.20320v2),
  section 5.3 and Figure 4, compare Euler and Heun on rectified flows; Appendix D, Algorithm 2,
  implements the Heun predictor and averaged velocities. This is sampling a flow model,
  independently of the paper's use of a diffusion teacher to generate training pairs.
- [Wang et al., *Taming Rectified Flow for Inversion and Editing*, ICML 2025](https://proceedings.mlr.press/v267/wang25ce.html),
  [Appendix B.1](https://openreview.net/pdf/0f4223921214bc987e12a04945fc1246f65e8efb.pdf),
  uses Heun with pretrained rectified-flow models as a generation/inversion baseline.
- The official [*Flow Matching in Latent Space* implementation](https://github.com/VinAIResearch/LFM)
  also documents fixed-step Heun sampling.

Therefore, neither Heun itself nor its application to flow matching can be claimed as new here.

### 2.2 TMPD: an exact shared special case

[Boys et al., TMPD](https://arxiv.org/html/2310.06721v3), equations (10)-(11), use a linear
observation map. For the scalar affine property $f(X)=a^\top X+b$, their covariance-frozen
guidance becomes

$$G_{\rm TMPD}=J^\top a\,
\frac{y-a^\top m-b}{s^2+a^\top\Sigma a}.$$

Because $H=0$, this is exactly the scalar linear-property limit of $G_{\rm SMG}$ with
the same covariance and derivative convention. Thus "guidance with two moments" is shared
prior work. The extra nonlinear-property mean term $c$ distinguishes v7 from this displayed
linear-observation formula; it does not establish priority over the rest of the literature.

### 2.3 On the Guidance of Flow Matching: the identity and plug-in guidance are shared

[On the Guidance of Flow Matching](https://arxiv.org/html/2502.02150v2), Proposition 3.4,
already gives the covariance/Jacobian identity. Equation (6) gives a time-scaled gradient
of an energy evaluated at the denoising mean. Set that energy to

$$E(X)=\frac{(f(X)-y)^2}{2s^2},\qquad
-\nabla_{x_t}E(m)=\frac{y-f(m)}{s^2}J^\top g.$$

This recovers the plug-in part, after accounting for the paper's conversion to velocity units.
Section 3.4 also treats Gaussian-posterior approximations, Monte Carlo integration and analytic
linear inverse-problem guidance. These are substantive overlaps. Equation (6) itself does not
contain our $c$ or $v_f$ terms; a difference from that equation is not a difference from every
method covered by the framework.

### 2.4 GGDOpt: quadratic energy guidance differs from quadratic property moments

[A Gradient Guided Diffusion Framework for Chance Constrained Programming](https://arxiv.org/html/2510.12238v1),
equation (20), expands an optimization objective. Rename that objective $E$ to avoid confusing
it with our property $f$. With $u=x_t$, posterior mean $m$, isotropic posterior variance
$\rho^2$, and tilt strength $\beta$, its displayed formula is

$$G_{\rm GGDOpt}=-\frac1{\rho^2}\left[
\left(B+\frac1{\beta\rho^2}I\right)^{-1}
\left(-Bu+a-\frac{m}{\beta\rho^2}\right)+m\right],
\quad a=\nabla E(u),\quad B=\nabla^2 E(u).$$

For a matched squared-property energy,

$$\nabla E(z)=\frac{f(z)-y}{s^2}\nabla f(z),\qquad
\nabla^2E(z)=\frac{\nabla f(z)\nabla f(z)^\top+(f(z)-y)\nabla^2f(z)}{s^2}.$$

This acts through an inverse matrix built from energy curvature. SMG instead corrects property
moments and multiplies $J^\top g$ by a scalar. They are not generally identical formulas.

**Our algebraic counterexample (not a benchmark):** take $f(z)=z^2$, $u=m=y=s=\beta=1$,
$\rho^2=\Sigma=0.1$, and $J=0.5$. Then $a=0$, $B=4$, so the displayed GGDOpt expression
is zero. SMG has $c=0.1$, $v_f=0.4$, and $Jg=1$, giving

$$G_{\rm SMG}=-\frac{0.1}{1.4}=-0.0714285714\ldots\neq0.$$

The moments are consistent with a Gaussian channel: prior mean $2/3$, prior variance $2/15$,
$\alpha=1/2$, noise variance $1/10$, and observation $u=1$ give exactly those $m,\Sigma,J$.
This proves nonidentity of these formulas at a valid state. It proves neither better performance
nor novelty. A matched-energy implementation remains a relevant comparator.

### 2.5 STSL and MMPS: further substantial overlaps

[Rout et al., *Beyond First-Order Tweedie*, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Rout_Beyond_First-Order_Tweedie_Solving_Inverse_Problems_using_Latent_Diffusion_CVPR_2024_paper.pdf)
already uses second-order Tweedie information for posterior sampling. Theorem 4.4 and equation (6)
give a surrogate guidance containing a data-fit gradient and a term proportional to
$-\nabla_{x_t}\operatorname{tr}(\nabla_{x_t}^2\log p_t(x_t))$.
This is noisy-density curvature, whereas SMG's $c$ contracts the clean-property Hessian with
posterior covariance. These displayed corrections differ, but STSL rules out the broad claim
that adding inexpensive second-order information to guidance is new.

[Rozet et al., *Learning Diffusion Priors from Observations by Expectation Maximization*](https://arxiv.org/html/2405.13712v5),
section 4.2, equations (19)-(22), derives moment matching posterior sampling (MMPS), with
Tweedie covariance and a neglected covariance derivative. Its scalar affine-observation case
also gives the formula in section 2.2. The paper's linear-observation derivation does not
establish the nonlinear-property mean correction; it does establish further overlap in the
moment approximation, uncertainty denominator, and derivative convention. The audited version
is v5 (31 Oct 2025), subsequent to its NeurIPS 2024 publication.

### 2.55 TFG's Monte Carlo smoothing: the closest competitor, and it was missed until 19 Sep

**Added 19 Sep after reading the released source.** Earlier drafts of this audit
treated plain plug-in guidance as the thing SMG improves on. That understated the
field. [TFG (NeurIPS 2024)](https://arxiv.org/html/2409.15761v2) is a *design space* rather
than one formula, with four knobs — gradient strength $\rho$, mean-refinement $\mu$,
recurrence, and iteration — and one of them attacks the same defect SMG does.

From [`methods/tfg.py`](https://github.com/YWolfeee/Training-Free-Guidance/blob/main/methods/tfg.py):

```python
mc_eps  = self.get_noise(std, x.shape, self.args.eps_bsz)     # eps_bsz = 4
flat_x0 = (x0[None] + mc_eps).reshape(-1, *x0.shape[1:])
avg_logprobs = torch.logsumexp(outs.reshape(...), dim=0) - math.log(mc_eps.shape[0])
```

Written out, with $m=\hat x_1$ and $K$ = `eps_bsz`:

$$L=\log\frac1K\sum_{k=1}^{K}p\big(y\mid m+\varepsilon_k\big),\qquad
\varepsilon_k\sim\mathcal N(0,\sigma^2 I)$$

$$\frac{\partial L}{\partial m}=\sum_k w_k\,\frac{y-f(m+\varepsilon_k)}{s^2}\,
\nabla f(m+\varepsilon_k),\qquad w_k=\operatorname{softmax}_k\big(\log p_k\big)$$

So TFG returns a **softmax-weighted average of per-sample plug-in gradients**, not a
single gradient at the mean. It is a Monte Carlo estimate of
$\log\mathbb E_\varepsilon[p(y\mid m+\varepsilon)]$ — an attack on exactly the fact that
motivates SMG, that the property of the mean is not the mean of the property.

**This must be stated plainly: it is a far closer competitor than plain plug-in, and
beating plug-in alone would not be an interesting result.**

What still separates the two, stated as differences to measure rather than as claims:

| | TFG Monte Carlo | SMG |
|---|---|---|
| spread used | **isotropic** $\sigma^2I$, $\sigma$ a tuned hyperparameter on a schedule | the model's own $\Sigma_t=\frac{\sigma_t^2}{\alpha_t}\partial\hat x_1/\partial x_t$ — **anisotropic, state-dependent, read off the denoiser** |
| estimator | $K$-sample Monte Carlo ($K=4$ in their molecule script) | deterministic, one JVP |
| what is averaged | the **likelihood**, in log space | the property **moments**, which then form the likelihood |
| cost per guided step | $K$ guide forwards + $K$ backwards | 1 JVP + $n_{\text{probe}}$ HVPs |

The sharp question this makes possible is better than the one we had:
**does the model's real anisotropic covariance beat a tuned isotropic $\sigma$ at
matched cost?** That is a substantive comparison; "SMG beats plug-in" was close to a
strawman.

Two consequences for the claim:

1. **The novelty statement narrows again.** "Correcting the mean/property mismatch" is
   occupied. What remains is the *particular* correction — a deterministic second-order
   term using the denoiser's own covariance — and whether it is better or cheaper than
   sampling with a tuned isotropic surrogate.
2. **If the tuned $\sigma$ matches SMG, that is a real negative result** and should be
   reported as one. It would say the anisotropy does not matter at this scale, which is
   worth knowing.

Implemented as `mode="tfg_mc"` in `proj1/src/guidance.py`, with `n_mc` and `sigma_mc`
exposed so the comparison can be run at matched cost and $\sigma$ swept, since their
schedule tunes it. Measured cost per guided field evaluation, from the harness:

| arm | generator | guide |
|---|---|---|
| plug-in | 2 fwd, 1 VJP | 1 fwd, 1 bwd |
| SMG (4 probes) | 2 fwd, 1 VJP, 5 JVP | 5 fwd, 1 bwd, 4 HVP |
| **TFG MC (K=4)** | 2 fwd, 1 VJP | **5 fwd, 5 bwd** |

Note neither is obviously cheaper: SMG moves work onto forward-mode derivatives of the
*generator*, TFG onto repeated backward passes through the *guide*. Which wins depends
on their relative size — and ours differ by 4x — so this has to be measured in wall
time, not counted.

### 2.56 Residual calibration (Haimo, v1): the same gap, learned instead of computed

**Added 19 Sep.** A team member's independent study, *Residual Calibration for Property Guidance:
A Two-Modality Experimental Study*, attacks **the same quantity SMG does** and must be treated as a
competitor rather than a side project. Its own abstract states the premise in our words:
"Nonlinear properties of denoised means can misestimate quantities needed for guidance."

It uses the identical flow-matching convention — $X_t=tX+(1-t)Z$, $m_\theta=x+(1-t)v_\theta$ — so
the constructions line up term by term.

**The method.** Replace the plug-in mean with a *learned* correction:

$$\hat\mu_t=f(m_\theta)+C_\phi(x,m_\theta,t),\qquad
C_\phi=(1-t)\,\phi(x,m_\theta,t)^\top\beta$$

where $\beta$ holds **28 coefficients** fitted by ridge regression against $f(X)-f(m_\theta)$ —
exactly the Jensen gap — on 768 examples at six noise levels (4,608 corruption pairs). Features are
time, RMS values, means, one state/velocity cosine, $f(m)$, and time interactions. The $(1-t)$
factor forces the correction to vanish at the clean endpoint. The likelihood is then Gaussian with
variance $s^2+V(t)$, where $V(t)$ is interpolated out-of-fold **residual MSE** — explicitly *not* a
state-specific posterior variance.

**Four ways to handle one defect.** This makes the comparison structure much clearer than it was:

| | corrected mean | variance | training? | Hessian? | cost per guided step |
|---|---|---|---|---|---|
| plug-in | $f(m)$ | $s^2$ | no | no | 1 guide fwd+bwd |
| **Haimo v1** | $f(m)+C_\phi$, **learned** | $V(t)$, **time-only** | **yes**, 4,608 pairs | **no** | ~free (28 dot products) |
| **TFG MC** | — (averages the *likelihood*) | implicit, isotropic tuned $\sigma$ | no | no | $K$ guide fwd+bwd |
| **SMG** | $f(m)+\tfrac12\operatorname{tr}(H\Sigma_t)$, **computed** | $g^\top\Sigma_t g$, **state-specific** | no | **yes** | 1 JVP + probes |

Read across the rows: SMG's distinctive claims are that the correction is **computed from the
denoiser's own covariance rather than fitted**, and that the variance is **state-specific rather
than a function of $t$ alone**. Haimo's $V(t)$ is precisely the weaker form of what $g^\top\Sigma_t g$
provides, which makes the pair a clean ablation of that one axis.

It is also **cheaper than SMG** — no Hessian, no JVP — at the price of needing supervised fitting,
so it is not training-free. That trade is the interesting comparison, not who wins on MAE alone.

**Its reported results, stated as it states them.** Against a tuned plug-in: 82.5% target-error
reduction on clipped images, 50.6% on decoded text representations. But "a direct head wins on
text, image postprocessing is competitive at half the latency, and fluent controlled text is not
demonstrated," and the paper declines to claim originality for the combination. The honest reading
is that a learned correction helps, and that a *direct* learned predictor of $f(X)$ sometimes helps
more — which is a control we should run too.

**Two practical details worth importing.** It activates guidance only on $t\in[0.2,0.95]$ — far
later than our $t_{\min}=0.05$, and independent evidence that guiding early is the problem we
measured. And it caps the increment with $\min(1,\mathrm{RMS}(v_\theta)/\mathrm{RMS}(d))$, the same
velocity-relative trust region we adopted after the unclipped field measured 1.6 million times the
base velocity.

**Scope.** Demonstrated on CIFAR-10 and continuous WikiText-2, not on molecules or the simplex.
Porting it to QM9 means implementing the 28-feature head and fitting it on our generator — which
requires the base model to be trained first, so it is downstream of the current blocker.

### 2.6 Novelty statement to use for now

**Candidate:** a practical nonlinear-property mean correction in moment-based guidance, evaluated
against plug-in, variance-only, quadratic-energy and Monte Carlo alternatives at matched cost --
including **TFG's own Monte Carlo smoothing (2.55)** and **residual calibration (2.56)**, both of
which attack the same defect. Those are the comparators that matter; beating plain plug-in is not
a result. What survives as distinctive is that the correction is *computed from the denoiser's own
covariance rather than fitted or sampled*, and that the variance is *state-specific rather than a
function of t alone*.
Do not claim invention of covariance-aware guidance, second-order guidance, the Jacobian identity,
or Heun. Do not infer novelty from different titles or notation. The formula audit above is bounded,
not an exhaustive novelty clearance.

The fact that $c=0$ for affine properties is an immediate consequence of $H=0$, not a standalone
innovation. Affine unit consistency and rejection sampling are standard checks. A useful
evaluation contribution would require a demonstrated, informative protocol and results, not
just naming these checks. The preliminary mean-versus-variance result must be repeated against
the same soft likelihood targeted by the sampler before it supports a performance claim.

## 3. The Heun formula to implement

Let the **full guided field** be $F(x,t)$ and $h=t_{n+1}-t_n$ be the signed time increment.

**Euler:**

$$x_{n+1}=x_n+hF(x_n,t_n).$$

**Explicit Heun (RK2):**

$$\boxed{\begin{aligned}
k_1&=F(x_n,t_n),\\
\tilde x&=x_n+hk_1,\\
k_2&=F(\tilde x,t_n+h),\\
x_{n+1}&=x_n+\tfrac h2(k_1+k_2).
\end{aligned}}$$

**Derivation.** From $x'=F(x,t)$, the chain rule gives

$$x''=\partial_tF+(\partial_xF)F.$$

Hence the exact trajectory has the local expansion

$$x(t+h)=x+hF+\tfrac{h^2}{2}\big(\partial_tF+(\partial_xF)F\big)+O(h^3).$$

Expanding $k_2$ at $(x_n,t_n)$ gives

$$k_2=F+h\big(\partial_tF+(\partial_xF)F\big)+O(h^2).$$

Substitution into Heun reproduces that trajectory expansion through $h^2$ without explicitly
forming $\partial_xF$ or $\partial_tF$. Under the usual smoothness/stability assumptions,
Heun has local error $O(|h|^3)$ and global error $O(h^2)$, versus Euler's $O(|h|^2)$ and $O(h)$.
This is accuracy relative to the chosen field, not a proof that the field targets the right distribution.

### 3.1 Full field for our independent-Gaussian linear flow matching

Use generation time $t:0\to1$ and $x_t=tX+(1-t)\varepsilon$ with independent noise. Then

$$m=x_t+(1-t)v_\theta(x_t,t),\quad
\Sigma\approx\frac{(1-t)^2}{t}\partial_{x_t}m,\quad
\boxed{F_{\rm FM}(x_t,t)=v_\theta(x_t,t)+w(t)\frac{1-t}{t}G(x_t,t).}$$

Use either $G_{\rm plug}$ or $G_{\rm SMG}$. The likelihood-defined comparison uses $w=1$;
tuned guidance weights are separate arms. The identity requires independent Gaussian corruption;
an optimal-transport coupling does not automatically satisfy that assumption.

At stage 2, recompute the generator, $m,J,\Sigma,f,g,H,c,v_f$, and all time-dependent schedules
at $(\tilde x,t+h)$. Holding the guidance fixed while correcting only the base velocity is a
different method and is not Heun applied to the full guided ODE. The existing stop-gradient
approximation inside $G_{\rm SMG}$ does not authorize reusing its values across stages.

### 3.2 Diffusion: use its probability-flow ODE

Define forward noising time $\tau:0\to T$ (clean to noisy), forward drift $b$, and scalar
noise amplitude $\kappa$:

$$dX=b(X,\tau)d\tau+\kappa(\tau)dW_\tau.$$

With the score estimate $s_\theta$ and guidance in score units $G$, integrate

$$\boxed{F_{\rm diff}(x,\tau)=b(x,\tau)-\tfrac12\kappa(\tau)^2
\big(s_\theta(x,\tau)+G(x,\tau)\big)}$$

from large to small $\tau$, so **$h<0$** in the same Heun equations. For VP diffusion,
$b=-\beta x/2$ and $\kappa^2=\beta$. The probability-flow conversion is established in
[Song et al.](https://arxiv.org/abs/2011.13456).
Do not substitute the reverse-SDE coefficient $\kappa^2$ for the ODE coefficient
$\kappa^2/2$, or add fresh Gaussian noise to these deterministic Heun stages.
This specifies an ODE comparison, not a new DDPM stochastic update.

### 3.3 Endpoint, randomness and cost conventions

- Never evaluate formulas with division by zero at $t=0$ or $\sigma=0$. Use a validated
  analytic endpoint limit, or a shared terminal Euler/denoising policy and record it.
  If the second stage is skipped on the final interval, $N$ steps cost $2N-1$ field evaluations.
- For a smoothness/convergence check, first use a fixed interval strictly inside the endpoints.
  End-to-end quality uses one declared endpoint policy for every arm; report cutoff sensitivity.
- For exact-mixture toy tests starting at $t_{\min}>0$, sample the actual marginal
  $t_{\min}X+(1-t_{\min})\varepsilon$. Pure standard Gaussian noise is not exactly that marginal.
- Use exact traces on the tiny toy. With random trace probes on real models, hold a seeded probe
  set fixed along each trajectory for the deterministic solver comparison. A fresh random field
  at every stage does not satisfy the ordinary deterministic convergence argument.
- Report field evaluations, generator/guide forward calls, JVP/VJP/HVP work, wall time and memory.
  Two Heun stages are not automatically twice the total cost, and equal NFE is not equal cost
  across different guidance estimators.

## 4. Required experiment and interpretation

Run a crossed design on **each model family**:

| Guidance | Euler | Heun |
|---|---|---|
| Unguided | Control | Solver-only control |
| Plug-in | Baseline | Stronger solver baseline |
| SMG, mean correction only | Ablation | Ablation |
| SMG, variance correction only | Ablation | Ablation |
| SMG, both | Candidate | Candidate plus established solver |

Minimum comparison: the plug-in/SMG-both 2-by-2 grid, plus unguided controls. Add the
quadratic-energy and Monte Carlo competitors under the selected common solver for the
novelty/efficiency comparison; implementing Heun alone does not address those competitors.

1. **Formula checks:** analytic autonomous, explicitly time-dependent, and coupled ODEs;
   verify convergence order, time direction, stage refresh and actual field-call counts.
2. **Same-field convergence:** compare paired trajectories to a refined reference for that
   exact guidance field; refine the reference again to show numerical convergence. This isolates
   solver error. It does not estimate error against the desired conditional distribution.
3. **Equal field evaluations:** use budgets such as 40, 80, 160 and 320. Match actual calls,
   including terminal fallbacks. Use nested schedules with the same endpoints and initial states.
4. **Equal measured wall time:** measure full guidance cost, adjust step counts, and report
   property error, in-band fraction, validity/stability, mode or structural coverage, and useful
   distinct outputs per attempt and per GPU-hour. Use paired seeds and independent batches.
5. **Consistent distribution reference:** for the Gaussian property likelihood, accept base-model
   samples with probability $\exp(-(f(X)-y)^2/(2s^2))$. A hard-band reference is a different target.
   On the exact toy use the known base distribution; on molecules use the declared base sampler
   and state its own numerical approximation. Keep in-band fraction as a separate application metric.

If SMG improves over plug-in under both converged solvers, that supports an estimator effect.
If the improvement disappears with Heun, solver error may have driven the Euler result.
Heun helping both arms is a solver benefit, not SMG novelty. Different formulas alone do not
establish a useful research contribution.

## 5. Verification status

`audit/heun_formula_checks.py` checks analytic ODEs and the scalar formula counterexample in
section 2.4. Its output is separate from the historical v7 results. These checks do not implement
a molecular sampler and do not run the crossed SMG experiment. M-4 remains planned until those
sampler comparisons are actually implemented and run.

**Checked 18 Sep:** the analytic audit passes. With 40, 80 and 160 steps, observed convergence
orders are approximately 0.98–0.99 for Euler and 1.99–2.04 for Heun on autonomous and explicitly
time-dependent equations. Coupled-state, decreasing-time, stage-refresh and field-call checks
pass. The valid-state guidance counterexample returns SMG $=-1/14$ and displayed GGDOpt $=0$.
Results are recorded in `audit/heun_formula_results.json`.
