# SMG — why innovative?

## Second-Moment Guidance: explained, proved where possible, and compared with prior work

**Prepared 18 September 2026.** For a reader with introductory linear algebra, calculus, and probability.

**The answer we can currently defend:** SMG is a proposed adaptation of established mathematics. Its candidate contribution is a particular way to correct nonlinear-property guidance using posterior uncertainty. Neither its research novelty nor its practical advantage has been established. The title is a question to investigate, not a claim that the answer is already yes.

This document is the consolidated explanation of the v7 proposal. It distinguishes exact identities, controlled approximations, practical design choices, and untested hypotheses. Where older brainstorming notes make stronger claims about approximation order, novelty, or experimental success, use the qualifications here.

Related project files: [Innovation v7](SMG_FULL_MATH_AND_PROOFS.md), [formula-level prior-work audit and Heun specification](SMG_PRIOR_WORK_AUDIT.md), and [experiment plan](../../archive/FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md).

### Reading map

1. [The big picture and the name](#1-the-big-picture-and-the-name)
2. [Notation and the exact problem](#2-notation-and-the-exact-problem)
3. [A numerical example](#3-a-numerical-example)
4. [Proof: covariance from the denoiser](#4-proof-covariance-from-the-denoiser)
5. [Proofs: the mean and variance of the property](#5-proofs-the-mean-and-variance-of-the-property)
6. [From property moments to guidance](#6-from-property-moments-to-guidance)
7. [Why these approximation orders](#7-why-these-approximation-orders)
8. [What higher orders would change](#8-what-higher-orders-would-change)
9. [Implementation and computational cost](#9-implementation-and-computational-cost)
10. [Diffusion, flow matching, and Heun](#10-diffusion-flow-matching-and-heun)
11. [Comparison with previous methods](#11-comparison-with-previous-methods)
12. [Exactly what could count as a contribution](#12-exactly-what-could-count-as-a-contribution)
13. [Assumptions, failure cases, and proof limits](#13-assumptions-failure-cases-and-proof-limits)
14. [Experiments needed to support the claim](#14-experiments-needed-to-support-the-claim)
15. [Current status and defensible wording](#15-current-status-and-defensible-wording)
16. [Can SMG be incorporated into training?](#16-can-smg-be-incorporated-into-training)

## 1. The big picture and the name

A pretrained generator knows how to produce plausible samples. Guidance tries to make those samples satisfy an additional request, such as a molecule having a desired dipole moment.

During generation, we do not yet know the final clean sample. We have a noisy intermediate state and a prediction of the clean sample. Ordinary plug-in guidance evaluates the requested property on that prediction and uses the resulting error to steer generation.

The difficulty is that a prediction of the average sample is not necessarily a prediction of the average property:

$$
f\big(\mathbb E[X\mid x_t]\big)
\ne
\mathbb E[f(X)\mid x_t].
$$

For a nonlinear property, the uncertainty around the predicted sample can change the expected property. SMG attempts to account for that difference before calculating the guidance update.

**SMG means Second-Moment Guidance.** The name refers to using the posterior covariance, a second central moment, in addition to the posterior mean. It does not mean that every part of the algorithm is a second-order approximation.

For clarity:

- A first moment is an average, such as $\mathbb E[X]$.
- A second central moment describes spread: $\Sigma=\mathbb E[(X-m)(X-m)^\top]$.
- The raw second moment is $\mathbb E[XX^\top]=\Sigma+mm^\top$.
- A first- or second-order Taylor approximation describes how many derivative terms we retain.
- A first- or second-order ODE solver describes numerical error as the integration step shrinks.

These are different uses of the word “order.”

**The proposed SMG runs at inference time, during sampling.** It is not a new SGD or Adam training optimizer, and its name is not SGD. The generator and any learned property predictor keep their trained parameters fixed while SMG steers the sample.

**The aim:** improve property targeting while preserving plausible and diverse outputs at an acceptable total cost. A smaller property error alone is insufficient if validity collapses, all samples become nearly identical, or runtime increases excessively.

### 1.1 SMG runs at inference time

Training learns the model's parameters from examples. Inference uses the trained model to generate a new sample. In diffusion and flow matching, generating one sample usually involves many successive sampling steps, so inference itself is an iterative process.

| Stage | What happens | Are model parameters updated? |
|---|---|---|
| Train the generator | Learn a denoiser, score, or velocity from training data | Yes, generator parameters |
| Prepare the property function | Use an analytic property, an existing predictor, or train a predictor if needed | Only if a predictor needs training |
| Generate with SMG | Evaluate the frozen models, calculate the correction, and advance the sample | No; the sample state changes |

The proposed workflow is: **train or obtain the models → choose a target property → generate samples with SMG guidance**.

At each sampling step where guidance is enabled, SMG:

1. Uses the generator to predict the clean-sample mean.
2. Estimates posterior uncertainty and the property's derivatives.
3. Calculates the corrected property mean and variance.
4. Converts the resulting guidance score into the appropriate velocity correction.
5. Updates the current sample and repeats.

Changing the requested target changes these sampling calculations. It does not require retraining the generator for each target.

**Why do we use gradients if this is inference?** A gradient is a mathematical derivative; it is not synonymous with training. During training, derivatives with respect to model parameters tell an optimizer how to update those parameters. During SMG sampling, derivatives with respect to the sample or predicted clean input tell the sampler how to steer that sample. The model parameters stay fixed.

For example, the denoiser Jacobian $J=\partial m/\partial x_t$ differentiates the network's output with respect to its input $x_t$, not with respect to its weights. Automatic differentiation may perform backward calculations during inference to obtain these derivatives. That still does not constitute model training. An implementation must allow the required input derivatives even while freezing the model parameters.

The variable $t$ denotes position along the sampling path or noise schedule, not a training epoch. Heun's second evaluation is also an inference-time calculation.

Calling SMG “training-free guidance” means that the guidance rule does not introduce an additional model-training stage. It does not mean that the base generator was never trained, that every property function is available without training, or that inference is free of gradient computation. SMG adds sampling-time computation and potentially memory cost; those costs must be measured.

## 2. Notation and the exact problem

### 2.1 The quantities we use

Fix a sampling time $t$. Unless otherwise stated, every expectation and covariance below is conditional on the current state $x_t$.

| Symbol | Meaning |
|---|---|
| $X\in\mathbb R^d$ | The unknown clean sample |
| $x_t\in\mathbb R^d$ | The current noisy/intermediate sample |
| $m=m(x_t,t)$ | Posterior mean $\mathbb E[X\mid x_t]$ |
| $\delta=X-m$ | Deviation from that mean |
| $\Sigma$ | Posterior covariance $\mathbb E[\delta\delta^\top\mid x_t]$ |
| $f:\mathbb R^d\to\mathbb R$ | Scalar property to control |
| $y$ | Desired property value |
| $s>0$ | Width of the desired property likelihood; smaller means stricter targeting |
| $g=\nabla f(m)$ | Property gradient at the predicted clean sample |
| $H=\nabla^2 f(m)$ | Property Hessian at that sample |
| $J=\partial m/\partial x_t$ | Denoiser Jacobian, with $J_{ij}=\partial m_i/\partial x_{t,j}$ |
| $\operatorname{tr}(A)$ | Sum of the diagonal entries of a square matrix |

A gradient tells us the direction of local change. A Hessian records how that gradient changes; it measures curvature. A Jacobian is the matrix of first derivatives of a vector-valued function.

### 2.2 What distribution are we trying to sample?

Let $p_0(X)$ denote the generator's clean-data distribution. Use a Gaussian property likelihood

$$
\ell_y(X)=\mathcal N(y;f(X),s^2)
=\frac{1}{\sqrt{2\pi s^2}}
\exp\left[-\frac{(y-f(X))^2}{2s^2}\right].
$$

The intended clean distribution is

$$
p_{\mathrm{target}}(X)
=\frac{p_0(X)\ell_y(X)}{\int p_0(u)\ell_y(u)\,du}.
\tag{1}
$$

Thus, plausible samples with properties closer to $y$ receive more probability. This is a soft target. It differs from conditioning on the hard event $|f(X)-y|\leq\Delta$.

At intermediate time, the likelihood we need is

$$
Z_t(x_t)=p(y\mid x_t)
=\mathbb E[\ell_y(X)\mid x_t].
\tag{2}
$$

By Bayes' rule,

$$
\nabla_{x_t}\log p_t(x_t\mid y)
=\nabla_{x_t}\log p_t(x_t)
+\underbrace{\nabla_{x_t}\log Z_t(x_t)}_{G_{\mathrm{exact}}}.
\tag{3}
$$

The first term is the unconditional score. A score is simply the gradient of a log probability density. The second term is the exact guidance score. Its conversion into a sampling velocity depends on the model and schedule; Section 10 explains that conversion.

### 2.3 What plug-in guidance approximates

The simplest approximation replaces the uncertain $X$ with its mean:

$$
Z_t(x_t)\approx\ell_y(m).
$$

Applying the chain rule gives the comparison field

$$
G_{\mathrm{plug}}
=\frac{y-f(m)}{s^2}J^\top g.
\tag{4}
$$

This is the Gaussian-likelihood plug-in template. Published implementations may additionally normalize gradients or tune schedules, so equation (4) is not a claim that every named guidance algorithm has exactly this update.

**Important distinction:** the exact target requires the expectation of a likelihood, $\mathbb E[\ell_y(X)]$. Correcting $\mathbb E[f(X)]$ addresses part of that calculation. It does not, by itself, compute the exact likelihood.

## 3. A numerical example

Suppose $X$ is scalar, its current conditional mean is $m=2$, and its variance is $q=1$. Choose $f(X)=X^2$ and target $y=4$.

Evaluating the property at the mean gives

$$f(m)=2^2=4.$$

But the expected property is

$$
\mathbb E[X^2]=(\mathbb E[X])^2+\operatorname{Var}(X)=4+1=5.
\tag{5}
$$

This identity holds for any distribution with those moments. Gaussianity is unnecessary.

| Calculation | Predicted mean property | Residual $y-\text{prediction}$ |
|---|---:|---:|
| Plug-in | $4$ | $0$ |
| Add variance to the guidance denominator only | $4$ | $0$ |
| Correct the property mean | $5$ | $-1$ |

A denominator correction cannot change a zero numerator. A mean correction can.

For this property, $f'(m)=4$ and $f''(m)=2$. The proposed estimates are

$$c=\tfrac12 f''(m)q=1,\qquad v_f=f'(m)^2q=16.$$

With $s=1$, SMG's multiplier of the common direction $J^\top g$ is $-1/17$, whereas the plug-in multiplier is zero.

This example proves that the two formulas can behave differently. It does not prove that SMG's update equals the exact likelihood gradient. If $X$ is Gaussian here, the exact property variance is $18$, not $16$; Section 8 derives the missing term.

## 4. Proof: covariance from the denoiser

Assume an independent isotropic Gaussian corruption model:

$$
x_t=\alpha_t X+\sigma_t\varepsilon,
\qquad \varepsilon\sim\mathcal N(0,I),\quad \varepsilon\perp X.
\tag{6}
$$

For the proof, fix $t$, abbreviate $\alpha_t,\sigma_t$ by $\alpha,\sigma$, and assume $\alpha>0$, $\sigma>0$, finite required moments, and enough regularity to differentiate inside integrals.

We will show

$$
\boxed{J=\frac{\alpha}{\sigma^2}\Sigma,
\qquad \Sigma=\frac{\sigma^2}{\alpha}J.}
\tag{7}
$$

This is established second-order Tweedie mathematics, used in covariance-aware diffusion methods and explicitly stated for the relevant Gaussian-path flow setting in [On the Guidance of Flow Matching, Proposition 3.4](https://arxiv.org/html/2502.02150v2). The following is a direct calculus derivation.

**Step 1: differentiate the observation density.**

For a possible clean value $u$,

$$
p(x_t\mid X=u)\propto
\exp\left[-\frac{\|x_t-\alpha u\|^2}{2\sigma^2}\right],
$$

so

$$
\partial_{x_{t,j}}\log p(x_t\mid X=u)
=-\frac{x_{t,j}-\alpha u_j}{\sigma^2}.
$$

**Step 2: differentiate the marginal density.**

Since $p_t(x_t)=\int p(x_t\mid u)p_0(u)\,du$,

$$
\partial_{x_{t,j}}\log p_t(x_t)
=-\frac{x_{t,j}-\alpha m_j}{\sigma^2}.
\tag{8}
$$

The integral replaces $u_j$ with its conditional average $m_j$.

**Step 3: differentiate the posterior density using Bayes' rule.**

Subtracting the marginal log derivative from the observation log derivative gives

$$
\partial_{x_{t,j}}\log p(u\mid x_t)
=\frac{\alpha}{\sigma^2}(u_j-m_j).
$$

Consequently,

$$
\partial_{x_{t,j}}p(u\mid x_t)
=p(u\mid x_t)\frac{\alpha}{\sigma^2}(u_j-m_j).
$$

**Step 4: differentiate the posterior mean.**

$$
\begin{aligned}
J_{ij}
&=\partial_{x_{t,j}}\int u_i p(u\mid x_t)\,du\\
&=\frac{\alpha}{\sigma^2}
\mathbb E[X_i(X_j-m_j)\mid x_t]\\
&=\frac{\alpha}{\sigma^2}
\left(\mathbb E[X_iX_j\mid x_t]-m_i m_j\right)\\
&=\frac{\alpha}{\sigma^2}\Sigma_{ij}.
\end{aligned}
$$

That proves equation (7).

**What was not assumed:** the clean prior or clean posterior need not be Gaussian. The Gaussian assumption concerns the independent corruption noise.

**What changes with a learned model:** a network $m_\theta$ only approximates the posterior mean. Its Jacobian can fail to be symmetric or positive semidefinite, so $(\sigma^2/\alpha)J_\theta$ is an estimated covariance, not a guaranteed valid covariance.

Equation (7) does not automatically apply to arbitrary optimal-transport couplings, discrete paths, or Dirichlet flow matching. Their conditional laws require separate derivations.

## 5. Proofs: the mean and variance of the property

### 5.1 First-order Taylor expansion

Because $\delta=X-m$,

$$\mathbb E[\delta]=0,\qquad \mathbb E[\delta\delta^\top]=\Sigma.$$

Linearize the property:

$$
f(m+\delta)\approx f(m)+g^\top\delta.
\tag{9}
$$

Taking expectations gives

$$
\mathbb E[f(X)]\approx f(m)+g^\top\mathbb E[\delta]=f(m).
\tag{10}
$$

The linear term gives no mean correction because positive and negative deviations cancel on average.

For the variance, the constant disappears:

$$
\begin{aligned}
\operatorname{Var}(f(X))
&\approx\operatorname{Var}(g^\top\delta)\\
&=\mathbb E[(g^\top\delta)^2]\\
&=g^\top\mathbb E[\delta\delta^\top]g\\
&=\boxed{g^\top\Sigma g}.
\end{aligned}
\tag{11}
$$

This is standard first-order uncertainty propagation. It is not our invention.

### 5.2 Second-order Taylor expansion and the mean correction

Include curvature:

$$
f(m+\delta)=f(m)+g^\top\delta
+\tfrac12\delta^\top H\delta+R(\delta).
\tag{12}
$$

The expectation of the quadratic term is

$$
\begin{aligned}
\mathbb E[\delta^\top H\delta]
&=\sum_{i,j}H_{ij}\mathbb E[\delta_i\delta_j]\\
&=\sum_{i,j}H_{ij}\Sigma_{ij}\\
&=\operatorname{tr}(H\Sigma),
\end{aligned}
$$

where the final equality uses covariance symmetry. Therefore,

$$
\boxed{
\mathbb E[f(X)]
=f(m)+\underbrace{\tfrac12\operatorname{tr}(H\Sigma)}_{c}
+\mathbb E[R(\delta)].
}
\tag{13}
$$

This is also established Taylor-expansion mathematics.

If $f$ is affine, $H=0$ and the mean and variance from equations (10)-(11) are exact. If $f$ is quadratic, $R=0$ and equation (13) gives its exact mean under any posterior with finite second moments. Neither statement makes the entire likelihood or sampler exact.

The sign of $c$ depends on $\operatorname{tr}(H\Sigma)$. It cannot generally be inferred from $\operatorname{tr}(H)$ alone when uncertainty is anisotropic.

### 5.3 A meaningful remainder bound

Suppose the third derivative satisfies

$$
|D^3f(z)[u,u,u]|\leq M_3\|u\|^3
$$

along the relevant segments between $m$ and $m+\delta$. Taylor's theorem gives

$$
|R(\delta)|\leq\frac{M_3}{6}\|\delta\|^3,
$$

and hence

$$
\left|\mathbb E[f(X)]-f(m)-c\right|
\leq\frac{M_3}{6}\mathbb E\|\delta\|^3.
\tag{14}
$$

This explains when the correction is useful: the property should be smooth over the posterior's spread, and the third-moment remainder should be small.

To give “order” a precise meaning, consider a family $\delta=\epsilon Z$, where $Z$ has fixed bounded relevant moments and $\epsilon\to0$. Then $\Sigma=\epsilon^2 C$, $c=O(\epsilon^2)$, and the bound is $O(\epsilon^3)$. Under this additional concentration model, it is reasonable to describe the error as order $\|\Sigma\|^{3/2}$.

Small covariance alone does not bound the third moment: rare, large deviations can invalidate that conclusion.

### 5.4 The variance also has a remainder

Write $L=g^\top\delta$ and $U=f(m+\delta)-f(m)-L$. Exactly,

$$
\operatorname{Var}(f(X))
=g^\top\Sigma g+2\operatorname{Cov}(L,U)+\operatorname{Var}(U).
\tag{15}
$$

Cauchy-Schwarz bounds the magnitude of the omitted part by

$$
2\sqrt{\operatorname{Var}(L)\operatorname{Var}(U)}
+\operatorname{Var}(U).
$$

For bounded local derivatives and $\delta=\epsilon Z$ with controlled moments, $L$ is order $\epsilon$ and $U$ is order $\epsilon^2$. The leading variance is order $\epsilon^2$, with a possible order-$\epsilon^3$ correction from asymmetry. Under suitable symmetry and stronger smoothness, that odd-order contribution vanishes.

This is a controlled local approximation, not a universal guarantee across all sampling times.

## 6. From property moments to guidance

Define the estimates

$$
\mu_f=f(m)+c,\qquad
c=\tfrac12\operatorname{tr}(H\Sigma),\qquad
v_f=g^\top\Sigma g.
\tag{16}
$$

### 6.1 The additional distributional approximation

SMG approximates the conditional property distribution by

$$f(X)\mid x_t\ \approx\ \mathcal N(\mu_f,v_f).$$

This is called a Gaussian moment approximation: represent a potentially complicated distribution using a Gaussian with estimated mean and variance.

It is a separate assumption from the Taylor expansions. Even a Gaussian $X$ need not yield a Gaussian $f(X)$; $X^2$ is an immediate counterexample.

Our observation model is $Y=f(X)+\eta$, with independent $\eta\sim\mathcal N(0,s^2)$. Under the Gaussian property approximation, the sum is Gaussian:

$$
\widetilde Z_t(x_t)=\mathcal N(y;\mu_f,V),
\qquad V=s^2+v_f.
\tag{17}
$$

Independence adds the variances. This explains the denominator in the eventual update.

### 6.2 Derive the full gradient of this surrogate

Let $r=y-\mu_f$. Up to a constant independent of $x_t$,

$$
\log\widetilde Z=-\tfrac12\log V-\frac{r^2}{2V}.
$$

Differentiate with respect to its two inputs:

$$
\frac{\partial\log\widetilde Z}{\partial\mu_f}=\frac rV,
\qquad
\frac{\partial\log\widetilde Z}{\partial V}
=\frac{r^2-V}{2V^2}.
$$

Since $\nabla_{x_t}\mu_f=J^\top g+\nabla_{x_t}c$, the chain rule yields

$$
\boxed{
G_{\mathrm{full}}
=\frac rV\left(J^\top g+\nabla_{x_t}c\right)
+\frac{r^2-V}{2V^2}\nabla_{x_t}v_f.
}
\tag{18}
$$

This is the exact gradient of the approximate Gaussian likelihood. It is not necessarily the exact gradient of the original likelihood in equation (2).

### 6.3 The practical v7 choice

Practical SMG holds $c$ and $v_f$ fixed while differentiating locally:

$$
\boxed{
G_{\mathrm{SMG}}
=\frac{y-f(m)-\tfrac12\operatorname{tr}(H\Sigma)}
{s^2+g^\top\Sigma g}\,J^\top g.
}
\tag{19}
$$

The numerical values of $c$ and $v_f$ are still recomputed at each field evaluation. “Hold fixed” refers to the derivative calculation, not to freezing them for the whole trajectory.

This saves differentiation through quantities that can involve second derivatives of the denoiser and third derivatives of the property. However, the omitted terms in equation (18) need not be smaller in uncertainty order than the corrections we retained.

For example, in one dimension with fixed variance $q$ and $m=m(x_t)$,

$$
\partial_{x_t}c=\tfrac12 q f'''(m)m',
\qquad
\partial_{x_t}v_f=2q f'(m)f''(m)m'.
$$

Both are order $q$. Thus, claiming that equation (19) is the complete first-order-in-covariance expansion of the exact guidance would be unjustified.

For one scalar property, equation (19) is a scalar multiple of the same direction $J^\top g$ used by plug-in guidance. The denominator changes strength; the mean correction can also change the sign. The omitted vector terms in equation (18) could change direction in higher dimensions.

### 6.4 The four useful variants

| Variant | Multiplier of $J^\top g$ |
|---|---|
| Plug-in | $(y-f(m))/s^2$ |
| Mean correction only | $(y-f(m)-c)/s^2$ |
| Variance correction only | $(y-f(m))/(s^2+v_f)$ |
| Full practical SMG | $(y-f(m)-c)/(s^2+v_f)$ |

The mean correction asks, “What property value are we predicting on average?” The variance correction asks, “How uncertain is that prediction?” These address different errors and must be tested separately.

## 7. Why these approximation orders

### 7.1 They retain the first nonzero uncertainty effect in each moment

For the mean, the linear term averages to zero. The quadratic term is the first term that can correct $f(m)$.

For the variance, the linear term already fluctuates around zero. Squaring it produces $g^\top\Sigma g$, the leading uncertainty contribution.

Under $\delta=\epsilon Z$:

| Quantity | First useful retained term | Scale |
|---|---|---|
| Mean correction | $\tfrac12\mathbb E[\delta^\top H\delta]$ | $O(\epsilon^2)$ |
| Property variance | $\mathbb E[(g^\top\delta)^2]$ | $O(\epsilon^2)$ |

So “second-order mean, first-order variance” is not an arbitrary mismatch. Both retain leading effects proportional to covariance. This motivates the moment estimates; it does not justify dropping their gradients.

### 7.2 A stronger connection to the likelihood

There is a local reason these two corrections fit together. The following derivation is a Taylor calculation, not a novelty claim.

Let $r_0=y-f(m)$, and keep $s>0$ fixed as uncertainty shrinks. For the clean-sample likelihood $\ell=\ell_y$,

$$
\nabla\log\ell(m)=\frac{r_0}{s^2}g,
\qquad
\nabla^2\log\ell(m)=\frac{r_0}{s^2}H-\frac1{s^2}gg^\top.
$$

Use the identity

$$
\frac{\nabla^2\ell}{\ell}
=\nabla^2\log\ell+
(\nabla\log\ell)(\nabla\log\ell)^\top.
$$

Taylor-expand $\mathbb E[\ell(X)]$ around $m$, then expand its logarithm. With sufficient derivative and moment bounds, $\delta=\epsilon Z$, and fixed $m,y,s$,

$$
\log Z
=\log\ell(m)
+\frac{r_0c}{s^2}
+\frac12\left(\frac{r_0^2}{s^4}-\frac1{s^2}\right)v_f
+O(\epsilon^3).
\tag{20}
$$

For a suitably symmetric shrinking posterior, stronger regularity can improve the remainder. Equation (20) is not uniform for arbitrarily small $s$ or extreme targets.

Now expand the Gaussian surrogate:

$$
-\tfrac12\log(s^2+v_f)-\frac{(r_0-c)^2}{2(s^2+v_f)}.
$$

Because $c,v_f=O(\epsilon^2)$, its expansion through order $\epsilon^2$ is exactly the same expression as equation (20), including the normalization term.

**What this supports:** the moment-based likelihood captures the leading local uncertainty correction under the stated conditions, even if the property is not exactly Gaussian.

**What it does not support:** differentiating a pointwise remainder requires additional uniform derivative control. More importantly, practical SMG then drops derivative terms. Therefore equation (20) is not a proof that practical SMG is an equally accurate guidance score.

### 7.3 Why not use zero order, or only first order everywhere?

Zero-order property approximation treats $f(X)$ as constant at $f(m)$ and loses uncertainty.

First order produces a useful variance but leaves the mean at $f(m)$. That may be sufficient if curvature is negligible. It misses the discrepancy in Section 3 when curvature matters.

The proposed extra mean term is the smallest Taylor extension that addresses that discrepancy. “Smallest useful extension” is a computational motivation, not proof of originality.

## 8. What higher orders would change

### 8.1 Second-order property expansion gives a richer variance

Start from the quadratic approximation

$$Q=f(m)+g^\top\delta+\tfrac12\delta^\top H\delta.$$

For scalar $X$, write $a=f'(m)$, $b=f''(m)$, $q=\mathbb E[\delta^2]$, $\mu_3=\mathbb E[\delta^3]$, and $\mu_4=\mathbb E[\delta^4]$. Expanding the variance gives

$$
\begin{aligned}
\operatorname{Var}(Q)
&=\operatorname{Var}(a\delta+\tfrac12 b\delta^2)\\
&=a^2q+ab\mu_3+\tfrac14b^2(\mu_4-q^2).
\end{aligned}
\tag{21}
$$

This shows the extra information required: third and fourth central moments. Mean and covariance alone do not determine this variance for a general posterior.

If $\delta\sim\mathcal N(0,q)$, then $\mu_3=0$ and $\mu_4=3q^2$, yielding

$$
\operatorname{Var}(Q)=f'(m)^2q+\tfrac12 f''(m)^2q^2.
\tag{22}
$$

In several dimensions, under a centered Gaussian posterior approximation,

$$
\boxed{v_{\mathrm{quad}}=g^\top\Sigma g+
\tfrac12\operatorname{tr}(H\Sigma H\Sigma).}
\tag{23}
$$

**Proof of the vector formula.** The linear–quadratic covariance is zero because centered Gaussian third moments vanish. The Gaussian fourth-moment identity is

$$
\mathbb E[\delta_i\delta_j\delta_k\delta_l]
=\Sigma_{ij}\Sigma_{kl}+\Sigma_{ik}\Sigma_{jl}+\Sigma_{il}\Sigma_{jk}.
$$

Substituting this into $\mathbb E[(\delta^\top H\delta)^2]$ gives

$$
\mathbb E[(\delta^\top H\delta)^2]
=\operatorname{tr}(H\Sigma)^2+2\operatorname{tr}(H\Sigma H\Sigma).
$$

Subtract the squared mean and multiply by $1/4$ to obtain the extra term in equation (23).

### 8.2 When the extra term matters

For Gaussian $X$ and $f(X)=X^2$,

$$
\mathbb E[f(X)]=m^2+q,
\qquad
\operatorname{Var}(f(X))=4m^2q+2q^2.
\tag{24}
$$

SMG's mean estimate is exact here, while its leading variance retains only $4m^2q$.

At $m=0$, the leading variance is zero, but the true variance is $2q^2$. This is a clear case where the richer variance estimate is preferable.

Nevertheless, practical SMG also has $g=2m=0$ at this point, so its guidance direction is zero. Fixing only the denominator cannot create a missing direction. A correct variance estimate and a useful guidance gradient are separate questions.

### 8.3 A subtlety: quadratic variance is not the full next-order variance for arbitrary properties

For $\delta\sim\mathcal N(0,q)$ and a sufficiently smooth scalar $f$, write

$$
f(m+\delta)=f(m)+a\delta+\tfrac12 b\delta^2+\tfrac16 d\delta^3+\cdots,
$$

where $d=f'''(m)$. Through order $q^2$,

$$
\boxed{
\operatorname{Var}(f(X))
=a^2q+\left(\tfrac12b^2+ad\right)q^2+o(q^2).
}
\tag{25}
$$

The missing $adq^2$ comes from twice the covariance of the linear and cubic terms:

$$
2\mathbb E[(a\delta)(d\delta^3/6)]
=\frac{ad}{3}\mathbb E[\delta^4]
=adq^2.
$$

Thus equation (23) is the exact variance of the quadratic Taylor polynomial under Gaussian uncertainty. It is not the complete order-$\Sigma^2$ expansion of the variance of every nonlinear property.

### 8.4 Third and fourth order for the mean

In one dimension,

$$
\mathbb E[f(X)]
\approx f(m)+\tfrac12 f''(m)q
+\tfrac16 f'''(m)\mu_3
+\tfrac1{24}f''''(m)\mu_4.
\tag{26}
$$

A skewed posterior can make the third-order term important. For a centered Gaussian posterior, $\mu_3=0$, and the fourth-order mean term becomes $f''''(m)q^2/8$.

Higher moments are not free. In the Gaussian observation channel, differentiating posterior moments can expose higher cumulants. For example, direct differentiation as in Section 4 gives

$$
\partial_{x_{t,k}}\Sigma_{ij}
=\frac{\alpha}{\sigma^2}
\mathbb E[\delta_i\delta_j\delta_k\mid x_t].
\tag{27}
$$

Combining this with equation (7), third central moments involve second derivatives of the exact denoiser. Fourth cumulants involve another derivative. This is mathematically possible but can be expensive and sensitive to network derivative errors.

### 8.5 Why stop here as a starting design?

| Option | What it adds | Main tradeoff |
|---|---|---|
| Plug-in | No uncertainty correction | Cheap, but loses spread and nonlinear mean shift |
| Variance-only | Leading property uncertainty | Established idea; misses the mean shift |
| Practical SMG | Leading mean shift and variance | Requires curvature computation and drops some gradients |
| Quadratic variance | Extra curvature-driven spread | Additional contractions; Gaussian closure or higher moments |
| Full surrogate gradient | Restores equation (18)'s terms | Higher derivatives and greater differentiation cost |
| Third/fourth-order moments | Skewness and higher local effects | More moments, derivatives, estimation error, and complexity |
| Monte Carlo marginalization | Evaluates the function on multiple possible clean outcomes | Sampling cost; depends on the quality of those posterior draws |

There is no theorem that the chosen order is optimal. A local Taylor series need not improve monotonically when uncertainty is large or the property varies sharply. More accurate derivatives of an inaccurate learned model can also be unhelpful.

The sensible reason to begin with practical SMG is that it is a small, interpretable extension using the first two posterior moments. The sensible reason to keep higher-order variants is that explicit failure cases already show where they may help. The experiment, not the order label, should decide.

## 9. Implementation and computational cost

### 9.1 Avoid materializing large matrices

We do not necessarily need to store the full $d\times d$ matrices $J$, $H$, or $\Sigma$.

A Jacobian-vector product computes $Jz$ directly. A vector-Jacobian product computes $J^\top z$. A Hessian-vector product computes $Hz$. Automatic differentiation can calculate these actions without constructing every matrix entry.

Let $k=\sigma_t^2/\alpha_t$. Under equation (7),

$$
v_f=k\,g^\top Jg
=k\,(J^\top g)^\top g.
\tag{28}
$$

Thus the common guidance direction $J^\top g$ can also supply the scalar variance estimate by a dot product. This makes the variance term relatively inexpensive in this formulation. It does not make the curvature correction free.

For the mean correction,

$$
c=\frac{k}{2}\operatorname{tr}(HJ).
$$

If a random probe $z$ satisfies $\mathbb E[zz^\top]=I$, then

$$
\mathbb E[z^\top HJz]
=\operatorname{tr}\big(HJ\,\mathbb E[zz^\top]\big)
=\operatorname{tr}(HJ).
$$

Using $K$ probes gives

$$
\widehat c=\frac{k}{2K}\sum_{r=1}^K z_r^\top H(Jz_r).
\tag{29}
$$

This trace estimator is unbiased for the supplied matrices. It does not remove network error or Taylor error, and a finite number of probes introduces estimation noise. “One extra JVP” is not a complete runtime accounting: the computation also requires property derivatives, Hessian actions, and possibly several probes.

Analytic quadratic properties can be particularly convenient because their Hessians are known. A learned property predictor with nonsmooth activations can be problematic: a local Hessian may vanish within a linear region while uncertainty crosses a kink. The smooth Taylor analysis then does not describe the whole posterior region.

### 9.2 One evaluation of the practical field

1. Evaluate the generator and obtain the clean-mean estimate $m$.
2. Evaluate the property $f(m)$ and gradient $g$.
3. Compute the common guidance direction $J^\top g$.
4. Estimate $v_f$ and $c$ using the selected covariance and trace scheme.
5. Form the residual $r=y-f(m)-c$ and total variance $V=s^2+v_f$.
6. Return $G_{\mathrm{SMG}}=(r/V)J^\top g$.
7. Convert $G_{\mathrm{SMG}}$ into a velocity correction using the model's schedule.

With exact covariance, $v_f\geq0$ and $V>0$. If a learned Jacobian produces negative $v_f$, this is evidence that the covariance approximation is failing. Symmetrization, clipping, or damping can be practical repairs, but must be specified and ablated because they change the estimator.

The runtime comparison must count generator evaluations, property evaluations, derivative work, memory, and wall-clock time. A small symbolic formula does not establish a cheap implementation.

## 10. Diffusion, flow matching, and Heun

### 10.1 Where the shared identity applies

The covariance proof depends on the conditional Gaussian channel in equation (6). It does not depend on calling the trained model “diffusion” or “flow matching.”

A standard diffusion marginal has the form $\alpha_tX+\sigma_t\varepsilon$. An independent linear flow path can have the form

$$
x_t=tX+(1-t)\varepsilon,\qquad t:0\longrightarrow1.
\tag{30}
$$

Both satisfy the required observation model at interior times. A general flow-matching construction need not do so.

### 10.2 Derive the field for independent linear flow matching

Differentiating equation (30) along a conditional path gives $X-\varepsilon$. Using $\varepsilon=(x_t-tX)/(1-t)$ and averaging conditionally,

$$
v(x_t,t)=\frac{m-x_t}{1-t},
\qquad
m=x_t+(1-t)v(x_t,t).
\tag{31}
$$

Let $m_y=\mathbb E[X\mid x_t,y]$. The desired conditional velocity is $(m_y-x_t)/(1-t)$.

Applying equation (8) with and without the property condition gives

$$
G_{\mathrm{exact}}
=\frac{t}{(1-t)^2}(m_y-m).
$$

Therefore,

$$
v_y-v=\frac{m_y-m}{1-t}
=\frac{1-t}{t}G_{\mathrm{exact}}.
\tag{32}
$$

The proposed practical guided field is consequently

$$
F_{\mathrm{FM}}(x_t,t)
=v_\theta(x_t,t)+w(t)\frac{1-t}{t}G_{\mathrm{SMG}}(x_t,t).
\tag{33}
$$

The likelihood-derived ideal uses $w(t)=1$ and exact quantities. A tuned weight is another design choice; it does not generally preserve the same declared target distribution.

Equations involving $1/t$ or $1/(1-t)$ must not be evaluated blindly at endpoints. Use justified limits or declared cutoff/terminal policies.

### 10.3 Diffusion probability-flow sampling

Use forward noising time $\tau$, with clean data at $\tau=0$, and a forward SDE

$$dX=b(X,\tau)\,d\tau+\kappa(\tau)\,dW_\tau.$$

Here $W_\tau$ is Brownian motion. For our deterministic sampling comparison, the relevant field is the probability-flow ODE:

$$
F_{\mathrm{diff}}(x,\tau)
=b(x,\tau)-\tfrac12\kappa(\tau)^2
\left(s_\theta(x,\tau)+G_{\mathrm{SMG}}(x,\tau)\right).
\tag{34}
$$

The symbol $s_\theta$ denotes the model's score estimate, not the property width $s$. Sampling integrates this ODE from large $\tau$ toward zero, using negative time steps. The ODE construction is established in [Song et al., Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).

The reverse SDE has a different score coefficient and a stochastic term. Deterministic Heun equations should not be presented as a new stochastic DDPM update.

Exact conditional transport also requires the correct starting distribution. At a finite-noise cutoff, the conditional marginal can still depend on the target. Starting from unconditional Gaussian noise is then an additional approximation; record it and test sensitivity to the cutoff.

### 10.4 Heun changes numerical integration, not the guidance formula

For either full field $F$, with signed step $h$:

$$
\begin{aligned}
k_1&=F(x_n,t_n),\\
\widetilde x&=x_n+hk_1,\\
k_2&=F(\widetilde x,t_n+h),\\
x_{n+1}&=x_n+\frac h2(k_1+k_2).
\end{aligned}
\tag{35}
$$

**Proof of second-order accuracy.** Differentiating $x'=F(x,t)$ gives

$$x''=\partial_tF+(\partial_xF)F.$$

Thus the exact trajectory has expansion

$$
x(t+h)=x+hF+\frac{h^2}{2}
\left(\partial_tF+(\partial_xF)F\right)+O(h^3).
$$

Expanding the predicted-stage evaluation gives

$$
k_2=F+h\left(\partial_tF+(\partial_xF)F\right)+O(h^2).
$$

Substitution into equation (35) reproduces the exact expansion through $h^2$. For a sufficiently smooth field on a finite interval with the usual stability conditions, local error is $O(|h|^3)$ and accumulated global error is $O(h^2)$. Euler has global error $O(h)$.

The second stage must reevaluate the entire guided field, including $m,c,v_f$ and schedules, at the new state and time. Reusing the old guidance changes the method.

Heun already appears in [Karras et al., NeurIPS 2022, Algorithm 1](https://papers.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf) for diffusion and [Lee et al., NeurIPS 2024, Section 5.3 and Appendix D](https://arxiv.org/html/2405.20320v2) for rectified flow. Its use here is an established solver control.

A more accurate solver can follow an inaccurate guidance field more accurately. Therefore solver convergence does not prove correct conditional sampling.

## 11. Comparison with previous methods

The comparison must distinguish four objects: the posterior approximation, the property or likelihood approximation, the derivative convention, and the numerical solver. Similar titles do not imply identical methods; different notation does not establish novelty.

### 11.1 Overview

| Method or component | Shared idea | What distinguishes our proposed formula, if anything |
|---|---|---|
| Plug-in guidance / DPS template | Differentiate a loss or likelihood through a clean estimate | SMG changes the scalar residual and uncertainty denominator |
| TMPD | Posterior moments and covariance-aware likelihood | Its displayed linear-observation case equals SMG's affine-property limit |
| MMPS | Tweedie covariance, moment matching, negligible covariance derivative | Further substantial overlap; same affine-property limit |
| On the Guidance of Flow Matching | General guidance framework, covariance identity, approximate and Monte Carlo variants | The specific nonlinear-property mean correction differs from its displayed plug-in equation; framework overlap remains |
| STSL | Efficient second-order Tweedie information in guidance | Published correction uses noisy-density curvature through a surrogate |
| GGDOpt | Gaussian posterior approximation and quadratic expansion | Expands the optimization energy; its displayed matrix formula differs from SMG's property-moment rule |
| ABMS | Reduce plug-in bias using several possible denoised outcomes | Uses an extra backward step and Monte Carlo estimation rather than our local moment rule |
| Heun | More accurate integration of an ODE | No solver novelty claimed |

### 11.2 DPS and the plug-in template

[Diffusion Posterior Sampling for General Noisy Inverse Problems](https://arxiv.org/abs/2209.14687) already handles nonlinear inverse problems. Therefore “guidance for nonlinear properties” is not a defensible broad novelty claim.

For a Gaussian squared-property likelihood, the plain plug-in comparison is equation (4). Practical DPS variants can change gradient scaling, so experiments must document which published implementation or controlled mathematical baseline is being used.

### 11.3 TMPD and MMPS: exact overlap in a special case

Set $f(X)=a^\top X+b$. Then $g=a$, $H=0$, and

$$
G_{\mathrm{SMG}}
=J^\top a\,
\frac{y-a^\top m-b}{s^2+a^\top\Sigma a}.
\tag{36}
$$

This is the scalar affine form of the covariance-frozen linear-observation guidance in [TMPD, equations (10)-(11)](https://arxiv.org/html/2310.06721v3). It is also obtained from [MMPS, Section 4.2, equations (19)-(22)](https://arxiv.org/html/2405.13712v5).

**Our algebraic conclusion:** the uncertainty denominator and this special case are shared prior work. The nonlinear mean term $c$ is absent from those linear-observation expressions because an affine function has no curvature to correct. That is appropriate for their setting, not a mathematical flaw.

MMPS also neglects the covariance derivative in its displayed guidance. That practical convention is not unique to SMG. The linked MMPS text is the October 2025 revision of work published at NeurIPS 2024.

### 11.4 On the Guidance of Flow Matching

[On the Guidance of Flow Matching](https://arxiv.org/html/2502.02150v2) already states the covariance–Jacobian identity in Proposition 3.4 under its independent Gaussian-path assumptions. Its equation (6) is a time-scaled gradient of an energy evaluated at the clean mean.

For the energy

$$E(X)=\frac{(f(X)-y)^2}{2s^2},$$

that plug-in gradient is

$$-\nabla_{x_t}E(m)=\frac{y-f(m)}{s^2}J^\top g.$$

Its Section 3.4 also discusses Gaussian posterior approximations and Monte Carlo guidance. Thus, “apply covariance-aware guidance to flow matching” is already covered in prior work. Our $c$ and $v_f$ are not present in that particular plug-in equation, but this does not prove a distinction from every method the broader framework can express.

### 11.5 STSL: second-order guidance already exists

[Beyond First-Order Tweedie: Solving Inverse Problems using Latent Diffusion](https://openaccess.thecvf.com/content/CVPR2024/papers/Rout_Beyond_First-Order_Tweedie_Solving_Inverse_Problems_using_Latent_Diffusion_CVPR_2024_paper.pdf), CVPR 2024, introduces STSL.

Its Theorem 4.4 and equation (6) use a surrogate correction involving

$$-\gamma\nabla_{x_t}\operatorname{tr}\left(\nabla_{x_t}^2\log p_t(x_t)\right),$$

alongside a data-fit gradient. The Hessian here concerns the noisy-state log density. SMG's mean correction instead contracts the clean-property Hessian with posterior covariance.

These displayed corrections are different. Nevertheless, STSL directly defeats the broad claim that efficient second-order information in diffusion guidance is new. Its existence is a reason to narrow our claim, not dismiss the paper because its formula has different symbols.

### 11.6 GGDOpt: expanding energy versus expanding the property

[A Gradient Guided Diffusion Framework for Chance Constrained Programming](https://arxiv.org/html/2510.12238v1), equation (20), gives quadratic-energy guidance. Rename its objective $E$, use $u=x_t$, isotropic posterior variance $\rho^2$, and strength $\beta$. Its displayed expression is

$$
G_{\mathrm{GGDOpt}}
=-\frac1{\rho^2}
\left[
\left(B+\frac1{\beta\rho^2}I\right)^{-1}
\left(-Bu+a-\frac{m}{\beta\rho^2}\right)+m
\right],
\tag{37}
$$

where $a=\nabla E(u)$ and $B=\nabla^2E(u)$.

For our squared-property energy,

$$
\nabla E=\frac{f-y}{s^2}\nabla f,\qquad
\nabla^2E=\frac{\nabla f\nabla f^\top+(f-y)\nabla^2f}{s^2}.
\tag{38}
$$

So even a common underlying property can produce different approximations when one expands $E$ and the other expands $f$.

**Our formula counterexample:** take $f(z)=z^2$, $u=m=y=s=\beta=1$, $\rho^2=\Sigma=0.1$, and $J=0.5$. Then $a=0$, $B=4$, so equation (37) returns zero. Equation (19) returns $-1/14$.

These moments are compatible with a Gaussian channel: a prior mean $2/3$, prior variance $2/15$, $\alpha=1/2$, $\sigma^2=1/10$, and observation $u=1$ yield this $m,\Sigma,J$. The difference is therefore not based on an inconsistent choice of moments.

This establishes nonidentity of these displayed formulas at one valid state. It establishes neither superiority nor priority over other methods or variants.

### 11.7 ABMS and Monte Carlo alternatives

[One step further with Monte-Carlo sampler to guide diffusion better](https://arxiv.org/html/2603.06685v1) introduces ABMS. Section 4.1 and Algorithm 1 use samples from an additional backward step, denoise them, and average a conditional function before constructing guidance. Section 5.3 includes molecular inverse design on QM9.

The paper's function need not be the same object as our scalar property $f$: a property, energy, likelihood, and log likelihood have different expectations. A comparison must align the intended target and inspect the implemented function, not simply match the letter used in a formula.

SMG replaces such sampling-based estimation with a local moment construction, possibly using random probes for its trace. A computational advantage must be measured. More Monte Carlo samples reduce sampling error for the distribution actually sampled; they do not automatically fix a misspecified approximate posterior.

Molecular application alone is not a new contribution in this comparison.

## 12. Exactly what could count as a contribution

### 12.1 What is certainly established mathematics or prior work

We should not claim invention of:

- Taylor approximation of a transformed mean or variance.
- The covariance–Jacobian / second-order Tweedie identity.
- Using posterior uncertainty in guidance.
- Guidance for nonlinear observations.
- Covariance-aware guidance in flow matching.
- Efficient second-order guidance in general.
- Heun sampling, including its use with diffusion and rectified flow.
- Rejection sampling as a reference technique.
- The observation that affine properties have zero Hessian.

### 12.2 The specific candidate

The proposed method combines:

1. A nonlinear-property mean shift $c=\tfrac12\operatorname{tr}(H_f\Sigma)$.
2. A leading property variance $v_f=g^\top\Sigma g$.
3. A Gaussian property-likelihood approximation.
4. A practical guidance derivative retaining $(y-f(m)-c)J^\top g/(s^2+v_f)$.
5. A computation based on covariance actions and trace estimation, adapted to the chosen generator.

A defensible candidate claim is:

> We investigate whether a curvature correction to the predicted nonlinear-property mean improves moment-based guidance beyond variance-only correction, at matched computational cost.

This is a research hypothesis. “We investigate” is currently justified; “we are the first” or “we outperform” is not.

### 12.3 What would make it a useful contribution?

| Possible contribution | Evidence needed |
|---|---|
| Distinct practical algorithm | A broader formula and implementation audit finding a meaningful difference from close methods |
| Better efficiency | Better quality at equal measured total time or memory, not merely fewer nominal steps |
| Better conditional sampling | A matched-target distribution reference and improved distributional metrics |
| A useful mechanism finding | Ablations showing when the mean term helps, fails, or dominates the variance term |
| A theoretical contribution | A result beyond the standard identities, such as a justified bound for the implemented field under explicit assumptions |
| An application contribution | A substantive, reproducible result in the chosen domain beyond simply transferring known formulas |

Standard ingredients can form a valuable method. But giving their combination a name does not establish novelty. Equally, an adaptation can be a worthwhile class project without being a new general-purpose research algorithm.

If a prior method already implements the same effective correction, describe our work as an adaptation or evaluation. An evaluation contribution still requires informative results; standard sanity checks alone do not establish one.

## 13. Assumptions, failure cases, and proof limits

### 13.1 The assumptions are separate

| Layer | Requirement | What can go wrong |
|---|---|---|
| Gaussian-channel identity | Independent Gaussian corruption, exact mean, regularity | Different coupling invalidates the identity; a learned mean introduces error |
| Mean Taylor approximation | Smooth property and controlled posterior spread | Multiple distant modes, strong curvature changes, or heavy tails |
| Leading variance | Linear property variation dominates locally | Near-zero gradient can hide curvature-driven variance |
| Gaussian property likelihood | Moments adequately summarize the relevant likelihood integral | Skewness, multiple modes, or narrow target width |
| Practical derivative | Dropped gradient terms are tolerable | Correct moments can still yield a poor guidance score |
| Numerical sampling | Appropriate step sizes and endpoint policy | Solver error can hide or imitate a guidance benefit |
| Scientific property | Predictor represents the desired real property | Guidance may exploit predictor error |

The exact covariance identity does not make the other rows exact.

### 13.2 An error ledger

There are at least five distinct sources of error:

1. Generator error in $m$ and its derivatives.
2. Property-predictor error, if $f$ is learned.
3. Taylor truncation and Gaussian likelihood approximation.
4. Dropped derivatives and finite-probe trace error.
5. Numerical ODE integration error.

Improving one source does not necessarily dominate the others. Heun addresses the fifth; higher property moments address part of the third; neither automatically resolves the rest.

### 13.3 What the proofs do and do not establish

| Statement | Status |
|---|---|
| $\Sigma=(\sigma^2/\alpha)J$ under equation (6) and an exact mean | Exact identity, proved in Section 4 |
| Quadratic mean correction and its third-moment remainder bound | Proved under Section 5's assumptions |
| Leading variance from linearization | Proved for the linearized property |
| Gaussian quadratic-polynomial variance | Proved in Section 8 |
| Full derivative of the Gaussian surrogate | Exact calculus result, Section 6 |
| Practical SMG is the exact derivative of that same surrogate | Generally false because terms are omitted |
| Practical SMG always improves conditional sampling | Unproved and not expected universally |
| SMG has established research priority | Not established by this audit |
| Heun is second order for a smooth specified field | Standard result, derived in Section 10 |

### 13.4 Units and transformations: useful checks, not novelty

For an affine change of property units, let

$$
f'=af+b,\qquad y'=ay+b,\qquad s'=|a|s,\qquad a\ne0.
$$

Then $g'=ag$, $H'=aH$, $c'=ac$, $v_f'=a^2v_f$, and the corrected residual becomes $r'=ar$. Consequently,

$$
G'_{\mathrm{SMG}}
=\frac{ar}{a^2V}J^\top(ag)=G_{\mathrm{SMG}}.
$$

Plug-in guidance is invariant under the same consistent transformation. This is a units check, not evidence favoring SMG.

A nonlinear monotone transformation preserves an appropriately transformed hard conditioning event. However, a new Gaussian likelihood in the transformed property generally defines a different soft target. Comparing those two Gaussian-guided samplers is not automatically a test of representation invariance.

Even exact agreement in property means does not establish agreement of the full generated distributions.

## 14. Experiments needed to support the claim

### 14.1 Establish what changes the result

Use the same base model, property, target, initial seeds, and declared likelihood:

| Guidance | Euler | Heun |
|---|---|---|
| Unguided | Control | Solver control |
| Plug-in | Baseline | Baseline with improved integration |
| Mean correction only | Ablation | Ablation |
| Variance correction only | Ablation | Ablation |
| Practical SMG | Candidate | Candidate |

On small problems, additionally compare the full surrogate gradient from equation (18), richer variance from equation (23), and an accurate likelihood-gradient reference.

This separates four questions: does the mean correction help, does variance scaling help, do omitted gradients matter, and is the observed difference merely numerical integration error?

### 14.2 Use the correct conditional reference

For target equation (1), draw $X\sim p_0$ and accept with probability

$$
A(X)=\exp\left[-\frac{(f(X)-y)^2}{2s^2}\right]\leq1.
$$

**Proof:** the density of an accepted draw is proportional to $p_0(X)A(X)$, which is equation (1) after normalization.

A reference accepting all samples inside a fixed band instead targets $p_0(X)\mathbf 1_{\{|f(X)-y|\leq\Delta\}}$. It answers a different distributional question.

For toy problems use the exact known prior. For molecular experiments, a reference drawn from a numerical base sampler is conditional on that sampler's distribution, which itself approximates the learned model.

### 14.3 Compare against substantive prior work

Beyond plug-in and our ablations, include matched-target covariance/moment guidance, a quadratic-energy comparator where applicable, and a Monte Carlo alternative. Use published implementations when possible, and document adaptations needed for the selected property and model.

Compare equal total wall time as well as equal field-evaluation budgets. Count derivative work and probe count. Equal Euler and Heun step counts do not mean equal cost.

### 14.4 Measure both success and damage

Report property error and in-band fraction together with sample validity, physical or structural stability, coverage/diversity, runtime, and memory. On tractable examples, also compare the sampled distribution with the correct conditional reference.

Use paired seeds, multiple independent batches, and uncertainty intervals. Test several property curvatures and posterior-noise levels rather than selecting one favorable example.

Measure the size of the mean shift, variance correction, and omitted gradient terms across time. Compare analytic properties with learned predictors to distinguish the mathematical mechanism from predictor artifacts.

For deterministic Heun convergence checks, use exact traces on small problems. If probes are needed, keep a seeded probe set fixed along each trajectory; a freshly randomized field at every stage does not meet the ordinary smooth deterministic ODE argument.

### 14.5 Results that would weaken or reject the proposal

- The curvature correction is negligible in the intended operating regime.
- Mean correction gives no benefit over variance-only or tuned plug-in guidance.
- Gains disappear once solver error is controlled.
- Full-gradient diagnostics show the omitted terms dominate.
- A close existing method reproduces the same behavior more simply.
- Monte Carlo guidance wins at matched cost.
- Improved property targeting comes with unacceptable validity or coverage loss.

Such findings would still be informative. They should change the conclusion rather than be hidden by a novelty claim.

## 15. Current status and defensible wording

The project contains analytic checks of covariance/moment identities, historical toy experiments, and an analytic Heun audit. The [Heun/formula audit](SMG_PRIOR_WORK_AUDIT.md#5-verification-status) records observed numerical orders close to one for Euler and two for Heun. These are tests of formulas and numerical integration, not evidence of molecular effectiveness.

The historical v7 guided toy used a hard-band reference while its guidance targeted a Gaussian soft likelihood. Those results require a matched-target rerun before they support a conditional-distribution accuracy claim. The required crossed solver/guidance comparison and molecular benefit remain unestablished in the existing records.

**Wording justified now:**

> Second-Moment Guidance is a proposed nonlinear-property guidance rule that combines a quadratic Taylor correction to the conditional property mean with a leading-order property variance estimate. It uses established posterior covariance identities and a Gaussian likelihood approximation, and simplifies the likelihood gradient by holding the correction moments fixed locally. We study whether the mean correction provides a useful improvement over existing moment-based and Monte Carlo guidance at matched cost.

**Wording not justified now:** “the first second-order guidance method,” “a new Tweedie identity,” “a novel Heun sampler,” “guaranteed unbiased guidance,” “universally better than plug-in,” or “proved innovative.”

The direct answer to “are we just using a first-order approximation for variance?” is: **the variance-only variant does exactly that. Full practical SMG also corrects the nonlinear property's mean using a quadratic Taylor term. Both mathematical ingredients are established. Their specific use may form a useful adaptation, but originality and advantage require evidence beyond the derivation.**

## 16. Can SMG be incorporated into training?

**Yes.** The current v7 method is inference-time guidance, but its computations can supply training targets or motivate new training losses. These are proposed extensions, not implemented results or established novelty claims.

Three routes answer different questions:

| Route | What is trained? | What changes during inference? |
|---|---|---|
| Distill SMG | A conditional model imitates SMG's guided field | Replace explicit SMG calculations with the trained field |
| Learn property moments | A network predicts conditional property mean and variance | Replace Taylor moment estimates with learned estimates |
| Add a property-aware auxiliary loss | The generator itself receives additional supervision | Generator behavior changes; explicit SMG may still be used |

### 16.1 Distill SMG into a model

Distillation means training a student to imitate a teacher. Freeze a pretrained generator and the property function. Together with SMG, they define a teacher field

$$
F_T(x_t,t,y,s)
=F_{\mathrm{base}}(x_t,t)+\Delta F_{\mathrm{SMG}}(x_t,t,y,s),
$$

where $\Delta F_{\mathrm{SMG}}$ includes the appropriate conversion from score to velocity in Section 10. Train a student field $F_\phi$ with parameters $\phi$:

$$
\mathcal L_{\mathrm{distill}}(\phi)
=\mathbb E_{(x_t,t,y,s)\sim Q}
\left[\left\|F_\phi(x_t,t,y,s)
-\operatorname{stopgrad}(F_T(x_t,t,y,s))\right\|^2\right].
$$

Here $Q$ is the training distribution over states, times, and requested targets. Stop-gradient means the teacher output is treated as a fixed label. Only student parameters are updated.

For a deterministic teacher and an unrestricted student, the pointwise squared error is minimized by reproducing the teacher field. This establishes what the training objective asks for; it does not guarantee that a finite network will learn it or that the teacher targets the exact desired distribution.

At inference, evaluate $F_\phi$ directly and integrate it with the chosen sampler. Explicit property Hessians and covariance actions can then be omitted. Several sampling steps may still be necessary: field distillation is not automatically one-step generation.

Training should cover the states visited by guided trajectories, not only convenient unconditional states. Include the target $y$ and width $s$ as inputs if both must vary; generalization to unseen target ranges is unproved. Teacher approximation errors remain in the desired student output, and student fitting error adds another source of error.

This is the most direct extension if the goal is to move repeated SMG computation from inference into a reusable training stage.

### 16.2 Learn the conditional property moments directly

Instead of calculating $f(m)+c$ and $g^\top\Sigma g$, learn functions

$$
a_\phi(x_t,t)\approx\mathbb E[f(X)\mid x_t],
\qquad
b_\psi(x_t,t)\approx\mathbb E[f(X)^2\mid x_t].
$$

Given clean training samples $X$, evaluate their property and construct $x_t$ using the same corruption path as the generator. Use

$$
\mathcal L_{\mathrm{mean}}=\mathbb E[(a_\phi(x_t,t)-f(X))^2],
\qquad
\mathcal L_{\mathrm{second}}=\mathbb E[(b_\psi(x_t,t)-f(X)^2)^2].
$$

**Why these targets work:** for a random label $U$ and fixed $x_t$,

$$
\mathbb E[(a-U)^2\mid x_t]
=(a-\mathbb E[U\mid x_t])^2+\operatorname{Var}(U\mid x_t).
$$

The second term does not depend on $a$, so the minimum is the conditional mean. Apply this identity once with $U=f(X)$ and once with $U=f(X)^2$. This argument requires the relevant finite moments, sufficient model capacity, and successful fitting.

Estimate the property variance by $b_\psi-a_\phi^2$. In finite models it may be negative or poorly calibrated, so positivity and calibration need explicit treatment. A positive-variance probabilistic head is another possible parameterization.

The learned moments can enter the Gaussian likelihood in Section 6, with guidance obtained by differentiating that surrogate with respect to $x_t$. This can avoid property Hessian evaluation at inference, but still requires input derivatives of the moment predictor. Accurate moments alone do not make the property distribution Gaussian.

This is a learned moment-guidance extension: it replaces the Taylor construction rather than proving the original SMG approximation more accurate. Its labels must correspond to the same clean distribution and corruption path for which guidance is intended.

### 16.3 Add an auxiliary loss during generator training

A more direct, but less cleanly justified, modification is

$$
\mathcal L_{\mathrm{total}}(\theta)
=\mathcal L_{\mathrm{base}}(\theta)
+\lambda\,
\mathbb E\left[
\left(f(m_\theta(x_t,t))+c_\theta(x_t,t)-f(X)\right)^2
\right].
$$

Here $\theta$ are generator parameters, $\mathcal L_{\mathrm{base}}$ is its usual diffusion or flow-matching loss, and the extra term supervises the corrected property prediction using the clean training sample's property.

This is a candidate auxiliary objective, not a derived likelihood-training principle. Differentiating through $c_\theta$ can require expensive mixed derivatives. More fundamentally, an additional objective can move $m_\theta$ away from the posterior mean learned by the base loss. Then interpreting its Jacobian as the original posterior covariance is no longer automatically justified.

The label is $f(X)$, not one fixed desired value $y$ for every training sample. Pushing every example toward a single target changes the intended generator distribution. A model for many targets would require an explicit conditional or reweighted training design.

One cannot simply add the SMG vector to a parameter gradient: $G_{\mathrm{SMG}}$ lives in sample space, while $\nabla_\theta\mathcal L$ lives in parameter space. A training loss and a chain-rule connection must be specified.

### 16.4 Which extension should we investigate first?

For reducing the inference cost of the existing proposal, first validate explicit SMG, then test distillation. Compare the student with its teacher, a distilled plug-in teacher, and an appropriate directly trained conditional baseline.

For improving the property-moment approximation, compare direct moment learning with the Taylor estimates using held-out conditional-moment tests. For changing the generator's training objective, establish a separate hypothesis and verify that sample quality and posterior-mean calibration survive.

Training guidance networks is already prior work: [On the Guidance of Flow Matching, Section 3.5 and Appendix A.6](https://arxiv.org/html/2502.02150v2) develops guidance-matching and related reweighted training losses. Its losses are not the same as the proposed frozen-SMG-teacher loss above, but their existence rules out claiming that moving guidance into training is itself new.

Any claimed speed benefit must include teacher-label generation and student training costs, together with the number of future samples needed to recover that upfront expense. None of these training extensions has been implemented or evaluated as part of this document update.
