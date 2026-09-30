# Original BTVG: adversarial mathematical audit

25 September 2026. Scope: the original Gaussian-KL proposal and the current
`btvg` implementation, not BTVG2. This audit tests both the defense of BTVG
and the reasoning previously used against it.

**Outcome: the objective is mathematically coherent; the original argument
that its variance correction must improve final targeting is not valid.**
Several earlier negative arguments also overreach. The defensible description
is an approximate local moment regularizer with conditional benefits, not a
proved controller of terminal residual variance or band success.

The checks in [btvg_soundness_checks.py](../../audit/btvg_soundness_checks.py)
passed all 12 groups. They include finite-difference checks of the losses,
explicit counterexamples, and an exactly solvable Gaussian flow. They are
mathematical checks, not new molecular experiments. The project virtual
environment could not launch, and the available Python environments checked
did not provide PyTorch, so the molecular implementation tests were not rerun.

## 1. What survives: the loss and its derivative

At fixed time, write

\[
\mu(x)=f_A(m_t(x)),\qquad e(x)=\mu(x)-y,\qquad
V(x)=g^T\Sigma_t(x)g,\quad g=\nabla_m f_A(m_t(x)).
\]

The original proposal assumes a Gaussian property law and minimizes

\[
L=\operatorname{KL}\bigl(N(\mu,V)\Vert N(y,\tau^2)\bigr)
=\frac12\left[\frac{e^2+V}{\tau^2}-1+\log\frac{\tau^2}{V}\right].
\]

For positive V and tau, its exact state gradient is

\[
-\nabla_x L=-\frac{e}{\tau^2}\nabla_x\mu
-\frac12\left(\frac1{\tau^2}-\frac1V\right)\nabla_xV.
\]

**Defense succeeds.** The algebra is correct. In unconstrained moment space
the loss is strictly convex and has its unique minimum at e=0, V=tau^2.
Under frozen-time gradient flow in x, dL/ds=-||grad L||^2. This is a local
surrogate-descent result; the realizable map x -> (mu,V) need not reach that
minimum or be convex.

### The implementation has a different, still coherent loss

The code sets the variance coefficient to zero below tau^2. Define

\[
L_+=\frac{e^2}{2\tau^2}+\psi(V/\tau^2),\qquad
\psi(u)=\begin{cases}0,&u\le1,\\
\frac12(u-1-\log u),&u>1.
\end{cases}
\]

For positive V this differentiable loss gives exactly the implemented
one-sided coefficient, provided the variance derivative is exact. The
variance penalty is zero for every V <= tau^2; it supplies no lower bound
on variance. Extending the zero penalty to a negative estimated V is possible
as an algebraic convention, but a negative V has no variance interpretation.

Thus “clipping the coefficient destroys every objective interpretation” is
false. However, calling the resulting field the gradient of the original KL
is also false. It neither restores V to tau^2 from below nor prevents collapse.

The sweep multiplies original-BTVG strength by (tau/s)^2. At matched nominal
strength, the resulting field for positive V is

\[
G=-\frac{e}{s^2}\nabla\mu
-\frac1{2s^2}\left(1-\frac{\tau^2}{V}\right)_+\nabla V.
\]

Its mean term is exactly plug-in guidance's mean term. Consequently, after
this normalization, an incremental improvement must come from the added
variance correction and its interaction with the sampler. A larger mean gain
is not an independent contribution in this comparison.

Sources: [original proposal](PROPOSALS_VARIANCE_TARGETED_GUIDANCE.md),
[gradient and field implementation](../../proj1/src/guidance.py), and
[strength normalization](../../proj1/scripts/guidance_sweep.py).

## 2. Attack: shrinking variance is not always the correct targeting action

For an exact Gaussian residual R~N(e,V), success in [-delta,delta] is

\[
p(e,V)=\Phi\left(\frac{\delta-e}{\sqrt V}\right)
-\Phi\left(\frac{-\delta-e}{\sqrt V}\right).
\]

Its variance derivative is

\[
\partial_Vp=
\frac{(e-\delta)\phi((\delta-e)/\sqrt V)
-(e+\delta)\phi((-\delta-e)/\sqrt V)}{2V^{3/2}}.
\]

**Defense succeeds in a restricted regime.** If |e| <= delta and V>0,
this derivative is negative. At fixed mean, reducing variance improves
coverage. Concentrating a calibrated distribution already centered near the
target is mathematically justified.

**The unrestricted defense fails.** At e=3 delta, shrinking standard
deviation from delta to delta/2 changes success from 0.02271846 to
0.00003167124, a 717-fold decrease. Narrowing around a wrong answer can
make success less likely. The likelihood's occasional preference for
widening is therefore not intrinsically “wrong for design.” The appropriate
action depends on the objective and current mean.

### Stronger counterexample: the mean term does not automatically rescue it

One could object that the example holds the mean fixed, whereas BTVG moves
both moments. The following counterexample includes both terms and exact
derivatives.

Let delta=1, tau=1/1.96, and in a neighborhood of x=(0,0) define an exact
Gaussian family with

\[
e(x)=3+0.1x_1,\qquad V(x)=1+10x_2.
\]

Take one gradient step x_new=x-10^-4 grad L. All variances remain above
tau^2, so original and one-sided BTVG agree.

| Quantity | Before | After |
|---|---:|---:|
| Mean error | 3.0000000000 | 2.9999884752 |
| Variance | 1.000000 | 0.985792 |
| KL loss | 18.03505553 | 18.01478693 |
| Band probability | 0.02271846 | 0.02195838 |

**Mean error decreases, variance decreases, and KL decreases, but success
decreases.** This smooth Gaussian-family example refutes the proposed general
implication even without estimator noise. It is a counterexample about the
objective, not a claim that this particular moment map is the QM9 posterior
or a specific independent-Gaussian interpolant.

Even the moment risk e^2+V need not decrease under KL descent. At x=0, let
e=1+x, V=2-3x, tau^2=1. Then G=-0.25, dL/ds=-0.0625, but
d(e^2+V)/ds=+0.25. Entropy regularization makes KL and squared-error risk
different objectives.

## 3. Attack: adding a descent direction cannot make mean guidance worse

Let a=grad mu, h=grad V, A=1/tau^2, and
B=max((1/tau^2-1/V)/2,0). With exact derivatives,

\[
G=-Ae a-Bh,
\]
\[
\frac{d(e^2/2)}{ds}=-Ae^2\|a\|^2-Be\,a^Th,
\qquad
\frac{dV}{ds}=-Ae\,a^Th-B\|h\|^2.
\]

The cross term has no fixed sign. The variance component alone decreases V,
but can increase mean error; the combined field need not decrease both.
For example, e=-1, V=4, tau^2=1, a=1, h=4 gives mean guidance +1,
variance guidance -1.5, and total guidance -0.5. The added term reverses
progress toward the target while the combined KL still decreases locally.

This conflict also occurs with an exact Gaussian interpolant, rather than
only an arbitrary moment map. Take independent X_1,epsilon~N(0,1), t=0.5,
and f(z)=z^2. Then X_1|X_t=x~N(x,0.5), J=1 and Sigma=0.5. At x=0.5,
y=1 and tau=0.1, original BTVG's proxies give mu=0.25, V=0.5,
dmu/dx=1 and dV/dx=2. Its mean field is +75, variance field -98,
and combined field -23. There is no generator or gradient estimation error
in this example; the nonlinear observable closure is still approximate.
For the true property band [1-0.196,1+0.196], an x edit of 10^-5 times
the combined field lowers exact posterior coverage from 0.09883485 to
0.09882342; mean-only raises it to 0.09887214. These are fixed-time local
interventions without clipping, not terminal performance measurements.

**A conditional defense is available:** if e a^T h >= 0, both local
derivatives above are nonpositive. Otherwise a trade-off must be resolved.
Actual-state projection can preserve the local mean while decreasing V, but
it only protects the declared local quantities, and the final combined clip
and finite step must also be accounted for. Such a repair changes the field
and does not retroactively prove the original BTVG claim.

## 4. Attack the earlier criticism: “BTVG ignores across-sample spread”

This criticism is too strong. For a common target y and any distribution
of states,

\[
\mathbb E[(\mu(X_t)-y)^2]
=\operatorname{Var}(\mu(X_t))+(\mathbb E[\mu(X_t)]-y)^2.
\]

BTVG penalizes per-state errors, not only the error of the population mean.
Its mean term therefore does penalize across-state dispersion. A zero
population bias does not imply that this term has finished its job.
With varying targets, apply the same identity to the residual mu(X_t)-Y;
raw property variance is not the targeting objective.

There is an exact example within the project's flow-matching assumptions.
Let X_1 and epsilon be independent N(0,1), X_t=t X_1+(1-t)epsilon, and
f(x)=x with target y=0. Write q=t^2+(1-t)^2. Then

\[
m_t(x)=\frac{t}{q}x,\qquad V_t=\frac{(1-t)^2}{q}.
\]

V is state-independent, so grad_x V=0 everywhere. With guidance beginning
at t=0.5, no clipping, s=1 and normalized weight w, BTVG equals plug-in
and has terminal variance exp(-w/2), compared with unguided variance 1.
The additional contraction follows from

\[
\int_{1/2}^1\frac{t(1-t)}{[t^2+(1-t)^2]^2}\,dt=\frac14.
\]

The numerical ODE check at w=1 gives 0.6065306597, matching exp(-1/2).
**Population concentration is possible with no variance-gradient term.**
This both defends the full method against the “ignores spread” criticism
and shows why improvement over an unguided model would not establish the
added term's value.

The following claims in the earlier
[variance diagnostic](../results/VARIANCE_DECOMPOSITION.md) need qualification:

- “No part of its objective touches” across-state spread is false by the
  identity above.
- Vanishing posterior V near t=1 does not prove earlier interventions had
  no effect. Under bounded derivatives it is expected from the noise schedule.
- A weak late correlation between V and final absolute error does not imply
  zero causal benefit from moving in -grad V. It tests a particular
  association, not the effect of an intervention or its gradient direction.
- A sum of an approximate f_A posterior variance and variance of f_A(m)
  is not automatically an exact total-variance decomposition of terminal
  f_B outcomes. The conditional law, outer law, property, and means must match.

The reported diagnostics can motivate skepticism. They do not establish
these stronger impossibility or causal conclusions. This audit leaves the
existing generated report and experimental records unchanged.

## 5. The real missing bridge: local moments versus sampler outcomes

There are three different mathematical objects:

1. The training interpolant posterior p(X_1 | X_t=x).
2. The proxy f_A(m_t(x)) and linearized variance g^T Sigma g used by BTVG.
3. The terminal property of the actual guided, discretized sampler.

For a deterministic ODE, its endpoint is fixed once its current state and
future controller are fixed. Its own conditional endpoint variance is zero.
The interpolant posterior can nevertheless have nonzero variance because
it describes a different coupling. That distinction is real.

However, it does **not** make interpolant posteriors irrelevant to guidance.
For an exact independent Gaussian interpolation, a terminal weight R(X_1)
defines

\[
h_t(x)=\mathbb E[R(X_1)\mid X_t=x],\qquad
v_t^R-v_t=\frac{1-t}{t}\nabla_x\log h_t(x).
\]

This obtains the reweighted marginal path even when the sampler is a
deterministic ODE. The justification is through marginal transport, not
random future branching of that ODE. See the primary derivation in
[On the Guidance of Flow Matching, Appendix A.3 and Proposition 3.4](https://arxiv.org/html/2502.02150v2).

**The missing proof for BTVG is that exp(-L_t), or exp(-L_{+,t}), forms
the appropriate h_t for the desired terminal law.** An arbitrary local
penalty does not acquire this property by using the score-to-velocity factor.
For example, under an exact Gaussian observable closure, integrating the
terminal weight exp(-(F-y)^2/(2tau^2)) gives

\[
h_t=\frac{\tau}{\sqrt{\tau^2+V}}
\exp\left[-\frac{e^2}{2(\tau^2+V)}\right],
\]

which is different from exp(-L). BTVG is allowed to choose another control
objective, but exact conditional sampling is then not established.

Likewise, under an exact gradient-based controller
dx/dt=v_t-lambda_t grad L_t,

\[
\frac{dL_t(X_t)}{dt}=\partial_tL_t+\nabla L_t^Tv_t
-\lambda_t\|\nabla L_t\|^2.
\]

Only the last term is controlled by the gradient-descent argument. Time
dependence, base flow, finite steps, and estimator error remain. Positive
scalar clipping preserves the exact guidance direction's instantaneous
descent, but does not remove those other terms or preserve baseline progress
when an additional direction changes the amount of joint clipping.

## 6. Estimator assumptions that cannot be skipped

Under an exact independent Gaussian channel, Sigma=((1-t)^2/t) J is a
valid covariance identity, where J=dm/dx. A learned mean map need not have
a symmetric positive semidefinite J. Nonpositive estimates cannot be called
calibrated uncertainty merely because the code suppresses their corrections.

Even a perfect covariance does not make g^T Sigma g the exact variance of a
nonlinear property. For X~N(m,Sigma) and quadratic f with Hessian H,

\[
\mathbb E[f(X)]=f(m)+\tfrac12\operatorname{tr}(H\Sigma),\quad
\operatorname{Var}(f(X))=g^T\Sigma g+
\tfrac12\operatorname{tr}(H\Sigma H\Sigma).
\]

For X~N(0,1), f(X)=X^2, BTVG's local linearized V is zero, while the true
variance is 2. Low proxy variance can mean a locally flat predictor rather
than a reliably targeted endpoint.

The exact derivative of the implemented scalar V=k g^T Jg, at fixed t, is

\[
\nabla_xV=kJ^TH(J+J^T)g+
k\nabla_x(g^TJg)\big|_{g\text{ fixed}}.
\]

The implemented 2 J^T H Sigma g expression assumes J=J^T. Its function
documentation acknowledges that assumption. Thus the current learned-model
field is not universally an exact gradient even of its stated proxy loss.
The original claim of a Hessian-free full variance derivative also fails:
movement of g contributes property curvature.

Finally, even exact first two moments do not certify Gaussian coverage.
A residual equal to 0 with probability .75 and to each of +/-2tau with
probability .125 has mean zero and variance tau^2, but only 75% of its mass
lies in +/-1.96tau. Smooth mixtures can approximate this example. Moment
matching alone does not establish 95% success.

## 7. The strongest claims that can be protected

**Gaussian distribution matching:** if Q=N(mu,V) is the true law being
evaluated, P=N(y,tau^2), tau=delta/1.96, and KL(Q||P)<=epsilon, Pinsker's
inequality gives

\[
Q(|F-y|\le\delta)\ge0.9500042-\sqrt{\epsilon/2}.
\]

This is a genuine certificate for a sufficiently small attained KL. It does
not say that every step reducing a large KL improves actual coverage. It
also does not apply to the full-KL value after silently substituting the
one-sided penalty, or to a miscalibrated Gaussian proxy.

**One-sided penalty at its ideal minimum:** e=0 and 0<V<=tau^2 gives at
least 95% Gaussian coverage when tau=delta/1.96. It provides no guarantee
of molecular validity or structural diversity.

**Moment-based risk without Gaussianity:** if mu and V are the actual
conditional moments of the outcome F under the law of interest, then

\[
\mathbb E[(F-y)^2\mid x]=(\mu(x)-y)^2+V(x),
\quad
\Pr(|F-y|>\delta)\le\frac{\mathbb E[(F-y)^2]}{\delta^2}.
\]

Thus jointly controlling squared bias and the appropriate variance has a
sound risk interpretation. Original BTVG does not automatically inherit it:
its KL includes an entropy term, its moments are approximate, and its local
posterior is not the guided sampler's actual continuation law.

**Incremental value of extra guidance:** the necessary comparison is the
same mean controller with and without the variance component, including
matched tuning and the same clipping/validity constraints. In the inspected
implementation, `btvg_var` contains only the variance term. The original
proposal's table described it as likelihood-mean plus variance; these are
different ablations. Failure of variance-only guidance does not rule out a
beneficial interaction with mean guidance, and success of the full method
over unguided does not establish that interaction.

## 8. Decision

- **Protect:** the Gaussian-KL formula; the one-sided loss interpretation;
  exact frozen-time descent under its assumptions; variance reduction near
  a correctly centered Gaussian target; and the conditional risk identities.
- **Withdraw:** automatic terminal-variance reduction, automatic band-success
  improvement, harmless addition of variance guidance, guaranteed diversity
  preservation, and exact probability calibration from the current proxy.
- **Correct the earlier rejection:** local and terminal variances differ,
  but the full method does penalize across-state error, and endpoint variance
  collapse or weak observational correlation cannot prove the extra term
  is useless.

The mathematical design survives as a precisely scoped surrogate. Its
original stronger explanation does not survive adversarial testing. A claim
about terminal success needs an additional, valid link to terminal outcomes;
that link is currently an assumption, not a theorem.
