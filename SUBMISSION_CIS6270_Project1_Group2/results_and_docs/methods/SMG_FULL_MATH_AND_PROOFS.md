# Innovation v7 — Second-Moment Guidance

For the consolidated beginner-friendly explanation, proofs, approximation-order choices, and qualified novelty assessment, read [Why innovative?](SMG_WHY_INNOVATIVE.md).

**Primary innovation under evaluation; updated 18 Sep 2026.** Original brainstorm: 17 Sep.
The current formula-level prior-work comparison and required Heun control are in
[SMG, Heun, and prior work](SMG_PRIOR_WORK_AUDIT.md). That audit supersedes the original
broad novelty claims below: covariance-aware guidance and second-order guidance already exist;
the nonlinear-property correction is a candidate contribution, not a verified priority claim.
Build plan, roles and timeline remain in `PROJECT_GUIDE.md`.

**Status labels** follow the house convention: `[PROVED]` derivation given · `[CHECKED]` verified
numerically, script named · `[ASSUMPTION]` required, not established · `[HYPOTHESIS]` an experiment
decides · `[TAKEN]` prior work · `[DESIGN]` a choice.

The original numerical checks are in `audit/marginalized_guidance_checks.py`, results in
`audit/marginalized_guidance_results.json`. Pure Python, no dependencies, runs in about four
minutes. **No molecular training, sampling, decoding or physical calculation has been performed.**
Additional analytic Heun/formula checks are in `audit/heun_formula_checks.py`, with separate
results in `audit/heun_formula_results.json`. They do not constitute a guided-sampler comparison.

---

## 1. The one-sentence version

> SMG corrects the difference between the property of the posterior mean and the posterior mean
> property in plug-in guidance. Its leading mean correction is computable from property curvature
> and estimated denoising covariance. The practical guidance remains approximate; usefulness,
> total cost and novelty must be established against the relevant existing methods.

The two are not generally the same thing. The original toy below shows a substantial difference
in mode coverage against a hard-band reference. Its Gaussian guidance target differs from that
reference; the matched-target and solver controls in §13 are required before a stronger claim.

---

## 2. What is actually wrong with the baseline

Write the interpolant that covers both families we are required to train:

$$x_t=\alpha_t x_1+\sigma_t\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I)\ \perp\ x_1$$

VP diffusion is $\alpha_t^2+\sigma_t^2=1$; independent-Gaussian linear flow matching is
$\alpha_t=t,\ \sigma_t=1-t$. An optimal-transport coupling is not automatically independent
and requires its own covariance/guidance derivation.
Let $\hat x_1(x_t)=\mathbb E[x_1\mid x_t]$ be the denoiser, $f$ the property, $y^\ast$ the target.

Training-free guidance needs $\nabla_{x_t}\log p(y^\ast\mid x_t)$, and

$$p(y^\ast\mid x_t)=\mathbb E_{x_1\sim p(x_1\mid x_t)}\big[p(y^\ast\mid x_1)\big].$$

The plain plug-in comparator replaces the likelihood expectation by evaluation at the mean:

$$p(y^\ast\mid x_t)\ \approx\ p\big(y^\ast\mid \hat x_1(x_t)\big).$$

For nonlinear properties, a related error is $\mathbb E[f(x_1)\mid x_t]\ne f(\hat x_1)$.
Correcting this property mean addresses one part of the likelihood approximation; the two
expectations are not equivalent. Published guidance variants can also use covariance or
Monte Carlo marginalization, so compare their actual estimators rather than grouping all of
them as plain plug-in. The moment expansion is established mathematics `[TAKEN]`; the scope
of the candidate methodological contribution is audited in the companion note.

---

## 3. The mathematics

### 3.1 Lemma 1 — the posterior covariance is the denoiser Jacobian `[PROVED]` `[CHECKED: check_A_tweedie]`

$$\boxed{\ \Sigma_t\ :=\ \operatorname{Cov}(x_1\mid x_t)\ =\ \frac{\sigma_t^2}{\alpha_t}\,
\frac{\partial\hat x_1}{\partial x_t}\ }$$

**Proof.** Condition on $x_t$. Since $p(x_t\mid x_1)=\mathcal N(\alpha_t x_1,\sigma_t^2 I)$,

$$p(x_1\mid x_t)\ \propto\ p(x_1)\exp\!\Big(\tfrac{\alpha_t}{\sigma_t^2}\,x_1^{\!\top}x_t-
\tfrac{\alpha_t^2}{2\sigma_t^2}\lVert x_1\rVert^2\Big),$$

an exponential family in the natural parameter $\theta=\alpha_t x_t/\sigma_t^2$ with sufficient
statistic $x_1$. For any exponential family, $\partial_\theta\mathbb E[x_1\mid\theta]=
\operatorname{Cov}(x_1\mid\theta)$. The chain rule $\partial_{x_t}=(\alpha_t/\sigma_t^2)\partial_\theta$
gives the result. $\square$

This is second-order Tweedie `[TAKEN]`; the exponential-family route is the shortest proof of it and
avoids ever writing $\nabla^2\log p_t$. Two consequences we use:

- It holds for **both** model families with only $(\alpha_t,\sigma_t)$ changing, so the FM-vs-diffusion
  comparison the assignment mandates becomes a mechanism experiment rather than a formality.
- For flow matching, $\hat x_1=x_t+(1-t)v_\theta$, so
  $\Sigma_t=\frac{(1-t)^2}{t}\big(I+(1-t)\,\partial v_\theta/\partial x_t\big)$ — one JVP.
- $\Sigma_t\succeq0$ and symmetric in exact arithmetic. A *learned* network need not satisfy that,
  and how badly it fails is a free diagnostic of checkpoint quality.

`[CHECKED]` 400 random 2-D Gaussian-mixture cases, random $\alpha,\sigma,x_t$: max relative error
between $\Sigma_t$ and $(\sigma^2/\alpha)\,\partial\hat x_1/\partial x_t$ is $8.1\times10^{-10}$.

### 3.2 Lemma 2 — the two moments of the property `[PROVED]` `[CHECKED: check_B, check_C]`

For $f\in C^3$ with gradient $g=\nabla f(\hat x_1)$ and Hessian $H=\nabla^2f(\hat x_1)$:

$$\mathbb E[f(x_1)\mid x_t]=f(\hat x_1)+\tfrac12\operatorname{tr}(H\Sigma_t)+R_3,\qquad
\operatorname{Var}[f(x_1)\mid x_t]=g^{\!\top}\Sigma_t\,g+R_3',$$

with $|R_3|\le\frac16 M_3\,\mathbb E\big[\lVert x_1-\hat x_1\rVert^3\mid x_t\big]$ and $M_3$ a bound
on the third derivative along the segment. Ordinary Taylor expansion about $\hat x_1$, using
$\mathbb E[x_1-\hat x_1\mid x_t]=0$.

Three exact corollaries, and they are the ones that matter:

| $f$ | $\tfrac12\operatorname{tr}(H\Sigma_t)$ | consequence |
|---|---|---|
| **linear** | identically $0$ | plug-in mean is **exactly unbiased**; variance term is exact |
| **quadratic** | exact, $R_3=0$ | correction is **exact**, no remainder |
| general smooth | leading term | $R_3=O(\lVert\Sigma_t\rVert^{3/2})$ |

`[CHECKED]` One state, $\operatorname{tr}\Sigma_t=1.56$, truth by 400 000 exact posterior draws
(Monte-Carlo s.e. in brackets):

| property | plug-in error | curvature term $\tfrac12\mathrm{tr}(H\Sigma)$ | corrected error |
|---|---|---|---|
| linear | $+0.00077$ (0.6 s.e. — zero) | $0.00000$ | $+0.00077$ (0.6 s.e.) |
| quadratic | $-0.65667$ (**497 s.e.**) | $0.65651$ | $-0.00015$ (0.1 s.e. — zero) |
| exponential | $-0.20492$ (**173 s.e.**) | $0.22180$ | $+0.01688$ (14.3 s.e.) |

The quadratic row is the sharp one: predicted correction $0.65651$, measured bias $0.65667$. They
agree to $1.6\times10^{-4}$, which is the Monte-Carlo noise.

### 3.3 The practical corrected guidance term `[DESIGN]` under stated approximations

Take $p(y\mid x_1)=\mathcal N(y;f(x_1),s^2)$ and approximate $f(x_1)\mid x_t$ as Gaussian with the
moments of Lemma 2. Then $p(y^\ast\mid x_t)\approx\mathcal N(y^\ast;\hat m_t,s^2+\hat v_t)$ with
$\hat m_t=f(\hat x_1)+\tfrac12\operatorname{tr}(H\Sigma_t)$ and $\hat v_t=g^{\!\top}\Sigma_t g$, and

$$\boxed{\ \nabla_{x_t}\log p(y^\ast\mid x_t)\ \approx\
\underbrace{\frac{y^\ast-f(\hat x_1)-\tfrac12\operatorname{tr}(H\Sigma_t)}
{s^2+g^{\!\top}\Sigma_t\,g}}_{\text{scalar}}\cdot
\underbrace{\Big(\tfrac{\partial\hat x_1}{\partial x_t}\Big)^{\!\top} g}_{\text{what DPS already computes}}\ }$$

**Dropped terms** `[ASSUMPTION]`. Write $c=\tfrac12\operatorname{tr}(H\Sigma_t)$,
$V=s^2+\hat v_t$, and $r=y^\ast-f(\hat x_1)-c$. The full derivative of this Gaussian
surrogate likelihood is

$$\nabla_{x_t}\log\widetilde p(y^\ast\mid x_t)
=\frac rV\left(J^\top g+\nabla_{x_t}c\right)
+\frac{r^2-V}{2V^2}\nabla_{x_t}\hat v_t.$$

The practical formula retains only $rJ^\top g/V$. The omitted terms can involve second
derivatives of the denoiser and third derivatives of the property; they are not generically
third derivatives of the denoiser. They need not be smaller in covariance order than the retained
correction. Measure them on a diagnostic subset. The moment derivation does not prove accuracy
of this truncated gradient or exactness of the resulting sampler.

### 3.4 Corollary 4 — the linear-property reduction `[PROVED]`

**For a linear measurement operator the mean correction is identically zero.** $H=0\Rightarrow
\operatorname{tr}(H\Sigma_t)=0$, and only the variance term survives.

This establishes a reduction to the linear-observation formulas, not a novelty proof.
TMPD's linear observation mean requires no curvature correction, and its scalar guidance
agrees with SMG when $H=0$, using the same covariance and derivative convention.
Covariance-aware guidance also exists in general flow-matching work, and nonlinear
quadratic-energy guidance exists. The precise overlaps and distinctions are recorded in
[the prior-work comparison](SMG_PRIOR_WORK_AUDIT.md#2-what-the-cited-methods-actually-share-with-smg).

`[CHECKED against the primary text, 17 Sep]` TMPD's observation model is $y=Hx_0+u$ and its
likelihood is $p(y\mid x_t)\approx\mathcal N\big(y;\,Hm_{0\mid t},\,HC_{0\mid t}H^{\!\top}+\sigma_y^2I\big)$.
The conditional mean passes straight through $H$ and the covariance appears **only** in the variance
— which is the correct and complete answer when $H$ is linear, and silently incomplete the moment it
is not. The authors note the gap themselves in a different guise, observing that latent diffusion
"would make any observation operator nonlinear."

We should expect a reviewer to try to break this claim, and the honest position is in §6.

### 3.5 Proposition 5 — the reduction table `[PROVED]`

The estimator is a strict generalisation, and every special case is a published method:

| set | recovers |
|---|---|
| $\Sigma_t\to0$ | DPS / FreeDoM / TFG / TFG-Flow plug-in guidance `[TAKEN]` |
| $H=0$, $\Sigma_t\approx r_t^2I$ | ΠGDM `[TAKEN]` |
| $H=0$, $\Sigma_t$ exact | TMPD `[TAKEN]` |
| $H=0$, $\hat v_t$ a scalar from a Gaussian analysis | reward damping $\lambda_t=\lambda/(1+2\lambda\sigma^2_{1\mid t})$ `[TAKEN]` |
| $H\neq0$, $\Sigma_t$ exact | **the candidate** |

Writing the table this way is also the fastest route to the related-work section, and it makes the
ablation grid obvious: each row is an arm.

### 3.6 Proposition 6 — reparameterization consistency `[PROVED]` `[CHECKED: check_D, check_G]`

This is the part I like most, because it is a correctness test that **needs no ground truth at all**.

Let $\varphi$ be strictly increasing. The conditioning *events* coincide exactly:

$$\{x_1:f(x_1)\in[l,h]\}\ =\ \{x_1:\varphi(f(x_1))\in[\varphi(l),\varphi(h)]\}.$$

So the exact conditional distribution is **identical** whether the target is expressed through $f$ or
through $\varphi\circ f$. Dipole in Debye or its logarithm, GC fraction or GC percentage cubed —
same set of molecules, same answer. Any guidance method that returns different sample distributions
for the two is wrong by that amount, and the discrepancy is measurable by sampling alone.

| estimator | defect under nonlinear $\varphi$ |
|---|---|
| plug-in | $\Theta(\lVert\Sigma_t\rVert)$ — the missing $\tfrac12\varphi''\hat v_t$ |
| second-moment corrected | $O(\lVert\Sigma_t\rVert^{3/2})$, and $O(\lVert\Sigma_t\rVert^{2})$ for a symmetric posterior |

**Affine $\varphi$ is a null control** `[PROVED]` `[CHECKED]`: plug-in is exactly affine-covariant
provided the likelihood width is carried across with the units ($s\mapsto|a|s$). Measured bias
$+0.0545$ under $f\mapsto-1.5f+0.75$ against $+0.0526$ unreparameterized, a difference inside noise.
Forget the rescaling and the effective strength changes by $a^2$; that is a units artefact and not
the effect we study. Both behaviours are in the check, because a diagnostic that fires on a units
mistake is worthless.

`[CHECKED]` Standardized mean error, one state, four reparameterizations, four covariance levels:

| $\operatorname{tr}\Sigma_t$ | identity | $\log$ | cube | $\sqrt{\ }$ |
|---|---|---|---|---|
| 1.702 | 0.287 / 0.021 | 0.001 / 0.056 | **0.532** / 0.048 | 0.149 / 0.027 |
| 0.788 | 0.192 / 0.016 | 0.002 / 0.017 | 0.432 / 0.073 | 0.098 / 0.010 |
| 0.244 | 0.092 / 0.007 | 0.004 / 0.006 | 0.255 / 0.017 | 0.044 / 0.005 |
| 0.067 | 0.049 / 0.002 | 0.002 / 0.001 | 0.138 / 0.002 | 0.025 / 0.002 |

(plug-in / corrected, in units of the true posterior sd.) Log-log slopes: **0.425 for plug-in,
1.080 for corrected**, against predicted $1/2$ and $1$ in standardized units.

The $\log$ column is the demonstration. Here $f=\exp(a^{\!\top}x)$, so $\log f$ is linear and the
plug-in estimator is *exactly right* ($z=0.001$). The same physical quantity written as a cube makes
the same estimator wrong by half a posterior standard deviation. The exact conditional cannot depend
on which of the two a chemist happened to write down.

### 3.7 Proposition 7 — the signed prediction `[PROVED]` `[CHECKED: check_G]`

Plug-in guidance drives $f(\hat x_1)\to y^\ast$, so the achieved property converges to
$y^\ast+\tfrac12\operatorname{tr}(H\Sigma_t)$ evaluated where guidance still has authority. Hence:

> **Plug-in guidance overshoots convex properties and undershoots concave ones.**

`[CHECKED]` end to end in the toy sampler:

| property | plug-in bias | corrected bias |
|---|---|---|
| convex, $\operatorname{tr}H=+2.0$ | $+0.0526$ | $-0.0097$ |
| concave, $\operatorname{tr}H=-2.0$ | $-0.0864$ | $+0.0127$ |

The sign flips with the curvature, as predicted, and the correction shrinks both by roughly an order
of magnitude. This is a falsifiable claim about *published methods*, and it is cheap to test on
their reported numbers.

### 3.8 Proposition 8 — no guidance weight can imitate the mean correction `[PROVED]`

Worth stating as mathematics rather than leaving to the empirical control in §4, because it is the
objection every reviewer reaches for first.

Plug-in guidance with weight $w$ produces the field $w\,\frac{y^\ast-f(\hat x_1)}{s^2}\,J^{\!\top}g$.
The corrected field is $\frac{y^\ast-f(\hat x_1)-c}{s^2+\hat v}\,J^{\!\top}g$ with
$c=\tfrac12\operatorname{tr}(H\Sigma_t)$. Since both are the same direction $J^{\!\top}g$, equality
requires

$$w\ =\ \frac{s^2}{s^2+\hat v_t}\left(1-\frac{c_t}{\,y^\ast-f(\hat x_1)\,}\right).$$

The right-hand side depends on $\hat v_t$ and $c_t$, which depend on the **state**. Two consequences:

- A constant $w$ cannot reproduce it, and neither can any schedule $w(t)$, because at a fixed $t$ the
  required value still differs across samples within one batch.
- **This settles the comparison with the closest 2026 competitor by proof.** The reward-damping
  schedule $\lambda_t=\lambda/(1+2\lambda\sigma^2_{1\mid t})$ is confirmed (§11, item 2) to be a
  function of $t$ alone, derived for quadratic rewards on isotropic Gaussian targets. It is therefore
  a $w(t)$, and by the argument above it cannot reproduce the curvature correction for any tuning of
  $\lambda$ or $\sigma$. It remains a mandatory arm, but the separation is not an empirical accident.
- The variance correction alone rescales the field and so *is* partially imitable by a schedule —
  which is consistent with it being the part that overlaps with published work (§3.5), and with it
  being the weaker of the two in §4.

The mean correction is not a strength adjustment. It moves the **fixed point**: plug-in guidance is
stationary where $f(\hat x_1)=y^\ast$, the corrected version where
$f(\hat x_1)+\tfrac12\operatorname{tr}(H\Sigma_t)=y^\ast$. No rescaling relocates a fixed point.

### 3.9 What is **not** proved

No molecular improvement. Nothing here establishes better decoded diversity, validity, accuracy or
yield. The Gaussian-posterior approximation in Proposition 3 is an approximation; the dropped
gradient terms in §3.3 are dropped; the correction can be swamped by network error, decoding
discontinuity, or solver error; and a learned $\partial\hat x_1/\partial x_t$ need not be PSD. Every
one of those is a measurement on the test list, not a claim.

---

## 4. The evidence so far, and the control that matters

A toy flow-matching sampler with the **exact** velocity field of a three-mode Gaussian mixture, so
there is no network error and every difference is attributable to the guidance estimator alone. The
reference is the model's own exact conditional, obtained by rejection sampling the prior
(acceptance rate 18%, $n=20\,000$) — an unbiased target that no method can game.

`[CHECKED: check_E_sampler]` $B=4000$, target 2.0, convex property:

| arm | achieved mean | bias | $W_1$ to reference | **mode TV** | mode fractions |
|---|---|---|---|---|---|
| reference (exact conditional) | 1.986 | — | — | — | 0.357 / 0.249 / 0.394 |
| unguided | 1.876 | $-0.124$ | 0.802 | 0.077 | 0.341 / 0.326 / 0.333 |
| **plug-in guidance** | 2.053 | $+0.053$ | 0.098 | **0.282** | **0.075** / 0.379 / 0.546 |
| **second-moment** | 1.995 | $-0.005$ | 0.088 | **0.076** | 0.281 / 0.319 / 0.400 |

Read the mode fractions. The exact conditional puts 35.7% of its mass on mode 0. Plug-in guidance
puts **7.5%** there — it has all but deleted a mode that genuinely satisfies the constraint. The
correction restores it to 28.1%. The property axis barely distinguishes the two arms; the
*distributional* axis separates them by a factor of nearly four.

### The control that decides whether this is real

If a tuned scalar guidance weight reproduced the effect, the contribution would be a schedule and we
would say so. `[CHECKED: check_F_controls]`, same reference, $B=2500$:

| arm | $w$ | bias | in band | mode TV |
|---|---|---|---|---|
| plug-in | 0.125 | $+0.574$ | 0.195 | 0.192 |
| plug-in | 0.25 | $+0.450$ | 0.188 | 0.237 |
| plug-in | 0.5 | $+0.182$ | 0.765 | 0.251 |
| plug-in | 1.0 | $+0.055$ | 1.000 | 0.297 |
| plug-in | 2.0 | $+0.025$ | 1.000 | 0.321 |
| plug-in | 4.0 | $+0.016$ | 1.000 | 0.438 |
| **mean correction only** | 1.0 | $-0.013$ | 1.000 | **0.138** |
| variance correction only | 1.0 | $+0.231$ | 0.662 | 0.258 |
| **both** | 1.0 | $-0.008$ | 0.700 | **0.105** |

Three things fall out, and the second one is inconvenient in a useful way.

1. **No $w$ works.** Mode TV degrades monotonically in $w$; the best value (0.125) still only reaches
   0.192 and pays a bias of $+0.574$ with 19% in-band. There is no scalar that buys both.
2. **The mean correction is the load-bearing part, not the variance correction.** Variance-only is
   barely better than plug-in. This is the *opposite* of where the prior literature has concentrated
   — and it is the component that is identically zero for linear operators (§3.4). Convenient for
   the novelty argument, so it deserves the most adversarial testing on real data.
3. **There is a real trade-off.** Both-corrections lands 70% in band against 100% for plug-in at
   $w=1$. Matching a tolerance band is not the same as matching the conditional, and the paper has to
   report both axes rather than pick the flattering one.

**Limits of this evidence.** Two dimensions, three modes, an exact score, a quadratic property, no
network, no decoding, no chemistry. It establishes that the mechanism is real and that the obvious
control does not explain it. It establishes nothing about molecules.

---

## 5. Fixing the modality

The user's question was whether the modality should be chosen to fit the innovation. It should, and
the answer is that **the pair the team already picked is the right pair, for a better reason than the
one currently written down.** "Two different geometries" becomes "two different posterior
structures", which is exactly the axis the innovation lives on.

| | **Modality 1 — QM9, 3D coordinates** | **Modality 2 — DNA enhancers on the simplex** |
|---|---|---|
| $p(x_1\mid x_t)$ | implicit; reachable only through the denoiser | **explicit** — the model emits a per-position categorical |
| $\Sigma_t$ | Gaussian approximation via Lemma 1, one JVP | not needed |
| the correction | second-order, approximate | **exact in closed form** for any $k$-mer functional |
| what it tests | does the estimator survive a real, hard case | does the *principle* hold when the estimator is exact |

That contrast is the strongest possible transfer argument, and the modality-2 half of it is now
**verified against the primary source rather than assumed** (§11, item 6). Dirichlet Flow Matching
trains its network by cross-entropy $\mathcal L(\theta)=-\mathbb E[\log\hat p(x_1\mid x;\theta)]$ and
builds its field as $\hat v=\sum_i u_t(x\mid x_1{=}e_i)\,\hat p(x_1{=}e_i\mid x;\theta)$. The
denoising posterior over clean tokens is not something we have to estimate — **it is the model's
own output.**

So on the simplex $\mathbb E[f(x_1)\mid x_t]$ and $\operatorname{Var}[f(x_1)\mid x_t]$ are computable
*exactly* for GC content and $k$-mer frequencies: no Tweedie, no Jacobian, no Hutchinson estimator,
no Gaussian assumption, no second-order truncation. Overlapping $k$-mer windows share positions so
the variance needs cross-window covariance terms, but positions stay independent and it is still
closed form.

Two things follow that are worth more than a transfer section usually is:

- The correction on modality 2 is not "second-moment" at all, it is **exact marginalization at
  negligible cost**. If the principle is right it should work *better* here, and if it does not, the
  principle is wrong and no amount of estimator engineering on molecules will save it.
- Running both gives a direct measurement of **what the Gaussian approximation in Proposition 3
  costs**, by comparing the exact correction against the second-order one on the same modality.

There is also a sharper version of the pathology here. On the simplex, $f(\hat x_1)$ evaluates the
property of a *blurred base composition* — a point in the interior of the simplex that is not a
sequence at all. Asking a cell-type classifier for the activity of "40% A, 30% C, 20% G, 10% T at
every position" is not a slightly noisy question; it is a question with no biological referent. The
plug-in estimator is at its worst exactly where guidance has the most authority to act.

### The three-tier property ladder `[DESIGN]`

This is what makes the claim checkable rather than merely plausible, and it costs almost nothing.
For each modality, run three properties whose curvature we know in advance:

| tier | Modality 1 | Modality 2 | theory says |
|---|---|---|---|
| **linear** | molecular weight (linear in one-hot type channels) | GC content (linear in the simplex state) | bias exactly **zero** — a bug detector |
| **quadratic** | squared radius of gyration | 2-mer frequency | correction **exact**, closed-form prediction |
| **learned, nonlinear** | dipole moment via the frozen guide | cell-type probability via the frozen classifier | correction approximate — the real test |

The linear tier cannot produce a flattering result: if we measure a bias there, we have a bug. The
quadratic tier has a closed-form prediction that either matches or does not. Only the third tier is
the usual kind of experiment. A reviewer who distrusts learned oracles has two tiers that do not
involve one.

One pleasing special case: for $R_g^2$ the Hessian is a constant projector, so
$\tfrac12\operatorname{tr}(H\Sigma_t)$ collapses to a multiple of the **divergence of the velocity
field** — a single scalar, one Hutchinson probe, and a quantity people already compute for
likelihoods.

---

## 6. Novelty — the honest position

Three rounds of search, 17 Sep 2026. "Not found" means nothing surfaced in a bounded search; it is
not a priority claim, and this document should never say "first".

### Taken, and cited without any claim

| piece | owner |
|---|---|
| guidance through the clean-data estimate | DPS; FreeDoM; MPGD; TFG; TFG-Flow |
| second-order Tweedie, $\operatorname{Cov}=(\sigma^2/\alpha)\partial\hat x_1/\partial x_t$ | standard; TMPD uses it for inverse problems |
| covariance/Jacobian identity and covariance-aware flow guidance | [On the Guidance of Flow Matching (2025)](https://arxiv.org/html/2502.02150v2), Proposition 3.4 and §3.3–§3.4 |
| quadratic-energy guidance using a Hessian and Gaussian posterior approximation | [GGDOpt (2025)](https://arxiv.org/html/2510.12238v1), equation (20); compare the energy, not just the paper's symbol $f$ |
| second-order Heun integration for diffusion sampling | [Karras et al. (2022)](https://papers.neurips.cc/paper_files/paper/2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf), Algorithm 1 |
| Heun applied to rectified-flow sampling | [Lee et al. (NeurIPS 2024)](https://arxiv.org/html/2405.20320v2), section 5.3 and Algorithm 2; [Wang et al. (ICML 2025)](https://proceedings.mlr.press/v267/wang25ce.html), Appendix B.1 |
| efficient second-order Tweedie guidance through a surrogate loss | [STSL (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Rout_Beyond_First-Order_Tweedie_Solving_Inverse_Problems_using_Latent_Diffusion_CVPR_2024_paper.pdf), Theorem 4.4 and equation (6) |
| moment matching guidance with Tweedie covariance and a neglected covariance derivative | [MMPS](https://arxiv.org/html/2405.13712v5), section 4.2, equations (19)-(22) |
| the FM version via the velocity divergence | [Divergence is Uncertainty (2026)](https://arxiv.org/abs/2605.00941) — a UQ paper, not guidance |
| isotropic posterior covariance in the likelihood | ΠGDM |
| exact covariance, **linear** operators | [TMPD (Boys et al.)](https://arxiv.org/abs/2310.06721); [Optimal Posterior Covariance (2024)](https://arxiv.org/abs/2402.02149) |
| the Jensen gap as the error source, **bounded** by $\tfrac12 L\operatorname{tr}\operatorname{Cov}$ | ABMS; DPS analyses |
| DPS bias characterised as covariance coupled to reward curvature — **analysis only** | [Feynman-Kac Analysis of Bias and Stability (2026)](https://arxiv.org/abs/2605.06538) |
| scalar reward damping $\lambda/(1+2\lambda\sigma^2_{1\mid t})$ from a Gaussian analysis | [Are we really tilting? (2026)](https://arxiv.org/abs/2606.02884) |
| **Monte-Carlo** marginalization of the guidance likelihood, incl. on QM9 | [ABMS (2026)](https://arxiv.org/abs/2603.06685); [CBG (2026)](https://arxiv.org/abs/2602.22428); TDS; multilevel SMC |

Earlier notes compared two additional close predecessors. These comparisons do not establish priority:

- **Feynman-Kac (2605.06538)** expresses the DPS bias as a Radon–Nikodym path weight coupling the
  conditional covariance to the reward curvature. Verified by reading: it is an **analysis**. It
  proposes no corrected estimator, no $\tfrac12\operatorname{tr}(H\Sigma)$ term and no variance
  inflation; its algorithm boxes are DPS itself and early guidance stopping. It is the best possible
  citation for our motivation — an independent theory paper saying DPS is biased in exactly this way.
- **ABMS (2603.06685)** averages the property over $M$ draws from the one-step denoising posterior,
  and **runs on QM9 quantum-property targeting**. Verified: its curvature/covariance appearance is a
  *bound* in Proposition 1, not an implemented correction. It is a mandatory baseline and the natural
  head-to-head: Monte-Carlo marginalization at $M=3$ costs roughly $2.8\times$ inference time by its
  own table. SMG's full cost includes property derivatives, covariance actions, trace probes,
  and guidance evaluation; its speed advantage has not been measured.

### Candidate contribution and evidence still needed — revised 18 Sep

| # | item | status |
|---|---|---|
| 1 | the estimator of §3.3, with the **curvature term**, for nonlinear scalar properties | candidate incremental method; specific priority and practical benefit unestablished |
| 2 | Corollary 4 — the mean correction vanishes for affine properties | standard consequence of zero Hessian; explains a shared special case, not an independent innovation |
| 3 | affine reparameterization consistency (§3.6) | standard sanity check; consistent plug-in guidance also passes when target widths are transformed |
| 4 | rejection sampling as a conditional reference | standard reference technique; must use the same likelihood as the guided sampler |
| 5 | mean correction outperforming variance correction | preliminary toy observation; matched-target and molecular tests remain necessary |

Item 1 is the main research hypothesis. Items 2-4 support explanation and evaluation. Item 5
could become an empirical contribution if it survives the corrected comparisons.

**Current assessment.** The constituent mathematics is established. The particular combination
may be a useful adaptation, but a different displayed formula is not sufficient evidence of
research novelty. Keep SMG as the primary method under evaluation; do not call its novelty or
publication prospects established. Heun supplies a solver control, not an additional innovation.

---

## 7. How it maps onto the required five questions

The assignment wants a sequence of questions answered. This innovation happens to answer them in
order rather than around them.

| assignment question | how it is answered here |
|---|---|
| which base model to build on | Lemma 1 holds for both families with only $(\alpha_t,\sigma_t)$ changing, so FM vs diffusion becomes a test of whether the *same* corrected guidance transfers across families at matched SNR — a mechanism experiment, not a formality |
| does guidance work | yes, and we additionally show *what it gets wrong*, against the model's own exact conditional |
| does the proposed change help | ablation is already prototyped: mean-only, variance-only, both, tuned-scalar, isotropic, MC-marginalization at matched cost |
| competitive with recent work | TFG-Flow, ABMS, MolGuidance, PropMolFlow, ΠGDM-style, reward damping — six, need three |
| does it transfer | modality 2 runs the **exact** version of the same correction (§5) |

---

## 8. Risks, and the gates that kill it cheaply

| risk | why it might happen | gate | cost |
|---|---|---|---|
| $\operatorname{tr}(H\Sigma_t)$ is negligible on the real checkpoint | guidance applied late only, or low property curvature | measure $\tfrac12\operatorname{tr}(H\Sigma_t)$ against $t$ before anything else | ~2 h |
| the learned Jacobian is not PSD | network, not mathematics | measure $g^{\!\top}\Sigma_t g<0$ frequency | ~1 h |
| the dropped gradient terms (§3.3) matter | third derivatives | finite differences on a diagnostic subset | ~2 h |
| Hutchinson variance eats the correction | $\operatorname{tr}(HJ)$ needs probes | variance of the estimator vs probe count; use analytic $H$ on tiers 1–2 | ~1 h |
| network error swamps it | real checkpoints are not exact scores | the tier-1 and tier-2 properties have closed-form predictions and will say so | free |
| ABMS wins at matched wall-clock | MC marginalization is unbiased, ours is $O(\Sigma^{3/2})$ | equal-wall-clock comparison, mandatory from day one | — |
| rejection reference too expensive | rare targets | stratify targets by rarity; report where the reference is affordable and use the §3.6 diagnostic where it is not | — |

**Every outcome is reportable.** If the correction is negligible, that is a measured statement about
when plug-in guidance is safe, with a formula for it. If MC marginalization wins at matched compute,
that is an efficiency result. If the mean correction fails to reproduce §4 on real molecules, the
tier-1/tier-2 ladder will say whether it is the theory or the implementation.

---

## 9. What happens to the existing work

Component D and Balanced Preview Sampling do not need to be thrown away, and §4 gives them a better
home.

The team's original motivation was "guidance collapses diversity, so add repulsion." Check E suggests
a large part of that collapse is **estimator bias, not a property of conditioning** — the corrected
sampler recovers the deleted mode with no repulsion term at all. That reframes the old work rather
than deleting it:

- It **dissolves the circularity problem** that `PROJECT_GUIDE.md` Part 3 spends five pages on. With
  no repulsion term there is nothing to accuse us of optimizing, and the target is agreement with a
  reference we did not choose.
- **Component D becomes a comparison arm**: repulsion-based diversity repair, against bias-removal
  based diversity repair. If D still adds something on top of unbiased guidance, that is a result. If
  it does not, that is also a result, and a cleaner one than the current plan can produce.
- The tolerance band, the ball/slab solver and their proofs stay correct and stay in `audit/`. They
  are simply no longer the headline.
- BPS is unaffected and remains parked.

The evaluation harness, the $f_A$/$f_B$ split, the within-condition protocol, the attempts-not-
survivors denominator and the DFT plan all carry over unchanged.

---

## 10. Checks performed

All in `audit/marginalized_guidance_checks.py`; results in
`audit/marginalized_guidance_results.json`.

| # | check | result |
|---|---|---|
| A | second-order Tweedie identity, 400 random cases | max rel. error $8.1\times10^{-10}$ |
| B | mean bias equals $\tfrac12\operatorname{tr}(H\Sigma)$; zero for linear, exact for quadratic | predicted 0.65651 vs measured 0.65667 |
| C | error order in $\lVert\Sigma\rVert$, exact closed-form truth | mean slope **0.999 → 1.997**; variance slope **1.022 → 2.005** |
| D | reparameterization consistency, 4 maps × 4 covariance levels | standardized slope **0.425 → 1.080** |
| E | toy sampler vs the model's own exact conditional | mode TV **0.282 → 0.076** |
| F | tuned-scalar control and mean/variance ablation | no $w$ matches; mean correction dominates |
| G | signed bias prediction, plus the affine null control | sign flips with curvature; affine control inside noise |

---

## 11. Open reading — **(verify)** before any of this enters the paper

House rule from `PROJECT_GUIDE.md` §8.2: anything that came from a search summary rather than the
primary text is marked and checked before it is quoted. Current list.

| # | claim | status |
|---|---|---|
| 4 | TMPD is restricted to linear operators — **Corollary 4 rests on this** | ✅ **verified 17 Sep.** Observation model is $y=Hx_0+u$, $u\sim\mathcal N(0,\sigma_y^2I)$, and the likelihood is $p(y\mid x_t)\approx\mathcal N(y;Hm_{0\mid t},HC_{0\mid t}H^{\!\top}+\sigma_y^2I)$ — the covariance enters the **variance only**, with the conditional mean passed straight through $H$. They state the limitation themselves: linear maps "do not suffice in the latent diffusion setting" |
| 6 | Dirichlet FM's denoiser emits a per-position categorical over clean bases — **all of §5's modality-2 argument rests on this** | ✅ **verified 17 Sep.** The network is trained by cross-entropy $\mathcal L(\theta)=-\mathbb E[\log\hat p(x_1\mid x;\theta)]$ and the field is built as $\hat v=\sum_i u_t(x\mid x_1{=}e_i)\,\hat p(x_1{=}e_i\mid x;\theta)$. The denoising posterior over clean tokens **is literally the model's output** |
| 2 | reward damping is $\lambda_t=\lambda/(1+2\lambda\sigma^2_{1\mid t})$ | ✅ **verified 17 Sep.** Proposition 4 of 2606.02884, with $\sigma^2_{1\mid t}=\sigma^2(1-t)^2/((1-t)^2+t^2\sigma^2)$. Confirmed to be a **scalar function of time alone**, derived for isotropic Gaussian targets and quadratic rewards $r(x)=-\lVert x-a\rVert^2$, with $\sigma$ tuned in practice. No curvature term anywhere |
| 1 | ABMS costs ~2.8× inference at $M=3$ (4.8 → 1.7 it/s) | search summary of its Table 4 — read the table, and re-time it ourselves regardless |
| 3 | ΠGDM uses $r_t^2=\sigma_t^2/(\sigma_t^2+\alpha_t^2)$ | from memory. Read the paper; it is an ablation row |
| 5 | the Feynman-Kac paper proposes no corrected estimator | read via HTML, one pass. Wants a second reader |
| 7 | QM9 rejection-sampling acceptance rate at useful dipole targets | unknown. Measure; it decides item 4 of §6 |

The two load-bearing items were checked first and both held. Item 2 held in a way that helps:
because the damping schedule is confirmed to be a function of $t$ alone, **Proposition 8 separates us
from the closest 2026 competitor by proof rather than by measurement.**

---

## 12. What I want torn apart

1. **The nonlinear-property guidance rule needs a novelty audit.** Corollary 4 itself is a
   standard affine special case. Check whether prior methods already incorporate
   $\tfrac12\operatorname{tr}(H\Sigma_t)$ in the predicted property mean and use the resulting
   residual with uncertainty scaling. If so, describe SMG as an adaptation; any evaluation
   contribution would still need substantial results beyond standard consistency checks.
2. **Is the rejection-sampling reference affordable at our targets?** 18% acceptance in a toy means
   nothing. If the QM9 acceptance rate at a useful dipole target is $10^{-4}$, item 4 is restricted
   to common targets and the §3.6 diagnostic carries the rest.
3. **Is the mean-over-variance finding an artefact of the toy's curvature?** $\operatorname{tr}H=2$
   with $\operatorname{tr}\Sigma\approx1$ is a large correction by construction. On QM9 the ratio may
   be tiny, and the first gate in §8 exists to find that out in two hours.
4. **Am I over-reading check E?** The mode-deletion result is the most striking number in this
   document and therefore the one most likely to be a toy artefact. Three modes and a quadratic
   property is a friendly setting.

---

## 13. Required Euler/Heun comparison — added 18 Sep

Heun is `[TAKEN]`, a solver control, and can be combined with either plug-in guidance or SMG.
For a full guided ODE $x'=F(x,t)$ and signed step $h$:

$$\boxed{\begin{aligned}
k_1&=F(x_n,t_n),&\widetilde x&=x_n+hk_1,\\
k_2&=F(\widetilde x,t_n+h),&x_{n+1}&=x_n+\tfrac h2(k_1+k_2).
\end{aligned}}$$

Recompute the entire guidance at stage 2. For independent-Gaussian linear FM,
$F=v_\theta+w(t)(1-t)G/t$ with increasing $t$. For diffusion's probability-flow ODE,
$F=b-\kappa^2(s_\theta+G)/2$ with decreasing forward-noising time. The derivation,
endpoint rules and comparison to prior formulas are in
[SMG_PRIOR_WORK_AUDIT.md](SMG_PRIOR_WORK_AUDIT.md).

Run at least the **plug-in/SMG × Euler/Heun** grid, plus unguided controls, at matched field
evaluations and measured wall time; see [M-4](../../archive/FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md#m-4-eulerheun--guidance-comparison--required-solver-control).
Keep the same target definition in every reference: the old toy's hard-band rejection target
differs from its Gaussian guidance likelihood. Its historical results are retained, but they
do not prove agreement with the exact distribution targeted by that likelihood.
