# The two-moment closure is invalid where a learned guide is off-distribution

**21 September 2026. A measured negative result, and the strongest finding the project has so
far.** It is not a compute problem, and more compute makes it worse.

---

## 1. The claim

Every closure in this space -- DPS, TMPD, MMPS, SMG, QPMG, PiGDM -- replaces the intractable
posterior expectation of the property by a two-term Taylor expansion about the posterior mean:

$$\mathbb{E}[f(X)\mid x_t] \;\approx\; f(m) + \tfrac12\operatorname{tr}(H\Sigma_t) \;=\; f(m) + c$$

**Nobody measures whether that approximation holds on their own checkpoint.** With a learned
denoiser there is no exact reference, so the assumption is inherited rather than tested.

It can be tested, cheaply, by comparing all three quantities directly:

| quantity | what it is |
|---|---|
| $f(m)$ | what plug-in guidance assumes |
| $f(m) + c$ | what every two-moment closure assumes |
| $\mathbb{E}[f(m+\delta)]$, $\delta\sim\mathcal N(0,\Sigma_t)$ | **the truth** |

**Result: on our flow-matching checkpoint with an EGNN dipole guide, the second-order term makes
the estimate worse for $t < 0.75$, by up to a factor of 19.**

---

## 2. The measurement

`results/bench/quad_validity.py`. Generator `fm_v1` (epoch 1500 EMA), guide `f_A_mu` (EGNNScalar
128x4, val MAE 0.0897 D), 16 real `train_a` molecules noised to each $t$, 256 posterior draws per
state, and $c$ estimated with 64 Hutchinson probes so that estimator variance is **not** the
explanation.

| $t$ | $f(m)$ | $\mathbb{E}[f(m+\delta)]$ | $f(m)+c$ | linear err | quadratic err | sd of $f$ | verdict |
|---|---|---|---|---|---|---|---|
| 0.20 | **-14.0** | 6.64 | 191 | 26.6 | 501.7 | 15.5 | quadratic **19x worse** |
| 0.30 | **-4.62** | 7.95 | 129.6 | 12.6 | 221.3 | 12.0 | 17.6x worse |
| 0.40 | 1.66 | 8.13 | 37.8 | 7.00 | 51.1 | 8.85 | 7.3x worse |
| 0.50 | 3.50 | 7.73 | -13.5 | 4.25 | 22.2 | 6.44 | 5.2x worse |
| 0.60 | 3.12 | 6.77 | 2.72 | 3.65 | 8.57 | 4.14 | 2.3x worse |
| 0.70 | 3.06 | 4.17 | 0.87 | 1.12 | 3.52 | 1.42 | 3.1x worse |
| **0.75** | 2.56 | 3.03 | 2.72 | 0.475 | **0.311** | 0.638 | **helps, 1.53x** |
| 0.80 | 2.55 | 2.75 | 2.59 | 0.204 | 0.172 | 0.411 | helps 1.18x |
| 0.85 | 2.58 | 2.68 | 2.59 | 0.0959 | 0.0883 | 0.286 | helps 1.09x |
| 0.90 | 2.60 | 2.64 | 2.61 | 0.0434 | 0.0382 | 0.187 | helps 1.14x |
| 0.95 | 2.62 | 2.63 | 2.62 | 0.0139 | 0.0117 | 0.0954 | helps 1.19x |
| 0.98 | 2.63 | 2.64 | 2.63 | 0.00408 | 0.00360 | 0.0430 | helps 1.13x |

**The crossover is sharp and sits at $t = 0.75$ for this guide.**

### It is guide-dependent, and that is part of the finding

The same measurement, same generator, same property, with a **different guide architecture**:

| guide | arch | clean-data MAE | crossover $t$ | worst ratio |
|---|---|---|---|---|
| `f_A_mu` | EGNN 128x4 | 0.0897 D | **0.75** | 19x worse at $t=0.2$ |
| `f_A_mu_transformer` | distance-bias transformer | 0.120 D | **0.95** | 12.5x worse at $t=0.4$ |
| `f_A_alpha_transformer` | transformer | 0.288 a.u. | **0.75** | 53x worse at $t=0.3$ |

Three points follow.

**The failure is general, the crossover is not.** Every guide tested breaks the closure over most
of the trajectory, but where it recovers differs by a factor of several in $1-t$. A late-$t$
guidance window is therefore **not a constant that can be looked up** -- it has to be measured per
guide, per property. That measurement is the instrument this project can contribute.

**Clean-data accuracy does not predict off-distribution curvature.** The transformer is only
slightly worse than the EGNN on held-out clean molecules (0.120 vs 0.0897 D) yet its closure is
invalid over a much wider range. Choosing a guide by its validation MAE says nothing about whether
a two-moment closure built on it will work.

**The effect is larger on polarizability than on dipole** (53x vs 19x), so it is not an artefact of one
property's scale.

### The mechanism is visible in the table

At $t = 0.2$ the guide returns $f(m) = -14.0$ **for a dipole moment**, which is physically
impossible: $\lvert\mu\rvert \ge 0$ by definition. $f_A$ was trained on clean QM9 molecules; at
small $t$ the posterior mean is a blurred, non-physical average of many molecules, and the network
is extrapolating far outside anything it has seen. Its value there is meaningless, and its
**curvature is worse than meaningless** -- $c$ reaches $+191$ against a true expectation of $6.6$.

This is a statement about the *guide*, not about the generator or the closure algebra.

---

## 3. Why no estimator can fix it

The distinction matters because it decides whether this is an engineering problem or a research
finding.

**Variance would be fixable.** Hutchinson's estimator has relative error $O(1/\sqrt{K})$ in the
probe count $K$. `results/bench/probe_noise.py` measures it: at $K=1$ the within-state estimator
noise *exceeds* the across-state signal at $t=0.6$ and $t=0.9$, and $K\approx 32$ is needed before
the estimate is signal-dominated. Hutch++, XTrace, antithetic probes or a learned control variate
would all reduce that further.

**Validity is not fixable.** The table above uses $K=64$, so $c$ is well estimated -- and the
answer is still worse than ignoring $c$ entirely. A better estimator converges *faster to a value
whose use is invalid*. Spending compute here buys precision on the wrong quantity.

The cost of pretending otherwise is large: matching SMG's correction to the residual would need
roughly **1,570 probes at $t=0.3$**, which at measured rates is hundreds of GPU-hours for a term
that should not be used at that $t$ at all.

---

## 4. What follows

**1. SMG and every two-moment closure are defensible only for $t \gtrsim 0.75$ on this guide.**
The honest way to run them is with a **late-$t$ guidance window**, disclosed, rather than over the
whole trajectory. Within that window the correction is real but modest (1.1-1.5x better than
linear).

**2. The Monte-Carlo arms are the principled answer, not the cheap approximation.** `lgd_mc`,
`osc` and `tfg_mc` evaluate $f$ at *actual perturbed points* and integrate the property's law
directly. They never form $H$, are immune to this failure by construction, and cost 125-273
ms/molecule against 800-2500 for the SMG family (`results/bench/time_guided.py`). This is very
likely why TFG-Flow chose Monte-Carlo smoothing over a curvature correction.

**3. It reframes the contribution.** "Our closure is better" was already a crowded claim, and
[Gleich and Schmidler (Jan 2026)](https://arxiv.org/abs/2601.21104) now occupy the posterior-shape
problem with an asymptotically exact SMC estimator. The claim this measurement supports is
different and defensible:

> **Measure when a closure suffices, and use it only there.** We show that the two-moment closure
> underpinning a family of published guidance methods fails on a learned property network over
> most of the sampling trajectory, identify the mechanism (the guide is evaluated off its training
> distribution), locate the crossover, and show that sampling-based estimators do not share the
> failure.

That is the frame `TOP6_GUIDANCE_RECOMMENDATIONS.md` section 9 argues is the only one in which a
cheap method can legitimately beat an exact one.

---

## 5. Limits of this result, stated plainly

- **Three guide/property combinations, one generator.** Measured on `f_A_mu` (EGNN),
  `f_A_mu_transformer` and `f_A_alpha_transformer`, all against `fm_v1`. **Not yet
  checked:** the EGNN guides for polarizability and gap (trained on the cluster, not
  yet pulled locally), any gap guide, SchNet at 0.0210 D, and the diffusion base
  model. SchNet is the most valuable remaining test: it is 4x more accurate than
  anything here and a different architecture again, so if it shows the same failure
  the result is about learned property networks in general, not about ours.
- **16 molecules, 256 draws.** Enough for a 19x effect, not enough for the 1.1x effects at large
  $t$; those need a larger sample before they are quoted.
- **The draws are trace-matched isotropic, not $\Sigma^{1/2}$.** A single JVP gives $\Sigma v$,
  never its square root, so the posterior spread is matched in trace but not in shape. An
  anisotropic draw (Lanczos or Chebyshev) could shift the crossover.
- **$c$ is Hutchinson-estimated at $K=64$**, not exact. The residual estimator noise at large $t$
  is of the same order as the 1.1x improvements there.
- **This says nothing about terminal sample quality.** Field accuracy is not sample accuracy;
  `TOP6_GUIDANCE_RECOMMENDATIONS.md` section 8 shows methods an order of magnitude apart in field
  error can land within noise of each other after a full sampling run. The guidance grid is what
  settles that.

## 6. Reproduce

```
python results/bench/quad_validity.py     # the table above
python results/bench/probe_noise.py       # variance vs signal, per t and probe count
```
