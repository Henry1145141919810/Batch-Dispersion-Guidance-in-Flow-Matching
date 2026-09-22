# Two proposed guidance methods, derived from what the sweep measured

**21 September 2026.** Both attack the one gap every measurement points at: **guidance solves the
mean and does nothing about the spread.** Both use a quantity our existing arms already compute
and throw away. Novelty is assessed honestly in Section 5 -- one is narrow, one is moderate, and
neither is claimed as first-of-its-kind.

---

## 1. The evidence these are built on

From `FINDINGS_SO_FAR.md`, all measured on `fm_v1` with f_B scoring:

| observation | number |
|---|---|
| bias of the best arm on mu | **0.0444** against a target of 2.4932 -- the mean is solved |
| spread of the same arm | **0.9936**, i.e. ~95% of the MAE |
| spread as a fraction of the raw data spread | **65-105%** -- guidance barely narrows, and sometimes widens |
| `plug` on alpha | spread **105%** -- *wider than the data it was trained on* |
| delta (the band we score against) | **6x tighter** than the achievable spread |
| arms targeting `Var[f(X)|x_t]` | **zero** |

And the quantity needed is already in hand: **`V_F = g' Sigma g` is exactly the property's
variance at state `x_t`**, computed by every SMG-family arm and used only as a denominator.

Its measured trajectory (sd of `f` over the posterior, mu, `f_A_mu`):

| t | 0.20 | 0.50 | 0.70 | 0.75 | 0.90 | 0.98 |
|---|---|---|---|---|---|---|
| sd of `f` | 15.51 | 6.44 | 1.42 | 0.638 | 0.187 | 0.043 |

**The property is decided between t = 0.5 and t = 0.75.** Before that it is undetermined; after
that it is fixed and guidance can only damage the molecule. That single fact motivates Proposal B.

---

## 2. Why the standard objective does not concentrate -- the derivation

Every arm uses the Gaussian-observable closure `F = f(X) | x_t ~ N(mu_F, V_F)` and the likelihood

```
p(y | x_t) = N(y ; mu_F, S),        S = s^2 + V_F
```

Guidance is `grad_x log p`, which has **two** terms:

```
grad log p = (y - mu_F)/S * grad(mu_F)                           <- the mean term
           + 1/2 * [ (y - mu_F)^2 / S^2  -  1/S ] * grad(V_F)    <- the variance term
```

Three things follow, and they explain the measurements exactly.

**(a) Almost every method drops the second term.** `plug`, `tmpd`, `smg`, `smg2` keep only the
mean term. Only `osc` (`cov_term`) and `smg2_curv` (the Stein direction) include a version of it.
So the variance is not merely untargeted -- for most arms it is not even differentiated.

**(b) When it is kept, its sign is wrong for our purpose.** The coefficient is positive when
`(y - mu_F)^2 > S`, and ascending `log p` then **increases** `V_F`. The likelihood hedges: when
the target is far, a wider property distribution puts more mass on it. That is correct inference
and exactly wrong for design. It is consistent with `plug` reaching **105% of the data spread** on
alpha.

**(c) The tolerance `s` was set 6-17x too wide** (Gap 0), which pushes `S` up, shrinks the mean
term, and makes the whole objective nearly indifferent to where the sample lands.

**The fix is not a better estimator of the same objective. It is a different objective.**

---

## 3. Proposal A -- Band-Targeted Variance Guidance (BTVG)

### 3.1 The objective

Replace "maximise the likelihood of `y*`" with "**make the property distribution equal to the
band**". Target `N(y*, r^2)` where `r` is the tolerance you are willing to accept, and descend

```
L_BTVG(x_t) = KL( N(mu_F, V_F) || N(y*, r^2) )
            = 1/2 [ (V_F + (mu_F - y*)^2)/r^2  -  1  +  ln(r^2 / V_F) ]
```

### 3.2 The guidance field

```
-grad L = -(mu_F - y*)/r^2 * grad(mu_F)                <- mean term, constant gain 1/r^2
          - 1/2 (1/r^2 - 1/V_F) * grad(V_F)            <- variance term, targets V_F -> r^2
```

then pulled back through `J^T` exactly like every other arm, so it shares the existing plumbing.

**Why this concentrates where the likelihood does not:**

| | likelihood objective | BTVG |
|---|---|---|
| mean gain | `1/S`, shrinks as `V_F` grows | `1/r^2`, constant |
| variance coefficient | `1/2[(y-mu_F)^2/S^2 - 1/S]` -- sign depends on the **residual** | `-1/2(1/r^2 - 1/V_F)` -- sign depends only on `V_F` vs `r^2` |
| behaviour when far from target | **widens** the distribution | still shrinks toward `r^2` |
| behaviour when `V_F < r^2` | keeps shrinking | **stops**, and gently re-widens |

The last row matters: BTVG is **self-limiting**. It drives `V_F` to `r^2` from either side and
does not over-concentrate, which is the failure mode that would collapse diversity and destroy
molecules.

### 3.3 Cost, and why it is cheap

`V_F = g' Sigma g` is a scalar already computed by `smg_var`/`smg`/`smg2`. The new object is
`grad_x V_F`, one reverse pass over that scalar -- **and we already have the code**:
`guidance.py::kappa3_skew` computes `grad_x (k * g' (dm/dx) g)` for the skew probe. BTVG reuses it.

| arm | measured ms/molecule |
|---|---|
| `plug` | 95.0 |
| `tmpd` | 241.8 |
| **BTVG (projected)** | **~260-280** |
| `smg2` | 802.7 |
| `smg2_curv` | 836.4 |

So it sits at TMPD's price, a third of `smg2`, and **needs no Hessian at all** -- which is what
makes it immune to the failure in `FINDING_QUADRATIC_CLOSURE_VALIDITY.md`. No `H`, no
`tr(H Sigma)`, no probe count.

### 3.4 Implementation

New mode `"btvg"` in `guidance_field`:

1. `post_fn` forward -> `m`, `k`            (already there)
2. `property_derivs` -> `f(m)`, `g`          (already there)
3. `sigma_times_vector(g)` -> `Sigma g`, and `V_F = g . Sigma g`   (already there)
4. `grad_x V_F` by one VJP of the scalar     (reuse `kappa3_skew`'s inner block)
5. weights `w_mu = -(f(m) - y)/r^2`, `w_V = -1/2 (1/r^2 - 1/V_F)`
6. `_pullback(w_mu * g)` plus the `V_F` gradient, which is already in `x_t` space

Roughly 40 lines, all from existing pieces. `r` defaults to `delta = 2 x f_B MAE` so the method
targets exactly the band it is scored against -- the mismatch of Gap 0 disappears by construction.

### 3.5 Ablations (the innovation must be ablated)

| arm | mean term | variance term | isolates |
|---|---|---|---|
| `plug` | likelihood | -- | baseline |
| `btvg_mean` | **BTVG** (`1/r^2` gain) | -- | is the constant gain alone the win? |
| `btvg_var` | likelihood | **BTVG** | is the variance term alone the win? |
| `btvg` | BTVG | BTVG | the proposal |
| `osc` | likelihood | likelihood (sampled) | the published version of a variance term |

That ladder separates "targeting the band" from "shrinking the variance" and from "doing both".

### 3.6 Go / no-go

**Go** if `btvg` reduces **spread / data-std** by >= 10 points against `plug` at matched molecule
stability, on mu at q50 and q90. Spread is the metric, not MAE -- MAE improves trivially if bias
improves, and bias is already solved.
**Stop** if `btvg_var` alone is null, i.e. the whole effect is the `1/r^2` mean gain. Then the
finding is "the tolerance was mis-set" (Gap 0), which is a fix, not a method.

---

## 4. Proposal B -- Resolution-Weighted Guidance (RWG)

### 4.1 The observation

The property is **decided between t = 0.5 and t = 0.75** (Section 1). Guidance applied when
`V_F` is already small cannot change the property -- the sample is committed -- but it *can* still
move the coordinates and break the molecule. That is precisely the trade-off measured in
`FINDINGS_SO_FAR.md` Section 3.3: narrowing the window monotonically improves stability and
worsens targeting.

Both ends of that trade are wasteful. Guidance should be spent where the property is **still
resolvable**, and `V_F` measures exactly that, per state, for free.

### 4.2 The method

Multiply any arm's field by a measured, per-state weight:

```
w(x_t) = w_0 * V_F(x_t) / (V_F(x_t) + r^2)
```

| regime | `V_F` vs `r^2` | weight | meaning |
|---|---|---|---|
| early | `V_F >> r^2` | -> `w_0` | property undetermined, push hard |
| resolving | `V_F ~ r^2` | `~ w_0/2` | the decisive window |
| committed | `V_F << r^2` | -> 0 | already inside the band; stop, and stop breaking molecules |

It is **not a time schedule**. Two samples at the same `t` get different weights if one has
resolved its property and the other has not. It is also **composable**: it multiplies `plug`,
`osc`, `lgd_mc`, `btvg` -- anything.

### 4.3 The estimator-switching variant (the time-interval combination)

Our own validity measurement gives a per-guide crossover `t*` (0.75 for `f_A_mu`, 0.95 for the
transformer guide). That licenses a **measured** switch rather than a guessed one:

```
t <  t*   ->  a sampling estimator (osc / lgd_mc) -- no H, immune to the closure failure
t >= t*   ->  the analytic closure (smg2) -- valid there, and cheaper than K guide calls
```

**Honest caveat, and it is the important one:** our sweep shows that above `t = 0.75` *all arms
converge* (alpha: 6.5849-6.6025, a 0.3% spread). So the late branch may be worth nothing, and the
switch would reduce to "use the MC arm in the resolving window and stop". **That is a legitimate
outcome and the experiment should be allowed to reach it.** RWG's weight already implements
"stop" automatically; the switch is the part under test.

### 4.4 Cost

`V_F` is already computed by every SMG-family arm. For `plug`/`tfg_mc`, which do not compute it,
RWG adds one JVP: `plug` 95 -> ~150 ms/molecule. The weight itself is arithmetic.

### 4.5 Ablations

| arm | what it isolates |
|---|---|
| `X` at fixed `w` | the baseline arm |
| `X` + fixed time window `[t_lo, t_hi]` | is a *time* schedule enough? (this is prior art -- TFG) |
| `X` + RWG weight | does the **per-state** weight beat the time schedule? |
| `X` + RWG + estimator switch at measured `t*` | does switching add anything over weighting? |

The second row is the one that matters: **if a fixed time window does as well, RWG is not a
contribution**, because strength schedules over `t` are established prior art.

### 4.6 Go / no-go

**Go** if RWG beats the best fixed window at matched compute on the **stability-at-fixed-MAE**
frontier -- i.e. it buys molecular quality without losing targeting.
**Stop** if a two-parameter time window `[t_lo, t_hi]` matches it. Then the per-state measurement
is decoration.

---

## 5. Novelty, assessed honestly

Searched: training-free guidance targeting observable variance; moment matching at inference;
timestep-scheduled guidance and estimator switching.

### What is established prior art

- **Guidance strength schedules over `t`.** TFG's design space tunes `s_rho(t)`, `s_mu(t)`
  explicitly; the literature on coarse-to-fine guidance allocation is substantial. **RWG's
  time-interval framing is NOT novel** -- only the per-state, measured weight is arguably new, and
  narrowly so.
- **A `grad V_F` term inside the likelihood.** Present in `osc` (`cov_term`) and in the Stein
  direction of `smg2_curv`, and in TMPD/MMPS lineage. **BTVG's variance term is not the first
  derivative of `V_F` to appear in guidance.**
- **[MMD Guidance (2026)](https://arxiv.org/html/2601.08379v2)** matches a generated *distribution*
  to a reference set and states it can reproduce target variance. Checked directly: it needs a
  **batch at inference and reference samples**, and does **not** target the variance of a scalar
  observable. Different mechanism, different requirements.
- **[Stein Diffusion Guidance (2025)](https://pith.science/paper/2507.05482)** minimises a KL to
  the true posterior with a Stein correction. Related in spirit; it corrects the *posterior*, not
  the *observable's* moments toward a chosen band.
- **[Multilevel/SMC guidance (Jan 2026)](https://arxiv.org/html/2601.21104)** is asymptotically
  exact and occupies the accuracy frontier. Neither proposal here should be pitched as beating it
  on accuracy -- only on cost-per-success.

### What appears to be new, stated narrowly

> **BTVG:** replacing the likelihood objective with a KL to a *band* `N(y*, r^2)`, so that the
> variance term's sign is set by `V_F` versus the tolerance rather than by the residual -- giving
> a self-limiting concentration force that the likelihood objective cannot produce, because the
> likelihood provably **widens** the observable when the target is far.

The sharp, checkable claim is the sign argument in Section 2(b), and the fact that **no existing
arm has a term that concentrates when the residual is large.** That is verifiable in code, not
just in prose.

> **RWG:** weighting guidance by the *measured, per-state* remaining property variance
> `V_F/(V_F + r^2)` rather than by `t`.

**Novelty rating: BTVG moderate, RWG narrow.** Both need a full citation trace before any claim
appears in the write-up. Neither is claimed as first-of-its-kind.

### Why this is a better story than SMG regardless

SMG's claim was "a better estimate of `E[f]`". We measured that its correction is
**anti-correlated** with the quantity it estimates (`corr = -0.44` at `t = 0.3`) and that the
whole closure is invalid below `t = 0.75`. BTVG does not estimate `E[f]` better -- it **changes
what is being optimised**, from a quantity that is already solved (bias 0.044) to the one that is
not (spread, ~95% of the error). Even a null result is informative, because it would establish
that the spread is irreducible by inference-time guidance, which is itself worth reporting.

---

## 6. What would falsify each

| proposal | falsified if |
|---|---|
| BTVG | `btvg_var` alone is null -- the effect is entirely the `1/r^2` mean gain, i.e. Gap 0 was the whole story |
| BTVG | spread does not fall below `plug`'s at matched molecule stability |
| BTVG | it concentrates but collapses diversity (uniqueness drops sharply) -- concentration bought by mode collapse is not a win |
| RWG | a two-parameter fixed time window matches it |
| RWG | `V_F` is too noisy per-state at `n_probe = 1` to weight with (measure this first) |
| both | the `s` sweep (Gap 0) alone closes the concentration gap -- then a mis-set parameter was the finding |

**Run the `s` sweep before either.** It is one axis over cells we can already produce, it is the
cheapest possible test of the same hypothesis, and if it closes the gap then neither proposal is
needed. That ordering is deliberate: the cheap experiment that can kill the expensive idea goes
first.

---

## 7. Suggested sequence

1. **`s` sweep** over `{delta, 2 delta, y_std/4, y_std}` -- ~2 h, and it can falsify both proposals.
2. **Measure `V_F` stability per state** at `n_probe` 1 / 4 / 16 -- ~30 min, gates RWG.
3. **Implement BTVG** (~40 lines, reuses `kappa3_skew`) and run the Section 3.5 ladder at n = 512.
4. **RWG as a multiplier** on the best arm from (3), against a fixed time window.
5. Only then, the estimator switch at the measured `t*`.

Steps 1 and 2 are gates: both are cheap and either can stop the programme before the expensive
parts.
