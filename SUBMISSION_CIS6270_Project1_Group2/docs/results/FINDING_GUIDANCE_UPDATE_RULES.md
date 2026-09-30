# Momentum, Adam and Muon as the inference-time guidance update rule

**22 September 2026. Null result, with one finding worth keeping.**

Treat the 100 sampling steps as an optimisation trajectory and the per-step guidance
direction `G` as its gradient. Then the shipped sampler is running plain first-order
descent on that signal, and the obvious question is whether the standard
preconditioners — momentum, Adam, Muon — do better. They do not. **All four variants
tie with the existing rule on every property**, on MAE and on the project's own band
coverage rubric alike.

**The mechanism is measured, not inferred.** These rules re-point the guidance vector,
and the guidance vector is only ~7 % of the sampling velocity. Rotating it by 67°
changes the actual sampling direction by **3.7°**, falling to **0.5°** over the last
third of the trajectory. No update rule that only changes direction can move this
sampler, and no sample size would have shown otherwise.

Nothing here touches the base model. This is inference-time only.

---

## 1. What was changed, and what was not

One insertion point: `_Base._guide` in `proj1/src/sampling.py`, where `guidance_field`
returns its direction. The rule maps that direction to the direction actually applied,
and everything downstream — the `(1-t)/t` score-to-velocity conversion, the strength
`w`, the velocity-relative clip, the solver, the endpoint policy — is untouched.

| rule | update | E(3)-equivariant? |
|---|---|---|
| `euler` | `D = G` — the shipped first-order rule, the control | yes |
| `momentum` | `D = m̂`, `m = β₁m + (1−β₁)G`, bias-corrected | yes |
| `adam` | `D = m̂ / (√v̂ + ε)`, elementwise second moment | **no** |
| `adam_eq` | as `adam`, but coordinates get ONE scalar per atom | yes |
| `muon` | `D = NewtonSchulz(m̂)`, the orthogonalised `UVᵀ` direction | yes |

β₁ = 0.9, β₂ = 0.999.

**`euler` is inert.** Verified by A/B against the pre-change sampler on a real guided
run: coordinates and features bit-identical, same field-eval count. The 690 finished
cells in `results/sweep/` are unaffected.

### The equivariance constraint, which removes one candidate

The coordinate block is `[B, N, 3]` and rotation acts on the right, `G_c → G_c Rᵀ`. A
rule is admissible only if it commutes with that.

- **Elementwise Adam does not.** Its second moment holds per-axis squared magnitudes, so
  dividing by `√v` scales x, y and z differently — an axis-aligned rescaling. Measured
  equivariance error 1.29 against 6.7e-16 for the admissible rules. It is kept in the
  comparison, flagged `equivariant: false` in every cell, so the breakage is measured
  rather than asserted. **It is not a candidate method.** `adam_eq` is the repair: one
  scalar per atom is a multiple of the identity on each 3-vector, and scalars commute
  with `R`.
- **Muon does, exactly.** Newton-Schulz iterates `p(X) = aX + b(XXᵀ)X + c(XXᵀ)²X`. Under
  `X → X Rᵀ` the Gram matrix `XXᵀ` is invariant, so `p(X Rᵀ) = p(X) Rᵀ` at every order:
  orthogonalisation commutes with rotation because rotation acts on the side the
  iteration never touches. The same argument with a permutation on the left covers atom
  ordering, and padded atoms (zero rows) stay zero because every term carries that row
  as a factor. Verified to 5.6e-15. So Muon applies to the `[N,3]` coordinate block and
  the `[N,K]` atoms-by-types block as genuine matrices — nothing is flattened into a
  matrix it was not already.

`ns_steps` is 8, not Muon's 5. Measured: a well-conditioned block reaches the quintic's
fixed-point band in 5 steps, but at condition number ~900 — which a 9×3 coordinate
gradient near rank 2 reaches easily — 5 steps leaves the smallest singular value at
0.24, i.e. not orthogonalised at all. The extra iterations are 3×3 matmuls.

### Magnitude is held; only direction is under test

Adam's output is ~unit per element and Muon's has unit singular values. Applied at our
tuned `w` that is a change of strength, not of rule, so the preconditioned direction is
rescaled to the raw direction's per-sample norm. This matters more than it sounds:
**without the rescale these rules would have applied 5.8× (momentum) to 35× (Muon) the
raw magnitude**, pinning every step against the clip, and the experiment would have
measured the clip. `--rescale none` is available and untested.

---

## 2. Protocol

`proj1/scripts/update_rule_bench.py`, 90 cells, 63 min on an RTX 5080, zero failures.

Frozen `fm_last.pt` (epoch 1500, the shipped generator), guide `f_A`, evaluator `f_B`,
`plug` arm, `t_min_guide = 0.5`, target `q50`, n = 512, 100 Euler steps, seed 20260921,
clip 1.0 — identical to the stage-v2 screening protocol, so these cells sit beside
`results/sweep/` rather than needing their own baseline.

Axes: 3 properties × 5 rules × 6 strengths `w ∈ {0.05, 0.25, 0.5, 1, 2, 4}`.

**Why a strength grid.** In the finished sweep, raising `w` lowers MAE and costs
chemistry (mu/plug: MAE 1.22 → 0.78, molecule stability 0.416 → 0.305). A comparison at
one `w` therefore rewards any rule that merely pushes harder. "Better" has to mean
better at matched stability.

---

## 3. Result: every rule ties

Regress euler's MAE on its molecule stability across all six strengths, then measure
each rule's cells against that line at their own stability. Negative = better than
euler at the same chemistry.

| property | euler line (R²) | momentum | adam | adam_eq | muon |
|---|---|---|---|---|---|
| mu | 0.92 | −0.3 se | −0.6 se | −0.6 se | **−1.8 se** |
| alpha | 0.73 | −0.1 se | −0.9 se | −0.6 se | +0.2 se |
| gap | 0.94 | −0.6 se | −0.6 se | +0.4 se | **−1.9 se** |

**No comparison reaches significance.** Muon is negative on two of three properties at
~1.8–1.9 se, which is a trend and not a result.

**On the project's own rubric** (`in_band_fraction` at a chemistry floor, not MAE),
measured the same way — regress euler's band coverage on its stability, then score each
rule against that line:

| property | momentum | adam | adam_eq | muon |
|---|---|---|---|---|
| mu | +1.0 se | +1.0 se | +1.8 se | **+2.4 se** |
| alpha | +0.0 se | −0.4 se | −0.5 se | −0.8 se |
| gap | −0.2 se | **+2.0 se** | +1.2 se | +1.7 se |

Two of twelve comparisons clear 2 se, which is about what twelve comparisons give by
chance, and the signs are inconsistent across properties (muon +2.4 on mu, −0.8 on
alpha). Read as ties, consistent with the MAE view. Muon is the only rule weakly
positive on both metrics.

**A methodological warning, because the first pass got this wrong.** An earlier analysis
interpolated euler's frontier pointwise from three strengths and reported gains of up to
10 se. With six strengths the frontier is visibly non-monotonic in stability — alpha
runs 0.412 → 0.400 → 0.373 → 0.375 → 0.369 → 0.361 against MAE 6.20 → 6.60 → 5.86 →
5.32 → 4.88 → 4.49 — because stability's own se is 0.021 at n = 512. Interpolating that
curve converts its noise into apparent significance. Any future frontier comparison in
this project should regress the whole curve, not interpolate between neighbours.

---

## 4. Why: guidance is a small correction to the base velocity

The null result has a mechanical explanation, and it is measurable in one cheap run
(`proj1/scripts/velocity_share_diag.py`, n = 128, ~1 min). At every guided step the raw
direction is pushed through the **identical** downstream conversion, strength and clip
as the transformed one, and both are evaluated at the same state, so the only difference
between the two total velocities is the rule. Writing `V` for the base FM velocity and
`C(.)` for that downstream pipeline:

| rule | r = \|C(D)\|/\|V\| | **angle(V+C(G), V+C(D))** | angle(C(G), C(D)) |
|---|---|---|---|
| adam | 0.068 | **3.68°** | 66.99° |
| muon | 0.064 | **3.26°** | 66.62° |
| momentum | 0.061 | **1.83°** | 51.46° |

mu / `plug` / w = 1 / `t_min` 0.5. Over sampling time the correction collapses:

| segment | t | r | total-velocity angle | guidance angle |
|---|---|---|---|---|
| early | 0.50–0.65 | 0.133 | 6.80° | 62.9° |
| mid | 0.66–0.81 | 0.071 | 4.18° | 69.5° |
| **late** | **0.82–0.99** | **0.0076** | **0.46°** | 68.4° |

At the strongest strength in the whole sweep (w = 4) the share only reaches r = 0.162
and 8.73°, still decaying to 2.05° late. The geometry is consistent: for r = 0.068 and a
67° rotation the predicted deflection is `atan(2r·sin(67°/2)) ≈ 4.3°` against 3.68°
measured, the residual being the correction's component along `V`.

**So the claim is not "direction does not matter".** It is:

> The guidance correction is too small a share of the sampling velocity for its
> direction to matter. Any intervention that only re-points the guidance vector is
> bounded by r ≈ 0.07 and cannot move the trajectory, however large the rotation.

That has a corollary worth acting on: **leverage lives in \|C(D)\|, not in its
direction** — strength, the clip, and the guidance window. It is consistent with the
sweep, where `t_min_guide` is the largest single signal (47 % MAE swing on alpha) while
arm choice is comparatively minor.

It also bears on the G1 gate in `GUIDANCE_EXPERIMENT_PLAN.md`. A partial G1 run
(mu, 3 arms, n = 64) shows arms collapsing onto each other late in the window —
between-arm field cosine 0.93 at t = 0.90 and 0.98 at t = 0.95, against a between-guide
cosine of 0.70–0.76. Late in sampling the arms are near-identical fields *and* near-zero
corrections at once. G1 is not yet run in full.

## 5. Staleness: real, and not the explanation

The stated hypothesis was that momentum would hurt because old directions go stale as
the trajectory moves. Measured per step, averaged over strengths, for `momentum`:

| property | cos(Gₜ, Gₜ₋₁) | cos(buffer, Gₜ) | early → late |
|---|---|---|---|
| mu | 0.820 | 0.440 | 0.431 → 0.448 |
| alpha | 0.916 | 0.475 | 0.560 → 0.393 |
| gap | 0.798 | 0.400 | 0.447 → 0.355 |

Consecutive raw directions agree well (0.80–0.92), but the accumulated buffer retains
only 0.40–0.48 alignment with what the guide asks for now, and on alpha and gap that
decays across the trajectory as predicted. **The buffer is genuinely stale.** It does
not hurt, because momentum ties — so staleness is confirmed as a phenomenon and
refuted as an explanation for anything.

---

## 6. Cost

| rule | ms/molecule | vs euler |
|---|---|---|
| euler | 70.3 | 1.00× |
| momentum | 70.7 | 1.00× |
| adam | 71.3 | 1.01× |
| adam_eq | 71.0 | 1.01× |
| muon | 71.3 | 1.01× |

All four are free. Cost is not what decides this.

---

## 7. What this does and does not license

**Does:** the claim that the inference-time update rule is not a lever on this arm, and
the direction-resolution finding in section 4.

**Does not:** a general claim about guidance. Three limits, in order of how much they
matter.

1. **`plug` only.** `smg` — our own method — is untested here. Given that 68° of
   direction change does nothing on `plug`, the prior is that `smg` behaves the same,
   but that is a prediction, not a measurement. ~2.7 h.
2. **n = 512 is underpowered**, and the analysis treats cells as independent when they
   are not: every rule generates from the same seed, so the molecules are paired. A
   per-molecule paired difference would cut the variance substantially and could settle
   Muon's 1.8 se **with no new sampling**. It needs `evaluate_samples` to return
   per-sample errors. This is the cheapest next step by a wide margin.
3. **Magnitude untested.** These results hold direction variable and magnitude fixed.
   `--rescale none` is the other half and has not been run.

---

## 8. Reproducing

```bash
python proj1/tests/test_update_rules.py          # 24 gates, incl. the equivariance proofs
python proj1/scripts/update_rule_bench.py --arms plug --props mu,alpha,gap \
    --strengths 0.05,0.25,0.5,1,2,4 --n 512 --steps 100
python proj1/scripts/update_rule_report.py       # the four tables
python proj1/scripts/velocity_share_diag.py      # section 4, ~1 min
```

Cells in `results/update_rule/` (90, one JSON each, resumable). An earlier 29-cell run
against `fm.pt` (epoch 1300) rather than the shipped `fm_last.pt` is kept in
`results/update_rule_fm1300/` — internally consistent, not comparable to the sweep, and
superseded.

New code: `proj1/src/guidance_update.py`, `proj1/tests/test_update_rules.py`,
`proj1/scripts/update_rule_bench.py`, `proj1/scripts/update_rule_report.py`,
`proj1/scripts/velocity_share_diag.py`. Two edits to `proj1/src/sampling.py`: the update
rule at the single application point, and `log_velocity=True`, which is opt-in and
changes no arithmetic on the sampling path (`euler` verified bit-identical after both).
Section 4's numbers are in `results/velocity_share.json` and
`results/velocity_share_w4.json`.
