# Findings so far, full metrics, and where the gaps are

**21 September 2026.** Written to be read before deciding what to build next. Everything is
measured unless marked as an estimate or an assumption.

**Evidence base:** 141 of 294 screening cells · 72,192 molecules generated · 512 molecules per
cell · one target only (**q50**) · NFE 100 Euler · `n_probe = 1` · generator `fm_v1` frozen ·
every number scored by **f_B**, the held-out evaluator. Section 8 separates safe claims from
provisional ones.

---

## 1. The base model: `fm_v1`

Flow matching, EGNN 256x8, 1500 epochs on `train_a` (51,527 molecules), epoch-1500 EMA weights.
3 seeds x 10,000 samples under the EDM protocol; bond tables verified identical to EDM's
`bond_analyze.py` over every HCNOF pair and distance.

| metric | `fm_v1` | real QM9 (ceiling) | EEGSDE half-data EDM | full-data EDM |
|---|---|---|---|---|
| atom stability | **0.9366 +- 0.0012** | 0.9934 | 0.9837 | 0.987 |
| molecule stability | **0.3993 +- 0.0085** | 0.9523 | 0.8174 | 0.820 |
| validity (EDM convention) | **0.7625 +- 0.0031** | 0.9767 | -- | 0.919 |
| uniqueness | 0.9939 +- 0.0009 | 1.0000 | -- | -- |
| novelty vs `train_a` | 0.8716 +- 0.0021 | -- | 0.8292 | -- |
| connected (single fragment) | 0.9255 +- 0.0021 | 0.9980 | -- | -- |

**Gap to the like-for-like baseline: 4.9 atom points, 42 molecule points.** Measured causes:

| candidate | verdict |
|---|---|
| half-data split | **ruled out** -- halving costs EDM 0.33 atom points at equal compute |
| training budget | 303k steps vs ~1.56M (5.2x) |
| **unscaled one-hot atom types** | **leading suspect** -- EDM's own ablation: 3 atom / 35 molecule points. The model beating us scales its one-hot by 1/8. **Untested; one retrain would settle it.** |
| sampler | measured, small: NFE 100 -> 500 buys 0.65 atom points, saturates by 500 |
| evaluator | **ruled out** -- our bond tables match EDM's exactly after the C#O fix |

Sampler sweep (seed 0, 10,000 samples): atom 0.933 / 0.939 / 0.940 / 0.939 at NFE 100 / 250 /
500 / 1000; Heun at 100 steps (200 evaluations) gives 0.936.

---

## 2. The property predictors

All measured on the same held-out `val` split (17,748 molecules), same metric, same batching.

| role | arch | mu (D) | alpha (a.u.) | gap (Ha) | params |
|---|---|---|---|---|---|
| **f_A (guide)** | **EGNN 128x4** | **0.0897** | **0.2447** | **0.00390** | 429,825 |
| **f_B (evaluator)** | **EGNN 128x4** | **0.0840** | **0.2407** | **0.00380** | 429,825 |
| f_A / f_B | transformer | 0.1201 / 0.1214 | 0.2882 / 0.2864 | 0.00411 / 0.00411 | 569,225 |
| f_A / f_B | ridge (floor) | 0.8008 / 0.8000 | 1.1751 / 1.1785 | 0.02055 / 0.02055 | 0 |
| external | TFG guide / oracle | 0.0659 / 0.0728 | 0.1551 / 0.1529 | 0.0029 / 0.0027 | -- |
| external | SchNet (mu only) | **0.0210** | -- | -- | -- |
| published | EDM L-bound | 0.043 | 0.10 | 0.00235 | -- |
| -- | chance (data std) | 1.539 | 8.200 | 0.0475 | -- |

**Ours are 1.6-4x behind published predictors**, because they are undertrained: 120 epochs at
128x4 against EEGSDE's 2000 epochs at 192x7. That matters because `delta = 2 x f_B MAE`, so our
tolerance bands are 1.6-4x looser than they need to be.

**The pair is fixed at EGNN/EGNN** (best on both sides, 0% data overlap, matches published
practice). External predictors overlap our `train_a` by **~37%** (expectation: their first half is
50,000 of our 133,885; an earlier draft said 38% using the wrong denominator -- **quote 37%**), so
they serve only as post-hoc robustness checks, never as the primary evaluator.

All three architectures pass exact E(3) gates -- rotation, reflection, translation, permutation,
padding, twice-differentiability -- to ~1e-17.

---

## 3. What guidance actually does

### 3.1 It aims almost perfectly. It does not concentrate.

`RMSE^2 = bias^2 + spread^2`, each arm at its own best strength, t_min = 0.05:

**mu**, target 2.4932, data mean 2.7038, data std 1.5394, delta 0.168

| arm | w | MAE (f_B) | in_band | \|bias\| | spread | spread / data std | mol_stab | validity |
|---|---|---|---|---|---|---|---|---|
| `plug` | 4 | **0.8006** | 0.109 | **0.0444** | 0.9936 | 65% | 0.277 | 0.615 |
| `lgd_mc` | 4 | 0.8490 | 0.100 | 0.1910 | 1.0278 | 67% | 0.387 | 0.717 |
| `tfg_mc` | 4 | 0.8717 | 0.113 | 0.1316 | 1.0753 | 70% | 0.352 | 0.662 |
| `osc` | 4 | 0.8817 | 0.104 | 0.2283 | 1.0387 | 67% | 0.385 | 0.729 |
| `smg` | 4 | 0.9661 | 0.090 | 0.0485 | 1.1911 | 77% | 0.324 | 0.734 |
| `smg2` | 4 | 1.0015 | 0.109 | 0.1936 | 1.2671 | 82% | 0.365 | 0.734 |
| `smg2_curv` | 4 | 1.0343 | 0.076 | 0.2834 | 1.2550 | 82% | 0.387 | 0.768 |

**alpha**, target 75.54, data mean 75.2178, data std 8.2004, delta 0.4814

| arm | w | MAE | in_band | \|bias\| | spread | spread / std | mol_stab | validity |
|---|---|---|---|---|---|---|---|---|
| `osc` | 4 | **4.3559** | 0.080 | 0.2998 | 5.6878 | 69% | 0.369 | 0.707 |
| `lgd_mc` | 4 | 4.6811 | 0.061 | 0.9077 | 5.9683 | 73% | 0.336 | 0.656 |
| `smg2_curv` | 4 | 5.6397 | 0.057 | 2.2512 | 6.7958 | 83% | 0.408 | 0.773 |
| `smg2` | 2 | 5.6711 | 0.055 | 1.7382 | 7.2603 | 89% | 0.395 | 0.787 |
| `tfg_mc` | 4 | 6.7688 | 0.043 | 3.7761 | 8.2246 | 100% | 0.270 | 0.521 |
| `smg` | 0.25 | 6.8849 | 0.055 | 2.7396 | 8.5276 | 104% | 0.410 | 0.836 |
| `plug` | 4 | 7.1804 | 0.047 | **4.1221** | 8.5736 | **105%** | **0.232** | **0.480** |

**gap**, target 0.2496, data mean 0.2513, data std 0.0475, delta 0.0076

| arm | w | MAE | in_band | \|bias\| | spread | spread / std | mol_stab | validity |
|---|---|---|---|---|---|---|---|---|
| `tfg_mc` | 4 | **0.0309** | 0.131 | 0.0126 | 0.0349 | 73% | 0.328 | 0.664 |
| `plug` | 4 | 0.0319 | 0.107 | 0.0161 | 0.0339 | 71% | 0.223 | 0.525 |
| `lgd_mc` | 4 | 0.0322 | 0.133 | 0.0010 | 0.0395 | 83% | 0.395 | 0.719 |
| `osc` | 4 | 0.0334 | 0.146 | 0.0026 | 0.0405 | 85% | 0.381 | 0.715 |
| `smg` | 4 | 0.0349 | 0.133 | 0.0016 | 0.0417 | 88% | 0.354 | 0.723 |
| `smg2` | 4 | 0.0359 | 0.133 | 0.0060 | 0.0427 | 90% | 0.363 | 0.744 |
| `smg2_curv` | 4 | 0.0370 | 0.125 | 0.0064 | 0.0438 | 92% | 0.387 | 0.758 |

**Read this as: bias is essentially solved, spread is not.** `plug` lands the mu distribution
0.0444 from a target of 2.4932. Its entire 0.80 MAE is spread -- 0.9936, **six times wider than
delta = 0.168**. Only ~10% of samples fall in band because the distribution is wide, not because
it is mis-aimed.

### 3.2 The quality trade-off separates the arms sharply

On alpha, `plug` is simultaneously the **worst MAE** (7.18), **largest bias** (4.12), **widest
spread** (105% -- wider than the raw data) and **worst molecule stability** (0.232) and **validity**
(0.480). `osc` beats it on every one of those. The Monte-Carlo arms centre better, concentrate
better and break fewer molecules at the same time.

### 3.3 Narrowing the guidance window trades targeting for quality, monotonically

**mu**, each arm's best, across windows:

| t_min | best MAE | that arm's bias | spread | mol_stab |
|---|---|---|---|---|
| 0.05 | 0.8006 (`plug`) | 0.0444 | 0.9936 (65%) | 0.277 |
| 0.50 | 0.9822 (`plug`) | 0.1638 | 1.1850 (77%) | 0.373 |
| 0.75 | 1.0941 (`plug`) | 0.3041 | 1.3228 (86%) | 0.408 |

Guide less, get better molecules and worse targeting. No arm escapes this.

### 3.4 No reward hacking

`|f_A - f_B|` in units of delta, mu, t_min = 0.05, across strengths:

| arm | w=0.25 | w=0.5 | w=1 | w=2 | w=4 |
|---|---|---|---|---|---|
| `plug` | 1.2 | 1.2 | 1.3 | 1.2 | 1.3 |
| `lgd_mc` | 1.1 | 0.9 | 1.0 | 1.0 | 1.0 |
| `osc` | 0.9 | 1.0 | 1.0 | 1.1 | 1.1 |

Flat. If guidance were exploiting the guide this would grow with strength. **The disjoint-split
protocol works**, and no arm is fooling f_A.

---

## 4. The central negative result: the two-moment closure fails here

Four independent measurements.

### 4.1 The closure is invalid over most of the trajectory

`f(m)` vs `f(m) + c` vs the true posterior average, 16 molecules, 256 draws, `c` at 64 probes:

| t | f(m) | true E[f] | f(m)+c | linear err | quadratic err | verdict |
|---|---|---|---|---|---|---|
| 0.20 | **-14.0** | 6.64 | 191 | 26.6 | 501.7 | **19x worse** |
| 0.30 | **-4.62** | 7.95 | 129.6 | 12.6 | 221.3 | 17.6x worse |
| 0.40 | 1.66 | 8.13 | 37.8 | 7.00 | 51.1 | 7.3x worse |
| 0.50 | 3.50 | 7.73 | -13.5 | 4.25 | 22.2 | 5.2x worse |
| 0.60 | 3.12 | 6.77 | 2.72 | 3.65 | 8.57 | 2.3x worse |
| 0.70 | 3.06 | 4.17 | 0.87 | 1.12 | 3.52 | 3.1x worse |
| **0.75** | 2.56 | 3.03 | 2.72 | 0.475 | **0.311** | **helps 1.53x** |
| 0.80 | 2.55 | 2.75 | 2.59 | 0.204 | 0.172 | helps 1.18x |
| 0.90 | 2.60 | 2.64 | 2.61 | 0.0434 | 0.0382 | helps 1.14x |
| 0.98 | 2.63 | 2.64 | 2.63 | 0.00408 | 0.00360 | helps 1.13x |

Crossover **t = 0.75**, and **guide-dependent**: 0.95 for the transformer guide, 0.75 for
`f_A_alpha_transformer` (53x worst case). Mechanism: at t = 0.2 the dipole guide returns
**-14.0 D**, physically impossible. It is extrapolating on blurred non-molecules.

**Validity failure, not variance.** 64 probes means `c` is well estimated, and it is still worse
than ignoring `c`. More compute converges faster to a wrong value.

### 4.2 The correction does not track what it is correcting

| t | mean `c` | mean true gap `f(x_1)-f(m)` | sd of `c` | **corr(c, gap)** |
|---|---|---|---|---|
| 0.30 | 63.06 | 4.806 | 497.6 | **-0.441** |
| 0.50 | 1.825 | -0.395 | 29.46 | 0.238 |
| 0.70 | 0.285 | -0.157 | 3.735 | 0.202 |
| 0.80 | 0.0259 | -0.0135 | 0.185 | 0.115 |
| 0.90 | 0.0045 | -0.0119 | 0.0282 | 0.128 |

**Anti-correlated at t = 0.3**, 13x too large in the mean, sd 498 against a 1.5 D property scale.

### 4.3 The gap IS learnable -- the analytic estimate just is not tracking it

Ridge head on 30 E(3)-invariant features (Haimo v1's parameterisation), refitted after the
scalar-`k` bug was fixed, n = 1024, 8 probes:

| target | train R^2 | **held-out R^2** | target std |
|---|---|---|---|
| **residual** (predict `c`) | 0.0437 | **0.0165** | 423.6 |
| **direct** (predict `f(x_1)-f(m)`) | 0.7663 | **0.7551** | 15.36 |

A cheap lookup explains **1.7%** of `c` but **76%** of the true gap. So Haimo v1's hypothesis
(`c` is mostly a function of `t`) is refuted -- and so is `c` itself as an estimator.

### 4.4 The valid window is the window where guidance stops working

At t_min = 0.75 on alpha, all seven arms land between **6.5849 and 6.6025** (0.3% spread), all
with bias ~0.39 and spread ~104% of the data. Every arm has reverted to the unguided
distribution. Only 25 of 100 steps are guided.

**The region where the closure becomes valid is the region where guidance has no effect.** That
is structural, not a tuning problem.

---

## 5. Where our own arms stand

Rank of 7, each at its best strength, scored by f_B:

| | mu .05 | mu .5 | mu .75 | alpha .05 | alpha .5 | alpha .75 | gap .05 |
|---|---|---|---|---|---|---|---|
| **`smg`** | 5 | 6 | 5 | 6 | 3 | 1* | 5 |
| **`smg2`** | 6 | 5 | 7 | 4 | 6 | 6 | 6 |
| **`smg2_curv`** | 7 | 7 | 6 | 3 | 7 | 7 | 7 |
| `plug` | 1 | 1 | 1 | 7 | 1 | 2 | 2 |
| `lgd_mc` | 2 | 3 | 2 | 2 | 5 | 5 | 3 |
| `tfg_mc` | 3 | 2 | 4 | 5 | 2 | 3 | 1 |
| `osc` | 4 | 4 | 3 | 1 | 4 | 4 | 4 |

\* the only first place, in the column where all seven arms are within 0.3%.

`smg2_curv` -- most elaborate, most expensive (836 ms/molecule vs `plug`'s 95) -- is last or
second-to-last in 6 of 7 columns. Caveats: q90 untested, and `n_probe = 1` is below the probe-noise
signal threshold.

### Measured cost per arm, NFE 100

| arm | ms/molecule | x unguided |
|---|---|---|
| unguided | 34.5 | 1.0 |
| `plug` | 95.0 | 2.8 |
| `tfg_mc` (K=4) | 125.5 | 3.6 |
| `tmpd` | 241.8 | 7.0 |
| `lgd_mc` / `osc` (K=4) | 271.2 / 273.3 | 7.9 |
| `smg` (1 probe) | 421.5 | 12.2 |
| `smg2` (1 probe) | 802.7 | 23.3 |
| `smg2_curv` (1 probe) | 836.4 | 24.3 |
| `smg2` + kappa3 | 1173.1 | 34.0 |

---

## 6. THE GAPS

### Gap 0. **The tolerance `s` was never swept, and it is set 6-17x too wide** (NEW, and likely large)

Every arm's likelihood is `exp(-(y - f(x))^2 / 2 s^2)`. **`s` is the tolerance the guidance aims
for.** We set `s = y_std` and never varied it:

| property | `s` used | delta (what we score) | ratio |
|---|---|---|---|
| mu | 1.539 | 0.168 | **9.2x too wide** |
| alpha | 8.200 | 0.4814 | **17.0x too wide** |
| gap | 0.0475 | 0.0076 | **6.2x too wide** |

We told guidance "anywhere within 1.5 D is acceptable" and scored it on "within 0.168 D". This is
very likely a major contributor to Section 3.1: a wide likelihood has no incentive to concentrate.
**Sweeping `s` in {delta, 2 delta, y_std/4, y_std} is cheap and should be the next experiment run.**

### Gap 1. Nothing targets the property VARIANCE (the biggest structural gap)

Guidance acts on the score, so it moves `E[f(X)|x_t]`. **No arm targets `Var[f(X)|x_t]`.** Our own
measurement says bias is 0.044 -- solved -- and ~100% of the error is spread. Every method is
optimising the term that is already fine.

And the variance is **already computed**: `g' Sigma g` is exactly the property's spread at each
state. Every SMG arm calculates it and uses it only as a denominator.

**Accepting a band +-r turns an impossible goal (variance -> 0) into a reachable one
(variance -> r^2).** The quantity needed is in hand at no extra cost. This is the most promising
direction in this document. Novelty is **unverified** and needs a citation trace -- TMPD/PiGDM/SMG
all use `g' Sigma g` as an uncertainty denominator, none as a *target*.

**Distinguish it from Component D**, which is the same intuition in rejected form: D imposes the
band as a hard QP constraint, and the TOP6 memo rejects it ("the tolerance band is the zero-loss
set of a dead-zone loss; both halves standard; a local acceptance test is not a terminal
guarantee"). Targeting the variance is a different mechanism from constraining each step.

### Gap 2. The learned correction beats the analytic one and nobody uses it

R^2 0.755 vs 0.017 for the same quantity (4.3). Keeping SMG's framework but replacing `c` with the
learned head costs plug-in prices at inference and is Haimo's parameterisation. **Untested.**

### Gap 3. The guide is off-distribution and nothing fixes it

Every failure in Section 4 traces to `f_A` being evaluated on blurred non-molecules. The standard
fix is a time-conditioned guide trained on noised inputs (EEGSDE does this). **The honest
consequence:** such a guide learns `E[f(x_1)|x_t]` directly, the Jensen gap vanishes by
construction, and **SMG has nothing left to correct.** That is itself a finding.

### Gap 4. The easy target hides everything

q50 sits 0.04-0.14 sigma from the data mean -- the model already produces it. q90 (1.2-1.4 sigma)
is queued and untested. **No claim about which arm is best is safe until it lands.**

### Gap 5. The concentration-quality frontier is unplotted and needs no new compute

We report MAE and stability separately. Nobody has plotted: at a fixed molecule stability, which
arm gets closest to the target? That is the figure a reader wants, and it is a re-analysis of
cells already on disk.

---

## 7. Open items

| item | state |
|---|---|
| Sweep completion (`unguided`, real `tmpd`, strengths 0.01/0.05, **q90**) | running |
| Diffusion base model | **done 28 Sep** (trained by Bobo; 18 v3 cells, 0.81 GPU-h) |
| **`s` (tolerance) sweep** | **not run -- see Gap 0** |
| `smg_mean` ablation rung | not run; the ladder is incomplete without it |
| Stage 2 at n = 4,096 | after the arm cut |
| Robustness check (SchNet mu, TFG oracle alpha/gap) | after a winner exists |
| **M2 DNA simplex** | **base model trained and validated, sweep built, NOT RUN** (26 Sep). `proj1/m2/`, checkpoint and data tracked. Outstanding: the run, plus four defects in [MODALITY2_V3_PLAN.md](../protocol/MODALITY2_V3_PLAN.md) §2.2 |
| `escalate` arm | designed, not implemented |
| `band` (Component D) | implemented, diverges above w = 0.05, held out |
| Feature-scaling retrain | not done; would test the 4.9-point base-model gap |

---

## 8. What is safe to claim

**Safe now:** the base-model benchmark and its causes; the closure-validity crossover and
mechanism; `c` not tracking the gap; the learned head beating it; no reward hacking; the
bias/spread decomposition at q50; measured per-arm costs.

**Provisional:** any arm ranking (q90 untested); any claim an arm beats `unguided` (baseline
running); SMG at a fair probe count (`n_probe = 1` is below the noise floor); anything affected by
the mis-set `s` (Gap 0) -- **which is every concentration number in Section 3**.

**Not claimed:** terminal quality at publication sample size (n = 512 gives molecule-stability
SE 0.022); anything about the diffusion family.
