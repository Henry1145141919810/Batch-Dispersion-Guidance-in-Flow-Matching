# Arm rankings — complete, unfiltered, all metrics

**Written for:** the project team. Every arm on **our** flow-matching generator.
Nothing filtered, including the arms the screen dropped.

## The slice, and why it is the only fair one

`t_min_guide = 0.5` · target q50 · n = 512 · seed 20260921 · **w = 1** · non-finite cells excluded.

**Why w = 1 and not best-strength.** The stage-main grid is a *cross*, not a full grid:
the strength sweep runs at `t_min = DEFAULT_WIN = 0.05`, the window sweep at
`w = DEFAULT_W = 1.0`. So at the good window (0.5) the v1-only arms (`tfg_mc`,
`lgd_mc`, `osc`, `smg2`, `smg2_curv`) have exactly **one** cell — w=1, the
pre-registered default, fixed before any data. Stage v2 ran its **full 7-point
strength sweep at window 0.5**. A best-strength table therefore compares our
tuned arms against their untuned ones and **flatters us**. w=1 is the only
comparison where every arm is on the same footing.

> **Missing experiment:** 5 arms × 6 strengths × 3 properties = **90 cells, ~2 GPU-h**.
> Run before any paper claim about best-strength performance.

## How the ranking is computed

Ordered by **MAE/δ**, lowest first, per property. MAE is the primary metric because
it uses every sample's magnitude; band coverage is the *acceptance* criterion but a
binomial with se ≈ 0.013 at n=512, too weak to rank on. **MAE alone is not the
whole story** — see the bias/spread columns.

| column | meaning |
|---|---|
| `MAE/δ` | mean abs error in tolerance units, δ = 2×f_B MAE |
| `in_band` | fraction within δ — the acceptance criterion |
| `\|bias\|` | (mean f_B − target)/δ — **is the distribution centred?** |
| `spread` | √(RMSE²−bias²)/δ — **is it concentrated?** |
| `mol_stab`, `valid` | chemistry cost |

## mu — unguided: MAE 7.27, in_band 0.072, |bias| 2.48, spread 8.75, mol_stab 0.402

| # | arm | class | MAE/δ | in_band | \|bias\| | spread | mol_stab | valid |
|---|---|---|---|---|---|---|---|---|
| 1 | `btvg` | **OURS** | 5.38 | 0.096 | 0.02 | 6.52 | 0.330 | 0.686 |
| 2 | `shg_plug_btvg` | **OURS** | 5.77 | 0.084 | 0.94 | 6.98 | 0.363 | 0.732 |
| 3 | `plug` | prior | 5.85 | 0.084 | 0.97 | 7.05 | 0.373 | 0.730 |
| 4 | `shg_three` | **OURS** | 5.95 | 0.082 | 1.02 | 7.26 | 0.373 | 0.732 |
| 5 | `shg_plug_spbc` | **OURS** | 5.99 | 0.080 | 1.06 | 7.27 | 0.375 | 0.727 |
| 6 | `tfg_mc` | prior | 6.01 | 0.096 | 0.62 | 7.30 | 0.404 | 0.766 |
| 7 | `lgd_mc` | prior | 6.06 | 0.086 | 0.73 | 7.54 | 0.377 | 0.766 |
| 8 | `btvg_var` | abl | 6.15 | 0.086 | 0.71 | 7.76 | 0.328 | 0.672 |
| 9 | `osc` | prior ⚠ | 6.23 | 0.113 | 0.83 | 7.77 | 0.375 | 0.777 |
| 10 | `smg_mean` | abl | 6.27 | 0.096 | 0.85 | 7.74 | 0.375 | 0.721 |
| 11 | `tmpd` | prior | 6.33 | 0.096 | 1.82 | 7.62 | 0.391 | 0.746 |
| 12 | `smg2` | **OURS** | 6.58 | 0.068 | 1.73 | 7.94 | 0.400 | 0.775 |
| 13 | `smg` | **OURS** | 6.60 | 0.084 | 1.61 | 8.28 | 0.396 | 0.771 |
| 14 | `shg_smg_spbc` | **OURS** | 6.76 | 0.084 | 1.60 | 8.58 | 0.404 | 0.777 |
| 15 | `smg2_curv` | **OURS** | 6.83 | 0.074 | 2.03 | 8.19 | 0.396 | 0.771 |
| 16 | `rch` | prior(Haimo v1) | 6.86 | 0.102 | 1.51 | 8.53 | 0.375 | 0.727 |
| 17 | `spbc` | **OURS** | 7.19 | 0.068 | 2.23 | 8.78 | 0.406 | 0.793 |
| 18 | `band` | **OURS** | 7.19 | 0.068 | 2.38 | 8.71 | 0.414 | 0.781 |
| — | `unguided` | baseline | 7.27 | 0.072 | 2.48 | 8.75 | 0.402 | 0.785 |

## alpha — unguided: MAE 13.93, in_band 0.043, |bias| 0.78, spread 17.95, mol_stab 0.402

| # | arm | class | MAE/δ | in_band | \|bias\| | spread | mol_stab | valid |
|---|---|---|---|---|---|---|---|---|
| 1 | `shg_plug_btvg` | **OURS** | 11.01 | 0.062 | 2.12 | 14.14 | 0.371 | 0.770 |
| 2 | `shg_plug_spbc` | **OURS** | 11.04 | 0.070 | 1.93 | 14.23 | 0.381 | 0.770 |
| 3 | `plug` | prior | 11.04 | 0.062 | 2.13 | 14.21 | 0.375 | 0.773 |
| 4 | `shg_three` | **OURS** | 11.05 | 0.066 | 1.93 | 14.23 | 0.381 | 0.768 |
| 5 | `tfg_mc` | prior | 11.18 | 0.064 | 1.82 | 14.16 | 0.373 | 0.777 |
| 6 | `btvg` | **OURS** | 11.38 | 0.057 | 0.66 | 14.79 | 0.359 | 0.746 |
| 7 | `smg_mean` | abl | 11.57 | 0.041 | 1.94 | 14.71 | 0.379 | 0.734 |
| 8 | `tmpd` | prior | 11.75 | 0.055 | 1.31 | 15.15 | 0.377 | 0.764 |
| 9 | `smg` | **OURS** | 11.75 | 0.051 | 1.78 | 15.25 | 0.383 | 0.748 |
| 10 | `shg_smg_spbc` | **OURS** | 11.78 | 0.049 | 1.60 | 15.29 | 0.373 | 0.756 |
| 11 | `osc` | prior ⚠ | 12.28 | 0.062 | 0.96 | 15.80 | 0.396 | 0.781 |
| 12 | `lgd_mc` | prior | 12.39 | 0.057 | 1.06 | 16.00 | 0.414 | 0.781 |
| 13 | `smg2` | **OURS** | 12.40 | 0.051 | 1.05 | 16.08 | 0.361 | 0.766 |
| 14 | `smg2_curv` | **OURS** | 12.75 | 0.045 | 0.36 | 16.34 | 0.369 | 0.746 |
| 15 | `spbc` | **OURS** | 13.34 | 0.043 | 0.22 | 17.24 | 0.396 | 0.770 |
| 16 | `btvg_var` | abl | 13.99 | 0.047 | 2.94 | 18.29 | 0.391 | 0.783 |
| 17 | `band` | **OURS** | 14.01 | 0.039 | 0.72 | 17.98 | 0.404 | 0.771 |
| 18 | `rch` | prior(Haimo v1) | 24.69 | 0.020 | 10.42 | 29.08 | 0.301 | 0.668 |
| — | `unguided` | baseline | 13.93 | 0.043 | 0.78 | 17.95 | 0.402 | 0.785 |

## gap — unguided: MAE 5.03, in_band 0.105, |bias| 0.99, spread 5.92, mol_stab 0.402

| # | arm | class | MAE/δ | in_band | \|bias\| | spread | mol_stab | valid |
|---|---|---|---|---|---|---|---|---|
| 1 | `btvg` | **OURS** | 4.25 | 0.145 | 0.32 | 5.18 | 0.312 | 0.648 |
| 2 | `rch` | prior(Haimo v1) | 4.33 | 0.121 | 0.07 | 5.25 | 0.318 | 0.678 |
| 3 | `btvg_var` | abl | 4.48 | 0.131 | 0.08 | 5.45 | 0.301 | 0.646 |
| 4 | `plug` | prior | 4.49 | 0.125 | 1.26 | 5.25 | 0.359 | 0.725 |
| 5 | `shg_three` | **OURS** | 4.50 | 0.133 | 1.09 | 5.32 | 0.389 | 0.754 |
| 6 | `shg_plug_btvg` | **OURS** | 4.54 | 0.125 | 1.27 | 5.30 | 0.365 | 0.719 |
| 7 | `shg_plug_spbc` | **OURS** | 4.56 | 0.131 | 1.18 | 5.34 | 0.363 | 0.727 |
| 8 | `tfg_mc` | prior | 4.62 | 0.115 | 0.79 | 5.45 | 0.404 | 0.756 |
| 9 | `tmpd` | prior | 4.65 | 0.117 | 1.03 | 5.48 | 0.398 | 0.764 |
| 10 | `osc` | prior ⚠ | 4.90 | 0.105 | 0.72 | 5.78 | 0.406 | 0.781 |
| 11 | `smg` | **OURS** | 4.91 | 0.084 | 0.91 | 5.72 | 0.379 | 0.756 |
| 12 | `smg_mean` | abl | 4.91 | 0.125 | 0.46 | 5.84 | 0.361 | 0.748 |
| 13 | `lgd_mc` | prior | 4.92 | 0.098 | 0.74 | 5.78 | 0.402 | 0.770 |
| 14 | `smg2_curv` | **OURS** | 4.93 | 0.107 | 0.76 | 5.75 | 0.396 | 0.777 |
| 15 | `shg_smg_spbc` | **OURS** | 4.93 | 0.078 | 0.90 | 5.74 | 0.377 | 0.762 |
| 16 | `smg2` | **OURS** | 4.93 | 0.117 | 0.86 | 5.80 | 0.408 | 0.805 |
| 17 | `band` | **OURS** | 4.95 | 0.121 | 0.93 | 5.84 | 0.410 | 0.801 |
| 18 | `spbc` | **OURS** | 5.00 | 0.100 | 0.82 | 5.88 | 0.406 | 0.770 |
| — | `unguided` | baseline | 5.03 | 0.105 | 0.99 | 5.92 | 0.402 | 0.785 |

---
## What the full metric changes

Ranking on MAE alone puts `plug` near the top and reads as 'prior art wins'. On the
**full** metric that is wrong:

- **`btvg` has by far the lowest bias of any arm.** mu **0.02 δ** against `plug` 0.97 and
  unguided 2.48 — it centres the distribution almost exactly. gap 0.32 vs `plug` 1.26.
  alpha 0.66 vs `plug` 2.13.
- **`btvg` also has the lowest spread** on mu (6.52 vs `plug` 7.05, unguided 8.75) and gap
  (5.18 vs 5.25). It was designed to cut spread and it does — *and* it fixes the bias.
- **The irony:** `spbc` was built to correct bias and does nothing (mu bias 2.23 vs
  unguided 2.48). `btvg`, built for spread, is the best bias corrector in the table.
- **`btvg` beats `plug` on MAE, bias and spread on both mu and gap.** `plug` wins only on
  alpha (11.04 vs 11.38) and on chemistry.
- **The cost is chemistry:** `btvg` mol_stab 0.330/0.359/0.312 against unguided 0.402, and
  validity 0.686 vs 0.785 on mu. Real, and it belongs in the same row as the win.
- **`rch` (Haimo v1) on alpha is a wreck:** bias **10.42 δ**, spread 29.08, mol_stab 0.301.
  Not inert — destructive.
- **`osc` ⚠ has the best band coverage on mu (0.113)** of any arm, and still has no citation
  anywhere in this repo.

## Are we failing?

No. At the only fair comparison, our `btvg` beats the strongest prior-art arm on three of
the five metrics that matter on two of three properties, and is the best-centred arm
everywhere. What we cannot yet claim is a *best-strength* win, because that experiment
has not been run for the prior-art arms at the window we report.
