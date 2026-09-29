# Full-scale run: dist target, n=5000 x 3 seeds (20261001, 20261002, 20261003)

PRIMARY set (FR3a): each arm at its best-MAE q90 compare-stage strength among those with mol_stability >= 0.9 x unguided. MAE, bias and spread are in units of delta. Rubric: in_band gain over unguided, subject to the same floor measured here at full scale. Every se is conditional on these test molecules (see the docstring).

## mu  (delta = 0.168)

| arm | w | in_band | +-se | seed sd | MAE/d | +-se | |bias|/d | resid sd/d | mol_stab | valid | chem floor | rubric |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unguided | 1 | 0.076 | 0.002 | 0.003 | 9.012 | 0.062 | 0.934 | 11.760 | 0.402 | 0.760 | ok | reference |
| plug | 0.05 | 0.079 | 0.002 | 0.004 | 8.684 | 0.058 | 0.746 | 11.218 | 0.402 | 0.765 | ok | gain +0.004 |
| tmpd | 0.5 | 0.083 | 0.002 | 0.005 | 8.160 | 0.054 | 0.656 | 10.465 | 0.398 | 0.760 | ok | gain +0.007 |
| lgd_mc | 4 | 0.104 | 0.002 | 0.005 | 6.479 | 0.043 | 1.326 | 8.223 | 0.380 | 0.735 | ok | gain +0.029 |
| btvg | 0.05 | 0.081 | 0.002 | 0.003 | 8.614 | 0.058 | 0.458 | 11.175 | 0.400 | 0.758 | ok | gain +0.006 |
| btvg_var | 0.01 | 0.077 | 0.002 | 0.004 | 8.957 | 0.061 | 0.829 | 11.623 | 0.404 | 0.760 | ok | gain +0.002 |

FR5 -- btvg minus each arm. d in_band: + favours btvg. d MAE: - favours btvg. Verdict on in_band sigma_ind >= 3, both arms above the floor.

| vs | d in_band | sigma_ind | sigma_pair | d MAE/d | sigma_ind | sigma_pair | verdict |
|---|---|---|---|---|---|---|---|
| unguided | +0.0059 | +1.91 | +2.73 | -0.398 | -4.67 | -9.56 | tie |
| plug | +0.0021 | +0.68 | +0.99 | -0.069 | -0.84 | -1.83 | tie |
| tmpd | -0.0012 | -0.38 | -0.55 | +0.454 | +5.73 | +11.67 | tie |
| lgd_mc | -0.0229 | -6.83 | -7.30 | +2.135 | +29.56 | +39.47 | lgd_mc beats btvg |
| btvg_var | +0.0041 | +1.32 | +2.00 | -0.342 | -4.06 | -9.18 | tie |

Steering breakdown: in_band by |target - unguided mean at that atom count| / sd(targets) (sd 1.487). Mean distance 0.725 sd.

| arm | [0, 0.5) sd | [0.5, 1) sd | [1, 1.5) sd | [1.5, inf) sd |
|---|---|---|---|---|
| share of targets | 0.41 | 0.32 | 0.19 | 0.08 |
| unguided | 0.092 | 0.081 | 0.053 | 0.020 |
| plug | 0.100 | 0.083 | 0.054 | 0.018 |
| tmpd | 0.104 | 0.089 | 0.053 | 0.021 |
| lgd_mc | 0.116 | 0.116 | 0.089 | 0.035 |
| btvg | 0.102 | 0.086 | 0.054 | 0.022 |
| btvg_var | 0.098 | 0.079 | 0.055 | 0.019 |

## alpha  (delta = 0.4814)

| arm | w | in_band | +-se | seed sd | MAE/d | +-se | |bias|/d | resid sd/d | mol_stab | valid | chem floor | rubric |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unguided | 1 | 0.058 | 0.002 | 0.004 | 11.824 | 0.083 | 0.015 | 15.594 | 0.402 | 0.760 | ok | reference |
| plug | 0.5 | 0.063 | 0.002 | 0.006 | 10.669 | 0.071 | 1.274 | 13.726 | 0.396 | 0.758 | ok | gain +0.004 |
| tmpd | 1 | 0.064 | 0.002 | 0.005 | 10.262 | 0.069 | 1.451 | 13.181 | 0.391 | 0.754 | ok | gain +0.006 |
| lgd_mc | 4 | 0.080 | 0.002 | 0.006 | 8.376 | 0.057 | 2.218 | 10.695 | 0.385 | 0.743 | ok | gain +0.022 |
| btvg | 1 | 0.065 | 0.002 | 0.003 | 9.693 | 0.065 | 0.690 | 12.500 | 0.387 | 0.753 | ok | gain +0.006 |
| btvg_var | 0.01 | 0.059 | 0.002 | 0.005 | 11.808 | 0.083 | 0.000 | 15.587 | 0.401 | 0.760 | ok | gain +0.000 |

FR5 -- btvg minus each arm. d in_band: + favours btvg. d MAE: - favours btvg. Verdict on in_band sigma_ind >= 3, both arms above the floor.

| vs | d in_band | sigma_ind | sigma_pair | d MAE/d | sigma_ind | sigma_pair | verdict |
|---|---|---|---|---|---|---|---|
| unguided | +0.0065 | +2.33 | +3.05 | -2.131 | -20.25 | -37.64 | tie |
| plug | +0.0021 | +0.76 | +1.04 | -0.976 | -10.14 | -21.00 | tie |
| tmpd | +0.0007 | +0.26 | +0.36 | -0.569 | -6.03 | -12.80 | tie |
| lgd_mc | -0.0157 | -5.25 | -5.93 | +1.317 | +15.25 | +23.60 | lgd_mc beats btvg |
| btvg_var | +0.0061 | +2.21 | +2.89 | -2.114 | -20.08 | -37.32 | tie |

Steering breakdown: in_band by |target - unguided mean at that atom count| / sd(targets) (sd 8.053). Mean distance 0.492 sd.

| arm | [0, 0.5) sd | [0.5, 1) sd | [1, 1.5) sd | [1.5, inf) sd |
|---|---|---|---|---|
| share of targets | 0.62 | 0.27 | 0.07 | 0.04 |
| unguided | 0.076 | 0.037 | 0.018 | 0.000 |
| plug | 0.080 | 0.043 | 0.022 | 0.005 |
| tmpd | 0.082 | 0.042 | 0.021 | 0.004 |
| lgd_mc | 0.095 | 0.067 | 0.036 | 0.014 |
| btvg | 0.080 | 0.049 | 0.022 | 0.004 |
| btvg_var | 0.076 | 0.037 | 0.018 | 0.000 |

## gap  (delta = 0.007598)

| arm | w | in_band | +-se | seed sd | MAE/d | +-se | |bias|/d | resid sd/d | mol_stab | valid | chem floor | rubric |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unguided | 1 | 0.114 | 0.003 | 0.008 | 5.828 | 0.036 | 0.986 | 7.209 | 0.402 | 0.760 | ok | reference |
| plug | 0.25 | 0.125 | 0.003 | 0.007 | 5.355 | 0.033 | 1.148 | 6.601 | 0.386 | 0.749 | ok | gain +0.011 |
| tmpd | 1 | 0.126 | 0.003 | 0.008 | 5.273 | 0.032 | 1.091 | 6.508 | 0.382 | 0.740 | ok | gain +0.012 |
| lgd_mc | 4 | 0.152 | 0.003 | 0.005 | 4.412 | 0.028 | 0.447 | 5.565 | 0.372 | 0.737 | ok | gain +0.038 |
| btvg | 0.05 | 0.116 | 0.003 | 0.007 | 5.680 | 0.035 | 0.891 | 7.047 | 0.383 | 0.748 | ok | gain +0.002 |
| btvg_var | 0.05 | 0.115 | 0.003 | 0.002 | 5.790 | 0.036 | 0.846 | 7.191 | 0.384 | 0.747 | ok | gain +0.001 |

FR5 -- btvg minus each arm. d in_band: + favours btvg. d MAE: - favours btvg. Verdict on in_band sigma_ind >= 3, both arms above the floor.

| vs | d in_band | sigma_ind | sigma_pair | d MAE/d | sigma_ind | sigma_pair | verdict |
|---|---|---|---|---|---|---|---|
| unguided | +0.0021 | +0.58 | +0.86 | -0.148 | -2.98 | -5.85 | tie |
| plug | -0.0087 | -2.32 | -3.32 | +0.325 | +6.78 | +12.29 | tie |
| tmpd | -0.0098 | -2.60 | -3.61 | +0.407 | +8.55 | +14.88 | tie |
| lgd_mc | -0.0360 | -9.16 | -11.19 | +1.268 | +28.40 | +40.22 | lgd_mc beats btvg |
| btvg_var | +0.0013 | +0.34 | +0.72 | -0.110 | -2.21 | -6.41 | tie |

Steering breakdown: in_band by |target - unguided mean at that atom count| / sd(targets) (sd 0.04781). Mean distance 0.694 sd.

| arm | [0, 0.5) sd | [0.5, 1) sd | [1, 1.5) sd | [1.5, inf) sd |
|---|---|---|---|---|
| share of targets | 0.39 | 0.36 | 0.18 | 0.06 |
| unguided | 0.144 | 0.122 | 0.064 | 0.020 |
| plug | 0.148 | 0.144 | 0.069 | 0.023 |
| tmpd | 0.154 | 0.139 | 0.074 | 0.024 |
| lgd_mc | 0.165 | 0.178 | 0.110 | 0.042 |
| btvg | 0.143 | 0.129 | 0.063 | 0.023 |
| btvg_var | 0.143 | 0.125 | 0.068 | 0.016 |

## SECONDARY -- FR3 as registered (unconstrained best-MAE strength), seed 20261001 only

The pre-registered choice, kept so the registered analysis is reported. NOT the headline: at these strengths several arms fail the chemistry floor. FR5 is not applied here.

| mu | w | in_band | MAE/d | mol_stab | chem floor |
|---|---|---|---|---|---|
| unguided | 1 | 0.073 | 8.945 | 0.394 | ok |
| plug | 4 | 0.134 | 5.276 | 0.295 | NO |
| tmpd | 4 | 0.113 | 5.632 | 0.322 | NO |
| lgd_mc | 4 | 0.099 | 6.492 | 0.398 | ok |
| btvg | 4 | 0.117 | 5.658 | 0.273 | NO |
| btvg_var | 0.01 | 0.075 | 8.901 | 0.400 | ok |

| alpha | w | in_band | MAE/d | mol_stab | chem floor |
|---|---|---|---|---|---|
| unguided | 1 | 0.056 | 11.864 | 0.394 | ok |
| plug | 4 | 0.078 | 8.157 | 0.358 | ok |
| tmpd | 4 | 0.080 | 8.430 | 0.376 | ok |
| lgd_mc | 4 | 0.080 | 8.413 | 0.383 | ok |
| btvg | 4 | 0.083 | 7.654 | 0.355 | ok |
| btvg_var | 0.01 | 0.056 | 11.846 | 0.395 | ok |

| gap | w | in_band | MAE/d | mol_stab | chem floor |
|---|---|---|---|---|---|
| unguided | 1 | 0.116 | 5.857 | 0.394 | ok |
| plug | 4 | 0.179 | 3.861 | 0.258 | NO |
| tmpd | 4 | 0.160 | 4.241 | 0.284 | NO |
| lgd_mc | 4 | 0.158 | 4.413 | 0.378 | ok |
| btvg | 4 | 0.140 | 4.510 | 0.267 | NO |
| btvg_var | 4 | 0.111 | 5.672 | 0.290 | NO |
