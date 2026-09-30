# Joint strength x start-time tune, q90, n = 1000: results

**Written for:** the project team, for writing the paper. This is the only write-up of the 25 Sep tune sweep. **It is a q90 run, and the paper reports q50 only, so no number on this page goes into the paper or the slides.** It is here so the team knows what the sweep found, and why the stage-2 table it was meant to feed does not exist.

> **A screen, not a verdict.** Every cell is n = 1000 at one seed (20260925). Every pick is the maximum over a grid, so its in-band is biased upward. The in-band δ is **not** v2's, so no in-band number here can be compared with the v2 tables. The Caveats section has the numbers behind each of these points.

## What the run is

The sweep ran our own flow-matching base (`fm_last.pt`) at the fixed q90 target for all seven arms (`unguided`, `plug`, `tmpd`, `lgd_mc`, `tfg`, `btvg`, `btvg_var`), sweeping guidance strength w (9 values, 0.01 to 16) **jointly** with the guidance start time t_min (0.05, 0.5, 0.75). v2 had fixed the start time at 0.5. For each (property, arm) it takes two picks. `floor` is the best continuous in-band among cells whose molecule stability is at least 0.9 x unguided's. `open` is the best in-band with no chemistry constraint. `open` is a ceiling, used to measure what the floor costs, and is not an operating point. The in-band half-width is δ = 2 x f_B's MAE on val molecules near q90 (`results/local_fb_mae.json`), not v2's pre-registered 2 x f_B's global MAE. The plan is [TUNE_SWEEP_PLAN.md](../protocol/TUNE_SWEEP_PLAN.md), pre-registered 25 Sep. Stage 2 was supposed to re-run both picks at n = 4000 x 3 seeds, chained behind the freeze. It never ran (next section).

## What ran, and what did not

- **Stage 1 (the sweep) is complete.** All 489 of the 489 planned cells are on disk (3 unguided + 6 guided arms x 3 properties x 9 strengths x 3 start times). There are no `.failed` markers and no `CELL FAILED` lines, and all 20 jobs exited with rc = 0: array 8706732 (18 tasks) plus insurance links 8706733 and 8706734, which found nothing left to do. The sweep used 30.80 GPU-h; the plan had costed it at 24.6. There are 4 non-finite samples in the whole tree, all in `tfg` cells on mu at w >= 8 (pathology table below).
- **The freeze and stage 2 never ran, at least up to the pull.** `pull_0927.tgz` (at the repo root until 28 Sep, now `bundles/results/pull_0927.tgz`) is a snapshot of Betty's `results/` and `logs/` trees. It has 5,451 entries, and its newest is timestamped 2026-09-27 06:19; the sweep's last link wrote its log at 2026-09-25 14:17. (Timestamps are as `tar -tv` lists them on this machine.) The snapshot carries other runs' cells and freeze files, for example `results/full/n5000/frozen_q90.json`, `results/full/n5000/frozen_q90_tfg.json`, `results/full/v2/n5000/frozen_v2.json`, `results/transfer_equifm/full/n5000/frozen_q90.json` and the 26-27 Sep `results/v3/` cells. But it has:
  - no `results/tune/n1000/frozen_tune.json`, which `tune_freeze.slurm` writes;
  - no `logs/tunefreeze-*` or `logs/tunedfull-*` log;
  - no `results/full/v2_tuned/`.

  None of the 20 sweep logs mentions a freeze or a refusal. The freeze job did not *refuse*: a refusal still writes a `tunefreeze-*` log. So at pull time the freeze had either never been submitted, or been submitted and not started (still pending, or cancelled). The stage-2 jobs chained behind it with `afterok` could not have run in any of these cases. One suggestive detail: the job IDs right after the sweep's are taken by other work (the pull holds `logs/basecmp-8706737.out` and `logs/basecmp-8706738.out`). The plan's one-paste chain would probably have produced the freeze and full-run jobs in that range. IDs are cluster-wide, so this is suggestive only.
- **The freeze would not have refused this tree.** Run here on the local code, `freeze_tune.py` exits 0 and schedules 37 stage-2 cells per seed after de-duplication. Its output went to a scratch directory, **not** into `results/tune/n1000/`. Writing it there would make it look as if the chained freeze had run.
- **So no n = 4000 x 3-seed numbers exist for these picks.** Everything below is the n = 1000 screen. The sweep also ran without `--per-mol`, so there are no per-molecule sidecars, on Betty or here. Paired comparisons and cluster-robust standard errors are impossible on this tree.

## Provenance

| item | value |
|---|---|
| cells | `results/tune/n1000/*__tune.json`, 489 files, aggregates only (no sidecars) |
| pulled in | `pull_0927.tgz`, newest entry 2026-09-27 06:19 (tarball timestamp as listed on this machine) |
| cells written | 2026-09-25 03:39 to 12:05 (tarball timestamps; the extracted files' mtimes agree) |
| SLURM jobs | array 8706732, tasks 0-17 (job IDs 8706732 and 8707193-8708467, from the logs), insurance links 8706733 and 8706734; logs `logs/tune-*.out` |
| n, seed | 1000, 20260925; every cell reseeds with this seed, so all cells share initial noise and molecule sizes |
| target | q90, fixed per property: mu 4.6627 D, alpha 85.05 Bohr³, gap 0.3162 Ha |
| δ | `results/local_fb_mae.json`: mu 0.16992, alpha 0.46754, gap 0.00739 (v2 used 0.16799 / 0.48135 / 0.00760) |
| generator | `fm_last.pt`, md5 `a190ac8394902027d4a951f8d30e8c5c` |
| device | NVIDIA B200 MIG 2g.45gb, torch 2.11.0+cu128 |
| picks | `proj1/scripts/freeze_tune.py`, unchanged; `tune_table.py` calls its own pick functions and cross-checks all 36 picks against its output |

The generated provenance table, including the per-job log summary, opens the generated block below.

## Regenerate

From the repo root in Git Bash, read-only on `results/`. CPU only, seconds each. Use the venv interpreter (`.venv/Scripts/python.exe` on Windows, `python` with the venv active elsewhere). The global `python` has no torch, and `freeze_tune.py` imports `guidance_sweep`.

```
SCRATCH="$(mktemp -d)"
.venv/Scripts/python.exe proj1/scripts/freeze_tune.py --root results/tune/n1000 --json-out "$SCRATCH/frozen_tune.json" --md-out "$SCRATCH/TUNE_SWEEP_TABLE.md"
.venv/Scripts/python.exe proj1/scripts/tune_table.py --root results/tune/n1000 --frozen "$SCRATCH/frozen_tune.json" --logs "logs/tune-*.out" --md-out docs/results/TUNE_N1000_RESULTS.md
```

The first command is the freeze. It writes the picks and the full sweep grid to scratch, deliberately not into `results/`. The second rewrites **only** the two generated blocks on this page, between the `BEGIN`/`END` markers, and keeps the prose. `--frozen` is optional; with it, the script refuses if any pick disagrees with the freeze's.

## Tables

<!-- BEGIN generated by proj1/scripts/tune_table.py -- do not hand-edit -->

_Generated 2026-09-27 by `proj1/scripts/tune_table.py` from `results/tune/n1000`._

### Provenance (generated)

| item | value |
|---|---|
| cells read / planned | 489 / 489 (3 unguided + 6 guided arms x 3 properties x 9 strengths x 3 windows) |
| `.failed` markers in the tree | 0 |
| `frozen_tune.json` in the tree | **no** |
| freeze cross-check | all 36 picks agree with `frozen_tune.json` |
| n per cell, seed | 1000, 20260925 (every cell reseeded with the same seed) |
| target | `q90`: mu 4.6627, alpha 85.0500, gap 0.3162 |
| in-band delta (this run) | mu 0.16992, alpha 0.46754, gap 0.00739 |
| delta source | `local_fb_mae.json` (delta = 2 x MAE of f_B over val molecules whose true property is in [Q_train_a(0.90 - h), Q_train_a(0.90 + h)]) |
| v2's pre-registered delta (NOT used here) | mu 0.16799 (+1.2 %), alpha 0.48135 (-2.9 %), gap 0.00760 (-2.8 %) |
| generator | `fm_last.pt` md5 `a190ac8394902027d4a951f8d30e8c5c` |
| device, torch | NVIDIA B200 MIG 2g.45gb, 2.11.0+cu128 |
| sampler | 100-step euler, velocity clip 1, batch 128, n_mc 4, sigma_mc 0.1 |
| strengths | 0.01, 0.05, 0.25, 0.5, 1, 2, 4, 8, 16 |
| t_min_guide | 0.05, 0.5, 0.75 |
| non-finite samples | 4 across all cells |
| cell file mtimes (this machine's clock, as preserved by the pull) | 2026-09-25 03:39 to 2026-09-25 12:05 |
| unguided cells | one generation scored three ways (identical stability, validity, uniqueness and SMILES sample across properties) |

SLURM logs matched by `logs/tune-*.out`: 20.

| log | job | array task | cells run | GPU-min | cells on disk at end | rc | CELL FAILED lines | freeze/refuse text |
|---|---|---|---|---|---|---|---|---|
| `tune-8706732.out` | 8706732 | 8706732 task 17 | 27 | 70.7 | 489 / 489 | 0 | 0 | no |
| `tune-8706733.out` | 8706733 | none task none | 0 | 0.0 | 489 / 489 | 0 | 0 | no |
| `tune-8706734.out` | 8706734 | none task none | 0 | 0.0 | 489 / 489 | 0 | 0 | no |
| `tune-8707193.out` | 8707193 | 8706732 task 0 | 28 | 108.6 | 110 / 489 | 0 | 0 | no |
| `tune-8707228.out` | 8707228 | 8706732 task 1 | 28 | 108.7 | 124 / 489 | 0 | 0 | no |
| `tune-8707230.out` | 8707230 | 8706732 task 2 | 28 | 108.4 | 126 / 489 | 0 | 0 | no |
| `tune-8707231.out` | 8707231 | 8706732 task 3 | 27 | 67.9 | 73 / 489 | 0 | 0 | no |
| `tune-8707380.out` | 8707380 | 8706732 task 4 | 27 | 67.9 | 181 / 489 | 0 | 0 | no |
| `tune-8707428.out` | 8707428 | 8706732 task 5 | 27 | 67.5 | 234 / 489 | 0 | 0 | no |
| `tune-8707443.out` | 8707443 | 8706732 task 6 | 27 | 44.7 | 218 / 489 | 0 | 0 | no |
| `tune-8707445.out` | 8707445 | 8706732 task 7 | 27 | 44.6 | 220 / 489 | 0 | 0 | no |
| `tune-8707484.out` | 8707484 | 8706732 task 8 | 27 | 44.7 | 253 / 489 | 0 | 0 | no |
| `tune-8707503.out` | 8707503 | 8706732 task 9 | 27 | 205.2 | 361 / 489 | 0 | 0 | no |
| `tune-8707504.out` | 8707504 | 8706732 task 10 | 27 | 204.8 | 360 / 489 | 0 | 0 | no |
| `tune-8707511.out` | 8707511 | 8706732 task 11 | 27 | 205.0 | 370 / 489 | 0 | 0 | no |
| `tune-8707546.out` | 8707546 | 8706732 task 12 | 27 | 119.4 | 320 / 489 | 0 | 0 | no |
| `tune-8708238.out` | 8708238 | 8706732 task 13 | 27 | 119.5 | 434 / 489 | 0 | 0 | no |
| `tune-8708425.out` | 8708425 | 8706732 task 14 | 27 | 119.4 | 481 / 489 | 0 | 0 | no |
| `tune-8708426.out` | 8708426 | 8706732 task 15 | 27 | 70.5 | 443 / 489 | 0 | 0 | no |
| `tune-8708467.out` | 8708467 | 8706732 task 16 | 27 | 70.6 | 455 / 489 | 0 | 0 | no |

Total: 489 cells run, 1848.1 GPU-min = 30.80 GPU-h.

### Unguided reference and the chemistry floor (generated)

| property | delta | in_band ±se | dec | MAE/d (dec) | bias/d (dec) | resid sd/d (dec) | mol_stab ±se | floor = 0.9 x (±se) | valid | uniq |
|---|---|---|---|---|---|---|---|---|---|---|
| mu | 0.16992 | 0.033 ±0.0056 | 0.033 | 12.01 (12.00) | -10.45 (-10.30) | 9.00 (9.12) | 0.420 ±0.0156 | 0.3780 (±0.0140) | 0.768 | 0.999 |
| alpha | 0.46754 | 0.028 ±0.0052 | 0.030 | 22.99 (22.88) | -21.25 (-21.07) | 18.70 (18.74) | 0.420 ±0.0156 | 0.3780 (±0.0140) | 0.768 | 0.999 |
| gap | 0.00739 | 0.058 ±0.0074 | 0.055 | 10.28 (10.36) | -10.07 (-10.16) | 6.17 (6.12) | 0.420 ±0.0156 | 0.3780 (±0.0140) | 0.768 | 0.999 |

The floor is 0.9 x ONE unguided cell at n = 1000, so it carries that cell's sampling error (the bracketed se).

### (a) The picks, full metric block (generated)

`k` = number of candidate cells the pick is the maximum of (floor: cells clearing the floor; open: all 27). MAE, bias and residual sd are over delta, continuous with decoded in brackets. Flags: `w=max` / `w=min` = pick at the top (16) / bottom (0.01) of the strength grid; `t-open` = at an end of the t_min grid AND beats the neighbouring window at the same w (freeze_tune's window_edge rule); `t-edge` = at an end of the t_min grid but not rising; `dec!=` = the decoded metric would pick the cell shown; `collapse` = uniqueness of valid < 0.95; `below floor` = open pick fails the floor.

**mu** (delta 0.16992, floor 0.3780)

| arm | pick | w | t_min | k | in_band ±se | dec | MAE/d | bias/d | resid sd/d | mol_stab | valid | uniq | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `unguided` | -- | -- | -- | 1 | 0.033 ±0.0056 | 0.033 | 12.01 (12.00) | -10.45 (-10.30) | 9.00 (9.12) | 0.420 | 0.768 | 0.999 | reference |
| `plug` | floor | 0.5 | 0.5 | 10 | 0.046 ±0.0066 | 0.046 | 10.32 (10.24) | -8.79 (-8.62) | 8.40 (8.47) | 0.378 | 0.735 | 1.000 | -- |
| `plug` | open | 16 | 0.5 | 27 | 0.113 ±0.0100 | 0.110 | 5.37 (5.69) | -4.17 (-4.01) | 5.25 (5.83) | 0.186 | 0.518 | 1.000 | w=max, below floor |
| `tmpd` | floor | 0.5 | 0.05 | 13 | 0.052 ±0.0070 | 0.049 | 10.99 (10.98) | -9.38 (-9.30) | 8.85 (8.92) | 0.425 | 0.767 | 0.997 | t-open |
| `tmpd` | open | 16 | 0.5 | 27 | 0.115 ±0.0101 | 0.116 | 5.67 (6.15) | -4.30 (-4.09) | 5.62 (6.51) | 0.182 | 0.493 | 1.000 | w=max, below floor |
| `lgd_mc` | floor | 0.01 | 0.05 | 17 | 0.042 ±0.0063 | 0.036 | 11.96 (11.89) | -10.37 (-10.20) | 9.12 (9.18) | 0.406 | 0.760 | 0.999 | w=min, t-open, dec!= (w=0.05,t=0.05) |
| `lgd_mc` | open | 16 | 0.05 | 27 | 0.091 ±0.0091 | 0.080 | 7.27 (7.45) | -6.35 (-6.18) | 6.33 (6.72) | 0.285 | 0.646 | 1.000 | w=max, t-open, dec!= (w=16,t=0.5), below floor |
| `tfg` | floor | 0.05 | 0.05 | 6 | 0.048 ±0.0068 | 0.041 | 10.48 (10.74) | -9.25 (-9.34) | 8.00 (8.25) | 0.405 | 0.752 | 0.999 | t-open |
| `tfg` | open | 4 | 0.5 | 27 | 0.423 ±0.0156 | 0.346 | 818.36 (813.20) | +816.18 (+810.05) | 23751.07 (23772.38) | 0.104 | 0.446 | 1.000 | f_B outliers (sd>100 d), below floor |
| `btvg` | floor | 0.01 | 0.05 | 9 | 0.041 ±0.0063 | 0.043 | 11.89 (11.86) | -10.59 (-10.49) | 8.51 (8.66) | 0.400 | 0.749 | 0.997 | w=min, t-open |
| `btvg` | open | 16 | 0.05 | 27 | 0.112 ±0.0100 | 0.105 | 5.96 (6.09) | -5.02 (-4.63) | 5.57 (6.08) | 0.150 | 0.467 | 1.000 | w=max, t-open, below floor |
| `btvg_var` | floor | 0.05 | 0.5 | 13 | 0.040 ±0.0062 | 0.039 | 12.10 (12.10) | -10.84 (-10.68) | 8.52 (8.69) | 0.401 | 0.755 | 1.000 | dec!= (w=0.01,t=0.05) |
| `btvg_var` | open (= floor) | 0.05 | 0.5 | 27 | 0.040 ±0.0062 | 0.039 | 12.10 (12.10) | -10.84 (-10.68) | 8.52 (8.69) | 0.401 | 0.755 | 1.000 | dec!= (w=0.01,t=0.05) |

**alpha** (delta 0.46754, floor 0.3780)

| arm | pick | w | t_min | k | in_band ±se | dec | MAE/d | bias/d | resid sd/d | mol_stab | valid | uniq | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `unguided` | -- | -- | -- | 1 | 0.028 ±0.0052 | 0.030 | 22.99 (22.88) | -21.25 (-21.07) | 18.70 (18.74) | 0.420 | 0.768 | 0.999 | reference |
| `plug` | floor | 8 | 0.75 | 11 | 0.032 ±0.0056 | 0.037 | 20.98 (20.82) | -19.36 (-19.10) | 17.27 (17.40) | 0.403 | 0.735 | 1.000 | t-edge |
| `plug` | open | 16 | 0.05 | 27 | 0.073 ±0.0082 | 0.073 | 11.40 (12.47) | -3.44 (-2.01) | 18.76 (19.74) | 0.147 | 0.401 | 0.983 | w=max, t-open, below floor |
| `tmpd` | floor | 0.5 | 0.05 | 17 | 0.042 ±0.0063 | 0.043 | 18.54 (18.50) | -15.54 (-15.43) | 17.10 (17.17) | 0.385 | 0.722 | 1.000 | t-open |
| `tmpd` | open | 16 | 0.5 | 27 | 0.061 ±0.0076 | 0.061 | 11.62 (11.62) | -9.47 (-9.07) | 12.04 (12.30) | 0.195 | 0.522 | 1.000 | w=max, below floor |
| `lgd_mc` | floor | 16 | 0.75 | 16 | 0.035 ±0.0058 | 0.039 | 20.17 (20.08) | -18.67 (-18.44) | 16.89 (17.12) | 0.395 | 0.728 | 1.000 | w=max, t-edge |
| `lgd_mc` | open | 4 | 0.05 | 27 | 0.062 ±0.0076 | 0.061 | 13.03 (13.13) | -9.61 (-9.43) | 14.00 (14.23) | 0.317 | 0.616 | 0.998 | t-open, below floor |
| `tfg` | floor | 0.01 | 0.05 | 6 | 0.034 ±0.0057 | 0.038 | 21.30 (21.21) | -19.30 (-19.15) | 17.94 (18.02) | 0.421 | 0.752 | 1.000 | w=min, t-open, TIE with w=0.25,t=0.75 |
| `tfg` | open | 8 | 0.05 | 27 | 0.298 ±0.0145 | 0.162 | 6.43 (8.68) | -3.45 (-2.38) | 14.51 (16.68) | 0.072 | 0.325 | 0.985 | t-open, dec!= (w=16,t=0.05), below floor |
| `btvg` | floor | 8 | 0.75 | 12 | 0.031 ±0.0055 | 0.035 | 20.99 (20.85) | -19.38 (-19.12) | 17.27 (17.43) | 0.402 | 0.728 | 1.000 | t-edge |
| `btvg` | open | 16 | 0.05 | 27 | 0.077 ±0.0084 | 0.073 | 9.81 (10.46) | -3.06 (-2.02) | 14.53 (15.40) | 0.138 | 0.380 | 0.997 | w=max, t-open, below floor |
| `btvg_var` | floor | 4 | 0.75 | 19 | 0.029 ±0.0053 | 0.030 | 23.04 (22.94) | -21.34 (-21.13) | 18.66 (18.73) | 0.416 | 0.770 | 0.999 | t-open, dec!= (w=0.01,t=0.75), TIE with w=8,t=0.75; w=16,t=0.75 |
| `btvg_var` | open (= floor) | 4 | 0.75 | 27 | 0.029 ±0.0053 | 0.030 | 23.04 (22.94) | -21.34 (-21.13) | 18.66 (18.73) | 0.416 | 0.770 | 0.999 | t-open, dec!= (w=0.01,t=0.75), TIE with w=8,t=0.75; w=16,t=0.75 |

**gap** (delta 0.00739, floor 0.3780)

| arm | pick | w | t_min | k | in_band ±se | dec | MAE/d | bias/d | resid sd/d | mol_stab | valid | uniq | flags |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `unguided` | -- | -- | -- | 1 | 0.058 ±0.0074 | 0.055 | 10.28 (10.36) | -10.07 (-10.16) | 6.17 (6.12) | 0.420 | 0.768 | 0.999 | reference |
| `plug` | floor | 2 | 0.75 | 10 | 0.068 ±0.0080 | 0.064 | 9.38 (9.53) | -9.14 (-9.31) | 5.94 (5.90) | 0.380 | 0.739 | 0.999 | t-edge, dec!= (w=0.25,t=0.5) |
| `plug` | open | 16 | 0.05 | 27 | 0.161 ±0.0116 | 0.137 | 4.94 (5.67) | -4.60 (-5.34) | 4.26 (4.66) | 0.166 | 0.506 | 1.000 | w=max, t-open, below floor |
| `tmpd` | floor | 2 | 0.75 | 14 | 0.065 ±0.0078 | 0.064 | 9.50 (9.60) | -9.26 (-9.38) | 5.95 (5.93) | 0.388 | 0.728 | 0.999 | t-edge |
| `tmpd` | open | 16 | 0.5 | 27 | 0.138 ±0.0109 | 0.125 | 4.98 (5.54) | -4.65 (-5.21) | 4.23 (4.58) | 0.208 | 0.526 | 1.000 | w=max, below floor |
| `lgd_mc` | floor | 2 | 0.5 | 20 | 0.094 ±0.0092 | 0.087 | 7.98 (8.10) | -7.64 (-7.78) | 5.79 (5.77) | 0.391 | 0.738 | 0.996 | dec!= (w=2,t=0.05), TIE with w=2,t=0.05 |
| `lgd_mc` | open | 16 | 0.05 | 27 | 0.211 ±0.0129 | 0.188 | 3.75 (4.06) | -2.90 (-3.22) | 4.04 (4.24) | 0.313 | 0.725 | 0.993 | w=max, t-open, below floor |
| `tfg` | floor | 0.05 | 0.75 | 4 | 0.071 ±0.0081 | 0.068 | 8.91 (9.15) | -8.67 (-8.93) | 5.86 (5.82) | 0.378 | 0.700 | 0.999 | t-edge |
| `tfg` | open | 8 | 0.5 | 27 | 0.337 ±0.0149 | 0.278 | 3.06 (3.67) | -2.70 (-3.28) | 3.47 (3.97) | 0.120 | 0.415 | 1.000 | below floor |
| `btvg` | floor | 2 | 0.75 | 10 | 0.073 ±0.0082 | 0.069 | 9.49 (9.62) | -9.29 (-9.43) | 5.95 (5.91) | 0.385 | 0.742 | 1.000 | t-edge |
| `btvg` | open | 16 | 0.5 | 27 | 0.097 ±0.0094 | 0.095 | 6.15 (6.42) | -5.94 (-6.22) | 4.30 (4.40) | 0.221 | 0.552 | 1.000 | w=max, below floor |
| `btvg_var` | floor | 0.01 | 0.05 | 13 | 0.064 ±0.0077 | 0.058 | 10.06 (10.15) | -9.86 (-9.95) | 6.23 (6.19) | 0.387 | 0.765 | 1.000 | w=min, t-open, dec!= (w=0.05,t=0.05) |
| `btvg_var` | open | 0.25 | 0.05 | 27 | 0.085 ±0.0088 | 0.086 | 9.20 (9.34) | -8.99 (-9.13) | 6.05 (5.98) | 0.352 | 0.716 | 0.999 | t-open, below floor |

Tallies over the 18 floor picks:

- at one of the two weakest strengths (w <= 0.05): 7 -- mu/lgd_mc, mu/tfg, mu/btvg, mu/btvg_var, alpha/tfg, gap/tfg, gap/btvg_var
- at the top of the strength grid while clearing the floor (grid-limited, a lower bound): 1 -- alpha/lgd_mc
- clearing the floor by less than one se of the stability proportion: 8 -- mu/plug (0.378 vs floor 0.3780, se 0.0153); alpha/tmpd (0.385 vs floor 0.3780, se 0.0154); gap/plug (0.380 vs floor 0.3780, se 0.0153); gap/tmpd (0.388 vs floor 0.3780, se 0.0154); gap/lgd_mc (0.391 vs floor 0.3780, se 0.0154); gap/tfg (0.378 vs floor 0.3780, se 0.0153); gap/btvg (0.385 vs floor 0.3780, se 0.0154); gap/btvg_var (0.387 vs floor 0.3780, se 0.0154)
- decided by an exact in-band tie (the tie-break, not the data, chose the cell): 4 -- alpha/tfg floor, alpha/btvg_var floor, alpha/btvg_var open, gap/lgd_mc floor
- the decoded in-band would pick a different cell: floor 6 (mu/lgd_mc, mu/btvg_var, alpha/btvg_var, gap/plug, gap/lgd_mc, gap/btvg_var); open 4 (mu/lgd_mc, mu/btvg_var, alpha/tfg, alpha/btvg_var)

### (b) What the chemistry floor costs: open minus floor (generated)

| property | arm | floor in_band | open in_band | open - floor | z | dec: open - floor | floor mol_stab | open mol_stab | stab given up | valid given up | uniq given up | open pick grid-capped? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mu | `plug` | 0.046 | 0.113 | +0.067 | +5.58 | +0.064 | 0.378 | 0.186 | -0.192 | -0.217 | +0.000 | yes (w=16) |
| mu | `tmpd` | 0.052 | 0.115 | +0.063 | +5.13 | +0.067 | 0.425 | 0.182 | -0.243 | -0.274 | +0.003 | yes (w=16) |
| mu | `lgd_mc` | 0.042 | 0.091 | +0.049 | +4.42 | +0.044 | 0.406 | 0.285 | -0.121 | -0.114 | +0.001 | yes (w=16) |
| mu | `tfg` | 0.048 | 0.423 | +0.375 | +22.03 | +0.305 | 0.405 | 0.104 | -0.301 | -0.306 | +0.001 | no |
| mu | `btvg` | 0.041 | 0.112 | +0.071 | +6.03 | +0.062 | 0.400 | 0.150 | -0.250 | -0.282 | +0.003 | yes (w=16) |
| mu | `btvg_var` | 0.040 | 0.040 | 0 (same cell) | -- | 0 | 0.401 | 0.401 | 0 | 0 | 0 | no |
| alpha | `plug` | 0.032 | 0.073 | +0.041 | +4.13 | +0.036 | 0.403 | 0.147 | -0.256 | -0.334 | -0.017 | yes (w=16) |
| alpha | `tmpd` | 0.042 | 0.061 | +0.019 | +1.92 | +0.018 | 0.385 | 0.195 | -0.190 | -0.200 | +0.000 | yes (w=16) |
| alpha | `lgd_mc` | 0.035 | 0.062 | +0.027 | +2.82 | +0.022 | 0.395 | 0.317 | -0.078 | -0.112 | -0.002 | no |
| alpha | `tfg` | 0.034 | 0.298 | +0.264 | +16.97 | +0.124 | 0.421 | 0.072 | -0.349 | -0.427 | -0.015 | no |
| alpha | `btvg` | 0.031 | 0.077 | +0.046 | +4.57 | +0.038 | 0.402 | 0.138 | -0.264 | -0.348 | -0.003 | yes (w=16) |
| alpha | `btvg_var` | 0.029 | 0.029 | 0 (same cell) | -- | 0 | 0.416 | 0.416 | 0 | 0 | 0 | no |
| gap | `plug` | 0.068 | 0.161 | +0.093 | +6.60 | +0.073 | 0.380 | 0.166 | -0.214 | -0.233 | +0.001 | yes (w=16) |
| gap | `tmpd` | 0.065 | 0.138 | +0.073 | +5.45 | +0.061 | 0.388 | 0.208 | -0.180 | -0.202 | +0.001 | yes (w=16) |
| gap | `lgd_mc` | 0.094 | 0.211 | +0.117 | +7.38 | +0.101 | 0.391 | 0.313 | -0.078 | -0.013 | -0.003 | yes (w=16) |
| gap | `tfg` | 0.071 | 0.337 | +0.266 | +15.64 | +0.210 | 0.378 | 0.120 | -0.258 | -0.285 | +0.001 | no |
| gap | `btvg` | 0.073 | 0.097 | +0.024 | +1.93 | +0.026 | 0.385 | 0.221 | -0.164 | -0.190 | +0.000 | yes (w=16) |
| gap | `btvg_var` | 0.064 | 0.085 | +0.021 | +1.79 | +0.028 | 0.387 | 0.352 | -0.035 | -0.049 | -0.001 | no |

Tallies over the 18 (property, arm) pairs: floor pick = open pick in 2; open pick at the top of the strength grid (w = 16, so the ceiling is a lower bound) in 11.

Where the picks differ (excluding `tfg`, 13 pairs): open - floor in-band +0.019 to +0.117; molecule stability given up -0.264 to -0.035.

Where the picks differ (`tfg` only, 3 pairs): open - floor in-band +0.264 to +0.375; molecule stability given up -0.349 to -0.258.

**Cells with non-finite samples or f_B outliers** (residual sd > 100 delta), anywhere in the grid (generated):

| property | arm | w | t_min | non-finite | resid sd/d | MAE/d | guide-eval gap max | in_band | mol_stab | a pick? |
|---|---|---|---|---|---|---|---|---|---|---|
| mu | `tfg` | 4 | 0.5 | 0 | 2.38e+04 | 818 | 1.96e+04 | 0.423 | 0.104 | open |
| mu | `tfg` | 8 | 0.5 | 1 | 1.02e+03 | 34.4 | 1.07e+04 | 0.371 | 0.076 | no |
| mu | `tfg` | 8 | 0.75 | 1 | 3.15 | 2.21 | 2.5 | 0.362 | 0.069 | no |
| mu | `tfg` | 16 | 0.05 | 0 | 4.02e+04 | 1.28e+03 | 3.34e+05 | 0.230 | 0.020 | no |
| mu | `tfg` | 16 | 0.5 | 1 | 7.03e+11 | 2.23e+10 | 1.93e+13 | 0.304 | 0.029 | no |
| mu | `tfg` | 16 | 0.75 | 1 | 71.1 | 6.23 | 59 | 0.268 | 0.023 | no |
| alpha | `tfg` | 2 | 0.75 | 0 | 5.58e+14 | 1.76e+13 | 8.99e+14 | 0.112 | 0.157 | no |
| alpha | `tfg` | 8 | 0.75 | 0 | 1.65e+14 | 5.21e+12 | 2.04e+14 | 0.246 | 0.076 | no |

8 of 489 cells.

### (c) Floor picks head-to-head: vs unguided and vs plug's floor pick (generated)

Unpaired iid z at n = 1000 per cell, one seed. Screen-level only.

| property | arm (floor pick) | in_band | vs unguided: diff | z | dec diff | dec z | stab diff | MAE/d diff | vs plug: diff | z | dec diff | dec z | stab diff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mu | `plug` (w=0.5, t=0.5) | 0.046 | +0.013 | +1.49 | +0.013 | +1.49 | -0.042 | -1.69 | -- | -- | -- | -- | -- |
| mu | `tmpd` (w=0.5, t=0.05) | 0.052 | +0.019 | +2.11 | +0.016 | +1.81 | +0.005 | -1.02 | +0.006 | +0.62 | +0.003 | +0.32 | +0.047 |
| mu | `lgd_mc` (w=0.01, t=0.05) | 0.042 | +0.009 | +1.06 | +0.003 | +0.37 | -0.014 | -0.05 | -0.004 | -0.44 | -0.010 | -1.13 | +0.028 |
| mu | `tfg` (w=0.05, t=0.05) | 0.048 | +0.015 | +1.70 | +0.008 | +0.95 | -0.015 | -1.53 | +0.002 | +0.21 | -0.005 | -0.55 | +0.027 |
| mu | `btvg` (w=0.01, t=0.05) | 0.041 | +0.008 | +0.95 | +0.010 | +1.17 | -0.020 | -0.13 | -0.005 | -0.55 | -0.003 | -0.33 | +0.022 |
| mu | `btvg_var` (w=0.05, t=0.5) | 0.040 | +0.007 | +0.83 | +0.006 | +0.72 | -0.019 | +0.09 | -0.006 | -0.66 | -0.007 | -0.78 | +0.023 |
| alpha | `plug` (w=8, t=0.75) | 0.032 | +0.004 | +0.52 | +0.007 | +0.87 | -0.017 | -2.02 | -- | -- | -- | -- | -- |
| alpha | `tmpd` (w=0.5, t=0.05) | 0.042 | +0.014 | +1.70 | +0.013 | +1.55 | -0.035 | -4.46 | +0.010 | +1.19 | +0.006 | +0.68 | -0.018 |
| alpha | `lgd_mc` (w=16, t=0.75) | 0.035 | +0.007 | +0.90 | +0.009 | +1.10 | -0.025 | -2.83 | +0.003 | +0.37 | +0.002 | +0.23 | -0.008 |
| alpha | `tfg` (w=0.01, t=0.05) | 0.034 | +0.006 | +0.77 | +0.008 | +0.99 | +0.001 | -1.70 | +0.002 | +0.25 | +0.001 | +0.12 | +0.018 |
| alpha | `btvg` (w=8, t=0.75) | 0.031 | +0.003 | +0.40 | +0.005 | +0.63 | -0.018 | -2.01 | -0.001 | -0.13 | -0.002 | -0.24 | -0.001 |
| alpha | `btvg_var` (w=4, t=0.75) | 0.029 | +0.001 | +0.13 | +0.000 | +0.00 | -0.004 | +0.04 | -0.003 | -0.39 | -0.007 | -0.87 | +0.013 |
| gap | `plug` (w=2, t=0.75) | 0.068 | +0.010 | +0.92 | +0.009 | +0.85 | -0.040 | -0.90 | -- | -- | -- | -- | -- |
| gap | `tmpd` (w=2, t=0.75) | 0.065 | +0.007 | +0.65 | +0.009 | +0.85 | -0.032 | -0.78 | -0.003 | -0.27 | +0.000 | +0.00 | +0.008 |
| gap | `lgd_mc` (w=2, t=0.5) | 0.094 | +0.036 | +3.04 | +0.032 | +2.79 | -0.029 | -2.30 | +0.026 | +2.13 | +0.023 | +1.95 | +0.011 |
| gap | `tfg` (w=0.05, t=0.75) | 0.071 | +0.013 | +1.18 | +0.013 | +1.21 | -0.042 | -1.37 | +0.003 | +0.26 | +0.004 | +0.36 | -0.002 |
| gap | `btvg` (w=2, t=0.75) | 0.073 | +0.015 | +1.36 | +0.014 | +1.30 | -0.035 | -0.79 | +0.005 | +0.44 | +0.005 | +0.45 | +0.005 |
| gap | `btvg_var` (w=0.01, t=0.05) | 0.064 | +0.006 | +0.56 | +0.003 | +0.29 | -0.033 | -0.22 | -0.004 | -0.36 | -0.006 | -0.56 | +0.007 |

### (d) The start-time effect (generated)

**d1. Best floor-clearing in-band at each t_min** (the strength that achieves it in brackets; `--` = no cell at that t_min clears the floor). Bold = the arm's floor pick.

| property | arm | t=0.05 | t=0.5 | t=0.75 | best - worst t_min | z |
|---|---|---|---|---|---|---|
| mu | `plug` | 0.045 (w=0.01) | **0.046 (w=0.5)** | 0.036 (w=0.5) | +0.010 | +1.13 |
| mu | `tmpd` | **0.052 (w=0.5)** | 0.043 (w=0.5) | 0.037 (w=0.5) | +0.015 | +1.63 |
| mu | `lgd_mc` | **0.042 (w=0.01)** | 0.039 (w=1) | 0.038 (w=2) | +0.004 | +0.46 |
| mu | `tfg` | **0.048 (w=0.05)** | 0.046 (w=0.05) | 0.037 (w=0.05) | +0.011 | +1.22 |
| mu | `btvg` | **0.041 (w=0.01)** | 0.040 (w=0.05) | 0.039 (w=0.5) | +0.002 | +0.23 |
| mu | `btvg_var` | 0.039 (w=0.01) | **0.040 (w=0.05)** | 0.035 (w=0.01) | +0.005 | +0.59 |
| alpha | `plug` | -- | 0.030 (w=0.01) | **0.032 (w=8)** | +0.002 | +0.26 |
| alpha | `tmpd` | **0.042 (w=0.5)** | 0.032 (w=0.05) | 0.031 (w=4) | +0.011 | +1.31 |
| alpha | `lgd_mc` | 0.033 (w=0.25) | 0.027 (w=0.01) | **0.035 (w=16)** | +0.008 | +1.03 |
| alpha | `tfg` | **0.034 (w=0.01)** | 0.031 (w=0.01) | 0.034 (w=0.25) | +0.003 | +0.38 |
| alpha | `btvg` | -- | 0.029 (w=0.01) | **0.031 (w=8)** | +0.002 | +0.26 |
| alpha | `btvg_var` | 0.026 (w=0.01) | 0.028 (w=0.01) | **0.029 (w=4)** | +0.003 | +0.41 |
| gap | `plug` | 0.060 (w=0.01) | 0.066 (w=0.25) | **0.068 (w=2)** | +0.008 | +0.73 |
| gap | `tmpd` | 0.063 (w=0.05) | 0.061 (w=0.5) | **0.065 (w=2)** | +0.004 | +0.37 |
| gap | `lgd_mc` | 0.094 (w=2) | **0.094 (w=2)** | 0.072 (w=8) | +0.022 | +1.78 |
| gap | `tfg` | 0.064 (w=0.01) | 0.067 (w=0.01) | **0.071 (w=0.05)** | +0.007 | +0.62 |
| gap | `btvg` | 0.058 (w=0.01) | 0.058 (w=0.05) | **0.073 (w=2)** | +0.015 | +1.36 |
| gap | `btvg_var` | **0.064 (w=0.01)** | 0.058 (w=0.05) | 0.063 (w=1) | +0.006 | +0.56 |

**d2. Best unconstrained in-band at each t_min** (strength in brackets; mol_stab after the slash).

| property | arm | t=0.05 | t=0.5 | t=0.75 |
|---|---|---|---|---|
| mu | `plug` | 0.105 (w=16) / 0.132 | 0.113 (w=16) / 0.186 | 0.065 (w=16) / 0.267 |
| mu | `tmpd` | 0.083 (w=16) / 0.121 | 0.115 (w=16) / 0.182 | 0.065 (w=16) / 0.269 |
| mu | `lgd_mc` | 0.091 (w=16) / 0.285 | 0.085 (w=16) / 0.284 | 0.047 (w=8) / 0.345 |
| mu | `tfg` | 0.331 (w=2) / 0.105 | 0.423 (w=4) / 0.104 | 0.400 (w=2) / 0.106 |
| mu | `btvg` | 0.112 (w=16) / 0.150 | 0.081 (w=16) / 0.162 | 0.052 (w=16) / 0.278 |
| mu | `btvg_var` | 0.039 (w=0.01) / 0.423 | 0.040 (w=0.05) / 0.401 | 0.035 (w=0.01) / 0.421 |
| alpha | `plug` | 0.073 (w=16) / 0.147 | 0.068 (w=16) / 0.196 | 0.039 (w=16) / 0.377 |
| alpha | `tmpd` | 0.059 (w=16) / 0.139 | 0.061 (w=16) / 0.195 | 0.040 (w=16) / 0.376 |
| alpha | `lgd_mc` | 0.062 (w=4) / 0.317 | 0.044 (w=16) / 0.327 | 0.035 (w=16) / 0.395 |
| alpha | `tfg` | 0.298 (w=8) / 0.072 | 0.257 (w=8) / 0.095 | 0.260 (w=4) / 0.105 |
| alpha | `btvg` | 0.077 (w=16) / 0.138 | 0.064 (w=8) / 0.312 | 0.038 (w=16) / 0.374 |
| alpha | `btvg_var` | 0.026 (w=0.01) / 0.399 | 0.028 (w=0.01) / 0.418 | 0.029 (w=4) / 0.416 |
| gap | `plug` | 0.161 (w=16) / 0.166 | 0.152 (w=16) / 0.220 | 0.091 (w=16) / 0.305 |
| gap | `tmpd` | 0.120 (w=16) / 0.142 | 0.138 (w=16) / 0.208 | 0.082 (w=16) / 0.313 |
| gap | `lgd_mc` | 0.211 (w=16) / 0.313 | 0.184 (w=16) / 0.293 | 0.081 (w=16) / 0.364 |
| gap | `tfg` | 0.308 (w=4) / 0.114 | 0.337 (w=8) / 0.120 | 0.193 (w=2) / 0.117 |
| gap | `btvg` | 0.092 (w=1) / 0.268 | 0.097 (w=16) / 0.221 | 0.082 (w=16) / 0.335 |
| gap | `btvg_var` | 0.085 (w=0.25) / 0.352 | 0.074 (w=4) / 0.269 | 0.063 (w=1) / 0.405 |

Which start time gives the highest / lowest unconstrained ceiling, over the 18 pairs: highest t=0.05: 9, t=0.5: 8, t=0.75: 1; lowest t=0.05: 2, t=0.5: 1, t=0.75: 15.

**d3. When vs how, inside the chemistry floor.** For each arm, among floor-clearing cells only: *when* = best in-band at the best t_min minus best in-band at the worst t_min (each t_min at its own best strength; from d1). *how* = the same range taken over strengths (each strength at its own best t_min; strengths with no floor-clearing cell are skipped). z is the unpaired z of the two cells that define each range. **The two ranges are NOT directly comparable:** *when* is a range over at most 3 groups and *how* over up to 9, and the range of more noisy draws is larger by construction. So each range is set against an illustrative noise null for its own number of groups k: the expected range of k iid normal draws, 2 x E[max of k N(0,1)] x se, with se = sqrt(p(1-p)/n) at p = the mean of the k group maxima. It treats the group maxima as iid draws with one true in-band, which they are not (they are themselves maxima, and share a seed), so it is a yardstick, not a test.

| property | arm | when: range | z | k | null range | how: range | z | k | null range |
|---|---|---|---|---|---|---|---|---|---|
| mu | `plug` | +0.010 | +1.13 | 3 | 0.011 | +0.011 | +1.25 | 5 | 0.014 |
| mu | `tmpd` | +0.015 | +1.63 | 3 | 0.011 | +0.019 | +2.11 | 5 | 0.014 |
| mu | `lgd_mc` | +0.004 | +0.46 | 3 | 0.010 | +0.004 | +0.46 | 6 | 0.016 |
| mu | `tfg` | +0.011 | +1.22 | 3 | 0.011 | +0.012 | +1.34 | 2 | 0.007 |
| mu | `btvg` | +0.002 | +0.23 | 3 | 0.010 | +0.006 | +0.70 | 5 | 0.014 |
| mu | `btvg_var` | +0.005 | +0.59 | 3 | 0.010 | +0.011 | +1.35 | 9 | 0.017 |
| alpha | `plug` | +0.002 | +0.26 | 2 | 0.006 | +0.004 | +0.52 | 8 | 0.015 |
| alpha | `tmpd` | +0.011 | +1.31 | 3 | 0.010 | +0.013 | +1.57 | 8 | 0.016 |
| alpha | `lgd_mc` | +0.008 | +1.03 | 3 | 0.009 | +0.007 | +0.90 | 9 | 0.016 |
| alpha | `tfg` | +0.003 | +0.38 | 3 | 0.010 | +0.005 | +0.64 | 3 | 0.009 |
| alpha | `btvg` | +0.002 | +0.26 | 2 | 0.006 | +0.003 | +0.40 | 8 | 0.015 |
| alpha | `btvg_var` | +0.003 | +0.41 | 3 | 0.009 | +0.002 | +0.27 | 9 | 0.016 |
| gap | `plug` | +0.008 | +0.73 | 3 | 0.013 | +0.008 | +0.73 | 6 | 0.020 |
| gap | `tmpd` | +0.004 | +0.37 | 3 | 0.013 | +0.006 | +0.56 | 6 | 0.019 |
| gap | `lgd_mc` | +0.022 | +1.78 | 3 | 0.015 | +0.035 | +2.95 | 8 | 0.023 |
| gap | `tfg` | +0.007 | +0.62 | 3 | 0.013 | +0.004 | +0.35 | 2 | 0.009 |
| gap | `btvg` | +0.015 | +1.36 | 3 | 0.013 | +0.013 | +1.17 | 7 | 0.021 |
| gap | `btvg_var` | +0.006 | +0.56 | 3 | 0.013 | +0.006 | +0.56 | 9 | 0.022 |

Over the 18 (property, arm) pairs: the largest *when* z is +1.78 (gap/lgd_mc) and the largest *how* z is +2.95 (gap/lgd_mc); a range exceeds its noise null in 5 pairs for *when* and 3 for *how*.

Where the picks land in t_min, over the 18 (property, arm) pairs: floor picks t=0.05: 7, t=0.5: 3, t=0.75: 8; open picks t=0.05: 9, t=0.5: 8, t=0.75: 1.

**d4. Every arm at the fixed strength w = 1, across t_min** -- the shape of the project's original window comparison (one strength, several start times). Each cell: in_band (z vs unguided) / MAE/d / mol_stab.

| property | arm | t=0.05 | t=0.5 | t=0.75 |
|---|---|---|---|---|
| mu | `plug` | 0.046 (+1.49) / 10.34 / 0.241 | 0.054 (+2.31) / 9.58 / 0.331 | 0.035 (+0.25) / 11.02 / 0.381 |
| mu | `tmpd` | 0.051 (+2.01) / 10.34 / 0.352 | 0.049 (+1.81) / 10.12 / 0.367 | 0.035 (+0.25) / 11.24 / 0.388 |
| mu | `lgd_mc` | 0.041 (+0.95) / 11.49 / 0.386 | 0.039 (+0.72) / 11.37 / 0.412 | 0.035 (+0.25) / 11.47 / 0.406 |
| mu | `tfg` | 0.295 (+16.92) / 2.45 / 0.119 | 0.322 (+18.27) / 2.25 / 0.130 | 0.343 (+19.33) / 2.11 / 0.127 |
| mu | `btvg` | 0.037 (+0.49) / 10.52 / 0.262 | 0.042 (+1.06) / 10.67 / 0.257 | 0.035 (+0.25) / 11.20 / 0.384 |
| mu | `btvg_var` | 0.019 (-1.97) / 13.57 / 0.342 | 0.022 (-1.50) / 13.01 / 0.329 | 0.034 (+0.12) / 11.99 / 0.416 |
| alpha | `plug` | 0.028 (+0.00) / 18.22 / 0.242 | 0.048 (+2.34) / 16.92 / 0.350 | 0.029 (+0.13) / 22.47 / 0.417 |
| alpha | `tmpd` | 0.048 (+2.34) / 15.96 / 0.340 | 0.030 (+0.27) / 18.29 / 0.386 | 0.029 (+0.13) / 22.50 / 0.414 |
| alpha | `lgd_mc` | 0.047 (+2.24) / 16.62 / 0.360 | 0.021 (-1.01) / 20.60 / 0.394 | 0.029 (+0.13) / 22.53 / 0.419 |
| alpha | `tfg` | 0.125 (+8.30) / 7.83 / 0.145 | 0.092 (+6.08) / 7.63 / 0.170 | 0.056 (+3.13) / 10.97 / 0.247 |
| alpha | `btvg` | 0.044 (+1.92) / 15.22 / 0.227 | 0.028 (+0.00) / 18.07 / 0.366 | 0.029 (+0.13) / 22.47 / 0.417 |
| alpha | `btvg_var` | 0.018 (-1.49) / 22.75 / 0.368 | 0.019 (-1.33) / 25.10 / 0.405 | 0.027 (-0.14) / 23.03 / 0.419 |
| gap | `plug` | 0.055 (-0.29) / 9.18 / 0.233 | 0.085 (+2.35) / 7.72 / 0.300 | 0.064 (+0.56) / 9.61 / 0.399 |
| gap | `tmpd` | 0.062 (+0.38) / 9.60 / 0.347 | 0.062 (+0.38) / 8.94 / 0.372 | 0.063 (+0.47) / 9.81 / 0.404 |
| gap | `lgd_mc` | 0.082 (+2.11) / 8.91 / 0.391 | 0.074 (+1.44) / 9.00 / 0.398 | 0.061 (+0.28) / 9.79 / 0.420 |
| gap | `tfg` | 0.259 (+12.80) / 3.43 / 0.138 | 0.291 (+14.42) / 3.18 / 0.151 | 0.155 (+7.12) / 5.36 / 0.149 |
| gap | `btvg` | 0.092 (+2.89) / 8.12 / 0.268 | 0.071 (+1.18) / 8.25 / 0.281 | 0.066 (+0.74) / 9.76 / 0.403 |
| gap | `btvg_var` | 0.074 (+1.44) / 8.96 / 0.273 | 0.056 (-0.19) / 9.22 / 0.288 | 0.063 (+0.47) / 10.11 / 0.405 |

### Resolving power and selection bias (generated)

| property | unguided in_band | its se at n = 1000 | se of a difference of two such cells |
|---|---|---|---|
| mu | 0.033 | 0.0056 | 0.0080 |
| alpha | 0.028 | 0.0052 | 0.0074 |
| gap | 0.058 | 0.0074 | 0.0105 |

Expected maximum of k iid N(0, 1) draws (the upward bias of a pick, in se units, if all k candidates had the same true in-band and were independent; the cells share a seed, so this is an illustration, not a correction): k=4: 1.03, k=6: 1.27, k=9: 1.49, k=10: 1.54, k=11: 1.59, k=12: 1.63, k=13: 1.67, k=14: 1.70, k=16: 1.77, k=17: 1.79, k=19: 1.84, k=20: 1.87, k=27: 2.00.

Candidate-set sizes seen at the floor picks: mu/plug 10, mu/tmpd 13, mu/lgd_mc 17, mu/tfg 6, mu/btvg 9, mu/btvg_var 13, alpha/plug 11, alpha/tmpd 17, alpha/lgd_mc 16, alpha/tfg 6, alpha/btvg 12, alpha/btvg_var 19, gap/plug 10, gap/tmpd 14, gap/lgd_mc 20, gap/tfg 4, gap/btvg 10, gap/btvg_var 13.

<!-- END generated by proj1/scripts/tune_table.py -->

## What each outcome shows

1. **The sweep is complete and clean, apart from `tfg` at high strength.** It has 489 / 489 cells and no failures. All 8 cells with non-finite samples or f_B outliers are `tfg` cells. One of them is `tfg`'s open pick on mu.

2. **The floor is the same number, 0.378, on all three properties.** The unguided cell is one generation scored three ways (molecule stability 0.420), since unguided sampling ignores the target.

3. **Inside the chemistry floor, guidance barely moves in-band at q90.** Floor picks sit +0.001 to +0.036 above unguided (table c). Only `gap`/`lgd_mc` crosses z = 3 (+0.036, z +3.04), and even that is not evidence of an effect, for three reasons. On the decoded metric it drops to z +2.79. It is the maximum of k = 20 candidate cells, and under independence the expected selection inflation at k = 20 is 1.87 se. And it is one of 18 floor picks compared against unguided. No arm separates from `plug`; the largest gap is `gap`/`lgd_mc` at +0.026, z +2.13. Two further findings make the floor picks fragile. 7 of 18 floor picks sit at w <= 0.05: for those arms the floor lets through almost no guidance. And 8 of 18 clear the floor by less than one se of the stability proportion. At n = 1000, which cell is "the floor pick" is itself noisy.

4. **The floor costs a lot, and the measured ceiling is only a lower bound.** Excluding `tfg`, where the picks differ, `open` beats `floor` by +0.019 to +0.117 in-band, at a molecule-stability cost of -0.035 to -0.264. 11 of the 18 open picks sit at the top of the strength grid (w = 16), so the true unconstrained ceiling is higher than measured. Only `btvg_var` on mu and alpha has open = floor. The largest non-`tfg` gain is `lgd_mc` on gap: +0.117 (z +7.38) for -0.078 stability.

5. **`tfg`'s open-pick in-band should not be read as coverage.** Its open picks gain +0.264 to +0.375 in-band, but at molecule stability 0.072 to 0.120. Around them the evaluator returns absurd values: the mu open pick has residual sd 2.38e4 δ and MAE 818 δ, and two alpha cells (w = 2 and w = 8, both at t_min 0.75, neither a pick) have residual sd of 5.58e14 δ and 1.65e14 δ. On the decoded metric the alpha open pick falls from 0.298 to 0.162.

6. **`btvg_var` does not raise in-band on mu or alpha anywhere in the grid.** Its best cell clears the floor (open = floor) and sits at unguided's level: +0.007 (z +0.83) on mu, +0.001 (z +0.13) on alpha. At w = 1 it is below unguided on mu at t_min 0.05 and 0.5 (z -1.97, -1.50).

7. **Start time: at n = 1000, "when you guide matters more than how" does not hold inside the floor.** Among floor-clearing cells, neither knob moves in-band by a resolvable amount (table d3).
   - No start-time range and no strength range reaches z = 3. The largest start-time z is +1.78 and the largest strength z is +2.95, both for `gap`/`lgd_mc`.
   - The two ranges cannot be compared head-to-head: the strength range is taken over up to 9 groups and the start-time range over at most 3, so the strength range is larger by construction. Against a noise yardstick matched to each range's number of groups, the start-time range exceeds its yardstick in 5 of 18 pairs and the strength range in 3. That supports neither knob dominating.
   - The floor picks spread over all three start times (7 at 0.05, 3 at 0.5, 8 at 0.75). These counts partly reflect the tie-break: 4 picks were decided by exact in-band ties (listed under table a), which the pre-registered rule breaks toward smaller w first and the later start second.
   - **What does hold is a statement about the ceiling.** Starting late caps what strong guidance can reach: t_min = 0.75 gives the lowest unconstrained best in 15 of 18 pairs and the highest in 1. The earlier two start times split the highest (9 and 8).
   - **Fixed-strength cut.** At w = 1, `plug` on alpha scores in-band 0.028 (z +0.00) at t_min 0.05, 0.048 (z +2.34) at 0.5, and 0.029 at 0.75. The sign matches the earlier result (0.05 worse than 0.5), but at 0.05 `plug` is *not* worse than no guidance: its MAE/δ is 18.22 against unguided's 22.99. The project's −8.2σ / +4.5σ figures came from a different experiment. They were MAE σ against unguided, on the **q50** target, at n = 512 (SCOPE_FM_GUIDANCE_STATUS.md §8 for t_min 0.05: seed 20260921, w = 1; §10c for t_min 0.5: seeds 20260921 and 20260922). The target, metric and δ all differ, so this run neither replicates nor refutes them.

8. **At q90, bias dominates the error, and the floor leaves most of it in place.** Unguided bias is -10.45 / -21.25 / -10.07 δ, against a residual sd of 9.00 / 18.70 / 6.17 δ (mu / alpha / gap). The best floor pick on gap (`lgd_mc`) still carries -7.64 δ of bias. The open picks remove much more of it, for example `plug` on alpha at -3.44 δ, but at open-pick chemistry.

9. **The continuous and decoded metrics disagree on 6 of 18 floor picks and 4 of 18 open picks.** Selection used the continuous metric (v2's rule). The flagged picks are listed under table (a).

## Caveats

- **q90 only.** The paper reports q50 statistics only (scope fixed 26 Sep), so nothing on this page goes into the paper or the slides as a number. That includes the floor cost, the head-to-heads and the start-time finding.
- **A screen, not a verdict.** n = 1000 x 1 seed. At unguided's level the in-band se is 0.0056 / 0.0052 / 0.0074 (mu / alpha / gap), and the se of a difference between two such cells is 0.0080 / 0.0074 / 0.0105. Every z here is unpaired iid and screen-level. The floor itself (0.9 x one unguided cell) has an se of its own, given in the unguided table, and 8 of 18 floor picks clear it by less than one se. v2 already showed that a screen pick can fail the floor at full scale (`lgd_mc`, n = 512 to n = 5000).
- **Winner's curse.** Each pick is the maximum over k candidate cells: 4 to 20 at the floor, 27 for open. If all k had the same true in-band and were independent, the expected maximum would sit 1.03 se (k = 4) to 2.00 se (k = 27) above it. The cells share a seed, so this illustrates the size of the bias rather than correcting for it. Stage 2's disjoint seeds were the planned remedy, and stage 2 never ran.
- **Ties.** 4 picks were decided by an exact in-band tie with other candidate cells: `alpha`/`tfg` floor, `alpha`/`btvg_var` floor and open, and `gap`/`lgd_mc` floor. Their (w, t_min), their edge flags, and the start-time counts that include them were set by the pre-registered tie-break (smaller w, then larger t_min), not by the data.
- **δ differs from v2.** The local δ is +1.2 % / -2.9 % / -2.8 % against v2's pre-registered δ, and it also sets BTVG's τ (δ / 1.96). So in-band here is not comparable to any v2 table, and the `btvg` / `btvg_var` cells are different experiments from v2's. The plan records that this δ was adopted after its effect on v2's `plug`-on-gap headline had been seen (TUNE_SWEEP_PLAN.md §2).
- **No per-molecule sidecars.** Bias and residual sd come from the cell aggregates (bias = f_B mean - target, sd = sqrt(RMSE² - bias²), over finite molecules). This is exact for a fixed target, but a paired z or a cluster-robust se cannot be computed. All cells share a seed, so the unpaired z ignores a correlation whose size and sign cannot be measured here.
- **Grid edges.** 11 of 18 open picks are at w = 16, and `alpha`/`lgd_mc`'s floor pick is at w = 16 while still clearing the floor. Those numbers are lower bounds. Start-time edges are flagged per pick in table (a).
- **The selection metric is the continuous in-band** (v2's rule). Where the decoded metric would pick differently, the pick is flagged.
- **f_B's error on generated molecules is unmeasured.** δ is f_B's error on real val molecules. Among generated molecules, only 0.420 are molecule-stable unguided, the floor picks sit at 0.378 to 0.425, and the open picks go down to 0.072. This affects every δ equally.
- **To get stage 2 now**, the freeze and the full run must be submitted by hand, following TUNE_SWEEP_PLAN.md §7 ("If the freeze refuses": the freeze plus the chained full run) and the five steps in [BETTY_RUNBOOK.md](../protocol/BETTY_RUNBOOK.md). Nothing on this page was produced by stage 2.

<!-- BEGIN appendix generated by proj1/scripts/tune_table.py -- do not hand-edit -->

### Appendix: the full grid (generated)

Each cell: `in_band (dec) / mol_stab`. **Bold** = clears the floor. `[F]` floor pick, `[O]` open pick.

**mu** (floor 0.3780; unguided 0.033 (0.033) / 0.420)

`plug`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.045 (0.035) / 0.390** | **0.033 (0.032) / 0.420** | **0.034 (0.032) / 0.420** |
| 0.05 | **0.035 (0.039) / 0.386** | **0.040 (0.035) / 0.402** | **0.034 (0.033) / 0.411** |
| 0.25 | 0.042 (0.044) / 0.352 | 0.041 (0.045) / 0.376 | **0.035 (0.034) / 0.401** |
| 0.5 | 0.048 (0.054) / 0.305 | **0.046 (0.046) / 0.378** `[F]` | **0.036 (0.036) / 0.404** |
| 1 | 0.046 (0.058) / 0.241 | 0.054 (0.059) / 0.331 | **0.035 (0.035) / 0.381** |
| 2 | 0.048 (0.047) / 0.212 | 0.066 (0.066) / 0.267 | 0.040 (0.040) / 0.368 |
| 4 | 0.056 (0.052) / 0.173 | 0.082 (0.079) / 0.217 | 0.044 (0.046) / 0.335 |
| 8 | 0.087 (0.080) / 0.151 | 0.095 (0.087) / 0.194 | 0.049 (0.052) / 0.305 |
| 16 | 0.105 (0.105) / 0.132 | 0.113 (0.110) / 0.186 `[O]` | 0.065 (0.070) / 0.267 |

`tmpd`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.032 (0.032) / 0.418** | **0.032 (0.032) / 0.419** | **0.033 (0.033) / 0.420** |
| 0.05 | **0.034 (0.034) / 0.410** | **0.035 (0.033) / 0.419** | **0.033 (0.032) / 0.414** |
| 0.25 | **0.039 (0.039) / 0.415** | **0.042 (0.042) / 0.403** | **0.034 (0.032) / 0.402** |
| 0.5 | **0.052 (0.049) / 0.425** `[F]` | **0.043 (0.043) / 0.402** | **0.037 (0.036) / 0.403** |
| 1 | 0.051 (0.053) / 0.352 | 0.049 (0.053) / 0.367 | **0.035 (0.035) / 0.388** |
| 2 | 0.045 (0.051) / 0.293 | 0.051 (0.052) / 0.321 | 0.039 (0.036) / 0.365 |
| 4 | 0.068 (0.066) / 0.200 | 0.074 (0.076) / 0.252 | 0.046 (0.041) / 0.342 |
| 8 | 0.076 (0.076) / 0.164 | 0.096 (0.084) / 0.225 | 0.051 (0.049) / 0.310 |
| 16 | 0.083 (0.081) / 0.121 | 0.115 (0.116) / 0.182 `[O]` | 0.065 (0.066) / 0.269 |

`lgd_mc`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.042 (0.036) / 0.406** `[F]` | **0.032 (0.032) / 0.420** | **0.034 (0.033) / 0.420** |
| 0.05 | **0.040 (0.043) / 0.409** | **0.034 (0.033) / 0.414** | **0.034 (0.032) / 0.419** |
| 0.25 | **0.036 (0.033) / 0.406** | **0.038 (0.034) / 0.415** | **0.034 (0.033) / 0.407** |
| 0.5 | **0.032 (0.035) / 0.406** | **0.038 (0.034) / 0.413** | **0.035 (0.032) / 0.406** |
| 1 | **0.041 (0.042) / 0.386** | **0.039 (0.036) / 0.412** | **0.035 (0.030) / 0.406** |
| 2 | **0.028 (0.033) / 0.380** | 0.052 (0.045) / 0.357 | **0.038 (0.037) / 0.392** |
| 4 | 0.057 (0.048) / 0.346 | 0.055 (0.058) / 0.339 | 0.044 (0.038) / 0.371 |
| 8 | 0.065 (0.066) / 0.286 | 0.066 (0.063) / 0.302 | 0.047 (0.046) / 0.345 |
| 16 | 0.091 (0.080) / 0.285 `[O]` | 0.085 (0.080) / 0.284 | 0.047 (0.056) / 0.312 |

`tfg`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.034 (0.035) / 0.409** | **0.035 (0.032) / 0.413** | **0.036 (0.032) / 0.415** |
| 0.05 | **0.048 (0.041) / 0.405** `[F]` | **0.046 (0.037) / 0.403** | **0.037 (0.035) / 0.402** |
| 0.25 | 0.100 (0.071) / 0.262 | 0.093 (0.071) / 0.277 | 0.082 (0.063) / 0.279 |
| 0.5 | 0.183 (0.114) / 0.149 | 0.206 (0.127) / 0.177 | 0.183 (0.109) / 0.180 |
| 1 | 0.295 (0.224) / 0.119 | 0.322 (0.216) / 0.130 | 0.343 (0.215) / 0.127 |
| 2 | 0.331 (0.245) / 0.105 | 0.370 (0.274) / 0.106 | 0.400 (0.279) / 0.106 |
| 4 | 0.331 (0.228) / 0.080 | 0.423 (0.346) / 0.104 `[O]` | 0.390 (0.307) / 0.085 |
| 8 | 0.298 (0.237) / 0.040 | 0.371 (0.322) / 0.076 | 0.362 (0.286) / 0.069 |
| 16 | 0.230 (0.202) / 0.020 | 0.304 (0.235) / 0.029 | 0.268 (0.209) / 0.023 |

`btvg`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.041 (0.043) / 0.400** `[F]` | **0.034 (0.032) / 0.412** | **0.035 (0.034) / 0.421** |
| 0.05 | **0.032 (0.033) / 0.390** | **0.040 (0.037) / 0.401** | **0.035 (0.034) / 0.418** |
| 0.25 | 0.031 (0.030) / 0.332 | 0.027 (0.031) / 0.331 | **0.036 (0.033) / 0.413** |
| 0.5 | 0.032 (0.034) / 0.299 | 0.040 (0.041) / 0.327 | **0.039 (0.035) / 0.401** |
| 1 | 0.037 (0.041) / 0.262 | 0.042 (0.043) / 0.257 | **0.035 (0.032) / 0.384** |
| 2 | 0.047 (0.045) / 0.239 | 0.049 (0.052) / 0.237 | 0.037 (0.039) / 0.365 |
| 4 | 0.052 (0.051) / 0.183 | 0.067 (0.069) / 0.226 | 0.040 (0.038) / 0.345 |
| 8 | 0.076 (0.068) / 0.184 | 0.071 (0.064) / 0.192 | 0.049 (0.044) / 0.328 |
| 16 | 0.112 (0.105) / 0.150 `[O]` | 0.081 (0.084) / 0.162 | 0.052 (0.053) / 0.278 |

`btvg_var`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.039 (0.040) / 0.423** | **0.037 (0.037) / 0.412** | **0.035 (0.034) / 0.421** |
| 0.05 | **0.033 (0.038) / 0.408** | **0.040 (0.039) / 0.401** `[F]` `[O]` | **0.034 (0.033) / 0.421** |
| 0.25 | 0.023 (0.022) / 0.347 | 0.029 (0.034) / 0.358 | **0.033 (0.032) / 0.418** |
| 0.5 | 0.022 (0.026) / 0.330 | 0.021 (0.023) / 0.338 | **0.034 (0.031) / 0.415** |
| 1 | 0.019 (0.019) / 0.342 | 0.022 (0.028) / 0.329 | **0.034 (0.032) / 0.416** |
| 2 | 0.018 (0.018) / 0.296 | 0.020 (0.023) / 0.317 | **0.035 (0.034) / 0.415** |
| 4 | 0.028 (0.029) / 0.329 | 0.024 (0.023) / 0.271 | **0.032 (0.029) / 0.405** |
| 8 | 0.028 (0.028) / 0.302 | 0.026 (0.028) / 0.297 | **0.029 (0.029) / 0.393** |
| 16 | 0.023 (0.026) / 0.280 | 0.027 (0.031) / 0.288 | **0.035 (0.030) / 0.393** |

**alpha** (floor 0.3780; unguided 0.028 (0.030) / 0.420)

`plug`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | 0.030 (0.029) / 0.332 | **0.030 (0.031) / 0.420** | **0.028 (0.030) / 0.420** |
| 0.05 | 0.028 (0.031) / 0.299 | **0.029 (0.031) / 0.422** | **0.028 (0.030) / 0.419** |
| 0.25 | 0.032 (0.030) / 0.280 | **0.026 (0.026) / 0.404** | **0.028 (0.031) / 0.412** |
| 0.5 | 0.028 (0.029) / 0.267 | 0.029 (0.028) / 0.375 | **0.029 (0.031) / 0.413** |
| 1 | 0.028 (0.030) / 0.242 | 0.048 (0.049) / 0.350 | **0.029 (0.031) / 0.417** |
| 2 | 0.029 (0.026) / 0.210 | 0.050 (0.048) / 0.297 | **0.029 (0.033) / 0.422** |
| 4 | 0.040 (0.038) / 0.176 | 0.055 (0.061) / 0.267 | **0.031 (0.032) / 0.423** |
| 8 | 0.049 (0.047) / 0.161 | 0.053 (0.054) / 0.233 | **0.032 (0.037) / 0.403** `[F]` |
| 16 | 0.073 (0.073) / 0.147 `[O]` | 0.068 (0.072) / 0.196 | 0.039 (0.041) / 0.377 |

`tmpd`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.029 (0.032) / 0.423** | **0.029 (0.031) / 0.421** | **0.028 (0.030) / 0.420** |
| 0.05 | **0.031 (0.032) / 0.406** | **0.032 (0.032) / 0.422** | **0.028 (0.030) / 0.419** |
| 0.25 | **0.036 (0.033) / 0.407** | **0.029 (0.029) / 0.422** | **0.028 (0.031) / 0.412** |
| 0.5 | **0.042 (0.043) / 0.385** `[F]` | **0.027 (0.028) / 0.405** | **0.029 (0.031) / 0.413** |
| 1 | 0.048 (0.043) / 0.340 | **0.030 (0.033) / 0.386** | **0.029 (0.031) / 0.414** |
| 2 | 0.056 (0.053) / 0.300 | 0.045 (0.046) / 0.341 | **0.029 (0.031) / 0.422** |
| 4 | 0.050 (0.047) / 0.220 | 0.047 (0.046) / 0.290 | **0.031 (0.032) / 0.423** |
| 8 | 0.047 (0.042) / 0.183 | 0.052 (0.054) / 0.230 | **0.031 (0.035) / 0.404** |
| 16 | 0.059 (0.048) / 0.139 | 0.061 (0.061) / 0.195 `[O]` | 0.040 (0.041) / 0.376 |

`lgd_mc`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.027 (0.028) / 0.398** | **0.027 (0.029) / 0.421** | **0.028 (0.030) / 0.420** |
| 0.05 | 0.026 (0.023) / 0.376 | **0.023 (0.026) / 0.419** | **0.028 (0.030) / 0.419** |
| 0.25 | **0.033 (0.036) / 0.383** | **0.018 (0.022) / 0.408** | **0.028 (0.031) / 0.414** |
| 0.5 | 0.046 (0.044) / 0.368 | **0.024 (0.025) / 0.402** | **0.028 (0.031) / 0.413** |
| 1 | 0.047 (0.049) / 0.360 | **0.021 (0.021) / 0.394** | **0.029 (0.031) / 0.419** |
| 2 | 0.048 (0.047) / 0.331 | 0.026 (0.026) / 0.376 | **0.029 (0.031) / 0.412** |
| 4 | 0.062 (0.061) / 0.317 `[O]` | 0.024 (0.026) / 0.375 | **0.031 (0.031) / 0.422** |
| 8 | 0.055 (0.055) / 0.289 | 0.038 (0.037) / 0.367 | **0.031 (0.031) / 0.411** |
| 16 | 0.061 (0.059) / 0.293 | 0.044 (0.045) / 0.327 | **0.035 (0.039) / 0.395** `[F]` |

`tfg`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.034 (0.038) / 0.421** `[F]` | **0.031 (0.032) / 0.418** | **0.029 (0.031) / 0.414** |
| 0.05 | 0.037 (0.038) / 0.340 | **0.026 (0.025) / 0.409** | **0.029 (0.031) / 0.413** |
| 0.25 | 0.064 (0.067) / 0.216 | 0.045 (0.047) / 0.282 | **0.034 (0.035) / 0.401** |
| 0.5 | 0.094 (0.075) / 0.173 | 0.056 (0.052) / 0.226 | 0.039 (0.040) / 0.344 |
| 1 | 0.125 (0.074) / 0.145 | 0.092 (0.074) / 0.170 | 0.056 (0.043) / 0.247 |
| 2 | 0.199 (0.105) / 0.110 | 0.145 (0.097) / 0.122 | 0.112 (0.053) / 0.157 |
| 4 | 0.275 (0.126) / 0.096 | 0.253 (0.128) / 0.102 | 0.260 (0.076) / 0.105 |
| 8 | 0.298 (0.162) / 0.072 `[O]` | 0.257 (0.128) / 0.095 | 0.246 (0.092) / 0.076 |
| 16 | 0.248 (0.189) / 0.061 | 0.227 (0.155) / 0.067 | 0.202 (0.097) / 0.069 |

`btvg`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | 0.030 (0.031) / 0.317 | **0.029 (0.031) / 0.419** | **0.028 (0.030) / 0.420** |
| 0.05 | 0.032 (0.031) / 0.299 | **0.028 (0.030) / 0.424** | **0.028 (0.030) / 0.419** |
| 0.25 | 0.037 (0.035) / 0.273 | **0.025 (0.028) / 0.405** | **0.028 (0.031) / 0.412** |
| 0.5 | 0.031 (0.027) / 0.253 | **0.028 (0.025) / 0.382** | **0.029 (0.031) / 0.411** |
| 1 | 0.044 (0.037) / 0.227 | 0.028 (0.030) / 0.366 | **0.029 (0.032) / 0.417** |
| 2 | 0.061 (0.047) / 0.168 | 0.037 (0.039) / 0.355 | **0.029 (0.032) / 0.421** |
| 4 | 0.052 (0.047) / 0.179 | 0.049 (0.051) / 0.335 | **0.029 (0.031) / 0.419** |
| 8 | 0.061 (0.063) / 0.167 | 0.064 (0.058) / 0.312 | **0.031 (0.035) / 0.402** `[F]` |
| 16 | 0.077 (0.073) / 0.138 `[O]` | 0.056 (0.063) / 0.276 | 0.038 (0.037) / 0.374 |

`btvg_var`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.026 (0.025) / 0.399** | **0.028 (0.030) / 0.418** | **0.028 (0.030) / 0.420** |
| 0.05 | **0.021 (0.022) / 0.398** | **0.027 (0.028) / 0.423** | **0.028 (0.030) / 0.420** |
| 0.25 | **0.020 (0.017) / 0.395** | **0.022 (0.024) / 0.410** | **0.028 (0.030) / 0.419** |
| 0.5 | 0.024 (0.023) / 0.371 | **0.016 (0.020) / 0.401** | **0.028 (0.030) / 0.419** |
| 1 | 0.018 (0.020) / 0.368 | **0.019 (0.019) / 0.405** | **0.027 (0.030) / 0.419** |
| 2 | 0.022 (0.019) / 0.347 | **0.013 (0.014) / 0.390** | **0.027 (0.029) / 0.416** |
| 4 | 0.014 (0.015) / 0.351 | **0.012 (0.014) / 0.401** | **0.029 (0.030) / 0.416** `[F]` `[O]` |
| 8 | 0.019 (0.022) / 0.311 | 0.013 (0.013) / 0.354 | **0.029 (0.030) / 0.414** |
| 16 | 0.015 (0.016) / 0.333 | 0.006 (0.008) / 0.323 | **0.029 (0.029) / 0.412** |

**gap** (floor 0.3780; unguided 0.058 (0.055) / 0.420)

`plug`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.060 (0.059) / 0.413** | **0.058 (0.055) / 0.420** | **0.058 (0.055) / 0.420** |
| 0.05 | 0.049 (0.051) / 0.376 | **0.054 (0.058) / 0.420** | **0.061 (0.057) / 0.419** |
| 0.25 | 0.054 (0.051) / 0.353 | **0.066 (0.067) / 0.382** | **0.063 (0.061) / 0.411** |
| 0.5 | 0.049 (0.047) / 0.297 | 0.064 (0.066) / 0.345 | **0.064 (0.065) / 0.405** |
| 1 | 0.055 (0.049) / 0.233 | 0.085 (0.086) / 0.300 | **0.064 (0.064) / 0.399** |
| 2 | 0.077 (0.077) / 0.200 | 0.090 (0.088) / 0.277 | **0.068 (0.064) / 0.380** `[F]` |
| 4 | 0.092 (0.088) / 0.192 | 0.103 (0.094) / 0.240 | 0.071 (0.068) / 0.373 |
| 8 | 0.123 (0.107) / 0.185 | 0.125 (0.105) / 0.227 | 0.076 (0.072) / 0.336 |
| 16 | 0.161 (0.137) / 0.166 `[O]` | 0.152 (0.135) / 0.220 | 0.091 (0.075) / 0.305 |

`tmpd`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.059 (0.057) / 0.423** | **0.058 (0.055) / 0.424** | **0.058 (0.055) / 0.421** |
| 0.05 | **0.063 (0.061) / 0.415** | **0.058 (0.054) / 0.419** | **0.059 (0.055) / 0.419** |
| 0.25 | **0.061 (0.057) / 0.419** | **0.060 (0.059) / 0.410** | **0.058 (0.056) / 0.412** |
| 0.5 | **0.055 (0.052) / 0.388** | **0.061 (0.057) / 0.409** | **0.060 (0.058) / 0.413** |
| 1 | 0.062 (0.062) / 0.347 | 0.062 (0.063) / 0.372 | **0.063 (0.061) / 0.404** |
| 2 | 0.059 (0.059) / 0.279 | 0.068 (0.068) / 0.320 | **0.065 (0.064) / 0.388** `[F]` |
| 4 | 0.060 (0.051) / 0.217 | 0.087 (0.083) / 0.263 | 0.073 (0.067) / 0.371 |
| 8 | 0.078 (0.071) / 0.181 | 0.111 (0.091) / 0.221 | 0.071 (0.069) / 0.341 |
| 16 | 0.120 (0.110) / 0.142 | 0.138 (0.125) / 0.208 `[O]` | 0.082 (0.072) / 0.313 |

`lgd_mc`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.059 (0.056) / 0.420** | **0.057 (0.055) / 0.420** | **0.059 (0.056) / 0.420** |
| 0.05 | **0.057 (0.055) / 0.424** | **0.061 (0.059) / 0.419** | **0.059 (0.056) / 0.421** |
| 0.25 | **0.066 (0.062) / 0.423** | **0.063 (0.062) / 0.419** | **0.061 (0.055) / 0.422** |
| 0.5 | **0.069 (0.066) / 0.439** | **0.066 (0.065) / 0.404** | **0.061 (0.060) / 0.423** |
| 1 | **0.082 (0.075) / 0.391** | **0.074 (0.069) / 0.398** | **0.061 (0.065) / 0.420** |
| 2 | **0.094 (0.091) / 0.383** | **0.094 (0.087) / 0.391** `[F]` | **0.063 (0.063) / 0.406** |
| 4 | 0.119 (0.115) / 0.341 | 0.121 (0.112) / 0.342 | **0.069 (0.067) / 0.389** |
| 8 | 0.158 (0.149) / 0.330 | 0.176 (0.160) / 0.328 | **0.072 (0.068) / 0.378** |
| 16 | 0.211 (0.188) / 0.313 `[O]` | 0.184 (0.177) / 0.293 | 0.081 (0.076) / 0.364 |

`tfg`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.064 (0.062) / 0.395** | **0.067 (0.065) / 0.415** | **0.063 (0.059) / 0.410** |
| 0.05 | 0.087 (0.078) / 0.280 | 0.073 (0.071) / 0.298 | **0.071 (0.068) / 0.378** `[F]` |
| 0.25 | 0.165 (0.135) / 0.172 | 0.173 (0.154) / 0.212 | 0.109 (0.085) / 0.286 |
| 0.5 | 0.211 (0.167) / 0.126 | 0.220 (0.178) / 0.176 | 0.129 (0.100) / 0.212 |
| 1 | 0.259 (0.197) / 0.138 | 0.291 (0.224) / 0.151 | 0.155 (0.125) / 0.149 |
| 2 | 0.297 (0.223) / 0.112 | 0.316 (0.259) / 0.126 | 0.193 (0.149) / 0.117 |
| 4 | 0.308 (0.247) / 0.114 | 0.303 (0.269) / 0.119 | 0.185 (0.157) / 0.088 |
| 8 | 0.272 (0.242) / 0.084 | 0.337 (0.278) / 0.120 `[O]` | 0.185 (0.160) / 0.083 |
| 16 | 0.270 (0.246) / 0.087 | 0.259 (0.247) / 0.105 | 0.181 (0.155) / 0.072 |

`btvg`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.058 (0.055) / 0.393** | **0.057 (0.057) / 0.413** | **0.060 (0.056) / 0.420** |
| 0.05 | 0.059 (0.061) / 0.372 | **0.058 (0.057) / 0.395** | **0.060 (0.056) / 0.418** |
| 0.25 | 0.059 (0.056) / 0.312 | 0.058 (0.055) / 0.357 | **0.060 (0.059) / 0.412** |
| 0.5 | 0.053 (0.054) / 0.303 | 0.070 (0.070) / 0.301 | **0.060 (0.060) / 0.408** |
| 1 | 0.092 (0.087) / 0.268 | 0.071 (0.066) / 0.281 | **0.066 (0.061) / 0.403** |
| 2 | 0.074 (0.073) / 0.286 | 0.083 (0.078) / 0.280 | **0.073 (0.069) / 0.385** `[F]` |
| 4 | 0.082 (0.078) / 0.241 | 0.081 (0.077) / 0.255 | **0.071 (0.067) / 0.379** |
| 8 | 0.068 (0.066) / 0.207 | 0.088 (0.077) / 0.260 | 0.074 (0.066) / 0.356 |
| 16 | 0.080 (0.080) / 0.192 | 0.097 (0.095) / 0.221 `[O]` | 0.082 (0.075) / 0.335 |

`btvg_var`

| w \ t_min | 0.05 | 0.5 | 0.75 |
|---|---|---|---|
| 0.01 | **0.064 (0.058) / 0.387** `[F]` | **0.056 (0.056) / 0.412** | **0.060 (0.055) / 0.420** |
| 0.05 | **0.060 (0.061) / 0.390** | **0.058 (0.058) / 0.408** | **0.060 (0.056) / 0.416** |
| 0.25 | 0.085 (0.086) / 0.352 `[O]` | 0.046 (0.044) / 0.346 | **0.059 (0.058) / 0.414** |
| 0.5 | 0.066 (0.060) / 0.333 | 0.047 (0.051) / 0.315 | **0.060 (0.058) / 0.413** |
| 1 | 0.074 (0.068) / 0.273 | 0.056 (0.056) / 0.288 | **0.063 (0.061) / 0.405** |
| 2 | 0.077 (0.073) / 0.278 | 0.050 (0.051) / 0.281 | **0.062 (0.060) / 0.407** |
| 4 | 0.069 (0.063) / 0.260 | 0.074 (0.068) / 0.269 | **0.058 (0.057) / 0.403** |
| 8 | 0.070 (0.065) / 0.275 | 0.048 (0.045) / 0.250 | **0.061 (0.059) / 0.404** |
| 16 | 0.062 (0.059) / 0.263 | 0.049 (0.051) / 0.255 | **0.060 (0.059) / 0.404** |

<!-- END appendix generated by proj1/scripts/tune_table.py -->
