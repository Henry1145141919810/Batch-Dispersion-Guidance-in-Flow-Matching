# Status and run log

**Updated 18 Sep 2026, 16:05.** Live task state. Plan of record: **S + D + SMG**
(see [INNOVATION_IDEAS_INDEX.md](INNOVATION_IDEAS_INDEX.md), [FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md](FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md)).
Deadline **29 Sep 08:30**, defence 30 Sep.

## How Betty is reached

**VS Code Remote-SSH, pinned to `login01`.** Connected 18 Sep ~14:20 after Duo; the
vscode-server lives in `~/.vscode-server`. The Claude Code extension is to be installed *in*
that remote window ("Install in SSH: betty") so its shell runs on Betty directly.

The laptop-side unattended route (Kerberos ticket → Git Bash `ssh`) was abandoned. `kinit`
itself works, but Git Bash links Heimdal, which refuses a ccache on a `noacl` mount and then
falls into an NTLM/RC4 path that OpenSSL 3.5 blocks. SSH multiplexing is impossible on Windows
(MSYS sockets lack `SCM_RIGHTS`). Neither is worth more time; Remote-SSH sidesteps both.

Per instruction 18 Sep, **no compute runs on the local RTX 5080**. All training is on Betty.

**First job submitted 18 Sep ~14:35**: `cgm-fm`, 150 epochs, hidden 256 / 8 layers / batch
256. Two copies were queued by accident and both hit `QOSMaxWallDurationPerJobLimit` because
the script asked for 12 h; one was cancelled, the other resubmitted at 4 h.

## Task state

| Task | State | Notes |
|---|---|---|
| S0 assets — QM9 tensors | **done** | 133,885 molecules, zero skipped; 4-way split |
| S0 assets — predictors f_A / f_B | **done, gate passed** | val MAE 0.0897 / 0.0840 D vs 1.53 D chance |
| S1 symmetry checks | **done** | 12/12 pass, float64, incl. new finite-gradient check |
| FM trainer written | **done** | [train_fm.py](proj1/scripts/train_fm.py), smoke-tested |
| FM trainer — resume + time guard | **done, tested** | `--resume` chains 4 h jobs; `--max-minutes 225` pauses cleanly |
| FM trainer — full run (SUPERSEDED) | **discarded** | job 8495717, 150 epochs, val 1.8596. Trained on `train_ab`, violating the pre-registered split. Archived under `proj1/checkpoints/v2_*`; see [SPLIT_PROTOCOL.md](SPLIT_PROTOCOL.md) |
| **FM base model — FINAL** | **done, 20 Sep** | jobs 8524621-24 chained, 1500/1500 epochs on `train_a` (51,527), 27.6 s/epoch, ~11 h 25 m. **Selected epoch 1300: atom_stab 0.9390, mol_stab 0.3926, val 1.79762.** All links `COMPLETED 0:0`, clean time-guard stops, resume boundaries exact (492 -> 986 -> 1480 -> 1500) |
| **FM base model — benchmark** | **done, 20 Sep** | [BASE_MODEL_BENCHMARK.md](BASE_MODEL_BENCHMARK.md). EDM protocol, 3 seeds x 10k, corrected evaluator: **atom 0.935 +- 0.002, mol 0.39 +- 0.01, validity 0.75** at NFE 100. Like-for-like bar (EEGSDE Table 5, unconditional half-data EDM): 98.37 / 81.74. Gap is not the data split; leading suspect is unscaled one-hot atom types (EDM Table 10). NFE saturates by 500 (+0.5 atom). **Epoch 1500 beats selected 1300 on all 3 seeds x 10k (+0.18 atom, +1.0 mol); ship `fm_last.pt` ema (0.937 / 0.40 / 0.76) as a disclosed amendment to the pre-registered rule.** Audited twice by an independent agent. |
| Evaluator fix | **done, 20 Sep** | `evaluation.py` BONDS3 lacked EDM's C#O 113 pm triple entry; inflated generated-sample stability 0.2-1.1 pt, invisible on real data. Now entry-for-entry identical to EDM (brute-force verified). Packages rebuilt: fm `44e4ac00a001`, diff `149d1377a076`. Runs started with the old table need no restart (selection effect < noise). |
| Diffusion — full run | **not started** | job 8501085 cancelled (wrong split). Betty is now free; old `diff*.pt` are `train_ab`-era and must be archived first or the selection guard refuses to start. Parsa also has the diffusion package |
| Cluster scripts | **run on Betty** | setup completed; the `--resume` slurm version still needs pushing (`transfer/`) |
| Diffusion trainer | **done, smoke-tested** | VP schedule in `src/diffusion.py`; eps-prediction; same resume/guard as FM |
| Samplers (Euler + Heun) | **done, tested** | `src/sampling.py`, both families, full-field Heun, call counting |
| Guidance (plug-in + SMG) | **done, tested** | `src/guidance.py`: 4 modes, Hutchinson tr(HSigma), exact JVP for v_f |
| S2 evaluation harness (YIELD) | **written, calibrated** | `src/evaluation.py` + `scripts/s2_harness.py`; bond heuristic reproduces published QM9 data stability (99.5% atom / 96.0% mol vs Hoogeboom 99.0 / 95.2). Defines delta = 2 x f_B MAE = **0.168 D**. Needs one Betty run on `fm.pt` |
| M-0 reparameterisation control | **code ready** | needs an FM checkpoint to run against |
| M-1 signed-bias gate | **script ready** | reads delta from the S2 result; refuses to run without it. **First red decision point** |
| Generic job wrapper | **done** | `cluster/run.slurm`, `sbatch --export=ALL,CMD="..."` for tests / M-0 / M-1 |
| M-2 arms, M-3 reference, M-4 Heun control | blocked | |
| D-1 tolerance band | blocked | |

## Results so far

**Property predictors** (EGNN, single linear head, disjoint halves):

| | split | val MAE (Debye) | vs chance |
|---|---|---|---|
| f_A (guide) | train_a | 0.0897 | 17× |
| f_B (evaluator) | train_b | 0.0840 | 18× |

Published EGNN on µ is ~0.03 D. These are adequate for their purpose: the f_A−f_B gap is
the reward-hacking diagnostic, and independent training halves keep the blind spots
uncorrelated.

**FM trainer smoke test** (2,000 molecules, hidden 64, 3 layers): loss falls 3.91 → 3.62
over two epochs; coordinate and feature terms both decrease. Trivial-predictor loss is
≈3.25 (coord), so the model is learning, not merely centred.

**Resume chain test** (CPU, tiny model): guard pauses at epoch 1 → `--resume` continues at
epoch 2 → completes 6/6; a further `--resume` with nothing left exits cleanly.

**VP schedule** verified exact: alpha^2 + sigma^2 = 1 to 6 decimals across tau, alpha(0) = 1,
alpha(1) = 0.0066. Diffusion smoke test falls below the trivial eps-prediction loss of 1.0
(coord 0.79, feat 0.94), so it is learning rather than predicting zero.

**Guidance and sampler tests** (`proj1/tests/test_guidance.py`, CPU, float64) -- 9/9 pass:

| check | result |
|---|---|
| Euler = N field evals, Heun = 2N, both families | exact |
| affine f gives c = 0 | 0.0 exactly |
| affine f: SMG = plug-in scaled by s^2/(s^2+v_f) | 4.4e-16 |
| Hutchinson tr(H Sigma) unbiased vs exact Jacobian/Hessian | z = 1.65 over 1500 probes |
| v_f via one JVP vs exact | 0.0 exactly |
| Euler convergence order on the guided field | 1.03, 1.01 |
| Heun convergence order on the same field | 2.00, 2.00 |

The order check runs on an interior span [0.2, 0.8] with the endpoint policy off. Over the
full [0, 1] the guidance switch-on at `t_min_guide` is a discontinuity in t and the error does
not fall monotonically -- worth knowing before M-4 reads too much into a full-interval curve.

**M-1 design note.** E[f(x_1) | x_t] is obtained without posterior sampling by noising real
validation molecules: each x_t comes with its true x_1, and by the tower property the
population-average gap E[f(m) - f(x_1)] equals E_{x_t}[f(m) - E[f(x_1)|x_t]]. The script
reports that mean gap against -c, the regression slope of per-pair gap on -c (~1 if the
correction is right), sign agreement, and |gap|/delta, then prints a mechanical verdict from
the M-1 decision table. Smoke-tested end to end on CPU with a random generator.

**S2 harness on real QM9 (300 val molecules):** atom stability 0.9946, molecule stability
0.9600, RDKit validity 0.98, uniqueness 1.00. This is the ceiling every generator number is
read against, and it matches the published dataset figures, so the heuristic is faithful.
delta = 0.168 D. The guidance temperature `s` defaults to the property std (1.54 D), not delta:
1/s^2 multiplies the field and delta would give a x35 prefactor that overflows.

## Findings worth keeping

- **`sqrt(0)` NaN in the coordinate update.** `d2.sqrt()` in `EGNNLayer.forward` has
  infinite gradient where `d2 == 0` — the masked diagonal, and every pair of padded atoms,
  which all sit at the origin. The forward value is masked but `inf * 0 = NaN`, so *every*
  backward pass through a coordinate-updating layer produced NaN. Fixed with
  `torch.sqrt(d2 + 1e-8)`. The symmetry suite passed 11/11 throughout because its only
  backward pass ran through `EGNNScalar`, which has `update_coords=False`. A regression check
  now exists and was verified to fail (216 non-finite entries) with the bug reintroduced.
- **Checkpoint clobbering.** `best` reset to `inf` on relaunch, so a restart overwrote a
  better checkpoint on its first eval. Fixed in both trainers: `best` seeds from the checkpoint
  on disk; `--fresh` opts out.
- **Scheduler restore order.** `CosineAnnealingLR` steps recursively from the optimiser's
  current LR, so on resume the optimiser state must be loaded *before* the scheduler state.
- **Affine properties need `allow_unused=True` in the HVP.** For affine f the gradient is a
  constant, the double-backward graph never reaches the inputs, and `torch.autograd.grad`
  raises rather than returning zeros. H = 0 is the correct answer there.
- **`CUDA_VISIBLE_DEVICES=""` does not hide the GPU from torch on Windows** (`-1` does). Both
  trainers now take `--device cpu` so a local smoke test cannot occupy a reserved GPU.
- **SMG's novelty is narrower than first ranked.** TMPD and MMPS already contain SMG's
  scalar linear-property formula; the covariance/Jacobian identity and plug-in guidance are
  in *On the Guidance of Flow Matching*; STSL already puts second-order Tweedie information
  into guidance; Heun is long-established. What survives is the **nonlinear-property mean
  correction** ½ tr(HΣ), which vanishes for affine f — the claim rests entirely on properties
  with curvature. See [SMG_PRIOR_WORK_AUDIT.md](SMG_PRIOR_WORK_AUDIT.md).
- **The mode-collapse figure (0.357 → 0.075) is provisional.** It must be re-run against the
  same soft likelihood the sampler targets before it supports any performance claim.

## Naming on Betty

No course identifiers anywhere on the cluster. Working tree
`/vast/projects/ajw/wharton/hyhuang/cgm`, job names `cgm-*`.

## Cluster constraints

- Partition `b200-mig45`, resources fixed at exactly `--gpus=1 --cpus-per-task=6 --mem=48G`;
  the submit filter has rejected other ratios since 2026-08-03. `--mem` must also be an exact
  multiple of the partition's 8 GB per CPU, which 6 × 8 = 48 satisfies.
- **QOS wall cap is 4 hours** (`wharton-dgx-b200`, `wharton-phd-student-dgx-b200`), although
  the partition itself allows 2 days. Anything longer sits `PD` forever with
  `QOSMaxWallDurationPerJobLimit`. Every long run is a chain of `--resume` jobs.
- SLURM binaries live in `/cm/local/apps/slurm/current/bin` and drop off `PATH` mid-session;
  `srun: command not found` means re-add that directory, not that SLURM is gone.
- `/tmp` is node-local. Everything a job reads or writes goes on `/vast`.
- Never run compute on a login node. Check `$SLURM_JOB_ID` is set before anything heavy.
- Quotas as of 18 Sep: home 8.11 / 50 GB, allocation 113.28 GB / 1 TB.
