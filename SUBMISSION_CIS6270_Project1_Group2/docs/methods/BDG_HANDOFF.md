# BDG — Batch-Dispersion Guidance: complete handoff

> **⚠️ Reviewed 25 Sep 2026 — read [BDG_REVIEW.md](BDG_REVIEW.md) before relying on
> anything below.** This document is kept unchanged as the author's record. The
> review verified the mechanics (the gradient, the §3 reduction, η = 0 ≡ plug, same
> cost as plug, a monotone knob reproduced by an independent port), but found that
> several claims below do not hold as written:
> - **What BDG is.** Plug with the centring gain fixed and only the *deviation* gain `w_eff = 1+ηe` servoed. It is not plug at a negative weight: its widening cells still pull the batch mean toward the target.
> - **The equilibrium is not the setpoint.** The law settles where w_eff = 0, i.e. `V* = τ²(1−1/η)`, an asymptote the 50-step window never reaches. At η ≤ 1 the controller's own field is never repulsive. "V_b does not vanish, so it is stable" is not the reason; boundedness (`w_eff ≥ 1−η`) is.
> - **"Does not reduce to any fixed schedule" is refuted as stated.** A schedule replayed on another seed's noise recovers 69.5–96 % of the effect; feedback adds a small real remainder.
> - **The target is never stated.** The plug control rules out q90, so it is q50 or `dist`. At the project's q90 headline the §1 premise and "widening lowers coverage by definition" are false.
> - **"No existing arm widens reproducibly / 1.010×" is false.** `rch` widens 1.42×/1.455× on both seeds.
> - **The floor 0.362109375 is borrowed** from `results/sweep`, and the gap e4t0.5 "FAILS" is knife-edge.
> - **§8 mis-describes MGD and omits the closed-loop guidance literature.**
>
> The required changes are listed in BDG_REVIEW.md §5. This file's code lives in another member's Betty tree; an independent port is on branch `worktree-wf_bc7c0f18-844-2`, with its cells in `results/bdg_port/`.

**Written for:** Henry, and an AI agent working on his behalf. Assumes familiarity with the repo's
guidance arms but *no* knowledge of this work. Nothing here needs to be re-derived.

**Status as of 25 Sep 2026.** Implemented, tested, run on two seeds, audited. The method works at
what it claims and fails on a different axis, both measured. **Read §3 before claiming anything.**

---

## 1. One paragraph

Band coverage (`in_band`) is governed by how tightly the batch clusters, not where it is centred —
`delta/sigma = 0.056–0.169`, so the acceptance window is 6–17% of one standard deviation. But every
guidance arm optimises a *centring* objective and exposes one knob (strength) that moves centring and
spread together. BDG adds a feedback loop: measure the batch's current spread of the predicted
property, compare it to a requested setpoint, and modulate the guidance coefficient by the error.
Spread becomes a first-class control. **It works** — monotone requested-to-achieved on 6/6 curves
across two seeds, spanning `sd/unguided` 0.65 to 1.17, including *widening*, which no existing arm
here can do reproducibly. **It does not improve `in_band`**, and we can prove neither direction of the
knob can: tightening enough to matter breaks the chemistry floor, widening lowers coverage by
definition.

---

## 2. The method

Per guided sampling step, for a batch of `B` trajectories.

### Baseline (`plug` / DPS), for contrast

```
m_i  = x_t[i] + (1-t) * v_theta(x_t[i], t)     endpoint estimate
F_i  = f_A(m_i)                                predicted property, one scalar per molecule
g_i  = grad f_A(m_i)
push_i = [ (y - F_i) / s^2 ] * g_i             then J^T pullback
```

### BDG adds two batch statistics and an error signal

```
F_bar = (1/B) sum_i F_i                            where the batch is centred
V_b   = (1/(B-1)) sum_i (F_i - F_bar)^2            HOW TIGHTLY IT IS GROUPED
e     = (V_b - tau^2) / tau^2                      relative error vs the SETPOINT

push_i = [ (y - F_i) - eta * e * (F_i - F_bar) ] / s^2 * g_i
           \________/   \_____________________/
            centring          dispersion
```

`tau` is **the knob** (requested spread); `eta` is the gain. **Note:** `tau_mult = 1.0` is *not* a
no-op — the measured fixed points are ~0.57–0.87 depending on property and gain, so cell names like
`e4t1` should not be read as "hold". The bit-identity control is `e0t1` (`eta = 0`). In the sweep `tau = tau_mult * s` with
`s = f_A.y_std`, so `tau_mult` reads as "fraction of the natural spread".

### Why that dispersion term is the gradient of the grouping

```
dV_b/dF_i = 2/(B-1) * (F_i - F_bar)        the F_bar dependence cancels, since sum_j (F_j - F_bar) = 0
dV_b/dm_i = 2/(B-1) * (F_i - F_bar) * g_i  chain rule
```

Push each molecule along **its own** property gradient, proportional to its signed deviation from the
batch mean. The `2/(B-1)` is absorbed into `eta` so the gain does not depend on batch size.

### Signs — the whole mechanism

```
V_b > tau^2  ->  e > 0  ->  high-F molecules pushed DOWN, low-F UP  ->  TIGHTEN
V_b = tau^2  ->  e = 0  ->  term vanishes, reduces to plug exactly
V_b < tau^2  ->  e < 0  ->  signs reverse                           ->  WIDEN
```

**No clamp.** The existing `btvg` arm clamps its coefficient at `<= 0` (guidance.py:1471) so it can
only ever tighten. That clamp is forced: `btvg` servos the *per-sample* posterior variance
`V_F = g' Sigma g` with `Sigma = k*J`, `k = (1-t)^2/t -> 0`, so `V_F < tau^2` becomes true on **every**
trajectory late in `t` and the widening branch fires unconditionally (measured runaway coefficients
+5.3, +110.8, +78.6). `V_b` does not vanish, so the fixed point is stable and the clamp comes off.
**That is what buys the widening direction.**

**Cost: zero extra NFE.** Both terms are a scalar times `g_i`, so they share the one backward pass
`plug` already performs. Cost dict is byte-identical to `plug`.

---

## 3. THE REDUCTION — read this before claiming novelty

`num_i` is **affine in `F_i`**, so it factors exactly:

```
(y - F_i) - eta*e*(F_i - F_bar)  =  (1 + eta*e) * ( y_eff - F_i )
                                      \_________/
                                         w_eff        y_eff = (y + eta*e*F_bar)/(1 + eta*e)
```

Verified numerically to **1e-14**. Both `w_eff` and `y_eff` are single numbers shared by the batch.

**So BDG adds no new direction.** At any instant it is `plug` at a rescaled weight aiming at a shifted
target — a *batch-feedback-adaptive guidance weight*, not a variance controller. **Do not call it
variance control in the paper.** State this reduction yourself in §3.5; a reviewer will find it.

This generalises: **any guidance whose per-sample coefficient is affine in `F_i` is `plug` reweighted.**
That covers BDG, the "two-gain" idea, `spbc`/MGD's mean-only corrector, and `btvg`'s mean term. To get
a genuinely new *direction* you need a nonlinear function of `F_i`, or a direction other than `g_i`.

**What does NOT reduce: the feedback.** Freezing `w_eff` at its time-average and running plain `plug`
at that constant weight **fails to reproduce BDG** — mu widened 2.53x instead of 1.17x and
`mol_stability` collapsed from 0.377 to 0.252. The loop self-limits: as `V_b` grows, `e` falls, and the
weight pulls itself back. A fixed schedule cannot do that. Cells: `results/bdg/*__plug__*weff*.json`.

---

## 4. Code inventory

Pre-change copies are saved as `*.bak_bdg`; diff against those to review.

| file | change | lines |
|---|---|---|
| `proj1/src/guidance.py` | `"bdg"` in `KNOWN_MODES`; kwargs `bdg_eta`/`bdg_tau`/`bdg_onesided` on `guidance_field`; the dispersion block in the shared plug/SMG tail; 6 diagnostics | +65 / −2 |
| `proj1/src/sampling.py` | 3 kwargs on `_Base.__init__`, stored, forwarded in `_guide`; **6 keys added to `DIAG_KEYS`** | +16 / −3 |
| `proj1/scripts/guidance_sweep.py` | `arm_kwargs` bdg branch (variant `e<eta>t<tau_mult>[o]`); `bdg_tau` resolved in `run_cell` once `s` is known; `plan_bdg_cells`; `bdg` stage; `--only-variant` / `--only-arm`; constants `BDG_TAU_MULT`, `BDG_ETA`, `BDG_W`, `BDG_FIXED_POINT`, `BDG_WEFF` | +120 / −1 |
| `proj1/tests/test_bdg.py` | 28 exact gates | new, 161 |
| `proj1/scripts/bdg_table.py` | results → §4.4 table, with two automatic gates | new, 113 |
| `proj1/cluster/bdg_cpu.slurm` | main 45-cell array | new, 81 |
| `proj1/cluster/bdg_hold.slurm` | per-property fixed-point cells | new, 43 |
| `proj1/cluster/bdg_vf.slurm` | `btvg_var` (V_F contrast), 8 cores / batch 128 after OOM | new, 45 |
| `proj1/cluster/bdg_weff.slurm` | the reduction control (plug at negative w) | new, 34 |
| `proj1/cluster/bdg_seed2.slurm` | seed-2 replication, 19 cells | new, 47 |

`proj1/cluster/bdg.slurm` is the **abandoned GPU version** — kept for reference only; the GPU queue was
not the bottleneck but CPU was faster. Do not run it.

**`DIAG_KEYS` is load-bearing.** `_accumulate_diag` (sampling.py:172) only accumulates keys listed
there. Without the six additions every cell lands with the controller state silently absent — the same
gap that made `osc`'s mechanism unfalsifiable. This bug was present and fixed before any cell was
written.

**Clean up before shipping:** `*.bak_bdg` (3 files) and `proj1/scripts/guidance_sweep.py.bak_0154`
(from an earlier session, unrelated). Delete or gitignore; do not leave a teammate guessing which is
authoritative.

---

## 5. Reproducing

```bash
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf
cd $PROJECT_ROOT
PY=$PROJECT_ROOT/redacted

# 1. gates (fast, no cluster). Expect "all BDG gates pass".
$PY proj1/tests/test_bdg.py
# regression: BDG must not have broken any existing arm
$PY proj1/tests/test_v2_arms.py; $PY proj1/tests/test_btvg2.py; $PY proj1/tests/test_chem_guard.py

# 2. plan check, no compute
$PY proj1/scripts/guidance_sweep.py --stage bdg --props mu,alpha,gap \
    --fm weights/fm_ema.pt --dry-run

# 3. the runs (CPU arrays, one cell per task -- tensors are too small for
#    torch intra-op parallelism, so parallelism must be ACROSS cells)
sbatch --array=0-44 proj1/cluster/bdg_cpu.slurm      # 45 cells, seed 20260921
sbatch --array=0-5  proj1/cluster/bdg_hold.slurm     # fixed-point cells
sbatch --array=0-5  proj1/cluster/bdg_vf.slurm       # V_F contrast
sbatch --array=0-3  proj1/cluster/bdg_weff.slurm     # reduction control
sbatch --array=0-18 proj1/cluster/bdg_seed2.slurm    # seed 20260922

# 4. the table
$PY proj1/scripts/bdg_table.py results/bdg
$PY proj1/scripts/bdg_table.py results/bdg_seed2
```

**Cluster constraints that actually bite:** `SLURM_CONF` must be exported or you get DNS SRV errors.
`genoa-std-mem` gives 5632 MB/CPU implicitly — do not pass `--mem`. 4 cores per task is right for BDG;
`btvg_var` needs **8 cores and batch 128** because its HVP OOMs at batch 512.

**Protocol facts you must not break:**

- **CHECKPOINT: corrected 25 Sep.** An earlier version of this doc said BDG numbers could never be
  compared to `results/sweep`. That was too strong. `weights/fm_ema.pt` (md5 `e19ccc06`) carries
  `source_md5 = a190ac83`, the md5 stamped in all 1080 sweep cells, and `load_fm(..., use_ema=True)`
  reads `("ema_state_dict", "ema")` from **both** layouts — so sampling consumed the *same EMA tensors*
  either way. `weights/README.md` records `max |w_slim - w_full| = 0.0` on every parameter.
  Measured on 18 matched (prop, arm, q, w, t_min) pairs: `in_band` differs by **+0.0012 ± 0.0129**
  (max 0.0234), against 0.0188 for two independent draws — i.e. *below* independent-draw noise, t=0.38.
  `mol_stability` drifts **-0.0077 ± 0.0211**, consistently signed but not significant.
  **So:** BDG cells and `results/sweep` cells MAY appear in the same comparison table, provided the
  equivalence check above is printed as a protocol table. The real differences are DEVICE (B200 GPU vs
  CPU) and BATCH (128 vs 512), not weights.
  **BUT keep floor calls within-job.** FR3a = 0.362109 and `mol_stability` drifts -0.008, which flips
  marginal cells. Pass/fail verdicts use the in-job control; comparison tables may mix.
- BDG runs `--n 512 --batch 512` (one batch) on purpose: **for BDG the batch IS the estimator.** The
  usual 128-in-512 would run four independent controllers.
- `btvg_var` runs at batch 128. This is sound *only* because `V_F` is per-sample and cannot depend on
  batch size. State this asymmetry in the paper rather than hiding it.

Cost: ~40–70 min wall per array (all tasks run concurrently), ~4 core-hours total per seed.

---

## 6. Results

Two seeds: `20260921` (`results/bdg/`, 55 cells) and `20260922` (`results/bdg_seed2/`, 19 cells).
`n=512`, 100 steps, `t_min_guide=0.5`, CPU, per-molecule sidecars written. Floor FR3a = 0.362109375.

### 6.1 The knob works — `sd/unguided`, seed1 / seed2

| property | e4t0.5 (tight) | e4t1 | e4t1.21 | e4t1.5 (wide) |
|---|---|---|---|---|
| mu | 0.739 / 0.730 | 0.951 / 0.931 | 1.016 / 1.065 | **1.168 / 1.094** |
| alpha | 0.681 / 0.654 | 0.905 / 0.884 | — | **1.034 / 1.042** |
| gap | 0.787 / 0.742 | 0.984 / 0.964 | — | **1.096 / 1.117** |

Monotone on **6/6 curves, both seeds**. Widening replicates on all three properties; mu and alpha
clear the floor on both seeds, gap fails it on seed 2 (but so does its *control* — gap chemistry is
marginal on that seed generally). For scale: a prior scan of 1,766 existing cells found the largest
reproducible floor-clearing widening anywhere was **1.010x**, not seed-reproducible.

### 6.2 The failure — `in_band` does not improve inside the floor

The bind, in two adjacent rows of the same property:

```
gap  e4t0.5   sd 0.787   in_band 0.1523  <- best score anywhere   mol_stab 0.3477  FAILS floor
gap  e4t1     sd 0.984   in_band 0.1230                           mol_stab 0.3867  passes
```

`plug w=4` shows the same thing: best raw `in_band` on all three (0.1309 / 0.0684 / 0.1602) and fails
the floor on all three (0.2930 / 0.3438 / 0.2812). Every `e4t0.5` cell fails the floor on alpha and
gap. Widening lowers `in_band` every time, as predicted.

### 6.3 `V_b` vs `V_F` — which batch statistic matters

| arm | servos | spread range over its sweep | floor |
|---|---|---|---|
| `bdg` | `V_b` (across-batch, survives) | **31–43%** | mostly PASS |
| `btvg_var` | `V_F` (per-sample, vanishes) | **0.4–8.6%** | **5 of 6 FAIL** |

This is the scientific core, and both arms were run on the *same* checkpoint, seed, device and window
so the comparison is valid.

---

## 7. What is verified

| check | evidence |
|---|---|
| field matches the spec | brute-force autograd, **2.8e-14**, non-diagonal coupled toy |
| equivariance | 28 checks (global + per-sample rotations, reflection, translation, atom and batch permutation, padding inertness), max err **8.7e-19** |
| `eta=0` ≡ `plug` | bit-identical at field level AND cell level, all 3 properties |
| denominator | `den == s^2` exactly; no SMG branch fires |
| cost | dict byte-identical to `plug` |
| one-sided ablation | `e4t1.5o` reproduces `plug` exactly — clamped, asked to widen, does **nothing** |
| reduction | `w_eff` factorisation verified to 1e-14; open-loop control does **not** reproduce it |

---

## 8. Novelty — the honest position

A final literature pass (25 Sep) found **more prior art than earlier checks did**. Read this before
writing §3.5 — three framings that look attractive are already dead.

**Dead framings, do not use any of these:**

| framing | killed by |
|---|---|
| "the guidance weight as a feedback controller" | **CFG-Ctrl, arXiv:2603.03281** — reinterprets CFG as control on the generative flow and *explicitly* calls vanilla CFG "a proportional controller (P-control) with fixed gain". Scale-as-gain is in print. |
| "negative guidance weight widens the distribution" | **Ventura, Achilli, Ambrogioni & Lucibello, arXiv:2602.00716** (31 Jan 2026) — proposes a schedule with a negative-guidance window and states the mechanism outright: *"negative guidance reduces means and expands variances"*, with theory and Stable Diffusion experiments. Our `plug` at `w<0` widening 1.15–2.53x is a **replication in a different guidance family, not a discovery.** Cite it. |
| "state-dependent / self-regulating guidance scale" | **FBG, arXiv:2506.06085** — self-regulating scale from the model's own predictions, challenging guidance-as-fixed-hyperparameter. |
| "drive an empirical batch second moment to a setpoint, two-sided" | **MGD, arXiv:2602.17211** — empirical ensemble moments, a prescribed target, a corrector whose RHS is literally (empirical moment − setpoint), sign-reversing with the mismatch, and second moments included whenever the feature map has quadratic entries. |

**Also not new:** batch-coupled guidance (Particle Guidance, Corso et al. ICLR 2024 — *repulsive*, in
sample space, no setpoint); spread targeting through a batch distribution (Variance-Tilted Diffusion,
arXiv:2606.22239 — *monotone*, can only widen, fixed linear feature); MMD Guidance (arXiv:2601.08379);
adaptive scales generally (autoguidance, guidance intervals, CFG rescaling); and **the one-sided
version of this exact controller**, specified in this repo's own
`Three_New_Guidance_Ideas_Variance_and_Switching.md` §5.2 as `b_V = -eta_V (V - tau^2)_+`, never
implemented, and abandoned in its own revision.

**What actually survives — a narrow combination claim plus a negative result.** No paper found sets
the scalar weight of a *property-gradient* guidance term from the *across-batch second moment of the
property net's own endpoint prediction*, against a *user-chosen* setpoint, signed so the weight passes
through zero. Each ingredient is separately published; the combination is not. MGD is closest and
differs in three ways worth stating: no property predictor (its feature map is fixed), its setpoint is
the interpolant's own moment trajectory rather than a user choice, and it solves a Gram-matrix system
for multipliers rather than reducing to a reweighted guidance term.

**The real content is the reduction (§3), not the method.** `w_eff = 1+eta*e` is a structural result:
it says the repo's own unimplemented RATV 2x2 solve is unnecessary, and that any affine-in-`F_i`
coefficient is `plug` reweighted. Pair that with the measured finding that both ends of the knob are
blocked and you have a defensible course-project contribution. **Sell the diagnosis, not the arm.**

**Do not claim** to be first to control variance, to have invented feedback guidance, or to have
discovered negative-weight widening.

Verdict: ~0.85 confidence of clearing a "slightly novel" bar for a course project, ~0.1 for a
workshop paper.

---

## 9. Open items, ranked by fragility

1. **`eta=1` was only run on seed 1.** The cells that beat `plug` *while clearing the floor* are all
   `eta=1` (alpha 0.0645 vs 0.0449; gap 0.1348 vs 0.1133), and they are maxima over ~13 cells per
   property — the grid-search artifact that produced two false results earlier in this project.
   **Cheapest fix: 6 cells, ~1 h.** This is the highest-value remaining run.
2. **`plug` at negative `w` beats BDG on alpha** (1.153x at `mol_stab` 0.397 vs BDG's 1.034 / 0.373).
   The simpler thing wins there — report it, do not bury it. **But it is NOT a claim:** negative-weight
   widening is published (arXiv:2602.00716). Present it as a replication and as the honest control that
   BDG must beat.
3. **`bdg_dev` and `bdg_disp` are structurally uninformative** — both are mean-zero over the batch by
   construction and `_accumulate_diag` takes a batch mean, so they record ~1e-7 in every cell. Persist
   the batch RMS instead.
4. **`bdg_e` is recorded post-clamp**, so the one-sided cells report `e = +0.000` by construction and
   cannot show how deep the widening branch would have gone. Persist the raw `e` too.
5. **`bdg_table.py:34`'s `se_prop`** assumes independent samples, which is wrong for BDG cells (all 512
   are coupled through `F_bar` and `V_b`). The printed z-scores are understated.
6. **`BDG_FIXED_POINT` is wrong for mu and alpha.** It was estimated from cells whose own widening
   inflated `V_b`. Measured: mu ≈ 0.574–0.678, alpha ≈ 0.84–0.87, gap ≈ 0.673. The fixed point is also
   `eta`-dependent, contradicting the planner comment.

---

## 10. Mapping to the paper

- **§3.5 (0.625 pt)** — problem (coverage is σ-governed), mechanism (§2), the reduction (§3, state it
  yourself), controls named.
- **§4.4 (0.675 pt, largest Results item)** — five ablations exist: `eta=0` base (bit-identical),
  one-sided vs two-sided, the τ ladder, `V_b` vs `V_F`, and the `w_eff` reduction control.
- **§4.3** — guidance works; the rubric explicitly rewards investigating a *non*-improvement.
- **§4.7** — two failure modes, both measured: the chemistry price of contraction, and the geometric
  impossibility of widening helping coverage.
- **§3.6 / §4.6 (Modality 2)** — `V_b` is the batch variance of a scalar and carries no geometry, so
  only `f_A` and the pullback change. **Verify this claim before relying on it** — it has not been
  tested on the simplex.

---

# 11. Modality 2 — DNA enhancers on the probability simplex

Added 26 Sep. **BDG transferred with zero code change**, which is the §3.6 claim made
literal. Read §11.4 before writing §4.6 — the task is *not* the published benchmark.

## 11.1 Data — and one trap

`$PROJECT_ROOT/redacted
- `KC_regions.fa` — 6,126 **Drosophila** Kenyon-cell enhancers, native 500 bp (`chr2L..chrX`)
- `MEL_regions.fa` — 3,885 **human** melanoma enhancers (`chr1..chr22`)

**Do not merge them.** Different species. Training on the mixture makes the GC
distribution bimodal, and BDG steering that spread would partly be steering the
species mixing proportion — a confound sitting on the measured quantity. Use KC
alone, all 6,126 (no train/val split needed: the evaluator is an exact count, so
no leakage is possible).

`DeepFlyBrain_data.pkl` (554 MB), `DeepMEL2_data.pkl` (440 MB) and
`classifier_ckpt/enhancer_class.ckpt` (133 B) are **git-LFS pointers, not files**.
The hparams beside them ARE real (`num_cls: 47`, `mel_enhancer: true`) and are the
Dirichlet-FM repo's MEL classifier config.

## 11.2 Code

| file | what |
|---|---|
| `proj1/m2/gate.py` | the go/no-go on the naive base. **Passed**: decode confidence 0.983, generated GC sd 0.0572 vs real 0.0668 (86%), 3-mer JS 7.6x better than uniform |
| `proj1/m2/simplex_fm.py` | data → simplex, model, training, `gc_soft` (f_A) / `gc_hard` (exact evaluator). GPU + 500 bp + validation early stopping |
| `proj1/m2/m2_sweep.py` | guidance field + sampler + metrics, one cell per invocation |
| `proj1/cluster/m2_{gate,train,sweep,train_gpu}.slurm` | the runs |
| `results/m2/` | 26 cells, 2 seeds, at 200 bp |

## 11.3 What transferred

State space went from R^3 molecular coordinates to (Delta^3)^L; the property went
from a trained EGNN to an exact GC count; **the controller did not change**.

| signature | QM9 | DNA |
|---|---|---|
| `eta=0` ≡ plug | bit-identical | **bit-identical** |
| one-sided at widening setpoint ≡ plug | exact | **exact** (`w_eff = +1.000`) |
| widening reached | 1.168 / 1.094 | **1.0055 / 1.0078** (`w_eff` −1.381 / −1.451) |
| monotone in tau | 0.68→0.91→1.03 | 0.81→1.00→1.01 |
| cost identical to plug | yes | **yes**, byte-identical |
| contraction breaks the fidelity floor | FR3a fails at `e4t0.5` | **k-mer floor fails at `e4t0.5`** |
| in_band improves | **no** | **no** |

**The conclusion transferred too:** in neither modality does spread control raise
band coverage, and in both the contraction rung is the one that breaks fidelity.

**The magnitude is 2.6x smaller** (0.20 vs 0.52 span). Not the clip — it fired on
0-3% of steps and the widening cell clipped *once* in 25,600. Three real causes:
GC is **affine** (zero curvature, gradient identical at every position, so the
simplex renormalisation partly undoes it); GC is a **200-position average**; and
the band spans only **4.3 attainable GC values** (granularity 1/200 = 0.005,
delta = 0.0107), a hard ceiling on what any method can move.

The lever for a stronger effect is a **richer property**, not more data: a trained
cell-type classifier logit is nonlinear, motif-driven and continuous-valued. Use
classifier **guidance**, never classifier **conditioning** — a conditional model
gives BDG nothing to steer and breaks the M1↔M2 parallel §3.6 is scored on.

## 11.4 THIS IS NOT THE PUBLISHED BENCHMARK — disclose it

Verified 26 Sep against Dirichlet FM (arXiv 2402.05841), Fisher Flow (NeurIPS
2024), Gumbel-Softmax FM (arXiv 2503.17361) and MOG-DFM (arXiv 2505.07086).

**The standard enhancer protocol:** class-conditional generation, **500 bp**,
DeepFlyBrain (**81 classes**, 104k seqs) / DeepMEL2 (**47 classes**, 89k), scored by
**FBD** — Wasserstein distance between Gaussians fit to a pretrained classifier's
penultimate embeddings, **10k samples each** — plus target-class probability.
Dataset: Zenodo 10184648, a single **26 GB** tarball.

**GC content is used by none of them.** Their targets are cell-type class
(enhancers), Sei-predicted activity (promoters), and DNA shape HelT/Rise (MOG-DFM).

**Our deviations:** GC-band coverage instead of class-conditional FBD; 6,126
sequences instead of 83,968; 1.3M sample-views instead of 40.3M; 200 bp in the
shipped cells instead of 500.

**Three disclosures that are free rubric credit:**
1. **Gumbel-Softmax FM does not run the enhancer benchmark at all** (promoter,
   protein and peptide tasks only). Only 2 of the 3 named baselines apply here.
2. **Fisher Flow switches the headline metric to perplexity and disputes FBD**,
   reporting the cell-type classifiers score only **11.5% / 11.2%** test accuracy.
   You can cite a NeurIPS paper saying the standard evaluator is weak.
3. **MOG-DFM's own guided demo runs at `--length 100`** while reporting base FBD at
   500 bp. Short-length guided demonstrations are precedented in this literature.

**Our base model IS their "Linear FM" baseline** (published FBD 19.6 melanoma /
15.0 fly brain at 100 NFE, vs Dirichlet FM 5.3 / 15.2), trained on a 6k subset.
So the base is a *named published baseline*, not something non-standard — but the
numbers are not comparable. **Quote theirs for context; never put them in one
table with ours.**

Also: at 200 bp their classifier does **not** crash — it global-average-pools, so
it is length-agnostic and returns numbers that are meaningless. Silent garbage,
not an error. Another reason the 500 bp rerun matters.

## 11.5 In flight / next

- **GPU job 8716992** (`b200-mig90`): 500 bp, hidden 256, 12 layers, batch 256,
  validation early stopping, max 100k steps. Records `best_it`, `sample_views`,
  `val_loss` for §A.2. When it lands, re-run the 26-cell sweep at 500 bp.
- **Memorization check, not yet run.** 6,126 sequences on a 0.83M-param model: take
  nearest-neighbour Hamming from each generated sequence to the training set. ~20
  lines, no new sampling (the `.permol.pt` sidecars exist). This is the novelty
  metric the rubric lists for sequence modalities, and it converts a question
  someone will ask at the defense into a reported number.
- **QM9 gap runs, not queued** (see §9): eta=1 on seed 2 is the highest-value one.
