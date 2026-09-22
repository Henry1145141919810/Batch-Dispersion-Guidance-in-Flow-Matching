# Transfer experiment: our guidance arms on a borrowed base model

**Written for:** the project team (Bobo, Idea, Haimo), and the defence. This is a
pre-registration, not a report — it is written before any transfer cell has run,
and the decision rules below are fixed now so that they cannot be chosen after
seeing the numbers.

**Status as of 2026-09-22.** Harness built, adversarially reviewed, revised, and
gated (**61/61**). Base-model checkpoint not yet downloaded. Zero cells run.
What the review found and what changed is §8 — read it before trusting §3.

---

## 1. Why this exists

`guidance_sweep.py` measures every arm on **our** flow-matching generator,
steered by **our** `f_A` and scored by **our** `f_B`. The `train_a` / `train_b`
split makes `f_A` and `f_B` disjoint, so an arm cannot win by exploiting the
scorer. But generator, guide and evaluator still come out of one pipeline, built
by one group, on one preprocessing of QM9, and our base model sits **42 molecule
points** below the like-for-like published bar
([../results/BASE_MODEL_BENCHMARK.md](../results/BASE_MODEL_BENCHMARK.md)).

Two questions follow that nothing in our own sweep can answer:

1. **Is a ranking of arms a property of the methods, or of our stack?** On our
   generator the ranking already inverts completely between properties — `plug`
   and `tfg_mc` are the two best arms on `mu` and the two worst on `alpha`
   ([../status/SCOPE_FM_GUIDANCE_STATUS.md](../status/SCOPE_FM_GUIDANCE_STATUS.md) §8).
   If it inverts again on a different generator, "method X is best" is not a
   claim this project can make at all, and saying so is the finding.
2. **Do our conclusions survive on a generator that does not have our known
   defect?** The leading suspect for our stability gap is unscaled one-hot atom
   features. The borrowed generator scales them by 1/8, which is exactly the
   choice we did not make. This experiment therefore doubles as a partial test
   of that hypothesis's consequences for guidance.

The professor's framing is the operative one: **the innovation is the guidance,
so the guidance is what should be portable.** A method that only works on the
generator its authors trained is not a method.

---

## 2. What is borrowed, and from where

Everything except the guidance field comes from TFG (Ye et al., *Unified
Training-Free Guidance for Diffusion Models*, NeurIPS 2024), whose QM9 release
happens to contain all three pieces for the same three properties we study.

| role | artifact | what it is | already in repo? |
|---|---|---|---|
| base model | `EDMsecond/generative_model_ema.npy` | unconditional EDM trained on one 50K half of QM9 | ❌ **download** |
| guide `f_A` | `tf_predict_{mu,alpha,gap}/model_ema_2000.npy` | time-dependent EGNN property network (192×7, attention) | ✅ vendored |
| oracle `f_B` | `evaluate_{mu,alpha,gap}/best_checkpoint.npy` | EDM's `main_qm9_prop` EGNN classifier | ✅ vendored |

`python proj1/scripts/fetch_tfg_assets.py` pulls the one missing folder and
writes `weights/tfg_manifest.json` with an md5 per file. Every transfer cell
stamps the generator's md5, so a cluster run and a local run can always be
compared.

**Why TFG and not one of the others.** The candidates were TFG, TFG-Flow,
PropMolFlow, OC-Flow and MolGuidance (all vendored under
`audit/fa_fb_search/`). TFG wins on four counts, in order of weight:

1. **It is the only one that releases a base model, a guide and an independent
   oracle for all three of our properties.** TFG-Flow releases a guide
   classifier but its base model is a discrete-state (CTMC) flow over atom
   types, which our continuous arms cannot drive without a relaxation we would
   then have to defend. PropMolFlow scores stability under a revised
   formal-charge rule on explicitly generated bond orders — a different metric
   on a re-curated QM9.
2. **Its guide and oracle are already loaded and verified in this repo.**
   `audit/fa_fb_search/check_candidates.py` checks E(3) invariance, padding
   behaviour and second-derivative finiteness on both; `disjointness_test.py`
   tests whether they trained on different molecules. That work is done and it
   is not small.
3. **`tfg_mc` — TFG's own method — is already one of our comparator arms.** So
   the transfer puts our arms on TFG's home turf against TFG's own method, run
   with TFG's own guide and scored with TFG's own oracle. That is the least
   favourable ground we can choose for ourselves, which is the point.
4. **Its base model is the like-for-like bar from the benchmark.** EEGSDE's
   half-data EDM row and TFG's `EDMsecond` are the same recipe, so the
   generator underneath this experiment is the model we are 42 points behind.

---

## 3. What was built

| file | what it does |
|---|---|
| `proj1/src/external/edm_schedule.py` | ports EDM's tabulated `polynomial_2` schedule to a continuous VP schedule, deriving `beta(tau) = sigma^2 gamma'(tau)` so the probability-flow ODE is defined on it |
| `proj1/src/external/tfg_assets.py` | loads all three checkpoints behind our interfaces; calibrates the two property networks from raw normalised output to physical units |
| `proj1/scripts/transfer_sweep.py` | the driver: resumable, one JSON per cell, `--preflight` |
| `proj1/scripts/fetch_tfg_assets.py` | downloads and hashes the missing checkpoint(s); `--models EDMsecond,EDMfull` |
| `proj1/scripts/benchmark_transfer_base.py` | **the hard gate** — runs the borrowed model through OUR evaluator (`score_samples` imported from `benchmark_base.py`, not reimplemented). Doubles as the apples-to-apples base-model row; protocol in [../results/BASE_MODEL_BENCHMARK.md](../results/BASE_MODEL_BENCHMARK.md) §9 |
| `proj1/tests/test_transfer_backend.py` | **61 gates**, all passing |
| `proj1/src/sampling.py` | `VPSampler` gained `noise_schedule`, `tau_max_guide` and `grid`. The default path is unchanged and gated as such |

**`guidance.py` was not touched.** That is the whole point of the design: the
arms run here byte-for-byte as they run in the main sweep, because
`guidance_field` only ever sees `(f_net, post_fn, coords, feats, mask, y, s)`
and never knows which generator produced the posterior.

### 3.1 The two things most likely to be wrong, and how they are gated

**Calibration.** Neither released property network ships the `(mean, MAD)` it
was normalised by — TFG computes them from a dataloader at runtime. Both are
therefore calibrated here by a two-parameter least-squares fit of raw output to
the physical property, on the same 3,000 molecules, identically for guide and
oracle. The fit never sees the answer, so the check is independent: **the fitted
slope should recover QM9's mean-absolute-deviation.** It does, to 0.5–1.6% on
all six networks, and the intercept recovers the property mean to within 3% of a
MAD. A missing or doubled one-hot division — the single most likely defect in
the whole adapter, and a silent one — moves this by a factor of 8 or 64, so the
gate is not vacuous. It is the first thing to re-check if any number here looks
strange.

**The schedule.** `beta = sigma^2 gamma'` is the one derivation in the transfer
that is ours rather than ported. It is gated against `d log alpha / d tau =
-beta / 2` by finite difference (max relative error 4.9e-8) and against the VP
identity `alpha^2 + sigma^2 = 1` (2.2e-16), and the interpolated gamma is gated
to agree with EDM's own lookup **exactly** at grid points. The first version of
this file took `gamma'` from a central difference of the table rather than the
slope of the interpolant `alpha` actually comes from; the identity gate caught
it at 1.3%, which is what the gate is for.

---

## 4. The protocol

### 4.1 Held fixed across every arm

| item | value | why |
|---|---|---|
| base model | `EDMsecond`, EMA weights, md5 stamped per cell | one generator, no per-arm choice |
| guide `f_A` | `tf_predict_<p>`, time channel **0** | our arms evaluate at the posterior mean `m`, an estimate of a *clean* molecule |
| oracle `f_B` | `evaluate_<p>` | the network TFG reports its own MAE with |
| sampler | 100-step Euler on the PF-ODE, `tau_min = 1e-3` | the same solver and budget as the main sweep |
| guidance window | `tau_max_guide = 0.5` | the VP image of the main sweep's `t_min_guide = 0.5`. **Not a free parameter** — see §5.7 |
| clip | 1.0 (velocity-relative trust region) | same as the main sweep |
| targets | q50, q90 of the QM9 property distribution | q50 interpolates, q90 asks for the tail |
| `delta` | `choose_delta(oracle MAE, k=2)` | the pre-registered rule, through the same helper the main sweep uses |
| n per cell | 512 | matches the main sweep's screening scale |
| seed | 20260922 | not equal to the main sweep's 20260921 |

### 4.2 The arms

Deliberately short. The main sweep's job is to screen twenty arms; this one's
job is to **re-test a verdict**, and each cell here costs ~3× a main-sweep cell
(EDM is 192×9 with attention and a dense edge list).

| arm | class | what it is |
|---|---|---|
| `unguided` | baseline | the reference. **Not** EDM's published unconditional row — see §5 |
| `plug` | COMPARE | DPS-style plug-in |
| `tmpd` | COMPARE | TMPD / ΠGDM uncertainty denominator |
| `tfg_mc` | COMPARE | **TFG's own method**, on TFG's own stack |
| `lgd_mc` | COMPARE | loss-guided diffusion |
| `osc` | COMPARE | observable-space closure ⚠ *see §7* |
| `smg` | **OURS** | SMG as shipped |
| `smg2` | **OURS** | completed-moment closure |
| `btvg` | **OURS** | band-targeted variance guidance — **the arm stage v2 says survived** (+8.9 / +6.7 / +7.0 σ, tied best on all three) |
| `shg_plug_btvg` | **OURS** | scheduled handoff, plug early → btvg late. Highest band coverage on `mu` of any arm (0.131) |

**`spbc` is deliberately absent.** Stage v2 (commit `4b800ed`, 342 cells × 2
seeds) drops it on **both** seeds — +0.2 / +1.3 / +0.1 σ, it does nothing — and
re-testing a replicated null on a second backend buys nothing for ~22 cells of
wall-clock. It stays a clean negative result reported from the main sweep. The
arm list here is chosen from **what v2 measured, not from what was designed.**

**SHG schedules are mirrored into VP time.** `SHG_SCHEDULES` is written in flow
time (0 noise → 1 data); VP time runs the other way (1 noise → 0 data), and
`_Base.active()` looks the schedule up with whatever scalar the sampler passes.
Handing a flow-time schedule to `VPSampler` therefore runs every phase in the
wrong half of the trajectory — a band meant for the nearly-formed molecule
would fire in pure noise — and nothing would raise. `flow_to_vp_schedule`
maps `[lo, hi)` to `[1-hi, 1-lo)`; four gates cover it. `shg_plug_btvg` becomes
`[(0.0, 0.2, btvg), (0.2, 0.5, plug)]`, which also sits entirely inside the
`tau_max_guide = 0.5` window, and its btvg phase is normalised **per phase**
(finding S1: keying it off the arm name ran that field 3139× too strong).

Grid: 3 properties × 10 arms × 2 targets × 4 strengths, minus the unguided arm's
missing strength axis = **222 cells**.

### 4.3 Decision rules, fixed now

These are written before any cell has run and are not to be revised afterwards.

**R1 — the portability claim.** An arm of ours is *portable* if, on at least two
of the three properties, it beats `unguided` on `prop_mae_eval` by ≥ 3 combined
standard errors at its best strength. We will report portability per arm, not a
single verdict.

**R2 — the ranking claim.** Compute Spearman's ρ between the arm ranking on our
generator (main sweep, best strength per arm, same property) and on the borrowed
one, per property. **We commit to reporting all three ρ whatever they are.** If
ρ is low, the finding is that arm rankings are generator-specific, and that is a
more useful result for the field than a win.

**R3 — the home-turf test.** `tfg_mc` on TFG's own generator, guide and oracle
is the hardest single comparison available to us. Any of our arms that beats it
there is worth a sentence in the paper; any that loses to it there is reported
as losing.

**R4 — chemistry is a cost, not a metric to optimise.** An arm that improves MAE
while dropping `mol_stability` below the unguided value by more than 5 points is
reported with that drop in the same row, never separately.

**R5 — no post-hoc arm additions.** If an arm not in §4.2 is run later, it is
reported as a separate, post-hoc analysis with that label.

---

## 5. Caveats that travel with every number

These are not boilerplate. Each one is a way this experiment could mislead, and
each is stated in the driver's own docstring so it cannot be lost.

1. **Guide/oracle disjointness is inferred, not constructed.** Both TFG arg
   files record `dataset: qm9_second_half`, and TFG's training script mutates
   `args.dataset` *after* building its training loader, so that string describes
   an auxiliary test loader rather than the training split.
   `audit/fa_fb_search/disjointness_test.py` finds the error structure of a
   disjoint pair (30.9% / 40.9% / 28.2% against 37.3% / 37.3% / 25.3% predicted)
   and returns **DISJOINT CONSISTENT** — strong, but circumstantial. Our own
   pair is disjoint *by construction*. **This is the transfer's main weakness
   and it cannot be fixed from the released artifacts.** It must be stated
   whenever a transfer number is quoted beside one of ours.
2. **`delta` is optimistic.** It is 2 × the oracle's MAE measured on QM9
   molecules of which an unknown ~50% sit in the oracle's own training half, so
   the measured MAE is lower than a clean held-out MAE and `delta` is *tighter*
   than it should be. A tighter band lowers in-band coverage for every arm
   equally, so it favours no arm — it makes the absolute coverage numbers
   pessimistic, and they must be read that way.
3. **The sampler is ours, not EDM's.** EDM samples a 1000-step ancestral SDE;
   TFG samples 100 DDIM steps at η = 1. We integrate the PF-ODE at 100 Euler
   steps, because that is what our guidance fields are defined on and because
   holding the sampler fixed across arms is what makes the comparison internal.
   **Consequence: the `unguided` row here is not EDM's published unconditional
   row and must never be quoted as one.**
4. **The guide is asked at t = 0, and only at t = 0.** TFG's own method
   evaluates its time-dependent guide at the noisy state `x_tau` with that
   `tau`; our arms evaluate at the posterior mean, so they ask for t = 0. **The
   alternative is not measured and there is no flag for it.** Measuring it
   honestly needs a calibration refitted per `t`, because `calibrate` fits once
   at t = 0 — a switch alone would read a t-dependent network through a t = 0
   calibration. An earlier version of the driver carried exactly such a switch,
   wired to nothing; see §8.
5. **Molecule sizes come from our `val` split.** The driver draws its atom-count
   masks from the same `val` slice the main sweep starts from, so the two
   experiments see the same size distribution and a difference between them
   cannot be a difference in molecule size. This is **not** a leak: only the
   *number of atoms per molecule* crosses over, never coordinates, features,
   property values or any of our trained weights. A size histogram is public
   information about QM9.
6. **The guidance window is chosen, and it is the largest lever in the
   project.** `VPSampler` historically had no counterpart to
   `FlowSampler.t_min_guide`, so guidance ran at every `tau` including pure
   noise. The main sweep's own measurement is that this matters more than any
   arm difference — alpha MAE **10.02 → 5.32** as `t_min_guide` goes 0.05 → 0.5
   — so the transfer sets `tau_max_guide = 0.5`, the mirror of that window
   under the VP time convention. It also matters mechanically: `vp_posterior`
   divides by `alpha`, and `alpha(1.0) = 0.0032` under EDM's schedule, so
   guiding near `tau = 1` amplifies any epsilon error by ~300×. Had this been
   left open, a ranking inversion — the headline this experiment exists to
   produce — would not have been attributable to the backend.
7. **TFG's guide is a much better predictor than ours.** On the calibration set
   its MAE is 0.033 D (`mu`), 0.074 Bohr³ (`alpha`), 0.0020 Ha (`gap`) against
   our `f_A`'s 0.090 / 0.245 / 0.0039 — though see caveat 2 on why that is
   flattering. A stronger guide changes the difficulty of the task, so transfer
   MAEs are **not** comparable in absolute terms to main-sweep MAEs. Only
   improvement over the matched `unguided` baseline, and arm ordering, transfer.

---

## 6. Order of work

| # | step | command | cost |
|---|---|---|---|
| 1 | download the base model(s) | `fetch_tfg_assets.py --models EDMsecond,EDMfull` | minutes |
| 2 | full gates, generator included | `test_transfer_backend.py --require-edm` | seconds |
| 3 | **THE HARD GATE — benchmark the borrowed model through our evaluator** | `benchmark_transfer_base.py --edm-dir weights/EDMsecond --n 2000 --steps 100 --grid gamma --out results/bench/edmsecond_gate.json` | ~20 min |
| 4 | preflight — one cell per arm, writes nothing | `transfer_sweep.py --preflight` | ~2 min |
| 5 | the sweep | `transfer_sweep.py --n 512 --steps 100` | est. 6–9 h |
| 6 | the base-model comparison row (3 seeds × 10,000) | `benchmark_transfer_base.py ... --n 10000 --seed {0,1,2}` | ~3 h |
| 7 | read it | `select_arms.py` needs a `--out-dir` pass for `results/transfer` | — |

**Step 3 is a hard gate, and the plan stops there if it fails.** It is now a
script rather than an instruction: `benchmark_transfer_base.py` exits non-zero
if the borrowed model scores atom < 0.95 or molecule < 0.60 against EDM's
published 0.9837 / 0.8174. The threshold is deliberately loose — it asks *is
the adapter driving this checkpoint at all*, not *does our ODE reproduce their
SDE* — and our own model loses only 0.65 atom / 2.8 molecule points going from
500 steps to 100 ([../results/BASE_MODEL_BENCHMARK.md](../results/BASE_MODEL_BENCHMARK.md)
§3.2), so a correct adapter should clear it comfortably.

If it fails, **no guidance cell should be run until that is resolved**, because
every guided number would inherit the fault while looking plausible. The order
of investigation is in the script's docstring: re-run at the other `--grid` and
at `--steps 1000`; if that fixes it the cause is discretisation (F7), not the
adapter; if not, suspect the epsilon sign, the time direction, the one-hot
scale, or the EMA key.

Step 3 doubles as the **base-model comparison** the paper needs: it is the same
evaluator our own numbers come from, so it turns one row of the published
column from *quoted* into *measured*. Protocol and the table to fill in are
[../results/BASE_MODEL_BENCHMARK.md](../results/BASE_MODEL_BENCHMARK.md) §9.

Record step 3's output in
[../status/SCOPE_FM_GUIDANCE_STATUS.md](../status/SCOPE_FM_GUIDANCE_STATUS.md)
before starting step 5.

---

## 7. Open items

- **`osc` still has no citation.** It is labelled prior art in every document in
  this repo and is the best arm on `alpha` in the main sweep, and no citation
  for it exists anywhere in the repository (`SCOPE_FM_GUIDANCE_STATUS.md` §10b).
  Until that is resolved it must not be presented as a COMPARE arm in the
  transfer table either. Either find the citation or relabel it as ours —
  and if it is ours, it is our strongest result on `alpha` and the paper
  currently gives it away.
- **`select_arms.py` does not yet read `results/transfer/`.** It assumes the
  main sweep's cell-name grammar. Adding an `--out-dir` and a cell-name parser
  is small, and it is not done.
- **Cost parity is recorded but not yet enforced.** Every cell stamps
  `cost.{gen_fwd, gen_vjp, gen_jvp, guide_fwd, guide_bwd, guide_hvp}`, so an
  "at matched compute" comparison is available; nobody has written it.
- **Only one external base model.** A second (TFG-Flow, with a relaxation for
  its discrete type channel) would make the portability claim much stronger and
  is out of budget before the 29 Sep deadline.

---

## 8. The adversarial review, and what it changed

The harness was handed to an independent agent with the brief *find silent
defects; an adapter that feeds a network the wrong scale still returns finite
numbers and still produces a table.* It reported **44 of 44 gates passing** and
then found a blocker that made the experiment impossible to run. Both facts
belong in this record.

| # | finding | severity | action |
|---|---|---|---|
| **F1** | `EGNN.py` **imports** `remove_mean` and `remove_mean_with_mask` from a `utils.py` that is not vendored. `definitions()` raised `KeyError`; with the names simply dropped, `_forward` raised `NameError` on the first generator call. **All 222 cells would have failed.** | **blocker** | ✅ both ported into `tfg_assets.py` and injected into the exec namespace. The upstream `assert` in `remove_mean_with_mask` uses `.item()` — a device sync per generator call — and is replaced by a masked multiply that enforces the same property without the sync |
| **F2** | `evaluation.py::embedding_diversity` called `f_net_eval.net.embed(...)`, reaching *past* the wrapper that knows which feature scale the inner network wants. The borrowed oracle was therefore embedding one-hot values of **0.125**. Measured: `diversity_logdet` off by 12%, and the embeddings at **cos 0.64** to the correct ones — a different geometry, not a rescaling | **silent, corrupts a metric in every cell** | ✅ one-line fix to `f_net_eval.embed(...)`. Invisible for the entire main sweep because `PhysicalProperty.embed` and `.net.embed` coincide there; the transfer harness is the first caller for which they differ |
| **F3** | `--guide-time current` had **no caller anywhere** — `set_time` was never invoked, so it was bit-identical to `zero` and would have produced an ablation table reading "the choice does not matter" | dead ablation | ✅ flag and `time_mode` machinery **deleted**, not patched: a working version also needs a per-`t` calibration, which nobody had built. §5.4 now says only t = 0 was measured |
| **F4** | cell filenames encoded neither `n`, `steps`, `solver`, `seed` nor the window, so a smoke run at `--n 64` would be silently kept by a later `--n 512` sweep and the result set would mix sample sizes | resumability hole | ✅ `config_tag()` appends n / steps / solver / window / seed to every cell name |
| **F5** | `VPSampler` had **no counterpart to `t_min_guide`**, so every transfer cell would have guided at every `tau` including pure noise — below the worst point the project has already measured, and undisclosed | **methodological, undisclosed** | ✅ `tau_max_guide` added, default `0.5` in the driver (§5.6). This is the finding with the largest effect on the result |
| **F6** | `--k-delta` was accepted, written into every cell, and ignored (`delta = 2.0 * mae` hardcoded) | wrong provenance field | ✅ routed through `evaluation.choose_delta` |
| **F8** | two comments stated the opposite of the code: the schedule docstring claimed central differences (the code uses the interpolant slope, deliberately), and `dense_edges` claimed self-loops were removed from the edge list (they are removed from the *mask*, and the stated NaN rationale is false) | misleading | ✅ both rewritten. The second mattered: acting on it would have broken edge-list parity with EDM's `get_adj_matrix` |
| **F9** | `EDMGenerator.double()` left the schedule in float32; `Calibrated.y_mean` was dead and implied a `net*std+mean` form that does not hold; `fetch_tfg_assets.py` stored **no reference hash**, so `--verify` compared a file against itself | minor | ✅ `_apply` override; `y_mean` removed; `EXPECTED` table plus `--record` |

### 8.1 What the review verified as correct, numerically

Worth recording, because it is what the transfer's credibility rests on:

- **The property adapters are bitwise identical** (max abs diff `0.000e+00`) to
  the pre-existing, independently written loaders in `check_candidates.py` and
  `disjointness_test.py`, for all three properties. `TFGGuide.embed` and
  `TFGOracle.embed` are likewise `0.000e+00` against the true `graph_dec` input
  captured by a forward hook on the reference networks — so no layer, mask,
  `edge_attr` or `sin_embedding` branch is dropped or mis-ordered.
- **`beta = sigma^2 gamma'` is right**, re-derived independently, and holds to
  `1.5e-6` across **all 1000 cells**. `EDMSchedule.gamma` matches a live
  `PredefinedNoiseSchedule('polynomial_2', 1000, 1e-5)` on **all 1001 grid
  points**; the time direction agrees with both EDM and `diffusion.py`.
- **The generator wiring is right**: `in_node_nf = n_types + charges + 1`
  matches EDM's `dynamics_in_node_nf`; the pre-seeded edge list is byte-identical
  to EDM's own `get_adj_matrix`; the cache cannot go stale across a short final
  batch; and the `batch == 1` branch gives identical output.
- **`sampling.py`'s default path is unchanged** — `VPSampler(net, mask).beta is
  diffusion.beta` returns `True`. All existing suites pass.
- **No leakage path was found.** Taking molecule-size masks from our `val`
  split is benign: only the atom count per slot crosses over, which is a public
  statistic of QM9, and sharing it between the two experiments is what makes
  them comparable. Targets land within 0.1–0.3% of the main sweep's.

### 8.2 The lesson for the rest of the project

**The gate file reported "ALL PASS (44 gates)" with the entire generator path
unexecuted**, because every generator gate sat behind `if not have: return`. That
is the vacuous-gate failure this project has already caught twice (README,
*Testing discipline*). Two changes followed:

- `check_generator_defs()` now drives a randomly-initialised `EGNN_dynamics_QM9`
  of the real class, so the vendored definitions and the injected helpers execute
  **without the checkpoint**. It fails on F1.
- The embed gate previously called `cal.net.embed(c, cal._feats(f), m)` — it did
  the conversion itself and then reached past the wrapper, i.e. it tested a call
  production never makes. It now calls `embedding_diversity` exactly as
  `evaluate_samples` does, and carries a **negative control** asserting that
  feeding the wrong feature space actually changes the answer.

Gate count: 44 → 56 → **61** (the last five cover the gamma-uniform
time grid added as the F7 mitigation, §8.3).

### 8.3 Still open from the review

- **F7, now mitigated and measurable.** `--grid gamma` was added after the
  review: `EDMSchedule.tau_of_gamma` inverts the schedule so the time grid can
  be spaced evenly in log-SNR. On the 101-point grid the benchmark integrates,
  the worst single step falls from **3.702 nats to 0.228** (our own linear-beta
  schedule, for scale: 0.20). Same ODE, same endpoints, steps placed where the
  distribution moves. `gamma` is the default and `uniform` is kept so the
  difference is reported rather than assumed. The residual risk, unchanged: EDM's schedule is far stiffer
  than ours at the noise end under the same 100-step uniform-in-`tau` grid:
  measured on the grid the driver uses, the first Euler step changes log-SNR by
  **3.70** (ours: 0.20), and `|1 - beta*h/2| = 2.68` (ours: 1.10). The stiff
  terms largely cancel when epsilon is accurate, so this is not a guaranteed
  blow-up, but it multiplies epsilon error by ~17x over the first few steps
  relative to our own schedule. **This is exactly what step 4 of §6 tests**, and
  it is the most likely reason for that gate to fail. Mitigations if it does, in
  order of cost: a `tau` grid uniform in `gamma` rather than in `tau`;
  `--solver heun`; more steps. Setting `tau_max_guide = 0.5` already keeps
  *guidance* out of the stiffest region, but the base trajectory still passes
  through it.
- **Whether `EDMsecond` trained on a half disjoint from `tf_predict_*` /
  `evaluate_*` is unknown.** `disjointness_test.py` tests guide-vs-oracle only.
  Nothing in the released artifacts addresses the generator's half. State it.
- **Unverifiable until the checkpoint is downloaded:** whether `dynamics.*` keys
  load strictly; whether the generator's `args.pickle` records
  `include_charges=True`, in which case `tfg_assets.py` refuses loudly and the
  experiment is blocked a second time; real epsilon magnitudes; sample quality;
  wall-clock. The generator gates used random weights, so **structure and wiring
  are verified, numerics are not.**
