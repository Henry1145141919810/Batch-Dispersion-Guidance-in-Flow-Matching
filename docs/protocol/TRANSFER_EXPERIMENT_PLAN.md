# Transfer experiment: our guidance arms on a borrowed base model

**Written for:** the project team (Bobo, Idea, Haimo), and the defence. This is a
pre-registration, not a report — it is written before any transfer cell has run,
and the decision rules below are fixed now so that they cannot be chosen after
seeing the numbers.

**Status as of 2026-09-22.** Harness built and gated (44/44). Base-model
checkpoint not yet downloaded. Zero cells run.

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
| `proj1/scripts/fetch_tfg_assets.py` | downloads and hashes the missing checkpoint |
| `proj1/tests/test_transfer_backend.py` | **44 gates**, all passing |
| `proj1/src/sampling.py` | one change: `VPSampler` takes an optional `noise_schedule` |

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
| clip | 1.0 (velocity-relative trust region) | same as the main sweep |
| targets | q50, q90 of the QM9 property distribution | q50 interpolates, q90 asks for the tail |
| `delta` | 2 × the oracle's measured MAE | the pre-registered rule, unchanged |
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
| `spbc` | **OURS** | shape-preserving bias correction |
| `btvg` | **OURS** | band-targeted variance guidance |

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
4. **The guide is asked at t = 0.** TFG's own method evaluates its
   time-dependent guide at the noisy state `x_tau` with that `tau`; our arms
   evaluate at the posterior mean, so they ask for t = 0. `--guide-time current`
   measures the other choice. The pre-registered setting is `zero`; if the
   alternative is run, it is a labelled ablation.
5. **Molecule sizes come from our `val` split.** The driver draws its atom-count
   masks from the same `val` slice the main sweep starts from, so the two
   experiments see the same size distribution and a difference between them
   cannot be a difference in molecule size. This is **not** a leak: only the
   *number of atoms per molecule* crosses over, never coordinates, features,
   property values or any of our trained weights. A size histogram is public
   information about QM9.
6. **TFG's guide is a much better predictor than ours.** On the calibration set
   its MAE is 0.033 D (`mu`), 0.074 Bohr³ (`alpha`), 0.0020 Ha (`gap`) against
   our `f_A`'s 0.090 / 0.245 / 0.0039 — though see caveat 2 on why that is
   flattering. A stronger guide changes the difficulty of the task, so transfer
   MAEs are **not** comparable in absolute terms to main-sweep MAEs. Only
   improvement over the matched `unguided` baseline, and arm ordering, transfer.

---

## 6. Order of work

| # | step | command | cost |
|---|---|---|---|
| 1 | download the base model | `python proj1/scripts/fetch_tfg_assets.py` | minutes |
| 2 | full gates, generator included | `python proj1/tests/test_transfer_backend.py --require-edm` | seconds |
| 3 | preflight — one cell per arm, writes nothing | `python proj1/scripts/transfer_sweep.py --preflight` | ~2 min |
| 4 | **sanity: unguided only, all three properties** | `python proj1/scripts/transfer_sweep.py --arms unguided` | ~15 min |
| 5 | the sweep | `python proj1/scripts/transfer_sweep.py --n 512 --steps 100` | est. 6–9 h |
| 6 | read it | `select_arms.py` needs a `--out-dir` pass for `results/transfer` | — |

**Step 4 is a hard gate, and the plan stops there if it fails.** An unguided
EDM sampled with our PF-ODE at 100 Euler steps should produce chemistry in the
neighbourhood of a published EDM — atom stability well above 0.95 and molecule
stability well above 0.6. If it does not, the adapter is wrong somewhere the
gates do not reach (most likely the schedule or the epsilon sign), and **no
guidance cell should be run until that is resolved**, because every guided
number would inherit the fault while looking plausible.

Record step 4's output in
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
