# Property-targeted generation with inference-time guidance

Group 2 — Bobo Li · Henry Huang · Idea Idehpour · Haimo Fang

> ### 📦 Reviewing the code submission? Start at **[SUBMISSION.md](SUBMISSION.md)**
>
> It is the entry point for the CIS 6270 code review: repository structure,
> environment and setup, dataset preparation, how to run training / sampling /
> evaluation / figures, where the innovation is implemented, and a table mapping
> **every paper table and figure to the script that produced it**.
>
> The other two files worth knowing:
> **[docs/results/DATA_INDEX.md](docs/results/DATA_INDEX.md)** says which data
> the paper uses and which runs are superseded;
> **[requirements.txt](requirements.txt)** pins the environment the committed
> results were produced in.

Steer a **frozen** continuous generative model to hit a target property value at
sampling time, without retraining it. Two modalities: **QM9 3D molecules** (most
of this repo) and **DNA sequences on the probability simplex**. **Both
modalities are now run end to end** — see the section immediately below, which
supersedes [MODALITY2_V3_PLAN.md](docs/protocol/MODALITY2_V3_PLAN.md)'s "not
started" framing.

## Blade run, 27–28 Sep: what is finished, and what it changes

**Read this before editing the paper.** All of it postdates the v4 draft, and
five findings below contradict things a reader would otherwise assume. Every
number here comes from committed cells; every claim names the file it is in.

### Everything ran, and validated

| stage | cells | |
|---|---|---|
| M1 headline | 144/144 | 3 backends × 3 properties × 3 seeds |
| M1 ablation | 612/612 | 17-arm η×τ grid at w ∈ {1,4} |
| M2 gc | 153/153 | headline at w ∈ {1,4} + the 17-arm ablation |
| M2 cpg | 42/42 | replication at w ∈ {1,4} |

**18,738 validation checks, 0 failures** (`proj1/scripts/v3_sanity.py`,
`proj1/m2/m2_sanity.py`). Result cells are in the repo; the per-cell sample
tensors are ~300 MB and stay on blade, gitignored, and regenerate exactly.

### The headline results

- **M1: BDG beats DPS by +0.95 pp in-band, paired, t = 5.71** over 18 pairs;
  holds separately on both backends (+1.04 fm, +0.86 equifm). Chemistry cost
  −7.15 pp molecule stability against DPS's −4.49.
- **The τ setpoint reverses sign in both modalities.** τ = 0.5 rows rise with
  η; τ = 1.5 rows fall and go negative, on all three M1 properties, both base
  models, and M2. Force cannot reverse the sign of its own parameter, so this
  is the controller — [M2_ABLATION_GRID.md](docs/results/M2_ABLATION_GRID.md).
- **M2/cpg is our strongest transfer number**: bdg_e4t0.5 **+18.30 pp**
  (t = +21) against plug's +1.47 at w = 4.

> **Cross-check, 28 Sep** (recomputed twice, independently, from the committed
> cells; [V3_BLADE_READOUT.md](docs/results/V3_BLADE_READOUT.md)). The +0.95 pp
> is `bdg_e4t0.5` − `plug` on continuous in-band, pooled over 18 cells. At the
> protocol's pre-registered per-cell bar (unpaired z = 2.99, FULL_RUN_V3_PROTOCOL.md
> §6.1) it is **0 above / 0 below / 6 ties**, on continuous, decoded and
> second-oracle in-band alike. The other headline BDG arm, `bdg_e4t1`, pooled the
> same way is **−1.21 pp (t = −5.97)**, with 3 of 6 cells significantly below plug.
> Plug's like-for-like chemistry is **−3.96 pp** over the same 6 (base, property)
> cells; the −4.49 includes edm, which has no BDG arm. In the M1 ablation, the
> τ = 1.5 losses hold in 12 of 12 grids, while the τ = 0.5 row rises with η in
> 5 of 12, all of them at w = 1. The paired, useful-yield reading that the Betty run
> got still has to be run on blade (readout §5).

### Five findings that change what the paper may say

1. **On M2 the guidance WINDOW is worth ~10× the guidance METHOD.** Guiding
   from t = 0 rather than v3's t ≥ 0.5 gives plug **+22.8 pp**; the best any
   method achieves inside v3's window is +2.5 pp. And at t = 0 **BDG loses to
   plug** (−2.33 pp). Cause: the (1−t)/t factor diverges, the clip truncates
   30 % of steps, and arms converge. **No BDG claim may be stated without the
   window it was measured at.** [M2_WINDOW_DOMINATES.md](docs/results/M2_WINDOW_DOMINATES.md)
2. **Three of M2's four baselines are one baseline in v3's window.** plug, tmpd
   and lgd_mc return identical numbers (max|d| ≤ 4.2e-6) because the measured
   `v_f` collapses 38,000× by t = 0.5, so TMPD's denominator → s² and LGD's
   draw scale → 0. Reporting them as three agreeing baselines would claim
   independent confirmation that does not exist. Protocol §3.0.
3. **BDG raises M2 in-band by CONTRACTING, while the bias worsens** (+0.985 →
   +1.277 δ). On M1 it does both. The transfer is partial: the dispersion half
   transfers, the target-seeking half has nothing to do because M2's base
   already sits on target. [M2_MECHANISM.md](docs/results/M2_MECHANISM.md)
4. **`tmpd` was silently DPS until 27 Sep.** `v_f` was read from the table as
   one scalar broadcast over the batch, so TMPD had no per-sample weighting at
   all — on both properties. Any pre-fix TMPD claim is void. Fixed; M1's
   implementation was always correct.
5. **EquiFM is not bit-reproducible** (same arm, same seed differs by 1.5e-3
   in-band); fm and edm are bit-exact. EquiFM's error bars are therefore not
   purely seed-to-seed, and its η = 0 control needs a tolerance.

> **Cross-check, 28 Sep, of the five findings.** Findings 2 and 5 hold. Finding 1: the
> +22.8 pp and the clip figure hold, but the best result inside v3's window is
> **+4.23 pp** (`bdg_e8t0.5`, w = 4, [M2_ABLATION_GRID.md](docs/results/M2_ABLATION_GRID.md)),
> not +2.5, so **on gc** the window is worth about 5× the method, not 10×. Both terms
> are gc-only: cpg has no t = 0 cell, and its in-window gain reaches +18.30 pp, so no
> ratio holds for M2 as a whole. At t = 0, against plug, `bdg_e4t0.5` is a tie
> (−2.33 pp, z −2.67), while `bdg_e4t1` is a clear loss (−20.80 pp, z −26.58).
> Finding 3 holds for M2, but "on M1 it
> does both" fails on our own base: |bias| worsens on fm/alpha and fm/gap
> ([V3_RESULTS_fm.md](docs/results/V3_RESULTS_fm.md)). Finding 4 applies to M2's
> tmpd only. Below, the DeepFlyBrain "verified identical to Keras" figures are in
> no committed file, and `deepflybrain.py` says the port was validated *without* TF.
> Details and the open items are in
> [SCOPE_FM_GUIDANCE_STATUS.md](docs/status/SCOPE_FM_GUIDANCE_STATUS.md).

### M2 now has a functional evaluation, and the base model is the weak link

M2 had no analogue of RDKit validity — every ACGT string is legal, so in-band
could be bought with strength. Two gates now close that:

- `proj1/m2/enhancer_gate.py` — a discriminator trained against a **per-sequence
  order-1 Markov null**, which matches each sequence's own GC and CpG, so the
  gate is **blind to what guidance steers** (corr +0.001 GC, −0.004 CpG). AUC
  0.992. A global Markov null was tried first and **rejected**: it leaves GC
  spread usable (|GC−mean| alone gives AUC 0.73) and would have punished BDG
  for narrowing spread, which is its mechanism.
- `proj1/m2/deepflybrain.py` — **DeepFlyBrain itself** (Janssens et al., Nature
  2022), the accessibility predictor trained on our corpus, ported to PyTorch
  and **verified numerically identical to the published Keras model**
  (max|d| 4.5e-08, r = 1.0000000000). Note for anyone reloading it: Keras 2.2.4
  used `recurrent_activation='hard_sigmoid'`; modern TF defaults to `sigmoid`
  and silently gives a different model.

| | max topic | frac predicted accessible |
|---|---|---|
| real regions | 0.279 | 0.13–0.14 |
| **our base (unguided)** | **0.190** | **0.030** |
| strongest guided rung (w = 64) | 0.189 | 0.029 |

**Two things follow, and the second reframes the first.** Guidance costs
essentially nothing functionally — at most −0.003 across every arm and
strength. But **our samples are ~4.5× less likely to be predicted accessible
than real enhancers** (0.030 vs 0.13–0.14 frac active), which no distributional
metric showed. That is a limitation of the frozen base and belongs in §4.7.
Caveat: DeepFlyBrain is a predictor with its own error, not an assay, and our
corpus is its training data.

### Why nothing degrades on M2 — and why that is a limitation, not a win

"Guidance at maximum strength damaged nothing" does not survive contact with
the sequences. **It is true, it is not a bug, and it says something weaker than
it sounds.** Checked three ways:

- **The stored samples are the guided ones.** Every cell's stored sequences
  reproduce its own recorded `gc_mean` exactly.
- **Guidance barely moves the sequences.** At w = 64 it changes **0.63–0.80 %
  of positions — 3 to 4 bases in 500.** The output is 99.2 % identical to
  unguided, so a motif-level predictor correctly sees almost nothing.
- **The predictor is not blind, it is correctly insensitive.** Randomising the
  same fraction gives the same non-effect, and it responds properly to real
  damage:

  | bases randomised | Δ dfb_max_topic |
  |---|---|
  | 0.8 % (what guidance changes) | −0.0003 |
  | 5 % | −0.0080 |
  | 20 % | −0.0311 |

  Guidance at w = 64 lands at −0.003, exactly where a perturbation of its size
  belongs.

**The arithmetic behind it.** GC is a mean over 500 bases, so one flip moves it
by 1/500 = 0.002, and the band is δ = 0.00883 — **±4.4 bases**. Hitting the
target is therefore a few-substitution edit. It also explains the modest gain:
14.6 % of sequences are already inside, 27.4 % are within 4 flips, but the
median sequence outside needs **14.6 flips**. Guidance moves the borderline
ones in and leaves the rest — the +4 pp we measure.

**So the correct claim is NOT "BDG preserves quality on DNA where competitors
do not."** It is: *the DNA task, with a ±4.4-base band on a 500-base sequence,
is not one where any method must trade quality for accuracy.* No fidelity
trade-off exists to win. That is structurally unlike M1, where moving a
molecular property means changing the molecule and chemistry pays for it
(TFG: −19.7 pp stability). **The M1 comparison is the load-bearing one.**

### Bugs fixed in shared code

* `v3_table.py` refused **every** equifm table (δ compared exactly, but the GPU
  calibration agrees only to 8 significant figures) and **every** ablation table
  (the w = 1 assertion applied to the ablation, which sweeps w ∈ {1,4}; plus
  `KeyError: 'edm'` on the `--both` path, a follow-on to 2df1d74).
* `batch_memprobe.py` reports a false **zero** for edm — it probes with
  `bdg_e4t0.5`, which edm does not plan.
* M2's `kmer_freq` crashed on GPU; `m2_sanity` returned **success on an empty
  tree**, so a stage that produced nothing passed its gate.

## Start here

| if you want… | read |
|---|---|
| **what is done, what is running, what is missing** | [docs/status/SCOPE_FM_GUIDANCE_STATUS.md](docs/status/SCOPE_FM_GUIDANCE_STATUS.md) |
| **every guidance method, ours vs prior art, with its proof doc** | [docs/methods/GUIDANCE_METHODS_INDEX.md](docs/methods/GUIDANCE_METHODS_INDEX.md) |
| the timetable and who owns what | [docs/status/PLAN_AND_TIMETABLE.md](docs/status/PLAN_AND_TIMETABLE.md) |
| how the data is split and why | [docs/protocol/SPLIT_PROTOCOL.md](docs/protocol/SPLIT_PROTOCOL.md) |
| how the base model compares to published work | [docs/results/BASE_MODEL_BENCHMARK.md](docs/results/BASE_MODEL_BENCHMARK.md) |
| **the headline guidance result** (v2, fixed target) | [docs/results/FULL_RUN_V2_RESULTS.md](docs/results/FULL_RUN_V2_RESULTS.md), pre-registered in [docs/protocol/FULL_RUN_V2_PROTOCOL.md](docs/protocol/FULL_RUN_V2_PROTOCOL.md) |

**The two status documents above are living documents — update them, do not
write new ones.**

## Layout

```
proj1/src/          models, guidance, samplers, evaluation
proj1/scripts/      training, sweeps, selection, benchmarking
proj1/tests/        43 closed-form gates (run them; they are fast)
proj1/cluster/      SLURM job scripts for PARCC Betty
blade_runs/         the blade GPU box's lanes and queues (no scheduler there)
proj1/m2/           Modality 2: DNA sequences on the simplex
weights/            inference-ready checkpoints (24 MB) -- see weights/README.md
results/            one JSON per experiment cell -- THE experimental record
                    (results/v3/<be>/v3|v3abl/n2000/ is the blade run;
                    results/v3/<be>/n5000/ is the earlier Betty run)
docs/status/        living status + plan
docs/methods/       guidance methods and their proofs
docs/results/       benchmarks and verified findings
docs/protocol/      split, predictor decisions, experiment plan, Betty runbook
docs/reference/     outside reading: Haimo's v1 study, the writing guide
paper/              the manuscript (see "The paper and the slides" below)
slides/             the defense deck; slides/handoff/ is its content spec
course/             course-supplied: assignment, paper template, citation.bib
audit/              prior-art audit; bdg_review/ and top_guidance_2026/ tracked
```

Not in the repo, by design: `data/` (430 MB, rebuild with
`prepare_qm9.py`), `betty_pull/` (cluster sync), `logs/` (cluster job logs),
`bundles/` (every tarball shipped to or pulled from Betty; its README says
which `code_vN` is next), most of `audit/` (vendored reference repos),
`archive/` (brainstorms, dropped ideas and the superseded root scratch
folders; six of its eight docs are cited from `docs/`, and `archive/INDEX.md`
maps each one).

**Two exceptions under `audit/` are tracked**: `fa_fb_search/TFG/` and
`fa_fb_search/OC-Flow/` (28 MB). `proj1/src/external/tfg_assets.py` loads model
definitions and checkpoints from them, so the transfer experiment does not run
from a fresh clone without them. Origin, commits and licences are in
[audit/fa_fb_search/PROVENANCE.md](audit/fa_fb_search/PROVENANCE.md). None of
that code or those weights are ours.

## Reproducing

**New to the project? Read [ONBOARDING.md](ONBOARDING.md) instead** — it is the
same ground with every step's expected output, so you can tell success from a
silent failure.

```bash
python -m venv .venv && ./.venv/Scripts/activate      # or source .venv/bin/activate
pip install torch rdkit

python proj1/scripts/download_qm9.py                   # raw QM9, SHA-256 verified
python proj1/scripts/prepare_qm9.py                    # rebuilds data/qm9.pt (seeded)
python proj1/tests/test_v2_arms.py                     # expect ALL PASS
python proj1/tests/test_arms_exact.py                  # expect ALL PASS

# one throwaway cell per arm, ~15 s, writes nothing -- run this before any sweep
python proj1/scripts/guidance_sweep.py --stage v2 --props mu --preflight

# the screening sweep (resumable: one JSON per cell, completed cells are skipped)
python proj1/scripts/guidance_sweep.py --props mu,alpha,gap --n 512 --steps 100

# read the result
python proj1/scripts/select_arms.py --stage main
```

The **transfer experiment** — every guidance arm re-run on a borrowed base
model, guide and oracle, so that nothing in it but the guidance field is ours —
needs one download and then runs from the clone:

```bash
python proj1/scripts/fetch_tfg_assets.py            # TFG's EDMsecond, ~100 MB
python proj1/tests/test_transfer_backend.py         # expect ALL PASS (61 gates)

# the hard gate: the borrowed model through OUR evaluator. If this fails,
# nothing downstream is trustworthy -- see the script's docstring.
python proj1/scripts/benchmark_transfer_base.py --edm-dir weights/EDMsecond     --n 2000 --steps 100 --grid gamma --out results/bench/edmsecond_gate.json

python proj1/scripts/transfer_sweep.py --preflight  # one cell per arm, ~2 min
```

Read [docs/protocol/TRANSFER_EXPERIMENT_PLAN.md](docs/protocol/TRANSFER_EXPERIMENT_PLAN.md)
first — it is a pre-registration, and §5 lists the caveats that must travel
with every number it produces.

Training the generator from scratch is `proj1/scripts/train_fm.py --split train_a`
(1500 epochs); the predictors are `proj1/scripts/train_predictor.py`. Both are
long GPU jobs — the published weights in `weights/` let you skip them.

## Testing discipline

Every guidance arm has a **closed-form gate**: a fixture with a diagonal
posterior and a quadratic property, where every quantity has an exact analytic
value, so a wrong sign or a dropped factor moves a number.

`results/bench/mutation_test.py` injects **29 deliberate defects** and requires
every one to be caught. It runs against a sandbox copy, never the working tree.
Three of those 29 are real bugs this project shipped and the gates missed at the
time — they are in there so they cannot come back.

Gates that pass *vacuously* are worse than no gates. Two were found and fixed:
one compared against `≤ 0`, which a zero gradient satisfies, and one
short-circuited on an empty comparison set. Both now assert their fixture is
actually exercising them.

## The paper and the slides

The current paper is **manuscript v4**, dated 28 September 2026. BDG is the
selected innovation. Final comparative results are pending and appear as `P`
(the `\pend` macro). The v4 draft predates the blade run, so its `P` cells can
now be filled from `docs/results/V3_RESULTS*.md` and `docs/results/M2_*` —
read the blade-run section above first.
The main text occupies five pages; references and appendix follow.

- `paper/main.tex`: the complete editable manuscript, including the overview,
  tables, algorithm, and appendix.
- `paper/main.pdf`: current compiled paper.
- `paper/versions/v4_main.pdf`: numbered release for tracking.
- `paper/versions/v4_2026-09-28/`: rebuildable snapshot with bibliographies and style.
- `paper/versions/v3_revision_notes.pdf`: reasons for the v3 revision and rubric map.
- `paper/versions/CHANGELOG.md`: version history; v1 to v3 remain preserved.
- `paper/body.tex`: compatibility notice directing edits to main.tex.
- `paper/figs/`: earlier figures retained for historical versions.
- `slides/defense.tex`: existing defense deck, maintained separately.

Build the paper from `paper/`:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
python tools/check.py
python tools/pagecheck.py
```

The page check requires `pypdf` and includes every main-text float before the
references. The integrity check validates section labels, citations to floats,
and selected style/scope constraints; it does not certify empirical completion.
`tools/prosecount.py` reads the marked main-text region of main.tex. All checks
must be rerun after final results replace `P` values, because longer tables can
change pagination. Pending cells stay visible until actual results replace them.

The implementation/protocol map is in Appendix A.7 and the separate revision
notes. Historical pilot tables must not be substituted for the final molecular
protocol or the unfinished sequence comparisons. No experiment was run during
the v3 paper revision.

Build the existing slides separately from `slides/` with `pdflatex defense`
(two passes). This paper revision does not update the deck.

### External code we adapted, and what is not ours

| What | Where it came from | What we did |
|---|---|---|
| TFG model definitions, `EDMsecond` checkpoint | TFG release, vendored under `audit/fa_fb_search/TFG/` | loaded by `proj1/src/external/tfg_assets.py`; not modified |
| EDM noise schedule (`polynomial_2`, clipping) | EDM reference implementation | ported verbatim in `proj1/src/external/edm_schedule.py`, gated to 1e-5 |
| EDM stability and validity conventions, bond tables | EDM `bond_analyze.py` | reimplemented and verified entry-for-entry, 0 mismatches |
| OC-Flow reference | vendored under `audit/fa_fb_search/OC-Flow/` | reference only |
| Enhancer sequence data | MOG-DFM dataset release | Kenyon-cell half only; see `docs/methods/BDG_HANDOFF.md` §11 |

Provenance, commits and licences: [audit/fa_fb_search/PROVENANCE.md](audit/fa_fb_search/PROVENANCE.md).
None of that code or those weights are ours, and the paper labels every borrowed
row as borrowed.

## Status in one line

Both modalities have run v3 end to end: 144 + 612 M1 cells and 153 + 42 M2 cells,
with 0 validator failures. At the pre-registered per-cell bar, **BDG does not beat
plug on M1** (`bdg_e4t0.5` 0/0/6 ties; `bdg_e4t1` 0 above / 3 below). On M2
(bar inherited from M1, since M2 pre-registered none), `bdg_e4t0.5` beats plug on 3 of
4 (property, w) cells. It gets there by contracting the spread while its bias
worsens, and at about 10× plug's applied correction on gc. `bdg_e4t1` ties plug in all
4. Our own VP diffusion model **has results** (28 Sep): 18 v3 cells, `--backend vp`, checkpoint `weights/diff_ema.pt` md5 `8a3390a6`. **Flow matching beats it on molecule stability, validity and atom stability at matched everything** -- see [docs/results/V3_RESULTS_vp.md](docs/results/V3_RESULTS_vp.md) and [DATA_INDEX.md](docs/results/DATA_INDEX.md). Start from
[V3_FINAL_SUMMARY.md](docs/results/V3_FINAL_SUMMARY.md); the paper fill map is
[PAPER_TABLE_FILL_V3.md](docs/results/PAPER_TABLE_FILL_V3.md).
