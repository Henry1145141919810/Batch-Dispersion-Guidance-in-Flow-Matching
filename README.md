# Property-targeted generation with inference-time guidance

Group 2 — Bobo Li · Henry Huang · Idea Idehpour · Haimo Fang

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
weights/            inference-ready checkpoints (24 MB) -- see weights/README.md
results/sweep/      one JSON per experiment cell -- THE experimental record
docs/status/        living status + plan
docs/methods/       guidance methods and their proofs
docs/results/       benchmarks and verified findings
docs/protocol/      split, predictor decisions, experiment plan
```

Not in the repo, by design: `data/` (430 MB, rebuild with
`prepare_qm9.py`), `betty_pull/` (cluster sync), most of `audit/` (vendored
reference repos), `archive/` (brainstorms and dropped ideas — every one is
cited from the methods index, with the reason it was dropped).

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

```
paper/main.tex        the manuscript; body.tex holds the main text, so the
                      5-page limit can be measured independently of the appendix
paper/body.tex        abstract, sections 1-5, the five result tables
paper/refs_extra.bib  references beyond the course-supplied citation.bib
paper/figs/           overview.tex (Figure 1) and make_mechanism.py (Figure 2)
slides/defense.tex    the 29-slide defense deck, labelled 1a-3e
slides/README_SLIDES.md  speaker split, timing, and what must not be said
```

Build both, from the repository root:

```bash
cd paper/figs && pdflatex overview.tex && cd ../..    # Figure 1
./.venv/Scripts/python.exe paper/figs/make_mechanism.py   # Figure 2

cd paper  && pdflatex main && bibtex main && pdflatex main && pdflatex main
          && ../.venv/Scripts/python.exe tools/pagecheck.py   # enforces 5 pages
cd ../slides && pdflatex defense && pdflatex defense
```

`paper/tools/pagecheck.py` reports the main-text page count against the 5-page
limit and counts the unresolved `\TODO` markers; `paper/tools/prosecount.py`
reports the prose budget per section. `paper/tools/check.py` is the one to run
before sharing a build: it verifies the rubric's structural elements, that every
table and figure is referenced from the text, that the q50-only scope holds, and
that no phrase our own adversarial review refuted has crept back in. All three are
advisory and produce no result.

The main text sits at **exactly 5.00 pages with no slack**, so adding a sentence
needs a sentence cut to pay for it. Page count is quantized by float placement, so
small trims often move nothing; see `paper/versions/CHANGELOG.md`.

Pending numbers are marked three ways, all rendered red while `\DRAFTtrue` is set
in `paper/main.tex`: `\pend` is one missing value in a table cell, `\phfig` is a
figure reserved at its final height, and `\TODO` is an inline note. Setting
`\DRAFTfalse` hides all three for a clean build.

### Which script produced which table or figure

| Paper item | Produced by | Reads |
|---|---|---|
| Table 1, base models (§4.2) | `proj1/scripts/benchmark_base.py` | `weights/fm_ema.pt`, `results/bench/rescored/*.json` |
| Table 2, guidance (§4.3) | `proj1/scripts/onegen_full_metrics.py` | `results/bdg_local/*.json` |
| Table 3, BDG ablations (§4.4) | `proj1/scripts/bdg_full_metrics.py` | `results/bdg_port/cells/*.json` |
| Table 4, recent methods (§4.5) | `proj1/scripts/transfer_sweep.py` → `v3_table.py` | `results/v3/` (pending) |
| Table 5, Modality 2 (§4.6) | `proj1/m2/m2_sweep.py` → `make_vf_table.py` | `results/m2/` (pending) |
| Figure 1, overview (§3.7) | `paper/figs/overview.tex` | nothing; it is a diagram |
| Table 8, the full ladder (App. A.3) | `proj1/scripts/bdg_table.py` | `results/bdg_port/table.txt` |
| Figure 2, the controller (App. A.3) | `paper/figs/make_mechanism.py` | `results/bdg_port/table.txt` |
| The innovation itself | `proj1/src/guidance.py`, the `mode == "bdg"` block | — |
| Its gates | `proj1/tests/test_bdg.py` (25 checks) | — |

Appendix Table 11 carries this same map inside the paper, so a reader never has to
leave it to find what produced a number.

The v3 headline and ablation stages are pre-registered in
[docs/protocol/FULL_RUN_V3_PROTOCOL.md](docs/protocol/FULL_RUN_V3_PROTOCOL.md)
and [docs/protocol/ABLATION_V3_PROTOCOL.md](docs/protocol/ABLATION_V3_PROTOCOL.md).
Every `\TODO` in the paper and the slides points at a cell from those two.

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

Both modalities are run end to end and validated: 144 + 612 M1 cells, 153 + 42
M2 cells, 18,738 checks, 0 failures. BDG beats DPS by +0.95 pp paired on M1
(t = 5.71) and the τ setpoint reverses sign in both modalities. Read the blade
run section above before editing the paper — five findings there change what may
be claimed.
