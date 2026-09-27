# Property-targeted generation with inference-time guidance

Group 2 — Bobo Li · Henry Huang · Idea Idehpour · Haimo Fang

Steer a **frozen** continuous generative model to hit a target property value at
sampling time, without retraining it. Two modalities: **QM9 3D molecules** (most
of this repo) and **DNA sequences on the probability simplex** — the Modality 2
base model is **trained, validated and in this repository**
([`proj1/m2/`](proj1/m2/)); its guidance sweep is built and has not been run.
See [MODALITY2_V3_PLAN.md](docs/protocol/MODALITY2_V3_PLAN.md).

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

The base generator is trained and benchmarked; nine guidance arms are measured
on three properties; the three new arms built to fix why our method loses have
**not** been measured yet. See the status doc.
