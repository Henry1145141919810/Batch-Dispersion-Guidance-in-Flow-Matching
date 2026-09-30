# BDG: Batch-Dispersion Guidance — CIS 6270 Project 1, Group 2

Bobo Li, Henry Huang, Idea Idehpour, Haimo Fang
Department of Computer and Information Science, University of Pennsylvania

**The paper is [`paper/main.pdf`](paper/main.pdf)** — five pages of main text,
then references, then the appendix. Everything else in this folder exists to
support it.

---

## What we did, in one paragraph

Property-targeted generation asks for samples inside a tolerance band around a
target. Gradient guidance does that with one strength dial that moves a batch's
centre and its spread together. We introduce **Batch-Dispersion Guidance (BDG)**,
an inference-time rule that gives the deviation term its own coefficient, servoed
by the gap between the batch variance of predicted endpoint properties and a
requested setpoint. We test it on QM9 molecular coordinates and on DeepFlyBrain
DNA sequences on a probability simplex.

**The honest result: BDG does not improve band coverage over the rule it
modifies.** What it does do is reach operating points that rule cannot. Both
findings are in the paper, and this README reports them the same way.

---

## The headline numbers

All at the **q50** target (the median of each property's training distribution),
window `t >= 0.5`, 100-step Euler, n = 2,000 per cell pooled over three seeds to
6,000 samples, verdicts at the pre-registered Bonferroni bar **z = 2.99**.

**Base models** (unguided, percentages, mean ± seed sd):

| Generator | Mol. stable | Valid | Unique | NFE | s/sample |
|---|---|---|---|---|---|
| **FM, ours** | **39.7 ± 0.6** | **75.6 ± 1.1** | 99.8 | 100 | 0.051 |
| VP diffusion, ours | 28.8 ± 1.3 | 66.2 ± 0.7 | 99.9 | 101 | 0.048\* |
| EquiFM (flow, pretrained) | 86.2 ± 0.3 | 93.8 ± 0.6 | 99.7 | 100 | 0.086 |
| EDMsecond (diffusion, pretrained) | 67.2 ± 1.3 | 86.0 ± 1.3 | 99.8 | 101 | 0.087 |

Our two models share data, architecture, parameter count and training budget and
differ only in the generative family, so the first two rows are the matched
comparison. Flow matching wins all three chemistry metrics by 7 to 13 times the
seed standard deviation, which is why it is carried forward. \*That one timing is
a B200 MIG number; the rest are RTX A6000, and runtimes are not hardware matched.

**Guidance works, and is paid for in chemistry.** The plug-in rule raises decoded
coverage in 8 of 9 (generator, property) pairs and clears the bar in 5 of them,
at a cost of 3.96 points of molecule stability.

**BDG against the arm it modifies** (plug-in *is* BDG at zero gain), twelve
contrasts of two setpoints over three properties and two generators:

| | above the bar | below | tie |
|---|---|---|---|
| BDG vs plug-in | **0** | 3 | 9 |

**What the setpoint does reach.** Holding strength fixed, coverage tracks the
*measured* deviation gain at Spearman 0.92 to 0.99 in all twelve ablation grids.
The expanding setpoint widens the batch past unguided (R = 1.15 to 1.25) in
**6 of 6** cells; the strength dial reaches R > 1 in **0 of 6**, because raising
strength only ever contracts.

**For context**, TFG leads coverage in 5 of 6 cells while spending 8 to 27 points
of molecule stability. With no chemistry floor in this protocol, a coverage
leaderboard would simply crown whichever arm destroyed the most chemistry, which
is why every coverage figure in the paper is printed beside the chemistry it cost.

Every number above is regenerated from the cells by a script; see
**Connection to the paper** below.

---

## Layout

```
paper/                     the submission itself
  main.pdf                 THE PAPER
  main.tex                 the whole manuscript; body.tex is only a pointer
  figs/  tools/            figure sources; page, rubric and style checks
  CHANGELOG.md             what changed between drafts and why

code/                      everything that produced a number
  src/                     models, guidance rules, samplers, evaluation
  scripts/                 training, sweeps, table generation
  tests/                   closed-form gates, including 25 for BDG
  m2/                      Modality 2, the DNA simplex
  cluster/                 SLURM submission scripts

results_and_docs/
  results/                 53 script-generated result documents
  protocol/                the pre-registrations, written before the runs
  methods/                 method derivations and the prior-art audit
  status/                  what is done, running, or missing

weights/                   checkpoints plus README.md, the weight protocol
assignment_and_rubric/     the course assignment and paper template
MANIFEST.md                file count, sizes, SHA-256 of the graded artefacts
```

---

## Environment and setup

Python 3.12, PyTorch with CUDA for anything that samples.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch rdkit numpy scipy pandas pymupdf
```

`rdkit` is required for validity scoring; `pymupdf` only for the paper checks.

---

## Reproducing the results

**1. Build the dataset** (~430 MB, not shipped):

```bash
python code/scripts/prepare_qm9.py
```

QM9 arrives via DeepChem, pinned by SHA-256 on both the source archive and the
built tensor. One seeded permutation gives four disjoint splits: `train_a` and
`train_b` (51,527 each), `val` (17,748), `test` (13,083). The generator and the
guide `f_A` train on `train_a`; the evaluator `f_B` trains on `train_b`, so
nothing that scores a sample was trained on what steered it.

**2. Train**, or skip this and use the shipped checkpoints:

```bash
python code/scripts/train_fm.py         --split train_a   # ~11.5 h
python code/scripts/train_diffusion.py  --split train_a   # ~12 h
python code/scripts/train_predictor.py  --split train_a --prop mu   # f_A
python code/scripts/train_predictor.py  --split train_b --prop mu   # f_B
```

**3. Sample and score.** The headline stage and the ablation grid:

```bash
python code/scripts/transfer_sweep.py --stage v3    --n 2000 --backends fm,equifm,edm,vp
python code/scripts/transfer_sweep.py --stage v3abl --n 2000 --backends fm,equifm
python code/m2/m2_sweep.py --n 2000                        # Modality 2
```

**4. Regenerate every table and figure:**

```bash
python code/scripts/v3_table.py              --md-out results_and_docs/results/V3_RESULTS.md
python code/scripts/v3_blade_readout.py      --md-out results_and_docs/results/V3_BLADE_READOUT.md
python code/scripts/paper_controllability.py --md-out results_and_docs/results/PAPER_CONTROLLABILITY.md
python code/scripts/paper_fill_v3.py         --md-out results_and_docs/results/PAPER_TABLE_FILL_V3.md
```

**5. Build the paper:**

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex && bibtex main
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
python tools/pagecheck.py     # five-page limit, overfull boxes, dangling refs
python tools/check.py         # rubric structure, scope, house style
```

Both must pass before the paper is shared. `check.py` enforces things easy to
break by accident: every required section present, four or five contribution
bullets, every table and figure referenced from the text, no q90 number anywhere
(this study reports q50 only), and no phrase our own adversarial reviews refuted.

**The main text has no slack.** It sits at exactly five pages, and the count
moves in jumps rather than smoothly because LaTeX float placement absorbs small
edits. Adding a sentence needs a sentence cut to pay for it.

---

## The weights

`weights/README.md` is the weight protocol: what each checkpoint is, how it was
selected, and its hash. In short:

| File | What | Notes |
|---|---|---|
| `fm_ema.pt` | our flow-matching generator | EGNN 256x8, 3,753,229 params, epoch 1500, EMA 0.9999, `train_a`, seed 20260918 |
| `diff_ema.pt` | our VP diffusion generator | same backbone, params, budget and split; only the family differs |
| `f_A_{mu,alpha,gap}.pt` | the guides that steer | trained on `train_a` |
| `f_B_{mu,alpha,gap}.pt` | the evaluators that score | trained on `train_b`, disjoint from the guides |
| `deepflybrain/` | the Modality 2 activity gate | |

**Accept a generator checkpoint only if its recorded `args` show the right
family, split `train_a`, hidden 256, layers 8 and epoch 1500.** An earlier
`train_ab` checkpoint violated the split, because a generator trained on
`train_ab` would have seen the evaluator's training data.

Not included: **EDMsecond**, a third party's released checkpoint. We re-fetch it
rather than redistribute it — `code/scripts/fetch_tfg_assets.py`, and its md5 is
recorded in `weights/tfg_manifest.json`.

---

## Connection to the paper

No number in the paper is typed by hand. Each is emitted by a script that reads
recorded cells:

| Paper object | Produced by | Reads |
|---|---|---|
| Table 1, base models | `scripts/v3_table.py` | `results/v3/{fm,vp,equifm,edm}/v3/n2000/` |
| Table 2, guided vs unguided | `scripts/v3_table.py` | the same headline cells |
| Table 3, the ablation | `scripts/v3_blade_readout.py` | `results/v3/{fm,equifm}/v3abl/n2000/` |
| Table 4, recent methods | `scripts/v3_table.py` | the headline cells |
| Table 5, Modality 2 | `m2/m2_v3_results.py` | `results/m2/` |
| The R and exchange-rate findings | `scripts/paper_controllability.py` | the ablation grid |
| Figure 1, the overview | `paper/figs/overview.tex` | nothing; it is a diagram |
| BDG itself | `src/guidance.py`, the `mode == "bdg"` branch | |
| Its gates | `tests/test_bdg.py`, 25 checks | |

`results_and_docs/results/PAPER_TABLE_FILL_V3.md` is the audit that closes the
loop: it recomputes **every number printed in the paper** from the raw cells and
reports each as matching or not. At the last run, 93 of 93 table numbers matched.

---

## What is deliberately not here

| | Why |
|---|---|
| `data/` (430 MB) | rebuilt by `scripts/prepare_qm9.py`, pinned by SHA-256 |
| the raw cell tree (619 MB) | one JSON per cell; the script-generated summaries in `results_and_docs/results/` carry every number the paper uses |
| `weights/EDMsecond/` | a third party's released checkpoint; re-fetched, not redistributed |
| vendored reference repositories | external code we read but did not modify |
| build artefacts and caches | regenerated by the commands above |

Absolute cluster paths have been replaced by `$PROJECT_ROOT`, which the SLURM
scripts already accept, so no personal or shared-filesystem path ships here.

---

## Where the rubric is answered

| Rubric item | Where |
|---|---|
| Abstract | `paper/main.pdf` p.1 |
| 1 Introduction, contributions | p.1 |
| 2 Related Work, four paragraphs | p.1–2 |
| 3.1 Modalities, data, splits, conditioning | p.2, Appendix A.1 |
| 3.2 Flow-matching baseline | p.2 |
| 3.3 Diffusion baseline | p.2, Appendix A.2 |
| 3.4 Guided generation | p.2 |
| 3.5 The innovation, with hypothesis and controls | p.3 |
| 3.6 Transfer to Modality 2 | p.3 |
| 3.7 Model overview figure | p.3, Figure 1 |
| 4.1 Evaluation protocol, metrics, budget, statistics | p.3 |
| 4.2 Flow matching vs diffusion | p.4, Table 1 |
| 4.3 Guided vs unguided | p.4, Table 2 |
| 4.4 Innovation ablations | p.4, Table 3 |
| 4.5 Comparison with recent methods | p.4, Table 4 |
| 4.6 Modality 2 transfer | p.5, Table 5 |
| 4.7 Cross-modality analysis and failure modes | p.5 |
| 5 Discussion, limitations, narrowest claim | p.5 |
| Code: README and setup | this file |
| Code: organisation and comments | `code/`, and `tests/` for the gates |
| Code: connection to the paper | the table above, and `PAPER_TABLE_FILL_V3.md` |

---

## Known limitations, stated plainly

The paper states these; they are repeated here so nothing is discovered only by
reading closely.

- **Three of five ablation controls did not run**: a one-sided residual branch,
  an open-loop replay, and a signed-strength plug-in. Without the replay the grid
  shows that the deviation weight moves coverage, but cannot show that the
  *feedback* is what does it.
- **No external sequence comparator ran.** Three rows of the Modality 2 table
  have no data, and are marked as such.
- **The borrowed generator's sampler is not deterministic** on its accelerator;
  its re-runs move coverage by as much as the effects measured on it.
- **On Modality 2 the guidance window matters more than the rule.** Opening it to
  `t >= 0` buys the plug-in rule 22.8 points, about five times the best result
  obtained inside the protocol window.
- **Useful yield is not reported.** The metric needs per-molecule records that
  exist only for a superseded run whose flow-matching cells were steered by a
  different predictor pair, so the paper reports decoded band coverage throughout
  rather than mix two experiments.
