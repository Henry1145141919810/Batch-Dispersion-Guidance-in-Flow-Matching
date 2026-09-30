# BDG: Batch-Dispersion Guidance — CIS 6270 Project 1, Group 2

Bobo Li, Henry Huang, Idea Idehpour, Haimo Fang
Department of Computer and Information Science, University of Pennsylvania

**The paper is [`paper/main.pdf`](paper/main.pdf)**, manuscript v10: main text,
then references, then the appendix. Everything else in this folder exists to
support it. Table, figure and section numbers below are v10's.

---

## What we did, in one paragraph

Property-targeted generation asks for samples inside a tolerance band around a
target. Gradient guidance does that with one strength dial that moves a batch's
centre and its spread together. We introduce **Batch-Dispersion Guidance (BDG)**,
an inference-time rule that gives the deviation term its own coefficient, servoed
by the gap between the batch variance of predicted endpoint properties and a
requested setpoint `tau`. At zero gain it is exactly the plug-in rule it modifies.
We test it on QM9 molecular coordinates and on DeepFlyBrain DNA sequences on a
probability simplex.

**The result, stated the way the paper states it.** On molecules, contracting
BDG raises mean band coverage over plug-in in all six comparisons, but by 0.45 to
1.52 points, and none clears the pre-registered bar. What it adds is reach: the
setpoint moves spread in both directions, including past unguided, which the
strength dial cannot. On DNA the effect has a sign, and the setpoint predicts it:
BDG gains 7.08 points on GC and **loses 4.88 on CpG**, because on CpG the plug-in
baseline is already tighter than the tightest setpoint in our grid. What
transfers is the mechanism and its sign rule, not a coverage gain.

---

## The headline numbers

At the **q50** target (the median of each property's training distribution),
100-step Euler, n = 2,000 per cell in batches of 500, three seeds, mean ± seed
sd. Molecular verdicts use the pre-registered Bonferroni bar **|z| ≥ 2.99**.

**Base models** (unguided, percentages; paper Table 1):

| Generator | Mol. stable | Valid | Unique | NFE | s/sample |
|---|---|---|---|---|---|
| **FM, ours** | **39.7 ± 0.6** | **75.6 ± 1.1** | 99.8 | 100 | 0.051 |
| VP diffusion, ours | 28.8 ± 1.3 | 66.2 ± 0.7 | 99.9 | 101 | 0.048\* |
| EquiFM (flow, pretrained) | 86.2 ± 0.3 | 93.8 ± 0.6 | 99.7 | 100 | 0.086 |
| EDMsecond (diffusion, pretrained) | 67.2 ± 1.3 | 86.0 ± 1.3 | 99.8 | 101 | 0.087 |

Our two models share data, architecture, parameter count and training budget and
differ only in the generative family, so the first two rows are the matched
comparison. Flow matching wins stability by 10.9 points and validity by 9.4, which
is why it is carried forward. \*That one timing is a B200 MIG number; the rest are
RTX A6000, and runtimes are not hardware matched.

**Guidance works, and is paid for in chemistry** (Table 2; `w = 1`, `t >= 0.5`).
Plug-in raises decoded coverage on all six (model, property) pairs of our FM and
VP models. Dipole and gap clear the bar on both (+2.75 / +2.35 points on FM,
+2.13 / +2.60 on VP); polarizability does not. Molecular stability falls by 2.5 to
6.2 points.

**BDG against the arm it modifies** (Table S8): twelve contrasts, two setpoints ×
three properties × two flow backbones:

| | above the bar | below | tie |
|---|---|---|---|
| BDG vs plug-in | **0** | 3 | 9 |

All three losses are at the widening setpoint `tau_m = 1`. At the contracting
setpoint `tau_m = 0.5`, all six differences are positive (+0.45 to +1.52 points)
and mean absolute target error is lower in all six, but none is resolved.

**What the setpoint reaches** (Figure 2a, Appendix A.6). At fixed strength
(`eta = 8`, `w = 1`), decoded spread rises monotonically with the setpoint in all
six (backbone, property) curves, reaching about 1.15 to 1.25 times unguided at
`tau_m = 1.5`. The tested plug-in strengths `w ∈ {1, 4}` stay below unguided
spread. Across the twelve ablation grids, coverage tracks the measured deviation
coefficient at Spearman 0.92 to 0.99. That association is descriptive: it does
not separate the feedback loop from the size of the coefficient.

**For context**, TFG leads coverage in 5 of 6 molecular cells while spending 8 to
27 points of molecular stability relative to plug-in (Tables 4, S4, S5). With no
chemistry floor in this protocol, a coverage leaderboard would crown whichever arm
destroyed the most chemistry, which is why every coverage figure in the paper is
printed beside the chemistry it cost.

**Modality 2, DNA** (Table 5; `w = 4`, guidance from `t >= 0.3` for both
properties):

| | plug-in | BDG, `tau_m = 0.5` | difference (paired seed sd) |
|---|---|---|---|
| GC coverage | 30.25 | 37.33 | **+7.08** (4.53) |
| CpG coverage | 91.51 | 86.63 | **−4.88** (1.33) |

BDG contracts only a batch wider than `tau`. Plug-in leaves GC at 0.62 s, wider
than the `tau_m = 0.5` setpoint, and CpG at 0.37 s, already inside it. BDG
therefore tightens GC to 0.59 s and relaxes CpG to 0.45 s, and coverage follows
the sign. An exploratory setpoint below CpG's achieved spread, `tau_m = 0.25`,
contracts it to 0.298 s and recovers +2.28 points, with all three seeds agreeing
in sign and a higher 3-mer JSD (Table S17). That setpoint was chosen after seeing
the registered result, so it tests the explanation and is not a headline number.

The **pre-registered DNA window is `t >= 0.5`** (Tables S18, S19). There,
plug-in, TMPD-inspired and LGD-MC give identical coverage on both properties, and
BDG at `tau_m = 0.5` is ahead on both (GC 14.1 → 16.4, CpG 51.6 → 68.5). The
`t >= 0.3` follow-up was added after diagnosing that overlap, and it is the window
Table 5 reports. The two windows are never pooled.

Every number above is regenerated from the result cells by a script; see
**Connection to the paper** below.

---

## Layout

```
paper/                     the submission itself
  main.pdf                 THE PAPER (v10)
  main.tex                 the whole manuscript; body.tex is only a pointer
  figs/  tools/            figure sources; table generator, page and rubric checks
  CHANGELOG.md             what changed between drafts and why

code/                      everything that produced a number (proj1/ in the repository)
  src/                     models, guidance rules, samplers, evaluation
  scripts/                 training, sweeps, table generation
  tests/                   gate suites, including test_bdg.py for BDG
  m2/                      Modality 2, the DNA simplex
    blade_bundle/          the M2 generator checkpoint and the DNA dataset
  cluster/                 SLURM submission scripts

results_and_docs/
  results/                 53 script-generated result documents
  protocol/                the pre-registrations, written before the runs,
                           and the dated addendum for the DNA window follow-up
  methods/                 method derivations and the prior-art audit
  status/                  project status and run logs (historical)

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
pip install torch rdkit numpy scipy pandas matplotlib pymupdf
```

`rdkit` is required for validity scoring; `pymupdf` only for the paper checks.

**One step before running anything.** The scripts import from the repository
layout, where `code/` is named `proj1/`. From the root of this folder:

```bash
ln -s code proj1                   # Windows: mklink /D proj1 code
```

Without it, the training and sampling scripts stop with
`ModuleNotFoundError`. The commands below use the `proj1/` paths.

---

## Reproducing the results

**1. Build the data.** QM9 (~430 MB, not shipped) is fetched, verified by
SHA-256 and split:

```bash
python proj1/scripts/download_qm9.py
python proj1/scripts/prepare_qm9.py
```

One seeded permutation gives four disjoint splits: `train_a` and `train_b`
(51,527 each), `val` (17,748), `test` (13,083). The generator and the guide
`f_A` train on `train_a`; the evaluator `f_B` trains on `train_b`, so nothing
that scores a sample was trained on what steered it.

The DeepFlyBrain 500-bp split used for Modality 2 ships as
`code/m2/blade_bundle/dfb500.npz` (83,722 / 10,505 / 10,434
train / validation / test).

**2. Train**, or skip this and use the shipped checkpoints:

```bash
python proj1/scripts/train_fm.py         --split train_a   # ~11.5 h
python proj1/scripts/train_diffusion.py  --split train_a   # ~12 h
python proj1/scripts/train_predictor.py  --split train_a --prop mu   # f_A (also alpha, gap)
python proj1/scripts/train_predictor.py  --split train_b --prop mu   # f_B (also alpha, gap)
python proj1/m2/simplex_fm.py --dfb --crop 500 --hidden 128 --layers 10 \
    --batch 256 --lr 2e-3 --epochs 1500 --val-every-epochs 5 --patience 300 \
    --out proj1/m2/ckpt/fm_m2_dfb500.pt                    # the DNA generator
```

`--patience 300` is the recorded setting: it keeps early stopping from firing, so
the run reaches 1,500 epochs and validation selects the best epoch within it.

**3. Sample and score.** One cell is one (backend, property, arm set, seed).
Molecules, the headline stage and the ablation grid, per backend `fm`, `vp`,
`equifm`, `edm`, property and seed:

```bash
V3_ARMS=unguided,plug,tmpd,lgd_mc,tfg,bdg_e4t0.5,bdg_e4t1
ABL_ARMS=bdg_e0t1$(for e in 1 2 4 8; do for t in 0.5 0.75 1 1.5; do printf ',bdg_e%st%s' $e $t; done; done)
python proj1/scripts/transfer_sweep.py --stage v3    --backend fm --props mu \
    --arms $V3_ARMS  --n 2000 --batch 500 --seed 20261001 --per-mol
python proj1/scripts/transfer_sweep.py --stage v3abl --backend fm --props mu \
    --arms $ABL_ARMS --n 2000 --batch 500 --seed 20261001 --per-mol
```

`code/cluster/v3_run.slurm` is the exact job that tiled every cell, with the
seed list, the arm sets and the checkpoint pins. `transfer_sweep.py --help` lists
every flag.

DNA, the registered `t >= 0.5` window at both strengths, then the `t >= 0.3`
follow-up one cell at a time (arms `plug`, `tmpd`, `lgd_mc`, `tfg_mc`,
`unguided`, and `bdg` with variants `e4t0.5`, `e4t1`, `e0t1`; seeds 20260921-23):

```bash
python proj1/m2/run_sweep.py --stage m2 --w 1
python proj1/m2/run_sweep.py --stage m2 --w 4
python proj1/m2/m2_sweep.py --prop cpg --arm bdg --variant e4t0.5 --w 4 \
    --t-min 0.3 --stage m2win --seed 20260921
```

**4. Regenerate every table and figure.** The paper's tables and Figure 2 are
written into `main.tex` by one script, which reads the result cells directly:

```bash
python paper/tools/build_results.py           # rebuild table blocks and figures
python paper/tools/build_results.py --check   # read-only: every block and source hash
```

The raw cell tree (`results/`, 619 MB) is not in this folder, so this step runs
in the full repository. The markdown pages in `results_and_docs/results/` come
from the scripts named at the top of each page (for example `v3_table.py`,
`v3_blade_readout.py`, `m2_v3_results.py`, `m2_window_compare.py`).

**5. Build the paper:**

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex && bibtex main
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
python tools/pagecheck.py     # five-page limit, overfull boxes, dangling refs
python tools/check.py         # rubric structure, scope, house style
```

Both checks must pass before the paper is shared. `check.py` enforces things easy
to break by accident: every required section present, four or five contribution
bullets, every table and figure referenced from the text, no q90 number anywhere
(this study reports q50 only), and no phrase our own adversarial reviews refuted.

---

## The weights

`weights/README.md` is the weight protocol: what each checkpoint is, how it was
selected, and its hash. In short:

| File | What | Notes |
|---|---|---|
| `weights/fm_ema.pt` | our flow-matching generator | EGNN 256x8, 3,753,229 params, epoch 1500, EMA 0.9999, `train_a`, seed 20260918 |
| `weights/diff_ema.pt` | our VP diffusion generator | same backbone, params, budget and split; validation selected epoch 1475 |
| `weights/f_A_{mu,alpha,gap}.pt` | the guides that steer | trained on `train_a` |
| `weights/f_B_{mu,alpha,gap}.pt` | the evaluators that score | trained on `train_b`, disjoint from the guides |
| `code/m2/blade_bundle/fm_m2_dfb500.pt` | our DNA simplex generator | dilated CNN 128x10, 1,500 epochs, validation loss 0.0634, seed 20260921 |
| `code/m2/enhancer_gate.pt` | our enhancer-vs-Markov discriminator | a Modality 2 diagnostic, not a headline metric |
| `weights/deepflybrain/DeepFlyBrain.json` | the published DeepFlyBrain architecture | weights are re-fetched, see below |

**Accept a molecular generator checkpoint only if its recorded `args` show the
right family, split `train_a`, hidden 256, layers 8 and epoch 1500.** An earlier
`train_ab` checkpoint violated the split, because a generator trained on
`train_ab` would have seen the evaluator's training data.

Third-party weights are re-fetched rather than redistributed:
**EDMsecond** by `code/scripts/fetch_tfg_assets.py` (md5 in
`weights/tfg_manifest.json`), **EquiFM** by `code/scripts/fetch_equifm_assets.py`,
and **DeepFlyBrain.hdf5** from zenodo.org/record/5153337 (md5 in
`code/m2/deepflybrain.py`).

---

## Connection to the paper

No number in the paper is typed by hand. Each is emitted by a script that reads
recorded cells:

| Paper object | Produced by | Reads |
|---|---|---|
| Table 1, base models | `paper/tools/build_results.py` | `results/v3/{fm,vp,equifm,edm}/v3/n2000/` |
| Table 2, guided vs unguided | `build_results.py` | the same headline cells, FM and VP |
| Table 3, dipole ablation | `build_results.py` | `results/v3/fm/v3abl/n2000/` |
| Table 4, recent methods | `build_results.py` | the FM headline cells |
| Table 5, DNA at `t >= 0.3` | `build_results.py` | `results/m2/m2win/n2000/` |
| Figure 2 | `build_results.py` | the ablation grid and the DNA cells |
| Tables S3 to S20 | `build_results.py` | as above, plus `results/m2/m2/` (`t >= 0.5`) and `results/m2/m2tau/` (setpoint probe) |
| Figure 1, the overview | `paper/figs/overview.tex` | nothing; it is a diagram |
| BDG itself | `code/src/guidance.py`, the `mode == "bdg"` branch | |
| Its gates | `code/tests/test_bdg.py` | eta = 0 is bit-identical to plug-in; the factorisation holds on the real field |

`build_results.py --check` is the audit that closes the loop: it recomputes every
generated table block from the cells and verifies each source by hash against
`paper/results_manifest_v10.json` in the repository. At the v10 build it passed:
23 tables, 1,073 source hashes. Appendix A.9 of the paper is the implementation
and evidence map.

---

## What is deliberately not here

| | Why |
|---|---|
| `data/` (430 MB) | rebuilt by `download_qm9.py` and `prepare_qm9.py`, pinned by SHA-256 |
| the raw cell tree (619 MB) | one JSON per cell; the script-generated pages in `results_and_docs/results/` carry every number the paper uses |
| `weights/EDMsecond/`, EquiFM, `DeepFlyBrain.hdf5` | third parties' released weights; re-fetched, not redistributed |
| the aborted 489-epoch DNA checkpoint | superseded by the 1,500-epoch one; shipping it invites loading the wrong file |
| vendored reference repositories | external code we read but did not modify |
| build artefacts and caches | regenerated by the commands above |

Absolute cluster paths have been replaced by `$PROJECT_ROOT`, which the SLURM
scripts already accept, so no personal or shared-filesystem path ships here.

---

## Where the rubric is answered

| Rubric item | Where in `paper/main.pdf` |
|---|---|
| Abstract | Abstract |
| 1 Introduction, contributions | §1, four bullets |
| 2 Related Work | §2: continuous generation, dispersion and guidance, molecules, the measured comparators, sequences |
| 3.1 Modalities, data, splits, conditioning | §3.1, Appendix A.1, Figure S1 |
| 3.2 Flow-matching baseline | §3.2, Eq. 1, Table S1 |
| 3.3 Diffusion baseline | §3.3, Eq. 2, Appendix A.3 |
| 3.4 Guided generation | §3.4, Eq. 3 |
| 3.5 The innovation | §3.5, Eq. 4, Algorithm 1 (Appendix A.8) |
| 3.6 Transfer to Modality 2 | §3.6, Appendices A.3 and A.7 |
| 3.7 Model overview figure | §3.7, Figure 1 |
| 4.1 Evaluation protocol, metrics, budget, statistics | §4.1, Appendix A.4, Table S8 |
| 4.2 Flow matching vs diffusion | §4.2, Table 1 |
| 4.3 Guided vs unguided | §4.3, Table 2, Tables S4 to S7 |
| 4.4 Innovation ablations | §4.4, Table 3, Figure 2a, Appendix A.6 |
| 4.5 Comparison with recent methods | §4.5, Table 4, Tables S3 to S6 |
| 4.6 Modality 2 transfer | §4.6, Table 5, Figure 2b, Tables S16 to S20 |
| 4.7 Cross-modality analysis and failure modes | §4.7, Appendix A.7 |
| 5 Discussion, limitations, next steps | §5 |
| Code: README and setup | this file |
| Code: organisation and comments | `code/`, and `code/tests/` for the gates |
| Code: connection to the paper | the table above, and `build_results.py --check` |

---

## Known limitations, stated plainly

The paper states these; they are repeated here so nothing is discovered only by
reading closely.

- **The molecular coverage gains are not resolved.** All six contracting-setpoint
  contrasts are positive and none clears the pre-registered bar.
- **Three of five planned ablation controls did not run**: a one-sided residual,
  an open-loop replay and a signed-strength plug-in. Without the replay, the grid
  shows that the deviation coefficient moves coverage, but cannot show that the
  *feedback* is what does it.
- **BDG loses on CpG at the registered setpoint.** The sign rule explains the
  loss, and the `tau_m = 0.25` probe that reverses it is exploratory, chosen
  after the registered result was seen.
- **The headline DNA window is a follow-up, not the registered one.** At the
  registered `t >= 0.5`, three of the four comparators give identical coverage.
  Both windows are reported and never pooled.
- **On DNA, the guidance window matters more than the rule.** Moving from
  `t >= 0.5` to `t >= 0.3` raises plug-in's coverage from 14.1% to 30.3% on GC and
  from 51.6% to 91.5% on CpG, more than any rule-to-rule difference in either
  window.
- **No external sequence generator was run.** The four DNA comparators are
  ports of the molecular guidance rules onto our own frozen simplex model.
  Dirichlet FM, Fisher Flow and MOG-DFM are cited, not reproduced.
- **The borrowed EquiFM sampler is not bitwise reproducible.** Same-seed reruns
  differ by up to 0.007 in continuous coverage, so its `eta = 0` control matches
  plug-in only to that tolerance. Our FM reproduces exactly.
- **Useful yield is not reported.** Coverage counts invalid molecules; the joint
  in-band-and-stable yield is listed as future work rather than inferred from the
  marginals.
