# CIS 6270 Project 1 — Group 2 — code submission

**Bobo Li · Henry Huang · Idea Idehpour · Haimo Fang**

Property-targeted generation with inference-time guidance.
**Modality 1**: 3D molecular coordinates (QM9). **Modality 2**: probability-simplex
DNA sequences (DeepFlyBrain enhancers).

This file is the entry point for the code review. It answers the three things the
assignment asks of the code, in order: **how to set up and run it** (§1–§4),
**how it is organised and where the innovation lives** (§5), and **which script
produced which number in the paper** (§6–§7). §8 defines the submission archive.

If you only read one other file, read
**[docs/results/DATA_INDEX.md](docs/results/DATA_INDEX.md)** — it names the
canonical run, the four base models, the base data, and every superseded tree
with what replaced it.

---

## 1. Repository structure

```
proj1/                  ALL OUR CODE
  src/                    library: models, samplers, guidance, evaluation
    models/egnn.py          E(n)-equivariant backbone, shared by FM and VP
    sampling.py             FlowSampler / VPSampler / EquiFMSampler + guidance hooks
    guidance.py             the guidance fields, including our innovation (BDG)
    diffusion.py            the VP noise schedule
    evaluation.py           stability, validity, uniqueness, in-band
    external/               loaders for borrowed checkpoints (not our models)
  scripts/                74 entry points: training, sweeps, tables, audits
  m2/                     Modality 2: its own data, sweep and tables
  tests/                  gate suites, run before any sweep
  cluster/                SLURM job files for Betty
blade_runs/             drivers for the off-cluster GPU box
results/                every committed result cell (JSON, one per experiment cell)
docs/
  protocol/               the pre-registered protocols
  results/                generated result pages + DATA_INDEX.md
  status/                 running project status
paper/                  manuscript source, figures, and the tooling that builds them
slides/                 defense deck source and its content spec
weights/                model checkpoints (see §8 — not all are in the archive)
course/                 the assignment and paper template we were given
data/                   QM9, built by §3. Not in the repo and not in the archive
logs/                   cluster job logs (not in the archive)

README.md               project overview and current status
SUBMISSION.md           this file — the entry point for the code review
requirements.txt        the pinned environment
ONBOARDING.md           how a new teammate gets running
```

**Our code is `proj1/`, `blade_runs/` and `paper/tools/`.** Everything under
`audit/` is third-party and is not in the archive (§7).

---

## 2. Environment and setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python **3.12.14**. The one line to get right is torch — [requirements.txt](requirements.txt)
explains the CUDA variants. Everything runs on CPU, but **CPU is ~26× slower**,
which turns a 10-minute sampling cell into four hours.

Versions are pinned to what produced the committed results. Every result cell
records its own torch/CUDA build and device in its `prov` block, so any number
can be checked against the environment that made it.

---

## 3. Dataset preparation

**Modality 1 — QM9.** Not in the archive (430 MB). Rebuild:

```bash
python proj1/scripts/download_qm9.py      # fetches the raw archive
python proj1/scripts/prepare_qm9.py       # builds data/qm9.pt with the seeded split
```

The split is seeded (**20260917**) and reproducible. Generators train on
`train_a` (51,527 molecules); the guide `f_A` trains on `train_a` and the oracle
`f_B` on `train_b` — **disjoint by construction**, which is what makes the
acceptance band honest. Full rules:
[docs/protocol/SPLIT_PROTOCOL.md](docs/protocol/SPLIT_PROTOCOL.md).

**Modality 2 — DeepFlyBrain enhancers.** `proj1/m2/deepflybrain.py` reads the
HDF5 source; needs `h5py`.

**Borrowed checkpoints** (external baselines, §7):

```bash
python proj1/scripts/fetch_tfg_assets.py      # TFG's EDMsecond, ~100 MB
python proj1/scripts/fetch_equifm_assets.py   # EquiFM
```

---

## 4. How to run

### 4.1 Training

| what | command |
|---|---|
| flow-matching generator | `python proj1/scripts/train_fm.py --epochs 1500 --batch 256 --hidden 256 --layers 8 --split train_a` |
| VP diffusion generator | `python proj1/scripts/train_diffusion.py --epochs 1500 --batch 256 --hidden 256 --layers 8 --split train_a` |
| guide / oracle predictors | `python proj1/scripts/train_predictor.py --prop mu --split train_a` (then `train_b`, and for `alpha`, `gap`) |
| Modality 2 generator | `proj1/m2/dfb_train.slurm` |

Both generators are **identical apart from the corruption and the regression
target** — same backbone, parameter count, epochs, batch, EMA and seed. That is
what makes the flow-vs-diffusion comparison in the paper a controlled one.
On a cluster use `proj1/cluster/train_{fm,diffusion}.slurm` (4-hour QOS cap, so
`--max-minutes 225` and `--resume` chain the links).

### 4.2 Sampling and evaluation

The sweep script does both: it samples under an arm and scores the samples.

```bash
# always first: one throwaway cell per arm, writes nothing, ~15 s
python proj1/scripts/transfer_sweep.py --stage v3 --backend fm \
    --props mu --arms unguided,plug --n 2000 --batch 500 --seed 20261001 --preflight

# one cell
python proj1/scripts/transfer_sweep.py --stage v3 --backend fm \
    --props mu --arms unguided,plug,tmpd,lgd_mc,tfg,bdg_e4t0.5,bdg_e4t1 \
    --n 2000 --batch 500 --seed 20261001 --per-mol
```

Resumable: one JSON per cell, completed cells are skipped. Whole runs:
`bash proj1/cluster/submit_v3.sh` (fm/equifm/edm) and
`bash proj1/cluster/submit_vp_bench.sh` (our VP diffusion).
Modality 2: `python proj1/m2/m2_sweep.py --prop gc --arm bdg --w 4` (`--arm` is
required; see [MODALITY2_V3_PROTOCOL.md](docs/protocol/MODALITY2_V3_PROTOCOL.md)).

### 4.3 Result pages and figures

```bash
python proj1/scripts/v3_sanity.py                 # integrity: must be 0 fail
python proj1/scripts/v3_table.py --both --stage v3 --n 2000 \
    --seeds 20261001,20261002,20261003 --md-out docs/results/V3_RESULTS.md
python proj1/scripts/v3_final_summary.py --md-out docs/results/V3_FINAL_SUMMARY.md
python paper/tools/build_results.py               # every paper table + figure
```

`DATA_INDEX.md` §6 has the full ordered regeneration block. Run it in order —
the later scripts cross-check the earlier ones and refuse on a mismatch.

### 4.4 Tests

```bash
python proj1/tests/test_v3.py                 # ALL PASS (106 gates)
python proj1/tests/test_bdg.py                # 25/25 — the innovation's gates
python proj1/tests/test_transfer_backend.py   # ALL PASS (74 gates)
```

These are **gates, not unit tests**: each asserts a property that, if it broke,
would produce plausible-looking wrong numbers rather than an error. Run them
before any sweep.

---

## 5. Organisation, configs, and where the innovation is

### The innovation — Batch-Dispersion Guidance (BDG)

| | |
|---|---|
| **implementation** | [`proj1/src/guidance.py`](proj1/src/guidance.py) — the field |
| **sampler wiring** | [`proj1/src/sampling.py`](proj1/src/sampling.py) — controller state, clipping |
| **gates** | [`proj1/tests/test_bdg.py`](proj1/tests/test_bdg.py) — 25 checks |
| **arms** | `bdg_e<eta>t<tau_mult>`, e.g. `bdg_e4t0.5`. `eta` = dispersion gain, `tau_mult` = setpoint multiplier |
| **ablation grid** | η ∈ {0,1,2,4,8} × τ_mult ∈ {0.5,0.75,1,1.5}, both strengths — [ABLATION_V3_PROTOCOL.md](docs/protocol/ABLATION_V3_PROTOCOL.md) |
| **the η = 0 control** | `bdg_e0t1` — BDG with the dispersion term switched off. This is the atomic control the ablation turns on |

The non-obvious parts carry comments where they are: why `eta` defaults to 0
rather than 1, why the batch is the estimator rather than a speed knob, and why
the clip is load-bearing (removing it produced 222 non-finite samples against 0).

### Experiment settings

There is **no YAML config layer** — settings are declared in code, in one place
per concern, and every result cell records the settings it ran under:

| setting | declared in |
|---|---|
| which backend runs which arms, with which property pair | `transfer_sweep.V3_BACKENDS` |
| which backends the cluster chain submits | `transfer_sweep.V3_CHAIN_BACKENDS` |
| seeds, target, strength, guidance window | `transfer_sweep.V3_SEEDS`, `V3_TARGET`, `V3_W`, `V3_T_START` |
| what a valid v3 cell must record | `v3_sanity.PINNED` |
| the ablation grid | `v3_final_summary.ABL_ARMS` |

A cell that disagrees with `PINNED` is rejected by `v3_sanity.py`, so a stray
setting cannot reach a table quietly.

---

## 6. Connection to the paper

**Every *data* table and figure in the paper is generated**, not hand-written:
`paper/tools/build_results.py` reads the committed cells and injects them into
`% BEGIN DATA` blocks in `main.tex`. Three provenance tables --
`tab:training`, `tab:hashes`, `tab:provenance` -- are hand-written records of
how the models were trained and which artifacts were evaluated.

| paper object | produced by | from |
|---|---|---|
| `tab:fmvd` (FM vs VP vs external) | `paper/tools/build_results.py` | `results/v3/{fm,vp,equifm,edm}/v3/n2000/` |
| `tab:ablation`, `fig:abl-fm`, `fig:abl-equifm` | `paper/tools/build_results.py` | `results/v3/{fm,equifm}/v3abl/n2000/` |
| `tab:recent`, `fig:results` | `paper/tools/build_results.py` | `results/v3/*/v3/n2000/` |
| `tab:m2`, `tab:full-m2` | `paper/tools/build_results.py` | `results/m2/` |
| `tab:full-{fm,equifm,edm,vp}`, `tab:grid-*`, `tab:contrasts` | `paper/tools/build_results.py` | the same cells, per backend |
| `tab:training`, `tab:hashes`, `tab:provenance` | **hand-written** in `main.tex` | checkpoint metadata + cell `prov`, cross-checked by `paper_fill_v3.py --latex-check` |
| `fig:overview` | TikZ, inline in `paper/main.tex` | — |

**Which script produced the experiments themselves**

| paper section | experiment | script |
|---|---|---|
| 4.2 FM vs diffusion | both generators, unguided, matched | `train_fm.py`, `train_diffusion.py` → `transfer_sweep.py --stage v3` |
| 4.3 Guidance | guided vs unguided, matched | `transfer_sweep.py --stage v3`, arms `unguided` / `plug` |
| 4.4 Innovation ablations | the η × τ_mult grid | `transfer_sweep.py --stage v3abl`, driven by `proj1/cluster/v3_run.slurm` |
| 4.5 Recent methods | TMPD, LGD-MC, TFG | the same sweep; external guides via `proj1/src/external/` |
| 4.6 Modality 2 transfer | BDG on the simplex | `proj1/m2/m2_sweep.py --prop <gc\|cpg> --arm <arm>` |

**Cross-check.** `proj1/scripts/paper_fill_v3.py --latex-check` recomputes every
number the paper prints from the cells and reports any that disagree. It is the
reason the tables and the data cannot drift apart silently.

---

## 7. External code and models we did not write

Origin, commit and licence for TFG and OC-Flow:
[audit/fa_fb_search/PROVENANCE.md](audit/fa_fb_search/PROVENANCE.md). For
EquiFM: [docs/protocol/EQUIFM_USABILITY_AUDIT.md](docs/protocol/EQUIFM_USABILITY_AUDIT.md).

⚠️ **`audit/` is third-party and is NOT in the archive, and that has a
consequence worth stating.** About 30 MB of it is load-bearing, not just
reference: `proj1/src/external/tfg_assets.py` loads TFG's model definitions from
`audit/fa_fb_search/TFG/`, and the borrowed `f_A`/`f_B` property networks live
there too. **So the `edm` and `equifm` backends do not run from the archive
alone**, and `test_transfer_backend.py` needs that directory. `fetch_tfg_assets.py`
re-downloads the EDMsecond *generator* only — its own docstring says the property
networks are expected to be committed. Every paper table still regenerates from
the archive, because those come from the committed cells, not from re-sampling.

| what | whose | how it is used |
|---|---|---|
| **EDMsecond** checkpoint (`edm` backend) | TFG release (Ye et al.) | a *borrowed* QM9 diffusion baseline, frozen. Loaded by `proj1/src/external/tfg_assets.py` |
| **EquiFM** checkpoint (`equifm` backend) | Song et al. | a *borrowed* flow-matching baseline, frozen |
| **TFG** property predictors | TFG release | the external guide/oracle pair used to score `equifm` |
| TFG / OC-Flow model definitions | their repos, vendored unmodified | loaded, never edited |
| E(3)-EDM noise schedule | Hoogeboom et al. | re-implemented in `proj1/src/external/edm_schedule.py` to sample their checkpoint under its own schedule |

⚠️ **`edm` and `vp` are both diffusion baselines and only `vp` is ours.** `edm`
is TFG's released EDMsecond. The paper labels every borrowed row as borrowed.

Our own models — the flow-matching generator, the VP diffusion generator, the
`f_A`/`f_B` predictors, the Modality 2 generator, and all guidance including BDG
— are trained by the scripts in §4.1 and implemented in `proj1/src/`.

---

## 8. The submission archive

Built by a script so the contents are the same every time and the exclusions are
written down rather than improvised:

```bash
python proj1/scripts/make_submission.py          # dry run: print the manifest
python proj1/scripts/make_submission.py --zip    # write cis6270_p1_group2_code.zip
```

**~1,600 files, ~23 MB**, against a working repository of 6,934 files and 2.6 GB.

| in | why |
|---|---|
| `proj1/`, `blade_runs/`, `paper/`, `docs/`, `slides/handoff/`, `course/` | the code, the protocols, the manuscript and its tooling |
| `results/v3/`, `results/m2/` — the cells (4.5 MB) | **every paper table regenerates from the archive alone**, with no GPU and no download |
| `requirements.txt`, `README.md`, this file | setup and entry points |

| out | why |
|---|---|
| `data/` (430 MB) | QM9 itself; rebuilt by `prepare_qm9.py`, seeded and reproducible |
| `betty_pull/` (582 MB), `bundles/` (396 MB) | cluster syncs and already-shipped tarballs |
| `audit/` (344 MB) | third-party code; fetched by script, provenance recorded |
| `archive/`, `scratch_schnet/`, `slides/deck_versions/` | superseded material and frozen snapshots |
| superseded result trees | listed in `DATA_INDEX.md` §5 with what replaced each |
| `weights/*.pt` (40 MB) | **the one judgement call** — see below |

**On weights.** They are excluded by default, so the archive reproduces every
*table* but not fresh *sampling*. `weights/README.md` documents each checkpoint
by md5, and every result cell records the md5 of the generator that produced it,
so any number can be traced to its checkpoint. If the submission limit allows
**40 MB** more, `--weights` adds all 8 checkpoints; say so in the submission note
if you do.

The script **refuses to build** an archive missing `SUBMISSION.md`,
`DATA_INDEX.md`, `requirements.txt`, `README.md`, `transfer_sweep.py` or
`build_results.py` — a submission that cannot rebuild its own tables is not
worth handing in.

### Known limits of the archive

Stated rather than discovered:

1. **The `edm` and `equifm` backends do not run from the archive alone** — they
   need `audit/fa_fb_search/`, which is third-party and excluded (§7). The `fm`
   and `vp` backends, and every table, do run.
2. **Five Modality-2 helper scripts load from a hard-coded cluster path** at
   import: `proj1/m2/{calib_k,diag_strength,gate_arms,make_vf_table,nfe_check}.py`
   raise `FileNotFoundError` off Betty. They are diagnostics, not on any results
   path; `m2_sweep.py`, `m2_table.py` and `m2_v3_results.py` are the entry
   points that matter and they are portable.
3. **`prepare_qm9.py` takes no arguments and has no `--help`** — running it with
   any argument starts a full dataset rebuild. It is seeded and deterministic,
   so a stray rebuild is harmless, but it is not a no-op.
