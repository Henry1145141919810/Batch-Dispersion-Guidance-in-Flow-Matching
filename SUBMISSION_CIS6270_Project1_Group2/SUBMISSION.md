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
`audit/` is third-party; only its load-bearing part, `audit/fa_fb_search/`, is
in the archive (§7).

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

**Borrowed checkpoints** (§7). Both scripts download from the upstream release
and check every file against a recorded hash (SHA-256 for EquiFM, md5 for TFG):

```bash
python proj1/scripts/fetch_equifm_assets.py   # EquiFM + OC-Flow's property nets, 30 MB -- needed for ANY v3 sampling
python proj1/scripts/fetch_tfg_assets.py      # TFG's EDMsecond, 21 MB -- needed for --backend edm
```

The first one is not optional even for our own generators: every v3 cell, on
every backend, is also scored by a second, independently trained oracle
(OC-Flow's property EGNN), and those weights come from that download.

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
python proj1/tests/test_transfer_backend.py   # ALL PASS (74 gates; 61 before fetch_tfg_assets.py)
```

`test_transfer_backend.py` skips its 13 EDMsecond gates, and says so, until
`fetch_tfg_assets.py` has run; the other two need nothing beyond QM9.

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

**Cross-check.** This recomputes every number the paper's tables print from the
cells and reports any that disagree (currently 28 of 28 match):

```bash
python proj1/scripts/paper_fill_v3.py --md-out docs/results/PAPER_TABLE_FILL_V3.md --latex-check
```

It is the reason the tables and the data cannot drift apart silently.

---

## 7. External code and models we did not write

Origin, commit and licence for TFG and OC-Flow:
[audit/fa_fb_search/PROVENANCE.md](audit/fa_fb_search/PROVENANCE.md). For
EquiFM: [docs/protocol/EQUIFM_USABILITY_AUDIT.md](docs/protocol/EQUIFM_USABILITY_AUDIT.md).

**The load-bearing part of `audit/` IS in the submission folder.**
`audit/fa_fb_search/` (TFG and OC-Flow, vendored unmodified, 29.9 MB) ships
because our code imports it: `proj1/src/external/tfg_assets.py` loads TFG's model
definitions from `audit/fa_fb_search/TFG/`, and the borrowed `f_A`/`f_B` property
networks that score the `equifm` and `edm` backends are checkpoints inside it.

Those checkpoints look like third-party weights we should link rather than ship,
and an earlier build excluded them on that reasoning. That was wrong, and
`fetch_tfg_assets.py` says why in its first paragraph: the *generators* are
missing from this repository, the *property networks* are not — they are
committed here, and the script **checks them rather than re-downloading**. Drop
them and nothing brings them back. The rest of `audit/` (reference clones
including `TFG-Flow`, the BDG review's evidence) stays in the git repository
only.

**What the archive does not contain is the third-party releases** that
upstream publishes itself. Two scripts fetch them and check each file against
a recorded hash (EquiFM: SHA-256 at the commit the usability audit pinned;
TFG: md5 of the download every transfer cell was produced with):

```bash
python proj1/scripts/fetch_equifm_assets.py   # EquiFM + OC-Flow property nets -> audit/equifm_20260922/  (ALL v3 sampling)
python proj1/scripts/fetch_tfg_assets.py      # TFG's EDMsecond                -> weights/EDMsecond/      (--backend edm)
```

**The first is needed even for our own backends (`fm`, `vp`).**
`transfer_sweep.py` scores every v3 cell on every backend with a second,
independently trained oracle (OC-Flow's property EGNN, `oracle2` in each cell),
so that no generator column gets a second opinion another lacks. Without the
fetch, `--preflight` stops at `FileNotFoundError` on
`.../exp_class_mu/args.pickle`. Modality 2, the tests (see §4.4) and every table
rebuild need no download beyond QM9.

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

## 8. The submission folder

`SUBMISSION_CIS6270_Project1_Group2/` is built by one script, so its contents are
the same every time, the exclusions are written down rather than improvised, and
the folder is **run before it is handed in**:

```bash
python proj1/scripts/build_submission.py               # build, then verify by running it
python proj1/scripts/build_submission.py --zip         # also write the .zip
python proj1/scripts/build_submission.py --no-verify   # build only
```

**1,711 files, 100.3 MB**, against a working repository of ~6,900 files and 2.6 GB.

**The folder keeps the repository's layout.** `proj1/`, `docs/`, `results/`,
`paper/`, `weights/` land under exactly those names, so every command in this
file and in `README.md` runs inside the folder verbatim. That is not cosmetic:
each script resolves its imports as `ROOT/proj1/src`. An earlier build renamed
`proj1/` to `code/` and `docs/` to `results_and_docs/` because it read more
nicely, and in that folder `python code/scripts/transfer_sweep.py` died with
`ModuleNotFoundError: No module named 'evaluation'` — every documented command
was broken, and it shipped that way. Do not rename them.

| in | size | why |
|---|---|---|
| `proj1/` — `src`, `scripts`, `tests`, `m2`, `cluster` | 19.1 MB | the code, including the Modality-2 generator bundle |
| `weights/` | 40.5 MB | FM and VP generators and the six `f_A`/`f_B` predictors, so the folder can **sample**, not only rebuild tables. md5 of each in `weights/README.md` |
| `results/v3/`, `results/m2/`, `results/bdg_port/` | 4.8 MB | the measured cells, JSON. **Every paper table and three of the four figures regenerate from the folder alone**, with no GPU and no download |
| `docs/` — `protocol`, `results`, `methods`, `status` | 3.7 MB | the pre-registered protocols, the generated result pages, `DATA_INDEX.md` |
| `paper/` | 1.8 MB | the manuscript, `main.pdf`, its figures, and the tooling that builds them |
| `audit/fa_fb_search/` | 29.9 MB | external code we import, `PROVENANCE.md`, and the six **vendored** TFG property networks — 28 MB of `.npy` that no fetch script restores (§7) |
| `blade_runs/`, `course/` | 0.4 MB | the off-cluster drivers; the assignment and paper template |
| `README.md`, `SUBMISSION.md`, `requirements.txt` | — | the two entry points and the pinned environment |

| out | why |
|---|---|
| `data/` (430 MB) | QM9 itself; rebuilt by `prepare_qm9.py`, seeded and reproducible |
| `*.permol.pt` sidecars (500 MB) | per-molecule samples. No paper table or figure reads one — `build_results.py` works from the seed-level cell JSONs. See limit 2 |
| third-party **released** weights: `weights/EDMsecond/`, `audit/equifm_20260922/`, `DeepFlyBrain.hdf5` | ours to link, not to redistribute. Fetched and hash-checked by script (§7). TFG's property networks are a different case and do ship — see the row above |
| `audit/fa_fb_search/TFG-Flow/` | a reference clone nothing in `proj1/` imports; its two checkpoint zips alone are 40 MB |
| `audit/.../TFG/tf_predict_mu/logs.txt` (79 MB) | a third party's training log. A 2 MB per-file cap on the vendored tree drops it; it was half the weight of the first build |
| versioned drafts: `figs/*_v9.pdf`, `results_manifest_v*.json`, `paper/versions/`, `paper/tmp/` | superseded. The folder shows one version of the work, not its history |
| `betty_pull/`, `bundles/`, `archive/`, `scratch_schnet/`, `slides/deck_versions/`, the rest of `audit/` | cluster syncs and superseded material; in the git repository |
| `logs/`, caches, `__pycache__`, LaTeX build artefacts | regenerated. A `.pyc` also records the absolute path it was compiled from |

Absolute cluster paths, `@upenn.edu` addresses, login names and the build
machine's home directory are rewritten on the way in — 1,185 substitutions
across 1,055 files in the current build. Institutional hostnames
(`parcc.upenn.edu`, `blade.seas.upenn.edu`) are left readable on purpose: they
say which machines produced the numbers.

### The build refuses to ship a folder that does not work

It ends by running the folder it just made. Every one of these failed on the
folder that was built before this was added:

| check | what it catches |
|---|---|
| 13 required files present | a folder missing something a graded requirement needs: either entry point, `requirements.txt`, `transfer_sweep.py`, `build_results.py`, `DATA_INDEX.md`, both generators |
| `proj1/scripts/transfer_sweep.py --help` | imports that do not resolve in the folder — the failure described above |
| `proj1/scripts/v3_sanity.py` | cells absent or inconsistent. It now **fails** on an empty tree; it used to print "0 fail" and exit 0, which is how a folder containing no cells at all was certified sound |
| `proj1/tests/test_bdg.py` | the innovation's 25 gates |
| a scan for the account name | a scrub rule silently dropped. It caught 81 files after one rewrite of this script |

Current state: all pass — 15,243 cell checks, 0 fail; all BDG gates pass. And,
run inside the folder, `python paper/tools/build_results.py` rebuilds 23 table
blocks and 3 vector figures and leaves `paper/main.tex` **byte-identical**. The
tables in the PDF are the tables the shipped cells produce.

**Weights are in by default.** The checkpoints shipped are the inference copies
(EMA weights only). They are byte-identical in every parameter to the training
checkpoints that produced the results, and every result cell records the md5 of
the generator that made it. Resuming *training* needs the full checkpoints, with
optimiser state, which are on the cluster.

### Known limits of the folder

Stated rather than discovered:

1. **QM9 sampling needs one download, two for `edm`.** `fetch_equifm_assets.py`
   (30 MB) for every v3 backend including our own `fm` and `vp`, because each
   cell is also scored by OC-Flow's oracle; plus `fetch_tfg_assets.py` (21 MB)
   for `edm` (§7). Everything the code *imports* is in the folder. Modality 2,
   the tests and every table rebuild need no download beyond QM9.
2. **The paired and useful-yield readouts cannot be recomputed here.**
   `paper_fill_v3.py`, `v3_blade_readout.py` and `v3_paired.py` read the
   `.permol.pt` sidecars, which are 500 MB and stay in the repository. The pages
   they produced ship instead, under `docs/results/`. Nothing in the paper's
   tables or figures depends on them.
3. **The property-distribution figure needs QM9.** `paper/tools/distributions.py`
   reads `data/qm9.pt`; the other three figures come from the cells. The values
   it drew are recorded in `paper/property_distributions.json`, which ships.
4. **Unzip to a short path on Windows.** The longest path in the folder is 147
   characters including its root; below a ~110-character parent directory it
   stays under Windows' 260-character limit.
5. **Five Modality-2 helper scripts load from a hard-coded cluster path** at
   import: `proj1/m2/{calib_k,diag_strength,gate_arms,make_vf_table,nfe_check}.py`
   raise `FileNotFoundError` off Betty. They are diagnostics, not on any results
   path; `m2_sweep.py`, `m2_table.py` and `m2_v3_results.py` are the entry points
   that matter, and they are portable.
6. **`prepare_qm9.py` takes no arguments and has no `--help`** — running it with
   any argument starts a full dataset rebuild. It is seeded and deterministic, so
   a stray rebuild is harmless, but it is not a no-op.
