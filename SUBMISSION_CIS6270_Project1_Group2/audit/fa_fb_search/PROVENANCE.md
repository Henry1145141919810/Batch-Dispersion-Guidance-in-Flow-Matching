# Vendored third-party code and checkpoints

**None of the code or weights in `TFG/` and `OC-Flow/` is ours.** They are
committed to this repository unmodified so that the transfer experiment runs
from a fresh clone. Everything else under `audit/` remains untracked.

## Why these two subtrees are committed when the rest of `audit/` is not

`proj1/src/external/tfg_assets.py` loads model definitions and checkpoints from
them at import time. Until 22 September 2026 `audit/` was ignored wholesale,
which meant the transfer harness — `transfer_sweep.py`,
`benchmark_transfer_base.py` and all 61 gates in `test_transfer_backend.py` —
ran only on the machine it was written on. A teammate cloning the repository
got an immediate failure inside `definitions()`.

## What is here

### `TFG/` — Training-Free Guidance

| | |
|---|---|
| paper | Ye, Lin, Yang, Zhang, Chen, Ermon (NeurIPS 2024), *TFG: Unified Training-Free Guidance for Diffusion Models*, [arXiv:2409.15761](https://arxiv.org/abs/2409.15761) |
| repository | <https://github.com/YWolfeee/Training-Free-Guidance> |
| commit | `f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77` |
| licence | **MIT** (stated in their README; `LICENSE.md` was not part of the partial clone taken here) |

Used by this project:

| path | role |
|---|---|
| `tasks/networks/egnn/EGNN.py` | `EGNN_dynamics_QM9`, the EDM generator backbone |
| `tasks/networks/egnn/energy.py` | the guide's `EGNN`, and the `polynomial_2` noise schedule our `edm_schedule.py` ports |
| `tf_predict_{mu,alpha,gap}/` | the time-dependent property networks used as `f_A` |
| `evaluate_{mu,alpha,gap}/` | the independent property classifiers used as `f_B` |

The checkpoints are TFG's released weights, downloaded from the Google Drive
folder their README links. They were **not retrained, fine-tuned or altered**.

`EDMsecond`, the base generator, is *not* committed — it is fetched by
`proj1/scripts/fetch_tfg_assets.py` into `weights/EDMsecond/`.

### `OC-Flow` — two property-prediction source files

| | |
|---|---|
| paper | Wang, Xue, Shi et al. (2025), *Guided Flow Matching with Optimal Control* |
| repository | <https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control> |
| commit | `ca218ba616f7e3a58a05924fdb547bd579b3c900` |
| licence | **not stated** in the files vendored here — see the note below |

Used by this project: `molecule/qm9/property_prediction/models_property.py` and
`models/gcl.py`, which define the `EGNN` classifier that TFG's `evaluate_*`
checkpoints load into.

**Licence note, stated rather than assumed.** OC-Flow's repository does not
carry a licence in the portion cloned here. These two files are, however,
byte-identical (md5 `4d98e2a3…` for `models_property.py`) to the same files in
**EDM**, `github.com/ehoogeboom/e3_diffusion_for_molecules`, which is **MIT**
(Hoogeboom, Satorras, Vignac, Welling, ICML 2022). The real upstream is
therefore EDM, and the licence chain runs through it. We vendor them from the
OC-Flow path only because that is the path our loader was written against; a
future tidy-up should repoint it at `e3_diffusion_for_molecules/` and drop the
`OC-Flow/` copy.

## What is deliberately NOT committed

| | why |
|---|---|
| `TFG/**/logs.txt` | a 79 MB training log, three quarters of the subtree, read by nothing |
| `TFG/*.html` | scraped Google Drive listings, kept locally as a record of how the file ids were obtained |
| every other repository under `audit/` | TFG-Flow, PropMolFlow, MolGuidance, e3_diffusion_for_molecules, `guidance_baselines_2026/` — consulted for the prior-art audit, not imported by any code in `proj1/` |

## Verifying what you have

```bash
python proj1/scripts/fetch_tfg_assets.py --verify
```

reports an md5 for every vendored checkpoint and writes
`weights/tfg_manifest.json`. `audit/fa_fb_search/TFG/downloaded_weight_hashes.json`
holds SHA-256 for the two files that were hashed at download time.

## If these need to be removed

A takedown or a licence objection is handled by deleting the two exemptions in
`.gitignore` and the files from the working tree. The harness then needs the
manual setup path: clone both repositories at the commits above and download
TFG's checkpoints from their Drive link. **Note that removing them from
`.gitignore` does not remove them from git history** — that needs a history
rewrite, which is a decision for the repository owner.
