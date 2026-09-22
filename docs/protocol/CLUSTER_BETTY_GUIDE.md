# PARCC Betty Cluster — Complete Guide (CS2 Win-Probability Project)

**Consolidated from:** `e:/CLAUDE_CONTEXT.md` (§3–§8, §13–§15), `docs/cluster_runbook.md`,
`docs/BETTY_benchmark_guide.md`, `docs/BETTY_defuse_guide.md`, the `jobs/*.sh` sbatch scripts, and
the project memory notes (`parcc-betty-cluster`, `tcn-first-result-betty`).
**Last consolidated:** 2026-09-17.

This is the single place to look for *how to use Betty* for this project: login, rules, storage,
environment, Slurm, data transfer, job scripts, utilities, troubleshooting, and what has already
been run there.

---

## Table of contents

1. [What Betty is and what we have](#1-what-betty-is-and-what-we-have)
2. [Login and authentication](#2-login-and-authentication)
3. [Hard rules (account-suspension risk)](#3-hard-rules-account-suspension-risk)
4. [Storage layout](#4-storage-layout)
5. [Software environment (conda)](#5-software-environment-conda)
6. [Slurm — partitions, CLI filter, commands, templates](#6-slurm--partitions-cli-filter-commands-templates)
7. [Data transfer](#7-data-transfer)
8. [Project job scripts in this repo](#8-project-job-scripts-in-this-repo)
9. [Standard workflow (smoke → full → retrieve)](#9-standard-workflow-smoke--full--retrieve)
10. [Benchmark / holdout / defuse runs (what to run, what to send back)](#10-benchmark--holdout--defuse-runs)
11. [PARCC utility tools](#11-parcc-utility-tools)
12. [Troubleshooting](#12-troubleshooting)
13. [What has already been run on Betty](#13-what-has-already-been-run-on-betty)
14. [Publication acknowledgment](#14-publication-acknowledgment)
15. [Contacts and links](#15-contacts-and-links)
16. [Quick-reference cheatsheet](#16-quick-reference-cheatsheet)

---

## 1. What Betty is and what we have

**Betty** is the Penn Advanced Research Computing Center (PARCC) flagship HPC cluster: three login
nodes (`login01`, `login02`, `login03`), Globus data-transfer nodes, and Slurm-managed compute
nodes (AMD EPYC Genoa CPU nodes and NVIDIA B200 GPU nodes). It is shared by hundreds of users.

**Our allocation** (via Prof. Abraham Wyner / Wharton):

| Item | Value |
|---|---|
| Project storage | **1 TB** at `/vast/projects/ajw/wharton` |
| Symlink | `~/projects` → `/vast/projects/ajw/wharton` |
| Project directory | `/vast/projects/ajw/wharton/cs2-rwp/` (== `~/projects/cs2-rwp/`) |
| CPU concurrency | up to 128 cores |
| GPU concurrency | up to 4 GPUs |
| Wharton software | `source /vast/projects/wharton/software/bin/wharton_parcc.sh` (in `~/.bashrc`) |
| PennKey used on Betty | `hyhuang` (from `docs/BETTY_defuse_guide.md`) |

The cluster is the project's **production compute**: everything GPU-gated (TCN / Transformer /
GAT / deep bootstrap) runs here; classical models and feature building run locally on the laptop.

---

## 2. Login and authentication

### 2.1 Prerequisites
- Be on **PennNet or the Penn VPN** (https://vpn.upenn.edu/).
- Duo two-step enrolled: https://isc.upenn.edu/pennkey/two-step-verification-enrollment-instructions
- Betty uses **two-of-three factor auth**: Kerberos ticket (`kinit`) + SSH key + Duo. Any two must
  succeed.

### 2.2 Connect (every session)
```bash
# On the LOCAL laptop:
conda deactivate                              # local conda breaks kinit — deactivate first
kinit <PennKey>@UPENN.EDU                     # realm MUST be uppercase; ticket lasts 10 h; Duo prompt
ssh <PennKey>@login.betty.parcc.upenn.edu
hostname                                      # note login01/02/03 — tmux sessions live on ONE node
```
If you use `tmux`, you must SSH back to the **same** login node to reattach
(`ssh <PennKey>@login02.betty.parcc.upenn.edu`, etc.).

### 2.3 Local `~/.ssh/config`
```
Host *.parcc.upenn.edu
    VerifyHostKeyDNS yes
    GSSAPIAuthentication yes
    ControlMaster auto
    ControlPath ~/.ssh/control:%h:%p:%r
```
`ControlMaster` reuses one authenticated connection, so subsequent `ssh`/`scp` calls don't re-prompt.

### 2.4 Browser access — Open OnDemand
https://ood.betty.parcc.upenn.edu/ — file browser, shell, and job submission from the browser (no
`kinit` required). Useful for quick log inspection.

### 2.5 Once logged in (every session)
```bash
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
cd /vast/projects/ajw/wharton/cs2-rwp
```
(These `module`/`source` lines can live in `~/.bashrc` — see §5.)

---

## 3. Hard rules (account-suspension risk)

> These are PARCC terms of use. Violating them can suspend the account and revoke queue access.

| # | Rule | Do instead |
|---|---|---|
| 1 | **Never run compute on a login node** — no `python train.py`, no demo parsing, no notebooks, no MATLAB, no `./binary` | `sbatch job.sh` or `srun ... --pty bash` |
| 2 | **Never rsync / bulk-copy / extract `.rar` on a login node** | Globus for bulk; `scp` only for files < 1 GB; extract via a Slurm job |
| 3 | **Never share credentials** (PennKey, SSH keys) | — |
| 4 | **All compute goes through Slurm**; never SSH straight into a compute node | `sbatch` / `srun` |
| 5 | **Clean up `/tmp/$USER`** on compute nodes before the job exits | `cp -r /tmp/$USER/out ~/projects/cs2-rwp/data/ && rm -r /tmp/$USER` |
| 6 | **Home quota is 50 GB** — code and conda only, never data. A full home dir = cannot log in | All data → `/vast/projects/ajw/wharton/cs2-rwp/` |
| 7 | **`pip install --user` is banned** | Use the conda env (`uv pip install` inside it) |
| 8 | Long bootstrap loops must **checkpoint every 10 iterations** | see §6.6 |

**What IS allowed on login nodes:** `sbatch`, `squeue`, `scancel`, `scontrol`, `sacct`; editing
(`vim`/`nano`); `git`; `module load`; `conda activate`; `scp` of small files; `parcc_*.py`
utilities; `kinit`; `mkdir`, `ln -s`, `ls`, `cat`, `tail -f`; light compilation (`make -j4`, not
`-j20`).

---

## 4. Storage layout

| Path | Quota | Purpose |
|---|---|---|
| `/vast/home/<letter>/<PennKey>` (`~`) | **50 GB** | Code, `.bashrc`, conda env (`~/envs/cs2-rwp`). **Never data.** |
| `/vast/projects/ajw/wharton` (`~/projects`) | **1 TB** | All project data — demos, parquet, models, checkpoints, logs, outputs |
| `/ceph/projects/...` | variable | Long-term inactive storage (not used in jobs) |
| `/tmp/$USER` | ~500 GB shared | Local scratch **on compute nodes only** — must clean up |

Set up the symlink once: `ln -s /vast/projects/ajw/wharton/ ~/projects`.

### Project space layout (`/vast/projects/ajw/wharton/cs2-rwp/`)
The repo is cloned **directly into this directory** (`git clone ... .`), so `src/`, `jobs/`,
`docs/` are the repo, and the gitignored data directories sit alongside:

```
cs2-rwp/
├── src/, jobs/, docs/, ...                 # the git repo (github.com/Henry1145141919810/CS2_Win_Probability_Model)
├── data/
│   ├── training_dataset.parquet            # 2024–25 training table (~81 MB; scp'd from laptop)
│   ├── trajectory_dataset.parquet          # per-player trajectories for GAT (~68 MB)
│   ├── test_dataset_2026_lag2025.parquet   # 2026 holdout, lagged-2025 firepower
│   ├── test_dataset_2026_sameyr.parquet    # 2026 holdout, same-year firepower (leaky best case)
│   ├── training_dataset_defuse.parquet     # re-parsed tables with defuse-progress columns
│   ├── test_dataset_2026_defuse.parquet
│   ├── parquet/{ticks,kills,rounds,bomb,grenades}/   # (only if parsing on cluster)
│   └── holdout2026/parquet/ticks/          # isolated 2026 tick channels
├── demos/{raw,extracted}/                  # .rar via Globus → .dem via Slurm extraction
├── features/{economy,mapcontrol,firepower,tactical,combined}/
├── models/                                 # .pkl / .pt
├── checkpoints/                            # tcn.pt, tcn_smoke.pt, bootstrap .pkl (every 10 iters)
├── outputs/                                # oof_*.parquet, holdout_*.parquet, figures/
└── logs/                                   # Slurm logs: <jobname>_<jobid>.out / .err
```
Create the non-repo dirs once: `mkdir -p logs checkpoints data outputs`.

---

## 5. Software environment (conda)

### `~/.bashrc`
```bash
### WHARTON RESOURCES
source /vast/projects/wharton/software/bin/wharton_parcc.sh

### MODULES
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
```

### The env: `cs2-rwp`
| | |
|---|---|
| Location | `$HOME/envs/cs2-rwp` (prefix env, activated by path) |
| Python | 3.11 |
| CUDA | **12.8 — always the `cu128` PyTorch index** (only version Betty's B200s support) |
| Activate | `conda activate "$HOME/envs/cs2-rwp"` |

### Build it (ONE time, on a COMPUTE node — never the login node)
Preferred: `sbatch jobs/setup_env.sh` (genoa-std-mem, 4 CPU, 16 G, 40 min), then
`tail -f logs/cs2-setup-env_<JOBID>.out` until it prints `ENV READY`. What it runs:

```bash
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$HOME/envs/cs2-rwp" python=3.11 uv -c conda-forge
conda activate "$HOME/envs/cs2-rwp"
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # Betty B200 = cu128
uv pip install polars numpy scikit-learn pyarrow
python -c "import torch; print('torch', torch.__version__)"
```
The deep models only need torch/polars/numpy/scikit-learn/pyarrow. The **full research stack**
(only needed if you also parse demos / build features / run classical models on the cluster):
```bash
uv pip install torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install awpy polars pandas numpy scipy scikit-learn xgboost lightgbm catboost \
               torch-geometric shap pyarrow matplotlib seaborn
```
Alternative to `sbatch`: do it inside an interactive session
(`srun --partition=genoa-std-mem --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash`).

To add a package later: activate the env and `uv pip install <pkg>` — never `pip install --user`.

---

## 6. Slurm — partitions, CLI filter, commands, templates

### 6.1 Partitions

| Partition | Hardware | Account limit | Use for |
|---|---|---|---|
| `dgx-b200` | Full NVIDIA B200 (8/node, 180 GB VRAM) | 32 GPUs | Full deep runs, bootstrap, large sweeps |
| `b200-mig90` | MIG slice, 90 GB VRAM | 8 MIGs | Medium GPU tasks |
| `b200-mig45` | MIG slice, 45 GB VRAM | 8 MIGs | **Default for our deep jobs** (small models), smoke tests, debugging |
| `genoa-std-mem` | AMD EPYC Genoa, standard RAM | 640 CPUs | Demo parsing, XGBoost, features, env build, extraction |
| `genoa-lrg-mem` (`genoa-large-mem`) | Genoa, high RAM | 128 CPUs | Jobs needing > 256 GB RAM |

Our models are tiny (TCN ~0.3 s/epoch on a B200) — `b200-mig45` is enough for nearly everything.

### 6.2 CLI-filter rule (since 2026-08-03) — **read before writing any GPU job**
`sbatch` now **rejects** jobs with the wrong CPU:GPU ratio (`CPUS_PER_GPU_MISMATCH`) or the wrong
memory (`MEM_PER_CPU_MISMATCH`). Docs: https://docs.parcc.upenn.edu/docs/notes/filter.html

| Partition | Required `--cpus-per-task` per GPU | `--mem` per GPU (= CPUs × 8 G) | VRAM |
|---|---|---|---|
| `dgx-b200` | **28** | **224G** | 180 GB |
| `b200-mig90` | **14** | **112G** | 90 GB |
| `b200-mig45` | **6** | **48G** | 45 GB |

CPU partitions: `genoa-std-mem` 5632 MB/CPU, `genoa-lrg-mem` 15872 MB/CPU.

- `--mem` must be **exactly** CPUs × 8 G on GPU partitions (on mig45: `--mem=48G`; 32G/16G are rejected).
- A valid mig45 GPU job is therefore `--gpus=1 --cpus-per-task=6 --mem=48G`. All repo job scripts
  on mig45 were fixed to this on 2026-08-31.
- Interactive `srun` has been observed to slip through with 4 CPU / 32 G, but `sbatch` does not.
  Use the correct values in both.
- **`jobs/tcn_train.sh` still uses `dgx-b200` with `--cpus-per-task=8 --mem=64G` and would be
  rejected under the filter** — change it to `--cpus-per-task=28 --mem=224G`, or move it to
  `b200-mig45` with 6 / 48G (the model is small enough).

### 6.3 Core commands
```bash
sbatch job.sh                       # submit batch job
srun [opts] cmd                     # run one-off / interactive
squeue -u $USER                     # your jobs (PD pending, R running, CG completing, CD done, F failed, TO timeout)
scancel <JOBID>                     # cancel one
scancel -u $USER                    # cancel ALL yours
scontrol show job <JOBID>           # why is it pending? details
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS     # post-run stats
tail -f logs/<jobname>_<JOBID>.out  # live log
```

### 6.4 GPU job template (matches the repo's scripts)
```bash
#!/bin/bash
#SBATCH --job-name=cs2-<name>
#SBATCH --output=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.out
#SBATCH --error=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6          # mig45 = 6 ; mig90 = 14 ; dgx-b200 = 28
#SBATCH --mem=48G                  # = CPUs x 8G  (mig90 112G ; dgx-b200 224G)
#SBATCH --time=01:00:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"

PROJ=/vast/projects/ajw/wharton/cs2-rwp
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"
nvidia-smi

python src/models/deep/tcn.py --data "$PROJ/data/training_dataset.parquet" ...
```

### 6.5 CPU job template
```bash
#!/bin/bash
#SBATCH --job-name=cs2-parse-demos
#SBATCH --output=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.out
#SBATCH --error=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
echo "Running on: $(hostname) at $(date)"
python ~/projects/cs2-rwp/scripts/parse_demos_batch.py \
    --demo_dir /vast/projects/ajw/wharton/cs2-rwp/demos/extracted/ \
    --output_dir /vast/projects/ajw/wharton/cs2-rwp/data/parquet/
```

### 6.6 Interactive sessions (debugging only, not production)
```bash
# CPU
srun --partition=genoa-std-mem --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
# GPU (mig45, filter-compliant)
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=00:20:00 --pty bash
# then inside:
module load anaconda3 && source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp" && cd /vast/projects/ajw/wharton/cs2-rwp
```
GPU sanity check (30 s): `srun -p b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G -t 00:02:00 nvidia-smi`

### 6.7 Mandatory checkpointing for long loops
Bootstrap scripts must save every 10 iterations to `checkpoints/` (durable project space) and
resume from it:
```python
checkpoint_file = "/vast/projects/ajw/wharton/cs2-rwp/checkpoints/MODEL_bootstrap.pkl"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "rb") as f:
        state = pickle.load(f)
    completed_aucs = state["aucs"]; start_b = len(completed_aucs)
...
if b % 10 == 0:
    with open(checkpoint_file, "wb") as f:
        pickle.dump({"aucs": completed_aucs}, f)
```
(The TCN/Transformer/GAT scripts already checkpoint their best epoch.)

---

## 7. Data transfer

| Size / kind | Method |
|---|---|
| Bulk (HLTV `.rar` demos, many GB) | **Globus** — required |
| Single files < 1 GB (parquet tables, logs, checkpoints) | `scp` — fine |
| Extracting `.rar` | Slurm job on `genoa-std-mem` — never on login node |

### 7.1 Globus
1. https://globus.org → log in with Penn credentials (search "University of Pennsylvania").
2. Find the **PARCC VAST** collection.
3. Navigate to `/vast/projects/ajw/wharton/cs2-rwp/demos/raw/`.
4. Transfer from the source endpoint (laptop via Globus Connect Personal, or Wharton HPC3).
5. Email arrives when complete.

### 7.2 scp (from the local laptop; repo root)
```bash
# up
scp data/training_dataset.parquet \
    <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/data/
scp data/test_dataset_2026_lag2025.parquet data/test_dataset_2026_sameyr.parquet \
    <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/data/
# down
scp <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/logs/cs2-tcn_<JOBID>.out .
scp <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/outputs/holdout_tcn_lag2025.parquet outputs/
```
From WSL the repo is at `/mnt/e/CS2_Win_Prob_Model`; from Git Bash it is `/e/CS2_Win_Prob_Model`.
Data files are gitignored, so **every rebuilt parquet must be re-scp'd** — a stale table on Betty
has already caused one crash (see §12).

### 7.3 Extracting `.rar` via Slurm
```bash
srun --partition=genoa-std-mem --cpus-per-task=2 --mem=8G --time=02:00:00 \
     bash -c "cd /vast/projects/ajw/wharton/cs2-rwp/demos/raw && for f in *.rar; do unrar e \"\$f\" ../extracted/; done"
```

### 7.4 Code sync
Code goes through **git**, not scp:
```bash
cd /vast/projects/ajw/wharton/cs2-rwp
git pull                                   # or: git fetch origin && git checkout <branch>
```

---

## 8. Project job scripts in this repo

All in `jobs/`. Every script: `set -euo pipefail`, loads anaconda3, activates `cs2-rwp`, `cd`s to
`$PROJ`, prints host/date (+ `nvidia-smi`), logs to `logs/<jobname>_<jobid>.out`.

| Script | Job name | Partition / resources | Time | What it does |
|---|---|---|---|---|
| `setup_env.sh` | `cs2-setup-env` | genoa-std-mem, 4 CPU, 16G | 40 min | One-time conda env build (§5) |
| `tcn_smoke.sh` | `cs2-tcn-smoke` | mig45, 1 GPU, 6 CPU, 48G | 30 min | 20 matches × 3 epochs; validates pipeline → `checkpoints/tcn_smoke.pt` |
| `tcn_train.sh` | `cs2-tcn` | **dgx-b200, 8 CPU, 64G — pre-filter, needs 28/224G** | 4 h | Single-split TCN, 30 epochs → `checkpoints/tcn.pt` |
| `tcn_cv.sh` | `cs2-tcn-cv` | mig45, 6 CPU, 48G | 30 min | 5-fold GroupKFold OOF, full metrics + CIs → `outputs/oof_tcn.parquet` |
| `tcn_sweep.sh` | `cs2-tcn-sweep` | mig45, 6 CPU, 48G | 1 h | 8 configs (dropout/hidden/lr/seq-len), 5-fold each; read with `grep -E "CONFIG\|OOF" logs/cs2-tcn-sweep_*.out` |
| `tcn_seeds.sh` | `cs2-tcn-seeds` | mig45, 6 CPU, 48G | 1 h | Seeds 0–4, 5-fold each → mean ± std (deep-model CI) |
| `tcn_holdout.sh` | `cs2-tcn-holdout` | mig45, 6 CPU, 48G | 30 min | Train on 2024–25, eval on `test_dataset_2026_lag2025` → `outputs/holdout_tcn_lag2025.parquet` |
| `tcn_holdout_sameyr.sh` | `cs2-tcn-holdout-sy` | mig45, 6 CPU, 48G | 30 min | Same, on `test_dataset_2026_sameyr` |
| `transformer_cv.sh` | `cs2-tf-cv` | mig45, 6 CPU, 48G | 1 h | Causal Transformer 5-fold OOF → `outputs/oof_transformer.parquet` |
| `transformer_holdout.sh` | `cs2-tf-holdout` | mig45, 6 CPU, 48G | 1 h | Transformer out-of-time on lagged holdout |
| `transformer_holdout_sameyr.sh` | `cs2-tf-holdout-sy` | mig45, 6 CPU, 48G | — | Transformer out-of-time on same-year holdout |
| `gat_smoke.sh` | `cs2-gat-smoke` | mig45, 6 CPU, 48G | 15 min | GAT 20 matches × 5 epochs |
| `gat_cv.sh` | `cs2-gat-cv` | mig45, 6 CPU, 48G | 1 h | GAT 5-fold OOF on `data/trajectory_dataset.parquet` |

Model entry points: `src/models/deep/tcn.py`, `src/models/deep/transformer.py`,
`src/models/deep/gat.py`. Key flags: `--data`, `--holdout`, `--cv`, `--epochs`, `--patience`,
`--limit-matches`, `--seed`, `--dropout`, `--hidden`/`--dim`, `--lr`, `--seq-len`, `--checkpoint`,
`--save-oof`. The ensemble (`src/models/ensemble_oof.py`) runs **locally** on the saved OOF files.

**Adding a new deep job:** copy `jobs/tcn_cv.sh`, change job name / python line, keep the mig45
`6 / 48G` header.

---

## 9. Standard workflow (smoke → full → retrieve)

```bash
# 0. local: get a ticket and connect
conda deactivate && kinit <PennKey>@UPENN.EDU && ssh <PennKey>@login.betty.parcc.upenn.edu

# 1. on Betty: sync code, check resources
cd /vast/projects/ajw/wharton/cs2-rwp && git pull
parcc_quota.py && parcc_sfree.py && parcc_sqos.py

# 2. local (other terminal): scp any rebuilt parquet tables up  (§7.2)

# 3. GPU sanity (optional, 30 s)
srun -p b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G -t 00:02:00 nvidia-smi

# 4. SMOKE TEST FIRST — always
sbatch jobs/tcn_smoke.sh
squeue -u $USER                                  # PD -> R -> CD
tail -f logs/cs2-tcn-smoke_<JOBID>.out           # expect: device=cuda (NVIDIA B200 ...), val_AUC lines, checkpoint saved

# 5. Full run(s)
sbatch jobs/tcn_cv.sh ; sbatch jobs/transformer_cv.sh ; sbatch jobs/gat_cv.sh

# 6. Read results
grep -E "OOF|OUT-OF-TIME|baseline|CI" logs/cs2-*_<JOBID>.out

# 7. Retrieve small files to local (§7.2): logs, outputs/*.parquet, checkpoints/*.pt
```
Interactive smoke alternative (when testing an untested code path such as `--holdout`):
```bash
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=00:20:00 --pty bash
module load anaconda3 && source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp" && cd /vast/projects/ajw/wharton/cs2-rwp
python src/models/deep/tcn.py --data data/training_dataset.parquet \
    --holdout data/test_dataset_2026_lag2025.parquet --epochs 3 --limit-matches 20
exit
```

---

## 10. Benchmark / holdout / defuse runs

### 10.1 Out-of-time (2026 lagged-prior holdout) — `docs/BETTY_benchmark_guide.md`
Question: do the sequence models degrade out-of-time more than the classical models, or do they
also recover once firepower uses lagged-2025 stats?

1. `git pull`; scp `data/test_dataset_2026_lag2025.parquet` up; confirm `training_dataset.parquet`
   on Betty is the **current** (firepower-v2) table.
2. Interactive smoke with `--holdout ... --epochs 3 --limit-matches 20` → expect
   `TCN OUT-OF-TIME [test_dataset_2026_lag2025]  AUC 0.8x ...`.
3. `sbatch jobs/tcn_holdout.sh` and `sbatch jobs/transformer_holdout.sh` (plus the `_sameyr`
   variants for the leaky best case).
4. Report back per model: the `OUT-OF-TIME` line (AUC, log-loss, Brier, ECE, BSS, cAUC), the B=500
   bootstrap CI block, and the `outputs/holdout_*_lag2025.parquet` prediction files.
5. Optional: refresh in-time OOF with `tcn_cv.sh` / `transformer_cv.sh`.

GAT out-of-time is **deferred**: it needs a 2026 trajectory dataset (assembly pass over
`data/holdout2026/parquet/ticks`), and GAT is the weakest in-time model.

### 10.2 Defuse-progress feature — `docs/BETTY_defuse_guide.md`
The re-parse adds 4 columns (`defuse_in_progress, defuse_elapsed_sec, defuse_progress_frac,
defuse_beats_fuse`). Deep models consume every column, so only the data changes.

1. scp `data/training_dataset_defuse.parquet` and `data/test_dataset_2026_defuse.parquet` up;
   check out the branch with the defuse code.
2. Smoke as above with the `_defuse` tables; feature-column count should be **+4** (the
   schema-mismatch guard prints differing columns otherwise).
3. Edit `tcn_holdout.sh` / `transformer_holdout.sh`: `--data` → `_defuse` table, `--holdout` →
   `test_dataset_2026_defuse.parquet`, `--save-oof outputs/holdout_{tcn,transformer}_defuse.parquet`
   (do not overwrite the earlier predictions). Submit.
4. Report the `OUT-OF-TIME` line + CI **and the per-second curve on defusing rows** (snippet in the
   defuse guide: filter `defuse_in_progress == 1`, log-loss/Brier, and actual-vs-predicted by
   `defuse_progress_frac` bucket).
5. Optional isolation: build `_nodefuse` copies of both tables (drop the 4 columns) and run the same
   job twice to measure the deep marginal effect.

---

## 11. PARCC utility tools

Available on the login nodes (safe to run there):
```bash
parcc_quota.py                              # storage quotas (home 50 GB / project 1 TB) — run before big jobs/transfers
parcc_du.py /vast/projects/ajw/wharton      # space usage by subdirectory
parcc_sfree.py                              # free GPUs/CPUs/nodes per partition
parcc_sqos.py                               # QOS limits for the account — check before sbatch
parcc_sreport.py --user <PennKey>           # recent job usage report
parcc_sdebug.py --job <JOBID>               # debug a failed / stuck job
```

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `kinit` fails | Local conda still active → `conda deactivate`; realm must be `@UPENN.EDU` (uppercase); be on PennNet/VPN |
| `ssh` asks for password / Duo repeatedly | Ticket expired (10 h) → `kinit` again; check `~/.ssh/config` has `GSSAPIAuthentication yes` |
| Cannot log in at all | Home dir may be full (50 GB) — use OOD or ask support; move data to project space |
| `sbatch` rejected: `CPUS_PER_GPU_MISMATCH` | Wrong CPUs per GPU → mig45 = 6, mig90 = 14, dgx-b200 = 28 |
| `sbatch` rejected: `MEM_PER_CPU_MISMATCH` | `--mem` must equal CPUs × 8G → mig45 `48G`, mig90 `112G`, dgx-b200 `224G` |
| Job stuck `PD` | `scontrol show job <ID>` for the reason; `parcc_sfree.py`; use `b200-mig45` instead of `dgx-b200` |
| `CUDA not available` in log | Wrong partition (no `--gpus=1`) or wrong env (torch not cu128) |
| `ModuleNotFoundError` | Activate env and `uv pip install <pkg>`; never `pip install --user` |
| OOM | Lower `--batch` (e.g. 32) or `--seq-len` |
| Deep job crashes on column/shape mismatch | Betty's `training_dataset.parquet` is **stale** vs the local rebuild (happened 2026-07: firepower-v1 table on Betty) → re-scp the current table; the schema guard prints which columns differ |
| tmux session "missing" | You're on a different login node → `ssh <PennKey>@login0X.betty.parcc.upenn.edu` |
| `--holdout` path never GPU-tested locally | The laptop has no CUDA (RTX 4060 not used for these runs) — always run the 3-epoch smoke first |

---

## 13. What has already been run on Betty

(From `docs/results_checkpoint.md`, `docs/notes_lagged_holdout.md`, memory notes.)

- **2026-06-25 — first deep run succeeded end-to-end** (env via `setup_env.sh`, scp upload,
  sbatch, B200 training ~0.3 s/epoch, checkpoint). Single-split TCN val AUC 0.8429 (overfit).
- **TCN 5-fold OOF:** AUC 0.8489 · ECE 0.0118 · cAUC 0.572 (classical logreg EFB2: 0.8515 / 0.016 / 0.596).
- **TCN sweep** (`tcn_sweep.sh`): best dropout 0.5 / hidden 48 / lr 1e-3 / **seq-len 160** → OOF
  0.8493. seq-len is critical (160 → 0.849, 100 → 0.826, 64 → 0.780).
- **TCN multi-seed** (`tcn_seeds.sh`): 0.8485 ± 0.0004.
- **GAT 5-fold OOF** (`gat_cv.sh`, trajectory dataset 68 MB): 0.8465 — weakest.
- **Transformer 5-fold OOF** (`transformer_cv.sh`): 0.8473, ECE 0.009.
- Ensemble (local, on saved OOF): 4-model soft-vote 0.8531 — best overall.
- **2026-07-29 — out-of-time deep runs done** (`tcn_holdout.sh`, `transformer_holdout.sh`) on the
  lagged-2025 holdout after re-syncing the current training table.
- **2026-08-31 — all mig45 job scripts and both Betty guides updated** for the CLI-filter rule
  (6 CPU / 48G).

Verdict so far: deep models tie the calibrated classical model on ~220 matches, win on
calibration, lose on contested rounds; the lever is more data.

---

## 14. Publication acknowledgment

PARCC terms of use require that every publication using Betty:
- **acknowledges the PARCC allocation**, and
- is **uploaded to the ColdFront portal** (URL: ask Prof. Wyner), and
- uses the resources only for the research described in the allocation.

Suggested acknowledgment text:
> Computational resources were provided by the Penn Advanced Research Computing Center (PARCC)
> under allocation [allocation ID — check ColdFront].

(Paper v12 already includes a PARCC acknowledgment; `docs/roadmap.md` tracks the item.)

---

## 15. Contacts and links

| Resource | URL / contact |
|---|---|
| PARCC docs — CLI filter | https://docs.parcc.upenn.edu/docs/notes/filter.html |
| PARCC training | https://parcc.upenn.edu/training |
| PARCC support | https://parcc.upenn.edu/support |
| Wharton Research Computing | research-computing@wharton.upenn.edu (Slurm script conversion help: email in a **separate thread**) |
| Penn VPN | https://vpn.upenn.edu/ |
| Open OnDemand | https://ood.betty.parcc.upenn.edu/ |
| Globus | https://globus.org (search "University of Pennsylvania") |
| ColdFront portal | ask Prof. Wyner |
| Duo / two-step | https://isc.upenn.edu/pennkey/two-step-verification-enrollment-instructions |
| Repo | https://github.com/Henry1145141919810/CS2_Win_Probability_Model |

---

## 16. Quick-reference cheatsheet

```
# LOGIN (local)
conda deactivate
kinit <PennKey>@UPENN.EDU                      # uppercase realm, 10 h
ssh <PennKey>@login.betty.parcc.upenn.edu
hostname                                       # login0X for tmux

# SESSION (Betty)
module load anaconda3 && source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
cd /vast/projects/ajw/wharton/cs2-rwp && git pull

# STORAGE
~            50 GB  code + conda only
~/projects   1 TB   = /vast/projects/ajw/wharton  (all data / logs / checkpoints / outputs)

# GPU JOB HEADER (mig45)   --gpus=1 --cpus-per-task=6 --mem=48G
#           (mig90)        --gpus=1 --cpus-per-task=14 --mem=112G
#           (dgx-b200)     --gpus=1 --cpus-per-task=28 --mem=224G

# RUN
sbatch jobs/tcn_smoke.sh        # smoke first, always
sbatch jobs/tcn_cv.sh
squeue -u $USER ; tail -f logs/<job>_<ID>.out ; scancel <ID>

# CHECK
parcc_quota.py ; parcc_sfree.py ; parcc_sqos.py

# TRANSFER (local)
scp data/<table>.parquet <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/data/
# bulk demos → Globus only

# NEVER
python anything.py on a login node | rsync / unrar on a login node | pip install --user | data in ~
```
