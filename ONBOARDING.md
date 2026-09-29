# Getting this repo running

**Written for:** Bobo, Idea and Haimo, on a machine that has never seen this
project. Follow it top to bottom; every step says what it should print, so you
can tell a success from a silent failure without knowing the codebase.

Total: about **35 minutes**, most of it one download and one CPU-bound parse.
Nothing here needs a GPU.

---

## 0. What you are setting up

A frozen generative model of 3D molecules, and code that steers it toward a
target property **at sampling time, without retraining it**. Three things have
to exist before anything runs:

| | what | how you get it | size |
|---|---|---|---|
| **code** | this repository | `git clone` | 30 MB |
| **QM9 data** | the molecules | step 3 — download + parse | 430 MB |
| **weights** | our FM and VP generators and predictors | already in `weights/` | 40.5 MB |

The borrowed TFG checkpoints used by the transfer experiment are **committed**
(`audit/fa_fb_search/`, see its `PROVENANCE.md`), so you do not fetch those.
Only TFG's base model is a download, and only if you work on that experiment.

---

## 1. Clone and make an environment

```bash
git clone https://github.com/Henry1145141919810/cis6270-project1-group2.git
cd cis6270-project1-group2

python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\activate

pip install torch rdkit
pip install gdown          # only if you will do step 6
```

Python **3.10+**. `torch` is the only heavy dependency; `rdkit` is needed for
validity and SMILES; `gdown` fetches one checkpoint from Google Drive and is
needed only by the transfer experiment. If you will run the sweeps on a GPU, install the CUDA
build of torch from <https://pytorch.org> rather than the default wheel.

Check it took:

```bash
python -c "import torch, rdkit; print(torch.__version__, torch.cuda.is_available())"
```

---

## 2. Confirm the repo is complete

```bash
python proj1/scripts/fetch_tfg_assets.py --verify
```

Expect, as the first line:

```
vendored property networks: 12/12 present  (tracked in git; see audit/fa_fb_search/PROVENANCE.md)
```

**If it says fewer than 12/12, stop.** Your clone is incomplete — those files
are committed, so something went wrong in the clone rather than in your setup.
Try `git checkout -- audit/`.

The `NOT READY` line about `weights/EDMsecond` at the end is expected; that is
step 6 and only matters for the transfer experiment.

---

## 3. Get the data (~25 min, mostly waiting)

```bash
python proj1/scripts/download_qm9.py      # ~45 MB download, then extracts
python proj1/scripts/prepare_qm9.py       # ~10 min, CPU bound
```

The first prints `ok` for each file and ends with `READY`. The second ends
with the split sizes:

```
split sizes: train_a=51527  train_b=51527  val=17748  test=13083
```

**Those exact numbers matter.** If yours differ, you have a different QM9 and
none of your results will be comparable to anyone else's. Check with:

```bash
python proj1/scripts/download_qm9.py --verify
```

which should report `qm9.pt ... ok -- identical to the file the published
numbers used`.

> **Why the fuss.** "QM9" names several distributions that are not
> interchangeable. We use the DeepChem/MoleculeNet packaging and keep all
> 133,885 molecules; the Cormorant/EDM download removes 3,054 of them and
> applies a different split. Both parse without error. The hashes are the only
> thing that catches it.

---

## 4. Check the install with the test suites

```bash
python proj1/tests/test_arms_exact.py
python proj1/tests/test_v2_arms.py
python proj1/tests/test_transfer_backend.py
```

Each ends with `ALL PASS`. The third prints `ALL PASS (61 gates)` and a note
that generator gates were skipped — expected until step 6.

These are **closed-form gates**: each one is a quantity with an exact analytic
value, so a wrong sign or a dropped factor moves a number. If one fails, do
not work around it — say so in the group chat. They have caught real bugs
repeatedly, including three that shipped.

---

## 5. Run something small

One guidance cell per arm, ~15 s, writes nothing:

```bash
python proj1/scripts/guidance_sweep.py --stage v2 --props mu --preflight
```

Expect a header naming the checkpoint it chose, then one line per arm:

```
generator: fm_ema.pt  [weights]  family=flow  epoch=1500
  mu     f_A 0.08971  f_B 0.08399  delta 0.16799
  [  1/ 14] mu     unguided  ...  mae=1.2167
```

`[weights]` says the published checkpoint was used. On a machine that has
pulled from Betty it will say `[betty_pull]` instead — same model, newer file.
**If two people get different numbers, check this label first.**

One line is *expected* to say `PREFLIGHT SKIP rch`: the residual-calibration
heads are not published, so that arm has nothing to load. Everything else
should run. A nonzero exit means a real failure.

Run this before any real sweep — it catches a missing checkpoint or a wiring
error in a minute instead of after a four-hour queue window.

The real screening sweep is resumable, one JSON per cell, and skips work
already on disk:

```bash
python proj1/scripts/guidance_sweep.py --props mu,alpha,gap --n 512 --steps 100
python proj1/scripts/select_arms.py --stage main        # read the result
```

---

## 6. Only if you are on the transfer experiment

This re-runs every guidance arm on a **borrowed** base model, guide and oracle,
so nothing in it but the guidance field is ours.

```bash
pip install gdown                                 # if you have not already
python proj1/scripts/fetch_tfg_assets.py          # TFG's EDMsecond, 21 MB
python proj1/scripts/fetch_equifm_assets.py       # EquiFM + OC-Flow's second oracle, 30 MB
python proj1/tests/test_transfer_backend.py --require-edm

# THE HARD GATE -- do not skip
python proj1/scripts/benchmark_transfer_base.py --edm-dir weights/EDMsecond \
    --n 2000 --steps 100 --grid gamma \
    --out results/bench/edmsecond_gate.json

python proj1/scripts/transfer_sweep.py --preflight
```

The gate runs the borrowed model through **our own evaluator** and exits
nonzero if it scores below atom 0.95 / molecule 0.60. **If it fails, do not run
guidance cells** — every guided number would inherit the fault while looking
plausible. The script's docstring gives the order to investigate.

Read [docs/protocol/TRANSFER_EXPERIMENT_PLAN.md](docs/protocol/TRANSFER_EXPERIMENT_PLAN.md)
before quoting anything from it. It is a pre-registration: the decision rules
are fixed in §4.3 before results are seen, and §5 lists caveats that must
travel with every number.

---

## 7. Where things live

```
proj1/src/          models, guidance fields, samplers, evaluation
proj1/scripts/      training, sweeps, selection, benchmarking
proj1/tests/        closed-form gates -- run them, they are fast
proj1/cluster/      SLURM scripts for PARCC Betty
weights/            inference-ready checkpoints
results/sweep/      one JSON per experiment cell -- THE experimental record
audit/              vendored third-party code; see fa_fb_search/PROVENANCE.md
```

Start reading here, in this order:

| if you want… | read |
|---|---|
| what is done, running, missing | [docs/status/SCOPE_FM_GUIDANCE_STATUS.md](docs/status/SCOPE_FM_GUIDANCE_STATUS.md) |
| every guidance method, ours vs prior art | [docs/methods/GUIDANCE_METHODS_INDEX.md](docs/methods/GUIDANCE_METHODS_INDEX.md) |
| how the base model compares to published work | [docs/results/BASE_MODEL_BENCHMARK.md](docs/results/BASE_MODEL_BENCHMARK.md) |
| how the data is split and why | [docs/protocol/SPLIT_PROTOCOL.md](docs/protocol/SPLIT_PROTOCOL.md) |

The two status documents are **living documents** — update them, do not write
new ones.

---

## 8. Conventions worth knowing before you change anything

- **`results/sweep/*.json` is the experimental record.** One file per cell.
  Never edit one by hand; re-run the cell.
- **Every guidance arm needs a closed-form gate** in `proj1/tests/`. A new arm
  without one will not be trusted, and rightly.
- **A gate that passes vacuously is worse than no gate.** Two were found doing
  it: one compared against `<= 0`, which a zero gradient satisfies; one
  short-circuited on an empty set. If you add a gate, also check it *fails*
  when you deliberately break the thing it guards.
- **Deviations get disclosed, not hidden.** The repo records its own defects,
  including numbers that were wrong and are now fixed. Keep that up.

---

## 9. When something does not work

| symptom | cause |
|---|---|
| `ModuleNotFoundError: torch` | venv not activated |
| `data/qm9.pt is missing` | step 3 |
| `missing data/gdb9.sdf` | step 3, first command |
| split sizes differ from §3 | wrong QM9 — run `download_qm9.py --verify` |
| `vendored property networks: 0/12` | incomplete clone — `git checkout -- audit/` |
| `missing weights/EDMsecond/...` | step 6, first command |
| `FileNotFoundError: betty_pull/fm_last.pt` | you are on a build from before 22 Sep 2026 — `git pull` |
| two people get different sweep numbers | compare the `[weights]` / `[betty_pull]` label in the header |
| a gate fails | **do not work around it** — report it |
| CUDA out of memory | lower `--batch` (128 → 32); it does not change results, only how many samples are in flight |
