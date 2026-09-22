# Property-targeted generation with inference-time guidance

Group 2 — Bobo Li · Henry Huang · Idea Idehpour · Haimo Fang

Steer a **frozen** continuous generative model to hit a target property value at
sampling time, without retraining it. Two modalities: **QM9 3D molecules**
(this repo's current content) and **DNA sequences on the probability simplex**
(not started).

## Start here

| if you want… | read |
|---|---|
| **what is done, what is running, what is missing** | [docs/status/SCOPE_FM_GUIDANCE_STATUS.md](docs/status/SCOPE_FM_GUIDANCE_STATUS.md) |
| **every guidance method, ours vs prior art, with its proof doc** | [docs/methods/GUIDANCE_METHODS_INDEX.md](docs/methods/GUIDANCE_METHODS_INDEX.md) |
| the timetable and who owns what | [docs/status/PLAN_AND_TIMETABLE.md](docs/status/PLAN_AND_TIMETABLE.md) |
| how the data is split and why | [docs/protocol/SPLIT_PROTOCOL.md](docs/protocol/SPLIT_PROTOCOL.md) |
| how the base model compares to published work | [docs/results/BASE_MODEL_BENCHMARK.md](docs/results/BASE_MODEL_BENCHMARK.md) |

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

```bash
python -m venv .venv && ./.venv/Scripts/activate      # or source .venv/bin/activate
pip install torch rdkit

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

## Status in one line

The base generator is trained and benchmarked; nine guidance arms are measured
on three properties; the three new arms built to fix why our method loses have
**not** been measured yet. See the status doc.
