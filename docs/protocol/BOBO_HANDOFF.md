# Running the v3 benchmark from a fresh clone

**Written for: Bobo**, running protocol v3 on a cluster that is not Henry's
Betty tree. 26 September 2026.

Everything below was checked against this repository, not assumed. Where a
weight is *not* in the clone, this says so and says how to get it.

---

## 1. What you get in the clone, and what you must fetch

| what | where | in the clone? |
|---|---|---|
| **our FM generator** | `weights/fm_ema.pt` (14.4 MB) | ✅ **yes** |
| **our property pair** f_A, f_B × {mu, alpha, gap} | `weights/f_{A,B}_<p>.pt` (1.7 MB each) | ✅ **yes** |
| **TFG's property pair** (guide + oracle) | `audit/fa_fb_search/TFG/` (28 MB) | ✅ **yes**, deliberately tracked |
| **Modality 2 model + data** | `proj1/m2/blade_bundle/` (15 MB) | ✅ **yes** |
| **EquiFM generator + OC-Flow oracles** | `audit/equifm_20260922/` | ❌ fetch — see below |
| **QM9 diffusion generator** (TFG's EDMsecond) | `weights/EDMsecond/` | ❌ fetch — see below |
| **QM9 dataset** | `data/qm9.pt` (430 MB built) | ❌ build — ~35 min |

Three commands, all verified by hash, none needing cluster credentials:

```
python proj1/scripts/download_qm9.py && python proj1/scripts/prepare_qm9.py
python proj1/scripts/fetch_equifm_assets.py      # GitHub raw, sha256-checked
python proj1/scripts/fetch_tfg_assets.py         # Google Drive via gdown, md5-checked
```

`fetch_tfg_assets.py` needs `pip install gdown`. Both have a `--verify` mode
that re-checks hashes without downloading; the job's preflight calls it.

> **Needs outbound internet.** If the compute nodes have none, run the three
> fetches on a login node first — they only write files.

### The generator: you do NOT need `fm_last.pt`

The v3 job used to demand `proj1/checkpoints/fm_last.pt` and check its md5.
That file is 57.5 MB of weights **plus optimiser and scheduler state**, it is
git-ignored, and it exists only on Betty — so a clean clone failed in seconds.

`weights/fm_ema.pt` holds **the same EMA tensors** with the optimiser state
stripped (`max |w_slim − w_full| = 0.0` across every parameter), and records
`source_md5` naming its parent. The check is now on the **model**, not the
file:

```
python proj1/scripts/verify_generator.py
# -> generator OK: weights/fm_ema.pt (source_md5 matches the pin a190ac83)
```

It accepts either file and rejects anything else, so you cannot silently run a
different generator.

---

## 2. Point the job at your checkout

The job file defaults to Henry's `/vast/...` path. Override it — no editing:

```
export CGM_PROJ=/path/to/your/clone
export CGM_VENV=/path/to/your/venv        # only if not $CGM_PROJ/.venv
```

`submit_v3.sh` forwards both through `--export`. You will also want to change
the `#SBATCH --output=` / `--error=` lines in `proj1/cluster/v3_run.slurm`,
which still point at Henry's log directory, and `--partition` if yours is not
`b200-mig45`.

---

## 3. Choose n — this is the one real decision

**v3 pre-registers no cell size.** That is deliberate: it is yours to pick for
the cluster you are on. But **n sets the run's statistical power, not just its
cost**, so pick it from the power table, not the GPU-hour column:

```
python proj1/scripts/v3_power.py --md-out docs/results/V3_POWER.md
```

| n per cell | min. detectable in-band difference | headline GPU-h | ablation GPU-h |
|---|---|---|---|
| 500 | 3.13 pp | 3.8 | 6.2 |
| 1000 | 2.21 pp | 7.7 | 12.5 |
| **2000** | **1.56 pp** | **15.3** | **24.9** |
| 3000 | 1.28 pp | 23.0 | 37.4 |
| 5000 | 0.99 pp | 38.3 | 62.3 |

**A BDG-vs-plug effect is expected at about 1 pp**, and a null is only evidence
against BDG if the run could have seen it. If you cannot afford an n that
resolves 1 pp, that is fine — but the write-up has to report the minimum
detectable difference beside the null rather than saying "no difference".

GPU-hours are a **floor**: per-task fixed cost (checkpoint load, the
3000-molecule calibration) does not scale with n. Read the real rate off the
`[timing]` line after the first tasks land.

**n must be divisible by 500.** The batch is BDG's *estimator*, not a speed
knob, and a remainder batch is a second, far noisier controller pooled in as
an equal. Every entry point refuses an indivisible pair.

---

## 4. Run it

```
V3_N=2000 bash proj1/cluster/submit_v3.sh            # headline  (the default)
V3_N=2000 bash proj1/cluster/submit_v3.sh ablation   # the grid, afterwards
```

One paste, chained jobs, nothing to resubmit unless the table refuses. Submit
the **headline first**: its `[timing]` lines settle the real per-pass cost of
the non-`fm` backends — the one assumed number in the whole budget — before
the ablation, which is the larger stage, commits.

Use the **same `V3_N` for both**. Nothing enforces it across separate
submissions, and matching n is what makes the two stages comparable.

Watch it:

```
squeue -u $USER -o "%.18i %.14j %.9T %.10M %.28E"
grep -h '\[timing\]' logs/v3-*.out | tail -20
```

---

## 5. What the run produces, and the one trap in reading it

Cells land in `results/v3/<backend>/<stage>/n<N>/seed<S>/`, with `<stage>` one
of `v3` (headline) or `v3abl` (ablation). Tables:

```
python proj1/scripts/v3_table.py --both --stage v3    --n 2000
python proj1/scripts/v3_table.py --both --stage v3abl --n 2000
```

### ⚠️ Do not compare `in_band` between backends

Each base model is scored with a specific property pair (Henry's 26 Sep
decision):

| backend | generator | f_A / f_B | arms |
|---|---|---|---|
| `fm` | ours | **ours** | all 7 |
| `equifm` | EquiFM | **TFG's** | all 7 |
| `edm` | QM9 diffusion (EDMsecond) | **ours** | `unguided`, `plug` only |

δ = k × MAE(f_B), so **the pair sets the width of the acceptance band**, and
the two oracles are not equally accurate. Measured on the same 3000
calibration molecules ([V3_PAIR_DELTA.md](../results/V3_PAIR_DELTA.md)):

| property | δ ours | δ TFG | ours / TFG |
|---|---|---|---|
| mu | 0.136757 | 0.156754 | **0.87×** — ours narrower |
| alpha | 0.430463 | 0.164797 | **2.61×** — ours wider |
| gap | 0.005381 | 0.003736 | **1.44×** — ours wider |

So an ours-pair backend is scored in a different band, **narrower on mu and
wider on alpha and gap**, and its `in_band` moves for that reason alone —
before any base model or any arm is considered. Do not say "ours is the wider
band" without naming the property.

**Within one backend every arm shares one δ**, so the arm-against-arm
comparison is completely unaffected. That is the comparison this project is
about, and it is safe. The cross-backend one is not: `v3_table.py` refuses to
pool cells whose `pair` disagrees, and every cell records which pair scored it.

---

## 6. Things that are NOT ready, so you do not go looking

- **Modality 2 is not a v3 backend.** Its model and data are in the clone and
  its sweep is built, but it is a separate stack with its own driver, and four
  defects must be fixed before any run — see
  [MODALITY2_V3_PLAN.md](MODALITY2_V3_PLAN.md) §2.2. Notably its cell names
  omit `n`, so a smoke run and a real run collide silently.
- **Our own VP-diffusion QM9 model is not in the repo** and is not wired into
  `transfer_sweep.py`. The QM9-diffusion backend in v3 is **TFG's EDMsecond**,
  not ours.
- The cluster memory figures in the protocol are a **laptop extrapolation**.
  The preflight measures your real card before the array starts — read what it
  writes, especially at large n.

---

## 7. If something refuses

Every refusal in this chain is deliberate and names its reason. The common
ones:

| message | meaning |
|---|---|
| `V3_N is not set` | pick n (§3) |
| `batch 500 does not divide n` | choose an n that is a multiple of 500 |
| `is not the pinned generator` | wrong or corrupt `fm_ema.pt`; re-clone |
| `PROJ=... does not exist` | set `CGM_PROJ` (§2) |
| `TFG / EquiFM assets missing or corrupt` | run the fetch scripts (§1) |
| table: `REFUSING ... cells absent` | the run is incomplete; the message lists which (property, arm) |
| table: `... disagree` on `pair`/`n`/`delta_mode` | you pooled two different configurations — check `--stage` and `--n` |

The full gate suite is `python proj1/tests/test_v3.py` (80 closed-form checks,
no GPU needed). Run it after any edit to the protocol, the job or the planner;
it is what keeps the three files in step.
