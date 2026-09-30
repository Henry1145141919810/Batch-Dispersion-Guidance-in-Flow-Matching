# Betty runbook — the same five steps, every time

**Written for:** Henry, and anyone else on the team who has to drive the cluster.
This is the ONLY procedure. Every interaction with Betty is these five steps in
this order. If something here does not work, the runbook is wrong and should be
fixed — do not improvise around it.

`PROJ = $PROJECT_ROOT`

**The current run is [Protocol v3](#protocol-v3-26-sep--the-current-headline--the-same-five-steps)**
(26 Sep). The campaign sections after it — the transfer, the base-model
comparison, the TFG arm — are historical: their trees still exist, but nothing in
them is scheduled.

---

## Rule 0 — the two lines that start every session

```
cd $PROJECT_ROOT
source .venv/bin/activate
```

**Your prompt must change to `(.venv) <user>@login01:...`.** If it does not,
nothing else on this page will work — every `python` command will fail with
`ModuleNotFoundError: No module named 'torch'`.

This is the single most common failure. It has cost us two rounds already.

**Also:** when copying commands out of chat, copy ONLY the command. Arrows,
`# comments` explaining what to expect, and table pipes are not part of it —
the shell will try to run them.

---

## Step 1 — SHIP THE CODE (laptop)

Nothing on Betty updates itself. Any code change means a new tarball.

```
cd "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf bundles/code/code_vN.tgz proj1/scripts proj1/src proj1/tests proj1/cluster
scp bundles/code/code_vN.tgz betty:$PROJECT_ROOT/
```

**On the laptop, every tarball lives in `bundles/`** (since 27 Sep 2026; they
used to pile up in the project root). Only the laptop path changed: `scp` still
drops the file straight into `$PROJ` on Betty, so every Betty-side command below
is unchanged. The highest number used so far is **`code_v19`**, so the next
one is `code_v20`; `ls bundles/code/` shows every name that is taken. The
historical campaign sections further down predate this and still use root
paths. On the laptop, read their `code_vNN.tgz` as `bundles/code/code_vNN.tgz`,
`fm_transfer_assets_v1.tgz` as `bundles/assets/fm_transfer_assets_v1.tgz`, and
their `scp betty:... "…/Project 1/"` pulls as landing in `…/Project 1/bundles/results/`.

Then on Betty, **extract from `$PROJ`, never from inside `proj1/`** — otherwise
you get `proj1/proj1/...` and the jobs silently run the old code:

```
tar tzf code_vN.tgz | head
tar xzf code_vN.tgz
```

The first line must show paths starting `proj1/`.

---

## Step 2 — VERIFY BEFORE SUBMITTING

Always. Both commands below are genuinely seconds — tiny tensors and a plan
count, no model and no dataset loaded — so they are safe on the login node.

```
python proj1/tests/test_v2_arms.py
```
```
python proj1/tests/test_dflow.py
```
```
python proj1/tests/test_full_run.py
```
```
python proj1/tests/test_btvg2_xproj.py
```
```
python -c "import sys; sys.path[:0]=['proj1/scripts','proj1/src']; import guidance_sweep as g; print(g.COMPARE_SET); print(len(g.plan_compare_cells(['mu','alpha','gap'])),'compare cells')"
```

Expect `ALL PASS` four times, then the arm list and **`258 compare cells`**.
The arm list must include `btvg` and `btvg_var`. If it doesn't, Betty has an
old tarball — go back to Step 1.

**If anything fails, stop.** Do not submit. Send me the output.

### Do NOT run `--preflight` on the login node

It loads the full generator and dataset and samples every arm — measured at
**6.5 minutes of CPU**, `dflow` alone 150 s. That is compute on a login node.

You do not need to: `guidance_sweep_v2.slurm` runs the same preflight **inside
the job**, under `set -e`, before the sweep starts. A broken arm kills the job
in about two minutes having written nothing. That is the preflight's real home.

If you want it by hand, give it a GPU:

```
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=00:20:00 --pty bash
```

then run it inside that shell.

### Name every tarball uniquely

`code_v7.tgz` was rebuilt under the same name after a fix, the old one stayed
on Betty, and the compare stage silently lost two of its seven arms. **Never
reuse a tarball name.** Bump the number every time.

---

## Step 3 — SUBMIT

Always `--export=ALL,...`. Without `ALL` the job does not inherit `PATH` from
the venv activation and dies on `import torch` inside the job.

```
for P in mu alpha gap; do
  J=$(sbatch --parsable --export=ALL,SWEEP_PROPS=$P proj1/cluster/guidance_sweep.slurm)
  sbatch --dependency=afterany:$J --export=ALL,SWEEP_PROPS=$P proj1/cluster/guidance_sweep.slurm
  echo "$P -> $J"
done
```

Pattern: **one head job per property, plus one `afterany` insurance link.** The
sweep is resumable, so the second link finishes whatever the 4-hour QOS window
cut off, and exits in ~3 minutes if there is nothing left.

To run a different stage, add `STAGE=`:

```
--export=ALL,SWEEP_PROPS=$P,STAGE=compare
```

Stages: `main` (v1 arms) · `v2` (new arms) · `compare` (the controlled table).

### The full run is different: one array job + two sweepers

`full_run.slurm` is NOT a per-property chain. First check the compare stage
is finished (seconds of CPU, reads JSONs only — fine on the login node):

```
python proj1/scripts/check_fullrun_go.py --target q90
```

`INCOMPLETE` → wait. A table ending in `FR3a HEADLINE strengths` → submit:

```
FULL=$(sbatch --parsable --array=0-11 --export=ALL proj1/cluster/full_run.slurm)
```
```
S1=$(sbatch --parsable --dependency=afterany:$FULL --export=ALL proj1/cluster/full_run.slurm)
```
```
sbatch --dependency=afterany:$S1 --export=ALL proj1/cluster/full_run.slurm
```

The first is 12 tasks: 0-8 the PRIMARY set (3 properties × 3 seeds, 6 arms,
~2 h each on the slowest slice), 9-11 the SECONDARY set (FR3 as registered,
seed 1, ~1-1.5 h each). dflow is not in it (23 Sep decision). Primary tasks have the lowest indices, so
they start first. The other two jobs have no `--array`, so they are insurance
sweepers: each waits for the one before it, then finishes anything left. Two,
because the time guard never refuses a job's first cell, so a sweeper can
start a 31-min btvg cell with 25 minutes left and be killed. With nothing
left each exits in minutes.

The job re-checks FR1 and the compare grid itself (`STOP: the compare stage is
incomplete` / `STOP: FR1 FAILS on q50`), so a job that starts too early wastes
minutes, not hours. Exit codes of `check_fullrun_go.py`: 0 pass, **10** FR1
fail, 2 incomplete, 3 crash.

---

### The geometry-repair re-test: one job + one insurance link

`xproj_run.slurm` re-tests one claim: that BTVG-2's variance term adds nothing.
The old comparison was confounded, because BTVG-2 imposed its invariants before
the generator pullback and neither survived it (measured leak 3.59 against the
repaired arm's 3.9e-7). Twelve cells, both arms, n = 2048, ~2 h. Nothing it
writes can touch `results/full`, `results/sweep` or `frozen_q90.json`.

```
XP=$(sbatch --parsable --export=ALL proj1/cluster/xproj_run.slurm)
```
```
sbatch --dependency=afterany:$XP --export=ALL proj1/cluster/xproj_run.slurm
```

The second job is the insurance link, the same pattern as everywhere else: the
script is resumable (a cell with a JSON is skipped), so the link finishes
whatever the 4-hour window cut off and exits in minutes if there is nothing
left. **The job's own exit code is trustworthy here:** it counts the 12 cells
at the end and returns 1 if any are missing, so `INCOMPLETE` in the log means
resubmit. `guidance_sweep.py` alone would have exited 0 in that case.

---

## Step 4 — CHECK IT IS DOING THE RIGHT THING

```
squeue -u $USER -o "%.10i %.16j %.8T %.10M %.22E"
```

`PD (Priority)` = waiting for a free GPU, normal.
`PD (Dependency)` = waiting for its parent, normal.
`R` = running.

Once a job starts, confirm it planned the cells you expected:

```
grep -m1 'cells total' logs/sweep-*.out
grep -m2 'cells total' logs/sweepv2-*.out
```

`-m2` for `sweepv2` and `full` logs: those jobs run the preflight inside the
job first, and its line (`cells total 7`, one cell per arm) comes first.
`-m1` stops on it and looks like a wrong stage. It is not.

Each job runs **one property** (`SWEEP_PROPS=$P`), so the count is per
property, a third of the stage total:

| stage | per job (one property) | whole stage |
|---|---|---|
| main | 160 | 480 |
| v2 | 114 | 342 |
| compare | 86 | 258 |
| full | 6 per primary task, 3-4 per secondary | 64 (54 primary + 10 secondary) |
| xproj | 6 per strength (w=16, then w=8) | 12 |

**A wrong number here means `STAGE` did not export** — cancel those jobs and
resubmit. The `stage : v2` line in the log header is a hardcoded string; ignore
it and trust the cell count.

---

## Step 5 — COLLECT

```
ls results/sweep/*.json | wc -l
ls results/sweep/*.failed 2>/dev/null | wc -l
ls results/sweep/*__cmp.json | wc -l
python proj1/scripts/check_fullrun_go.py --target q50
python proj1/scripts/check_fullrun_go.py --target q90
```

Want: 1080 JSON once main + v2 + compare are all done, 0 failed, 258
compare cells. The job logs end in `SWEEP COMPLETE` when a property is done.
`select_arms.py` has no `compare` stage (it takes `main`, `v2`, `all`), so
the compare readout is `check_fullrun_go.py`: seconds, reads JSONs only.

To pull results back to the laptop for analysis:

```
tar -czf cells.tgz results/sweep results/sweep_v2_seed2
```
```
scp betty:$PROJECT_ROOT/cells.tgz "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1/bundles/results/"
```

Pulled tarballs land in `bundles/results/`, not the project root. Extract them
**from the project root**, e.g. `tar xzf bundles/results/cells.tgz ...`, so
their `results/...` paths land in the tree.

For the full run: `ls results/full/n5000/seed*/*__full.json | wc -l` (want 64),
then `python proj1/scripts/full_run_table.py`. It refuses to print a partial or
inconsistent run and says what is missing. To pull it back:
`tar -czf full.tgz results/full`.

On the laptop, `tar xzf bundles/results/cells.tgz --skip-old-files` keeps any file that
already exists locally — the **local** copy wins, not Betty's. That is only
safe if the overlapping cells are identical. Cells are deterministic (the
compare stage reproduced 75 earlier cells exactly), and on 23 Sep all 690
overlapping cells were byte-identical. But a cell re-run on Betty after a fix
would silently lose to its stale local copy. So extract into a scratch folder,
compare the overlap, then merge. Claude does this step.

---

## The gotchas, all of them, in one place

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: torch` | venv not active | `source .venv/bin/activate` |
| job dies instantly on `import torch` | `--export` without `ALL` | `--export=ALL,...` |
| `MEM_PER_CPU_MISMATCH` | memory not a multiple | omit `--mem`, or `b200-mig45` = 6 CPU / 48G; `genoa-std-mem` = 5632 MB per CPU (4 CPU → 22528M) |
| `cells total` is wrong | `STAGE` did not export | resubmit with `--export=ALL,SWEEP_PROPS=$P,STAGE=x` |
| `proj1/proj1/...` appears | extracted from inside `proj1/` | extract from `$PROJ` |
| job killed at 60 min | `run.slurm` defaults to `--time=01:00:00` | add `--time=04:00:00` on the sbatch line |
| `.failed` cells for one arm | missing checkpoint for that arm | that arm is skipped, the rest still run; fit the checkpoint and the next link picks it up |
| command runs my explanation text | pasted the `→ ...` annotation too | copy only the command line |
| job runs old behaviour after you shipped a fix | **`sbatch` snapshots the `.slurm` file at submission.** Only the *Python* is read at start time | if a `.slurm` changed, `scancel` and resubmit. Check what a queued job will run: `scontrol write batch_script <JOBID> -` |
| every dependent job waits on the same parent | a loop picked one job ID for all properties | chain each property to its **own** parent; check the `DEPENDENCY` column in `squeue -o "%.22E"` |
| a stage silently lost arms | a tarball was rebuilt under the same name and the old one stayed on Betty | never reuse a tarball name — bump `code_vN` every time |

## Never

- run training or a sweep on a login node — `sbatch` or `srun`, always
- `ssh` directly into a compute node
- assume Betty has your latest code — ship it (Step 1)
- submit without Step 2

---

## Protocol v3 (26 Sep) — the current headline — the same five steps

**This is the run to drive.** Protocol:
[FULL_RUN_V3_PROTOCOL.md](FULL_RUN_V3_PROTOCOL.md). Everything below it on this
page — the transfer, the base-model comparison, the TFG arm — is **historical**,
kept because those trees still exist, not because they are scheduled. The
`basecmp` chain was cancelled on 26 Sep; its protocol differs from v3 on target,
strength and arm set, so its cells cannot be pooled with v3's.

Own job file `proj1/cluster/v3_run.slurm`, own tree `results/v3/<backend>/`. It
touches nothing under `results/sweep`, `results/full`, `results/basecmp` or the
transfer tree.

| | |
|---|---|
| headline | 7 arms × 3 properties × 3 seeds × 2 bases = **126 cells**, array `0-17`, tree `n2000/` |
| ablation | 17 BDG arms × 3 properties × 1 seed × 2 bases = **102 cells**, array `0-5`, its **own** tree `n1000/` |
| per cell | headline n = **2000** in **4 batches of 500**; ablation n = **1000** in 2. w = 1, q50, t ≥ 0.5 |
| BDG | headline η = 4 × τ_mult ∈ **{0.5, 1.0}**; ablation η ∈ {0,1,2,4,8} × τ_mult ∈ {0.5,0.75,1,1.5} |
| floor | **none** — nothing is excluded for chemistry; chemistry is reported |
| cost | headline ≈ **12.8 GPU-h**, ablation ≈ **4.2**, both ≈ **17.0**; longest task ≈ 75 min against the 4 h wall |

**The two stages must keep DIFFERENT n.** `n` is the only thing separating their
trees, and `v3_table.py` refuses if the ablation's single-seed arms land in the
headline's table. Do not set `V3_ABL_N=2000`.

### Step 1 — ship the code (laptop)

**Only the code.** `proj1/checkpoints/` is deliberately NOT in the tarball, so
shipping code never disturbs the weights already on Betty.

```
cd "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf bundles/code/code_v20.tgz proj1/scripts proj1/src proj1/tests proj1/cluster results/v3_batch_memory.json
scp bundles/code/code_v20.tgz betty:$PROJECT_ROOT/
```

This stage first shipped as `code_v17`; v17, v18 and v19 are all taken, so the
commands above and in Step 2 say `code_v20`, the next free name as of 27 Sep.
**If `bundles/code/code_v20.tgz` already exists, bump the number in all four
lines**; never rebuild a name that has shipped.

`results/v3_batch_memory.json` travels with the code on purpose: it is the
measurement that chose `--batch`, and `test_v3.py` gates the job's batch against
it. Without it the gates cannot run.

### Step 2 — verify (Betty)

```
cd $PROJECT_ROOT
source .venv/bin/activate
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf

md5sum code_v20.tgz
tar xzf code_v20.tgz
md5sum proj1/checkpoints/fm_last.pt
python proj1/tests/test_v3.py
python proj1/tests/test_bdg.py
```

Want: `fm_last.pt` = `a190ac8394902027d4a951f8d30e8c5c` (the slurm FATALs in
seconds on any other generator — every cell in this project used that one, and a
different one would make v3's two halves incomparable to each other and to v2),
`ALL PASS (54 gates)`, `22/22 checks pass`.

**Then confirm the batch on a GPU node, with the real checkpoint.** This is the
one number the laptop cannot settle: the slope was measured on a 16 GB card with
`weights/fm_ema.pt`, and the run uses a 45 GB slice with `fm_last.pt`.

```
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=00:25:00 \
  python proj1/scripts/batch_memprobe.py --confirm 500 \
    --json-out results/v3_batch_memory_betty.json \
    --md-out docs/results/V3_BATCH_MEMORY_BETTY.md
```

Want roughly `fm ~20 GiB`, `equifm ~28 GiB`, both inside the 33.5 GiB budget. If
EquiFM lands above ~33 GiB, submit with `V3_BATCH=250` (still divides 5000; 20
controllers; 9 % standard error on the variance) rather than pressing on. **A
batch that does not divide `V3_N` is refused** by both the submit script and the
job: a remainder batch is a second, far noisier BDG controller pooled in as an
equal.

> ⚠️ **`V3_BATCH` can only be changed BEFORE the first cell is written.** The batch
> is part of every cell's filename (`…_b500_…`) and is in `v3_table.py`'s
> `SAME_KEYS`. Change it mid-run and you get a tree holding two batch sizes: the
> old cells are not resumed (different names), arms are no longer paired on the same
> initial noise, and the table **refuses** on every page. If you must change it
> after cells exist, move `results/v3` aside and start the tree again.

### Step 3 — submit (one command)

```
bash proj1/cluster/submit_v3.sh
```

Five chained jobs: headline array → insurance → ablation array → insurance →
table. `afterany`, not `afterok`, because a task that runs out of its 4-hour
window exits non-zero with its finished cells on disk and the next link must
still run. The ablation is chained **after** the headline rather than beside it
because they share one tree — run at once, two jobs would write the same three
BDG cells at the same instant. Ordered, the headline's cells exist first and the
ablation skips them (≈ 8 GPU-h saved).

Overrides, both forwarded through `--export`: `V3_N=...` lowers n, `V3_BATCH=...`
changes the controller size.

### Step 4 — check

```
squeue -u $USER -o "%.18i %.14j %.9T %.10M %.28E"
grep -h '[timing]' logs/v3-*.out | tail -20
ls results/v3/fm/n2000/seed20261001/ | wc -l        # headline: counts up to 21 (7 arms x 3 props)
ls results/v3/equifm/n2000/seed20261001/ | wc -l    # same
ls results/v3/fm/n1000/seed20261001/ | wc -l        # ablation: counts up to 51 (17 arms x 3 props)
```

**Read the `[timing]` lines after the first few tasks land.** EquiFM's per-cell
cost is the one unmeasured number in the budget: both bases run the *same* 350
network passes per guided cell, so EquiFM's extra cost is per pass only (it is
the bigger net, nf = 256, 9 layers), but the 1.83× in the budget is **borrowed**
from the TFG/EDMsecond backend, not measured on EquiFM. The split is sized for
2.5×. If `[timing]` shows worse than that, lower `V3_N` before the ablation's 36
tasks queue.

### Step 5 — collect

The table job writes these on Betty; it **refuses** rather than warns if the run
is incomplete, and names the missing seed or cell.

```
docs/results/V3_RESULTS.md          both bases side by side
docs/results/V3_RESULTS_fm.md       ours
docs/results/V3_RESULTS_equifm.md   EquiFM
```

Re-run just the table after filling a gap:

```
sbatch --export=ALL,V3_N=5000,V3_BATCH=500,STAGE=table proj1/cluster/v3_run.slurm
```

### Reading v3 — three things that travel with every number

1. **`at w = 1`.** Equal `w` is not equal force: the measured correction share at
   w = 1 spans 10× across arms, 5.8× across the v3 arms it covers, and `tfg` has
   no share to measure at all. v3 is "what does each method do at one dial
   setting", not "which method is best".
2. **No in-band leaderboard.** With no floor, the arm furthest above its own best
   strength looks strongest on in-band while destroying the most chemistry — at
   w = 1, `tfg` takes the highest in-band on all three properties at validity
   0.62. `v3_table.py` refuses to rank on in-band alone, by design.
3. **BDG's ladder is a strength ladder at q50.** `w_eff = 1 + η(1/τ_mult² − 1)`
   exactly, so the headline's three BDG arms sit at `w_eff` of **13.0 / 4.11 /
   1.00** — `bdg_e4t1` *is* plug at onset — and τ_mult 1.5 **reverses** the deviation
   term. Order BDG rows by `w_eff`, never by τ_mult, and state the strength
   confound with any BDG gain. See FULL_RUN_V3_PROTOCOL.md §4.1.

---

## QM9 diffusion, OURS (`--backend vp`) — the same five steps

**Not `edm`.** `edm` is TFG's borrowed EDMsecond; `vp` is the model we trained.
Both are diffusion and only `vp` is ours. Protocol:
[FULL_RUN_V3_PROTOCOL.md §1.3](FULL_RUN_V3_PROTOCOL.md) (the 28 Sep amendment).

| | |
|---|---|
| grid | 2 arms (`unguided`, `plug`) × 3 properties × 3 seeds × w = 1 = **18 cells** |
| per cell | n = **2000** in 4 batches of 500, q50, t ≥ 0.5 |
| ablation | **none** — `vp` sits it out, as `edm` does |
| cost | ≈ **1.5 GPU-h** total; longest task ≈ 10 min against the 4 h wall |
| tree | `results/v3/vp/v3/n2000/seed<S>/` |

**It fits in one window, so do NOT use `submit_v3.sh`** — that chain re-plans
`fm`, `equifm` and `edm` too. Nine independent tasks is the whole run.

### Prerequisite — the checkpoint EXISTS. Skip to Step 1.

**Done 28 Sep**: `weights/diff_ema.pt`, md5 `8a3390a6`, selected epoch 1475 of
1500, trained by Bobo. The training recipe in the rest of this subsection is
**archival** — keep it for a retrain, do not run it now.

*History, for the record.* Job 8590174 — the first run with both the right split
and the right architecture — died at epoch ≈80 of 1500 (`logs/diff-8590174.err`:
`CANCELLED … DUE TO NODE FAILURE`, 2026-09-21T16:58:59) and was never restarted.
Its epoch-75 remains later got picked up by the resolver and produced a full set
of 18 **wrong** cells at atom stability 0.763 instead of 0.905; the gate now pins
the checkpoint md5 (`vp_bench.slurm`). A field check is not an identity check.

**Clear the stale `train_ab` checkpoints first. There are TWO guards, not one,
and the `--resume` guard on `diff_last.pt` fires *before* the selection guard
on `diff.pt`.** `--fresh` does not help: `train_diffusion.slurm` hardcodes
`--resume`, and the resume guard is checked at `train_diffusion.py:293`, before
`--fresh` is consulted at `:340`. With `afterany`, skipping this does not fail
one link — **all five run and all five die in under a minute, and the night
yields nothing.** The audited recipe
([QUEUE_CHECK.md](../../results/bench/audit4/QUEUE_CHECK.md) §B):

```
cd $PROJECT_ROOT
mkdir -p proj1/checkpoints/old_train_ab
mv proj1/checkpoints/diff* proj1/checkpoints/old_train_ab/ 2>/dev/null
ls proj1/checkpoints/diff* 2>/dev/null && echo "STILL THERE - DO NOT SUBMIT" || echo "clear"
```

The `diff*` glob is deliberate — `diff_last_prev.pt`, `diff_ep*.pt` and
`diff_history.json` are all clobbered or misread otherwise. Nothing `fm*` is
touched by `train_diffusion.py`.

**Then chain the links link-to-link, never all-to-link-1** — five jobs that
all depend on the first would run concurrently, and five `atomic_save`s of
`diff_last.pt` would interleave and destroy the run:

```
source .venv/bin/activate
J=$(sbatch --parsable proj1/cluster/train_diffusion.slurm)
for i in 2 3 4 5; do
    J=$(sbatch --parsable --dependency=afterany:$J proj1/cluster/train_diffusion.slurm)
done
squeue -u $USER -o "%.10i %.10P %.12j %.8T %.20E"   # every link but the first must show a Dependency
```

`afterany`, not `afterok`: a node failure should not stall the tail.

**Cost: ≈12 GPU-h.** 28.5 s/epoch measured on job 8590174 itself
(`logs/diff-8590174.out`, ep 25 → 80: (2332−765)/55), which agrees to 0.7 %
with QUEUE_CHECK's independent 28.69 s/epoch projection. **Four links are
needed and the fifth is a spare** — links 1–3 are full 470-epoch/3 h 45 m
links, link 4 is ep 1413–1500 (≈42 min), link 5 exits in ≈20 s. The same shape
as FM's measured 11 h 23 m.

### Step 1 — ship the code (laptop)

```
cd "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf bundles/code/code_v20.tgz proj1/scripts proj1/src proj1/tests proj1/cluster results/v3_batch_memory.json
scp bundles/code/code_v20.tgz betty:$PROJECT_ROOT/
```

`code_v19` was the last shipped, so **`code_v20` is the next free name**
(bundles/README.md keeps that count). Never rebuild a name that has shipped. The tarball excludes `proj1/checkpoints/` on purpose, so
shipping code cannot disturb the weights already on Betty.

### Step 2 — verify (Betty)

```
cd $PROJECT_ROOT
source .venv/bin/activate
export SLURM_CONF=/cm/shared/apps/slurm/etc/slurm/slurm.conf

md5sum code_v20.tgz
tar xzf code_v20.tgz

md5sum proj1/checkpoints/diff_last.pt
python -c "import torch;st=torch.load('proj1/checkpoints/diff_last.pt',map_location='cpu',weights_only=False);print(st.get('family'),st.get('epoch'),st['args'])"

python proj1/tests/test_v3.py
python proj1/tests/test_bdg.py
```

Want `family=vp_diffusion`, `epoch=1500`, and `args` showing `hidden 256,
layers 8, split train_a, seed 20260918`. Then `ALL PASS (106 gates)` and
`25/25`. **Record that md5** — it lands in every cell's provenance as
`gen_md5`, and it is how a reader tells a `vp` cell from an `edm` one.

Then the GPU smoke, **not on the login node**:

```
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=00:20:00 \
  python proj1/scripts/transfer_sweep.py --stage v3 --backend vp \
    --props mu --arms unguided,plug --n 2000 --batch 500 --seed 20261001 --preflight
```

### Step 3 — submit (nine independent tasks)

```
for P in mu alpha gap; do
  for S in 20261001 20261002 20261003; do
    sbatch --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G --time=04:00:00 \
      --job-name=vp-$P-$S --output=logs/vp-%j.out \
      --wrap="cd $PROJECT_ROOT && source .venv/bin/activate && \
        python -u proj1/scripts/transfer_sweep.py --stage v3 --backend vp \
          --props $P --arms unguided,plug --n 2000 --batch 500 --seed $S --per-mol"
  done
done
```

Everything else is argparse default, and that is what the cells record:
`--steps 100`, `--solver euler`, `--grid uniform`, `--tau-max-guide 0.5`,
`--clip 1.0`, `--k-delta 2.0`.

> ⚠️ **Do not pass `--grid gamma` on `vp`.** The gamma grid needs a noise
> schedule object to invert, and `vp` passes `noise_schedule=None` — which
> *is* the linear-beta schedule our generator was trained under, not an
> omission. It does **not** fail fast: `VPSampler.__init__` raises a plain
> `ValueError`, `run_cell` catches it per cell, writes a `.failed` marker and
> continues, so the task burns its whole wall and *then* returns 1. Leave
> `--grid` at its default.

### Step 4 — check

```
squeue -u $USER -o "%.18i %.14j %.9T %.10M %.28E"
grep -h '\[timing\]' logs/vp-*.out | tail -20
ls results/v3/vp/v3/n2000/seed20261001/*.json | wc -l   # 6 (2 arms x 3 props)
```

### Step 5 — collect

**Not** the chain's table job — `v3_run.slurm`'s `BACKENDS` excludes `vp` by
design, so `STAGE=table` will never produce a vp page. Call the table directly:

```
python proj1/scripts/v3_sanity.py                       # takes no arguments
python proj1/scripts/v3_table.py --backend vp --stage v3 --n 2000 \
    --seeds 20261001,20261002,20261003 \
    --md-out docs/results/V3_RESULTS_vp.md
```

`--n` is **required** and has no default (v3 pre-registers no cell size), and
without `--md-out` the table goes to stdout only and no file is written. Use
`--backend vp` writes only the vp page. `--both` now probes the tree and walks
every backend that has cells for the stage, so it includes `vp` as well.
It **refuses** rather than warns if a cell is missing, and names it.

### Reading `vp` — what it does and does not settle

1. **`fm` vs `vp` is the matched comparison** — same backbone, parameters,
   epochs, batch, EMA, split and seed. It isolates the generator family and
   nothing else. This is the cell `tab:fmvd` had never had until 28 Sep.
   **Result: `fm` wins on all three** — mol stab 0.3970 vs 0.2883, validity
   0.7562 vs 0.6617, atom stab 0.9356 vs 0.9070.
2. **`vp` vs `edm` is ours-against-borrowed, on one sampler and one pair.** Both
   run through *our* 100-step probability-flow ODE, not EDM's native 1000-step
   ancestral SDE, so neither row is EDM's published unconditional number and
   neither may be quoted as one.
3. **`vp` is not comparable to `equifm` on `in_band`.** Different pair, different
   band width. `v3_table.py` refuses to pool them; do not do by hand what it
   refuses to do for you.

---

## The transfer (borrowed FM model: EquiFM) — the same five steps

Its own job file, `proj1/cluster/transfer_run.slurm`, and its own results tree,
`results/transfer_equifm/`. It never touches `results/sweep/` or `results/full/`, so
it can run beside the main full run. Protocol:
[TRANSFER_EXPERIMENT_PLAN.md](TRANSFER_EXPERIMENT_PLAN.md) §10. `BACKEND=equifm`
is the default. The EDMsecond (diffusion) transfer of §9 was dropped on 23 Sep
when FM was chosen; it still runs with `BACKEND=edm` but is not scheduled.

### Step 1 — ship the code AND the TFG assets (laptop)

The transfer needs files the main sweep never did: the EquiFM checkpoint and
its network definitions, TFG's guide and oracle networks, and OC-Flow's three
property networks (the second oracle). They travel in their own bundle:

```
cd "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf code_v13.tgz proj1/scripts proj1/src proj1/tests proj1/cluster
scp code_v13.tgz fm_transfer_assets_v1.tgz betty:$PROJECT_ROOT/
```

`fm_transfer_assets_v1.tgz` is already built, in `bundles/assets/` since 27 Sep (55 MB, 42
files, md5 `7eb4ab32…`).
`code_v11` and `code_v12` are already taken (v12 shipped the TFG arm). Next
code change: `code_v14.tgz`, never `v13` again.

### Step 2 — verify (Betty, login node, seconds)

Rule 0 first, then:

```
tar tzf code_v13.tgz | head -3
tar xzf code_v13.tgz
tar xzf fm_transfer_assets_v1.tgz
python proj1/scripts/fetch_tfg_assets.py --verify --models ""
python proj1/scripts/fetch_equifm_assets.py --verify
python proj1/tests/test_transfer_protocol.py
python proj1/scripts/transfer_sweep.py --dry-run
python proj1/scripts/transfer_sweep.py --stage eqtune --dry-run
```

Expect paths starting `proj1/`, then **`READY`** twice (TFG networks, then
EquiFM + OC-Flow files), then **`ALL PASS (22 gates)`**, then
**`cells total 330 | done 0 | to run 330`** (7 arms × 9 strengths × q50/q90)
and **`cells total 183 | done 0 | to run 183`** (the equal-chemistry tune
block). Anything else: stop and send the output.

### Step 3 — submit both chains

Two independent chains; they can run side by side. Each stage waits
(`afterany`) for the previous stage's last insurance sweeper, and every stage
refuses to start if its input is incomplete, so nothing is computed from a
partial grid.

**Chain 1 — the headline: compare → extend (edge rule) → full.**

```
CMP=$(sbatch --parsable --array=0-11 --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm)
C1=$(sbatch --parsable --dependency=afterany:$CMP --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm)
C2=$(sbatch --parsable --dependency=afterany:$C1 --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm)
EXT=$(sbatch --parsable --dependency=afterany:$C2 --array=0-2 --export=ALL,STAGE=extend proj1/cluster/transfer_run.slurm)
E1=$(sbatch --parsable --dependency=afterany:$EXT --export=ALL,STAGE=extend proj1/cluster/transfer_run.slurm)
FT=$(sbatch --parsable --dependency=afterany:$E1 --array=0-23 --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm)
F1=$(sbatch --parsable --dependency=afterany:$FT --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm)
F2=$(sbatch --parsable --dependency=afterany:$F1 --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm)
echo $CMP $C1 $C2 $EXT $E1 $FT $F1 $F2
```

**Chain 2 — the equal-chemistry comparison: eqtune → eqextend (edge rule) → eqconfirm.**

```
EQT=$(sbatch --parsable --array=0-56 --export=ALL,STAGE=eqtune proj1/cluster/transfer_run.slurm)
Q1=$(sbatch --parsable --dependency=afterany:$EQT --export=ALL,STAGE=eqtune proj1/cluster/transfer_run.slurm)
Q2=$(sbatch --parsable --dependency=afterany:$Q1 --export=ALL,STAGE=eqtune proj1/cluster/transfer_run.slurm)
EQX=$(sbatch --parsable --dependency=afterany:$Q2 --array=0-2 --export=ALL,STAGE=eqextend proj1/cluster/transfer_run.slurm)
X1=$(sbatch --parsable --dependency=afterany:$EQX --export=ALL,STAGE=eqextend proj1/cluster/transfer_run.slurm)
EQC=$(sbatch --parsable --dependency=afterany:$X1 --array=0-17 --export=ALL,STAGE=eqconfirm proj1/cluster/transfer_run.slurm)
R1=$(sbatch --parsable --dependency=afterany:$EQC --export=ALL,STAGE=eqconfirm proj1/cluster/transfer_run.slurm)
R2=$(sbatch --parsable --dependency=afterany:$R1 --export=ALL,STAGE=eqconfirm proj1/cluster/transfer_run.slurm)
echo $EQT $Q1 $Q2 $EQX $X1 $EQC $R1 $R2
```

Cost at 5080 speed (plan §11.4): chain 1 ≈ 60 GPU-h, chain 2 ≈ 80 GPU-h.

### Step 4 — check

```
grep -h "cells to run\|preflight passed\|STOP\|frozen\|refused" logs/transfer-*.out | tail -40
python proj1/scripts/transfer_sweep.py --dry-run
python proj1/scripts/transfer_sweep.py --stage eqtune --dry-run
```

`to run 0` on both means the two screening stages are done. If a later stage
logged `STOP ... refused` because its input was not finished when it
started, resubmit just that stage's line once the input says `to run 0`.

### Step 5 — collect (laptop)

```
scp -r betty:$PROJECT_ROOT/results/transfer_equifm results/
python proj1/scripts/full_run_table.py --root results/transfer_equifm/full/n5000
python proj1/scripts/full_run_table.py --root results/transfer_equifm/eqchem/confirm/n5000 --seeds 20261004,20261005,20261006 --frozen results/transfer_equifm/eqchem/frozen_eqtune.json
python proj1/scripts/dist_report.py --backend equifm --dir "results/transfer_equifm/full/n5000/seed*" --frozen results/transfer_equifm/full/n5000/frozen_q90.json
```

The equal-chemistry frontier (in_band against mol_stability across w, and
each arm's in_band at the chemistry floor) is inside
`results/transfer_equifm/eqchem/frozen_eqtune.json` under `"frontier"`, and
was printed in the log of the first eqconfirm task. **`--backend equifm` is
not optional** for `dist_report.py`.

---

## The base-model comparison (25 Sep) — the same five steps

Our base model against EquiFM, both through ONE external property pair, all
seven arms, screening strength **and** guidance start-time jointly; then the v2
full run on EquiFM at both picks. Its own job file,
`proj1/cluster/basecmp_run.slurm`, its own submit script, and its own results
tree `results/basecmp/`. It never touches `results/sweep/`, `results/full/`,
`results/tune/` or `results/transfer_equifm/`, so it can run beside anything.
Protocol: [BASECMP_PROTOCOL.md](BASECMP_PROTOCOL.md).

**The whole thing is one command in Step 3.** Seven stages are queued at once
and chained with `--dependency`; nothing needs resubmitting.

### Step 1 — ship the code and the assets (laptop)

Same assets as the transfer (EquiFM + TFG's pair + OC-Flow), already built.
`code_v14` is taken; this tarball has its own name so it cannot collide with
another session's:

```
cd "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf code_v15_basecmp.tgz proj1/scripts proj1/src proj1/tests proj1/cluster
scp code_v15_basecmp.tgz fm_transfer_assets_v1.tgz betty:$PROJECT_ROOT/
```

`fm_transfer_assets_v1.tgz` may already be on Betty from the transfer work; it is
55 MB and re-copying it is harmless.

### Step 2 — verify (Betty, login node, seconds)

Rule 0 first, then:

```
tar tzf code_v15_basecmp.tgz | head -3
tar xzf code_v15_basecmp.tgz
tar xzf fm_transfer_assets_v1.tgz
python proj1/scripts/fetch_tfg_assets.py --verify --models ""
python proj1/scripts/fetch_equifm_assets.py --verify
md5sum proj1/checkpoints/fm_last.pt
python proj1/tests/test_basecmp.py
python proj1/tests/test_transfer_protocol.py
python proj1/scripts/transfer_sweep.py --stage basecmp --backend fm --dry-run
python proj1/scripts/transfer_sweep.py --stage basecmp --backend equifm --dry-run
python proj1/scripts/transfer_sweep.py --stage compare --backend edm --dry-run
```

Expect, in order: paths starting `proj1/`; **`READY`** twice; md5
**`a190ac8394902027d4a951f8d30e8c5c`**; **`ALL PASS (30 gates)`**;
**`ALL PASS (22 gates)`**; then
**`cells total 363 | done 0 | to run 363`** twice — once per base model — and
finally **`cells total 330 | done 216 | to run 114`**.

That last line is the regression check, and it is the important one: it proves
the per-cell guidance window did not change any existing cell's filename. **If
it says `done 0`, stop** — every finished cell in the project has just been
orphaned and nothing should be submitted.

Do **not** run `--preflight` here. Each job runs it inside itself, on a GPU,
before sampling anything.

### Step 3 — submit (one command)

```
bash proj1/cluster/submit_basecmp.sh
```

It prints a line per stage with its job id and dependency, then what to read
when it lands. To see what it would do without submitting:

```
DRY=1 bash proj1/cluster/submit_basecmp.sh
```

To change the compute budget — this is the one knob, and it decides `n` for both
the screen and the full run:

```
BUDGET_GPUH=30 bash proj1/cluster/submit_basecmp.sh
```

The default is 96 GPU-hours. On the cost model's current estimate that buys the
screen at **n = 500** and the full run at **n = 1000** (~79 GPU-h); the screen at
the full n = 1000 needs **`BUDGET_GPUH=130`**. The exact choice is made from the
probe's own measurements, so read `results/basecmp/plan.json` rather than
trusting these numbers.

The chain refuses to run a full run below n = 1000 whatever the budget, because
below that the differences v2 calls verdicts sit inside their own standard error.
It also prints the most expensive single array task against the 225-minute guard,
so you can see whether the array will leave work for the sweepers.

### Step 4 — check

```
squeue -u $USER -o "%.10i %.16j %.8T %.10M %.22E"
cat results/basecmp/plan.json
grep -h "cells to run\|preflight passed\|CHOSEN\|STOP\|INCOMPLETE\|REFUSING\|frozen" logs/basecmp-*.out | tail -40
```

`plan.json` appears after the `size` stage and is the first thing to read: it
records the measured per-cell cost on **both** base models, the cost surface, and
which `n` the rule picked. The probe exists because EquiFM had never had a single
cell run on it, so this is the first real measurement of what it costs.

Per-task cell counts to expect on the screen, across its 96 tasks: **10** for the
42 two-arm tasks, **5** for the 48 single-arm ones (`btvg` and `btvg_var` get a
task each — at `t_start = 0.05` about 95 of the 100 steps are guided, so they are
the expensive ones), and **11** for the six that also carry their property's
single unguided cell. 726 cells in total across both base models.

A stage that logs `STOP ... INCOMPLETE` started before its input finished. That
is not a failure of the run — resubmit just that stage's sweeper once the stage
before it reports `to run 0`:

```
sbatch --export=ALL,BUDGET_GPUH=96,STAGE=screen proj1/cluster/basecmp_run.slurm
```

### Step 5 — collect

On Betty:

```
ls results/basecmp/fm/screen/*.json | wc -l
ls results/basecmp/equifm/screen/*.json | wc -l
ls results/basecmp/*/screen/*.failed 2>/dev/null | wc -l
ls results/basecmp/equifm/full/n*/seed*/*__full.json | wc -l
tar -czf basecmp.tgz results/basecmp docs/results/BASECMP_SCREEN_*.md
```

Want at least **363** screen cells per base model (more once the refine stage has
added cells), **0** failed, and up to **117** full-run cells (fewer where an
arm's two picks coincided — those are computed once and reported under both).

On the laptop:

```
scp betty:$PROJECT_ROOT/basecmp.tgz "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1/"
```

Then read, in this order:

```
docs/results/BASECMP_SCREEN_fm.md          our base: the picks, then every cell
docs/results/BASECMP_SCREEN_equifm.md      EquiFM: the same
results/basecmp/<base>/frozen_basecmp.json the frozen (w, t), both sets, + statuses
results/basecmp/<base>/screen_table.csv    the same table, machine-readable
docs/results/BASECMP_FULL_equifm.md        the full run under v2's V4-V8
```

**Read the statuses before the numbers.** `floor_limited` means the arm never
cleared the chemistry floor anywhere on the grid — it is a result, not a win.
`grid_edge` means its best cell sits at the edge of the screened strengths, so
the optimum may be outside them. `collapse_contaminated` means v2's distinctness
guard fired. All three are in the frozen file, the csv and both markdown tables.

Claude extracts into a scratch folder, checks nothing overlaps, and merges.

---

## The TFG arm (added 23 Sep, replaces dflow) — the same five steps

Its own job file, `proj1/cluster/tfg_run.slurm`, two stages. It writes into
the SAME trees as the main run (`results/sweep/`, `results/full/n5000/seed*`)
because its cells must pair with the existing ones, but it only ever ADDS
`*__tfg__*` cells and two new files (`frozen_q90_tfg.json`,
`fr1_q50_tfg.json`). It never rewrites `frozen_q90.json` or any existing
cell. Protocol: SCOPE_FM_GUIDANCE_STATUS.md, "AMENDMENT FR2a".

### Step 1 — ship the code (laptop)

```
cd "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf code_v12.tgz proj1/scripts proj1/src proj1/tests proj1/cluster
scp code_v12.tgz betty:$PROJECT_ROOT/
```

### Step 2 — verify (Betty, login node, seconds)

Rule 0 first, then:

```
tar tzf code_v12.tgz | head -3
tar xzf code_v12.tgz
python proj1/tests/test_tfg.py
python proj1/tests/test_full_run.py
md5sum proj1/checkpoints/fm_last.pt
ls results/full/n5000/frozen_q90.json
python proj1/scripts/guidance_sweep.py --stage compare --arms tfg --dry-run
```

Expect paths starting `proj1/`, then **`71 gates, 0 failed`**, then
**`ALL PASS`**, then md5 **`a190ac8394902027d4a951f8d30e8c5c`**, then the
frozen file listed (not "No such file"), then
**`cells total 42 | already done 0 | to run 42`**. Anything else: stop.

### Step 3 — submit both stages, chained

```
CMP=$(sbatch --parsable --array=0-2 --export=ALL,STAGE=compare proj1/cluster/tfg_run.slurm)
S1=$(sbatch --parsable --dependency=afterany:$CMP --export=ALL,STAGE=compare proj1/cluster/tfg_run.slurm)
FULL=$(sbatch --parsable --dependency=afterany:$S1 --array=0-3 --export=ALL,STAGE=full proj1/cluster/tfg_run.slurm)
S2=$(sbatch --parsable --dependency=afterany:$FULL --export=ALL,STAGE=full proj1/cluster/tfg_run.slurm)
echo $CMP $S1 $FULL $S2
```

Compare: three tasks, one per property, 14 cells each at n = 512. Full:
tasks 0-2 are the primary set (one seed each, three cells), task 3 the
secondary set (seed 1, only cells whose strength differs). The first full
task freezes the strength into `frozen_q90_tfg.json`, after checking that
every other arm's strength and the floor come out identical to
`frozen_q90.json`. If the compare stage is not finished when the full stage
starts, it prints `STOP` and runs nothing; then resubmit the last two lines.

### Step 4 — check

```
squeue -u $USER -o "%.10i %.16j %.8T %.10M %.22E"
grep -h "cells to run\|preflight passed\|strengths frozen\|consistent with\|STOP\|NOTE\|MISMATCH" logs/tfg-*.out
```

Want `14 cells to run` per compare task, `preflight passed`, then for the
full stage `consistent with .../frozen_q90.json`, `strengths frozen`, and
`3 cells to run` per primary task (0 to 3 for task 3). A `NOTE (post hoc,
report only)` line about FR1 is information, not an error. **`MISMATCH`
means stop and send me the log.**

### Step 5 — collect

On Betty:

```
ls results/sweep/*__tfg__*__cmp.json | wc -l
ls results/full/n5000/seed*/*__tfg__*__full.json | wc -l
tar -czf tfg_cells.tgz results/sweep/*__tfg__* results/full/n5000/frozen_q90_tfg.json results/full/n5000/fr1_q50_tfg.json results/full/n5000/seed*/*__tfg__*
```

Want 42, then 9 to 12. On the laptop:

```
scp betty:$PROJECT_ROOT/tfg_cells.tgz "C:/Users/<user>/Downloads/pennstuff/cis 6270/Project 1/"
```

Claude extracts it into a scratch folder, checks nothing overlaps, merges,
and reads it with
`python proj1/scripts/full_run_table.py --frozen results/full/n5000/frozen_q90_tfg.json`.
