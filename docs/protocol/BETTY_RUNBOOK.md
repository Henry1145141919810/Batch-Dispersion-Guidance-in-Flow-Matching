# Betty runbook — the same five steps, every time

**Written for:** Henry, and anyone else on the team who has to drive the cluster.
This is the ONLY procedure. Every interaction with Betty is these five steps in
this order. If something here does not work, the runbook is wrong and should be
fixed — do not improvise around it.

`PROJ = /vast/projects/ajw/wharton/hyhuang/cgm`

---

## Rule 0 — the two lines that start every session

```
cd /vast/projects/ajw/wharton/hyhuang/cgm
source .venv/bin/activate
```

**Your prompt must change to `(.venv) hyhuang@login01:...`.** If it does not,
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
cd "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf code_vN.tgz proj1/scripts proj1/src proj1/tests proj1/cluster
scp code_vN.tgz betty:/vast/projects/ajw/wharton/hyhuang/cgm/
```

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
python -c "import sys; sys.path[:0]=['proj1/scripts','proj1/src']; import guidance_sweep as g; print(g.COMPARE_SET); print(len(g.plan_compare_cells(['mu','alpha','gap'])),'compare cells')"
```

Expect `ALL PASS` three times, then the arm list and **`258 compare cells`**.
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
scp betty:/vast/projects/ajw/wharton/hyhuang/cgm/cells.tgz "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1/"
```

For the full run: `ls results/full/n5000/seed*/*__full.json | wc -l` (want 64),
then `python proj1/scripts/full_run_table.py`. It refuses to print a partial or
inconsistent run and says what is missing. To pull it back:
`tar -czf full.tgz results/full`.

On the laptop, `tar xzf cells.tgz --skip-old-files` keeps any file that
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

## The transfer (TFG's EDMsecond) — the same five steps

Its own job file, `proj1/cluster/transfer_run.slurm`, and its own results tree,
`results/transfer/`. It never touches `results/sweep/` or `results/full/`, so
it can run beside the main full run. Protocol:
[TRANSFER_EXPERIMENT_PLAN.md](TRANSFER_EXPERIMENT_PLAN.md) §9.

### Step 1 — ship the code AND the TFG assets (laptop)

The transfer needs files the main sweep never did: the EDMsecond checkpoint
and TFG's six property networks (plus two network-definition files from the
OC-Flow tree). They travel in their own bundle, built once:

```
cd "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf code_v11.tgz proj1/scripts proj1/src proj1/tests proj1/cluster
scp code_v11.tgz tfg_assets_v1.tgz betty:/vast/projects/ajw/wharton/hyhuang/cgm/
```

`tfg_assets_v1.tgz` is already built in the project root (47 MB, 32 files).
Next code change: `code_v12.tgz`, never `v11` again.

### Step 2 — verify (Betty, login node, seconds)

Rule 0 first, then:

```
tar tzf code_v11.tgz | head -3
tar xzf code_v11.tgz
tar xzf tfg_assets_v1.tgz
python proj1/scripts/fetch_tfg_assets.py --verify
python proj1/scripts/transfer_sweep.py --dry-run
```

Expect paths starting `proj1/`, then **`READY`** at the end of the hash
table (every md5 matches the one pinned on the laptop), then
**`cells total 258 | done 0 | to run 258`** (216 before `tfg` joined the
set on 23 Sep; a `216` means Betty has an old tarball). `HASH MISMATCH` or
`NOT READY` means stop.

### Step 3 — submit the compare stage (n = 512)

```
CMP=$(sbatch --parsable --array=0-11 --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm)
S1=$(sbatch --parsable --dependency=afterany:$CMP --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm)
S2=$(sbatch --parsable --dependency=afterany:$S1 --export=ALL,STAGE=compare proj1/cluster/transfer_run.slurm)
echo $CMP $S1 $S2
```

Twelve array tasks (property × target × arm group, 29 or 14 cells each, ~1–1.5 h)
plus two insurance sweepers that finish anything a task's time guard left.
Group A is unguided, plug, tmpd, lgd_mc, tfg; group B is btvg, btvg_var.

### Step 4 — check

```
grep -h "cells to run\|preflight passed\|STOP" logs/transfer-*.out
python proj1/scripts/transfer_sweep.py --dry-run
```

`to run 0` means the compare stage is done.

### Then the full stage (dist, n = 5000, 3 seeds)

Only once the compare stage says `to run 0`. First the freeze, by hand, to
read the table (seconds; reads JSONs only):

```
python proj1/scripts/transfer_sweep.py --stage freeze
```

A table ending in `FR3a strengths ...` means ready. `INCOMPLETE` means wait.
Then:

```
FT=$(sbatch --parsable --array=0-23 --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm)
F1=$(sbatch --parsable --dependency=afterany:$FT --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm)
F2=$(sbatch --parsable --dependency=afterany:$F1 --export=ALL,STAGE=full proj1/cluster/transfer_run.slurm)
echo $FT $F1 $F2
```

Twenty-four tasks (18 primary: property × seed × arm group; 6 secondary). The
first task freezes the strengths once into
`results/transfer/full/n5000/frozen_q90.json`; every other task reuses it.

### Step 5 — collect (laptop)

```
scp -r betty:/vast/projects/ajw/wharton/hyhuang/cgm/results/transfer results/
python proj1/scripts/full_run_table.py --root results/transfer/full/n5000
python proj1/scripts/dist_report.py --backend tfg --dir "results/transfer/full/n5000/seed*" --frozen results/transfer/full/n5000/frozen_q90.json
```

**`--backend tfg` is not optional.** Without it `dist_report.py` builds its
ladder with OUR f_B and OUR δ; it now refuses transfer cells rather than
mis-scoring them, but the flag is what makes it work.

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
cd "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
tar --exclude='__pycache__' -czf code_v12.tgz proj1/scripts proj1/src proj1/tests proj1/cluster
scp code_v12.tgz betty:/vast/projects/ajw/wharton/hyhuang/cgm/
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

Expect paths starting `proj1/`, then **`62 gates, 0 failed`**, then
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
scp betty:/vast/projects/ajw/wharton/hyhuang/cgm/tfg_cells.tgz "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1/"
```

Claude extracts it into a scratch folder, checks nothing overlaps, merges,
and reads it with
`python proj1/scripts/full_run_table.py --frozen results/full/n5000/frozen_q90_tfg.json`.
