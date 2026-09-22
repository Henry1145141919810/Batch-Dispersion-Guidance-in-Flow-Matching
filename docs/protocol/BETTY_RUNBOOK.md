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
python -c "import sys; sys.path[:0]=['proj1/scripts','proj1/src']; import guidance_sweep as g; print(g.COMPARE_SET); print(len(g.plan_compare_cells(['mu','alpha','gap'])),'compare cells')"
```

Expect `ALL PASS` twice, then the arm list and **`258 compare cells`**.
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
grep -m1 'cells total' logs/sweepv2-*.out
```

| stage | expected |
|---|---|
| main | 480 total |
| v2 | 342 total |
| compare | 258 total |

**A wrong number here means `STAGE` did not export** — cancel those jobs and
resubmit. The `stage : v2` line in the log header is a hardcoded string; ignore
it and trust the cell count.

---

## Step 5 — COLLECT

```
ls results/sweep/*.json | wc -l
ls results/sweep/*.failed 2>/dev/null | wc -l          # want 0
python proj1/scripts/select_arms.py --stage compare
```

To pull results back to the laptop for analysis:

```
tar -czf cells.tgz results/sweep results/sweep_v2_seed2
```
```
scp betty:/vast/projects/ajw/wharton/hyhuang/cgm/cells.tgz "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1/"
```

On the laptop, extract with `--skip-old-files` so Betty's cells win over any
local copies:

```
tar xzf cells.tgz --skip-old-files
```

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
