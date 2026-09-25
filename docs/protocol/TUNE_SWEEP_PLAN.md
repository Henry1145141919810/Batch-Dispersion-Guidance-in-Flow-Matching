# Joint strength x start-time tuning, then the full run — the plan

**25 September 2026. Pre-registration. Nothing here has been run.** Written for
Henry, to check before the chain is queued.

Everything is queued in ONE paste and runs unattended. Stage 2 is chained to
stage 1 with `afterok` on the freeze, so nothing has to be resubmitted by hand.

---

## 1. What this answers, and why it is not a repeat of v2

v2 froze each arm's **strength** on an n = 512 screen at a **single** guidance
window (`t_min_guide = 0.5`). It never swept the two together. But the
project's best-supported claim is that **when you guide matters more than how**
— `plug` measured at −8.2σ on alpha at `t_min = 0.05` and +4.5σ at 0.5 — and
that claim came from a window sweep done at one fixed strength. Neither screen
could see an interaction between the two knobs.

This run sweeps them **jointly**, on our own FM base, for all seven arms, and
freezes **two** operating points per (property, arm):

| pick | rule | what it is for |
|---|---|---|
| **floor** | best in-band among cells with molecule stability ≥ 0.9 × unguided | the reportable operating point, v2's chemistry budget unchanged |
| **open** | best in-band, **no** chemistry constraint | the ceiling the method reaches when chemistry is allowed to collapse — so the floor's cost is *measured*, not asserted |

`open` is **not a recommendation**. It exists because "the floor costs us X" has
been asserted repeatedly in this project and never measured at full scale.

---

## 2. Settings, and every deliberate departure from v2

| setting | value | same as v2? |
|---|---|---|
| generator | `fm_last.pt`, md5 `a190ac83` | yes — every cell this project has ever run |
| target | q90, fixed per property: mu 4.6627 D, alpha 85.05 Bohr³, gap 0.3162 Ha | yes |
| arms | unguided, plug, tmpd, lgd_mc, tfg, **btvg, btvg_var** | v2's first pass had 5; btvg and btvg_var were its deferred second pass and are included here from the start |
| **in-band δ** | **2 × f_B's MAE on val molecules near q90** (0.16992 / 0.46754 / 0.00739) | **NO — this is the change you asked for** |
| screen n | 1000, one seed (20260925) | v2 screened at 512 |
| strengths | 0.01, 0.05, 0.25, 0.5, 1, 2, 4, 8, 16 (9) | v2 used 7, topping out at 4 |
| **windows** | **0.05, 0.5, 0.75 — swept jointly with strength** | **NO — v2 fixed this at 0.5** |
| full-run n | **4000** × 3 seeds (20261001/2/3) | v2 used 5000; trimmed for the 15 h budget (§5) |
| sampler | 100-step Euler, velocity clip 1.0, sizes from `val` | yes |
| sidecars | per-molecule, with coordinates | yes |

### The δ is not blind, and that must be on the record

This run adopts the local δ **after** its effect on the v2 headline has been
seen. The rescoring is committed (`92113c5`): on gap, `plug` vs unguided on the
decoded metric with the cluster-robust se goes **+2.75 (fails V4's z ≥ 3) under
the pre-registered δ, +3.04 (clears) under this one.** No other headline
comparison changes sign or crosses the bar.

So this is not a blind pre-registration of δ. It is a deliberate adoption of a
δ whose effect is partly known, on the argument that f_B's error near the
target is the right tolerance for a target-directed run — an argument that
stands on its own and was made before the rescoring. It is recorded here so a
reader can discount it, and the arm it helps (`plug` on gap) should be read
with that in mind.

**The status doc is now stale on this point.** `SCOPE_FM_GUIDANCE_STATUS.md` §6
says the local δ is "used for: the sensitivity analysis only" and the
pre-registered δ for "every reported result". That remains true of the v2
tables; it is not true of this run.

### δ is not only a scoring knob — it changes what is sampled

BTVG's τ is δ/1.96 (`guidance_sweep.arm_kwargs`), so `btvg` and `btvg_var`
cells sampled under the pre-registered δ and under this one are **different
experiments**. Every cell records `delta` and `delta_source`, and
`freeze_tune.py` **refuses** a tree that mixes two. This is why the whole sweep
is re-run rather than reusing the q90 cells already on disk — those were
sampled under the old δ, at n = 512, at one window.

### Selection uses the CONTINUOUS in-band, and that is a deliberate choice

v2's rule, unchanged. V4 warns the continuous metric flatters arms that push
the continuous atom-type features, and the v2 full run showed two of three
headline comparisons crossing z = 3 on continuous but not decoded. So the
freeze **also** computes what each pick would have been on the decoded metric
and flags every disagreement. Changing the selection metric is a protocol
change; it is not made silently here, and the flag is the evidence for making
it deliberately later.

---

## 3. Stage 1 — the sweep (`tune_sweep.slurm`)

**489 cells** = 3 unguided (one per property) + 6 guided arms × 3 properties ×
9 strengths × 3 windows, at n = 1000.

`unguided` gets one cell per property, not 27: it applies no guidance, so
neither knob can change its output.

**Ordering is chosen so a truncated job is still usable.** Pass 1 writes the
three unguided cells — without them the chemistry floor is undefined and
nothing can be frozen at all. Pass 2 is every arm's strength curve at
`t_min = 0.5`, which reproduces the comparison v2 already made. Pass 3 is the
other two windows.

**Array layout:** `--array=0-8`, task = 3 × window_index + property_index. Each
task is one (property, window) slice, about 2.6 h — inside the 4-hour QOS
window. The sweep is resumable: a finished cell is a file, and a file on disk
is never recomputed, so the insurance link finishes whatever a cut-off task
left.

---

## 4. Stage 2 — the full run (`tuned_full_run.slurm`)

Runs the v2 protocol at **both** frozen operating points, at n = 4000 × 3
seeds, with per-molecule sidecars.

**Cells are de-duplicated.** When an arm's best in-band already clears the
chemistry floor, its two picks coincide and the cell is sampled **once**;
`frozen_tune.json` records which file each set points at. Worst case (the two
picks differ for all six guided arms) is 13 cells per (property, seed) = 117
cells; best case 7 per group = 63.

**Array layout:** `--array=0-8`, task = 3 × seed_index + property_index. About
2.4 h per task in the worst case.

**The freeze is a separate job, and the full run is chained to it with
`afterok`.** If `freeze_tune.py` refuses — missing cells, no unguided
reference, two deltas in one tree, a degenerate floor — the full run never
starts and no GPU time is spent on a table that could not be interpreted.

---

## 5. The budget, and how the array is split

Costed from measured numbers: per-arm NFE from the v1 full run, and
`v2_run.slurm`'s measured 58 min for a 23.5-unit task at n = 5000.

**Cost scales with the guided fraction, which is why the array is split by
window AND by arm.** Guidance runs only for t ≥ `t_min_guide`, so at
`t_min = 0.05` a cell guides 95 of 100 steps and costs ~3.7× the same cell at
0.75. A whole (property, window) task at 0.05 needs **4.6 h — over the 4-hour
wall.** Each slice is therefore halved into two cost-balanced arm groups:

| | group A (`unguided`, `lgd_mc`, `tfg`, `plug`) | group B (`btvg`, `btvg_var`, `tmpd`) |
|---|---|---|
| t_min = 0.05 | 2.40 h | 2.18 h |
| t_min = 0.5 | 1.27 h | 1.15 h |
| t_min = 0.75 | 0.64 h | 0.57 h |

**Sweep: 24.6 GPU-h over 18 tasks, longest 2.40 h.**

**Full run: 9.9–36.9 GPU-h, and the range is the point.** Its cost depends on
where the freeze puts the picks — the thing the sweep exists to discover:

| if every pick lands at | GPU-h | per task (18 tasks) |
|---|---|---|
| t_min = 0.75 | 9.9 | 1.10 h |
| t_min = 0.5 | 19.5 | 2.17 h |
| t_min = 0.05 | 36.9 | **4.10 h — would not fit undivided** |

Split into 18 tasks the worst case is ~2.1 h per task. **Total: 35–62 GPU-h.**

**Wall clock.** At 18-way concurrency, ≈ 2.4 h (sweep) + 0.3 h (freeze) +
2.1 h (full run) ≈ **5 h plus queueing**. At 5-way, ≈ 13 h. Below about
4-way it will not fit 15 h, and the knobs below are the answer.

**n = 4000 rather than the protocol's 5000** is part of that margin. Pooled
over three seeds that is 12,000 molecules against v2's 15,000, so the iid se of
an in-band proportion near 0.05 moves from 0.0018 to 0.0020 — a real but small
loss. (Both are iid; the run's own design effect of ~1.03–1.15 applies on top,
and the reporter prints the cluster-robust se beside the iid one.)

**If the queue is deep:** `FULL_N=3000` takes the full run's worst case to
27.7 GPU-h; `TUNE_N=512` takes the sweep to ~12.6 GPU-h. Both are set on the
submit line. `FULL_N=5000` restores the protocol number for ~9 GPU-h more.

## 6. What this run can and cannot settle

**Can:** which (strength, start-time) each arm is best at under a fixed
chemistry budget, on our own base, with all seven arms including ours; what
that budget costs each arm; and whether the start time interacts with strength
at all — which no screen so far could see.

**Cannot, and must not be claimed:**

- **It is one screen seed.** The picks are selected on n = 1000 at seed
  20260925 and scored at n = 4000 × 3 disjoint seeds, so selection cannot
  inflate the reported number — but a different screen seed could pick
  different cells. v2 already showed this bites: `lgd_mc` cleared the floor on
  the n = 512 screen and fell below it at n = 5000.
- **Ties will be ties.** At n = 1000 the se of an in-band proportion near 0.05
  is 0.0069, so adjacent grid points are frequently not distinguishable. The
  freeze records the whole grid, and the reporting must show the runner-up
  comparison, not just the winner.
- **`open` is a ceiling, not an operating point.** Cells that fail the floor
  fail it by producing chemistry we would not ship.
- **δ is f_B's error on real molecules.** In-band scores generated ones, only
  ~35–40 % of which are molecule-stable; f_B's error on that population is
  unmeasured. This affects the pre-registered δ and this one equally.
- **Nothing here compares base models.** That is the EquiFM work in another
  session; it must use these same three δ values, since δ is a property of the
  oracle and not of the generator.

---

## 7. Submitting — the whole chain, one paste

### First: ship the δ file, which the code tarball does NOT carry

`tar ... proj1/scripts proj1/src proj1/tests proj1/cluster` ships code only, so
`results/local_fb_mae.json` never reaches Betty. It is committed and
deterministic, so **copy it rather than regenerating it** — that pins the exact
δ this plan was written against:

```
scp results/local_fb_mae.json betty:/vast/projects/ajw/wharton/hyhuang/cgm/results/
```

Regenerating on Betty would also work (the same f_B gives the same numbers),
but `local_fb_mae.py` runs f_B over 17,748 `val` molecules for each of three
properties — **that is not a login-node job.** If you must, give it a GPU:

```
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=48G \
     --time=00:20:00 python proj1/scripts/local_fb_mae.py
```

Either way the run re-checks it: `--delta-json` refuses a file whose
`global_mae` is not the f_B this box actually loads, so a stale or
foreign δ file cannot silently move the in-band bar (or BTVG's τ).

### Then: the chain

Run from `$PROJ` with the venv active. Each `sbatch` prints the id the next
line depends on, so nothing has to be typed twice.

```
A=$(sbatch --parsable --array=0-17 --export=ALL proj1/cluster/tune_sweep.slurm)
B=$(sbatch --parsable --dependency=afterany:$A --export=ALL proj1/cluster/tune_sweep.slurm)
C=$(sbatch --parsable --dependency=afterany:$B --export=ALL proj1/cluster/tune_sweep.slurm)
D=$(sbatch --parsable --dependency=afterany:$C --export=ALL proj1/cluster/tune_freeze.slurm)
E=$(sbatch --parsable --dependency=afterok:$D --array=0-17 --export=ALL proj1/cluster/tuned_full_run.slurm)
F=$(sbatch --parsable --dependency=afterany:$E --export=ALL proj1/cluster/tuned_full_run.slurm)
echo "sweep $A, links $B $C, freeze $D, full $E, link $F"
```

**Two insurance links after the sweep, not one.** A single no-array job walks
every slice inside one 4-hour window and cannot finish 24.6 GPU-h alone; it is
a finisher, not a substitute. Two links close what an interrupted array left.

**`afterok` on the freeze is load-bearing.** If `freeze_tune.py` refuses, the
full run never starts and no GPU time is spent on a table that could not be
interpreted.

### If the freeze refuses

It refuses on a missing unguided cell, missing planned cells, two deltas in one
tree, a degenerate floor, or a non-finite metric — and names which. **The
already-queued stage 2 (`$E`, `$F`) will then sit as `DependencyNeverSatisfied`
and be purged, so it must be resubmitted by hand** after the cause is fixed:

```
G=$(sbatch --parsable --export=ALL proj1/cluster/tune_sweep.slurm)
H=$(sbatch --parsable --dependency=afterany:$G --export=ALL proj1/cluster/tune_freeze.slurm)
I=$(sbatch --parsable --dependency=afterok:$H --array=0-17 --export=ALL proj1/cluster/tuned_full_run.slurm)
sbatch --dependency=afterany:$I --export=ALL proj1/cluster/tuned_full_run.slurm
```

To freeze from a deliberately partial sweep instead, run `freeze_tune.py
--allow-partial` by hand; the result is labelled partial and records what was
missing.

### Reading the results

```
python proj1/scripts/tuned_full_table.py   --root results/full/v2_tuned/n4000   --frozen results/tune/n1000/frozen_tune.json   --md-out docs/results/TUNED_FULL_RUN_RESULTS.md
```

`full_run_v2_table.py` **cannot** read this tree and is not meant to: it
requires one cell per (property, arm) and one shared `t_min_guide`, which two
operating points and per-arm start times both violate. Those refusals are right
for v2, so this run has its own reader.

## 8. Files

| what | where |
|---|---|
| sweep stage, both plans | `proj1/scripts/guidance_sweep.py` (`--stage tune`, `--stage tuned_full`, `--delta-json`, `--windows`) |
| the freeze and the sweep table | `proj1/scripts/freeze_tune.py` |
| δ definition and values | `proj1/scripts/local_fb_mae.py` → `results/local_fb_mae.json` |
| gates (48, closed-form) | `proj1/tests/test_tune.py` |
| cluster jobs | `proj1/cluster/tune_sweep.slurm`, `tune_freeze.slurm`, `tuned_full_run.slurm` |
| the reporter | `proj1/scripts/tuned_full_table.py` → `docs/results/TUNED_FULL_RUN_RESULTS.md` |
| sweep cells | `results/tune/n1000/*__tune.json` |
| frozen picks | `results/tune/n1000/frozen_tune.json` |
| sweep table | `docs/results/TUNE_SWEEP_TABLE.md` |
| full-run cells | `results/full/v2_tuned/n4000/seed<S>/*__tuned.json` |

Nothing above touches `results/sweep`, `results/full/v2`, or the EquiFM assets.
