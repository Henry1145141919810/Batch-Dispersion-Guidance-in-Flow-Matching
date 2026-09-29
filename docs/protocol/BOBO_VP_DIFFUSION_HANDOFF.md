# Bobo — run our QM9 diffusion base model through v3 (`--backend vp`)

**28 September 2026.** You have the trained QM9 diffusion model; we have the
backend wired and gated. This is what to run, in order.

**Priority order, and please do them in this order:**

| | | time |
|---|---|---|
| **1** | **Check your checkpoint against our protocol** (§1) | 1 min |
| **2** | **Smoke it** (§2) | 2 min |
| **3** | **Run the 18 cells** (§3) | ≈1.5 h |
| **4** | **Send everything back** (§5) | — |

**Step 1 is not a formality.** If your model was trained on a different split,
width, depth or budget from our flow-matching model, the comparison the paper
needs does not exist and the cells would be misleading rather than merely
wrong. Two minutes now saves the run.

---

## 0. What this is for, briefly

The v3 run has a diffusion backend called **`edm`** — TFG's released
`EDMsecond`, borrowed and frozen. It is **not ours**, and it cannot answer the
question the paper asks, because it differs from our FM model in data,
architecture and budget all at once.

Our paper has a row for "VP, trained here" (`tab:fmvd`, `paper/main.tex:141`)
in which every cell is `\pend`, and `paper/main.tex:297` says so in print:
*"Our VP & pending benchmark & no result row substituted"*.

**Your model fills it** — provided it matches the flow-matching model on
everything except the generator family. That is what makes `fm` vs `vp` a
controlled comparison instead of two unrelated numbers.

---

## 1. FIRST — check your checkpoint against the protocol

```bash
cd /scratch2/boboli/projects/cis6270-project1-group2

python -c "
import torch,sys
st=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
print('family:',st.get('family'))
print('epoch :',st.get('epoch'))
print('args  :',st.get('args'))
" <path-to-your-checkpoint.pt>
```

Compare against what `weights/fm_ema.pt` records — these are the values the
whole comparison rests on:

| field | required | if it differs |
|---|---|---|
| `family` | **`vp_diffusion`** | 🔴 the loader refuses; it is not a VP model |
| `split` | **`train_a`** | 🔴 the loader refuses; the generator saw `f_B`'s data |
| `hidden` | **256** | 🔴 the loader refuses; not matched to `fm` |
| `layers` | **8** | 🔴 the loader refuses; not matched to `fm` |
| `epochs` / `epoch` | **1500** | 🟠 runs, but **tell us** — see below |
| `batch` | **256** | 🟠 runs, but tell us |
| `ema` | **0.9999** | 🟠 runs, but tell us |
| `seed` | **20260918** | 🟠 runs, but tell us |

**Red rows are refused automatically** — the loader exits 1 and names the
field. You do not have to police those.

**Amber rows are not yet checked in code and are the ones I need you to read
off and report.** A different seed is defensible if we declare it. A different
epoch budget, batch or EMA means `fm` vs `vp` is no longer matched on
"everything but the family", and we would have to say so in the paper rather
than discover it later.

⚠️ **If your checkpoint is a *selected* one, its `epoch` can legitimately be
below 1500** — our trainer selects on highest validation atom stability at
NFE 100, so a good 1500-epoch run routinely delivers epoch ~1425. What matters
is that the run *trained* to 1500. If you have a `_last.pt` alongside it,
report that epoch too; it is the one that shows completeness.

**If anything in the red rows differs, stop and message me — do not run §3.**
Appendix A has the training recipe if it turns out we need a fresh one.

### Where to put it

The sweep finds the checkpoint on its own if you place it at
`proj1/checkpoints/diff.pt`. Otherwise pass `--vp-ckpt <path>` to every command
below. Resolution order is `proj1/checkpoints/diff.pt` → `diff_last.pt` →
`weights/vp_ema.pt`.

---

## 2. Smoke it — 2 minutes, do not skip

```bash
mkdir -p logs
python proj1/scripts/transfer_sweep.py --stage v3 --backend vp \
    --props mu --arms unguided,plug --n 2000 --batch 500 --seed 20261001 --preflight
```

Want, in this order:

```
base model: VP diffusion (ours)  md5 <...>  nf=256 layers=8  grid=uniform
property pair: ours (weights/f_{A,B}_<p>.pt)
  mu   guide MAE ... | oracle MAE ... | delta 0.17541 | scale check ...
  [  1/  2] mu  unguided  q50  w=1  t=0.5  ...
  [  2/  2] mu  plug      q50  w=1  t=0.5  ...
preflight OK: 2 arms ran end to end, nothing written
```

**`delta` must read 0.17541.** That is our property pair's pre-registered band
for mu. A different number means the wrong pair loaded and nothing downstream is
comparable.

If the acceptance guard refuses, it names the field and why. **Send me the
message rather than working around it** — every one of those refusals is
protecting the comparison, not being fussy.

---

## 3. THE RUN — 18 cells, ≈1.5 h

### The protocol. Pre-registered. Please do not vary any of it.

| axis | value |
|---|---|
| **arms** | **`unguided` and `plug`. Those two, nothing else.** |
| properties | mu, alpha, gap |
| seeds | **20261001, 20261002, 20261003** |
| n | **2000**, in 4 batches of 500 |
| strength | **w = 1, a single value. No sweep.** |
| target | q50 |
| f_A / f_B | **ours** — `weights/f_{A,B}_<p>.pt` |
| ablation | **none** — `vp` sits it out, exactly as `edm` does |
| **cells** | **2 × 3 × 3 = 18** (36,000 molecules) |

Two arms because `vp` mirrors `edm` exactly, and that is what makes
`vp`-vs-EDMsecond a legal head-to-head. The rest of the arm set answers a
different question and is not part of this job.

```bash
mkdir -p logs
for P in mu alpha gap; do
  for S in 20261001 20261002 20261003; do
    CUDA_VISIBLE_DEVICES=0 python -u proj1/scripts/transfer_sweep.py --stage v3 --backend vp \
      --props $P --arms unguided,plug --n 2000 --batch 500 --seed $S --per-mol \
      2>&1 | tee -a logs/vp_sweep.log
  done
done
```

Everything else is argparse default, and that is exactly what the cells record:
`--steps 100`, `--solver euler`, `--grid uniform`, `--tau-max-guide 0.5`,
`--clip 1.0`, `--k-delta 2.0`.

Sequential on one GPU is fine — the `edm` equivalent took a recorded **1.48 h
for the same 18 cells on your A6000**, longest cell 7 min. If you would rather
spread it across your three cards, the `blade_runs/lane.sh` queue pattern is
already there, but it is not worth the setup for 1.5 h.

> ⚠️ **Do not pass `--grid gamma`.** The gamma grid needs a schedule object to
> invert, and `vp` passes `noise_schedule=None` — which *is* the linear-beta
> schedule the model was trained under, not an omission. It raises on the first
> batch of every cell, writes a `.failed` marker for each, and exits 1 within
> minutes having produced nothing.

> ⚠️ **Do not use `submit_v3.sh`.** `vp` is deliberately not in its chain, and
> that script would re-plan `fm`, `equifm` and `edm` on top of this.

---

## 4. Check and build the table

```bash
ls results/v3/vp/v3/n2000/seed20261001/*.json | wc -l   # 6 (2 arms x 3 props)
python proj1/scripts/v3_sanity.py                       # takes no arguments; must pass

python proj1/scripts/v3_table.py --backend vp --stage v3 --n 2000 \
    --seeds 20261001,20261002,20261003 \
    --md-out docs/results/V3_RESULTS_vp.md
```

Three things that will bite otherwise: `--n` is **required** and has no default
(v3 pre-registers no cell size); without `--md-out` the table only goes to
stdout and no file is written; and use `--backend vp`, **not `--both`** —
`--both` walks only the chain's three backends by design.

Count `*.json`, not every file — `--per-mol` writes a `.permol.pt` sidecar
beside each cell, so a bare `ls | wc -l` reads 12.

---

## 5. Send everything back — I will tidy it up

**Send a tarball, not a commit.** `proj1/checkpoints/*.pt` and `*.permol.pt` are
both gitignored; pushed through git, the weights and all 18 sidecars are
silently dropped.

### 5.1 The model

- [ ] The checkpoint itself, **plus its md5** (`md5sum <file>`)
- [ ] `_last.pt` too, if you have one — it is what shows the run completed
- [ ] `diff_history.json` (or your trainer's equivalent), the full per-epoch record

### 5.2 How it was trained — the run detail

Whatever of this you have. If it was not our `train_diffusion.py`, this matters
more, not less, because I have to describe it accurately in the paper.

- [ ] The **training log**
- [ ] Which **script and flags** — the exact command line
- [ ] **Split**, and how the halves were drawn
- [ ] **Epochs trained**, batch, lr schedule, EMA decay, seed
- [ ] **Which GPU**, and **s/epoch + total wall clock** — nobody has timed this
      architecture on an A6000 and it is worth recording
- [ ] Anything you changed from our defaults, however small

### 5.3 The stats

- [ ] **Selected checkpoint's `atom_stab`, `mol_stab`, `val_loss`**, and at
      which epoch — this is the line our trainer prints as
      `SELECTED checkpoint: atom_stab … mol_stab … val_loss …`
- [ ] The **selection rule** you used (ours is highest validation atom
      stability at NFE 100, ties broken by validation loss)
- [ ] Final/best training and validation loss

### 5.4 The run output

- [ ] `results/v3/vp/` — **all 18 cells**, including the `.permol.pt` sidecars
- [ ] `docs/results/V3_RESULTS_vp.md`
- [ ] `logs/vp_sweep.log`

### 5.5 And tell me in plain words

- [ ] Anything that **surprised** you, failed, or that you worked around
- [ ] Whether any amber row in §1 differs from the protocol

---

## 6. Three things not to let through

1. **Never call `edm` ours.** It is TFG's borrowed EDMsecond. If a table, slide
   or caption says otherwise, that is a bug.
2. **`vp` vs `equifm` is not comparable on `in_band`.** Different property
   pair, different band width (mu: 0.17541 ours against 0.15675 TFG's).
   `v3_table.py` refuses to pool them — do not do by hand what it refuses to do
   for you.
3. **Neither `vp` nor `edm` is EDM's published unconditional number.** Both run
   through *our* 100-step probability-flow ODE, not EDM's native 1000-step
   ancestral SDE. Do not quote either as a reproduction.

---

## Appendix A — only if §1 says the checkpoint does not match

The trainer is in the repo and pre-registered. ≈12 GPU-h on a B200
(28.5 s/epoch measured); **unmeasured on an A6000, budget 13–20 h.**

**Clear the old checkpoints first — and start with the `cd`.** From the wrong
directory the `mv` silently does nothing and the check still prints `clear`.
There are two guards and the `--resume` one fires first; `--fresh` does not
help, because `train_diffusion.slurm` hardcodes `--resume` and the resume guard
is checked at `train_diffusion.py:293`, before `--fresh` at `:340`.

```bash
cd /scratch2/boboli/projects/cis6270-project1-group2
mkdir -p proj1/checkpoints/old_split logs
mv proj1/checkpoints/diff* proj1/checkpoints/old_split/ 2>/dev/null
ls proj1/checkpoints/diff* 2>/dev/null && echo "STILL THERE - DO NOT SUBMIT" || echo "clear"
```

Blade, one continuous run — `--max-minutes` defaults to `0.0`, which disables
the time guard, so there is no chaining and no 4-hour cap:

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u proj1/scripts/train_diffusion.py \
    --epochs 1500 --batch 256 --hidden 256 --layers 8 \
    --split train_a --tag diff --resume \
    > logs/vp_train.log 2>&1 &
```

Use one GPU — it is a single model and the trainer is not distributed. At
epoch 1 the log must print `parameters: 3753229  hidden=256 layers=8
ema=0.9999` and `VP diffusion on train_a: 51527 train, 17748 val`. If either
differs, stop.
