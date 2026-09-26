# Read this first

You are picking up one narrow job from another Claude session running on the
PARCC cluster. That session has the full project context; you do not need it.

**Your job: train one model, report five numbers, stop.**

Everything below is already decided and tested. Please do not relitigate it —
if something looks wrong, say so in your report rather than fixing it silently,
because the two sessions must stay in agreement about what was run.

---

## The one thing to run

```bash
nvidia-smi                 # pick a genuinely idle GPU
python bench.py cuda       # 30-second probe — ALWAYS before the long run
./train_blade.sh <gpu>     # benchmarks, then trains under nohup
tail -f train.log
```

Read `HANDOFF.md` for the protocol. Read it before changing any flag.

---

## What this model is for, so you calibrate effort correctly

The project compares **guidance methods** — inference-time steering rules
applied to a *frozen* generative model. This model is the frozen substrate.
It is **not** the contribution and is not being compared against other
generative models.

Every guidance method reads the same weights, same sampler, same seeds. So the
comparison is internally valid at any base quality. **A better base changes
absolute sample fidelity, not the ordering of arms.** Do not gold-plate it.

---

## Decisions already made — do not undo

- **Epochs, not steps.** The loop reshuffles with `randperm` each epoch.
  An older copy used `torch.randint` (i.i.d. with replacement), which left
  ~37% of data unseen per epoch-equivalent. If you see `randint` in the
  training loop, your copy is stale — `git pull`.
- **489 epochs** = 328 steps/epoch x 256 batch = 41.1M sample-views, chosen to
  match the published Linear FM budget (157,440 steps, 40.3M). Not arbitrary.
- **Channel order is ACGT.** Verified Watson–Crick symmetric
  (A .2728 ~ T .2720, C .2276 ~ G .2276). `gc_soft` reads channels 1:3.
- **4 of 83,726 training sequences were dropped**, not imputed. They carried
  ambiguous all-zero rows. A 0.25-filled row reads GC=0.5 under the
  differentiable `gc_soft` but argmaxes to A (GC=0) under the evaluated
  `gc_hard` — a 0.1 disagreement between the guided property and the scored
  one. After dropping, they agree to exactly 0.0. Keep it that way.
- **Cosine LR anneals over the full run.** Killing it early leaves LR
  un-annealed. Change `--epochs`; do not ctrl-C at a step count you like.

---

## Sanity check on load — reproduce these or stop

```
train (83722, 500, 4)   val (10505, 500, 4)   test (10434, 500, 4)
GC mean 0.4552   sd 0.0552
gc_soft == gc_hard: max|d| = 0.00e+00
```

If any differs, something is wrong with the data. Report it; do not work around it.

---

## Report these five, and nothing needs to be prettier than plain text

1. final validation loss, and the step/epoch it was reached at
2. whether early stopping fired, or it ran the full 489
3. wall time, and the `bench.py` throughput in sample-views/sec
4. **generated GC sd, against the real 0.0552** — this one matters most
5. anything in the run that surprised you

On (4): the guidance method controls the *spread* of GC across a batch. If the
base cannot reproduce the data's spread, the setpoint grid is aimed at the
wrong place, and the other session needs to know before it runs the sweep.

Then hand back `fm_m2_dfb500.pt`. The guidance sweep runs elsewhere.

---

## Do not

- **Do not tune the base against guidance results.** The base is frozen before
  any guidance is applied. Selecting it by downstream guidance performance
  would leak the comparison and invalidate every arm. This is the one rule
  that, if broken, silently destroys the project's main claim.
- **Do not "improve" the method.** No new losses, no architecture search, no
  extra tricks. A different model is worse than a modest expected one.
- **Do not change the data, the split, or the crop.** The split is the
  published one so the numbers sit on the same test set as the baselines.
- **Do not run the guidance sweep here** unless explicitly asked.

---

## blade-specific hazards

- **Shared box, no scheduler.** 6x RTX A6000. Nothing reserves a GPU and
  nothing protects your process. Check `nvidia-smi` immediately before
  launching, and again if throughput drops — someone may have landed on you.
- As of 26 Sep 02:05, GPUs 0–1 and 4–5 were busy (two users), 2–3 free.
  Do not trust that; re-check.
- `nohup` survives logout but not OOM and not a reboot.
- Checkpoints are written on **every validation improvement**, so a kill still
  leaves a usable model. Early stopping ending before 489 epochs is correct
  behaviour, not a failure.

---

## If throughput disappoints

The estimate was 1.5–6.5 h for the full run, from a CPU measurement of 170
views/s and an assumption about GPU efficiency that was never verified. If
`bench.py` comes in at the slow end, that is expected — report the number
rather than compensating by cutting epochs. If it comes in fast and you have
hours to spare, `--hidden 256 --layers 12` (4.87M params) is a sanctioned
upgrade; anything else is not.
