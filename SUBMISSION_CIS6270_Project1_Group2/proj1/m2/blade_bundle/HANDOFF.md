# Modality 2 base model — blade handoff

**Audience: an agent or collaborator working on `blade.seas.upenn.edu` with no prior context.**

Goal: train ONE unconditional flow-matching model on DNA enhancer sequences,
then stop. This model is a **frozen substrate**, not a contribution.

---

## 1. Why this model exists

The project compares **guidance methods** — inference-time steering rules applied
to a frozen generative model. Guidance has no parameters and never trains. So the
comparison is controlled by construction: every method reads the *same weights*,
runs the *same sampler* at the same NFE, on the *same seeds*, scored by the *same*
metrics. Only the steering rule varies.

Our contribution is **BDG (batch-dispersion guidance)**, which servos the
across-batch variance of a predicted property to a setpoint:

```
num_i = (y - F_i) - eta * e * (F_i - Fbar),     e = (V_b - tau^2) / tau^2
```

`V_b` is the variance of the property across the batch. This reduces
algebraically to an adaptive weight `w_eff = 1 + eta*e`; `w_eff < 0` is a
*widening* regime that pushes samples apart.

**None of that depends on base-model quality.** A better base changes absolute
sample fidelity, not the ordering of arms. Do not over-invest here.

---

## 2. What to run

```bash
tar xzf blade_bundle.tgz && cd blade_bundle
python bench.py cuda            # 30s throughput probe — ALWAYS run this first
./train_blade.sh 2              # 2 = an idle GPU index; check nvidia-smi
```

`train_blade.sh` benchmarks, then launches training under `nohup`.
Watch with `tail -f train.log`.

### Pick an idle GPU
`nvidia-smi` — the box has 6× RTX A6000 (49 GB each) and is **shared, with no
scheduler**. As of 26 Sep 02:05, GPUs 0–1 and 4–5 were busy; 2–3 were free.
Nothing protects your process from contention or OOM. Re-check before launching.

---

## 3. The data

`dfb500.npz` — the **DeepFlyBrain** corpus on its **official published split**,
stored as `uint8` base indices `[N, 500]`, expanded to one-hot `[N, 500, 4]` at
load. Channel order is **ACGT** (verified Watson–Crick symmetric: A .2728 ≈ T
.2720, C .2276 ≈ G .2276).

| split | n | length |
|---|---|---|
| train | 83,722 | 500 bp |
| valid | 10,505 | 500 bp |
| test  | 10,434 | 500 bp |

GC mean **0.4552**, sd **0.0552**. These are *Drosophila* enhancer regions —
accessibility peaks cut to fixed 500 bp windows, pooled over 81 cell types.

Four of the original 83,726 training sequences carried ambiguous (all-zero)
rows and were **dropped**, not imputed. A 0.25-filled row reads as GC = 0.5 under
the differentiable `gc_soft` but argmaxes to A (GC = 0) under the evaluated
`gc_hard` — a 0.1 disagreement between the guided property and the scored one.
After dropping, the two agree to **exactly 0.0**. Preserve this.

Provenance: `DeepFlyBrain_data.pkl`, sha256 `539426217f24e2…`, 554,306,501 bytes.
Packed to 12 MB here purely for transfer.

---

## 4. Epochs, and why

One epoch is one pass over the training set with a fresh shuffle:

```python
order = torch.randperm(Xa.shape[0], device=dev)      # every sequence once
```

This matches both published enhancer baselines — Dirichlet FM
(`train_dna.py:82-93`) and Fisher Flow (`dna_enhancer_datamodule.py:83`) each
use `DataLoader(shuffle=True)` with `max_epochs` — and our own Modality 1,
which reshuffles with `randperm`.

An **earlier version of this file sampled minibatches i.i.d. WITH replacement**
(`torch.randint`). That is also unbiased, but it leaves roughly **37%** of the
data unseen in any epoch's worth of steps, since (1 - 1/N)^N -> 1/e, and it
defines no epoch boundary to report against published budgets. If you see
`randint` in the training loop, you have an outdated copy.

Note the distinction is *with vs without replacement*, NOT full-batch gradient
descent vs SGD. Both variants are mini-batch at batch 256.

The unit that compares across datasets and against published work is
**sample-views = steps x batch**:

| | steps | batch | views | epochs |
|---|---|---|---|---|
| published Dirichlet FM | 436,240 | 256 | 111.7M | 1,330 |
| published **Linear FM** (our base's method) | 157,440 | 256 | 40.3M | 480 |
| **this run** | 160,392 | 256 | **41.1M** | **489** |

328 steps/epoch x 489 epochs. Chosen to land on the published Linear FM budget.

## 5. Config, and what not to change

```
--hidden 128 --layers 10     1.02M params
--batch 256 --lr 2e-3        cosine anneal over epochs x steps_per_epoch
--epochs 489                 160,392 steps, 41.1M sample-views
--val-every-epochs 5 --patience 10
--crop 500
```

- **Cosine LR anneals over the full `--epochs`.** Cutting the run short leaves
  LR un-annealed. Change `--epochs` rather than killing early.
- **Checkpoints save on every validation improvement**, so a crash or OOM still
  leaves the best model at `fm_m2_dfb500.pt`. Early stopping may end it before
  489 epochs — that is correct behaviour, not a failure.
- If the benchmark shows lots of headroom, `--hidden 256 --layers 12` (4.87M
  params) is a reasonable upgrade. At 83,722 sequences that is 58 params/seq;
  1.02M is 12/seq. QM9's ratio was 28.

---

## 6. When training finishes

Report: final val loss, step reached, whether early stopping fired, wall time,
and the **generated GC sd vs the real 0.0552**. That last number matters most —
BDG controls the spread of GC, so if the base cannot reproduce the data's spread,
the setpoint grid is aimed at the wrong place.

Then hand `fm_m2_dfb500.pt` back. The guidance sweep runs elsewhere.

**Do not** tune the base against guidance results. The base is frozen before any
guidance is applied; choosing it by downstream guidance performance would leak
the comparison and invalidate every arm.

---

## 7. Known-good reference numbers

Reproduce these on load or something is wrong:

```
train (83722, 500, 4)   val (10505, 500, 4)   test (10434, 500, 4)
GC mean 0.4552   sd 0.0552
gc_soft == gc_hard: max|d| = 0.00e+00
one-hot rows sum to 1: max err 0.0e+00
```

Earlier 200 bp run on a 6,126-sequence slice, for orientation only (different
data, not comparable): 20k steps, final loss 0.06485, gc_std 0.06711.
