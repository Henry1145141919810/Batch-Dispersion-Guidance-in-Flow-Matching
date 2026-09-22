# Which model trains on which data — the protocol, and the evidence for it

**19 September 2026.** Single source of truth. If any file disagrees with this one, this one is
right and the other is a bug.

This exists because the protocol was broken once, by me, and it cost a day of compute. The
mistake was easy to make and the reasoning behind the correct answer was scattered, so it is all
here.

---

## The protocol

| model | trains on | size |
|---|---|---|
| **generator** (flow matching AND diffusion) | `train_a` | 51,527 |
| **$f_A$**, the guide | `train_a` | 51,527 |
| **$f_B$**, the evaluator | **`train_b`** | 51,527 |
| checkpoint selection | `val` | 17,748 |
| final targets | `test` | 13,083 |

In one line: **the generator and the guide share a half; the evaluator gets the other half
alone.**

Concretely:

```bash
python proj1/scripts/train_fm.py        --split train_a    # generator
python proj1/scripts/train_diffusion.py --split train_a    # generator
# the predictors are trained in the main project, not in the shipping packages:
#   train_predictor.py --split train_a --prop mu     # f_A
#   train_predictor.py --split train_b --prop mu     # f_B
```

`prepare_qm9.py` fixes these groups with seed `20260917`. Never change that seed.

---

## Why the generator and $f_A$ may share

Guidance is an optimiser pointed at $f_A$. Given enough strength it will find inputs that fool
$f_A$ rather than molecules that truly have the property — reward hacking. The thing that must be
independent is whatever **checks** the result, which is $f_B$.

If $f_A$ and $f_B$ shared training data they would share blind spots, and the $f_A - f_B$ gap
would go quiet exactly when guidance had found one. The generator sharing with $f_A$ costs
nothing, because the generator is not judging anything.

---

## Why the generator must NOT see `train_b`

Three reasons, in descending order of how well they hold up. The third is the one usually quoted
and is the weakest, which is worth knowing.

### 1. It is what the papers we compare against do — and they had the alternative

This is decisive. TFG's released checkpoint collection contains **three** generator folders:

```
EDM           Shared folder
EDMfull       Shared folder     <- trained on the full 100k
EDMsecond     Shared folder     <- trained on ONE HALF
```

and `scripts/molecule_property.sh` selects:

```bash
generators_path=$base_dir/EDMsecond/generative_model_ema.npy
energy_path=$base_dir/tf_predict_$target/model_ema_2000.npy       # guide
classifiers_path=$base_dir/evaluate_$target/best_checkpoint.npy   # oracle
```

**They had `EDMfull` in the same folder and deliberately chose the half-trained generator.**

The same convention appears across the field: EDM's conditional protocol splits the QM9 training
partition into two 50K halves, training the classifier on one and the generative model on the
other. PropMolFlow and MolGuidance both ship a `use_first_half_training_set` flag. TFG's
`qm9_first_half` / `qm9_second_half` machinery is vendored straight from EDM.

Train on 103k and our stability and property numbers would be produced on **twice the data** of
every number we quote them beside.

### 2. It keeps the independence argument to one sentence

With `train_a` we can write: *"the evaluator trained on molecules neither the generator nor the
guide has ever seen."* A reviewer accepts that and moves on.

With `train_ab` we would write *"the generator saw the evaluator's training data; we believe the
effect is small"* — and then be asked to prove it, which needs a novelty study we have not built.

### 3. The leak argument, which is weaker than it sounds

`PROJECT_GUIDE.md` section 4.1 says a generator trained on both halves produces samples the
oracle scores "optimistically."

**Think that through before repeating it.** If the generator emits a near-copy of a `train_b`
molecule, $f_B$ predicts its property *unusually accurately*, because it memorised it. That makes
the measurement more truthful on those samples, not less. The direction of the bias is arguable.

What genuinely breaks is **calibration**: $\delta$ is derived from $f_B$'s error on held-out data,
and if generated samples are partly memorised-by-$f_B$, its error on them is not exchangeable with
that held-out figure. So $\delta$ stops describing the evaluation we are actually doing.

That is a real objection, but it is subtler than "optimistic scoring," and reasons 1 and 2 carry
the decision on their own.

---

## The mistake, so it is not repeated

On 18 September the generator default was changed to `train_ab` (103,054 molecules), reasoning
that only $f_A$ and $f_B$ need to be disjoint. That reasoning is incomplete — it ignores reasons
1 and 2 entirely — and the justification was put in argparse help text rather than raised as a
decision. It ran for ~250 epochs before an adversarial review caught it.

**What should have happened:** a change to a pre-registered protocol is a decision, not a default.
It gets raised, argued and written down *before* compute is spent.

**Guards now in place:**

- `--split` defaults to `train_a` in both trainers, with the rationale in the help text
- both trainers print a large, unmissable warning if anything other than `train_a` is used
- `--resume` refuses outright if the saved `split` differs from the requested one
- **and so does checkpoint selection**: an existing `<tag>.pt` trained on a different split
  will not be used to seed `best_stab`. Without this the new run inherits a score it cannot
  fairly be measured against, may never write a deliverable, and leaves the OLD model sitting
  on disk under the right filename. Exits non-zero; `--fresh` is the deliberate override.
- this file, referenced from that warning
- the shipping packages' `README_FIRST.md` and `AGENT_BRIEF.md` list the split as a hard invariant

---

## The consequence for our numbers, stated so it is not forgotten

**Our generator sees 51,527 molecules. E(3)-EDM's headline unconditional figure (atom stability
0.987, molecule stability ~0.82) comes from a model trained on ~100k.** Those are not comparable,
and we must not quote ours beside that number without saying so.

The right comparison is a half-trained generator in the conditional setting, which is lower. An
earlier draft of our plan set "atom stability 0.97" as the target by reference to the 100k figure;
that is a stiffer bar than the literature actually clears here, and the target should be restated
against a like-for-like baseline before it is used to judge the run.

If we ever want the "does more data help?" ablation, **`EDMfull` is available off the shelf** in
TFG's collection — no training required.

### The like-for-like bar, measured (20 September)

EEGSDE (Bao et al., ICLR 2023, Table 5, first row) trained an **unconditional** EDM on one 50K
half of the QM9 training set with EDM's conditional recipe and scored 10K samples with EDM's
official code, with hydrogens: **atom stability 98.37, molecule stability 81.74, novelty 82.92.**
Full-data EDM is 98.7 / 82.0 at essentially the same compute (~1.7M vs ~1.56M iterations), so
halving the data costs about 0.3 points. That is the bar our generator is measured against.

Our `fm.pt` (epoch 1300, 3 seeds x 10,000 samples, corrected evaluator) scores **93.5 / 39 / 75**
(atom / molecule / validity) at 100 Euler steps. The gap is not the data split; see
`BASE_MODEL_BENCHMARK.md` for the full comparison, the metric-alignment analysis, and the
evidence that unscaled one-hot atom-type features are the leading suspect.

**Comparability trap:** TFG-Flow (Lin et al., ICLR 2025) models heavy atoms only and its
evaluator counts an atom as stable when its bond-order sum is *at most* the allowed valence.
The same samples that score 38% molecule stability under EDM's rule score 93% under theirs.
Never place a TFG-Flow stability figure beside an EDM-rule figure without the conversion.

---

## Considered and rejected: an uneven split

Since `train_b` only trains $f_B$, why give it a full half? A 70/30 split would hand the
generator ~72k and leave $f_B$ ~31k.

The cost to $f_B$ really is small -- property-prediction error scales roughly as $N^{-0.4}$, so a
1.3x data cut is about an 11% accuracy cut, 0.084 -> ~0.093 D. And the generator, which is our
actual bottleneck, would gain 37% more data.

**Rejected anyway, for the same reason as `train_ab` and for the same evidence.** TFG's
`EDMsecond` trains on ~50k; our 51,527 matches it almost exactly. At 70/30 our generator sees 40%
more data than the model whose published numbers ours sit beside -- the same category error,
merely smaller. No published work in this setting uses anything but 50/50.

If comparability is ever judged not worth its cost, 70/30 is the sensible deviation and $f_B$
barely suffers. Recorded here so it does not have to be re-derived.

## Legitimate exceptions

`--split train_ab` and `--split train_b` remain available, for exactly one purpose: a deliberate,
disclosed ablation. If you use one, say so in the write-up, and do not put its numbers in a table
beside published work without a note.
