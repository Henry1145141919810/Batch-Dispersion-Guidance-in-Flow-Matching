# Base model benchmark: flow-matching generator on QM9

**20 September 2026.** Unconditional generation quality of `fm.pt` (epoch 1300 of a 1500-epoch
run on `train_a`), measured under the EDM protocol and compared against published base models in
the same half-data property-guidance setting. No guidance anywhere in this document.

Every number for our model was produced by code in this repository from raw samples saved under
`results/bench/`. Every published number was read from the paper PDF (`results/bench/papers/`).
An independent agent then audited 29 stated claims and the code (`results/bench/audit/
AUDIT_REPORT.md`); it found one evaluator defect and six errors in the draft, all of which are
fixed below and listed in Section 7. All of our numbers are from the **corrected** evaluator.

---

## 1. Verdict

Under EDM's metric (with hydrogens, exact valence), our base model reaches **atom stability
0.935, molecule stability 0.39, validity 0.75** at its operating point of 100 Euler steps
(3 seeds x 10,000 samples). The like-for-like published baseline -- an **unconditional** EDM
trained on one 50K half and scored with EDM's own code -- reaches **98.37 / 81.74** (EEGSDE,
Table 5). The gap is 4.9 atom points and 43 molecule points, and it is real.

It is **not** caused by the half-data split: halving the data costs EDM 0.33 atom points at equal
compute. It is **not** what it looks like beside TFG-Flow, whose 98-99% atom stability is
computed on heavy atoms only under a rule that passes under-bonded atoms; under that rule as
their code computes it, our model is above their molecule- and atom-stability range and their
validity, and below their uniqueness (Section 4.3). The
leading suspect on the evidence is a **feature-scaling choice**: EDM's own ablation shows that
leaving its categorical channels unscaled (one-hot and charge together, single run) costs 3 atom
points and 35 molecule points; we use unscaled one-hot; and the half-data EDM that beats us
scales its one-hot by 1/8 with a *narrower* network than ours, if it used EDM's released recipe
as its authors state. That is a hyperparameter, testable in one training run, and it could be
wrong.

**One caveat on all of the above, now being closed.** Every published number in
this document is read from a PDF, and Section 4.1 shows one fixed set of
molecules scoring 38%, 84% or 93% molecule-stable depending on whose rule is
applied. Section 9 sets up the fix: TFG's *released* EDM checkpoints run through
**our own evaluator**, so one row of the published column becomes measured
rather than quoted. It is set up and gated; it has not been run.

Three smaller findings from the sweep. The sampler is not the bottleneck: going from 100 to 500
Euler steps buys 0.65 atom points (and 2.8 molecule, 2.1 validity) and 1000 steps buys nothing
more. EMA versus raw weights is not distinguishable at this sample size. And the 512-sample
selection signal could not separate the last 200 epochs: on 10,000 fresh samples the final
epoch (1500) beats the selected epoch (1300) by 0.18 atom and 1.0 molecule points on average
over three seeds, the same sign on every seed and metric. Shipping epoch 1500 instead is a
**disclosed amendment** to the pre-registered rule, argued in Section 3.4, not an a-priori
choice. **Its numbers: atom 0.937 +- 0.001, molecule 0.40 +- 0.01, validity 0.76 +- 0.003.**

---

## 2. What was measured, exactly

| item | choice | why |
|---|---|---|
| checkpoint | `betty_pull/fm.pt`, EMA weights, epoch 1300, `split=train_a` | the deliverable; EMA is what EDM samples too |
| samples | 10,000 per run (5,000 at NFE 1000); 2,000 per epoch snapshot | EDM protocol |
| seeds | 0, 1, 2; none equal to the trainer's eval seed (4321) or the inspector's (99991) | fresh draw |
| sizes | drawn from the `train_a` size distribution | EDM draws n from the training histogram |
| bond inference | Hoogeboom distance tables, **now entry-for-entry identical to EDM's `bond_analyze.py`** (verified by brute force over every HCNOF pair and distance, 0 mismatches) | see Section 7: the C#O triple entry was missing until this audit |
| atom stability | **with hydrogens**; stable iff bond-order sum **equals** allowed valence | EDM |
| molecule stability | every atom stable | EDM |
| validity | RDKit sanitises; **largest connected fragment** kept | EDM's `compute_validity` |
| uniqueness | distinct canonical SMILES among valid | EDM |
| novelty | unique valid SMILES not in the training set; training SMILES inferred from 3D by the same bond code | reported vs `train_a` and vs `train_a`+`train_b`; **not a quality metric** (a worse fit is "more novel") |
| connected | fraction of samples that are one fragment | not an EDM metric; validity alone hides fragments |
| calibration | real QM9 through the same code: atom 0.994, mol 0.956, validity 0.98 (EDM: 99.0 / 95.2 / 97.7) | evaluator faithful on data |

Scripts: `proj1/scripts/benchmark_base.py` (sampling, EDM metrics, saves raw coordinates; the
same `score_samples()` re-scores saved samples with `--rescore`) and
`proj1/scripts/eval_conventions.py` (re-scores saved samples under other papers' rules).

**Scope caveats that apply to every comparison below** (from the audit):

- **Our QM9 is not the published preprocessing.** `prepare_qm9.py` keeps all 133,885 molecules
  (EDM, EEGSDE and GeoLDM exclude the 3,054 "uncharacterized" ones), uses a private random
  split, and its halves are 51,527 not 50,000. `train_a` is 3% larger than EDM's `Db` and
  contains ~2.3% molecules the published protocol removes. The script's own docstring says to
  switch to the official split before quoting against published work. Magnitude unknown;
  disclosed here, not fixed.
- **Sampler.** Ours is a 100-step Euler ODE; EDM and EEGSDE sample a 1000-step SDE. Section 3.2
  measures the step count; the ODE-vs-SDE difference is not measured.
- **Checkpoint selection.** Ours is the best of 61 evaluations on the reported metric (atom
  stability, 512 samples). EDM and TFG-Flow select on loss. Section 3.4 shows the last epoch
  beside the selected one; the fresh-seed re-evaluation removes the winner's curse from the
  number but not the mild bias in the choice.
- **Training settings deviate from the pre-registration.** `PROJECT_GUIDE.md` Section 4.5
  fixes Adam lr 1e-4, batch 128, EMA 0.999, checks every ~5 epochs on 1,000 samples. The run
  used lr 2e-4, batch 256, EMA 0.9999 (a deliberate 19 Sep fix, documented in
  `archive/ship_packages_2026-09-19/WHAT_CHANGED.md`, local-only), checks every 25 epochs on 512 samples. The pre-registered
  *checkpoint rule* (highest validation atom stability at NFE 100) was followed. The
  deviations were not written down until this audit.
- **Architecture.** Ours is a plain dense EGNN, 256 wide, 8 layers, no attention. EDM's
  conditional recipe is 192 wide, 9 layers, with attention. Parameter counts, not layer counts,
  are the fair comparison and are not quoted here.
- **Published budgets are upper bounds.** TFG-Flow early-stops on validation loss; EEGSDE's
  checkpoints sit at epochs 1820-2080; EDM saves on validation NLL.
- **What EEGSDE's paper does not state.** That its half-data EDM includes hydrogens, and its
  width (192) and one-hot scaling (1/8), are inferred from "the same setting with EDM" plus
  EDM's released conditional command; the paper itself does not say them. Its unconditional
  row is a single run (no std given).
- **TFG-Flow differs from us in a second material way** besides heavy atoms: its atom types
  are generated by a discrete-state (CTMC) flow, not a continuous one-hot relaxation.

---

## 3. Our numbers (corrected evaluator)

### 3.1 Main table

| run | NFE | n | atom stab | mol stab | validity | uniqueness | valid x unique | novelty (train_a) | novelty (train_a+b) | connected |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 0 | 100 | 10000 | 0.9334 | 0.3796 | 0.7464 | 0.9949 | 0.7426 | 0.8692 | 0.7474 | 0.9234 |
| seed 1 | 100 | 10000 | 0.9368 | 0.4015 | 0.7559 | 0.9936 | 0.7511 | 0.8725 | 0.7502 | 0.9276 |
| seed 2 | 100 | 10000 | 0.9344 | 0.3878 | 0.7522 | 0.9931 | 0.7470 | 0.8731 | 0.7560 | 0.9256 |
| **mean +- std (3 seeds)** | 100 | 10000 | 0.9349 +- 0.0017 | 0.3896 +- 0.0111 | 0.7515 +- 0.0048 | 0.9939 +- 0.0009 | 0.7469 +- 0.0043 | 0.8716 +- 0.0021 | 0.7512 +- 0.0044 | 0.9255 +- 0.0021 |

Standard error on one 10,000-sample run: atom stability ~0.0006, molecule stability ~0.005, validity ~0.004. The seed-to-seed spread above is the honest uncertainty.

### 3.2 Effect of the sampler (same checkpoint, seed 0)

| run | NFE | n | atom stab | mol stab | validity | uniqueness | valid x unique | novelty (train_a) | novelty (train_a+b) | connected |
|---|---|---|---|---|---|---|---|---|---|---|
| euler, 100 steps | 100 | 10000 | 0.9334 | 0.3796 | 0.7464 | 0.9949 | 0.7426 | 0.8692 | 0.7474 | 0.9234 |
| heun, 100 steps | 200 | 10000 | 0.9362 | 0.3971 | 0.7491 | 0.9940 | 0.7446 | 0.8654 | 0.7373 | 0.9311 |
| euler, 250 steps | 250 | 10000 | 0.9388 | 0.4035 | 0.7645 | 0.9935 | 0.7595 | 0.8689 | 0.7427 | 0.9334 |
| euler, 500 steps | 500 | 10000 | 0.9399 | 0.4078 | 0.7675 | 0.9936 | 0.7626 | 0.8680 | 0.7408 | 0.9347 |
| euler, 1000 steps | 1000 | 5000 | 0.9394 | 0.4088 | 0.7666 | 0.9958 | 0.7634 | 0.8635 | 0.7330 | 0.9285 |

### 3.3 Stability versus training epoch (2,000 samples each, fresh seed 7, NFE 100)

| epoch | atom stab (trainer, 512 samples, seed 4321) | mol stab (trainer) | atom stab (fresh, 2000, seed 7) | mol stab (fresh) | validity (fresh) | connected (fresh) |
|---|---|---|---|---|---|---|
| 100 | 0.8592 | 0.1172 | 0.8559 | 0.1040 | 0.3995 | 0.7785 |
| 200 | 0.8898 | 0.1934 | 0.8868 | 0.1735 | 0.5190 | 0.8449 |
| 300 | 0.9053 | 0.2246 | 0.9031 | 0.2240 | 0.5885 | 0.8819 |
| 400 | 0.9135 | 0.2617 | 0.9092 | 0.2585 | 0.6310 | 0.8906 |
| 500 | 0.9180 | 0.2832 | 0.9149 | 0.2875 | 0.6460 | 0.9156 |
| 600 | 0.9197 | 0.2988 | 0.9207 | 0.3150 | 0.6790 | 0.9109 |
| 700 | 0.9260 | 0.3418 | 0.9253 | 0.3375 | 0.7030 | 0.9118 |
| 800 | 0.9306 | 0.3594 | 0.9261 | 0.3295 | 0.7045 | 0.9226 |
| 900 | 0.9280 | 0.3379 | 0.9297 | 0.3555 | 0.7225 | 0.9183 |
| 1000 | 0.9304 | 0.3594 | 0.9312 | 0.3680 | 0.7170 | 0.9184 |
| 1100 | 0.9323 | 0.3711 | 0.9324 | 0.3740 | 0.7355 | 0.9184 |
| 1200 | 0.9334 | 0.3594 | 0.9344 | 0.3890 | 0.7425 | 0.9239 |
| 1300 | 0.9390 | 0.3926 | 0.9336 | 0.3860 | 0.7530 | 0.9216 |
| 1400 | 0.9343 | 0.3867 | 0.9363 | 0.3945 | 0.7630 | 0.9279 |
| 1500 | 0.9365 | 0.3984 | 0.9363 | 0.3975 | 0.7635 | 0.9273 |

The trainer column is what selection saw: the best of 61 evaluations on one fixed 512-sample draw (the old bond table; differences of ~0.002 from that are immaterial here). The fresh column is the unbiased view of the same checkpoints under the corrected evaluator. Where they disagree, the fresh one is right: the trainer's 0.9390 at epoch 1300 is a 512-sample high draw, and the curve is still rising at 1500.

### 3.4 Two checks on the selection rule

**Selected epoch 1300 vs final epoch 1500** (10,000 samples each, seed 0, NFE 100, EMA weights):

| checkpoint | atom stab | mol stab | validity |
|---|---|---|---|
| epoch 1300 (selected, `fm.pt`) | 0.9334 | 0.3796 | 0.7464 |
| epoch 1500 (`fm_last.pt`) | 0.9358 | 0.3913 | 0.7589 |

**The 512-sample selection signal could not distinguish these two checkpoints.** The trainer
scored epoch 1300 at 0.9390 and epoch 1500 at 0.9365 on its fixed 512-sample draw -- one standard
error apart at that sample size -- and kept 1300. On 10,000 fresh samples with the same seed,
epoch 1500 is better by +0.0023 atom, +0.0117 molecule and +0.0125 validity. The comparison is
genuinely paired (identical size draws by index; per-molecule agreement 0.905 against 0.526 if
independent), and a paired bootstrap puts the three differences at 4.1 / 3.9 / 3.6 standard
errors; the fresh-seed curve in 3.3 agrees in sign (2.0 / 1.6 SE). So the difference is real.
It is also small: smaller than the epoch-1300 checkpoint's own seed-to-seed spread in 3.1.

**Recommendation, stated as an amendment.** Ship the epoch-1500 EMA weights (`fm_last.pt`, key
`ema`). This overrides the pre-registered rule of `PROJECT_GUIDE.md` Section 4.5. The
comparison itself was pre-specified in `run_all.sh` before any result was seen; the decision to
act on it was not, so it is a disclosed post-hoc amendment, justified by the rule's evaluation
sample (512) being too small to resolve differences of the size that exist between late
checkpoints. **Confirmed on seeds 1 and 2** (10,000 samples each, corrected evaluator):

| seed | epoch 1300 atom / mol / validity | epoch 1500 atom / mol / validity | paired delta |
|---|---|---|---|
| 0 | 0.9334 / 0.3796 / 0.7464 | 0.9358 / 0.3913 / 0.7589 | +0.0023 / +0.0117 / +0.0125 |
| 1 | 0.9368 / 0.4015 / 0.7559 | 0.9381 / 0.4082 / 0.7648 | +0.0013 / +0.0067 / +0.0089 |
| 2 | 0.9344 / 0.3878 / 0.7522 | 0.9361 / 0.3985 / 0.7637 | +0.0017 / +0.0107 / +0.0115 |
| **mean +- std** | 0.9349 +- 0.0017 / 0.3896 +- 0.0111 / 0.7515 +- 0.0048 | **0.9366 +- 0.0012 / 0.3993 +- 0.0085 / 0.7625 +- 0.0031** | +0.0018 / +0.0097 / +0.0110 |

Every seed, every metric, the same sign. Epoch 1500 is the better checkpoint; the amendment is
justified. For the diffusion run and any retrain, raise `--stab-n` to 2048 so the rule can
resolve this itself.

**EMA vs raw weights** (epoch 1300, NFE 100, seed 0):

| weights | n | atom stab | mol stab | validity |
|---|---|---|---|---|
| EMA (what we ship) | 10000 | 0.9334 | 0.3796 | 0.7464 |
| raw | 5000 | 0.9319 | 0.3746 | 0.7496 |

Paired on the 5,000 shared indices, EMA is ahead by 0.12 atom points (1.2 SE), 0.9 SE on
molecule stability and behind by 0.8 SE on validity. **Not a measurable difference at this
sample size.** EMA is kept because it is the published convention, not because this shows it
helps.

---

## 4. The comparison, with metrics and scope aligned

### 4.1 First, the thing that decides everything: which rule

The same samples (seed 0, NFE 100, 10,000 molecules), scored three ways. Connectivity is divided
by all generated molecules, as TFG-Flow's evaluator does.

| convention | atoms scored | rule | atom stab | mol stab | validity | connectivity |
|---|---|---|---|---|---|---|
| **EDM** (Hoogeboom 2022): ours, EEGSDE, GeoLDM | all, incl. H | bond-order sum **= valence** | **0.933** | **0.380** | 0.746 | 0.689 |
| **TFG-Flow** (Lin 2025), rule as intended | heavy only | bond-order sum **<= valence** | 0.974 | 0.839 | 0.840 | 0.810 |
| **TFG-Flow**, rule as their code counts it | heavy only | as above, but each atom counts only bonds to higher-indexed atoms | 0.992 | 0.931 | 0.840 | 0.810 |
| *real QM9 under EDM's rule* | all | = | 0.994 | 0.956 | 0.982 | 0.980 |
| *real QM9 under TFG-Flow's rule (as coded)* | heavy | <= | 1.000 | 0.998 | 0.983 | 0.981 |

Seeds 1 and 2 agree with these to within 0.025 on every entry -- molecule stability moves by up to
0.022 between seeds, connectivity by 0.012 (`results/bench/rescored/conv_*`).

**Molecule stability for one fixed set of molecules is 38%, 84% or 93% depending on the
definition.** A table that mixes rows from these conventions is meaningless. TFG-Flow's rule
passes an under-bonded atom (the missing bonds are assumed to be hydrogens it does not model),
and its loop increments only `atom_orders[i]` for each pair, never `[j]` -- both verified in
their released `utils/evaluator.py`, and both reproduced here. Under that rule even real QM9
scores 99.8%.

### 4.2 Like-for-like: unconditional EDM on one 50K half (EEGSDE, ICLR 2023)

EEGSDE re-trained EDM on one 50K half (`Db`) with EDM's conditional recipe (~2000 epochs, batch
64, EMA 0.9999, nf 192, one-hot scaled by 1/8, no charges) and scored 10K samples with **EDM's
official code, with hydrogens**. Their Table 5 opens with a row that has neither conditioning nor
guidance -- an unconditional half-data EDM -- and Table 8 gives the conditional model per
property:

| model | data | novelty | atom stab | mol stab |
|---|---|---|---|---|
| **EDM, unconditional, half data** (Table 5, row 1) | 50K | 82.92 | **98.37** | **81.74** |
| EDM, conditional, half data (Table 8; range over 6 properties) | 50K | 83.6-84.6 | 98.13-98.30 | 79.33-81.95 |
| EDM, unconditional, full data (EDM Table 1) | 100K | -- | 98.7 +- 0.1 | 82.0 +- 0.4 |
| **ours, unconditional** (3-seed mean) | 51.5K | 87.2 | **93.5** | **39** |

This is the fair bar. Two things follow. **Halving the data is not the problem:** 98.7 -> 98.37
atom and 82.0 -> 81.74 molecule, at essentially equal compute (EDM ~1.7M iterations per its
paper, EEGSDE ~1.56M). And **our gap to the bar is 4.9 atom points and 43 molecule points**, on a
model with 5.2x fewer gradient steps, no attention, a 100-step ODE instead of a 1000-step SDE,
and unscaled type features (Section 5). Novelty is listed for completeness only; ours being
higher is not an advantage.

### 4.3 Closest sibling: TFG-Flow (ICLR 2025), scored under its own rule

TFG-Flow is the nearest published model to ours in every respect but one: flow matching, EGNN
(9 x 256), first 50K half, batch 256, 1200 epochs (~235k steps -- fewer than our 303k), atom
types + coordinates only, 100 Euler steps. The one difference: **heavy atoms only.** They report
no unconditional row. Their Table 8 (TFG-Flow, guided) has six property rows, but the gap, HOMO
and LUMO rows are digit-for-digit identical to Table 6 (their conditional-flow baseline) while
the standard deviations differ -- a transcription error in the paper. Only the alpha, mu and Cv
rows can be read as TFG-Flow numbers:

| | validity | uniqueness | novelty | mol stab | atom stab | connectivity |
|---|---|---|---|---|---|---|
| TFG-Flow Table 8, alpha / mu / Cv rows (4,096 guided samples each) | 75.3-76.3 | 99.0-99.4 | 89.9-93.4 | 86.9-89.9 | 98.3-98.8 | 63.6-68.6 |
| **ours, their rule as coded, H stripped** (10,000, 3 seeds) | **84.0-84.7** | 97.9-98.1 | -- | **93.1-93.8** | **99.2-99.3** | **81.0-82.0** |
| ours, their rule as intended | 84.0-84.7 | 97.9-98.1 | -- | 83.9-84.7 | 97.4-97.6 | 81.0-82.0 |

Under their rule as coded, our base model is above their range on molecule stability, atom
stability, validity and connectivity, and **below it on uniqueness** (97.9-98.1 vs 99.0-99.4).
Under the rule as evidently intended, we are 2-6 points below on molecule stability and about
one point below on atom stability, above on validity and connectivity, below on uniqueness.
Caveats, in both directions: their rows are guided samples over 4,096 molecules with a
hyperparameter search that keeps validity above 75%, so their validity is a search floor, not a
free measurement; and stripping hydrogens after generation both removes our H-placement errors
from the score (favours us) and means our heavy-atom skeleton was built under the harder
all-atom task (favours them). Novelty is not compared: different reference set and atom set.

### 4.4 Full-data reference (not like-for-like; the numbers people know)

QM9 with hydrogens, 10,000 samples, 3 runs. From EDM Table 1, EDM Table 10, GeoLDM Table 1 and
EEGSDE Table 5:

| model | data | atom stab | mol stab | validity | valid x unique |
|---|---|---|---|---|---|
| Data | -- | 99.0 | 95.2 | 97.7 | 97.7 |
| E-NF | 100K | 85.0 | 4.9 | 40.2 | 39.4 |
| G-Schnet | 100K | 95.7 | 68.1 | 85.5 | 80.3 |
| GDM (non-equivariant EDM) | 100K | 97.0 | 63.2 | -- | -- |
| GDM-aug | 100K | 97.6 | 71.6 | 90.4 | 89.5 |
| **EDM with unscaled one-hot** (Table 10, single run) | 100K | **95.7** | **46.9** | -- | -- |
| EDM | 100K | 98.7 | 82.0 | 91.9 | 90.7 |
| EDM-Bridge | 100K | 98.8 | 84.6 | 92.0 | 90.7 |
| GeoLDM | 100K | 98.9 | 89.4 | 93.8 | 92.7 |
| **EDM, unconditional, half data** (EEGSDE Table 5) | **50K** | **98.37** | **81.74** | -- | -- |
| **ours** | **51.5K** | **93.5** | **39** | **75** | **75** |

### 4.5 Not comparable, and why

- **TFG** (Ye et al., NeurIPS 2024) uses the same half-data EDM (`EDMsecond`) sampled with DDIM
  at 100 steps, but reports only MAE and RDKit validity for guided samples (4,096 each). No
  stability. Its guided validity of 62.5-85.5% across methods is the only published figure for
  "half-data EDM at 100 steps", and it is not the base model's number.
- **PropMolFlow** (2025) trains on 50K with explicit hydrogens and 2000 epochs, but scores
  stability with a revised formal-charge-aware rule on **explicitly generated bond orders**,
  on a re-curated QM9, and notes that earlier metrics tolerate inconsistent bond-charge
  combinations. Different metric, different data; main text has box plots only. Not to be
  placed beside any row above.
- **GeoLDM conditional / EDM conditional** (the papers themselves) report MAE only for their
  half-data models.

---

## 5. Why the gap: ranked by evidence

**1. Unscaled atom-type features. The leading suspect, not a demonstrated cause.**
EDM's Table 10 changes the scale of both categorical channels at once -- one-hot 1.0 -> 0.25 and
integer charge 1.0 -> 0.1 -- and reports a single run for the unscaled row. We have no charge
channel, so only the one-hot half can apply to us:

| feature scaling | atom stab | mol stab |
|---|---|---|
| `[x, 1.00 h_onehot, 1.0 h_charge]` (single run) | 95.7 | 46.9 |
| `[x, 0.25 h_onehot, 0.1 h_charge]` (EDM default, 3 runs) | 98.7 | 82.0 |

Their explanation: "it seems that it is easier to learn a denoising process where the atom type
is decided later, when the atom coordinates are already relatively well defined." Three facts
line up behind this:

- **Our features are plain `{0,1}` one-hot, used unscaled.** Verified: `data/qm9.pt` row sums
  are exactly 1 on real atoms; `train_fm.py` and `egnn.py` apply no factor.
- **The half-data EDM that beats us by 4.9 atom points scales its one-hot by 1/8** -- if, as
  EEGSDE states, it used EDM's conditional recipe (`--normalize_factors [1,8,1]
  --include_charges False`, `--nf 192`), which also makes it *narrower* than ours.
- The mechanism transfers to the linear interpolant: with unscaled types the type signal
  exceeds the noise from t > 0.5; scaled by 0.25, not until t > 0.8; by 1/8, t > 0.89. Types
  are decided after the geometry has formed.

This is a hypothesis for flow matching: the ablation is on diffusion, single-run, and confounded
with the charge channel. It is the only candidate whose measured size is of the order of our gap,
and it costs one training run to test. It could also be wrong.

**2. Training budget.** 303k gradient steps versus ~1.56M for the half-data EDM (5.2x) and
~1.7M for full EDM (5.7x; the EDM paper reports 1100 epochs, its README's 3000 is not what the
paper used). Our own curve (Section 3.3) was still rising when the cosine schedule hit its floor.

**3. Sampling. Measured, and small.** Section 3.2: 100 -> 250 -> 500 -> 1000 Euler steps gives
atom stability 0.933 -> 0.939 -> 0.940 -> 0.939, molecule stability 0.380 -> 0.404 -> 0.408 ->
0.409, validity 0.746 -> 0.765 -> 0.768 -> 0.767 (seed 0). Heun at 100 steps (200 evaluations)
gives 0.936 / 0.397 / 0.749. The integrator is worth 0.65 atom, 2.8 molecule and 2.1 validity
points and saturates by 500 steps. The ODE-vs-SDE difference is unmeasured.

**4. Architecture.** No attention. Width vs depth roughly offset (256 x 8 vs 192 x 9).

**Ruled out as a major factor: data.** 98.7 -> 98.37 atom for halving at equal compute.
**Ruled out: the evaluator.** After the fix in Section 7 it is identical to EDM's.

---

## 6. What to do about it, in order of cost

1. **Scale the one-hot features (0.25, or EDM-conditional's 1/8) and retrain.** One run,
   ~11.5 h on Betty. Everything above says this is the first thing to try, before any extension
   of the current schedule.
2. **Ship epoch 1500, not 1300** (`fm_last.pt`, key `ema`): +0.23 atom, +1.2 molecule points.
   This overrides the pre-registered checkpoint rule and must be recorded as an amendment
   with the reason (the 512-sample signal could not resolve the last 200 epochs; the fresh
   comparison was pre-specified in `run_all.sh`, the decision to act on it was not). Confirmed
   on all three seeds (Section 3.4): +0.18 atom, +1.0 molecule, +1.1 validity points on average,
   same sign every time. For the diffusion run and any retrain, raise `--stab-n` to 2048 so
   selection can resolve differences of this size.
3. **Sample at 250-500 steps when quoting a number** (+0.65 atom, +2.8 molecule points;
   nothing beyond 500). Guidance experiments can stay at 100.
4. **Only then** consider a longer schedule.

For the write-up: quote our number beside EEGSDE's unconditional half-data row (same protocol,
same code); beside TFG-Flow only under TFG-Flow's rule with the conversion in Section 4.1 shown;
never beside PropMolFlow; and disclose the preprocessing difference in Section 2.

---

## 7. Independent check, and what it changed

An adversarial agent verified 29 claims against the PDFs, arXiv HTML, the EDM and TFG-Flow
GitHub code, and our code and data: 21 confirmed, 6 partially, 1 refuted, 1 unverifiable
(`results/bench/audit/AUDIT_REPORT.md`). A second pass on the finished draft
(`AUDIT_REPORT_PASS2.md`) re-derived every table from the JSONs (all reproduce), tested the
paired-comparison claim, and found nine further errors of wording, sign or rounding in the prose
-- two of them comparison signs against TFG-Flow that had been stated the wrong way round -- all
corrected in this version. What the two passes found and what was done:

| finding | severity | action |
|---|---|---|
| `evaluation.py` `BONDS3` lacked EDM's C#O triple entry (113 pm). Zero real-QM9 pairs fall in that window, so calibration looked perfect; 2.5% of generated molecules had one. Reported numbers were inflated by 0.2 (atom), 0.3 (mol), 1.1 (validity) points. | **evaluator defect** | entry added; brute-force grid confirms 0 mismatches with EDM's `get_bond_order`; every sample set re-scored; all numbers in this document are post-fix. The trainer's checkpoint selection used the old table; effect on selection immaterial (Section 3.4). |
| TFG-Flow re-implementation used our (then wrong) table and divided connectivity by valid rather than all molecules | refuted claim | fixed; auditor's independent numbers reproduced to 4 decimals |
| TFG-Flow Table 8 gap/HOMO/LUMO rows duplicate Table 6 | their transcription error | only alpha/mu/Cv rows used |
| EDM budget quoted from README (3000 epochs, 4.7M) not the paper (1100, ~1.7M); "half-data EDM had 3x fewer steps" therefore false | draft error | corrected: 5.7x, equal compute |
| EEGSDE Table 5 has an **unconditional** half-data row | missed comparator | now the primary like-for-like row |
| PropMolFlow "batch 128" is its inference batch; its inflation remark targets other revised metrics, not Hoogeboom's | draft error | corrected |
| QM9 preprocessing differs from the published protocol; checkpoint selected on the reported metric; heavy-atom stripping cuts both ways; ODE vs SDE; novelty not a quality metric | caveats | all stated in Section 2 / 4.3 |
| Pass 2: "above TFG-Flow on every column" was false (below on uniqueness; below on atom stability under the intended rule); "EMA helps" was 1.2 SE; EDM Table 10 changes two channels, not one; TFG validity range mis-stated; several roundings | draft errors | all corrected; hedges added for EEGSDE's inferred recipe, its single-run row, and TFG-Flow's discrete-state type flow |
| Pass 2: training used lr 2e-4 / batch 256 / EMA 0.9999 / 512-sample checks against a pre-registration of 1e-4 / 128 / 0.999 / 1k; shipping epoch 1500 overrides the pre-registered checkpoint rule | undisclosed deviations | both now disclosed (Section 2, Section 3.4) and the latter framed as an amendment requiring confirmation on seeds 1-2 |

---

## 8. Reproduce

```
python proj1/scripts/benchmark_base.py --ckpt betty_pull/fm.pt --n 10000 --steps 100 --seed 0 \
    --out results/bench/fm_nfe100_euler_s0.json
python proj1/scripts/benchmark_base.py --rescore results/bench/*_samples.pt --out results/bench/rescored
python proj1/scripts/eval_conventions.py results/bench/fm_nfe100_euler_s0_samples.pt
python proj1/scripts/eval_conventions.py --data 2000
```

`results/bench/run_all.sh` is the sweep; `results/bench/rescored/` holds every result under the
corrected evaluator; `results/bench/papers/` the PDFs; `results/bench/CLAIMS_TO_VERIFY.md` the
claims; `results/bench/audit/AUDIT_REPORT.md` the audit.

---

## 9. Head-to-head against a borrowed checkpoint, through our own evaluator

**Status: set up, not yet run.** No numbers below are filled in. The
checkpoints are not downloaded and the table is here so the protocol is fixed
before any result is seen.

### 9.1 Why, given Sections 4 and 5 already compare against published work

Every published row in Section 4 is read from a PDF. That comparison carries a
caveat this document spends Section 4.1 demonstrating: **one fixed set of
molecules scores 38%, 84% or 93% molecule-stable depending on whose rule is
applied.** We control for that by quoting only rows we believe share EDM's
convention — but "believe" is doing work there, and EEGSDE's unconditional
half-data row, the single most important comparator in this document, is a
**single run with no standard deviation** whose recipe we infer from "the same
setting with EDM" plus EDM's released command.

Running TFG's *released* EDM checkpoint through **this repository's evaluator**
removes that caveat for one row. `benchmark_transfer_base.py` imports
`score_samples` and `dataset_smiles` from `benchmark_base.py` — imported, not
reimplemented — so the borrowed model and ours pass through the same bond
tables, the same largest-fragment validity rule, the same SMILES
canonicalisation. A difference in the output is then a difference in the
models, and nothing else.

This also does something Section 4 cannot: it puts a **measured** number on the
published column. If TFG's `EDMsecond` scores near 98.4 / 81.7 under our
evaluator, the published column is calibrated and Section 4's comparisons are
sound as stated. If it does not, the gap in Section 1 is partly an
evaluator-convention artefact and this document needs revising.

### 9.2 What is and is not held equal

| item | ours | borrowed | equal? |
|---|---|---|---|
| evaluator, bond tables, validity rule, SMILES | `score_samples` | **the same function object** | ✅ |
| molecule sizes | drawn from `train_a` | drawn from `train_a`, same generator and seed | ✅ paired |
| sample count, seeds | 10,000 × 3 | 10,000 × 3 | ✅ |
| QM9 preprocessing | ours (133,885, private split) | ours | ✅ *applies to both* |
| training data | our `train_a` half (51,527) | TFG's half (50,000), identity unknown | ⚠️ different halves |
| training budget | 303k steps | ~1.56M steps | ❌ **not equal** |
| architecture | EGNN 256×8, no attention | EGNN **256×9**, attention (read from the checkpoint's args, 23 Sep) | ❌ **not equal** |
| feature scaling | one-hot ×1 | one-hot **×1/4** (`normalize_factors` [1, 4, 10], read from the checkpoint) | ❌ **the hypothesis in Section 5** |
| sampler | 100-step Euler PF-ODE | **the same** | ✅ *not EDM's own SDE* |

**Drawing sizes from `train_a` for both is the right call, and it is measured,
not asserted.** The two halves' size histograms have total variation **0.0064**
(mean atoms 18.003 vs 17.985), so conditioning the borrowed model on our half's
histogram costs it nothing, and in exchange the two runs are paired molecule
for molecule rather than merely equal in distribution.

**The sampler is the one deliberate asymmetry against the published number.**
EDM samples a 1000-step ancestral SDE; we integrate the probability-flow ODE at
100 steps, because that is the sampler our guidance fields are defined on and
holding it fixed is what makes the transfer experiment internal. So a shortfall
against EEGSDE's published row is **not by itself evidence of an adapter
defect** — `--steps` and `--grid` separate the two.

### 9.3 The stiffness problem, and the grid that fixes it

EDM's `polynomial_2` schedule is far stiffer at the noise end than the
linear-beta schedule our own diffusion model uses. Measured on the 101-point
grid the benchmark actually integrates:

| grid | worst Δγ in one step | best | comment |
|---|---|---|---|
| uniform in τ (naive) | **3.702** | 0.061 | the first Euler step crosses 3.7 nats of log-SNR — SNR ×40 |
| uniform in γ (`--grid gamma`) | **0.228** | 0.228 | flat by construction |
| *our own linear-β schedule, uniform in τ* | 0.20 | 0.11 | for scale |

A naive uniform grid therefore spends its first step in a regime our own model
never enters, at 18× the step size. `--grid gamma` re-spaces the same
trajectory — identical ODE, identical endpoints — so the steps land where the
distribution is moving. **`gamma` is the default**, and `uniform` is kept so
the difference can be reported rather than assumed.

### 9.4 The table to fill in

10,000 samples, 3 seeds, 100 Euler steps, `--grid gamma`, our evaluator:

| model | data | source | atom stab | mol stab | validity (EDM) | valid × unique | connected |
|---|---|---|---|---|---|---|---|
| Real QM9 (calibration) | — | — | 0.994 | 0.956 | 0.982 | — | 0.980 |
| **ours** (`fm_last.pt`) | 51.5K | this repo | **0.9366 ± .0012** | **0.3993 ± .0085** | **0.7625 ± .0031** | 0.746 | 0.926 |
| TFG `EDMsecond` | 50K | TFG release | — | — | — | — | — |
| TFG `EDMfull` *(optional)* | 100K | TFG release | — | — | — | — | — |
| *EDM half-data, as published* | 50K | EEGSDE Tab. 5 | *0.9837* | *0.8174* | — | — | — |
| *EDM full, as published* | 100K | EDM Tab. 1 | *0.987* | *0.820* | *0.919* | *0.907* | — |

The two italic rows are what Sections 4.2 and 4.4 already quote. The point of
the two blank rows is to sit directly above them and say whether they survive
re-measurement.

**Measured 23 Sep — the gate run, not yet the table's protocol.** n = 2,000,
**one seed**, sizes from `train_a`, 100 Euler steps, local RTX 5080, md5
`6abbd010…`. It passes the gate at both grids; the 10,000 × 3 row above is
still to be run, and these numbers must not be quoted in its place.

| grid | atom stab | mol stab | validity (EDM) | valid × unique | connected (of valid) |
|---|---|---|---|---|---|
| `uniform` | **0.9644** | **0.6725** | 0.849 | 0.8485 | 0.952 |
| `gamma` | 0.9589 | 0.6380 | 0.831 | 0.8305 | 0.938 |

Three things follow. (1) Through **our** evaluator at 100 steps, the borrowed
model beats ours by ~2.8 atom / ~27–31 molecule points; it is itself ~2 / ~15
below its published row, the price of our 100-step ODE against EDM's
1000-step SDE (§9.2). (2) **Section 9.3's prediction did not hold**: the
gamma-uniform grid is *worse* than the naive one, by 0.6 / 3.5 points. The
transfer sweep therefore runs `--grid uniform`. (3) The model the gate
measures scales one-hot by **1/4** and is **256** wide — not the 1/8 / 192
recipe this document infers for EEGSDE's half-data row (§1, §2, §5). That
inference was about EEGSDE's model, which is not released; but the one
half-data EDM we *can* inspect uses EDM's unconditional default, and any
sentence saying "the model beating us scales by 1/8" should say which model
it means.

**Novelty is not in this table and must not be added to it.** `score_samples`
computes `novelty_vs_train_a` against *our* half; the borrowed models' own
training halves are unknown, so their novelty numbers measure nothing about
them. The field is written to the JSON (schema parity) and flagged in the
output's `caveats`.

### 9.5 The gate

`benchmark_transfer_base.py` exits non-zero if the borrowed model scores
**atom < 0.95 or molecule < 0.60**. That is deliberately loose — it asks *is
the adapter driving this checkpoint at all*, not *does our ODE reproduce their
SDE*. Our own model loses only 0.65 atom and 2.8 molecule points going from 500
steps to 100 (Section 3.2), so a correct adapter on a model published at
98.4 / 81.7 should clear this comfortably.

**If it fails, no guidance cell runs.** Every guided number would inherit the
fault while looking plausible. The order of investigation is in the script's
docstring: re-run at the other `--grid` and at `--steps 1000`; if that fixes
it, the cause is discretisation, not the adapter; if not, suspect the epsilon
sign, the time direction, the one-hot scale, or the EMA key.

### 9.6 Reproduce

```bash
python proj1/scripts/fetch_tfg_assets.py --models EDMsecond,EDMfull
python proj1/tests/test_transfer_backend.py --require-edm      # 74 gates

python proj1/scripts/benchmark_transfer_base.py --edm-dir weights/EDMsecond \
    --n 2000 --steps 100 --grid gamma --seed 0 \
    --out results/bench/edmsecond_nfe100_gamma_s0.json          # the gate

# then the full 3-seed runs, and the uniform-grid comparison
for s in 0 1 2; do
  python proj1/scripts/benchmark_transfer_base.py --edm-dir weights/EDMsecond \
      --n 10000 --steps 100 --grid gamma --seed $s \
      --out results/bench/edmsecond_nfe100_gamma_s$s.json
done
python proj1/scripts/benchmark_transfer_base.py --edm-dir weights/EDMsecond \
    --n 10000 --steps 100 --grid uniform --seed 0 \
    --out results/bench/edmsecond_nfe100_uniform_s0.json
```
