# Which networks supply f_A and f_B, and why

**19 September 2026.** Decision record for the property predictors. Companion to
[REQUIREMENTS_FOR_FA_FB.md](REQUIREMENTS_FOR_FA_FB.md) (what we need) and
[FA_FB_INTERNET_SEARCH.md](FA_FB_INTERNET_SEARCH.md) (what exists). This file records what we
measured, what we chose, and what remains unverified.

---

## 1. The two roles are not symmetric

| | $f_A$ — the **guide** | $f_B$ — the **evaluator** (TFG calls it the *oracle*) |
|---|---|---|
| what it does | its **gradient** steers sampling, inside the loop | **scores** finished samples, never touches generation |
| needs | $\nabla f$ w.r.t. coordinates **and continuous atom features**; $\nabla^2 f$ for SMG | forward passes only |
| accuracy matters because | guidance converges to $f_A(\hat x_1)=y^\ast$, so **$f_A$'s error is a hard floor** on true property error | it sets $\delta = k\times\text{MAE}$, the tolerance every in-band number is measured against |

That asymmetry drives everything below. A network can be an excellent $f_B$ and a useless $f_A$
if it is not differentiable in the right variables.

**A trap worth stating once.** $\delta$ must be set by the **worse** of the two. If $f_B$ is very
accurate ($\delta$ small) while $f_A$ is not, guidance cannot reach the band no matter how good
the method is, and the experiment measures $f_A$'s weakness instead of the guidance. An earlier
plan of ours had $f_A$ at 0.12 D against $\delta = 0.042$ and would have produced exactly that.

---

## 2. Our own network

`EGNNScalar` — an E(n)-equivariant graph network, **not** an MLP:

```
one-hot atom type + t  ->  Linear  ->  h                    [B, N, 128]
4 x EGNN layer:
      m_ij = MLP(h_i, h_j, ||r_i - r_j||^2)        message per ordered atom pair
      h_i  = h_i + MLP(h_i, sum_j m_ij)            equal-weight aggregation
mean-pool over valid atoms  ->  MLP  ->  Linear  ->  scalar
```

Invariance is structural: messages depend only on **squared distance**, which no rotation or
translation changes, and pooling is permutation-symmetric. Verified to 1e-17 in
`proj1/tests/test_egnn_symmetry.py`.

*No attention* means $\sum_j m_{ij}$ weights every neighbour equally. With attention it would be
$\sum_j a_{ij}m_{ij}$, $a_{ij}=\sigma(\text{MLP}(m_{ij}))$ — a learned per-edge weight, which is
what EDM, TFG and OC-Flow all use and we do not.

**429,825 parameters, hidden 128, 4 layers, 120 epochs, 23 minutes per model.**
Validation MAE on $\mu$: $f_A$ **0.0897 D**, $f_B$ **0.0840 D**, against a 1.53 D chance baseline.

---

## 3. The candidates, measured

### 3.1 Pretrained SchNet (quantum-machine.org, via PyTorch Geometric)

Different architecture family: continuous-filter convolutions over a radius graph, not
message-passing on squared distances.

**Measured on 512 molecules SchNet never trained on:**

| | MAE |
|---|---|
| SchNet | **0.0210 D** |
| our $f_B$ | 0.0722 D |
| **correlation of their errors** | **0.187** |

That 0.187 is the important number. It says architecture diversity decorrelates errors far more
than shared training data correlates them — SchNet shares **74.8%** of our `train_b` molecules
(measured directly; its published split indexes the same 133,885-molecule list in the same order)
and the errors still barely agree.

**As $f_B$: excellent, and working today** — `proj1/src/schnet_ref.py`, loaded without a
schnetpack install by stubbing the module tree, with a dense radius graph replacing the
`pyg-lib` kernel that we avoid.

**As $f_A$: possible but unverified.** SchNet embeds atom type by integer lookup,
`self.embedding(z)`, which gives **no gradient with respect to continuous atom features** — and
our sampler carries continuous features until the final argmax. The fix is the linear-embedding
adapter the search document used for TFG-Flow, `h = feats @ embedding_weights`, which recovers
feature derivatives and reproduces outputs exactly at one-hot inputs. Second derivatives through
the dipole readout $\lVert\sum_i q_i(r_i-r_{\text{com}})\rVert$ still need checking; a norm is
smooth away from zero but SMG needs the Hessian.

### 3.2 TFG (NeurIPS 2024) — a matched pair

Covers **all three of our properties** (μ, α, gap) plus HOMO, LUMO, Cv. Adopting it means
**zero property training**.

| | TFG guide | TFG oracle |
|---|---|---|
| base network | EDM `egnn_dynamics` + energy head | EDM property regressor |
| parameters | **1,671,566** | **743,944** |
| width / layers | 192 / 7 | 128 / 7 |
| attention | yes | yes |
| charges | `include_charges: False` | **`True`, `charge_power: 2`** |
| feature scaling | `[1, 8, 1]` | none |

The two are **not** the same architecture as each other — the guide is a diffusion *dynamics*
network repurposed with an energy readout, which is why it carries `condition_time: True` (they
feed $t=0$; the time input is inherited, not evidence of noisy training).

**Disjointness is claimed but not certified.** Both saved argument files record
`dataset: qm9_second_half`, and the released training script mutates `args.dataset` *after*
building its training loader, so that string describes an auxiliary test loader:

```python
dataloaders, _ = dataset.retrieve_dataloaders(args)   # trains on the DEFAULT: first_half
args.dataset = "qm9_second_half"                      # overwritten here
dataloaders["test"] = dataloaders_aux["train"]        # second half becomes TEST
```

**We tested it empirically instead** (`audit/fa_fb_search/disjointness_test.py`). A network fits
its training molecules better, so the joint structure of the two error sets reveals the split:

| | guide wins >2x | oracle wins >2x | neither |
|---|---|---|---|
| observed over 3,000 molecules | 30.9% | 40.9% | 28.2% |
| predicted by a 50/50 split of EDM's 100k train set | 37.4% | 37.4% | 25.3% |

with 6–7x MAE gaps inside each winning population, and both models weak on the leftover quarter —
which is EDM's val/test, the molecules neither saw. **Consistent with disjoint halves.** Same-half
training cannot produce that structure.

**Replicated independently on all three properties**, which is much stronger than one result:

| property | guide wins | oracle wins | neither | MAE gap in winning population |
|---|---|---|---|---|
| μ (Debye) | 30.9% | 40.9% | 28.2% | 7x |
| α (Bohr³) | 42.6% | 37.3% | 20.1% | 9x |
| gap (Hartree) | 39.8% | 39.6% | 20.6% | 11x |
| *predicted* | *37.3%* | *37.3%* | *25.3%* | — |

Each uses different weights trained on the same nominal split, so three separate model pairs all
reproduce the same signature. The gap pair is the cleanest — 39.8 / 39.6, almost exactly symmetric.

Held-out accuracy, on the molecules neither model memorised, in each property's own units:

| | μ (D) | α (Bohr³) | gap (Hartree) |
|---|---|---|---|
| TFG guide | 0.0659 | 0.1551 | 0.0029 |
| TFG oracle | 0.0728 | 0.1529 | 0.0027 |

The gap figure is ~0.074 eV and α ~0.155 Bohr³, both in the range published EGNN work reports,
which is an independent check that the fitted raw→physical rescaling is sane.

*A correction worth recording:* the first version of this test used **error correlation** as the
discriminator and reported "same half suspected" at $+0.886$. That was wrong. High $|{\rm error}|$
correlation comes from shared **molecule difficulty** — both models struggle on the same awkward
geometries — and persists for genuinely disjoint pairs. The population fractions are the right
statistic.

**Accuracy, honestly.** Raw MAE over all 3,000 is 0.044/0.046 D, but that is inflated by
memorisation. On molecules **neither** memorised:

| | held-out MAE |
|---|---|
| TFG guide | 0.0659 D |
| TFG oracle | 0.0728 D |
| our $f_B$ | 0.0840 D |

A ~20% improvement, not the 2x the raw figures suggest. Their networks are simply 2–4x larger.

### 3.3 Considered and rejected

| | why not |
|---|---|
| **TFG-Flow** | genuine disjoint pair, but **heavy atoms only** — hydrogens removed, and we model explicit H. Released MAE 0.20/0.18 D, worse than ours. |
| **EEGSDE** | guide is **time-conditioned on noisy inputs**. Not a technicality: a noise-aware guide learns $\mathbb E[f(x_1)\mid x_t]$ directly, the Jensen gap vanishes by construction, and **SMG has nothing left to correct.** Its clean evaluators remain usable. |
| **PropMolFlow / MolGuidance** | consume learned **bond-feature** embeddings. Our intermediates are point clouds with no bonds. |
| **DimeNet** | no matched pair, no gap model, angle terms fragile at atom collisions. |
| **OC-Flow** | six clean EGNN checkpoints at **0.0454 D** — the best unexamined single-side asset. Still EGNN; no partner. Worth an audit if we want a stronger $f_A$ without training. |

---

## 4. Decision

**Evaluator panel, not a single evaluator.** No available option gives both data-disjointness and
architecture-diversity, so we get both across a set of three:

| | role | catches |
|---|---|---|
| **TFG guide** | $f_A$ | — |
| **TFG oracle** | $f_B$ | its own pair's data split |
| **our $f_B$** (`train_b`) | second evaluator | data blind spots, independent of TFG entirely |
| **SchNet** | third evaluator | **architecture** blind spots |

Reporting all three per arm means TFG's unverified disjointness stops being fatal: if their pair
were secretly non-disjoint, our $f_B$ and SchNet would still catch the reward hacking.

**$\delta$ from the evaluator we trust, reported with its $k$,** and every in-band number labelled
with which evaluator produced it. Switching $f_B$ changes $\delta$ — 2 x 0.073 = 0.146 with TFG's
oracle against 0.168 with ours — so numbers from before and after a switch are not comparable.

**We still train our own six** (3 properties x 2 splits, ~50 min wall-clock at 3 parallel jobs).
Cheap, and they are the panel member that owes nothing to anyone else's split.

### Open items

1. **TFG oracle's charge features.** `include_charges: True, charge_power: 2` means its inputs
   include powers of nuclear charge that our one-hot features do not supply. The disjointness
   test fed it 5-dim one-hot and it still produced sensible dipoles, so the adapter is close —
   but its absolute numbers are not trustworthy until this is settled.
2. **Output normalisation.** TFG outputs are raw normalised scalars; the property mean and MAD
   must be recovered for true Debye. The linear fit used in the disjointness test is a stand-in.
3. **The raw predictor.** `EnergyDiffusion.forward` returns negative squared target error, not
   the property. SMG's curvature must be computed on $f$, not on the reward.
4. **SchNet as a second $f_A$** — needs the linear-embedding adapter and a second-derivative check.

### A transformer f_A?

Worth considering precisely because **SchNet's architecture diversity currently only reaches the
evaluator side.** If $f_A$ is always EGNN-family, a shared inductive-bias failure in the guide is
invisible. A distance-biased invariant transformer would give the guide side its own axis.

Cost: ~40 min per model, ~1.4 h wall-clock for all six at 3 parallel jobs, plus implementation.

**Cheaper first:** try SchNet-as-$f_A$ via the embedding adapter (weights already downloaded and
verified). If its second derivatives behave, that gives guide-side architecture diversity for
free, and the transformer becomes optional rather than necessary.

### A linear f_A, as a null control

Not a competitor — a **correctness test**. A linear property has $H = 0$ exactly, so SMG must
collapse onto plug-in *identically*. We verified that analytically (4.4e-16 in the test suite);
running it end-to-end through the real sampler would catch an implementation bug no accuracy
metric would reveal. A linear model on invariant descriptors reaches ~0.5–1.0 D on μ, far worse
than $\delta$, so it can never be a working guide. About a minute to fit, being a least-squares
solve.
