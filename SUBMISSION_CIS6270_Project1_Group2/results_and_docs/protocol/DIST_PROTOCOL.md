# The `dist` protocol — literature backing, theory, metrics

**23 September 2026.** How the full run scores guidance on per-molecule targets, why
that is the published protocol, and what each number means. Code:
`proj1/src/dist_metrics.py`, `proj1/scripts/dist_report.py`, gates in
`proj1/tests/test_dist_metrics.py` (32/32).

**Relation to the pre-registration.** The headline verdict is FR5 in
`proj1/scripts/full_run_table.py`: BTVG "beats" an arm only at an *independent-samples*
z ≥ 3, fixed on 22 Sep before any full-run cell existed. Nothing here re-decides that.
This document adds the reference ladder that places the full run beside the literature,
plus supplementary paired statistics. Where the two disagree, FR5 is the answer.

---

## 1. What `dist` is

Molecule *i* of a batch is generated **at the atom count of held-out test molecule
*i***, and its target is **that same molecule's real property**. Sizes and targets come
from the same molecule, because size alone carries much of the property
(corr(size, α) = +0.755). Pairing them separately once produced a 10-atom molecule asked
to hit a 25-atom molecule's polarisability (`SCOPE_FM_GUIDANCE_STATUS.md`, FR6).

Our generator only sees the atom count. Every QM9 mask is a prefix of ones (verified
on all 133,885 molecules), so the count is the only information the mask carries.

---

## 2. Who else does this — verified against the papers

`SCOPE_FM_GUIDANCE_STATUS.md` flagged this as unverified: *"cite the EDM/EEGSDE sampler
code or paper text for the conditioning protocol."* Both papers were read on 22 Sep.

**EDM** (Hoogeboom et al., ICML 2022), §5.2 and Appendix E:

> "We split the QM9 training partition into two halves D_a, D_b of 50K samples each.
> The classifier φ_c is trained on the first half D_a, while the Conditional EDM is
> trained on the second half D_b."

> "we define the generative process by first sampling c, M ∼ p(c, M) and then
> x, h ∼ p(x, h | c, M). We compute c, M ∼ p(c, M) on the training partition as a
> parametrized two dimensional categorical distribution"

Its baselines, and how to read them:

> "If 'EDM' overcomes 'Naive (Upper-Bound)' it should be able to incorporate conditional
> property information into the generated molecules. If it overcomes '#Atoms' it should
> be able to incorporate it into the molecular structure beyond the number of atoms."

**EEGSDE** (Bao et al., ICLR 2023), §5.1 and Appendix F:

> "The training set is further divided into two non-overlapping halves D_a, D_b equally.
> The noise prediction network and the time-dependent property prediction model of the
> energy function are trained on D_b separately."

> "we train another property prediction model φ_p on D_a ... For fairness, the property
> prediction model φ_p is different from g(z_t, t) used in the energy function."

> "Following the EDM, we firstly sample the number of atoms in a molecule M ∼ p(M) and
> the property value c ∼ p(c|M) ... Then we generate a molecule given M, c."

**The mapping is exact on the part that matters:**

| | EDM / EEGSDE | ours |
|---|---|---|
| generator + guide | D_b | `train_a` |
| evaluator | D_a, a different network | `f_B` on `train_b` |
| (c, M) | joint, from the training partition | joint, empirical pairs from `test` |
| metric | MAE of evaluator vs target | same, plus band coverage |
| samples | K = 10,000 | n = 5,000 × 3 seeds (FR4 amendment) |

### Deviations, stated so they are disclosed rather than discovered

1. **Our generator is unconditional.** EEGSDE's headline guides a *conditional* network.
   The comparable published row is EEGSDE **Table 5, unconditional network + guidance**:
   μ MAE 1.415 at s = 0.1 and 1.241 at s = 0.5. **That is the row to quote beside ours**,
   not EEGSDE's 0.777.
2. **(c, M) come from `test`, not a parametric fit on training data.** Non-parametric,
   no binning, and it avoids reusing the training partition. It is the same joint law.
3. **Our evaluator is weaker.** Our L-bound is 0.085 D against their 0.043 D. Everything
   below the evaluator's own error is unresolvable for us and not for them.

---

## 3. The theory: atom count is the confound, so read everything against a ladder

A generator handed the right atom count, and **ignoring the target completely**, already
hits part of every target. So a raw MAE says nothing until it is placed against
references that each know a different amount. All four are computed from real test
molecules and `f_B` alone, with no sampling and no RNG, as exact expectations:

| rung | what it computes | knows |
|---|---|---|
| **U-bound** | `f_B`(another test molecule) vs cᵢ | nothing |
| **size-shuffle** *(ours)* | `f_B`(another molecule **of size Mᵢ**) vs cᵢ | M, not c |
| **#Atoms** | median(c \| M) from `train_a`, vs cᵢ | M, as a point |
| **L-bound** | `f_B`(the real molecule *i*) vs cᵢ | evaluator floor |

U-bound, #Atoms and L-bound are EDM's rows. **Size-shuffle is new, and it is the one
that matters for us.**

**Why a fourth rung.** #Atoms is a deterministic point predictor. A *sampler* that knows
the size but ignores the target draws from p(c | M) rather than sitting on its median,
and pays for its own spread. For a Gaussian within-size law, E|c − c′| = √2 · E|c − median|.
So **a perfect generator that ignores the target lands on size-shuffle, not on #Atoms.**
Measured on real QM9, the ratio is **1.406 (μ), 1.437 (α), 1.381 (gap)** against √2 = 1.414.

Two consequences:

- **#Atoms is a harsher bar than it looks for any sampler.** EEGSDE's own
  unconditional-plus-guidance row (1.241 D) *fails* EDM's #Atoms test (1.053) yet clearly
  beats size-shuffle (1.496). Judged by EDM's criterion alone, unconditional guidance looks
  like no gain; against size-shuffle, it is a real partial gain. Both are reported.
- **The size-shuffle rung is empirically right.** Our own unguided generator, scored on
  the same 512 dist molecules as the rung, lands on it on every axis:

  | | unguided | size-shuffle |
  |---|---|---|
  | MAE (D) | 1.537 (95 % CI 1.43–1.66) | 1.499 |
  | in-band | 0.082 | 0.083 |
  | gap_closure | −0.03 | 0 by definition |
  | partial_corr | 0.05 | ~0 by construction |

  A real target-blind generator lands exactly on the rung built to describe one.
  Artifact: `results/dist_checks/n512_unguided/` (cell, sidecar and `report.json`).

From the ladder, one number comparable across properties:

```
gap_closure = (size_shuffle − MAE) / (size_shuffle − L_bound)
```

0 means the arm behaves like a size-aware generator that ignores the target. 1 means it
does as well as scoring the real molecule.

And the continuous form of EDM's "beyond the number of atoms" criterion: **`partial_corr`**,
the correlation of `f_B`(generated) with cᵢ *within* atom-count groups. A generator that
uses size but ignores the target scores ~0, however well it matches #Atoms, because all of
its tracking is between size groups. The ceiling is the real molecules' 0.99.

---

## 4. Validation: our ladder reproduces EDM's published baselines

Before any arm is scored, the protocol is checked by recomputing EDM's own baselines on
our data (full test set, n = 13,083):

| | ours | EDM Table 3 | diff |
|---|---|---|---|
| μ U-bound (D) | 1.647 | 1.616 | +1.9 % |
| μ #Atoms (D) | 1.064 | 1.053 | +1.1 % |
| α U-bound (Bohr³) | 8.971 | 9.01 | −0.4 % |
| α #Atoms (Bohr³) | 3.875 | 3.86 | +0.4 % |
| gap U-bound (meV) | 1479 | 1470 | +0.6 % |
| gap #Atoms (meV) | 862.5 | 866 | −0.4 % |

**All six are within 2 %.** Our data, split and target construction reproduce the
published setup. (L-bounds are not expected to match: that rung measures the evaluator,
and ours is weaker.)

### The full ladder

| | μ (D) | α (Bohr³) | gap (meV) |
|---|---|---|---|
| δ (tolerance) | 0.168 | 0.481 | 206.7 |
| U-bound — MAE / in-band | 1.647 / 0.069 | 8.971 / 0.035 | 1479 / 0.090 |
| **size-shuffle** — MAE / in-band | **1.496 / 0.082** | **5.567 / 0.063** | **1192 / 0.120** |
| #Atoms — MAE / in-band | 1.064 / 0.107 | 3.875 / 0.093 | 862.5 / 0.143 |
| L-bound — MAE / in-band | 0.085 / 0.884 | 0.240 / 0.895 | 104.1 / 0.875 |

**The sobering row is #Atoms in-band.** A lookup table of median property by atom count
puts **10.7 %** of μ targets in band. On the project's primary metric, an arm must beat
that just to outperform a table that never looks at the molecule.

---

## 5. Metrics

### Primary — the project rubric and FR5, unchanged

- **`in_band`**, gain over `unguided`, subject to `mol_stability` ≥ 0.9 × unguided
  (the rubric in `project-arm-ranking-rubric`), with Wilson 95 % CI.
- **FR5 verdict** from `full_run_table.py`: independent z ≥ 3.

### Supplementary — what this protocol adds

| metric | answers |
|---|---|
| **MAE** in published units, bootstrap 95 % CI | the number EDM/EEGSDE report; gap converted Hartree → meV |
| **gap_closure** | how much of the target information the arm captured, comparable across properties |
| **beats #Atoms** | EDM's stated criterion, verbatim |
| **partial_corr** | tracking *beyond* atom count, against a ceiling of 0.99 |
| **valid ∧ in-band**, **distinct valid in-band per attempt** | the project's headline yield. It needs per-molecule SMILES, now recorded in the sidecar |
| **hack_rate** | P(guide says in band ∧ evaluator disagrees): EEGSDE's "for fairness" separation, quantified |
| **seed sd of MAE** | error bars measured from seeds, not assumed |
| **in-band by steering demand** | near / mid / far from a typical molecule **of the same size** |
| **in-band by size tercile** | an arm can win overall by working on one size only |

### Why the demand strata are not optional

The smoke test showed the aggregate in-band can penalise guidance for doing the right
thing. Unguided in-band was **0.37 on near targets and 0.095 on far ones**: a target-blind
generator scores well only where targets are typical. Guidance traded near-target coverage
for far-target coverage (btvg far 0.19). The aggregate hides that trade. *(n = 64, a
pipeline check, not a result; artifact `results/dist_checks/n64_smoke/`.)*

That smoke run also produced a false alarm worth recording: unguided in-band **0.234**
(15/64), +4.3 sd above the size-shuffle expectation, and "significant" at p = 0.002 on a
within-size permutation test. The test was run *because* the number looked odd, which is
how post-hoc significance misleads. At n = 512 the same generator scored 0.082, on the
rung. No leak: every mask carries only the atom count. **Do not read a dist cell below
n ≈ 500.**

The demand here is |cᵢ − median(c | Mᵢ)| / sd(c | Mᵢ): distance from a typical molecule
**of the same size**, because the size is fixed. `full_run_table.py`'s steering breakdown
uses distance from the unguided generator's own mean, which is a different quantity. Both
are reported, and neither replaces the other.

---

## 6. Statistical protocol

1. **Same molecules, same targets, same noise across arms.** Every arm in a seed shares the
   5,000 test molecules and the initial noise, so comparisons pair exactly.
2. **Pairing on (seed, mol_idx), never on row position.** Matched rows are *required* to
   have equal targets or the comparison raises. That guard exists because the batch loop
   once sliced every batch's targets from index 0, which would have guided 75 % of
   molecules toward another molecule's target while scoring them against their own.
3. **Seeds pooled, not tested separately.** Seeds are replicates of one dist set, so each
   comparison is one test on the pooled rows. Pairing buys variance, not the point estimate:
   the mean paired difference equals the difference of means. The test gate shows a correct
   pairing at |z| = 35 against 6.4 for a wrong one.
4. **One strength per arm.** Frozen by FR3 (q90 best-MAE strength). Arm-vs-plug is only
   paired at frozen strengths, because nominal `w` is not a shared axis (btvg's w = 4 is
   applied as 0.0124 against plug's 4.0).
5. **Supplementary family:** each arm vs `unguided` and vs `plug`, Holm-adjusted within
   each property. Exact sign test for paired in-band.

---

## 7. Can we generate at a restricted atom count? Yes, natively

The generator is conditioned on atom count through the mask. Set the mask to *k* atoms and
every sample has exactly *k* atoms. `dist` already does this per molecule. EDM does the
same thing deliberately: its Figure 9 sweeps α over [73.6, 101.6] "while keeping the
reparametrization noise ε fixed and the number of nodes M = 19".

**Limits worth knowing before relying on it:**

- **Total count only, not composition.** The mask fixes atoms *including hydrogens*.
  Heavy-atom count, or "exactly one nitrogen", is chosen by the model through the type
  features and is **not** constrained. That would need type conditioning, inpainting, or a
  composition guide.
- **Stay inside the trained range.** QM9 has 3–29 atoms (1–9 heavy). Test-set sizes below
  9 or above 25 are nearly empty (for example, 3 molecules at 29), so those sizes are barely
  trained.
- **The target must be feasible at that size.** α is strongly size-determined. A fixed-*k*
  sweep should take its targets from p(c | M = k), for example its quantiles, not from the
  global distribution, or the arm is asked for something no molecule of that size has.

---

## 8. What is not backed, and open items

- **The 3-seed × 5,000 scale is below EEGSDE's K = 10,000.** It was halved before any cell
  existed (FR4 amendment, compute). Disclosed; CIs come from the pooled 15,000.
- **EDM computes its references on D_b; ours use `test`.** The same law, and section 4
  confirms the values agree.
- **`partial_corr` and size-shuffle are our constructions.** They are motivated by EDM's
  criterion and tested against closed forms, but they are not published metrics. Say so.
- **SMILES in the sidecar need the updated `evaluation.py` on Betty.** Without it,
  `distinct_valid_in_band_per_attempt` reports "–" and everything else still runs.

---

## 9. Reproduce

```bash
python proj1/tests/test_dist_metrics.py                        # 32 gates
python proj1/scripts/dist_report.py --ladder-only --n 13083     # section 4, ~1 min
python proj1/scripts/dist_report.py --dir "results/full/n5000/seed*"
```

The full run itself is `proj1/cluster/full_run.slurm`, already written. Submit it with the
procedure in `docs/protocol/BETTY_RUNBOOK.md` (step 3, "The full run is different"). It
writes the `.permol.pt` sidecars this report reads.
