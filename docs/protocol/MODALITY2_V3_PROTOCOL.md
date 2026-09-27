# Modality 2 under v3: the same protocol, on a modality that changes four of its terms

**Pre-registration. Written 26 September 2026, before any M2 guidance cell has
been run.** It supersedes the scoping note
[MODALITY2_V3_PLAN.md](MODALITY2_V3_PLAN.md), which asked *what it would take*;
this says *what will be run*.

It is deliberately the **same protocol** as
[FULL_RUN_V3_PROTOCOL.md](FULL_RUN_V3_PROTOCOL.md): one target, one headline
strength, a separate strength-swept ablation, three seeds, the batch as BDG's
estimator, no fidelity floor, and every configuration in the cell name. Where
M2 departs, it departs because the modality forces it, and §2 gives the reason
for each of the four departures.

> **Status of the numbers below.** Every parameter is marked
> **[RECORDED]** (a measurement exists in a file in this repository, cited),
> **[DIAGNOSTIC]** (measured and reported, but it does not set a parameter —
> every parameter comes from v3), or **[CHOICE]** (a free parameter, argued but not
> measurable — there is exactly one, and §2.3 is about it).
> An earlier audit of this stack found that most M2 settings existed only in
> code comments and commit messages: the scripts that produced them print to
> stdout and write no file, and `results/m2*` has never existed in any commit.
> This protocol does not inherit an unrecorded number.

---

## 1. What M2 is, in one table

| | M1 (v3) | **M2 (this protocol)** |
|---|---|---|
| data | QM9 molecules | DeepFlyBrain enhancers, 500 bp, official split |
| state | coords + feats + mask, EGNN | `[B, 500, 4]` on the simplex (Δ³)⁵⁰⁰, dilated 1-D CNN |
| generator | 3 base models (`fm`, `equifm`, `edm`) | **one**: `fm_m2_dfb500.pt`, frozen |
| step | Euler/Heun on ℝ³ⁿ | Euler **+ clamp-renormalise projection** each step |
| properties | mu, alpha, gap (3 learned pairs) | **gc, cpg** (2 analytic pairs) |
| f_A (guided) | a trained network | `gc_soft` / `cpg_soft`, differentiable, **exact** |
| f_B (scored) | a *different* trained network | `gc_hard` / `cpg_hard`, argmax-decoded, **exact** |
| δ | 2 × f_B's calibration MAE | **a choice** (§2.2) — f_B has no error to read |
| target | q50 | q50 — same |
| seeds | 20261001/2/3 | 20260921/22/23 |
| n, batch | 2000, 500 | 2000, 500 — same |
| NFE | 100 | 100 — same |
| window | t ≥ 0.5 | t ≥ 0.5 — same |
| floor | none; fidelity reported | none; fidelity reported |

**One generator means M2 has no cross-backend trap.** The whole of v3 §1.2 and
§5's "do not compare in_band between backends" is inapplicable here: every M2
cell is scored by the same δ against the same corpus, so every M2 row is
directly comparable to every other M2 row. This is the one respect in which M2
is *simpler* than M1.

### 1.1 The two properties, and why both

| property | f_A | curvature | s (corpus sd) |
|---|---|---|---|
| `gc` | GC content of the decoded sequence | **exactly affine** (Hessian identically zero) | 0.0552 |
| `cpg` | CpG dinucleotide density | **quadratic** | 0.01475 |

**Order of work: `gc` first, `cpg` as replication if time allows** (owner's
decision, 26 Sep). A third property is possible later and is not planned here.

> **The curvature justification these two properties used to carry is
> withdrawn.** Both this document's first draft and `m2_sweep.py`'s comments
> said the pair "isolates when a Tweedie second-moment correction can matter".
> **No arm in either modality computes such a correction.** The arms that would
> differ — `smg_mean`, `smg2` — are not in v3's arm set
> (`unguided,plug,tmpd,lgd_mc,tfg,bdg×2`) and M2's `guidance_field` has no
> Hessian term at all. The contrast as stated had nothing to test.
>
> **What `cpg` actually buys is replication**: a second property with a
> different functional form, so that a result is not a quirk of GC in
> particular. That is a real reason and it is the one to write up. It is also
> why `cpg` is deferred rather than dropped — replication is worth having, but
> it is not worth having before the primary property is measured at all.

M1 cannot run a known-curvature property at all: all three of its properties
are learned networks. That remains a genuine asymmetry in M2's favour — it is
just not one the current arm set exploits.

### 1.2 `tmpd` was a duplicate of `plug`, and the clip still makes it one

Three measurements, 26 Sep, in the order they were found. The first two are
fixed in code; **the third is not fixable by a code change and constrains how
`tmpd` may be reported.**

**(a) `v_f` was batch-constant, so TMPD had no per-sample weighting.** TMPD's
defining feature is that each sample's step is divided by *that sample's*
uncertainty `s² + v_f_i`. `m2_sweep.py` read `v_f` from `vf_table.json` as one
scalar broadcast over the batch (`torch.full_like`), on **both** properties, so
every sample got the identical factor. Measured spread of `s²/(s²+v_f_i)`
across a batch of 64: **~1e-6 (float noise) at every t, on `gc` and `cpg`
alike**. `tmpd` was DPS with a global step-size schedule. A second bug fed it:
`_const_grad` returned a gradient for `gc` and **zeros** for `cpg`, so `gAg`
was identically 0 there — though it was moot, since the table value overrode it
anyway.

**Fixed**: the per-sample gradient at `m` now comes from autograd and
`v_f_i = k_t · gAg_i`, with `k_t` calibrated so the batch **mean** of `v_f`
still equals the measured table value. The table keeps setting the scale; the
spread is restored. Verified: mean matches the table exactly at every t, and
the per-sample spread goes from ~1e-6 to **0.33 (`gc`) and 0.44 (`cpg`)** at
t = 0.1. Modality 1 already did this correctly with a JVP
(`guidance.py:1689-1691`), so this makes the two modalities run the same
method — which the transfer claim requires.

**(b) The clip erases what the fix restored.** TMPD modulates per-sample
*magnitude*. The clip caps per-sample magnitude at `clip × |v|`. They act on
the same axis, and the clip wins: any clipped sample gets rescaled to exactly
`clip × |v|`, identical for `plug` and `tmpd`. Measured at t = 0.1, `gc`,
64 samples:

| w | fraction of samples clipped | relative \|tmpd − plug\| after the clip |
|---|---|---|
| 1 | 94 % | 0.0073 |
| 4 | 97 % | 0.0005 |
| 16 | 97 % | 0.0021 |
| 64 | 97 % | 0.0081 |

So **under `clip = 1.0` and `t_min = 0`, `tmpd` is indistinguishable from
`plug` at every strength in the grid**, whatever `v_f` does.

**(c) The cause is the window, not the clip — and M1 proves it.** The
score-to-velocity factor is `(1-t)/max(t, 1e-6)`, which diverges as t → 0 and
is capped only at 1e6. Guiding from t = 0 therefore requests enormous
corrections at small t, and the clip truncates essentially all of them.
Modality 1 guides from t ≥ 0.5, where the factor is ~1 — and in M1's own
finished v3 cells **under 0.2 % of sample-steps clip** (64–406 of 200,000 on
our base), with `plug` and `tmpd` differing by 0.001–0.021 in-band and carrying
visibly different clip counts (406 vs 19). TMPD is a live baseline in M1 and a
dead one in M2, and the window is the whole difference.

**Consequence for the window (§2).** The window is no longer a one-sided
argument, and this is why M2 ends up on v3's t ≥ 0.5 rather than its own. Guiding
late steers after the sequence has committed; guiding from 0 runs at the clip
ceiling, where `w` barely matters and every magnitude-only method collapses
onto `plug`. `measure_window.py` sweeps `t_min ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5}`
and records **gap closure and clip saturation together**, so the choice of v3's
window is supported by this project's own measurement rather than asserted.

**Consequence for the write-up.** If the chosen window still clips heavily,
`tmpd` must be reported as *clip-limited*, not as "a baseline that performed
like DPS". Those are different claims and only the first is true.

## 2. M2 runs v3's protocol exactly, and the one thing it cannot

The transfer question the project asks is whether **the same innovation, under
the same protocol, still helps on a different modality**. Every setting M2
changes is a confound in that comparison, so M2 changes nothing it does not
have to. Wherever v3 fixes a number, M2 uses v3's number:

| | v3 (M1) | M2 | why |
|---|---|---|---|
| target | q50 | q50 | identical |
| strength | w = 1, every arm | w = 1, every arm | identical |
| window | t ≥ 0.5 | t ≥ 0.5 | identical |
| NFE | 100 | 100 | identical |
| n / batch | 2000 / 500 | 2000 / 500 | identical (§2.4) |
| seeds | 3 | 3 | identical count |
| arms (headline) | 7 | 7 | mapped one-for-one (§3) |
| ablation | 17 arms × w ∈ {1, 4} | 17 arms × w ∈ {1, 4} | identical grid |
| fidelity floor | none; reported | none; reported | identical (§3.1) |
| **δ** | 2 × f_B's calibration MAE | **a chosen band** | **§2.2 — impossible here** |

**Two earlier M2 departures are withdrawn**, both of which this document
previously argued for at length:

- **The window is v3's t ≥ 0.5, not t ≥ 0.** The case for t ≥ 0 rested on
  comments that no script produced and that contradicted each other (8.7 % vs
  17 % for the same condition, §2.1). Measuring it revealed the opposite
  problem: guiding from 0 saturates the clip on 94–97 % of sample-steps, so the
  clip rather than `w` sets the strength and every magnitude-only arm collapses
  onto `plug` (§1.2b). v3's window avoids that by construction — the
  score-to-velocity factor `(1-t)/t` is ~1 there, which is why **under 0.2 % of
  M1's own sample-steps clip**. Following M1 exactly is both the more faithful
  choice and the one that keeps the arms distinguishable.
- **NFE is v3's 100, not 400.** The case for 400 was real but is a *reporting*
  matter, not a protocol one: see §2.3.

The early window is not discarded, it is **demoted to a diagnostic**. The
ablation carries a `t_min = 0` rung on `plug` and the two headline BDG arms, so
the write-up can state what guiding from 0 would have done, from this project's
own cells, rather than from a comment.

### 2.2 δ is a **choice**, and it is the only unavoidable deviation **[CHOICE]**

This is the departure that constrains how every M2 number may be written up.

M1's rule is δ = 2 × f_B's calibration MAE: the band is "within the evaluator's
own resolution", and the evaluator is a network that *has* a resolution. **M2's
f_B is an exact count over the decoded sequence.** It has no error. There is
nothing to read a δ off, so δ is set by argument instead of by measurement:

```
delta = max(delta_ratio * s, 4.4 * quantum)      # m2_sweep.py:413-414
```

with `delta_ratio = 0.16`, `s` the corpus sd of the property, and `quantum` the
spacing of attainable values of the discrete decoded observable.

Two things about it are recorded here so they are not re-derived or quietly
changed:

- **Why a floor at all.** The observable is discrete. At ratio 0.16 the band
  spans 8.8 attainable values for `gc` (s = 0.0552) but only 2.4 for `cpg`
  (s = 0.01475) — `cpg`'s in-band would be quantisation-limited, measuring the
  lattice rather than the method. The `4.4 × quantum` floor makes `cpg` span
  the same ~8.8 values. **4.4 is not derived anywhere in the repository**; it
  is the number that makes the two properties span the same count, and that is
  its entire justification. It is recorded here as such.
- **A real inconsistency, fixed.** `quantum` is coded as `1/(crop-1)` for both
  properties, but `gc_hard` averages over **L = 500** positions while `cpg`
  averages over **L − 1 = 499** dinucleotides. `gc`'s quantum is 1/500.
  The difference is 0.2 % and changes no conclusion, but the protocol uses the
  per-property quantum and §7's gate asserts it.

**Consequence, and it is not optional.** An M1 in-band and an M2 in-band are
set by different *kinds* of rule — one measured from an oracle's error, one
chosen against a lattice. **They may never be pooled, averaged, or placed in
one column.** The comparison that *is* valid across the two modalities is the
**within-modality arm ranking**: does the same guidance method win on DNA as on
molecules? That needs no shared band, which is exactly why it is the question
to ask.

### 2.3 Two limitations that follow from matching v3 — **reported, not engineered away** **[RECORDED]**

The assignment's §4.7 asks for at least two meaningful limitations. These are
M2's, and both are consequences of running v3's protocol rather than a tuned
variant of it. Engineering either away would cost the exact comparability the
transfer claim depends on, so both are measured and reported instead.

**(a) At NFE 100 the guided quantity and the scored quantity differ by 5.5 %.**
BDG servos `gc_soft`; the table scores `gc_hard`. On real sequences the two
agree exactly; on generated ones they diverge until the sampler converges
(`blade_bundle/nfe_sweep.log`):

| NFE | `gc_hard` sd | `gc_soft` sd | gap |
|---|---|---|---|
| **100 (v3's, and ours)** | **0.0511** | **0.0483** | **5.5 %** |
| 200 | 0.0525 | 0.0505 | 3.8 % |
| 400 | 0.0531 | 0.0517 | 2.6 % |
| 1000 | 0.0534 | 0.0526 | 1.5 % |

So a setpoint τ lands ~5.5 % wide of where it was aimed — systematically, in
the same direction, for every BDG arm. NFE 400 would halve it, and an earlier
draft of this protocol chose 400 for that reason. **It is not worth a protocol
departure**: the bias is common to every arm at a given NFE, so it cancels in
the arm-against-arm comparison this study is about, and it costs 4× the
compute. The same table is the evidence for the limitation and for why the
limitation is tolerable. Composition converges the same way (G over-produced
0.240 vs 0.227 at NFE 100, 0.2287 at NFE 1000), so this is Euler
discretisation, not a learned bias.

**(b) M2's in-band is not commensurable with M1's** (§2.2). Different kinds of
rule set the two bands. They may be reported side by side and never pooled.

### 2.4 The batch is BDG's estimator here too **[RECORDED]**

Identical in principle to v3 §2.4, and it is a **change to M2's code**, which
currently draws one batch of `n` and therefore runs **one** controller.

`V_b` is the variance of the guided property across whatever tensor the sampler
is handed, so the batch size *is* the controller's sample size. v3 runs
`n ÷ 500` controllers of 500, each with a 6.3 % standard error on the variance.
**M2 must match it, or its BDG arm is a differently-estimated controller and
the two modalities' BDG rows are not the same method.**

- `--batch 500`, independent of `n`. At n = 2000 that is 4 controllers.
- **n must be divisible by the batch.** A remainder batch is a second, far
  noisier controller pooled in as an equal.
- Memory is not the binding constraint it is in M1 — the M2 network is 1.02 M
  parameters against EquiFM's, and one batch of 2000 would fit. 500 is chosen
  **to match M1's controller sample size**, so that a BDG row means the same
  thing in both modalities. That is the whole reason, and it is why this
  number is not re-derived from this card's memory.

---

## 3. The headline run

One target (q50), one strength, seven arms, two properties, three seeds.

| arm | what it is | M1's counterpart |
|---|---|---|
| `unguided` | the reference; no strength, no window | same |
| `plug` | DPS (Chung et al. 2023) — also BDG at η = 0 | `plug` |
| `tmpd` | TMPD / ΠGDM (Boys et al. 2024) | `tmpd` |
| `lgd_mc` | LGD (Song et al. 2023) | `lgd_mc` |
| `tfg_mc` | **TFG's MC-smoothing ingredient only**, not full TFG | `tfg` ≠ this |
| `bdg_e4t0.5` | ours, η = 4, τ_mult = 0.5 | same |
| `bdg_e4t1` | ours, η = 4, τ_mult = 1.0 | same |

> **`tfg_mc` is not `tfg`.** M1 runs TFG's full update; M2 implements only its
> Monte-Carlo smoothing ingredient. The write-up must not present them as the
> same arm, and no M2 sentence may say "TFG" unqualified. This is a difference
> in what was implemented, not a protocol choice.

**Cells:** 7 arms × 2 properties × 3 seeds = **42 headline cells.**

**Headline strength w = 1 [FROM v3].** v3's headline is w = 1 for every arm,
and v3 §2.2 is careful that this is equal *nominal* strength, not equal force —
the applied-correction share spans 5.8× across the arms it covers. M2 inherits
both the number and the caveat, so every M2 sentence carries "at w = 1" exactly
as every M1 sentence does.

An earlier draft fixed M2's strength by its own measurement (the smallest w
separating `plug` from `unguided`). **That is withdrawn**: a separately chosen
strength would mean the two modalities were not run under the same protocol,
which is the one thing the transfer claim cannot afford.
`measure_strength.py` is kept and its sweep over w ∈ {1, 4, 16, 64} is
**reported as a diagnostic**, because whether w = 1 is a strong or a weak
setting on DNA is itself part of "what had to be adapted for the new modality".

**Pairing.** Every arm sees the same seed, the same batch, the same initial
noise, the same target and the same δ, so arms are paired and a paired test is
valid — exactly as in v3 §3.

**Recorded per cell** (already produced by `m2_sweep.py:319-331`): `gc_mean`,
`gc_sd`, `bias_delta`, `in_band_fraction` (on the **decoded** property, so it
is M1's `in_band_fraction_dec`), `decode_conf`, `kmer_js`, `diversity`,
`clipped_sample_steps`, the `cost` dict, and per-arm `diag`. BDG cells also
record the controller state: `e`, `V_b/τ²`, `w_eff`.

### 3.1 No fidelity floor — the same decision v3 made, and the same consequence

v3 removed v2's chemistry gate: nothing is disqualified, fidelity is
**reported**. M2 adopts this, which settles an open item rather than adding one.

`m2_sweep.py:28-34` describes a k-mer-JS floor ("a cell passes if its 3-mer JS
is no worse than 1/0.9 × the unguided cell's") that **is implemented nowhere** —
`kmer_js` is recorded and never compared to anything. That prose is removed
rather than implemented, because implementing it would put M2 on v2's protocol
while M1 is on v3's, and the two modalities' tables would then mean different
things.

The consequence travels with every M2 number, as it does in M1: **an in-band
leaderboard would crown the arm that destroyed the most sequence fidelity.**
Every in-band figure prints beside `kmer_js`, `decode_conf` and `diversity`;
the summary names the best arm that did **not** lose fidelity against unguided;
no verdict is computed.

**The fidelity metrics have a floor of their own from the base model**, which is
what makes them readable: the frozen base scores 3-mer JS 0.0007 against real
test sequences, 24× better than uniform-random and 19× better than a
composition-matched control, with decode confidence 0.958 at NFE 100 and 0.988
at NFE 1000 (`blade_bundle/check_quality.log`, `REPORT.txt`). A guided arm's
`kmer_js` is read against that, not against zero.

---

## 4. The ablation

The same two knobs as M1's, on the same grid, **at two strengths**, so that the
BDG confound v3 §4.1 names — "controlling spread" and "pushing harder" being
one dial — is separated here as it is there.

```
eta ∈ {0, 1, 2, 4, 8}   ×   tau_mult ∈ {0.5, 0.75, 1, 1.5}
```

η = 0 kills the feedback regardless of τ, so it appears once: **17 arms**,
matching M1's `ABL_ARMS` exactly.

**Strengths:** the headline w and 4 × it, mirroring v3's w ∈ {1, 4}.

**Cells:** 17 arms × 2 strengths × 2 properties × 3 seeds = **204 ablation
cells**, ~5× the headline — the same shape of imbalance v3 has, for the same
reason. **Run the headline first.**

### 4.1 The two checks that make the BDG numbers admissible

Both are pre-registered as **pass/fail**, and a failure voids the BDG rows
rather than being reported as a result.

1. **`bdg` at η = 0 must equal `plug` at the same w, bit for bit.** It *is*
   BDG with the feedback switched off, so any difference is an implementation
   error, not a finding. Commit `01a9d17` claims this held to 0.00e+00 on four
   metrics; like everything else in that message it was never recorded, so it
   is asserted here as a gate on the run's own cells.
2. **`tmpd` must be exactly parallel to `plug` on `gc`**, with field ratio
   `s²/(s²+v_f(t))`. `gc` is exactly affine, so the second-moment correction is
   identically zero and only the uncertainty denominator separates them. This
   is the known-answer control of §1.1, and M1 has no equivalent. **Verified
   26 Sep**: cos = 1.0000000000 and the ratio matches the prediction to 7
   decimals at t ∈ {0, 0.1, 0.3, 0.5}. It is a gate on the field, not on end
   metrics — the clip and the sampler wash the difference out downstream, so
   comparing `in_band` would pass a broken implementation.

   Note what this control does **not** say: `tmpd` and `plug` are *not* equal,
   and an earlier draft of this protocol asserted that they were. They differ
   by the denominator, which at t = 0 is a factor of two. §1.2 records why the
   same check passing on `cpg` is a defect rather than a reassurance.

A third check is **reported, not gated**: `clipped_sample_steps` across the two
strengths. v3's ablation warns that the clip is applied *after* w, so
quadrupling w on a request the clip already truncates may change nothing but
how much is thrown away. Compare the clip counts across strengths **before**
attributing any difference to strength, and mark a rung clip-limited where it
is.

---

## 5. Cost, and choosing n

**n = 2000, batch 500, 3 seeds**, matching M1 so that the two modalities'
tables have the same statistical footing.

`in_band` is a proportion, so its standard error is √(p(1−p)/n): **1.1 pp at
n = 2000** against 1.5 pp at n = 1024 (M2's old default). Arms in this project
separate by ~3 pp, so n = 1024 could resolve a large effect and not a small
one. As in v3 §6.1, **a null is only evidence against BDG if the run could have
seen the effect** — any null must be reported beside the minimum difference the
run could detect.

M2 is far cheaper per cell than M1: 1.02 M parameters against an EGNN, an
analytic property instead of a network forward-and-backward, and no RDKit
validity pass. The NFE 400 of §2.2 spends some of that back. **The per-cell
cost is measured in §7 step 3 and recorded before the full run is launched** —
this protocol does not carry a GPU-hour estimate it has not measured, which is
the error v3's own budget section had to withdraw.

---

## 6. What this protocol can and cannot settle

**Can:**
- Rank the guidance arms *within* M2, paired, at one shared strength.
- Say whether the second-moment correction does anything where its curvature is
  known to be zero (§1.1) — which M1 cannot.
- Separate BDG's two knobs, and say where the clip limits the answer (§4).
- Support the cross-modality question that matters: **does the arm ranking
  transfer from molecules to DNA?**

**Cannot:**
- Compare an M2 in-band to an M1 in-band (§2.3). Not with a caveat; not at all.
- Say which arm is "best" at its own optimum — every number carries "at w = ⟨the
  headline strength⟩", exactly as v3's carry "at w = 1".
- Say anything about full TFG (§3).
- Generalise past DeepFlyBrain enhancers at 500 bp.

---

## 7. Running it

Every parameter comes from v3, so there is nothing to measure first. The two
diagnostic measurements can run before, after or alongside the sweep; they
inform the write-up, not the settings.

```bash
# the sweep. gc first; cpg is replication and runs only if time allows (§1.1)
python proj1/m2/run_sweep.py --stage m2    --props gc --n 2000 --device cuda
python proj1/m2/run_sweep.py --stage m2abl --props gc --n 2000 --device cuda

# split across GPUs by seed -- cells are independent and skipped if present
python proj1/m2/run_sweep.py --stage m2 --props gc --seeds 20260921 --device cuda

# check cells as they land; exits 1 on any failure
python proj1/m2/m2_sanity.py --stage m2 --n 2000

# the tables
python proj1/m2/m2_table.py --stage m2    --n 2000
python proj1/m2/m2_table.py --stage m2abl --n 2000 --w 1
python proj1/m2/m2_table.py --stage m2abl --n 2000 --w 4

# diagnostics for the write-up (§2.1 window, §3 strength) -- not parameters
python proj1/m2/measure_window.py --props gc --device cuda
python proj1/m2/measure_strength.py --props gc --device cuda
```

Cells land in `results/m2/<stage>/n<N>/seed<S>/`, named
`<prop>__<arm>__q50__w<w>__<variant>__n2000_nfe100_win0.5_b500_dr0.16__s<seed>.json`.

### 7.1 Code prerequisites — **all six done, 26 Sep**, listed with what each prevents

Each was found by the audit in [MODALITY2_V3_PLAN.md](MODALITY2_V3_PLAN.md)
§2.2 and each would corrupt a result silently rather than loudly.

1. **The cell name must carry the configuration.** It is currently
   `<prop>__<arm>__<target>__w<w>__<variant>__s<seed>.json` — no `n`, no
   `steps`, no `t_min`, no `batch`, no `delta_ratio`. Resume is
   skip-if-exists, so **an n = 128 smoke cell and an n = 2000 real cell have
   the same filename and the smoke cell is silently kept.** This is the exact
   failure M1's `config_tag` exists to prevent. Mirror it:
   `n2000_nfe400_win0_b500_dr0.16`.
2. **`--batch`** (§2.4), with a refusal when it does not divide `n`.
3. **A stage directory**, mirroring v3: `results/m2/<stage>/n<N>/seed<S>/`.
4. **The default checkpoint is the wrong model.** `m2_sweep.py` defaults
   `--ckpt` to a 200 bp model while `run_sweep.py` defaults to the 500 bp one,
   and the corpus and k-mer reference are chosen by branching on the
   checkpoint's `crop`. A bare `m2_sweep.py` call loads the wrong model *and*
   scores it against the wrong reference distribution. Make the default the
   500 bp checkpoint, or refuse without `--ckpt`.
5. **`m2_table.py` did not exist** — nothing in this repository read M2 cells.
   It now refuses to pool cells whose `n`, `batch`, `steps`, `t_min`, `delta`,
   `delta_ratio`, `target` or `ckpt` disagree, refuses to pool the ablation's
   two strengths, and computes no verdict (§3.1).
6. **The unimplemented floor prose is removed** (§3.1).

### 7.2 Gates

`proj1/m2/test_m2_protocol.py` mirrors `proj1/tests/test_v3.py`: **38
closed-form checks, no GPU**, run after any edit to this protocol, the driver,
the cell runner or the table — it is what keeps those four in step.

It asserts the cell-name config tag carries all five fields, that `n % batch`
is refused *by running the driver* rather than grepping it, that the per-property
quantum is used (§2.3), that NFE 400 / window 0 / batch 500 are the defaults in
**both** entry points, that the ablation grid is exactly M1's 17 arms over
η ∈ {0,1,2,4,8} × τ ∈ {0.5,0.75,1,1.5} at two strengths, that the driver
refuses a missing `--w`, and that no fidelity floor exists in code.

**The gates are mutation-tested.** Reverting NFE to 100, removing the batch
refusal, removing the `--w` requirement, or restoring the shared `1/(L-1)`
quantum each makes exactly the corresponding gate fail — so a gate that passes
is checking the thing it names, not a string that happens to be present.

---

## 8. Status

**Pre-registration, not a result.** No M2 guidance cell exists at the time of
writing — `results/m2*` has never existed in this repository, in the working
tree or in any commit. Nothing in this document may be cited as evidence of an
outcome, and the three measurements in §7 are the only things that may change
its parameters. If one of them contradicts a pre-registered value, the value
changes **here, before the run**, with the contradiction recorded — not
afterwards.
