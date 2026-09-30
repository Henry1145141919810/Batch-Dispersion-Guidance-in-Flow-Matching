# v3 ablation: BDG's two knobs, on the same protocol as the headline

**26 September 2026. Pre-registration. Nothing here has been run.** This
document governs `--stage v3abl` only. The headline is
[FULL_RUN_V3_PROTOCOL.md](FULL_RUN_V3_PROTOCOL.md), and everything this stage
does not override is inherited from it.

**Final form, 26 Sep (Henry).** This stage runs the **same n and the same
three seeds** as the headline. Two earlier same-day drafts had it smaller —
one seed, then a smaller n as well — and both are withdrawn. What they were
working around was that `n` used to be the only thing separating the two
stages' result trees; `transfer_sweep` now writes
`results/v3/<backend>/<stage>/n<N>/seed<S>/`, so the **stage directory** does
that job and the two stages are free to match.

⚠️ **The figure "51.3 GPU-h" that earlier drafts contrasted this run against
was never the headline.** It was *this stage's own first plan* — 14 grid arms
in a shared n = 5000 tree. Anywhere that number appears as "the headline", it
is wrong.

---

## 1. What it is

| setting | ablation | headline, for contrast |
|---|---|---|
| grid | η ∈ {0, 1, 2, 4, 8} × τ_mult ∈ {0.5, 0.75, 1.0, 1.5} | η = 4 × τ_mult ∈ {0.5, 1.0} |
| arms | **17** (all of the grid) | 7 (5 comparison + 2 BDG) |
| comparison arms | **none** — see §4 | all five |
| n per cell | **the same as the headline**, whatever the operator chose | — |
| seeds | **the same three** | the same three |
| properties | mu, alpha, gap | the same |
| base models | **`fm` and `equifm`** | those two **+ `edm`** |
| target | q50 | the same |
| **strength** | **w ∈ {1, 4} — swept** | w = 1, fixed |
| window | t ≥ 0.5 | the same |
| property pair | per backend, inherited (`fm` ours, `equifm` TFG's) | the same |
| chemistry floor | none | none |
| **cells** | **612** = 17 arms × **2 strengths** × 3 props × 3 seeds × 2 backends | 162 |
| tree | `results/v3/<backend>/v3abl/n<N>/seed<S>/` | `…/v3/n<N>/…` |

**This stage is much the LARGER of the two** — 17 arms at two strengths
against 7 arms at one, so roughly **3× the headline's cost** (at n = 2000:
~50 GPU-h against ~15; see [V3_POWER.md](../results/V3_POWER.md)). That is why
`submit_v3.sh` defaults to the headline and leaves this for a second window,
once the headline's `[timing]` lines have settled the real per-pass rate.

**The QM9 diffusion base sits this out.** It is declared `unguided`, `plug`
only, and the grid plans neither, so its nine array tasks exit 0 immediately
and say so. The array grid stays uniform (backend × property × seed) because a
ragged one is how stride bugs happen.

**η = 0 appears once, at τ_mult = 1.0.** At η = 0 the dispersion term is
multiplied by zero, so τ_mult cannot change those samples; four τ_mults would
be one computation under four names and pooling them would look like four
independent measurements of one thing. It is the identity rung: `w_eff` ≡ 1
whatever the controller measures, which is why it reproduces `plug` **in the
guidance field** regardless of `V_b`.

**All 17 arms run, including the headline's two** (η = 4 at τ_mult 0.5 and
1.0). The two stages write separate trees, so there is nothing to reuse — and
that duplication is useful: the two copies are at the same n and the same
seeds, so they should agree within seed noise, which is a free consistency
check on the whole harness.

### 1.1 Why the strength is swept here and nowhere else

The headline fixes **w = 1 for every arm** and keeps the caveat that equal `w`
is equal *dial setting*, not equal *force* (headline §2.2). Normalising the
strengths instead — equalising applied correction share, or chemistry cost, or
renormalising the field — was considered and **declined** (Henry, 26 Sep). The
headline is unchanged.

This stage sweeps **w ∈ {1, 4}**, and not for coverage. It is the control the
BDG confound needs.

**The algebra, read off the code.** BDG's numerator is
`num_i = (y − F_i − c) − η·e·(F_i − F̄)`, and
[`sampling.py:595-597`](../../proj1/src/sampling.py#L595-L597) multiplies the
whole guidance field by `w`. Grouping the terms:

> **correction ∝ w · [ (y − c − F̄) − w_eff·(F_i − F̄) ] / den**, with
> **w_eff = 1 + η·e**

So **`w` scales the mean term and the deviation term together, while `w_eff`
reweights the deviation term alone.** They are two different knobs on one
field, not one knob under two names.

That matters because the headline protocol's §6 lists, as something v3
*cannot* settle: *"nothing separates 'controlling spread helped' from 'pushing
harder helped' — at q50 those are the same dial."* With a strength axis they
are no longer the same dial. If raising `w` at fixed (η, τ_mult) reproduces
what raising `w_eff` does, the dispersion term is doing nothing that a larger
strength would not have done; if it does not, the controller is contributing
something of its own.

**⚠️ A w = 4 row may be clip-limited rather than controller-limited, and you
must check before reading it.** The correction passes a velocity-relative clip
that is applied **after** `w` ([`sampling.py:598-602`](../../proj1/src/sampling.py#L598-L602)).
At τ_mult 0.5 the measured `w_eff` is already of order **10³**, with **2516**
clipped sample steps against `plug`'s 884
([BDG_LADDER_MEASURED.md](../results/BDG_LADDER_MEASURED.md)). Quadrupling `w`
on a request the clip is already truncating may change nothing except how much
is thrown away.

Every cell records `clipped_sample_steps`. **Compare it across the two
strengths before attributing any w = 1 → w = 4 difference to strength**, and
if the clipped count rises while the metrics do not move, report the rung as
clip-limited rather than as evidence about the controller. The clip cannot
simply be widened: removing it produced 222 non-finite samples against 0
([CLIP_PILOT.md](../results/CLIP_PILOT.md)).

**Tables are per strength.** `v3_table.py` refuses to pool cells whose `w`
disagrees — averaging w = 1 and w = 4 would average the axis this stage exists
to sweep — so the chain writes `V3_RESULTS_ABL_w1*.md` and
`V3_RESULTS_ABL_w4*.md` and they are read side by side.

### 1.2 This stage carries the τ_mult 0.75 and 1.5 rungs alone

The headline keeps only the endpoints, 0.5 and 1.0. So **any claim that the τ
ladder is monotone, or that it curves, must be read off this grid's four
points** — the headline's two can show a difference and not a shape.

## 2. The batch is the estimator, and it does not depend on n

For every other arm the batch is a performance knob. For BDG it is **the
estimator**: `V_b` is the variance of the guide's predictions across the
molecules in the batch, so the batch size *is* the controller's sample size.

**`--batch 500`, whatever n is.** It is the largest round batch that fits the
tighter backend inside the working budget, and that is a property of the card,
not of n. A cell therefore runs **n ÷ 500 controllers**, each estimating `V_b`
from 500 molecules — a **6.3 % standard error on the variance, independent of
n**. Choosing a larger n buys *more* controllers, not better ones, so the
cell-level controller diagnostics get less noisy while each controller stays
equally good. **n must be divisible by 500**; `test_v3.py` and the job both
refuse otherwise.

⚠️ **The memory figures are a laptop extrapolation, not a cluster
measurement.** An earlier draft of this section stated "17.57 GiB on our base
and 26.65 GiB on EquiFM … confirmed on Betty 26 Sep"; **that measurement does
not exist in this repository** and is withdrawn — see FULL_RUN_V3_PROTOCOL.md
§2.4. The chain's `preflight` link does run the real check
(`batch_memprobe.py --confirm 500`) before the array starts. Read what it
writes, especially if the operator picked a large n.

---

## 3. What the grid is expected to show, before it runs

Measured on real trajectories at this run's own configuration
([BDG_LADDER_MEASURED.md](../results/BDG_LADDER_MEASURED.md)), η = 4:

| | τ_mult 0.5 | 0.75 | 1.0 | 1.5 |
|---|---|---|---|---|
| `V_b/τ²` run-mean | 256.6 | 114.1 | 64.0 | 28.6 |
| `w_eff` run-mean | 1019 | 456 | 253 | 111 |
| `w_eff` RMS | 4815 | 2142 | 1204 | 534 |
| share of guided steps `w_eff` < 0 | 0.00 | 0.08 | 0.16 | **0.62** |
| clipped steps (plug = 884) | **2516** | 2270 | 1853 | 1752 |

Four things this pre-registers as the expectation:

1. **`V_b/τ²` follows `1/τ_mult²` to within 0.5 %.** The controller's setpoint is
   `V_b/τ² = 1`; extrapolating the same law puts it at **τ_mult ≈ 8**. Every rung
   in this grid therefore sits **28× to 257× above the setpoint.** The grid
   explores the concentrating, saturated end of the knob and nothing else.
2. **The closed form `w_eff = 1 + η(1/τ_mult² − 1)` is withdrawn.** It predicted
   13.00 / 4.11 / 1.00 / −1.22 and is two orders of magnitude low, because it
   assumed `V_b = y_std²`, which holds on one synthetic `x_t` and not during
   sampling. Order rows by **measured** `w_eff`; never quote the closed form.
3. **`w_eff` is not mean-like.** RMS exceeds the signed mean by ~4.7× on every
   rung, and at τ_mult 1.5 the deviation term **reverses on 62 % of guided
   steps** — spreading the batch instead of concentrating it. Cells therefore
   record `bdg_w_eff_sq` (→ RMS) and `bdg_w_eff_neg` (→ that fraction) as well as
   the signed mean, which alone was unfalsifiable. These were added on 26 Sep and
   are free at run time but **unrecoverable afterwards**.
4. **The clip is a live confound and may be the dominant one.** BDG rungs clip
   ~2.8× more often than `plug`. At `w_eff` of order 10³ the dispersion term sits
   far outside the velocity-relative trust region, so what reaches the sample is
   the clip's truncation of the request, not the request. The clip cannot be
   widened — removing it produced 222 non-finite samples against 0.

---

## 4. What this ablation can and cannot settle

**Can.** Whether the deviation weight moves coverage at q50 at all; whether
that effect is monotone in **measured** `w_eff`; whether η does anything beyond
rescaling an already-saturated term; whether the η = 0 identity holds; whether
any of it looks different on a borrowed base model; and — new with the strength
axis (§1.1) — **whether the dispersion term does anything a larger `w` would
not have done**, since `w` scales the mean and deviation terms together while
`w_eff` reweights the deviation term alone.

**Cannot, and must not be claimed:**

- **That any rung beat `plug`, `tmpd`, `lgd_mc` or `tfg` — from this stage
  alone.** There is **no comparison arm here** (Henry, 26 Sep, explicitly
  declined), so the grid is primarily read against itself. What *has* changed
  is that the headline's `plug` is now at the **same n, the same three seeds
  and the same property pair**, in the sibling stage tree — so a comparison
  against it is defensible where it was not before. It is a comparison
  **across stage trees**, which must be stated, and it inherits the full
  17-way selection problem below.
- **That any rung won, without a multiplicity correction.** 17 settings × 3
  properties × 2 bases. "Did *any* rung win?" is a **max-over-17** selection;
  `BDG_REVIEW.md` already named "the max-over-13 selection" as the larger
  threat to the port's apparent result. A max-T or Bonferroni threshold must
  be stated beside any win claimed from this grid, and `v3_table.py`'s printed
  z is **not** selection-adjusted.
- **A difference smaller than the run can resolve.** The resolvable difference
  is set by the n the operator chose — see
  [V3_POWER.md](../results/V3_POWER.md) — and this stage's 17-way selection
  makes its own threshold wider than the headline's at the same n. Report the
  minimum detectable difference beside a null.
- **That a rung's result came from the controller rather than the clip**
  (§3.4). Measured: `plug` clips on 884 sample steps, `bdg_e4t0.5` on
  **2516**. This is the single most likely explanation for a null and must be
  offered as such, not as "spread control does not help". **The w = 4 rows are
  the most exposed to it**, because the clip is applied after `w` — check
  `clipped_sample_steps` across the two strengths before reading anything into
  a strength difference (§1.1).
- **That `w` and `w_eff` are the same knob.** They are not (§1.1), and the
  strength axis is what demonstrates it. But the converse claim also needs
  care: showing that `w` and `w_eff` move a metric *similarly* does not prove
  the controller is useless, only that at these settings it buys nothing a
  cheaper dial would not.
- **Anything about the setpoint regime.** No rung is within 28× of it (§3.1).
  If the grid comes back flat, the honest reading includes "we never tested
  τ_mult near the value where the controller could equilibrate" — τ_mult ≈ 2–8
  is the obvious follow-up and is *not* in this run.
- **A monotone reading of the τ_mult ladder.** On 62 % of guided steps at
  τ_mult 1.5 the deviation term reverses (§3). Order rows by **measured**
  `w_eff`, never by τ_mult and never by the withdrawn closed form.

---

## 5. Running it

```
V3_N=<n> bash proj1/cluster/submit_v3.sh ablation
```

**`V3_N` is required and must match the headline's** — that is what makes the
two stages comparable, and nothing enforces it across separate submissions, so
check it. **This stage is ~3× the headline's cost** (17 arms at two strengths
against 7 at one); price it with `v3_power.py` before submitting. Four chained
jobs: **preflight → array → insurance → table**. The
preflight is joined with `afterok`, so a broken arm or a batch that will not
fit stops everything before the grid runs. The array range is derived from the
job file (backends × properties × seeds) rather than typed.

| | |
|---|---|
| stage | `transfer_sweep.py --stage v3abl` |
| grid | `V3_ABL_ETAS` × `V3_ABL_TAU_MULTS` in `transfer_sweep.py` |
| arms literal | `ABL_ARMS` in `proj1/cluster/v3_run.slurm` |
| cells | `results/v3/<backend>/v3abl/n<N>/seed<S>/tr__*.json` |
| strengths | `V3_ABL_WS` in `transfer_sweep.py` |
| tables | `v3_table.py --stage v3abl --n <n> --w 1` and `--w 4` → `V3_RESULTS_ABL_w1*.md`, `V3_RESULTS_ABL_w4*.md`. Pooling the two is refused |
| power | [V3_POWER.md](../results/V3_POWER.md) |
| gates | `proj1/tests/test_v3.py` |
