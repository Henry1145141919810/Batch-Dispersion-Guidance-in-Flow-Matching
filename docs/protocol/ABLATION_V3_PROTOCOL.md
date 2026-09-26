# v3 ablation: BDG's two knobs, at n = 1000 on one seed

**26 September 2026. Pre-registration. Nothing here has been run.** This document
governs `--stage v3abl` only. The headline run is
[FULL_RUN_V3_PROTOCOL.md](FULL_RUN_V3_PROTOCOL.md) and where the two disagree on
`n` or seeds, each governs its own tree.

**Revised 26 Sep**, when the headline moved to n = 2000: this stage moved to
**n = 1000**, because equal n would put both stages in one tree under one stage
label and the table refuses that (§1.1). Nothing else changed.

**The ablation is deliberately cheaper than the headline** (Henry, 26 Sep):
**4.2 GPU-h against the headline's 12.8**. It buys that by dropping from three
seeds to one and from n = 2000 to n = 1000. What that costs is stated in §4 and
it is not small — read §4 before quoting any number from this run.

⚠️ **The figure "51.3 GPU-h" that earlier drafts contrasted this run against was
never the headline.** It was *this stage's own first plan* — 14 grid arms in a
shared n = 5000 tree across three seeds. The headline was 35.6 GPU-h then and is
12.8 now. Anywhere that number appears as "the headline", it is wrong.

---

## 1. What it is

| setting | ablation | headline, for contrast |
|---|---|---|
| grid | η ∈ {0, 1, 2, 4, 8} × τ_mult ∈ {0.5, 0.75, 1.0, 1.5} | η = 4 × τ_mult ∈ {0.5, 1.0} |
| arms | **17** (all of the grid) | 7 (5 comparison + 2 BDG) |
| n per cell | **1000**, in batches of **500** → **2 controllers** | 2000 in batches of 500 → 4 controllers |
| seeds | **one — 20261001** | three |
| properties | mu, alpha, gap | the same |
| base models | **both** — ours and EquiFM | the same |
| target / strength / window | q50, w = 1, t ≥ 0.5 | the same |
| chemistry floor | none | none |
| **cells** | **102** = 17 × 3 × 1 × 2 | 126 |
| **cost** | **≈ 4.2 GPU-h** (fm 1.5 + EquiFM 2.7) | ≈ 12.8 GPU-h |
| tree | `results/v3/<backend>/n1000/seed20261001/` | `…/n2000/seed<S>/` |

**This stage now carries the τ_mult 0.75 rung alone.** The headline dropped it,
keeping only the endpoints 0.5 and 1.0. So any claim that the τ ladder is
*monotone*, or that it curves, must be read off this grid's four points — the
headline's two cannot show either. That raises this stage's importance and does
not change its design.

### 1.1 Why n = 1000 and not 2000

Both stages write one stage label (`"v3"`) into
`results/v3/<backend>/n<N>/seed<S>/`, so **`n` is the only thing separating their
trees**. While the headline was n = 5000 that separation was free. When the
headline moved to 2000 it stopped being free: at equal n the two stages produce
byte-identical paths, and [`v3_table.py`](../../proj1/scripts/v3_table.py) — which
requires every (property, arm) pair at **all three seeds** — would glob this
stage's 15 single-seed arms into the headline's table and refuse with 45 problems
per backend.

Three fixes were available: move this stage's n, split the stage label, or
special-case the table's glob. Moving n was chosen because it also **preserves
the §4 doctrine** that an ablation row is not comparable to a headline row — that
doctrine rests on n, seed count and controller count differing, and equal n would
have removed one of the three and left the gate asserting it false.

`n` must stay divisible by the batch, 500. 1000 gives **2 whole controllers**
per cell, each still estimating `V_b` from 500 molecules (a 6.3 % standard error
on the variance). Controller **count** shrank; controller **quality** did not.

**η = 0 appears once, at τ_mult = 1.0.** At η = 0 the dispersion term is
multiplied by zero, so τ_mult cannot change those samples; four τ_mults would be
one computation under four names and pooling them would look like four
independent measurements of one thing. It is the identity rung: `w_eff` ≡ 1
whatever the controller measures, which is why it reproduces `plug` **in the
guidance field** regardless of `V_b`.

**All 17 arms run, including the headline's two** (η = 4 at τ_mult 0.5 and 1.0).
`n = 1000` writes to a different directory from the headline's `n2000/`, so there
is nothing to reuse. This is a feature: the grid is **self-contained**, and every
rung is comparable to every other rung at the same n, batch, seed and controller
count.

---

## 2. The batch does not shrink with n

For every other arm the batch is a performance knob. For BDG it is **the
estimator**: `V_b` is the variance of the guide's predictions across the molecules
in the batch, so the batch size *is* the controller's sample size.

`2000 / 500 = 4`, exactly. So a cell runs **four controllers of 500** instead of
the headline's ten, and **each one still estimates `V_b` from 500 molecules — a
6.3 % standard error on the variance.** Controller *count* was economised;
controller *quality* was not. That is the only defensible way to make a BDG run
cheaper, and it is why n was cut to 2000 rather than to a number that does not
divide 500.

Batch 500 is measured, not assumed: on `b200-mig45` it reserves 17.57 GiB on our
base and 26.65 GiB on EquiFM against a 33.5 GiB budget
([V3_BATCH_MEMORY.md](../results/V3_BATCH_MEMORY.md), confirmed on Betty 26 Sep).
`test_v3.py` refuses a batch that does not divide the ablation's n.

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

**Can.** Whether the deviation weight moves coverage at q50 at all; whether that
effect is monotone in **measured** `w_eff`; whether η does anything beyond
rescaling an already-saturated term; whether the η = 0 identity holds; and whether
any of it looks different on a borrowed base model.

**Cannot, and must not be claimed:**

- **That any rung beat `plug`, `tmpd`, `lgd_mc` or `tfg`.** There is **no
  comparison arm in this stage** (Henry, 26 Sep — explicitly declined). The
  headline's `plug` is at n = 2000, three seeds, four controllers, in a different
  tree. **This grid is read only against itself.**
- **Anything with a seed-to-seed error bar.** One seed. A difference between two
  rungs carries the binomial se at n = 1000 and *no* across-seed component, so it
  understates the true uncertainty. At n = 1000 that se is ~0.9pp at p ≈ 0.09,
  so only differences of roughly 2.5pp or more are resolvable at all, before any
  correction for the 17-way selection. The headline's three seeds are the only place
  this project can estimate that spread.
- **A number transferable to the headline.** Different n (1000 vs 2000),
  different seed count (1 vs 3), two controllers instead of four. An ablation row and a headline row are not two
  measurements of the same quantity. The gate that used to enforce equal n was
  inverted on 26 Sep to assert the *difference*, so this cannot drift silently.
- **That a rung's result came from the controller rather than the clip** (§3.4).
  This is the single most likely explanation for a null and must be offered as
  such, not as "spread control does not help".
- **Anything about the setpoint regime.** No rung is within 28× of it (§3.1). If
  the grid comes back flat, the honest reading includes "we never tested τ_mult
  near the value where the controller could equilibrate" — τ_mult ≈ 2–8 is the
  obvious follow-up and is *not* in this run.

---

## 5. Running it

```
bash proj1/cluster/submit_v3.sh ablation
```

Four chained jobs: **preflight → array `0-5` → insurance → table.** The preflight
is joined with `afterok`, so a broken arm or a batch that will not fit stops
everything before the grid runs.

| | |
|---|---|
| array | **6 tasks** = 2 backends × 3 properties (one seed, all 17 arms per task) |
| longest task | 59 min ours, ~107 min EquiFM at 1.83×, 147 min at the 2.5× margin, against `BUDGET_MIN` 225 |
| wall | 1.8 h at 6-way · 2.8 h at 3-way · 8.3 h serial |
| gates | `proj1/tests/test_v3.py` (65, closed-form) |
| cells | `results/v3/<backend>/n2000/seed20261001/tr__*.json` — 51 per backend |
| table | `proj1/scripts/v3_table.py --n 2000` |

Overrides: `V3_ABL_N` changes n (it must stay divisible by the batch, or the job
refuses); `V3_BATCH` changes the controller size, but **only before the first cell
exists** — the batch is in every filename and in the table's consistency keys, so
changing it mid-run splits the tree and the table refuses.

Costs are estimates on the same 2.468 min/unit rate as the headline, with EquiFM
at a **borrowed** 1.83× (the repo's one timed external backend). The headline's
`[timing]` lines measure that factor directly; read them before trusting the
figures above.
