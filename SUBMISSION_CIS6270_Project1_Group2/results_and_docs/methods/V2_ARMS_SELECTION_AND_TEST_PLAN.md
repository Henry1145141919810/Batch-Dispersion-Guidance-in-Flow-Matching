# v2 guidance arms — selection, implementation, and the run plan

Written for: the project team (Bobo, Idea, Haimo) and anyone reading the method
section later. It records *why* these three were chosen over the other four
proposed, what is actually implemented, what the tests do and do not prove, and
the exact order the cluster jobs must run in.

---

## 0. URGENT, affects the v1 sweep too

**There is not a single `unguided` cell in `results/sweep/`.** 141 cells on
disk; the arms present are `lgd_mc`, `osc`, `plug`, `smg`, `smg2`, `smg2_curv`,
`tfg_mc` only. The three q50 baselines sit at **position 0 of the todo list** —
they were never written.

Without a baseline, `select_arms.py` was a silent no-op: the divergence filter
became `inf`, the "no better than unguided" test could never be satisfied, so
**no arm could ever be dropped**, and it printed `dropped (0): none` while
looking like a working screen. It now refuses to emit a verdict instead.

The same hole makes any v1 conclusion "guidance beats/loses to doing nothing"
unsupported. The cells cost seconds:

```bash
python proj1/scripts/guidance_sweep.py --arms unguided --props mu,alpha,gap
```

---

## 1. What was selected, and why

Seven candidates were on the table across the two proposal documents. Three
were implemented. The selection criterion was **whether the idea attacks a
failure this project has actually measured**, not whether it is elegant.

### SPBC — Shape-Preserving Bias Correction

**The measured problem.** The population bias is large in the units that decide
acceptance. On `alpha`, plug-in guidance sits **13.99 δ** off centre at the wide
window; `alpha`/SMG is −5.69 δ; `gap`/plug is −2.12 δ. A mean error of several
band half-widths cannot be fixed by concentrating harder — no amount of
variance reduction moves a distribution that is centred in the wrong place.

**The mechanism.** For any scalar outcome law, subtracting the bias
`b = E[Y] − y*` leaves the mean at `y*` and leaves **every centred quantile and
every pairwise difference unchanged**. So the ideal edit gives every trajectory
the *same* property increment `ν`. A shared guidance multiplier does not do
this: per-sample property change is (residual) × (that sample's response
gradient), so a common coefficient produces *different* increments and distorts
the very shape it is supposed to preserve. SPBC instead solves, per sample,

```
min ‖d_i‖² / 2   subject to   a_i · d_i = ν      =>     d_i = (ν / r_i) a_i
```

with `a_i = Jᵀ∇f` (one VJP, already shared with every arm) and `r_i = ‖a_i‖²`.

**Why it is not prior art.** Classifier guidance, DPS, ΠGDM/TMPD and the MC
families all apply a *shared coefficient to a per-sample gradient*. SPBC
inverts that: a *per-sample coefficient* chosen so the realised property
increment is shared. The distinction is the whole point and it is testable —
the gate `spbc_increment_is_common` measures it at 1.8e-15.

**Honest limits, stated up front.** The forecast is the cheap `f_A(m_t)`, not a
rollout, so it inherits the off-distribution error in
`FINDING_QUADRATIC_CLOSURE_VALIDITY.md`. Preserving scalar property spacings
does **not** imply preserving structural diversity. And centring is not
guaranteed to raise band coverage — a skewed law can lose coverage when centred.
That is why coverage is measured rather than assumed.

### BTVG — Band-Targeted Variance Guidance

**The measured problem.** Spread runs at 4–18 δ, and `plug` reaches **105 % of
the data spread** on `alpha` — guidance is making the property distribution
*wider* than the unguided model.

**Why that happens, and it is not a bug in `plug`.** Expanding the usual
Gaussian likelihood score gives two terms:

```
(y − μ_F)/S · ∇μ_F   +   ½[ (y − μ_F)²/S² − 1/S ] · ∇V_F
```

The second coefficient is **positive** whenever `(y − μ_F)² > S`. A wider
observable puts more mass on a distant target, so maximising the likelihood of
`y*` genuinely prefers to widen when the target is far. That is correct
inference and wrong for design.

**The mechanism.** Replace the objective. Descend
`KL( N(μ_F, V_F) ‖ N(y*, τ²) )` instead. Its variance coefficient is
`−½(1/τ² − 1/V_F)`, whose sign depends on **V_F versus the tolerance**, not on
the residual. So it concentrates even when far from target.

It is also self-limiting — but that had to be *enforced*, not assumed. Left
alone the sign flip does not "stop", it **reverses**: `V_F` is a per-sample
posterior variance that goes to 0 as t → 1 by construction (`k = (1−t)²/t`
falls ~780× over t ∈ [0.5, 0.975]), so `V_F < τ²` happens on every trajectory
regardless of the batch, and the measured coefficient went to **+5.3, +110.8,
+78.6** — BTVG spent the last ~15 % of every run actively widening. The
coefficient is now clamped at ≤ 0: it may concentrate, and it may stop, but it
never widens. Gated by `btvg_var_independent_of_residual`,
`btvg_var_stops_below_tau` and `btvg_var_off_when_V_nonpositive`.

**τ is not δ.** A centred Gaussian with sd = δ gives only **68.3 %** band
coverage; 95 % needs sd ≈ δ/1.96. The default is δ/1.96 and the sweep brackets
it at {0.5, 1, 2}×.

### SHG — Scheduled Handoff Guidance

**The measured problem.** The guidance-window effect is the **largest single
signal in the sweep**: on `alpha` at fixed `w = 1`, MAE goes 10.02 → 5.32 →
6.59 as `t_min_guide` goes 0.05 → 0.5 → 0.75. That is a 47 % improvement from a
scheduling choice alone, and it is U-shaped, not monotone — so there is an
interior optimum that a single fixed arm cannot exploit.

**The mechanism.** Let different arms own different time bands. The band
boundaries are set by measurement, not taste: nothing starts below t = 0.5
(the window result), and the curvature-aware arms are only ever scheduled above
t ≈ 0.75 (where `FINDING_QUADRATIC_CLOSURE_VALIDITY.md` shows the quadratic
closure is actually valid). Four schedules are queued, including a three-phase
approach → curvature → centre.

### Dropped, and why

| Candidate | Why not |
|---|---|
| RWG | Subsumed by SHG; the novelty that remained was narrow. |
| RBAG | HIG prior art is too close to claim as a contribution. |
| OSHG adaptive order gate | Expensive per step; the fixed handoff (SHG) captures the measured effect at a fraction of the cost. |
| `escalate` | Designed, not implemented — it depends on the SHG result, so it is downstream of this round. |

---

## 2. Also added: the v1 arms that were never queued

These were implemented and unit-tested earlier but never made it into a sweep,
so the ablation ladder has holes. They run in stage v2:

- **`smg_mean`** — completes the ladder: mean-only / var-only / both.
- **`rch` (Haimo v1)** — the residual calibration head. Now run in **two
  variants**, because they are different claims: `residual` (fit to
  `c = ½ tr(HΣ)`, the quantity SMG computes — held-out R² **0.017**) and
  `direct` (fit to the realised endpoint gap — held-out R² **0.755**). The
  first is the deflation test for SMG; the second is the one that could work.
  Running one in place of the other would silently swap the claim.
- **`band`** — the tolerance-band step.

`rch` **refuses to run** without a head fitted for that property rather than
falling back to an unfitted one. Only `rch_mu.pt` exists, so
`fit_rch.slurm` is a hard dependency of the v2 sweep.

---

## 3. Four defects found before anything was submitted

Three of these were found by the gates or by the small-scale data, not by
reading the code. Listing them because the pattern matters: **two of the four
were invisible to the tests and only showed up in measured output.**

### 3a. The variance gradient was missing half of itself (found by a gate)

`grad_property_variance` computed only one of the two terms of ∇ₓV_F. Writing
`g = ∇f(m(x))`, `Σ = kJ`:

```
∇ₓ V  =  2 Jᵀ H (Σ g)            <- g moves with x, since dg/dx = HJ
         + k ∇ₓ[ gᵀ J(x) g ]|_g   <- the mean map's own curvature
```

Differentiating the scalar with `g` detached gives **only the second term**.
That is not a small correction: whenever the posterior mean is close to linear
in `x_t` — the regime the entire Tweedie construction assumes — the second term
nearly vanishes and the first term *is* the gradient. In the test fixture the
implemented gradient was **identically zero** against an autograd reference,
which means BTVG's variance half, the decisive component, would have been a
silent no-op there and mis-scaled everywhere else.

Fixed. Cost is one JVP + one HVP + one VJP, comparable to `smg2_curv`. For an
affine property (the ridge descriptor) H is exactly 0 and the term is correctly
skipped.

### 3b. SHG ignored the strength knob entirely (found by the smoke data)

Every SHG cell reported **the same number to four decimals** at w = 0.01, 0.05,
0.25, 1 and 4 (alpha MAE 4.811 at all five). `active()` was returning the
schedule's own weight and discarding `self.w`, so the strength sweep was inert
for all four SHG arms — 60 cells that would have measured nothing.

This is the same signature as the earlier `band` dead-code bug. The schedule's
third field is a *multiplier* on `self.w`; it is now applied as one. After the
fix the same five cells read 6.94 / 6.64 / 6.58 / **4.81** / 5.05 — a real
interior optimum.

No gate covered this because `active()` lives in `sampling.py` and the mutation
test only mutated `guidance.py`. It now mutates both.

### 3c. `rch` (Haimo v1) crashed on every cell (found by running it)

Inside `run_cell`, the local holding the RCH parameterisation was named `tgt` —
**shadowing the parameter that holds the q50/q90 target name**. Every `rch`
cell raised `KeyError` on `TARGETS[prop][tgt]`. The sweep's error handling is
correct, so it would have written `.failed` for all of them and carried on;
a four-hour cluster window would have been spent producing failure files for
the one arm this round was specifically meant to stop overlooking.

Fixed, and this is why `--preflight` now exists (§3e).

### 3d. The units gate was wrong, and exposed a real precondition

The first version of the sampler-units gate compared against the raw
`guidance_field` output, but `_guide` returns `zero_com(G_c, mask)`. The gate
was wrong, not the code — the applied multiplier is exactly `w(1−t)/t` for a
score arm and exactly `w` for a displacement arm, both to ~2e-16.

The detour surfaced something worth stating. SPBC's edit is parallel to
`a_i = Jᵀg`, and the sampler projects the coordinate block to zero centre of
mass afterwards. Projection changes `a_i · d_i` in general, so **the common
increment survives only when `a_i` already has zero COM**. That needs *both* a
translation-invariant property (so `g` has zero COM) *and* a posterior whose
Jacobian preserves the zero-COM subspace. The real generator satisfies the
second — `m = x + (1−t)v` with a zero-COM EGNN velocity — but an
elementwise-diagonal posterior does not. Both directions are now gated, and the
precondition is written down rather than assumed.

### 3e. Consequence: `--preflight`

`guidance_sweep.py --preflight` runs one throwaway cell per arm at n=8,
4 steps, writes nothing, and exits nonzero if any arm raises. It covers all 12
v2 arms in **15 seconds** and is the first thing `guidance_sweep_v2.slurm`
does. 3c is exactly the failure it exists to catch.

---

## 4. What the tests prove — and what they don't

`proj1/tests/test_v2_arms.py` — **27 closed-form gates, all passing to ~1e-15.**
The fixture uses a diagonal posterior mean and a quadratic property, so every
quantity has an exact closed form and Hutchinson estimation is exact. The
whole suite (8 files) is green.

`results/bench/mutation_test.py` — **25 / 25 injected defects caught**, up from
8, and it now mutates `sampling.py` as well as `guidance.py`. Three of the
mutations are the real bugs from §3 rather than hypothetical ones.

Two gates were strengthened after they passed *vacuously*:

- `btvg_var_reduces_V` originally tested `≤ 0`, which a zero gradient satisfies —
  it passed while the code was broken. It now requires a strictly negative
  directional derivative with a real margin.
- `spbc_zero_response_zero_edit` could not fail, because at `r = 0` the response
  `a_i` is also exactly zero and the edit vanishes either way. The dangerous
  case is `r` **small and nonzero**, where `ν/r` explodes while `a_i` is merely
  tiny. A separate gate now covers it, and it is what closed the last surviving
  mutation.

**What these do NOT prove.** The fixture's posterior mean is *linear*, so the
mean-map-curvature term of ∇ₓV_F is identically zero in it and is therefore
**not** covered by any gate. That is an open hole, flagged for the review pass.
Nothing here tests behaviour on the real EGNN at sampling scale — that is what
the cluster screen is for.

---

## 4b. Adversarial review — 18 findings, all addressed

An independent adversarial pass ran against the real generator on CPU. It found
**18 defects**; the serious ones are below. Two of its findings were that *my
gates* were wrong, not the code.

| # | Finding | Status |
|---|---|---|
| D1 | No `unguided` baseline ⇒ `select_arms.py` silently cannot drop anything | fixed — refuses, see §0 |
| D2 | SPBC's shape claim is **false end-to-end**: 8.03 δ drift in centred values, 12.3 δ in pairwise differences. Not clipping (0/320 steps), not zero-COM (retention 1.0000) — accumulated linearisation error over 20 steps | claim corrected |
| D3 | BTVG **clip-saturated at every strength**: its mean term carries 1/τ² where `plug` carries 1/s², and s²/τ² = 322 (mu), 1115 (alpha), 150 (gap). The whole grid measured clipped directions, not the objective | strength normalised by (τ/s)²; both values recorded |
| D4 | `V_F = gᵀΣg` comes out **negative** on real states (up to 3/16 samples). `clamp(min=1e-12)` turned that into a **+5e11** coefficient — the largest widening step the clip allows, exactly where the model had broken down. Root cause: Σ is not symmetric, measured asymmetry 0.045 → 0.63 | fixed + gated |
| D5 | The mean-map curvature term of ∇V_F is **identically zero in the fixture**; deleting it *or* scaling it ×100 passed the whole suite. It is 29–56% of the Hessian term on a nonlinear mean map | nonlinear gate added; both mutations now caught |
| D6 | Stage v2 scheduled **zero** `plug`/`tmpd`/`smg` cells — the references were filtered out by an `arms` test. And naive fixing would have collided with 5 existing stage-main cells | fixed with a `v2ref` tag; 0 collisions verified |
| D10 | `btvg_var_self_limits_below_tau` was a **vacuous pass** (empty `under` set). Real behaviour isn't "stops" but "reverses": V_F → 0 as t → 1 *by construction*, so BTVG widened over the last ~15% of every trajectory (b = +5.3, +110.8, +78.6) | coefficient clamped ≤ 0; gate de-vacuumed |
| D7 | SPBC closes **~50%** of the bias, not 100% — the increment is applied as a rate. And `mu` is a **minibatch** mean, so 4 different increments are applied within n=512 | documented with measurements |
| D8/D9 | `select_arms` pooled across targets/windows/sample sizes; `wins_property` leaked across properties so verdicts depended on alphabetical order | both fixed |
| D11–D18 | VPSampler ignored the schedule and displacement units; `schedule_log` miscounted and never reset; no v2 diagnostic reached the JSON; `arm_kwargs` raised on variant `"-"`; SPBC paid a redundant forward; RCH hardcoded `N_FEAT`; the mutation test wrote into the working tree | all fixed |

Two it confirmed correct, worth keeping: the **KL derivation is exactly right**
term by term (independently rederived), and **SPBC is step-count invariant**
(49.0 / 43.6 / 53.9 / 55.2 % closure at 10/20/40/80 steps) — the units concern
was unfounded.

One finding I acted on differently: its D4 fix suggestion implied a `V_ok`
guard, but `clamp(max=0)` already zeroes every `V ≤ 0` case (verified
numerically), so the guard was dead code that survived its own mutation. I
removed it and retargeted the mutation at the diagnostic instead.

**D17, unresolved and worth knowing:** the shipped `rch` head produces a
correction ~100× its own target (`c = −6.93` vs `C = +22.25` at t=0.5), so
`num = y − f − C` is −23.2 where the true residual is −1.00. Held-out R² is
0.0165 and nothing guards on it. Haimo v1's residual parameterisation looks
inert-to-harmful — which is a *result*, and the reason both parameterisations
are now measured.

After all fixes: **29/29 mutations caught**, all 8 test files pass, and
preflight runs all **15** arms clean.

---

## 5. Early signal from the smoke run — NOT decision data

43 cells on `alpha`, **n = 32, 20 steps, CPU**. The standard error on band
coverage at n = 32 is ±0.09, so nothing here decides anything. It is reported
because the *directions* are informative and because two of the four bugs above
were found in it.

| arm | in_band | MAE | f_B mean | mol_stab |
|---|---|---|---|---|
| unguided | 0.094 | 6.97 | 71.27 | 0.250 |
| `btvg` τ=1 | **0.375** | **1.30** | **75.54** | 0.094 |
| `btvg` τ=2 | 0.188 | 2.00 | 75.62 | 0.312 |
| `btvg_var` τ=1 | 0.000 | 9.36 | 68.71 | 0.281 |
| `shg_plug_spbc` | 0.125 | 4.81 | 73.44 | 0.250 |
| `spbc` w=4 | 0.000 | 5.82 | 74.36 | 0.156 |
| `smg_mean` | 0.062 | 6.67 | 71.05 | 0.219 |

Target is 75.54, δ = 0.481.

Three things worth carrying into the full run:

- **BTVG lands the mean exactly** (75.54 against a 75.54 target, from an
  unguided 71.27) and cuts MAE 5×. It costs stability, and τ=2 recovers most of
  the stability at half the MAE gain — so τ is a real knob, not a formality.
- **`btvg_var` alone moves the mean the *wrong way*** (71.27 → 68.71). This is
  expected and worth stating in the writeup: minimising `V_F = gᵀΣg` can be
  achieved by drifting toward regions where `‖∇f‖` is small, which is a
  systematic bias, not centring. The variance half needs the mean half; that is
  what the ablation is for.
- **SPBC is under-powered at every strength tested**, moving the mean only
  71.27 → 74.36 at w=4 when it needed +4.27. Either the trust radius or the
  rate interpretation needs re-tuning at full scale. Flagged, not resolved.

---

## 6. The run order on PARCC

Strictly sequential; each step depends on the one before.

```bash
# 0. Ship the code. Untar from $PROJ, NOT from inside proj1/.
#    laptop:
tar --exclude='__pycache__' -czf code_v2.tgz \
    proj1/scripts proj1/src proj1/tests proj1/cluster
scp code_v2.tgz betty:$PROJECT_ROOT/
#    cluster, from $PROJ:
cd $PROJECT_ROOT
tar tzf code_v2.tgz | head          # expect paths starting proj1/
tar xzf code_v2.tgz
python proj1/tests/test_v2_arms.py  # expect ALL PASS

# 1a. The missing v1 baselines -- seconds, and everything downstream needs them.
python proj1/scripts/guidance_sweep.py --arms unguided --props mu,alpha,gap

# 1b. Haimo v1 needs heads for alpha and gap (only mu exists).
sbatch proj1/cluster/fit_rch.slurm

# 2. Stage-2 screening at n=512 -- the SAME sample size as the v1 sweep,
#    which is what makes the two directly comparable. One job per property,
#    each chained afterany under the 4-hour QOS.
sbatch --export=ALL,SWEEP_PROPS=mu    proj1/cluster/guidance_sweep_v2.slurm
sbatch --export=ALL,SWEEP_PROPS=alpha proj1/cluster/guidance_sweep_v2.slurm
sbatch --export=ALL,SWEEP_PROPS=gap   proj1/cluster/guidance_sweep_v2.slurm

# 3. Read the screen. Nothing is dropped by hand.
python proj1/scripts/select_arms.py --stage v2

# 4. Full scale, survivors only, one job per seed for real error bars.
for S in 20260921 20260922 20260923; do
  sbatch --export=ALL,SEED=$S proj1/cluster/final_benchmark.slurm
done
```

Step 4 recomputes the survivor list from the screening cells *every time it
starts*, so the final run can never silently disagree with the screen it is
supposed to follow. If the screen has not run, the job stops instead of
guessing.

**Resume safety.** Stage-v2 cell names carry a variant suffix, so they cannot
collide with the 141 v1 cells already on disk. The stage-`main` plan was
verified byte-identical to the pre-patch version — the v1 sweep resumes exactly
where it stopped.

---

## 7. The selection rule, stated explicitly

`select_arms.py`. The default is **KEEP**. An arm is dropped only if, on *every*
property, both:

1. it is separated from the best arm by more than 3 combined standard errors on
   band coverage, **and**
2. it beats the unguided baseline on **neither** band coverage **nor** MAE by
   3 combined standard errors

…or it collapses chemistry on every property **and wins nothing**.

Uncertainty is computed, not assumed. Band coverage is a binomial
(`se = √(p(1−p)/n)`). For MAE, `|e|` has mean MAE and second moment RMSE² *by
definition*, so `var(|e|) = RMSE² − MAE²` exactly — no distributional
assumption.

Two corrections were made to this rule after running it on smoke cells:

- **Chemistry alone must not drop an arm.** The first version dropped `btvg`
  for stability while it held the best property MAE on the board by a factor of
  six. An arm that dominates the target metric and costs stability is a
  trade-off to report and re-tune at lower strength, not an "obvious
  non-competitor". It now proceeds, flagged.
- **The "no better than unguided" test used coverage only.** Coverage is a
  binomial with little power at n = 512; MAE uses every sample's magnitude.
  Requiring failure on *both* makes a drop strictly harder to justify.

**Divergence is not allowed to win.** Cells with nonfinite samples, or MAE above
10× unguided, are excluded from an arm's best cell and the arm is reported
`FRAGILE` with the count. This is the exact failure mode that let an
`alpha`/SMG cell with **MAE 43,599,656 and 47/512 nonfinite samples** hide
behind a "best strength" table earlier in this project.

---

## 8. What to check when the jobs land

| Check | Where | What would be bad |
|---|---|---|
| Did any cell fail? | `ls results/sweep/*.failed` | Non-empty means an arm crashed; the name says which |
| Is `rch` measurable on all three? | `fitrch-*.out` | A missing head means Haimo v1 is still only measured on `mu` |
| Does SPBC actually reduce bias? | `select_arms.py` bias column | Bias unchanged means the displacement units are wrong |
| Does BTVG reduce spread below `plug`? | spread / std column | If not, the KL objective is not buying anything |
| Does SHG beat its own components? | compare `shg_*` to `plug`, `smg`, `spbc` | If a schedule loses to both its parts, the handoff boundary is wrong, not the idea |
| `schedule_used` non-empty for SHG arms | the cell JSON | Empty means `active()` never fired and the arm silently ran unguided |
