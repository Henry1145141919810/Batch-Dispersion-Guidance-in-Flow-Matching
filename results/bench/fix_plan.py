"""Apply the audit's Task-B corrections to GUIDANCE_EXPERIMENT_PLAN.md. Run once."""
import io

p = "GUIDANCE_EXPERIMENT_PLAN.md"
s = io.open(p, encoding="utf-8").read()

# ---- cost table: every figure now measured
OLD = s[s.index("| # | arm | class |"):s.index("### Why both `smg` and `smg2` are kept")]
NEW = """| # | arm | class | what it computes | **measured** ms/mol @ NFE 100 |
|---|---|---|---|---|
| 1 | `unguided` | CONTROL | no guidance; the floor for property targeting | **34.5** |
| 2 | `plug` | COMPARE | `(y - f(m)) / s^2 * J^T g`. DPS-style. Needs strength tuning or it is catastrophic | **95.0** |
| 3 | `tmpd` | COMPARE | plug-in with the uncertainty denominator `s^2 + g'Sg` -- TMPD / PiGDM's published scalar form, and our `smg_var` ablation | **241.8** |
| 4 | `tfg_mc` | COMPARE | TFG's Monte Carlo smoothing: K isotropic perturbations at a **tuned** sigma, softmax-weighted (Ye et al. 2024) | **125.5** (K=4) |
| 5 | `lgd_mc` | COMPARE | likelihood marginalisation, isotropic draws at a scale **read off the model** (`r^2 = tr(Sigma)/d`) rather than tuned | **271.2** (K=4) |
| 6 | `osc` | COMPARE | observable-space closure: same guide budget as `lgd_mc`, scalar observable's law integrated analytically | **273.3** (K=4) |
| 7 | `smg` | OWN | **SMG as shipped.** Mean correction `c = 1/2 tr(H Sigma)`, denominator `s^2 + g'Sg` | **421.5** (1 probe), **959.0** (4) |
| 8 | `smg2` | OWN | **Completed-moment closure.** Denominator gains `1/2 tr((H Sigma)^2)` | **802.7** (1 probe), **2538.6** (4) |
| 9 | `smg2_curv` | OWN | `smg2` plus the Stein direction `((r^2-S)/S^2) Sigma H Sigma g` | **836.4** (1 probe) |
| 10 | `escalate` | OWN | closure-residual escalation. **Not yet implemented** -- design only | >= 1074 (smg2 + lgd_mc, plus escalated cells) |
| 11 | `kappa3` | OWN (probe) | logs `gamma = k3[g,g,g] / (g'Sg)^{3/2}`. A **column on another arm**, not a sampler | +370 on `smg2` (**1173.1** total) |

All figures measured on the RTX 5080 by `results/bench/time_guided.py`, which also prints the
generator and guide call counts behind each. **An earlier version of this table was estimated
from call counting and was wrong by up to 2x** (`smg2` was given as ~450, `lgd_mc` as ~130);
those estimates are replaced by the measurements above.

**`smg2` costs ~1.9x `smg` at the same probe count**, because the second trace needs two JVPs and
two HVPs per probe against one each for the first. At 4 probes it is 2.5 s/molecule, which is why
the probe count is settled by gate G3 before Stage 2 rather than assumed.

"""
s = s.replace(OLD, NEW)

# ---- lgd_mc / osc description fix (the Sigma^2 bug and what replaced it)
s = s.replace("""| 5 | `lgd_mc` | COMPARE | Monte Carlo likelihood marginalisation, drawing from the model's own `Sigma` rather than isotropic (LGD) | est. ~130 (K=4) |""", "")

# ---- corrected cost/timetable section
OLD_T = s[s.index("| stage | what | cells | GPU-hours (serial) |"):s.index("Predictor training is already done")]
NEW_T = """| stage | what | cells (all arms) | **GPU-hours, serial** | Betty, 4 slots |
|---|---|---|---|---|
| 0 | gates G1-G5 | -- | ~2 | ~2 h |
| 1 | screening + tuning, n = 512, 3 props x 3 targets x 5 strengths | 45 | **19.8** | ~5 h |
| 2 | reportable grid, n = 4,096, 3 props x 5 targets, 1 seed | 15 | **52.9** | ~13 h |
| 2b | mu at the median target, 2 extra seeds, n = 4,096 | 2 | **7.1** | ~2 h |
| -- | **total** | | **~82** | **~22 h** |

Arithmetic: summing the measured ms/molecule over the nine sampling arms gives 3,101 ms, so one
cell across all arms costs `3.101 s x n`. At n = 4,096 that is 3.53 GPU-hours per cell; at
n = 512, 0.44.

**The previous version of this table said ~58 GPU-hours and was wrong**: it costed Stages 1 and 2b
as though every arm were as cheap as `plug`. The corrected figure is ~82, and it assumes
`--n-probe 1`. **At 4 probes Stage 2 alone rises to ~150 GPU-hours**, which is the single largest
budget risk in the plan and is why G3 runs first.

**4-hour QOS cap.** At n = 4,096 and 1 probe the most expensive arm (`smg2_curv`) is 0.95 h per
cell, so 3-4 cells per array task is safe. At 4 probes `smg2` alone is 2.9 h per cell and only one
cell fits per task -- size the array accordingly if G3 forces a higher probe count.

"""
s = s.replace(OLD_T, NEW_T)

# ---- matched-compute control (the audit's fairness objection)
s = s.replace("""### Two protocol rules from the memo, adopted""",
"""### A matched-compute control, because the arms differ by 24x

`unguided` is 34.5 ms/molecule and `smg2_curv` is 836.4 -- a 24x spread. Comparing them at
equal *sample count* silently hands the expensive arms 24x the compute, and every expensive arm
here is an OWN arm. That would make a win uninterpretable.

So every table reports **two columns**: at matched samples, and at **matched compute**. The
matched-compute column gives each cheap arm the extra budget it is owed, spent the way its own
authors would spend it:

| arm | matched-compute variant at the `smg2_curv` budget (836 ms/mol) |
|---|---|
| `plug` | NFE 100 -> 880, or 8 restarts with best-of selection on `f_A` |
| `tfg_mc` | K = 4 -> K = 30 |
| `lgd_mc` / `osc` | K = 4 -> K = 12 |
| `tmpd` | NFE 100 -> 345 |

**If an OWN arm's advantage disappears in the matched-compute column, that is the result**, and
it goes in the paper. The memo's section 2 already names the stop condition: "stop the whole
programme if a strength-tuned plug-in arm matches it at equal cost on terminal metrics."

### Two protocol rules from the memo, adopted""")

# ---- stochastic sampler gap
s = s.replace("""- **Report both samplers.** With a stochastic sampler the reverse SDE absorbs local field error
  and methods an order of magnitude apart in field accuracy land within noise of each other; with
  the deterministic sampler the ordering survives. Reporting only one makes the result an
  artefact of the solver choice.""",
"""- **Report both samplers.** With a stochastic sampler the reverse SDE absorbs local field error
  and methods an order of magnitude apart in field accuracy land within noise of each other; with
  the deterministic sampler the ordering survives. Reporting only one makes the result an
  artefact of the solver choice. **We do not currently have a stochastic sampler**
  (`proj1/src/sampling.py` is deterministic Euler/Heun). Either add an `eta > 0` variant before
  Stage 2, or state in the write-up that all results are deterministic-sampler results and that
  the ordering is therefore the favourable case for field-accuracy claims. **This is an open
  item, not a solved one.**""")

# ---- Stage 1 rule: SE band on criterion 3
s = s.replace("""3. **property MAE not better than `unguided`** -- it provides no targeting at all.""",
"""3. **property MAE not better than `unguided` by at least 2 standard errors** -- it provides no
   targeting at all. The SE band matters: at n = 512 the MAE standard error is ~0.031 D, so a
   bare point-estimate comparison would eliminate an arm that is genuinely better by 0.03 D
   roughly half the time.

Criteria 1 and 2 are evaluated at **every** strength (an arm that diverges at one strength but is
healthy at another is kept, at the healthy strength). Only criterion 3 is evaluated at the arm's
best strength.""")

# ---- sigma = 0.7 provenance
s = s.replace("""- Property-MAE standard error is `sigma/sqrt(n)`. With `sigma ~ 0.7 D` on mu: **0.011 D** at""",
"""- Property-MAE standard error is `sigma/sqrt(n)`, where sigma is the spread of the per-molecule
  absolute error. **`sigma ~ 0.7 D` is an assumption, not a measurement** -- it is taken from the
  order of published MAEs on mu and must be replaced by the measured spread from Stage 1 before
  the Stage 2 sample size is final. With `sigma = 0.7 D`: **0.011 D** at""")

# ---- multiplicity
s = s.replace("""4. **`kappa3` is a column, not a row.**""",
"""4. **Multiplicity.** The grid is 9 arms x 3 properties x 5 targets = 135 comparisons. At the
   5% level, roughly 7 will look significant by chance. The headline claim must therefore rest
   on a **pre-specified primary cell** -- mu at the median target q50, matched compute,
   deterministic sampler -- with everything else reported as secondary and uncorrected, or on a
   correction stated in advance. Choosing the best cell after the fact is the same error the
   checkpoint selection made in `BASE_MODEL_BENCHMARK.md` section 3.4.
5. **`kappa3` is a column, not a row.**""")

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("plan corrected: measured costs, 82 GPU-h, matched-compute control, SE band, multiplicity, sampler gap")
