# Three new guidance ideas: reduce bias, improve band coverage, preserve useful spread

**Research memo — 21 September 2026**  
**Status:** three proposed mechanisms, a fresh audit of existing results, and reproducible mathematical checks. **No new molecular sampling experiments were run for this memo.**

**Revision after the follow-up:** the objective is now **bias correction and higher tolerance-band coverage with explicit spread preservation**. The earlier emphasis on contraction was incomplete. The filename is retained so previous links continue to work. Sections 3.3–3.7, 5.8–5.10, 6.7, and 7.8 develop the revised reasoning and mechanisms; the recommendation and experiment plan have also been updated.

## 1. What I recommend

The right objective is **put a diverse population in the acceptable region**, with small bias, rather than simply minimize its variance. Bias must be measured relative to the tolerance band: alpha `plug` is **8.56 band half-widths** off center, alpha `smg` is **5.69**, and gap `plug` is **2.12**. These are substantial biases even when variance accounts for most of MSE. The previously highlighted dipole `plug` cell is a comparatively well-centered special case, not grounds for treating bias as generally solved.

My revised recommendation is to **test a shape-preserving centering edit first**, accept it only when coverage and quality do not deteriorate, and then reshape the distribution only as much as needed for band coverage. Preserve structural diversity and a nonzero, robust spread within the band. Retain the original overall property variance only when it is compatible with the band and a controlled tail. This distinction is mathematical, not merely a preference: a broad property law cannot fit entirely inside a much narrower band while keeping its original variance.

The three proposals below are new relative to the earlier QPMG, DBFG, MAG, completed-moment, closure-escalation, and diversity-projection proposals in this project:

| Proposal | Concrete change | Why investigate it | Main way it could fail |
|---|---|---|---|
| **1. Reachability-Aware Target-Variance Guidance (RATV)** | Center through approximately common property increments, then improve coverage subject to moment and robust-spread constraints | Several arms have material bias in band units; stronger attraction can erase useful spread | Centering can itself reduce coverage; preserving only variance can hide shape changes or outliers |
| **2. Order-Sensitive Handoff Guidance (OSHG)** | Test whether centering should hand off to coverage repair using measured order, terminal benefit, and spread retention | The useful phase boundary depends on remaining bias and band occupancy, not only clock time | The gate may add nothing beyond simple lookahead or a fixed switch |
| **3. Reachable Band-Assignment Guidance (RBAG)** | Construct the least distorted feasible target slots with a nonzero central-spread floor, then assign them by trajectory response cost | The band admits many target values; they can preserve useful differences between molecules | Exact old spread may be impossible; sorting or ordinary target jitter may suffice |

**Build order:** repair the evidence; establish a translation/centering diagnostic and a diversity-preserving baseline; pilot revised RATV; test simple centering-to-coverage handoffs before adaptive OSHG; use revised RBAG when band-compatible target allocation is needed. Validate response heterogeneity before attributing an advantage to response-aware costs. Test the mechanisms separately before combining them.

**Novelty standard:** these are specific research hypotheses, not priority-cleared inventions. The underlying control, switching, and assignment mathematics is established. Sections 5–8 identify the narrower contributions that might survive experiments and prior-work review. Existing work already covers moment guidance, variance interactions, prescribed marginal distributions, optimal-control guidance, and histogram transport. Giving those primitives new names would not be a contribution.

## 2. Evidence inspected and the outcome table

The unnamed document in the request was interpreted as [FINDINGS_SO_FAR.md](../status/FINDINGS_SO_FAR.md), which contains the latest outcome tables and the target-variance suggestion. I also inspected:

- [GUIDANCE_EXPERIMENT_PLAN.md](../protocol/GUIDANCE_EXPERIMENT_PLAN.md), [FINDING_QUADRATIC_CLOSURE_VALIDITY.md](../results/FINDING_QUADRATIC_CLOSURE_VALIDITY.md), and [SPLIT_PROTOCOL.md](../protocol/SPLIT_PROTOCOL.md).
- [The previous three proposals](../../archive/guidance_learning_output/Three_New_Guidance_Proposals.md) and [the TOP6 review](../../audit/top_guidance_2026/TOP6_GUIDANCE_RECOMMENDATIONS.md).
- The actual screening, sampler, guidance, property-wrapper, and evaluation implementations under `proj1/`, and the closure diagnostic under `results/bench/`.
- All **141** JSON result cells currently in `results/sweep/`, excluding archived audit runs in other directories.

### 2.1 Census and reproducibility

| Item | Audited value | Interpretation |
|---|---:|---|
| Saved screening cells | 141 | 49 mu, 49 alpha, 43 gap |
| Generated attempts across cells | 72,192 | 141 × 512; not 72,192 independent replications |
| Targets | q50 only | No q90 result in this directory |
| Sampling seeds | One: 20260921 | Initial samples are reused across arms; uncertainty must respect pairing |
| Arms represented | Seven | No `unguided` or `tmpd` cell here |
| Current sweep plan | 294 cells | Reconstructed from the current script's constants |
| Saved cells belonging to that current plan | **99** | The other 42 use historical strengths 0.5 or 2 |
| Current-plan cells absent locally | **195** | Therefore “141 of 294 completed” is not an accurate current-plan completion count |
| Nonfinite trajectories across saved cells | **128** | Their treatment matters for interpreting MAE |

The raw cell files name `fm_last.pt`; `fm_v1` is the project documentation's checkpoint label. A basename is not a checkpoint identity. Future result records should include checkpoint and code hashes. Existing records also omit the actual likelihood width `s`; the inspected runner sets `s = f_A.y_std`.

The accompanying [audit script](../../audit/guidance_variance_switching_2026/audit_results_and_math.py) records SHA-256 hashes for all 141 input files, rebuilds the table, and runs the small mathematical checks. Outputs are [screening_audit.json](../../audit/guidance_variance_switching_2026/screening_audit.json), [reconstructed_screening_table.csv](../../audit/guidance_variance_switching_2026/reconstructed_screening_table.csv), and [mathematical_checks.json](../../audit/guidance_variance_switching_2026/mathematical_checks.json).

### 2.2 Reconstructed full-window outcomes

Each arm is shown at its **lowest observed evaluator MAE** among the available strengths at q50, t_min=0.05. This is retrospective screening selection, not an independently tested ranking. Standard deviation is reconstructed as

\[
\widehat V=\widehat{\mathrm{RMSE}}^2-(\overline f_B-y)^2,
\qquad \mathrm{spread}=\sqrt{\widehat V}.
\]

The identity uses population-denominator empirical variance, not the n−1 estimator. For a cell with nonfinite samples, the mean and RMSE are conditional on finite samples because that is how the evaluator computes them.

| Property | Arm | w | MAE | Signed bias | Spread | In band | Molecule stability | Validity |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mu | plug | 4 | 0.8006 | −0.0444 | 0.9936 | 0.1094 | 0.2773 | 0.6152 |
| mu | lgd_mc | 4 | 0.8490 | −0.1910 | 1.0278 | 0.0996 | 0.3867 | 0.7168 |
| mu | tfg_mc | 4 | 0.8717 | −0.1316 | 1.0753 | 0.1133 | 0.3516 | 0.6621 |
| mu | osc | 4 | 0.8817 | −0.2283 | 1.0387 | 0.1035 | 0.3848 | 0.7285 |
| mu | smg | 4 | 0.9661 | +0.0485 | 1.1911 | 0.0898 | 0.3242 | 0.7344 |
| mu | smg2 | 4 | 1.0015 | +0.1936 | 1.2671 | 0.1094 | 0.3652 | 0.7344 |
| mu | smg2_curv | 4 | 1.0343 | +0.2834 | 1.2550 | 0.0762 | 0.3867 | 0.7676 |
| alpha | osc | 4 | 4.3559 | +0.2998 | 5.6878 | 0.0801 | 0.3691 | 0.7070 |
| alpha | lgd_mc | 4 | 4.6811 | +0.9077 | 5.9683 | 0.0605 | 0.3359 | 0.6563 |
| alpha | smg2_curv | 4 | 5.6397 | −2.2512 | 6.7958 | 0.0566 | 0.4082 | 0.7734 |
| alpha | smg2 | 2 | 5.6711 | −1.7382 | 7.2603 | 0.0547 | 0.3945 | 0.7871 |
| alpha | tfg_mc | 4 | 6.7688 | +3.7761 | 8.2246 | 0.0430 | 0.2695 | 0.5215 |
| alpha | smg | 0.25 | 6.8849 | −2.7396 | 8.5276 | 0.0547 | 0.4102 | 0.8359 |
| alpha | plug | 4 | 7.1804 | +4.1221 | 8.5736 | 0.0469 | 0.2324 | 0.4805 |
| gap | tfg_mc | 4 | 0.03090 | −0.01263 | 0.03489 | 0.1309 | 0.3281 | 0.6641 |
| gap | plug | 4 | 0.03187 | −0.01612 | 0.03391 | 0.1074 | 0.2227 | 0.5254 |
| gap | lgd_mc | 4 | 0.03217 | −0.00101 | 0.03952 | 0.1328 | 0.3945 | 0.7188 |
| gap | osc | 4 | 0.03343 | −0.00263 | 0.04053 | 0.1465 | 0.3809 | 0.7148 |
| gap | smg | 4 | 0.03487 | +0.00162 | 0.04174 | 0.1328 | 0.3535 | 0.7227 |
| gap | smg2 | 4 | 0.03587 | −0.00600 | 0.04272 | 0.1328 | 0.3633 | 0.7441 |
| gap | smg2_curv | 4 | 0.03695 | −0.00644 | 0.04376 | 0.1250 | 0.3867 | 0.7578 |

The headline numbers in the findings memo reproduce to their displayed rounding. The problems are primarily interpretation and missing controls, rather than arithmetic transcription.

**What the table supports:** concentration is a serious candidate bottleneck; the trade-off with chemistry is material; the best-looking method depends on the property and metric. For example, dipole `lgd_mc` sacrifices about 0.0484 MAE relative to `plug` while improving molecule stability by about 10.94 percentage points. On gap, lowest MAE and highest in-band rate select different arms.

**What it does not support:** a universal winner, superiority to a matched unguided baseline, a working mixed schedule, or decoded valid-and-on-target yield. Those measurements are absent.

### 2.2a Bias and spread in the units that matter for acceptance

The same selected cells look different when divided by the actual scoring half-band delta:

| Property / arm | Signed bias / delta | Spread / delta | Observed in-band fraction | Diagnosis |
|---|---:|---:|---:|---|
| mu / plug | −0.26 | 5.91 | 0.1094 | Small center error, broad outcome law |
| mu / osc | −1.36 | 6.18 | 0.1035 | Both center error and breadth matter |
| alpha / plug | +8.56 | 17.81 | 0.0469 | Mean far outside the band and very broad outcomes |
| alpha / smg | −5.69 | 17.72 | 0.0547 | High validity does not imply acceptable targeting |
| alpha / osc | +0.62 | 11.82 | 0.0801 | Center closer to the band, still very broad |
| gap / plug | −2.12 | 4.46 | 0.1074 | Material center displacement |
| gap / tfg_mc | −1.66 | 4.59 | 0.1309 | Lowest selected MAE does not mean low bias in band units |

These are arithmetic transformations of the existing aggregate results, not new samples. **MSE decomposition answers where squared error comes from; bias/delta answers whether the center is acceptably located.** Both are necessary. The revised proposals use both, and distinguish population signed bias from MAE, which is average absolute per-sample error.

There is also actionable evidence that the large biases are not merely evaluator disagreement. For alpha `plug`, the guide's own mean bias is +8.14delta and the evaluator's is +8.56delta; for alpha `smg`, they are −5.67delta and −5.69delta. An online f_A-only controller can therefore see substantial miscentering in these cells. However,

\[
b_B=b_A+\mathbb E[f_B(X)-f_A(X)],
\]

so centering f_A does not generally center f_B, and neither predictor is physical ground truth. Report both signed biases and their difference. Do not use final-test f_B offsets to tune the online target. Any predictor calibration using labeled validation molecules is a separately disclosed comparator, and clean-data calibration need not transfer to generated molecules.

### 2.3 Corrections needed before building a new method on the findings

**A. The current property table uses continuous atom features.** `evaluate_samples` computes `f_A(coords, feats, mask)` and `f_B(coords, feats, mask)` directly. The wrapper does not one-hot decode `feats`. Chemistry is assessed through a separate decoded construction. Consequently, the property column is currently a continuous-feature evaluator metric, including invalid finite samples. It is not yet a property metric on valid decoded molecules.

Retain that historical metric for comparability, but add evaluator predictions after atom-type decoding, with an explicit valid-only summary and an all-attempt success denominator. These answer different questions; report both.

This representation diagnosis follows from the inspected scoring implementation. Because the historical cells lack code hashes, a different historical scoring revision cannot be ruled out from their metadata alone. Verify provenance or rescore saved samples before making a publication claim about exactly how those cells were evaluated.

**B. Validity × in-band is not valid-and-in-band yield.** The saved aggregates do not contain the intersection. Without sample-level records, the only general bounds are

\[
\max(0,p_{\mathrm{valid}}+p_{\mathrm{band}}-1)
\le p_{\mathrm{valid}\cap\mathrm{band}}
\le\min(p_{\mathrm{valid}},p_{\mathrm{band}}).
\]

For dipole `plug`, the bound is [0, 0.1094]. Multiplying 0.6152 by 0.1094 would assume an unmeasured independence relationship. Distinct valid-and-in-band graphs require still more information; five example SMILES per cell are insufficient.

**C. The window comparison is partly strength-confounded and not universally monotone.** At t_min=0.05 there is a strength sweep, while 0.5 and 0.75 have only w=1. A fairer existing comparison fixes `plug` at w=1:

| Property | t_min=0.05: MAE / stability | t_min=0.50: MAE / stability | t_min=0.75: MAE / stability |
|---|---|---|---|
| mu | 0.9753 / 0.3242 | 0.9822 / 0.3730 | 1.0941 / 0.4082 |
| alpha | 10.0239 / 0.2637 | **5.3164 / 0.3750** | 6.5924 / 0.4004 |
| gap | 0.03432 / 0.2852 | 0.03414 / 0.3594 | 0.03583 / 0.4023 |

Alpha targeting improves substantially when the early portion is removed. The phrase “guide less, get worse targeting, monotonically” is therefore too broad. No existing cell tests “A early, B late.”

**D. The closure reference is a proxy, and its covariance differs from the correction's covariance.** `quad_validity.py` compares an anisotropic Jacobian-based correction \(\tfrac12\operatorname{tr}(H\widehat\Sigma)\) with Monte Carlo draws using trace-matched isotropic covariance \(r^2I\). Neither is a verified true posterior. Their disagreement can reflect covariance mismatch as well as Taylor error and guide extrapolation.

A decisive counterexample: take \(f(x)=x_1^2\), m=0, and \(\Sigma=\operatorname{diag}(100,1)\). The second-order expansion is exact and gives 100. Trace-matched isotropic draws give expectation 50.5. Comparing those quantities would report a 49.5 “closure error” even though there is no Taylor remainder. This counterexample is checked in the audit.

Repair the diagnostic with two separate comparisons: isotropic MC against an isotropic trace correction; and, if feasible, stabilized anisotropic MC against the same stabilized anisotropic correction. Report guide-mean error separately from likelihood-gradient error. A smaller error in E[f] does not establish a better guidance field. The negative dipole predictions remain strong evidence of extrapolation; the precise t=0.75 crossover and universal closure-failure interpretation require this repair.

**E. “No reward hacking” is too strong.** The fairly flat guide/evaluator disagreement in selected cells is evidence against increasing disagreement under this particular diagnostic. Two EGNNs trained on disjoint data can still share architecture bias and off-distribution errors. It does not certify accurate physical properties, especially with soft features and invalid molecules. Use a separate architecture and, if available, a feasible physical-property check after selecting candidates.

**F. Divergences are present.** Alpha SMG has 2/512 nonfinite outputs at w=0.5, 27 at w=1, 47 at w=2, and 50 at w=4. Dipole SMG w=4 and gap SMG w=2 each have one. The evaluator excludes nonfinite samples from property means but counts them against in-band and validity. Any comparison based solely on MAE can therefore reward losing difficult samples. The best alpha SMG row uses w=0.25 and does not expose these failures.

**G. Small bias does not mean a mean-only objective has succeeded.** Squared error already satisfies

\[
\mathbb E[(Y-y)^2]=(\mathbb EY-y)^2+\operatorname{Var}(Y).
\]

Ordinary per-sample guidance can reduce variance; it is not intrinsically a mean-only mechanism. The proposed contribution must demonstrate a better allocation of contraction and chemical cost, not merely add a quantity already present in MSE. Also, the exact decomposition is for **MSE**, not MAE.

**H. “s is too wide” is a testable hypothesis, with an identifiability caveat.** In the current plug-in implementation,

\[
u_{\mathrm{plug}}\propto \frac{w}{s^2}(y-f(m))J^T\nabla f(m).
\]

Changing \((w,s)\) while holding w/s² fixed gives the same field, including the same deterministic radial clipping. For plug-in, a width sweep is redundant with a sufficiently broad strength sweep. For MC likelihood weights and denominators s²+v it generally is **not** redundant. A narrow s can also cause MC weight collapse. Record effective precision w/s², normalized weight ESS, and clipped-step fraction; do not label a reparameterized strength change as innovation. s=delta does not guarantee output spread delta.

**I. A learned conditional mean does not solve the full likelihood.** A time-conditioned guide may avoid the specific clean-guide mean bias, but \(\mathbb E[\exp(-(Y-y)^2/(2s^2))\mid x_t]\) is not determined by \(\mathbb E[Y\mid x_t]\). Conditional shape still matters. Similarly, held-out R² for fitting the mean gap is not evidence that the fitted head's derivatives or terminal guided outcomes are accurate.

## 3. Define “target variance” correctly

### 3.1 Three different quantities

Let Y be the terminal property, with the evaluator convention explicitly specified.

1. **Population outcome variance:** \(V_{\mathrm{out}}=\operatorname{Var}_{\text{initial seeds}}[Y]\). This is the spread relevant to your observed outcome table.
2. **Conditional uncertainty:** \(V_{\mathrm{cond}}(x_t)=\operatorname{Var}(Y\mid X_t=x_t)\), under a specified stochastic path or joint distribution.
3. **Local surrogate variance:** \(v_{\mathrm{lin}}=\nabla f(m)^T\widehat\Sigma_t\nabla f(m)\). This is a linearized quantity using an approximate covariance; it is not generally either of the first two.

For a coherent joint law, total variance gives

\[
\operatorname{Var}(Y)=\operatorname{Var}(\mathbb E[Y\mid X_t])+
\mathbb E[\operatorname{Var}(Y\mid X_t)].
\]

This identity cannot mix a forward corruption posterior with an unrelated deterministic sampler continuation. For a deterministic continuation given x_t, its own conditional terminal variance is zero. Different x_t still produce different outcomes. At the endpoint, conditional variance is zero even when the generated property population is extremely broad. Minimizing the SMG denominator's v_lin can therefore miss the problem completely.

With a quadratic observable under a Gaussian proposal, the exact surrogate variance is

\[
\operatorname{Var}(f(m)+g^T\xi+\tfrac12\xi^TH\xi)
=g^T\Sigma g+\tfrac12\operatorname{tr}((H\Sigma)^2).
\]

Even this completed expression is not the exact variance of an arbitrary nonlinear learned property. Nor is its gradient available “for free” merely because a scalar estimate has been logged: differentiating it involves additional derivatives, or additional perturbation evaluations.

### 3.2 A variance target is not a coverage guarantee

Let b=E[Y]−y, and let delta be the fixed scoring half-band. For |b|<delta, Chebyshev gives

\[
\Pr(|Y-y|>\delta)\le\frac{V_{\mathrm{out}}}{(\delta-|b|)^2}.
\]

A sufficient distribution-free condition for at least 1−epsilon coverage is

\[
V_{\mathrm{out}}\le\epsilon(\delta-|b|)^2.
\]

This is conservative. Under an actually Gaussian, centered outcome law, 95% coverage instead requires standard deviation approximately delta/1.96. Setting variance to delta² gives only **68.27%** Gaussian coverage. Setting it to zero is not a practical universal goal either: it may be infeasible, exploit the guide, or damage chemistry.

For the **optional concentration phase**, a nonzero target standard deviation tau such as delta_A/2 is a possible design choice, with delta_A a frozen guide-side tolerance. It is not the default instruction for the revised centering phase, which attempts to retain spread. Check feasibility and actual decoded coverage before selecting tau. Do not silently tune delta_A using test f_B outcomes. The official delta remains frozen from the evaluator protocol; its clean validation MAE is not a guaranteed error bound on generated molecules.

Concentrating the target property and retaining structural diversity are compatible in principle, but neither implies the other. Always measure both.

Even mean and variance together do not determine in-band rate. At target zero and half-band 0.5, a law with equal mass at −1 and +1 has mean zero, variance one, and zero coverage. A law with mass 0.9 at zero and 0.05 at each of ±sqrt(10) has the same mean and variance but 90% coverage. Therefore a moment controller must be judged on coverage and tails, not only its moment objective.

### 3.3 What exactly should be preserved?

The follow-up admits three useful interpretations; this revision addresses all three rather than treating them as interchangeable:

| Meaning of spread | Desired preservation | Relationship to band coverage |
|---|---|---|
| **Structural diversity** | Different valid graphs, scaffolds, and meaningful geometries among successful samples | Can coexist with a narrow property band if many structures realize it; feasibility and generator support still matter |
| **Property spread inside the band** | Nonzero central quantile width and a sensible range of accepted property values | Compatible with coverage, but limited by the band's width |
| **Current overall property spread** | The original standard deviation or centered distribution across all attempts | May conflict sharply with high coverage; can also be sustained artificially by rare, distant failures |

Until a single preference is fixed, the recommended primary objective is high decoded useful yield with structural diversity and a predeclared central-spread floor. Treat literal retention of current overall variance as a separately reported, stricter variant. Do not silently substitute one definition after seeing results.

For structural diversity, compare equal-sized sets of valid in-band molecules, using fixed graph fingerprints/scaffold definitions and atom-count strata. Also report total distinct successful yield. Pairwise diversity from a handful of survivors is unstable and can look high because the easy, common structures were lost. A mean pairwise distance alone does not rule out mode collapse.

For property spread, use the interquartile range, a central 80% width, and a width computed *among successful samples*, alongside variance and tail quantiles. A spread floor should be chosen from a held-out, matched valid in-band reference when available. Preserving the unguided model's unconditional structural distribution is generally impossible when that distribution is correlated with the target property.

### 3.4 Bias correction can preserve the whole centered shape

For any scalar outcome law with finite mean, define

\[
b=\mathbb E[Y]-y,\qquad Y'=Y-b.
\]

Then E[Y']=y and Var(Y')=Var(Y). More strongly, every centered quantile difference and every pairwise property difference is unchanged. On a batch, the desired scalar edit is simply

\[
\nu_i=d\quad\text{for every }i,\qquad d=-\eta_\mu(\mu-y).
\]

This gives a clean baseline for “fix bias, keep spread.” It also shows why a shared guidance multiplier is not enough: ordinary per-sample property changes depend on both residuals and different response gradients. A common field coefficient does not generally produce common property increments.

This is a statement about changing actual properties, not subtracting b from the reported predictions. A molecular implementation must realize the desired increments by state edits and verify the completed molecules. Translating atomic coordinates rigidly is not a solution either: the property models are translation-invariant.

Under the local response approximation in Section 4, the corresponding edit is

\[
d_i^{\mathrm{state}}=\frac{d}{r_i}M^{-1}a_i.
\]

It can be prohibitively costly for a low-mobility trajectory. The largest common allowed increment is bounded by \(\min_i R_i^{\mathrm{edit}}\sqrt{r_i}\). Thus literal shape preservation has a weakest-trajectory bottleneck. Relaxing that bottleneck requires an explicitly quantified loss of shape preservation, not a claim that individual clipping leaves the distribution unchanged.

### 3.5 Centering alone is not enough, and can sometimes hurt coverage

For an illustrative Gaussian law with bias b and standard deviation sigma, band probability is

\[
p(b,\sigma)=\Phi((\delta-b)/\sigma)-\Phi((-\delta-b)/\sigma).
\]

An exact translation to the center gives p(0,sigma). Using the observed moments only as parameters of this **unverified Gaussian illustration**:

| Cell | Gaussian coverage before centering | Gaussian coverage after centering at unchanged spread |
|---|---:|---:|
| mu / plug | 13.41% | 13.43% |
| alpha / plug | 3.99% | 4.48% |
| gap / plug | 15.86% | 17.73% |

These are not estimates of an executed molecular intervention, and their difference from observed coverage shows why the Gaussian assumption cannot be taken for granted. They illustrate the basic difficulty: translating a distribution whose width is many band widths does not make most of it fit.

There is no distribution-free guarantee that centering increases coverage. At target zero and half-band 0.5, a law with 90% at zero and 10% at ten has bias one, variance nine, and 90% coverage. Subtracting its bias preserves variance and reduces bias to zero but moves the bulk to −1, yielding zero coverage. This is why centering is a **candidate intervention with an acceptance check**, not an unconditional first action. When the mean is driven by a few outliers, a robust location objective or targeted tail repair may be preferable, with signed bias still reported.

With sample-level values, a cheap diagnostic is to calculate the entire translation response curve

\[
p(s)=B^{-1}\sum_i\mathbf1\{|z_i+s-y|\le\delta\}.
\]

Its unconstrained maximum is the fraction of samples contained in the densest interval of width 2delta. Compare the zero-bias shift with the best shift satisfying a small bias bound. This determines whether *any common translation* can solve the problem while preserving shape. It is an offline diagnostic; using observed f_B values this way is not an online permissible controller or a new molecular outcome. The saved aggregate JSONs cannot reconstruct this curve.

### 3.6 A precise limit on keeping wide spread inside a narrow band

If every output satisfies |Y−y|≤delta, then

\[
\boxed{\operatorname{Var}(Y)+b^2\le\delta^2.}
\]

Proof: E[(Y−y)²]≤delta², and E[(Y−y)²]=Var(Y)+b². Thus the current 4–18-delta standard deviations in the table cannot be preserved with 100% band coverage. This does not prevent preserving rich **structural** diversity.

For a softer requirement, suppose all residuals obey a declared tail cap |Y−y|≤R with R>delta, and let p be band probability. Then

\[
V+b^2\le p\delta^2+(1-p)R^2,
\qquad
\boxed{p\le\frac{R^2-V-b^2}{R^2-\delta^2}.}
\]

Intersect this necessary bound with [0,1]; a negative right-hand side means the requested moments violate the assumed tail cap. This is an upper bound, not a sufficient construction. It requires an actual support cap; a sample maximum is not a justified population cap. The same algebra is valid for a finite batch when its maximum residual is used strictly as a finite-batch statement.

For example, preserving V=2delta² with zero bias and tail cap R=3delta permits at most 87.5% coverage. Requesting 90% requires some residual magnitude at least sqrt(11)delta≈3.317delta. Without a tail cap, one can keep enormous variance by moving a small failure fraction arbitrarily far away. That is why preserved variance alone is an unsafe definition of success for this research objective.

### 3.7 Prefer “preserve differences that matter” to “preserve every source of variance”

With a structural or atom-count stratum S,

\[
\operatorname{Var}(Y)=\mathbb E[\operatorname{Var}(Y\mid S)]
+\operatorname{Var}(\mathbb E[Y\mid S]).
\]

Oppositely biased subgroups can cancel in the overall mean. Removing subgroup offsets can preserve all within-group property differences and the number of samples in each group while reducing overall variance. That decrease is not necessarily loss of useful diversity; it may remove systematic misalignment. Conversely, forcing the old overall variance to remain would preserve part of the defect.

The audit demonstrates this with groups {−3,−2,−1} and {1,2,3}. Their joint mean is zero, but their group means are −2 and +2. Centering each group preserves within-group variance 2/3 and all within-group spacings, while overall variance falls from 14/3 to 2/3. Molecular group-centering remains an untested intervention, and groups with no attainable target support must not be forced to hit by destroying chemistry.

A useful empirical objective is therefore: **small global and prespecified subgroup bias; high valid band coverage; preserved structural coverage and central within-band width; controlled tails; low edit cost.** These form a constrained trade-off, not a promise that every number can improve simultaneously.

An ideal distributional benchmark clarifies the structural goal. If the reference law is p_ref(x), Y=f(x) has density p_Y, and a desired band-compatible property law pi is supported where p_Y>0, then

\[
q(x)=p_{\rm ref}(x)\frac{\pi(f(x))}{p_Y(f(x))}
\]

has property marginal pi and preserves the reference conditional distribution of structures at each exact property level. This is the established marginal-reweighting idea of [Jeffrey guidance](https://arxiv.org/abs/2606.13240), not a new proposal here. It explains how property concentration can coexist with varied structures; it does not guarantee unchanged overall scaffold frequencies when those correlate with property. The practical controllers in this memo are not proven to sample this ideal law.

## 4. Shared construction and implementation contract

Use the same frozen generator and f_A. f_B evaluates completed candidates; it is never an online controller, switch selector, or assignment cost. Validation can select hyperparameters under a declared tuning budget; final testing uses fresh seeds and the existing held-out protocol.

Work on the padded, coordinate-centered state subspace with a fixed dimensionless metric M. M may give coordinates and atom features different declared scales. It must be positive definite **on that subspace**. In formulas below, inverses and gradients act there; implement padding and centering explicitly. A positive definite metric is not a chemical validity guarantee.

At an intervention anchor, define a **terminal forecast**

\[
z_i=F_t(x_i),\qquad a_i=\nabla_{x_i}F_t(x_i),\qquad
r_i=a_i^TM^{-1}a_i.
\]

Here r_i measures squared terminal-property sensitivity per unit edit cost. It is **not posterior variance**. Two forecast variants must remain distinct:

- **Reference experiment:** continue the frozen, deterministic sampler along a specified suffix \(\Psi_{t\to1}\), and set \(F_t=f_A\circ\Psi_{t\to1}\). Compute derivatives through that actual continuation, with checkpointing if necessary. This costs additional generator evaluations and reverse passes.
- **Practical approximation:** use f_A(m_t), or a short continuation followed by a clean-state estimate. This is cheaper but inherits the off-distribution and terminal-response errors already identified. Validate its predicted edits against full suffix outcomes before trusting it.

For the diagonal response formulas below, the reference suffix must act independently on each trajectory. If the forecast itself includes future batch-coupled RATV/RBAG decisions, each z_i can depend on every x_j; then a full cross-particle response matrix is required and the diagonal r_i derivation does not apply. Start with a fixed sample-wise reference suffix and treat later interventions through repeated re-estimation.

For an exact smooth continuation, F_t solves the transport relation \(\partial_tF_t+\nabla F_t^Tv=0\) along its reference field. Extra state velocity d therefore changes the forecast at rate aᵀd. For an approximate forecast, subtract its measured baseline drift when choosing the desired moment change. A discrete implementation must verify the complete baseline-plus-edit step; it cannot assume this continuous cancellation holds exactly.

The initial differentiable forecast uses continuous features. Score completed samples again with decoded features and compare the two. Do not backpropagate through argmax and pretend it is an ordinary derivative. A decoded finite-difference alternative is a separate, more expensive estimator that must be labeled and tested.

Proposals 1 and 3 are defined below as **state displacements at sparse intervention anchors**, not likelihood scores. Insert a displacement after a documented baseline step. Do not multiply it a second time by the flow score-to-velocity factor (1−t)/t. Ordinary score-based arms retain their existing conversion. Keep these units separate in code and logs.

No model retraining is required by these proposals. Pilot calibration, extra rollouts, assignment solving, and mathematical checks are still computational costs and must be reported.

## 5. Proposal 1 — Reachability-Aware Target-Variance Guidance

### 5.1 The research question

**Can we reduce terminal property bias with minimal distortion of the centered distribution, then raise band coverage while retaining a feasible amount of useful spread?**

For a biased batch, the first request is mean correction with little change in centered shape. For a well-centered batch, further mean attraction can be unhelpful; the request becomes improved coverage at a controlled spread. RATV separates those requests and allocates edits through measured terminal responses. The original contraction controller below is retained as an optional comparison, rather than the revised default.

Section 2.2a supplies the empirical motivation: several selected arms miss the center by more than one tolerance half-band, while their distributions are also much wider than the band. A large variance share of MSE does not resolve which edit preserves useful diversity. The revised centering and coverage ablations test that directly.

### 5.2 Derivation

For a batch of B forecasts z_i at the same target y, define

\[
\mu=\frac1B\sum_i z_i,\quad c_i=z_i-\mu,\quad
V=\frac1B\sum_i c_i^2.
\]

Let \(\nu_i=a_i^Td_i\) be the first-order property change produced by state displacement d_i. Then

\[
\Delta\mu=\frac1B\sum_i\nu_i,
\qquad \Delta V=\frac2B\sum_i c_i\nu_i+O(\|d\|^2).
\]

In the optional contraction variant, request small changes

\[
b_\mu=\eta_\mu(y-\mu),\qquad
b_V=-\eta_V(V-\tau^2)_+,
\]

where \((q)_+=\max(q,0)\), tau>0 is the desired maximum standard deviation, and both step parameters are nonnegative. This version contracts excess variance without deliberately expanding a batch already below target. At a centered mean, b_mu=0 imposes mean preservation to first order.

**Revised centering phase:** instead set b_mu=eta_mu(y−mu) and **b_V=0**, and use the shape regularizer in Section 5.8. Setting b_V=0 in the least-cost formula alone preserves variance only to first order; it need not preserve the centered shape or even finite-step variance.

For a fixed property change nu_i, solve

\[
\min_{d_i}\ \tfrac12d_i^TMd_i\quad\text{subject to}\quad a_i^Td_i=\nu_i.
\]

The Lagrange multiplier solution, for r_i>0, is

\[
d_i^*=\frac{\nu_i}{r_i}M^{-1}a_i,
\qquad \tfrac12(d_i^*)^TMd_i^*=\frac{\nu_i^2}{2r_i}.
\]

Define the 2×B matrix C by C_1i=1/B and C_2i=2c_i/B, and R=diag(r_i). The resulting batch problem is

\[
\min_\nu\ \tfrac12\nu^TR^{-1}\nu
\quad\text{subject to}\quad C\nu=b,
\qquad b=(b_\mu,b_V)^T.
\]

If CRCᵀ is invertible, its unique solution is

\[
\boxed{\nu^*=RC^T(CRC^T)^{-1}b.}
\]

Only a 2×2 linear solve is required after estimating the B sensitivities. Substitution verifies Cnu=b; stationarity verifies minimal edit energy. The solve itself is standard constrained least squares. The candidate contribution is its terminal-response use, mean-preserving concentration policy, and bounded molecular implementation.

### 5.3 What the formula actually guarantees

Under an exact smooth terminal forecast and continuously recomputed, feasible, uncapped controls, the optional contraction choice of moment *rates* produces

\[
\dot\mu=k_\mu(y-\mu),\qquad
\dot V=-k_V(V-\tau^2)_+.
\]

Thus the mean approaches y, and variance above tau² approaches tau². This is a local control result under exact response information. It is not a theorem about learned predictors, decoded chemistry, or the distribution generated by a finite number of interventions.

Even with exact linear property changes, a finite step satisfies

\[
V(z+\nu)=V(z)+\frac2B\sum_i c_i\nu_i+\operatorname{Var}(\nu).
\]

The extra nonnegative term prevents replacing the local prediction with a finite-step guarantee. With nonlinear F there are additional Taylor remainders. Use a line search on the actual forecast and verify the proposed baseline-plus-edit continuation.

**Degeneracies:** r_i=0 means that trajectory has no first-order property mobility; fix nu_i=0 rather than divide by a small number. If all usable c_i are identical, C loses rank and independent mean/variance control is impossible. A ridge term can improve numerical conditioning but changes the exact constraints. Record constraint residuals instead of claiming preservation.

### 5.4 Bounded practical version

Let the allowable state-edit radius be R_i^edit in the M metric. Since the minimum edit norm is |nu_i|/sqrt(r_i), impose

\[
|\nu_i|\le R_i^{\mathrm{edit}}\sqrt{r_i}.
\]

Use the constrained quadratic problem, with explicit slacks on the moment requests when infeasible. Prefer a small mean-error allowance and a soft variance-reduction objective over an infeasible exact command. Penalties are fixed on validation and disclosed.

One explicit implementation minimizes

\[
\tfrac12\nu^TR^{-1}\nu+
\tfrac{\lambda_\mu}{2}(C_1\nu-b_\mu)^2+
\tfrac{\lambda_V}{2}(C_2\nu-b_V)^2
\]

under the bounds above. Normalize the property by a fixed guide-side scale before selecting these penalties so they do not change arbitrarily between Debye, atomic units, and Hartree. This soft version does not exactly preserve the mean. When mean preservation is essential, impose C_1nu=b_mu as a hard constraint and relax only the variance request; if that constraint is itself infeasible, explicitly fall back and report it.

**Do not independently clip each closed-form d_i afterwards and continue claiming Cnu=b.** Such clipping destroys the coordinated moment constraints. Either solve with the bounds included, or scale every d_i by the same line-search factor and accept proportionally reduced progress.

Start with M=I after established coordinate/feature normalization. A learned chemical-cost metric would be a different method. A coordinate-only intervention can be a useful late-stage ablation, but freezing atom types can make the target unattainable and must not be assumed to protect validity.

```text
At a preselected intervention anchor:
  Forecast each trajectory's terminal property using f_A.
  Estimate its response gradient and response reliability.
  Compute the batch mean, variance, and requested moment changes.
  Remove zero-response directions; solve the bounded moment allocation.
  Convert scalar property changes to state edits.
  Backtrack using actual forecast evaluations and the declared trust radius.
  Apply the accepted edits; record predicted and realized moment changes.
Continue the sampler and evaluate decoded outputs with f_B only afterwards.
```

### 5.5 The crucial falsification: is this just MSE guidance?

With equal r_i, the uncapped solution simplifies to

\[
\nu_i=b_\mu+\frac{b_V}{2V}(z_i-\mu).
\]

If tau=0, b_mu=−k(mu−y), and b_V=−2kV, then

\[
\nu_i=-k(z_i-y).
\]

That is ordinary contraction toward y. The audit checks this reduction. A constant-r, zero-target-variance implementation cannot be sold as a new mechanism.

RATV must beat **normalized squared-error guidance with the same forecast, gradient information, trust budget, and intervention times**. It must also beat a simple batch variance loss plus a mean penalty. A win over the existing cheap endpoint approximation alone could come entirely from the more expensive forecast.

### 5.6 Novelty and prior work

Moment matching itself is occupied by [Moment Guided Diffusion](https://arxiv.org/abs/2602.17211). Batch variance interactions are occupied by [Variance-Tilted Diffusion](https://arxiv.org/abs/2606.22239), which explicitly targets spread through a batch distribution. Geometry-aware terminal control is occupied by [TOCFlow](https://arxiv.org/abs/2601.09474). These works rule out broad claims such as “the first guidance that controls variance.”

**Revised candidate claim:** a terminal-response controller that distinguishes approximately shape-preserving bias correction from coverage repair, with explicit central-spread and tail safeguards, and improves decoded useful yield at matched structural diversity and compute. Its incremental content depends on measured response heterogeneity and benefits beyond existing moment-control and constrained-gradient primitives.

This proposal deliberately does not claim exact sampling from a likelihood-tilted posterior. It is a finite-batch steering controller.

### 5.7 Required experiments and stop conditions

Use 128 trajectories per coupled batch initially, with independent batches as statistical units. Try sparse anchors such as 0.50, 0.65, and 0.75 only after response calibration; these times are hypotheses.

Compare: tuned plug-in; tuned MC baseline; normalized per-sample MSE; mean-only control; variance-only control; equal-r moment allocation; actual-r allocation; and full versus approximate terminal forecasts. All controller comparisons share their forecast cost.

Measure terminal mean, variance, decoded in-band rate, valid-and-in-band yield, molecule stability, response calibration, active bounds, infeasible requests, and full runtime. Repeat at q90; a median-only success may simply exploit an already-centered generator.

**Stop or simplify** if equal-r control is indistinguishable, normalized MSE explains the gain, soft-feature variance contracts without decoded improvement, response estimates are unreliable, or cost per useful molecule worsens. If the observed chemical deterioration is due to a weak base model rather than concentration allocation, say so.

### 5.8 Revision: preserve shape during centering, not merely a variance derivative

Let P=I−11ᵀ/B remove a batch mean. For scalar property increments nu, use

\[
\min_\nu\ \tfrac12\nu^TR^{-1}\nu+
\tfrac{\lambda_{\rm shape}}2\|P\nu\|^2
\quad\text{subject to}\quad C\nu=(b_\mu,0)^T.
\]

The first term is local state-edit cost. The second penalizes different property increments across trajectories, because those differences distort the centered outcome configuration. It is zero for a common translation. Indeed,

\[
\sum_{i<j}(\nu_i-\nu_j)^2=B\|P\nu\|^2.
\]

Define H=R⁻¹+lambda_shape P. For positive mobilities, H is positive definite and the uncapped solution is

\[
\boxed{\nu_b=H^{-1}C^T(CH^{-1}C^T)^{-1}(b_\mu,0)^T.}
\]

As lambda_shape increases, this penalizes pairwise shape distortion more strongly. If a common translation is feasible, the large-penalty limit approaches it. With trust bounds, exact translation can remain infeasible; record the distortion and any relaxed moment request. The one-parameter shape penalty exposes a meaningful trade-off between low edit cost and preservation, rather than hiding it in a generic guidance strength.

This is standard regularized constrained optimization. The research question is whether property-space shape preservation buys useful molecular outcomes beyond cheaper normalizations and the fixed-variance least-cost controller. The audit checks a heterogeneous five-trajectory example: increment variance decreases from 0.03480 at lambda_shape=0 to approximately 7.18×10⁻¹² at 100,000, while the requested mean/variance rates are satisfied numerically.

Do not infer structural diversity preservation from property-spacing preservation. A state edit can retain all scalar property differences and still damage graphs or move every sample toward the same structural family.

### 5.9 Revision: coverage improvement in directions that do not move the moments

When centering is adequate but coverage remains low, the next question is whether mass can be redistributed while approximately retaining both moments. Define a differentiable band utility

\[
h_\epsilon(e)=\operatorname{sigmoid}((e+\delta_A)/\epsilon)
-\operatorname{sigmoid}((e-\delta_A)/\epsilon),
\quad \widetilde p=B^{-1}\sum_i h_\epsilon(z_i-y),
\]

and let ell_i=h_epsilon'(z_i−y)/B. For a centered batch, solve the first-order coverage problem

\[
\max_\nu\ \ell^T\nu-\tfrac\eta2\nu^TH\nu
\quad\text{subject to}\quad C\nu=0,
\]

where eta>0. In the uncapped, full-rank case,

\[
Q_H=H^{-1}-H^{-1}C^T(CH^{-1}C^T)^{-1}CH^{-1},
\qquad
\boxed{\nu_{\rm cover}=\eta^{-1}Q_H\ell.}
\]

Since C Q_H=0 and Q_H is positive semidefinite,

\[
C\nu_{\rm cover}=0,\qquad
\ell^T\nu_{\rm cover}=\eta^{-1}\ell^TQ_H\ell\ge0.
\]

This proves nonnegative **first-order smooth-coverage** progress with zero mean and variance rates. It does not prove increased hard-band count, exact finite-step moments, validity, or retained structural diversity. If Q_H ell=0, the available first-order coverage gradient conflicts completely with the requested moment preservation. That is a useful infeasibility signal; do not amplify a numerically tiny direction indefinitely.

For simultaneous centering and coverage repair, the unconstrained formula is nu=nu_b+eta⁻¹ Q_H ell. The coverage component is nonnegative relative to nu_b at first order, but ellᵀnu_b can be negative. Therefore the total update still needs a coverage acceptance test, as the asymmetric counterexample in Section 3.5 demonstrates.

The key difficulty is that fixed-moment redistribution can move some samples **away** from the band to compensate for moving others into it. This is exactly the tail-inflation failure. The practical constrained variant must include response trust bounds, a central-width floor, and a tail safeguard; it must evaluate the full nonlinear forecast after a proposed step. If those conditions leave no feasible positive improvement, relax overall property-variance preservation by a disclosed amount or leave the baseline unchanged. Never manufacture outliers just to retain an attractive variance number.

The smooth band derivative is small far outside the band. Use the centering phase, a prespecified smoothing continuation, or target assignment to reach a region with a useful coverage signal. Aggressively widening epsilon changes the objective and must be counted in tuning.

For structural diversity, an optional **control**, not a new fourth proposal, uses a per-molecule tangent direction

\[
w_i^\perp=w_i-M^{-1}a_i\frac{a_i^Tw_i}{r_i},
\qquad a_i^Tw_i^\perp=0,
\]

where w_i is a differentiable structural-diversity proposal. This preserves each forecast property to first order, a stricter condition than preserving only the batch mean. Finite-step and decoded checks still apply. This machinery is related to [Harmless Diversity](https://proceedings.mlr.press/v162/gong22b.html) and already appears in this project's earlier tangent-control discussion; it is an important baseline, not a novelty claim. [Particle Guidance](https://arxiv.org/abs/2310.13102) is another existing diversity comparator.

### 5.10 Revised decision rule and the comparisons that can reject it

```text
At a calibrated anchor, using f_A forecasts only:
  Measure bias in band units, coverage, central spread, tails, and response quality.
  If bias is too large:
    propose a common-property centering edit or its bounded shape-regularized version;
    accept only if the declared coverage and quality guard allows it.
  If coverage is too low after that:
    try moment-preserving coverage repair with central-spread and tail safeguards;
    if infeasible, try the smallest disclosed spread relaxation or band-slot assignment.
  If targeting is already adequate:
    stop property correction; preserve diversity through the baseline or a tested tangent control.
  Verify completed decoded molecules on held-out evaluation after sampling.
```

Use a fixed reference distribution or reference batch recorded at the start of the correction block. If the preservation reference is reset to the latest degraded state at every step, many small accepted losses can compound into severe collapse. Report cumulative distortion from the original reference as well as each local step's change.

Add these controlled comparisons to Section 5.7: common-property translation; least-cost centering with b_V=0; shape-regularized centering; coverage repair with and without tail/central-width safeguards; standard variance contraction; and a published-style diversity control at matched target performance. Reject the revised mechanism if its only advantage is manufactured tails, if mean correction worsens useful yield, or if structural diversity falls at matched on-target sample count.

## 6. Proposal 2 — Order-Sensitive Handoff Guidance

### 6.1 Why “A from 0.05 to 0.75, then B” is not yet supported

Your proposed schedule is a sensible experiment. The current evidence provides two motivations: early guide behavior can be unreliable, and full-window arms show different concentration/quality trade-offs. It provides **no measured mixed-schedule benefit**.

Moreover, the current flow conversion is

\[
u_A(t)=w_A\frac{1-t}{t}G_A(x_t,t),
\]

before clipping. The scalar multiplier is 19 at t=0.05, 1 at 0.5, 1/3 at 0.75, and about 0.111 at 0.9. Its integral from 0.75 to 1 is only about **1.84%** of its integral from 0.05 to 1. This is an explanatory calculation, not a measure of actual guidance work: field norms, clipping, state changes, and the discrete grid all matter.

Therefore, weak late-only guidance at w=1 does not establish a structural impossibility of late control. It may have little effective strength under the current conversion. Nor should a huge late multiplier be used without a trust-region comparison: it can damage chemistry.

### 6.2 The proposed mechanism

The new hypothesis is that **local order sensitivity identifies when one guidance method prepares states that another can improve**. This differs from simply measuring whether a closure is accurate.

Let A(x,t) and B(x,t) denote the **full velocity fields**, including the common generator, conversion, strength, and clipping. At a saved state, compare two shadow interventions of equal duration and exposure:

\[
x_{AB}=\Phi_B^h\circ\Phi_A^h(x),\qquad
x_{BA}=\Phi_A^h\circ\Phi_B^h(x).
\]

Both use A for h and B for h. A difference cannot be attributed merely to assigning more nominal duration to one method.

For smooth, time-independent fields,

\[
x_{AB}-x_{BA}
=h^2\{D B\,A-D A\,B\}+O(h^3).
\]

For time-dependent fields, augment the state by time. The state component becomes

\[
\boxed{\mathcal B_{AB}=D_xB\,A-D_xA\,B+\partial_tB-\partial_tA.}
\]

The difference is h² times this augmented bracket to leading order. Omitting the time-derivative terms is incorrect for the actual sampler. The base field does not simply cancel inside the bracket; it contributes cross terms. The audit checks the sign and the time-dependent terms on an explicit example, plus a commuting null case.

For a smooth terminal loss \(L\circ\Psi\), the leading order preference is

\[
L(\Psi(x_{AB}))-L(\Psi(x_{BA}))
\approx h^2\nabla(L\circ\Psi)^T\mathcal B_{AB}.
\]

This gives a mechanism for an order effect. It does not predict the sign of a full 0.05–0.75 schedule, and a nonzero bracket is not automatically beneficial.

### 6.3 Practical gate without high-order autodifferentiation

Use the paired finite interventions themselves, followed by a common continuation policy, to estimate an order advantage with f_A. Use a prespecified soft band loss, for example

\[
L_A(z)=\operatorname{softplus}\!\left(\frac{|z-y|-\delta_A}{\epsilon_A}\right),
\]

or a differentiable smoothed absolute value inside this expression. Evaluate decoded validity of shadow endpoints where feasible. A valid shadow endpoint is a measured outcome of that shadow branch, not a guarantee for another suffix.

Define \(D_{\mathrm{order}}=L_A(\Psi(x_{BA}))-L_A(\Psi(x_{AB}))\). Positive values favor A then B. Also measure the separate **handoff advantage**: continuing with B after the shared prefix must improve on staying with A. Order advantage alone does not establish that either mixed policy beats A throughout.

A conservative first policy is:

1. Run A until one of at most two validation-chosen decision anchors.
2. Evaluate paired AB/BA shadow pulses and a stay-with-A continuation using common random numbers.
3. Switch permanently to B only if the calibrated order signal favors AB, the actual handoff comparison favors B, and shadow quality meets the predeclared criterion.
4. Otherwise keep A; permit one later decision, then stop testing.

Use a batch-level gate first. Per-molecule gates are more expressive but introduce selection noise and can change the outcome law substantially. A pilot-derived margin or confidence bound should account for paired variability; selecting the larger of two noisy estimates without a holdout creates optimism.

Freeze probe draws across paired finite differences. Redrawing Hutchinson or MC noise independently can make the estimated order effect mostly estimator noise. Clipping and decoder transitions are nonsmooth; the bracket approximation is limited away from those boundaries, although the directly measured finite interventions remain meaningful.

### 6.4 Concrete schedules worth testing first

All intervals below use flow time 0=noise and 1=data, with unguided sampling before 0.05. Let tau_switch be tested on a small prespecified grid, initially {0.50, 0.65, 0.75}.

| Property | A on [0.05, tau_switch) | B on [tau_switch, 1) | Evidence motivating the pair | Status |
|---|---|---|---|---|
| mu | lgd_mc | plug | Full-window LGD has much better stability; plug has lower observed MAE | Mechanistic hypothesis; segment advantage unknown |
| alpha | osc | plug | OSC is strong over the full interval; removing early plug guidance helps alpha | Especially useful for testing whether early plug is harmful |
| gap | tfg_mc | osc | TFG has lowest observed MAE; OSC has higher in-band rate and stability | Different metrics motivate a concentration/quality handoff |
| Any, after proposal 1 passes | strongest validated A | RATV | Replace generic late attraction with measured mean/variance control | Untested combination of two changes; run only after separate ablations |

**Additional mechanistic arm:** MC → `smg_mean` or `smg2` after a repaired closure diagnostic, compared with MC → `plug` and MC → off. This tests whether the late curvature term adds anything. Do not make it the default recommendation: the present SMG family is more expensive than these MC arms, and no evidence shows a late closure adds useful yield.

At tau_switch=0.75, the exact requested design is A on [0.05,0.75) and B on [0.75,1). The switch remains a tested hyperparameter until a mechanism-specific advantage is demonstrated.

### 6.5 The experiment that establishes empirical support

For each candidate A/B pair, use the same initial seeds, masks, and declared probe streams. Run:

| Policy | Before switch | After switch | Question answered |
|---|---|---|---|
| 00 | off | off | Matched baseline |
| A0 | A | off | What early A achieves alone |
| 0B | off | B | What late B achieves alone |
| AA | A | A | Whether a handoff is better than staying with A |
| BB | B | B | Whether a handoff is better than using B throughout |
| AB | A | B | Proposed fixed handoff |
| BA | B | A | Long-schedule order control |
| blend | fixed mixture of A/B | same mixture | Whether switching helps beyond simultaneous mixing |

The long AB/BA policies have unequal A/B durations unless tau_switch is the midpoint. They test practical schedule order but **do not isolate pure order**. The short equal-exposure pulse experiment does that.

Let U be distinct valid-and-in-band outputs per attempt. Estimate

\[
\Delta_{\mathrm{handoff}}=U_{AB}-U_{AA},\qquad
\Delta_{\mathrm{interaction}}=U_{AB}-U_{A0}-U_{0B}+U_{00}.
\]

The second quantity is a factorial interaction on this outcome scale. Positive interaction is useful mechanistic evidence; it is not required for AB to be useful, and it is not by itself proof of broad chemical synergy.

Save A's state at the switch and branch its suffix to A, B, and off. This makes the comparison about the same early trajectories. Also retain full independent-batch replications to quantify seed variability. Pair intervals at the batch level, with all failed attempts retained.

Give each arm its own tuned strength. Log realized edit energy \(\sum_k h_k\|u_k\|_M^2\), displacement exposure \(\sum_k h_k\|u_k\|_M\), clipping fraction, and actual runtime. Compare both equal-cost and equal-exposure variants; matching nominal w alone is insufficient. The blend gets the same overall clipping rule as the switched arm.

### 6.6 Novelty, strongest objection, and stopping rule

Restricting guidance to a time interval is already established by [Kynkäänniemi et al.](https://arxiv.org/abs/2404.07724). Rollout-based guidance and optimal control also have extensive precedent, including [TOCFlow](https://arxiv.org/abs/2601.09474). The Lie bracket is classical control mathematics; it is not invented here.

**Candidate claim:** an equal-exposure, terminal-scored order diagnostic that predicts beneficial molecular guidance handoffs beyond time, residual, guidance norm, and clipping, and supports an economical gate. This is narrower than claiming a new scheduler simply because it changes arms.

The strongest objection is that OSHG is just expensive lookahead selection with a theoretical interpretation attached. To answer it, compare with the same-cost direct “stay A versus switch B” lookahead **without the order test**. If the order diagnostic adds no predictive or performance value, drop that mechanism and describe the surviving result as a simple empirical handoff.

Before molecular claims, repeat local pulse tests at h and h/2, with Euler and Heun at controlled work. A score divided by h² should stabilize in a smooth regime. A benefit appearing only at one coarse step size is evidence for a solver artifact. If a fixed time switch achieves the same useful yield at lower cost, prefer it and reject the adaptive novelty claim.

### 6.7 Revision: switch objectives when the distribution needs a different edit

The follow-up strengthens the case for a **centering-to-coverage** handoff rather than assuming that late guidance should simply be stronger. A candidate A phase establishes valid structure and reduces systematic location error. A candidate B phase improves occupancy subject to retained central spread and structural diversity. These phases can use the same base estimator with different control objectives; changing the method name is not the essential mechanism.

Keep the requested 0.05–0.75 A / 0.75–1 B schedule as a fixed comparator. An adaptive proposal should switch only when the forecast bias is below a validation-fixed threshold, such as 0.5delta_A as a starting **design choice**, coverage is still below its target, response estimates are reliable, and the candidate B continuation improves predicted coverage without violating spread and quality guards. If bias remains high at 0.75, a blind switch can abandon an unfinished task. If targeting is already adequate earlier, continuing to attract toward the center can unnecessarily consume spread.

Do not turn the coverage-only shadow loss into the decision rule without the preservation checks. Compare candidate continuations lexicographically: first enforce validity/quality and a feasible preservation budget, then compare coverage and mean error, then compare cost. Avoid compensating a large diversity loss with an arbitrary numeric weight on a slight property gain.

For the OSHG order experiment, test **center then repair** against **repair then center** at equal exposure, and compare both with center alone, repair alone, and ordinary contraction. Evidence of a useful order effect must survive the central-width, tail, and structural-diversity checks. A sharp result would be that the first phase improves the attainable coverage direction without spending additional structural diversity; that remains a hypothesis, not a consequence of the bracket formula.

## 7. Proposal 3 — Reachable Band-Assignment Guidance

### 7.1 The research question

**If a whole interval counts as success, can we assign each trajectory a suitable target within it, while preserving a prescribed nonzero property spread and spending less molecular edit effort?**

RATV controls two moments without deciding where each trajectory should finish. RBAG instead specifies a finite target distribution and allocates its slots to trajectories. Hard-to-move trajectories receive nearby slots when possible; responsive trajectories carry more of the adjustment. This is a different mechanism from imposing the same dead-zone loss on every sample.

This is also a different goal from BPS: RBAG changes individual trajectories and their marginal property distribution. It makes no marginal-preservation claim. It does not clone, discard, or resample molecules.

### 7.2 Specify the finite target law

Choose B target slots inside a guide-side band, for example

\[
q_j=y+a\left(\frac{2(j-\tfrac12)}B-1\right),\qquad j=1,\ldots,B.
\]

These equally spaced midpoints have

\[
\frac1B\sum_jq_j=y,\qquad
\frac1B\sum_j(q_j-y)^2=a^2\frac{B^2-1}{3B^2}.
\]

Their largest absolute offset is a(1−1/B). Thus a≤delta_A places every slot strictly inside the guide-side band. The variance is approximately a²/3, not a². If prescribing tau² directly, use \(a=\tau\sqrt{3B^2/(B^2-1)}\) and check the support constraint. Arbitrary mean, variance, and bounded support requests are not always compatible.

A uniform-in-band target is a design choice, not the exact model conditional law. Compare it with truncated-Gaussian slots and the single-target baseline at equal tuning effort. Do not claim that preserving property spread automatically preserves graph or scaffold diversity.

### 7.3 Derive the trajectory-specific assignment cost

At a shared anchor use the forecast z_i, response a_i, and mobility r_i from Section 4. Assigning trajectory i to slot j requires first-order change

\[
\Delta_{ij}=q_j-z_i.
\]

The minimum local state-edit energy is

\[
\boxed{C_{ij}=\frac{(q_j-z_i)^2}{2r_i}.}
\]

The corresponding displacement is \(d_{ij}=M^{-1}a_i\Delta_{ij}/r_i\). This follows from the same one-constraint minimum-energy problem as RATV. The cost describes the declared state metric, not an independently validated chemical damage model.

Under an edit radius R_i^edit, a full local assignment is feasible only when

\[
|q_j-z_i|\le R_i^{\mathrm{edit}}\sqrt{r_i}.
\]

Set infeasible edges to infinity and solve

\[
\min_{\pi\text{ a permutation}}\sum_i C_{i,\pi(i)}.
\]

Every slot has capacity one. The matching objective prevents assigning all molecules to the easiest target value. A full matching need not exist even when every trajectory can reach at least one slot; too many trajectories may compete for the same subset of slots.

Use a standard linear-assignment solver; do not claim the optimizer as new. For B≈128, assignment itself is likely cheaper than neural response estimation, but this must be measured rather than assumed. Exact assignment has worst-case cubic scaling; a relaxed transport plan is a separate approximation and should not silently replace one-to-one targets.

### 7.4 Why ordinary sorted matching is an essential control

With equal r_i and no feasibility restrictions, the scalar squared-distance problem reduces to the familiar sorted assignment. That case is not a new guidance mechanism.

With unequal response costs, sorted matching can be wrong. Consider

\[
z=(-2,-1),\quad q=(-0.5,0.5),\quad r=(100,1).
\]

Omitting the common factor 1/2, sorted matching costs

\[
\frac{1.5^2}{100}+1.5^2=2.2725,
\]

whereas the reversed assignment costs

\[
\frac{2.5^2}{100}+0.5^2=0.3125.
\]

The more responsive trajectory accepts the farther target so the less responsive one can take the nearby target. With both r_i=1, sorted matching instead wins, at 4.5 versus 6.5. Both statements are exhaustively checked in the audit.

This example establishes a reason for response-aware assignment. It does **not** establish an advantage on molecular data. The first empirical gate is whether the actual response heterogeneity is large and predictive enough to change assignments profitably.

### 7.5 A safe implementation scope and explicit failures

Start from an existing A prefix, then perform a single assignment at a calibrated anchor. Freeze the assigned slots for that correction block to prevent target chattering. Recompute state gradients as needed, but do not continuously reassign without an explicit reassignment penalty and a separate ablation.

If full assignments are infeasible, choose one of two declared behaviors:

- **Strict feasibility variant:** apply no assignment intervention to that batch and continue the baseline; record infeasibility as an outcome.
- **Partial-progress variant:** use a common fraction lambda of the desired changes, subject to the local trust bounds, and reassess later. This changes the target law only partially and does not satisfy the slot distribution exactly.

For a fixed assignment with exact scalar interpolation, the latter has

\[
z_i'=(1-\lambda)z_i+\lambda q_{\pi(i)},
\]

\[
\mu'=(1-\lambda)\mu+\lambda y,
\]

\[
V'=(1-\lambda)^2V+\lambda^2V_q+
2\lambda(1-\lambda)\operatorname{Cov}(z,q_\pi).
\]

Thus a partial step does not automatically have variance V_q, and assignment order affects intermediate variance. Actual nonlinear state edits introduce further errors. Compare predicted and realized decoded outcomes after the suffix.

Keep masks and atom counts fixed across arms. Log results by atom-count strata because attainable properties can depend strongly on size. Optional within-size assignment avoids cross-stratum competition, but requires enough samples and may worsen feasibility. It cannot create molecules outside the generator's support.

For ties, use a documented exchangeable random tie-breaking rule. Individual-molecule permutation and rotation symmetries follow from invariant forecasts and a compatible metric. Batch permutation symmetry additionally requires that the assignment solver's tie behavior not systematically privilege array positions.

### 7.6 Novelty and the strongest competing explanations

[Histogram-constrained Image Generation](https://arxiv.org/abs/2606.31683) already uses optimal transport to impose distributional constraints during sampling, including explicit interventions on intermediate predictions. [Jeffrey guidance](https://arxiv.org/abs/2606.13240) already provides prescribed marginal control through a density-ratio construction. General distribution steering and “using OT during guidance” are therefore not novel claims.

**Candidate claim:** assigning a finite band of nonlinear molecular property targets using terminal-response edit costs and feasibility constraints, with unchanged attempt counts, and showing that heterogeneous costs outperform ordinary sorted assignment and independent target jitter. HIG is a particularly close conceptual comparator; the exact contribution requires formula-level comparison before publication.

The strongest alternative explanation is simply that different target offsets reduce overconcentration. Compare the same target slots under random assignment, sorted assignment, response-aware assignment, and fixed independent jitter. If response-aware allocation adds nothing, report that result and use the simpler method.

### 7.7 Tests and stopping rule

Use the same reference/approximate forecasts as RATV. Compare at identical slot laws, baseline prefixes, total state-edit energy, and neural evaluation budgets. Measure assignment feasibility, actual target residual per slot, response-cost calibration, terminal variance, decoded joint success, graph diversity, and runtime.

**Reject or reclassify** RBAG if it improves only the continuous-feature score, if its assignments barely differ from sorting, if gains disappear against random target offsets, or if it preserves target-property spread by sacrificing chemical validity. A negligible-cost sorted target control may still be useful engineering; it would not validate the proposed mechanism.

### 7.8 Revision: choose the least distorted feasible slots, with a central-spread floor

Uniform slots in Section 7.2 remain a clean baseline, but they can replace the entire original property shape unnecessarily. For the preservation objective, start instead from the sorted, centered reference forecasts

\[
\widetilde q_j=y+z_{(j)}-\mu.
\]

If these slots already lie in the band and are reachable, they preserve all centered property spacings. If not, choose q by the convex design problem

\[
\min_q\ \sum_j(q_j-\widetilde q_j)^2+
\lambda_{\rm gap}\sum_{j=1}^{B-1}
\left[(q_{j+1}-q_j)-(\widetilde q_{j+1}-\widetilde q_j)\right]^2
\]

subject to

\[
y-a\le q_1\le\cdots\le q_B\le y+a,\qquad
B^{-1}\sum_jq_j=y,\qquad
q_{j_{\rm hi}}-q_{j_{\rm lo}}\ge W_{\min}>0.
\]

Here a<delta_A leaves a declared margin, and j_lo/j_hi are fixed central order-statistic indices, for example the discrete 25% and 75% ranks with their exact convention specified. The last constraint is a central-spread floor. It is linear, unlike a lower bound on variance, which is generally a nonconvex constraint. W_min≤2a is necessary; the full constraints and later response assignment may impose stricter feasibility requirements.

The squared-distance term retains the reference quantile locations as far as possible. The gap term penalizes additional distortion of adjacent quantile spacings. Neither term guarantees preserved modes, graph diversity, or exact centered shape when the band forces compression. Choosing W_min very close to 2a can pile mass near band edges; evaluate the whole quantile curve and keep a safety margin.

Next solve the original response-cost assignment of these slots to trajectories. Slot design and assignment are distinct optimizations: the first keeps a feasible property shape, while the second asks which molecular trajectories can realize it cheaply. Their sequential solution is not a claimed global optimum of the joint nonlinear problem. Failure to match the slots within response bounds must trigger the declared partial-progress/fallback policy, not hidden sample deletion.

This revision makes preservation explicit: **minimize how much of the original centered property law must change, retain a nonzero central width, and assign the resulting targets without sacrificing validity.** The guarantee is only for the designed slots; generated outcomes must be measured. Compare against uniform slots and centered-reference slots without a gap penalty. If those simpler designs match the result, drop the extra slot-design machinery.

For literal overall-variance preservation with less than 100% coverage, one could add an outside-band slot fraction, but that explicitly budgets failures and invites tail inflation. It is not the default recommendation. Such a variant must declare its coverage target, tail cap, and unchanged all-attempt denominator before evaluation.

## 8. Prior-work audit and honest novelty boundaries

The following are primary sources consulted on 21 September 2026. The search was bounded and included guidance with target variance, moment constraints, switched guidance and Lie brackets, and target assignment/transport. A search that does not find an exact match is not proof of priority.

| Source | What it prevents us from claiming | How it affects this memo | Inspection level |
|---|---|---|---|
| [On the Guidance of Flow Matching, Feng et al.](https://proceedings.mlr.press/v267/feng25s.html) | General flow guidance or covariance-aware guidance as new | Preserve the distinction between a likelihood-score construction and a direct control edit | Primary publication page and paper method material |
| [Applying Guidance in a Limited Interval, Kynkäänniemi et al.](https://arxiv.org/abs/2404.07724) | A time window alone as a new guidance mechanism | Fixed switches and interval controls are mandatory | Primary abstract |
| [MGD: Moment Guided Diffusion](https://arxiv.org/abs/2602.17211) | Guiding moments toward prescribed values as new | RATV needs an incremental response-allocation claim | Primary abstract and HTML available; no full equivalence audit |
| [Variance-Tilted Diffusion](https://arxiv.org/abs/2606.22239) | Variance-dependent interacting guidance as new | Distinguish finite target contraction from that paper's variance-weighted batch target | Primary abstract and target/method sections |
| [TOCFlow](https://arxiv.org/abs/2601.09474) | Terminal constraints, geometric control, or global statistical constraints as new | Full-forecast and geometry-matched baselines matter | Primary abstract and formulation material; no full equivalence audit |
| [Jeffrey guidance](https://arxiv.org/abs/2606.13240) | Density-ratio marginal shaping as a new third proposal | A simple target-density/base-density recipe was deliberately not counted as an innovation | Primary abstract and density-ratio method section |
| [Histogram-constrained Image Generation](https://arxiv.org/abs/2606.31683) | Histogram matching or OT transformations during diffusion as new | RBAG's possible increment is the nonlinear terminal response cost and feasibility allocation | Primary explicit-transformation and OT sections |
| [Multilevel and Sequential Monte Carlo for Training-Free Diffusion Guidance](https://arxiv.org/abs/2601.21104) | Richer posterior simulation or multilevel estimation as a fresh general idea | More MC effort is a serious compute-matched comparator | Primary abstract |
| [Harmless Diversity](https://proceedings.mlr.press/v162/gong22b.html) | Adding diversity while protecting a main objective as new | Tangent and constrained-diversity controls must be credited and tested as baselines | Primary publication abstract; supplementary revision search |
| [Particle Guidance](https://arxiv.org/abs/2310.13102) | Interacting particles for diverse diffusion sampling as new | Provides a structural-diversity comparator, including molecular conformer precedent | Primary abstract; supplementary revision search |

No candidate has a novelty guarantee. In particular, a full equation-by-equation comparison with MGD/TOCFlow is still needed for RATV, and with HIG for RBAG. OSHG had no exact guidance-specific match surfaced by the bounded order/bracket searches, but its components are classical and its incremental value remains untested.

Compared with the earlier project ideas:

| Existing idea | Why the new proposals are different | Shared machinery that must not be re-claimed |
|---|---|---|
| QPMG / SMG2 | RATV controls ensemble outcomes; it does not approximate a nonlinear posterior likelihood with a Taylor closure | Terminal/property derivatives and local approximations |
| DBFG | RBAG assigns target values; it does not estimate flux across atom-type decoder facets | Need for decoded evaluation |
| MAG | OSHG measures paired interventions; it does not create a moment-matched posterior ambiguity set | Conservative abstention alone is not novel |
| Closure-residual escalation | OSHG tests order and terminal utility rather than merely comparing closure estimates | Adaptive switching and extra evaluations |
| Component D / TRE | RATV and RBAG control mean/spread or slots; they do not maximize a diversity direction within a per-sample property envelope | Terminal responses, local constraints, and trust regions |
| BPS | RBAG edits individual trajectories rather than thinning a pool with preserved marginals | Batch-level bookkeeping |

## 9. Empirical program: make the hypotheses falsifiable

### 9.1 First repair the measurements

1. **Add sample-level result records.** Save seed, batch, target, atom count, finite flag, decoded validity, canonical graph identifier, soft and decoded f_A/f_B values, and actual compute. Keep all attempts, including rejected or failed candidates, in the denominator.
2. **Run matched unguided controls** with the same masks, seeds, checkpoint, and evaluation convention as each comparison. A larger historical base benchmark is useful context but cannot substitute for a paired cell.
3. **Separate calibration from final claims.** Keep generator/f_A on train_a and f_B on train_b. Use validation for response calibration and fixed settings; use fresh generation seeds and the established test protocol for final reporting. The existing q50 sweep has already informed hypotheses and is not a confirmatory test.
4. **Repair the closure diagnostic** by matching covariance within each comparison. Test on states reached by the candidate guided prefixes as well as corrupted real molecules; their distributions need not match.
5. **Sweep effective strength and MC width responsibly.** For plug-in, deduplicate identical w/s² settings. For MC, examine {delta, 2delta, y_std/4, y_std} as a starting design, with guide-side tolerances and equal declared tuning budgets. Monitor ESS and increase K only in a counted comparison. A narrow likelihood with K=4 may simply select one accidental perturbation.
6. **Complete q90 and include a directional stress test.** Keep the current numerical q90 targets fixed. A q10 extension is useful only when predeclared; do not silently expand the current sweep and its completion denominator.

The current CLI does not expose s, and its result filenames omit s, clipping, MC configuration, checkpoint hash, and code version. Before a width or handoff sweep, use a separate output namespace and configuration hashes. Otherwise resumability can silently reuse a result from a different experiment. This memo does not modify the running sweep or launch new cluster jobs.

### 9.2 Cheap mechanism gates before expensive grids

| Gate | Minimal design | Evidence needed to advance |
|---|---|---|
| Terminal-response calibration | Saved states at 0.50, 0.65, 0.75; small positive/negative edits; full continuation; several atom-count strata | Predicted property changes track actual changes and improve over a cheap forecast baseline |
| RATV preservation gate | Matched batches under common translation, least-cost centering, shape-regularized centering, and normalized MSE | Better bias/coverage with retained central width and structural diversity; response-aware costs add value beyond the simpler controls |
| Fixed handoff gate | A0, 0B, AA, BB, AB, BA, 00 and blend; branch common prefixes; include center/repair objectives | At least one AB policy improves on its best pure arm at comparable quality, retained diversity, and cost |
| OSHG mechanism gate | Equal-exposure shadow pulses; different h; independent pilot/validation batches | Order signal predicts handoff gains beyond time, residual, clipping, and direct lookahead |
| RBAG heterogeneity gate | Same slots under random, sorted, and response-aware assignment; uniform versus least-distorted slot design | Response costs predict edit savings, and more elaborate slot design preserves useful width beyond the uniform baseline |
| Decoding gate | Soft versus decoded scores and decoded joint success | Improvement survives the actual output representation |

A workable small start is 64–128 trajectories per diagnostic batch, several independent batches, and three anchor times, followed by 512-attempt screening only for mechanisms that pass. These sizes are proposed feasibility budgets, not power guarantees. Do not launch the full cross-product of three methods, all switches, all widths, all targets, and all solvers before these gates.

### 9.3 Primary outcomes and statistical units

Retain MAE for comparison with prior tables, but the principal practical outcome should be

\[
U_\delta(y)=
\frac{\#\{\text{distinct valid decoded graphs whose evaluated property is in band}\}}
{\#\{\text{all attempted final outputs}\}}.
\]

Count a graph once per fixed target and evaluation batch according to a predeclared canonicalization rule. If multiple conformations share a graph, specify whether a graph is counted when any conformer succeeds; also report per-conformer success so that multiplicity cannot hide effort. Property values are evaluator predictions, not automatically physical ground truth.

Report U per attempt and successful distinct graphs per total GPU-hour. Include preview branches, failed attempts, calibration compute, and any rescoring in the appropriate accounting. Report both amortized inference cost and one-off method-development/calibration cost.

Additional outcomes: decoded property MAE, mean error, variance, tail quantiles, in-band rate, valid-only versions of those statistics, atom and molecule stability, uniqueness, atom-count distribution, and structural diversity within the band. Never use smaller property variance itself as the diversity metric.

**Revision-specific outcomes:** report |bias|/delta; variance-retention ratio to the fixed reference; central-width retention; within-band width; full centered quantile distortion; worst-tail behavior; and prespecified subgroup biases. Present the coverage-versus-preserved-diversity frontier. Report both overall property variance and useful central/structural spread so neither interpretation of “keep the spread” is hidden. An arm that keeps variance through more extreme failures fails the revised objective even if its mean and coverage improve.

RATV and RBAG couple the particles within a batch. **The independent unit is the independently generated batch**, not each molecule inside it. Use paired differences for shared-seed comparisons and a batch/bootstrap or seed-level interval. A binomial standard error treating all coupled particles as independent is unjustified. Distinct-graph yield is also a nonlinear batch statistic.

Do not extract uncertainty intervals or a distinct-yield frontier from the existing aggregate JSONs: the necessary sample-level information is absent. A descriptive MAE/stability Pareto scatter is possible, but it would remain a selected, one-seed screening picture.

### 9.4 Predeclare success and failure

Before new data, choose a useful-effect threshold and a quality noninferiority margin. For example, **as a design choice rather than a scientifically universal cutoff**, require a positive lower confidence bound on paired useful-yield improvement, point improvement of at least 1 percentage point, and no more than 2 percentage points loss in molecule stability against the relevant tuned baseline. Also require useful yield per GPU-hour not to worsen for an efficiency claim. Adjust these margins before running experiments if the available compute cannot resolve them; do not change them after seeing results.

For this revision, those conditions are insufficient without preservation criteria. Predeclare a structural-diversity noninferiority margin at matched successful sample count, a feasible central-width floor, a cumulative distortion budget relative to the fixed reference, and a tail safeguard. If literal overall-variance retention is the chosen goal, add its tolerance explicitly and check Section 3.6's feasibility limit first. No particular retention percentage is claimed to be universally appropriate; select it before the confirmatory run and show the resulting trade-off curve.

Use a separate screening set to select a small number of configurations. Final comparisons should include the selected strongest baseline, not merely the baseline that is easiest to beat. When many targets or variants are tested, identify a primary comparison and account for multiplicity or report all intervals as exploratory.

The project's planned 4,096-output stage is a reasonable starting convention, not a guarantee of statistical power. Estimate between-batch variance and paired effect variability in the pilot before deciding how many independent batches are needed. Report at least multiple independent seeds for any headline claim; one large seed cannot establish seed robustness.

### 9.5 Cost and integration details that affect conclusions

The existing cost measurements are approximately 95.0 ms/molecule for plug-in, 125.5 for TFG-MC, 271.2 for LGD-MC, 273.3 for OSC, 421.5 for SMG, 802.7 for SMG2, and 836.4 for SMG2-curv at their documented settings. They are historical measurements, not predicted costs for these proposals.

For reference forecasts, budget roughly an additional suffix rollout per refresh plus the associated differentiation cost. For handoff lookahead, budget every shadow suffix. Sharing a prefix can reduce repeated work, but does not make branches free. Record actual device, batch size, wall time, peak memory, forward calls, and derivative calls.

For Euler, switch using the actual field-evaluation time. For Heun, split an integration step at a discontinuous handoff and use one declared regime for each substep; otherwise a stage evaluated at the boundary can unintentionally blend the fields. Report the number of field evaluations rather than assuming that 100 Heun steps cost the same as 100 Euler steps.

The current VP sampler has no matching `t_min_guide` behavior: the runner only passes that window to `FlowSampler`. A label in a diffusion filename would therefore not establish a real guidance window. Implement and verify an explicit VP schedule before transferring the switch experiment. Flow's 0.75 is not automatically a physically corresponding diffusion time; compare noise level or log-SNR under the declared schedules.

For stochastic continuations, reuse noise within paired comparisons and average enough continuation draws to estimate expected response. The deterministic forecast derivative, zero conditional continuation variance, and bracket tests above cannot simply be reinterpreted as stochastic results without that additional analysis.

## 10. Self-audit: what was actually checked

The audit uses Python's standard library and requires no model inference. It checks identities and counterexamples, not molecular efficacy.

| Check | Executed result | Scope |
|---|---|---|
| Screening census and reconstruction | 141 cells, 72,192 attempts; 99 match the current 294-cell plan | Existing local files only |
| Moment allocation constraints | 1,000 random eight-trajectory problems; maximum residual **1.34×10⁻¹³** | Positive mobilities, uncapped feasible linear problem |
| Moment allocation stationarity | Maximum residual **1.78×10⁻¹⁵** | Confirms the stated quadratic optimum with equality constraints |
| Reduction to ordinary contraction | Maximum error **1.11×10⁻¹⁶** | Explicitly verifies a case where RATV adds no new mechanism |
| Time-dependent order formula | Maximum tested error **1.11×10⁻¹³** in the h²-scaled bracket | A deliberately simple smooth affine example; not a molecular convergence study |
| Commuting fields | Order discrepancy **5.55×10⁻¹⁷** | Required null behavior |
| Heterogeneous assignment example | Costs 2.2725 sorted versus 0.3125 reversed, omitting 1/2 | Exhaustive two-item local-cost comparison |
| Homogeneous assignment control | Sorted 4.5 versus reversed 6.5 | Confirms the standard sorted case |
| Uniform-slot variance | Formula error **2.78×10⁻¹⁷** for B=128, a=0.8 | Finite slot law only |
| Covariance-mismatch counterexample | Apparent error 49.5; matched Taylor error zero | Demonstrates a confound in the existing closure diagnostic |
| Variance delta² versus coverage | Centered Gaussian in-band probability **0.68268949** | Refutes a 95%-coverage interpretation |
| Endpoint conditional versus population variance | Local variance zero, ensemble variance four in a two-point example | Refutes identifying local uncertainty with generated spread |

Reproduce from the project root:

```powershell
python audit/guidance_variance_switching_2026/audit_results_and_math.py
```

The script rewrites only its audit outputs. If new result cells arrive later, its census and reconstructed tables may change; the recorded source hashes identify the inputs used. The numeric tables in this memo describe the snapshot audited on the date above.

**Not tested here:** constrained-QP implementation, nonlinear line-search convergence, molecular symmetry of a new controller, terminal-response reliability, actual mixed schedules, decoded useful yield, q90 performance, extra GPU runtime, or superiority of any proposal. Those are explicitly assigned to the empirical program rather than implied by toy arithmetic.

### 10.1 Additional checks for the bias-and-spread revision

The separate [revision check script](../../audit/guidance_variance_switching_2026/check_bias_and_spread_revision.py) and [its results](../../audit/guidance_variance_switching_2026/bias_spread_revision_checks.json) add:

| Check | Executed result | What it establishes |
|---|---|---|
| Common translation, 1,000 random batches | Maximum variance error 2.66×10⁻¹⁵; pairwise-difference error 8.88×10⁻¹⁶ | Property translation preserves centered scalar shape arithmetically |
| Moment-preserving smooth-coverage direction, 1,000 random batches | Maximum moment-rate residual 1.36×10⁻¹⁶; no negative coverage directional derivative | The stated local constrained-gradient identity, in the tested uncapped cases |
| Increasing shape penalty | Increment variance 0.03480 → approximately 7.18×10⁻¹² | The example approaches common increments while retaining requested moment rates |
| First-order preservation versus finite step | Predicted variance change 0; actual change 0.02 | Local preservation is insufficient for a finite intervention |
| Same variance with better coverage through distant tails | Variance stays 1; coverage rises 0 → 0.9; maximum residual rises 1 → sqrt(10) | A necessary adverse control against misleading variance retention |
| Centering a skewed law | Bias 1 → 0; variance stays 9; coverage falls 0.9 → 0 | Bias correction must be coverage-checked |
| Subgroup centering | Overall variance 14/3 → 2/3, within-group variance unchanged at 2/3 | Reduced aggregate spread need not erase within-group differences |
| Bounded-tail feasibility | V=2, delta=1, R=3 imply coverage ≤0.875 | Literal spread preservation can rule out a requested 0.9 coverage |

Run `python audit/guidance_variance_switching_2026/check_bias_and_spread_revision.py` from the project root. The Gaussian illustration and tolerance-normalized bias table are also reproduced there. These checks do not implement the new constrained controller or slot-design optimizer and do not constitute molecular evidence.

## 11. How to use this memo to develop the next contribution

The strongest starting question is now: **can we correct the measured bias while preserving the centered property shape and structural diversity, and how much additional shape change is actually necessary to raise band coverage?** Test that against common translation, normalized MSE, and established diversity controls before attributing any result to a new mechanism.

Your 0.05–0.75 then switch design should first be tested with branched suffix controls and explicit centering/coverage objectives. Compare a fixed handoff with one triggered by remaining bias, coverage, and preservation budget. A useful switch should improve how many valid molecules enter the band while retaining the distinctions between successful molecules; a lower scalar MAE alone does not establish that.

RBAG becomes attractive when the old property shape cannot fit in the band. Its revised question is how to choose and realize the **least distorted feasible spread** inside that band, with a nonzero central-width floor and measured structural diversity. Response-aware allocation matters only if it improves on simpler assignment controls.

The difficult case should be reported plainly: **if the user requires the original very large overall property variance, high band coverage, and no extreme tails simultaneously, those requirements may be incompatible.** The useful alternative is to preserve the part of spread that represents meaningful diversity, measure the minimum relaxation required, and report the trade-off instead of hiding it in outliers or invalid samples.

For each candidate, a defensible eventual paper claim needs all three parts: **a precisely stated mechanism beyond its closest simple control, a measured failure mode it addresses, and a decoded, compute-aware improvement on fresh experiments.** Until then, the formulas are checked proposals and the molecular benefits remain hypotheses.
