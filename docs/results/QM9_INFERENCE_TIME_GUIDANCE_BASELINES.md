# QM9 inference-time guidance baselines: research and implementation handoff

**Research checked 19 September 2026, America/New_York (20 September UTC).**

**Purpose:** compare recent guidance algorithms using our own frozen QM9 generator, with no new generator training for individual baselines. This is a research/implementation specification, not a claim that the new baselines have already been implemented or benchmarked.

## 0. Read this first: the decision

**Yes, there are enough suitable recent methods. My recommendation is to implement LGD-MC, D-Flow, and OC-Flow first, and add FlowGrad as the fourth external comparator if flow matching is selected.** All have publications in 2023–2025. This gives a close approximation-based competitor to SMG, source-noise optimization, and trajectory-control optimization. FlowGrad provides a further published ODE method and an informative unregularized control comparison.

For the paper, the intended FM table is **unguided + plug-in + LGD-MC + D-Flow + OC-Flow + FlowGrad + ours**. The first two are internal references; the next four are external methods. If resources allow only three external methods, use **LGD-MC + D-Flow + OC-Flow**, explicitly labeling LGD-MC as adapted to the independent Gaussian FM path. If avoiding a diffusion-to-FM adaptation is the overriding concern, **D-Flow + OC-Flow + FlowGrad** are the three native ODE choices; still retain LGD-MC as a low-cost scientific control for SMG.

If **VP diffusion** is selected instead, the natural main suite is **full TFG + LGD-MC + FreeDoM**, with **DPS** as a fourth, inexpensive reference. These can share our VP checkpoint; full TFG/FreeDoM require a discrete sampling driver, not a generator retrain. D-Flow can also run on a differentiable probability-flow ODE.

**We do not yet have full TFG in the local sampler.** `proj1/src/guidance.py:tfg_mc_weighted_grad` implements isotropic Monte Carlo likelihood smoothing and a denoiser pullback. It lacks TFG's clean-state refinement, recurrence, and discrete update. Its present behavior is closely aligned with the **LGD-MC estimator**, which predates TFG. Do not count the same function under both names as two external methods. TFG and **TFG-Flow** are also different papers.

### What is fixed, and what changes

| Fixed across the controlled comparison | Allowed to change by method |
|---|---|
| Generator checkpoint and EMA choice | How guidance is computed and applied |
| Guide `f_A`, evaluator `f_B`, units, decoder | Monte Carlo sample count, control variables, optimizer |
| Test target/atom-count pairs and starting noise | Method-specific schedules and sampling policy |
| Metric definitions and sample-attempt accounting | Inference compute, which must be measured |
| Generator parameters remain frozen | Per-sample latent/control tensors can be optimized |

“Training-free” here means no extra generative-model training or task-specific noisy-guide training for a baseline. A shared, already-trained clean property predictor is still required. Optimizing a sample's initial noise at inference time is not retraining the generator.

### What the assignment actually requires

The assignment PDF, pages 1 and 3, requires at least three relevant methods published within five years. The paper instructions explicitly say the two trained base models do not replace the external methods, require the strongest internal reference in addition, and require disclosure of changes in checkpoints/data/compute:

- [Assignment](CIS_6270_Fall_2026_Project1_Assignment.pdf), pages 1, 3, and 6.
- [Paper instructions](proj1_tex/Project1_Paper_Instructions.tex), particularly lines 321 and 394.

A shared-backbone comparison is a defensible **comparison of guidance algorithms**, particularly because guidance is our proposed contribution. It is not a reproduction of every original paper's complete generative system or evidence that our backbone beats their backbones. This is an interpretation of the assignment, not an instructor's explicit ruling. The same three-baseline requirement remains for Modality 2; this document addresses QM9 only.

## 1. Evidence and candidate ranking

All selected publication years are safely inside the September 2021–September 2026 window. Publication venue/year, rather than an old arXiv submission date alone, determines the labels below.

| Method | Publication | Same frozen FM checkpoint? | Same frozen VP checkpoint? | QM9 evidence | Recommendation |
|---|---|---|---|---|---|
| **LGD-MC** | ICML 2023 [S1] | Yes, with Gaussian-path score-to-velocity conversion | Yes | Included in TFG's later molecular benchmark; original paper is not a QM9 paper | First implementation; closest practical MC comparator |
| **D-Flow** | ICML 2024 [S2] | Yes | Yes, using deterministic ODE | Original paper §6.3 and appendix B.3 | Core baseline |
| **OC-Flow** | ICLR 2025 [S3] | Yes | Possible ODE adaptation; not first priority | Original paper §5.2, appendix E.2; released molecular code | Core baseline; distinguish paper and released implementation |
| **FlowGrad** | CVPR 2023 [S4] | Yes | Possible deterministic ODE adaptation | Original experiments are images; OC-Flow later evaluates a molecular adaptation | Fourth FM comparator / native ODE backup |
| **TFG** | NeurIPS 2024 [S5] | Partial mechanisms transfer; a full port needs additional choices | Yes, port discrete sampler to our schedule | Original paper appendix D.7 | Core if diffusion wins; do not rename current MC component “full TFG” |
| **FreeDoM** | ICCV 2023 [S6] | Not a direct deterministic FM port | Yes | TFG's later molecular benchmark | Third diffusion comparator |
| **DPS** | ICLR 2023 [S7] | Plug-in score approximation transfers | Yes | TFG's later molecular benchmark | Cheap external reference; distinguish practical normalization |
| **On the Guidance of Flow Matching** | ICML 2025 [S8] | Yes for relevant training-free variants | Some special cases | Original evaluations are not QM9 | Essential theory/reference; backup, not another name for plug-in |
| **TMPD** | TMLR 2024 [S9] | Gaussian-path adaptation possible | Yes for its observation setting | No verified original QM9 experiment | Mandatory related work for SMG; nonlinear extension is not a literal reproduction |
| **TFG-Flow** | ICLR 2025 [S10] | Not a faithful drop-in for our continuous atom-type path | No direct port | Original molecular experiments | Different discrete/continuous generator interface; defer |

The alternatives were selected for frozen-model compatibility, scientific relevance, implementation availability, and publication status, rather than merely using papers whose titles contain “guidance.” Source links and pinned code are in §12.

## 2. Local code audit: what Claude should assume

The working source is **`proj1/`**. `transfer/` and `_ship/` contain copies/packages; do not independently patch them and assume the active implementation changed.

| Local evidence | Consequence |
|---|---|
| `scripts/train_fm.py` trains `x_t = t*x_clean + (1-t)*eps` with independent Gaussian noise | The independent affine Gaussian formulas below apply to this path; not automatically to OT-paired noise/data |
| `src/sampling.py:FlowSampler` samples `t: 0 -> 1` | Do not copy reverse-time signs from EquiFM/OC-Flow wrappers |
| `src/diffusion.py` is continuous VP, epsilon prediction | Our checkpoint is not the published E(3)-EDM checkpoint or schedule |
| `src/sampling.py:VPSampler` integrates `tau: 1 -> 0.001`, then denoises | Its final denoising map must be included in end-to-end gradients |
| `src/guidance.py` exposes `plug`, `smg`, `smg_mean`, `smg_var`, `tfg_mc` | Only the MC component of TFG exists; there are no D-Flow/OC-Flow/FlowGrad drivers here |
| Base-field generator calls in `FlowSampler.field` and `VPSampler.field` are inside `torch.no_grad()` | These methods cannot be reused unchanged for differentiating through an entire trajectory |
| `tfg_mc` uses constant `sigma_mc`, default `0.1`; coordinate probe noise is masked but not centered | Add an explicit schedule and zero-CoM probes; do not claim the current noise is the calibrated posterior covariance |
| `Cost` already separates forward/VJP/JVP and guide calls | Extend it for optimizer closures, resampling, and batch/sample counts |
| `scripts/s2_harness.py` samples unguided and guided arms sequentially from one RNG | The arms are not automatically paired on identical initial noise; build a saved noise bank |
| `scripts/prepare_qm9.py` explicitly saves a random split, not the official Cormorant split | Published molecular numbers are context only; they are not directly comparable to our reproduced scores |
| `src/evaluation.py` decodes atom types for bonds but passes continuous `feats` to `f_A`/`f_B` | Final molecular property metrics must evaluate decoded one-hot types; retain soft-feature scores only as diagnostics |
| `choose_delta()` uses only evaluator MAE | This differs from the newer predictor planning document; settle and record one evaluation-tolerance policy |

The existing FM diagnosis document reports weak molecule stability for an earlier checkpoint. That is historical project evidence, not a fresh measurement in this research pass. Before expensive optimization, measure unguided quality of the **actual selected checkpoint**. A guidance sweep cannot establish a useful molecular method if essentially no valid molecules survive.

Do not inherit two overstrong claims from older project notes: errors from a shared evaluator do **not** necessarily cancel between methods, and training guide/evaluator on disjoint halves does **not** make their errors independent. Different methods can exploit different predictor errors. Keep evaluator independence, decoded scoring, and a second evaluator/physics subset where available.

## 3. Common mathematical and implementation contract

### 3.1 State, property objective, and units

Write the padded state as `z = (r, h)`, with coordinates `r[B,N,3]`, continuous type features `h[B,N,5]`, and a fixed mask per molecule. Let `P` mask features and remove the coordinate center of mass over active atoms. Atom counts are sampled once and held fixed throughout a trajectory.

Here “zero CoM” follows the repository's convention: subtract the **arithmetic coordinate mean**, not a mass-weighted physical center. Use that same linear projector throughout.

Use a common guide energy in the main controlled study:

\[
E(z;y)=\frac{(f_A(z)-y)^2}{2s_y^2},\qquad R(z;y)=-E(z;y).
\]

`f_A` and `y` must use the same physical units (dipole: Debye). `s_y > 0` is a reward width; it is not the evaluation success tolerance `delta`. In the existing harness the default width is the training property standard deviation; preserve this as a documented starting point, rather than suddenly substituting a narrow success band.

A useful implementation convention is a **per-molecule energy**, summed to obtain independent gradients. Using a batch mean changes gradient scale with batch size. For L-BFGS, a single flattened batch also shares line-search decisions, so batch size is an algorithmic setting; start with batch size 1 for parity.

Keep continuous atom features while taking gradients. Do not put `argmax`, RDKit sanitization, or integer atom embedding lookups on the guidance path. At final evaluation, decode features once and score the resulting discrete molecular representation. A soft embedding adapter for a pretrained guide must reproduce the original predictor at one-hot inputs and have verified input derivatives.

### 3.2 Endpoint mean versus actual terminal sample

For our FM path, with velocity `v_theta`,

\[
m_t(z)=z+(1-t)v_\theta(z,t),\quad J_t=\partial m_t/\partial z.
\]

This is a learned conditional-mean estimate, **not** the result of integrating the ODE from `t` to 1. LGD-MC and plug-in use the estimate. D-Flow, FlowGrad, and OC-Flow differentiate an actual numerical rollout. Replacing their rollout by this one-step estimate changes their core mechanism.

For VP, using amplitude `a(tau)` and standard deviation `b(tau)`,

\[
m_\tau(z)=\frac{z-b(\tau)\epsilon_\theta(z,\tau)}{a(\tau)}.
\]

Never confuse `a` with DDIM's cumulative variance coefficient `alpha_bar = a^2`.

### 3.3 A guidance score is not a velocity or a discrete displacement

For an approximate conditional score `G = grad_z log Z(z)`, the current FM path uses

\[
v_{\rm guided}(z,t)=v_\theta(z,t)+w\frac{1-t}{t}G(z,t).
\]

The independent affine Gaussian path is the assumption behind this conversion [S8]. For the exact tilted endpoint mean, `Delta v = (m_tilt-m)/(1-t)` and the Gaussian-channel score identity gives the same factor. This does not make an approximate `G` exact posterior guidance.

Do not evaluate this formula at `t=0`. Keep an explicit onset such as `t_min_guide`, chosen on validation. For VP probability-flow sampling,

\[
F(z,\tau)=-\tfrac12\beta(\tau)z-\tfrac12\beta(\tau)(s_\theta(z,\tau)+wG),
\]

with a **negative integration step**. A positive ascent score therefore moves samples toward the reward during reverse integration. The reverse SDE has a different coefficient; do not reuse the ODE's one-half factor in an ancestral sampler.

TFG's `Delta_t`, FlowGrad's state kicks, and OC-Flow's controls enter different equations. Do **not** multiply all of them by `(1-t)/t` merely because they are called guidance.

### 3.4 Geometry and numerical policy

- Freeze network parameters with `requires_grad_(False)` and `eval()`, but keep autograd enabled for inputs on guidance/rollout paths.
- Apply mask and coordinate projection to noise, gradients, controls, and the state. Do not center atom-feature channels as a routine symmetry projection.
- Work in the exact coordinate/feature scaling of the selected checkpoint. A Euclidean norm mixing Å and feature channels is a modeling choice; record the scaling.
- Use identical per-step clipping/onset for local-score comparisons where appropriate, and also disclose any change from a paper's native settings. Do not clip a whole optimized trajectory after the fact without differentiating through that policy.
- Do not backpropagate through stochastic probe resampling. Draw fixed reparameterized probes for an objective evaluation. Fix probes across the two Heun stages if claiming a deterministic-field solver comparison; separate that from a method using fresh probes at each stage.

## 4. LGD-MC: first implementation, and the closest practical comparator

**Published mechanism.** Song et al.'s ICML 2023 paper, equation (9), replaces the point estimate with Monte Carlo samples from an approximate clean posterior and differentiates the logarithm of their mean likelihood [S1]. The estimator is directly relevant to SMG's treatment of nonlinear properties.

For our adapter, define

\[
z_k=m_t(z)+r_t P\epsilon_k,\quad
\ell_k=-E(z_k;y),\quad
L(z)=\operatorname{logsumexp}_k\ell_k-\log K.
\]

With `r_t` independent of the state,

\[
G_{\rm LGD}=J_t^\top\sum_k\omega_k
\frac{y-f_A(z_k)}{s_y^2}\nabla f_A(z_k),\qquad
\omega_k=\operatorname{softmax}_k(\ell_k).
\]

This is **not** the gradient of the mean energy and is **not** an unweighted average of property gradients. The local `tfg_mc_weighted_grad` is a useful starting point for this calculation.

**Our port specification (not an original QM9 recipe from LGD).**

1. Add a canonical `lgd_mc` name; retain `tfg_mc` only as a deprecated alias for old run records.
2. Project coordinate probes to zero CoM and mask feature probes. Reuse one endpoint mean and one generator pullback.
3. Make `K`, coordinate/feature spread, and schedule explicit. Begin with `K=4`, then compare `K=16` under the compute budget. A `K=1` noisy sample does not equal plug-in unless the spread is zero.
4. Candidate spread schedule: for path `z=a*x+b*eps`, use `r_t = c*b/sqrt(a^2+b^2)`, with `c` tuned on validation. This is the posterior standard deviation for a unit-Gaussian prior, used only as a **heuristic** for molecular data. It is `c*b` for VP, and `c*(1-t)/sqrt(t^2+(1-t)^2)` for our FM. Also retain the old constant-spread arm as a diagnostic if useful.
5. Convert `G` to a sampler velocity using §3.3. Label the FM result **LGD-MC, Gaussian-FM adaptation**. This transfer is supported by [S8], but is not an original LGD QM9 experiment.

**Code audit warning.** The TFG authors' *reimplementation* of LGD, `methods/lgd.py`, computes `grad(logsumexp(logp)) / K`. The derivative of `log(mean(exp(logp)))` has **no additional `1/K`**: the `-log K` is constant. Implement the paper estimator above; if comparing exact repository behavior, expose the extra scaling as a named compatibility option. Without nonlinear clipping, it can be absorbed into strength for a fixed `K`; a fixed clipping threshold can also change that equivalence. Either way, it confounds an unadjusted sample-count sweep. This is not evidence that the LGD authors' own implementation has the same discrepancy.

**Cost and diagnostics.** One denoiser pullback plus `K` guide evaluations/derivatives per field evaluation, with possible batching. Log effective sample size `ESS = 1/sum(omega^2)`; `ESS≈1` means one probe dominates. Increasing `K` reduces Monte Carlo error for the chosen approximate posterior; it does not remove posterior-approximation bias.

That is an efficient target implementation, not the current call total. The local `guidance_field` computes the endpoint and `property_derivs` before branching to `tfg_mc`, then recomputes the endpoint for the pullback; the sampler also evaluated the generator for its base field. Count those extra calls if retained. Remove redundant work consistently before making efficiency claims.

**Acceptance checks:** finite-difference agreement with `L`; invariance to duplicating every probe; zero spread equals plug-in for every `K`; gradients of padded atoms vanish; rotation/permutation tests transform the probes along with the molecule.

## 5. D-Flow: optimize the initial noise through the frozen generator

**Published mechanism and relevance.** D-Flow optimizes the source point by differentiating the terminal objective through a numerical ODE solve. Its original QM9 experiment includes dipole targeting. The molecular appendix uses midpoint integration with 100 forward evaluations per rollout and L-BFGS with line search, five outer steps and up to five inner iterations [S2].

Let `Phi_theta(z0)` denote the entire deterministic sampler, including the terminal denoising map if one exists:

\[
\min_{z_0} E(\Phi_\theta(z_0);y),\qquad
\nabla_{z_0}E=D\Phi_\theta(z_0)^\top\nabla E.
\]

**Our implementation plan.**

```text
freeze generator and guide weights
z0 = saved initial noise, represented in the masked/zero-CoM subspace
define closure:
    clear z0 gradient
    z1 = differentiable_rollout(z0, frozen_generator, solver, time_grid)
    loss = sum(per_molecule_energy(z1))
    compute gradient w.r.t. z0
    count EVERY closure and its forwards/backwards
    return loss
run L-BFGS with strong-Wolfe line search
evaluate the actual accepted iterate, retain best finite guide-loss iterate
perform the final rollout and evaluate decoded molecules with f_B
```

**Do not use the existing `FlowSampler.field()` in this rollout:** its `no_grad` generator call breaks the required derivative. Add a differentiable **unguided** field and a differentiable terminal map. Discrete backpropagation through a fixed-step solver is a good correctness reference; checkpointing can reduce activation memory while retaining the same discrete gradient. Record recomputation cost.

Start with the paper's L-BFGS configuration, not an arbitrary Adam/SGD replacement. The paper also describes feature-wise normalization of source noise after optimization steps. Our molecule representation has different channels and padding, so make the normalization scope explicit: active atoms only; retain coordinate zero CoM; guard small standard deviations. Record whether the paper-inspired source normalization is used. An unnormalized or SGD arm is a labeled variant, not a silent reproduction.

Use two solver views when feasible: the shared solver/grid for controlled comparison, and a midpoint setting matching the paper's stated rollout budget. Do not count “100 NFE” as total D-Flow cost; that is one solve inside many optimizer closures.

**Source availability.** I did not verify a released original D-Flow implementation in the sources inspected. The pinned OC-Flow repository has `dflow_opt`, which is an **OC-Flow-author reimplementation**, uses Euler, and does not visibly implement all the original appendix choices. Use it for orientation and verify against [S2]. Do not describe it as official D-Flow code.

**Failure modes:** memory from the differentiable rollout; variable L-BFGS line-search cost; source optimization overfitting `f_A`; valid-only MAE improving because most other attempts fail. It is an optimization method, not a guarantee of independent samples from the exact reward-tilted distribution.

**Acceptance checks:** terminal loss finite differences w.r.t. source noise; zero optimization gives the corresponding unguided sampler exactly; generator/guide weights remain unchanged; final selected iterate's recorded loss matches a fresh evaluation; hard-decoded property and validity reported.

## 6. OC-Flow: optimize controls along the trajectory

**Published mechanism.** OC-Flow formulates steering as optimal control with terminal reward and a penalty on control energy. It has original QM9 experiments and publicly released molecular code. Its paper gives an iterative control update and discusses relationships to D-Flow and FlowGrad [S3].

The paper-level continuous form is

\[
\dot z=v_\theta(z,t)+u(t),\qquad
\min_u E(z_1;y)+\frac{\lambda}{2}\int_0^1\|u(t)\|^2dt.
\]

**Important choice: reproduce a named released variant.** The released molecular `oc_opt_du` directly optimizes a discretized objective with SGD or L-BFGS and includes an additional initial displacement. It is not identical to the paper's E-MSA update. For this project, start with **OC-Flow, released QM9 objective port**, using direct autograd, and disclose the variant. Do not attach the continuous theorem's guarantees to this finite, clipped, nonconvex implementation.

For a forward-time Euler port, make the objective explicit:

\[
z_0=z_{\rm noise}+b_0,\quad
z_{i+1}=P[z_i+h_i(v_\theta(z_i,t_i)+u_i)],\quad
\mathcal J=\sum_b E(z_{N,b};y_b)+\frac\lambda2\sum_i h_i\|u_i\|^2.
\]

The official molecular implementation's penalty averages over atom/batch dimensions, and its extra initial displacement is not part of the accumulated velocity penalty. Decide and record these details. A per-active-atom normalization in our padded representation changes the effective `lambda`; it must be exposed in configuration. If fixing `b0=0`, label that as the fixed-initial-state variant rather than claiming exact parity with `oc_opt_du`.

**Our implementation sequence:**

1. Implement the discretized objective using the differentiable rollout from D-Flow, with trainable **sample controls only** and frozen network parameters. Use integer step indexing, not `int(float_time * n_steps)`.
2. Start with L-BFGS settings corresponding to the released QM9 route; tune `lambda` and the iteration budget on validation. Retain a common-energy run and, if practical, an absolute-error objective sensitivity run because the upstream molecular loss uses MAE.
3. Keep control tensors in the valid state subspace and include the exact projection/terminal maps in derivatives.
4. Only after parity is established, consider a manual discrete adjoint or the separate E-MSA variant. Do not mix their updates under one method name.

For the unprojected Euler velocity-control step, an independent gradient check must recover

\[
\lambda_N=\nabla E(z_N),\quad
\lambda_i=(I+h_i Dv_i)^\top\lambda_{i+1},\quad
\nabla_{u_i}\mathcal J=h_i\lambda_{i+1}+\lambda h_i u_i.
\]

Here `lambda_i` is the adjoint and scalar `lambda` is the penalty coefficient; use different variable names in code. With projection, differentiate the projected map. Include the derivative of the VP terminal denoiser when applicable.

**Upstream discrepancy to avoid.** In `oc_opt_du_adjoint`, the forward Euler path uses `z+h*(v(z)+u)`, but the manual VJP is constructed from `z+u+v(z+u)` without the matching step-size factor. The molecular `flowgrad_opt` has a similar mismatch. These are visible code discrepancies, not a retraction of the papers. Use direct autograd of the declared map as ground truth; do not copy these routines blindly.

**Cost:** repeated full rollouts and backwards, plus control storage. More expensive than plug-in/LGD in typical settings; no new generator training. Log terminal loss and integrated control penalty separately.

**Acceptance checks:** control finite differences; exact zero-control reduction; penalty scaling under step refinement; masked/padded controls have no effect; source-shift inclusion explicit; all optimizer closures and final evaluations counted.

## 7. FlowGrad: a fourth external ODE method with a distinct control convention

**Published mechanism.** FlowGrad propagates terminal-objective gradients through a frozen generative ODE using decomposed VJPs, with optional grouping of nearly straight trajectory segments to accelerate backward computation [S4]. The original task is image manipulation. Its use on QM9 is a later adaptation in the OC-Flow paper, which should be cited as such [S3].

**Follow the original implementation's state-edit convention.** The inspected original `generate_traj` adds a state displacement before evaluating the velocity:

\[
q_i=z_i+d_i,\qquad z_{i+1}=P[q_i+h_i v_\theta(q_i,t_i)].
\]

This differs from OC-Flow's additive **velocity** control `z+h*(v(z)+u)`. With `P=I`, the exact discrete gradient w.r.t. a kick is

\[
\nabla_{d_i}E=(I+h_i Dv(q_i,t_i))^\top\lambda_{i+1}.
\]

**Our port:** initialize per-step kicks to zero, take terminal energy gradients, run the reverse VJP recurrence for the actual forward map, and update kicks by SGD. Start with all segments retained; this is the uncompressed FlowGrad variant. Add the original straightness-based grouping only after the uncompressed gradient matches autograd. Label whether compression is enabled and count retained backward segments. Do not claim the original speedup for an uncompressed port or assume our molecular trajectories are equally straight.

The original code assumes image-style time scaling (`t*999`) and batch size one in its optimization helper. Replace only the model/time/shape adapter; do not carry image scaling into our `t in [0,1]` model. A molecular port also needs masks and zero-CoM projection.

Use a separate method identifier from OC-Flow. Merely setting the OC penalty to zero can be a useful ablation, but does not reproduce every state-kick/compression detail of original FlowGrad. Likewise, the later OC-Flow authors' `flowgrad_opt` is not the original FlowGrad implementation.

**Initial tuning proposal:** SGD iteration budgets `{5,10,20}` and a small geometric learning-rate sweep after a 16-sample stability pilot. These are project settings, not published molecular optima.

**Acceptance checks:** state-kick finite differences; zero kicks exactly reproduce the reference rollout; compression disabled agrees with direct autograd; approximate compression error measured before claiming its speed benefit. As with D-Flow, watch guide exploitation and loss of diversity.

## 8. If diffusion wins: complete TFG, FreeDoM, and DPS

### 8.1 Full TFG on our VP checkpoint

**Published mechanism.** TFG combines guidance through the denoiser, iterative clean-state refinement, likelihood smoothing, and recurrence. The paper has a QM9 benchmark [S5]. A nontrivial instantiation of this full framework is meaningfully different from the current smoothing-only arm.

**Port the discrete update, not just `tilde_get_guidance`.** Use a decreasing grid `tau_n -> tau_next`; define `A = a(tau_n)^2`, `A_next = a(tau_next)^2`, and `q = A/A_next`. Then `0 < q <= 1`. For a VP-compatible DDIM base step,

\[
\sigma_\eta^2=\eta^2\frac{1-A_{\rm next}}{1-A}(1-q),\qquad
S=\sqrt{A_{\rm next}}m+\sqrt{1-A_{\rm next}-\sigma_\eta^2}\,\epsilon_\theta+\sigma_\eta P\xi.
\]

Use the **same** coordinate/feature scaling and VP amplitudes as our trained model. This equation adds a new sampler, not a new generator. Validate endpoints separately and do not clamp molecular coordinates to an image range such as `[-1,1]`.

The implementation sequence from the pinned `methods/tfg.py` is:

```text
for each noisy -> less-noisy interval:
    for each recurrence:
        m = denoiser_mean(z, current_time)
        choose fixed smoothing probes for this recurrence
        Delta_t = rho_n * grad_z(logmeanexp(-E(m + probes)))
        m_refined = detach(m)
        repeat I times:
            m_refined += mu_n * grad_m(logmeanexp(-E(m_refined + probes)))
        Delta_0 = m_refined - m
        z_next = S(z, m) + Delta_t / sqrt(q) + sqrt(A_next) * Delta_0
        if another recurrence remains:
            z = sqrt(q)*z_next + sqrt(1-q)*projected_fresh_noise
    advance with z_next
```

Mask and project every applicable term. `rho_n`, `mu_n`, smoothing spread, recurrence `R`, and clean iterations `I` are separate parameters. The official code normalizes its increase/decrease schedules over the entire grid; copying the names without their definitions is insufficient.

Start with `R=1`, `I in {1,4}`, nonzero clean refinement, then consider `R=2` if compute permits. Allow zero refinement in tuning but record when the selected TFG configuration collapses to a simpler special case. If it becomes numerically identical to LGD, do not present identical outputs under two names as evidence of two distinct comparisons; retain another distinct comparator.

**Current-code discrepancy:** at the pinned TFG revision, `scripts/molecule_property.sh` sets `eps_bsz=1`, `iter_steps=4`, `recur_steps=1`, `eta=1.0`, and labels some values as sweep placeholders. The local comment saying their molecular script uses four MC samples is not supported by this revision. Neither `K=1` nor the script's placeholder strengths should be advertised as universally optimal settings.

For a shared discrete comparison, LGD/SMG score guidance can be converted to epsilon units as `eps_guided = eps - b*w*G`, and then supplied to the same DDIM base. This is a declared sampler adaptation; keep the existing probability-flow-ODE study as a separate ablation. Alternatively evaluate every method's natural sampler and report the sampler difference. Do not silently compare an ODE and stochastic TFG as if only one scalar coefficient changed.

**Acceptance checks:** `rho=mu=0, R=1` gives the identical DDIM base with the same noise; recurrence uses the proper forward-noising ratio and projected noise; detached clean refinement does not accidentally trigger extra generator backwards; equation-level parity with the pinned code on a small synthetic example.

### 8.2 FreeDoM

**Published mechanism.** FreeDoM performs energy guidance with a repeated denoise/re-noise strategy (“time travel”) [S6]. It appears as a molecular comparator in TFG [S5]. It is useful because extra inference effort goes into revisiting states rather than Monte Carlo smoothing or global trajectory optimization.

For the VP port, use the same `S` and `q` above. Each recurrence computes `G = -grad_z E(m(z))`, sets `z_next = S + rho_n*G`, and, if needed, returns to the current noise level with `sqrt(q)*z_next + sqrt(1-q)*P*xi`. Preserve the recurrence schedule/window and strength schedule in the run record. The pinned TFG repository provides a FreeDoM **reimplementation**, not the FreeDoM authors' original code.

Run `R in {1,2,4}` under equal compute limits, with at least one genuine recurrent configuration. `R=1` with a matching energy/normalization can collapse to a plug-in/DPS-like update; do not claim recurrence was tested in that setting. A deterministic FM ODE has no automatically equivalent forward-noising transition: a new stochastic FM sampler would be an additional adaptation. Hence this is a diffusion recommendation.

### 8.3 DPS

**Published mechanism.** DPS approximates measurement guidance through a predicted clean sample [S7]. For our smooth scalar property it yields the basic pullback `-grad_z E(m(z))`, but practical published implementations also use residual-dependent normalization.

Our existing `plug` arm is a **Gaussian-energy plug-in / DPS-style** field. It is not automatically the original DPS sampler. The original pinned DPS `condition_methods.py` differentiates the residual L2 norm, while the TFG authors' DPS port divides a guidance gradient by an absolute log-likelihood. Those are different practical conventions.

Keep two names if both are used: `plugin_gaussian` for the matched-energy scientific control, and `dps_residual_norm` for a deliberately specified DPS practical port. Use per-sample residual norms, a documented epsilon at zero residual, and tests for batch independence. Do not silently take one norm over a multi-molecule batch. DPS is a useful inexpensive fourth diffusion method; it should not be the sole approximation-based competitor to SMG.

## 9. Other plausible methods: what to do with them

### TMPD: a close prior formula, but not a free extra QM9 baseline

TMPD uses denoiser-derived moment information for linear observation inverse problems [S9]. Under the independent Gaussian channel, let `Sigma = (b^2/a)*J` on the valid state subspace. For a scalar affine observation `f(z)=c^T z+d`, the covariance-frozen guidance is

\[
G=J^\top c\,\frac{y-c^Tm-d}{s_y^2+c^T\Sigma c}.
\]

This is the affine-property limit of our SMG expression. For nonlinear dipole prediction, replacing `c` by `grad f(m)` gives a **local-linear TMPD-inspired extension**, essentially the existing `smg_var` arm under the same conventions. Include it as an important ablation/related-work comparison, but do not count it as a faithful published nonlinear-QM9 TMPD implementation. Learned denoiser Jacobians need not yield symmetric positive-semidefinite covariances; log negative quadratic forms, define any stabilization, and do not describe clipped values as exact posterior covariance.

### On the Guidance of Flow Matching: theory and an optional distinct estimator

Feng et al. [S8] justify the Gaussian-path bridge used here and study several genuinely different estimators. Their classical gradient special case overlaps with our plug-in. Their Gaussian simulation estimator instead weights conditional velocities by endpoint rewards; a covariance-aware or marginal-MC implementation would be a separate project. Do not count `g_cov-G` and the algebraically equivalent plug-in as two methods. A purported `g_MC` needs the conditional/marginal sampling and density ingredients specified by that paper; simply drawing isotropic perturbations and calling them exact posterior samples is wrong.

### TFG-Flow: very relevant paper, incompatible discrete interface

Its released `ConditionalFlow.compute_next_state` uses categorical endpoint samples and mask-state transition rates for atom types, alongside continuous coordinate updates [S10]. Our model evolves five atom-type channels with Gaussian interpolation and velocity prediction. There is no categorical transition-rate head to plug in.

Options are: run the authors' released model as a **separate-system** comparison, or explicitly invent/validate a continuous relaxation. The former violates the same-backbone condition but can be an appendix reference; the latter is not a full TFG-Flow reproduction. Neither is necessary to reach three baselines, so defer both.

### MPGD, FK steering, and training-dependent molecular guidance

| Candidate | Why it is relevant | Why not in the first implementation wave |
|---|---|---|
| **MPGD, ICLR 2024** [S11] | Clean-estimate updates can avoid generator backwards; a low-cost alternative | Its stronger manifold-preserving claims involve assumptions/projections we do not establish for molecule geometry. An **MPGD without projection** port is possible on VP; label it precisely and do not promise chemical validity |
| **FK steering, ICML 2025** [S12] | Frozen-model particle resampling supports arbitrary rewards, including nondifferentiable rewards | Distinct population-sampling infrastructure, particle ancestry/ESS accounting, and diversity effects. In deterministic FM, duplicated particles have identical futures without a justified stochastic/rejuvenation mechanism |
| **MolGuidance, December 2025 preprint** [S13] | Directly relevant recent molecular guidance comparison | It includes conditional, autoguidance, and model-guidance systems tied to its trained molecular framework. Not a single interchangeable clean-regressor rule for our unconditional checkpoint |
| **EnFlow, December 2025 preprint, revised May 2026** [S14] | Recent energy-guided flow molecular work | Conformer generation/ground-state identification conditioned on a molecular graph; different task and training/path requirements |
| Conditional E(3)-EDM, conditional EquiFM, GeoLDM, classifier-free guidance | Useful molecular systems context | Their trained conditioning/latent interfaces are not created by changing an inference flag on our unconditional checkpoint |

Best-of-N sampling using the **guide** is also worth including as an internal compute-scaling control. Count every candidate and every guide evaluation. It does not provide a third named recent published method by itself, and selection must never consult the final evaluator.

## 10. Fair experimental protocol and feasible compute plan

### 10.1 Fix the experiment before launching sweeps

1. **Select and hash one generator checkpoint.** Carry forward the model-family decision supported by the assignment's matched FM-versus-diffusion experiment. Do not choose the family merely because its baselines are easier.
2. **Freeze the guide/evaluator pair for every method.** Reuse the chosen pair from [the predictor decision](FA_FB_ARCHITECTURE_DECISION.md); this handoff does not change that choice. Verify feature order, normalization, atom counts, and checkpoint split provenance. Existing historical claims of disjointness are not a replacement for index checks.
3. **Start with dipole `mu`.** Adding `alpha` or gap is useful later but not necessary to implement the first comparison. Source availability for another property does not establish a working adapter.
4. **Construct fixed target/atom-count pairs.** For broad targeting, sample `(n_atoms, target_mu)` jointly from a held-out evaluation pool and save them. For fixed-target experiments, choose target values/size strata using training/validation support. Do not pair extreme targets with impossible atom counts or allow methods to change `n_atoms` for an easier task.
5. **Separate tuning from final evaluation.** Use validation targets, seeds, and generated samples for tuning, then freeze configurations before final runs. The evaluator may support a fixed validation model-selection rule, but final test seeds/targets cannot determine hyperparameters or which generated candidates survive.
6. **Save paired initial noise.** Reuse by `(seed, sample_id, atom_count)`. Use separate named RNG streams for initial noise, MC probes, and recurrent/stochastic sampler noise so a larger `K` does not change later samples' initial states.

### 10.2 Metrics that answer the actual question

| Metric | Definition / guard against misleading results |
|---|---|
| Property MAE | `mean(abs(f_B(decoded_z)-target))`; report finite-all and valid-only denominators separately |
| Chemical quality | Atom stability, whole-molecule stability, RDKit validity, connected-valid fraction, nonfinite fraction |
| Target success | Fraction of **all attempts** satisfying validity and `abs(f_B(decoded_z)-target)<=delta` |
| Useful unique yield | Number of distinct decoded valid successful molecules divided by all attempts, and per GPU-hour; for fixed-target panels, count within each target stratum |
| Diversity | Canonical-SMILES uniqueness, fingerprint diversity among valid in-band molecules, and optional invariant-embedding diversity on matched sample counts |
| Novelty / coverage | Novel SMILES relative to training data; coverage of supported condition/size strata where feasible |
| Predictor exploitation | `f_A` versus `f_B` errors, soft-feature versus decoded errors, and second evaluator/physics subset if available |
| Efficiency | Wall time, peak GPU memory, full model/guide pass accounting, optimizer closures, attempts per second, useful yield per GPU-hour |

**Required harness corrections:**

- Hard-decode the atom types before final guide/evaluator scoring. Keep the original soft scores as a separate diagnostic; otherwise a fractional feature vector can hit the target while its actual molecule does not.
- `in_band_fraction` currently counts finite predictor successes without requiring validity. Add `valid_in_band_fraction`; never label the former useful molecular yield.
- Sanitizable disconnected fragments may still be RDKit-valid. Report connectivity explicitly instead of treating disconnected atom clouds as an unquestioned molecular success.
- Define no-valid/no-success outcomes as zero yield and unavailable conditional MAE/diversity, rather than dropping that run or displaying a misleading zero error.
- The present embedding metric is computed over all finite samples; add a valid-and-in-band view with a fixed subsample size. Do not compare raw Gram log-determinants at different set sizes or call size diversity conditional structural coverage.

**Tolerance:** choose a physical `delta` before testing, report predictor validation errors, and show a small sensitivity curve. `2*MAE` is a heuristic scale, not a confidence interval or proof of physical success. If retaining the project's `2*max(MAE_A,MAE_B)` convention, say so and freeze it for all arms. Recompute the validation errors using the final decoded adapter. `s_y`, the loss width, remains a separate parameter.

**Uncertainty:** use at least three sampling seeds if feasible, with paired targets/noise; report per-seed results and mean/dispersion. This quantifies sampling variability for one trained backbone, not uncertainty across generator training runs. Correlated particles/optimized populations are not independent replicates; use seed/batch-level resampling for their intervals.

### 10.3 Two comparisons, with different purposes

**Mechanism-controlled view:** hold checkpoint, reward, decoder, target pairs, base integration grid, and comparable numerical safeguards fixed. For SMG versus LGD versus plug-in, also match onset and solver. This asks whether the guidance estimator helps.

**Method/compute view:** allow each published algorithm its essential sampler/optimizer and give methods a common time or measured-compute budget. This asks which complete inference procedure buys more useful molecules. Report full Pareto curves over control error, validity/diversity, and cost; do not choose a very weak competitor strength after seeing the test outcomes.

These cannot always be collapsed into one perfectly matched table: TFG recurrence, FlowGrad compression, and D-Flow's repeated solves are part of their methods. Record all differences instead of deleting those mechanisms to make NFE look equal.

### 10.4 Cost measurement: use actual runs, not “steps”

The existing `Cost` fields are a starting point. Add:

```text
generator_forward_calls / generator_forward_sample_evals
generator_vjp_calls / generator_jvp_calls / recomputed_forwards
guide_forward_sample_evals / guide_backward_calls / guide_hvp_calls
optimizer_steps / optimizer_closure_calls / recurrent_updates
attempted_samples / finite_samples / valid_samples / valid_in_band_samples
wall_seconds / peak_cuda_memory_bytes / hardware / precision / batch_size
```

Record both calls and sample-evaluations because batch sizes differ. Keep forwards and backwards distinct; one VJP is not universally “one NFE.” Synchronize the GPU around timing, separate initialization from generation, and report whether final decoding/evaluation is included. Count discarded line-search candidates and failed samples. A checkpointed backward can re-execute forwards; include that in measured cost.

### 10.5 Execution order and budgets

These are proposed project budgets, not published optimal settings or performance promises.

| Stage | Work | Exit condition |
|---|---|---|
| A: common infrastructure | Frozen model/guide adapters, decoded evaluator, noise bank, differentiable rollout, logging | Gradient and representation checks pass; unguided sampler usable |
| B: cheap close competitor | Canonical LGD-MC plus plug-in and SMG controls | MC objective/gradient parity; finite small QM9 pilot |
| C: first trajectory method | D-Flow with L-BFGS | End-to-end finite differences; per-sample time/memory measured |
| D: reuse infrastructure | OC-Flow released-objective port | Control gradients correct; penalty/source-shift choices recorded |
| E: coverage/backup | FlowGrad from original state-kick implementation | Uncompressed VJP parity; optional compression benchmark |
| F: final experiments | Freeze configs, then paired multi-seed comparisons | At least three distinct external methods plus internal references |

For a diffusion winner, replace C–E with the shared discrete VP driver, full TFG, FreeDoM, and DPS. D-Flow is optional additional evidence there.

**Pilot first:** 16 attempts per configuration, then 64 validation attempts for viable configurations. Do not train new generators to get a baseline running. Abort nonfinite configurations and record their failure rate; do not repeatedly resample until a desired number of valid molecules appears.

**Bound tuning:** initially cap each method at 12 evaluated configurations and a stated GPU-hour ceiling. Some starting ranges:

- Local plug-in/LGD: guidance weights `{0.1,0.3,1.0}`, spread multiplier `{0.1,1.0}`, `K={4,16}`; a common validation-chosen onset and clipping policy. These scales assume the §3 energy and local sampler conventions.
- D-Flow: L-BFGS learning rate `{0.3,1.0}`, outer iterations `{1,3,5}`, inner limit 5; source-normalization policy chosen and recorded separately.
- OC-Flow: regularization `{0.001,0.01,0.1}` **after fixing its reduction convention**, outer iterations `{1,3,5}` with the same initial L-BFGS configuration.
- FlowGrad: use a small geometric learning-rate pilot, then iteration budgets `{5,10,20}`. State-kick learning rates are not comparable numerically with velocity-control learning rates.
- TFG: staged search of `rho`/`mu`, then `I`/smoothing/recurrence; avoid their full Cartesian product. Include the best simpler configuration in the search space, but detect duplicate selected algorithms.

**Final sample count:** target 1,024 attempts **total per seed**, balanced across predeclared target strata, for three seeds. If the measured trajectory-optimization cost makes that impossible, fix a common smaller count, e.g. 256 total attempts per seed, for the primary all-method table and label the larger cheap-method runs as supplementary. These are totals per seed, not per target value. Retain at least three sampling seeds before increasing the number of target properties.

Estimate the cost using `GPU-hours = attempts * measured_seconds_per_attempt / 3600`, with separately measured batch throughput. For illustration only, 768 attempts at 100 seconds each cost about 21.3 GPU-hours **for one method before tuning**. Published runtimes from another GPU/model are not forecasts for our model. Frozen inference can still be expensive.

If a chosen trajectory method is too expensive after a correct pilot, lower its recorded optimization budget or use the fourth comparator. A fair low-budget negative result is acceptable; silently turning D-Flow into one-step plug-in guidance is not.

## 11. Concrete Claude implementation contract

### 11.1 Suggested files, without a large refactor

Do not create a directory named `guidance/` beside `guidance.py` without deliberately fixing imports. The current flat module imports make that an unnecessary collision risk.

```text
proj1/src/guidance.py                  # add lgd_mc canonical mode and schedule
proj1/src/sampling.py                  # preserve existing public samplers
proj1/src/baseline_rollout.py           # differentiable pure base field + terminal map
proj1/src/baseline_dflow.py             # source optimization
proj1/src/baseline_ocflow.py            # explicit discretized control objective
proj1/src/baseline_flowgrad.py          # original-style state kicks and discrete VJPs
proj1/src/baseline_vp_discrete.py       # only if pursuing TFG/FreeDoM/DPS discrete ports
proj1/scripts/benchmark_guidance.py     # shared config, noise bank, attempts, metrics
proj1/configs/guidance_baselines/       # one resolved config per reported arm
proj1/tests/test_baseline_gradients.py  # meaningful finite-difference/parity checks
```

The public adapter should expose `base_field(z,t,mask,differentiable)`, `endpoint_mean(z,t,mask)`, `project(z,mask)`, `terminal_map(z,t,mask)`, `guide_energy(z,y,mask)`, and `decode(z,mask)`. The guide returns a vector of energies, not an already batch-averaged scalar. Do not require end-to-end methods to fit into a local `guidance_field` API; give them their own sampling drivers.

### 11.2 Configuration requirements

Every result must save its resolved configuration, for example:

```yaml
method: lgd_mc
method_variant: gaussian_fm_adaptation
publication: Song_et_al_ICML_2023
family: flow
generator_checkpoint: REQUIRED
generator_sha256: REQUIRED
guide_checkpoint: REQUIRED
guide_sha256: REQUIRED
evaluator_checkpoint: REQUIRED
evaluator_sha256: REQUIRED
use_ema: true
property: mu
property_units: Debye
energy: gaussian_squared_error
reward_width: REQUIRED_PHYSICAL_VALUE
evaluation_tolerance: REQUIRED_PHYSICAL_VALUE
solver: euler
steps: 100
guidance_onset: 0.1
guidance_weight: 0.3
velocity_relative_clip: 1.0
mc_samples: 4
mc_spread: gaussian_unit_prior_heuristic
mc_spread_multiplier: 0.1
mc_coordinate_projection: zero_com
final_property_input: decoded_one_hot
target_pair_manifest: REQUIRED
initial_noise_manifest: REQUIRED
sampling_seed: REQUIRED
```

This is a **schema/example**, not a runnable configuration with validated optimal values. Reject missing `REQUIRED` fields before a real run. A hash and method name are not sufficient without preprocessing and source-variant information.

### 11.3 Checks that decide whether an implementation is ready

1. **Model freeze:** parameter tensors and buffers unchanged after optimization; no model optimizer created.
2. **Differentiability:** finite differences match source/control/MC gradients on a small smooth test problem and a tiny molecule with frozen network weights. Use float64 for the synthetic reference.
3. **Geometry:** masked values/gradients vanish, coordinate CoM stays zero, translations/rotations/permutations behave consistently; transform random probes when checking pathwise symmetry.
4. **Reduction:** zero guidance/control recovers the declared underlying sampler, including its terminal map. For recurrent samplers, use recurrence 1 for trajectory-equality tests.
5. **Representation:** clean one-hot adapter matches its source predictor; final evaluation uses decoded features; atom-order and units verified.
6. **Statistical bookkeeping:** attempted counts fixed, failures retained, no evaluator-driven candidate selection, pairing survives batch-size changes.
7. **Cost:** instrumented closure totals agree with a tiny known optimizer run; checkpoint recomputations and final rerolls included.
8. **Names:** `tfg_mc` and `lgd_mc` aliases cannot create duplicate “external methods”; variant names distinguish paper-native and adapted choices.

### 11.4 What has actually been checked in this research pass

The public source snapshots are in [audit/guidance_baselines_2026](audit/guidance_baselines_2026). Each repository has a `manifest.json` with its commit, exact source URLs, and file SHA-256 hashes. Downloaded source was inspected as text; it was not installed or executed.

Independent synthetic checks were run using [check_formulas.py](audit/guidance_baselines_2026/check_formulas.py), with results in [formula_checks.json](audit/guidance_baselines_2026/formula_checks.json):

| Check | Maximum absolute discrepancy |
|---|---:|
| LGD weighted-gradient formula versus finite differences | `3.59e-11` (rounded upward) |
| Duplicating MC probes leaves the log-mean-exp gradient unchanged | `2.78e-17` |
| Euler velocity-control adjoint versus finite differences | `5.36e-11` |
| FlowGrad-style state-kick adjoint versus finite differences | `3.41e-11` (rounded upward) |
| Gaussian-channel FM score-to-velocity conversion | `4.45e-16` (rounded upward) |
| Zero-CoM projection idempotence | `2.23e-16` (rounded upward) |

These checks validate the displayed implementation formulas on small examples. They do **not** establish QM9 effectiveness, runtime, reproduction of published scores, or a novelty claim for SMG. No project sampler/trainer was modified and no QM9 baseline experiment was launched as part of this research task.

## 12. Primary sources and pinned implementation references

The short citations throughout this document link to primary papers. The explanations of how to fit the methods to our code, the recommended hyperparameters, and the implementation order are project recommendations, not statements copied from the papers.

- **[S1] Song et al., LGD-MC, ICML 2023.** [Proceedings and bibliography](https://proceedings.mlr.press/v202/song23k.html); [paper, equation (9) and §4.3](https://proceedings.mlr.press/v202/song23k/song23k.pdf). No original-author LGD repository was verified in this pass; the TFG code is a later reimplementation.
- **[S2] Ben-Hamu et al., D-Flow, ICML 2024.** [Proceedings and bibliography](https://proceedings.mlr.press/v235/ben-hamu24a.html); [paper, §6.3 and appendix B.3](https://arxiv.org/html/2402.14017v2). Distinguish it from the unrelated D-peptide paper also called D-Flow.
- **[S3] Wang et al., Training Free Guided Flow Matching with Optimal Control, ICLR 2025.** [Conference paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/19a94fdf9e1c5b387830e4c4fef6972a-Paper-Conference.pdf); [searchable paper, algorithm 1 and appendix E.2](https://arxiv.org/html/2410.18070v3); [official repository](https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control).
- **[S4] Liu et al., FlowGrad, CVPR 2023.** [Proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_FlowGrad_Controlling_the_Output_of_Generative_ODEs_With_Gradients_CVPR_2023_paper.html); [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_FlowGrad_Controlling_the_Output_of_Generative_ODEs_With_Gradients_CVPR_2023_paper.pdf); [original repository](https://github.com/gnobitab/FlowGrad).
- **[S5] Ye et al., TFG, NeurIPS 2024.** [Conference paper](https://proceedings.nips.cc/paper_files/paper/2024/file/2818054fc6de6dacdda0f142a3475933-Paper-Conference.pdf); [searchable paper, algorithm 1 and appendix D.7](https://arxiv.org/html/2409.15761v2); [official repository](https://github.com/YWolfeee/Training-Free-Guidance). Its reimplementations of other methods are not those other methods' original code.
- **[S6] Yu et al., FreeDoM, ICCV 2023.** [Conference paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.pdf); [author publication page](https://villa.jianzhang.tech/publication/100070/).
- **[S7] Chung et al., DPS, ICLR 2023.** [Paper](https://arxiv.org/abs/2209.14687); [official repository](https://github.com/DPS2022/diffusion-posterior-sampling).
- **[S8] Feng et al., On the Guidance of Flow Matching, ICML 2025.** [Proceedings](https://proceedings.mlr.press/v267/feng25s.html); [searchable paper, §3.3–3.4](https://arxiv.org/html/2502.02150v3); [official repository](https://github.com/AI4Science-WestlakeU/flow_guidance). The v3 covariance prose has a wording inconsistency (“inverse covariance”) next to an equation for covariance; use the equation and independent derivation, not that word alone.
- **[S9] Boys et al., TMPD, TMLR 2024.** [Paper](https://arxiv.org/html/2310.06721v3); [journal record](https://openreview.net/forum?id=4unJi0qrTE); [author bibliography confirming TMLR 2024 and later ICLR journal track](https://odakyildiz.com/papers.html); [PyTorch implementation](https://github.com/bb515/tmpdtorch). Do not label it AISTATS 2024 or confuse its later journal-track appearance with original publication.
- **[S10] Lin et al., TFG-Flow, ICLR 2025.** [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/aca042ae71283d25b3d063fc011db9e8-Abstract-Conference.html); [paper](https://arxiv.org/html/2501.14216v2); [official repository](https://github.com/linhaowei1/TFG-Flow).
- **[S11] He et al., MPGD, ICLR 2024.** [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c355566ce402de341c3320cf69a10750-Abstract-Conference.html); [paper](https://arxiv.org/abs/2311.16424).
- **[S12] Singhal et al., A General Framework for Inference-time Scaling and Steering of Diffusion Models, ICML 2025.** [Proceedings and software link](https://proceedings.mlr.press/v267/singhal25b.html).
- **[S13] Jin et al., MolGuidance.** [December 2025 arXiv record](https://arxiv.org/abs/2512.12198); [authors' repository](https://github.com/Liu-Group-UF/MolGuidance). Used for screening, not counted as a venue-verified core comparator here.
- **[S14] Xu et al., EnFlow.** [ArXiv record, initially December 2025, revised May 2026](https://arxiv.org/abs/2512.22597); [authors' implementation and training stages](https://github.com/Rich-XGK/EnFlow). The current v2 title is *Energy-Guided Generative Modeling for Low-Energy Molecular Structure Discovery*; the original title and repository description use *Energy-Guided Flow Matching Enables Few-Step Conformer Generation and Ground-State Identification*. Used for task/compatibility screening only.

### Audited revisions

| Repository | Pinned revision | Most useful files / local copy |
|---|---|---|
| TFG | `f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77` | [TFG](https://github.com/YWolfeee/Training-Free-Guidance/blob/f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77/methods/tfg.py), [LGD](https://github.com/YWolfeee/Training-Free-Guidance/blob/f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77/methods/lgd.py), [base sampler](https://github.com/YWolfeee/Training-Free-Guidance/blob/f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77/methods/base.py), [local manifest](audit/guidance_baselines_2026/TFG/manifest.json) |
| OC-Flow | `ca218ba616f7e3a58a05924fdb547bd579b3c900` | [molecular methods](https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control/blob/ca218ba616f7e3a58a05924fdb547bd579b3c900/molecule/guided_sample.py), [driver](https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control/blob/ca218ba616f7e3a58a05924fdb547bd579b3c900/molecule/main_guided.py), [local manifest](audit/guidance_baselines_2026/OC-Flow/manifest.json) |
| FlowGrad | `0cc3717e51ce2810bc44f5755397d11a8c43989a` | [original trajectory/optimization implementation](https://github.com/gnobitab/FlowGrad/blob/0cc3717e51ce2810bc44f5755397d11a8c43989a/utils/flowgrad_utils.py), [local manifest](audit/guidance_baselines_2026/FlowGrad/manifest.json) |
| Flow guidance framework | `b47872e9f72c0b8360c6232fa8ae45f64159bdae` | [repository snapshot](https://github.com/AI4Science-WestlakeU/flow_guidance/tree/b47872e9f72c0b8360c6232fa8ae45f64159bdae), [local manifest](audit/guidance_baselines_2026/FlowGuidance/manifest.json) |
| TFG-Flow | `4e7ef1655d77d564e2d0a37b016abab0780b5d04` | [mixed discrete/continuous guidance](https://github.com/linhaowei1/TFG-Flow/blob/4e7ef1655d77d564e2d0a37b016abab0780b5d04/diffusion/guidance.py), [local manifest](audit/guidance_baselines_2026/TFG-Flow/manifest.json) |
| DPS | `effbde7325b22ce8dc3e2c06c160c021e743a12d` | [original conditioning code](https://github.com/DPS2022/diffusion-posterior-sampling/blob/effbde7325b22ce8dc3e2c06c160c021e743a12d/guided_diffusion/condition_methods.py), [local manifest](audit/guidance_baselines_2026/DPS/manifest.json) |

## 13. Wording for the eventual paper

Suggested scope statement, to use only after these experiments actually run:

> We compare inference-time guidance algorithms on a shared, independently trained QM9 generator. All generator and property-network parameters are frozen during sampling. We implement the published guidance mechanisms through a common molecular representation and disclose changes to samplers, reward normalization, and optimization settings. These experiments measure guidance performance under a controlled backbone and do not reproduce the original papers' complete training pipelines.

Use method labels such as `LGD-MC (Gaussian-FM adaptation)`, `D-Flow (our backbone)`, `OC-Flow (released QM9 objective port)`, and `FlowGrad (molecular, uncompressed)` where applicable. Do not claim “we outperform TFG” using only the current `tfg_mc` component, or “we sample the exact conditional distribution” from an approximate-guidance or terminal-optimization experiment.

**Claude's next action:** implement the shared differentiable rollout and corrected decoded evaluation, finish LGD-MC, then build D-Flow and OC-Flow on that rollout. Add FlowGrad to broaden the comparison. Freeze the base checkpoint; use the validation pilot to settle feasible inference budgets before large runs.

[S1]: https://proceedings.mlr.press/v202/song23k.html
[S2]: https://proceedings.mlr.press/v235/ben-hamu24a.html
[S3]: https://proceedings.iclr.cc/paper_files/paper/2025/file/19a94fdf9e1c5b387830e4c4fef6972a-Paper-Conference.pdf
[S4]: https://openaccess.thecvf.com/content/CVPR2023/papers/Liu_FlowGrad_Controlling_the_Output_of_Generative_ODEs_With_Gradients_CVPR_2023_paper.pdf
[S5]: https://proceedings.nips.cc/paper_files/paper/2024/file/2818054fc6de6dacdda0f142a3475933-Paper-Conference.pdf
[S6]: https://openaccess.thecvf.com/content/ICCV2023/papers/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.pdf
[S7]: https://arxiv.org/abs/2209.14687
[S8]: https://proceedings.mlr.press/v267/feng25s.html
[S9]: https://arxiv.org/html/2310.06721v3
[S10]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/aca042ae71283d25b3d063fc011db9e8-Abstract-Conference.html
[S11]: https://proceedings.iclr.cc/paper_files/paper/2024/hash/c355566ce402de341c3320cf69a10750-Abstract-Conference.html
[S12]: https://proceedings.mlr.press/v267/singhal25b.html
[S13]: https://arxiv.org/abs/2512.12198
[S14]: https://arxiv.org/abs/2512.22597
