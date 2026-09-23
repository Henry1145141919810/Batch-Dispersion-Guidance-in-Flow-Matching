# EquiFM usability audit

**22 September 2026.** Scope: determine whether released EquiFM weights and external property predictors can support this project's comparison of existing and proposed guidance methods. No model training, production sampler changes, or guidance-ranking sweep was performed.

## Decision

**EquiFM is executable, produces credible molecules in a local pilot, and has compatible external clean property predictors. It is a useful candidate for a supplementary transfer experiment. The released stack does not meet the full current design, even when exact split IDs are allowed to differ.**

The two remaining substantive mismatches are:

1. The available unconditional generator was trained on the entire 100K upstream training partition. Our protocol places generator and guide on one half, evaluator on the other.
2. Its hybrid, geometry-aligned training path does not justify the scalar Gaussian posterior-covariance formula used by our uncertainty-based methods. This affects the methods' interpretation, beyond whether their code can execute.

The property-pair issue is more tractable: **the already-downloaded TFG clean guide and OC-Flow property predictor have strong complementary-half evidence for all three properties**. They can supply an entirely external guide/evaluator pair without new training. They do not fix the two generator issues above.

## 1. What was actually checked

Sources and artifacts:

- [EquiFM paper, section 5.3](https://papers.nips.cc/paper_files/paper/2023/file/01d64478381c33e29ed611f1719f5a37-Paper-Conference.pdf).
- [Official MolFM repository](https://github.com/GenSI-THUAIR/MolFM), snapshot `2c83c0f065e12a79e9206035235c72e06cbb279c`.
- [Original NeurIPS supplemental archive](https://papers.nips.cc/paper_files/paper/2023/file/01d64478381c33e29ed611f1719f5a37-Supplemental-Conference.zip), including the nested `code.tar.gz` training source.
- [OC-Flow repository](https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control), snapshot `ca218ba616f7e3a58a05924fdb547bd579b3c900`; [paper, appendix E.2](https://arxiv.org/html/2410.18070v3#A5.SS2).
- [TFG paper, appendix D.7](https://arxiv.org/html/2409.15761v2#A4.SS7), plus the TFG weights already present in this workspace.

All local audit artifacts are under `audit/equifm_20260922/`. The GitHub downloads are pinned and hashed in `manifest.json`. Checkpoints were loaded as tensors with `weights_only=True`; metadata was read through the existing restricted metadata reader. Selected reviewed source definitions were loaded without running the repositories' module-level setup.

The same EquiFM generator occurs in three places: MolFM sampling, OC-Flow, and the original supplement. **All three are byte-identical**, SHA-256:

`47b40ae673c1117219432f0c81ae61f72ac380761200144ed04ebb9f179d3527`

I found no additional half-trained unconditional generator or conditional checkpoint in the checked repository tree or supplemental inventory. This is a statement about the checked releases, not proof that the authors have no other weights.

## 2. The split: the paper does use halves, but the released model is different

The user is correct about the original experiment. EquiFM section 5.3 trains a property evaluator on one 50K half and **property-conditioned generators** on the other. That is a generator/evaluator split. It does not itself supply our separate clean guide and evaluator.

The downloadable checkpoint has `dataset=qm9`, `conditioning=[]`, and `context_node_nf=0`. OC-Flow appendix E.2 explicitly identifies its EquiFM prior as trained on the whole QM9 training partition. The identical checkpoint hashes connect that statement to the model audited here.

There are two misleading metadata patterns to avoid:

- OC-Flow's `molecule/utils.py` changes the generator argument to `qm9_second_half` **after loading metadata, at inference**. This cannot change what the weights trained on.
- Its property-training script constructs the training loader with default `qm9_first_half`, then changes `args.dataset` to `qm9_second_half` to construct an auxiliary evaluation loader. The subsequently saved arguments therefore say second half even for a predictor trained on the first. Its logged `best_test` concerns that auxiliary half, not the official held-out test partition.

Allowing different split seeds/IDs is sufficient to accept an external *disjoint* protocol. It does not turn a full-training generator into a half-training generator. The present project rule is in [SPLIT_PROTOCOL.md](SPLIT_PROTOCOL.md).

### Reconstruction and empirical evidence

The audit downloaded the original QM9 XYZ archive and exclusion list, rather than substituting a reprocessed dataset:

1. 133,885 original molecules; remove the 3,054 excluded entries.
2. Seed 0 upstream permutation: 100,000 train, 17,748 validation, 13,083 test.
3. Preserve original TAR iteration order when constructing the processed training array. The upstream filter tests membership; it does **not** retain split-permutation order. TAR IDs were verified to be sequential.
4. Seed 42 permutation of that training array, yielding the two 50,000-molecule halves.
5. Sample 1,024 molecules independently from each partition with audit seed 20260922.

Use the entire official validation partition to recover each property's mean/MAD. No regression fit was used to calibrate these audit predictions. The first/second-half error signatures below support the source-level split interpretation; they are not a substitute for an author-supplied checkpoint training-ID manifest.

## 3. Property predictors: usable, with an available external partner

MolFM itself does not provide the property weights in the checked release. **OC-Flow provides six separate clean property EGNNs alongside its EquiFM copy.** The three tested here are dipole moment, polarizability, and HOMO–LUMO gap. The other three were downloaded but not numerically audited.

These networks take coordinates and continuous five-channel atom features, with a mask and complete geometric edge list; they do not require predicted bonds or SMILES. Their SiLU networks expose derivatives in both coordinates and atom features.

Measured absolute errors on the reconstructed original data:

| Property | Units | OC first half | OC second half | OC held-out test |
|---|---|---:|---:|---:|
| mu | Debye | 0.00154 | 0.04914 | 0.05172 |
| alpha | Bohr cubed | 0.00229 | 0.10021 | 0.09602 |
| gap | Hartree | 0.0000536 | 0.002293 | 0.002374 |

Each entry is a 1,024-molecule subset, not a full-partition benchmark. Test standard errors are respectively 0.00486 D, 0.00723 Bohr cubed, and 0.0000824 Ha. Gap test MAE is approximately 64.6 meV. These are errors on real QM9 geometries, not a DFT validation of generated molecules.

Property derivative smoke checks used two real molecules in double precision: finite, nonzero first derivatives and Hessian-vector products for all three properties; rotation/translation/permutation discrepancies below 2e-15 in physical output; padded outputs/gradients unchanged; padded gradients zero; tested coincident-atom case finite. These bounded checks do not certify every out-of-distribution intermediate.

Physical normalizers recovered from the official validation partition:

| Property | Mean | Mean absolute deviation |
|---|---:|---:|
| mu, D | 2.67508745 | 1.17573273 |
| alpha, Bohr cubed | 75.37343597 | 6.27277231 |
| gap, Ha | 0.25221726298 | 0.03902422264 |

The original loader uses eV for gap. Converting both mean and MAD consistently to Hartree gives the same normalized network output and reports in our project's units.

### A partner without new training

TFG documents a clean guide trained on the second half with time input fixed at zero. The audit also evaluated those existing weights using the reconstructed split and the same exact physical normalizers:

| Property | TFG guide first half | TFG guide second half | TFG guide held-out test |
|---|---:|---:|---:|
| mu, D | 0.04145 | 0.001218 | 0.04388 |
| alpha, Bohr cubed | 0.09211 | 0.003521 | 0.09176 |
| gap, Ha | 0.002481 | 0.0000457 | 0.002540 |

Thus a credible external pair is **TFG clean guide as f_A, OC-Flow clean predictor as f_B**. Reversing their roles is also technically possible but would be a separate experimental choice. Membership evidence is much stronger than inferring disjointness solely from error correlations: each fits the expected opposite half dramatically better, across all three properties.

TFG's existing evaluator instead fits the first half, like OC-Flow's predictor. Their tensors differ, but **different weights do not mean disjoint training sets**. Pairing those two evaluators would not solve the requirement.

### Do not reuse the rounding wrapper for guidance

OC-Flow's public scoring helper first decodes the generated atom channels using `torch.round`. The rounded branch has zero direct derivative through atom-type inputs. A tested mu feature-gradient norm fell from 1.843 to exactly zero after adding that rounding operation.

Call the underlying property network on unrounded, correctly scaled features during guidance; apply decoding for final scoring. This preserves the network weights but changes the evaluation point. Monitor the discrepancy: on the 500-step pilot, mean absolute soft-versus-rounded prediction differences were **0.0981 D, 0.2082 Bohr cubed, and 0.001685 Ha**. Those are not negligible compared with real-data predictor error.

## 4. The frozen generator works locally

The released EMA generator contains 5,339,920 parameters, a 256-wide, nine-layer equivariant network. Its native state contains three coordinate channels, five continuous atom-type channels, and one additional nuclear-charge channel. It includes hydrogens and represents H/C/N/O/F.

Preserve its scales: coordinates / 1, one-hot features / 4, nuclear charge / 10. The extra charge channel remains part of the generator state even though the property model does not consume it directly. The guidance adapter test found nonzero gradients into it through the generator.

Local environment: RTX 5080, PyTorch 2.11.0+cu128. Strict weight loading passed. Generator forward, input gradients, second gradients, and JVP smoke checks were finite. Native sampling was performed from time 1 to 0 with Euler, keeping the released velocity and state conventions.

| Metric | 100 Euler steps | 500 Euler steps |
|---|---:|---:|
| Molecules | 256 | 256 |
| Finite samples | 256 | 256 |
| Atom stability | 98.37% | 98.67% |
| Molecule stability | 86.72% | 86.33% |
| Project EDM-style validity | 94.53% | 94.53% |
| Uniqueness among valid molecules | 100% | 100% |
| Connected among valid molecules | 97.11% | 98.35% |
| Sampling plus chemistry scoring time | 12.85 s | 63.51 s |

Both pilots use common seed 20260922 and the same 256 atom-count masks sampled from the official test subset, batch size 32. The project chemistry scorer was reused. These are small usability checks, not reproductions of the paper's reported benchmark; the node-count choice, solver, sample count and metric implementation matter. Novelty was not assessed. The tiny stability difference does not establish that 100 steps is better than 500.

## 5. Where our guidance mathematics stops transferring directly

This is continuous flow matching, including continuous atom-type channels; it is not a discrete atom-state CTMC. But it is also not the project's single independent Gaussian interpolation.

The original supplemental training source confirms that the released `HB_path` uses:

- Coordinates: a nearly linear path after target-dependent noise rotation/alignment and Hungarian assignment.
- Atom and charge features: a VP path with fresh Gaussian noise, beta increasing linearly from 0.1 to 20.
- An additional coordinate angle penalty, enabled in checkpoint metadata.

The source evidence is in local supplemental `equivariant_diffusion/cnflows.py`: alignment at 665–680, hybrid path at 704–736, feature target at 306–314, and angle loss at 901–915. The native sampler multiplies only the feature head by the VP factor.

### Endpoint proxy: implementable

Use native time tau (1 noise, 0 data), epsilon = 1e-4, b = 1-epsilon, s_x = epsilon+b*tau, and a = exp(-19.9*tau^2/4 - 0.1*tau/2). Let q_x and q_h denote the **raw** network outputs before the sampler's VP multiplication. The targets in the released training code give the following endpoint proxies in native normalized units:

```
m_x = b*x_tau - s_x*q_x
m_h = a*h_tau + q_h/a
```

The plus sign in the feature expression follows the implementation's `VP_field` return value, which is `a*h_0 - a^2*h_tau`. Treating the raw head as epsilon prediction or as an ordinary linear-flow velocity gives the wrong formula.

For a pure squared-loss conditional-expectation velocity, the coordinate mean relation follows algebraically even with coupled endpoints. **Alignment alone does not invalidate that mean identity.** The additional angle penalty and finite model error mean the released coordinate expression should nevertheless be described as an endpoint proxy, not a certified exact conditional mean.

An audit-only adapter passed the existing `guidance_field` interface for `plug` and `tfg_mc`: both returned finite nonzero coordinate/feature gradients, with zero padded gradients, at native times 0.5, 0.2 and 0.05 on two sampled states. This verifies the differentiation path, not guided-generation quality. Its dummy covariance coefficient is deliberately unused by those two modes and must never be used for uncertainty methods.

### Covariance: the blocking mathematical mismatch

Our `Posterior` interface and uncertainty methods use `Sigma = k * Dm`, derived for an independent additive Gaussian channel. The coordinate noise here is aligned as a function of the clean geometry. The conditional kernel in the represented coordinate space therefore is not that independent Gaussian channel, and the same Tweedie covariance identity is not established.

Even if alignment were removed, the hybrid schedules would require different coordinate and feature coefficients, not the current shared scalar k. Under an actually independent block Gaussian channel the relation would be `Sigma = Dm * diag(s_x^2/(1-tau), (1-a^2)/a)`, including cross-block derivatives and working in the centered coordinate subspace. That hypothetical formula does not fix the released aligned coordinate kernel.

The existing sampler's single score-to-velocity conversion also assumes the original path. A complete EquiFM implementation must specify the guidance-to-velocity mapping; passing the endpoint proxy to our existing FM sampler is insufficient.

| Current arm | What this audit establishes |
|---|---|
| unguided | Native frozen sampling works. |
| plug, tfg_mc | Endpoint-proxy differentiation works; need a justified or explicitly empirical velocity correction and full pilot evaluation. |
| tmpd / smg_var | Need an EquiFM-compatible covariance operator. |
| smg, smg2, smg2_curv | Same covariance issue, including the curvature moments. Derivative support itself is available. |
| lgd_mc, osc as currently implemented | Their uncertainty estimates depend on the same covariance construction. |
| btvg and its variance component | Same issue for property variance and its derivative. |
| shg_plug_btvg | Inherits the BTVG issue; must also reverse the schedule into native time. |

Using an arbitrary positive covariance surrogate could make every arm executable, but would measure a new approximation choice alongside method portability. The conclusion would be an empirical robustness result, not a faithful validation of the current posterior-moment derivation.

## 6. Recommended use

**For the full current design:** do not call the public EquiFM stack a ready replacement. Obtaining a half-trained unconditional checkpoint would repair the data protocol but leave the hybrid/EOT covariance problem. A compatible covariance construction is a separate research/implementation task.

**For a supplementary external-FM experiment:** the actual generator and external property models are now available and audited. Use the TFG clean guide and OC-Flow evaluator as the candidate independent pair; explicitly disclose the full-training generator and any covariance surrogate. Keep one frozen base, one property pair, shared targets/seeds/atom counts/solver budgets, and report evaluator error, validity/stability, and guide/evaluator disagreement. Do not optimize and assess all arms with the same property network.

Start with a complete unguided/plug/tfg_mc pilot after specifying the velocity correction. Add proposed uncertainty methods only when their covariance approximation is explicitly defined and checked. A comparison omitting the proposed methods does not by itself answer the user's intended portability question.

No guided outcome ranking is claimed in this audit. The completed work establishes executable assets, a credible external property pairing, and precisely which design changes remain necessary.

## 7. Local evidence and reproduction

- `audit/equifm_20260922/check_usability.py`: original-data reconstruction, three property audits, generator derivatives, two sampling pilots.
- `audit/equifm_20260922/usability_results.json`: exact measurements, metadata, normalizers, and sampling metrics.
- `audit/equifm_20260922/check_adapter.py` / `adapter_results.json`: endpoint-proxy gradient checks and supplemental checkpoint comparison.
- `audit/equifm_20260922/check_partner.py` / `partner_results.json`: TFG guide/evaluator behavior on reconstructed halves.
- `audit/equifm_20260922/manifest.json`: pinned repository files and hashes.
- `audit/equifm_20260922/supplement_code_inventory.json`: contents of the original training-code archive.
- `audit/equifm_20260922/samples_euler100.pt` and `samples_euler500.pt`: pilot outputs, including both soft and rounded features.

Run the three audit scripts with the existing project Python environment. They require the downloaded assets listed above, NumPy, PyTorch, and the project's chemistry evaluator dependencies. They do not require retraining, importing the full external repositories, or installing their Docker environments.
