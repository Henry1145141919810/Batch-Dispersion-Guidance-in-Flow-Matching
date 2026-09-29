# Manuscript v8: revision and evidence map

29 September 2026. **BDG: Batch-Dispersion Guidance for Property-Band Generation**.
Authors: Bobo Li, Henry Huang, Idea Idehpour.

This release revises the paper using the attached **BDG Defense — Group 2 v1.pptx**
as reference material. Claims in the deck were checked against the result files;
the deck was not edited. The paper has **five main pages, one reference page and
seventeen appendix pages**. The template style, type sizes and spacing are unchanged.

## Changes from v7

- Reorganized the argument around model selection, guidance, the added BDG term,
  and transfer. Restored four related-work paragraphs and the rubric's three
  Discussion paragraphs. The abstract is 205 whitespace-delimited words.
- Retained the completed own-VP benchmark and its checkpoint/hardware disclosures.
  Extended the guided-versus-unguided interpretation to both trained families:
  dipole/gap coverage gains are 2.75/2.35 percentage points for FM and 2.13/2.60
  for VP. These four comparisons exceed the protocol threshold; neither
  polarizability change does.
- Made the zero-gain limit and coefficient bound explicit: eta = 0 gives plug-in;
  positive tau is required, and w_eff >= 1 - eta. Expansion needs eta > 1.
- Expanded main Table 2 with the stronger plug-in control (w = 4) and molecular
  stability. It remains the dipole q50 experiment, y = 2.4932 D, band +/-0.17541 D.
- Added dipole distinct-valid yield and measured runtime to main Table 3; added
  CpG runtime to main Table 4 while retaining GC/CpG Hamming diversity.
- Redrew the overview as two modality paths through one shared BDG coefficient.
  Figure 2 now shows all six molecular spread curves: three properties on both
  our FM and EquiFM, with distinct line styles and marker fills.
- Added Table S3 with separate generator forward, VJP, JVP and guide calls.
  BDG matches plug-in's neural calls; LGD-MC and TFG use five times as many guide
  forward/backward calls in the reported molecular configuration.
- Added exploratory Table S9 comparing coverage gain per stability point lost
  against a shared zero-gain w = 1 reference. BDG at eta = 8 exceeds the stronger
  plug-in ratio in five of six tasks; at eta = 4 it does so in four of six.
  These are descriptive ratios, not joint usable yield or significance tests.
- Added the descriptive coefficient/coverage correlation audit: Spearman
  0.92-0.99 across the sixteen nonzero configurations in each of twelve grids.
  The text distinguishes association from evidence that feedback is necessary.
- Removed a duplicate QM9 bibliography entry that made BibTeX return an error.
  The integrity checker now checks duplicate/missing bibliography keys and the
  three-paragraph Discussion requirement.

## Deck claims checked before inclusion

| Topic | Paper treatment |
|---|---|
| FM versus diffusion | Quality supports the selected FM checkpoint; shared settings do not erase different checkpoint selection or hardware. No family-wide or speed-superiority claim. |
| DNA baseline overlap | Plug-in, TMPD and LGD-MC have identical decoded coverage on both GC and CpG at w = 1/4 in the late window. TFG-MC differs. No blanket abstract claim. |
| CpG mechanism | GC is affine and CpG quadratic, but their band widths and clipping also differ. Larger CpG gains do not prove that curvature causes the difference. |
| Cost accounting | Forward and derivative operations are reported separately. Their sum is not presented as an equal-cost forward-call count. |
| Stability exchange | Eta = 4 and eta = 8 results are distinguished. A result from eta = 8 is not attributed to the eta = 4 headline setting. |
| Expansion | All six eta = 8, w = 1 curves widen monotonically over the four tested setpoints. The comparison with plug-in is limited to tested w = 1/4. |
| Target convergence | Lower decoded MAE in all six contracting headline comparisons and closer dipole ablation means are supported. Universal mean-bias improvement and faster convergence are not measured. |
| Early-window GC | The 22.82-point plug-in gain is decoded coverage, not only a pre-decoding effect. It is a separate GC-only diagnostic. |

## Current evidence and rubric map

| Requirement | Location |
|---|---|
| Single question, two modalities | Abstract; Introduction; shared-rule overview, Figure 1 |
| Trained FM versus diffusion and justified choice | Sections 3.2-3.3, 4.2; Table 1; training records, Tables S1-S2 |
| Guided versus unguided | Section 4.3; Table 3; own-VP Table S7 |
| Innovation, removal and parameter ablations | Equation 4; Section 4.4; numeric Table 2; Figure 2a; Table S9; full Tables S10-S15 |
| Recent comparisons, quality/diversity/cost | Section 4.5; Table 3; operation counts S3; full S4-S6 |
| Second-modality transfer | Section 4.6; Table 4; Figure 2b; complete DNA Table S16 |
| Uncertainty and failure analysis | Section 4.7; Discussion; Appendix A.3/A.5/A.6; contrast Table S8 |
| Reproducibility | Appendix A.7-A.8; Algorithm 1; provenance Table S17; results_manifest_v8.json |

The DNA comparators are guidance adaptations on the shared backbone. Dirichlet
FM, Fisher Flow and MOG-DFM remain background references, not newly evaluated
generators. Sample-level joint molecular yield and batch-aware intervals remain
unperformed. The manuscript states those limits explicitly.

## Validation and preservation

- Eighteen inline table blocks regenerated from the same final result cells as
  the three vector figures. All **973 source hashes** verified: 774 molecular
  files (162 headline, including 18 own-VP; 612 ablation) and 199 sequence files.
- Verified own-FM/VP guidance contrasts, all six monotone spread curves, the
  six molecular MAE improvements, the exchange ratios, correlations and DNA
  baseline identities and coverage contrasts.
- All fourteen required subsections, three Introduction paragraphs, four
  contribution bullets, three Discussion paragraphs and the abstract word
  range pass the structure checks. No unresolved or unreferenced floats.
- Built with installed LaTeX/BibTeX; five main pages, zero overfull boxes,
  zero undefined citations/references. All 23 pages rendered and visually reviewed.
- The built-in editor compiler reports a platform-directory error. The editable
  source remains available; the installed compiler produced the verified PDF.
- `v7_pre_v8_2026-09-29/` preserves the exact files before this revision.
  Prior numbered releases remain unchanged. `v8_2026-09-29/` contains a self-contained
  LaTeX/PDF snapshot and SHA-256 inventory; `v8_main.tex` / `v8_main.pdf` are the
  numbered source/PDF copies. All live prose and tables stay in `paper/main.tex`.

Base repository revision: `02209e303952eab43dc3627776c8450873938069`.
Reference deck SHA-256:
`4eed9c287a55154a166d025272e9d1145ee7ba173adbca07b50745da62535739`.
