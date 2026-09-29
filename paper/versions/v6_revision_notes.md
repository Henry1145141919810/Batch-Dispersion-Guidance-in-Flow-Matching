# Manuscript v6: evidence and grading map

28 September 2026. Current manuscript: `paper/main.tex`; compiled paper:
`paper/main.pdf`. The revision uses the local final run: 756 molecular cells
and 199 DNA result files. DNA contains 12 repeat configurations across stages;
the repeats are checked but never counted as extra seeds. Earlier n=5000
results and pilot data are excluded from headline results.

## Scientific conclusion

BDG changes sampled property spread at fixed nominal strength. At the declared
molecular comparison threshold, its two headline settings yield zero improvements,
three losses and nine unresolved contrasts against plug-in. An unresolved
contrast is not proof of equality. The contracting DNA setting improves CpG
coverage over plug-in by 4.82 pp at w=1 and 16.83 pp at w=4; GC gains are 0.42
and 2.32 pp, respectively. Fidelity and applied-force limitations accompany these
results. This is property-dependent transfer, not a null across both modalities.

## Rubric coverage

| Assignment item | Evidence in v6 | Remaining limit |
|---|---|---|
| Abstract (.225) | One 215-word paragraph: problem, modalities, method, results, transfer and limits | Trained VP benchmark explicitly pending |
| Introduction (.525) | Exactly three paragraphs, one hypothesis and four supported contributions | No claim that FM already won model selection |
| Related work (.425) | Four idea/modality paragraphs; near-neighbor batch/moment work and four guidance baselines | Adaptations distinguished from original benchmark reproductions |
| Data (.3) | Representations, conditioning, disjoint molecular predictor halves, all split counts | Random sequence splits do not establish cluster generalization |
| FM (.475) | Path/loss, EGNN size, training settings, evaluated checkpoint and Euler sampler | Full training wall time not recorded |
| Diffusion (.475) | VP process, noise objective, reverse ODE and configured matched architecture | Actual benchmark/checkpoint selection must be filled when delivered |
| Guidance (.3) | Endpoint pullback, calibration scale, strength/window, clip and frozen components | Nominal strength is not applied-force matching |
| Innovation (.625) | Residual, detached coefficients, reduction, repulsion condition, algorithm | No exact setpoint or feedback-necessity claim |
| Transfer methods (.2) | Same coefficient; CNN, simplex path/projection, analytic properties and decoding | Biological activity not inferred from composition |
| Overview (.2) | Revised original schematic distinguishing training, frozen sampling and shared BDG | VP branch clearly pending evaluation |
| Evaluation (.4) | n, seeds, batches, uncertainty, decoded metrics, quality and time | Binomial approximation omits batch coupling and rerun noise |
| FM vs diffusion (.45) | Table 1 includes our FM, pending own VP, separate borrowed models | This empirical criterion remains incomplete until VP statistics arrive |
| Guidance result (.325) | Section 4.3 and common Table 3: matched unguided/plug-in plus chemistry | Alpha gain unresolved |
| Ablations (.675) | Table 2, Figure 2a, all 12 supplementary grids | Descriptive fixed-strength effects do not isolate feedback from force |
| Recent M1 methods (.45) | DPS, TMPD-inspired, LGD-MC, TFG, unguided and both BDG settings; full quality/time tables | Equal-w study is not a tuned-method ranking |
| M2 transfer (.4) | Same four guidance adaptations, both properties/strengths, matched fidelity and seed spread | Three adaptations collapse to one effective comparator; TFG-MC is restricted. No independent generator comparison to Dirichlet FM/Fisher Flow/MOG-DFM was run locally |
| Failure analysis (.2) | Chemistry loss, sequence mismatch, clipping, window sensitivity and baseline degeneracy | Joint useful-yield analysis requires sample-level records |
| Discussion (.325) | Three paragraphs separating finding, mechanism and limitations | Final claim is property-dependent dispersion adjustment |
| Code/docs (.525) | Single manuscript source; table/figure builder, source-hash manifest, checks, README and source map | Existing project training/evaluation code is preserved |

The DNA comparison contains four published-method adaptations, which is the
literal method-comparison design. Because several collapse numerically and the
sequence TFG port is restricted, this should not be presented to graders as three
independent competitive sequence generators. The paper states the limitation.
No score or full rubric completion is guaranteed.

## Important corrections from v5

- Own diffusion is awaiting its benchmark, not described as cancelled.
- Positive CpG findings now appear in the abstract, table, figure and conclusion.
- Table 4 no longer juxtaposes w=4 coverage with w=1 fidelity.
- The molecular scale s is the calibration-pool property standard deviation,
  assigned by the inference wrapper; it is not the raw checkpoint's y_std.
- Checkpoint inspection confirms DNA validation selection at epoch 1450,
  completed training of 1500 epochs, Adam and no EMA. Predictor seeds are 20260918.
- Wall-clock sums are 81.11 h for molecules and 5.26 h for DNA, including scoring.
  These are neither elapsed run time nor measured accelerator utilization.
- Wider spread is established against the tested positive strengths only.
  The algebra does not prove that all strength schedules fail or explain a null.
- Negative residual weakens contraction; net repulsion requires w_eff < 0.
- The discarded '11 of 84' evaluator assertion and the universal '10x window'
  assertion are absent. Baseline degeneracy and unverified activity-port
  equivalence are disclosed.

## Verification

- `paper/tools/build_results.py --check`: eight generated table blocks, all 955
  source hashes, molecular verdict counts, six widening comparisons and four
  DNA gain claims.
- `paper/tools/check.py`: 200-250-word abstract, exactly three introduction
  paragraphs, four/five contribution bullets, all 14 required sections, all float
  references, target scope and house-style checks.
- `paper/tools/pagecheck.py`: five main-text pages, no overfull boxes, no
  undefined citations or references. Template margins and typography unchanged.
- Manual visual review of main text, references, tables, plots and appendix.
- The built-in editor was opened but its compiler could not initialize
  ('Unable to find standard directories for platform'). The existing local
  LaTeX installation produced the verified PDF; no TeX software was installed.
- Closest newer references checked against their primary records:
  [MGD](https://arxiv.org/abs/2602.17211),
  [Variance-Tilted Diffusion](https://arxiv.org/abs/2606.22239), and
  [Fisher Flow](https://arxiv.org/abs/2405.14664).

## When the VP benchmark arrives

Replace Table 1's P cells with the matching three-seed benchmark; record the
selected checkpoint, training budget, sample count, sampler and hardware.
Then update the provisional FM choice in Sections 3.3/4.2, the abstract,
overview and discussion together. Keep borrowed EDMsecond separate.
