# BDG Defense, PowerPoint v3

Release date: 29 September 2026. Source: the user-supplied **BDG Defense — Group 2 v2.pptx**.
Evidence baseline: `paper/main.tex`, paper v8, Git commit `3068433`, and
`paper/results_manifest_v8.json` (973 recorded result files).

The released deck is `output/BDG_Defense_Group_2_v3_revised.pptx`.
The original file is preserved as `source/BDG_Defense_Group_2_v2.pptx`.
`manifest.json` records exact hashes, sources and counts. Earlier deck and paper versions are unchanged.

## Content and rubric map

| Slides | Rubric item | Content |
|---|---|---|
| 1 | Cover | Three current authors, project title, defense date |
| 2 | 1a | Property-band problem and both state spaces |
| 3–4 | 1b | Splits, representations, conditioning, decoded metrics |
| 5–6 | 1c | FM path, 29×8 velocity/error, masked loss, training and selection |
| 7–8 | 1d | VP noise process, objective, selected model, sampler |
| 9–10 | 1e | Fixed masks and target, uncertainty, fairness and hardware |
| 11 | 1f | Study overview |
| 12 | 2a | Completed own FM/VP benchmark and model-family decision |
| 13–14 | 2b | Guidance rule and guided/unguided comparison |
| 15–18 | 2c | Motivation, mechanism, exact reduction, both-backbone spread plots |
| 19–20 | 2d | Numeric q50 dipole ablation and supported interpretation |
| 21–22 | 2e | Four recent adaptations, quality/diversity/time, comparison scope |
| 23 | 3a | DNA transfer and training |
| 24 | 3b | DNA coverage, JSD, Hamming diversity and timing |
| 25 | 3c | Transfer contrast chart with numerical uncertainty |
| 26 | 3d | Measured failures and limitations |
| 27 | 3e | Supported conclusions and next experiments |
| 28–29 | Bibliography | Numbered references retained from v2 |
| 30–47 | Q&A appendix | EquiFM, EGNN, f_A/f_B, size protocol, cost, algorithm, DNA, decoding, full ablation grids |

The talk retains the required 1a–3e order and top-right labels. The bibliography
ends the main presentation. The appendix is backup material outside the 15-minute
talk. Plan roughly five minutes per current presenter. Notes contain source
metadata, not a presentation script.

## Evidence corrections

- Our VP results are included. Its uniqueness is 99.9%, and its B200 MIG timing is disclosed.
- EquiFM is flow matching, EDMsecond is diffusion. The model-family decision uses our matched pair.
- The FM slide explains continuous atom-type channels, coordinate/type velocities and the separately normalized masked loss on one slide.
- η = 0 removes the BDG term. τ = 0 is undefined, not the plug-in reduction.
- At η = 4, τₘ = 0.5, all six molecular tasks have higher mean coverage and lower decoded MAE than plug-in. Registered coverage contrasts remain unresolved. Higher seed variation is scoped to our FM tasks.
- The full 612-cell molecular ablation coverage grid is preserved as editable numeric tables, split by backbone. The main ablation specifies μ, q50, y and δ and reports decoded mean, MAE and spread.
- DNA keeps all four comparator rows separate. Both GC and CpG have exactly matching decoded IB for plug-in, TMPD and LGD-MC in the tested late window. Tiny differences in other metrics are explicitly scoped.
- DNA tables include Hamming, separate GC/CpG JSD and matching-strength timing. Higher coverage does not establish enhancer function.
- The early-window GC +22.82 pp figure compares **t ≥ 0 versus t ≥ 0.5 plug-in**, at w = 1. It is a decoded result, not a pre-decoding metric or an unguided contrast.
- Atom counts are the first 2,000 validation masks. Verified distribution: min 8, median 18, mean 17.865999, max 29. All sizes receive the same q50 target.
- EGNN explanation distinguishes dipole magnitude from dipole direction, explains dense masked messages and both model classes, and follows the actual mean-pool/MLP/linear-head implementation.
- Molecular decoding follows actual argmax, EDM distance thresholds, bond-order valence and RDKit sanitization code. Property scoring remains separate from bond inference and includes invalid molecules.

## Reproducible source

`source/build_v3.mjs` is the Artifact Tool builder. `source/deck_data.json` is the
numeric snapshot derived by `source/prepare_data.py` from the paper's result loader.
The versioned v2 input supplies the original theme, images and bibliography.
The builder uses the project's private `slides/.build_v8` directory for intermediates.
To rebuild, copy the builder and snapshot there and use the bundled Node runtime
with the Artifact Tool packages. Set `FINAL_NAME` to a new filename. The finalizer
refuses to overwrite an existing release. The original builder run used the local
paths encoded in the file. `prepare_data.py` requires the project's scientific
Python dependencies and all result cells.

The PPTX contains native tables and charts. Slide images were rendered from the
released PPTX and visually reviewed. Package/layout checks passed with no findings
or warnings. Native Microsoft PowerPoint execution was not tested.
