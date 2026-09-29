# BDG Defense, PowerPoint v4

Released 29 September 2026. The original supplied PowerPoint v2 is preserved
in `source/BDG_Defense_Group_2_v2.pptx`. The preceding v3 checkpoint remains
in its own folder. This release aligns with paper v9 and the 22:47 DNA updates
(cfc5d64, f568f68), merged locally by 498dc40.

Open `output/BDG_Defense_Group_2_v4_revised.pptx`.

| Slides | Grading item | Content |
|---|---|---|
| 1 | Cover | Title and three current authors |
| 2 | 1a | Question and two state spaces |
| 3-4 | 1b | Data, splits, representation and decoded metrics |
| 5-6 | 1c | FM path, continuous atom types, split loss, training and sampling |
| 7-8 | 1d | VP process, training, selected checkpoint and sampling |
| 9-11 | 1e | Fixed sizes, empirical QM9 bands, fairness and uncertainty |
| 12 | 1f | Model overview |
| 13 | 2a | Completed own FM/VP and external family comparison |
| 14-15 | 2b | Plug-in guidance and guided versus unguided results |
| 16-19 | 2c | BDG mechanism, eta=0 limit, six spread curves |
| 20-21 | 2d | Numerical q50 dipole ablation and supported interpretation |
| 22-23 | 2e | Four adaptations, coverage, quality, diversity and cost |
| 24-25 | 3a | DNA transfer and empirical GC/CpG bands |
| 26 | 3b | New GC >=0.3 and original CpG >=0.5, explicitly labelled |
| 27 | 3c | Mean gains and paired seed variation |
| 28 | 3d | Four measured limitations |
| 29 | 3e | Conclusion; size and target-distribution confounding |
| 30-31 | Bibliography | Numbered sources |
| 32-51 | Q&A appendix | EGNN, f_A/f_B, decoding, grids, original/new DNA windows |

The 1a-3e order, upper-right rubric labels, source citations and bibliography
before the appendix follow the assignment. Rehearse the 15-minute main talk:
roughly 5 minutes each for three presenters, with appendix slides reserved
for questions. A possible allocation is 1-12, 13-23, and 24-29, keeping
time for the title, bibliography transition and handoffs. Notes contain source
metadata, not a reading script.

## Data and interpretation

- New GC data: t >= 0.3, w=4, three seeds of 2,000. BDG tau_m=.5 gains
  7.08 points over plug-in, with paired seed SD 4.53. CpG retains t >= 0.5.
- Original late-window GC and CpG results remain in the appendix. Baseline
  overlap is scoped to this original window. TFG-MC is labelled as a DNA port.
- Real-data distributions mark q50 and band edges for both modalities, with
  formulas. Full bandwidth is 2delta; BDG changes tau, not the evaluation band.
- eta=0, not tau=0, is the plug-in ablation. Both flow backbones show higher
  mean molecular coverage/lower MAE at tau_m=.5, with uncertainty retained.
- The own VP benchmark is complete. Cross-family runtime hardware differs.
- All previous EGNN, FM-loss, f_A/f_B, fixed-mask and decoder explanations remain.

See `docs/protocol/MODALITY2_WINDOW_ADDENDUM_2026-09-29.md` and the paper v9
manifests for exact source settings. The original protocol is not rewritten
as if the new follow-up had been preregistered.

## Versioning and verification

`manifest.json` records hashes and evidence sources. `source/build_v4.mjs`
and `source/deck_data_v4.json` reconstruct the deck using Artifact Tool and the
versioned v2 input. Copy them into the private `slides/.build_v8` folder used
by the builder; use the bundled Node runtime and set FINAL_NAME to a fresh
filename. The finalizer refuses overwrite. Source paths are local to this
workspace and must be adjusted if rebuilding on another machine.

The authoring/derivation scripts are also retained as provenance. Regenerating
numerical data needs the repository result loaders, the v9 paper builder, all
result cells and local training datasets. The frozen deck_data snapshot allows
rebuilding the presentation without recomputing those datasets.

Final PPTX reimport and package/layout gates pass with zero findings or warnings.
All 51 slides were individually rendered and inspected. Quantitative displays
include 39 native tables and eight native charts; charts contain native numeric
series, not embedded spreadsheets. IBM Plex Sans and Source Serif 4 match the
reference. Native Microsoft PowerPoint execution was not tested.
