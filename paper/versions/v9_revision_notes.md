# Paper v9 and PowerPoint v4: 29 September 2026

The title remains **BDG: Batch-Dispersion Guidance for Property-Band Generation**.
The author list is Bobo Li, Henry Huang and Idea Idehpour.

This revision incorporates commits cfc5d64 and f568f68, both from 28 September
at 22:47, already merged as 498dc40. It preserves the original t >= 0.5 DNA
comparison and adds the new GC-only t >= 0.3 follow-up at w=4. It does not
invent CpG early-window results or a GC tau_m=1 follow-up. Table 4 labels both
windows; Tables S16-S18 retain the new detail and original results.

GC BDG coverage is 37.33% versus plug-in 30.25%, a +7.08-point mean difference
with paired seed SD 4.53 points. The new window removes the numerical overlap
among the four baseline means. The window was selected after diagnosis, so
this is reported as an exploratory follow-up. The original DNA 32-contrast
threshold is not silently extended to the new comparison. The n=500 window
and w=16 strength diagnostics are kept separate from n=2000, w=4 results.

The new empirical distributions show all three QM9 properties (train_a,
51,527 molecules) and both DNA properties (83,722 training sequences).
They mark q50, y +/- delta and the bandwidth formulas. DNA bins align to the
exact count lattice to avoid artificial spikes from unequal lattice occupancy.
Delta is the fixed evaluation half-width; tau is the controller spread setting.
QM9 plots use our predictor pair and do not substitute its bands for EquiFM's.

Teaching additions cover the 29x8 FM error split, continuous atom types,
EGNN symmetry and layer equations, generator/predictor roles, f_A/f_B training
and scoring, and the point-cloud-to-molecule decoder. The fixed 2,000-mask
atom-count protocol is explicit. The discussion adds size control and local
target-distribution evaluation as next steps for reducing confounding.

Earlier requested corrections remain: completed own VP benchmark, FM-family
selection rationale, numeric q50 dipole ablation, complete other-property
ablation tables, DNA Hamming, positive molecular mean gains with their seed
variation, and the valid eta=0 plug-in limit (tau=0 is undefined).

Validation: 20 generated table blocks match 995 result-cell files; histograms
match the frozen targets and preserve full observed support. Five main pages,
one reference page and eighteen appendix pages (24 total). LaTeX reports zero
overfull boxes and no undefined references or citations. All PDF pages and all
51 PPTX slides were rendered and inspected. PowerPoint structural checks,
layout checks and Artifact Tool reimport pass; native Microsoft PowerPoint
execution was not tested. The deck has 39 native tables and eight native charts.

Paper v8 and PowerPoint v2/v3 remain preserved. `v9_2026-09-29/` contains the
frozen paper, figures, source and manifests; `slides/deck_versions/
pptx_v4_2026-09-29/` contains the released PPTX and editable source snapshot.
