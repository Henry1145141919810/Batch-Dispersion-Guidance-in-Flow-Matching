# v7 revision and evidence map

28 September 2026. Title: **BDG: Batch-Dispersion Guidance for Property-Band Generation**.
Current authors: Bobo Li, Henry Huang, Idea Idehpour.

The manuscript has five main pages, one reference page and fifteen appendix pages.
The original template style is unchanged. All prose and tables remain in `main.tex`.

## The eight requested changes

| Request | Completed revision and evidence |
|---|---|
| Remove Haimo | Removed from the current manuscript author list. Historical releases remain archival records. |
| Check DNA baseline overlap | Removed the blanket abstract statement. Independent per-seed checks show exactly equal decoded IB for plug-in, TMPD and LGD-MC on **both** GC and CpG, at w=1/4 and t>=0.5. TFG-MC differs. Section 4.6 and Appendix A.6 state this restricted finding. |
| Show Hamming diversity | Main Table 4 now shows normalized Hamming distance for GC/CpG beside matching-strength coverage and JSD. It remains near 0.7456-0.7457. |
| Justify model-family choice | The newly arrived own-VP data now fill Table 1 and Table S6. Our FM has 39.7% stability and 75.6% validity, versus VP's 28.8% and 66.2%. EquiFM is explicitly a flow model and EDMsecond a diffusion model; their quality comparison supplies external context. |
| Numeric Table 2 | Uses our FM dipole moment at q50 y=2.4932 D, half-width 0.17541 D, w=1. Shows eta=0, eta=4 at tau_m=.5/1, and eta=8 at tau_m=.5/1.5, including IB with seed SD, MAE, generated mean and spread. Tables S8-S13 contain all 204 configurations x 3 seeds = 612 cells, including the other properties and both strengths/backbones. |
| Positive, accurate Table 3 interpretation | Leads with higher mean IB and lower decoded MAE across all six molecular headline comparisons at eta=4, tau_m=.5. Our FM has higher seed variability than plug-in. The molecular IB differences still fall below the registered threshold; that uncertainty remains stated. Tau_m=.5/1 clearly changes occupancy and stability. |
| Favor measured BDG benefits | Abstract, results and discussion emphasize adjustable spread, band occupancy, lower mean absolute target error, and positive CpG transfer. Dipole ablation means move closer to target. We avoid asserting universal improvement of signed bias, faster convergence, or exact terminal variance. |
| Highlight reduction to plug-in | Corrected the parameter: **eta=0**, for any positive tau, removes the added term. **Tau=0 is undefined**, since e=(V-tau^2)/tau^2. Main methods, Table 2, Table 3's plug-in label/caption, and the algorithm appendix make this explicit. |

## Newly arrived diffusion results

- 18 final cells: three properties, unguided/plug-in, three seeds of 2000.
- Local checkpoint MD5 `8a3390a693ed39ebad154b0640145acc` matches every cell.
- Selected epoch 1475, family `vp_diffusion`, EMA, training split `train_a`,
  seed 20260918, eight layers of width 256, batch 256, Adam 2e-4, cosine floor
  5%, EMA .9999, configured budget 1500 epochs.
- The evaluated FM export is epoch-1500 EMA. The VP export does not itself prove
  completion of all 1500 epochs; checkpoint-selection differences are disclosed.
- VP evaluation used B200 MIG 2g.45gb; other molecular evaluation used RTX A6000.
  Table 1 explicitly flags that runtime is not hardware matched.
- Molecular measured wall time now sums to 81.9242 hours over 774 cells,
  including scoring and 0.8128 hours for VP. This is not elapsed wall time or GPU utilization.

## Numerical interpretation

Contracting BDG's molecular mean IB gains over plug-in range from 0.45 to
1.52 percentage points. All six decoded MAEs are lower. This statement concerns
eta=4, tau_m=.5, w=1 headline cells, not every setting in the full grid.

For the selected dipole ablation, the zero-gain control has MAE 0.969 D and
pooled mean 2.615 D. Eta=4, tau_m=.5 gives 0.903 D and 2.609 D; eta=8 gives
0.861 D and 2.603 D. The target is 2.4932 D. These are endpoint improvements,
not a measurement of convergence speed.

Relative to plug-in, late-window TMPD/LGD-MC differences across property mean,
property SD, JSD and Hamming are bounded by 4.22e-6 for GC and 3.40e-6 for CpG.
TFG-MC has single-seed IB differences up to 0.05 percentage points for GC and
0.75 for CpG. The source-field names `gc_mean`/`gc_sd` are reused for the selected
sequence property by the evaluation code.

## Rubric and validation

- Preserved all fourteen required subsections, exactly three introduction
  paragraphs, four contribution bullets and a 214-word abstract.
- Main text includes model selection, guided/unguided comparison, numeric
  innovation ablation, recent adaptations, second-modality transfer and failure analysis.
- Main Table 3 keeps uncertainty and molecular quality alongside coverage.
  Main Table 4 adds diversity beside coverage and sequence fidelity.
- Sixteen generated table blocks match the source cells; 973 result-file hashes
  are recorded in `results_manifest_v7.json`. Regenerated all three vector figures.
- Verified the positive coverage/MAE claims, all six widening comparisons,
  own FM/VP quality, per-property DNA baseline identity, and four DNA coverage gains.
- PDF compiled with installed LaTeX and all 21 pages visually inspected.
  Five main pages, no overfull boxes, no undefined references or citations.
- The built-in editor compiler reported a platform-directory error; the editable
  source is preserved and the local compiler produced the verified PDF.

The measured DNA baselines remain adaptations, not reproductions of Dirichlet
FM, Fisher Flow or MOG-DFM. Joint molecular yield and batch-aware confidence
intervals still need sample-level records. This revision does not turn those
unperformed analyses into completed claims.
