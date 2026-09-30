# Paper versions

In the repository, `paper/versions/` keeps every numbered snapshot; `v1_*` and
`v2_*` there preserve the earliest sources and PDFs. The live manuscript is
now `paper/main.tex`; `paper/body.tex` points readers there. Numbered snapshots
are retained when the live paper changes. Paper version numbers and experiment
protocol version numbers are independent.

## v10 - 29 Sep 2026

A correctness pass over v9, not a new experiment. Authors: Bobo Li, Henry Huang.

- **The DNA headline reversed, and the paper now says so.** v9's table was
  captioned "t >= 0.3 for both properties" but its CpG column still held the
  t >= 0.5 numbers. At t >= 0.3, BDG is +7.08 points on GC and −4.88 on CpG
  (Table 5), not +16.83. The abstract, 4.6, 4.7 and the discussion now state the
  sign rule that explains both: BDG contracts only a batch wider than `tau`.
- **The sign rule is tested.** The `tau_m = 0.25` CpG probe (three seeds) contracts
  spread to 0.298 s and recovers +2.28 points, every seed agreeing in sign. It is
  reported as exploratory in Table S17 and kept out of every contrast family.
- **New tables.** Table 2, guided versus unguided on our FM and VP (4.3 had prose
  and no table), and Table S20, the complete t >= 0.3 DNA grid. Table S16 gains a
  CpG panel; Table S19 states its window, t >= 0.5.
- **Stale claims corrected.** The follow-up covers both properties at w ∈ {1, 4};
  the CpG t >= 0.3 and GC `tau_m = 1` cells are reported, not "not run"; the
  `eta = 0` tolerance matches the code; the reproduction map (Table S21) adds the
  setpoint-probe stage.
- **Tables mark best and runner-up** wherever one direction is better, ranked
  within each property panel, with ties left unmarked.
- **First compiled build of this source.** 0 errors, 0 undefined references,
  0 overfull boxes; `tools/check.py` passes. 23 generated table blocks and 1,073
  source hashes pass `build_results.py --check`; 27 pages overall.

Revision notes and frozen snapshot, in the repository: `paper/versions/v10_revision_notes.md`, `paper/versions/v10_2026-09-29/`.

## v9 - 29 Sep 2026

Adds the two 22:47 commits: GC guidance starts at t >= 0.3 in the new follow-up;
CpG remains at t >= 0.5. Original comparisons are retained. Adds measured
property distributions, q50 and bandwidth formulas for both modalities,
expanded EGNN/decoder explanations, fixed atom-count protocol, and next steps
on atom-count and target-density confounding. Twenty generated table blocks,
995 source cells; five main pages and 24 pages overall. Paper v8 is unchanged.

Revision notes and frozen snapshot, in the repository: `paper/versions/v9_revision_notes.md`, `paper/versions/v9_2026-09-29/`.
The matching defense deck is PowerPoint v4, a separate version sequence.

## v8 - 29 Sep 2026

Further manuscript revision informed by the supplied defense deck and checked
against the final result cells. Five main pages, one reference page and seventeen
appendix pages; the previous live files are preserved in `v7_pre_v8_2026-09-29/`.

- Clarified the study's logic and restored the three-paragraph Discussion.
- Retained own-VP results and compared guidance on both trained model families.
- Expanded the selected dipole ablation with stronger plug-in and stability.
- Added main-table diversity/yield and runtime context, plus separate neural-call
  counts in Table S3.
- Redrew the overview and plotted all six molecular spread curves.
- Added exploratory stability-exchange ratios and coefficient/coverage diagnostics,
  separating eta = 4 from eta = 8 and descriptive findings from causal claims.
- Regenerated eighteen table blocks and three vector figures; verified all 973
  source hashes. Removed the duplicate QM9 bibliography entry.
- All 23 PDF pages visually reviewed; five-page main-text limit and layout,
  citation, structure and numerical checks pass without changing template style.

See `v8_revision_notes.md` for the evidence audit, rubric map and preserved limits.
The rebuildable snapshot is `v8_2026-09-29/`, with SHA-256 checksums.

## v7 - 28 Sep 2026

Revised all eight requested points and incorporated the newly arrived own-VP
benchmark. The paper remains five main pages, with one reference page and
fifteen appendix pages. Previous releases remain preserved.

- Removed Haimo Fang from the current author list.
- Replaced the abstract's broad DNA-collapse statement with the measured BDG
  findings; separately checked both GC and CpG and distinguished TFG-MC.
- Added Hamming diversity to main Table 4 at matched strength.
- Filled our VP results and justified carrying FM forward from stability and
  validity. Explicitly labeled EquiFM as flow and EDMsecond as diffusion.
- Replaced qualitative Table 2 with numeric dipole ablations at q50 = 2.4932 D.
  Added all 612 molecular ablation cells as six complete appendix tables.
- Emphasized BDG's six positive mean-coverage and lower-MAE comparisons,
  while retaining seed variation and the registered uncertainty result.
- Highlighted the correct zero-gain limit: eta = 0 gives plug-in; tau = 0 is
  undefined. Setpoint changes demonstrably adjust spread and occupancy.
- Rebuilt sixteen inline table blocks and three vector figures, with hashes
  and audit results for 973 input files. All pages visually reviewed; no
  overflowing boxes or unresolved references.

See `v7_revision_notes.md` for the request-by-request evidence map.

## v6 - 28 Sep 2026

**Full evidence and presentation revision using the completed local v3-final
run. Our trained VP benchmark remains pending.** The prior live v5 source/PDF
is preserved in `v5_pre_v6_2026-09-28/`.

- Rewrote the abstract, introduction, methods, interpretation and discussion.
  Removed the false cross-modality null: contracting BDG has measured CpG gains.
  Molecular differences remain unresolved under the declared test; this is not
  a proof of equivalence or a mechanistic prediction of failure.
- Restored the own-VP benchmark row as pending and removed the assertion that
  the user's comparison was cancelled. Borrowed EDMsecond remains separate.
- Generated eight inline numerical table blocks from the final cells, with
  seed variation and separate strength/backend definitions. DNA coverage and
  fidelity now come from the same strength; both GC and CpG are visible.
- Replaced the overview and added a final-run two-panel result graphic plus
  complete molecular ablation heatmaps. The old pilot mechanism graphic is not
  used. All 955 source files have hashes in `results_manifest_v6.json`.
- Corrected the molecular guidance scale to the calibration-pool standard
  deviation, verified the DNA checkpoint's best epoch (1450 of 1500), filled
  predictor training seeds, and separated solver NFE from actual neural calls.
- Removed GPU-hour, universal-strength, feedback-necessity and across-property
  window claims that the data do not establish. Added coupled-batch and rerun
  uncertainty limitations; threshold-sized differences are not called power.
- Described DNA baselines as implemented guidance adaptations. Their numerical
  collapse is explicit; unrun sequence generators are no longer empty result rows.
- Preserved five main-text pages with the unmodified template. Expanded the
  appendix, verified all references/float links, and inspected rendered pages.

See `v6_revision_notes.md` for the grading map and exact remaining limitations.
`v6_2026-09-28/` is the rebuildable source/PDF snapshot.

## v5 - 28 Sep 2026

**The run landed, so the paper reports it.** Every result now comes from the
v3-final blade cells: `results/v3/{fm,equifm,edm}/{v3,v3abl}/n2000/` (756 cells)
and `results/m2/` (199), seeds 20261001/2/3 and 20260921/22/23, n = 2,000 per
cell pooled to 6,000 per arm, target q50, window t >= 0.5, 100-step Euler,
81.2 GPU-hours. Five pages, 0 overfull boxes, `tools/check.py` passing.

`docs/results/PAPER_TABLE_FILL_V3.md` section 1b recomputes every number in the
paper from the raw cells. **Tables: 93 of 93 correct.** It found five prose
defects, all fixed here.

### The headline changed

BDG does not beat the arm it modifies. Against plug-in, which is BDG at
`eta = 0`, twelve contrasts (two setpoints x three properties x two generators)
give **no gain at the pre-registered bar, three losses and nine ties**.

### One run, not two

The earlier n = 5,000 execution is dropped entirely. Its flow-matching cells were
*guided* by a different predictor pair, and its band width on alpha differs by a
factor of 3.07, so it was never comparable with the tables it sat beside. It also
carried a BDG setpoint later dropped from the design, which is why its contrast
count was 18 where the final run gives 12.

**Useful yield goes with it.** That metric needs per-molecule sidecars which
exist only for the dropped run, so the paper reports decoded band coverage
throughout. One consequence: TFG is **5 of 6** rather than 6 of 6, since
EquiFM/mu is a tie at z = +1.87.

### What the ablation does support, which the paper was missing

Coverage does not improve, but the setpoint reaches operating points the strength
dial cannot. Generated by `proj1/scripts/paper_controllability.py` into
`docs/results/PAPER_CONTROLLABILITY.md`:

- The expanding setpoint widens the batch past unguided, **R from 1.15 to 1.25,
  in 6 of 6 cells**. The strength dial at w = 4 reaches R > 1 in **0 of 6**; it
  only ever contracts. This is categorical, both generators.
- On our generator the setpoint is also the cheaper dial, buying 0.52 to 0.81
  coverage points per point of molecule stability against the strength dial's
  0.38 to 0.53. On the borrowed generator the two are indistinguishable, so the
  claim carries that qualifier.

### Claims the run does not support, now stated as such

- **The matched flow-against-diffusion comparison never ran.** Job 8501085 was
  cancelled for a wrong split and never restarted; no VP checkpoint exists in the
  repository. The only `family: vp_diffusion` artefacts are 16,231-parameter
  stubs written by `results/bench/audit4/mkfake.py` with hard-coded metrics, for
  auditing whether the harness trusts a checkpoint's self-report. The v3 `edm`
  backend is TFG's released EDMsecond. Table 1 carries three honestly-labelled
  generators and no VP row.
- **Plug-in at w = 4 never ran.** The strength axis exists only inside the
  ablation; that row is gone from the guidance table.
- Three of five ablation controls did not run: a one-sided residual, an open-loop
  replay, a signed-strength plug-in. Without the replay the grid cannot show the
  feedback is necessary, only that the deviation weight is what moves coverage.
- The three external sequence comparators have no cells.

### Prose defects the fill-check caught

| Was | Now |
|---|---|
| `tab:m2` mixed strengths in one column: unguided and plug JSD at w = 1, BDG's at w = 4 | all at w = 1; BDG's is 0.00049 |
| `tab:m2` showed only the better BDG setpoint | both, which the project's own rule requires |
| the DNA window "roughly ten times" the rule | **about five times**; 10x holds only against BDG's w = 4 gain |
| "12 of 12 past the bar", "5 of 12 all at w = 1" | those are *continuous* counts; on the decoded coverage the paper defines, 11 of 12 and four at w = 1 |
| Modality 2 quoted GC only | CpG at w = 1 is +4.82 points and clears the bar |

Two more caught in review: section 4.3 said guidance raises coverage on *every*
generator and property, where EquiFM/alpha is -0.0002 (now 8 of 9); and the
Discussion still credited "the two gains attributed to BDG", of which there are
now none.

### Statistics

Section 4.1.3's three assertions were all false of the run, confirmed three ways.
Seeds are *pooled* into one binomial proportion, not averaged with a standard
deviation; contrasts are *unpaired* with a separate-variance standard error; and
the batch is not the unit of uncertainty. Rewritten, and the paper now states the
pre-registered bar `z = 2.99` (alpha = 0.05 over eighteen contrasts, two-sided)
and its 1.56-point resolution, where before it stated no threshold at all.

Files: `v5_main.tex`, `v5_main.pdf`, and `v5_2026-09-28/` with `SHA256SUMS.txt`.

## v4 - 28 Sep 2026

**Calibrated against the course paper template**, section by section, after a
conformance audit of all 14 template pages. Five pages of main text, 0 overfull
boxes, 0 undefined references, `tools/check.py` passing.

### The conditional-flow-matching question, settled from the code

`proj1/scripts/train_fm.py` uses `x_t = t*x_1 + (1-t)*eps` with target
`u = x_1 - eps`. That **is** the conditional flow-matching objective, the
template's own Equation (2): the target is the velocity conditioned on the
endpoint pair. The paper wrote the equation without naming it, which read as if
we had used something non-standard. Section 3.2 now names it and states that the
conditioning is on the pair, not on a property.

Separately, the trainer records that *"atom count is not generated, we condition
on a mask drawn from the data"*. That is a conditioning variable and the template
requires any such variable to be defined precisely. Section 3.1 now states that
generation is unconditional in the property sense and defines the atom-count mask
`M`; the fields are written `v_theta(x_t,t;M)` and `epsilon_phi(x_u,u;M)`, which
also resolves a contradiction where 3.1 declared a condition and 3.2 denied one.

### Section 4.1, rebuilt to the template's structure

It covered one modality, gave no compute budget and no statistical reporting. It
now has the three required subsubsections: Metrics (both modalities, quality plus
diversity plus controllability, each with a stated limitation), Fair Comparison
and Computational Budget (what is matched, and the mismatches stated rather than
glossed), and Statistical Reporting (three seeds, mean and standard deviation,
paired differences, and the batch as the unit of uncertainty because BDG couples
samples within one).

### Section 3.7 and Figure 1

Section 3.7 was a single sentence and the figure showed neither the source
distribution, the two pathways, the guidance signal, nor the pretrained
components. The figure was redrawn to carry all of them with a trained / frozen /
pretrained key, and it now sits on the same page as Section 3.7. Panel (b) had a
solid arrow from the trained simplex generator to the analytic property map,
which asserted that the generator produces that map; it now runs generator to
frozen backbone, mirroring panel (a).

### Also fixed

- Section 3.5 is the largest rubric item and was missing both bookends the
  template mandates: it opened on a definition and closed on a cost remark. It
  now opens with the weakness and the hypothesis and closes with the three
  controls that isolate the innovation.
- Five foundational works the template names were sitting in the bibliography
  **uncited**: stochastic interpolants, SiT, rectified flow, classifier-free
  guidance and Guided Flows. A guidance paper that never cites classifier-free
  guidance is a visible gap. All five are now cited.
- Section 4.2 never stated which family is carried forward or why.
- Section 4.7 delivered none of its synthesis duties and did not end with an
  answer to the paper's central question.
- The Discussion had two of three mandated paragraphs; the missing one is the
  limitations paragraph, which the rubric prices separately. It now carries two
  limitations, the experiment that would settle each, and the narrowest claim the
  evidence supports.
- Table 4 had no internal base-model row, which the template requires beside the
  external methods. Table 3 had no guidance-strength axis.
- Appendix floats now number S1, S2, ... as the template sets them.
- `tools/check.py` counted em dashes inside LaTeX comments, so TikZ section
  banners tripped it. Style checks now run on rendered text only.

### Cost

The template-required additions ran the main text to a full six pages. It is back
to five, paid for by moving training settings, protocol constants, split
arithmetic and metric limits into the appendix, which does not count toward the
limit. One regression caught in review: a compression pass cut the abstract to
140 words, under the template's 200 floor. It is 205.

Files: `v4_main.tex`, `v4_main.pdf`, and `v4_2026-09-28/` with `SHA256SUMS.txt`.

## v3 - 27 Sep 2026

**BDG selected; final results pending.** Revised against the assignment rubric,
paper template, Chatterjee Lab guide, current protocols, and implemented method.

- BDG is the sole proposed innovation; BTVG is absent from the current manuscript.
- Removed final-performance and transfer conclusions inferred from older pilots.
  Unreleased comparisons use a defined `P` marker. No experimental results were invented.
- Corrected the setpoint scale, net-widening condition, zero-gain factorization,
  one-sided identity condition, detached pullback, and scope of efficiency claims.
- Aligned the molecular description with q50, w=1, batch 500, three seeds,
  operator-selected n, and no chemistry-floor exclusion.
- Updated DNA to the shipped 500-bp DeepFlyBrain setup. Final sequence settings
  and the three external comparisons remain explicitly pending.
- Restored the exact introduction structure and retained all required sections.
  Consolidated prose, tables, overview, algorithm, and appendix in main.tex.
- Fixed BibTeX parsing and verified active reference records; corrected EquiFM's
  title, used the published Dirichlet FM reference, and added the QM9 source.
- Five-page main text, standard template typography; references and appendix
  follow. The separate rationale maps each rubric item to the remaining evidence.

Files:

- `v3_main.pdf` and `v3_main.tex`: numbered manuscript release.
- `v3_2026-09-27/`: complete rebuildable source snapshot and PDF, with hashes.
- `v3_revision_notes.md` and `v3_revision_notes.pdf`: explanation and rubric map.
- `v2_pre_v3_2026-09-27/`: exact live source, PDF, bibliographies, style, and
  figures found before this revision. The original v1/v2 snapshots are unchanged.

The built-in compiler could not initialize; the existing local LaTeX installation
produced the PDF. Validation includes compilation, five-page enforcement,
resolved citations/references, float references, house-style checks, and a visual
review of every manuscript and rationale page. This validates the draft's form,
not completion of its pending empirical requirements.

## v2 — 26 Sep 2026

Three adversarial passes were run against v1: a rubric grader, a hostile reviewer
checking every claim against the repository's own result files, and an audit against
the lab writing guide. A fourth pass verified the five unattributed citations.

### Claims v1 made that its own data contradicted

Each of these was verified against the file named before it was changed.

| v1 said | The data says | Source |
|---|---|---|
| ladder reaches `1.219` / `1.22`, on "six of six curves" | `1.219` is **mu at q90**; this study reports q50 only, where the range is `0.670`–`1.187` over **three** properties | `results/bdg_port/table.txt` |
| "No rung beats plug-in at \|z\|>=3: the largest is 3.07" | the max is **-3.07**, a BDG **loss** (gap q50, e4t1.5); the sentence also refuted itself, since 3.07 >= 3 | `docs/results/BDG_FULL_METRICS.md:549` |
| "more chemistry at matched coverage" as the headline win | `z = 1.73`, `p = 0.083`, one seed; and **both** cells fail the project's own chemistry floor of 0.390 | `results/bdg_port/table.txt` floor column |
| clip counts `884` vs `2,516` | that is a different run (n=128, seed 20261001) from the cells it is said to confound; the matched pair is **701** vs **4,457** | `docs/results/ONEGEN_FULL_METRICS.md` |
| batches sit "one or two" band widths off, spread "five to nine" | `0.55`–`3.03` and `5.95`–`16.37` | `BDG_FULL_METRICS.md` spread-vs-bias lever |
| Real QM9 `0.9940 / 0.9560 / 0.9820` | `0.994 / 0.956 / 0.982`; the fourth digit was introduced by v1's column-alignment pass and is invented | `docs/results/BASE_MODEL_BENCHMARK.md:215` |
| guidance raises in-band `1.9x` on gap, TFG `6.0x` | `1.81x` and `5.94x` | Table 2's own printed values |
| Modality 2 "every signature reproduces" | the widening rung reaches `1.006`–`1.008`, inside the `1.010` ratio the team's own review calls indistinguishable from zero | `docs/methods/BDG_REVIEW.md:210` |
| reduction verified to `6.5e-7` | the cited check reports `6e-7` | `BDG_REVIEW.md:62` |

### Also corrected

- Figure 1 coloured the diffusion model as **external pretrained**. We trained it.
- Three-seed error bars were present in the original Table 1 and were dropped by v1's
  alignment pass. Restored.
- `\pend` pointed at Appendix A.6, which is Algorithms. Now A.7.
- Table 3's caption claimed an ordering by `w_eff`, a column the table does not have,
  and never named its property (mu).
- BDG's own row carried Status "reproduced".
- Eight floats, including four of the five tables and Algorithm 1, were never
  referenced from the text. `tools/check.py` now fails the build if that recurs.
- Five citations had no authors. All five verified against arXiv; **two had wrong or
  truncated titles**. `CFG-Ctrl`'s recorded subtitle does not exist, and
  `ensemblevar2025` is a climate-downscaling paper that sets spread by DDIM step count,
  so it no longer sits in the near-our-mechanism sentence.
- Guide-audit fixes: two rhetorical questions, personified methods, present-tense
  results, "demonstrate" without a test, a claim-colon-slogan pattern, and a citation
  bundle attached to no claim.

### Scope and disclosures added

- The chemistry floor is now defined in the protocol and reported in Table 3.
- Spread ratios are disclosed as scored by the guide `f_A`, in-band by held-out `f_B`.
- Modality 2 inherits the `t >= 0.5` window, where GC batch spread is nearly flat; that
  is now stated rather than silent.
- Table 4's pass counts are defined as network calls at batch granularity, and the text
  no longer implies a cost comparison the generator column contradicts.
- The reduction now carries the answer to "then this is a reparameterisation".

### Cost

Main text went 5.00 -> 5.65 pages under these additions and was brought back to 5.00 by
cutting ~330 words of prose that the appendix or another section already carried. No
rubric line item lost its evidence; `tools/check.py` verifies the structural ones.

## v1 — 26 Sep 2026

First complete draft. Five-page main text, five tables, one figure, appendix A.1–A.7.
