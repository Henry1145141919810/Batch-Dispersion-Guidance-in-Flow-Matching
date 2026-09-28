# Paper versions

`v1_*` and `v2_*` preserve the earlier sources and PDFs. The live manuscript is
now `paper/main.tex`; `paper/body.tex` points readers there. Numbered snapshots
are retained when the live paper changes. Paper version numbers and experiment
protocol version numbers are independent.

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
