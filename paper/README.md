# The paper and the defense deck

Group 2, CIS 6270 Project 1. Start here if you want to read the paper, edit it, or
find the slide content.

## Read it

`paper/main.pdf` is the current **v8** build, revised on 29 September 2026.
The main text is five pages; references and the expanded appendix follow.
The five-page limit covers the main text only. See
`versions/v8_revision_notes.md` for the changes, factual corrections and grading map.

## Edit it

**Edit manuscript prose in `main.tex`.** All prose, tables, the overview,
algorithm and appendix remain in this one source. Eighteen table blocks are
regenerated between `% BEGIN DATA` / `% END DATA` markers by
`tools/build_results.py`; edit that builder when changing those table layouts.
Figures and numerical tables read the same final result cells.

`body.tex` is a two-line pointer that says so. It is not the paper any more. If you
open it expecting prose, that is why it is empty.

```
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Run it twice after bibtex or cross-references come out as `??`.

Before compiling, from the repository root, regenerate the measured figures and
tables with `python paper/tools/build_results.py` (NumPy, Matplotlib and SciPy required).
Then run `python paper/tools/build_results.py --check` for a read-only check of
all eighteen table blocks, the key numerical claims, and all 973 source hashes.
The builder reads only the final `n2000` trees and keeps strengths, backends,
stages, seed identities and the early-window diagnostic separate.

## Check it before you share it

```
../.venv/Scripts/python.exe tools/pagecheck.py   # 5-page limit, overfull boxes, bad refs
../.venv/Scripts/python.exe tools/check.py       # rubric structure, scope, house style
```

`check.py` must say `all checks pass`. It enforces the 200-250-word abstract,
exactly three introduction and Discussion paragraphs, and things that are easy to break by
accident: that all fourteen required sections exist, that the contribution list has
four or five bullets, that every table and figure is referenced from the text, that
no q90 number appears (this paper reports q50 only), and that no phrase our own
adversarial review refuted has crept back in.

**The main text has no slack.** It sits at exactly five pages. Adding a sentence
needs a sentence cut to pay for it, and the page count moves in jumps rather than
smoothly, because LaTeX float placement absorbs small edits. Do not try to fit
something by shrinking type or spacing; the writing guide forbids it and a reviewer
will notice.

## Evidence status

The local final run has landed: 774 molecular and 199 sequence result files.
Our own VP diffusion benchmark is included in Table 1 and Table S7. Its selected
epoch-1475 EMA checkpoint matches the result-cell fingerprint. FM has higher
stability and validity, supporting the choice to carry FM forward. VP was sampled
on B200 MIG; the other molecular models used RTX A6000, so runtime is not matched.
Both models specify a 1500-epoch budget; their evaluated checkpoint selection differs.
Borrowed EquiFM (flow matching) and EDMsecond (diffusion) remain separate references.

The completed DNA comparisons are ports of DPS, TMPD, LGD-MC and restricted
TFG-MC. DPS/TMPD/LGD-MC have identical decoded coverage on both GC and CpG
in the measured late window; TFG-MC differs. Dirichlet FM, Fisher Flow and MOG-DFM are
background references, not evaluated generators. Joint molecular useful yield
and batch-aware intervals require sample-level files on the source machine.
The correct plug-in ablation is **eta = 0**, with positive tau. Tau = 0 is undefined.
All 612 molecular ablation cells are represented in numeric Tables S10-S15.
Table S3 separates sampling operation counts; Table S9 reports exploratory
coverage gained per stability point lost, distinguishing eta = 4 from eta = 8.

## Current graphics

- Figure 1: the overview drawn directly in `main.tex`.
- Figure 2: `figs/results_v8.pdf`, all six molecular spread curves and DNA coverage gains.
- Supplement: `figs/ablation_fm_v8.pdf` and `figs/ablation_equifm_v8.pdf`, all 12 grids.
- `results_manifest_v8.json`: source hashes, seed contrasts, target-error and DNA identity audits, plus exchange ratios and coefficient/coverage diagnostics.

`figs/mechanism.pdf` and its older builder use pilot data and are historical;
they are not included in v8. Earlier standalone overview files are also historical.

## Files

```
paper/
  main.tex           THE MANUSCRIPT. Edit this.
  main.pdf           current build
  body.tex           a pointer to main.tex; not the paper
  citation.bib       course-supplied references
  refs_extra.bib     references we added; every entry verified against arXiv
  neurips_2026.sty   template style file, unmodified
  figs/
    results_v8.pdf   Figure 2, generated from final cells
    ablation_*_v8.pdf  complete molecular ablation heatmaps
    overview.*      historical standalone versions; live overview is in main.tex
    mechanism.*     historical pilot-based figure, not used
  tools/
    pagecheck.py     page limit and layout
    check.py         rubric, scope and style gate
    prosecount.py    word budget per section
  versions/
    CHANGELOG.md     WHAT CHANGED AND WHY, per version. Read this first.
    v8_main.tex      the numbered v8 release
    v8_main.pdf
    v8_2026-09-29/   rebuildable v8 snapshot with SHA256SUMS.txt
    v8_revision_notes.md   corrections, deck audit and rubric map
    v7_pre_v8_2026-09-29/  exact files before this revision
    v7_*, v6_*, ...  earlier releases, kept
    v2_pre_v3_2026-09-27/  the exact source as it stood before the v3 revision
```

## Which script produced which table

Appendix A of the paper carries this same map, so a reader never has to leave the
paper to find what produced a number. The repo-root `README.md` has the fuller
version with the input artefacts.

## The defense deck

The deck itself is a Claude artifact; Henry has the link. This repo holds the
**ingredients**, so the deck can be rebuilt or corrected without guessing.

```
slides/
  handoff/                  THE CONTENT SPEC. Feed these to the deck builder.
    00_START_HERE.md        read order and what the deck must achieve
    01_DECK_SPEC.md         structure, slide count, visual rules
    02_CONTENT_part1_setup.md      problem, data, both base models
    03_CONTENT_part2_modality1.md  guidance, BDG, the ablations
    04_CONTENT_part3_transfer.md   Modality 2, limitations, close
    05_BIBLIOGRAPHY.md      every citation the deck may show
    06_SPEAKER_PLAN.md      who says what, with timings
    07_DO_NOT_SAY.md        claims our own data does not support. Read it.
    08_PLACEHOLDERS.md      numbers still pending, and the verified ones
    figures/                figure files at 300 dpi, PNG and PDF
  README_SLIDES.md          how the earlier Beamer deck was built
  defense.tex, defense.pdf  a superseded Beamer version, kept for reference
  deck_versions/            earlier deck builds and their extracted assets
```

**`07_DO_NOT_SAY.md` is the important one.** Three adversarial passes found nine
claims in the v1 paper that our own result files contradicted. The corrections are
in `08_PLACEHOLDERS.md` and in the deck content files. If you rebuild the deck from
an older copy of these notes you will reintroduce those errors, so start from what
is here.

The numbers most often got wrong, and the right values:

| Do not say | Say |
|---|---|
| ladder "six of six curves", `0.670`-`1.219` | three of three q50 curves, `0.670`-`1.187`. `1.219` is a q90 number and this paper is q50 only |
| "largest \|z\| is 3.07", implying a near-win | the largest effect is a BDG **loss** at `z = -3.07` |
| "more chemistry at matched coverage" | unresolved: `z = 1.73`, one seed, and both cells fall below our own chemistry floor |
| clip counts `884` vs `2,516` | `701` vs `4,457`; the other pair is from a different run |
| Real QM9 `0.9940 / 0.9560 / 0.9820` | `0.994 / 0.956 / 0.982`; only three significant figures exist |
