# The paper and the defense deck

Group 2, CIS 6270 Project 1. Start here if you want to read the paper, edit it, or
find the slide content.

## Read it

`paper/main.pdf` is the current build. Ten pages: five of main text, then
references, then the appendix. The five-page limit covers the main text only.

## Edit it

**Edit `main.tex`. Nothing else.** The whole manuscript lives there as of v3, under
the lab single-source rule: prose, all tables, the algorithm, and the appendix.

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

## Check it before you share it

```
../.venv/Scripts/python.exe tools/pagecheck.py   # 5-page limit, overfull boxes, bad refs
../.venv/Scripts/python.exe tools/check.py       # rubric structure, scope, house style
```

`check.py` must say `all checks pass`. It enforces things that are easy to break by
accident: that all fourteen required sections exist, that the contribution list has
four or five bullets, that every table and figure is referenced from the text, that
no q90 number appears (this paper reports q50 only), and that no phrase our own
adversarial review refuted has crept back in.

**The main text has no slack.** It sits at exactly five pages. Adding a sentence
needs a sentence cut to pay for it, and the page count moves in jumps rather than
smoothly, because LaTeX float placement absorbs small edits. Do not try to fit
something by shrinking type or spacing; the writing guide forbids it and a reviewer
will notice.

## Pending numbers

The full run has not landed. Cells waiting on it carry a defined marker rather than
a zero, so nobody mistakes a placeholder for a measurement. `check.py` counts them.
When a number arrives, replace the marker; that is roughly length-neutral and safe.

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
    overview.tex     Figure 1, the study overview, a standalone TikZ document
    overview.pdf     its build, included by main.tex
    make_mechanism.py  builds the controller figure from measured data
    mechanism.pdf    its output
  tools/
    pagecheck.py     page limit and layout
    check.py         rubric, scope and style gate
    prosecount.py    word budget per section
  versions/
    CHANGELOG.md     WHAT CHANGED AND WHY, per version. Read this first.
    v3_main.tex      the numbered v3 release
    v3_main.pdf
    v3_2026-09-27/   rebuildable v3 snapshot with SHA256SUMS.txt
    v3_revision_notes.md   the rubric map: which item rests on which evidence
    v2_*, v1_*       earlier releases, kept
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
