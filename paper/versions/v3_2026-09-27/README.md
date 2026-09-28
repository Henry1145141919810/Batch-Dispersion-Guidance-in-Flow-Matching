# Manuscript v3 — 27 September 2026

This snapshot preserves the BDG manuscript before final experimental results
are available. `P` means pending, not zero. Manuscript version numbers and
experiment protocol version numbers are tracked independently.

`main.tex` contains the manuscript, tables, diagram, algorithm, and appendix.
The bibliographies and unchanged course style are included. The retained
`figs/` directory is historical; the current diagram is drawn inside the source.
The matching numbered release is `../v3_main.pdf`. The separate explanation
and rubric map are `../v3_revision_notes.pdf` and `../v3_revision_notes.md`.

## Rebuild

With a LaTeX installation, run these commands from this directory:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Keep this numbered snapshot for tracking; make later revisions in
`paper/main.tex` and give the next release a new version number.

## Validation record

- 10 PDF pages: 5 main-text pages including all main-text floats, 1 reference
  page, and 4 appendix pages.
- No overfull boxes or unresolved citations/references in the final build.
- All 14 required Methods/Results subsections are present.
- Every manuscript and revision-note page was visually inspected.
- The live, numbered, and snapshot manuscript source/PDF files have matching
  SHA-256 hashes. The template style matches the pre-revision archive.
- `SHA256SUMS.txt` records the preserved manuscript files. This README is
  explanatory packaging and is not part of that manifest.

These checks establish document integrity and layout, not completion of the
pending experiments or satisfaction of empirical grading requirements.

## Requirement-file locations

The repository was reorganized after the source review. The assignment and
template now live under `course/`. The writing guide is now
`docs/reference/chatterjee_lab_scientific_writing_guide.md`; the revision notes
record the original path used during the review.
