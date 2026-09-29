# Manuscript v8 snapshot

Frozen 29 September 2026. Main text: 5 pages; references: 1; appendix: 17.
Read `main.pdf`; edit the live `paper/main.tex` for the next version.
This directory contains everything needed to rebuild the PDF with LaTeX/BibTeX:

```
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`SHA256SUMS.txt` inventories the frozen files. `REVISION_NOTES.md` records the
changes and validation. Historical snapshots must not be edited in place.

The tools are archived as provenance. To regenerate numerical tables/figures,
run the corresponding version of `paper/tools/build_results.py` from the
repository's live paper directory with the original `results/` trees and the
repository's result loaders. The snapshot itself does not duplicate the 973
input JSON files or their loaders. Their hashes and derived audits are preserved
in `results_manifest_v8.json`. NumPy, Matplotlib and SciPy are required.
