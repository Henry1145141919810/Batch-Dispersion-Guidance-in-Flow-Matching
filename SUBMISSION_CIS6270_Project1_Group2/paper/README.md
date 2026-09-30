# Current paper and defense deck

The live manuscript is **paper v9**, revised 29 September 2026:
[PDF](main.pdf), [editable source](main.tex),
[revision notes](versions/v9_revision_notes.md).
Its main text is five pages; references and the appendix make 24 pages total.
The matching defense deck is [PowerPoint v4](../slides/deck_versions/pptx_v4_2026-09-29/output/BDG_Defense_Group_2_v4_revised.pptx).

Paper v9 incorporates the new GC t >= 0.3 follow-up from the two 22:47 commits.
CpG remains at t >= 0.5. It preserves the original DNA window in appendix tables,
shows empirical property distributions with q50 and fixed evaluation bands,
and includes the completed VP benchmark and expanded implementation explanations.

Edit prose in main.tex. DATA blocks come from tools/build_results.py; rebuild
them with Python, then use --check for a read-only source/table audit.
tools/distributions.py rebuilds the measured training histograms using the
local QM9 and DeepFlyBrain data. Build the multi-file document with pdflatex,
bibtex and two more pdflatex passes; run tools/check.py and tools/pagecheck.py.

The frozen [v9 snapshot](versions/v9_2026-09-29/README.md) includes source, figures,
PDF and hashes. [Earlier versions](versions/CHANGELOG.md) remain unchanged.
The historical slides/handoff files and Beamer deck predate these results;
use the versioned PowerPoint release and its rubric map for the current defense.
