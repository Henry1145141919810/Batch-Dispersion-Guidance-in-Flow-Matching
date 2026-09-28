# Defense deck: version log

The live deck is the Claude Slides artifact **"BDG Defense — Group 2"** (claude.ai → Artifacts).
Each version here is a frozen snapshot of that deck, so an older draft can always be recovered.

Every version folder holds:

| Path | What it is |
|---|---|
| `BDG_defense_vN_preview.pdf` | full render of that version (v1: 29 pages, v2: 36), for quick viewing (a local render; for submission, export PDF from the artifact itself) |
| `project/deck.json`, `project/slides/*.html` | the exact slide sources published to the artifact |
| `assets/*.png` | the equation images and the two figures the slides use |
| `assets/blob_map.json` | maps each `/_blob/<id>` in the slide HTML to its file in `assets/` |

New versions are added as new folders; nothing here is overwritten.

---

## v1 — 2026-09-26 (artifact version 3, id `1790477453-9f4e`)

First complete draft, built from `slides/handoff/`.

- 29 slides: title, 26 content slides in the mandatory 1a → 3e order, 2 bibliography slides.
- Label (1a–3e) top-right on every content slide; numbered citations with sources bottom-right; 43 red `[TODO]` markers matching `08_PLACEHOLDERS.md`.
- Figure 1 uses the fixed overview (`slides/handoff/figures/fig1_overview_v2.*`): overlapping text lines and the connector through the Diffusion box corrected.
- Bibliography entries 16–19 completed with author lists checked on arXiv.
- Reference [3] cited as Chen et al. 2025 (bibliography order).
- Slide 3b: simplex-baseline TODOs placed in the DNA column; one-line fair discussion added.
- Slide 1e-ii: four identical/none fairness rows merged into two.
- Slide 3e: "At the power we have, it does not", per `07_DO_NOT_SAY.md`.
- Still pending: every `[TODO]` (v3 headline, v3 ablation, Job A, Job B).

## v2 — 2026-09-27 (artifact version 4, id `1790537699-31fa`)

Adds a Q&A backup appendix after the bibliography; the 15-minute talk (slides 1–29) is unchanged except one cell.

- **Appendix (7 slides, labelled A1–A4, not presented):** divider; A1 EquiFM ×2 (checkpoint, hybrid path, how our arms run on it); A2 property predictors ×2 (our f_A/f_B vs TFG's guide/oracle: architecture, training, held-out accuracy, why we keep ours); A3 data split (who trains on what, why halves); A4 guidance arms ×1 (each arm's correction, what it changes, passes per cell).
- Bibliography: added [26] EquiFM (Song et al. 2023) and [27] EEGSDE (Bao et al. 2023), cited by the appendix.
- Slide 1b: the `val` row now reads "checkpoints, atom counts". δ is calibrated on 3,000 molecules from `train_a` + `train_b`, not `val` (`FULL_RUN_V3_PROTOCOL.md` §2.3).
- Appendix numbers come from `V3_PAIR_DELTA.md`, `FA_FB_ARCHITECTURE_DECISION.md`, `EQUIFM_USABILITY_AUDIT.md`, `SPLIT_PROTOCOL.md`, `FULL_RUN_V3_PROTOCOL.md` and `weights/README.md`; no q90 number appears.
- 44 red `[TODO]` markers (v1's 43 plus D-Flow's pass count in A4).
