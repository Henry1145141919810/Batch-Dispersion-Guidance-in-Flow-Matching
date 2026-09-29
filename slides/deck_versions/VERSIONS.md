# Defense deck: version log

## Current PowerPoint release: PPTX v4, 29 September 2026

[Open revised PowerPoint](pptx_v4_2026-09-29/output/BDG_Defense_Group_2_v4_revised.pptx)
([release record](pptx_v4_2026-09-29/README.md)). Matches paper v9.
Adds empirical property distributions with q50 and band formulas, new GC
t >= 0.3 results, and explicit original/new window comparisons. All previous
FM-loss, EGNN, predictor, decoding and benchmark revisions are retained.
51 slides: 31 main/bibliography plus 20 appendix. Prior versions preserved.

## Previous PowerPoint checkpoint: PPTX v3, 29 September 2026

[Open revised PowerPoint](pptx_v3_2026-09-29/output/BDG_Defense_Group_2_v3_revised.pptx)
([release record](pptx_v3_2026-09-29/README.md)).

This release starts from the user's **BDG Defense — Group 2 v2.pptx** and follows
paper **v8**, not the old pending-data LaTeX deck. It contains 47 slides:
the original 29-slide main/bibliography sequence plus 18 appendix slides.
The supplied v2, audited numeric snapshot and editable builder are preserved in
the release folder. PowerPoint v3 is a separate version sequence from the older
Claude Slides artifact versions below. Previous releases remain unchanged.

The earlier live deck was the Claude Slides artifact **"BDG Defense — Group 2"**.
The following entries are historical snapshots of that artifact.

> ### ⚠️ Standing note, 28 September 2026 — read before trusting any "pending" below
>
> **Our own VP diffusion model is trained and benchmarked.** `weights/diff_ema.pt`,
> md5 `8a3390a6`, selected epoch 1475 of 1500; 18 v3 cells. Every entry below that
> calls the VP benchmark, "Job A" or the matched FM-vs-VP row *pending* is a record
> of what was true when that version was frozen — **the data is no longer pending.**
>
> Result: **flow matching beats our own VP diffusion on all three chemistry
> metrics** at matched backbone, parameters, epochs, batch, EMA, split and seed
> (mol stab 0.3970 vs 0.2883, validity 0.7562 vs 0.6617, atom stab 0.9356 vs
> 0.9070). Canonical source:
> [docs/results/DATA_INDEX.md](../../docs/results/DATA_INDEX.md) §3.
>
> **The deck source itself is NOT yet revised**: `slides/defense.tex` still carries
> `\TODO{v3 Job A}` / `\TODO{v3 A}`. The handoff docs under `slides/handoff/` **are**
> corrected, so they and the deck currently disagree. Fix the deck before rehearsing.

Every version folder holds:

| Path | What it is |
|---|---|
| `BDG_defense_vN_preview.pdf` | full render of that version (v1: 29 pages, v2–v3: 36), for quick viewing (a local render; for submission, export PDF from the artifact itself) |
| `project/deck.json`, `project/slides/*.html` | the exact slide sources published to the artifact |
| `assets/*.png` | the equation images and the two figures the slides use |
| `assets/blob_map.json` | maps each `/_blob/<id>` in the slide HTML to its file in `assets/` |
| `figure_source/` (v3 on) | the LaTeX/TikZ source of the figures we drew, so they can be re-rendered |

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

## v3 — 2026-09-27 (artifact version 5, id `1790548646-65be`)

A language pass over every slide, plus a new figure for the innovation. Slide count, order, labels, citations and numbers are unchanged.

- **New BDG mechanism figure** (`figure_source/fig3_bdg.tex`), in the style of the DooABLe figure: the batch → the guide's predicted properties with measured spread against the requested setpoint → the controller → what the sign of e does, with the feedback arrow closing the loop, and the rule banded underneath. It replaces the equation stack on slide 2c-ii; the statistics and the rule now live in the picture.
- **Prose tightened throughout.** Sentences shortened, hedges cut, body text 24 → 26 px, key terms bolded. Every takeaway box now leads with the claim rather than the word "Takeaway".
- Slide 2a: the pending own-diffusion row is marked `[TODO: v3 Job A]` in the callout again (v2 lost the marker in an edit).
- Appendix A4's subtitle no longer says "1 of 2" — the second slide was never written.
- Checked after the rewrite: label sequence 1a → 3e intact, every on-slide citation still has its bottom-right source, 44 red `[TODO]`s, no q90, no forbidden phrasing from `07_DO_NOT_SAY.md`, no slide overflows its canvas.

## v4 — 2026-09-28 (artifact version 8, id `1790559215-7ff7`)

Second language pass, three sections reframed, and D-Flow removed throughout.

- **D-Flow is gone** — its row in the §4.5 benchmark (2e) and in the appendix arm table (A4), and its bibliography entry. Entries after it renumbered: Gumbel-Softmax FM `[24]`, EquiFM `[25]`, EEGSDE `[26]`; every citing slide updated. The external comparison is now four methods (plug-in, TMPD, LGD-MC, TFG), which still clears the assignment's "at least three".
- **1e reframed as our FM against our own diffusion.** 1e-i is the measurement plan, stated as scored end to end by our own f_A/f_B. 1e-ii is a matched-protocol table: what is identical by construction (data, backbone, optimiser, budget, selection, sampler, evaluation) against what must differ (corruption process, prediction head, the diffusion model's extra Tweedie step). The borrowed-checkpoint mismatch rows moved out — they belong to 2a and the appendix.
- **1f rebuilt as a reading guide.** The figure now sits beside five numbered steps, each pointing at the slide that expands it, rather than two paragraphs of caption.
- **2a: EquiFM is the only external model.** Both published EDM rows removed; real QM9 kept as the ceiling. The slide opens by saying every row is unconditional and unguided. EquiFM's row is our own 256-molecule pilot, labelled one seed and twice our training data.
- **The trust-region line now explains itself:** uncapped, a correction can overwhelm the velocity, the trajectory diverges and coordinates overflow — 222 broken molecules in that diagnostic run, zero with the cap.
- **3e rebuilt around the answer.** A dark "No." panel carries the verdict; three short blocks carry what the evidence does support, why chemistry survives, and the one experiment that could withdraw the claim.
- Body text is 26–28 px, slide padding evened out, and one edit made in the Slides editor (s2b1's normalised styling) was preserved rather than overwritten.
- 36 red `[TODO]`s (v3's 44, minus the six D-Flow cells and its two markers). Checks re-run: label sequence 1a → 3e, every citation sourced bottom-right, no q90, nothing from `07_DO_NOT_SAY.md`, no slide overflowing.

## v5 — 2026-09-28 (artifact version 11, id `1790567881-a908`)

An audit against the newest manuscript (`paper/main.tex`, 28 Sep) plus eight requested fixes. Slide count, order and labels unchanged.

- **Aligned with the paper.** 2c and 2d now follow the manuscript exactly and carry no pilot numbers: 2c-i motivates BDG from the plug-in coupling (one w moves the mean and shrinks the spread); 2c-iii states the reduction for w_eff ≠ 0, the bound, the expansion condition (η > 1 and V_b < τ²(1 − 1/η)) and the property-ODE picture as an idealisation, not a guarantee; 2c-iv shows the paper's Algorithm 1; 2d-i is the paper's registered ablation table; 2d-ii states in advance what each outcome would establish. The pilot "controller, measured" figure is off the deck.
- **Mechanism figure corrected** (`figure_source/fig3_bdg_v2.tex`): e < 0 only weakens contraction; expansion needs w_eff < 0. The old panel said e < 0 expands.
- **Numbers the paper's changelog refuted are gone**: a q90 value (1.219) on 2d-ii, "largest z = 3.07" (really −3.07, a BDG loss), "one or two band widths off", "chemistry retained at matched coverage", clip counts 884 vs 2,516, and real QM9's invented fourth digit (now 0.994 / 0.956 / 0.982).
- **Comparison set on 2c-i**: where BDG sits against plug-in, TMPD, LGD-MC and TFG. TMPD's second moment is each sample's own endpoint covariance; BDG's is the batch's spread in property space. Prior art (Particle Guidance, MGD, VTD, CFG-Ctrl, FBG, negative guidance) kept in one line.
- **2b rebuilt around plug-in**: 2b-i gives the rule and how it enters both samplers; 2b-ii is the FM/VP × unguided/plug-in table with the setting stated. FM cells are the pilot (n = 256, one seed, w = 1): error falls 0.8–2.0 band widths, in-band change unresolved, 0.06–0.10 stability lost. VP cells are TODO (Job A); the rule for choosing the lead property is fixed now.
- **Modality 2 is DeepFlyBrain** (500 bp, GC and CpG, 83,722 / 10,505 / 10,434), as in the paper. Old 200-bp Kenyon-cell results removed; 3a–3e rewritten; 3b/3c are TODO tables; 3e now reads "Not yet shown."
- **1f is a native workflow chart** (two lanes, four stages, the shared BDG rule linked), replacing the dense figure.
- **Wording**: 1a, 1b-ii, 1c, 1d, 1e rewritten in plain scholarly terms; NFE defined on 1c-ii and 2a; diffusion notation now matches the paper (u, x₁, εφ); 1e-ii discloses the checkpoint-selection mismatch (FM final weights vs VP best validation).
- Bibliography: Gumbel-Softmax FM dropped (no longer cited); EquiFM → [24], EEGSDE → [25]; Dirichlet FM cited as ICML 2024.
- 90 red `[TODO]`s. Checks re-run: label sequence 1a → 3e, every citation sourced bottom-right, no q90, nothing from `07_DO_NOT_SAY.md`, no slide overflowing.

## v6 — 2026-09-28 (artifact version 12, id `1790629322-1aba`)

Filled every landed cell from Bobo's v3-final blade run (`results/v3`, `results/m2`, n = 2,000, 3 seeds), aligned to `paper/main.tex` v6. The Betty n = 5,000 run is not used (different predictor pair). The only red `[TODO]`s left are our own VP diffusion base model (Job A), which has been trained but not yet benchmarked.

- **2a base models filled.** FM (ours) atom 93.6, mol 39.7±0.6, valid 75.6±1.1, uniq 99.8, 0.051 s/sample; EquiFM 98.6 / 86.2±0.3 / 93.8±0.6 / 99.7 / 0.086 s; real QM9 ceiling 99.4 / 95.6 / 98.2. VP row is the only TODO (Job A). The 256-molecule EquiFM pilot is gone (A1 now shows the 3×2,000 unguided run).
- **2b-ii filled for flow matching:** plug-in raises decoded in-band μ +2.75 (z 5.22), gap +2.35 (z 4.04); α +1.07 (z 2.56, unresolved); stability −2.5 to −6.2, validity −1.9 to −5.7. VP cells remain TODO; the lead-property rule is kept.
- **2c-iv is now "BDG at work"** with the setpoint→spread figure (R 0.67–0.76 → 1.15–1.24 on FM, EquiFM to 1.25; plug-in narrows in 6/6), and Algorithm 1 moved to a new appendix slide **A5** (deck grows to 37 slides).
- **2d-i/ii filled:** 612-cell grid, the FM ablation heatmap, the paper's five controls with numbers, and the outcome reading — η=0 identity holds; 0/12 beat plug-in at the registered bar (3 losses, 9 unresolved); spread follows τ_m over four setpoints (widening 6/6); replay and signed-strength not run.
- **2e-i/ii filled:** the paper's `tab:recent` (unguided, plug-in, TMPD, LGD-MC, TFG, BDG τ_m 0.5 and 1) with passes per cell (gen/guide) and s/sample; BDG costs plug-in's calls (1,000/400), a fifth of LGD-MC's and TFG's guide passes.
- **Part 3 (DNA) filled from the M2 blade cells:** 3b is the GC/CpG × w=1/w=4 table with JSD; BDG−plug CpG +4.82 (w1) / +16.83 (w4), GC +0.42 / +2.32; plug/TMPD/LGD-MC collapse to one comparator; Dirichlet FM / Fisher Flow / MOG-DFM not evaluated. 3c carries the DNA panel and the transfer verdict; 3d states measured failure modes (clip 7.4–9.2% vs ≤0.21%, ~10× correction share, ~5.4× window, guided≠scored, batch coupling); 3e is "No on QM9. Yes on CpG." with the registered-bar phrasing.
- **1c/1d/1e updated:** FM ships epoch-1,500 EMA (MD5 e19ccc06); VP is trained (Bobo) with a checkpoint audit + 18-cell benchmark pending; every cell is 3 seeds × 2,000 pooled to 6,000; the registered test is unpaired z at bar 2.99.
- 19 red `[TODO]`s, all our VP diffusion (Job A) cells. Checks re-run: label sequence 1a → 3e, every citation sourced bottom-right, no q90, nothing from `07_DO_NOT_SAY.md`, no slide overflowing.

## v7 — 2026-09-29 (artifact version 13, id `1790649080-c2b2`)

Our own VP diffusion benchmark landed (`results/v3/vp`, 18 cells, md5 `8a3390a6`, epoch-1475 EMA, 0.81 GPU-h on a B200 MIG slice), so the deck has **no `[TODO]` left**. Plus a 28-point revision pass against `course/CIS_6270_Fall_2026_Project1_Assignment.pdf`. 38 slides (A6 added).

- **The base-model question is decided on measured evidence.** 2a now carries five rows — our FM, our VP, borrowed EquiFM, borrowed EDMsecond, real QM9. At matched data, backbone, recipe and budget our FM beats our VP by **+10.9 molecule-stability points** (39.7 vs 28.8), +9.4 validity, +2.9 atom, with one fewer network call. The VP timing ran on a different accelerator and the slide says so.
- **2b fills both families.** Plug-in clears the bar on μ and gap for *both* (FM +2.75/+2.35, VP +2.13/+2.60) and leaves α unresolved on both; the setting box states plainly that the rule is DPS-style plug-in for both rows.
- **A registered-protocol slide (1e-i).** Target, band rule and δ values, 3 seeds × 2,000, batch 500, t ≥ 0.5, unpaired z at 2.99, the ~0.8–1.9-point resolution, the f_A/f_B architecture (EGNN 128 × 4, 429,825 parameters each) and the **hardware mismatch**, all stated as the truth rather than a side note.
- **1f redrawn.** Four stages × two lanes with the shared BDG band spanning both, and the comparison sets (plug-in, TMPD, LGD-MC, TFG / TFG-MC) now visible where they are actually used.
- **2d rebuilt around what ran.** Five controls, each phrased as the question it answers; the "not run" rows are gone from the table and live in 3d. Positive reading first: the setpoint is a dial (0.67× → 1.25×), coverage tracks the measured w_eff at Spearman 0.92–0.99, and dispersion buys more coverage per stability point than strength. The honest bound — all six gains below the 2.99 bar — is kept beside it.
- **3e reframed.** "Yes to the dial. Yes on CpG." The evidence block leads; where we yield to TFG's convergence is stated, with the 8–27 stability points it costs.
- **Rubric calibration.** 2e and 3b now report coverage, quality, **diversity** and **efficiency** as the rubric's 4.5/4.6 tables require; GC, CpG and 3-mer JSD are defined once in 1b instead of on page 25.
- **Presentation fixes.** Haimo Fang removed from the cover; BDG's accent changed from red to violet (`#5B3E9B`) across 1a, 1f, 2c, 2d, 2e, 3a, 3b, 3c, 3e and the mechanism figure (`figure_source/fig3_bdg_v3.tex`); the NFE box, the "why the cap" box and four other stray callouts dropped; 1a and 1b text enlarged; new appendix **A6** documents the simplex-FM architecture.
- 0 red `[TODO]`s. Checks re-run: label sequence 1a → 3e, every citation sourced bottom-right, no q90, nothing from `07_DO_NOT_SAY.md`, no slide overflowing.
