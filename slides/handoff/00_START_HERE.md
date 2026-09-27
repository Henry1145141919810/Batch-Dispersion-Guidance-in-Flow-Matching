# Defense slides: everything needed to build the deck

**CIS 6270 Project 1, Group 2.** Defense is Wednesday 30 September, 30 minutes:
**15-minute presentation (hard limit) + 15 minutes Q&A.** Presentation is worth 4.5
points, Q&A 3 points.

This folder is a complete handoff. Feed all of it to the Claude app and ask it to build
the deck. Nothing outside this folder is needed.

---

## What to say to the Claude app

> Build a 15-minute defense deck from this folder. `01_DECK_SPEC.md` is the contract:
> follow it exactly, especially the 1a-3e slide sequence, the label in the top-right
> corner of every content slide, the numbered citations with sources in small text at
> the bottom-right, and the bibliography slide at the end. The slide content is in
> files 02, 03 and 04, one section per file, already written slide by slide. Use the
> two figures in `figures/`. Keep every `[TODO: ...]` marker visible and red. Do not
> invent numbers.

---

## What is in here

| File | What it is |
|---|---|
| `00_START_HERE.md` | this file |
| `01_DECK_SPEC.md` | **the contract**: rubric requirements, slide sequence, formatting rules, what the graders check |
| `02_CONTENT_part1_setup.md` | slides **1a-1f**: frame the study, data, FM baseline, diffusion baseline, metrics, overview |
| `03_CONTENT_part2_modality1.md` | slides **2a-2e**: base model choice, guidance, the innovation, ablations, benchmarking |
| `04_CONTENT_part3_transfer.md` | slides **3a-3e**: transfer, results, verdict, limitations, conclusion |
| `05_BIBLIOGRAPHY.md` | the numbered source list for the final slide |
| `06_SPEAKER_PLAN.md` | who presents what, timing per slide, Q&A preparation |
| `07_DO_NOT_SAY.md` | **read this**: claims our own adversarial review refuted. Saying them invites a question we cannot answer |
| `08_PLACEHOLDERS.md` | every pending number, what it is waiting on, and where it goes |
| `figures/` | both figures as PNG (300 dpi) and PDF (vector), with a README |

---

## The three things that decide the grade

1. **The 1a-3e sequence is mandatory and ordered.** Slides must progress 1a, 1b, 1c,
   1d, 1e, 1f, 2a, 2b, 2c, 2d, 2e, 3a, 3b, 3c, 3d, 3e. More than one slide per item is
   allowed. The label must appear in the **top-right corner** of each content slide.

2. **Every source must be cited on the slide**, numbered, with the matching source in a
   very small font at the **bottom-right**. Any uncited figure must be original. Both
   figures here are ours; their READMEs say so.

3. **Talking time must be roughly equal across all four presenters.** Teams are
   penalised if one person dominates. `06_SPEAKER_PLAN.md` has a split.

**No notes or flashcards are permitted during the presentation.** Nothing in this
folder goes on a slide as a speaker note.

---

## Scope rule that applies everywhere

This project reports the **q50 target only**. There must be no q90 number anywhere in
the deck. See `07_DO_NOT_SAY.md`.

---

## The honest shape of the result, so the deck does not oversell

The innovation (BDG) **works as a spread controller and does not raise band coverage**.
Both halves belong in the deck. The rubric rewards a well-analysed negative result, and
our own adversarial review will be visible to anyone who reads the repository. The deck
should:

- show the controller does what it claims (monotone setpoint, both modalities, zero
  extra cost),
- state the reduction ourselves before anyone finds it,
- report the coverage null with its statistics,
- and name the one place BDG is genuinely ahead: it keeps more chemistry at matched
  coverage.
