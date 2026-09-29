# Defense deck: how to build it, who presents what, and what still needs data

**Deck:** `defense.tex` → `defense.pdf`, 29 slides.
**Build:** `pdflatex defense && pdflatex defense` (two passes; the second fixes the
`shrink` frames).

The deck depends on two figures built from the paper tree:

```
.venv/Scripts/python.exe paper/figs/make_mechanism.py   # figs/mechanism.pdf
cd paper/figs && pdflatex overview.tex                  # figs/overview.pdf
```

---

## 1. What the rubric requires, and where it is satisfied

| Requirement | Where |
|---|---|
| Slides progress in order **1a → 3e** | every content slide, in sequence |
| Label **1a–3e in the top-right corner** | set by `\lbl{...}`; rendered as a red badge by the `headline` template |
| **Numbered on-slide citations** | `[1]`, `[2]`, … inline; numbering restarts per slide, which the assignment permits |
| **Source list, very small, bottom-right** | set by `\srcs{...}` |
| **Bibliography slide at the end** | slides 27–28 |
| **All uncited figures original** | the only figure is `paper/figs/overview.pdf`, which is ours; its source line says so |
| Evenly distributed talking time | see §2 |
| No notes | nothing on the slides is a speaker note; this file is not presented |

---

## 2. Speaker allocation (4 presenters, 15 minutes hard limit)

Roughly equal content and equal time. The teaching team penalises uneven splits.

| Presenter | Slides | Items | Target |
|---|---|---|---|
| **1** | 2–6 | 1a, 1b×2, 1c×2 | 3:30 |
| **2** | 7–12 | 1d×2, 1e×2, 1f, 2a | 3:45 |
| **3** | 13–20 | 2b×2, 2c×4, 2d×2 | 4:15 |
| **4** | 21–27 | 2e×2, 3a, 3b, 3c, 3d, 3e | 3:30 |

Slide 1 (title) and 28–29 (bibliography) are not spoken.

**Balance by time, not by slide count.** Presenter 3 has the most slides because the
innovation lives there, but several are figure-led and go quickly. Rehearse with a
timer and move the 2b boundary if presenter 3 runs long.

**Pacing:** 26 spoken slides in 15:00 is ~35 s each. Budget ~60 s for the table
slides (12, 14, 19, 21, 24) and the two figures (11, 18), and ~25 s for the text
slides. Slide 18 (the measured controller) is the one to linger on: it carries the
whole innovation argument in one picture.

**Q&A is 15 minutes and questions go to individuals.** Every member must be able to
answer on the whole project, not only their own slides. The highest-risk questions,
given what this deck claims:

1. *Why is BDG not just a hyperparameter sweep?* Answer with the mean/deviation form:
   the centring gain is fixed at `w` and only the deviation gain is servoed, and
   `w_eff` changes sign, which `w` cannot do without reversing the centring term.
2. *You admit it reduces to plug-in reweighted. So what is new?* The composition, the
   widening-side bound `w_eff >= 1 - eta`, and the pull-back sign property. Say the
   reduction before they find it.
3. *Your ablation shows no coverage gain. Why is this in the paper?* Because the
   controlled comparison establishes the mechanism, and the null is bounded and
   quantified (72 paired tests, max |z| = 3.07, Šidák p = 0.14).
4. *Which model did you actually carry forward and why?* Flow matching, for the
   closed-form endpoint estimate, not for a fidelity win. Do not overclaim here.
5. *Is your diffusion baseline yours?* Say plainly: **yes, and there are two.** `vp`
   is ours (trained here, 18 v3 cells, 28 Sep); `edm` is TFG's borrowed EDMsecond, which
   2a labels. Flow matching beats our own VP at matched everything (mol stab 0.3970 vs 0.2883, validity 0.7562 vs 0.6617, atom stab 0.9356 vs 0.9070; every gap 7-14x the sd of the difference).

---

## 3. Things that must not be said (they are refuted in our own records)

These come from the adversarial review in `docs/methods/BDG_REVIEW.md`. Saying any of
them invites a question we cannot answer.

- "variance control" — call it *a feedback-servoed deviation gain*.
- "the controller holds its setpoint" — the equilibrium is an asymptote the 50-step
  window never reaches.
- "no fixed schedule can reproduce the loop" — an open-loop replay recovers 69.5–96%.
- "no existing arm widens reproducibly" — `rch` widens 1.42× on both seeds.
- "we can prove neither direction can help" / "by definition".
- Anything at **q90**. This study reports **q50 only**.
- Any claim that a BDG rung beat `plug` without the multiplicity correction.

---

## 4. Placeholders still to fill

Every `\TODO{...}` renders as a small red tag. Grep them with:

```
grep -n "TODO{" defense.tex
```

| Slide | Needs |
|---|---|
| 12 (2a) | our own VP diffusion row (v3 Job A) |
| 14 (2b) | three-seed confirmation of the guidance table |
| 19 (2d) | the three-seed v3 ablation grid; the signed-`w` plug-in control |
| 21 (2e) | all five external-method rows and the fair-discussion verdict |
| 23 (3a) | M2 architecture and training budget (v3 Job B) |
| 24 (3b) | the three simplex baselines and our transferred row |

Slide 18's figure regenerates itself from `results/bdg_port/table.txt`, so it needs
no manual edit; re-run `make_mechanism.py` if those cells are ever re-run.

When the run lands, fill these and rebuild. Nothing else in the deck changes.

---

## 5. Submission note

Slides may not be changed after submission. Build once, check
`defense.pdf` opens and all 28 slides render, and submit the PDF.
