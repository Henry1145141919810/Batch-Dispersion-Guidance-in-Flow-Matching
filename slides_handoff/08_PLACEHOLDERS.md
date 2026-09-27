# Every pending number, and what it is waiting on

Each `[TODO: ...]` in files 02, 03 and 04 is listed here. Render them **red and
visible** in the deck so none is missed before submission.

Two runs produce all of them. Both are pre-registered, so the protocol cannot be
changed after seeing results.

| Run | What it is | Produces |
|---|---|---|
| **v3 headline** | 7 arms, q50, w = 1, 3 seeds, 3 backends | slides 2a, 2b-ii, 2e-i |
| **v3 ablation** | η × τ_mult grid at w ∈ {1,4}, 3 seeds | slide 2d-i |
| **v3 Job A** | our own VP diffusion on QM9, unguided + plug-in | slides 1d-ii, 1e-ii, 2a |
| **v3 Job B** | Modality 2 sweep on the simplex | slides 3a, 3b |

---

## Slide by slide

| Slide | Marker | What is needed |
|---|---|---|
| 1d-ii | `[TODO: v3 Job A]` | our VP diffusion's training status line |
| 1e-i | `[TODO: v3 n]` | the sample count per guidance cell the operator chooses |
| 1e-ii | `[TODO: v3 Job A]` | training compute for our diffusion row |
| 2a | `[TODO: v3 Job A]` ×4 | atom stability, molecule stability, validity, uniqueness for our VP diffusion; uniqueness for `EDMsecond` |
| 2b-ii | `[TODO: v3]` | three-seed confirmation of the guidance table |
| 2d-i | `[TODO: not run]` | the **signed-w plug-in control**; the drop rule applies to it |
| 2d-i | `[TODO: v3 ablation]` | the three-seed η × τ_mult grid |
| 2e-i | `[TODO: v3]` ×~18 | in-band, molecule stability and diversity for all five external methods, the internal reference and BDG; D-Flow's pass counts |
| 2e-ii | `[TODO: v3]` | completes the better/comparable/worse verdict |
| 3a | `[TODO: v3 Job B]` | M2 architecture and training budget |
| 3b | `[TODO: v3 Job B]` ×4 | the three simplex baselines and our transferred row |

---

## Numbers that are already final and must not be marked pending

Do not add a TODO to any of these; they are measured and will not change.

- Base model: atom 0.9366 ±0.0012, molecule 0.3993 ±0.0085, validity 0.7625 ±0.0031,
  uniqueness 0.9939 (3 seeds × 10,000)
- `EDMsecond` through our evaluator: 0.9644 / 0.6725 / 0.8490 (n = 2,000, one seed)
- Real QM9 ceiling: 0.994 / 0.956 / 0.982
- Splits: 51,527 / 51,527 / 17,748 / 13,083 of 133,885
- Parameters: ours 3,753,229; `EDMsecond` 5,340,409
- Training: 1500 epochs, 303k steps, 11.4 h
- The whole guidance table on slide 2b-ii (n = 256, one seed)
- The whole ablation table on slide 2d-i, gates and ladder and controls
- Monotone ladder: 3/3 q50 curves, 0.670 → 1.187 (guide f_A). The evaluator-scored
  range 0.650 → 1.197 spans q50 and q90; do not quote it beside a q50 claim.
- The null: 72 paired tests, max |z| = 3.07, Šidák p = 0.14
- w_eff: 34 sign changes, span [−2.64, +10.35], bound ≥ −3
- Open-loop replay: recovers 69.5–96%, gap 0.3–6.6% against plug-in's 8.4–30.5%
- Clip counts for the guidance-table cells: 701 (plug-in w=8) vs 4,457 (BDG w=32).
  The 222 non-finite figure is from a q90 pilot on arms not in this paper; qualify it.
- Cost: 4,000 guide passes vs 20,000 for LGD-MC and TFG
- All Modality 2 structural signatures, and the 2.6× magnitude gap (0.20 vs 0.52)
- M2 causes: GC affine, 200-position average, 4.3 attainable values at 1/200 granularity
- M2 clip: fired on 0–3% of steps, expanding cell clipped once in 25,600

---

## Before submitting

1. Search the deck for `TODO` and confirm every remaining one is intentional.
2. Confirm no **q90** number appears anywhere.
3. Confirm the label sequence reads 1a 1b 1b 1c 1c 1d 1d 1e 1e 1f 2a 2b 2b 2c 2c 2c 2c
   2d 2d 2e 2e 3a 3b 3c 3d 3e.
4. Confirm every slide with a borrowed idea or image has a numbered citation and a
   bottom-right source line.
5. Rehearse to 15:00.

Slides may not be changed after submission.
