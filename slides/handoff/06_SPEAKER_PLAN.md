# Speaker plan, timing, and Q&A preparation

**15 minutes, hard limit. No notes or flashcards are permitted.**
Teams are penalised if one member dominates while others barely speak.

---

## Split

Four presenters, roughly equal time. Balance by **time, not slide count**: the figure
slides go quickly, the table slides do not.

| Presenter | Items | Slides | Target |
|---|---|---|---|
| **1** | 1a, 1b ×2, 1c ×2 | ~5 | 3:30 |
| **2** | 1d ×2, 1e ×2, 1f, 2a | ~6 | 3:45 |
| **3** | 2b ×2, 2c ×4, 2d ×2 | ~8 | 4:15 |
| **4** | 2e ×2, 3a, 3b, 3c, 3d, 3e | ~7 | 3:30 |

Presenter 3 has the most slides because the innovation lives there, and two of them are
figure-led. If they run long in rehearsal, move 2b to presenter 2.

**Hand-off points are natural:** presenter 1 ends after the flow-matching baseline,
presenter 2 after the base-model choice, presenter 3 after the ablation conclusions.

---

## Pacing

26 spoken slides in 15:00 is about **35 s each**. Budget:

- **~60 s**: the table slides (2a, 2b-ii, 2d-i, 2e-i, 3b) and the two figures (1f, 2c-iv)
- **~25 s**: the text slides
- **Slide 2c-iv (the measured controller figure) is the one to linger on.** It carries
  the whole innovation argument in one picture.

Rehearse with a timer at least twice. The limit is hard, and slides cannot be changed
after submission.

---

## Q&A: 15 minutes, questions go to individuals

**Questions are directed at one student and cannot be delegated.** Everyone must know
the whole project: datasets, preprocessing, both formulations, training and sampling,
guidance, the innovation, ablations, baselines, metrics, figures, conclusions,
implementation details, and the Modality 2 transfer.

### The seven questions most likely to be asked

1. **"Why is this not just a hyperparameter sweep?"** The assignment explicitly excludes
   tuning. Answer with the mean/deviation form: the centring gain is fixed at w and only
   the deviation gain is servoed, and w_eff changes sign, which w cannot do without
   reversing the centring term. Also: τ is in property units, w is a dimensionless push.

2. **"You admit it reduces to the plug-in rule reweighted. So what is new?"** The
   composition, the widening-side bound w_eff ≥ 1 − η, and the pull-back sign property.
   Say the reduction before they raise it.

3. **"Your ablation shows no coverage gain. Why is this in the paper?"** Because the
   controlled comparison establishes the mechanism, and the null is bounded and
   quantified: 72 paired tests, max |z| = 3.07, Šidák p = 0.14. And BDG keeps more
   more chemistry at the same coverage, but say it is unresolved: z = 1.73, one seed,
   and both cells sit below our chemistry floor.

4. **"Which model did you carry forward, and why?"** Flow matching, for the closed-form
   endpoint estimate, **not** for a fidelity win. Do not overclaim here.

5. **"Is your diffusion baseline actually yours?"** Say it plainly: **yes, and there
   are two.** `vp` is ours -- trained here to epoch 1500, selected 1475, 18 v3 cells.
   `edm` is TFG's released EDMsecond, borrowed and labelled as such on 2a. Against our
   own VP at matched everything, flow matching wins (mol stab 0.3970 vs 0.2883, validity 0.7562 vs 0.6617, atom stab 0.9356 vs 0.9070; every gap 7-14x the sd of the difference).

6. **"Why should in-band be the metric rather than MAE?"** Because a design campaign
   accepts anything inside the tolerance band, and MAE rewards an arm that is close on
   average while landing nothing inside it. And we never rank on in-band alone, because
   it rewards contraction onto implausible structures.

7. **"What would change your mind?"** The signed-w plug-in control. We pre-registered a
   drop rule: if it reaches BDG's spread at equal chemistry, we withdraw the controller
   claim and report the decomposition instead.

### Know your own numbers

Each presenter should be able to state, without looking: the split sizes, the parameter
counts, NFE, δ's definition, the three properties and their units, what the trust region
does, and the three measured causes of the smaller Modality 2 effect.

### If asked something we have not measured

Say so, and say what would measure it. `08_PLACEHOLDERS.md` lists every pending number.
Guessing is worse than "that run has not landed; it is the v3 ablation stage, three
seeds, and it would settle exactly that question."
