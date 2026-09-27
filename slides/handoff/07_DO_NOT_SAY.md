# Claims that must not appear in the deck or the Q&A

Every line below was refuted or narrowed by our own adversarial review of the
innovation (38 agents, per-objection rulings; see `docs/methods/BDG_REVIEW.md` in the
repository). Anyone who reads the repository can find these. Saying one on a slide
invites a question we cannot answer, and the review is the strongest evidence we have
that the work was checked properly, so it is worth being on the right side of it.

---

## Forbidden phrasings, and what to say instead

| Do not say | Say instead | Why |
|---|---|---|
| "variance control" | "a feedback-servoed **deviation gain**" | Only the deviation gain is servoed; the centring gain stays at w. Calling it variance control overstates what the rule does. |
| "the controller holds its setpoint" | "the setpoint moves achieved spread monotonically" | The law's equilibrium V* = τ²(1 − 1/η) is an **asymptote the 50-step window never reaches**. Measured V_b/τ² run-means stay far above it. |
| "no fixed schedule can reproduce the loop" | "the schedule largely transfers across seeds; feedback adds a small measurable remainder" | An open-loop replay on a different seed's noise recovers **69.5–96%** of the effect. The original claim rested on a confounded control. |
| "no existing arm widens reproducibly" | say nothing about uniqueness | The `rch` arm widens **1.42× / 1.455×** on both seeds and clears the floor. The old "1.010× yardstick" is inside noise. |
| "we can prove neither direction can help" / "by definition" | "we did not observe an improvement, at this power" | Delete "we can prove" and "by definition" entirely. |
| "widening is plug-in at a negative weight" | "the deviation gain goes negative while the centring gain stays at w" | Rebutted. Negative-weight plug-in reverses the centring term too; BDG's widening cells still move the batch mean *toward* the target. |
| "BDG beat plug-in" | "no rung beats plug-in at a corrected significance level" | 72 paired tests, max \|z\| = 3.07, Šidák p = 0.14. |
| "tightening costs chemistry" | "the chemistry cost tracks \|w_eff\|" | The cost follows the magnitude of the deviation gain, not the direction of the knob. |
| anything about **q90** | nothing | **This study reports q50 only.** There must be no q90 number anywhere. |
| "state of the art" | nothing | Not supported under any protocol we ran. |

---

## Three things we must claim in our own voice, before anyone asks

1. **The reduction.** num_i is affine in F_i, so BDG factors exactly as
   (1 + η·e)(y_eff − F_i). At any instant it is the plug-in field at a rescaled weight
   aiming at a shifted target. **Say this on slide 2c-iii.** A reviewer will find it
   otherwise, and finding it themselves is much worse than hearing it from us.

2. **The prior art.** Each ingredient is published: the scale as a control gain
   (CFG-Ctrl, FBG), ensemble moments to a target (MGD), batch spread targeting (VTD),
   batch-coupled repulsion (Particle Guidance), negative-weight widening. **Slide 2c-i
   lists them.** What is unclaimed is only the narrow composition, and we say the search
   was bounded (about eight queries, abstracts only).

3. **The base-model comparison is incomplete.** Our own VP diffusion has not finished
   training at scale. Slide 2a says flow matching is carried forward *without a fidelity
   win*, because the matched row is pending and the measured comparator is borrowed.

---

## What is genuinely ours, and safe to claim

- The **composition**: a scalar guidance weight set by the batch variance of a *learned
  property predictor's own endpoint prediction*, against a *two-sided user-chosen*
  setpoint, signed through zero.
- The **widening-side bound** w_eff ≥ 1 − η, which is what licenses running two-sided
  with no clamp.
- The **pull-back sign property**: the dispersion term reaches each molecule's property
  as a squared norm, so its sign survives the Jacobian transpose.
- The **reduction itself** as a structural result: any guidance affine in F_i is the
  plug-in rule reweighted.
- Every **measured** number in the deck.
