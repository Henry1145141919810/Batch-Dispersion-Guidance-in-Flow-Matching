# On M2 the guidance window matters ~10x more than the guidance method

> **Correction, 28 Sep (cross-check of the blade run; see SCOPE_FM_GUIDANCE_STATUS.md).**
> The best result any method achieves *inside* v3's window is **+4.23 pp**
> (`bdg_e8t0.5` at w = 4, 3 seeds, [M2_ABLATION_GRID.md](M2_ABLATION_GRID.md)), not
> +2.5 pp, so **on gc** the window is worth about **5x** the method, not 10x. Both
> terms are gc-only: cpg has no t = 0 cell, and its in-window gain reaches +18.30 pp
> (`bdg_e4t0.5`, w = 4), so the ratio does not hold for M2 as a whole. The +22.82 pp
> window gain and the 30-32 % clip at t >= 0 check out against the cells. At t >= 0,
> against plug, the two headline BDG arms land on opposite sides of the bar.
> `bdg_e4t0.5` is a **tie** (-2.33 pp, unpaired z -2.67; read it as "no advantage at
> t >= 0"). `bdg_e4t1` is a **clear loss** (-20.80 pp, z -26.58). Full table:
> [M2_V3_RESULTS.md](M2_V3_RESULTS.md) section 6. The text below is Bobo's original
> and is left as written.

This is the most consequential M2 result and it does not flatter our method, so
it is recorded before the write-up is drafted rather than after.

## The numbers (gc, n=2000, 3 seeds, w=1, paired by seed)

| arm | t >= 0.5 (protocol) | t >= 0 (diagnostic) | window buys |
|---|---|---|---|
| `plug` | 0.1403 | **0.3685** | **+22.82 pp** |
| `bdg_e4t0.5` | 0.1445 | 0.3452 | +20.07 pp |
| `bdg_e4t1` | 0.1402 | 0.1605 | +2.03 pp |
| unguided | 0.1395 | -- | -- |

Guiding from t = 0 rather than t = 0.5 is worth **+22.8 pp** to `plug`. The best
our method achieves *within* v3's window is **+2.5 pp**. The window is worth
roughly **ten times** the method.

And at t = 0 it is not merely tighter but better on every axis: bias/delta
improves from +0.985 to +0.734, gc_sd falls from 0.0515 to 0.0232, and k-mer JS
*improves* from 0.00045 to 0.00029 -- the samples are closer to real enhancers,
not further.

## BDG's advantage exists at one window and not the other

Paired, same seed:

| window | w | bdg_e4t0.5 - plug | t |
|---|---|---|---|
| t >= 0.5 | 1 | **+0.42 pp** | +9.45 |
| t >= 0.5 | 4 | **+2.32 pp** | +14.57 |
| t >= 0 | 1 | **-2.33 pp** | -1.81 |

At v3's window BDG beats DPS clearly and reproducibly. At the early window it
does **not** -- it is slightly worse, and `bdg_e4t1` is worse by 20.8 pp
(t = -18.2). Any claim that BDG helps must carry the window it was measured at.

## Why, and why this is a coherent finding rather than a contradiction

At t = 0 the score-to-velocity factor `(1-t)/t` diverges, so every arm requests
a correction far outside the velocity-relative trust region and **the clip
truncates 30-32 % of guided sample-steps**. A clipped step carries no magnitude
information: what reaches the sample is the clip's ceiling, identical for every
arm. The methods therefore converge, and what is left is the window.

So the two windows measure two different regimes:

- **t >= 0 is clip-saturated.** Guidance is running flat out, the arms collapse
  toward each other, and the gain is the *schedule*, not the steering rule.
  This is the same mechanism that makes `tmpd` and `lgd_mc` identical to `plug`
  (protocol 3.0), seen from the other side.
- **t >= 0.5 is unsaturated** (0-9 % clipped below the top rung). The arms are
  separable and the controller's contribution is measurable -- which is what
  the eta sweep at fixed w demonstrates.

v3's window is therefore the right one for the question this project asks
(*which steering rule is better*), and the wrong one for the question *how good
can guided generation on this modality get*. Both belong in the paper, and they
are different questions.

## What the write-up must not say

- Not "BDG improves in-band on Modality 2" without "at v3's window, w = 1-4".
- Not a comparison of M2's +2.5 pp against M1's gains as though the schedules
  were matched -- they are, but the M2 base is already on target (M2_MECHANISM.md).
- Not a claim that the early window is simply better: it is better *and*
  clip-limited, so it does not answer the method question at all.
