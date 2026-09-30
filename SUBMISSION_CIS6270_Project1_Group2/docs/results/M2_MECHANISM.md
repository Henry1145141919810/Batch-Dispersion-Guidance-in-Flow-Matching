# How BDG raises in-band on M2, and how that differs from M1

> **Correction, 28 Sep.** The M2 half of this page checks out against the cells
> (+0.985 is unguided's bias/delta on gc; +1.277 is `bdg_e8t0.5` at w = 4, a rung
> that clips 19.2 % of guided steps). The M1 contrast does **not** hold on our own
> base: in the v3 headline, `bdg_e4t0.5` makes |bias|/delta **worse** than unguided on
> fm/alpha (-1.085 -> +1.495, an overshoot) and fm/gap (-0.893 -> -1.242), and on
> fm/mu plug alone does almost all the debiasing (+2.095 -> +0.496; BDG +0.439)
> ([V3_RESULTS_fm.md](V3_RESULTS_fm.md)). So "on M1 BDG does both" is true of fm/mu
> only. The text below is Bobo's original.

in-band is a single number that two different things can move: shifting the
mean toward the target, or contracting the spread around it. Reporting the
gain without saying which would let a reader assume the property was steered to
target. On M2 it was not.

## M2 (gc), w = 4, tau_mult = 0.5, 3 seeds

| arm | in_band | gc_sd | sd / unguided | bias / delta |
|---|---|---|---|---|
| unguided | 0.1395 | 0.05154 | 1.000 | +0.985 |
| bdg eta=1 | 0.1453 | 0.04873 | 0.945 | +1.091 |
| bdg eta=2 | 0.1505 | 0.04748 | 0.921 | +1.158 |
| bdg eta=4 | 0.1642 | 0.04619 | 0.896 | +1.231 |
| bdg eta=8 | 0.1818 | 0.04525 | 0.878 | +1.277 |

The spread contracts monotonically and the bias **worsens** monotonically: the
batch mean moves from just inside the +/- 1 delta band to outside it. in-band
still rises because the contraction outweighs the drift.

At `tau_mult = 1.5` the mirror image holds -- the spread *widens* (1.009 to
1.031 x) and in-band falls. The controller is doing exactly what its setpoint
specifies, in both directions. That sign change is the strongest single piece
of evidence that the mechanism is the dispersion feedback and not extra force.

## M1 (mu), same arms

| backend | arm | in_band | bias/delta | rmse/delta |
|---|---|---|---|---|
| FM (ours) | unguided | 0.0780 | +2.095 | 8.935 |
| FM (ours) | bdg_e4t0.5 | 0.1152 | **+0.439** | 6.024 (0.67x) |
| EquiFM | unguided | 0.0858 | +2.437 | 9.275 |
| EquiFM | bdg_e4t0.5 | 0.1193 | **+0.843** | 6.228 (0.67x) |

On M1 BDG does **both**: it removes most of the bias (-1.66 delta on our base)
*and* contracts the residual to 0.67x.

## Why the modalities differ, and what it means

BDG's numerator is `(y - F_i) - eta*e*(F_i - Fbar)`. The first term corrects
the mean, the second controls the dispersion.

- **M1's base is far off-target** (+2.1 delta), so the mean-correction term has
  a large error to work on and dominates the visible gain.
- **M2's base is already on target** -- the unguided mean sits 0.985 delta
  above q50, i.e. essentially at the band edge, because the base reproduces the
  corpus median well. There is almost no mean error to remove, so only the
  dispersion term does anything, and the small residual drift is uncorrected.

**So the transfer is partial, and that is the finding.** The dispersion
controller transfers intact and is measurably responsible for the gain (the eta
sweep at fixed w). The target-seeking half does not visibly transfer, because
the second modality gives it nothing to do. A write-up that reported only
"in-band improved on both modalities" would be hiding the more interesting
result.

## Fidelity, across the same sweep

| w | eta | in_band | k-mer JS | x unguided | decode conf | diversity | clipped |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.1453 | 0.00051 | 1.13x | 0.9553 | 0.7456 | 1.0% |
| 4 | 2 | 0.1505 | 0.00055 | 1.21x | 0.9525 | 0.7457 | 3.2% |
| 4 | 4 | 0.1642 | 0.00059 | 1.31x | 0.9475 | 0.7457 | 9.2% |
| 4 | 8 | 0.1818 | 0.00062 | 1.37x | 0.9400 | 0.7457 | 19.2% |

The cost is real, monotonic and small in absolute terms: at the strongest rung
k-mer JS is 0.00062 against the base's 0.00045, still **27x closer to real
enhancers than uniform-random sequences** (0.0170). Decode confidence falls
1.8 points. **Diversity does not move at all** (0.7456 -> 0.7457), so the
contraction is in the guided property, not a collapse of the samples.

The `eta = 8` rung at `w = 4` clips on 19.2 % of guided sample-steps, so it is
partly clip-limited and should be reported as such (protocol 1.2b).
