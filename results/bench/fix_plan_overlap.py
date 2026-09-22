"""Record the corrected overlap figure and the robustness-check protocol. Run once."""
import io

p = "GUIDANCE_EXPERIMENT_PLAN.md"
s = io.open(p, encoding="utf-8").read()

anchor = """**Not to be used as f_A, ever:** `ridge`."""
add = """### External predictors: usable as a CHECK, never as the primary pair

Published predictors are more accurate than ours on every property, but all of
them trained on someone else's split, and ours is a private random split.

| predictor | mu (D) | alpha (Bohr^3) | gap (Ha) | overlap with our `train_a` |
|---|---|---|---|---|
| **our EGNN f_B** | **0.0840** | **0.2407** | **0.00380** | **0%, by construction** |
| TFG oracle (EGNN) | 0.0728 | 0.1529 | 0.0027 | ~37% (see below) |
| TFG guide (EGNN) | 0.0659 | 0.1551 | 0.0029 | ~37% |
| SchNet (mu only) | **0.0210** | -- | -- | ~all of QM9 |
| EDM published L-bound | 0.043 | 0.10 | 0.00235 | ~37% |

**The ~37% figure, and how it was obtained.** It is an EXPECTATION, not a
measurement. EDM (and TFG, which vendors EDM's machinery) builds its halves
with `np.random.seed(42)`, permutes its 100,000-molecule train partition and
takes the first 50,000 (`tasks/networks/qm9/data/utils.py`). A molecule drawn
from OUR 133,885 therefore has probability 50,000/133,885 = **37.3%** of
sitting in their first half, i.e. ~19,240 of our 51,527 `train_a` molecules.

Two caveats to carry into any write-up:

- It assumes their half is a uniform draw independent of ours. Both splits are
  seeded permutations, so this is reasonable but unverified.
- An **earlier draft said 38%**, using their post-exclusion pool (130,831) as
  the denominator. That was wrong: the draw is from OUR set, so OUR total is
  the denominator. **Quote 37%, not 38%.**

The exact overlap is computable -- both splits are deterministic -- but it
needs their full download-and-process pipeline reproduced, and no decision
turns on the precise value: at 30% or 45% the conclusion is the same.

**Why this rules them out as primary f_B.** The generator trained on `train_a`.
An evaluator that memorised those molecules scores their near-copies
optimistically -- exactly the leak `SPLIT_PROTOCOL.md` exists to prevent.

**Why they are still worth running.** The leak is identical across arms, since
every arm samples from the same generator. So absolute MAE is optimistic but
the RANKING is informative. Protocol, on the winning arm only, re-scoring saved
samples with no re-sampling:

| property | check evaluator | what it tests |
|---|---|---|
| mu | SchNet, 0.0210 D | 4x better AND a different architecture -- tests both accuracy and shared blind spots |
| alpha, gap | TFG oracle | the only external option; same EGNN family, so it tests the data-leak question only |

If the winner's advantage survives a 4x-better evaluator of a different
architecture, the claim is far stronger. If it does not, that is the finding.

**Not to be used as f_A, ever:** `ridge`."""

assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("plan: 37%% overlap recorded with provenance and the 38%% correction noted")
