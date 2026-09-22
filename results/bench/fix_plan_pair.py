"""Lock the f_A / f_B pair to one fixed choice per property. Run once."""
import io

p = "GUIDANCE_EXPERIMENT_PLAN.md"
s = io.open(p, encoding="utf-8").read()

anchor = "## 2. The property functions\n"
add = """## 2. The property functions

### THE PAIR IS FIXED: EGNN f_A, EGNN f_B, one pair per property

There is exactly one guide and one evaluator per property for the whole
comparison, and they do not vary across arms, targets, strengths or windows:

    f_A = proj1/checkpoints/f_A_<prop>.pt      EGNNScalar 128x4, trained on train_a
    f_B = proj1/checkpoints/f_B_<prop>.pt      EGNNScalar 128x4, trained on train_b

`guidance_sweep.py` loads exactly these and nothing else. The guide/evaluator is
a nuisance variable: varying it would multiply the grid and make every number
conditional on a choice the reader has to track.

**Why EGNN on both sides, when we also have a transformer.** Three reasons, in
order:

1. **It is the most accurate on both sides for every property** (table below),
   and f_B's accuracy sets `delta = 2 x MAE`. Using the transformer as evaluator
   would widen the mu band from 0.168 to 0.243 D -- 45% looser -- which flatters
   every arm and compresses exactly the differences the experiment measures.
2. **It is what published work does.** EDM uses an EGNN classifier on both
   sides; EEGSDE uses Satorras' EGNN for the oracle and an EGNN energy for the
   guide; TFG reuses EEGSDE's oracle and trains its own EGNN guide. Same
   architecture on both sides is standard; the **disjoint data split** is the
   protection against reward hacking, and `SPLIT_PROTOCOL.md` guarantees it.
3. Architecture diversity between f_A and f_B is our own extra-rigour idea, not
   a requirement, and it costs measurement sensitivity.

**The cost, disclosed.** f_A and f_B share architectural blind spots, so
guidance that exploits a quirk of the EGNN inductive bias could fool both. The
disjoint split makes that unlikely, not impossible.

**The mitigation, and it is cheap.** Once the sweep names a winner, re-score
**only that arm's samples** with the transformer f_B
(`f_B_<prop>_transformer.pt`, a genuinely different inductive bias). One extra
run. If the advantage survives a different evaluator architecture, the claim is
much stronger; if it does not, that is the finding. This is a post-hoc
robustness check on the winner, never a second primary result.

**Not to be used as f_A, ever:** `ridge`. At 0.80 D on mu it is 9x worse than
the EGNN, and when f_A is much worse than f_B guidance cannot reach the band no
matter how good the method is -- the experiment would measure f_A's weakness
instead of the guidance. `proj1/scripts/predictor_table.py` flags this pairing
automatically as a "delta trap". Ridge exists only as the accuracy floor.

"""
assert s.count(anchor) == 1
s = s.replace(anchor, add, 1)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("plan: f_A/f_B pair locked to EGNN/EGNN with the transformer as a post-hoc check")
