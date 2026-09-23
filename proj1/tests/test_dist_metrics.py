"""Gates for the dist-protocol metrics.

Every check compares against a brute-force computation or a closed-form
answer, not against "it runs". The ones that matter most:

  * the ladder's exact expectations equal the explicit double loop
  * size-shuffle sits at sqrt(2) x #Atoms for a Gaussian within-size law --
    the reason size-shuffle, not #Atoms, is the fair reference for a sampler
  * a generator that uses size but ignores the target gets partial_corr ~ 0
    and gap_closure ~ 0; scoring the real molecules gets gap_closure = 1
  * paired_compare matches on mol_idx, not position, and REFUSES to pair rows
    whose targets differ (the batch-slicing bug that motivated it)

Run: python proj1/tests/test_dist_metrics.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
import dist_metrics as dm  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %-58s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


rng = np.random.default_rng(0)

# ---- 1. sorted-prefix expectations == brute force ------------------------
a = rng.normal(size=500)
c = rng.normal(size=50)
ref = dm._SortedRef(a)
bf_sum = np.array([np.abs(a - ci).sum() for ci in c])
bf_cnt = np.array([(np.abs(a - ci) <= 0.3).sum() for ci in c])
check("abs_sum == brute force", bool(np.allclose(ref.abs_sum(c), bf_sum)),
      "max err %.1e" % np.abs(ref.abs_sum(c) - bf_sum).max())
check("count_within == brute force",
      bool(np.array_equal(ref.count_within(c, 0.3), bf_cnt)))

# ---- a synthetic QM9: property = size effect + Gaussian within-size noise --
N_TEST, N_TRAIN, SIG = 4000, 20000, 1.0


def synth(n):
    m = rng.integers(10, 20, size=n)
    y = 0.5 * m + rng.normal(0, SIG, size=n)
    return m, y


m_te, y_te = synth(N_TEST)
m_tr, y_tr = synth(N_TRAIN)
fB_te = y_te + rng.normal(0, 0.05, size=N_TEST)   # a good evaluator
pos = np.arange(300)
delta = 0.2
lad = dm.reference_ladder(y_te, fB_te, m_te, pos, y_tr, m_tr, delta)

# ---- 2. ladder == explicit double loop -----------------------------------
u_bf = np.mean([np.mean(np.abs(np.delete(fB_te, i) - y_te[i])) for i in pos])
s_bf = []
for i in pos:
    g = np.nonzero(m_te == m_te[i])[0]
    g = g[g != i]
    s_bf.append(np.mean(np.abs(fB_te[g] - y_te[i])))
s_bf = float(np.mean(s_bf))
check("U-bound exact expectation == double loop",
      abs(lad["u_bound"]["mae"] - u_bf) < 1e-9,
      "%.6f vs %.6f" % (lad["u_bound"]["mae"], u_bf))
check("size-shuffle exact expectation == double loop",
      abs(lad["size_shuffle"]["mae"] - s_bf) < 1e-9,
      "%.6f vs %.6f" % (lad["size_shuffle"]["mae"], s_bf))
check("L-bound == evaluator's own error on the real molecules",
      abs(lad["l_bound"]["mae"] - np.abs(fB_te[pos] - y_te[pos]).mean()) < 1e-12)

# ---- 3. the ladder is ordered, and size-shuffle / #Atoms ~ sqrt(2) --------
L = lad
ordered = (L["u_bound"]["mae"] > L["size_shuffle"]["mae"]
           > L["atoms_median"]["mae"] > L["l_bound"]["mae"])
check("ladder ordered U > size-shuffle > #Atoms > L", ordered,
      "%.3f > %.3f > %.3f > %.3f" % (L["u_bound"]["mae"],
                                     L["size_shuffle"]["mae"],
                                     L["atoms_median"]["mae"],
                                     L["l_bound"]["mae"]))
ratio = L["size_shuffle"]["mae"] / L["atoms_median"]["mae"]
check("size-shuffle / #Atoms ~ sqrt(2) for a Gaussian within-size law",
      abs(ratio - math.sqrt(2)) < 0.12, "ratio %.3f vs %.3f" % (ratio, math.sqrt(2)))
check("#Atoms MAE ~ sigma*sqrt(2/pi) (median of a Gaussian)",
      abs(L["atoms_median"]["mae"] - SIG * math.sqrt(2 / math.pi)) < 0.08,
      "%.3f vs %.3f" % (L["atoms_median"]["mae"], SIG * math.sqrt(2 / math.pi)))

# ---- 4. partial correlation isolates "beyond atom count" ------------------
size_only = 0.5 * m_te[pos] + rng.normal(0, 0.01, size=pos.size)
pc_size = dm.partial_corr(size_only, y_te[pos], m_te[pos])
pr_size = dm._pearson(size_only, y_te[pos])
check("size-only predictor: high Pearson, partial_corr ~ 0",
      pr_size > 0.5 and abs(pc_size) < 0.15,
      "pearson %.3f partial %.3f" % (pr_size, pc_size))
pc_true = dm.partial_corr(y_te[pos] + rng.normal(0, 0.05, pos.size),
                          y_te[pos], m_te[pos])
check("target-tracking predictor: partial_corr ~ 1", pc_true > 0.95,
      "partial %.3f" % pc_true)

# ---- 5. ranks with ties ---------------------------------------------------
check("average ranks with ties",
      np.array_equal(dm._rank([10, 20, 20, 30]), [0, 1.5, 1.5, 3]))


# ---- 6. cell_metrics on two limiting generators ---------------------------
def pm_from(fb_vals, idx, smiles=None, valid=None, fa=None):
    n = idx.size
    return {"f_A": fb_vals if fa is None else fa, "f_B": fb_vals,
            "y": y_te[idx], "finite": np.ones(n, bool),
            "mol_stable": np.ones(n, bool),
            "valid": np.ones(n, bool) if valid is None else valid,
            "n_atoms": m_te[idx], "mol_idx": idx,
            **({"smiles": smiles} if smiles is not None else {})}


# (a) "scores the real molecule": exactly the L-bound -> closure 1
cm_real = dm.cell_metrics(pm_from(fB_te[pos], pos), delta, lad, "mu",
                          y_tr, m_tr)
check("scoring the real molecules gives gap_closure = 1",
      abs(cm_real["gap_closure"] - 1.0) < 1e-9, "%.6f" % cm_real["gap_closure"])

# (b) "knows the size, ignores the target": f_B of a random SAME-SIZE molecule
ign = np.empty(pos.size)
for k, i in enumerate(pos):
    g = np.nonzero(m_te == m_te[i])[0]
    ign[k] = fB_te[rng.choice(g[g != i])]
cm_ign = dm.cell_metrics(pm_from(ign, pos), delta, lad, "mu", y_tr, m_tr)
check("size-aware, target-ignoring generator: gap_closure ~ 0",
      abs(cm_ign["gap_closure"]) < 0.15, "%.3f" % cm_ign["gap_closure"])
check("...and does NOT beat #Atoms (it samples, #Atoms is a point)",
      not cm_ign["beats_atoms"], "MAE %.3f vs #Atoms %.3f"
      % (cm_ign["mae"], lad["atoms_median"]["mae"]))
check("...and has partial_corr ~ 0",
      abs(cm_ign["partial_corr"]) < 0.2, "%.3f" % cm_ign["partial_corr"])

# ---- 7. joint yield counts DISTINCT valid in-band molecules ---------------
sm = ["C", "C", "CC", None] + ["X%d" % i for i in range(pos.size - 4)]
vd = np.array([True, True, True, False] + [False] * (pos.size - 4))
cm_y = dm.cell_metrics(pm_from(y_te[pos], pos, smiles=sm, valid=vd), delta,
                       lad, "mu")
check("distinct valid in-band: duplicates counted once",
      abs(cm_y["distinct_valid_in_band_per_attempt"] - 2 / pos.size) < 1e-12,
      "%.4f" % cm_y["distinct_valid_in_band_per_attempt"])

# ---- 8. reward hacking: guide in band, evaluator not ----------------------
fa_hack = y_te[pos].copy()                       # guide says: perfect
fb_off = y_te[pos] + 10 * delta                  # evaluator says: way off
cm_h = dm.cell_metrics(pm_from(fb_off, pos, fa=fa_hack), delta, lad, "mu")
check("hack_rate = 1 when the guide is fooled on every molecule",
      abs(cm_h["hack_rate"] - 1.0) < 1e-12, "%.3f" % cm_h["hack_rate"])

# ---- 9. paired_compare ----------------------------------------------------
pa = pm_from(fB_te[pos], pos)
pb = pm_from(fB_te[pos] + 0.3, pos)              # uniformly worse
res = dm.paired_compare(pa, pb, delta)
check("paired: detects a uniform shift (A better)",
      res["d_mae"] < 0 and res["d_mae_p"] < 1e-6,
      "dMAE %.3f p %.1e" % (res["d_mae"], res["d_mae_p"]))
same = dm.paired_compare(pa, pa, delta)
check("paired: identical arms -> dMAE 0, in-band p = 1",
      same["d_mae"] == 0.0 and same["in_band_p"] == 1.0)
perm = rng.permutation(pos.size)
pb_shuf = {k: (v[perm] if hasattr(v, "__len__") and not isinstance(v, str)
               else v) for k, v in pb.items()}
res_s = dm.paired_compare(pa, pb_shuf, delta)
check("paired: matches on mol_idx, not row position",
      abs(res_s["d_mae"] - res["d_mae"]) < 1e-12,
      "%.6f vs %.6f" % (res_s["d_mae"], res["d_mae"]))
pb_bad = dict(pb)
pb_bad["y"] = np.roll(pb["y"], 1)                # targets misaligned
try:
    dm.paired_compare(pa, pb_bad, delta)
    raised = False
except ValueError:
    raised = True
check("paired: REFUSES rows whose targets differ", raised)

# ---- 10. exact sign test and Holm ----------------------------------------
check("sign test k=0,n=10 -> p = 2/1024",
      abs(dm._binom_two_sided(0, 10) - 2 / 1024) < 1e-12)
check("sign test symmetric k=5,n=10 -> p = 1", dm._binom_two_sided(5, 10) == 1.0)
h = dm.holm({"a": 0.01, "b": 0.04, "c": 0.03})
check("Holm adjusted values (0.03, 0.06, 0.06)",
      abs(h["a"] - 0.03) < 1e-12 and abs(h["c"] - 0.06) < 1e-12
      and abs(h["b"] - 0.06) < 1e-12, str(h))

# ---- 11. steering demand strata partition the set ------------------------
dem = dm.steering_demand(y_te[pos], m_te[pos], y_tr, m_tr)
cm_d = dm.cell_metrics(pm_from(fB_te[pos], pos), delta, lad, "mu", y_tr, m_tr)
tot = sum(v["n"] for v in cm_d["in_band_by_demand"].values())
check("demand strata partition every molecule", tot == pos.size,
      "%d of %d" % (tot, pos.size))
check("mean demand ~ sqrt(2/pi) sd for a Gaussian law",
      abs(dem.mean() - math.sqrt(2 / math.pi)) < 0.12, "%.3f" % dem.mean())

# ---- 12. units: gap converts Hartree -> meV -------------------------------
cm_gap = dm.cell_metrics(pm_from(fB_te[pos], pos), delta, lad, "gap")
check("gap MAE reported in meV (x 27211.386)",
      abs(cm_gap["mae_published_units"] - cm_gap["mae"] * 27211.386245988) < 1e-6
      and cm_gap["published_unit"] == "meV")

# ---- 13. pooling seeds: replicates stay paired, never collide -------------
p1 = pm_from(fB_te[pos], pos)
p2 = pm_from(fB_te[pos] + 0.01, pos)
q1 = pm_from(fB_te[pos] + 0.2, pos)
q2 = pm_from(fB_te[pos] + 0.21, pos)
A = dm.pool([(1, p1), (2, p2)])
B = dm.pool([(1, q1), (2, q2)])
check("pool concatenates replicates with a per-row seed",
      A["y"].size == 2 * pos.size and set(A["seed"].tolist()) == {1, 2})
rp = dm.paired_compare(A, B, delta)
check("pooled pairing matches all 2n rows on (seed, mol_idx)",
      rp["n_matched"] == 2 * pos.size, "%d" % rp["n_matched"])
rp2 = dm.paired_compare(A, dm.pool([(2, q2), (1, q1)]), delta)
check("pooled pairing is independent of seed order",
      abs(rp2["d_mae"] - rp["d_mae"]) < 1e-12)
# What pairing buys is VARIANCE, not the point estimate: the mean of paired
# differences is the difference of means, so a wrong pairing leaves d_mae
# unchanged and only destroys the power. Two seeds with independent outputs,
# B = A + 0.1 within each seed: paired correctly the differences are tight;
# with the seed labels swapped each molecule is compared against a different
# draw and the difference is mostly noise.
#
# B's errors are A's scaled by 1.3 -- a REAL effect. (An earlier version added
# a constant offset, which leaves a zero-mean error's MAE unchanged in
# expectation: both z came out ~0 and the gate passed on nothing. Hence the
# floor on the correct z below: this gate must not be able to pass empty.)
# Expected: correct |z| ~ 32 (d = 0.3|s|, tight); swapped ~ 6.
s1 = rng.normal(0, 1.0, pos.size)
s2 = rng.normal(0, 1.0, pos.size)
A2 = dm.pool([(1, pm_from(y_te[pos] + s1, pos)),
              (2, pm_from(y_te[pos] + s2, pos))])
Bok = dm.pool([(1, pm_from(y_te[pos] + 1.3 * s1, pos)),
               (2, pm_from(y_te[pos] + 1.3 * s2, pos))])
Bsw = dm.pool([(1, pm_from(y_te[pos] + 1.3 * s2, pos)),
               (2, pm_from(y_te[pos] + 1.3 * s1, pos))])
r_ok, r_sw = dm.paired_compare(A2, Bok, delta), dm.paired_compare(A2, Bsw, delta)
check("pairing leaves d_mae unchanged (it is the difference of means)",
      abs(r_ok["d_mae"] - r_sw["d_mae"]) < 1e-12)
check("correct seed pairing gives the power: |z| >> a wrong pairing",
      abs(r_ok["d_mae_z"]) > 10
      and abs(r_ok["d_mae_z"]) > 3 * abs(r_sw["d_mae_z"]),
      "|z| %.1f correct vs %.1f swapped"
      % (abs(r_ok["d_mae_z"]), abs(r_sw["d_mae_z"])))

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  FAILED: " + f)
sys.exit(1 if FAIL else 0)
