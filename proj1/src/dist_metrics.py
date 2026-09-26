"""Metrics for the `dist` protocol: per-molecule property targets on QM9.

THE PROTOCOL THIS SCORES (docs/protocol/DIST_PROTOCOL.md). Molecule i of a
batch is generated at the atom count M_i of held-out test molecule i, and its
target is that same molecule's real property c_i. That is the conditional
protocol of EDM (Hoogeboom et al. 2022, sec. 5.2 and app. E: "first sampling
c, M ~ p(c, M) and then x, h ~ p(x, h | c, M)") and EEGSDE (Bao et al. 2023,
app. F.3: "M ~ p(M) and ... c ~ p(c | M)"), with (c, M) taken as empirical
pairs from `test` rather than a parametric fit on the training partition.

WHY ATOM COUNT IS THE CENTRAL CONFOUND. Size alone carries a large share of
the property: corr(size, alpha) = +0.755 on QM9. A generator that is handed
M_i and IGNORES c_i therefore already "hits" part of every target. EDM built
its baselines around exactly this, and states the reading criterion
verbatim:

    "If 'EDM' overcomes 'Naive (Upper-Bound)' it should be able to incorporate
     conditional property information into the generated molecules. If it
     overcomes '#Atoms' it should be able to incorporate it into the molecular
     structure beyond the number of atoms."

So every metric here is read against a LADDER of references that each know a
different amount, all computed from real data and the evaluator alone -- no
sampling, no RNG, exact expectations:

    U-bound        f_B(x_j) vs c_i, j any other test molecule      knows nothing
    size-shuffle   f_B(x_j) vs c_i, j another molecule of size M_i knows M, not c
    #Atoms         median(c | M) from train_a, vs c_i              knows M, point
    L-bound        f_B(x_i) vs c_i on the real molecule            evaluator floor

U-bound, #Atoms and L-bound are EDM's rows (and EEGSDE's, which reuses them).
SIZE-SHUFFLE IS OURS, and it is the fair "ignores the target" reference for a
SAMPLER. #Atoms is a deterministic point predictor; any method that draws
from p(c | M) instead is penalised by its own spread -- for a Gaussian law,
E|c - c'| is sqrt(2) times E|c - median|. A perfect generator that ignores
the target lands on size-shuffle, not on #Atoms, so size-shuffle is where
"no target information" sits for every arm in this project. #Atoms is kept
because it is the published bar and because beating it is EDM's stated test.

A proportion of the target information captured is then

    gap_closure = (size_shuffle - MAE) / (size_shuffle - L_bound)

0 = behaves like a size-aware generator that ignores c; 1 = as good as scoring
the real molecule. It is comparable across properties, which raw MAE is not.

Pure numpy on purpose: the sidecars are torch tensors, converted at entry, so
the statistics can be unit-tested without a GPU or a checkpoint.
"""
from __future__ import annotations

import math

import numpy as np

HARTREE_TO_MEV = 27211.386245988

# Published units, so a number can be placed beside EDM/EEGSDE directly.
# Our gap is stored in Hartree; both papers report it in meV.
PUBLISHED_UNIT = {"mu": ("D", 1.0), "alpha": ("Bohr^3", 1.0),
                  "gap": ("meV", HARTREE_TO_MEV)}

# Verified against the paper PDFs, 22 Sep 2026. EDM Table 3; EEGSDE Table 1
# (conditional network) and Table 5 (UNCONDITIONAL network + guidance -- the
# row that matches our setting: a frozen unconditional generator plus a
# training-free guide). Units as published.
PUBLISHED = {
    "mu": {"EDM U-bound": 1.616, "EDM #Atoms": 1.053,
           "EDM conditional": 1.111, "EDM L-bound": 0.043,
           "EEGSDE uncond+guide s=0.1": 1.415,
           "EEGSDE uncond+guide s=0.5": 1.241,
           "EEGSDE cond+guide best (s=2)": 0.777},
    "alpha": {"EDM U-bound": 9.01, "EDM #Atoms": 3.86,
              "EDM conditional": 2.76, "EDM L-bound": 0.10,
              "EEGSDE cond+guide best (s=3)": 2.50},
    "gap": {"EDM U-bound": 1470.0, "EDM #Atoms": 866.0,
            "EDM conditional": 655.0, "EDM L-bound": 64.0,
            "EEGSDE cond+guide s=1": 542.0},
}

# Steering-demand strata: how far each target sits from a typical molecule
# OF ITS OWN SIZE, in within-size sd. Separates concentration from steering
# inside the dist task (FR7 makes the same distinction across q50/q90).
DEMAND_EDGES = (0.5, 1.0)


def to_np(x):
    """Tensor, list or array -> numpy, without importing torch here."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# exact expectations over a reference set, by sorted prefix sums
# ---------------------------------------------------------------------------

class _SortedRef:
    """E|a_j - c| and P(|a_j - c| <= delta) over a fixed set {a_j}, exactly.

    Sorted once; each query is two binary searches. For sorted a with prefix
    sums S, and k = #{a_j <= c}:
        sum_j |a_j - c| = c k - S[k] + (S[n] - S[k]) - c (n - k)
    """

    def __init__(self, a):
        self.a = np.sort(np.asarray(a, dtype=np.float64))
        self.S = np.concatenate([[0.0], np.cumsum(self.a)])
        self.n = self.a.size

    def abs_sum(self, c):
        c = np.asarray(c, dtype=np.float64)
        k = np.searchsorted(self.a, c, side="right")
        return c * k - self.S[k] + (self.S[-1] - self.S[k]) - c * (self.n - k)

    def count_within(self, c, delta):
        c = np.asarray(c, dtype=np.float64)
        return (np.searchsorted(self.a, c + delta, side="right")
                - np.searchsorted(self.a, c - delta, side="left"))


def _atom_median_table(y_train, m_train):
    """median / mean / sd of the property for each atom count in train_a."""
    y_train, m_train = np.asarray(y_train, float), np.asarray(m_train, int)
    tab = {}
    for M in np.unique(m_train):
        v = y_train[m_train == M]
        tab[int(M)] = (float(np.median(v)), float(v.mean()),
                       float(v.std()) if v.size > 1 else float("nan"), v.size)
    return tab


def _lookup(tab, M):
    """Nearest atom count present in the table (sizes 4-7 are near-empty)."""
    if M in tab:
        return tab[M], False
    keys = np.array(sorted(tab))
    return tab[int(keys[np.argmin(np.abs(keys - M))])], True


def reference_ladder(y_test, fB_test, m_test, dist_pos, y_train, m_train,
                     delta):
    """The four references for one dist set, as exact expectations.

    y_test, fB_test, m_test : the WHOLE test split (true property, evaluator
                              prediction on the real molecule, atom count)
    dist_pos                : positions within test used as the dist set
    y_train, m_train        : train_a, for the #Atoms predictor (never test)
    delta                   : the in-band tolerance

    Returns MAE and in-band for each rung, plus the within-size tracking of
    the real molecules (the ceiling for `partial_corr`).
    """
    y_test = np.asarray(y_test, float)
    fB_test = np.asarray(fB_test, float)
    m_test = np.asarray(m_test, int)
    pos = np.asarray(dist_pos, int)
    c = y_test[pos]
    self_err = np.abs(fB_test[pos] - c)
    n_all = y_test.size

    # U-bound: every OTHER test molecule, uniformly
    ref = _SortedRef(fB_test)
    u_mae = (ref.abs_sum(c) - self_err) / (n_all - 1)
    u_in = (ref.count_within(c, delta) - (self_err <= delta)) / (n_all - 1)

    # size-shuffle: every OTHER test molecule of the same atom count
    s_mae = np.full(pos.size, np.nan)
    s_in = np.full(pos.size, np.nan)
    for M in np.unique(m_test[pos]):
        grp = np.nonzero(m_test == M)[0]
        if grp.size < 2:
            continue                        # no other molecule of this size
        r = _SortedRef(fB_test[grp])
        sel = np.nonzero(m_test[pos] == M)[0]
        cs, se = c[sel], self_err[sel]
        s_mae[sel] = (r.abs_sum(cs) - se) / (grp.size - 1)
        s_in[sel] = (r.count_within(cs, delta) - (se <= delta)) / (grp.size - 1)
    s_ok = np.isfinite(s_mae)

    # #Atoms: a point predictor of c from M alone, fit on train_a
    tab = _atom_median_table(y_train, m_train)
    med = np.empty(pos.size)
    mean = np.empty(pos.size)
    n_fallback = 0
    for k, M in enumerate(m_test[pos]):
        (md, mn, _, _), fell = _lookup(tab, int(M))
        med[k], mean[k] = md, mn
        n_fallback += fell
    a_err = np.abs(med - c)
    a_err_mean = np.abs(mean - c)

    return {
        "n": int(pos.size),
        "u_bound": {"mae": float(u_mae.mean()), "in_band": float(u_in.mean())},
        "size_shuffle": {"mae": float(s_mae[s_ok].mean()),
                         "in_band": float(s_in[s_ok].mean()),
                         "n_no_same_size": int((~s_ok).sum())},
        "atoms_median": {"mae": float(a_err.mean()),
                         "in_band": float((a_err <= delta).mean()),
                         "n_size_fallback": int(n_fallback)},
        "atoms_mean": {"mae": float(a_err_mean.mean()),
                       "in_band": float((a_err_mean <= delta).mean())},
        "l_bound": {"mae": float(self_err.mean()),
                    "in_band": float((self_err <= delta).mean()),
                    "partial_corr": partial_corr(fB_test[pos], c, m_test[pos])},
        "delta": float(delta),
    }


# ---------------------------------------------------------------------------
# tracking: does the output follow the per-molecule target?
# ---------------------------------------------------------------------------

def _rank(x):
    """Average ranks, ties shared -- Spearman without scipy."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size)
    r[order] = np.arange(x.size, dtype=float)
    xs = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def partial_corr(pred, target, n_atoms, min_group=3):
    """Correlation of prediction with target WITHIN atom-count groups.

    Both variables are centred on their own group mean and the residuals
    pooled. This is EDM's "beyond the number of atoms" criterion made
    continuous: a generator that uses M but ignores c scores ~0 here however
    well it matches #Atoms, because all of its tracking is between groups.
    Groups smaller than `min_group` carry no within-group signal and are
    dropped rather than contributing a residual of exactly zero.
    """
    p = np.asarray(pred, float)
    t = np.asarray(target, float)
    m = np.asarray(n_atoms, int)
    ok = np.isfinite(p) & np.isfinite(t)
    p, t, m = p[ok], t[ok], m[ok]
    rp, rt = [], []
    for M in np.unique(m):
        g = m == M
        if g.sum() < min_group:
            continue
        rp.append(p[g] - p[g].mean())
        rt.append(t[g] - t[g].mean())
    if not rp:
        return float("nan")
    return _pearson(np.concatenate(rp), np.concatenate(rt))


def steering_demand(y, n_atoms, y_train, m_train):
    """|c_i - median(c | M_i)| / sd(c | M_i), from train_a.

    How far each target is from a typical molecule of its OWN size. The size
    is already fixed by the protocol, so this is the steering the guidance
    actually has to supply -- not distance from the global mean.
    """
    tab = _atom_median_table(y_train, m_train)
    glob_sd = float(np.std(y_train))
    out = np.empty(len(y))
    for k, (c, M) in enumerate(zip(np.asarray(y, float),
                                   np.asarray(n_atoms, int))):
        (md, _, sd, cnt), _ = _lookup(tab, int(M))
        sd = sd if (cnt >= 5 and sd == sd and sd > 0) else glob_sd
        out[k] = abs(c - md) / sd
    return out


# ---------------------------------------------------------------------------
# one cell
# ---------------------------------------------------------------------------

def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (mid - half, mid + half)


def _boot_mean_ci(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def cell_metrics(pm, delta, ladder, prop, y_train=None, m_train=None,
                 ref_mol_stability=None, atom_stability=None, n_boot=2000):
    """Every dist metric for one cell, from its per-molecule sidecar.

    pm                : the `.permol.pt` dict (f_A, f_B, y, finite,
                        mol_stable, valid, n_atoms, mol_idx, optional smiles)
    ladder            : reference_ladder() for the SAME dist set
    ref_mol_stability : unguided's molecule stability, for the chemistry floor
    """
    fA, fB, y = to_np(pm["f_A"]).astype(float), to_np(pm["f_B"]).astype(float), \
        to_np(pm["y"]).astype(float)
    fin = to_np(pm["finite"]).astype(bool) & np.isfinite(fB)
    stab = to_np(pm["mol_stable"]).astype(bool)
    valid = to_np(pm["valid"]).astype(bool)
    m = to_np(pm["n_atoms"]).astype(int)
    n = y.size
    unit, scale = PUBLISHED_UNIT.get(prop, ("", 1.0))

    err = np.where(fin, np.abs(fB - y), np.inf)
    inb = fin & (err <= delta)
    e = err[fin]
    mae = float(e.mean()) if e.size else float("nan")
    k_in = int(inb.sum())
    # Signed residual, split into its two parts: MAE conflates a distribution
    # that is off-centre with one that is centred but wide, and the two have
    # opposite remedies. `resid_sd` is what caps in-band coverage -- a perfectly
    # centred Gaussian of that width scores P(|Z| <= delta / resid_sd), however
    # well it is aimed. That ceiling is reported beside in_band by the results
    # page, so "guidance does not help" can be told apart from "guidance steers
    # but cannot concentrate".
    res = (fB - y)[fin]
    bias = float(res.mean()) if res.size else float("nan")
    resid_sd = float(res.std(ddof=1)) if res.size > 1 else float("nan")

    out = {
        "n": int(n), "n_nonfinite": int((~fin).sum()),
        "mae": mae,
        "mae_ci": _boot_mean_ci(e, n_boot),
        "rmse": float(np.sqrt((e ** 2).mean())) if e.size else float("nan"),
        "bias": bias, "resid_sd": resid_sd,
        "in_band_ceiling": (2.0 * 0.5 * (1.0 + math.erf(
            delta / resid_sd / math.sqrt(2.0))) - 1.0)
        if resid_sd == resid_sd and resid_sd > 0 else float("nan"),
        "mae_published_units": mae * scale, "published_unit": unit,
        "in_band": k_in / n, "in_band_ci": _wilson(k_in, n),
        "mol_stability": float(stab.mean()),
        "validity": float(valid.mean()),
        "atom_stability": atom_stability,
    }

    # ---- the ladder: where this arm sits between "knows nothing" and truth
    ss, lb = ladder["size_shuffle"]["mae"], ladder["l_bound"]["mae"]
    out["gap_closure"] = (ss - mae) / (ss - lb) if ss > lb else float("nan")
    out["beats_atoms"] = bool(mae < ladder["atoms_median"]["mae"])
    out["beats_size_shuffle"] = bool(mae < ss)
    out["beats_u_bound"] = bool(mae < ladder["u_bound"]["mae"])
    out["in_band_gain_vs_size_shuffle"] = (out["in_band"]
                                           - ladder["size_shuffle"]["in_band"])

    # ---- tracking
    out["pearson"] = _pearson(fB[fin], y[fin])
    out["spearman"] = _pearson(_rank(fB[fin]), _rank(y[fin]))
    out["partial_corr"] = partial_corr(fB[fin], y[fin], m[fin])
    out["partial_corr_ceiling"] = ladder["l_bound"]["partial_corr"]

    # ---- the joint yield the project's headline is defined on:
    # distinct valid molecules inside the band, per molecule ATTEMPTED
    vib = valid & inb
    out["valid_in_band"] = float(vib.mean())
    smi = pm.get("smiles")
    if smi is not None:
        good = {s for s, ok in zip(smi, vib) if ok and s}
        out["distinct_valid_in_band_per_attempt"] = len(good) / n
    else:
        out["distinct_valid_in_band_per_attempt"] = None

    # ---- chemistry floor (the rubric: mol_stability >= 0.9 x unguided)
    if ref_mol_stability is not None and ref_mol_stability > 0:
        out["chem_ratio"] = out["mol_stability"] / ref_mol_stability
        out["passes_chem_floor"] = bool(out["chem_ratio"] >= 0.9)
    else:
        out["chem_ratio"] = out["passes_chem_floor"] = None

    # ---- reward hacking: the guide says in band, the evaluator disagrees
    errA = np.where(fin, np.abs(fA - y), np.inf)
    out["guide_eval_gap"] = float(np.abs(fA - fB)[fin].mean()) if fin.any() \
        else float("nan")
    out["hack_rate"] = float(((errA <= delta) & ~inb & fin).mean())
    out["guide_in_band"] = float((errA <= delta).mean())

    # ---- strata: an arm can win overall by working on one size only
    terc = np.quantile(m, [1 / 3, 2 / 3])
    size_bin = np.digitize(m, terc, right=True)
    out["in_band_by_size"] = {
        "small (<=%d)" % terc[0]: float(inb[size_bin == 0].mean())
        if (size_bin == 0).any() else None,
        "mid": float(inb[size_bin == 1].mean()) if (size_bin == 1).any() else None,
        "large (>%d)" % terc[1]: float(inb[size_bin == 2].mean())
        if (size_bin == 2).any() else None,
    }
    if y_train is not None and m_train is not None:
        dem = steering_demand(y, m, y_train, m_train)
        lo, hi = DEMAND_EDGES
        strata = {"near (<=%.1f sd)" % lo: dem <= lo,
                  "mid": (dem > lo) & (dem <= hi),
                  "far (>%.1f sd)" % hi: dem > hi}
        out["in_band_by_demand"] = {
            k: {"in_band": float(inb[s].mean()) if s.any() else None,
                "n": int(s.sum())} for k, s in strata.items()}
        out["mean_demand_sd"] = float(dem.mean())
    return out


# ---------------------------------------------------------------------------
# two arms, paired
# ---------------------------------------------------------------------------

def _norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2.0))


def _binom_two_sided(k, n):
    """Exact two-sided sign test p for k successes in n at p = 1/2."""
    if n == 0:
        return 1.0
    if n > 1000:
        return _norm_p((k - n / 2.0) / math.sqrt(n / 4.0))
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1))
    return min(1.0, 2.0 * tail / (2.0 ** n))


def _keys(pm, seed):
    """(seed, mol_idx) per row. A pooled sidecar carries its own per-row
    `seed`, so replicate seeds of the same molecule never collide."""
    idx = to_np(pm["mol_idx"]).astype(np.int64)
    if "seed" in pm:
        return [(int(s), int(i)) for s, i in zip(to_np(pm["seed"]), idx)]
    return [(seed, int(i)) for i in idx]


POOL_KEYS = ("f_A", "f_B", "y", "finite", "mol_stable", "valid", "n_atoms",
             "mol_idx")


def pool(sidecars):
    """Concatenate one arm's sidecars across seeds into one paired sample.

    `sidecars` is [(seed, pm), ...]. Seeds are replicates of the same dist
    set -- same molecules, same targets, different initial noise -- so the
    pre-registered comparison is ONE test on the pooled rows, not one test per
    seed that a multiplicity correction then has to pay for three times. Each
    row keeps its seed, so pairing is still exact on (seed, mol_idx).
    """
    out = {k: np.concatenate([to_np(pm[k]) for _, pm in sidecars])
           for k in POOL_KEYS}
    out["seed"] = np.concatenate([np.full(len(to_np(pm["y"])), int(s))
                                  for s, pm in sidecars])
    if all("smiles" in pm for _, pm in sidecars):
        out["smiles"] = [x for _, pm in sidecars for x in pm["smiles"]]
    return out


def paired_compare(pm_a, pm_b, delta, seed_a=0, seed_b=0, n_boot=2000):
    """Arm A against arm B on the SAME molecules: same size, same target,
    same initial noise.

    Rows are matched on (seed, mol_idx), never on position, and the targets of
    matched rows are REQUIRED to agree. That guard is not decorative: the
    batch loop once sliced every batch's targets from index 0, which would
    have guided 75% of molecules toward another molecule's target while
    scoring them against their own (DIST fix note, 22 Sep). A pairing that
    silently compared different targets would reproduce that failure here.

    Returns, for A minus B (negative = A better on error):
      dMAE    mean paired difference in |error|, bootstrap CI, paired z and p
      in_band discordant counts and an exact sign (McNemar) test
    """
    ka, kb = _keys(pm_a, seed_a), _keys(pm_b, seed_b)
    ib = {k: j for j, k in enumerate(kb)}
    pairs = [(i, ib[k]) for i, k in enumerate(ka) if k in ib]
    if not pairs:
        raise ValueError("no molecules in common -- different dist sets or seeds")
    ia = np.array([p[0] for p in pairs])
    jb = np.array([p[1] for p in pairs])

    ya = to_np(pm_a["y"]).astype(float)[ia]
    yb = to_np(pm_b["y"]).astype(float)[jb]
    if not np.allclose(ya, yb, rtol=0, atol=1e-6):
        bad = int((~np.isclose(ya, yb, rtol=0, atol=1e-6)).sum())
        raise ValueError("%d matched rows have DIFFERENT targets -- the two "
                         "cells were not run on the same dist set" % bad)

    def errs(pm, ix):
        fb = to_np(pm["f_B"]).astype(float)[ix]
        fin = to_np(pm["finite"]).astype(bool)[ix] & np.isfinite(fb)
        return np.where(fin, np.abs(fb - to_np(pm["y"]).astype(float)[ix]),
                        np.inf), fin

    ea, fa = errs(pm_a, ia)
    eb, fb_ = errs(pm_b, jb)
    both = fa & fb_
    d = ea[both] - eb[both]
    n_p = int(d.size)
    sd = float(d.std(ddof=1)) if n_p > 1 else float("nan")
    z = float(d.mean() / (sd / math.sqrt(n_p))) if n_p > 1 and sd > 0 else 0.0

    ina, inb = fa & (ea <= delta), fb_ & (eb <= delta)
    a_only = int((ina & ~inb).sum())
    b_only = int((~ina & inb).sum())

    return {
        "n_matched": int(len(pairs)),
        "n_paired_finite": n_p,
        "n_excluded_nonfinite": int(len(pairs) - n_p),
        "d_mae": float(d.mean()) if n_p else float("nan"),
        "d_mae_ci": _boot_mean_ci(d, n_boot),
        "d_mae_z": z, "d_mae_p": _norm_p(z),
        "in_band_a": float(ina.mean()), "in_band_b": float(inb.mean()),
        "d_in_band": float(ina.mean() - inb.mean()),
        "in_band_a_only": a_only, "in_band_b_only": b_only,
        "in_band_p": _binom_two_sided(a_only, a_only + b_only),
    }


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a dict {name: p}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - rank) * p))
        adj[name] = running
    return adj
