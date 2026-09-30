"""The v3-final run (blade, n = 2000), summarised one page per component.

    python proj1/scripts/v3_final_summary.py --md-out docs/results/V3_FINAL_SUMMARY.md

WHAT THIS IS. The page a teammate reads before writing the paper. Four
components, in this order, each with the same structure (what ran, a
full-metric table, head-to-heads with verdicts at the pre-registered bar,
chemistry or fidelity cost, cost in time, outcome, caveats):

  1. QM9 diffusion  = the `edm` backend: TFG's released EDMsecond checkpoint,
                      BORROWED, scored with our property pair; unguided + plug.
  1b. `vp`          = our OWN trained VP diffusion model (28 Sep): same pair,
                      unguided + plug, 18 cells. The matched partner to `fm`.
  2. `fm`           = our flow-matching EGNN + our pair: headline + ablation.
  3. `equifm`       = EquiFM (borrowed) + TFG's pair: headline + ablation.
  4. Modality 2     = simplex flow matching on DeepFlyBrain enhancers (gc, cpg).

then "Across the run" (verdict scoreboard, costs, what is pending, pointers)
and a consistency section that re-derives rows of the generated docs this page
must agree with and refuses to write if an M1 row disagrees.

WHAT IT READS -- committed stats JSONs only:
  results/v3/<be>/v3/n2000/seed2026100{1,2,3}/tr__*.json      M1 headline
  results/v3/<be>/v3abl/n2000/seed2026100{1,2,3}/tr__*.json   M1 ablation
  results/m2/{m2,m2abl}/n2000/seed2026092{1,2,3}/*.json   M2 (m2wsweep: seed 20260921 only)
  results/m2_share.json, m2_gate_scores.json, m2_dfb_activity.json (M2 diagnostics)
It never reads results/v3/<be>/n5000/ (the Betty run), v2, q90, results/full,
results/sweep or the BDG port trees.

STATISTICS. FULL_RUN_V3_PROTOCOL.md section 6.1: unpaired binomial
se = sqrt(p(1-p)/N) per arm, N = 3n = 6000, contrast z = d / sqrt(se_a^2 +
se_b^2), Bonferroni 0.05/18 two-sided -> z = 2.99. MODALITY2_V3_PROTOCOL.md
states no decision rule (section 3.1: "no verdict is computed"), so M2 contrasts
use the same unpaired z = 2.99 and the page says so.

It REFUSES (sys.exit with the reason) on a missing, duplicated or unexpected
cell, on a cell whose configuration differs from the run's, and on any M1 row
that disagrees with V3_RESULTS*.md or V3_BLADE_READOUT.md.

M1 statistics reuse v3_table.load/pooled and v3_blade_readout's helpers, so
the numbers here are computed by the same code as the docs they must match.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import v3_table as V                                   # noqa: E402
import v3_blade_readout as BR                          # noqa: E402

# The detailed M2 page's builder. Its loader and pooling are used when it
# imports cleanly, so this page and M2_V3_RESULTS.md define every M2 number the
# same way; otherwise the cells are read here with the same definitions.
# Appended, not prepended, so proj1/m2 cannot shadow anything imported later.
sys.path.append(os.path.join(ROOT, "proj1", "m2"))
try:
    import m2_v3_results as M2R                        # noqa: E402
    if not (callable(getattr(M2R, "load", None)) and callable(getattr(M2R, "pooled", None))):
        raise ImportError("m2_v3_results has no load()/pooled()")
    M2R_ERR = None
except Exception as _exc:                              # noqa: BLE001
    M2R, M2R_ERR = None, "%s: %s" % (type(_exc).__name__, _exc)

NL = "\n"
CMD = "python proj1/scripts/v3_final_summary.py --md-out docs/results/V3_FINAL_SUMMARY.md"
N = 2000
BATCH = 500
M1_SEEDS = ["20261001", "20261002", "20261003"]
M2_SEEDS = [20260921, 20260922, 20260923]
PROPS = V.PROPS                                        # mu, alpha, gap
M2_PROPS = ("gc", "cpg")
HEAD_ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg", "bdg_e4t0.5", "bdg_e4t1")
# `vp` is OUR OWN QM9 diffusion, added 28 Sep. Two arms, mirroring `edm`, so
# vp-vs-EDMsecond is a legal head-to-head; see FULL_RUN_V3_PROTOCOL.md 1.3.
BACKEND_ARMS = {"fm": HEAD_ARMS, "equifm": HEAD_ARMS, "edm": ("unguided", "plug"),
                "vp": ("unguided", "plug")}
BDG_HEAD = BR.BDG_HEAD                                 # ("bdg_e4t0.5", "bdg_e4t1")
ETAS = (1, 2, 4, 8)
TAUS = BR.TAUS                                         # "0.5", "0.75", "1", "1.5"
ABL_ARMS = ("bdg_e0t1",) + tuple("bdg_e%dt%s" % (e, t) for e in ETAS for t in TAUS)
M2_HEAD_ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg_mc", "bdg_e4t0.5", "bdg_e4t1")
M2_WIN0_ARMS = ("plug", "bdg_e4t0.5", "bdg_e4t1")
M2_WSWEEP = (("plug", 16.0), ("plug", 64.0), ("bdg_e4t0.5", 16.0), ("bdg_e4t0.5", 64.0))
M2_CKPT = "fm_m2_dfb500.pt"
ALPHA, N_CONTRASTS = 0.05, 18
ZBAR = BR.ZBAR                                         # 2.99
BE_TITLE = {"edm": "QM9 diffusion: `edm` = TFG's EDMsecond (borrowed), our property pair",
            "fm": "Our flow-matching model: `fm` = our FM EGNN, our property pair",
            "equifm": "EquiFM: `equifm` = EquiFM (borrowed), TFG's property pair",
            "vp": "Our QM9 diffusion: `vp` = our VP diffusion EGNN (trained here), "
                  "our property pair -- the matched partner to `fm`"}
DAGGER = "&dagger;"


def die(msg):
    sys.exit("REFUSING: " + msg)


# ------------------------------------------------------------------ stats
def se(p, n):
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


zdiff = BR.zdiff
verdict = BR.verdict
pp = BR.pp


def mdd(pa, pb, Na, Nb):
    """Minimum detectable difference at the bar: ZBAR x the unpaired contrast se
    (v3_power.py's and m2_v3_results.py's definition). A `tie` is |d| < MDD by
    construction, so the MDD is what a tie cannot rule out."""
    return ZBAR * math.sqrt(se(pa, Na) ** 2 + se(pb, Nb) ** 2)


def mdd_txt(pa, pb, Na, Nb):
    return "%.2f" % (100 * mdd(pa, pb, Na, Nb))


def mean(xs):
    return sum(xs) / len(xs)


def tally(vs):
    return "%d / %d / %d" % (vs.count("above"), vs.count("below"), vs.count("tie"))


def arms_txt(arms):
    return ", ".join("`%s`" % a for a in arms) if arms else "none"


# ------------------------------------------------------------------ M1 load
def m1_load(root, be, stage, w=None):
    cells, problems = V.load(root, be, N, M1_SEEDS, stage=stage, w=w)
    if problems:
        die("%s/%s n%d w=%s: %s" % (be, stage, N, w, "; ".join(problems)))
    want = BACKEND_ARMS[be] if stage == "v3" else ABL_ARMS
    exp = {(p, a) for p in PROPS for a in want}
    missing, extra = sorted(exp - set(cells)), sorted(set(cells) - exp)
    if missing or extra:
        die("%s/%s w=%s: missing %s, unexpected %s" % (be, stage, w, missing, extra))
    want_w = 1.0 if stage == "v3" else float(w)
    for k, rows in cells.items():
        seeds = [r["_seed"] for r in rows]
        if seeds != M1_SEEDS:
            die("%s/%s %s: seeds %s (duplicated or missing cell)" % (be, stage, k, seeds))
        for r in rows:
            bad = [(f, r.get(f)) for f, v in (("n", N), ("batch", BATCH), ("steps", 100),
                                               ("solver", "euler"), ("target_name", "q50"))
                   if r.get(f) != v]
            if abs(float(r["t_start"]) - 0.5) > 1e-12 or abs(float(r["w"]) - want_w) > 1e-12:
                bad.append(("t_start/w", (r["t_start"], r["w"])))
            if r.get("oracle2") is None or r.get("guided_steps") is None or r.get("seconds") is None:
                bad.append(("oracle2/guided_steps/seconds", "absent"))
            if bad:
                die("%s/%s %s seed %s: %s" % (be, stage, k, r["_seed"], bad))
    return cells


def pool(rows):
    """v3_table.pooled plus the second oracle, the per-sample-step clip
    fraction, the seed-mean w_eff and the recorded time."""
    P = V.pooled(rows)
    Nn = P["N"]
    P["o2"] = sum(r["oracle2"]["in_band"] * r["n"] for r in rows) / Nn
    P["o2_dec"] = sum(r["oracle2"]["in_band_dec"] * r["n"] for r in rows) / Nn
    den = 0.0
    for r in rows:
        nb = r["n"] // r["batch"]
        g = int(r["guided_steps"])
        # guided_steps is SUMMED over the n/batch batches (transfer_sweep), so
        # the steps one sample is guided on is guided_steps / (n/batch)
        if g % nb:
            die("%s: guided_steps %d not divisible by n/batch %d" % (r["_file"], g, nb))
        den += (g // nb) * r["n"]
    P["clip_frac"] = (P["clipped"] / den) if den else 0.0
    P["guided_per_batch"] = int(rows[0]["guided_steps"]) // (rows[0]["n"] // rows[0]["batch"])
    P["w_eff_seeds"] = BR.w_eff(rows)
    P["secs"] = sum(float(r["seconds"]) for r in rows)
    P["secs_cell"] = P["secs"] / len(rows)
    return P


def pooled_all(cells):
    return {k: pool(v) for k, v in cells.items()}


# ------------------------------------------------------------------ M1 tables
def m1_metric_table(A, P, arms, label_w=None):
    A("| prop (delta) | arm | in_band +- se | dec | o2 | o2 dec | MAE/d | bias/d | sd/d | "
      "mol_stab | atom_stab | valid | uniq | DV | diversity |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PROPS:
        for a in arms:
            r = P[(p, a)]
            d = r["delta"]
            A("| %s (%.5f) | `%s` | %.4f +- %.4f | %.4f | %.4f | %.4f | %.3f | %+.3f | %.3f | "
              "%.4f | %.4f | %.4f | %.4f | %.4f | %.4f |"
              % (p, d, a, r["in_band_fraction"], r["se_ib"], r["in_band_fraction_dec"],
                 r["o2"], r["o2_dec"], r["prop_mae_eval"] / d, r["bias"] / d, r["sd"] / d,
                 r["mol_stability"], r["atom_stability"], r["validity"],
                 r["uniqueness_of_valid"], r["unique_valid_per_sample"],
                 r["diversity_mean_pairwise"]))
    A("")


def contrast(a, b):
    """d and z on continuous, decoded and second-oracle in-band, a minus b."""
    out = {}
    for key, lab in (("in_band_fraction", "cont"), ("in_band_fraction_dec", "dec"), ("o2", "o2")):
        d = a[key] - b[key]
        out[lab] = (d, zdiff(a[key], b[key], a["N"], b["N"]))
    d = a["mol_stability"] - b["mol_stability"]
    out["mol"] = (d, zdiff(a["mol_stability"], b["mol_stability"], a["N"], b["N"]))
    out["val"] = a["validity"] - b["validity"]
    out["dv"] = a["unique_valid_per_sample"] - b["unique_valid_per_sample"]
    return out


def m1_vs_unguided(A, P, arms, U=None, wlab=""):
    """Every guided arm against unguided. U overrides the unguided cells (the
    w = 4 rows are read against the headline's unguided, in the other tree)."""
    U = U or P
    A("| prop | arm%s | d in_band (pp) | z | verdict | MDD (pp) | d dec (pp) | z | verdict | d o2 (pp) "
      "| z | verdict | d mol_stab (pp) | z%s | d valid (pp) | d DV (pp) |" % (wlab, DAGGER))
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for p in PROPS:
        u = U[(p, "unguided")]
        for a in arms:
            if a == "unguided":
                continue
            r = P[(p, a)]
            c = contrast(r, u)
            res[(p, a)] = c
            A("| %s | `%s` | %s | %+.2f | %s | %s | %s | %+.2f | %s | %s | %+.2f | %s | %s | %+.2f | %s | %s |"
              % (p, a, pp(c["cont"][0]), c["cont"][1], verdict(c["cont"][1]),
                 mdd_txt(r["in_band_fraction"], u["in_band_fraction"], r["N"], u["N"]),
                 pp(c["dec"][0]), c["dec"][1], verdict(c["dec"][1]),
                 pp(c["o2"][0]), c["o2"][1], verdict(c["o2"][1]),
                 pp(c["mol"][0]), c["mol"][1], pp(c["val"]), pp(c["dv"])))
    A("")
    A("%s z on molecule stability uses the same unpaired binomial form but is **not "
      "pre-registered**; it is printed to size the chemistry cost, not to call a verdict. "
      "MDD = %.2f x the contrast se on continuous in-band: a `tie` is |d| < MDD by "
      "construction, so it is a limit on this run's resolution, not a measured equality."
      % (DAGGER, ZBAR))
    A("")
    return res


def m1_bdg_vs_plug_rows(head, abl4, Ph, Pa4):
    """(w, prop, arm, P_bdg, P_plug, rows_bdg, rows_plug); plug at w = 4 is
    bdg_e0t1 in the ablation tree (eta = 0 is plug's field)."""
    out = []
    for p in PROPS:
        for a in BDG_HEAD:
            out.append((1, p, a, Ph[(p, a)], Ph[(p, "plug")], head[(p, a)], head[(p, "plug")]))
    for p in PROPS:
        for a in BDG_HEAD:
            out.append((4, p, a, Pa4[(p, a)], Pa4[(p, "bdg_e0t1")], abl4[(p, a)],
                        abl4[(p, "bdg_e0t1")]))
    return out


def m1_bdg_vs_plug(A, rows):
    A("| w | prop | arm | in_band BDG / plug | d cont (pp) | z | v | MDD (pp) | d dec (pp) | z | v "
      "| d o2 (pp) | z | v | seeds BDG > plug | bias/d BDG / plug | sd/d BDG / plug "
      "| d mol_stab (pp) | d valid (pp) | d DV (pp) | clip % BDG / plug |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for (w, p, a, b, pl, rb, rp) in rows:
        c = contrast(b, pl)
        c["mdd"] = mdd(b["in_band_fraction"], pl["in_band_fraction"], b["N"], pl["N"])
        pos = sum(1 for x, y in zip(rb, rp) if x["in_band_fraction"] > y["in_band_fraction"])
        d = b["delta"]
        res[(w, p, a)] = c
        A("| %d | %s | `%s` | %.4f / %.4f | %s | %+.2f | %s | %.2f | %s | %+.2f | %s | %s | %+.2f | %s "
          "| %d of 3 | %+.3f / %+.3f | %.3f / %.3f | %s | %s | %s | %.2f / %.2f |"
          % (w, p, a, b["in_band_fraction"], pl["in_band_fraction"],
             pp(c["cont"][0]), c["cont"][1], verdict(c["cont"][1]), 100 * c["mdd"],
             pp(c["dec"][0]), c["dec"][1], verdict(c["dec"][1]),
             pp(c["o2"][0]), c["o2"][1], verdict(c["o2"][1]), pos,
             b["bias"] / d, pl["bias"] / pl["delta"], b["sd"] / d, pl["sd"] / pl["delta"],
             pp(c["mol"][0]), pp(c["val"]), pp(c["dv"]),
             100 * b["clip_frac"], 100 * pl["clip_frac"]))
    A("")
    ties = [c["mdd"] for c in res.values() if verdict(c["cont"][1]) == "tie"]
    if ties:
        A("MDD = %.2f x the contrast se on continuous in-band. The %d continuous ties above sit "
          "against MDDs of %.2f to %.2f pp: a tie is |d| < MDD by construction, so each is a limit "
          "on this run's resolution, **not a measured equality**. The `seeds BDG > plug` count is "
          "descriptive and not pre-registered." % (ZBAR, len(ties), 100 * min(ties), 100 * max(ties)))
        A("")
    return res


def abl_summary(abl, be):
    """The blade readout's 12-grid summary, per backend; plus e8t1.5's chemistry."""
    out = []
    for p in PROPS:
        for w in (1, 4):
            c = abl[(be, w)]
            P = lambda a: pool(c[(p, a)])
            base = P("bdg_e0t1")
            b0 = base["in_band_fraction"]
            xs, ys, row05 = [], [], []
            for tau in TAUS:
                for eta in ETAS:
                    a = BR.arm_name(eta, tau)
                    r = P(a)
                    d = r["in_band_fraction"] - b0
                    xs.append(BR.w_eff(c[(p, a)]))
                    ys.append(d)
                    if tau == "0.5":
                        row05.append(d)
            mono = all(row05[i + 1] >= row05[i] for i in range(len(row05) - 1)) and row05[0] >= 0
            ea, eb = P("bdg_e8t0.5"), P("bdg_e8t1.5")
            za = zdiff(ea["in_band_fraction"], b0, ea["N"], base["N"])
            zb = zdiff(eb["in_band_fraction"], b0, eb["N"], base["N"])
            out.append(dict(be=be, prop=p, w=w, b0=b0, mol0=base["mol_stability"], mono=mono,
                            da=ea["in_band_fraction"] - b0, za=za,
                            dma=ea["mol_stability"] - base["mol_stability"],
                            db=eb["in_band_fraction"] - b0, zb=zb,
                            dmb=eb["mol_stability"] - base["mol_stability"],
                            clipa=ea["clip_frac"], clip0=base["clip_frac"],
                            rho=BR.spearman(xs, ys)))
    return out


def blade_summary_row(s):
    """Exactly V3_BLADE_READOUT.md's 'Summary over the 12 grids' row."""
    return ("| %s | %s | %d | %s | %s | %+.2f | %s | %s | %+.2f | %+.2f |"
            % (s["be"], s["prop"], s["w"], "yes" if s["mono"] else "**no**", pp(s["da"]),
               s["za"], pp(s["dma"]), pp(s["db"]), s["zb"], s["rho"]))


def identity(head, abl1):
    """bdg_e0t1 (ablation, w=1) vs plug (headline), and the headline BDG arms
    re-run in the ablation: max |d| over properties and seeds."""
    keys = ("in_band_fraction", "in_band_fraction_dec", "mol_stability", "validity",
            "unique_valid_per_sample")
    mi = ms = ri = rs = 0.0
    allk = clipd = 0.0
    for p in PROPS:
        for x, y in zip(abl1[(p, "bdg_e0t1")], head[(p, "plug")]):
            mi = max(mi, abs(x["in_band_fraction"] - y["in_band_fraction"]))
            ms = max(ms, abs(x["mol_stability"] - y["mol_stability"]))
            allk = max(allk, max(abs(float(x[k]) - float(y[k])) for k in keys))
            clipd = max(clipd, abs(x["clipped_sample_steps"] - y["clipped_sample_steps"]))
        for a in BDG_HEAD:
            for x, y in zip(abl1[(p, a)], head[(p, a)]):
                ri = max(ri, abs(x["in_band_fraction"] - y["in_band_fraction"]))
                rs = max(rs, abs(x["mol_stability"] - y["mol_stability"]))
    return dict(mi=mi, ms=ms, ri=ri, rs=rs, allk=allk, clipd=int(clipd))


def ung_copies(head):
    """The three unguided cells per seed (one per property) are the same
    generation run -- guidance is off, and chemistry does not depend on the
    property -- so their chemistry must agree if the backend is bit-exact."""
    keys = ("mol_stability", "atom_stability", "validity", "uniqueness_of_valid")
    worst = {k: 0.0 for k in keys}
    for i in range(len(M1_SEEDS)):
        cs = [head[(p, "unguided")][i] for p in PROPS]
        for k in keys:
            worst[k] = max(worst[k], max(abs(a[k] - b[k]) for a in cs for b in cs))
    return worst


def m1_cost_table(A, P, arms):
    A("| arm | cells | mean s / cell | min / 1000 molecules | generator passes | guide passes "
      "| clip % mu / alpha / gap |")
    A("|---|---|---|---|---|---|---|")
    for a in arms:
        rs = [P[(p, a)] for p in PROPS]
        secs = mean([r["secs_cell"] for r in rs])
        A("| `%s` | %d | %.0f | %.2f | %d | %d | %s |"
          % (a, sum(r["seeds"] for r in rs), secs, secs / N * 1000 / 60,
             rs[0]["gen_passes"], rs[0]["guide_passes"],
             " / ".join("%.2f" % (100 * r["clip_frac"]) for r in rs)))
    A("")


# ------------------------------------------------------------------ M2 load
def m2_arm(r):
    return "bdg_" + r["variant"] if r["arm"] == "bdg" else r["arm"]


def m2_read(root):
    """[(stage, path, cell)] -- through m2_v3_results.load when it imported."""
    out = []
    if M2R is not None:
        for r in M2R.load(root, N):
            out.append((r["_stage"], r["_path"], r))
        return out
    for st in ("m2", "m2abl", "m2wsweep"):
        for fn in sorted(glob.glob(os.path.join(root, st, "n%d" % N, "seed*", "*.json"))):
            with open(fn, encoding="utf-8") as f:
                out.append((st, fn, json.load(f)))
    return out


def m2_load(root):
    rows = {}
    for st, fn, r in m2_read(root):
        if r.get("stage") != st:
            die("%s is a %r cell under %s/" % (fn, r.get("stage"), st))
        bad = [(k, r.get(k)) for k, v in (("n", N), ("batch", BATCH), ("steps", 100),
                                           ("delta_ratio", 0.16), ("target_name", "q50"),
                                           ("ckpt", M2_CKPT), ("clip", 1.0))
               if r.get(k) != v]
        if r.get("seed") not in M2_SEEDS:
            bad.append(("seed", r.get("seed")))
        if bad:
            die("%s: %s" % (fn, bad))
        r["_st"], r["_file"] = st, os.path.relpath(fn, ROOT).replace(os.sep, "/")
        k = (st, r["prop"], m2_arm(r), float(r["w"]), float(r["t_min_guide"]), r["seed"])
        if k in rows:
            die("duplicated M2 cell %s: %s and %s" % (k, rows[k]["_file"], r["_file"]))
        rows[k] = r
    exp = set()
    for s in M2_SEEDS:
        for p in M2_PROPS:
            for a in M2_HEAD_ARMS:
                for w in (1.0, 4.0):
                    exp.add(("m2", p, a, w, 0.5, s))
        for a in ABL_ARMS:
            for w in (1.0, 4.0):
                exp.add(("m2abl", "gc", a, w, 0.5, s))
        for a in M2_WIN0_ARMS:
            exp.add(("m2abl", "gc", a, 1.0, 0.0, s))
    for a, w in M2_WSWEEP:
        exp.add(("m2wsweep", "gc", a, w, 0.5, M2_SEEDS[0]))
    missing, extra = sorted(exp - set(rows)), sorted(set(rows) - exp)
    if missing or extra:
        die("M2 tree: missing %s; unexpected %s" % (missing[:6], extra[:6]))
    for p in M2_PROPS:
        ds = {round(r["delta"], 12) for k, r in rows.items() if k[1] == p}
        if len(ds) != 1:
            die("M2 %s cells scored at different deltas %s" % (p, sorted(ds)))
    return rows


def m2_guided_per_batch(r):
    """Guided steps per batch. m2_sweep.run_cell increments cost['gen_vjp'] once
    per guided step and sums it over the n/batch batches; the loop guides every
    step i with i/steps >= t_min. The two must agree."""
    nb = r["n"] // r["batch"]
    loop = sum(1 for i in range(r["steps"]) if i * (1.0 / r["steps"]) >= r["t_min_guide"])
    if r["arm"] == "unguided":
        if r["cost"]["gen_vjp"] or r["clipped_sample_steps"]:
            die("%s: unguided cell with guided steps or clips" % r["_file"])
        return 0
    g = r["cost"]["gen_vjp"]
    if g % nb or g // nb != loop:
        die("%s: cost gen_vjp %d is not %d batches x %d guided steps" % (r["_file"], g, nb, loop))
    return loop


def m2_pool(rs):
    """N-weighted pooling over seeds. The property sd is the sd of the POOLED
    sample (per-seed sds plus the between-seed shift of the mean), bias/d is the
    pooled mean's offset over delta -- m2_v3_results.pooled's definitions, used
    through it when it imported. The clip fraction is recomputed here from
    m2_sweep.run_cell's loop and must agree with it."""
    Nn = sum(r["n"] for r in rs)
    den = sum(m2_guided_per_batch(r) * r["n"] for r in rs)
    cl = sum(r["clipped_sample_steps"] for r in rs)
    clip = (cl / den) if den else 0.0
    ibs = [r["in_band_fraction"] for r in rs]
    if M2R is not None:
        o = M2R.pooled(rs, what="a v3_final_summary row")
        if abs(o["clip_frac"] - clip) > 1e-12 or o["N"] != Nn:
            die("m2_v3_results.pooled's clip fraction %r disagrees with run_cell's loop %r"
                % (o["clip_frac"], clip))
        return dict(N=Nn, ib=o["in_band_fraction"], se=o["se"], seeds=o["seeds"],
                    bias=o["bias_delta"], sd=o["sd"], mean=o["gc_mean"], kmer=o["kmer_js"],
                    conf=o["decode_conf"], div=o["diversity"], clip=clip, clipped=cl,
                    mins=o["minutes"], mins_tot=o["minutes_total"], w_eff=o["w_eff"],
                    delta=o["delta"], y=o["y"], ibs=ibs)
    wm = lambda k: sum(r[k] * r["n"] for r in rs) / Nn
    ib, gm = wm("in_band_fraction"), wm("gc_mean")
    sdp = math.sqrt(sum((r["gc_sd"] ** 2 + (r["gc_mean"] - gm) ** 2) * r["n"] for r in rs) / Nn)
    we = [r["diag"].get("bdg_w_eff") for r in rs if r.get("diag")]
    we = [x for x in we if x is not None]
    return dict(N=Nn, ib=ib, se=se(ib, Nn), seeds=len(rs),
                bias=(gm - rs[0]["y"]) / rs[0]["delta"], sd=sdp, mean=gm,
                kmer=wm("kmer_js"), conf=wm("decode_conf"), div=wm("diversity"),
                clip=clip, clipped=cl,
                mins=mean([r["minutes"] for r in rs]), mins_tot=sum(r["minutes"] for r in rs),
                w_eff=mean(we) if we else None, delta=rs[0]["delta"], y=rs[0]["y"], ibs=ibs)


def m2_cells(rows, st, p, a, w, t=0.5):
    out = [rows[(st, p, a, float(w), t, s)] for s in M2_SEEDS]
    return out


# ------------------------------------------------------------------ doc checks
def doc_sections(path):
    """{(backend, key): [lines]} for a v3_table.py page; key is a property, 'cost'
    or 'settings'."""
    out, be, key = {}, None, None
    with open(path, encoding="utf-8") as f:
        for line in f.read().split(NL):
            m = re.match(r"^## Base model: .*\(`--backend (\w+)`\)", line)
            if m:
                be, key = m.group(1), "settings"
                continue
            m = re.match(r"^### (\w+) \(delta", line)
            if m:
                key = m.group(1)
                continue
            if line.startswith("### Cost"):
                key = "cost"
                continue
            if line.startswith("### "):
                key = "other"
                continue
            out.setdefault((be, key), []).append(line)
    return out


def v3_expected(be, cells, P, stage):
    """The rows v3_table.emit prints for this backend, as {(be, key): [lines]}."""
    meas = {}
    for a in {x for (_p, x) in cells}:
        vals = [P[(p, a)]["bdg_w_eff"] for p in PROPS
                if (p, a) in P and P[(p, a)].get("bdg_w_eff") is not None]
        if vals:
            meas[a] = sum(vals) / len(vals)
    arms = V.order_arms({a for (_p, a) in cells}, measured=meas)
    exp = {}
    for p in PROPS:
        av = [a for a in arms if (p, a) in P]
        d = P[(p, av[0])]["delta"]
        rows = []
        for a in av:
            r = P[(p, a)]
            rows.append("| `%s` | %.4f | %.4f | %.4f | %.3f | %+.3f | %.3f | %.4f | %.4f | "
                        "%.4f | %.4f (%.4f) | %.4f | %.4f | %.3f |"
                        % (a, r["in_band_fraction"], r["se_ib"], r["in_band_fraction_dec"],
                           r["prop_mae_eval"] / d, r["bias"] / d, r["sd"] / d,
                           r["mol_stability"], r["atom_stability"], r["validity"],
                           r["uniqueness_of_valid"], r["uniq_min_cell"],
                           r["unique_valid_per_sample"], r["diversity_mean_pairwise"],
                           r["guide_eval_gap_mean"] / d))
        u = P.get((p, "unguided"))
        if u is not None:
            for a in av:
                if a == "unguided":
                    continue
                r = P[(p, a)]
                rows.append("| `%s` | %+.4f | %+.2f | %+.4f | %+.2f | %+.4f | %+.4f |"
                            % (a, r["in_band_fraction"] - u["in_band_fraction"],
                               V.z2(r, u, "in_band_fraction", "se_ib"),
                               r["in_band_fraction_dec"] - u["in_band_fraction_dec"],
                               V.z2(r, u, "in_band_fraction_dec", "se_ib_dec"),
                               r["mol_stability"] - u["mol_stability"],
                               r["validity"] - u["validity"]))
        exp[(be, p)] = rows
    cost = []
    plug_tot = sum(P[(p, "plug")]["clipped"] for p in PROPS if (p, "plug") in P) or None
    for a in arms:
        r = next((P[(p, a)] for p in PROPS if (p, a) in P), None)
        clips, tot = [], 0
        for p in PROPS:
            rp = P.get((p, a))
            clips.append("%d" % rp["clipped"] if rp else "--")
            tot += rp["clipped"] if rp else 0
        ratio = ("%.2fx" % (tot / plug_tot)) if plug_tot and a != "plug" else "--"
        cost.append("| `%s` | %d | %d | %s | %s |"
                    % (a, r["gen_passes"], r["guide_passes"], " | ".join(clips), ratio))
    exp[(be, "cost")] = cost
    secs = sum(float(r.get("seconds") or 0.0) for rs in cells.values() for r in rs)
    devs = sorted({(r.get("prov") or {}).get("device") for rs in cells.values() for r in rs} - {None})
    exp[(be, "settings")] = ["| measured cost | %.1f GPU-h over %d cells, on %s |"
                             % (secs / 3600.0, sum(len(rs) for rs in cells.values()),
                                ", ".join(devs) or "an unrecorded device")]
    return exp


def check_doc(path, expected, report, name):
    if not os.path.isfile(path):
        die("%s is missing; this page must agree with it" % path)
    got = doc_sections(path)
    n_ok, bad = 0, []
    for k, lines in expected.items():
        have = set(got.get(k, []))
        for ln in lines:
            if ln in have:
                n_ok += 1
            else:
                bad.append("%s %s: %s" % (name, k, ln))
    if bad:
        die("%d row(s) of %s disagree with the cells:\n  %s"
            % (len(bad), name, "\n  ".join(bad[:10])))
    report.append((name, n_ok))


def check_lines(path, lines, report, name):
    with open(path, encoding="utf-8") as f:
        have = set(f.read().split(NL))
    bad = [ln for ln in lines if ln not in have]
    if bad:
        die("%d row(s) of %s disagree with the cells:\n  %s"
            % (len(bad), name, "\n  ".join(bad[:10])))
    report.append((name, len(lines)))


# ------------------------------------------------------------------ sections
def per_prop(head, get):
    """One template for a per-property name ('..._mu.pt' -> '..._<p>.pt'), or
    the three names if they do not follow one template."""
    names = {p: str(get(head[(p, "plug")][0])) for p in PROPS}
    ts = {re.sub(r"_%s\b" % p, "_<p>", names[p]) for p in PROPS}
    return ts.pop() if len(ts) == 1 else "; ".join("%s: %s" % (p, names[p]) for p in PROPS)


def m1_header(A, be, head, abl=None):
    r = head[(PROPS[0], "plug")][0]
    prov = r.get("prov") or {}
    gen = prov.get("gen") or r.get("backend") or "?"
    has_bdg = any(a.startswith("bdg_") for a in BACKEND_ARMS[be])
    A("**What ran.** %d headline cells (%d arms x 3 properties x 3 seeds, w = 1)%s, n = %d per "
      "cell, batch %d (%d batches per cell%s), %d-step %s on the `%s` grid, guidance for "
      "t >= %g (%d guided steps per batch), target q50, seeds %s, device %s."
      % (sum(len(v) for v in head.values()), len(BACKEND_ARMS[be]),
         (" + %d ablation cells (17 BDG arms x 3 x 3 x w in {1, 4})"
          % sum(len(v) for c in abl for v in c.values())) if abl else "",
         N, BATCH, N // BATCH, ", so 4 BDG controllers" if has_bdg else "",
         r["steps"], r["solver"], r.get("grid"), float(r["t_start"]),
         pool(head[(PROPS[0], "plug")])["guided_per_batch"], ", ".join(M1_SEEDS),
         prov.get("device")))
    A("")
    A("| | |")
    A("|---|---|")
    # prov.diffusion_steps is the checkpoint's diffusion length T (its noise
    # schedule's step count), NOT a training length; the edm cells record none.
    # This branch was written for `edm` alone and said "this borrowed
    # checkpoint" unconditionally. `vp` also records a noise_schedule and is
    # OURS, so the old wording printed our own model as borrowed -- inside the
    # page DATA_INDEX tells readers to start from. It also has no
    # diffusion_steps, so the T clause printed "T = None".
    extra = ""
    if prov.get("noise_schedule"):
        extra += ", noise schedule `%s`" % prov.get("noise_schedule")
        if prov.get("diffusion_steps") is not None:
            extra += (", T = %s diffusion steps (`prov.diffusion_steps`; the cells "
                      "record no training length for it)" % prov.get("diffusion_steps"))
    if prov.get("epoch") is not None:
        extra += ", checkpoint epoch %s (`prov.epoch`)" % prov.get("epoch")
    A("| generator | %s, md5 `%s`%s |" % (gen, str(prov.get("gen_md5"))[:8], extra))
    A("| property pair | `%s`: guide f_A = %s; oracle f_B = %s |"
      % (r["pair"], per_prop(head, lambda c: c.get("guide")),
         per_prop(head, lambda c: c.get("oracle"))))
    A("| delta (2 x MAE(f_B)) | %s |" % ", ".join("%s %.5f" % (p, head[(p, "unguided")][0]["delta"])
                                            for p in PROPS))
    A("| second oracle (o2) | %s, its own delta (%s) |"
      % (per_prop(head, lambda c: c["oracle2"].get("oracle2")),
         ", ".join("%s %.5f" % (p, head[(p, "unguided")][0]["oracle2"]["delta"]) for p in PROPS)))
    A("")


def section_edm(A, head, P, store):
    be = "edm"
    A("## 1. %s" % BE_TITLE[be])
    A("")
    A("> **This is not our diffusion model.** `edm` is TFG's released EDMsecond checkpoint, "
      "borrowed and frozen. **Our own** QM9 VP diffusion model is the separate `vp` backend "
      "(section below, 18 cells as of 28 Sep); it, not `edm`, is what fills the paper's "
      "\"VP, trained here\" row (`tab:fmvd`). v3 declares `edm` unguided + plug only "
      "(FULL_RUN_V3_PROTOCOL.md section 1.2), so it has no BDG arm, no ablation, and no "
      "BDG-vs-plug contrast.")
    A("")
    m1_header(A, be, head)
    A("### Full metric block")
    A("")
    m1_metric_table(A, P, BACKEND_ARMS[be])
    A("### Head-to-head: plug against unguided (z >= %.2f)" % ZBAR)
    A("")
    res = m1_vs_unguided(A, P, BACKEND_ARMS[be])
    store["vs_ung"][(be, 1)] = res
    A("### Chemistry cost")
    A("")
    for p in PROPS:
        u, r = P[(p, "unguided")], P[(p, "plug")]
        A("- %s: molecule stability %.4f -> %.4f (%s pp, %+.1f %%), validity %.4f -> %.4f, "
          "DV %.4f -> %.4f."
          % (p, u["mol_stability"], r["mol_stability"], pp(r["mol_stability"] - u["mol_stability"]),
             100 * (r["mol_stability"] / u["mol_stability"] - 1), u["validity"], r["validity"],
             u["unique_valid_per_sample"], r["unique_valid_per_sample"]))
    A("")
    A("### Cost in time")
    A("")
    m1_cost_table(A, P, BACKEND_ARMS[be])
    tot = sum(r["secs"] for r in P.values())
    store["cost"].append(("QM9 diffusion (`edm`, borrowed)", "headline", len(P) * 3,
                          head[(PROPS[0], "plug")][0]["prov"]["device"], tot))
    A("Total %.1f GPU-h over %d cells (the cells' own `seconds`). Clip %% is clipped sample-steps "
      "over (guided steps per batch x n); `edm` guides %d steps per batch on its uniform grid, "
      "not 50." % (tot / 3600, len(P) * 3, P[(PROPS[0], "plug")]["guided_per_batch"]))
    A("")
    uc = ung_copies(head)
    A("### Outcome")
    A("")
    cl = [p for p in PROPS if res[(p, "plug")]["cont"][1] >= ZBAR]
    cd = [p for p in PROPS if res[(p, "plug")]["dec"][1] >= ZBAR]
    co = [p for p in PROPS if res[(p, "plug")]["o2"][1] >= ZBAR]
    A("- `plug` raises continuous in-band over unguided at z >= %.2f on **%s** (%s); decoded on "
      "%s; second oracle on %s."
      % (ZBAR, ", ".join(cl) or "no property",
         ", ".join("%s %s pp, z %+.2f" % (p, pp(res[(p, "plug")]["cont"][0]), res[(p, "plug")]["cont"][1])
                   for p in PROPS),
         ", ".join(cd) or "none", ", ".join(co) or "none"))
    lost = [p for p in PROPS if res[(p, "plug")]["mol"][0] < 0]
    worst = min(PROPS, key=lambda p: res[(p, "plug")]["mol"][0])
    A("- Molecule stability against unguided: %s pp; lower on %s, the largest drop on %s."
      % (", ".join("%s %s" % (p, pp(res[(p, "plug")]["mol"][0])) for p in PROPS),
         "every property" if len(lost) == len(PROPS) else (", ".join(lost) or "no property"),
         worst))
    A("- This is a borrowed base with our pair: it says plug-in guidance moves a diffusion "
      "model's in-band at w = 1 at a chemistry cost. It says **nothing about our VP model**, and "
      "its in-band is not comparable with `fm` or `equifm` (different generator; `equifm` also a "
      "different pair and band).")
    A("")
    A("### Caveats")
    A("")
    A("- Borrowed checkpoint; never label it \"ours\". Our own VP diffusion is `vp`, a "
      "separate backend with its own 18 cells.")
    A("- w = 1 is not plug's best strength and a w = 1 number is not a ranking (section 6.2).")
    others = {b: max(v.values()) for b, v in store["ungc"].items() if b != "edm"}
    if max(uc.values()) > 0:
        A("- **Not bit-reproducible either.** The three unguided cells per seed (one per "
          "property) are the same generation run, since guidance is off and chemistry does not "
          "depend on the property. On %s they agree %s; on `edm` they differ by up to %.1e in "
          "molecule stability and %.1e in validity. The README's \"fm and edm are bit-exact\" "
          "is contradicted for `edm` by the committed cells."
          % (" and ".join("`%s`" % b for b in others),
             "exactly" if max(others.values()) == 0 else
             "to %.1e" % max(others.values()), uc["mol_stability"], uc["validity"]))
    else:
        A("- The three unguided cells per seed (one per property, the same generation run) "
          "agree exactly on chemistry.")
    A("- Pre-registered bar: 18 contrasts for 6 guided arms; `edm` has 1 guided arm, so z = "
      "%.2f is conservative here." % ZBAR)
    A("- Useful yield (decoded in-band AND stable, per molecule) is **pending**: it needs the "
      "`*.permol.pt` sidecars on blade.")
    A("")


def section_vp(A, head, P, Pfm, store):
    """Our OWN QM9 diffusion. Numbered 1b so `edm` stays 1 and nothing renumbers.

    The point of this section is the MATCHED contrast against `fm`: same
    EGNNVelocity backbone, same parameter count, epochs, batch, EMA, split and
    seed, so fm-vs-vp isolates the generator family and nothing else. That is
    the cell `tab:fmvd` never had, and it is reported here from the unguided
    arm, which is property-independent.
    """
    be = "vp"
    A("## 1b. %s" % BE_TITLE[be])
    A("")
    A("> **This IS our diffusion model** -- not to be confused with `edm` above, which is "
      "TFG's borrowed EDMsecond. Both are diffusion; only `vp` is ours. It is matched to "
      "`fm` on backbone, parameter count, epochs, batch, EMA, split and training seed, so "
      "the fm-vs-vp contrast isolates the generator **family**. v3 declares `vp` unguided + "
      "plug only (FULL_RUN_V3_PROTOCOL.md section 1.3), so it has no BDG arm and no ablation.")
    A("")
    m1_header(A, be, head)
    A("### Full metric block")
    A("")
    m1_metric_table(A, P, BACKEND_ARMS[be])
    A("### The matched comparison: `fm` against `vp`, unguided")
    A("")
    A("Unguided sampling is property-independent at a fixed seed, so these are one "
      "generation run scored three times; the three properties agree by construction.")
    A("")
    A("| metric | `fm` (flow) | `vp` (diffusion) | vp - fm |")
    A("|---|---|---|---|")
    for k, lab in (("mol_stability", "molecule stability"), ("validity", "validity"),
                   ("unique_valid_per_sample", "unique valid per sample")):
        a = sum(Pfm[(p, "unguided")][k] for p in PROPS) / len(PROPS)
        b = sum(P[(p, "unguided")][k] for p in PROPS) / len(PROPS)
        A("| %s | %.4f | %.4f | **%+.4f** |" % (lab, a, b, b - a))
    A("")
    A("**Flow matching wins on every one.** This is the evidence the base-model choice "
      "previously lacked.")
    A("")
    A("### Head-to-head: plug against unguided (z >= %.2f)" % ZBAR)
    A("")
    res = m1_vs_unguided(A, P, BACKEND_ARMS[be])
    store["vs_ung"][(be, 1)] = res
    A("### Chemistry cost")
    A("")
    for p in PROPS:
        u, r = P[(p, "unguided")], P[(p, "plug")]
        A("- %s: molecule stability %.4f -> %.4f (%s pp, %+.1f %%), validity %.4f -> %.4f, "
          "DV %.4f -> %.4f."
          % (p, u["mol_stability"], r["mol_stability"], pp(r["mol_stability"] - u["mol_stability"]),
             100 * (r["mol_stability"] / u["mol_stability"] - 1), u["validity"], r["validity"],
             u["unique_valid_per_sample"], r["unique_valid_per_sample"]))
    A("")
    A("### Cost in time")
    A("")
    m1_cost_table(A, P, BACKEND_ARMS[be])
    tot = sum(r["secs"] for r in P.values())
    dev = head[(PROPS[0], "plug")][0]["prov"]["device"]
    store["cost"].append(("QM9 diffusion (`vp`, OURS)", "headline", len(P) * 3, dev, tot))
    A("Total %.1f GPU-h over %d cells (the cells' own `seconds`), on %s."
      % (tot / 3600, len(P) * 3, dev))
    A("")
    A("### Outcome")
    A("")
    cl = [p for p in PROPS if res[(p, "plug")]["cont"][1] >= ZBAR]
    A("- `plug` raises continuous in-band over unguided at z >= %.2f on **%s** (%s)."
      % (ZBAR, ", ".join(cl) or "no property",
         ", ".join("%s %s pp, z %+.2f" % (p, pp(res[(p, "plug")]["cont"][0]),
                                          res[(p, "plug")]["cont"][1]) for p in PROPS)))
    A("- Molecule stability against unguided: %s pp."
      % ", ".join("%s %s" % (p, pp(res[(p, "plug")]["mol"][0])) for p in PROPS))
    A("- Against `fm` on the same protocol, this is a **base-model** result, not a guidance "
      "one: it says the flow family generates better chemistry than the diffusion family at "
      "matched budget. Its in-band is comparable with `fm` and `edm` (same pair and band) "
      "but NOT with `equifm` (different pair, different band width).")
    A("")
    A("### Caveats")
    A("")
    A("- **`s/sample` here is a B200 MIG number.** The `fm`, `equifm` and `edm` rows of "
      "`tab:fmvd` were all timed on an RTX A6000 and the table's caption says so. Footnote "
      "the difference or re-time this cell on the same card; do not paste it in silently.")
    A("- **The clip binds far harder than on `fm`**: `plug` clips on about 6-7.5 %% of guided "
      "sample-steps against M1's under 1 %%. The score carries a 1/sigma factor near the "
      "noisy end, so a diffusion base asks more of the trust region than a flow base does. "
      "`v3_sanity.py` reports these as warnings, not failures.")
    A("- w = 1 is not plug's best strength and a w = 1 number is not a ranking (section 6.2).")
    A("- Pre-registered bar: 18 contrasts for 6 guided arms; `vp` has 1 guided arm, so "
      "z = %.2f is conservative here." % ZBAR)
    A("- Unlike the blade backends, **`vp`'s `*.permol.pt` sidecars are in the repository "
      "tree**, so useful yield can be computed for it without a further pull.")
    A("")


def section_flow(A, num, be, head, abl, Ph, Pa, store):
    A("## %d. %s" % (num, BE_TITLE[be]))
    A("")
    if be == "equifm":
        A("> **EquiFM is not bit-reproducible.** Same-seed re-runs of the same configuration "
          "(the headline's two BDG arms, re-run in the ablation) move in-band by up to "
          "**%.1e**, the same order as its BDG - plug differences. Its binomial se therefore "
          "understates its noise, and every `equifm` verdict below carries that."
          % identity(head, abl[1])["ri"])
        A("")
    m1_header(A, be, head, abl=[abl[1], abl[4]])
    A("### Full metric block, headline (w = 1)")
    A("")
    m1_metric_table(A, Ph, HEAD_ARMS)
    A("### Head-to-head: every guided arm against unguided (w = 1)")
    A("")
    res = m1_vs_unguided(A, Ph, HEAD_ARMS)
    store["vs_ung"][(be, 1)] = res
    # identity
    idn = identity(head, abl[1])
    exact = idn["allk"] == 0.0 and idn["clipd"] == 0
    A("### plug at w = 4, and the eta = 0 identity it rests on")
    A("")
    A("The headline runs plug at w = 1 only. The ablation's `bdg_e0t1` is BDG at eta = 0, which "
      "is plug's guidance field (test_bdg.py A9/B-a, bit-identical), and it runs at w = 1 and 4. "
      "Checked first at w = 1 against the headline's plug, same n and seeds, max |d| over "
      "properties and seeds: in-band %.2e, molecule stability %.2e, %.2e over in-band, decoded "
      "in-band, stability, validity and DV, and %d clipped sample-steps. %s"
      % (idn["mi"], idn["ms"], idn["allk"], idn["clipd"],
         "**`fm` is bit-exact**, so `bdg_e0t1` at w = 4 *is* plug at w = 4." if exact else
         "**Not exact**: `equifm`'s re-run noise (above) is the same size, so `bdg_e0t1` at w = 4 "
         "is plug's field, not a bit-identical plug cell."))
    A("")
    A("At w = 4 the identity rests on the field-level gate only (there is no headline plug at "
      "w = 4). The w = 4 rows below read `bdg_e0t1` against the headline's `unguided`, which is "
      "a comparison **across stage trees** (same n, seeds, pair and delta; unguided does not "
      "depend on w).")
    A("")
    A("| arm | w | in_band +- se | dec | o2 | MAE/d | bias/d | sd/d | mol_stab | valid | uniq "
      "| DV | clip % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PROPS:
        for lab, w, r in (("unguided", 0, Ph[(p, "unguided")]), ("plug", 1, Ph[(p, "plug")]),
                          ("plug (= bdg_e0t1)", 4, Pa[4][(p, "bdg_e0t1")])):
            d = r["delta"]
            A("| %s: %s | %d | %.4f +- %.4f | %.4f | %.4f | %.3f | %+.3f | %.3f | %.4f | %.4f "
              "| %.4f | %.4f | %.2f |"
              % (p, lab, w, r["in_band_fraction"], r["se_ib"], r["in_band_fraction_dec"], r["o2"],
                 r["prop_mae_eval"] / d, r["bias"] / d, r["sd"] / d, r["mol_stability"],
                 r["validity"], r["uniqueness_of_valid"], r["unique_valid_per_sample"],
                 100 * r["clip_frac"]))
    A("")
    A("Against unguided at w = 4 (arms present in the ablation tree only):")
    A("")
    U = {(p, "unguided"): Ph[(p, "unguided")] for p in PROPS}
    P4 = dict(Pa[4])
    res4 = m1_vs_unguided(A, P4, ("unguided", "bdg_e0t1") + BDG_HEAD, U=U)
    store["vs_ung"][(be, 4)] = res4
    # every ablation cell against the headline's unguided (the scoreboard shows
    # only the three arms above; this is the rest of the tree)
    above, below = {}, []
    for w in (1, 4):
        for p in PROPS:
            for a in ABL_ARMS:
                c = contrast(Pa[w][(p, a)], U[(p, "unguided")])
                if c["cont"][1] >= ZBAR:
                    above[w] = above.get(w, 0) + 1
                if min(c[k][1] for k in ("cont", "dec", "o2")) <= -ZBAR:
                    below.append((w, p, a, c))
    store.setdefault("abl_vs_ung", {})[be] = (above, below)
    txt = ("**The whole ablation tree against unguided** (17 arms x 3 properties at each w, "
           "single contrasts against the headline's `unguided`, not selection-adjusted): %d cells "
           "clear it on continuous in-band at w = 1 and %d at w = 4; **%d fall below it** at "
           "z <= -%.2f on at least one of continuous / decoded / second oracle"
           % (above.get(1, 0), above.get(4, 0), len(below), ZBAR))
    if below:
        txt += (" (d continuous, z continuous / decoded / second oracle): "
                + "; ".join("%s w = %d `%s` %s pp (z %+.2f / %+.2f / %+.2f)"
                            % (p, w, a, pp(c["cont"][0]), c["cont"][1], c["dec"][1], c["o2"][1])
                            for (w, p, a, c) in below) + ".")
        taus = sorted({a.split("t", 1)[1] for (_w, _p, a, _c) in below})
        bw = [BR.w_eff(abl[w][(p, a)]) for (w, p, a, _c) in below]
        negs = [[(r.get("diag") or {}).get("bdg_w_eff_neg") for r in abl[w][(p, a)]]
                for (w, p, a, _c) in below]
        if any(x is None for v in negs for x in v) or any(x is None for x in bw):
            die("%s: an ablation cell below unguided has no diag.bdg_w_eff / bdg_w_eff_neg" % be)
        neg = [mean(v) for v in negs]
        txt += (" All are at tau_mult %s. Their measured w_eff is %.2f to %.2f and the deviation "
                "term reverses (w_eff < 0) on %.0f-%.0f %% of guided steps (seed means of "
                "`diag.bdg_w_eff`, `diag.bdg_w_eff_neg`): there BDG spreads the batch and loses "
                "coverage against not guiding at all."
                % (" / ".join(taus), min(bw), max(bw), 100 * min(neg), 100 * max(neg)))
    else:
        txt += "."
    A(txt)
    A("")
    A("### BDG against plug, per cell, both headline BDG arms")
    A("")
    A("plug is BDG's eta = 0 limit, so it is the base arm for the dispersion term. w = 1 rows "
      "are the headline; w = 4 rows are the ablation tree, plug = `bdg_e0t1`. Verdicts are "
      "continuous / decoded / second oracle at z >= %.2f." % ZBAR)
    A("")
    rows = m1_bdg_vs_plug_rows(head, abl[4], Ph, Pa[4])
    bvp = m1_bdg_vs_plug(A, rows)
    store["bvp"][be] = bvp
    A("### Ablation, compact")
    A("")
    A("In-band against `bdg_e0t1` at fixed w (eta = 0: plug's field). The full 4 x 4 grids and "
      "their measured w_eff are in V3_BLADE_READOUT.md section 3; the full metric blocks in "
      "V3_RESULTS_ABL_w1.md / _w4.md. Spearman is over the 16 eta > 0 rungs, against the "
      "**measured** w_eff (seed mean of the cells' `diag.bdg_w_eff`).")
    A("")
    summ = abl_summary({(be, 1): abl[1], (be, 4): abl[4]}, be)
    store["abl"][be] = summ
    A("| prop | w | eta = 0 in_band | d e8t0.5 (pp) | z | v | its d mol_stab (pp) | its clip % "
      "(eta 0) | d e8t1.5 (pp) | z | v | its d mol_stab (pp) | tau 0.5 row rises with eta "
      "| Spearman(w_eff, d) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summ:
        A("| %s | %d | %.4f | %s | %+.2f | %s | %s | %.2f (%.2f) | %s | %+.2f | %s | %s | %s | %+.2f |"
          % (s["prop"], s["w"], s["b0"], pp(s["da"]), s["za"], verdict(s["za"]), pp(s["dma"]),
             100 * s["clipa"], 100 * s["clip0"], pp(s["db"]), s["zb"], verdict(s["zb"]),
             pp(s["dmb"]), "yes" if s["mono"] else "**no**", s["rho"]))
    A("")
    A("Single contrasts against eta = 0, not a max over the 16 rungs. A claim that *some* rung "
      "beat plug in *some* grid is a selection over 16 x 3 x 2 x 2 = 192 contrasts. "
      "ABLATION_V3_PROTOCOL.md section 4 counts \"17 settings x 3 properties x 2 bases\" (no w "
      "axis) and requires a stated max-T or Bonferroni bar beside any such claim; Bonferroni over "
      "192 two-sided contrasts is z = %.2f (**derived here, not in the protocol**)."
      % V.NORM_Q(1.0 - ALPHA / (2 * 192)))
    A("")
    # chemistry
    A("### Chemistry cost")
    A("")
    A("Mean change in molecule stability against unguided over the three properties, w = 1: %s. "
      "At w = 4: %s."
      % (", ".join("`%s` %s pp" % (a, pp(mean([res[(p, a)]["mol"][0] for p in PROPS])))
                   for a in HEAD_ARMS[1:]),
         ", ".join("`%s` %s pp" % (a, pp(mean([res4[(p, a)]["mol"][0] for p in PROPS])))
                   for a in ("bdg_e0t1",) + BDG_HEAD)))
    A("")
    held = {p: [a for a in HEAD_ARMS[1:] if res[(p, a)]["mol"][0] >= 0] for p in PROPS}
    A("Arms whose molecule stability is not below unguided's at w = 1: %s."
      % "; ".join("%s: %s" % (p, arms_txt(held[p])) for p in PROPS))
    A("")
    # cost
    A("### Cost in time")
    A("")
    m1_cost_table(A, Ph, HEAD_ARMS)
    th = sum(r["secs"] for r in Ph.values())
    ta = {w: sum(r["secs"] for r in Pa[w].values()) for w in (1, 4)}
    dev = head[(PROPS[0], "plug")][0]["prov"]["device"]
    store["cost"].append(("`%s`" % be, "headline", len(Ph) * 3, dev, th))
    store["cost"].append(("`%s`" % be, "ablation w = 1", len(Pa[1]) * 3, dev, ta[1]))
    store["cost"].append(("`%s`" % be, "ablation w = 4", len(Pa[4]) * 3, dev, ta[4]))
    clips = [(pool([r])["clip_frac"], r["_file"]) for w in (1, 4) for rs in abl[w].values() for r in rs]
    A("Headline %.1f GPU-h over %d cells; ablation %.1f (w = 1) + %.1f (w = 4) = %.1f GPU-h over "
      "%d cells. Ablation clip fraction per cell: max %.1f %%, %d of %d cells above 5 %%. "
      "(Clip %% = clipped sample-steps / (guided steps per batch x n); `v3_sanity.py` divides by "
      "the summed guided steps and understates it %dx.)"
      % (th / 3600, len(Ph) * 3, ta[1] / 3600, ta[4] / 3600, (ta[1] + ta[4]) / 3600,
         (len(Pa[1]) + len(Pa[4])) * 3, 100 * max(c for c, _ in clips),
         sum(1 for c, _ in clips if c > 0.05), len(clips), N // BATCH))
    A("")
    # outcome
    A("### Outcome")
    A("")
    clr = lambda p, k: [a for a in HEAD_ARMS[1:] if res[(p, a)][k][1] >= ZBAR]
    big = {p: max(HEAD_ARMS[1:], key=lambda a: res[(p, a)]["cont"][0]) for p in PROPS}
    if all(big[p] == "tfg" for p in PROPS):
        # v3 has no chemistry floor (FULL_RUN_V3_PROTOCOL 2.1) and no M1 tfg strength
        # sweep, so the only strength evidence is 2.1's v2 figure, 2.2's "unknown",
        # and this run's clip fractions.
        why = ("that arm is `tfg` on every property. FULL_RUN_V3_PROTOCOL.md section 2.1 records "
               "that under v2's chemistry floor `tfg` froze at w = 0.01-0.05, so w = 1 is 20-100x "
               "that (a v2 measurement: v3 has no floor and no M1 `tfg` strength sweep), and "
               "section 2.2 records its force at w = 1 as unknown, not small; here it clips "
               "%s %% of sample-steps against plug's %s %%. So this is a statement about "
               "strength, not a ranking"
               % ("/".join("%.2f" % (100 * Ph[(p, "tfg")]["clip_frac"]) for p in PROPS),
                  "/".join("%.2f" % (100 * Ph[(p, "plug")]["clip_frac"]) for p in PROPS)))
    else:
        why = ("w = 1 is not that arm's best strength and equal w is not equal force "
               "(FULL_RUN_V3_PROTOCOL.md section 2.2), so this is a statement about strength, "
               "not a ranking")
    A("- **Against unguided, w = 1** (arms clearing z >= %.2f on continuous / decoded / second "
      "oracle): %s. Largest in-band gain per property: %s; %s."
      % (ZBAR, "; ".join("%s: %s / %s / %s" % (p, arms_txt(clr(p, "cont")), arms_txt(clr(p, "dec")),
                                              arms_txt(clr(p, "o2"))) for p in PROPS),
         "; ".join("%s `%s` %s pp at d mol_stab %s pp"
                   % (p, big[p], pp(res[(p, big[p])]["cont"][0]), pp(res[(p, big[p])]["mol"][0]))
                   for p in PROPS), why))
    parts = []
    for a in BDG_HEAD:
        v1 = {k: [verdict(bvp[(1, p, a)][k][1]) for p in PROPS] for k in ("cont", "dec", "o2")}
        v4 = {k: [verdict(bvp[(4, p, a)][k][1]) for p in PROPS] for k in ("cont", "dec", "o2")}
        parts.append("`%s`: w = 1 %s | %s | %s, w = 4 %s | %s | %s; d in_band at w = 1 %s pp "
                     "(MDD %s pp); mean d mol_stab vs plug %s pp"
                     % (a, tally(v1["cont"]), tally(v1["dec"]), tally(v1["o2"]), tally(v4["cont"]),
                        tally(v4["dec"]), tally(v4["o2"]),
                        ", ".join(pp(bvp[(1, p, a)]["cont"][0]) for p in PROPS),
                        ", ".join("%.2f" % (100 * bvp[(1, p, a)]["mdd"]) for p in PROPS),
                        pp(mean([bvp[(1, p, a)]["mol"][0] for p in PROPS]))))
    A("- **BDG against plug** (above / below / tie over 3 properties, continuous | decoded | "
      "second oracle): %s. A tie is a difference below the MDD beside it, not an established "
      "null; the property-averaged mol_stab means are descriptive, not pre-registered."
      % "; ".join(parts))
    ub = Ph
    sd_b = {p: ub[(p, "bdg_e4t0.5")]["sd"] / ub[(p, "bdg_e4t0.5")]["delta"] for p in PROPS}
    sd_u = {p: ub[(p, "unguided")]["sd"] / ub[(p, "unguided")]["delta"] for p in PROPS}
    b_b = {p: ub[(p, "bdg_e4t0.5")]["bias"] / ub[(p, "bdg_e4t0.5")]["delta"] for p in PROPS}
    b_u = {p: ub[(p, "unguided")]["bias"] / ub[(p, "unguided")]["delta"] for p in PROPS}
    narrower = [p for p in PROPS if sd_b[p] < sd_u[p]]
    worse = [p for p in PROPS if abs(b_b[p]) > abs(b_u[p])]
    A("- `bdg_e4t0.5` against unguided: residual sd/d %s on %s (%s vs %s); |bias|/d larger on %s "
      "(%s vs %s).%s"
      % ("smaller" if narrower else "not smaller", ", ".join(narrower) or "no property",
         ", ".join("%.3f" % sd_b[p] for p in PROPS), ", ".join("%.3f" % sd_u[p] for p in PROPS),
         ", ".join(worse) or "no property",
         ", ".join("%+.3f" % b_b[p] for p in PROPS), ", ".join("%+.3f" % b_u[p] for p in PROPS),
         " So \"BDG fixes bias and spread\" does not hold on this base." if worse else ""))
    A("- Ablation: in-band tracks the measured w_eff (Spearman %+.2f to %+.2f in all 6 grids); "
      "e8t1.5 falls below eta = 0 at z <= -%.2f in %d of 6; e8t0.5 clears it at z >= %.2f in "
      "%d of 6; the tau 0.5 row rises with eta in %d of 3 grids at w = 1 and %d of 3 at w = 4. "
      "This supports \"the deviation weight w_eff, not the global w, carries the effect\"; it "
      "does not show the feedback loop is necessary (no fixed-w_eff control ran)."
      % (min(s["rho"] for s in summ), max(s["rho"] for s in summ), ZBAR,
         sum(1 for s in summ if s["zb"] <= -ZBAR), ZBAR, sum(1 for s in summ if s["za"] >= ZBAR),
         sum(1 for s in summ if s["mono"] and s["w"] == 1),
         sum(1 for s in summ if s["mono"] and s["w"] == 4)))
    A("")
    A("### Caveats")
    A("")
    A("- in_band here is in `%s`'s own band (pair `%s`); it is not comparable with any other "
      "backend or with M2." % (be, head[(PROPS[0], "plug")][0]["pair"]))
    A("- w = 1 is not any arm's optimum and equal w is not equal force (FULL_RUN_V3_PROTOCOL.md "
      "section 2.2); nothing here says which method is best.")
    A("- From the headline alone `bdg - plug` cannot separate controlling spread from pushing "
      "harder (section 6.2): `bdg_e4t0.5` clips %s %% of sample-steps against plug's %s %%."
      % ("/".join("%.2f" % (100 * Ph[(p, "bdg_e4t0.5")]["clip_frac"]) for p in PROPS),
         "/".join("%.2f" % (100 * Ph[(p, "plug")]["clip_frac"]) for p in PROPS)))
    # clip-limited (FULL_RUN_V3_PROTOCOL 6.2, ABLATION_V3_PROTOCOL 1.1 / 4)
    cw = {w: [pool([r])["clip_frac"] for rs in abl[w].values() for r in rs] for w in (1, 4)}
    e8 = max(PROPS, key=lambda p: Pa[4][(p, "bdg_e8t0.5")]["clip_frac"])
    A("- **Clip-limited, not only controller-limited.** The clip is applied *after* w, so a rung "
      "may be clip-limited rather than controller-limited; FULL_RUN_V3_PROTOCOL.md section 6.2 "
      "names this the single most likely explanation for a BDG null, so it is the explanation "
      "offered for the BDG - plug ties above, not \"spread control does not help\" -- weighed "
      "against each row's clip %% BDG / plug (where BDG clips barely more than plug, the clip "
      "cannot be what limits it). `bdg_e4t0.5` clips "
      "%s %% (w = 1) and %s %% (w = 4) of sample-steps on mu/alpha/gap against plug's %s and %s "
      "%%; the e8t0.5 rung reaches %.2f %% at w = 4 (%s); %d of %d ablation cells at w = 1 and "
      "%d of %d at w = 4 clip more than 5 %%. ABLATION_V3_PROTOCOL.md section 1.1: compare the "
      "clip across the two strengths before attributing any w = 1 -> 4 difference to strength, "
      "and read the high-w_eff rungs as possibly clip-limited."
      % ("/".join("%.2f" % (100 * Ph[(p, "bdg_e4t0.5")]["clip_frac"]) for p in PROPS),
         "/".join("%.2f" % (100 * Pa[4][(p, "bdg_e4t0.5")]["clip_frac"]) for p in PROPS),
         "/".join("%.2f" % (100 * Ph[(p, "plug")]["clip_frac"]) for p in PROPS),
         "/".join("%.2f" % (100 * Pa[4][(p, "bdg_e0t1")]["clip_frac"]) for p in PROPS),
         100 * Pa[4][(e8, "bdg_e8t0.5")]["clip_frac"], e8,
         sum(1 for c in cw[1] if c > 0.05), len(cw[1]), sum(1 for c in cw[4] if c > 0.05),
         len(cw[4])))
    # the measured w_eff against the protocol's pre-run magnitude
    we = [BR.w_eff(abl[w][(p, a)]) for w in (1, 4) for p in PROPS for a in ABL_ARMS
          if a != "bdg_e0t1"]
    if any(x is None for x in we):
        die("%s: an ablation cell has no diag.bdg_w_eff" % be)
    hw = {a: [BR.w_eff(head[(p, a)]) for p in PROPS] for a in BDG_HEAD}
    store.setdefault("weff", {})[be] = (min(we), max(we))
    A("- **The measured w_eff is not the protocol's.** The pre-run figure of w_eff ~ 10^2-10^3 "
      "(FULL_RUN_V3_PROTOCOL.md sections 4.1 and 6.2, ABLATION_V3_PROTOCOL.md section 3, "
      "BDG_LADDER_MEASURED.md: 1019 at tau_mult 0.5 and 253 at 1.0) is not what v3 measured: "
      "this backend's cells record w_eff (seed means of `diag.bdg_w_eff`) from %.2f to %.2f "
      "over the 16 eta > 0 rungs at both w, and the headline arms measure %s (`bdg_e4t0.5`) and "
      "%s (`bdg_e4t1`) on mu/alpha/gap. Quote the measured values. The protocol's \"at w_eff of "
      "order 10^3 the clip truncates most of the dispersion term\" rests on that magnitude and "
      "is withdrawn for this run; the clip caveat above stands on the measured clip fractions, "
      "not on it."
      % (min(we), max(we), "/".join("%.2f" % x for x in hw["bdg_e4t0.5"]),
         "/".join("%.2f" % x for x in hw["bdg_e4t1"])))
    A("- V3_RESULTS.md's \"Selection\" line uses z = 3.51 (max of 6 arms at the 3-sigma family "
      "rate); this page uses the pre-registered per-contrast z = %.2f. Continuous contrasts "
      "between the two bars: %s."
      % (ZBAR, arms_txt(["%s/%s" % (p, a) for p in PROPS for a in HEAD_ARMS[1:]
                         if ZBAR <= res[(p, a)]["cont"][1] < 3.51]).replace("`", "") or "none"))
    if be == "equifm":
        bdiff = [abs(bvp[(1, p, a)]["cont"][0]) for p in PROPS for a in BDG_HEAD]
        A("- **Reproducibility**: same-seed re-runs move in-band by up to %.1e (%.2f pp) and "
          "molecule stability by up to %.1e, on top of seed noise; the eta = 0 control differs "
          "from plug by %.1e in-band. equifm's BDG - plug differences at w = 1 are %.2f to %.2f "
          "pp in size, so they are the same order as noise the binomial se does not include."
          % (idn["ri"], 100 * idn["ri"], idn["rs"], idn["mi"], 100 * min(bdiff), 100 * max(bdiff)))
    elif exact and idn["ri"] == 0 and idn["rs"] == 0:
        A("- `fm` is bit-exact in this run (eta = 0 = plug, and same-seed re-runs are identical); "
          "V3_REPRO_ENVELOPE.md recorded it as not bit-stable at n = 64, which has not been "
          "reconciled.")
    else:
        A("- `%s` is **not** bit-exact in this run: eta = 0 vs plug max |d in_band| %.1e, "
          "same-seed re-runs %.1e." % (be, idn["mi"], idn["ri"]))
    A("- Useful yield (decoded in-band AND stable, per molecule) is **pending** on blade; it is "
      "not the product of the marginals above.")
    A("")


def m2_window_best(rows):
    """The largest gc in-band inside v3's window (every distinct arm x w setting
    of the headline and the ablation), and what moving plug to t >= 0 is worth.
    bdg_e4t0.5 / bdg_e4t1 are in both trees (a same-seed re-run of one
    configuration), so each (arm, w) is counted once, from the headline tree."""
    cands = [("m2", a, w) for w in (1.0, 4.0) for a in M2_HEAD_ARMS[1:]]
    seen = {(a, w) for (_st, a, w) in cands}
    cands += [("m2abl", a, w) for w in (1.0, 4.0) for a in ABL_ARMS if (a, w) not in seen]
    if len({(a, w) for (_st, a, w) in cands}) != len(cands):
        die("M2 window candidates are not distinct")
    ung = m2_pool(m2_cells(rows, "m2", "gc", "unguided", 1.0))
    best = max(cands, key=lambda c: m2_pool(m2_cells(rows, c[0], "gc", c[1], c[2]))["ib"])
    bp = m2_pool(m2_cells(rows, best[0], "gc", best[1], best[2]))
    late = m2_pool(m2_cells(rows, "m2", "gc", "plug", 1.0))
    early = m2_pool([rows[("m2abl", "gc", "plug", 1.0, 0.0, s)] for s in M2_SEEDS])
    gain = early["ib"] - late["ib"]
    # cpg: no t = 0 and no eta-sweep cell, so its largest in-window gain is over
    # the headline arms only, and no window ratio is formed on it
    cpg_u = m2_pool(m2_cells(rows, "m2", "cpg", "unguided", 1.0))
    cpg_c = [(a, w) for w in (1.0, 4.0) for a in M2_HEAD_ARMS[1:]]
    cpg_b = max(cpg_c, key=lambda c: m2_pool(m2_cells(rows, "m2", "cpg", c[0], c[1]))["ib"])
    cpg_g = m2_pool(m2_cells(rows, "m2", "cpg", cpg_b[0], cpg_b[1]))["ib"] - cpg_u["ib"]
    return dict(cands=cands, best=best, bp=bp, ung=ung, win_gain=gain,
                ratio=gain / (bp["ib"] - ung["ib"]), n_config=len(
                    [c for c in cands if c[1] != "bdg_e0t1"]), cpg_best=cpg_b, cpg_gain=cpg_g)


def ratio_txt(x):
    """The window/method ratio as the other docs quote it (M2_V3_RESULTS.md,
    SCOPE_FM_GUIDANCE_STATUS.md, M2_WINDOW_DOMINATES.md: "about 5x"), with the
    exact quotient beside it."""
    return "about %.0fx (%.1f)" % (x, x)


def m2_ung_cells(rows, p):
    """unguided is w-independent (no field is applied) and its w = 1 and w = 4
    cells are the same samples, so it is pooled ONCE from its w = 1 cells --
    M2_V3_RESULTS.md's convention."""
    return m2_cells(rows, "m2", p, "unguided", 1.0)


def m2_arm_table(A, rows, p, w):
    u = m2_pool(m2_ung_cells(rows, p))
    A("| arm | in_band +- se | bias/d | %s sd | sd / unguided | k-mer JS | decode conf | diversity "
      "| clip %% | min / cell |" % p)
    A("|---|---|---|---|---|---|---|---|---|---|")
    P = {}
    for a in M2_HEAD_ARMS:
        r = u if a == "unguided" else m2_pool(m2_cells(rows, "m2", p, a, w))
        P[a] = r
        A("| `%s` | %.4f +- %.4f | %+.3f | %.5f | %.3f | %.5f | %.4f | %.4f | %.2f | %.2f |"
          % (a, r["ib"], r["se"], r["bias"], r["sd"], r["sd"] / u["sd"], r["kmer"], r["conf"],
             r["div"], 100 * r["clip"], r["mins"]))
    A("")
    return P


def section_m2(A, rows, store, root):
    A("## 4. Modality 2: simplex flow matching on DeepFlyBrain enhancers (gc, cpg)")
    A("")
    r0 = rows[("m2", "gc", "plug", 1.0, 0.5, M2_SEEDS[0])]
    n_head = sum(1 for k in rows if k[0] == "m2")
    n_abl = sum(1 for k in rows if k[0] == "m2abl")
    n_ws = sum(1 for k in rows if k[0] == "m2wsweep")
    A("**What ran.** One frozen generator (`%s`, 500 bp), %d headline cells (7 arms x {gc, cpg} "
      "x w in {1, 4} x 3 seeds), %d ablation cells (gc only: 17 BDG arms x w in {1, 4} x 3 "
      "seeds, plus plug / bdg_e4t0.5 / bdg_e4t1 at t_min = 0, w = 1, 3 seeds) and %d "
      "strength-sweep cells (gc, w = 16 and 64, **one seed**). n = %d, batch %d (%d controllers), "
      "%d Euler steps + simplex projection, guidance for t >= %g (%d guided steps per batch), "
      "q50 target, seeds %s. `cpg` ran the headline only (no ablation)."
      % (M2_CKPT, n_head, n_abl, n_ws, N, BATCH, r0["n_controllers"], r0["steps"],
         r0["t_min_guide"], m2_guided_per_batch(r0), ", ".join(str(s) for s in M2_SEEDS)))
    A("")
    A("| | |")
    A("|---|---|")
    for p in M2_PROPS:
        rp = rows[("m2", p, "unguided", 1.0, 0.5, M2_SEEDS[0])]
        A("| %s | target y = %.5f, corpus sd s = %.5f, delta = %.5f (a **chosen** band: "
          "max(0.16 s, 4.4 quantum), quantum %.6f) |"
          % (p, rp["y"], rp["s"], rp["delta"], rp["quantum"]))
    A("| arms | unguided, plug (DPS), tmpd, lgd_mc, **tfg_mc (TFG's MC-smoothing ingredient "
      "only, not full TFG)**, bdg_e4t0.5, bdg_e4t1 |")
    A("| cell fields | `gc_mean` / `gc_sd` hold the guided property for **both** gc and cpg; "
      "`in_band_fraction` is on the decoded sequence |")
    A("")
    # the headline contrast family this page reports, sized before the tables
    fam = []
    for p in M2_PROPS:
        u0 = m2_pool(m2_ung_cells(rows, p))
        for w in (1.0, 4.0):
            pl0 = m2_pool(m2_cells(rows, "m2", p, "plug", w))
            for a in M2_HEAD_ARMS[1:]:
                r = m2_pool(m2_cells(rows, "m2", p, a, w))
                fam.append(("unguided", zdiff(r["ib"], u0["ib"], r["N"], u0["N"])))
                if a in BDG_HEAD:
                    fam.append(("plug", zdiff(r["ib"], pl0["ib"], r["N"], pl0["N"])))
    n_u = sum(1 for k, _ in fam if k == "unguided")
    n_p = len(fam) - n_u
    zfam = V.NORM_Q(1.0 - ALPHA / (2 * len(fam)))
    ab = [abs(z) for _k, z in fam if abs(z) >= ZBAR]
    A("**Decision rule.** MODALITY2_V3_PROTOCOL.md states **no** decision rule (section 3.1: "
      "\"no verdict is computed\"; section 5 asks that any null be reported beside the minimum "
      "difference the run could detect, which the tables below print as MDD). This page "
      "therefore uses **the same unpaired z = %.2f as M1**, se = sqrt(p(1-p)/N), N = 3n = %d. "
      "That bar was derived for M1's 18 contrasts. This page's M2 headline tables report **%d "
      "contrasts** (%d against `unguided`, %d BDG against `plug`), whose Bonferroni bar is z = "
      "%.2f, so the inherited %.2f is slightly permissive here. %s"
      % (ZBAR, 3 * N, len(fam), n_u, n_p, zfam, ZBAR,
         ("It changes nothing: the weakest verdict past %.2f is |z| = %.2f, so all %d survive the "
          "wider bar." % (ZBAR, min(ab), len(ab))) if ab and min(ab) >= zfam else
         ("%d of the %d verdicts past %.2f do not clear %.2f."
          % (sum(1 for z in ab if z < zfam), len(ab), ZBAR, zfam)) if ab else
         "No contrast clears either bar."))
    A("")
    if M2R is not None:
        A("**Loader.** Cells are read and pooled through `proj1/m2/m2_v3_results.py` (`load`, "
          "`pooled`), the builder of the detailed M2 page, so both pages define every M2 number "
          "the same way. This script then checks the tree against the planned cell set and the "
          "pinned configuration, and recomputes every clip fraction from `m2_sweep.run_cell`'s "
          "loop (it refuses if the two disagree). Sources: "
          "`results/m2/{m2,m2abl,m2wsweep}/n2000/seed*/*.json`.")
    else:
        A("**Loader.** `proj1/m2/m2_v3_results.py` did not import (%s), so the cells are read "
          "directly here (`results/m2/{m2,m2abl,m2wsweep}/n2000/seed*/*.json`) with the same "
          "definitions, checked against the planned cell set and the pinned configuration."
          % M2R_ERR)
    A("")
    A("**Definitions.** in-band is N-weighted over seeds with a binomial se on N = %d; the "
      "property sd is the sd of the **pooled** sample (per-seed sds plus the between-seed shift "
      "of the mean), so it can differ in the 5th decimal from M2_RESULTS_gc.txt and "
      "M2_MECHANISM.md, which average the per-seed sds; bias/d is the pooled mean's offset from "
      "the target over delta." % (3 * N))
    A("")
    # identities
    M = ("in_band_fraction", "gc_mean", "gc_sd", "kmer_js")
    MA = M + ("bias_delta", "decode_conf", "diversity")
    e0 = {}
    for w in (1.0, 4.0):
        e0[w] = max(abs(float(x[k]) - float(y[k]))
                    for s in M2_SEEDS
                    for x, y in [(rows[("m2abl", "gc", "bdg_e0t1", w, 0.5, s)],
                                  rows[("m2", "gc", "plug", w, 0.5, s)])]
                    for k in MA + ("clipped_sample_steps",))
    same = {}
    same_all = {}
    for p in M2_PROPS:
        for w in (1.0, 4.0):
            for s in M2_SEEDS:
                pl = rows[("m2", p, "plug", w, 0.5, s)]
                for a in ("tmpd", "lgd_mc"):
                    x = rows[("m2", p, a, w, 0.5, s)]
                    same[(p, w)] = max(same.get((p, w), 0.0), max(abs(x[k] - pl[k]) for k in M))
                    same_all[(p, w)] = max(same_all.get((p, w), 0.0),
                                           max(abs(x[k] - pl[k]) for k in MA))
    ungd = max(abs(rows[("m2", p, "unguided", 1.0, 0.5, s)][k] - rows[("m2", p, "unguided", 4.0, 0.5, s)][k])
               for p in M2_PROPS for s in M2_SEEDS for k in MA)
    if ungd != 0.0:
        die("M2 unguided cells at w = 1 and w = 4 differ (max |d| %.1e); pooling unguided once "
            "from its w = 1 cells would be wrong" % ungd)
    rerun_ib = rerun_cl = rerun_ct = 0.0
    for w in (1.0, 4.0):
        for a in BDG_HEAD:
            for s in M2_SEEDS:
                x, y = rows[("m2", "gc", a, w, 0.5, s)], rows[("m2abl", "gc", a, w, 0.5, s)]
                rerun_ib = max(rerun_ib, abs(x["in_band_fraction"] - y["in_band_fraction"]))
                rerun_cl = max(rerun_cl, abs(x["clipped_sample_steps"] - y["clipped_sample_steps"]))
                rerun_ct = max(rerun_ct, max(abs(x[k] - y[k]) for k in MA))
    store["m2_ident"] = dict(e0=e0, same=same, ungd=ungd, rerun_ib=rerun_ib)
    A("### Identity checks (read before the tables)")
    A("")
    A("| check | result |")
    A("|---|---|")
    A("| pre-registered gate: `bdg_e0t1` (m2abl) = `plug` (m2), gc, bit for bit | w = 1: max \\|d\\| "
      "%.1e; w = 4: %.1e over 7 metrics + clipped steps -> **%s** |"
      % (e0[1.0], e0[4.0], "PASS" if e0[1.0] == 0 and e0[4.0] == 0 else "FAIL: BDG rows void"))
    A("| plug = tmpd = lgd_mc inside the window (protocol section 3.0) | max \\|d\\| over in-band, "
      "mean, sd, k-mer JS: %s; over all 7 metrics: %.1e |"
      % (", ".join("%s w=%g %.1e" % (p, w, same[(p, w)]) for p in M2_PROPS for w in (1.0, 4.0)),
         max(same_all.values())))
    A("| unguided cells at w = 1 and w = 4 (same run twice) | max \\|d\\| %.1e |" % ungd)
    A("| headline BDG arms re-run in m2abl (gc, same seed) | max \\|d in_band\\| %.1e, over all 7 "
      "metrics %.1e, max \\|d clipped steps\\| %d |" % (rerun_ib, rerun_ct, int(rerun_cl)))
    A("")
    A("**So plug, tmpd and lgd_mc are numerically one baseline here** (v_f/s^2 is 2.5e-5 at "
      "t = 0.5, so TMPD's denominator is s^2 and LGD's draws collapse to a point). They are "
      "listed below for completeness; their three verdicts are one verdict.")
    A("")
    # headline tables
    Pm2 = {}
    for p in M2_PROPS:
        for w in (1.0, 4.0):
            A("### %s, w = %g: full metric block" % (p, w))
            A("")
            Pm2[(p, w)] = m2_arm_table(A, rows, p, w)
    store["m2"] = Pm2
    um = {(p, w): mean([r["minutes"] for r in m2_cells(rows, "m2", p, "unguided", w)])
          for p in M2_PROPS for w in (1.0, 4.0)}
    A("sd / unguided is the pooled property sd divided by unguided's; clip %% = clipped "
      "sample-steps / (%d guided steps per batch x n), from "
      "`m2_sweep.run_cell` (cost `gen_vjp` is summed over the %d batches). min / cell is the "
      "cells' `minutes`; the device is **not recorded** in M2 cells. `unguided` is "
      "w-independent, so it is pooled **once from its w = 1 cells** in both strengths' blocks "
      "(M2_V3_RESULTS.md's convention; the w = 4 copies are the same samples, max |d| %.1e on "
      "every metric). Its min / cell is therefore the w = 1 copy's (%s); the w = 4 re-runs of the "
      "same samples took %s, which is scheduling noise, not work."
      % (m2_guided_per_batch(r0), N // BATCH, ungd,
         ", ".join("%s %.2f" % (p, um[(p, 1.0)]) for p in M2_PROPS),
         ", ".join("%s %.2f" % (p, um[(p, 4.0)]) for p in M2_PROPS)))
    A("")
    A("### Head-to-head: every guided arm against unguided")
    A("")
    A("| prop | w | arm | d in_band (pp) | z | verdict | MDD (pp) | d bias/d | sd ratio "
      "| d k-mer JS | d decode conf |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    vs = {}
    for p in M2_PROPS:
        for w in (1.0, 4.0):
            P = Pm2[(p, w)]
            u = P["unguided"]
            for a in M2_HEAD_ARMS[1:]:
                r = P[a]
                z = zdiff(r["ib"], u["ib"], r["N"], u["N"])
                vs[(p, w, a)] = (r["ib"] - u["ib"], z, mdd(r["ib"], u["ib"], r["N"], u["N"]))
                A("| %s | %g | `%s` | %s | %+.2f | %s | %s | %+.3f | %.3f | %+.5f | %+.4f |"
                  % (p, w, a, pp(r["ib"] - u["ib"]), z, verdict(z),
                     mdd_txt(r["ib"], u["ib"], r["N"], u["N"]), r["bias"] - u["bias"],
                     r["sd"] / u["sd"], r["kmer"] - u["kmer"], r["conf"] - u["conf"]))
    A("")
    tm = [x[2] for x in vs.values() if verdict(x[1]) == "tie"]
    A("MDD = %.2f x the contrast se. The %d ties above sit against MDDs of %.2f to %.2f pp: each is "
      "a limit on this run's resolution (|d| < MDD by construction), **not a measured "
      "equality** (MODALITY2_V3_PROTOCOL.md section 5)." % (ZBAR, len(tm), 100 * min(tm), 100 * max(tm)))
    A("")
    store["m2_vs"] = vs
    A("### BDG against plug (plug is BDG's eta = 0 limit)")
    A("")
    A("| prop | w | arm | in_band BDG / plug | d (pp) | z | verdict | MDD (pp) | seeds BDG > plug "
      "| bias/d BDG / plug | sd ratio BDG / plug | k-mer JS BDG / plug | decode conf BDG / plug "
      "| clip % BDG / plug |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    bvp = {}
    for p in M2_PROPS:
        for w in (1.0, 4.0):
            P = Pm2[(p, w)]
            u, pl = P["unguided"], P["plug"]
            for a in BDG_HEAD:
                b = P[a]
                z = zdiff(b["ib"], pl["ib"], b["N"], pl["N"])
                bvp[(p, w, a)] = (b["ib"] - pl["ib"], z, mdd(b["ib"], pl["ib"], b["N"], pl["N"]))
                pos = sum(1 for x, y in zip(b["ibs"], pl["ibs"]) if x > y)
                A("| %s | %g | `%s` | %.4f / %.4f | %s | %+.2f | %s | %s | %d of 3 | %+.3f / %+.3f "
                  "| %.3f / %.3f | %.5f / %.5f | %.4f / %.4f | %.2f / %.2f |"
                  % (p, w, a, b["ib"], pl["ib"], pp(b["ib"] - pl["ib"]), z, verdict(z),
                     mdd_txt(b["ib"], pl["ib"], b["N"], pl["N"]), pos,
                     b["bias"], pl["bias"], b["sd"] / u["sd"], pl["sd"] / u["sd"], b["kmer"],
                     pl["kmer"], b["conf"], pl["conf"], 100 * b["clip"], 100 * pl["clip"]))
    A("")
    tb = [x[2] for x in bvp.values() if verdict(x[1]) == "tie"]
    if tb:
        A("The %d ties above sit against MDDs of %.2f to %.2f pp, a limit on resolution, not a "
          "measured equality. `seeds BDG > plug` is descriptive and not pre-registered."
          % (len(tb), 100 * min(tb), 100 * max(tb)))
        A("")
    store["m2_bvp"] = bvp
    # ablation compact
    A("### Ablation (gc only), compact")
    A("")
    A("In-band against `bdg_e0t1` at fixed w; Spearman over the 16 eta > 0 rungs against the "
      "measured w_eff (seed mean of `diag.bdg_w_eff`). The full grid, as gains over unguided, is "
      "M2_ABLATION_GRID.md.")
    A("")
    A("| w | eta = 0 in_band | d e8t0.5 (pp) | z | v | its k-mer JS / eta 0 | its clip % "
      "| d e8t1.5 (pp) | z | v | tau 0.5 row rises with eta | Spearman(w_eff, d) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    m2abl = {}
    for w in (1.0, 4.0):
        base = m2_pool(m2_cells(rows, "m2abl", "gc", "bdg_e0t1", w))
        xs, ys, r05 = [], [], []
        for tau in TAUS:
            for eta in ETAS:
                r = m2_pool(m2_cells(rows, "m2abl", "gc", BR.arm_name(eta, tau), w))
                xs.append(r["w_eff"])
                ys.append(r["ib"] - base["ib"])
                if tau == "0.5":
                    r05.append(r["ib"] - base["ib"])
        mono = all(r05[i + 1] >= r05[i] for i in range(3)) and r05[0] >= 0
        ea = m2_pool(m2_cells(rows, "m2abl", "gc", "bdg_e8t0.5", w))
        eb = m2_pool(m2_cells(rows, "m2abl", "gc", "bdg_e8t1.5", w))
        za = zdiff(ea["ib"], base["ib"], ea["N"], base["N"])
        zb = zdiff(eb["ib"], base["ib"], eb["N"], base["N"])
        rho = BR.spearman(xs, ys)
        m2abl[w] = dict(base=base, ea=ea, eb=eb, za=za, zb=zb, rho=rho, mono=mono)
        A("| %g | %.4f | %s | %+.2f | %s | %.5f / %.5f | %.2f | %s | %+.2f | %s | %s | %+.2f |"
          % (w, base["ib"], pp(ea["ib"] - base["ib"]), za, verdict(za), ea["kmer"], base["kmer"],
             100 * ea["clip"], pp(eb["ib"] - base["ib"]), zb, verdict(zb),
             "yes" if mono else "**no**", rho))
    A("")
    store["m2abl"] = m2abl
    # largest inside window
    wb = m2_window_best(rows)
    cands, best, bp, ung = wb["cands"], wb["best"], wb["bp"], wb["ung"]
    zbest = zdiff(bp["ib"], ung["ib"], bp["N"], ung["N"])
    store["m2_best"] = (best, bp, ung, zbest)
    A("Largest gc in-band anywhere inside v3's window (t >= 0.5, w in {1, 4}, all %d distinct "
      "arm x w settings of the two trees; %d distinct configurations, since `bdg_e0t1` is plug): "
      "`%s` at w = %g, %.4f against unguided %.4f, **%s pp** (z %+.2f, a max over those "
      "settings, so not a pre-specified contrast), at k-mer JS %.5f (unguided %.5f), bias/d "
      "%+.3f (unguided %+.3f), and **%.1f %% of guided sample-steps clipped**. cpg has no "
      "ablation and no t = 0 cell; its largest in-window gain over unguided is %s pp (`%s`, "
      "w = %g, headline arms only). No window ratio is formed on cpg, and the gc and cpg gains "
      "are not compared with each other (each property has its own band; unguided in-band %.4f "
      "on gc, %.4f on cpg)."
      % (len(cands), wb["n_config"], best[1], best[2], bp["ib"], ung["ib"],
         pp(bp["ib"] - ung["ib"]), zbest, bp["kmer"], ung["kmer"], bp["bias"], ung["bias"],
         100 * bp["clip"], pp(wb["cpg_gain"]), wb["cpg_best"][0], wb["cpg_best"][1],
         ung["ib"], Pm2[("cpg", 1.0)]["unguided"]["ib"]))
    A("")
    # clip-limited and measured w_eff, for the M2 grid
    wq = [m2_pool(m2_cells(rows, "m2abl", "gc", a, w)) for w in (1.0, 4.0) for a in ABL_ARMS
          if a != "bdg_e0t1"]
    if any(r["w_eff"] is None for r in wq):
        die("an M2 ablation cell has no diag.bdg_w_eff")
    cl5 = {w: sum(1 for a in ABL_ARMS for s in M2_SEEDS
                  if m2_pool([rows[("m2abl", "gc", a, w, 0.5, s)]])["clip"] > 0.05)
           for w in (1.0, 4.0)}
    store["m2_weff"] = (min(r["w_eff"] for r in wq), max(r["w_eff"] for r in wq))
    A("**Clip-limited, and the measured w_eff.** The clip is applied after w, so a rung may be "
      "clip-limited rather than controller-limited (MODALITY2_V3_PROTOCOL.md section 4.1: compare "
      "the clip across strengths and mark such rungs). %d of %d gc ablation cells at w = 1 and %d "
      "of %d at w = 4 clip more than 5 %% of guided sample-steps; the e8t0.5 rung clips %.2f %% "
      "at w = 1 and %.2f %% at w = 4, so the top of the w = 4 ladder may be the clip, not the "
      "controller. Measured w_eff (seed means of `diag.bdg_w_eff`) spans %.2f to %.2f over the 16 "
      "eta > 0 rungs; quote the measured values."
      % (cl5[1.0], len(ABL_ARMS) * 3, cl5[4.0], len(ABL_ARMS) * 3,
         100 * m2abl[1.0]["ea"]["clip"], 100 * m2abl[4.0]["ea"]["clip"],
         store["m2_weff"][0], store["m2_weff"][1]))
    A("")
    # window
    A("### The window: t >= 0 against t >= 0.5 (gc, w = 1)")
    A("")
    A("| arm | in_band t >= 0.5 | in_band t >= 0 | window buys (pp) | clip % t >= 0.5 / t >= 0 "
      "| bias/d t >= 0 | k-mer JS t >= 0 |")
    A("|---|---|---|---|---|---|---|")
    win = {}
    for a in M2_WIN0_ARMS:
        late = Pm2[("gc", 1.0)][a]
        early = m2_pool([rows[("m2abl", "gc", a, 1.0, 0.0, s)] for s in M2_SEEDS])
        win[a] = (late, early)
        A("| `%s` | %.4f | %.4f | %s | %.2f / %.2f | %+.3f | %.5f |"
          % (a, late["ib"], early["ib"], pp(early["ib"] - late["ib"]), 100 * late["clip"],
             100 * early["clip"], early["bias"], early["kmer"]))
    A("")
    e_p = win["plug"][1]
    wz = {}
    for a in BDG_HEAD:
        e_b = win[a][1]
        eb_rows = [rows[("m2abl", "gc", a, 1.0, 0.0, s)] for s in M2_SEEDS]
        ep_rows = [rows[("m2abl", "gc", "plug", 1.0, 0.0, s)] for s in M2_SEEDS]
        wz[a] = dict(d=e_b["ib"] - e_p["ib"], z=zdiff(e_b["ib"], e_p["ib"], e_b["N"], e_p["N"]),
                     mdd=mdd(e_b["ib"], e_p["ib"], e_b["N"], e_p["N"]),
                     neg=sum(1 for x, y in zip(eb_rows, ep_rows)
                             if x["in_band_fraction"] < y["in_band_fraction"]))
    store["m2_win"] = (win, wz)
    clips0 = [win[a][1]["clip"] for a in M2_WIN0_ARMS]
    beats = [a for a in BDG_HEAD if wz[a]["z"] >= ZBAR]
    loses = [a for a in BDG_HEAD if wz[a]["z"] <= -ZBAR]
    A("At t >= 0, against plug: %s. %s The t quoted in M2_WINDOW_DOMINATES.md is a seed-paired "
      "t, not pre-registered, and the seed counts are descriptive."
      % ("; ".join("`%s` %s pp (unpaired z %+.2f, %s, MDD %.2f pp, below plug on %d of 3 seeds)"
                   % (a, pp(wz[a]["d"]), wz[a]["z"], verdict(wz[a]["z"]), 100 * wz[a]["mdd"],
                      wz[a]["neg"]) for a in BDG_HEAD),
         ("Neither headline BDG arm beats plug in the early window%s."
          % ((", and %s loses to it at the bar" % arms_txt(loses)) if loses else ""))
         if not beats else ("%s beats plug at the bar." % arms_txt(beats))))
    A("")
    A("On gc, moving plug from t >= 0.5 to t >= 0 is worth %s pp against the %s pp of the best "
      "in-window gc setting above (`%s`, w = %g), %s. No t = 0 or eta-sweep cell exists on cpg, "
      "whose largest in-window gain is %s pp (`%s`, w = %g); the ratio is not formed across "
      "properties. At t >= 0 the three arms clip %.1f-%.1f %% of guided sample-steps "
      "(denominator 100 guided steps per batch there), so that regime answers \"how good can "
      "guidance get\", not \"which rule is better\" (M2_WINDOW_DOMINATES.md)."
      % (pp(e_p["ib"] - win["plug"][0]["ib"]), pp(bp["ib"] - ung["ib"]), best[1], best[2],
         ratio_txt((e_p["ib"] - win["plug"][0]["ib"]) / (bp["ib"] - ung["ib"])),
         pp(wb["cpg_gain"]), wb["cpg_best"][0], wb["cpg_best"][1],
         100 * min(clips0), 100 * max(clips0)))
    A("")
    # strength sweep
    A("### Strength sweep beyond w = 4 (gc, **ONE seed**, %d)" % M2_SEEDS[0])
    A("")
    u1 = rows[("m2", "gc", "unguided", 1.0, 0.5, M2_SEEDS[0])]
    A("| arm | w | in_band | d vs unguided, same seed (pp) | bias/d | k-mer JS | clip % |")
    A("|---|---|---|---|---|---|---|")
    for a, w in M2_WSWEEP:
        r = m2_pool([rows[("m2wsweep", "gc", a, w, 0.5, M2_SEEDS[0])]])
        A("| `%s` | %g | %.4f | %s | %+.3f | %.5f | %.2f |"
          % (a, w, r["ib"], pp(r["ib"] - u1["in_band_fraction"]), r["bias"], r["kmer"],
             100 * r["clip"]))
    A("")
    A("One seed at n = 2000: no verdict is called on these rows.")
    A("")
    # force share
    with open(os.path.join(ROOT, "results", "m2_share.json"), encoding="utf-8") as f:
        share = json.load(f)
    sg = share["by_prop"]["gc"]
    A("### Fidelity cost, and the force confound")
    A("")
    for p in M2_PROPS:
        for w in (1.0, 4.0):
            P = Pm2[(p, w)]
            u = P["unguided"]
            A("- %s, w = %g: k-mer JS unguided %.5f; `bdg_e4t0.5` %.5f (%.2fx), plug %.5f "
              "(%.2fx); decode conf %.4f -> %.4f (bdg_e4t0.5); diversity %.4f -> %.4f."
              % (p, w, u["kmer"], P["bdg_e4t0.5"]["kmer"], P["bdg_e4t0.5"]["kmer"] / u["kmer"],
                 P["plug"]["kmer"], P["plug"]["kmer"] / u["kmer"], u["conf"],
                 P["bdg_e4t0.5"]["conf"], u["div"], P["bdg_e4t0.5"]["div"]))
    A("- Applied correction share on gc (results/m2_share.json: n = %d, seed %d, a diagnostic): "
      "plug %.4f / `bdg_e4t0.5` %.4f / `bdg_e4t1` %.4f at w = 1 and %.4f / %.4f / %.4f at w = 4. "
      "`bdg_e4t0.5` pushes about %.0fx harder than plug at the same w, so part of any "
      "`bdg_e4t0.5` gain is force; the eta sweep at fixed w above is what separates them."
      % (share["n"], share["seed"], sg["plug"]["w1"]["share"], sg["bdg_e4t0.5"]["w1"]["share"],
         sg["bdg_e4t1"]["w1"]["share"], sg["plug"]["w4"]["share"], sg["bdg_e4t0.5"]["w4"]["share"],
         sg["bdg_e4t1"]["w4"]["share"], sg["bdg_e4t0.5"]["w1"]["share"] / sg["plug"]["w1"]["share"]))
    with open(os.path.join(ROOT, "results", "m2_dfb_activity.json"), encoding="utf-8") as f:
        dfb = json.load(f)
    with open(os.path.join(ROOT, "results", "m2_gate_scores.json"), encoding="utf-8") as f:
        gate = json.load(f)
    uc = [c for c in dfb["cells"].values() if c["arm"] == "unguided" and c["w"] == 1.0]
    w64 = [c for c in dfb["cells"].values() if c["w"] == 64.0]
    A("- DeepFlyBrain (results/m2_dfb_activity.json): max topic / fraction predicted accessible "
      "is %.3f / %.2f-%.2f on real sequences and %.3f / %.3f on unguided samples (%d cells); "
      "the w = 64 rows (%s) are **one seed**. The enhancer gate's AUC is %.3f "
      "(m2_gate_scores.json). The port's \"verified identical to Keras\" is in **no committed "
      "file**; `deepflybrain.py` says it was validated without TensorFlow."
      % (dfb["reference"]["real_train"]["max_topic_mean"],
         min(dfb["reference"]["real_train"]["frac_active"], dfb["reference"]["real_test"]["frac_active"]),
         max(dfb["reference"]["real_train"]["frac_active"], dfb["reference"]["real_test"]["frac_active"]),
         mean([c["dfb_max_topic"] for c in uc]), mean([c["dfb_frac_active"] for c in uc]), len(uc),
         ", ".join("%s %s: %.3f / %.3f"
                   % ("bdg_" + c["variant"] if c["arm"] == "bdg" else c["arm"], c["prop"],
                      c["dfb_max_topic"], c["dfb_frac_active"]) for c in w64),
         gate["gate_auc"]))
    A("")
    # cost
    A("### Cost in time")
    A("")
    A("| arm | cells (m2 tree) | mean min / cell | min / 1000 sequences |")
    A("|---|---|---|---|")
    for a in M2_HEAD_ARMS:
        rs = [rows[("m2", p, a, w, 0.5, s)] for p in M2_PROPS for w in (1.0, 4.0) for s in M2_SEEDS]
        m = mean([r["minutes"] for r in rs])
        A("| `%s` | %d | %.2f | %.2f |" % (a, len(rs), m, m / N * 1000))
    A("")
    mt = {st: sum(r["minutes"] for k, r in rows.items() if k[0] == st) for st in ("m2", "m2abl", "m2wsweep")}
    for st in ("m2", "m2abl", "m2wsweep"):
        store["cost"].append(("M2 (`%s`)" % M2_CKPT, st, sum(1 for k in rows if k[0] == st),
                              "not recorded", 60 * mt[st]))
    A("Total recorded time: %.1f h (m2) + %.1f h (m2abl) + %.2f h (m2wsweep) over %d cells. Per-cell "
      "minutes vary by up to ~6x for identical work (cpg unguided: 0.29 to 1.69 min), so treat "
      "these as wall-clock under unknown concurrency. This table counts every cell the m2 tree "
      "holds, so `unguided`'s 12 include its w = 4 re-runs of the w = 1 samples (the metric "
      "blocks above pool it once, from w = 1)."
      % (mt["m2"] / 60, mt["m2abl"] / 60, mt["m2wsweep"] / 60, len(rows)))
    A("")
    # outcome
    A("### Outcome")
    A("")
    A("- **Against unguided** (arms clearing z >= %.2f): %s."
      % (ZBAR, "; ".join("%s w=%g: %s" % (p, w, arms_txt([a for a in M2_HEAD_ARMS[1:]
                                                          if vs[(p, w, a)][1] >= ZBAR]))
                         for w in (1.0, 4.0) for p in M2_PROPS)))
    A("- **BDG against plug**: %s."
      % "; ".join("`%s`: %s" % (a, ", ".join("%s w=%g %s pp (z %+.2f, %s)"
                                             % (p, w, pp(bvp[(p, w, a)][0]), bvp[(p, w, a)][1],
                                                verdict(bvp[(p, w, a)][1]))
                                             for p in M2_PROPS for w in (1.0, 4.0)))
                  for a in BDG_HEAD))
    contr = [(p, w) for p in M2_PROPS for w in (1.0, 4.0)
             if Pm2[(p, w)]["bdg_e4t0.5"]["sd"] < Pm2[(p, w)]["plug"]["sd"]
             and abs(Pm2[(p, w)]["bdg_e4t0.5"]["bias"]) > abs(Pm2[(p, w)]["plug"]["bias"])]
    A("- `bdg_e4t0.5` has a smaller property sd **and** a larger |bias|/d than plug in %d of 4 "
      "(property, w) cells (%s): its in-band gain comes from **contracting the spread, not from "
      "re-centring on the target** (M2_MECHANISM.md)."
      % (len(contr), ", ".join("%s w=%g" % c for c in contr) or "none"))
    A("- On gc the eta sweep at fixed w tracks measured w_eff (Spearman %+.2f at w = 1, %+.2f at "
      "w = 4); e8t0.5 vs eta = 0 is %s pp (w = 1, z %+.2f) and %s pp (w = 4, z %+.2f), at %.1f %% "
      "clipped steps for the w = 4 rung."
      % (m2abl[1.0]["rho"], m2abl[4.0]["rho"], pp(m2abl[1.0]["ea"]["ib"] - m2abl[1.0]["base"]["ib"]),
         m2abl[1.0]["za"], pp(m2abl[4.0]["ea"]["ib"] - m2abl[4.0]["base"]["ib"]), m2abl[4.0]["za"],
         100 * m2abl[4.0]["ea"]["clip"]))
    A("- The window dominates on gc: moving plug from t >= 0.5 to t >= 0 is worth %s pp, against "
      "%s pp for the best in-window gc setting (`%s`, w = %g), %s; cpg has no t = 0 cell. At "
      "t >= 0 against plug: %s."
      % (pp(win["plug"][1]["ib"] - win["plug"][0]["ib"]), pp(bp["ib"] - ung["ib"]), best[1],
         best[2], ratio_txt((win["plug"][1]["ib"] - win["plug"][0]["ib"]) / (bp["ib"] - ung["ib"])),
         ", ".join("`%s` %s pp (z %+.2f, %s)" % (a, pp(wz[a]["d"]), wz[a]["z"], verdict(wz[a]["z"]))
                   for a in BDG_HEAD)))
    A("")
    A("### Caveats")
    A("")
    A("- M2's delta is a **chosen** band (delta_ratio 0.16, floor 4.4 quantum), so M2 in-band is "
      "not commensurable with M1's; never pool or rank across modalities.")
    A("- v3's window t >= 0.5 vs t = 0, **on gc**: the window is worth %s the method (plug "
      "%s pp from the window against the best in-window gc rung, %s pp at `%s` w = %g). cpg has "
      "no t = 0 or eta-sweep cell (largest in-window gain %s pp), so no ratio is formed on it or "
      "across properties."
      % (ratio_txt((win["plug"][1]["ib"] - win["plug"][0]["ib"]) / (bp["ib"] - ung["ib"])),
         pp(win["plug"][1]["ib"] - win["plug"][0]["ib"]),
         pp(bp["ib"] - ung["ib"]), best[1], best[2], pp(wb["cpg_gain"])))
    A("- plug, tmpd and lgd_mc are one baseline inside the window (v_f collapses); `tfg_mc` is "
      "not full TFG.")
    A("- The clip is applied after w: the gc ablation's top rung (e8t0.5) clips %.2f %% of guided "
      "sample-steps at w = 4 and `bdg_e4t0.5` clips %.2f / %.2f %% (gc / cpg) at w = 4, so those "
      "rows may be clip-limited rather than controller-limited. Measured w_eff spans %.2f to "
      "%.2f over the gc grid (seed means); quote the measured values."
      % (100 * m2abl[4.0]["ea"]["clip"], 100 * Pm2[("gc", 4.0)]["bdg_e4t0.5"]["clip"],
         100 * Pm2[("cpg", 4.0)]["bdg_e4t0.5"]["clip"], store["m2_weff"][0], store["m2_weff"][1]))
    c4 = Pm2[("cpg", 4.0)]
    A("- `cpg` has no ablation, so its BDG gains cannot be split into force and feedback. Its "
      "largest one (`bdg_e4t0.5`, w = 4, %s pp over unguided) moves bias/d from %+.3f to %+.3f "
      "and k-mer JS from %.5f to %.5f, at %.1f %% clipped steps."
      % (pp(c4["bdg_e4t0.5"]["ib"] - c4["unguided"]["ib"]), c4["unguided"]["bias"],
         c4["bdg_e4t0.5"]["bias"], c4["unguided"]["kmer"], c4["bdg_e4t0.5"]["kmer"],
         100 * c4["bdg_e4t0.5"]["clip"]))
    A("- The w = 16 / 64 rows are one seed; the DeepFlyBrain-vs-Keras equivalence is in no "
      "committed file; m2_dfb_activity.json and m2_gate_scores.json hold one entry per cell "
      "file name, so cells present in both the m2 and m2abl trees collide there and some "
      "ablation cells are absent (not used in this page beyond the rows quoted).")
    A("- Same-seed re-runs agree on in-band to %.1e but are not bit-identical (continuous "
      "metrics differ by up to %.1e, clipped steps by up to %d)."
      % (rerun_ib, rerun_ct, int(rerun_cl)))
    A("")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "v3"))
    ap.add_argument("--m2-root", default=os.path.join(ROOT, "results", "m2"))
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    zb = V.NORM_Q(1.0 - ALPHA / (2 * N_CONTRASTS))
    if abs(zb - ZBAR) > 0.005:
        die("the pre-registered bar recomputes to %.4f, not %.2f" % (zb, ZBAR))

    # DERIVED from the registry, not a literal: this tuple was ("fm","equifm",
    # "edm") while BACKEND_ARMS already carried a fourth backend, and the
    # section that wanted it died on a KeyError.
    M1_BACKENDS = tuple(BACKEND_ARMS)
    head = {be: m1_load(args.root, be, "v3") for be in M1_BACKENDS}
    abl = {be: {w: m1_load(args.root, be, "v3abl", w=w) for w in (1, 4)} for be in ("fm", "equifm")}
    for _be in ("edm", "vp"):
        if os.path.isdir(os.path.join(args.root, _be, "v3abl")):
            die("results/v3/%s/v3abl exists; v3 plans no %s ablation" % (_be, _be))
    Ph = {be: pooled_all(c) for be, c in head.items()}
    Pa = {be: {w: pooled_all(c) for w, c in d.items()} for be, d in abl.items()}
    m2rows = m2_load(args.m2_root)

    # -------- consistency with the generated docs (refuse on any M1 mismatch)
    report = []
    RD = os.path.join(ROOT, "docs", "results")
    exp_all = {}
    for be in M1_BACKENDS:
        e = v3_expected(be, head[be], Ph[be], "v3")
        exp_all.update(e)
        check_doc(os.path.join(RD, "V3_RESULTS_%s.md" % be), e, report, "V3_RESULTS_%s.md" % be)
    check_doc(os.path.join(RD, "V3_RESULTS.md"), exp_all, report, "V3_RESULTS.md")
    for w in (1, 4):
        ew = {}
        for be in ("fm", "equifm"):
            ew.update(v3_expected(be, abl[be][w], Pa[be][w], "v3abl"))
        check_doc(os.path.join(RD, "V3_RESULTS_ABL_w%d.md" % w), ew, report,
                  "V3_RESULTS_ABL_w%d.md" % w)
    br = []
    for be in ("fm", "equifm"):
        for p in PROPS:
            pl, plr = Ph[be][(p, "plug")], head[be][(p, "plug")]
            for arm in BDG_HEAD:
                b, brr = Ph[be][(p, arm)], head[be][(p, arm)]
                z = zdiff(b["in_band_fraction"], pl["in_band_fraction"], b["N"], pl["N"])
                zd = zdiff(b["in_band_fraction_dec"], pl["in_band_fraction_dec"], b["N"], pl["N"])
                zo = zdiff(b["o2"], pl["o2"], b["N"], pl["N"])
                pos = sum(1 for x, y in zip(brr, plr) if x["in_band_fraction"] > y["in_band_fraction"])
                br.append("| %s | %s | `%s` | %.4f / %.4f | %s | %+.2f | %s | %+.2f | %s | %+.2f | "
                          "%d of 3 | %s | %s | %s / %s / %s |"
                          % (be, p, arm, b["in_band_fraction"], pl["in_band_fraction"],
                             pp(b["in_band_fraction"] - pl["in_band_fraction"]), z,
                             pp(b["in_band_fraction_dec"] - pl["in_band_fraction_dec"]), zd,
                             pp(b["o2"] - pl["o2"]), zo, pos,
                             pp(b["mol_stability"] - pl["mol_stability"]),
                             pp(b["validity"] - pl["validity"]), verdict(z), verdict(zd),
                             verdict(zo)))
        for s in abl_summary({(be, 1): abl[be][1], (be, 4): abl[be][4]}, be):
            br.append(blade_summary_row(s))
        idn = identity(head[be], abl[be][1])
        br.append("| %s | %.2e | %.2e | %.2e | %.2e |" % (be, idn["mi"], idn["ms"], idn["ri"], idn["rs"]))
    check_lines(os.path.join(RD, "V3_BLADE_READOUT.md"), br, report, "V3_BLADE_READOUT.md")

    # M2 docs: reported, not refused on (they are not generated from one loader)
    m2checks = []
    # M2_RESULTS_gc.txt (m2_table.py, gc, w = 1)
    with open(os.path.join(RD, "M2_RESULTS_gc.txt"), encoding="utf-8") as f:
        txt = f.read().split(NL)
    ok = bad = 0
    badl = []
    for ln in txt:
        m = re.match(r"^(\S+)\s+([\d.]+) \+/- [\d.]+\s+([\d.]+)\s+([+-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%", ln)
        if not m or m.group(1) not in M2_HEAD_ARMS:
            continue
        # m2_table.py's own definitions: plain means over seeds, sd = mean of per-seed sds
        rs = m2_cells(m2rows, "m2", "gc", m.group(1), 1.0)
        mm = lambda k: mean([x[k] for x in rs])
        clipm = mean([x["clipped_sample_steps"] / float(m2_guided_per_batch(x) * x["n"])
                      if m2_guided_per_batch(x) else 0.0 for x in rs])
        mine = ("%.4f" % mm("in_band_fraction"), "%.5f" % mm("gc_sd"), "%+.3f" % mm("bias_delta"),
                "%.5f" % mm("kmer_js"), "%.3f" % mm("decode_conf"), "%.1f" % (100 * clipm))
        theirs = (m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7))
        if mine == theirs:
            ok += 1
        else:
            bad += 1
            badl.append("%s: doc %s, cells %s" % (m.group(1), theirs, mine))
    m2checks.append(("M2_RESULTS_gc.txt (gc, w = 1: in-band, sd, bias/d, k-mer JS, conf, clip %)",
                     ok, bad, badl))
    # M2_ABLATION_GRID.md: replicate m2_ablation_grid.py (it pools the m2 and m2abl copies)
    ungs = {(k[1], k[5]): r["in_band_fraction"] for k, r in m2rows.items() if k[2] == "unguided"}
    grid = {}
    for k, r in m2rows.items():
        if k[0] in ("m2", "m2abl") and k[1] == "gc" and k[2].startswith("bdg_") and k[4] == 0.5:
            m = re.match(r"^e([0-9.]+)t([0-9.]+)$", r["variant"])
            grid.setdefault((k[3], float(m.group(2)), float(m.group(1))), []).append(
                r["in_band_fraction"] - ungs[("gc", k[5])])
    for w in (1.0, 4.0):
        for tau in (0.5, 0.75, 1.0, 1.5):
            grid.setdefault((w, tau, 0.0), grid[(w, 1.0, 0.0)])
    with open(os.path.join(RD, "M2_ABLATION_GRID.md"), encoding="utf-8") as f:
        gtxt = f.read().split(NL)
    ok = bad = 0
    badl = []
    cur_w, in_gc = None, False
    for ln in gtxt:
        if ln.startswith("## "):
            in_gc = ln.strip() == "## gc"
        m = re.match(r"^### w = ([\d.]+)", ln)
        if m:
            cur_w = float(m.group(1))
        m = re.match(r"^\| ([\d.]+) \| (.*) \|$", ln)
        if in_gc and cur_w and m:
            tau = float(m.group(1))
            for eta, cell in zip((0, 1, 2, 4, 8), m.group(2).split(" | ")):
                mine = "%+.2f" % (100 * mean(grid[(cur_w, tau, float(eta))]))
                if mine == cell.strip():
                    ok += 1
                else:
                    bad += 1
                    badl.append("w %g tau %g eta %d: doc %s, cells %s" % (cur_w, tau, eta, cell, mine))
    m2checks.append(("M2_ABLATION_GRID.md (gc, 2 x 20 cells, its own pooling of both trees)",
                     ok, bad, badl))
    # M2_WINDOW_DOMINATES.md (hand-written): the window table
    with open(os.path.join(RD, "M2_WINDOW_DOMINATES.md"), encoding="utf-8") as f:
        wtxt = f.read().split(NL)
    ok = bad = 0
    badl = []
    for ln in wtxt:
        m = re.match(r"^\| `([\w.]+)` \| ([\d.]+) \| \**([\d.]+)\** \| \**([+-][\d.]+) pp\** \|", ln)
        if not m or m.group(1) not in M2_WIN0_ARMS:
            continue
        late = m2_pool(m2_cells(m2rows, "m2", "gc", m.group(1), 1.0))
        early = m2_pool([m2rows[("m2abl", "gc", m.group(1), 1.0, 0.0, s)] for s in M2_SEEDS])
        mine = ("%.4f" % late["ib"], "%.4f" % early["ib"], "%+.2f" % (100 * (early["ib"] - late["ib"])))
        theirs = (m.group(2), m.group(3), m.group(4))
        if mine == theirs:
            ok += 1
        else:
            bad += 1
            badl.append("%s: doc %s, cells %s" % (m.group(1), theirs, mine))
    m2checks.append(("M2_WINDOW_DOMINATES.md window table (hand-written)", ok, bad, badl))
    # M2_V3_RESULTS.md (the detailed M2 page, written by a separate builder): its
    # four headline sections' metric rows and its vs-unguided / vs-plug rows
    m2v = os.path.join(RD, "M2_V3_RESULTS.md")
    if os.path.isfile(m2v):
        with open(m2v, encoding="utf-8") as f:
            vtxt = f.read().split(NL)
        ok = bad = 0
        badl = []
        sec, tab = None, None
        for ln in vtxt:
            m = re.match(r"^#### `(gc|cpg)` at w = (\d+)", ln)
            if m:
                sec, tab = (m.group(1), float(m.group(2))), None
                continue
            if ln.startswith("## ") or ln.startswith("### "):
                sec, tab = None, None
                continue
            if sec is None:
                continue
            if ln.startswith("| arm | in_band |"):
                tab = "metric"
                continue
            if ln.startswith("| arm vs `unguided`"):
                tab = "unguided"
                continue
            if ln.startswith("| arm vs `plug`"):
                tab = "plug"
                continue
            m = re.match(r"^\| `([\w.]+)` \| (.*) \|$", ln)
            if not (m and tab and m.group(1) in M2_HEAD_ARMS):
                continue
            a, cells = m.group(1), [c.strip() for c in m.group(2).split(" | ")]
            P = {x: m2_pool(m2_cells(m2rows, "m2", sec[0], x, sec[1])) for x in M2_HEAD_ARMS}
            r = P[a]
            if tab == "metric":
                mine = ["%.4f" % r["ib"], "%.4f" % r["se"], "%+.3f" % r["bias"], "%.5f" % r["sd"],
                        "%.3f" % (r["sd"] / P["unguided"]["sd"]), "%.5f" % r["kmer"],
                        "%.4f" % r["conf"], "%.4f" % r["div"]]
                theirs = [cells[i] for i in (0, 1, 2, 3, 4, 6, 8, 9)]
            else:
                b = P[tab]
                z = zdiff(r["ib"], b["ib"], r["N"], b["N"])
                mine = ["%+.2f pp" % (100 * (r["ib"] - b["ib"])), "%+.2f" % z, "**%s**" % verdict(z)]
                theirs = cells[:3]
            if mine == theirs:
                ok += 1
            else:
                bad += 1
                badl.append("%s w=%g %s vs %s: doc %s, here %s" % (sec[0], sec[1], a, tab, theirs, mine))
        m2checks.append(("M2_V3_RESULTS.md headline sections (metric, vs-unguided and vs-plug rows)",
                         ok, bad, badl))
    else:
        m2checks.append(("M2_V3_RESULTS.md (not present when this page was generated)", 0, 0, []))

    # ------------------------------------------------------------ the page
    L = []
    A = L.append
    store = {"vs_ung": {}, "bvp": {}, "abl": {}, "cost": [],
             "ungc": {be: ung_copies(head[be]) for be in head}}
    A("# v3-final summary: the blade run (n = 2000), one page per component")
    A("")
    A("Generated by `proj1/scripts/v3_final_summary.py`. Do not hand-edit; re-run it:")
    A("")
    A("```")
    A(CMD)
    A("```")
    A("")
    n_m1 = (sum(len(v) for c in head.values() for v in c.values())
            + sum(len(v) for d in abl.values() for c in d.values() for v in c.values()))
    ws_seeds = sorted({k[5] for k in m2rows if k[0] == "m2wsweep"})
    A("**Sources** (committed stats JSONs only, %d M1 + %d M2 cells): M1 headline "
      "`results/v3/{fm,equifm,edm,vp}/v3/n2000/seed{20261001,20261002,20261003}/`; M1 ablation "
      "`results/v3/{fm,equifm}/v3abl/n2000/` (w = 1 and 4); M2 `results/m2/{m2,m2abl}/"
      "n2000/seed{20260921,20260922,20260923}/`; M2 strength sweep `results/m2/m2wsweep/n2000/"
      "seed%s/` (**%s**); M2 diagnostics `results/m2_share.json`, "
      "`m2_dfb_activity.json`, `m2_gate_scores.json`. n = %d per cell, batch %d, 100-step Euler, "
      "target q50, guidance window t >= 0.5, headline w = 1, ablation w in {1, 4}. Protocols: "
      "FULL_RUN_V3_PROTOCOL.md, ABLATION_V3_PROTOCOL.md, MODALITY2_V3_PROTOCOL.md. The older Betty "
      "run (`results/v3/<be>/n5000/`) is not read."
      % (n_m1, len(m2rows), ",".join(str(s) for s in ws_seeds),
         "one seed" if len(ws_seeds) == 1 else "%d seeds" % len(ws_seeds), N, BATCH))
    A("")
    A("**Statistics.** Every M1 verdict is the **pre-registered** unpaired contrast: se = "
      "sqrt(p(1-p)/N) per arm, N = 3n = %d, z = d / sqrt(se_a^2 + se_b^2), Bonferroni 0.05/18 "
      "two-sided, **z = %.2f** (FULL_RUN_V3_PROTOCOL.md section 6.1). M2 verdicts use the same "
      "form at the same bar, **inherited rather than pre-registered** (MODALITY2_V3_PROTOCOL.md "
      "computes no verdict; section 4). `above` / `below` = z >= "
      "%.2f / z <= -%.2f, else `tie`. Beside every null the tables print the **minimum detectable "
      "difference**, MDD = %.2f x the contrast se (v3_power.py's definition; FULL_RUN_V3_PROTOCOL.md "
      "section 6.1 and MODALITY2_V3_PROTOCOL.md section 5 require it): a tie is |d| < MDD by "
      "construction, so it is a limit on this run's resolution, not a measured equality. Unpaired "
      "is conservative (arms share initial noise), so a tie can be a paired win. No paired or "
      "pooled statistic carries a verdict here; the Spearman correlations, the property-averaged "
      "chemistry means, the seeds-BDG > plug counts and the molecule-stability z are descriptive "
      "and **not pre-registered**. M2's protocol states no decision rule, so M2 uses the same bar "
      "(section 4)." % (3 * N, ZBAR, ZBAR, ZBAR, ZBAR))
    A("")
    A("## Read this first")
    A("")
    idf = identity(head["fm"], abl["fm"][1])
    fm_exact = idf["allk"] == 0 and idf["clipd"] == 0 and idf["ri"] == 0 and idf["rs"] == 0
    uce = ung_copies(head["edm"])
    wb = m2_window_best(m2rows)
    we_all = [BR.w_eff(abl[be][w][(p, a)]) for be in abl for w in (1, 4) for p in PROPS
              for a in ABL_ARMS if a != "bdg_e0t1"]
    if any(x is None for x in we_all):
        die("an M1 ablation cell has no diag.bdg_w_eff")
    for c in (
        "**q50 only.** No q90 number appears anywhere on this page.",
        "**in_band is not comparable across M1 backends** (each has its own property pair, so its "
        "own delta and band width) **nor between M1 and M2** (M2's delta is a chosen band). "
        "Nothing is tested or ranked across backends or modalities.",
        "**w = 1 is not any arm's best strength and equal w is not equal force.** Nothing here "
        "says which method is best (FULL_RUN_V3_PROTOCOL.md section 6.2).",
        "**Both headline BDG arms are always reported separately** (`bdg_e4t0.5`, `bdg_e4t1`); "
        "they are never pooled, and plug (BDG at eta = 0) is the base arm for the dispersion term.",
        "**The ablation supports \"the deviation weight w_eff, not the global w, carries the "
        "effect\". It does not show that the feedback loop is necessary.** The measured w_eff "
        "(seed means, %.2f to %.2f over the 12 M1 grids) is not the protocol's pre-run 10^2-10^3, "
        "and the high-w_eff rungs may be clip-limited (sections 2-3 caveats)."
        % (min(we_all), max(we_all)),
        "**EquiFM is not bit-reproducible**: same-seed re-runs move in-band by up to %.1e, so its "
        "binomial se understates its noise. `fm` %s in this run; `edm`'s same-seed unguided runs "
        "%s (section 1)."
        % (identity(head["equifm"], abl["equifm"][1])["ri"],
           "is bit-exact" if fm_exact else "is **not** bit-exact (section 2)",
           ("differ by up to %.1e in molecule stability" % uce["mol_stability"])
           if max(uce.values()) > 0 else "agree exactly"),
        "**Clip fraction** = clipped sample-steps / (guided steps per batch x n). M1's "
        "`guided_steps` is summed over the n/batch = %d batches; `v3_sanity.py` divides by the sum "
        "and understates every M1 fraction %dx. M2's denominator is derived from "
        "`m2_sweep.run_cell` (section 4)." % (N // BATCH, N // BATCH),
        "**`edm` is TFG's EDMsecond, a BORROWED diffusion checkpoint; `vp` is our OWN trained "
        "QM9 VP diffusion (18 cells, 28 Sep). Both are diffusion; only `vp` is ours.**",
        "**Useful yield** (decoded in-band AND stable, per molecule) **is pending**: the "
        "`*.permol.pt` sidecars stay on blade. It is never approximated by multiplying marginals.",
        "**M2**: on gc, moving plug from t >= 0.5 to t >= 0 is worth %s pp against the %s pp of "
        "the best in-window gc setting (`%s`, w = %g), %s; no t = 0 or eta-sweep cell exists on "
        "cpg, whose largest in-window gain is %s pp (`%s`, w = %g), and the ratio is not formed "
        "across properties. Plug, tmpd and lgd_mc are one baseline in the window; `tfg_mc` is "
        "not full TFG; the w = 16 / 64 rows are one seed; the DeepFlyBrain-vs-Keras equivalence "
        "is in no committed file."
        % (pp(wb["win_gain"]), pp(wb["bp"]["ib"] - wb["ung"]["ib"]), wb["best"][1],
           wb["best"][2], ratio_txt(wb["ratio"]), pp(wb["cpg_gain"]), wb["cpg_best"][0],
           wb["cpg_best"][1]),
    ):
        A("- " + c)
    A("")
    A("Column key (M1): `in_band` = continuous, oracle f_B; `dec` = on decoded molecules; `o2` / "
      "`o2 dec` = the second oracle (OC-Flow classifier, its own delta); MAE, bias and sd are "
      "divided by delta; `DV` = distinct valid molecules per attempt; `MDD` = minimum detectable "
      "difference; `v` = verdict. `diversity` = mean pairwise distance of the centred, "
      "unit-normalised embeddings **in the oracle f_B's own network** (`evaluation.embedding_diversity`), so it "
      "is a different measurement for each property and each pair: never compare it across "
      "properties, backends or modalities. The same unguided `fm` molecules score %s on its "
      "mu / alpha / gap oracles. (M2's `diversity` is a mean pairwise Hamming distance between "
      "decoded sequences, another quantity again.)"
      % " / ".join("%.4f" % Ph["fm"][(p, "unguided")]["diversity_mean_pairwise"] for p in PROPS))
    A("")
    A("Contents: 1. QM9 diffusion (`edm`, borrowed) - 1b. QM9 diffusion (`vp`, OURS) - "
      "2. Our flow-matching model (`fm`) - 3. EquiFM (`equifm`, borrowed) - 4. Modality 2 - "
      "5. Across the run - 6. Consistency with the generated docs.")
    A("")
    section_edm(A, head["edm"], Ph["edm"], store)
    section_vp(A, head["vp"], Ph["vp"], Ph["fm"], store)
    section_flow(A, 2, "fm", head["fm"], abl["fm"], Ph["fm"], Pa["fm"], store)
    section_flow(A, 3, "equifm", head["equifm"], abl["equifm"], Ph["equifm"], Pa["equifm"], store)
    section_m2(A, m2rows, store, args.m2_root)

    # ------------------------------------------------------------ across
    A("## 5. Across the run")
    A("")
    A("### Verdict scoreboard (z >= %.2f; nothing here compares backends)" % ZBAR)
    A("")
    A("Arms that clear unguided, per component, property and strength. M1 lists continuous / "
      "decoded / second-oracle; M2 has one (decoded) in-band. w = 4 M1 rows hold only the three "
      "arms carried forward from the headline (plug = `bdg_e0t1`, `bdg_e4t0.5`, `bdg_e4t1`), read "
      "against the headline's unguided; the other 14 ablation arms are summarised under the "
      "table (below unguided) and in sections 2-3 (both directions).")
    A("")
    A("| component | prop | w | clear unguided: continuous | decoded | second oracle | below "
      "unguided (any) |")
    A("|---|---|---|---|---|---|---|")
    for be in ("edm", "fm", "equifm"):
        for w in ((1,) if be == "edm" else (1, 4)):
            res = store["vs_ung"][(be, w)]
            for p in PROPS:
                arms = [a for (pp_, a) in res if pp_ == p]
                lab = lambda a: "plug" if a == "bdg_e0t1" else a
                cl = {k: [lab(a) for a in arms if res[(p, a)][k][1] >= ZBAR] for k in ("cont", "dec", "o2")}
                bl = sorted({lab(a) for a in arms for k in ("cont", "dec", "o2") if res[(p, a)][k][1] <= -ZBAR})
                A("| `%s` | %s | %d | %s | %s | %s | %s |"
                  % (be, p, w, arms_txt(cl["cont"]), arms_txt(cl["dec"]), arms_txt(cl["o2"]),
                     arms_txt(bl)))
    for p in M2_PROPS:
        for w in (1.0, 4.0):
            cl = [a for a in M2_HEAD_ARMS[1:] if store["m2_vs"][(p, w, a)][1] >= ZBAR]
            bl = [a for a in M2_HEAD_ARMS[1:] if store["m2_vs"][(p, w, a)][1] <= -ZBAR]
            A("| M2 | %s | %g | -- | %s | -- | %s |" % (p, w, arms_txt(cl), arms_txt(bl)))
    A("")
    bel = [(be, w, p, a, c) for be in ("fm", "equifm") for (w, p, a, c) in store["abl_vs_ung"][be][1]]
    zsel = V.NORM_Q(1.0 - ALPHA / (2 * 192))
    A("**The rest of the ablation tree.** The `none`s in the last column cover the three arms "
      "shown only. Across all 17 ablation arms, **%d M1 ablation cells fall below unguided** at "
      "z <= -%.2f on at least one of continuous / decoded / second oracle (single contrasts, "
      "not selection-adjusted; %d of them also pass z <= -%.2f on continuous, Bonferroni over "
      "the 192 eta > 0 cells, derived here): %s. See sections 2 and 3 for their z and measured w_eff.%s"
      % (len(bel), ZBAR, sum(1 for x in bel if x[4]["cont"][1] <= -zsel), zsel,
         "; ".join("`%s` %s w = %d `%s` %s pp" % (be, p, w, a, pp(c["cont"][0]))
                   for (be, w, p, a, c) in bel) or "none",
         (" Every `equifm` one carries its re-run noise (up to %.1e in-band)."
          % identity(head["equifm"], abl["equifm"][1])["ri"])
         if any(x[0] == "equifm" for x in bel) else ""))
    A("")
    A("**BDG against plug, above / below / tie, per component** (M1: continuous | decoded | "
      "second oracle over 3 properties; M2: in-band over gc and cpg):")
    A("")
    A("| component | w | `bdg_e4t0.5` | `bdg_e4t1` |")
    A("|---|---|---|---|")
    for be in ("fm", "equifm"):
        bvp = store["bvp"][be]
        for w in (1, 4):
            cells = []
            for a in BDG_HEAD:
                cells.append(", ".join("%s %s" % (lab, tally([verdict(bvp[(w, p, a)][k][1])
                                                              for p in PROPS]).replace(" ", ""))
                                       for k, lab in (("cont", "cont"), ("dec", "dec"), ("o2", "o2"))))
            A("| `%s` | %d | %s | %s |" % (be, w, cells[0], cells[1]))
    for w in (1.0, 4.0):
        cells = ["in-band " + tally([verdict(store["m2_bvp"][(p, w, a)][1])
                                     for p in M2_PROPS]).replace(" ", "") for a in BDG_HEAD]
        A("| M2 | %g | %s | %s |" % (w, cells[0], cells[1]))
    A("")
    A("`edm` has no BDG arm. Sums over fm + equifm at w = 1 reproduce V3_BLADE_READOUT.md "
      "section 1. Every `equifm` verdict in both tables carries its re-run noise (up to %.1e "
      "in-band on a same-seed re-run), which the binomial se does not include."
      % identity(head["equifm"], abl["equifm"][1])["ri"])
    A("")
    A("### Costs")
    A("")
    A("| component | stage | cells | device | recorded time (h) | mean min / cell | min / 1000 samples |")
    A("|---|---|---|---|---|---|---|")
    for comp, st, nc, dev, secs in store["cost"]:
        A("| %s | %s | %d | %s | %.2f | %.2f | %.2f |"
          % (comp, st, nc, dev, secs / 3600, secs / nc / 60, secs / nc / 60 / N * 1000))
    A("")
    A("M1 time is the cells' `seconds` on one RTX A6000 per cell (V3_RESULTS*.md print the same "
      "totals as GPU-h). M2 cells record `minutes` but not the device. SCOPE_FM_GUIDANCE_STATUS.md's "
      "\"11.9 + 21.1 GPU-h for the ablation\" is the **w = 1 half only**; the full ablation is "
      "%.1f + %.1f GPU-h."
      % (sum(s for c, st, n_, d, s in store["cost"] if c == "`fm`" and st.startswith("ablation")) / 3600,
         sum(s for c, st, n_, d, s in store["cost"] if c == "`equifm`" and st.startswith("ablation")) / 3600))
    A("")
    A("### What is still pending")
    A("")
    A("- **Useful yield** (decoded in-band AND stable, paired per molecule) for the blade cells. "
      "The sidecars stay on blade: `python proj1/scripts/v3_paired.py --n 2000 --backends "
      "fm,equifm --md-out docs/results/V3_BLADE_PAIRED.md`, and `--stage v3abl --w 1` (then 4) "
      "for the grid.")
    A("- **Our own QM9 VP diffusion model has run** (`vp`, 28 Sep, 18 cells, checkpoint md5 "
      "`8a3390a6`, selected epoch 1475). It, not `edm`, fills `tab:fmvd`'s \"VP, trained "
      "here\" row. Matched to `fm` on backbone, parameters, epochs, batch, EMA, split and "
      "seed, so the contrast isolates the generator family.")
    A("- **Paper-table rows with no run**: `tab:fmvd` (VP row; the FM row has no matched "
      "comparison); `tab:ablation` \"One-sided residual\", \"Open-loop replay\" and "
      "\"Signed-strength plug-in\" (none of these arms is in v3), and its eta = 0 and gain/setpoint "
      "rows are defined as **paired** changes, which wait for `v3_paired.py`; `tab:m2` Dirichlet "
      "FM, Fisher Flow and MOG-DFM (no cells); `tab:training` provenance entries.")
    A("- **Fillable now from this page**: `tab:recent` (fm and equifm headline, both BDG rows, "
      "time from the cost tables), `tab:guidance` (fm unguided, plug w = 1, plug w = 4 = "
      "`bdg_e0t1`), and the simplex-FM rows of `tab:m2`.")
    A("- A fixed-w_eff (open-loop) control, without which \"the feedback is necessary\" cannot be "
      "claimed.")
    A("")
    A("### Detailed docs")
    A("")
    m2doc = os.path.isfile(os.path.join(RD, "M2_V3_RESULTS.md"))
    for d, what in (
        ("V3_RESULTS.md", "M1 headline full metric blocks, all four backends (+ `_fm`, `_equifm`, `_edm`, `_vp`)"),
        ("V3_RESULTS_ABL_w1.md", "M1 ablation full metric blocks at w = 1 (`_w4` for w = 4)"),
        ("V3_BLADE_READOUT.md", "BDG vs plug per cell, the pooled README figure, all 12 grids with measured w_eff, blade vs Betty"),
        ("M2_V3_RESULTS.md", "detailed M2 page (proj1/m2/m2_v3_results.py)" + ("" if m2doc else "; **not present when this page was generated**")),
        ("M2_ABLATION_GRID.md", "M2 gc eta x tau grid as gains over unguided"),
        ("M2_RESULTS_gc.txt", "m2_table.py output, gc w = 1"),
        ("M2_SHARE.md", "M2 applied correction share"),
        ("M2_MECHANISM.md", "hand-written: contraction vs re-centring (with its 28 Sep correction)"),
        ("M2_WINDOW_DOMINATES.md", "hand-written: the window (with its 28 Sep correction)"),
    ):
        A("- [%s](%s): %s" % (d, d, what))
    A("- Status: [SCOPE_FM_GUIDANCE_STATUS.md](../status/SCOPE_FM_GUIDANCE_STATUS.md), top section "
      "\"v3 FINISHED ON BLADE\".")
    A("")
    # ------------------------------------------------------------ consistency
    A("## 6. Consistency with the generated docs")
    A("")
    A("Every M1 number on this page is computed by `v3_table.load/pooled` and "
      "`v3_blade_readout`'s helpers. Before writing, the script re-derives, from the same cells, "
      "the rows those docs print and requires each to appear **verbatim** in the committed doc "
      "(it refuses to write otherwise):")
    A("")
    A("| doc | rows re-derived and found verbatim |")
    A("|---|---|")
    for name, n_ok in report:
        A("| %s | %d |" % (name, n_ok))
    A("")
    A("The V3_RESULTS* rows are the full metric rows, the vs-unguided rows, the clip-count rows "
      "and the measured-cost line; the V3_BLADE_READOUT rows are section 1 (BDG vs plug), the "
      "12-grid summary and the eta = 0 control. The M2 docs are checked and reported, not "
      "enforced:")
    A("")
    A("| M2 doc | agree | disagree |")
    A("|---|---|---|")
    for name, ok, bad, badl in m2checks:
        A("| %s | %d | %d%s |" % (name, ok, bad, (": " + "; ".join(badl[:4])) if badl else ""))
    A("")

    out = NL.join(L) + NL
    if args.md_out:
        dst = args.md_out if os.path.isabs(args.md_out) else os.path.join(ROOT, args.md_out)
        with open(dst, "w", encoding="utf-8", newline="\n") as f:
            f.write(out)
        print("wrote %s (%d lines)" % (dst, len(L)), file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
