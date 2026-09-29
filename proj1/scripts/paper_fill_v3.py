"""Map every pending cell (\\pend) of the paper's result tables to the v3-final
number that fills it, and cross-check the generated V3_RESULTS docs on the way.

    python proj1/scripts/paper_fill_v3.py --md-out docs/results/PAPER_TABLE_FILL_V3.md --latex-check

WHAT IT READS, and nothing else:
  * paper/main.tex. The six tables that hold \\pend cells (tab:fmvd, tab:guidance,
    tab:ablation, tab:recent, tab:m2, tab:training): rows, columns, captions.
    The paper is READ ONLY. This script never writes under paper/.
  * results/v3/<be>/v3/n2000/seed{20261001,20261002,20261003}/   M1 headline, blade
    (fm, equifm, edm)
  * results/v3/<be>/v3abl/n2000/seed{...}/   M1 ablation, blade (fm, equifm; w = 1, 4)
  * results/m2/{m2,m2abl,m2wsweep}/n2000/seed{20260921,20260922,20260923}/   M2
  * weights/*.pt and proj1/m2/blade_bundle/fm_m2_dfb500.pt: md5 and stored
    training arguments, for tab:training only (these are NOT v3 numbers)
  * docs/results/V3_RESULTS*.md: only to check them against the cells.

It never reads results/v3/<be>/n5000/ (the older Betty run), v2, q90 or any
pilot tree.

STATISTICS. M1 verdicts use the pre-registered rule of FULL_RUN_V3_PROTOCOL.md
section 6.1: unpaired binomial se = sqrt(p(1-p)/N) per arm, N = 3n = 6000,
z = d / sqrt(se_a^2 + se_b^2), Bonferroni 0.05/18 two-sided, |z| >= 2.99.
MODALITY2_V3_PROTOCOL.md states no significance threshold, so M2 uses the same
rule and says so. Nothing paired is computed: the per-molecule sidecars stay
on blade.

It REFUSES (exits non-zero with the reason) on a missing, duplicated or
misconfigured cell, and on a paper table whose rows no longer match the map
written here, rather than emitting a stale or partial fill.

--latex-check additionally copies the paper to a temporary directory, measures
every proposed tabular with pdflatex, compiles the paper with the recommended
and the alternative tables substituted, and reports the page count the way
paper/tools/pagecheck.py does. The temporary directory is deleted afterwards.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import v3_table as V  # noqa: E402  (reused loader, pooled() and binomial z)

NL = "\n"
SCRIPT = "proj1/scripts/paper_fill_v3.py"
REGEN = ("python proj1/scripts/paper_fill_v3.py --md-out "
         "docs/results/PAPER_TABLE_FILL_V3.md --latex-check")
# The \pend cells are defined by the committed v4 manuscript. paper/main.tex is
# the live file, which other revisions fill while this runs; it is AUDITED here
# (section 1b of the doc), never used as the map's anchor and never written.
PAPER = os.path.join(ROOT, "paper", "versions", "v4_main.tex")
PAPER_LIVE = os.path.join(ROOT, "paper", "main.tex")
V3_ROOT = os.path.join(ROOT, "results", "v3")
M2_ROOT = os.path.join(ROOT, "results", "m2")
N_CELL = 2000
BATCH = 500
SEEDS_M1 = ["20261001", "20261002", "20261003"]
SEEDS_M2 = [20260921, 20260922, 20260923]
PROPS = ("mu", "alpha", "gap")
PLABEL = {"mu": "mu", "alpha": "alpha", "gap": "gap"}
PTEX = {"mu": r"$\mu$", "alpha": r"$\alpha$", "gap": "gap"}
HEAD_ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg", "bdg_e4t0.5", "bdg_e4t1")
GUIDED = HEAD_ARMS[1:]
BDG_HEAD = ("bdg_e4t0.5", "bdg_e4t1")
BE_LABEL = {"fm": "our FM (flow-matching EGNN, trained here) + our property pair",
            "equifm": "EquiFM (borrowed release) + TFG's property pair",
            "edm": "EDMsecond (TFG's released diffusion checkpoint, BORROWED) + our pair"}
BE_SHORT = {"fm": "our FM", "equifm": "EquiFM", "edm": "EDM (ext.)"}
M2_PROPS = ("gc", "cpg")
M2_ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg_mc", "bdg_e4t0.5", "bdg_e4t1")
ABL_ETAS = (1, 2, 4, 8)
ABL_TAUS = ("0.5", "0.75", "1", "1.5")
ABL_ARMS = ["bdg_e0t1"] + ["bdg_e%dt%s" % (e, t) for e in ABL_ETAS for t in ABL_TAUS]


def die(msg):
    sys.stderr.write("REFUSING: " + msg + NL)
    raise SystemExit(1)


# ----------------------------------------------------------------- statistics
# The pre-registered bar: Bonferroni over 6 guided arms x 3 properties = 18
# contrasts against unguided, alpha = 0.05 two-sided.
Z_BAR = V.NORM_Q(1.0 - 0.05 / (2.0 * 18))
if abs(Z_BAR - 2.99) > 0.01:
    die("the Bonferroni bar computed to %.4f, not the protocol's 2.99" % Z_BAR)
# The ablation grid is a selection (ABLATION_V3_PROTOCOL.md sec. 4: "17 settings x
# 3 properties x 2 bases"; a max-T or Bonferroni threshold must be stated beside
# any win claimed from it, and it is wider than the headline's). Bonferroni over
# that family, alpha = 0.05 two-sided. Nothing about the grid is pre-registered.
N_SEL = 17 * 3 * 2
Z_SEL = V.NORM_Q(1.0 - 0.05 / (2.0 * N_SEL))
if abs(Z_SEL - 3.49) > 0.01:
    die("the ablation family's Bonferroni bar computed to %.4f, not 3.49" % Z_SEL)


def bse(p, n):
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def zun(pa, pb, n):
    """Unpaired binomial z, n per arm (the pre-registered test)."""
    s = math.sqrt(bse(pa, n) ** 2 + bse(pb, n) ** 2)
    return (pa - pb) / s if s > 0 else float("nan")


def verdict(z):
    if z != z:
        return "n/a"
    return "above" if z >= Z_BAR else ("below" if z <= -Z_BAR else "tie")


def mdd(p, n):
    """Minimum difference detectable at the bar, for two arms near p."""
    return Z_BAR * math.sqrt(2.0) * bse(p, n)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def sdev(xs):
    xs = list(xs)
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan")


def f3(x):
    return "%.3f" % x


def f4(x):
    return "%.4f" % x


def pp(x, d=2):
    return ("%+." + str(d) + "f") % (100.0 * x)


# ------------------------------------------------------------------- M1 cells
def m1_scan_duplicates(be, stage):
    """load() files every tr__*.json of a seed into one list per (prop, arm), so
    a duplicate in one seed plus a gap in another would still count 3 rows.
    Check uniqueness per (seed, prop, arm, w) from the JSON itself."""
    seen = {}
    for s in SEEDS_M1:
        d = os.path.join(V3_ROOT, be, stage, "n%d" % N_CELL, "seed%s" % s)
        for fn in sorted(glob.glob(os.path.join(d, "tr__*.json"))):
            with open(fn, encoding="utf-8") as fh:
                r = json.load(fh)
            k = (s, r["prop"], r["arm"], float(r["w"]))
            if k in seen:
                die("duplicate M1 cell %s/%s seed %s: %s and %s"
                    % (be, stage, s, os.path.basename(seen[k]), os.path.basename(fn)))
            seen[k] = fn
            if str(r.get("seed")) != s:
                die("%s sits in seed%s/ but records seed %s" % (fn, s, r.get("seed")))
    return seen


def m1_config_check(be, stage, cells):
    for (p, a), rows in cells.items():
        for r in rows:
            bad = []
            if r["n"] != N_CELL:
                bad.append("n=%s" % r["n"])
            if r.get("batch") != BATCH:
                bad.append("batch=%s" % r.get("batch"))
            if r["steps"] != 100:
                bad.append("steps=%s" % r["steps"])
            if r["target_name"] != "q50":
                bad.append("target=%s" % r["target_name"])
            if abs(float(r["t_start"]) - 0.5) > 1e-12:
                bad.append("t_start=%s" % r["t_start"])
            if a != "unguided":
                g = r["guided_steps"] / (r["n"] / r["batch"])
                if abs(g - round(g)) > 1e-9:
                    bad.append("guided_steps %s not a multiple of n/batch" % r["guided_steps"])
            if bad:
                die("%s/%s/%s/%s seed %s: %s" % (be, stage, p, a, r["_seed"], ", ".join(bad)))


def load_m1():
    """{(be, stage, w): {(prop, arm): [rows by seed]}}; exits on any problem."""
    out = {}
    # `vp` is OUR OWN QM9 diffusion, added 28 Sep: headline only, no ablation,
    # same as `edm`. crosscheck_docs reads every docs/results/V3_RESULTS_*.md
    # page and indexes M1 by the backend it finds there, so a page that exists
    # without a matching entry here is a KeyError, not a skipped check.
    plan = [("fm", "v3", None), ("equifm", "v3", None), ("edm", "v3", None),
            ("vp", "v3", None),
            ("fm", "v3abl", 1), ("fm", "v3abl", 4),
            ("equifm", "v3abl", 1), ("equifm", "v3abl", 4)]
    for be, stage, w in plan:
        m1_scan_duplicates(be, stage)
        c, pr = V.load(V3_ROOT, be, N_CELL, SEEDS_M1, stage=stage, w=w)
        pr = V.check(c, pr, stage=stage, backend=be)
        if pr:
            die("M1 %s/%s w=%s: %s" % (be, stage, w, "; ".join(pr)))
        m1_config_check(be, stage, c)
        out[(be, stage, 1 if w is None else w)] = c
    # the arms the paper's removed rows would need: none may exist anywhere
    return out


def m1_arm_census():
    arms = set()
    for fn in glob.glob(os.path.join(V3_ROOT, "*", "v3*", "n%d" % N_CELL, "seed*", "tr__*.json")):
        with open(fn, encoding="utf-8") as fh:
            arms.add(json.load(fh)["arm"])
    return sorted(arms)


def m1_stats(rows):
    P = V.pooled(rows)
    N = P["N"]
    wm = lambda f: sum(f(r) * r["n"] for r in rows) / N  # noqa: E731
    s = dict(P)
    s["ib"] = P["in_band_fraction"]
    s["ib_dec"] = P["in_band_fraction_dec"]
    s["o2"] = wm(lambda r: r["oracle2"]["in_band"])
    s["o2_dec"] = wm(lambda r: r["oracle2"]["in_band_dec"])
    s["o2_delta"] = rows[0]["oracle2"]["delta"]
    bias_dec = wm(lambda r: r["f_B_dec_mean"]) - wm(lambda r: r["target_mean"])
    s["bias_dec"] = bias_dec
    s["sd_dec"] = math.sqrt(max(wm(lambda r: r["prop_rmse_eval_dec"] ** 2) - bias_dec ** 2, 0.0))
    # v3_table.pooled squares the MEAN rmse; the pooled variance needs the mean
    # of the SQUARES. Kept separately so the size of that difference is visible.
    s["sd_exact"] = math.sqrt(max(wm(lambda r: r["prop_rmse_eval"] ** 2) - P["bias"] ** 2, 0.0))
    den = sum((r["guided_steps"] / (r["n"] / r["batch"])) * r["n"] for r in rows)
    s["clip_frac"] = (sum(r["clipped_sample_steps"] for r in rows) / den) if den else 0.0
    s["s1k"] = [r["seconds"] * 1000.0 / r["n"] for r in rows]
    s["seed"] = {k: [r[k] for r in rows] for k in
                 ("mol_stability", "validity", "uniqueness_of_valid",
                  "unique_valid_per_sample", "in_band_fraction_dec")}
    d0 = [r.get("diag") or {} for r in rows]
    if all("bdg_w_eff" in d for d in d0):
        s["weff3"] = mean(d["bdg_w_eff"] for d in d0)
        s["weff_rms3"] = math.sqrt(max(mean(d["bdg_w_eff_sq"] for d in d0), 0.0))
        s["weff_neg3"] = mean(d["bdg_w_eff_neg"] for d in d0)
    return s


# ------------------------------------------------------------------- M2 cells
def m2_guided_steps(steps, t_min):
    dt = 1.0 / steps
    return sum(1 for i in range(steps) if i * dt >= t_min)


M2_EXPECT = {}
for _p in M2_PROPS:
    for _a in M2_ARMS:
        for _w in (1.0, 4.0):
            M2_EXPECT[("m2", _p, _a, _w, 0.5)] = 3
for _a in ABL_ARMS:
    for _w in (1.0, 4.0):
        M2_EXPECT[("m2abl", "gc", _a, _w, 0.5)] = 3
for _a in ("plug", "bdg_e4t0.5", "bdg_e4t1"):
    M2_EXPECT[("m2abl", "gc", _a, 1.0, 0.0)] = 3
for _a in ("plug", "bdg_e4t0.5"):
    for _w in (16.0, 64.0):
        M2_EXPECT[("m2wsweep", "gc", _a, _w, 0.5)] = 1


def load_m2():
    cells = {}
    for stage in ("m2", "m2abl", "m2wsweep"):
        for fn in sorted(glob.glob(os.path.join(M2_ROOT, stage, "n%d" % N_CELL, "seed*", "*.json"))):
            with open(fn, encoding="utf-8") as fh:
                r = json.load(fh)
            sd = os.path.basename(os.path.dirname(fn))
            if sd != "seed%d" % r["seed"]:
                die("%s sits in %s/ but records seed %s" % (fn, sd, r["seed"]))
            if r["seed"] not in SEEDS_M2:
                die("%s: seed %s is not one of M2's three" % (fn, r["seed"]))
            name = "%s__%s__%s__w%g__%s__n%d_nfe%d_win%g_b%d_dr%g__s%d.json" % (
                r["prop"], r["arm"], r["target_name"], r["w"], r["variant"], r["n"],
                r["steps"], r["t_min_guide"], r["batch"], r["delta_ratio"], r["seed"])
            if name != os.path.basename(fn):
                die("%s: its fields say it should be named %s" % (fn, name))
            bad = []
            for k, want in (("n", N_CELL), ("batch", BATCH), ("steps", 100),
                            ("target_name", "q50"), ("delta_ratio", 0.16),
                            ("ckpt", "fm_m2_dfb500.pt"), ("stage", stage),
                            ("clip", 1.0)):
                if r.get(k) != want:
                    bad.append("%s=%r" % (k, r.get(k)))
            if r["n"] // r["batch"] != r["n_controllers"]:
                bad.append("n_controllers=%s" % r["n_controllers"])
            if bad:
                die("%s: %s" % (fn, ", ".join(bad)))
            arm = r["arm"] if r["arm"] != "bdg" else "bdg_" + r["variant"]
            key = (stage, r["prop"], arm, float(r["w"]), float(r["t_min_guide"]))
            per = cells.setdefault(key, {})
            if r["seed"] in per:
                die("duplicate M2 cell %s seed %s" % (key, r["seed"]))
            r["_file"] = os.path.relpath(fn, ROOT).replace(os.sep, "/")
            per[r["seed"]] = r
    missing = [k for k, ns in M2_EXPECT.items() if len(cells.get(k, {})) != ns]
    if missing:
        die("M2 cells missing or incomplete: %s" % ", ".join(
            "%s (%d of %d seeds)" % (k, len(cells.get(k, {})), M2_EXPECT[k]) for k in missing[:6]))
    extra = sorted(str(k) for k in cells if k not in M2_EXPECT)
    for p in M2_PROPS:
        ds = {round(r["delta"], 12) for k, per in cells.items() if k[1] == p for r in per.values()}
        ys = {round(r["y"], 12) for k, per in cells.items() if k[1] == p for r in per.values()}
        if len(ds) != 1 or len(ys) != 1:
            die("M2 %s cells disagree on delta %s or target %s" % (p, ds, ys))
    # guided-step accounting, which the clip fraction's denominator rests on
    for k, per in cells.items():
        for r in per.values():
            want = 0 if k[2] == "unguided" else m2_guided_steps(r["steps"], r["t_min_guide"])
            got = r["cost"]["gen_vjp"] / r["n_controllers"]
            if abs(got - want) > 1e-9:
                die("%s seed %s: gen_vjp/n_controllers = %s, expected %d guided steps"
                    % (k, r["seed"], got, want))
    return cells, extra


def m2_rows(cells, stage, prop, arm, w, t=0.5):
    per = cells[(stage, prop, arm, float(w), float(t))]
    return [per[s] for s in sorted(per)]


def m2_stats(rows):
    N = sum(r["n"] for r in rows)
    ib = sum(r["in_band_fraction"] * r["n"] for r in rows) / N
    M = sum(r["gc_mean"] * r["n"] for r in rows) / N
    ss = sum((r["n"] - 1) * r["gc_sd"] ** 2 + r["n"] * (r["gc_mean"] - M) ** 2 for r in rows)
    y, dl = rows[0]["y"], rows[0]["delta"]
    # gc_sd is torch's unbiased sd (m2_sweep.py: gc.std()), so this pooled sd is
    # exact; sd_pop treats gc_sd as a population sd, as other M2 docs do
    ss_pop = sum(r["n"] * r["gc_sd"] ** 2 + r["n"] * (r["gc_mean"] - M) ** 2 for r in rows)
    s = {"N": N, "ib": ib, "mean": M, "sd": math.sqrt(ss / (N - 1)), "sd_pop": math.sqrt(ss_pop / N),
         "bias_d": (M - y) / dl, "bias_d_cells": mean(r["bias_delta"] for r in rows),
         "js": mean(r["kmer_js"] for r in rows), "conf": mean(r["decode_conf"] for r in rows),
         "div": mean(r["diversity"] for r in rows), "delta": dl, "y": y,
         "s1k": [r["minutes"] * 60.0 * 1000.0 / r["n"] for r in rows],
         "seeds": len(rows)}
    den = sum(r["cost"]["gen_vjp"] * r["batch"] for r in rows)
    s["clip_frac"] = (sum(r["clipped_sample_steps"] for r in rows) / den) if den else 0.0
    if all("bdg_w_eff" in (r.get("diag") or {}) for r in rows):
        s["weff"] = mean(r["diag"]["bdg_w_eff"] for r in rows)
    return s


M2_FIELDS = ("in_band_fraction", "gc_mean", "gc_sd", "kmer_js", "decode_conf", "diversity")


def m2_functional(M2):
    """The committed functional scores: the realism gate and DeepFlyBrain activity
    (results/m2_gate_scores.json, results/m2_dfb_activity.json). NOT part of any
    pre-registered gate. Keyed by the cell's file stem (+ .permol.pt, the blade
    sidecar the scorer read), and cross-checked against the cell's own in-band."""
    stems = {}
    for k, per in M2.items():
        for r in per.values():
            stem = os.path.basename(r["_file"])[:-len(".json")] + ".permol.pt"
            # the m2abl re-runs of the headline BDG cells share their stem, and
            # their samples (in-band identical): one scored cell covers both
            stems.setdefault(stem, (k, r))
    out = {"n_expected": len(stems)}
    for tag, fn, fields in (("gate", "m2_gate_scores.json", ("gate_mean", "gate_pass")),
                            ("dfb", "m2_dfb_activity.json", ("dfb_max_topic", "dfb_frac_active"))):
        with open(os.path.join(ROOT, "results", fn), encoding="utf-8") as fh:
            d = json.load(fh)
        cells = d["cells"]
        unknown = sorted(set(cells) - set(stems))
        if unknown:
            die("results/%s scores cells no M2 cell file matches: %s" % (fn, unknown[:3]))
        grp = {}
        for st, c in cells.items():
            k, r = stems[st]
            if abs(c["in_band"] - r["in_band_fraction"]) > 1e-6:
                die("results/%s: %s has in_band %.6f, its cell %.6f" % (fn, st, c["in_band"], r["in_band_fraction"]))
            grp.setdefault((k[1], k[2], k[3], k[4]), {})[r["seed"]] = {f: c[f] for f in fields}
        # the largest drop against unguided, read two ways: on the same seeds
        # (paired; the right reading for the one-seed cells) and against
        # unguided's 3-seed mean (how M2_V3_RESULTS.md reads it)
        worst, worst_m = None, None
        for key, per in grp.items():
            if key[1] == "unguided":
                continue
            ug = grp.get((key[0], "unguided", 1.0, 0.5), {})
            common = [s for s in per if s in ug]
            if not common:
                continue
            dlt = mean(per[s][fields[0]] - ug[s][fields[0]] for s in common)
            if worst is None or dlt < worst[0]:
                worst = (dlt, key, len(common))
            dm = mean(v[fields[0]] for v in per.values()) - mean(v[fields[0]] for v in ug.values())
            if worst_m is None or dm < worst_m[0]:
                worst_m = (dm, key, len(per))
        miss = {}
        for st in set(stems) - set(cells):
            k, r = stems[st]
            miss.setdefault((k[1], k[2], k[3], k[4]), []).append(r["seed"])
        out[tag] = {"ref": d.get("reference"), "auc": d.get("gate_auc"), "grp": grp, "worst": worst, "worst_m": worst_m,
                    "fields": fields, "n_scored": len(cells), "miss": miss}
    return out


def m2_maxdiff(ra, rb):
    return max(abs(a[k] - b[k]) for a, b in zip(ra, rb) for k in M2_FIELDS)


# ------------------------------------------------------------------ the paper
def paper_tables(path=PAPER, strict=True, src=None):
    if src is None:
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
    lines = src.split(NL)
    out = {}
    for lab in ("tab:fmvd", "tab:guidance", "tab:ablation", "tab:recent", "tab:m2", "tab:training"):
        key = "\\label{" + lab + "}"
        if src.count(key) != 1:
            if not strict:
                out[lab] = None
                continue
            die("%s has %d copies of %s" % (os.path.relpath(path, ROOT), src.count(key), key))
        i = src.index(key)
        a = src.rfind("\\begin{table}", 0, i)
        b = src.index("\\end{table}", i) + len("\\end{table}")
        env = src[a:b]
        c0 = env.index("\\caption{") + len("\\caption{")
        depth, j = 1, c0
        while depth:
            depth += {"{": 1, "}": -1}.get(env[j], 0)
            j += 1
        caption = env[c0:j - 1]
        t0 = env.index("\\begin{tabular}")
        t1 = env.index("\\end{tabular}") + len("\\end{tabular}")
        tab = env[t0:t1]
        spec = re.match(r"\\begin\{tabular\}\{([^}]*)\}", tab).group(1)
        body = tab[len("\\begin{tabular}{" + spec + "}"):-len("\\end{tabular}")]
        rows = []
        for ln in body.split("\\\\"):
            ln = re.sub(r"\\(toprule|midrule|bottomrule)", "", ln).strip()
            if ln:
                rows.append([c.strip() for c in ln.split("&")])
        out[lab] = {"env": env, "caption": caption, "tabular": tab, "spec": spec,
                    "header": rows[0], "rows": rows[1:], "pend": env.count("\\pend"),
                    "line": src[:a].count(NL) + 1}
    return src, lines, out


EXPECT_ROWS = {
    "tab:fmvd": ["FM, trained here", "VP, trained here"],
    "tab:guidance": ["Unguided", "Plug-in", "Plug-in"],
    "tab:ablation": ["$\\eta=0$", "Gain/setpoint, $w=1,4$", "One-sided residual",
                     "Open-loop replay", "Signed-strength plug-in"],
    "tab:recent": ["Unguided, selected backbone", "DPS-style plug-in \\citep{chung2023dps}",
                   "TMPD-inspired \\citep{boys2024tmpd}", "LGD-MC \\citep{song2023lgd}",
                   "TFG \\citep{ye2024tfg}", "BDG (ours)"],
    "tab:m2": ["Simplex FM, unguided", "Simplex FM + plug-in", "Simplex FM + BDG",
               "Dirichlet FM \\citep{stark2024dirichlet}", "Fisher Flow \\citep{davis2024fisher}",
               "MOG-DFM \\citep{chen2025multi}"],
    "tab:training": ["Architecture", "Parameters", "Training split", "Objective", "Epochs",
                     "Batch", "Optimizer", "Learning rate", "EMA", "Run seed",
                     "Hashes / hardware"],
}
EXPECT_PEND = {"tab:fmvd": 8, "tab:guidance": 15, "tab:ablation": 15, "tab:recent": 24,
               "tab:m2": 24, "tab:training": 7}


def check_paper(T):
    for lab, want in EXPECT_ROWS.items():
        got = [r[0] for r in T[lab]["rows"]]
        if got != want:
            die("%s rows changed since this map was written:%s  paper: %s%s  map:   %s"
                % (lab, NL, got, NL, want))
        if T[lab]["pend"] != EXPECT_PEND[lab]:
            die("%s holds %d \\pend, the map was written for %d"
                % (lab, T[lab]["pend"], EXPECT_PEND[lab]))


# ------------------------------------------------------- live paper audit
NUMRE = re.compile(r"\d*\.\d+|\d+")


def cell_nums(c):
    return [("0" + t) if t.startswith(".") else t for t in NUMRE.findall(c)]


def m1_arm_of(label):
    lo = label.lower()
    if "unguided" in lo:
        return "unguided"
    if "bdg" in lo:
        if "0.5" in lo:
            return "bdg_e4t0.5"
        if re.search(r"\{?=\}?\s*1(\.0)?(?![.\d])", label):
            return "bdg_e4t1"
        return None
    for k, a in (("plug", "plug"), ("tmpd", "tmpd"), ("lgd", "lgd_mc"), ("tfg", "tfg")):
        if k in lo:
            return a
    return None


def m1_key(h):
    lo = h.lower()
    if re.match(r"\s*ib\b", lo):
        return "ib_dec"
    for k, key in (("mol", "mol_stability"), ("valid", "validity"), ("uniq", "uniqueness_of_valid"),
                   ("dv", "unique_valid_per_sample"), ("nfe", "nfe"), ("s/sample", "time"),
                   ("time", "time")):
        if k in lo:
            return key
    return None


def audit_live(F, R, TL, fm_params=None):
    """Compare every number in the live tables with the recomputation.
    Each entry: (table, row, column, printed, recomputed, status, note)."""
    out, skipped = [], []
    # tab:fmvd -- one row per generator, mean (+ sd), NFE, time per sample
    t = TL.get("tab:fmvd")
    if t:
        for row in t["rows"]:
            lo = row[0].lower()
            be = "equifm" if "equifm" in lo else "edm" if "edm" in lo else "fm" if ("ours" in lo or "flow" in lo or "fm" in lo) else None
            if be is None:
                skipped.append("tab:fmvd row %r: no generator recognised" % row[0])
                continue
            m = re.search(r"\(\$?([\d.]+)\$?\s*\\?,?\s*M\)", row[0])
            if m and be == "fm" and fm_params:
                out.append(("tab:fmvd", row[0], "parameter count in the label", m.group(1) + " M",
                            "%.2f M (%d values in the EMA state dict of `weights/fm_ema.pt`)" % (fm_params / 1e6, fm_params),
                            "ok" if close(fm_params / 1e6, m.group(1)) else "MISMATCH", "counted from the checkpoint"))
            elif m:
                skipped.append("the parameter count %s M in the `%s` row label (no local checkpoint of this generator "
                               "is counted; check it against the release)" % (m.group(1), be))
            for h, c in zip(t["header"][1:], row[1:]):
                key, nums = m1_key(h), cell_nums(c)
                if not key or not nums:
                    continue
                if key == "nfe":
                    ok = any(close(x, nums[0]) for x in R[be]["nfe"])
                    out.append(("tab:fmvd", row[0], h, nums[0], "/".join("%g" % x for x in R[be]["nfe"]),
                                "ok" if ok else "MISMATCH", ""))
                    continue
                if key == "time":
                    cands = {"%s cells" % p: mean(F.h[be][(p, "unguided")]["s1k"]) / 1000 for p in PROPS}
                    cands["all 9 cells"] = mean(s["s1k"] for s in R[be]["seed"]) / 1000
                else:
                    cands = {"%s cells" % p: F.h[be][(p, "unguided")][key] for p in PROPS}
                    cands["mean over properties"] = mean(s[key] for s in R[be]["seed"])
                hit = [k for k, v in cands.items() if close(v, nums[0])]
                out.append(("tab:fmvd", row[0], h, nums[0], "%.4f" % cands.get("mu cells", cands["mu cells"]),
                            "ok" if hit else "MISMATCH", ("matches the %s" % hit[0]) if hit else ""))
                if len(nums) > 1 and key != "time":
                    sds = {"%s cells" % p: sdev(r[key] for r in F.M1[(be, "v3", 1)][(p, "unguided")]) for p in PROPS}
                    shit = [k for k, v in sds.items() if close(v, nums[1])]
                    out.append(("tab:fmvd", row[0], h + " (sd)", nums[1], "%.4f" % sds["mu cells"],
                                "ok" if shit else "MISMATCH", ("seed sd of the %s" % shit[0]) if shit else ""))
    # tab:guidance / tab:recent -- mu/alpha/gap triples on our FM at w = 1
    for lab in ("tab:guidance", "tab:recent"):
        t = TL.get(lab)
        if not t:
            continue
        cap = t["caption"].lower()
        if "our flow" not in cap and "our frozen flow" not in cap:
            skipped.append("%s: caption does not say it is our FM; audited as fm anyway" % lab)
        for row in t["rows"]:
            arm = m1_arm_of(row[0])
            if arm is None:
                skipped.append("%s row %r: no arm recognised" % (lab, row[0]))
                continue
            for h, c in zip(t["header"][1:], row[1:]):
                key, nums = m1_key(h), cell_nums(c)
                if key is None or len(nums) != 3:
                    if nums:
                        skipped.append("%s %r / %r: expected a mu/alpha/gap triple" % (lab, row[0], h))
                    continue
                for p, tok in zip(PROPS, nums):
                    v = F.h["fm"][(p, arm)][key]
                    out.append((lab, row[0], "%s %s" % (h, p), tok, "%.4f" % v,
                                "ok" if close(v, tok) else "MISMATCH", ""))
    # tab:m2 -- GC, IB at w = 1 and 4, JSD, diversity
    t = TL.get("tab:m2")
    if t:
        prop = "gc" if "gc" in t["caption"].lower() and "cpg" not in t["caption"].lower() else None
        shown = set()
        cov_ws, fid_cols = set(), []   # strengths the coverage columns name; fidelity columns without a w
        for row in t["rows"]:
            lo = row[0].lower()
            nums_all = [cell_nums(c) for c in row[1:]]
            if any(k in lo for k in ("dirichlet", "fisher", "mog")):
                out.append(("tab:m2", row[0], "all", "-" if not any(nums_all) else "numbers", "not run",
                            "ok" if not any(nums_all) else "MISMATCH", "no cells exist for this model"))
                continue
            if "bdg" in lo:
                a = m1_arm_of(row[0])   # tau_mult 0.5 / 1 in the label, or None
                arms = [a] if a else ["bdg_e4t0.5", "bdg_e4t1"]
            else:
                arms = (["unguided"] if "unguided" in lo else ["plug"] if "plug" in lo else
                        ["tfg_mc"] if "tfg" in lo else [])
            if not arms or prop is None:
                skipped.append("tab:m2 row %r: arm or property not recognised" % row[0])
                continue
            for arm in arms:
                res = []
                for h, c, nums in zip(t["header"][1:], row[1:], nums_all):
                    if not nums:
                        continue
                    hl = h.lower()
                    m = re.search(r"w\{?=\}?\s*(\d+)", h)
                    ws = [float(m.group(1))] if m else [1.0, 4.0]
                    key = "ib" if hl.strip().startswith("ib") else "js" if "jsd" in hl else "div" if "divers" in hl else None
                    if key is None:
                        continue
                    if key == "ib" and m:
                        cov_ws.add(float(m.group(1)))
                    if key != "ib" and not m and h not in fid_cols:
                        fid_cols.append(h)
                    hits = [w for w in ws if close(F.m2s(prop, arm, w)[key], nums[0])]
                    res.append((h, nums[0], "/".join("%.5f" % F.m2s(prop, arm, w)[key] for w in ws),
                                "ok" if hits else "MISMATCH",
                                ("matches w = %s" % "/".join("%g" % w for w in hits)) if len(ws) > 1 and hits else "",
                                "\\mathbf" in c))
                if len(arms) > 1 and any(r[3] == "MISMATCH" for r in res):
                    continue
                shown.add(arm)
                for h, tok, mine, st, note, bold in res:
                    out.append(("tab:m2", "%s [= `%s`]" % (row[0], arm), h, tok, mine, st,
                                "; ".join(x for x in (note, "printed in bold" if bold else "") if x)))
                if len(arms) > 1:
                    break
        # `if prop` for the same reason the block below it guards: when no M2
        # property matched, prop is None and F.m2s(None, ...) raises KeyError
        # rather than reporting anything. The guard existed two lines down but
        # not here, so a run with this M2 state aborted the whole document.
        for other in BDG_HEAD if prop else ():
            if other not in shown:
                out.append(("tab:m2", "Simplex FM + BDG", "--", "--", "--", "NOTE",
                            "`%s` (IB %.4f at w = 1, %.4f at w = 4) has no row, which rule 'both headline BDG arms, always' forbids"
                            % (other, F.m2s(prop, other, 1)["ib"], F.m2s(prop, other, 4)["ib"])))
        if prop and len(cov_ws) > 1 and fid_cols:
            b1, b4, u4 = F.m2s(prop, "bdg_e4t0.5", 1), F.m2s(prop, "bdg_e4t0.5", 4), F.m2s(prop, "unguided", 4)
            out.append(("tab:m2", "all rows", ", ".join(c.replace("$\\downarrow$", "").replace("$\\uparrow$", "") for c in fid_cols),
                        "--", "--", "NOTE",
                        "coverage is shown at w = %s but these fidelity columns at one strength (the cells match w = 1); "
                        "at w = 4 `bdg_e4t0.5`'s k-mer JS is %.5f (%.2fx unguided's %.5f, against %.5f at w = 1) and its decode "
                        "confidence %.4f (unguided %.4f, %.4f at w = 1), so the w = 4 coverage gain sits beside the w = 1 "
                        "fidelity; give the fidelity columns per w, or keep the table at one strength"
                        % (" and ".join("%g" % w for w in sorted(cov_ws)), b4["js"], b4["js"] / u4["js"], u4["js"], b1["js"],
                           b4["conf"], u4["conf"], b1["conf"])))
    return out, skipped


WORDNUM = {"no": 0, "none": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
           "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def _num(tok, total=None):
    """'4', 'four', or 'all' (= total)."""
    tok = tok.lower()
    if tok == "all":
        return total
    return int(tok) if tok.isdigit() else WORDNUM.get(tok)


def claim_checks(F, X, live_src):
    """Numbers the live prose (and text tables) state, recomputed.
    Each: (quote, claim, recomputed, status, where). LIVE['unmatched'] lists the
    checks that found no sentence, so a reworded claim does not vanish silently."""
    txt = re.sub(r"\s+", " ", live_src)
    out, tried = [], []
    secs = [(m.start(), m.group(1)) for m in re.finditer(r"\\(?:sub)*section\*?\{([^{}]*)\}", txt)]
    ab = re.search(r"\\begin\{abstract\}.*?\\end\{abstract\}", txt)
    app = txt.find("\\appendix")
    tabs = []
    for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", txt):
        lab = re.search(r"\\label\{(tab:[^}]*)\}", m.group(0))
        tabs.append((m.start(), m.end(), lab.group(1) if lab else "a table"))

    def where(i):
        if ab and ab.start() <= i < ab.end():
            return "abstract"
        for a, b, lab in tabs:
            if a <= i < b:
                return "`%s`" % lab
        prev = [t for p, t in secs if p <= i]
        return ("sec. \"%s\"" % prev[-1] if prev else "front matter") + (" (appendix)" if 0 <= app <= i else "")

    def sentence(i, j):
        """The sentence holding txt[i:j]; a period inside a number does not end it."""
        ends = [m.end() for m in re.finditer(r"[.!?](?=\s+[A-Z\\])", txt[:i])]
        a = ends[-1] if ends else 0
        m = re.search(r"[.!?](?=\s+[A-Z\\]|\s*$)", txt[j:])
        b = j + m.end() if m else len(txt)
        return txt[a:b].strip()

    def add(m, claim, mine, st, quote=None):
        out.append((quote if quote is not None else m.group(0), claim, mine, st, where(m.start())))

    def each(name, pat):
        ms = list(re.finditer(pat, txt))
        tried.append((name, bool(ms)))
        return ms

    m2s = F.m2s
    # ---- M1 headline: plug against unguided, decoded, 9 cells
    cl = [(be, p) for be in ("fm", "equifm", "edm") for p in PROPS
          if F.zdec(F.h[be][(p, "plug")], F.h[be][(p, "unguided")]) >= Z_BAR]
    for m in each("plug clears the bar in k of 9",
                  r"clears (?:\$z\{=\}2\.99\$ )?(?:on decoded coverage )?in (?:only )?(\d+) of (?:those )?(\d+)"):
        add(m, "%s of %s" % m.groups(), "%d of 9 (%s)" % (len(cl), ", ".join("%s/%s" % c for c in cl)),
            "ok" if int(m.group(1)) == len(cl) else "MISMATCH")
    up = [(be, p) for be in ("fm", "equifm", "edm") for p in PROPS
          if F.h[be][(p, "plug")]["ib_dec"] > F.h[be][(p, "unguided")]["ib_dec"]]
    for m in each("guidance raises decoded coverage in k of 9", r"raises decoded coverage in (\d+) of (\d+)"):
        add(m, "%s of %s" % m.groups(), "plug-in above unguided (any size, decoded) in %d of 9; not in %s"
            % (len(up), ", ".join("%s/%s" % (be, p) for be in ("fm", "equifm", "edm") for p in PROPS if (be, p) not in up)),
            "ok" if int(m.group(1)) == len(up) else "MISMATCH")
    for m in each("stability cost of plug", r"(?:[Tt]he cost is|at a cost of) \$?([\d.]+)\$? points of molecule stabili"):
        v = -100 * mean(F.h[be][(p, "plug")]["mol_stability"] - F.h[be][(p, "unguided")]["mol_stability"]
                        for be in ("fm", "equifm") for p in PROPS)
        add(m, m.group(1), "%.2f (mean over fm and EquiFM, 6 cells)" % v, "ok" if close(v, m.group(1)) else "MISMATCH")
    # ---- run cost: every occurrence, with its stage split where the sentence gives one
    g, h2 = X["gpuh"], X["m2_hours"]
    tot = g["v3"] + g["v3abl"]
    unit = ("summed per-cell wall-clock `seconds` including scoring, cells run concurrently on a shared box "
            "(caveat 11), so **not a GPU-hour measurement**; **M1 only**: M2's %d cells add %.2f h of recorded "
            "`minutes` (m2 %.2f, m2abl %.2f, m2wsweep %.2f) on an unrecorded device"
            % (X["m2_ncells"], sum(h2.values()), h2["m2"], h2["m2abl"], h2["m2wsweep"]))
    for m in each("run cost in GPU-hours", r"\$([\d.]+)\$ GPU-hours"):
        s = sentence(m.start(), m.end())
        ms = re.search(r"\$([\d.]+)\$ for the headline stage and \$([\d.]+)\$ for the larger ablation", s)
        st = []
        if close(tot, m.group(1)):
            st.append("total ok")
        elif abs(round(g["v3"], 1) + round(g["v3abl"], 1) - float(m.group(1))) < 1e-9:
            st.append("MISMATCH at the printed digit: the exact total %.2f rounds to %.1f; %s is the sum of the rounded stages"
                      % (tot, tot, m.group(1)))
        else:
            st.append("MISMATCH")
        if ms:
            st.append("stages %s" % ("ok" if close(g["v3"], ms.group(1)) and close(g["v3abl"], ms.group(2)) else "MISMATCH"))
        st.append("unit wrong: summed cell wall-clock, not GPU-hours")
        out.append((s, m.group(1) + (" = %s + %s" % ms.groups() if ms else ""),
                    "%.2f h = headline %.2f + ablation %.2f; %s" % (tot, g["v3"], g["v3abl"], unit),
                    "; ".join(st), where(m.start())))
    for m in each("smallest resolvable difference (M1)", r"smallest resolvable difference in coverage is \$?([\d.]+)\$? points"):
        v = 100 * Z_BAR * math.sqrt(2 * 0.09 * 0.91 / 6000)
        add(m, m.group(1), "%.2f (at p = 0.09, the protocol's power-table p)" % v,
            "ok" if abs(v - float(m.group(1))) < 0.011 else "MISMATCH")
    # ---- M2
    p0 = m2s("gc", "unguided", 1)["ib"]
    for m in each("M2 resolvable difference", r"(?:bar resolves|below the) \$?([\d.]+)\$? (?:points|this \$n\$ resolves)"):
        v = 100 * mdd(p0, 6000)
        add(m, m.group(1), "%.2f (at gc unguided p = %.4f; unpaired, not pre-registered for M2)" % (v, p0),
            "ok" if abs(v - float(m.group(1))) < 0.011 else "MISMATCH")

    def vs_plug(prop, arm, w):
        s, pl = m2s(prop, arm, w), m2s(prop, "plug", w)
        return 100 * (s["ib"] - pl["ib"]), zun(s["ib"], pl["ib"], 6000)

    def both_arms(w):
        return "; ".join("`%s`: GC %+.2f (z %+.2f), CpG %+.2f (z %+.2f)" % ((a,) + vs_plug("gc", a, w) + vs_plug("cpg", a, w))
                         for a in BDG_HEAD)
    sh = X["m2_share"]["r"]
    clip = lambda arm, w: "/".join("%.2f" % (100 * m2s(pr, arm, w)["clip_frac"]) for pr in M2_PROPS)  # noqa: E731
    for m in each("M2 BDG gain over plug at w = 1",
                  r"gain over plug-in is \$([\d.]+)\$ points(?: on GC)?(?:, below the \$[\d.]+\$ this \$n\$ resolves, but \$([\d.]+)\$ on CpG)?"):
        a, b = vs_plug("gc", "bdg_e4t0.5", 1)[0], vs_plug("cpg", "bdg_e4t0.5", 1)[0]
        ok = close(a, m.group(1)) and (m.group(2) is None or close(b, m.group(2)))
        add(m, m.group(1) + (" / %s" % m.group(2) if m.group(2) else ""),
            "w = 1, unpaired z, not pre-registered for M2: %s. The sentence names one arm; `bdg_e4t1` is not mentioned. Force: `bdg_e4t0.5` applies %.1fx plug's correction share on GC at w = 1 (`bdg_e4t1` %.2fx); CpG is not measured"
            % (both_arms(1), sh[("bdg_e4t0.5", 1)], sh[("bdg_e4t1", 1)]) +
            ("" if m.group(2) else "; the CpG figure (%+.2f) is omitted" % b),
            ("ok, one BDG arm only (rule 3)" if ok else "MISMATCH"), quote=sentence(m.start(), m.end()))
    for m in each("M2 BDG gain over plug at w = 4",
                  r"at \$w\{=\}4\$ it is \$([\d.]+)\$(?: points)? on GC and \$([\d.]+)\$ on CpG"):
        a, b = vs_plug("gc", "bdg_e4t0.5", 4)[0], vs_plug("cpg", "bdg_e4t0.5", 4)[0]
        add(m, "%s / %s" % m.groups(),
            "w = 4: %s. Confounds the sentence omits: `bdg_e4t0.5` applies %.1fx plug's correction share on GC (CpG not measured) and clips %s %% of guided sample-steps (GC/CpG) against plug's %s %%"
            % (both_arms(4), sh[("bdg_e4t0.5", 4)], clip("bdg_e4t0.5", 4), clip("plug", 4)),
            "ok, one BDG arm only, force and clip confounds missing" if close(a, m.group(1)) and close(b, m.group(2)) else "MISMATCH")
    u = m2s("gc", "unguided", 1)["ib"]
    for m in each("M2 eta sweep", r"moves coverage from \$\+([\d.]+)\$ to \$\+([\d.]+)\$ points"):
        a = 100 * (m2s("gc", "bdg_e0t1", 4, "m2abl")["ib"] - u)
        b = 100 * (m2s("gc", "bdg_e8t0.5", 4, "m2abl")["ib"] - u)
        add(m, "+%s / +%s" % m.groups(), "%+.2f / %+.2f (w = 4, against unguided; `bdg_e8t0.5` clips %.1f %%)"
            % (a, b, 100 * m2s("gc", "bdg_e8t0.5", 4, "m2abl")["clip_frac"]),
            "ok" if close(a, m.group(1)) and close(b, m.group(2)) else "MISMATCH")
    for m in each("M2 inverted setpoint", r"reverses the sign to \$[-−]([\d.]+)\$"):
        v = 100 * (m2s("gc", "bdg_e8t1.5", 4, "m2abl")["ib"] - u)
        add(m, "-" + m.group(1), "%+.2f" % v, "ok" if close(-v, m.group(1)) else "MISMATCH")
    win = X["m2win"]["plug_t0"] - X["m2win"]["plug_t05"]
    best = X["m2win"]["best_in"] - X["m2win"]["ung"]
    bdgp = m2s("gc", "bdg_e4t0.5", 4)["ib"] - m2s("gc", "plug", 4)["ib"]
    for m in each("M2 window gain", r"buys plug-in \$([\d.]+)\$ points"):
        add(m, m.group(1), "%.2f" % (100 * win), "ok" if close(100 * win, m.group(1)) else "MISMATCH")
    for m in each("M2 window ratio, 'roughly ten times'", r"roughly ten times"):
        add(m, "~10x", "%.1fx the best in-window result (+%.2f pp vs unguided, `bdg_e8t0.5` w = 4); %.1fx only against BDG's gain over plug at w = 4 (+%.2f pp)"
            % (win / best, 100 * best, win / bdgp, 100 * bdgp),
            "MISMATCH: the recorded correction is ~5x (SCOPE status doc, M2_WINDOW_DOMINATES.md)",
            quote=sentence(m.start(), m.end()))
    for m in each("M2 window ratio, 'about five times'", r"about five times"):
        add(m, "~5x", "%.1fx the best in-window result (+%.2f pp, `bdg_e8t0.5` w = 4)" % (win / best, 100 * best),
            "ok" if 4.5 <= win / best < 5.5 else "MISMATCH", quote=sentence(m.start(), m.end()))
    # ---- the ablation grid (prose and the text table)
    c = X["grid_counts"]
    for m in each("expanding row: falls in k of 12, past the bar in j",
                  r"(?:lowers it in|IB falls in) (\d+) of 12(?:,? (every one past the bar|and clears the bar in (\d+)|(\d+) past the bar))?"):
        falls = int(m.group(1))
        past = 12 if m.group(2) == "every one past the bar" else (int(m.group(3) or m.group(4)) if m.group(2) else None)
        mine = ("e8t1.5 below eta = 0 (decoded): %d of 12; past 2.99 %d (continuous %d); past the grid's selection-adjusted %.2f %d"
                % (c["lo_neg"], c["lo_sig"], c["lo_sig_c"], Z_SEL, c["lo_sel"]))
        if falls != c["lo_neg"]:
            st = "MISMATCH"
        elif past is None:
            st = "ok"
        elif past == c["lo_sig"]:
            st = "ok at 2.99, which is not selection-adjusted; %d at %.2f, the bar ABLATION_V3_PROTOCOL.md 4 requires beside a grid claim" % (c["lo_sel"], Z_SEL)
        elif past == c["lo_sig_c"]:
            st = "ok on continuous only"
        else:
            st = "MISMATCH"
        add(m, m.group(0), mine, st)
    for m in each("contracting row: rises in k of 12, j at w = 1",
                  r"(?:raises coverage in|IB rises in) (\d+) of 12(?: grids)?,? (all|\w+)(?: of them)? at \$w\{=\}1\$"):
        k, j = int(m.group(1)), _num(m.group(2), int(m.group(1)))
        mine = "decoded %d of 12 (w = 1: %d); continuous %d of 12 (w = 1: %d)" % (c["mono_d"], c["mono_d_w1"], c["mono_c"], c["mono_c_w1"])
        st = ("ok" if (k, j) == (c["mono_d"], c["mono_d_w1"]) else
              "ok on continuous only" if (k, j) == (c["mono_c"], c["mono_c_w1"]) else "MISMATCH")
        add(m, "%d of 12, %s at w = 1" % (k, j), mine, st)
    for m in each("Spearman range", r"Spearman \$0\.92\$(?: to |--)\$?0\.99\$?"):
        add(m, "0.92 to 0.99", "decoded %.2f to %.2f" % (c["rho_min"], c["rho_max"]),
            "ok" if "%.2f" % c["rho_min"] == "0.92" and "%.2f" % c["rho_max"] == "0.99" else "MISMATCH")
    for m in each("eta = 0 identity", r"bit-exact on ours; \$4\\times10\^\{-3\}\$ on EquiFM"):
        v = max(X["eta0"][("equifm", p)]["ib"] for p in PROPS)
        add(m, "4e-3", "%.1e (continuous; decoded %.1e)" % (v, max(X["eta0"][("equifm", p)]["ib_dec"] for p in PROPS)),
            "ok" if abs(v - 4e-3) < 5e-4 else "MISMATCH")
    # ---- M2 fidelity
    for m in each("M2 pairwise distinctness", r"Pairwise distinctness is \$([\d.]+)\$ for every arm"):
        ds = [m2s(pr, a, w)["div"] for pr in M2_PROPS for a in M2_ARMS for w in (1, 4)]
        add(m, m.group(1), "%.4f to %.4f over the M2 headline cells" % (min(ds), max(ds)),
            "ok" if all(close(d, m.group(1)) for d in ds) else "ok to 3 decimals")
    for m in each("diversity ceiling", r"(?:random-sequence|diversity) ceiling is \$?([\d.]+)\$?"):
        v = 0.75 * (1 - 1 / 256.0)
        add(m, m.group(1), "%.4f = 3/4 x 255/256 (the diversity code includes the 256 self-pairs)" % v,
            "ok" if close(v, m.group(1)) else "MISMATCH")
    # ---- BDG against plug, and TFG, at the headline
    vb = X["bdg_vs_plug_dec"]
    for m in each("BDG vs plug: k contrasts, gains / losses / ties",
                  r"(twelve|eighteen|\d+) contrasts[^.;]{0,90}? give (no|\w+) gains?(?: at the bar)?, (\w+) loss(?:es)? and (\w+) ties"):
        k, a, b, t = (_num(g, None) for g in m.groups())
        add(m, "%s: %s / %s / %s" % m.groups(),
            "v3-final, unpaired decoded at 2.99, fm and EquiFM x 2 arms x 3 props: %d contrasts, %d above / %d below / %d tie; all %d losses are `bdg_e4t1` (section 9), and EquiFM's carry caveat 5"
            % (sum(vb), vb[0], vb[1], vb[2], vb[1]),
            "ok" if (k, a, b, t) == (sum(vb), vb[0], vb[1], vb[2]) else "MISMATCH")
    lead = [(be, p) for be in ("fm", "equifm") for p in PROPS
            if max(GUIDED + ("unguided",), key=lambda a: F.h[be][(p, a)]["ib_dec"]) == "tfg"]
    for m in each("TFG leads coverage in k of 6, spending a to b stability points",
                  r"TFG leads coverage in (\d+) of (\d+) cells[^.]{0,60}?spending (\d+) to (\d+) points of molecule stability"):
        dp = [100 * (F.h[be][(p, "plug")]["mol_stability"] - F.h[be][(p, "tfg")]["mol_stability"]) for be in ("fm", "equifm") for p in PROPS]
        du = [100 * (F.h[be][(p, "unguided")]["mol_stability"] - F.h[be][(p, "tfg")]["mol_stability"]) for be in ("fm", "equifm") for p in PROPS]
        ok = int(m.group(1)) == len(lead) and round(min(dp)) == int(m.group(3)) and round(max(dp)) == int(m.group(4))
        add(m, "%s of %s; %s to %s" % m.groups(),
            "highest decoded IB of the 7 arms in %d of 6 (not in %s); stability cost %.1f to %.1f pp against plug-in, %.1f to %.1f against unguided"
            % (len(lead), ", ".join("%s/%s" % c for c in [(be, p) for be in ("fm", "equifm") for p in PROPS] if c not in lead) or "none",
               min(dp), max(dp), min(du), max(du)),
            "ok (stability read against plug-in)" if ok else "MISMATCH", quote=sentence(m.start(), m.end()))
    for m in each("stability cost in words", r"cost of (four|three|five) points of molecule stability"):
        v = -100 * mean(F.h[be][(p, "plug")]["mol_stability"] - F.h[be][(p, "unguided")]["mol_stability"]
                        for be in ("fm", "equifm") for p in PROPS)
        add(m, m.group(1), "%.2f (mean over fm and EquiFM, 6 cells)" % v, "ok" if round(v) == _num(m.group(1)) else "MISMATCH")
    # the expanding setpoint's spread: R of e8t1.5 at w = 1, against plug-in at w = 4
    Rx = {(be, p): F.a[(be, 1)][(p, "bdg_e8t1.5")]["sd_dec"] / F.h[be][(p, "unguided")]["sd_dec"] for be in ("fm", "equifm") for p in PROPS}
    R4 = {(be, p): F.a[(be, 4)][(p, "bdg_e0t1")]["sd_dec"] / F.h[be][(p, "unguided")]["sd_dec"] for be in ("fm", "equifm") for p in PROPS}
    for m in each("expanding setpoint R range, k of 6 cells",
                  r"\$R\$ \$([\d.]+)\$ to \$([\d.]+)\$, in (\d+) of (\d+) cells"):
        k = sum(v > 1 for v in Rx.values())
        ok = close(min(Rx.values()), m.group(1)) and close(max(Rx.values()), m.group(2)) and int(m.group(3)) == k
        add(m, "R %s to %s, %s of %s" % m.groups(),
            "`bdg_e8t1.5` at w = 1: decoded R %.2f to %.2f, above 1 in %d of 6 (fm, EquiFM x 3 props); plug-in at w = 4 (`bdg_e0t1`): R %.2f to %.2f, below 1 in %d of 6. Not pre-registered; EquiFM carries caveat 5"
            % (min(Rx.values()), max(Rx.values()), k, min(R4.values()), max(R4.values()), sum(v < 1 for v in R4.values())),
            "ok" if ok else "MISMATCH", quote=sentence(m.start(), m.end()))
    # ---- claims from another run
    for m in each("TFG useful yield",
                  r"TFG leads useful yield in 6 of 6 cells, by \$([\d.]+)\$ to \$([\d.]+)\$ points over plug-in, while losing (\d+) to (\d+) points of molecule stability"):
        dv = [100 * (F.h[be][(p, "tfg")]["mol_stability"] - F.h[be][(p, "unguided")]["mol_stability"])
              for be in ("fm", "equifm") for p in PROPS]
        dp = [100 * (F.h[be][(p, "tfg")]["mol_stability"] - F.h[be][(p, "plug")]["mol_stability"])
              for be in ("fm", "equifm") for p in PROPS]
        add(m, "yield +%s to +%s; stability -%s to -%s" % m.groups(),
            "yield: not computable from v3-final cells (Betty n = 5000 sidecars); blade stability loss vs unguided %.1f to %.1f pp, vs plug %.1f to %.1f pp"
            % (-max(dv), -min(dv), -max(dp), -min(dp)), "NOT v3-FINAL")
    betty = ("v3-final has 12 BDG-vs-plug contrasts (2 arms x 3 props x fm, EquiFM); unpaired decoded: %d above / %d below / %d tie. "
             "The 18 are the Betty n = 5000 run's, whose arm set still held `bdg_e4t0.75` and whose fm cells were guided by "
             "TFG's f_A **and scored by TFG's oracle**" % tuple(X["bdg_vs_plug_dec"]))
    for name, pat in (("18 paired contrasts (abstract form)", r"eighteen paired contrasts give two gains, five losses and eleven ties"),
                      ("18 paired contrasts (body form)", r"paired useful-yield reading over 18 contrasts is 2 above, 5 below and 11 ties")):
        for m in each(name, pat):
            add(m, "18 paired contrasts: 2 / 5 / 11", betty, "NOT v3-FINAL")
    for m in each("useful yield, any sentence", r"useful[- ]yield"):
        s = sentence(m.start(), m.end())
        if "TFG leads useful yield" in s or "paired useful-yield reading over 18" in s or any(s == o[0] for o in out):
            continue
        out.append((s, "useful yield", "not computable from v3-final cells: the blade sidecars stay on blade (rule 11); the only paired reading is Betty n = 5000",
                    "NOT v3-FINAL", where(m.start())))
    # the second evaluator: verdict changes over the v3-final contrasts
    flips, ncon = 0, 0
    for be in ("fm", "equifm", "edm"):
        H = F.h[be]
        for p in PROPS:
            pairs = [(a, "unguided") for a in GUIDED if (p, a) in H] + [(a, "plug") for a in BDG_HEAD if (p, a) in H]
            for a, b in pairs:
                ncon += 1
                s, r = H[(p, a)], H[(p, b)]
                flips += verdict(zun(s["ib_dec"], r["ib_dec"], s["N"])) != verdict(zun(s["o2_dec"], r["o2_dec"], s["N"]))
    for m in each("second evaluator changes k of N verdicts", r"changes (\d+) of (\d+) (?:paired )?verdicts"):
        ok = (int(m.group(1)), int(m.group(2))) == (flips, ncon)
        add(m, "%s of %s" % m.groups(), "v3-final: the second oracle (decoded, unpaired, 2.99) changes %d of the %d headline contrasts (every guided arm vs unguided and each BDG arm vs plug, fm, EquiFM, EDM); %s of %s is not a count these cells give, and its source run is not stated"
            % (flips, ncon, m.group(1), m.group(2)), "ok" if ok else "not reproduced from v3-final cells",
            quote=sentence(m.start(), m.end()))
    LIVE["unmatched"] = [n for n, hit in tried if not hit]
    return out


def md_table_of(T, lab):
    """The paper's table reproduced in markdown, P for a pending cell."""
    t = T[lab]
    clean = lambda c: (c.replace("\\pend", "**P**").replace("$\\uparrow$", " (up)")  # noqa: E731
                       .replace("$\\downarrow$", " (down)").replace("|", "/"))
    L = ["| " + " | ".join(clean(c) or "(row label)" for c in t["header"]) + " |",
         "|" + "---|" * len(t["header"])]
    for r in t["rows"]:
        L.append("| " + " | ".join(clean(c) for c in r) + " |")
    return L


# ------------------------------------------------------- tab:training sources
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def training_sources(M1):
    import torch  # noqa: WPS433 (only for checkpoint metadata)
    out = {"md5": {}, "args": {}}
    files = ["weights/fm_ema.pt"] + ["weights/f_%s_%s.pt" % (ab, p) for ab in "AB" for p in PROPS] \
        + ["proj1/m2/blade_bundle/fm_m2_dfb500.pt"]
    for f in files:
        path = os.path.join(ROOT, f)
        if not os.path.isfile(path):
            die("tab:training source %s is missing" % f)
        out["md5"][f] = md5(path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if f == "weights/fm_ema.pt":
            if not hasattr(ck.get("ema"), "items"):
                die("weights/fm_ema.pt has no 'ema' state dict to count parameters from")
            out["fm_params"] = sum(v.numel() for v in ck["ema"].values() if hasattr(v, "numel"))
        if "args" in ck:
            out["args"][f] = dict(ck["args"], **{k: ck[k] for k in ("epoch",) if k in ck})
        else:
            out["args"][f] = {k: v for k, v in ck.items()
                              if isinstance(v, (int, float, str)) and k != "state_dict"}
    gen = {r["prov"]["gen_md5"] for rs in M1[("fm", "v3", 1)].values() for r in rs}
    out["fm_cell_md5"] = sorted(gen)
    devs = set()
    for key, c in M1.items():
        for rs in c.values():
            for r in rs:
                devs.add((r.get("prov") or {}).get("device"))
    out["devices"] = sorted(str(d) for d in devs)
    with open(os.path.join(ROOT, "proj1", "m2", "simplex_fm.py"), encoding="utf-8") as fh:
        sfm = fh.read()
    out["m2_adam_line"] = next((i + 1 for i, ln in enumerate(sfm.split(NL))
                                if "torch.optim.Adam(" in ln), None)
    out["m2_has_ema"] = bool(re.search(r"\bema\b", sfm, re.I))
    with open(os.path.join(ROOT, "proj1", "scripts", "train_fm.py"), encoding="utf-8") as fh:
        out["fm_adam_line"] = next((i + 1 for i, ln in enumerate(fh.read().split(NL))
                                    if "torch.optim.Adam(" in ln), None)
    tl = os.path.join(ROOT, "proj1", "m2", "blade_bundle", "train.log")
    with open(tl, encoding="utf-8", errors="replace") as fh:
        out["m2_train_device"] = fh.readline().strip()
    return out


# ---------------------------------------------------- V3_RESULTS* cross-check
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def ndec(tok):
    return len(tok.split(".")[1]) if "." in tok else 0


def close(mine, tok):
    return abs(mine - float(tok)) <= 0.5 * 10 ** (-ndec(tok)) + 1e-9


def crosscheck_docs(M1):
    files = sorted(glob.glob(os.path.join(ROOT, "docs", "results", "V3_RESULTS*.md")))
    rep = {"files": [], "n": 0, "bad": [], "weff_first": 0, "weff_first_bad": 0,
           "weff_mean_bad": [], "bar_diff": [], "gpuh_bad": [], "sd_digit": 0, "sd_rel": 0.0}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            txt = fh.read()
        rel = os.path.relpath(f, ROOT).replace(os.sep, "/")
        title = txt.split(NL, 1)[0]
        if "ablation" in title:
            stage = "v3abl"
            w = int(re.search(r"at w = (\d+)", title).group(1))
        else:
            stage, w = "v3", 1
        rep["files"].append(rel)
        be = prop = mode = None
        for ln in txt.split(NL):
            m = re.match(r"## Base model: .*\(`--backend (\w+)`\)", ln)
            if m:
                be, prop, mode = m.group(1), None, None
                continue
            m = re.match(r"### (mu|alpha|gap) \(delta", ln)
            if m:
                prop, mode = m.group(1), "metric"
                continue
            if ln.startswith("### BDG controller state"):
                prop, mode = None, "ctrl"
                continue
            if ln.startswith("### Cost and clip"):
                prop, mode = None, "cost"
                continue
            if ln.startswith("| arm vs unguided"):
                mode = "vs"
                continue
            m = re.match(r"\| measured cost \| ([\d.]+) GPU-h over (\d+) cells", ln)
            if m and be:
                cells = M1[(be, stage, w)]
                secs = sum(r["seconds"] for rs in cells.values() for r in rs)
                ncell = sum(len(rs) for rs in cells.values())
                rep["n"] += 2
                if not close(secs / 3600.0, m.group(1)) or int(m.group(2)) != ncell:
                    rep["gpuh_bad"].append("%s %s: doc %s GPU-h / %s cells, cells %.2f / %d"
                                           % (rel, be, m.group(1), m.group(2), secs / 3600.0, ncell))
                continue
            m = re.match(r"\| `([^`]+)` \|(.*)", ln)
            if not (m and be):
                continue
            arm, rest = m.group(1), m.group(2)
            cells = M1[(be, stage, w)]
            toks = NUM.findall(rest)
            if mode == "metric" and prop:
                s = m1_stats(cells[(prop, arm)])
                d = s["delta"]
                mine = [s["ib"], s["se_ib"], s["ib_dec"], s["prop_mae_eval"] / d, s["bias"] / d,
                        s["sd"] / d, s["mol_stability"], s["atom_stability"], s["validity"],
                        s["uniqueness_of_valid"], s["uniq_min_cell"], s["unique_valid_per_sample"],
                        s["diversity_mean_pairwise"], s["guide_eval_gap_mean"] / d]
                names = ["in_band", "se", "dec", "MAE/d", "bias/d", "sd/d", "mol_stab",
                         "atom_stab", "valid", "uniq", "uniq_min", "yield", "diversity", "gap/d"]
                if len(toks) != len(mine):
                    rep["bad"].append("%s %s/%s/%s: could not parse the row" % (rel, be, prop, arm))
                    continue
                for nm, a, t in zip(names, mine, toks):
                    rep["n"] += 1
                    if not close(a, t):
                        rep["bad"].append("%s %s/%s/%s %s: doc %s, cells %.6f"
                                          % (rel, be, prop, arm, nm, t, a))
                # would the exact pooled sd change the printed digit?
                if not close(s["sd_exact"] / d, toks[5]):
                    rep["sd_digit"] += 1
                rep["sd_rel"] = max(rep["sd_rel"], abs(s["sd_exact"] - s["sd"]) / s["sd"])
            elif mode == "vs" and prop:
                u = m1_stats(cells[(prop, "unguided")])
                s = m1_stats(cells[(prop, arm)])
                mine = [s["ib"] - u["ib"], V.z2(s, u, "in_band_fraction", "se_ib"),
                        s["ib_dec"] - u["ib_dec"], V.z2(s, u, "in_band_fraction_dec", "se_ib_dec"),
                        s["mol_stability"] - u["mol_stability"], s["validity"] - u["validity"]]
                names = ["d_ib", "z", "d_dec", "z_dec", "d_mol", "d_val"]
                for nm, a, t in zip(names, mine, toks):
                    rep["n"] += 1
                    if not close(a, t):
                        rep["bad"].append("%s %s/%s/%s %s: doc %s, cells %.6f"
                                          % (rel, be, prop, arm, nm, t, a))
                # the doc's bar (Bonferroni over the arms on that property at a
                # 3-sigma family rate) against the pre-registered 2.99
                m_tests = len([k for k in cells if k[0] == prop and k[1] != "unguided"])
                zb = V.NORM_Q(1.0 - (2.0 * V.NORM_SF(V.SIGMA)) / (2.0 * m_tests))
                z = mine[1]
                if (abs(z) >= zb) != (abs(z) >= Z_BAR):
                    rep["bar_diff"].append("%s %s/%s/%s: z = %+.2f clears %s but not %s"
                                           % (rel, be, prop, arm, z,
                                              "2.99" if abs(z) >= Z_BAR else "%.2f" % zb,
                                              "%.2f" % zb if abs(z) >= Z_BAR else "2.99"))
            elif mode == "ctrl":
                vals = toks[2:]
                for i, p in enumerate(PROPS):
                    if (p, arm) not in cells:
                        continue
                    rows = cells[(p, arm)]
                    first = rows[0]["diag"]
                    f_first = [first["bdg_w_eff"], math.sqrt(max(first["bdg_w_eff_sq"], 0.0)),
                               first["bdg_w_eff_neg"]]
                    s = m1_stats(rows)
                    f_mean = [s["weff3"], s["weff_rms3"], s["weff_neg3"]]
                    tk = vals[3 * i:3 * i + 3]
                    for a, b, t in zip(f_first, f_mean, tk):
                        rep["weff_first"] += 1
                        rep["n"] += 1
                        if not close(a, t):
                            rep["weff_first_bad"] += 1
                            rep["bad"].append("%s %s/%s/%s controller: doc %s, first seed %.4f"
                                              % (rel, be, p, arm, t, a))
                        if not close(b, t):
                            rep["weff_mean_bad"].append((rel, be, p, arm, t, b))
            elif mode == "cost":
                clips = toks[2:5]
                for p, t in zip(PROPS, clips):
                    if (p, arm) not in cells:
                        continue
                    rep["n"] += 1
                    tot = sum(r["clipped_sample_steps"] for r in cells[(p, arm)])
                    if int(float(t)) != tot:
                        rep["bad"].append("%s %s/%s/%s clipped: doc %s, cells %d"
                                          % (rel, be, p, arm, t, tot))
    return rep


# ------------------------------------------------------------- LaTeX builders
def _sgn(z, bar):
    if z is None or z != z or abs(z) < bar:
        return 0
    return 1 if z > 0 else -1


def marks(z_ung=None, z_plug=None):
    """One superscript, and every mark carries its DIRECTION: dagger+/- is above /
    below unguided, ddagger+/- a BDG arm above / below plug-in, both at the
    pre-registered unpaired |z| >= 2.99. (An unsigned mark put beside an IB level
    read as a win where every M1 BDG-vs-plug mark is a loss.)"""
    parts = []
    s = _sgn(z_ung, Z_BAR)
    if s:
        parts.append(r"\dagger" + ("+" if s > 0 else "-"))
    s = _sgn(z_plug, Z_BAR)
    if s:
        parts.append(r"\ddagger" + ("+" if s > 0 else "-"))
    return "$^{%s}$" % ",".join(parts) if parts else ""


def smark(z):
    """The ablation's own mark: |z| >= Z_SEL against the row's reference arm. The
    cell it sits on is a signed Delta, so the sign is the Delta's."""
    return r"$^{\S}$" if _sgn(z, Z_SEL) else ""


MARK_MEANING = [(r"\dagger+", r"above unguided"), (r"\dagger-", r"below unguided"),
                (r"\ddagger+", r"BDG above plug-in"), (r"\ddagger-", r"BDG below plug-in")]


def legend(body, m2=False):
    """The caption sentence that defines exactly the marks a body carries."""
    have = [(m, t) for m, t in MARK_MEANING if m in body]
    tail = (r"unpaired $|z|\ge2.99$, $N{=}6{,}000$ per arm" +
            (r", not pre-registered for this modality" if m2 else ""))
    if not have:
        return r"No contrast with unguided or plug-in reaches %s." % tail
    return "%s; %s." % (", ".join(r"$^{%s}$ %s" % (m, t) for m, t in have), tail)


# Tighter column separation for the bodies that carry many packed cells; it goes
# inside the float, before \begin{tabular}, so it stays local to the table.
TABSEP = r"\setlength{\tabcolsep}{4pt}"


def nz(x, d=2):
    """A fraction without its leading zero (.36), as the live paper prints them;
    used only in the ablation's triple-packed cells to keep it inside the text width."""
    t = "%.*f" % (d, x)
    return t[1:] if t.startswith("0.") else t


def s1(x):
    """Signed one-decimal for a LaTeX cell: an exact zero prints as 0.0 (the fm
    identity), and a negative value is set in math so it gets a minus sign, not
    a hyphen."""
    if x == 0:
        return "0.0"
    return "+%.1f" % x if x > 0 else "$-%.1f$" % abs(x)


def md_label(tex):
    """A LaTeX row label, readable in markdown."""
    t = tex.replace("$", "").replace("{=}", " = ").replace("=", " = ").replace("  ", " ")
    for a, b in (("\\eta", "eta"), ("\\tau", "tau_mult"), ("w_{\\mathrm{eff}}", "w_eff")):
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


class Fill:
    """Every number the tables need, computed once from the cells."""

    def __init__(self, M1, M2):
        self.M1, self.M2 = M1, M2
        self.h = {be: {k: m1_stats(v) for k, v in M1[(be, "v3", 1)].items()}
                  for be in ("fm", "equifm", "edm")}
        self.a = {(be, w): {k: m1_stats(v) for k, v in M1[(be, "v3abl", w)].items()}
                  for be in ("fm", "equifm") for w in (1, 4)}
        self.m2 = {}
        for k in M2:
            self.m2[k] = m2_stats([M2[k][s] for s in sorted(M2[k])])

    def zdec(self, s, r):
        return zun(s["ib_dec"], r["ib_dec"], s["N"])

    def m2s(self, prop, arm, w, stage="m2", t=0.5):
        if arm == "unguided":
            w = 1.0
        return self.m2[(stage, prop, arm, float(w), float(t))]


def fmvd_rows(F):
    """Unguided base-model chemistry per backend. fm/equifm write the same
    samples into all three property cells of a seed; edm does not (checked)."""
    out = {}
    # `vp` is our OWN QM9 diffusion (28 Sep). It is here because tab:fmvd's
    # "VP, trained here" row is built from it -- it is the matched partner to
    # `fm`, and the only backend in this list that is not either ours-but-flow
    # or borrowed.
    for be in ("fm", "equifm", "edm", "vp"):
        cells = F.M1[(be, "v3", 1)]
        per_seed = []
        spread = 0.0
        for i in range(3):
            rs = [cells[(p, "unguided")][i] for p in PROPS]
            for k in ("mol_stability", "validity", "uniqueness_of_valid"):
                spread = max(spread, max(r[k] for r in rs) - min(r[k] for r in rs))
            per_seed.append({k: mean(r[k] for r in rs) for k in
                             ("mol_stability", "validity", "uniqueness_of_valid",
                              "unique_valid_per_sample")}
                            | {"s1k": mean(r["seconds"] * 1000.0 / r["n"] for r in rs),
                               "nfe": {r["cost"]["gen_fwd"] / (r["n"] / r["batch"]) for r in rs}})
        nfe = set().union(*[s["nfe"] for s in per_seed])
        out[be] = {"seed": per_seed, "spread": spread, "nfe": sorted(nfe)}
    return out


def pm(xs, d=3):
    return ("%." + str(d) + "f $\\pm$ %." + str(d) + "f") % (mean(xs), sdev(xs))


def tex_fmvd(F, R, option):
    L = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Model & Mol. stab.$\uparrow$ & Validity$\uparrow$ & Uniqueness$\uparrow$ & NFE$\downarrow$ & Time (s/1k)$\downarrow$\\",
         r"\midrule"]

    def row(label, be):
        ss = R[be]["seed"]
        return "%s & %s & %s & %s & %d & %s\\\\" % (
            label, pm([s["mol_stability"] for s in ss]), pm([s["validity"] for s in ss]),
            pm([s["uniqueness_of_valid"] for s in ss]), int(R[be]["nfe"][0]),
            pm([s["s1k"] for s in ss], 1))
    L.append(row("FM, trained here", "fm"))
    if option == "A":
        L.append(row("EDM, external release", "edm"))
    else:
        # Our own VP diffusion RAN on 28 Sep (18 cells, md5 8a3390a6), so this
        # emits real numbers where it used to emit "not run".
        L.append(row("VP, trained here", "vp"))
    L += [r"\bottomrule", r"\end{tabular}"]
    return NL.join(L)




# Proposed captions, one per option, built from the bodies so that each defines
# exactly the marks its body carries (bar, test, reference arm, direction).
# Section 12 of the doc measures each one in the compiled paper.
TIME_NOTE = r"time is wall-clock s per 1{,}000 samples including scoring, on a shared machine"


def captions(X, bodies):
    rr = 100 * X["rerun"]["equifm"]["ib"]
    B = {k: v for k, v in bodies.items()}
    return {
        "tab:fmvd": {
            "A": (r"QM9 base models, seed mean $\pm$ sd over three seeds of 2{,}000. EDM is a released "
                  r"checkpoint, not trained here and not matched to ours. NFE includes terminal "
                  r"denoising; " + TIME_NOTE + "."),
            "B": (r"Matched QM9 base models, seed mean $\pm$ sd over three seeds of 2{,}000; the VP "
                  r"model was not evaluated. NFE includes terminal denoising; " + TIME_NOTE + ".")},
        "tab:guidance": {
            "A": (r"QM9 guidance on our FM against a matched unguided control, by property. $w$ is "
                  r"nominal strength; plug-in at $w{=}4$ is the ablation's $\eta{=}0$ arm, so its marks "
                  r"are not pre-registered. " + legend(B["tab:guidance"]["A"]) +
                  r" Validity, uniqueness and DV are in the appendix."),
            "B": (r"QM9 guidance on our FM against a matched unguided control, by property. $w$ is "
                  r"nominal strength; plug-in at $w{=}4$ is the ablation's $\eta{=}0$ arm, so its marks "
                  r"are not pre-registered. DV is distinct-valid molecules per attempt. "
                  + legend(B["tab:guidance"]["B"]))},
        "tab:ablation": {
            "A": (r"BDG controls on our FM, cells $\mu/\alpha/$gap; $\tau$ is $\tau_{\mathrm{mult}}$. "
                  r"$\Delta$IB (points) is against plug-in (row 1), $\eta{=}0$ at $w{=}1$ (rows 2--4) "
                  r"and $\eta{=}0$ at $w{=}4$ (row 5); $^{\S}$: unpaired $|z|\ge%.2f$, Bonferroni over "
                  r"the grid's %d contrasts; none is pre-registered. $R$ is decoded property sd over "
                  r"unguided; clip is the \%% of guided sample-steps clipped. One-sided, replay and "
                  r"signed-strength controls did not run." % (Z_SEL, N_SEL)),
            "B": (r"BDG controls on our FM. $\Delta$IB (points, $\mu/\alpha/$gap) is against plug-in "
                  r"(row 1) and against $\eta{=}0$ at the same $w$ (row 2, the range over 16 settings, "
                  r"three properties and both $w$). $R$ is decoded property sd over unguided; clip is "
                  r"the \% of guided sample-steps clipped.")},
        "tab:recent": {
            "A": (r"Molecular guidance at $w{=}1$ on our FM, per property. " + legend(B["tab:recent"]["A"]) +
                  r" Prior-method rows are adaptations; EquiFM, EDM, validity, uniqueness and DV are "
                  r"in the appendix."),
            "B": (r"Molecular guidance at $w{=}1$, per property and flow backend; each backend has its "
                  r"own property pair and band, so compare rows only within a block. "
                  + legend(B["tab:recent"]["B"]) +
                  r" EquiFM is not bit-reproducible: re-runs move IB by up to %.2f points, which its "
                  r"$z$ does not include. Prior-method rows are adaptations." % rr)},
        "tab:m2": {
            "A": (r"DeepFlyBrain, $L{=}500$, $w{=}1$, $t\ge0.5$; the band is chosen and not comparable "
                  r"with QM9. $^{a}$Equals TMPD and LGD-MC numerically in this window. "
                  + legend(B["tab:m2"]["A"], m2=True) +
                  r" JSD is 3-mer against training data; " + TIME_NOTE + ", not a benchmark."),
            "B": (r"DeepFlyBrain GC, $L{=}500$, $w{=}1$, $t\ge0.5$. " + legend(B["tab:m2"]["B"], m2=True) +
                  r" JSD is 3-mer against training data; diversity is mean normalized Hamming "
                  r"distance; " + TIME_NOTE + ", not a benchmark.")},
    }


def tex_guidance(F, option):
    fm = F.h["fm"]
    a4 = F.a[("fm", 4)]
    rows = [("Unguided", "0", lambda p: fm[(p, "unguided")]),
            ("Plug-in", "1", lambda p: fm[(p, "plug")]),
            ("Plug-in", "4", lambda p: a4[(p, "bdg_e0t1")])]
    if option == "A":
        L = [r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & & \multicolumn{3}{c}{IB$\uparrow$} & \multicolumn{3}{c}{Mol. stab.$\uparrow$}\\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Arm & $w$ & $\mu$ & $\alpha$ & gap & $\mu$ & $\alpha$ & gap\\", r"\midrule"]
        for lab, w, g in rows:
            ib = []
            for p in PROPS:
                s, u = g(p), fm[(p, "unguided")]
                ib.append(f3(s["ib_dec"]) + ("" if lab == "Unguided" else marks(F.zdec(s, u))))
            ms = [f3(g(p)["mol_stability"]) for p in PROPS]
            L.append("%s & $%s$ & %s & %s\\\\" % (lab, w, " & ".join(ib), " & ".join(ms)))
    else:
        L = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Arm & $w$ & IB$\uparrow$ & Mol. stab.$\uparrow$ & Validity$\uparrow$ & Uniqueness$\uparrow$ & DV$\uparrow$\\"]
        for p in PROPS:
            L.append(r"\midrule")
            L.append(r"\multicolumn{7}{l}{\emph{%s}}\\" % PTEX[p])
            for lab, w, g in rows:
                s, u = g(p), fm[(p, "unguided")]
                L.append("%s & $%s$ & %s%s & %s & %s & %s & %s\\\\" % (
                    lab, w, f3(s["ib_dec"]), "" if lab == "Unguided" else marks(F.zdec(s, u)),
                    f3(s["mol_stability"]), f3(s["validity"]), f3(s["uniqueness_of_valid"]),
                    f3(s["unique_valid_per_sample"])))
    L += [r"\bottomrule", r"\end{tabular}"]
    return NL.join(L)




def abl_cells(F):
    """The five contrasts of the recommended tab:ablation layout, on fm. Each row
    names its reference: row 1 is the only cross-tree one (the ablation's eta = 0
    against the headline's plug); rows 2-5 stay inside the ablation tree."""
    a1, a4, h = F.a[("fm", 1)], F.a[("fm", 4)], F.h["fm"]
    spec = [
        (r"$\eta{=}0$, $w{=}1$", "identity", lambda p: (a1[(p, "bdg_e0t1")], h[(p, "plug")]),
         "`plug`, w = 1 (headline tree)"),
        (r"$\eta{=}0$, $w{=}4$", "strength alone", lambda p: (a4[(p, "bdg_e0t1")], a1[(p, "bdg_e0t1")]),
         "eta = 0, w = 1 (ablation tree)"),
        (r"$\eta{=}8$, $\tau{=}0.5$, $w{=}1$", r"largest $w_{\mathrm{eff}}$",
         lambda p: (a1[(p, "bdg_e8t0.5")], a1[(p, "bdg_e0t1")]), "eta = 0, w = 1 (ablation tree)"),
        (r"$\eta{=}8$, $\tau{=}1.5$, $w{=}1$", r"negative $w_{\mathrm{eff}}$",
         lambda p: (a1[(p, "bdg_e8t1.5")], a1[(p, "bdg_e0t1")]), "eta = 0, w = 1 (ablation tree)"),
        (r"$\eta{=}8$, $\tau{=}0.5$, $w{=}4$", r"$w_{\mathrm{eff}}$ at $w{=}4$",
         lambda p: (a4[(p, "bdg_e8t0.5")], a4[(p, "bdg_e0t1")]), "eta = 0, w = 4 (ablation tree)"),
    ]
    return spec


def tex_ablation(F, option):
    """The clip is applied after w, so a rung may be clip-limited (FULL_RUN_V3_PROTOCOL.md
    6.2, ABLATION_V3_PROTOCOL.md 1.1 and 4): each row's clip fraction travels in the
    body, and marks use the grid's selection-adjusted bar Z_SEL."""
    h = F.h["fm"]
    L = [TABSEP + r"\begin{tabular}{llcccc}", r"\toprule",
         r"Control & Effect isolated & $\Delta$IB (pp) & $R$ & Mol. stab.$\uparrow$ & Clip (\%)\\", r"\midrule"]
    if option == "A":
        for lab, eff, g, _ref in abl_cells(F):
            d, R, ms, cl = [], [], [], []
            for p in PROPS:
                s, ref = g(p)
                z = F.zdec(s, ref)
                d.append("%s%s" % (s1(100 * (s["ib_dec"] - ref["ib_dec"])), smark(z)))
                R.append(nz(s["sd_dec"] / h[(p, "unguided")]["sd_dec"]))
                ms.append(nz(s["mol_stability"]))
                cl.append("%.1f" % (100 * s["clip_frac"]))
            L.append("%s & %s & %s & %s & %s & %s\\\\" % (lab, eff, "/".join(d), "/".join(R), "/".join(ms),
                                                           "/".join(cl)))
    else:
        a1, a4 = F.a[("fm", 1)], F.a[("fm", 4)]
        d, R, ms, cl = [], [], [], []
        for p in PROPS:
            s, ref = a1[(p, "bdg_e0t1")], h[(p, "plug")]
            d.append(s1(100 * (s["ib_dec"] - ref["ib_dec"])))
            R.append(nz(s["sd_dec"] / h[(p, "unguided")]["sd_dec"]))
            ms.append(nz(s["mol_stability"]))
            cl.append("%.1f" % (100 * s["clip_frac"]))
        L.append(r"$\eta=0$ & identity & %s & %s & %s & %s\\" % ("/".join(d), "/".join(R), "/".join(ms),
                                                                  "/".join(cl)))
        ds, Rs, mss, cls = [], [], [], []
        for aw in (a1, a4):
            for p in PROPS:
                ref = aw[(p, "bdg_e0t1")]
                for arm in ABL_ARMS[1:]:
                    s = aw[(p, arm)]
                    ds.append(100 * (s["ib_dec"] - ref["ib_dec"]))
                    Rs.append(s["sd_dec"] / h[(p, "unguided")]["sd_dec"])
                    mss.append(s["mol_stability"])
                    cls.append(100 * s["clip_frac"])
        L.append(r"Gain/setpoint, $w=1,4$ & dispersion vs.\ strength & $%+.1f$ to $%+.1f$ & %s--%s & %s--%s & %.1f--%.1f\\"
                 % (min(ds), max(ds), nz(min(Rs)), nz(max(Rs)), nz(min(mss)), nz(max(mss)), min(cls), max(cls)))
        for lab, eff in (("One-sided residual", "negative-residual branch"),
                         ("Open-loop replay", "feedback vs.\\ schedule"),
                         ("Signed-strength plug-in", "scalar reweighting")):
            L.append(r"%s & %s & \multicolumn{4}{c}{not run in v3}\\" % (lab, eff))
    L += [r"\bottomrule", r"\end{tabular}"]
    return NL.join(L)



RECENT_ROWS = [("unguided", "Unguided, selected backbone"),
               ("plug", r"DPS-style plug-in \citep{chung2023dps}"),
               ("tmpd", r"TMPD-inspired \citep{boys2024tmpd}"),
               ("lgd_mc", r"LGD-MC \citep{song2023lgd}"),
               ("tfg", r"TFG \citep{ye2024tfg}"),
               ("bdg_e4t0.5", r"BDG (ours), $\tau_{\mathrm{mult}}=0.5$"),
               ("bdg_e4t1", r"BDG (ours), $\tau_{\mathrm{mult}}=1$")]


def recent_line(F, be, arm, label):
    H = F.h[be]
    ib, ms = [], []
    for p in PROPS:
        s, u = H[(p, arm)], H[(p, "unguided")]
        mark = marks(F.zdec(s, u) if arm != "unguided" else None,
                     F.zdec(s, H[(p, "plug")]) if arm.startswith("bdg") else None)
        ib.append(f3(s["ib_dec"]) + mark)
        ms.append(f3(s["mol_stability"]))
    return "%s & %s & %s\\\\" % (label, " & ".join(ib), " & ".join(ms))


def tex_recent(F, option):
    L = [r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{3}{c}{IB$\uparrow$} & \multicolumn{3}{c}{Mol. stab.$\uparrow$}\\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Method & $\mu$ & $\alpha$ & gap & $\mu$ & $\alpha$ & gap\\", r"\midrule"]
    bes = ["fm"] if option == "A" else ["fm", "equifm"]
    for i, be in enumerate(bes):
        if option == "B":
            if i:
                L.append(r"\midrule")
            L.append(r"\multicolumn{7}{l}{\emph{%s}}\\" % (
                "Our FM, our property pair" if be == "fm" else
                "EquiFM (released), TFG's property pair; not bit-reproducible"))
        for arm, label in RECENT_ROWS:
            if option == "B" and arm == "unguided":
                label = "Unguided"
            L.append(recent_line(F, be, arm, label))
            if arm == "unguided" and option == "A":
                L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    return NL.join(L)



M2_ROWS = [("unguided", r"Simplex FM, unguided"),
           ("plug", r"\quad + plug-in$^{a}$"),
           ("tfg_mc", r"\quad + TFG-MC ingredient"),
           ("bdg_e4t0.5", r"\quad + BDG, $\tau_{\mathrm{mult}}=0.5$"),
           ("bdg_e4t1", r"\quad + BDG, $\tau_{\mathrm{mult}}=1$")]


def m2_mark(F, prop, arm, w):
    if arm == "unguided":
        return ""
    s, u = F.m2s(prop, arm, w), F.m2s(prop, "unguided", w)
    zp = None
    if arm.startswith("bdg"):
        zp = zun(s["ib"], F.m2s(prop, "plug", w)["ib"], s["N"])
    return marks(zun(s["ib"], u["ib"], s["N"]), zp)


def tex_m2(F, option, w=1):
    if option == "A":
        L = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{IB, $w{=}%d$$\uparrow$} & \multicolumn{2}{c}{JSD ($10^{-4}$)$\downarrow$} & Div.$\uparrow$ & Time (s/1k)$\downarrow$\\" % w,
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & GC & CpG & GC & CpG & & GC/CpG\\", r"\midrule"]
        for arm, label in M2_ROWS:
            g, c = F.m2s("gc", arm, w), F.m2s("cpg", arm, w)
            div = f3(g["div"]) if f3(g["div"]) == f3(c["div"]) else "%s/%s" % (f3(g["div"]), f3(c["div"]))
            L.append("%s & %s%s & %s%s & %.2f & %.2f & %s & %.0f/%.0f\\\\" % (
                label, f3(g["ib"]), m2_mark(F, "gc", arm, w), f3(c["ib"]), m2_mark(F, "cpg", arm, w),
                1e4 * g["js"], 1e4 * c["js"], div, mean(g["s1k"]), mean(c["s1k"])))
    else:
        L = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & IB, $w{=}%d$$\uparrow$ & JSD ($10^{-4}$)$\downarrow$ & Diversity$\uparrow$ & Time (s/1k)$\downarrow$\\" % w,
             r"\midrule"]
        labels = [("unguided", "Simplex FM, unguided"), ("plug", "Simplex FM + plug-in"),
                  ("bdg_e4t0.5", r"Simplex FM + BDG, $\tau_{\mathrm{mult}}=0.5$"),
                  ("bdg_e4t1", r"Simplex FM + BDG, $\tau_{\mathrm{mult}}=1$")]
        for arm, label in labels:
            g = F.m2s("gc", arm, w)
            L.append("%s & %s%s & %.2f & %s & %.0f\\\\" % (label, f3(g["ib"]), m2_mark(F, "gc", arm, w),
                                                          1e4 * g["js"], f3(g["div"]), mean(g["s1k"])))
        for lab in (r"Dirichlet FM \citep{stark2024dirichlet}", r"Fisher Flow \citep{davis2024fisher}",
                    r"MOG-DFM \citep{chen2025multi}"):
            L.append(r"%s & \multicolumn{4}{c}{not run}\\" % lab)
    L += [r"\bottomrule", r"\end{tabular}"]
    return NL.join(L)




# ------------------------------------------------------------ LaTeX checking
def run_tex(d, name, full):
    cmds = [["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name]]
    if full:
        cmds += [["bibtex", name], cmds[0], cmds[0]]
    for c in cmds:
        subprocess.run(c, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=180, check=False)
    with open(os.path.join(d, name + ".log"), encoding="utf-8", errors="replace") as fh:
        return fh.read()


PREAMBLE = r"""\documentclass{article}
\PassOptionsToPackage{numbers,compress}{natbib}
\usepackage[final,main]{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,graphicx,microtype,url}
\usepackage{xcolor}
\newcommand{\pend}{\textcolor{black!60}{\textsf{P}}}
\renewcommand{\citep}[1]{[99]}
\newsavebox{\tb}
\makeatletter\def\@captype{table}\makeatother
"""


def latex_check(T, bodies, captions, extra=None):
    """`extra`: {name: tabular} measured for width only (the appendix bodies)."""
    if not (shutil.which("pdflatex") and shutil.which("bibtex")):
        die("--latex-check needs pdflatex and bibtex on PATH")
    from pypdf import PdfReader
    res = {"measure": {}, "compile": {}, "caption": {}, "labs": list(bodies)}
    tmp = tempfile.mkdtemp(prefix="paperfill_")
    try:
        pdir = os.path.join(ROOT, "paper")
        for f in ("neurips_2026.sty", "citation.bib", "refs_extra.bib"):
            shutil.copy(os.path.join(pdir, f), tmp)
        if os.path.isdir(os.path.join(pdir, "figs")):
            shutil.copytree(os.path.join(pdir, "figs"), os.path.join(tmp, "figs"))
        # 1. every tabular, measured in a box at the paper's \small; every caption
        # (v4's and each proposal) set at \small in the text width with a
        # "Table 9:" prefix, which approximates article's \@makecaption
        doc = [PREAMBLE, r"\begin{document}",
               r"\typeout{PFV TEXTWIDTH \the\textwidth}\typeout{PFV TEXTHEIGHT \the\textheight}"]
        items = [("current|" + lab, T[lab]["tabular"]) for lab in bodies]
        for lab, opts in bodies.items():
            for o, b in opts.items():
                items.append(("%s|%s" % (lab, o), b))
        for nm, tab in (extra or {}).items():
            items.append(("%s|A" % nm, tab))
        caps = [("cap|current|" + lab, T[lab]["caption"]) for lab in bodies]
        for lab, opts in captions.items():
            for o, c in opts.items():
                caps.append(("cap|%s|%s" % (lab, o), c))
        for nm, tab in items:
            doc.append(r"\sbox{\tb}{\small %s}\typeout{PFV %s \the\wd\tb\space \the\ht\tb\space \the\dp\tb}"
                       % (tab, nm))
        for nm, cap in caps:
            doc.append(r"\setbox\tb\vbox{\hsize=\textwidth\small\caption{%s}}"
                       r"\typeout{PFV %s \the\wd\tb\space \the\ht\tb\space \the\dp\tb}" % (cap, nm))
        items += caps
        doc.append(r"\end{document}")
        with open(os.path.join(tmp, "measure.tex"), "w", encoding="utf-8", newline=NL) as fh:
            fh.write(NL.join(doc))
        log = run_tex(tmp, "measure", False)
        for m in re.finditer(r"PFV (\S+) ([\d.]+)pt ([\d.]+)pt ([\d.]+)pt", log):
            res["measure"][m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4))
        m = re.search(r"PFV TEXTWIDTH ([\d.]+)pt", log)
        res["textwidth"] = float(m.group(1)) if m else float("nan")
        m = re.search(r"PFV TEXTHEIGHT ([\d.]+)pt", log)
        res["textheight"] = float(m.group(1)) if m else float("nan")
        if len(res["measure"]) != len(items):
            die("--latex-check measured %d of %d tabulars and captions"
                % (len(res["measure"]), len(items)))
        for k in [k for k in res["measure"] if k.startswith("cap|")]:
            res["caption"][k[4:]] = res["measure"].pop(k)
        res["extra"] = {k.split("|")[0]: res["measure"].pop(k) for k in list(res["measure"])
                        if k.split("|")[0] in (extra or {})}
        # 2. the whole paper: unchanged, all recommended, all alternatives
        with open(PAPER, encoding="utf-8") as fh:
            src = fh.read()
        live = LIVE["src"]   # the snapshot section 1b audited, not a re-read
        probe = (r"\par\typeout{PFV ENDMAIN page=\thepage\space pagetotal=\the\pagetotal"
                 r"\space pagegoal=\the\pagegoal}")
        # every option set whole, the live file as found, and each table's option
        # alone (so a cost can be attributed to one table, caption included)
        variants = ["current", "A", "B", "live"] + ["%s|%s" % (lab, o) for lab in bodies for o in ("A", "B")]
        for variant in variants:
            s = live if variant == "live" else src
            if variant in ("A", "B"):
                todo = [(lab, variant) for lab in bodies]
            elif "|" in variant:
                todo = [tuple(variant.split("|"))]
            else:
                todo = []
            for lab, o in todo:
                n0 = len(s)
                s = s.replace(T[lab]["tabular"], bodies[lab][o], 1)
                s = s.replace("\\caption{" + T[lab]["caption"] + "}",
                              "\\caption{" + captions[lab][o] + "}", 1)
                if len(s) == n0 and bodies[lab][o] == T[lab]["tabular"]:
                    die("--latex-check: substitution for %s option %s changed nothing" % (lab, o))
            s = s.replace("% MAIN_TEXT_END", probe + NL + "% MAIN_TEXT_END", 1)
            with open(os.path.join(tmp, "main.tex"), "w", encoding="utf-8", newline=NL) as fh:
                fh.write(s)
            for f in glob.glob(os.path.join(tmp, "main.*")):
                if not f.endswith(".tex"):
                    os.remove(f)
            log = run_tex(tmp, "main", True)
            pdf = os.path.join(tmp, "main.pdf")
            if not os.path.isfile(pdf):
                if variant == "live":
                    # someone else's file, possibly mid-edit: report, do not refuse
                    res["compile"][variant] = None
                    continue
                die("--latex-check: variant %s did not compile" % variant)
            pages = [p.extract_text() or "" for p in PdfReader(pdf).pages]
            ref = next((i for i, t in enumerate(pages) if re.match(r"^\s*References\b", t)), None)
            m = re.search(r"PFV ENDMAIN page=(\d+) pagetotal=([\d.]+)pt pagegoal=([\d.]+)pt", log)
            res["compile"][variant] = {
                "pages": len(pages), "main": ref,
                "overfull": len(re.findall(r"Overfull \\[hv]box", log)),
                "undefined": len(re.findall(r"(?:Citation|Reference).*undefined", log)),
                "end_page": int(m.group(1)) if m else None,
                "slack": (float(m.group(3)) - float(m.group(2))) if m else None}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


# ------------------------------------------------------ stale-prose locator
STALE = [
    ("Final comparative results are pending", "abstract: results now exist for M1 (fm, equifm, edm) and M2"),
    ("compares linear flow matching with variance-preserving diffusion under one matched protocol",
     "abstract: no VP trained here was evaluated; the only diffusion cells are the borrowed EDMsecond"),
    ("The VP baseline matches the representation, split and network size",
     "sec:diff describes a VP model that now HAS 18 cells (`vp`, 28 Sep)"),
    ("an open-loop replay separates feedback from schedule",
     "sec:bdg: open-loop replay was never run in v3; the grid cannot show the feedback is necessary"),
    ("Every principal result reports seed means, standard deviations and paired per-seed differences",
     "sec:protocol: the pre-registered test is the UNPAIRED binomial z at 2.99; nothing paired exists off blade"),
    ("The two baselines share split, preprocessing, conditioning",
     "sec:protocol: true of fm vs our (unrun) VP, not of fm vs EDMsecond"),
    ("otherwise the guidance sections re-instantiate unchanged on the VP ODE",
     "sec:fmvd: no VP guidance cell exists"),
    ("clipping frequency and realized gain must accompany attribution to dispersion",
     "sec:ablations: the per-cell clip fractions and measured w_eff are in this doc"),
    ("reserves four recent guidance comparisons per flow backend",
     "sec:recent: v3 filled them; 'reserves' is stale"),
    ("these are the three planned external sequence comparators",
     "related work: Dirichlet FM, Fisher Flow and MOG-DFM were not run"),
    ("These rows are pending evaluations",
     "app:comparisons: the three external sequence rows were not run"),
    ("Final sequence timing, batch size, $n$, seeds, and comparator alignment remain unspecified",
     "app:m2: M2 ran n=2000, batch 500, seeds 20260921-3, NFE 100, t>=0.5"),
    ("The driver defaults to $t_{\\min}=0$",
     "app:m2: m2_sweep.py now defaults to --t-min 0.5, and the run used 0.5"),
    ("the current floor uses the latter for both",
     "app:m2: fixed; gc's quantum is 1/L = 0.002 in every gc cell (field `quantum`)"),
    ("Three-mer JSD compares generated and held-out frequencies",
     "app:m2: kmer_js is computed against the TRAINING split (m2_sweep.py: real_kmer = kmer_freq(X), X = train)"),
    ("with the pair-sampling rule recorded",
     "app:m2: the rule is: first 256 sequences of the cell, all 256^2 ordered pairs including self-pairs"),
    ("Sequence protocol status", "tab:provenance points at MODALITY2_V3_PLAN.md, superseded by MODALITY2_V3_PROTOCOL.md"),
    ("Final output trees and run manifests are pending",
     "tab:provenance: the trees exist (results/v3/<be>/{v3,v3abl}/n2000, results/m2/*/n2000)"),
    ("The central question is unanswered", "sec:failure: v3 now answers part of it"),
]


def stale_lines(src, live):
    out = []
    for phrase, why in STALE:
        i = src.find(phrase)
        out.append((src[:i].count(NL) + 1 if i >= 0 else None, phrase, why, phrase in live))
    return out


LIVE = {}   # filled in main(): the live paper/main.tex, its tables and its audit
LXG = {}    # filled in main() when --latex-check ran: the measured page costs
CAPS = {}   # filled in main(): the proposed caption of every option, printed beside its body


def body_block(A, lab, opt, tex, rec=False):
    """A LaTeX body with its proposed caption, ready to paste."""
    A("**(3) LaTeX, option A**%s, with its proposed caption:" % (" (recommended)" if rec else "")
      if opt == "A" else "**Option %s**, with its proposed caption:" % opt)
    A("")
    A("```latex")
    A(r"\caption{%s}" % CAPS[lab][opt])
    A(tex)
    A("```")
    A("")


def own_change(LX, lab, opt):
    """(tabular, caption) height change of one option against v4's, in pt: what the
    table itself costs, before LaTeX decides where its float goes."""
    ht = lambda t: t[1] + t[2]  # noqa: E731
    dt = ht(LX["measure"]["%s|%s" % (lab, opt)]) - ht(LX["measure"]["current|" + lab])
    dc = ht(LX["caption"]["%s|%s" % (lab, opt)]) - ht(LX["caption"]["current|" + lab])
    return dt, dc


def end_moved(LX, c):
    """How far the end of the main text moved against v4, in pt. Includes any float
    LaTeX pushed to another page, so it is set by float placement as much as by
    the change itself."""
    if c is None or c["slack"] is None:
        return None
    base = LX["compile"]["current"]
    return base["slack"] - c["slack"] + (c["end_page"] - base["end_page"]) * LX["textheight"]


def mcost(lab, opt):
    """One line for an option bullet: what the table itself costs, and what the
    compiled paper then does."""
    if not LXG:
        return "  *Measured cost:* not measured in this generation (run with `--latex-check`)."
    c = LXG["compile"].get("%s|%s" % (lab, opt))
    if not c or c["slack"] is None:
        return "  *Measured cost:* the compile did not report it."
    dt, dc = own_change(LXG, lab, opt)
    k = end_moved(LXG, c)
    return ("  *Measured cost* (this option alone, in v4): tabular and caption grow by about %+.0f pt "
            "(%+.1f lines: tabular %+.0f, caption %+.0f). Compiled, the main text runs to **%d** page%s "
            "against the limit of 5 and its end moves by %+.0f pt, which includes any float LaTeX moved "
            "(section 12)."
            % (dt + dc, (dt + dc) / 11.0, dt, dc, c["main"], "" if c["main"] == 1 else "s", k))


def sec_live(A, F, X):
    A("## 1b. Audit of the live `paper/main.tex` (filled by another revision)")
    A("")
    if X["is_v4"]:
        A("The live file is still v4; nothing to audit.")
        A("")
        return
    A("When this ran, `paper/main.tex` (md5 `%s` of its bytes, as `md5sum` prints it; modified %s) no longer matched v4: its result tables had been filled and much of its prose rewritten by a revision that is not committed%s. Every number in its tables that the parser recognises, and every prose or text-table number one of the checks below matches, is compared with this script's recomputation from the v3-final cells, at the printed precision; each prose row names **where** the sentence sits. **This is a snapshot**: re-run the script after the next revision." % (
        LIVE["md5"][:8], LIVE["mtime"],
        " (untracked `%s` are present)" % "`, `".join(LIVE["revs"]) if LIVE["revs"] else ""))
    A("")
    aud = LIVE["audit"]
    n_bad = sum(1 for r in aud if r[5] == "MISMATCH")
    A("**Tables.** %d numbers compared; **%d mismatch**." % (sum(1 for r in aud if r[5] in ("ok", "MISMATCH")), n_bad))
    A("")
    A("| table | row | column | printed | recomputed | status | note |")
    A("|---|---|---|---|---|---|---|")
    for tab, row, col, tok, mine, st, note in aud:
        if st == "ok" and not note:
            continue
        A("| `%s` | %s | %s | %s | %s | %s | %s |" % (tab, row.replace("|", "/"), col.replace("|", "/"), tok, mine,
                                                    "**%s**" % st if st != "ok" else st, note))
    A("")
    A("Rows not listed matched exactly at the printed precision. **Not audited**: quantities no cell records other than the fm parameter count (citations, captions' wording, the text of `tab:backends`, `tab:provenance` and the appendix)%s; `tab:ablation` is a text table, audited through the prose checks below." % (
        ("; " + "; ".join(LIVE["skipped"])) if LIVE["skipped"] else ""))
    A("")
    A("**Prose and text tables.** Numbers stated in the live text, recomputed:")
    A("")
    A("| where | live text | claims | recomputed from v3-final cells | status |")
    A("|---|---|---|---|---|")
    for quote, claim, mine, st, wh in LIVE["claims"]:
        A("| %s | \"%s\" | %s | %s | %s |" % (wh, quote.replace("|", "/")[:320], claim.replace("|", "/"), mine,
                                            st if st == "ok" else "**%s**" % st))
    A("")
    if LIVE.get("unmatched"):
        A("Checks that found no matching sentence in the live text (the claim is gone, or is now worded in a way this script does not parse; read those sentences by hand): %s." % "; ".join(LIVE["unmatched"]))
        A("")
    A("**What the audit means for the live paper.**")
    A("")
    A("- Its **table numbers match the v3-final cells**: %d of %d at the printed precision." % (
        sum(1 for r in aud if r[5] == "ok"), sum(1 for r in aud if r[5] in ("ok", "MISMATCH"))))
    for b in live_issues(X, long=True):
        A("- " + b)
    A("")


def live_issues(X, long=False):
    """The live paper's problems, each with where it sits (sections 1b and 13)."""
    aud, cl = LIVE["audit"], LIVE["claims"]
    out = []
    lv = (LXG.get("compile") or {}).get("live") if LXG else None
    if lv and lv["main"] and lv["main"] > 5:
        out.append("It runs to **%d main-text pages** under pagecheck's rule (section 12)." % lv["main"])
    nb = [r for r in aud if r[5] == "MISMATCH"]
    if nb:
        out.append("%d table number(s) do not match the cells (%s)." % (len(nb), ", ".join(sorted({"`%s`" % r[0] for r in nb}))))
    jsd_w = {}
    for tab, row, col, tok, mine, st, note in aud:
        m = re.match(r"matches w = (\d+)", note)
        if tab == "tab:m2" and m and ("JSD" in col or "divers" in col.lower()):
            jsd_w.setdefault(col, set()).add(m.group(1))
    LIVE["mixed_cols"] = [c.replace("$\\downarrow$", "").replace("$\\uparrow$", "") for c, ws in jsd_w.items() if len(ws) > 1]
    for col in LIVE["mixed_cols"]:
        out.append("**`tab:m2`'s %s column mixes strengths**: its rows match the cells at different w while the header names none." % col)
    for r in aud:
        if r[5] == "NOTE":
            out.append("`%s`: %s." % (r[0], r[6]))

    def locs(pred):
        seen = []
        for c in cl:
            if pred(c) and c[4] not in seen:
                seen.append(c[4])
        return ", ".join(seen)
    nf = lambda c: c[3] == "NOT v3-FINAL"  # noqa: E731
    if any(nf(c) for c in cl):
        if long:
            out.append("Its **useful-yield and paired-contrast claims are not v3-final** (in: %s). Useful yield needs the per-molecule sidecars, which exist in the repo only for the older Betty run (`results/v3/<be>/n5000/`, V3_BETTY_PAIRED.md). That run's arm set still held `bdg_e4t0.75`, so its \"18 contrasts\" are 3 BDG arms x 3 properties x 2 generators, and its fm guided cells were **guided by TFG's f_A and scored by TFG's oracle**, so their band and in-band belong to a different property pair from the blade fm tables they sit beside. The v3-final reading of BDG against plug is the unpaired one: decoded %d above / %d below / %d tie over 12 contrasts (section 9)."
                       % ((locs(nf),) + tuple(X["bdg_vs_plug_dec"])))
        else:
            out.append("its useful-yield and paired-contrast claims come from the Betty n = 5000 run, not v3-final (%s)" % locs(nf))
    nr = lambda c: c[3] == "not reproduced from v3-final cells"  # noqa: E731
    for c in cl:
        if nr(c):
            out.append("\"%s\" (%s) is not reproduced by the v3-final cells, and the sentence does not say which run it comes from: %s." % (
                c[1], c[4], c[2]))
    ten = lambda c: c[1] == "~10x" and c[3].startswith("MISMATCH")  # noqa: E731
    if any(ten(c) for c in cl):
        five = locs(lambda c: c[1] == "~5x" and c[3] == "ok")
        out.append("**\"roughly ten times\"** for the M2 window contradicts the recorded correction (~5x); it survives in: %s%s." % (
            locs(ten), (", while %s already says about five times" % five) if five else ""))
    gh = [c for c in cl if "GPU-hours" in c[0]]
    if gh:
        tot = X["gpuh"]["v3"] + X["gpuh"]["v3abl"]
        bad = [c for c in gh if "total ok" not in c[3]]
        out.append("**The run-cost figure** (%s)%s. It is also the wrong unit and scope: summed per-cell wall-clock `seconds` including scoring, with cells run concurrently, not GPU-hours, and Modality 1 only (Modality 2 adds %.2f h of recorded minutes on an unrecorded device). Say e.g. \"the molecular run's cells recorded %.2f hours of wall-clock, including scoring (%.2f headline, %.2f ablation); the sequence run's %.2f hours\"." % (
            locs(lambda c: "GPU-hours" in c[0]),
            (": the exact total is %.2f h, which rounds to %.1f; the printed %s is the sum of the rounded stages" % (tot, tot, bad[0][1].split(" ")[0])) if bad else "",
            sum(X["m2_hours"].values()), tot, X["gpuh"]["v3"], X["gpuh"]["v3abl"], sum(X["m2_hours"].values())))
    for c in cl:
        if c[3].startswith("ok at 2.99"):
            out.append("\"%s\" (%s) is stated at the headline bar; at the grid's selection-adjusted bar %.2f it is %d (ABLATION_V3_PROTOCOL.md 4 requires that bar beside a grid claim)." % (
                c[1], c[4], Z_SEL, X["grid_counts"]["lo_sel"]))
        elif c[3] == "ok on continuous only":
            out.append("\"%s\" (%s) holds on continuous in-band only, not on the decoded IB the paper defines: %s." % (c[1], c[4], c[2]))
        elif c[3].startswith("ok, one BDG arm only"):
            out.append("The M2 sentence claiming %s (%s) names only `bdg_e4t0.5`: %s." % (c[1], c[4], c[2]))
        elif c[3] == "MISMATCH" or (c[3].startswith("MISMATCH") and c[1] != "~10x" and "GPU-hours" not in c[0]):
            out.append("\"%s\" (%s) does not match: %s." % (c[1], c[4], c[2]))
    return out


# ------------------------------------------------------------------- the doc
def write_doc(args, T, src, F, M2, m2_extra, R, TS, X, LX, arm_census):
    L = []
    A = L.append
    A("# Paper table fill from the v3-final run (blade, n = 2000)")
    A("")
    A("Generated by `%s`. Do not hand-edit; re-run it:" % SCRIPT)
    A("")
    A("```")
    A(REGEN)
    A("```")
    A("")
    A("A map from every pending cell (`\\pend`) of the paper's result tables to the v3-final "
      "number that fills it, with its source, its format, a ready-to-paste LaTeX body, what "
      "no v3 run can fill, and where the table's design does not match the data. **The paper "
      "is read, not edited.** The pending cells are those of the committed v4 manuscript, "
      "`paper/versions/v4_main.tex` (CHANGELOG: v4 is current; identical to `paper/main.tex` "
      "at commit 7697670). %s Every table number below is recomputed from the cells by this "
      "script (the few quoted diagnostics name their source file); section 10 checks the "
      "V3_RESULTS docs against the same cells."
      % ("The live `paper/main.tex` is still identical to it." if X["is_v4"] else
         "**The live `paper/main.tex` has since been filled by another revision**, so section 1b "
         "audits every number it now carries against the same cells."))
    A("")
    A("## 0. Sources, statistics and the caveats that travel with every number")
    A("")
    A("| | |")
    A("|---|---|")
    A("| M1 headline | `results/v3/{fm,equifm,edm}/v3/n2000/seed{20261001,20261002,20261003}/tr__*.json` (63 + 63 + 18 cells) |")
    A("| M1 ablation | `results/v3/{fm,equifm}/v3abl/n2000/seed{...}/tr__*.json`, w = 1 and w = 4 (306 + 306 cells) |")
    A("| M2 | `results/m2/m2/n2000/` (headline, gc and cpg, w = 1 and 4), `results/m2/m2abl/n2000/` (gc grid at w = 1, 4 plus three t_min = 0 cells), `results/m2/m2wsweep/n2000/` (w = 16, 64, **one seed**); seeds 20260921/22/23; %d cells |"
      % sum(len(v) for v in M2.values()))
    A("| run | Bobo's blade run: n = 2000 per cell, batch 500 (4 controllers per cell), 100-step Euler, target q50, guidance for t >= 0.5, headline w = 1, ablation w in {1, 4}; M1 on %s |"
      % ", ".join(sorted({d for d in TS["devices"]})))
    A("| not used | `results/v3/<be>/n5000/` (the older Betty run), v2, q90, results/full, results/sweep, results/bdg_port, results/bdg_local |")
    A("| pooling | N-weighted over the three seed cells, N = 6000 per arm. With equal n per seed this equals the seed mean |")
    A("| IB | the paper defines IB on **decoded** samples (sec. 4.1), so every IB below is `in_band_fraction_dec` unless marked *continuous* |")
    A("| test (M1) | pre-registered (FULL_RUN_V3_PROTOCOL.md 6.1): unpaired binomial se = sqrt(p(1-p)/N) per arm, z = d / sqrt(se_a^2 + se_b^2), Bonferroni 0.05/18 two-sided, **\\|z\\| >= %.2f** |" % Z_BAR)
    A("| test (M2) | MODALITY2_V3_PROTOCOL.md states **no** significance threshold (it says a paired test would be valid, sec. 3, and that a null must be reported beside the detectable difference, sec. 5). M2 verdicts here therefore use M1's unpaired z at %.2f, **labelled not pre-registered for M2** |" % Z_BAR)
    A("| detectable difference | at the bar, two arms near p = 0.10 need %.2f pp (M1); M2 gc near 0.14 needs %.2f pp, cpg near 0.50 needs %.2f pp |"
      % (100 * mdd(0.10, 6000), 100 * mdd(0.14, 6000), 100 * mdd(0.50, 6000)))
    A("")
    A("**Caveats. Each applies wherever its subject appears below.**")
    A("")
    A("1. **q50 only.** No q90 number is used or printed.")
    A("2. **Both headline BDG arms, always separately**: `bdg_e4t0.5` (eta 4, tau_mult 0.5) and `bdg_e4t1` (eta 4, tau_mult 1). Never pooled, never only the better one.")
    A("3. **w = 1 is not any arm's best strength and equal w is not equal force** (FULL_RUN_V3_PROTOCOL.md 2.2, 6.2). No table here ranks methods or names a best method. TFG at w = 1 buys its in-band with the largest chemistry loss.")
    A("4. **in_band is not comparable across M1 backends** (each has its own property pair, so its own delta), nor between M1 and M2 (M2's delta is a chosen band, `delta_ratio` 0.16). Nothing here is tested or ranked across backends.")
    A("5. **EquiFM is not bit-reproducible.** The headline's two BDG arms re-run in the ablation move in-band by up to **%.1e** (continuous; %.1e decoded), the size of EquiFM's BDG effects. Its binomial se understates its noise; every EquiFM verdict carries this. fm is bit-exact (0 difference on every re-run and on eta = 0 vs plug)."
      % (X["rerun"]["equifm"]["ib"], X["rerun"]["equifm"]["ib_dec"]))
    A("6. **`edm` is TFG's released EDMsecond, a BORROWED diffusion checkpoint** (checkpoint `weights/EDMsecond/generative_model_ema.npy`, md5 `%s` as every edm cell records it in `prov.edm_md5`%s; its arguments come from `args.pickle`, `prov.edm_args`). **Our own** QM9 VP diffusion model is the SEPARATE `vp` backend (md5 `8a3390a6`, selected epoch 1475, 18 cells as of 28 Sep), and it is what fills the \"VP, trained here\" cells. Both are diffusion; only `vp` is ours. `edm` and `vp` each ran unguided and plug only."
      % (X["edm_md5"]["cells"][:8], X["edm_md5"]["note"]))
    A("7. **Clip fraction.** M1 `guided_steps` is summed over the n/batch = 4 batches, so the per-sample-step fraction is `clipped_sample_steps / ((guided_steps / (n/batch)) * n)`. `v3_sanity.py` divides by `guided_steps * n` and understates it 4x; not used. M2 counts one `gen_vjp` per guided step per batch, so its fraction is `clipped_sample_steps / ((gen_vjp / n_controllers) * n)`; the script checks `gen_vjp / n_controllers` equals the guided steps of the t grid (50 at t >= 0.5, 100 at t >= 0) in every cell.")
    A("8. **Useful yield** (decoded in-band AND stable, per molecule) **is pending**: the per-molecule `*.permol.pt` sidecars stay on blade. It is not approximated by multiplying marginals.")
    A("9. **The ablation shows that the deviation weight w_eff, not the global w, carries the effect. It does not show the feedback loop is necessary**: no open-loop replay or fixed-w_eff control was run.")
    A("10. **M2 caveats.** (a) v3's window t >= 0.5 against t >= 0: plug gains %s pp from the window, while the best result inside the window is %s pp (`bdg_e8t0.5` at w = 4, vs unguided), so the window is worth about %.1fx the method. (b) plug, tmpd and lgd_mc are numerically one baseline in M2's window (max \\|d\\| %.1e over in-band, GC mean, GC sd, k-mer JS, decode confidence and diversity; v_f collapses for t >= 0.5). (c) M2's delta is a chosen band, not commensurable with M1. (d) The w = 16 and w = 64 rows are **one seed**. (e) The DeepFlyBrain-vs-Keras equivalence is in no committed file."
      % (pp(X["m2win"]["plug_t0"] - X["m2win"]["plug_t05"]), pp(X["m2win"]["best_in"] - X["m2win"]["ung"]),
         (X["m2win"]["plug_t0"] - X["m2win"]["plug_t05"]) / (X["m2win"]["best_in"] - X["m2win"]["ung"]),
         X["m2_same"]))
    A("11. **Time** is each cell's wall-clock `seconds` (M1) or `minutes` (M2), per 1,000 samples, **including scoring** (f_A, f_B, second oracle, RDKit). It is not a controlled benchmark: cells ran concurrently on a shared 4-GPU box, and e.g. fm/mu/tmpd's three seed cells took %s s per 1,000, and M2's unguided cpg cells, identical samples at w = 1 and w = 4, took %.1f and %.1f s per 1,000. Summed, it is cell wall-clock hours, not GPU-hours. M2 cells record **no device**."
      % ("/".join("%.0f" % x for x in F.h["fm"][("mu", "tmpd")]["s1k"]),
         mean(F.m2[("m2", "cpg", "unguided", 1.0, 0.5)]["s1k"]), mean(F.m2[("m2", "cpg", "unguided", 4.0, 0.5)]["s1k"])))
    ce = X["clip_ext"]
    A("12. **The clip is applied after w, so a rung may be clip-limited rather than controller-limited** (FULL_RUN_V3_PROTOCOL.md 6.2; ABLATION_V3_PROTOCOL.md 1.1 and 4). It is the single most likely explanation for a BDG null and must be offered before \"spread control does not help\"; the clip cannot be removed (222 non-finite samples without it, CLIP_PILOT.md). The grid's extreme rungs are its most clipped: fm mu at w = 4 clips %.2f %% of guided sample-steps at e8t1.5 and %.2f %% at e8t0.5, against eta = 0's %.2f %% (pooled over each rung's 3 seeds; the largest single cell is %.2f %%, %s). Every clip fraction below uses caveat 7's denominator and travels beside the in-band it qualifies."
      % (ce["fm_mu_w4"]["bdg_e8t1.5"], ce["fm_mu_w4"]["bdg_e8t0.5"], ce["fm_mu_w4"]["bdg_e0t1"],
         ce["cell_max"]["fm"][0], ce["cell_max"]["fm"][1]))
    A("")
    # ------------------------------------------------------------- census
    A("## 1. The pending cells, and how many v3 can fill")
    A("")
    A("| table | v4 line | `\\pend` cells in v4 | fillable from v3-final cells | cannot be filled by any v3 run | `\\pend` left in live main.tex |")
    A("|---|---|---|---|---|---|")
    # (fillable, cannot-fill). v4 holds 8 pending cells here: 4 for the FM row
    # and 4 for the VP row. All 8 are now fillable -- the VP four by the `vp`
    # backend, which did not exist when this census was first written.
    census = {"tab:fmvd": (8, 0, "none -- the `VP, trained here` row is filled by the `vp` backend (28 Sep)"),
              "tab:guidance": (15, 0, "none, but 'by property' needs a layout choice (sec. 3)"),
              "tab:ablation": (6, 9, "the 9 cells of one-sided residual, open-loop replay, signed-strength plug-in (never run)"),
              "tab:recent": (24, 0, "none, but 'per property and backend' and two BDG rows need a layout choice"),
              "tab:m2": (12, 12, "the 12 cells of Dirichlet FM, Fisher Flow, MOG-DFM (not run in M2)"),
              "tab:training": (1, 6, "training provenance; v3 cells fill only the molecular-FM hash and sampling hardware")}
    tot = [0, 0, 0]
    for lab in ("tab:fmvd", "tab:guidance", "tab:ablation", "tab:recent", "tab:m2", "tab:training"):
        a, b, why = census[lab]
        if a + b != T[lab]["pend"]:
            die("census for %s adds to %d, the paper holds %d" % (lab, a + b, T[lab]["pend"]))
        tot[0] += T[lab]["pend"]
        tot[1] += a
        tot[2] += b
        tl = LIVE["T"].get(lab)
        A("| `%s` | %d | %d | %d | %d: %s | %s |" % (lab, T[lab]["line"], T[lab]["pend"], a, b, why,
                                                    tl["pend"] if tl else "table absent"))
    A("| **total** | | **%d** | **%d** | **%d** | %d |" % (tuple(tot) + (LIVE["src"].count("\\pend") - 1,)))
    A("")
    A("One more `\\pend` sits in prose (sec. 4.1, \"`\\pend{}` marks a pending value\"); it becomes stale once the tables are filled (section 11).")
    A("")
    A("**Page budget.** The main text is exactly 5.00 pages with no slack (CHANGELOG v4), so every option below is priced in lines. A `\\small` booktabs row is about one text line (~11 pt); a grouped header (`\\multicolumn` + `\\cmidrule`) adds one. Section 12 measures every proposed tabular with pdflatex and compiles the paper with each set substituted.")
    A("")
    A("**Formats used in the bodies.** IB, stability, validity, uniqueness and DV as fractions to 3 decimals (N = 6000, so se <= 0.0065 and a fourth decimal is noise); Delta IB in percentage points to 1 decimal; R to 2 decimals; clip in % of guided sample-steps to 1 decimal; JSD in units of 1e-4; time in s per 1,000 samples (caveat 11). The ablation bodies print R and molecule stability to 2 decimals without the leading zero (.36, as the live paper does), and the ablation and appendix bodies set `\\tabcolsep` to 4 pt inside the float, so that they fit the text width (section 12).")
    A("")
    A("**Marks, and every mark carries its direction.** On IB levels (tab:guidance, tab:recent, tab:m2): `$^{\\dagger+}$` / `$^{\\dagger-}$` = above / below unguided, `$^{\\ddagger+}$` / `$^{\\ddagger-}$` = a BDG arm above / below plug-in, all at the pre-registered unpaired \\|z\\| >= %.2f (M1); on M2 the same test is **not pre-registered** and the caption says so. %s On the ablation's signed Delta IB cells: `$^{\\S}$` = \\|z\\| >= %.2f against **that row's** reference arm, the Bonferroni bar over the grid's own family of %d contrasts (17 settings x 3 properties x 2 bases, ABLATION_V3_PROTOCOL.md 4), which the protocol requires beside any grid claim; no grid contrast is pre-registered. Each body below is printed with a proposed caption that defines exactly the marks it carries." % (
        Z_BAR,
        ("In M1's bodies every `\\ddagger` is a **loss** (%d BDG-vs-plug contrasts below plug-in, none above), so an unsigned mark would read as BDG's win." % X["bdg_vs_plug_dec"][1])
        if X["bdg_vs_plug_dec"][0] == 0 and X["bdg_vs_plug_dec"][1] else
        "In M1's bodies the BDG-vs-plug marks carry both signs (%d above, %d below)." % tuple(X["bdg_vs_plug_dec"][:2]),
        Z_SEL, N_SEL))
    A("")
    # ------------------------------------------------------------- tab:fmvd
    sec_live(A, F, X)
    sec_fmvd(A, T, F, R)
    sec_guidance(A, T, F)
    sec_ablation(A, T, F, X)
    sec_recent(A, T, F)
    sec_m2(A, T, F, M2, X, m2_extra)
    sec_training(A, T, TS)
    sec_blocks(A, F)
    sec_h2h(A, F, X)
    sec_check(A, X, arm_census)
    A("## 11. Prose the fill makes stale (not `\\pend` cells)")
    A("")
    A("Filling the tables makes these sentences of v4 false or out of date. Line numbers are in `paper/versions/v4_main.tex`; the last column says whether the sentence is still in the live `paper/main.tex`. Nothing here was edited.")
    A("")
    A("| v4 line | text | why | still in live main.tex |")
    A("|---|---|---|---|")
    for ln, phrase, why, alive in stale_lines(src, LIVE["src"]):
        A("| %s | \"%s\" | %s | %s |" % (ln if ln else "not found", phrase.replace("|", "/"), why,
                                        "**yes**" if alive else "no, already changed"))
    A("")
    sec_latex(A, LX)
    A("## 13. Open items")
    A("")
    A("1. **Useful yield and every paired statistic** need the blade sidecars: `python proj1/scripts/v3_paired.py --n 2000 --backends fm,equifm --md-out docs/results/V3_BLADE_PAIRED.md`, then `--stage v3abl --w 1` and `--w 4`. `tab:ablation`'s caption says \"paired change\"; the point estimate below is exact (a mean of per-molecule paired differences equals the difference of the two means when both arms hold the same molecules), only its paired interval is missing.")
    A("2. **The VP row of `tab:fmvd` is RESOLVED** (28 Sep): our own VP checkpoint was sampled under v3's exact settings, 2 arms x 3 properties x 3 seeds x n = 2000. Section 2's relabel-or-withdraw analysis is retained for the record but its premise is gone. Outstanding on that row: its `s/sample` was measured on a B200 MIG slice while every other row is RTX A6000, which the caption states -- footnote it or re-time that cell.")
    A("3. **The layout decisions** in sections 2-7 are the authors'. The recommended set is length-measured in section 12.")
    A("4. **Prose** listed in section 11 must change with the tables.")
    A("5. **The DeepFlyBrain-vs-Keras equivalence** must be committed or withdrawn before any DeepFlyBrain activity number is cited.")
    A("6. **EquiFM's run-to-run noise** (caveat 5) is not in any error bar; a same-seed re-run envelope for EquiFM would size it.")
    if not X["is_v4"]:
        issues = live_issues(X)
        A("7. **The live `paper/main.tex`** (snapshot: md5 `%s` of its bytes, modified %s; section 1b lists each item with where it sits):%s Re-run this script after each revision." % (
            LIVE["md5"][:8], LIVE["mtime"], ("" if not issues else NL + NL.join("   - " + i.rstrip(".") + "." for i in issues) + NL + "  ")
            if issues else " no problem found."))
    A("")
    out = NL.join(L) + NL
    dst = args.md_out if os.path.isabs(args.md_out) else os.path.join(ROOT, args.md_out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8", newline=NL) as fh:
        fh.write(out)
    print("wrote %s (%d lines)" % (dst, out.count(NL)))


def cellsrc(be, stage, prop, arm, w=1):
    # the grid tag in the file name differs by backend: flow (fm), native (equifm), uniform (edm)
    solver = {"fm": "flow", "equifm": "native", "edm": "uniform"}.get(be, "<grid>")
    return ("`results/v3/%s/%s/n2000/seed<S>/tr__%s__%s__q50__w%d__n2000_s100_euler_%s_win0.5_b500_seed<S>.json`"
            % (be, stage, prop, arm, w, solver))


def sec_fmvd(A, T, F, R):
    A("## 2. `tab:fmvd`: matched QM9 base models")
    A("")
    A("Caption: \"%s\"" % T["tab:fmvd"]["caption"])
    A("")
    A("**(1) Structure** (2 rows x 5 columns; NFE is given):")
    A("")
    for ln in md_table_of(T, "tab:fmvd"):
        A(ln)
    A("")
    A("**(2) Cell map.** Source: the unguided headline cells, %s, fields `mol_stability`, `validity`, `uniqueness_of_valid`, `seconds`; `cost.gen_fwd / (n/batch)` gives NFE. The three property cells of one seed are averaged first, then mean +- sample sd over the 3 seeds (the caption's \"seed mean +- standard deviation\"). On fm and EquiFM the three property cells of a seed are **the same samples** (max spread 0); on EDM they are not (max spread %.4f), which is EDM's own GPU nondeterminism."
      % (cellsrc("fm", "v3", "<prop>", "unguided"), R["edm"]["spread"]))
    A("")
    A("| row | Mol. stab. | Validity | Uniqueness | DV | NFE (measured) | Time, s/1k | per-seed mol. stab. | fillable |")
    A("|---|---|---|---|---|---|---|---|---|")
    for be, lab, ok in (("fm", "FM, trained here", "yes"),
                        ("edm", "EDMsecond, **external** (not the paper's VP)", "only if the row is relabelled"),
                        ("equifm", "EquiFM, **external** (reference only)", "no row for it")):
        ss = R[be]["seed"]
        A("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            lab, pm([s["mol_stability"] for s in ss]).replace("$\\pm$", "+-"),
            pm([s["validity"] for s in ss]).replace("$\\pm$", "+-"),
            pm([s["uniqueness_of_valid"] for s in ss]).replace("$\\pm$", "+-"),
            pm([s["unique_valid_per_sample"] for s in ss]).replace("$\\pm$", "+-"),
            "/".join("%g" % x for x in R[be]["nfe"]),
            pm([s["s1k"] for s in ss], 1).replace("$\\pm$", "+-"),
            ", ".join(f4(s["mol_stability"]) for s in ss), ok))
    ss = R["vp"]["seed"]
    A("| VP, trained here (**ours**) | %s | %s | %s | %s | %s | %s | %s | **yes** |" % (
        pm([x["mol_stability"] for x in ss]).replace("$\pm$", "+-"),
        pm([x["validity"] for x in ss]).replace("$\pm$", "+-"),
        pm([x["uniqueness_of_valid"] for x in ss]).replace("$\pm$", "+-"),
        pm([x["unique_valid_per_sample"] for x in ss]).replace("$\pm$", "+-"),
        "/".join("%g" % x for x in R["vp"]["nfe"]),
        pm([x["s1k"] for x in ss], 1).replace("$\pm$", "+-"),
        ", ".join(f4(x["mol_stability"]) for x in ss)))
    A("")
    A("**(4) All four `VP, trained here` cells are now fillable** from the `vp` backend (28 Sep). The NFE 100 of the FM row is confirmed by the cells (%s per batch); `edm` runs %s and `vp` runs %s."
      % ("/".join("%g" % x for x in R["fm"]["nfe"]), "/".join("%g" % x for x in R["edm"]["nfe"]),
         "/".join("%g" % x for x in R["vp"]["nfe"])))
    A("")
    A("**(5) RESOLVED 28 Sep: the matched FM-vs-VP comparison now exists.** `vp` is matched to `fm` on backbone, parameter count, epochs, batch, EMA, split and training seed, so the contrast isolates the generator family. **The two options below are SUPERSEDED** -- kept only because the v4 manuscript was written against them. Neither relabelling nor withdrawing the row is needed.")
    A("")
    A("- **Option A, relabel the second row as the external EDMsecond** (body below). Table rows unchanged, but the caption grows. It must drop \"Matched\" and say the row is borrowed and unmatched in data (EDM's preprocessing removes the 3,054 uncharacterized molecules this project keeps), architecture and training. It also forces rewording in the abstract, sec. 3.3, sec. 4.1 (fair comparison), sec. 4.2 and Figure 1(a), which all describe a VP trained here (section 11). What it buys: a diffusion reference scored by the same sampler settings and the same evaluator, which shows the released diffusion model far ahead of our FM on stability (%s vs %s), a statement about two checkpoints, not about the two families."
      % (f3(mean(s["mol_stability"] for s in R["edm"]["seed"])), f3(mean(s["mol_stability"] for s in R["fm"]["seed"]))))
    A(mcost("tab:fmvd", "A"))
    A("- **Option B, keep the VP row and mark it not run.** Rows and caption about the same. The rubric's FM-vs-diffusion section is then empty and sec. 3.3 describes a model with no result; both must say so.")
    A(mcost("tab:fmvd", "B"))
    A("- **Recommendation: A.** The comparison the section is required to make is otherwise absent, and the external row is honest if labelled. Adding EquiFM as a third row would pair a released FM with a released diffusion model (+1 row). The authors decide.")
    A("")
    body_block(A, "tab:fmvd", "A", tex_fmvd(F, R, "A"), rec=True)
    body_block(A, "tab:fmvd", "B", tex_fmvd(F, R, "B"))


def sec_guidance(A, T, F):
    fm, a4 = F.h["fm"], F.a[("fm", 4)]
    A("## 3. `tab:guidance`: guided against unguided")
    A("")
    A("Caption: \"%s\"" % T["tab:guidance"]["caption"])
    A("")
    A("**(1) Structure** (3 rows x 5 metric columns, one column set, \"by property\"):")
    A("")
    for ln in md_table_of(T, "tab:guidance"):
        A(ln)
    A("")
    A("**(2) Cell map, our FM (the selected backbone), all three properties.** Unguided and plug-in w = 1: %s. **Plug-in w = 4 exists only as the ablation's `bdg_e0t1` at w = 4** (%s): eta = 0 multiplies the dispersion term by zero, so its field is plug's (bit-identical at w = 1 on fm: every seed, every property, 0 difference against the headline plug; see section 4). Fields: IB `in_band_fraction_dec`, `mol_stability`, `validity`, `uniqueness_of_valid`, DV `unique_valid_per_sample`; N-weighted over 3 seeds."
      % (cellsrc("fm", "v3", "<prop>", "unguided|plug"), cellsrc("fm", "v3abl", "<prop>", "bdg_e0t1", 4)))
    A("")
    A("| property | arm | w | IB (dec) | z vs ung. | verdict | IB cont. | Mol. stab. | Validity | Uniqueness | DV | clip % |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PROPS:
        u = fm[(p, "unguided")]
        for lab, w, s in (("Unguided", 0, u), ("Plug-in", 1, fm[(p, "plug")]), ("Plug-in", 4, a4[(p, "bdg_e0t1")])):
            z = F.zdec(s, u) if w else float("nan")
            A("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %.2f |" % (
                p, lab, w, f4(s["ib_dec"]), "--" if not w else "%+.2f" % z, "--" if not w else verdict(z),
                f4(s["ib"]), f4(s["mol_stability"]), f4(s["validity"]), f4(s["uniqueness_of_valid"]),
                f4(s["unique_valid_per_sample"]), 100 * s["clip_frac"]))
    A("")
    A("The same rows for EquiFM (plug-in w = 4 from its ablation) and for EDM (no w = 4 cell: EDM ran no ablation) are in section 7.2.")
    A("")
    A("**(4) Cannot be filled:** nothing on fm. EDM has no plug-in at w = 4.")
    A("")
    A("**(5) Design mismatch: \"by property\" with one column set, three properties, three backends.**")
    A("")
    A("- **Option A, one IB and one stability column per property** (3 rows kept, 6 numeric columns, +1 grouped-header line). Validity, uniqueness and DV move to the two appendix tables of section 7.2 (the first holds DV, the second validity and uniqueness, with the same three rows), and the caption says so. Molecule stability stays beside in-band.")
    A(mcost("tab:guidance", "A"))
    A("- **Option B, three property blocks** (9 rows + 3 block labels) with all five metrics.")
    A(mcost("tab:guidance", "B"))
    A("- **Recommendation: A**, on page budget; it keeps every property in the main text. The authors decide.")
    A("")
    A("Reading, fm: plug-in at w = 1 raises decoded IB over unguided on mu and gap at the bar and not on alpha (z %+.2f); at w = 4 all three clear it. Every step up in w costs molecule stability on every property (w = 1: %s pp; w = 4: %s pp vs unguided)."
      % (F.zdec(fm[("alpha", "plug")], fm[("alpha", "unguided")]),
         "/".join(pp(fm[(p, "plug")]["mol_stability"] - fm[(p, "unguided")]["mol_stability"], 1) for p in PROPS),
         "/".join(pp(a4[(p, "bdg_e0t1")]["mol_stability"] - fm[(p, "unguided")]["mol_stability"], 1) for p in PROPS)))
    A("")
    body_block(A, "tab:guidance", "A", tex_guidance(F, "A"), rec=True)
    body_block(A, "tab:guidance", "B", tex_guidance(F, "B"))


def sec_ablation(A, T, F, X):
    A("## 4. `tab:ablation`: BDG controls")
    A("")
    A("Caption: \"%s\"" % T["tab:ablation"]["caption"])
    A("")
    A("**(1) Structure** (5 rows x 3 metric columns):")
    A("")
    for ln in md_table_of(T, "tab:ablation"):
        A(ln)
    A("")
    A("**(2) What each column means here.** Delta IB: decoded in-band minus **the row's reference arm's**, same backend, property and seeds; the reference is not the same in every row (see the table below, and the caption of each body). **The paper calls it a paired change. The point estimate is exact from these cells**: both arms hold the same 6000 index-paired molecules, so the mean of per-molecule differences equals the difference of means. Only the paired interval needs the blade sidecars. The z is the unpaired binomial one; **no ablation contrast is pre-registered**, and the bodies read it at the grid's selection-adjusted bar \\|z\\| >= %.2f (Bonferroni over the protocol's %d-contrast family), with the headline bar 2.99 given beside it only as a label. R: decoded property sd, `sqrt(mean(prop_rmse_eval_dec^2) - (mean(f_B_dec_mean) - target)^2)` over the 3 seed cells, divided by the same for the headline unguided arm. Mol. stab.: `mol_stability`. Clip: caveat 7's fraction, pooled over the 3 seeds (caveat 12)." % (Z_SEL, N_SEL))
    A("")
    A("| row of option A | cell | reference arm | Delta IB mu/alpha/gap (pp) | unpaired z | past 3.49 / past 2.99 |")
    A("|---|---|---|---|---|---|")
    for lab, eff, g, ref_name in abl_cells(F):
        ds, zs = [], []
        for p in PROPS:
            s, ref = g(p)
            ds.append(pp(s["ib_dec"] - ref["ib_dec"]))
            zs.append(F.zdec(s, ref))
        A("| %s | %s | %s | %s | %s | %d of 3 / %d of 3 |" % (
            md_label(lab), eff.replace("$", "").replace("w_{\\mathrm{eff}}", "w_eff").replace("{=}", " = "), ref_name,
            "/".join(ds), "/".join("%+.2f" % z for z in zs),
            sum(abs(z) >= Z_SEL for z in zs), sum(abs(z) >= Z_BAR for z in zs)))
    A("")
    A("Row 1 is the only cross-tree contrast (the ablation's `bdg_e0t1` against the headline's `plug`), which ABLATION_V3_PROTOCOL.md 4 says must be stated. Row 2 is **not** a same-w contrast: it is eta = 0 at w = 4 against eta = 0 at w = 1, which on fm is bit-identical to the headline plug at w = 1; against eta = 0 at its own w it is 0.0 by construction.")
    A("")
    A("Row `eta = 0`: `bdg_e0t1` (ablation) against `plug` (headline), same n and seeds:")
    A("")
    A("| backend | property | max \\|d\\| in-band over seeds | decoded | mol. stab. | re-run of `bdg_e4t0.5` / `bdg_e4t1` (headline vs ablation copy), max \\|d\\| in-band |")
    A("|---|---|---|---|---|---|")
    for be in ("fm", "equifm"):
        for p in PROPS:
            e = X["eta0"][(be, p)]
            A("| %s | %s | %.1e | %.1e | %.1e | %.1e / %.1e |" % (be, p, e["ib"], e["ib_dec"], e["ms"],
                                                                 X["rerun_pp"][(be, p, "bdg_e4t0.5")],
                                                                 X["rerun_pp"][(be, p, "bdg_e4t1")]))
    A("")
    A("fm is bit-exact: the identity row is 0.0 on every seed and property. On EquiFM the eta = 0 difference (%.1e) sits inside its own re-run noise (%.1e), so it passes as an identity only up to that noise (caveat 5)."
      % (max(X["eta0"][("equifm", p)]["ib"] for p in PROPS), X["rerun"]["equifm"]["ib"]))
    A("")
    A("**The grid, per (backend, property, w)**: decoded Delta IB against eta = 0 at the same w, with the rung's **measured** w_eff (3-seed mean of `diag.bdg_w_eff`) and clip fraction (pooled over the rung's 3 seeds). Extremes over the 16 BDG rungs. EquiFM rows carry caveat 5: its re-runs move in-band by up to %.2f pp, which no z here includes." % (100 * X["rerun"]["equifm"]["ib"]))
    A("")
    A("| backend | prop | w | eta=0 IB (dec) | eta=0 mol. stab. | eta=0 clip % | Delta IB range (pp) | at e8t0.5: Delta IB (z) | at e8t1.5: Delta IB (z) | Spearman(w_eff, Delta IB) | e8t0.5 is max w_eff / e8t1.5 is min | w_eff range | R range | mol. stab. range | clip % range, pooled over each rung's 3 seeds |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, g in X["grid"].items():
        be, p, w = key
        A("| %s | %s | %d | %s | %s | %.2f | %+.2f to %+.2f | %+.2f (%+.2f) | %+.2f (%+.2f) | %+.2f | %s / %s | %.2f to %.2f | %.2f to %.2f | %.4f to %.4f | %.2f to %.2f |" % (
            be, p, w, f4(g["e0"]["ib_dec"]), f4(g["e0"]["mol_stability"]), 100 * g["e0"]["clip_frac"],
            g["dmin"], g["dmax"], g["d_hi"], g["z_hi"], g["d_lo"], g["z_lo"], g["rho"],
            "yes" if g["hi_is_max"] else "**no**", "yes" if g["lo_is_min"] else "**no**",
            g["wmin"], g["wmax"], g["Rmin"], g["Rmax"], g["msmin"], g["msmax"], g["cmin"], g["cmax"]))
    A("")
    c = X["grid_counts"]
    A("- Spearman(measured w_eff, decoded Delta IB) over the 16 rungs: %.2f to %.2f, positive in %d of %d grids." % (c["rho_min"], c["rho_max"], c["rho_pos"], c["n"]))
    A("- **The corner counts, at the grid's selection-adjusted bar \\|z\\| >= %.2f** (Bonferroni over ABLATION_V3_PROTOCOL.md 4's %d-contrast family; the protocol requires this bar beside any grid claim), decoded in-band: e8t1.5 below eta = 0 in **%d of %d** grids; e8t0.5 above eta = 0 in **%d of %d** (%d of 6 at w = 1, %d of 6 at w = 4). At the headline bar 2.99, which is **not** selection-adjusted and not pre-registered for the grid, the same counts are %d of %d and %d of %d (%d and %d at w = 1 / w = 4). e8t1.5's Delta is negative in %d of %d grids. EquiFM contributes %d of the %d below and %d of the %d above at 2.99, and its re-run noise (up to %.2f pp, caveat 5) is not in those z." % (
        Z_SEL, N_SEL, c["lo_sel"], c["n"], c["hi_sel"], c["n"], c["hi_sel_w1"], c["hi_sel_w4"],
        c["lo_sig"], c["n"], c["hi_sig"], c["n"], c["hi_sig_w1"], c["hi_sig_w4"], c["lo_neg"], c["n"],
        c["lo_sig_eq"], c["lo_sig"], c["hi_sig_eq"], c["hi_sig"], 100 * X["rerun"]["equifm"]["ib"]))
    A("- The corners were chosen as the rungs of largest and smallest **measured** w_eff, so they are selected, not pre-specified contrasts; **none is pre-registered**. A claim that *some* rung beat eta = 0 or plug is a max-over-16 selection (ABLATION_V3_PROTOCOL.md 4).")
    at = c["dc_at"]
    A("- **Decoded (the paper's IB) and continuous in-band differ by up to %.2f pp** in a grid Delta (%s/%s/w = %d/`%s`: %+.2f decoded against %+.2f continuous), and they disagree on %d corner verdict%s at 2.99: %s. On continuous in-band the 2.99 counts are %d of %d below and %d of %d above, which is what V3_BLADE_READOUT.md reports (its grid is continuous)." % (
        c["dc_max"], at[0], at[1], at[2], at[3], at[4], at[5], len(c["flips"]), "" if len(c["flips"]) == 1 else "s",
        "; ".join("%s/%s/w = %d/`%s` decoded %+.2f (z %+.2f, %s), continuous %+.2f (z %+.2f, %s)" % f for f in c["flips"]),
        c["lo_sig_c"], c["n"], c["hi_sig_c"], c["n"]))
    A("- The pre-run prediction of w_eff of order 10^2-10^3 (FULL_RUN_V3_PROTOCOL.md 4.1, BDG_LADDER_MEASURED.md) is **not what the run measured**: v3's cells record w_eff between %.2f and %.2f across all 12 grids. Quote the measured values." % (c["wmin"], c["wmax"]))
    A("")
    A("**(4) Cannot be filled by any v3 run:** `One-sided residual` (no one-sided `...o` BDG variant exists in any v3 or M2 tree), `Open-loop replay` (never run in v3; BDG_REVIEW.md's replay is a pre-v3 pilot and may not stand in), `Signed-strength plug-in` (never run). Arms present in the v3 trees: %s." % ", ".join("`%s`" % a for a in X["arm_census"]))
    A("")
    A("**(5) Design mismatch: one row per control, but the data are 2 backends x 3 properties x 2 strengths, three of the five controls were never run, and Delta IB is defined as paired.**")
    A("")
    A("- **Option A, re-purpose the rows to contrasts v3 did run** (5 rows kept, fm only, each cell mu/alpha/gap; EquiFM's grid in the appendix): eta = 0 at w = 1 (identity), eta = 0 at w = 4 (strength alone), and the grid's corners eta = 8 at tau_mult 0.5 and 1.5 (largest and smallest measured w_eff) at w = 1, and eta = 8, tau_mult 0.5 at w = 4. Rows unchanged; the caption must say the three removed controls were not run and that Delta IB's z is unpaired. It also makes the text of sec. 3.5 and 4.4 promising an open-loop replay stale (section 11).")
    A(mcost("tab:ablation", "A"))
    A("- **Option B, keep the paper's five rows**: fill eta = 0 and put the grid's range in the gain/setpoint row; the three unrun rows read \"not run in v3\". Honest, but a range from %+.1f to %+.1f pp says nothing about the mechanism, and three of five rows are empty." % (c["dmin_all"], c["dmax_all"]))
    A(mcost("tab:ablation", "B"))
    A("- **Recommendation: A.** Every row is then measured, and rows 2 vs 3-5 separate strength from deviation weight, which is what the grid can support (caveat 9). The authors decide.")
    A("")
    A("Realized gain and clip frequency, which sec. 4.4 says must accompany the attribution (fm, 3-seed means):")
    A("")
    A("| row of option A | measured w_eff mu/alpha/gap | clip % mu/alpha/gap |")
    A("|---|---|---|")
    for lab, eff, g, _ref in abl_cells(F):
        ws, cs = [], []
        for p in PROPS:
            s, _ = g(p)
            ws.append("%.2f" % s["weff3"] if "weff3" in s else "--")
            cs.append("%.2f" % (100 * s["clip_frac"]))
        A("| %s | %s | %s |" % (md_label(lab), "/".join(ws), "/".join(cs)))
    A("")
    body_block(A, "tab:ablation", "A", tex_ablation(F, "A"), rec=True)
    body_block(A, "tab:ablation", "B", tex_ablation(F, "B"))


def sec_recent(A, T, F):
    A("## 5. `tab:recent`: comparison with recent methods")
    A("")
    A("Caption: \"%s\"" % T["tab:recent"]["caption"])
    A("")
    A("**(1) Structure** (6 rows x 4 columns, \"per property and backend\"):")
    A("")
    for ln in md_table_of(T, "tab:recent"):
        A(ln)
    A("")
    A("**(2) Cell map.** Source: %s for each arm, on fm and equifm (EDM ran unguided and plug only). IB `in_band_fraction_dec`, Mol. stab. `mol_stability`, DV `unique_valid_per_sample`, Time `seconds` per 1,000. The BDG row must be **two rows** (the caption already says \"reported separately\")." % cellsrc("<be>", "v3", "<prop>", "<arm>"))
    A("")
    for be in ("fm", "equifm", "edm"):
        H = F.h[be]
        A("*%s*%s" % (BE_LABEL[be], " (EquiFM verdicts carry caveat 5)" if be == "equifm" else ""))
        A("")
        A("| arm | IB mu | IB alpha | IB gap | Mol. stab. mu | alpha | gap | DV mu | alpha | gap | Time s/1k mu | alpha | gap |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for arm in HEAD_ARMS:
            if (PROPS[0], arm) not in H:
                continue
            A("| `%s` | %s | %s | %s | %s |" % (
                arm, " | ".join(f4(H[(p, arm)]["ib_dec"]) for p in PROPS),
                " | ".join(f4(H[(p, arm)]["mol_stability"]) for p in PROPS),
                " | ".join(f4(H[(p, arm)]["unique_valid_per_sample"]) for p in PROPS),
                " | ".join("%.0f" % mean(H[(p, arm)]["s1k"]) for p in PROPS)))
        A("")
    A("**(4) Cannot be filled:** nothing, for fm and EquiFM. TMPD, LGD-MC, TFG and BDG have no EDM cells.")
    A("")
    A("**(5) Design mismatch: one column set for 2 flow backends x 3 properties, and one BDG row for two arms.**")
    A("")
    A("- **Option A, our FM only, one IB and one stability column per property, BDG split** (7 rows, 6 numeric columns: +1 row and +1 grouped-header line). DV, EquiFM and EDM go to the appendix tables of section 7.2 (the second of which also carries validity and uniqueness); time leaves the table and belongs in the cost paragraph of app. A.3 (per-arm values in the tables above; wall-clock with scoring, caveat 11). This drops DV and Time from the main text; the caption says where DV is.")
    A(mcost("tab:recent", "A"))
    A("- **Option B, fm and EquiFM blocks** in the same column layout (14 rows + 2 block labels). The two blocks must not be compared (caveat 4).")
    A(mcost("tab:recent", "B"))
    A("- **Recommendation: A.** The authors decide.")
    A("")
    body_block(A, "tab:recent", "A", tex_recent(F, "A"), rec=True)
    body_block(A, "tab:recent", "B", tex_recent(F, "B"))
    parts = []
    for a, lab in (("bdg_e4t0.5", "tau_mult 0.5"), ("bdg_e4t1", "tau_mult 1")):
        sig, tie = [], 0
        for be in ("fm", "equifm"):
            for p in PROPS:
                s, r = F.h[be][(p, a)], F.h[be][(p, "plug")]
                z = F.zdec(s, r)
                if abs(z) >= Z_BAR:
                    sig.append("%s %s %s pp (z %+.2f)" % (BE_SHORT[be], p, pp(s["ib_dec"] - r["ib_dec"]), z))
                else:
                    tie += 1
        parts.append("%s: %s%d of 6 cells tie plug-in" % (lab, ("%s past the bar; " % ", ".join(sig)) if sig else "", tie))
    A("**Reading for the text, both BDG arms against plug-in at the pre-registered bar (decoded IB, fm and EquiFM):** %s. Every BDG-vs-plug mark in these bodies is a `$^{\\ddagger-}$`, a **loss**; EquiFM's carries caveat 5. Molecule stability moves the other way (section 9)." % "; ".join(parts))
    A("")


def sec_m2(A, T, F, M2, X, m2_extra):
    A("## 6. `tab:m2`: transfer to DeepFlyBrain")
    A("")
    A("Caption: \"%s\"" % T["tab:m2"]["caption"])
    A("")
    A("**(1) Structure** (6 rows x 4 columns, \"by property\"):")
    A("")
    for ln in md_table_of(T, "tab:m2"):
        A(ln)
    A("")
    A("**(2) Cell map.** Source: `results/m2/m2/n2000/seed<S>/<prop>__<arm>__q50__w<w>__<variant>__n2000_nfe100_win0.5_b500_dr0.16__s<S>.json`, S in {20260921, 20260922, 20260923}. IB `in_band_fraction` (on the argmax-decoded sequence, exact count), JSD `kmer_js` (3-mer, **against the training split**), Diversity `diversity` (mean normalized Hamming over the first 256 sequences of the cell, all ordered pairs including self-pairs), Time `minutes` x 60 x 1000 / n. IB is N-weighted over seeds (N = 6000); JSD, decode confidence and diversity are seed means. Property sd is the pooled sd over the 6000 sequences from `gc_mean`/`gc_sd` (named `gc_*` in cpg cells too). The headline tree holds unguided at w = 1 and w = 4; they are the same samples (max \\|d\\| %.1e), so unguided enters once." % X["m2_ung_same"])
    A("")
    A("Property sd is pooled exactly over the 6000 sequences from the cells' `gc_sd`, which `m2_sweep.py` computes with torch's unbiased `gc.std()`; M2_V3_RESULTS.md and V3_FINAL_SUMMARY.md pool as if it were a population sd, so their sd can differ from these in the 5th decimal (e.g. gc unguided %.5f there, %.5f here). No ratio or conclusion moves." % (
        F.m2s("gc", "unguided", 1)["sd_pop"], F.m2s("gc", "unguided", 1)["sd"]))
    A("")
    sh = X["m2_share"]
    A("The full M2 block, per property and strength. Verdicts use the unpaired binomial z at 2.99, **not pre-registered for M2** (section 0); N = 6000 per arm. On gc, plug applies a correction share of %.4f at w = 1 against M1 plug's 0.055 (the figure `results/m2_share.json` quotes from FULL_RUN_V3_PROTOCOL.md 2.2); `bdg_e4t0.5` applies %.1fx plug's share at w = 1 and %.1fx at w = 4, `bdg_e4t1` %.2fx and %.2fx (n = %d, one seed; gc only). Read every BDG-vs-plug row as a statement about strength as much as method." % (
        sh["plug"], sh["r"][("bdg_e4t0.5", 1)], sh["r"][("bdg_e4t0.5", 4)], sh["r"][("bdg_e4t1", 1)], sh["r"][("bdg_e4t1", 4)], sh["n"]))
    A("")
    for w in (1, 4):
        for prop in M2_PROPS:
            u = F.m2s(prop, "unguided", w)
            A("*%s, w = %d* (delta %.5f, target %.4f)" % (prop, w, u["delta"], u["y"]))
            A("")
            A("| arm | IB | +-se | z vs ung. | z vs plug | Delta vs ung. (pp) | bias/delta | prop. sd | sd / unguided | k-mer JS | decode conf. | diversity | clip % | w_eff | Time s/1k |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            pl = F.m2s(prop, "plug", w)
            for arm in M2_ARMS:
                s = F.m2s(prop, arm, w)
                zu = zun(s["ib"], u["ib"], s["N"]) if arm != "unguided" else float("nan")
                zp = zun(s["ib"], pl["ib"], s["N"]) if arm.startswith("bdg") else float("nan")
                A("| `%s` | %s | %s | %s | %s | %s | %+.3f | %.5f | %.3f | %.6f | %.4f | %.4f | %.2f | %s | %.0f |" % (
                    arm, f4(s["ib"]), f4(bse(s["ib"], s["N"])),
                    "--" if zu != zu else "%+.2f %s" % (zu, verdict(zu)),
                    "--" if zp != zp else "%+.2f %s" % (zp, verdict(zp)),
                    "--" if arm == "unguided" else pp(s["ib"] - u["ib"]), s["bias_d"], s["sd"],
                    s["sd"] / u["sd"], s["js"], s["conf"], s["div"], 100 * s["clip_frac"],
                    "%.2f" % s["weff"] if "weff" in s else "--",
                    # unguided's samples are shared across w, its wall-clock is not:
                    # print each strength's own cells' time
                    mean(F.m2[("m2", prop, arm, float(w), 0.5)]["s1k"])))
            A("")
    A("Unguided enters every block with the w = 1 cells' samples (identical at w = 4, above); its Time column is each strength's **own** cells' wall-clock, which differs for identical work (caveat 11).")
    A("")
    A("Diagnostics outside the headline (not table cells):")
    A("")
    A("| cell | IB | vs | Delta (pp) | clip % | seeds |")
    A("|---|---|---|---|---|---|")
    ung = F.m2s("gc", "unguided", 1)
    for arm in ("plug", "bdg_e4t0.5", "bdg_e4t1"):
        s = F.m2s("gc", arm, 1, stage="m2abl", t=0.0)
        A("| gc `%s` w = 1, **t >= 0** | %s | same arm at t >= 0.5 | %s | %.1f | 3 |" % (
            arm, f4(s["ib"]), pp(s["ib"] - F.m2s("gc", arm, 1)["ib"]), 100 * s["clip_frac"]))
    for arm in ("plug", "bdg_e4t0.5"):
        for w in (16, 64):
            per = M2[("m2wsweep", "gc", arm, float(w), 0.5)]
            (sd_,) = per.keys()
            s = F.m2s("gc", arm, w, stage="m2wsweep")
            u1 = M2[("m2", "gc", "unguided", 1.0, 0.5)][sd_]["in_band_fraction"]
            A("| gc `%s` w = %d | %s | unguided, same seed %d (%s) | %s | %.1f | **1** |" % (
                arm, w, f4(s["ib"]), sd_, f4(u1), pp(s["ib"] - u1), 100 * s["clip_frac"]))
    b8 = F.m2s("gc", "bdg_e8t0.5", 4, stage="m2abl")
    A("| gc `bdg_e8t0.5` w = 4 (best inside the window) | %s | unguided | %s | %.1f | 3 |" % (f4(b8["ib"]), pp(b8["ib"] - ung["ib"]), 100 * b8["clip_frac"]))
    A("")
    z05 = zun(F.m2s("gc", "bdg_e4t0.5", 1, "m2abl", 0.0)["ib"], F.m2s("gc", "plug", 1, "m2abl", 0.0)["ib"], 6000)
    z1 = zun(F.m2s("gc", "bdg_e4t1", 1, "m2abl", 0.0)["ib"], F.m2s("gc", "plug", 1, "m2abl", 0.0)["ib"], 6000)
    c0 = [100 * F.m2s("gc", a, 1, "m2abl", 0.0)["clip_frac"] for a in ("plug", "bdg_e4t0.5", "bdg_e4t1")]
    A("BDG against plug at t >= 0 (w = 1, gc; the three t >= 0 cells clip on %.0f-%.0f %% of guided sample-steps): `bdg_e4t0.5` %s pp (unpaired z %+.2f, %s), `bdg_e4t1` %s pp (z %+.2f, %s)." % (
        min(c0), max(c0), pp(F.m2s("gc", "bdg_e4t0.5", 1, "m2abl", 0.0)["ib"] - F.m2s("gc", "plug", 1, "m2abl", 0.0)["ib"]), z05, verdict(z05),
        pp(F.m2s("gc", "bdg_e4t1", 1, "m2abl", 0.0)["ib"] - F.m2s("gc", "plug", 1, "m2abl", 0.0)["ib"]), z1, verdict(z1)))
    A("")
    e0 = X["m2_e0_full"]
    e0_zero = all(v["ib"] == 0 and v["fields"] == 0 and v["bias"] == 0 and v["clip_max"] == 0 for v in e0.values())
    rr = X["m2_rerun_full"]
    A("Pre-registered M2 gate (MODALITY2_V3_PROTOCOL.md 4.1): `bdg_e0t1` (m2abl) against `plug` (m2), gc, over the 3 seeds, max \\|d\\| at w = 1 / w = 4: in-band %.1e / %.1e; GC mean, GC sd, k-mer JS, decode confidence and diversity %.1e / %.1e; bias/delta %.1e / %.1e; clipped sample-steps %d / %d. The gate requires bit-for-bit; %s." % (
        e0[1.0]["ib"], e0[4.0]["ib"], e0[1.0]["fields"], e0[4.0]["fields"], e0[1.0]["bias"], e0[4.0]["bias"],
        e0[1.0]["clip_max"], e0[4.0]["clip_max"],
        "it passes on every recorded metric" if e0_zero else "**it fails at float precision; the protocol voids BDG rows on a failure**"))
    A("")
    A("Headline BDG cells re-run in m2abl (the same %d gc cells, same seeds): in-band %.1e; max \\|d\\| %.1e over GC mean, GC sd, k-mer JS, decode confidence and diversity; %.1e on bias/delta (the same GC-mean difference divided by delta = %.5f); clipped sample-steps differ by up to %d in %d of %d cells. Quote this whole line, not one of its figures, as M2's re-run reproducibility." % (
        rr["n"], rr["ib"], rr["fields"], rr["bias"], F.m2s("gc", "unguided", 1)["delta"], rr["clip_max"], rr["clip_cells"], rr["n"]))
    if m2_extra:
        A("")
        A("Cells present but not in the expected M2 set: %s." % ", ".join(m2_extra))
    A("")
    A("**(4) Cannot be filled:** the 12 cells of Dirichlet FM, Fisher Flow and MOG-DFM. M2 ran unguided, plug, tmpd, lgd_mc, tfg_mc and BDG on our simplex FM only; no external sequence model was run.")
    A("")
    A("**(5) Design mismatch: three rows name methods that were not run, the methods that were run (TMPD, LGD-MC, TFG-MC) have no row, BDG is two arms, and there are two properties.**")
    A("")
    A("- **Option A, rows = what ran, both properties** (5 rows: unguided, plug-in, TFG-MC, BDG x 2; plug-in's row is also TMPD and LGD-MC, which equal it numerically in this window; columns IB, JSD for GC and CpG, diversity, time): one row fewer and one grouped-header line more. Requires rewording the related-work sentence and app. A.4, which call the three external models planned comparators (section 11).")
    A(mcost("tab:m2", "A"))
    A("- **Option B, keep the planned rows**: GC only, BDG split (+1 row), the three external rows \"not run\", CpG in the appendix. Leaves half the table as not run.")
    A(mcost("tab:m2", "B"))
    sh = X["m2_share"]
    A("- **Recommendation: A.** The authors decide. Whichever layout, the BDG rows must be read with the force confound of MODALITY2_V3_PROTOCOL.md sec. 3: on gc, `bdg_e4t0.5` applies %.1fx plug's correction share at w = 1 and %.1fx at w = 4, while `bdg_e4t1` applies %.2fx and %.2fx (`results/m2_share.json`, n = %d, seed %s; **no committed file measures it on CpG**). A `bdg_e4t0.5` gain is therefore partly force; the eta sweep at fixed w (M2_ABLATION_GRID.md) is what separates them. Neither body carries w = 4; if the text quotes it, it must quote both arms and the confounds together, e.g.:" % (
        sh["r"][("bdg_e4t0.5", 1)], sh["r"][("bdg_e4t0.5", 4)], sh["r"][("bdg_e4t1", 1)], sh["r"][("bdg_e4t1", 4)],
        sh["n"], sh["seed"]))
    A("")
    A("  > %s" % m2_bdg_sentence(F, X))
    A("")
    A("- **The fidelity columns must carry the strength of the coverage column beside them.** A w = 4 coverage figure placed next to w = 1 JSD and diversity shows a coverage gain without its fidelity cost: at w = 4 `bdg_e4t0.5`'s k-mer JS on gc is %.6f (%.2fx unguided's %.6f) and its decode confidence %.4f (unguided %.4f); at w = 1 they are %.6f and %.4f. Either give JSD and diversity per w, or keep the table at one strength." % (
        F.m2s("gc", "bdg_e4t0.5", 4)["js"], F.m2s("gc", "bdg_e4t0.5", 4)["js"] / F.m2s("gc", "unguided", 4)["js"],
        F.m2s("gc", "unguided", 4)["js"], F.m2s("gc", "bdg_e4t0.5", 4)["conf"], F.m2s("gc", "unguided", 4)["conf"],
        F.m2s("gc", "bdg_e4t0.5", 1)["js"], F.m2s("gc", "bdg_e4t0.5", 1)["conf"]))
    A("")
    body_block(A, "tab:m2", "A", tex_m2(F, "A"), rec=True)
    body_block(A, "tab:m2", "B", tex_m2(F, "B"))
    sec_m2_functional(A, F, X)


def m2_bdg_sentence(F, X):
    """The M2 BDG-vs-plug reading, both arms, both properties, both strengths, with
    the force and clip confounds attached (rule 3; MODALITY2_V3_PROTOCOL.md 3, 4.1)."""
    def arm_txt(arm, w):
        out = []
        for pr, nm in (("gc", "GC"), ("cpg", "CpG")):
            s, pl = F.m2s(pr, arm, w), F.m2s(pr, "plug", w)
            z = zun(s["ib"], pl["ib"], s["N"])
            out.append("%s pp on %s (z %+.2f, %s)" % (pp(s["ib"] - pl["ib"]), nm, z, verdict(z)))
        return " and ".join(out)
    segs = ["at w = %d, BDG with tau_mult 0.5 is %s, and with tau_mult 1 %s" % (w, arm_txt("bdg_e4t0.5", w), arm_txt("bdg_e4t1", w))
            for w in (1, 4)]
    clip = lambda arm, w: " / ".join("%.2f %%" % (100 * F.m2s(pr, arm, w)["clip_frac"]) for pr in M2_PROPS)  # noqa: E731
    sh = X["m2_share"]["r"]
    return ("Against plug-in (unpaired z at %.2f, not pre-registered for M2): %s; %s. tau_mult 0.5 applies %.1fx plug's correction share on GC at w = 1 and %.1fx at w = 4 (not measured on CpG), and clips %s of guided sample-steps at w = 1 and %s at w = 4 (GC / CpG), against plug's %s and %s."
            % (Z_BAR, segs[0], segs[1], sh[("bdg_e4t0.5", 1)], sh[("bdg_e4t0.5", 4)],
               clip("bdg_e4t0.5", 1), clip("bdg_e4t0.5", 4), clip("plug", 1), clip("plug", 4)))


def sec_m2_functional(A, F, X):
    fx = X["m2_func"]
    g, d = fx["gate"], fx["dfb"]
    A("### 6.1 Functional evaluation: committed, not pre-registered")
    A("")
    A("M2 has no analogue of RDKit validity, so in-band could in principle be bought at no visible cost. Two scorers answer that, and both committed a summary: the realism gate (`results/m2_gate_scores.json`, `proj1/m2/enhancer_gate.py`) and DeepFlyBrain activity (`results/m2_dfb_activity.json`, `proj1/m2/deepflybrain.py`). **Neither is part of any pre-registered gate**: MODALITY2_V3_PROTOCOL.md names no functional test, so these numbers decide nothing. Each scored cell's in-band in both files was checked against its cell file (all agree to 1e-6).")
    A("")
    ug_g = g["grp"][("gc", "unguided", 1.0, 0.5)]
    ug_d = d["grp"][("gc", "unguided", 1.0, 0.5)]
    A("- **Realism gate**: held-out AUC %.4f (real enhancers against a per-sequence order-1 Markov null, which matches each sequence's GC and CpG by construction). Reference means: real train %.4f, real test %.4f, Markov null %.4f; unguided gc samples %.4f." % (
        g["auc"], g["ref"]["real_train"], g["ref"]["real_test"], g["ref"]["markov_null"], mean(v["gate_mean"] for v in ug_g.values())))
    A("- **DeepFlyBrain** (re-implemented in PyTorch; caveat 10(e)): real test max-topic %.4f, %.1f %% predicted accessible; our unguided samples %.4f and %.1f %%. The frozen base, not the guidance, is the weak link on this reading." % (
        d["ref"]["real_test"]["max_topic_mean"], 100 * d["ref"]["real_test"]["frac_active"],
        mean(v["dfb_max_topic"] for v in ug_d.values()), 100 * mean(v["dfb_frac_active"] for v in ug_d.values())))
    for tag, nm, fld in (("gate", "gate mean", "gate_mean"), ("dfb", "DeepFlyBrain max-topic", "dfb_max_topic")):
        dl, key, ns = fx[tag]["worst"]
        dm, km, nm_ = fx[tag]["worst_m"]
        A("- Largest %s drop against unguided over every scored cell: **%+.4f** against unguided on the same seeds (%s `%s` w = %g, t >= %g, %d scored seed%s); **%+.4f** against unguided's 3-seed mean, the reading M2_V3_RESULTS.md prints (%s `%s` w = %g, %d scored seed%s).%s" % (
            nm, dl, key[0], key[1], key[2], key[3], ns, "" if ns == 1 else "s",
            dm, km[0], km[1], km[2], nm_, "" if nm_ == 1 else "s",
            " They differ because a partly scored cell's seeds need not have unguided's mean score; the same-seed reading is the paired one." if abs(dl - dm) > 5e-5 else ""))
    A("")
    A("Headline arms (mean over the scored seeds, and the change against unguided on the same seeds):")
    A("")
    A("| prop | arm | w | seeds scored | gate mean | vs unguided | DFB max-topic | vs unguided |")
    A("|---|---|---|---|---|---|---|---|")
    for pr in M2_PROPS:
        ug1, ud1 = g["grp"][(pr, "unguided", 1.0, 0.5)], d["grp"][(pr, "unguided", 1.0, 0.5)]
        for w in (1.0, 4.0):
            for arm in M2_ARMS:
                if arm == "unguided" and w == 4.0:
                    continue
                pg, pd = g["grp"].get((pr, arm, w, 0.5), {}), d["grp"].get((pr, arm, w, 0.5), {})
                if not pg:
                    A("| %s | `%s` | %g | **0** | -- | -- | -- | -- |" % (pr, arm, w))
                    continue
                cm = [s for s in pg if s in ug1]
                A("| %s | `%s` | %g | %d | %.4f | %+.4f | %.4f | %+.4f |" % (
                    pr, arm, w, len(pg), mean(v["gate_mean"] for v in pg.values()),
                    mean(pg[s]["gate_mean"] - ug1[s]["gate_mean"] for s in cm),
                    mean(v["dfb_max_topic"] for v in pd.values()),
                    mean(pd[s]["dfb_max_topic"] - ud1[s]["dfb_max_topic"] for s in cm if s in pd)))
    A("")
    same = set(g["miss"]) == set(d["miss"]) and all(sorted(g["miss"][k]) == sorted(d["miss"][k]) for k in g["miss"])
    A("**Coverage: %d of %d distinct cells scored by the gate, %d by DeepFlyBrain%s.** The scorers read the per-sequence `*.permol.pt` sidecars, which stay on blade, so an unscored cell cannot be scored from this repository. Unscored%s:" % (
        g["n_scored"], fx["n_expected"], d["n_scored"], " (the same cells)" if same else " (**different cells**; the gate's list is shown)",
        "" if same else " by the gate"))
    A("")
    A("| prop | arm | w | t_min | seeds unscored |")
    A("|---|---|---|---|---|")
    for key in sorted(g["miss"], key=lambda k: (k[0], k[1], k[2], k[3])):
        A("| %s | `%s` | %g | %g | %d |" % (key[0], key[1], key[2], key[3], len(g["miss"][key])))
    t0 = sum(len(v) for k, v in g["miss"].items() if k[3] == 0.0)
    t0_all = sum(len(F.M2[k]) for k in F.M2 if k[4] == 0.0)
    b8 = len(g["miss"].get(("gc", "bdg_e8t0.5", 4.0, 0.5), []))
    A("")
    A("Two gaps matter: **%d of the %d t >= 0 cells are unscored**, so the early window has %s functional reading; and **%d of 3 seeds of `bdg_e8t0.5` at w = 4**, the grid's strongest rung (+4.23 pp against unguided), are unscored. Caveats that travel with these numbers: the DeepFlyBrain port's equivalence to the published Keras model is in **no committed file** (caveat 10(e)), so no DeepFlyBrain number should be cited until it is committed or withdrawn (open item 5); DeepFlyBrain is a predictor with its own error, not an assay, and our corpus is its training data; the per-sequence correlation between the gate and GC or CpG is uncommitted." % (
        t0, t0_all, "no" if t0 == t0_all else "a partial", b8))
    A("")


def sec_training(A, T, TS):
    A("## 7. `tab:training`, and the appendix bodies the options move columns into")
    A("")
    A("### 7.1 `tab:training`")
    A("")
    for ln in md_table_of(T, "tab:training"):
        A(ln)
    A("")
    A("This is a training-provenance table. **A v3 run records sampling, not training**, so it fills only two things: the molecular FM's checkpoint hash (every fm cell records `prov.gen_md5` = `%s`, which equals the md5 of `weights/fm_ema.pt` = `%s`) and the sampling hardware (%s). Everything else below comes from checkpoint metadata or training files in the repo, **not from the v3 run**, and is listed so the authors can verify it; the VP column has no checkpoint at all." % (
        ", ".join(TS["fm_cell_md5"]), TS["md5"]["weights/fm_ema.pt"], ", ".join(TS["devices"])))
    A("")
    args_fa = [TS["args"]["weights/f_%s_%s.pt" % (ab, p)] for ab in "AB" for p in PROPS]
    seeds_fa = sorted({a.get("seed") for a in args_fa})
    m2a = TS["args"]["proj1/m2/blade_bundle/fm_m2_dfb500.pt"]
    fma = TS["args"]["weights/fm_ema.pt"]
    A("| pending cell | candidate value | source | from the v3 run? |")
    A("|---|---|---|---|")
    A("| Optimizer, Sequence FM | Adam | `proj1/m2/simplex_fm.py:%s` (`torch.optim.Adam`) | no |" % TS["m2_adam_line"])
    A("| EMA, Sequence FM | %s | `simplex_fm.py` %s; the checkpoint stores no EMA weights | no |" % (
        "none" if not TS["m2_has_ema"] else "check", "contains no EMA" if not TS["m2_has_ema"] else "mentions EMA"))
    A("| Run seed, f_A / f_B | %s | `args.seed` of all six `weights/f_{A,B}_{mu,alpha,gap}.pt` | no |" % "/".join(str(s) for s in seeds_fa))
    A("| Run seed, Sequence FM | %s | `seed` key of `proj1/m2/blade_bundle/fm_m2_dfb500.pt` | no |" % m2a.get("seed"))
    A("| Hashes / hardware, Molecular FM / VP | FM md5 `%s`; VP: no checkpoint | cells' `prov.gen_md5`; file md5 | **FM hash: yes** |" % TS["md5"]["weights/fm_ema.pt"][:8])
    A("| Hashes / hardware, f_A / f_B | %s | file md5 (cells record only the path) | no |" % ", ".join(
        "%s `%s`" % (f.split("/")[-1][:-3], TS["md5"][f][:8]) for f in sorted(TS["md5"]) if "/f_" in f))
    A("| Hashes / hardware, Sequence FM | md5 `%s`; trained on \"%s\" | file md5 (M2 cells record only the file name, no hash, no device); `blade_bundle/train.log` line 1 | no |" % (
        TS["md5"]["proj1/m2/blade_bundle/fm_m2_dfb500.pt"][:8], TS["m2_train_device"]))
    A("")
    A("The molecular FM checkpoint's stored arguments verify five of the six asterisked settings for **FM only**: epochs %s (checkpoint `epoch` %s), batch %s, lr %s (cosine down to %s x lr, not to zero), EMA %s, seed %s; the sixth, Adam, is in `proj1/scripts/train_fm.py:%s`, code rather than a stored argument. The asterisks cannot be removed for VP. Training hardware for the molecular FM and the predictors is not recorded in their checkpoints." % (
        fma.get("epochs"), fma.get("epoch"), fma.get("batch"), fma.get("lr"), fma.get("lr_min_frac"), fma.get("ema"),
        fma.get("seed"), TS["fm_adam_line"]))
    A("")
    A("**Design mismatch:** none. Only rows the v3 run cannot fill. Recommendation: fill the FM hash from the cells; take the rest from the checkpoint records above after checking them; leave VP as P or delete the VP half of the column.")
    A("")


APP_NAMES = {"unguided": "unguided", "plug": "plug-in", "tmpd": "TMPD", "lgd_mc": "LGD-MC", "tfg": "TFG",
             "bdg_e4t0.5": r"BDG, $\tau_{\mathrm{mult}}{=}0.5$", "bdg_e4t1": r"BDG, $\tau_{\mathrm{mult}}{=}1$"}
# The two appendix bodies: (1) coverage, stability and DV; (2) validity and
# uniqueness, the columns options A of tab:guidance / tab:recent move out.
APP_BLOCKS = {"app_cov": (("ib_dec", "IB$\\uparrow$"), ("mol_stability", "Mol. stab.$\\uparrow$"),
                          ("unique_valid_per_sample", "DV$\\uparrow$")),
              "app_chem": (("validity", "Validity$\\uparrow$"), ("uniqueness_of_valid", "Uniqueness$\\uparrow$"))}


def tex_appendix(F, which):
    keys = APP_BLOCKS[which]
    nc = 2 + 3 * len(keys)
    L = [TABSEP + r"\begin{tabular}{ll%s}" % ("c" * 3 * len(keys)), r"\toprule",
         " & & " + " & ".join(r"\multicolumn{3}{c}{%s}" % h for _k, h in keys) + r"\\",
         "".join(r"\cmidrule(lr){%d-%d}" % (3 + 3 * i, 5 + 3 * i) for i in range(len(keys))),
         "Backend & Arm & " + " & ".join(r"$\mu$ & $\alpha$ & gap" for _ in keys) + r"\\"]
    assert L[-1].count("&") == nc - 1
    for be in ("fm", "equifm", "edm"):
        L.append(r"\midrule")
        H = F.h[be]
        first = True
        rows = [(APP_NAMES[a], H, a) for a in HEAD_ARMS if ("mu", a) in H]
        if be != "edm":
            rows.append((r"plug-in, $w{=}4$", F.a[(be, 4)], "bdg_e0t1"))
        for name, src, arm in rows:
            L.append("%s & %s & %s\\\\" % (
                BE_SHORT[be] if first else "", name,
                " & ".join(" & ".join(f3(src[(p, arm)][k]) for p in PROPS) for k, _h in keys)))
            first = False
    L += [r"\bottomrule", r"\end{tabular}"]
    return NL.join(L)


def sec_blocks(A, F):
    A("### 7.2 Appendix bodies: the columns and backends the options A move out of the main text")
    A("")
    A("Two tables for all three QM9 backends at w = 1, plus the w = 4 plug-in rows (the ablation's `bdg_e0t1` at w = 4, another stage tree). The first carries decoded IB, molecule stability and DV per property; the second carries validity and uniqueness, so that together they hold every column option A of `tab:guidance` and `tab:recent` removes. Rows compare only within a backend (caveat 4); EquiFM carries caveat 5; EDM is external (caveat 6). %s" % (
        ("Widths at the paper's `\\small`: %s (text width %.1f pt; an appendix table wider than that needs `\\footnotesize` or a split)." % (
            "; ".join("%s %.1f pt" % (k, LXG["extra"][k][0]) for k in APP_BLOCKS), LXG["textwidth"]))
        if LXG else "Widths are measured only with `--latex-check`."))
    for k, title in (("app_cov", "IB, molecule stability, DV"), ("app_chem", "validity, uniqueness")):
        A("")
        A("*%s:*" % title)
        A("")
        A("```latex")
        A(tex_appendix(F, k))
        A("```")
    A("")
    A("## 8. The full metric block behind every M1 number above")
    A("")
    A("Headline, w = 1, N = 6000 per arm. `dec` is the paper's IB; `cont` is in-band on the continuous types; `o2` is the second oracle (OC-Flow, its own delta) on decoded types. MAE, bias and sd are divided by that backend's delta (continuous). **sd/d is the exact pooled residual sd**, `sqrt(mean(prop_rmse_eval^2) - bias^2)` over the 3 seed cells; V3_RESULTS squares the mean rmse instead (section 10), so its sd/d can differ from these in the third decimal (%d of the %d rows below). R is the decoded sd over unguided's. DV is distinct-valid per attempt. Clip %% uses the corrected denominator (caveat 7). Useful yield: **pending** (caveat 8)." % (
        sum(1 for be in ("fm", "equifm", "edm") for k, s in F.h[be].items()
            if "%.3f" % (s["sd"] / s["delta"]) != "%.3f" % (s["sd_exact"] / s["delta"])),
        sum(len(F.h[be]) for be in ("fm", "equifm", "edm"))))
    A("")
    for be in ("fm", "equifm", "edm"):
        H = F.h[be]
        for p in PROPS:
            u = H[(p, "unguided")]
            A("*%s, %s* (delta %.5f; second-oracle delta %.5f)%s" % (BE_LABEL[be], p, u["delta"], u["o2_delta"],
                                                                     " - caveat 5 applies" if be == "equifm" else ""))
            A("")
            A("| arm | IB dec | +-se | IB cont | IB o2 | MAE/d | bias/d | sd/d | R | mol. stab. | atom stab. | validity | uniq. | DV | clip % | w_eff (3-seed) | Time s/1k |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for arm in HEAD_ARMS:
                if (p, arm) not in H:
                    continue
                s = H[(p, arm)]
                d = s["delta"]
                A("| `%s` | %s | %s | %s | %s | %.3f | %+.3f | %.3f | %.3f | %s | %s | %s | %s | %s | %.2f | %s | %.0f |" % (
                    arm, f4(s["ib_dec"]), f4(s["se_ib_dec"]), f4(s["ib"]), f4(s["o2_dec"]),
                    s["prop_mae_eval"] / d, s["bias"] / d, s["sd_exact"] / d, s["sd_dec"] / u["sd_dec"],
                    f4(s["mol_stability"]), f4(s["atom_stability"]), f4(s["validity"]),
                    f4(s["uniqueness_of_valid"]), f4(s["unique_valid_per_sample"]), 100 * s["clip_frac"],
                    "%.2f" % s["weff3"] if "weff3" in s else "--", mean(s["s1k"])))
            A("")


def sec_h2h(A, F, X):
    A("## 9. Head-to-heads at the pre-registered bar (M1, headline, w = 1)")
    A("")
    A("Every guided arm against unguided, and each BDG arm against plug (BDG's eta = 0 limit). Unpaired binomial z, \\|z\\| >= %.2f. Differences in percentage points. Chemistry differences are printed beside every in-band contrast." % Z_BAR)
    A("")
    for be in ("fm", "equifm", "edm"):
        H = F.h[be]
        A("*%s*%s" % (BE_LABEL[be], (" - caveat 5: re-run noise up to %.2f pp is not in these se" % (100 * X["rerun"]["equifm"]["ib"])) if be == "equifm" else ""))
        A("")
        A("| prop | contrast | d IB dec | z | verdict | d IB cont | z | d IB o2 | z | d mol. stab. | d validity | d DV |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in PROPS:
            u = H[(p, "unguided")]
            pairs = [(a, "unguided") for a in GUIDED if (p, a) in H] + \
                    [(a, "plug") for a in BDG_HEAD if (p, a) in H]
            for a, b in pairs:
                s, r = H[(p, a)], H[(p, b)]
                zd = zun(s["ib_dec"], r["ib_dec"], s["N"])
                zc = zun(s["ib"], r["ib"], s["N"])
                zo = zun(s["o2_dec"], r["o2_dec"], s["N"])
                A("| %s | `%s` - `%s` | %s | %+.2f | %s | %s | %+.2f | %s | %+.2f | %s | %s | %s |" % (
                    p, a, b, pp(s["ib_dec"] - r["ib_dec"]), zd, verdict(zd), pp(s["ib"] - r["ib"]), zc,
                    pp(s["o2_dec"] - r["o2_dec"]), zo, pp(s["mol_stability"] - r["mol_stability"]),
                    pp(s["validity"] - r["validity"]), pp(s["unique_valid_per_sample"] - r["unique_valid_per_sample"])))
        A("")
    # verdict census
    A("Census of the BDG-vs-plug contrasts (fm and EquiFM, 3 properties each):")
    A("")
    A("| arm vs plug | decoded: above / below / tie | continuous | second oracle (decoded) | mean d mol. stab. vs unguided (pp), BDG / plug, same 6 cells |")
    A("|---|---|---|---|---|")
    for a in BDG_HEAD:
        cnt = {"dec": [0, 0, 0], "cont": [0, 0, 0], "o2": [0, 0, 0]}
        dms, dmp = [], []
        for be in ("fm", "equifm"):
            H = F.h[be]
            for p in PROPS:
                s, r, u = H[(p, a)], H[(p, "plug")], H[(p, "unguided")]
                for k, key in (("dec", "ib_dec"), ("cont", "ib"), ("o2", "o2_dec")):
                    v = verdict(zun(s[key], r[key], s["N"]))
                    cnt[k][("above", "below", "tie").index(v)] += 1
                dms.append(s["mol_stability"] - u["mol_stability"])
                dmp.append(r["mol_stability"] - u["mol_stability"])
        A("| `%s` | %s | %s | %s | %s / %s |" % (a, " / ".join(map(str, cnt["dec"])), " / ".join(map(str, cnt["cont"])),
                                             " / ".join(map(str, cnt["o2"])), pp(mean(dms)), pp(mean(dmp))))
    A("")


def sec_check(A, X, arm_census):
    rep = X["docs"]
    A("## 10. Independent check against docs/results/V3_RESULTS*.md")
    A("")
    A("Every number in the metric, contrast, controller-state and clip tables of %s was parsed and compared with this script's recomputation from the cells, at the doc's printed precision (a mismatch is a difference larger than half a unit in the last printed digit)." % ", ".join("`%s`" % f.split("/")[-1] for f in rep["files"]))
    A("")
    A("- Values compared: **%d**. Mismatches: **%d**." % (rep["n"], len(rep["bad"])))
    for b in rep["bad"][:40]:
        A("  - %s" % b)
    if len(rep["bad"]) > 40:
        A("  - ... and %d more" % (len(rep["bad"]) - 40))
    A("- GPU-hour and cell-count lines: %s." % ("all agree" if not rep["gpuh_bad"] else "; ".join(rep["gpuh_bad"])))
    A("- **The \"BDG controller state\" tables report seed 20261001's cell only, not a 3-seed mean.** `v3_table.pooled()` reads `rows[0]['diag']`. All %d printed controller values match the first seed; **%d** of them differ from the 3-seed mean at the printed precision. The paper should quote 3-seed means (sections 4 and 8 here use them)." % (rep["weff_first"], len(rep["weff_mean_bad"])))
    if rep["weff_mean_bad"]:
        worst = sorted(rep["weff_mean_bad"], key=lambda t: -abs(float(t[4]) - t[5]))[:5]
        A("  - largest: " + "; ".join("%s %s/%s/%s doc %s vs 3-seed %.3f" % (f.split("/")[-1], be, p, a, t, v)
                                       for f, be, p, a, t, v in worst))
    A("- **sd/d is computed from the mean rmse** (`pooled()` squares the mean rmse; the pooled variance needs the mean of the squares). The exact pooled sd changes the printed third decimal in %d of the rows compared (largest relative difference %.1e); no conclusion moves." % (rep["sd_digit"], rep["sd_rel"]))
    A("- **Significance bar.** The V3_RESULTS docs print a bar of z = 3.51 (Bonferroni over the 6 arms of one property at a 3-sigma family rate), and their \"arms clearing it\" use continuous in-band. The pre-registered bar is %.2f over 18 contrasts; the paper's IB is decoded. Contrasts whose continuous-IB verdict differs between the two bars: %s." % (
        Z_BAR, "none" if not rep["bar_diff"] else "; ".join(rep["bar_diff"])))
    A("- **Clip counts** in V3_RESULTS are raw sums over seeds and are correct; only `v3_sanity.py`'s fraction is 4x low (caveat 7).")
    A("- M2's `M2_RESULTS_gc.txt` pools unguided's w = 1 and w = 4 cells (identical samples) as 6 seeds, which understates its seed se by about 37 %% (0.0035 printed; 3 distinct seeds give %.4f). Its other rows use the 3-seed se." % X["m2_ung_se3"])
    A("")


def sec_latex(A, LX):
    A("## 12. Length check of the proposed bodies")
    A("")
    if LX is None:
        A("Not run in this generation. Re-run with `--latex-check` (needs pdflatex, bibtex and pypdf) to measure every tabular and compile the paper with each option substituted.")
        A("")
        return
    tw = LX["textwidth"]
    A("Measured with pdflatex on a temporary copy of `paper/` (deleted afterwards): each tabular boxed at the paper's `\\small`, text width %.1f pt, text height %.1f pt. Then the full paper compiled four times: v4 with its pending tables, v4 with every option A (tabular + proposed caption), v4 with every option B, and the live `paper/main.tex` as found (section 1b). Prose edits of section 11 are **not** included in the A and B variants." % (tw, LX["textheight"]))
    A("")
    base = LX["compile"]["current"]
    cost = lambda c: end_moved(LX, c)  # noqa: E731

    A("Two different quantities are reported and must not be confused. **Own change** is the height the option adds to the paper: its tabular plus its caption, measured in boxes against v4's (the caption set at `\\small` in the text width with a `Table 9:` prefix, which approximates the class's caption layout). **End moved** is how far the end of the main text moved in the compiled paper; it includes any float LaTeX pushed to another page, so it is set by float placement at least as much as by the option, and several variants land on the identical spill position.")
    A("")
    A("| table | option | tabular width (pt) | fits the text width | tabular height change (pt) | caption height change (pt) | own change (pt, ~lines) | paper with only this change: main-text pages (pagecheck) | probe: last page, space left (pt) | end moved (pt), float placement included |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for nm, (wd, ht, dp) in LX["measure"].items():
        lab, var = nm.split("|", 1)
        if lab == "current":
            continue
        dt, dc = own_change(LX, lab, var)
        c = LX["compile"].get(nm)
        k = cost(c)
        A("| `%s` | %s | %.1f | %s | %+.1f | %+.1f | %+.0f (%+.1f) | %s | %s | %s |" % (
            lab, var, wd, "yes" if wd <= tw else "**no, overfull**", dt, dc, dt + dc, (dt + dc) / 11.0,
            ("%d" % c["main"] if c["main"] <= 5 else "**%d**" % c["main"]) if c else "?",
            "page %d, %.1f" % (c["end_page"], c["slack"]) if c else "?",
            "%+.0f" % k if k is not None else "?"))
    A("")
    A("The v4 main text ends on page %d with %.1f pt to spare, so **any change costing more than that pushes it to page %d**. The pagecheck column is the criterion. The probe (`\\pagegoal - \\pagetotal` where the main text ends) can lag the page builder and ignores glue that can still shrink, so a small negative value may fit or not." % (
        base["end_page"], base["slack"], base["end_page"] + 1))
    fits = [nm for nm, c in LX["compile"].items() if "|" in nm and c and c["main"] <= 5]
    own = {o: [own_change(LX, lab, o) for lab in LX["labs"]] for o in ("A", "B")}
    A("")
    for o in ("A", "B"):
        dts, dcs = [a for a, _b in own[o]], [b for _a, b in own[o]]
        c = LX["compile"][o]
        A("- **All five option %s together**: own change %+.1f pt (~%+.1f lines; tabulars %+.1f, captions %+.1f). Compiled together, the main text runs to %d pages and its end moves by %+.0f pt; the %+.0f pt between the two is float placement (a float that no longer fits is carried to the next page), not text to cut." % (
            o, sum(dts) + sum(dcs), (sum(dts) + sum(dcs)) / 11.0, sum(dts), sum(dcs), c["main"], cost(c),
            cost(c) - sum(dts) - sum(dcs)))
    A("")
    A("**Budget.** Options that fit on their own: %s. The v4 main text has %.1f pt of slack, so any set whose own change exceeds it needs that much prose cut; beyond that, what must be recovered is the float that moved to page %d, which depends on where LaTeX places it, not a count of lines. The prose edits of section 11 change all of this again, so re-measure after them." % (
        ", ".join("`%s` %s" % tuple(nm.split("|")) for nm in fits) or "none", base["slack"], base["end_page"] + 1))
    if LX.get("extra"):
        A("")
        A("Appendix bodies (section 7.2), measured for width only: %s; text width %.1f pt." % (
            "; ".join("`%s` %.1f pt (%s)" % (k, v[0], "fits" if v[0] <= tw else "**overfull at `\\small`**")
                      for k, v in LX["extra"].items()), tw))
    A("")
    A("| paper variant | total pages | main text pages (pagecheck rule) | last main-text page | space left on it (pt) | end moved vs v4 (pt), float placement included | overfull boxes | undefined refs |")
    A("|---|---|---|---|---|---|---|---|")
    names = {"current": "v4, tables pending", "A": "v4 + all option A", "B": "v4 + all option B",
             "live": "live `paper/main.tex` as found"}
    for v, c in LX["compile"].items():
        if v not in names:
            continue
        if c is None:
            A("| %s | did not compile | | | | | | |" % names[v])
            continue
        k = cost(c) if v != "live" else None
        A("| %s | %d | %s | %s | %s | %s | %d | %d |" % (
            names[v], c["pages"], c["main"], c["end_page"],
            "%.1f" % c["slack"] if c["slack"] is not None else "?",
            "%+.0f" % k if k is not None else "--", c["overfull"], c["undefined"]))
    A("")
    lv = LX["compile"].get("live")
    A("The limit is 5 main-text pages. \"Space left\" is `\\pagegoal - \\pagetotal` at the end of the main text; floats that LaTeX defers can make the page count jump rather than slide, so a variant that fits here can still overflow after the prose edits.%s" % (
        (" **The live `paper/main.tex`, as found, runs to %s main-text pages under pagecheck's rule.**" % lv["main"])
        if lv and lv["main"] and lv["main"] > 5 else ""))
    A("")


# ------------------------------------------------------------------ the checks
def cross_checks(F, M1, M2):
    X = {}
    X["eta0"], X["rerun"], X["rerun_pp"] = {}, {}, {}
    for be in ("fm", "equifm"):
        h, a1 = M1[(be, "v3", 1)], M1[(be, "v3abl", 1)]
        rr = {"ib": 0.0, "ib_dec": 0.0}
        for p in PROPS:
            e0, pl = a1[(p, "bdg_e0t1")], h[(p, "plug")]
            X["eta0"][(be, p)] = {
                "ib": max(abs(x["in_band_fraction"] - y["in_band_fraction"]) for x, y in zip(e0, pl)),
                "ib_dec": max(abs(x["in_band_fraction_dec"] - y["in_band_fraction_dec"]) for x, y in zip(e0, pl)),
                "ms": max(abs(x["mol_stability"] - y["mol_stability"]) for x, y in zip(e0, pl))}
            for arm in BDG_HEAD:
                d = max(abs(x["in_band_fraction"] - y["in_band_fraction"]) for x, y in zip(a1[(p, arm)], h[(p, arm)]))
                dd = max(abs(x["in_band_fraction_dec"] - y["in_band_fraction_dec"]) for x, y in zip(a1[(p, arm)], h[(p, arm)]))
                X["rerun_pp"][(be, p, arm)] = d
                rr["ib"] = max(rr["ib"], d)
                rr["ib_dec"] = max(rr["ib_dec"], dd)
        X["rerun"][be] = rr
    if X["rerun"]["fm"]["ib"] != 0 or any(X["eta0"][("fm", p)]["ib"] != 0 for p in PROPS):
        die("fm is no longer bit-exact on re-runs / eta = 0; the caveats written here would be false")
    # the grid
    X["grid"] = {}
    cnt = {"n": 0, "rho_pos": 0, "rho_min": 9, "rho_max": -9, "lo_sig": 0, "hi_sig": 0,
           "hi_sig_w1": 0, "lo_sig_c": 0, "hi_sig_c": 0, "hi_sig_w4": 0, "wmin": 1e9, "wmax": -1e9, "dmin_all": 1e9, "dmax_all": -1e9}
    for be in ("fm", "equifm"):
        for w in (1, 4):
            A_ = F.a[(be, w)]
            for p in PROPS:
                e0 = A_[(p, "bdg_e0t1")]
                u = F.h[be][(p, "unguided")]
                rungs = ABL_ARMS[1:]
                ds = [100 * (A_[(p, a)]["ib_dec"] - e0["ib_dec"]) for a in rungs]
                we = [A_[(p, a)]["weff3"] for a in rungs]
                rho = spearman(we, ds)
                hi, lo = A_[(p, "bdg_e8t0.5")], A_[(p, "bdg_e8t1.5")]
                g = {"e0": e0, "dmin": min(ds), "dmax": max(ds), "rho": rho,
                     "d_hi": 100 * (hi["ib_dec"] - e0["ib_dec"]), "z_hi": F.zdec(hi, e0),
                     "d_lo": 100 * (lo["ib_dec"] - e0["ib_dec"]), "z_lo": F.zdec(lo, e0), "z_hi_c": zun(hi["ib"], e0["ib"], hi["N"]), "z_lo_c": zun(lo["ib"], e0["ib"], lo["N"]),
                     "hi_is_max": hi["weff3"] == max(we), "lo_is_min": lo["weff3"] == min(we),
                     "wmin": min(we), "wmax": max(we),
                     "Rmin": min(A_[(p, a)]["sd_dec"] / u["sd_dec"] for a in rungs),
                     "Rmax": max(A_[(p, a)]["sd_dec"] / u["sd_dec"] for a in rungs),
                     "msmin": min(A_[(p, a)]["mol_stability"] for a in rungs),
                     "msmax": max(A_[(p, a)]["mol_stability"] for a in rungs),
                     "cmin": min(100 * A_[(p, a)]["clip_frac"] for a in rungs),
                     "cmax": max(100 * A_[(p, a)]["clip_frac"] for a in rungs)}
                # V3_BLADE_READOUT.md's "tau_mult 0.5 row rises with eta": the four
                # eta > 0 deltas at tau 0.5 are non-decreasing and the first is >= 0
                for tag, key in (("c", "ib"), ("d", "ib_dec")):
                    r05 = [A_[(p, "bdg_e%dt0.5" % e)][key] - e0[key] for e in ABL_ETAS]
                    mono = all(r05[i + 1] >= r05[i] for i in range(3)) and r05[0] >= 0
                    cnt["mono_" + tag] = cnt.get("mono_" + tag, 0) + mono
                    cnt["mono_%s_w%d" % (tag, w)] = cnt.get("mono_%s_w%d" % (tag, w), 0) + mono
                X["grid"][(be, p, w)] = g
                cnt["n"] += 1
                cnt["rho_pos"] += rho > 0
                cnt["rho_min"] = min(cnt["rho_min"], rho)
                cnt["rho_max"] = max(cnt["rho_max"], rho)
                cnt["lo_sig"] += g["z_lo"] <= -Z_BAR
                cnt["lo_sig_c"] += g["z_lo_c"] <= -Z_BAR
                cnt["hi_sig_c"] += g["z_hi_c"] >= Z_BAR
                cnt["hi_sig"] += g["z_hi"] >= Z_BAR
                cnt["hi_sig_w%d" % w] += g["z_hi"] >= Z_BAR
                cnt["wmin"] = min(cnt["wmin"], min(we))
                cnt["wmax"] = max(cnt["wmax"], max(we))
                if be == "fm":
                    cnt["dmin_all"] = min(cnt["dmin_all"], min(ds))
                    cnt["dmax_all"] = max(cnt["dmax_all"], max(ds))
    for k in ("mono_c", "mono_d", "mono_c_w1", "mono_d_w1", "mono_c_w4", "mono_d_w4"):
        cnt.setdefault(k, 0)
    # the corners at the grid's selection-adjusted bar, and where the decoded
    # (paper's IB) and continuous readings disagree
    cnt["hi_sel"] = sum(g["z_hi"] >= Z_SEL for g in X["grid"].values())
    cnt["lo_sel"] = sum(g["z_lo"] <= -Z_SEL for g in X["grid"].values())
    cnt["hi_sel_w1"] = sum(g["z_hi"] >= Z_SEL for k, g in X["grid"].items() if k[2] == 1)
    cnt["hi_sel_w4"] = sum(g["z_hi"] >= Z_SEL for k, g in X["grid"].items() if k[2] == 4)
    cnt["lo_neg"] = sum(g["d_lo"] < 0 for g in X["grid"].values())
    cnt["hi_sig_eq"] = sum(g["z_hi"] >= Z_BAR for k, g in X["grid"].items() if k[0] == "equifm")
    cnt["lo_sig_eq"] = sum(g["z_lo"] <= -Z_BAR for k, g in X["grid"].items() if k[0] == "equifm")
    dc_max, dc_at, flips = 0.0, None, []
    for (be, p, w), g in X["grid"].items():
        A_ = F.a[(be, w)]
        e0 = A_[(p, "bdg_e0t1")]
        for a in ABL_ARMS[1:]:
            s = A_[(p, a)]
            dd, dc = 100 * (s["ib_dec"] - e0["ib_dec"]), 100 * (s["ib"] - e0["ib"])
            if abs(dd - dc) > dc_max:
                dc_max, dc_at = abs(dd - dc), (be, p, w, a, dd, dc)
            if a in ("bdg_e8t0.5", "bdg_e8t1.5"):
                vd, vc = verdict(zun(s["ib_dec"], e0["ib_dec"], s["N"])), verdict(zun(s["ib"], e0["ib"], s["N"]))
                if vd != vc:
                    flips.append((be, p, w, a, dd, zun(s["ib_dec"], e0["ib_dec"], s["N"]), vd,
                                  dc, zun(s["ib"], e0["ib"], s["N"]), vc))
    cnt["dc_max"], cnt["dc_at"], cnt["flips"] = dc_max, dc_at, flips
    X["grid_counts"] = cnt
    # the clip at the grid's extremes: pooled over a rung's 3 seeds, and per cell
    ce = {"fm_mu_w4": {a: 100 * F.a[("fm", 4)][("mu", a)]["clip_frac"] for a in ("bdg_e0t1", "bdg_e8t0.5", "bdg_e8t1.5")},
          "pooled_max": {}, "cell_max": {}}
    for be in ("fm", "equifm"):
        best, bestc = (0.0, ""), (0.0, "")
        for w in (1, 4):
            for (p, a), rows in M1[(be, "v3abl", w)].items():
                v = 100 * F.a[(be, w)][(p, a)]["clip_frac"]
                if v > best[0]:
                    best = (v, "%s/%s/`%s`/w = %d" % (be, p, a, w))
                for r in rows:
                    c = 100 * r["clipped_sample_steps"] / ((r["guided_steps"] / (r["n"] / r["batch"])) * r["n"])
                    if c > bestc[0]:
                        bestc = (c, "%s/%s/`%s`/w = %d, seed %s" % (be, p, a, w, r["_seed"]))
        ce["pooled_max"][be], ce["cell_max"][be] = best, bestc
    X["clip_ext"] = ce
    # edm's checkpoint hash: what the cells record, against the local file
    em = {(r.get("prov") or {}).get("edm_md5") for rs in M1[("edm", "v3", 1)].values() for r in rs}
    if len(em) != 1 or None in em:
        die("edm cells disagree on, or lack, prov.edm_md5: %s" % em)
    em = em.pop()
    fp = os.path.join(ROOT, "weights", "EDMsecond", "generative_model_ema.npy")
    X["edm_md5"] = {"cells": em, "note": ("; the local file has the same md5" if os.path.isfile(fp) and md5(fp) == em
                                          else "; **the local file's md5 differs**" if os.path.isfile(fp)
                                          else "; no local copy to check")}
    # measured cost per stage, from the cells' own `seconds`
    X["gpuh"] = {"v3": sum(r["seconds"] for be in ("fm", "equifm", "edm")
                           for rs in M1[(be, "v3", 1)].values() for r in rs) / 3600.0,
                 "v3abl": sum(r["seconds"] for be in ("fm", "equifm") for w in (1, 4)
                              for rs in M1[(be, "v3abl", w)].values() for r in rs) / 3600.0}
    # BDG against plug, decoded, unpaired: above / below / tie over the 12 contrasts
    vb = [0, 0, 0]
    for be in ("fm", "equifm"):
        for p in PROPS:
            for a in BDG_HEAD:
                v = verdict(F.zdec(F.h[be][(p, a)], F.h[be][(p, "plug")]))
                vb[("above", "below", "tie").index(v)] += 1
    X["bdg_vs_plug_dec"] = vb
    # M2
    ung1 = m2_rows(M2, "m2", "gc", "unguided", 1)
    same = 0.0
    for prop in M2_PROPS:
        a, b = m2_rows(M2, "m2", prop, "unguided", 1), m2_rows(M2, "m2", prop, "unguided", 4)
        same = max(same, m2_maxdiff(a, b))
    X["m2_ung_same"] = same
    if same != 0:
        die("M2 unguided w=1 and w=4 cells differ (%g); they cannot be merged" % same)
    X["m2_ung_se3"] = sdev(r["in_band_fraction"] for r in ung1) / math.sqrt(3)
    trio = 0.0
    for prop in M2_PROPS:
        for w in (1, 4):
            pl = m2_rows(M2, "m2", prop, "plug", w)
            for arm in ("tmpd", "lgd_mc"):
                trio = max(trio, m2_maxdiff(pl, m2_rows(M2, "m2", prop, arm, w)))
    X["m2_same"] = trio
    X["m2_e0"] = {w: m2_maxdiff(m2_rows(M2, "m2abl", "gc", "bdg_e0t1", w), m2_rows(M2, "m2", "gc", "plug", w))
                  for w in (1.0, 4.0)}
    X["m2_rerun"] = max(m2_maxdiff(m2_rows(M2, "m2abl", "gc", a, w), m2_rows(M2, "m2", "gc", a, w))
                        for a in BDG_HEAD for w in (1.0, 4.0))

    # the same comparisons over EVERY recorded metric, not only the six fields:
    # in-band, the six fields, bias/delta, and the clipped sample-step count
    def full_diff(pairs):
        out = {"ib": 0.0, "fields": 0.0, "bias": 0.0, "clip_max": 0, "clip_cells": 0, "n": 0}
        for ra, rb in pairs:
            for x, y in zip(ra, rb):
                out["n"] += 1
                out["ib"] = max(out["ib"], abs(x["in_band_fraction"] - y["in_band_fraction"]))
                out["fields"] = max(out["fields"], max(abs(x[k] - y[k]) for k in M2_FIELDS if k != "in_band_fraction"))
                out["bias"] = max(out["bias"], abs(x["bias_delta"] - y["bias_delta"]))
                dcl = abs(x["clipped_sample_steps"] - y["clipped_sample_steps"])
                out["clip_max"] = max(out["clip_max"], dcl)
                out["clip_cells"] += dcl > 0
        return out
    X["m2_rerun_full"] = full_diff([(m2_rows(M2, "m2abl", "gc", a, w), m2_rows(M2, "m2", "gc", a, w))
                                    for a in BDG_HEAD for w in (1.0, 4.0)])
    X["m2_e0_full"] = {w: full_diff([(m2_rows(M2, "m2abl", "gc", "bdg_e0t1", w), m2_rows(M2, "m2", "gc", "plug", w))])
                       for w in (1.0, 4.0)}
    X["m2_hours"] = {st: sum(r["minutes"] for k, per in M2.items() if k[0] == st for r in per.values()) / 60.0
                     for st in ("m2", "m2abl", "m2wsweep")}
    X["m2_ncells"] = sum(len(v) for v in M2.values())
    best_in = max(F.m2s("gc", a, w, stage="m2abl")["ib"] for a in ABL_ARMS for w in (1, 4))
    best_head = max(F.m2s("gc", a, w)["ib"] for a in M2_ARMS for w in (1, 4))
    X["m2win"] = {"plug_t0": F.m2s("gc", "plug", 1, "m2abl", 0.0)["ib"],
                  "plug_t05": F.m2s("gc", "plug", 1)["ib"],
                  "best_in": max(best_in, best_head), "ung": F.m2s("gc", "unguided", 1)["ib"]}
    if abs(X["m2win"]["best_in"] - F.m2s("gc", "bdg_e8t0.5", 4, stage="m2abl")["ib"]) > 1e-12:
        die("the best in-window M2 cell is no longer bdg_e8t0.5 at w = 4; update the caveat text")
    return X


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split(NL)[0])
    ap.add_argument("--md-out", required=True)
    ap.add_argument("--latex-check", action="store_true",
                    help="measure the proposed tabulars and compile the paper with them (pdflatex)")
    args = ap.parse_args()

    src, _lines, T = paper_tables()
    check_paper(T)
    M1 = load_m1()
    census = m1_arm_census()
    if any(a.endswith("o") or "replay" in a or "signed" in a for a in census):
        die("a one-sided / replay / signed arm now exists in the v3 trees (%s); "
            "tab:ablation's 'cannot be filled' text is stale" % census)
    M2, m2_extra = load_m2()
    F = Fill(M1, M2)
    R = fmvd_rows(F)
    X = cross_checks(F, M1, M2)
    X["arm_census"] = census
    with open(os.path.join(ROOT, "results", "m2_share.json"), encoding="utf-8") as fh:
        sh = json.load(fh)
    if set(sh["by_prop"]) != {"gc"}:
        die("results/m2_share.json now covers %s; the doc says it measures gc only" % sorted(sh["by_prop"]))
    gs = sh["by_prop"]["gc"]
    X["m2_share"] = {"plug": gs["plug"]["w1"]["share"], "ratio": gs["bdg_e4t0.5"]["w1"]["share"] / gs["plug"]["w1"]["share"],
                     "n": sh["n"], "seed": sh.get("seed"),
                     "r": {(a, w): gs[a]["w%d" % w]["share"] / gs["plug"]["w%d" % w]["share"]
                           for a in BDG_HEAD for w in (1, 4)},
                     "plug4": gs["plug"]["w4"]["share"], "t05_4": gs["bdg_e4t0.5"]["w4"]["share"]}
    X["m2_func"] = m2_functional(M2)
    # the live file: read ONCE, as bytes, so the md5 is the one `md5sum` prints
    # (the file is CRLF; a hash of the text-mode string would match nothing a
    # teammate can check) and the audit and the page check see the same snapshot
    # of a file other revisions are editing while this runs
    import datetime
    mtime = os.path.getmtime(PAPER_LIVE)
    with open(PAPER_LIVE, "rb") as fh:
        live_raw = fh.read()
    live_src = live_raw.decode("utf-8").replace("\r\n", "\n")
    _s, _l, TL = paper_tables(PAPER_LIVE, strict=False, src=live_src)
    X["is_v4"] = live_src == src
    LIVE.update({
        "src": live_src, "T": TL,
        "md5": hashlib.md5(live_raw).hexdigest(),
        "mtime": datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "revs": sorted(os.path.relpath(f, ROOT).replace(os.sep, "/")
                       for f in glob.glob(os.path.join(ROOT, "paper", "_rev*.py")))})
    TS = training_sources(M1)
    LIVE["audit"], LIVE["skipped"] = audit_live(F, R, {k: v for k, v in TL.items() if v}, TS.get("fm_params"))
    LIVE["claims"] = claim_checks(F, X, live_src)
    X["docs"] = crosscheck_docs(M1)
    bodies = {"tab:fmvd": {"A": tex_fmvd(F, R, "A"), "B": tex_fmvd(F, R, "B")},
              "tab:guidance": {"A": tex_guidance(F, "A"), "B": tex_guidance(F, "B")},
              "tab:ablation": {"A": tex_ablation(F, "A"), "B": tex_ablation(F, "B")},
              "tab:recent": {"A": tex_recent(F, "A"), "B": tex_recent(F, "B")},
              "tab:m2": {"A": tex_m2(F, "A"), "B": tex_m2(F, "B")}}
    caps = captions(X, bodies)
    CAPS.update(caps)
    extra = {k: tex_appendix(F, k) for k in APP_BLOCKS}
    LX = latex_check(T, bodies, caps, extra) if args.latex_check else None
    if LX:
        LXG.update(LX)
    write_doc(args, T, src, F, M2, m2_extra, R, TS, X, LX, census)


if __name__ == "__main__":
    main()
