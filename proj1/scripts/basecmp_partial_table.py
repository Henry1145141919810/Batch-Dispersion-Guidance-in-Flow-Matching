"""Tabulate the CANCELLED base-model comparison (results/basecmp) as a PARTIAL record.

    python proj1/scripts/basecmp_partial_table.py \
        --md-out docs/results/BASECMP_PARTIAL_RESULTS.md

WHY THIS EXISTS. The basecmp chain (BASECMP_PROTOCOL.md) was cancelled on 26 Sep
2026 part-way through its screen, because protocol v3 superseded it. Its two
official readers cannot report it, by design:

  * `basecmp_table.py` reads only the full run (`<base>/full/n*/seed*`), which
    never started, so it finds no cells.
  * `basecmp_freeze.py` refuses a screen with holes, because its picks are an
    argmax and a hole is invisible in an argmax.

Both refusals are correct, so this script does NOT pick and does NOT issue
verdicts. It prints what exists, cell by cell, with the full metric block and
its standard errors, and the like-for-like base comparison on the cells both
bases actually ran. Everything it writes is labelled PARTIAL.

READ-ONLY on results/ and logs/. It writes only the file named by --md-out
(and --csv-out when given).

WHAT IT COMPUTES
  * how far the chain got: stage by stage from the job logs, and cell coverage
    against the pre-registered 363 per base;
  * plan.json's measured cost model, the cost surface the sizer printed, and
    the screen's measured cost against that model's prediction;
  * the probe (n = 256, mu, w = 1) on both bases, plus the 10 EquiFM probe cells
    that two concurrent jobs computed twice (a measurement of run-to-run
    nondeterminism at a fixed seed);
  * per base, per property, per arm, per (w, t_start): in-band continuous and
    decoded (+- binomial se), MAE / delta, bias / delta and residual sd / delta
    (continuous and decoded; over finite rows, population sd -- the repo's
    convention), molecule stability against the chemistry floor, validity,
    uniqueness, and oracle-2 in-band;
  * the matched base comparison: for every (property, arm, w, t_start) both
    bases ran, EquiFM minus ours on in-band and chemistry with an UNPAIRED z
    (the two generators share molecule sizes but no molecules).

THE CHEMISTRY FLOOR FOR alpha AND gap IS BORROWED, and says so everywhere. Only
mu's unguided cell ran on either base (the alpha and gap unguided cells sat in
array tasks that never started). The unguided sampler reads no property --
`f_net=None`, sizes `val[:n]`, seed 20260925 -- so its molecules are the same
for every property up to GPU nondeterminism, and mu's unguided molecule
stability is that base's unguided chemistry. Unguided IN-BAND for alpha and gap
is a different matter: it needs the alpha / gap oracle on unguided molecules,
which were never scored, and it is not computable from this tree.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from transfer_sweep import (BACKENDS, BASECMP_FLOOR, BASECMP_ROOT,  # noqa: E402
                            BASECMP_SEED, BASECMP_STRENGTHS, BASECMP_T_STARTS)

BASES = ("fm", "equifm")
LABEL = {"fm": "ours (FM)", "equifm": "EquiFM"}
PROPS = ("mu", "alpha", "gap")
GUIDED = ("plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var")
UNIT = {"mu": "D", "alpha": "Bohr^3", "gap": "Ha"}
Z_MARK = 3.0


# ---------------------------------------------------------------- statistics
def se_prop(p, n):
    p = min(max(float(p), 0.0), 1.0)
    return math.sqrt(p * (1.0 - p) / n) if n else float("nan")


def z_unpaired(p1, n1, p2, n2):
    """(p2 - p1) / sqrt(se1^2 + se2^2): independent samples."""
    s = math.sqrt(se_prop(p1, n1) ** 2 + se_prop(p2, n2) ** 2)
    return (p2 - p1) / s if s > 0 else float("nan")


def f(x, d=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return "%.*f" % (d, x)


def fz(z):
    return "n/a" if not math.isfinite(z) else "%+.2f" % z


# ---------------------------------------------------------------- cells
def sidecar_stats(path, delta):
    """bias and residual sd of (f_B - y), continuous and decoded, over FINITE
    rows, population sd (unbiased=False), in units of delta. None if absent."""
    if not os.path.exists(path):
        return None
    import torch
    d = torch.load(path, weights_only=False)
    fin = d["finite"].bool()
    out = {"n_finite": int(fin.sum()), "mol_idx": d["mol_idx"].clone(),
           "n_atoms": d["n_atoms"].clone()}
    for key, tag in (("f_B", ""), ("f_B_dec", "_dec")):
        r = (d[key].double() - d["y"].double())[fin]
        out["bias" + tag] = float(r.mean()) / delta
        out["sd" + tag] = float(r.std(unbiased=False)) / delta
        out["inband_recomputed" + tag] = float((r.abs() <= delta).sum()) / len(fin)
    return out


def agg_stats(r, delta):
    """bias and residual sd from the aggregate block alone: bias = mean f_B -
    target, sd = sqrt(RMSE^2 - bias^2). Used where no sidecar exists (probe)."""
    out = {}
    for mean_k, rmse_k, tag in (("f_B_mean", "prop_rmse_eval", ""),
                                ("f_B_dec_mean", "prop_rmse_eval_dec", "_dec")):
        if r.get(mean_k) is None or r.get(rmse_k) is None:
            out["bias" + tag] = out["sd" + tag] = float("nan")
            continue
        b = float(r[mean_k]) - float(r["target_mean"])
        rm = float(r[rmse_k])
        out["bias" + tag] = b / delta
        out["sd" + tag] = math.sqrt(max(rm * rm - b * b, 0.0)) / delta
    return out


def load_cells(root, base, stage):
    cells = []
    for fn in sorted(glob.glob(os.path.join(root, base, stage, "tr__*.json"))):
        r = json.load(open(fn))
        if r.get("backend") != BACKENDS[base]:
            raise SystemExit("REFUSING: %s is a %r cell under %s/"
                             % (fn, r.get("backend"), base))
        dl = float(r["delta"])
        o2 = r.get("oracle2") or {}
        cal = r.get("calibration") or {}
        c = {"base": base, "stage": stage, "file": os.path.basename(fn),
             "prop": r["prop"], "arm": r["arm"], "w": float(r["w"]),
             "t": float(r["t_start"]), "n": int(r["n"]), "seed": int(r["seed"]),
             "delta": dl, "delta_global": cal.get("delta_global"),
             "delta_mode": cal.get("delta_mode"),
             "delta_detail": cal.get("delta_detail") or {},
             "delta2": o2.get("delta"), "delta2_mode": o2.get("delta_mode"),
             "ib": float(r["in_band_fraction"]),
             "ib_dec": float(r["in_band_fraction_dec"]),
             "mae": float(r["prop_mae_eval"]) / dl,
             "mae_dec": float(r["prop_mae_eval_dec"]) / dl,
             "stab": float(r["mol_stability"]), "valid": float(r["validity"]),
             "uniq": float(r["uniqueness_of_valid"]),
             "uvps": float(r["unique_valid_per_sample"]),
             "o2_ib": o2.get("in_band"), "o2_ib_dec": o2.get("in_band_dec"),
             "nonfinite": int(r.get("n_nonfinite", 0)),
             "seconds": float(r.get("seconds") or float("nan")),
             "clipped": int(r.get("clipped_sample_steps", 0)),
             "guided_steps": int(r.get("guided_steps", 0)),
             "f_B_mean": r.get("f_B_mean"), "target": float(r["target"]),
             "gen_md5": (r.get("prov") or {}).get("gen_md5"),
             "device": (r.get("prov") or {}).get("device"),
             "torch": (r.get("prov") or {}).get("torch"),
             "grid": r.get("grid"), "steps": r.get("steps"),
             "solver": r.get("solver"), "clip": r.get("clip"),
             "batch": r.get("batch"), "target_name": r.get("target_name"),
             "guide": r.get("guide"), "oracle": r.get("oracle"),
             "oracle2": o2.get("oracle2"), "w_scale": r.get("w_scale")}
        sc = sidecar_stats(fn[:-5] + ".permol.pt", dl)
        ag = agg_stats(r, dl)
        c["agg"] = ag
        if sc is not None:
            c.update({k: sc[k] for k in ("bias", "sd", "bias_dec", "sd_dec")})
            c["stat_src"] = "sidecar"
            c["_side"] = sc
        else:
            c.update(ag)
            c["stat_src"] = "aggregate"
        cells.append(c)
    return cells


# ---------------------------------------------------------------- logs
HDR = re.compile(r"^(host|job|array|stage|budget|started|finished)\s*:\s*(.*)$")
TASK = re.compile(r"^=== task (\d+) : (.*)$")
DONE = re.compile(r"^(\d+) cells this run, ([\d.]+) min; STAGE COMPLETE")
CELL = re.compile(r"^\s*\[\s*\d+/\s*\d+\]\s+(\w+)\s+(\w+)\s+q90\s+w=([\d.]+)\s+"
                  r"t=([\d.]+)\s+MAE\s+([\d.]+).*?in-band ([\d.]+) \(dec ([\d.]+)\)"
                  r"\s+mol-stab ([\d.]+)\s+clipped (\d+)\s+([\d.]+)s")
CANCEL = re.compile(r"CANCELLED AT (\S+)")


def arg_of(spec, name):
    m = re.search(r"--%s (\S+)" % name, spec)
    return m.group(1) if m else None


def parse_logs(logdir):
    jobs = []
    for fn in sorted(glob.glob(os.path.join(logdir, "basecmp-*.out"))):
        txt = open(fn, encoding="utf-8", errors="replace").read().splitlines()
        j = {"log": os.path.basename(fn), "cells_logged": [], "task_spec": None,
             "done_cells": None, "done_min": None, "already_complete": False}
        past_preflight = False
        for ln in txt:
            m = HDR.match(ln)
            if m and m.group(1) not in j:
                j[m.group(1)] = m.group(2).strip()
            m = TASK.match(ln)
            if m:
                j["task"] = int(m.group(1))
                j["task_spec"] = m.group(2)
            if "already complete" in ln:
                j["already_complete"] = True
            if ln.strip() == "preflight passed":
                past_preflight = True
            m = CELL.match(ln)
            if m and past_preflight:
                j["cells_logged"].append({
                    "prop": m.group(1), "arm": m.group(2), "w": float(m.group(3)),
                    "t": float(m.group(4)), "mae": float(m.group(5)),
                    "ib": float(m.group(6)), "ib_dec": float(m.group(7)),
                    "stab": float(m.group(8)), "clipped": int(m.group(9)),
                    "sec": float(m.group(10))})
            m = DONE.match(ln)
            if m:
                j["done_cells"], j["done_min"] = int(m.group(1)), float(m.group(2))
        err = fn[:-4] + ".err"
        j["cancelled_at"] = None
        if os.path.exists(err):
            m = CANCEL.search(open(err, encoding="utf-8", errors="replace").read())
            if m:
                j["cancelled_at"] = m.group(1)
        if j.get("task_spec"):
            j["backend"] = arg_of(j["task_spec"], "backend")
            j["n"] = int(arg_of(j["task_spec"], "n") or 0)
            j["props"] = (arg_of(j["task_spec"], "props") or "").split(",")
            j["arms"] = (arg_of(j["task_spec"], "arms") or "").split(",")
            j["t_starts"] = [float(x) for x in
                             (arg_of(j["task_spec"], "t-starts") or "").split(",") if x]
            j["strengths"] = arg_of(j["task_spec"], "strengths")
        m = re.match(r"(\d+) task (\S+)", j.get("array", ""))
        j["array_id"] = m.group(1) if m else None
        j["job"] = j.get("job")
        jobs.append(j)
    return jobs


def parse_size_log(path):
    """The cost surface and the CHOSEN block, exactly as the sizer printed them."""
    txt = open(path, encoding="utf-8", errors="replace").read().splitlines()
    surf, chosen, cols = {}, [], None
    for i, ln in enumerate(txt):
        if ln.strip().startswith("n_full ->"):
            cols = [int(x) for x in ln.split("->")[1].split()]
        m = re.match(r"^\s*n_screen\s+(\d+)\s+(.*)$", ln)
        if m and cols:
            surf[int(m.group(1))] = dict(zip(cols, [float(x) for x in m.group(2).split()]))
        if ln.startswith("CHOSEN"):
            chosen = [ln.strip()] + [x.strip() for x in txt[i + 1:i + 5]
                                     if x.startswith("  ")]
    return surf, cols, chosen


def fmt_ts(s):
    return (s or "").replace("T", " ")[:19]


def parse_ts(s):
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:                                          # noqa: BLE001
        return None


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=BASECMP_ROOT)
    ap.add_argument("--logs", default=os.path.join(ROOT, "logs"))
    ap.add_argument("--md-out", default="")
    ap.add_argument("--csv-out", default="")
    args = ap.parse_args()

    plan = json.load(open(os.path.join(args.root, "plan.json")))
    spm = plan["sec_per_molecule"]
    margin = float(plan["margin"])

    cells = {(b, s): load_cells(args.root, b, s)
             for b in BASES for s in ("probe", "screen")}
    jobs = parse_logs(args.logs)
    extra_dirs = {b: sorted(d for d in os.listdir(os.path.join(args.root, b))
                            if os.path.isdir(os.path.join(args.root, b, d)))
                  for b in BASES}
    failed = sorted(glob.glob(os.path.join(args.root, "*", "*", "*.failed")))

    L = []                                                   # the document
    A = L.append
    ctx_jdev, ctx_dup = {}, {}                               # filled by section 5
    P = []                                                   # stdout summary

    def say(s=""):
        P.append(s)
        print(s)

    # ---------------------------------------------------------- consistency
    meta = {}
    for (b, s), cs in cells.items():
        meta[(b, s)] = sorted({(c["n"], c["seed"], c["gen_md5"], c["batch"],
                                c["steps"], c["solver"], c["clip"], c["grid"],
                                c["target_name"], c["delta_mode"], c["delta2_mode"])
                               for c in cs})
        if len(meta[(b, s)]) != 1:
            raise SystemExit("INCONSISTENT %s/%s: %s" % (b, s, meta[(b, s)]))
    scr = {b: cells[(b, "screen")] for b in BASES}
    prb = {b: cells[(b, "probe")] for b in BASES}
    n_scr = {b: scr[b][0]["n"] for b in BASES}
    n_prb = {b: prb[b][0]["n"] for b in BASES}
    # the MIG slice each cell ran on: cost is not comparable across slice types
    devs = {(b, s): collections.Counter(c["device"] for c in cells[(b, s)])
            for b in BASES for s in ("probe", "screen")}

    # delta: one rule, identical on both bases up to float noise
    dtab = {}
    for p in PROPS:
        ds = [c["delta"] for b in BASES for c in scr[b] if c["prop"] == p]
        d2 = [c["delta2"] for b in BASES for c in scr[b] if c["prop"] == p]
        dg = [c["delta_global"] for b in BASES for c in scr[b] if c["prop"] == p]
        det = next(c["delta_detail"] for c in scr["fm"] if c["prop"] == p)
        dtab[p] = {"lo": min(ds), "hi": max(ds), "rel": (max(ds) - min(ds)) / max(ds),
                   "d2lo": min(d2), "d2hi": max(d2), "glo": min(dg), "ghi": max(dg),
                   "det": det}

    # sizes: every sidecar on both bases must carry the same val[:n] molecules
    ref_idx, idx_ok, n_side = None, True, 0
    for b in BASES:
        for c in scr[b]:
            sd = c.get("_side")
            if sd is None:
                continue
            n_side += 1
            if ref_idx is None:
                ref_idx = sd["mol_idx"]
            elif not bool((sd["mol_idx"] == ref_idx).all()):
                idx_ok = False
    # aggregate-derived bias/sd vs sidecar, so the probe's derived columns are
    # known to mean the same thing
    agg_err = max(max(abs(c["agg"][k] - c[k]) for k in ("bias", "sd", "bias_dec", "sd_dec"))
                  for b in BASES for c in scr[b] if c["stat_src"] == "sidecar")
    ib_err = max(abs(c["_side"]["inband_recomputed"] - c["ib"])
                 for b in BASES for c in scr[b] if c["stat_src"] == "sidecar")

    # ---------------------------------------------------------- coverage
    want = {(p, a, float(w), float(t)) for p in PROPS for a in GUIDED
            for w in BASECMP_STRENGTHS for t in BASECMP_T_STARTS}
    want |= {(p, "unguided", 1.0, 0.5) for p in PROPS}
    have = {b: {(c["prop"], c["arm"], c["w"], c["t"]) for c in scr[b]} for b in BASES}
    extra = {b: sorted(have[b] - want) for b in BASES}
    miss = {b: sorted(want - have[b]) for b in BASES}
    matched = sorted(have["fm"] & have["equifm"])

    # ---------------------------------------------------------- floor
    ung = {b: next(c for c in scr[b] if c["arm"] == "unguided" and c["prop"] == "mu")
           for b in BASES}
    floor = {b: BASECMP_FLOOR * ung[b]["stab"] for b in BASES}
    ung_p = {b: next(c for c in prb[b] if c["arm"] == "unguided") for b in BASES}
    floor_p = {b: BASECMP_FLOOR * ung_p[b]["stab"] for b in BASES}
    ung_have = {b: sorted(p for p in PROPS if (p, "unguided", 1.0, 0.5) in have[b])
                for b in BASES}
    for b in BASES:
        for c in scr[b]:
            c["clears"] = c["stab"] >= floor[b] - 1e-12
        for c in prb[b]:
            c["clears"] = c["stab"] >= floor_p[b] - 1e-12

    # feature-scale check (basecmp_freeze.scale_sanity, the same function)
    try:
        from basecmp_freeze import scale_sanity
        scale = {b: scale_sanity([{"prop": "mu", "arm": "unguided",
                                   "f_B_mean": ung[b]["f_B_mean"]}], ["mu"])
                 for b in BASES}
    except Exception as exc:                                    # noqa: BLE001
        scale = {b: ["(not computed: %r)" % exc] for b in BASES}

    # ---------------------------------------------------------- jobs
    by_stage = {}
    for j in jobs:
        by_stage.setdefault(j.get("stage"), []).append(j)
    screen_jobs = sorted(by_stage.get("screen", []), key=lambda j: j.get("task", -1))
    arrays = sorted({(j.get("stage"), j.get("budget"), j.get("array_id") or j["job"])
                     for j in jobs})
    starts = [parse_ts(j.get("started")) for j in screen_jobs]
    ends = []
    for j in screen_jobs:
        e = j.get("finished", "")
        e = parse_ts(e.split("rc=")[0].strip()) if e else None
        if e is None and j["cancelled_at"]:
            e = parse_ts(j["cancelled_at"] + "-04:00")
        ends.append(e)
    conc = 0
    ev = sorted([(s, 1) for s in starts if s] + [(e, -1) for e in ends if e],
                key=lambda x: (x[0], x[1]))
    cur = 0
    for _t, dlt in ev:
        cur += dlt
        conc = max(conc, cur)

    def pred_spm(b, arm, t):
        a, bb = spm[b][arm]
        return max(a + bb * (1.0 - t), 0.0)

    def task_pred_min(j):
        tot = 0.0
        for p in j["props"]:
            for arm in j["arms"]:
                if arm == "unguided":
                    tot += j["n"] * pred_spm(j["backend"], arm, 0.5)
                    continue
                for t in j["t_starts"]:
                    tot += j["n"] * pred_spm(j["backend"], arm, t) * len(BASECMP_STRENGTHS)
        return tot * margin / 60.0

    size_logs = sorted(by_stage.get("size", []), key=lambda j: j.get("started", ""))
    surf, surf_cols, chosen = {}, [], {}
    for j in size_logs:
        s_, c_, ch = parse_size_log(os.path.join(args.logs, j["log"]))
        if s_:
            surf, surf_cols = s_, c_
        chosen[j["job"]] = (j.get("budget"), ch, j.get("started"))

    # ================================================================ WRITE
    today = datetime.date.today().isoformat()
    A("# Base-model comparison (basecmp): the partial screen, cancelled 26 Sep")
    A("")
    A("**Written for:** the project team (Group 2), as the record of a cluster run "
      "that was cancelled before it produced any result. **Nothing here is a "
      "finding for the paper or the slides.**")
    A("")
    A("**Generated by `proj1/scripts/basecmp_partial_table.py` on %s. Do not edit by "
      "hand.** Source: `results/basecmp/` (read-only) and `logs/basecmp-*.out`."
      % today)
    A("")
    # The project requires an independent adversarial pass before a result is
    # cited. This page has not had one, and saying so is part of the page.
    A("> **NOT INDEPENDENTLY CHECKED (27 Sep).** The adversarial re-check this "
      "project requires before a result is cited was launched and did **not** "
      "complete -- it hit a session rate limit. Every number here was produced "
      "by this script and is self-consistent, but no independent pass has "
      "confirmed the claims or hunted for defects, and comparable passes on the "
      "v3 and tune write-ups each found real defects. **Re-run the check before "
      "quoting this page.** One partial finding did land and is in section 8: "
      "the two bases are not guided equally at t_start = 0.05.")
    A("")
    A("> **PARTIAL, CANCELLED, q90.** The chain was cancelled during its screen. No "
      "(w, t_start) was frozen, no refine or full run exists, and no verdict is "
      "issued. The target is q90 and the paper reports q50 only, so no number on "
      "this page goes into the paper or the slides. At the screen's n = %d with one "
      "seed, the se of a single in-band fraction near 0.05 is about %.3f "
      "(printed on every row). None of these cells can be pooled with protocol v3's."
      % (n_scr["fm"], se_prop(0.05, n_scr["fm"])))
    A("")

    A("<!-- WHAT-IT-WAS -->")
    A("")

    # ---- how far it got
    A("## 1. How far it got")
    A("")
    n_ok_scr = sum(1 for j in screen_jobs if j["done_min"] is not None)
    n_can = sum(1 for j in screen_jobs if j["cancelled_at"])
    A("| stage | what exists | status |")
    A("|---|---|---|")
    for st in ("probe", "size", "screen", "plan", "refine", "freeze", "full", "table"):
        js = by_stage.get(st, [])
        if st == "probe":
            desc = "; ".join("%s: %d cells (n = %d)" % (LABEL[b], len(prb[b]), n_prb[b])
                             for b in BASES)
            stt = "complete on both bases (%d job logs)" % len(js)
        elif st == "size":
            desc = "`results/basecmp/plan.json` (n_screen %d, n_full %d, budget %g GPU-h)" % (
                plan["n_screen"], plan["n_full"], plan["budget_gpuh"])
            stt = "ran twice, once per chain (jobs %s)" % ", ".join(j["job"] for j in js)
        elif st == "screen":
            desc = "; ".join("%s: %d of %d cells" % (LABEL[b], len(scr[b]),
                                                        plan["cells"]["screen"][b])
                             for b in BASES)
            stt = ("%d of 96 array tasks logged: %d completed, %d cancelled mid-task "
                   "at %s; tasks %d-95 never started"
                   % (len(screen_jobs), n_ok_scr, n_can,
                      fmt_ts(next((j["cancelled_at"] for j in screen_jobs
                                   if j["cancelled_at"]), "")),
                      max(j["task"] for j in screen_jobs) + 1))
        else:
            desc = ("no `%s/` directory under either base" % st
                    if st in ("refine", "full") else "nothing written")
            stt = "never started (%d job logs)" % len(js)
        A("| %s | %s | %s |" % (st, desc, stt))
    A("")
    A("`.failed` cells: **%d**. Sub-directories present: %s. Cells outside the "
      "pre-registered grid: %s."
      % (len(failed), "; ".join("%s `%s`" % (b, "`, `".join(extra_dirs[b]))
                                 for b in BASES),
         ", ".join("%s %d" % (b, len(extra[b])) for b in BASES)))
    A("")
    A("**Two chains were queued and both ran their probe and size stages.** The "
      "job logs show arrays %s. The budget-96 size stage (job %s, %s) picked "
      "n_screen = 500. The budget-130 one (job %s, %s) ran 43 s later, picked "
      "n_screen = 750 and overwrote `plan.json`. The only screen array in the "
      "logs (%s) belongs to the budget-96 chain, but every job reads n from "
      "`plan.json` at run time, so every screen cell has n = %d. The logs hold no "
      "screen job from the budget-130 chain."
      % (", ".join("%s (budget %s, %s)" % (a[2], a[1], a[0]) for a in arrays
                   if a[0] in ("probe", "screen")),
         size_logs[0]["job"], fmt_ts(size_logs[0].get("started")),
         size_logs[-1]["job"], fmt_ts(size_logs[-1].get("started")),
         screen_jobs[0]["array_id"], n_scr["fm"]))
    A("")
    A("Screen wall clock: first task started %s, the chain was cancelled at %s; at "
      "most %d array tasks ran at once."
      % (fmt_ts(screen_jobs[0].get("started")),
         fmt_ts(next((j["cancelled_at"] for j in screen_jobs if j["cancelled_at"]), "")),
         conc))
    A("")

    A("### 1.1 Which cells exist")
    A("")
    A("Count of strengths present, out of 5 (w = %s), per start-time. The "
      "pre-registered grid is 6 guided arms x 5 w x 4 t_start x 3 properties, plus "
      "one unguided cell per property = 363 per base."
      % ", ".join("%g" % w for w in BASECMP_STRENGTHS))
    A("")
    A("| property | arm | %s |" % " | ".join(
        "%s t=%g" % (("ours" if b == "fm" else "EquiFM"), t)
        for b in BASES for t in BASECMP_T_STARTS))
    A("|---|---|%s" % ("---|" * (len(BASES) * len(BASECMP_T_STARTS))))
    for p in PROPS:
        for a in GUIDED:
            row = []
            for b in BASES:
                for t in BASECMP_T_STARTS:
                    k = sum((p, a, float(w), float(t)) in have[b] for w in BASECMP_STRENGTHS)
                    row.append("**%d**" % k if 0 < k < 5 else str(k))
            A("| %s | %s | %s |" % (p, a, " | ".join(row)))
        A("| %s | unguided (w=1, t=0.5) | %s |" % (p, " | ".join(
            ("yes" if (p, "unguided", 1.0, 0.5) in have[b] else "**no**") if t == 0.5 else "-"
            for b in BASES for t in BASECMP_T_STARTS)))
    A("")
    for b in BASES:
        A("- **%s: %d present, %d missing.** Bold counts are cells whose array task was "
          "cancelled part-way." % (LABEL[b], len(have[b] & want), len(miss[b])))
    A("- **t_start = 0.75 never ran on either base, and t_start = 0.5 ran only for mu.** "
      "The array went in task order: property-major within each start-time, "
      "t = 0.05 first. It was cancelled inside the mu block at t = 0.5.")
    A("- **Unguided ran for mu only** (%s). The alpha and gap unguided cells sat in "
      "tasks that never started. See section 4 for how the chemistry floor is set "
      "for them." % "; ".join("%s: %s" % (LABEL[b], ", ".join(ung_have[b]))
                             for b in BASES))
    A("- **Matched cells** (the same property, arm, w and t_start on both bases): "
      "**%d**. Ours has %d the other lacks (%s), and EquiFM has %d."
      % (len(matched), len(have["fm"] - have["equifm"]),
         ", ".join("%s/%s/w%g/t%g" % k for k in sorted(have["fm"] - have["equifm"])[:10]),
         len(have["equifm"] - have["fm"])))
    A("")

    A("### 1.2 What the official readers say (verbatim)")
    A("")
    A("```")
    A("$ python proj1/scripts/basecmp_table.py --backend equifm      # and --backend fm")
    A("no cells under ...\\results\\basecmp\\equifm\\full\\n*\\seed*\\tr__*__full.json")
    A("rc=2")
    A("")
    A("$ python proj1/scripts/basecmp_freeze.py --backend fm         # BEFORE the fix below")
    A("INCONSISTENT delta for mu: [0.251947229461, 0.251947248755, ... 0.251947372746] "
      "-- in-band means a different thing in each cell")
    A("rc=2")
    A("")
    A("$ python proj1/scripts/basecmp_freeze.py --backend fm         # after the fix")
    A("INCOMPLETE -- %d of 363 base-grid cells missing, e.g. alpha/btvg/w0.05/t0.5, ..."
      % len(miss["fm"]))
    A("rc=2")
    A("$ python proj1/scripts/basecmp_freeze.py --backend equifm")
    A("INCOMPLETE -- %d of 363 base-grid cells missing, e.g. alpha/btvg/w0.05/t0.5, ..."
      % len(miss["equifm"]))
    A("rc=2")
    A("```")
    A("")
    A("**A defect that would have stopped the chain even without the cancellation.** "
      "`basecmp_freeze.load_screen` compared δ across cells rounded to 12 decimals. "
      "But δ is recomputed inside every array task (`local_delta`, float32 on the "
      "GPU), so one property's cells differ around the 7th significant figure. The "
      "relative spread here is at most %.1e. `STAGE=plan` calls "
      "`basecmp_freeze.py --emit-refine`, which goes through the same loader, so it "
      "would have logged `STOP: the refine plan refused` on a complete screen. "
      "Fixed (section 10): the check now uses a relative tolerance of 1e-5, which "
      "still refuses any real change of rule (local and global δ differ by 5-60 %%). "
      "The fix does not change a number on this page. The freeze was **not** run "
      "with `--allow-partial`, because that prints picks over a grid with holes, and "
      "this page makes none."
      % max(dtab[p]["rel"] for p in PROPS))
    A("")

    # ---- provenance
    A("## 2. Provenance")
    A("")
    A("| | ours (FM) | EquiFM |")
    A("|---|---|---|")
    A("| generator md5 | `%s` | `%s` |" % (scr["fm"][0]["gen_md5"], scr["equifm"][0]["gen_md5"]))
    A("| sampler grid | %s | %s |" % (scr["fm"][0]["grid"], scr["equifm"][0]["grid"]))
    A("| screen cells / n / seed | %d / %d / %d | %d / %d / %d |" % (
        len(scr["fm"]), n_scr["fm"], scr["fm"][0]["seed"],
        len(scr["equifm"]), n_scr["equifm"], scr["equifm"][0]["seed"]))
    A("| probe cells / n / seed | %d / %d / %d | %d / %d / %d |" % (
        len(prb["fm"]), n_prb["fm"], prb["fm"][0]["seed"],
        len(prb["equifm"]), n_prb["equifm"], prb["equifm"][0]["seed"]))
    A("| steps / solver / clip / batch | %s / %s / %s / %s | %s / %s / %s / %s |" % (
        scr["fm"][0]["steps"], scr["fm"][0]["solver"], scr["fm"][0]["clip"], scr["fm"][0]["batch"],
        scr["equifm"][0]["steps"], scr["equifm"][0]["solver"], scr["equifm"][0]["clip"],
        scr["equifm"][0]["batch"]))
    A("| screen cells by device | %s | %s |" % tuple(
        ", ".join("%s: %d" % (k.replace("NVIDIA B200 MIG ", "B200 MIG "), v)
                  for k, v in sorted(devs[(b, "screen")].items())) for b in BASES))
    A("| probe cells by device | %s | %s |" % tuple(
        ", ".join("%s: %d" % (k.replace("NVIDIA B200 MIG ", "B200 MIG "), v)
                  for k, v in sorted(devs[(b, "probe")].items())) for b in BASES))
    A("| torch | %s | %s |" % (scr["fm"][0]["torch"], scr["equifm"][0]["torch"]))
    A("")
    A("Guide `%s`, oracle `%s`, oracle 2 `%s` (per property), **identical on both "
      "bases**. The target is q90 per property (mu 4.6627 D, alpha 85.05 Bohr^3, "
      "gap 0.3162 Ha). Sizes come from `val[:%d]`: every sidecar on both bases "
      "carries the same molecule indices (%s, %d sidecars checked). The sizes are "
      "shared across bases, the molecules are not."
      % (scr["fm"][0]["guide"].replace("alpha", "<p>").replace("mu", "<p>").replace("gap", "<p>"),
         scr["fm"][0]["oracle"].replace("alpha", "<p>").replace("mu", "<p>").replace("gap", "<p>"),
         "OC-Flow exp_class_<p>", n_scr["fm"], "all identical" if idx_ok else "NOT identical",
         n_side))
    A("")
    A("**The δ rule was `local`, for the oracle and oracle 2 alike, on every cell of "
      "both bases.** δ = 2 x f_B's MAE over `val` molecules whose TRUE property lies "
      "in [Q_train_a(0.85), Q_train_a(0.95)]. That is the protocol's section 3 rule "
      "applied to TFG's oracle, and it contains no generator. The logs' `delta "
      "0.25195` for mu is this local δ. The `(local)` tag is printed on the oracle-2 "
      "line and applies to both. δ is the same on both bases within float noise:")
    A("")
    A("| property | local δ (range over all cells, both bases) | rel. spread | global δ (2 x overall MAE), not used | local window | val molecules | coverage at δ | at 1 x MAE | oracle-2 δ |")
    A("|---|---|---|---|---|---|---|---|---|")
    for p in PROPS:
        t_ = dtab[p]
        det = t_["det"]
        A("| %s | %.6g .. %.6g %s | %.1e | %.6g | [%.4g, %.4g] | %s | %.3f | %.3f | %.6g |"
          % (p, t_["lo"], t_["hi"], UNIT[p], t_["rel"], t_["glo"],
             det["window"][0], det["window"][1], det.get("n"),
             det.get("coverage_at_delta", float("nan")),
             det.get("coverage_at_1x", float("nan")), t_["d2lo"]))
    A("")
    A("Strengths: `btvg` and `btvg_var` multiply w by (tau / s)^2 before it reaches "
      "the sampler (`strength_scale`, recorded as `w_scale`), so a given w asks the "
      "same of every arm. Job IDs: probe %s; size %s; screen array %s (job logs "
      "%s..%s). Dates: probe started %s, screen cancelled %s."
      % (", ".join(sorted(j["job"] for j in by_stage.get("probe", []))),
         ", ".join(j["job"] for j in size_logs), screen_jobs[0]["array_id"],
         screen_jobs[0]["job"], screen_jobs[-1]["job"],
         fmt_ts(min(j.get("started", "9") for j in by_stage.get("probe", []))),
         fmt_ts(next((j["cancelled_at"] for j in screen_jobs if j["cancelled_at"]), ""))))
    A("")

    # ---- cost
    A("## 3. Cost: the first measurement of EquiFM")
    A("")
    A("`plan.json` fits seconds per molecule as `a + b x (1 - t_start)` for each "
      "(base, arm), from the probe's three start-times (n = 256, mu, w = 1). The "
      "table shows the fit at the two ends of the screened grid and the EquiFM / "
      "ours ratio.")
    A("")
    A("| arm | ours a | ours b | EquiFM a | EquiFM b | ours s/mol t=0.05 | EquiFM s/mol t=0.05 | ratio t=0.05 | ours s/mol t=0.75 | EquiFM s/mol t=0.75 | ratio t=0.75 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    ratios = {}
    for a in ("unguided",) + GUIDED:
        fa, fb = spm["fm"][a], spm["equifm"][a]
        r05 = pred_spm("equifm", a, 0.05) / pred_spm("fm", a, 0.05)
        r75 = pred_spm("equifm", a, 0.75) / pred_spm("fm", a, 0.75)
        ratios[a] = (r05, r75)
        A("| %s | %.4f | %.4f | %.4f | %.4f | %.3f | %.3f | %.2fx | %.3f | %.3f | %.2fx |"
          % (a, fa[0], fa[1], fb[0], fb[1], pred_spm("fm", a, 0.05),
             pred_spm("equifm", a, 0.05), r05, pred_spm("fm", a, 0.75),
             pred_spm("equifm", a, 0.75), r75))
    A("")
    A("(`unguided` was probed at one start-time only, so its slope is 0 by construction.)")
    A("")
    A("**The cost surface** (GPU-hours including the %.2fx margin, exactly as the "
      "sizer printed it; both size jobs printed the same surface):" % margin)
    A("")
    A("| n_screen \\ n_full | %s |" % " | ".join(str(c) for c in surf_cols))
    A("|---|%s" % ("---|" * len(surf_cols)))
    for ns in sorted(surf, reverse=True):
        A("| %d | %s |" % (ns, " | ".join("%.1f" % surf[ns][c] for c in surf_cols)))
    A("")
    for jid, (bud, ch, st) in chosen.items():
        A("- job %s (budget %s): `%s`" % (jid, bud, ch[0] if ch else "(no CHOSEN line)"))
        for x in ch[1:]:
            A("  - `%s`" % x)
    A("")
    A("`plan.json` (the budget-130 decision): n_screen %d, n_full %d; estimated "
      "screen %.1f + refine %.1f + full %.1f = %.1f GPU-h; worst single task `%s` "
      "at %.0f min against the %d-min guard."
      % (plan["n_screen"], plan["n_full"], plan["est_gpuh"]["screen"],
         plan["est_gpuh"]["refine"], plan["est_gpuh"]["full"], plan["est_gpuh"]["total"],
         plan["worst_task"]["which"], plan["worst_task"]["minutes"],
         plan["worst_task"]["guard_minutes"]))
    A("")
    # measured vs predicted, screen
    A("**The screen against the model.** Measured seconds per molecule (cell "
      "`seconds` / n, mean over the strengths present), with the model's "
      "prediction without margin in brackets:")
    A("")
    A("| arm | %s |" % " | ".join("%s t=%g" % ("ours" if b == "fm" else "EquiFM", t)
                                 for b in BASES for t in (0.05, 0.25, 0.5)))
    A("|---|%s" % ("---|" * 6))
    meas_ratio = []
    for a in GUIDED:
        row = []
        for b in BASES:
            for t in (0.05, 0.25, 0.5):
                cs = [c for c in scr[b] if c["arm"] == a and c["t"] == t]
                if not cs:
                    row.append("-")
                    continue
                m = sum(c["seconds"] / c["n"] for c in cs) / len(cs)
                pr = pred_spm(b, a, t)
                meas_ratio.append((b, a, t, m / pr))
                row.append("%.3f (%.3f)" % (m, pr))
        A("| %s | %s |" % (a, " | ".join(row)))
    A("")
    rr = {b: [x[3] for x in meas_ratio if x[0] == b] for b in BASES}
    tot_h = {b: sum(c["seconds"] for c in scr[b]) / 3600.0 for b in BASES}
    tot_hp = {b: sum(c["seconds"] for c in prb[b]) / 3600.0 for b in BASES}
    pred_h = {b: sum(c["n"] * pred_spm(b, c["arm"], c["t"]) for c in scr[b]) / 3600.0
              for b in BASES}
    A("Measured / predicted over these (base, arm, t) groups: ours %.2f-%.2fx, "
      "EquiFM %.2f-%.2fx. GPU time inside the cells (the sum of `seconds`, which "
      "leaves out model loading and the per-task preflight): screen ours %.1f h, "
      "EquiFM %.1f h. The model without margin predicts %.1f h and %.1f h for the "
      "same cells. The probe used %.2f h and %.2f h. Because the EquiFM probe ran "
      "twice (below), %.2f h of EquiFM probe time was duplicated."
      % (min(rr["fm"]), max(rr["fm"]), min(rr["equifm"]), max(rr["equifm"]),
         tot_h["fm"], tot_h["equifm"], pred_h["fm"], pred_h["equifm"],
         tot_hp["fm"], tot_hp["equifm"],
         sum(c["sec"] for j in by_stage.get("probe", []) if j.get("backend") == "equifm"
             for c in j["cells_logged"]) / 3600.0 - tot_hp["equifm"]))
    A("")
    # ---- device-matched cost: the only like-for-like base cost comparison
    short = lambda d: (d or "?").replace("NVIDIA B200 MIG ", "")       # noqa: E731
    cgrp = collections.defaultdict(list)
    for b in BASES:
        for c in scr[b]:
            cgrp[(b, c["arm"], c["t"], short(c["device"]))].append(c["seconds"] / c["n"])
    A("**Device-matched cost, EquiFM against ours.** The screen ran on two MIG "
      "slice types, so the like-for-like cost ratio compares screen cells of the "
      "same arm and t_start **on the same slice type**. Seconds per molecule "
      "(cells):")
    A("")
    A("| arm | t_start | slice | ours s/mol | EquiFM s/mol | EquiFM / ours |")
    A("|---|---|---|---|---|---|")
    dm_ratio = []
    for a in ("unguided",) + GUIDED:
        for t in BASECMP_T_STARTS:
            for dv in sorted({k[3] for k in cgrp}):
                o, e = cgrp.get(("fm", a, t, dv)), cgrp.get(("equifm", a, t, dv))
                if not (o and e):
                    continue
                mo, me = sum(o) / len(o), sum(e) / len(e)
                dm_ratio.append((a, t, dv, me / mo))
                A("| %s | %g | %s | %.3f (%d) | %.3f (%d) | %.2fx |"
                  % (a, t, dv, mo, len(o), me, len(e), me / mo))
    A("")
    sl = []
    for b in BASES:
        for a in GUIDED:
            for t in BASECMP_T_STARTS:
                x1, x2 = cgrp.get((b, a, t, "1g.45gb")), cgrp.get((b, a, t, "2g.45gb"))
                if x1 and x2:
                    sl.append((sum(x1) / len(x1)) / (sum(x2) / len(x2)))
    ung_dev = {b: short(ung[b]["device"]) for b in BASES}
    A("Guided arms, device-matched: EquiFM costs **%.2f-%.2fx** ours per molecule "
      "(%d comparisons). The two unguided cells ran on different slices (ours %s "
      "%.3f s/mol, EquiFM %s %.3f s/mol), so they have no device-matched ratio. "
      "Within a base, the same (arm, t_start) on a 1g.45gb slice took %.2f-%.2fx "
      "as long as on a 2g.45gb slice (%d comparisons)."
      % (min(r[3] for r in dm_ratio), max(r[3] for r in dm_ratio), len(dm_ratio),
         ung_dev["fm"], ung["fm"]["seconds"] / ung["fm"]["n"], ung_dev["equifm"],
         ung["equifm"]["seconds"] / ung["equifm"]["n"], min(sl), max(sl), len(sl)))
    A("")
    pdev = {b: collections.Counter((short(c["device"]), c["t"]) for c in prb[b]) for b in BASES}
    # which end of each EquiFM arm's probe line sat on the slower 1g slice
    slow_late, slow_early = [], []
    for a in GUIDED:
        dv = {c["t"]: short(c["device"]) for c in prb["equifm"] if c["arm"] == a}
        if dv.get(0.75) == "1g.45gb" and dv.get(0.05) != "1g.45gb":
            slow_late.append(a)
        elif dv.get(0.05) == "1g.45gb" and dv.get(0.75) == "1g.45gb" and dv.get(0.5) != "1g.45gb":
            slow_early.append(a)
    A("**The probe fit mixes slice types on EquiFM only.** Probe cells by (slice, "
      "t_start): ours %s; EquiFM %s. Each EquiFM line through three start-times "
      "therefore joins points measured on different hardware. The fitted intercept "
      "a should sit near the unguided cost (EquiFM %.3f s/mol, on %s). It comes out "
      "at %s for %s, whose late (t = 0.75) point ran on the slower slice, and at %s "
      "for %s, whose t = 0.5 point alone ran on the faster one. Both are the "
      "direction the slice mix predicts. Use the device-matched screen ratio above "
      "for EquiFM's cost."
      % (tuple(", ".join("%s t=%g: %d" % (k[0], k[1], v) for k, v in sorted(pdev[b].items()))
               for b in BASES)
         + (spm["equifm"]["unguided"][0], short(ung_p["equifm"]["device"]),
            ", ".join("%.3f" % spm["equifm"][a][0] for a in slow_late) or "-",
            ", ".join(slow_late) or "-",
            ", ".join("%.3f" % spm["equifm"][a][0] for a in slow_early) or "-",
            ", ".join(slow_early) or "-")))
    A("")
    A("Per array task, logged wall minutes against the model's prediction with the "
      "%.2fx margin. These are the tasks that finished; the preflight (a few "
      "seconds) is included:" % margin)
    A("")
    A("| task | base | property | arms | t_start | cells | wall min | predicted min (with margin) | wall / predicted |")
    A("|---|---|---|---|---|---|---|---|---|")
    tr = []
    for j in screen_jobs:
        if j["done_min"] is None:
            A("| %d | %s | %s | %s | %s | - | cancelled at %s | %.1f | - |"
              % (j["task"], j["backend"], ",".join(j["props"]), ",".join(j["arms"]),
                 ",".join("%g" % t for t in j["t_starts"]), fmt_ts(j["cancelled_at"]),
                 task_pred_min(j)))
            continue
        pm = task_pred_min(j)
        tr.append((j["backend"], j["done_min"] / pm, j))
        A("| %d | %s | %s | %s | %s | %d | %.1f | %.1f | %.2f |"
          % (j["task"], j["backend"], ",".join(j["props"]), ",".join(j["arms"]),
             ",".join("%g" % t for t in j["t_starts"]), j["done_cells"], j["done_min"],
             pm, j["done_min"] / pm))
    A("")
    worst_eq = max((x for x in tr if x[0] == "equifm"), key=lambda x: x[2]["done_min"])
    worst_fm = max((x for x in tr if x[0] == "fm"), key=lambda x: x[2]["done_min"])
    A("Longest finished task: EquiFM task %d (%s, %s, t=%s) at %.1f min; ours task "
      "%d (%s, %s, t=%s) at %.1f min. Nothing came near the 225-min guard."
      % (worst_eq[2]["task"], ",".join(worst_eq[2]["props"]), ",".join(worst_eq[2]["arms"]),
         ",".join("%g" % t for t in worst_eq[2]["t_starts"]), worst_eq[2]["done_min"],
         worst_fm[2]["task"], ",".join(worst_fm[2]["props"]), ",".join(worst_fm[2]["arms"]),
         ",".join("%g" % t for t in worst_fm[2]["t_starts"]), worst_fm[2]["done_min"]))
    A("")

    # ---- unguided + floor
    A("## 4. The unguided reference and the chemistry floor")
    A("")
    A("The floor is 0.9 x **that base's own** unguided molecule stability "
      "(protocol section 6). One unguided cell per base exists, for mu (n = %d, w = 1, "
      "t = 0.5, which the unguided sampler never reads)." % n_scr["fm"])
    A("")
    A("| base | in-band (cont / dec) | +- se | MAE/δ (cont / dec) | bias/δ (cont / dec) | resid sd/δ (cont / dec) | mol stab +- se | floor (0.9x) | validity | uniq of valid | oracle-2 in-band (cont / dec) | f_B mean (D) |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in BASES:
        u = ung[b]
        A("| %s | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s +- %s | **%s** | %s | %s | %s / %s | %s |"
          % (LABEL[b], f(u["ib"]), f(u["ib_dec"]), f(se_prop(u["ib"], u["n"]), 4),
             f(se_prop(u["ib_dec"], u["n"]), 4), f(u["mae"], 2), f(u["mae_dec"], 2),
             f(u["bias"], 2), f(u["bias_dec"], 2), f(u["sd"], 2), f(u["sd_dec"], 2),
             f(u["stab"]), f(se_prop(u["stab"], u["n"]), 3), f(floor[b], 4),
             f(u["valid"]), f(u["uniq"]), f(u["o2_ib"]), f(u["o2_ib_dec"]),
             f(u["f_B_mean"], 3)))
    A("")
    zs = z_unpaired(ung["fm"]["stab"], n_scr["fm"], ung["equifm"]["stab"], n_scr["equifm"])
    zv = z_unpaired(ung["fm"]["valid"], n_scr["fm"], ung["equifm"]["valid"], n_scr["equifm"])
    zi = z_unpaired(ung["fm"]["ib_dec"], n_scr["fm"], ung["equifm"]["ib_dec"], n_scr["equifm"])
    A("Unguided, EquiFM minus ours (unpaired z): mol stability %+.3f (z = %s), "
      "validity %+.3f (z = %s), decoded in-band %+.4f (z = %s)."
      % (ung["equifm"]["stab"] - ung["fm"]["stab"], fz(zs),
         ung["equifm"]["valid"] - ung["fm"]["valid"], fz(zv),
         ung["equifm"]["ib_dec"] - ung["fm"]["ib_dec"], fz(zi)))
    A("")
    fmarg = {b: (ung[b]["stab"] - floor[b]) / se_prop(ung[b]["stab"], ung[b]["n"])
             for b in BASES}
    A("**The floor is not equally hard to breach by noise on the two bases.** It "
      "sits 10 %% below unguided on both, which is %.1f binomial se of one cell's "
      "stability below the unguided value on ours and %.1f se on EquiFM. A guided "
      "cell whose true stability equals the unguided generator's would land below "
      "the floor by chance with probability about %.3f on ours and %.1e on EquiFM. "
      "That counts both cells' noise, since the floor itself comes from one n = %d "
      "cell. Floor-clearance counts (sections 6-7) are therefore not like-for-like "
      "across bases for cells near the floor."
      % (fmarg["fm"], fmarg["equifm"], 0.5 * math.erfc(fmarg["fm"] / 2.0),
         0.5 * math.erfc(fmarg["equifm"] / 2.0), n_scr["fm"]))
    A("")
    A("**The floor for alpha and gap is borrowed from mu's unguided cell.** No alpha "
      "or gap unguided cell ran. The unguided sampler reads no property (`f_net=None`; "
      "sizes `val[:n]`; seed %d), so its molecules, and therefore their stability, "
      "are the same for every property up to GPU nondeterminism. That makes mu's "
      "unguided stability the base's unguided chemistry. What is **not** available "
      "is unguided in-band for alpha and gap: it needs the alpha or gap oracle on "
      "unguided molecules, which were never scored, and coordinates are not saved "
      "on this path." % BASECMP_SEED)
    A("")
    A("Feature-scale check (`basecmp_freeze.scale_sanity`: the oracle's mean on "
      "unguided samples against QM9's train_a mean, in MAD; above 1.0 MAD would "
      "suggest a wrong sampler divisor):")
    A("")
    for b in BASES:
        for line in scale[b]:
            A("- %s: `%s`" % (LABEL[b], line))
    A("")

    # ---- probe
    A("## 5. The probe (n = %d, mu, w = 1, both bases)" % n_prb["fm"])
    A("")
    A("The probe exists to measure cost; these are its quality numbers too. The "
      "probe cells have no sidecars, so bias/δ and resid sd/δ here are derived from "
      "the aggregate block (bias = mean f_B - target, sd = sqrt(RMSE^2 - bias^2)). "
      "On the %d screen cells that have sidecars, the derived and the sidecar values "
      "agree to %.1e δ. The probe floor is 0.9 x the probe's own unguided: ours "
      "%.4f, EquiFM %.4f."
      % (sum(1 for b in BASES for c in scr[b] if c["stat_src"] == "sidecar"),
         agg_err, floor_p["fm"], floor_p["equifm"]))
    A("")
    A("| base | arm | t_start | in-band (cont / dec) | +- se (cont / dec) | MAE/δ (cont / dec) | bias/δ (cont / dec) | resid sd/δ (cont / dec) | mol stab | clears probe floor | validity | uniq | oracle-2 in-band (cont / dec) | s/mol |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in BASES:
        for c in sorted(prb[b], key=lambda c: ((("unguided",) + GUIDED).index(c["arm"]),
                                               c["t"])):
            A("| %s | %s | %g | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s | %s | %s | %s | %s / %s | %.3f |"
              % (LABEL[b], c["arm"], c["t"], f(c["ib"]), f(c["ib_dec"]),
                 f(se_prop(c["ib"], c["n"])), f(se_prop(c["ib_dec"], c["n"])),
                 f(c["mae"], 2), f(c["mae_dec"], 2), f(c["bias"], 2), f(c["bias_dec"], 2),
                 f(c["sd"], 2), f(c["sd_dec"], 2), f(c["stab"]),
                 "-" if c["arm"] == "unguided" else ("yes" if c["clears"] else "**no**"),
                 f(c["valid"]), f(c["uniq"]), f(c["o2_ib"]), f(c["o2_ib_dec"]),
                 c["seconds"] / c["n"]))
    A("")
    # duplicated EquiFM probe cells
    pj = [j for j in by_stage.get("probe", []) if j.get("backend") == "equifm"
          and j["cells_logged"]]
    dup = []
    if len(pj) == 2:
        j1, j2 = sorted(pj, key=lambda j: j.get("started", ""))
        k1 = {(c["arm"], c["t"]): c for c in j1["cells_logged"]}
        k2 = {(c["arm"], c["t"]): c for c in j2["cells_logged"]}
        filecell = {(c["arm"], c["t"]): c for c in prb["equifm"]}
        for k in sorted(set(k1) & set(k2), key=lambda k: (k[1], GUIDED.index(k[0]))):
            fc = filecell.get(k)
            src = "?"
            if fc:
                raw = fc["mae"] * fc["delta"]
                src = ("job %s" % j2["job"] if abs(raw - k2[k]["mae"]) < 6e-5 else
                       "job %s" % j1["job"] if abs(raw - k1[k]["mae"]) < 6e-5 else "?")
            dup.append((k, k1[k], k2[k], src))
        # the slice each run used, read off the files each job is known to have written
        jdev = {j1["job"]: set(), j2["job"]: set()}
        for c in prb["equifm"]:
            raw, k = c["mae"] * c["delta"], (c["arm"], c["t"])
            for jj, kk in ((j2, k2), (j1, k1)):
                if k in kk and abs(raw - kk[k]["mae"]) < 6e-5:
                    jdev[jj["job"]].add(short(c["device"]))
                    break
        ctx_jdev.update(jdev)
        A("**The EquiFM probe ran twice, concurrently.** Job %s (budget-96 chain, "
          "%s) started %s. Job %s (budget-130 chain, %s) started %s, found %d of its "
          "19 cells done, and recomputed the other %d while the first job was still "
          "writing them: the same cells, the same seed and the same code, on "
          "**different MIG slice types**. Both runs are in the logs, so this measures "
          "how far a fixed seed reproduces across runs and slices (n = %d). It is "
          "not a same-hardware replicate. The last column says whose file survived; "
          "the cost fit in `plan.json` read the surviving files."
          % (j1["job"], "/".join(sorted(jdev[j1["job"]])) or "?", fmt_ts(j1.get("started")),
             j2["job"], "/".join(sorted(jdev[j2["job"]])) or "?", fmt_ts(j2.get("started")),
             19 - len(j2["cells_logged"]), len(j2["cells_logged"]), n_prb["equifm"]))
        A("")
        A("| arm | t_start | in-band run 1 / run 2 | dec run 1 / run 2 | mol stab run 1 / run 2 | MAE (D) run 1 / run 2 | s run 1 / run 2 | file on disk from |")
        A("|---|---|---|---|---|---|---|---|")
        for k, c1, c2, src in dup:
            A("| %s | %g | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.4f / %.4f | %.1f / %.1f | %s |"
              % (k[0], k[1], c1["ib"], c2["ib"], c1["ib_dec"], c2["ib_dec"], c1["stab"],
                 c2["stab"], c1["mae"], c2["mae"], c1["sec"], c2["sec"], src))
        A("")
        dmax_stab = max(abs(c1["stab"] - c2["stab"]) for _k, c1, c2, _s in dup)
        dmax_ib = max(abs(c1["ib_dec"] - c2["ib_dec"]) for _k, c1, c2, _s in dup)
        n_ident = sum(1 for _k, c1, c2, _s in dup if abs(c1["mae"] - c2["mae"]) < 1e-9)
        g05 = max(abs(c1["stab"] - c2["stab"]) for k, c1, c2, _s in dup if k[1] == 0.05)
        g75 = max(abs(c1["stab"] - c2["stab"]) for k, c1, c2, _s in dup if k[1] == 0.75)
        ctx_dup.update({"n_ident": n_ident, "n": len(dup), "dstab": dmax_stab,
                        "dib": dmax_ib, "g05": g05, "g75": g75})
        A("%d of %d re-runs reproduce the MAE to 4 decimals. The largest gaps are "
          "%.3f in molecule stability and %.3f in decoded in-band, against a binomial "
          "se of about %.3f and %.3f at this n. The largest stability gap is %.3f at "
          "t_start = 0.05 (about 95 guided steps) against %.3f at t_start = 0.75 (about "
          "25). At a fixed seed, guided EquiFM sampling is not reproducible across "
          "these runs."
          % (n_ident, len(dup), dmax_stab, dmax_ib, se_prop(0.75, n_prb["equifm"]),
             se_prop(0.07, n_prb["equifm"]), g05, g75))
        A("")

    # ---- floor clearance summary + in-band profile at t = 0.25
    A("## 6. Per property, per arm: the screen in summary")
    A("")
    A("Floors: ours %.4f, EquiFM %.4f (section 4). **Cells clearing the floor, out of "
      "those present**, per base:" % (floor["fm"], floor["equifm"]))
    A("")
    A("| property | arm | ours: clear / present | EquiFM: clear / present | ours: mol stab range | EquiFM: mol stab range | ours: in-band dec range | EquiFM: in-band dec range |")
    A("|---|---|---|---|---|---|---|---|")
    never = {b: [] for b in BASES}
    for p in PROPS:
        for a in GUIDED:
            row = [p, a]
            rng = []
            for b in BASES:
                cs = [c for c in scr[b] if c["prop"] == p and c["arm"] == a]
                k = sum(c["clears"] for c in cs)
                if cs and k == 0:
                    never[b].append("%s/%s" % (a, p))
                row.append(("**0** / %d" % len(cs)) if cs and k == 0 else "%d / %d" % (k, len(cs)))
                rng.append((cs, b))
            for cs, b in rng:
                row.append("%.3f-%.3f" % (min(c["stab"] for c in cs), max(c["stab"] for c in cs)))
            for cs, b in rng:
                row.append("%.3f-%.3f" % (min(c["ib_dec"] for c in cs), max(c["ib_dec"] for c in cs)))
            A("| %s |" % " | ".join(row))
    A("")
    for b in BASES:
        A("- **%s: no present cell clears the floor for** %s."
          % (LABEL[b], ", ".join(never[b]) if never[b] else "(none: every arm has at least one)"))
    A("")
    A("These ranges are over an incomplete grid: t = 0.75 is absent everywhere and "
      "t = 0.5 exists only for mu. They are descriptive spans, not picks. The "
      "maximum of 10-15 noisy cells is biased upward, which is one reason "
      "`basecmp_freeze.py` will not pick from an incomplete grid.")
    A("")

    # ---- the weakest strength: does near-zero guidance leave chemistry alone?
    def clip_frac(c):
        nb = math.ceil(c["n"] / float(c["batch"]))
        per = c["guided_steps"] / nb if nb else 0
        return c["clipped"] / (c["n"] * per) if per else float("nan")

    weak = {}
    A("**The weakest strength, w = 0.05.** If guidance at w = 0.05 were negligible, "
      "these cells would sit at unguided stability (ours %.3f, EquiFM %.3f, se %.3f "
      "and %.3f). Molecule stability over every (property, t_start) cell present at "
      "w = 0.05, and the fraction of guided sample-steps where the velocity clip "
      "bound (`clipped_sample_steps` / (n x guided steps per sample)):"
      % (ung["fm"]["stab"], ung["equifm"]["stab"], se_prop(ung["fm"]["stab"], n_scr["fm"]),
         se_prop(ung["equifm"]["stab"], n_scr["equifm"])))
    A("")
    A("| arm | cells | ours: mean mol stab (min-max) | ours: mean Δ vs unguided | ours: clipped fraction | EquiFM: mean mol stab (min-max) | EquiFM: mean Δ vs unguided | EquiFM: clipped fraction |")
    A("|---|---|---|---|---|---|---|---|")
    for a in GUIDED:
        row = [a]
        cs_all = {b: [c for c in scr[b] if c["arm"] == a and c["w"] == 0.05] for b in BASES}
        row.append(" / ".join(str(len(cs_all[b])) for b in BASES))
        for b in BASES:
            cs = cs_all[b]
            ms = sum(c["stab"] for c in cs) / len(cs)
            cf = sum(clip_frac(c) for c in cs) / len(cs)
            weak[(b, a)] = (ms, ms - ung[b]["stab"], cf)
            row += ["%.3f (%.3f-%.3f)" % (ms, min(c["stab"] for c in cs), max(c["stab"] for c in cs)),
                    "%+.3f" % (ms - ung[b]["stab"]), "%.3f" % cf]
        A("| %s |" % " | ".join(row))
    A("")
    # ---- GUIDED-STEP SYMMETRY. The whole point of basecmp is one property pair
    # and one grid across two generators. If the two bases do not receive the
    # same NUMBER of guided steps in a window, they were not given equal
    # guidance there, and every matched-cell difference in that window carries
    # that asymmetry. Measured from the cells, not assumed from the grid.
    steps_by = {}
    for b in BASES:
        for cell in scr[b]:
            if cell["arm"] == "unguided" or not cell.get("guided_steps"):
                continue
            nb = math.ceil(cell["n"] / float(cell["batch"]))
            if nb:
                steps_by.setdefault((b, cell["t"]), set()).add(
                    round(cell["guided_steps"] / nb))
    tstarts = sorted({t for (_b, t) in steps_by})
    mism = [t for t in tstarts
            if len({tuple(sorted(steps_by.get((b, t), set()))) for b in BASES
                    if (b, t) in steps_by}) > 1]
    A("**Guided steps per sample, measured per base.** The comparison assumes "
      "both generators are guided equally often inside a window.")
    A("")
    A("| t_start | %s | equal? | matched cells here |"
      % " | ".join(LABEL[b] for b in BASES))
    A("|---|" + "---|" * (len(BASES) + 2))
    for t in tstarts:
        vals = [", ".join(str(v) for v in sorted(steps_by.get((b, t), []))) or "--"
                for b in BASES]
        n_match = sum(1 for k in matched if k[3] == t)
        A("| %g | %s | %s | %d |"
          % (t, " | ".join(vals), "no" if t in mism else "yes", n_match))
    A("")
    if mism:
        A("> **The two bases are NOT guided equally at t_start = %s.** Every "
          "matched-cell difference in that window includes the extra step as "
          "well as the generator, and it is the window holding the largest "
          "block of matched cells. The gap is about 1 %% of the guided steps, "
          "far smaller than the chemistry difference in outcome 1, but it is a "
          "systematic asymmetry in exactly the direction this design exists to "
          "measure, so a *small* matched difference in that window should not "
          "be attributed to the generator alone."
          % ", ".join("%g" % t for t in mism))
        A("")
    A("This is descriptive. The clip counts are reported beside the stability, "
      "and nothing on this page tests whether one causes the other.")
    A("")

    # ---- soft vs decoded: moving the type channels without moving the molecule
    gapsd = {}
    A("**Continuous minus decoded in-band**, mean over every cell present (the "
      "protocol selects on decoded because an arm can move the continuous atom-type "
      "channels without changing the molecule that decodes):")
    A("")
    A("| arm | ours: mean (cont - dec) | ours: max | EquiFM: mean (cont - dec) | EquiFM: max |")
    A("|---|---|---|---|---|")
    for a in GUIDED:
        row = [a]
        for b in BASES:
            ds = [c["ib"] - c["ib_dec"] for c in scr[b] if c["arm"] == a]
            gapsd[(b, a)] = (sum(ds) / len(ds), max(ds))
            row += ["%+.4f" % gapsd[(b, a)][0], "%+.4f" % gapsd[(b, a)][1]]
        A("| %s |" % " | ".join(row))
    A("")
    for t0 in (0.25, 0.05):
        A("**Strength profile at t_start = %g** (complete on both bases for every "
          "property and arm). Each entry is ours / EquiFM. Decoded in-band first, "
          "then molecule stability; `*` marks a cell below its base's floor. One "
          "cell's in-band se is about %.3f."
          % (t0, se_prop(0.06, n_scr["fm"])))
        A("")
        A("| property | arm | %s | %s |" % (
            " | ".join("in-band dec w=%g" % w for w in BASECMP_STRENGTHS),
            " | ".join("mol stab w=%g" % w for w in BASECMP_STRENGTHS)))
        A("|---|---|%s" % ("---|" * (2 * len(BASECMP_STRENGTHS))))
        idx = {(b, c["prop"], c["arm"], c["w"], c["t"]): c for b in BASES for c in scr[b]}
        for p in PROPS:
            for a in GUIDED:
                e1, e2 = [], []
                for w in BASECMP_STRENGTHS:
                    co, ce = idx.get(("fm", p, a, float(w), t0)), idx.get(("equifm", p, a, float(w), t0))
                    e1.append("%s / %s" % (f(co["ib_dec"]) if co else "-", f(ce["ib_dec"]) if ce else "-"))
                    e2.append("%s%s / %s%s" % (f(co["stab"]) if co else "-",
                                               "*" if co and not co["clears"] else "",
                                               f(ce["stab"]) if ce else "-",
                                               "*" if ce and not ce["clears"] else ""))
                A("| %s | %s | %s | %s |" % (p, a, " | ".join(e1), " | ".join(e2)))
        A("")
        A("Unguided mu for reference: decoded in-band %s / %s, molecule stability %s / %s."
          % (f(ung["fm"]["ib_dec"]), f(ung["equifm"]["ib_dec"]), f(ung["fm"]["stab"]),
             f(ung["equifm"]["stab"])))
        A("")

    # ---- matched comparison summary
    A("## 7. Like-for-like: ours against EquiFM on matched cells")
    A("")
    A("Both bases went through **one** property pair, one δ, one target, one set of "
      "molecule sizes, one strength grid and one 100-step Euler sampler, so a cell "
      "present on both isolates the generator further than anything else in this "
      "project does. It does not isolate it perfectly: each generator brings its "
      "own path geometry and its own per-arm approximation, and the cells ran on "
      "two MIG slice types (caveats 6 and 8). There are %d "
      "such cells. Each entry is EquiFM minus ours; z is unpaired, because the two "
      "generators share molecule sizes but no molecules. Counts of |z| >= %g are "
      "descriptive: one seed, no multiplicity correction, and the cells are not "
      "independent of one another (one arm's five strengths share its noise). "
      "**They are not verdicts.**" % (len(matched), Z_MARK))
    A("")
    A("| property | arm | matched cells | mean Δ in-band dec | EquiFM higher, z >= %g | ours higher, z <= -%g | mean Δ in-band cont | mean Δ mol stab | EquiFM stabler, z >= %g | ours stabler, z <= -%g | mean Δ validity | cells clearing floor: ours / EquiFM |"
      % (Z_MARK, Z_MARK, Z_MARK, Z_MARK))
    A("|---|---|---|---|---|---|---|---|---|---|---|---|")
    idx = {(b, c["prop"], c["arm"], c["w"], c["t"]): c for b in BASES for c in scr[b]}
    mrows = []
    for k in matched:
        co, ce = idx[("fm",) + k], idx[("equifm",) + k]
        mrows.append({
            "k": k, "o": co, "e": ce,
            "d_ibd": ce["ib_dec"] - co["ib_dec"],
            "z_ibd": z_unpaired(co["ib_dec"], co["n"], ce["ib_dec"], ce["n"]),
            "d_ib": ce["ib"] - co["ib"],
            "z_ib": z_unpaired(co["ib"], co["n"], ce["ib"], ce["n"]),
            "d_st": ce["stab"] - co["stab"],
            "z_st": z_unpaired(co["stab"], co["n"], ce["stab"], ce["n"]),
            "d_va": ce["valid"] - co["valid"],
            "z_va": z_unpaired(co["valid"], co["n"], ce["valid"], ce["n"])})
    summ = {}
    for p in PROPS:
        for a in GUIDED + ("unguided",):
            rs = [r for r in mrows if r["k"][0] == p and r["k"][1] == a]
            if not rs:
                continue
            s_ = {"n": len(rs),
                  "d_ibd": sum(r["d_ibd"] for r in rs) / len(rs),
                  "e_hi": sum(1 for r in rs if r["z_ibd"] >= Z_MARK),
                  "o_hi": sum(1 for r in rs if r["z_ibd"] <= -Z_MARK),
                  "d_ib": sum(r["d_ib"] for r in rs) / len(rs),
                  "d_st": sum(r["d_st"] for r in rs) / len(rs),
                  "e_st": sum(1 for r in rs if r["z_st"] >= Z_MARK),
                  "o_st": sum(1 for r in rs if r["z_st"] <= -Z_MARK),
                  "d_va": sum(r["d_va"] for r in rs) / len(rs),
                  "cl_o": sum(r["o"]["clears"] for r in rs),
                  "cl_e": sum(r["e"]["clears"] for r in rs)}
            summ[(p, a)] = s_
            A("| %s | %s | %d | %+.4f | %d | %d | %+.4f | %+.3f | %d | %d | %+.3f | %d / %d |"
              % (p, a, s_["n"], s_["d_ibd"], s_["e_hi"], s_["o_hi"], s_["d_ib"], s_["d_st"],
                 s_["e_st"], s_["o_st"], s_["d_va"], s_["cl_o"], s_["cl_e"]))
    A("")
    tot = {"n": len(mrows),
           "e_hi": sum(1 for r in mrows if r["z_ibd"] >= Z_MARK),
           "o_hi": sum(1 for r in mrows if r["z_ibd"] <= -Z_MARK),
           "e_hi_c": sum(1 for r in mrows if r["z_ib"] >= Z_MARK),
           "o_hi_c": sum(1 for r in mrows if r["z_ib"] <= -Z_MARK),
           "e_st": sum(1 for r in mrows if r["z_st"] >= Z_MARK),
           "o_st": sum(1 for r in mrows if r["z_st"] <= -Z_MARK),
           "d_ibd": sum(r["d_ibd"] for r in mrows) / len(mrows),
           "d_st": sum(r["d_st"] for r in mrows) / len(mrows),
           "both_clear": sum(1 for r in mrows if r["o"]["clears"] and r["e"]["clears"]),
           "o_only": sum(1 for r in mrows if r["o"]["clears"] and not r["e"]["clears"]),
           "e_only": sum(1 for r in mrows if r["e"]["clears"] and not r["o"]["clears"]),
           "neither": sum(1 for r in mrows if not r["o"]["clears"] and not r["e"]["clears"])}
    for p in PROPS:
        rs = [r for r in mrows if r["k"][0] == p]
        tot["d_ibd_" + p] = sum(r["d_ibd"] for r in rs) / len(rs)
        tot["e_hi_" + p] = sum(1 for r in rs if r["z_ibd"] >= Z_MARK)
        tot["o_hi_" + p] = sum(1 for r in rs if r["z_ibd"] <= -Z_MARK)
        tot["n_" + p] = len(rs)
    A("Over all %d matched cells: decoded in-band, EquiFM higher at z >= %g in %d "
      "and ours higher at z <= -%g in %d (continuous in-band: %d and %d). Molecule "
      "stability: EquiFM stabler at z >= %g in %d and ours in %d. Mean Δ decoded "
      "in-band %+.4f, mean Δ molecule stability %+.3f. Against each base's own "
      "floor: %d cells clear on both, %d on ours only, %d on EquiFM only, %d on "
      "neither."
      % (tot["n"], Z_MARK, tot["e_hi"], Z_MARK, tot["o_hi"], tot["e_hi_c"], tot["o_hi_c"],
         Z_MARK, tot["e_st"], tot["o_st"], tot["d_ibd"], tot["d_st"], tot["both_clear"],
         tot["o_only"], tot["e_only"], tot["neither"]))
    A("")

    A("<!-- WHAT-IT-SHOWS -->")
    A("")
    A("<!-- CAVEATS -->")
    A("")

    # ---- script changes
    A("## 10. Script changes this page rests on")
    A("")
    A("Two, both minimal and backward compatible.")
    A("")
    A("1. **`proj1/scripts/basecmp_freeze.py`** — `load_screen` compared each "
      "property's δ across cells for exact equality at 12 decimals. δ is "
      "recomputed inside every array task in float32 on the GPU, so a real screen "
      "differs in about the 7th significant figure and the check refused every "
      "one (section 1.2). It now calls a new `delta_consistent()` with a relative "
      "tolerance of 1e-5. A genuine change of δ rule is 5-60 % away (local "
      "against global, section 2), so the looser check still catches it. All 30 "
      "gates in `proj1/tests/test_basecmp.py` pass, `freeze_refuses_delta` "
      "included. **This fix changes no number on this page**, which makes none of "
      "the picks the freeze exists to make.")
    A("2. **`proj1/scripts/basecmp_partial_table.py`** — new; it generates this "
      "page and nothing else. It is read-only on `results/` and `logs/` and "
      "writes only the paths given to `--md-out` and `--csv-out`. It makes no "
      "pick and issues no verdict, because the screen is incomplete.")
    A("")
    A("Nothing else was changed. `basecmp_table.py`, `basecmp_size.py`, "
      "`transfer_sweep.py` and the cluster scripts are untouched, and no file "
      "under `results/` or `logs/` was modified.")
    A("")
    A("One defect is **recorded and not fixed**, because it is outside this "
      "page's scope: `transfer_sweep.py:1344` runs the same exact-equality δ "
      "check over its own stages, but only prints `INCONSISTENT delta` and "
      "carries on, so it mislabels float noise as a rule change without stopping "
      "anything. Whoever next touches that reader should give it the same "
      "tolerance.")
    A("")

    # ---- regenerate
    A("## Regenerate")
    A("")
    A("```")
    A("cd \"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1\"")
    A("# this page (read-only on results/ and logs/)")
    A(".venv/Scripts/python.exe proj1/scripts/basecmp_partial_table.py --md-out docs/results/BASECMP_PARTIAL_RESULTS.md")
    A("# optional: every screen cell as csv, anywhere outside results/")
    A(".venv/Scripts/python.exe proj1/scripts/basecmp_partial_table.py --csv-out <scratch>/basecmp_partial_cells.csv")
    A("# the official readers' refusals, as quoted in section 1.2")
    A(".venv/Scripts/python.exe proj1/scripts/basecmp_table.py --backend equifm")
    A(".venv/Scripts/python.exe proj1/scripts/basecmp_freeze.py --backend fm")
    A(".venv/Scripts/python.exe proj1/scripts/basecmp_freeze.py --backend equifm")
    A("# the gates, after the delta-tolerance fix")
    A(".venv/Scripts/python.exe proj1/tests/test_basecmp.py")
    A("```")
    A("")

    # ---- appendix: every cell
    A("## Appendix A. Every screen cell, per base")
    A("")
    A("n = %d per cell, seed %d. Bias and residual sd are over finite rows with the "
      "population sd (the `btvg2_table.py` convention), from the `.permol.pt` "
      "sidecars. In-band counts a non-finite sample as a miss, over all n. `se` is "
      "the binomial se of that cell's own fraction. It is **not** the se of a "
      "difference: within a base, cells share sizes, target and initial noise, so "
      "within-base comparisons are paired, and that paired se is not computed here. "
      "In-band recomputed from the sidecars matches the cell files to %.1e. "
      "Non-finite samples across all screen cells: ours %d, EquiFM %d."
      % (n_scr["fm"], BASECMP_SEED, ib_err, sum(c["nonfinite"] for c in scr["fm"]),
         sum(c["nonfinite"] for c in scr["equifm"])))
    A("")
    for b in BASES:
        A("### A.%d %s (floor %.4f; the alpha and gap floor is borrowed from mu's unguided cell)"
          % (BASES.index(b) + 1, LABEL[b], floor[b]))
        A("")
        A("| property | arm | t_start | w | in-band (cont / dec) | +- se (cont / dec) | MAE/δ (cont / dec) | bias/δ (cont / dec) | resid sd/δ (cont / dec) | mol stab | floor | validity | uniq | oracle-2 in-band (cont / dec) |")
        A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        order = ("unguided",) + GUIDED
        for c in sorted(scr[b], key=lambda c: (PROPS.index(c["prop"]), order.index(c["arm"]),
                                               c["t"], c["w"])):
            A("| %s | %s | %g | %g | %s / %s | %s / %s | %s / %s | %s / %s | %s / %s | %s | %s | %s | %s | %s / %s |"
              % (c["prop"], c["arm"], c["t"], c["w"], f(c["ib"]), f(c["ib_dec"]),
                 f(se_prop(c["ib"], c["n"])), f(se_prop(c["ib_dec"], c["n"])),
                 f(c["mae"], 2), f(c["mae_dec"], 2), f(c["bias"], 2), f(c["bias_dec"], 2),
                 f(c["sd"], 2), f(c["sd_dec"], 2), f(c["stab"]),
                 "ok" if c["clears"] else "**below**", f(c["valid"]), f(c["uniq"]),
                 f(c["o2_ib"]), f(c["o2_ib_dec"])))
        A("")

    A("## Appendix B. Every matched cell: EquiFM minus ours")
    A("")
    A("Unpaired z (independent generators, n = %d each). `floor` shows each base "
      "against its own floor (ours %.4f, EquiFM %.4f)." % (n_scr["fm"], floor["fm"], floor["equifm"]))
    A("")
    A("| property | arm | t_start | w | in-band dec: ours / EquiFM | Δ | z | in-band cont Δ | z | mol stab: ours / EquiFM | Δ | z | validity Δ | z | MAE dec/δ: ours / EquiFM | floor: ours / EquiFM |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = ("unguided",) + GUIDED
    for r in sorted(mrows, key=lambda r: (PROPS.index(r["k"][0]), order.index(r["k"][1]),
                                          r["k"][3], r["k"][2])):
        co, ce = r["o"], r["e"]
        A("| %s | %s | %g | %g | %s / %s | %+.3f | %s | %+.3f | %s | %s / %s | %+.3f | %s | %+.3f | %s | %s / %s | %s / %s |"
          % (r["k"][0], r["k"][1], r["k"][3], r["k"][2], f(co["ib_dec"]), f(ce["ib_dec"]),
             r["d_ibd"], fz(r["z_ibd"]), r["d_ib"], fz(r["z_ib"]), f(co["stab"]), f(ce["stab"]),
             r["d_st"], fz(r["z_st"]), r["d_va"], fz(r["z_va"]), f(co["mae_dec"], 2),
             f(ce["mae_dec"], 2), "ok" if co["clears"] else "below",
             "ok" if ce["clears"] else "below"))
    A("")

    # ---------------------------------------------------------- stdout summary
    say("basecmp PARTIAL: screen ours %d/363, EquiFM %d/363, matched %d, failed %d"
        % (len(scr["fm"]), len(scr["equifm"]), len(matched), len(failed)))
    say("floors: ours %.4f (unguided %.4f), EquiFM %.4f (unguided %.4f)"
        % (floor["fm"], ung["fm"]["stab"], floor["equifm"], ung["equifm"]["stab"]))
    say("matched: EquiFM higher in-band dec z>=3 in %d, ours in %d; stabler E %d, O %d"
        % (tot["e_hi"], tot["o_hi"], tot["e_st"], tot["o_st"]))
    say("never clear floor: ours %s | EquiFM %s" % (never["fm"], never["equifm"]))
    say("sizes identical across sidecars: %s (%d); agg-vs-sidecar max err %.2e; "
        "in-band recompute max err %.2e" % (idx_ok, n_side, agg_err, ib_err))

    ctx = {"plan": plan, "floor": floor, "ung": ung, "tot": tot, "summ": summ,
           "never": never, "scr": scr, "n_scr": n_scr, "miss": miss,
           "matched": matched, "tot_h": tot_h, "pred_h": pred_h, "rr": rr,
           "zs": zs, "zv": zv, "zi": zi, "weak": weak, "gapsd": gapsd,
           "dm_ratio": dm_ratio, "sl": sl, "fmarg": fmarg, "tr": tr,
           "jdev": ctx_jdev, "dupst": ctx_dup, "screen_jobs": screen_jobs}

    if args.csv_out:
        cols = ["base", "prop", "arm", "w", "t", "n", "ib", "ib_dec", "mae", "mae_dec",
                "bias", "bias_dec", "sd", "sd_dec", "stab", "clears", "valid", "uniq",
                "uvps", "o2_ib", "o2_ib_dec", "nonfinite", "seconds", "delta", "file"]
        with open(args.csv_out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            for b in BASES:
                for c in scr[b]:
                    wr.writerow(c)
        say("wrote %s" % args.csv_out)

    if args.md_out:
        text = "\n".join(L) + "\n"
        text = narrative(text, ctx)
        tmp = args.md_out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, args.md_out)
        say("wrote %s" % args.md_out)
    return 0


def narrative(text, ctx):
    """The prose sections, filled in at the three markers. Every number in them
    is read from `ctx`, which the table code computed; none is typed in."""
    c = ctx
    scr, ung, floor, tot, summ = c["scr"], c["ung"], c["floor"], c["tot"], c["summ"]
    n = c["n_scr"]["fm"]

    # ---------------------------------------------------------- section 0
    W = []
    W.append("## 0. What the run was, and why it was cancelled")
    W.append("")
    W.append("The base-model comparison (`docs/protocol/BASECMP_PROTOCOL.md`, "
             "pre-registered 25 Sep) asked how much of a guidance method's "
             "behaviour belongs to the method and how much to the generator it "
             "runs on. It ran **our FM base** (`fm_last.pt`) and **EquiFM** "
             "through **one external property pair**: guide TFG `tf_predict_<p>`, "
             "oracle TFG `evaluate_<p>` (EDM's classifier), second oracle OC-Flow's "
             "clean EGNN. δ, target, molecule sizes and sampler settings were the "
             "same on both, so a difference between the two columns can be put on "
             "the generator. All seven arms ran (unguided, plug, tmpd, lgd_mc, tfg, "
             "btvg, btvg_var) at the q90 target, with strength and guidance "
             "start-time screened jointly. The chain was probe -> size -> screen "
             "-> plan/refine -> freeze (one `floor` and one `free` pick per arm) -> "
             "the v2 full run on EquiFM at both picks -> table.")
    W.append("")
    W.append("It was **cancelled on 26 Sep, part-way through the screen**, because "
             "protocol v3 superseded it. v3 differs on target, strength rule, arm "
             "set, property pair and δ rule (section 9), so these cells could not "
             "have been pooled with v3's either way. The tree was pulled after the "
             "cancellation and had never been tabulated; this page is its only "
             "write-up.")
    W.append("")

    # ---------------------------------------------------------- section 8
    S = []
    S.append("## 8. What each outcome shows")
    S.append("")
    S.append("Plain words, with the caveats of section 9 attached to every line. "
             "None of this is a verdict or a pick.")
    S.append("")
    S.append("**1. Chemistry is where the two generators differ, and by a lot.** "
             "Unguided, EquiFM makes molecule-stable molecules %.1f %% of the time "
             "against our %.1f %% (z = %s), and valid ones %.1f %% against %.1f %% "
             "(z = %s). This survives guidance: EquiFM is stabler at z >= 3 in %d "
             "of the %d matched cells, and ours in %d. The mean gap is %+.3f. The "
             "pair, δ, sizes and sampler were identical, so this is the generator "
             "and its path, not the scoring (caveat 6). The feature-scale check "
             "puts both unguided oracle means within 0.2 MAD of QM9's, so a "
             "scoring-scale defect does not explain it either. It is also the one "
             "result here that is far too large to be a single-seed artefact: the "
             "unguided gap alone is %s se."
             % (100 * ung["equifm"]["stab"], 100 * ung["fm"]["stab"], fz(c["zs"]),
                100 * ung["equifm"]["valid"], 100 * ung["fm"]["valid"], fz(c["zv"]),
                tot["e_st"], tot["n"], tot["o_st"], tot["d_st"], fz(c["zs"])))
    S.append("")
    mu_pos = sum(1 for a in GUIDED if summ[("mu", a)]["d_ibd"] > 0)
    gap_neg = sum(1 for a in GUIDED if summ[("gap", a)]["d_ibd"] < 0)
    top = {b: max(scr[b], key=lambda x: x["ib_dec"]) for b in BASES}
    amax = {b: max(x["ib_dec"] for x in scr[b] if x["prop"] == "alpha") for b in BASES}
    S.append("**2. Hitting the target does not differ systematically between the "
             "generators.** Over the %d matched cells the mean difference in "
             "decoded in-band is %+.4f. EquiFM is higher at z >= 3 in %d cells and "
             "ours in %d. Under independence about %.1f of %d cells would pass "
             "|z| >= 3 by chance, and these cells are not independent. The sign "
             "depends on the property. On mu, EquiFM is ahead in %d of 6 guided "
             "arms (mean %+.4f; %d cells at z >= 3 for EquiFM, %d for ours). On "
             "gap, ours is ahead in %d of 6 (mean %+.4f; %d and %d). On alpha the "
             "mean difference is %+.4f (%d and %d), and the best cell on either "
             "base reaches only %.3f decoded in-band. Unguided mu is %.3f against "
             "%.3f decoded (z = %s)."
             % (tot["n"], tot["d_ibd"], tot["e_hi"], tot["o_hi"],
                2 * 0.5 * math.erfc(3 / math.sqrt(2)) * tot["n"], tot["n"],
                mu_pos, tot["d_ibd_mu"], tot["e_hi_mu"], tot["o_hi_mu"],
                gap_neg, tot["d_ibd_gap"], tot["e_hi_gap"], tot["o_hi_gap"],
                tot["d_ibd_alpha"], tot["e_hi_alpha"], tot["o_hi_alpha"],
                max(amax.values()), ung["fm"]["ib_dec"], ung["equifm"]["ib_dec"],
                fz(c["zi"])))
    S.append("")
    S.append("**3. At q90, every arm on both bases sits far from the target.** "
             "Unguided bias is %.1f δ on ours and %.1f δ on EquiFM, with residual "
             "sd %.1f δ and %.1f δ: the band is a sliver of the upper tail. The "
             "highest decoded in-band cell anywhere is %.3f (ours, %s/%s, w = %g, "
             "t = %g), at molecule stability %.3f against a floor of %.3f; on "
             "EquiFM it is %.3f (%s/%s, w = %g, t = %g) at stability %.3f against "
             "%.3f. Both are far below their base's floor. High in-band here is "
             "bought with chemistry, and the strength profiles (section 6) show "
             "the trade cell by cell."
             % (ung["fm"]["bias"], ung["equifm"]["bias"], ung["fm"]["sd"],
                ung["equifm"]["sd"],
                top["fm"]["ib_dec"], top["fm"]["prop"], top["fm"]["arm"],
                top["fm"]["w"], top["fm"]["t"], top["fm"]["stab"], floor["fm"],
                top["equifm"]["ib_dec"], top["equifm"]["prop"], top["equifm"]["arm"],
                top["equifm"]["w"], top["equifm"]["t"], top["equifm"]["stab"],
                floor["equifm"]))
    S.append("")
    cp = {(b, a): (sum(x["clears"] for x in scr[b] if x["arm"] == a),
                   sum(1 for x in scr[b] if x["arm"] == a)) for b in BASES for a in GUIDED}
    S.append("**4. Relative to its own unguided chemistry, EquiFM absorbs guidance "
             "better, but the floor comparison is partly a resolution effect.** Of "
             "the %d matched cells, %d clear both floors, %d clear EquiFM's only, "
             "%d clear ours only and %d clear neither. The extreme case is "
             "`btvg_var`, which clears EquiFM's floor in %d of %d cells and ours in "
             "%d of %d. But ours' floor sits only %.1f se below its unguided value, "
             "against %.1f se on EquiFM (section 4), so noise alone pushes some of "
             "our near-unguided cells under it. Section 6's weakest-strength table "
             "separates the two effects."
             % (tot["n"], tot["both_clear"], tot["e_only"], tot["o_only"], tot["neither"],
                cp[("equifm", "btvg_var")][0], cp[("equifm", "btvg_var")][1],
                cp[("fm", "btvg_var")][0], cp[("fm", "btvg_var")][1],
                c["fmarg"]["fm"], c["fmarg"]["equifm"]))
    S.append("")
    wk = c["weak"]
    lost = [a for a in GUIDED if wk[("fm", a)][1] <= -0.03]
    kept_o = [a for a in GUIDED if a not in lost]
    kept_e = [a for a in GUIDED if abs(wk[("equifm", a)][1]) < 0.03]
    lost_e = [a for a in GUIDED if a not in kept_e]
    S.append("**5. On our base, even the weakest strength costs chemistry for most "
             "arms. On EquiFM it mostly does not.** At w = 0.05, averaged over "
             "every (property, t_start) cell present, %d of our 6 arms sit at "
             "least 0.03 below unguided stability (%s, by %.3f to %.3f), while %s "
             "stay within it (%s). On EquiFM %d of 6 stay within 0.03 of unguided "
             "(%s); the exceptions are %s (%s). The clip binds on %.2f-%.2f of guided "
             "sample-steps across our arms at this strength and %.2f-%.2f across "
             "EquiFM's; those counts are reported beside the stability, not tested "
             "as its cause (the clipping mechanism may not be asserted without a "
             "test of its own). Either way the pattern depends on the generator, "
             "which is what this comparison was built to expose."
             % (len(lost), ", ".join("`%s`" % a for a in lost),
                -max(wk[("fm", a)][1] for a in lost) if lost else float("nan"),
                -min(wk[("fm", a)][1] for a in lost) if lost else float("nan"),
                ", ".join("`%s`" % a for a in kept_o) or "none",
                ", ".join("%s %+.3f" % (a, wk[("fm", a)][1]) for a in kept_o),
                len(kept_e), ", ".join("`%s`" % a for a in kept_e) or "none",
                ", ".join("`%s`" % a for a in lost_e) or "none",
                ", ".join("%s %+.3f" % (a, wk[("equifm", a)][1]) for a in lost_e) or "-",
                min(wk[("fm", a)][2] for a in GUIDED), max(wk[("fm", a)][2] for a in GUIDED),
                min(wk[("equifm", a)][2] for a in GUIDED),
                max(wk[("equifm", a)][2] for a in GUIDED)))
    S.append("")
    g = c["gapsd"]
    others = max(abs(g[(b, a)][0]) for b in BASES for a in GUIDED if a != "tfg")
    S.append("**6. `tfg` moves EquiFM's atom-type channels without moving its "
             "molecules; on our base it does not.** Continuous minus decoded "
             "in-band averages %+.4f for `tfg` on EquiFM (max %+.4f), against "
             "%+.4f on ours. No other arm on either base exceeds %.4f in absolute "
             "mean. So on EquiFM a large part of `tfg`'s continuous in-band never "
             "reaches a decoded molecule. That is why the protocol selects on "
             "decoded, and it is a method property that depends on the generator "
             "(EquiFM's types sit on a VP schedule in its own normalised space; "
             "protocol section 9)."
             % (g[("equifm", "tfg")][0], g[("equifm", "tfg")][1], g[("fm", "tfg")][0],
                others))
    S.append("")
    dmr = [x[3] for x in c["dm_ratio"]]
    wall = [x[1] for x in c["tr"]]
    S.append("**7. Cost: EquiFM is dearer than our base per molecule, and the "
             "sizer's model held.** On device-matched screen cells, guided EquiFM "
             "costs %.2f-%.2fx ours per molecule. Without its margin the "
             "probe-fitted model predicted the screen's own cells to within "
             "%.2f-%.2fx on ours and %.2f-%.2fx on EquiFM. With the %.2fx margin "
             "every finished array task came in under its prediction (wall / "
             "predicted %.2f-%.2f), and the longest ran %.0f min against the "
             "225-min guard. This is the first cost measurement EquiFM has ever "
             "had in this project, and it is the part of the run that is worth "
             "keeping: a later EquiFM queue can be sized from it instead of from "
             "an extrapolation off another network. Read it with the slice caveat "
             "in section 3 -- the probe's EquiFM fit mixes MIG slice types."
             % (min(dmr), max(dmr), min(c["rr"]["fm"]), max(c["rr"]["fm"]),
                min(c["rr"]["equifm"]), max(c["rr"]["equifm"]), c["plan"]["margin"],
                min(wall), max(wall), max(x[2]["done_min"] for x in c["tr"])))
    S.append("")
    d = c["dupst"]
    if d:
        S.append("**8. A fixed seed did not reproduce EquiFM across the two probe "
                 "runs.** The same 10 EquiFM probe cells were computed twice, at "
                 "the same seed, by two concurrent jobs on different MIG slice "
                 "types. %d of %d reproduce their MAE to 4 decimals; the largest "
                 "gaps are %.3f in molecule stability and %.3f in decoded in-band. "
                 "The gap is larger at t_start = 0.05 (%.3f, about 95 guided steps) "
                 "than at 0.75 (%.3f, about 25). This is a run-and-hardware "
                 "replicate, not a same-hardware one, so it bounds reproducibility "
                 "rather than isolating a cause. It matters for reading every "
                 "single-seed EquiFM number on this page."
                 % (d["n_ident"], d["n"], d["dstab"], d["dib"], d["g05"], d["g75"]))
        S.append("")
    S.append("**What the run cannot say.** Nothing about which arm to use on "
             "either base: that is what the freeze and the full run were for, and "
             "neither exists. Nothing about t_start, the axis the protocol called "
             "the project's largest measured single effect -- two of its four "
             "values ran, and only for mu. Nothing about alpha or gap unguided "
             "in-band. And nothing at q50, which is the target the paper reports.")
    S.append("")

    # ---------------------------------------------------------- section 9
    C = []
    C.append("## 9. Caveats")
    C.append("")
    C.append("**1. Cancelled and incomplete, so there are no picks and no "
             "verdicts.** %d of 363 cells are missing on ours and %d on EquiFM. "
             "The refine, freeze and full stages never ran. `basecmp_freeze.py` "
             "refuses this tree (section 1.2) because a pick is an argmax and a "
             "missing cell cannot be seen in one -- it simply never wins. This "
             "page therefore reports cells, not choices, and every ranking-shaped "
             "statement in section 8 is a description of what is present."
             % (len(c["miss"]["fm"]), len(c["miss"]["equifm"])))
    C.append("")
    C.append("**2. q90, and the paper reports q50 only.** The target is the q90 "
             "quantile per property. The paper's scope decision of 26 Sep is q50 "
             "only. **No number on this page goes into the paper or the slides.** "
             "They are also not comparable in either direction: at q90 every arm "
             "sits several δ from the target (outcome 3), which is not where q50 "
             "sits.")
    C.append("")
    C.append("**3. Screen-size n, one seed, so the standard errors are large.** "
             "n = %d per cell, one seed (%d). A single in-band fraction near 0.05 "
             "has se %.4f, near 0.10 has %.4f, and molecule stability near 0.85 "
             "has %.4f. An unpaired difference between two cells has about %.1fx "
             "those. Every table prints its own se. Differences smaller than about "
             "%.3f in in-band are not resolvable here at all."
             % (n, scr["fm"][0]["seed"], se_prop(0.05, n), se_prop(0.10, n),
                se_prop(0.85, n), math.sqrt(2), 3 * math.sqrt(2) * se_prop(0.05, n)))
    C.append("")
    C.append("**4. Not poolable with v3.** v3 (`docs/protocol/FULL_RUN_V3_PROTOCOL.md`) "
             "changed the target (q50), the strength rule (w = 1 for every arm, "
             "unnormalised), the arm set (`btvg` dropped, BDG added), the chemistry "
             "floor (none), the property pair (per backend, not one shared pair) "
             "and the δ rule (the global calibration MAE, not this local one). v3 "
             "says so itself in its section 1. A basecmp cell cannot be put in a v3 "
             "table, and the reverse holds too.")
    C.append("")
    C.append("**5. The in-band bar is optimistic in a way that is the same for "
             "both bases.** δ here is 2 x TFG's oracle MAE on real `val` molecules "
             "near the target. TFG's `evaluate_<p>` trained on an unknown ~50 % of "
             "QM9, so some of those val molecules are probably in its training set "
             "and this MAE is low, making the band tighter than it should be. It is "
             "also f_B's error on **real** molecules, while in-band scores generated "
             "ones. Both effects hit both bases identically, so they do not bear on "
             "the comparison; they do make the absolute coverage numbers "
             "pessimistic. The guide and the oracle are disjoint by inference, not "
             "by construction -- the protocol's section 2 calls this the main "
             "weakness of every number the run produces, and it is unfixable from "
             "the released artifacts.")
    C.append("")
    C.append("**6. The generators differ in more than their weights.** Our base is "
             "a linear flow path with independent Gaussian noise; EquiFM's is a "
             "hybrid geometry-aligned path with types on a VP schedule. `plug` is "
             "exact on both, while `tmpd`, `lgd_mc`, `btvg` and `btvg_var` inherit "
             "a coordinate-block approximation on EquiFM, and `tfg` runs TFG's "
             "schedules on the coordinate clock there. Neither unguided row is a "
             "published number: EquiFM's paper sampled dopri5, and both are "
             "re-measured here under one 100-step Euler sampler. So \"the generator\" "
             "in section 8 means the whole generator-plus-its-approximation, not "
             "the weights alone.")
    C.append("")
    C.append("**7. The chemistry floor for alpha and gap is borrowed from mu's "
             "unguided cell** (section 4), and it sits %.1f se below unguided on "
             "ours against %.1f se on EquiFM, so floor-clearance counts are not "
             "like-for-like near the floor."
             % (c["fmarg"]["fm"], c["fmarg"]["equifm"]))
    C.append("")
    C.append("**8. Two hardware confounds.** Cells ran on two MIG slice types, so "
             "raw seconds are comparable only within a slice type (section 3 gives "
             "the device-matched ratios). And EquiFM does not reproduce exactly "
             "across runs at a fixed seed (outcome 8), which no single-seed number "
             "here can separate from a real effect.")
    C.append("")
    C.append("**9. The counts of |z| >= 3 in section 7 are descriptive.** One "
             "seed, no multiplicity adjustment, and the cells are not independent "
             "-- one arm's five strengths share a generator, a seed and the same "
             "initial noise. v2's verdict rule (z >= 3, Holm-adjusted within a "
             "property, over three seeds at the full n) is not met by anything "
             "here, and is not attempted.")
    C.append("")

    return (text.replace("<!-- WHAT-IT-WAS -->", "\n".join(W))
                .replace("<!-- WHAT-IT-SHOWS -->", "\n".join(S))
                .replace("<!-- CAVEATS -->", "\n".join(C)))


if __name__ == "__main__":
    raise SystemExit(main())
