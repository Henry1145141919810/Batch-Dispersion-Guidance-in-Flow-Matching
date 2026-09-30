"""Every cell of the BDG port run, the full metric block, and every comparator
from that same run, paired molecule for molecule.

    python proj1/scripts/bdg_full_metrics.py
    python proj1/scripts/bdg_full_metrics.py --md-out docs/results/BDG_FULL_METRICS.md

WHAT IS ON DISK. results/bdg_port/cells: the 64 cells of the independent BDG
port that docs/methods/BDG_REVIEW.md section 2 describes, each a .json and a
.permol.pt sidecar. Per property (mu, alpha, gap) and fixed target (q50, q90):

  bdg e0t1                   eta = 0; must be bit-identical to plug w1 (gate)
  bdg e4t0.5 ... e4t1.5      eta = 4, the tau ladder (tau = tau_mult * s)
  bdg e4t1.5o                the one-sided clamp asked to widen (mu, gap only;
                             must be bit-identical to plug w1, gate)
  plug w1 (bdgctl, tgt)      two copies, must be identical (gate)
  plug w4 (tgt)
  unguided w0 (bdgctl)       the run's own control: its mol stability sets the
                             floor (review section 4.6)

All share one seed, n, a single batch of n, one device and one mol_idx order,
so the comparisons are PAIRED. This script verifies that (mol_idx, y and
n_atoms identical in every block) and refuses to pair otherwise. Every metric is
recomputed from the sidecars and checked against the cell JSON.

WORDING. The page follows the review's rulings (sections 1, 4, 6). BDG is plug
with the centring gain fixed at w and only the deviation gain w_eff = 1 + eta*e
servoed on e = V_b/tau^2 - 1. The script refuses to write a page containing any
phrase on the review's banned list (FORBIDDEN below).

CONTEXT FROM OTHER RUNS, NEVER PAIRED. The spread/bias ranges of plug, btvg_var
and btvg come from results/sweep (n = 512, seed 20260921, batch 128; most cells
on B200 MIG slices, but some on the port's own RTX 5080: the page computes and
marks the device of every group against its unguided divisor); the sensitivity
floors come from results/sweep and the v2 full run. They enter as aggregates
only. The handoff's own section 6 numbers are quoted verbatim from
docs/methods/BDG_HANDOFF.md (the author's Betty cells, not in this repo); every
handoff value the page states is parsed from that file, and every review figure
from docs/methods/BDG_REVIEW.md, refusing on a parse failure or a mismatch with
what this script recomputes. Whether results/sweep's fm_last.pt and the port's
fm_ema.pt are the same sampling weights is read from weights/README.md and checked
against the md5s the cells record and the md5 of weights/fm_ema.pt on disk.

min/max of w_eff and the per-step V_b/tau^2 are not in the cell JSONs (they hold
run-means). They are read from audit/bdg_review/weff_traj.json, a per-step re-run
of 12 of these configurations, and used only where its run-mean reproduces the
cell's to 1e-6.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
NL = chr(10)

CELLS_DIR = os.path.join(ROOT, "results", "bdg_port", "cells")
REVIEW = os.path.join(ROOT, "docs", "methods", "BDG_REVIEW.md")
HANDOFF = os.path.join(ROOT, "docs", "methods", "BDG_HANDOFF.md")
WEFF_TRAJ = os.path.join(ROOT, "audit", "bdg_review", "weff_traj.json")
WEIGHTS_DIR = os.path.join(ROOT, "weights")
WEIGHTS_README = os.path.join(WEIGHTS_DIR, "README.md")
V2_UNGUIDED = os.path.join(ROOT, "results", "full", "v2", "n5000", "seed*",
                           "*__unguided__q90__w1__tmin0.5__full.json")
DEFAULT_MD = os.path.join("docs", "results", "BDG_FULL_METRICS.md")

PROPS = ("mu", "alpha", "gap")
TGTS = ("q50", "q90")
BDG_VARIANTS = ("e0t1", "e4t0.5", "e4t0.75", "e4t1", "e4t1.25", "e4t1.5", "e4t1.5o")
ONESIDED_PROPS = ("mu", "gap")
SIGMA = 3.0
FLOOR_FRAC = 0.9
SWEEP_SEED, SWEEP_N = 20260921, 512
SWEEP_ARMS = ("plug", "btvg_var", "btvg")
SWEEP_RE = re.compile(r"^(mu|alpha|gap)__(plug|btvg_var|btvg|unguided)__(q50|q90)"
                      r"__w([0-9.]+)__tmin0\.5__(.+)\.json$")
# Banned anywhere in the page's own voice (every line outside the verbatim
# handoff quote). The review's rulings: sections 1, 4.1-4.5, 5 items 2-5.
FORBIDDEN = ("variance control", "holds its setpoint", "holds the setpoint",
             "no fixed schedule", "negative-weight plug", "negative weight plug",
             "plug at a negative weight", "no existing arm", "only arm",
             "does not reduce to any fixed", "irreducible", "decoupled",
             "we can prove", "by definition", "first-class control",
             "does not reach", "never reach")
# Claims the review ruled against. Outside the quote they may appear only on a
# line that states the ruling (contains "ruled"). "1.010" alone is not banned: it
# is also a legitimate sd ratio in the tables; the yardstick is "1.010x".
RULED_ONLY = (r"1\.010\s*(x|×)", r"scientific core", r"comparison is valid",
              r"fails floor")
GEN_KEYS = ("gen_fwd", "gen_vjp", "gen_jvp")
GUIDE_KEYS = ("guide_fwd", "guide_bwd", "guide_hvp")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def refuse(problems, why):
    print("REFUSING: " + why)
    for p in problems:
        print("  * " + p)
    raise SystemExit(1)


def isnan(v):
    return v is None or (isinstance(v, float) and v != v)


def f(v, fmt="%.4f"):
    return "n/a" if isnan(v) else fmt % v


def zf(z):
    if isnan(z):
        return "n/a"
    s = "%+.2f" % z
    return "**%s**" % s if abs(z) >= SIGMA else s


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def gauss_inband(b, s):
    """P(|X| <= 1) for X ~ N(b, s^2): the band is +-1 in units of delta."""
    return Phi((1.0 - b) / s) - Phi((-1.0 - b) / s)


def best_scale(b, s):
    """The sd scale k that maximises gauss_inband(b, k*s) at fixed bias b.

    d/dsigma [Phi((1-b)/sigma) - Phi((-1-b)/sigma)] = 0 has, for |b| > 1, the
    single root sigma*^2 = 2|b| / ln((|b|+1)/(|b|-1)); for |b| <= 1 the in-band
    probability rises monotonically as sigma -> 0 (tighter is always better),
    so there is no finite optimum and None is returned. Checked against a grid.
    """
    ab = abs(b)
    if ab <= 1.0:
        return None
    sig = math.sqrt(2.0 * ab / math.log((ab + 1.0) / (ab - 1.0)))
    grid = [sig * (0.5 + 0.001 * i) for i in range(1001)]
    g = max(grid, key=lambda x: gauss_inband(b, x))
    assert abs(gauss_inband(b, g) - gauss_inband(b, sig)) < 1e-9, (b, s, sig, g)
    return sig / s


def sidak(z, m):
    p1 = math.erfc(abs(z) / math.sqrt(2.0))
    if p1 >= 1.0:
        return 1.0
    return -math.expm1(m * math.log1p(-p1))


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
                r[order[k]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def parse_variant(v):
    m = re.fullmatch(r"e([0-9.]+)t([0-9.]+)(o?)", v)
    return float(m.group(1)), float(m.group(2)), bool(m.group(3))


def row_specs(prop):
    """(arm, w, variant, label) in display order."""
    rs = [("unguided", 0.0, "bdgctl", "unguided w0"),
          ("plug", 1.0, "bdgctl", "plug w1"),
          ("plug", 1.0, "tgt", "plug w1 (tgt copy)"),
          ("plug", 4.0, "tgt", "plug w4")]
    for v in BDG_VARIANTS:
        if v.endswith("o") and prop not in ONESIDED_PROPS:
            continue
        rs.append(("bdg", 1.0, v, "bdg " + v))
    return rs


def ladder(prop):
    """The eta = 4 two-sided tau ladder, sorted by tau_mult."""
    vs = [v for v in BDG_VARIANTS if v.startswith("e4") and not v.endswith("o")]
    return sorted(vs, key=lambda v: parse_variant(v)[1])


# ---------------------------------------------------------------------------
# load and verify
# ---------------------------------------------------------------------------

def load():
    import torch
    cells, problems = {}, []
    for p in sorted(glob.glob(os.path.join(CELLS_DIR, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            r = json.load(fh)
        side = p[:-5] + ".permol.pt"
        key = (r["prop"], r["target_name"], r["arm"], float(r["w"]), r["variant"])
        if key in cells:
            problems.append("duplicate cell %s" % (key,))
            continue
        if not os.path.exists(side):
            problems.append("no sidecar for %s" % os.path.basename(p))
            continue
        cells[key] = (r, torch.load(side, weights_only=False),
                      os.path.relpath(p, ROOT).replace(os.sep, "/"))
    want = {(p, t, a, w, v) for p in PROPS for t in TGTS for (a, w, v, _) in row_specs(p)}
    for k in sorted(want - set(cells), key=str):
        problems.append("missing cell %s" % (k,))
    for k in sorted(set(cells) - want, key=str):
        problems.append("unexpected cell %s" % (k,))
    return cells, problems


def check_consistency(cells):
    problems = []
    rows = [c[0] for c in cells.values()]
    for key in ("n", "seed", "steps", "solver", "t_min_guide", "clip", "batch", "fm",
                "k_delta", "delta_source"):
        vals = {json.dumps(r.get(key)) for r in rows}
        if len(vals) > 1:
            problems.append("mixed %s: %s" % (key, sorted(vals)))
    for key in ("fm_md5", "device", "torch", "cuda"):
        vals = {(r.get("prov") or {}).get(key) for r in rows}
        if len(vals) > 1:
            problems.append("mixed prov.%s: %s" % (key, sorted(map(str, vals))))
    if rows[0]["batch"] != rows[0]["n"]:
        problems.append("batch %s != n %s: not one batch" % (rows[0]["batch"], rows[0]["n"]))
    for p in PROPS:
        vals = {r["delta"] for r in rows if r["prop"] == p}
        if len(vals) > 1:
            problems.append("%s: mixed delta %s" % (p, sorted(vals)))
        for t in TGTS:
            vals = {r["target"] for r in rows if r["prop"] == p and r["target_name"] == t}
            if len(vals) > 1:
                problems.append("%s/%s: mixed target %s" % (p, t, sorted(vals)))
    for (p, t, a, w, v), (r, pm, _) in cells.items():
        if float(r.get("w_applied", w)) != w:
            problems.append("%s/%s %s %s: w_applied %s != w %s" % (p, t, a, v, r.get("w_applied"), w))
        if pm["f_B"].numel() != r["n"]:
            problems.append("%s/%s %s %s: sidecar rows %d != n %d" % (p, t, a, v, pm["f_B"].numel(), r["n"]))
        if a != "bdg":
            continue
        eta, mult, one = parse_variant(v)
        if (float(r["bdg_eta"]) != eta or float(r["bdg_tau_mult"]) != mult
                or bool(r["bdg_onesided"]) != one):
            problems.append("%s/%s %s: variant name disagrees with bdg_eta/tau_mult/onesided" % (p, t, v))
        if abs(r["bdg_tau"] - mult * r["bdg_s"]) > 1e-6 * max(1.0, abs(r["bdg_tau"])):
            problems.append("%s/%s %s: bdg_tau != tau_mult * bdg_s" % (p, t, v))
    return problems


def check_pairing(cells):
    import torch
    problems = []
    ref0 = cells[(PROPS[0], TGTS[0], "unguided", 0.0, "bdgctl")][1]["mol_idx"]
    same_order_everywhere = True
    for p in PROPS:
        for t in TGTS:
            ref = cells[(p, t, "unguided", 0.0, "bdgctl")][1]
            for (a, w, v, lab) in row_specs(p):
                pm = cells[(p, t, a, w, v)][1]
                for k in ("mol_idx", "y", "n_atoms"):
                    if not torch.equal(pm[k], ref[k]):
                        problems.append("%s/%s %s: %s differs from unguided's; will not pair"
                                        % (p, t, lab, k))
                if not torch.equal(pm["mol_idx"], ref0):
                    same_order_everywhere = False
    return problems, same_order_everywhere


def metrics(r, pm):
    """The full block for one cell, recomputed from its sidecar."""
    n = pm["f_B"].numel()
    d = r["delta"]
    fin = pm["finite"].bool()
    y = pm["y"]
    ib = ((pm["f_B"] - y).abs() <= d) & fin
    ibd = ((pm["f_B_dec"] - y).abs() <= d) & fin
    e = (pm["f_B"].double() - y.double())[fin]
    ed = (pm["f_B_dec"].double() - y.double())[fin]
    ms = pm["mol_stable"].bool() & fin
    va = pm["valid"].bool() & fin
    smi = [s for s, ok in zip(pm["smiles"], va.tolist()) if ok]
    cost = r.get("cost") or {}
    guided = sum((r.get("schedule_used") or {}).values())
    p_ib = float(ib.double().mean())
    return {
        "n": n,
        "in_band": p_ib,
        "se": math.sqrt(max(p_ib * (1.0 - p_ib), 0.0) / n),
        "in_band_dec": float(ibd.double().mean()),
        "mae": float(e.abs().mean()) / d,
        "mae_dec": float(ed.abs().mean()) / d,
        "bias": float(e.mean()) / d,
        "bias_dec": float(ed.mean()) / d,
        "resid": float(e.std(unbiased=False)) / d,
        "resid_dec": float(ed.std(unbiased=False)) / d,
        "sd_fa": float(pm["f_A"][fin].double().std(unbiased=False)),
        "sd_fb": float(pm["f_B"][fin].double().std(unbiased=False)),
        "sd_fbd": float(pm["f_B_dec"][fin].double().std(unbiased=False)),
        "n_stab": int(ms.sum()),
        "mol_stab": float(ms.double().mean()),
        "valid": float(va.double().mean()),
        "n_valid": len(smi),
        "uniq": len(set(smi)) / max(len(smi), 1),
        "nonfinite": int((~fin).sum()),
        "atom_stab": r["atom_stability"],
        "div": r["diversity_mean_pairwise"],
        "gap": r["guide_eval_gap_mean"] / d,
        "clip": (r.get("clipped_sample_steps", 0) / (n * guided)) if guided else None,
        "guided_steps": guided,
        "gen": sum(cost.get(k, 0) for k in GEN_KEYS),
        "guide": sum(cost.get(k, 0) for k in GUIDE_KEYS),
        "cost": cost,
        "diag": r.get("diag") or {},
    }


def check_json_agreement(cells, M):
    problems, worst = [], collections.defaultdict(float)
    for k, (r, pm, path) in cells.items():
        m = M[k]
        pairs = (("in_band", "in_band_fraction", 1e-12, False),
                 ("in_band_dec", "in_band_fraction_dec", 1e-12, False),
                 ("mol_stab", "mol_stability", 1e-12, False),
                 ("valid", "validity", 1e-12, False),
                 ("uniq", "uniqueness_of_valid", 1e-12, False),
                 ("mae", "prop_mae_eval", 1e-5, True),
                 ("mae_dec", "prop_mae_eval_dec", 1e-5, True))
        for mk, jk, tol, scaled in pairs:
            ref = r[jk] / r["delta"] if scaled else r[jk]
            diff = abs(m[mk] - ref) / (max(abs(ref), 1e-12) if scaled else 1.0)
            worst[mk] = max(worst[mk], diff)
            if diff > tol:
                problems.append("%s: recomputed %s %.8g vs JSON %.8g" % (path, mk, m[mk], ref))
        if m["nonfinite"] != r["n_nonfinite"]:
            problems.append("%s: non-finite %d vs JSON %d" % (path, m["nonfinite"], r["n_nonfinite"]))
    return problems, worst


# ---------------------------------------------------------------------------
# external reads: the review, the handoff, the trajectories, other runs
# ---------------------------------------------------------------------------

def _section(txt, head, nxt):
    i = txt.index(head)
    return txt[i:txt.index(nxt, i + len(head))]


def read_review():
    """Every review figure the page states, parsed; refuses if any is missing."""
    txt = open(REVIEW, encoding="utf-8").read()
    miss = []

    def need(pat, where=txt, flags=0, what=None):
        m = re.search(pat, where, flags)
        if not m:
            miss.append(what or pat)
        return m

    m = need(r"design effect on in-band is only ([0-9.]+)\s*[–-]\s*([0-9.]+)")
    t = need(r"tightening to BDG's ([0-9.]+)")
    s42 = _section(txt, "### 4.2", "### 4.3")
    s43 = _section(txt, "### 4.3", "### 4.4")
    s44 = _section(txt, "### 4.4", "### 4.5")
    s45 = _section(txt, "### 4.5", "### 4.6")
    s46 = _section(txt, "### 4.6", "### 4.7")
    s47 = _section(txt, "### 4.7", "### 4.8")
    s5 = _section(txt, "## 5.", "## 6.")
    s8 = _section(txt, "## 8.", "## 9.")
    # the rulings on 6.1's "Widening replicates on all three properties" (4.5 / R5) and
    # 6.2's "Widening lowers in_band every time" (4.4 / R2)
    alpha45 = need(r"\*\*BDG's own alpha widening \(([0-9.]+) / ([0-9.]+)\) is ([0-9.]+) and "
                   r"([0-9.]+) se from none\*\*.*?\(to ~([0-9.]+) and ([0-9.]+) se\), so it is "
                   r"(borderline rather than null)", s45, re.S, what="4.5 alpha widening ruling")
    r5 = need(r"^\| R5 \| [^\n]*\| (partially defended \([0-9.]+\)) — \*(alpha borderline once "
              r"paired)\* \|$", s8, re.M, what="8 R5 ruling")
    r2 = need(r"\*Ruling R2: (partially defended \([0-9.]+\))\. The premise is right at (q\d+) and "
              r"wrong at (q\d+)\.\*", s44, what="4.4 R2 ruling")
    r2b = need(r"are false at (q\d+)\*\*, where widening slightly \*raises\*\s+coverage", s44,
               what="4.4 widening raises coverage at the flipped target")
    drag = need(r"It is worth ([0-9.]+)–([0-9.]+) δ over the ladder, so do not call the two "
                r"gains \"decoupled\"", s42, what="4.2 drag range")
    w42 = need(r"Measured on the port's paired cells at (q\d+), the widening cells", s42,
               what="4.2 target of the widening mean shift")
    rep = need(r"open-loop replay of a schedule from another seed \| \*\*([0-9.]+)–([0-9.]+) %\*\* "
               r"\((\d+)/(\d+) configurations\)", s43, what="4.3 replay remainder")
    scope = need(r"\*\*Scope:\*\* (one seed pair), (\w+) and (\w+) only, τ_mult ([0-9.]+) and "
                 r"([0-9.]+), n = (\d+)", s43, what="4.3 replay scope")
    tgt = need(r"q50 versus `dist` cannot be separated", s44, what="4.4 q50-or-dist")
    se_sd = need(r"se of an sd ratio is ([0-9.]+) / ([0-9.]+) / ([0-9.]+) for (\w+) / (\w+) / "
                 r"(\w+)", s45, what="4.5 sd-ratio se")
    rows45 = re.findall(r"^\| `(\w+)` (\w+) (q\d+), w = ([0-9.]+) \| ([0-9.]+)× \| "
                        r"(?:([0-9.]+)×|—) \| ([^|]+) \|", s45, re.M)
    if not rows45:
        miss.append("4.5 widening table")
    yard = need(r"The 1\.010× yardstick is inside noise", s45, what="4.5 yardstick ruling")
    knife = need(r"It is −([0-9.]+) se below ([0-9.]+) and −([0-9.]+) se below the pooled "
                 r"([0-9.]+)", s46, what="4.6 knife-edge z")
    knife2 = need(r"Report it as knife-edge, not as \"FAILS\"", s46, what="4.6 ruling")
    p47 = need(r"plug alone, with no variance term, sweeps it by \*\*([0-9.]+)–([0-9.]+) %\*\*",
               s47, what="4.7 plug range")
    i12 = need(r"^12\. \*\*Controls and tests for any §4\.4 row\*\*.*?matched signed-w plug "
               r"control", s5, re.M | re.S, what="5 item 12 signed-w plug control")
    if miss:
        raise SystemExit("REFUSING: could not read from %s: %s" % (REVIEW, "; ".join(miss)))
    if r2.group(3) != r2b.group(1):
        raise SystemExit("REFUSING: review 4.4's R2 says wrong at %s but its widening sentence "
                         "says false at %s" % (r2.group(3), r2b.group(1)))
    return {"alpha45": dict(h=(alpha45.group(1), alpha45.group(2)),
                            unpaired=(alpha45.group(3), alpha45.group(4)),
                            paired=(alpha45.group(5), alpha45.group(6)),
                            word=alpha45.group(7), r5=r5.group(1), r5_note=r5.group(2)),
            "r2": dict(ruling=r2.group(1), right=r2.group(2), wrong=r2.group(3)),
            "deff": (m.group(1), m.group(2)), "tight": float(t.group(1)),
            "tight_s": t.group(1), "drag": (drag.group(1), drag.group(2)),
            "w42_tgt": w42.group(1),
            "replay": (rep.group(1), rep.group(2), int(rep.group(3)), int(rep.group(4))),
            "replay_scope": dict(pairs=scope.group(1), props=(scope.group(2), scope.group(3)),
                                 mults=(scope.group(4), scope.group(5)), n=int(scope.group(6))),
            "se_sd": dict(zip(se_sd.groups()[3:], map(float, se_sd.groups()[:3]))),
            "w45": [dict(arm=a, prop=p, tgt=tg, w=float(w), s1=float(s1),
                         s2=float(s2) if s2 else None, floor=fl.strip())
                    for a, p, tg, w, s1, s2, fl in rows45],
            "knife": dict(z=float(knife.group(1)), floor=float(knife.group(2)),
                          z_pooled=float(knife.group(3)), pooled=float(knife.group(4))),
            "plug47": (p47.group(1), p47.group(2))}


def read_handoff():
    txt = open(HANDOFF, encoding="utf-8").read()
    i = txt.index("### 6.1")
    j = txt.index(NL + "## 7.", i)
    sec = txt[i:j].rstrip()
    if sec.endswith("---"):              # the handoff's section separator, not content
        sec = sec[:-3].rstrip()
    s61 = sec[:sec.index("### 6.2")]
    s62 = sec[sec.index("### 6.2"):sec.index("### 6.3")]
    s63 = sec[sec.index("### 6.3"):]
    lines = [l for l in s61.splitlines() if l.startswith("|")]
    header = [h.strip() for h in lines[0].strip().strip("|").split("|")]
    cols = [re.match(r"(e\d+t[0-9.]+)", h).group(1) for h in header[1:]]
    t61 = {}
    for l in lines[2:]:
        cs = [x.strip().strip("*").strip() for x in l.strip().strip("|").split("|")]
        vals = {}
        for col, val in zip(cols, cs[1:]):
            m = re.match(r"([0-9.]+)\s*/\s*([0-9.]+)", val)
            vals[col] = (float(m.group(1)), float(m.group(2))) if m else None
        t61[cs[0]] = vals
    bind = {}
    for m in re.finditer(r"^(\w+)\s+(e\d+t[0-9.]+)\s+sd ([0-9.]+)\s+in_band ([0-9.]+).*?"
                         r"mol_stab ([0-9.]+)\s+(FAILS floor|passes)", s62, re.M):
        bind[m.group(2)] = {"prop": m.group(1), "sd": float(m.group(3)),
                            "in_band": float(m.group(4)), "mol_stab": float(m.group(5)),
                            "passes": m.group(6) == "passes"}
    m = re.search(r"best raw `in_band` on all three \(([0-9.]+) / ([0-9.]+) / ([0-9.]+)\)"
                  r" and fails\s+the floor on all three \(([0-9.]+) / ([0-9.]+) / ([0-9.]+)\)", s62)
    r63, f63 = {}, {}
    for arm in ("bdg", "btvg_var"):
        mm = re.search(r"\| `%s` \|.*?\*\*([0-9.]+)[–-]([0-9.]+)%%\*\*\s*\|\s*([^|]+?)\s*\|"
                       % arm, s63)
        r63[arm] = (float(mm.group(1)), float(mm.group(2))) if mm else None
        f63[arm] = mm.group(3).strip("*").strip() if mm else None
    if not t61 or len(bind) < 2 or not m or None in r63.values() or None in f63.values():
        raise SystemExit("REFUSING: could not parse the handoff's section 6")
    plug4 = {"in_band": dict(zip(PROPS, map(float, m.groups()[:3]))),
             "mol_stab": dict(zip(PROPS, map(float, m.groups()[3:])))}
    # section 6's intro and section 5: the Betty cells' provenance
    one = lambda s: re.sub(r"\s+", " ", s)
    intro = one(txt[txt.index(NL + "## 6.") + 1:i])
    s5 = one(txt[txt.index(NL + "## 5.") + 1:txt.index(NL + "## 6.")])
    s61o, s62o, s63o = one(s61), one(s62), one(s63)
    miss = []

    def need(pat, where, what):
        mm = re.search(pat, where)
        if not mm:
            miss.append(what)
        return mm

    seeds = need(r"Two seeds: `(\d+)` .*? and `(\d+)`", intro, "section 6 seeds")
    nn = need(r"`n=(\d+)`", intro, "section 6 n")
    dev = need(r"`t_min_guide=[0-9.]+`, (CPU|GPU)", intro, "section 6 device")
    floor = need(r"Floor FR3a = ([0-9]+\.[0-9]+)", intro, "section 6 floor")
    batch = need(r"BDG runs `--n (\d+) --batch (\d+)`", s5, "section 5 batch")
    vbatch = need(r"`btvg_var` runs at batch (\d+)", s5, "section 5 btvg_var batch")
    mono = need(r"Monotone on \*\*(\d+)/(\d+) curves, (both seeds)\*\*", s61o, "6.1 monotone")
    fl61 = need(r"(\w+) and (\w+) clear the floor on both seeds, (\w+) fails it on seed (\d) "
                r"\(but so does its \*control\*", s61o, "6.1 floor sentence")
    yard = need(r"largest reproducible floor-clearing widening anywhere was \*\*([0-9.]+)x\*\*",
                s61o, "6.1 yardstick")
    tightf = need(r"Every `(e\d+t[0-9.]+)` cell fails the floor on (\w+) and (\w+)\.", s62o,
                  "6.2 tight-cell floor sentence")
    every = need(r"Widening lowers `in_band` (every time), as predicted", s62o,
                 "6.2 widening sentence")
    repl = need(r"(Widening replicates on all three properties)", s61o, "6.1 replicates sentence")
    core = need(r"This is the scientific core, and both arms were run on the \*same\* ([^.]+?) "
                r"so the comparison is valid", s63o, "6.3 core sentence")
    if miss:
        raise SystemExit("REFUSING: could not parse the handoff: %s" % "; ".join(miss))
    if int(batch.group(1)) != int(nn.group(1)):
        raise SystemExit("REFUSING: handoff section 5 n %s != section 6 n %s"
                         % (batch.group(1), nn.group(1)))
    return {"text": sec, "cols": cols, "t61": t61, "bind": bind, "plug4": plug4, "r63": r63,
            "f63": f63, "seeds": (int(seeds.group(1)), int(seeds.group(2))),
            "n": int(nn.group(1)), "batch": int(batch.group(2)), "device": dev.group(1),
            "btvg_var_batch": int(vbatch.group(1)),
            "floor": float(floor.group(1)), "floor_s": floor.group(1),
            "mono": (int(mono.group(1)), int(mono.group(2)), mono.group(3)),
            "fl61": dict(clear=(fl61.group(1), fl61.group(2)), fail=fl61.group(3),
                         fail_seed=int(fl61.group(4))),
            "yard": float(yard.group(1)), "yard_s": yard.group(1),
            "tightf": dict(v=tightf.group(1), props=(tightf.group(2), tightf.group(3))),
            "every": every.group(1), "every_s": every.group(0), "repl_s": repl.group(1),
            "core_same": core.group(1)}


def load_weff(cells):
    """{(prop, tgt, variant): {...}} for the trajectories whose run-mean
    reproduces the cell's recorded run-mean w_eff."""
    if not os.path.exists(WEFF_TRAJ):
        return {}, []
    out, rejected = {}, []
    for t in json.load(open(WEFF_TRAJ, encoding="utf-8")):
        v = "e%gt%g" % (float(t["eta"]), float(t["tau_mult"]))
        key = (t["prop"], t["target"], "bdg", 1.0, v)
        w = [s["bdg_w_eff"] for s in t["traj"]]
        e = [s["bdg_e_raw"] for s in t["traj"]]
        if key not in cells or not w:
            rejected.append("%s/%s %s: no matching cell" % (t["prop"], t["target"], v))
            continue
        r = cells[key][0]
        mean = sum(w) / len(w)
        guided = sum((r.get("schedule_used") or {}).values())
        if abs(mean - r["diag"]["bdg_w_eff"]) > 1e-6 or len(w) != guided:
            rejected.append("%s/%s %s: run-mean %.6f vs cell %.6f, %d vs %d steps"
                            % (t["prop"], t["target"], v, mean, r["diag"]["bdg_w_eff"],
                               len(w), guided))
            continue
        # per-step V_b/tau^2 against the law's equilibrium V* = 1 - 1/eta (in tau^2)
        eta = float(t["eta"])
        V = [s["bdg_V_over_tau2"] for s in t["traj"]]
        law = max(abs(wi - (1.0 + eta * (vi - 1.0))) for wi, vi in zip(w, V))
        if law > 1e-5 or any(s["bdg_e"] != s["bdg_e_raw"] for s in t["traj"]):
            rejected.append("%s/%s %s: per-step w_eff is not 1 + eta (V_b/tau^2 - 1) (max "
                            "diff %.1e) or e is clamped" % (t["prop"], t["target"], v, law))
            continue
        vstar = 1.0 - 1.0 / eta
        side = [x > vstar for x in V]
        cross = [i for i in range(1, len(V)) if side[i] != side[i - 1]]
        sign = sum(1 for a, b in zip(w, w[1:]) if (a > 0) != (b > 0))
        if len(cross) != sign:
            raise SystemExit("REFUSING: %s/%s %s: %d crossings of V* but %d w_eff sign changes"
                             % (t["prop"], t["target"], v, len(cross), sign))
        out[(t["prop"], t["target"], v)] = {
            "min": min(w), "max": max(w), "steps": len(w), "sign": sign,
            "e_min": min(e), "match": abs(mean - r["diag"]["bdg_w_eff"]), "law": law,
            "prop": t["prop"], "target": t["target"], "tau_mult": float(t["tau_mult"]),
            "eta": eta, "vstar": vstar, "V0": V[0], "Vend": V[-1], "Vmin": min(V),
            "Vmax": max(V), "Vmin_step": V.index(min(V)) + 1, "Vmean": sum(V) / len(V),
            "cross": len(cross), "first_cross": (cross[0] + 1) if cross else None,
            "last_cross": cross[-1] if cross else None,
            "dclose": min(abs(x - vstar) for x in V), "dend": abs(V[-1] - vstar),
            "w_pos_all": all(x > 0 for x in w), "above0": side[0]}
    return out, rejected


def sensitivity_floors():
    sw = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "sweep", "*__unguided__*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        if r.get("seed") == SWEEP_SEED and r.get("n") == SWEEP_N:
            sw.append(r)
    vals = {r["mol_stability"] for r in sw}
    if len(vals) != 1:
        raise SystemExit("REFUSING: results/sweep unguided stability is not one value: %s" % vals)
    s1 = vals.pop()
    v2 = collections.defaultdict(set)
    for p in sorted(glob.glob(V2_UNGUIDED)):
        r = json.load(open(p, encoding="utf-8"))
        v2[(r["seed"], r["n"])].add(r["mol_stability"])
    if not v2 or any(len(s) != 1 for s in v2.values()):
        raise SystemExit("REFUSING: v2 unguided stability is missing or not one value per seed")
    tot = sum(n for (_, n) in v2)
    pooled = sum(next(iter(s)) * n for (_, n), s in v2.items()) / tot
    devs = sorted({str((r.get("prov") or {}).get("device") or "device not recorded")
                   for r in sw})
    return {
        "sweep": FLOOR_FRAC * s1, "sweep_ug": s1, "sweep_cells": len(sw),
        "sweep_batch": sorted({r.get("batch") for r in sw}), "sweep_dev": devs,
        "v2": FLOOR_FRAC * pooled, "v2_ug": pooled, "v2_seeds": sorted(s for (s, _) in v2),
        "v2_n": tot,
    }


def weights_same(port_fm, port_md5, sweep_fm, sweep_md5):
    """Is the port's weights file the same sampling weights as results/sweep's?

    Read from weights/README.md (the port file's own section: what it was stripped
    from, the verified max parameter difference, the source md5) and checked against
    the md5s the cells record and the md5 of the file on disk. Returns (True, facts)
    or (False, the reason)."""
    import hashlib
    if not os.path.exists(WEIGHTS_README):
        return False, "weights/README.md not found"
    txt = open(WEIGHTS_README, encoding="utf-8").read()
    m = re.search(r"^## %s\s*$(.*?)(?=^## |\Z)" % re.escape(str(port_fm)), txt, re.M | re.S)
    if not m:
        return False, "weights/README.md has no section for %s" % port_fm
    sec = m.group(1)
    src = re.search(r"Stripped from `([^`]+)`", sec)
    dif = re.search(r"Verified: `max \|w_slim - w_full\| = ([0-9.eE+-]+)`", sec)
    md5 = re.search(r"source md5 \(full checkpoint\): `([0-9a-f]{32})`", sec)
    ema = re.search(r"The EMA weights are what sampling\s+uses", sec)
    if not (src and dif and md5 and ema):
        return False, "weights/README.md's %s section did not parse" % port_fm
    why = []
    if list(sweep_fm) != [src.group(1)]:
        why.append("results/sweep used %s, the README's source is %s"
                   % ("/".join(map(str, sweep_fm)), src.group(1)))
    if float(dif.group(1)) != 0.0:
        why.append("the README's max parameter difference is %s" % dif.group(1))
    if list(sweep_md5) != [md5.group(1)]:
        why.append("results/sweep md5 %s is not the README's source md5 %s"
                   % ("/".join(map(str, sweep_md5)), md5.group(1)))
    path = os.path.join(WEIGHTS_DIR, str(port_fm))
    disk = None
    if os.path.exists(path):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        disk = h.hexdigest()
    if disk != port_md5:
        why.append("the port's recorded md5 %s is not weights/%s's (%s)" % (port_md5, port_fm, disk))
    if why:
        return False, "; ".join(why)
    return True, dict(src=src.group(1), diff=dif.group(1), src_md5=md5.group(1),
                      port_md5=port_md5)


def v2_targets():
    out = {}
    for p in sorted(glob.glob(V2_UNGUIDED)):
        r = json.load(open(p, encoding="utf-8"))
        out.setdefault(r["prop"], set()).add(r["target"])
    return out


def sweep_context(delta_port):
    found, skipped, problems = [], collections.Counter(), []
    for d in sorted(glob.glob(os.path.join(ROOT, "results", "sweep*"))):
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*__tmin0.5__*.json"))):
            m = SWEEP_RE.match(os.path.basename(p))
            if not m:
                continue
            r = json.load(open(p, encoding="utf-8"))
            rel = os.path.relpath(d, ROOT).replace(os.sep, "/")
            if (r.get("n") != SWEEP_N or r.get("seed") != SWEEP_SEED
                    or r.get("target_name") != m.group(3)):
                skipped[(rel, r.get("n"), r.get("seed"))] += 1
                continue
            found.append({"dir": rel, "prop": m.group(1), "arm": m.group(2), "tgt": m.group(3),
                          "w": float(m.group(4)), "tag": m.group(5), "r": r})
    # the unguided reference: every copy must agree
    ug = {}
    for c in found:
        if c["arm"] != "unguided":
            continue
        r = c["r"]
        sig = (r["prop_rmse_eval"], r["f_B_mean"], r["target_mean"], r["mol_stability"])
        k = (c["prop"], c["tgt"])
        if k in ug and ug[k]["sig"] != sig:
            problems.append("%s/%s: unguided copies in results/sweep disagree" % k)
        u = ug.setdefault(k, {"sig": sig, "r": r, "tags": set(), "devs": set()})
        u["tags"].add(c["tag"])
        u["devs"].add(str((r.get("prov") or {}).get("device")))
    for p in PROPS:
        for t in TGTS:
            if (p, t) not in ug:
                problems.append("%s/%s: no unguided cell in results/sweep" % (p, t))
    if problems:
        return None, problems

    def sd_bias(r):
        b = r["f_B_mean"] - r["target_mean"]
        return math.sqrt(max(r["prop_rmse_eval"] ** 2 - b * b, 0.0)), b

    for k, u in ug.items():
        u["sd"], u["bias"] = sd_bias(u["r"])
    groups = collections.defaultdict(list)
    for c in found:
        if c["arm"] == "unguided":
            continue
        sd, b = sd_bias(c["r"])
        dl = c["r"]["delta"]
        if abs(dl - delta_port[c["prop"]]) > 1e-9:
            problems.append("%s: results/sweep delta %.8f != port delta %.8f"
                            % (c["prop"], dl, delta_port[c["prop"]]))
        u = ug[(c["prop"], c["tgt"])]
        groups[(c["arm"], c["tag"])].append(dict(c, ratio=sd / u["sd"], bias_d=b / dl,
                                                 stab=c["r"]["mol_stability"],
                                                 dev=(c["r"].get("prov") or {}).get("device")))
    allc = [g for v in groups.values() for g in v]
    meta = {
        "n_found": len(allc), "skipped": skipped, "ug": ug,
        "devices": sorted({g["dev"] for g in allc}, key=str),
        "models": collections.Counter(dev_model(g["dev"]) for g in allc),
        "batch": sorted({g["r"].get("batch") for g in allc}, key=str),
        "fm": sorted({g["r"].get("fm") for g in allc}, key=str),
        "fm_md5": sorted({(g["r"].get("prov") or {}).get("fm_md5") for g in allc}
                         | {(u["r"].get("prov") or {}).get("fm_md5") for u in ug.values()},
                         key=str),
        "n": sorted({g["r"].get("n") for g in allc}, key=str),
        "seed": sorted({g["r"].get("seed") for g in allc}, key=str),
        # unguided copies on more than one MIG slice that agree (sig checked above)
        "ug_multislice": sum(1 for u in ug.values() if len(u["devs"]) > 1
                             and len({dev_model(d) for d in u["devs"]}) == 1),
    }
    return (groups, meta), problems


def dev_model(d):
    """The GPU model, without the MIG slice: 'NVIDIA B200 MIG 1g.45gb' -> 'NVIDIA B200'."""
    return re.sub(r"\s+MIG\s+.*$", "", str(d))


def group_devices(g, ug):
    """(the group's device models, its unguided divisors' device models)."""
    mine = sorted({dev_model(x["dev"]) for x in g})
    div = sorted({dev_model(d) for x in g for d in ug[(x["prop"], x["tgt"])]["devs"]})
    return mine, div


def widening_counterexamples(rows45, ug):
    """Recompute the review's section 4.5 widening cells from results/sweep (seed
    SWEEP_SEED, n SWEEP_N) against results/sweep's unguided at the same target, and
    refuse if the ratio disagrees with the review's seed-1 column to 1e-3."""
    out, problems = [], []
    for row in rows45:
        pat = os.path.join(ROOT, "results", "sweep", "%s__%s__%s__w%g__tmin0.5__*.json"
                           % (row["prop"], row["arm"], row["tgt"], row["w"]))
        rs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(pat))]
        rs = [r for r in rs if r.get("seed") == SWEEP_SEED and r.get("n") == SWEEP_N]
        u = ug.get((row["prop"], row["tgt"]))
        if len(rs) != 1 or u is None:
            problems.append("%s %s %s w %g: %d results/sweep cells, unguided %s"
                            % (row["arm"], row["prop"], row["tgt"], row["w"], len(rs),
                               "found" if u else "missing"))
            continue
        r = rs[0]
        b = r["f_B_mean"] - r["target_mean"]
        sd = math.sqrt(max(r["prop_rmse_eval"] ** 2 - b * b, 0.0))
        ratio = sd / u["sd"]
        if abs(ratio - row["s1"]) > 1e-3:
            problems.append("%s %s %s w %g: recomputed %.4f vs review %.3f"
                            % (row["arm"], row["prop"], row["tgt"], row["w"], ratio, row["s1"]))
        out.append(dict(row, ratio=ratio, stab=r["mol_stability"],
                        shift=(r["f_B_mean"] - u["r"]["f_B_mean"]) / r["delta"]))
    return out, problems


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------

def paired(A, B, delta, what, dec=False):
    """(mean d, paired z, d) over rows finite in both; z is nan when sd(d) = 0."""
    key = "f_B_dec" if dec else "f_B"
    ok = A["finite"].bool() & B["finite"].bool()
    if what == "ib":
        xa = ((A[key] - A["y"]).abs() <= delta).double()
        xb = ((B[key] - B["y"]).abs() <= delta).double()
        diff = (xa - xb)[ok]
    elif what == "mae":
        diff = ((A[key].double() - A["y"].double()).abs()
                - (B[key].double() - B["y"].double()).abs())[ok] / delta
    elif what == "mean":
        diff = (A[key].double() - B[key].double())[ok] / delta
    else:
        diff = (A["mol_stable"].double() - B["mol_stable"].double())[ok]
    n = diff.numel()
    m = float(diff.mean())
    sd = float(diff.std(unbiased=True)) if n > 1 else 0.0
    z = m / (sd / math.sqrt(n)) if sd > 0 else float("nan")
    return m, z, diff


def verdict(z):
    if isnan(z):
        return "n/a"
    return "beats" if z >= SIGMA else ("loses" if z <= -SIGMA else "tie")


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-out", default=DEFAULT_MD)
    ap.add_argument("--json-out", default="",
                    help="optional machine-readable summary (write it outside the repo)")
    ap.add_argument("--print", action="store_true", help="also print the page")
    args = ap.parse_args()

    import torch

    cells, problems = load()
    if problems:
        refuse(problems, "the port's cell set is not complete")
    problems = check_consistency(cells)
    if problems:
        refuse(problems, "the port's cells are not one run")
    problems, same_order = check_pairing(cells)
    if problems:
        refuse(problems, "the cells do not pair")
    M = {k: metrics(r, pm) for k, (r, pm, _) in cells.items()}
    problems, worst = check_json_agreement(cells, M)
    if problems:
        refuse(problems, "the sidecars do not reproduce the cell JSONs")

    RV = read_review()
    HO = read_handoff()
    WT, WT_rejected = load_weff(cells)
    SF = sensitivity_floors()
    V2T = v2_targets()

    C = lambda p, t, a, w, v: cells[(p, t, a, w, v)]
    any_r = C(PROPS[0], TGTS[0], "unguided", 0.0, "bdgctl")[0]
    prov = any_r.get("prov") or {}
    N = any_r["n"]
    DELTA = {p: C(p, TGTS[0], "unguided", 0.0, "bdgctl")[0]["delta"] for p in PROPS}
    TARGET = {(p, t): C(p, t, "unguided", 0.0, "bdgctl")[0]["target"] for p in PROPS for t in TGTS}
    ref_bdg = C(PROPS[0], TGTS[0], "bdg", 1.0, "e4t1")[0]
    GUIDED = sum((ref_bdg.get("schedule_used") or {}).values())
    W_BDG = sorted({float(cells[k][0]["w"]) for k in cells if k[2] == "bdg"})
    ETAS = sorted({float(cells[k][0]["bdg_eta"]) for k in cells if k[2] == "bdg"})
    ETA = max(ETAS)
    VSTAR = 1.0 - 1.0 / ETA
    total_nonfinite = sum(m["nonfinite"] for m in M.values())

    sweep_res, sweep_problems = sweep_context(DELTA)

    # ---- gates ----------------------------------------------------------------
    gates = []
    for t in TGTS:
        for p in PROPS:
            plug = C(p, t, "plug", 1.0, "bdgctl")[1]
            for lab, other in (("bdg e0t1", ("bdg", 1.0, "e0t1")),
                               ("bdg e4t1.5o", ("bdg", 1.0, "e4t1.5o")),
                               ("plug w1 (tgt copy)", ("plug", 1.0, "tgt"))):
                if (p, t) + other not in cells:
                    continue
                o = C(p, t, *other)[1]
                dfb = float((o["f_B"] - plug["f_B"]).abs().max())
                dfbd = float((o["f_B_dec"] - plug["f_B_dec"]).abs().max())
                dfa = float((o["f_A"] - plug["f_A"]).abs().max())
                same = (dfb == 0.0 and dfbd == 0.0 and dfa == 0.0
                        and torch.equal(o["mol_stable"], plug["mol_stable"])
                        and o["smiles"] == plug["smiles"])
                gates.append(dict(p=p, t=t, lab=lab, dfb=dfb, dfbd=dfbd, dfa=dfa, same=same))
    cost_same = [k for k in cells if k[2] == "bdg"
                 and cells[k][0]["cost"] == C(k[0], k[1], "plug", 1.0, "bdgctl")[0]["cost"]]
    n_bdg = sum(1 for k in cells if k[2] == "bdg")

    def gate_count(lab):
        g = [x for x in gates if x["lab"] == lab]
        return sum(1 for x in g if x["same"]), len(g)

    # ---- per-block computations -------------------------------------------------
    BLK = {}
    PT = []            # every paired test
    seen = set()       # (block, comparator, metric, diff bytes) for de-duplication
    for t in TGTS:
        for p in PROPS:
            dl = DELTA[p]
            ug = M[(p, t, "unguided", 0.0, "bdgctl")]
            floor_mol = FLOOR_FRAC * ug["n_stab"]
            fl = floor_mol / N
            se_mol = math.sqrt(N * fl * (1.0 - fl))
            rows = []
            for (a, w, v, lab) in row_specs(p):
                m = M[(p, t, a, w, v)]
                r = C(p, t, a, w, v)[0]
                margin = m["n_stab"] - floor_mol
                eta = mult = one = None
                if a == "bdg":
                    eta, mult, one = parse_variant(v)
                rows.append(dict(
                    key=(p, t, a, w, v), a=a, w=w, v=v, lab=lab, m=m, r=r,
                    eta=eta, mult=mult, one=one,
                    rA=m["sd_fa"] / ug["sd_fa"], rB=m["sd_fb"] / ug["sd_fb"],
                    rBd=m["sd_fbd"] / ug["sd_fbd"],
                    margin=margin, passes=m["n_stab"] >= floor_mol,
                    knife=abs(margin) <= se_mol,
                    s3621=m["mol_stab"] >= SF["sweep"], s3571=m["mol_stab"] >= SF["v2"]))
            pu = ug["mol_stab"]
            BLK[(p, t)] = dict(rows=rows, floor=fl, floor_mol=floor_mol, se_mol=se_mol,
                               floor_se=FLOOR_FRAC * math.sqrt(pu * (1.0 - pu) / N),
                               ug=ug, cmp={})
            for comp, ckey in (("plug w1", (p, t, "plug", 1.0, "bdgctl")),
                               ("unguided", (p, t, "unguided", 0.0, "bdgctl"))):
                B_ = C(*ckey)[1]
                Mb = M[ckey]
                out = []
                for row in rows:
                    if not (row["a"] == "bdg" or (row["a"] == "plug" and row["w"] == 4.0)):
                        continue
                    A_ = C(*row["key"])[1]
                    res = {}
                    for name, what, dec in (("ib", "ib", False), ("ibd", "ib", True),
                                            ("mae", "mae", False), ("maed", "mae", True),
                                            ("stab", "stab", False), ("mean", "mean", False)):
                        dm, z, diff = paired(A_, B_, dl, what, dec)
                        res[name] = (dm, z)
                        if name in ("ib", "ibd") and not isnan(z):
                            raw = diff.numpy().tobytes()
                            sig = (p, t, comp, name, raw)
                            dup = sig in seen
                            seen.add(sig)
                            PT.append(dict(p=p, t=t, comp=comp, metric=name, cell=row["lab"],
                                           v=row["v"], a=row["a"], d=dm, z=z, dup=dup,
                                           raw=raw, passes=row["passes"], knife=row["knife"]))
                    res["dbias"] = abs(row["m"]["bias"]) - abs(Mb["bias"])
                    res["dsd"] = row["m"]["resid"] - Mb["resid"]
                    res["identical"] = all(x[0] == 0.0 and isnan(x[1]) for x in
                                           (res["ib"], res["ibd"], res["mae"], res["maed"],
                                            res["stab"], res["mean"]))
                    out.append((row, res))
                BLK[(p, t)]["cmp"][comp] = out

    # ---- page -----------------------------------------------------------------------
    L = []
    A = L.append
    A("# BDG port: every cell, the full metric block, and every comparator from the same run")
    A("")
    A("Generated by proj1/scripts/bdg_full_metrics.py. Do not hand-edit; re-run the script.")
    A("")
    A("**The port run only: n = %d, one seed, one batch of %d, %s, fixed targets q50 and "
      "q90. Not the v2 q90 headline run.** Per [BDG_REVIEW.md](../methods/BDG_REVIEW.md) "
      "section 1, no number on this page belongs in a q90 headline table. The iid se of an "
      "in-band proportion here is up to %.3f (at p = 0.5)."
      % (N, any_r["batch"], prov.get("device", "?"), math.sqrt(0.25 / N)))
    A("")
    lo05 = [cells[k][0]["diag"]["bdg_V_over_tau2"] for k in cells
            if k[2] == "bdg" and k[4].startswith("e4") and not k[4].endswith("o")
            and parse_variant(k[4])[1] == min(parse_variant(v)[1] for v in ladder("mu"))]
    hi15 = [cells[k][0]["diag"]["bdg_V_over_tau2"] for k in cells
            if k[2] == "bdg" and k[4].startswith("e4") and not k[4].endswith("o")
            and parse_variant(k[4])[1] == max(parse_variant(v)[1] for v in ladder("mu"))]
    mults = [parse_variant(v)[1] for v in ladder("mu")]
    # per-step V_b/tau^2 against V*, from the trajectories that reproduce their cell
    if any(abs(x["vstar"] - VSTAR) > 1e-12 for x in WT.values()):
        raise SystemExit("REFUSING: a trajectory's eta differs from the cells' eta = %g" % ETA)

    def wt_sel(**kw):
        return [x for x in WT.values() if all(x[k] == v for k, v in kw.items())]

    def rng(vals, fmt="%.2f"):
        vals = list(vals)
        lo, hi = min(vals), max(vals)
        return (fmt % lo) if fmt % lo == fmt % hi else "%s-%s" % (fmt % lo, fmt % hi)

    wt_props = sorted({x["prop"] for x in WT.values()}, key=PROPS.index)
    step_bits = []
    for p in wt_props:
        xs_ = wt_sel(prop=p)
        crossing = [x for x in xs_ if x["cross"]]
        never = [x for x in xs_ if not x["cross"]]
        bit = "%s: %d of %d cross V*" % (p, len(crossing), len(xs_))
        if crossing:
            nc = rng((x["cross"] for x in crossing), "%d")
            bit += (" (%s each, at tau_mult %s, the first time at guided step %s)"
                    % ("once" if nc == "1" else nc + " times", "/".join(
                        "%g" % m_ for m_ in sorted({x["tau_mult"] for x in crossing})),
                       rng((x["first_cross"] for x in crossing), "%d")))
        if never:
            grp = collections.OrderedDict()
            for x in sorted(never, key=lambda x: (x["tau_mult"], TGTS.index(x["target"]))):
                grp.setdefault((x["tau_mult"], x["above0"]), []).append(x["target"])
            bit += "; the other %d never do: %s" % (len(never), ", ".join(
                "tau_mult %g at %s stays %s it" % (m_, "both targets" if len(ts_) == len(TGTS)
                                                   else "/".join(ts_),
                                                   "above" if ab_ else "below")
                for (m_, ab_), ts_ in grp.items()))
        step_bits.append(bit)
    tight_m = min(mults)
    away = []
    for p in wt_props:
        xs_ = wt_sel(prop=p, tau_mult=tight_m)
        if not xs_:
            continue
        s_ = ("%s from %s to %s" % (p, rng(x["V0"] for x in xs_), rng(x["Vend"] for x in xs_)))
        dips = [x for x in xs_ if x["Vmin"] < x["V0"] - 1e-12]
        if dips:
            s_ += (" after a dip to %s by step %s"
                   % (rng(x["Vmin"] for x in dips), rng((x["Vmin_step"] for x in dips), "%d")))
        if all(x["w_pos_all"] for x in xs_):
            s_ += ", with w_eff > 0 at every step"
        away.append(s_)
    tight_wt = [x for x in WT.values() if x["tau_mult"] == tight_m]
    ends_away = bool(tight_wt) and all(x["dend"] > x["dclose"] for x in tight_wt)
    A("**What BDG is.** BDG is plug with the centring gain fixed at w and only the "
      "deviation gain w_eff = 1 + eta*e servoed on e = V_b/tau^2 - 1. Here V_b is the batch "
      "variance of the guide's prediction F_i = f_A(m_i) at the endpoint estimate, tau = "
      "tau_mult x s with s = f_A's y_std, and the per-molecule numerator is "
      "w[(y - F_bar) - w_eff (F_i - F_bar)] (review section 4.1). The centring gain on the "
      "batch mean is the nominal w at every eta and tau (w = %s in every bdg cell here); only "
      "the deviation gain moves. At eta = 0, w_eff = 1 and BDG is plug at w (gated below). "
      "The law's equilibrium is where w_eff = 0, i.e. V* = tau^2 (1 - 1/eta) = %.2f tau^2 at "
      "eta = %g: an asymptote of the law, not a state the loop settles at in the %d-step "
      "guided window. The run-mean V_b/tau^2 is %.2f-%.2f at tau_mult %g and %.2f-%.2f at "
      "tau_mult %g. Per step (the %d re-run trajectories below, %s only), w_eff = 0 exactly at "
      "V*, so every w_eff sign change is a crossing of V*: %s. It does not settle there. At "
      "tau_mult %g V_b/tau^2 comes within %s of V* %s, ending %s from it at step %d: %s."
      % ("/".join("%g" % w for w in W_BDG), VSTAR, ETA, GUIDED, min(lo05), max(lo05),
         min(mults), min(hi15), max(hi15), max(mults), len(WT), " and ".join(wt_props),
         "; ".join(step_bits), tight_m, rng((x["dclose"] for x in tight_wt), "%.3f"),
         "and moves away again on every trajectory" if ends_away
         else "but does not end farther from it on every trajectory",
         rng(x["dend"] for x in tight_wt), GUIDED, "; ".join(away)))
    A("")
    A("**Per-step V_b/tau^2 against V* = %.2f** (`audit/bdg_review/weff_traj.json`; each "
      "row's run-mean w_eff reproduces its cell's, and w_eff = 1 + eta (V_b/tau^2 - 1) holds "
      "at every step to %.1e). Steps are numbered 1-%d over the guided window."
      % (VSTAR, max([x["law"] for x in WT.values()] or [0.0]), GUIDED))
    A("")
    A("| prop | target | tau_mult | V_b/tau^2 step 1 -> step %d | min (step) / max | run-mean | "
      "crossings of V* (= w_eff sign changes) | first crossing (step) | w_eff > 0 every step |"
      % GUIDED)
    A("|---|---|---|---|---|---|---|---|---|")
    for x in sorted(WT.values(), key=lambda x: (PROPS.index(x["prop"]), TGTS.index(x["target"]),
                                                x["tau_mult"])):
        A("| %s | %s | %g | %.3f -> %.3f | %.3f (%d) / %.3f | %.3f | %d | %s | %s |"
          % (x["prop"], x["target"], x["tau_mult"], x["V0"], x["Vend"], x["Vmin"],
             x["Vmin_step"], x["Vmax"], x["Vmean"], x["cross"],
             x["first_cross"] if x["first_cross"] else "never",
             "yes" if x["w_pos_all"] else "no"))
    A("")

    # ---- provenance ----------------------------------------------------------------
    A("## Provenance")
    A("")
    A("| field | value (read from the cell JSONs) |")
    A("|---|---|")
    A("| cells | %d, each a .json and a .permol.pt sidecar, in `results/bdg_port/cells/` |"
      % len(cells))
    A("| n, batch | %d, %d (one batch: every molecule of a cell shares one F_bar and V_b) |"
      % (N, any_r["batch"]))
    A("| seed | %d (every cell) |" % any_r["seed"])
    A("| device | %s; torch %s, CUDA %s |" % (prov.get("device"), prov.get("torch"),
                                              prov.get("cuda")))
    A("| weights | `%s`, md5 `%s` |" % (any_r.get("fm"), prov.get("fm_md5")))
    A("| sampler | %d-step %s, guidance window t >= %g (%d guided steps), clip %g |"
      % (any_r["steps"], any_r["solver"], any_r["t_min_guide"], GUIDED, any_r["clip"]))
    A("| delta | k_delta = %g, source: %s |" % (any_r["k_delta"], any_r["delta_source"]))
    for p in PROPS:
        same_v2 = all(abs(TARGET[(p, "q90")] - x) < 1e-9 for x in V2T.get(p, {float("nan")}))
        A("| %s | delta %.5f; target q50 = %g, q90 = %g%s |"
          % (p, DELTA[p], TARGET[(p, "q50")], TARGET[(p, "q90")],
             " (the same q90 value as the v2 run; the run itself is not v2)" if same_v2 else ""))
    A("| pairing | mol_idx, per-molecule y and n_atoms identical across every cell of each "
      "block (verified; refuses otherwise)%s |"
      % ("; the same mol_idx order in all %d cells" % len(cells) if same_order else ""))
    A("| sidecar vs JSON | in-band, stability, validity and uniqueness reproduce exactly "
      "(max abs diff %.1e); MAE to %.1e relative |"
      % (max(worst["in_band"], worst["in_band_dec"], worst["mol_stab"], worst["valid"],
             worst["uniq"]), max(worst["mae"], worst["mae_dec"])))
    A("| non-finite samples | %d over all %d cells |" % (total_nonfinite, len(cells)))
    A("| cost | the cost dict of %d of %d bdg cells is identical to plug w1's (%d generator, "
      "%d guide passes per cell) |"
      % (len(cost_same), n_bdg, M[(PROPS[0], TGTS[0], "plug", 1.0, "bdgctl")]["gen"],
         M[(PROPS[0], TGTS[0], "plug", 1.0, "bdgctl")]["guide"]))
    n_eta = sum(1 for k in cells if k[2] == "bdg" and float(cells[k][0]["bdg_eta"]) == ETA)
    A("| w_eff min/max | %d of the %d eta = %g cells have a per-step trajectory in "
      "`audit/bdg_review/weff_traj.json` (a re-run of the same configuration) whose run-mean "
      "reproduces the cell's (max diff %.1e); only those show a min/max%s |"
      % (len(WT), n_eta, ETA, max([x["match"] for x in WT.values()] or [0.0]),
         ("; rejected: " + "; ".join(WT_rejected)) if WT_rejected else ""))
    A("")

    # ---- how to read ------------------------------------------------------------------
    A("## How to read the blocks")
    A("")
    A("- **Denominators.** In-band (continuous and decoded), mol stability and validity "
      "are over all n = %d rows, counting a non-finite sample as a failure. MAE, bias and "
      "every sd are over finite rows only. This run has %d non-finite samples, so the two "
      "conventions coincide here." % (N, total_nonfinite))
    A("- **Units.** Error columns are in units of delta, the band half-width. bias = "
      "mean(f - y); residual sd and every sd ratio use the population sd (unbiased=False, "
      "the btvg2_table.py convention). sd ratios divide by the same quantity on the same "
      "block's unguided cell. sd(f_A) is what the controller acts on; sd(f_B) is whether "
      "that survives to the held-out evaluator; sd(f_B_dec) is the decoded view.")
    A("- **se.** +- se is the iid binomial se sqrt(p(1-p)/n). Trajectories inside a bdg "
      "cell are coupled through F_bar and V_b, so its rows are not independent. Review "
      "section 4.9 puts the coupling design effect on in-band at %s-%s, so the iid se is "
      "not materially wrong for in-band. On sd the review finds the servo suppresses "
      "replicate spread, so iid statements about sd are conservative." % RV["deff"])
    A("- **Floor.** %g x the SAME block's unguided mol stability, the run's own control "
      "(review section 4.6). Margin = stable molecules - %g x unguided's stable molecules. "
      "**Knife-edge** = |margin| within one binomial se at the floor, sqrt(n f (1 - f)) "
      "molecules. **A row below its own floor carries no in-band verdict.** Sensitivity "
      "columns only, never a verdict: %.4f = %g x %.4f, the unguided stability of the %d "
      "`results/sweep` cells at seed %d (%s, batch %s), and %.4f = %g x %.4f, the pooled v2 "
      "unguided (seeds %s, %d molecules). Both come from other runs with a different batch, "
      "so neither transfers here (review section 4.6)."
      % (FLOOR_FRAC, FLOOR_FRAC, SF["sweep"], FLOOR_FRAC, SF["sweep_ug"], SF["sweep_cells"],
         SWEEP_SEED, ", ".join(map(str, SF["sweep_dev"])),
         "/".join(map(str, SF["sweep_batch"])), SF["v2"], FLOOR_FRAC, SF["v2_ug"],
         ", ".join(map(str, SF["v2_seeds"])), SF["v2_n"]))
    fse = [BLK[k]["floor_se"] for k in BLK]
    ug_st = sorted({BLK[k]["ug"]["mol_stab"] for k in BLK})
    pw1_below = sum(1 for k in BLK for row in BLK[k]["rows"]
                    if row["lab"] == "plug w1" and not row["passes"])
    ug0 = C(PROPS[0], TGTS[0], "unguided", 0.0, "bdgctl")[1]
    ug_same = all(torch.equal(ug0["mol_stable"], C(p, t, "unguided", 0.0, "bdgctl")[1]["mol_stable"])
                  and ug0["smiles"] == C(p, t, "unguided", 0.0, "bdgctl")[1]["smiles"]
                  for p in PROPS for t in TGTS)
    A("- **The own floor is itself an estimate.** Unguided mol stability here is %s%s, so "
      "the floor carries its own se of %.4f (%.1f molecules, against %.1f for a row sitting "
      "at the floor). This run's unguided sits %+.4f above the pooled v2 unguided, and plug "
      "w1 itself is below the own floor in %d of 6 blocks: read verdicts near the floor as "
      "knife-edge."
      % ("/".join("%.4f" % s for s in ug_st),
         " (the six unguided cells have identical stability flags and SMILES: the sampler "
         "does not see the target, so it is one draw)" if ug_same else "",
         max(fse), max(fse) * N, max(BLK[k]["se_mol"] for k in BLK),
         ug_st[0] - SF["v2_ug"], pw1_below))
    A("- **Controller.** V_b/tau^2, e and w_eff are run-means over the %d guided steps "
      "(the cell's `diag`). For the one-sided cell e is recorded after the clamp, so the raw "
      "e is shown beside it. disp_rms and dev_rms are the batch RMS of the dispersion and "
      "deviation terms. Run-means average over steps whose sign flips; min/max come from the "
      "trajectory file where it matches." % GUIDED)
    A("- **Paired z** = mean(d) / (sd(d, unbiased=True) / sqrt(n)) over rows finite in both "
      "cells; n/a when sd(d) = 0 (the two cells agree on every molecule). Bold = |z| >= %g." % SIGMA)
    A("")

    # ---- gates ------------------------------------------------------------------------
    A("## Gates")
    A("")
    A("Each row compares a cell with plug w1 (bdgctl) of the same block, molecule by "
      "molecule. Bit-identical = max |d| = 0 on f_B, f_B_dec and f_A, and the same "
      "stability flags and SMILES.")
    A("")
    A("| target | prop | cell | max abs d f_B | max abs d f_B_dec | max abs d f_A | bit-identical |")
    A("|---|---|---|---|---|---|---|")
    for g in gates:
        A("| %s | %s | %s | %.3g | %.3g | %.3g | %s |"
          % (g["t"], g["p"], g["lab"], g["dfb"], g["dfbd"], g["dfa"],
             "yes" if g["same"] else "**NO**"))
    A("")
    e0 = gate_count("bdg e0t1")
    e1o = gate_count("bdg e4t1.5o")
    tg = gate_count("plug w1 (tgt copy)")
    raw1o = [cells[k][0]["diag"].get("bdg_e_raw", float("nan")) for k in cells
             if k[2] == "bdg" and k[4].endswith("o")]
    A("**eta = 0 is plug w1 on %d of %d blocks; the one-sided cell asked to widen is plug "
      "w1 on %d of %d; the two plug w1 copies agree on %d of %d.** The one-sided cells' "
      "run-mean raw (pre-clamp) e is %+.4f to %+.4f, so the clamp is asked to widen, and it "
      "leaves the field at plug's."
      % (e0[0], e0[1], e1o[0], e1o[1], tg[0], tg[1], min(raw1o), max(raw1o)))
    A("")

    # ---- blocks -----------------------------------------------------------------------
    A("## Full metric blocks")
    A("")
    A("Six blocks, one per (property, target). q50 and q90 are never merged. Cell names: "
      "`e<eta>t<tau_mult>`, `o` = one-sided clamp.")
    A("")
    for t in TGTS:
        for p in PROPS:
            b = BLK[(p, t)]
            dl = DELTA[p]
            A("### %s, target %s (y = %g, delta = %.5f)" % (p, t, TARGET[(p, t)], dl))
            A("")
            A("*Port run, n = %d, seed %d, %s. Not the v2 q90 headline run.* Own floor = "
              "%g x %d stable unguided molecules = %.1f molecules (%.4f); one binomial se at "
              "the floor = %.1f molecules; the floor's own se = %.1f molecules."
              % (N, any_r["seed"], prov.get("device"), FLOOR_FRAC, b["ug"]["n_stab"],
                 b["floor_mol"], b["floor"], b["se_mol"], b["floor_se"] * N))
            A("")
            A("**Property metrics.**")
            A("")
            A("| cell | eta | tau_mult | 1-sided | in-band cont +- se | in-band dec | "
              "MAE/d cont / dec | bias/d cont / dec | resid sd/d | sd(f_A)/ug | "
              "sd(f_B)/ug | sd(f_B_dec)/ug |")
            A("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for row in b["rows"]:
                m = row["m"]
                A("| %s | %s | %s | %s | %.4f +- %.4f | %.4f | %.3f / %.3f | %+.3f / %+.3f | "
                  "%.3f | %.3f | %.3f | %.3f |"
                  % (row["lab"], f(row["eta"], "%g"), f(row["mult"], "%g"),
                     ("yes" if row["one"] else "no") if row["a"] == "bdg" else "n/a",
                     m["in_band"], m["se"], m["in_band_dec"], m["mae"], m["mae_dec"],
                     m["bias"], m["bias_dec"], m["resid"], row["rA"], row["rB"], row["rBd"]))
            A("")
            A("**Chemistry and floor.**")
            A("")
            A("| cell | mol stab | own floor (margin, molecules) | vs %.4f | vs %.4f | "
              "atom stab | validity | uniqueness | non-finite | diversity |"
              % (SF["sweep"], SF["v2"]))
            A("|---|---|---|---|---|---|---|---|---|---|")
            for row in b["rows"]:
                m = row["m"]
                fv = "PASS" if row["passes"] else "FAIL"
                if row["knife"]:
                    fv += ", knife-edge"
                A("| %s | %.4f | %s (%+.1f) | %s | %s | %.4f | %.4f | %.4f | %d | %.4f |"
                  % (row["lab"], m["mol_stab"], fv, row["margin"],
                     "pass" if row["s3621"] else "fail", "pass" if row["s3571"] else "fail",
                     m["atom_stab"], m["valid"], m["uniq"], m["nonfinite"], m["div"]))
            A("")
            A("**Controller and cost.**")
            A("")
            A("| cell | tau | V_b/tau^2 | e (raw) | w_eff run-mean | w_eff min / max "
              "(sign changes) | disp_rms | dev_rms | clipped guided steps | guide-eval gap/d | "
              "gen / guide passes |")
            A("|---|---|---|---|---|---|---|---|---|---|---|")
            for row in b["rows"]:
                m, r = row["m"], row["r"]
                dg = m["diag"]
                if row["a"] == "bdg":
                    e_s = "%+.4f" % dg["bdg_e"]
                    if "bdg_e_raw" in dg and dg["bdg_e_raw"] != dg["bdg_e"]:
                        e_s += " (raw %+.4f)" % dg["bdg_e_raw"]
                    wt = WT.get((row["key"][0], row["key"][1], row["v"]))
                    mm = ("%+.3f / %+.3f (%d)" % (wt["min"], wt["max"], wt["sign"])
                          if wt else "not recorded")
                    ctl = ("%.4g | %.4f | %s | %+.4f | %s | %.4g | %.4g"
                           % (r["bdg_tau"], dg["bdg_V_over_tau2"], e_s, dg["bdg_w_eff"], mm,
                              dg["bdg_disp_rms"], dg["bdg_dev_rms"]))
                else:
                    ctl = "n/a | n/a | n/a | n/a | n/a | n/a | n/a"
                A("| %s | %s | %s | %.3f | %d / %d |"
                  % (row["lab"], ctl,
                     ("%.1f %%" % (100 * m["clip"])) if m["clip"] is not None else "n/a",
                     m["gap"], m["gen"], m["guide"]))
            A("")
            for comp in ("plug w1", "unguided"):
                A("**Paired, minus %s.** Positive = the cell is higher; for MAE negative is "
                  "better. The last two columns are descriptive (not tested)."
                  % ("plug w1 (bdgctl)" if comp == "plug w1" else "unguided w0"))
                A("")
                A("| cell | own floor | d in-band cont | z | d in-band dec | z | d MAE/d cont "
                  "| z | d MAE/d dec | z | d mol stab | z | d mean f_B/d | z | "
                  "d abs(bias)/d | d resid sd/d | in-band verdict cont / dec |")
                A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                for row, res in b["cmp"][comp]:
                    fv = "PASS" if row["passes"] else "FAIL"
                    if row["knife"]:
                        fv += " (knife-edge)"
                    if res["identical"]:
                        vd = "identical (gate)"
                    elif not row["passes"]:
                        vd = "no verdict (below floor)"
                    else:
                        vd = "%s / %s" % (verdict(res["ib"][1]), verdict(res["ibd"][1]))
                    A("| %s | %s | %+.4f | %s | %+.4f | %s | %+.3f | %s | %+.3f | %s | %+.4f "
                      "| %s | %+.3f | %s | %+.3f | %+.3f | %s |"
                      % (row["lab"], fv, res["ib"][0], zf(res["ib"][1]), res["ibd"][0],
                         zf(res["ibd"][1]), res["mae"][0], zf(res["mae"][1]), res["maed"][0],
                         zf(res["maed"][1]), res["stab"][0], zf(res["stab"][1]),
                         res["mean"][0], zf(res["mean"][1]), res["dbias"], res["dsd"], vd))
                A("")

    # ---- multiplicity ------------------------------------------------------------------
    A("## Multiplicity")
    A("")
    # within-metric repeats (identical by construction) are dropped; continuous and decoded
    # are separate hypotheses and stay separate even where their d coincide on this sample
    n_dup = collections.Counter(x["comp"] for x in PT if x["dup"])
    kept = [x for x in PT if not x["dup"]]
    by_cell = collections.defaultdict(dict)
    for x in kept:
        by_cell[(x["p"], x["t"], x["comp"], x["cell"])][x["metric"]] = x
    coinc = [k for k, v in by_cell.items() if "ib" in v and "ibd" in v
             and v["ib"]["raw"] == v["ibd"]["raw"]]
    coinc_c = collections.Counter(k[2] for k in coinc)
    coinc_bdg_pw = sum(1 for k in coinc if k[2] == "plug w1" and by_cell[k]["ib"]["a"] == "bdg")

    def ncomp(c_):
        return "%d against %s%s" % (coinc_c[c_], c_, (" (%s)" % ", ".join(
            "%s %s %s" % (k[0], k[1], k[3]) for k in sorted(
                [k for k in coinc if k[2] == c_], key=lambda k: (TGTS.index(k[1]),
                                                                  PROPS.index(k[0]), k[3]))))
                                   if coinc_c[c_] else "")

    A("Paired in-band tests (continuous and decoded) from the tables above. A test whose "
      "d is identical, for the same metric, to one already counted in the same block against "
      "the same comparator is counted once: the one-sided cell against unguided repeats "
      "e0t1's, by the gate (%d tests dropped). Tests with sd(d) = 0 have no z and are not "
      "counted. Continuous and decoded in-band are separate hypotheses, so they are counted "
      "separately even where their d happen to coincide on this sample: %s, and %s."
      % (sum(n_dup.values()), ncomp("plug w1"), ncomp("unguided")))
    A("")
    A("| family | tests | abs z >= %g | max abs z (where) | single-test p | Sidak p over the family |"
      % SIGMA)
    A("|---|---|---|---|---|---|")
    MULT = {}
    for fam, sel in (("minus plug w1", lambda x: x["comp"] == "plug w1"),
                     ("minus unguided", lambda x: x["comp"] == "unguided"),
                     ("all", lambda x: True)):
        T = [x for x in PT if sel(x) and not x["dup"]]
        big = [x for x in T if abs(x["z"]) >= SIGMA]
        top = max(T, key=lambda x: abs(x["z"]))
        MULT[fam] = dict(m=len(T), big=len(big), top=top, p1=math.erfc(abs(top["z"]) / math.sqrt(2)),
                         ps=sidak(top["z"], len(T)))
        A("| %s | %d | %d | %+.2f (%s %s, %s, %s) | %.2g | %.2g |"
          % (fam, len(T), len(big), top["z"], top["p"], top["t"], top["cell"],
             "cont" if top["metric"] == "ib" else "dec", MULT[fam]["p1"], MULT[fam]["ps"]))
    A("")
    fam_pw = [x for x in PT if x["comp"] == "plug w1" and not x["dup"]]
    fam_pw_bdg = [x for x in fam_pw if x["a"] == "bdg"]
    top_bdg = max(fam_pw_bdg, key=lambda x: abs(x["z"]))
    A("Against plug w1 the bdg cells alone give %d tests; the largest is %s %s %s (%s), z = "
      "%+.2f, Sidak p = %.2g over those %d. **The grid is correlated**: the tests share "
      "molecules, noise and comparators, and the tau ladder is a smooth family, so Sidak's "
      "independence is only an approximation and the tests are not %d independent looks."
      % (len(fam_pw_bdg), top_bdg["p"], top_bdg["t"], top_bdg["cell"],
         "cont" if top_bdg["metric"] == "ib" else "dec", top_bdg["z"],
         sidak(top_bdg["z"], len(fam_pw_bdg)), len(fam_pw_bdg), len(fam_pw_bdg)))
    A("")
    alt = {"minus plug w1": MULT["minus plug w1"]["m"] - coinc_c["plug w1"],
           "minus unguided": MULT["minus unguided"]["m"] - coinc_c["unguided"],
           "all": MULT["all"]["m"] - len(coinc)}
    alt_bdg = len(fam_pw_bdg) - coinc_bdg_pw
    A("If each coincident continuous/decoded pair were counted once instead, the families "
      "would be %d (minus plug w1), %d (minus unguided) and %d (all), and %d for the bdg "
      "cells against plug w1, with Sidak p %.2g, %.2g, %.2g and %.2g. The max-z rows do not "
      "change."
      % (alt["minus plug w1"], alt["minus unguided"], alt["all"], alt_bdg,
         sidak(MULT["minus plug w1"]["top"]["z"], alt["minus plug w1"]),
         sidak(MULT["minus unguided"]["top"]["z"], alt["minus unguided"]),
         sidak(MULT["all"]["top"]["z"], alt["all"]), sidak(top_bdg["z"], alt_bdg)))
    A("")

    # ---- spread versus bias lever --------------------------------------------------------
    A("## Spread versus bias lever")
    A("")
    A("**An idealised Gaussian bound at fixed centring, not a prediction** (review section "
      "4.4). For each block, take the unguided cell's measured bias b and residual sd s (in "
      "delta) and treat the residual as N(b, s^2). The columns give that model's in-band "
      "now; with the bias removed (b -> 0, s unchanged); with s x %s (BDG's tight end, "
      "review section 4.4) and with s x the port's own tightest sd(f_B)/ug on the eta = %g "
      "ladder; and the sd scale k* that maximises in-band at fixed b. For |b| <= 1 the "
      "model's in-band rises monotonically as the spread shrinks, so there is no finite "
      "optimum; for |b| > 1, k* s = sqrt(2|b| / ln((|b|+1)/(|b|-1))). It does not model what "
      "any arm would do to the centring."
      % (RV["tight_s"], ETA))
    A("")
    A("| prop | target | view | measured in-band | b | s | Gaussian now | b -> 0 | "
      "s x %s | s x port tightest (k) | k* | in-band at k* | spread-lever gain | "
      "bias-lever gain |" % RV["tight_s"])
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    LEVER = {}
    for t in TGTS:
        for p in PROPS:
            ug = BLK[(p, t)]["ug"]
            tight = min(row["rB"] for row in BLK[(p, t)]["rows"]
                        if row["a"] == "bdg" and row["v"] in ladder(p))
            for view, b, s, meas in (("cont", ug["bias"], ug["resid"], ug["in_band"]),
                                     ("dec", ug["bias_dec"], ug["resid_dec"], ug["in_band_dec"])):
                now = gauss_inband(b, s)
                deb = gauss_inband(0.0, s)
                t65 = gauss_inband(b, RV["tight"] * s)
                tpt = gauss_inband(b, tight * s)
                k = best_scale(b, s)
                if k is None:
                    kstar, pk, gain_s = "none (tighter is better)", "-> 1", float("nan")
                else:
                    pk_v = gauss_inband(b, k * s)
                    kstar, pk, gain_s = "%.2f" % k, "%.4f" % pk_v, pk_v - now
                gain_b = deb - now
                LEVER[(p, t, view)] = dict(b=b, s=s, now=now, deb=deb, t65=t65, k=k,
                                           gain_s=gain_s, gain_b=gain_b)
                A("| %s | %s | %s | %.4f | %+.2f | %.2f | %.4f | %.4f | %.4f | %.4f (%.3f) | %s "
                  "| %s | %s | %+.4f |"
                  % (p, t, view, meas, b, s, now, deb, t65, tpt, tight, kstar, pk,
                     f(gain_s, "%+.4f"), gain_b))
    A("")
    q90g = [LEVER[(p, "q90", "cont")] for p in PROPS]
    q50g = [LEVER[(p, "q50", "cont")] for p in PROPS]
    A("On the continuous view, removing the bias is worth %+.4f to %+.4f at q90 against a "
      "whole spread lever of %s; at q50 the bias lever is %+.4f to %+.4f and tightening to "
      "s x %s changes in-band by %+.4f to %+.4f. At q90, tightening to s x %s changes it by "
      "%+.4f to %+.4f."
      % (min(x["gain_b"] for x in q90g), max(x["gain_b"] for x in q90g),
         ", ".join(f(x["gain_s"], "%+.4f") for x in q90g),
         min(x["gain_b"] for x in q50g), max(x["gain_b"] for x in q50g), RV["tight_s"],
         min(x["t65"] - x["now"] for x in q50g), max(x["t65"] - x["now"] for x in q50g),
         RV["tight_s"], min(x["t65"] - x["now"] for x in q90g),
         max(x["t65"] - x["now"] for x in q90g)))
    A("")

    # ---- context from results/sweep ---------------------------------------------------------
    A("## Context, a different run: spread and bias ranges over each arm's sweep")
    A("")
    port_dev = dev_model(prov.get("device"))

    def differs(label, port_v, sweep_vs):
        sv = "/".join(map(str, sweep_vs))
        return None if [port_v] == list(sweep_vs) else "%s (%s vs the port's %s)" % (label, sv, port_v)

    CTX = {}
    W45 = []
    if sweep_res is not None:
        W45, w45_problems = widening_counterexamples(RV["w45"], sweep_res[1]["ug"])
        if w45_problems:
            refuse(w45_problems, "the review's section 4.5 widening cells do not reproduce "
                                 "from results/sweep")
        for x in W45:
            if x["floor"].startswith("clears") != (x["stab"] >= SF["sweep"]):
                refuse(["%s %s %s w %g: stab %.4f vs floor %.4f, review says '%s'"
                        % (x["arm"], x["prop"], x["tgt"], x["w"], x["stab"], SF["sweep"],
                           x["floor"])], "the review's section 4.5 floor column does not "
                                         "reproduce")
    if sweep_res is None:
        A("**Aggregate only, never paired.** Review section 5 item 8 asks for the spread range "
          "of plug's w-sweep, btvg_var's and btvg's beside BDG's tau ladder, with a bias column.")
        A("")
        A("**Not computed:** " + "; ".join(sweep_problems))
        A("")
    else:
        groups, meta = sweep_res
        w_diff = differs("weights file", any_r.get("fm"), meta["fm"])
        if w_diff:
            w_same, w_fact = weights_same(any_r.get("fm"), prov.get("fm_md5"), meta["fm"],
                                          meta["fm_md5"])
            if w_same:
                w_diff = w_diff[:-1] + (
                    "; a file-name difference only: per weights/README.md, %s is %s with the "
                    "optimiser and scheduler state stripped, and the EMA weights that sampling "
                    "uses are byte-identical, max abs parameter difference %s; the results/sweep "
                    "cells record md5 `%s`, the README's source md5, and the port records "
                    "`%s`, the md5 of weights/%s; review section 9)"
                    % (any_r.get("fm"), w_fact["src"], w_fact["diff"], w_fact["src_md5"][:8],
                       w_fact["port_md5"][:8], any_r.get("fm")))
            else:
                w_diff = w_diff[:-1] + "; not verified to be the same weights: %s)" % w_fact
        diffs = [d_ for d_ in (differs("n", N, meta["n"]), differs("seed", any_r["seed"], meta["seed"]),
                               differs("batch", any_r["batch"], meta["batch"]), w_diff) if d_]
        order = [(a, tag) for a in SWEEP_ARMS for tag in
                 sorted({k[1] for k in groups if k[0] == a},
                        key=lambda s: (s != "cmp", s))]
        GDEV = {k: group_devices(groups[k], meta["ug"]) for k in order}
        off = [k for k in order if GDEV[k][0] != GDEV[k][1]]
        div_models = sorted({dev_model(d) for u in meta["ug"].values() for d in u["devs"]})
        A("**Aggregate only, never paired.** Review section 5 item 8 asks for the spread range "
          "of plug's w-sweep, btvg_var's and btvg's beside BDG's tau ladder, with a bias column. "
          "The three sweeps come from `results/sweep*` cells with n = %s and seed %s. Against "
          "the port they differ in %s, so only aggregate ranges are compared. Device, read "
          "from each cell: %s; the port ran on the %s. sd = sqrt(rmse^2 - bias^2) with rmse = "
          "prop_rmse_eval and bias = f_B_mean - target_mean, both over finite rows; the ratio "
          "divides by the results/sweep unguided cell at the same target, which ran on the %s "
          "for every (property, target). %s The port rows use sd(f_B) over finite rows "
          "(population sd), the same quantity. Range = max - min of sd(f_B)/ug over the sweep. "
          "See also [VARIANCE_DECOMPOSITION.md](VARIANCE_DECOMPOSITION.md), including its caveat "
          "that the across-molecule variance concentrates then re-widens on unguided runs."
          % ("/".join(map(str, meta["n"])), "/".join(map(str, meta["seed"])),
             ", ".join(diffs) if diffs else "nothing recorded",
             ", ".join("%d of %d cells on the %s" % (c_, meta["n_found"], m_)
                       for m_, c_ in sorted(meta["models"].items(), key=lambda kv: -kv[1])),
             port_dev, " / ".join(div_models),
             ("**%s ran on a different device from their unguided divisor** (%s), so their "
              "ratio is not a same-device ratio; they are marked in the tables and used "
              "nowhere else on this page." % (
                  " and ".join("The %s (%s) cells" % k_ for k_ in off),
                  "; ".join("%s (%s): %s against %s" % (k_[0], k_[1], "/".join(GDEV[k_][0]),
                                                      "/".join(GDEV[k_][1])) for k_ in off)))
             if off else "Every group ran on the same device model as its divisor."))
        A("")
        if sweep_problems:
            A("**Warnings:** " + "; ".join(sweep_problems))
            A("")
        A("Cells used: %d (devices %s; batch %s). Unguided copies on two MIG slices of the same "
          "model agree exactly on rmse, mean and stability in %d of %d (property, target) pairs "
          "(checked; the script refuses on disagreement). Skipped by the n/seed filter: %s."
          % (meta["n_found"], ", ".join(map(str, meta["devices"])),
             "/".join(map(str, meta["batch"])), meta["ug_multislice"], len(meta["ug"]),
             "; ".join("%s n=%s seed=%s: %d" % (k[0], k[1], k[2], v)
                       for k, v in sorted(meta["skipped"].items(), key=str)) or "none"))
        A("")
        for t in TGTS:
            A("**Target %s.**" % t)
            A("")
            A("| prop | arm (cell tag) | run | cells | w grid | sd(f_B)/ug min | max | range | "
              "bias/d min | max |")
            A("|---|---|---|---|---|---|---|---|---|---|")
            for p in PROPS:
                ugc = meta["ug"][(p, t)]
                ug_models = sorted({dev_model(d) for d in ugc["devs"]})
                A("| %s | unguided (reference) | results/sweep, %s | 1 | - | 1.000 | 1.000 | "
                  "0.000 | %+.3f | %+.3f |" % (p, "/".join(ug_models), ugc["bias"] / DELTA[p],
                                               ugc["bias"] / DELTA[p]))
                for (a, tag) in order:
                    g = sorted([x for x in groups[(a, tag)] if x["prop"] == p and x["tgt"] == t],
                               key=lambda x: x["w"])
                    if not g:
                        continue
                    rs = [x["ratio"] for x in g]
                    bs = [x["bias_d"] for x in g]
                    mine = sorted({dev_model(x["dev"]) for x in g})
                    same_dev = mine == ug_models
                    CTX[(p, t, a, tag)] = dict(lo=min(rs), hi=max(rs), rng=max(rs) - min(rs),
                                               blo=min(bs), bhi=max(bs), n=len(g),
                                               same_dev=same_dev,
                                               w_hi=max(g, key=lambda x: x["ratio"])["w"],
                                               w_lo=min(g, key=lambda x: x["ratio"])["w"])
                    run = ("results/sweep, %s" % "/".join(mine) if same_dev else
                           "results/sweep, **%s; divisor on %s**" % ("/".join(mine),
                                                                     "/".join(ug_models)))
                    A("| %s | %s (%s) | %s | %d | %s | %.3f | %.3f | %.3f | %+.3f "
                      "| %+.3f |"
                      % (p, a, tag, run, len(g), ", ".join("%g" % x["w"] for x in g), min(rs),
                         max(rs), max(rs) - min(rs), min(bs), max(bs)))
                lad = [row for row in BLK[(p, t)]["rows"]
                       if row["a"] == "bdg" and row["v"] in ladder(p)]
                rs = [row["rB"] for row in lad]
                bs = [row["m"]["bias"] for row in lad]
                CTX[(p, t, "bdg", "port")] = dict(lo=min(rs), hi=max(rs), rng=max(rs) - min(rs),
                                                  blo=min(bs), bhi=max(bs), n=len(lad))
                A("| %s | **bdg tau ladder (eta = %g)** | port, %s | %d | tau_mult %s | %.3f | "
                  "%.3f | %.3f | %+.3f | %+.3f |"
                  % (p, ETA, port_dev, len(lad), ", ".join("%g" % row["mult"] for row in lad),
                     min(rs), max(rs), max(rs) - min(rs), min(bs), max(bs)))
                pl = [row for row in BLK[(p, t)]["rows"]
                      if row["a"] == "plug" and row["lab"] != "plug w1 (tgt copy)"]
                rs = [row["rB"] for row in pl]
                bs = [row["m"]["bias"] for row in pl]
                A("| %s | plug (port control) | port, %s | %d | %s | %.3f | %.3f | %.3f | %+.3f | "
                  "%+.3f |"
                  % (p, port_dev, len(pl), ", ".join("%g" % row["w"] for row in pl), min(rs),
                     max(rs), max(rs) - min(rs), min(bs), max(bs)))
            A("")
        # missing
        A("**Missing from each sweep** (the grid is the union of w values that arm and tag "
          "has at either target for any property):")
        A("")
        for (a, tag) in order:
            g = groups[(a, tag)]
            W = sorted({x["w"] for x in g})
            msgs = []
            for t in TGTS:
                at_t = [x for x in g if x["tgt"] == t]
                if not at_t:
                    msgs.append("%s: not run" % t)
                    continue
                miss = []
                for p in PROPS:
                    have = {x["w"] for x in at_t if x["prop"] == p}
                    lack = [w for w in W if w not in have]
                    if lack:
                        miss.append("%s w %s" % (p, ", ".join("%g" % w for w in lack)))
                msgs.append("%s: %s" % (t, "; ".join(miss) if miss else "complete"))
            A("- %s (%s), grid w = %s: %s" % (a, tag, ", ".join("%g" % w for w in W),
                                              " | ".join(msgs)))
        A("")
        cmp_rows = {a: [CTX[(p, t, a, "cmp")]["rng"] for p in PROPS for t in TGTS
                        if (p, t, a, "cmp") in CTX] for a in SWEEP_ARMS}
        bdg_rng = [CTX[(p, t, "bdg", "port")]["rng"] for p in PROPS for t in TGTS]
        pc = [(p, t, CTX[(p, t, "plug", "cmp")]) for p in PROPS for t in TGTS]
        bc = [(p, t, CTX[(p, t, "bdg", "port")]) for p in PROPS for t in TGTS]
        if not all(c_["same_dev"] for _, _, c_ in pc):
            raise SystemExit("REFUSING: a plug cmp group ran on a different device from its "
                             "unguided divisor; the direction statement assumes it did not")
        se_sd = RV["se_sd"]
        plug_over = [(c_["hi"] - 1.0) / se_sd[p] for p, _, c_ in pc]
        plug_in_noise = all(x <= 1.0 for x in plug_over)
        A("Over the six (property, target) pairs, the sd(f_B)/ug range (max - min) is "
          "%.3f-%.3f for plug's `cmp` w-sweep, %.3f-%.3f for btvg_var's, %.3f-%.3f for btvg's, "
          "and %.3f-%.3f for BDG's tau ladder in the port. Plug with no variance term moves "
          "the spread over a range of the same order (review section 4.7), **but in one "
          "direction**: over its `cmp` w-sweep sd(f_B)/ug spans %.3f-%.3f; on each pair its "
          "maximum is %.3f-%.3f (at w = %s, the weak end) and its minimum %.3f-%.3f (at w = "
          "%s). %s BDG's ladder spans %.3f-%.3f, with a per-pair maximum of %.3f-%.3f, above "
          "1 on %d of 6 (unpaired aggregates, no se attached). A range of the same size does "
          "not mean plug covers the widening side: whether a signed-w plug does is the "
          "untested control (review section 5 item 12). No uniqueness claim is made (review "
          "section 4.5; see the appendix row on the 6.1 yardstick)."
          % (min(cmp_rows["plug"]), max(cmp_rows["plug"]), min(cmp_rows["btvg_var"]),
             max(cmp_rows["btvg_var"]), min(cmp_rows["btvg"]), max(cmp_rows["btvg"]),
             min(bdg_rng), max(bdg_rng),
             min(c_["lo"] for *_, c_ in pc), max(c_["hi"] for *_, c_ in pc),
             min(c_["hi"] for *_, c_ in pc), max(c_["hi"] for *_, c_ in pc),
             "/".join("%g" % w for w in sorted({c_["w_hi"] for *_, c_ in pc})),
             min(c_["lo"] for *_, c_ in pc), max(c_["lo"] for *_, c_ in pc),
             "/".join("%g" % w for w in sorted({c_["w_lo"] for *_, c_ in pc})),
             ("So plug's range is contraction from about unguided's spread: its maximum "
              "exceeds 1 on %d of 6 pairs, by at most %.2f of the review's se of an sd ratio "
              "at n = %d (%s; section 4.5)."
              % (sum(1 for *_, c_ in pc if c_["hi"] > 1.0), max(plug_over), SWEEP_N,
                 " / ".join("%s %.3f" % (p, se_sd[p]) for p in PROPS)))
             if plug_in_noise else
             ("Plug's maximum exceeds 1 by more than one of the review's sd-ratio se on %d "
              "of 6 pairs." % sum(1 for x in plug_over if x > 1.0)),
             min(c_["lo"] for *_, c_ in bc), max(c_["hi"] for *_, c_ in bc),
             min(c_["hi"] for *_, c_ in bc), max(c_["hi"] for *_, c_ in bc),
             sum(1 for *_, c_ in bc if c_["hi"] > 1.0)))
        A("")

    # ---- what this supports ------------------------------------------------------------------
    A("## What this supports, and what it does not")
    A("")
    lad_rows = [(t, row) for t in TGTS for p in PROPS for row in BLK[(p, t)]["rows"]
                if row["a"] == "bdg" and row["v"] in ladder(p)]
    res_pw = {(row["key"]): res for t in TGTS for p in PROPS
              for row, res in BLK[(p, t)]["cmp"]["plug w1"]}
    res_ug = {(row["key"]): res for t in TGTS for p in PROPS
              for row, res in BLK[(p, t)]["cmp"]["unguided"]}
    S = {}
    for t in TGTS:
        lr = [row for tt, row in lad_rows if tt == t]
        clear = [row for row in lr if row["passes"]]
        below = [row for row in lr if not row["passes"]]
        beat_c = [row for row in clear if res_pw[row["key"]]["ib"][1] >= SIGMA]
        beat_d = [row for row in clear if res_pw[row["key"]]["ibd"][1] >= SIGMA]
        lose_c = [row for row in clear if res_pw[row["key"]]["ib"][1] <= -SIGMA]
        lose_d = [row for row in clear if res_pw[row["key"]]["ibd"][1] <= -SIGMA]
        pos_c = [row for row in clear if res_pw[row["key"]]["ib"][0] > 0]
        best = max(clear, key=lambda row: (row["m"]["in_band"], row["m"]["in_band_dec"])) if clear else None
        S[t] = dict(n=len(lr), clear=clear, below=below, beat_c=beat_c, beat_d=beat_d,
                    lose_c=lose_c, lose_d=lose_d, pos_c=pos_c,
                    knife_below=[row for row in below if row["knife"]],
                    knife_clear=[row for row in clear if row["knife"]])
    mono_A = mono_B = 0
    for t in TGTS:
        for p in PROPS:
            lad = [row for row in BLK[(p, t)]["rows"] if row["a"] == "bdg" and row["v"] in ladder(p)]
            lad.sort(key=lambda r: r["mult"])
            mono_A += all(b_["rA"] >= a_["rA"] for a_, b_ in zip(lad, lad[1:]))
            mono_B += all(b_["rB"] >= a_["rB"] for a_, b_ in zip(lad, lad[1:]))
    allB = [row["rB"] for _, row in lad_rows]
    allA = [row["rA"] for _, row in lad_rows]
    A("Counts are over the eta = %g two-sided tau ladder (%d cells per block, %d in all); "
      "e0t1 and the one-sided cell are plug w1 by the gates and carry no test."
      % (ETA, len(ladder("mu")), len(lad_rows)))
    A("")
    A("**Supported (review sections 3 and 6).**")
    A("")
    A("- **The gates.** eta = 0 is bit-identical to plug w1 on %d of %d blocks and the "
      "one-sided cell asked to widen on %d of %d; the cost dict equals plug's in %d of %d bdg "
      "cells." % (e0[0], e0[1], e1o[0], e1o[1], len(cost_same), n_bdg))
    A("- **The knob moves the spread monotonically.** requested -> achieved is monotone on "
      "%d of 6 curves for sd(f_A) and %d of 6 for sd(f_B); sd(f_A)/ug spans %.3f-%.3f and "
      "sd(f_B)/ug %.3f-%.3f over the ladder." % (mono_A, mono_B, min(allA), max(allA),
                                                min(allB), max(allB)))
    wmin = min([x["min"] for x in WT.values()] or [float("nan")])
    emin = min([x["e_min"] for x in WT.values()] or [float("nan")])
    A("- **Bounded on the widening side.** Over the %d recorded trajectories the smallest "
      "raw e is %+.4f (the bound is e >= -1) and the smallest w_eff is %+.3f (the bound is "
      "1 - eta = %+g). The largest is %+.3f: w_eff is unbounded above."
      % (len(WT), emin, wmin, 1 - ETA, max([x["max"] for x in WT.values()] or [float("nan")])))
    def shift(akey, bkey, field):
        """Paired mean shift of `field` (A - B) in delta, and its z."""
        A_, B_ = cells[akey][1], cells[bkey][1]
        ok = A_["finite"].bool() & B_["finite"].bool()
        d_ = (A_[field].double() - B_[field].double())[ok] / DELTA[akey[0]]
        return float(d_.mean()), float(d_.mean()) / (float(d_.std(unbiased=True))
                                                    / math.sqrt(d_.numel()))

    def fmt_sh(xs_):
        return ", ".join("%s %+.2f d (z %+.1f)" % (p, m_, z_) for p, (m_, z_) in xs_)

    wide_m = parse_variant(ladder("mu")[-1])[1]
    kw = lambda p, t: (p, t, "bdg", 1.0, ladder(p)[-1])
    ku = lambda p, t: (p, t, "unguided", 0.0, "bdgctl")
    kp = lambda p, t: (p, t, "plug", 1.0, "bdgctl")
    rowof = lambda k: [r_ for r_ in BLK[(k[0], k[1])]["rows"] if r_["key"] == k][0]
    # per target: the paired mean shifts, and the sign test for "toward the target" (a
    # shift opposite in sign to the reference cell's bias moves the mean toward y)
    WU = {}
    for t in TGTS:
        wu_B = [(p, shift(kw(p, t), ku(p, t), "f_B")) for p in PROPS]
        for (p, a_) in wu_B:
            assert abs(a_[0] - res_ug[kw(p, t)]["mean"][0]) < 1e-12, (p, t, a_)
        pu_B = [(p, shift(kp(p, t), ku(p, t), "f_B")) for p in PROPS]
        ugb = {p: BLK[(p, t)]["ug"]["bias"] for p in PROPS}
        WU[t] = dict(
            B=wu_B, A=[(p, shift(kw(p, t), ku(p, t), "f_A")) for p in PROPS], pu_B=pu_B,
            ugb=ugb, ug_neg=sum(1 for p in PROPS if ugb[p] < 0),
            toward=[p for p, (m_, _) in wu_B if (m_ > 0) != (ugb[p] > 0)],
            p_toward=[p for p, (m_, _) in pu_B if (m_ > 0) != (ugb[p] > 0)],
            fell=[p for p in PROPS if res_ug[kw(p, t)]["dbias"] < 0],
            rows=[rowof(kw(p, t)) for p in PROPS])
    t42 = RV["w42_tgt"]
    if t42 not in TGTS:
        raise SystemExit("REFUSING: review section 4.2 states its widening shift at %s, not a "
                         "port target" % t42)
    others42 = [t for t in TGTS if t != t42]

    def plist(ps):
        ps = list(ps)
        return ("all three" if len(ps) == len(PROPS) else
                "none" if not ps else " and ".join(ps))

    def dir_s(t, key):
        tw_ = WU[t][key]
        aw_ = [p for p in PROPS if p not in tw_]
        return ("toward the target on %s" % plist(tw_)) + (
            " and away on %s" % plist(aw_) if tw_ and aw_ else "")

    def floor_s(t):
        rs_ = WU[t]["rows"]
        below_ = [r_ for r_ in rs_ if not r_["passes"]]
        parts = ", ".join("%s %s%s" % (r_["key"][0], "PASS" if r_["passes"] else "FAIL",
                                        " (knife-edge)" if r_["knife"] else "") for r_ in rs_)
        if len(below_) == len(rs_):
            tail = "all three sit below it"
        elif below_:
            tail = "%d of 3 %s below it" % (len(below_), "sits" if len(below_) == 1 else "sit")
        else:
            tail = "all three clear it"
        return ("Against their own floor the %s tau_mult %g cells are %s: %s"
                % (t, wide_m, parts, tail))

    def bias_path(t):
        return ", ".join("%s %+.2f -> %+.2f" % (p, WU[t]["ugb"][p], M[kw(p, t)]["bias"])
                         for p in PROPS)

    W = WU[t42]
    pu_B = W["pu_B"]
    wp_B = [(p, shift(kw(p, t42), kp(p, t42), "f_B")) for p in PROPS]
    wp_A = [(p, shift(kw(p, t42), kp(p, t42), "f_A")) for p in PROPS]
    plug_share = [pb_[0] / wb_[0] for (_, pb_), (_, wb_) in zip(pu_B, W["B"])]
    share_word = ("Most of that is plug's own centring" if all(s_ >= 0.5 for s_ in plug_share)
                  else "Part of that is plug's own centring")
    pw_neg = sum(1 for p in PROPS if M[kp(p, t42)]["bias"] < 0)
    away_p = ["%s at z %+.1f" % (p, z_) for p, (m_, z_) in wp_B
              if (m_ > 0) == (M[kp(p, t42)]["bias"] > 0)]
    drag_B = [shift(row["key"], kp(row["key"][0], row["key"][1]), "f_B")[0] for _, row in lad_rows]
    drag_A = [shift(row["key"], kp(row["key"][0], row["key"][1]), "f_A")[0] for _, row in lad_rows]
    all_tw = len(W["toward"]) == len(PROPS)
    mech = ("so the shift is a statement about the mechanism, not an in-band verdict"
            if any(not r_["passes"] for r_ in W["rows"]) else
            "the shift is a statement about the mechanism; in-band is tested separately below")
    A("- **At %s, the widening cells keep plug's centring gain; against unguided they move the "
      "batch mean toward the target%s, against plug w1 the deviation mode drags it** (review "
      "section 4.2, which states this at %s only). At %s, tau_mult %g moves mean f_B by %s "
      "relative to unguided (paired; on the guide's own f_A %s); unguided's bias is negative "
      "in %d of 3, and by the sign test (a shift opposite in sign to unguided's bias) the "
      "shift is %s; bias/d goes %s. %s (plug w1's shift is %s of the widening "
      "cell's): plug w1 moves mean f_B by %s against unguided. Against plug w1, which has the "
      "same centring gain, tau_mult %g moves mean f_B by %s (f_A %s); plug w1's bias is "
      "negative in %d of 3, and a shift of the same sign as its bias moves away from the "
      "target: %s. Over all %d ladder cells (both targets) the paired shift of the mean "
      "against plug w1 spans %+.2f to %+.2f d on f_B and %+.2f to %+.2f d on f_A; the review "
      "puts the covariance drag at %s-%s d over the ladder. The centring gain is fixed, but "
      "its effect on the mean is not independent of the deviation gain. %s, %s. %s"
      % (t42, "" if all_tw else " on %d of 3 properties" % len(W["toward"]), t42, t42, wide_m,
         fmt_sh(W["B"]), fmt_sh(W["A"]), W["ug_neg"], dir_s(t42, "toward"), bias_path(t42),
         share_word,
         ", ".join("%s %.0f %%" % (p, 100 * s_) for p, s_ in zip(PROPS, plug_share)),
         fmt_sh(pu_B), wide_m, fmt_sh(wp_B),
         fmt_sh(wp_A), pw_neg, ", ".join(away_p) if away_p else "none", len(lad_rows),
         min(drag_B), max(drag_B), min(drag_A), max(drag_A), RV["drag"][0], RV["drag"][1],
         floor_s(t42), mech,
         " ".join("At %s the same sign test gives %s (%s)."
                  % (t, dir_s(t, "toward"),
                     "under Not supported below" if len(WU[t]["toward"]) < len(PROPS)
                     else "the same direction as at %s" % t42) for t in others42)))
    # the other target(s): the same comparison, computed from the cells
    OTHER42 = []
    for t in others42:
        Wt = WU[t]
        lad_db = {p: [res_ug[row["key"]]["dbias"] for row in BLK[(p, t)]["rows"]
                      if row["a"] == "bdg" and row["v"] in ladder(p)] for p in PROPS}
        away_w = [p for p in PROPS if p not in Wt["toward"]]
        away_clear = [p for p in away_w if rowof(kw(p, t))["passes"]]
        OTHER42.append((len(Wt["toward"]) == len(PROPS),
            "- **Moving the batch mean toward the target at %s** (outside review section 4.2's "
            "%s scope). At %s, tau_mult %g moves mean f_B by %s relative to unguided (paired; on "
            "f_A %s). Unguided's bias is %s d, negative in %d of 3, so by the same sign test the "
            "shift is %s; bias/d goes %s. Over the %d ladder cells, d abs(bias)/d against "
            "unguided spans %s (|bias| rises on %s). Plug w1, with the same centring gain, "
            "moves mean f_B by %s "
            "against unguided: %s. %s%s."
            % (t, t42, t, wide_m, fmt_sh(Wt["B"]), fmt_sh(Wt["A"]),
               ", ".join("%s %+.2f" % (p, Wt["ugb"][p]) for p in PROPS), Wt["ug_neg"],
               dir_s(t, "toward"), bias_path(t), len(ladder("mu")),
               ", ".join("%s %+.2f to %+.2f" % (p, min(lad_db[p]), max(lad_db[p]))
                         for p in PROPS),
               ", ".join("%d of %d for %s" % (sum(1 for x in lad_db[p] if x > 0),
                                               len(lad_db[p]), p) for p in PROPS),
               fmt_sh(Wt["pu_B"]), dir_s(t, "p_toward"), floor_s(t),
               ("; the away shift on %s is on a cell that clears its own floor"
                % plist(away_clear)) if away_clear else "")))
    for ok_, txt_ in OTHER42:
        if ok_:
            A(txt_)
    q90b = [LEVER[(p, "q90", "cont")] for p in PROPS]
    A("- **At q90 bias governs coverage, not spread** (review section 4.4, idealised bound "
      "above): removing the unguided bias is worth %s in-band against a best spread lever of "
      "%s."
      % (", ".join("%s %+.4f" % (p, x["gain_b"]) for p, x in zip(PROPS, q90b)),
         ", ".join("%s %s" % (p, f(x["gain_s"], "%+.4f")) for p, x in zip(PROPS, q90b))))
    # chemistry cost against |w_eff| and against the knob: collinear in this grid
    xs = [abs(row["m"]["diag"]["bdg_w_eff"]) for _, row in lad_rows]
    ys = [row["m"]["mol_stab"] - BLK[(row["key"][0], row["key"][1])]["ug"]["mol_stab"]
          for _, row in lad_rows]
    ts = [row["mult"] for _, row in lad_rows]
    rho_w, rho_t, rho_wt = spearman(xs, ys), spearman(ts, ys), spearman(xs, ts)
    RHO_BLK = []
    for t in TGTS:
        for p in PROPS:
            sel = [i for i, (tt, row) in enumerate(lad_rows) if tt == t and row["key"][0] == p]
            bx, by, bt = [xs[i] for i in sel], [ys[i] for i in sel], [ts[i] for i in sel]
            RHO_BLK.append((p, t, spearman(bx, by), spearman(bt, by), spearman(bx, bt)))
    tau_wins = [b_ for b_ in RHO_BLK if abs(b_[3]) > abs(b_[2])]
    A("")
    A("**Not supported (review sections 1 and 4).**")
    A("")
    for ok_, txt_ in OTHER42:
        if not ok_:
            A(txt_)
    A("- **Telling |w_eff| from the knob as the driver of chemistry cost.** Review section 4.2 "
      "says the cost tracks |w_eff|. This grid cannot separate the two: over the %d ladder "
      "cells the Spearman correlation of d mol stab vs unguided with |run-mean w_eff| is "
      "%+.2f and with tau_mult %+.2f, but |w_eff| and tau_mult are themselves correlated at "
      "%+.2f over the same cells. Per block (rho with |w_eff| / with tau_mult / between the "
      "two, 5 cells each): %s; tau_mult has the larger |rho| in %d of 6 blocks. Descriptive, "
      "not a test: run-means hide sign flips and the cells are coupled."
      % (len(lad_rows), rho_w, rho_t, rho_wt,
         "; ".join("%s %s %+.2f / %+.2f / %+.2f" % b_ for b_ in RHO_BLK), len(tau_wins)))
    for t in TGTS:
        s_ = S[t]
        if s_["clear"]:
            tested = ("%d of those beat plug w1 at z >= %g on continuous in-band and %d on "
                      "decoded; %d lose at z <= -%g continuous and %d decoded; %d of the %d "
                      "point estimates are positive."
                      % (len(s_["beat_c"]), SIGMA, len(s_["beat_d"]), len(s_["lose_c"]), SIGMA,
                         len(s_["lose_d"]), len(s_["pos_c"]), len(s_["clear"])))
        else:
            tested = "so no ladder cell at this target carries an in-band verdict."
        A("- **An in-band gain over plug w1 at %s.** %d of %d ladder cells clear their own "
          "floor (%d of them knife-edge); %s %d cells sit below their own floor and carry no "
          "verdict (%d of them knife-edge)."
          % (t, len(s_["clear"]), s_["n"], len(s_["knife_clear"]), tested, len(s_["below"]),
             len(s_["knife_below"])))
    any_beat = any(S[t]["beat_c"] or S[t]["beat_d"] for t in TGTS)
    A("- **So: %s floor-clearing bdg cell beats plug w1 on in-band at |z| >= %g at either "
      "target.** With %d paired in-band tests against plug w1 the max |z| is %.2f (Sidak p = "
      "%.2g)."
      % ("a" if any_beat else "no", SIGMA, MULT["minus plug w1"]["m"],
         abs(MULT["minus plug w1"]["top"]["z"]), MULT["minus plug w1"]["ps"]))
    # the same question without the floor, and under the two sensitivity floors
    any_c = sum(1 for _, row in lad_rows if res_pw[row["key"]]["ib"][1] >= SIGMA)
    any_d = sum(1 for _, row in lad_rows if res_pw[row["key"]]["ibd"][1] >= SIGMA)
    sens = []
    for lab_, keyf in (("%.4f" % SF["sweep"], "s3621"), ("%.4f" % SF["v2"], "s3571")):
        for t in TGTS:
            cl = [row for tt, row in lad_rows if tt == t and row[keyf]]
            bt = [row for row in cl if res_pw[row["key"]]["ib"][1] >= SIGMA
                  or res_pw[row["key"]]["ibd"][1] >= SIGMA]
            sens.append("%s at %s: %d clear, %d beat" % (lab_, t, len(cl), len(bt)))
    A("- **The answer does not hinge on the floor.** Ignoring the floor entirely, %d of the "
      "%d ladder cells beat plug w1 at z >= %g on continuous in-band and %d on decoded. "
      "Under the sensitivity floors (never a verdict): %s."
      % (any_c, len(lad_rows), SIGMA, any_d, "; ".join(sens)))
    pw4 = [(p, t, res_pw[(p, t, "plug", 4.0, "tgt")],
            [row for row in BLK[(p, t)]["rows"] if row["a"] == "plug" and row["w"] == 4.0][0])
           for t in TGTS for p in PROPS]
    A("- **Strength alone.** plug w4 against plug w1: %s. It clears its own floor in %d of "
      "6 blocks."
      % ("; ".join("%s %s %+.4f (z %+.1f)" % (p, t, r_["ib"][0], r_["ib"][1])
                   for p, t, r_, _ in pw4), sum(1 for *_, row in pw4 if row["passes"])))
    n_cross = sum(1 for x in WT.values() if x["cross"])
    A("- **The loop settling, at its setpoint or at its equilibrium.** The setpoint is "
      "V_b/tau^2 = 1 and the law's equilibrium is V*/tau^2 = %.2f; the run-mean V_b/tau^2 "
      "is %.2f-%.2f at the tight end and %.2f-%.2f at the wide end. Per step (the table at "
      "the top), V_b/tau^2 crosses V* on %d of the %d re-run trajectories and stays on one "
      "side of it on the other %d. Nothing here says the loop settles within the window."
      % (VSTAR, min(lo05), max(lo05), min(hi15), max(hi15), n_cross, len(WT),
         len(WT) - n_cross))
    rsc = RV["replay_scope"]
    A("- **Anything about feedback versus a replayed schedule, uniqueness of widening, or "
      "the q90 headline.** These %d cells contain no open-loop replay; the review's port "
      "replay (section 4.3: a %s-%s %% remainder to the closed loop on achieved spread, %d/%d "
      "configurations, %s, %s and %s only, tau_mult %s and %s, n = %d) is separate and not "
      "re-verified here. There is no signed-w plug control (section 5 item 12), and the run "
      "is not the v2 run."
      % (len(cells), RV["replay"][0], RV["replay"][1], RV["replay"][2], RV["replay"][3],
         rsc["pairs"], rsc["props"][0], rsc["props"][1], rsc["mults"][0], rsc["mults"][1],
         rsc["n"]))
    A("")

    # ---- appendix: the handoff's own numbers -----------------------------------------------
    A("## Appendix: the handoff's section 6, and whether the port reproduces its direction")
    A("")
    tight_v = ladder("mu")[0]
    mid_v = [v for v in ladder("mu") if parse_variant(v)[1] == 1.0][0]
    wide = ladder("mu")[-1]
    hb = HO["bind"]
    if HO["tightf"]["v"] != tight_v or tight_v not in hb or mid_v not in hb:
        raise SystemExit("REFUSING: the handoff's 6.2 cells (%s, %s) are not the port's tight "
                         "and mid cells (%s, %s)" % (HO["tightf"]["v"], sorted(hb), tight_v, mid_v))
    if HO["cols"][-1] != wide:
        raise SystemExit("REFUSING: the handoff's widest 6.1 column %s is not the port's %s"
                         % (HO["cols"][-1], wide))
    # the handoff's floor: is it the results/sweep unguided x 0.9 (review section 4.6)?
    borrowed = abs(HO["floor"] - FLOOR_FRAC * SF["sweep_ug"]) < 1e-12
    if not borrowed or abs(RV["knife"]["floor"] - HO["floor"]) > 5e-5:
        raise SystemExit("REFUSING: the handoff floor %s is not %g x the results/sweep unguided "
                         "%.8g, or the review's %s" % (HO["floor_s"], FLOOR_FRAC, SF["sweep_ug"],
                                                     RV["knife"]["floor"]))
    bfl = ("borrowed floor %s = %g x %.8g, the results/sweep unguided"
           % (HO["floor_s"], FLOOR_FRAC, SF["sweep_ug"]))
    # the review's knife-edge z for the handoff's gap tight cell, recomputed
    k_st = round(hb[tight_v]["mol_stab"] * HO["n"])
    if abs(k_st / HO["n"] - hb[tight_v]["mol_stab"]) > 5e-5:
        raise SystemExit("REFUSING: handoff mol_stab %s is not a count out of n = %d"
                         % (hb[tight_v]["mol_stab"], HO["n"]))
    p_st = k_st / HO["n"]
    z_knife = (p_st - HO["floor"]) / math.sqrt(p_st * (1.0 - p_st) / HO["n"])
    if abs(abs(z_knife) - RV["knife"]["z"]) > 0.01:
        raise SystemExit("REFUSING: recomputed knife-edge z %.3f vs the review's -%.2f"
                         % (z_knife, RV["knife"]["z"]))
    port_diffs = []
    for lab_, hv_, pv_ in (("device", HO["device"], prov.get("device")),
                           ("batch", HO["batch"], any_r["batch"]), ("n", HO["n"], N),
                           ("seed", "/".join(map(str, HO["seeds"])), any_r["seed"])):
        if str(hv_) != str(pv_):
            port_diffs.append("%s (%s vs the port's %s)" % (lab_, hv_, pv_))
    A("**Author's Betty cells (seeds %s, n = %d, %s, batch %d, target not stated: q50 or dist "
      "per review section 4.4); not in this repo; not re-verifiable.** Read from "
      "[BDG_HANDOFF.md](../methods/BDG_HANDOFF.md) sections 5 and 6. Quoted verbatim, section "
      "6.1-6.3:" % ("/".join(map(str, HO["seeds"])), HO["n"], HO["device"], HO["batch"]))
    A("")
    A("**The quote carries claims the review ruled against:** 6.1's \"%s\" (review section "
      "4.5, R5: %s, %s), 6.1's yardstick for the largest reproducible widening elsewhere "
      "(section 4.5), 6.2's floor verdict on gap %s (section 4.6), 6.2's \"%s\" (section 4.4, "
      "R2: %s, the premise is right at %s and wrong at %s), and 6.3's framing of the bdg / "
      "btvg_var contrast (section 4.7). The quote is left as the author wrote it; each of "
      "these has a row with the review's ruling in the table below."
      % (HO["repl_s"], RV["alpha45"]["r5"], RV["alpha45"]["r5_note"], tight_v, HO["every_s"],
         RV["r2"]["ruling"], RV["r2"]["right"], RV["r2"]["wrong"]))
    A("")
    q0 = len(L)
    for line in HO["text"].splitlines():
        A(("> " + line) if line.strip() else ">")
    q1 = len(L)
    A("")
    A("The port differs from these cells in %s, so only the **direction** of each claim can "
      "be compared. Both port targets are shown because the handoff's target is unknown."
      % ", ".join(port_diffs))
    A("")
    A("| handoff claim | handoff value | port q50 | port q90 | port reproduces the direction? |")
    A("|---|---|---|---|---|")
    APPX = []

    def sgn(x):
        return 1 if x > 1.0 else (-1 if x < 1.0 else 0)

    def call(ok):
        return ("yes" if all(ok.values()) else ("no" if not any(ok.values()) else
                                                "q50 only" if ok["q50"] else "q90 only"))

    for p in PROPS:
        for col in HO["cols"]:
            hv = HO["t61"].get(p, {}).get(col)
            if hv is None:
                continue
            pk = [(t, (p, t, "bdg", 1.0, col)) for t in TGTS]
            if any(k not in cells for _, k in pk):
                A("| 6.1 %s %s sd/ug | %.3f / %.3f | no %s cell | no %s cell | not testable |"
                  % (p, col, hv[0], hv[1], col, col))
                APPX.append("not testable")
                continue
            hd = {sgn(hv[0]), sgn(hv[1])}
            vals = {}
            for t, k in pk:
                row = [r_ for r_ in BLK[(p, t)]["rows"] if r_["key"] == k][0]
                vals[t] = (row["rA"], row["rB"])
            dirs = [sgn(x) for t in TGTS for x in vals[t]]
            if len(hd) > 1:
                vd = "handoff seeds disagree"
            else:
                h = hd.pop()
                k_ = sum(1 for d_ in dirs if d_ == h)
                vd = "yes" if k_ == len(dirs) else ("no" if k_ == 0 else
                                                     "partly (%d of %d)" % (k_, len(dirs)))
            APPX.append(vd)
            A("| 6.1 %s %s sd/ug %s 1 | %.3f / %.3f | f_A %.3f, f_B %.3f | f_A %.3f, f_B %.3f | %s |"
              % (p, col, "<" if hv[0] < 1 else ">", hv[0], hv[1], vals["q50"][0], vals["q50"][1],
                 vals["q90"][0], vals["q90"][1], vd))
    mono = {}
    for t in TGTS:
        a_ = b_ = 0
        for p in PROPS:
            lad = sorted([row for row in BLK[(p, t)]["rows"]
                          if row["a"] == "bdg" and row["v"] in ladder(p)], key=lambda r: r["mult"])
            a_ += all(y["rA"] >= x["rA"] for x, y in zip(lad, lad[1:]))
            b_ += all(y["rB"] >= x["rB"] for x, y in zip(lad, lad[1:]))
        mono[t] = (a_, b_)
    A("| 6.1 monotone requested -> achieved | %d/%d curves, %s | f_A %d/3, f_B %d/3 | "
      "f_A %d/3, f_B %d/3 | %s |"
      % (HO["mono"][0], HO["mono"][1], HO["mono"][2], mono["q50"][0], mono["q50"][1],
         mono["q90"][0], mono["q90"][1],
         "yes" if sum(mono["q50"]) + sum(mono["q90"]) == 12 else
         "partly (%d of 12)" % (sum(mono["q50"]) + sum(mono["q90"]))))
    hw = [p for p in PROPS if HO["t61"].get(p, {}).get(wide) and min(HO["t61"][p][wide]) > 1.0]
    wr = {t: sum(1 for p in PROPS
                 for row in BLK[(p, t)]["rows"] if row["v"] == wide and row["rB"] > 1.0)
          for t in TGTS}
    a45 = RV["alpha45"]
    A("| 6.1 widening on all three properties (%s): \"%s\" | sd/ug > 1 on %d/%d, both seeds | "
      "f_B > 1 on %d/3 | f_B > 1 on %d/3 | %s in direction; \"replicates\" is ruled borderline "
      "on alpha (review section 4.5, R5: %s, %s): the handoff's alpha widening %s / %s is %s "
      "and %s se from none unpaired and about %s and %s se paired, %s |"
      % (wide, HO["repl_s"], len(hw), len([p for p in PROPS if HO["t61"].get(p, {}).get(wide)]),
         wr["q50"], wr["q90"], "yes" if wr["q50"] + wr["q90"] == 6 else
         "partly (%d of 6)" % (wr["q50"] + wr["q90"]), a45["r5"], a45["r5_note"], a45["h"][0],
         a45["h"][1], a45["unpaired"][0], a45["unpaired"][1], a45["paired"][0],
         a45["paired"][1], a45["word"]))

    def get(p, t, v):
        return [row for row in BLK[(p, t)]["rows"] if row["v"] == v and row["a"] == "bdg"][0]

    def fl(row):
        return ("PASS" if row["passes"] else "FAIL") + (" (knife-edge)" if row["knife"] else "")

    def plugrow(p, t):
        return [row for row in BLK[(p, t)]["rows"] if row["lab"] == "plug w1"][0]

    # the two sensitivity floors, by their flag in the rows; the handoff's own is the first
    SENS = (("the handoff's borrowed %.4f" % SF["sweep"], "s3621"), ("%.4f" % SF["v2"], "s3571"))

    # 6.1's floor claim on the widening cells
    f61 = HO["fl61"]

    def ok61(key):
        return {t: all(get(p, t, wide)[key] for p in f61["clear"])
                and not get(f61["fail"], t, wide)[key] for t in TGTS}

    def sens_note(okf):
        """The sensitivity-floor outcomes, and at which targets the own-floor disagreement
        with the handoff comes from the change of floor (the handoff's borrowed floor, the
        first of SENS, reproduces the claim there)."""
        own = okf("passes")
        outs = [(lab_, okf(k_)) for lab_, k_ in SENS]
        s_ = "Sensitivity, not a verdict: " + "; ".join(
            "under %s: %s" % (lab_, call(o_)) for lab_, o_ in outs)
        flip = [t for t in TGTS if not own[t] and outs[0][1][t]]
        if flip:
            s_ += ("; at %s the own-floor difference from the handoff comes from the change of "
                   "floor, not from the port disagreeing" % " and ".join(flip))
        return s_

    ok = ok61("passes")
    knife_ok = [get(p, t, wide)["knife"] for t in TGTS if ok[t] for p in f61["clear"]]
    call61 = (call(ok) + " against the own floor (the verdict)"
              + ("; every pass is knife-edge" if knife_ok and all(knife_ok) else "")
              + "; one port seed. " + sens_note(ok61))
    A("| 6.1 floor on the widening cells (%s): %s clear, %s fails | %s clear on both seeds; "
      "%s fails on seed %d, as does its control (%s) | %s | %s | %s |"
      % (wide, " and ".join(f61["clear"]), f61["fail"], " and ".join(f61["clear"]), f61["fail"],
         f61["fail_seed"], bfl,
         "; ".join(["%s %s" % (p, fl(get(p, "q50", wide))) for p in f61["clear"]]
                   + ["%s %s, its plug w1 control %s" % (f61["fail"], fl(get(f61["fail"], "q50", wide)),
                                                          fl(plugrow(f61["fail"], "q50")))]),
         "; ".join(["%s %s" % (p, fl(get(p, "q90", wide))) for p in f61["clear"]]
                   + ["%s %s, its plug w1 control %s" % (f61["fail"], fl(get(f61["fail"], "q90", wide)),
                                                          fl(plugrow(f61["fail"], "q90")))]),
         call61))
    # 6.1's yardstick, recomputed from results/sweep
    if W45:
        h_s2 = [x["s2"] for x in W45 if x["s2"] is not None]
        A("| 6.1 yardstick: the largest reproducible floor-clearing widening among existing "
          "cells | %sx, not seed-reproducible | not a port claim; recomputed here from "
          "results/sweep (seed %d, n = %d) against its unguided: %s | not a port claim | "
          "ruled false (review section 4.5): these cells widen %.3f-%.3fx on seed %d (%.3f-%.3fx "
          "on seed 2 per the review) and the yardstick is inside noise; no uniqueness claim |"
          % (HO["yard_s"], SWEEP_SEED, SWEEP_N,
             "; ".join("%s %s %s w %g %.3fx, stab %.4f (%s %.4f), mean shift %+.2f d"
                       % (x["arm"], x["prop"], x["tgt"], x["w"], x["ratio"], x["stab"],
                          ">=" if x["stab"] >= SF["sweep"] else "<", SF["sweep"], x["shift"])
                       for x in W45),
             min(x["ratio"] for x in W45), max(x["ratio"] for x in W45), SWEEP_SEED,
             min(h_s2) if h_s2 else float("nan"), max(h_s2) if h_s2 else float("nan")))
        APPX.append("ruled false")
    ok = {t: get("gap", t, tight_v)["m"]["in_band"] > get("gap", t, mid_v)["m"]["in_band"]
          for t in TGTS}
    A("| 6.2 gap: %s in-band above %s | %.4f vs %.4f | %.4f vs %.4f | %.4f vs %.4f | %s |"
      % (tight_v, mid_v, hb[tight_v]["in_band"], hb[mid_v]["in_band"],
         get("gap", "q50", tight_v)["m"]["in_band"], get("gap", "q50", mid_v)["m"]["in_band"],
         get("gap", "q90", tight_v)["m"]["in_band"], get("gap", "q90", mid_v)["m"]["in_band"],
         call(ok)))
    ok = {t: (not get("gap", t, tight_v)["passes"]) and get("gap", t, mid_v)["passes"] for t in TGTS}
    A("| 6.2 gap: %s fails the floor, %s passes | %s (stab %.4f = %d/%d) / %s (%s) | %s / %s | "
      "%s / %s | %s; the handoff's fail is ruled knife-edge (review section 4.6: %+.2f se from "
      "the %s, not a FAIL) |"
      % (tight_v, mid_v, "passes" if hb[tight_v]["passes"] else "fails", p_st, k_st, HO["n"],
         "passes" if hb[mid_v]["passes"] else "fails", bfl,
         fl(get("gap", "q50", tight_v)), fl(get("gap", "q50", mid_v)),
         fl(get("gap", "q90", tight_v)), fl(get("gap", "q90", mid_v)), call(ok), z_knife,
         "borrowed floor"))
    # plug w4 best raw in-band, with ties reported
    GATE_COPIES = {"plug w1 (tgt copy)", "bdg e0t1", "bdg e4t1.5o"}
    best4, tie4, fail4 = {}, {}, {}
    for t in TGTS:
        best4[t], tie4[t] = 0, []
        for p in PROPS:
            rows_ = BLK[(p, t)]["rows"]
            top = max(r_["m"]["in_band"] for r_ in rows_)
            p4 = [r_ for r_ in rows_ if r_["a"] == "plug" and r_["w"] == 4.0][0]
            if p4["m"]["in_band"] != top:
                continue
            others = [r_["lab"] for r_ in rows_ if r_ is not p4 and r_["m"]["in_band"] == top]
            if "plug w1" in others:
                others = [o for o in others if o not in GATE_COPIES]
            if others:
                tie4[t].append("%s, with %s" % (p, " and ".join(others)))
            else:
                best4[t] += 1
        fail4[t] = sum(1 for p in PROPS for row in BLK[(p, t)]["rows"]
                       if row["a"] == "plug" and row["w"] == 4.0 and not row["passes"])
    n_tie = sum(len(v) for v in tie4.values())
    n_best = best4["q50"] + best4["q90"]

    def best_s(t):
        return "best on %d/3%s" % (best4[t], (", tied on %d/3 (%s)" % (len(tie4[t]), "; ".join(tie4[t])))
                                   if tie4[t] else "")

    A("| 6.2 plug w4: best raw in-band on all three | %s | %s | %s | %s |"
      % (" / ".join("%.4f" % HO["plug4"]["in_band"][p] for p in PROPS), best_s("q50"),
         best_s("q90"), "yes" if n_best == 6 else
         "partly (%d of 6 outright%s)" % (n_best, (", %d tie" % n_tie) if n_tie else "")))
    def fails_s(rows_by_p, t):
        """'fails k/n (alpha knife-edge)': the port's own-floor fails, knife-edge marked."""
        fr = [(p, r_) for p, r_ in rows_by_p(t) if not r_["passes"]]
        kn = [p for p, r_ in fr if r_["knife"]]
        return ("fails %d/%d" % (len(fr), len(rows_by_p(t)))
                + (" (%s knife-edge)" % " and ".join(kn) if kn else "")), len(fr), len(kn)

    def fails_call(rows_by_p):
        res_ = [fails_s(rows_by_p, t) for t in TGTS]
        nf, nk = sum(x[1] for x in res_), sum(x[2] for x in res_)
        tot = sum(len(rows_by_p(t)) for t in TGTS)
        return (("yes" if nf == tot else "partly (%d of %d)" % (nf, tot))
                + ("; knife-edge on %d" % nk if nk else ""))

    p4_rows = lambda t: [(p, [row for row in BLK[(p, t)]["rows"]
                              if row["a"] == "plug" and row["w"] == 4.0][0]) for p in PROPS]
    A("| 6.2 plug w4: fails the floor on all three | stab %s (%s) | %s | %s | %s |"
      % (" / ".join("%.4f" % HO["plug4"]["mol_stab"][p] for p in PROPS), bfl,
         fails_s(p4_rows, "q50")[0], fails_s(p4_rows, "q90")[0], fails_call(p4_rows)))
    tp = HO["tightf"]["props"]
    tv_rows = lambda t: [(p, get(p, t, tight_v)) for p in tp]
    A("| 6.2 every %s cell fails the floor on %s | fails on %d/%d named (%s) | %s | %s | %s |"
      % (tight_v, " and ".join(tp), len(tp), len(tp), ", ".join(tp),
         fails_s(tv_rows, "q50")[0], fails_s(tv_rows, "q90")[0], fails_call(tv_rows)))
    wl = {t: sum(1 for p in PROPS if get(p, t, wide)["m"]["in_band"] < get(p, t, mid_v)["m"]["in_band"])
          for t in TGTS}
    wlp = {t: sum(1 for p in PROPS if get(p, t, wide)["m"]["in_band"]
                  < M[(p, t, "plug", 1.0, "bdgctl")]["in_band"]) for t in TGTS}
    r2w = RV["r2"]["wrong"]
    kw1 = sum(1 for p in PROPS if LEVER[(p, r2w, "cont")]["k"] is not None
              and LEVER[(p, r2w, "cont")]["k"] > 1.0)
    A("| 6.2 widening lowers in-band %s: \"%s\" | %s | %s below %s on %d/3, below "
      "plug w1 on %d/3 | %s below %s on %d/3, below plug w1 on %d/3 | %s; the premise is "
      "ruled right at %s and wrong at %s (review section 4.4, R2: %s): in the idealised "
      "bound above, the in-band-maximising spread scale k* at %s is above 1 on %d of 3 "
      "(continuous view) |"
      % (HO["every"], HO["every_s"], HO["every"], wide, mid_v, wl["q50"], wlp["q50"], wide,
         mid_v, wl["q90"], wlp["q90"], "yes" if wl["q50"] + wl["q90"] == 6 else
         "partly (%d of 6 vs %s)" % (wl["q50"] + wl["q90"], mid_v), RV["r2"]["right"], r2w,
         RV["r2"]["ruling"], r2w, kw1))
    h_b = HO["r63"]["bdg"]
    A("| 6.3 bdg spread range over its sweep | %g-%g %% | %s | %s | %s |"
      % (h_b[0], h_b[1],
         ", ".join("%s %.0f %%" % (p, 100 * CTX[(p, "q50", "bdg", "port")]["rng"]) for p in PROPS),
         ", ".join("%s %.0f %%" % (p, 100 * CTX[(p, "q90", "bdg", "port")]["rng"]) for p in PROPS),
         "same order (magnitude only; the port's range is sd(f_B))"))
    h_v = HO["r63"]["btvg_var"]
    if all((p, t, "btvg_var", "cmp") in CTX for p in PROPS for t in TGTS):
        bigger = sum(1 for p in PROPS for t in TGTS
                     if CTX[(p, t, "bdg", "port")]["rng"] > CTX[(p, t, "btvg_var", "cmp")]["rng"])
        A("| 6.3 btvg_var spread range | %g-%g %% | no btvg_var in the port; results/sweep `cmp`: "
          "%s | results/sweep `cmp`: %s | bdg's range exceeds btvg_var's in %d of 6 (different "
          "runs, aggregate) |"
          % (h_v[0], h_v[1],
             ", ".join("%s %.0f %%" % (p, 100 * CTX[(p, "q50", "btvg_var", "cmp")]["rng"]) for p in PROPS),
             ", ".join("%s %.0f %%" % (p, 100 * CTX[(p, "q90", "btvg_var", "cmp")]["rng"]) for p in PROPS),
             bigger))
    if all((p, t, "plug", "cmp") in CTX for p in PROPS for t in TGTS):
        A("| 6.3 the bdg / btvg_var contrast: \"the scientific core\", \"the comparison is "
          "valid\" (same %s) | as quoted; the handoff's section 5 runs btvg_var at batch %d and "
          "bdg at batch %d | plug with no variance term, results/sweep `cmp`: %s | "
          "results/sweep `cmp`: %s | ruled incomplete (review section 4.7): there is no plug "
          "control, plug alone sweeps sd by %s-%s %% per the review, and the contrast needs "
          "plug's w-sweep, btvg_var and the BDG ladder with a bias column before it can be "
          "called the core |"
          % (HO["core_same"], HO["btvg_var_batch"], HO["batch"],
             ", ".join("%s %.0f %%" % (p, 100 * CTX[(p, "q50", "plug", "cmp")]["rng"]) for p in PROPS),
             ", ".join("%s %.0f %%" % (p, 100 * CTX[(p, "q90", "plug", "cmp")]["rng"]) for p in PROPS),
             RV["plug47"][0], RV["plug47"][1]))
        APPX.append("ruled incomplete")
    lp_ = lambda key: {t: sum(1 for tt, r_ in lad_rows if tt == t and r_[key]) for t in TGTS}
    lp = lp_("passes")
    maj = lambda key: {t: lp_(key)[t] > S[t]["n"] / 2 for t in TGTS}
    f63b = HO["f63"]["bdg"]
    if f63b.lower().startswith("mostly pass"):
        c63 = call(maj("passes"))
        vd63 = ("%s against the own floor (the verdict; a majority must pass). "
                % c63) + sens_note(maj) + " (%s)" % "; ".join(
                    "%s: %s" % (lab_, ", ".join("%d of %d at %s" % (lp_(k_)[t], S[t]["n"], t)
                                                 for t in TGTS)) for lab_, k_ in SENS)
    else:
        vd63 = "not interpreted (%s)" % f63b
    A("| 6.3 bdg floor: %s | %s | %d of %d pass own floor | %d of %d | %s |"
      % (f63b, f63b, lp["q50"], S["q50"]["n"], lp["q90"], S["q90"]["n"], vd63))
    A("| 6.3 btvg_var floor: %s | %s | not testable (no btvg_var cell in the port) | not "
      "testable | not testable |" % (HO["f63"]["btvg_var"], HO["f63"]["btvg_var"]))
    A("")

    out = NL.join(L) + NL
    # banned wording: every line outside the verbatim quote
    bad = []
    for i, line in enumerate(L):
        if q0 <= i < q1:
            continue
        lo_ = line.lower()
        bad += ["%r (page line ~%d)" % (phr, i + 1) for phr in FORBIDDEN if phr in lo_]
        bad += ["%r without a ruling (page line ~%d)" % (pat, i + 1) for pat in RULED_ONLY
                if re.search(pat, lo_) and "ruled" not in lo_]
    quoted = [l_ for l_ in L[q0:q1]]
    if quoted != [("> " + l_) if l_.strip() else ">" for l_ in HO["text"].splitlines()]:
        bad.append("the handoff quote is not verbatim")
    if bad:
        raise SystemExit("REFUSING to write: the page contains banned wording: %s" % "; ".join(bad))

    if args.print:
        print(out)
    if args.md_out:
        dst = args.md_out if os.path.isabs(args.md_out) else os.path.join(ROOT, args.md_out)
        with open(dst, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
        print("wrote %s (%d lines)" % (dst, out.count(NL)), file=sys.stderr)

    # ---- summary ---------------------------------------------------------------------------
    headline = []
    for t in TGTS:
        for p in PROPS:
            blk = BLK[(p, t)]
            clear = [row for row in blk["rows"] if row["a"] == "bdg" and row["v"] in ladder(p)
                     and row["passes"]]
            best = max(clear, key=lambda r: (r["m"]["in_band"], r["m"]["in_band_dec"])) if clear else None
            pw1 = M[(p, t, "plug", 1.0, "bdgctl")]
            ug = blk["ug"]
            hb_ = dict(prop=p, target=t, plug_w1_in_band=pw1["in_band"],
                       unguided_in_band=ug["in_band"])
            p4row = [row for row in blk["rows"] if row["a"] == "plug" and row["w"] == 4.0][0]
            p4res = res_pw[(p, t, "plug", 4.0, "tgt")]
            if best:
                rs = res_pw[best["key"]]
                hb_.update(best_floor_clearing_bdg=best["v"], in_band=best["m"]["in_band"],
                           delta_vs_plug_w1=rs["ib"][0], z_vs_plug_w1=rs["ib"][1],
                           note="dec in-band %.4f (plug w1 %.4f, d %+.4f z %+.2f); stab %.4f "
                                "vs own floor %.4f%s; plug w4 in-band %.4f (d %+.4f z %+.2f) "
                                "stab %.4f %s own floor; unguided dec %.4f"
                                % (best["m"]["in_band_dec"], pw1["in_band_dec"], rs["ibd"][0],
                                   rs["ibd"][1], best["m"]["mol_stab"], blk["floor"],
                                   " knife-edge" if best["knife"] else "",
                                   p4row["m"]["in_band"], p4res["ib"][0], p4res["ib"][1],
                                   p4row["m"]["mol_stab"],
                                   "clears" if p4row["passes"] else "below",
                                   ug["in_band_dec"]))
            else:
                lad = [row for row in blk["rows"] if row["a"] == "bdg" and row["v"] in ladder(p)]
                bo = max(lad, key=lambda r: (r["m"]["in_band"], r["m"]["in_band_dec"]))
                rs = res_pw[bo["key"]]
                hb_.update(best_floor_clearing_bdg="none",
                           note="no ladder cell clears the own floor %.4f (plug w1 %.4f is below "
                                "it too); best ladder cell regardless of floor: %s in-band %.4f "
                                "(d %+.4f z %+.2f vs plug w1), dec %.4f (d %+.4f z %+.2f), stab "
                                "%.4f; plug w4 in-band %.4f (d %+.4f z %+.2f) stab %.4f"
                                % (blk["floor"], pw1["mol_stab"], bo["v"], bo["m"]["in_band"],
                                   rs["ib"][0], rs["ib"][1], bo["m"]["in_band_dec"],
                                   rs["ibd"][0], rs["ibd"][1], bo["m"]["mol_stab"],
                                   p4row["m"]["in_band"], p4res["ib"][0], p4res["ib"][1],
                                   p4row["m"]["mol_stab"]))
            headline.append(hb_)
    summ = dict(
        lines=out.count(NL), gates=dict(e0t1=e0, onesided=e1o, plug_copies=tg),
        headline=headline, n_tests_all=MULT["all"]["m"], n_big_all=MULT["all"]["big"],
        max_abs_z_all=abs(MULT["all"]["top"]["z"]),
        n_tests_pw1=MULT["minus plug w1"]["m"], n_big_pw1=MULT["minus plug w1"]["big"],
        max_abs_z_pw1=abs(MULT["minus plug w1"]["top"]["z"]),
        sidak_pw1=MULT["minus plug w1"]["ps"],
        top_pw1="%s %s %s %s" % (MULT["minus plug w1"]["top"]["p"], MULT["minus plug w1"]["top"]["t"],
                                 MULT["minus plug w1"]["top"]["cell"], MULT["minus plug w1"]["top"]["metric"]),
        any_beat=any_beat, beat_ignoring_floor=(any_c, any_d), sens=sens,
        S={t: dict(n=S[t]["n"], clear=len(S[t]["clear"]), beat_c=len(S[t]["beat_c"]),
                   beat_d=len(S[t]["beat_d"]), lose_c=len(S[t]["lose_c"]),
                   lose_d=len(S[t]["lose_d"]), pos_c=len(S[t]["pos_c"]),
                   below=len(S[t]["below"])) for t in TGTS},
        mono=(mono_A, mono_B), appendix=APPX, lo05=(min(lo05), max(lo05)),
        hi15=(min(hi15), max(hi15)), rho=(rho_w, rho_t), rho_w_tau=rho_wt,
        rho_blocks=RHO_BLK, sweep_problems=sweep_problems, wt_rejected=WT_rejected,
        vstar_cross={"%s/%s/t%g" % (x["prop"], x["target"], x["tau_mult"]):
                     (x["cross"], x["first_cross"], x["V0"], x["Vend"]) for x in WT.values()},
        plug4_best=(best4, tie4), drag=(min(drag_B), max(drag_B), min(drag_A), max(drag_A)),
        w45=[(x["arm"], x["prop"], x["tgt"], x["w"], x["ratio"], x["stab"]) for x in W45],
        z_knife=z_knife)
    print(json.dumps({k: v for k, v in summ.items() if k != "headline"}, indent=1, default=str))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(summ, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
