"""Write docs/results/FULL_RUN_RESULTS.md: every results table, raw and normalised.

    python proj1/scripts/results_tables.py

Reads only files on disk, so every number in the document traces to a cell:
  results/full/n5000/seed*/*__full.json   the full run (7 arms incl. tfg)
  results/full/n5000/frozen_q90_tfg.json  FR3a / FR3 strengths, all 7 arms
  results/dist_report_tfg.json            dist_report.py --frozen <that file>
                                          (gap closure, published units)
  results/sweep/*__cmp.json               our compare stage (n=512)
  results/transfer/compare/tr__*.json     the EDMsecond transfer compare stage
  results/dist_report_all.json            dist_report.py with NO --frozen: every
                                          (arm, strength) in the full run, so the
                                          FR3 secondary (w=4, seed 1) is included
                                          (tracking, residual sd, centring ceiling)
  results/force_share/mu__<arm>__w<w>.json velocity_share_diag.py --rules euler:
                                          the guidance correction's share of the
                                          sampling velocity at each strength
and runs full_run_table.py for the pre-registered FR5 verdicts. Regenerate
the two dist reports first if any cell changed:
    python proj1/scripts/dist_report.py --dir "results/full/n5000/seed*" \
        --frozen results/full/n5000/frozen_q90_tfg.json --out results/dist_report_tfg.json
    python proj1/scripts/dist_report.py --dir "results/full/n5000/seed*" \
        --out results/dist_report_all.json
and the force shares with proj1/scripts/force_share.py.

Sections 1-4 are unchanged in content and numbering (other docs cite 4).
Sections 5-9 answer "does guidance work?" for a reader who has only seen the
pre-registered table, where FR3a's chemistry floor froze most arms at a
strength that is almost off.

NORMALISATIONS, so a reader need not know each property's units:
  in-band x unguided   how many times more molecules land within delta of
                       their target than with no guidance (1.00x = no help)
  MAE cut              % reduction of mean absolute error vs unguided
  gap closure          0 = no better than a same-size molecule that ignores
                       the target (size-shuffle), 1 = the real test molecule
  chemistry %          mol_stability / unguided's; the rubric needs >= 90%
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "docs", "results", "FULL_RUN_RESULTS.md")
PROPS = ("mu", "alpha", "gap")
ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var")
TARMS = ("unguided", "plug", "tmpd", "lgd_mc", "btvg", "btvg_var")   # no tfg there
HA_MEV = 27211.386
DEC = {"mu": 3, "alpha": 2, "gap": 0}


def phys(p, x):
    """Our cells hold gap in Hartree; the tables use meV like EDM/EEGSDE."""
    return x * HA_MEV if p == "gap" else x


def f(x, nd=3):
    return ("%%.%df" % nd) % x


def pct(x):
    """A fraction as a percent that keeps its sign: whole percents, one decimal
    under 1% so a small loss (-0.4%) is not printed as 0%."""
    v = 100 * x
    if abs(v) < 1:
        s = "%.1f%%" % v
        return "0.0%" if s in ("-0.0%", "0.0%") else s
    return "%d%%" % round(v)


def f2(x):
    """Two decimals, with -0.00 shown as 0.00 (it is a rounding sign only)."""
    s = "%.2f" % x
    return "0.00" if s == "-0.00" else s


def rjson(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fr5_rows():
    """{(prop, arm): (d_in_band, sigma_ind, verdict)} from full_run_table.py."""
    txt = subprocess.run(
        [sys.executable, os.path.join(HERE, "full_run_table.py"), "--frozen",
         os.path.join(ROOT, "results", "full", "n5000", "frozen_q90_tfg.json")],
        capture_output=True, text=True, encoding="utf-8", check=True).stdout
    rows = {}
    for block in re.split(r"^## ", txt, flags=re.M):
        m = re.match(r"(mu|alpha|gap) ", block)
        if not m:
            continue
        for line in block.splitlines():
            mm = re.match(r"\| (\w+) \| ([+-][\d.]+) \| ([+-][\d.]+) \| [^|]+ \| [^|]+ \| "
                          r"[^|]+ \| [^|]+ \| (.+) \|$", line)
            if mm:
                rows[(m.group(1), mm.group(1))] = (float(mm.group(2)), float(mm.group(3)),
                                                   mm.group(4))
    return rows


def fr3a(store, p, a, tgt="q90"):
    """(cell, n strengths clearing the floor): the full run's FR3a rule."""
    u = store[(p, "unguided", tgt, 1.0)]
    if a == "unguided":
        return u, 1
    rs = [r for (pp, aa, tt, _), r in store.items() if pp == p and aa == a and tt == tgt]
    ok = [r for r in rs if r["mol_stability"] >= 0.9 * u["mol_stability"] - 1e-12]
    if ok:
        return min(ok, key=lambda r: (r["prop_mae_eval"], float(r["w"]))), len(ok)
    return max(rs, key=lambda r: (r["mol_stability"], -float(r["w"]))), 0


def load_store(pattern):
    store = {}
    for fn in glob.glob(pattern):
        r = rjson(fn)
        store[(r["prop"], r["arm"], r["target_name"], float(r["w"]))] = r
    return store


def main():
    rep = rjson(os.path.join(ROOT, "results", "dist_report_tfg.json"))
    fz = rjson(os.path.join(ROOT, "results", "full", "n5000", "frozen_q90_tfg.json"))
    W, WM = fz["frozen_w"], fz["frozen_w_mae"]
    A = {p: {k.split("@")[0]: v for k, v in rep[p]["arms"].items()} for p in PROPS}
    missing = [(p, a) for p in PROPS for a in ARMS if a not in A[p]]
    if missing:
        raise SystemExit("dist_report_tfg.json lacks %s -- regenerate it (see docstring)" % missing)
    fr5 = fr5_rows()
    L = []
    out = L.append

    out("# Full-run results: every arm, raw and normalised")
    out("")
    out("**Generated by `python proj1/scripts/results_tables.py`. Do not edit by hand; "
        "re-run the script.** Sources and the regenerate command are in its docstring. "
        "The pre-registered verdict is `full_run_table.py` (full output with tfg: "
        "[FULL_RUN_TABLE_WITH_TFG.md](FULL_RUN_TABLE_WITH_TFG.md)); this page adds the "
        "normalised views and the transfer screen. Protocol and caveats: "
        "[../status/SCOPE_FM_GUIDANCE_STATUS.md](../status/SCOPE_FM_GUIDANCE_STATUS.md), "
        "FULL-RUN RESULT and AMENDMENT FR2a.")
    out("")
    out("**How to read the normalised columns.** *in-band x unguided*: how many times more "
        "molecules land within delta of their target than with no guidance (1.00x = no "
        "help). *MAE cut*: % reduction of mean absolute error vs unguided. *Gap closure*: "
        "0 = no better than a same-size molecule that ignores the target, 1 = the real "
        "test molecule, negative = worse than ignoring it. *Chemistry %*: molecule "
        "stability relative to unguided; the rubric requires >= 90%.")
    out("")
    out("**What must travel with these numbers.** (1) `tfg` is POST HOC: it replaced "
        "dflow on 23 Sep after this run was read, same seeds/targets/settings, strength "
        "frozen by the same FR3a rule on its own compare cells. (2) The strength "
        "confound: the chemistry floor held plug, tmpd, btvg and tfg to weak strengths "
        "(tfg at 1-5% of its published step), while lgd_mc kept w = 4. (3) Property "
        "metrics score the continuous atom features every arm ends on; for tfg the "
        "decoded (argmax one-hot) view is shown too. At the frozen strengths it is small "
        "except on mu, where decoding removes about a quarter of tfg's gain; at high "
        "strength it is large. Unguided has no decoded view, so that comparison is not "
        "like-for-like. "
        "(4) Section 4 is a one-seed n = 512 screen, not a verdict.")
    out("")

    # the headline and the metric glossary; sections 5-9 at the end. Imported
    # here, not at module level, because results_evidence imports this module.
    import results_evidence as ev
    ALL = rjson(os.path.join(ROOT, "results", "dist_report_all.json"))
    ev.read_first(out, ALL, W, WM)

    # 1. main
    out("## 1. Main result: our flow model, full run "
        "(dist, n = 5,000 x 3 seeds, FR3a strengths)")
    out("")
    out("### 1a. Raw")
    out("")
    out("| arm | w (mu/alpha/gap) | in-band mu | in-band alpha | in-band gap | MAE mu (D) "
        "| MAE alpha (Bohr^3) | MAE gap (meV) | mol stab mu / alpha / gap |")
    out("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        out("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            "**tfg** (post hoc)" if a == "tfg" else a,
            " / ".join("%g" % W[a][p] for p in PROPS),
            *(f(A[p][a]["in_band"]) for p in PROPS),
            *(f(A[p][a]["mae_published_units"], DEC[p]) for p in PROPS),
            " / ".join(f(A[p][a]["mol_stability"]) for p in PROPS)))
    out("")
    out("### 1b. Normalised (each arm against unguided, same molecules)")
    out("")
    out("| arm | in-band x unguided (mu/alpha/gap) | MAE cut (mu/alpha/gap) "
        "| gap closure (mu/alpha/gap) | chemistry % of unguided (mu/alpha/gap) |")
    out("|---|---|---|---|---|")
    for a in ARMS:
        u = {p: A[p]["unguided"] for p in PROPS}
        out("| %s | %s | %s | %s | %s |" % (
            "**tfg**" if a == "tfg" else a,
            " / ".join("%.2fx" % (A[p][a]["in_band"] / u[p]["in_band"]) for p in PROPS),
            " / ".join(pct(1 - A[p][a]["mae"] / u[p]["mae"]) for p in PROPS),
            " / ".join(f2(A[p][a]["gap_closure"]) for p in PROPS),
            " / ".join("%.0f%%" % (100 * A[p][a]["mol_stability"] / u[p]["mol_stability"])
                       for p in PROPS)))
    out("")
    out("tfg on decoded atom types at these strengths (in-band): %s, against %s on "
        "the continuous features." % (
            " / ".join(f(A[p]["tfg"].get("in_band_dec", float("nan"))) for p in PROPS)
            if all("in_band_dec" in A[p]["tfg"] for p in PROPS) else
            " / ".join(f(_dec_in_band(p, W["tfg"][p])) for p in PROPS),
            " / ".join(f(A[p]["tfg"]["in_band"]) for p in PROPS)))
    out("")
    out("### 1c. Pre-registered verdict (FR5): btvg against each arm, on in-band")
    out("")
    out("sigma = independent-samples z; only |sigma| >= 3 decides, both arms above the "
        "chemistry floor. Positive d favours btvg. The supplementary PAIRED z (same "
        "molecules, same noise; full table in FULL_RUN_TABLE_WITH_TFG.md) crosses 3 "
        "among FR5 ties in three places: alpha versus unguided +3.05, gap versus "
        "plug -3.32, and gap versus tmpd -3.61. These are supplementary comparisons; "
        "an FR5 tie is failure to establish superiority, not evidence of equivalence. "
        "The additional target-cluster analysis in WORKSHOP_CLAIM_AUDIT.md groups "
        "repeated targets and accounts for its 36-contrast comparison family.")
    out("")
    out("| btvg vs | mu: d in-band (sigma) | alpha: d in-band (sigma) "
        "| gap: d in-band (sigma) | verdict |")
    out("|---|---|---|---|---|")
    for a in ARMS:
        if a == "btvg":
            continue
        cells = [fr5.get((p, a)) for p in PROPS]
        verd = "; ".join(sorted({c[2] for c in cells if c}))
        out("| %s | %s | %s | %s | %s |" % (
            a, *("%+.4f (%+.2f)" % (c[0], c[1]) if c else "-" for c in cells), verd))
    out("")

    # 2. secondary
    out("## 2. Unconstrained strengths "
        "(FR3 as registered: best MAE, no chemistry floor; seed 20261001, n = 5,000)")
    out("")
    out("| arm | w (mu/alpha/gap) | in-band mu / alpha / gap | in-band x unguided "
        "| mol stab mu / alpha / gap | chemistry % of unguided | clears floor? |")
    out("|---|---|---|---|---|---|---|")
    sec = {}
    for p in PROPS:
        for a in ARMS:
            fn = glob.glob(os.path.join(ROOT, "results", "full", "n5000", "seed20261001",
                                        "%s__%s__dist__w%g__tmin0.5__full.json" % (p, a, WM[a][p])))
            sec[(p, a)] = rjson(fn[0])
    for a in ARMS:
        r = [sec[(p, a)] for p in PROPS]
        u = [sec[(p, "unguided")] for p in PROPS]
        out("| %s | %s | %s | %s | %s | %s | %s |" % (
            "**tfg**" if a == "tfg" else a,
            " / ".join("%g" % WM[a][p] for p in PROPS),
            " / ".join(f(x["in_band_fraction"]) for x in r),
            " / ".join("%.2fx" % (x["in_band_fraction"] / y["in_band_fraction"]) for x, y in zip(r, u)),
            " / ".join(f(x["mol_stability"]) for x in r),
            " / ".join("%.0f%%" % (100 * x["mol_stability"] / y["mol_stability"]) for x, y in zip(r, u)),
            " / ".join("yes" if x["mol_stability"] >= 0.9 * y["mol_stability"] - 1e-12 else "**no**"
                       for x, y in zip(r, u))))
    t = [sec[(p, "tfg")] for p in PROPS]
    out("")
    out("tfg on decoded atom types (the molecules that exist): in-band %s, against %s "
        "on the continuous features. Floor here = 90%% of this seed's own unguided." % (
            " / ".join(f(x["in_band_fraction_dec"]) for x in t),
            " / ".join(f(x["in_band_fraction"]) for x in t)))
    out("")

    # 3. ladder
    out("## 3. Reference ladder (published units)")
    out("")
    out("| reference | mu (D) | alpha (Bohr^3) | gap (meV) | meaning |")
    out("|---|---|---|---|---|")
    for k, name, mean in (("u_bound", "U-bound (random molecule)", "ignores size and target"),
                          ("size_shuffle", "size-shuffle (ours)", "right size, target ignored: gap closure 0"),
                          ("atoms_median", "#Atoms (EDM's bar)", "best guess from size alone"),
                          ("l_bound", "L-bound (the real molecule)", "gap closure 1")):
        out("| %s | %s | %s | %s | %s |" % (name, *(
            f(phys(p, rep[p]["ladder"][k]["mae"]), DEC[p]) for p in PROPS), mean))
    best = {p: min(ARMS[1:], key=lambda a: A[p][a]["mae"]) for p in PROPS}
    out("| best arm (%s) | %s | %s | %s | beats #Atoms on %s |" % (
        "/".join(sorted(set(best.values()))),
        *(f(A[p][best[p]]["mae_published_units"], DEC[p]) for p in PROPS),
        ", ".join(p for p in PROPS if A[p][best[p]]["beats_atoms"]) or "none"))
    out("| EDM, trained conditional | 1.111 | 2.76 | 655 | published (Hoogeboom et al. 2022) |")
    out("")

    # 4. transfer
    edm = load_store(os.path.join(ROOT, "results", "transfer", "compare", "tr__*.json"))
    ours = load_store(os.path.join(ROOT, "results", "sweep", "*__cmp.json"))
    out("## 4. EDMsecond transfer: compare stage only "
        "(TFG's generator, guide and oracle; n = 512, seed 20260922, q90)")
    out("")
    out("Ran on Betty on 23 Sep before tfg joined the transfer set, so tfg is absent; the "
        "full stage never ran (the transfer moved to EquiFM the same evening). "
        "Same FR3a rule as the main run: best q90 MAE among strengths clearing 90% "
        "of unguided's mol stability.")
    out("")
    out("| arm | w (mu/alpha/gap) | in-band mu / alpha / gap | in-band x unguided | MAE cut "
        "| mol stab mu / alpha / gap | strengths clearing floor (of 7) |")
    out("|---|---|---|---|---|---|---|")
    for a in TARMS:
        rr = [fr3a(edm, p, a) for p in PROPS]
        uu = [edm[(p, "unguided", "q90", 1.0)] for p in PROPS]
        out("| %s | %s | %s | %s | %s | %s | %s |" % (
            a, " / ".join("%g" % r["w"] for r, _ in rr),
            " / ".join(f(r["in_band_fraction"]) for r, _ in rr),
            " / ".join("%.2fx" % (r["in_band_fraction"] / u["in_band_fraction"])
                       if u["in_band_fraction"] else "n/a" for (r, _), u in zip(rr, uu)),
            " / ".join(pct(1 - r["prop_mae_eval"] / u["prop_mae_eval"])
                       for (r, _), u in zip(rr, uu)),
            " / ".join(f(r["mol_stability"]) for r, _ in rr),
            "-" if a == "unguided" else " / ".join(str(n) for _, n in rr)))
    ue = edm[("mu", "unguided", "q90", 1.0)]
    n_hits = [round(edm[(p, "unguided", "q90", 1.0)]["in_band_fraction"] * 512) for p in PROPS]
    out("")
    out("Unguided on EDMsecond: mol stability %s, validity %s (ours: 0.402 / 0.785). "
        "Unguided in-band counts are %s of 512 molecules, so in-band ratios on alpha "
        "are noise; MAE cut is the steadier column." % (
            f(ue["mol_stability"]), f(ue["validity"]), " / ".join(map(str, n_hits))))
    out("")
    out("### 4b. Same rule on both generators (q90, n = 512): in-band x unguided, rank in brackets")
    out("")
    out("| arm | ours mu | EDM mu | ours alpha | EDM alpha | ours gap | EDM gap |")
    out("|---|---|---|---|---|---|---|")
    ratio = {}
    for store, tag in ((ours, "ours"), (edm, "edm")):
        for p in PROPS:
            u = store[(p, "unguided", "q90", 1.0)]["in_band_fraction"]
            vals = {a: fr3a(store, p, a)[0]["in_band_fraction"] / u for a in TARMS[1:]}
            order = sorted(vals, key=lambda a: -vals[a])
            for a in vals:
                ratio[(tag, p, a)] = (vals[a], order.index(a) + 1)
    for a in TARMS[1:]:
        out("| %s | %s |" % (a, " | ".join("%.2fx (%d)" % ratio[(tag, p, a)]
                                            for p in PROPS for tag in ("ours", "edm"))))
    out("")
    out("Ties are broken by listing order. At n = 512 the difference between two arms "
        "has 2 se of about 0.024 in-band at p ~ 0.04, so most rank gaps here are within "
        "noise; the clear exception is lgd_mc on our gap.")
    out("")

    # 5-9: does guidance work? (results_evidence.py)
    ev.sections(out, ALL, W, WM, ours)

    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (OUT, len(L)))


def _dec_in_band(p, w):
    """Seed-mean decoded in-band for tfg at strength w (the dist report has no _dec)."""
    vals = []
    for fn in glob.glob(os.path.join(ROOT, "results", "full", "n5000", "seed*",
                                     "%s__tfg__dist__w%g__tmin0.5__full.json" % (p, w))):
        vals.append(rjson(fn)["in_band_fraction_dec"])
    return sum(vals) / len(vals)


if __name__ == "__main__":
    main()
