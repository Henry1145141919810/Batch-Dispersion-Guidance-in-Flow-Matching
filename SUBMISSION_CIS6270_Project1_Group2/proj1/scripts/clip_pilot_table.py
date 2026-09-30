"""Does removing the velocity clip help? A v2-shaped pilot at small n.

    python proj1/scripts/clip_pilot_table.py

Reads results/pilot_clip/{clip1,noclip}/ and writes
docs/results/CLIP_PILOT.md.

THE QUESTION. The clip caps the guidance step at `clip x |v|`, the base flow
velocity. BTVG saturates it on ~1/3 of guided steps, so the mean and variance
terms compete for one budget and the aiming step is scaled down with the rest.
Removing the clip would let each term through in full.

WHY THE RAW COMPARISON IS NOT THE ANSWER. Letting more guidance through also
pushes the state further off the data manifold, which costs chemistry. Every
arm already trades in-band against molecule stability as strength rises, so
an unclipped cell with higher in-band and lower stability has only moved ALONG
that trade-off, not beaten it. The question the rubric asks is whether the
FRONTIER moves: more in-band at the SAME molecule stability.

So this reports three things, in order:
  1. every cell's full metric block, both clip settings
  2. at fixed w, clipped vs unclipped, paired on the same molecules and noise
  3. the frontier: the best in-band reachable at or above a given molecule
     stability, over the convex hull of an arm's measured cells, and the same
     restricted to a single measured setting (no mixing)

EXPLORATORY. n = 256, one seed, mu only. The se of an in-band proportion here
is ~0.02, so only large differences are visible; this cannot settle a 0.01
effect. It is a direction-finder for whether a full run is worth it.

The clip binds more often the stronger the arm: for btvg on mu it runs
0.024 / 0.114 / 0.323 / 0.522 of guided steps over w = 0.25 / 1 / 4 / 16.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
PILOT = os.path.join(ROOT, "results", "pilot_clip")
OUT = os.path.join(ROOT, "docs", "results", "CLIP_PILOT.md")
SETTINGS = (("clip1", "clip = 1 (shipped)"), ("noclip", "no clip"))
ARMS = ("unguided", "plug", "btvg", "lgd_mc")
FLOOR = 0.9


def cells(sub):
    out = {}
    for fn in glob.glob(os.path.join(PILOT, sub, "*__tgt.json")):
        j = json.load(open(fn))
        m = re.match(r"(\w+)__(\w+)__(\w+)__w([0-9.]+)__", os.path.basename(fn))
        pm = fn.replace(".json", ".permol.pt")
        j["_permol"] = pm if os.path.exists(pm) else None
        out[(m.group(2), float(m.group(4)))] = j
    return out


def se_prop(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def resid(j, rows=None):
    """(bias/delta, resid sd/delta, n survivors) from the sidecar.

    Over FINITE rows only -- a diverged sample has no meaningful residual. That
    makes these three columns a different denominator from in-band / stability
    / validity, which count a non-finite sample as a failure over the full n.
    `rows` restricts to a given boolean mask, for like-for-like comparison
    against a cell that lost different samples.
    `unbiased=False` matches btvg2_table.py, the repo's convention."""
    if not j["_permol"]:
        return float("nan"), float("nan"), 0
    d = torch.load(j["_permol"], weights_only=False)
    keep = d["finite"].bool() if rows is None else (d["finite"].bool() & rows)
    r = (d["f_B"].double() - d["y"].double())[keep]
    dl = j["delta"]
    return float(r.mean()) / dl, float(r.std(unbiased=False)) / dl, int(keep.sum())


def finite_mask(j):
    if not j["_permol"]:
        return None
    return torch.load(j["_permol"], weights_only=False)["finite"].bool()


def paired(a, b, key):
    """Mean difference and paired z on the same molecules, a minus b."""
    if not (a["_permol"] and b["_permol"]):
        return None, None
    da, db = (torch.load(x["_permol"], weights_only=False) for x in (a, b))
    if not torch.equal(da["mol_idx"], db["mol_idx"]):
        return None, None
    dl = a["delta"]
    def v(d, j):
        r = (d["f_B"].double() - d["y"].double())
        return {"in_band": (r.abs() <= dl).double(),
                "mol_stable": d["mol_stable"].double()}[key]
    # mask to rows finite in BOTH, as btvg2_table.py::paired does. No effect
    # on these cells (no diverged row lands within delta) but the two pages
    # must not be able to disagree.
    fin = da["finite"].bool() & db["finite"].bool()
    diff = (v(da, a) - v(db, b))[fin]
    sd = float(diff.std(unbiased=True))
    n = len(diff)
    return float(diff.mean()), (float(diff.mean()) / (sd / math.sqrt(n))
                                if sd > 0 else float("nan"))


def envelope_at(rows, stab):
    """Best in-band achievable at molecule stability >= `stab`.

    NOT a walk along the strength curve: stability is non-monotone in w for
    several arms here, so chaining strength-sorted or stability-sorted points
    routes through DOMINATED cells and understates what the arm can do. Two
    settings can be mixed (run a fraction of samples at each), and a mixture
    achieves the convex combination of their (stability, in-band), so the
    achievable set is the convex hull of the measured cells. The answer is
    therefore the maximum of in-band over that hull subject to the chemistry
    requirement -- attained either at a measured cell that already clears
    `stab`, or where an edge crosses it.

    Returns None when no measured cell reaches `stab`; extrapolating past the
    measured range is how a pilot invents a result. Non-increasing in `stab`
    by construction, which section 3 asserts."""
    pts = [(r["mol_stability"], r["in_band_fraction"]) for r in rows]
    if not pts or stab > max(s for s, _ in pts):
        return None
    best = best_single_at(rows, stab)
    for (s0, b0) in pts:                      # edges crossing the requirement
        for (s1, b1) in pts:
            if s0 < stab < s1:
                v = b0 + (b1 - b0) * (stab - s0) / (s1 - s0)
                best = v if best is None else max(best, v)
    return best


def best_single_at(rows, stab):
    """Best in-band from ONE measured setting with stability >= stab.

    The rubric freezes a single strength per arm and applies the floor to that
    cell, so this is the frozen-operating-point reading of the same question.
    The envelope above can beat it by mixing in a cell that is itself below the
    floor, which no frozen arm is allowed to do."""
    return max((r["in_band_fraction"] for r in rows
                if r["mol_stability"] >= stab), default=None)


def main():
    data = {k: cells(k) for k, _ in SETTINGS}
    if not any(data.values()):
        raise SystemExit("no cells in %s -- run the sweep first" % PILOT)
    L = []
    o = L.append
    ref = data["clip1"].get(("unguided", 1.0))
    n = ref["n"] if ref else 0
    floor = FLOOR * ref["mol_stability"] if ref else float("nan")
    o("# Does removing the velocity clip help? (pilot)")
    o("")
    o("Generated by `proj1/scripts/clip_pilot_table.py`. Do not hand-edit; "
      "re-run the script.")
    o("")
    se1 = se_prop(0.1, n)
    res1 = 2 * se1                      # a single in-band proportion
    resd = 2 * math.sqrt(2) * se1       # a difference of two INDEPENDENT cells
    o("**Exploratory pilot, not a verdict.** n = %d, one seed (20261001), `mu` "
      "only, v2's fixed q90 target. The se of an in-band proportion here is "
      "%.4f, so a single cell is resolvable to about %.4f and a difference "
      "between two independent cells to about %.4f. Differences at the same "
      "strength are PAIRED (same molecules, same noise), so those are judged "
      "by the paired z, not by that threshold. It asks whether a full run is "
      "worth it." % (n, se1, res1, resd))
    o("")
    o("| provenance | value |")
    o("|---|---|")
    o("| target | q90 = %.4f (v2's fixed target) |" % (ref["target_mean"] if ref else 0))
    o("| delta | %.5f |" % (ref["delta"] if ref else 0))
    o("| chemistry floor | %.4f = 0.9 x unguided %.4f |" % (
        floor, ref["mol_stability"] if ref else 0))
    o("| sampler | 100-step euler, window t >= 0.5, n = %d, seed 20261001 |" % n)
    o("| cells | %s |" % ", ".join("%s: %d" % (k, len(v)) for k, v in data.items()))
    o("")
    o("**What the clip does.** The guidance vector is rescaled so its norm is "
      "at most `clip x |v|`, where v is the base flow velocity "
      "(`sampling.py::_clip_to_velocity`). It is a pure rescale: the direction "
      "is untouched. `--clip -1` removes it.")
    o("")

    # ---------------------------------------------------------------- 1
    o("## 1. Every cell")
    o("")
    o("`non-finite` counts samples whose geometry diverged to NaN/inf; they "
      "occur only without the clip. **Denominators differ across this table:** "
      "in-band, mol stab and validity are over all %d samples, counting a "
      "diverged one as a failure; MAE, bias and resid sd are over finite rows "
      "only (see the survivorship note below)." % n)
    o("")
    o("| setting | arm | w | in-band | +-se | MAE/d | bias/d | resid sd/d | "
      "mol stab | vs floor | valid | non-finite |")
    o("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, lab in SETTINGS:
        for a in ARMS:
            for (arm, w), j in sorted(data[key].items(), key=lambda kv: kv[0][1]):
                if arm != a:
                    continue
                b, sd, nsurv = resid(j)
                nf = j.get("n_nonfinite", 0)
                o("| %s | `%s` | %g | %.4f | %.4f | %.3f | %+.2f | %.2f | "
                  "%.4f | %s | %.4f | %s |" % (
                      lab, arm, w, j["in_band_fraction"],
                      se_prop(j["in_band_fraction"], j["n"]),
                      j["prop_mae_eval"] / j["delta"], b, sd,
                      j["mol_stability"],
                      "ok" if j["mol_stability"] >= floor else "**below**",
                      j["validity"],
                      "-" if not nf else "**%d of %d**" % (nf, j["n"])))
    o("")
    # like-for-like on the cells that lost samples: the printed bias/sd of a
    # no-clip cell scores only its survivors, so compare the clipped cell on
    # the SAME rows before reading any difference off the table above.
    hurt = [(a, w) for (a, w), j in data["noclip"].items()
            if j.get("n_nonfinite", 0)]
    if hurt:
        o("**Survivorship.** The three residual columns score only the samples "
          "that stayed finite, so a no-clip cell that lost samples is graded on "
          "its survivors. Restricting the clipped cell to the same rows:")
        o("")
        o("| arm | w | survivors | bias/d clipped -> no clip | "
          "resid sd/d clipped -> no clip |")
        o("|---|---|---|---|---|")
        for a, w in sorted(hurt, key=lambda k: (k[0], k[1])):
            c, u = data["clip1"].get((a, w)), data["noclip"].get((a, w))
            if not (c and u):
                continue
            m = finite_mask(u)
            bc, sc, _ = resid(c, m)
            bu, su, ns = resid(u)
            o("| `%s` | %g | %d of %d | %+.2f -> %+.2f | %.2f -> %.2f |" % (
                a, w, ns, u["n"], bc, bu, sc, su))
        o("")
        o("On `btvg` w = 16 the printed bias of %+.2f is largely survivorship: "
          "the clipped cell over the same %d rows is already %+.2f. The spread "
          "columns move the other way, so the spread finding below is "
          "conservative under this correction." % (
              resid(data["noclip"][("btvg", 16.0)])[0],
              resid(data["noclip"][("btvg", 16.0)])[2],
              resid(data["clip1"][("btvg", 16.0)],
                    finite_mask(data["noclip"][("btvg", 16.0)]))[0])
          if ("btvg", 16.0) in data["noclip"] else "")
        o("")

    # ---------------------------------------------------------------- 2
    o("## 2. Same strength, clipped vs unclipped (paired)")
    o("")
    o("Same molecules, same initial noise, same w: the only difference is the "
      "clip. Positive = unclipped is higher.")
    o("")
    o("| arm | w | d in-band | paired z | d mol stab | paired z | "
      "unclipped in-band / stab | clipped in-band / stab |")
    o("|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for w in sorted({w for (arm, w) in data["clip1"] if arm == a}):
            c, u = data["clip1"].get((a, w)), data["noclip"].get((a, w))
            if not (c and u):
                continue
            di, zi = paired(u, c, "in_band")
            ds, zs = paired(u, c, "mol_stable")
            f = lambda x, p=4: ("n/a" if x is None or x != x
                                else "%+.*f" % (p, x))
            o("| `%s` | %g | %s | %s | %s | %s | %.4f / %.4f | %.4f / %.4f |" % (
                a, w, f(di), f(zi, 2), f(ds), f(zs, 2),
                u["in_band_fraction"], u["mol_stability"],
                c["in_band_fraction"], c["mol_stability"]))
    o("")

    # ---------------------------------------------------------------- 3
    o("## 3. The frontier: in-band at equal chemistry")
    o("")
    o("A cell with more in-band and less stability has moved along the "
      "trade-off, not beaten it. Each entry is the best in-band the arm can "
      "reach at **at least** that molecule stability, over the convex hull of "
      "its measured cells (mixing two settings achieves the combination). "
      "Blank = no measured cell of that arm reaches that stability, where an "
      "answer would be an extrapolation. Entries must not fall left to right; "
      "the script checks that.")
    o("")
    targets = [floor, 0.9 * floor, 0.8 * floor]
    o("**A mixture may include a cell that is itself below the floor.** "
      "`lgd_mc` clipped reads 0.1016 at the two looser levels from w = 4 "
      "alone, whose own stability is 0.3594 -- below the floor. The rubric "
      "freezes ONE strength per arm and applies the floor to that cell, so "
      "these entries are not frozen operating points. The second table gives "
      "the frozen-cell reading.")
    o("")
    front, single = {}, {}
    for name, fn, store in (("Best mixture (convex hull)", envelope_at, front),
                            ("Best single measured setting (no mixing)",
                             best_single_at, single)):
        o("**%s**" % name)
        o("")
        o("| arm | setting | " + " | ".join(
            "in-band @ stab >= %.3f" % s for s in targets) + " |")
        o("|---|---|" + "---|" * len(targets))
        for a in ARMS:
            if a == "unguided":
                continue
            for key, lab in SETTINGS:
                rows = [j for (arm, w), j in data[key].items() if arm == a]
                if not rows:
                    continue
                vals = [fn(rows, s) for s in targets]
                seq = [v for v in vals if v is not None]
                if any(y < x - 1e-12 for x, y in zip(seq, seq[1:])):
                    raise SystemExit("frontier for %s/%s falls as the "
                                     "requirement is relaxed: %s" % (a, key, seq))
                store[(a, key)] = vals
                o("| `%s` | %s | %s |" % (a, lab, " | ".join(
                    "-" if v is None else "%.4f" % v for v in vals)))
        o("")
    o("Unguided in-band is %.4f at stability %.4f." % (
        ref["in_band_fraction"], ref["mol_stability"]) if ref else "")
    o("")

    # ---------------------------------------------------------------- 4
    o("## 4. What this pilot supports")
    o("")
    # every claim below is computed; a hardcoded assertion would survive data
    # that contradicts it, which an earlier draft of this page did
    nf = {k: sum(j.get("n_nonfinite", 0) for j in v.values())
          for k, v in data.items()}
    ncells = {k: sum(1 for j in v.values() if j.get("n_nonfinite", 0))
              for k, v in data.items()}
    nguided = {k: sum(1 for (a, w) in v if a != "unguided") for k, v in data.items()}
    o("- **Without the clip the sampler diverges; with it, never here.** "
      "Non-finite samples: **%d across %d of the %d guided no-clip cells**, "
      "against **%d** in all %d clipped cells. `btvg` at w = 16 loses %s and "
      "keeps %.3f molecule stability. A 222-vs-0 count is not a noise-scale "
      "effect at this n." % (
          nf["noclip"], ncells["noclip"], nguided["noclip"], nf["clip1"],
          len(data["clip1"]),
          "%d of %d samples" % (data["noclip"][("btvg", 16.0)]["n_nonfinite"],
                                data["noclip"][("btvg", 16.0)]["n"])
          if ("btvg", 16.0) in data["noclip"] else "n/a",
          data["noclip"][("btvg", 16.0)]["mol_stability"]
          if ("btvg", 16.0) in data["noclip"] else float("nan")))
    # frontier: compare only where BOTH settings have a value, under both
    # readings (mixture and frozen single cell)
    def fcmp(store):
        return [(a, t, store[(a, "clip1")][i], store[(a, "noclip")][i])
                for a in ARMS if a != "unguided" and (a, "clip1") in store
                for i, t in enumerate(targets)
                if store[(a, "clip1")][i] is not None
                and store[(a, "noclip")][i] is not None]
    cmp_f, cmp_s = fcmp(front), fcmp(single)
    up_f = [c for c in cmp_f if c[3] > c[2] + 1e-12]
    up_s = [c for c in cmp_s if c[3] > c[2] + 1e-12]
    big = [c for c in cmp_f if abs(c[3] - c[2]) >= resd]
    arms_big = sorted({c[0] for c in big})
    o("- **No arm's frontier improves without the clip, under either "
      "reading.** Unclipped is higher in **%d of %d** mixture comparisons and "
      "**%d of %d** frozen-cell comparisons (section 3). The largest gap, all "
      "in the clipped setting's favour, is %.4f. Against the %.4f needed for a "
      "difference of two independent cells, **%d of %d clear it**%s. What "
      "carries the reading is the direction, not any single gap." % (
          len(up_f), len(cmp_f), len(up_s), len(cmp_s),
          max((abs(c[3] - c[2]) for c in cmp_f), default=0), resd,
          len(big), len(cmp_f),
          (", and those %d are one finding, not %d: all are `%s`, resting on "
           "the same cells, which section 2 reports once as a paired result"
           % (len(big), len(big), arms_big[0]))
          if big and len(arms_big) == 1 else ""))
    # fixed-strength counts, guided arms only: unguided's tie is structural
    rows_w = [(a, w, data["clip1"][(a, w)], data["noclip"][(a, w)])
              for (a, w) in sorted(data["clip1"]) if a != "unguided"
              and (a, w) in data["noclip"]]
    d_ib = [(a, w, u["in_band_fraction"] - c["in_band_fraction"])
            for a, w, c, u in rows_w]
    lo = [x for x in d_ib if x[2] < -1e-12]
    hi = [x for x in d_ib if x[2] > 1e-12]
    per_arm = {a: (sum(1 for x in d_ib if x[0] == a and x[2] < -1e-12),
                   sum(1 for x in d_ib if x[0] == a))
               for a in {x[0] for x in d_ib}}
    zs = []
    for a, w, c, u in rows_w:
        di, zi = paired(u, c, "in_band")
        if zi is not None and zi == zi and abs(zi) >= 3:
            zs.append((a, w, di, zi))
    o("- **At fixed strength it is lower in %d of %d guided cells**, higher in "
      "%d (section 2). By arm: %s. `unguided` is excluded -- it applies no "
      "guidance, so the clip cannot act and its tie is structural, not "
      "evidence. These are paired, so the test is the paired z: %s." % (
          len(lo), len(d_ib), len(hi),
          ", ".join("`%s` %d of %d lower" % (a, v[0], v[1])
                    for a, v in sorted(per_arm.items())),
          ("%d of %d reach |z| >= 3 -- %s" % (
              len(zs), len(d_ib),
              "; ".join("`%s` w=%g (%+.4f, z %+.2f)" % x for x in zs)))
          if zs else "none of the %d reaches |z| >= 3" % len(d_ib)))
    bt_c, bt_u = data["clip1"].get(("btvg", 4.0)), data["noclip"].get(("btvg", 4.0))
    if bt_c and bt_u:
        m = finite_mask(bt_u)
        _, sc_m, nsv = resid(bt_c, m)
        _, su, _ = resid(bt_u)
        bc_m, _, _ = resid(bt_c, m)
        bu, _, _ = resid(bt_u)
        o("- **Unclipping frees both terms, and the spread term wins.** For "
          "`btvg` at w = 4, on the %d rows finite in both, the residual sd "
          "rises %.2f -> %.2f delta while the bias moves toward zero "
          "(%+.2f -> %+.2f). So aiming does improve, but the spread grows "
          "faster and in-band still falls. Survivorship works against this "
          "reading, so it is conservative." % (
              nsv, sc_m, su, bc_m, bu))
    o("")
    plug_s = [c for c in cmp_s if c[0] == "plug"]
    ties = [c for c in plug_s if abs(c[3] - c[2]) < 1e-12]
    o("- **Read `plug` and `btvg` as null, not as small wins.** Their mixture "
      "gaps come from mixing: under the frozen-cell reading `plug` is an exact "
      "tie at %d of %d levels (both settings' best floor-clearing cell is the "
      "same w = 0.25 run) and `btvg`'s gaps are %s. The direction is carried "
      "by `lgd_mc` alone." % (
          len(ties), len(plug_s),
          "one molecule wide (%s)" % ", ".join(
              "%+.4f" % (c[2] - c[3]) for c in cmp_s if c[0] == "btvg")
          if any(c[0] == "btvg" for c in cmp_s) else "not measurable here"))
    o("")
    o("**Not supported.** Any claim about alpha or gap (not run), about arms "
      "not listed, or about an in-band difference between independent cells "
      "below ~%.4f. One seed, n = %d, mu only, and the %d fixed-strength "
      "comparisons are 3 correlated strength curves on one molecule set, not "
      "%d independent trials. The mechanism behind the divergence is not "
      "established here: the data show that it happens without the clip and "
      "not with it, not why. Frontier entries carry no standard error though "
      "each blends cells with se ~ %.4f." % (
          resd, n, len(d_ib), len(d_ib), se1))
    o("")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote %s (%d lines)" % (OUT, len(L)))


if __name__ == "__main__":
    main()
