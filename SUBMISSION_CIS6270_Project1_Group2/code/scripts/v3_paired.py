"""Paired per-molecule reading of a protocol-v3 run: useful yield, tested paired.

    python proj1/scripts/v3_paired.py --n 5000 --layout flat --backends fm,equifm \
        --md-out docs/results/V3_BETTY_PAIRED.md

WHY THIS EXISTS. `v3_table.py` reports UNPAIRED z values and says so. Within one
(backend, property, seed) every arm starts from the same initial noise and the
same molecule sizes, so molecule i of one arm is molecule i of every other, and
the unpaired se overstates the noise. BDG_REVIEW.md ruled that a BDG comparison
"must be recomputed paired". This reads the `*.permol.pt` sidecars and does that.

THE METRIC. v3 has no chemistry floor, so in-band alone crowns whichever arm
destroyed the most chemistry. The reported metric is this project's own
**useful yield** (SCOPE_FM_GUIDANCE_STATUS.md): **decoded in-band AND
molecule-stable, per attempt**, with an RDKit-validity cross-check that must not
contradict it. Useful yield does NOT make an arm immune to spending chemistry --
an arm can still buy yield by pushing hard -- it *charges* for it, because a
molecule that stops being stable leaves the numerator. Three companion columns
are printed so a verdict cannot rest on one scoring choice: the continuous
(undecoded) yield, the same yield under the SECOND oracle (OC-Flow, its own
delta -- a re-measurement in a different band, not a reproduction), and the
strict yield that also requires validity.

THE DECISION BAR IS THE PRE-REGISTERED ONE. FULL_RUN_V3_PROTOCOL.md section 6
and V3_POWER.md pre-register Bonferroni alpha = 0.05 / 18 contrasts, two-sided:
**z = 2.99**. That is what verdicts are called at here. A 3-sigma family rate
over the arms present gives 3.55 instead; it is printed for reference and
labelled, because changing the bar after the data landed is exactly the move
this file exists to make visible.

McNEMAR, WITH THE CONTINUITY CORRECTION. z = (|b10 - b01| - 1)/sqrt(b10 + b01),
signed. Three baseline verdicts sit inside the correction, so the uncorrected
form is printed beside it rather than chosen silently.

WHAT IT VERIFIES BEFORE COMPUTING ANYTHING. Every sidecar must reproduce its
cell's in-band (continuous and decoded), second-oracle in-band, molecule
stability and validity to --tol; every arm in a (backend, property, seed) must
share `mol_idx` and `n_atoms` elementwise; every arm must have exactly one cell
per seed at one strength and one stage. Otherwise it refuses.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import v3_table as V                                   # noqa: E402

NL = chr(10)
PROPS = V.PROPS
BASE_ORDER = ("unguided", "plug", "tmpd", "lgd_mc", "tfg")
# Pre-registered in FULL_RUN_V3_PROTOCOL.md section 6 / V3_POWER.md: Bonferroni
# alpha = 0.05 over 18 contrasts, two-sided. Verdicts are called at this bar.
PREREG_ALPHA, PREREG_CONTRASTS = 0.05, 18
# every cell pooled here must agree on these, as in v3_table.SAME_KEYS
PAIR_KEYS = ("n", "steps", "solver", "batch", "target_name", "w", "t_start",
             "target", "delta")


def prereg_bar():
    return V.NORM_Q(1.0 - (PREREG_ALPHA / PREREG_CONTRASTS) / 2.0)


def cell_paths(root, backend, n, seed, stage, layout):
    d = (os.path.join(root, backend, "n%d" % n, "seed%s" % seed)
         if layout == "flat"
         else os.path.join(root, backend, stage, "n%d" % n, "seed%s" % seed))
    return sorted(glob.glob(os.path.join(d, "tr__*.json")))


def load_cell(js_path, tol, stage, w):
    """(arm, prop, dict) or (None, None, problem-string)."""
    r = json.load(open(js_path))
    base = os.path.basename(js_path)
    if r.get("stage") != stage:
        return None, None, "%s is a %r cell, not %r" % (base, r.get("stage"), stage)
    if w is not None and abs(float(r.get("w", 1.0)) - float(w)) > 1e-12:
        return None, None, None                        # a strength we did not ask for
    pm_path = js_path[:-len(".json")] + ".permol.pt"
    if not os.path.exists(pm_path):
        return None, None, "%s: no per-molecule sidecar" % base
    x = torch.load(pm_path, map_location="cpu", weights_only=False)
    tgt, delta = float(r["target"]), float(r["delta"])
    finite = x["finite"].bool()
    hit = ((x["f_B"] - tgt).abs() <= delta) & finite
    hit_dec = ((x["f_B_dec"] - tgt).abs() <= delta) & finite
    stable, valid = x["mol_stable"].bool(), x["valid"].bool()
    for key, t in (("in_band_fraction", hit), ("in_band_fraction_dec", hit_dec),
                   ("mol_stability", stable), ("validity", valid)):
        if key in r and abs(float(t.float().mean()) - float(r[key])) > tol:
            return None, None, ("%s: sidecar %s %.6f != cell %.6f"
                                % (base, key, float(t.float().mean()), float(r[key])))
    out = {"hit": hit, "hit_dec": hit_dec, "stable": stable, "valid": valid,
           "hs": hit & stable, "hs_dec": hit_dec & stable,
           "hsv_dec": hit_dec & stable & valid,
           "mol_idx": x.get("mol_idx"), "n_atoms": x.get("n_atoms"),
           "seconds": float(r.get("seconds") or 0.0),
           "keys": {k: r.get(k) for k in PAIR_KEYS},
           "rmse_over_mae": (float(r["prop_rmse_eval"]) / float(r["prop_mae_eval"])
                             if r.get("prop_mae_eval") else float("nan")),
           "max_dev_over_delta": float((x["f_B"] - tgt).abs().max()) / delta}
    o2 = r.get("oracle2") or {}
    if "f_B2_dec" in x and o2.get("delta") is not None:
        d2 = float(o2["delta"])
        out["hs_o2_dec"] = (((x["f_B2_dec"] - tgt).abs() <= d2) & finite & stable)
        if o2.get("in_band_dec") is not None:
            got = float((((x["f_B2_dec"] - tgt).abs() <= d2) & finite).float().mean())
            if abs(got - float(o2["in_band_dec"])) > tol:
                return None, None, ("%s: sidecar oracle2 in_band_dec %.6f != cell %.6f"
                                    % (base, got, float(o2["in_band_dec"])))
    return r["arm"], r["prop"], out


def mcnemar(a, b, continuity=True):
    """Signed McNemar z over the SAME molecules, plus the discordant counts.

    b10 = a hit where b missed. With the continuity correction the statistic is
    (|b10 - b01| - 1)/sqrt(b10 + b01), carrying the sign of (b10 - b01) and
    floored at 0. Concordant pairs carry no information about the difference.
    """
    b10 = int((a & ~b).sum())
    b01 = int((b & ~a).sum())
    n_d = b10 + b01
    if n_d == 0:
        return float("nan"), b10, b01
    diff = b10 - b01
    num = (max(abs(diff) - 1.0, 0.0) if continuity else abs(diff))
    return math.copysign(num / math.sqrt(n_d), diff or 1.0), b10, b01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "v3"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seeds", default="20261001,20261002,20261003")
    ap.add_argument("--stage", default="v3", choices=["v3", "v3abl"])
    ap.add_argument("--layout", default="staged", choices=["staged", "flat"])
    ap.add_argument("--backends", default="fm")
    ap.add_argument("--w", type=float, default=None,
                    help="which strength to read. Required when the tree holds "
                         "more than one, as the ablation's does.")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()
    seeds = [s for s in args.seeds.split(",") if s]
    backends = [b for b in args.backends.split(",") if b]
    ZBAR = prereg_bar()

    data, problems, cost, flags = {}, [], {}, []
    for be in backends:
        per_prop, seen_w, counts = {}, set(), {}
        for seed in seeds:
            paths = cell_paths(args.root, be, args.n, seed, args.stage, args.layout)
            if not paths:
                problems.append("%s: no cells for seed %s" % (be, seed))
                continue
            group = {}
            for p in paths:
                try:
                    r = json.load(open(p))
                except Exception as exc:               # noqa: BLE001
                    problems.append("%s: %s unreadable (%s)"
                                    % (be, os.path.basename(p), exc))
                    continue
                seen_w.add(float(r.get("w", 1.0)))
                arm, prop, got = load_cell(p, args.tol, args.stage, args.w)
                if arm is None:
                    if got:
                        problems.append("%s: %s" % (be, got))
                    continue
                if prop in group and arm in group[prop]:
                    problems.append("%s/%s/%s: two cells for arm %s -- refusing "
                                    "rather than letting one overwrite the other"
                                    % (be, prop, seed, arm))
                    continue
                group.setdefault(prop, {})[arm] = got
                counts[(prop, arm)] = counts.get((prop, arm), 0) + 1
            for prop, arms in group.items():
                ref_arm = sorted(arms)[0]
                ref = arms[ref_arm]
                for a, g in sorted(arms.items()):
                    for key in ("mol_idx", "n_atoms"):
                        if ref[key] is None or g[key] is None:
                            problems.append("%s/%s/%s: %s has no %s, cannot pair"
                                            % (be, prop, seed, a, key))
                        elif not torch.equal(ref[key], g[key]):
                            problems.append("%s/%s/%s: %s and %s disagree on %s -- "
                                            "NOT paired" % (be, prop, seed, ref_arm,
                                                            a, key))
                    for k, v in (g["keys"] or {}).items():
                        if ref["keys"].get(k) != v:
                            problems.append("%s/%s/%s: %s and %s disagree on %s "
                                            "(%r vs %r)" % (be, prop, seed, ref_arm,
                                                            a, k, ref["keys"].get(k), v))
                    if g["max_dev_over_delta"] > 1e3:
                        flags.append("`%s`/%s/%s seed %s: one sample lands %.0fx delta "
                                     "from target (RMSE/MAE %.1f) -- its MAE, bias and "
                                     "sd are not meaningful; threshold metrics are "
                                     "unaffected" % (be, prop, a, seed,
                                                     g["max_dev_over_delta"],
                                                     g["rmse_over_mae"]))
                for a, g in arms.items():
                    slot = per_prop.setdefault(prop, {}).setdefault(a, {})
                    for k, v in g.items():
                        if torch.is_tensor(v) and v.dtype == torch.bool:
                            slot[k] = torch.cat([slot[k], v]) if k in slot else v
                    cost[be] = cost.get(be, 0.0) + g["seconds"]
        if args.w is None and len(seen_w) > 1:
            problems.append("%s: this tree holds %d strengths (%s); pass --w"
                            % (be, len(seen_w),
                               ", ".join("%g" % x for x in sorted(seen_w))))
        for (prop, arm), c in sorted(counts.items()):
            if c != len(seeds):
                problems.append("%s/%s/%s: %d of %d seeds"
                                % (be, prop, arm, c, len(seeds)))
        data[be] = per_prop
    if problems:
        print("REFUSING: the paired reading is not safe.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)

    L = ["# Protocol v3 -- paired per-molecule reading (useful yield)", ""]
    L.append("Generated by `proj1/scripts/v3_paired.py`. Do not hand-edit; re-run it.")
    L.append("")
    L.append("Companion to [V3_BETTY_HEADLINE.md](V3_BETTY_HEADLINE.md). Same "
             "cells, read per molecule.")
    L.append("")
    L.append("> **The metric is useful yield: decoded in-band AND "
             "molecule-stable, per attempt** -- this project's own definition. "
             "It does **not** make an arm immune to spending chemistry: an arm "
             "can still buy yield by pushing harder. It *charges* for it, "
             "because a molecule that stops being stable leaves the numerator. "
             "Read it beside the stability column, never instead of it.")
    L.append("")
    L.append("> **Decision bar: the pre-registered z = %.2f** (Bonferroni "
             "alpha = %.2f / %d contrasts, two-sided; FULL_RUN_V3_PROTOCOL.md "
             "section 6, V3_POWER.md). A 3-sigma family rate over the arms "
             "present would give a different bar; where that changes a verdict "
             "it is stated, because moving the bar after the data landed is the "
             "thing this page exists to make visible."
             % (ZBAR, PREREG_ALPHA, PREREG_CONTRASTS))
    L.append("")
    L.append("> **Paired, and verified paired.** Every arm in a (backend, "
             "property, seed) was checked to share `mol_idx` and `n_atoms` "
             "elementwise and to agree on %s; every sidecar was checked to "
             "reproduce its cell's own in-band (continuous, decoded and "
             "second-oracle), stability and validity to %g. `z` is McNemar "
             "**with the continuity correction**, `z_unc` without it."
             % (", ".join("`%s`" % k for k in PAIR_KEYS), args.tol))
    L.append("")
    L.append("> **What useful yield leaves out: uniqueness.** It counts a "
             "stable in-band molecule whether or not it duplicates another. "
             "The strict column additionally requires RDKit validity.")
    L.append("")
    if flags:
        L.append("> **Outlier cells** (flagged, not excluded):")
        for f in sorted(set(flags)):
            L.append("> - %s" % f)
        L.append("")
    for be in backends:
        L.append("## `%s` -- %s" % (be, V.BACKEND_ARCH.get(be, be)))
        L.append("")
        L.append("Measured cost %.1f GPU-h (summed across unequal MIG slices -- "
                 "not one unit). n = %d x %d seeds = %d molecules per arm."
                 % (cost.get(be, 0.0) / 3600.0, args.n, len(seeds),
                    args.n * len(seeds)))
        L.append("")
        for prop in PROPS:
            arms_d = data[be].get(prop) or {}
            if not arms_d:
                continue
            arms = ([a for a in BASE_ORDER if a in arms_d]
                    + sorted(a for a in arms_d if a not in BASE_ORDER))
            N = int(next(iter(arms_d.values()))["hit"].numel())
            has_o2 = all("hs_o2_dec" in arms_d[a] for a in arms)
            L.append("### %s" % prop)
            L.append("")
            head = ("| arm | **yield (dec in-band & stable)** | +-se | yield, "
                    "continuous | yield, 2nd oracle | strict (+valid) | in_band "
                    "dec | stable | valid |")
            L.append(head)
            L.append("|---|---|---|---|---|---|---|---|---|")
            for a in arms:
                g = arms_d[a]
                y = float(g["hs_dec"].float().mean())
                se = math.sqrt(max(y * (1 - y), 0.0) / N)
                L.append("| `%s` | **%.4f** | %.4f | %.4f | %s | %.4f | %.4f | "
                         "%.4f | %.4f |"
                         % (a, y, se, float(g["hs"].float().mean()),
                            ("%.4f" % float(g["hs_o2_dec"].float().mean()))
                            if has_o2 else "--",
                            float(g["hsv_dec"].float().mean()),
                            float(g["hit_dec"].float().mean()),
                            float(g["stable"].float().mean()),
                            float(g["valid"].float().mean())))
            L.append("")
            for ref in ("unguided", "plug"):
                if ref not in arms_d:
                    continue
                others = [a for a in arms if a != ref]
                if not others:
                    continue
                L.append("**Paired against `%s`** -- verdicts at the "
                         "pre-registered z = %.2f" % (ref, ZBAR))
                L.append("")
                L.append("| arm | d(yield) | z | z_unc | discordant (arm / %s) | "
                         "z, continuous | z, 2nd oracle | d(stable) | d(valid) |"
                         % ref)
                L.append("|---|---|---|---|---|---|---|---|---|")
                for a in others:
                    g, h = arms_d[a], arms_d[ref]
                    z, w1, w2 = mcnemar(g["hs_dec"], h["hs_dec"])
                    zu, _, _ = mcnemar(g["hs_dec"], h["hs_dec"], continuity=False)
                    zc, _, _ = mcnemar(g["hs"], h["hs"])
                    z2 = (mcnemar(g["hs_o2_dec"], h["hs_o2_dec"])[0]
                          if has_o2 else float("nan"))
                    L.append("| `%s` | %+.4f | %+.2f | %+.2f | %d / %d | %+.2f | "
                             "%s | %+.4f | %+.4f |"
                             % (a, float(g["hs_dec"].float().mean()
                                         - h["hs_dec"].float().mean()),
                                z, zu, w1, w2, zc,
                                ("%+.2f" % z2) if has_o2 else "--",
                                float(g["stable"].float().mean()
                                      - h["stable"].float().mean()),
                                float(g["valid"].float().mean()
                                      - h["valid"].float().mean())))
                L.append("")

                def side(arm, key):
                    z, _, _ = mcnemar(arms_d[arm][key], arms_d[ref][key])
                    return z
                wins = [a for a in others if side(a, "hs_dec") >= ZBAR]
                losses = [a for a in others if side(a, "hs_dec") <= -ZBAR]
                L.append("Above `%s`: %s. Below `%s`: %s. Tie: %d."
                         % (ref, ", ".join("`%s`" % a for a in wins) or "**none**",
                            ref, ", ".join("`%s`" % a for a in losses) or "**none**",
                            len(others) - len(wins) - len(losses)))
                # where a second scoring choice would change the verdict
                dis = []
                for a in others:
                    v = lambda z: (1 if z >= ZBAR else (-1 if z <= -ZBAR else 0))
                    base = v(side(a, "hs_dec"))
                    if v(side(a, "hs")) != base:
                        dis.append("`%s` (continuous)" % a)
                    if has_o2 and v(side(a, "hs_o2_dec")) != base:
                        dis.append("`%s` (2nd oracle)" % a)
                L.append("")
                L.append("Verdicts that a different scoring choice would change: %s."
                         % (", ".join(dis) or "**none**"))
                L.append("")
    out = NL.join(L) + NL
    print(out)
    if args.md_out:
        dst = (args.md_out if os.path.isabs(args.md_out)
               else os.path.join(ROOT, args.md_out))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8").write(out)
        print("wrote %s" % dst, file=sys.stderr)


if __name__ == "__main__":
    main()
