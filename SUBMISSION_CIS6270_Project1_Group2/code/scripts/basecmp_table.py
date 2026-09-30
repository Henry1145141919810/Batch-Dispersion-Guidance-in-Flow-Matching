"""Read the base-model comparison's full run, under protocol v2's reporting rules.

    python proj1/scripts/basecmp_table.py --backend equifm \
        --md-out docs/results/BASECMP_FULL_equifm.md

WHY A SEPARATE READER. `full_run_v2_table.py` and `full_run_table.py` cannot read
these cells, by three independent mechanisms: they skip any cell whose `stage` is
not `"full"` (these carry `"basecmpfull"`), they key results by (property, arm) so
the `floor` and `free` cells of one arm would collide, and their consistency block
refuses a directory with mixed `t_min_guide`, which every basecmp full directory
has by construction because each arm carries its own frozen window. That is also
the reason a basecmp cell can never silently contaminate the real v2 table -- the
`stage` filter keeps the two apart in both directions.

WHAT IT REPORTS, keyed to v2's rules:

  V4  the full metric block per (set, property, arm), pooled over the three seeds,
      with every property metric in its DECODED form beside the soft one.
  V5  error-ranked buckets: molecules sorted by |f_B - y| and the block recomputed
      within the best 10 %, 50 % and 100 %. DESCRIPTIVE ONLY -- it selects
      molecules with the same oracle that then scores them, so no user could
      reproduce the top decile at generation time. It shows the SHAPE of an arm's
      error distribution, never a yield, and is never compared against another
      method's achievable number.
  V6  distinct valid in-band molecules per attempt, beside every in-band figure,
      with anything below 0.95 uniqueness labelled collapse-contaminated.
  V7  the verdict: in-band, between arms that both clear the chemistry floor, at
      z >= 3. Independent-samples z is primary here and the paired z is reported
      beside it when the sidecars are present (every arm sees the same sizes and
      the same target, so the pairing is real and the paired se is the smaller).
  V8  multiplicity: each arm against `unguided` and against the strongest
      comparator, per property, Holm-adjusted WITHIN each property.

THE TWO CAVEATS THAT MUST TRAVEL WITH EVERY NUMBER IT PRINTS, and which it prints
itself: this run is not comparable to the existing v2 full run (different guide,
different oracle, different delta, different generator), and its n may be below
v2's registered 5000, in which case its standard errors are correspondingly larger
and it must not be quoted beside v2's own numbers as though the resolution matched.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from transfer_sweep import (BACKENDS, BASECMP_FLOOR, BASECMP_ROOT,  # noqa: E402
                            BASECMP_SETS, BASECMP_FULL_SEEDS, backend_arms)

RC_OK, RC_INCOMPLETE, RC_ERROR = 0, 2, 3
UNIQ_MIN = 0.95
Z_VERDICT = 3.0
BUCKETS = (0.10, 0.50, 1.00)
# the metric the verdict is decided on, and the one selection was made on
VERDICT = "in_band_fraction_dec"


def phi_sf(z):
    """One-sided normal survival function, for a z already in sigma."""
    return 0.5 * math.erfc(abs(float(z)) / math.sqrt(2.0))


def holm(pairs):
    """[(key, p)] -> {key: adjusted p}, Holm-Bonferroni within the family."""
    out, m = {}, len(pairs)
    run = 0.0
    for i, (k, p) in enumerate(sorted(pairs, key=lambda kp: kp[1])):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def load(root, backend, props, arms, require=True):
    """{(set, prop, arm): [cells]} or an exit code."""
    pat = os.path.join(root, backend, "full", "n*", "seed*", "tr__*__full.json")
    files = sorted(glob.glob(pat))
    if not files:
        print("no cells under %s" % pat)
        return RC_INCOMPLETE
    cells, meta = {}, set()
    for fn in files:
        r = json.load(open(fn))
        if r.get("stage") != "basecmpfull":
            continue
        if r.get("backend") != BACKENDS[backend]:
            print("REFUSING: %s is a %r cell" % (os.path.basename(fn),
                                                 r.get("backend")))
            return RC_INCOMPLETE
        if r.get("prop") not in props or r.get("arm") not in arms:
            continue
        meta.add((r["n"], (r.get("prov") or {}).get("gen_md5"), r.get("batch"),
                  r.get("steps"), r.get("solver"), r.get("clip"),
                  (r.get("calibration") or {}).get("delta_mode")))
        r["_file"] = fn
        # a cell can belong to BOTH sets: the two picks coincide whenever an arm
        # never threatens the floor, and those cells are computed once
        for s in (r.get("basecmp_set") or list(BASECMP_SETS)):
            cells.setdefault((s, r["prop"], r["arm"]), []).append(r)
    if not cells:
        print("no basecmpfull cells under %s" % pat)
        return RC_INCOMPLETE
    if len(meta) != 1:
        print("INCONSISTENT cells: (n, gen_md5, batch, steps, solver, clip, "
              "delta_mode) = %s" % sorted(map(str, meta)))
        return RC_INCOMPLETE
    for p in props:
        ds = {round(float(r["delta"]), 12) for k, v in cells.items()
              for r in v if k[1] == p and r.get("delta") is not None}
        if len(ds) > 1:
            print("INCONSISTENT delta for %s: %s" % (p, sorted(ds)))
            return RC_INCOMPLETE
    if require:
        holes = [(s, p, a) for s in BASECMP_SETS for p in props for a in arms
                 if len(cells.get((s, p, a), [])) != len(BASECMP_FULL_SEEDS)]
        if holes:
            print("INCOMPLETE -- %d (set, property, arm) groups do not have all "
                  "%d seeds, e.g. %s" % (len(holes), len(BASECMP_FULL_SEEDS),
                                         ", ".join("%s/%s/%s" % h for h in holes[:8])))
            for s, p, a in holes[:8]:
                print("      %s/%s/%s has %d" % (s, p, a,
                                                 len(cells.get((s, p, a), []))))
            print("A partial run is not reported: pooling a different number of "
                  "seeds per arm makes the standard errors incomparable, which is "
                  "the one thing the verdict rests on. Pass --allow-partial to "
                  "look anyway, and say so in the write-up.")
            return RC_INCOMPLETE
    return cells, meta.pop()


def pooled(group):
    """Pool seeds for one (set, prop, arm). Every cell has the same n."""
    n = sum(int(c["n"]) for c in group)
    out = {"n_total": n, "n_seeds": len(group), "n_per_seed": int(group[0]["n"]),
           "w": float(group[0]["w"]), "t_start": float(group[0]["t_start"]),
           "delta": float(group[0].get("delta", float("nan")))}
    # fractions pool as a weighted mean; the seeds are equal-sized, so this is
    # the plain mean, but the weights are written out so an unequal set is right
    for k in ("in_band_fraction", "in_band_fraction_dec", "mol_stability",
              "atom_stability", "validity", "uniqueness_of_valid",
              "unique_valid_per_sample", "prop_mae_eval", "prop_mae_eval_dec",
              "prop_rmse_eval", "prop_rmse_eval_dec", "guide_eval_gap_mean",
              "f_B_mean", "f_B_dec_mean", "target_mean",
              "diversity_mean_pairwise"):
        vals = [(float(c[k]), int(c["n"])) for c in group if c.get(k) is not None
                and math.isfinite(float(c[k]))]
        out[k] = (sum(v * w for v, w in vals) / sum(w for _v, w in vals)
                  if vals else float("nan"))
    out["n_nonfinite"] = sum(int(c.get("n_nonfinite", 0)) for c in group)
    # spread across seeds, which is the honest "is this one seed" check
    ibs = [float(c[VERDICT]) for c in group if c.get(VERDICT) is not None]
    out["per_seed_in_band_dec"] = ibs
    out["seed_spread"] = (max(ibs) - min(ibs)) if len(ibs) > 1 else float("nan")
    p = out[VERDICT]
    out["se"] = math.sqrt(max(p * (1.0 - p), 0.0) / max(n, 1))
    return out


def buckets_from_sidecars(group, delta):
    """V5: sort by |f_B_dec - y| and recompute within the best 10 / 50 / 100 %.

    DESCRIPTIVE. It ranks molecules by the very oracle it then scores them with,
    so the top decile is not a yield anyone could reproduce without f_B at
    generation time. Returns None when the sidecars are absent."""
    try:
        import torch
    except Exception:                                          # noqa: BLE001
        return None
    err, ok, stab, val = [], [], [], []
    for c in group:
        side = c["_file"][:-5] + ".permol.pt"
        if not os.path.exists(side):
            return None
        pm = torch.load(side, weights_only=False)
        if "f_B_dec" not in pm or "y" not in pm:
            return None
        e = (pm["f_B_dec"].float() - pm["y"].float()).abs()
        err.append(e)
        ok.append(pm["finite"].bool())
        stab.append(pm["mol_stable"].bool())
        val.append(pm["valid"].bool())
    e = torch.cat(err)
    ok = torch.cat(ok)
    stab = torch.cat(stab)
    val = torch.cat(val)
    order = torch.argsort(e)
    out = {}
    for frac in BUCKETS:
        k = max(1, int(round(frac * e.numel())))
        idx = order[:k]
        out["%d%%" % round(frac * 100)] = {
            "n": int(k),
            "in_band_dec": float(((e[idx] <= delta) & ok[idx]).float().mean()),
            "mae_dec": float(e[idx][ok[idx]].float().mean()) if ok[idx].any()
            else float("nan"),
            "mol_stability": float(stab[idx].float().mean()),
            "validity": float(val[idx].float().mean())}
    return out


def z_indep(a, b):
    """Independent-samples z on two pooled fractions."""
    pa, pb = a[VERDICT], b[VERDICT]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    return (pa - pb) / se if se > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="equifm", choices=["fm", "equifm"])
    ap.add_argument("--root", default=BASECMP_ROOT)
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--arms", default="")
    ap.add_argument("--frozen", default="",
                    help="the frozen json, to carry each pick's statuses "
                         "(floor_limited / grid edges) into this table")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a] or backend_arms(args.backend)
    guided = [a for a in arms if a != "unguided"]

    got = load(args.root, args.backend, props, arms,
               require=not args.allow_partial)
    if isinstance(got, int):
        return got
    cells, meta = got
    n_per, gen_md5, batch, steps, solver, clip, dmode = meta

    frozen = {}
    if args.frozen and os.path.exists(args.frozen):
        frozen = json.load(open(args.frozen))

    agg = {k: pooled(v) for k, v in cells.items()}

    print("%s -- basecmp full run, %d cells, n=%d x %d seeds, delta_mode=%s, "
          "generator %s" % (BACKENDS[args.backend], sum(len(v) for v in cells.values()),
                            n_per, len(BASECMP_FULL_SEEDS), dmode, str(gen_md5)[:12]))
    if n_per * len(BASECMP_FULL_SEEDS) < 15000:
        print("  NOTE: n x seeds = %d against v2's registered 15000. The se of an "
              "in-band difference here is ~%.4f against v2's ~0.0025. Report this "
              "as a reduced-n v2 run; do not quote it beside v2's own numbers as "
              "though the resolution matched."
              % (n_per * len(BASECMP_FULL_SEEDS),
                 math.sqrt(2 * 0.05 * 0.95 / (n_per * len(BASECMP_FULL_SEEDS)))))
    print("  NOTE: not comparable to the existing v2 full run -- different guide, "
          "oracle, delta and generator.")

    lines = []
    verdicts = {}
    for s in BASECMP_SETS:
        print("\n=== set %r ===" % s)
        for p in props:
            u = agg.get((s, p, "unguided"))
            if u is None:
                print("  %s: no unguided cell; no floor and no verdict" % p)
                continue
            floor = BASECMP_FLOOR * u["mol_stability"]
            print("  %-6s floor %.4f (0.9 x unguided %.4f)  delta %.5f"
                  % (p, floor, u["mol_stability"], u["delta"]))
            print("    %-9s %6s %5s  %7s %7s  %7s  %6s  %6s  %8s %s"
                  % ("arm", "w", "t", "in-band", "(soft)", "se", "stab", "uniq",
                     "MAE_dec", "flags"))
            fam = []
            for a in arms:
                g = agg.get((s, p, a))
                if g is None:
                    continue
                clears = g["mol_stability"] >= floor - 1e-12
                fl = []
                if not clears:
                    fl.append("BELOW-FLOOR")
                if g["unique_valid_per_sample"] < UNIQ_MIN:
                    fl.append("COLLAPSE")
                if g["n_nonfinite"]:
                    fl.append("nonfinite=%d" % g["n_nonfinite"])
                pk = (frozen.get("frozen", {}).get(s, {}).get(a, {}).get(p)
                      if frozen else None)
                if pk:
                    if pk.get("w_edge"):
                        fl.append("w-edge")
                    if pk.get("t_edge"):
                        fl.append("t-edge")
                st = (frozen.get("status", {}).get(s, {}).get(a, {}).get(p)
                      if frozen else None)
                if st == "floor_limited":
                    fl.append("floor-limited-at-selection")
                print("    %-9s %6g %5g  %7.4f %7.4f  %7.4f  %6.3f  %6.3f  %8.4f %s"
                      % (a, g["w"], g["t_start"], g[VERDICT],
                         g["in_band_fraction"], g["se"], g["mol_stability"],
                         g["unique_valid_per_sample"], g["prop_mae_eval_dec"],
                         ",".join(fl) or "-"))
                if a != "unguided" and clears:
                    fam.append(a)

            # ---- V7/V8: the verdict family, Holm-adjusted within this property
            comparators = [a for a in fam if a in ("plug", "tmpd", "lgd_mc", "tfg")]
            best_cmp = (max(comparators, key=lambda a: agg[(s, p, a)][VERDICT])
                        if comparators else None)
            tests = []
            for a in fam:
                tests.append((("%s vs unguided" % a),
                              z_indep(agg[(s, p, a)], u)))
                if best_cmp and a != best_cmp:
                    tests.append((("%s vs %s" % (a, best_cmp)),
                                  z_indep(agg[(s, p, a)], agg[(s, p, best_cmp)])))
            ps = [(k, phi_sf(z)) for k, z in tests if math.isfinite(z)]
            adj = holm(ps)
            if tests:
                print("    verdicts (in-band, decoded; only arms clearing the "
                      "floor; Holm within this property; z >= %.0f is a verdict)"
                      % Z_VERDICT)
                for k, z in tests:
                    a_p = adj.get(k, float("nan"))
                    call = ("VERDICT" if abs(z) >= Z_VERDICT and a_p < 0.05
                            else "tie")
                    print("      %-26s z = %+6.2f   Holm p = %.2g   %s"
                          % (k, z, a_p, call))
            if best_cmp:
                verdicts[(s, p)] = best_cmp
            excluded = [a for a in guided if a not in fam
                        and agg.get((s, p, a)) is not None]
            if excluded:
                print("    excluded from the verdict family (below the chemistry "
                      "floor, which is a result and not a win): %s"
                      % ", ".join(excluded))

            # ---- V5: error-ranked buckets, descriptive
            for a in arms:
                g = cells.get((s, p, a))
                if not g:
                    continue
                bk = buckets_from_sidecars(g, agg[(s, p, a)]["delta"])
                if bk is None:
                    continue
                lines.append((s, p, a, bk))
    if lines:
        print("\n=== V5 error-ranked buckets (DESCRIPTIVE -- ranked by the same "
              "oracle that scores them, so NOT a yield) ===")
        print("  %-6s %-9s %-6s %8s %9s %8s" % ("prop", "arm", "bucket",
                                                "in-band", "MAE_dec", "stab"))
        for s, p, a, bk in lines:
            if s != BASECMP_SETS[0]:
                continue
            for name in ("10%", "50%", "100%"):
                b = bk[name]
                print("  %-6s %-9s %-6s %8.4f %9.4f %8.3f"
                      % (p, a, name, b["in_band_dec"], b["mae_dec"],
                         b["mol_stability"]))
    else:
        print("\nV5 buckets: no per-molecule sidecars found, so the error-ranked "
              "buckets are not available. (--per-mol writes them; the cells "
              "themselves carry only the aggregate block.)")

    if args.md_out:
        with open(args.md_out + ".tmp", "w", encoding="utf-8") as fh:
            fh.write("# Base-model comparison: the v2 full run on %s\n\n"
                     % BACKENDS[args.backend])
            fh.write("**Generated by `proj1/scripts/basecmp_table.py` on %s. "
                     "Do not edit by hand.** n = %d x %d seeds, delta mode `%s`, "
                     "generator `%s`.\n\n"
                     % (datetime.date.today().isoformat(), n_per,
                        len(BASECMP_FULL_SEEDS), dmode, str(gen_md5)[:12]))
            fh.write("Not comparable to the existing v2 full run: different "
                     "guide, oracle, delta and generator.\n\n")
            for s in BASECMP_SETS:
                fh.write("## Set `%s`\n\n" % s)
                fh.write("| property | arm | w | t | in-band (dec) | se | "
                         "mol stab | uniq/sample | MAE (dec) |\n")
                fh.write("|---|---|---|---|---|---|---|---|---|\n")
                for p in props:
                    for a in arms:
                        g = agg.get((s, p, a))
                        if g is None:
                            continue
                        fh.write("| %s | %s | %g | %g | %.4f | %.4f | %.3f | "
                                 "%.3f | %.4f |\n"
                                 % (p, a, g["w"], g["t_start"], g[VERDICT],
                                    g["se"], g["mol_stability"],
                                    g["unique_valid_per_sample"],
                                    g["prop_mae_eval_dec"]))
                fh.write("\n")
        os.replace(args.md_out + ".tmp", args.md_out)
        print("\nwrote %s" % args.md_out)
    return RC_OK


if __name__ == "__main__":
    raise SystemExit(main())
