"""Score every `dist` cell against the reference ladder, paired.

RELATION TO THE PRE-REGISTERED TABLE -- READ FIRST. The headline verdict for
the full run is `full_run_table.py`, and it is not superseded here. FR5 fixed,
on 22 Sep and before any full-run cell existed, that BTVG "beats" an arm only
at an INDEPENDENT-samples z >= 3, with paired z as supplementary. This script
does not re-decide that. It adds what full_run_table.py does not have: the
EDM/EEGSDE reference ladder (U-bound, #Atoms, L-bound), our size-shuffle rung,
gap closure, tracking beyond atom count, the joint valid-in-band yield, the
hack rate and published units -- i.e. what makes the table placeable beside
the literature. Its paired tests are SUPPLEMENTARY, consistent with FR5's own
designation of paired tests, and its comparison family (each arm vs unguided
and vs plug) is not FR5's (btvg vs every arm). Where the two disagree,
full_run_table.py is the pre-registered answer.

    python proj1/scripts/dist_report.py --ladder-only        # no cells needed
    python proj1/scripts/dist_report.py --dir results/sweep  # screening cells
    python proj1/scripts/dist_report.py --dir "results/full/n5000/seed*"
                                         # the full run: seeds POOLED

Reads the cells the sweep writes for target `dist` together with their
`.permol.pt` sidecars (run the sweep with --per-mol, or there is nothing to
pair). Four tables, in the order a reader should take them:

  1. THE LADDER. U-bound / size-shuffle / #Atoms / L-bound for our data and
     evaluator, beside EDM's and EEGSDE's published rows. Computed from the
     real test molecules and f_B alone, so it exists before any cell runs, and
     our U-bound and #Atoms should land on EDM's -- which checks that our data
     and protocol are the published ones before any arm is scored.
  2. THE ARMS. Per cell: MAE with bootstrap CI and in published units,
     in-band with Wilson CI, where the arm sits on the ladder (gap_closure,
     beats #Atoms), tracking beyond atom count (partial_corr), the joint
     valid-in-band yield, the chemistry floor, and the hack rate.
  3. PAIRED. Each arm against `unguided` and against `plug`, on the same
     molecules: paired dMAE with bootstrap CI, exact sign test on in-band,
     Holm-adjusted within each property. The family is fixed here, before any
     full-run dist cell exists -- but it is SUPPLEMENTARY (see the top): FR5's
     pre-registered test lives in full_run_table.py.

     ONE STRENGTH PER ARM, or the family is meaningless. A screening grid
     holds seven strengths per arm, and pairing every strength of `btvg`
     with every strength of `plug` is 49 comparisons of arbitrary strength
     pairs -- nominal w is not even a shared axis (btvg's w=4 is applied as
     0.0124 against plug's 4.0) -- and Holm over ~100 of them removes all
     power. So: against `unguided` every cell is paired (unguided has one
     cell per seed); arm-against-`plug` is paired only at FROZEN strengths,
     given by --frozen (the check_fullrun_go.py json) or implied when each
     arm has exactly one strength, which is what --stage full produces.
  4. STRATA. In-band by molecule size and by steering demand.

The ladder needs f_B on all 13,083 real test molecules; that is one chunked
forward pass per property and is cached beside the report.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
import dist_metrics as dm  # noqa: E402
from evaluation import _chunked, choose_delta  # noqa: E402
from m1_signed_bias import PhysicalProperty  # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT = os.path.join(ROOT, "proj1", "checkpoints")
PAIR_REFS = ("unguided", "plug")          # supplementary family; FR5 is full_run_table.py


# Which (generator, f_A, f_B) stack the cells came from. "ours" is the main
# sweep; "tfg" is transfer_sweep.py, scored by TFG's released oracle. The
# ladder MUST be built with the same oracle and the same delta as the cells it
# is compared against, so the backend picks both -- and the cache file, which
# is keyed by it, so one backend can never read the other's f_B predictions.
BACKENDS = ("ours", "tfg")
CELL_BACKEND = {"ours": None, "tfg": "TFG/EDMsecond"}


def _tfg_pair(d, prop, dev):
    """TFG's oracle and delta, built exactly as transfer_sweep.py builds them."""
    import transfer_sweep as ts
    scale = ts.generator_feat_scale(os.path.join(ROOT, "weights", "EDMsecond"))
    _, f_B, delta, rep = ts.build_pair(prop, d, ts.calibration_indices(d), dev,
                                       2.0, scale)
    return f_B, delta, rep


def fB_on_real_test(d, prop, dev, cache_dir, backend="ours"):
    """f_B on every real test molecule, cached. The L-bound and both shuffle
    references are built from these, so they are the evaluator's view of
    REAL molecules -- exactly EDM's construction."""
    tag = "" if backend == "ours" else backend + "_"
    path = os.path.join(cache_dir, "dist_fB_test_%s%s.pt" % (tag, prop))
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    if backend == "tfg":
        f_B, _, rep = _tfg_pair(d, prop, dev)
    else:
        f_B = PhysicalProperty(os.path.join(CKPT, "f_B_%s.pt" % prop),
                               len(d["types"]), dev)
    te = d["split"]["test"]
    c = d["coords"][te].to(dev).float()
    f = d["feats"][te].to(dev).float()
    m = d["mask"][te].to(dev).float()
    if backend == "tfg":
        # `Calibrated` takes the SAMPLER's feature space (the generator's
        # one-hot / normalize_factors[1]) and rescales to what its network
        # wants; the data file holds RAW one-hot. Passing it straight in fed
        # TFG's oracle features 8x too large: L-bound MAE 542 D against EDM's
        # published 0.043 D, finite and silent.
        f = f / rep["sampler_feat_scale"]
    with torch.no_grad():
        pred = _chunked(f_B, c, f, m).float().cpu()
    torch.save(pred, path)
    return pred


def build_ladder(d, prop, n, dev, cache_dir, backend="ours"):
    pi = d["props"].index(prop)
    te, tra = d["split"]["test"], d["split"]["train_a"]
    y_te = d["y"][te, pi].float().numpy()
    m_te = d["mask"][te].sum(1).round().int().numpy()
    y_tr = d["y"][tra, pi].float().numpy()
    m_tr = d["mask"][tra].sum(1).round().int().numpy()
    fB = fB_on_real_test(d, prop, dev, cache_dir, backend).numpy()
    if backend == "tfg":
        delta = _tfg_pair(d, prop, dev)[1]
    else:
        mae_b = float(torch.load(os.path.join(CKPT, "f_B_%s.pt" % prop),
                                 map_location="cpu", weights_only=False)["val_mae"])
        delta = choose_delta(mae_b, 2.0)
    # the dist set is test[:n] -- positions 0..n-1 of the test array, which is
    # exactly what guidance_sweep.run_cell takes as te_idx
    lad = dm.reference_ladder(y_te, fB, m_te, np.arange(n), y_tr, m_tr, delta)
    return lad, (y_tr, m_tr), delta


def fmt(v, p=4):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return ("%." + str(p) + "g") % v if v == v else "nan"
    return str(v)


def print_ladder(prop, lad):
    unit, scale = dm.PUBLISHED_UNIT[prop]
    print("\n=== %s  (n=%d, delta=%.4g, published unit %s) ==="
          % (prop, lad["n"], lad["delta"], unit))
    print("  %-26s %12s %12s %9s" % ("reference", "MAE (ours)", "MAE (" + unit + ")",
                                     "in_band"))
    for key, name in (("u_bound", "U-bound (shuffled)"),
                      ("size_shuffle", "size-shuffle  [ours]"),
                      ("atoms_median", "#Atoms (median | M)"),
                      ("atoms_mean", "#Atoms (mean | M)"),
                      ("l_bound", "L-bound (real molecule)")):
        r = lad[key]
        print("  %-26s %12.5g %12.5g %9.3f"
              % (name, r["mae"], r["mae"] * scale, r["in_band"]))
    print("  within-size tracking ceiling (partial_corr of real molecules): %.3f"
          % lad["l_bound"]["partial_corr"])
    print("  published:")
    for k, v in dm.PUBLISHED.get(prop, {}).items():
        print("    %-34s %9.4g %s" % (k, v, unit))


def load_cells(roots, prop, backend="ours"):
    """Cells for `prop` with target `dist`, from one or more directories.

    `roots` is a comma-separated list of directories or globs, so the full
    run's per-seed layout (results/full/n5000/seed*/) is read in one call.
    Only cells from `backend` are read; a cell from the other one is refused
    rather than skipped, because scoring it against this ladder is wrong."""
    dirs = []
    for pat in [x for x in roots.split(",") if x]:
        dirs += sorted(d for d in glob.glob(pat) if os.path.isdir(d))
    files = []
    pre = "tr__" if backend == "tfg" else ""
    for dd in dirs:
        files += sorted(glob.glob(os.path.join(dd, "%s%s__*.json" % (pre, prop))))
        # the other backend's cells do not match this glob, so without this
        # check a wrong --backend reads NOTHING and prints a cell-less ladder
        # instead of an error
        other = sorted(glob.glob(os.path.join(dd, "%s%s__*.json" % (
            "" if backend == "tfg" else "tr__", prop))))
        if other:
            raise SystemExit("%s holds %s-backend cells (e.g. %s); rerun with "
                             "--backend %s" % (dd, "ours" if backend == "tfg"
                                               else "tfg", os.path.basename(other[0]),
                                               "ours" if backend == "tfg" else "tfg"))
    cells = []
    for jf in files:
        with open(jf) as fh:
            r = json.load(fh)
        if r.get("target_name") != "dist":
            continue
        if r.get("backend") != CELL_BACKEND[backend]:
            raise SystemExit("%s is a %r cell, but this report was built for "
                             "--backend %s" % (jf, r.get("backend"), backend))
        side = jf[:-5] + ".permol.pt"
        if not os.path.exists(side):
            print("  SKIP %s: no .permol.pt sidecar (run the sweep with --per-mol)"
                  % os.path.basename(jf))
            continue
        cells.append((r, torch.load(side, weights_only=False)))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "results", "sweep"))
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--n", type=int, default=0,
                    help="dist-set size for --ladder-only (default: from cells, "
                         "else 512)")
    ap.add_argument("--ladder-only", action="store_true")
    ap.add_argument("--frozen", default="",
                    help="json with frozen_w[arm][prop] (check_fullrun_go.py "
                         "--json-out). Restricts each arm to that one strength, "
                         "which is what makes arm-vs-plug pairing meaningful.")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--backend", default="ours", choices=BACKENDS,
                    help="ours = the main sweep's cells, scored by our f_B; "
                         "tfg = transfer_sweep.py's cells, scored by TFG's "
                         "evaluate_<p> oracle with its own delta")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if not args.out:
        args.out = os.path.join(ROOT, "results", "dist_report%s.json"
                                % ("" if args.backend == "ours" else "_" + args.backend))
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    cache = os.path.join(ROOT, "results")
    d = torch.load(DATA, weights_only=False)

    frozen_w = {}
    if args.frozen:
        with open(args.frozen) as fh:
            frozen_w = json.load(fh).get("frozen_w") or {}

    report = {}
    for prop in [p for p in args.props.split(",") if p]:
        cells = [] if args.ladder_only else load_cells(args.dir, prop,
                                                       args.backend)
        if frozen_w:
            cells = [(r, pm) for r, pm in cells
                     if r["arm"] == "unguided"
                     or abs(r["w"] - float(frozen_w.get(r["arm"], {})
                                           .get(prop, float("nan")))) < 1e-12]
        n = args.n or (cells[0][0]["n"] if cells else 512)
        lad, (y_tr, m_tr), delta = build_ladder(d, prop, n, dev, cache,
                                                args.backend)
        # the ladder and the cells must share one delta, or in_band means two
        # different things in one table. Relative 1e-5, not exact: the TFG
        # calibration is re-fitted on the GPU every run and CUDA reductions
        # are not bit-deterministic (measured spread ~2e-7 relative); a
        # wrong-backend delta differs at O(1).
        off = sorted({r["delta"] for r, _ in cells
                      if abs(r["delta"] - delta) > 1e-5 * abs(delta)})
        if off:
            raise SystemExit("%s: cells were scored with delta %s but the %s "
                             "ladder uses %.6g -- wrong --backend?"
                             % (prop, off, args.backend, delta))
        print_ladder(prop, lad)
        report[prop] = {"ladder": lad, "arms": {}, "paired": {}}
        if args.ladder_only or not cells:
            if not args.ladder_only:
                print("  (no dist cells with sidecars in %s)" % args.dir)
            continue

        # ---- pool each (arm, strength) across seeds: seeds are replicates of
        # the same dist set, so one arm is ONE pooled sample, and the spread of
        # the per-seed MAEs is reported as the measured error bar
        groups = {}
        for r, pm in cells:
            groups.setdefault((r["arm"], r["w"]), []).append((r, pm))
        pooled = {}
        for key, lst in groups.items():
            seeds = [r["seed"] for r, _ in lst]
            if len(set(seeds)) != len(seeds):
                raise SystemExit("%s@w%g has two cells with the same seed -- "
                                 "two runs of one cell in --dir" % key)
            pooled[key] = (lst, dm.pool([(r["seed"], pm) for r, pm in lst]))

        ung = [v for (a, _), v in pooled.items() if a == "unguided"]
        ref_stab = (float(dm.to_np(ung[0][1]["mol_stable"]).mean())
                    if ung else None)

        print("\n  %-10s %-5s %5s %9s %17s %8s %8s %13s %7s %6s %7s %6s %6s %5s"
              % ("arm", "w", "seeds", "MAE", "MAE 95% CI", "seed sd", "in_band",
                 "in_band CI", "closure", "#Atom", "partial", "v&in", "chem",
                 "hack"))
        pms = {}
        for (arm, w), (lst, pm) in sorted(pooled.items()):
            atom_st = [r.get("atom_stability") for r, _ in lst]
            atom_st = (float(np.mean(atom_st))
                       if all(a is not None for a in atom_st) else None)
            cm = dm.cell_metrics(pm, delta, lad, prop, y_tr, m_tr,
                                 ref_mol_stability=ref_stab,
                                 atom_stability=atom_st)
            per_seed = [dm.cell_metrics(p1, delta, lad, prop, n_boot=2)["mae"]
                        for _, p1 in lst]
            cm["seeds"] = sorted(r["seed"] for r, _ in lst)
            cm["mae_per_seed"] = per_seed
            cm["mae_seed_sd"] = (float(np.std(per_seed, ddof=1))
                                 if len(per_seed) > 1 else None)
            key = "%s@w%g" % (arm, w)
            report[prop]["arms"][key] = cm
            pms.setdefault(arm, []).append(({"w": w, "arm": arm}, pm))
            print("  %-10s %-5g %5d %9.4g [%7.4g,%7.4g] %8s %8.3f [%5.3f,%5.3f] "
                  "%7.3f %6s %7.3f %6.3f %6s %5.3f"
                  % (arm, w, len(lst), cm["mae"], cm["mae_ci"][0], cm["mae_ci"][1],
                     fmt(cm["mae_seed_sd"], 3), cm["in_band"],
                     cm["in_band_ci"][0], cm["in_band_ci"][1],
                     cm["gap_closure"], fmt(cm["beats_atoms"]),
                     cm["partial_corr"], cm["valid_in_band"],
                     fmt(cm["passes_chem_floor"]), cm["hack_rate"]))

        # ---- paired, against the pre-specified references, Holm within prop
        print("\n  paired -- SUPPLEMENTARY; the pre-registered FR5 verdict is "
              "full_run_table.py (independent z >= 3)")
        print("  (A minus ref; negative dMAE = A better)")
        pvals, rows = {}, []
        one_w = {a: len({r["w"] for r, _ in lst}) == 1 for a, lst in pms.items()}
        # pooled sidecars carry per-row seeds, so pairing is on (seed, mol_idx)
        for arm, lst in sorted(pms.items()):
            if arm == "unguided":
                continue        # a reference only: "unguided vs plug" is the
                #                 same test as "plug vs unguided", and counting it
                #                 twice inflates the Holm family
            for refname in PAIR_REFS:
                if arm == refname or refname not in pms:
                    continue
                # arm-vs-plug needs ONE strength on each side (see docstring);
                # unguided is strength-free, so it always pairs
                if refname != "unguided" and not (one_w[arm] and one_w[refname]):
                    continue
                for (ra, pa) in lst:
                    for (rb, pb) in pms[refname]:
                        res = dm.paired_compare(pa, pb, delta)
                        tag = "%s@w%g vs %s@w%g" % (arm, ra["w"], refname,
                                                    rb["w"])
                        rows.append((tag, res))
                        pvals[tag + " |mae"] = res["d_mae_p"]
                        pvals[tag + " |band"] = res["in_band_p"]
        skipped_vs_plug = [a for a in pms if a not in ("unguided", "plug")
                           and "plug" in pms and not (one_w[a] and one_w["plug"])]
        if skipped_vs_plug:
            print("    (vs plug skipped for %s: several strengths per arm and no "
                  "--frozen -- pass the frozen-strength json to pair them)"
                  % ", ".join(sorted(skipped_vs_plug)))
        adj = dm.holm(pvals) if pvals else {}
        for tag, res in rows:
            print("    %-40s dMAE %+.4g [%+.4g,%+.4g] p_holm %.3g   "
                  "d_in_band %+.3f (%d vs %d) p_holm %.3g"
                  % (tag, res["d_mae"], res["d_mae_ci"][0], res["d_mae_ci"][1],
                     adj[tag + " |mae"], res["d_in_band"], res["in_band_a_only"],
                     res["in_band_b_only"], adj[tag + " |band"]))
            res["d_mae_p_holm"] = adj[tag + " |mae"]
            res["in_band_p_holm"] = adj[tag + " |band"]
            report[prop]["paired"][tag] = res

        # ---- strata
        print("\n  in-band by steering demand (distance from a typical molecule "
              "of the same size)")
        for key, cm in report[prop]["arms"].items():
            s = cm.get("in_band_by_demand", {})
            print("    %-24s %s" % (key, "   ".join(
                "%s %s (n=%d)" % (k, fmt(v["in_band"], 3), v["n"])
                for k, v in s.items())))
        print("  in-band by size tercile")
        for key, cm in report[prop]["arms"].items():
            print("    %-24s %s" % (key, "   ".join(
                "%s %s" % (k, fmt(v, 3)) for k, v in cm["in_band_by_size"].items())))

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
