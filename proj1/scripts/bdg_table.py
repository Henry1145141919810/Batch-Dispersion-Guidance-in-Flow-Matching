"""Turn results/bdg/ into the section 4.4 ablation table.

Every comparison here is WITHIN the BDG job: same checkpoint, seed, window,
device and batch. That is deliberate for the FLOOR calls -- mol_stability drifts
about -0.008 between the GPU/batch-128 sweep and this CPU/batch-512 run, enough
to flip cells sitting near FR3a = 0.362109.

It is NOT because the checkpoints differ. weights/fm_ema.pt holds the same EMA
tensors as the sweep's fm_last.pt (source_md5 matches the sweep's fm_md5;
load_fm reads "ema" from both layouts). Measured on 18 matched pairs, in_band
differs by +0.0012 +/- 0.0129, below independent-draw noise. results/sweep cells
MAY be used in comparison tables alongside these.

Reports, per property:
  sd/unguided   the population spread of f_B, the quantity in_band responds to
  bias/delta    the centring, which BDG is not trying to move
  in_band       the registered acceptance metric
  mol_stab      chemistry, against the registered floor FR3a = 0.362109375
  V_b, e        the controller's own state, averaged over guided steps.
                e < 0 means the servo was in its WIDENING branch.

Run: python proj1/scripts/bdg_table.py [results_dir]
"""
import glob
import json
import math
import os
import sys

FR3A = 0.362109375
D = sys.argv[1] if len(sys.argv) > 1 else "results/bdg"


def spread(d):
    """sd of f_B about the target, with the bias removed."""
    fb, tg, rm = d.get("f_B_mean"), d.get("target_mean"), d.get("prop_rmse_eval")
    if None in (fb, tg, rm):
        return float("nan")
    return math.sqrt(max(rm ** 2 - (fb - tg) ** 2, 0.0))


def se_prop(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def load():
    rows = {}
    for f in sorted(glob.glob(os.path.join(D, "*.json"))):
        d = json.load(open(f))
        variant = os.path.basename(f).split("__")[-1][:-5]
        dg = d.get("diag") or {}
        rows[(d["prop"], d["arm"], float(d.get("w", 1)), variant)] = dict(
            sd=spread(d), bias=(d["f_B_mean"] - d["target_mean"]) / d["delta"],
            inband=d["in_band_fraction"], ms=d["mol_stability"],
            mae=d["prop_mae_eval"], n=d.get("n", 512),
            nonfinite=d.get("n_nonfinite", 0),
            clip=d.get("clipped_sample_steps", 0),
            valid=d.get("validity"), uniq=d.get("uniqueness_of_valid"),
            V_b=dg.get("bdg_V_b"), e=dg.get("bdg_e"),
            widen=dg.get("bdg_widening"), disp=dg.get("bdg_disp"))
    return rows


def main():
    rows = load()
    if not rows:
        print("no cells in %s yet" % D)
        return 1
    print("cells: %d   floor FR3a = %.6f" % (len(rows), FR3A))
    for prop in ("mu", "alpha", "gap"):
        u = rows.get((prop, "unguided", 1.0, "bdgref"))
        if not u:
            continue
        se = se_prop(u["inband"], u["n"])
        print("\n=== %s   UNGUIDED  sd=%.4f  bias=%+.3f  in_band=%.4f (se %.4f)"
              "  mol_stab=%.4f" % (prop, u["sd"], u["bias"], u["inband"], se,
                                   u["ms"]))
        print("  %-10s %-9s %7s %7s %8s %8s %7s %7s %6s"
              % ("arm", "variant", "sd/ung", "bias/d", "in_band", "mol_stab",
                 "V_b", "e", "floor"))
        order = {"unguided": 0, "plug": 1, "bdg": 2, "btvg_var": 3}
        keys = [k for k in rows if k[0] == prop and k[1] != "unguided"]
        for k in sorted(keys, key=lambda x: (order.get(x[1], 9), str(x[3]), x[2])):
            r = rows[k]
            lab = k[3] if k[1] == "bdg" else "w=%g" % k[2]
            z = ((r["inband"] - u["inband"])
                 / math.sqrt(se ** 2 + se_prop(r["inband"], r["n"]) ** 2)) \
                if se else float("nan")
            print("  %-10s %-9s %7.4f %+7.3f %8.4f %8.4f %7s %7s %6s  z=%+.2f%s"
                  % (k[1], lab, r["sd"] / u["sd"], r["bias"], r["inband"],
                     r["ms"],
                     ("%.4g" % r["V_b"]) if r["V_b"] is not None else "-",
                     ("%+.3f" % r["e"]) if r["e"] is not None else "-",
                     "PASS" if r["ms"] >= FR3A else "fail", z,
                     "  NONFINITE=%d" % r["nonfinite"] if r["nonfinite"] else ""))

        # --- the gates the paper needs -----------------------------------
        p = rows.get((prop, "plug", 1.0, "bdgref"))
        b0 = rows.get((prop, "bdg", 1.0, "e0t1"))
        if p and b0:
            same = all(abs(p[f] - b0[f]) < 1e-12
                       for f in ("sd", "bias", "inband", "ms", "mae"))
            print("  [control] bdg(eta=0) == plug at CELL level: %s"
                  % ("YES" if same else "NO  <-- the base control is INVALID"))
            if not same:
                for f in ("sd", "bias", "inband", "ms", "mae"):
                    if abs(p[f] - b0[f]) >= 1e-12:
                        print("      %-7s plug %.8g  bdg0 %.8g" % (f, p[f], b0[f]))
        # widening: did anything exceed the unguided spread while clearing FR3a?
        w = [(k, r) for k, r in rows.items()
             if k[0] == prop and r["sd"] > u["sd"] and r["ms"] >= FR3A]
        print("  [widening] floor-clearing cells with sd > unguided: %s"
              % (", ".join("%s/%s %.3fx" % (k[1], k[3], r["sd"] / u["sd"])
                           for k, r in w) if w else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
