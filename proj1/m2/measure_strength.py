"""Fix the headline strength by measurement, and WRITE IT. Protocol section 3.

v3's headline is w = 1 for every arm, and v3 is careful that this is equal
NOMINAL strength, not equal force. M2 cannot copy the number: the property
scales, the state space and the clip all differ. The only claim on record --
that guidance lifts in_band 0.062 -> 0.430 at w = 16 -- exists solely in a
commit message, from a script that prints and writes nothing, at N=128 and
NFE=100, neither of which is this protocol.

THE PRE-REGISTERED RULE, so this is a measurement and not a tuning knob:

    the headline w is the SMALLEST w in {1, 4, 16, 64} at which `plug`
    separates from `unguided` by more than the seed-to-seed standard error.

Smallest-that-separates, not best-looking. Picking the best would make the
headline an arm's own optimum, which is exactly what v3 refuses to do.
Both properties must clear it; if they disagree the larger w is taken and the
disagreement is recorded rather than averaged away.

Writes results/m2_strength.json and docs/results/M2_STRENGTH.md.

  python proj1/m2/measure_strength.py --device cuda
"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_fm as S      # noqa: E402
import m2_sweep as M        # noqa: E402


def mean_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=M.DEFAULT_CKPT)
    ap.add_argument("--props", default="gc,cpg")
    ap.add_argument("--ws", default="1,4,16,64")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--seeds", default="20260921,20260922,20260923")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json-out", default="results/m2_strength.json")
    ap.add_argument("--md-out", default="docs/results/M2_STRENGTH.md")
    a = ap.parse_args()
    ws = [float(x) for x in a.ws.split(",")]
    seeds = [int(s) for s in a.seeds.split(",")]
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = S.SimplexFM(ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state_dict"]); net.eval(); net = net.to(dev)
    X, _, _ = S.load_dfb(crop=ck["crop"])
    real_kmer = M.kmer_freq(X)

    res = {"ckpt": os.path.basename(a.ckpt), "n": a.n, "batch": a.batch,
           "steps": a.steps, "t_min": a.t_min, "seeds": seeds, "ws": ws,
           "rule": "smallest w where plug - unguided > combined seed-to-seed se",
           "by_prop": {}}

    for prop in a.props.split(","):
        real = S.PROPS[prop][1](X)
        s, y = float(real.std()), float(real.median())
        quantum = 1.0 / (ck["crop"] if prop == "gc" else ck["crop"] - 1)
        delta = max(0.16 * s, 4.4 * quantum)
        print("[%s] s=%.5f y=%.5f delta=%.5f" % (prop, s, y, delta))

        def cell(arm, w, seed):
            r, _ = M.run_cell(net, ck, arm, "-", w, y, s, None, 0.0, False,
                              a.n, a.steps, a.t_min, 1.0, seed, delta,
                              real_kmer, prop=prop, dev=dev, batch=a.batch)
            return r

        ung = [cell("unguided", 1.0, sd)["in_band_fraction"] for sd in seeds]
        um, use = mean_se(ung)
        print("      unguided  in_band %.4f +/- %.4f" % (um, use))
        rows, chosen = [], None
        for w in ws:
            vals = [cell("plug", w, sd) for sd in seeds]
            ib = [v["in_band_fraction"] for v in vals]
            kj = [v["kmer_js"] for v in vals]
            pm, pse = mean_se(ib)
            comb = math.sqrt(use ** 2 + pse ** 2)
            sep = (pm - um) > comb
            rows.append({"w": w, "in_band": pm, "se": pse,
                         "kmer_js": mean_se(kj)[0], "separates": bool(sep),
                         "combined_se": comb})
            print("      plug w=%-4g in_band %.4f +/- %.4f  (unguided + %.4f, "
                  "combined se %.4f) %s"
                  % (w, pm, pse, pm - um, comb, "SEPARATES" if sep else "-"))
            if sep and chosen is None:
                chosen = w
        res["by_prop"][prop] = {"s": s, "y": y, "delta": delta,
                                "unguided": {"in_band": um, "se": use},
                                "rows": rows, "smallest_separating": chosen}

    picks = [v["smallest_separating"] for v in res["by_prop"].values()]
    if any(p is None for p in picks):
        res["headline_w"] = None
        print("\nNO w SEPARATED on at least one property. The headline strength "
              "is NOT set;\nreport this rather than widening the grid to find "
              "one.")
    else:
        res["headline_w"] = max(picks)
        res["properties_agree"] = len(set(picks)) == 1
        print("\nheadline w = %g%s" % (res["headline_w"], "" if res[
            "properties_agree"] else "  (properties disagreed: %s -- the larger "
            "is taken and the disagreement recorded)" % picks))

    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    json.dump(res, open(a.json_out, "w"), indent=1)
    if a.md_out:
        os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
        with open(a.md_out, "w") as fh:
            fh.write("# M2 headline strength, measured\n\n")
            fh.write("Protocol section 3. Generated by "
                     "`proj1/m2/measure_strength.py`; n=%d, NFE=%d, t_min=%g, "
                     "seeds %s.\n\nRule: %s\n\n"
                     % (a.n, a.steps, a.t_min, seeds, res["rule"]))
            for prop, d in res["by_prop"].items():
                fh.write("## %s\n\nunguided in_band %.4f +/- %.4f\n\n"
                         "| w | in_band | se | kmer_js | separates? |\n"
                         "|---|---|---|---|---|\n"
                         % (prop, d["unguided"]["in_band"], d["unguided"]["se"]))
                for r in d["rows"]:
                    fh.write("| %g | %.4f | %.4f | %.5f | %s |\n"
                             % (r["w"], r["in_band"], r["se"], r["kmer_js"],
                                "yes" if r["separates"] else "no"))
                fh.write("\nsmallest separating w: %s\n\n"
                         % d["smallest_separating"])
            fh.write("**headline w = %s**\n" % res["headline_w"])
        print("wrote %s and %s" % (a.json_out, a.md_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
