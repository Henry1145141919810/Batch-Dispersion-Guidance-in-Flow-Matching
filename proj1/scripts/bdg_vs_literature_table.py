"""The full metric block for BDG beside plug, and beside the literature arms.

WHY THIS IS TWO TABLES AND NOT ONE. BDG has never been run on the generator the
literature comparators were run on. Every BDG cell available here comes from the
review's independent port (`results/bdg_port/`, `fm_ema.pt`, md5 e19ccc06,
n = 256, one seed, RTX 5080). Every tmpd / lgd_mc / tfg cell comes from the v2
full run (`fm_last.pt`, md5 a190ac83, n = 5000, three seeds, B200). BDG_HANDOFF
§5 states the rule this enforces -- "No BDG number may be compared to
`results/sweep`" -- and the same applies to the v2 tree. Joining the two would
compare arms across generators, batch sizes and sample counts at once.

So: table A is BDG's own controlled comparison (its unguided and plug controls
were re-run inside the same jobs, which is what makes it valid). Table B is the
literature comparison at the same target, on its own generator. The only honest
cross-reading is of SHAPES -- does the bias dominate, does any arm clear the
chemistry floor -- never of levels.

Columns. Everything the cells carry, in units of delta where it is a property
error. `ceiling` is the coverage a perfectly centred Gaussian of that residual
sd would score; in_band above it means the residual law is more peaked than
Gaussian, below means less. `z vs plug` is a paired McNemar sign test on the
same molecule indices, plug as the base arm, computed only within a table.

Usage:  python proj1/scripts/bdg_vs_literature_table.py [--json-out PATH]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

PORT = "results/bdg_port/cells"
V2 = "results/full/v2/n5000/seed*"


def z_(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load(path):
    d = json.load(open(path))
    side = path.replace(".json", ".permol.pt")
    if os.path.exists(side):
        t = torch.load(side, map_location="cpu", weights_only=False)
        fin = t["finite"].numpy()
        d["_r"] = (t["f_B"].numpy() - t["y"].numpy())[fin].astype(np.float64)
        d["_hit"] = (np.abs(d["_r"]) <= d["delta"]).astype(int)
    return d


def row(d, usd, floor):
    r, dl = d.get("_r"), d["delta"]
    b = float(r.mean()) if r is not None else float("nan")
    sd = float(r.std(ddof=1)) if r is not None else float("nan")
    g = lambda k: d.get(k)
    return {
        "arm": d["arm"], "w": d.get("w"), "variant": d.get("variant") or "-",
        "n": d["n"],
        "in_band": g("in_band_fraction"), "in_band_dec": g("in_band_fraction_dec"),
        "mae_d": g("prop_mae_eval") / dl,
        "mae_dec_d": (g("prop_mae_eval_dec") / dl) if g("prop_mae_eval_dec") else None,
        "rmse_over_mae": g("prop_rmse_eval") / g("prop_mae_eval"),
        "bias_d": b / dl, "sd_d": sd / dl, "sd_over_unguided": sd / usd,
        "ceiling": z_(dl / sd) * 2 - 1 if sd == sd and sd > 0 else None,
        "mol_stab": g("mol_stability"), "validity": g("validity"),
        "atom_stab": g("atom_stability"),
        "uniq_valid": g("uniqueness_of_valid"),
        "uvps": g("unique_valid_per_sample"),
        "guide_eval_gap_d": (g("guide_eval_gap_mean") / dl)
        if g("guide_eval_gap_mean") is not None else None,
        "diversity": g("diversity_mean_pairwise"),
        "clipped": g("clipped_sample_steps"),
        "floor_ok": (g("mol_stability") >= floor) if floor else None,
    }


def mcnemar(hit_a, hit_b):
    """z for arm A against base B, paired on molecule index."""
    if hit_a is None or hit_b is None or len(hit_a) != len(hit_b):
        return None
    a_only = int(((hit_a == 1) & (hit_b == 0)).sum())
    b_only = int(((hit_a == 0) & (hit_b == 1)).sum())
    m = a_only + b_only
    return (a_only - b_only) / math.sqrt(m) if m else 0.0


def emit(title, note, rows, base_hit):
    print("=" * 128)
    print(title)
    print(note)
    print("-" * 128)
    h = ("%-9s %-8s %6s %7s %7s %7s %7s %7s %7s %7s %7s %6s %6s %6s %6s %7s %5s %6s"
         % ("arm", "variant", "n", "in_bd", "in_bdD", "MAE/d", "bias/d", "sd/d",
            "sd/ung", "ceil", "rmse/mae", "mstab", "valid", "astab", "uvps",
            "gap/d", "floor", "z vs plug"))
    print(h)
    f = lambda v, s="%7.4f": (s % v) if isinstance(v, (int, float)) and v == v else "%7s" % "-"
    for r, hit in rows:
        print("%-9s %-8s %6d %s %s %s %s %s %s %s %s %s %s %s %s %s %5s %s"
              % (r["arm"], r["variant"], r["n"],
                 f(r["in_band"]), f(r["in_band_dec"]), f(r["mae_d"], "%7.2f"),
                 f(r["bias_d"], "%7.2f"), f(r["sd_d"], "%7.2f"),
                 f(r["sd_over_unguided"], "%7.3f"), f(r["ceiling"]),
                 f(r["rmse_over_mae"], "%7.3f"), f(r["mol_stab"], "%6.4f"),
                 f(r["validity"], "%6.4f"), f(r["atom_stab"], "%6.4f"),
                 f(r["uvps"], "%6.4f"), f(r["guide_eval_gap_d"], "%7.2f"),
                 {True: "ok", False: "FAIL", None: "-"}[r["floor_ok"]],
                 f(mcnemar(hit, base_hit), "%6.2f") if hit is not None else "%6s" % "-"))
    print()


def table_port(prop, tgt):
    def p(n):
        f = os.path.join(PORT, n + ".json")
        return load(f) if os.path.exists(f) else None
    u = p("%s__unguided__%s__w0__tmin0.5__bdgctl" % (prop, tgt))
    if u is None:
        return None
    usd = float(u["_r"].std(ddof=1))
    floor = 0.9 * u["mol_stability"]
    names = [("unguided", "%s__unguided__%s__w0__tmin0.5__bdgctl" % (prop, tgt)),
             ("plug w1", "%s__plug__%s__w1__tmin0.5__bdgctl" % (prop, tgt)),
             ("plug w4", "%s__plug__%s__w4__tmin0.5__tgt" % (prop, tgt))]
    names += [("bdg " + v, "%s__bdg__%s__w1__tmin0.5__%s" % (prop, tgt, v))
              for v in ("e0t1", "e4t0.5", "e4t0.75", "e4t1", "e4t1.25", "e4t1.5")]
    base = p("%s__plug__%s__w1__tmin0.5__bdgctl" % (prop, tgt))
    out = []
    for lab, n in names:
        d = p(n)
        if d is None:
            continue
        r = row(d, usd, floor)
        r["variant"] = lab.split(" ", 1)[1] if " " in lab else "-"
        out.append((r, d.get("_hit")))
    return out, base.get("_hit"), floor


def table_v2(prop):
    cells = {}
    for f in sorted(glob.glob(os.path.join(V2, "%s__*__q90__*full.json" % prop))):
        d = load(f)
        cells.setdefault(d["arm"], []).append(d)
    if "unguided" not in cells:
        return None
    usd = float(np.mean([c["_r"].std(ddof=1) for c in cells["unguided"]]))
    floor = 0.9 * float(np.mean([c["mol_stability"] for c in cells["unguided"]]))
    out = []
    order = ["unguided", "plug", "tmpd", "lgd_mc", "tfg"]
    base_hit = None
    for arm in order:
        if arm not in cells:
            continue
        rs = [row(c, usd, floor) for c in cells[arm]]
        agg = {k: (float(np.mean([x[k] for x in rs]))
                   if isinstance(rs[0][k], (int, float)) and rs[0][k] is not None
                   else rs[0][k]) for k in rs[0]}
        agg["arm"] = arm
        agg["variant"] = "w=%g" % cells[arm][0]["w"]
        agg["n"] = sum(x["n"] for x in rs)
        agg["floor_ok"] = agg["mol_stab"] >= floor
        hit = np.concatenate([c["_hit"] for c in cells[arm]])
        if arm == "plug":
            base_hit = hit
        out.append((agg, hit))
    return out, base_hit, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="results/bdg_vs_literature.json")
    a = ap.parse_args()
    dump = {}
    for prop in ("mu", "alpha", "gap"):
        for tgt in ("q50", "q90"):
            t = table_port(prop, tgt)
            if t is None:
                continue
            rows, base, floor = t
            emit("TABLE A -- %s, target %s : BDG and its own controls" % (prop.upper(), tgt),
                 "generator fm_ema.pt (e19ccc06), n=256, 1 seed (20260925), RTX 5080, "
                 "batch=n. Chemistry floor = 0.9 x unguided = %.4f. NOT comparable "
                 "to any results/sweep or v2 number." % floor, rows, base)
            dump["A_%s_%s" % (prop, tgt)] = [r for r, _ in rows]
    for prop in ("mu", "alpha", "gap"):
        t = table_v2(prop)
        if t is None:
            continue
        rows, base, floor = t
        emit("TABLE B -- %s, target q90 : the literature arms (v2 full run)" % prop.upper(),
             "generator fm_last.pt (a190ac83), n=5000 x 3 seeds, B200. Strengths "
             "frozen by rule V2. Chemistry floor = %.4f. BDG WAS NEVER RUN HERE."
             % floor, rows, base)
        dump["B_%s_q90" % prop] = [r for r, _ in rows]

    prov = {"script": os.path.relpath(__file__).replace("\\", "/"),
            "script_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
            "port_generator": "fm_ema.pt e19ccc06", "v2_generator": "fm_last.pt a190ac83",
            "numpy": np.__version__, "torch": torch.__version__}
    try:
        prov["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   text=True).strip()
    except Exception:
        prov["git_head"] = None
    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    json.dump({"prov": prov, "tables": dump}, open(a.json_out, "w"), indent=1,
              default=float)
    print("wrote %s" % a.json_out)


if __name__ == "__main__":
    main()
