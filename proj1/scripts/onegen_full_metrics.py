"""Full metric block, one generator, every arm including BDG.

This is the table the two-generator versions could not be. Every cell here was
produced by the SAME generator (`weights/fm_ema.pt`, md5 e19ccc06), the same
seed (20260925), the same n (256), the same window (t_min_guide = 0.5) and the
same 100-step Euler solver, on one RTX 5080. The BDG cells and the comparator
cells are therefore directly comparable in LEVEL, not only in change -- which
was the whole point of re-running the comparators locally.

SELECTION: each arm at the strength with the best in_band over its own grid,
with NO chemistry floor (Henry, 25 Sep). `mstab` is still reported so the cost
is visible; it simply does not gate the pick. The comparators sweep w over
{0.05, 0.25, 1, 2, 4}. BDG sweeps the 2-D grid w x tau_mult, 5 x 5 = 25 cells
per (property, target), eta = 4 throughout. `grid` says how many of the 5
strengths have landed; anything below 5/5 is a partial pick and is marked.

THE OPEN CONFOUND, which this table cannot settle. BDG reduces exactly to plug
at a rescaled weight (BDG_HANDOFF §3): w_eff = 1 + eta*e. At BDG's winning
cells the run-mean `w x w_eff` is 5-10, while the comparator grid stops at
w = 4. So a BDG win may be a longer lever rather than a mechanism. The `plugeq`
column prints w * mean(w_eff) for BDG rows so the reader can see where each
pick actually sits. Resolving it needs plug at w in {8, 16, 32}; until those
run, no BDG win in this table should be quoted.

POWER: n = 256, so se on in_band at p ~ 0.1 is about 0.019 and the smallest
paired difference resolvable at z = 3 is ~0.05. Every gap in this table is
smaller than that. It is a screen.

Usage:  python proj1/scripts/onegen_full_metrics.py [--md-out PATH]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess

import numpy as np
import torch

CELLS = "results/bdg_local/*.json"
COLS = ["sel", "plugeq", "grid", "in_bd", "d_ung", "in_bdD", "MAE/d", "bias/d",
        "sd/d", "sd/ung", "ceil", "rm/mae", "mstab", "valid", "astab", "uvps",
        "gap/d", "divers", "clip"]


def ceiling(sd, dl):
    return 2.0 * 0.5 * (1.0 + math.erf(dl / sd / math.sqrt(2.0))) - 1.0 if sd > 0 else float("nan")


def load():
    ung, arms = {}, {}
    for p in sorted(glob.glob(CELLS)):
        d = json.load(open(p))
        s = p.replace(".json", ".permol.pt")
        if os.path.exists(s):
            t = torch.load(s, map_location="cpu", weights_only=False)
            fin = t["finite"].numpy()
            d["_r"] = (t["f_B"].numpy() - t["y"].numpy())[fin].astype(np.float64)
        k = (d["target_name"], d["prop"])
        if d["arm"] == "unguided":
            ung[k] = d
        elif d["arm"] == "bdg":
            if d.get("bdg_onesided"):
                continue
            arms.setdefault((k, "bdg"), {})[(d["w"], d["bdg_tau_mult"])] = d
        else:
            arms.setdefault((k, d["arm"]), {})[d["w"]] = d
    return ung, arms


def metrics(d, usd, uin):
    dl = d["delta"]
    r = d.get("_r")
    if r is not None:
        b, sd = float(r.mean()), float(r.std(ddof=1))
    else:
        b = d["f_B_mean"] - d["target"]
        n = d["n"]                       # rmse uses ddof=0; match the sidecar's ddof=1
        sd = math.sqrt(max(d["prop_rmse_eval"] ** 2 - b * b, 0.0) * n / (n - 1.0))
        if d.get("target_name") == "dist":
            raise ValueError("the rmse fallback assumes a constant target; "
                             "`dist` cells vary y per molecule -- pass sidecars")
    g = d.get
    we = (d.get("diag") or {}).get("bdg_w_eff")
    return {
        "in_bd": g("in_band_fraction"), "d_ung": g("in_band_fraction") - uin,
        "in_bdD": g("in_band_fraction_dec"),
        "MAE/d": g("prop_mae_eval") / dl, "bias/d": b / dl, "sd/d": sd / dl,
        "sd/ung": sd / usd, "ceil": ceiling(sd, dl),
        "rm/mae": g("prop_rmse_eval") / g("prop_mae_eval"),
        "mstab": g("mol_stability"), "valid": g("validity"),
        "astab": g("atom_stability"), "uvps": g("unique_valid_per_sample"),
        "gap/d": g("guide_eval_gap_mean") / dl,
        "divers": g("diversity_mean_pairwise"), "clip": g("clipped_sample_steps"),
        "plugeq": (d["w"] * we) if we is not None else None, "_sd": sd,
    }


def fmt(v, w=7, p=4):
    if v is None or (isinstance(v, float) and v != v):
        return "%*s" % (w, "-")
    if isinstance(v, str):
        return "%*s" % (w, v)
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return "%*d" % (w, v)
    return "%*.*f" % (w, p, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-out", default="docs/results/ONEGEN_FULL_METRICS.md")
    ap.add_argument("--json-out", default="results/onegen_full_metrics.json")
    a = ap.parse_args()
    ung, arms = load()
    md, dump = [], {}
    md.append("# Full metric block, one generator, all arms incl. BDG\n")
    md.append("`fm_ema.pt` e19ccc06 | n=256 | seed 20260925 | window 0.5 | 100-step Euler "
              "| RTX 5080. Best strength per arm, **no chemistry floor**. "
              "se(in_band) ~ 0.019; nothing under ~0.05 is resolvable. "
              "`plugeq` = w x mean(w_eff) for BDG: where the pick sits on plug's "
              "FIELD-MAGNITUDE scale. BDG's centring gain on (y - F_i) is still "
              "exactly w, so this is NOT 'BDG = plug at w=plugeq'. Comparator grid stops at w=4, so plugeq > 4 is "
              "outside it.\n")
    for tgt in ("q50", "q90"):
        for prop in ("mu", "alpha", "gap"):
            k = (tgt, prop)
            if k not in ung:
                continue
            u = ung[k]
            um = metrics(u, 1.0, 0.0)
            usd, uin = um["_sd"], u["in_band_fraction"]
            um = metrics(u, usd, uin)
            rows = [("unguided", "-", 0, um)]
            ngrid = 0
            for (kk, arm), ws in arms.items():
                if kk != k:
                    continue
                # tie-break on the SMALLER strength, not on dict/filename order.
                # max() keeps the FIRST maximal element, so sort ascending by w.
                bw, bc = max(sorted(ws.items(), key=lambda kv: kv[0]),
                             key=lambda kv: kv[1]["in_band_fraction"])
                nstr = len({w for w, _ in ws} if arm == "bdg" else set(ws))
                ngrid = max(ngrid, nstr)
                sel = ("w=%g,t=%g" % bw) if arm == "bdg" else "w=%g" % bw
                rows.append((arm, sel, nstr, metrics(bc, usd, uin)))
            rows = [rows[0]] + sorted(rows[1:], key=lambda r: -r[3]["in_bd"])
            title = "## %s / %s   (delta = %.5f, unguided in_band %.4f)" % (
                tgt, prop, u["delta"], uin)
            hdr = "%-9s " % "arm" + " ".join("%7s" % c for c in COLS)
            print("=" * len(hdr)); print(title); print(hdr); print("-" * len(hdr))
            md.append("\n" + title + "\n")
            md.append("| arm | " + " | ".join(COLS) + " |")
            md.append("|" + "---|" * (len(COLS) + 1))
            for arm, sel, nstr, m in rows:
                m = dict(m, sel=sel,
                         grid=("-" if arm == "unguided" else "%d/%d" % (nstr, ngrid)))
                line = "%-9s " % arm + " ".join(
                    fmt(m[c], 7, 3 if c in ("sd/ung", "rm/mae", "divers", "plugeq") else 4)
                    for c in COLS)
                print(line)
                md.append("| " + arm + " | " +
                          " | ".join(fmt(m[c], 1).strip() for c in COLS) + " |")
            print()
            dump["%s_%s" % (tgt, prop)] = [
                dict({kk: vv for kk, vv in m.items() if not kk.startswith("_")},
                     arm=arm, sel=sel, grid=nstr) for arm, sel, nstr, m in rows]
    prov = {"script": os.path.relpath(__file__).replace("\\", "/"),
            "script_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
            "generator": "weights/fm_ema.pt e19ccc06", "n": 256,
            "seed": 20260925, "window": 0.5, "eta": 4.0,
            "selection": "best in_band over own grid; NO chemistry floor",
            "open_confound": "plug grid stops at w=4; BDG picks sit at plugeq 5-10",
            "numpy": np.__version__, "torch": torch.__version__}
    try:
        prov["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   text=True).strip()
    except Exception:
        prov["git_head"] = None
    md.append("\n---\n\n```json\n" + json.dumps(prov, indent=1) + "\n```\n")
    os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
    open(a.md_out, "w").write("\n".join(md))
    json.dump({"prov": prov, "tables": dump}, open(a.json_out, "w"), indent=1,
              default=float)
    print("wrote %s and %s" % (a.md_out, a.json_out))


if __name__ == "__main__":
    main()
