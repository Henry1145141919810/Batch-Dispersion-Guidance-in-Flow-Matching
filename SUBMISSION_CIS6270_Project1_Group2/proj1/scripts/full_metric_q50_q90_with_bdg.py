"""One full metric table per (target, property), with BDG inside it.

SELECTION RULE (no chemistry floor -- Henry, 25 Sep): each arm is shown at the
strength with the best in_band over the axis that was actually swept for it.
  comparators  sweep w over 7-10 points at tau fixed      -> best over w
  bdg          sweeps tau_mult over 5 points at w = 1     -> best over tau_mult
`sel` names the selected point. BDG's w axis has never been run, so its row is
a best-of-5-on-one-line against the comparators' best-of-7-to-10 on a grid --
the remaining unfairness, and it runs in BDG's favour.

THE TWO GENERATORS ARE IN ONE TABLE BUT ARE NOT COMPARABLE IN LEVEL. Every
comparator row is `fm_last.pt` (a190ac83, n = 512, seed 20260921). Every BDG row
is the review's port, `fm_ema.pt` (e19ccc06, n = 256, seed 20260925). The `src`
column says which. The only column that crosses the two honestly is `d_ung`,
each row against the unguided baseline of ITS OWN generator, and even that
controls for the baseline level rather than for how each generator responds to
guidance. Both unguided rows are printed so the offset is visible.

bias and resid_sd for comparator cells are derived, since results/sweep carries
no per-molecule sidecar: bias = f_B_mean - target and rmse^2 = bias^2 + sd^2.
Port rows use the sidecar directly. Decoded columns exist only for port cells.

The `cmp`/`v2ref` duplicates in results/sweep are deduplicated by preferring the
provenance-stamped copy; where both exist they agree exactly (checked).

Usage:  python proj1/scripts/full_metric_q50_q90_with_bdg.py [--md-out PATH]
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

SWEEP = "results/sweep/*.json"
PORT = "results/bdg_port/cells/*.json"
COLS = ["src", "n", "sel", "in_bd", "d_ung", "in_bdD", "MAE/d", "bias/d", "sd/d",
        "sd/ung", "ceil", "rm/mae", "mstab", "valid", "astab", "uvps", "uniqV",
        "gap/d", "divers", "clip"]


def ceiling(sd, dl):
    return 2.0 * 0.5 * (1.0 + math.erf(dl / sd / math.sqrt(2.0))) - 1.0 if sd > 0 else float("nan")


def metrics(d, src, sel, usd):
    dl = d["delta"]
    if "_r" in d:
        b, sd = float(d["_r"].mean()), float(d["_r"].std(ddof=1))
    else:                                   # derive: rmse^2 = bias^2 + sd^2
        b = d["f_B_mean"] - d["target"]
        sd = math.sqrt(max(d["prop_rmse_eval"] ** 2 - b * b, 0.0))
    g = d.get
    return {
        "arm": d["arm"], "src": src, "n": d["n"], "sel": sel,
        "in_bd": g("in_band_fraction"), "d_ung": None,
        "in_bdD": g("in_band_fraction_dec"),
        "MAE/d": g("prop_mae_eval") / dl, "bias/d": b / dl, "sd/d": sd / dl,
        "sd/ung": sd / usd if usd else None, "ceil": ceiling(sd, dl),
        "rm/mae": g("prop_rmse_eval") / g("prop_mae_eval"),
        "mstab": g("mol_stability"), "valid": g("validity"),
        "astab": g("atom_stability"), "uvps": g("unique_valid_per_sample"),
        "uniqV": g("uniqueness_of_valid"),
        "gap/d": (g("guide_eval_gap_mean") / dl)
        if g("guide_eval_gap_mean") is not None else None,
        "divers": g("diversity_mean_pairwise"), "clip": g("clipped_sample_steps"),
        "_sd": sd,
    }


def load_sweep():
    """(target, prop, arm) -> {w: cell}, provenance-stamped copy preferred."""
    out, unguided = {}, {}
    for p in glob.glob(SWEEP):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if d.get("t_min_guide") != 0.5 or d.get("n") != 512:
            continue
        t = d.get("target_name")
        if t not in ("q50", "q90"):
            continue
        stamped = bool((d.get("prov") or {}).get("fm_md5"))
        if d["arm"] == "unguided":
            k = (t, d["prop"])
            if k not in unguided or stamped:
                unguided[k] = d
            continue
        k = (t, d["prop"], d["arm"])
        slot = out.setdefault(k, {})
        w = d["w"]
        if w not in slot or (stamped and not slot[w][1]):
            slot[w] = (d, stamped)
    return {k: {w: v[0] for w, v in s.items()} for k, s in out.items()}, unguided


def load_port():
    bdg, ung = {}, {}
    for p in glob.glob(PORT):
        d = json.load(open(p))
        s = p.replace(".json", ".permol.pt")
        if os.path.exists(s):
            t = torch.load(s, map_location="cpu", weights_only=False)
            fin = t["finite"].numpy()
            d["_r"] = (t["f_B"].numpy() - t["y"].numpy())[fin].astype(np.float64)
        k = (d["target_name"], d["prop"])
        if d["arm"] == "unguided":
            ung[k] = d
        elif d["arm"] == "bdg" and not d.get("bdg_onesided"):
            bdg.setdefault(k, {})[d["bdg_tau_mult"]] = d
    return bdg, ung


def fmt(v, w=7, p=4):
    if v is None or (isinstance(v, float) and v != v):
        return "%*s" % (w, "-")
    if isinstance(v, str):
        return "%*s" % (w, v[:w])
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        return "%*d" % (w, v)
    return "%*.*f" % (w, p, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-out", default="docs/results/FULL_METRICS_Q50_Q90_WITH_BDG.md")
    ap.add_argument("--json-out", default="results/full_metrics_q50_q90_with_bdg.json")
    a = ap.parse_args()

    sweep, sw_ung = load_sweep()
    bdg, pt_ung = load_port()
    dump, md = {}, []
    md.append("# Full metric block, q50 and q90, with BDG\n")
    md.append("Script-generated by `proj1/scripts/full_metric_q50_q90_with_bdg.py`. "
              "Selection: best in_band over each arm's own swept axis, **no chemistry "
              "floor**. `src` = generator: `L` = fm_last.pt a190ac83 n=512 seed 20260921; "
              "`P` = port fm_ema.pt e19ccc06 n=256 seed 20260925. Levels do NOT cross "
              "generators; only `d_ung` does, and only approximately.\n")

    for tgt in ("q50", "q90"):
        for prop in ("mu", "alpha", "gap"):
            if (tgt, prop) not in sw_ung:
                continue
            u = sw_ung[(tgt, prop)]
            ur = metrics(u, "L", "-", 1.0)
            usd = ur["_sd"]
            ur = metrics(u, "L", "-", usd)
            ur["d_ung"] = 0.0
            rows = [ur]
            for (t, pp, arm), ws in sweep.items():
                if t != tgt or pp != prop:
                    continue
                w, c = max(ws.items(), key=lambda kv: kv[1]["in_band_fraction"])
                r = metrics(c, "L", "w=%g" % w, usd)
                r["d_ung"] = r["in_bd"] - ur["in_bd"]
                rows.append(r)
            rows = [rows[0]] + sorted(rows[1:], key=lambda r: -r["in_bd"])
            pu = pt_ung.get((tgt, prop))
            if pu is not None:
                pur = metrics(pu, "P", "-", 1.0)
                pusd = pur["_sd"]
                pur = metrics(pu, "P", "-", pusd)
                pur["d_ung"] = 0.0
                rows.append(pur)
                tb = bdg.get((tgt, prop), {})
                if tb:
                    tm, c = max(tb.items(), key=lambda kv: kv[1]["in_band_fraction"])
                    r = metrics(c, "P", "tau=%g" % tm, pusd)
                    r["d_ung"] = r["in_bd"] - pur["in_bd"]
                    r["arm"] = "BDG(eta4)"
                    rows.append(r)

            title = "## %s / %s   (delta = %.5f)" % (tgt, prop, u["delta"])
            print("=" * 150)
            print(title)
            hdr = "%-14s " % "arm" + " ".join("%7s" % c for c in COLS)
            print(hdr)
            print("-" * len(hdr))
            md.append("\n" + title + "\n")
            md.append("| arm | " + " | ".join(COLS) + " |")
            md.append("|" + "---|" * (len(COLS) + 1))
            for r in rows:
                line = "%-14s " % r["arm"] + " ".join(
                    fmt(r[c], 7, 3 if c in ("sd/ung", "rm/mae", "divers") else 4)
                    if c not in ("src", "n", "sel", "clip") else fmt(r[c])
                    for c in COLS)
                print(line)
                md.append("| " + r["arm"] + " | " +
                          " | ".join(str(fmt(r[c], 1)).strip() for c in COLS) + " |")
            print()
            dump["%s_%s" % (tgt, prop)] = [{k: v for k, v in r.items()
                                            if not k.startswith("_")} for r in rows]

    prov = {"script": os.path.relpath(__file__).replace("\\", "/"),
            "script_md5": hashlib.md5(open(__file__, "rb").read()).hexdigest(),
            "selection": "best in_band over each arm's swept axis; NO chemistry floor",
            "comparators": "fm_last.pt a190ac83, n=512, seed 20260921, window 0.5",
            "bdg": "port fm_ema.pt e19ccc06, n=256, seed 20260925, w=1 only",
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
