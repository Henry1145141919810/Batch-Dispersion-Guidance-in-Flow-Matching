"""Blue-team doability check for BDG (read-only; CPU; no sampling).

Uses the 64 local-port cells (RTX 5080, fm_ema.pt, n=256, seed 20260925) that
the gpu lens produced, and asks: which of the ten protocol mismatches the red
team lists are (a) non-issues, (b) removable at zero compute from sidecars,
(c) need a re-run.  Prints the full metric block per (prop, target) block with
the cell's OWN unguided floor, decoded + continuous in_band at BOTH deltas,
bias/delta, sd/delta, sd ratio, and z vs the same-run plug w=1 control.
"""
import json, math, glob, os
import torch

SP = r"C:/Users/MOOOOO~1/AppData/Local/Temp/claude/c--Users-mooooonesy-Downloads-pennstuff-cis-6270-Project-1/7822c1d3-5ed2-4605-bb2c-4e0412afabed/scratchpad/bdg"
REPO = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
CELLS = SP + "/cells"
LOCAL = json.load(open(REPO + "/results/local_fb_mae.json"))
D_LOCAL = LOCAL["delta"]
D_PRE = LOCAL["delta_preregistered"]


def load(prop, arm, tgt, w, tag):
    stem = f"{CELLS}/{prop}__{arm}__{tgt}__w{w}__tmin0.5__{tag}"
    if not os.path.exists(stem + ".json"):
        return None
    j = json.load(open(stem + ".json"))
    s = torch.load(stem + ".permol.pt", weights_only=False)
    return j, s


def stats(j, s, delta):
    fin = s["finite"]
    y = s["y"][fin].double()
    out = {}
    for key, fk in (("c", "f_B"), ("d", "f_B_dec")):
        f = s[fk][fin].double()
        r = f - y
        out["ib_" + key] = float((r.abs() <= delta).double().mean())
        out["bias_" + key] = float(r.mean() / delta)
        out["sd_" + key] = float(f.std() / delta)
    out["mol"] = float(s["mol_stable"].double().mean())
    out["n"] = int(fin.sum())
    out["weff"] = j.get("diag", {}).get("bdg_w_eff")
    out["clip"] = j.get("clipped_sample_steps")
    return out


def z2(p1, p2, n1, n2):
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    return (p1 - p2) / se


# ---- 1. delta independence of BDG sampling -------------------------------
print("=" * 100)
print("1. Is delta a SAMPLING knob for bdg (as it is for btvg, tau=delta/1.96)?")
for f in sorted(glob.glob(CELLS + "/*__bdg__*.json"))[:3] + sorted(glob.glob(CELLS + "/*__bdg__*.json"))[-3:]:
    j = json.load(open(f))
    print(f"   {os.path.basename(f):48s} tau={j['bdg_tau']:.6f} = tau_mult {j['bdg_tau_mult']} x s {j['bdg_s']:.6f}"
          f" -> {j['bdg_tau_mult']*j['bdg_s']:.6f}; delta recorded {j['delta']:.5f} (unused by the field)")

# ---- 2. full block per (prop, target) ------------------------------------
LADDER = ["e0t1", "e4t0.5", "e4t0.75", "e4t1", "e4t1.25", "e4t1.5"]
summary = {}
for tgt in ("q50", "q90"):
    for prop in ("mu", "alpha", "gap"):
        ug = load(prop, "unguided", tgt, 0, "bdgctl")
        pl = load(prop, "plug", tgt, 1, "bdgctl")
        if ug is None or pl is None:
            continue
        rows = [("unguided", ug), ("plug w1", pl)]
        for tag in LADDER:
            c = load(prop, "bdg", tgt, 1, tag)
            if c is not None:
                rows.append(("bdg " + tag, c))
        pl4 = load(prop, "plug", tgt, 4, "tgt")
        if pl4 is not None:
            rows.append(("plug w4", pl4))
        su = stats(*ug, D_PRE[prop])
        floor = 0.9 * su["mol"]
        print("=" * 100)
        print(f"{prop} {tgt}: own-unguided floor = 0.9 x {su['mol']:.4f} = {floor:.4f}   "
              f"(borrowed FR3a 0.3621)   delta_pre {D_PRE[prop]:.5f}  delta_local {D_LOCAL[prop]:.5f}")
        print(f"   {'cell':14s} {'w_eff':>7s} {'ib_c':>6s} {'ib_d':>6s} {'ib_dL':>6s} {'bias/d':>7s} {'sd/d':>6s}"
              f" {'sd/ug':>6s} {'mol':>6s} {'floor':>5s} {'z_d vs plug1':>12s}")
        sp = stats(*pl, D_PRE[prop])
        blk = {}
        for name, (j, s) in rows:
            a = stats(j, s, D_PRE[prop])
            aL = stats(j, s, D_LOCAL[prop])
            z = z2(a["ib_d"], sp["ib_d"], a["n"], sp["n"])
            ok = "PASS" if a["mol"] >= floor else "FAIL"
            we = f"{a['weff']:+.3f}" if a["weff"] is not None else "   -  "
            print(f"   {name:14s} {we:>7s} {a['ib_c']:.4f} {a['ib_d']:.4f} {aL['ib_d']:.4f} {a['bias_d']:+7.2f}"
                  f" {a['sd_d']:6.2f} {a['sd_d']/su['sd_d']:6.3f} {a['mol']:.4f} {ok:>5s} {z:+12.2f}")
            blk[name] = dict(a, ib_d_local=aL["ib_d"], sd_ratio=a["sd_d"] / su["sd_d"], floor_pass=ok, z_vs_plug1=z)
        ladder = [blk["bdg " + t]["sd_ratio"] for t in LADDER[1:] if "bdg " + t in blk]
        mono = all(b > a for a, b in zip(ladder, ladder[1:]))
        print(f"   decoded sd/unguided ladder {['%.3f' % x for x in ladder]}  monotone={mono}")
        blk["_monotone_decoded"] = mono
        blk["_floor"] = floor
        summary[f"{prop}_{tgt}"] = blk

# ---- 3. what a full-protocol q90 BDG run could possibly show -------------
print("=" * 100)
print("3. Power: smallest in_band gain a q90 BDG run could resolve at z=3 vs a same-run control")
w = json.load(open(REPO + "/results/btvg3_widening_sim.json"))
for prop, p0 in (("mu", 0.0347), ("alpha", 0.0241), ("gap", 0.0594)):
    for n in (512, 5000, 15000):
        se = math.sqrt(2 * p0 * (1 - p0) / n)
        print(f"   {prop:5s} p0={p0:.4f} n/arm={n:6d}: se(diff)={se:.4f}  -> needs +{3*se:.4f} for z=3")
print("   widening ceiling from real v2 residuals (btvg3_widening_sim / data lens): +0.0001 .. +0.007")

json.dump(summary, open(SP + "/defend/doability_defence.json", "w"), indent=1, default=float)
