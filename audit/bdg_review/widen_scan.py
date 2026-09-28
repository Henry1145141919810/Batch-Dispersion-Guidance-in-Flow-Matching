"""Can any EXISTING arm in this repo widen the property law while clearing the
chemistry floor?  (BDG's novelty claim: 'the largest reproducible floor-clearing
widening anywhere was 1.010x'.)

For a fixed scalar target y, the cell JSON gives MAE, RMSE and f_B_mean, so

    bias = f_B_mean - y
    sd   = sqrt(RMSE^2 - bias^2)          exact, no distributional assumption

is the spread of the predicted property.  Ratio each guided cell's sd to the
unguided cell in the SAME (property, target, window, stage-suffix) group, and
flag the ones that clear the 0.9 x unguided molecule-stability floor.
Read-only; no sampling.
"""
import glob
import json
import math
import os
import collections

REPO = "C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"

cells = []
for f in glob.glob(os.path.join(REPO, "results/sweep*/*.json")):
    d = json.load(open(f))
    if d.get("target_name") == "dist" or d.get("target") is None:
        continue
    if d.get("n", 0) < 512:
        continue
    rmse = d["prop_rmse_eval"]
    bias = d["f_B_mean"] - d["target"]
    v = rmse * rmse - bias * bias
    if v < 0:
        continue
    suffix = os.path.basename(f)[:-5].split("__")
    tag = suffix[-1] if not suffix[-1].startswith("tmin") else ""
    cells.append(dict(f=os.path.basename(f)[:-5], prop=d["prop"], arm=d["arm"],
                      w=d["w"], tmin=d["t_min_guide"], tgt=d["target_name"],
                      tag=tag, seedtree="seed2" if "sweep_v2_seed2" in f else "seed1",
                      sd=math.sqrt(v), bias=bias, mol=d["mol_stability"],
                      inb=d["in_band_fraction"], var=d.get("variant", "-"),
                      seed=d.get("seed"), n=d["n"]))

print("cells scanned:", len(cells))
key = lambda c: (c["prop"], c["tgt"], c["tmin"], c["tag"], c["seedtree"])
ung = {key(c): c for c in cells if c["arm"] == "unguided"}
print("unguided reference groups:", len(ung))

rows = []
for c in cells:
    if c["arm"] == "unguided":
        continue
    u = ung.get(key(c))
    if u is None:
        continue
    rows.append(dict(c, ratio=c["sd"] / u["sd"], floor=0.9 * u["mol"],
                     u_sd=u["sd"], u_mol=u["mol"], u_inb=u["inb"]))

print("guided cells with a matched unguided reference:", len(rows))
clear = [r for r in rows if r["mol"] >= r["floor"]]
print("of which clear the 0.9x chemistry floor:", len(clear))
wide = sorted([r for r in clear if r["ratio"] > 1.0], key=lambda r: -r["ratio"])
print("\n--- widest floor-clearing cells (sd ratio > 1) ---")
print("%-58s %-7s %-6s ratio  mol_stab floor  in_band  n" % ("cell", "arm", "tree"))
for r in wide[:20]:
    print("%-58s %-7s %-6s %.3f  %.4f  %.4f  %.4f  %d"
          % (r["f"][:58], r["arm"], r["seedtree"], r["ratio"], r["mol"], r["floor"], r["inb"], r["n"]))
print("\ncount of floor-clearing cells with ratio > 1.05:",
      sum(1 for r in clear if r["ratio"] > 1.05))
print("count of floor-clearing cells with ratio > 1.10:",
      sum(1 for r in clear if r["ratio"] > 1.10))

# reproducibility across the two seed trees: same (prop,arm,w,tmin,tgt) in both
byk = collections.defaultdict(dict)
for r in rows:
    byk[(r["prop"], r["arm"], r["w"], r["tmin"], r["tgt"], r["var"])][r["seedtree"]] = r
rep = [(k, v["seed1"], v["seed2"]) for k, v in byk.items() if len(v) == 2]
print("\nconfigs present in BOTH seed trees:", len(rep))
both = [(k, a, b) for k, a, b in rep
        if min(a["ratio"], b["ratio"]) > 1.0
        and a["mol"] >= a["floor"] and b["mol"] >= b["floor"]]
both.sort(key=lambda t: -min(t[1]["ratio"], t[2]["ratio"]))
print("configs that widen AND clear the floor in BOTH trees:", len(both))
for k, a, b in both[:15]:
    print("  %-5s %-10s w=%-5s tmin=%-5s %-4s  ratio %.3f / %.3f   mol %.4f / %.4f"
          % (k[0], k[1], k[2], k[3], k[4], a["ratio"], b["ratio"], a["mol"], b["mol"]))

# what the tightening side looks like, for contrast
tight = sorted([r for r in clear], key=lambda r: r["ratio"])[:10]
print("\n--- tightest floor-clearing cells (for contrast) ---")
for r in tight:
    print("%-58s %-7s ratio %.3f  mol %.4f floor %.4f in_band %.4f (unguided %.4f)"
          % (r["f"][:58], r["arm"], r["ratio"], r["mol"], r["floor"], r["inb"], r["u_inb"]))
