import json, glob, math, os, re
from collections import defaultdict
ROOT = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
SE = {("mu",512):0.0662, ("alpha",512):0.0537, ("gap",512):0.0372}

def load(p):
    try: d = json.load(open(p))
    except Exception: return None
    if not isinstance(d, dict) or "prop_rmse_eval" not in d: return None
    return d

def sd_of(d):
    b = d["f_B_mean"] - d["target_mean"]; r = d["prop_rmse_eval"]
    return math.sqrt(max(r*r - b*b, 0.0)), b

trees = {"seed1":"results/sweep", "seed2":"results/sweep_v2_seed2"}
data = {}
for tag, tree in trees.items():
    ung, cells = {}, []
    for p in glob.glob(os.path.join(ROOT, tree, "*.json")):
        d = load(p)
        if d is None: continue
        prop, arm, tgt = d.get("prop"), d.get("arm"), d.get("target_name")
        if arm is None or prop is None: continue
        if arm == "unguided":
            sd, b = sd_of(d); ung[(prop, tgt)] = (sd, d["mol_stability"], b/d["delta"])
        else:
            cells.append((p, d))
    data[tag] = (ung, cells)

rows = defaultdict(dict)
for tag, (ung, cells) in data.items():
    for p, d in cells:
        prop, tgt = d["prop"], d.get("target_name")
        if (prop, tgt) not in ung:                       # need matched unguided
            continue
        if tgt not in ("q50","q90"): continue
        sd, b = sd_of(d)
        u_sd, u_mol, u_bd = ung[(prop, tgt)]
        key = (prop, d["arm"], tgt, str(d.get("w_applied")), str(d.get("variant")), str(d.get("t_min_guide")))
        rows[key][tag] = dict(ratio=sd/u_sd, mol=d["mol_stability"], floor=0.9*u_mol,
                              bias_d=b/d["delta"], u_bias_d=u_bd, ib=d["in_band_fraction"],
                              nonfin=d.get("n_nonfinite",0), clip=d.get("clipped_sample_steps",0),
                              prop=prop)
both = {k:v for k,v in rows.items() if len(v)==2}
print("keys present in both seed trees:", len(both))

def widen_clear(v):   # widens and clears the floor on this seed
    return v["ratio"] > 1.0 and v["mol"] >= v["floor"] and v["nonfin"]==0
c_any = [k for k,v in both.items() if all(widen_clear(x) for x in v.values())]
print("[S1] widen(>1.0) + floor on BOTH seeds:", len(c_any))
c_2se = [k for k in c_any if all((both[k][t]["ratio"]-1.0) >= 2*SE[(both[k][t]["prop"],512)] for t in both[k])]
print("[S2] ... and >= 2 se on BOTH seeds:", len(c_2se))
for k in sorted(c_2se):
    v = both[k]
    print("      ", k)
    for t in ("seed1","seed2"):
        x=v[t]; print("        %s ratio %.3f (z %+.2f) mol %.4f floor %.4f bias %+.2f d (ung %+.2f) in_band %.4f clip %d"
              %(t,x["ratio"],(x["ratio"]-1)/SE[(x["prop"],512)],x["mol"],x["floor"],x["bias_d"],x["u_bias_d"],x["ib"],x["clip"]))
c_bias = [k for k in c_2se if all(abs(both[k][t]["bias_d"]-both[k][t]["u_bias_d"]) <= 2.0 for t in both[k])]
print("[S3] ... and mean move <= 2 delta on BOTH seeds:", len(c_bias), c_bias)

# arms that can go BOTH ways (some cell <0.95, some cell >1.05) within one (prop,target)
dirs = defaultdict(lambda: [1e9,-1e9])
for k,v in rows.items():
    if "seed1" not in v: continue
    prop, arm, tgt = k[0], k[1], k[2]
    r = v["seed1"]["ratio"]
    d = dirs[(prop,arm,tgt)]
    d[0]=min(d[0],r); d[1]=max(d[1],r)
both_ways = {k:v for k,v in dirs.items() if v[0]<0.95 and v[1]>1.05}
print("\n[S4] (prop,arm,target) blocks whose w-ladder spans BOTH <0.95 and >1.05 (seed1):", len(both_ways))
for k,v in sorted(both_ways.items()): print("      %-28s min %.3f max %.3f"%(str(k),v[0],v[1]))
# per-unit mean displacement for the real counterexamples
print("\n[S5] delta of mean move per unit of sd ratio (seed1, alpha q50):")
for arm in ("rch","dflow","btvg_var","plug"):
    pts=[(v["seed1"]["ratio"],v["seed1"]["bias_d"],v["seed1"]["u_bias_d"]) for k,v in rows.items()
         if k[0]=="alpha" and k[1]==arm and k[2]=="q50" and "seed1" in v]
    if not pts: continue
    pts=sorted(pts)
    lo,hi=pts[0],pts[-1]
    rng=hi[0]-lo[0]
    if rng>1e-6:
        print("      %-9s ladder sd %.3f->%.3f  bias %+.2f->%+.2f d  => %.1f delta per unit sd"
              %(arm,lo[0],hi[0],lo[1],hi[1],abs(hi[1]-lo[1])/rng))
