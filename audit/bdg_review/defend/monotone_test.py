import json, glob, math, os
from collections import defaultdict
ROOT = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
def load(p):
    try: d=json.load(open(p))
    except Exception: return None
    return d if isinstance(d,dict) and "prop_rmse_eval" in d else None
def sd_of(d):
    b=d["f_B_mean"]-d["target_mean"]; r=d["prop_rmse_eval"]
    return math.sqrt(max(r*r-b*b,0.0)), b
ung={}; lad=defaultdict(list)
for p in glob.glob(os.path.join(ROOT,"results/sweep","*.json")):
    d=load(p)
    if d is None or d.get("prop") is None: continue
    if d.get("t_min_guide")!=0.5: continue
    if d["arm"]=="unguided":
        sd,_=sd_of(d); ung[(d["prop"],d.get("target_name"))]=(sd,d["mol_stability"])
    else:
        sd,b=sd_of(d)
        lad[(d["prop"],d["arm"],d.get("target_name"),str(d.get("variant")))].append(
            (d.get("w_applied"),sd,b/d["delta"],d["mol_stability"],d["in_band_fraction"],d.get("clipped_sample_steps",0)))
blocks=[("alpha","dflow","q50"),("alpha","dflow","q90"),("alpha","plug","q90"),("alpha","smg","q50"),
        ("alpha","smg","q90"),("alpha","lgd_mc","q90"),("alpha","tmpd","q90"),("mu","dflow","q90"),
        ("mu","smg_mean","q50"),("alpha","rch","q50")]
for prop,arm,tgt in blocks:
    for k,v in lad.items():
        if k[:3]!=(prop,arm,tgt): continue
        v=[x for x in v if x[0] is not None]
        if len(v)<4: continue
        v.sort()
        u=ung.get((prop,tgt))
        if u is None: continue
        rats=[x[1]/u[0] for x in v]
        inc=all(b>=a for a,b in zip(rats,rats[1:])); dec=all(b<=a for a,b in zip(rats,rats[1:]))
        turns=sum(1 for i in range(1,len(rats)-1) if (rats[i]-rats[i-1])*(rats[i+1]-rats[i])<0)
        print("%-9s %-9s %-4s %-9s n=%d  monotone=%s turns=%d  ratios=%s"%(
            prop,arm,tgt,k[3],len(rats),("inc" if inc else "dec" if dec else "NO"),turns,
            " ".join("%.3f"%r for r in rats)))
        print("            floor %.4f  mol=%s"%(0.9*u[1]," ".join("%.3f"%x[3] for x in v)))
        print("            bias/d=%s"%" ".join("%+.1f"%x[2] for x in v))
