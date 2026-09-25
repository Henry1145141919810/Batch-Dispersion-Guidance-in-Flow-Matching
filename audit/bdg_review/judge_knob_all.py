import json, math, os, glob, re, collections
R = "results/sweep"
def bs(p):
    j=json.load(open(p)); d=j["delta"]; b=j["f_B_mean"]-j["target_mean"]
    sd=math.sqrt(max(j["prop_rmse_eval"]**2-b**2,0))
    return b/d, sd/d, j.get("mol_stability"), j.get("seed")
groups=collections.defaultdict(list)
for p in glob.glob(R+"/*.json"):
    pb=os.path.basename(p)
    m=re.match(r"(\w+?)__(\w+?)__(q50|q90)__w([0-9.]+)__tmin0\.5(?:__(\w+))?\.json$",pb)
    if not m: continue
    prop,arm,tgt,w,suf=m.groups()
    if arm=="unguided": continue
    try: b,s,mol,seed=bs(p)
    except Exception as e: continue
    groups[(prop,arm,tgt,suf)].append((float(w),b,s,mol))
rows=[]
for k,v in groups.items():
    if len(v)<4: continue
    B=[r[1] for r in v]; S=[r[2] for r in v]
    rb=max(B)-min(B); rs=max(S)-min(S)
    rows.append((k, len(v), rs, rb, rs/max(rb,1e-9)))
rows.sort(key=lambda r:-r[4])
for r in rows:
    if r[0][2]=="q90": print(r[0], "n=",r[1], "dsd=%.2f dbias=%.2f ratio=%.2f"%(r[2],r[3],r[4]))
print("total groups",len(rows))
