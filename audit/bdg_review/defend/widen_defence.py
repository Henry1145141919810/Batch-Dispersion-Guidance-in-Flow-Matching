import torch, glob, math, itertools, json
import numpy as np
ROOT = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
rng = np.random.default_rng(20260925)

# ---- [D1] bootstrap se of an sd RATIO at n=512 and n=256 from the real unguided f_B pool
pools = {}
for prop in ("mu","alpha","gap"):
    vals = []
    for p in glob.glob(ROOT+f"/results/full/v2/n5000/seed*/{prop}__unguided__q90__w1__tmin0.5__full.permol.pt"):
        d = torch.load(p, map_location="cpu", weights_only=False)
        fb = d["f_B"].double().numpy()
        fin = d["finite"].numpy().astype(bool) if "finite" in d else np.ones_like(fb, bool)
        vals.append(fb[fin])
    pools[prop] = np.concatenate(vals)
print("[D1] pool sizes:", {k: len(v) for k,v in pools.items()})
se = {}
for prop, pool in pools.items():
    for n in (256, 512):
        B = 4000
        a = pool[rng.integers(0, len(pool), size=(B, n))].std(axis=1, ddof=1)
        b = pool[rng.integers(0, len(pool), size=(B, n))].std(axis=1, ddof=1)
        r = a/b
        se[(prop,n)] = float(r.std(ddof=1))
        print(f"[D1] {prop:5s} n={n}: se(sd ratio) = {r.std(ddof=1):.4f}   P(ratio>1.010|null) = {(r>1.010).mean():.3f}")

# ---- [D2] z for every reported widening, incl. the independent local n=256 replication
cells = [
 # (label, prop, ratio, n)
 ("handoff seed1 alpha e4t1.5","alpha",1.034,512),
 ("handoff seed2 alpha e4t1.5","alpha",1.042,512),
 ("local  n256 alpha q50 t1.5","alpha",1.187,256),
 ("local  n256 alpha q90 t1.5","alpha",1.145,256),
 ("handoff seed1 mu   e4t1.5","mu",1.168,512),
 ("handoff seed2 mu   e4t1.5","mu",1.094,512),
 ("local  n256 mu q50 t1.5","mu",1.093,256),
 ("local  n256 mu q90 t1.5","mu",1.219,256),
 ("handoff seed1 gap  e4t1.5","gap",1.096,512),
 ("handoff seed2 gap  e4t1.5","gap",1.117,512),
 ("local  n256 gap q50 t1.5","gap",1.073,256),
 ("local  n256 gap q90 t1.5","gap",1.129,256),
]
byprop = {}
for lab,prop,r,n in cells:
    z = (r-1.0)/se[(prop,n)]
    byprop.setdefault(prop,[]).append(z)
    print(f"[D2] {lab:28s} ratio {r:.3f}  se {se[(prop,n)]:.4f}  z {z:+.2f}")
for prop, zs in byprop.items():
    print(f"[D2] {prop:5s} Stouffer over {len(zs)} runs: z = {sum(zs)/math.sqrt(len(zs)):+.2f}"
          f"   (fully-correlated worst case = mean z {np.mean(zs):+.2f})")

# ---- [D3] the controller test on the red team's own counterexamples
import json as J
def cell(p):
    d = J.load(open(ROOT+"/results/sweep/"+p))
    b = d["f_B_mean"]-d["target_mean"]; r = d["prop_rmse_eval"]
    return dict(sd=math.sqrt(max(r*r-b*b,0)), bias_d=b/d["delta"], ib=d["in_band_fraction"],
                mol=d["mol_stability"], clip=d.get("clipped_sample_steps"), w=d.get("w_applied"))
u = cell("alpha__unguided__q50__w1__tmin0.5__cmp.json")
print("\n[D3] alpha q50 unguided: sd %.3f bias %+.2f d  in_band %.4f mol %.4f clip %d"%(u["sd"],u["bias_d"],u["ib"],u["mol"],u["clip"]))
for name,p in [("rch w0.01","alpha__rch__q50__w0.01__tmin0.5__residual.json"),
               ("rch w0.05","alpha__rch__q50__w0.05__tmin0.5__residual.json"),
               ("rch w0.25","alpha__rch__q50__w0.25__tmin0.5__residual.json"),
               ("rch w1","alpha__rch__q50__w1__tmin0.5__residual.json"),
               ("rch w4","alpha__rch__q50__w4__tmin0.5__residual.json"),
               ("dflow w1","alpha__dflow__q50__w1__tmin0.5__cmp.json"),
               ("dflow w2","alpha__dflow__q50__w2__tmin0.5__cmp.json"),
               ("dflow w4","alpha__dflow__q50__w4__tmin0.5__cmp.json"),
               ("btvg_var w2","alpha__btvg_var__q50__w2__tmin0.5__tau1.json")]:
    c = cell(p)
    print("[D3] %-12s sd/ung %.3f  bias %+7.2f d (move %+6.2f d)  in_band %.4f (%+.4f)  mol %.4f  clip %5d"
          %(name, c["sd"]/u["sd"], c["bias_d"], c["bias_d"]-u["bias_d"], c["ib"], c["ib"]-u["ib"], c["mol"], c["clip"]))

# ---- [D4] monotone-ordering p under a random-ordering null, for the ladder claim
# handoff: 6 curves (3 props x 2 seeds); 4-point ladder for mu, 3-point for alpha/gap
p_h = (1/math.factorial(4))**2 * (1/math.factorial(3))**4
# local replication: 6 curves x 5 points
p_l = (1/math.factorial(5))**6
print("\n[D4] P(all 6 handoff curves monotone | random ordering) = %.3e" % p_h)
print("[D4] P(all 6 local n=256 curves monotone | random ordering) = %.3e" % p_l)
print("[D4] joint (independent runs) = %.3e" % (p_h*p_l))
