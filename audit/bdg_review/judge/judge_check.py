# Judge's independent check: target of BDG's plug w=4 control, and q50/q90 spread-vs-debias lever.
import json, glob, math, os
import numpy as np, torch
os.chdir("C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1")
bdg = {"mu": (0.1309, 0.2930), "alpha": (0.0684, 0.3438), "gap": (0.1602, 0.2812)}
print("== plug w=4 tmin0.5 control vs results/sweep ==")
chi = {}
tg = {}
for q in ["q50", "q90"]:
    c2 = 0
    for p in ["mu", "alpha", "gap"]:
        J = json.load(open(f"results/sweep/{p}__plug__{q}__w4__tmin0.5__cmp.json"))
        def g(d, k):
            if k in d: return d[k]
            for v in d.values():
                if isinstance(v, dict):
                    r = g(v, k)
                    if r is not None: return r
            return None
        ib, ms, n = g(J, "in_band_fraction"), g(J, "mol_stability"), g(J, "n") or 512
        tg.setdefault(q, {})[p] = (g(J, "target"), g(J, "delta"))
        for a, b in [(bdg[p][0], ib), (bdg[p][1], ms)]:
            pp = (a + b) / 2; se = math.sqrt(pp * (1 - pp) * (2 / 512))
            c2 += ((a - b) / se) ** 2
        print(q, p, "sweep in_band %.4f mol %.4f | bdg %.4f %.4f" % (ib, ms, *bdg[p]), "target", tg[q][p])
    print(q, "chi2(6) =", round(c2, 2))

print("\n== lever on v2 unguided sidecars (pooled 3 seeds) ==")
S = np.round(np.arange(0.3, 2.51, 0.01), 2)
for p in ["mu", "alpha", "gap"]:
    for key in ["f_B", "f_B_dec"]:
        fs = []
        for f in sorted(glob.glob(f"results/full/v2/n5000/seed*/{p}__unguided__*.permol.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            fin = d["finite"].numpy().astype(bool)
            fs.append(d[key].numpy()[fin].astype(np.float64))
        f = np.concatenate(fs); n = f.size
        J = json.load(open(glob.glob(f"results/full/v2/n5000/seed20261001/{p}__unguided__*.json")[0]))
        delta = float(J["delta"])
        m, sd = f.mean(), f.std(ddof=1)
        for q in ["q50", "q90"]:
            y = tg[q][p][0]
            b = m - y
            base = np.mean(np.abs(f - y) <= delta)
            deb = np.mean(np.abs(f - m) <= delta)
            curve = np.array([np.mean(np.abs(b + s * (f - m)) <= delta) for s in S])
            i = curve.argmax()
            reach = (S >= 0.65) & (S <= 1.22)
            ir = np.argmax(np.where(reach, curve, -1))
            at065 = curve[np.argmin(abs(S - 0.65))]
            print(f"{p:5s} {key:7s} {q} |b|/sd={abs(b)/sd:.3f} delta/sd={delta/sd:.3f} base={base:.4f} "
                  f"debias+={deb-base:+.4f} best s*={S[i]:.2f} spread+={curve[i]-base:+.4f} "
                  f"reach(.65-1.22) s={S[ir]:.2f} +{curve[ir]-base:+.4f} at0.65={at065:.4f} n={n}")
