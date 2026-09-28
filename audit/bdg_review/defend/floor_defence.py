"""Blue-team check of the 'borrowed floor' objection against BDG_HANDOFF.md s6/s6.2.

CPU only, reads repo results, writes nothing to the repo.

A. Inventory of every unguided cell on the real generator (fm md5 a190ac83):
   which are independent draws, what the pooled population value is, and
   where the load-bearing row (gap e4t0.5, mol_stab 0.3477, n=512) sits
   against each floor convention the project itself uses.
B. Paired design: per-molecule correlation of mol_stable between unguided and a
   guided arm run on the SAME noise (v2 sidecars). Decides whether an own-control
   floor (paired, n=512) or a pooled fixed floor (n=15000) gives the less noisy
   verdict.
C. Probabilities: control-noise flip vs the row's own noise.
D. Would a PASS change s6.2's conclusion? z vs the floor-clearing plug reference,
   and the results/sweep q50 gap plug ladder as a same-weights proxy frontier.
"""
import glob, json, math, os, sys
from collections import defaultdict

ROOT = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
MD5 = "a190ac8394902027d4a951f8d30e8c5c"
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))

# ---------------- A. inventory ----------------
rows = []
for p in glob.glob(os.path.join(ROOT, "results", "**", "*unguided*.json"), recursive=True):
    if "transfer" in p.replace("\\", "/").split("/results/")[1]:
        continue
    try:
        d = json.load(open(p))
    except Exception:
        continue
    if not isinstance(d, dict) or "mol_stability" not in d:
        continue
    prov = d.get("prov") or {}
    md5 = prov.get("fm_md5")
    if md5 != MD5:
        continue
    dev = str(prov.get("device", "?"))
    devtype = "cpu" if "cpu" in dev.lower() else "cuda"
    mask = "dist" if d.get("target_name") == "dist" else "fixed"
    key = (d.get("seed"), d.get("n"), d.get("batch"), devtype, mask, d.get("steps"), d.get("solver"))
    rows.append((key, d["mol_stability"], d.get("prop"), os.path.relpath(p, ROOT), dev))

groups = defaultdict(list)
for key, ms, prop, rel, dev in rows:
    groups[key].append((ms, prop, rel, dev))

print("A. unguided cells on fm md5 a190ac83 (transfer/EDMsecond excluded)")
print(f"   {len(rows)} cells in {len(groups)} (seed,n,batch,devtype,mask,steps,solver) groups")
indep = []
for key, lst in sorted(groups.items(), key=lambda kv: str(kv[0])):
    vals = sorted({round(x[0], 10) for x in lst})
    print(f"   {key}: {len(lst)} cells, distinct mol_stab values {vals}  e.g. {lst[0][2]}")
    if len(vals) == 1:
        indep.append((key, vals[0], key[1]))
    else:
        # different values inside a nominally identical group -> keep each distinct value
        for v in vals:
            indep.append((key, v, key[1]))

# independent draws for the pooled estimate: n=5000 v2 fixed-target (3 seeds) is the
# project's own 'pooled' floor; also build a pool over every distinct draw.
v2 = [(k, v, n) for k, v, n in indep if n == 5000 and k[4] == "fixed"]
allp = indep
def pool(lst):
    num = sum(v * n for _, v, n in lst); den = sum(n for _, _, n in lst)
    p = num / den
    return p, math.sqrt(p * (1 - p) / den), den
p_v2, se_v2, n_v2 = pool(v2)
p_all, se_all, n_all = pool(allp)
print(f"\n   pooled v2 fixed-target n=5000 x{len(v2)}: p = {p_v2:.5f} +/- {se_v2:.5f} (N={n_v2}); floor 0.9p = {0.9*p_v2:.5f}")
print(f"   pooled over every distinct draw ({len(allp)} draws): p = {p_all:.5f} +/- {se_all:.5f} (N={n_all}); floor = {0.9*p_all:.5f}")
# heterogeneity check across independent draws
chi2 = sum(n * (v - p_all) ** 2 / (p_all * (1 - p_all)) for _, v, n in allp)
print(f"   heterogeneity across draws: chi2 = {chi2:.2f} on {len(allp)-1} df")

G, nG = 0.3477, 512           # gap e4t0.5 mol_stab (handoff s6.2), n=512
G_exact = round(G * nG) / nG
print(f"\n   load-bearing row: gap e4t0.5 mol_stab {G} -> {round(G*nG)}/512 = {G_exact:.5f}")
seG = math.sqrt(G_exact * (1 - G_exact) / nG)
floors = {
    "borrowed (0.9 x results/sweep seed-1 B200/128 screen)": 0.9 * 0.40234375,
    "project pooled convention (0.9 x v2 3x5000)": 0.9 * p_v2,
    "pooled over every distinct draw": 0.9 * p_all,
    "objection's seed-2 floor (0.9 x sweep_v2_seed2 B200/128)": 0.9 * 0.369140625,
}
for name, f in floors.items():
    verdict = "PASS" if G_exact >= f else "FAIL"
    print(f"   vs {name}: floor {f:.5f}  margin {G_exact-f:+.5f} = {(G_exact-f)/seG:+.2f} se(G)  -> {verdict}")

# ---------------- B. paired correlation ----------------
print("\nB. per-molecule corr(mol_stable unguided, mol_stable guided) on the SAME noise (v2 sidecars)")
try:
    import torch
    rhos = []
    for sd in sorted(glob.glob(os.path.join(ROOT, "results/full/v2/n5000/seed*"))):
        for prop in ("mu", "alpha", "gap"):
            ug = glob.glob(os.path.join(sd, f"{prop}__unguided__*.permol.pt"))
            if not ug:
                continue
            U = torch.load(ug[0], weights_only=False)
            for p in sorted(glob.glob(os.path.join(sd, f"{prop}__*.permol.pt"))):
                if "__unguided__" in p:
                    continue
                A = torch.load(p, weights_only=False)
                assert torch.equal(A["mol_idx"], U["mol_idx"]) and torch.equal(A["n_atoms"], U["n_atoms"])
                u = U["mol_stable"].float(); a = A["mol_stable"].float()
                r = torch.corrcoef(torch.stack([u, a]))[0, 1].item()
                agree = (u == a).float().mean().item()
                arm = os.path.basename(p).split("__")[1] + "@w" + os.path.basename(p).split("__")[3][1:]
                rhos.append((os.path.basename(sd), prop, arm, r, agree, u.mean().item(), a.mean().item()))
    for s, prop, arm, r, ag, mu_, ma in rhos:
        print(f"   {s} {prop:5s} {arm:14s} rho {r:+.3f}  same-verdict {ag:.3f}  unguided {mu_:.4f} guided {ma:.4f}")
    rs = [r for *_, r, _, _, _ in [(x[0], x[1], x[2], x[3], x[4], x[5], x[6]) for x in rhos]]
    rs = [x[3] for x in rhos]
    print(f"   rho range {min(rs):+.3f} .. {max(rs):+.3f}, median {sorted(rs)[len(rs)//2]:+.3f}")
    # which floor gives the lower-variance verdict?
    # paired: Var(G - 0.9U) = sG^2 + 0.81 sU^2 - 1.8 rho sG sU ; fixed pooled: sG^2 + 0.81 se_pool^2
    sU = math.sqrt(p_v2 * (1 - p_v2) / 512)
    for rho in (min(rs), sorted(rs)[len(rs)//2], max(rs)):
        vp = seG**2 + 0.81 * sU**2 - 1.8 * rho * seG * sU
        vf = seG**2 + 0.81 * se_v2**2
        print(f"   rho {rho:+.3f}: sd(verdict) own-control {math.sqrt(vp):.4f} vs pooled fixed {math.sqrt(vf):.4f}")
    print(f"   break-even rho (own-control as precise as pooled) = {(0.81*sU**2 - 0.81*se_v2**2)/(1.8*seG*sU):.3f}")
except Exception as ex:
    print("   (skipped:", ex, ")")

# ---------------- C. probabilities ----------------
print("\nC. probabilities")
thr = G_exact / 0.9
sU = math.sqrt(p_v2 * (1 - p_v2) / 512)
print(f"   row passes an own-control floor iff U <= {thr:.5f}; P(U <= thr) with U~N({p_v2:.4f},{sU:.4f}) = {Phi((thr-p_v2)/sU):.3f}")
for name, f in floors.items():
    print(f"   P(true mu_G >= {f:.4f} | G observed, flat prior) = {Phi((G_exact - f)/seG):.3f}   [{name}]")

# ---------------- D. would a PASS change s6.2? ----------------
print("\nD. if the row passed: in_band vs floor-clearing plug in the BDG tree (s9.1: 0.1133)")
a, b = 0.1523, 0.1133
ka, kb = round(a * 512), round(b * 512)
pa, pb = ka / 512, kb / 512
pp = (ka + kb) / 1024
z = (pa - pb) / math.sqrt(pp * (1 - pp) * 2 / 512)
p1 = 1 - Phi(z)
print(f"   {ka}/512 vs {kb}/512: diff {pa-pb:+.4f}, z = {z:.2f}, one-sided p = {p1:.3f}, Sidak over 13 cells p = {1-(1-p1)**13:.3f}")
print("   project bar: z >= 3 with Holm within property (FULL_RUN_V2_PROTOCOL V7/V8)")

print("\n   same-weights proxy frontier: results/sweep gap plug q50 tmin0.5 (B200/128, seeds 1 and 2)")
for tree in ("sweep", "sweep_v2_seed2"):
    for p in sorted(glob.glob(os.path.join(ROOT, "results", tree, "gap__plug__q50__w*__tmin0.5__*.json"))):
        d = json.load(open(p))
        print(f"   {tree:15s} w={d['w']:<5} in_band {d['in_band_fraction']:.4f}  mol_stab {d['mol_stability']:.4f}  seed {d['seed']}  {os.path.basename(p)}")
    for p in sorted(glob.glob(os.path.join(ROOT, "results", tree, "gap__unguided__q50__*tmin0.5*.json"))):
        d = json.load(open(p))
        print(f"   {tree:15s} unguided in_band {d['in_band_fraction']:.4f}  mol_stab {d['mol_stability']:.4f}  {os.path.basename(p)}")
