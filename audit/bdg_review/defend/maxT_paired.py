"""Zero-compute test for handoff §9.1: paired, selection-adjusted in_band comparison of every BDG cell in a
family against ONE control cell run on the same initial noise.
  python maxT_paired.py <cell_dir> <prop> <target> <control_stem> [arm_glob]
Reads <stem>.permol.pt sidecars (f_B, y, finite, mol_stable) and <stem>.json (delta).
For each arm j: D_ij = in_band_ij - in_band_i,control. Reports b (arm-only hits), c (control-only hits),
paired z = sum D / sqrt(sum D^2 - (sum D)^2/n)  (McNemar with continuity-free variance),
and FWER-adjusted p for max_j z_j by (a) joint per-molecule sign-flip permutation of the D rows and
(b) max over MVN(0, R) with R the empirical correlation of the D_j columns (Hothorn-Bretz-Westfall).
Floor-clearing flag uses 0.9 x the family's OWN unguided mol_stability if an unguided sidecar is given via
env UNGUIDED=<stem>; otherwise it is not applied."""
import sys, os, glob, json, math, numpy as np, torch
cd, prop, tg, ctl = sys.argv[1:5]
pat = sys.argv[5] if len(sys.argv) > 5 else f'{prop}__bdg__{tg}__*'
def load(stem):
    d = torch.load(os.path.join(cd, stem + '.permol.pt'), weights_only=False, map_location='cpu')
    j = json.load(open(os.path.join(cd, stem + '.json')))
    ib = (((d['f_B'] - d['y']).abs() <= j['delta']) & d['finite']).numpy().astype(float)
    return ib, float(d['mol_stable'].float().mean()), j
i0, ms0, j0 = load(ctl)
floor = None
if os.environ.get('UNGUIDED'):
    floor = 0.9 * load(os.environ['UNGUIDED'])[1]
stems = sorted(set(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(cd, pat + '.json'))))
rows, D = [], []
for s in stems:
    if s == ctl: continue
    i1, ms1, j1 = load(s)
    if j1.get('seed') != j0.get('seed') or len(i1) != len(i0): continue
    dvec = i1 - i0
    if not dvec.any(): continue          # bit-identical arms (eta=0, one-sided) carry no information
    D.append(dvec); rows.append((s, i1.mean(), ms1, int((dvec > 0).sum()), int((dvec < 0).sum())))
D = np.stack(D); n = D.shape[1]
S = D.sum(1); V = (D ** 2).sum(1) - S ** 2 / n
z = S / np.sqrt(np.maximum(V, 1e-12))
rng = np.random.default_rng(0); P = 20000
flips = rng.choice([-1.0, 1.0], size=(P, n))
Sp = flips @ D.T                         # (P, arms): sign flip preserves D^2, so V is unchanged under H0 flips
zp = (Sp - 0) / np.sqrt(np.maximum(((D ** 2).sum(1))[None, :] - Sp ** 2 / n, 1e-12))
mx_perm = zp.max(1)
R = np.corrcoef(D); L = np.linalg.cholesky(R + 1e-9 * np.eye(len(R)))
mx_mvn = (rng.standard_normal((200000, len(R))) @ L.T).max(1)
print(f'control {ctl}: in_band {i0.mean():.4f}  mol_stab {ms0:.4f}  n={n}  floor={floor}')
print(f'arms in family (non-identical): {len(rows)}   mean off-diag corr(D) = {R[~np.eye(len(R),dtype=bool)].mean():.3f}')
out = []
for k, (s, p1, ms1, b, c) in enumerate(rows):
    p_perm = float((mx_perm >= z[k]).mean()); p_mvn = float((mx_mvn >= z[k]).mean())
    ok = '' if floor is None else ('PASS' if ms1 >= floor else 'FAIL')
    print(f'{s.split("__",1)[1]:45s} in_band {p1:.4f} d={p1-i0.mean():+.4f} b={b:3d} c={c:3d} z_paired={z[k]:+.2f}  FWER p: perm {p_perm:.3f} mvn {p_mvn:.3f}  mol {ms1:.4f} {ok}')
    out.append(dict(arm=s, in_band=p1, b=b, c=c, z=float(z[k]), p_perm=p_perm, p_mvn=p_mvn, mol=ms1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'maxT_{prop}_{tg}.json'), 'w'), indent=1)
