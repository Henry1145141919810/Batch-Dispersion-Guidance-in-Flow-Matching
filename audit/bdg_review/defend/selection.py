"""Selection-penalty arithmetic for the handoff's eta=1 cells (alpha 33/512 vs 23/512, gap 69/512 vs 58/512).
(1) unpaired two-proportion z, Sidak over M=13, E[max of 13 iid N(0,1)]  (the objection's numbers)
(2) the same with the dependence the design actually has: a COMMON control (Dunnett, rho=0.5) and shared
    initial noise (rho estimated from the 5080 cells as corr of per-molecule difference vectors D_j = ib_j - ib_plugw1)
(3) paired (McNemar-type) z using the measured paired/unpaired variance ratio r from pairing.json
(4) power / sample size, unpaired vs paired.
CPU only. No sampling."""
import numpy as np, json, os, math, glob, torch
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(H, '..', 'cells')
rng = np.random.default_rng(0)
def z2(x1, x0, n):
    p1, p0 = x1/n, x0/n; pp = (x1+x0)/(2*n)
    return (p1-p0)/math.sqrt(pp*(1-pp)*2/n), (p1-p0)/math.sqrt((p1*(1-p1)+p0*(1-p0))/n)
M = 13
res = {}
Zmax_iid = rng.standard_normal((400000, M)).max(1)
print('E[max of %d iid N(0,1)] = %.3f' % (M, Zmax_iid.mean()))
def emax(rho, N=400000):
    z0 = rng.standard_normal(N)[:, None]; zi = rng.standard_normal((N, M))
    return math.sqrt(rho)*z0 + math.sqrt(1-rho)*zi
# (2) empirical rho among difference vectors, per block
rhos = {}
for prop in ['mu', 'alpha', 'gap']:
    for tg in ['q50', 'q90']:
        base = f'{prop}__plug__{tg}__w1__tmin0.5__bdgctl'
        def ib(fn):
            d = torch.load(os.path.join(C, fn+'.permol.pt'), weights_only=False, map_location='cpu')
            j = json.load(open(os.path.join(C, fn+'.json')))
            return (((d['f_B']-d['y']).abs() <= j['delta']) & d['finite']).float().numpy()
        i0 = ib(base)
        arms = [f'{prop}__bdg__{tg}__w1__tmin0.5__e4t{t}' for t in ['0.5', '0.75', '1', '1.25', '1.5']] + [f'{prop}__plug__{tg}__w4__tmin0.5__tgt']
        D = np.stack([ib(a) - i0 for a in arms])
        Cm = np.corrcoef(D); off = Cm[~np.eye(len(arms), dtype=bool)]
        rhos[(prop, tg)] = float(off.mean())
        print(f'{prop} {tg}: mean off-diag corr of D_j over {len(arms)} arms = {off.mean():.3f} (min {off.min():.3f}, max {off.max():.3f})')
rho_emp = float(np.mean([rhos[k] for k in rhos if k[1] == 'q50']))
print('mean rho at q50 = %.3f' % rho_emp)
pairing = json.load(open(os.path.join(H, 'pairing.json')))
def r_of(prop, tg, sel):
    return [p['var_ratio_paired_over_unpaired'] for p in pairing if p['prop'] == prop and p['tgt'] == tg and sel(p['other'])]
for name, x1, x0, prop in [('alpha', 33, 23, 'alpha'), ('gap', 69, 58, 'gap')]:
    n = 512
    zp, zu = z2(x1, x0, n)
    p1 = 1-stats.norm.cdf(zp)
    sidak = 1-(1-p1)**M
    out = dict(z_pooled=round(zp, 3), z_unpooled=round(zu, 3), p_one=round(p1, 4), sidak13=round(sidak, 3),
               frac_null_max_iid_ge_z=round(float((Zmax_iid >= zp).mean()), 3))
    for rho in [0.5, rho_emp]:
        mx = emax(rho).max(1)
        out[f'E_max13_rho{rho:.2f}'] = round(float(mx.mean()), 3)
        out[f'p_max13_ge_z_rho{rho:.2f}'] = round(float((mx >= zp).mean()), 3)
    # paired
    rs_t1 = r_of(prop, 'q50', lambda o: o.endswith('e4t1'))
    rs_bdg = r_of(prop, 'q50', lambda o: o.startswith('bdg__') and 'e0t1' not in o and not o.endswith('o'))
    rs_w4 = r_of(prop, 'q50', lambda o: 'plug' in o and 'w4' in o)
    for lab, rr in [('r_e4t1', rs_t1), ('r_bdg_median', [float(np.median(rs_bdg))]), ('r_bdg_max', [max(rs_bdg)]), ('r_plugw4', rs_w4)]:
        r = rr[0]
        zpair = zu/math.sqrt(r)
        out[lab] = round(r, 3); out['zpaired_'+lab] = round(zpair, 2)
        mx = emax(rho_emp).max(1)
        out['p_max13_zpaired_'+lab] = round(float((mx >= zpair).mean()), 3)
    # power / sample size for d = observed diff, 80% power, one-sided 0.05 and two-sided 0.05
    pa, pb = x1/n, x0/n; d = pa-pb; v = pa*(1-pa)+pb*(1-pb)
    for side, za in [('one', 1.645), ('two', 1.96)]:
        nu = (za+0.8416)**2*v/d**2
        out[f'n80_unpaired_{side}'] = int(round(nu))
        out[f'n80_paired_rmedian_{side}'] = int(round(nu*float(np.median(rs_bdg))))
    # power of the handoff's proposed n=512 rerun (one-sided 0.05) if true diff = observed
    se_u = math.sqrt(v/n); se_p = math.sqrt(float(np.median(rs_bdg))*v/n)
    out['power512_unpaired'] = round(1-stats.norm.cdf(1.645-d/se_u), 3)
    out['power512_paired_rmedian'] = round(1-stats.norm.cdf(1.645-d/se_p), 3)
    res[name] = out
    print(name, json.dumps(out, indent=0))
json.dump(dict(res=res, rhos={f'{k[0]}_{k[1]}': v for k, v in rhos.items()}, rho_emp_q50=rho_emp), open(os.path.join(H, 'selection.json'), 'w'), indent=1)
