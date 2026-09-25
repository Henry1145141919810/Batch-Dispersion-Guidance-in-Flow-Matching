"""How correlated are per-molecule in_band outcomes between two arms run on the SAME initial noise?
Uses the GPU-lens cells (fm_ema, RTX 5080, n=256 one batch, seed 20260925). CPU only, reads sidecars.
Output: phi, discordant pairs, McNemar z vs unpaired two-proportion z, and the variance ratio
Var(paired diff)/Var(unpaired diff) = (b+c)/n / (p1q1+p0q0) (approx)."""
import torch, json, glob, os, math
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cells')
def load(fn):
    d = torch.load(os.path.join(C, fn + '.permol.pt'), weights_only=False, map_location='cpu')
    j = json.load(open(os.path.join(C, fn + '.json')))
    ib = ((d['f_B'] - d['y']).abs() <= j['delta']) & d['finite']
    return d, j, ib
out = []
for prop in ['mu', 'alpha', 'gap']:
    for tg in ['q50', 'q90']:
        ref = f'{prop}__plug__{tg}__w1__tmin0.5__bdgctl'
        if not os.path.exists(os.path.join(C, ref + '.json')):
            continue
        d0, j0, ib0 = load(ref)
        others = sorted(set(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(C, f'{prop}__*__{tg}__*.json'))))
        for o in others:
            if o == ref: continue
            d1, j1, ib1 = load(o)
            if j1.get('seed') != j0.get('seed') or len(ib1) != len(ib0):
                continue
            n = len(ib0)
            p0, p1 = ib0.float().mean().item(), ib1.float().mean().item()
            b = int((ib1 & ~ib0).sum()); c = int((ib0 & ~ib1).sum())
            a = ib0.float(); bb = ib1.float()
            phi = torch.corrcoef(torch.stack([a, bb]))[0, 1].item() if a.std() > 0 and bb.std() > 0 else float('nan')
            rfb = torch.corrcoef(torch.stack([d0['f_B'], d1['f_B']]))[0, 1].item()
            vr = ((b + c) / n - (p1 - p0) ** 2) / (p1 * (1 - p1) + p0 * (1 - p0)) if (p1*(1-p1)+p0*(1-p0))>0 else float('nan')
            out.append(dict(prop=prop, tgt=tg, other=o.split('__', 1)[1], n=n, p_plugw1=round(p0, 4), p_other=round(p1, 4),
                            b=b, c=c, phi=round(phi, 3), corr_fB=round(rfb, 3), var_ratio_paired_over_unpaired=round(vr, 3)))
for r in out:
    print(r)
vrs = [r['var_ratio_paired_over_unpaired'] for r in out if 'bdg' in r['other'] and 'e0t1' not in r['other'] and 'o.' not in r['other'] and not r['other'].endswith('o')]
vrs = [v for v in vrs if v == v]
print('\nguided-vs-guided (bdg eta=4, non-identical) variance ratio: n=%d median %.3f min %.3f max %.3f' % (len(vrs), sorted(vrs)[len(vrs)//2], min(vrs), max(vrs)))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pairing.json'), 'w'), indent=1)
