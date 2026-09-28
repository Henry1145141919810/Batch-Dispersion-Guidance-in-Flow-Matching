"""Paired se of the batch-mean shift (in delta units) between an arm and plug w=1 on shared noise, 5080 cells."""
import torch, json, os, numpy as np
C = '../cells'
def ld(s):
    d = torch.load(os.path.join(C, s+'.permol.pt'), weights_only=False, map_location='cpu'); j = json.load(open(os.path.join(C, s+'.json')))
    return d['f_B'].double().numpy(), d['finite'].numpy(), j['delta']
for prop in ['mu', 'alpha', 'gap']:
    for tg in ['q50', 'q90']:
        f0, m0, dl = ld(f'{prop}__plug__{tg}__w1__tmin0.5__bdgctl')
        out = []
        for a in [f'{prop}__plug__{tg}__w4__tmin0.5__tgt', f'{prop}__bdg__{tg}__w1__tmin0.5__e4t0.5']:
            f1, m1, _ = ld(a); m = m0 & m1
            dd = (f1[m] - f0[m]) / dl
            out.append(f'{a.split("__")[1]}{"w4" if "plug" in a else "e4t0.5"}: dbias {dd.mean():+.2f} +/- {dd.std(ddof=1)/np.sqrt(m.sum()):.2f} (z {dd.mean()/(dd.std(ddof=1)/np.sqrt(m.sum())):+.1f})')
        print(prop, tg, ' | '.join(out))
