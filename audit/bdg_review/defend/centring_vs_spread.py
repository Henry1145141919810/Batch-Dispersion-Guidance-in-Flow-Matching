"""At ANY eta, num_i = (y - F_bar) - w_eff (F_i - F_bar): BDG keeps plug's batch-mean shift at unit gain and changes
ONLY the contraction gain about the batch mean. Plug at weight w scales BOTH by w. Check (1) the identity numerically,
(2) empirically on the 5080 cells whether BDG (eta=4, w=1) moves the batch MEAN less than plug w=4 moves it while
moving the SPREAD comparably -- i.e. whether BDG-vs-plug-w1 is closer to a spread-only contrast than a strength sweep.
Reads the GPU lens's weff_traj.json for run-mean w_eff if present."""
import numpy as np, torch, json, os, glob
H = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(H, '..', 'cells')
rng = np.random.default_rng(1)
err = 0
for _ in range(2000):
    B = rng.integers(2, 600); F = rng.normal(size=B) * rng.uniform(.1, 5); y = rng.normal() * 3
    eta = rng.uniform(0, 8); tau2 = rng.uniform(.05, 5)
    Fb = F.mean(); V = F.var(ddof=1); e = (V - tau2) / tau2; w = 1 + eta * e
    lhs = (y - F) - eta * e * (F - Fb); rhs = (y - Fb) - w * (F - Fb)
    err = max(err, np.abs(lhs - rhs).max() / max(1, np.abs(lhs).max()))
print('identity num_i = (y-Fbar) - w_eff (F_i-Fbar): max rel err %.2e' % err)
weff = {}
wt = os.path.join(H, '..', 'weff_traj.json')
if os.path.exists(wt):
    W = json.load(open(wt))
    for k, v in (W.items() if isinstance(W, dict) else []):
        if isinstance(v, dict) and 'mean' in v: weff[k] = v['mean']
def stats(stem):
    d = torch.load(os.path.join(C, stem + '.permol.pt'), weights_only=False, map_location='cpu')
    j = json.load(open(os.path.join(C, stem + '.json')))
    f = d['f_B'][d['finite']].double().numpy(); y = float(j['target']); dl = j['delta']
    return (f.mean() - y) / dl, f.std(ddof=1) / dl
for prop in ['mu', 'alpha', 'gap']:
    for tg in ['q50', 'q90']:
        b0, s0 = stats(f'{prop}__plug__{tg}__w1__tmin0.5__bdgctl')
        bu, su = stats(f'{prop}__unguided__{tg}__w0__tmin0.5__bdgctl')
        b4, s4 = stats(f'{prop}__plug__{tg}__w4__tmin0.5__tgt')
        line = [f'{prop} {tg}: unguided bias {bu:+.2f} sd {su:.2f} | plug w1 bias {b0:+.2f} sd {s0:.2f} | plug w4 dbias {b4-b0:+.2f} dsd {s4/s0:.3f}']
        for t in ['0.5', '0.75', '1.25', '1.5']:
            b, s = stats(f'{prop}__bdg__{tg}__w1__tmin0.5__e4t{t}')
            line.append(f'e4t{t} dbias {b-b0:+.2f} dsd {s/s0:.3f}')
        print(' | '.join(line))
