"""Experiment 4: does SMG-2 survive stochastic trace estimation?  d=16, Gaussian prior, f = x^T A x with a
random symmetric A (mixed-sign spectrum). Exact target samples by importance resampling of prior draws.
Hutchinson probes (Rademacher) replace the exact traces c = 0.5 tr(H Sigma), v2 = 0.5 tr((H Sigma)^2)."""
import numpy as np, json, sys
from lib import *

d = 16
rs = np.random.default_rng(7)
Q, _ = np.linalg.qr(rs.standard_normal((d, d)))
ev = np.concatenate([np.linspace(1.5, 0.2, 10), -np.linspace(0.2, 1.0, 6)])
A = (Q * ev) @ Q.T
tg = Target("quadform", lambda x: np.einsum("...i,ij,...j->...", x, A, x), lambda x: 2 * x @ A,
            lambda x: np.broadcast_to(2 * A, x.shape[:-1] + (d, d)))
prior = GMM(np.zeros((1, d)), [1.0], [1.0])
X0 = rs.standard_normal((3_000_000, d)); f0 = tg.f(X0)
y = float(np.quantile(f0, 0.85)); s = 0.3
w = np.exp(-0.5 * (f0 - y) ** 2 / s ** 2); w /= w.sum()
print(f"f under prior: mean {f0.mean():.2f}, sd {f0.std():.2f}; target y={y:.2f}, s={s}; reference ESS = {1/(w**2).sum():.0f}")
ref = X0[rs.choice(len(X0), 3000, p=w)]; f_ref = np.sort(tg.f(ref)[:1500])
del X0, f0

def smg2_hutch(m, S, K, rng):
    g = tg.grad(m); H = 2 * A
    cs, v2s = 0.0, 0.0
    for _ in range(K):
        z = rng.choice([-1.0, 1.0], size=m.shape)
        Hz = z @ H; Sz = np.einsum("nij,nj->ni", S, z)
        HSz = Sz @ H; SHz = np.einsum("nij,nj->ni", S, Hz)
        cs = cs + 0.5 * (Hz * Sz).sum(1) / K
        v2s = v2s + 0.5 * (SHz * HSz).sum(1) / K
    vf = np.einsum("ni,nij,nj->n", g, S, g)
    r = y - tg.f(m) - cs
    return np.einsum("nij,nj->ni", S, (r / (s ** 2 + vf + np.maximum(v2s, 0.0)))[:, None] * g)

M = {
 "SMG var-only":        lambda m,S,x,t,r: est_smg(tg,y,s,m,S,mean=False),
 "SMG (as stated)":     lambda m,S,x,t,r: est_smg(tg,y,s,m,S),
 "SMG-2 exact traces":  lambda m,S,x,t,r: est_smg(tg,y,s,m,S,second_var=True),
 "SMG-2 Hutchinson K=1":lambda m,S,x,t,r: smg2_hutch(m,S,1,r),
 "SMG-2 Hutchinson K=4":lambda m,S,x,t,r: smg2_hutch(m,S,4,r),
 "LGD-MC n=16":         lambda m,S,x,t,r: est_lgd_mc(tg,y,s,m,S,n=16,rng=r),
 "OSC0 n=16":           lambda m,S,x,t,r: est_osc(tg,y,s,m,S,n=16,rng=r,cov_term=False),
}
out = {}
print(f"  {'method':22s} {'eta':>4s} {'ED':>16s} {'W1(f)/s':>9s} {'mean(f-y)/s':>12s} {'sd(f-y)/s':>10s} {'clip':>6s}")
fr = (tg.f(ref) - y) / s
print(f"  {'exact target':22s} {'':4s} {'':16s} {'':9s} {fr.mean():12.3f} {fr.std():10.3f}")
for eta in [0.0, 1.0]:
    for k, fn in M.items():
        E, W, mu, sd, cl = [], [], [], [], []
        for seed in [41, 42, 43]:
            r = np.random.default_rng(seed)
            x, clip = ddim_sample(prior, lambda m,S,xx,t: fn(m,S,xx,t,r), 1500, r, steps=100, eta=eta, clip=50.0)
            f = tg.f(x)
            E.append(energy_distance(x, ref, r)); W.append(np.abs(np.sort(f) - f_ref).mean() / s)
            mu.append(((f - y) / s).mean()); sd.append(((f - y) / s).std()); cl.append(clip)
        out[f"{k}|eta={eta}"] = [float(np.mean(E)), float(np.std(E)), float(np.mean(W)), float(np.mean(mu)), float(np.mean(sd))]
        print(f"  {k:22s} {eta:4.1f} {np.mean(E):8.4f}+-{np.std(E):.4f} {np.mean(W):9.3f} {np.mean(mu):12.3f} {np.mean(sd):10.3f} {np.mean(cl):6.3f}")
        sys.stdout.flush()
json.dump(out, open("exp4_results.json", "w"), indent=1)
