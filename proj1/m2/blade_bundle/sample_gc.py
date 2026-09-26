"""Generated GC spread of a trained checkpoint (report item 4).

Unguided Euler sampling on the simplex with the same path training used:
x_0 ~ Dirichlet(1), dx/dt = v_theta(x_t, t), projected back with to_simplex
after every step. Reports gc_hard (the evaluator) and gc_soft (the guided
property) across the generated batch, against the real data's spread.
"""
import argparse

import torch

import simplex_fm as S

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="fm_m2_dfb500.pt")
ap.add_argument("--npz", default="dfb500.npz")
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--nfe", type=int, nargs="+", default=[100])
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
net = S.SimplexFM(ck["hidden"], ck["layers"])
net.load_state_dict(ck["state_dict"])
net.to(a.device).eval()
L = ck["crop"]
print("ckpt %s | best_it %s  val %.5f  %.1fM views  complete=%s"
      % (a.ckpt, ck.get("best_it"), ck.get("val_loss", float("nan")),
         ck.get("sample_views", 0) / 1e6, ck.get("complete", True)))

Xa, _, Xt = S.load_npz(a.npz, crop=L)
for name, X in (("train", Xa), ("test", Xt)):
    g = S.gc_hard(X)
    print("real %-5s  GC mean %.4f  sd %.4f" % (name, g.mean(), g.std()))

dir1 = torch.distributions.Dirichlet(torch.ones(4, device=a.device))
for nfe in a.nfe:
    torch.manual_seed(a.seed)
    outs = []
    with torch.no_grad():
        for s in range(0, a.n, a.batch):
            b = min(a.batch, a.n - s)
            x = dir1.sample((b, L))
            for k in range(nfe):
                t = torch.full((b,), k / nfe, device=a.device)
                x = S.to_simplex(x + net(x, t) / nfe)
            outs.append(x.cpu())
    X = torch.cat(outs)
    gh, gs = S.gc_hard(X), S.gc_soft(X)
    print("gen  nfe %3d n %d  gc_hard mean %.4f sd %.4f | gc_soft mean %.4f sd %.4f"
          " | decode conf %.3f"
          % (nfe, X.shape[0], gh.mean(), gh.std(), gs.mean(), gs.std(),
             X.max(-1).values.mean()))
