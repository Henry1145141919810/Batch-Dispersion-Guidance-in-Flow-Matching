"""GO/NO-GO gate for Modality 2: does NAIVE linear simplex flow matching work?

Dirichlet Flow Matching (Stark et al., arXiv 2402.05841) exists because linear
paths on the simplex underperform. That is not the same as failing. BDG is a
GUIDANCE rule that sits on top of whatever base exists, so the base does not
need to be competitive -- it needs three things, and this script tests exactly
those three and nothing else:

  G1 DECODES        median max-p per position > 0.5
                    A mushy state means the path collapsed toward the simplex
                    interior and argmax decoding is meaningless.
  G2 SPREAD EXISTS  sd of generated GC within 2x of the data's own sd
                    BDG steers the batch spread of a property. If the generated
                    GC spread is ~0 there is nothing to steer and the transfer
                    experiment cannot run.
  G3 LEARNED        3-mer JS divergence(generated, real) < JS(uniform, real)
                    The weakest possible "it learned something" bar.

Pass all three -> build Modality 2 on this. Fail -> switch to low-res images,
where mean brightness is the exact analogue of GC content, and lose one hour
rather than one day.

NAIVE BY DESIGN. Source = Dirichlet(1) (uniform on the simplex); path is the
straight line x_t = t*x_1 + (1-t)*x_0, which stays inside because the simplex is
convex; target velocity is x_1 - x_0; Euler sampling with a clamp-and-renormalise
projection after each step. No Dirichlet path, no Fisher-Rao metric, no
Gumbel-Softmax. If this clears the gate, those are upgrades, not requirements.

Run: python proj1/m2/gate.py [--steps N] [--crop L]
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

FASTA = ("<path redacted>"
         "enhancer_data/KC_regions.fa")
IDX = {c: i for i, c in enumerate("ACGT")}


# ---------------------------------------------------------------- data
def load_seqs(path, crop):
    seqs, cur = [], []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line.strip().upper())
    if cur:
        seqs.append("".join(cur))
    return [s[:crop] for s in seqs if len(s) >= crop]


def one_hot(seqs, L):
    """[N, L, 4]. Any non-ACGT letter becomes the uniform distribution, which is
    the honest simplex encoding of 'unknown base' rather than a silent A."""
    x = torch.zeros(len(seqs), L, 4)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            k = IDX.get(c)
            if k is None:
                x[i, j] = 0.25
            else:
                x[i, j, k] = 1.0
    return x


def gc_hard(x):
    """GC of the argmax-decoded sequence. EXACT -- this is the f_B analogue."""
    tok = x.argmax(-1)
    return ((tok == 1) | (tok == 2)).float().mean(-1)


def gc_soft(x):
    """GC read off the simplex coordinates. Differentiable -- the f_A analogue."""
    return x[..., 1:3].sum(-1).mean(-1)


def kmer_freq(x, k=3):
    tok = x.argmax(-1)                                  # [N, L]
    B, L = tok.shape
    pw = torch.tensor([4 ** i for i in range(k)])
    idx = sum(tok[:, i:L - k + 1 + i] * pw[i] for i in range(k))
    cnt = torch.zeros(4 ** k)
    cnt.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return cnt / cnt.sum()


def js(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + eps).log() - (b + eps).log())).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


# ---------------------------------------------------------------- model
class TimeEmb(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        a = t[:, None] * f[None] * 1000.0
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class SimplexFM(nn.Module):
    """Dilated 1D CNN velocity field. Small on purpose: this is a gate, not a
    submission. Dilations give a ~100bp receptive field so 3-mer structure is
    reachable."""
    def __init__(self, L, hidden=128, layers=6):
        super().__init__()
        self.temb = TimeEmb(hidden)
        self.inp = nn.Conv1d(4, hidden, 5, padding=2)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            d = 2 ** (i % 4)
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2 * d, dilation=d),
                nn.GroupNorm(8, hidden), nn.SiLU(),
                nn.Conv1d(hidden, hidden, 1)))
        self.out = nn.Conv1d(hidden, 4, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        h = self.inp(x.transpose(1, 2))
        te = self.temb(t)[:, :, None]
        for blk in self.blocks:
            h = h + blk(h + te)
        return self.out(h).transpose(1, 2)


def to_simplex(x):
    """Project back after an Euler step. The naive fix: clamp negatives, renormalise."""
    x = x.clamp(min=0.0)
    return x / x.sum(-1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def sample(net, n, L, steps, gen):
    x = torch.distributions.Dirichlet(torch.ones(4)).sample((n, L))
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt)
        x = to_simplex(x + dt * net(x, t))
    return x


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n-sample", type=int, default=512)
    ap.add_argument("--solver-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260921)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    seqs = load_seqs(FASTA, a.crop)
    X = one_hot(seqs, a.crop)
    print("data: %d seqs x %d bp" % tuple(X.shape[:2]))
    real_gc = gc_hard(X)
    print("real GC: mean %.4f  sd %.4f" % (real_gc.mean(), real_gc.std()))

    net = SimplexFM(a.crop)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    dir1 = torch.distributions.Dirichlet(torch.ones(4))
    print("model: %.2fM params | training %d steps" % (npar / 1e6, a.steps))

    t0, run = time.time(), 0.0
    for it in range(1, a.steps + 1):
        i = torch.randint(0, X.shape[0], (a.batch,))
        x1 = X[i]
        x0 = dir1.sample((a.batch, a.crop))
        t = torch.rand(a.batch)
        xt = t[:, None, None] * x1 + (1 - t)[:, None, None] * x0
        loss = F.mse_loss(net(xt, t), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step()
        run += float(loss)
        if it % 500 == 0:
            print("  it %5d  loss %.5f  %.1f min" % (it, run / 500, (time.time() - t0) / 60))
            run = 0.0

    net.eval()
    g = torch.Generator().manual_seed(a.seed)
    Xg = sample(net, a.n_sample, a.crop, a.solver_steps, g)

    conf = Xg.max(-1).values
    gen_gc = gc_hard(Xg)
    Xr = torch.distributions.Dirichlet(torch.ones(4)).sample((a.n_sample, a.crop))
    kr, kg, ku = kmer_freq(X), kmer_freq(Xg), kmer_freq(Xr)
    js_gen, js_uni = js(kg, kr), js(ku, kr)

    print("\n" + "=" * 62)
    g1 = float(conf.median()); p1 = g1 > 0.5
    print("G1 DECODES       median max-p = %.4f   (need >0.50)   %s"
          % (g1, "PASS" if p1 else "FAIL"))
    g2 = float(gen_gc.std()); lo, hi = float(real_gc.std()) / 2, float(real_gc.std()) * 2
    p2 = lo <= g2 <= hi
    print("G2 SPREAD        gen GC sd = %.4f   (need %.4f-%.4f)  %s"
          % (g2, lo, hi, "PASS" if p2 else "FAIL"))
    p3 = js_gen < js_uni
    print("G3 LEARNED       JS(gen,real) = %.5f  <  JS(unif,real) = %.5f  %s"
          % (js_gen, js_uni, "PASS" if p3 else "FAIL"))
    print("=" * 62)
    print("gen GC: mean %.4f  sd %.4f  (real: %.4f / %.4f)"
          % (gen_gc.mean(), gen_gc.std(), real_gc.mean(), real_gc.std()))
    print("VERDICT: %s" % ("GO -- build Modality 2 on this naive base"
                           if (p1 and p2 and p3) else
                           "NO-GO -- switch to low-res images"))
    return 0 if (p1 and p2 and p3) else 1


if __name__ == "__main__":
    raise SystemExit(main())
