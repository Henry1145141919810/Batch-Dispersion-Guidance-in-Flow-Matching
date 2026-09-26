"""Is the base good enough to be the frozen substrate? Unguided samples vs the real
test split, each metric bracketed by a floor (real train vs real test) and
baselines that know nothing about sequence structure.

  - k-mer JS divergence (k = 3, 4, 6), log2, pooled counts
  - base composition and GC spread, across seeds
  - decode confidence (mean max simplex coordinate before argmax)
  - longest homopolymer run (catches degenerate repeats)
  - nearest-train identity (catches memorisation)
"""
import argparse

import torch

import simplex_fm as S

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="fm_m2_dfb500.pt")
ap.add_argument("--npz", default="dfb500.npz")
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--nfe", type=int, default=100)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu")
net = S.SimplexFM(ck["hidden"], ck["layers"])
net.load_state_dict(ck["state_dict"])
net.to(dev).eval()
L = ck["crop"]
Xa, _, Xt = S.load_npz(a.npz, crop=L)
tr_tok, te_tok = Xa.argmax(-1), Xt.argmax(-1)
dir1 = torch.distributions.Dirichlet(torch.ones(4, device=dev))


def sample(n, seed, batch=1024):
    torch.manual_seed(seed)
    outs = []
    with torch.no_grad():
        for s in range(0, n, batch):
            b = min(batch, n - s)
            x = dir1.sample((b, L))
            for k in range(a.nfe):
                t = torch.full((b,), k / a.nfe, device=dev)
                x = S.to_simplex(x + net(x, t) / a.nfe)
            outs.append(x.cpu())
    return torch.cat(outs)


def kmer_hist(tok, k):
    w = 4 ** torch.arange(k - 1, -1, -1)
    idx = tok.unfold(1, k, 1).mul(w).sum(-1).flatten()
    h = torch.bincount(idx, minlength=4 ** k).double()
    return h / h.sum()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda x, y: (x[x > 0] * (x[x > 0] / y[x > 0]).log2()).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def longest_run(tok):
    change = torch.ones_like(tok, dtype=torch.bool)
    change[:, 1:] = tok[:, 1:] != tok[:, :-1]
    best = torch.zeros(tok.shape[0], dtype=torch.long)
    cur = torch.zeros(tok.shape[0], dtype=torch.long)
    for j in range(tok.shape[1]):
        cur = torch.where(change[:, j], torch.ones_like(cur), cur + 1)
        best = torch.maximum(best, cur)
    return best.float()


def nn_identity(q_tok, ref_tok, chunk=8192):
    """Max fraction of aligned positions shared with any reference sequence."""
    q = torch.nn.functional.one_hot(q_tok, 4).flatten(1).to(dev, torch.float16)
    best = torch.zeros(q.shape[0], device=dev)
    for s in range(0, ref_tok.shape[0], chunk):
        r = torch.nn.functional.one_hot(ref_tok[s:s + chunk], 4).flatten(1)
        best = torch.maximum(best, (q @ r.to(dev, torch.float16).T).float().max(1).values)
    return (best / L).cpu()


def describe(name, tok, conf=None):
    comp = torch.bincount(tok.flatten(), minlength=4).double()
    comp /= comp.sum()
    gc = ((tok == 1) | (tok == 2)).float().mean(-1)
    lr = longest_run(tok)
    row = [name, *("%.4f" % c for c in comp), "%.4f" % gc.mean(), "%.4f" % gc.std(),
           "%.1f" % lr.mean(), "%.3f" % (lr >= 20).float().mean()]
    row += ["%.4f" % js(kmer_hist(tok, k), ref_h[k]) for k in (3, 4, 6)]
    row += ["%.3f" % conf if conf is not None else "-"]
    print("%-14s A %s C %s G %s T %s | GC %s sd %s | maxrun %s  P(run>=20) %s"
          " | JS3 %s JS4 %s JS6 %s | conf %s" % tuple(row))


ref_h = {k: kmer_hist(te_tok, k) for k in (3, 4, 6)}
g = torch.Generator().manual_seed(0)
comp = torch.bincount(tr_tok.flatten(), minlength=4).double()
comp /= comp.sum()
floor_tok = tr_tok[torch.randperm(tr_tok.shape[0], generator=g)[:a.n]]
unif_tok = torch.randint(0, 4, (a.n, L), generator=g)
iid_tok = torch.multinomial(comp.float(), a.n * L, replacement=True,
                            generator=g).view(a.n, L)

print("ckpt %s  best_it %d  val %.5f | n %d  nfe %d | reference = real test (%d seqs)"
      % (a.ckpt, ck["best_it"], ck["val_loss"], a.n, a.nfe, te_tok.shape[0]))
describe("real test", te_tok)
describe("real train", floor_tok)
describe("uniform rand", unif_tok)
describe("iid real comp", iid_tok)
gens = []
for seed in a.seeds:
    X = sample(a.n, seed)
    tok = X.argmax(-1)
    gens.append(tok)
    describe("gen seed %d" % seed, tok, float(X.max(-1).values.mean()))

m = 512
print("nearest-train identity (first %d seqs; 1.0 = exact copy of a training seq):" % m)
for name, tok in (("real test", te_tok[:m]), ("gen seed 0", gens[0][:m]),
                  ("iid real comp", iid_tok[:m])):
    idn = nn_identity(tok, tr_tok)
    print("  %-14s mean %.3f  median %.3f  max %.3f  P(>=0.9) %.3f"
          % (name, idn.mean(), idn.median(), idn.max(), (idn >= 0.9).float().mean()))
