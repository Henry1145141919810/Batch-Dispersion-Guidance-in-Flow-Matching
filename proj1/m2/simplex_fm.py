"""Modality 2 base model: linear flow matching on the probability simplex.

STATE SPACE. A length-L DNA sequence over {A,C,G,T} is L points on the 3-simplex
    Delta^3 = { p in R^4 : p >= 0, sum p = 1 }
with the four letters as its corners. Real sequences sit exactly AT corners, so
the relaxation adds the interior as room to move through without approximating
the data. This is a genuinely different state space from Modality 1's R^3
coordinates: bounded, convex, with the data on its boundary.

PATH. Deliberately the naive one, x_t = t*x_1 + (1-t)*x_0, which stays inside
because the simplex is convex. Dirichlet Flow Matching (Stark et al., arXiv
2402.05841) exists because linear simplex paths underperform, and they do -- but
BDG is a GUIDANCE rule layered on whatever base exists, so "underperforms" is
survivable. proj1/m2/gate.py measured the three properties BDG actually needs and
all three cleared: decode confidence 0.983, generated GC sd 0.0572 against the
data's 0.0668 (86% of it), 3-mer JS 7.6x better than uniform-random sequences.

SHARED WITH MODALITY 1. Same interpolant family, same endpoint estimate
m = x_t + (1-t)*v_theta, same Euler solver, same seeds. Only the state space and
the property net change, which is exactly the claim section 3.6 has to make.

Run: python proj1/m2/simplex_fm.py --steps 20000 --out proj1/m2/ckpt/fm_m2.pt
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

FASTA = ("/vast/projects/pranam/lab/pranam/MOG-DFM/dataset/"
         "enhancer_data/KC_regions.fa")
IDX = {c: i for i, c in enumerate("ACGT")}


# ------------------------------------------------------------------ data
def load_seqs(path=FASTA, crop=200):
    seqs, cur = [], []
    for line in open(path):
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        else:
            cur.append(line.strip().upper())
    if cur:
        seqs.append("".join(cur))
    return [s[:crop] for s in seqs if len(s) >= crop]


def one_hot(seqs, L):
    """[N, L, 4]. Non-ACGT becomes the uniform distribution -- the honest simplex
    encoding of an unknown base, not a silent A."""
    x = torch.zeros(len(seqs), L, 4)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s):
            k = IDX.get(c)
            if k is None:
                x[i, j] = 0.25
            else:
                x[i, j, k] = 1.0
    return x


DFB_PKL = ("/vast/projects/pranam/lab/nnori/hadsbm-hiv/MOG-DFM/dataset/"
           "enhancer_data/DeepFlyBrain_data.pkl")


def load_dfb(path=DFB_PKL, crop=500):
    """The FULL DeepFlyBrain corpus on its OFFICIAL split, already one-hot
    [N, 500, 4] int8 with channel order ACGT (verified: A .2728 ~ T .2720,
    C .2276 ~ G .2276, Watson-Crick symmetric).

    Returns (train, val, test). We keep the published split rather than
    re-partitioning, so our numbers sit on the same test set the baselines
    report on. ~0.02% of positions are all-zero (ambiguous N); those become the
    uniform 0.25 simplex point, the same convention one_hot() uses for the
    fasta path -- an honest unknown base, not a silent A."""
    import pickle
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    out = []
    for key in ("train_data", "valid_data", "test_data"):
        x = torch.from_numpy(d[key][:, :crop].astype("float32"))
        # Exactly 4 of 83,726 training sequences carry ambiguous (all-zero) rows,
        # up to 20% of their length; valid and test carry none. Dropping those 4
        # is cleaner than imputing 0.25, because a 0.25 row reads as GC=0.5 under
        # gc_soft but argmaxes to A (GC=0) under gc_hard -- a 0.1 disagreement
        # between the guided property and the evaluated one. Dropping makes the
        # two agree to machine precision on real data.
        keep = (x.sum(-1) > 0).all(dim=1)
        out.append(x[keep].contiguous())
    return out


def splits(X, seed=20260921):
    """train_a / train_b / val, mirroring Modality 1's split protocol. Only
    train_a is used to fit the generator; the others exist so a learned property
    net could be added later without leaking."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(X.shape[0], generator=g)
    n = X.shape[0]
    a, b = int(0.45 * n), int(0.90 * n)
    return X[idx[:a]], X[idx[a:b]], X[idx[b:]]


# ------------------------------------------------------- the property
def gc_soft(x):
    """GC read off the simplex coordinates. DIFFERENTIABLE -- this is f_A, the
    quantity guidance steers. C is index 1, G is index 2."""
    return x[..., 1:3].sum(-1).mean(-1)


def gc_hard(x):
    """GC of the argmax-decoded sequence. EXACT -- this is the evaluator.

    Note what this buys over Modality 1: f_B there is a trained network with its
    own error, and f_A/f_B disagree 2-3x more on generated molecules than their
    validation MAEs suggest. Here the evaluator is a count. There is no model
    error and no guide-transfer confound, so the gap between guided and scored
    values is purely the DECODING gap."""
    tok = x.argmax(-1)
    return ((tok == 1) | (tok == 2)).float().mean(-1)


# ----------------------------------------------------------------- model
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
    """Dilated 1D CNN velocity field. Dilations 1,2,4,8 repeated give a receptive
    field of ~100 bp at 8 layers, enough for the k-mer structure the fidelity
    metric scores."""

    def __init__(self, hidden=128, layers=8):
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
    """Project back after an Euler step: clamp negatives, renormalise. The naive
    fix, and the one place the simplex geometry needs handling at all."""
    x = x.clamp(min=0.0)
    return x / x.sum(-1, keepdim=True).clamp(min=1e-8)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=500)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--device", default="auto")
    # EARLY STOPPING, not a fixed step budget. Stark et al. train their linear-FM
    # baseline to 40.3M sample-views over 83,968 sequences. We have 6,126, so
    # matching their step count would be ~6,500 epochs of pure memorisation.
    # "Matched in strength" for a smaller dataset means trained until validation
    # stops improving, and reporting where that happened.
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dfb", action="store_true",
                    help="train on the full DeepFlyBrain corpus (83,726 seqs) on its\n                          official split, instead of the 6,126-sequence KC slice")
    ap.add_argument("--out", default="proj1/m2/ckpt/fm_m2_500.pt")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print("device: %s%s" % (dev, " [" + torch.cuda.get_device_name(0) + "]"
                            if dev == "cuda" else ""))

    if a.dfb:
        Xa, Xv, Xt = load_dfb(crop=a.crop)
        print("DeepFlyBrain official split -> train %d  val %d  test %d  (L=%d)"
              % (Xa.shape[0], Xv.shape[0], Xt.shape[0], a.crop))
    else:
        X = one_hot(load_seqs(crop=a.crop), a.crop)
        Xa, Xb, Xv = splits(X, a.seed)
        print("data %s -> train_a %d  train_b %d  val %d"
              % (tuple(X.shape), Xa.shape[0], Xb.shape[0], Xv.shape[0]))
    gc = gc_hard(Xa)
    print("train_a GC: mean %.4f  sd %.4f" % (gc.mean(), gc.std()))

    Xa, Xv = Xa.to(dev), Xv.to(dev)
    net = SimplexFM(a.hidden, a.layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    dir1 = torch.distributions.Dirichlet(torch.ones(4, device=dev))
    print("model %.2fM params | %d steps" % (
        sum(p.numel() for p in net.parameters()) / 1e6, a.steps))

    def val_loss(n_rep=8):
        """Validation loss on held-out sequences, averaged over several noise
        draws so the stochastic target does not dominate the estimate."""
        net.eval(); tot = 0.0
        with torch.no_grad():
            for r in range(n_rep):
                gg = torch.Generator(device=dev).manual_seed(1234 + r)
                x1 = Xv
                x0 = dir1.sample((x1.shape[0], a.crop))
                t = torch.rand(x1.shape[0], device=dev, generator=gg)
                xt = t[:, None, None] * x1 + (1 - t)[:, None, None] * x0
                tot += float(F.mse_loss(net(xt, t), x1 - x0))
        net.train(); return tot / n_rep

    t0, run = time.time(), 0.0
    best_val, bad, best_state, best_it = float("inf"), 0, None, 0
    for it in range(1, a.steps + 1):
        i = torch.randint(0, Xa.shape[0], (a.batch,), device=dev)
        x1 = Xa[i]
        x0 = dir1.sample((a.batch, a.crop))
        t = torch.rand(a.batch, device=dev)
        xt = t[:, None, None] * x1 + (1 - t)[:, None, None] * x0
        loss = F.mse_loss(net(xt, t), x1 - x0)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        run += float(loss.detach())
        if it % a.val_every == 0:
            tr = run / a.val_every; run = 0.0
            vl = val_loss()
            mark = ""
            if vl < best_val - 1e-6:
                best_val, bad, best_it = vl, 0, it
                best_state = {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}
                # WRITE IT NOW. SLURM can kill this job at its wall limit, and a
                # best_state living only in memory would be lost. Every improved
                # checkpoint is durable, so a killed run still yields a usable
                # model and the recorded sample_views say how far it got.
                os.makedirs(os.path.dirname(a.out), exist_ok=True)
                torch.save({"state_dict": best_state, "hidden": a.hidden,
                            "layers": a.layers, "crop": a.crop,
                            "steps": a.steps, "seed": a.seed,
                            "final_loss": best_val, "best_it": it,
                            "sample_views": it * a.batch, "val_loss": vl,
                            "device": dev, "gc_mean": float(gc.mean()),
                            "gc_std": float(gc.std()), "complete": False},
                           a.out)
                mark = " *best, saved*"
            else:
                bad += 1
                mark = "  (no improve %d/%d)" % (bad, a.patience)
            print("  it %6d  train %.5f  val %.5f  lr %.2e  %.1f min  %.1fM views%s"
                  % (it, tr, vl, sch.get_last_lr()[0], (time.time() - t0) / 60,
                     it * a.batch / 1e6, mark))
            if bad >= a.patience:
                print("  EARLY STOP: val has not improved for %d checks. Best was"
                      " it %d (val %.5f, %.1fM sample-views)."
                      % (a.patience, best_it, best_val, best_it * a.batch / 1e6))
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    best = best_val

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()},
                "hidden": a.hidden, "layers": a.layers, "crop": a.crop,
                "steps": a.steps, "seed": a.seed, "final_loss": best,
                "best_it": best_it, "sample_views": best_it * a.batch,
                "val_loss": best_val, "device": dev,
                # the data's own GC scale. This plays the role f_A.y_std plays in
                # Modality 1: it sets s, and tau = tau_mult * s.
                "gc_mean": float(gc.mean()), "gc_std": float(gc.std())},
               a.out)
    print("saved %s  (gc_std %.5f -> this is s)" % (a.out, gc.std()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
