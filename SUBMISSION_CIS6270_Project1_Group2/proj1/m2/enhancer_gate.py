"""A HARD quality constraint for Modality 2, which otherwise has none.

Modality 1 has molecular stability and RDKit validity: crank the guidance and
molecules stop being molecules, visibly. DNA has no such wall -- every string
over ACGT is a legal sequence -- so in-band can be bought with strength and the
only cost is a gradual drift in k-mer statistics. That asymmetry makes M2's
in-band a softer number than M1's, and it is the gap this file closes.

THE GATE. Train a discriminator to separate real enhancers from an ORDER-1
MARKOV chain fitted to those same enhancers. Then report, per guided batch, the
fraction of generated sequences the discriminator calls enhancer-like.

WHY THAT NULL, and it is the whole design. An order-1 Markov chain reproduces
the mononucleotide AND dinucleotide frequencies of the training data by
construction. GC content is a function of mononucleotide frequencies; CpG
density is a function of dinucleotide frequencies. So both properties this
project steers are IDENTICAL in expectation between the two classes, and a
discriminator cannot use either to tell them apart -- it is forced onto
third-order and longer structure, which is where motif content lives.

That independence is what makes it a fair gate rather than a second copy of the
objective. A guidance method could drive GC anywhere it likes without moving
this score, so a drop in the score means the guidance damaged sequence
structure, not that it changed the property it was asked to change.

    python proj1/m2/enhancer_gate.py --train          # fit and save the gate
    python proj1/m2/enhancer_gate.py --score-cells    # score every stored cell
"""
import argparse
import glob
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_fm as S      # noqa: E402

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enhancer_gate.pt")


def per_seq_markov(tok, gen, dev="cpu"):
    """A null matched to EACH sequence's own dinucleotide composition.

    A single global order-1 chain matches the corpus MEAN of GC and CpG but not
    their SPREAD: measured, real GC sd 0.0550 against the null's 0.0231. That
    gap is usable -- |GC - mean| alone separates the two classes at AUC 0.734,
    and a gate trained on it scored +0.46 correlated with |GC - mean|. Such a
    gate would penalise BDG for NARROWING GC spread, which is its mechanism and
    not a defect, so it would be measuring the wrong thing entirely.

    Fitting the chain per sequence fixes that: each negative carries its own
    parent's dinucleotide statistics, so GC and CpG match sequence by sequence
    and their spreads match by construction. What is left for a discriminator
    is third-order and longer structure.
    """
    B, L = tok.shape
    t = tok.to(dev)
    idx = t[:, :-1] * 4 + t[:, 1:]                       # [B, L-1] in 0..15
    cnt = torch.zeros(B, 16, device=dev)
    cnt.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float))
    T = cnt.view(B, 4, 4)
    T = T + 1e-6                                         # unseen transitions
    T = T / T.sum(-1, keepdim=True)
    out = torch.empty(B, L, dtype=torch.long, device=dev)
    out[:, 0] = t[:, 0]                                  # same first base
    for j in range(1, L):
        probs = T[torch.arange(B, device=dev), out[:, j - 1]]
        out[:, j] = torch.multinomial(probs, 1, generator=gen).squeeze(1)
    return out.cpu()


def markov1_fit(tok):
    """Order-1 transition counts, plus the initial-base distribution."""
    a, b = tok[:, :-1].reshape(-1), tok[:, 1:].reshape(-1)
    T = torch.zeros(4, 4, dtype=torch.double)
    T.index_put_((a, b), torch.ones(a.numel(), dtype=torch.double), accumulate=True)
    T = T / T.sum(1, keepdim=True).clamp(min=1)
    p0 = torch.bincount(tok[:, 0], minlength=4).double()
    return T.float(), (p0 / p0.sum()).float()


def markov1_sample(T, p0, n, L, gen):
    """Negatives with the SAME mono- and dinucleotide statistics as the data."""
    out = torch.empty(n, L, dtype=torch.long)
    out[:, 0] = torch.multinomial(p0, n, replacement=True, generator=gen)
    for j in range(1, L):
        out[:, j] = torch.multinomial(T[out[:, j - 1]], 1, generator=gen).squeeze(1)
    return out


class Gate(nn.Module):
    """Small dilated CNN -- the same receptive-field logic as the generator, so
    it can see motif-scale structure without being able to memorise a 500-mer."""

    def __init__(self, h=64, layers=4):
        super().__init__()
        self.inp = nn.Conv1d(4, h, 7, padding=3)
        self.blocks = nn.ModuleList()
        for i in range(layers):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(h, h, 5, padding=2 * d, dilation=d),
                nn.GroupNorm(8, h), nn.GELU()))
        self.head = nn.Linear(2 * h, 1)

    def forward(self, x):                       # x: [B, L, 4] float one-hot
        z = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            z = z + b(z)
        z = torch.cat([z.mean(-1), z.amax(-1)], -1)
        return self.head(z).squeeze(-1)


def train(dev, epochs, bs=256):
    Xa, Xv, _ = S.load_dfb(crop=500)
    tr, va = Xa.argmax(-1), Xv.argmax(-1)
    T, p0 = markov1_fit(tr)
    g = torch.Generator().manual_seed(0)
    gd = torch.Generator(device=dev).manual_seed(0)
    print("PER-SEQUENCE order-1 null. match on what guidance steers (real vs null):")
    neg_v = per_seq_markov(va, gd, dev=dev)
    for name, t in (("real valid", va), ("markov null", neg_v)):
        gc = ((t == 1) | (t == 2)).float().mean(-1)
        cpg = ((t[:, :-1] == 1) & (t[:, 1:] == 2)).float().mean(-1)
        print("   %-12s GC %.4f+/-%.4f   CpG %.4f+/-%.4f"
              % (name, gc.mean(), gc.std(), cpg.mean(), cpg.std()))

    net = Gate().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    n = tr.shape[0]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (n // bs))
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for k in range(n // bs):
            idx = perm[k * bs:(k + 1) * bs]
            pos = tr[idx]
            neg = per_seq_markov(pos, gd, dev=dev)      # paired to THIS batch
            x = F.one_hot(torch.cat([pos, neg]), 4).float().to(dev)
            y = torch.cat([torch.ones(bs), torch.zeros(bs)]).to(dev)
            loss = F.binary_cross_entropy_with_logits(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += float(loss)
        net.eval()
        with torch.no_grad():
            sp = torch.cat([net(F.one_hot(va[i:i + 512], 4).float().to(dev))
                            for i in range(0, va.shape[0], 512)])
            sn = torch.cat([net(F.one_hot(neg_v[i:i + 512], 4).float().to(dev))
                            for i in range(0, neg_v.shape[0], 512)])
        auc = (sp[:, None] > sn[None, :]).float().mean()
        acc = ((sp > 0).float().mean() + (sn <= 0).float().mean()) / 2
        print("  epoch %d  loss %.4f  held-out AUC %.4f  balanced acc %.4f"
              % (ep, tot / max(1, n // bs), auc, acc))
    torch.save({"state_dict": net.state_dict(), "T": T, "p0": p0,
                "null": "per_seq_markov",
                "auc": float(auc), "acc": float(acc)}, CKPT)
    print("wrote %s" % CKPT)

    # THE CHECK THAT MATTERS: the gate must be blind to what guidance steers.
    with torch.no_grad():
        gc = ((va == 1) | (va == 2)).float().mean(-1)
        cpg = ((va[:, :-1] == 1) & (va[:, 1:] == 2)).float().mean(-1)
        alls = torch.cat([sp.cpu(), sn.cpu()])
        gcn = ((neg_v == 1) | (neg_v == 2)).float().mean(-1)
        cpn = ((neg_v[:, :-1] == 1) & (neg_v[:, 1:] == 2)).float().mean(-1)
        mu = float(gc.mean())
        for nm, pr, ng in (("GC", gc, gcn), ("CpG", cpg, cpn),
                           ("|GC-mean|", (gc - mu).abs(), (gcn - mu).abs())):
            c = torch.corrcoef(torch.stack([alls, torch.cat([pr, ng])]))[0, 1]
            print("  corr(gate score, %-10s) across BOTH classes = %+.4f" % (nm, c))
        au = float(((gc - mu).abs()[:, None] > (gcn - mu).abs()[None, :]).float().mean())
        print("  |GC-mean| alone as a classifier: AUC %.4f (0.5 = carries nothing)" % au)
    return net


def load(dev):
    d = torch.load(CKPT, map_location="cpu")
    net = Gate(); net.load_state_dict(d["state_dict"]); net.eval()
    return net.to(dev), d


def score_tokens(net, tok, dev, bs=512):
    out = []
    with torch.no_grad():
        for i in range(0, tok.shape[0], bs):
            x = F.one_hot(tok[i:i + bs].long(), 4).float().to(dev)
            out.append(torch.sigmoid(net(x)).cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--score-cells", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/m2_gate_scores.json")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    if a.train:
        train(dev, a.epochs)
        return 0
    if a.score_cells:
        net, meta = load(dev)
        print("gate: held-out AUC %.4f, balanced acc %.4f" % (meta["auc"], meta["acc"]))
        Xa, _, Xt = S.load_dfb(crop=500)
        ref = {"real_train": score_tokens(net, Xa.argmax(-1)[:4000], dev).mean(),
               "real_test": score_tokens(net, Xt.argmax(-1)[:4000], dev).mean()}
        g = torch.Generator().manual_seed(1)
        neg = markov1_sample(meta["T"], meta["p0"], 4000, 500, g)
        ref["markov_null"] = score_tokens(net, neg, dev).mean()
        print("  reference: real train %.4f  real test %.4f  markov null %.4f"
              % (ref["real_train"], ref["real_test"], ref["markov_null"]))
        res = {"gate_auc": meta["auc"], "reference": {k: float(v) for k, v in ref.items()},
               "cells": {}}
        pts = sorted(glob.glob("results/m2/**/*.permol.pt", recursive=True))
        done = 0
        for p in pts:
            d = torch.load(p, map_location="cpu")
            if "tok" not in d:
                continue
            j = p.replace(".permol.pt", ".json")
            if not os.path.exists(j):
                continue
            c = json.load(open(j))
            s = score_tokens(net, d["tok"], dev)
            res["cells"][os.path.basename(p)] = {
                "arm": c["arm"], "variant": c.get("variant"), "w": c["w"],
                "prop": c["prop"], "seed": c["seed"], "t_min": c.get("t_min_guide"),
                "in_band": c["in_band_fraction"], "kmer_js": c["kmer_js"],
                "gate_mean": float(s.mean()), "gate_pass": float((s > 0.5).float().mean())}
            done += 1
            if done % 25 == 0:
                print("  scored %d cells" % done, flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("scored %d cells -> %s" % (done, a.out))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
