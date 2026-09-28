"""DeepFlyBrain in PyTorch: a real activity predictor for our generated enhancers.

WHY. Every M2 quality metric so far measures statistical RESEMBLANCE -- k-mer
divergence, decode confidence, and the composition-controlled realism gate in
enhancer_gate.py. None of them says whether a sequence would actually behave as
a regulatory element. Modality 1 has RDKit validity and molecular stability,
which are properties of the object itself; M2 had no equivalent.

DeepFlyBrain (Janssens et al., Nature 2022) is the model trained on the very
corpus our base model learned from: adult fly brain scATAC, Kenyon cells, T
neurons and glia, 81 topics. Its input is (500, 4) one-hot ACGT -- exactly our
sequence format, no resampling or cropping needed. So it scores our samples on
the task the data was collected for.

  weights  zenodo.org/record/5153337  DeepFlyBrain.hdf5  md5 3de4f58d0c541170...
  licence  MIT

WHY A PORT RATHER THAN THE ORIGINAL. The published model needs Keras 2.2.4 on
TensorFlow 1.14 and Python 3.7. Rather than stand that up, the architecture is
rebuilt here from the published JSON and the HDF5 weights are loaded into it.
Two details are easy to get wrong and are handled explicitly:

  * Keras LSTM uses recurrent_activation='hard_sigmoid', which is
    clip(0.2x + 0.5, 0, 1) and NOT the logistic sigmoid torch.nn.LSTM applies.
    The cell is therefore written out by hand, with Keras's [i, f, c, o] gate
    order.
  * Conv1D padding='same' with an EVEN kernel (24) pads asymmetrically in
    Keras: 11 left, 12 right. torch's padding=12 would shift the sequence.

The model is applied to the forward strand and to its reverse complement with
SHARED weights, and the two 256-d branches are concatenated before the output
layer -- that is what `conv1d_1` and `dropout_3` each having two inbound nodes
in the published graph means.

    python proj1/m2/deepflybrain.py --selftest     # real vs shuffled controls
    python proj1/m2/deepflybrain.py --score-cells
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

HDF5 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "weights", "deepflybrain", "DeepFlyBrain.hdf5")


def hard_sigmoid(x):
    """Keras's, not the logistic one: clip(0.2x + 0.5, 0, 1)."""
    return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)


class KerasLSTM(nn.Module):
    """One direction, Keras gate order [i, f, c, o] and hard_sigmoid gates."""

    def __init__(self, n_in, units, reverse=False):
        super().__init__()
        self.units, self.reverse = units, reverse
        self.W = nn.Parameter(torch.zeros(n_in, 4 * units))
        self.U = nn.Parameter(torch.zeros(units, 4 * units))
        self.b = nn.Parameter(torch.zeros(4 * units))

    def forward(self, x):                       # x: [B, T, n_in]
        B, T, _ = x.shape
        u = self.units
        h = x.new_zeros(B, u)
        c = x.new_zeros(B, u)
        xs = x.flip(1) if self.reverse else x
        z_x = xs @ self.W + self.b              # precompute the input term
        out = []
        for t in range(T):
            z = z_x[:, t] + h @ self.U
            i = hard_sigmoid(z[:, :u])
            f = hard_sigmoid(z[:, u:2 * u])
            g = torch.tanh(z[:, 2 * u:3 * u])
            o = hard_sigmoid(z[:, 3 * u:])
            c = f * c + i * g
            h = o * torch.tanh(c)
            out.append(h)
        y = torch.stack(out, 1)
        return y.flip(1) if self.reverse else y


class DeepFlyBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 1024, 24)      # padding applied manually
        self.td = nn.Linear(1024, 128)
        self.fwd_lstm = KerasLSTM(128, 128, reverse=False)
        self.bwd_lstm = KerasLSTM(128, 128, reverse=True)
        self.dense2 = nn.Linear(10496, 256)
        self.dense3 = nn.Linear(512, 81)

    def branch(self, x):                        # x: [B, 500, 4]
        z = F.pad(x.transpose(1, 2), (11, 12))  # Keras 'same', even kernel
        z = F.relu(self.conv(z))                # [B, 1024, 500]
        z = F.max_pool1d(z, 12, 12)             # [B, 1024, 41]
        z = z.transpose(1, 2)                   # [B, 41, 1024]
        z = F.relu(self.td(z))                  # [B, 41, 128]
        z = torch.cat([self.fwd_lstm(z), self.bwd_lstm(z)], -1)   # [B, 41, 256]
        z = z.flatten(1)                        # [B, 10496]
        return F.relu(self.dense2(z))           # [B, 256]

    def forward(self, x):
        rc = x.flip(1).flip(2)                  # reverse + complement (ACGT->TGCA)
        return torch.sigmoid(self.dense3(
            torch.cat([self.branch(x), self.branch(rc)], -1)))


def load(dev="cuda", path=HDF5):
    import h5py
    f = h5py.File(path, "r")
    g = lambda p: torch.tensor(f[p][()])                        # noqa: E731
    m = DeepFlyBrain()
    with torch.no_grad():
        m.conv.weight.copy_(g("conv1d_1/conv1d_1/kernel:0").permute(2, 1, 0))
        m.conv.bias.copy_(g("conv1d_1/conv1d_1/bias:0"))
        m.td.weight.copy_(g("time_distributed_1/time_distributed_1/kernel:0").T)
        m.td.bias.copy_(g("time_distributed_1/time_distributed_1/bias:0"))
        for tag, cell in (("forward_lstm_1", m.fwd_lstm),
                          ("backward_lstm_1", m.bwd_lstm)):
            p = "bidirectional_1/bidirectional_1/%s/" % tag
            cell.W.copy_(g(p + "kernel:0"))
            cell.U.copy_(g(p + "recurrent_kernel:0"))
            cell.b.copy_(g(p + "bias:0"))
        m.dense2.weight.copy_(g("dense_2/dense_2/kernel:0").T)
        m.dense2.bias.copy_(g("dense_2/dense_2/bias:0"))
        m.dense3.weight.copy_(g("dense_3/dense_3/kernel:0").T)
        m.dense3.bias.copy_(g("dense_3/dense_3/bias:0"))
    return m.eval().to(dev)


def score(m, tok, dev, bs=128):
    """Per-sequence: max topic probability, and the mean over the 81 topics."""
    mx, mn = [], []
    with torch.no_grad():
        for i in range(0, tok.shape[0], bs):
            x = F.one_hot(tok[i:i + bs].long(), 4).float().to(dev)
            p = m(x)
            mx.append(p.max(-1).values.cpu()); mn.append(p.mean(-1).cpu())
    return torch.cat(mx), torch.cat(mn)


def selftest(dev):
    """The port has to be validated without TF. A correct DeepFlyBrain scores
    real brain regions far above composition-matched controls; a mis-wired one
    returns noise. That is the check."""
    m = load(dev)
    Xa, _, Xt = S.load_dfb(crop=500)
    tr, te = Xa.argmax(-1), Xt.argmax(-1)
    g = torch.Generator().manual_seed(0)
    n = 1000
    real = te[:n]
    shuf = torch.stack([s[torch.randperm(500, generator=g)] for s in real])
    unif = torch.randint(0, 4, (n, 500), generator=g)
    print("DeepFlyBrain port self-test (max topic probability per sequence)\n")
    print("  %-26s %-10s %-10s %s" % ("sequence set", "mean", "median", "frac > 0.5"))
    res = {}
    for name, t in (("real held-out enhancers", real),
                    ("real, positions shuffled", shuf),
                    ("uniform random", unif),
                    ("real training enhancers", tr[:n])):
        mx, _ = score(m, t, dev)
        res[name] = mx
        print("  %-26s %-10.4f %-10.4f %.3f"
              % (name, mx.mean(), mx.median(), (mx > 0.5).float().mean()))
    ok = res["real held-out enhancers"].mean() > 3 * res["uniform random"].mean()
    print("\n  real >> random by 3x or more: %s" % ("PASS" if ok else "**FAIL**"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--score-cells", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/m2_dfb_activity.json")
    a = ap.parse_args()
    dev = a.device if torch.cuda.is_available() else "cpu"
    if a.selftest:
        return selftest(dev)
    if a.score_cells:
        m = load(dev)
        Xa, _, Xt = S.load_dfb(crop=500)
        ref = {}
        for nm, t in (("real_train", Xa.argmax(-1)[:2000]),
                      ("real_test", Xt.argmax(-1)[:2000])):
            mx, _ = score(m, t, dev)
            ref[nm] = {"max_topic_mean": float(mx.mean()),
                       "frac_active": float((mx > 0.5).float().mean())}
        print("reference:", json.dumps(ref))
        res = {"reference": ref, "cells": {}}
        done = 0
        for p in sorted(glob.glob("results/m2/**/*.permol.pt", recursive=True)):
            d = torch.load(p, map_location="cpu")
            j = p.replace(".permol.pt", ".json")
            if "tok" not in d or not os.path.exists(j):
                continue
            c = json.load(open(j))
            mx, mn = score(m, d["tok"], dev)
            res["cells"][os.path.basename(p)] = {
                "arm": c["arm"], "variant": c.get("variant"), "w": c["w"],
                "prop": c["prop"], "seed": c["seed"], "t_min": c.get("t_min_guide"),
                "in_band": c["in_band_fraction"], "kmer_js": c["kmer_js"],
                "dfb_max_topic": float(mx.mean()),
                "dfb_frac_active": float((mx > 0.5).float().mean())}
            done += 1
            if done % 20 == 0:
                print("  scored %d cells" % done, flush=True)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print("scored %d cells -> %s" % (done, a.out))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
