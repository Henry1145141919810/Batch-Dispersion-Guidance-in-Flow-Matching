"""Teaching demo: compute the flow-matching training loss by hand.

    python proj1/scripts/demo_fm_loss.py

Walks one training step of proj1/scripts/train_fm.py::fm_loss on four real QM9
molecules (methane first) with the shipped generator, printing every tensor on
the way. Then it checks a hand-written masked loss against the repo's own
fm_loss, drawn from the same random noise and times, and shows what forgetting
the mask would do. CPU only, a few seconds. Writes nothing.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, HERE)
from m1_signed_bias import load_fm  # noqa: E402

_spec = importlib.util.spec_from_file_location("train_fm", os.path.join(HERE, "train_fm.py"))
TF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TF)


def masked_loss(v_c, v_f, u_c, u_f, mask):
    """The slide's loss: ||z||_M^2 with z = v - u, for a whole batch.

    v_c, u_c : [B, 29, 3]  predicted / true velocity, coordinate channels
    v_f, u_f : [B, 29, 5]  predicted / true velocity, atom-type channels
    mask     : [B, 29]     1.0 for a real atom, 0.0 for a padding slot

    Return one number: the coordinate part and the type part, each averaged
    over REAL entries only, added with equal weight.
    """
    # TODO(human)
    raise NotImplementedError("fill in masked_loss")


def main():
    torch.set_grad_enabled(False)
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    idx = torch.tensor([0, 1, 5000, 20000])               # methane first
    x1_c, x1_f, mask = d["coords"][idx], d["feats"][idx], d["mask"][idx]
    print("STEP 1  batch of real molecules")
    print("  coords %s  feats %s  mask %s" % (tuple(x1_c.shape), tuple(x1_f.shape), tuple(mask.shape)))
    print("  real atoms per molecule:", mask.sum(1).int().tolist(), " (the rest of the 29 slots are padding)")
    print("  methane mask:", mask[0].int().tolist())

    fm = next(p for p in (os.path.join(ROOT, "betty_pull", "fm_last.pt"),
                          os.path.join(ROOT, "proj1", "checkpoints", "fm_last.pt"),
                          os.path.join(ROOT, "weights", "fm_ema.pt")) if os.path.exists(p))
    net, _ = load_fm(fm, x1_f.shape[-1], "cpu")

    # The same random draws, in the same order, that fm_loss makes internally.
    torch.manual_seed(0)
    t = torch.rand(len(idx))
    m = mask.unsqueeze(-1)
    eps_c = TF.sample_noise(x1_c, mask)                    # x_0, coordinates (re-centred)
    eps_f = torch.randn_like(x1_f) * m                     # x_0, type channels
    print("\nSTEP 2  one random time per molecule: t =", [round(float(v), 3) for v in t])

    tb = t.view(-1, 1, 1)
    xt_c = (tb * x1_c + (1 - tb) * eps_c) * m              # STEP 4: the noisy state
    xt_f = (tb * x1_f + (1 - tb) * eps_f) * m
    u_c = (x1_c - eps_c) * m                               # STEP 5: true velocity
    u_f = (x1_f - eps_f) * m
    v_c, v_f = net(xt_c, xt_f, mask, t)                    # STEP 6: the network's guess

    print("\nSTEPS 3-6  methane, atom 0 (a carbon) vs padding slot 28")
    print("  real one-hot type      ", x1_f[0, 0].tolist())
    print("  noisy type at t=%.3f   " % float(t[0]), [round(float(v), 2) for v in xt_f[0, 0]])
    print("  true type velocity  u^f", [round(float(v), 2) for v in u_f[0, 0]])
    print("  predicted velocity  v^f", [round(float(v), 2) for v in v_f[0, 0]])
    print("  padding slot: x_t %s  u %s  v %s  (all zero: the mask did its job)" % (
        xt_c[0, 28].tolist(), u_c[0, 28].tolist(), [round(float(v), 6) for v in v_c[0, 28]]))

    mine = masked_loss(v_c, v_f, u_c, u_f, mask)
    torch.manual_seed(0)                                   # replay the same draws
    t2 = torch.rand(len(idx))
    ref, ref_c, ref_f = TF.fm_loss(net, x1_c, x1_f, mask, t2, 1.0)
    print("\nSTEP 7  the loss")
    print("  yours          %.6f" % float(mine))
    print("  repo fm_loss   %.6f   (coord part %.6f + type part %.6f)" % (float(ref), float(ref_c), float(ref_f)))
    print("  match:", "YES" if abs(float(mine) - float(ref)) < 1e-6 else "NO -- check the denominators")

    naive = ((v_c - u_c) ** 2).mean() + ((v_f - u_f) ** 2).mean()
    print("\n  forgetting the mask (averaging over all 29 slots): %.6f" % float(naive))
    print("  -> %.0f%% too small: %d of the %d slots here are padding, contributing zero error but"
          % (100 * (1 - float(naive) / float(ref)), int((1 - mask).sum()), mask.numel()))
    print("     inflating the denominator. Small molecules would look better than they are.")


if __name__ == "__main__":
    main()
