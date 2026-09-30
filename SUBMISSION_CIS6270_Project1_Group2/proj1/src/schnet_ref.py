"""Pretrained SchNet as an INDEPENDENT reference predictor.

Loads the SchNet checkpoints released with the original paper
(quantum-machine.org), which cover QM9 targets 0 (dipole moment), 1
(isotropic polarisability) and 4 (HOMO-LUMO gap) among others, and exposes them
behind our f(coords, feats, mask) signature.

WHY THIS FILE EXISTS, AND WHAT IT CANNOT DO
-------------------------------------------
Its training split is published alongside the weights, and we measured the
overlap against ours directly (indices are into the same 133,885-molecule list,
in the same order):

    our train_a   74.8% seen by SchNet
    our train_b   74.8% seen by SchNet
    our test      74.2% seen by SchNet

So this model **cannot** be f_A or f_B under the disjointness rule in
REQUIREMENTS_FOR_FA_FB.md: whichever role it took, it would share about three
quarters of its partner's training molecules, and the f_A - f_B gap would stop
being a reward-hacking detector.

It has exactly one sound use, and it depends on how the split is chosen:

  * SchNet's own val+test indices (33,885 molecules) are ones it never trained
    on. Train our guide f_A on a subset of THOSE and the pair
    (f_A = ours, f_B = SchNet) has **zero** shared training molecules and two
    different architectures -- strictly better independence than our current
    pair, which shares an architecture. `unseen_indices()` returns that set.

Loading needs no schnetpack install: the released file is a pickled 0.x
AtomisticModel, so we stub the module tree, patch the Module internals newer
PyTorch expects, and copy the tensors into PyTorch Geometric's SchNet.

The dipole model's readout is |sum_i q_i (r_i - r_com)| with mass-weighted
centre, so it is invariant, differentiable and returns Debye.
"""
from __future__ import annotations

import os
import sys
import types
import warnings
import zipfile

import numpy as np
import torch

URL = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"
TARGETS = {"mu": ("qm9_dipole_moment", 0),
           "alpha": ("qm9_isotropic_polarizability", 1),
           "gap": ("qm9_gap", 4)}
# our one-hot order -> atomic number
Z_OF = [1, 6, 7, 8, 9]


def _stub_schnetpack():
    class _Stub(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

        def __setstate__(self, st):
            if isinstance(st, dict):
                object.__setattr__(self, "__dict__", {**self.__dict__, **st})

    class _Mod(types.ModuleType):
        def __getattr__(self, name):
            cls = type(name, (_Stub,), {})
            setattr(self, name, cls)
            return cls

    for mod in ("schnetpack", "schnetpack.representation",
                "schnetpack.representation.schnet", "schnetpack.atomistic",
                "schnetpack.atomistic.output_modules", "schnetpack.atomistic.model",
                "schnetpack.nn", "schnetpack.nn.base", "schnetpack.nn.acsf",
                "schnetpack.nn.blocks", "schnetpack.nn.cfconv", "schnetpack.nn.cutoff",
                "schnetpack.nn.neighbors", "schnetpack.nn.activations",
                "schnetpack.data", "schnetpack.data.atoms"):
        sys.modules.setdefault(mod, _Mod(mod))


_MODULE_DEFAULTS = {
    "_state_dict_pre_hooks": dict, "_load_state_dict_post_hooks": dict,
    "_state_dict_hooks": dict, "_non_persistent_buffers_set": set,
    "_load_state_dict_pre_hooks": dict, "_forward_pre_hooks_with_kwargs": dict,
    "_forward_hooks_with_kwargs": dict, "_forward_hooks_always_called": dict,
    "_backward_pre_hooks": dict,
}


def _patch(m):
    """Old pickles predate several nn.Module internals; add the missing ones."""
    for k, factory in _MODULE_DEFAULTS.items():
        if k not in m.__dict__:
            object.__setattr__(m, k, factory())
    if "_compiled_call_impl" not in m.__dict__:
        object.__setattr__(m, "_compiled_call_impl", None)
    for c in m.__dict__.get("_modules", {}).values():
        if c is not None:
            _patch(c)


def ensure_archive(root="scratch_schnet"):
    os.makedirs(root, exist_ok=True)
    zp = os.path.join(root, "trained_schnet_models.zip")
    if not os.path.exists(zp):
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(URL, context=ctx) as r, open(zp, "wb") as f:
            f.write(r.read())
    return zp


def unseen_indices(prop="mu", root="scratch_schnet"):
    """Molecule indices SchNet never trained on (its own val + test).

    Indices are into the full 133,885-molecule QM9 list in SDF order, the same
    indexing prepare_qm9.py uses, so they can be compared to our splits
    directly. Training our guide inside this set makes SchNet a genuinely
    independent evaluator.
    """
    name = TARGETS[prop][0]
    zp = ensure_archive(root)
    member = "trained_schnet_models/%s/split.npz" % name
    with zipfile.ZipFile(zp) as z:
        z.extract(member, root)
    d = np.load(os.path.join(root, member))
    return (torch.from_numpy(d["train_idx"]).long(),
            torch.cat([torch.from_numpy(d["val_idx"]).long(),
                       torch.from_numpy(d["test_idx"]).long()]))


def load_schnet(prop="mu", root="scratch_schnet", device="cpu"):
    """Return a PyG SchNet with the released weights, in physical units."""
    from torch_geometric.nn.models import SchNet
    import ase.units

    name, target = TARGETS[prop]
    zp = ensure_archive(root)
    member = "trained_schnet_models/%s/best_model" % name
    path = os.path.join(root, member)
    if not os.path.exists(path):
        with zipfile.ZipFile(zp) as z:
            z.extract(member, root)

    _stub_schnetpack()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = torch.load(path, map_location="cpu", weights_only=False)
    _patch(state)

    net = SchNet(hidden_channels=128, num_filters=128, num_interactions=6,
                 num_gaussians=50, cutoff=10.0, dipole=(target == 0),
                 atomref=None)
    net.embedding.weight = state.representation.embedding.weight
    for int1, int2 in zip(state.representation.interactions, net.interactions):
        int2.mlp[0].weight = int1.filter_network[0].weight
        int2.mlp[0].bias = int1.filter_network[0].bias
        int2.mlp[2].weight = int1.filter_network[1].weight
        int2.mlp[2].bias = int1.filter_network[1].bias
        int2.lin.weight = int1.dense.weight
        int2.lin.bias = int1.dense.bias
        int2.conv.lin1.weight = int1.cfconv.in2f.weight
        int2.conv.lin2.weight = int1.cfconv.f2out.weight
        int2.conv.lin2.bias = int1.cfconv.f2out.bias
    om = state.output_modules[0]
    net.lin1.weight = om.out_net[1].out_net[0].weight
    net.lin1.bias = om.out_net[1].out_net[0].bias
    net.lin2.weight = om.out_net[1].out_net[1].weight
    net.lin2.bias = om.out_net[1].out_net[1].bias

    from torch_geometric.nn.aggr import MeanAggregation, SumAggregation
    avg = bool(om.atom_pool.average)
    net.readout = MeanAggregation() if avg else SumAggregation()
    net.mean = om.standardize.mean.item()
    net.std = om.standardize.stddev.item()
    net.atomref = None
    units = [1.0] * 12
    units[0] = ase.units.Debye
    units[1] = ase.units.Bohr ** 3
    net.scale = 1.0 / units[target]
    net.interaction_graph = _DenseInteractionGraph(10.0)
    return net.to(device).eval()


class _DenseInteractionGraph(torch.nn.Module):
    """Radius graph without the C++ extension.

    PyG's radius_graph needs pyg-lib/torch-cluster, which we avoid (a source
    build against a new CUDA). QM9 has at most 29 atoms and SchNet's cutoff is
    10 A, so every pair fits in memory and an exact dense computation is both
    simpler and faster than a neighbour-list kernel at this size.
    """

    def __init__(self, cutoff):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, pos, batch):
        d = torch.cdist(pos, pos)
        same = batch.unsqueeze(0) == batch.unsqueeze(1)
        off_diag = ~torch.eye(pos.shape[0], dtype=torch.bool, device=pos.device)
        keep = same & off_diag & (d <= self.cutoff)
        row, col = keep.nonzero(as_tuple=True)
        return torch.stack([row, col], dim=0), d[row, col]


class SchNetReference(torch.nn.Module):
    """Wraps PyG SchNet behind our f(coords, feats, mask) signature.

    Dense padded tensors are flattened to the (z, pos, batch) form SchNet
    wants, with padded atoms dropped rather than masked -- SchNet builds its
    own radius graph, so a phantom atom at the origin would create real edges.
    """

    def __init__(self, prop="mu", root="scratch_schnet", device="cpu"):
        super().__init__()
        self.net = load_schnet(prop, root, device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.prop = prop
        self.dev = device

    def forward(self, coords, feats, mask):
        B, N, _ = coords.shape
        keep = mask > 0.5
        z = torch.tensor(Z_OF, device=coords.device)[feats.argmax(-1)][keep]
        pos = coords[keep]
        batch = torch.arange(B, device=coords.device).unsqueeze(1).expand(B, N)[keep]
        return self.net(z.long(), pos, batch).view(-1)
