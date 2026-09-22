"""Load TFG's released QM9 assets and expose them through our interfaces.

TFG (Ye et al., NeurIPS 2024, "Unified Training-Free Guidance for Diffusion
Models") publishes, for QM9, exactly the three things this project needs from
an outside source:

  EDMsecond          an unconditional EDM trained on ONE HALF of QM9 -- the
                     base model, net(coords, feats, mask, tau) -> eps
  tf_predict_<p>     a time-dependent EGNN property network -- the GUIDE
  evaluate_<p>       an independent EGNN property classifier -- the ORACLE

for p in {mu, alpha, gap}, the same three properties we already study. Taking
all three from TFG means no component of the transfer experiment is ours: we
contribute only the guidance field, which is the thing under test.

WHAT EACH NETWORK ACTUALLY OUTPUTS, since neither is a property regressor on
its face:

  * the guide's released weights are an `EnergyDiffusion` whose `forward`
    returns -(phi(x,t) - target)^2, a Gaussian log-likelihood. `phi` -- the
    inner `energy.EGNN` -- is the property prediction, in TFG's (mean, MAD)
    normalised units. We load `dynamics.egnn.*` and call phi directly, because
    our guidance arms need f(m), not log p(y | m).
  * the oracle is EDM's `main_qm9_prop` EGNN classifier, also in normalised
    units.

NEITHER CHECKPOINT SHIPS ITS (mean, MAD). TFG computes them from a dataloader
at runtime and the saved `args.pickle` does not record them. So both networks
are calibrated here by a two-parameter least-squares fit of raw output to the
physical property over real QM9 molecules -- the same procedure, the same
molecules, for guide and oracle alike. Two parameters over thousands of points
cannot invent accuracy, and the fitted slope should recover the published MAD;
`tests/test_transfer_backend.py` asserts that it does.

THE SPLIT CAVEAT, WHICH MUST TRAVEL WITH EVERY NUMBER THIS PRODUCES. Both
saved arg files record `dataset: qm9_second_half`, and TFG's training script
mutates `args.dataset` AFTER building its training loader, so that string
describes an auxiliary test loader, not the split the model trained on.
`audit/fa_fb_search/disjointness_test.py` tests guide-vs-oracle disjointness
from the joint structure of their errors and returns "DISJOINT CONSISTENT" --
strong, but circumstantial. Our own f_A/f_B pair is disjoint BY CONSTRUCTION;
TFG's is disjoint by inference. That is a weakness of the borrowed pair and it
is not fixable from the released artifacts.

SAFE LOADING. Nothing here imports the vendored repository. `definitions()`
parses a file, keeps only the named class and function definitions, and
executes those -- module-level code in the vendored files never runs.
`metadata()` unpickles the arg files through an allowlist admitting only
`argparse.Namespace` and `torch.device`.
"""
from __future__ import annotations

import argparse
import ast
import io
import math
import pickle
from pathlib import Path

import torch
from torch import nn

from .edm_schedule import EDMSchedule

PROPS = ("mu", "alpha", "gap")
PROP_INDEX = {"mu": 0, "alpha": 1, "gap": 2}
PROP_UNITS = {"mu": "D", "alpha": "Bohr^3", "gap": "Ha"}

# QM9 mean-absolute-deviation per property, mean(|y - mean(y)|) over all
# 133,885 molecules in data/qm9.pt. This is the constant EDM and TFG divide
# their targets by -- `compute_mean_mad_from_dataloader` in qm9/utils.py --
# and it is NOT the standard deviation (the two differ by ~0.78, the Gaussian
# ratio sqrt(2/pi)). Quoting the sd here was the first version of this file and
# it made the calibration look 24% wrong when it was right.
#
# Used ONLY as an independent check on the fitted calibration slope: the fit
# sees the property values, never this table, so recovering MAD to ~1% is
# evidence that the adapter is driving these networks correctly.
QM9_MAD = {"mu": 1.1895, "alpha": 6.3003, "gap": 0.0397}
QM9_MEAN = {"mu": 2.7060, "alpha": 75.1913, "gap": 0.2511}


# --------------------------------------------------------------------------
# safe loading of vendored definitions and argument files
# --------------------------------------------------------------------------

class _MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("argparse", "Namespace"): argparse.Namespace,
                   ("torch", "device"): torch.device}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(
                "refusing to unpickle %s.%s from an external args file"
                % (module, name))
        return allowed[module, name]


def metadata(path) -> dict:
    return vars(_MetadataReader(io.BytesIO(Path(path).read_bytes())).load())


def definitions(path, names, namespace=None) -> dict:
    """Execute ONLY the named class/function definitions from a vendored file.

    Where a file defines the same name twice -- `models_property.py` defines
    `EGNN` twice, identically -- the later definition wins, exactly as it
    would on a normal import.
    """
    import numpy as np
    scope = dict(torch=torch, nn=nn, math=math, np=np, F=torch.nn.functional)
    scope.update(namespace or {})
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    kept = [n for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    missing = set(names) - {n.name for n in kept}
    if missing:
        raise KeyError("%s does not define %s" % (path, sorted(missing)))
    tree.body = kept
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def dense_edges(mask: torch.Tensor):
    """EDM's fully-connected edge list for a padded batch, plus its edge mask.

    The edge LIST is every ordered pair including self-loops, byte-identical to
    EDM's own `get_adj_matrix(n_nodes, batch_size)` -- gated by comparison in
    the review of this file. Self-loops are excluded in the MASK, not the list,
    which is exactly what EDM does. Do not "fix" the list to drop them: the
    networks index into it positionally, so removing entries silently breaks
    parity with the released weights.

    They are harmless numerically as well as structurally -- `coord2diff` uses
    `sqrt(radial + 1e-8) + norm_constant`, which is finite at distance 0 -- so
    the mask is the only thing doing work here.
    """
    b, n = mask.shape
    idx = torch.arange(b * n, device=mask.device).reshape(b, n)
    row = idx[:, :, None].expand(b, n, n).reshape(-1)
    col = idx[:, None, :].expand(b, n, n).reshape(-1)
    pm = mask[:, :, None] * mask[:, None, :]
    eye = torch.eye(n, dtype=mask.dtype, device=mask.device)[None]
    pm = pm * (1.0 - eye)
    return [row, col], pm.reshape(-1, 1)


def _egnn_path(args_path) -> Path:
    """Locate the vendored EGNN.py above a checkpoint's args file."""
    p = Path(args_path).resolve()
    for parent in p.parents:
        cand = parent / "tasks" / "networks" / "egnn" / "EGNN.py"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "could not find tasks/networks/egnn/EGNN.py above %s" % args_path)


def _coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    diff = x[row] - x[col]
    radial = (diff ** 2).sum(1).unsqueeze(1)
    return radial, diff / (radial.sqrt() + norm_constant)


# EDM's `equivariant_diffusion/utils.py`, ported. `EGNN.py` IMPORTS these two
# from that module rather than defining them, and the module is not vendored --
# so `definitions()` cannot extract them and `EGNN_dynamics_QM9._forward`
# raises NameError on its very first call. They are injected into the exec
# namespace instead.
#
# The upstream `remove_mean_with_mask` asserts that padded entries are already
# zero, via `.item()`. That is a device synchronisation on every one of the
# ~100 generator calls per sample, in the inner loop of every cell, to check an
# invariant `_forward` establishes three lines earlier (`xh = xh * node_mask`,
# then `vel = (x_final - x) * node_mask`). The assertion is kept as a masked
# multiply that costs nothing and cannot sync; `test_transfer_backend.py` gates
# the padded output at exactly zero, which is the same property.

def _remove_mean(x):
    return x - x.mean(dim=1, keepdim=True)


def _remove_mean_with_mask(x, node_mask):
    x = x * node_mask
    n = node_mask.sum(1, keepdim=True).clamp(min=1.0)
    return (x - x.sum(dim=1, keepdim=True) / n) * node_mask


# --------------------------------------------------------------------------
# the base model
# --------------------------------------------------------------------------

class EDMGenerator(nn.Module):
    """TFG's `EDMsecond` as net(coords, feats, mask, tau) -> (eps_c, eps_f).

    UNITS. Everything this module sees is in EDM's NORMALISED space: the
    sampler's state is (x / norm_values[0], onehot / norm_values[1]), which for
    EDM's QM9 recipe is (x, onehot / 8). That is deliberate -- it is the space
    the network was trained in and the space its epsilon lives in, so no
    rescaling happens on the hot path where a factor could go missing. The
    conversion back to physical coordinates and one-hot features happens once,
    at the boundary, in `denormalise`.

    The 1/8 is not cosmetic. It is the feature scaling our own generator does
    NOT do, and which BASE_MODEL_BENCHMARK.md Section 5 names as the leading
    suspect for our stability gap. Running the arms here therefore also asks
    whether our guidance conclusions survive on a generator whose type channel
    is scaled the published way.
    """

    def __init__(self, weights, args_path, n_types: int = 5, device="cpu"):
        super().__init__()
        a = metadata(args_path)
        self.args = a
        if a.get("include_charges", False):
            raise ValueError(
                "this adapter handles EDM's include_charges=False recipe only; "
                "%s records include_charges=True, whose state has a sixth node "
                "channel our 5-type data cannot fill" % args_path)
        self.norm_values = [float(v) for v in a["normalize_factors"]]
        if self.norm_values[0] != 1.0:
            # Downstream, samples are scored in this module's normalised space
            # on the strength of coordinates being unscaled -- the bond tables
            # are in angstroms. A checkpoint that scales coordinates would
            # break that silently, so refuse it here.
            raise ValueError(
                "normalize_factors[0] = %g, but the transfer pipeline scores "
                "samples in normalised space and assumes coordinates are "
                "unscaled angstroms" % self.norm_values[0])
        self.n_types = n_types
        self.dev = device

        egnn = definitions(
            _egnn_path(args_path),
            {"EGNN_dynamics_QM9", "EGNN", "GCL", "EquivariantBlock",
             "EquivariantUpdate", "SinusoidsEmbeddingNew", "coord2diff",
             "unsorted_segment_sum"},
            {"remove_mean": _remove_mean,
             "remove_mean_with_mask": _remove_mean_with_mask})
        # +1 for the time channel EDM concatenates onto h. EDM's own builder
        # calls this `dynamics_in_node_nf`; the class does NOT add it itself,
        # so omitting it builds a 5-wide input embedding that the released
        # 6-wide weights refuse to load into.
        in_node_nf = n_types + int(a.get("include_charges", False)) + 1
        self.dynamics = egnn["EGNN_dynamics_QM9"](
            in_node_nf=in_node_nf, context_node_nf=a.get("context_node_nf", 0),
            n_dims=3, device=device, hidden_nf=a["nf"], act_fn=torch.nn.SiLU(),
            n_layers=a["n_layers"], attention=a["attention"], tanh=a["tanh"],
            mode=a.get("model", "egnn_dynamics"), norm_constant=a["norm_constant"],
            inv_sublayers=a["inv_sublayers"], sin_embedding=a["sin_embedding"],
            normalization_factor=a["normalization_factor"],
            aggregation_method=a["aggregation_method"])

        state = torch.load(weights, map_location="cpu", weights_only=True)
        sub = {k[len("dynamics."):]: v for k, v in state.items()
               if k.startswith("dynamics.")}
        if not sub:
            raise KeyError(
                "no 'dynamics.*' keys in %s -- this is not an EDM "
                "generative_model checkpoint" % weights)
        self.dynamics.load_state_dict(sub)
        self.dynamics.to(device).eval().requires_grad_(False)
        self.dynamics.device = device
        self.schedule = EDMSchedule.from_args(a, device=device)

    def forward(self, coords, feats, mask, tau):
        """eps-hat in EDM's normalised space. `tau` is [B], in [0, 1]."""
        b, n, _ = coords.shape
        edges, edge_mask = dense_edges(mask)
        xh = torch.cat([coords, feats], dim=-1)
        t = torch.as_tensor(tau, device=coords.device,
                            dtype=coords.dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(b)
        # `_forward` caches an edge list per (n_nodes, batch_size) and builds
        # it on the CPU with a Python double loop. Pre-seeding the cache with
        # ours keeps the device right and skips that loop on every call.
        self.dynamics._edges_dict.setdefault(n, {})[b] = [
            e.to(coords.device) for e in edges]
        out = self.dynamics._forward(t.reshape(b, 1), xh, mask,
                                     edge_mask.reshape(b, n, n), None)
        return out[..., :3], out[..., 3:]

    # -- boundary conversions ---------------------------------------------

    def normalise(self, coords, feats):
        return coords / self.norm_values[0], feats / self.norm_values[1]

    def denormalise(self, coords, feats):
        return coords * self.norm_values[0], feats * self.norm_values[1]


# --------------------------------------------------------------------------
# the property functions
# --------------------------------------------------------------------------

class TFGGuide(nn.Module):
    """`tf_predict_<prop>`: TFG's time-dependent property network, as f_A.

    Returns the RAW normalised prediction. `Calibrated` puts it in physical
    units; nothing else should.

    TIME CHANNEL, FIXED AT ZERO, AND WHY THERE IS NO OPTION. The network takes
    a time scalar as a sixth node feature. Our arms evaluate the property at the
    posterior mean m = E[x_0 | x_tau], which is an estimate of a CLEAN molecule,
    so the time to ask for is t = 0.

    An earlier version of this class carried a `time_mode="current"` switch that
    fed the sampler's tau instead -- what TFG's own method does. It is removed
    rather than fixed, and the reason is worth keeping: it could not have worked
    without two further changes nobody had made. Nothing called its setter, so
    it was bit-identical to t = 0 and would have produced an ablation table
    showing "the choice does not matter"; and the calibration in
    `transfer_sweep.calibrate` is fitted once at t = 0, so a working `current`
    mode would have been reading a t-dependent network through a t = 0
    calibration. Measuring that choice honestly needs a per-t calibration and a
    hook in the sampler. Until someone does that, this file measures t = 0 and
    says so.

    FEATURE UNITS. `feats` arrives ALREADY divided by norm_values[1], because
    the sampler runs in EDM's normalised space, and that is exactly what this
    network expects -- so there is no second division here. Dividing again (the
    obvious way to write it) would feed one-hot values of 1/64 and the guide
    would predict a near-constant property: guidance would still run, and its
    gradient would be tiny but finite, which is a silent failure rather than a
    crash. `test_transfer_backend.py` gates the scale against real molecules.
    """

    def __init__(self, root, prop: str, device="cpu"):
        super().__init__()
        root = Path(root)
        a = metadata(root / ("tf_predict_%s/args_2000.pickle" % prop))
        self.args, self.prop = a, prop
        self.norm_values = [float(v) for v in a["normalize_factors"]]
        if self.norm_values[0] != 1.0:
            # Same guard as EDMGenerator: `_inputs` below passes coordinates
            # through unscaled, which is only right when this factor is 1.
            raise ValueError(
                "tf_predict_%s has normalize_factors[0] = %g; this adapter "
                "passes coordinates through unscaled" % (prop, self.norm_values[0]))
        d = definitions(root / "tasks/networks/egnn/energy.py",
                        {"GCL", "EquivariantBlock", "EquivariantUpdate", "EGNN",
                         "SinusoidsEmbeddingNew", "coord2diff",
                         "unsorted_segment_sum"})
        self.egnn = d["EGNN"](
            in_node_nf=6, in_edge_nf=1, hidden_nf=a["nf"], n_layers=a["n_layers"],
            attention=a["attention"], tanh=a["tanh"],
            norm_constant=a["norm_constant"], inv_sublayers=a["inv_sublayers"],
            sin_embedding=a["sin_embedding"],
            normalization_factor=a["normalization_factor"],
            aggregation_method=a["aggregation_method"])
        state = torch.load(root / ("tf_predict_%s/model_ema_2000.npy" % prop),
                           map_location="cpu", weights_only=True)
        pre = "dynamics.egnn."
        sub = {k[len(pre):]: v for k, v in state.items() if k.startswith(pre)}
        if not sub:
            raise KeyError("no 'dynamics.egnn.*' keys in tf_predict_%s" % prop)
        self.egnn.load_state_dict(sub)
        self.egnn.to(device).eval().requires_grad_(False)

    def _inputs(self, coords, feats, mask):
        b, n, _ = coords.shape
        tcol = torch.zeros((b, n, 1), device=coords.device, dtype=coords.dtype)
        h = (torch.cat([feats, tcol], dim=-1) * mask[..., None]).reshape(-1, 6)
        x = (coords * mask[..., None]).reshape(-1, 3)
        edges, edge_mask = dense_edges(mask)
        return h, x, edges, edge_mask, n

    def forward(self, coords, feats, mask):
        h, x, edges, edge_mask, n = self._inputs(coords, feats, mask)
        return self.egnn(h, x, edges, node_mask=mask.reshape(-1, 1),
                         edge_mask=edge_mask, n_nodes=n)

    def embed(self, coords, feats, mask):
        """Pooled invariant embedding: the guide's readout minus `graph_dec`.

        The diversity metrics need one. Without it `embedding_diversity`
        raises, which is better than the alternative failure mode -- a missing
        `embed` also makes `guidance._diversity_direction` return the zero
        direction, which silently degenerates the `band` arm to no edit at all.
        """
        h, x, edges, edge_mask, n = self._inputs(coords, feats, mask)
        e = self.egnn
        dist, _ = _coord2diff(x, edges)
        if e.sin_embedding is not None:
            dist = e.sin_embedding(dist)
        hh = e.embedding(h)
        for i in range(e.n_layers):
            hh, _ = e._modules["e_block_%d" % i](
                hh, x, edges, node_mask=mask.reshape(-1, 1),
                edge_mask=edge_mask, edge_attr=dist)
        hh = e.node_dec(hh) * mask.reshape(-1, 1)
        return hh.view(coords.shape[0], n, -1).sum(1)


class TFGOracle(nn.Module):
    """`evaluate_<prop>`: EDM's `main_qm9_prop` EGNN classifier, as f_B.

    Takes PHYSICAL coordinates and raw one-hot features, so the caller
    denormalises first. It is the network TFG scores its own published MAE
    with, which is the point: the transfer's property numbers become directly
    comparable to TFG's Table 3 rather than to anything of ours.
    """

    def __init__(self, root, prop: str, device="cpu"):
        super().__init__()
        root = Path(root)
        a = metadata(root / ("evaluate_%s/args.pickle" % prop))
        self.args, self.prop = a, prop
        gcl = definitions(
            root.parent / "OC-Flow/molecule/qm9/property_prediction/models/gcl.py",
            {"E_GCL", "unsorted_segment_sum", "unsorted_segment_mean"})
        d = definitions(
            root.parent / "OC-Flow/molecule/qm9/property_prediction/models_property.py",
            {"E_GCL_mask", "EGNN"}, gcl)
        self.egnn = d["EGNN"](in_node_nf=5, in_edge_nf=0, hidden_nf=a["nf"],
                              n_layers=a["n_layers"], attention=a["attention"],
                              node_attr=a["node_attr"])
        self.egnn.load_state_dict(
            torch.load(root / ("evaluate_%s/best_checkpoint.npy" % prop),
                       map_location="cpu", weights_only=True))
        self.egnn.to(device).eval().requires_grad_(False)
        if a["node_attr"]:
            # node_attr=1 feeds h0 back into every layer, so `embed` below
            # would have to thread it through. The released checkpoints all
            # record node_attr=0; refuse rather than embed the wrong thing.
            raise ValueError("evaluate_%s has node_attr=1, which `embed` below "
                             "does not reproduce" % prop)

    def forward(self, coords, feats, mask):
        b, n, _ = coords.shape
        edges, edge_mask = dense_edges(mask)
        return self.egnn(feats.reshape(-1, 5), coords.reshape(-1, 3), edges,
                         None, mask.reshape(-1, 1), edge_mask, n)

    def embed(self, coords, feats, mask):
        """The oracle's readout minus `graph_dec` -- see `TFGGuide.embed`."""
        b, n, _ = coords.shape
        edges, edge_mask = dense_edges(mask)
        e = self.egnn
        h = e.embedding(feats.reshape(-1, 5))
        for i in range(e.n_layers):
            h, _, _ = e._modules["gcl_%d" % i](
                h, edges, coords.reshape(-1, 3), mask.reshape(-1, 1), edge_mask,
                edge_attr=None, node_attr=None, n_nodes=n)
        h = e.node_dec(h) * mask.reshape(-1, 1)
        return h.view(b, n, -1).sum(1)


# --------------------------------------------------------------------------
# calibration: raw normalised output -> physical units
# --------------------------------------------------------------------------

class Calibrated(nn.Module):
    """A raw TFG network plus the affine map that puts it in physical units.

    Exposes `forward(coords, feats, mask)` and `.net.embed(...)`, which is the
    pair `guidance.py` and `evaluation.py` require of any f_A / f_B, and
    `.y_std`, which the sweep uses to set the likelihood scale.

    `feats_are_normalised` records which space this predictor wants, so the
    driver cannot hand the oracle EDM-scaled features by accident: the guide
    takes onehot/8, the oracle takes raw onehot, and getting that backwards
    produces plausible numbers rather than an error.
    """

    def __init__(self, inner, slope, intercept, prop, y_std,
                 feats_are_normalised: bool, feat_scale: float = 1.0):
        super().__init__()
        self.net = inner
        self.slope, self.intercept = float(slope), float(intercept)
        self.prop = prop
        # `y_std` is the property's own spread, which is what the sweep uses to
        # set the likelihood scale `s`. There is deliberately no `y_mean`:
        # `PhysicalProperty` has one because its network predicts a
        # standardised target, but this wrapper's `slope` is TFG's MAD, not a
        # standard deviation, so `net * y_std + y_mean` is NOT the map applied
        # here and exposing a `y_mean` would invite that reading.
        self.y_std = float(y_std)
        self.feats_are_normalised = bool(feats_are_normalised)
        self.feat_scale = float(feat_scale)

    def _feats(self, feats):
        """Convert from the SAMPLER's space (EDM-normalised) to this net's."""
        return feats if self.feats_are_normalised else feats * self.feat_scale

    def forward(self, coords, feats, mask):
        return self.net(coords, self._feats(feats), mask) * self.slope \
            + self.intercept

    def embed(self, coords, feats, mask):
        return self.net.embed(coords, self._feats(feats), mask)


def fit_calibration(fn, coords, feats, mask, truth, batch=64):
    """Least-squares a*raw + b -> physical units.

    Two parameters over thousands of molecules. Identical treatment for guide
    and oracle, so the comparison between them is unaffected by it.
    """
    raw = []
    with torch.no_grad():
        for i in range(0, coords.shape[0], batch):
            raw.append(fn(coords[i:i + batch].contiguous(),
                          feats[i:i + batch].contiguous(),
                          mask[i:i + batch].contiguous()).reshape(-1))
    raw = torch.cat(raw).double()
    truth = truth.double()
    A = torch.stack([raw, torch.ones_like(raw)], dim=1)
    sol = torch.linalg.lstsq(A, truth.unsqueeze(1)).solution.squeeze(1)
    a, b = float(sol[0]), float(sol[1])
    mae = float((a * raw + b - truth).abs().mean())
    return a, b, mae
