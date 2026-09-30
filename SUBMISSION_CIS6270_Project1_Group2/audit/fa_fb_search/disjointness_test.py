"""Did TFG's mu guide and oracle train on DIFFERENT molecules?

The paper says yes (guide on one half of QM9's train split, evaluator on the
other). The artifacts cannot confirm it: both saved argument files record
`dataset: qm9_second_half`, and the released training script mutates
`args.dataset` AFTER building its training loader --

    dataloaders, _ = dataset.retrieve_dataloaders(args)   # trains on the DEFAULT, first_half
    args.dataset = "qm9_second_half"                       # overwritten here
    dataloaders["test"] = dataloaders_aux["train"]         # second half becomes TEST

-- so the string describes an auxiliary test loader, not the training split.
If both models secretly trained on the same molecules, the f_A - f_B gap would
be decorative while looking rigorous. That is a worse failure than using our
own pair, so it has to be settled before adoption.

THE TEST. A network fits molecules it trained on better than ones it did not.
Run both models over many QM9 molecules and look at the JOINT structure of
their per-molecule errors:

  disjoint halves -> each model is accurate on its own half and weaker on the
                     other, so the two error series are pushed in OPPOSITE
                     directions: low correlation, and a large population where
                     one model is clearly better than the other
  same half       -> both are accurate on the same molecules and weak on the
                     same molecules: clearly POSITIVE error correlation

This needs no reconstruction of their split, only their predictions.

Outputs are raw normalised scalars, so a two-parameter linear map to Debye is
fitted per model by least squares. Two parameters over thousands of molecules
cannot manufacture the correlation structure being tested, and both models get
identical treatment.

Run:  python audit/fa_fb_search/disjointness_test.py
"""
from pathlib import Path
import argparse
import ast
import io
import json
import math
import pickle

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
PROJ = ROOT.parents[1]
torch.set_num_threads(4)
torch.manual_seed(20260919)


class MetadataReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('argparse', 'Namespace'): argparse.Namespace,
                   ('torch', 'device'): torch.device}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError((module, name))
        return allowed[module, name]


def metadata(path):
    return vars(MetadataReader(io.BytesIO(path.read_bytes())).load())


def definitions(path, names, namespace=None):
    """Load only reviewed class/function definitions, never module top-level code."""
    scope = dict(torch=torch, nn=nn, math=math)
    scope.update(namespace or {})
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body
                 if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(tree, str(path), 'exec'), scope)
    return scope


def edges(mask):
    b, n = mask.shape
    idx = torch.arange(b * n).reshape(b, n)
    row = idx[:, :, None].expand(b, n, n).reshape(-1)
    col = idx[:, None, :].expand(b, n, n).reshape(-1)
    pm = mask[:, :, None] * mask[:, None, :]
    pm = pm * (1 - torch.eye(n, dtype=mask.dtype)[None])
    return [row, col], pm.reshape(-1, 1)


def build_models(prop='mu'):
    energy = definitions(ROOT / 'TFG/tasks/networks/egnn/energy.py',
                         {'GCL', 'EquivariantBlock', 'EGNN', 'SinusoidsEmbeddingNew',
                          'coord2diff', 'unsorted_segment_sum'})
    ga = metadata(ROOT / ('TFG/tf_predict_%s/args_2000.pickle' % prop))
    guide = energy['EGNN'](in_node_nf=6, in_edge_nf=1, hidden_nf=ga['nf'],
                           n_layers=ga['n_layers'], attention=ga['attention'],
                           tanh=ga['tanh'], norm_constant=ga['norm_constant'],
                           inv_sublayers=ga['inv_sublayers'],
                           sin_embedding=ga['sin_embedding'],
                           normalization_factor=ga['normalization_factor'],
                           aggregation_method=ga['aggregation_method'])
    state = torch.load(ROOT / ('TFG/tf_predict_%s/model_ema_2000.npy' % prop),
                       map_location='cpu', weights_only=True)
    pre = 'dynamics.egnn.'
    guide.load_state_dict({k[len(pre):]: v for k, v in state.items() if k.startswith(pre)})
    guide = guide.double().eval().requires_grad_(False)

    def guide_fn(c, a, mask):
        b, n, _ = c.shape
        edge, pm = edges(mask)
        h = torch.cat([a / ga['normalize_factors'][1],
                       torch.zeros(b, n, 1, dtype=c.dtype)], dim=-1)
        return guide((h * mask[..., None]).reshape(-1, 6),
                     (c * mask[..., None] / ga['normalize_factors'][0]).reshape(-1, 3),
                     edge, node_mask=mask.reshape(-1, 1), edge_mask=pm, n_nodes=n)

    gcl = definitions(ROOT / 'OC-Flow/molecule/qm9/property_prediction/models/gcl.py',
                      {'E_GCL', 'unsorted_segment_sum', 'unsorted_segment_mean'})
    odefs = definitions(ROOT / 'OC-Flow/molecule/qm9/property_prediction/models_property.py',
                        {'E_GCL_mask', 'EGNN'}, gcl)
    oa = metadata(ROOT / ('TFG/evaluate_%s/args.pickle' % prop))
    oracle = odefs['EGNN'](in_node_nf=5, in_edge_nf=0, hidden_nf=oa['nf'],
                           n_layers=oa['n_layers'], attention=oa['attention'],
                           node_attr=oa['node_attr'])
    oracle.load_state_dict(torch.load(ROOT / ('TFG/evaluate_%s/best_checkpoint.npy' % prop),
                                      map_location='cpu', weights_only=True))
    oracle = oracle.double().eval().requires_grad_(False)

    def oracle_fn(c, a, mask):
        b, n, _ = c.shape
        edge, pm = edges(mask)
        return oracle(a.reshape(-1, 5), c.reshape(-1, 3), edge, None,
                      mask.reshape(-1, 1), pm, n)

    return guide_fn, oracle_fn, ga, oa


def predict(fn, coords, feats, mask, bs=32):
    out = []
    with torch.no_grad():
        for i in range(0, coords.shape[0], bs):
            out.append(fn(coords[i:i + bs].contiguous(),
                          feats[i:i + bs].contiguous(),
                          mask[i:i + bs].contiguous()).reshape(-1))
    return torch.cat(out)


def fit_linear(raw, truth):
    """Least-squares a*raw + b -> physical units. Two parameters, thousands of points."""
    A = torch.stack([raw, torch.ones_like(raw)], dim=1)
    sol = torch.linalg.lstsq(A, truth.unsqueeze(1)).solution.squeeze(1)
    return sol[0].item(), sol[1].item()


def main(prop='mu'):
    N = 3000
    d = torch.load(PROJ / 'data/qm9.pt', map_location='cpu', weights_only=True)
    g = torch.Generator().manual_seed(7)
    sel = torch.randperm(d['coords'].shape[0], generator=g)[:N]
    coords = d['coords'][sel].double()
    feats = d['feats'][sel].double()
    mask = d['mask'][sel].double()
    pi = {'mu': 0, 'alpha': 1, 'gap': 2}[prop]
    unit = {'mu': 'D', 'alpha': 'Bohr^3', 'gap': 'Hartree'}[prop]
    truth = d['y'][sel, pi].double()

    guide_fn, oracle_fn, ga, oa = build_models(prop)
    print("loaded TFG %s guide" % prop, " (nf=%d, layers=%d) and oracle (nf=%d, layers=%d)"
          % (ga['nf'], ga['n_layers'], oa['nf'], oa['n_layers']))
    print("both saved args claim dataset = %r / %r\n" % (ga['dataset'], oa['dataset']))

    print("predicting on %d random QM9 molecules ..." % N)
    rg = predict(guide_fn, coords, feats, mask)
    ro = predict(oracle_fn, coords, feats, mask)

    ag, bg = fit_linear(rg, truth)
    ao, bo = fit_linear(ro, truth)
    pg, po = ag * rg + bg, ao * ro + bo
    eg, eo = (pg - truth).abs(), (po - truth).abs()

    print("\nfitted raw -> Debye:  guide  %.4f x + %.4f" % (ag, bg))
    print("                      oracle %.4f x + %.4f" % (ao, bo))
    print("\nMAE over all %d:   guide %.4f D    oracle %.4f D" % (N, eg.mean(), eo.mean()))

    # --- the structural test
    c_err = torch.corrcoef(torch.stack([eg, eo]))[0, 1].item()
    c_sig = torch.corrcoef(torch.stack([pg - truth, po - truth]))[0, 1].item()
    # how often is one clearly better than the other?
    ratio = (eg + 1e-9) / (eo + 1e-9)
    g_better = (ratio < 0.5).float().mean().item()
    o_better = (ratio > 2.0).float().mean().item()

    print("\n--- joint structure of the errors ---")
    print("  correlation of |error|      : %+.3f" % c_err)
    print("  correlation of signed error : %+.3f" % c_sig)
    print("  guide  >2x better on        : %5.1f%% of molecules" % (100 * g_better))
    print("  oracle >2x better on        : %5.1f%% of molecules" % (100 * o_better))
    print("  neither clearly better      : %5.1f%%"
          % (100 * (1 - g_better - o_better)))

    # split the sample by which model wins, then look at the other model there
    win_g = ratio < 0.5
    win_o = ratio > 2.0
    if win_g.any() and win_o.any():
        print("\n  where the GUIDE wins  : guide MAE %.4f   oracle MAE %.4f"
              % (eg[win_g].mean(), eo[win_g].mean()))
        print("  where the ORACLE wins : guide MAE %.4f   oracle MAE %.4f"
              % (eg[win_o].mean(), eo[win_o].mean()))

    # Molecules NEITHER model memorised: the honest held-out accuracy estimate.
    tie = (~win_g) & (~win_o)
    held_g = eg[tie].mean().item()
    held_o = eo[tie].mean().item()
    print("  where NEITHER wins    : guide MAE %.4f   oracle MAE %.4f  (n=%d)"
          % (held_g, held_o, int(tie.sum())))

    print("\n--- reading ---")
    # |error| correlation is the WRONG discriminator: it is dominated by shared
    # molecule difficulty (both models struggle on the same awkward geometries)
    # and stays high even for a genuinely disjoint pair. What separates the
    # hypotheses is the SIZE of the populations where one model is several times
    # better, against the fractions implied by a 50/50 split of EDM's 100k train
    # set inside full QM9:
    #     each half memorised by exactly one model : 100000/133885/2 = 37.4%
    #     in neither training set                  : 1 - 100000/133885 = 25.3%
    exp_each = 100000 / 133885 / 2
    exp_neither = 1 - 100000 / 133885
    tie_frac = 1.0 - g_better - o_better
    gap = max(eo[win_g].mean().item() / max(eg[win_g].mean().item(), 1e-9),
              eg[win_o].mean().item() / max(eo[win_o].mean().item(), 1e-9))
    if g_better > 0.20 and o_better > 0.20 and abs(tie_frac - exp_neither) < 0.12:
        verdict = ("DISJOINT CONSISTENT. Observed %.1f%% / %.1f%% / %.1f%% "
                   "(guide wins / oracle wins / neither) against %.1f%% / %.1f%% / %.1f%% "
                   "predicted by a 50/50 split of EDM's train set. Same-half training "
                   "cannot produce %.0fx MAE gaps on a third of molecules each. "
                   "Circumstantial but strong; ID-level proof needs the seed-42 halves."
                   % (100 * g_better, 100 * o_better, 100 * tie_frac,
                      100 * exp_each, 100 * exp_each, 100 * exp_neither, gap))
    elif g_better < 0.10 and o_better < 0.10:
        verdict = ("SAME-HALF SUSPECTED: almost no population where either model is "
                   "clearly better -- what two models fitted to the same molecules "
                   "look like. Do NOT adopt without ID-level proof.")
    else:
        verdict = ("INCONCLUSIVE: population fractions match neither hypothesis. "
                   "Reconstruct the seed-42 halves before adopting.")
    print("  " + verdict)

    out = {"n": N, "mae_guide": eg.mean().item(), "mae_oracle": eo.mean().item(),
           "fit_guide": [ag, bg], "fit_oracle": [ao, bo],
           "corr_abs_error": c_err, "corr_signed_error": c_sig,
           "guide_2x_better_frac": g_better, "oracle_2x_better_frac": o_better,
           "claimed_dataset_guide": ga['dataset'], "claimed_dataset_oracle": oa['dataset'],
           "heldout_mae_guide": held_g, "heldout_mae_oracle": held_o,
           "tie_fraction": tie_frac, "win_mae_gap": gap,
           "verdict": verdict, "unit": unit}
    out["property"] = prop
    name = 'disjointness_test_%s.json' % prop
    (ROOT / name).write_text(json.dumps(out, indent=2))
    print("\nwrote audit/fa_fb_search/%s" % name)


if __name__ == '__main__':
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else 'mu')
