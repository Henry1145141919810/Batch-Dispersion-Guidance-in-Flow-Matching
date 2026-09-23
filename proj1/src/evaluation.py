"""S2: the evaluation harness. Every later comparison is measured with this.

Four groups of metric, and one number the gates depend on.

1. STRUCTURE -- does the sample look like a molecule at all?
   Bonds are inferred from interatomic distances with the lookup table of
   Hoogeboom et al. (E(3) Equivariant Diffusion for Molecule Generation in 3D,
   ICML 2022), which is the standard QM9 protocol so our numbers are comparable
   to published ones. An atom is "stable" when its inferred valency equals the
   allowed valency for its element; a molecule is stable when every atom is.

2. CHEMISTRY -- validity and uniqueness via RDKit, after building a molecule
   from the inferred bonds. Validity = sanitisation succeeds. Uniqueness =
   distinct canonical SMILES among the valid ones.

3. PROPERTY -- error against the target, measured with f_B, the evaluator that
   never took part in generation. The f_A - f_B gap is reported alongside: f_A
   is what guidance optimised, so a large gap is the reward-hacking signature.

4. DIVERSITY -- mean pairwise distance in f_B's invariant embedding, and the
   log-determinant of the embedding Gram matrix. Both are computed in the
   evaluator's coordinates, not the guide's, for the same reason.

delta, the tolerance, is NOT a free parameter. An in-band fraction is only
meaningful if the band exceeds the evaluator's own resolution, so delta is
derived from f_B's validation MAE (see `choose_delta`). Picking delta smaller
than that measures evaluator noise, not generation quality.

Limitations, stated because they affect what the numbers can support:
  * Distance-based bonds are a heuristic. A geometry that is slightly off can
    lose a bond and read as unstable even when the intent is clear.
  * Uniqueness over canonical SMILES ignores stereochemistry and conformation.
  * No DFT here. Property numbers are a learned evaluator's opinion, so they
    are comparable ACROSS OUR ARMS but not to a DFT-validated result.
"""
from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# bond inference: Hoogeboom et al. tables, lengths in picometres
# ---------------------------------------------------------------------------

BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}
BONDS3 = {
    "C": {"C": 120, "N": 116, "O": 113},
    "N": {"C": 116, "N": 110},
    "O": {"C": 113},
}
# The C#O entry (113 pm) was missing until 20 Sep 2026. On real QM9 no pair
# falls in that window, so calibration looked perfect; on generated samples
# ~2.5% of molecules had one, and scoring them with the entry absent inflated
# atom stability by ~0.2 pt and validity by ~1.1 pt relative to EDM's own
# bond_analyze.py. The table now matches EDM's entry for entry (verified by a
# brute-force grid over every HCNOF pair and distance).
MARGIN1, MARGIN2, MARGIN3 = 10, 5, 3
ALLOWED_VALENCE = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}


def bond_order(s1: str, s2: str, dist_angstrom: float) -> int:
    """0, 1, 2 or 3. `dist` in angstrom; the tables are in picometres."""
    d = 100.0 * dist_angstrom
    if s1 not in BONDS1 or s2 not in BONDS1[s1]:
        return 0
    if d >= BONDS1[s1][s2] + MARGIN1:
        return 0
    # inside single-bond range; tighten to double / triple if it fits
    if s1 in BONDS2 and s2 in BONDS2[s1] and d < BONDS2[s1][s2] + MARGIN2:
        if s1 in BONDS3 and s2 in BONDS3[s1] and d < BONDS3[s1][s2] + MARGIN3:
            return 3
        return 2
    return 1


def stability(coords, feats, mask, types):
    """Per-molecule (atoms_stable, n_atoms, molecule_is_stable, bond_matrix).

    coords [B,N,3] angstrom, feats [B,N,K] (argmax gives the element),
    mask [B,N]. Returns lists of length B.
    """
    B, N, _ = coords.shape
    idx = feats.argmax(-1)
    out = []
    for b in range(B):
        n = int(mask[b].sum().item())
        syms = [types[int(idx[b, i])] for i in range(n)]
        valence = [0] * n
        bonds = {}
        for i in range(n):
            for j in range(i + 1, n):
                d = float(torch.linalg.norm(coords[b, i] - coords[b, j]))
                o = bond_order(syms[i], syms[j], d)
                if o:
                    valence[i] += o
                    valence[j] += o
                    bonds[(i, j)] = o
        n_stable = sum(1 for i in range(n)
                       if valence[i] == ALLOWED_VALENCE.get(syms[i], -1))
        out.append((n_stable, n, n_stable == n, bonds, syms))
    return out


def to_smiles(bonds, syms):
    """Canonical SMILES via RDKit, or None if it will not sanitise."""
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        return None
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
             3: Chem.BondType.TRIPLE}
    m = Chem.RWMol()
    for s in syms:
        m.AddAtom(Chem.Atom(s))
    for (i, j), o in bonds.items():
        m.AddBond(i, j, order[o])
    try:
        mol = m.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# delta
# ---------------------------------------------------------------------------

def choose_delta(mae_eval, k=2.0):
    """Tolerance in physical units.

    delta = k * (evaluator validation MAE). With k = 2 the band is twice the
    evaluator's own typical error, so "in band" is a statement about the
    sample rather than about f_B's noise. k is reported with every result; a
    single in-band number without its k is not interpretable.
    """
    return k * mae_eval


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

EVAL_CHUNK = 1024


def _chunked(fn, coords, feats, mask, chunk=None):
    """Apply a per-molecule network in slices of at most `chunk` molecules.

    The predictors are per-molecule (eval mode, no batch statistics), so this
    equals one call up to float rounding -- and at n <= chunk it IS one call,
    so every n=512 screening cell is unchanged. At n=5000 a single call builds
    5000 x 29 x 29 edge tensors in every layer, and an OOM there would strike
    AFTER the whole sampling run, losing a cell that took up to an hour.
    `chunk` defaults to the module's EVAL_CHUNK read at CALL time, so a test
    can shrink it and drive evaluate_samples through the sliced path.
    """
    chunk = EVAL_CHUNK if chunk is None else chunk
    if coords.shape[0] <= chunk:
        return fn(coords, feats, mask)
    return torch.cat([fn(coords[i:i + chunk], feats[i:i + chunk],
                         mask[i:i + chunk])
                      for i in range(0, coords.shape[0], chunk)])


def embedding_diversity(f_net_eval, coords, feats, mask, eps=1e-6):
    """(mean pairwise distance, normalised log-det) of the evaluator embedding.

    log-det of the Gram matrix of unit-normalised embeddings, divided by the
    batch size, is a volume measure: it falls when samples collapse together
    even if a few outliers keep the mean distance high.
    """
    with torch.no_grad():
        # `f_net_eval.embed`, NOT `f_net_eval.net.embed`. The wrapper is what
        # knows which feature scale its inner network wants; reaching through
        # to `.net` skips that conversion. For our own `PhysicalProperty` the
        # two are identical, so this was invisible for the whole main sweep --
        # but the borrowed evaluator in `external/tfg_assets.py` takes raw
        # one-hot while the sampler carries one-hot/8, and going through `.net`
        # fed it features 8x too small: the embeddings came out at cos 0.64 to
        # the correct ones, not a rescaling of them, moving diversity_logdet by
        # 12%. Finite, plausible, and wrong.
        h = _chunked(f_net_eval.embed, coords, feats, mask)
    h = h - h.mean(0, keepdim=True)
    hn = h / h.norm(dim=1, keepdim=True).clamp(min=eps)
    d = torch.cdist(hn, hn)
    B = h.shape[0]
    mean_pair = (d.sum() / max(B * (B - 1), 1)).item()
    G = hn @ hn.T + eps * torch.eye(B, device=h.device, dtype=h.dtype)
    logdet = torch.linalg.slogdet(G).logabsdet.item() / B
    return mean_pair, logdet


def decode_types(feats, mask, hot_value=1.0):
    """The atom types that actually decode: argmax -> one-hot, masked, at the
    SAMPLER's feature scale (`hot_value` = 1 / the generator's one-hot
    divisor; 1 for our flow model, 1/4 for EDMsecond).

    The property metrics were always computed on the continuous features a
    trajectory ends on, which a predictor reads even where they are not a
    valid one-hot. An arm that pushes the FEATURES along grad f (TFG's
    clean-space step does, on every guided step including the last) can move
    that number without moving the molecule. Measured on the tfg pilot
    (mu, w=1, n=64): MAE 0.31 -> 0.44 D decoded, where plug moved 0.90 ->
    0.94. So every cell now also carries the decoded view.
    """
    oh = torch.nn.functional.one_hot(feats.argmax(-1), feats.shape[-1])
    return oh.to(feats.dtype) * hot_value * mask.unsqueeze(-1)


def evaluate_samples(coords, feats, mask, types, f_A, f_B, y, delta,
                     per_mol=False, hot_value=1.0):
    """The full metric block for one arm. Returns a plain dict.

    `y` is the target in physical units, one per sample. f_A is the guide
    (what guidance optimised), f_B the held-out evaluator (what we believe).

    per_mol=True adds "_per_mol": CPU tensors, one row per sample (f_A, f_B,
    y, finite, mol_stable, valid, n_atoms). The caller must pop it before
    json-serialising. It exists so two arms run on the same noise and the same
    targets can be compared PAIRED, which the aggregate numbers cannot do.

    NOTE diversity_logdet depends on n (the Gram matrix is n x n but its rank
    is at most the embedding width), so it is comparable only between cells of
    the same n.

    Every property metric also has a `_dec` twin scored on the DECODED atom
    types (decode_types; `hot_value` is the sampler's one-hot scale). The
    un-suffixed metrics are unchanged, so every existing cell stays comparable.
    """
    B = coords.shape[0]
    # An exploded trajectory is a failed sample. It counts against stability,
    # validity and in-band, and is excluded from the property averages so one
    # NaN cannot erase the metrics of the whole arm.
    finite = torch.isfinite(coords).all((1, 2)) & torch.isfinite(feats).all((1, 2))
    n_bad = int((~finite).sum().item())
    coords = torch.where(finite.view(-1, 1, 1), coords, torch.zeros_like(coords))
    feats = torch.where(finite.view(-1, 1, 1), feats, torch.zeros_like(feats))

    st = stability(coords, feats, mask, types)
    n_stable_atoms = sum(s[0] for s, ok in zip(st, finite) if ok)
    n_atoms = sum(s[1] for s in st)
    mol_stable = sum(1 for s, ok in zip(st, finite) if ok and s[2])

    smiles = [to_smiles(s[3], s[4]) if ok else None for s, ok in zip(st, finite)]
    valid = [s for s in smiles if s is not None]
    uniq = len(set(valid))

    feats_dec = decode_types(feats, mask, hot_value)
    with torch.no_grad():
        fa = _chunked(f_A, coords, feats, mask)
        fb = _chunked(f_B, coords, feats, mask)
        fa_d = _chunked(f_A, coords, feats_dec, mask)
        fb_d = _chunked(f_B, coords, feats_dec, mask)
    pm = None
    if per_mol:
        pm = {"f_A": fa.detach().float().cpu(),
              "f_B": fb.detach().float().cpu(),
              "f_A_dec": fa_d.detach().float().cpu(),
              "f_B_dec": fb_d.detach().float().cpu(),
              "y": torch.as_tensor(y).detach().float().cpu(),
              "finite": finite.detach().cpu(),
              "mol_stable": torch.tensor([bool(ok) and bool(s[2])
                                          for s, ok in zip(st, finite)]),
              "valid": torch.tensor([s is not None for s in smiles]),
              "n_atoms": mask.float().sum(1).round().to(torch.int32).cpu(),
              # canonical SMILES per row (None where invalid). The headline
              # metric -- DISTINCT valid molecules inside the band per attempt
              # -- needs identity per molecule, which the aggregate
              # `uniqueness_of_valid` cannot give once it is intersected with
              # the band.
              "smiles": list(smiles)}
    err_b = (fb - y).abs()
    in_band = ((err_b <= delta) & finite).float().mean().item()
    gap = (fa - fb).abs()
    err_d = (fb_d - y).abs()
    in_band_d = ((err_d <= delta) & finite).float().mean().item()
    gap_d = (fa_d - fb_d).abs()
    fb_d_m = fb_d[finite].mean().item() if finite.any() else float("nan")
    if finite.any():
        err_b, gap = err_b[finite], gap[finite]
        err_d, gap_d = err_d[finite], gap_d[finite]
        fa_m, fb_m = fa[finite].mean().item(), fb[finite].mean().item()
        mean_pair, logdet = embedding_diversity(
            f_B, coords[finite], feats[finite], mask[finite])
    else:
        fa_m = fb_m = mean_pair = logdet = float("nan")

    out = {
        "n": B,
        "n_nonfinite": n_bad,
        "atom_stability": n_stable_atoms / max(n_atoms, 1),
        "mol_stability": mol_stable / B,
        "validity": len(valid) / B,
        "uniqueness_of_valid": uniq / max(len(valid), 1),
        "unique_valid_per_sample": uniq / B,
        "prop_mae_eval": err_b.mean().item(),
        "prop_rmse_eval": err_b.pow(2).mean().sqrt().item(),
        "in_band_fraction": in_band,
        "delta": delta,
        "guide_eval_gap_mean": gap.mean().item(),
        "guide_eval_gap_max": gap.max().item(),
        "f_A_mean": fa_m,
        "f_B_mean": fb_m,
        "target_mean": float(y.mean()),
        "diversity_mean_pairwise": mean_pair,
        "diversity_logdet": logdet,
        "smiles_sample": valid[:5],
        # the same property metrics on the DECODED atom types (decode_types)
        "prop_mae_eval_dec": err_d.mean().item(),
        "prop_rmse_eval_dec": err_d.pow(2).mean().sqrt().item(),
        "in_band_fraction_dec": in_band_d,
        "guide_eval_gap_dec_mean": gap_d.mean().item(),
        "f_B_dec_mean": fb_d_m,
    }
    if pm is not None:
        out["_per_mol"] = pm
    return out
