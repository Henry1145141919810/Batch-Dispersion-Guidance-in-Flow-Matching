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
        h = f_net_eval.embed(coords, feats, mask)
    h = h - h.mean(0, keepdim=True)
    hn = h / h.norm(dim=1, keepdim=True).clamp(min=eps)
    d = torch.cdist(hn, hn)
    B = h.shape[0]
    mean_pair = (d.sum() / max(B * (B - 1), 1)).item()
    G = hn @ hn.T + eps * torch.eye(B, device=h.device, dtype=h.dtype)
    logdet = torch.linalg.slogdet(G).logabsdet.item() / B
    return mean_pair, logdet


def evaluate_samples(coords, feats, mask, types, f_A, f_B, y, delta):
    """The full metric block for one arm. Returns a plain dict.

    `y` is the target in physical units, one per sample. f_A is the guide
    (what guidance optimised), f_B the held-out evaluator (what we believe).
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

    with torch.no_grad():
        fa = f_A(coords, feats, mask)
        fb = f_B(coords, feats, mask)
    err_b = (fb - y).abs()
    in_band = ((err_b <= delta) & finite).float().mean().item()
    gap = (fa - fb).abs()
    if finite.any():
        err_b, gap = err_b[finite], gap[finite]
        fa_m, fb_m = fa[finite].mean().item(), fb[finite].mean().item()
        mean_pair, logdet = embedding_diversity(
            f_B, coords[finite], feats[finite], mask[finite])
    else:
        fa_m = fb_m = mean_pair = logdet = float("nan")

    return {
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
    }
