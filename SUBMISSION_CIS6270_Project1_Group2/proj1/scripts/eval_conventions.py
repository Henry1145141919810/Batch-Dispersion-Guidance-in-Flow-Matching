"""Score saved samples under several published metric conventions.

    python proj1/scripts/eval_conventions.py results/bench/fm_nfe100_euler_s0_samples.pt
    python proj1/scripts/eval_conventions.py --data 10000     # real QM9, for calibration

Different papers call different things "atom stability". Putting our number
beside theirs is only honest if both are computed the same way, so this
scores the SAME samples under each convention:

  edm_H        Hoogeboom et al. 2022, with hydrogens. Bond orders from the
               distance lookup table; an atom is stable iff its bond-order sum
               EQUALS its allowed valence. This is what train_fm.py selects on
               and what benchmark_base.py reports.
  tfgflow      Lin et al. 2025 (TFG-Flow), utils/evaluator.py, as published:
               heavy atoms only, and an atom is stable iff its bond-order sum is
               <= the allowed valence (under-bonded atoms pass; the missing
               bonds are implicitly hydrogens). Reproduced from their code.
  tfgflow_ascoded
               The same, but replicating a detail of their loop: only
               atom_orders[i] is incremented for each pair (i, j), never
               atom_orders[j], so each atom counts only its bonds to
               higher-indexed atoms. This is what their script actually prints.
               Recorded so the comparison to their tables is exact; the row
               above is what their metric is evidently intended to be.

Validity / uniqueness / connectivity are computed on the same atom set as the
stability metric (all atoms for edm_H, heavy atoms for the tfgflow rows), with
EDM's largest-fragment convention for validity. Connectivity is divided by ALL
generated molecules, as TFG-Flow's evaluator does; the of-valid figure is kept
under a separate key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from evaluation import bond_order, ALLOWED_VALENCE  # noqa: E402

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    Chem = None


def bonds_for(coords, syms):
    """{(i,j): order} over the given atoms, same lookup table as evaluation.py."""
    n = len(syms)
    out = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = float(torch.linalg.norm(coords[i] - coords[j]))
            o = bond_order(syms[i], syms[j], d)
            if o:
                out[(i, j)] = o
    return out


def smiles_largest_fragment(bonds, syms):
    if Chem is None:
        return None
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    m = Chem.RWMol()
    for s in syms:
        m.AddAtom(Chem.Atom(s))
    for (i, j), o in bonds.items():
        m.AddBond(i, j, order[o])
    try:
        mol = m.GetMol()
        Chem.SanitizeMol(mol)
        whole = Chem.MolToSmiles(mol)
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        largest = max(frags, default=mol, key=lambda x: x.GetNumAtoms())
        return Chem.MolToSmiles(largest), ("." not in whole)
    except Exception:
        return None, False


def score(coords, syms, rule):
    """Per-molecule (stable_atoms, n_atoms). rule in {'exact','le','le_ascoded'}."""
    n = len(syms)
    bonds = bonds_for(coords, syms)
    val_i = [0] * n            # both endpoints
    val_ascoded = [0] * n      # TFG-Flow's loop: only i
    for (i, j), o in bonds.items():
        val_i[i] += o
        val_i[j] += o
        val_ascoded[i] += o
    stable = 0
    for k in range(n):
        allowed = ALLOWED_VALENCE.get(syms[k], -1)
        if isinstance(allowed, list):
            allowed = max(allowed)
        if rule == "exact":
            ok = val_i[k] == allowed
        elif rule == "le":
            ok = val_i[k] <= allowed
        else:
            ok = val_ascoded[k] <= allowed
        stable += int(ok)
    return stable, n, bonds


def evaluate(C, T, M, type_names, finite, ref_smiles=None):
    out = {}
    for conv, keep_h, rule in (("edm_H", True, "exact"),
                               ("tfgflow", False, "le"),
                               ("tfgflow_ascoded", False, "le_ascoded")):
        n_at = n_sa = n_ms = 0
        n_mol = 0
        smis, connected = [], 0
        for b in range(C.shape[0]):
            if not finite[b]:
                n_mol += 1
                continue
            n = int(M[b].sum())
            syms = [type_names[int(T[b, i])] for i in range(n)]
            idx = [i for i in range(n) if keep_h or syms[i] != "H"]
            if not idx:
                n_mol += 1
                continue
            c = C[b, idx]
            s = [syms[i] for i in idx]
            st, na, bonds = score(c, s, rule)
            n_at += na; n_sa += st; n_ms += int(st == na); n_mol += 1
            smi, conn = smiles_largest_fragment(bonds, s)
            if smi is not None:
                smis.append(smi); connected += int(conn)
        uniq = set(smis)
        out[conv] = {
            "atom_stability": n_sa / max(n_at, 1),
            "mol_stability": n_ms / max(n_mol, 1),
            "validity": len(smis) / max(n_mol, 1),
            "uniqueness": len(uniq) / max(len(smis), 1),
            # TFG-Flow's evaluator divides connectivity by ALL generated
            # molecules (evaluator.py lines 91-100), not by the valid ones.
            "connectivity": connected / max(n_mol, 1),
            "connectivity_of_valid": connected / max(len(smis), 1),
            "n_mol": n_mol, "n_atoms": n_at,
        }
        if ref_smiles is not None and conv in ref_smiles:
            out[conv]["novelty"] = sum(1 for x in uniq if x not in ref_smiles[conv]) / max(len(uniq), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="?", help="a *_samples.pt written by benchmark_base.py")
    ap.add_argument("--data", type=int, default=0, help="score N real train_a molecules instead")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    type_names = d["types"]
    if args.data:
        g = torch.Generator().manual_seed(123)
        tr = d["split"]["train_a"]
        pick = tr[torch.randperm(len(tr), generator=g)[: args.data]]
        C, T, M = d["coords"][pick].float(), d["feats"][pick].argmax(-1), d["mask"][pick].bool()
        finite = torch.ones(len(pick), dtype=torch.bool)
        label = "REAL QM9 train_a, %d molecules" % args.data
    else:
        s = torch.load(args.samples, weights_only=False)
        C, T, M, finite = s["coords"].float(), s["types"].long(), s["mask"].bool(), s["finite"].bool()
        type_names = s["type_names"]
        label = os.path.basename(args.samples)

    res = evaluate(C, T, M, type_names, finite)
    print(label)
    print("  %-18s %9s %9s %9s %9s %9s" % ("convention", "atom", "mol", "valid", "uniq", "connect"))
    for k, v in res.items():
        print("  %-18s %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (k, v["atom_stability"], v["mol_stability"], v["validity"], v["uniqueness"], v["connectivity"]))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"label": label, **res}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
