"""Chemistry scored by a rule INDEPENDENT of the evaluator's bond table.

    python proj1/scripts/independent_chem.py results/pilot_chem/seed20261001_n2048

The sweep's mol_stability uses the Hoogeboom distance table (evaluation.py).
Chemistry-safe guidance (guidance.py, *_chem modes) guards a smooth relaxation
of THAT SAME table, so a stability gain could in principle be the table being
gamed rather than chemistry being kept. This script re-scores the saved final
molecules (sidecars written with --per-mol --save-coords) with RDKit's own
bond perception, rdDetermineBonds (covalent-radius connectivity + Hueckel /
xyz2mol bond orders, total charge 0), and asks RDKit to sanitise the result.

  indep_valid   the molecule gets a consistent neutral bond assignment and
                sanitises -- no reference to the Hoogeboom table anywhere
  agree         fraction of molecules where indep_valid == the sweep's
                mol_stable, i.e. how often the two rules agree

If a guard's gain in mol_stability is not matched by a gain in indep_valid,
the gain is the table, not the chemistry.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import torch

TYPES = ("H", "C", "N", "O", "F")


def indep_valid(coords, types, n):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDetermineBonds
    RDLogger.DisableLog("rdApp.*")
    lines = [str(n), ""]
    for i in range(n):
        x, y, z = (float(v) for v in coords[i])
        lines.append("%s %.6f %.6f %.6f" % (TYPES[int(types[i])], x, y, z))
    try:
        mol = Chem.MolFromXYZBlock("\n".join(lines) + "\n")
        if mol is None:
            return False
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def score_sidecar(path):
    s = torch.load(path, weights_only=False)
    if "coords" not in s:
        return None
    C, T, M = s["coords"], s["types"], s["mask"]
    ok = []
    for b in range(C.shape[0]):
        n = int(M[b].sum())
        ok.append(indep_valid(C[b, :n], T[b, :n], n))
    ok = torch.tensor(ok)
    st = s["mol_stable"].bool()
    return {"n": int(ok.numel()), "indep_valid": ok.double().mean().item(),
            "mol_stable": st.double().mean().item(),
            "agree": (ok == st).double().mean().item(), "_ok": ok}


def main(argv):
    root = argv[1] if len(argv) > 1 else "."
    out = {}
    for p in sorted(glob.glob(os.path.join(root, "*.permol.pt"))):
        r = score_sidecar(p)
        if r is None:
            continue
        name = os.path.basename(p)[:-len(".permol.pt")]
        print("%-52s n=%d  indep_valid %.3f  mol_stable %.3f  agree %.3f"
              % (name, r["n"], r["indep_valid"], r["mol_stable"], r["agree"]))
        out[name] = {k: v for k, v in r.items() if not k.startswith("_")}
        torch.save(r["_ok"], p[:-len(".permol.pt")] + ".indep.pt")
    with open(os.path.join(root, "independent_chem.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
