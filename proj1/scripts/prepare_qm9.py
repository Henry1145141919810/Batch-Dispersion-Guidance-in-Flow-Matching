"""S0: turn gdb9.sdf + gdb9.sdf.csv into tensors, with the four-way split.

Reads 3D coordinates and atom types with RDKit, joins the DFT properties from
the CSV by mol_id, and writes one .pt file.

The four-way split exists so the guide and the evaluator never share training
data (see FEASIBILITY_TESTS_HOW_TO_RUN_AND_DECIDE.md S0):

  train_a  -> guide f_A   (and the generator, under the pre-registered
              protocol in PROJECT_GUIDE section 4.1 -- see --split)
  train_b  -> evaluator f_B only
  val      -> checkpoint selection, hyperparameters, tolerance choice
  test     -> the (N, y*) targets, never trained on

Split is random with a fixed seed. The official Cormorant/E(3)-EDM split and its
3,054 uncharacterized exclusions are deliberately NOT used here: feasibility only
needs a predictor that is roughly right, and chasing the official split files is
not on the critical path. Phase 1 should switch to them before any number is
quoted against published work.

Run: python proj1/scripts/download_qm9.py   (once, fetches + verifies the raw data)
     python proj1/scripts/prepare_qm9.py
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")
SDF = os.path.join(DATA, "gdb9.sdf")
CSV = os.path.join(DATA, "gdb9.sdf.csv")
OUT = os.path.join(DATA, "qm9.pt")

TYPES = ["H", "C", "N", "O", "F"]
T2I = {s: i for i, s in enumerate(TYPES)}
MAX_ATOMS = 29
# Properties we target. Units: mu in Debye, alpha in Bohr^3, gap in Hartree.
PROPS = ["mu", "alpha", "gap"]
SEED = 20260917


def load_properties():
    """mol_id -> [mu, alpha, gap]"""
    out = {}
    with open(CSV, newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                out[row["mol_id"]] = [float(row[p]) for p in PROPS]
            except (KeyError, ValueError):
                continue
    return out


def main():
    for p in (SDF, CSV):
        if not os.path.exists(p):
            print("missing %s\n\nRun:  python proj1/scripts/download_qm9.py\n\n"
                  "That fetches the DeepChem QM9 packaging this project was "
                  "built on and\nverifies it by SHA-256 -- a different QM9 "
                  "distribution parses fine and\nsilently changes every "
                  "number in docs/results/." % p)
            return 1

    t0 = time.time()
    props = load_properties()
    print("properties loaded: %d molecules" % len(props))

    supplier = Chem.SDMolSupplier(SDF, removeHs=False, sanitize=False)
    coords_all, feats_all, mask_all, y_all = [], [], [], []
    n_skipped = {"no_mol": 0, "no_conf": 0, "too_big": 0, "bad_type": 0,
                 "no_props": 0}

    for i, mol in enumerate(supplier):
        if i % 20000 == 0 and i:
            print("  %6d / 133885  (%.0fs)" % (i, time.time() - t0))
        if mol is None:
            n_skipped["no_mol"] += 1
            continue
        mol_id = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if mol_id not in props:
            n_skipped["no_props"] += 1
            continue
        if mol.GetNumConformers() == 0:
            n_skipped["no_conf"] += 1
            continue
        n = mol.GetNumAtoms()
        if n > MAX_ATOMS:
            n_skipped["too_big"] += 1
            continue
        syms = [a.GetSymbol() for a in mol.GetAtoms()]
        if any(s not in T2I for s in syms):
            n_skipped["bad_type"] += 1
            continue

        pos = mol.GetConformer().GetPositions()          # [n, 3] angstrom
        c = np.zeros((MAX_ATOMS, 3), dtype=np.float32)
        f = np.zeros((MAX_ATOMS, len(TYPES)), dtype=np.float32)
        m = np.zeros((MAX_ATOMS,), dtype=np.float32)
        c[:n] = pos - pos.mean(axis=0, keepdims=True)    # zero centre of mass
        for j, s in enumerate(syms):
            f[j, T2I[s]] = 1.0
        m[:n] = 1.0

        coords_all.append(c)
        feats_all.append(f)
        mask_all.append(m)
        y_all.append(props[mol_id])

    coords = torch.from_numpy(np.stack(coords_all))
    feats = torch.from_numpy(np.stack(feats_all))
    mask = torch.from_numpy(np.stack(mask_all))
    y = torch.tensor(y_all, dtype=torch.float32)
    n_mol = coords.shape[0]
    print("\nkept %d molecules, skipped %s" % (n_mol, n_skipped))

    # ---- four-way split
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_mol, generator=g)
    n_test, n_val = 13083, 17748
    n_rest = n_mol - n_test - n_val
    n_a = n_rest // 2
    idx = {
        "train_a": perm[:n_a],
        "train_b": perm[n_a:n_rest],
        "val": perm[n_rest:n_rest + n_val],
        "test": perm[n_rest + n_val:],
    }

    # ---- normalisation from train_a ONLY, so the evaluator's half never leaks
    ya = y[idx["train_a"]]
    y_mean, y_std = ya.mean(0), ya.std(0)

    torch.save({
        "coords": coords, "feats": feats, "mask": mask, "y": y,
        "types": TYPES, "props": PROPS, "max_atoms": MAX_ATOMS,
        "split": idx, "y_mean": y_mean, "y_std": y_std, "seed": SEED,
        "note": "random split; official Cormorant split NOT used - see docstring",
    }, OUT)

    print("\nsplit sizes: " + "  ".join(
        "%s=%d" % (k, len(v)) for k, v in idx.items()))
    print("atoms per molecule: min=%d mean=%.2f max=%d"
          % (mask.sum(1).min(), mask.sum(1).mean(), mask.sum(1).max()))
    print("\nproperty         mean      std       min       max")
    for j, p in enumerate(PROPS):
        print("%-12s %9.4f %9.4f %9.4f %9.4f"
              % (p, y[:, j].mean(), y[:, j].std(), y[:, j].min(), y[:, j].max()))
    print("\nnormalisation from train_a: mean=%s std=%s"
          % (y_mean.tolist(), y_std.tolist()))
    print("wrote %s (%.1f MB) in %.0fs"
          % (OUT, os.path.getsize(OUT) / 1e6, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
