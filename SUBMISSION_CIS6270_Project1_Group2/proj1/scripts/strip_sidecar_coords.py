"""Drop the 3D payload from .permol.pt sidecars so they can be committed.

    python proj1/scripts/strip_sidecar_coords.py results/full/v2 --dry-run
    python proj1/scripts/strip_sidecar_coords.py results/full/v2 results/xproj

WHY. The v2 run was sampled with `--save-coords`, so each sidecar carries
`coords` [n, 29, 3], `types` [n, 29] and `mask` [n, 29] -- about 2.4 MB a cell,
106 MB for the run. This repo has no LFS and v1's committed sidecars are 390 kB
with none of those keys. Stripping the three 3D tensors leaves exactly v1's key
set plus the decoded scores, which is what every analysis in docs/ actually
reads:

    f_A  f_B  f_A_dec  f_B_dec  y  finite  mol_stable  valid  n_atoms
    mol_idx  smiles

WHAT IS LOST, AND WHERE IT STILL EXISTS. Only the geometry: per-atom positions,
atom types and the padding mask. Nothing in the v2 tables, the paired z, the
bias/residual-sd decomposition or the V5 buckets uses them -- those read scores
and flags. Any future analysis that DOES need geometry (bond-length
distributions, conformer work, re-decoding under different rules) must go back
to the untouched originals, which live in the pull tarball `bundles/results/v2_cells.tgz` in
the laptop checkout (gitignored) and on the cluster at
`$PROJECT_ROOT/results/`. Re-extracting the tarball
over the tree restores them.

This edits files in place and is not reversible from the repo alone. It
refuses to run on a file whose key set it does not recognise, prints exactly
what it will do under --dry-run, and skips files already stripped.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DROP = ("coords", "types", "mask")
# v1's committed key set; a stripped sidecar must end up a superset of it.
REQUIRED = ("f_A", "f_B", "finite", "mol_idx", "mol_stable", "n_atoms",
            "smiles", "valid", "y")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="directories to walk")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch

    files = []
    for r in args.roots:
        r = r if os.path.isabs(r) else os.path.join(ROOT, r)
        files += sorted(glob.glob(os.path.join(r, "**", "*.permol.pt"),
                                  recursive=True))
    if not files:
        raise SystemExit("no .permol.pt found under %s" % ", ".join(args.roots))

    todo, already, before, after_est = [], 0, 0, 0
    for fn in files:
        d = torch.load(fn, map_location="cpu", weights_only=False)
        if not isinstance(d, dict):
            raise SystemExit("%s is not a dict sidecar" % fn)
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            raise SystemExit("%s lacks %s -- refusing to touch it"
                             % (fn, ", ".join(missing)))
        present = [k for k in DROP if k in d]
        sz = os.path.getsize(fn)
        before += sz
        if not present:
            already += 1
            after_est += sz
            continue
        todo.append((fn, present, sz))

    print("%d sidecars, %d already stripped, %d to strip (%.1f MB on disk)"
          % (len(files), already, len(todo), before / 1e6))
    if args.dry_run:
        for fn, present, sz in todo[:5]:
            print("  would drop %s from %s (%.1f MB)"
                  % (", ".join(present), os.path.relpath(fn, ROOT), sz / 1e6))
        if len(todo) > 5:
            print("  ... and %d more" % (len(todo) - 5))
        return

    freed = 0
    for fn, present, sz in todo:
        d = torch.load(fn, map_location="cpu", weights_only=False)
        for k in present:
            del d[k]
        tmp = fn + ".tmp"
        torch.save(d, tmp)
        os.replace(tmp, fn)
        freed += sz - os.path.getsize(fn)
    print("stripped %d files, freed %.1f MB" % (len(todo), freed / 1e6))
    print("originals remain in the pull tarball and on the cluster "
          "(see this script's docstring)")


if __name__ == "__main__":
    main()
