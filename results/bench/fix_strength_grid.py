"""Make STRENGTHS the union of both grids, and record run provenance. Run once.

THE GRID SPLIT. The cluster ran w in {0.25, 0.5, 1, 2, 4}. STRENGTHS was later
changed to {0.01, 0.05, 0.25, 1, 4} to reach the low end plug needs. Neither
contains the other, so:

  * 42 finished cells (every arm's w=0.5 and w=2) fell OUT of the plan --
    they would never be regenerated if lost, and `SWEEP COMPLETE` would still
    print with them missing;
  * any arm run from now on lands on a different grid from the eight arms
    already on disk, so a "best over strength" column compares different grids;
  * mu/plug and mu/tfg_mc already have SEVEN cells where every other mu arm has
    five, and `select_arms.py` takes a max over them -- a best-of-7 draw against
    best-of-5. At se(in_band) ~ 0.0134 that is ~ +0.0025 of free coverage for
    exactly the two arms currently reported as the winners.

The union fixes all three at once: the 42 orphans rejoin the plan, and every
arm converges to the same 7-point curve as cells fill in.

PROVENANCE. Nothing in the 141 cluster cells records the checkpoint path, its
hash, the torch version or the GPU -- `fm` is only a basename, and both
machines' paths end in "fm_last.pt", so a checkpoint mismatch would have been
undetectable. Now recorded per cell.
"""
import io

NL = chr(10)
p = "proj1/scripts/guidance_sweep.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''# plug-in needs ~0.003-0.01 to avoid catastrophe (TOP6 memo section 8); at
# 0.25 it was already 75-94% clip-saturated, so the old grid could not tune it.
STRENGTHS = [0.01, 0.05, 0.25, 1.0, 4.0]'''
NEW = '''# THE UNION OF BOTH GRIDS RUN SO FAR, deliberately.
# The cluster ran {0.25, 0.5, 1, 2, 4}; this list was later narrowed to reach
# the low end plug-in needs (~0.003-0.01; at 0.25 it is already heavily
# clip-saturated). Neither grid contains the other, which orphaned 42 finished
# cells and meant any newly-run arm could not be compared against the eight
# already on disk at the same strengths. Keeping the union means the orphans
# stay in the plan and every arm converges to the same 7-point curve.
STRENGTHS = [0.01, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0]'''
assert s.count(OLD) == 1, "strengths anchor"
s = s.replace(OLD, NEW)

# ---- provenance -----------------------------------------------------------
OLD_I = '''import argparse
import itertools
import json
import os
import sys
import time'''
NEW_I = '''import argparse
import hashlib
import itertools
import json
import os
import sys
import time'''
assert s.count(OLD_I) == 1, "import anchor"
s = s.replace(OLD_I, NEW_I)

OLD_H = '''def load_rch(path, dev, target="residual"):'''
NEW_H = '''def file_md5(path, chunk=1 << 20):
    """Hash of the generator checkpoint, recorded in every cell.

    Cells produced on different machines are pooled into one table, and the
    only provenance previously recorded was `os.path.basename(args.fm)` --
    which is "fm_last.pt" on both the cluster and locally, so a checkpoint
    mismatch would have been invisible. One hash per run, not per cell.
    """
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def load_rch(path, dev, target="residual"):'''
assert s.count(OLD_H) == 1, "load_rch anchor"
s = s.replace(OLD_H, NEW_H)

OLD_P = '''    guides, evals, deltas = {}, {}, {}'''
NEW_P = '''    prov = {"fm_path": os.path.abspath(args.fm),
            "fm_md5": file_md5(args.fm),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": (torch.cuda.get_device_name(0)
                       if dev == "cuda" and torch.cuda.is_available() else "cpu")}
    print("provenance: %s  md5 %s  torch %s  cuda %s  on %s"
          % (os.path.basename(prov["fm_path"]), prov["fm_md5"][:12],
             prov["torch"], prov["cuda"], prov["device"]))

    guides, evals, deltas = {}, {}, {}'''
assert s.count(OLD_P) == 1, "guides anchor"
s = s.replace(OLD_P, NEW_P)

OLD_R = '''                  "variant": variant, "stage": args.stage,'''
NEW_R = '''                  "variant": variant, "stage": args.stage, "prov": prov,'''
assert s.count(OLD_R) == 1, "record anchor"
s = s.replace(OLD_R, NEW_R)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("STRENGTHS unioned; provenance recorded per cell")
