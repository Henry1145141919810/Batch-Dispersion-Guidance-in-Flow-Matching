"""Build the Project 1 code-submission archive, or just print what would go in.

    python proj1/scripts/make_submission.py              # dry run: print the manifest
    python proj1/scripts/make_submission.py --zip        # write the archive

WHY THIS EXISTS. The working repository is ~2.5 GB across 6,936 tracked files:
QM9 itself, cluster syncs, shipped tarballs, vendored reference repos, deck
snapshots, per-molecule sidecars. None of that is the code, and a grader asked
to review 2.6 GB will not find the four scripts that matter. This script names
the submission exactly, so what we hand in is the same set every time and the
decision about what to leave out is written down rather than improvised.

WHAT GOES IN (two tiers, both included by default)

  A. code, docs, paper source, configs           ~480 files,  ~18 MB
  B. the result cells the paper's tables read    ~1,117 files, ~4.5 MB

Tier B is optional in principle and included in practice: with it, every table
in the paper regenerates from the archive alone, with no GPU and no download.
That is worth 4.5 MB. `--no-results` drops it.

WHAT STAYS OUT, AND WHY

  data/            QM9 itself, 430 MB. Rebuilt by proj1/scripts/prepare_qm9.py;
                   the split is seeded (20260917) and reproducible.
  betty_pull/      cluster sync, 582 MB of checkpoints we did not author here.
  bundles/         the code tarballs already shipped to Betty, 396 MB. Shipping
                   a copy of every previous shipment inside this one is silly.
  audit/           vendored third-party repos. 344 MB on disk but only
                   ~30 MB tracked, and that 30 MB IS load-bearing: tfg_assets.py
                   loads TFG's model definitions from it and the borrowed f_A/f_B
                   property networks live there. Excluding it means the `edm` and
                   `equifm` backends do not run from the archive alone -- see
                   SUBMISSION.md section 7. Every paper table still regenerates,
                   because those come from the committed cells.
  archive/         superseded brainstorms and dropped ideas, 64 MB.
  scratch_schnet/  a reference implementation on no results path, 32 MB.
  slides/deck_versions/   frozen deck snapshots, 30 MB. The live source travels.
  *.pt *.npy       model weights. See WEIGHTS below -- this is the one real
                   judgement call in the manifest.
  *.permol.pt      per-molecule sidecars; the committed JSON cells carry every
                   number the paper reports.
  results/ other   superseded runs (n5000, basecmp, transfer*, v2/q90, tune,
                   bdg_port). docs/results/DATA_INDEX.md section 5 lists each
                   with what replaced it. Cite none of them; ship none of them.

WEIGHTS. `weights/*.pt` is 39 MB and is NOT included. The archive therefore
reproduces every TABLE (tier B carries the cells) but not fresh SAMPLING, which
needs the generators. If the submission instructions allow 39 MB more, pass
`--weights` and say so in the README; otherwise weights/README.md documents
every checkpoint by md5 so any of them can be matched to the cells that used it.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Tier A. Everything a reader needs to understand and re-run the study.
TIER_A_PREFIX = (
    "proj1/",          # all our code: src, scripts, tests, cluster job files
    "paper/",          # the manuscript source, its tooling and its figures
    "docs/",           # protocols, results pages, the data index
    "blade_runs/",     # the off-cluster run drivers
    "course/",         # the assignment and the paper template we were given
    "slides/handoff/", # the deck's content spec (the deck itself is a separate deliverable)
    "weights/README",  # what each checkpoint is, by md5, without the weights
)
TIER_A_FILES = ("README.md", "ONBOARDING.md", "requirements.txt",
                ".gitignore", ".gitattributes", "SUBMISSION.md",
                "slides/README_SLIDES.md")

# Tier B. The cells the paper's tables are computed from -- the canonical run
# only. docs/results/DATA_INDEX.md section 1 defines this set.
TIER_B_PREFIX = ("results/v3/", "results/m2/")

# Never, whatever tier asked for it.
DROP_EXT = (".pt", ".npy", ".npz", ".zip", ".tgz", ".sdf", ".h5")
DROP_SUBSTR = ("__pycache__", ".ipynb_checkpoints")


def tracked():
    """Only git-tracked files: .gitignore already encodes what is not ours."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def select(files, results=True, weights=False):
    keep = []
    for f in files:
        if any(s in f for s in DROP_SUBSTR):
            continue
        # `--weights` has to be part of the TIER-A test, not a late escape
        # hatch: weights/*.pt matches neither TIER_A_PREFIX ("weights/README")
        # nor TIER_B_PREFIX, so it was dropped two lines before the branch that
        # was supposed to rescue it. --weights was a silent no-op.
        is_weight = weights and f.startswith("weights/") and f.endswith(".pt")
        in_a = f.startswith(TIER_A_PREFIX) or f in TIER_A_FILES or is_weight
        in_b = results and f.startswith(TIER_B_PREFIX) and f.endswith(".json")
        if not (in_a or in_b):
            continue
        # paper/figs/ carries the paper's actual figures; nothing there should
        # be filtered by extension. (.pdf is not in DROP_EXT, so today this
        # only guards against a future .npz/.pt landing in that directory.)
        if f.endswith(DROP_EXT) and not f.startswith("paper/figs/") and not is_weight:
            continue
        keep.append(f)
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", action="store_true", help="write the archive (default: dry run)")
    ap.add_argument("--out", default=os.path.join(ROOT, "cis6270_p1_group2_code.zip"))
    ap.add_argument("--no-results", action="store_true", help="tier A only")
    ap.add_argument("--weights", action="store_true",
                    help="also include weights/*.pt (+60 MB); say so in the submission note")
    args = ap.parse_args()

    files = select(tracked(), results=not args.no_results, weights=args.weights)
    if not files:
        sys.exit("nothing selected -- is this a git checkout?")

    by_top, total = {}, 0
    for f in files:
        n = os.path.getsize(os.path.join(ROOT, f))
        total += n
        by_top.setdefault(f.split("/")[0], [0, 0])
        by_top[f.split("/")[0]][0] += 1
        by_top[f.split("/")[0]][1] += n
    for k, (n, s) in sorted(by_top.items(), key=lambda kv: -kv[1][1]):
        print("  %-16s %5d files %8.2f MB" % (k, n, s / 1e6))
    print("  %-16s %5d files %8.2f MB" % ("TOTAL", len(files), total / 1e6))

    # A submission that cannot rebuild its own tables is not worth handing in.
    # SUBMISSION.md is first: it is the entry point for the code review, and it
    # was untracked on the first build, so the archive shipped without it.
    need = ["SUBMISSION.md", "docs/results/DATA_INDEX.md", "requirements.txt",
            "README.md", "proj1/scripts/transfer_sweep.py",
            "paper/tools/build_results.py"]
    missing = [f for f in need if f not in files]
    if missing:
        sys.exit("REFUSING: the manifest is missing %s" % ", ".join(missing))

    if not args.zip:
        print("\ndry run. Add --zip to write %s" % args.out)
        return 0

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(ROOT, f), os.path.join("cis6270_p1_group2", f))
    print("\nwrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))
    return 0


if __name__ == "__main__":
    sys.exit(main())
