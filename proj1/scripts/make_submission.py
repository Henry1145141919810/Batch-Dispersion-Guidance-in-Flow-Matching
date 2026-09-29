"""Build the Project 1 code-submission archive, or just print what would go in.

    python proj1/scripts/make_submission.py              # dry run: print the manifest
    python proj1/scripts/make_submission.py --zip        # write the archive

WHY THIS EXISTS. The working repository is ~2.5 GB across 6,936 tracked files:
QM9 itself, cluster syncs, shipped tarballs, vendored reference repos, deck
snapshots, per-molecule sidecars. None of that is the code, and a grader asked
to review 2.6 GB will not find the four scripts that matter. This script names
the submission exactly, so what we hand in is the same set every time and the
decision about what to leave out is written down rather than improvised.

WHAT GOES IN (four tiers, all included by default)

  A. code, docs, paper source, configs           ~480 files,  ~18 MB
  B. the result cells the paper's tables read    ~1,117 files, ~4.5 MB
  W. the model weights that fresh sampling needs    ~13 files, ~57 MB
  V. vendored third-party code the external
     backends load at import time                  ~37 files, ~30 MB

Total ~1,700 files, ~112 MB.

Tier B: with it, every table in the paper regenerates from the archive alone,
with no GPU and no download. `--no-results` drops it.

Tier W: every checkpoint WE trained, so a reader can sample, not just re-read
tables -- the flow-matching and VP-diffusion generators and the six f_A/f_B
predictors (weights/*.pt, documented by md5 in weights/README.md), and the
Modality 2 generator with the 12 MB DeepFlyBrain split it samples against
(proj1/m2/blade_bundle/{fm_m2_dfb500.pt,dfb500.npz}). `--no-weights` drops it.

Tier V: audit/fa_fb_search/ -- TFG and OC-Flow, vendored unmodified, with their
origin, commit and licence in PROVENANCE.md. It is not our code, but it is
load-bearing: proj1/src/external/tfg_assets.py imports TFG's model definitions
from it, and the borrowed f_A/f_B property networks the `equifm` backend is
scored with are checkpoints inside it. Without it the `edm` and `equifm`
backends cannot even import. `--no-vendored` drops it.

What the archive still cannot contain is the two BORROWED generators, which are
third-party releases: fetch them with proj1/scripts/fetch_tfg_assets.py (TFG's
EDMsecond) and fetch_equifm_assets.py (EquiFM). SUBMISSION.md section 7.

WHAT STAYS OUT, AND WHY

  data/            QM9 itself, 430 MB. Rebuilt by proj1/scripts/prepare_qm9.py;
                   the split is seeded (20260917) and reproducible.
  betty_pull/      cluster sync, 582 MB of checkpoints we did not author here.
  bundles/         the code tarballs already shipped to Betty, 396 MB. Shipping
                   a copy of every previous shipment inside this one is silly.
  audit/ other     everything in audit/ except tier V: reference clones and the
                   BDG review's evidence (docs/methods/BDG_REVIEW.md cites it;
                   it is in the git repository, not the archive).
  archive/         superseded brainstorms and dropped ideas, 64 MB.
  scratch_schnet/  a reference implementation on no results path, 32 MB.
  slides/deck_versions/   frozen deck snapshots, 30 MB. The live source travels.
  weights/EDMsecond/  TFG's borrowed generator, 21 MB, not ours and not tracked;
                   fetch_tfg_assets.py downloads it.
  other *.pt/*.npy anything binary outside tiers W and V -- cluster checkpoints,
                   per-molecule sidecars (*.permol.pt). The committed JSON cells
                   carry every number the paper reports.
  results/ other   superseded runs (n5000, basecmp, transfer*, v2/q90, tune,
                   bdg_port). docs/results/DATA_INDEX.md section 5 lists each
                   with what replaced it. Cite none of them; ship none of them.
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
    "weights/README",  # what each checkpoint is, by md5
)
TIER_A_FILES = ("README.md", "ONBOARDING.md", "requirements.txt",
                ".gitignore", ".gitattributes", "SUBMISSION.md",
                "slides/README_SLIDES.md")

# Tier B. The cells the paper's tables are computed from -- the canonical run
# only. docs/results/DATA_INDEX.md section 1 defines this set.
TIER_B_PREFIX = ("results/v3/", "results/m2/",
                 # the batch-memory probe that chose v3's batch size of 500.
                 # proj1/tests/test_v3.py reads it (`batch_probe_record_exists`)
                 # and FAILS without it: it sat outside results/v3/, so an
                 # earlier manifest shipped a test suite that could not pass.
                 "results/v3_batch_memory",
                 # Modality 2's measured summaries (correction share, realism
                 # gate, DeepFlyBrain activity, strength and window scans),
                 # 127 KB. paper_fill_v3.py, v3_final_summary.py and
                 # m2_v3_results.py open them unconditionally, so without them
                 # the paper cross-check stops at FileNotFoundError.
                 "results/m2_")

# Tier W. The checkpoints WE trained. Everything tracked under weights/ (the
# FM and VP generators, the six predictors, and their README / manifests),
# plus the Modality 2 generator and the data split it samples against, which
# live beside the M2 code rather than in weights/.
TIER_W_PREFIX = ("weights/",)
TIER_W_FILES = ("proj1/m2/blade_bundle/fm_m2_dfb500.pt",
                "proj1/m2/blade_bundle/dfb500.npz")

# Tier V. Third-party code, vendored unmodified, that our code imports. Its
# binary files (.npy checkpoints, .pickle args) are the borrowed property
# networks, so they must survive the DROP_EXT filter below.
TIER_V_PREFIX = ("audit/fa_fb_search/",)

# Never, whatever tier asked for it -- EXCEPT a file that tier W or V names,
# which is exactly the binary those tiers exist to carry.
DROP_EXT = (".pt", ".npy", ".npz", ".zip", ".tgz", ".sdf", ".h5")
DROP_SUBSTR = ("__pycache__", ".ipynb_checkpoints", ".permol.pt")


def tracked():
    """Only git-tracked files: .gitignore already encodes what is not ours."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def select(files, results=True, weights=True, vendored=True):
    keep = []
    for f in files:
        if any(s in f for s in DROP_SUBSTR):
            continue
        # A tier-W/V file must be tested BEFORE the extension filter, not
        # rescued after it: an earlier version dropped weights/*.pt by
        # extension two lines before the branch meant to keep them, so
        # `--weights` was a silent no-op. `carries_binary` is the exemption.
        in_w = weights and (f.startswith(TIER_W_PREFIX) or f in TIER_W_FILES)
        in_v = vendored and f.startswith(TIER_V_PREFIX)
        carries_binary = in_w or in_v or f.startswith("paper/figs/")
        in_a = f.startswith(TIER_A_PREFIX) or f in TIER_A_FILES
        in_b = results and f.startswith(TIER_B_PREFIX) and f.endswith(".json")
        if not (in_a or in_b or in_w or in_v):
            continue
        if f.endswith(DROP_EXT) and not carries_binary:
            continue
        keep.append(f)
    return sorted(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", action="store_true", help="write the archive (default: dry run)")
    ap.add_argument("--out", default=os.path.join(ROOT, "cis6270_p1_group2_code.zip"))
    ap.add_argument("--no-results", action="store_true", help="tier A only")
    ap.add_argument("--no-weights", action="store_true",
                    help="drop tier W, our checkpoints (~57 MB): tables still "
                         "rebuild, fresh sampling no longer runs")
    ap.add_argument("--no-vendored", action="store_true",
                    help="drop tier V, audit/fa_fb_search (~30 MB): the edm and "
                         "equifm backends no longer import")
    ap.add_argument("--copy-to", default=None, metavar="DIR",
                    help="copy the manifest, unzipped, into DIR (must not exist). "
                         "The way to test that the archive runs on its own: no "
                         "betty_pull/, no local checkpoints, no .git")
    args = ap.parse_args()

    files = select(tracked(), results=not args.no_results,
                   weights=not args.no_weights, vendored=not args.no_vendored)
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
    # ...and one that cannot SAMPLE is not what the README promises. Named
    # file by file, so a renamed checkpoint fails the build instead of
    # shipping an archive that stops at FileNotFoundError on the reader's side.
    if not args.no_weights:
        need += ["weights/fm_ema.pt", "weights/diff_ema.pt", "weights/README.md",
                 *("weights/f_%s_%s.pt" % (s, p) for s in "AB"
                   for p in ("mu", "alpha", "gap")),
                 *TIER_W_FILES]
    if not args.no_vendored:
        need += ["audit/fa_fb_search/PROVENANCE.md"]
    missing = [f for f in need if f not in files]
    if missing:
        sys.exit("REFUSING: the manifest is missing %s" % ", ".join(missing))

    if args.copy_to:
        import shutil
        if os.path.exists(args.copy_to):
            sys.exit("REFUSING: %s exists; give a fresh directory" % args.copy_to)
        for f in files:
            dst = os.path.join(args.copy_to, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, f), dst)
        print("\ncopied %d files to %s" % (len(files), args.copy_to))
        return 0

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
