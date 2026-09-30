"""Download the TFG checkpoints the transfer experiment needs, and verify them.

The generators are missing from this repository; the property networks are not
-- they are committed under `audit/fa_fb_search/TFG/` (see that directory's
PROVENANCE.md), so a fresh clone already has everything but the base model.
`--models` selects which generators to pull (see FOLDERS): `EDMsecond`, the
half-data EDM the transfer experiment runs on, is the default, and `EDMfull`
is worth adding for the base-model comparison. The six property networks
(`tf_predict_{mu,alpha,gap}`, `evaluate_{mu,alpha,gap}`) are already vendored
under `audit/fa_fb_search/TFG/`, and this script checks them rather than
re-downloading.

Everything comes from the Drive folder TFG's README links, whose file ids were
already enumerated into `audit/fa_fb_search/TFG/drive_pretrained_files.json`
when the property networks were pulled. `gdown` is the only extra dependency.

After a successful run:

    weights/<model>/generative_model_ema.npy
    weights/<model>/args.pickle

and `--verify` re-reports every hash so a cluster copy can be checked against
a local one without re-downloading.

WHY THE HASHES MATTER HERE MORE THAN USUAL. This checkpoint is the base model
for an entire experiment whose point is that nothing in it is ours. If the
file silently differed between the machine that produced the numbers and the
machine that reproduces them, the experiment would be unfalsifiable. Every
transfer cell stamps this md5, so the two can always be compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TFG_ROOT = os.path.join(ROOT, "audit", "fa_fb_search", "TFG")
DEST = os.path.join(ROOT, "weights", "EDMsecond")

# The two files inside TFG's "EDMsecond Shared folder" (id
# 1wnGEaxVPKZOwlBuI0yytOFfgMybxnX2o, recorded in drive_pretrained_files.json).
# Folder id is used rather than per-file ids because the enumeration in
# drive_predictor_files.json covers the predictor folders only.
# TFG releases three unconditional EDM generators. Which ones matter:
#
#   EDMsecond  trained on ONE HALF of QM9. THE one the transfer experiment
#              needs: it is the same recipe as EEGSDE's half-data row, i.e.
#              the like-for-like bar our own half-data model is measured
#              against, and it is the generator TFG's own guided results use.
#   EDMfull    trained on ALL of QM9. Not needed for the guidance transfer --
#              guiding a full-data model and comparing to our half-data one
#              would confound the base model with the data budget -- but it is
#              the single most useful EXTRA row for the base-model comparison,
#              because it puts EDM's published 98.7 / 82.0 through OUR
#              evaluator and so calibrates the whole published column.
#   EDM        TFG's third variant; same recipe family, no separate role here.
#
# Default fetches EDMsecond. `--models EDMsecond,EDMfull` gets both.
FOLDERS = {
    "EDMsecond": "1wnGEaxVPKZOwlBuI0yytOFfgMybxnX2o",
    "EDMfull": "1kXy0DaAcei3-9qOkSnhEtbpXBH9EzfgS",
    "EDM": "1zxvQlc0aRBlhHNdKfud8KT16mjzLagcp",
}
EDM_FOLDER_ID = FOLDERS["EDMsecond"]          # kept for callers that import it
NEEDED = ("generative_model_ema.npy", "args.pickle")

# Already vendored; checked, never fetched.
VENDORED = [os.path.join("tf_predict_%s" % p, f)
            for p in ("mu", "alpha", "gap")
            for f in ("model_ema_2000.npy", "args_2000.pickle")] + \
           [os.path.join("evaluate_%s" % p, f)
            for p in ("mu", "alpha", "gap")
            for f in ("best_checkpoint.npy", "args.pickle")]


# Reference hashes. EMPTY until the first successful download, then filled in
# by `--record`. `--verify` compares against these; with an empty table it can
# only report what is on disk, which cannot detect a truncated or substituted
# download -- the exact failure the docstring above says hashes are for. Record
# them once, commit them, and every later run is checked.
# Recorded 23 Sep 2026 from the first download (gdown, TFG's Drive folder),
# the file the hard gate and every local transfer cell were produced with.
EXPECTED = {
    'weights/EDMsecond/generative_model_ema.npy': '6abbd010d766a3e44c06448e0aa94bd3',
    'weights/EDMsecond/args.pickle': '28d7ffff0ecdd138017de5d74748e632',
    'audit/fa_fb_search/TFG/tf_predict_mu/model_ema_2000.npy': 'fe4ba055e39f8afa4dd96041c652ab74',
    'audit/fa_fb_search/TFG/tf_predict_mu/args_2000.pickle': 'a588297715df64d02aa0f2ca55718b1d',
    'audit/fa_fb_search/TFG/tf_predict_alpha/model_ema_2000.npy': '1b8bf52cac44d2e0e907da2e7c4b985b',
    'audit/fa_fb_search/TFG/tf_predict_alpha/args_2000.pickle': '3323f22c5e043fc2c514d58600b800be',
    'audit/fa_fb_search/TFG/tf_predict_gap/model_ema_2000.npy': '29c21e4a61d49d24bc9ef3d51fe0e234',
    'audit/fa_fb_search/TFG/tf_predict_gap/args_2000.pickle': 'd7cef4ca43e1f126aa4ab591f565cfa0',
    'audit/fa_fb_search/TFG/evaluate_mu/best_checkpoint.npy': 'f2e74eed23303485b94739ef34fe37be',
    'audit/fa_fb_search/TFG/evaluate_mu/args.pickle': '4c65f459cea4783ce4a5dad77892393d',
    'audit/fa_fb_search/TFG/evaluate_alpha/best_checkpoint.npy': '15cc4ae1dc21739b450469a988b3d60f',
    'audit/fa_fb_search/TFG/evaluate_alpha/args.pickle': 'feae2b0ba62b2f6f66ef9bd404be230f',
    'audit/fa_fb_search/TFG/evaluate_gap/best_checkpoint.npy': '6ff449ceae5e1b8bda44a016d277c462',
    'audit/fa_fb_search/TFG/evaluate_gap/args.pickle': '7115230d41bf24697f1fd1e041104954',
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def check_vendored():
    missing, ok = [], []
    for rel in VENDORED:
        p = os.path.join(TFG_ROOT, rel)
        (ok if os.path.exists(p) else missing).append(rel)
    print("vendored property networks: %d/%d present  (tracked in git; see "
          "audit/fa_fb_search/PROVENANCE.md)" % (len(ok), len(VENDORED)))
    for rel in missing:
        print("   MISSING  %s" % rel)
    if missing:
        print("   These are committed to the repository. If they are absent "
              "your clone is incomplete -- try `git checkout -- audit/`.")
    return not missing


def fetch_edm(model="EDMsecond", force=False):
    dest = os.path.join(ROOT, "weights", model)
    target = os.path.join(dest, "generative_model_ema.npy")
    if model not in FOLDERS:
        raise SystemExit("unknown model %r; known: %s"
                         % (model, ", ".join(sorted(FOLDERS))))
    if os.path.exists(target) and not force:
        print("%s already present at %s -- pass --force to refetch"
              % (model, dest))
        return True
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Either:\n"
              "    pip install gdown\n"
              "or download TFG's %r folder by hand from the Drive link\n"
              "in audit/fa_fb_search/TFG/README.md and place\n"
              "    generative_model_ema.npy\n    args.pickle\n"
              "into %s" % (model, dest), file=sys.stderr)
        return False
    os.makedirs(dest, exist_ok=True)
    print("downloading TFG %s (folder %s) -> %s"
          % (model, FOLDERS[model], dest))
    gdown.download_folder(id=FOLDERS[model], output=dest, quiet=False,
                          use_cookies=False)
    return True


def report(models=("EDMsecond",)):
    rows = []
    for model in models:
        for name in NEEDED:
            p = os.path.join(ROOT, "weights", model, name)
            if os.path.exists(p):
                rows.append({"path": os.path.relpath(p, ROOT), "md5": md5(p),
                             "bytes": os.path.getsize(p)})
            else:
                rows.append({"path": os.path.relpath(p, ROOT), "md5": None,
                             "bytes": None})
    for rel in VENDORED:
        p = os.path.join(TFG_ROOT, rel)
        if os.path.exists(p):
            rows.append({"path": os.path.relpath(p, ROOT), "md5": md5(p),
                         "bytes": os.path.getsize(p)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="EDMsecond",
                    help="comma list from %s. EDMsecond is what the transfer "
                         "sweep needs; add EDMfull for the extra base-model "
                         "comparison row." % ", ".join(sorted(FOLDERS)))
    ap.add_argument("--force", action="store_true",
                    help="refetch even if already present")
    ap.add_argument("--verify", action="store_true",
                    help="hash what is present and exit; download nothing")
    ap.add_argument("--record", action="store_true",
                    help="print an EXPECTED table to paste into this file, so "
                         "later --verify runs are checked against a reference "
                         "rather than against themselves")
    ap.add_argument("--manifest",
                    default=os.path.join(ROOT, "weights", "tfg_manifest.json"))
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    have_props = check_vendored()
    if not args.verify:
        for m in models:
            fetch_edm(m, force=args.force)

    rows = report(models)
    with open(args.manifest, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n%-64s %-34s %s" % ("path", "md5", "bytes"))
    for r in rows:
        print("%-64s %-34s %s" % (r["path"], r["md5"] or "-- MISSING --",
                                  r["bytes"] if r["bytes"] else "-"))
    print("\nmanifest written to %s" % os.path.relpath(args.manifest, ROOT))

    if args.record:
        print("\nEXPECTED = {")
        for r in rows:
            if r["md5"]:
                print('    %r: %r,' % (r["path"].replace("\\", "/"), r["md5"]))
        print("}")

    mismatched = [r for r in rows
                  if EXPECTED.get(r["path"].replace("\\", "/"))
                  and r["md5"] != EXPECTED[r["path"].replace("\\", "/")]]
    for r in mismatched:
        print("  HASH MISMATCH  %s" % r["path"])
    if not EXPECTED:
        print("\nNOTE: no reference hashes recorded yet. Run with --record "
              "once and paste the table into this file.")

    have_edm = all(os.path.exists(os.path.join(ROOT, "weights", m, n))
                   for m in models for n in NEEDED)
    if mismatched:
        print("\nNOT READY: a file does not match its recorded hash.")
        return 1
    if not (have_props and have_edm):
        print("\nNOT READY: the transfer sweep needs every row above to exist.")
        return 1
    print("\nREADY: run proj1/scripts/transfer_sweep.py --preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
