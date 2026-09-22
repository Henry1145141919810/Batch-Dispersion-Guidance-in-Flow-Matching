"""Download the TFG checkpoints the transfer experiment needs, and verify them.

Only ONE file is genuinely missing from this repository: `EDMsecond`, TFG's
unconditional EDM trained on one half of QM9. The six property networks
(`tf_predict_{mu,alpha,gap}`, `evaluate_{mu,alpha,gap}`) are already vendored
under `audit/fa_fb_search/TFG/`, and this script checks them rather than
re-downloading.

Everything comes from the Drive folder TFG's README links, whose file ids were
already enumerated into `audit/fa_fb_search/TFG/drive_pretrained_files.json`
when the property networks were pulled. `gdown` is the only extra dependency.

After a successful run:

    weights/EDMsecond/generative_model_ema.npy
    weights/EDMsecond/args.pickle

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
EDM_FOLDER_ID = "1wnGEaxVPKZOwlBuI0yytOFfgMybxnX2o"
NEEDED = ("generative_model_ema.npy", "args.pickle")

# Already vendored; checked, never fetched.
VENDORED = [os.path.join("tf_predict_%s" % p, f)
            for p in ("mu", "alpha", "gap")
            for f in ("model_ema_2000.npy", "args_2000.pickle")] + \
           [os.path.join("evaluate_%s" % p, f)
            for p in ("mu", "alpha", "gap")
            for f in ("best_checkpoint.npy", "args.pickle")]


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
    print("vendored property networks: %d/%d present"
          % (len(ok), len(VENDORED)))
    for rel in missing:
        print("   MISSING  %s" % rel)
    return not missing


def fetch_edm(force=False):
    target = os.path.join(DEST, "generative_model_ema.npy")
    if os.path.exists(target) and not force:
        print("EDMsecond already present at %s -- pass --force to refetch" % DEST)
        return True
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Either:\n"
              "    pip install gdown\n"
              "or download TFG's 'EDMsecond' folder by hand from the Drive link\n"
              "in audit/fa_fb_search/TFG/README.md and place\n"
              "    generative_model_ema.npy\n    args.pickle\n"
              "into %s" % DEST, file=sys.stderr)
        return False
    os.makedirs(DEST, exist_ok=True)
    print("downloading TFG EDMsecond (folder %s) -> %s" % (EDM_FOLDER_ID, DEST))
    gdown.download_folder(id=EDM_FOLDER_ID, output=DEST, quiet=False,
                          use_cookies=False)
    return True


def report():
    rows = []
    for name in NEEDED:
        p = os.path.join(DEST, name)
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
    ap.add_argument("--force", action="store_true",
                    help="refetch EDMsecond even if it is already present")
    ap.add_argument("--verify", action="store_true",
                    help="hash what is present and exit; download nothing")
    ap.add_argument("--manifest",
                    default=os.path.join(ROOT, "weights", "tfg_manifest.json"))
    args = ap.parse_args()

    have_props = check_vendored()
    if not args.verify:
        fetch_edm(force=args.force)

    rows = report()
    with open(args.manifest, "w") as fh:
        json.dump(rows, fh, indent=1)
    print("\n%-64s %-34s %s" % ("path", "md5", "bytes"))
    for r in rows:
        print("%-64s %-34s %s" % (r["path"], r["md5"] or "-- MISSING --",
                                  r["bytes"] if r["bytes"] else "-"))
    print("\nmanifest written to %s" % os.path.relpath(args.manifest, ROOT))

    have_edm = all(os.path.exists(os.path.join(DEST, n)) for n in NEEDED)
    if not (have_props and have_edm):
        print("\nNOT READY: the transfer sweep needs every row above to exist.")
        return 1
    print("\nREADY: run proj1/scripts/transfer_sweep.py --preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
