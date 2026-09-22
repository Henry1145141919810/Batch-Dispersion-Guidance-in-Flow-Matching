"""Download the raw QM9 archive this project was built on, and verify it.

    python proj1/scripts/download_qm9.py          # download, check, extract
    python proj1/scripts/download_qm9.py --verify # check what is already there

Produces `data/gdb9.sdf` and `data/gdb9.sdf.csv`, which
`proj1/scripts/prepare_qm9.py` then turns into `data/qm9.pt`.

WHICH QM9, AND WHY IT MATTERS. "QM9" names several distributions that are not
interchangeable. This project uses the **DeepChem / MoleculeNet** packaging:
a `gdb9.tar.gz` holding exactly `gdb9.sdf` (3D conformers, 133,885 molecules,
ids `gdb_1`...) and `gdb9.sdf.csv` (the DFT properties, keyed on `mol_id`, with
the column schema `mol_id,A,B,C,mu,alpha,homo,lumo,gap,...`).

Other packagings differ in ways that are silent rather than loud:

  * the **Figshare original** (`dsgdb9nsd.xyz.tar.bz2`) ships one .xyz per
    molecule with properties in the comment line, in different units and a
    different order -- `prepare_qm9.py` cannot read it at all;
  * the **Cormorant / EDM** download applies the official split and removes
    the 3,054 "uncharacterized" molecules. Our `prepare_qm9.py` keeps all
    133,885 and takes its own seeded random split, which is a documented
    deviation (`docs/results/BASE_MODEL_BENCHMARK.md` Section 2). Starting
    from their file would silently change the molecule set underneath every
    number in this repository.

So the archive is checked by SHA-256 rather than trusted. A different QM9 that
happens to parse would produce a different `qm9.pt`, a different train/val
split, different property targets and different `delta` -- and nothing would
raise. The hash is the only thing standing between a teammate and a set of
numbers that look fine and are not comparable to anyone else's.

IF THE URL HAS MOVED. It is a third-party S3 bucket and may not live forever.
The hashes below are the specification; the URL is a convenience. Obtain
`gdb9.tar.gz` from any MoleculeNet mirror, drop it in `data/`, and run
`--verify`. If the hash matches, it is the right file regardless of where it
came from.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tarfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data")

URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

# SHA-256 of the files this project's results were produced from. Recorded
# 22 Sep 2026 from the working copy that built data/qm9.pt.
EXPECTED = {
    "gdb9.tar.gz":  "45255048ac6d83ea4b923ecdf7d6fb6dc62bfec5e80fbc5bcfd93a62157a31db",
    "gdb9.sdf":     "98c4e97d50ac549b8c9f0b2114b348a9a944718e17e50d9a724b729f1deaa28e",
    "gdb9.sdf.csv": "73a67793e3cfa9660f001278bd019c143f57e4785db537a01811cf2ce72aa7eb",
}
# And of the tensor file prepare_qm9.py builds from them. Checked by
# --verify so a rebuild can be confirmed byte-identical to the one the
# published numbers used.
QM9_PT_SHA = "05f118203ace8b0e70855f34ba672ad69a1f616d0c9b9eaded9edf11631deffd"

EXTRACTED = ("gdb9.sdf", "gdb9.sdf.csv")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done = 0
    t0 = time.time()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
            done += len(b)
            if size > (1 << 26) and time.time() - t0 > 2:
                sys.stdout.write("\r    hashing %s ... %3.0f%%"
                                 % (os.path.basename(path), 100.0 * done / size))
                sys.stdout.flush()
                t0 = time.time()
    if size > (1 << 26):
        sys.stdout.write("\r" + " " * 60 + "\r")
    return h.hexdigest()


def check(name, quiet=False):
    """(exists, matches). Prints a line unless quiet."""
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        if not quiet:
            print("  %-14s MISSING" % name)
        return False, False
    got = sha256(p)
    want = EXPECTED[name]
    ok = got == want
    if not quiet:
        print("  %-14s %10.1f MB  %s"
              % (name, os.path.getsize(p) / 1e6,
                 "ok" if ok else "*** HASH MISMATCH ***"))
        if not ok:
            print("      expected %s" % want)
            print("      got      %s" % got)
    return True, ok


def download(url, dest):
    print("downloading %s" % url)
    print("  -> %s  (~45 MB)" % dest)
    t0 = time.time()

    def hook(blocks, bs, total):
        if total <= 0:
            return
        pct = 100.0 * min(blocks * bs, total) / total
        sys.stdout.write("\r  %3.0f%%  %.1f/%.1f MB  %.0fs"
                         % (pct, min(blocks * bs, total) / 1e6, total / 1e6,
                            time.time() - t0))
        sys.stdout.flush()

    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp, hook)
    except Exception as exc:                                  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print("\n  download failed: %r" % exc, file=sys.stderr)
        print("  The URL is a convenience, not the specification -- see this\n"
              "  script's docstring. Fetch gdb9.tar.gz from any MoleculeNet\n"
              "  mirror, put it in data/, and re-run with --verify.",
              file=sys.stderr)
        return False
    # Only move into place once it is complete, so an interrupted download
    # cannot leave a truncated file that looks present to the next run.
    os.replace(tmp, dest)
    print()
    return True


def extract(tgz):
    print("extracting")
    with tarfile.open(tgz, "r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if os.path.basename(m.name) in EXTRACTED]
        names = {os.path.basename(m.name) for m in members}
        missing = set(EXTRACTED) - names
        if missing:
            raise SystemExit(
                "archive does not contain %s -- this is not the DeepChem QM9 "
                "packaging (see this script's docstring)" % sorted(missing))
        for m in members:
            # Flatten: write to data/<basename>, never to a path the archive
            # chooses. Guards against an archive with absolute or ../ members.
            target = os.path.join(DATA, os.path.basename(m.name))
            src = tf.extractfile(m)
            if src is None:
                raise SystemExit("could not read %s from the archive" % m.name)
            with open(target, "wb") as out:
                while True:
                    chunk = src.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
            print("  %-14s %10.1f MB"
                  % (os.path.basename(m.name), os.path.getsize(target) / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="hash what is present and exit; download nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-download and re-extract even if files are present")
    ap.add_argument("--url", default=URL)
    ap.add_argument("--keep-archive", action="store_true",
                    help="keep gdb9.tar.gz after extracting (default: keep)")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    tgz = os.path.join(DATA, "gdb9.tar.gz")

    if args.verify:
        print("checking data/ against the recorded hashes")
        allok = True
        for name in ("gdb9.tar.gz",) + EXTRACTED:
            exists, ok = check(name)
            allok = allok and exists and ok
        qp = os.path.join(DATA, "qm9.pt")
        if os.path.exists(qp):
            got = sha256(qp)
            same = got == QM9_PT_SHA
            print("  %-14s %10.1f MB  %s"
                  % ("qm9.pt", os.path.getsize(qp) / 1e6,
                     "ok -- identical to the file the published numbers used"
                     if same else "differs from the recorded build"))
            if not same:
                print("      A different qm9.pt means a different molecule set "
                      "or split.\n      Every number in docs/results/ assumes "
                      "the recorded one.")
        else:
            print("  %-14s not built yet -- run prepare_qm9.py" % "qm9.pt")
        return 0 if allok else 1

    have_extracted = all(os.path.exists(os.path.join(DATA, n))
                         for n in EXTRACTED)
    if have_extracted and not args.force:
        print("data/gdb9.sdf and data/gdb9.sdf.csv already present; verifying")
        ok = all(check(n)[1] for n in EXTRACTED)
        if ok:
            print("\nREADY: python proj1/scripts/prepare_qm9.py")
            return 0
        print("\nHashes do not match. Re-run with --force to replace them.")
        return 1

    if not os.path.exists(tgz) or args.force:
        if not download(args.url, tgz):
            return 1
    else:
        print("data/gdb9.tar.gz already present; verifying before extracting")

    exists, ok = check("gdb9.tar.gz")
    if not ok:
        print("\nThe archive does not match the recorded hash. It is either a\n"
              "different QM9 packaging or a corrupted download; either way it\n"
              "would silently change every number in this repository. Not\n"
              "extracting. See this script's docstring.")
        return 1

    extract(tgz)
    print("\nverifying extracted files")
    allok = all(check(n)[1] for n in EXTRACTED)
    if not allok:
        return 1
    print("\nREADY: python proj1/scripts/prepare_qm9.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
