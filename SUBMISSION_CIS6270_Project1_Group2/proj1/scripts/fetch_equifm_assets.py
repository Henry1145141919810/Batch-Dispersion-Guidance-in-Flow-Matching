"""Download and verify the EquiFM generator and OC-Flow's property networks.

`audit/` is git-ignored, so a fresh clone has none of the files the EquiFM
transfer backend (proj1/src/external/equifm_backend.py) loads. This script
fetches exactly those files from the OC-Flow repository at the commit the
usability audit pinned (docs/protocol/EQUIFM_USABILITY_AUDIT.md), checks each
against its recorded SHA-256, and writes them where the backend looks:

    audit/equifm_20260922/OC-Flow/<path in the repo>

    python proj1/scripts/fetch_equifm_assets.py            # fetch what is missing
    python proj1/scripts/fetch_equifm_assets.py --verify   # hash what is present

The EquiFM generator in this repo is byte-identical to the one in MolFM's
sampling folder and in the NeurIPS supplement (SHA-256 47b40ae6...), which is
why one source suffices. TFG's guide and oracle are fetched separately by
fetch_tfg_assets.py (they are tracked in git).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEST = os.path.join(ROOT, "audit", "equifm_20260922", "OC-Flow")
REPO = "WangLuran/Guided-Flow-Matching-with-Optimal-Control"
COMMIT = "ca218ba616f7e3a58a05924fdb547bd579b3c900"

# path in the repo -> sha256, from audit/equifm_20260922/manifest.json
FILES = {
    "molecule/equifm/args.pickle":
        "cb8c719d5ff72efb9c96bf9b4659c3dbbb034a354a31f072680109f557e13468",
    "molecule/equifm/cnf_models.py":
        "0d3e9d7886601e2414804823fb95fd9576158047f3106c450bd98f1373a39328",
    "molecule/equifm/egnn.py":
        "8df3567f9cf18f0ea82a0bdd39ba07f582b033d71e574a4e755b4065dd6d31d6",
    "molecule/equifm/generative_model_ema_0.npy":
        "47b40ae673c1117219432f0c81ae61f72ac380761200144ed04ebb9f179d3527",
    "molecule/qm9/property_prediction/models/gcl.py":
        "05c956dab2c8d4a7e3d7297aa0325629f1077cb8d1d2b1074206f98497a97575",
    "molecule/qm9/property_prediction/models_property.py":
        "147269527b348fd58215fa1378769617d64cdac545890fc91e0b5abefd64f208",
    "molecule/qm9/property_prediction/outputs/exp_class_mu/args.pickle":
        "d07f33d1cf51a223ddfc025d90ae6ebeb30e7ca72300f2c7eabfb04d63443db1",
    "molecule/qm9/property_prediction/outputs/exp_class_mu/best_checkpoint.npy":
        "4928eb07caa6f4b0202d101ab698d2ed5521c0bf9a61f30ba68ffa9612820337",
    "molecule/qm9/property_prediction/outputs/exp_class_alpha/args.pickle":
        "ea84a20522042639674cb58a82dfe884a7dd3b6e321be224e81bcdf745bb57ec",
    "molecule/qm9/property_prediction/outputs/exp_class_alpha/best_checkpoint.npy":
        "dae7c3e79fb5e64d7241882fc3b08ac5a0ee316a7bd8dfca8414cdfd190ab070",
    "molecule/qm9/property_prediction/outputs/exp_class_gap/args.pickle":
        "ebf363333f8760aced04fe38ac9c55bd6422b823dadfdff5ed9f00969568562f",
    "molecule/qm9/property_prediction/outputs/exp_class_gap/best_checkpoint.npy":
        "1b08735b5d32d7c3a6b59eb09c5f7103ff05bb3aece304027f2a9a65671ecf74",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def download(path):
    """Raw file bytes at the pinned commit; follows a Git LFS pointer."""
    urls = ["https://raw.githubusercontent.com/%s/%s/%s" % (REPO, COMMIT, path),
            "https://media.githubusercontent.com/media/%s/%s/%s" % (REPO, COMMIT, path)]
    last = None
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
        except Exception as exc:                      # noqa: BLE001
            last = exc
            continue
        if data.startswith(b"version https://git-lfs"):
            continue                                  # a pointer: try the media URL
        return data
    raise RuntimeError("could not fetch %s (%r)" % (path, last))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify", action="store_true",
                    help="hash the files present and exit; download nothing")
    ap.add_argument("--dest", default=DEST)
    args = ap.parse_args()

    bad = 0
    for path, want in FILES.items():
        out = os.path.join(args.dest, *path.split("/"))
        if os.path.exists(out):
            got = sha256(open(out, "rb").read())
            status = "ok" if got == want else "HASH MISMATCH"
        elif args.verify:
            status = "MISSING"
        else:
            data = download(path)
            if sha256(data) != want:
                status = "HASH MISMATCH (download)"
            else:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out + ".part", "wb") as fh:
                    fh.write(data)
                os.replace(out + ".part", out)
                status = "fetched"
        bad += status not in ("ok", "fetched")
        print("%-24s %s" % (status, path))
    if bad:
        print("\nNOT READY: %d file(s) missing or mismatched" % bad)
        return 1
    print("\nREADY: EquiFM backend assets verified (%d files)" % len(FILES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
