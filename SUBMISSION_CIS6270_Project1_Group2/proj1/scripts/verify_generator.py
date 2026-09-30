"""Is this checkpoint the generator every cell in this project was produced with?

    python proj1/scripts/verify_generator.py                    # resolve and check
    python proj1/scripts/verify_generator.py --ckpt weights/fm_ema.pt
    python proj1/scripts/verify_generator.py --print-path       # just the path

Exit 0 if the checkpoint holds the pinned model, 1 otherwise, with a message
that says what to do about it.

WHY THIS IS NOT `md5sum`. The pin is the model, not the file. Two files in
this project carry the SAME EMA weights under different names:

  proj1/checkpoints/fm_last.pt   the training checkpoint: model + ema + opt +
                                 sched + rng. 57.5 MB. md5 a190ac83...
                                 NOT in the repository -- .gitignore excludes
                                 it, and it is the one only Betty has.
  weights/fm_ema.pt              the same EMA tensors with the optimiser and
                                 scheduler state dropped. 14.35 MB, tracked,
                                 in every clone. weights/README.md records
                                 `max |w_slim - w_full| = 0.0` across every
                                 parameter, and the file stores
                                 `source_md5 = a190ac83...` to say which
                                 checkpoint it came from.

A file-md5 gate accepts the first and rejects the second, so a teammate with
a clean clone -- who has the right weights -- is told the generator is wrong.
That is what this script exists to stop. It accepts either: the file's own
md5, or `source_md5` recorded inside it.

It does NOT accept a checkpoint that merely looks similar. `source_md5` is
written by the stripping script and names the exact parent; anything else
fails, so a different training run cannot slip through under the same name.
"""
import argparse
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

# The generator every cell in this project was produced with: our
# flow-matching EGNN, hidden 256, 8 layers, train_a, seed 20260918, epoch 1500
# EMA. Pinned because a different generator would make v3's backends
# incomparable to each other and to v1/v2.
PINNED_MD5 = "a190ac8394902027d4a951f8d30e8c5c"


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def check(path, pinned=PINNED_MD5):
    """-> (ok, how, detail). `how` is 'file md5' or 'source_md5'."""
    if not os.path.exists(path):
        return False, "missing", "no such file"
    got = file_md5(path)
    if got == pinned:
        return True, "file md5", got
    # not the full checkpoint -- is it the slim one stripped from it?
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:                       # noqa: BLE001
        return False, "unreadable", "file md5 %s, and torch.load failed: %s" % (got, exc)
    if not isinstance(ck, dict):
        return False, "not a checkpoint", "file md5 %s" % got
    src = ck.get("source_md5")
    if src == pinned:
        return True, "source_md5", "file md5 %s, stripped from %s" % (got, src)
    return False, "mismatch", ("file md5 %s, source_md5 %r" % (got, src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="",
                    help="default: checkpoint_paths.default_generator()")
    ap.add_argument("--pinned", default=PINNED_MD5)
    ap.add_argument("--print-path", action="store_true",
                    help="print the resolved path and exit 0, no checking")
    a = ap.parse_args()

    if a.ckpt:
        path = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(ROOT, a.ckpt)
    else:
        from checkpoint_paths import default_generator
        path = default_generator()

    if a.print_path:
        print(path)
        return 0

    ok, how, detail = check(path, a.pinned)
    rel = os.path.relpath(path, ROOT)
    if ok:
        print("generator OK: %s (%s matches the pin %s)" % (rel, how, a.pinned[:8]))
        return 0

    print("FATAL: %s is not the pinned generator." % rel, file=sys.stderr)
    print("  %s: %s" % (how, detail), file=sys.stderr)
    print("  expected the model whose checkpoint md5 is %s" % a.pinned, file=sys.stderr)
    print("", file=sys.stderr)
    print("Every cell in this project used that generator; a different one", file=sys.stderr)
    print("would make the backends incomparable to each other and to v1/v2.", file=sys.stderr)
    print("", file=sys.stderr)
    if how == "missing":
        print("From a clean clone, weights/fm_ema.pt is the one you want and", file=sys.stderr)
        print("it is already tracked -- it holds the same EMA tensors with the", file=sys.stderr)
        print("optimiser state dropped. Check that the clone is complete.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
