"""Backfill the stored sequences for M2 cells that predate sample storage.

Cells written before 27 Sep kept only the scored property and the decode
confidence per sequence, so they could not answer a question they had not been
asked at run time. m2_sweep now stores the decoded sequences themselves.

This regenerates each existing cell into a scratch directory, CHECKS THAT THE
REGENERATED JSON MATCHES THE ORIGINAL -- which is the point, since it proves
the stored samples are the ones the published numbers came from -- and only
then moves the new per-sequence file into place. The original JSON is never
touched, so a mismatch costs nothing but a log line.

    python proj1/m2/backfill_samples.py --stage m2 --device cuda
    python proj1/m2/backfill_samples.py --stage m2 --seeds 20260921 --device cuda
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

KEYS = ["in_band_fraction", "gc_mean", "gc_sd", "kmer_js", "diversity",
        "bias_delta", "clipped_sample_steps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/m2")
    ap.add_argument("--stage", default="m2")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="proj1/m2/blade_bundle/fm_m2_dfb500.pt")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    seeds = {int(s) for s in a.seeds.split(",") if s.strip()}

    todo = []
    for p in sorted(glob.glob(os.path.join(a.out_dir, a.stage, "n%d" % a.n,
                                           "seed*", "*.json"))):
        r = json.load(open(p))
        if seeds and r["seed"] not in seeds:
            continue
        pm = p.replace(".json", ".permol.pt")
        if os.path.exists(pm):
            import torch
            try:
                if "tok" in torch.load(pm, map_location="cpu"):
                    continue                      # already has its sequences
            except Exception:                     # noqa: BLE001
                pass
        todo.append((p, r))
    print("%d cells to backfill in %s/%s/n%d" % (len(todo), a.out_dir, a.stage, a.n))
    if a.dry or not todo:
        return 0

    ok = bad = 0
    tmp = tempfile.mkdtemp(prefix="m2backfill_")
    try:
        for i, (p, r) in enumerate(todo, 1):
            cmd = [sys.executable, "proj1/m2/m2_sweep.py", "--ckpt", a.ckpt,
                   "--prop", r["prop"], "--arm", r["arm"],
                   "--variant", str(r.get("variant", "-")),
                   "--w", str(r["w"]), "--target", r["target_name"],
                   "--seed", str(r["seed"]), "--t-min", str(r["t_min_guide"]),
                   "--n", str(r["n"]), "--batch", str(r["batch"]),
                   "--steps", str(r["steps"]), "--stage", a.stage,
                   "--device", a.device, "--out-dir", tmp]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print("[%d/%d] FAIL %s" % (i, len(todo), os.path.basename(p)))
                print("   ", (res.stderr.strip().splitlines() or ["?"])[-1][:160])
                bad += 1
                continue
            new = os.path.join(tmp, a.stage, "n%d" % r["n"],
                               "seed%d" % r["seed"], os.path.basename(p))
            if not os.path.exists(new):
                print("[%d/%d] FAIL no regenerated cell" % (i, len(todo)))
                bad += 1
                continue
            nr = json.load(open(new))
            d = max(abs(nr.get(k, 0) - r.get(k, 0)) for k in KEYS)
            if d > 1e-9:
                # The regenerated batch is NOT the published one, so its samples
                # would misrepresent the numbers already reported. Leave it.
                print("[%d/%d] MISMATCH %s max|d|=%.3e -- not replacing"
                      % (i, len(todo), os.path.basename(p), d))
                bad += 1
                continue
            shutil.copyfile(new.replace(".json", ".permol.pt"),
                            p.replace(".json", ".permol.pt"))
            ok += 1
            if i % 10 == 0 or i == len(todo):
                print("[%d/%d] %d verified and backfilled, %d skipped"
                      % (i, len(todo), ok, bad), flush=True)
            for f in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
                if os.path.isfile(f):
                    os.remove(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nbackfilled %d, skipped %d" % (ok, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
