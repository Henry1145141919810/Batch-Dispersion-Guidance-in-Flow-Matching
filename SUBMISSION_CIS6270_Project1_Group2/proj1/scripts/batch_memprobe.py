#!/usr/bin/env python
"""How large a batch fits, and therefore how many BDG controllers a cell runs.

BDG's controller state is estimated over whatever batch the sampler is handed
(`V_b` in guidance.py), so the batch is not a performance knob for this method --
it is the estimator's sample size. guidance.py asks for one batch per cell. That
is not affordable: this script is what measured why.

    peak allocation is LINEAR in the batch and FLAT in the step count

so a few small batches fit a line that extrapolates to any batch, and the
largest divisor of n that fits the card is the answer. On an RTX 5080 the fit
said one batch of 5000 would need ~107 GiB on our base and ~197 GiB on EquiFM,
against Betty's 45 GB slice -- so v3 runs ten controllers of 500, and says so.

RUN THIS ON BETTY BEFORE THE FULL RUN. The laptop fit is an extrapolation from a
16 GB card to a 45 GB slice; --confirm measures the real thing directly, which is
worth two minutes against a 60-hour run:

    python proj1/scripts/batch_memprobe.py --confirm 500          # the real test
    python proj1/scripts/batch_memprobe.py --batches 32,64,128    # refit

Peak allocation is per process, so another job sharing the card changes whether
you OOM but not what this prints. Wall-clock is NOT reported: it is worthless on
a contended card (the same cell timed 17.9 s and then 12.2 s on the laptop).
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TS = os.path.join("proj1", "scripts", "transfer_sweep.py")


def child(a):
    """Run ONE cell, then print this process's peak allocation as JSON."""
    import atexit

    import torch

    def report():
        d = {"backend": a.backend, "batch": a.batch,
             "alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
             "resv_gib": torch.cuda.max_memory_reserved() / 2 ** 30}
        sys.stdout.write("PROBE_JSON " + json.dumps(d) + "\n")

    atexit.register(report)
    # A short out-dir on purpose: a cell's filename is ~130 characters and
    # Windows still refuses paths over 260, which is how this probe first
    # "failed" -- after doing all the work.
    #
    # AND IT IS EMPTIED FIRST. transfer_sweep resumes by file existence, so a
    # second probe at the same batch found its cell already there, skipped the
    # work, and reported a peak of ZERO -- which the fit read as "no memory per
    # molecule" and turned into "--batch 5000 fits". A silent zero is the one
    # answer this script must never give, so the cell is cleared here and a zero
    # is refused in probe() as well.
    out = os.path.join("results", "_memprobe", a.backend)
    if os.path.isdir(out):
        import shutil
        shutil.rmtree(out)
    sys.argv = [TS, "--stage", "v3", "--backend", a.backend,
                "--props", a.prop, "--arms", a.arm,
                "--n", str(a.batch), "--batch", str(a.batch),
                "--steps", str(a.steps), "--seed", str(a.seed),
                "--fm-ckpt", a.fm_ckpt,
                "--out-dir", os.path.join("results", "_memprobe", a.backend)]
    import runpy
    runpy.run_path(TS, run_name="__main__")


def probe(a, backend, batch):
    """Spawn one child; return its peak allocation, or None if it OOMed."""
    cmd = [sys.executable, os.path.abspath(__file__), "--child",
           "--backend", backend, "--batch", str(batch), "--arm", a.arm,
           "--prop", a.prop, "--steps", str(a.steps), "--seed", str(a.seed),
           "--fm-ckpt", a.fm_ckpt]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    out = (p.stdout or "") + (p.returncode and (p.stderr or "") or "")
    m = re.search(r"PROBE_JSON (\{.*\})", out)
    if not m:
        oom = "out of memory" in out.lower() or "OutOfMemory" in out
        print("  %-7s batch %5d  %s" % (backend, batch, "OOM" if oom else "FAILED"))
        if not oom:
            tail = [l for l in (p.stderr or "").splitlines() if l.strip()][-3:]
            for l in tail:
                print("      | %s" % l[:150])
        return None
    r = json.loads(m.group(1))
    if r["alloc_gib"] <= 0.0:
        # No GPU work happened: a resumed cell, or torch fell back to CPU. Either
        # way the number is not a measurement, and a zero slope would recommend
        # the largest batch there is.
        print("  %-7s batch %5d  ZERO peak -- no GPU work was done. The cell was"
              " already on disk (probe cells are cleared, so this means a stale"
              " tree elsewhere) or torch has no CUDA device." % (backend, batch))
        return None
    print("  %-7s batch %5d  peak alloc %7.3f GiB   reserved %7.3f GiB"
          % (backend, batch, r["alloc_gib"], r["resv_gib"]))
    return r


def fit(points):
    """Least-squares GiB-per-molecule and intercept over the probed batches."""
    n = len(points)
    xs = [p["batch"] for p in points]
    out = {}
    for key in ("alloc_gib", "resv_gib"):
        ys = [p[key] for p in points]
        mx, my = sum(xs) / n, sum(ys) / n
        var = sum((x - mx) ** 2 for x in xs)
        slope = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
                 if var else 0.0)
        out[key] = {"per_mol": slope, "intercept": my - slope * mx}
    return out


def divisors(n):
    return sorted(d for d in range(1, n + 1) if n % d == 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", default="fm,equifm")
    ap.add_argument("--batches", default="32,64,128",
                    help="batches to fit the line over (peak is linear in this)")
    ap.add_argument("--confirm", type=int, default=0,
                    help="instead of fitting, run ONE cell at this batch and "
                         "report whether it fits. This is the real test; run it "
                         "on the machine that will do the full run")
    ap.add_argument("--arm", default="bdg_e4t0.5",
                    help="a BDG arm: the controller's VJP is the memory peak")
    ap.add_argument("--prop", default="mu")
    ap.add_argument("--steps", type=int, default=10,
                    help="peak allocation is flat in the step count, so 10 is "
                         "as informative as 100 and far cheaper")
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--n", type=int, default=5000, help="the run's n per cell")
    ap.add_argument("--vram-gb", type=float, default=45.0,
                    help="the target card in GB, as the MIG profile names it "
                         "(b200-mig45 = 45 GB). Converted to GiB internally: a "
                         "45 GB slice is 41.9 GiB, and reading it as 45 GiB is a "
                         "7 %% overstatement that is enough to pick a batch "
                         "which OOMs -- it recommended 625 (35.0 GiB) against a "
                         "real budget of 33.5")
    ap.add_argument("--headroom", type=float, default=0.80,
                    help="fraction of VRAM a batch may reserve")
    ap.add_argument("--json-out", default="results/v3_batch_memory.json")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--fm-ckpt", default="proj1/checkpoints/fm_last.pt")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--backend", default="fm", help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=64, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.child:
        return child(a)

    if not os.path.exists(os.path.join(ROOT, a.fm_ckpt)):
        raise SystemExit("--fm-ckpt %s not found (cwd %s)" % (a.fm_ckpt, ROOT))
    backends = [b for b in a.backends.split(",") if b]
    vram_gib = a.vram_gb * 1e9 / 2 ** 30      # GB -> GiB; see --vram-gb
    budget = vram_gib * a.headroom
    rec = {"arm": a.arm, "prop": a.prop, "steps": a.steps, "n": a.n,
           "vram_gb": a.vram_gb, "vram_gib": vram_gib, "headroom": a.headroom,
           "budget_gib": budget, "points": [], "fits": {}, "confirm": {}}

    if a.confirm:
        print("CONFIRM: one cell at batch %d on this machine" % a.confirm)
        if a.n % a.confirm:
            print("  WARNING: %d does not divide n=%d -- a remainder batch is a"
                  " second, noisier controller" % (a.confirm, a.n))
        for be in backends:
            r = probe(a, be, a.confirm)
            rec["confirm"][be] = r      # None means it did not fit
            if r:
                rec["points"].append(r)
    else:
        for be in backends:
            pts = []
            for b in [int(x) for x in a.batches.split(",") if x]:
                r = probe(a, be, b)
                if r:
                    pts.append(r)
            rec["points"].extend(pts)
            if len(pts) < 2:
                print("  %s: need 2 probed batches to fit a line" % be)
                continue
            f = fit(pts)
            rec["fits"][be] = f
            per = f["resv_gib"]["per_mol"]
            biggest = [d for d in divisors(a.n)
                       if per * d + f["resv_gib"]["intercept"] <= budget]
            rec["fits"][be]["largest_divisor_of_n"] = biggest[-1] if biggest else 0
            rec["fits"][be]["one_batch_resv_gib"] = per * a.n
            print("  %-7s %.4f GiB reserved/molecule -> one batch of %d needs"
                  " %.0f GiB; largest divisor of %d fitting %.1f GiB: %s"
                  % (be, per, a.n, per * a.n, a.n, budget,
                     biggest[-1] if biggest else "NONE"))

    if rec["fits"]:
        agreed = [v["largest_divisor_of_n"] for v in rec["fits"].values()]
        rec["recommended_batch"] = min(agreed) if agreed else 0
        print("\nRECOMMENDED --batch %s (the tightest backend governs; a cell "
              "then runs %s controllers)"
              % (rec["recommended_batch"],
                 a.n // rec["recommended_batch"] if rec["recommended_batch"] else "?"))

    jo = os.path.join(ROOT, a.json_out)
    os.makedirs(os.path.dirname(jo), exist_ok=True)
    with open(jo, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    print("wrote %s" % a.json_out)
    if a.md_out:
        md = os.path.join(ROOT, a.md_out)
        os.makedirs(os.path.dirname(md), exist_ok=True)
        with open(md, "w") as fh:
            fh.write(render(rec, a))
        print("wrote %s" % a.md_out)
    return 0


def render(rec, a):
    L = ["# How large a batch fits, and how many BDG controllers a cell runs",
         "",
         "Generated by `proj1/scripts/batch_memprobe.py`. Peak allocation is "
         "per process, linear in the batch and flat in the step count, so a "
         "contended card changes whether a probe OOMs but not what it "
         "measures.", "",
         "Probe: arm `%s`, property %s, %d steps, seed %d. Target card "
         "%.0f GB = %.1f GiB, usable at %.0f%%: %.1f GiB."
         % (rec["arm"], rec["prop"], rec["steps"], a.seed, rec["vram_gb"],
            rec["vram_gib"], rec["headroom"] * 100, rec["budget_gib"]),
         "", "| backend | batch | peak allocated | peak reserved |",
         "|---|---|---|---|"]
    for p in rec["points"]:
        L.append("| %s | %d | %.3f GiB | %.3f GiB |"
                 % (p["backend"], p["batch"], p["alloc_gib"], p["resv_gib"]))
    if rec["fits"]:
        L += ["", "| backend | GiB reserved / molecule | one batch of %d |"
                  " largest divisor of %d that fits |" % (rec["n"], rec["n"]),
              "|---|---|---|---|"]
        for be, f in sorted(rec["fits"].items()):
            L.append("| %s | %.4f | %.0f GiB | **%d** |"
                     % (be, f["resv_gib"]["per_mol"], f["one_batch_resv_gib"],
                        f["largest_divisor_of_n"]))
        b = rec.get("recommended_batch", 0)
        if b:
            L += ["", "**Recommended `--batch %d`** -- the tightest backend "
                      "governs, and a cell then runs %d independent "
                      "controllers, each estimating `V_b` from %d samples "
                      "(a %.1f %% standard error on the variance)."
                  % (b, rec["n"] // b, b, 100.0 * (2.0 / (b - 1)) ** 0.5)]
        else:
            L += ["", "**No divisor of %d fits.** Lower `--n`." % rec["n"]]
    if rec["confirm"]:
        L.append("")
        for be, r in sorted(rec["confirm"].items()):
            L.append("- Confirmed on this machine: %s at batch %d %s."
                     % (be, a.confirm,
                        ("reserved %.1f GiB of a %.1f GiB budget"
                         % (r["resv_gib"], rec["budget_gib"])) if r
                        else "**DID NOT FIT**"))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
