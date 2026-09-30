#!/usr/bin/env python
"""How much does a cell move when NOTHING changes? The run-to-run envelope.

The seed fixes the initial noise, not the trajectory: the EGNN's scatter kernels
use atomics, so two identical runs of the same arm on the same seed do not agree
bitwise, and the velocity-relative clip turns a 1e-8 difference into a DIFFERENT
CLIP DECISION and from there a different molecule.

This matters for one pre-registered claim in particular. BDG at eta = 0 is plug:
`test_bdg.py` A9 and B-a prove the guidance field is BIT-identical, max err
0.00e+00, on a real x_t on CUDA. It is tempting to also assert that a bdg_e0
CELL equals a plug cell. That assertion would be testing the GPU. This script
measures the envelope so the gate can be stated at the level it actually holds:

    run the SAME arm twice  -> |A - B| is what "no change" looks like
    run the comparison arm  -> |A - C| is the difference under test

usage:
    python proj1/scripts/repro_envelope.py --arm plug --against bdg_e0t1
    python proj1/scripts/repro_envelope.py --arm plug --repeats 4   # tighter envelope

A single pair is a very noisy estimate of a spread, so --repeats 3+ is worth the
minutes if anything is going to be concluded from a field being "outside".
"""
import argparse
import glob
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TS = os.path.join("proj1", "scripts", "transfer_sweep.py")
SKIP = {"seconds"}          # wall-clock is not a property of the sampler


def run(a, arm, tag):
    out = os.path.join("results", "_repro", tag)
    full = os.path.join(ROOT, out)
    if os.path.isdir(full):
        import shutil
        shutil.rmtree(full)     # transfer_sweep resumes by file existence
    cmd = [sys.executable, TS, "--stage", "v3", "--backend", a.backend,
           "--props", a.prop, "--arms", arm, "--n", str(a.n),
           "--batch", str(a.n), "--steps", str(a.steps), "--seed", str(a.seed),
           "--fm-ckpt", a.fm_ckpt, "--out-dir", out]
    p = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                       capture_output=True, text=True)
    hits = glob.glob(os.path.join(full, "*.json"))
    if not hits:
        sys.stdout.write("  %s FAILED\n" % tag)
        for line in [l for l in (p.stderr or "").splitlines() if l.strip()][-4:]:
            sys.stdout.write("    | %s\n" % line[:160])
        return None
    return json.load(open(hits[0]))


def numeric(d):
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and k not in SKIP}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="plug", help="run this arm --repeats times")
    ap.add_argument("--against", default="bdg_e0t1",
                    help="the arm whose difference is under test ('' to skip)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--backend", default="fm")
    ap.add_argument("--prop", default="mu")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--fm-ckpt", default="proj1/checkpoints/fm_last.pt")
    ap.add_argument("--json-out", default="results/repro_envelope.json")
    ap.add_argument("--md-out", default="")
    a = ap.parse_args()
    if a.repeats < 2:
        raise SystemExit("--repeats must be >= 2: the envelope IS the repeat")

    print("envelope: %s x %d, then %s, all at seed %d, n=%d, %d steps"
          % (a.arm, a.repeats, a.against or "(nothing)", a.seed, a.n, a.steps))
    reps = [run(a, a.arm, "%s_%d" % (a.arm, i)) for i in range(a.repeats)]
    reps = [r for r in reps if r]
    if len(reps) < 2:
        raise SystemExit("need >= 2 successful repeats, got %d" % len(reps))
    other = run(a, a.against, "against") if a.against else None

    keys = sorted(set(numeric(reps[0])).intersection(*[set(numeric(r)) for r in reps[1:]]))
    rows = []
    for k in keys:
        vals = [r[k] for r in reps]
        env = max(abs(x - y) for x, y in itertools.combinations(vals, 2))
        d = abs(other[k] - vals[0]) if (other and k in numeric(other)) else None
        rows.append({"field": k, "envelope": env, "delta": d,
                     "outside": (d is not None and d > env)})

    moved = [r for r in rows if r["envelope"] > 0]
    print("\n%-28s %13s %13s" % ("field", "envelope", a.against or ""))
    for r in rows:
        if r["envelope"] or r["delta"]:
            print("%-28s %13.3e %13s %s"
                  % (r["field"], r["envelope"],
                     "%.3e" % r["delta"] if r["delta"] is not None else "-",
                     "OUTSIDE" if r["outside"] else ""))
    print("\n%d of %d fields MOVE when nothing changes (%d repeats of %s)"
          % (len(moved), len(rows), len(reps), a.arm))
    rec = {"arm": a.arm, "against": a.against, "repeats": len(reps),
           "backend": a.backend, "prop": a.prop, "n": a.n, "steps": a.steps,
           "seed": a.seed, "rows": rows,
           "n_fields": len(rows), "n_moving": len(moved),
           "n_outside": sum(1 for r in rows if r["outside"])}
    if other:
        print("%d of %d fields put %s OUTSIDE that envelope"
              % (rec["n_outside"], len(rows), a.against))
    jo = os.path.join(ROOT, a.json_out)
    os.makedirs(os.path.dirname(jo), exist_ok=True)
    json.dump(rec, open(jo, "w"), indent=2, sort_keys=True)
    print("wrote %s" % a.json_out)
    if a.md_out:
        md = os.path.join(ROOT, a.md_out)
        os.makedirs(os.path.dirname(md), exist_ok=True)
        open(md, "w").write(render(rec))
        print("wrote %s" % a.md_out)
    return 0


def render(rec):
    L = ["# The run-to-run envelope: what a cell does when nothing changes", "",
         "Generated by `proj1/scripts/repro_envelope.py`. %d repeats of `%s` at "
         "one seed (%d), %s, %s, n = %d, %d steps -- then `%s` for comparison."
         % (rec["repeats"], rec["arm"], rec["seed"], rec["backend"], rec["prop"],
            rec["n"], rec["steps"], rec["against"]), "",
         "**%d of %d numeric fields move when nothing changes.** The seed fixes "
         "the initial noise; it does not fix the trajectory, because the EGNN's "
         "scatter kernels use atomics and the velocity-relative clip is a "
         "threshold -- once it fires on a different step the molecule differs."
         % (rec["n_moving"], rec["n_fields"]), "",
         "| field | envelope (same arm, %d runs) | `%s` - `%s` | outside? |"
         % (rec["repeats"], rec["against"], rec["arm"]),
         "|---|---|---|---|"]
    for r in rec["rows"]:
        if r["envelope"] or r["delta"]:
            L.append("| `%s` | %.3e | %s | %s |"
                     % (r["field"], r["envelope"],
                        "%.3e" % r["delta"] if r["delta"] is not None else "-",
                        "**yes**" if r["outside"] else ""))
    L += ["", "%d of %d fields fall outside the envelope. With only %d repeats "
              "the envelope is itself a noisy estimate, so 'outside' here is "
              "not evidence of a real difference -- and for `%s` it cannot be, "
              "because `test_bdg.py` A9 and B-a prove the guidance field is "
              "bit-identical to `%s` (max err 0.00e+00) on a real `x_t` on CUDA. "
              "What this table measures is the amplification, not a discrepancy."
          % (rec["n_outside"], rec["n_fields"], rec["repeats"], rec["against"],
             rec["arm"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
