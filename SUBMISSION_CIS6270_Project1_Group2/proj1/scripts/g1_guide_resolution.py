"""G1 -- the guide-resolution gate.

    python proj1/scripts/g1_guide_resolution.py --props mu,alpha,gap

GUIDANCE_EXPERIMENT_PLAN.md section 3 calls this "the single most important
check in the plan" and it has never been run.

THE QUESTION. Every arm builds its field by differentiating f_A. If two arms'
fields differ by less than the noise f_A itself carries, then the 390-cell arm
comparison is ranking arms on the guide's measurement error rather than on any
property of the methods, and no amount of extra sampling fixes that.

THE TEST, and why this form of it is decisive. "The guide's error" is 0.0897 D
on mu -- a scalar, in Debye, which is not directly comparable to a field in
state space. But we have TWO guides: f_A and f_B are the same architecture
trained on disjoint halves (SPLIT_PROTOCOL.md), with val MAE 0.0897 and 0.0840.
Their disagreement IS the guide-error scale, expressed in exactly the space the
fields live in. So:

    between-arm spread    cos( G_armI(f_A), G_armJ(f_A) )   same guide, two arms
    between-guide spread  cos( G_arm(f_A),  G_arm(f_B)  )   same arm, two guides

and the gate reads:

    between-guide agreement < between-arm agreement
        -> swapping the GUIDE changes the field more than swapping the ARM.
           The arm comparison is measuring f_A, not the methods. FAIL.

    between-guide agreement > between-arm agreement
        -> the arms are genuinely further apart than the guide's own noise,
           and the grid resolves them. PASS.

States come from real unguided sampling trajectories, snapshotted inside the
guided window, so the fields are evaluated where the sampler actually goes
rather than at noised data points that no trajectory visits.

f_B IS USED HERE AND ONLY HERE AS A PERTURBATION. It is not scoring anything,
so this does not spend the evaluator's independence -- no sample is selected,
ranked or reported on the basis of f_B in this script.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from checkpoint_paths import default_generator  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from evaluation import choose_delta  # noqa: E402
from guidance import fm_posterior, guidance_field, Cost  # noqa: E402
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from sampling import FlowSampler, initial_noise  # noqa: E402


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "guidance_sweep", os.path.join(HERE, "guidance_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GS = _load_sweep()
DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT = os.path.join(ROOT, "proj1", "checkpoints")


def flat(c, f):
    return torch.cat([c.reshape(c.shape[0], -1), f.reshape(f.shape[0], -1)], 1)


def cos(a, b):
    an, bn = a.norm(dim=1).clamp(min=1e-30), b.norm(dim=1).clamp(min=1e-30)
    return ((a * b).sum(1) / (an * bn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=None,
                    help="generator checkpoint; default resolves betty_pull -> "
                         "proj1/checkpoints -> weights, so a fresh clone works")
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--arms", default="plug,tmpd,smg,smg2,btvg,btvg_var")
    ap.add_argument("--times", default="0.5,0.6,0.7,0.8,0.9,0.95")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--n-probe", type=int, default=1)
    ap.add_argument("--k-delta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "g1_guide_resolution.json"))
    args = ap.parse_args()
    if args.fm is None:
        args.fm = default_generator()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a]
    times = [float(t) for t in args.times.split(",") if t]

    d = torch.load(DATA, weights_only=False)
    types = d["types"]
    net, ck = load_fm(args.fm, len(types), dev)
    sel = d["split"]["val"][: args.n]
    mask = d["mask"][sel].to(dev)
    print("G1 guide resolution | %s | n=%d states per time | times %s"
          % (os.path.basename(args.fm), args.n, times))

    # --- states: snapshot a real unguided trajectory inside the window ---
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    c, f = initial_noise(mask, len(types), gen)
    smp = FlowSampler(net, mask, f_net=None, w=0.0, clip=None)
    grid = smp.time_grid(args.steps)
    snaps, want = {}, sorted(times)
    for i in range(args.steps):
        t0, t1 = float(grid[i]), float(grid[i + 1])
        for tw in want:
            if t0 <= tw < t1 and tw not in snaps:
                snaps[tw] = (c.clone(), f.clone())
        k_c, k_f = smp.field(c, f, t0)
        c, f = c + (t1 - t0) * k_c, f + (t1 - t0) * k_f
    print("snapshotted %d states\n" % len(snaps))

    out = {}
    for prop in props:
        f_A = PhysicalProperty(os.path.join(CKPT, "f_A_%s.pt" % prop), len(types), dev)
        f_B = PhysicalProperty(os.path.join(CKPT, "f_B_%s.pt" % prop), len(types), dev)
        mae_a = float(torch.load(os.path.join(CKPT, "f_A_%s.pt" % prop),
                                 map_location="cpu", weights_only=False)["val_mae"])
        mae_b = float(torch.load(os.path.join(CKPT, "f_B_%s.pt" % prop),
                                 map_location="cpu", weights_only=False)["val_mae"])
        delta = choose_delta(mae_b, args.k_delta)
        target = GS.TARGETS[prop]["q50"]
        s = f_A.y_std
        print("=" * 78)
        print("%s   f_A err %.4f   f_B err %.4f   delta %.4f   target %.4f"
              % (prop, mae_a, mae_b, delta, target))
        print("=" * 78)

        between_arm, between_guide = [], []
        rows = []
        for t in sorted(snaps):
            c0, f0 = snaps[t]
            B = c0.shape[0]
            tt = torch.full((B,), t, device=dev)
            y = torch.full((B,), target, device=dev)
            post = lambda cc, ff: fm_posterior(net, cc, ff, mask, tt)  # noqa: E731

            fields = {}
            for arm in arms:
                extra = GS.arm_kwargs(arm, "-", delta)
                for tag, guide in (("A", f_A), ("B", f_B)):
                    g = torch.Generator(device=dev).manual_seed(args.seed)
                    Gc, Gf, _ = guidance_field(
                        guide, post, c0, f0, mask, y, s, arm,
                        args.n_probe, g, Cost(), t_scalar=t, **extra)
                    fields[(arm, tag)] = flat(Gc, Gf)

            # same guide, different arms
            pa = []
            for i, ai in enumerate(arms):
                for aj in arms[i + 1:]:
                    pa.append(float(cos(fields[(ai, "A")], fields[(aj, "A")]).mean()))
            # same arm, different guide
            pg = [float(cos(fields[(a, "A")], fields[(a, "B")]).mean()) for a in arms]
            between_arm += pa
            between_guide += pg
            rows.append({"t": t, "between_arm_cos": statistics.fmean(pa),
                         "between_guide_cos": statistics.fmean(pg)})
            print("  t=%.2f   between-ARM cos %.4f   between-GUIDE cos %.4f   %s"
                  % (t, statistics.fmean(pa), statistics.fmean(pg),
                     "guide noise dominates" if statistics.fmean(pg)
                     < statistics.fmean(pa) else "arms resolved"))

        ba, bg = statistics.fmean(between_arm), statistics.fmean(between_guide)
        verdict = "FAIL" if bg < ba else "PASS"
        print("\n  OVERALL  between-ARM cos %.4f   between-GUIDE cos %.4f   -> %s"
              % (ba, bg, verdict))
        print("  %s\n" % (
            "Swapping the guide moves the field MORE than swapping the arm: the "
            "arm ranking is reading f_A's error." if verdict == "FAIL" else
            "Arms differ by more than the guide's own noise: the grid resolves them."))

        out[prop] = {"between_arm_cos": ba, "between_guide_cos": bg,
                     "verdict": verdict, "per_time": rows,
                     "f_A_val_mae": mae_a, "f_B_val_mae": mae_b, "delta": delta}

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
