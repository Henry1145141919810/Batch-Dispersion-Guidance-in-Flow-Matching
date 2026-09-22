"""Benchmark the inference-time guidance UPDATE RULE, everything else frozen.

    python proj1/scripts/update_rule_bench.py --arms plug --props mu,alpha,gap

The axis under test is how the guidance direction becomes a step -- euler (the
shipped first-order rule), momentum, adam, adam_eq, muon -- and nothing else.
Same frozen generator, same guide, same sampler, same 100 steps, same seeds,
same targets, same window, same clip, same evaluator as the main sweep. Cells
are written one JSON per cell and skipped if present, so this resumes exactly
like `guidance_sweep.py`.

WHY A STRENGTH GRID RATHER THAN ONE w. In the finished sweep, MAE falls
monotonically in w and molecule stability falls with it: mu/plug goes
1.22 -> 0.78 MAE and 0.416 -> 0.305 stability as w goes 0.01 -> 4. A comparison
at a single w therefore rewards any rule that merely pushes harder, and every
one of these preconditioners changes the effective push. The rules are run over
the same grid and read off the MAE-versus-stability frontier, so "better" has
to mean better at matched chemistry rather than further along the same curve.

The magnitude is also held directly: `rescale="norm"` makes the applied
direction carry the raw direction's per-sample norm, so w, the (1-t)/t
conversion and the velocity-relative clip all keep their meaning. The
`pre_scale` column records what the preconditioner's own magnitude would have
been, which is how much that convention is doing.

Per-step diagnostics are written with every cell: the direction norm, the
cosine between consecutive raw directions, and the cosine between the momentum
buffer and the current direction -- the staleness measurement.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from evaluation import choose_delta, evaluate_samples  # noqa: E402
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from sampling import FlowSampler, VPSampler, initial_noise, integrate  # noqa: E402
from guidance_update import EQUIVARIANT, RULES  # noqa: E402


def _load_sweep():
    """Reuse the sweep's constants and kwargs builders rather than copying them.

    TARGETS, arm_kwargs and strength_scale decide what a cell MEANS. Duplicating
    them here would let this benchmark and the main sweep drift into measuring
    two different things under one name, which is the failure `scale_schedule`
    already caused once.
    """
    spec = importlib.util.spec_from_file_location(
        "guidance_sweep", os.path.join(HERE, "guidance_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GS = _load_sweep()
DATA = os.path.join(ROOT, "data", "qm9.pt")
CKPT = os.path.join(ROOT, "proj1", "checkpoints")
OUT = os.path.join(ROOT, "results", "update_rule")


def cell_name(prop, arm, rule, w):
    return "%s__%s__%s__w%g.json" % (prop, arm, rule, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=os.path.join(ROOT, "betty_pull", "fm.pt"))
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--arms", default="plug",
                    help="guidance arms to run the rules on. The rule is "
                         "orthogonal to the arm, so this is a separate axis.")
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--strengths", default="0.25,1,4")
    ap.add_argument("--target", default="q50")
    ap.add_argument("--t-min-guide", type=float, default=0.5,
                    help="0.5 is the measured optimum of the window effect and "
                         "is what stage v2 screens at")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--solver", default="euler")
    ap.add_argument("--n-probe", type=int, default=1)
    ap.add_argument("--n-mc", type=int, default=4)
    ap.add_argument("--sigma-mc", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--k-delta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--rescale", default="norm", choices=["norm", "none"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0.0)
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a]
    rules = [r for r in args.rules.split(",") if r]
    strengths = [float(x) for x in args.strengths.split(",") if x]
    for r in rules:
        if r not in RULES:
            raise SystemExit("unknown rule %r; known: %s" % (r, ",".join(RULES)))

    cells = [(p, a, r, w) for p in props for a in arms
             for r in rules for w in strengths]
    todo = [c for c in cells
            if not os.path.exists(os.path.join(out, cell_name(*c)))]
    print("cells total %d | done %d | to run %d" % (len(cells),
                                                    len(cells) - len(todo),
                                                    len(todo)))
    if args.dry_run:
        for c in todo:
            print("   ", cell_name(*c))
        return 0
    if not todo:
        print("nothing to do")
        return 0

    d = torch.load(DATA, weights_only=False)
    types = d["types"]
    net, ck = load_fm(args.fm, len(types), dev)
    family = ck.get("family", "flow")
    sel = d["split"]["val"][: args.n]
    mask_v = d["mask"][sel].to(dev)
    print("generator %s  family=%s  device=%s  n=%d steps=%d"
          % (os.path.basename(args.fm), family, dev, args.n, args.steps))

    guides, deltas, mae_bs = {}, {}, {}
    for prop in props:
        ga = os.path.join(CKPT, "f_A_%s.pt" % prop)
        gb = os.path.join(CKPT, "f_B_%s.pt" % prop)
        guides[prop] = (PhysicalProperty(ga, len(types), dev),
                        PhysicalProperty(gb, len(types), dev))
        mae_bs[prop] = float(torch.load(gb, map_location="cpu",
                                        weights_only=False)["val_mae"])
        deltas[prop] = choose_delta(mae_bs[prop], args.k_delta)
        print("  %-6s delta %.5f" % (prop, deltas[prop]))

    def run_cell(prop, arm, rule, w):
        f_A, f_B = guides[prop]
        delta = deltas[prop]
        target = GS.TARGETS[prop][args.target]
        extra = GS.arm_kwargs(arm, "-", delta)
        s = f_A.y_std
        w_scale = GS.strength_scale(arm, extra, s)
        w_applied = w * w_scale
        y_t = torch.full((args.n,), target, device=dev)
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        cs, fs, calls = [], [], 0
        cost = {k: 0 for k in ("gen_fwd", "gen_vjp", "gen_jvp",
                               "guide_fwd", "guide_bwd", "guide_hvp")}
        clipped = 0
        step_rows = []
        t_start = time.time()
        for i in range(0, args.n, args.batch):
            m = mask_v[i:i + args.batch]
            c0, f0 = initial_noise(m, len(types), gen)
            kw = dict(f_net=f_A, y=y_t[: m.shape[0]], s=s, mode=arm,
                      w=w_applied, clip=args.clip, n_probe=args.n_probe,
                      n_mc=args.n_mc, sigma_mc=args.sigma_mc,
                      update_rule=rule,
                      update_kw=dict(beta1=args.beta1, beta2=args.beta2,
                                     rescale=args.rescale),
                      **extra)
            if family == "vp_diffusion":
                smp = VPSampler(net, m, **kw)
            else:
                smp = FlowSampler(net, m, t_min_guide=args.t_min_guide, **kw)
            c, f, nc = integrate(smp, c0, f0, args.steps, args.solver)
            cs.append(c)
            fs.append(f)
            calls += nc
            for k in cost:
                cost[k] += getattr(smp.cost, k)
            clipped += smp.n_clipped
            step_rows.append(smp.updater.rows)
        wall = time.time() - t_start
        C, F = torch.cat(cs), torch.cat(fs)
        r = evaluate_samples(C, F, mask_v, types, f_A, f_B, y_t, delta)
        r.pop("delta", None)

        # Per-step diagnostics, averaged over the batches at the same step
        # index. Batches are the same length, so this is a clean mean; it is
        # kept as a curve because the staleness question is about where in the
        # trajectory the buffer goes wrong, not about a single average.
        traj = []
        if step_rows and step_rows[0]:
            for j in range(len(step_rows[0])):
                agg = {}
                for key in ("t", "g_norm", "cos_prev", "cos_mom_g",
                            "cos_applied_g", "pre_scale"):
                    vals = [b[j][key] for b in step_rows
                            if j < len(b) and b[j].get(key) is not None]
                    if vals:
                        agg[key] = statistics.fmean(vals)
                traj.append(agg)

        def curve_mean(key, lo=0.0, hi=1.0):
            vals = [row[key] for row in traj
                    if key in row and lo <= row.get("t", 0.0) < hi]
            return statistics.fmean(vals) if vals else None

        r.update({
            "prop": prop, "arm": arm, "rule": rule, "w": w,
            "equivariant": EQUIVARIANT[rule],
            "target_name": args.target, "target": target,
            "w_applied": w_applied, "w_scale": w_scale,
            "t_min_guide": args.t_min_guide, "n": args.n,
            "steps": args.steps, "solver": args.solver,
            "n_probe": args.n_probe, "delta": delta, "mae_B": mae_bs[prop],
            "cost": cost, "field_evals": calls, "clipped_sample_steps": clipped,
            "seed": args.seed, "fm": os.path.basename(args.fm),
            "clip": args.clip, "batch": args.batch,
            "beta1": args.beta1, "beta2": args.beta2, "rescale": args.rescale,
            "wall_s": wall, "ms_per_mol": 1000.0 * wall / args.n,
            "traj": traj,
            "g_norm_mean": curve_mean("g_norm"),
            "cos_prev_mean": curve_mean("cos_prev"),
            "cos_prev_early": curve_mean("cos_prev", 0.5, 0.75),
            "cos_prev_late": curve_mean("cos_prev", 0.75, 1.01),
            "cos_mom_g_mean": curve_mean("cos_mom_g"),
            "cos_mom_g_early": curve_mean("cos_mom_g", 0.5, 0.75),
            "cos_mom_g_late": curve_mean("cos_mom_g", 0.75, 1.01),
            "cos_applied_g_mean": curve_mean("cos_applied_g"),
            "pre_scale_mean": curve_mean("pre_scale"),
        })
        return r

    t0 = time.time()
    done, failed = 0, []
    for (prop, arm, rule, w) in todo:
        if args.max_minutes and done:
            used = (time.time() - t0) / 60.0
            if used + used / done > args.max_minutes:
                print("  time guard: stopping with %d cells left"
                      % (len(todo) - done))
                break
        name = cell_name(prop, arm, rule, w)
        cs_t = time.time()
        try:
            r = run_cell(prop, arm, rule, w)
        except Exception as e:
            failed.append(name)
            with open(os.path.join(out, name + ".failed"), "w") as fh:
                json.dump({"cell": name,
                           "error": "%s: %s" % (type(e).__name__, e)}, fh)
            print("  CELL FAILED %s: %s: %s" % (name, type(e).__name__, e))
            continue
        tmp = os.path.join(out, name + ".tmp")
        with open(tmp, "w") as fh:
            json.dump(r, fh)
        os.replace(tmp, os.path.join(out, name))
        done += 1
        print("  [%3d/%3d] %-6s %-6s %-9s w=%-5g mae=%-8.4f band=%-6.3f "
              "molstab=%-6.3f cos_mom=%-6s %.0fs"
              % (done, len(todo), prop, arm, rule, w,
                 r["prop_mae_eval"], r["in_band_fraction"],
                 r["mol_stability"],
                 ("%.3f" % r["cos_mom_g_mean"]) if r["cos_mom_g_mean"]
                 else "-", time.time() - cs_t))

    print("\n%d cells, %.1f min" % (done, (time.time() - t0) / 60.0))
    if failed:
        print("%d FAILED:" % len(failed))
        for f in failed:
            print("   " + f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
