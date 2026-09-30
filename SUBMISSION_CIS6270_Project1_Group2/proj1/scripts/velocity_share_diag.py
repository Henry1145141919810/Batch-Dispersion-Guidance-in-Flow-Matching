"""How much of the sampling velocity is guidance, and does rotating it survive?

    python proj1/scripts/velocity_share_diag.py

The update-rule benchmark found that Adam and Muon point the guidance direction
~68 degrees away from the raw one with no measurable change in outcome. The
hypothesis under test here is the deflationary one: guidance is a small
correction on top of the base flow-matching velocity, so a large rotation of a
small vector barely moves the total step, and the trajectory never notices.

Two numbers per guided step, both measured inside the real sampler
(`log_velocity=True` is pure logging; it changes no arithmetic on the path):

    r_t      = |C(D_t)| / |V_t|                        the correction's share
    angle    = angle( V_t + C(G_t),  V_t + C(D_t) )    what the step actually does

`C(.)` is the existing downstream conversion, strength and clip, applied
identically to both directions, so the ONLY difference between the two total
velocities is the update rule. The raw counterfactual C(G_t) is evaluated at
the same state the transformed run is actually at, which makes this an
instantaneous comparison rather than a comparison of two diverged trajectories.

Deliberately cheap: one property, one strength, Adam against Euler, n=128.
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
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from evaluation import choose_delta  # noqa: E402
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from sampling import FlowSampler, initial_noise, integrate  # noqa: E402


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "guidance_sweep", os.path.join(HERE, "guidance_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GS = _load_sweep()


def q(vals):
    v = sorted(vals)
    return (statistics.fmean(v), v[len(v) // 2], v[0], v[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=os.path.join(ROOT, "betty_pull", "fm_last.pt"))
    ap.add_argument("--prop", default="mu")
    ap.add_argument("--arm", default="plug")
    ap.add_argument("--rules", default="adam,muon,momentum")
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--t-min-guide", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "velocity_share.json"))
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    net, _ = load_fm(args.fm, len(types), dev)
    ckpt = os.path.join(ROOT, "proj1", "checkpoints")
    f_A = PhysicalProperty(os.path.join(ckpt, "f_A_%s.pt" % args.prop),
                           len(types), dev)
    mae_b = float(torch.load(os.path.join(ckpt, "f_B_%s.pt" % args.prop),
                             map_location="cpu", weights_only=False)["val_mae"])
    delta = choose_delta(mae_b, 2.0)
    target = GS.TARGETS[args.prop]["q50"]
    sel = d["split"]["val"][: args.n]
    mask = d["mask"][sel].to(dev)
    extra = GS.arm_kwargs(args.arm, "-", delta)
    w_applied = args.w * GS.strength_scale(args.arm, extra, f_A.y_std)

    print("velocity-share diagnostic | %s | %s arm=%s w=%g (applied %.4g) "
          "t_min=%g n=%d steps=%d"
          % (os.path.basename(args.fm), args.prop, args.arm, args.w,
             w_applied, args.t_min_guide, args.n, args.steps))
    print("r_t = |C(D)| / |V| ;  angle = between the two TOTAL velocities\n")

    out = {}
    for rule in [r for r in args.rules.split(",") if r]:
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        c0, f0 = initial_noise(mask, len(types), gen)
        smp = FlowSampler(net, mask, t_min_guide=args.t_min_guide,
                          f_net=f_A,
                          y=torch.full((args.n,), target, device=dev),
                          s=f_A.y_std, mode=args.arm, w=w_applied,
                          clip=args.clip, n_probe=1,
                          update_rule=rule, log_velocity=True, **extra)
        integrate(smp, c0, f0, args.steps, "euler")
        rows = smp.vel_rows
        rt = [r["r_t"] for r in rows]
        ang = [r["total_vel_angle_deg"] for r in rows]
        gang = [r["guidance_angle_deg"] for r in rows]
        m_rt, md_rt, lo_rt, hi_rt = q(rt)
        m_a, md_a, _, hi_a = q(ang)
        m_g, md_g, _, _ = q(gang)
        print("== %s   (%d guided steps)" % (rule, len(rows)))
        print("   r_t            mean %.4f  median %.4f  min %.4f  max %.4f"
              % (m_rt, md_rt, lo_rt, hi_rt))
        print("   total-vel angle mean %.2f deg  median %.2f  max %.2f"
              % (m_a, md_a, hi_a))
        print("   guidance angle  mean %.2f deg  median %.2f   <- for reference"
              % (m_g, md_g))
        # over sampling time, in thirds of the guided window
        k = max(len(rows) // 3, 1)
        for name, seg in (("early", rows[:k]), ("mid", rows[k:2 * k]),
                          ("late", rows[2 * k:])):
            if not seg:
                continue
            print("     %-5s t=%.2f-%.2f  r_t %.4f   total-vel %.2f deg   "
                  "guidance %.2f deg"
                  % (name, seg[0]["t"], seg[-1]["t"],
                     statistics.fmean([r["r_t"] for r in seg]),
                     statistics.fmean([r["total_vel_angle_deg"] for r in seg]),
                     statistics.fmean([r["guidance_angle_deg"] for r in seg])))
        print()
        out[rule] = {"r_t_mean": m_rt, "r_t_median": md_rt,
                     "r_t_min": lo_rt, "r_t_max": hi_rt,
                     "total_vel_angle_mean": m_a, "total_vel_angle_median": md_a,
                     "total_vel_angle_max": hi_a,
                     "guidance_angle_mean": m_g, "guidance_angle_median": md_g,
                     "n_guided_steps": len(rows), "rows": rows}

    out["_config"] = {"prop": args.prop, "arm": args.arm, "w": args.w,
                      "w_applied": w_applied, "t_min_guide": args.t_min_guide,
                      "n": args.n, "steps": args.steps, "clip": args.clip,
                      "seed": args.seed, "fm": os.path.basename(args.fm)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
