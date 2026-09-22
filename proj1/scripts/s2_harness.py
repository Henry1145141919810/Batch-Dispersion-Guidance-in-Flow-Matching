"""S2: run the evaluation harness end to end and fix delta.

Three arms, so the numbers have a reference on both sides:

  data      real validation molecules through the same metrics. This is the
            ceiling: whatever stability/validity the bond heuristic assigns to
            REAL QM9 is the most any generator can be expected to reach.
  unguided  samples from the base model. The floor for property targeting and
            the reference for structure.
  plug-in   the baseline guidance arm at one fixed target, so the harness is
            exercised on a guided trajectory before M-1/M-2 rely on it.

delta is set from f_B's validation MAE (k = 2 by default) and written to the
results file; every later arm reads the same delta from there.

Usage (Betty, inside a job):
  python proj1/scripts/s2_harness.py --fm proj1/checkpoints/fm.pt \
      --guide proj1/checkpoints/f_A_mu.pt --eval proj1/checkpoints/f_B_mu.pt \
      --n 256 --steps 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, HERE)
from evaluation import choose_delta, evaluate_samples  # noqa: E402
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from sampling import FlowSampler, VPSampler, initial_noise, integrate  # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True, help="generator checkpoint (fm.pt or diff.pt)")
    ap.add_argument("--guide", required=True, help="f_A checkpoint")
    ap.add_argument("--eval", required=True, help="f_B checkpoint")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--solver", default="euler", choices=["euler", "heun"])
    ap.add_argument("--target", type=float, default=None,
                    help="property target in physical units; default = data mean + 1 std")
    ap.add_argument("--s", type=float, default=None,
                    help="Gaussian target width; default = property std. NOT delta: "
                         "1/s^2 multiplies the guidance, and delta (~0.17 D) would "
                         "give x35 on top of the (1-t)/t factor and overflow")
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0,
                    help="cap the guidance correction at this multiple of the base "
                         "velocity norm, per sample. Pass a negative value to disable "
                         "(the ablation that shows why it is needed).")
    ap.add_argument("--mode", default="plug",
                    choices=["plug", "smg", "smg_mean", "smg_var", "smg2",
                             "smg2_curv", "tfg_mc", "lgd_mc", "osc", "rch",
                             "band"],
                    help="see GUIDANCE_EXPERIMENT_PLAN.md section 1 for what "
                         "each arm is and whether it is OWN or COMPARE. "
                         "smg_var is TMPD/PiGDM's published scalar form; "
                         "rch needs --rch <fitted head>; band needs --band-tau")
    ap.add_argument("--rch", default=None, help="fitted rch_<prop>.pt for --mode rch")
    ap.add_argument("--band-tau", type=float, default=None,
                    help="tolerance for --mode band; defaults to delta = 2 x f_B MAE")
    ap.add_argument("--want-kappa3", action="store_true",
                    help="log the observable's standardised skew alongside any arm")
    ap.add_argument("--n-mc", type=int, default=4, help="TFG eps_bsz")
    ap.add_argument("--sigma-mc", type=float, default=0.1,
                    help="TFG perturbation std; their schedule tunes this")
    ap.add_argument("--n-probe", type=int, default=4,
                    help="Hutchinson probes for tr(H Sigma); only used by SMG modes")
    ap.add_argument("--k-delta", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=os.path.join(ROOT, "proj1", "results", "s2_harness.json"))
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    d = torch.load(DATA, weights_only=False)
    types = d["types"]
    net, ck = load_fm(args.fm, len(types), dev)
    family = ck.get("family", "flow")
    f_A = PhysicalProperty(args.guide, len(types), dev)
    f_B = PhysicalProperty(args.eval, len(types), dev)
    mae_B = float(torch.load(args.eval, map_location="cpu", weights_only=False)["val_mae"])
    delta = choose_delta(mae_B, args.k_delta)
    y_mean, y_std = f_A.y_mean, f_A.y_std
    target = args.target if args.target is not None else y_mean + y_std
    s = args.s if args.s is not None else y_std

    print("generator : %s  family=%s  val_loss=%.4f" % (args.fm, family, ck.get("val_loss", float("nan"))))
    print("guide f_A : %s   evaluator f_B: %s (val MAE %.4f)" % (args.guide, args.eval, mae_B))
    print("delta     : %.4f  (= %.1f x f_B MAE)   target y*=%.3f   s=%.3f   w=%.2f"
          % (delta, args.k_delta, target, s, args.w))
    print("sampling  : n=%d  steps=%d  solver=%s  device=%s" % (args.n, args.steps, args.solver, dev))

    va = d["split"]["val"][: args.n]
    coords_v, feats_v, mask_v = d["coords"][va].to(dev), d["feats"][va].to(dev), d["mask"][va].to(dev)
    y_t = torch.full((args.n,), target, device=dev)

    clip = None if args.clip < 0 else args.clip

    def make_sampler(mask, guided):
        kw = dict(f_net=f_A, y=y_t[: mask.shape[0]], s=s, mode=args.mode, w=args.w,
                  clip=clip, n_probe=args.n_probe,
                  n_mc=args.n_mc, sigma_mc=args.sigma_mc) if guided else {}
        if family == "vp_diffusion":
            return VPSampler(net, mask, **kw)
        return FlowSampler(net, mask, **kw)

    def sample(guided):
        cs, fs = [], []
        calls = 0
        cost = {"gen_fwd": 0, "gen_vjp": 0, "gen_jvp": 0,
                "guide_fwd": 0, "guide_bwd": 0, "guide_hvp": 0}
        clipped = 0
        for i in range(0, args.n, args.batch):
            m = mask_v[i:i + args.batch]
            c0, f0 = initial_noise(m, len(types), gen)
            smp = make_sampler(m, guided)
            c, f, n_calls = integrate(smp, c0, f0, args.steps, args.solver)
            cs.append(c)
            fs.append(f)
            calls += n_calls
            for k in cost:
                cost[k] += getattr(smp.cost, k)
            clipped += smp.n_clipped
        cost["field_evals"] = calls
        cost["clipped_sample_steps"] = clipped
        return torch.cat(cs), torch.cat(fs), cost

    results = {"delta": delta, "k_delta": args.k_delta, "mae_B": mae_B,
               "target": target, "s": s, "w": args.w, "steps": args.steps,
               "solver": args.solver, "family": family, "arms": {}}
    t0 = time.time()

    # ---- arm 1: real data
    r = evaluate_samples(coords_v, feats_v, mask_v, types, f_A, f_B,
                         torch.full((args.n,), y_mean, device=dev), delta)
    r["note"] = "real validation molecules; target set to the data mean so in-band is the natural rate"
    results["arms"]["data"] = r
    print("\n[data]     atom_stab %.3f  mol_stab %.3f  valid %.3f  uniq %.3f  |  f_B MAE vs mean %.3f  (%.0fs)"
          % (r["atom_stability"], r["mol_stability"], r["validity"], r["uniqueness_of_valid"],
             r["prop_mae_eval"], time.time() - t0))

    # ---- arm 2: unguided samples
    c, f, cost = sample(guided=False)
    r = evaluate_samples(c, f, mask_v, types, f_A, f_B, y_t, delta)
    r["cost"] = cost
    results["arms"]["unguided"] = r
    print("[unguided] atom_stab %.3f  mol_stab %.3f  valid %.3f  uniq %.3f  |  f_B MAE %.3f  in-band %.3f  gap %.3f  (%.0fs)"
          % (r["atom_stability"], r["mol_stability"], r["validity"], r["uniqueness_of_valid"],
             r["prop_mae_eval"], r["in_band_fraction"], r["guide_eval_gap_mean"], time.time() - t0))

    # ---- arm 3: plug-in guidance
    c, f, cost = sample(guided=True)
    r = evaluate_samples(c, f, mask_v, types, f_A, f_B, y_t, delta)
    r["cost"] = cost
    results["arms"]["plugin"] = r
    print("[plug-in]  atom_stab %.3f  mol_stab %.3f  valid %.3f  uniq %.3f  |  f_B MAE %.3f  in-band %.3f  gap %.3f  (%.0fs)"
          % (r["atom_stability"], r["mol_stability"], r["validity"], r["uniqueness_of_valid"],
             r["prop_mae_eval"], r["in_band_fraction"], r["guide_eval_gap_mean"], time.time() - t0))

    # ---- readout
    dat, ung, plg = results["arms"]["data"], results["arms"]["unguided"], results["arms"]["plugin"]
    print("\nharness readout")
    print("  bond-heuristic ceiling on REAL data: mol_stab %.3f  valid %.3f  -- generator numbers are read against this"
          % (dat["mol_stability"], dat["validity"]))
    print("  unguided -> plug-in : f_B MAE %.3f -> %.3f   in-band %.3f -> %.3f   mol_stab %.3f -> %.3f"
          % (ung["prop_mae_eval"], plg["prop_mae_eval"], ung["in_band_fraction"],
             plg["in_band_fraction"], ung["mol_stability"], plg["mol_stability"]))
    print("  reward-hacking gap |f_A - f_B|: unguided %.3f   plug-in %.3f   (delta = %.3f)"
          % (ung["guide_eval_gap_mean"], plg["guide_eval_gap_mean"], delta))
    print("  delta = %.4f fixed for all later arms" % delta)
    pc = plg["cost"]
    print("  measured guided cost: gen fwd %d, VJP %d, JVP %d | guide fwd %d, bwd %d, HVP %d"
          % (pc["gen_fwd"], pc["gen_vjp"], pc["gen_jvp"],
             pc["guide_fwd"], pc["guide_bwd"], pc["guide_hvp"]))
    print("  (field evals alone = %d, which under-counts the real work)" % pc["field_evals"])
    print("  guidance clip: %s   steps clipped: %d   non-finite samples: %d"
          % ("off" if clip is None else "%.2f x |v|" % clip,
             pc["clipped_sample_steps"], plg.get("n_nonfinite", 0)))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("wrote " + args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
