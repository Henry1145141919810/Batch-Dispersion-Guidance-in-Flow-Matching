"""Follow ONE molecule through BTVG and print what each term did, step by step.

    python proj1/scripts/btvg_walkthrough.py --prop mu --w 4

A worked example for explaining the arm, not a benchmark. At every guided step
it evaluates, AT THE SAME STATE the real `btvg` run is at:

  - the mean term alone   (`btvg_mean`, which after strength normalisation is
    exactly `plug`)
  - the variance term alone (`btvg_var`)
  - the KL coefficients and V_F / tau^2, so the "band-targeting" switch is
    visible
  - the base flow velocity, the applied correction, and whether the velocity
    clip bound -- i.e. whether the two terms were competing for one budget

The extra evaluations are counterfactual read-outs at the current state; the
trajectory itself is the ordinary `btvg` run and is unchanged.

Writes results/btvg_walkthrough/<prop>_w<w>.json alongside the printout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from evaluation import choose_delta                      # noqa: E402
from guidance import guidance_field                      # noqa: E402
from m1_signed_bias import PhysicalProperty, load_fm     # noqa: E402
from sampling import FlowSampler, initial_noise, integrate, fm_posterior  # noqa: E402


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "guidance_sweep", os.path.join(HERE, "guidance_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GS = _load_sweep()


def _n(a, b):
    return float(torch.sqrt((a ** 2).sum() + (b ** 2).sum()))


class Tracer(FlowSampler):
    """FlowSampler that records the two BTVG halves without changing the path."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.rows = []

    def _guide(self, coords, feats, t, post_fn, mode=None, t_scalar=None):
        # the real step (unchanged; this is what the trajectory follows)
        G_c, G_f = super()._guide(coords, feats, t, post_fn, mode,
                                  t_scalar=t_scalar)
        diag = self.last_diag
        # the same state, each half on its own -- pure read-out
        halves = {}
        for half in ("btvg_mean", "btvg_var"):
            H_c, H_f, _ = guidance_field(
                self.f_net, post_fn, coords, feats, self.mask, self.y, self.s,
                half, self.n_probe, None, self.cost, self.n_mc, self.sigma_mc,
                t_scalar=t, tau=self.tau)
            halves[half] = _n(H_c, H_f)
        # base velocity at this state, to see the clip
        B = coords.shape[0]
        with torch.no_grad():
            v_c, v_f = self.net(coords, feats, self.mask,
                                torch.full((B,), float(t_scalar),
                                           device=coords.device))
        mult = float(self.w) * (1.0 - float(t_scalar)) / float(t_scalar)
        raw = mult * _n(G_c, G_f)
        vn = _n(v_c, v_f)
        tau2 = float(self.tau) ** 2
        V = float(diag["btvg_V_raw"][0])
        f = float(diag["f"][0])
        y = float(self.y[0])
        self.rows.append({
            "t": round(float(t_scalar), 4),
            "f": f, "resid": y - f,
            "V": V, "V_over_tau2": V / tau2,
            "b_coeff": -0.5 * (1.0 / tau2 - 1.0 / V) if V > 0 else None,
            "b_clamped_to_zero": (V <= 0) or (V < tau2),
            "mean_norm": halves["btvg_mean"], "var_norm": halves["btvg_var"],
            "var_share": halves["btvg_var"] / max(
                halves["btvg_mean"] + halves["btvg_var"], 1e-30),
            "raw_over_v": raw / max(vn, 1e-30),
            "clipped": raw > self.clip * vn,
        })
        return G_c, G_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=os.path.join(ROOT, "betty_pull", "fm_last.pt"))
    ap.add_argument("--prop", default="mu")
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--target", default="q90")
    ap.add_argument("--mol", type=int, default=0, help="index into val")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--compare", action="store_true",
                    help="also run the same molecule and noise unguided and "
                         "with the mean term alone (= plug)")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    net, _ = load_fm(args.fm, len(types), dev)
    ck = os.path.join(ROOT, "proj1", "checkpoints")
    f_A = PhysicalProperty(os.path.join(ck, "f_A_%s.pt" % args.prop), len(types), dev)
    f_B = PhysicalProperty(os.path.join(ck, "f_B_%s.pt" % args.prop), len(types), dev)
    mae_b = float(torch.load(os.path.join(ck, "f_B_%s.pt" % args.prop),
                             map_location="cpu", weights_only=False)["val_mae"])
    delta = choose_delta(mae_b, 2.0)
    target = GS.TARGETS[args.prop][args.target]

    sel = d["split"]["val"][args.mol: args.mol + 1]
    mask = d["mask"][sel].to(dev)
    extra = GS.arm_kwargs("btvg", "-", delta)
    w_applied = args.w * GS.strength_scale("btvg", extra, f_A.y_std)
    tau = extra.get("tau")

    print("BTVG walkthrough | %s | one molecule (%d atoms) | target %s = %.4f"
          % (args.prop, int(mask.sum()), args.target, target))
    print("delta = %.5f   tau = %.5f (= delta/1.96)   tau^2 = %.3e"
          % (delta, tau, tau ** 2))
    print("w nominal %g -> applied %.5g   clip 1.0   guidance from t = 0.5\n"
          % (args.w, w_applied))

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    c0, f0 = initial_noise(mask, len(types), gen)
    smp = Tracer(net, mask, t_min_guide=0.5, f_net=f_A,
                 y=torch.full((1,), target, device=dev), s=f_A.y_std,
                 mode="btvg", w=w_applied, clip=1.0, n_probe=1, **extra)
    c1, f1, _ = integrate(smp, c0, f0, args.steps, "euler")

    print("%-6s %9s %9s %11s %9s %9s %8s %8s %s" % (
        "t", "f(m)", "y-f(m)", "V/tau^2", "|mean|", "|var|", "var sh", "|G|/|v|", "clip"))
    for r in smp.rows:
        if round(r["t"] * 100) % 5 and r["t"] < 0.95:
            continue
        print("%-6.2f %9.3f %9.3f %11.1f %9.3g %9.3g %8.3f %8.3f %s" % (
            r["t"], r["f"], r["resid"], r["V_over_tau2"], r["mean_norm"],
            r["var_norm"], r["var_share"], r["raw_over_v"],
            "YES" if r["clipped"] else ""))

    n = len(smp.rows)
    clip = sum(r["clipped"] for r in smp.rows)
    stop = sum(r["b_clamped_to_zero"] for r in smp.rows)
    neg = sum(r["V"] <= 0 for r in smp.rows)
    share = sum(r["var_share"] for r in smp.rows) / n
    with torch.no_grad():
        got = float(f_B(c1, f1, mask)[0])
    print("\n%d guided steps | clip bound %d (%.0f%%) | V <= tau^2 (term off) "
          "%d (%.0f%%) | V <= 0 %d" % (n, clip, 100 * clip / n, stop,
                                       100 * stop / n, neg))
    print("mean variance-term share of the raw field: %.3f" % share)
    print("final f_B = %.4f   target %.4f   |error| = %.4f = %.1f delta   %s"
          % (got, target, abs(got - target), abs(got - target) / delta,
             "IN BAND" if abs(got - target) <= delta else "outside the band"))

    cmp_out = {}
    if args.compare:
        print("")
        print("Same molecule, same initial noise, same applied strength:")
        print("%-26s %9s %9s %9s" % ("arm", "f_B", "err/delta", "in band"))
        for label, arm in (("unguided", None), ("mean term only (= plug)",
                                                "btvg_mean"), ("btvg (both terms)", "btvg")):
            g2 = torch.Generator(device=dev).manual_seed(args.seed)
            a0, b0 = initial_noise(mask, len(types), g2)
            kw = dict(f_net=f_A, y=torch.full((1,), target, device=dev),
                      s=f_A.y_std, w=w_applied, clip=1.0, n_probe=1, **extra)
            if arm is None:
                s2 = FlowSampler(net, mask, t_min_guide=0.5)
            else:
                s2 = FlowSampler(net, mask, t_min_guide=0.5, mode=arm, **kw)
            x1, y1, _ = integrate(s2, a0, b0, args.steps, "euler")
            with torch.no_grad():
                v = float(f_B(x1, y1, mask)[0])
            e = abs(v - target) / delta
            print("%-26s %9.4f %9.1f %9s" % (label, v, e, "yes" if e <= 1 else "no"))
            cmp_out[label] = {"f_B": v, "err_over_delta": e}

    out = os.path.join(ROOT, "results", "btvg_walkthrough")
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, "%s_w%g.json" % (args.prop, args.w))
    json.dump({"prop": args.prop, "target": target, "delta": delta, "tau": tau,
               "w_nominal": args.w, "w_applied": w_applied, "mol": args.mol,
               "seed": args.seed, "final_f_B": got,
               "compare": cmp_out, "rows": smp.rows},
              open(fn, "w"), indent=1)
    print("wrote", fn)


if __name__ == "__main__":
    main()
