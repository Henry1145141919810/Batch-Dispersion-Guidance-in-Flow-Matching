"""Is V_F the variance BTVG needed to reduce? A direct measurement.

    python proj1/scripts/variance_decomposition.py --prop mu --n 256

BTVG's objective descends V_F = g' Sigma g, the LOCAL predictive variance of
one trajectory's property under the interpolant posterior. The metric it is
meant to improve is the ACROSS-MOLECULE spread of terminal residuals. Those
are different quantities, and the law of total variance names the relation:

    Var_i[f(x_1^i)]  =  E_i[ Var(f | x_t^i) ]  +  Var_i[ E(f | x_t^i) ]
                        \_ within: ~ E_i[V_F] _/   \_ across: ~ Var_i f(m) _/

BTVG acts only on the WITHIN term. For a deterministic sampler the within term
goes to 0 as t -> 1 by construction (k = (1-t)^2/t), so the terminal spread is
entirely the ACROSS term -- the one the objective never touches. If that is
right, then no repair of the estimator, the geometry, the gate or the clip can
make this objective control the terminal spread, which is what the four
completed repairs each found the hard way.

This measures the split at every guided step, on real trajectories:

  within(t)   = mean_i V_F,i(t)              what BTVG minimises
  across(t)   = Var_i[ f(m_i(t)) ]           what it does not touch
  terminal    = Var_i[ f_B(x_1^i) ]          what the metric scores

and three falsifiable predictions, each of which would sink the diagnosis if
it came out the other way:

  P1  within(t) -> 0 while the terminal spread stays O(several delta)
  P2  across(t) already dominates well before the end of the window
  P3  V_F late in sampling does not predict which molecules miss: the
      correlation of V_F,i with |f_B,i - y| is ~0

Run with `--arm btvg --w 4` to check the same on a guided trajectory, so the
claim is not an artefact of looking only at unguided states.

IMPLEMENTATION. An unguided trajectory with full logging is obtained by running
mode `btvg_var` at w = 0: the field is computed (so V_F and f(m) are recorded)
and then multiplied by zero, so the path is bit-identical to unguided. That is
checked against a real unguided run rather than asserted.
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
from m1_signed_bias import PhysicalProperty, load_fm     # noqa: E402
from sampling import FlowSampler, initial_noise, integrate  # noqa: E402


def _load_sweep():
    spec = importlib.util.spec_from_file_location(
        "guidance_sweep", os.path.join(HERE, "guidance_sweep.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GS = _load_sweep()


class Recorder(FlowSampler):
    """FlowSampler that stores per-molecule f(m) and V_F at each guided step."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.log = []

    def _guide(self, coords, feats, t, post_fn, mode=None, t_scalar=None):
        G_c, G_f = super()._guide(coords, feats, t, post_fn, mode,
                                  t_scalar=t_scalar)
        d = self.last_diag
        if "btvg_V_raw" in d:
            self.log.append({"t": float(t_scalar),
                             "f": d["f"].detach().cpu().clone(),
                             "V": d["btvg_V_raw"].detach().cpu().clone()})
        return G_c, G_f


def run(net, mask, types, f_A, target, dev, arm, w_applied, extra, steps, seed,
        record=True, batch=64):
    """Chunked: V_F needs a JVP and an HVP per step, which will not fit for a
    large batch.

    The initial noise is drawn once for the full set and sliced, and zero_com
    reduces per molecule, so that part is exact. That is NOT sufficient on its
    own: `_Base.__init__` seeds `probe_gen` per sampler, i.e. per batch, so any
    arm drawing Hutchinson probes would see batch-dependent trajectories. The
    btvg family is safe because `grad_property_variance` uses an exact
    `torch.func.jvp` along g and the mean term only pulls back -- neither
    touches the generator. Other arms are refused below."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    c0, f0 = initial_noise(mask, len(types), gen)
    if arm and not arm.startswith("btvg"):
        raise SystemExit("chunking is only verified safe for the btvg family: "
                         "_Base seeds probe_gen PER SAMPLER, so any arm that "
                         "draws probes (smg*, tfg_mc, lgd_mc, osc) would get "
                         "different probes per batch. btvg's variance gradient "
                         "uses an exact jvp with no probe.")
    logs, cs, fs = [], [], []
    for i in range(0, mask.shape[0], batch):
        sl = slice(i, min(i + batch, mask.shape[0]))
        m = mask[sl]
        cls = Recorder if record else FlowSampler
        kw = dict(t_min_guide=0.5, f_net=f_A,
                  y=target[sl].clone(),
                  s=f_A.y_std, mode=arm, w=w_applied, clip=1.0, n_probe=1, **extra)
        smp = cls(net, m, **kw) if arm else cls(net, m, t_min_guide=0.5)
        c1, f1, _ = integrate(smp, c0[sl], f0[sl], steps, "euler")
        cs.append(c1)
        fs.append(f1)
        if record:
            logs.append(smp.log)
        del smp
        if dev == "cuda":
            torch.cuda.empty_cache()
    merged = []
    if record and logs:
        for k in range(len(logs[0])):
            merged.append({"t": logs[0][k]["t"],
                           "f": torch.cat([L[k]["f"] for L in logs]),
                           "V": torch.cat([L[k]["V"] for L in logs])})
    return merged, torch.cat(cs), torch.cat(fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default=os.path.join(ROOT, "betty_pull", "fm_last.pt"))
    ap.add_argument("--prop", default="mu")
    ap.add_argument("--arm", default="", help="'' = unguided (logged at w=0)")
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--target", default="q90")
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
    # `dist` draws BOTH sizes and targets from the SAME held-out test
    # molecules, as guidance_sweep does, so the size-property coupling
    # survives. q90 uses val sizes and one fixed target.
    if args.target == "dist":
        sel = d["split"]["test"][: args.n]
        yv = d["y"][sel, d["props"].index(args.prop)].to(dev).float()
        target = yv                       # per-molecule
    else:
        sel = d["split"]["val"][: args.n]
        target = torch.full((args.n,), GS.TARGETS[args.prop][args.target],
                            device=dev)
    mask = d["mask"][sel].to(dev)

    # the logged arm: btvg_var at w=0 (an unguided path) or the named arm
    arm = args.arm or "btvg_var"
    wn = 0.0 if not args.arm else args.w
    extra = GS.arm_kwargs(arm, "-", delta)
    w_applied = wn * GS.strength_scale(arm, extra, f_A.y_std)
    label = args.arm or "unguided (logged at w=0)"
    print("variance decomposition | %s | %s | n=%d seed=%d target %s "
          "(mean %.4f) delta=%.5f" % (args.prop, label, args.n, args.seed,
                                      args.target, float(target.mean()), delta))

    log, c1, f1 = run(net, mask, types, f_A, target, dev, arm, w_applied,
                      extra, args.steps, args.seed, batch=args.batch)
    with torch.no_grad():
        fb = f_B(c1, f1, mask).cpu().double()

    # The w=0 path must equal a real unguided run, or the logging changed what
    # it claims to observe. Compares TERMINAL f_B, not the whole path.
    # `not (gap < tol)` so a NaN gap aborts: NaN is exactly the failure this
    # trick can produce (0.0 * non-finite dV), and `gap >= tol` would let it
    # through.
    gapv = None
    if not args.arm:
        _, c1u, f1u = run(net, mask, types, f_A, target, dev, None, 0.0, {},
                          args.steps, args.seed, record=False, batch=args.batch)
        with torch.no_grad():
            fbu = f_B(c1u, f1u, mask).cpu().double()
        gapv = float((fb - fbu).abs().max())
        ok = gapv < 1e-4
        print("  w=0 terminal f_B vs a real unguided run: max |d| = %.3g %s"
              % (gapv, "OK" if ok else "*** PATHS DIFFER ***"))
        if not ok:
            raise SystemExit("the w=0 trick changed the trajectory; the "
                             "decomposition below would not be unguided")

    ycpu = target.cpu().double()
    resid = fb - ycpu
    natoms = mask.sum(1).cpu()
    term_var = float(resid.var(unbiased=True))
    print("  terminal: bias %+.2f d   sd %.2f d   in-band %.4f" % (
        float(resid.mean()) / delta, term_var ** 0.5 / delta,
        float((resid.abs() <= delta).double().mean())))
    print()
    print("%-6s %12s %12s %12s %10s %10s" % (
        "t", "within/d^2", "across/d^2", "sum/d^2", "within sd/d", "within %"))
    rows = []
    d2 = delta ** 2
    for r in log:
        within = float(r["V"].double().mean())
        across = float((r["f"].double() - ycpu).var(unbiased=True))
        tot = within + across
        rows.append({"t": r["t"], "within": within, "across": across,
                     "within_frac": within / tot if tot > 0 else float("nan")})
        if round(r["t"] * 100) % 5 == 0 or r["t"] >= 0.97:
            print("%-6.2f %12.3g %12.3g %12.3g %10.2f %9.1f%%" % (
                r["t"], within / d2, across / d2, tot / d2,
                (within ** 0.5) / delta, 100 * within / tot if tot > 0 else 0))
    print("\nterminal Var(f_B)/d^2 = %.3g   (sd %.2f d)"
          % (term_var / d2, term_var ** 0.5 / delta))

    # P3: does late V_F predict which molecules miss?
    #
    # SPEARMAN on the RAW SIGNED V, not Pearson on sqrt(V). sqrt() needs V > 0,
    # and V <= 0 on up to ~17% of molecules at some late steps -- dropping them
    # is a selection whose direction cannot be checked after the fact. A rank
    # correlation keeps every molecule and is robust to V spanning orders of
    # magnitude. Pearson-on-sqrt is kept beside it so the two can be compared.
    # 1e-3 tolerance on the window edge: t is float32, so 0.90 is stored as
    # 0.8999999 and `>= 0.9` would silently drop that step.
    def _within_group(x, y, g):
        """Rank correlation pooled WITHIN atom-count groups. On `dist` both V_F
        and |residual| grow with molecule size, so an unconditioned correlation
        can be pure size confound -- the same reason the dist protocol scores
        partial_corr within atom-count groups."""
        num = den_x = den_y = 0.0
        for q in g.unique():
            m = g == q
            if int(m.sum()) < 5:
                continue
            rx = x[m].argsort().argsort().double()
            ry = y[m].argsort().argsort().double()
            rx, ry = rx - rx.mean(), ry - ry.mean()
            num += float((rx * ry).sum())
            den_x += float(rx.pow(2).sum())
            den_y += float(ry.pow(2).sum())
        return num / ((den_x * den_y) ** 0.5 + 1e-30)

    def spearman(x, y):
        rx = x.argsort().argsort().double()
        ry = y.argsort().argsort().double()
        rx, ry = rx - rx.mean(), ry - ry.mean()
        return float((rx * ry).sum() / ((rx.pow(2).sum() * ry.pow(2).sum()).sqrt() + 1e-30))

    late = [r for r in log if r["t"] >= 0.9 - 1e-3]
    cors, permol = [], []
    ae = resid.abs()
    for r in late:
        v = r["V"].double()
        rho = spearman(v, ae)
        pos = torch.isfinite(v) & (v > 0)
        pear = float("nan")
        if int(pos.sum()) > 10:
            a, b = v[pos].sqrt(), ae[pos]
            pear = float(((a - a.mean()) * (b - b.mean())).mean()
                         / (a.std(unbiased=False) * b.std(unbiased=False) + 1e-30))
        # is the dropped set systematically different? (the bias Pearson hides)
        drop = int((~pos).sum())
        dmean = float(ae[~pos].mean()) / delta if drop else float("nan")
        kmean = float(ae[pos].mean()) / delta if int(pos.sum()) else float("nan")
        rho_w = _within_group(v, ae, natoms)
        cors.append({"t": r["t"], "spearman": rho, "spearman_within_size": rho_w,
                     "pearson_sqrt": pear,
                     "n": len(v), "n_dropped": drop,
                     "mean_ae_dropped": dmean, "mean_ae_kept": kmean})
        permol.append({"t": r["t"], "V": v.tolist()})
    if cors:
        print("\nP3  rank corr( V_F at t , |terminal residual| )  [all molecules]")
        print("      %-6s %9s %14s %11s %8s" % (
            "t", "spearman", "within-size", "pearson_sq", "dropped"))
        for c in cors:
            print("      %-6.2f %+9.3f %+14.3f %+11.3f %8d" % (
                c["t"], c["spearman"], c["spearman_within_size"],
                c["pearson_sqrt"], c["n_dropped"]))

    out = os.path.join(ROOT, "results", "variance_decomposition")
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, "%s_%s_%s.json" % (
        args.prop, args.target, args.arm or "unguided"))
    json.dump({"prop": args.prop, "arm": args.arm or "unguided", "w": wn,
               "n": args.n, "seed": args.seed,
               "target_name": args.target,
               "target_mean": float(target.mean()),
               "steps": args.steps, "batch": args.batch,
               "t_min_guide": 0.5, "unguided_gap": gapv,
               "delta": delta, "terminal_var": term_var,
               "terminal_bias": float(resid.mean()),
               "in_band": float((resid.abs() <= delta).double().mean()),
               "rows": rows,
               "late_corr": cors,
               "resid_over_delta": (resid / delta).tolist(),
               "n_atoms": natoms.tolist(),
               "late_V": permol},
              open(fn, "w"), indent=1)
    print("\nwrote", fn)


if __name__ == "__main__":
    main()
