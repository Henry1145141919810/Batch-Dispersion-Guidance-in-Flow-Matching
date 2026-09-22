"""Run our guidance arms on a BORROWED base model and BORROWED property pair.

THE QUESTION THIS ANSWERS. `guidance_sweep.py` measures every arm on our
flow-matching generator, steered by our `f_A` and scored by our `f_B`. The
split protocol makes f_A and f_B disjoint, so an arm cannot win by exploiting
the scorer -- but generator, guide and evaluator still come from one pipeline,
built by one group, on one preprocessing of QM9. A reviewer is entitled to ask
whether an arm's advantage is a property of the METHOD or a property of our
STACK. Nothing in our own sweep can answer that.

So: take TFG's released `EDMsecond` generator, TFG's released `tf_predict_<p>`
guide and TFG's released `evaluate_<p>` oracle, for the three properties we
already study, and re-run the arms unchanged. Every component except the
guidance field is then external. If our arms keep their ranking here, the
ranking is about the arms. If it inverts, we have learned something more
important than a win.

WHAT IS HELD FIXED ACROSS ARMS, so that differences are attributable:
  base model      EDMsecond (one checkpoint, one md5, stamped per cell)
  guide  f_A      tf_predict_<p>, evaluated at time channel 0
  oracle f_B      evaluate_<p>, the network TFG reports its own MAE with
  sampler         100-step Euler on the probability-flow ODE under EDM's
                  polynomial_2 schedule
  targets         q50 / q90 of the QM9 property distribution
  delta           k x the oracle's measured MAE, k = 2 -- the same
                  pre-registered rule, via the same `choose_delta` helper
  window          guidance off while tau > tau_max_guide (default 0.5)
  clip            velocity-relative trust region, clip = 1.0

WHAT DIFFERS FROM THE MAIN SWEEP, and must be disclosed with every number:

  1. GUIDE/ORACLE DISJOINTNESS IS INFERRED, NOT CONSTRUCTED. Both TFG arg
     files record `dataset: qm9_second_half`, and their training script
     mutates `args.dataset` after building the training loader, so the string
     is unreliable. `audit/fa_fb_search/disjointness_test.py` finds the error
     structure of a disjoint pair and says "DISJOINT CONSISTENT", which is
     strong but circumstantial. Our own pair is disjoint by construction. This
     is the transfer's main weakness and it cannot be fixed from the released
     artifacts.
  2. DELTA IS OPTIMISTIC. It is 2 x the oracle's MAE measured on QM9
     molecules of which an unknown ~50% are in the oracle's own training half,
     so the measured MAE is lower than a clean held-out MAE and delta is
     TIGHTER than it should be. A tighter band lowers in-band coverage for
     every arm equally, so it does not favour any arm -- it makes the absolute
     coverage numbers pessimistic, and they should be read that way.
  3. THE SAMPLER IS OURS, NOT EDM'S. EDM samples a 1000-step ancestral SDE and
     TFG samples 100 DDIM steps at eta = 1. We integrate the probability-flow
     ODE at 100 Euler steps, because that is what our guidance fields are
     defined on and because holding the sampler fixed across arms is what
     makes the comparison internal. Consequence: the UNGUIDED row here is not
     EDM's published unconditional row, and must not be quoted as one.
  4. THE GUIDE IS ASKED AT t = 0, AND ONLY AT t = 0. TFG's own method
     evaluates its time-dependent guide at the noisy state x_tau with that
     tau. Our arms evaluate at the posterior mean m, an estimate of a clean
     molecule, so they ask for t = 0. The alternative is NOT measured here and
     there is no flag for it: doing it honestly needs a calibration refitted
     per t, since `calibrate` below fits once at t = 0. An earlier version of
     this file carried a `--guide-time current` switch that was wired to
     nothing and would have reported "the choice does not matter".
  5. THE GUIDANCE WINDOW IS SET, NOT INHERITED. `--tau-max-guide` defaults to
     0.5, the VP image of the main sweep's `t_min_guide = 0.5` (flow time runs
     0 -> 1 toward data, VP time 1 -> 0, so the windows mirror). This is the
     largest single effect measured anywhere in the project -- alpha MAE
     10.02 -> 5.32 as `t_min_guide` goes 0.05 -> 0.5 -- so leaving `VPSampler`
     at its historical "guide everywhere" default would have run every arm at
     the bad end of that curve, and any ranking inversion would have been
     unattributable. It also matters mechanically: `vp_posterior` divides by
     alpha, and alpha(1.0) = 0.0032 under EDM's schedule, so guiding near
     tau = 1 amplifies epsilon error by ~300x. Run `--tau-max-guide 1.0` as a
     labelled comparison if the window's effect on this backend is wanted.

Resumable: one JSON per cell, completed cells are skipped, so a partial run is
still a complete experiment over the cells it finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

from evaluation import choose_delta, evaluate_samples        # noqa: E402
from external.tfg_assets import (Calibrated, EDMGenerator, PROP_INDEX,  # noqa: E402
                                 PROP_UNITS, QM9_MAD, TFGGuide, TFGOracle,
                                 fit_calibration)
from sampling import VPSampler, initial_noise, integrate     # noqa: E402

DATA = os.path.join(ROOT, "data", "qm9.pt")
TFG_ROOT = os.path.join(ROOT, "audit", "fa_fb_search", "TFG")
OUT = os.path.join(ROOT, "results", "transfer")

# --------------------------------------------------------------------------
# the arms under test
# --------------------------------------------------------------------------
# Deliberately a SHORT list. The main sweep's job is to screen 20+ arms; this
# one's job is to re-test a verdict, so it runs the published comparators that
# a reviewer will recognise plus the arms we claim as ours, and nothing else.
# Cost per cell here is ~3x a main-sweep cell (EDM is 192x9 with attention and
# a dense edge list), so breadth is bought with wall-clock we do not have.

COMPARE_ARMS = [
    "plug",        # DPS-style plug-in (Chung et al. 2023)
    "tmpd",        # TMPD / PiGDM uncertainty denominator (Boys et al. 2024)
    "tfg_mc",      # TFG's own MC smoothing (Ye et al. 2024) -- ITS HOME TURF
    "lgd_mc",      # loss-guided diffusion (Song et al. 2023)
    "osc",         # observable-space closure
]
OUR_ARMS = [
    "smg",         # SMG as shipped
    "smg2",        # completed-moment closure
    "spbc",        # shape-preserving bias correction
    "btvg",        # band-targeted variance guidance
]
BASELINE = ["unguided"]

# Targets: quantiles of the QM9 property distribution, computed from the data
# rather than hard-coded, because this experiment does not use our train_a
# split and quoting train_a's quantiles here would be quietly wrong.
TARGET_QUANTILES = {"q50": 0.50, "q90": 0.90}

STRENGTHS = [0.05, 0.25, 1.0, 4.0]


def file_md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def cell_name(prop, arm, tgt, w, cfg=""):
    """One filename per cell, carrying the configuration that produced it.

    Resumability skips any cell whose file exists, so a name that omits part of
    the configuration silently merges incompatible runs: an `--n 64` smoke test
    followed by an `--n 512` sweep would leave the small cells in place and the
    result set would mix sample sizes without saying so. `cfg` is built once in
    `main` from every argument that changes a cell's meaning.
    """
    return "tr__%s__%s__%s__w%g%s.json" % (prop, arm, tgt, w, cfg)


def config_tag(args):
    """The run-configuration suffix baked into every cell name."""
    return "__n%d_s%d_%s_win%g_seed%d" % (
        args.n, args.steps, args.solver, args.tau_max_guide, args.seed)


def arm_kwargs(arm, delta):
    """Per-arm extra state, mirroring guidance_sweep.arm_kwargs.

    BTVG's tau is the target sd that makes the band a 95% interval, not delta
    itself: a centred Gaussian with sd = delta covers only 68.3% of the band.
    """
    if arm in ("btvg", "btvg_var", "btvg_mean"):
        return {"tau": delta / 1.96}
    if arm == "band":
        return {"band_tau": delta}
    return {}


def strength_scale(arm, kw, s):
    """Put every arm's strength knob on one scale, as the main sweep does.

    BTVG's coefficient carries 1/tau^2 where the likelihood arms carry 1/s^2,
    so w = 1 means something different for it unless the ratio is divided out.
    Without this, a BTVG cell at w = 1 ran thousands of times stronger than a
    plug cell at w = 1 and was clip-saturated at every strength -- which is
    finding S1 in SCOPE_FM_GUIDANCE_STATUS.md section 10b.
    """
    if arm.startswith("btvg") and kw.get("tau"):
        return (kw["tau"] / s) ** 2
    return 1.0


def load_backend(args, dev):
    """Generator, guide and oracle, all external, all calibrated together."""
    gen_w = os.path.join(args.edm_dir, "generative_model_ema.npy")
    gen_a = os.path.join(args.edm_dir, "args.pickle")
    for p in (gen_w, gen_a):
        if not os.path.exists(p):
            raise SystemExit(
                "missing %s.\nRun:  python proj1/scripts/fetch_tfg_assets.py\n"
                "The EDMsecond checkpoint is not in the repository; it is "
                "TFG's release and must be downloaded." % p)
    net = EDMGenerator(gen_w, gen_a, n_types=5, device=dev)
    return net, {"edm_md5": file_md5(gen_w), "edm_args": gen_a}


def calibrate(prop, d, sel, dev, k_delta):
    """Build (f_A, f_B) in physical units, plus delta and the fit report.

    Guide and oracle are fitted on the SAME molecules with the SAME two-
    parameter least squares, so nothing about the comparison between them
    depends on the calibration.
    """
    idx = PROP_INDEX[prop]
    c = d["coords"][sel].to(dev)
    f = d["feats"][sel].to(dev)
    m = d["mask"][sel].to(dev)
    truth = d["y"][sel, idx].to(dev)

    raw_guide = TFGGuide(TFG_ROOT, prop, device=dev)
    raw_oracle = TFGOracle(TFG_ROOT, prop, device=dev)

    # The guide wants onehot / norm_values[1] and the oracle wants raw onehot.
    # Fit each in the space it wants -- `f` here is raw one-hot from the data
    # file -- then record which space in the wrapper so the sampler cannot mix
    # them up later. The factor is read off the guide's own args rather than
    # written as 8, so a checkpoint with a different recipe cannot be fitted
    # under the wrong one.
    fscale = raw_guide.norm_values[1]
    ag, bg, mae_g = fit_calibration(
        lambda C, F, M: raw_guide(C, F / fscale, M), c, f, m, truth)
    ao, bo, mae_o = fit_calibration(raw_oracle, c, f, m, truth)

    y_std = float(truth.double().std())
    f_A = Calibrated(raw_guide, ag, bg, prop, y_std,
                     feats_are_normalised=True)
    f_B = Calibrated(raw_oracle, ao, bo, prop, y_std,
                     feats_are_normalised=False, feat_scale=fscale)
    # The project's pre-registered rule, via the same helper the main sweep
    # uses -- NOT a hardcoded 2.0. `k_delta` was previously accepted, recorded
    # in every cell, and ignored, so `--k-delta 3` produced cells whose stated
    # k did not describe their delta.
    delta = choose_delta(mae_o, k_delta)
    report = {"guide_slope": ag, "guide_intercept": bg, "guide_mae": mae_g,
              "oracle_slope": ao, "oracle_intercept": bo, "oracle_mae": mae_o,
              "qm9_mad": QM9_MAD[prop], "y_std": y_std,
              "slope_over_mad_guide": ag / QM9_MAD[prop],
              "slope_over_mad_oracle": ao / QM9_MAD[prop],
              "n_calibration": int(len(sel))}
    return f_A, f_B, delta, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--edm-dir", default=os.path.join(ROOT, "weights", "EDMsecond"),
                    help="directory holding generative_model_ema.npy + args.pickle")
    ap.add_argument("--props", default="mu,alpha,gap")
    ap.add_argument("--arms", default="",
                    help="comma list; default is baseline + comparators + ours")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--solver", default="euler", choices=["euler", "heun"])
    ap.add_argument("--tau-min", type=float, default=1e-3)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--k-delta", type=float, default=2.0)
    ap.add_argument("--n-probe", type=int, default=1)
    ap.add_argument("--n-mc", type=int, default=4)
    ap.add_argument("--sigma-mc", type=float, default=0.1)
    ap.add_argument("--tau-max-guide", type=float, default=0.5,
                    help="skip guidance while tau > this, i.e. in the noisy "
                         "part of the trajectory. 0.5 is the VP image of the "
                         "main sweep's t_min_guide = 0.5 (V2_WIN); 1.0 guides "
                         "everywhere. NOT a free parameter -- see the module "
                         "docstring, item 5.")
    ap.add_argument("--n-calib", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="one throwaway cell per arm at minimum cost, writes "
                         "nothing, exits nonzero if any arm raises")
    args = ap.parse_args()

    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else args.device)
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    props = [p for p in args.props.split(",") if p]
    arms = ([a for a in args.arms.split(",") if a] if args.arms
            else BASELINE + COMPARE_ARMS + OUR_ARMS)

    if args.preflight:
        args.n, args.batch, args.steps = 8, 8, 10
    cfg = config_tag(args)

    cells = []
    for prop in props:
        for arm in arms:
            for tgt in TARGET_QUANTILES:
                # The unguided baseline has no strength knob. Running it at
                # every w would write identical cells under different names
                # and inflate the "done" count.
                for w in ([1.0] if arm == "unguided" else STRENGTHS):
                    cells.append((prop, arm, tgt, w))
    if args.preflight:
        seen, first = set(), []
        for c in cells:
            if c[0] != props[0] or c[1] in seen:
                continue
            seen.add(c[1])
            first.append(c)
        cells = first

    todo = cells if args.preflight else [
        c for c in cells
        if not os.path.exists(os.path.join(out, cell_name(*c, cfg=cfg)))]
    print("cells total %d | done %d | to run %d"
          % (len(cells), len(cells) - len(todo), len(todo)))
    if args.dry_run:
        for c in todo[:20]:
            print("   ", cell_name(*c, cfg=cfg))
        return 0
    if not todo:
        print("nothing to do -- the transfer sweep is complete")
        return 0

    d = torch.load(DATA, weights_only=True)
    types = d["types"]
    net, prov = load_backend(args, dev)
    prov.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                 "device": (torch.cuda.get_device_name(0)
                            if dev == "cuda" and torch.cuda.is_available()
                            else "cpu"),
                 "norm_values": net.norm_values,
                 "noise_schedule": net.args["diffusion_noise_schedule"],
                 "diffusion_steps": net.args["diffusion_steps"]})
    print("base model: EDMsecond  md5 %s  nf=%d layers=%d  schedule=%s"
          % (prov["edm_md5"][:12], net.args["nf"], net.args["n_layers"],
             prov["noise_schedule"]))

    g = torch.Generator().manual_seed(4242)
    calib_sel = torch.randperm(d["coords"].shape[0], generator=g)[:args.n_calib]

    guides, evals, deltas, reports = {}, {}, {}, {}
    for prop in props:
        f_A, f_B, delta, rep = calibrate(prop, d, calib_sel, dev, args.k_delta)
        guides[prop], evals[prop] = f_A, f_B
        deltas[prop] = delta
        reports[prop] = rep
        print("  %-6s guide MAE %.5f %s | oracle MAE %.5f | delta %.5f | "
              "slope/MAD %.3f (guide) %.3f (oracle)"
              % (prop, rep["guide_mae"], PROP_UNITS[prop], rep["oracle_mae"],
                 delta, rep["slope_over_mad_guide"], rep["slope_over_mad_oracle"]))
        # A slope far from the published MAD means the adapter is feeding the
        # network the wrong units, and every number downstream would be
        # meaningless while looking fine. Fail loudly here instead.
        for who in ("guide", "oracle"):
            r = rep["slope_over_mad_%s" % who]
            if not 0.9 < r < 1.1:
                raise SystemExit(
                    "%s %s calibration slope is %.3f x QM9's MAD; expected ~1. "
                    "The adapter is almost certainly feeding it features on the "
                    "wrong scale." % (prop, who, r))

    targets = {}
    for prop in props:
        col = d["y"][:, PROP_INDEX[prop]].double()
        targets[prop] = {k: float(torch.quantile(col, q))
                         for k, q in TARGET_QUANTILES.items()}
        print("  %-6s targets %s" % (prop, targets[prop]))

    # Molecule sizes: the same val slice the main sweep starts from, so the
    # two experiments see the same size distribution and a difference between
    # them cannot be a difference in how big the molecules are.
    va = d["split"]["val"][: args.n]
    mask_v = d["mask"][va].to(dev)
    clip = None if args.clip < 0 else args.clip

    t0, done, failed = time.time(), 0, []

    def run_cell(prop, arm, tgt, w):
        f_A, f_B = guides[prop], evals[prop]
        delta = deltas[prop]
        extra = arm_kwargs(arm, delta)
        s = f_A.y_std
        w_applied = w * strength_scale(arm, extra, s)
        target = targets[prop][tgt]
        y_t = torch.full((args.n,), target, device=dev)
        gen = torch.Generator(device=dev).manual_seed(args.seed)

        cs, fs, calls = [], [], 0
        cost = {k: 0 for k in ("gen_fwd", "gen_vjp", "gen_jvp",
                               "guide_fwd", "guide_bwd", "guide_hvp")}
        clipped, diag_log = 0, {}
        for i in range(0, args.n, args.batch):
            m = mask_v[i:i + args.batch]
            c0, f0 = initial_noise(m, len(types), gen)
            smp = VPSampler(
                net, m, tau_min=args.tau_min, noise_schedule=net.schedule,
                tau_max_guide=args.tau_max_guide,
                f_net=(None if arm == "unguided" else f_A),
                y=y_t[: m.shape[0]], s=s, mode=arm, w=w_applied, clip=clip,
                n_probe=args.n_probe, n_mc=args.n_mc, sigma_mc=args.sigma_mc,
                **extra)
            c, f, nc = integrate(smp, c0, f0, args.steps, args.solver)
            cs.append(c)
            fs.append(f)
            calls += nc
            for k in cost:
                cost[k] += getattr(smp.cost, k)
            clipped += smp.n_clipped
            for dk, dv in smp.diag_summary().items():
                tot, cnt = diag_log.get(dk, (0.0, 0))
                diag_log[dk] = (tot + dv, cnt + 1)

        # Scored in the SAMPLER's space, which is EDM's normalised space, and
        # that is correct for all three consumers:
        #   * coordinates -- norm_values[0] is 1 for EDM's QM9 recipe, asserted
        #     at load time, so normalised coordinates ARE angstroms and the
        #     bond-distance tables apply unchanged;
        #   * atom types -- `stability` and `to_smiles` read the element by
        #     argmax over the feature channels, which a common positive scale
        #     cannot change;
        #   * f_A and f_B -- `Calibrated` takes sampler-space features by
        #     contract and each rescales to what its own network wants.
        # Denormalising here instead would silently feed f_A features 8x too
        # large, and its predictions would be wrong while nothing raised.
        C, F = torch.cat(cs), torch.cat(fs)
        r = evaluate_samples(C, F, mask_v, types, f_A, f_B, y_t, delta)
        r.pop("delta", None)
        r.update({
            "prop": prop, "arm": arm, "target_name": tgt, "target": target,
            "arm_class": ("OURS" if arm in OUR_ARMS else
                          "COMPARE" if arm in COMPARE_ARMS else "BASELINE"),
            "backend": "TFG/EDMsecond", "prov": prov,
            "calibration": reports[prop],
            "w": w, "w_applied": w_applied, "n": args.n, "steps": args.steps,
            "solver": args.solver, "delta": delta, "mae_B": reports[prop]["oracle_mae"],
            "n_probe": args.n_probe, "n_mc": args.n_mc, "sigma_mc": args.sigma_mc,
            "clip": args.clip, "k_delta": args.k_delta, "seed": args.seed,
            "guide_time": "zero", "tau_min": args.tau_min,
            "tau_max_guide": args.tau_max_guide,
            "cost": cost, "field_evals": calls,
            "clipped_sample_steps": clipped,
            "diag": {k: tot / cnt for k, (tot, cnt) in diag_log.items()},
            "batch": args.batch})
        return r

    for (prop, arm, tgt, w) in todo:
        if args.max_minutes and done:
            used = (time.time() - t0) / 60.0
            if used + used / done > args.max_minutes:
                print("  time guard: %.1f min used, stopping with %d cells left"
                      % (used, len(todo) - done))
                break
        cs_t = time.time()
        try:
            r = run_cell(prop, arm, tgt, w)
        except Exception as exc:                       # noqa: BLE001
            failed.append((cell_name(prop, arm, tgt, w, cfg), repr(exc)))
            print("  FAILED %s: %r" % (cell_name(prop, arm, tgt, w, cfg), exc))
            continue
        done += 1
        if not args.preflight:
            with open(os.path.join(out, cell_name(prop, arm, tgt, w, cfg)),
                      "w") as fh:
                json.dump(r, fh, indent=1)
        print("  %-44s MAE %9.4f (%5.2f d)  in-band %.3f  mol-stab %.3f  %.1fs"
              % (cell_name(prop, arm, tgt, w), r["prop_mae_eval"],
                 r["prop_mae_eval"] / r["delta"], r["in_band_fraction"],
                 r["mol_stability"], time.time() - cs_t))

    if failed:
        print("\n%d cell(s) failed:" % len(failed))
        for name, exc in failed:
            print("   %s  %s" % (name, exc))
        return 1
    if args.preflight:
        print("\npreflight OK: %d arms ran end to end, nothing written" % done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
