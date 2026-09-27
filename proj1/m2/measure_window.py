"""Measure the guidance window, and WRITE THE RESULT. Protocol section 2.1.

M2 departs from v3's t >= 0.5 and guides from t = 0. The repository asserts two
things in support, both only in comments, and they disagree with each other
(8.7 % in one file, 17 % in another, for the same stated condition). No script
here computed either. This one does, and writes a file:

  A. the batch sd of the GUIDED property at the endpoint estimate, versus t.
     If the property's spread is settled before t = 0.5, then guiding only
     after 0.5 steers a decision already made.
  B. the gap closure actually achieved at t_min in {0, 0.5}, same arm, same
     strength, same seeds -- which is the claim in the operational form.

Writes results/m2_window.json and docs/results/M2_WINDOW.md.

  python proj1/m2/measure_window.py --device cuda
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_fm as S      # noqa: E402
import m2_sweep as M        # noqa: E402


def trajectory_spread(net, ck, prop, n, steps, seed, dev, grid):
    """Batch sd of f_soft(m) at the endpoint estimate m, along an UNGUIDED path."""
    L = ck["crop"]
    g = torch.Generator().manual_seed(seed)
    e = torch.empty(n, L, 4).exponential_(generator=g)
    x = (e / e.sum(-1, keepdim=True)).to(dev)
    dt, out = 1.0 / steps, {}
    soft = S.PROPS[prop][0]
    want = sorted(grid)
    with torch.no_grad():
        for i in range(steps):
            t = i * dt
            v = net(x, torch.full((n,), t, device=dev))
            while want and t >= want[0] - 1e-9:
                m = x + (1.0 - t) * v          # the endpoint estimate guidance reads
                out[round(want.pop(0), 4)] = float(soft(m).std())
            x = M._step(x, v, dt)
        # Any grid point beyond the last step the sampler actually takes (only
        # possible when steps is too coarse to reach it) is recorded from the
        # final state rather than dropped, so the table has no silent holes.
        if want:
            m = x
            for t in want:
                out[round(t, 4)] = float(soft(m).std())
        out["final"] = float(S.PROPS[prop][1](x).std())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=M.DEFAULT_CKPT)
    ap.add_argument("--props", default="gc,cpg")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--w", type=float, default=16.0)
    ap.add_argument("--arm", default="plug")
    ap.add_argument("--seeds", default="20260921,20260922,20260923")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json-out", default="results/m2_window.json")
    ap.add_argument("--md-out", default="docs/results/M2_WINDOW.md")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = S.SimplexFM(ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state_dict"]); net.eval(); net = net.to(dev)
    X, _, _ = S.load_dfb(crop=ck["crop"])
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.49, 0.6, 0.8, 0.99]

    res = {"ckpt": os.path.basename(a.ckpt), "n": a.n, "steps": a.steps,
           "w": a.w, "arm": a.arm, "seeds": seeds, "device": dev,
           "spread": {}, "closure": {}}

    for prop in a.props.split(","):
        real = S.PROPS[prop][1](X)
        s, y = float(real.std()), float(real.median())
        quantum = 1.0 / (ck["crop"] if prop == "gc" else ck["crop"] - 1)
        delta = max(0.16 * s, 4.4 * quantum)
        real_kmer = M.kmer_freq(X)

        print("[%s] A. batch sd of %s_soft(m) along an unguided path" % (prop, prop))
        sp = trajectory_spread(net, ck, prop, a.n, a.steps, seeds[0], dev, grid)
        res["spread"][prop] = sp
        for t in grid:
            print("      t=%.2f  sd=%.5f" % (t, sp[round(t, 4)]))
        print("      final (decoded) sd=%.5f   corpus sd=%.5f" % (sp["final"], s))

        print("[%s] B. gap closure at t_min in {0, 0.5}, arm=%s w=%g"
              % (prop, a.arm, a.w))
        res["closure"][prop] = {"s": s, "y": y, "delta": delta}
        base = None
        for t_min in (0.0, 0.5):
            means, ibs = [], []
            for sd_ in seeds:
                r, _ = M.run_cell(net, ck, a.arm, "-", a.w, y, s, None, 0.0,
                                  False, a.n, a.steps, t_min, 1.0, sd_, delta,
                                  real_kmer, prop=prop, dev=dev, batch=a.batch)
                means.append(r["gc_mean"]); ibs.append(r["in_band_fraction"])
            res["closure"][prop]["t%g" % t_min] = {
                "prop_mean": sum(means) / len(means),
                "in_band": sum(ibs) / len(ibs)}
            print("      t_min=%.1f  mean=%.5f  in_band=%.4f"
                  % (t_min, sum(means) / len(means), sum(ibs) / len(ibs)))
        # unguided reference, so "closure" has a start point
        means = []
        for sd_ in seeds:
            r, _ = M.run_cell(net, ck, "unguided", "-", a.w, y, s, None, 0.0,
                              False, a.n, a.steps, 0.0, 1.0, sd_, delta,
                              real_kmer, prop=prop, dev=dev, batch=a.batch)
            means.append(r["gc_mean"])
        u = sum(means) / len(means)
        res["closure"][prop]["unguided_mean"] = u
        for t_min in (0.0, 0.5):
            gm = res["closure"][prop]["t%g" % t_min]["prop_mean"]
            frac = (gm - u) / (y - u) if abs(y - u) > 1e-12 else float("nan")
            res["closure"][prop]["t%g" % t_min]["gap_closed"] = frac
            print("      t_min=%.1f  closes %.1f%% of the gap (unguided %.5f -> "
                  "target %.5f)" % (t_min, 100 * frac, u, y))

    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    json.dump(res, open(a.json_out, "w"), indent=1)
    if a.md_out:
        os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
        with open(a.md_out, "w") as fh:
            fh.write("# M2 guidance window, measured\n\n")
            fh.write("Protocol section 2.1. Generated by "
                     "`proj1/m2/measure_window.py`; n=%d, NFE=%d, seeds %s, "
                     "arm `%s` at w=%g.\n\n" % (a.n, a.steps, seeds, a.arm, a.w))
            for prop in res["spread"]:
                fh.write("## %s\n\n### A. batch sd of the guided property "
                         "at the endpoint estimate\n\n| t | sd |\n|---|---|\n"
                         % prop)
                for t in grid:
                    fh.write("| %.2f | %.5f |\n" % (t, res["spread"][prop][round(t, 4)]))
                fh.write("| final (decoded) | %.5f |\n\n"
                         % res["spread"][prop]["final"])
                c = res["closure"][prop]
                fh.write("### B. gap closure\n\nunguided mean %.5f, target %.5f"
                         "\n\n| t_min | mean | in_band | gap closed |\n"
                         "|---|---|---|---|\n" % (c["unguided_mean"], c["y"]))
                for t_min in (0.0, 0.5):
                    d = c["t%g" % t_min]
                    fh.write("| %.1f | %.5f | %.4f | %.1f%% |\n"
                             % (t_min, d["prop_mean"], d["in_band"],
                                100 * d["gap_closed"]))
                fh.write("\n")
        print("wrote %s and %s" % (a.json_out, a.md_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
