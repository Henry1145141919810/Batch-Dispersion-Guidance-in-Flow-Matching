"""Applied correction share: how hard does each arm actually push, at w = 1?

v3 §2.2 makes the point for Modality 1 -- equal `w` is not equal force, and it
measures `mean |C(G)| / |V|` over guided steps to say by how much. That number
is what lets a reader interpret "at w = 1" instead of taking it as "at equal
strength".

Modality 2 needs the same number for a sharper reason. A CPU probe at v3's
settings (w = 1, t_min = 0.5, NFE = 100) found M2's guidance moving the
property by ~0.3 % of the acceptance band, with the clip inactive -- so a null
M2 headline would be a statement about the STRENGTH, not about the method. This
script measures the share directly so the write-up can say which it is, with a
number rather than an inference.

Writes results/m2_share.json and docs/results/M2_SHARE.md.

  python proj1/m2/measure_share.py --props gc --device cuda
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simplex_fm as S      # noqa: E402
import m2_sweep as M        # noqa: E402

ARMS = [("unguided", "-"), ("plug", "-"), ("tmpd", "-"), ("lgd_mc", "-"),
        ("tfg_mc", "-"), ("bdg", "e4t0.5"), ("bdg", "e4t1")]


def share(net, ck, prop, y, s, arm, variant, w, t_min, steps, n, seed, dev):
    """mean |correction| / |velocity| over guided steps, plus the clipped share."""
    L = ck["crop"]
    g = torch.Generator().manual_seed(seed)
    e = torch.empty(n, L, 4).exponential_(generator=g)
    x = (e / e.sum(-1, keepdim=True)).to(dev)
    dt = 1.0 / steps
    eta, tau, onesided = 0.0, None, False
    if arm == "bdg":
        import re
        m = re.match(r"^e([0-9.]+)t([0-9.]+)(o?)$", variant)
        eta, onesided = float(m.group(1)), bool(m.group(3))
        tau = float(m.group(2)) * s
    tot, cnt, clipped, nguided = 0.0, 0, 0, 0
    for i in range(steps):
        t = i * dt
        with torch.no_grad():
            v = net(x, torch.full((n,), t, device=dev))
        if arm != "unguided" and t >= t_min:
            G, _ = M.guidance_field(net, x, t, y, s, arm, eta, tau, onesided,
                                    gen=g, prop=prop)
            fac = (1.0 - t) / max(t, 1e-6)
            corr = w * fac * G
            vn = v.reshape(n, -1).norm(dim=1)
            cn = corr.reshape(n, -1).norm(dim=1)
            over = cn > 1.0 * vn
            clipped += int(over.sum()); nguided += n
            sc = torch.where(over, 1.0 * vn / cn.clamp(min=1e-12),
                             torch.ones_like(cn))
            applied = (cn * sc)                       # after the clip
            tot += float((applied / vn.clamp(min=1e-12)).sum()); cnt += n
            v = v + corr * sc.view(-1, 1, 1)
        x = M._step(x, v, dt)
    return (tot / max(cnt, 1), clipped / max(nguided, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=M.DEFAULT_CKPT)
    ap.add_argument("--props", default="gc")
    ap.add_argument("--ws", default="1,4,16,64")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--t-min", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json-out", default="results/m2_share.json")
    ap.add_argument("--md-out", default="docs/results/M2_SHARE.md")
    a = ap.parse_args()
    ws = [float(x) for x in a.ws.split(",")]
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = S.SimplexFM(ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state_dict"]); net.eval(); net = net.to(dev)
    X, _, _ = S.load_dfb(crop=ck["crop"])

    res = {"n": a.n, "steps": a.steps, "t_min": a.t_min, "seed": a.seed,
           "ws": ws, "note": "mean |correction|/|velocity| over guided steps, "
                             "after the clip; v3 section 2.2 reports 0.055 for "
                             "plug on M1 mu at w=1", "by_prop": {}}
    for prop in a.props.split(","):
        real = S.PROPS[prop][1](X)
        s, y = float(real.std()), float(real.median())
        res["by_prop"][prop] = {}
        print("[%s] s=%.5f y=%.5f  (M1 plug at w=1 applies 0.055)" % (prop, s, y))
        for arm, var in ARMS:
            if arm == "unguided":
                continue
            row = {}
            for w in ws:
                sh, cl = share(net, ck, prop, y, s, arm, var, w, a.t_min,
                               a.steps, a.n, a.seed, dev)
                row["w%g" % w] = {"share": sh, "clipped": cl}
                print("   %-11s w=%-4g share %.4f   clipped %.1f%%"
                      % (arm if arm != "bdg" else "bdg_" + var, w, sh, 100 * cl))
            res["by_prop"][prop][arm if arm != "bdg" else "bdg_" + var] = row

    os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
    json.dump(res, open(a.json_out, "w"), indent=1)
    if a.md_out:
        os.makedirs(os.path.dirname(a.md_out) or ".", exist_ok=True)
        with open(a.md_out, "w") as fh:
            fh.write("# M2 applied correction share\n\n"
                     "`mean |correction| / |velocity|` over guided steps, after "
                     "the clip. Generated by `proj1/m2/measure_share.py`; "
                     "n=%d, NFE=%d, t_min=%g, seed %d.\n\n"
                     "For scale, v3 §2.2 measures **0.055** for `plug` on M1's "
                     "`mu` at w = 1, and 0.024-0.138 across its arms. A share "
                     "far below that is why an M2 arm at w = 1 may not move the "
                     "property: it is a statement about the strength, not about "
                     "the method.\n\n" % (a.n, a.steps, a.t_min, a.seed))
            for prop, arms in res["by_prop"].items():
                fh.write("## %s\n\n| arm | %s |\n|---|%s|\n"
                         % (prop, " | ".join("w=%g" % w for w in ws),
                            "---|" * len(ws)))
                for arm, row in arms.items():
                    fh.write("| %s | %s |\n" % (arm, " | ".join(
                        "%.4f" % row["w%g" % w]["share"] for w in ws)))
                fh.write("\nclipped share of guided sample-steps:\n\n"
                         "| arm | %s |\n|---|%s|\n"
                         % (" | ".join("w=%g" % w for w in ws), "---|" * len(ws)))
                for arm, row in arms.items():
                    fh.write("| %s | %s |\n" % (arm, " | ".join(
                        "%.1f%%" % (100 * row["w%g" % w]["clipped"]) for w in ws)))
                fh.write("\n")
        print("wrote %s and %s" % (a.json_out, a.md_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
