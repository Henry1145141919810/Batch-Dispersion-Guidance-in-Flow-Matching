"""BLUE-TEAM check 4: is tau_mult a PORTABLE knob in a way a fixed weight is not?

A fixed schedule (constant or time-varying) applied to a different plant
(property/target) lands at a different spread, because the per-step gain and the
flow's drift differ by plant (d2: early gain ~0.3 for mu, ~0.1 for gap). Feedback
should compensate. Test on the 36 GPU cells at seed 20260925 (n=256, same noise
for every cell of a (prop,target) block): at each knob setting, how much does the
achieved sd(f_A)/unguided vary ACROSS the 6 (prop,target) configs? Compare with an
iid-bootstrap null for the ratio (the part of that variation noise alone makes)."""
import os, json
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); BDG = os.path.dirname(HERE)
rng = np.random.default_rng(1)
def fa(prop, tgt, arm, w, var):
    pm = torch.load(os.path.join(BDG, "cells", "%s__%s__%s__%s__tmin0.5__%s.permol.pt" % (prop, arm, tgt, w, var)), weights_only=False)
    return pm["f_A"][pm["finite"]].double().numpy()
def boot_ratio_se(x, u, nb=4000):
    # PAIRED by molecule index would understate; treat as independent -> conservative (larger) null
    ix = rng.integers(0, len(x), (nb, len(x))); iu = rng.integers(0, len(u), (nb, len(u)))
    r = x[ix].std(1, ddof=1) / u[iu].std(1, ddof=1)
    return r.std(ddof=1)
cfg = [(p, t) for p in ("mu", "alpha", "gap") for t in ("q50", "q90")]
knobs = [("plug w1", "plug", "w1", "tgt"), ("plug w4", "plug", "w4", "tgt")] + \
        [("bdg t%s" % m, "bdg", "w1", "e4t%s" % m) for m in ("0.5", "0.75", "1", "1.25", "1.5")]
out = {}
print("%-10s | %-6s %-6s %-6s | %-8s | %s" % ("knob", "mean", "sd", "range", "noise sd", "sd/noise  (ratios per config)"))
for lab, arm, w, var in knobs:
    rs, ses = [], []
    for p, t in cfg:
        u = fa(p, t, "unguided", "w0", "bdgctl"); x = fa(p, t, arm, w, var)
        rs.append(x.std(ddof=1) / u.std(ddof=1)); ses.append(boot_ratio_se(x, u))
    rs = np.array(rs); noise = float(np.sqrt(np.mean(np.array(ses) ** 2)))
    out[lab] = dict(ratios=rs.tolist(), mean=float(rs.mean()), sd=float(rs.std(ddof=1)), noise=noise)
    print("%-10s | %.3f  %.3f  %.3f  | %.4f   | %.2f      %s" % (lab, rs.mean(), rs.std(ddof=1), np.ptp(rs), noise, rs.std(ddof=1) / noise,
          " ".join("%.3f" % v for v in rs)))
json.dump(out, open(os.path.join(HERE, "d4_portability.json"), "w"), indent=1)
print("\nconfig order:", cfg)
print("noise sd = rms bootstrap se of each ratio treating numerator and unguided as independent (conservative)")
