"""BLUE-TEAM check 1: re-read the GPU open-loop replay as the PAIRED design it is.

The objection says the replay 'sits inside the ~4.4% noise floor'. That floor is
the iid se of an sd at n=256 -- the right yardstick for two INDEPENDENT batches.
But `replay` and `closed` both run on seed B's initial noise; on a deterministic
pipeline (eta=0 == plug is bit-identical at cell level on this GPU) the paired
null for |replay - closed| is ~0, not 4.4%. So |replay - closed| is the size of
the batch-specific part of the schedule, not noise.

The question a control engineer asks is different anyway: does the loop REJECT
batch-level disturbances that a schedule transmits? That is a comparison of
seed-to-seed variation under each policy, with the SAME schedule source:

  open-loop seed-to-seed   |replay_B - closed_A|   (schedule e_A on noise A vs noise B;
                                                    closed_A IS e_A applied to noise A)
  closed-loop seed-to-seed |closed_B - closed_A|

Calibrate both against an iid bootstrap of the seed-A cell's own per-molecule f_A
(what two independent unregulated batches from that distribution would differ by).
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BDG = os.path.dirname(HERE)
R = json.load(open(os.path.join(BDG, "gpu", "openloop_replay_gpu.json")))
rng = np.random.default_rng(0)


def cell(prop, tgt, variant, arm="bdg", w="w1"):
    stem = "%s__%s__%s__%s__tmin0.5__%s" % (prop, arm, tgt, w, variant)
    pm = torch.load(os.path.join(BDG, "cells", stem + ".permol.pt"), weights_only=False)
    fa = pm["f_A"][pm["finite"]].double().numpy()
    return fa


def boot_se_sd(x, nb=20000):
    n = len(x)
    idx = rng.integers(0, n, size=(nb, n))
    s = x[idx].std(axis=1, ddof=1)
    return s.std(ddof=1) / x.std(ddof=1)


rows = []
print("%-4s %-4s %-4s | %-8s %-8s %-8s | %-7s %-7s %-7s | %-6s | %s"
      % ("prop", "tgt", "tau", "sd_A", "sd_C(B)", "sd_R(B)", "|R-C|", "|C-A|", "|R-A|",
         "iid", "cell==seedA?"))
for r in R:
    A, C, Rp = r["seed_A"]["sd_fa"], r["closed"]["sd_fa"], r["replay"]["sd_fa"]
    tm = r["tau_mult"]
    fa = cell(r["prop"], r["target"], "e4t%g" % tm)
    same = abs(fa.std(ddof=1) - A) / A
    se = boot_se_sd(fa)
    iid_diff = np.sqrt(2) * se           # se of |sd1 - sd2|/sd for two iid batches
    rc, ca, ra = abs(Rp - C) / C, abs(C - A) / A, abs(Rp - A) / A
    rows.append(dict(prop=r["prop"], tgt=r["target"], tau=tm, rc=rc, ca=ca, ra=ra,
                     iid=iid_diff, pc=abs(r["plug"]["sd_fa"] - C) / C))
    print("%-4s %-4s %-4g | %.4f   %.4f   %.4f   | %.4f  %.4f  %.4f  | %.4f | rel diff %.1e"
          % (r["prop"], r["target"], tm, A, C, Rp, rc, ca, ra, iid_diff, same))

print("\nsummary by branch (mean over the 4 configs; iid = sqrt(2)*bootstrap se of sd)")
for tm, lab in ((0.5, "TIGHTEN (w_eff>0, contractive open loop)"),
                (1.5, "WIDEN   (w_eff<0, expansive open loop)")):
    sub = [x for x in rows if x["tau"] == tm]
    m = lambda k: np.mean([x[k] for x in sub])
    print("  %s" % lab)
    print("     |replay-closed| paired      %.4f" % m("rc"))
    print("     closed-loop seed-to-seed    %.4f   (|C-A|)" % m("ca"))
    print("     open-loop   seed-to-seed    %.4f   (|R-A|, same schedule e_A)" % m("ra"))
    print("     iid two-batch expectation   %.4f   (E|Z|*sqrt2*se = 0.798*%.4f)"
          % (0.798 * m("iid"), m("iid")))
    print("     count |R-C| > |C-A|         %d/4" % sum(x["rc"] > x["ca"] for x in sub))
    print("     count |R-A| > |C-A|         %d/4" % sum(x["ra"] > x["ca"] for x in sub))

# chi-square of closed-loop seed-to-seed against the iid null, per branch and pooled
from math import erf, sqrt
def chi2_cdf(x, k):
    # regularized lower gamma for integer/2 via series (k small)
    from math import exp, lgamma, log
    a = k / 2.0; xx = x / 2.0
    s, term, n = 0.0, 1.0 / a, 0
    while term > 1e-15 * max(s, 1e-300) or n < 5:
        s += term; n += 1; term *= xx / (a + n)
        if n > 10000: break
    return exp(-xx + a * log(xx) - lgamma(a)) * s

for keyname, key in (("closed-loop |C-A|", "ca"), ("open-loop |R-A|", "ra")):
    for tm in (0.5, 1.5, None):
        sub = [x for x in rows if tm is None or x["tau"] == tm]
        q = sum((x[key] / x["iid"]) ** 2 for x in sub)
        print("  %-18s tau %-5s sum z^2 = %6.3f on %d dof ; P(chi2 <= this | no regulation) = %.3f"
              % (keyname, "all" if tm is None else tm, q, len(sub), chi2_cdf(q, len(sub))))

# the e schedules themselves: how batch-specific are they?
print("\ne schedule, seed A vs seed B closed loop (50 guided steps)")
for r in R:
    ea = np.array(r["seed_A"]["e_used"]); eb = np.array(r["closed"]["e_used"])
    wa, wb = 1 + 4 * ea, 1 + 4 * eb
    print("  %-4s %-4s t%-4g  rms(w_A - w_B) %.3f  rms(w_B) %.3f  corr %.3f  "
          "first-10 rms diff %.3f  last-10 rms diff %.3f"
          % (r["prop"], r["target"], r["tau_mult"], np.sqrt(np.mean((wa - wb) ** 2)),
             np.sqrt(np.mean(wb ** 2)), np.corrcoef(wa, wb)[0, 1],
             np.sqrt(np.mean((wa[:10] - wb[:10]) ** 2)),
             np.sqrt(np.mean((wa[-10:] - wb[-10:]) ** 2))))
json.dump(rows, open(os.path.join(HERE, "d1_paired_replay.json"), "w"), indent=1)
