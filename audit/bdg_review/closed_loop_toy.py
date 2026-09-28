"""Three claims about the BDG loop, on a toy where the answer is knowable.

No generator, no GPU: a batch of scalars F_i driven by exactly BDG's numerator,
F_i <- F_i + h * [ (y - F_i) - eta * e * (F_i - F_bar) ], which is what the real
arm does to the predicted property to first order (the pullback contributes a
common positive factor ||g||^2/s^2, folded into h).

CLAIM 1 (handoff section 2). "The 2/(B-1) is absorbed into eta so the gain does
not depend on batch size."  Worth checking, because the FIELD obviously does not
depend on B while the DESCENT RATE ON V_b might. Analytically:
    dV_b/dt = sum_i (dV_b/dF_i)(dF_i/dt),  dV_b/dF_i = 2(F_i-F_bar)/(B-1)
    dispersion part = -eta e (2/(B-1)) sum_i (F_i-F_bar)^2 = -2 eta e V_b
since sum_i (F_i-F_bar)^2 = (B-1) V_b. The (B-1) CANCELS, so the relative
contraction rate is B-independent. Checked numerically below.

CLAIM 2 (handoff section 2, "the fixed point is stable and the clamp comes
off"). The handoff licenses removing btvg's <=0 clamp by arguing V_b does not
vanish. The stronger fact is ALGEBRAIC and needs no dynamics: V_b >= 0 forces
    e = V_b/tau^2 - 1 >= -1
so the widening coefficient is bounded by eta no matter what the batch does,
and w_eff = 1 + eta e >= 1 - eta. btvg's -0.5(1/tau^2 - 1/V_F) has no such
bound because V_F -> 0. Checked by driving V_b to 0 on purpose.

CLAIM 3 (the estimator question the handoff raises but does not measure).
V_b over B samples is a noisy estimate of the population spread; the handoff
runs B = 512 for that reason. How much does the achieved spread degrade at
B = 32, 64, 128? This is the sample-size cost of "the batch is the estimator".
"""
import json
import os

import torch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "closed_loop_toy.json")


def roll(B, eta, tau, y=0.0, h=0.02, steps=400, sd0=1.0, seed=0, onesided=False):
    g = torch.Generator().manual_seed(seed)
    F = torch.randn(B, generator=g, dtype=torch.float64) * sd0
    hist = []
    for _ in range(steps):
        Fb = F.mean()
        Vb = F.var(unbiased=True)
        e = (Vb - tau ** 2) / tau ** 2
        if onesided:
            e = e.clamp(min=0.0)
        F = F + h * ((y - F) - eta * e * (F - Fb))
        hist.append((float(Vb.sqrt()), float(e), float(1.0 + eta * e)))
    return F, hist


def main():
    res = {}
    print("CLAIM 1 -- is the contraction rate batch-size independent?")
    print("  relative dV_b/V_b over the first step, eta=4, tau=0.5, sd0=1")
    rows = []
    for B in (16, 32, 64, 128, 256, 512, 2048):
        # average over seeds so the ESTIMATOR noise does not masquerade as a
        # batch-size effect -- that is claim 3, measured separately below
        rates = []
        for sd in range(40):
            F, h = roll(B, 4.0, 0.5, steps=1, seed=sd)
            V0 = h[0][0] ** 2
            V1 = float(F.var(unbiased=True))
            rates.append((V1 - V0) / V0)
        r = sum(rates) / len(rates)
        rows.append((B, r))
        print("    B=%-5d  mean relative dV_b = %+0.6f" % (B, r))
    spread = max(r for _, r in rows) - min(r for _, r in rows)
    print("  spread across B = %.2e  -> batch-size independent: %s"
          % (spread, abs(spread) < 5e-3))
    res["claim1_rates"] = rows

    print("\nCLAIM 2 -- is the widening branch bounded without a clamp?")
    # start the batch ALREADY collapsed and ask for a huge spread: the worst
    # case for an unclamped controller
    F, hist = roll(64, 4.0, tau=50.0, sd0=1e-6, steps=300, seed=1)
    e_min = min(e for _, e, _ in hist)
    w_min = min(w for _, _, w in hist)
    print("  tau=50 with sd0=1e-6 (V_b/tau^2 ~ 4e-16): min e %+0.8f  "
          "min w_eff %+0.4f  final sd %.4g  finite=%s"
          % (e_min, w_min, float(F.std()), bool(torch.isfinite(F).all())))
    print("  algebraic bound: e >= -1 and w_eff >= 1-eta = -3.  respected: %s"
          % (e_min >= -1.0 - 1e-12 and w_min >= -3.0 - 1e-12))
    # the btvg contrast: its coefficient has no such bound
    for Vf in (1e-2, 1e-4, 1e-8, 1e-12):
        b = -0.5 * (1.0 / 0.5 ** 2 - 1.0 / Vf)
        print("    btvg coefficient at V_F=%-8.0e : %+0.3e" % (Vf, b))
    res["claim2"] = {"e_min": e_min, "w_eff_min": w_min,
                     "final_sd": float(F.std())}

    print("\nCLAIM 3 -- the estimator cost of a small batch")
    print("  requested tau=0.5 from sd0=1.0, eta=4, 400 steps, 60 seeds")
    rows = []
    for B in (16, 32, 64, 128, 256, 512):
        ach = []
        for sd in range(60):
            F, _ = roll(B, 4.0, 0.5, steps=400, seed=1000 + sd)
            ach.append(float(F.std(unbiased=True)))
        m = sum(ach) / len(ach)
        v = (sum((a - m) ** 2 for a in ach) / (len(ach) - 1)) ** 0.5
        rows.append((B, m, v))
        print("    B=%-4d achieved sd %.4f +- %.4f   (bias %+0.4f, "
              "rel scatter %.1f%%)" % (B, m, v, m - 0.5, 100 * v / m))
    res["claim3"] = rows
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
