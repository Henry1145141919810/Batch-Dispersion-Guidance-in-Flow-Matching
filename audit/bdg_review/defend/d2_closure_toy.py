"""D2. Is the objection's 'correct reason' (e >= -1  =>  w_eff >= 1-eta) sufficient
to license removing the clamp? Or is closure (the measured statistic responds to
the actuator; the handoff's 'V_b does not vanish') the load-bearing half?

Same bounded law in every arm:  w_eff = 1 + eta*e,  e = S/tau^2 - 1 >= -1.
Deviation dynamics (math lens, exact for the linearised plug/BDG step):
      d_i <- d_i * (1 - kappa_n * w_eff),   kappa_n = k0 * (1-t)/t
  A  S = V_b(d)                     closed loop (BDG)
  B  S = V_b(d_0) * within(t)/within(0.5)   the SAME starting value, but decaying
     on btvg's measured schedule (real mu/unguided within(t), 20956x fall) and
     blind to the batch -- i.e. BDG's bounded law fed a V_F-like statistic
  Bc B with the positive-part clamp e -> max(e, 0)   (btvg's fix)

No flow drift (the batch's own across(t) dip/re-widen is omitted; caveat).
Units: natural spread = 1, so tau = tau_mult.
"""
import json
import os

import numpy as np

ROOT = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
VD = json.load(open(os.path.join(ROOT, "results", "variance_decomposition",
                                 "mu_q90_unguided.json")))
T_LOG = np.array([r["t"] for r in VD["rows"]])
W_LOG = np.array([r["within"] for r in VD["rows"]])


def within_rel(t):
    return float(np.exp(np.interp(t, T_LOG, np.log(W_LOG)))) / W_LOG[0]


def run(signal, tau_mult, k0, eta=4.0, steps=50, B=4096, seed=0, t_end=0.99,
        fade=True):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(B)
    d = (d - d.mean()) / d.std(ddof=1)          # V_b(0) = 1 exactly
    V0 = 1.0
    tau2 = tau_mult ** 2
    ts = np.linspace(0.5, t_end, steps)
    n_widen = 0
    w_hist = []
    for t in ts:
        Vb = d.var(ddof=1)
        if signal == "A":
            S = Vb
        else:
            S = V0 * within_rel(min(t, T_LOG[-1]))
        e = S / tau2 - 1.0
        if signal == "Bc":
            e = max(e, 0.0)
        w = 1.0 + eta * e
        kap = k0 * ((1 - t) / t if fade else 1.0)
        d = d * (1.0 - kap * w)
        n_widen += (w < 0)
        w_hist.append(w)
        if not np.isfinite(d).all() or d.std() > 1e6:
            return dict(sd=float("inf"), frac_widen=n_widen / steps, w_min=min(w_hist),
                        w_max=max(w_hist))
    return dict(sd=float(d.std(ddof=1)), frac_widen=n_widen / steps,
                w_min=float(min(w_hist)), w_max=float(max(w_hist)))


def main():
    out = {}
    TM = [0.5, 0.75, 1.0, 1.25, 1.5]
    for k0, fade, steps, lab in [(0.01, True, 50, "realistic window: kappa=k0(1-t)/t, k0=0.01, 50 steps"),
                                 (0.05, True, 50, "5x gain, same window"),
                                 (0.02, False, 400, "structural: constant kappa=0.02, 400 steps")]:
        print("\n== %s" % lab)
        print("   %-8s | %-30s | %-30s | %-30s" % ("tau_mult", "A  closed loop on V_b",
                                                    "B  bounded e fed V_F-like", "Bc B + clamp"))
        rows = {}
        for tm in TM:
            r = {s: run(s, tm, k0, steps=steps, fade=fade) for s in ("A", "B", "Bc")}
            rows[tm] = r
            print("   %-8g | sd %7.4g widen %4.0f%% w[%+.2f,%+.2f] | sd %7.4g widen %4.0f%% w[%+.2f,%+.2f] | sd %7.4g widen %4.0f%%"
                  % (tm, r["A"]["sd"], 100 * r["A"]["frac_widen"], r["A"]["w_min"], r["A"]["w_max"],
                     r["B"]["sd"], 100 * r["B"]["frac_widen"], r["B"]["w_min"], r["B"]["w_max"],
                     r["Bc"]["sd"], 100 * r["Bc"]["frac_widen"]))
        for s in ("A", "B", "Bc"):
            sds = [rows[tm][s]["sd"] for tm in TM]
            mono = all(b >= a for a, b in zip(sds, sds[1:]))
            print("   %-3s achieved sd monotone in request: %s" % (s, mono))
        if not fade:
            print("   A fixed point prediction sd = tau*sqrt(1-1/eta) = %s"
                  % ", ".join("%.4f" % (tm * 0.75 ** 0.5) for tm in TM))
        out[lab] = {str(k): v for k, v in rows.items()}
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "d2_closure_toy.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
