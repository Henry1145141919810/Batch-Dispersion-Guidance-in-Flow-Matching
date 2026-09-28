"""D1. Feed BDG's OWN bounded error law e = S/tau^2 - 1 (e >= -1, so w_eff >= 1-eta)
with the two candidate statistics S, using the repo's measured curves
(results/variance_decomposition/*.json, n=256, seed 20261001, q90):

  S = within(t) = mean_i V_F,i(t)   -- btvg's statistic
  S = across(t) = Var_i f(m_i(t))   -- BDG's V_b

Question: is boundedness (w_eff >= 1-eta, the objection's 'correct reason')
enough to make a two-sided law safe? Or is the load-bearing fact the one the
handoff states -- that V_b is not driven to zero by the schedule -- i.e. that
the SIGN of e is conditional on the batch rather than on t?

For each run and each requested tau (tau_mult * terminal sd, plus btvg's own
default tau = delta/1.96) report:
  frac_widen  fraction of the logged guided window with e < 0 (widening sign)
  t_last_flip last t at which sign(e) changes (None = never changes)
  e_end       e at the last logged t
"""
import json
import os

ROOT = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
D = os.path.join(ROOT, "results", "variance_decomposition")
RUNS = ["mu_q90_unguided.json", "mu_q90_btvg.json", "alpha_q90_unguided.json"]


def summarise(ts, S, tau2):
    e = [s / tau2 - 1.0 for s in S]
    widen = [x < 0 for x in e]
    flips = [ts[i] for i in range(1, len(e)) if widen[i] != widen[i - 1]]
    return (sum(widen) / len(widen), flips[-1] if flips else None, e[-1], min(e))


def main():
    out = {}
    for f in RUNS:
        d = json.load(open(os.path.join(D, f)))
        rows = d["rows"]
        ts = [r["t"] for r in rows]
        W = [r["within"] for r in rows]
        A = [r["across"] for r in rows]
        tv = d["terminal_var"]
        delta = d["delta"]
        name = f.replace(".json", "")
        print("\n== %s  (terminal var %.4g, delta %.4g, %d logged steps, t %.2f..%.2f)"
              % (name, tv, delta, len(ts), ts[0], ts[-1]))
        print("   within: %.4g -> %.4g   across: %.4g -> %.4g"
              % (W[0], W[-1], A[0], A[-1]))
        reqs = [("tau_mult 0.5 (TIGHTEN request)", 0.25 * tv),
                ("tau_mult 1.0", tv),
                ("tau_mult 1.5 (WIDEN request)", 2.25 * tv),
                ("btvg default tau=delta/1.96", (delta / 1.96) ** 2)]
        out[name] = {}
        for lab, tau2 in reqs:
            fw_F, fl_F, eF_end, eF_min = summarise(ts, W, tau2)
            fw_b, fl_b, eb_end, eb_min = summarise(ts, A, tau2)
            print("   %-32s | S=V_F: widen %5.1f%% of window, last flip t=%s, e_end %+.4f"
                  "  | S=V_b: widen %5.1f%%, last flip t=%s, e_end %+.3f"
                  % (lab, 100 * fw_F, "%.2f" % fl_F if fl_F else "never", eF_end,
                     100 * fw_b, "%.2f" % fl_b if fl_b else "never", eb_end))
            out[name][lab] = dict(VF_frac_widen=fw_F, VF_last_flip=fl_F, VF_e_end=eF_end,
                                  Vb_frac_widen=fw_b, Vb_last_flip=fl_b, Vb_e_end=eb_end)
        # smallest tau_mult for which V_F's END-OF-WINDOW sign is NOT widening
        tm_min = (W[-1] / tv) ** 0.5
        print("   V_F's final sign is WIDEN for every tau_mult > %.4f (i.e. any request above "
              "%.1f%% of the natural spread)" % (tm_min, 100 * tm_min))
        out[name]["VF_widen_for_tau_mult_above"] = tm_min

    # per-molecule: fraction of TRAJECTORIES with V_F,i < tau^2 at late t (btvg's own tau)
    d = json.load(open(os.path.join(D, "mu_q90_unguided.json")))
    tau2 = (d["delta"] / 1.96) ** 2
    print("\n== per-molecule, mu/unguided, btvg default tau^2 = %.3g" % tau2)
    for lv in d["late_V"]:
        V = lv["V"]
        below = sum(1 for v in V if v < tau2) / len(V)
        print("   t=%.3f  frac of molecules with V_F,i < tau^2 (raw coef would WIDEN): %.3f"
              % (lv["t"], below))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "d1_error_signal.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
