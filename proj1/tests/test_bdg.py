"""Exact gates for BDG: two-sided setpoint control of the across-batch variance.

Reuses test_v2_arms' diagonal fixture so every quantity has a closed form:

  posterior   m = D * x elementwise  =>  J = diag(D)
  property    f(m) = 1/2 sum a_i m_i^2 + sum b_i m_i

Closed forms checked here:
  F_bar = mean_i f(m_i)                    over the BATCH axis
  V_b   = var_i  f(m_i)   (unbiased)       the summand that SURVIVES
  e     = (V_b - tau^2) / tau^2
  num_i = (y - F_i) - eta * e * (F_i - F_bar)      [c_term = 0, this family]
  G_i   = J^T ( num_i / s^2 * g_i )

THE GATE THAT MATTERS: eta = 0 must reproduce `plug` BIT-IDENTICALLY, not
approximately. That is the base-model control for the ablation table, and it is
only free if it is exact.

Run: python proj1/tests/test_bdg.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "tests"))
from guidance import (DISPLACEMENT_MODES, KNOWN_MODES, Cost,  # noqa: E402
                      guidance_field)
from test_v2_arms import setup  # noqa: E402

torch.set_default_dtype(torch.float64)
R = {}
FAIL = []


def check(name, got, want, tol=0.0):
    R[name] = got
    ok = (got == want) if tol == 0.0 else (abs(got - want) <= tol)
    if not ok:
        FAIL.append("%s: got %r want %r (tol %g)" % (name, got, want, tol))


def main():
    E = setup()
    mask, coords, feats = E["mask"], E["coords"], E["feats"]
    f_net, post_fn = E["f_net"], E["post_fn"]
    m_c, m_f = E["m_c"], E["m_f"]
    y, s = 0.5, 1.7

    F = f_net(m_c, m_f, mask).detach()          # [B]
    Fbar = F.mean()
    V_b = F.var(unbiased=True)

    # ---- 1. registration -------------------------------------------------
    check("bdg_in_KNOWN_MODES", "bdg" in KNOWN_MODES, True)
    check("bdg_NOT_displacement", "bdg" in DISPLACEMENT_MODES, False)

    # ---- 2. THE BIT-IDENTITY GATE ---------------------------------------
    cp, cb = Cost(), Cost()
    Gp_c, Gp_f, dp = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                    mode="plug", cost=cp)
    Gb_c, Gb_f, db = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                    mode="bdg", bdg_eta=0.0,
                                    bdg_tau=float(V_b.sqrt()), cost=cb)
    check("eta0_coords_bitidentical", torch.equal(Gp_c, Gb_c), True)
    check("eta0_feats_bitidentical", torch.equal(Gp_f, Gb_f), True)
    check("eta0_cost_identical", cp.as_dict() == cb.as_dict(), True)

    # ---- 3. the closed form ---------------------------------------------
    for eta, tau_mult, one in [(1.0, 0.5, False), (4.0, 1.5, False),
                               (2.0, 1.0, False), (4.0, 1.5, True)]:
        tau = float(V_b.sqrt()) * tau_mult
        e = (V_b - tau ** 2) / tau ** 2
        if one:
            e = e.clamp(min=0.0)
        num = (y - F) - eta * e * (F - Fbar)
        want = num / s ** 2
        _, _, dg = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                  mode="bdg", bdg_eta=eta, bdg_tau=tau,
                                  bdg_onesided=one)
        tag = "e%gt%g%s" % (eta, tau_mult, "o" if one else "")
        check("num_closed_form_" + tag,
              (dg["num"] - want * s ** 2).abs().max().item(), 0.0, 1e-12)

    # ---- 4. SIGN: the whole point ---------------------------------------
    # tau below the current spread  => e > 0 => CONTRACT: molecules above the
    # batch mean get their coefficient pulled DOWN relative to plug.
    tau_lo = float(V_b.sqrt()) * 0.5
    _, _, d_lo = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                mode="bdg", bdg_eta=1.0, bdg_tau=tau_lo)
    hi = F > Fbar
    d_num = d_lo["num"] - dp["num"]
    check("contract_pulls_high_down", bool((d_num[hi] < 0).all()), True)
    check("contract_pushes_low_up", bool((d_num[~hi] > 0).all()), True)
    check("contract_e_positive", bool(d_lo["bdg_e"][0] > 0), True)
    check("contract_not_flagged_widening",
          float(d_lo["bdg_widening"][0]), 0.0)

    # tau above the current spread => e < 0 => WIDEN: the signs reverse.
    tau_hi = float(V_b.sqrt()) * 1.5
    _, _, d_hi = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                mode="bdg", bdg_eta=1.0, bdg_tau=tau_hi)
    d_num = d_hi["num"] - dp["num"]
    check("widen_pushes_high_up", bool((d_num[hi] > 0).all()), True)
    check("widen_pulls_low_down", bool((d_num[~hi] < 0).all()), True)
    check("widen_e_negative", bool(d_hi["bdg_e"][0] < 0), True)
    check("widen_flagged", float(d_hi["bdg_widening"][0]), 1.0)

    # ---- 5. the one-sided ablation kills exactly the widening branch -----
    _, _, d_os = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                mode="bdg", bdg_eta=1.0, bdg_tau=tau_hi,
                                bdg_onesided=True)
    check("onesided_kills_widening",
          torch.equal(d_os["num"], dp["num"]), True)
    _, _, d_os2 = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                 mode="bdg", bdg_eta=1.0, bdg_tau=tau_lo,
                                 bdg_onesided=True)
    check("onesided_keeps_contraction",
          torch.equal(d_os2["num"], d_lo["num"]), True)

    # ---- 6. fixed point --------------------------------------------------
    _, _, d_fp = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                mode="bdg", bdg_eta=4.0,
                                bdg_tau=float(V_b.sqrt()))
    check("fixed_point_is_plug", torch.equal(d_fp["num"], dp["num"]), True)

    # ---- 7. diagnostics are persisted -----------------------------------
    for key in ("bdg_V_b", "bdg_tau", "bdg_e", "bdg_dev", "bdg_disp",
                "bdg_widening"):
        check("diag_has_" + key, key in d_lo, True)
    check("diag_V_b_correct",
          abs(float(d_lo["bdg_V_b"][0]) - float(V_b)), 0.0, 1e-12)

    # ---- 8. guards -------------------------------------------------------
    for bad in (None, 0.0, -1.0):
        try:
            guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                           mode="bdg", bdg_eta=1.0, bdg_tau=bad)
            FAIL.append("bdg_tau=%r was accepted" % (bad,))
        except ValueError:
            pass
    R["bad_tau_rejected"] = True

    width = max(len(k) for k in R)
    for k, v in R.items():
        print("  %-*s  %s" % (width, k, v))
    print()
    if FAIL:
        print("FAIL (%d):" % len(FAIL))
        for f in FAIL:
            print("   ", f)
        return 1
    print("all BDG gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
