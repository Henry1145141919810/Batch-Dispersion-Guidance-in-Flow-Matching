"""D3. (c) 'setpoint-relative scale-free error' -- how far is it already in the repo's
CONTROL LAWS (not just the diagnostics at guidance.py:785/1502)?
  btvg   (guidance.py:1494): b      = -0.5 (1/tau^2 - 1/V)   = -(1/(2 tau^2)) (1 - 1/r), clamp <= 0
  btvg2  (guidance.py:733) : b      = 0.5/s^2 (1 - tau^2/V)_+ = (0.5/s^2) (1 - 1/r)_+
  bdg    (handoff §2)      : coef   = -eta e / s^2 = -(eta/s^2)(r - 1)
All three are functions of r = V/tau^2. Tabulate the WIDENING magnitude each
law would request, unclamped, as r -> 0 (the vanishing-statistic regime).

D4. 'the setpoint is never reached': where does the real loop sit in the last 20
guided steps (weff_traj.json, the GPU lens's per-step log), against the
closed-form P-control equilibrium V*/tau^2 = 1 - 1/eta = 0.75, and is it still
moving (slope over the last 10 steps)?
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
eta = 4.0

print("D3  widening coefficient requested by each law (positive = widen), unclamped")
print("    r=V/tau^2 | btvg  x tau^2 | btvg2 x s^2 (sign-flipped) | bdg x s^2 (eta=4)")
for r in (2.0, 1.0, 0.75, 0.5, 0.1, 1e-2, 1e-4, 1e-8):
    btvg = 0.5 * (1.0 / r - 1.0)          # -b*tau^2, widen if > 0
    btvg2 = 0.5 * (1.0 / r - 1.0)         # -(unclamped b)*s^2, widen if > 0
    bdg = -eta * (r - 1.0)                # widen if > 0, bounded by eta
    print("    %-9g | %+13.4g | %+26.4g | %+.4g" % (r, btvg, btvg2, bdg))
print("    => btvg and btvg2 share the form (1/r - 1)/2: unbounded as r->0. bdg's (1-r)*eta "
      "is bounded by eta. That boundedness is the ONLY formal difference in the error law.")

print("\nD4  late-window position of the real loop (GPU lens, n=256, eta=4, fm_ema.pt)")
d = json.load(open(os.path.join(HERE, "..", "weff_traj.json")))
rows = []
for r in d:
    tr = r["traj"]
    v = [x["bdg_V_over_tau2"] for x in tr]
    w = [x["bdg_w_eff"] for x in tr]
    L = v[-20:]
    m = sum(L) / len(L)
    last10 = v[-10:]
    slope = (last10[-1] - last10[0]) / 9.0
    wl = w[-20:]
    wm = sum(wl) / len(wl)
    near = sum(1 for x in wl if abs(x) < 0.25) / len(wl)
    sd_s = (v[-1] ** 0.5) * r["tau_mult"]
    rows.append((r["prop"], r["target"], r["tau_mult"], m, slope, wm, near, sd_s))
    print("    %-4s %-4s tau_mult %-4g  V/tau^2 last20 %.3f (eq 0.750)  slope/step %+.4f  "
          "w_eff last20 %+.3f  |w_eff|<0.25 %3.0f%%   achieved sd/s %.3f vs requested %.2f"
          % (r["prop"], r["target"], r["tau_mult"], m, slope, wm, 100 * near, sd_s,
             r["tau_mult"]))
by = {}
for p, tg, tm, m, sl, wm, nr, sds in rows:
    by.setdefault(tm, []).append((m, wm, sds))
print("\n    by request:")
for tm in sorted(by):
    ms = [x[0] for x in by[tm]]
    ws = [x[1] for x in by[tm]]
    ss = [x[2] for x in by[tm]]
    print("    tau_mult %-4g  V/tau^2 %.3f..%.3f   w_eff %+.2f..%+.2f   achieved sd/s %.3f..%.3f "
          "(requested %.2f; drooped eq %.3f)" % (tm, min(ms), max(ms), min(ws), max(ws),
                                                  min(ss), max(ss), tm, tm * 0.75 ** 0.5))
json.dump(rows, open(os.path.join(HERE, "d3_forms_and_setpoint.json"), "w"), indent=1)
