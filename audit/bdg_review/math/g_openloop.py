"""Is the handoff's 3 open-loop control ('freeze w_eff, run plain plug') the
right counterfactual?  In the LINEAR regime the deviation dynamics do not see
the target at all, so a frozen w_eff must reproduce the spread exactly.  Any
observed difference is therefore evidence of a NONLINEARITY, not of feedback."""
import numpy as np
ts = np.linspace(0,1,101)[:-1]; ts = ts[ts>=0.5]
B, eta, y, tau2 = 4096, 4.0, 0.0, 2.25
def run(base, mode, wfroz=None, yfroz=None, clip=None, seed=0, hetero=0.0):
    g = np.random.default_rng(seed)
    F = g.normal(0.6, 1.0, size=B)
    a = np.exp(g.normal(0, hetero, size=B)); a /= a.mean()
    ws = []
    for t in ts:
        kap = base*(1-t)/t
        Fb, V = F.mean(), F.var(ddof=1)
        e = (V-tau2)/tau2
        w = 1.0+eta*e
        if mode == "bdg":
            num = (y-F) - eta*e*(F-Fb)
        elif mode == "frozen_w_y":            # plug at w_eff_bar, aiming at y
            num = wfroz*(y-F)
        elif mode == "frozen_w_yeff":         # plug at w_eff_bar, aiming at y_eff_bar
            num = wfroz*(yfroz-F)
        step = kap*a*num
        if clip is not None:
            step = np.clip(step, -clip, clip)
        F = F + step
        ws.append(w)
    return F, np.array(ws)

for hetero, clip, lab in ((0.0, None, "linear, homogeneous, no clip"),
                          (0.0, 0.02, "clip at 0.02"),
                          (0.8, None, "heterogeneous a (sd 0.8 in log)"),
                          (0.8, 0.02, "heterogeneous + clip")):
    F0, ws = run(0.30, "bdg", clip=clip, hetero=hetero)
    wbar_t = ws.mean()                                   # unweighted time-average
    kaps = 0.30*(1-ts)/ts
    wbar_k = float((ws*kaps).sum()/kaps.sum())           # gain-weighted average
    # y_eff time-average uses Fbar ~ 0 here, so y_eff ~ y/w
    F1,_ = run(0.30, "frozen_w_y", wfroz=wbar_t, clip=clip, hetero=hetero)
    F2,_ = run(0.30, "frozen_w_y", wfroz=wbar_k, clip=clip, hetero=hetero)
    base_sd = np.random.default_rng(0).normal(0.6,1.0,size=B).std(ddof=1)
    print("  %-32s w_eff: start %+.3f end %+.3f | time-avg %+.3f gain-wtd-avg %+.3f"
          % (lab, ws[0], ws[-1], wbar_t, wbar_k))
    print("     sd/sd0:  BDG %.4f | frozen(time-avg) %.4f | frozen(gain-wtd) %.4f"
          % (F0.std(ddof=1)/base_sd, F1.std(ddof=1)/base_sd, F2.std(ddof=1)/base_sd))
    print("     mean:    BDG %+.4f | frozen(time-avg) %+.4f  <- the TARGET differs"
          % (F0.mean(), F1.mean()))
print("\n  In the linear no-clip row the frozen-weight run reproduces BDG's spread")
print("  to within the amount w_eff moved. The handoff reports mu 2.53x vs 1.17x,")
print("  a 6.0x gap in log-widening (ln2.53/ln1.168 = %.2f). That cannot come from"
      % (np.log(2.53)/np.log(1.168)))
print("  feedback alone in the linear regime; it needs a nonlinearity (clip/mean")
print("  drag) OR the control froze w_eff while still aiming at y instead of y_eff.")
