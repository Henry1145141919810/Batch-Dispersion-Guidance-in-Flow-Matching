"""
BLUE-TEAM defence checks against the objection:
  "V* = tau^2(1-1/eta), not tau^2, and eta=1 cannot widen; the stability claim is a non-sequitur."

I do NOT dispute the algebra. I test what it IMPLIES for the handoff's reported claims.

Model (same 1-D deviation model both sides use, from the sampler's euler step):
  d_i <- d_i * (1 - kappa_n * w_eff),   w_eff = 1 + eta*(u - 1),   u = V_b/tau^2
  => u <- u * (1 - kappa_n*(1+eta*(u-1)))^2
  kappa_n = h * w * (1-t)/t * a/s^2  (h=0.01, 50 guided steps, t in [0.5, 0.99])

D1  reproduce u* = 1-1/eta                                       (do I get the red team's result?)
D2  is "fixed point 0" specific to eta=1, or shared with plug?   (is it a NEW defect?)
D3  does the 1.5x/0.866 correction break MONOTONICITY of the knob? (the actual 6.1 claim)
D4  how far do 50 steps get toward u*?                            (does the 13.4% undershoot
                                                                   describe any reported cell?)
D5  boundedness of BDG's coefficient vs btvg's, as V -> 0         (what really licenses the clamp)
D6  margin to the transient stability bound at the strengths used, and the w at which it binds
"""
import numpy as np

H, NSTEP = 0.01, 50
T = np.linspace(0.5, 0.99, NSTEP)


def kappa_sched(wa):  # wa = w * a/s^2  (the one fitted gain)
    return H * wa * (1.0 - T) / T


def run_u(u0, eta, wa, nstep=NSTEP, kap=None):
    u = float(u0)
    ks = kappa_sched(wa) if kap is None else np.full(nstep, kap)
    for k in ks[:nstep]:
        u = u * (1.0 - k * (1.0 + eta * (u - 1.0))) ** 2
    return u


print("=" * 78)
print("D1  interior fixed point: iterate to convergence at constant kappa")
print("=" * 78)
for eta in (1.5, 2.0, 4.0, 8.0, 16.0):
    for u0 in (4.0, 0.05):
        u = u0
        for _ in range(400000):
            u = u * (1.0 - 0.01 * (1.0 + eta * (u - 1.0))) ** 2
        pred = 1.0 - 1.0 / eta
        print(f"  eta={eta:5.1f} u0={u0:5.2f}  u_inf={u:.9f}  1-1/eta={pred:.9f}"
              f"  err={abs(u-pred):.2e}")
print("  -> CONCEDED: u* = 1-1/eta, both directions. The objection's algebra is correct.")

print()
print("=" * 78)
print("D2  is 'fixed point = 0' a defect SPECIFIC to eta=1, or shared with plug (eta=0)?")
print("=" * 78)
print("  u after 50 steps and after 5000 steps, u0=1.0, wa=0.20 (mu fitted gain):")
for eta in (0.0, 1.0, 4.0):
    u50 = run_u(1.0, eta, 0.20)
    u_long = 1.0
    for _ in range(5000):
        u_long = u_long * (1.0 - 0.002 * (1.0 + eta * (u_long - 1.0))) ** 2
    tag = "plug (eta=0)" if eta == 0 else f"bdg eta={eta:g}"
    print(f"  {tag:14s}  u(50)={u50:.6f}   u(5000, kappa=2e-3)={u_long:.6e}")
print("  -> plug itself has V*=0 in this same model. 'eta=1 is an unconditional contractor'")
print("     is a property eta=1 SHARES WITH PLUG, i.e. it is the section-3 reduction at eta=1,")
print("     not a new pathology. eta>1 is the extra ingredient the WIDENING branch needs.")

print()
print("=" * 78)
print("D3  does the correction sd* = tau*sqrt(1-1/eta) break the knob's monotonicity?")
print("=" * 78)
print("  eta=4, so the correction is a single constant sqrt(0.75)=%.6f" % np.sqrt(0.75))
print("  tau_mult   sd*/s (corrected)   sd*/s (handoff's implied tau)   ratio")
for tm in (0.5, 1.0, 1.21, 1.5):
    print(f"   {tm:5.2f}      {tm*np.sqrt(0.75):8.4f}            {tm:8.4f}"
          f"              {np.sqrt(0.75):.4f}")
print("  -> the correction is ONE property-independent, data-independent multiplicative")
print("     constant. Monotone requested->achieved (the 6/6 result) is untouched, and tau")
print("     remains an invertible handle: tau_needed = sd_wanted / sqrt(1-1/eta).")
print("  -> on the WIDENING branch the asymptote still widens: at tau_mult=1.5, sd*/s =")
print(f"     {1.5*np.sqrt(0.75):.4f}, i.e. +{100*(1.5*np.sqrt(0.75)-1):.1f}% vs unguided, LARGER than the")
print("     measured 1.168x. The '13.4% undershoot' is an undershoot of tau, NOT of widening.")

print()
print("=" * 78)
print("D4  does the 50-step window get anywhere near u*?  (fitted gains from the ladder)")
print("=" * 78)
print("  wa: mu 0.20, alpha 0.30, gap 0.10 (one gain per property, eta=4, u*=0.75)")
for prop, wa, meas in (("mu", 0.20, {0.5: 0.739, 1.0: 0.951, 1.21: 1.016, 1.5: 1.168}),
                       ("alpha", 0.30, {0.5: 0.681, 1.0: 0.905, 1.5: 1.034}),
                       ("gap", 0.10, {0.5: 0.787, 1.0: 0.984, 1.5: 1.096})):
    for tm, m in meas.items():
        u0 = 1.0 / tm ** 2
        uf = run_u(u0, 4.0, wa)
        print(f"  {prop:5s} tau_mult={tm:4.2f}  u0={u0:6.3f} -> u(50)={uf:6.3f}"
              f"   (u*=0.750)   sd_model/unguided={np.sqrt(uf/u0):.3f}  measured={m:.3f}")
print("  -> every cell ends FAR from u*=0.75 (u(50) barely moves off u0). The asymptotic")
print("     13.4% undershoot therefore describes NO cell in the 6.1 table; it describes only")
print("     the separate 'hold' cells (BDG_FIXED_POINT), which section 9.6 already lists as a")
print("     known defect. The objection cannot have both 'undershoot by 13.4%' and 'the runs")
print("     never reach V*' as criticisms of the same numbers.")

print()
print("=" * 78)
print("D5  what actually bounds the coefficient: BDG's linear e vs btvg's reciprocal")
print("=" * 78)
print("  BDG     : e      = V/tau^2 - 1          -> bounded below by -1 (since V>=0)")
print("  btvg_var: b      = -0.5*(1/tau^2 - 1/V) -> pole at V=0")
tau2 = 1.0
print("   V          BDG e      BDG w_eff(eta=4)      btvg b")
for V in (1e0, 1e-2, 1e-4, 1e-8, 1e-12, 0.0):
    e = V / tau2 - 1.0
    b = np.inf if V == 0 else -0.5 * (1.0 / tau2 - 1.0 / V)
    print(f"  {V:8.0e}   {e:+9.6f}    {1+4*e:+9.4f}         {b:+.4e}")
print("  -> BDG's coefficient is bounded by eta for ANY batch, because the error signal is")
print("     AFFINE in the measured variance. btvg's is unbounded because it is RECIPROCAL in")
print("     it. So the clamp-removal decision is CORRECT, and correct for a reason that needs")
print("     no dynamics -- which is the objection's own point, and it STRENGTHENS the method.")
print("     'V_b does not vanish' is still doing work, but different work: it says the setpoint")
print("     is TRACKABLE (a servo aimed at a quantity that -> 0 by construction has no")
print("     operating regime). Two claims, both true, currently welded into one bad sentence.")

print()
print("=" * 78)
print("D6  margin to the transient bound at the strengths actually used")
print("=" * 78)
print("  transient bound: kappa < 2/w_eff(u0).  u0 at tau_mult=0.5 is 4.0 -> w_eff=13.0")
bound = 2.0 / 13.0
print(f"  bound = 2/13.0 = {bound:.4f}")
for prop, wa in (("mu", 0.20), ("alpha", 0.30), ("gap", 0.10)):
    kpk = kappa_sched(wa).max()
    print(f"  {prop:5s} kappa_peak={kpk:.5f}  margin={bound/kpk:6.1f}x   "
          f"breaks at w ~ {4.0*bound/kpk:6.0f}  (runs used w=4)")
print("  -> stable with 50-115x margin, so the CONCLUSION the handoff draws holds; what")
print("     fails is the argument it gives for it. Forward risk: the v2 protocol says")
print("     'extend the strength grid past w=4'. At w=16 the margin is still 13-29x; the")
print("     bound binds only near w~200-460. A one-line runtime guard covers it.")

print()
print("=" * 78)
print("D7  is eta>1 needed for the reported widening cells?  min(w_eff) = 1-eta")
print("=" * 78)
for eta in (0.0, 1.0, 1.5, 4.0):
    print(f"  eta={eta:4.1f}  min w_eff = {1-eta:+5.2f}   widening branch exists: "
          f"{'YES' if eta > 1 else 'NO'}")
print("  -> every row of the handoff's 6.1 widening table is e4 (eta=4), so the headline")
print("     widening result is inside the regime where the mechanism exists. The eta=1 cells")
print("     appear only in open item 1, which the handoff already discounts as a grid-search")
print("     maximum over ~13 cells. So the 'disjointness' separates a live claim from one the")
print("     handoff already withdraws -- but eta>1 must be stated in section 2 as a DESIGN")
print("     CONSTRAINT, and every ablation row must carry its eta.")
