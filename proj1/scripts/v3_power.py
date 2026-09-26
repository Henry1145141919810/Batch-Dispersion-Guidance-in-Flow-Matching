"""What cell size does protocol v3 need? The table the operator chooses n against.

    python proj1/scripts/v3_power.py
    python proj1/scripts/v3_power.py --md-out docs/results/V3_POWER.md

WHY THIS EXISTS. v3 deliberately pre-registers no n (Henry, 26 Sep): the
operator picks the cell size for the cluster they are on. That makes n a
POWER decision, not just a budget one, and nothing else in the repo says what
each choice buys. This script is that statement, and it is generated rather
than written out by hand so it cannot drift from the protocol's own constants.

WHAT IT ASSUMES, and every one of these is stated in the protocol:

  * The headline metric is `in_band_fraction`, a PROPORTION over n samples, so
    its standard error is the binomial sqrt(p(1-p)/n). Section 4 pre-registers
    "significance is the binomial standard error alone" -- no run-to-run
    correction -- because the counting metrics were measured bit-stable over
    three identical runs (V3_REPRO_ENVELOPE.md).

  * Three seeds pool into 3n samples per (arm, property). That is the number
    every published figure rests on.

  * A CONTRAST between two arms is what carries a claim, and it is reported
    UNPAIRED: se = sqrt(se_a^2 + se_b^2). Arms within a cell do share initial
    noise, so a paired test would be tighter -- roughly by sqrt(2) -- which
    makes every number here CONSERVATIVE. It can miss a real difference; it
    cannot manufacture one. Do not quote these as paired.

  * MULTIPLICITY. The headline runs 6 guided arms against `unguided` on 3
    properties = 18 contrasts, so the threshold is Bonferroni at alpha/18,
    two-sided. `unguided` is not counted against itself. The ablation's grid
    is a max-over-17 selection and needs its own, wider threshold; --arms and
    --props move both.

WHAT IT DOES NOT MODEL. Per-task fixed cost (checkpoint load, the calibration
pass) does not scale with n, so the GPU-hour column is a floor, not a forecast
-- read the real figure off the [timing] line. And p is a stand-in: in_band at
w = 1 was indicatively 0.07-0.10 for most arms but 0.38 for tfg, where the se
is larger. Pass --p to see any of them.
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

COMPARE_ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg")

# the rate the budget is costed at: v2_run.slurm's measured 58 min for a
# 23.5-unit task at n = 5000, on OUR base. Units are DEFINED at n = 5000 and
# taken to scale linearly in n. EquiFM's per-pass ratio is NOT measured; 1.83x
# is borrowed from the repo's one timed external backend (TFG/EDMsecond).
MIN_PER_UNIT = 2.468
UNITS_AT_5000 = {"compare_set": 23.5, "per_bdg_arm": 3.5}
N_REF = 5000
EQUIFM_RATIO = 1.83


def norm_ppf(q):
    """Inverse normal CDF (Acklam), so this file needs no scipy."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    if q > ph:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) / \
               ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1)
    r = q - 0.5
    s = r * r
    return (((((a[0] * s + a[1]) * s + a[2]) * s + a[3]) * s + a[4]) * s + a[5]) * r / \
           (((((b[0] * s + b[1]) * s + b[2]) * s + b[3]) * s + b[4]) * s + 1)


def se_cell(p, n):
    return math.sqrt(p * (1 - p) / n)


def se_pooled(p, n, seeds):
    return math.sqrt(p * (1 - p) / (n * seeds))


def se_contrast(p_a, p_b, n, seeds):
    return math.sqrt(p_a * (1 - p_a) / (n * seeds)
                     + p_b * (1 - p_b) / (n * seeds))


def stage_units(arms, n, comp_units, per_bdg):
    """n5000-equivalent units for one (property, seed, backend) task.

    The comparison set is costed as ONE bundle of `comp_units` because that is
    how it was measured (v2's 58 min covered all five arms together). A
    backend running only SOME of them -- the QM9 diffusion base runs unguided
    and plug -- is charged pro rata by arm count. That is crude and it
    OVERSTATES: unguided does no guidance at all and plug is the cheapest
    guided arm, so two-fifths of the bundle is more than their real share.
    Erring high is the right direction for a budget.
    """
    comp = [a for a in arms if a in COMPARE_ARMS]
    bdg = [a for a in arms if a.startswith("bdg_")]
    units = comp_units * (len(comp) / float(len(COMPARE_ARMS)))         + per_bdg * len(bdg)
    return units * (n / float(N_REF))


def gpu_hours(n, stage_arms, props, seeds, registry, ratio):
    """GPU-hours for one stage, summed over the backends that run it.

    Each backend is charged for the arms IT runs (the registry), not for the
    stage's full set, and every backend after the first at the borrowed
    `ratio`. `fm` is the one the rate was measured on, so it is charged at 1x
    and everything else at `ratio`.
    """
    total_min = 0.0
    for be, cfg in sorted(registry.items()):
        allow = cfg.get("arms")
        arms = list(stage_arms) if allow is None else             [a for a in stage_arms if a in allow]
        if not arms:
            continue                       # this backend sits this stage out
        u = stage_units(arms, n, UNITS_AT_5000["compare_set"],
                        UNITS_AT_5000["per_bdg_arm"])
        rate = 1.0 if be == "fm" else ratio
        total_min += u * MIN_PER_UNIT * props * seeds * rate
    return total_min / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.09,
                    help="in_band fraction to size against (default 0.09, the "
                         "indicative value for most arms at w=1; tfg was 0.38)")
    ap.add_argument("--p-ref", type=float, default=None,
                    help="the arm the contrast is against (default: same as --p)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="default: the protocol's own seed count")
    ap.add_argument("--props", type=int, default=3)
    ap.add_argument("--arms", type=int, default=None,
                    help="guided arms contrasted against unguided (default: "
                         "the headline's own count minus unguided)")
    # no --backends count: which base models run, and which arms each runs,
    # is the registry's to say. Overstating it was how the ablation's cost
    # came out larger than the headline's.
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--ns", default="500,1000,2000,3000,5000,10000")
    ap.add_argument("--md-out", default="")
    a = ap.parse_args()

    import transfer_sweep as T
    registry = T.V3_BACKENDS
    seeds = a.seeds if a.seeds else len(T.V3_SEEDS)
    head = list(T.v3_arms())
    abl = list(T.v3_arms(etas=T.V3_ABL_ETAS, tau_mults=T.V3_ABL_TAU_MULTS,
                         compare=False))
    n_contrasts = (a.arms if a.arms else len(head) - 1) * a.props
    z = norm_ppf(1 - a.alpha / n_contrasts / 2.0)
    p, p_ref = a.p, (a.p_ref if a.p_ref is not None else a.p)
    ns = [int(x) for x in a.ns.split(",") if x]

    L = []
    L.append("# Protocol v3: what each cell size buys")
    L.append("")
    L.append("Generated by `proj1/scripts/v3_power.py`. Do not hand-edit; re-run it.")
    L.append("")
    L.append("**v3 pre-registers no `n`.** The operator picks it. This is the "
             "table to pick against: `n` sets the run's POWER, not only its "
             "cost, and the effect BDG is expected to produce is small.")
    L.append("")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append("| in_band sized against | p = %.3f (contrast vs p = %.3f) |" % (p, p_ref))
    L.append("| seeds | %d, pooling to 3n per (arm, property) |" % seeds)
    L.append("| headline arms | %d (%s) |" % (len(head), ", ".join(head)))
    L.append("| ablation arms | %d |" % len(abl))
    L.append("| contrasts | %d = %d guided arms x %d properties |"
             % (n_contrasts, n_contrasts // a.props, a.props))
    L.append("| threshold | Bonferroni alpha=%.3f/%d, two-sided: **z = %.3f** |"
             % (a.alpha, n_contrasts, z))
    L.append("| base models | %d: %s |"
             % (len(registry),
                ", ".join("%s (%s pair%s)"
                          % (b, c["pair"],
                             "" if c["arms"] is None
                             else ", %s only" % "+".join(c["arms"]))
                          for b, c in sorted(registry.items()))))
    L.append("")
    L.append("All figures in percentage points of in_band.")
    L.append("")
    L.append("| n per cell | se, one cell | se, pooled (%dn) | se, contrast | "
             "**min. detectable difference** | headline GPU-h | ablation GPU-h |"
             % seeds)
    L.append("|---|---|---|---|---|---|---|")
    for n in ns:
        sc = se_cell(p, n) * 100
        sp = se_pooled(p, n, seeds) * 100
        sx = se_contrast(p, p_ref, n, seeds) * 100
        mdd = z * sx
        gh = gpu_hours(n, head, a.props, seeds, registry, EQUIFM_RATIO)
        ga = gpu_hours(n, abl, a.props, seeds, registry, EQUIFM_RATIO)
        L.append("| %d | %.3f | %.3f | %.3f | **%.2f** | %.1f | %.1f |"
                 % (n, sc, sp, sx, mdd, gh, ga))
    L.append("")
    L.append("**How to read the last column pair.** GPU-hours are costed from "
             "`v2_run.slurm`'s one measured rate (58 min for 23.5 units at "
             "n = %d on our base) scaled linearly in n. `fm` is charged at "
             "that rate; every other base model at the **borrowed** %.2fx "
             "ratio, which is not measured for EquiFM and is a stand-in. Each "
             "backend is charged only for the arms it runs, so the QM9 "
             "diffusion base costs a fraction of the others in the headline "
             "and nothing at all in the ablation. Per-task fixed cost does "
             "not scale with n, so these are a **floor**; read the real "
             "number off the `[timing]` line once the first tasks land."
             % (N_REF, EQUIFM_RATIO))
    L.append("")
    L.append("**The number that should decide it** is the minimum detectable "
             "difference. `BDG_REVIEW.md` found no floor-clearing BDG cell "
             "beating `plug` over 72 paired tests, and "
             "FULL_RUN_V3_PROTOCOL.md section 6 pre-registers a null as the "
             "likely outcome. A null is only evidence against BDG if the run "
             "could have seen the effect: choose n so the MDD sits **below** "
             "the difference you would care about, and if that is not "
             "affordable, say so in the write-up rather than reporting \"no "
             "difference\" flat.")
    L.append("")
    L.append("The ablation's grid is a **max-over-%d selection**, so its own "
             "threshold is wider than the z above and its MDD correspondingly "
             "larger. Re-run with `--arms %d` to size against it."
             % (len(abl), len(abl)))
    out = "\n".join(L) + "\n"
    print(out)
    if a.md_out:
        path = a.md_out if os.path.isabs(a.md_out) else os.path.join(ROOT, a.md_out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
        print("wrote %s" % path)


if __name__ == "__main__":
    main()
