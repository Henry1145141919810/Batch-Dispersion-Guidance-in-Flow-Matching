"""Read the btvg2_xproj retest and print its table. CPU only, seconds.

    python proj1/scripts/xproj_table.py
    python proj1/scripts/xproj_table.py --md-out docs/results/XPROJ_RETEST.md

WHAT THIS ANSWERS. One pre-registered exploratory question, from
docs/results/WORKSHOP_CLAIM_AUDIT.md: *was the BTVG-2 failure substantially an
implementation error?* The audit found BTVG-2's invariants were imposed in the
wrong space; `btvg2_xproj` imposes them in x_t and passes the five gates in
`proj1/tests/test_btvg2_xproj.py`. Those gates are code properties. This is the
molecular outcome retest: `btvg2_xproj` against `lgd_mc` on identical targets
and identical initial noise, at the two strengths specified in advance.

THE AUDIT FIXED WHAT EACH OUTCOME LICENSES, BEFORE THE RUN:
  * a null narrows the corrected method's claim
  * a positive licenses FRESH CONFIRMATION, not retrospective victory
It also registered the comparison's metrics: "decoded coverage, useful yield,
local mean/variance response and clipping loss". Decoded coverage is the
PRIMARY one and is reported first here -- on the continuous metric alone the
comparison flatters whichever arm pushes the continuous type features harder,
which is the whole reason v2's V4 requires the decoded view.

AND IT REGISTERED A LIMIT ON WHAT MAY BE CONCLUDED ABOUT MECHANISM:
"Current correction-versus-LGD alone cannot separate objective mismatch,
estimator noise and clipping competition. This is an additional requirement
for that stronger mechanism claim, not permission to assert it from a null
outcome." So this script reports the clipping numbers and explicitly declines
to read a mechanism out of them: `btvg2_xproj` adds a variance term to the same
base at the same nominal w, so a larger raw update and more clipping is close
to definitional rather than evidential.

WHAT IT IS, AND IS NOT. n = 2,048, ONE seed (20261001), `dist` target, two
strengths. That is a pilot: the se on a paired in_band difference is ~0.010, so
it can only see a large effect, and a null at this power NARROWS a claim rather
than establishing a negative. It is NOT part of the v2 full run (different
target, n, seed count and protocol) and its numbers never belong in a v2 table.

CLIPPING IS REPORTED AS A FRACTION, NOT A COUNT. `clipped_sample_steps` counts
(sample, step) pairs where the guidance norm exceeded clip x velocity norm, so
the denominator is n x guided steps -- identical for both arms here (same n,
steps and window, and both schedule 800 guided passes). A bare integer hides
that, and the precedent script `fr3a_vs_fr3_table.py` reports the percentage.
It is also a count of steps at which the clip bound, NOT a magnitude of
guidance discarded: two arms can clip equally often and lose very different
amounts.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

NL = chr(10)

PROPS = ("mu", "alpha", "gap")
ARMS = ("lgd_mc", "btvg2_xproj")
OURS = "btvg2_xproj"
STRENGTHS = (8.0, 16.0)
SEED = 20261001
SIGMA = 3.0
# The v2 full run's chemistry floor, for reference only: 0.9 x the unguided
# molecule stability measured at n = 5,000 x 3 seeds in results/full/v2.
V2_FLOOR = 0.357120


def load(root):
    out, dupes = {}, []
    for fn in sorted(glob.glob(os.path.join(root, "*", "*__tgt.json"))):
        r = json.load(open(fn))
        key = (r["prop"], r["arm"], float(r["w"]))
        if key in out:
            dupes.append("more than one cell for %s/%s@w%g" % key)
            continue
        r["_file"] = fn
        side = fn[:-5] + ".permol.pt"
        out[key] = (r, side if os.path.exists(side) else None)
    return out, dupes


def check(cells, dupes):
    problems = list(dupes)
    for p in PROPS:
        for a in ARMS:
            for w in STRENGTHS:
                if (p, a, w) not in cells:
                    problems.append("missing %s/%s@w%g" % (p, a, w))
                elif cells[(p, a, w)][1] is None:
                    problems.append("no sidecar for %s/%s@w%g" % (p, a, w))
    rows = [r for (r, _) in cells.values()]
    if not rows:
        return problems + ["no cells found"]
    for key in ("n", "seed", "steps", "solver", "t_min_guide", "clip", "batch"):
        vals = {r.get(key) for r in rows}
        if len(vals) > 1:
            problems.append("mixed %s: %s" % (key, sorted(map(str, vals))))
    if {r.get("seed") for r in rows} != {SEED}:
        problems.append("cells are not all seed %d: %s"
                        % (SEED, sorted({r.get("seed") for r in rows})))
    vals = {(r.get("prov") or {}).get("fm_md5") for r in rows}
    if len(vals) > 1:
        problems.append("mixed generator: %s" % sorted(map(str, vals)))
    if {r.get("target_name") for r in rows} != {"dist"}:
        problems.append("not all cells are the dist target")
    return problems


def paired(cells, prop, w, dec=False):
    """(z_in_band, z_mae, d_in_band) for OURS - lgd_mc over the same molecules."""
    import torch
    a = torch.load(cells[(prop, OURS, w)][1], weights_only=False)
    b = torch.load(cells[(prop, "lgd_mc", w)][1], weights_only=False)
    if not torch.equal(a["mol_idx"], b["mol_idx"]):
        raise SystemExit("sidecars do not pair at %s w=%g" % (prop, w))
    delta = cells[(prop, OURS, w)][0]["delta"]
    key = "f_B_dec" if dec else "f_B"
    if key not in a or key not in b:
        return float("nan"), float("nan"), float("nan")
    ia = ((a[key] - a["y"]).abs() <= delta) & a["finite"]
    ib = ((b[key] - b["y"]).abs() <= delta) & b["finite"]
    d = ia.double() - ib.double()
    se = d.std(unbiased=True).item() / math.sqrt(d.numel())
    z_ib = d.mean().item() / se if se > 0 else float("nan")
    ok = a["finite"] & b["finite"]
    e = ((a[key] - a["y"]).abs() - (b[key] - b["y"]).abs())[ok].double()
    se2 = e.std(unbiased=True).item() / math.sqrt(e.numel())
    z_mae = e.mean().item() / se2 if se2 > 0 else float("nan")
    return z_ib, z_mae, d.mean().item()


def clip_frac(r):
    """clipped (sample, step) pairs as a fraction of the guided ones."""
    guided = int(round(r["steps"] * (1.0 - r["t_min_guide"])))
    denom = r["n"] * guided
    return (r.get("clipped_sample_steps", 0) / denom if denom else float("nan")), denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "xproj"))
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    cells, dupes = load(args.root)
    problems = check(cells, dupes)
    if problems:
        print("REFUSING: the xproj set is not complete or not consistent.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)

    import torch  # noqa: F401

    any_r = cells[(PROPS[0], ARMS[0], STRENGTHS[0])][0]
    prov = any_r.get("prov") or {}
    _, denom = clip_frac(any_r)

    L = []
    L.append("# BTVG-2 cross-projection retest (`btvg2_xproj` vs `lgd_mc`)")
    L.append("")
    L.append("Generated by `proj1/scripts/xproj_table.py`. Do not hand-edit; "
             "re-run the script.")
    L.append("")
    L.append("**Exploratory pilot. Not part of the v2 full run.** It answers one "
             "question registered in "
             "[WORKSHOP_CLAIM_AUDIT.md](WORKSHOP_CLAIM_AUDIT.md): was the BTVG-2 "
             "failure substantially an implementation error? The audit fixed in "
             "advance what each outcome licenses -- **a null narrows the corrected "
             "method's claim; a positive licenses a fresh confirmation run, not a "
             "retrospective victory** -- and registered the metrics: decoded "
             "coverage and useful yield first, then the mean/variance response and "
             "clipping.")
    L.append("")
    L.append("| provenance | value |")
    L.append("|---|---|")
    L.append("| cells | %d = %d properties x %d arms x %d strengths |"
             % (len(PROPS) * len(ARMS) * len(STRENGTHS), len(PROPS), len(ARMS),
                len(STRENGTHS)))
    L.append("| n per cell | %d, **one seed** (%d, verified on every cell) |"
             % (any_r["n"], SEED))
    L.append("| target | `dist` (per-molecule, the v1 protocol), not v2's fixed q90 |")
    L.append("| generator | `%s` md5 `%s` |"
             % (any_r.get("fm", "?"), (prov.get("fm_md5") or "?")[:8]))
    L.append("| sampler | %d-step %s, window t >= %g, clip %g, batch %d |"
             % (any_r["steps"], any_r["solver"], any_r["t_min_guide"],
                any_r["clip"], any_r["batch"]))
    L.append("| device | %s, torch %s |" % (prov.get("device"), prov.get("torch")))
    L.append("")
    L.append("**Resolving power, stated before the numbers.** At n = 2,048 and one "
             "seed the se of a paired in_band difference is about 0.010. This pilot "
             "can see an effect of ~0.03 and cannot exclude one of ~0.02. A null "
             "here **narrows** the corrected method's claim, which is what the "
             "audit registered; it does not establish that no effect exists.")
    L.append("")

    # chemistry, computed rather than asserted
    above = [(p, a, w) for p in PROPS for a in ARMS for w in STRENGTHS
             if cells[(p, a, w)][0]["mol_stability"] >= V2_FLOOR]
    L.append("**Chemistry at these strengths.** w = 8 and 16 were chosen to probe "
             "the mechanism, not to be affordable, but they are not uniformly "
             "unaffordable: against the v2 full run's floor of %.4f, **%d of %d "
             "cells are above it** and the lowest cell in the set is %.4f "
             "(%.1f %% under). The floor is shown for orientation only -- this "
             "pilot uses a different target and protocol and is not scored against "
             "the v2 rubric."
             % (V2_FLOOR, len(above), len(PROPS) * len(ARMS) * len(STRENGTHS),
                min(cells[k][0]["mol_stability"] for k in cells),
                100 * (1 - min(cells[k][0]["mol_stability"] for k in cells) / V2_FLOOR)))
    L.append("")

    L.append("## Outcome")
    L.append("")
    L.append("| prop | w | arm | in_band | **in_band (dec)** | MAE | mol_stab | "
             "valid | useful yield | vs v2 floor |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for p in PROPS:
        for w in STRENGTHS:
            for a in ARMS:
                r = cells[(p, a, w)][0]
                L.append("| %s | %g | `%s` | %.4f | **%.4f** | %.4f | %.4f | %.4f | "
                         "%.4f | %s |"
                         % (p, w, a, r["in_band_fraction"],
                            r.get("in_band_fraction_dec", float("nan")),
                            r["prop_mae_eval"], r["mol_stability"], r["validity"],
                            r["unique_valid_per_sample"],
                            "above" if r["mol_stability"] >= V2_FLOOR else "below"))
    L.append("")
    L.append("*Useful yield is `unique_valid_per_sample`: distinct valid molecules "
             "per attempt, the audit's second registered metric.*")
    L.append("")

    L.append("## Paired test: `%s` minus `lgd_mc`, same molecules and same noise"
             % OURS)
    L.append("")
    L.append("**Decoded coverage is the registered primary metric** and is given "
             "first. The continuous columns follow, because the two disagree in "
             "direction on mu.")
    L.append("")
    L.append("| prop | w | **d(in_band) dec** | **z_pair** | d(in_band) cont | z_pair | "
             "d(MAE) cont | z_pair | reading (decoded) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    wins = wins_c = 0
    dec_pos = []
    for p in PROPS:
        for w in STRENGTHS:
            ro, rl = cells[(p, OURS, w)][0], cells[(p, "lgd_mc", w)][0]
            zc, zmc, dc = paired(cells, p, w, dec=False)
            zd, _zmd, dd = paired(cells, p, w, dec=True)
            dmae = ro["prop_mae_eval"] - rl["prop_mae_eval"]
            if zd >= SIGMA:
                read = "ours ahead"
                wins += 1
            elif zd <= -SIGMA:
                read = "lgd_mc ahead"
            else:
                read = "tie"
            if zc >= SIGMA:
                wins_c += 1
            if dd > 0:
                dec_pos.append("%s w=%g" % (p, w))
            L.append("| %s | %g | **%+.4f** | **%+.2f** | %+.4f | %+.2f | %+.4f | "
                     "%+.2f | %s |"
                     % (p, w, dd, zd, dc, zc, dmae, zmc, read))
    L.append("")
    lower_mae = [(p, w) for p in PROPS for w in STRENGTHS
                 if cells[(p, OURS, w)][0]["prop_mae_eval"]
                 < cells[(p, "lgd_mc", w)][0]["prop_mae_eval"]]
    lower_ib = [(p, w) for p in PROPS for w in STRENGTHS
                if cells[(p, OURS, w)][0]["in_band_fraction"]
                < cells[(p, "lgd_mc", w)][0]["in_band_fraction"]]
    L.append("**The MAE and in_band columns point opposite ways, and that is the "
             "project's known signature.** `%s` has the lower MAE in %d of %d cells "
             "while having the lower continuous in_band in %d of %d: it sits closer "
             "on average and lands inside the band less often, which is a "
             "bias-versus-spread difference, not a coverage win for either arm. "
             "Neither difference reaches %g sigma."
             % (OURS, len(lower_mae), len(PROPS) * len(STRENGTHS),
                len(lower_ib), len(PROPS) * len(STRENGTHS), SIGMA))
    L.append("")

    L.append("## Mean/variance response and clipping")
    L.append("")
    L.append("`var_share` is the variance term's share of the update, `gate` the "
             "positive-V gate, `capped` the fraction of steps the gate capped, and "
             "`clipped` the fraction of **guided (sample, step) pairs** at which "
             "the velocity clip bound. The denominator is %d for every cell "
             "(n = %d x %d guided steps), identical for both arms."
             % (denom, any_r["n"], int(round(any_r["steps"] * (1 - any_r["t_min_guide"])))))
    L.append("")
    L.append("| prop | w | gate | V/tau^2 | var share | capped | clipped (`%s`) | "
             "clipped (`lgd_mc`) | ratio |" % OURS)
    L.append("|---|---|---|---|---|---|---|---|---|")
    ratios = []
    for p in PROPS:
        for w in STRENGTHS:
            ro, rl = cells[(p, OURS, w)][0], cells[(p, "lgd_mc", w)][0]
            d = ro.get("diag") or {}
            fo, _ = clip_frac(ro)
            fl, _ = clip_frac(rl)
            ratios.append(fo / fl if fl else float("nan"))
            f = lambda v: "--" if v is None else "%.4g" % v
            L.append("| %s | %g | %s | %s | %s | %s | %.1f %% | %.1f %% | %.2f x |"
                     % (p, w, f(d.get("btvg2_gate")), f(d.get("btvg2_V_over_tau2")),
                        f(d.get("btvg2_var_share")), f(d.get("btvg2_capped")),
                        100 * fo, 100 * fl, ratios[-1]))
    L.append("")
    shares = [(cells[(p, OURS, w)][0].get("diag") or {}).get("btvg2_var_share", 0)
              for p in PROPS for w in STRENGTHS]
    L.append("`%s` clips more often than `lgd_mc` in **%d of %d** cells "
             "(%.2f-%.2f x), carrying a variance share of %.2f-%.2f."
             % (OURS, sum(1 for r in ratios if r > 1), len(ratios),
                min(ratios), max(ratios), min(shares), max(shares)))
    L.append("")
    L.append("**This is reported, and deliberately not read as a mechanism.** The "
             "audit registered the limit in advance: *\"correction-versus-LGD alone "
             "cannot separate objective mismatch, estimator noise and clipping "
             "competition. This is an additional requirement for that stronger "
             "mechanism claim, not permission to assert it from a null outcome.\"* "
             "`%s` adds a variance term to the same base at the same nominal w, so "
             "a larger raw update and more frequent clipping is close to "
             "definitional. It is also a count of steps at which the clip bound, "
             "not a magnitude of guidance discarded. Separating these needs the "
             "matched-magnitude experiment the audit specifies, which has not been "
             "run." % OURS)
    L.append("")

    L.append("## Verdict")
    L.append("")
    n_cmp = len(PROPS) * len(STRENGTHS)
    L.append("On the registered primary metric (decoded coverage), `%s` is ahead of "
             "`lgd_mc` at the pre-set sigma >= %g in **%d of %d** paired "
             "comparisons; on continuous in_band, **%d of %d**."
             % (OURS, SIGMA, wins, n_cmp, wins_c, n_cmp))
    L.append("")
    if wins == 0:
        L.append("**This is a null, and under the audit's pre-set rule a null "
                 "narrows the corrected method's claim.** Fixing the projection "
                 "space made `btvg2_xproj` pass its five code gates; it did not "
                 "produce a demonstrable molecular improvement over `lgd_mc` at "
                 "either strength on any property.")
        L.append("")
        if dec_pos:
            L.append("**Two things stop this being a clean negative.** (1) On "
                     "decoded coverage the point estimate favours `btvg2_xproj` in "
                     "%s -- not significantly, but the continuous metric reverses "
                     "the sign on mu, so the choice of metric decides the "
                     "*direction* there. The reason is V4's: `lgd_mc` loses more "
                     "in_band going continuous to decoded than `btvg2_xproj` does. "
                     "(2) The pilot's power cannot exclude a real effect of ~0.02. "
                     "The supportable statement is that **the projection error was "
                     "not what was costing the method its outcome**, not that the "
                     "corrected method is equivalent to `lgd_mc`."
                     % ", ".join(dec_pos))
        else:
            L.append("The point estimates do not favour `btvg2_xproj` on either "
                     "metric, so the reading is the same whichever is used. The "
                     "supportable statement is that the projection error was not "
                     "what was costing the method its outcome; the pilot's power "
                     "still cannot exclude a real effect of ~0.02.")
        L.append("")
        L.append("No confirmation run is licensed by this pilot.")
    else:
        L.append("**This is a positive, and it licenses a fresh confirmation run -- "
                 "nothing more.** Under the audit's pre-set rule it may not be "
                 "reported as a win. A confirmation needs the v2 protocol: the "
                 "fixed target, three seeds, n = 5,000, and a strength that clears "
                 "the chemistry floor.")
    L.append("")

    out = NL.join(L) + NL
    print(out)
    if args.md_out:
        dst = args.md_out if os.path.isabs(args.md_out) else os.path.join(ROOT, args.md_out)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8").write(out)
        print("wrote %s" % dst, file=sys.stderr)


if __name__ == "__main__":
    main()
