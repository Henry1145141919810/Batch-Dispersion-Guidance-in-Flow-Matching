"""Read a protocol-v3 run and print its full metric block, per base model.

    python proj1/scripts/v3_table.py --backend fm
    python proj1/scripts/v3_table.py --backend equifm --md-out docs/results/V3_EQUIFM.md
    python proj1/scripts/v3_table.py --both --md-out docs/results/V3_RESULTS.md

WHAT v3 IS. docs/protocol/FULL_RUN_V3_PROTOCOL.md. Every arm at w = 1, the q50
target for every property, guidance from t = 0.5, three seeds, n = 2000, on both
base models. BDG (eta = 4, tau_mult in {0.5, 0.75, 1}) is the innovation arm; the
ablation extends it over eta and tau_mult in the same tree.

THIS SCRIPT REFUSES TO RANK ON in_band ALONE, and that refusal is the point.

v3 has NO chemistry floor (Henry, 26 Sep): nothing is excluded for poor
chemistry. But `w = 1` is far from some arms' floor-clearing strength -- tfg
froze at 0.01-0.05 under v2, so w = 1 is 20-100x it. Measured on our base at
q50, n = 512, tfg at w = 1 takes the HIGHEST in-band of any arm on all three
properties (mu 0.3828 against plug's 0.0840) at molecule stability 0.2285 and
validity 0.62, against unguided's 0.4023 and 0.75. An in-band leaderboard would
therefore crown the arm that destroyed the most chemistry.

So every table here prints in_band beside mol_stability, validity, uniqueness
and distinct-valid-per-attempt, and the summary names a "best in-band" only with
its chemistry attached and a warning when that arm's stability is below
unguided's. A verdict is left to the reader, deliberately.

WHAT IS COMPARABLE, AND WHAT IS NOT.
  * Within one backend, arms ARE paired: same seed, same batch size, same
    molecule sizes and the same fixed target, so a paired test WOULD be valid --
    but it is NOT what this script reports. `z2` below is unpaired:
    (a-b)/sqrt(se_a^2 + se_b^2). Because an arm and its control share initial
    noise, pairing roughly halves the se, so every z printed here is
    CONSERVATIVE -- it can miss a real difference, not manufacture one. The
    `--per-mol` sidecars written for all 396 cells are what a paired test would
    read; nothing reads them yet. BDG_REVIEW.md already ruled that the port's
    comparison "must be recomputed paired", so do that before publishing a BDG
    claim rather than quoting these z values as paired.
  * ACROSS backends nothing is paired. Our FM base and EquiFM have different
    architectures, different noise schedules (one clock against two) and
    different state scales, so molecule i of one is not molecule i of the other.
    Cross-backend rows are printed side by side and their differences are NOT
    tested. `--both` puts them in one page; it does not put them in one test.
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
# The base models, and the property pair each is scored with. Kept in step
# with transfer_sweep.V3_BACKENDS by a gate in test_v3.py -- this module must
# not import torch (the cluster runs it on a login node), so it restates the
# names rather than importing the registry at module scope.
BACKENDS = ("fm", "equifm", "edm")
BACKEND_LABEL = {"fm": "ours (flow-matching EGNN), our property pair",
                 "equifm": "EquiFM (borrowed), TFG's property pair",
                 "edm": "QM9 diffusion (TFG/EDMsecond), our property pair"}
SIGMA = 3.0
UNIQ_MIN = 0.95


def NORM_SF(z):
    """P(Z > z) for a standard normal, via erfc -- no scipy on the cluster."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def NORM_Q(p):
    """Inverse standard normal CDF by bisection: exact enough for a threshold
    and free of any dependency. p in (0, 1)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 1.0 - NORM_SF(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
# every cell of one v3 run must agree on these or it is not one experiment
# Every cell pooled into one table must agree on all of these. `pair` is here
# because delta = k x MAE(f_B): two cells scored with different property pairs
# have different BAND WIDTHS, so averaging their in_band would compare a wide
# band against a narrow one and call the difference a result.
SAME_KEYS = ("n", "steps", "solver", "batch", "target_name", "w", "t_start",
             "delta_mode", "pair")
# a cell's delta_mode must be a real rule, not None: see load()
KNOWN_DELTA_MODES = ("global", "local")


def w_eff_nominal(arm):
    """ORDERING FALLBACK ONLY: 1 + eta*(1/tau_mult^2 - 1).

    This closed form is WITHDRAWN as a prediction. It assumes V_b = y_std^2,
    which holds on the one synthetic x_t in test_bdg.py part B and not during
    sampling: bdg_ladder_probe.py measured V_b/tau^2 run-means of 64-257 and so
    w_eff of 111-1019 where this formula says -1.22 to 13.00 -- two orders of
    magnitude out. What the measurement DID confirm is that the rung ORDER is
    preserved, so this is still usable to order rows when no measured w_eff is
    available (an ablation-only tree, or cells written before the diagnostics).
    Never print it as a value. See FULL_RUN_V3_PROTOCOL.md section 4.1.
    """
    if not arm.startswith("bdg_e"):
        return None
    eta, tau = arm[len("bdg_e"):].split("t")
    eta, tau = float(eta), float(tau.rstrip("o"))
    return 1.0 + eta * (1.0 / (tau * tau) - 1.0)


def load(root, backend, n, seeds, stage="v3"):
    """{(prop, arm): [rows, one per seed]} plus problems.

    `stage` selects BOTH the directory level and the label a cell must carry.
    The two v3 stages run at the same n and the same seeds, so the stage is the
    only thing that tells their cells apart; reading one tree with the other's
    expectations is the mistake this argument exists to prevent.
    """
    got, problems = {}, []
    for seed in seeds:
        d = os.path.join(root, backend, stage, "n%d" % n, "seed%s" % seed)
        if not os.path.isdir(d):
            problems.append("%s: no directory for seed %s" % (backend, seed))
            continue
        for fn in sorted(glob.glob(os.path.join(d, "tr__*.json"))):
            r = json.load(open(fn))
            if r.get("stage") != stage:
                # the directory says one stage and the cell says another: a
                # stray file, not something to quietly average in
                problems.append("%s: %s is a %r cell under %s/"
                                % (backend, os.path.basename(fn),
                                   r.get("stage"), stage))
                continue
            r["_seed"], r["_file"] = seed, os.path.basename(fn)
            # transfer_sweep records delta_mode under `calibration`, NOT at the
            # top level. Reading it with r.get("delta_mode") returned None for
            # every cell, which made the SAME_KEYS agreement check below vacuous
            # (one value, None, always agrees) and printed "None rule" in the
            # header. delta is not in a cell's filename, so a run that mixed the
            # global and local rules would have been pooled silently.
            r["delta_mode"] = ((r.get("calibration") or {}).get("delta_mode")
                               or r.get("delta_mode"))
            # cells written before 26 Sep have no `pair` because every backend
            # used TFG's; label them so they group with the cells they match
            # rather than against a None that agrees with everything
            r["pair"] = (r.get("pair")
                         or (r.get("calibration") or {}).get("pair")
                         or "tfg")
            got.setdefault((r["prop"], r["arm"]), []).append(r)
    cells = {}
    for k, rows in sorted(got.items()):
        if len(rows) != len(seeds):
            problems.append("%s/%s: %d of %d seeds (%s)"
                            % (k[0], k[1], len(rows), len(seeds),
                               ", ".join(sorted(r["_seed"] for r in rows))))
            continue
        cells[k] = sorted(rows, key=lambda r: r["_seed"])
    return cells, problems


def check(cells, problems, stage="v3", backend=None):
    rows = [r for rs in cells.values() for r in rs]
    if not rows:
        return problems + ["no v3 cells found"]
    # A WHOLLY missing (property, arm) never reaches `cells`, so nothing above
    # notices it: the table simply omitted the row and exited 0, which reads as
    # "this arm was not planned" rather than "this arm did not run". The
    # headline's arm set is not negotiable, so it is checked against the planner
    # rather than against whatever happens to be on disk.
    # THE EXPECTED ARM SET IS PER STAGE. It used to be v3_arms() whatever tree
    # was loaded, so the ablation -- which deliberately carries no comparison
    # arm -- was ALWAYS reported as missing all five and the table refused on a
    # perfect run. The planner is still the source of truth; only which plan is
    # asked for changed.
    try:
        import transfer_sweep as T
        if stage == "v3abl":
            want = list(T.v3_arms(etas=T.V3_ABL_ETAS,
                                  tau_mults=T.V3_ABL_TAU_MULTS, compare=False))
        else:
            want = list(T.v3_arms())
        # AND PER BACKEND. The QM9 diffusion base is declared unguided+plug
        # only, so expecting the full set there would refuse a complete run.
        if backend is not None:
            want = T.v3_backend_arms(backend, want)
    except Exception as exc:                       # noqa: BLE001
        problems.append("cannot import transfer_sweep to learn the planned arms "
                        "(%s), so a missing arm cannot be detected" % exc)
        want = []
    missing = [(p, a) for p in PROPS for a in want if (p, a) not in cells]
    if missing:
        problems.append("%d planned %s cells absent entirely (not one "
                        "seed): %s" % (len(missing), stage,
                                       ", ".join("%s/%s" % m for m in missing[:8])
                                       + (" ..." if len(missing) > 8 else "")))
    # ablation arms are optional here (the ablation is a separate stage), but an
    # arm present for SOME properties and absent for others is a hole, not a plan
    seen = {a for (_p, a) in cells}
    for a in sorted(seen):
        gaps = [p for p in PROPS if (p, a) not in cells]
        if gaps and a not in want:
            problems.append("%s ran for %s but not %s"
                            % (a, ", ".join(p for p in PROPS if (p, a) in cells),
                               ", ".join(gaps)))
    bad_mode = {r.get("delta_mode") for r in rows} - set(KNOWN_DELTA_MODES)
    if bad_mode:
        problems.append("cells record delta_mode %s; expected one of %s"
                        % (sorted(map(str, bad_mode)), list(KNOWN_DELTA_MODES)))
    for key in SAME_KEYS:
        vals = {json.dumps(r.get(key)) for r in rows}
        if len(vals) > 1:
            problems.append("cells disagree on %s: %s" % (key, sorted(vals)))
    vals = {(r.get("prov") or {}).get("gen_md5") for r in rows}
    if len(vals) > 1:
        problems.append("cells used different generators: %s" % sorted(map(str, vals)))
    for p in PROPS:
        ds = {round(float(r["delta"]), 12) for r in rows if r["prop"] == p}
        if len(ds) > 1:
            problems.append("%s: cells scored at different deltas %s" % (p, sorted(ds)))
    for (p, a), rs in cells.items():
        for r in rs:
            if abs(float(r["w"]) - 1.0) > 1e-12:
                problems.append("%s/%s seed %s ran at w=%g; v3 fixes w = 1"
                                % (p, a, r["_seed"], float(r["w"])))
    return problems


def pooled(rows):
    N = sum(r["n"] for r in rows)
    wm = lambda k: sum(r[k] * r["n"] for r in rows) / N
    out = {"N": N, "seeds": len(rows)}
    for k in ("in_band_fraction", "in_band_fraction_dec", "prop_mae_eval",
              "prop_mae_eval_dec", "prop_rmse_eval", "mol_stability",
              "atom_stability", "validity", "uniqueness_of_valid",
              "unique_valid_per_sample", "diversity_mean_pairwise",
              "guide_eval_gap_mean"):
        out[k] = wm(k) if k in rows[0] else float("nan")
    out["nonfinite"] = sum(r.get("n_nonfinite", 0) for r in rows)
    out["uniq_min_cell"] = min(r["uniqueness_of_valid"] for r in rows)
    out["delta"] = rows[0]["delta"]
    out["bias"] = wm("f_B_mean") - wm("target_mean")
    out["sd"] = math.sqrt(max(out["prop_rmse_eval"] ** 2 - out["bias"] ** 2, 0.0))
    c = rows[0].get("cost") or {}
    out["guide_passes"] = sum(c.get(k, 0) for k in ("guide_fwd", "guide_bwd", "guide_hvp"))
    out["gen_passes"] = sum(c.get(k, 0) for k in ("gen_fwd", "gen_vjp", "gen_jvp"))
    out["clipped"] = sum(r.get("clipped_sample_steps", 0) for r in rows)
    d = rows[0].get("diag_mean") or rows[0].get("diag") or {}
    for k in ("bdg_e_raw", "bdg_V_over_tau2", "bdg_w_eff", "bdg_disp_rms",
              # w_eff changes sign within a window (measured: 62 % of guided
              # steps at tau_mult 1.5), so its signed run-mean alone is
              # unfalsifiable. These two are what make it readable.
              "bdg_w_eff_sq", "bdg_w_eff_neg"):
        out[k] = d.get(k)
    if out.get("bdg_w_eff_sq") is not None:
        out["bdg_w_eff_rms"] = math.sqrt(max(0.0, out["bdg_w_eff_sq"]))
    out["se_ib"] = math.sqrt(max(out["in_band_fraction"] * (1 - out["in_band_fraction"]), 0) / out["N"])
    out["se_ib_dec"] = math.sqrt(max(out["in_band_fraction_dec"] * (1 - out["in_band_fraction_dec"]), 0) / out["N"])
    return out


def z2(a, b, key, se):
    s = math.sqrt(a[se] ** 2 + b[se] ** 2)
    return (a[key] - b[key]) / s if s > 0 else float("nan")


def order_arms(arms, measured=None):
    """Comparison arms first, then BDG by DESCENDING w_eff.

    Not by (eta, tau_mult): tau_mult is not a monotone axis -- the deviation term
    reverses on 62 % of guided steps at tau_mult 1.5 -- so ordering by tau_mult
    invites a monotone reading of a ladder that folds. `measured` maps arm ->
    the run's own mean w_eff and is preferred; the nominal closed form is only
    the fallback when a cell has no controller state.
    """
    base = [a for a in ("unguided", "plug", "tmpd", "lgd_mc", "tfg") if a in arms]
    measured = measured or {}

    def key(a):
        m = measured.get(a)
        return (-(m if m is not None else w_eff_nominal(a)), a)

    bdg = sorted([a for a in arms if a.startswith("bdg_e")], key=key)
    rest = [a for a in arms if a not in base and a not in bdg]
    return base + bdg + sorted(rest)


def emit(L, backend, cells, n, seeds):
    P = {k: pooled(v) for k, v in cells.items()}
    # order BDG by the run's OWN mean w_eff (averaged over the properties that
    # have it), falling back to the nominal form only where it was not recorded
    meas = {}
    for a in {x for (_p, x) in cells}:
        vals = [P[(p, a)]["bdg_w_eff"] for p in PROPS
                if (p, a) in P and P[(p, a)].get("bdg_w_eff") is not None]
        if vals:
            meas[a] = sum(vals) / len(vals)
    arms = order_arms({a for (_p, a) in cells}, measured=meas)
    any_row = next(iter(cells.values()))[0]
    prov = any_row.get("prov") or {}
    L.append("## Base model: %s (`--backend %s`)"
             % (BACKEND_LABEL.get(backend, backend), backend))
    L.append("")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append("| generator | %s, md5 `%s` |" % (prov.get("gen", "?"),
                                              str(prov.get("gen_md5"))[:8]))
    L.append("| target | `%s`, fixed per property |" % any_row["target_name"])
    L.append("| strength | w = %g for every arm |" % float(any_row["w"]))
    L.append("| guidance window | t >= %g |" % float(any_row["t_start"]))
    L.append("| n, seeds | %d x %d = %d molecules per arm |"
             % (any_row["n"], len(seeds), any_row["n"] * len(seeds)))
    # delta per property, from whatever arm that property HAS. Indexing
    # P[(p, arms[0])] raised KeyError on a tree where the first arm was missing
    # for one property -- a traceback instead of the refusal check() exists to
    # print, and the refusal is the useful output.
    def any_delta(p):
        for a in arms:
            if (p, a) in P:
                return "%s %.5f" % (p, P[(p, a)]["delta"])
        return "%s MISSING" % p
    L.append("| delta | %s rule: %s |"
             % (any_row.get("delta_mode") or "UNRECORDED",
                ", ".join(any_delta(p) for p in PROPS)))
    L.append("| sampler | %d-step %s, batch %d |"
             % (any_row["steps"], any_row["solver"], any_row["batch"]))
    L.append("| chemistry floor | **none** -- nothing is excluded; chemistry is reported |")
    L.append("| non-finite | %d across all cells |" % sum(P[k]["nonfinite"] for k in P))
    L.append("")
    for p in PROPS:
        av = [a for a in arms if (p, a) in P]
        if not av:
            continue
        u = P.get((p, "unguided"))
        L.append("### %s (delta %.5f)" % (p, P[(p, av[0])]["delta"]))
        L.append("")
        L.append("| arm | in_band | +-se | dec | MAE/d | bias/d | sd/d | mol_stab | "
                 "atom_stab | valid | uniq (min) | yield | diversity | guide-eval gap/d |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        d = P[(p, av[0])]["delta"]
        for a in av:
            r = P[(p, a)]
            L.append("| `%s` | %.4f | %.4f | %.4f | %.3f | %+.3f | %.3f | %.4f | %.4f | "
                     "%.4f | %.4f (%.4f) | %.4f | %.4f | %.3f |"
                     % (a, r["in_band_fraction"], r["se_ib"], r["in_band_fraction_dec"],
                        r["prop_mae_eval"] / d, r["bias"] / d, r["sd"] / d,
                        r["mol_stability"], r["atom_stability"], r["validity"],
                        r["uniqueness_of_valid"], r["uniq_min_cell"],
                        r["unique_valid_per_sample"], r["diversity_mean_pairwise"],
                        r["guide_eval_gap_mean"] / d))
        L.append("")
        if u is None:
            L.append("*No unguided cell for %s, so no reference comparison.*" % p)
            L.append("")
            continue
        L.append("| arm vs unguided | d(in_band) | z | d(in_band) dec | z | "
                 "d(mol_stab) | d(validity) |")
        L.append("|---|---|---|---|---|---|---|")
        for a in av:
            if a == "unguided":
                continue
            r = P[(p, a)]
            L.append("| `%s` | %+.4f | %+.2f | %+.4f | %+.2f | %+.4f | %+.4f |"
                     % (a, r["in_band_fraction"] - u["in_band_fraction"],
                        z2(r, u, "in_band_fraction", "se_ib"),
                        r["in_band_fraction_dec"] - u["in_band_fraction_dec"],
                        z2(r, u, "in_band_fraction_dec", "se_ib_dec"),
                        r["mol_stability"] - u["mol_stability"],
                        r["validity"] - u["validity"]))
        L.append("")
        # the summary, with chemistry attached and no bare ranking
        guided = [a for a in av if a != "unguided"]
        best = max(guided, key=lambda a: P[(p, a)]["in_band_fraction"])
        b = P[(p, best)]
        # MULTIPLICITY. `best` is a MAXIMUM over however many guided arms ran --
        # up to 21, of which 17 are BDG settings. Reporting its z against a fixed
        # 3-sigma is a selection effect, and BDG_REVIEW.md already named the
        # max-over-13 selection as the larger threat to the port's result. Print
        # the Bonferroni-adjusted threshold beside it so the comparison a reader
        # actually makes is the corrected one. (Bonferroni over the arms tested
        # on this property; it is conservative, and max-T on the sidecars would
        # be tighter -- see the paired-test note in this file's docstring.)
        m_tests = len(guided)
        z_bonf = NORM_Q(1.0 - (2.0 * NORM_SF(SIGMA)) / (2.0 * m_tests))
        L.append("**Selection:** the row below is the largest in-band of **%d** "
                 "guided arms on %s. An uncorrected %.1f-sigma bar is not valid "
                 "for a maximum; the Bonferroni-adjusted two-sided bar at the "
                 "same family error rate is **z = %.2f**. Arms clearing it "
                 "against unguided: %s."
                 % (m_tests, p, SIGMA, z_bonf,
                    ", ".join("`%s`" % a for a in guided
                              if abs(z2(P[(p, a)], u, "in_band_fraction", "se_ib"))
                              >= z_bonf) or "**none**"))
        L.append("")
        note = ""
        if b["mol_stability"] < u["mol_stability"]:
            note = (" **It reaches that by spending chemistry:** molecule stability "
                    "%.4f against unguided's %.4f (%+.1f %%), validity %.4f against "
                    "%.4f. At w = 1 an arm whose own floor-clearing strength is far "
                    "below 1 buys in-band this way, so this is NOT a statement that "
                    "it is the better method."
                    % (b["mol_stability"], u["mol_stability"],
                       100 * (b["mol_stability"] / u["mol_stability"] - 1),
                       b["validity"], u["validity"]))
        L.append("**%s: highest in-band is `%s`** at %.4f (z = %+.2f vs unguided, "
                 "%+.2f decoded).%s"
                 % (p, best, b["in_band_fraction"],
                    z2(b, u, "in_band_fraction", "se_ib"),
                    z2(b, u, "in_band_fraction_dec", "se_ib_dec"), note))
        L.append("")
        keep = [a for a in guided if P[(p, a)]["mol_stability"] >= u["mol_stability"]]
        if keep:
            k = max(keep, key=lambda a: P[(p, a)]["in_band_fraction"])
            r = P[(p, k)]
            L.append("Highest in-band among arms that do **not** lose chemistry "
                     "against unguided: `%s` at %.4f (z = %+.2f, %+.2f decoded), "
                     "molecule stability %.4f."
                     % (k, r["in_band_fraction"], z2(r, u, "in_band_fraction", "se_ib"),
                        z2(r, u, "in_band_fraction_dec", "se_ib_dec"), r["mol_stability"]))
        else:
            L.append("**No guided arm holds chemistry at unguided's level on %s.** "
                     "Every arm at w = 1 costs stability here, so every in-band gain "
                     "on this property is bought." % p)
        L.append("")
    # BDG's controller state, which is the ablation's whole point
    bdgs = [a for a in arms if a.startswith("bdg_e")]
    if bdgs:
        L.append("### BDG controller state (%s)" % backend)
        L.append("")
        L.append("`w_eff = 1 + eta*e` is the weight on the DEVIATION term; `V_b/tau^2` "
                 "is how far the batch sits from its setpoint. Each entry is "
                 "**signed mean / RMS / share of guided steps with w_eff < 0**, and "
                 "the mean is taken over guided steps and then over the %d "
                 "controllers a cell runs (one per batch)."
                 % (int(any_row["n"]) // int(any_row["batch"])
                    if any_row.get("batch") else 0))
        L.append("")
        L.append("Read RMS against the signed mean: w_eff changes sign inside a "
                 "window, so signed ~ 0 with a large RMS means an arm that was "
                 "violently active, not an inert one. `eta = 0` must reproduce "
                 "`plug` in the guidance FIELD (`test_bdg.py` A9/B-a, bit-identical); "
                 "a cell-level identity is NOT asserted -- see the run-to-run "
                 "envelope in FULL_RUN_V3_PROTOCOL.md section 2.4. The closed form "
                 "w_eff = 1 + eta(1/tau_mult^2 - 1) is WITHDRAWN: it is two orders "
                 "of magnitude below what real trajectories do (section 4.1).")
        L.append("")
        L.append("| arm | eta | tau_mult | " + " | ".join(
            "%s: mean / RMS / frac<0" % p for p in PROPS) + " |")
        L.append("|---|---|---|" + "---|" * len(PROPS))
        for a in bdgs:
            eta, tm = a[len("bdg_e"):].split("t")
            row = []
            for p in PROPS:
                r = P.get((p, a))
                if r is None or r.get("bdg_w_eff") is None:
                    row.append("--")
                elif r.get("bdg_w_eff_rms") is None:
                    # cells written before the RMS/sign diagnostics existed
                    row.append("%+.1f / n/a / n/a" % r["bdg_w_eff"])
                else:
                    row.append("%+.1f / %.1f / %.2f"
                               % (r["bdg_w_eff"], r["bdg_w_eff_rms"],
                                  r["bdg_w_eff_neg"]))
            L.append("| `%s` | %s | %s | %s |" % (a, eta, tm, " | ".join(row)))
        L.append("")
        if any(P.get((p, a), {}).get("bdg_w_eff_rms") is None
               for p in PROPS for a in bdgs if (p, a) in P):
            L.append("**Some cells predate `bdg_w_eff_sq`/`bdg_w_eff_neg`** and show "
                     "`n/a`: for those, only the signed mean exists and cancellation "
                     "is invisible. They cannot support a claim about the controller.")
            L.append("")
    L.append("### Cost and clip saturation (%s)" % backend)
    L.append("")
    L.append("Clipped steps are printed for **every property**, not just %s. The "
             "clip is a live confound on BDG's top rungs: at `w_eff` of order "
             "10^3 the dispersion term sits far outside the velocity-relative "
             "trust region, so what reaches the sample is the clip's truncation "
             "of it, and a rung can be clip-limited rather than "
             "controller-limited. Compare each row against `plug`'s. The clip "
             "cannot simply be widened -- removing it produced 222 non-finite "
             "samples against 0." % PROPS[0])
    L.append("")
    L.append("| arm | generator passes | guide passes | "
             + " | ".join("%s clipped" % p for p in PROPS) + " | vs plug |")
    L.append("|---|---|---|" + "---|" * (len(PROPS) + 1))
    plug_tot = sum(P[(p, "plug")]["clipped"] for p in PROPS
                   if (p, "plug") in P) or None
    for a in arms:
        r = next((P[(p, a)] for p in PROPS if (p, a) in P), None)
        if not r:
            continue
        clips, tot = [], 0
        for p in PROPS:
            rp = P.get((p, a))
            clips.append("%d" % rp["clipped"] if rp else "--")
            tot += rp["clipped"] if rp else 0
        ratio = ("%.2fx" % (tot / plug_tot)) if plug_tot and a != "plug" else "--"
        L.append("| `%s` | %d | %d | %s | %s |"
                 % (a, r["gen_passes"], r["guide_passes"], " | ".join(clips), ratio))
    L.append("")
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "v3"))
    ap.add_argument("--backend", default="fm", choices=BACKENDS)
    ap.add_argument("--both", action="store_true",
                    help="both base models on one page; they are NOT compared statistically")
    # NO DEFAULT. v3 pre-registers no cell size (the operator picks it), so a
    # default here would quietly tabulate a tree that may not be the one that
    # ran. Both stages now use the SAME n and the SAME seeds and are told apart
    # by --stage, which selects the directory level and the expected arm set.
    ap.add_argument("--n", type=int, required=True,
                    help="cell size the run used (results/v3/<be>/<stage>/n<N>/)")
    ap.add_argument("--stage", default="v3", choices=["v3", "v3abl"],
                    help="v3 = the headline (comparison set + BDG eta=4); "
                         "v3abl = the eta x tau_mult grid, which carries NO "
                         "comparison arm and is read against itself")
    ap.add_argument("--seeds", default="20261001,20261002,20261003")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()
    seeds = [s for s in args.seeds.split(",") if s]
    backends = list(BACKENDS) if args.both else [args.backend]

    loaded, problems = {}, []
    for be in backends:
        c, pr = load(args.root, be, args.n, seeds, stage=args.stage)
        problems += check(c, pr, stage=args.stage, backend=be)
        loaded[be] = c
    if problems:
        print("REFUSING: the v3 run is not complete or not consistent.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)

    L = ["# Protocol v3 %s results%s"
         % ("ablation" if args.stage == "v3abl" else "headline",
            " - both base models" if args.both else ""), ""]
    L.append("Generated by `proj1/scripts/v3_table.py`. Do not hand-edit; re-run it.")
    L.append("")
    L.append("Protocol: [FULL_RUN_V3_PROTOCOL.md](../protocol/FULL_RUN_V3_PROTOCOL.md).")
    L.append("")
    L.append("> **There is no chemistry floor in v3, and no arm is excluded.** Every "
             "in-band number below is printed beside the chemistry it cost. `w = 1` is "
             "not each arm's best strength -- tfg's floor-clearing strength under v2 "
             "was 0.01-0.05, so at w = 1 it is 20-100x that -- and an arm far above "
             "its own best will show high in-band with collapsed stability. **Do not "
             "read the in-band column as a ranking.**")
    L.append("")
    if args.both:
        L.append("> **The two base models are not paired.** Different architectures, "
                 "different noise schedules and different state scales, so molecule i "
                 "of one is not molecule i of the other. Rows are side by side; no "
                 "cross-backend difference is tested.")
        L.append("")
    for be in backends:
        emit(L, be, loaded[be], args.n, seeds)
    out = NL.join(L) + NL
    print(out)
    if args.md_out:
        dst = args.md_out if os.path.isabs(args.md_out) else os.path.join(ROOT, args.md_out)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "w", encoding="utf-8").write(out)
        print("wrote %s" % dst, file=sys.stderr)


if __name__ == "__main__":
    main()
