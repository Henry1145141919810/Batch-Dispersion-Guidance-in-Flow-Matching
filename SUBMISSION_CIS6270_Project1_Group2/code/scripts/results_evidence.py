"""Sections 5-9 of docs/results/FULL_RUN_RESULTS.md: does guidance work?

Called by results_tables.py; not run on its own. Kept in its own module so the
pre-existing, independently checked sections 1-4 of that page are untouched.

WHY THESE SECTIONS EXIST. Read alone, the pre-registered table makes guidance
look inert: FR3a's chemistry floor, applied to the q90 compare stage, froze
plug and btvg at w = 0.05 on mu, where the correction is almost off. A reader
then concludes guidance does nothing. These sections put beside that table
what the same data show:

  5  tracking beyond atom count (partial correlation with each molecule's own
     target, within atom-count groups) -- a test a target-ignoring generator
     cannot pass, and EDM's own stated criterion
  6  the q90 compare stage strength by strength: how far guidance moves the
     property, what chemistry it costs, and exactly why FR3a froze what it froze
  7  the correction's measured share of the sampling velocity at equal w
  8  the centring ceiling: why in-band stays low even when the mean is on target
  9  what this supports and what it does not

EVERY NUMBER IS COMPUTED HERE from files on disk; the prose explains what each
table measures and never states a number itself.
"""
from __future__ import annotations

import math
import os

import results_tables as rt

CMP_ARMS = ("plug", "tmpd", "lgd_mc", "dflow", "tfg", "btvg", "btvg_var")
GRID = (0.01, 0.05, 0.25, 0.5, 1.0, 2.0, 4.0)
FULL_ARMS = ("plug", "tmpd", "lgd_mc", "btvg", "btvg_var", "tfg")
FORCE_ARMS = ("plug", "tmpd", "lgd_mc", "btvg", "btvg_var")
FORCE_W = (0.05, 0.25, 1.0, 4.0)
Z_HALF = 0.6744897501960817          # P(|Z| <= Z_HALF) = 0.5


# ---------------------------------------------------------------- helpers

def mae_se(r):
    """se of a cell's MAE: |e| has mean MAE and second moment RMSE^2."""
    return math.sqrt(max(r["prop_rmse_eval"] ** 2 - r["prop_mae_eval"] ** 2, 0.0) / r["n"])


def r_se(r, n):
    """Large-sample se of a correlation coefficient."""
    return (1.0 - r * r) / math.sqrt(max(n - 3, 1))


def groups(ALL, p):
    """{arm: {w: cell_metrics}} from the all-strengths dist report."""
    g = {}
    for k, v in ALL[p]["arms"].items():
        arm, w = k.split("@w")
        g.setdefault(arm, {})[float(w)] = v
    return g


def n_fin(cm):
    return cm["n"] - cm["n_nonfinite"]


def force(p, a, w):
    fn = os.path.join(rt.ROOT, "results", "force_share", "%s__%s__w%g.json" % (p, a, w))
    if not os.path.exists(fn):
        return None
    return rt.rjson(fn)["euler"]["r_t_mean"]


def sets(a, p, W, WM):
    """[(label, w)] for the pre-registered strength sets an arm has in the run."""
    out = [("FR3a", float(W[a][p]))]
    if float(WM[a][p]) != float(W[a][p]):
        out.append(("FR3", float(WM[a][p])))
    else:
        out[0] = ("FR3a = FR3", out[0][1])
    return out


def name(a):
    return "**tfg** (post hoc)" if a == "tfg" else a


def full_cell(p, a, w, seed="20261001"):
    fn = os.path.join(rt.ROOT, "results", "full", "n5000", "seed%s" % seed,
                      "%s__%s__dist__w%g__tmin0.5__full.json" % (p, a, w))
    return rt.rjson(fn) if os.path.exists(fn) else None


def floor_of(p, cm, ALL):
    """The same floor sections 1 and 2 use: 0.9 x unguided's molecule stability,
    against the POOLED unguided for a 3-seed group (section 1b) and against that
    seed's OWN unguided for a one-seed group (section 2)."""
    if len(cm["seeds"]) == 1:
        ref = full_cell(p, "unguided", 1.0, str(cm["seeds"][0]))["mol_stability"]
    else:
        ref = groups(ALL, p)["unguided"][1.0]["mol_stability"]
    return 0.9 * ref


def clears_floor(p, cm, ALL):
    return cm["mol_stability"] >= floor_of(p, cm, ALL) - 1e-12


def tfg_fr3(ALL, WM):
    """tfg at its unconstrained strength: (soft in-band, decoded in-band, mol stab)."""
    out = []
    for p in rt.PROPS:
        c = full_cell(p, "tfg", float(WM["tfg"][p]))
        out.append((c["in_band_fraction"], c["in_band_fraction_dec"], c["mol_stability"]))
    return out


def ceiling_rows(ALL, W, WM):
    """Every row of section 8, as dicts. One function, so the headline, section 8 and
    section 9 cannot disagree about which rows clear the floor or sit at their ceiling.

    Keys: p, a, lab, w, ok (the floor convention of sections 1-2), ok_pooled (always
    against the pooled unguided), ratio (in-band / Gaussian ceiling), sd (residual sd
    in delta), narrow (1 - sd / unguided's sd), nseed, margin (molecules above the
    floor of `ok`; negative below it). A missing cell RAISES: silently skipping one
    would change every count and range quoted from these rows."""
    rows = []
    for p in rt.PROPS:
        G = groups(ALL, p)
        dl = ALL[p]["ladder"]["delta"]
        u = G["unguided"][1.0]
        todo = [("unguided", "reference", 1.0)]
        for a in FULL_ARMS:
            todo += [(a, lab, w) for lab, w in sets(a, p, W, WM)]
        for a, lab, w in todo:
            if w not in G.get(a, {}):
                raise SystemExit("dist_report_all.json has no %s@w%g for %s -- regenerate "
                                 "it (see results_tables.py docstring)" % (a, w, p))
            c = G[a][w]
            rows.append({"p": p, "a": a, "lab": lab, "w": w,
                         "ok": clears_floor(p, c, ALL),
                         "ok_pooled": c["mol_stability"] >= 0.9 * u["mol_stability"] - 1e-12,
                         "ratio": c["in_band"] / c["in_band_ceiling"],
                         "sd": c["resid_sd"] / dl,
                         "narrow": 1.0 - c["resid_sd"] / u["resid_sd"],
                         "nseed": len(c["seeds"]), "n": c["n"],
                         "margin": (c["mol_stability"] - floor_of(p, c, ALL)) * c["n"]})
    return rows


def narrow_text(R):
    """How much guidance narrows the residual, at floor-clearing strengths. The
    largest value can hang on the floor convention, so all three maxima are
    computed: own convention, pooled floor for every row, 3-seed rows only."""
    g = [r for r in R if r["a"] != "unguided"]
    best = max([r for r in g if r["ok"]], key=lambda r: r["narrow"])
    pool = max([r for r in g if r["ok_pooled"]], key=lambda r: r["narrow"])
    three = max([r for r in g if r["ok"] and r["nseed"] > 1], key=lambda r: r["narrow"])
    lab = lambda r: "%s %s on %s" % (r["a"], r["lab"], r["p"])
    if best is three:
        return "up to %s (%s)" % (rt.pct(best["narrow"]), lab(best))
    lo = min(pool["narrow"], three["narrow"])
    return ("%s to %s depending on the floor convention: the %s (%s) is one seed, "
            "clears its own floor by %.0f of %d molecules%s; under the pooled floor the most is "
            "%s (%s), and among 3-seed rows %s (%s)" % (
                rt.pct(lo), rt.pct(best["narrow"]), rt.pct(best["narrow"]), lab(best),
                best["margin"], best["n"],
                "" if best["ok_pooled"] else ", and fails the pooled floor",
                rt.pct(pool["narrow"]), lab(pool), rt.pct(three["narrow"]), lab(three)))


def atoms_passes(ALL, W, WM):
    """EDM's #Atoms bar: (FR3a cells that pass, FR3 plug/tmpd/btvg cells that pass,
    those clearing the section-2 floor, those clearing the pooled floor)."""
    G = {p: groups(ALL, p) for p in rt.PROPS}
    uu = {p: G[p]["unguided"][1.0]["mol_stability"] for p in rt.PROPS}
    fr3 = [(p, a, G[p][a][float(WM[a][p])]) for p in rt.PROPS for a in ("plug", "tmpd", "btvg")]
    beat = [(p, a, c) for p, a, c in fr3 if c["beats_atoms"]]
    return ([(p, a) for p in rt.PROPS for a in FULL_ARMS
             if G[p][a][float(W[a][p])]["beats_atoms"]],
            beat,
            [(p, a) for p, a, c in beat if clears_floor(p, c, ALL)],
            [(p, a) for p, a, c in beat if c["mol_stability"] >= 0.9 * uu[p] - 1e-12])


# ---------------------------------------------------------------- read first

def read_first(out, ALL, W, WM):
    G = {p: groups(ALL, p) for p in rt.PROPS}
    ung = [G[p]["unguided"][1.0]["partial_corr"] for p in rt.PROPS]
    trio = ("plug", "tmpd", "btvg")
    fr3 = [(p, a, G[p][a][float(WM[a][p])]) for p in rt.PROPS for a in trio]
    strong = [c["partial_corr"] for _, _, c in fr3]
    seeds = sorted({s for _, _, c in fr3 for s in c["seeds"]})
    ws = sorted({float(WM[a][p]) for p in rt.PROPS for a in trio})
    wp, wb = float(W["plug"]["mu"]), float(W["btvg"]["mu"])
    fp, fb = force("mu", "plug", wp), force("mu", "btvg", wb)
    out("## Read this first: does guidance work?")
    out("")
    out("**Yes, and the evidence is in this run.** Read alone, the pre-registered table "
        "below makes guidance look inert: on mu, `plug` and `btvg` sit almost on top of "
        "`unguided`. That is not because guidance fails. FR3a froze them at %s on mu, "
        "where %s (sections 6 and 7)." % (
            "w = %g" % wp if wp == wb else "w = %g and %g" % (wp, wb),
            "the correction is %s (plug) and %s (btvg) of the base velocity -- measured on "
            "the q50 target, n = 128, not on this run" % (
                "%.1f%%" % (100 * fp), "%.1f%%" % (100 * fb))
            if fp is not None and fb is not None
            else "the correction is barely distinguishable from switching guidance off"))
    out("")
    fr3a_beat, beat, beat_ok, beat_pooled = atoms_passes(ALL, W, WM)
    out("The decisive test is **section 5**: within each atom-count group, does the "
        "generated molecule's property follow *its own* target? The unguided generator "
        "never sees the target, and scores %s / %s / %s (mu / alpha / gap). At the "
        "pre-registered FR3 strength (w = %s; seed %s only), `plug`, `tmpd` and `btvg` "
        "score %s to %s. This partial correlation is our continuous form of EDM's "
        "criterion for a method that incorporates the property \"into the molecular "
        "structure beyond the number of atoms\" (Hoogeboom et al. 2022, section 5.2): a "
        "generator that ignores its target scores zero on it." % (
            *(rt.f(x) for x in ung), " / ".join("%g" % w for w in ws),
            ", ".join(str(s) for s in seeds), rt.f(min(strong), 2), rt.f(max(strong), 2)))
    out("")
    out("EDM's own published bar is stricter: beat, on MAE, a lookup of the median "
        "property for each atom count. %s At the FR3 strength, `plug`, `tmpd` and `btvg` "
        "pass it in %d of 9 property cells, %d of them above the chemistry floor (one "
        "seed; section 5)%s." % (
            "No arm passes it at its FR3a strength, on any property."
            if not fr3a_beat else "At the FR3a strength %d arm-property cells pass it."
            % len(fr3a_beat), len(beat), len(beat_ok),
            (" -- and %s only under the one-seed floor convention, failing the pooled one "
             "(section 8)" % ("both" if len(beat_ok) == 2 else "all")
             if beat_ok and not beat_pooled else "")))
    out("")
    R = ceiling_rows(ALL, W, WM)
    ok_g = [r for r in R if r["ok"] and r["a"] != "unguided"]
    tight = min(ok_g, key=lambda r: r["sd"])
    ok_all = [r for r in R if r["ok"]]
    out("**The price, and the limit.** At that strength most arms fall below the "
        "chemistry floor on mu and gap (section 2). Guidance does narrow the output: at "
        "strengths that clear the floor the residual sd falls against unguided by %s. "
        "The narrowest floor-clearing residual is %.2f delta "
        "(%s %s on %s). But in-band 0.50 would need about %.2f delta, and across the %d "
        "floor-clearing rows in-band is %.2f to %.2f times the Gaussian ceiling for its "
        "own width -- so the constraint is the width that remains, not the aim "
        "(section 8). At the strengths the rubric admits, guidance **steers, and narrows "
        "the spread only partly.**" % (
            narrow_text(R),
            tight["sd"], tight["a"], tight["lab"], tight["p"], 1.0 / Z_HALF,
            len(ok_all), min(r["ratio"] for r in ok_all), max(r["ratio"] for r in ok_all)))
    out("")
    t = tfg_fr3(ALL, WM)
    tr = [r for r in R if r["a"] == "tfg" and r["lab"] == "FR3"]
    out("Only tfg at its unconstrained strength (post hoc, one seed) gets the residual "
        "down to a few delta (%.2f to %.2f) and above its Gaussian ceiling, and it pays "
        "for it: molecule stability %s / %s / %s (mu / alpha / gap), far below the floor, "
        "and part of the gain is on the continuous atom features only -- decoding them "
        "to real atom types lowers its in-band from %s / %s / %s to %s / %s / %s. Equal w "
        "is not equal force either (section 7), so neither equal w nor each arm's "
        "floor-clearing best is a controlled comparison." % (
            min(r["sd"] for r in tr), max(r["sd"] for r in tr),
            *(rt.f(x[2]) for x in t), *(rt.f(x[0]) for x in t), *(rt.f(x[1]) for x in t)))
    out("")
    out("### How to read every metric on this page")
    out("")
    out("| metric | computed as | shows | does not show |")
    out("|---|---|---|---|")
    rows = (
        ("in-band", "share of molecules with abs(f_B - target) <= delta, delta = 2 x "
         "f_B's validation MAE", "the project's pre-registered acceptance criterion",
         "limited by the output's spread (section 8), so it can stay low while guidance "
         "steers"),
        ("MAE", "mean abs(f_B - target); f_B is the held-out evaluator",
         "that outputs are closer to their targets",
         "whether the gain is a mean shift or a narrowing"),
        ("partial corr.", "correlation of f_B(output) with the target, after removing "
         "each atom-count group's mean from both", "that outputs follow their own "
         "targets beyond what atom count explains", "magnitude: a correctly ordered but "
         "weak arm still scores high"),
        ("gap closure", "(size-shuffle MAE - MAE) / (size-shuffle MAE - L-bound MAE)",
         "share of the available target information captured; comparable across "
         "properties", "inherits MAE's sensitivity to spread"),
        ("beats #Atoms", "MAE below a median-property-by-atom-count lookup (EDM Table 3)",
         "EDM's published bar", "a point predictor is a hard bar for any sampler "
         "(DIST_PROTOCOL.md section 3)"),
        ("% of gap closed", "(guided mean f_B - unguided mean f_B) / (target - unguided "
         "mean f_B), q90 compare stage", "how far guidance moves the whole property "
         "distribution toward a tail target", "spread"),
        ("mol. stability", "share of molecules whose every atom has its allowed valency",
         "the chemistry price", "compounds over ~18 atoms, so it falls fast"),
        ("correction share", "mean abs(C(G)) / abs(V) over guided steps", "how hard an "
         "arm actually pushes, whatever its w", "whether the push is useful"),
        ("centring ceiling", "P(abs(Z) <= delta / residual sd) for a centred Gaussian",
         "the best in-band an arm's own spread allows", "non-Gaussian tails: a peaked "
         "residual can sit slightly above it"),
    )
    for r in rows:
        out("| %s | %s | %s | %s |" % r)
    out("")
    out("All property metrics are scored on the continuous atom features each arm ends "
        "on, including partial correlation. Only tfg has a decoded (argmax one-hot) view; "
        "it is shown where it exists.")
    out("")


# ---------------------------------------------------------------- 5 tracking

def section5(out, ALL, W, WM):
    out("## 5. Does guidance work? Tracking beyond atom count (full run, dist, n = 5,000)")
    out("")
    out("The `dist` protocol hands every arm the atom count of its target molecule, and "
        "atom count alone predicts much of each property, so a generator can post a "
        "respectable MAE while ignoring the target. Partial correlation removes that: it "
        "correlates output with target *within* each atom-count group. A generator that "
        "uses the size but ignores the target scores 0 however good its MAE; the real "
        "molecules set the ceiling.")
    out("")
    out("z = (arm - unguided) / sqrt(se_arm^2 + se_unguided^2) with the large-sample se "
        "of a correlation. Two caveats pull in opposite directions. It treats the two as "
        "independent samples, but they share their targets and their initial noise, so "
        "the se of the difference is smaller than this. And it is conditional on these "
        "5,000 test molecules: a fresh draw of targets would add variance it does not "
        "count. Read the large values as \"far outside noise\", not as exact z. FR3a rows "
        "pool 3 seeds; FR3 rows are seed 20261001 only.")
    out("")
    G = {p: groups(ALL, p) for p in rt.PROPS}
    out("| arm | set | w (mu / alpha / gap) | partial corr. mu | alpha | gap "
        "| z vs unguided (mu / alpha / gap) | beats #Atoms (mu / alpha / gap) |")
    out("|---|---|---|---|---|---|---|---|")
    u = {p: G[p]["unguided"][1.0] for p in rt.PROPS}
    out("| unguided | reference | - | %s | %s | %s | - | %s |" % (
        *(rt.f(u[p]["partial_corr"]) for p in rt.PROPS),
        " / ".join("yes" if u[p]["beats_atoms"] else "no" for p in rt.PROPS)))
    for a in FULL_ARMS:
        labels = [("FR3a", W), ("FR3", WM)]
        for lab, WW in labels:
            ws = [float(WW[a][p]) for p in rt.PROPS]
            if lab == "FR3" and ws == [float(W[a][p]) for p in rt.PROPS]:
                continue
            cm = [G[p][a][w] for p, w in zip(rt.PROPS, ws)]
            # an FR3 strength equal to FR3a's is the same pooled cell; say so
            # rather than print it twice as though it were a second measurement
            same = [lab == "FR3" and w == float(W[a][p]) for p, w in zip(rt.PROPS, ws)]
            zs = []
            for p, c in zip(rt.PROPS, cm):
                su = r_se(u[p]["partial_corr"], n_fin(u[p]))
                sa = r_se(c["partial_corr"], n_fin(c))
                zs.append((c["partial_corr"] - u[p]["partial_corr"]) / math.hypot(su, sa))
            out("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                name(a), lab, " / ".join("%g" % w for w in ws),
                *("same as FR3a" if s else rt.f(c["partial_corr"]) for c, s in zip(cm, same)),
                " / ".join("-" if s else "%+.1f" % z for z, s in zip(zs, same)),
                " / ".join("-" if s else ("**yes**" if c["beats_atoms"] else "no")
                           for c, s in zip(cm, same))))
    out("| *real molecules (ceiling)* | - | - | %s | %s | %s | - | - |" % tuple(
        rt.f(ALL[p]["ladder"]["l_bound"]["partial_corr"]) for p in rt.PROPS))
    out("")
    out("Rows whose partial correlation is several times unguided's and whose z is large "
        "are the evidence that guidance places target information in the molecule. Rows "
        "near zero are arms held at a strength where the correction is almost off. The "
        "chemistry cost of each row is in section 2 (FR3) and section 1 (FR3a); an arm "
        "below the floor cannot win or be beaten under the rubric, so these rows show that "
        "guidance works, not which arm is best.")
    out("")
    out("tfg's rows are scored on the continuous atom features, and partial correlation "
        "has no decoded view for any arm. For tfg that matters: at its unconstrained "
        "strength decoding removes much of its in-band gain (section 2), so read its FR3 "
        "partial correlation as an upper bound on what its decoded molecules achieve.")
    out("")


# ---------------------------------------------------------------- 6 q90

def q_distance(ours, p, tgt):
    """abs(target - unguided mean) / sd of the unguided output, from the unguided cell.
    The target is a constant there, so the residual's sd IS the output's sd:
    sd = sqrt(rmse^2 - bias^2) with bias = f_B mean - target."""
    u = ours[(p, "unguided", tgt, 1.0)]
    bias = u["f_B_mean"] - u["target_mean"]
    sd = math.sqrt(max(u["prop_rmse_eval"] ** 2 - bias ** 2, 0.0))
    return abs(bias) / sd


def section6(out, ours, W):
    out("## 6. The q90 compare stage, strength by strength (n = 512, t_min 0.5)")
    out("")
    d50 = [q_distance(ours, p, "q50") for p in rt.PROPS]
    d90 = [q_distance(ours, p, "q90") for p in rt.PROPS]
    out("q50 targets sit %.2f-%.2f sd from the unguided generator's own mean, so there is "
        "little to steer. **q90 targets sit %.2f-%.2f sd into the tail** (mu / alpha / "
        "gap: %s), where an unguided generator essentially cannot reach. If guidance did "
        "nothing, the property distribution would not move. These are also the cells "
        "FR3a froze the full run's strengths on, so the second table of each block shows "
        "exactly why each arm got the strength it got." % (
            min(d50), max(d50), min(d90), max(d90), " / ".join("%.2f" % x for x in d90)))
    out("")
    out("Notation. % of gap closed = (guided mean - unguided mean) / (target - unguided "
        "mean), all scored by f_B; the bracket is the MAE z against unguided. In the "
        "stability table x marks a strength below the floor (0.9 x unguided) and **bold** "
        "marks the strength FR3a chose: the best MAE among strengths clearing the floor, "
        "or the most stable if none does. n = 512 is a screen, not a verdict.")
    out("")
    for p in rt.PROPS:
        u = ours[(p, "unguided", "q90", 1.0)]
        T, U = u["target_mean"], u["f_B_mean"]
        dl = u["delta"]
        floor = 0.9 * u["mol_stability"]
        out("### 6%s. %s -- target %s, unguided mean %s, gap %s delta" % (
            "abc"[rt.PROPS.index(p)], p, rt.f(T, 4), rt.f(U, 4), rt.f((T - U) / dl, 1)))
        out("")
        out("Unguided: in-band %s, molecule stability %s, floor %s." % (
            rt.f(u["in_band_fraction"]), rt.f(u["mol_stability"]), rt.f(floor)))
        out("")
        hdr = "| arm | " + " | ".join("w=%g" % w for w in GRID) + " |"
        sep = "|---|" + "---|" * len(GRID)
        chosen = {}
        for a in CMP_ARMS:
            rr = [r for (pp, aa, tt, _), r in ours.items() if pp == p and aa == a and tt == "q90"]
            if rr:
                chosen[a] = float(rt.fr3a(ours, p, a)[0]["w"])
        out("% of gap closed (MAE z vs unguided):")
        out("")
        out(hdr)
        out(sep)
        for a in CMP_ARMS:
            cells = []
            for w in GRID:
                r = ours.get((p, a, "q90", w))
                if r is None:
                    cells.append("-")
                    continue
                z = (u["prop_mae_eval"] - r["prop_mae_eval"]) / math.hypot(mae_se(r), mae_se(u))
                cells.append("%s (%+.1f)" % (rt.pct((r["f_B_mean"] - U) / (T - U)), z))
            out("| %s | %s |" % (name(a), " | ".join(cells)))
        out("")
        out("Molecule stability (x = below the floor, **bold** = FR3a's choice):")
        out("")
        out(hdr)
        out(sep)
        for a in CMP_ARMS:
            cells = []
            for w in GRID:
                r = ours.get((p, a, "q90", w))
                if r is None:
                    cells.append("-")
                    continue
                s = rt.f(r["mol_stability"]) + ("" if r["mol_stability"] >= floor - 1e-12 else " x")
                cells.append("**%s**" % s if chosen.get(a) == w else s)
            out("| %s | %s |" % (name(a), " | ".join(cells)))
        out("")
        out("In-band:")
        out("")
        out(hdr)
        out(sep)
        for a in CMP_ARMS:
            cells = [rt.f(ours[(p, a, "q90", w)]["in_band_fraction"])
                     if (p, a, "q90", w) in ours else "-" for w in GRID]
            out("| %s | %s |" % (name(a), " | ".join(cells)))
        dec = [ours.get((p, "tfg", "q90", w), {}).get("in_band_fraction_dec") for w in GRID]
        if any(x is not None for x in dec):
            out("| tfg, decoded atom types | %s |" % " | ".join(
                "-" if x is None else rt.f(x) for x in dec))
        out("")
        # dflow left the full run, so its frozen strengths are only in frozen_q90.json
        WD = dict(W)
        if "dflow" not in WD:
            old_fz = os.path.join(rt.ROOT, "results", "full", "n5000", "frozen_q90.json")
            if os.path.exists(old_fz):
                WD.update({k: v for k, v in rt.rjson(old_fz)["frozen_w"].items()
                           if k == "dflow"})
        unchecked = [a for a in chosen if a not in WD]
        mism = [a for a in chosen if a in WD and abs(chosen[a] - float(WD[a][p])) > 1e-12]
        if unchecked:
            out("Check: no frozen file records %s, so its bold mark is the rule applied "
                "here, not a checked value." % ", ".join(unchecked))
            out("")
        if mism:
            out("Check: for %s the rule applied here disagrees with the frozen file; the "
                "frozen file is what the full run used." % ", ".join(mism))
            out("")
    # which arms' share of the gap rises with strength on every property
    # (a 2-point tolerance absorbs n = 512 noise between neighbouring strengths)
    def share(p, a, w):
        u = ours[(p, "unguided", "q90", 1.0)]
        r = ours[(p, a, "q90", w)]
        return (r["f_B_mean"] - u["f_B_mean"]) / (u["target_mean"] - u["f_B_mean"])
    TOL = 0.02
    drop = {}                       # arm -> (largest fall between neighbours, prop, w, w')
    for a in CMP_ARMS:
        if all((p, a, "q90", w) in ours for p in rt.PROPS for w in GRID):
            drop[a] = max((share(p, a, GRID[i]) - share(p, a, GRID[i + 1]), p,
                           GRID[i], GRID[i + 1])
                          for p in rt.PROPS for i in range(len(GRID) - 1))
    mono = [a for a in drop if drop[a][0] <= TOL]
    rest = [a for a in drop if a not in mono]
    gone = [a for a in CMP_ARMS if a not in drop]
    say = lambda a: "`%s` falls %.2f points (%s, w = %g to %g)" % (
        a, 100 * drop[a][0], drop[a][1], drop[a][2], drop[a][3])
    dipped = [a for a in mono if drop[a][0] > 0]
    out("Reading these blocks. In the first table the share of the gap closed rises with "
        "strength on every property for %s -- guidance working. \"Rises\" allows a fall "
        "of up to %g points between neighbouring strengths, as n = 512 noise%s. %s%s The "
        "second table shows "
        "the price, and why the floor admits only the weakest strengths for arms whose "
        "stability falls fast. The third shows in-band rising much less than the first: "
        "moving the mean toward a tail target is not the same as landing molecules "
        "within delta of it (section 8)." % (
            ", ".join("`%s`" % a for a in mono), 100 * TOL,
            ("; within that allowance %s" % "; ".join(say(a) for a in dipped))
            if dipped else "",
            ("Beyond it, at its largest fall, %s." % "; ".join(say(a) for a in rest))
            if rest else "",
            (" %s lack%s some strengths." % (", ".join("`%s`" % a for a in gone),
                                             "s" if len(gone) == 1 else ""))
            if gone else ""))
    out("")
    # arms that move the property AWAY from the target at w = 4: MAE z <= -3, the
    # |z| >= 3 threshold FR5 applies to in-band, borrowed here for MAE
    wrong = []
    for p in rt.PROPS:
        u = ours[(p, "unguided", "q90", 1.0)]
        for a in CMP_ARMS:
            r = ours.get((p, a, "q90", 4.0))
            if r is None:
                continue
            g = (r["f_B_mean"] - u["f_B_mean"]) / (u["target_mean"] - u["f_B_mean"])
            z = (u["prop_mae_eval"] - r["prop_mae_eval"]) / math.hypot(mae_se(r), mae_se(u))
            if g < 0 and z <= -3:
                wrong.append((a, p, g, z))
    if wrong:
        txt = ("**Moves the property the wrong way at w = 4** (share of the gap below "
               "zero and MAE worse than unguided at z <= -3 -- the |z| >= 3 threshold FR5 "
               "applies to in-band, borrowed here for MAE): %s. "
               "The guided distribution ends further from the target than no guidance "
               "does." % "; ".join("%s on %s (%s, z %+.1f)" % (a, p, rt.pct(g), z)
                                   for a, p, g, z in wrong))
        vw = [p for a, p, _, _ in wrong if a == "btvg_var"]
        if vw:
            txt += (" For `btvg_var`, the variance-only term, this is the ablation result "
                    "on %s: without the mean term it pushes a tail target away." % (
                        " and ".join(vw)))
        out(txt)
        out("")


# ---------------------------------------------------------------- 7 force

def section7(out):
    out("## 7. Equal w is not equal force (mu, q50, n = 128)")
    out("")
    out("What a guidance step does to a trajectory depends on the correction actually "
        "applied -- after the score-to-velocity factor and the clip -- relative to the "
        "base flow-matching velocity: correction share = mean abs(C(G)) / abs(V) over "
        "guided steps. `strength_scale` normalises some arms so that w is meant to be "
        "comparable; this measures whether it is. tfg and dflow replace the sampler and "
        "have no additive correction, so they are absent. Measured by "
        "`proj1/scripts/force_share.py`.")
    out("")
    vals = {(a, w): force("mu", a, w) for a in FORCE_ARMS for w in FORCE_W}
    if all(v is None for v in vals.values()):
        out("*Not yet measured: run `python proj1/scripts/force_share.py`.*")
        out("")
        return
    out("| arm | " + " | ".join("w=%g" % w for w in FORCE_W) + " |")
    out("|---|" + "---|" * len(FORCE_W))
    for a in FORCE_ARMS:
        out("| %s | %s |" % (a, " | ".join("-" if vals[(a, w)] is None else rt.f(vals[(a, w)])
                                          for w in FORCE_W)))
    out("")
    at1 = [vals[(a, 1.0)] for a in FORCE_ARMS if vals[(a, 1.0)]]
    if len(at1) > 1:
        out("At w = 1 the strongest arm pushes %.1f times harder than the weakest. So "
            "comparing arms at one w compares different forces under one label, and FR3a, "
            "which froze each arm at its own strongest floor-clearing w, froze them at "
            "different forces. A controlled comparison has to fix the force, or the "
            "chemistry cost, not w." % (max(at1) / min(at1)))
        out("")


# ---------------------------------------------------------------- 8 ceiling

def section8(out, ALL, W, WM):
    out("## 8. Why in-band stays low: the centring ceiling (full run, dist)")
    out("")
    out("MAE mixes two different things: an output that is off-centre, and one that is "
        "centred but wide. They need different remedies. Split the residual f_B - target "
        "into its bias and its sd. A perfectly centred Gaussian of that sd scores "
        "P(abs(Z) <= delta / sd) in-band, however well it is aimed. So two separate "
        "questions: did guidance make the residual narrower (compare the residual-sd "
        "column with unguided's), and is the arm's in-band near the ceiling for its own "
        "width (last column)? The ratio in the last column is scale-free: a narrower "
        "residual raises the in-band and its ceiling together, so a ratio near 1 says the "
        "residual is Gaussian-like and aim is not the constraint, not that it failed to "
        "narrow.")
    out("")
    out("Floor convention, as in sections 1 and 2: a 3-seed row is judged against the "
        "pooled unguided, a one-seed row against that seed's own unguided.")
    out("")
    G = {p: groups(ALL, p) for p in rt.PROPS}
    R = ceiling_rows(ALL, W, WM)
    for p in rt.PROPS:
        dl = ALL[p]["ladder"]["delta"]
        out("### 8%s. %s (delta = %s)" % ("abc"[rt.PROPS.index(p)], p, rt.f(dl, 4)))
        out("")
        out("| arm | set | w | seeds | clears floor? | bias / delta | residual sd / delta "
            "| narrower than unguided | in-band | ceiling if centred | in-band / ceiling |")
        out("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in [r for r in R if r["p"] == p]:
            c = G[p][r["a"]][r["w"]]
            out("| %s | %s | %g | %d | %s | %+.2f | %.2f | %s | %s | %s | %.2f |" % (
                name(r["a"]), r["lab"], r["w"], len(c["seeds"]),
                "yes" if r["ok"] else "**no**", c["bias"] / dl, r["sd"],
                "-" if r["a"] == "unguided" else rt.pct(r["narrow"]),
                rt.f(c["in_band"]), rt.f(c["in_band_ceiling"]), r["ratio"]))
        out("")
    ok_r = [r for r in R if r["ok"]]
    ok_g = [r for r in ok_r if r["a"] != "unguided"]
    hi = [r for r in R if r["ratio"] > 1.25]
    edge = [r for r in R if r["ok"] != r["ok_pooled"]]
    out("**Does guidance narrow the output?** Partly. At strengths that clear the floor the "
        "residual sd falls against unguided by %s. The narrowest "
        "floor-clearing residual is %.2f delta. **Is the aim the constraint?** No: across "
        "the %d floor-clearing rows in-band is %.2f to %.2f times the Gaussian ceiling for "
        "its own width. **So what limits in-band is the width that remains.** Reaching "
        "in-band 0.50 would need a residual sd of about %.2f delta." % (
            narrow_text(R),
            min(r["sd"] for r in ok_g), len(ok_r), min(r["ratio"] for r in ok_r),
            max(r["ratio"] for r in ok_r), 1.0 / Z_HALF))
    out("")
    if edge:
        out("Knife-edge rows, whose floor verdict flips between the two conventions: %s. "
            "Under the pooled convention for every row, the floor-clearing range above "
            "would be %.2f to %.2f over %d rows." % (
                "; ".join("%s %s w=%g on %s (%s here, %s pooled)" % (
                    r["a"], r["lab"], r["w"], r["p"], "clears" if r["ok"] else "fails",
                    "clears" if r["ok_pooled"] else "fails") for r in edge),
                min(r["ratio"] for r in R if r["ok_pooled"]),
                max(r["ratio"] for r in R if r["ok_pooled"]),
                len([r for r in R if r["ok_pooled"]])))
        out("")
    if hi:
        out("The rows well above their ceiling (ratio > 1.25) are %s. Their residual is "
            "not only narrower but peaked, with more mass near zero than a Gaussian of the "
            "same width. %s" % (
                ", ".join("%s %s w=%g on %s (%.2f, residual sd %.2f delta)" % (
                    r["a"], r["lab"], r["w"], r["p"], r["ratio"], r["sd"]) for r in hi),
                "None of them clears the chemistry floor, so none is admissible under the "
                "rubric." if not any(r["ok"] for r in hi) else
                "Some clear the floor; read those as genuine concentration."))
        out("")
    out("This remaining width is the problem BTVG's variance term was designed to "
        "address, and it does not solve it.")
    out("")


# ---------------------------------------------------------------- 9 claims

def section9(out, ALL, W, WM, ours):
    G = {p: groups(ALL, p) for p in rt.PROPS}
    ung = [G[p]["unguided"][1.0]["partial_corr"] for p in rt.PROPS]
    strong = [G[p][a][float(WM[a][p])]["partial_corr"]
              for p in rt.PROPS for a in ("plug", "tmpd", "btvg")]
    closed = []
    for p in rt.PROPS:
        u = ours[(p, "unguided", "q90", 1.0)]
        for a in ("plug", "tmpd", "btvg"):
            r = ours.get((p, a, "q90", 4.0))
            if r:
                closed.append((r["f_B_mean"] - u["f_B_mean"]) / (u["target_mean"] - u["f_B_mean"]))
    out("## 9. What this page supports, and what it does not")
    out("")
    out("**Supported.**")
    out("")
    seeds = sorted({s for p in rt.PROPS for a in ("plug", "tmpd", "btvg")
                    for s in G[p][a][float(WM[a][p])]["seeds"]})
    out("1. Guidance makes generated molecules follow their own targets beyond what atom "
        "count explains: partial correlation %s to %s unguided, %s to %s for plug / tmpd "
        "/ btvg at the FR3 strength (section 5). Those FR3 rows are %d seed%s (%s); the "
        "3-seed FR3a rows show the same direction at the weaker frozen strengths." % (
            rt.f(min(ung)), rt.f(max(ung)), rt.f(min(strong), 2), rt.f(max(strong), 2),
            len(seeds), "" if len(seeds) == 1 else "s", ", ".join(str(x) for x in seeds)))
    out("2. On the q90 steering task, those arms at w = 4 close %s to %s of a tail target's "
        "gap, rising with strength, on every property (section 6)." % (
            rt.pct(min(closed)), rt.pct(max(closed))))
    R = ceiling_rows(ALL, W, WM)
    fail_full = [p for p in rt.PROPS
                 if any(r["p"] == p and r["lab"] == "FR3" and not r["ok"]
                        and r["a"] in ("plug", "tmpd", "btvg") for r in R)]
    fail_q90 = []
    for p in rt.PROPS:
        u = ours[(p, "unguided", "q90", 1.0)]
        fl = 0.9 * u["mol_stability"]
        if all(ours[(p, a, "q90", 4.0)]["mol_stability"] < fl - 1e-12
               for a in ("plug", "tmpd", "btvg")):
            fail_q90.append(p)
    out("3. The gain has a measured chemistry price. At w = 4 in the full run, plug / tmpd "
        "/ btvg include arms below the floor on %s (section 2); on the n = 512 q90 screen "
        "all three are below it at w = 4 on %s (section 6)." % (
            ", ".join(fail_full) or "no property", ", ".join(fail_q90) or "no property"))
    ok_r = [r for r in R if r["ok"]]
    ok_g = [r for r in ok_r if r["a"] != "unguided"]
    hi = [r for r in R if r["ratio"] > 1.25]
    out("4. Guidance narrows the output only partly: at floor-clearing strengths the "
        "residual sd falls by %s. It never gets below %.2f delta, while "
        "in-band 0.50 would need about %.2f delta. In-band is %.2f to %.2f times the "
        "Gaussian ceiling for its own width across %d floor-clearing rows, so the limit "
        "is the width that remains, not the aim (section 8).%s" % (
            narrow_text(R),
            min(r["sd"] for r in ok_g), 1.0 / Z_HALF,
            min(r["ratio"] for r in ok_r), max(r["ratio"] for r in ok_r), len(ok_r),
            (" The rows that get the residual down to a few delta (%s) are all below the "
             "floor." % "; ".join("%s on %s" % (a, ", ".join(r["p"] for r in hi if r["a"] == a))
                                  for a in sorted({r["a"] for r in hi})))
            if hi and not any(r["ok"] for r in hi) else ""))
    vw = []
    for p in rt.PROPS:
        u = ours[(p, "unguided", "q90", 1.0)]
        r = ours.get((p, "btvg_var", "q90", 4.0))
        if r:
            vw.append((p, (r["f_B_mean"] - u["f_B_mean"]) / (u["target_mean"] - u["f_B_mean"])))
    if vw:
        away = [p for p, g in vw if g < 0]
        toward = [p for p, g in vw if g >= 0]
        txt = ("5. The variance-only term (`btvg_var`) is not a reliable steer: at w = 4 on "
               "the q90 task it closes %s of the gap (mu / alpha / gap), against %s to %s "
               "for the arms that carry a mean term (claim 2)." % (
                   " / ".join(rt.pct(g) for _, g in vw), rt.pct(min(closed)),
                   rt.pct(max(closed))))
        if away:
            txt += (" On %s the share is negative: it pushes the property away from the "
                    "target." % " and ".join(away))
        if toward:
            txt += (" On %s it moves the right way, but by much less than the mean-term "
                    "arms." % " and ".join(toward))
        out(txt + " (section 6)")
    out("")
    out("**Not supported.**")
    out("")
    out("- Any ranking of arms at w = 4 where an arm is below the chemistry floor: under the "
        "rubric such an arm can neither win nor be beaten.")
    fr3a_beat, beat, beat_ok, beat_pooled = atoms_passes(ALL, W, WM)
    out("- Saying guidance beats EDM's #Atoms bar at an admissible strength. %s%s "
        "Partial correlation is our continuous form of EDM's criterion, not EDM's "
        "criterion." % (
            "No arm does at its FR3a strength (section 5)." if not fr3a_beat else
            "At the FR3a strength %d arm-property cells pass it (section 5)." % len(fr3a_beat),
            (" At FR3, %d cell%s (%s) pass%s it and clear the floor, but on one seed%s "
             "(read-this-first block)." % (
                 len(beat_ok), "" if len(beat_ok) == 1 else "s",
                 ", ".join("%s on %s" % (a, p) for p, a in beat_ok),
                 "es" if len(beat_ok) == 1 else "",
                 " and only under the one-seed floor convention"
                 if not set(beat_ok) & set(beat_pooled) else ""))
            if beat_ok else ""))
    out("- Comparing arms at equal w as a controlled comparison (section 7).")
    out("- tfg's in-band numbers without its decoded view beside them (sections 1 and 2); "
        "at high strength decoding removes much of its gain.")
    out("- Reading the n = 512 compare stage (section 6) as a verdict. It is the screen the "
        "strengths were frozen on.")
    out("")


def sections(out, ALL, W, WM, ours):
    section5(out, ALL, W, WM)
    section6(out, ours, W)
    section7(out)
    section8(out, ALL, W, WM)
    section9(out, ALL, W, WM, ours)
