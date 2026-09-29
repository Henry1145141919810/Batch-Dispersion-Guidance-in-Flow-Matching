"""Check v3 cells AS THEY LAND, so a systematic fault is caught at 50 cells
instead of at 756.

Everything here is a check that a COMPLETE run could still fail: the cells all
exist, every number is finite, and the table would build -- and yet the result
would be wrong or meaningless. Those are the faults worth a re-run, and they
are the ones you cannot see by watching the log scroll past.

    python proj1/scripts/v3_sanity.py                 # everything on disk
    python proj1/scripts/v3_sanity.py --stage v3 --backend fm
    python proj1/scripts/v3_sanity.py --quiet         # only FAIL/WARN lines

Exit code is 1 if anything FAILED, so it can gate a chain.
"""
import argparse
import collections
import glob
import json
import os

# The configuration v3 pre-registers. A cell that disagrees is not a v3 cell.
PINNED = {"n": 2000, "batch": 500, "steps": 100, "t_min_guide": 0.5,
          "target_name": "q50", "clip": 1.0, "solver": "euler"}
# Keyed by the LABEL a cell records (transfer_sweep.BACKENDS[flag]), not the
# flag. An unlisted label makes `want` None and the check SKIPS rather than
# fails, so a new backend is silently exempt until it is added here.
PAIR_BY_BACKEND = {"FM (ours)": "ours", "EquiFM": "tfg", "TFG/EDMsecond": "ours",
                   "VP diffusion (ours)": "ours"}

# delta = 2 x f_B's calibration MAE, measured held-out (V3_PAIR_DELTA.md). Pinned
# to 5 decimals because the whole cross-backend warning in the handoff rests on
# these: fm and edm are scored in OUR wider band, equifm in TFG's narrower one.
# A cell whose delta drifts is being scored against a different band than the
# handoff says, and no other column would show it.
DELTA = {("ours", "mu"): 0.17541, ("ours", "alpha"): 0.50618,
         ("ours", "gap"): 0.00748, ("tfg", "mu"): 0.15675,
         ("tfg", "alpha"): 0.16480, ("tfg", "gap"): 0.00374}

# FULL_RUN_V3_PROTOCOL section 2.1, measured on our base at q50/mu/n=512, single
# seed. Indicative only -- different n and seeds -- so this is an ORDER-OF-
# MAGNITUDE check, not an assertion. It exists to catch an arm that silently
# stopped steering, which is the failure that looks fine in every other column.
REF_MU_FM = {"unguided": 0.0723, "plug": 0.0840, "tmpd": 0.0957,
             "lgd_mc": 0.0859, "tfg": 0.3828}

# Run-to-run reproducibility, MEASURED 27 Sep by re-running one arm at its own
# seed and diffing against the original cell:
#
#   FM (ours)      bit-exact (the eta=0 control lands at exactly 0.000e+00)
#   EquiFM         NOT reproducible -- same arm, same seed differs by
#                  1.5e-3 in_band, 6.1e-3 prop_mae, 4.0e-3 f_A_mean
#
# So an exact-equality control is meaningful on fm and impossible on equifm, and
# a failure there of ~8e-3 is the backend's own noise rather than a defect. The
# tolerance below is that measured noise with headroom; it is NOT a way of
# waving a real difference through, which on these metrics would be far larger.
REPRO_TOL = {"FM (ours)": 0.0, "TFG/EDMsecond": 0.0, "EquiFM": 2e-2}

FAILS, WARNS, CHECKS = [], [], 0


def fail(c, msg):
    FAILS.append("%s: %s" % (c, msg))


def warn(c, msg):
    WARNS.append("%s: %s" % (c, msg))


def cellname(r):
    a = r["arm"] if r["arm"] != "bdg" else "bdg_" + str(r.get("variant", ""))
    return "%s/%s/%s/s%s" % (r.get("backend", "?"), r.get("prop", "?"), a,
                             r.get("seed", "?"))


def clipped_frac(r):
    g = r.get("guided_steps") or r.get("steps", 1)
    return r.get("clipped_sample_steps", 0) / float(max(1, g) * r.get("n", 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/v3")
    ap.add_argument("--stage", default="")
    ap.add_argument("--backend", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    global CHECKS

    paths = sorted(glob.glob(os.path.join(a.root, "*", "*", "n*", "seed*",
                                          "tr__*.json")))
    rows = []
    for p in paths:
        try:
            r = json.load(open(p))
        except Exception as exc:                        # noqa: BLE001
            fail(os.path.basename(p), "unreadable: %s" % exc)
            continue
        if a.stage and r.get("stage") != a.stage:
            continue
        if a.backend and a.backend.lower() not in str(r.get("backend")).lower():
            continue
        rows.append(r)
    if not rows:
        print("no cells under %s" % a.root)
        return 0

    # ---------------------------------------------------------- per cell
    for r in rows:
        c = cellname(r)
        for k, v in PINNED.items():
            CHECKS += 1
            if k in r and r[k] != v:
                fail(c, "%s is %r, v3 pre-registers %r" % (k, r[k], v))
        CHECKS += 1
        if r.get("arm") != "unguided" and r.get("w") != 1.0 \
                and r.get("stage") == "v3":
            fail(c, "w is %r; the v3 headline is w = 1 for every arm" % r.get("w"))
        CHECKS += 1
        want = PAIR_BY_BACKEND.get(str(r.get("backend")))
        if want and r.get("pair") != want:
            fail(c, "scored by pair %r; %s must use %r"
                 % (r.get("pair"), r.get("backend"), want))
        CHECKS += 1
        exp = DELTA.get((r.get("pair"), r.get("prop")))
        if exp is not None and abs(r.get("delta", 0) - exp) > 5e-5:
            fail(c, "delta %.5f, but pair %r on %r calibrates to %.5f"
                 % (r.get("delta", 0), r.get("pair"), r.get("prop"), exp))
        CHECKS += 1
        if r.get("n_nonfinite", 0):
            fail(c, "%d non-finite samples" % r["n_nonfinite"])
        CHECKS += 1
        cf = clipped_frac(r)
        if cf > 0.05:
            warn(c, "clip binds on %.1f%% of guided sample-steps; M1's v3 runs "
                    "under 1%%, and a clipped step carries no magnitude "
                    "information" % (100 * cf))
        CHECKS += 1
        if r.get("validity", 1) <= 0 or r.get("mol_stability", 1) <= 0:
            fail(c, "chemistry collapsed: validity %.3f mol_stability %.3f"
                 % (r.get("validity", -1), r.get("mol_stability", -1)))
        CHECKS += 1
        if r.get("in_band_fraction") is None:
            fail(c, "no in_band_fraction")

    # ------------------------------------------- within one (be,prop,seed)
    grp = collections.defaultdict(dict)
    for r in rows:
        a_ = r["arm"] if r["arm"] != "bdg" else "bdg_" + str(r.get("variant"))
        grp[(r.get("backend"), r.get("prop"), r.get("seed"))][a_] = r

    for key, arms in sorted(grp.items(), key=lambda kv: str(kv[0])):
        tag = "%s/%s/s%s" % key
        # Relative tolerance, for the same reason v3_table.py uses one: delta is
        # 2 x f_B's MAE reduced on the GPU and is not bit-reproducible between
        # tasks (equifm's agree to 8 significant figures). What this check exists
        # to catch is cells scored by different PAIRS, which differ by 12-207%.
        deltas = [float(r["delta"]) for r in arms.values() if r.get("delta")]
        CHECKS += 1
        if deltas and (max(deltas) - min(deltas)) > 1e-6 * max(deltas):
            fail(tag, "arms in one group have different delta: %s"
                 % sorted({round(d, 12) for d in deltas}))
        tgts = {r.get("target") for r in arms.values()}
        CHECKS += 1
        if len(tgts) > 1:
            fail(tag, "arms in one group have different target: %s" % sorted(tgts))

        ung = arms.get("unguided")
        for name, r in sorted(arms.items()):
            if name == "unguided" or ung is None:
                continue
            CHECKS += 1
            same_ib = r["in_band_fraction"] == ung["in_band_fraction"]
            same_mae = abs(r.get("prop_mae_eval", 0)
                           - ung.get("prop_mae_eval", 0)) < 1e-9
            if same_ib and same_mae:
                fail(tag, "arm %r is IDENTICAL to unguided -- it did not steer"
                     % name)
        # two arms identical to each other: one of them is not the method it says
        names = sorted(n for n in arms if n != "unguided")
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                # bdg_e0t1 IS plug by construction -- that is the pre-registered
                # control, checked with its own tolerance above. Warning that the
                # two are identical would flag the control passing as a fault.
                if {x, y} == {"bdg_e0t1", "plug"}:
                    continue
                CHECKS += 1
                if abs(arms[x]["in_band_fraction"]
                       - arms[y]["in_band_fraction"]) < 1e-12 and \
                   abs(arms[x].get("prop_mae_eval", 0)
                       - arms[y].get("prop_mae_eval", 0)) < 1e-12:
                    warn(tag, "arms %r and %r are numerically identical; one of "
                              "them is not running the method it is labelled as"
                         % (x, y))

    # ------------------- the pre-registered eta=0 control: bdg(eta=0) IS plug
    # It is BDG with the feedback switched off, so any difference beyond the
    # backend's own reproducibility is an implementation fault and every BDG
    # number is void. The ablation carries bdg_e0t1; plug lives in the headline,
    # so this reaches across stages deliberately.
    plug = {}
    for r in rows:
        if r.get("arm") == "plug" and abs(float(r.get("w", 0)) - 1.0) < 1e-9:
            plug[(r.get("backend"), r.get("prop"), r.get("seed"))] = r
    for r in rows:
        if r.get("arm") != "bdg_e0t1" or abs(float(r.get("w", 0)) - 1.0) > 1e-9:
            continue
        ref = plug.get((r.get("backend"), r.get("prop"), r.get("seed")))
        if not ref:
            continue
        CHECKS += 1
        tol = REPRO_TOL.get(str(r.get("backend")), 0.0)
        d = max(abs(r.get(m, 0) - ref.get(m, 0)) for m in
                ("in_band_fraction", "prop_mae_eval", "mol_stability", "validity"))
        if d > tol:
            fail(cellname(r), "CONTROL: bdg(eta=0) != plug, max|d| = %.3e > "
                              "tolerance %.1e for %s. BDG with the feedback off "
                              "must BE plug; every BDG number is void until this "
                              "holds." % (d, tol, r.get("backend")))

    # ------------------------------------------------ vs the reference table
    for (be, prop, seed), arms in sorted(grp.items(), key=lambda kv: str(kv[0])):
        if be != "FM (ours)" or prop != "mu":
            continue
        for name, ref in REF_MU_FM.items():
            if name not in arms:
                continue
            CHECKS += 1
            got = arms[name]["in_band_fraction"]
            if got > 0 and (got / ref > 4.0 or ref / max(got, 1e-9) > 4.0):
                warn("FM/mu/s%s" % seed,
                     "arm %r in_band %.4f vs the protocol's indicative %.4f "
                     "(>4x apart; different n and seed, so check rather than "
                     "assume)" % (name, got, ref))

    # -------------------------------------------------- seed-to-seed spread
    by_arm = collections.defaultdict(list)
    for r in rows:
        a_ = r["arm"] if r["arm"] != "bdg" else "bdg_" + str(r.get("variant"))
        by_arm[(r.get("backend"), r.get("prop"), a_)].append(
            (r.get("seed"), r["in_band_fraction"]))
    for k, vals in sorted(by_arm.items(), key=lambda kv: str(kv[0])):
        if len(vals) < 2:
            continue
        ib = [v for _, v in vals]
        m = sum(ib) / len(ib)
        spread = max(ib) - min(ib)
        CHECKS += 1
        # in_band's own se at n=2000 is ~1.1pp; a spread far beyond that across
        # seeds means the seeds are not measuring the same thing.
        if spread > max(0.10, 4 * 0.011) and spread > 0.5 * max(m, 1e-9):
            warn("%s/%s/%s" % k, "seed-to-seed in_band spread %.4f around mean "
                 "%.4f over %d seeds" % (spread, m, len(ib)))

    # ------------------------------------------------------------- report
    done = len(rows)
    groups = len(grp)
    if not a.quiet:
        print("%d cells, %d (backend,prop,seed) groups, %d checks"
              % (done, groups, CHECKS))
        seen = collections.Counter((r.get("backend"), r.get("stage")) for r in rows)
        for k in sorted(seen, key=str):
            print("   %-16s %-6s %d cells" % (k[0], k[1], seen[k]))
    for w in WARNS:
        print("WARN  %s" % w)
    for f in FAILS:
        print("FAIL  %s" % f)
    print("\n%s: %d fail, %d warn, %d checks"
          % ("PROBLEMS" if FAILS else "clean", len(FAILS), len(WARNS), CHECKS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
