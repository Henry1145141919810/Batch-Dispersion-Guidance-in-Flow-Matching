"""Check M2 cells as they land. The counterpart of proj1/scripts/v3_sanity.py.

Same principle: these are the faults a COMPLETE run can still have -- every
cell present, every number finite, the table builds -- and the result is wrong
anyway.

M2 carries three failure modes M1 does not, and each has bitten this stack
already (protocol sections 1.2 and 2.4):

  * a clip-saturated cell, where the clip rather than w sets the strength and
    every magnitude-only arm collapses onto plug;
  * a cell whose batch does not equal the pre-registered controller size, which
    silently changes what BDG's estimator is;
  * a cell run at a different window or NFE from its neighbours, which the old
    cell names could not even express.

    python proj1/m2/m2_sanity.py --stage m2
    python proj1/m2/m2_sanity.py --stage m2 --window 0.1 --quiet

Exit code 1 if anything FAILED, so a chain can gate on it.
"""
import argparse
import collections
import glob
import json
import os

PINNED = {"batch": 500, "steps": 400, "target_name": "q50", "clip": 1.0,
          "delta_ratio": 0.16}
# Corpus scales the checkpoint and the protocol both record.
S_BY_PROP = {"gc": 0.05521, "cpg": 0.01475}
FAILS, WARNS = [], []
CHECKS = 0


def fail(c, m):
    FAILS.append("%s: %s" % (c, m))


def warn(c, m):
    WARNS.append("%s: %s" % (c, m))


def name(r):
    a = r["arm"] if r["arm"] != "bdg" else "bdg_" + str(r.get("variant"))
    return "%s/%s/w%g/s%s" % (r.get("prop"), a, r.get("w", 0), r.get("seed"))


def clipped_frac(r):
    guided = max(1, int(round(r["steps"] * (1.0 - r.get("t_min_guide", 0.0)))))
    return r["clipped_sample_steps"] / float(guided * r["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/m2")
    ap.add_argument("--stage", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--window", type=float, default=None,
                    help="the t_min Rule W selected; cells must all use it")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    global CHECKS

    pat = os.path.join(a.out_dir, a.stage or "*", "n*" if not a.n else "n%d" % a.n,
                       "seed*", "*.json")
    rows = []
    for p in sorted(glob.glob(pat)):
        if p.endswith(".permol.pt"):
            continue
        try:
            rows.append(json.load(open(p)))
        except Exception as exc:                          # noqa: BLE001
            fail(os.path.basename(p), "unreadable: %s" % exc)
    if not rows:
        print("no cells under %s" % pat)
        return 0

    for r in rows:
        c = name(r)
        for k, v in PINNED.items():
            CHECKS += 1
            if k in r and r[k] != v:
                fail(c, "%s is %r, the protocol pre-registers %r" % (k, r[k], v))
        CHECKS += 1
        if r["n"] % r["batch"]:
            fail(c, "batch %d does not divide n %d -- a remainder controller was "
                    "pooled in as an equal" % (r["batch"], r["n"]))
        CHECKS += 1
        if r.get("n_controllers") != r["n"] // r["batch"]:
            fail(c, "n_controllers %r disagrees with n/batch = %d"
                 % (r.get("n_controllers"), r["n"] // r["batch"]))
        CHECKS += 1
        if a.window is not None and abs(r.get("t_min_guide", -1) - a.window) > 1e-9:
            fail(c, "t_min %r, but Rule W selected %g"
                 % (r.get("t_min_guide"), a.window))
        CHECKS += 1
        s_exp = S_BY_PROP.get(r.get("prop"))
        if s_exp and abs(r.get("s", 0) - s_exp) > 5e-5:
            fail(c, "corpus sd s=%.5f, expected %.5f for %r -- the wrong corpus "
                    "or the wrong checkpoint" % (r.get("s", 0), s_exp, r["prop"]))
        CHECKS += 1
        # delta is the band; it must follow the protocol's own formula exactly.
        q = 1.0 / (500 if r["prop"] == "gc" else 499)
        exp_d = max(r.get("delta_ratio", 0.16) * r.get("s", 0), 4.4 * q)
        if abs(r.get("delta", 0) - exp_d) > 1e-6:
            fail(c, "delta %.6f, formula gives %.6f" % (r.get("delta", 0), exp_d))
        CHECKS += 1
        for k in ("in_band_fraction", "gc_mean", "gc_sd", "kmer_js", "decode_conf"):
            v = r.get(k)
            if v is None or v != v:
                fail(c, "%s is %r" % (k, v))
        CHECKS += 1
        if not (0.0 <= r.get("decode_conf", -1) <= 1.0):
            fail(c, "decode_conf %r outside [0,1]" % r.get("decode_conf"))
        CHECKS += 1
        cf = clipped_frac(r)
        if cf > 0.5:
            warn(c, "clip binds on %.0f%% of guided sample-steps: the clip, not "
                    "w, is setting the strength here (protocol 1.2b)" % (100 * cf))

    # ---------------------------------------------- within (prop, w, seed)
    grp = collections.defaultdict(dict)
    for r in rows:
        a_ = r["arm"] if r["arm"] != "bdg" else "bdg_" + str(r.get("variant"))
        grp[(r["prop"], r["w"], r["seed"])][a_] = r
    for k, arms in sorted(grp.items(), key=str):
        tag = "%s/w%g/s%s" % k
        for f in ("delta", "y", "s", "t_min_guide", "n", "batch", "steps"):
            CHECKS += 1
            vals = {r.get(f) for r in arms.values()}
            if len(vals) > 1:
                fail(tag, "arms disagree on %s: %s" % (f, sorted(map(str, vals))))
        # the pre-registered control: bdg at eta=0 IS plug
        CHECKS += 1
        if "bdg_e0t1" in arms and "plug" in arms:
            d = max(abs(arms["bdg_e0t1"][m] - arms["plug"][m])
                    for m in ("in_band_fraction", "gc_mean", "gc_sd", "kmer_js"))
            if d != 0.0:
                fail(tag, "CONTROL FAILED: bdg(eta=0) != plug, max|d| = %.3e. "
                          "Every BDG number is void until this holds." % d)
        ung = arms.get("unguided")
        for nm, r in sorted(arms.items()):
            if nm == "unguided" or ung is None:
                continue
            CHECKS += 1
            if r["in_band_fraction"] == ung["in_band_fraction"] and \
               abs(r["gc_mean"] - ung["gc_mean"]) < 1e-12:
                fail(tag, "arm %r is IDENTICAL to unguided -- it did not steer" % nm)

    done = len(rows)
    if not a.quiet:
        print("%d cells, %d (prop,w,seed) groups, %d checks" % (done, len(grp), CHECKS))
        c = collections.Counter((r.get("prop"), r.get("stage")) for r in rows)
        for k in sorted(c, key=str):
            print("   %-5s %-8s %d cells" % (k[0], k[1], c[k]))
    for w in WARNS:
        print("WARN  %s" % w)
    for f in FAILS:
        print("FAIL  %s" % f)
    print("\n%s: %d fail, %d warn, %d checks"
          % ("PROBLEMS" if FAILS else "clean", len(FAILS), len(WARNS), CHECKS))
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
