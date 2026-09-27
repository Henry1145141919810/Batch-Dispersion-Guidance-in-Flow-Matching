"""Read Modality 2 cells into a table. The counterpart of proj1/scripts/v3_table.py.

It REFUSES rather than averaging across a configuration that changed, because
every one of those fields changes what the number means:

    n, batch, steps (NFE), t_min, delta, delta_ratio, target, ckpt

and, in the ablation, `w` -- which is the axis the ablation exists to measure,
so pooling w = 1 and w = 4 would average away the experiment.

It also prints no verdict. Protocol section 3.1: there is no fidelity floor, so
an in_band leaderboard would crown the arm that destroyed the most sequence
fidelity. Every in_band figure prints beside kmer_js, decode_conf and diversity,
and the summary names the best arm that did NOT lose fidelity against unguided.

Run:
  python proj1/m2/m2_table.py --stage m2    --n 2000
  python proj1/m2/m2_table.py --stage m2abl --n 2000 --w 16
"""
import argparse
import glob
import json
import math
import os

# Fields that must agree across every cell in one table, with the reason they
# are here rather than in a footnote.
PINNED = ["n", "batch", "steps", "t_min_guide", "delta", "delta_ratio",
          "target_name", "ckpt"]
FIDELITY = ["kmer_js", "decode_conf", "diversity"]


def load(out_dir, stage, n, seeds, prop=None):
    pat = os.path.join(out_dir, stage, "n%d" % n, "seed*", "*.json")
    rows = []
    for p in sorted(glob.glob(pat)):
        r = json.load(open(p))
        if seeds and r["seed"] not in seeds:
            continue
        if prop and r["prop"] != prop:
            continue
        r["_path"] = p
        rows.append(r)
    return rows


def check_pinned(rows):
    """Refuse a table built from cells that do not share a configuration."""
    bad = []
    for f in PINNED:
        vals = {r.get(f) for r in rows}
        if len(vals) > 1:
            bad.append("%s: %s" % (f, sorted(map(str, vals))))
    if bad:
        raise SystemExit(
            "REFUSING to build one table from cells whose configuration "
            "disagrees.\nEach of these changes what the number means, so "
            "averaging them is not\na summary, it is a different quantity:\n  "
            + "\n  ".join(bad))


def arm_key(r):
    return r["arm"] if r["arm"] != "bdg" else "bdg_" + r["variant"]


def clipped_frac(r):
    """Clipped sample-steps as a fraction of the GUIDED ones. Guidance runs only
    for t >= t_min, so dividing by n*steps would understate it at a late window
    and make two windows incomparable."""
    guided = max(1, int(round(r["steps"] * (1.0 - r.get("t_min_guide", 0.0)))))
    return r["clipped_sample_steps"] / float(guided * r["n"])


def mean_se(xs):
    """Mean and the SEED-TO-SEED standard error -- not the within-cell one, so
    it captures seed variance rather than assuming samples are the only noise."""
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(v / len(xs))


def controls(groups, w):
    """The two pre-registered gates of protocol section 4.1. A failure voids the
    BDG rows; it is not a result."""
    out = []
    plug = groups.get(("plug", w))
    e0 = groups.get(("bdg_e0t1", w))
    if plug and e0:
        d = max(abs(mean_se([r[k] for r in plug])[0]
                    - mean_se([r[k] for r in e0])[0])
                for k in ["in_band_fraction", "gc_mean", "gc_sd"] + FIDELITY)
        out.append(("bdg eta=0 == plug", d == 0.0,
                    "max|d| = %.3e over 6 metrics" % d))
    return out


def table(rows, w, prop):
    groups = {}
    for r in rows:
        if r["arm"] != "unguided" and w is not None and r["w"] != w:
            continue
        # unguided is w-INDEPENDENT by definition -- no guidance field is applied
        # at all -- so its cells are keyed at w=None and collapse into one group.
        # Keying them by their nominal w printed the same reference row once per
        # strength that had been run, which reads as two different baselines.
        key = (arm_key(r), None if r["arm"] == "unguided" else r["w"])
        groups.setdefault(key, []).append(r)

    ung = groups.get(("unguided", None))
    ung_js = mean_se([r["kmer_js"] for r in ung])[0] if ung else None

    print("\n=== %s   w=%s   (%d arms, %d seeds each)"
          % (prop, w, len(groups), len(next(iter(groups.values())))))
    print("%-16s %-17s %-9s %-9s %-8s %-8s %s"
          % ("arm", "in_band", "gc_sd", "bias/d", "kmerJS", "conf", "clipped"))
    best_any, best_clean = None, None
    for k in sorted(groups):
        rs = groups[k]
        ib, se = mean_se([r["in_band_fraction"] for r in rs])
        sd = mean_se([r["gc_sd"] for r in rs])[0]
        bd = mean_se([r["bias_delta"] for r in rs])[0]
        kj = mean_se([r["kmer_js"] for r in rs])[0]
        cf = mean_se([r["decode_conf"] for r in rs])[0]
        # Clip saturation as a FRACTION of the guided sample-steps, which is what
        # makes it comparable across arms and windows. Protocol section 1.2:
        # tmpd modulates per-sample magnitude and the clip caps per-sample
        # magnitude, so a heavily clipped tmpd row is plug by construction and
        # must be reported as clip-limited rather than as a baseline that
        # happened to match DPS.
        cl = mean_se([clipped_frac(r) for r in rs])[0]
        tag = " [clip-limited]" if cl > 0.5 else ""
        print("%-16s %.4f +/- %.4f  %.5f   %+8.3f  %.5f  %.3f  %5.1f%%%s"
              % (k[0], ib, se, sd, bd, kj, cf, 100 * cl, tag))
        # unguided is the REFERENCE, not a candidate: "the best arm is no
        # guidance" is not a statement about a guidance method.
        if k[0] == "unguided":
            continue
        if best_any is None or ib > best_any[1]:
            best_any = (k[0], ib, kj)
        if ung_js is not None and kj <= ung_js and (
                best_clean is None or ib > best_clean[1]):
            best_clean = (k[0], ib, kj)
    if best_any:
        print("  best in_band          : %s at %.4f (kmerJS %.5f vs unguided %.5f)"
              % (best_any[0], best_any[1], best_any[2], ung_js or float("nan")))
    if best_clean:
        print("  best without losing fidelity: %s at %.4f (kmerJS %.5f)"
              % (best_clean[0], best_clean[1], best_clean[2]))
    else:
        print("  best without losing fidelity: NONE -- every arm scored worse "
              "kmerJS than unguided")
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/m2")
    ap.add_argument("--stage", default="m2", choices=["m2", "m2abl"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--props", default="gc,cpg")
    ap.add_argument("--md-out", default="")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    all_rows = load(a.out_dir, a.stage, a.n, seeds)
    if not all_rows:
        raise SystemExit("no %s cells under %s/%s/n%d"
                         % (a.stage, a.out_dir, a.stage, a.n))
    ws = sorted({r["w"] for r in all_rows if r["arm"] != "unguided"})
    if a.stage == "m2abl" and a.w is None and len(ws) > 1:
        raise SystemExit(
            "REFUSING to pool the ablation's strengths: this tree holds w = %s.\n"
            "w is the axis the ablation exists to measure -- averaging the rungs "
            "would\naverage away the experiment. Pass --w with one of them."
            % ", ".join("%g" % x for x in ws))
    w = a.w if a.w is not None else (ws[0] if ws else None)

    for prop in [p for p in a.props.split(",") if p.strip()]:
        rows = [r for r in all_rows if r["prop"] == prop]
        if not rows:
            print("\n=== %s: no cells" % prop)
            continue
        check_pinned(rows)
        groups = table(rows, w, prop)
        for name, ok, detail in controls(groups, w):
            print("  CONTROL %-22s %s   (%s)"
                  % (name, "PASS" if ok else "**FAIL -- BDG rows are void**",
                     detail))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
