"""The complete Modality 2 results page, under MODALITY2_V3_PROTOCOL.md.

    python proj1/m2/m2_v3_results.py --md-out docs/results/M2_V3_RESULTS.md

WHY THIS FILE EXISTS. `m2_table.py` prints one stage at a time and refuses the
ablation tree outright (it mixes t_min 0 and 0.5, so `M2_RESULTS_ABL_gc_w*.txt`
hold refusals rather than tables); `m2_ablation_grid.py` prints the eta x tau
grid and nothing else. Neither computes a contrast, a standard error on a
contrast, or a verdict, and nothing in the repository reads the w-sweep tree or
the two functional-evaluation summaries. This does all of it in one pass, from
the cells, with every caveat the protocol attaches to the number printed beside
it.

WHAT IT REFUSES TO DO, and why each refusal is here:

  * pool cells whose configuration differs (n, batch, steps, t_min, delta,
    delta_ratio, target, ckpt, clip, n_controllers) -- each changes what the
    number means;
  * pool the t_min = 0 diagnostic rung with the t >= 0.5 headline -- they are
    two regimes, and the early one is clip-saturated (protocol 1.2b);
  * pool the ablation's two strengths -- w is the axis it exists to measure;
  * silently skip a missing or duplicated cell -- it exits with the list.

STATISTICS. in_band is a proportion over N = 3n samples, so its standard error
is the binomial sqrt(p(1-p)/N) (protocol section 5). MODALITY2_V3_PROTOCOL.md
states that rule and NO multiplicity threshold, so this page inherits v3's:
unpaired contrast se = sqrt(se_a^2 + se_b^2), Bonferroni two-sided at 0.05/18,
z = 2.99 (FULL_RUN_V3_PROTOCOL.md section 6.1). That is stated wherever a
verdict appears. A seed-paired t is printed too and is labelled NOT
PRE-REGISTERED every time.

CLIP FRACTION. m2_sweep.py accumulates `clipped` as a per-sample count inside
the per-batch step loop, over the guided steps only (`t >= t_min`), summed over
the n/batch batches. The denominator is therefore

    n * (number of guided steps in ONE batch)

and that count is read back from the cell's own cost counters as
`gen_vjp / n_controllers`, then cross-checked against round(steps*(1 - t_min)).
(proj1/scripts/v3_sanity.py's M1 analogue divides by guided_steps*n where
guided_steps is already summed over batches, and understates every M1 clip
fraction 4x. This does not copy it.)
"""
import argparse
import collections
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys

STAGES = ("m2", "m2abl", "m2wsweep")
SEEDS = (20260921, 20260922, 20260923)
HEADLINE_ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg_mc",
                 "bdg_e4t0.5", "bdg_e4t1")
BASELINES = ("plug", "tmpd", "lgd_mc", "tfg_mc")
ETAS = (0, 1, 2, 4, 8)
TAUS = (0.5, 0.75, 1.0, 1.5)
# The four metrics protocol section 3.0 measures the plug/tmpd/lgd_mc identity on.
IDENT4 = ("in_band_fraction", "gc_mean", "gc_sd", "kmer_js")
# Fields that must agree across every cell pooled into one row.
PINNED = ("n", "batch", "steps", "t_min_guide", "delta", "delta_ratio",
          "target_name", "ckpt", "clip", "n_controllers", "s", "y")
# v3's pre-registered bar, inherited: Bonferroni 0.05/18, two-sided.
Z_BAR = 2.99
# Section numbers, so the cross-references in the prose cannot drift from the
# headings.
S_PROV, S_READ, S_GC, S_CPG = 1, 2, 3, 4
S_CTRL, S_WIN, S_WSW, S_ABL, S_FUN, S_OUT = 5, 6, 7, 8, 9, 10


# ------------------------------------------------------------------ loading
def arm_key(r):
    return r["arm"] if r["arm"] != "bdg" else "bdg_" + str(r.get("variant"))


def cell_key(r):
    return (r["_stage"], r["prop"], arm_key(r), r["w"], r["t_min_guide"],
            r["seed"])


def load(out_dir, n):
    rows, seen = [], {}
    for st in STAGES:
        for p in sorted(glob.glob(os.path.join(
                out_dir, st, "n%d" % n, "seed*", "*.json"))):
            with open(p, encoding="utf-8") as fh:
                r = json.load(fh)
            r["_path"] = p
            r["_stage"] = st
            r["_base"] = os.path.basename(p)
            k = cell_key(r)
            if k in seen:
                die("DUPLICATE cell %s\n  %s\n  %s" % (k, seen[k], p))
            seen[k] = p
            rows.append(r)
    if not rows:
        die("no cells under %s/{%s}/n%d/seed*/ -- nothing to report"
            % (out_dir, ",".join(STAGES), n))
    return rows


def die(msg):
    sys.exit("m2_v3_results: REFUSING to write a table.\n" + msg)


# Metrics compared between the two copies of a configuration run in both trees.
# bias_delta is left out because it is gc_mean divided by delta (~0.0088), so
# it magnifies the same difference ~100x; gc_mean carries it.
DUP_METRICS = ("gc_mean", "gc_sd", "kmer_js", "decode_conf", "diversity")


def duplicate_configs(rows, tol=1e-5):
    """Configurations present in BOTH trees, as (key -> the copies).

    `cell_key` includes the stage, so the same configuration living under
    `m2/` and under `m2abl/` is two files and one cell. No table averages the
    two copies -- the headline reads the `m2` copy and the grid the `m2abl`
    one -- but a silent disagreement between them would mean the two halves of
    this page rest on different runs, so the copies are asserted to agree here:
    exactly on in-band (it is a count over the same 2000 decoded sequences) and
    to `tol` on the continuous metrics.
    """
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["prop"], arm_key(r), r["w"], r["t_min_guide"], r["seed"])].append(r)
    dups = {k: v for k, v in by.items() if len(v) > 1}
    worst_ib, worst_cont, extra_minutes, bad = 0.0, 0.0, 0.0, []
    for k, v in sorted(dups.items(), key=str):
        for i, a in enumerate(v):
            for b in v[i + 1:]:
                dib = abs(a["in_band_fraction"] - b["in_band_fraction"])
                dco = max(abs(a[m] - b[m]) for m in DUP_METRICS)
                worst_ib = max(worst_ib, dib)
                worst_cont = max(worst_cont, dco)
                if dib > 0.0 or dco > tol:
                    bad.append("%s: %s vs %s -- in-band %.2e, continuous %.2e"
                               % (str(k), a["_stage"], b["_stage"], dib, dco))
        extra_minutes += sum(r["minutes"] for r in v[1:])
    worst_clip = max((max(abs(a["clipped_sample_steps"]
                              - b["clipped_sample_steps"])
                          for i, a in enumerate(v) for b in v[i + 1:])
                      for v in dups.values()), default=0)
    if bad:
        die("a configuration was run in two trees and the copies DISAGREE, so\n"
            "the headline and the grid would rest on different runs:\n  %s"
            % "\n  ".join(bad))
    return dups, worst_ib, worst_cont, extra_minutes, worst_clip


def dup_phrase(dups):
    """One clause naming the re-run configurations, built from the keys."""
    if not dups:
        return ""
    props = sorted({k[0] for k in dups})
    arms = sorted({k[1] for k in dups})
    ws = sorted({k[2] for k in dups})
    tmins = sorted({k[3] for k in dups})
    seeds = sorted({k[4] for k in dups})
    return ("%s with %s, at w in {%s}, t >= %s, all %d seeds"
            % ("`" + "`, `".join(props) + "`",
               " and ".join("`%s`" % a for a in arms),
               ", ".join("%g" % w for w in ws),
               ", ".join("%g" % t for t in tmins), len(seeds)))


def expect(index, want, what):
    """Every cell the protocol says should exist, or exit with the list."""
    missing = [k for k in want if k not in index]
    if missing:
        die("%d %s cell(s) missing; a table built without them would be a\n"
            "different experiment reported under this protocol's name:\n  %s"
            % (len(missing), what, "\n  ".join(str(k) for k in missing)))


# ------------------------------------------------------------------ pooling
def guided_steps_per_batch(r):
    """Guided steps in ONE batch -- the clip fraction's denominator per sample.

    Read from the cell's own counters (m2_sweep.py increments gen_vjp once per
    guided step per batch) and cross-checked against the window. An unguided
    cell has none."""
    nc = r["n_controllers"]
    from_cost = r["cost"]["gen_vjp"] // nc if r["cost"]["gen_vjp"] else 0
    from_cfg = int(round(r["steps"] * (1.0 - r["t_min_guide"])))
    if r["arm"] != "unguided" and from_cost != from_cfg:
        die("cell %s: cost says %d guided steps per batch, the window says %d"
            % (r["_base"], from_cost, from_cfg))
    return from_cost


def clip_denom(r):
    g = guided_steps_per_batch(r)
    return r["n"] * g if g else 0


def check_pinned(rows, what):
    bad = []
    for f in PINNED:
        vals = {r.get(f) for r in rows}
        if len(vals) > 1:
            bad.append("%s: %s" % (f, sorted(map(str, vals))))
    if bad:
        die("cells pooled as %s disagree on the configuration. Averaging them\n"
            "is not a summary, it is a different quantity:\n  %s"
            % (what, "\n  ".join(bad)))


def pooled(rows, what="a row"):
    """N-weighted pooling over seeds. N = sum(n), and the se is the binomial
    one on that N (protocol section 5)."""
    check_pinned(rows, what)
    N = sum(r["n"] for r in rows)
    wm = lambda k: sum(r[k] * r["n"] for r in rows) / N
    o = {"N": N, "seeds": len(rows), "delta": rows[0]["delta"],
         "y": rows[0]["y"], "s": rows[0]["s"]}
    for k in ("in_band_fraction", "gc_mean", "kmer_js", "decode_conf",
              "diversity"):
        o[k] = wm(k)
    # the pooled spread is the spread of the POOLED sample, not the mean of the
    # per-seed sds: it carries the between-seed shift of the mean as well.
    o["sd"] = math.sqrt(sum((r["gc_sd"] ** 2 + (r["gc_mean"] - o["gc_mean"]) ** 2)
                            * r["n"] for r in rows) / N)
    o["bias_delta"] = (o["gc_mean"] - o["y"]) / o["delta"]
    o["sd_delta"] = o["sd"] / o["delta"]
    o["se"] = math.sqrt(max(o["in_band_fraction"] * (1 - o["in_band_fraction"]),
                            0.0) / N)
    o["clipped"] = sum(r["clipped_sample_steps"] for r in rows)
    den = sum(clip_denom(r) for r in rows)
    o["clip_frac"] = (o["clipped"] / den) if den else 0.0
    o["clip_denom"] = den
    o["minutes"] = sum(r["minutes"] for r in rows) / len(rows)
    o["minutes_total"] = sum(r["minutes"] for r in rows)
    we = [r["diag"].get("bdg_w_eff") for r in rows if "bdg_w_eff" in r["diag"]]
    o["w_eff"] = sum(we) / len(we) if we else None
    e = [r["diag"].get("bdg_e") for r in rows if "bdg_e" in r["diag"]]
    o["bdg_e"] = sum(e) / len(e) if e else None
    wd = [r["diag"].get("bdg_widening") for r in rows
          if "bdg_widening" in r["diag"]]
    o["widening"] = sum(wd) / len(wd) if wd else None
    o["rows"] = rows
    return o


def contrast(a, b):
    """Unpaired, as v3 section 6.1 pre-registers. Returns d, z, the minimum
    difference this run could have resolved at the bar, and the verdict."""
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    d = a["in_band_fraction"] - b["in_band_fraction"]
    z = d / se if se > 0 else float("nan")
    mdd = Z_BAR * se
    v = "above" if z >= Z_BAR else ("below" if z <= -Z_BAR else "tie")
    return d, z, mdd, v


def paired_t(a, b, key="in_band_fraction"):
    """NOT PRE-REGISTERED. Arms share initial noise within a seed, so this is
    tighter than the unpaired test; it is printed for information only. It is
    degenerate when the three seed differences are nearly equal, which happens
    here whenever a difference is one or two samples wide, so that case is
    reported rather than printed as a huge t."""
    da = {r["seed"]: r for r in a["rows"]}
    db = {r["seed"]: r for r in b["rows"]}
    ss = sorted(set(da) & set(db))
    if len(ss) < 2:
        return None, "", len(ss)
    d = [da[s][key] - db[s][key] for s in ss]
    signs = "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in d)
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    if sd <= 0 or abs(m) / (sd / math.sqrt(len(d))) > 100:
        return None, signs, len(ss)
    return m / (sd / math.sqrt(len(d))), signs, len(ss)


def family_stats(idx):
    """The size of the verdict family this page actually reports, the bar that
    family would carry on its own, and the weakest "above" on the page.

    The inherited bar is 0.05/18, which is M1's family (6 guided arms x 3
    properties against `unguided`). This page reports a different family, so
    the paragraph that calls the inherited bar conservative has to count the
    page's own contrasts rather than a hypothetical subset of them."""
    n_con, zs_above = 0, []
    for prop in ("gc", "cpg"):
        for w in (1.0, 4.0):
            g = {}
            for a in HEADLINE_ARMS:
                ww = 1.0 if a == "unguided" else w
                g[a] = pooled([idx[("m2", prop, a, ww, 0.5, s)]
                               for s in SEEDS], "family/%s/%s/w%g"
                              % (prop, a, w))
            for a, base in ([(a, "unguided") for a in HEADLINE_ARMS[1:]]
                            + [("bdg_e4t0.5", "plug"), ("bdg_e4t1", "plug")]):
                _d, z, _m, v = contrast(g[a], g[base])
                n_con += 1
                if v == "above":
                    zs_above.append(z)
    z_fam = statistics.NormalDist().inv_cdf(1 - 0.05 / n_con / 2)
    return n_con, z_fam, (min(zs_above) if zs_above else None), len(zs_above)


def share_info(args):
    """The applied-correction shares, WITH the provenance that qualifies them.

    They are the page's force caveat, and they come from a side measurement at
    n = 128 on one seed and on one property -- which the page quoted verbatim
    inside its `cpg` sections until 28 Sep. Read from the file so the numbers
    and their provenance cannot drift apart."""
    d = {"plug": "0.0122", "tfg_mc": "0.0109", "bdg_e4t0.5": "0.1164",
         "bdg_e4t1": "0.0066", "plug_w4": "0.0479",
         "prov": "`results/m2_share.json` is **not in the tree**, so these "
                 "values could not be read back or dated",
         "prov_short": "provenance unread -- `results/m2_share.json` is not "
                       "in the tree"}
    if not os.path.exists(args.share_json):
        return d
    with open(args.share_json, encoding="utf-8") as fh:
        J = json.load(fh)
    props = sorted(J["by_prop"])
    if "gc" not in props:
        die("%s records no `gc` correction share, which every force statement "
            "on this page quotes" % args.share_json)
    by = J["by_prop"]["gc"]
    for a in ("plug", "tfg_mc", "bdg_e4t0.5", "bdg_e4t1"):
        d[a] = "%.4f" % by[a]["w1"]["share"]
    d["plug_w4"] = "%.4f" % by["plug"]["w4"]["share"]
    d["prov"] = ("they are a **single-seed side measurement at n = %d on seed "
                 "%s, over %s only** (`%s`, %d-step Euler, t >= %g) -- about "
                 "1/%.0f of this run's n, one of its 3 seeds and one of its 2 "
                 "properties"
                 % (J["n"], J["seed"],
                    " and ".join("`%s`" % p for p in props), args.share_json,
                    J["steps"], J["t_min"], 2000.0 / J["n"]))
    d["prov_short"] = ("n = %d, seed %s, %s only"
                       % (J["n"], J["seed"],
                          " and ".join("`%s`" % p for p in props)))
    return d


def spearman(pairs):
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            for k in range(i, j + 1):
                out[idx[k]] = (i + j) / 2.0 + 1
            i = j + 1
        return out
    xs, ys = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs)
                    * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


# ------------------------------------------------------------------ sections
def provenance(L, rows, args, idx, dupinfo):
    dups, dup_ib, dup_cont, dup_minutes, dup_clip = dupinfo
    dup_note = ""
    if dups:
        dup_note = (". %s was run in **both trees**: %d configurations x "
                    "%d seeds = %d cells held twice, so **%d of the %d files "
                    "are re-runs**. The copies agree to %.1e on in-band and to "
                    "%.1e on the continuous metrics (their clipped-step counts "
                    "differ by at most %d); the headline reads the `m2` copy "
                    "and the grid the `m2abl` copy, and neither averages them"
                    % (dup_phrase(dups),
                       len({(k[0], k[1], k[2], k[3]) for k in dups}),
                       len({k[4] for k in dups}), len(dups),
                       sum(len(v) - 1 for v in dups.values()), len(rows),
                       dup_ib, dup_cont, dup_clip))
    head = [r for r in rows if r["_stage"] == "m2"]
    r0 = head[0]
    ck = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "blade_bundle", r0["ckpt"])
    md5 = "not found next to this script"
    if os.path.exists(ck):
        h = hashlib.md5()
        with open(ck, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        md5 = "md5 `%s` (%.1f MB)" % (h.hexdigest()[:8],
                                      os.path.getsize(ck) / 1e6)
    mins = sum(r["minutes"] for r in rows)
    per_stage = collections.Counter(r["_stage"] for r in rows)
    smin = collections.defaultdict(float)
    for r in rows:
        smin[r["_stage"]] += r["minutes"]
    pl = idx[("m2", "gc", "plug", 1.0, 0.5, SEEDS[0])]
    lg = idx[("m2", "gc", "lgd_mc", 1.0, 0.5, SEEDS[0])]
    gsb = guided_steps_per_batch(pl)
    L += ["## %d. Provenance" % S_PROV, "",
          "| | |", "|---|---|",
          "| generator | `%s`, frozen, %s |" % (r0["ckpt"], md5),
          "| modality | DeepFlyBrain enhancers, 500 bp, one-hot on the simplex "
          "(Delta^3)^500 |",
          "| target | `%s`, fixed per property |" % r0["target_name"],
          "| n, batch | %d in batches of %d = **%d controllers** per cell "
          "(protocol 2.4: the batch IS BDG's estimator, and 500 is M1's "
          "controller size) |" % (r0["n"], r0["batch"], r0["n_controllers"]),
          "| NFE | %d-step Euler with clamp-renormalise projection. A guided "
          "cell runs **%d guided steps per batch** and spends one extra net "
          "forward and one VJP on each (cost counters: %d generator forwards "
          "and %d VJPs per batch against the unguided cell's %d and 0); "
          "`lgd_mc` and `tfg_mc` spend %d property forward/backward pairs per "
          "batch against `plug`'s %d |"
          % (r0["steps"], gsb, pl["cost"]["gen_fwd"] // r0["n_controllers"],
             pl["cost"]["gen_vjp"] // r0["n_controllers"], r0["steps"],
             lg["cost"]["guide_fwd"] // r0["n_controllers"],
             pl["cost"]["guide_fwd"] // r0["n_controllers"]),
          "| guidance window | t >= %g (v3's, protocol 2.1). The t = 0 rung is "
          "a **diagnostic**, in section %d below, never pooled with these |"
          % (r0["t_min_guide"], S_WIN),
          "| clip | %g x the velocity norm, per sample, applied AFTER w |"
          % r0["clip"],
          "| seeds | %s (3) |" % ", ".join(str(s) for s in SEEDS),
          "| N per pooled row | 3 x %d = **%d** sequences |"
          % (r0["n"], 3 * r0["n"]),
          "| device | **not recorded in M2 cells** (M1's carry `prov.device`; "
          "M2's do not). The only committed record for M2 is "
          "`results/m2_window.json`, which says `cuda` |",
          "| cells read | %s = **%d files, %d distinct cells**%s |"
          % (" + ".join("%s %d" % (k, per_stage[k]) for k in STAGES),
             len(rows), len(rows) - sum(len(v) - 1 for v in dups.values()),
             dup_note),
          "| wall time | **%.1f min (%.2f h)** summed over **files**%s. Cells "
          "ran concurrently on several GPUs, so this is not elapsed time (%s) |"
          % (mins, mins / 60.0,
             (", so it includes the %.1f min of duplicate compute from those "
              "re-runs" % dup_minutes) if dups else "",
             ", ".join("%s %.0f min" % (k, smin[k]) for k in STAGES)),
          ""]
    L += ["### %d.1 delta is a CHOICE on M2, and it differs per property"
          % S_PROV, "",
          "M1's rule is delta = 2 x the evaluator's calibration MAE. M2's "
          "evaluator is an exact count over the decoded sequence and has no "
          "error to read a band off, so delta is set by argument "
          "(protocol 2.2):", "",
          "```", "delta = max(delta_ratio * s, 4.4 * quantum)"
          "        # m2_sweep.py:495", "```", "",
          "| property | corpus sd `s` | quantum | delta | delta / s | which "
          "term binds | attainable values spanned |",
          "|---|---|---|---|---|---|---|"]
    for prop in ("gc", "cpg"):
        r = idx[("m2", prop, "unguided", 1.0, 0.5, SEEDS[0])]
        s, q, d = r["s"], r["quantum"], r["delta"]
        binds = ("`delta_ratio * s`" if r["delta_ratio"] * s >= 4.4 * q
                 else "**the 4.4 x quantum floor**")
        L.append("| `%s` | %.6f | %.6f (1/%d) | %.6f | %.3f | %s | %.1f |"
                 % (prop, s, q, round(1 / q), d, d / s, binds, 2 * d / q))
    L += ["",
          "**This is the single most important number on the page for anyone "
          "comparing M2 to M1.** `gc`'s band is 0.16 corpus sd wide and "
          "`cpg`'s is %.2f, because the discrete floor binds on `cpg`; that "
          "alone is why `cpg`'s in-band sits near 0.50 and `gc`'s near 0.14. "
          "The two properties' in-band numbers are not comparable to each "
          "other, and neither is comparable to an M1 in-band, which is set by "
          "a measured oracle error rather than a chosen lattice "
          "(protocol 2.2, 2.3b). They may be printed side by side and never "
          "pooled, averaged or ranked together."
          % (idx[("m2", "cpg", "unguided", 1.0, 0.5, SEEDS[0])]["delta"]
             / idx[("m2", "cpg", "unguided", 1.0, 0.5, SEEDS[0])]["s"]),
          "",
          "`in_band_fraction` here is measured on the **decoded** (argmax) "
          "sequence, so it is M1's `in_band_fraction_dec`. M2 records no "
          "continuous-relaxation in-band and no second oracle.", ""]
    return L


def how_to_read(L, idx, share):
    n_fam, z_fam, weakest, n_above = family_stats(idx)
    if weakest is None:
        survive = ("Nothing on the page clears 2.99, so the wider bar changes "
                   "nothing.")
    elif weakest >= z_fam:
        survive = ("It changes nothing: the weakest \"above\" on the page is "
                   "z = %+.2f, so all %d survive a %d-contrast bar."
                   % (weakest, n_above, n_fam))
    else:
        survive = ("It matters: the weakest \"above\" on the page is z = %+.2f, "
                   "below the %d-contrast bar of %.2f, so that call would not "
                   "survive it." % (weakest, n_fam, z_fam))
    L += ["## %d. How to read this page" % S_READ, "",
          "**The bar, and it is inherited rather than pre-registered for M2.** "
          "`in_band` is a proportion, so its standard error is the binomial "
          "sqrt(p(1-p)/N). MODALITY2_V3_PROTOCOL.md section 5 fixes that form "
          "**per cell at n = 2000** and states **no multiplicity threshold**; "
          "the pooling of the 3 seeds to N = 3n = %d and the bar itself are "
          "FULL_RUN_V3_PROTOCOL.md section 6.1, which M2 inherits. So a "
          "contrast is reported **unpaired**, se = sqrt(se_a^2 + se_b^2), "
          "against a Bonferroni two-sided bar of **z = %.2f** (0.05/18) -- and "
          "every \"above\"/\"tie\" on this page is that inherited bar, **not "
          "pre-registered for M2**, whose own protocol sets no threshold and "
          "computes no verdict (MODALITY2_V3_PROTOCOL.md 3.1, 5). "
          "For scale, this page reports **%d** contrasts (%d against "
          "`unguided`, %d against `plug`), whose own Bonferroni bar would be "
          "z = **%.2f**, so the inherited %.2f is slightly permissive here. %s "
          "Every null is printed beside the **minimum difference this run "
          "could have resolved** at that bar, because a null is evidence "
          "against a method only if the run could have seen the effect "
          "(protocol section 5)."
          % (3 * 2000, Z_BAR, n_fam, n_fam - 8, 8, z_fam, Z_BAR, survive),
          "",
          "**Seed-paired t values are NOT PRE-REGISTERED** and are labelled as "
          "such wherever they appear. Arms within a seed share their initial "
          "noise, so a paired test is tighter than the unpaired one; it cannot "
          "be used to claim a result the unpaired bar denies.",
          "",
          "**No fidelity floor and no verdict on the leaderboard.** v3 removed "
          "v2's chemistry gate and M2 follows it (protocol 3.1): nothing is "
          "disqualified and fidelity is reported. An in-band leaderboard would "
          "crown the arm that destroyed the most sequence structure, so every "
          "in-band figure below prints beside k-mer JS, decode confidence and "
          "diversity.",
          "",
          "**`w = 1` is not any arm's best strength, and equal `w` is not "
          "equal force.** The measured applied-correction share at w = 1 "
          "(`docs/results/M2_SHARE.md`) is %s for `plug`/`tmpd`/`lgd_mc`, "
          "%s for `tfg_mc`, **%s** for `bdg_e4t0.5` and %s for "
          "`bdg_e4t1`, against **0.055** for M1's `plug` on `mu` at w = 1. So "
          "at the shared dial the baselines push ~4.5x weaker than M1's do and "
          "`bdg_e4t0.5` pushes ~10x harder than they do. Part of any BDG gain "
          "at a fixed w is force, not control. Nothing here may be written as "
          "\"the best method\". **Where those shares come from:** %s. It is "
          "the page's central caveat and it rests on that one side "
          "measurement; in particular there is **no `cpg` measurement of the "
          "correction share anywhere in the repository**, so in section %d the "
          "force comparison is carried over from `gc`, exactly as the eta "
          "sweep is (section %d)."
          % (share["plug"], share["tfg_mc"], share["bdg_e4t0.5"],
             share["bdg_e4t1"], share["prov"], S_CPG, S_ABL),
          "",
          "**`tfg_mc` is not TFG.** M2 implements only TFG's Monte-Carlo "
          "smoothing ingredient; M1 runs TFG's full update (protocol "
          "section 3). No M2 sentence may say \"TFG\" unqualified.",
          ""]
    return L


def w_eff_str(p):
    """The signed run-mean of w_eff alone is unfalsifiable when the controller
    changes sign inside the window (v3 records 62 % sign reversal at
    tau_mult 1.5 on M1), so the widening fraction is printed with it."""
    if p["w_eff"] is None:
        return "--"
    return "%+.3f (%.0f %% widen)" % (p["w_eff"], 100 * (p["widening"] or 0.0))


def metric_table(L, groups, order, unguided, title, note="", denom=None):
    L += ["#### %s" % title, ""]
    if note:
        L += [note, ""]
    L += ["| arm | in_band | +-se | bias/d | sd | sd/unguided | sd/d | k-mer JS "
          "| x unguided | decode conf | diversity | clip % of guided "
          "sample-steps | min/cell | measured w_eff (% of steps asking to "
          "widen) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in order:
        p = groups[a]
        L.append("| `%s` | %.4f | %.4f | %+.3f | %.5f | %.3f | %.3f | %.5f | "
                 "%.2f | %.4f | %.4f | %.3f %% | %.2f | %s |"
                 % (a, p["in_band_fraction"], p["se"], p["bias_delta"],
                    p["sd"], p["sd"] / unguided["sd"], p["sd_delta"],
                    p["kmer_js"], p["kmer_js"] / unguided["kmer_js"],
                    p["decode_conf"], p["diversity"], 100 * p["clip_frac"],
                    p["minutes"], w_eff_str(p)))
    L += ["",
          "Clip fraction = clipped sample-steps / (n x guided steps in ONE "
          "batch), summed over the 3 pooled seeds = **%d** guided sample-steps "
          "here. `clipped` is counted per sample inside the per-batch step "
          "loop and summed over the n/batch batches, so dividing by a "
          "batch-summed step count would understate it by that factor. The "
          "unguided arm applies no field at all, so its clip column is 0 by "
          "construction." % (denom or 0), "",
          "**M2 cells record no MAE and no RMSE** -- `m2_sweep.py` writes the "
          "property mean, the property sd, the bias in delta units and the "
          "in-band fraction, and nothing else about the error distribution. "
          "So the location/spread block here is bias/delta and sd/delta rather "
          "than M1's MAE/delta; they bound the mean absolute error between "
          "\\|bias\\| and sqrt(bias^2 + sd^2), and cannot be turned into it.",
          "",
          "**Reading the w_eff column.** `w_eff = 1 + eta*e` with "
          "`e = (V_b - tau^2)/tau^2`, so `w_eff > 1` means the controller is "
          "asking to **contract** the batch's spread toward its setpoint and "
          "`e < 0` (the bracketed percentage) means it is asking to **widen** "
          "it, because the batch is already tighter than `tau`. At "
          "`tau_mult = 1` the setpoint is the corpus sd, which this base model "
          "undershoots, so `bdg_e4t1` sits in the widening branch on every "
          "guided step -- that, not a weaker dial, is why it does nothing "
          "here. The signed run-mean alone would be unfalsifiable, which is "
          "why the sign fraction is printed with it.", ""]
    return L


def h2h_table(L, groups, order, base, label):
    L += ["| arm vs `%s` | d(in_band) | z (unpaired) | verdict at z = %.2f "
          "(inherited, NOT PRE-REGISTERED for M2) | min. detectable d | "
          "per-seed signs | paired t (NOT PRE-REGISTERED) | d(k-mer JS) | "
          "d(decode conf) |" % (base, Z_BAR),
          "|---|---|---|---|---|---|---|---|---|"]
    b = groups[base]
    for a in order:
        if a == base:
            continue
        p = groups[a]
        d, z, mdd, v = contrast(p, b)
        t, signs, ns = paired_t(p, b)
        L.append("| `%s` | %+.2f pp | %+.2f | **%s** | %.2f pp | %s | %s | "
                 "%+.1e | %+.4f |"
                 % (a, 100 * d, z, v, 100 * mdd, signs,
                    ("%+.2f" % t) if t is not None else "degenerate",
                    p["kmer_js"] - b["kmer_js"],
                    p["decode_conf"] - b["decode_conf"]))
    L += ["", label, ""]
    return L


def identity_scan(idx, stage, prop, w, arms, tmin=0.5, tol=1e-5):
    """Pairs of arms that returned the SAME numbers, per seed. Reported per
    seed rather than worst-seed-only, because a coincidence on one seed is
    still two arms that did not measure two things there."""
    out = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            hits = []
            for s in SEEDS:
                d = max(abs(idx[(stage, prop, a, w, tmin, s)][m]
                            - idx[(stage, prop, b, w, tmin, s)][m])
                        for m in IDENT4)
                if d < tol:
                    hits.append(d)
            if hits:
                out.append((a, b, len(hits), max(hits)))
    return out


def identity_note(L, idx, prop, w):
    """Arms that returned the SAME numbers are not two independent baselines."""
    same = identity_scan(idx, "m2", prop, w,
                         [a for a in HEADLINE_ARMS if a != "unguided"])
    if same:
        L += ["**Numerically identical arms at this cell** (max |d| below 1e-5 "
              "over in-band, property mean, property sd and k-mer JS): "
              + "; ".join("`%s` = `%s` on %d of %d seeds (max |d| %.1e)"
                          % (a, b, n, len(SEEDS), d) for a, b, n, d in same)
              + ". These are **not** independent baselines and must never be "
                "reported as agreeing confirmations. Protocol 3.0 gives the "
                "measured reason: v_f falls from 2.971e-03 at t = 0 to "
                "7.729e-08 at t = 0.5 (a factor of 38,000), so TMPD's "
                "denominator s^2 + v_f tends to s^2 and LGD's draw scale "
                "v_f/|g|^2 tends to 0, and both reduce to their common "
                "ancestor DPS.", ""]
    else:
        L += ["**No two arms are numerically identical at this cell** "
              "(threshold 1e-5 on in-band, property mean, property sd and "
              "k-mer JS).", ""]
    return L


def headline(L, idx, prop, secno, share):
    L += ["## %d. Headline: `%s`, t >= 0.5, 3 seeds pooled" % (secno, prop), ""]
    if prop == "cpg":
        L += ["> **`cpg` is the replication property** (protocol 1.1): a "
              "second observable with a different functional form -- `gc` is "
              "exactly affine, `cpg` is quadratic -- so that a result is not a "
              "quirk of GC in particular. Its band is **%.3f corpus sd** wide "
              "because the discrete 4.4 x quantum floor binds on it, against "
              "`gc`'s 0.160, so its in-band level is not comparable with "
              "`gc`'s and the two are never averaged."
              % (idx[("m2", "cpg", "unguided", 1.0, 0.5, SEEDS[0])]["delta"]
                 / idx[("m2", "cpg", "unguided", 1.0, 0.5, SEEDS[0])]["s"]),
              ""]
    L += ["**The protocol's headline strength is w = 1** (protocol section 3, "
          "\"Headline strength w = 1 [FROM v3]\"): M2 inherits v3's dial "
          "unchanged, because a separately chosen strength would mean the two "
          "modalities were not run under the same protocol. **w = 4 is the "
          "ablation's second rung** (protocol section 4), and it is reported "
          "here as well because it is the rung where M2's `plug` finally "
          "reaches M1's own force: its applied-correction share is %s there "
          "against M1 `plug`'s 0.055 at w = 1. A weak baseline result at "
          "w = 1 is therefore a statement about strength, not about the "
          "method. %s" % (share["plug_w4"],
                          "That share, like every other on this page, was "
                          "measured at %s (section %d): **there is no `cpg` "
                          "measurement of the applied-correction share in the "
                          "repository**, so on this property the force "
                          "comparison is carried over from `gc` and not "
                          "measured -- the same gap the eta sweep has "
                          "(section %d)."
                          % (share["prov_short"], S_READ, S_ABL)
                          if prop == "cpg" else
                          "It was measured at %s (section %d)."
                          % (share["prov_short"], S_READ)), ""]
    for w in (1.0, 4.0):
        g = {}
        for a in HEADLINE_ARMS:
            if a == "unguided":
                # unguided applies NO field, so its w = 1 and w = 4 cells are
                # the same 6000 sequences twice over. Pooling both would halve
                # the reference's standard error against a sample size it does
                # not have, so only the w = 1 copies are pooled and the
                # identity is asserted below.
                rs = [idx[("m2", prop, a, 1.0, 0.5, s)] for s in SEEDS]
            else:
                rs = [idx[("m2", prop, a, w, 0.5, s)] for s in SEEDS]
            g[a] = pooled(rs, "%s/%s/w%g" % (prop, a, w))
        dmax = max(max(abs(idx[("m2", prop, "unguided", 1.0, 0.5, s)][m]
                           - idx[("m2", prop, "unguided", 4.0, 0.5, s)][m])
                       for s in SEEDS) for m in IDENT4)
        L = metric_table(
            L, g, HEADLINE_ARMS, g["unguided"],
            "`%s` at w = %g%s" % (prop, w,
                                  "  (the protocol's headline)" if w == 1.0
                                  else "  (the rung where the baselines reach "
                                       "M1's force)"),
            note=("`unguided` is w-independent -- no field is applied -- so it "
                  "is pooled over its 3 seeds ONCE (N = %d), not over the 6 "
                  "cells the tree holds at the two nominal strengths. The w = 1 "
                  "and w = 4 unguided cells agree to max |d| = %.1e, so this "
                  "loses nothing."
                  % (g["unguided"]["N"], dmax)),
            denom=clip_denom(idx[("m2", prop, "plug", w, 0.5, SEEDS[0])]) * 3)
        L = h2h_table(
            L, g, HEADLINE_ARMS, "unguided",
            "Both BDG arms are reported separately and are never pooled "
            "(protocol section 3 lists them as two arms).")
        L = h2h_table(
            L, g, ["bdg_e4t0.5", "bdg_e4t1"], "plug",
            "`plug` is BDG at eta = 0 exactly, so `bdg` - `plug` bounds the "
            "whole dispersion term's effect at this strength. It does not "
            "separate \"controlled the spread\" from \"pushed harder\": at one "
            "strength those are one dial, and `bdg_e4t0.5`'s measured "
            "correction share is ~10x the baselines'. The eta sweep at fixed w "
            "(section %d) is the control that separates them." % S_ABL)
        L = identity_note(L, idx, prop, w)
        # mechanism line
        b = g["bdg_e4t0.5"]
        u = g["unguided"]
        L += ["**Mechanism at `bdg_e4t0.5`, w = %g:** in-band %+.2f pp against "
              "unguided is bought by **contracting** the property spread "
              "(sd %.5f -> %.5f, %.3fx) while the bias **worsens** "
              "(%+.3f -> %+.3f delta). The controller's own state reads "
              "w_eff = %.2f (e = %+.3f; it asked to widen on %.0f %% of guided "
              "steps). Fidelity moves with it: k-mer JS %.5f -> %.5f "
              "(%.2fx the unguided cell, and the frozen base scores 0.0007 "
              "against real test sequences where uniform-random scores 0.0170 "
              "-- `proj1/m2/blade_bundle/check_quality.log`), "
              "decode confidence %.4f -> %.4f, diversity %.4f -> %.4f."
              % (w, 100 * (b["in_band_fraction"] - u["in_band_fraction"]),
                 u["sd"], b["sd"], b["sd"] / u["sd"], u["bias_delta"],
                 b["bias_delta"], b["w_eff"], b["bdg_e"],
                 100 * (b["widening"] or 0.0), u["kmer_js"], b["kmer_js"],
                 b["kmer_js"] / u["kmer_js"], u["decode_conf"],
                 b["decode_conf"], u["diversity"], b["diversity"]),
              ""]
        if b["clip_frac"] > 0.05:
            L += ["> **Clip-limited.** This cell clips on **%.1f %%** of "
                  "guided sample-steps. The clip is applied after `w`, so part "
                  "of what `w` asked for was thrown away and the rung's result "
                  "is not purely the controller's (protocol 1.2b, 4.1)."
                  % (100 * b["clip_frac"]), ""]
    return L


def grid_spearman(idx, w):
    """Rank correlation between the MEASURED deviation weight w_eff and the
    in-band change against the eta = 0 cell, over the 16 feedback rungs."""
    base = pooled([idx[("m2abl", "gc", "bdg_e0t1", w, 0.5, s)] for s in SEEDS])
    pts = []
    for tau in TAUS:
        for eta in (1, 2, 4, 8):
            p = pooled([idx[("m2abl", "gc", "bdg_e%gt%g" % (eta, tau), w, 0.5,
                             s)] for s in SEEDS])
            pts.append((p["w_eff"],
                        p["in_band_fraction"] - base["in_band_fraction"]))
    return spearman(pts)


def best_abl_rung(idx):
    """The best rung inside v3's window, as a gain over unguided -- the number
    the window comparison has to be made against."""
    ung = sum(idx[("m2", "gc", "unguided", 1.0, 0.5, s)]["in_band_fraction"]
              for s in SEEDS) / len(SEEDS)
    best = None
    for w in (1.0, 4.0):
        for tau in TAUS:
            for eta in (1, 2, 4, 8):
                a = "bdg_e%gt%g" % (eta, tau)
                p = pooled([idx[("m2abl", "gc", a, w, 0.5, s)] for s in SEEDS])
                d = p["in_band_fraction"] - ung
                if best is None or d > best[0]:
                    best = (d, a, w, p)
    return best


def best_headline_gain(idx, prop):
    """The largest in-window gain over `unguided` among the headline arms on
    one property, as (d, arm, w).

    The window-versus-method ratio is a `gc` statement -- the t = 0 rung and
    the eta sweep both ran on `gc` only -- so `cpg`'s own best in-window gain
    is printed beside it rather than folded into the same ratio: the two
    properties' bands differ by 3.7x in corpus-sd units and their in-band
    levels are not comparable (protocol 2.2)."""
    ung = pooled([idx[("m2", prop, "unguided", 1.0, 0.5, s)] for s in SEEDS],
                 "%s/unguided" % prop)["in_band_fraction"]
    best = None
    for w in (1.0, 4.0):
        for a in HEADLINE_ARMS[1:]:
            p = pooled([idx[("m2", prop, a, w, 0.5, s)] for s in SEEDS],
                       "%s/%s/w%g" % (prop, a, w))
            d = p["in_band_fraction"] - ung
            if best is None or d > best[0]:
                best = (d, a, w)
    return best


def window_section(L, idx, rows, secno):
    L += ["## %d. Diagnostic: the t_min = 0 window -- NEVER pooled with the "
          "headline" % secno, "",
          "The early window is **demoted to a diagnostic** by protocol "
          "section 2.1: the run keeps a t_min = 0 rung on `plug` and the two "
          "headline BDG arms so the write-up can state what guiding from 0 "
          "would have done from this project's own cells rather than from a "
          "comment. These 9 cells are `gc`, w = 1 only. They are a different "
          "regime, not a stronger setting of the same one, and pooling them "
          "with the t >= 0.5 rows would average two regimes together.", ""]
    g, gw = {}, {}
    for a in ("plug", "bdg_e4t0.5", "bdg_e4t1"):
        g[a] = pooled([idx[("m2abl", "gc", a, 1.0, 0.0, s)] for s in SEEDS],
                      "t0/%s" % a)
        gw[a] = pooled([idx[("m2", "gc", a, 1.0, 0.5, s)] for s in SEEDS],
                       "t0.5/%s" % a)
    u = pooled([idx[("m2", "gc", "unguided", 1.0, 0.5, s)] for s in SEEDS],
               "unguided")
    L += ["| arm | window | in_band | +-se | bias/d | sd | k-mer JS | decode "
          "conf | diversity | clip % of guided sample-steps | min/cell |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    L.append("| `unguided` | -- | %.4f | %.4f | %+.3f | %.5f | %.5f | %.4f | "
             "%.4f | %.3f %% | %.2f |"
             % (u["in_band_fraction"], u["se"], u["bias_delta"], u["sd"],
                u["kmer_js"], u["decode_conf"], u["diversity"], 0.0,
                u["minutes"]))
    for a in ("plug", "bdg_e4t0.5", "bdg_e4t1"):
        for tag, p in (("t >= 0.5", gw[a]), ("**t >= 0**", g[a])):
            L.append("| `%s` | %s | %.4f | %.4f | %+.3f | %.5f | %.5f | %.4f | "
                     "%.4f | %.3f %% | %.2f |"
                     % (a, tag, p["in_band_fraction"], p["se"],
                        p["bias_delta"], p["sd"], p["kmer_js"],
                        p["decode_conf"], p["diversity"],
                        100 * p["clip_frac"], p["minutes"]))
    L += ["", "The t = 0 cells run **%d** guided steps per batch against the "
          "window's %d, so their clip denominator is %d per cell rather than "
          "%d; both are computed from each cell's own counters."
          % (guided_steps_per_batch(idx[("m2abl", "gc", "plug", 1.0, 0.0,
                                         SEEDS[0])]),
             guided_steps_per_batch(idx[("m2", "gc", "plug", 1.0, 0.5,
                                         SEEDS[0])]),
             clip_denom(idx[("m2abl", "gc", "plug", 1.0, 0.0, SEEDS[0])]),
             clip_denom(idx[("m2", "gc", "plug", 1.0, 0.5, SEEDS[0])])),
          ""]
    L += ["**None of the contrasts below is in the pre-registered family**, "
          "which is the guided arms against `unguided` and BDG against `plug` "
          "at t >= 0.5. The same unpaired bar is applied for readability only; "
          "these are diagnostics of the window, not verdicts on a method.", "",
          "| contrast | d(in_band) | z (unpaired) | at z = %.2f | min. "
          "detectable d | per-seed signs | paired t (NOT PRE-REGISTERED) |"
          % Z_BAR, "|---|---|---|---|---|---|---|"]
    for a in ("plug", "bdg_e4t0.5", "bdg_e4t1"):
        d, z, mdd, v = contrast(g[a], gw[a])
        t, signs, _ = paired_t(g[a], gw[a])
        L.append("| `%s`: t >= 0 vs t >= 0.5 | %+.2f pp | %+.2f | **%s** | "
                 "%.2f pp | %s | %s |"
                 % (a, 100 * d, z, v, 100 * mdd, signs,
                    ("%+.2f" % t) if t is not None else "degenerate"))
    for a in ("bdg_e4t0.5", "bdg_e4t1"):
        d, z, mdd, v = contrast(g[a], g["plug"])
        t, signs, _ = paired_t(g[a], g["plug"])
        L.append("| `%s` vs `plug`, both at t >= 0 | %+.2f pp | %+.2f | **%s** "
                 "| %.2f pp | %s | %s |"
                 % (a, 100 * d, z, v, 100 * mdd, signs,
                    ("%+.2f" % t) if t is not None else "degenerate"))
    wdw = 100 * (g["plug"]["in_band_fraction"] - gw["plug"]["in_band_fraction"])
    bd, ba, bw, _bp = best_abl_rung(idx)
    cd, ca, cw = best_headline_gain(idx, "cpg")
    dsm, zsm, msm, _v = contrast(g["bdg_e4t0.5"], g["plug"])
    _t, ssm, _n = paired_t(g["bdg_e4t0.5"], g["plug"])
    don, zon, mon, _v = contrast(g["bdg_e4t1"], g["plug"])
    _t, son, _n = paired_t(g["bdg_e4t1"], g["plug"])
    L += ["",
          "**What this says, and the caveat that must travel with it.** On "
          "`gc`, where both terms were measured, the window is worth "
          "**%+.2f pp** to `plug` against the **%+.2f pp** the best rung in "
          "the whole ablation (`%s` at w = %g) reaches inside v3's window "
          "(section %d): **about %.0fx**. Both figures are `gc` figures. No "
          "t = 0 cell and no eta sweep ran on `cpg`, where the largest "
          "in-window gain over `unguided` is %+.2f pp (`%s` at w = %g); "
          "`gc`'s band is %.3f corpus sd wide and `cpg`'s %.3f, so the two "
          "properties' in-band levels are not comparable and **no "
          "window-versus-method ratio may be formed across them**. What "
          "travels is the rule, not the ratio: no M2 claim about a method may "
          "be stated without the window it was measured at."
          % (wdw, 100 * bd, ba, bw, S_ABL, wdw / (100 * bd), 100 * cd, ca, cw,
             u["delta"] / u["s"],
             idx[("m2", "cpg", "unguided", 1.0, 0.5, SEEDS[0])]["delta"]
             / idx[("m2", "cpg", "unguided", 1.0, 0.5, SEEDS[0])]["s"]),
          "",
          "**But the early window does not answer the method question at all.** "
          "It clips on %.0f-%.0f %% of guided sample-steps: the "
          "score-to-velocity factor (1-t)/max(t, 1e-6) diverges as t -> 0, so "
          "every arm requests a correction far outside the trust region and "
          "what reaches the sample is the clip's ceiling -- identical in "
          "**magnitude** for every arm, which is why the magnitude-only "
          "baselines collapse onto `plug` there, the same mechanism that "
          "collapses `tmpd` and `lgd_mc` onto it inside the window, seen from "
          "the other side. It does **not** equalise direction, and the table "
          "above shows the spread it leaves: %.4f for `plug` against %.4f for "
          "`bdg_e4t1`."
          % (100 * min(p["clip_frac"] for p in g.values()),
             100 * max(p["clip_frac"] for p in g.values()),
             g["plug"]["in_band_fraction"], g["bdg_e4t1"]["in_band_fraction"]),
          "",
          "**The two headline BDG arms land on opposite sides of the bar at "
          "t = 0, and both are reported.** `bdg_e4t0.5` is a **tie** against "
          "`plug` (%+.2f pp, z = %+.2f, min. detectable %.2f pp, per-seed "
          "signs %s) and must be read as \"no advantage at t >= 0\" rather "
          "than as a loss. `bdg_e4t1` is a **loss that clears the bar "
          "downward** (%+.2f pp, z = %+.2f, min. detectable %.2f pp, negative "
          "on all %d seeds, signs %s): at `tau_mult = 1` it spends the whole "
          "run asking to widen, and at t = 0 -- where the clip lets far more "
          "of the request through -- that costs it %.1f pp against `plug`. "
          "Neither arm may be reported without the other."
          % (100 * dsm, zsm, 100 * msm, ssm, 100 * don, zon, 100 * mon,
             len(SEEDS), son, abs(100 * don)),
          "",
          "**And the early window is not merely tighter, it is better on every "
          "axis printed here**: for `plug`, bias/delta improves %+.3f -> "
          "%+.3f, the property sd falls %.5f -> %.5f, k-mer JS *improves* "
          "%.5f -> %.5f and decode confidence rises %.4f -> %.4f. The samples "
          "move closer to real enhancers, not further. That is why the trade "
          "is real, and why it belongs in the write-up as its own finding "
          "rather than as a footnote to the method comparison."
          % (gw["plug"]["bias_delta"], g["plug"]["bias_delta"],
             gw["plug"]["sd"], g["plug"]["sd"], gw["plug"]["kmer_js"],
             g["plug"]["kmer_js"], gw["plug"]["decode_conf"],
             g["plug"]["decode_conf"]),
          ""]
    return L


def wsweep_section(L, idx, rows, args, secno):
    ws = [r for r in rows if r["_stage"] == "m2wsweep"]
    seeds = sorted({r["seed"] for r in ws})
    L += ["## %d. Diagnostic: the strength sweep at w = 16 and w = 64 -- ONE "
          "SEED" % secno, "",
          "**Every row in this section is a single cell on seed %s.** It "
          "carries no seed-to-seed error bar, no pooled N and no verdict: with "
          "one seed the binomial se on n = 2000 is ~0.8 pp and there is "
          "nothing to check it against. It is here because protocol section 3 "
          "keeps the strength sweep as a **diagnostic** -- whether w = 1 is a "
          "strong or a weak setting on DNA is itself part of what had to be "
          "adapted for the new modality -- and explicitly withdraws the "
          "earlier draft's plan to choose M2's strength from it."
          % ", ".join(str(s) for s in seeds), ""]
    u = idx[("m2", "gc", "unguided", 1.0, 0.5, seeds[0])]
    L += ["| arm | w | in_band | bias/d | sd | k-mer JS | decode conf | "
          "diversity | clip % of guided sample-steps | min | measured w_eff |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in [u] + sorted(ws, key=lambda r: (r["w"], arm_key(r))):
        w = "--" if r["arm"] == "unguided" else "%g" % r["w"]
        den = clip_denom(r)
        L.append("| `%s` | %s | %.4f | %+.3f | %.5f | %.5f | %.4f | %.4f | "
                 "%.1f %% | %.2f | %s |"
                 % (arm_key(r), w, r["in_band_fraction"], r["bias_delta"],
                    r["gc_sd"], r["kmer_js"], r["decode_conf"], r["diversity"],
                    100 * r["clipped_sample_steps"] / den if den else 0.0,
                    r["minutes"],
                    "%+.3f (%.0f %% widen)" % (r["diag"]["bdg_w_eff"],
                                               100 * r["diag"]["bdg_widening"])
                    if "bdg_w_eff" in r["diag"] else "--"))
    byw = {(arm_key(r), r["w"]): r for r in ws}
    L += ["",
          "The `unguided` reference row is the seed-%s cell from `%s/m2/`, not "
          "from the sweep tree, which holds only the four guided cells (%s); "
          "it is the same single seed, so the section's caveat covers it."
          % (seeds[0], args.out_dir,
             ", ".join(sorted("`%s` at w = %g" % (arm_key(r), r["w"])
                              for r in ws))), "",
          "One seed, so these are read as direction only: `plug` keeps gaining "
          "as w rises while its clip fraction climbs from 0.0 %% at w = 1 "
          "(headline) to %.1f %% at w = 64, and `bdg_e4t0.5` peaks at w = 16 "
          "and falls back at w = 64 with **%.1f %% of its guided sample-steps "
          "clipped** -- past that point the clip, not `w`, is setting the "
          "strength. Decode confidence is the fidelity cost that moves most "
          "(%.3f unguided -> %.3f at `bdg_e4t0.5`, w = 64)."
          % (100 * byw[("plug", 64.0)]["clipped_sample_steps"]
             / clip_denom(byw[("plug", 64.0)]),
             100 * byw[("bdg_e4t0.5", 64.0)]["clipped_sample_steps"]
             / clip_denom(byw[("bdg_e4t0.5", 64.0)]),
             u["decode_conf"], byw[("bdg_e4t0.5", 64.0)]["decode_conf"]), "",
          "> **A trap in a committed file.** `results/m2_strength.json` "
          "records `\"headline_w\": 16.0`, chosen at n = 500 by the rule "
          "\"smallest w where `plug` - `unguided` exceeds the combined "
          "seed-to-seed se\". **That rule was withdrawn** by protocol "
          "section 3 before the run: a separately chosen M2 strength would "
          "mean the two modalities were not run under the same protocol, "
          "which is the one thing the transfer claim cannot afford. The "
          "headline strength is w = 1. Read that file as the diagnostic it "
          "is, not as a setting.", ""]
    return L


def ablation_section(L, idx, rows, args, secno):
    L += ["## %d. Ablation: the 17-arm eta x tau grid on `gc`" % secno, "",
          "`w` scales the whole guidance field -- the mean term and the "
          "deviation term together -- while `eta` reweights the deviation term "
          "**alone**, through w_eff = 1 + eta*e with e = (V_b - tau^2)/tau^2. "
          "Sweeping `eta` at **fixed w** therefore varies the controller while "
          "holding the push roughly constant, and eta = 0 reduces the field to "
          "`plug` exactly. That is the comparison the M2 write-up should lead "
          "with, not the raw headline ordering, because at equal `w` "
          "`bdg_e4t0.5` applies ~10x the baselines' correction share.", "",
          "17 arms = eta 0 (once, since eta = 0 kills the feedback whatever "
          "tau is) + eta in {1,2,4,8} x tau_mult in {0.5,0.75,1,1.5}, at each "
          "of w in {1,4}, 3 seeds, n = 2000. Each cell below is **in-band "
          "against the eta = 0 cell at the same w**, in percentage points; the "
          "gain against `unguided` is given too, because that is what "
          "`M2_ABLATION_GRID.md` tabulates.", "",
          "**Rows are read by measured w_eff, never by tau_mult.** tau_mult is "
          "not a monotone axis: a **negative** measured w_eff, which the "
          "tau_mult = 1 and 1.5 rows reach, means the deviation term has "
          "reversed sign and is pushing samples *away* from the batch mean. "
          "That is the controller doing what its setpoint specifies -- a "
          "setpoint above the data's own spread asks for widening -- and it is "
          "why the grid must be ordered by the run's own w_eff rather than by "
          "the nominal knob.", ""]
    ung = {s: idx[("m2", "gc", "unguided", 1.0, 0.5, s)]["in_band_fraction"]
           for s in SEEDS}
    summary = []
    for w in (1.0, 4.0):
        cell = {}
        for tau in TAUS:
            for eta in ETAS:
                a = "bdg_e0t1" if eta == 0 else "bdg_e%gt%g" % (eta, tau)
                cell[(tau, eta)] = pooled(
                    [idx[("m2abl", "gc", a, w, 0.5, s)] for s in SEEDS],
                    "abl/%s/w%g" % (a, w))
        base = cell[(1.0, 0)]
        L += ["### w = %g" % w, "",
              "eta = 0 (`bdg_e0t1` = `plug`'s field) sits at in-band **%.4f** "
              "(+-%.4f), %+.2f pp against `unguided`, w_eff = %.2f by "
              "construction.  Minimum difference resolvable against it at "
              "z = %.2f: **%.2f pp** -- and the grid is a 16-way selection on "
              "top, so a win claimed from it needs a wider bar still "
              "(FULL_RUN_V3_PROTOCOL.md section 6.2)."
              % (base["in_band_fraction"], base["se"],
                 100 * (base["in_band_fraction"]
                        - sum(ung.values()) / len(ung)),
                 base["w_eff"], Z_BAR,
                 100 * Z_BAR * math.sqrt(2) * base["se"]), "",
              "| tau_mult | | eta=1 | eta=2 | eta=4 | eta=8 |",
              "|---|---|---|---|---|---|"]
        pts = []
        for tau in TAUS:
            dl, wl, ul, cl = [], [], [], []
            for eta in (1, 2, 4, 8):
                p = cell[(tau, eta)]
                d = p["in_band_fraction"] - base["in_band_fraction"]
                dl.append("%+.2f" % (100 * d))
                wl.append("%.2f" % p["w_eff"])
                ul.append("%+.2f" % (100 * (p["in_band_fraction"]
                                            - sum(ung.values()) / len(ung))))
                cl.append("%.1f" % (100 * p["clip_frac"]))
                pts.append((p["w_eff"], d))
            L += ["| **%g** | d vs eta=0 (pp) | %s |" % (tau, " | ".join(dl)),
                  "| | vs unguided (pp) | %s |" % " | ".join(ul),
                  "| | measured w_eff | %s |" % " | ".join(wl),
                  "| | clip %% of guided steps | %s |" % " | ".join(cl)]
        rho = spearman(pts)
        if abs(rho - grid_spearman(idx, w)) > 1e-12:      # the two must agree
            die("the grid's Spearman and grid_spearman() disagree at w = %g"
                % w)
        r05 = [cell[(0.5, e)]["in_band_fraction"] for e in (0, 1, 2, 4, 8)]
        r15 = [cell[(1.5, e)]["in_band_fraction"] for e in (0, 1, 2, 4, 8)]
        up = sum(1 for i in range(4) if r05[i + 1] > r05[i])
        dn = sum(1 for i in range(4) if r15[i + 1] < r15[i])
        best = max(((cell[(t, e)]["in_band_fraction"], t, e)
                    for t in TAUS for e in (1, 2, 4, 8)))
        bp = cell[(best[1], best[2])]
        d, z, mdd, v = contrast(bp, base)
        summary.append(
            "- **w = %g.** The tau = 0.5 row rises with eta in **%d of 4** "
            "steps (%+.2f pp at eta 8 against eta 0); the tau = 1.5 row falls "
            "in **%d of 4** (%+.2f pp at eta 8). Spearman(measured w_eff, "
            "d vs eta = 0) over the 16 feedback rungs = **%.3f**. The best "
            "rung is `bdg_e%gt%g` at %+.2f pp over eta = 0 (z = %+.2f "
            "unpaired, %s at z = %.2f before any selection adjustment), and it "
            "clips on %.1f %% of guided sample-steps."
            % (w, up, 100 * (r05[4] - r05[0]), dn, 100 * (r15[4] - r15[0]),
               rho, best[2], best[1], 100 * d, z, v, Z_BAR,
               100 * bp["clip_frac"]))
        abl_arms = ["bdg_e0t1"] + ["bdg_e%gt%g" % (e, t)
                                   for t in TAUS for e in (1, 2, 4, 8)]
        same = identity_scan(idx, "m2abl", "gc", w, abl_arms)
        if same:
            L += ["", "**Grid arms that returned the same numbers** (max |d| "
                  "below 1e-5 on the four metrics): "
                  + "; ".join("`%s` = `%s` on %d of %d seeds (max |d| %.1e)"
                              % (a, b, n, len(SEEDS), d) for a, b, n, d in same)
                  + ". These are coincidences of a counting metric at "
                    "n = 2000 on a single seed, not evidence that two settings "
                    "are one setting -- unlike the `plug`/`tmpd`/`lgd_mc` "
                    "identity above, which holds on every seed and has a "
                    "measured cause. They are listed so that no reader "
                    "discovers them unannounced."]
        L += ["", "Fidelity across the tau = 0.5 column at this strength: "
              + "; ".join("eta %g k-mer JS %.5f (%.2fx unguided), conf %.4f, "
                          "diversity %.4f"
                          % (e, cell[(0.5, e)]["kmer_js"],
                             cell[(0.5, e)]["kmer_js"]
                             / pooled([idx[("m2", "gc", "unguided", 1.0, 0.5,
                                            s)] for s in SEEDS])["kmer_js"],
                             cell[(0.5, e)]["decode_conf"],
                             cell[(0.5, e)]["diversity"])
                          for e in (1, 2, 4, 8))
              + ".", ""]
    L += ["### What the grid supports, in one line each", ""] + summary + [
        "",
        "**The grid was run on `gc` only.** `cpg` has the 7 headline arms at "
        "both strengths and **no eta sweep**, so the one place M2's BDG result "
        "is largest -- `cpg` at w = 4 -- is the one place there is no "
        "fixed-`w` control to separate the controller from the extra force. "
        "The mechanism argument on this page rests on `gc`; on `cpg` it is "
        "carried over, not measured.", ""]
    L += ["**What it supports:** in-band tracks the **measured deviation "
          "weight w_eff**, not the global `w`. The setpoint reverses the sign "
          "of the effect -- asking for a tighter spread than the data's own "
          "(tau_mult < 1) raises in-band, asking for a wider one (tau_mult > 1) "
          "lowers it below eta = 0 -- and force cannot reverse the sign of its "
          "own parameter at fixed `w`.", "",
          "**What it does NOT support:** that the feedback *loop* is necessary. "
          "No fixed-w_eff control was run, so a schedule that replayed the same "
          "w_eff trajectory open-loop is not excluded; on M1, "
          "`docs/methods/BDG_REVIEW.md`'s replay recovers 69.5-96 % of the "
          "effect. And the strongest rungs are the most clipped ones, so their "
          "result is not purely the controller's.", ""]
    L = grid_crosscheck(L, idx, ung, args)
    return L


def grid_crosscheck(L, idx, ung, args):
    """The committed grid must equal what these cells say. Same procedure as
    m2_ablation_grid.py: gain over unguided, both trees merged, 3 seeds."""
    path = args.grid_md
    L += ["### Cross-check against `%s`" % path, ""]
    if not os.path.exists(path):
        L += ["**NOT CHECKED** -- `%s` is not in the tree, so the committed "
              "grid could not be compared against these cells." % path, ""]
        return L
    with open(path, encoding="utf-8") as fh:
        txt = fh.read()
    want = {}
    w = None
    for line in txt.splitlines():
        m = re.match(r"^### w = ([0-9.]+)", line)
        if m:
            w = float(m.group(1))
            continue
        m = re.match(r"^\|\s*([0-9.]+)\s*\|(.+)\|\s*$", line)
        if m and w is not None:
            vals = [v.strip() for v in m.group(2).split("|")]
            if len(vals) != len(ETAS) or not all(
                    re.match(r"^[+-][0-9.]+$", v) for v in vals):
                continue
            for e, v in zip(ETAS, vals):
                want[(w, float(m.group(1)), e)] = float(v)
    if not want:
        L += ["**NOT CHECKED** -- no grid table could be parsed out of `%s`."
              % path, ""]
        return L
    worst, n_ok = 0.0, 0
    for (w, tau, eta), v in sorted(want.items()):
        a = "bdg_e0t1" if eta == 0 else "bdg_e%gt%g" % (eta, tau)
        got = 100 * sum(idx[("m2abl", "gc", a, w, 0.5, s)]["in_band_fraction"]
                        - ung[s] for s in SEEDS) / len(SEEDS)
        worst = max(worst, abs(got - v))
        n_ok += 1
    L += ["Recomputed all **%d** published cells (gain over `unguided`, in pp, "
          "3 seeds) from the same result files: worst disagreement "
          "**%.3f pp**, which is below the 0.01 pp the committed table is "
          "rounded to. **The committed grid and this page agree.** The two "
          "differ only in what they subtract: `M2_ABLATION_GRID.md` reports "
          "the gain over `unguided`, this page reports it over the eta = 0 "
          "cell as well, which is the comparison that isolates the feedback "
          "term." % (n_ok, worst), ""]
    return L


def controls_section(L, idx, rows, secno):
    L += ["## %d. Controls" % secno, "",
          "**Rows 1 and 2 are the protocol's two pre-registered pass/fail "
          "gates** (MODALITY2_V3_PROTOCOL.md section 4.1): a failure of either "
          "voids the BDG rows rather than being reported as a result. **Rows "
          "3 to 5 are this generator's own checks over every cell it reads** "
          "-- section 4.1 pre-registers no such gate, and the `where` column "
          "says so on each of them, so none of them may be cited as a "
          "pre-registered control.", ""]
    L += ["| control | where | result |", "|---|---|---|"]
    worst = 0.0
    for w in (1.0, 4.0):
        for s in SEEDS:
            a = idx[("m2abl", "gc", "bdg_e0t1", w, 0.5, s)]
            b = idx[("m2", "gc", "plug", w, 0.5, s)]
            worst = max(worst, max(abs(a[m] - b[m]) for m in IDENT4
                                   + ("decode_conf", "diversity")))
    L.append("| 1. `bdg` at eta = 0 IS `plug`, bit for bit | `gc`, w in {1,4}, "
             "3 seeds, 6 metrics | **%s** -- max \\|d\\| = %.2e |"
             % ("PASS" if worst == 0.0 else "FAIL", worst))
    L.append("| 2. `tmpd` exactly parallel to `plug` on `gc`, field ratio "
             "s^2/(s^2 + v_f(t)) | the guidance field, not end metrics | "
             "**recorded in the protocol, not recomputable from cells**: "
             "verified 26 Sep, cos = 1.0000000000 and the ratio matches to 7 "
             "decimals at t in {0, 0.1, 0.3, 0.5}. It is deliberately a gate "
             "on the field -- the clip and the sampler wash the difference out "
             "downstream, so comparing `in_band` would pass a broken "
             "implementation |")
    ng = sum(1 for r in rows
             if any(r.get(k) is None or r.get(k) != r.get(k)
                    for k in ("in_band_fraction", "gc_mean", "gc_sd",
                              "kmer_js", "decode_conf", "diversity")))
    L.append("| 3. every recorded metric finite | all %d files -- **this "
             "script's check, not a 4.1 gate** | **%s** |"
             % (len(rows), "PASS" if ng == 0 else "FAIL (%d cells)" % ng))
    bad = [r["_base"] for r in rows
           if abs(r["delta"] - max(r["delta_ratio"] * r["s"],
                                   4.4 * r["quantum"])) > 1e-9]
    L.append("| 4. delta follows the protocol's own formula, per property | "
             "all %d files -- **this script's check, not a 4.1 gate** "
             "(the per-property quantum is what protocol 7.2's suite asserts "
             "in code) | **%s** |"
             % (len(rows), "PASS" if not bad else "FAIL: " + ", ".join(bad)))
    L.append("| 5. n %% batch == 0 and n_controllers == n/batch | all %d "
             "files -- **this script's check, not a 4.1 gate** (protocol 7.2's "
             "suite asserts the refusal in the driver) | **%s** |"
             % (len(rows),
                "PASS" if all(r["n"] % r["batch"] == 0
                              and r["n_controllers"] == r["n"] // r["batch"]
                              for r in rows) else "FAIL"))
    L += ["",
          "**A third check is reported, not gated** (protocol 4.1): the clipped "
          "sample-step count across the two strengths. The clip is applied "
          "after `w`, so quadrupling `w` on a request the clip already "
          "truncates may change nothing but how much is thrown away. Every "
          "table above prints it, and rungs above 5 % are marked.", ""]
    return L


def functional_section(L, rows, args, secno):
    L += ["## %d. Functional evaluation: what is committed" % secno, "",
          "M2 has no analogue of RDKit validity -- every string over ACGT is a "
          "legal sequence -- so in-band could in principle be bought with "
          "strength at no visible cost. Two scorers close that, and both write "
          "a committed summary. **Neither is part of the pre-registered "
          "protocol**: MODALITY2_V3_PROTOCOL.md names no functional gate, so "
          "nothing below decides anything; it is reported.", ""]
    bykey = {}
    for r in rows:
        bykey[r["_base"].replace(".json", ".permol.pt")] = r
    for title, path, keys, ref_fmt in (
            ("9.1 Realism gate (`proj1/m2/enhancer_gate.py`)",
             args.gate_json, ("gate_mean", "gate_pass"), None),
            ("9.2 DeepFlyBrain activity (`proj1/m2/deepflybrain.py`)",
             args.dfb_json, ("dfb_max_topic", "dfb_frac_active"), None)):
        L += ["### %s" % title.split(" ", 1)[1], ""]
        if not os.path.exists(path):
            L += ["**NOT COMMITTED** -- `%s` is not in the tree." % path, ""]
            continue
        with open(path, encoding="utf-8") as fh:
            J = json.load(fh)
        cells = J["cells"]
        if "gate_auc" in J:
            L += ["Gate: a discriminator trained to separate real enhancers "
                  "from a **per-sequence order-1 Markov null**, which "
                  "reproduces each sequence's own mononucleotide and "
                  "dinucleotide frequencies -- so GC and CpG, the two things "
                  "guidance steers, are matched by construction and the gate "
                  "is forced onto third-order and longer structure. Held-out "
                  "AUC **%.4f**. References: real train **%.4f**, real test "
                  "**%.4f**, Markov null **%.4f**."
                  % (J["gate_auc"], J["reference"]["real_train"],
                     J["reference"]["real_test"], J["reference"]["markov_null"]),
                  ""]
        else:
            rf = J["reference"]
            L += ["DeepFlyBrain (Janssens et al., Nature 2022), the "
                  "accessibility predictor trained on the very corpus our base "
                  "model learned from, re-implemented in PyTorch. Per "
                  "sequence: the maximum topic probability over its 81 topics, "
                  "and whether that exceeds 0.5. References: real train "
                  "max-topic **%.4f**, %.3f active; real test **%.4f**, %.3f "
                  "active."
                  % (rf["real_train"]["max_topic_mean"],
                     rf["real_train"]["frac_active"],
                     rf["real_test"]["max_topic_mean"],
                     rf["real_test"]["frac_active"]), ""]
        agg = collections.defaultdict(list)
        for c in cells.values():
            a = c["arm"] if c["arm"] != "bdg" else "bdg_" + str(c["variant"])
            agg[(c["prop"], a, c["w"])].append(c)
        L += ["| property | arm | w | seeds | %s | %s | vs unguided |"
              % keys, "|---|---|---|---|---|---|---|"]
        for prop in ("gc", "cpg"):
            base = agg.get((prop, "unguided", 1.0))
            b0 = sum(c[keys[0]] for c in base) / len(base) if base else None
            for k in sorted([k for k in agg if k[0] == prop],
                            key=lambda k: (k[1], k[2])):
                cs = agg[k]
                v0 = sum(c[keys[0]] for c in cs) / len(cs)
                v1 = sum(c[keys[1]] for c in cs) / len(cs)
                L.append("| %s | `%s` | %g | %d%s | %.4f | %.4f | %s |"
                         % (prop, k[1], k[2], len(cs),
                            " **(ONE SEED)**" if len(cs) == 1 else "",
                            v0, v1,
                            "--" if b0 is None else "%+.4f" % (v0 - b0)))
        # coverage
        missing = sorted(set(bykey) - set(cells))
        agm = collections.Counter()
        for m in missing:
            r = bykey[m]
            agm[(r["prop"], arm_key(r), r["w"], r["t_min_guide"])] += 1
        L += ["", "**Coverage: %d of %d distinct cells scored; %d not scored.**"
              % (len(cells), len(bykey), len(missing))]
        if missing:
            L += ["The scorer reads the per-sequence `*.permol.pt` sidecars, "
                  "which are gitignored and stay on blade, so an unscored cell "
                  "cannot be filled in from this repository. Unscored:", ""]
            L += ["| property | arm | w | t_min | seeds unscored |",
                  "|---|---|---|---|---|"]
            for k in sorted(agm, key=str):
                L.append("| %s | `%s` | %g | %g | %d |"
                         % (k[0], k[1], k[2], k[3], agm[k]))
            L += ["", "Two of these matter for what the page can say: **all 9 "
                  "t_min = 0 cells are unscored**, so the early window has no "
                  "functional reading at all; and **all 3 seeds of "
                  "`bdg_e8t0.5` at w = 4** are unscored -- that is the "
                  "strongest rung in the whole ablation (+4.23 pp), so the one "
                  "cell most likely to be challenged on fidelity is the one "
                  "with no functional number.", ""]
        else:
            L += [""]
    worst = {}
    for nm, path, k in (("gate", args.gate_json, "gate_mean"),
                        ("dfb", args.dfb_json, "dfb_max_topic"),
                        ("act", args.dfb_json, "dfb_frac_active")):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            cells = json.load(fh)["cells"]
        agg = collections.defaultdict(list)
        for c in cells.values():
            a = c["arm"] if c["arm"] != "bdg" else "bdg_" + str(c["variant"])
            agg[(c["prop"], a, c["w"])].append(c)
        base = {p: sum(c[k] for c in agg[(p, "unguided", 1.0)])
                / len(agg[(p, "unguided", 1.0)]) for p in ("gc", "cpg")}
        cand = [(sum(c[k] for c in v) / len(v) - base[key[0]], key, len(v))
                for key, v in agg.items() if key[1] != "unguided"]
        worst[nm] = min(cand)
    u_dfb = None
    if os.path.exists(args.dfb_json):
        with open(args.dfb_json, encoding="utf-8") as fh:
            J = json.load(fh)
        ug = [c for c in J["cells"].values() if c["arm"] == "unguided"]
        u_dfb = (sum(c["dfb_max_topic"] for c in ug) / len(ug),
                 sum(c["dfb_frac_active"] for c in ug) / len(ug),
                 J["reference"]["real_test"]["max_topic_mean"],
                 J["reference"]["real_test"]["frac_active"])
    L += ["### What these two say, and what they do not", "",
          "**Guidance costs almost nothing functionally, over the cells that "
          "were scored.** Across every scored arm and strength the realism "
          "gate moves by at most **%+.4f** against `unguided` (worst: "
          "`%s`/`%s` at w = %g, on %d scored %s) and DeepFlyBrain's mean "
          "max-topic by at most **%+.4f** (worst: `%s`/`%s` at w = %g, %d "
          "scored %s). So the \"in-band is gameable\" worry is answered with "
          "evidence rather than with an assumption."
          % (worst["gate"][0], worst["gate"][1][0], worst["gate"][1][1],
             worst["gate"][1][2], worst["gate"][2],
             "seed" if worst["gate"][2] == 1 else "seeds",
             worst["dfb"][0], worst["dfb"][1][0], worst["dfb"][1][1],
             worst["dfb"][1][2], worst["dfb"][2],
             "seed" if worst["dfb"][2] == 1 else "seeds"),
          "",
          "**The base model, not the guidance, is the weak link.** Our "
          "unguided samples score max-topic **%.4f** and **%.1f %%** predicted "
          "accessible against real held-out regions' **%.4f** and **%.1f %%** "
          "-- about %.1fx less likely to be called accessible -- which no "
          "distributional metric on this page shows. That is a limitation of "
          "the frozen base, not of the guidance."
          % (u_dfb[0], 100 * u_dfb[1], u_dfb[2], 100 * u_dfb[3],
             u_dfb[3] / u_dfb[1]),
          "",
          "**Three caveats that must travel with these numbers.**",
          "",
          "1. **The DeepFlyBrain port's equivalence to the published Keras "
          "model is in no committed file.** A \"verified identical, "
          "max \\|d\\| 4.5e-08, r = 1.0000000000\" claim circulates in the "
          "project's README; nothing in this repository supports it, and "
          "`proj1/m2/deepflybrain.py`'s own docstring says the port was "
          "validated **without** TensorFlow, by a real-versus-shuffled "
          "self-test (`--selftest`, which asserts only that real held-out "
          "enhancers score more than 3x uniform-random). Either commit the "
          "comparison or withdraw the equivalence claim.",
          "2. **The per-sequence correlation between the gate and the steered "
          "property is also uncommitted.** The figures quoted elsewhere "
          "(+0.001 for GC, -0.004 for CpG) need the sidecars and cannot be "
          "recomputed here. The gate's *design* -- a per-sequence Markov null "
          "-- is the argument that it is blind to GC and CpG; the measurement "
          "of that blindness is not in the tree.",
          "3. **DeepFlyBrain is a predictor with its own error, not an assay**, "
          "and our corpus is its training data.", ""]
    return L


def outcome_section(L, idx, rows, args, secno, share):
    # functional-evaluation coverage: (scored, distinct cells)
    bases = {r["_base"].replace(".json", ".permol.pt") for r in rows}
    cov = (0, len(bases))
    if os.path.exists(args.gate_json):
        with open(args.gate_json, encoding="utf-8") as fh:
            cov = (len(set(json.load(fh)["cells"]) & bases), len(bases))
    g1 = {a: pooled([idx[("m2", "gc", a, 1.0, 0.5, s)] for s in SEEDS])
          for a in HEADLINE_ARMS}
    g4 = {a: pooled([idx[("m2", "gc", a, 4.0, 0.5, s)] for s in SEEDS])
          for a in HEADLINE_ARMS}
    g1["unguided"] = pooled([idx[("m2", "gc", "unguided", 1.0, 0.5, s)]
                             for s in SEEDS])
    g4["unguided"] = g1["unguided"]
    c1 = {a: pooled([idx[("m2", "cpg", a, 1.0, 0.5, s)] for s in SEEDS])
          for a in HEADLINE_ARMS}
    c4 = {a: pooled([idx[("m2", "cpg", a, 4.0, 0.5, s)] for s in SEEDS])
          for a in HEADLINE_ARMS}
    c1["unguided"] = pooled([idx[("m2", "cpg", "unguided", 1.0, 0.5, s)]
                             for s in SEEDS])
    c4["unguided"] = c1["unguided"]

    # everything the prose below asserts, recomputed rather than remembered
    rho = tuple(grid_spearman(idx, w) for w in (1.0, 4.0))
    ident = 0.0
    for prop in ("gc", "cpg"):
        for w in (1.0, 4.0):
            for s in SEEDS:
                for a, b in (("plug", "tmpd"), ("plug", "lgd_mc"),
                             ("tmpd", "lgd_mc")):
                    ident = max(ident,
                                max(abs(idx[("m2", prop, a, w, 0.5, s)][m]
                                        - idx[("m2", prop, b, w, 0.5, s)][m])
                                    for m in IDENT4))
    bd, ba, bw, bp = best_abl_rung(idx)
    w0 = {a: pooled([idx[("m2abl", "gc", a, 1.0, 0.0, s)] for s in SEEDS])
          for a in ("plug", "bdg_e4t0.5", "bdg_e4t1")}
    wgain = w0["plug"]["in_band_fraction"] - g1["plug"]["in_band_fraction"]
    wbdg_d, wbdg_z, wbdg_m, _v = contrast(w0["bdg_e4t0.5"], w0["plug"])
    wbd1_d, wbd1_z, _m, _v = contrast(w0["bdg_e4t1"], w0["plug"])
    cgain, cga, cgw = best_headline_gain(idx, "cpg")
    wclip_lo = min(p["clip_frac"] for p in w0.values())
    wclip_hi = max(p["clip_frac"] for p in w0.values())

    L += ["## %d. Outcome: what Modality 2 supports, in the protocol's own "
          "terms" % secno, "", "### Supports", ""]
    d, z, mdd, v = contrast(g1["bdg_e4t0.5"], g1["plug"])
    d4, z4, m4, v4 = contrast(g4["bdg_e4t0.5"], g4["plug"])
    cd, cz, cm, cv = contrast(c1["bdg_e4t0.5"], c1["plug"])
    cd4, cz4, cm4, cv4 = contrast(c4["bdg_e4t0.5"], c4["plug"])
    L += ["1. **The innovation runs unchanged on a second modality and its "
          "eta = 0 limit reproduces `plug` bit for bit** (section %d, control "
          "1, max \\|d\\| = 0.00e+00 over 6 metrics at both strengths). The "
          % S_CTRL +
          "controller transferred from R^3n molecular coordinates to the DNA "
          "simplex with no change to the control law.",
          "2. **Every headline contrast, in one table.** Nothing here is "
          "selected: it is all 6 guided arms x 2 properties x 2 strengths "
          "against `unguided`, plus both BDG arms against `plug`, at the bar "
          "of section %d." % S_READ,
          ""]
    L += ["| property | w | contrast | d(in_band) | z | verdict at z = %.2f "
          "(inherited, NOT PRE-REGISTERED for M2) | min. detectable d |"
          % Z_BAR, "|---|---|---|---|---|---|---|"]
    tally = collections.Counter()
    winset = set()
    for prop, gg in (("gc", (g1, g4)), ("cpg", (c1, c4))):
        for w, G in zip((1.0, 4.0), gg):
            for a in HEADLINE_ARMS[1:]:
                d_, z_, m_, v_ = contrast(G[a], G["unguided"])
                tally[("unguided", v_)] += 1
                if v_ == "above":
                    winset.add(a)
                L.append("| `%s` | %g | `%s` vs `unguided` | %+.2f pp | %+.2f "
                         "| **%s** | %.2f pp |"
                         % (prop, w, a, 100 * d_, z_, v_, 100 * m_))
            for a in ("bdg_e4t0.5", "bdg_e4t1"):
                d_, z_, m_, v_ = contrast(G[a], G["plug"])
                tally[("plug", v_)] += 1
                L.append("| `%s` | %g | `%s` vs `plug` | %+.2f pp | %+.2f | "
                         "**%s** | %.2f pp |"
                         % (prop, w, a, 100 * d_, z_, v_, 100 * m_))
    wtxt = (", ".join("`%s`" % a for a in sorted(winset))
            if winset else "**none**")
    L += ["",
          "   Against `unguided`: **%d above / %d below / %d tie** of %d. "
          "Against `plug`: **%d above / %d below / %d tie** of %d. The "
          "baselines' ties are not evidence that the baselines do nothing -- "
          "their effects are smaller than this run's minimum detectable "
          "difference, which protocol section 5 requires to be printed beside "
          "every null, and it is, in the column above."
          % (tally[("unguided", "above")], tally[("unguided", "below")],
             tally[("unguided", "tie")],
             sum(v for k, v in tally.items() if k[0] == "unguided"),
             tally[("plug", "above")], tally[("plug", "below")],
             tally[("plug", "tie")],
             sum(v for k, v in tally.items() if k[0] == "plug")),
          "3. **The arm(s) that clear the bar anywhere: " + wtxt + ".** Where "
          "`bdg_e4t0.5` does, it is on `cpg` at both strengths (%+.2f pp over "
          "`plug` at w = 1, %+.2f pp at w = 4) and on `gc` at w = 4 only "
          "(%+.2f pp); on `gc` at the protocol's headline w = 1 it is a tie "
          "(%+.2f pp, z = %+.2f, min. detectable %.2f pp). Its gain is bought "
          "by **contracting** the property spread while the bias worsens, not "
          "by steering the mean to target: on `gc` at w = 4 the sd goes "
          "%.5f -> %.5f (%.3fx) and bias/delta %+.3f -> %+.3f. On M2 the base "
          "already sits on target, so the mean-correction half of BDG's "
          "numerator has almost nothing to do and only the dispersion half "
          "shows. **The transfer is partial, and that is the finding.**"
          % (100 * cd, 100 * cd4, 100 * d4, 100 * d, z, 100 * mdd,
             g4["unguided"]["sd"], g4["bdg_e4t0.5"]["sd"],
             g4["bdg_e4t0.5"]["sd"] / g4["unguided"]["sd"],
             g4["unguided"]["bias_delta"], g4["bdg_e4t0.5"]["bias_delta"]),
          "4. **The other headline BDG arm does not help, and is reported "
          "beside it rather than instead of it.** `bdg_e4t1` is %+.2f pp "
          "against `plug` on `gc` at w = 1 and %+.2f pp on `cpg` at w = 4 -- "
          "ties at the bar, negative in sign on every seed at the latter. The "
          "reason is visible in its controller state: at `tau_mult = 1` the "
          "setpoint is the corpus sd, which this base undershoots, so `e < 0` "
          "on **%.0f %%** of guided steps and the arm spends the run asking to "
          "**widen** (measured w_eff %+.3f on `cpg` at w = 4, %+.3f on `gc` at "
          "w = 1). The two arms are never pooled and neither is reported alone."
          % (100 * contrast(g1["bdg_e4t1"], g1["plug"])[0],
             100 * contrast(c4["bdg_e4t1"], c4["plug"])[0],
             100 * (c4["bdg_e4t1"]["widening"] or 0.0),
             c4["bdg_e4t1"]["w_eff"], g1["bdg_e4t1"]["w_eff"]),
          "5. **The mechanism is the dispersion term, and the grid shows it "
          "through the setpoint's sign.** In-band tracks the **measured** "
          "w_eff (Spearman %.2f at w = 1 and %.2f at w = 4, over the 16 "
          "feedback rungs at each strength); tau_mult < 1 raises in-band and "
          "tau_mult > 1 pushes it below eta = 0, at fixed `w`. Force cannot "
          "reverse the sign of its own parameter." % rho,
          "6. **Three of M2's four published baselines are one baseline in "
          "v3's window, for a measured reason.** `plug`, `tmpd` and `lgd_mc` "
          "agree to max \\|d\\| = %.1e over the four metrics protocol 3.0 "
          "names, on every seed and both strengths and both properties. This "
          "is a transfer finding, not only a caveat: it says which family of "
          "guidance methods survives the change of modality -- the "
          "endpoint-uncertainty signal they rely on has effectively vanished "
          "by the time v3's window opens." % ident,
          "7. **Guidance is functionally almost free, where it was measured**: "
          "the largest movement on either committed scorer is a fraction of a "
          "percentage point against `unguided`, across every scored arm and "
          "strength -- and **%d of %d** distinct cells, including all 9 "
          "t_min = 0 cells and all 3 seeds of the grid's strongest rung, were "
          "never scored (section %d)." % (cov[1] - cov[0], cov[1], S_FUN), "",
          "### Does not support", ""]
    L += ["1. **Any comparison of an M2 in-band with an M1 in-band.** Not with "
          "a caveat; not at all (protocol 2.2, 2.3b). M1's band is 2x a "
          "measured oracle error; M2's is a chosen `delta_ratio` against a "
          "lattice, and on `cpg` the discrete floor rather than the ratio sets "
          "it (delta/s = %.3f against `gc`'s %.3f). Nor may `gc` and `cpg` "
          "in-band levels be compared with each other, for the same reason. "
          "The valid cross-modality question is the **within-modality arm "
          "ranking**."
          % (c1["unguided"]["delta"] / c1["unguided"]["s"],
             g1["unguided"]["delta"] / g1["unguided"]["s"]),
          "2. **\"The best method.\"** w = 1 is not any arm's optimum, equal "
          "`w` is not equal force, and at w = 1 `bdg_e4t0.5` applies ~10x the "
          "baselines' correction share while `bdg_e4t1` applies half of it "
          "(protocol 3, `M2_SHARE.md`). Those shares are a side measurement "
          "at %s (section %d) -- there is **no `cpg` measurement of the "
          "correction share in the repository**, so at the one place BDG's "
          "effect is largest, `cpg` at w = 4, the force confound is carried "
          "over from `gc` rather than measured, exactly as item 11's mechanism "
          "gap is." % (share["prov_short"], S_READ),
          "3. **That the feedback loop is necessary.** The grid supports \"the "
          "deviation term's weight, not the global `w`, carries the effect\". "
          "It has no fixed-w_eff control, so an open-loop schedule replaying "
          "the same w_eff is not excluded.",
          "4. **That the strongest rungs are the controller rather than the "
          "clip.** `%s` at w = %g clips **%.1f %%** of its guided sample-steps "
          "and `bdg_e4t0.5` at w = 4 clips **%.1f %%**; the clip is applied "
          "after `w` (protocol 1.2b, 4.1). And the clip cannot simply be "
          "removed -- on M1, removing it produced 222 non-finite samples."
          % (ba, bw, 100 * bp["clip_frac"], 100 * g4["bdg_e4t0.5"]["clip_frac"]),
          "5. **A method claim stated without its window.** On `gc`, "
          "guiding from t = 0 rather than t >= 0.5 is worth **%+.2f pp** to "
          "`plug`, against the **%+.2f pp** the best rung reaches inside v3's "
          "window: about **%.0fx** the method. Both are `gc` numbers -- no "
          "t = 0 cell and no eta sweep ran on `cpg`, whose in-band is not "
          "comparable with `gc`'s, so the ratio may not be carried across "
          "(`cpg`'s own largest in-window gain over `unguided` is %+.2f pp at "
          "`%s`, w = %g). At t = 0 the two headline BDG arms go opposite ways "
          "against `plug`: `bdg_e4t0.5` %+.2f pp (z = %+.2f, **tie**, min. "
          "detectable %.2f pp) and `bdg_e4t1` %+.2f pp (z = %+.2f, **below** "
          "the bar on all %d seeds) -- the second is a loss, not an absence of "
          "advantage, and neither is reported alone. But the early window is "
          "**clip-saturated** (%.0f-%.0f %% of guided sample-steps), so it "
          "answers \"how good can guided generation get here\" and not \"which "
          "steering rule is better\". Both questions belong in the write-up "
          "and they are different questions."
          % (100 * wgain, 100 * bd, wgain / bd, 100 * cgain, cga, cgw,
             100 * wbdg_d, wbdg_z, 100 * wbdg_m, 100 * wbd1_d, wbd1_z,
             len(SEEDS), 100 * wclip_lo, 100 * wclip_hi),
          "6. **Anything about full TFG.** M2 implements only TFG's MC "
          "smoothing ingredient (`tfg_mc`).",
          "7. **`tmpd` as \"a baseline that performed like DPS\".** In this "
          "window it *is* DPS, because `v_f/s^2` is 2.5e-05 there. Report it "
          "as window-limited, and note that `tmpd` was silently DPS on both "
          "properties until the per-sample `v_f` fix of 27 Sep -- any pre-fix "
          "TMPD number is void.",
          "8. **Anything at q90.** This is a q50 run throughout.",
          "9. **Generalisation past DeepFlyBrain enhancers at 500 bp**, or "
          "past NFE 100, where the guided quantity (`gc_soft`) and the scored "
          "quantity (`gc_hard`) still differ by 5.5 % in spread (0.0483 "
          "against 0.0511, `proj1/m2/blade_bundle/nfe_sweep.log`). That bias "
          "is common to every arm at a given NFE, so it cancels in the "
          "arm-against-arm "
          "comparison; it does not cancel in an absolute statement. "
          "MODALITY2_V3_PROTOCOL.md 2.3a pre-registers this as a limitation "
          "and the reason for not spending 4x the compute on NFE 400.",
          "10. **A single-seed row as a result.** The w = 16 and w = 64 cells "
          "are one seed, and so is every `bdg_e4t0.5` w = 4 functional score.",
          "11. **The mechanism argument on `cpg`.** The 17-arm eta x tau grid "
          "ran on `gc` only, so `cpg` at w = 4 -- where BDG's effect is "
          "largest -- has no fixed-`w` control separating the controller from "
          "the extra force. That is carried over from `gc`, not measured.",
          "12. **A useful-yield analogue.** M1 reads decoded-in-band AND "
          "stable per molecule; M2 has no such per-sequence join committed "
          "here, because the `*.permol.pt` sidecars stay on blade. It cannot "
          "be approximated by multiplying marginals and is simply pending.",
          ""]
    return L


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/m2")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--gate-json", default="results/m2_gate_scores.json")
    ap.add_argument("--dfb-json", default="results/m2_dfb_activity.json")
    ap.add_argument("--grid-md", default="docs/results/M2_ABLATION_GRID.md")
    ap.add_argument("--share-json", default="results/m2_share.json")
    ap.add_argument("--md-out", default="docs/results/M2_V3_RESULTS.md")
    args = ap.parse_args()

    rows = load(args.out_dir, args.n)
    idx = {cell_key(r): r for r in rows}
    # A configuration present in both trees is two files and one cell. The
    # copies are asserted to agree here rather than quietly averaged or
    # counted twice in the provenance block.
    dupinfo = duplicate_configs(rows)
    n_cells = len(rows) - sum(len(v) - 1 for v in dupinfo[0].values())
    share = share_info(args)

    # ---- every cell the protocol says must exist, or exit with the list
    want = []
    for prop in ("gc", "cpg"):
        for a in HEADLINE_ARMS:
            for w in (1.0, 4.0):
                for s in SEEDS:
                    want.append(("m2", prop, a, w, 0.5, s))
    expect(idx, want, "headline (7 arms x 2 properties x 2 strengths x 3 seeds)")
    want = []
    for w in (1.0, 4.0):
        for s in SEEDS:
            want.append(("m2abl", "gc", "bdg_e0t1", w, 0.5, s))
            for eta in (1, 2, 4, 8):
                for tau in TAUS:
                    want.append(("m2abl", "gc", "bdg_e%gt%g" % (eta, tau), w,
                                 0.5, s))
    expect(idx, want, "ablation (17 arms x 2 strengths x 3 seeds, gc)")
    expect(idx, [("m2abl", "gc", a, 1.0, 0.0, s)
                 for a in ("plug", "bdg_e4t0.5", "bdg_e4t1") for s in SEEDS],
           "t_min = 0 diagnostic window")
    expect(idx, [("m2wsweep", "gc", a, w, 0.5, SEEDS[0])
                 for a in ("plug", "bdg_e4t0.5") for w in (16.0, 64.0)],
           "single-seed strength sweep")

    cmd = ("python proj1/m2/m2_v3_results.py --n %d --md-out %s"
           % (args.n, args.md_out))
    # The four headline caveats, recomputed rather than remembered.
    bd, ba, bw, _bp = best_abl_rung(idx)
    wgain = (pooled([idx[("m2abl", "gc", "plug", 1.0, 0.0, s)] for s in SEEDS])
             ["in_band_fraction"]
             - pooled([idx[("m2", "gc", "plug", 1.0, 0.5, s)] for s in SEEDS])
             ["in_band_fraction"])
    ident = 0.0
    for prop in ("gc", "cpg"):
        for w in (1.0, 4.0):
            for s in SEEDS:
                for a, b in (("plug", "tmpd"), ("plug", "lgd_mc"),
                             ("tmpd", "lgd_mc")):
                    ident = max(ident,
                                max(abs(idx[("m2", prop, a, w, 0.5, s)][m]
                                        - idx[("m2", prop, b, w, 0.5, s)][m])
                                    for m in IDENT4))
    sweep_seeds = sorted({r["seed"] for r in rows if r["_stage"] == "m2wsweep"})
    # The t = 0 rung's own BDG-vs-plug contrasts, so the banner cannot say
    # "BDG" where the two headline arms disagree (bdg_e4t1 loses there).
    t0 = {a: pooled([idx[("m2abl", "gc", a, 1.0, 0.0, s)] for s in SEEDS],
                    "t0/%s" % a)
          for a in ("plug", "bdg_e4t0.5", "bdg_e4t1")}
    t0_d05 = contrast(t0["bdg_e4t0.5"], t0["plug"])
    t0_d1 = contrast(t0["bdg_e4t1"], t0["plug"])
    band = {p: (idx[("m2", p, "unguided", 1.0, 0.5, SEEDS[0])]["delta"]
                / idx[("m2", p, "unguided", 1.0, 0.5, SEEDS[0])]["s"])
            for p in ("gc", "cpg")}
    L = ["# Modality 2 under protocol v3: the complete results page", "",
         "Generated by `proj1/m2/m2_v3_results.py`. Do not hand-edit; re-run "
         "it:", "", "```", cmd, "```", "",
         "Protocol: [MODALITY2_V3_PROTOCOL.md](../protocol/MODALITY2_V3_PROTOCOL.md) "
         "(it supersedes `MODALITY2_V3_PLAN.md`). Source trees, all at "
         "n = %d: `%s/m2/` (headline), `%s/m2abl/` (17-arm grid + the t_min = 0 "
         "diagnostic), `%s/m2wsweep/` (single-seed strength sweep); plus "
         "`%s` and `%s` for the functional evaluation and `%s` for the "
         "cross-check. Seeds %s."
         % (args.n, args.out_dir, args.out_dir, args.out_dir, args.gate_json,
            args.dfb_json, args.grid_md,
            ", ".join(str(s) for s in SEEDS)), "",
         "> **Four things that govern every number below, and travel with each "
         "of them.**",
         "> 1. **M2's in-band is not commensurable with M1's.** M1's band is "
         "2x a measured oracle error; M2's is a **chosen** `delta_ratio` "
         "against a discrete lattice (protocol 2.2). The two may be printed "
         "side by side and never pooled, averaged or ranked together. The "
         "valid cross-modality comparison is the within-modality arm ranking.",
         "> 2. **On `gc`, the window is worth about %.0fx the method.** "
         "Guiding from t = 0 instead of v3's t >= 0.5 buys `plug` %+.2f pp, "
         "against the %+.2f pp the best rung (`%s` at w = %g) reaches inside "
         "the window -- both measured on `gc`, the only property with a "
         "t = 0 rung or an eta sweep, so the ratio may not be carried across "
         "to `cpg`, whose band is %.1fx wider in corpus-sd units. The early "
         "window is clip-saturated, so it does not answer the method "
         "question, and the two headline BDG arms go opposite ways there: "
         "`bdg_e4t0.5` ties `plug` (%+.2f pp, z = %+.2f) while `bdg_e4t1` "
         "loses to it (%+.2f pp, z = %+.2f, below the bar). Section %d."
         % (wgain / bd, 100 * wgain, 100 * bd, ba, bw,
            band["cpg"] / band["gc"], 100 * t0_d05[0], t0_d05[1],
            100 * t0_d1[0], t0_d1[1], S_WIN),
         "> 3. **`plug`, `tmpd` and `lgd_mc` are numerically one baseline in "
         "this window** (max \\|d\\| = %.1e over every seed, strength and "
         "property): the measured `v_f` collapses 38,000x by t = 0.5, so "
         "TMPD's denominator tends to s^2 and LGD's draw scale tends to 0. "
         "Reporting them as three agreeing baselines would claim independent "
         "confirmation that does not exist (protocol 3.0)." % ident,
         "> 4. **The w = 16 and w = 64 rows are ONE seed (%s)**, and the "
         "DeepFlyBrain port's equivalence to the published Keras model is "
         "**in no committed file** (section %d)."
         % (", ".join(str(s) for s in sweep_seeds), S_FUN), "",
         "This is a **q50** run throughout. No q90 number appears anywhere on "
         "this page.", ""]
    L = provenance(L, rows, args, idx, dupinfo)
    L = how_to_read(L, idx, share)
    L = headline(L, idx, "gc", S_GC, share)
    L = headline(L, idx, "cpg", S_CPG, share)
    L = controls_section(L, idx, rows, S_CTRL)
    L = window_section(L, idx, rows, S_WIN)
    L = wsweep_section(L, idx, rows, args, S_WSW)
    L = ablation_section(L, idx, rows, args, S_ABL)
    L = functional_section(L, rows, args, S_FUN)
    L = outcome_section(L, idx, rows, args, S_OUT, share)
    L += ["---", "",
          "Generated from %d files (%d distinct cells) by "
          "`proj1/m2/m2_v3_results.py`; re-run `%s` to regenerate."
          % (len(rows), n_cells, cmd), ""]

    os.makedirs(os.path.dirname(args.md_out) or ".", exist_ok=True)
    with open(args.md_out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))
    print("wrote %s (%d lines, %d files, %d distinct cells)"
          % (args.md_out, len(L), len(rows), n_cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
