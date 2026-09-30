"""Pick two operating points per (property, arm) from the joint tune sweep.

    python proj1/scripts/freeze_tune.py \
        --root results/tune/n1000 \
        --json-out results/tune/n1000/frozen_tune.json \
        --md-out docs/results/TUNE_SWEEP_TABLE.md

WHAT IT PICKS, for every (property, arm):

  floor   the (strength, t_min_guide) with the best in-band among cells whose
          molecule stability is at least FLOOR x the unguided cell's -- the
          v2 rubric's chemistry budget, unchanged.
  open    the (strength, t_min_guide) with the best in-band, with NO chemistry
          constraint. This is deliberately not a recommendation: it is the
          ceiling the method can reach when chemistry is allowed to collapse,
          and it exists so the floor's cost is visible rather than asserted.

TIE-BREAKS, fixed here before the data exists, applied in order: higher
in-band, then SMALLER strength, then LARGER t_min_guide (start guiding later
= intervene less). Both are "least intervention wins", which is the same
direction v2's "ties to the smaller w" already had.

THE SELECTION METRIC IS THE CONTINUOUS in_band, which is v2's rule
unchanged. That is a deliberate choice, not an oversight: V4 warns the
continuous metric flatters arms that push the continuous atom-type features,
and the v2 full run showed two of three headline comparisons crossing z = 3 on
continuous but not decoded. So this script ALSO computes what each pick would
have been on the decoded metric and flags every (property, arm) where the two
disagree. Changing the selection metric is a protocol change and is not made
silently here; the flag is the evidence for making it deliberately later.

WHAT IT REFUSES (exit 2), because each would silently corrupt the freeze:
  * a missing unguided cell -- the floor is undefined without it
  * a missing (arm, strength, window) cell, unless --allow-partial
  * cells sampled under different deltas, n, seed, generator or sampler
    settings. delta is not only a scoring knob: it sets BTVG's tau, so btvg
    cells from two deltas are different experiments.
  * a non-finite in-band or stability

WHAT IT ONLY FLAGS, because refusing would block a chained full run for
something a reader can weigh:
  * grid-limited: the floor pick sits at the top of the strength grid AND
    still clears the floor, so the grid, not chemistry, is capping the arm
  * window-limited: the pick sits at the edge of the window grid
  * collapse: uniqueness of valid below UNIQ_MIN at the pick
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

NL = chr(10)
PROPS = ("mu", "alpha", "gap")
FLOOR = 0.9            # the saved rubric: mol_stability >= 0.9 x unguided
UNIQ_MIN = 0.95        # V6's collapse threshold
# Our FM base sits at ~0.39-0.40 unguided molecule stability on every screen so
# far. Anything near zero means the run is broken, not that the arm is good.
MIN_UNGUIDED_STAB = 0.15
SCHEMA = "frozen_tune/1"
# every cell of one tune tree must agree on these, or the cells are not one
# experiment. `delta` is here because it sets BTVG's tau (guidance_sweep
# .arm_kwargs), so it changes what was SAMPLED, not just how it was scored.
SAMPLER_KEYS = ("n", "seed", "steps", "solver", "clip", "batch", "k_delta",
                "n_mc", "sigma_mc", "target_name")


def load(root):
    """{(prop, arm, w, win): row}, plus problems found while reading."""
    cells, problems = {}, []
    for fn in sorted(glob.glob(os.path.join(root, "*.json"))):
        if os.path.basename(fn).startswith("frozen"):
            continue
        try:
            r = json.load(open(fn))
        except Exception as e:                       # noqa: BLE001
            problems.append("%s: unreadable (%s)" % (os.path.basename(fn), e))
            continue
        if "arm" not in r or "prop" not in r:
            continue
        k = (r["prop"], r["arm"], float(r["w"]), float(r["t_min_guide"]))
        if k in cells:
            problems.append("two cells for %s/%s w=%g tmin=%g" % k)
            continue
        r["_file"] = os.path.basename(fn)
        cells[k] = r
    return cells, problems


def check(cells, problems, props, arms, strengths, windows, allow_partial):
    rows = list(cells.values())
    if not rows:
        return problems + ["no cells found"]
    for key in SAMPLER_KEYS:
        vals = {r.get(key) for r in rows}
        if len(vals) > 1:
            problems.append("cells disagree on %s: %s"
                            % (key, sorted(map(str, vals))))
    vals = {(r.get("prov") or {}).get("fm_md5") for r in rows}
    if len(vals) > 1:
        problems.append("cells used different generators: %s"
                        % sorted(map(str, vals)))
    for p in props:
        ds = {round(float(r["delta"]), 12) for r in rows if r["prop"] == p}
        if len(ds) > 1:
            problems.append("%s: cells sampled under different deltas %s -- "
                            "btvg's tau is delta/1.96, so these are not one "
                            "experiment" % (p, sorted(ds)))
        srcs = {r.get("delta_source") for r in rows if r["prop"] == p}
        if len(srcs) > 1:
            problems.append("%s: mixed delta_source %s" % (p, sorted(map(str, srcs))))
    for p in props:
        u = [r for k, r in cells.items() if k[0] == p and k[1] == "unguided"]
        if not u:
            problems.append("%s: no unguided cell -- the chemistry floor is "
                            "undefined" % p)
            continue
        # A floor of ~0 is worse than no floor: every cell "clears" it, the
        # two picks collapse into one, and the table looks fine. This is what
        # a too-small n or a broken generator produces.
        ms = float(u[0]["mol_stability"])
        if ms < MIN_UNGUIDED_STAB:
            problems.append(
                "%s: unguided molecule stability is %.4f, below %.2f. The "
                "chemistry floor would be %.4f, which every cell clears "
                "trivially, so the floor-constrained and unconstrained picks "
                "would be the same number by construction. Check n and the "
                "generator before freezing."
                % (p, ms, MIN_UNGUIDED_STAB, FLOOR * ms))
    missing = []
    for p in props:
        for a in arms:
            if a == "unguided":
                continue
            for w in strengths:
                for win in windows:
                    if (p, a, float(w), float(win)) not in cells:
                        missing.append("%s/%s w=%g tmin=%g" % (p, a, w, win))
    if missing:
        msg = "%d planned cells are missing (e.g. %s)" % (
            len(missing), ", ".join(missing[:4]))
        if allow_partial:
            print("WARNING: %s -- continuing because --allow-partial" % msg)
        else:
            problems.append(msg)
    for k, r in cells.items():
        for f in ("in_band_fraction", "in_band_fraction_dec", "mol_stability"):
            v = r.get(f)
            if v is None or not math.isfinite(float(v)):
                problems.append("%s/%s w=%g tmin=%g has non-finite %s" % (k + (f,)))
    return problems


def unguided_of(cells, prop):
    got = [r for k, r in cells.items() if k[0] == prop and k[1] == "unguided"]
    return got[0] if got else None


def best(cands, key):
    """Highest `key`; ties to the smaller strength, then the larger window."""
    return sorted(cands, key=lambda r: (-float(r[key]), float(r["w"]),
                                        -float(r["t_min_guide"])))[0]


def pick_for(cells, prop, arm, floor_value, metric):
    cand = [r for k, r in cells.items() if k[0] == prop and k[1] == arm
            and r.get(metric) is not None]
    if not cand:
        return None, None
    ok = [r for r in cand if float(r["mol_stability"]) >= floor_value - 1e-12]
    return (best(ok, metric) if ok else None), best(cand, metric)


def as_pick(r, floor_value, strengths, windows):
    if r is None:
        return None
    return {
        "w": float(r["w"]), "t_min_guide": float(r["t_min_guide"]),
        "in_band": float(r["in_band_fraction"]),
        "in_band_dec": r.get("in_band_fraction_dec"),
        "mol_stability": float(r["mol_stability"]),
        "validity": float(r["validity"]),
        "uniqueness_of_valid": float(r["uniqueness_of_valid"]),
        "unique_valid_per_sample": float(r["unique_valid_per_sample"]),
        "prop_mae_eval": float(r["prop_mae_eval"]),
        "clears_floor": bool(float(r["mol_stability"]) >= floor_value - 1e-12),
        "at_max_strength": float(r["w"]) >= max(strengths) - 1e-12,
        "at_window_edge": float(r["t_min_guide"]) in (min(windows), max(windows)),
        "window_open_ended": False,   # set by flag_window_open_ended below
        "collapse": float(r["uniqueness_of_valid"]) < UNIQ_MIN,
        "file": r["_file"],
    }


def flag_window_open_ended(cells, prop, arm, pick, windows):
    """True when the pick sits at an END of the window grid AND beats the
    window next to it, at the same strength -- i.e. the trend is still rising
    when the grid runs out, so the best start time may lie outside what was
    measured. Flagging every edge pick instead would fire on two of three
    windows and mean nothing."""
    if pick is None or not pick["at_window_edge"]:
        return False
    ws = sorted(windows)
    t = float(pick["t_min_guide"])
    nb = ws[1] if t == ws[0] else ws[-2]
    here = cells.get((prop, arm, float(pick["w"]), t))
    there = cells.get((prop, arm, float(pick["w"]), float(nb)))
    if here is None or there is None:
        return False
    return float(here["in_band_fraction"]) > float(there["in_band_fraction"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "tune", "n1000"))
    ap.add_argument("--json-out", default="")
    ap.add_argument("--md-out", default="")
    ap.add_argument("--props", default=",".join(PROPS))
    ap.add_argument("--arms", default="")
    ap.add_argument("--allow-partial", action="store_true",
                    help="freeze from whatever is on disk. The result is "
                         "labelled partial and records what was missing.")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    from guidance_sweep import TUNE_ARMS, TUNE_STRENGTHS, TUNE_WINDOWS

    props = [p for p in args.props.split(",") if p]
    arms = [a for a in args.arms.split(",") if a] or list(TUNE_ARMS)
    root = args.root if os.path.isabs(args.root) else os.path.join(ROOT, args.root)

    cells, problems = load(root)
    problems = check(cells, problems, props, arms, TUNE_STRENGTHS,
                     TUNE_WINDOWS, args.allow_partial)
    if problems:
        print("REFUSING to freeze: the tune sweep is not a usable experiment.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(2)

    any_row = next(iter(cells.values()))
    out = {
        "schema": SCHEMA,
        "rule": ("floor = best in_band with mol_stability >= %g x unguided; "
                 "open = best in_band, unconstrained. Ties: higher in_band, "
                 "then smaller w, then larger t_min_guide." % FLOOR),
        "selection_metric": "in_band_fraction (continuous), v2's rule",
        "floor_factor": FLOOR,
        "target": any_row.get("target_name"),
        "n": any_row.get("n"), "seed": any_row.get("seed"),
        "delta": {}, "delta_source": any_row.get("delta_source"),
        "floor": {}, "unguided": {},
        "strengths": list(TUNE_STRENGTHS), "windows": list(TUNE_WINDOWS),
        "partial": bool(args.allow_partial),
        "n_cells_read": len(cells),
        "prov": {k: (any_row.get("prov") or {}).get(k)
                 for k in ("fm_path", "fm_md5", "torch", "cuda", "device")},
        "picks": {a: {} for a in arms},
        "flags": {"grid_limited": [], "window_edge": [], "collapse": [],
                  "no_floor_clearing_cell": [], "metric_disagrees": [],
                  "no_cells": []},
    }

    for p in props:
        u = unguided_of(cells, p)
        fl = FLOOR * float(u["mol_stability"])
        out["floor"][p] = fl
        out["delta"][p] = float(u["delta"])
        out["unguided"][p] = {
            "mol_stability": float(u["mol_stability"]),
            "in_band": float(u["in_band_fraction"]),
            "in_band_dec": u.get("in_band_fraction_dec"),
            "validity": float(u["validity"]), "file": u["_file"],
        }
        for a in arms:
            if a == "unguided":
                out["picks"][a][p] = {
                    "floor": as_pick(u, fl, TUNE_STRENGTHS, TUNE_WINDOWS),
                    "open": as_pick(u, fl, TUNE_STRENGTHS, TUNE_WINDOWS),
                    "same": True}
                continue
            f_r, o_r = pick_for(cells, p, a, fl, "in_band_fraction")
            fd, od = pick_for(cells, p, a, fl, "in_band_fraction_dec")
            fp = as_pick(f_r, fl, TUNE_STRENGTHS, TUNE_WINDOWS)
            op = as_pick(o_r, fl, TUNE_STRENGTHS, TUNE_WINDOWS)
            same = bool(fp and op and fp["w"] == op["w"]
                        and fp["t_min_guide"] == op["t_min_guide"])
            out["picks"][a][p] = {
                "floor": fp, "open": op, "same": same,
                "floor_decoded": as_pick(fd, fl, TUNE_STRENGTHS, TUNE_WINDOWS),
                "open_decoded": as_pick(od, fl, TUNE_STRENGTHS, TUNE_WINDOWS)}
            if fp is None and op is None:
                out["flags"]["no_cells"].append("%s/%s" % (p, a))
            elif fp is None:
                out["flags"]["no_floor_clearing_cell"].append("%s/%s" % (p, a))
            else:
                if fp["at_max_strength"]:
                    out["flags"]["grid_limited"].append("%s/%s floor" % (p, a))
                if flag_window_open_ended(cells, p, a, fp, TUNE_WINDOWS):
                    fp["window_open_ended"] = True
                    out["flags"]["window_edge"].append("%s/%s floor" % (p, a))
                if fp["collapse"]:
                    out["flags"]["collapse"].append("%s/%s floor" % (p, a))
            if op:
                if op["collapse"]:
                    out["flags"]["collapse"].append("%s/%s open" % (p, a))
                if op["at_max_strength"]:
                    # `open` exists to measure the ceiling; a grid-capped
                    # ceiling is a lower bound wearing a ceiling's label
                    out["flags"]["grid_limited"].append("%s/%s open" % (p, a))
            # would the decoded metric have chosen differently?
            for which, cont, dec in (("floor", f_r, fd), ("open", o_r, od)):
                if cont is not None and dec is not None and (
                        float(cont["w"]) != float(dec["w"])
                        or float(cont["t_min_guide"]) != float(dec["t_min_guide"])):
                    out["flags"]["metric_disagrees"].append(
                        "%s/%s %s: continuous picks w=%g t=%g, decoded picks "
                        "w=%g t=%g" % (p, a, which, float(cont["w"]),
                                       float(cont["t_min_guide"]),
                                       float(dec["w"]), float(dec["t_min_guide"])))

    # the deduplicated cell list the full run will sample
    runs, seen = [], set()
    for p in props:
        for a in arms:
            for which in ("floor", "open"):
                pk = out["picks"][a][p].get(which)
                if not pk:
                    continue
                k = (p, a, pk["w"], pk["t_min_guide"])
                if k in seen:
                    continue
                seen.add(k)
                runs.append({"prop": p, "arm": a, "w": pk["w"],
                             "t_min_guide": pk["t_min_guide"]})
    out["full_run_cells"] = runs
    out["n_full_run_cells_per_seed"] = len(runs)

    print("froze %d (property, arm) pairs from %d cells" % (len(props) * len(arms), len(cells)))
    for p in props:
        print("  %-6s floor = %.4f (0.9 x unguided %.4f), delta %.5f"
              % (p, out["floor"][p], out["unguided"][p]["mol_stability"], out["delta"][p]))
        for a in arms:
            pk = out["picks"][a][p]
            f, o = pk["floor"], pk["open"]
            print("    %-9s floor: %s | open: %s%s"
                  % (a,
                     "none" if not f else "w=%-5g t=%-4g ib=%.4f stab=%.4f"
                     % (f["w"], f["t_min_guide"], f["in_band"], f["mol_stability"]),
                     "none" if not o else "w=%-5g t=%-4g ib=%.4f stab=%.4f"
                     % (o["w"], o["t_min_guide"], o["in_band"], o["mol_stability"]),
                     "  [same]" if pk["same"] else ""))
    print("full run will sample %d cells per seed (after de-duplication)" % len(runs))
    for k, v in out["flags"].items():
        if v:
            print("FLAG %s: %s" % (k, "; ".join(v[:6]) + (" ..." if len(v) > 6 else "")))

    if args.json_out:
        dst = args.json_out if os.path.isabs(args.json_out) else os.path.join(ROOT, args.json_out)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(out, open(dst, "w"), indent=1, sort_keys=True)
        print("wrote %s" % dst)
    if args.md_out:
        write_table(out, cells, props, arms, args.md_out)
    return out


def write_table(out, cells, props, arms, md_out):
    """The full sweep table: every (w, t_min) cell, per property and arm."""
    L = []
    L.append("# Joint strength x start-time sweep (tune stage)")
    L.append("")
    L.append("Generated by `proj1/scripts/freeze_tune.py`. Do not hand-edit; "
             "re-run the script.")
    L.append("")
    L.append("| setting | value |")
    L.append("|---|---|")
    L.append("| target | `%s`, fixed per property |" % out["target"])
    L.append("| n per cell, seed | %s, %s |" % (out["n"], out["seed"]))
    L.append("| in-band delta | %s |" % out["delta_source"])
    L.append("| | " + ", ".join("%s %.5f" % (p, out["delta"][p]) for p in props) + " |")
    L.append("| generator | `%s` md5 `%s` |"
             % (os.path.basename(str(out["prov"].get("fm_path"))),
                str(out["prov"].get("fm_md5"))[:8]))
    L.append("| strengths | %s |" % ", ".join("%g" % w for w in out["strengths"]))
    L.append("| t_min_guide | %s |" % ", ".join("%g" % w for w in out["windows"]))
    L.append("| cells read | %d%s |" % (out["n_cells_read"],
                                        " (PARTIAL)" if out["partial"] else ""))
    L.append("| selection | %s |" % out["selection_metric"])
    L.append("| rule | %s |" % out["rule"])
    L.append("")
    L.append("**Two operating points per arm.** `floor` is the best in-band that "
             "keeps molecule stability at or above %g x unguided; `open` is the "
             "best in-band with no chemistry constraint. `open` is **not a "
             "recommendation** -- it is the ceiling the method reaches when "
             "chemistry is allowed to fall, reported so the floor's cost is "
             "visible." % out["floor_factor"])
    L.append("")

    L.append("## The picks")
    L.append("")
    L.append("| property | arm | floor: w, t | in_band | dec | stab | valid | "
             "open: w, t | in_band | dec | stab | valid | same? |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    fmt = lambda v, f="%.4f": "--" if v is None else f % v
    for p in props:
        for a in arms:
            pk = out["picks"][a][p]
            f, o = pk["floor"], pk["open"]
            cell = lambda x: ("none" if not x else "w=%g, t=%g" % (x["w"], x["t_min_guide"]))
            L.append("| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                     % (p, a, cell(f),
                        fmt(f and f["in_band"]), fmt(f and f["in_band_dec"]),
                        fmt(f and f["mol_stability"]), fmt(f and f["validity"]),
                        cell(o),
                        fmt(o and o["in_band"]), fmt(o and o["in_band_dec"]),
                        fmt(o and o["mol_stability"]), fmt(o and o["validity"]),
                        "yes" if pk["same"] else "no"))
    L.append("")
    L.append("Chemistry floor per property: " +
             ", ".join("%s %.4f (0.9 x unguided %.4f)"
                       % (p, out["floor"][p], out["unguided"][p]["mol_stability"])
                       for p in props) + ".")
    L.append("")
    flag_help = {
        "grid_limited": "the floor pick sits at the TOP of the strength grid and "
                        "still clears the floor, so the grid -- not chemistry -- "
                        "is capping this arm. Its floor number is a lower bound.",
        "window_edge": "the pick sits at an END of the t_min grid AND beats the "
                       "window next to it, so the best start time may lie "
                       "outside the three that were measured.",
        "collapse": "uniqueness of valid below %.2f at this pick: the in-band "
                    "number is collapse-contaminated." % UNIQ_MIN,
        "no_floor_clearing_cell": "this arm ran, but no cell of it cleared the "
                                  "chemistry floor anywhere in the grid.",
        "no_cells": "this arm produced no usable cell at all -- it did not run, "
                    "or every cell of it failed. Not a result about the arm.",
        "metric_disagrees": "the continuous and decoded metrics choose different "
                            "operating points. Selection used the continuous one "
                            "(v2's rule); this is the evidence for revisiting it.",
    }
    any_flag = any(out["flags"].values())
    L.append("## Flags")
    L.append("")
    if not any_flag:
        L.append("None. No arm is grid-limited, no pick sits at a window edge, "
                 "no pick is collapse-contaminated, and the continuous and "
                 "decoded metrics agree on every pick.")
    for k, v in out["flags"].items():
        if not v:
            continue
        L.append("**%s** -- %s" % (k, flag_help.get(k, "")))
        L.append("")
        for item in v:
            L.append("- %s" % item)
        L.append("")
    L.append("")

    L.append("## The full grid")
    L.append("")
    L.append("Each cell is `in_band / mol_stab`. **Bold** = clears the chemistry "
             "floor. The floor pick is marked `[F]`, the open pick `[O]`.")
    L.append("")
    wins = out["windows"]
    for p in props:
        fl = out["floor"][p]
        L.append("### %s (floor %.4f, delta %.5f)" % (p, fl, out["delta"][p]))
        L.append("")
        for a in arms:
            if a == "unguided":
                u = out["unguided"][p]
                L.append("`unguided`: in_band %.4f, mol_stab %.4f, validity %.4f "
                         "(one cell; guidance knobs do not apply)."
                         % (u["in_band"], u["mol_stability"], u["validity"]))
                L.append("")
                continue
            pk = out["picks"][a][p]
            L.append("**`%s`**" % a)
            L.append("")
            L.append("| w \\ t_min | " + " | ".join("%g" % t for t in wins) + " |")
            L.append("|---|" + "---|" * len(wins))
            for w in out["strengths"]:
                row = []
                for t in wins:
                    r = cells.get((p, a, float(w), float(t)))
                    if r is None:
                        row.append("--")
                        continue
                    ib, ms = float(r["in_band_fraction"]), float(r["mol_stability"])
                    txt = "%.4f / %.3f" % (ib, ms)
                    if ms >= fl - 1e-12:
                        txt = "**%s**" % txt
                    for which, tag in (("floor", "F"), ("open", "O")):
                        x = pk.get(which)
                        if x and x["w"] == float(w) and x["t_min_guide"] == float(t):
                            txt += " `[%s]`" % tag
                    row.append(txt)
                L.append("| %g | %s |" % (w, " | ".join(row)))
            L.append("")
    out_txt = NL.join(L) + NL
    dst = md_out if os.path.isabs(md_out) else os.path.join(ROOT, md_out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(out_txt)
    print("wrote %s" % dst)


if __name__ == "__main__":
    main()
