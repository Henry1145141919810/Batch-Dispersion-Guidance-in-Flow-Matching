"""Read the v2 fixed-target full run and print the headline table. CPU only.

    python proj1/scripts/full_run_v2_table.py
    python proj1/scripts/full_run_v2_table.py --md-out docs/results/FULL_RUN_V2_RESULTS.md

v2 is the pre-registered fixed-target protocol in
docs/protocol/FULL_RUN_V2_PROTOCOL.md (rules V1-V8). It differs from v1
(full_run_table.py, `dist` target) in three ways that matter here:

  * ONE target per property, the q90 of QM9 train, so `target_name == "q90"`
    and every molecule in a cell shares a target.
  * The strength of each arm was frozen on an n=512 screen (V2) and the run
    scores n=5000 x 3 seeds. Selection is not evaluation.
  * A fixed target makes mode collapse a winning strategy in principle, so
    V6 requires in_band to be reported beside distinctness, always.

THE FLOOR IS RE-MEASURED HERE, AT FULL SCALE, AND REPORTED WITH ITS
UNCERTAINTY. V2 picks each arm's strength among strengths clearing 0.9 x
unguided molecule stability ON THE SCREEN; V7 then decides the verdict only
between arms that clear the chemistry floor. Those are two measurements of one
quantity and they can disagree.

  Re-measuring at evaluation is what the documents license, not a goalpost
  move: under a selection-only reading V7's "between arms that both clear the
  chemistry floor" is vacuous, because V2 guarantees every frozen strength
  already cleared the screen (`frozen_v2.json` has floor_limited == []), and a
  pre-registered clause is not read as a no-op. V7 says "As v1's FR5", and
  v1's implementation states the floor is "measured here at full scale"
  (full_run_table.py). The project applied the same rule at full scale against
  the same arm in FR3A_VS_FR3_COMPARISON.md.

  BUT THE MARGIN IS NOT RESOLVED BY THIS RUN. mol_stability is a proportion
  with its own se, and so is the unguided reference the floor is 0.9 x. This
  script propagates both and prints the margin in sigma. On this run lgd_mc is
  -1.1 and -1.6 sigma under the floor on mu and gap -- floor-limited under the
  rule, indistinguishable from the floor in fact. It also prints the per-seed
  check, because a pooled mean can hide a seed that passes (FR3A precedent);
  on gap, one seed of three does pass. Nothing here supports "cannot hold the
  verdict whatever its in_band" as a statement about the world; it is a
  statement about the rule.

WHAT IT CHECKS BEFORE PRINTING, and refuses on:
  * every seed directory exists with every (property, arm) cell, exactly ONE
    file per (property, arm) -- a stale duplicate at a different strength is
    refused rather than resolved by filename order
  * each cell's seed/target/stage match its location and its strength equals
    frozen_v2.json's pick
  * one n, one generator, one frozen file, and one SAMPLER CONFIGURATION
    (t_min_guide, steps, solver, clip, batch) across every cell, and each
    property's target and delta single-valued
  * the three seeds are genuinely distinct runs (f_B hashes all differ)

WHAT IT PRINTS
  V4  the metric block per (property, arm) pooled over seeds, CONTINUOUS AND
      DECODED. The decoded view is not optional: V4 requires it "because arms
      that push the continuous type features are otherwise flattered", and on
      this run it changes which comparisons clear z >= 3.
  V5  error-ranked buckets, descriptive only (see the section's own caveat).
  V6  distinctness, per-cell minimum as well as pooled.
  V7  the verdict on in_band at sigma >= 3, on BOTH metrics, with the
      winner-vs-runner-up comparison shown, because V7 requires ties to be
      reported as ties and the runner-up comparison is where the ties are.
  V8  Holm within each property over the registered family of 12 tests.

Every sigma here is conditional on this target set and these sizes: all seeds
reuse the same 5,000 val sizes. The script computes the cluster-robust design
effect (clusters = molecule) and reports it, rather than leaving the caveat
in a docstring no reader of the table sees.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

from full_run_table import _sd, _z, pooled  # noqa: E402  same conventions

NL = chr(10)

PROPS = ("mu", "alpha", "gap")
# v2 first pass. btvg/btvg_var are the LATER pass (protocol Sec 4).
ARMS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg")
REFERENCE = "lgd_mc"          # v1's winner; V8's second comparator
SEEDS = ("20261001", "20261002", "20261003")
FLOOR = 0.9                   # saved rubric: mol_stability >= 0.9 x unguided
SIGMA = 3.0                   # V7
UNIQ_MIN = 0.95               # V6 collapse threshold
BUCKETS = (0.10, 0.50, 1.00)  # V5
# V8, protocol Sec 7: "each arm against unguided and against lgd_mc ...
# That is 2 x 6 x 3 = 36 tests", i.e. 12 per property as registered.
REGISTERED_FAMILY = 12
# Sec 2 caveat (b): alpha correlates +0.755 with size, so in_band MUST be
# reported by size stratum for it.
SIZE_STRATA = ((0, 17), (17, 20), (20, 23), (23, 99))
# Sec 6, measured on the n=512 screen: the winner changes with the budget.
BUDGET_TABLE = (
    ("mu", "tfg 0.053", "tmpd 0.061", "tfg 0.109"),
    ("alpha", "lgd_mc 0.045", "tmpd 0.066", "tmpd 0.066"),
    ("gap", "lgd_mc 0.113", "lgd_mc 0.113", "tfg 0.150"),
)
BUDGET_LINE = ("at a chemistry budget of 0.9x unguided stability")


def load_cells(root):
    """{seed: {(prop, arm): (row, sidecar_path)}}. Refuses duplicates."""
    out, dupes = {}, []
    for d in sorted(glob.glob(os.path.join(root, "seed*"))):
        seed = os.path.basename(d)[4:]
        out.setdefault(seed, {})
        for fn in sorted(glob.glob(os.path.join(d, "*__full.json"))):
            r = json.load(open(fn))
            if r.get("stage") != "full":
                continue
            key = (r["prop"], r["arm"])
            if key in out[seed]:
                dupes.append("seed %s has more than one cell for %s/%s (%s, %s)"
                             % (seed, key[0], key[1],
                                os.path.basename(out[seed][key][0]["_file"]),
                                os.path.basename(fn)))
                continue
            r["_file"] = fn
            side = fn[:-5] + ".permol.pt"
            out[seed][key] = (r, side if os.path.exists(side) else None)
    return out, dupes


def check(cells, dupes, frozen):
    """Refuse a partial, duplicated or inconsistent set."""
    problems = list(dupes)
    W = frozen["frozen_w"]
    for s in SEEDS:
        if s not in cells:
            problems.append("seed %s: no directory" % s)
    for seed in SEEDS:
        c = cells.get(seed, {})
        miss = ["%s/%s" % (p, a) for p in PROPS for a in ARMS if (p, a) not in c]
        if miss:
            problems.append("seed %s missing %s" % (seed, ", ".join(miss)))
        for p in PROPS:
            for a in ARMS:
                if (p, a) not in c:
                    continue
                r, side = c[(p, a)]
                if side is None:
                    problems.append("seed %s: no sidecar for %s/%s" % (seed, p, a))
                if str(r.get("seed")) != seed:
                    problems.append("%s/%s in seed%s says seed=%s"
                                    % (p, a, seed, r.get("seed")))
                if r.get("target_name") != "q90":
                    problems.append("%s/%s seed %s has target %r, not q90"
                                    % (p, a, seed, r.get("target_name")))
                if abs(float(r["w"]) - float(W[a][p])) > 1e-12:
                    problems.append("%s/%s seed %s ran at w=%g, frozen_v2 says %g"
                                    % (p, a, seed, float(r["w"]), float(W[a][p])))
    rows = [r for c in cells.values() for (r, _) in c.values() if r.get("arm") in ARMS]
    if not rows:
        return problems + ["no cells found at all"]
    # one generator, one frozen file, one n, ONE SAMPLER CONFIGURATION
    for key in ("n", "steps", "solver", "t_min_guide", "clip", "batch"):
        vals = {r.get(key) for r in rows}
        if len(vals) > 1:
            problems.append("mixed %s across cells: %s" % (key, sorted(map(str, vals))))
    for key, what in (("fm_md5", "generator"), ("frozen_md5", "frozen file")):
        vals = {(r.get("prov") or {}).get(key) for r in rows}
        if len(vals) > 1:
            problems.append("mixed %s across cells: %s" % (what, sorted(map(str, vals))))
    for p in PROPS:
        for key in ("target", "delta"):
            vals = {round(float(r[key]), 10) for r in rows if r["prop"] == p}
            if len(vals) > 1:
                problems.append("%s has mixed %s: %s" % (p, key, sorted(vals)))
    return problems


def check_seeds_distinct(cells):
    """The three seeds must be three runs, not one copied. Hash f_B."""
    import torch
    problems, seen = [], {}
    for p in PROPS:
        for a in ARMS:
            hs = {}
            for seed in SEEDS:
                side = cells[seed][(p, a)][1]
                t = torch.load(side, weights_only=False)["f_B"]
                h = hashlib.md5(t.double().numpy().tobytes()).hexdigest()
                hs.setdefault(h, []).append(seed)
            for h, ss in hs.items():
                if len(ss) > 1:
                    problems.append("%s/%s: seeds %s have identical f_B -- "
                                    "a duplicated run, not distinct seeds"
                                    % (p, a, ", ".join(ss)))
            seen[(p, a)] = hs
    return problems


def per_mol(cells, prop, arm):
    """One arm's sidecars concatenated in the fixed seed order."""
    import torch
    parts = []
    for seed in SEEDS:
        side = cells[seed][(prop, arm)][1]
        if side is None:
            return None
        parts.append(torch.load(side, weights_only=False))
    cat = lambda k: torch.cat([p[k] for p in parts])
    keys = ("f_B", "y", "finite", "mol_idx", "n_atoms", "mol_stable", "valid")
    out = {k: cat(k) for k in keys}
    for k in ("f_B_dec",):
        if all(k in p for p in parts):
            out[k] = cat(k)
    return out


def ib_vec(pm, delta, dec=False):
    """The in_band indicator, continuous or decoded."""
    import torch
    f = pm["f_B_dec"] if (dec and "f_B_dec" in pm) else pm["f_B"]
    return (((f - pm["y"]).abs() <= delta) & pm["finite"]).double()


def paired_z(a, b, delta, dec=False):
    """z of (a - b) in_band over the same molecules and the same noise."""
    import torch
    if not torch.equal(a["mol_idx"], b["mol_idx"]):
        raise SystemExit("sidecars do not pair: molecule order differs")
    d = ib_vec(a, delta, dec) - ib_vec(b, delta, dec)
    se = d.std(unbiased=True).item() / math.sqrt(d.numel())
    return d.mean().item() / se if se > 0 else float("nan")


def cluster_se(pm, delta, dec=False):
    """Cluster-robust se of an arm's in_band, clusters = molecule, and the
    design effect against the iid se. All seeds reuse the same molecules, so
    the rows are not independent."""
    import torch
    x = ib_vec(pm, delta, dec)
    N = x.numel()
    m = x.mean()
    r = x - m
    g = pm["mol_idx"].long()
    sums = torch.zeros(int(g.max()) + 1, dtype=torch.float64).index_add_(0, g, r)
    var_cl = float((sums ** 2).sum()) / (N ** 2)
    var_iid = float(m * (1 - m)) / N
    return math.sqrt(max(var_cl, 0.0)), (var_cl / var_iid if var_iid > 0 else float("nan"))


def prop_se(p, n):
    return math.sqrt(max(p * (1 - p), 0.0) / n)


def floor_margin(P, p):
    """(margin, se, z) of an arm against 0.9 x unguided, both estimated."""
    u = P[(p, "unguided")]
    out = {}
    for a in ARMS:
        r = P[(p, a)]
        fl = FLOOR * u["mol_stab"]
        se = math.sqrt(prop_se(r["mol_stab"], r["N"]) ** 2
                       + (FLOOR * prop_se(u["mol_stab"], u["N"])) ** 2)
        out[a] = (r["mol_stab"] - fl, se, (r["mol_stab"] - fl) / se if se > 0 else float("nan"))
    return out


def buckets(pm, delta):
    """V5. The metric block within the best q of the cell by |f_B - y|."""
    import torch
    e = (pm["f_B"] - pm["y"]).abs()
    e = torch.where(pm["finite"], e, torch.full_like(e, float("inf")))
    order = torch.argsort(e)
    out = []
    for q in BUCKETS:
        k = max(1, int(round(q * e.numel())))
        sel = order[:k]
        ok = pm["finite"][sel]
        err = e[sel][ok]
        out.append({
            "q": q, "k": k,
            "in_band": float((err <= delta).double().mean()) if err.numel() else float("nan"),
            "mae": float(err.mean()) if err.numel() else float("nan"),
            "mol_stab": float(pm["mol_stable"][sel].double().mean()),
            "validity": float(pm["valid"][sel].double().mean()),
        })
    return out


def pval(z):
    return math.erfc(abs(z) / math.sqrt(2)) if z == z else float("nan")


def holm(tests, m):
    order = sorted(tests, key=lambda t: pval(t[1]))
    out, prev = {}, 0.0
    for i, (key, z) in enumerate(order):
        p = min(1.0, max(prev, pval(z) * max(m - i, 1)))
        out[key] = p
        prev = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(ROOT, "results", "full", "v2", "n5000"))
    ap.add_argument("--frozen", default="")
    ap.add_argument("--md-out", default="")
    args = ap.parse_args()

    root = args.root
    frozen_path = args.frozen or os.path.join(root, "frozen_v2.json")
    frozen = json.load(open(frozen_path))
    cells, dupes = load_cells(root)

    problems = check(cells, dupes, frozen)
    if problems:
        print("REFUSING: the v2 set is not complete or not consistent.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)

    import torch  # noqa: F401
    problems = check_seeds_distinct(cells)
    if problems:
        print("REFUSING: the seeds are not distinct runs.")
        for p in problems:
            print("  * " + p)
        raise SystemExit(1)

    L = []
    W = lambda a, p: float(frozen["frozen_w"][a][p])
    any_cell = cells[SEEDS[0]][(PROPS[0], ARMS[0])][0]
    prov = any_cell.get("prov") or {}
    DELTA = {p: cells[SEEDS[0]][(p, ARMS[0])][0]["delta"] for p in PROPS}
    TARGET = {p: cells[SEEDS[0]][(p, ARMS[0])][0]["target"] for p in PROPS}

    P, PM = {}, {}
    for p in PROPS:
        for a in ARMS:
            rows = [cells[s][(p, a)][0] for s in SEEDS]
            pm = per_mol(cells, p, a)
            PM[(p, a)] = pm
            d = pooled(rows, pm)
            N = d["N"]
            d["in_band_dec"] = sum(r["in_band_fraction_dec"] * r["n"] for r in rows) / N
            d["mae_dec"] = sum(r["prop_mae_eval_dec"] * r["n"] for r in rows) / N
            d["se_ib_dec"] = prop_se(d["in_band_dec"], N)
            d["uniq_valid"] = sum(r["uniqueness_of_valid"] * r["n"] for r in rows) / N
            d["uvps"] = sum(r["unique_valid_per_sample"] * r["n"] for r in rows) / N
            d["uniq_min_cell"] = min(r["uniqueness_of_valid"] for r in rows)
            d["atom_stab"] = sum(r["atom_stability"] * r["n"] for r in rows) / N
            d["diversity"] = sum(r["diversity_mean_pairwise"] * r["n"] for r in rows) / N
            d["guide_gap"] = sum(r["guide_eval_gap_mean"] * r["n"] for r in rows) / N
            d["nonfinite"] = sum(r["n_nonfinite"] for r in rows)
            d["field_evals"] = rows[0].get("field_evals", 0)
            c = rows[0].get("cost") or {}
            d["nfe"] = sum(v for v in c.values() if isinstance(v, (int, float)))
            d["clip_steps"] = sum(r.get("clipped_sample_steps", 0) for r in rows)
            cse, deff = cluster_se(pm, DELTA[p])
            d["cse_ib"], d["deff"] = cse, deff
            P[(p, a)] = d

    FL = {p: floor_margin(P, p) for p in PROPS}
    clears = lambda p, a: P[(p, a)]["mol_stab"] >= FLOOR * P[(p, "unguided")]["mol_stab"] - 1e-12

    # ---- header ---------------------------------------------------------
    L.append("# Full run v2 - fixed-target results (q90), n = 5,000 x 3 seeds")
    L.append("")
    L.append("Generated by `proj1/scripts/full_run_v2_table.py`. Do not hand-edit; "
             "re-run the script.")
    L.append("")
    L.append("Protocol: [FULL_RUN_V2_PROTOCOL.md](../protocol/FULL_RUN_V2_PROTOCOL.md) "
             "(rules V1-V8, pre-registered 24 Sep 2026, before any v2 cell existed).")
    L.append("")
    L.append("> **Every verdict on this page holds %s, and nowhere else.** The "
             "protocol's Sec 6 measured on the screen that the ranking changes at "
             "0.7x and 0.5x; only the 0.9x budget was run. The budget-dependence "
             "table is reproduced below and must travel with any sentence quoted "
             "from here." % BUDGET_LINE)
    L.append("")
    L.append("| provenance | value |")
    L.append("|---|---|")
    L.append("| cells | %d = %d properties x %d arms x %d seeds |"
             % (len(PROPS) * len(ARMS) * len(SEEDS), len(PROPS), len(ARMS), len(SEEDS)))
    L.append("| distinct generations | %d - the %d `unguided` cells are %d runs "
             "reused across properties, since unguided ignores the target |"
             % (len(PROPS) * len(ARMS) * len(SEEDS) - 2 * len(SEEDS),
                len(PROPS) * len(SEEDS), len(SEEDS)))
    L.append("| n per cell | %d |" % any_cell["n"])
    L.append("| seeds | %s (f_B hashes verified distinct) |" % ", ".join(SEEDS))
    L.append("| generator | `%s` md5 `%s` |"
             % (any_cell.get("fm", "?"), (prov.get("fm_md5") or "?")[:8]))
    L.append("| frozen strengths | `%s` md5 `%s` |"
             % (os.path.relpath(frozen_path, ROOT).replace(os.sep, "/"),
                (prov.get("frozen_md5") or "?")[:8]))
    L.append("| screen the strengths came from | stage `%s`, target `%s`, n = %s, seed %s |"
             % (frozen.get("source_stage"), frozen.get("target"),
                frozen.get("source_n"), frozen.get("source_seed")))
    L.append("| sampler | %d-step %s, guidance window t >= %g, velocity clip %g, batch %d |"
             % (any_cell["steps"], any_cell["solver"], any_cell["t_min_guide"],
                any_cell["clip"], any_cell["batch"]))
    L.append("| device | %s, torch %s |" % (prov.get("device"), prov.get("torch")))
    L.append("| non-finite samples | %d across all cells |"
             % sum(P[(p, a)]["nonfinite"] for p in PROPS for a in ARMS))
    L.append("")
    L.append("**Arms present: %s. `btvg` and `btvg_var` are NOT in this run.** The "
             "protocol queues them as a later pass (Sec 4), so everything below "
             "compares the external methods and the ablation rung `plug` to each "
             "other and to `unguided`. No sentence here is a verdict on our own "
             "method." % ", ".join("`%s`" % a for a in ARMS))
    L.append("")
    L.append("**Figures.** The pick is one point on a curve, and V2b requires the "
             "curve: `../results/figs/strength_{mu,alpha,gap}.png` (what each "
             "strength buys and costs, pick starred) and `pareto_{mu,alpha,gap}.png` "
             "(the trade-off every arm is on). **No claim here rests on the pick "
             "alone.** Regenerate with `python proj1/scripts/protocol_v2_figures.py`.")
    L.append("")
    L.append("### Sec 6 - the ranking depends on the chemistry budget (screen, n = 512)")
    L.append("")
    L.append("| property | at 0.9x unguided (run here) | at 0.7x | at 0.5x |")
    L.append("|---|---|---|---|")
    for row in BUDGET_TABLE:
        L.append("| %s | **%s** | %s | %s |" % row)
    L.append("")
    L.append("Only the first column was run at full scale. The other two are the "
             "screen's answer and are shown because reporting the single-budget "
             "ranking as if it were budget-free would be wrong on our own data.")
    L.append("")

    # ---- V4 -------------------------------------------------------------
    L.append("## V4 - the metric block, continuous and decoded")
    L.append("")
    L.append("**The decoded columns are not a footnote.** V4 requires every "
             "property metric in its decoded (argmax one-hot atom types) form "
             "\"because arms that push the continuous type features are otherwise "
             "flattered\". On this run the two metrics disagree about which "
             "comparisons clear z >= %g, so the continuous columns alone would "
             "overstate the result." % SIGMA)
    L.append("")
    for p in PROPS:
        u = P[(p, "unguided")]
        fl = FLOOR * u["mol_stab"]
        L.append("### %s (target %.4f, delta %.5f)" % (p, TARGET[p], DELTA[p]))
        L.append("")
        L.append("Floor = %.4f = 0.9 x unguided %.4f, both measured at n = %d. "
                 "The floor is itself an estimate; the margin column gives each "
                 "arm's distance from it in sigma, propagating the se of both."
                 % (fl, u["mol_stab"], u["N"]))
        L.append("")
        L.append("| arm | w | in_band | +-se | in_band (dec) | MAE/d | MAE/d (dec) | "
                 "bias/d | resid sd/d | mol_stab | margin vs floor | atom_stab | valid |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        d = DELTA[p]
        for a in sorted(ARMS, key=lambda x: -P[(p, x)]["in_band"]):
            r = P[(p, a)]
            mg, mse, mz = FL[p][a]
            L.append("| `%s` | %g | %.4f | %.4f | %.4f | %.3f | %.3f | %+.3f | %.3f | "
                     "%.4f | %+.4f (%+.2f s) | %.4f | %.4f |"
                     % (a, W(a, p), r["in_band"], r["se_ib"], r["in_band_dec"],
                        r["mae"] / d, r["mae_dec"] / d, r["bias"] / d,
                        r["resid_sd"] / d, r["mol_stab"], mg, mz,
                        r["atom_stab"], r["validity"]))
        L.append("")
        L.append("| arm | RMSE/d | diversity | guide-eval gap | non-finite | "
                 "field evals | total NFE | clipped steps |")
        L.append("|---|---|---|---|---|---|---|---|")
        for a in sorted(ARMS, key=lambda x: -P[(p, x)]["in_band"]):
            r = P[(p, a)]
            L.append("| `%s` | %.3f | %.4f | %.4f | %d | %d | %d | %d |"
                     % (a, r["rmse"] / d, r["diversity"], r["guide_gap"],
                        r["nonfinite"], r["field_evals"], r["nfe"], r["clip_steps"]))
        L.append("")

    # ---- floor, in detail -----------------------------------------------
    L.append("### The chemistry floor, and how well this run resolves it")
    L.append("")
    L.append("An arm below the floor is **floor-limited** and does not hold the "
             "verdict under V7. That is a rule, and the rule is applied here. It "
             "is not the same as a measured finding that the arm is worse: the "
             "margins below are small relative to their own uncertainty.")
    L.append("")
    L.append("| property | arm | mol_stab | floor | margin | se | sigma | per-seed pass |")
    L.append("|---|---|---|---|---|---|---|---|")
    for p in PROPS:
        fl = FLOOR * P[(p, "unguided")]["mol_stab"]
        for a in ARMS:
            if a == "unguided":
                continue
            mg, mse, mz = FL[p][a]
            per = []
            for s in SEEDS:
                av = cells[s][(p, a)][0]["mol_stability"]
                uv = cells[s][(p, "unguided")][0]["mol_stability"]
                per.append("Y" if av >= FLOOR * uv else "n")
            L.append("| %s | `%s` | %.4f | %.4f | %+.5f | %.5f | **%+.2f** | %s |"
                     % (p, a, P[(p, a)]["mol_stab"], fl, mg, mse, mz,
                        " ".join(per)))
    L.append("")
    lim = [(p, a) for p in PROPS for a in ARMS if a != "unguided" and not clears(p, a)]
    if lim:
        L.append("**Floor-limited: %s.** Each is within %.1f sigma of the floor, so "
                 "this run does not resolve whether the arm is truly below it -- it "
                 "resolves only that the rule's estimate puts it below. The per-seed "
                 "column shows the pooled call is not unanimous everywhere."
                 % (", ".join("`%s` on %s" % (a, p) for p, a in lim),
                    max(abs(FL[p][a][2]) for p, a in lim)))
        L.append("")
        L.append("**What is not measured:** no full-scale cell exists for a "
                 "floor-limited arm at any *other* strength on these properties, so "
                 "what it would deliver at a strength that does clear the floor is "
                 "unknown. The screen suggests it would be less; the run cannot say.")
        L.append("")

    # ---- alpha size strata (Sec 2 caveat b) ------------------------------
    L.append("### in_band by size stratum (alpha)")
    L.append("")
    L.append("Sec 2 caveat (b): alpha correlates +0.755 with molecule size, so a "
             "fixed high alpha is far easier at 23 atoms than at 15, and in_band "
             "**must** be reported by stratum for it.")
    L.append("")
    L.append("| arm | " + " | ".join("%d-%d atoms" % s if s[1] < 99 else "%d+ atoms" % s[0]
                                     for s in SIZE_STRATA) + " |")
    L.append("|---|" + "---|" * len(SIZE_STRATA))
    for a in sorted(ARMS, key=lambda x: -P[("alpha", x)]["in_band"]):
        pm = PM[("alpha", a)]
        v = ib_vec(pm, DELTA["alpha"])
        cellsr = []
        for lo, hi in SIZE_STRATA:
            sel = (pm["n_atoms"] >= lo) & (pm["n_atoms"] < hi)
            k = int(sel.sum())
            cellsr.append("%.4f (n=%d)" % (float(v[sel].mean()), k) if k else "--")
        L.append("| `%s` | %s |" % (a, " | ".join(cellsr)))
    L.append("")

    # ---- V6 -------------------------------------------------------------
    L.append("## V6 - the distinctness guard")
    L.append("")
    L.append("A fixed target makes emitting one in-band molecule 5,000 times a "
             "winning strategy in principle, so in_band is reported beside "
             "distinctness. Threshold fixed in the protocol: uniqueness of valid "
             "below %.2f is collapse-contaminated." % UNIQ_MIN)
    L.append("")
    L.append("| property | arm | uniqueness (pooled) | lowest single cell | "
             "distinct valid per attempt | verdict |")
    L.append("|---|---|---|---|---|---|")
    worst_cell = 1.0
    for p in PROPS:
        for a in ARMS:
            r = P[(p, a)]
            worst_cell = min(worst_cell, r["uniq_min_cell"])
            L.append("| %s | `%s` | %.4f | %.5f | %.4f | %s |"
                     % (p, a, r["uniq_valid"], r["uniq_min_cell"], r["uvps"],
                        "ok" if r["uniq_min_cell"] >= UNIQ_MIN else "**COLLAPSE**"))
    L.append("")
    L.append("Lowest uniqueness of any **single cell** in the run: **%.5f**, against "
             "the %.2f threshold. Collapse is not happening; the guard is reported "
             "because it was promised, not because it separates the arms."
             % (worst_cell, UNIQ_MIN))
    L.append("")

    # ---- V7 -------------------------------------------------------------
    L.append("## V7 - the verdict")
    L.append("")
    L.append("Decided on in_band at independent-samples z >= %g, between arms that "
             "both clear the chemistry floor. Paired z (same molecules, same "
             "initial noise) supplementary. **V7 also requires ties to be reported "
             "as ties**, so the winner-vs-runner-up comparison is shown: it is "
             "where the ties are." % SIGMA)
    L.append("")
    for p in PROPS:
        d = DELTA[p]
        elig = [a for a in ARMS if clears(p, a)]
        best = max(elig, key=lambda a: P[(p, a)]["in_band"])
        L.append("### %s" % p)
        L.append("")
        L.append("| arm | in_band | vs unguided z_ind | z_pair | **z_ind (dec)** | "
                 "vs `%s` z_ind | z_pair | chem |" % REFERENCE)
        L.append("|---|---|---|---|---|---|---|---|")
        for a in sorted(ARMS, key=lambda x: -P[(p, x)]["in_band"]):
            r, u, ref = P[(p, a)], P[(p, "unguided")], P[(p, REFERENCE)]
            f = lambda v: "--" if v != v else "%+.2f" % v
            zu = (float("nan") if a == "unguided" else
                  _z(r["in_band"] - u["in_band"],
                     math.sqrt(r["se_ib"] ** 2 + u["se_ib"] ** 2)))
            zud = (float("nan") if a == "unguided" else
                   _z(r["in_band_dec"] - u["in_band_dec"],
                      math.sqrt(r["se_ib_dec"] ** 2 + u["se_ib_dec"] ** 2)))
            zr = (float("nan") if a == REFERENCE else
                  _z(r["in_band"] - ref["in_band"],
                     math.sqrt(r["se_ib"] ** 2 + ref["se_ib"] ** 2)))
            zup = paired_z(PM[(p, a)], PM[(p, "unguided")], d) if a != "unguided" else float("nan")
            zrp = paired_z(PM[(p, a)], PM[(p, REFERENCE)], d) if a != REFERENCE else float("nan")
            L.append("| `%s` | %.4f | %s | %s | **%s** | %s | %s | %s |"
                     % (a, r["in_band"], f(zu), f(zup), f(zud), f(zr), f(zrp),
                        "ok" if clears(p, a) else "**floor-limited**"))
        L.append("")
        # winner vs runner-up
        rest = sorted([a for a in elig if a != best],
                      key=lambda a: -P[(p, a)]["in_band"])
        L.append("| top arm vs | in_band gap | z_ind | z_pair | separated at %g s? |" % SIGMA)
        L.append("|---|---|---|---|---|")
        ties = []
        for a in rest:
            gap = P[(p, best)]["in_band"] - P[(p, a)]["in_band"]
            z = _z(gap, math.sqrt(P[(p, best)]["se_ib"] ** 2 + P[(p, a)]["se_ib"] ** 2))
            zp = paired_z(PM[(p, best)], PM[(p, a)], d)
            if abs(z) < SIGMA:
                ties.append(a)
            L.append("| `%s` | %+.4f | %+.2f | %+.2f | %s |"
                     % (a, gap, z, zp, "yes" if abs(z) >= SIGMA else "**no - tie**"))
        L.append("")
        zb = _z(P[(p, best)]["in_band"] - P[(p, "unguided")]["in_band"],
                math.sqrt(P[(p, best)]["se_ib"] ** 2 + P[(p, "unguided")]["se_ib"] ** 2))
        zbd = _z(P[(p, best)]["in_band_dec"] - P[(p, "unguided")]["in_band_dec"],
                 math.sqrt(P[(p, best)]["se_ib_dec"] ** 2 + P[(p, "unguided")]["se_ib_dec"] ** 2))
        excl = [a for a in ARMS if a not in elig]
        s = ("**%s: the highest point estimate among floor-clearing arms is `%s`** "
             "at in_band %.4f (z = %+.2f vs unguided on the continuous metric, "
             "**%+.2f decoded**)." % (p, best, P[(p, best)]["in_band"], zb, zbd))
        if ties:
            s += (" It is **not separated** from %s (all below %g sigma), so under "
                  "V7 these are reported as tied, not as a win."
                  % (", ".join("`%s`" % a for a in ties), SIGMA))
        if excl:
            s += (" %s %s floor-limited here and does not hold the verdict under the "
                  "rule; see the margin table for how thin that call is."
                  % (", ".join("`%s`" % a for a in excl),
                     "is" if len(excl) == 1 else "are"))
        if abs(zbd) < SIGMA:
            s += (" **On the decoded metric V4 requires, this arm does not clear "
                  "z >= %g against unguided.**" % SIGMA)
        L.append(s)
        L.append("")

    # ---- clustered se ---------------------------------------------------
    L.append("### Resolving power, and what the sigmas are conditional on")
    L.append("")
    L.append("All three seeds reuse the same 5,000 `val` sizes and the same fixed "
             "target, so rows are clustered by molecule and the iid se understates "
             "a claim about the target population. The design effect is computed "
             "here rather than assumed:")
    L.append("")
    L.append("| property | design effect (min-max over arms) | iid se | clustered se |")
    L.append("|---|---|---|---|")
    for p in PROPS:
        ds = [P[(p, a)]["deff"] for a in ARMS]
        L.append("| %s | %.2f - %.2f | %.4f | %.4f |"
                 % (p, min(ds), max(ds), P[(p, "unguided")]["se_ib"],
                    P[(p, "unguided")]["cse_ib"]))
    L.append("")
    L.append("The inflation is **small** (v1's sqrt(3) worst case does not apply: "
             "the fixed target removes v1's per-molecule target clustering). "
             "Applying it to the headline comparisons:")
    L.append("")
    L.append("| comparison | z (iid) | z (clustered) | still >= %g? |" % SIGMA)
    L.append("|---|---|---|---|")
    for p in PROPS:
        elig = [a for a in ARMS if clears(p, a)]
        best = max(elig, key=lambda a: P[(p, a)]["in_band"])
        r, u = P[(p, best)], P[(p, "unguided")]
        z1 = _z(r["in_band"] - u["in_band"], math.sqrt(r["se_ib"] ** 2 + u["se_ib"] ** 2))
        z2 = _z(r["in_band"] - u["in_band"], math.sqrt(r["cse_ib"] ** 2 + u["cse_ib"] ** 2))
        L.append("| %s: `%s` vs unguided | %+.2f | %+.2f | %s |"
                 % (p, best, z1, z2, "yes" if abs(z2) >= SIGMA else "**no**"))
    L.append("")

    # ---- V8 -------------------------------------------------------------
    L.append("## V8 - multiplicity")
    L.append("")
    L.append("Holm within each property over the registered family: each arm "
             "against `unguided` and against `%s`, which the protocol fixes at "
             "2 x 6 x 3 = 36 tests, i.e. **%d per property**. With "
             "`btvg`/`btvg_var` absent only the tests below exist, and Holm is "
             "applied at the registered size."
             % (REFERENCE, REGISTERED_FAMILY))
    L.append("")
    L.append("**Why the registered size is the right one to use, and why it is "
             "conservative:** Holm's adjusted p at fixed m is a running maximum of "
             "p_(j) x (m - j + 1). Any test added later has a raw p no larger than "
             "the test it displaces, so adding the two missing arms can only "
             "**lower** these adjusted p-values. The column below is therefore a "
             "valid upper bound on the final answer -- a result significant here "
             "stays significant when the family is completed.")
    L.append("")
    L.append("**A tension this run exposes, unresolved:** V7 says a floor-limited "
             "arm can neither win nor be beaten, yet the family's second half tests "
             "every arm *against* `%s`, which is floor-limited on mu and gap. Those "
             "rows are reported and marked, but they cannot both be valid tests and "
             "comparisons against an arm that cannot be beaten. Treat them as "
             "descriptive." % REFERENCE)
    L.append("")
    for p in PROPS:
        tests, seen = [], set()
        for base in ("unguided", REFERENCE):
            for a in ARMS:
                if a == base:
                    continue
                key = tuple(sorted((a, base)))
                if key in seen:
                    continue
                seen.add(key)
                r, b = P[(p, a)], P[(p, base)]
                z = _z(r["in_band"] - b["in_band"],
                       math.sqrt(r["se_ib"] ** 2 + b["se_ib"] ** 2))
                tests.append(((a, base), z))
        h = holm(tests, REGISTERED_FAMILY)
        L.append("### %s" % p)
        L.append("")
        L.append("| test | z_ind | raw p | Holm (m = %d) | z >= %g | note |"
                 % (REGISTERED_FAMILY, SIGMA))
        L.append("|---|---|---|---|---|---|")
        for key, z in sorted(tests, key=lambda t: pval(t[1])):
            note = ("vs floor-limited arm" if (REFERENCE in key and not clears(p, REFERENCE))
                    else "")
            L.append("| `%s` vs `%s` | %+.2f | %.2e | %.4f | %s | %s |"
                     % (key[0], key[1], z, pval(z), h[key],
                        "yes" if abs(z) >= SIGMA else "no", note))
        L.append("")

    # ---- V5 -------------------------------------------------------------
    L.append("## V5 - error-ranked buckets")
    L.append("")
    L.append("**Descriptive only.** Molecules are sorted by |f_B - y| and the block "
             "is reported within the best 10 %, 50 % and 100 % of each cell. This "
             "ranks molecules by the same oracle it then scores them with, so no "
             "row here is a yield anyone could reproduce -- a user does not have "
             "f_B at generation time. Its purpose is the SHAPE of an arm's error "
             "distribution. These numbers are never compared against another "
             "method's achievable yield.")
    L.append("")
    L.append("**The in_band columns are arithmetic, not evidence.** When a cell's "
             "in_band is at or below 10 %, every in-band molecule already sits "
             "inside the best-10 % bucket, so `in_band @10 %` is exactly 10 x "
             "`@100 %` and `@50 %` exactly 2 x. The first of these holds for every "
             "cell in this run except `gap`/`lgd_mc` (in_band 0.117 > 0.10, so its "
             "@10 % saturates at 1.0000); the `@50 %` identity holds for that cell "
             "too. The informative columns here are **MAE/d**, **mol_stab** and "
             "**valid**: they say whether an arm's best decile is also its soundest.")
    L.append("")
    for p in PROPS:
        d = DELTA[p]
        L.append("### %s" % p)
        L.append("")
        L.append("| arm | in_band @10 / 50 / 100 % | MAE/d @10 / 50 / 100 % | "
                 "mol_stab @10 / 50 / 100 % | valid @10 / 50 / 100 % |")
        L.append("|---|---|---|---|---|")
        for a in sorted(ARMS, key=lambda x: -P[(p, x)]["in_band"]):
            b = buckets(PM[(p, a)], d)
            L.append("| `%s` | %.4f / %.4f / %.4f | %.3f / %.3f / %.3f | "
                     "%.4f / %.4f / %.4f | %.4f / %.4f / %.4f |"
                     % (a, b[0]["in_band"], b[1]["in_band"], b[2]["in_band"],
                        b[0]["mae"] / d, b[1]["mae"] / d, b[2]["mae"] / d,
                        b[0]["mol_stab"], b[1]["mol_stab"], b[2]["mol_stab"],
                        b[0]["validity"], b[1]["validity"], b[2]["validity"]))
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
