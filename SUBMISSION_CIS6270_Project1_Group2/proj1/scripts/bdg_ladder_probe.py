#!/usr/bin/env python
"""What BDG's controller actually does on a real trajectory, per rung.

WHY THIS EXISTS. FULL_RUN_V3_PROTOCOL.md section 4.1 once pre-registered a
closed form for BDG's deviation weight,

    w_eff = 1 + eta*(1/tau_mult^2 - 1)   ->   13.00 / 4.11 / 1.00 / -1.22

derived from `test_bdg.py` part B, which evaluates the field on ONE synthetic
x_t where V_b happens to equal y_std^2. On a real guided trajectory it does not:
V_b/tau^2 run-means at 64-257, so w_eff run-means at 111-1023, two orders of
magnitude above the closed form. The rung ORDER survives; the numbers do not.

WHAT IT MEASURES, and why a mean is not enough. w_eff is not mean-like: the
review measured it changing sign up to 34 times in a 50-step window. So this
prints three numbers per rung, from the diagnostics guidance.py records:

    w_eff mean    the signed run-mean (what a cell used to report alone)
    w_eff RMS     sqrt(mean(bdg_w_eff_sq)) -- magnitude, immune to cancellation
    frac < 0      mean(bdg_w_eff_neg) -- the share of guided steps on which the
                  deviation term REVERSED, i.e. spread the batch instead of
                  concentrating it

plus the clipped-step count beside plug's, because the clip is load-bearing here:
at w_eff ~ 1000 the dispersion term is far outside the trust region and the clip
truncates most of it, so a rung can be clip-limited rather than controller-limited.

    python proj1/scripts/bdg_ladder_probe.py                       # eta=4 ladder
    python proj1/scripts/bdg_ladder_probe.py --etas 0,1,2,4,8 --tau-mults 0.5,1
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TS = os.path.join("proj1", "scripts", "transfer_sweep.py")


def fmt(arm):
    return arm if arm == "plug" else arm


def run(a, arm):
    out = os.path.join("results", "_ladder", arm)
    full = os.path.join(ROOT, out)
    if os.path.isdir(full):
        import shutil
        shutil.rmtree(full)          # transfer_sweep resumes by file existence
    cmd = [sys.executable, TS, "--stage", "v3", "--backend", a.backend,
           "--props", a.prop, "--arms", arm, "--n", str(a.n),
           "--batch", str(a.n), "--steps", str(a.steps), "--seed", str(a.seed),
           "--fm-ckpt", a.fm_ckpt, "--out-dir", out]
    p = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                       capture_output=True, text=True)
    hits = glob.glob(os.path.join(full, "*.json"))
    if not hits:
        print("  %-14s FAILED" % arm)
        for l in [x for x in (p.stderr or "").splitlines() if x.strip()][-3:]:
            print("     | %s" % l[:150])
        return None
    r = json.load(open(hits[0]))
    d = r.get("diag") or {}
    row = {"arm": arm, "in_band": r["in_band_fraction"],
           "in_band_dec": r.get("in_band_fraction_dec"),
           "clipped": r["clipped_sample_steps"],
           "guided_steps": r.get("guided_steps"),
           "mol_stab": r.get("mol_stability"), "validity": r.get("validity")}
    if "bdg_w_eff" in d:
        row.update({"v_over_tau2": d["bdg_V_over_tau2"],
                    "w_eff": d["bdg_w_eff"],
                    "w_eff_rms": math.sqrt(max(0.0, d.get("bdg_w_eff_sq", 0.0))),
                    "frac_neg": d.get("bdg_w_eff_neg"),
                    "tau": d.get("bdg_tau"), "V_b": d.get("bdg_V_b"),
                    "disp_rms": d.get("bdg_disp_rms"),
                    "dev_rms": d.get("bdg_dev_rms")})
        if "bdg_w_eff_sq" not in d:
            row["stale"] = True
    print("  %-14s in_band %.4f  clipped %5d%s"
          % (arm, row["in_band"], row["clipped"],
             ("  w_eff %.1f (rms %.1f, %.0f%% negative)"
              % (row["w_eff"], row["w_eff_rms"], 100 * row["frac_neg"]))
             if "w_eff" in row else ""))
    return row


def closed_form(eta, tau):
    return 1.0 + eta * (1.0 / (tau * tau) - 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--etas", default="4")
    ap.add_argument("--tau-mults", default="0.5,0.75,1,1.5")
    ap.add_argument("--with-eta0", action="store_true", default=True,
                    help="include bdg_e0t1, the identity rung (w_eff == 1 by "
                         "construction, whatever V_b does)")
    ap.add_argument("--backend", default="fm")
    ap.add_argument("--prop", default="mu")
    ap.add_argument("--n", type=int, default=128, help="also the batch: one controller")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261001)
    ap.add_argument("--fm-ckpt", default="proj1/checkpoints/fm_last.pt")
    ap.add_argument("--json-out", default="results/bdg_ladder_probe.json")
    ap.add_argument("--md-out", default="")
    a = ap.parse_args()

    etas = [float(x) for x in a.etas.split(",") if x]
    taus = [float(x) for x in a.tau_mults.split(",") if x]
    arms = ["plug"] + (["bdg_e0t1"] if a.with_eta0 else [])
    for eta in etas:
        for t in taus:
            arm = "bdg_e%gt%g" % (eta, t)
            if arm not in arms:
                arms.append(arm)

    print("ladder on %s / %s, n = batch = %d (ONE controller), %d steps, seed %d"
          % (a.backend, a.prop, a.n, a.steps, a.seed))
    rows = [r for r in (run(a, arm) for arm in arms) if r]
    if any(r.get("stale") for r in rows):
        print("\nWARNING: cells lack bdg_w_eff_sq/_neg -- the code predates them, "
              "so only the signed mean is available and cancellation is invisible.")
    rec = {"backend": a.backend, "prop": a.prop, "n": a.n, "steps": a.steps,
           "seed": a.seed, "etas": etas, "tau_mults": taus, "rows": rows}
    jo = os.path.join(ROOT, a.json_out)
    os.makedirs(os.path.dirname(jo), exist_ok=True)
    json.dump(rec, open(jo, "w"), indent=2, sort_keys=True)
    print("\nwrote %s" % a.json_out)
    if a.md_out:
        md = os.path.join(ROOT, a.md_out)
        os.makedirs(os.path.dirname(md), exist_ok=True)
        open(md, "w").write(render(rec))
        print("wrote %s" % a.md_out)
    return 0


def render(rec):
    base = next((r for r in rec["rows"] if r["arm"] == "plug"), None)
    L = ["# What BDG's controller actually does, per rung", "",
         "Generated by `proj1/scripts/bdg_ladder_probe.py`. %s / %s, "
         "n = batch = %d (ONE controller, so no pooling over batches), %d steps, "
         "guidance on for t >= 0.5, seed %d."
         % (rec["backend"], rec["prop"], rec["n"], rec["steps"], rec["seed"]),
         "",
         "`w_eff` is BDG's weight on the **deviation** term. The signed mean is "
         "not enough on its own: the quantity changes sign within a window, so a "
         "violently active arm can average to nearly nothing. **RMS** gives its "
         "magnitude and **frac < 0** the share of guided steps on which the term "
         "reversed -- spreading the batch rather than concentrating it.", "",
         "| arm | V_b/tau^2 | w_eff mean | w_eff RMS | frac < 0 | closed form | in_band | clipped steps |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rec["rows"]:
        cf = "-"
        if r["arm"].startswith("bdg_e"):
            eta, tau = r["arm"][len("bdg_e"):].split("t")
            cf = "%.2f" % closed_form(float(eta), float(tau.rstrip("o")))
        g = lambda k, f: (f % r[k]) if r.get(k) is not None else "-"   # noqa: E731
        L.append("| `%s` | %s | %s | %s | %s | %s | %.4f | %d |"
                 % (r["arm"], g("v_over_tau2", "%.1f"), g("w_eff", "%.1f"),
                    g("w_eff_rms", "%.1f"), g("frac_neg", "%.3f"), cf,
                    r["in_band"], r["clipped"]))
    L += ["", "**The closed form is wrong on a real trajectory.** "
              "`w_eff = 1 + eta(1/tau_mult^2 - 1)` assumes `V_b = y_std^2`, which "
              "holds on the single synthetic `x_t` in `test_bdg.py` part B and "
              "not during sampling, where `V_b/tau^2` run-means in the tens to "
              "hundreds. The rung ORDER is preserved; the magnitudes are two "
              "orders of magnitude out. Order rows by `w_eff`, and quote the "
              "measured value, never the closed form."]
    if base:
        worst = max((r for r in rec["rows"] if r.get("w_eff") is not None),
                    key=lambda r: r["clipped"], default=None)
        if worst:
            L += ["", "**The clip is a live confound.** `plug` clips on %d sample "
                      "steps; `%s` clips on **%d**. At `w_eff` of order 10^3 the "
                      "dispersion term sits far outside the velocity-relative "
                      "trust region, so what reaches the sample is the clip's "
                      "truncation of it. A rung can be clip-limited rather than "
                      "controller-limited, and the two are not distinguished by "
                      "any metric in this run."
                  % (base["clipped"], worst["arm"], worst["clipped"])]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
