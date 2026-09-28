"""The w_eff TRAJECTORY, which the cell-level run-mean hides.

Why this exists. The handoff's section 3 concedes that BDG is exactly
`plug` at weight w_eff = 1 + eta*e aiming at y_eff, and then argues the
FEEDBACK is irreducible because freezing w_eff at its TIME-AVERAGE and running
plain plug fails to reproduce BDG. That control is only as strong as the
time-average is representative. If w_eff swings across a wide range -- or
changes SIGN -- over the guided window, then "a constant equal to the mean of a
sign-changing sequence does not reproduce the sequence" is close to a
tautology, and the control cannot carry the irreducibility claim.

So: record e, V_b/tau^2 and w_eff at EVERY guided step, for the same setup the
grid runs, and report the range, the sign changes and the time-average.

Also checks the bound that actually licenses removing btvg's clamp:
V_b >= 0 forces e >= -1 EXACTLY, hence w_eff >= 1 - eta, with no dynamical
argument needed. btvg's coefficient -0.5(1/tau^2 - 1/V_F) has no such bound
because V_F -> 0.

One sampling process, n=256, batch=256 -- the same resource envelope as a
grid cell.
"""
import json
import os
import sys

import torch

ROOT = os.environ.get("BDG_ROOT")
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from checkpoint_paths import require_predictor  # noqa: E402
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from sampling import FlowSampler, initial_noise, integrate  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weff_traj.json")
TARGETS = {"mu": {"q50": 2.4932, "q90": 4.6627},
           "gap": {"q50": 0.2496, "q90": 0.3162}}


def run(prop, tgt, eta, mult, dev, n=256, steps=100, seed=20260925):
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    net, _ = load_fm(os.path.join(ROOT, "weights", "fm_ema.pt"), len(types), dev)
    f_A = PhysicalProperty(require_predictor("f_A_%s.pt" % prop), len(types), dev)
    idx = d["split"]["val"][:n]
    mask = d["mask"][idx].to(dev)
    s = float(f_A.y_std)
    y = torch.full((n,), TARGETS[prop][tgt], device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    c0, f0 = initial_noise(mask, len(types), gen)
    smp = FlowSampler(net, mask, f_net=f_A, y=y, s=s, mode="bdg", w=1.0,
                      clip=1.0, t_min_guide=0.5,
                      bdg_eta=eta, bdg_tau=mult * s)
    traj = []
    orig = smp._accumulate_diag

    def hook(diag):
        orig(diag)
        if "bdg_e" in diag:
            traj.append({k: float(diag[k].double().mean())
                         for k in ("bdg_e", "bdg_e_raw", "bdg_V_over_tau2",
                                   "bdg_w_eff", "bdg_disp_rms", "bdg_dev_rms")})

    smp._accumulate_diag = hook
    integrate(smp, c0, f0, steps, "euler")
    return {"prop": prop, "target": tgt, "eta": eta, "tau_mult": mult,
            "s": s, "tau": mult * s, "n_guided_steps": len(traj), "traj": traj}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for prop in ("mu", "gap"):
        for tgt in ("q50", "q90"):
            for mult in (0.5, 1.0, 1.5):
                r = run(prop, tgt, 4.0, mult, dev)
                w = [x["bdg_w_eff"] for x in r["traj"]]
                e = [x["bdg_e"] for x in r["traj"]]
                r["w_eff_min"], r["w_eff_max"] = min(w), max(w)
                r["w_eff_mean"] = sum(w) / len(w)
                r["w_eff_sign_changes"] = sum(
                    1 for a, b in zip(w, w[1:]) if (a > 0) != (b > 0))
                r["e_min"], r["e_max"] = min(e), max(e)
                print("%-4s %-4s t%-4g  steps %3d  e [%+8.4f, %+8.4f]  "
                      "w_eff [%+8.3f, %+8.3f] mean %+8.3f  sign changes %d"
                      % (prop, tgt, mult, len(w), r["e_min"], r["e_max"],
                         r["w_eff_min"], r["w_eff_max"], r["w_eff_mean"],
                         r["w_eff_sign_changes"]))
                out.append(r)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("\nwrote %s" % OUT)
    print("\nBOUND CHECK: e >= -1 exactly (V_b >= 0), so w_eff >= 1 - eta = -3.")
    print("observed min e = %+.6f, min w_eff = %+.4f"
          % (min(r["e_min"] for r in out), min(r["w_eff_min"] for r in out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
