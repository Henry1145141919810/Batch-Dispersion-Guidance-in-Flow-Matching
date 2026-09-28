"""THE CONTROL THE REDUCTION ACTUALLY NEEDS (run on the real generator, GPU).

Handoff section 3 concedes BDG is exactly `plug` at weight w_eff = 1 + eta*e
aiming at y_eff, then rescues the novelty claim with one control: freeze w_eff
at its TIME-AVERAGE, run plain plug, observe it does not reproduce BDG. That
control conflates two different things -- "no feedback" and "constant in time".
The w_eff trajectory measured on this generator changes sign up to 34 times
inside a single 50-step guided window and spans [-2.64, +10.35], so a constant
was never going to reproduce it, and the control cannot carry the claim.

The clean version separates the two. Record the e SCHEDULE from a closed-loop
run on seed A, then on FRESH NOISE (seed B) run:

    plug     eta = 0                       no schedule at all
    closed   the live controller           feedback ON,  schedule from seed B
    replay   e supplied from seed A's log  feedback OFF, schedule from seed A

If `replay` lands on `closed`, the feedback adds nothing the schedule does not
already carry, and BDG is a transferable time-varying weight schedule -- the
strong form of the section-3 reduction, and fatal to "the loop is irreducible".
If `replay` misses `closed` and sits nearer `plug`, the per-batch feedback is
doing real work and the handoff's claim survives in a form a reviewer cannot
wave away.

Replaying on seed A would be circular -- the closed loop on seed A produced
exactly that schedule -- so the whole test is that the noise is different.
"""
import json
import os
import sys

import torch

ROOT = os.environ["BDG_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from checkpoint_paths import require_predictor  # noqa: E402
from evaluation import evaluate_samples  # noqa: E402
from m1_signed_bias import PhysicalProperty, load_fm  # noqa: E402
from sampling import FlowSampler, initial_noise, integrate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "openloop_replay_gpu.json")
TARGETS = {"mu": {"q50": 2.4932, "q90": 4.6627},
           "gap": {"q50": 0.2496, "q90": 0.3162}}
SEED_A, SEED_B = 20260925, 20260926


class Replay(FlowSampler):
    """A FlowSampler that hands guidance_field a pre-recorded e each step."""

    def __init__(self, *a, e_schedule=None, **kw):
        super().__init__(*a, **kw)
        self.e_schedule = e_schedule
        self.i_guided = 0

    def field(self, coords, feats, t_scalar):
        if self.e_schedule is not None:
            j = min(self.i_guided, len(self.e_schedule) - 1)
            self.bdg_e_override = float(self.e_schedule[j])
        before = self.n_guided
        out = super().field(coords, feats, t_scalar)
        if self.n_guided > before:            # this step really was guided
            self.i_guided += 1
        return out


def load_env(prop, dev):
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    net, _ = load_fm(os.path.join(ROOT, "weights", "fm_ema.pt"), len(types), dev)
    fa_p, fb_p = (require_predictor("f_A_%s.pt" % prop),
                  require_predictor("f_B_%s.pt" % prop))
    f_A = PhysicalProperty(fa_p, len(types), dev)
    f_B = PhysicalProperty(fb_p, len(types), dev)
    mae_b = float(torch.load(fb_p, map_location="cpu",
                             weights_only=False)["val_mae"])
    return d, types, net, f_A, f_B, 2.0 * mae_b


def one(env, prop, tgt, eta, mult, seed, e_schedule=None, n=256, steps=100):
    d, types, net, f_A, f_B, delta = env
    dev = next(net.parameters()).device
    idx = d["split"]["val"][:n]
    mask = d["mask"][idx].to(dev)
    s = float(f_A.y_std)
    y = torch.full((n,), TARGETS[prop][tgt], device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    c0, f0 = initial_noise(mask, len(types), gen)
    smp = Replay(net, mask, f_net=f_A, y=y, s=s, mode="bdg", w=1.0, clip=1.0,
                 t_min_guide=0.5, bdg_eta=eta, bdg_tau=mult * s,
                 e_schedule=e_schedule)
    log = []
    orig = smp._accumulate_diag

    def hook(diag):
        orig(diag)
        if "bdg_e" in diag:
            log.append((float(diag["bdg_e"].double().mean()),
                        float(diag["bdg_e_measured"].double().mean())))

    smp._accumulate_diag = hook
    C, F = integrate(smp, c0, f0, steps, "euler")[:2]
    r = evaluate_samples(C, F, mask, types, f_A, f_B, y, delta, per_mol=True)
    pm = r.pop("_per_mol")
    fin = pm["finite"]
    return {"in_band": r["in_band_fraction"],
            "in_band_dec": r["in_band_fraction_dec"],
            "mae_over_delta": r["prop_mae_eval"] / delta,
            "bias_over_delta": float((pm["f_B"][fin] - pm["y"][fin]).mean()) / delta,
            "sd_fa": float(pm["f_A"][fin].std(unbiased=True)),
            "sd_fb": float(pm["f_B"][fin].std(unbiased=True)),
            "mol_stab": r["mol_stability"], "valid": r["validity"],
            "e_used": [a for a, _ in log], "e_measured": [b for _, b in log]}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for prop in ("mu", "gap"):
        env = load_env(prop, dev)
        for tgt in ("q50", "q90"):
            ug = one(env, prop, tgt, 0.0, 1.0, SEED_B)   # sd reference, seed B
            for mult in (0.5, 1.5):
                a = one(env, prop, tgt, 4.0, mult, SEED_A)      # schedule source
                closed = one(env, prop, tgt, 4.0, mult, SEED_B)
                replay = one(env, prop, tgt, 4.0, mult, SEED_B,
                             e_schedule=a["e_used"])
                plug = one(env, prop, tgt, 0.0, mult, SEED_B)
                row = {"prop": prop, "target": tgt, "tau_mult": mult,
                       "seed_A": a, "closed": closed, "replay": replay,
                       "plug": plug, "unguided": ug}
                out.append(row)
                print("\n%s %s tau_mult %g   (seed B = %d, schedule from seed %d)"
                      % (prop, tgt, mult, SEED_B, SEED_A))
                print("  %-9s %-8s %-8s %-9s %-9s %-8s %s"
                      % ("run", "in_band", "MAE/d", "bias/d", "sdA/ug",
                         "molstab", "mean e used"))
                for nm in ("plug", "closed", "replay"):
                    r = row[nm]
                    print("  %-9s %-8.4f %-8.3f %+-9.3f %-9.4f %-8.4f %+.4f"
                          % (nm, r["in_band"], r["mae_over_delta"],
                             r["bias_over_delta"], r["sd_fa"] / ug["sd_fa"],
                             r["mol_stab"],
                             sum(r["e_used"]) / max(len(r["e_used"]), 1)))
                dsd = abs(replay["sd_fa"] - closed["sd_fa"]) / closed["sd_fa"]
                dpl = abs(plug["sd_fa"] - closed["sd_fa"]) / closed["sd_fa"]
                print("  |replay-closed|/closed on sd(f_A) = %.4f   "
                      "(plug is %.4f away)   replay reproduces closed: %s"
                      % (dsd, dpl, dsd < 0.25 * max(dpl, 1e-9)))
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
