"""Measure the real per-molecule cost of each guidance arm on fm_v1.

Answers "how long does the guidance cross product actually take?" with
measurements instead of the 3-6x multiplier I inferred from call counting.

    python results/bench/time_guided.py

Times a short guided integration per arm and reports ms/molecule at NFE 100,
plus the measured generator/guide call counts so the scaling to other NFE and
probe counts is arithmetic rather than another guess.
"""
from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from models.egnn import EGNNVelocity  # noqa: E402
from guidance import Cost, fm_posterior, guidance_field  # noqa: E402
from sampling import initial_noise  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from m1_signed_bias import PhysicalProperty  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 100


def main():
    ck = torch.load(os.path.join(ROOT, "betty_pull", "fm.pt"), map_location="cpu",
                    weights_only=False)
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    ca = ck["args"]
    net = EGNNVelocity(len(types), ca["hidden"], ca["layers"]).to(DEV).eval()
    net.load_state_dict(ck["ema_state_dict"])
    for p in net.parameters():
        p.requires_grad_(False)

    gpath = os.path.join(ROOT, "proj1", "checkpoints", "f_A_mu.pt")
    guide = PhysicalProperty(gpath, len(types), DEV)
    print("guide: f_A_mu (EGNNScalar), physical units")

    B = 64
    sel = d["split"]["val"][:B]
    mask = d["mask"][sel].to(DEV)
    g = torch.Generator(device=DEV).manual_seed(7)
    c0, f0 = initial_noise(mask, len(types), g)

    arms = [
        ("plug",       dict(mode="plug")),
        ("tmpd",       dict(mode="smg_var")),
        ("tfg_mc4",    dict(mode="tfg_mc", n_mc=4)),
        ("lgd_mc4",    dict(mode="lgd_mc", n_mc=4)),
        ("osc4",       dict(mode="osc", n_mc=4)),
        ("smg_p1",     dict(mode="smg", n_probe=1)),
        ("smg_p4",     dict(mode="smg", n_probe=4)),
        ("smg2_p1",    dict(mode="smg2", n_probe=1)),
        ("smg2_p4",    dict(mode="smg2", n_probe=4)),
        ("smg2curv_p1",dict(mode="smg2_curv", n_probe=1)),
        ("smg2_k3",    dict(mode="smg2", n_probe=1, want_kappa3=True)),
    ]
    print("\n%-10s %10s %10s   %s" % ("arm", "ms/mol", "x unguided", "calls per field eval"))
    print("-" * 78)

    # unguided reference: one plain network forward per step
    t = torch.full((B,), 0.5, device=DEV)
    torch.cuda.synchronize() if DEV == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        for _ in range(20):
            net(c0, f0, mask, t)
    torch.cuda.synchronize() if DEV == "cuda" else None
    base_per_eval = (time.time() - t0) / 20
    base_ms = base_per_eval * STEPS * 1000 / B
    print("%-13s %10.2f %10s   1 gen fwd" % ("(unguided)", base_ms, "1.00"))

    for name, kw in arms:
        cost = Cost()
        post = lambda c, f: fm_posterior(net, c, f, mask, t)
        # warm up once (compiles kernels, populates caches)
        guidance_field(guide, post, c0, f0, mask, 2.0, 0.5, generator=g, cost=Cost(), **kw)
        torch.cuda.synchronize() if DEV == "cuda" else None
        t0 = time.time()
        reps = 5
        for _ in range(reps):
            guidance_field(guide, post, c0, f0, mask, 2.0, 0.5, generator=g, cost=cost, **kw)
        torch.cuda.synchronize() if DEV == "cuda" else None
        per_eval = (time.time() - t0) / reps
        ms = per_eval * STEPS * 1000 / B
        cd = {k: v // reps for k, v in cost.as_dict().items()}
        calls = " ".join("%s=%d" % (k.replace("gen_", "g.").replace("guide_", "f."), v)
                         for k, v in cd.items() if v)
        print("%-13s %10.2f %10.2f   %s" % (name, ms, ms / base_ms, calls))

    print("\nNFE %d, batch %d, device %s. ms/mol scales linearly in NFE." % (STEPS, B, DEV))
    print("A cell of n molecules costs n * (ms/mol) / 1000 seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
