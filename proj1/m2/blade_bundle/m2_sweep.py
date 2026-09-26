"""Modality 2 sweep: unguided / plug / BDG on the DNA simplex.

THE POINT OF THIS FILE is that the guidance math is IDENTICAL to Modality 1.
Compare `guidance_field` below with the bdg branch of proj1/src/guidance.py: the
state space changed from R^3 molecular coordinates to (Delta^3)^L, the property
changed from a trained EGNN to an exact GC count, and the controller did not
change at all. That is the section 3.6 claim, made literal.

    m_i   = x_t + (1-t) * v_theta          endpoint estimate
    F_i   = f(m_i)                         scalar property, one per sequence
    g_i   = grad f(m_i)
    F_bar = mean_i F_i                     over the BATCH axis
    V_b   = var_i F_i   (unbiased)         the summand that survives to t=1
    e     = (V_b - tau^2) / tau^2          relative error vs the SETPOINT
    num_i = (y - F_i) - eta * e * (F_i - F_bar)
    G_i   = J^T ( num_i / s^2 * g_i )      J = dm/dx_t, one VJP

Two-sided on purpose: e < 0 flips the sign and WIDENS. The one-sided ablation
clamps e at min=0, which is the restriction the btvg arm carries in Modality 1.
eta = 0 must reproduce plug bit-identically.

SPECS MIRRORED FROM MODALITY 1: n=512, batch=512 (ONE batch -- for BDG the batch
IS the estimator), 100 Euler steps, t_min_guide=0.5, clip=1.0 relative to the
velocity norm, w=1.0 for the ladder, eta in {1,4}, tau_mult in {0.5,1.0,1.5},
seeds 20260921 / 20260922. s = the data's GC sd, playing the role f_A.y_std plays
in Modality 1, so tau_mult reads as "fraction of the natural spread".

THE FIDELITY FLOOR. Modality 1 gates on FR3a = 0.9 x unguided mol_stability.
Every argmax decode here is a syntactically valid sequence, so validity is
trivially 1 and useless as a floor. The analogue that carries the same meaning --
"guidance must not destroy the thing we are generating" -- is k-mer fidelity:
a cell passes if its 3-mer JS divergence to real enhancers is no worse than
1/0.9 times the unguided cell's. Decode confidence is reported beside it.

Run: python proj1/m2/m2_sweep.py --arm bdg --variant e4t1.5 --seed 20260921
"""
import argparse
import json
import os
import time

import torch

import simplex_fm as S


# ------------------------------------------------------------ metrics
def kmer_freq(x, k=3):
    tok = x.argmax(-1)
    B, L = tok.shape
    pw = torch.tensor([4 ** i for i in range(k)])
    idx = sum(tok[:, i:L - k + 1 + i] * pw[i] for i in range(k))
    cnt = torch.zeros(4 ** k)
    cnt.scatter_add_(0, idx.reshape(-1), torch.ones(idx.numel()))
    return cnt / cnt.sum()


def js(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * ((a + eps).log() - (b + eps).log())).sum()
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def diversity(x, n=256):
    """Mean pairwise Hamming distance among decoded sequences, subsampled."""
    tok = x.argmax(-1)[:n]
    B = tok.shape[0]
    d, c = 0.0, 0
    for i in range(0, B, 32):
        blk = tok[i:i + 32]
        ne = (blk[:, None, :] != tok[None, :, :]).float().mean(-1)
        d += float(ne.sum()); c += ne.numel()
    return d / max(c, 1)


# ----------------------------------------------------------- guidance
def guidance_field(net, x_t, t, y, s, mode, eta=0.0, tau=None, onesided=False):
    """Returns the guidance field in SCORE units, plus the controller state.

    Identical in structure to proj1/src/guidance.py's shared tail: build the
    endpoint estimate, read the property and its gradient there, form a scalar
    coefficient per sample, and pull back through J = dm/dx_t with one VJP."""
    xt = x_t.detach().requires_grad_(True)
    tv = torch.full((xt.shape[0],), float(t), device=xt.device)
    m = xt + (1.0 - float(t)) * net(xt, tv)          # posterior mean
    fval = S.gc_soft(m)                              # [B]  the property at m
    diag = {}

    num = y - fval                                   # plug / DPS
    if mode == "bdg":
        # BATCH-DISPERSION GUIDANCE, verbatim from Modality 1.
        # V_b does not vanish (unlike the per-sample posterior variance, which
        # goes to 0 as the interpolant closes), so e may change sign and the
        # controller needs no clamp. That is what buys the widening branch.
        B = fval.shape[0]
        fbar = fval.mean()
        V_b = fval.var(unbiased=True) if B > 1 else fval.new_zeros(())
        tau_t = torch.as_tensor(float(tau), device=fval.device, dtype=fval.dtype)
        e = (V_b - tau_t ** 2) / (tau_t ** 2)
        if onesided:
            e = e.clamp(min=0.0)                     # the btvg restriction
        num = num - float(eta) * e * (fval - fbar)
        ed = float(e.detach()); Vd = float(V_b.detach())
        diag = {"bdg_V_b": Vd, "bdg_tau": float(tau_t),
                "bdg_e": ed, "bdg_w_eff": 1.0 + float(eta) * ed,
                "bdg_widening": float(ed < 0)}

    # ONE VJP does the whole pullback. autograd.grad(fval, xt, grad_outputs=c)
    # returns sum_i c_i * d fval_i / d xt, and fval_i depends only on sample i,
    # so this is exactly J^T (c_i * grad f(m_i)) per sample -- the same single
    # backward that _pullback performs in Modality 1. No extra forward, no extra
    # backward: the cost dict matches plug's.
    G = torch.autograd.grad(fval, xt, grad_outputs=(num / s ** 2))[0]
    return G.detach(), diag


@torch.no_grad()
def _step(x, v, dt):
    return S.to_simplex(x + dt * v)


def run_cell(net, ck, arm, variant, w, y, s, tau, eta, onesided,
             n, steps, t_min, clip, seed, delta, real_kmer):
    L = ck["crop"]
    g = torch.Generator().manual_seed(seed)
    x = torch.distributions.Dirichlet(torch.ones(4)).sample((n, L))
    dt = 1.0 / steps
    cost = {"gen_fwd": 0, "gen_vjp": 0, "guide_fwd": 0, "guide_bwd": 0}
    clipped, dlog = 0, {}

    for i in range(steps):
        t = i * dt
        with torch.no_grad():
            v = net(x, torch.full((n,), t))
        cost["gen_fwd"] += 1
        if arm != "unguided" and t >= t_min:
            G, d = guidance_field(net, x, t, y, s, arm, eta, tau, onesided)
            cost["gen_fwd"] += 1; cost["gen_vjp"] += 1
            cost["guide_fwd"] += 1; cost["guide_bwd"] += 1
            # score -> velocity, exactly Modality 1's convention
            fac = (1.0 - t) / max(t, 1e-6)
            corr = w * fac * G
            vn = v.reshape(n, -1).norm(dim=1)
            cn = corr.reshape(n, -1).norm(dim=1)
            over = cn > clip * vn
            clipped += int(over.sum())
            sc = torch.where(over, clip * vn / cn.clamp(min=1e-12),
                             torch.ones_like(cn))
            v = v + corr * sc.view(-1, 1, 1)
            for k, val in d.items():
                dlog[k] = dlog.get(k, 0.0) + val
            dlog["_n"] = dlog.get("_n", 0) + 1
        x = _step(x, v, dt)

    nstep = max(dlog.pop("_n", 1), 1)
    diag = {k: v / nstep for k, v in dlog.items()}
    gc = S.gc_hard(x)
    conf = float(x.max(-1).values.mean())
    kj = js(kmer_freq(x), real_kmer)
    return {
        "arm": arm, "variant": variant, "w": w, "y": y, "s": s,
        "tau": tau, "eta": eta, "onesided": onesided,
        "n": n, "steps": steps, "t_min_guide": t_min, "clip": clip,
        "seed": seed, "delta": delta,
        "gc_mean": float(gc.mean()), "gc_sd": float(gc.std()),
        "bias_delta": float((gc.mean() - y) / delta),
        "in_band_fraction": float((gc - y).abs().le(delta).float().mean()),
        "decode_conf": conf, "kmer_js": kj,
        "diversity": diversity(x),
        "clipped_sample_steps": clipped, "cost": cost, "diag": diag,
    }, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="proj1/m2/ckpt/fm_m2.pt")
    ap.add_argument("--arm", required=True)          # unguided | plug | bdg
    ap.add_argument("--variant", default="-")        # e<eta>t<mult>[o] for bdg
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--target", default="q50", choices=["q50", "q90"])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--t-min", type=float, default=0.5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--delta-ratio", type=float, default=0.16)   # delta/sigma
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out-dir", default="results/m2")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = S.SimplexFM(ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state_dict"]); net.eval()

    # s plays f_A.y_std's role; delta matches Modality 1's delta/sigma band.
    s = ck["gc_std"]
    delta = a.delta_ratio * s
    X = S.one_hot(S.load_seqs(crop=ck["crop"]), ck["crop"])
    gc_real = S.gc_hard(X)
    y = float(gc_real.median()) if a.target == "q50" else float(
        gc_real.quantile(0.90))
    real_kmer = kmer_freq(X)

    eta, tau, onesided = 0.0, None, False
    if a.arm == "bdg":
        import re
        m = re.match(r"^e([0-9.]+)t([0-9.]+)(o?)$", a.variant)
        if not m:
            raise SystemExit("bdg variant must look like e4t1.5 or e4t1.5o")
        eta, onesided = float(m.group(1)), bool(m.group(3))
        tau = float(m.group(2)) * s

    name = "gc__%s__%s__w%g__%s__s%d.json" % (
        a.arm, a.target, a.w, a.variant, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, name)
    if os.path.exists(path):
        print("done already:", name); return 0

    print("cell %s | y=%.5f s=%.5f delta=%.5f tau=%s eta=%g"
          % (name, y, s, delta, tau, eta))
    t0 = time.time()
    r, x = run_cell(net, ck, a.arm, a.variant, a.w, y, s, tau, eta, onesided,
                    a.n, a.steps, a.t_min, a.clip, a.seed, delta, real_kmer)
    r["target_name"] = a.target
    r["minutes"] = (time.time() - t0) / 60
    json.dump(r, open(path, "w"), indent=1)
    torch.save({"gc": S.gc_hard(x), "conf": x.max(-1).values.mean(-1)},
               path.replace(".json", ".permol.pt"))
    print("  in_band %.4f | gc %.4f +/- %.4f | conf %.3f | kmerJS %.5f | div %.4f | %.1f min"
          % (r["in_band_fraction"], r["gc_mean"], r["gc_sd"], r["decode_conf"],
             r["kmer_js"], r["diversity"], r["minutes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
