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
# v_f(t) measured on the held-out split by make_vf_table.py. See the tmpd block
# for why this replaces Tweedie's k = (1-t)^2/t on the simplex.
_VF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vf_table.json")
try:
    with open(_VF_PATH) as _fh:
        VF_TABLE = json.load(_fh)
except Exception:
    VF_TABLE = None


def _vf_at(prop, t):
    """Linear interpolation into the measured v_f(t) table."""
    grid, vals = VF_TABLE["grid"], VF_TABLE["v_f"][prop]
    if t <= grid[0]:
        return vals[0]
    if t >= grid[-1]:
        return vals[-1]
    for i in range(1, len(grid)):
        if t <= grid[i]:
            u = (t - grid[i - 1]) / (grid[i] - grid[i - 1])
            return vals[i - 1] * (1 - u) + vals[i] * u
    return vals[-1]


_GRAD_CACHE = {}


def _const_grad(prop, L, device, dtype):
    """The constant gradient of an AFFINE observable. Only gc is affine; for a
    non-affine observable this is not used (tmpd takes the general path)."""
    key = (prop, L, str(device), str(dtype))
    if key not in _GRAD_CACHE:
        g = torch.zeros(L, 4, device=device, dtype=dtype)
        if prop == "gc":
            g[:, 1] = 1.0 / L; g[:, 2] = 1.0 / L
        _GRAD_CACHE[key] = g
    return _GRAD_CACHE[key]


def guidance_field(net, x_t, t, y, s, mode, eta=0.0, tau=None, onesided=False,
                   n_mc=8, sigma_mc=0.35, gen=None, cost=None, prop="gc"):
    """Returns the guidance field in SCORE units, plus the controller state.

    Identical in structure to proj1/src/guidance.py's shared tail: build the
    endpoint estimate, read the property and its gradient there, form a scalar
    coefficient per sample, and pull back through J = dm/dx_t with one VJP."""
    xt = x_t.detach().requires_grad_(True)
    tv = torch.full((xt.shape[0],), float(t), device=xt.device)
    m = xt + (1.0 - float(t)) * net(xt, tv)          # posterior mean
    tv_ = tv
    if mode not in ("unguided", "plug", "tmpd", "lgd_mc", "tfg_mc", "bdg"):
        # Previously an unrecognised mode fell through to the plug tail and
        # returned a plug field under the wrong label -- a typo'd arm name would
        # have produced plausible numbers for a method that never ran.
        raise ValueError("unknown guidance mode %r" % (mode,))
    f_soft = S.PROPS[prop][0]
    fval = f_soft(m)                                 # [B]  the property at m
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

    # ---------------------------------------------------------------- tmpd
    # TMPD / PiGDM: the denominator gains the property variance v_f = g^T Sigma g,
    # with Sigma = k J, k = (1-t)^2/t and J = dm/dx_t. Modality 1 computes it with
    # sigma_times_vector (one JVP) at guidance.py:1679. Same here -- but note
    # gc_soft is EXACTLY affine (Hessian identically zero, gradient constant at
    # 1/L on channels C and G), so v_f is exact rather than a probe estimate, and
    # the curvature terms that separate smg_mean from plug and smg2 from smg in
    # Modality 1 are identically zero here. That is a property of the observable,
    # not a shortcut: see the note in the write-up.
    den = s ** 2
    if mode == "tmpd":
        # v_f = g^T Sigma g. Because f is affine its gradient g is a CONSTANT, so
        # the single VJP we already take gives A = J^T g and v_f = k <g, A> -- no
        # probe, no JVP, no finite difference, and exact to machine precision.
        #
        # k, however, is NOT (1-t)^2/t here. That relation is Tweedie's for a
        # Gaussian VP diffusion with unit per-coordinate noise; our source is
        # Dirichlet(1) with per-coordinate variance 3/80, and the transplanted
        # constant overstates Var(f(x1)|x_t) by 30x at t=0.1 and 2.0e4x at t=0.7
        # (see calib_k.py). We therefore read v_f from a table MEASURED on the
        # held-out split, which for affine f is exactly E[(f(x1) - f(m))^2].
        A = torch.autograd.grad(fval.sum(), xt, retain_graph=True)[0]
        gv = _const_grad(prop, xt.shape[-2], xt.device, xt.dtype)
        gAg = (gv.unsqueeze(0) * A).sum(dim=(1, 2))
        if VF_TABLE is not None:
            v_f = torch.full_like(fval, _vf_at(prop, float(t)))
            k_t = float(v_f.mean()) / max(float(gAg.mean()), 1e-30)
        else:                                    # fall back to M1's constant
            k_t = (1.0 - float(t)) ** 2 / max(float(t), 1e-6)
            v_f = k_t * gAg
        den = den + v_f.clamp(min=0.0)
        diag = {"tmpd_v_f": float(v_f.mean()), "tmpd_k_eff": k_t,
                "tmpd_gAg": float(gAg.mean()),
                "tmpd_vf_over_s2": float((v_f / s ** 2).mean()),
                "tmpd_vf_clamped": float((v_f < 0).float().mean())}

    # ------------------------------------------------------- lgd_mc / tfg_mc
    # Both marginalise the likelihood over draws around m and take the gradient
    # of log mean_i exp(-(y - f(X_i))^2 / 2 s^2) -- a softmax-weighted average of
    # per-draw plug-in gradients (guidance.py:540, sigma_mc_weighted_grad).
    #
    # The draws are ISOTROPIC in both arms. Modality 1 is explicit that drawing
    # `Sigma z` would give covariance Sigma^2, not Sigma, and that a true
    # N(0, Sigma) draw needs Sigma^{1/2}, which one JVP cannot give. It therefore
    # matches Sigma in TRACE: r^2 = tr(Sigma)/d. We do the same, so lgd_mc's
    # spread is read off the model while tfg_mc's is a TUNED hyperparameter --
    # that is the only difference between the two arms, and it is the difference
    # the comparison is about.
    if mode in ("lgd_mc", "tfg_mc"):
        B = fval.shape[0]
        if mode == "lgd_mc":
            # LGD marginalises over the model's own posterior. The draws are
            # ISOTROPIC (Modality 1 is explicit that a true N(0, Sigma) draw
            # needs Sigma^{1/2}, which one JVP cannot give), so the only free
            # quantity is their SCALE. Modality 1 sets it by trace-matching
            # Sigma = k J; here k is the transplanted Gaussian-diffusion constant
            # that overstates the conditional variance by up to 2.0e4x, and using
            # it gave r = 8.3 on an observable confined to [0, 1] -- the draws
            # scrambled the property completely (obs_sd 0.51).
            #
            # Instead match the induced OBSERVABLE spread to the measured
            # conditional variance: isotropic draws of scale r induce
            # Var(f) = r^2 |g|^2, so r^2 = v_f / |g|^2. This is what LGD is
            # trying to achieve, computed from data instead of from a constant.
            gm = torch.autograd.grad(fval.sum(), m, retain_graph=True)[0]
            g2 = (gm ** 2).sum(dim=(1, 2)).clamp(min=1e-30)
            v_meas = (_vf_at(prop, float(t)) if VF_TABLE is not None
                      else (1.0 - float(t)) ** 2 / max(float(t), 1e-6) * 0.004)
            r = (torch.as_tensor(v_meas, device=m.device, dtype=m.dtype)
                 / g2).clamp(min=0.0).sqrt().view(-1, 1, 1)
        else:
            r = torch.full((B, 1, 1), float(sigma_mc), device=m.device)

        K = max(int(n_mc), 1)
        md = m.detach()
        F, Gm = [], []
        for _ in range(K):
            z = torch.randn(md.shape, generator=gen).to(md.device)
            mi = (md + r * z).requires_grad_(True)
            fi = f_soft(mi)
            gi = torch.autograd.grad(fi.sum(), mi)[0]
            F.append(fi.detach()); Gm.append(gi)
            if cost is not None:
                cost["guide_fwd"] += 1; cost["guide_bwd"] += 1
        F = torch.stack(F)                              # [K, B]
        lw = -(y - F) ** 2 / (2.0 * s ** 2)             # log weights
        sw = torch.softmax(lw, dim=0)                   # [K, B]
        coef = (y - F) / s ** 2                         # [K, B]
        # m-space weighted gradient, then ONE pullback -- Modality 1's structure
        W = sum(sw[i].view(-1, 1, 1) * coef[i].view(-1, 1, 1) * Gm[i]
                for i in range(K))
        G = torch.autograd.grad(m, xt, grad_outputs=W)[0]
        diag = {"mc_n": float(K),
                "mc_r": float(r.mean()),
                "mc_obs_sd": float(F.std(0).mean()) if K > 1 else 0.0,
                "mc_logp_spread": float((lw.max(0).values
                                         - lw.min(0).values).mean())}
        return G.detach(), diag

    # ONE VJP does the whole pullback. autograd.grad(fval, xt, grad_outputs=c)
    # returns sum_i c_i * d fval_i / d xt, and fval_i depends only on sample i,
    # so this is exactly J^T (c_i * grad f(m_i)) per sample -- the same single
    # backward that _pullback performs in Modality 1. No extra forward, no extra
    # backward: the cost dict matches plug's.
    G = torch.autograd.grad(fval, xt, grad_outputs=(num / den))[0]
    return G.detach(), diag


@torch.no_grad()
def _step(x, v, dt):
    return S.to_simplex(x + dt * v)


def run_cell(net, ck, arm, variant, w, y, s, tau, eta, onesided,
             n, steps, t_min, clip, seed, delta, real_kmer,
             n_mc=8, sigma_mc=0.35, prop="gc", dev="cpu"):
    L = ck["crop"]
    g = torch.Generator().manual_seed(seed)
    # Dirichlet(1,...,1) is uniform on the simplex, and if E_i ~ Exp(1) i.i.d.
    # then E / sum(E) ~ Dirichlet(1). We draw it that way because
    # torch.distributions.Dirichlet.sample() accepts NO generator and silently
    # uses the GLOBAL RNG -- so `seed` did not control the initial state at all,
    # and every cell started from different noise. Arm-to-arm differences were
    # then dominated by the start, not the guidance.
    # Drawn on CPU with the seeded generator, then moved: a cell is then
    # bit-reproducible whether it runs on cpu or cuda, which matters because the
    # sweep runs off-cluster on a GPU while the checks were done here.
    _e = torch.empty(n, L, 4).exponential_(generator=g)
    x = (_e / _e.sum(-1, keepdim=True)).to(dev)
    dt = 1.0 / steps
    cost = {"gen_fwd": 0, "gen_vjp": 0, "guide_fwd": 0, "guide_bwd": 0}
    clipped, dlog = 0, {}

    for i in range(steps):
        t = i * dt
        with torch.no_grad():
            v = net(x, torch.full((n,), t, device=dev))
        cost["gen_fwd"] += 1
        if arm != "unguided" and t >= t_min:
            G, d = guidance_field(net, x, t, y, s, arm, eta, tau, onesided,
                                  n_mc=n_mc, sigma_mc=sigma_mc, gen=g,
                                  cost=cost, prop=prop)
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
    gc = S.PROPS[prop][1](x)
    conf = float(x.max(-1).values.mean())
    kj = js(kmer_freq(x), real_kmer)
    return {
        "arm": arm, "prop": prop, "variant": variant, "w": w, "y": y, "s": s,
        "tau": tau, "eta": eta, "onesided": onesided,
        "n": n, "steps": steps, "t_min_guide": t_min, "clip": clip,
        "seed": seed, "delta": delta,
        "n_mc": n_mc, "sigma_mc": sigma_mc,
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
    ap.add_argument("--prop", default="gc", choices=["gc", "cpg"])
    # gc  is AFFINE (Hessian identically zero), so the Tweedie second-moment
    #     corrections vanish and smg_mean==plug, smg2==smg exactly.
    # cpg is QUADRATIC, so they do not. The contrast between the two is the
    #     point: it shows WHEN the corrections matter.
    ap.add_argument("--arm", required=True, choices=[
        "unguided", "plug", "tmpd", "lgd_mc", "tfg_mc", "bdg"])
    # plug = DPS (Chung et al., ICLR 2023)
    # tmpd = TMPD (Boys et al. 2023) / PiGDM (Song et al., ICLR 2023)
    # lgd_mc = LGD (Song et al., ICML 2023)
    # tfg_mc = TFG's MC-smoothing ingredient (Ye et al., NeurIPS 2024),
    #          NOT full TFG -- the write-up must preserve that distinction
    # bdg  = ours
    ap.add_argument("--variant", default="-")        # e<eta>t<mult>[o] for bdg
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--target", default="q50", choices=["q50", "q90"])
    ap.add_argument("--n", type=int, default=1024)
    # in_band is a PROPORTION, so its se is sqrt(p(1-p)/n): 0.042 at n=128,
    # 0.021 at n=512, 0.015 at n=1024. Arms separate by ~0.03 in in_band, which
    # n=512 CANNOT resolve (needs n>=2022 for a single cell). We use n=1024 x 3
    # seeds and report the mean with a seed-to-seed se, which also captures seed
    # variance rather than assuming samples within a cell are the only noise.
    ap.add_argument("--steps", type=int, default=400)   # NFE; see the
    # NFE study -- gc_soft (what BDG controls) and gc_hard (what we score)
    # differ by 6% at NFE 100 and 3% at 400, so setpoints land off intent
    # at the old default of 100.
    ap.add_argument("--n-mc", type=int, default=8)
    ap.add_argument("--sigma-mc", type=float, default=0.35)
    # 0.02 made tfg_mc a NULL ARM: it induces nu = sigma^2|g|^2 = 0.0005 s^2, so
    # the softmax weights are uniform and the field collapses onto plug (measured
    # difference exactly 0.000000). 0.35 anchors it on the observable's own scale.
    ap.add_argument("--t-min", type=float, default=0.0)
    # GUIDE FROM THE START. t_min=0.5 was inherited from Modality 1, where the
    # molecule commits late. Here the property is settled by t~0.5: the batch sd
    # of gc_soft(m) rises 0.00125 -> 0.0517 over t in [0, 0.49] and is flat after
    # (0.0497 at t=0.99). Guiding only for t >= 0.5 therefore steers after the
    # decision is made, and closes at most 17% of the gap to target even with the
    # clip disabled. At t_min=0 the same w closes 75%.
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--delta-ratio", type=float, default=0.16)   # delta/sigma
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--device", default="auto")   # auto | cpu | cuda
    ap.add_argument("--out-dir", default="results/m2")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") \
        if a.device == "auto" else a.device
    net = S.SimplexFM(ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state_dict"]); net.eval()
    net = net.to(dev)
    print("device: %s" % dev)

    # s plays f_A.y_std's role; delta matches Modality 1's delta/sigma band.
    # Per-property scale. gc_std is recorded in the checkpoint; anything else is
    # measured from the same corpus the model trained on.
    pass  # s and delta are set below, once the data is loaded
    # The target and the k-mer reference MUST come from the same corpus the base
    # model was trained on. This previously read the 6,126-sequence KC fasta while
    # the model was trained on DeepFlyBrain (GC 0.4552/0.0552 vs KC 0.4757/0.0422),
    # so the target sat ~0.5 sd off and kmer_js was scored against the wrong
    # reference distribution.
    if ck.get("crop") == 500 and "dfb" not in getattr(a, "data", ""):
        X, _Xv, _Xt = S.load_dfb(crop=ck["crop"])
    else:
        X = S.one_hot(S.load_seqs(crop=ck["crop"]), ck["crop"])
    gc_real = S.PROPS[a.prop][1](X)
    s = float(gc_real.std())
    # delta is a CHOICE here -- unlike Modality 1, where delta = 2 x f_B's
    # validation MAE, there is no predictor and hence no predictor error to read
    # it off. The observable is DISCRETE (quantum 1/(L-1) = 0.002), so we set
    # delta per property to span a comparable ~9 attainable values, otherwise
    # the metric is quantisation-limited: at delta/sigma = 0.16, gc spans 8.8
    # values but cpg spans only 2.4. Modality 1's own delta/sigma likewise varies
    # by property (0.059 / 0.109 / 0.160), so a fixed ratio was never the rule.
    quantum = 1.0 / (ck["crop"] - 1)
    delta = max(a.delta_ratio * s, 4.4 * quantum)
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

    name = "%s__%s__%s__w%g__%s__s%d.json" % (
        a.prop, a.arm, a.target, a.w, a.variant, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, name)
    if os.path.exists(path):
        print("done already:", name); return 0

    print("cell %s | y=%.5f s=%.5f delta=%.5f tau=%s eta=%g"
          % (name, y, s, delta, tau, eta))
    t0 = time.time()
    r, x = run_cell(net, ck, a.arm, a.variant, a.w, y, s, tau, eta, onesided,
                    a.n, a.steps, a.t_min, a.clip, a.seed, delta, real_kmer,
                    n_mc=a.n_mc, sigma_mc=a.sigma_mc, prop=a.prop, dev=dev)
    r["target_name"] = a.target
    r["minutes"] = (time.time() - t0) / 60
    json.dump(r, open(path, "w"), indent=1)
    torch.save({"prop": S.PROPS[a.prop][1](x),
                "conf": x.max(-1).values.mean(-1)},
               path.replace(".json", ".permol.pt"))
    print("  in_band %.4f | gc %.4f +/- %.4f | conf %.3f | kmerJS %.5f | div %.4f | %.1f min"
          % (r["in_band_fraction"], r["gc_mean"], r["gc_sd"], r["decode_conf"],
             r["kmer_js"], r["diversity"], r["minutes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
