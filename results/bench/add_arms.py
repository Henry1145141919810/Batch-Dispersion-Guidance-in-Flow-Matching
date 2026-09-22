"""Add the new guidance arms to proj1/src/guidance.py. Run once."""
import io

p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- new helpers
HELPERS = '''

def hutchinson_tr_HS_squared(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
                             n_probe, generator=None, cost=_NOCOST):
    """1/2 tr((H Sigma)^2) by Hutchinson: E_z[ z^T (H Sigma H Sigma) z ] / 2.

    This is the term SMG omits. For the quadratic surrogate
    q(u) = f(m) + g'u + 1/2 u'Hu with u ~ N(0, Sigma),

        E[q]   = f(m) + 1/2 tr(H Sigma)          <- SMG has this
        Var[q] = g' Sigma g + 1/2 tr((H Sigma)^2) <- SMG is missing this

    so a method named "second-moment guidance" was not using the second moment.
    See TOP6_GUIDANCE_RECOMMENDATIONS.md section 2.

    (H Sigma)^2 z is applied as Sigma -> H -> Sigma -> H, i.e. two JVPs and two
    HVPs per probe, then contracted with z. Vanishes identically for affine f,
    which is the G2 null test.
    """
    total = torch.zeros(coords.shape[0], device=coords.device)
    msk = mask.unsqueeze(-1)
    for _ in range(n_probe):
        z_c = (torch.randint(0, 2, coords.shape, generator=generator,
                             device=coords.device, dtype=coords.dtype) * 2.0 - 1.0) * msk
        z_f = (torch.randint(0, 2, feats.shape, generator=generator,
                             device=feats.device, dtype=feats.dtype) * 2.0 - 1.0) * msk
        v_c, v_f = z_c, z_f
        for _stage in range(2):                      # (H Sigma) applied twice
            sv_c, sv_f = sigma_times_vector(post_fn, coords, feats, mask,
                                            v_c, v_f, k, cost)
            hv_c, hv_f = _hvp(f_net, m_c, m_f, mask, sv_c, sv_f, cost)
            if hv_c is None:
                return torch.zeros_like(total)       # affine f: the term is 0
            v_c, v_f = hv_c, hv_f
        total = total + ((z_c * v_c).sum(dim=(1, 2))
                         + (z_f * v_f).sum(dim=(1, 2))).detach()
    return 0.5 * total / n_probe


def _hvp(f_net, m_c, m_f, mask, w_c, w_f, cost=_NOCOST):
    """H w at m, as a (coords, feats) pair. Returns (None, None) for affine f."""
    mc = m_c.detach().requires_grad_(True)
    mf = m_f.detach().requires_grad_(True)
    val = f_net(mc, mf, mask)
    gc, gf = torch.autograd.grad(val.sum(), (mc, mf), create_graph=True)
    cost.guide_fwd += 1
    cost.guide_hvp += 1
    scalar = (gc * w_c).sum() + (gf * w_f).sum()
    if not scalar.requires_grad:
        return None, None                            # H == 0 exactly
    hv_c, hv_f = torch.autograd.grad(scalar, (mc, mf), allow_unused=True)
    hv_c = torch.zeros_like(mc) if hv_c is None else hv_c
    hv_f = torch.zeros_like(mf) if hv_f is None else hv_f
    return hv_c.detach(), hv_f.detach()


def kappa3_skew(f_net, post_fn, coords, feats, mask, m_c, m_f, k, g_c, g_f,
                v_f, cost=_NOCOST):
    """Standardised skew of the observable F = f(X): gamma = k3[g,g,g] / (g'Sg)^{3/2}.

    Because p(X | x_t) is an exponential family in eta = alpha x_t / sigma^2,
    differentiating Tweedie's covariance identity once more gives
    dSigma/dx_t = (alpha/sigma^2) k3, so with Sigma = k * dm/dx_t,

        k3[g,g] = k * grad_{x_t} ( g' Sigma(x_t) g )

    i.e. the gradient of the scalar g'Sigma g, which is itself one JVP plus a
    dot product. One extra reverse pass gives the third cumulant contracted
    with g twice; contracting once more with g and standardising by
    (g'Sigma g)^{3/2} gives the skew.

    This is a DIAGNOSTIC, not a field: it measures whether a two-moment closure
    is trustworthy at this state. Both obvious uses of it failed on synthetic
    problems (as a closure it breaks on multimodal posteriors; as an abstention
    gate it loses to the time index), so it is logged, not acted on.
    See TOP6_GUIDANCE_RECOMMENDATIONS.md section 1.
    """
    c_in = coords.detach().requires_grad_(True)
    f_in = feats.detach().requires_grad_(True)

    def quad(c, f):
        p = post_fn(c, f)
        sg_c, sg_f = torch.func.jvp(
            lambda cc, ff: (post_fn(cc, ff).mean_coords, post_fn(cc, ff).mean_feats),
            (c, f), (g_c, g_f))[1]
        return k * ((g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2)))

    try:
        q = quad(c_in, f_in)
        d_c, d_f = torch.autograd.grad(q.sum(), (c_in, f_in), allow_unused=True)
        cost.gen_fwd += 2
        cost.gen_jvp += 1
        cost.gen_vjp += 1
    except RuntimeError:
        return torch.full((coords.shape[0],), float("nan"), device=coords.device)
    if d_c is None:
        return torch.zeros(coords.shape[0], device=coords.device)
    k3_gg_c = k * d_c.detach()
    k3_gg_f = k * (d_f.detach() if d_f is not None else torch.zeros_like(f_in))
    m3 = (k3_gg_c * g_c).sum(dim=(1, 2)) + (k3_gg_f * g_f).sum(dim=(1, 2))
    return m3 / v_f.clamp(min=1e-12).pow(1.5)


def sigma_mc_weighted_grad(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
                           y, s, n_mc, generator=None, cost=_NOCOST,
                           observable_closure=False):
    """LGD-MC (observable_closure=False) and the observable-space closure (True).

    Both draw n_mc perturbations of m from the model's OWN covariance -- via
    Sigma^{1/2} z approximated by one JVP per draw, which is the same operator
    every other arm uses -- rather than TFG's tuned isotropic sigma. That is
    the one difference from tfg_mc_weighted_grad, and it is the point: no
    hyperparameter.

      LGD-MC:  grad of log mean_i w_i, w_i = exp(-(y - f(X_i))^2 / 2 s^2)
               -- a softmax-weighted average of per-draw plug-in gradients.
      OSC:     fit a Gaussian to the SCALAR F = f(X) across draws and integrate
               the likelihood against it analytically, giving
               (r/S) gbar + ((r^2 - S)/S^2) Cov(F, grad f),
               with r = y - mean(F), S = s^2 + Var(F). Same guide budget,
               lower variance where F is near-Gaussian.
    """
    K = max(int(n_mc), 1)
    msk = mask.unsqueeze(-1)
    fs, gcs, gfs = [], [], []
    for _ in range(K):
        z_c = torch.randn(m_c.shape, generator=generator, device=m_c.device,
                          dtype=m_c.dtype) * msk
        z_f = torch.randn(m_f.shape, generator=generator, device=m_f.device,
                          dtype=m_f.dtype) * msk
        d_c, d_f = sigma_times_vector(post_fn, coords, feats, mask, z_c, z_f, k, cost)
        mc = (m_c + d_c).detach().requires_grad_(True)
        mf = (m_f + d_f).detach().requires_grad_(True)
        val = f_net(mc, mf, mask)
        gc, gf = torch.autograd.grad(val.sum(), (mc, mf))
        cost.guide_fwd += 1
        cost.guide_bwd += 1
        fs.append(val.detach())
        gcs.append(gc.detach())
        gfs.append(gf.detach())
    F = torch.stack(fs)                                  # [K, B]
    GC = torch.stack(gcs)
    GF = torch.stack(gfs)

    if not observable_closure:
        lw = -((y - F) ** 2) / (2.0 * s ** 2)
        w = torch.softmax(lw, dim=0).view(K, -1, 1, 1)
        coef = ((y - F) / s ** 2).view(K, -1, 1, 1)
        return (w * coef * GC).sum(0), (w * coef * GF).sum(0), F

    mu = F.mean(0)
    V = F.var(0, unbiased=True) if K > 1 else torch.zeros_like(mu)
    gbar_c, gbar_f = GC.mean(0), GF.mean(0)
    S = s ** 2 + V
    r = y - mu
    a = (r / S).view(-1, 1, 1)
    out_c, out_f = a * gbar_c, a * gbar_f
    if K > 1:
        dF = (F - mu).view(K, -1, 1, 1)
        cov_c = (dF * (GC - gbar_c)).sum(0) / (K - 1)
        cov_f = (dF * (GF - gbar_f)).sum(0) / (K - 1)
        b = ((r ** 2 - S) / S ** 2).view(-1, 1, 1)
        out_c = out_c + b * cov_c
        out_f = out_f + b * cov_f
    return out_c, out_f, F
'''

anchor = "\n\n# --------------------------------------------------------------------------\n# the fields\n# --------------------------------------------------------------------------"
assert s.count(anchor) == 1
s = s.replace(anchor, HELPERS + anchor)

# ---------------------------------------------------------------- dispatch
OLD_DOC = '''    mode:
      "plug"        (y - f(m)) / s^2                      the baseline
      "smg_mean"    numerator gets -c, denominator s^2     ablation
      "smg_var"     denominator gets +v_f                  ablation
      "smg"         both -- the candidate
      "tfg_mc"      TFG's Monte Carlo smoothing -- the closest published
                    competitor, K samples at isotropic sigma_mc
    """'''
NEW_DOC = '''    mode:
      "plug"        (y - f(m)) / s^2                      the baseline (DPS-style)
      "smg_mean"    numerator gets -c, denominator s^2     ablation
      "smg_var"     denominator gets +v_f                  ablation; this is also
                    TMPD / PiGDM's published scalar form
      "smg"         both -- SMG as shipped
      "smg2"        smg + 1/2 tr((H Sigma)^2) in the denominator: the COMPLETED
                    second moment. Reduces to smg for affine f.
      "smg2_curv"   smg2 plus the first Stein-series direction
                    ((r^2 - S)/S^2) Sigma H Sigma g -- a different direction,
                    not a rescaling
      "tfg_mc"      TFG's Monte Carlo smoothing: K isotropic perturbations at
                    tuned sigma_mc (Ye et al. 2024)
      "lgd_mc"      likelihood marginalisation drawing from the model's own
                    Sigma instead of isotropic -- no tuned spread
      "osc"         observable-space closure: same guide budget as lgd_mc, but
                    the scalar observable's law is integrated analytically

    Every mode also returns diagnostics. Pass want_kappa3=True to log the
    standardised skew of the observable alongside the field; it costs 2-3 extra
    generator passes and changes nothing about the field itself.
    """'''
assert s.count(OLD_DOC) == 1
s = s.replace(OLD_DOC, NEW_DOC)

s = s.replace('''def guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                   mode="plug", n_probe=1, generator=None, cost=_NOCOST,
                   n_mc=4, sigma_mc=0.1):''',
'''def guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                   mode="plug", n_probe=1, generator=None, cost=_NOCOST,
                   n_mc=4, sigma_mc=0.1, want_kappa3=False):''')

# lgd_mc / osc dispatch, inserted before the tfg_mc branch
OLD_TFG = '''    if mode == "tfg_mc":'''
NEW_MC = '''    if mode in ("lgd_mc", "osc"):
        w_c, w_f, F = sigma_mc_weighted_grad(
            f_net, post_fn, coords, feats, mask, m_c, m_f, k, y, s, n_mc,
            generator, cost, observable_closure=(mode == "osc"))
        G_c, G_f = _pullback(post_fn, coords, feats, w_c, w_f, cost)
        return G_c, G_f, {"f": fval, "c": torch.zeros_like(fval),
                          "v_f": torch.zeros_like(fval), "k": k, "n_mc": n_mc,
                          "obs_mean": F.mean(0), "obs_var": F.var(0, unbiased=True)
                          if F.shape[0] > 1 else torch.zeros_like(fval)}

    if mode == "tfg_mc":'''
assert s.count(OLD_TFG) == 1
s = s.replace(OLD_TFG, NEW_MC)

# smg2 / smg2_curv terms
OLD_TERMS = '''    c_term = torch.zeros_like(fval)
    v_f = torch.zeros_like(fval)
    if mode in ("smg_mean", "smg"):
        c_term = hutchinson_tr_H_Sigma(f_net, post_fn, coords, feats, mask,
                                       m_c, m_f, k, n_probe, generator, cost)
    if mode in ("smg_var", "smg"):
        sg_c, sg_f = sigma_times_vector(post_fn, coords, feats, mask,
                                        g_c, g_f, k, cost)
        v_f = (g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2))

    num = y - fval - c_term
    den = s ** 2 + v_f
    scale = (num / den).view(-1, 1, 1)

    G_c, G_f = _pullback(post_fn, coords, feats, scale * g_c, scale * g_f, cost)

    diag = {"f": fval, "c": c_term, "v_f": v_f, "k": k,
            "num": num, "den": den}
    return G_c.detach(), G_f.detach(), diag'''
NEW_TERMS = '''    SMG_MEAN = ("smg_mean", "smg", "smg2", "smg2_curv")
    SMG_VAR = ("smg_var", "smg", "smg2", "smg2_curv")
    SMG_VAR2 = ("smg2", "smg2_curv")

    c_term = torch.zeros_like(fval)
    v_f = torch.zeros_like(fval)
    v2 = torch.zeros_like(fval)
    if mode in SMG_MEAN:
        c_term = hutchinson_tr_H_Sigma(f_net, post_fn, coords, feats, mask,
                                       m_c, m_f, k, n_probe, generator, cost)
    sg_c = sg_f = None
    if mode in SMG_VAR:
        sg_c, sg_f = sigma_times_vector(post_fn, coords, feats, mask,
                                        g_c, g_f, k, cost)
        v_f = (g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2))
    if mode in SMG_VAR2:
        # The term SMG omits. Zero for affine f, so smg2 == smg there.
        v2 = hutchinson_tr_HS_squared(f_net, post_fn, coords, feats, mask,
                                      m_c, m_f, k, n_probe, generator, cost)

    num = y - fval - c_term
    den = s ** 2 + v_f + v2
    scale = (num / den).view(-1, 1, 1)
    w_c, w_f = scale * g_c, scale * g_f

    if mode == "smg2_curv":
        # First Stein-series direction: ((r^2 - S)/S^2) Sigma H Sigma g.
        # Sigma g is already in hand; H (Sigma g) is one HVP, and the outer
        # Sigma is supplied by the shared J^T pull-back, so this costs one HVP.
        hsg_c, hsg_f = _hvp(f_net, m_c, m_f, mask, sg_c, sg_f, cost)
        if hsg_c is not None:
            b = ((num ** 2 - den) / den ** 2).view(-1, 1, 1)
            w_c = w_c + b * hsg_c
            w_f = w_f + b * hsg_f

    G_c, G_f = _pullback(post_fn, coords, feats, w_c, w_f, cost)

    diag = {"f": fval, "c": c_term, "v_f": v_f, "v2": v2, "k": k,
            "num": num, "den": den}
    if want_kappa3:
        if sg_c is None:
            sg_c, sg_f = sigma_times_vector(post_fn, coords, feats, mask,
                                            g_c, g_f, k, cost)
            v_f = (g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2))
        diag["kappa3_skew"] = kappa3_skew(f_net, post_fn, coords, feats, mask,
                                          m_c, m_f, k, g_c, g_f, v_f, cost)
    return G_c.detach(), G_f.detach(), diag'''
assert s.count(OLD_TERMS) == 1
s = s.replace(OLD_TERMS, NEW_TERMS)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance.py: added smg2, smg2_curv, lgd_mc, osc, kappa3 probe")
