"""Guidance fields: plug-in baseline and SMG, for both model families.

Notation follows SMG_PRIOR_WORK_AUDIT.md section 2 throughout:

    m     = E[x_1 | x_t]                    posterior mean (the endpoint proxy)
    J     = dm / dx_t                       denoiser Jacobian
    Sigma = Cov(x_1 | x_t)                  posterior covariance
    g     = grad f(m)                       property gradient at the mean
    H     = hess f(m)                       property Hessian at the mean
    c     = 1/2 tr(H Sigma)                 the nonlinear mean correction
    v_f   = g^T Sigma g                     property variance

    G_plug = (y - f(m)) / s^2                 * J^T g
    G_smg  = (y - f(m) - c) / (s^2 + v_f)     * J^T g

The ONLY term that is a candidate for novelty is c. It vanishes whenever f is
affine (H = 0), at which point G_smg reduces exactly to TMPD's published scalar
linear-observation formula. Everything else here -- the covariance, the
uncertainty denominator, the Jacobian identity, plug-in guidance -- is prior
work. See SMG_PRIOR_WORK_AUDIT.md sections 2.2 to 2.5.

Cost accounting, which the M-2 comparison depends on:

  * J^T (g + correction) is ONE VJP through the generator, i.e. one backward.
  * c and v_f both need Sigma acting on a vector. With the Tweedie identity
    Sigma = k * dm/dx_t, that is one JVP per probe vector. We use Hutchinson
    probes for tr(H Sigma) and a single exact JVP for Sigma g.
  * c and v_f are treated as CONSTANTS in the local derivative (stop-gradient),
    but are recomputed at every field evaluation, including both Heun stages.

Every routine takes and returns the (coords, feats) pair as a flat tuple so the
same code serves flow matching and diffusion; the family only changes how
(m, J, Sigma) are formed, which is the job of the Posterior classes.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------
# posterior mean / Jacobian / covariance, per model family
# --------------------------------------------------------------------------

@dataclass
class Posterior:
    """m, and the scalar k with Sigma = k * dm/dx_t.

    Flow matching, x_t = t x_1 + (1 - t) eps:
        m = x_t + (1 - t) v_theta,        k = (1 - t)^2 / t
    VP diffusion, x_tau = alpha x_0 + sigma eps:
        m = (x_tau - sigma eps_theta) / alpha,   k = sigma^2 / alpha

    Both are the standard second-order Tweedie relation for an independent
    Gaussian channel; they are NOT valid under an optimal-transport coupling.
    """

    mean_coords: torch.Tensor
    mean_feats: torch.Tensor
    k: torch.Tensor          # [B], per sample -- NOT a scalar (see fix note below)


def fm_posterior(net, coords, feats, mask, t):
    """Flow matching: m = x + (1-t) v, k = (1-t)^2 / t."""
    v_c, v_f = net(coords, feats, mask, t)
    tb = t.view(-1, 1, 1)
    m_c = coords + (1.0 - tb) * v_c
    m_f = feats + (1.0 - tb) * v_f
    tv = t.reshape(-1).clamp(min=1e-6)
    k = (1.0 - tv) ** 2 / tv                     # [B]
    return Posterior(m_c, m_f, k)


def vp_posterior(net, coords, feats, mask, tau, alpha, sigma):
    """VP diffusion: m = (x - sigma eps) / alpha, k = sigma^2 / alpha."""
    e_c, e_f = net(coords, feats, mask, tau)
    a = alpha.view(-1, 1, 1)
    s = sigma.view(-1, 1, 1)
    m_c = (coords - s * e_c) / a
    m_f = (feats - s * e_f) / a
    av = alpha.reshape(-1).clamp(min=1e-6)
    sv = sigma.reshape(-1)
    k = sv ** 2 / av                             # [B]
    return Posterior(m_c, m_f, k)


# --------------------------------------------------------------------------
# the correction terms
# --------------------------------------------------------------------------

def _flat(*ts):
    return torch.cat([t.reshape(t.shape[0], -1) for t in ts], dim=1)


class Cost:
    """Every pass through the generator and the guide, so an arm's cost is
    measured rather than inferred from step counts. field_evals alone
    under-counts a guided step by roughly 3 forwards and a VJP."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.gen_fwd = self.gen_vjp = self.gen_jvp = 0
        self.guide_fwd = self.guide_bwd = self.guide_hvp = 0

    def as_dict(self):
        return dict(vars(self))

    def __repr__(self):
        return ("gen fwd=%d vjp=%d jvp=%d | guide fwd=%d bwd=%d hvp=%d"
                % (self.gen_fwd, self.gen_vjp, self.gen_jvp,
                   self.guide_fwd, self.guide_bwd, self.guide_hvp))


_NOCOST = Cost()


def property_derivs(f_net, m_c, m_f, mask, cost=_NOCOST):
    """f(m), and grad f(m) as a (coords, feats) pair. One backward."""
    m_c = m_c.detach().requires_grad_(True)
    m_f = m_f.detach().requires_grad_(True)
    val = f_net(m_c, m_f, mask)
    g_c, g_f = torch.autograd.grad(val.sum(), (m_c, m_f), create_graph=False)
    cost.guide_fwd += 1
    cost.guide_bwd += 1
    return val.detach(), g_c.detach(), g_f.detach()


def sigma_times_vector(post_fn, coords, feats, mask, vec_c, vec_f, k, cost=_NOCOST):
    """Sigma v = k * (dm/dx_t) v -- one JVP (forward mode), no graph retained."""
    def mfun(c, f):
        p = post_fn(c, f)
        return p.mean_coords, p.mean_feats

    _, (jv_c, jv_f) = torch.func.jvp(mfun, (coords, feats), (vec_c, vec_f))
    cost.gen_jvp += 1
    kb = torch.as_tensor(k, device=jv_c.device, dtype=jv_c.dtype).reshape(-1, 1, 1)
    return kb * jv_c.detach(), kb * jv_f.detach()


def hutchinson_tr_H_Sigma(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
                          n_probe, generator=None, cost=_NOCOST):
    """1/2 tr(H Sigma) by Hutchinson: E_z[ z^T H (Sigma z) ] with z Rademacher.

    H acts via one HVP at m; Sigma acts via one JVP through the generator. So
    each probe costs one JVP plus one HVP. Probes are drawn from `generator` so
    a trajectory can hold its probe set fixed, which the deterministic-solver
    comparison in M-4 requires.
    """
    total = torch.zeros(coords.shape[0], device=coords.device)
    msk = mask.unsqueeze(-1)
    for _ in range(n_probe):
        z_c = torch.randint(0, 2, coords.shape, generator=generator,
                            device=coords.device, dtype=coords.dtype)
        z_f = torch.randint(0, 2, feats.shape, generator=generator,
                            device=feats.device, dtype=feats.dtype)
        z_c = (z_c * 2.0 - 1.0) * msk
        z_f = (z_f * 2.0 - 1.0) * msk

        sz_c, sz_f = sigma_times_vector(post_fn, coords, feats, mask, z_c, z_f, k, cost)

        mc = m_c.detach().requires_grad_(True)
        mf = m_f.detach().requires_grad_(True)
        val = f_net(mc, mf, mask)
        gc, gf = torch.autograd.grad(val.sum(), (mc, mf), create_graph=True)
        cost.guide_fwd += 1
        cost.guide_hvp += 1
        # directional derivative of grad f along (sz) = H (sz).
        # An AFFINE property has a constant gradient: the scalar below then
        # carries no graph at all and differentiating it raises, so short
        # circuit to H = 0, which is the correct answer. allow_unused covers
        # the softer case where a graph exists but does not reach the inputs.
        scalar = (gc * sz_c).sum() + (gf * sz_f).sum()
        if not scalar.requires_grad:
            continue                       # H == 0, contributes nothing
        hv_c, hv_f = torch.autograd.grad(scalar, (mc, mf), allow_unused=True)
        if hv_c is None:
            hv_c = torch.zeros_like(mc)
        if hv_f is None:
            hv_f = torch.zeros_like(mf)
        total = total + ((z_c * hv_c).sum(dim=(1, 2))
                         + (z_f * hv_f).sum(dim=(1, 2))).detach()
    return 0.5 * total / n_probe


def _pullback(post_fn, coords, feats, w_c, w_f, cost=_NOCOST):
    """J^T (w_c, w_f) -- one VJP through the generator, shared by every arm.

    The weights are stop-gradient constants, which is why they can be folded in
    before the transpose rather than after.
    """
    c_in = coords.detach().requires_grad_(True)
    f_in = feats.detach().requires_grad_(True)
    p = post_fn(c_in, f_in)
    out = (p.mean_coords * w_c).sum() + (p.mean_feats * w_f).sum()
    G_c, G_f = torch.autograd.grad(out, (c_in, f_in))
    cost.gen_fwd += 1
    cost.gen_vjp += 1
    return G_c.detach(), G_f.detach()


def tfg_mc_weighted_grad(f_net, m_c, m_f, mask, y, s, n_mc, sigma,
                         generator=None, cost=_NOCOST):
    """TFG's Monte Carlo smoothing (methods/tfg.py), as a competitor to SMG.

    TFG perturbs the denoised estimate with isotropic Gaussian noise, evaluates
    the likelihood at each sample and averages in log space:

        L = logsumexp_k [ -(y - f(m + eps_k))^2 / 2 s^2 ] - log K,
        eps_k ~ N(0, sigma^2 I)

    which estimates log E_eps[ p(y | m + eps) ]. Differentiating,

        dL/dm = sum_k w_k * (y - f(m + eps_k))/s^2 * grad f(m + eps_k),
        w_k = softmax over k of the per-sample log-likelihood,

    so it returns a softmax-WEIGHTED average of per-sample plug-in gradients,
    rather than one gradient at the mean. Pulling it back through J is identical
    to every other arm.

    This attacks the same defect SMG does -- the property of the mean is not the
    mean of the property -- so it is the closest competitor we have, much closer
    than plain plug-in. What still differs:

      * the spread here is ISOTROPIC with sigma a tuned hyperparameter on a
        schedule; SMG uses the model's own posterior covariance
        Sigma_t = (sigma_t^2/alpha_t) d xhat_1/d x_t, which is anisotropic and
        read off the denoiser rather than tuned;
      * this is K-sample Monte Carlo (K = 4 here, our choice: TFG's QM9 runs
        use eps_bsz = 1, and its molecule code crashes for > 1, issue #10);
        SMG is deterministic at one JVP;
      * this averages the LIKELIHOOD; SMG corrects the property MOMENTS and then
        forms the likelihood.

    Cost: K guide forwards + K guide backwards, then the shared VJP.
    """
    K = max(int(n_mc), 1)
    msk = mask.unsqueeze(-1)
    lps, gcs, gfs = [], [], []
    for _ in range(K):
        e_c = torch.randn(m_c.shape, generator=generator, device=m_c.device,
                          dtype=m_c.dtype) * sigma * msk
        e_f = torch.randn(m_f.shape, generator=generator, device=m_f.device,
                          dtype=m_f.dtype) * sigma * msk
        mc = (m_c + e_c).detach().requires_grad_(True)
        mf = (m_f + e_f).detach().requires_grad_(True)
        val = f_net(mc, mf, mask)
        lp = -((y - val) ** 2) / (2.0 * s ** 2)
        gc, gf = torch.autograd.grad(lp.sum(), (mc, mf))
        cost.guide_fwd += 1
        cost.guide_bwd += 1
        lps.append(lp.detach())
        gcs.append(gc.detach())
        gfs.append(gf.detach())
    LP = torch.stack(lps)                       # [K, B]
    w = torch.softmax(LP, dim=0)                # [K, B]
    W = w.view(K, -1, 1, 1)
    G_c = (W * torch.stack(gcs)).sum(0)
    G_f = (W * torch.stack(gfs)).sum(0)
    return G_c, G_f, LP


# --------------------------------------------------------------------------
# TFG, the full per-step update (Ye et al., NeurIPS 2024), N_recur = 1
# --------------------------------------------------------------------------
#
# NOT `tfg_mc`. That arm is one of TFG's three ingredients -- the MC
# smoothing -- at settings TFG itself does not use on QM9 (K=4, a fixed
# sigma). TFG's own QM9 search (App. E.3, Table 11) turned the smoothing
# almost OFF (gamma_bar 1e-4 .. 0.1) and the clean-space "mean guidance" ON
# on all six properties. These two functions are the whole of
# methods/tfg.py:96-129 at N_recur = 1; the sampler (sampling.py, the
# `tfg` branch) supplies the time schedules and the step geometry, because
# only it knows the grid.

def tfg_rescale_grad(g_c, g_f, mask, clip_scale):
    """TFG's gradient safeguard, tasks/utils.py:rescale_grad, in effect.

    ms = per-sample mean of g^2 over active atoms and all 3+T channels; the
    gradient is multiplied by min(ms, clip)/ms. It is ms, NOT sqrt(ms): above
    the threshold the gradient's RMS becomes clip/RMS, so a larger raw
    gradient comes out SMALLER. That is TFG's code, reproduced rather than
    corrected. One deliberate difference: ms == 0 returns the (zero) gradient
    unchanged, where TFG's 0/0 would give NaN.

    Returns (g_c, g_f, hit) with hit[B] = 1.0 where the clamp bound.
    """
    n_act = mask.sum(dim=1).clamp(min=1.0)
    ch = g_c.shape[-1] + g_f.shape[-1]
    ms = ((g_c ** 2).sum(dim=(1, 2)) + (g_f ** 2).sum(dim=(1, 2))) / (ch * n_act)
    hit = ms > clip_scale
    coef = torch.where(hit, clip_scale / ms.clamp(min=1e-300),
                       torch.ones_like(ms)).view(-1, 1, 1)
    return g_c * coef, g_f * coef, hit.to(g_c.dtype)


def tfg_logf(f_net, c, f, mask, y, mad):
    """TFG's molecule log-likelihood, -((f - y) / mad)^2.

    energy.py:369 squares the difference of a MAD-normalised prediction and a
    MAD-normalised target, so in physical units it is this. It is NOT the
    project's -(y - f)^2 / (2 s^2): the two differ by the constant
    2 s^2 / mad^2, which a strength sweep would absorb -- except that
    rescale_grad's threshold is absolute, so TFG's published (rho, mu, gamma)
    only mean what they meant in TFG's own energy.
    """
    return -((f_net(c, f, mask) - y) / mad) ** 2


def tfg_components(f_net, post_fn, coords, feats, mask, y, mad, std, mu_step,
                   n_iter=4, eps_bsz=1, want_var=True, var_scale=1.0,
                   clip_scale=100.0, generator=None, cost=_NOCOST):
    """One TFG step's two guidance pieces (methods/tfg.py:96-123, N_recur = 1).

      eps_k   K draws of std * N(0, I) -- zero-CoM coordinates, masked
              features -- drawn ONCE and shared by both pieces (tfg.py:99).
              std == 0 draws nothing and uses a single zero perturbation,
              exactly as TFG's get_noise does.
      G       rescale_grad(var_scale * grad_{x_t} L(m(x_t))),
              L(x0) = log mean_k exp(logf(x0 + eps_k)),
              i.e. TFG's Delta_t / rho, through the denoiser.
      D0      x0' <- m; repeat n_iter: x0' += mu_step * rescale_grad(
              grad_{x0'} L(x0')); D0 = x0' - m. TFG's Delta_0, clean space
              only -- no generator backprop.

    `var_scale` puts the gradient in the space TFG's rescale_grad saw: TFG
    differentiates w.r.t. its VP-normalised state, so on a flow path
    x_vp = x_t / c_t the gradient is c_t times ours, and the clip must see
    that (1 on a VP model). G is returned WITHOUT rho: the sampler applies rho
    and the step geometry. mu_step IS applied here, because D0 is nonlinear in
    it -- the iterations compound.

    Cost: want_var -> 1 generator forward with graph + 1 VJP; otherwise one
    no-grad forward for m. Guide: K forwards/backwards for G (if want_var)
    plus n_iter * K for D0 (none when mu_step == 0 -- TFG skips the loop).
    """
    import math

    from models.egnn import zero_com

    msk = mask.unsqueeze(-1)
    if std is None or float(std) == 0.0:
        eps = [(torch.zeros_like(coords), torch.zeros_like(feats))]
    else:
        eps = []
        for _ in range(max(int(eps_bsz), 1)):
            e_c = torch.randn(coords.shape, generator=generator,
                              device=coords.device, dtype=coords.dtype)
            e_f = torch.randn(feats.shape, generator=generator,
                              device=feats.device, dtype=feats.dtype)
            eps.append((zero_com(e_c * msk, mask) * float(std),
                        e_f * msk * float(std)))
    K = len(eps)

    def L(c, f):
        lp = torch.stack([tfg_logf(f_net, c + e_c, f + e_f, mask, y, mad)
                          for (e_c, e_f) in eps])                # [K, B]
        cost.guide_fwd += K
        return torch.logsumexp(lp, dim=0) - math.log(K)

    zeros_B = torch.zeros(coords.shape[0], device=coords.device, dtype=coords.dtype)
    G_c = torch.zeros_like(coords)
    G_f = torch.zeros_like(feats)
    hit_var = zeros_B
    if want_var:
        c_in = coords.detach().requires_grad_(True)
        f_in = feats.detach().requires_grad_(True)
        p = post_fn(c_in, f_in)
        g_c, g_f = torch.autograd.grad(L(p.mean_coords, p.mean_feats).sum(),
                                       (c_in, f_in))
        cost.gen_fwd += 1
        cost.gen_vjp += 1
        cost.guide_bwd += K
        m_c, m_f = p.mean_coords.detach(), p.mean_feats.detach()
        G_c, G_f, hit_var = tfg_rescale_grad(var_scale * g_c.detach() * msk,
                                             var_scale * g_f.detach() * msk,
                                             mask, clip_scale)
    else:
        with torch.no_grad():
            p = post_fn(coords, feats)
        cost.gen_fwd += 1
        m_c, m_f = p.mean_coords.detach(), p.mean_feats.detach()

    x_c, x_f = m_c.clone(), m_f.clone()
    hits0 = []
    if float(mu_step) != 0.0:
        for _ in range(int(n_iter)):
            xc = x_c.detach().requires_grad_(True)
            xf = x_f.detach().requires_grad_(True)
            gc, gf = torch.autograd.grad(L(xc, xf).sum(), (xc, xf))
            cost.guide_bwd += K
            gc, gf, h0 = tfg_rescale_grad(gc.detach() * msk, gf.detach() * msk,
                                          mask, clip_scale)
            hits0.append(h0)
            x_c = x_c + float(mu_step) * gc
            x_f = x_f + float(mu_step) * gf
    D_c, D_f = (x_c - m_c).detach(), (x_f - m_f).detach()
    diag = {"tfg_var_rescaled": hit_var,
            "tfg_d0_rescaled": (torch.stack(hits0).mean(0) if hits0 else zeros_B),
            "tfg_eps_bsz": K}
    return G_c.detach(), G_f.detach(), D_c, D_f, diag


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

    def _mean(cc, ff):
        p = post_fn(cc, ff)
        return p.mean_coords, p.mean_feats

    def quad(c, f):
        # ONE jvp, which internally does one forward. An earlier version also
        # called post_fn on its own line and again inside the lambda's second
        # element, tripling the generator cost for nothing.
        sg_c, sg_f = torch.func.jvp(_mean, (c, f), (g_c, g_f))[1]
        kb = torch.as_tensor(k, device=sg_c.device, dtype=sg_c.dtype).reshape(-1)
        return kb * ((g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2)))

    q = quad(c_in, f_in)
    d_c, d_f = torch.autograd.grad(q.sum(), (c_in, f_in), allow_unused=True)
    cost.gen_fwd += 1
    cost.gen_jvp += 1
    cost.gen_vjp += 1
    if d_c is None:
        return torch.zeros(coords.shape[0], device=coords.device)
    kb = torch.as_tensor(k, device=d_c.device, dtype=d_c.dtype).reshape(-1, 1, 1)
    k3_gg_c = kb * d_c.detach()
    k3_gg_f = kb * (d_f.detach() if d_f is not None else torch.zeros_like(f_in))
    m3 = (k3_gg_c * g_c).sum(dim=(1, 2)) + (k3_gg_f * g_f).sum(dim=(1, 2))
    return m3 / v_f.clamp(min=1e-12).pow(1.5)


def trace_scale(post_fn, coords, feats, mask, k, n_probe=1, generator=None,
                cost=_NOCOST, com_free=True):
    """r^2 = tr(Sigma)/d by Hutchinson, the isotropic scale that matches Sigma
    in trace. E_z[z' Sigma z] = tr(Sigma) for z Rademacher, and Sigma z is one
    JVP, so this is n_probe JVPs.

    Used to set the spread of the Monte-Carlo arms from the MODEL rather than
    from a tuned hyperparameter. See sigma_mc_weighted_grad for why the draws
    cannot simply be Sigma^{1/2} z.
    """
    msk = mask.unsqueeze(-1)
    # The isotropic scale must divide tr(Sigma) by the RANK of Sigma, not by the
    # ambient size. With the real generator, coordinates live on the zero-CoM
    # subspace, so Sigma has exactly 3 null directions and the coordinate block
    # contributes 3(N-1), not 3N. com_free=False is for a posterior that does
    # not project (test fixtures, and any future non-CoM model).
    n_at = mask.sum(dim=1)
    n_co = (3.0 * (n_at - 1.0).clamp(min=1.0)) if com_free else (3.0 * n_at)
    d = (n_co + n_at * feats.shape[-1]).clamp(min=1.0)
    tot = torch.zeros(coords.shape[0], device=coords.device)
    for _ in range(max(int(n_probe), 1)):
        z_c = (torch.randint(0, 2, coords.shape, generator=generator,
                             device=coords.device, dtype=coords.dtype) * 2.0 - 1.0) * msk
        z_f = (torch.randint(0, 2, feats.shape, generator=generator,
                             device=feats.device, dtype=feats.dtype) * 2.0 - 1.0) * msk
        sz_c, sz_f = sigma_times_vector(post_fn, coords, feats, mask, z_c, z_f, k, cost)
        tot = tot + ((z_c * sz_c).sum(dim=(1, 2)) + (z_f * sz_f).sum(dim=(1, 2))).detach()
    return (tot / max(int(n_probe), 1) / d).clamp(min=0.0)


def sigma_mc_weighted_grad(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
                           y, s, n_mc, generator=None, cost=_NOCOST,
                           observable_closure=False, iso_r2=None, com_free=True):
    """LGD-MC (observable_closure=False) and the observable-space closure (True).

    THE DRAWS ARE ISOTROPIC, with a scale read off the model.

    An earlier version of this function drew `Sigma z` with z ~ N(0, I) and
    described it as sampling from the posterior covariance. That is wrong: the
    covariance of Sigma z is Sigma Sigma^T = Sigma^2, not Sigma. Sampling from
    N(0, Sigma) needs Sigma^{1/2} z, and a single JVP through the generator
    gives Sigma v, never its square root. The error was not a constant factor --
    it is sqrt(lambda) per eigendirection, so with k = (1-t)^2/t the draws came
    out ~3x too large early in the trajectory and ~80x too small late, which
    collapsed both arms onto plug-in exactly where the spread should matter.

    The fix keeps the property that distinguishes these arms from TFG -- the
    spread is not a tuned hyperparameter -- by matching Sigma in TRACE:
    r^2 = tr(Sigma)/d, one extra JVP. The draws are then r z, z ~ N(0, I).

    What this costs: the draws are isotropic, so they do not see Sigma's
    anisotropy. A genuinely anisotropic draw needs Sigma^{1/2}, i.e. a Lanczos
    or Chebyshev approximation at several JVPs per draw. That is a legitimate
    future arm; it is not what this one does, and the write-up must not claim
    otherwise. The memo's own reference implementation offers exactly this
    isotropic variant (`lib._draw(..., iso_r2=...)`).

      LGD-MC:  grad of log mean_i w_i, w_i = exp(-(y - f(X_i))^2 / 2 s^2)
               -- a softmax-weighted average of per-draw plug-in gradients.
      OSC:     fit a Gaussian to the SCALAR F = f(X) across draws and integrate
               the likelihood against it analytically, giving
               (r/S) gbar + ((r^2 - S)/S^2) Cov(F, grad f),
               with r = y - mean(F), S = s^2 + Var(F). Same guide budget,
               lower variance where F is near-Gaussian.

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
    if iso_r2 is None:
        iso_r2 = trace_scale(post_fn, coords, feats, mask, k, 1, generator, cost,
                             com_free=com_free)
    r = iso_r2.clamp(min=0.0).sqrt().view(-1, 1, 1)
    fs, gcs, gfs = [], [], []
    for _ in range(K):
        z_c = torch.randn(m_c.shape, generator=generator, device=m_c.device,
                          dtype=m_c.dtype) * msk
        z_f = torch.randn(m_f.shape, generator=generator, device=m_f.device,
                          dtype=m_f.dtype) * msk
        d_c, d_f = r * z_c, r * z_f
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


def btvg2_weighted_grad(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
                        y, s, tau, n_mc, generator=None, cost=_NOCOST,
                        gate=True, orth=True, cap=1.0, com_free=True):
    """BTVG-2: lgd_mc's mean term plus BTVG's variance term, both read off the
    SAME K smoothed draws. Returned in m-space; the caller pulls back with J^T.

    WHY. The full run showed the binding constraint is chemistry per unit of
    push, and lgd_mc's smoothed estimator is the only one that kept chemistry
    at w = 4. BTVG's analytic variance term (a) needs an HVP and a symmetric-J
    assumption, (b) goes non-positive on ~5% of steps, (c) drags the property
    mean away from far targets, and (d) eats the clip budget. Each piece below
    answers one of those.

    Draws are made exactly as sigma_mc_weighted_grad makes them (same RNG
    calls, same order), so at the first guided step the mean term is
    bit-identical to lgd_mc's and `btvg2 - lgd_mc` isolates the variance term.

        mean    M     = sum_i w_i (y - F_i)/s^2 g_i        lgd_mc, unchanged
        moments mu    = mean_i F_i,  V = var_i F_i         unbiased, V >= 0
        grad V  dV/dm = 2/(K-1) sum_i (F_i - mu) g_i       exact for fixed draws
        coeff   b     = 1/2 (1/s^2) (1 - tau^2/V)_+        forward KL to
                        N(y, tau^2), in plug units (x tau^2/s^2); never widens
        gate    gamma = exp(-(y - mu)^2 / (2 V))           concentrate only when
                        y is plausible under the current predictive law
        orth    P v   = v - (v.gbar / gbar.gbar) gbar       gbar = mean_i g_i, so
                        the variance step leaves mu_hat unchanged to first order
        cap     |gamma b P dV| <= cap max(|y - mu|, sqrt V) |gbar| / s^2
                        i.e. no larger than the mean step for the LARGER of
                        the actual miss and a one-sd miss (the residual floor
                        keeps the far-target protection; the sd floor keeps
                        it alive on target). Not |M|: M -> 0 when the target is hit, and
                        a |M| cap then crushed the variance step (measured 17x)
                        in exactly the centred-but-wide regime it exists for.
                        Near an extremum of f (gbar ~ 0) it binds hard, which
                        is where flatness-seeking is least trustworthy.
        field   M - gamma b P(dV/dm)

    GEOMETRY CAVEAT (23 Sep audit, docs/methods/BTVG_FAILURE_AUDIT_AND_REDESIGN.md
    section 3): `orth` and "never widens" hold in m-SPACE only. The step is
    pulled back by J^T and applied to x_t, where the mean changes at rate
    g^T J J^T S_m != 0 in general (exact counterexample with symmetric J;
    real-checkpoint |cos| 0.45-0.51 at t=0.8). Repair = project against
    Q J^T g and Q J^T h in x-space. Not implemented here; the chemistry
    guard (chem_safe_project) does project in x-space.

    WHAT "DESCENDING V" MEANS HERE, stated plainly. The draws are isotropic
    with a detached radius, so dV/dm measures how FLAT f is around m under
    that probe -- not the generator's commitment (J), and not Sigma's shape.
    It is "move to where the remaining noise can no longer change f". That can
    also mean "where the predictor is flat but the data is not" (an
    off-manifold / reward-hacking risk); the sweep's guide_eval_gap and
    mol_stability are the measurements that would show it.
    "Leaves mu unchanged" is exact for mu_hat, the K-draw isotropic estimate;
    it says nothing about the true posterior mean of f (SMG's c term).

    n_mc must be >= 2 (a variance needs two draws). It is refused rather than
    bumped, because bumping would silently break the bit-identical mean term
    against lgd_mc at the same n_mc.
    """
    if int(n_mc) < 2:
        raise ValueError("btvg2 needs n_mc >= 2 (got %r)" % (n_mc,))
    K = int(n_mc)
    msk = mask.unsqueeze(-1)
    iso_r2 = trace_scale(post_fn, coords, feats, mask, k, 1, generator, cost,
                         com_free=com_free)
    r = iso_r2.clamp(min=0.0).sqrt().view(-1, 1, 1)
    fs, gcs, gfs = [], [], []
    for _ in range(K):
        z_c = torch.randn(m_c.shape, generator=generator, device=m_c.device,
                          dtype=m_c.dtype) * msk
        z_f = torch.randn(m_f.shape, generator=generator, device=m_f.device,
                          dtype=m_f.dtype) * msk
        mc = (m_c + r * z_c).detach().requires_grad_(True)
        mf = (m_f + r * z_f).detach().requires_grad_(True)
        val = f_net(mc, mf, mask)
        gc, gf = torch.autograd.grad(val.sum(), (mc, mf))
        cost.guide_fwd += 1
        cost.guide_bwd += 1
        fs.append(val.detach())
        gcs.append(gc.detach())
        gfs.append(gf.detach())
    F = torch.stack(fs)                                   # [K, B]
    GC = torch.stack(gcs)
    GF = torch.stack(gfs)
    B = F.shape[1]

    # the mean term: lgd_mc exactly
    lw = -((y - F) ** 2) / (2.0 * s ** 2)
    wts = torch.softmax(lw, dim=0).view(K, -1, 1, 1)
    coef = ((y - F) / s ** 2).view(K, -1, 1, 1)
    M_c, M_f = (wts * coef * GC).sum(0), (wts * coef * GF).sum(0)

    mu = F.mean(0)
    V = F.var(0, unbiased=True)                           # >= 0 by construction
    dF = (F - mu).view(K, -1, 1, 1)
    dV_c = 2.0 * (dF * GC).sum(0) / (K - 1)
    dV_f = 2.0 * (dF * GF).sum(0) / (K - 1)

    tau2 = torch.as_tensor(tau, device=F.device, dtype=F.dtype) ** 2
    V_safe = V.clamp(min=1e-12)
    b = 0.5 / s ** 2 * (1.0 - tau2 / V_safe).clamp(min=0.0)
    if gate == "band":
        # concentrate only once the predicted property is INSIDE the band
        # (width delta = 1.96 tau): hold what is already a hit, and leave
        # everything else to the mean term. The plausibility gate below
        # concentrated at |y - mu| ~ sqrt V, i.e. locked samples in just
        # outside the band (pilot: -0.008 in_band vs lgd_mc, MAE slightly
        # better, combined z ~ -2 over five comparisons).
        band = 1.96 * torch.as_tensor(tau, device=F.device, dtype=F.dtype)
        gamma = torch.exp(-((y - mu) ** 2) / (2.0 * band ** 2))
    elif gate:
        gamma = torch.exp(-((y - mu) ** 2) / (2.0 * V_safe))
    else:
        gamma = torch.ones_like(mu)

    gb_c, gb_f = GC.mean(0), GF.mean(0)
    if orth:
        num = (dV_c * gb_c).sum(dim=(1, 2)) + (dV_f * gb_f).sum(dim=(1, 2))
        den = (gb_c ** 2).sum(dim=(1, 2)) + (gb_f ** 2).sum(dim=(1, 2))
        proj = torch.where(den > 0, num / den.clamp(min=1e-30),
                           torch.zeros_like(num)).view(-1, 1, 1)
        dV_c, dV_f = dV_c - proj * gb_c, dV_f - proj * gb_f

    a = (gamma * b).view(-1, 1, 1)
    S_c, S_f = -a * dV_c, -a * dV_f                       # the variance step
    capped = torch.zeros(B, dtype=torch.bool, device=F.device)
    nM = ((M_c ** 2).sum(dim=(1, 2)) + (M_f ** 2).sum(dim=(1, 2))).sqrt()
    if cap is not None:
        nS = ((S_c ** 2).sum(dim=(1, 2)) + (S_f ** 2).sum(dim=(1, 2))).sqrt()
        ng = ((gb_c ** 2).sum(dim=(1, 2)) + (gb_f ** 2).sum(dim=(1, 2))).sqrt()
        lim = cap * torch.maximum((y - mu).abs(), V.sqrt()) * ng / s ** 2
        capped = nS > lim
        sc = torch.where(capped, lim / nS.clamp(min=1e-30),
                         torch.ones_like(nS)).view(-1, 1, 1)
        S_c, S_f = S_c * sc, S_f * sc
        nS = nS * sc.view(-1)
    else:
        nS = ((S_c ** 2).sum(dim=(1, 2)) + (S_f ** 2).sum(dim=(1, 2))).sqrt()
    # the variance step's share of the field, in [0, 1] -- bounded, so its
    # running mean is not dominated by on-target samples where |M| -> 0
    share = torch.where(nS + nM > 0, nS / (nS + nM).clamp(min=1e-30),
                        torch.zeros_like(nM))
    diag = {"btvg2_gate": gamma.detach(),
            "btvg2_V_over_tau2": (V / tau2).detach(),
            "btvg2_capped": capped.to(F.dtype).detach(),
            "btvg2_var_share": share.detach(),
            "obs_mean": mu.detach(), "obs_var": V.detach()}
    return M_c + S_c, M_f + S_f, diag


# --------------------------------------------------------------------------
# chemistry-safe guidance (CSG, 23 Sep, after the full run and the BTVG-2
# pilot): remove from any arm's step the part that would break valence
# --------------------------------------------------------------------------
#
# WHY. Every pre-registered comparison so far says the binding constraint is
# chemistry per unit of push: a stronger push hits more targets and breaks
# more molecules, and lgd_mc won by breaking fewer at w = 4. CSG attacks that
# constraint directly. At each guided step it measures, on the predicted clean
# molecule m, how far each atom is from its allowed valence -- a smooth
# relaxation of the SAME distance rule the evaluator uses (Hoogeboom et al.;
# evaluation.bond_order) -- and removes from the step only the component that
# would increase that violation to first order.
#
# ONE-SIDED, BY DESIGN. The component that would REDUCE the violation is left
# alone and no chemistry push is ever ADDED. But every guarded step is then
# violation-non-increasing to first order (and strictly decreasing wherever
# the property push happened to help), so the guided trajectory is biased
# toward lower violation and CAN end up MORE stable than unguided (pilot, mu
# w=8: 0.417 vs unguided 0.390). An earlier comment here said that could not
# happen; it was wrong. Because the relaxation mirrors the evaluator's own
# table, a stability gain could be the table being gamed, so every CSG result
# must be scored with an INDEPENDENT rule too (RDKit rdDetermineBonds,
# covalent radii): proj1/scripts/independent_chem.py.
#
# FIXED BEFORE ANY RUN: kappa = 0.03 A (the sigmoid width, the size of the
# table's own margins, 3-10 pm) and a type temperature of 0.1 on the one-hot
# features. Neither was tuned.

QM9_TYPES = ("H", "C", "N", "O", "F")
CHEM_KAPPA = 0.03
CHEM_TYPE_TEMP = 0.1


def _bond_threshold_tables(device, dtype):
    """t_k[a, b] in angstrom, k = single/double/triple, INCLUDING the margins,
    so bond_order(a, b, d) == sum_k 1[d < t_k[a, b]] exactly (the tables are
    nested: t3 < t2 < t1 wherever they exist). Missing entries -> -10 A, i.e.
    that order is impossible for the pair."""
    from evaluation import BONDS1, BONDS2, BONDS3, MARGIN1, MARGIN2, MARGIN3
    n = len(QM9_TYPES)
    out = []
    for table, margin in ((BONDS1, MARGIN1), (BONDS2, MARGIN2), (BONDS3, MARGIN3)):
        t = torch.full((n, n), -10.0, device=device, dtype=dtype)
        for i, a in enumerate(QM9_TYPES):
            for j, b in enumerate(QM9_TYPES):
                if a in table and b in table[a]:
                    t[i, j] = (table[a][b] + margin) / 100.0
        out.append(t)
    return torch.stack(out)                               # [3, n, n]


def soft_valence_violation(m_c, m_f, mask, kappa=CHEM_KAPPA, temp=CHEM_TYPE_TEMP,
                           per_atom=False):
    """P = sum_i mask_i (v_i - A_i)^2, a smooth version of "atom i is unstable".

      p_ia   = softmax(m_f / temp)                          soft element
      b_ij   = sum_ab p_ia p_jb sum_k sigmoid((t_k[a,b] - d_ij) / kappa)
      v_i    = sum_{j != i} b_ij                            soft valence
      A_i    = sum_a p_ia allowed(a)                        soft allowed valence

    In the limit kappa -> 0 with one-hot types this is EXACTLY the evaluator's
    integer valence (gate C1), so P = 0 iff every atom is stable.
    """
    if m_f.shape[-1] != len(QM9_TYPES):
        raise ValueError("soft_valence_violation assumes the QM9 types %s"
                         % (QM9_TYPES,))
    from evaluation import ALLOWED_VALENCE
    T = _bond_threshold_tables(m_c.device, m_c.dtype)     # [3, 5, 5]
    allowed = torch.tensor([ALLOWED_VALENCE[a] for a in QM9_TYPES],
                           device=m_c.device, dtype=m_c.dtype)
    msk = mask.to(m_c.dtype)
    p = torch.softmax(m_f / temp, dim=-1) * msk.unsqueeze(-1)       # [B,N,5]
    diff = m_c.unsqueeze(2) - m_c.unsqueeze(1)                      # [B,N,N,3]
    d = torch.sqrt((diff ** 2).sum(-1) + 1e-12)                     # [B,N,N]
    # S[b,i,j,a,c] = sum_k sigmoid((t_k[a,c] - d_ij)/kappa)
    S = torch.sigmoid((T.view(3, 1, 1, 1, 5, 5)
                       - d.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)) / kappa).sum(0)
    b = torch.einsum("bia,bjc,bijac->bij", p, p, S)
    pair = msk.unsqueeze(2) * msk.unsqueeze(1)
    eye = torch.eye(m_c.shape[1], device=m_c.device, dtype=m_c.dtype)
    b = b * pair * (1.0 - eye)
    v = b.sum(-1)
    A = (p * allowed).sum(-1)
    viol = ((v - A) ** 2) * msk
    return viol if per_atom else viol.sum(-1)


def chem_safe_project(G_c, G_f, a_c, a_f):
    """One-sided projection: G' = G - max(0, <a,G>)/|a|^2 a, per sample.
    a = J^T grad_m P is the direction in x_t that raises the violation, so
    <a, G'> <= 0: the step can no longer increase P to first order. Samples
    where <a, G> <= 0 (the step does not hurt) are returned untouched."""
    num = (a_c * G_c).sum(dim=(1, 2)) + (a_f * G_f).sum(dim=(1, 2))
    den = (a_c ** 2).sum(dim=(1, 2)) + (a_f ** 2).sum(dim=(1, 2))
    active = (num > 0) & (den > 0)
    coef = torch.where(active, num / den.clamp(min=1e-30),
                       torch.zeros_like(num)).view(-1, 1, 1)
    P_c, P_f = G_c - coef * a_c, G_f - coef * a_f
    nG = ((G_c ** 2).sum(dim=(1, 2)) + (G_f ** 2).sum(dim=(1, 2))).sqrt()
    nR = (((G_c - P_c) ** 2).sum(dim=(1, 2)) + ((G_f - P_f) ** 2).sum(dim=(1, 2))).sqrt()
    removed = torch.where(nG > 0, nR / nG.clamp(min=1e-30), torch.zeros_like(nG))
    return P_c, P_f, active, removed


def tolerance_band_step(u_c, u_f, b_c, b_f, e, tau, eta=1.0, radius=None):
    """Component D: the largest diversifying edit that does not worsen the property.

    Solves, per sample,

        d* = argmax_d [ u'd - ||d||^2 / (2 eta) ]
             s.t.  l <= b'd <= h,   ||d|| <= R

    with the BASELINE-RELATIVE envelope from D_WHY_INNOVATIVE.md section 3:

        A = max(tau, |e|),   l = -A - e,   h = A - e

    so when the guide is already outside the tolerance band (|e| > tau) the rule
    degrades to "do not make it worse" rather than "get inside the band now",
    and tau is inert there. b'd is the first-order change in the property caused
    by the edit.

    Closed form (section 4): with r = eta u, n = b/||b||, and w the projection of
    r onto the ball, return w if it already satisfies the slab; otherwise

        s = clip(n'w, l/||b||, h/||b||)
        d = s n + min(1, sqrt(max(0, R^2 - s^2))/||v_r||) v_r,   v_r = r - (n'r) n

    There is no division by the active bound, so the degenerate cases need no
    special handling. The zero-CoM subspace projection Q is supplied by the
    caller's pull-back, as for every other arm.

    NOTE ON STATUS. The TOP6 memo rejects D as a method (section 7): the
    dead-zone objective and the norm-constrained QP are both standard, and a
    local acceptance test is not a terminal guarantee. It is implemented here
    because it is in the project's plan of record and the benchmark, not an
    argument, should decide. It is marked OWN-REJECTED in the plan.
    """
    B = u_c.shape[0]
    flat_u = _flat(u_c, u_f)
    flat_b = _flat(b_c, b_f)
    r = eta * flat_u
    bn = flat_b.norm(dim=1, keepdim=True).clamp(min=1e-12)
    n = flat_b / bn
    A = torch.maximum(tau, e.abs()).view(-1, 1)
    ev = e.view(-1, 1)
    lo, hi = (-A - ev) / bn, (A - ev) / bn

    if radius is None:
        R = r.norm(dim=1, keepdim=True)
    else:
        R = torch.as_tensor(radius, dtype=r.dtype, device=r.device).expand(B, 1)
    rn = r.norm(dim=1, keepdim=True).clamp(min=1e-12)
    w = r * torch.minimum(torch.ones_like(rn), R / rn)

    proj = (w * n).sum(1, keepdim=True)
    inside = (proj >= lo) & (proj <= hi)

    sv = torch.clamp((w * n).sum(1, keepdim=True), lo, hi)
    v_r = r - (r * n).sum(1, keepdim=True) * n
    vn = v_r.norm(dim=1, keepdim=True).clamp(min=1e-12)
    tang = torch.sqrt(torch.clamp(R ** 2 - sv ** 2, min=0.0))
    d = sv * n + torch.minimum(torch.ones_like(vn), tang / vn) * v_r
    d = torch.where(inside, w, d)

    nc = u_c[0].numel()
    return d[:, :nc].view_as(u_c), d[:, nc:].view_as(u_f)


class ResidualCalibrationHead(torch.nn.Module):
    """RCH (Haimo v1): LEARN SMG's correction instead of deriving it.

    SMG pays a JVP and an HVP per step to compute c = 1/2 tr(H Sigma). Haimo v1
    observes that for the properties tested, c is mostly a function of t, so a
    ridge head on cheap E(3)-invariant summary statistics of the state should
    reproduce it at plug-in cost (one VJP, no JVP, no HVP).

        C = (1 - t) * phi(state)' beta,     guide with  y - f(m) - C

    The (1-t) factor makes the correction vanish at the clean endpoint, which is
    correct: Sigma -> 0 there, so c -> 0.

    This is the cheapest test that can DEFLATE the SMG claim: if a lookup table
    reproduces c, then SMG's state dependence is not earning its cost. It is a
    COMPARE arm, never a contribution -- the parameterisation is prior work
    (Residual grad-DB, Liu et al. ICLR 2025 Eq. 15; VGG-Flow, NeurIPS 2025
    Eq. 16), and Reward Score Matching (2026) App. G.1 finds such residuals
    "effectively negligible".

    Fit with `fit_rch.py`, which caches (phi, c_exact) pairs from real molecules
    noised to random t and solves the ridge system in closed form.
    """

    N_FEAT = 30

    def __init__(self, n_feat=None):
        super().__init__()
        n = n_feat or self.N_FEAT
        self.register_buffer("beta", torch.zeros(n))
        self.fitted = False

    @staticmethod
    def features(coords, feats, mask, t, k, n_feat=None):
        """E(3)-invariant summary statistics. Rotation/translation invariant by
        construction: only norms, pairwise distances and type histograms."""
        m = mask.unsqueeze(-1)
        n = mask.sum(1).clamp(min=1.0)
        # Centre on the valid atoms' centre of mass FIRST: a radius measured
        # from the origin is not translation invariant, and the data only
        # happens to be centred already.
        c = coords * m
        com = c.sum(1, keepdim=True) / n.view(-1, 1, 1)
        c = (c - com) * m
        r2 = (c ** 2).sum(-1)
        d = torch.cdist(c, c)
        dpad = d + (1 - mask).unsqueeze(1) * 1e3
        dmin = dpad.masked_fill(torch.eye(d.shape[-1], device=d.device,
                                          dtype=torch.bool), 1e3).min(-1).values
        dmin = (dmin * mask).sum(1) / n
        tv = t.view(-1)
        kk = torch.as_tensor(k, device=coords.device, dtype=coords.dtype).reshape(-1)
        if kk.numel() == 1:
            kk = kk.expand_as(tv)
        hist = (feats * m).sum(1) / n.unsqueeze(-1)
        base = [torch.ones_like(tv), tv, tv ** 2, tv ** 3, (1 - tv),
                (1 - tv) ** 2, kk, kk * tv, n / 29.0,
                (r2 * mask).sum(1) / n, ((r2 * mask).sum(1) / n).sqrt(),
                dmin, dmin ** 2,
                (feats ** 2 * m).sum((1, 2)) / n,
                # NOT the L1 norm of coordinates -- that is not rotation
                # invariant. Mean pairwise distance is.
                (d * (mask.unsqueeze(1) * mask.unsqueeze(2))).sum((1, 2))
                / (n * (n - 1).clamp(min=1.0))]
        cols = base + [hist[:, i] for i in range(hist.shape[1])]
        cols = cols + [hist[:, i] * tv for i in range(hist.shape[1])]
        X = torch.stack(cols, dim=1)
        want = (ResidualCalibrationHead.N_FEAT if n_feat is None
                else int(n_feat))
        if X.shape[1] < want:
            X = torch.cat([X, torch.zeros(X.shape[0], want - X.shape[1],
                                          device=X.device, dtype=X.dtype)], 1)
        return X[:, :want]

    def forward(self, coords, feats, mask, t, k):
        phi = self.features(coords, feats, mask, t, k, self.beta.numel())
        return (1.0 - t.view(-1)) * (phi * self.beta).sum(1)


def _diversity_direction(f_net, post_fn, coords, feats, mask, m_c, m_f,
                         cost=_NOCOST):
    """Ascent direction for batch diversity, in the guide's embedding space.

    Component D moves for diversity subject to the property staying in band, so
    it needs a diversity objective distinct from the property. We use pairwise
    repulsion among the batch in the guide's pooled invariant embedding:

        D = (1 / B(B-1)) sum_{i != j} || h_i - h_j ||^2

    whose gradient pushes each sample away from the batch mean embedding. The
    embedding is the same space COMPONENTS_ABCD_MATH_AND_PROOFS.md section 3.8
    uses for the alignment argument, so the two are talking about the same
    coordinates. Pulled back through J like every other arm.

    Falls back to the zero direction if the guide exposes no `embed`, in which
    case the band arm degenerates to "do not worsen the property", which is
    still a valid (if weaker) reading of the rule.
    """
    if not hasattr(f_net, "embed"):
        return torch.zeros_like(coords), torch.zeros_like(feats)
    mc = m_c.detach().requires_grad_(True)
    mf = m_f.detach().requires_grad_(True)
    h = f_net.embed(mc, mf, mask)
    B = h.shape[0]
    if B < 2:
        return torch.zeros_like(coords), torch.zeros_like(feats)
    hbar = h.mean(0, keepdim=True)
    D = ((h - hbar) ** 2).sum() / (B - 1)
    w_c, w_f = torch.autograd.grad(D, (mc, mf), allow_unused=True)
    cost.guide_fwd += 1
    cost.guide_bwd += 1
    w_c = torch.zeros_like(mc) if w_c is None else w_c.detach()
    w_f = torch.zeros_like(mf) if w_f is None else w_f.detach()
    return _pullback(post_fn, coords, feats, w_c, w_f, cost)


def grad_property_variance(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
                           g_c, g_f, cost=_NOCOST, include_hessian=True):
    """(V_F, grad_x V_F) with V_F = g' Sigma g, the property's variance at x_t.

    V_F is already computed by every SMG-family arm and used only as a
    denominator. Its GRADIENT is what a variance-targeting objective needs.

    THE GRADIENT HAS TWO TERMS, and the obvious implementation only has one.
    Writing g = grad f(m(x)) and Sigma = k J with J = dm/dx,

        V(x) = k g(m(x))' J(x) g(m(x))

        grad_x V = 2 J' H (Sigma g)          <- g moves with x, since dg/dx = H J
                   + k grad_x[g' J(x) g]|_g  <- the mean map's own curvature

    Differentiating the scalar with `g` detached gives ONLY the second term. It
    is not a small correction: whenever the posterior mean is close to linear in
    x_t -- which is the regime the whole Tweedie construction assumes -- the
    second term nearly vanishes and the first term IS the gradient. An earlier
    revision shipped with only the second term and the exact gate in
    proj1/tests/test_v2_arms.py measured it as identically zero against an
    autograd reference.

    THE 2x FACTOR ASSUMES J IS SYMMETRIC. Exactly, the gradient is
    J^T H Sigma g + k J^T H J^T g + k grad(g'Jg); the two Hessian terms are
    equal, giving 2 J^T H Sigma g, only when J = J^T. J is a posterior-mean
    Jacobian and IS symmetric for an exact score, but the trained network is
    not exact: the measured asymmetry E|u'Jv - v'Ju| / rms on the real
    generator runs 0.045 (t=0.5) -> 0.63 (t=0.8). Calibrated against an
    autograd reference, that level of asymmetry costs 15-40% in magnitude with
    the direction preserved (cos > 0.95). Accepted deliberately: the exact form
    needs an extra VJP and HVP per step, and a 15-40% magnitude error on a term
    whose scale is set by a swept strength knob is not what limits this arm.
    The same assumption underlies every SMG-family arm already in the sweep.

    COST: one JVP (Sigma g), one HVP, one VJP, plus the reverse pass over the
    curvature scalar -- comparable to smg2_curv, and the HVP is the same
    construction _hvp already uses. `include_hessian=False` drops the first term
    and is kept only so the ablation can measure what it is worth; it is not a
    correct gradient and no arm should use it by default.

    For an affine property (the ridge descriptor) H is exactly 0, `_hvp`
    reports that by returning None, and the first term is correctly skipped.

    Returned in x_t space, so the caller does NOT pull it back again: unlike
    the mean term, which is a weight on g that J^T then transports, this is
    already a derivative with respect to x_t.
    """
    c_in = coords.detach().requires_grad_(True)
    f_in = feats.detach().requires_grad_(True)

    def _mean(cc, ff):
        p = post_fn(cc, ff)
        return p.mean_coords, p.mean_feats

    sg_c, sg_f = torch.func.jvp(_mean, (c_in, f_in), (g_c, g_f))[1]
    kb = torch.as_tensor(k, device=sg_c.device, dtype=sg_c.dtype).reshape(-1)
    V = kb * ((g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2)))
    d_c, d_f = torch.autograd.grad(V.sum(), (c_in, f_in), allow_unused=True)
    cost.gen_fwd += 1
    cost.gen_jvp += 1
    cost.gen_vjp += 1
    d_c = torch.zeros_like(c_in) if d_c is None else d_c.detach()
    d_f = torch.zeros_like(f_in) if d_f is None else d_f.detach()

    if include_hessian:
        # Sigma g = k J g, and the JVP above already produced J g.
        kv = kb.reshape(-1, 1, 1)
        hs_c, hs_f = _hvp(f_net, m_c, m_f, mask,
                          kv * sg_c.detach(), kv * sg_f.detach(), cost)
        if hs_c is not None:
            p_c, p_f = _pullback(post_fn, coords, feats, hs_c, hs_f, cost)
            d_c = d_c + 2.0 * p_c
            d_f = d_f + 2.0 * p_f
    return V.detach(), d_c, d_f


def spbc_displacement(f_net, post_fn, coords, feats, mask, m_c, m_f, g_c, g_f,
                      y, eta=1.0, max_radius=None, cost=_NOCOST, fval=None):
    """SPBC -- shape-preserving bias correction. Returns a STATE DISPLACEMENT.

    MOTIVATION, measured. The population bias is large in the units that decide
    acceptance: alpha/plug sits **13.99 delta** off centre at the wide window,
    alpha/smg -5.69, gap/plug -2.12. A mean error of several band half-widths
    cannot be fixed by concentrating harder.

    THE MECHANISM. For any scalar outcome law, subtracting the bias
    b = E[Y] - y* leaves the mean at y* and leaves **every centered quantile
    and every pairwise difference unchanged** (memo section 3.4). So the ideal
    edit gives EVERY trajectory the SAME property increment

        nu_i = d = -eta * (mean_i f(m_i) - y*)

    WHAT ACTUALLY HAPPENS, MEASURED -- the paragraph above is the ideal, not
    the result. End to end on the real generator (alpha, B=16, 40 steps,
    t_min 0.5, w = eta = 1): max change in a centred f_A value is 3.868
    = 8.03 delta, and max change in a pairwise difference is 5.932 = 12.3
    delta, against a batch sd of 13.76. The shape is NOT preserved in absolute
    terms.

    The cause is neither clipping (0/320 sample-steps clipped at the swept
    strengths; clip=1.0 and clip=None give bit-identical output) nor the
    zero-COM projection (retention 1.0000). a_i . d_i = nu holds exactly at
    every step -- that is what the gate measures -- but f(x_1) does not follow
    the first-order forecast once 20 such steps are composed.

    THE DEFENSIBLE CLAIM IS THE RELATIVE ONE: distortion per unit of mean
    shift is 4.2 (w=1) and 2.9 (w=4) against 14.5 / 21.0 / 20.7 / 8.8 for
    `plug` at w = 0.002 / 0.01 / 0.05 / 0.25 -- roughly 3x less shape
    distortion for the same movement of the mean. Write it that way.

    AND THE INCREMENT IS A RATE, NOT A ONE-SHOT EDIT. The integrator multiplies
    the returned field by dt, so nu is dnu/dt and the realised closure of the
    bias is 1 - exp(-eta * w * (1 - t_min)), i.e. about half at the
    pre-registered eta = w = 1, t_min = 0.5. Measured: 49.0 / 43.6 / 53.9 /
    55.2 % at 10 / 20 / 40 / 80 steps -- step-count invariant, which is the
    property that matters, but NOT the full correction. w = 4 gives 147%
    (overshoot). Tune w, do not assume one pass centres the batch.

    FINALLY, `mu` IS A MINIBATCH MEAN. The sweep runs batch=128 within n=512,
    so four different increments are applied while evaluation pools all 512 --
    the common-increment property holds within a batch, not across the sample.
    At sd(f_A) = 13.76 the standard error of that batch mean is ~1.22 alpha
    units, about 2.5 delta. Raise the batch, or read the spread accordingly.

    A shared guidance multiplier does NOT do this: per-sample property change is
    (residual) x (that sample's response gradient), so a common coefficient
    produces different increments and distorts the shape it is meant to keep.

    THE EDIT. Minimum-energy state change realising a requested nu_i, with
    metric M = I:

        min ||d_i||^2 / 2  s.t.  a_i . d_i = nu_i   =>   d_i = (nu_i / r_i) a_i

    with a_i = grad_x f(m(x_t)) = J^T g (one VJP, already shared with every
    arm) and r_i = ||a_i||^2 the squared response, i.e. property change per
    unit edit.

    BOUNDS. |nu_i| <= R_i sqrt(r_i) is the largest increment reachable inside a
    trust radius. When one trajectory cannot reach the common increment, the
    whole batch is scaled by the same factor rather than clipped per sample --
    per-sample clipping destroys the common-increment property that is the
    entire point (memo section 5.4).

    r_i = 0 means the trajectory has no first-order property mobility; it gets
    nu_i = 0 rather than a division by a small number.

    UNITS: this is a displacement in x_t, NOT a score. The sampler must add it
    without the (1-t)/t score-to-velocity conversion.

    DEGENERATE UNDER PER-MOLECULE TARGETS. `d_common` centres the BATCH MEAN of
    f(m) on the MEAN of y, so with the `dist` protocol -- where every molecule
    has its own target -- SPBC ignores each molecule's target entirely and
    corrects only the aggregate. Under `dist` the mean target ~ the data mean ~
    the unguided generator's mean, so the correction is ~0 BY CONSTRUCTION, not
    by measurement. Any `dist` result for spbc (or for the SHG schedules that
    contain an spbc phase) is a null of the method's definition, not evidence
    about it. Correcting this needs a per-molecule formulation, which is a
    different arm.

    HONEST LIMITS. The forecast is the cheap one, f_A(m_t), not a rollout
    through the sampler suffix; it inherits the off-distribution error measured
    in FINDING_QUADRATIC_CLOSURE_VALIDITY.md -- and the 8-delta shape drift
    above is that error, compounded. Preserving scalar property spacings does
    NOT imply preserving structural diversity. And centering is not guaranteed
    to raise band coverage -- a skewed law can lose coverage when centred
    (memo section 3.5), so it needs an acceptance check.
    """
    a_c, a_f = _pullback(post_fn, coords, feats, g_c, g_f, cost)     # a_i = J^T g
    r = (a_c ** 2).sum(dim=(1, 2)) + (a_f ** 2).sum(dim=(1, 2))      # r_i = ||a_i||^2
    if fval is None:
        with torch.no_grad():
            z = f_net(m_c, m_f, mask)                                # forecast
        cost.guide_fwd += 1
    else:
        z = fval          # guidance_field already paid for this one
    mu = z.mean()
    d_common = -eta * (mu - y if not torch.is_tensor(y) else mu - y.mean())

    live = r > 1e-12
    if max_radius is not None:
        # largest common increment every LIVE trajectory can reach
        reach = (max_radius * r.clamp(min=1e-12).sqrt())[live]
        if reach.numel():
            cap = reach.min()
            d_common = torch.clamp(d_common, -cap, cap)

    nu = torch.where(live, torch.full_like(r, float(d_common)), torch.zeros_like(r))
    scale = (nu / r.clamp(min=1e-12)).view(-1, 1, 1)
    return (scale * a_c).detach(), (scale * a_f).detach(), {
        "spbc_bias": (mu - (y if not torch.is_tensor(y) else y.mean())).detach(),
        "spbc_increment": d_common.detach() if torch.is_tensor(d_common)
        else torch.tensor(float(d_common)),
        "spbc_live_fraction": live.double().mean().detach(),
        "spbc_mean_response": r[live].mean().detach() if live.any()
        else torch.tensor(0.0)}


# --------------------------------------------------------------------------
# the fields
# --------------------------------------------------------------------------

# "tmpd" is the published name for the uncertainty-denominator field; it is
# exactly our smg_var ablation. Aliased rather than duplicated so the two can
# never drift apart.
_MODE_ALIASES = {"tmpd": "smg_var", "pigdm": "smg_var", "dps": "plug"}
KNOWN_MODES = {"plug", "smg_mean", "smg_var", "smg", "smg2", "smg2_curv",
               "tfg_mc", "lgd_mc", "osc", "rch", "band",
               # v2 arms, see PROPOSALS_*.md and the selection memo
               "spbc", "btvg", "btvg_mean", "btvg_var",
               # BTVG-2 (23 Sep, after the full run): see btvg2_weighted_grad
               "btvg2", "btvg2_nogate", "btvg2_noorth", "btvg2_nocap",
               "btvg2_band",
               # chemistry-safe guidance: any base arm + the valence guard
               "lgd_mc_chem", "plug_chem"}
# the base arm each CSG mode wraps; the guard never changes the base field's
# own computation, so `X_chem - X` isolates the guard exactly
CHEM_BASE = {"lgd_mc_chem": "lgd_mc", "plug_chem": "plug"}

# Arms that return a STATE DISPLACEMENT rather than a score. The sampler
# must not apply the (1-t)/t score-to-velocity conversion to these.
# Arms the sampler must NOT push through the (1-t)/t score-to-velocity factor.
#
# `spbc` because its edit IS a length: d_i = (nu/r_i) a_i with nu a physical
# property increment and r_i the measured response.
#
# `band` for a different reason, and NOT because its output is a length -- it
# is measured to be eta*u orthogonally projected onto the feasible set
# (||d||/||u|| = cos(d,u) = 0.3833 to four digits), so it is a length only if
# eta carries units of [x]^2. It belongs here because its input `u` is
# grad(diversity), not grad log p(y|x_t): the (1-t)/t factor is derived for the
# Gaussian path's SCORE and has no justification for a non-score direction
# either way, so not applying it is the choice consistent with this module's
# convention.
#
# NOTE: this does NOT explain band's recorded divergence. Measured: with
# clip=None band goes non-finite at w >= 0.05 both with and without the factor,
# and with the sweep's clip=1.0 neither version diverges at any strength.
# Divergence is a magnitude problem. The factor does change the magnitude a
# lot (x19 at t=0.05, x0.053 at t=0.95) and it makes band MORE clip-bound at
# t_min=0.5, which is why the low-strength grid extension exists.
DISPLACEMENT_MODES = {"spbc", "band"}

# Arms that are not a field at all: the SAMPLER builds their step, because it
# needs the time grid (schedules normalised over every step, the next time
# point, the step size). guidance_field refuses them by name.
SAMPLER_MODES = {"tfg"}


def guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                   mode="plug", n_probe=1, generator=None, cost=_NOCOST,
                   n_mc=4, sigma_mc=0.1, want_kappa3=False, rch=None,
                   band_tau=None, band_eta=1.0, band_radius=None, t_scalar=None,
                   tau=None, spbc_eta=1.0, spbc_radius=None):
    """Return (G_coords, G_feats, diagnostics) in SCORE units.

    mode:
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
                    tuned sigma_mc (Ye et al. 2024) -- ONE ingredient of TFG,
                    not TFG. The full method is the sampler-level `tfg` arm
                    (SAMPLER_MODES), which guidance_field refuses.
      "lgd_mc"      likelihood marginalisation drawing from the model's own
                    Sigma instead of isotropic -- no tuned spread
      "osc"         observable-space closure: same guide budget as lgd_mc, but
                    the scalar observable's law is integrated analytically
      "rch"         Haimo v1's Residual Calibration Head: a LEARNED stand-in for
                    c at plug-in cost. Needs a fitted head via rch=.
      "band"        Component D, the tolerance band: the largest diversifying
                    edit whose first-order property change stays in the band
      "spbc"        shape-preserving bias correction. Gives EVERY trajectory the
                    same property increment, which leaves all centered quantiles
                    and pairwise differences intact. Returns a DISPLACEMENT.
      "btvg"        band-targeted variance guidance: descend KL to N(y*, tau^2)
                    rather than maximise the likelihood of y*. Needs tau.
      "btvg_mean"   the mean half alone   (ablation)
      "btvg_var"    the variance half alone (ablation -- the decisive one)

    Every mode also returns diagnostics. Pass want_kappa3=True to log the
    standardised skew of the observable alongside the field; it costs 2-3 extra
    generator passes and changes nothing about the field itself.
    """
    # An unknown mode used to fall through to the plug-in branch and return a
    # plausible field under the wrong name. Alias the published name, then
    # refuse anything unrecognised.
    mode = _MODE_ALIASES.get(mode, mode)
    if mode in SAMPLER_MODES:
        raise ValueError("mode %r is built by the sampler (FlowSampler/VPSampler "
                         "`tfg` branch), not by guidance_field" % mode)
    if mode not in KNOWN_MODES:
        raise ValueError("unknown guidance mode %r; known: %s"
                         % (mode, ", ".join(sorted(KNOWN_MODES))))

    if mode in CHEM_BASE:
        # 1. the base arm's field, computed exactly as the base arm computes it
        #    (same RNG calls, so at the first guided step it is bit-identical)
        G_c, G_f, dg = guidance_field(
            f_net, post_fn, coords, feats, mask, y, s, mode=CHEM_BASE[mode],
            n_probe=n_probe, generator=generator, cost=cost, n_mc=n_mc,
            sigma_mc=sigma_mc, want_kappa3=want_kappa3, rch=rch,
            band_tau=band_tau, band_eta=band_eta, band_radius=band_radius,
            t_scalar=t_scalar, tau=tau, spbc_eta=spbc_eta,
            spbc_radius=spbc_radius)
        # 2. the valence-violation gradient at the predicted clean molecule,
        #    pulled back to x_t: the direction in x_t that raises P(m(x_t))
        with torch.no_grad():
            post = post_fn(coords, feats)
        cost.gen_fwd += 1
        mc = post.mean_coords.detach().requires_grad_(True)
        mf = post.mean_feats.detach().requires_grad_(True)
        P = soft_valence_violation(mc, mf, mask)
        c_c, c_f = torch.autograd.grad(P.sum(), (mc, mf))
        a_c, a_f = _pullback(post_fn, coords, feats, c_c.detach(), c_f.detach(),
                             cost)
        # 3. remove only the part of the step that would raise it
        G_c, G_f, active, removed = chem_safe_project(G_c.detach(), G_f.detach(),
                                                      a_c, a_f)
        dg = dict(dg)
        dg.update({"chem_active": active.to(G_c.dtype).detach(),
                   "chem_removed": removed.detach(),
                   "chem_P": P.detach()})
        return G_c.detach(), G_f.detach(), dg

    with torch.no_grad():
        post = post_fn(coords, feats)
    cost.gen_fwd += 1
    m_c, m_f, k = post.mean_coords, post.mean_feats, post.k
    fval, g_c, g_f = property_derivs(f_net, m_c, m_f, mask, cost)

    if mode == "spbc":
        d_c, d_f, dg = spbc_displacement(f_net, post_fn, coords, feats, mask,
                                         m_c, m_f, g_c, g_f, y, spbc_eta,
                                         spbc_radius, cost, fval=fval)
        dg.update({"f": fval, "c": torch.zeros_like(fval),
                   "v_f": torch.zeros_like(fval), "k": k, "units": "displacement"})
        return d_c, d_f, dg

    if mode in ("btvg", "btvg_mean", "btvg_var"):
        # Band-Targeted Variance Guidance: descend
        #   KL( N(mu_F, V_F) || N(y*, tau^2) )
        # instead of maximising the Gaussian likelihood of y*.
        #
        # WHY THE OBJECTIVE, NOT THE ESTIMATOR, IS THE CHANGE. Expanding the
        # usual likelihood score gives TWO terms,
        #     (y-mu_F)/S grad(mu_F)  +  1/2[(y-mu_F)^2/S^2 - 1/S] grad(V_F),
        # and the second is POSITIVE when (y-mu_F)^2 > S: the likelihood WIDENS
        # the observable when the target is far, because a wider law puts more
        # mass on a distant y. That is correct inference and wrong for design,
        # and it matches `plug` reaching 105% of the data spread on alpha.
        #
        # KL replaces that coefficient with -1/2(1/tau^2 - 1/V_F), whose sign
        # depends on V_F vs the TOLERANCE, not on the residual. It therefore
        # concentrates even when far from target, and is self-limiting: once
        # V_F < tau^2 it stops rather than collapsing the batch. "Stops" is
        # enforced, not emergent: the coefficient is clamped at <= 0 below.
        #
        # tau IS NOT delta. A centred Gaussian with sd = delta gives only
        # 68.3% band coverage; 95% needs sd ~ delta/1.96. The caller passes
        # tau explicitly and the default in the sweep is delta/1.96.
        if tau is None:
            raise ValueError("mode %r needs tau (target property sd)" % mode)
        tau_t = torch.as_tensor(tau, device=coords.device, dtype=coords.dtype)
        V, dV_c, dV_f = grad_property_variance(f_net, post_fn, coords, feats,
                                               mask, m_c, m_f, k, g_c, g_f,
                                               cost)
        # V_F CAN COME OUT NEGATIVE, and clamping it to +1e-12 is worse than
        # useless: 1/V then becomes 1e12 and b = +5e11, the largest widening
        # step the clip allows, exactly on the samples where the variance model
        # has broken down. Sigma = k dm/dx is symmetric only for an exact
        # score, and the measured asymmetry of the real generator is 0.05-0.63
        # over t in [0.5, 0.8], so the quadratic form is not PSD in practice.
        # Where V <= 0 the variance term is switched off instead.
        V_ok = V > 0
        V_safe = V.clamp(min=1e-12)
        G_c = torch.zeros_like(coords)
        G_f = torch.zeros_like(feats)
        if mode in ("btvg", "btvg_mean"):
            a = (-(fval - y) / tau_t ** 2).view(-1, 1, 1)
            mc, mf = _pullback(post_fn, coords, feats, a * g_c, a * g_f, cost)
            G_c, G_f = G_c + mc, G_f + mf
        if mode in ("btvg", "btvg_var"):
            # ...and the coefficient is clamped to be <= 0, so BTVG may
            # concentrate and may stop, but never widens. V_F is a per-sample
            # posterior variance and goes to 0 as t -> 1 by construction
            # (k = (1-t)^2/t falls by ~780x over t in [0.5, 0.975]), while
            # tau^2 is a population band half-width. So V_F < tau^2 eventually
            # happens on EVERY trajectory regardless of the batch, and without
            # this clamp the sign flip turns the last ~15% of every run into
            # active widening (measured b = +5.3, +110.8, +78.6).
            # The clamp alone already handles V <= 0: V_safe is then 1e-12,
            # which is below tau^2, so the raw coefficient is POSITIVE
            # (+5e11 at the measured tau) and the clamp zeroes it. An extra
            # torch.where(V_ok, ...) here was dead code -- it survived its
            # own mutation because nothing could distinguish it. V_ok is
            # kept, but only to report how often this happened.
            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V_safe)).clamp(max=0.0)
            b = b.view(-1, 1, 1)
            G_c, G_f = G_c + b * dV_c, G_f + b * dV_f
        return G_c.detach(), G_f.detach(), {
            "f": fval, "c": torch.zeros_like(fval), "v_f": V, "k": k,
            "btvg_tau": tau_t.expand_as(fval).detach(),
            "btvg_V_raw": V.detach(),
            "btvg_V_nonpositive": (~V_ok).to(V.dtype).detach(),
            "btvg_V_over_tau2": (V / tau_t ** 2).detach()}

    if mode == "rch":
        # Learned stand-in for c, at plug-in cost: one VJP, no JVP, no HVP.
        if rch is None or not getattr(rch, "fitted", False):
            raise ValueError("mode='rch' needs a fitted ResidualCalibrationHead")
        tt = (t_scalar if t_scalar is not None
              else torch.full((coords.shape[0],), 0.5, device=coords.device))
        C = rch(coords, feats, mask, tt, k)
        num = y - fval - C
        scale = (num / s ** 2).view(-1, 1, 1)
        G_c, G_f = _pullback(post_fn, coords, feats, scale * g_c, scale * g_f, cost)
        return G_c.detach(), G_f.detach(), {"f": fval, "c": C, "v_f": torch.zeros_like(fval),
                                            "k": k, "num": num,
                                            "den": torch.full_like(fval, s ** 2)}

    if mode == "band":
        # Component D. Two DIFFERENT vectors, which is the whole point:
        #   b = J^T grad f   the property direction, which the band constrains
        #   u = diversity    the direction we actually want to move in
        # An earlier version passed u = b, which makes the tangential component
        # v_r = r - (n'r)n identically ZERO and reduces the arm to plain
        # ascent along g -- the diversification was dead code.
        e = fval - y
        tau = (band_tau if torch.is_tensor(band_tau)
               else torch.full_like(fval, float(band_tau if band_tau is not None else 0.0)))
        b_c, b_f = _pullback(post_fn, coords, feats, g_c, g_f, cost)
        u_c, u_f = _diversity_direction(f_net, post_fn, coords, feats, mask,
                                        m_c, m_f, cost)
        d_c, d_f = tolerance_band_step(u_c, u_f, b_c, b_f, e, tau,
                                       band_eta, band_radius)
        return d_c.detach(), d_f.detach(), {"f": fval, "c": torch.zeros_like(fval),
                                            "v_f": torch.zeros_like(fval), "k": k,
                                            "band_e": e, "band_tau": tau}

    if mode.startswith("btvg2"):
        if tau is None:
            raise ValueError("mode %r needs tau (target property sd)" % mode)
        w_c, w_f, dg = btvg2_weighted_grad(
            f_net, post_fn, coords, feats, mask, m_c, m_f, k, y, s, tau, n_mc,
            generator, cost,
            gate=("band" if mode == "btvg2_band" else mode != "btvg2_nogate"),
            orth=(mode != "btvg2_noorth"),
            cap=(None if mode == "btvg2_nocap" else 1.0))
        G_c, G_f = _pullback(post_fn, coords, feats, w_c, w_f, cost)
        dg.update({"f": fval, "c": torch.zeros_like(fval),
                   "v_f": torch.zeros_like(fval), "k": k, "n_mc": n_mc})
        return G_c, G_f, dg

    if mode in ("lgd_mc", "osc"):
        w_c, w_f, F = sigma_mc_weighted_grad(
            f_net, post_fn, coords, feats, mask, m_c, m_f, k, y, s, n_mc,
            generator, cost, observable_closure=(mode == "osc"))
        G_c, G_f = _pullback(post_fn, coords, feats, w_c, w_f, cost)
        return G_c, G_f, {"f": fval, "c": torch.zeros_like(fval),
                          "v_f": torch.zeros_like(fval), "k": k, "n_mc": n_mc,
                          "obs_mean": F.mean(0), "obs_var": F.var(0, unbiased=True)
                          if F.shape[0] > 1 else torch.zeros_like(fval)}

    if mode == "tfg_mc":
        w_c, w_f, LP = tfg_mc_weighted_grad(f_net, m_c, m_f, mask, y, s,
                                            n_mc, sigma_mc, generator, cost)
        G_c, G_f = _pullback(post_fn, coords, feats, w_c, w_f, cost)
        return G_c, G_f, {"f": fval, "c": torch.zeros_like(fval),
                          "v_f": torch.zeros_like(fval), "k": k,
                          "n_mc": n_mc, "sigma_mc": sigma_mc,
                          "mc_logp_spread": (LP.max(0).values - LP.min(0).values)}

    SMG_MEAN = ("smg_mean", "smg", "smg2", "smg2_curv")
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
    return G_c.detach(), G_f.detach(), diag
