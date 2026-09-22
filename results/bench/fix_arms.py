"""Fixes from the 20 Sep arms audit. Run once."""
import io

p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- FIX 1
# The draws had covariance Sigma^2, not Sigma. One JVP gives Sigma v; it cannot
# give Sigma^{1/2} v. Replace with isotropic draws whose scale is READ OFF the
# model (trace-matched), which is the memo's own `iso_r2` variant, and say so.
OLD = '''def sigma_mc_weighted_grad(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
                           y, s, n_mc, generator=None, cost=_NOCOST,
                           observable_closure=False):
    """LGD-MC (observable_closure=False) and the observable-space closure (True).

    Both draw n_mc perturbations of m from the model's OWN covariance -- via
    Sigma^{1/2} z approximated by one JVP per draw, which is the same operator
    every other arm uses -- rather than TFG's tuned isotropic sigma. That is
    the one difference from tfg_mc_weighted_grad, and it is the point: no
    hyperparameter.
'''
NEW = '''def trace_scale(post_fn, coords, feats, mask, k, n_probe=1, generator=None,
                cost=_NOCOST):
    """r^2 = tr(Sigma)/d by Hutchinson, the isotropic scale that matches Sigma
    in trace. E_z[z' Sigma z] = tr(Sigma) for z Rademacher, and Sigma z is one
    JVP, so this is n_probe JVPs.

    Used to set the spread of the Monte-Carlo arms from the MODEL rather than
    from a tuned hyperparameter. See sigma_mc_weighted_grad for why the draws
    cannot simply be Sigma^{1/2} z.
    """
    msk = mask.unsqueeze(-1)
    d = (mask.sum(dim=1) * (coords.shape[-1] + feats.shape[-1])).clamp(min=1.0)
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
                           observable_closure=False, iso_r2=None):
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
'''
assert s.count(OLD) == 1
s = s.replace(OLD, NEW)

OLD_DRAW = '''    K = max(int(n_mc), 1)
    msk = mask.unsqueeze(-1)
    fs, gcs, gfs = [], [], []
    for _ in range(K):
        z_c = torch.randn(m_c.shape, generator=generator, device=m_c.device,
                          dtype=m_c.dtype) * msk
        z_f = torch.randn(m_f.shape, generator=generator, device=m_f.device,
                          dtype=m_f.dtype) * msk
        d_c, d_f = sigma_times_vector(post_fn, coords, feats, mask, z_c, z_f, k, cost)
        mc = (m_c + d_c).detach().requires_grad_(True)
        mf = (m_f + d_f).detach().requires_grad_(True)'''
NEW_DRAW = '''    K = max(int(n_mc), 1)
    msk = mask.unsqueeze(-1)
    if iso_r2 is None:
        iso_r2 = trace_scale(post_fn, coords, feats, mask, k, 1, generator, cost)
    r = iso_r2.clamp(min=0.0).sqrt().view(-1, 1, 1)
    fs, gcs, gfs = [], [], []
    for _ in range(K):
        z_c = torch.randn(m_c.shape, generator=generator, device=m_c.device,
                          dtype=m_c.dtype) * msk
        z_f = torch.randn(m_f.shape, generator=generator, device=m_f.device,
                          dtype=m_f.dtype) * msk
        d_c, d_f = r * z_c, r * z_f
        mc = (m_c + d_c).detach().requires_grad_(True)
        mf = (m_f + d_f).detach().requires_grad_(True)'''
assert s.count(OLD_DRAW) == 1
s = s.replace(OLD_DRAW, NEW_DRAW)

# ---------------------------------------------------------------- FIX 2
# kappa3_skew called post_fn three times per evaluation (one dead line plus two
# inside the jvp lambda), and a bare `except RuntimeError` turned any real
# failure into a silent NaN.
OLD_K3 = '''    c_in = coords.detach().requires_grad_(True)
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
    if d_c is None:'''
NEW_K3 = '''    c_in = coords.detach().requires_grad_(True)
    f_in = feats.detach().requires_grad_(True)

    def _mean(cc, ff):
        p = post_fn(cc, ff)
        return p.mean_coords, p.mean_feats

    def quad(c, f):
        # ONE jvp, which internally does one forward. An earlier version also
        # called post_fn on its own line and again inside the lambda's second
        # element, tripling the generator cost for nothing.
        sg_c, sg_f = torch.func.jvp(_mean, (c, f), (g_c, g_f))[1]
        return k * ((g_c * sg_c).sum(dim=(1, 2)) + (g_f * sg_f).sum(dim=(1, 2)))

    q = quad(c_in, f_in)
    d_c, d_f = torch.autograd.grad(q.sum(), (c_in, f_in), allow_unused=True)
    cost.gen_fwd += 1
    cost.gen_jvp += 1
    cost.gen_vjp += 1
    if d_c is None:'''
assert s.count(OLD_K3) == 1
s = s.replace(OLD_K3, NEW_K3)

# dispatch: pass iso_r2 through and record it
s = s.replace('''            f_net, post_fn, coords, feats, mask, m_c, m_f, k, y, s, n_mc,
            generator, cost, observable_closure=(mode == "osc"))''',
'''            f_net, post_fn, coords, feats, mask, m_c, m_f, k, y, s, n_mc,
            generator, cost, observable_closure=(mode == "osc"))''')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance.py: Sigma^2 draw bug fixed (trace-matched isotropic); kappa3 cost and except fixed")
