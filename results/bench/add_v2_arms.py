"""Add the three selected v2 guidance arms. Run once.

Selected on measured evidence, after the corrections in
Three_New_Guidance_Ideas_Variance_and_Switching.md:

  SPBC  shape-preserving bias correction   -> attacks bias, measured up to 13.99 delta
  BTVG  band-targeted variance guidance    -> attacks spread, measured at 4-18 delta
  SHG   scheduled handoff (sampler-level)  -> window effect, measured 47% on alpha

Dropped: RWG (subsumed by SHG, narrow novelty), RBAG (HIG prior art too close),
OSHG's adaptive order gate (expensive; the fixed handoff is SHG).
"""
import io

p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

NEW = '''

def grad_property_variance(f_net, post_fn, coords, feats, mask, g_c, g_f, k,
                           cost=_NOCOST):
    """(V_F, grad_x V_F) with V_F = g' Sigma g, the property's variance at x_t.

    V_F is already computed by every SMG-family arm and used only as a
    denominator. Its GRADIENT is what a variance-targeting objective needs, and
    it is one reverse pass over the scalar -- the same construction
    kappa3_skew uses for the third cumulant.

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
    return V.detach(), d_c, d_f


def spbc_displacement(f_net, post_fn, coords, feats, mask, m_c, m_f, g_c, g_f,
                      y, eta=1.0, max_radius=None, cost=_NOCOST):
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

    HONEST LIMITS. The forecast is the cheap one, f_A(m_t), not a rollout
    through the sampler suffix; it inherits the off-distribution error measured
    in FINDING_QUADRATIC_CLOSURE_VALIDITY.md. Preserving scalar property
    spacings does NOT imply preserving structural diversity. And centering is
    not guaranteed to raise band coverage -- a skewed law can lose coverage
    when centred (memo section 3.5), so it needs an acceptance check.
    """
    a_c, a_f = _pullback(post_fn, coords, feats, g_c, g_f, cost)     # a_i = J^T g
    r = (a_c ** 2).sum(dim=(1, 2)) + (a_f ** 2).sum(dim=(1, 2))      # r_i = ||a_i||^2
    with torch.no_grad():
        z = f_net(m_c, m_f, mask)                                    # forecast
    cost.guide_fwd += 1
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
'''

anchor = "\n\n# --------------------------------------------------------------------------\n# the fields\n# --------------------------------------------------------------------------"
assert s.count(anchor) == 1
s = s.replace(anchor, NEW + anchor)

# ---- register the new modes
s = s.replace('KNOWN_MODES = {"plug", "smg_mean", "smg_var", "smg", "smg2", "smg2_curv",\n'
              '               "tfg_mc", "lgd_mc", "osc", "rch", "band"}',
              'KNOWN_MODES = {"plug", "smg_mean", "smg_var", "smg", "smg2", "smg2_curv",\n'
              '               "tfg_mc", "lgd_mc", "osc", "rch", "band",\n'
              '               # v2 arms, see PROPOSALS_*.md and the selection memo\n'
              '               "spbc", "btvg", "btvg_mean", "btvg_var"}\n'
              '\n'
              '# Arms that return a STATE DISPLACEMENT rather than a score. The sampler\n'
              '# must not apply the (1-t)/t score-to-velocity conversion to these.\n'
              'DISPLACEMENT_MODES = {"spbc"}')

s = s.replace('''                   mode="plug", n_probe=1, generator=None, cost=_NOCOST,
                   n_mc=4, sigma_mc=0.1, want_kappa3=False, rch=None,
                   band_tau=None, band_eta=1.0, band_radius=None, t_scalar=None):''',
'''                   mode="plug", n_probe=1, generator=None, cost=_NOCOST,
                   n_mc=4, sigma_mc=0.1, want_kappa3=False, rch=None,
                   band_tau=None, band_eta=1.0, band_radius=None, t_scalar=None,
                   tau=None, spbc_eta=1.0, spbc_radius=None):''')

s = s.replace('''    if mode == "rch":''',
'''    if mode == "spbc":
        d_c, d_f, dg = spbc_displacement(f_net, post_fn, coords, feats, mask,
                                         m_c, m_f, g_c, g_f, y, spbc_eta,
                                         spbc_radius, cost)
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
        # V_F < tau^2 it stops rather than collapsing the batch.
        #
        # tau IS NOT delta. A centred Gaussian with sd = delta gives only
        # 68.3% band coverage; 95% needs sd ~ delta/1.96. The caller passes
        # tau explicitly and the default in the sweep is delta/1.96.
        if tau is None:
            raise ValueError("mode %r needs tau (target property sd)" % mode)
        tau_t = torch.as_tensor(tau, device=coords.device, dtype=coords.dtype)
        V, dV_c, dV_f = grad_property_variance(f_net, post_fn, coords, feats,
                                               mask, g_c, g_f, k, cost)
        V = V.clamp(min=1e-12)
        G_c = torch.zeros_like(coords)
        G_f = torch.zeros_like(feats)
        if mode in ("btvg", "btvg_mean"):
            a = (-(fval - y) / tau_t ** 2).view(-1, 1, 1)
            mc, mf = _pullback(post_fn, coords, feats, a * g_c, a * g_f, cost)
            G_c, G_f = G_c + mc, G_f + mf
        if mode in ("btvg", "btvg_var"):
            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V)).view(-1, 1, 1)
            G_c, G_f = G_c + b * dV_c, G_f + b * dV_f
        return G_c.detach(), G_f.detach(), {
            "f": fval, "c": torch.zeros_like(fval), "v_f": V, "k": k,
            "btvg_tau": tau_t.expand_as(fval).detach(),
            "btvg_V_over_tau2": (V / tau_t ** 2).detach()}

    if mode == "rch":''')

s = s.replace('''      "band"        Component D, the tolerance band: the largest diversifying
                    edit whose first-order property change stays in the band''',
'''      "band"        Component D, the tolerance band: the largest diversifying
                    edit whose first-order property change stays in the band
      "spbc"        shape-preserving bias correction. Gives EVERY trajectory the
                    same property increment, which leaves all centered quantiles
                    and pairwise differences intact. Returns a DISPLACEMENT.
      "btvg"        band-targeted variance guidance: descend KL to N(y*, tau^2)
                    rather than maximise the likelihood of y*. Needs tau.
      "btvg_mean"   the mean half alone   (ablation)
      "btvg_var"    the variance half alone (ablation -- the decisive one)''')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance.py: added spbc, btvg, btvg_mean, btvg_var + grad_property_variance")
