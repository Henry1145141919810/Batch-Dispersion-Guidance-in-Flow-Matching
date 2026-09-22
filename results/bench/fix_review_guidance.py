"""Adversarial-review fixes to guidance.py: D4, D10, D15. Run once.

D4  V_F = g'Sigma g comes out NEGATIVE on real states (measured: 1/16 samples
    at t=0.7, 3/16 at t=0.95). Sigma = k dm/dx is only symmetric when the
    network is an exact score; the measured asymmetry E|u'Jv - v'Ju|/rms runs
    0.045 -> 0.63 over t in [0.5, 0.8], so the quadratic form is not PSD.
    `V.clamp(min=1e-12)` then turned a negative V into a coefficient of
    +5.0e11 against a legitimate |b| ~ 8, i.e. the largest step the clip allows
    in the WIDENING direction, precisely when the variance model had broken
    down. Now: where V <= 0 the variance term is switched OFF.

D10 The variance coefficient is clamped to be <= 0, so BTVG can concentrate
    but never widen. Measured median V/tau^2 falls 2705 -> 0.0686 over
    t in [0.5, 0.975] -- V_F is a per-sample posterior variance that goes to 0
    as t -> 1 by construction, while tau^2 is a population band half-width, so
    the crossing is set by the noise schedule, not by the batch. Without the
    clamp the arm spends the last ~15% of every trajectory actively widening
    (measured b = +5.3, +110.8, +78.6). "Self-limiting" now means it stops,
    which is what the design intends and what the docstring may claim.

D15 SPBC recomputed f(m) that guidance_field had already computed, charging a
    second guide forward. That is a ~50% overstatement of SPBC's guide cost in
    a comparison whose whole point is matched compute.
"""
import io

NL = chr(10)          # written literally: shell heredocs here mangle backslashes

p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

# ---------------------------------------------------------------- D4 + D10
OLD = '''        V = V.clamp(min=1e-12)
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
            "btvg_V_over_tau2": (V / tau_t ** 2).detach()}'''
NEW = '''        # V_F CAN COME OUT NEGATIVE, and clamping it to +1e-12 is worse than
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
            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V_safe)).clamp(max=0.0)
            b = torch.where(V_ok, b, torch.zeros_like(b)).view(-1, 1, 1)
            G_c, G_f = G_c + b * dV_c, G_f + b * dV_f
        return G_c.detach(), G_f.detach(), {
            "f": fval, "c": torch.zeros_like(fval), "v_f": V, "k": k,
            "btvg_tau": tau_t.expand_as(fval).detach(),
            "btvg_V_raw": V.detach(),
            "btvg_V_nonpositive": (~V_ok).to(V.dtype).detach(),
            "btvg_V_over_tau2": (V / tau_t ** 2).detach()}'''
assert s.count(OLD) == 1, "btvg branch anchor"
s = s.replace(OLD, NEW)

# the docstring comment above the branch claimed unconditional self-limiting
OLD_C = '''        # V_F < tau^2 it stops rather than collapsing the batch.'''
NEW_C = '''        # V_F < tau^2 it stops rather than collapsing the batch. "Stops" is
        # enforced, not emergent: the coefficient is clamped at <= 0 below.'''
assert s.count(OLD_C) == 1, "comment anchor"
s = s.replace(OLD_C, NEW_C)

# ---------------------------------------------------------------- D15
OLD_S = '''def spbc_displacement(f_net, post_fn, coords, feats, mask, m_c, m_f, g_c, g_f,
                      y, eta=1.0, max_radius=None, cost=_NOCOST):'''
NEW_S = '''def spbc_displacement(f_net, post_fn, coords, feats, mask, m_c, m_f, g_c, g_f,
                      y, eta=1.0, max_radius=None, cost=_NOCOST, fval=None):'''
assert s.count(OLD_S) == 1, "spbc signature anchor"
s = s.replace(OLD_S, NEW_S)

OLD_F = '''    with torch.no_grad():
        z = f_net(m_c, m_f, mask)                                    # forecast
    cost.guide_fwd += 1
    mu = z.mean()'''
NEW_F = '''    if fval is None:
        with torch.no_grad():
            z = f_net(m_c, m_f, mask)                                # forecast
        cost.guide_fwd += 1
    else:
        z = fval          # guidance_field already paid for this one
    mu = z.mean()'''
assert s.count(OLD_F) == 1, "spbc forward anchor"
s = s.replace(OLD_F, NEW_F)

OLD_CALL = '''        d_c, d_f, dg = spbc_displacement(f_net, post_fn, coords, feats, mask,
                                         m_c, m_f, g_c, g_f, y, spbc_eta,
                                         spbc_radius, cost)'''
NEW_CALL = '''        d_c, d_f, dg = spbc_displacement(f_net, post_fn, coords, feats, mask,
                                         m_c, m_f, g_c, g_f, y, spbc_eta,
                                         spbc_radius, cost, fval=fval)'''
assert s.count(OLD_CALL) == 1, "spbc call anchor"
s = s.replace(OLD_CALL, NEW_CALL)

# ---------------------------------------------------------------- D16
# `features` hardcoded the class constant while load_rch builds the head with
# n_feat = beta.numel(). A head of any other width raised a shape error. The
# width now comes from the caller, defaulting to the class constant.
OLD_FEAT = """    @staticmethod
    def features(coords, feats, mask, t, k):"""
NEW_FEAT = """    @staticmethod
    def features(coords, feats, mask, t, k, n_feat=None):"""
assert s.count(OLD_FEAT) == 1, "features signature anchor"
s = s.replace(OLD_FEAT, NEW_FEAT)

OLD_WANT = "        want = ResidualCalibrationHead.N_FEAT"
NEW_WANT = NL.join([
    "        want = (ResidualCalibrationHead.N_FEAT if n_feat is None",
    "                else int(n_feat))"])
assert s.count(OLD_WANT) == 1, "want anchor"
s = s.replace(OLD_WANT, NEW_WANT)

OLD_FWD = """        phi = self.features(coords, feats, mask, t, k)"""
NEW_FWD = """        phi = self.features(coords, feats, mask, t, k, self.beta.numel())"""
assert s.count(OLD_FWD) == 1, "forward anchor"
s = s.replace(OLD_FWD, NEW_FWD)

# ---------------------------------------------------------------- D5 (docs)
OLD_H = '''    COST: one JVP (Sigma g), one HVP, one VJP, plus the reverse pass over the
    curvature scalar -- comparable to smg2_curv, and the HVP is the same
    construction _hvp already uses. `include_hessian=False` drops the first term
    and is kept only so the ablation can measure what it is worth; it is not a
    correct gradient and no arm should use it by default.'''
NEW_H = '''    THE 2x FACTOR ASSUMES J IS SYMMETRIC. Exactly, the gradient is
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
    correct gradient and no arm should use it by default.'''
assert s.count(OLD_H) == 1, "hessian doc anchor"
s = s.replace(OLD_H, NEW_H)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance.py: D4, D10, D15, D16, D5-docs applied")
