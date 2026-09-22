"""Exact gates for the v2 arms: SPBC, BTVG, and the SHG schedule.

Same discipline as test_arms_exact.py: a diagonal, analytically known setting
so every quantity has a closed form and a wrong sign or dropped factor moves a
number. An earlier gate file passed seven injected defects; these are written
to fail under mutation (results/bench/mutation_test.py covers them).

  posterior   m = D * x elementwise  =>  J = diag(D),  Sigma = k diag(D)
  property    f(m) = 1/2 sum a_i m_i^2 + sum b_i m_i  =>  H = diag(a)

Closed forms used:
  a_i (response)   = J^T grad f = D * g
  r_i              = ||D * g||^2
  V_F = g' Sigma g = k sum D_j g_j^2
  grad_x V_F       = 2 k D^2 g * (a*D*... ) -- verified by autograd reference
  SPBC edit        = (nu / r) * (D * g),  nu common across the batch
  BTVG mean term   = -(f(m) - y)/tau^2 * J^T g
  BTVG var term    = -1/2 (1/tau^2 - 1/V_F) * grad_x V_F

Run: python proj1/tests/test_v2_arms.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from guidance import (DISPLACEMENT_MODES, KNOWN_MODES, Posterior,  # noqa: E402
                      fm_posterior, grad_property_variance, guidance_field,
                      spbc_displacement)
from models.egnn import zero_com  # noqa: E402
from sampling import FlowSampler  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from guidance_sweep import SHG_SCHEDULES  # noqa: E402

torch.set_default_dtype(torch.float64)
R = {}
B, N, K = 4, 3, 2


class Quad(torch.nn.Module):
    def __init__(self, a_c, a_f, b_c, b_f):
        super().__init__()
        self.a_c, self.a_f, self.b_c, self.b_f = a_c, a_f, b_c, b_f

    def forward(self, coords, feats, mask):
        return (0.5 * (self.a_c * coords ** 2).sum(dim=(1, 2))
                + 0.5 * (self.a_f * feats ** 2).sum(dim=(1, 2))
                + (self.b_c * coords).sum(dim=(1, 2))
                + (self.b_f * feats).sum(dim=(1, 2)))


def setup(seed=7):
    torch.manual_seed(seed)
    mask = torch.ones(B, N)
    coords, feats = torch.randn(B, N, 3), torch.randn(B, N, K)
    D_c, D_f = torch.rand(B, N, 3) + 0.5, torch.rand(B, N, K) + 0.5
    a_c, a_f = torch.rand(B, N, 3) + 0.2, torch.rand(B, N, K) + 0.2
    b_c, b_f = torch.randn(B, N, 3), torch.randn(B, N, K)
    k = torch.full((B,), 0.8)
    f_net = Quad(a_c, a_f, b_c, b_f)

    def post_fn(c, f):
        return Posterior(D_c * c, D_f * f, k)

    m_c, m_f = D_c * coords, D_f * feats
    g_c, g_f = a_c * m_c + b_c, a_f * m_f + b_f       # grad f at m
    return dict(mask=mask, coords=coords, feats=feats, D_c=D_c, D_f=D_f,
                a_c=a_c, a_f=a_f, k=k, f_net=f_net, post_fn=post_fn,
                m_c=m_c, m_f=m_f, g_c=g_c, g_f=g_f)


def main():
    E = setup()
    mask, coords, feats = E["mask"], E["coords"], E["feats"]
    D_c, D_f, k = E["D_c"], E["D_f"], E["k"]
    f_net, post_fn = E["f_net"], E["post_fn"]
    m_c, m_f, g_c, g_f = E["m_c"], E["m_f"], E["g_c"], E["g_f"]
    kb = k.view(-1, 1, 1)

    # ---------------- V_F and its gradient, against autograd
    V_got, dV_c, dV_f = grad_property_variance(f_net, post_fn, coords, feats,
                                               mask, m_c, m_f, k, g_c, g_f)
    V_exact = k * ((D_c * g_c ** 2).sum(dim=(1, 2)) + (D_f * g_f ** 2).sum(dim=(1, 2)))
    R["V_F_exact"] = (V_got - V_exact).abs().max().item()

    ci = coords.clone().requires_grad_(True)
    fi = feats.clone().requires_grad_(True)
    Vref = (k * ((D_c * (E["a_c"] * (D_c * ci) + f_net.b_c) ** 2).sum(dim=(1, 2))
                 + (D_f * (E["a_f"] * (D_f * fi) + f_net.b_f) ** 2).sum(dim=(1, 2))))
    rc, rf = torch.autograd.grad(Vref.sum(), (ci, fi))
    R["grad_V_F_vs_autograd"] = max((dV_c - rc).abs().max().item(),
                                    (dV_f - rf).abs().max().item())

    # ---------------- the mean-map curvature term, on a NONLINEAR mean map
    # grad_x V has two terms and the diagonal fixture above kills one of them:
    # with m = D*x the Jacobian is constant, so k*grad(g'J(x)g) == 0 and
    # nothing constrains it. Here m = D*x + lam*x^3 elementwise, which keeps J
    # diagonal (hence symmetric, so the 2x Hessian factor stays exact) while
    # making the curvature term live.
    lam = 0.4
    Dc3, Df3 = E["D_c"], E["D_f"]
    k3 = E["k"]

    def post3(c, f):
        return Posterior(Dc3 * c + lam * c ** 3, Df3 * f + lam * f ** 3, k3)

    m3_c = Dc3 * coords + lam * coords ** 3
    m3_f = Df3 * feats + lam * feats ** 3
    g3_c, g3_f = E["a_c"] * m3_c + f_net.b_c, E["a_f"] * m3_f + f_net.b_f
    V3, d3_c, d3_f = grad_property_variance(f_net, post3, coords, feats, mask,
                                            m3_c, m3_f, k3, g3_c, g3_f)

    c3 = coords.clone().requires_grad_(True)
    f3 = feats.clone().requires_grad_(True)
    mm_c = Dc3 * c3 + lam * c3 ** 3
    mm_f = Df3 * f3 + lam * f3 ** 3
    gg_c = E["a_c"] * mm_c + f_net.b_c
    gg_f = E["a_f"] * mm_f + f_net.b_f
    Jc = Dc3 + 3.0 * lam * c3 ** 2            # diagonal of dm/dx
    Jf = Df3 + 3.0 * lam * f3 ** 2
    Vref3 = k3 * ((Jc * gg_c ** 2).sum(dim=(1, 2))
                  + (Jf * gg_f ** 2).sum(dim=(1, 2)))
    r3c, r3f = torch.autograd.grad(Vref3.sum(), (c3, f3))
    R["V_F_nonlinear_exact"] = (V3 - Vref3.detach()).abs().max().item()
    R["grad_V_F_nonlinear_vs_autograd"] = max(
        (d3_c - r3c).abs().max().item(), (d3_f - r3f).abs().max().item())

    # ...and the curvature term must actually be CARRYING weight here, or this
    # gate is as vacuous as the linear one it replaces.
    _V3b, h3_c, h3_f = grad_property_variance(f_net, post3, coords, feats, mask,
                                              m3_c, m3_f, k3, g3_c, g3_f,
                                              include_hessian=False)
    curv = torch.sqrt((h3_c ** 2).sum() + (h3_f ** 2).sum())
    full = torch.sqrt((d3_c ** 2).sum() + (d3_f ** 2).sum())
    R["curvature_term_is_live"] = 0.0 if float(curv / full.clamp(min=1e-30)) > 0.05 else 1.0

    # ---------------- SPBC gives EVERY trajectory the same property increment
    y = 1.0
    d_c, d_f, dg = spbc_displacement(f_net, post_fn, coords, feats, mask,
                                     m_c, m_f, g_c, g_f, y, eta=1.0)
    a_resp_c, a_resp_f = D_c * g_c, D_f * g_f                    # a_i = J^T g
    nu = (a_resp_c * d_c).sum(dim=(1, 2)) + (a_resp_f * d_f).sum(dim=(1, 2))
    R["spbc_increment_is_common"] = (nu - nu.mean()).abs().max().item()

    with torch.no_grad():
        z = f_net(m_c, m_f, mask)
    R["spbc_increment_equals_minus_bias"] = (nu.mean() - (-(z.mean() - y))).abs().item()

    # minimum-energy: d_i must be parallel to a_i
    cosang = ((a_resp_c * d_c).sum(dim=(1, 2)) + (a_resp_f * d_f).sum(dim=(1, 2))) / (
        (a_resp_c ** 2).sum(dim=(1, 2)) + (a_resp_f ** 2).sum(dim=(1, 2))).sqrt().clamp(min=1e-30) / (
        (d_c ** 2).sum(dim=(1, 2)) + (d_f ** 2).sum(dim=(1, 2))).sqrt().clamp(min=1e-30)
    R["spbc_edit_is_min_energy"] = (cosang.abs() - 1.0).abs().max().item()

    # a zero-response trajectory must get a zero edit, not a blow-up
    E2 = setup(11)
    zg_c, zg_f = torch.zeros_like(E2["g_c"]), torch.zeros_like(E2["g_f"])
    zd_c, zd_f, _ = spbc_displacement(E2["f_net"], E2["post_fn"], E2["coords"],
                                      E2["feats"], E2["mask"], E2["m_c"],
                                      E2["m_f"], zg_c, zg_f, y)
    R["spbc_zero_response_zero_edit"] = max(zd_c.abs().max().item(),
                                            zd_f.abs().max().item())

    # The dangerous case is not r == 0 (then a_i == 0 and the edit vanishes on
    # its own) but r SMALL AND NONZERO: nu/r explodes while a_i is merely tiny,
    # so the product is large. The guard must refuse these outright.
    tiny = 1e-8
    td_c, td_f, _ = spbc_displacement(E2["f_net"], E2["post_fn"], E2["coords"],
                                      E2["feats"], E2["mask"], E2["m_c"],
                                      E2["m_f"], tiny * E2["g_c"],
                                      tiny * E2["g_f"], y)
    R["spbc_tiny_response_is_refused"] = max(td_c.abs().max().item(),
                                             td_f.abs().max().item())

    # the trust radius scales the WHOLE batch, never per sample
    b_c, b_f, _ = spbc_displacement(f_net, post_fn, coords, feats, mask, m_c,
                                    m_f, g_c, g_f, y, eta=1.0, max_radius=1e-4)
    nub = (a_resp_c * b_c).sum(dim=(1, 2)) + (a_resp_f * b_f).sum(dim=(1, 2))
    R["spbc_bounded_still_common"] = (nub - nub.mean()).abs().max().item()
    R["spbc_bound_respected"] = max(
        0.0, ((b_c ** 2).sum(dim=(1, 2)) + (b_f ** 2).sum(dim=(1, 2))).sqrt().max().item() - 1e-4 - 1e-12)

    # ---------------- BTVG against its closed form
    tau = 0.37
    fval = f_net(m_c, m_f, mask)
    ref_mean_c = (-(fval - y) / tau ** 2).view(-1, 1, 1) * (D_c * g_c)
    ref_mean_f = (-(fval - y) / tau ** 2).view(-1, 1, 1) * (D_f * g_f)
    # the implementation clamps this coefficient at <= 0 (it may concentrate,
    # never widen); the fixture sits well above tau^2, so the clamp is inactive
    # here and the reference is the raw coefficient. Asserted, not assumed.
    assert bool((V_exact > tau ** 2).all()), "fixture must sit above tau^2"
    bcoef = (-0.5 * (1.0 / tau ** 2 - 1.0 / V_exact)).view(-1, 1, 1)
    assert bool((bcoef <= 0).all()), "clamp must be inactive in this fixture"
    ref_var_c, ref_var_f = bcoef * rc, bcoef * rf

    for mode, (rc_, rf_) in (("btvg_mean", (ref_mean_c, ref_mean_f)),
                             ("btvg_var", (ref_var_c, ref_var_f)),
                             ("btvg", (ref_mean_c + ref_var_c, ref_mean_f + ref_var_f))):
        G_c, G_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, y,
                                     0.5, mode=mode, tau=tau)
        R["field_%s_exact" % mode] = max((G_c - rc_).abs().max().item(),
                                         (G_f - rf_).abs().max().item())

    # the decisive property: the variance term's SIGN follows V_F vs tau^2,
    # never the residual. Far-from-target must still CONTRACT.
    far = 1e4                                   # residual enormous
    Gf_c, Gf_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, far, 0.5,
                                   mode="btvg_var", tau=tau)
    Gn_c, Gn_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, y, 0.5,
                                   mode="btvg_var", tau=tau)
    R["btvg_var_independent_of_residual"] = max(
        (Gf_c - Gn_c).abs().max().item(), (Gf_f - Gn_f).abs().max().item())

    # ...and it points STRICTLY DOWNHILL in V_F whenever V_F > tau^2, even at
    # that enormous residual -- the property `plug` does not have. The margin
    # is required to be real, so a zero field (the defect the gradient fix
    # repaired) fails here instead of passing vacuously.
    over = (V_exact > tau ** 2)
    assert bool(over.any()), "fixture must put some samples above tau^2"
    dirn = (Gf_c * rc).sum(dim=(1, 2)) + (Gf_f * rf).sum(dim=(1, 2))
    R["btvg_var_strictly_reduces_V"] = 0.0 if bool(
        (dirn[over] < -1e-6).all()) else 1.0
    # BELOW tau^2 THE ARM MUST STOP, NOT REVERSE. The previous version of this
    # check short-circuited on an empty `under` set and so tested nothing. The
    # fixture cannot produce V < tau^2 at a sane tau, so it is produced here by
    # raising tau instead -- and guarded, so it fails if that stops working.
    tau_big = float(V_exact.max().sqrt() * 10.0)
    V_over_big = V_exact / tau_big ** 2
    assert bool((V_over_big < 1.0).all()), "fixture must sit below tau_big^2"
    Gb_c, Gb_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, y, 0.5,
                                   mode="btvg_var", tau=tau_big)
    R["btvg_var_stops_below_tau"] = max(Gb_c.abs().max().item(),
                                        Gb_f.abs().max().item())

    # D4: a NON-POSITIVE V_F means the variance model has broken down. Sigma is
    # only symmetric for an exact score, and on the real generator V came out
    # <= 0 on up to 3/16 samples. Clamping V to +1e-12 turned those into a
    # coefficient of +5e11 -- the largest widening step the clip allows,
    # exactly where the model was least trustworthy. It must now contribute
    # nothing. Forced here by a property whose gradient at m is zero, which
    # makes V identically 0.
    class _Flat(torch.nn.Module):
        def forward(self, c, f, mask):
            return (0.0 * c).sum(dim=(1, 2)) + (0.0 * f).sum(dim=(1, 2))

    Gz_c, Gz_f, dz = guidance_field(_Flat(), post_fn, coords, feats, mask, y,
                                    0.5, mode="btvg_var", tau=tau)
    assert bool((dz["btvg_V_raw"] <= 0).all()), "fixture must force V <= 0"
    R["btvg_var_off_when_V_nonpositive"] = max(Gz_c.abs().max().item(),
                                               Gz_f.abs().max().item())
    R["btvg_reports_V_nonpositive"] = abs(
        float(dz["btvg_V_nonpositive"].mean()) - 1.0)

    # tau is required, and an unknown mode is refused
    try:
        guidance_field(f_net, post_fn, coords, feats, mask, y, 0.5, mode="btvg")
        R["btvg_requires_tau"] = 1.0
    except ValueError:
        R["btvg_requires_tau"] = 0.0

    # ---------------- SHG: the schedule must switch on time AND scale with w
    class _Stub(object):
        """The real `active`, bound to a bare attribute bag. Constructing a
        FlowSampler needs a network; the schedule logic needs none, and testing
        the real function rather than a copy of it is the point."""
        active = FlowSampler.active

        def __init__(self, **kw):
            self.mode, self.w = "plug", 1.0
            self.schedule, self.schedule_log = None, {}
            for k, v in kw.items():
                setattr(self, k, v)

    sched = [(0.50, 0.85, "plug", 1.0), (0.85, 1.00, "spbc", 0.5)]
    sm = _Stub(schedule=sched, w=2.0)
    got = [sm.active(t) for t in (0.10, 0.50, 0.84, 0.85, 0.99)]
    want = [(None, 0.0), ("plug", 2.0), ("plug", 2.0), ("spbc", 1.0),
            ("spbc", 1.0)]
    R["shg_selects_mode_and_weight"] = max(
        abs(g[1] - w[1]) + (0.0 if g[0] == w[0] else 1.0)
        for g, w in zip(got, want))

    # the decisive one: doubling w must double the weight at every time. The
    # screening sweep produced five identical cells at w = 0.01 ... 4 because
    # this was false.
    s1 = _Stub(schedule=sched, w=1.0)
    s2 = _Stub(schedule=sched, w=2.0)
    R["shg_honours_strength_knob"] = max(
        abs(2.0 * s1.active(t)[1] - s2.active(t)[1]) for t in (0.6, 0.9))

    # no schedule => the plain (mode, w) pair, unchanged
    s0 = _Stub(schedule=None, mode="smg", w=3.0)
    R["shg_absent_is_passthrough"] = (
        0.0 if s0.active(0.7) == ("smg", 3.0) else 1.0)

    # every mode a schedule can name must be a real mode, or `active` can hand
    # control to something guidance_field will reject mid-sample
    bad = [m for sc in SHG_SCHEDULES.values() for (_lo, _hi, m, _w) in sc
           if m not in KNOWN_MODES]
    R["shg_schedules_name_real_modes"] = float(len(bad))

    # ---------------- the units, measured through the REAL sampler
    # A displacement arm must NOT be put through the score-to-velocity factor
    # (1-t)/t, and a score arm must be. Set membership alone does not test
    # that the sampler acts on it, so this drives FlowSampler.field and reads
    # off the multiplier actually applied. t = 0.8 makes (1-t)/t = 0.25, so
    # the two conventions cannot be confused for one another.
    t_u, w_u = 0.8, 1.7
    conv = (1.0 - t_u) / t_u

    class _Net(torch.nn.Module):
        """Linear velocity field: m = x + (1-t) v is then affine in x, so the
        posterior is exactly the diagonal setting the other gates use."""
        def __init__(self, ac, af):
            super().__init__()
            self.ac, self.af = ac, af

        def forward(self, c, f, mask, t):
            return self.ac * c, self.af * f

    E3 = setup(23)
    net = _Net(0.3 * torch.ones_like(E3["coords"]),
               0.2 * torch.ones_like(E3["feats"]))
    tvec = torch.full((B,), t_u)

    def _post(c, f):
        return fm_posterior(net, c, f, E3["mask"], tvec)

    for mode, expect in (("spbc", w_u), ("plug", w_u * conv)):
        kw = dict(f_net=E3["f_net"], y=1.0, s=1.0, mode=mode, w=w_u,
                  clip=None, spbc_radius=None)
        smp = FlowSampler(net, E3["mask"], t_min_guide=0.0, **kw)
        v_c, v_f = net(E3["coords"], E3["feats"], E3["mask"], tvec)
        out_c, _out_f = smp.field(E3["coords"], E3["feats"], t_u)
        applied_c = out_c - v_c
        raw_c, _raw_f, _d = guidance_field(
            E3["f_net"], _post, E3["coords"], E3["feats"], E3["mask"], 1.0,
            1.0, mode=mode, spbc_radius=None)
        # _guide projects the coordinate block to zero COM and masks the
        # features before the multiplier is applied; the reference must do the
        # same or this measures the projection instead of the multiplier.
        raw_c = zero_com(raw_c, E3["mask"])
        denom = raw_c.abs().max().clamp(min=1e-30)
        R["units_%s_multiplier" % mode] = (
            (applied_c - expect * raw_c).abs().max() / denom).item()

    # ---------------- SPBC survives the zero-COM projection iff f is
    # translation invariant. The sampler projects the coordinate block after
    # the edit is built, and projection changes a_i . d_i in general -- so the
    # common increment, which is the entire claim, depends on this.
    class _TransInv(torch.nn.Module):
        """Depends on coordinates only through pairwise differences, so it is
        translation invariant by construction -- like every physical property
        here and like the EGNN predictor."""
        def __init__(self, w_c, a_f, b_f):
            super().__init__()
            self.w_c, self.a_f, self.b_f = w_c, a_f, b_f

        def forward(self, coords, feats, mask):
            d = coords.unsqueeze(2) - coords.unsqueeze(1)       # [B,N,N,3]
            r2 = (d ** 2).sum(-1)
            return ((self.w_c * r2).sum(dim=(1, 2))
                    + 0.5 * (self.a_f * feats ** 2).sum(dim=(1, 2))
                    + (self.b_f * feats).sum(dim=(1, 2)))

    # THE PRECONDITION, STATED. SPBC's edit is parallel to a_i = J^T g. For the
    # projection to be a no-op, a_i must already have zero COM, which needs
    # BOTH a translation-invariant property (so g has zero COM) AND a posterior
    # whose Jacobian maps the zero-COM subspace to itself (so J^T keeps it
    # there). The real generator satisfies the second: m = x + (1-t)v with an
    # EGNN velocity that is zero-COM by construction. The elementwise-diagonal
    # posterior the other gates use does NOT -- D * g leaves the subspace --
    # so this gate builds a COM-preserving posterior instead of pretending the
    # claim is unconditional.
    E4 = setup(31)
    a_scalar, k4 = 0.7, E4["k"]

    def post4(c, f):
        return Posterior(a_scalar * zero_com(c, E4["mask"]), E4["D_f"] * f, k4)

    E4["post_fn"] = post4
    E4["m_c"] = a_scalar * zero_com(E4["coords"], E4["mask"])
    ti = _TransInv(torch.rand(B, N, N) + 0.2, E4["a_f"], f_net.b_f)
    mi_c = E4["m_c"].detach().requires_grad_(True)
    mi_f = E4["m_f"].detach().requires_grad_(True)
    gi_c, gi_f = torch.autograd.grad(ti(mi_c, mi_f, E4["mask"]).sum(),
                                     (mi_c, mi_f))
    di_c, di_f, _ = spbc_displacement(ti, E4["post_fn"], E4["coords"],
                                      E4["feats"], E4["mask"], E4["m_c"],
                                      E4["m_f"], gi_c, gi_f, y)
    ai_c, ai_f = a_scalar * zero_com(gi_c, E4["mask"]), E4["D_f"] * gi_f
    nu_raw = (ai_c * di_c).sum(dim=(1, 2)) + (ai_f * di_f).sum(dim=(1, 2))
    pj_c = zero_com(di_c, E4["mask"])
    nu_prj = (ai_c * pj_c).sum(dim=(1, 2)) + (ai_f * di_f).sum(dim=(1, 2))
    R["spbc_survives_zero_com_when_invariant"] = (nu_prj - nu_raw).abs().max().item()
    R["spbc_increment_common_after_projection"] = (
        nu_prj - nu_prj.mean()).abs().max().item()

    # ...and the converse: for a property that is NOT translation invariant the
    # projection does move the increment. If this is ever ~0, the fixture has
    # stopped testing anything.
    d2_c, d2_f, _ = spbc_displacement(f_net, post_fn, coords, feats, mask, m_c,
                                      m_f, g_c, g_f, y)
    n2 = ((a_resp_c * d2_c).sum(dim=(1, 2))
          + (a_resp_f * d2_f).sum(dim=(1, 2)))
    n2p = ((a_resp_c * zero_com(d2_c, mask)).sum(dim=(1, 2))
           + (a_resp_f * d2_f).sum(dim=(1, 2)))
    R["noninvariant_property_IS_disturbed"] = (
        0.0 if (n2 - n2p).abs().max().item() > 1e-6 else 1.0)

    # ---------------- SHG bookkeeping
    R["spbc_is_displacement_mode"] = 0.0 if "spbc" in DISPLACEMENT_MODES else 1.0
    R["btvg_is_not_displacement"] = 0.0 if "btvg" not in DISPLACEMENT_MODES else 1.0

    tol = 1e-9
    print("%-38s %13s   pass" % ("check", "max abs error"))
    print("-" * 62)
    ok = True
    for kk, v in R.items():
        p = v < tol
        ok = ok and p
        print("%-38s %13.3e   %s" % (kk, v, "yes" if p else "NO"))
    print("-" * 62)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
