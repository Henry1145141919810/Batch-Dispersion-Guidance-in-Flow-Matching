"""Exact gates for the `tfg` arm (TFG, Ye et al. NeurIPS 2024, N_recur = 1).

Every gate compares against an INDEPENDENT reference -- TFG's own code copied
verbatim, autograd of the declared objective, or TFG's VP update written out
by hand -- so a dropped factor, a flipped sign or a wrong time index moves a
number. Toy models (linear velocity / epsilon field, quadratic property) keep
every quantity exact in float64.

  T1  rescale_grad == TFG's tasks/utils.py:rescale_grad, bound and unbound,
      with padded atoms
  T2  G  == autograd of log mean_k exp(logf(m(x) + eps_k)) w.r.t. x, rescaled
      with var_scale (K = 1 and K = 3, draws replayed)
  T3  D0 == a hand-written n_iter loop with the SAME draws and per-iteration
      rescale
  T4  std = 0 draws nothing (generator untouched) and uses one zero eps
  T5  std = 0, no clip: G == (2 s^2 / mad^2) x plug's field -- TFG's
      variance step IS the plug-in direction, only rescaled
  T6  cost counters; mu_step = 0 skips the loop; want_var False skips the VJP
  T7  schedules: sum_i rho_i = rho_bar * T, 'increase' is proportional to
      alpha_i, std_i = gamma_bar sqrt(1 - abar_i); flow abar = t^2/c_t^2
  T8  flow geometry == TFG's VP update mapped through x_vp = x / c_t, and the
      DDIM base step == our Euler step (both to 1e-15)
  T9  FlowSampler one step == v + (w rho_i k_var G + t1 D0(w mu_i)) / h
  T10 VPSampler   one step == base + (w rho_i G / sqrt(alpha) +
                             sqrt(abar_next) D0) / h
  T11 w = 0 reproduces the unguided trajectory bit for bit
  T12 padded atoms get no correction; the coordinate correction is zero-CoM
  T13 refusals: Heun stage 2, a field call outside integrate, a missing
      config, a non-euler update rule, guidance_field(mode='tfg')
  T14 determinism, and a tfg run does not move the initial-noise stream
  T15 sweep wiring: tfg replaces dflow in COMPARE_SET, Table 11 values,
      w not rescaled, check_fullrun_go agrees, 42 compare cells
  T16 the flow var_scale reaches rescale_grad: a clip that binds on c_t G but
      not on G does bind

Run: python proj1/tests/test_tfg.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from guidance import (Cost, Posterior, fm_posterior, guidance_field,  # noqa: E402
                      tfg_components, tfg_rescale_grad)
from models.egnn import zero_com  # noqa: E402
from sampling import FlowSampler, VPSampler, integrate, initial_noise  # noqa: E402

torch.set_default_dtype(torch.float64)
R = {}
B, N, T = 4, 5, 3          # T = atom-type channels
TOL = 1e-10


# ---------------------------------------------------------------- fixtures
class Net(torch.nn.Module):
    """Linear, time-dependent, zero-CoM field. Serves as a flow velocity and,
    in the VP gates, as an epsilon prediction."""

    def __init__(self, seed=5):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.randn(1, generator=g) * 0.3
        self.W = torch.randn(T, T, generator=g) * 0.3

    def forward(self, c, f, mask, t):
        tb = t.view(-1, 1, 1)
        m3 = mask.unsqueeze(-1)
        v_c = self.a * c * (1.0 + tb)
        v_f = f @ self.W * (1.0 - 0.5 * tb)
        return zero_com(v_c * m3, mask), v_f * m3


class Quad(torch.nn.Module):
    """A nonlinear property, masked, translation-invariant in coords."""

    def __init__(self, seed=9):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.rand(T, generator=g) + 0.3
        self.b = torch.randn(T, generator=g)

    def forward(self, c, f, mask):
        m3 = mask.unsqueeze(-1)
        cc = zero_com(c * m3, mask)
        return (0.2 * (cc ** 2).sum((1, 2)) + 0.5 * ((f * m3) ** 2 * self.a).sum((1, 2))
                + (f * m3 * self.b).sum((1, 2)))


def make_mask():
    mask = torch.ones(B, N)
    mask[0, -1] = 0.0
    mask[2, -2:] = 0.0
    return mask


def state(mask, seed=1):
    g = torch.Generator().manual_seed(seed)
    c = zero_com(torch.randn(B, N, 3, generator=g) * mask.unsqueeze(-1), mask)
    f = torch.randn(B, N, T, generator=g) * mask.unsqueeze(-1)
    return c, f


def diag_post(mask, seed=4):
    """m = D * x, an exact linear posterior with a known Jacobian."""
    g = torch.Generator().manual_seed(seed)
    D_c = torch.rand(B, N, 3, generator=g) + 0.5
    D_f = torch.rand(B, N, T, generator=g) + 0.5

    def post_fn(c, f):
        return Posterior(D_c * c, D_f * f, torch.full((B,), 0.7))
    return post_fn


def replay_eps(shape_c, shape_f, mask, std, K, seed):
    """The draws tfg_components makes, in the order it makes them."""
    g = torch.Generator().manual_seed(seed)
    m3 = mask.unsqueeze(-1)
    out = []
    for _ in range(K):
        e_c = torch.randn(shape_c, generator=g)
        e_f = torch.randn(shape_f, generator=g)
        out.append((zero_com(e_c * m3, mask) * std, e_f * m3 * std))
    return out


def tfg_ref_rescale(grad, clip_scale, node_mask):
    """TFG tasks/utils.py:rescale_grad, VERBATIM (grad [B,N,3+T], mask [B,N,1])."""
    scale = (grad ** 2).mean(dim=-1)
    if node_mask is not None:
        scale = scale.sum(dim=-1) / node_mask.float().squeeze(-1).sum(dim=-1)
        clipped_scale = torch.clamp(scale, max=clip_scale)
        co_ef = clipped_scale / scale
        grad = grad * co_ef.view(-1, 1, 1)
    return grad


def ref_rescale_pair(g_c, g_f, mask, clip):
    out = tfg_ref_rescale(torch.cat([g_c, g_f], -1), clip, mask.unsqueeze(-1))
    return out[..., :3], out[..., 3:]


def mx(*pairs):
    return max(float((a - b).abs().max()) for a, b in pairs)


CFG = dict(rho=0.016, mu=0.004, gamma=0.1, mad=0.8, n_iter=4, eps_bsz=1,
           rho_schedule="increase", mu_schedule="increase",
           sigma_schedule="decrease", clip_scale=100.0)


def main():
    mask = make_mask()
    m3 = mask.unsqueeze(-1)
    c, f = state(mask)
    post_fn = diag_post(mask)
    f_net = Quad()
    y = torch.tensor([1.0, -0.5, 2.0, 0.3])
    mad = 0.8

    # ---- T1 rescale_grad parity
    for clip, tag in ((1e9, "unbound"), (0.05, "bound")):
        g_c = torch.randn(B, N, 3) * m3 * 3.0
        g_f = torch.randn(B, N, T) * m3 * 3.0
        a_c, a_f, hit = tfg_rescale_grad(g_c, g_f, mask, clip)
        r_c, r_f = ref_rescale_pair(g_c, g_f, mask, clip)
        R["T1_rescale_matches_tfg_" + tag] = mx((a_c, r_c), (a_f, r_f))
    R["T1_bound_case_actually_bound"] = 0.0 if float(hit.min()) == 1.0 else 1.0

    # ---- T2 G == autograd reference, K = 1 and K = 3
    def ref_G(K, std, seed, var_scale, clip):
        eps = replay_eps(c.shape, f.shape, mask, std, K, seed)
        ci = c.clone().requires_grad_(True)
        fi = f.clone().requires_grad_(True)
        p = post_fn(ci, fi)
        lp = torch.stack([-((f_net(p.mean_coords + e_c, p.mean_feats + e_f, mask) - y)
                            / mad) ** 2 for (e_c, e_f) in eps])
        Lx = torch.logsumexp(lp, 0) - math.log(K)
        gc, gf = torch.autograd.grad(Lx.sum(), (ci, fi))
        return ref_rescale_pair(var_scale * gc * m3, var_scale * gf * m3, mask, clip)

    for K in (1, 3):
        for clip, tag in ((1e9, "unbound"), (1e-3, "bound")):
            G_c, G_f, _, _, _ = tfg_components(
                f_net, post_fn, c, f, mask, y, mad, 0.3, 0.0, n_iter=4,
                eps_bsz=K, want_var=True, var_scale=1.7, clip_scale=clip,
                generator=torch.Generator().manual_seed(21))
            r_c, r_f = ref_G(K, 0.3, 21, 1.7, clip)
            R["T2_G_autograd_K%d_%s" % (K, tag)] = mx((G_c, r_c), (G_f, r_f))

    # ---- T3 D0 == hand loop, same draws, per-iteration rescale
    for K, clip, tag in ((1, 1e9, "K1"), (3, 1e9, "K3"), (1, 0.02, "K1_bound")):
        mu_step, n_iter = 0.37, 4
        _, _, D_c, D_f, _ = tfg_components(
            f_net, post_fn, c, f, mask, y, mad, 0.2, mu_step, n_iter=n_iter,
            eps_bsz=K, want_var=True, var_scale=1.0, clip_scale=clip,
            generator=torch.Generator().manual_seed(33))
        eps = replay_eps(c.shape, f.shape, mask, 0.2, K, 33)
        p = post_fn(c, f)
        xc, xf = p.mean_coords.clone(), p.mean_feats.clone()
        for _ in range(n_iter):
            a = xc.clone().requires_grad_(True)
            b = xf.clone().requires_grad_(True)
            lp = torch.stack([-((f_net(a + e_c, b + e_f, mask) - y) / mad) ** 2
                              for (e_c, e_f) in eps])
            gc, gf = torch.autograd.grad((torch.logsumexp(lp, 0) - math.log(K)).sum(),
                                         (a, b))
            gc, gf = ref_rescale_pair(gc * m3, gf * m3, mask, clip)
            xc, xf = xc + mu_step * gc, xf + mu_step * gf
        R["T3_D0_hand_loop_" + tag] = mx((D_c, xc - p.mean_coords),
                                        (D_f, xf - p.mean_feats))
        if tag == "K1":
            R["T3_D0_is_nonzero"] = 0.0 if float(D_c.abs().max()) > 1e-6 else 1.0

    # ---- T4 std = 0 draws nothing
    g0 = torch.Generator().manual_seed(77)
    st0 = g0.get_state().clone()
    _, _, _, _, dg0 = tfg_components(f_net, post_fn, c, f, mask, y, mad, 0.0,
                                     0.1, eps_bsz=4, generator=g0)
    R["T4_std0_generator_untouched"] = float((g0.get_state() != st0).sum())
    R["T4_std0_single_zero_eps"] = abs(dg0["tfg_eps_bsz"] - 1)

    # ---- T5 variance step == (2 s^2/mad^2) * plug's field when std = 0
    s = 1.3
    G_c, G_f, _, _, _ = tfg_components(f_net, post_fn, c, f, mask, y, mad, 0.0,
                                       0.0, want_var=True, var_scale=1.0,
                                       clip_scale=1e30)
    P_c, P_f, _ = guidance_field(f_net, post_fn, c, f, mask, y, s, mode="plug")
    k = 2.0 * s ** 2 / mad ** 2
    R["T5_var_step_is_scaled_plug"] = mx((G_c, k * P_c * m3), (G_f, k * P_f * m3))

    # ---- T6 cost
    for (want, mu_step, n_it, K, exp) in (
            (True, 0.1, 4, 1, dict(gen_fwd=1, gen_vjp=1, guide_fwd=5, guide_bwd=5)),
            (True, 0.1, 4, 3, dict(gen_fwd=1, gen_vjp=1, guide_fwd=15, guide_bwd=15)),
            (True, 0.0, 4, 1, dict(gen_fwd=1, gen_vjp=1, guide_fwd=1, guide_bwd=1)),
            (False, 0.1, 4, 1, dict(gen_fwd=1, gen_vjp=0, guide_fwd=4, guide_bwd=4))):
        cst = Cost()
        _, _, D_c, _, _ = tfg_components(f_net, post_fn, c, f, mask, y, mad, 0.1,
                                         mu_step, n_iter=n_it, eps_bsz=K,
                                         want_var=want, cost=cst,
                                         generator=torch.Generator().manual_seed(1))
        got = {kk: getattr(cst, kk) for kk in exp}
        R["T6_cost_var%d_mu%g_K%d" % (want, mu_step, K)] = float(got != exp)
        if mu_step == 0.0:
            R["T6_mu0_D0_zero"] = float(D_c.abs().max())

    # ---- T7 schedules on the flow grid
    net = Net()
    fs = FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=1.0, tfg=CFG,
                     t_min_guide=0.0)
    steps = 20
    fs._grid_ts = fs.time_grid(steps)
    rows = [fs._tfg_schedule(i) for i in range(steps)]
    ts = fs._grid_ts.double()
    ab = ts ** 2 / (ts ** 2 + (1 - ts) ** 2)
    alpha = ab[:-1] / ab[1:]
    R["T7_rho_sums_to_rho_bar_T"] = abs(sum(r[0] for r in rows) - CFG["rho"] * steps)
    R["T7_mu_sums_to_mu_bar_T"] = abs(sum(r[1] for r in rows) - CFG["mu"] * steps)
    prof = torch.tensor([r[0] for r in rows]) / (CFG["rho"] * alpha * steps / alpha.sum())
    R["T7_increase_is_alpha"] = float((prof[1:] - 1.0).abs().max())
    R["T7_std_decrease"] = max(abs(r[2] - CFG["gamma"] * math.sqrt(1 - float(ab[i])))
                               for i, r in enumerate(rows))
    R["T7_flow_abar"] = max(abs(r[3] - float(ab[i])) + abs(r[4] - float(ab[i + 1]))
                            for i, r in enumerate(rows))
    dec = dict(CFG, rho_schedule="decrease")
    fs2 = FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", tfg=dec, t_min_guide=0.0)
    fs2._grid_ts = fs._grid_ts
    r_dec = torch.tensor([fs2._tfg_schedule(i)[0] for i in range(steps)])
    want_dec = CFG["rho"] * (1 - alpha) * steps / (1 - alpha).sum()
    R["T7_decrease_is_one_minus_alpha"] = float((r_dec - want_dec).abs().max())

    # ---- T8 flow geometry == TFG's VP update through x_vp = x / c_t
    worst_geo = worst_ddim = 0.0
    g = torch.Generator().manual_seed(3)
    for t, t1 in ((0.5, 0.51), (0.73, 0.74), (0.9, 0.95), (0.05, 0.2)):
        c_t = math.sqrt(t * t + (1 - t) ** 2)
        c_1 = math.sqrt(t1 * t1 + (1 - t1) ** 2)
        ab_t, ab_1 = t * t / c_t ** 2, t1 * t1 / c_1 ** 2
        al = ab_t / ab_1
        vs, k_var, k_0 = FlowSampler._tfg_geometry(t, t1, ab_t, ab_1, al)
        G_fm = torch.randn(6, generator=g)
        D0 = torch.randn(6, generator=g)
        rho = 0.37
        # TFG, VP state: gradient w.r.t. x_vp is c_t * G_fm; then map back
        A_vp = rho * (c_t * G_fm) / math.sqrt(al) + math.sqrt(ab_1) * D0
        ours = rho * k_var * (vs * G_fm) + k_0 * D0
        worst_geo = max(worst_geo, float((c_1 * A_vp - ours).abs().max()))
        # base step: DDIM(eta = 0) on the VP state == Euler on the flow state
        x = torch.randn(6, generator=g)
        m = torch.randn(6, generator=g)
        v = (m - x) / (1 - t)
        x_vp = x / c_t
        eps_hat = (x_vp - math.sqrt(ab_t) * m) / math.sqrt(1 - ab_t)
        ddim = math.sqrt(ab_1) * m + math.sqrt(1 - ab_1) * eps_hat
        worst_ddim = max(worst_ddim, float((c_1 * ddim - (x + (t1 - t) * v)).abs().max()))
    R["T8_flow_geometry_is_tfg_vp_update"] = worst_geo
    R["T8_ddim_base_is_euler"] = worst_ddim

    # ---- T9 FlowSampler one step, no clip
    w = 0.7
    for i_step in (10, 15):
        fsm = FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=w, tfg=CFG,
                          t_min_guide=0.0, clip=None)
        fsm._grid_ts = fsm.time_grid(steps)
        fsm._grid_i = i_step
        t0 = float(fsm._grid_ts[i_step])
        t1 = float(fsm._grid_ts[i_step + 1])
        out_c, out_f = fsm.field(c, f, t0)
        rho_i, mu_i, std_i, ab_c, ab_n, al = fsm._tfg_schedule(i_step)
        vs, k_var, k_0 = FlowSampler._tfg_geometry(t0, t1, ab_c, ab_n, al)
        tt = torch.full((B,), t0)
        pf = lambda cc, ff: fm_posterior(net, cc, ff, mask, tt)  # noqa: E731
        G_c, G_f, D_c, D_f, _ = tfg_components(
            f_net, pf, c, f, mask, y, CFG["mad"], std_i, w * mu_i, n_iter=4,
            eps_bsz=1, want_var=True, var_scale=vs, clip_scale=100.0,
            generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            v_c, v_f = net(c, f, mask, tt)
        h = t1 - t0
        e_c = v_c + zero_com((w * rho_i * k_var * G_c + k_0 * D_c) / h, mask)
        e_f = v_f + (w * rho_i * k_var * G_f + k_0 * D_f) / h * m3
        R["T9_flow_step_i%d" % i_step] = mx((out_c, e_c), (out_f, e_f))
        R["T9_flow_step_moves_i%d" % i_step] = 0.0 if float((out_c - v_c).abs().max()) > 1e-8 else 1.0

    # ---- T10 VPSampler one step, no clip
    vsm = VPSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=w, tfg=CFG, clip=None)
    vsm._grid_ts = vsm.time_grid(steps)
    i_step = 12
    vsm._grid_i = i_step
    tau0, tau1 = float(vsm._grid_ts[i_step]), float(vsm._grid_ts[i_step + 1])
    out_c, out_f = vsm.field(c, f, tau0)
    rho_i, mu_i, std_i, ab_c, ab_n, al = vsm._tfg_schedule(i_step)
    from diffusion import alpha_sigma, beta
    a0, s0 = alpha_sigma(torch.tensor([tau0]))
    a1, _ = alpha_sigma(torch.tensor([tau1]))
    R["T10_vp_abar_from_schedule"] = abs(ab_c - float(a0) ** 2) + abs(ab_n - float(a1) ** 2)
    tv = torch.full((B,), tau0)
    av, sv = alpha_sigma(tv)
    from guidance import vp_posterior
    pv = lambda cc, ff: vp_posterior(net, cc, ff, mask, tv, av, sv)  # noqa: E731
    G_c, G_f, D_c, D_f, _ = tfg_components(
        f_net, pv, c, f, mask, y, CFG["mad"], std_i, w * mu_i, n_iter=4,
        eps_bsz=1, want_var=True, var_scale=1.0, clip_scale=100.0,
        generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        e_c, e_f = net(c, f, mask, tv)
    bb = beta(tv).view(-1, 1, 1)
    sb = sv.view(-1, 1, 1)
    base_c = -0.5 * bb * c - 0.5 * bb * (-e_c / sb)
    base_f = -0.5 * bb * f - 0.5 * bb * (-e_f / sb)
    h = tau1 - tau0
    D_vc = w * rho_i * G_c / math.sqrt(al) + math.sqrt(ab_n) * D_c
    D_vf = w * rho_i * G_f / math.sqrt(al) + math.sqrt(ab_n) * D_f
    exp_c = zero_com(base_c + zero_com(D_vc / h, mask), mask)
    exp_f = (base_f + D_vf / h * m3) * m3
    R["T10_vp_step"] = mx((out_c, exp_c), (out_f, exp_f))
    R["T10_vp_h_is_negative"] = 0.0 if h < 0 else 1.0

    # ---- T11 w = 0 is the unguided trajectory, bit for bit
    STEPS = 12
    c0, f0 = state(mask, seed=8)
    ung = FlowSampler(net, mask, f_net=None, t_min_guide=0.5)
    uc, uf, _ = integrate(ung, c0.clone(), f0.clone(), STEPS, "euler")
    z = FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=0.0, tfg=CFG,
                    t_min_guide=0.5)
    zc, zf, _ = integrate(z, c0.clone(), f0.clone(), STEPS, "euler")
    R["T11_w0_is_unguided_bitwise"] = mx((uc, zc), (uf, zf))
    R["T11_w0_guided_steps_ran"] = 0.0 if z.n_guided > 0 else 1.0

    # ---- T12 padding and CoM, through a real integrate with the clip on
    gd = FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=2.0, tfg=CFG,
                     t_min_guide=0.5)
    gc_, gf_, _ = integrate(gd, c0.clone(), f0.clone(), STEPS, "euler")
    pad = (1 - mask).unsqueeze(-1)
    R["T12_padded_atoms_untouched"] = float(((gc_ - uc) * pad).abs().max()
                                            + ((gf_ - uf) * pad).abs().max())
    com = ((gc_ - uc) * m3).sum(1) / mask.sum(1, keepdim=True)
    R["T12_coord_change_zero_com"] = float(com.abs().max())
    R["T12_guided_differs"] = 0.0 if float((gc_ - uc).abs().max()) > 1e-8 else 1.0
    dsum = gd.diag_summary()
    R["T12_diag_d0_frac_logged"] = 0.0 if 0.0 < dsum.get("tfg_d0_frac", -1) <= 1.0 else 1.0
    R["T12_diag_corr_over_v_logged"] = 0.0 if dsum.get("tfg_corr_over_v", -1) > 0 else 1.0

    # ---- T13 refusals
    def raises(fn, exc=Exception):
        try:
            fn()
        except exc:
            return 0.0
        return 1.0
    R["T13_heun_refused"] = raises(lambda: integrate(
        FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", tfg=CFG, t_min_guide=0.0),
        c0.clone(), f0.clone(), 6, "heun"), NotImplementedError)
    R["T13_field_outside_integrate"] = raises(lambda: FlowSampler(
        net, mask, f_net=f_net, y=y, mode="tfg", tfg=CFG, t_min_guide=0.0).field(c, f, 0.5),
        RuntimeError)
    R["T13_missing_config"] = raises(lambda: FlowSampler(
        net, mask, f_net=f_net, y=y, mode="tfg", tfg={"rho": 1.0}), ValueError)
    R["T13_update_rule"] = raises(lambda: FlowSampler(
        net, mask, f_net=f_net, y=y, mode="tfg", tfg=CFG, update_rule="adam"), ValueError)
    R["T13_guidance_field_refuses"] = raises(lambda: guidance_field(
        f_net, post_fn, c, f, mask, y, 1.0, mode="tfg"), ValueError)

    # ---- T14 determinism, and the initial-noise stream is untouched
    a1 = integrate(FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=1.5,
                               tfg=CFG, t_min_guide=0.5), c0.clone(), f0.clone(), STEPS)
    a2 = integrate(FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=1.5,
                               tfg=CFG, t_min_guide=0.5), c0.clone(), f0.clone(), STEPS)
    R["T14_deterministic"] = mx((a1[0], a2[0]), (a1[1], a2[1]))
    gA = torch.Generator().manual_seed(123)
    initial_noise(mask, T, gA)
    integrate(FlowSampler(net, mask, f_net=f_net, y=y, mode="tfg", w=1.5, tfg=CFG,
                          t_min_guide=0.5), c0.clone(), f0.clone(), STEPS)
    nA = initial_noise(mask, T, gA)
    gB = torch.Generator().manual_seed(123)
    initial_noise(mask, T, gB)
    nB = initial_noise(mask, T, gB)
    R["T14_initial_noise_stream_untouched"] = mx((nA[0], nB[0]), (nA[1], nB[1]))

    # ---- T15 sweep wiring
    import guidance_sweep as gs
    import check_fullrun_go as cg
    R["T15_tfg_in_compare_set"] = 0.0 if "tfg" in gs.COMPARE_SET else 1.0
    R["T15_dflow_out_of_compare_set"] = 0.0 if "dflow" not in gs.COMPARE_SET else 1.0
    R["T15_table11_alpha"] = 0.0 if gs.TFG_QM9["alpha"] == (0.016, 0.001, 0.0001) else 1.0
    R["T15_table11_mu"] = 0.0 if gs.TFG_QM9["mu"] == (0.001, 0.002, 0.1) else 1.0
    R["T15_table11_gap"] = 0.0 if gs.TFG_QM9["gap"] == (0.032, 0.001, 0.001) else 1.0
    cfg = gs.tfg_config("mu", 1.23)
    R["T15_config_complete"] = float(sorted(cfg) != sorted(CFG))
    R["T15_config_fixed_values"] = float(
        (cfg["n_iter"], cfg["eps_bsz"], cfg["mu_schedule"], cfg["clip_scale"])
        != (4, 1, "increase", 100.0))
    R["T15_w_not_rescaled"] = abs(gs.strength_scale("tfg", {"tfg": cfg}, 1.5) - 1.0)
    R["T15_check_script_agrees"] = float(tuple(gs.COMPARE_SET) != cg.ALL_ARMS)
    R["T15_registered_fr1_kept"] = float(cg.REGISTERED_COMPETITORS
                                         != ("plug", "tmpd", "lgd_mc", "dflow"))
    cells = gs.plan_compare_cells(["mu", "alpha", "gap"], ["tfg"])
    R["T15_42_compare_cells"] = float(len(cells) != 42 or {x[1] for x in cells} != {"tfg"})
    R["T15_cell_names_are_cmp"] = float(not all(
        gs.cell_name(*x).endswith("__tmin0.5__cmp.json") for x in cells))
    full = gs.plan_compare_cells(["mu"])
    R["T15_default_plan_unchanged_size"] = float(len(full) != 86)

    # ---- T16 the flow var_scale reaches rescale_grad
    t_s = 0.5
    c_t = math.sqrt(t_s ** 2 + (1 - t_s) ** 2)          # 0.707: c_t G < G
    Gc, Gf, _, _, _ = tfg_components(f_net, post_fn, c, f, mask, y, mad, 0.0, 0.0,
                                     var_scale=1.0, clip_scale=1e30)
    ms_plain = ((Gc ** 2).sum((1, 2)) + (Gf ** 2).sum((1, 2))) / ((3 + T) * mask.sum(1))
    # a threshold between ms(c_t G) = c_t^2 ms(G) and ms(G), for sample 1
    thr = float(ms_plain[1]) * (c_t ** 2 + 1.0) / 2.0
    _, _, _, _, d_scaled = tfg_components(f_net, post_fn, c, f, mask, y, mad, 0.0, 0.0,
                                          var_scale=c_t, clip_scale=thr)
    _, _, _, _, d_plain = tfg_components(f_net, post_fn, c, f, mask, y, mad, 0.0, 0.0,
                                         var_scale=1.0, clip_scale=thr)
    R["T16_var_scale_changes_the_clip_decision"] = float(
        not (float(d_plain["tfg_var_rescaled"][1]) == 1.0
             and float(d_scaled["tfg_var_rescaled"][1]) == 0.0))

    # ---- T17 the decoded-type view in evaluate_samples
    import evaluation as ev

    class Stub(torch.nn.Module):
        def forward(self, cc, ff, mm):
            return (cc ** 2 * mm.unsqueeze(-1)).sum((1, 2)) + (ff * torch.arange(
                1.0, ff.shape[-1] + 1)).sum((1, 2))

        def embed(self, cc, ff, mm):
            return torch.cat([cc, ff], -1).mul(mm.unsqueeze(-1)).sum(1)

    types5 = ["H", "C", "N", "O", "F"]
    Bq, Nq = 7, 6
    mq = torch.ones(Bq, Nq)
    mq[:, -2:] = 0.0
    cq = torch.randn(Bq, Nq, 3, generator=torch.Generator().manual_seed(2)) * mq.unsqueeze(-1)
    oh = torch.nn.functional.one_hot(torch.randint(0, 5, (Bq, Nq), generator=torch.Generator(
        ).manual_seed(3)), 5).double() * mq.unsqueeze(-1)
    yq = torch.linspace(0.0, 3.0, Bq)
    r_exact = ev.evaluate_samples(cq, oh, mq, types5, Stub(), Stub(), yq, 0.7, per_mol=True)
    R["T17_onehot_dec_equals_soft"] = (
        abs(r_exact["prop_mae_eval_dec"] - r_exact["prop_mae_eval"])
        + abs(r_exact["in_band_fraction_dec"] - r_exact["in_band_fraction"])
        + float((r_exact["_per_mol"]["f_B_dec"] - r_exact["_per_mol"]["f_B"]).abs().max()))
    r_q = ev.evaluate_samples(cq, oh / 4.0, mq, types5, Stub(), Stub(), yq, 0.7,
                              hot_value=0.25)
    r_q_soft = ev.evaluate_samples(cq, oh / 4.0, mq, types5, Stub(), Stub(), yq, 0.7)
    R["T17_scaled_onehot_hot_value"] = abs(r_q["prop_mae_eval_dec"] - r_q["prop_mae_eval"])
    R["T17_wrong_hot_value_would_differ"] = 0.0 if abs(
        r_q_soft["prop_mae_eval_dec"] - r_q_soft["prop_mae_eval"]) > 1e-6 else 1.0
    soft = oh + 0.3 * torch.randn(Bq, Nq, 5, generator=torch.Generator().manual_seed(4)) * mq.unsqueeze(-1)
    r_s = ev.evaluate_samples(cq, soft, mq, types5, Stub(), Stub(), yq, 0.7, per_mol=True)
    dec = torch.nn.functional.one_hot(soft.argmax(-1), 5).double() * mq.unsqueeze(-1)
    want = (Stub()(cq, dec, mq) - yq).abs().mean()
    R["T17_dec_is_argmax_onehot"] = abs(r_s["prop_mae_eval_dec"] - float(want))
    R["T17_dec_differs_from_soft"] = 0.0 if abs(
        r_s["prop_mae_eval_dec"] - r_s["prop_mae_eval"]) > 1e-6 else 1.0
    R["T17_decode_masks_padding"] = float(ev.decode_types(soft, mq)[:, -2:].abs().max())

    # ---------------------------------------------------------------- report
    bad = {k: v for k, v in R.items() if not (v <= TOL)}
    width = max(len(k) for k in R)
    for k, v in R.items():
        print("  %-*s  %.3e  %s" % (width, k, v, "ok" if v <= TOL else "FAIL"))
    print("\n%d gates, %d failed" % (len(R), len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
