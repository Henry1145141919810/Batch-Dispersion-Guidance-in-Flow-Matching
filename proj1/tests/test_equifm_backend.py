"""Gates for the EquiFM backend (proj1/src/external/equifm_backend.py).

Every check compares the adapter against an EXPLICIT reference built a
different way, never against itself:

  load_*              strict load of the released checkpoint, pinned SHA-256,
                      the audit's parameter count
  native_parity       EquiFMSampler (unguided) reproduces the checkpoint's own
                      Cnflows.forward Euler loop, the loop the usability audit
                      measured 98.4 / 86.7 with
  mean_inverts_velocity  at real states, the posterior-mean formulas are the
                      exact inverse of the velocity the native sampler uses:
                      v_x = ((1-eps) x - m_x)/s_x,  v_h = M (a m_h - a^2 h)
  mean_exact_targets  feeding the regression TARGETS of the training code in
                      place of the network recovers x_0 and h_0 exactly
  sigma_blocks        guidance.sigma_times_vector through the tangent-scaling
                      posterior (k = 1) equals K (J v) from an unscaled JVP
  vjp_unscaled        guidance._pullback is untouched by the tangent scaling
  btvg_variance_grad  guidance.grad_property_variance (reverse-over-forward
                      through the custom op) equals a hand-written per-block
                      reference, value and gradient, Hessian term included
  velocity_delta      the guided velocity shift equals the native velocity map
                      at m + K G minus at m
  padding_*           padded atoms receive no guidance and no velocity
  tfg_*               TFG's port: per-block geometry equals FlowSampler's
                      (coords, to eps) and VPSampler's (types, exactly); the
                      displacement equals a hand-built reference; w = 0 is
                      the unguided field; the step moves f toward the target

Run: python proj1/tests/test_equifm_backend.py
CPU, float64 for the exactness gates; ~1-2 GB RAM.
"""
import hashlib
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

import guidance as G                                               # noqa: E402
from external import equifm_backend as E                           # noqa: E402
from external.tfg_assets import dense_edges                        # noqa: E402
from sampling import initial_noise, integrate                      # noqa: E402

R = {}


def gate(name, err, tol):
    R[name] = (float(err), float(tol))


def rel(a, b):
    return float((a - b).abs().max() / (b.abs().max() + 1e-30))


def toy_f(m_c, m_f, mask):
    """A smooth, invariant, non-linear stand-in property (nonzero Hessian in
    both blocks)."""
    m3 = mask.unsqueeze(-1)
    d = ((m_c[:, :, None] - m_c[:, None]) ** 2).sum(-1)
    pm = mask[:, :, None] * mask[:, None, :]
    w = torch.tensor([0.3, -0.2, 0.5, 0.1, -0.4], dtype=m_f.dtype, device=m_f.device)
    return (0.05 * (d * pm).sum((1, 2)) + ((m_f * m3) @ w).sum(1)
            + 0.5 * ((m_f * m3) ** 2).sum((1, 2)))


def unscaled_post(gen, coords, types, charge, mask, tau):
    """The same mean WITHOUT the custom op: the reference for J."""
    q_x, q_h = gen.raw(coords, torch.cat([types, charge], dim=-1), mask, tau)
    a = E.a_of(tau)
    m3 = mask.unsqueeze(-1)
    m_c = E.zero_com(((1.0 - E.EPS) * coords - E.s_x_of(tau) * q_x) * m3, mask)
    m_f = (a * types + q_h[..., :E.N_TYPES] / a) * m3
    return m_c, m_f


def main():
    torch.manual_seed(0)
    # --- load
    sha = hashlib.sha256(E.EQUIFM_WEIGHTS.read_bytes()).hexdigest()
    gate("load_sha256_pinned", 0.0 if sha == E.EQUIFM_SHA256 else 1.0, 0.5)
    gen = E.EquiFMGenerator(device="cpu")
    n_par = sum(p.numel() for p in gen.flow.parameters())
    gate("load_param_count_5339920", abs(n_par - 5339920), 0.5)

    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=True)
    te = d["split"]["test"][:3]
    mask = d["mask"][te].float()
    mask[2, 12:] = 0.0                       # a genuinely padded molecule
    B = mask.shape[0]

    # --- native parity (float32, the production precision)
    g = torch.Generator().manual_seed(7)
    c0, f0 = initial_noise(mask, 6, g)
    smp = E.EquiFMSampler(gen, mask)
    c_ours, f_ours, _ = integrate(smp, c0.clone(), f0.clone(), 10, "euler")
    flow = gen.flow
    flow.set_conditional_param(mask[..., None], dense_edges(mask)[1])
    z = torch.cat([c0, f0], -1) * mask[..., None]
    with torch.no_grad():
        for i in range(10):
            t = torch.tensor(1 - i / 10)
            z = (z - flow(t, z) / 10) * mask[..., None]
    gate("native_parity_coords", (c_ours - z[..., :3]).abs().max(), 1e-4)
    gate("native_parity_feats", (f_ours - z[..., 3:]).abs().max(), 1e-4)

    # --- exactness gates in float64
    gen = gen.double()
    mask = mask.double()
    x = E.zero_com(torch.randn(B, 29, 3, dtype=torch.float64) * mask[..., None], mask)
    h = torch.randn(B, 29, 6, dtype=torch.float64) * mask[..., None]
    types, charge = h[..., :5], h[..., 5:]
    for tau in (0.5, 0.2, 0.05):
        v_x, v_h = gen.velocity(x, h, mask, tau)
        m_c, m_f = unscaled_post(gen, x, types, charge, mask, tau)
        a = E.a_of(tau)
        vx_m = ((1 - E.EPS) * x - m_c) / E.s_x_of(tau)
        vh_m = E.feat_multiplier(tau) * (a * m_f - a * a * types)
        m3 = mask[..., None]
        gate("mean_inverts_velocity_x__tau%g" % tau,
             rel(E.zero_com(v_x * m3, mask), vx_m * m3), 1e-9)
        gate("mean_inverts_velocity_h__tau%g" % tau,
             rel(v_h[..., :5] * m3, vh_m * m3), 1e-9)

    # exact regression targets in place of the network
    x0 = E.zero_com(torch.randn(B, 29, 3, dtype=torch.float64) * mask[..., None], mask)
    h0 = torch.randn(B, 29, 5, dtype=torch.float64) * mask[..., None]
    zc = E.zero_com(torch.randn_like(x0) * mask[..., None], mask)
    zh = torch.randn_like(h0) * mask[..., None]
    tau = 0.37
    a = E.a_of(tau)
    xt = (1 - tau) * x0 + E.s_x_of(tau) * zc
    ht = a * h0 + (1 - a * a) ** 0.5 * zh

    class Oracle:
        def raw(self, c, f, m, t):
            return (1 - E.EPS) * zc - x0, torch.cat(
                [a * h0 - a * a * ht, torch.zeros_like(ht[..., :1])], -1)
    p = E.equifm_posterior(Oracle(), xt, ht, torch.zeros_like(ht[..., :1]), mask, tau)
    gate("mean_exact_targets_x", rel(p.mean_coords, x0), 1e-12)
    gate("mean_exact_targets_h", rel(p.mean_feats, h0), 1e-12)

    # Sigma v per block, the pull-back, and BTVG's variance gradient
    for tau in (0.45, 0.15):
        post_s = lambda c, f: E.equifm_posterior(gen, c, f, charge, mask, tau)  # noqa: E731
        post_u = lambda c, f: unscaled_post(gen, c, f, charge, mask, tau)      # noqa: E731
        k_c, k_f = E.block_k(tau)
        vc = E.zero_com(torch.randn_like(x) * mask[..., None], mask)
        vf = torch.randn_like(types) * mask[..., None]
        sc, sf = G.sigma_times_vector(post_s, x, types, mask, vc, vf, 1.0)
        _, (jc, jf) = torch.func.jvp(post_u, (x, types), (vc, vf))
        gate("sigma_blocks_x__tau%g" % tau, rel(sc, k_c * jc), 1e-10)
        gate("sigma_blocks_h__tau%g" % tau, rel(sf, k_f * jf), 1e-10)
        # the two blocks really are scaled differently (the gate has teeth)
        gate("sigma_blocks_differ__tau%g" % tau,
             0.0 if abs(k_c / k_f - 1) > 0.2 else 1.0, 0.5)

        wc = torch.randn_like(x) * mask[..., None]
        wf = torch.randn_like(types) * mask[..., None]
        pc, pf = G._pullback(post_s, x, types, wc, wf)
        ci = x.detach().requires_grad_(True)
        fi = types.detach().requires_grad_(True)
        mc, mf = post_u(ci, fi)
        rc, rf = torch.autograd.grad((mc * wc).sum() + (mf * wf).sum(), (ci, fi))
        gate("vjp_unscaled_x__tau%g" % tau, rel(pc, rc), 1e-12)
        gate("vjp_unscaled_h__tau%g" % tau, rel(pf, rf), 1e-12)

        m_c, m_f = (t.detach() for t in post_u(x, types))
        _, g_c, g_f = G.property_derivs(toy_f, m_c, m_f, mask)
        V, d_c, d_f = G.grad_property_variance(toy_f, post_s, x, types, mask,
                                               m_c, m_f, 1.0, g_c, g_f)
        # reference: V = k_c g_c'(Jg)_c + k_f g_f'(Jg)_f, g held fixed, plus
        # 2 J^T H Sigma g with Sigma g = K J g -- written out per block
        ci = x.detach().requires_grad_(True)
        fi = types.detach().requires_grad_(True)
        jg_c, jg_f = torch.func.jvp(post_u, (ci, fi), (g_c, g_f))[1]
        V_ref = k_c * (g_c * jg_c).sum((1, 2)) + k_f * (g_f * jg_f).sum((1, 2))
        r_c, r_f = torch.autograd.grad(V_ref.sum(), (ci, fi))
        hs_c, hs_f = G._hvp(toy_f, m_c, m_f, mask, k_c * jg_c.detach(),
                            k_f * jg_f.detach())
        post_uP = lambda c, f: G.Posterior(*post_u(c, f), 1.0)  # noqa: E731
        p_c, p_f = G._pullback(post_uP, x, types, hs_c, hs_f)
        r_c, r_f = r_c + 2 * p_c, r_f + 2 * p_f
        gate("btvg_variance_value__tau%g" % tau, rel(V, V_ref.detach()), 1e-10)
        gate("btvg_variance_grad_x__tau%g" % tau, rel(d_c, r_c), 1e-9)
        gate("btvg_variance_grad_h__tau%g" % tau, rel(d_f, r_f), 1e-9)

        # the velocity shift is the native velocity map at m + K G minus at m
        Gc = E.zero_com(torch.randn_like(x) * mask[..., None], mask)
        Gf = torch.randn_like(types) * mask[..., None]
        dv_c, dv_f = E.velocity_delta(tau, Gc, Gf)
        aa = E.a_of(tau)
        ref_c = (((1 - E.EPS) * x - (m_c + k_c * Gc)) - ((1 - E.EPS) * x - m_c)) / E.s_x_of(tau)
        ref_f = E.feat_multiplier(tau) * aa * ((m_f + k_f * Gf) - m_f)
        gate("velocity_delta_x__tau%g" % tau, rel(dv_c, ref_c), 1e-12)
        gate("velocity_delta_h__tau%g" % tau, rel(dv_f, ref_f), 1e-12)

    # ---- TFG on EquiFM: geometry, displacement, zero strength, sign
    from sampling import FlowSampler
    for tau, tau1 in ((0.45, 0.44), (0.15, 0.14)):
        geo = E.tfg_block_geometry(tau, tau1)
        fs = FlowSampler._tfg_geometry(1 - tau, 1 - tau1, None, None, None)
        # coordinates: TFG's flow mapping at t = 1 - tau (differs by eps only)
        gate("tfg_geom_x_is_flow__tau%g" % tau,
             max(abs(a - b) for a, b in zip(geo["x"], fs)), 1e-3)
        # types: exactly TFG's VP mapping (1, 1/sqrt(alpha), sqrt(abar'))
        vp = (1.0, E.a_of(tau1) / E.a_of(tau), E.a_of(tau1))
        gate("tfg_geom_h_is_vp__tau%g" % tau,
             max(abs(a - b) for a, b in zip(geo["h"], vp)), 1e-12)
    ts = torch.tensor([1.0, 0.5, 0.2, 0.0], dtype=torch.float64)
    gate("tfg_clock_is_flow_clock",
         (E.EquiFMSampler._tfg_abar(None, ts)
          - FlowSampler._tfg_abar(1 - ts).double()).abs().max(), 1e-3)

    tcfg = dict(rho=0.02, mu=0.01, gamma=0.0, mad=1.0, n_iter=4, eps_bsz=1,
                rho_schedule="increase", mu_schedule="increase",
                sigma_schedule="decrease", clip_scale=100.0)
    y_hi = toy_f(*(t.detach() for t in unscaled_post(gen, x, types, charge, mask, 0.3)),
                 mask) + 5.0                                   # a target above f(m)
    n_steps = 10
    grid = torch.linspace(1.0, 0.0, n_steps + 1)
    i = 7                                                      # tau = 0.3
    tau, tau1 = float(grid[i]), float(grid[i + 1])
    smp = E.EquiFMSampler(gen, mask, f_net=toy_f, y=y_hi, s=1.0, mode="tfg",
                          w=1.0, clip=None, tfg=tcfg)
    smp._grid_ts, smp._grid_i = grid, i
    post_s = lambda c, f: E.equifm_posterior(gen, c, f, charge, mask, tau)  # noqa: E731
    v0c, v0f = gen.velocity(x, h, mask, tau)
    C_c, C_f = smp._tfg_step(x, types, tau, post_s, 1.0, v0c, v0f[..., :5])
    hstep = tau1 - tau
    # explicit reference, built without tfg_components' var_scale path:
    rho_i, mu_i, std_i, _, _, _ = smp._tfg_schedule(i)
    gate("tfg_std_zero_when_gamma_zero", abs(std_i), 1e-15)
    geo = E.tfg_block_geometry(tau, tau1)
    ci = x.detach().requires_grad_(True)
    fi = types.detach().requires_grad_(True)
    mc_, mf_ = unscaled_post(gen, ci, fi, charge, mask, tau)
    L = G.tfg_logf(toy_f, mc_, mf_, mask, y_hi, 1.0)
    gx, gh = torch.autograd.grad(L.sum(), (ci, fi))
    m3 = mask[..., None]
    Gx, Gh, _ = G.tfg_rescale_grad(geo["x"][0] * gx * m3, gh * m3, mask, 100.0)
    xc, xf = mc_.detach().clone(), mf_.detach().clone()
    for _ in range(4):
        a_ = xc.detach().requires_grad_(True)
        b_ = xf.detach().requires_grad_(True)
        g1, g2 = torch.autograd.grad(G.tfg_logf(toy_f, a_, b_, mask, y_hi, 1.0).sum(), (a_, b_))
        g1, g2, _ = G.tfg_rescale_grad(g1 * m3, g2 * m3, mask, 100.0)
        xc, xf = xc + mu_i * g1, xf + mu_i * g2
    Dx, Dh = xc - mc_.detach(), xf - mf_.detach()
    ref_c = E.zero_com((rho_i * geo["x"][1] * Gx + geo["x"][2] * Dx) * m3, mask)
    ref_f = (rho_i * geo["h"][1] * Gh + geo["h"][2] * Dh) * m3
    gate("tfg_displacement_x", rel(C_c * hstep, ref_c), 1e-9)
    gate("tfg_displacement_h", rel(C_f * hstep, ref_f), 1e-9)
    # sign: the displacement raises f(m) toward a target above it
    gate("tfg_moves_toward_target",
         0.0 if float((gx * ref_c).sum() + (gh * ref_f).sum()) > 0 else 1.0, 0.5)
    # zero strength is exactly the unguided field
    smp0 = E.EquiFMSampler(gen, mask, f_net=toy_f, y=y_hi, s=1.0, mode="tfg",
                           w=0.0, clip=None, tfg=tcfg)
    smp0._grid_ts, smp0._grid_i = grid, i
    vgc, vgf = smp0.field(x, h, tau)
    vuc, vuf = E.EquiFMSampler(gen, mask).field(x, h, tau)
    gate("tfg_w0_is_unguided", max(float((vgc - vuc).abs().max()),
                                   float((vgf - vuf).abs().max())), 1e-12)

    # padding: a guided field puts nothing on padded atoms
    smp = E.EquiFMSampler(gen, mask, f_net=toy_f, y=torch.zeros(B, dtype=torch.float64),
                          s=1.0, mode="plug", w=1.0, clip=None)
    vc, vf = smp.field(x, h, 0.3)
    pad = mask == 0
    gate("padding_velocity_coords", vc[pad].abs().max(), 1e-12)
    gate("padding_velocity_feats", vf[pad].abs().max(), 1e-12)
    gate("guided_field_moves", 0.0 if smp.n_guided == 1 else 1.0, 0.5)
    # and the guided field differs from the unguided one (not a no-op)
    v0c, v0f = E.EquiFMSampler(gen, mask).field(x, h, 0.3)
    gate("guided_differs_from_unguided",
         0.0 if (vc - v0c).abs().max() > 1e-6 else 1.0, 0.5)

    print("%-44s %13s %11s   pass" % ("check", "error", "tolerance"))
    print("-" * 78)
    ok = True
    for k, (v, tol) in R.items():
        p = v < tol
        ok = ok and p
        print("%-44s %13.3e %11.1e   %s" % (k, v, tol, "yes" if p else "NO"))
    print("-" * 78)
    print("ALL PASS (%d gates)" % len(R) if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
