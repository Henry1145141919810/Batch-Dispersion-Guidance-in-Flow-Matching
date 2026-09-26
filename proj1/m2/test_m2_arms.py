"""Numerical self-tests for the Modality-2 guidance arms.

These test the MATH INVARIANTS that any correct implementation of plug / tmpd /
lgd_mc / tfg_mc / bdg on the simplex must satisfy. They do not test m2_sweep.py's
internals, so they stay valid while that file is refactored; where m2_sweep
exposes `guidance_field` the last group checks it against these invariants.

Why these particular tests. gc_soft is EXACTLY affine in x (verified in T1), so
    f(m) = <g, m>,   g[l, {C,G}] = 1/L,   g[l, {A,T}] = 0,   H = 0,  |g|^2 = 2/L
and every arm's field collapses to ONE scalar per sample times the SAME pulled
back direction A = J^T g. That makes each arm's K -> infinity / spread -> 0 limit
a closed form, which is what T5 to T8 check. An implementation that gets the
pullback, the denominator or the draw scale wrong fails one of them by more than
its stated tolerance.

Run:  srun -p genoa-std-mem -n1 -c32 -t 20 python proj1/m2/test_m2_arms.py
"""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import simplex_fm as S                                            # noqa: E402

CKPT = os.path.join(HERE, "blade_bundle", "fm_m2_dfb500.pt")
FAILED = []


def check(name, ok, detail=""):
    print("  %-52s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAILED.append(name)


def gvec(B, L, dtype=torch.float32):
    """grad gc_soft: constant, 1/L on channels C(1) and G(2)."""
    g = torch.zeros(B, L, 4, dtype=dtype)
    g[:, :, 1] = 1.0 / L
    g[:, :, 2] = 1.0 / L
    return g


def load(dtype=torch.float32):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = S.SimplexFM(ck["hidden"], ck["layers"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    if dtype is torch.float64:
        net = net.double()
    return net, ck


def mean_and_A(net, x, t, dtype=torch.float32):
    """m = x + (1-t) v(x,t), f = gc_soft(m), and A = J^T g from ONE VJP.

    autograd.grad(f, x, grad_outputs=1) is exactly J^T g per sample because
    f_i = <g, m_i> and m_i depends only on sample i. Every arm's field is a
    per-sample scalar times this A, which is why they all cost one VJP.
    """
    B = x.shape[0]
    xt = x.detach().to(dtype).requires_grad_(True)
    tv = torch.full((B,), float(t), dtype=dtype)
    m = xt + (1.0 - float(t)) * net(xt, tv)
    f = S.gc_soft(m)
    A = torch.autograd.grad(f, xt, grad_outputs=torch.ones(B, dtype=dtype),
                            retain_graph=True)[0]
    return m, f.detach(), A.detach(), xt


def mc_coef(F, spread, K, s2, y, seed, gn2):
    """The lgd_mc / tfg_mc coefficient, scalar-draw form (exact for affine f):
    F_k = F + spread*|g|*xi_k,  c = sum_k softmax_k(-(y-F_k)^2/2s^2) (y-F_k)/s^2."""
    gg = torch.Generator().manual_seed(seed)
    xi = torch.randn((K, F.shape[0]), generator=gg, dtype=F.dtype)
    sp = spread if torch.is_tensor(spread) else torch.full_like(F, float(spread))
    Fk = F.unsqueeze(0) + sp.unsqueeze(0) * math.sqrt(gn2) * xi
    w = torch.softmax(-((y - Fk) ** 2) / (2.0 * s2), dim=0)
    return (w * (y - Fk) / s2).sum(0)


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    net32, ck = load()
    L, s = ck["crop"], ck["gc_std"]
    s2, y, gn2 = s * s, float(ck["gc_mean"]), 2.0 / L
    print("L=%d  s=%.6f  s^2=%.8f  |g|^2=%.6g  y=%.5f" % (L, s, s2, gn2, y))

    # ---- T1  the property is exactly affine ------------------------------
    print("\nT1 gc_soft is exactly affine (H = 0, g constant)")
    torch.manual_seed(1)
    xa = torch.distributions.Dirichlet(torch.ones(4)).sample((4, L)).requires_grad_(True)
    ga = torch.autograd.grad(S.gc_soft(xa).sum(), xa)[0]
    xb = torch.distributions.Dirichlet(torch.ones(4)).sample((4, L)).requires_grad_(True)
    gb = torch.autograd.grad(S.gc_soft(xb).sum(), xb)[0]
    check("grad is state-independent (H == 0)", torch.equal(ga, gb),
          "max|ga-gb|=%.1e" % float((ga - gb).abs().max()))
    check("grad equals the closed form g", torch.allclose(ga, gvec(4, L), atol=0),
          "|g|^2=%.6g vs 2/L=%.6g" % (float((ga[0] ** 2).sum()), gn2))
    z = torch.distributions.Dirichlet(torch.ones(4)).sample((4, L))
    check("gc_soft(x) == <g,x> with no offset",
          float((S.gc_soft(z) - (gvec(4, L) * z).sum(dim=(1, 2))).abs().max()) < 1e-7)

    # ---- T2  the pullback against a central finite difference -------------
    print("\nT2 pullback J^T(c g) vs central FD, float64 (must be O(h^2))")
    net64, _ = load(torch.float64)
    B, t = 4, 0.7
    x0 = torch.distributions.Dirichlet(torch.ones(4)).sample((B, L)).double()
    tv = torch.full((B,), t, dtype=torch.float64)
    c = torch.randn(B, generator=torch.Generator().manual_seed(3)).double()
    xt = x0.clone().requires_grad_(True)
    G = torch.autograd.grad(S.gc_soft(xt + (1 - t) * net64(xt, tv)), xt,
                            grad_outputs=c)[0]
    u = torch.randn((B, L, 4), generator=torch.Generator().manual_seed(4)).double()
    u = u / u.reshape(B, -1).norm(dim=1).view(-1, 1, 1)
    ana, rels = float((G * u).sum()), []
    for h in (1e-2, 1e-3, 1e-4):
        with torch.no_grad():
            fp = S.gc_soft(x0 + h * u + (1 - t) * net64(x0 + h * u, tv))
            fm = S.gc_soft(x0 - h * u + (1 - t) * net64(x0 - h * u, tv))
        fd = float((c * (fp - fm) / (2 * h)).sum())
        rels.append(abs(ana - fd) / abs(fd))
    check("FD agrees at h=1e-4", rels[-1] < 1e-7, "rel=%.1e" % rels[-1])
    check("error falls ~100x per 10x in h (second order)",
          rels[0] / rels[1] > 50 and rels[1] / rels[2] > 50,
          "rels=%s" % ["%.1e" % r for r in rels])

    # ---- T3  one VJP serves every arm -------------------------------------
    print("\nT3 one VJP serves every arm: grad(f, x_t, c) == c_i * A_i")
    _, _, A64, _ = mean_and_A(net64, x0, t, torch.float64)
    pred = c.view(-1, 1, 1) * A64
    rel = float((G - pred).norm() / pred.norm())
    check("scalar coefficients commute with the pullback", rel < 1e-12,
          "rel Frobenius = %.1e" % rel)

    # ---- T4  v_f costs nothing --------------------------------------------
    print("\nT4 v_f = k g^T Sigma g / k = k <g, A> from the SAME VJP (no JVP)")
    k = (1 - t) ** 2 / t
    vf_free = k * (gvec(B, L, torch.float64) * A64).sum(dim=(1, 2))
    _, Jg = torch.func.jvp(lambda q: q + (1 - t) * net64(q, tv), (x0,),
                           (gvec(B, L, torch.float64),))
    vf_jvp = k * (gvec(B, L, torch.float64) * Jg.detach()).sum(dim=(1, 2))
    rel = float(((vf_free - vf_jvp).abs() / vf_jvp.abs()).max())
    check("<g,A> route == torch.func.jvp route", rel < 1e-12, "rel=%.1e" % rel)
    check("v_f > 0 (denominator stays positive)", bool((vf_free > 0).all()),
          "v_f/s^2 = %.4f" % (float(vf_free.mean()) / s2))

    # ---- T5  every arm reduces to plug ------------------------------------
    print("\nT5 reduction to plug in the degenerate limit")
    B, t = 32, 0.7
    xx = torch.distributions.Dirichlet(torch.ones(4)).sample((B, L))
    m32, F, A32, _ = mean_and_A(net32, xx, t)
    plug = (y - F) / s2
    for lab, sp in (("tfg_mc at sigma_mc = 0", 0.0), ("lgd_mc at r = 0", 0.0)):
        d = float((mc_coef(F, sp, 8, s2, y, 11, gn2) - plug).abs().max())
        check("%s -> plug" % lab, d <= 8e-7 * float(plug.abs().max()),
              "max|diff|=%.1e (softmax of equal logits, not bit-identical)" % d)
    ratios = []
    for tt in (0.9, 0.99, 0.999):
        _, Ft, At, _ = mean_and_A(net32, xx, tt)
        vf = ((1 - tt) ** 2 / tt) * (gvec(B, L) * At).sum(dim=(1, 2))
        ratios.append(float((s2 / (s2 + vf)).mean()))
    check("tmpd -> plug as t -> 1 (v_f -> 0)",
          ratios[0] < ratios[1] < ratios[2] and ratios[-1] > 0.9999,
          "den ratio s^2/(s^2+v_f) = %s" % ["%.6f" % r for r in ratios])

    # ---- T6  the MC arms' K -> infinity closed form -----------------------
    print("\nT6 MC arms -> (y-F)/(s^2+nu) with nu = spread^2 |g|^2 (affine f)")
    for lab, sp in (("sigma_mc = 0.35 (matched)", 0.35), ("sigma_mc = 0.02", 0.02)):
        nu = sp ** 2 * gn2
        closed = float(((y - F) / (s2 + nu)).mean())
        got = sum(float(mc_coef(F, sp, 1 << 16, s2, y, 500 + j, gn2).mean())
                  for j in range(2)) / 2
        check("%s: K=65536 matches closed form" % lab,
              abs(got / closed - 1) < 3e-3,
              "got %.5f closed %.5f rel %+.4f  nu/s^2=%.4f" % (got, closed, got / closed - 1, nu / s2))

    # ---- T7  lgd_mc in observable space IS tmpd ---------------------------
    print("\nT7 lgd_mc with spread^2 = v_f -> tmpd exactly as K -> infinity")
    vf32 = ((1 - t) ** 2 / t) * (gvec(B, L) * A32).sum(dim=(1, 2))
    tmpd = (y - F) / (s2 + vf32)
    rs = []
    for K in (4, 64, 1 << 16):
        spread = (vf32 / gn2).sqrt()            # per sample: nu_i = spread_i^2 |g|^2 = v_f,i
        got = sum(float(mc_coef(F, spread, K, s2, y, 400 + j, gn2).mean())
                  for j in range(4)) / 4
        rs.append(got / float(tmpd.mean()) - 1)
    check("K=65536 reproduces tmpd", abs(rs[-1]) < 3e-3,
          "rel by K(4,64,65536) = %s" % ["%+.4f" % r for r in rs])
    check("finite-K deviation shrinks with K", abs(rs[0]) > abs(rs[1]) > abs(rs[2]))
    check("K=4 deviation is large enough to matter", abs(rs[0]) > 0.02,
          "-> do not run the MC arms at K=4 without saying so")

    # ---- T8  tensor draws == scalar draws, exactly -------------------------
    print("\nT8 full-tensor draws == scalar draws when xi_k := <g,z_k>/|g|")
    gg = torch.Generator().manual_seed(77)
    Zs = [torch.randn((B, L, 4), generator=gg) for _ in range(64)]
    Ft = torch.stack([S.gc_soft(m32.detach() + 0.3 * z) for z in Zs])
    xi = torch.stack([(gvec(B, L) * z).sum(dim=(1, 2)) / math.sqrt(gn2) for z in Zs])
    Fs = F.unsqueeze(0) + 0.3 * math.sqrt(gn2) * xi
    check("F from the two draw forms agrees to float32 eps",
          float((Ft - Fs).abs().max()) < 2e-7,
          "max|diff|=%.1e" % float((Ft - Fs).abs().max()))

    # ---- T9  Hutchinson trace is unbiased, on a toy net -------------------
    print("\nT9 Rademacher Hutchinson tr(J) vs the exact Jacobian trace (toy net)")
    Lt, tt = 6, 0.7
    toy = S.SimplexFM(16, 2).double()
    torch.manual_seed(0)
    for p in toy.parameters():
        with torch.no_grad():
            p.copy_(torch.randn_like(p) * 0.2)
    xs = torch.distributions.Dirichlet(torch.ones(4)).sample((1, Lt)).double()
    tvt = torch.full((1,), tt, dtype=torch.float64)
    mfun = lambda q: (q + (1 - tt) * toy(q, tvt)).reshape(-1)          # noqa: E731
    Jex = torch.autograd.functional.jacobian(mfun, xs.reshape(1, Lt, 4))
    trJ = float(torch.einsum("iji", Jex.reshape(Lt * 4, Lt * 4, 1)[:, :, 0]
                             .unsqueeze(-1)).sum()) if False else float(
        Jex.reshape(Lt * 4, -1).diagonal().sum())
    gh = torch.Generator().manual_seed(4)
    est = []
    for _ in range(4000):
        zr = (torch.randint(0, 2, (1, Lt, 4), generator=gh).double() * 2 - 1)
        xr = xs.clone().requires_grad_(True)
        mr = xr + (1 - tt) * toy(xr, tvt)
        vj = torch.autograd.grad((mr * zr).sum(), xr)[0]
        est.append(float((zr * vj).sum()))
    hut = sum(est) / len(est)
    check("VJP-Hutchinson converges to tr(J)", abs(hut / trJ - 1) < 0.02,
          "hutchinson %.5f  exact %.5f  rel %+.4f" % (hut, trJ, hut / trJ - 1))

    # ---- T10  optional: m2_sweep's own guidance_field ---------------------
    print("\nT10 m2_sweep.guidance_field (skipped if the signature has moved)")
    try:
        import m2_sweep as MS
        sig = MS.guidance_field.__code__.co_varnames[:MS.guidance_field.__code__.co_argcount]
        Gp, _ = MS.guidance_field(net32, xx, t, y, s, "plug")
        exp = ((y - F) / s2).view(-1, 1, 1) * A32
        check("plug field == ((y-F)/s^2) * A", float((Gp - exp).norm() / exp.norm()) < 1e-5,
              "rel=%.1e" % float((Gp - exp).norm() / exp.norm()))
        Gb, _ = MS.guidance_field(net32, xx, t, y, s, "bdg", 0.0, s, False)
        check("bdg at eta=0 is bit-identical to plug", torch.equal(Gb, Gp))
        try:
            MS.guidance_field(net32, xx, t, y, s, "no_such_arm")
            check("unknown arm is refused", False, "it returned a field instead")
        except Exception as exc:
            check("unknown arm is refused", "no_such_arm" in str(exc) or
                  isinstance(exc, (ValueError, KeyError)), type(exc).__name__)
        print("     signature: %s" % (sig,))
    except Exception as exc:
        print("     SKIPPED: %s: %s" % (type(exc).__name__, exc))

    print("\n%d checks failed%s" % (len(FAILED), (": " + ", ".join(FAILED)) if FAILED else ""))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
