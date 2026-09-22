"""Gate the mean-map curvature term of grad V_F. Run once.

Every other gate uses m = D*x, so J is CONSTANT, the curvature term
k*grad(g'J(x)g) is identically zero, and `autograd.grad` returns None -- only
the `allow_unused` path ever runs. Deleting that term, or multiplying it by
100, both pass the full suite. It is not a small term either: with a nonlinear
mean map its norm is 29-56% of the Hessian term's.

This gate uses m = D*x + lam*x^3 elementwise. J = diag(D + 3*lam*x^2) is still
diagonal, hence symmetric, so the 2x Hessian factor remains exact and this
isolates the curvature term rather than confounding it with the symmetry
assumption.
"""
import io

p = "proj1/tests/test_v2_arms.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''    # ---------------- SPBC gives EVERY trajectory the same property increment'''
NEW = '''    # ---------------- the mean-map curvature term, on a NONLINEAR mean map
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

    # ---------------- SPBC gives EVERY trajectory the same property increment'''
assert s.count(OLD) == 1, "spbc section anchor"
s = s.replace(OLD, NEW)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("nonlinear mean-map gate added")
