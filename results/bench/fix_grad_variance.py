"""grad_property_variance was missing the term that flows through g itself.

Caught by proj1/tests/test_v2_arms.py against an autograd reference: in a
linear-mean setting the implemented gradient is identically ZERO while the true
one is 2 k a D^2 g, so BTVG's variance half would have been a silent no-op
there and mis-scaled everywhere else. Run once.
"""
import io

p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''def grad_property_variance(f_net, post_fn, coords, feats, mask, g_c, g_f, k,
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
    return V.detach(), d_c, d_f'''

NEW = '''def grad_property_variance(f_net, post_fn, coords, feats, mask, m_c, m_f, k,
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
    return V.detach(), d_c, d_f'''

assert s.count(OLD) == 1, "anchor missed"
s = s.replace(OLD, NEW)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("grad_property_variance: Hessian term added")
