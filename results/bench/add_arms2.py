"""Add the Tolerance Band (D) and RCH arms to guidance.py. Run once."""
import io

p = "proj1/src/guidance.py"
s = io.open(p, encoding="utf-8").read()

NEW = '''

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
    def features(coords, feats, mask, t, k):
        """E(3)-invariant summary statistics. Rotation/translation invariant by
        construction: only norms, pairwise distances and type histograms."""
        m = mask.unsqueeze(-1)
        n = mask.sum(1).clamp(min=1.0)
        c = coords * m
        r2 = (c ** 2).sum(-1)
        d = torch.cdist(c, c) + (1 - mask).unsqueeze(1) * 1e3
        dmin = d.masked_fill(torch.eye(d.shape[-1], device=d.device,
                                       dtype=torch.bool), 1e3).min(-1).values
        dmin = (dmin * mask).sum(1) / n
        tv = t.view(-1)
        kk = torch.as_tensor(float(k), device=coords.device).expand_as(tv)
        hist = (feats * m).sum(1) / n.unsqueeze(-1)
        base = [torch.ones_like(tv), tv, tv ** 2, tv ** 3, (1 - tv),
                (1 - tv) ** 2, kk, kk * tv, n / 29.0,
                (r2 * mask).sum(1) / n, ((r2 * mask).sum(1) / n).sqrt(),
                dmin, dmin ** 2,
                (feats ** 2 * m).sum((1, 2)) / n,
                (c.abs() * m).sum((1, 2)) / n]
        cols = base + [hist[:, i] for i in range(hist.shape[1])]
        cols = cols + [hist[:, i] * tv for i in range(hist.shape[1])]
        X = torch.stack(cols, dim=1)
        want = ResidualCalibrationHead.N_FEAT
        if X.shape[1] < want:
            X = torch.cat([X, torch.zeros(X.shape[0], want - X.shape[1],
                                          device=X.device, dtype=X.dtype)], 1)
        return X[:, :want]

    def forward(self, coords, feats, mask, t, k):
        phi = self.features(coords, feats, mask, t, k)
        return (1.0 - t.view(-1)) * (phi * self.beta).sum(1)
'''

anchor = "\n\n# --------------------------------------------------------------------------\n# the fields\n# --------------------------------------------------------------------------"
assert s.count(anchor) == 1
s = s.replace(anchor, NEW + anchor)

# dispatch: rch and band
s = s.replace('''                   mode="plug", n_probe=1, generator=None, cost=_NOCOST,
                   n_mc=4, sigma_mc=0.1, want_kappa3=False):''',
'''                   mode="plug", n_probe=1, generator=None, cost=_NOCOST,
                   n_mc=4, sigma_mc=0.1, want_kappa3=False, rch=None,
                   band_tau=None, band_eta=1.0, band_radius=None, t_scalar=None):''')

s = s.replace('''    if mode in ("lgd_mc", "osc"):''',
'''    if mode == "rch":
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
        # Component D. The unconstrained edit is the plug-in direction; the
        # band constrains its first-order property change.
        e = fval - y
        tau = (band_tau if torch.is_tensor(band_tau)
               else torch.full_like(fval, float(band_tau if band_tau is not None else 0.0)))
        u_c, u_f = _pullback(post_fn, coords, feats, g_c, g_f, cost)
        d_c, d_f = tolerance_band_step(u_c, u_f, u_c, u_f, e, tau,
                                       band_eta, band_radius)
        return d_c.detach(), d_f.detach(), {"f": fval, "c": torch.zeros_like(fval),
                                            "v_f": torch.zeros_like(fval), "k": k,
                                            "band_e": e, "band_tau": tau}

    if mode in ("lgd_mc", "osc"):''')

s = s.replace('''      "osc"         observable-space closure: same guide budget as lgd_mc, but
                    the scalar observable's law is integrated analytically''',
'''      "osc"         observable-space closure: same guide budget as lgd_mc, but
                    the scalar observable's law is integrated analytically
      "rch"         Haimo v1's Residual Calibration Head: a LEARNED stand-in for
                    c at plug-in cost. Needs a fitted head via rch=.
      "band"        Component D, the tolerance band: the largest diversifying
                    edit whose first-order property change stays in the band''')

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("guidance.py: added band (Component D) and rch (Haimo v1)")
