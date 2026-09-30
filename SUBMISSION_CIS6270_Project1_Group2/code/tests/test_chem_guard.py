"""Exact gates for chemistry-safe guidance (soft_valence_violation,
chem_safe_project, and the `*_chem` modes in guidance_field).

  C1  sharp limit == the evaluator: with kappa -> 0 and one-hot types, the
      soft valence equals evaluation.stability's integer valence atom for
      atom, on real QM9 molecules AND on jittered (partly unstable) ones
  C2  grad P == central finite differences at the shipped kappa
  C3  projection: <a, G'> <= 0 always; untouched where <a, G> <= 0; the
      exact formula where active; `removed` is |G - G'| / |G|
  C4  first order, for real: the directional derivative of P is > 0
      along G where active, and <= 0 along G' everywhere
  C5  `lgd_mc_chem` == chem_safe_project(lgd_mc field, J^T grad P), i.e.
      the wrapper changes nothing but the projection; same for plug_chem
  C5n the norm-preserving variant keeps the base length and chem's direction
  C6  registration: modes known, CHEM_BASE correct, no strength rescaling

Run: python proj1/tests/test_chem_guard.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from evaluation import ALLOWED_VALENCE, stability  # noqa: E402
from guidance import (CHEM_BASE, KNOWN_MODES, Posterior,  # noqa: E402
                      _pullback, chem_safe_project, guidance_field,
                      soft_valence_violation)
from guidance_sweep import strength_scale  # noqa: E402

torch.set_default_dtype(torch.float64)
R = {}


def check(name, ok, detail=""):
    R[name] = bool(ok)
    print("%-4s %-66s %s" % ("PASS" if ok else "FAIL", name, detail))


def real_molecules(n=256, seed=0):
    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    idx = d["split"]["val"][:n]
    return (d["coords"][idx].double(), d["feats"][idx].double(),
            d["mask"][idx].double(), d["types"])


def evaluator_valence_match(coords, feats, mask, types):
    """Per-atom agreement between soft (sharp limit) and the evaluator."""
    viol = soft_valence_violation(coords, feats, mask, kappa=1e-5, temp=1e-4,
                                  per_atom=True)
    soft_stable = (viol < 0.25) & mask.bool()          # |v - A| < 0.5
    ev = stability(coords.float(), feats.float(), mask.float(), types)
    agree, total, n_unstable = 0, 0, 0
    for b, (n_st, n, _, bonds, syms) in enumerate(ev):
        val = [0] * n
        for (i, j), o in bonds.items():
            val[i] += o
            val[j] += o
        for i in range(n):
            ev_ok = val[i] == ALLOWED_VALENCE.get(syms[i], -1)
            agree += int(ev_ok == bool(soft_stable[b, i]))
            total += 1
            n_unstable += int(not ev_ok)
    return agree, total, n_unstable


class Lin(torch.nn.Module):
    """A property with a known gradient, so the base arms are exact."""
    def __init__(self, w_c, w_f):
        super().__init__()
        self.w_c, self.w_f = w_c, w_f

    def forward(self, c, f, mask):
        return (self.w_c * c).sum(dim=(1, 2)) + (self.w_f * f).sum(dim=(1, 2))


def main():
    coords, feats, mask, types = real_molecules()

    # C1 on real molecules and on jittered ones
    a1, t1, u1 = evaluator_valence_match(coords, feats, mask, types)
    g = torch.Generator().manual_seed(1)
    jit = coords + 0.08 * torch.randn(coords.shape, generator=g) * mask.unsqueeze(-1)
    a2, t2, u2 = evaluator_valence_match(jit, feats, mask, types)
    check("C1 sharp-limit soft valence == evaluator, real + jittered",
          a1 == t1 and a2 == t2 and u2 > 50,
          "real %d/%d  jittered %d/%d (%d unstable atoms)" % (a1, t1, a2, t2, u2))

    # C2 gradient vs central differences at the shipped kappa
    B = 8
    c = jit[:B].clone().requires_grad_(True)
    f = (feats[:B] * 0.8 + 0.05).clone().requires_grad_(True)
    m = mask[:B]
    P = soft_valence_violation(c, f, m)
    gc, gf = torch.autograd.grad(P.sum(), (c, f))
    dc = torch.randn(c.shape, generator=g) * m.unsqueeze(-1)
    df = torch.randn(f.shape, generator=g) * m.unsqueeze(-1)
    h = 1e-6
    with torch.no_grad():
        fd = ((soft_valence_violation(c + h * dc, f + h * df, m)
               - soft_valence_violation(c - h * dc, f - h * df, m)) / (2 * h))
    ad = (gc * dc).sum(dim=(1, 2)) + (gf * df).sum(dim=(1, 2))
    rel = ((fd - ad).abs() / ad.abs().clamp(min=1e-8)).max().item()
    check("C2 grad P == central finite differences", rel < 1e-5,
          "max rel err %.1e" % rel)

    # C3 projection algebra on random vectors
    G_c, G_f = torch.randn(64, 5, 3, generator=g), torch.randn(64, 5, 5, generator=g)
    A_c, A_f = torch.randn(64, 5, 3, generator=g), torch.randn(64, 5, 5, generator=g)
    P_c, P_f, act, rem = chem_safe_project(G_c, G_f, A_c, A_f)
    dot = lambda x1, y1, x2, y2: (x1 * x2).sum(dim=(1, 2)) + (y1 * y2).sum(dim=(1, 2))
    after = dot(A_c, A_f, P_c, P_f)
    before = dot(A_c, A_f, G_c, G_f)
    untouched = torch.equal(P_c[~act], G_c[~act]) and torch.equal(P_f[~act], G_f[~act])
    k = (before / dot(A_c, A_f, A_c, A_f)).view(-1, 1, 1)
    exact = torch.allclose(P_c[act], (G_c - k * A_c)[act]) and torch.allclose(
        P_f[act], (G_f - k * A_f)[act])
    nG = dot(G_c, G_f, G_c, G_f).sqrt()
    rem_ref = dot(G_c - P_c, G_f - P_f, G_c - P_c, G_f - P_f).sqrt() / nG
    check("C3 <a,G'> <= 0; untouched if <a,G> <= 0; exact formula; removed",
          bool((after <= 1e-12).all()) and untouched and exact
          and bool(act.any()) and bool((~act).any())
          and torch.allclose(rem, rem_ref),
          "%d active of 64" % int(act.sum()))

    # C4 first-order behaviour on real molecules through a toy posterior
    # m = D * x (J = diag D), with the guidance step taken in x
    Bm = 32
    x_c, x_f = jit[:Bm].clone(), (feats[:Bm] * 0.8 + 0.05).clone()
    mk = mask[:Bm]
    D_c = 0.9 + 0.2 * torch.rand(x_c.shape, generator=g)
    D_f = 0.9 + 0.2 * torch.rand(x_f.shape, generator=g)
    kk = torch.full((Bm,), 0.5)

    def post_fn(cc, ff):
        return Posterior(D_c * cc, D_f * ff, kk)

    Gx_c = torch.randn(x_c.shape, generator=g) * mk.unsqueeze(-1)
    Gx_f = torch.randn(x_f.shape, generator=g) * mk.unsqueeze(-1)
    mc = (D_c * x_c).requires_grad_(True)
    mf = (D_f * x_f).requires_grad_(True)
    Pm = soft_valence_violation(mc, mf, mk)
    cc_, cf_ = torch.autograd.grad(Pm.sum(), (mc, mf))
    a_c, a_f = _pullback(post_fn, x_c, x_f, cc_, cf_)
    Q_c, Q_f, act4, _ = chem_safe_project(Gx_c, Gx_f, a_c, a_f)

    def P_at(xc, xf):
        return soft_valence_violation(D_c * xc, D_f * xf, mk)

    # CENTRAL differences = the directional derivative (the first-order
    # change); a forward difference is swamped by the sigmoid's curvature
    # (1/kappa^2 ~ 1e3 per A^2) and measured that instead
    eps = 1e-6

    def ddir(vc, vf):
        return (P_at(x_c + eps * vc, x_f + eps * vf)
                - P_at(x_c - eps * vc, x_f - eps * vf)) / (2 * eps)

    rise_G = ddir(Gx_c, Gx_f)
    rise_Q = ddir(Q_c, Q_f)
    ok4 = (bool(act4.any())
           and bool((rise_G[act4] > 0).all())
           and bool((rise_Q <= 1e-6 * rise_G.abs().max()).all()))
    check("C4 first-order change of P: > 0 along G (active), <= 0 along G'", ok4,
          "active %d/%d; max dP along G %.3g, along G' %.2g"
          % (int(act4.sum()), Bm, rise_G.max().item(), rise_Q.max().item()))

    # C5 the wrapper == base field + projection, for lgd_mc and plug
    w_c = torch.randn(x_c.shape, generator=g) * 0.1
    w_f = torch.randn(x_f.shape, generator=g) * 0.1
    f_net = Lin(w_c, w_f)
    y = torch.randn(Bm, generator=g)
    ok5 = True
    for mode, base_mode in (("lgd_mc_chem", "lgd_mc"), ("plug_chem", "plug")):
        Gb_c, Gb_f, _ = guidance_field(f_net, post_fn, x_c, x_f, mk, y, 1.3,
                                       mode=base_mode, n_mc=4,
                                       generator=torch.Generator().manual_seed(5))
        Gw_c, Gw_f, dgw = guidance_field(f_net, post_fn, x_c, x_f, mk, y, 1.3,
                                         mode=mode, n_mc=4,
                                         generator=torch.Generator().manual_seed(5))
        R_c, R_f, act5, _ = chem_safe_project(Gb_c, Gb_f, a_c, a_f)
        ok5 &= torch.allclose(Gw_c, R_c, rtol=1e-10, atol=1e-12)
        ok5 &= torch.allclose(Gw_f, R_f, rtol=1e-10, atol=1e-12)
        ok5 &= torch.equal(dgw["chem_active"].bool(), act5)
    check("C5 X_chem == project(X field, J^T grad P) for lgd_mc and plug", ok5)

    # C5n norm-preserving variant: same direction as lgd_mc_chem, length of the
    # base step, and still <a, G'> <= 0
    Gb_c, Gb_f, _ = guidance_field(f_net, post_fn, x_c, x_f, mk, y, 1.3,
                                   mode="lgd_mc", n_mc=4,
                                   generator=torch.Generator().manual_seed(5))
    Gp_c, Gp_f, _ = guidance_field(f_net, post_fn, x_c, x_f, mk, y, 1.3,
                                   mode="lgd_mc_chem", n_mc=4,
                                   generator=torch.Generator().manual_seed(5))
    Gn_c, Gn_f, _ = guidance_field(f_net, post_fn, x_c, x_f, mk, y, 1.3,
                                   mode="lgd_mc_chemn", n_mc=4,
                                   generator=torch.Generator().manual_seed(5))
    nb = dot(Gb_c, Gb_f, Gb_c, Gb_f).sqrt()
    nn_ = dot(Gn_c, Gn_f, Gn_c, Gn_f).sqrt()
    npj = dot(Gp_c, Gp_f, Gp_c, Gp_f).sqrt()
    cosn = dot(Gn_c, Gn_f, Gp_c, Gp_f) / (nn_ * npj)
    shrunk = npj < nb * (1 - 1e-6)
    ok5n = (torch.allclose(nn_, nb, rtol=1e-10)
            and bool((cosn > 1 - 1e-10).all())
            and bool((dot(a_c, a_f, Gn_c, Gn_f) <= 1e-10).all())
            and bool(shrunk.any()))
    check("C5n chemn: |G'| == |base|, direction == chem's, <a,G'> <= 0", ok5n,
          "%d/%d steps were shrunk by the plain guard" % (int(shrunk.sum()), Bm))

    # C6 registration
    ok6 = ({"lgd_mc_chem", "plug_chem", "lgd_mc_chemn"} <= KNOWN_MODES
           and CHEM_BASE == {"lgd_mc_chem": "lgd_mc", "plug_chem": "plug",
                             "lgd_mc_chemn": "lgd_mc"}
           and strength_scale("lgd_mc_chem", {}, 1.5) == 1.0
           and strength_scale("plug_chem", {}, 1.5) == 1.0)
    check("C6 modes registered, bases correct, no strength rescaling", ok6)

    ok = all(R.values())
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
