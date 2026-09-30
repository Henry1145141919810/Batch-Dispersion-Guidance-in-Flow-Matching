"""Exact gates for BTVG-2 (btvg2_weighted_grad), the post-full-run revision.

Same toy as test_v2_arms.py: a diagonal posterior m = D * x and a quadratic
property, so every quantity has an autograd reference. Each gate is written so
a dropped factor, a flipped sign, or a skipped branch moves a number.

  G1  variance term off (tau^2 >= V)  => field == lgd_mc's, bit for bit
  G2  dV/dm == autograd of the sample variance over the SAME draws
  G3  orth: the variance step is orthogonal to gbar = mean_i g_i
  G4  cap <= max(|y-mu|, sqrt V)|gbar|/s^2: invariant, not crushed on
      target (the |M|-cap defect), lands exactly on the limit when binding
  G5  gate == exp(-(y - mu)^2 / (2 V)) from the draws; G5b the band gate
  G6  never widens: <step, dV> <= 0 everywhere, and b == 0 when V <= tau^2
  G7  guidance_field('btvg2') == J^T (field), and the ablation modes dispatch
  G8  strength_scale: btvg2 is NOT rescaled, btvg still is
  G9  n_mc < 2 is refused

Run: python proj1/tests/test_btvg2.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from guidance import (KNOWN_MODES, Posterior, btvg2_weighted_grad,  # noqa: E402
                      guidance_field, sigma_mc_weighted_grad, trace_scale)

sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from guidance_sweep import arm_kwargs, strength_scale  # noqa: E402

torch.set_default_dtype(torch.float64)
B, N, T, K = 5, 3, 2, 4
R = {}


class Quad(torch.nn.Module):
    def __init__(self, a_c, a_f, b_c, b_f):
        super().__init__()
        self.a_c, self.a_f, self.b_c, self.b_f = a_c, a_f, b_c, b_f

    def forward(self, coords, feats, mask):
        return (0.5 * (self.a_c * coords ** 2).sum(dim=(1, 2))
                + 0.5 * (self.a_f * feats ** 2).sum(dim=(1, 2))
                + (self.b_c * coords).sum(dim=(1, 2))
                + (self.b_f * feats).sum(dim=(1, 2)))


def setup(seed=11):
    torch.manual_seed(seed)
    mask = torch.ones(B, N)
    coords, feats = torch.randn(B, N, 3), torch.randn(B, N, T)
    D_c, D_f = torch.rand(B, N, 3) + 0.5, torch.rand(B, N, T) + 0.5
    a_c, a_f = torch.rand(B, N, 3) + 0.2, torch.rand(B, N, T) + 0.2
    b_c, b_f = torch.randn(B, N, 3), torch.randn(B, N, T)
    k = torch.full((B,), 0.8)
    f_net = Quad(a_c, a_f, b_c, b_f)

    def post_fn(c, f):
        return Posterior(D_c * c, D_f * f, k)

    return dict(mask=mask, coords=coords, feats=feats, D_c=D_c, D_f=D_f,
                k=k, f_net=f_net, post_fn=post_fn,
                m_c=D_c * coords, m_f=D_f * feats)


def gen(seed=3):
    return torch.Generator().manual_seed(seed)


def draws(E, seed=3):
    """Replay btvg2's draws: same generator calls in the same order."""
    g = gen(seed)
    r2 = trace_scale(E["post_fn"], E["coords"], E["feats"], E["mask"], E["k"],
                     1, g, com_free=True)
    r = r2.clamp(min=0).sqrt().view(-1, 1, 1)
    out = []
    for _ in range(K):
        z_c = torch.randn(E["m_c"].shape, generator=g) * E["mask"].unsqueeze(-1)
        z_f = torch.randn(E["m_f"].shape, generator=g) * E["mask"].unsqueeze(-1)
        out.append((r * z_c, r * z_f))
    return out


def call(E, y, s, tau, **kw):
    return btvg2_weighted_grad(E["f_net"], E["post_fn"], E["coords"],
                               E["feats"], E["mask"], E["m_c"], E["m_f"],
                               E["k"], y, s, tau, K, gen(), **kw)


def dot(a_c, a_f, b_c, b_f):
    return (a_c * b_c).sum(dim=(1, 2)) + (a_f * b_f).sum(dim=(1, 2))


def check(name, ok, detail=""):
    R[name] = bool(ok)
    print("%-4s %-62s %s" % ("PASS" if ok else "FAIL", name, detail))


def main():
    E = setup()
    f, mask = E["f_net"], E["mask"]
    y = torch.randn(B) * 3.0
    s = 1.3

    # lgd_mc's own field with the same generator
    L_c, L_f, _ = sigma_mc_weighted_grad(f, E["post_fn"], E["coords"],
                                         E["feats"], mask, E["m_c"], E["m_f"],
                                         E["k"], y, s, K, gen())

    # G1: tau huge => b = 0 => btvg2 == lgd_mc exactly
    c1, f1, _ = call(E, y, s, tau=1e6)
    check("G1 variance off => field == lgd_mc bit for bit",
          torch.equal(c1, L_c) and torch.equal(f1, L_f))

    # autograd reference over the replayed draws
    D = draws(E)
    m_c = E["m_c"].clone().requires_grad_(True)
    m_f = E["m_f"].clone().requires_grad_(True)
    Fs = torch.stack([f(m_c + dc, m_f + df, mask) for dc, df in D])
    Vref = Fs.var(0, unbiased=True)
    dVc_ref, dVf_ref = torch.autograd.grad(Vref.sum(), (m_c, m_f))
    Fd = Fs.detach()
    mu_ref, V_ref = Fd.mean(0), Fd.var(0, unbiased=True)
    gs = []
    for dc, df in D:
        mc = (E["m_c"] + dc).requires_grad_(True)
        mf = (E["m_f"] + df).requires_grad_(True)
        gs.append(torch.autograd.grad(f(mc, mf, mask).sum(), (mc, mf)))
    gb_c = torch.stack([g[0] for g in gs]).mean(0)
    gb_f = torch.stack([g[1] for g in gs]).mean(0)

    # G2: tau tiny, no gate/orth/cap => step = -1/2 (1/s^2) dV exactly
    tiny = 1e-9
    c2, f2, _ = call(E, y, s, tau=tiny, gate=False, orth=False, cap=None)
    b_ref = 0.5 / s ** 2 * (1 - tiny ** 2 / V_ref)
    dVc = -(c2 - L_c) / b_ref.view(-1, 1, 1)
    dVf = -(f2 - L_f) / b_ref.view(-1, 1, 1)
    err = max((dVc - dVc_ref).abs().max().item(), (dVf - dVf_ref).abs().max().item())
    check("G2 dV/dm == autograd of var_i f(m + r z_i)", err < 1e-9,
          "max err %.2e" % err)

    # G3: orth => <step, gbar> = 0, and it actually removed something
    c3, f3, _ = call(E, y, s, tau=tiny, gate=False, orth=True, cap=None)
    S_c, S_f = c3 - L_c, f3 - L_f
    rel = (dot(S_c, S_f, gb_c, gb_f).abs()
           / (dot(S_c, S_f, S_c, S_f).sqrt() * dot(gb_c, gb_f, gb_c, gb_f).sqrt()))
    raw = (dot(dVc_ref, dVf_ref, gb_c, gb_f).abs()
           / (dot(dVc_ref, dVf_ref, dVc_ref, dVf_ref).sqrt()
              * dot(gb_c, gb_f, gb_c, gb_f).sqrt()))
    check("G3 orth: cos(step, gbar) == 0 (raw dV was not orthogonal)",
          rel.max().item() < 1e-10 and raw.min().item() > 1e-3,
          "cos after %.1e, before >= %.3f" % (rel.max().item(), raw.min().item()))

    # G4: cap = max(|y - mu|, sqrt V) |gbar| / s^2. Three checks:
    #  (a) the invariant |S| <= lim holds against a reference lim;
    #  (b) ON TARGET the step is NOT crushed to |M| (the defect of a |M| cap);
    #  (c) a small cap multiplier binds on every sample, exactly at the limit.
    def lim_ref(yy, Vr, mur, gbc, gbf):
        return (torch.maximum((yy - mur).abs(), Vr.sqrt())
                * dot(gbc, gbf, gbc, gbf).sqrt() / s ** 2)
    c4, f4, dg4 = call(E, y, s, tau=tiny, gate=False, orth=True, cap=1.0)
    nS = dot(c4 - L_c, f4 - L_f, c4 - L_c, f4 - L_f).sqrt()
    ok_a = bool((nS <= lim_ref(y, V_ref, mu_ref, gb_c, gb_f) * (1 + 1e-9)).all())
    y_on = mu_ref.clone()
    Mn_c, Mn_f, _ = sigma_mc_weighted_grad(f, E["post_fn"], E["coords"],
                                           E["feats"], mask, E["m_c"], E["m_f"],
                                           E["k"], y_on, s, K, gen())
    c4b, f4b, _ = call(E, y_on, s, tau=tiny, gate=False, orth=True, cap=1.0)
    nSb = dot(c4b - Mn_c, f4b - Mn_f, c4b - Mn_c, f4b - Mn_f).sqrt()
    nMb = dot(Mn_c, Mn_f, Mn_c, Mn_f).sqrt()
    ok_b = bool((nSb > nMb).any()) and bool(
        (nSb <= lim_ref(y_on, V_ref, mu_ref, gb_c, gb_f) * (1 + 1e-9)).all())
    # (c) a tiny cap multiplier must bind everywhere and land EXACTLY on it
    c4c, f4c, dg4c = call(E, y, s, tau=tiny, gate=False, orth=True, cap=1e-3)
    nSc = dot(c4c - L_c, f4c - L_f, c4c - L_c, f4c - L_f).sqrt()
    limc = 1e-3 * lim_ref(y, V_ref, mu_ref, gb_c, gb_f)
    ok_c = (bool(dg4c["btvg2_capped"].all())
            and torch.allclose(nSc, limc, rtol=1e-9, atol=0))
    check("G4 cap: invariant; not crushed on target; exact when binding",
          ok_a and ok_b and ok_c,
          "a=%s b=%s (on-target |S|/|M| max %.1f) c=%s"
          % (ok_a, ok_b, (nSb / nMb).max().item(), ok_c))

    # G5: gate value
    _, _, dg5 = call(E, y, s, tau=tiny)
    g_ref = torch.exp(-((y - mu_ref) ** 2) / (2 * V_ref))
    y_b = mu_ref + torch.tensor([0.0, 0.2, 0.5, 1.0, -0.4])
    _, _, dg5b = call(E, y_b, s, tau=0.3, gate="band")
    gb_ref = torch.exp(-((y_b - mu_ref) ** 2) / (2 * (1.96 * 0.3) ** 2))
    g_pl = torch.exp(-((y_b - mu_ref) ** 2) / (2 * V_ref))
    check("G5b band gate == exp(-(y-mu)^2 / 2 (1.96 tau)^2)",
          torch.allclose(dg5b["btvg2_gate"], gb_ref, rtol=1e-10, atol=0)
          and bool(((gb_ref > 0.05) & (gb_ref <= 1)).all())
          and not torch.allclose(gb_ref, g_pl),
          "gate %s" % [round(x, 3) for x in gb_ref.tolist()])
    check("G5 gate == exp(-(y-mu)^2 / 2V)",
          torch.allclose(dg5["btvg2_gate"], g_ref, rtol=1e-10, atol=0),
          "gate %s" % [round(x, 3) for x in g_ref.tolist()])

    # G6: never widens; b = 0 when V <= tau^2
    c6, f6, _ = call(E, y, s, tau=tiny, gate=True, orth=True, cap=1.0)
    S_c, S_f = c6 - L_c, f6 - L_f
    widen = dot(S_c, S_f, dVc_ref, dVf_ref)
    tau_mid = float(V_ref.median().sqrt())
    c6b, f6b, _ = call(E, y, s, tau=tau_mid, gate=False, orth=False, cap=None)
    low = V_ref <= tau_mid ** 2
    off = ((c6b - L_c)[low].abs().max().item() == 0.0
           and (f6b - L_f)[low].abs().max().item() == 0.0)
    check("G6 <step, dV> <= 0 always; step == 0 where V <= tau^2",
          bool((widen <= 1e-12).all()) and off and bool(low.any()),
          "max <S,dV> %.2e, %d samples at V<=tau^2" % (widen.max().item(), int(low.sum())))

    # G7: guidance_field pulls back with J^T (= D here) and dispatches ablations
    ok7 = True
    for mode, kw in (("btvg2", {}), ("btvg2_nogate", {"gate": False}),
                     ("btvg2_band", {"gate": "band"}),
                     ("btvg2_noorth", {"orth": False}), ("btvg2_nocap", {"cap": None})):
        Gc, Gf, _ = guidance_field(f, E["post_fn"], E["coords"], E["feats"], mask,
                                   y, s, mode=mode, n_mc=K, generator=gen(),
                                   tau=0.05)
        wc, wf, _ = call(E, y, s, tau=0.05, **kw)
        ok7 &= torch.allclose(Gc, E["D_c"] * wc, rtol=1e-10, atol=1e-12)
        ok7 &= torch.allclose(Gf, E["D_f"] * wf, rtol=1e-10, atol=1e-12)
    check("G7 guidance_field == J^T field, all five modes dispatch", ok7)

    # G8: strength normalisation
    kw = arm_kwargs("btvg2", "tgt", 0.2)
    ok8 = (abs(kw["tau"] - 0.2 / 1.96) < 1e-15
           and strength_scale("btvg2", kw, 1.5) == 1.0
           and strength_scale("btvg2_nogate", kw, 1.5) == 1.0
           and abs(strength_scale("btvg", kw, 1.5) - (kw["tau"] / 1.5) ** 2) < 1e-15
           and {"btvg2", "btvg2_nogate", "btvg2_noorth", "btvg2_nocap", "btvg2_band"} <= KNOWN_MODES)
    check("G8 btvg2 not rescaled, btvg still (tau/s)^2, modes registered", ok8)

    # G9: n_mc < 2 is refused, not silently bumped
    try:
        btvg2_weighted_grad(f, E["post_fn"], E["coords"], E["feats"], mask,
                            E["m_c"], E["m_f"], E["k"], y, s, 0.05, 1, gen())
        ok9 = False
    except ValueError:
        ok9 = True
    check("G9 n_mc < 2 raises instead of drawing a different K", ok9)

    ok = all(R.values())
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
