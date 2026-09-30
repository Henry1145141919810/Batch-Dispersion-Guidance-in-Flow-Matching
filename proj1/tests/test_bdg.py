"""Exact gates for BDG (Batch-Dispersion Guidance), ported from
docs/methods/BDG_HANDOFF.md section 2.

Two halves:

  PART A -- a diagonal analytic setting (same harness as test_v2_arms.py), so
  every quantity has a closed form and a dropped factor or a wrong sign moves
  a number.

  PART B -- THE REAL GENERATOR AND THE REAL GUIDE, at a real x_t taken from an
  actual unguided trajectory. Part A cannot catch a plumbing error (a kwarg
  that never reaches guidance_field, a diagnostic that never reaches the cell),
  and plumbing is where this project has been bitten before: `band` reported an
  identical number at every strength because its direction was dead code, and
  `osc`'s mechanism was unfalsifiable because its diagnostics were never in
  DIAG_KEYS. Part B is skipped if the weights are absent.

The four gates the port had to clear before any sampling run:
  (a) eta = 0 is BIT-IDENTICAL to plug on a real batch
  (b) the section-3 factorisation holds on the real field:
      BDG == plug at weight (1 + eta e) aiming at y_eff
  (c) the one-sided variant ASKED TO WIDEN reproduces plug exactly
  (d) the existing arms still pass (run test_v2_arms.py and test_tune.py)

Run: python proj1/tests/test_bdg.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
from guidance import (DISPLACEMENT_MODES, KNOWN_MODES, Cost,  # noqa: E402
                      Posterior, fm_posterior, guidance_field)
from sampling import FlowSampler, initial_noise, integrate  # noqa: E402

R = {}


def chk(name, ok, detail=""):
    R[name] = bool(ok)
    print("  %-52s %s   %s" % (name, "PASS" if ok else "FAIL", detail))


# --------------------------------------------------------------------------
# PART A: analytic, diagonal, float64
# --------------------------------------------------------------------------
B, N, K = 6, 3, 2


class Quad(torch.nn.Module):
    """f(m) = 1/2 sum a m^2 + sum b m -- grad is known in closed form."""

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
    g_c, g_f = a_c * m_c + b_c, a_f * m_f + b_f
    return dict(mask=mask, coords=coords, feats=feats, D_c=D_c, D_f=D_f,
                f_net=f_net, post_fn=post_fn, m_c=m_c, m_f=m_f,
                g_c=g_c, g_f=g_f)


def part_a():
    torch.set_default_dtype(torch.float64)
    E = setup()
    mask, coords, feats = E["mask"], E["coords"], E["feats"]
    f_net, post_fn, D_c, D_f = E["f_net"], E["post_fn"], E["D_c"], E["D_f"]
    m_c, m_f, g_c, g_f = E["m_c"], E["m_f"], E["g_c"], E["g_f"]
    s = 1.7
    y = torch.linspace(-1.0, 2.0, B)
    F = f_net(m_c, m_f, mask)                      # the predicted property
    eta, tau = 4.0, 0.9

    def field(mode, **kw):
        return guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                              mode=mode, **kw)

    print("\nPART A -- analytic diagonal setting (float64)")
    chk("A0 'bdg' is a known mode", "bdg" in KNOWN_MODES)

    # closed form: J = diag(D), so the pullback is elementwise D *
    F_bar = F.mean()
    V_b = F.var(unbiased=True)
    e = (V_b - tau ** 2) / tau ** 2
    num = (y - F) - eta * e * (F - F_bar)
    ref_c = (num / s ** 2).view(-1, 1, 1) * g_c * D_c
    ref_f = (num / s ** 2).view(-1, 1, 1) * g_f * D_f

    G_c, G_f, dg = field("bdg", bdg_eta=eta, bdg_tau=tau)
    err = max(float((G_c - ref_c).abs().max()), float((G_f - ref_f).abs().max()))
    chk("A1 field equals the closed form", err < 1e-12, "max err %.2e" % err)

    # the dispersion term IS the gradient of the batch variance: check
    # dV_b/dm_i against autograd on the real V_b, not against our own algebra
    mc = m_c.detach().clone().requires_grad_(True)
    mf = m_f.detach().clone().requires_grad_(True)
    Vb_auto = f_net(mc, mf, mask).var(unbiased=True)
    dc, df = torch.autograd.grad(Vb_auto, (mc, mf))
    ana_c = (2.0 / (B - 1)) * (F - F_bar).view(-1, 1, 1) * g_c
    ana_f = (2.0 / (B - 1)) * (F - F_bar).view(-1, 1, 1) * g_f
    derr = max(float((dc - ana_c).abs().max()), float((df - ana_f).abs().max()))
    chk("A2 dispersion direction == autograd grad V_b", derr < 1e-12,
        "max err %.2e" % derr)

    # e = 0 exactly -> the term vanishes -> plug. tau^2 = V_b by construction.
    tau_fp = float(V_b.sqrt())
    Gc0, Gf0, _ = field("bdg", bdg_eta=eta, bdg_tau=tau_fp)
    Pc, Pf, _ = field("plug")
    fp = max(float((Gc0 - Pc).abs().max()), float((Gf0 - Pf).abs().max()))
    chk("A3 tau^2 == V_b (e=0) reduces to plug", fp < 1e-12, "max err %.2e" % fp)

    # signs: V_b > tau^2 must push high-F molecules DOWN relative to plug
    Gc_t, _, _ = field("bdg", bdg_eta=eta, bdg_tau=0.3 * tau_fp)     # tighten
    Gc_w, _, _ = field("bdg", bdg_eta=eta, bdg_tau=3.0 * tau_fp)     # widen
    hi = int(F.argmax())
    # project the CHANGE from plug onto that molecule's own response direction
    a_hi = (g_c * D_c)[hi]
    d_t = float(((Gc_t - Pc)[hi] * a_hi).sum())
    d_w = float(((Gc_w - Pc)[hi] * a_hi).sum())
    chk("A4 V_b>tau^2 pushes the top molecule DOWN", d_t < 0, "<dG,a> %.4g" % d_t)
    chk("A5 V_b<tau^2 pushes the top molecule UP", d_w > 0, "<dG,a> %.4g" % d_w)

    # one-sided: the positive part. Asked to widen it must do nothing.
    Gc_o, Gf_o, dgo = field("bdg", bdg_eta=eta, bdg_tau=3.0 * tau_fp,
                            bdg_onesided=True)
    oer = max(float((Gc_o - Pc).abs().max()), float((Gf_o - Pf).abs().max()))
    chk("A6 one-sided asked to widen == plug exactly", oer == 0.0,
        "max err %.2e" % oer)
    chk("A7 one-sided still records the RAW e (open item 4)",
        float(dgo["bdg_e_raw"].mean()) < -0.5 and float(dgo["bdg_e"].mean()) == 0.0,
        "e_raw %.4f  e %.4f" % (float(dgo["bdg_e_raw"].mean()),
                                float(dgo["bdg_e"].mean())))
    # ...and asked to TIGHTEN it must equal the two-sided arm
    Gc_o2, _, _ = field("bdg", bdg_eta=eta, bdg_tau=0.3 * tau_fp,
                        bdg_onesided=True)
    chk("A8 one-sided asked to tighten == two-sided",
        float((Gc_o2 - Gc_t).abs().max()) == 0.0)

    # eta = 0 is bit-identical to plug (analytic half of gate (a))
    Gc_e0, Gf_e0, _ = field("bdg", bdg_eta=0.0, bdg_tau=tau)
    z = max(float((Gc_e0 - Pc).abs().max()), float((Gf_e0 - Pf).abs().max()))
    chk("A9 eta=0 is BIT-identical to plug", z == 0.0, "max err %.2e" % z)

    # the section-3 factorisation, analytic
    w_eff = 1.0 + eta * e
    y_eff = (y + eta * e * F_bar) / w_eff
    Gc_r, Gf_r, _ = guidance_field(f_net, post_fn, coords, feats, mask, y_eff,
                                   s, mode="plug")
    rerr = max(float((G_c - w_eff * Gc_r).abs().max()),
               float((G_f - w_eff * Gf_r).abs().max()))
    scale = float(G_c.abs().max())
    chk("A10 reduction: BDG == (1+eta e) * plug(y_eff)", rerr / scale < 1e-12,
        "rel err %.2e" % (rerr / scale))

    # cost: zero extra NFE over plug
    cb, cp = Cost(), Cost()
    guidance_field(f_net, post_fn, coords, feats, mask, y, s, mode="bdg",
                   cost=cb, bdg_eta=eta, bdg_tau=tau)
    guidance_field(f_net, post_fn, coords, feats, mask, y, s, mode="plug",
                   cost=cp)
    same = all(getattr(cb, k) == getattr(cp, k) for k in
               ("gen_fwd", "gen_vjp", "gen_jvp", "guide_fwd", "guide_bwd",
                "guide_hvp"))
    chk("A11 cost dict identical to plug", same,
        "gen_fwd %d/%d guide_bwd %d/%d" % (cb.gen_fwd, cp.gen_fwd,
                                           cb.guide_bwd, cp.guide_bwd))

    # batch permutation: V_b and F_bar are symmetric, so the field must permute
    perm = torch.randperm(B)
    Gp_c, _, _ = guidance_field(f_net, post_fn, coords, feats, mask, y, s,
                                mode="bdg", bdg_eta=eta, bdg_tau=tau)
    fq = Quad(f_net.a_c[perm], f_net.a_f[perm], f_net.b_c[perm], f_net.b_f[perm])

    def post_q(c, f):
        return Posterior(D_c[perm] * c, D_f[perm] * f, torch.full((B,), 0.8))

    Gq_c, _, _ = guidance_field(fq, post_q, coords[perm], feats[perm],
                                mask[perm], y[perm], s, mode="bdg",
                                bdg_eta=eta, bdg_tau=tau)
    perr = float((Gq_c - Gp_c[perm]).abs().max())
    chk("A12 equivariant under batch permutation", perr < 1e-12,
        "max err %.2e" % perr)

    # THE BATCH IS THE ESTIMATOR -- this is a property, not a bug. The gate
    # records it so the batch-dependence is asserted rather than assumed: a cell
    # run at a different `batch` is a different controller, which is why v3 pins
    # batch = 500 alongside n = 2000 instead of leaving it to the operator.
    def sub(j):
        """The same problem restricted to the first j trajectories."""
        fq = Quad(f_net.a_c[:j], f_net.a_f[:j], f_net.b_c[:j], f_net.b_f[:j])

        def pq(c, f):
            return Posterior(D_c[:j] * c, D_f[:j] * f, torch.full((j,), 0.8))

        return fq, pq

    fq3, pq3 = sub(3)
    Gh_c, _, dh = guidance_field(fq3, pq3, coords[:3], feats[:3], mask[:3],
                                 y[:3], s, mode="bdg", bdg_eta=eta, bdg_tau=tau)
    diff = float((Gh_c - G_c[:3]).abs().max())
    chk("A13 half-batch gives a DIFFERENT field (batch is the estimator)",
        diff > 1e-6, "max diff %.3g, V_b %.4g vs %.4g"
        % (diff, float(dh["bdg_V_b"].mean()), float(V_b)))

    # B = 1: the unbiased variance is undefined; degrade to plug, not to nan
    fq1, pq1 = sub(1)
    G1_c, _, _ = guidance_field(fq1, pq1, coords[:1], feats[:1], mask[:1],
                                y[:1], s, mode="bdg", bdg_eta=eta, bdg_tau=tau)
    P1_c, _, _ = guidance_field(fq1, pq1, coords[:1], feats[:1], mask[:1],
                                y[:1], s, mode="plug")
    chk("A14 B=1 degrades to plug, no nan",
        bool(torch.isfinite(G1_c).all())
        and float((G1_c - P1_c).abs().max()) == 0.0)

    # a missing setpoint must raise, not default
    try:
        guidance_field(f_net, post_fn, coords, feats, mask, y, s, mode="bdg",
                       bdg_eta=eta)
        ok = False
    except ValueError:
        ok = True
    chk("A15 missing bdg_tau raises", ok)

    # A16-A18 come from bobo's original test_bdg.py, kept when the two were
    # merged on 26 Sep. A16 pins the routing: `bdg` adds to plug's NUMERATOR
    # and must never be treated as a displacement mode, which would send it
    # down spbc_displacement instead. A17/A18 pin `bdg_widening`, the one diag
    # key of bobo's that carries a signal ours did not already record: it is
    # e < 0, "the controller ASKED to widen", which is NOT bdg_w_eff_neg
    # (1 + eta*e < 0, "the deviation term actually reversed").
    chk("A16 'bdg' is NOT a displacement mode",
        "bdg" not in DISPLACEMENT_MODES)
    _, _, d_tight = field("bdg", bdg_eta=eta, bdg_tau=0.3 * tau_fp)
    _, _, d_wide = field("bdg", bdg_eta=eta, bdg_tau=3.0 * tau_fp)
    chk("A17 tighten is not flagged widening",
        float(d_tight["bdg_widening"][0]) == 0.0
        and float(d_tight["bdg_e"][0]) > 0,
        "widening %.1f e %.4g" % (float(d_tight["bdg_widening"][0]),
                                  float(d_tight["bdg_e"][0])))
    chk("A18 widen IS flagged widening",
        float(d_wide["bdg_widening"][0]) == 1.0
        and float(d_wide["bdg_e"][0]) < 0,
        "widening %.1f e %.4g" % (float(d_wide["bdg_widening"][0]),
                                  float(d_wide["bdg_e"][0])))


# --------------------------------------------------------------------------
# PART B: the real generator, the real guide, a real x_t
# --------------------------------------------------------------------------
def real_batch(dev, n=32, t_stop=0.6, steps=40, seed=20260925):
    """Run the UNGUIDED sampler to t_stop and return (net, f_A, x_t, mask, t)."""
    from m1_signed_bias import PhysicalProperty, load_fm
    from checkpoint_paths import require_predictor

    d = torch.load(os.path.join(ROOT, "data", "qm9.pt"), weights_only=False)
    types = d["types"]
    net, _ = load_fm(os.path.join(ROOT, "weights", "fm_ema.pt"), len(types), dev)
    f_A = PhysicalProperty(require_predictor("f_A_mu.pt"), len(types), dev)
    idx = d["split"]["val"][:n]
    mask = d["mask"][idx].to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed)
    c, f = initial_noise(mask, len(types), gen)
    smp = FlowSampler(net, mask, f_net=None)        # unguided: no guidance
    ts = smp.time_grid(steps)
    with torch.no_grad():
        for i in range(steps):
            t0, t1 = float(ts[i]), float(ts[i + 1])
            if t0 >= t_stop:
                break
            v_c, v_f = smp.field(c, f, t0)
            c, f = c + (t1 - t0) * v_c, f + (t1 - t0) * v_f
    return net, f_A, c.detach(), f.detach(), mask, t_stop


def part_b(dev):
    torch.set_default_dtype(torch.float32)
    print("\nPART B -- real generator + real f_A on a real x_t (device=%s)" % dev)
    net, f_A, coords, feats, mask, t_stop = real_batch(dev)
    tb = torch.full((coords.shape[0],), float(t_stop), device=coords.device)

    def post_fn(c, f):
        return fm_posterior(net, c, f, mask, tb)

    s = float(f_A.y_std)
    y = torch.full((coords.shape[0],), 4.6627, device=coords.device)  # mu q90
    eta = 4.0

    def field(mode, **kw):
        return guidance_field(f_A, post_fn, coords, feats, mask, y, s,
                              mode=mode, **kw)

    Pc, Pf, _ = field("plug")

    # GATE (a): eta = 0, bit-identical
    Gc, Gf, _ = field("bdg", bdg_eta=0.0, bdg_tau=1.0 * s)
    z = max(float((Gc - Pc).abs().max()), float((Gf - Pf).abs().max()))
    chk("B-a eta=0 BIT-identical to plug on the real field", z == 0.0,
        "max abs diff %.3e" % z)

    # GATE (b): the section-3 factorisation on the real field
    worst = 0.0
    for mult in (0.5, 1.0, 1.5):
        Gc, Gf, dg = field("bdg", bdg_eta=eta, bdg_tau=mult * s)
        e = float(dg["bdg_e"].mean())
        w_eff = 1.0 + eta * e
        F_bar = float(dg["f"].mean())
        y_eff = (y + eta * e * F_bar) / w_eff
        Rc, Rf, _ = guidance_field(f_A, post_fn, coords, feats, mask, y_eff, s,
                                   mode="plug")
        num = max(float((Gc - w_eff * Rc).abs().max()),
                  float((Gf - w_eff * Rf).abs().max()))
        den = max(float(Gc.abs().max()), float(Gf.abs().max()))
        worst = max(worst, num / den)
        print("      tau_mult %-4g e %+8.4f  w_eff %+8.4f  rel err %.2e"
              % (mult, e, w_eff, num / den))
    chk("B-b reduction BDG == (1+eta e) * plug(y_eff)", worst < 1e-6,
        "worst rel err %.2e" % worst)

    # GATE (c): one-sided asked to widen == plug exactly
    Gc, Gf, dg = field("bdg", bdg_eta=eta, bdg_tau=8.0 * s, bdg_onesided=True)
    o = max(float((Gc - Pc).abs().max()), float((Gf - Pf).abs().max()))
    chk("B-c one-sided asked to widen == plug exactly", o == 0.0,
        "max abs diff %.3e, raw e %+.5f (clamped to %+.5f)"
        % (o, float(dg["bdg_e_raw"].mean()), float(dg["bdg_e"].mean())))

    # diagnostics actually reach the sampler's accumulator (the `osc` trap)
    from sampling import FlowSampler as FS
    smp = FS(net, mask, f_net=f_A, y=y, s=s, mode="bdg", w=1.0,
             t_min_guide=0.5, bdg_eta=eta, bdg_tau=1.0 * s)
    smp.field(coords, feats, 0.6)
    got = smp.diag_summary()
    need = {"bdg_e", "bdg_e_raw", "bdg_V_b", "bdg_V_over_tau2", "bdg_tau",
            "bdg_dev_rms", "bdg_disp_rms", "bdg_w_eff", "bdg_batch"}
    chk("B-d every BDG diagnostic survives _accumulate_diag",
        need <= set(got), "missing %s" % sorted(need - set(got)))
    chk("B-e dispersion RMS is non-trivial (open item 3)",
        got.get("bdg_disp_rms", 0.0) > 1e-3,
        "disp_rms %.5g (a mean would read ~1e-7)" % got.get("bdg_disp_rms", 0.0))
    chk("B-f sampler refuses bdg without a setpoint",
        _raises(lambda: FS(net, mask, f_net=f_A, y=y, s=s, mode="bdg", w=1.0)))


def _raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def main():
    print("BDG gates")
    part_a()
    have = all(os.path.exists(os.path.join(ROOT, p)) for p in
               ("weights/fm_ema.pt", "weights/f_A_mu.pt", "data/qm9.pt"))
    if have:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        part_b(dev)
    else:
        print("\nPART B skipped: weights/ or data/qm9.pt absent")
    bad = [k for k, v in R.items() if not v]
    print("\n%d/%d checks pass" % (len(R) - len(bad), len(R)))
    if bad:
        print("FAILED: %s" % ", ".join(bad))
        return 1
    print("all BDG gates pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
