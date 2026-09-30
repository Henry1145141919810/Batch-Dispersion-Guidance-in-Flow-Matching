"""Gates for the x-space geometry repair (`btvg2_xproj`).

WHAT WENT WRONG, AND WHY test_btvg2.py COULD NOT SEE IT.
btvg2 makes its variance step orthogonal to the mean direction and
V-decreasing in m-space, then applies J^T. Neither property survives the
pullback (docs/methods/BTVG_FAILURE_AUDIT_AND_REDESIGN.md section 3). Gates
G3/G6 there check m-space; G7 checks that a pullback happens. Their
conjunction never checks the space the sampler actually updates, so the
defect passed ten gates. These gates check x_t space, and X2 is written so
that the UNREPAIRED arm fails it -- if it ever starts passing, the gate has
gone vacuous.

  X1  the audit's exact counterexample, as pure algebra: projecting in m-space
      then pulling back gives mean derivative +4 and variance derivative
      +3.5; projecting in x-space gives 0 and <= 0
  X2  real code path, ASYMMETRIC non-diagonal J: along btvg2_xproj's variance
      step the frozen-draw mu_hat is unchanged and V_hat does not rise, both
      by central differences; along btvg2's, at least one fails
  X3  tau huge => the variance coefficient is 0 => btvg2_xproj is
      bit-identical to lgd_mc
  X4  cost: three VJPs against lgd_mc's one, as the audit says it must be
  X5  the field is exactly zero on padded atoms, on a fixture where the
      pullback demonstrably would leak into them (so dropping the mask fails)
  X6  the cap's LIMIT equals max(|y-mu|, sqrt V)|a_x|/s^2, rebuilt
      independently, and |S| stays under it. Checked via the returned limit
      because the cap never binds at real scale, so a gate that waited for it
      to fire would be vacuous (building the limit from the m-space |gbar|
      instead passed silently before this)

Run: python proj1/tests/test_btvg2_xproj.py
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from guidance import (Cost, KNOWN_MODES, Posterior,  # noqa: E402
                      _pullback, guidance_field, trace_scale)

torch.set_default_dtype(torch.float64)
B, N, T, K = 6, 4, 3, 4
R = {}


def check(name, ok, detail=""):
    R[name] = bool(ok)
    print("%-4s %-63s %s" % ("PASS" if ok else "FAIL", name, detail))


class Quad(torch.nn.Module):
    def __init__(self, a_c, a_f, b_c, b_f):
        super().__init__()
        self.a_c, self.a_f, self.b_c, self.b_f = a_c, a_f, b_c, b_f

    def forward(self, coords, feats, mask):
        return (0.5 * (self.a_c * coords ** 2).sum(dim=(1, 2))
                + 0.5 * (self.a_f * feats ** 2).sum(dim=(1, 2))
                + (self.b_c * coords).sum(dim=(1, 2))
                + (self.b_f * feats).sum(dim=(1, 2)))


def setup_asym(seed=13):
    """A posterior whose Jacobian is NON-diagonal and NON-symmetric, which is
    the regime where the m-space projection provably breaks."""
    torch.manual_seed(seed)
    mask = torch.ones(B, N)
    coords, feats = torch.randn(B, N, 3), torch.randn(B, N, T)
    Wc = torch.eye(3) + 0.6 * torch.randn(3, 3)
    Wf = torch.eye(T) + 0.6 * torch.randn(T, T)
    k = torch.full((B,), 0.7)
    a_c, a_f = torch.rand(B, N, 3) + 0.2, torch.rand(B, N, T) + 0.2
    b_c, b_f = torch.randn(B, N, 3), torch.randn(B, N, T)

    def post_fn(c, f):
        return Posterior(c @ Wc, f @ Wf, k)

    return dict(mask=mask, coords=coords, feats=feats, k=k,
                f_net=Quad(a_c, a_f, b_c, b_f), post_fn=post_fn,
                Wc=Wc, Wf=Wf)


def setup_partial(seed=17, curv=1.0):
    """Two deliberate departures from setup_asym:
    (1) `mask` has padding, and the toy property does NOT mask its own input,
        so the pullback carries nonzero gradient into padded rows. The real
        f_net architectures all mask internally, so in production the mask
        multiply in the arm is defensive -- this fixture is what makes it
        load-bearing, i.e. a gate that fails if it is dropped.
    (2) `curv` scales the quadratic coefficients. Large curvature makes the
        per-draw gradients disagree, so |dV/dm| grows against |gbar| and the
        cap binds -- otherwise that branch is never exercised."""
    torch.manual_seed(seed)
    mask = torch.ones(B, N)
    mask[:, -1] = 0.0                        # one padded slot per molecule
    mask[0, -2] = 0.0                        # and a second on one of them
    coords, feats = torch.randn(B, N, 3), torch.randn(B, N, T)
    Wc = torch.eye(3) + 0.6 * torch.randn(3, 3)
    Wf = torch.eye(T) + 0.6 * torch.randn(T, T)
    k = torch.full((B,), 0.7)
    a_c = (torch.rand(B, N, 3) + 0.2) * curv
    a_f = (torch.rand(B, N, T) + 0.2) * curv
    b_c, b_f = torch.randn(B, N, 3), torch.randn(B, N, T)

    def post_fn(c, f):
        return Posterior(c @ Wc, f @ Wf, k)

    return dict(mask=mask, coords=coords, feats=feats, k=k,
                f_net=Quad(a_c, a_f, b_c, b_f), post_fn=post_fn)


def gen(seed=5):
    return torch.Generator().manual_seed(seed)


def dot(x1, y1, x2, y2):
    return (x1 * x2).sum(dim=(1, 2)) + (y1 * y2).sum(dim=(1, 2))


def frozen_draws(E, seed=5):
    """Replay btvg2's probes: same generator calls, same order. r is detached
    in the arm, so the surrogate whose invariants are claimed holds BOTH the
    offsets and the radius fixed. That is what these gates differentiate."""
    g = gen(seed)
    r2 = trace_scale(E["post_fn"], E["coords"], E["feats"], E["mask"], E["k"],
                     1, g, com_free=True)
    r = r2.clamp(min=0).sqrt().view(-1, 1, 1)
    msk = E["mask"].unsqueeze(-1)
    p = E["post_fn"](E["coords"], E["feats"])
    Z = []
    for _ in range(K):
        z_c = torch.randn(p.mean_coords.shape, generator=g) * msk
        z_f = torch.randn(p.mean_feats.shape, generator=g) * msk
        Z.append((r * z_c, r * z_f))
    return Z


def mu_V_at(E, x_c, x_f, Z):
    """(mu_hat, V_hat) at state x with the offsets frozen."""
    p = E["post_fn"](x_c, x_f)
    F = torch.stack([E["f_net"](p.mean_coords + dc, p.mean_feats + df, E["mask"])
                     for dc, df in Z])
    return F.mean(0), F.var(0, unbiased=True)


def main():
    # ---------------- X1: the audit's counterexample, pure algebra ----------
    J = torch.diag(torch.tensor([3.0, 1.0]))
    g = torch.tensor([1.0, 1.0])
    h = torch.tensor([1.0, 2.0])

    def proj(v, u):
        return v - u * (u @ v) / (u @ u)

    S_m = -proj(h, g)                        # m-space: orthogonal to g, cuts V
    S_x_bad = J.T @ S_m                      # ...then pulled back
    a_x, b_x = J.T @ g, J.T @ h
    mean_bad, var_bad = a_x @ S_x_bad, b_x @ S_x_bad
    S_x_good = -proj(b_x, a_x)               # x-space repair
    mean_good, var_good = a_x @ S_x_good, b_x @ S_x_good
    check("X1 audit counterexample: m-space breaks, x-space holds",
          abs(S_m[0] - 0.5) < 1e-12 and abs(mean_bad - 4.0) < 1e-12
          and abs(var_bad - 3.5) < 1e-12
          and abs(mean_good) < 1e-12 and var_good <= 1e-12,
          "m-space gives mean %+.1f var %+.1f; x-space %+.0e / %+.2f"
          % (mean_bad, var_bad, mean_good.item(), var_good.item()))

    # ---------------- X2: real code path, asymmetric J ----------------------
    E = setup_asym()
    y = torch.randn(B) * 2.0
    s, tau = 1.2, 0.25
    Z = frozen_draws(E)

    fields = {}
    for mode in ("lgd_mc", "btvg2", "btvg2_xproj"):
        kw = {} if mode == "lgd_mc" else {"tau": tau}
        fields[mode] = guidance_field(E["f_net"], E["post_fn"], E["coords"],
                                      E["feats"], E["mask"], y, s, mode=mode,
                                      n_mc=K, generator=gen(), **kw)[:2]

    def directional(S_c, S_f, eps=1e-6):
        """Central differences of (mu_hat, V_hat) along S, at the state."""
        mp, vp = mu_V_at(E, E["coords"] + eps * S_c, E["feats"] + eps * S_f, Z)
        mm, vm = mu_V_at(E, E["coords"] - eps * S_c, E["feats"] - eps * S_f, Z)
        return (mp - mm) / (2 * eps), (vp - vm) / (2 * eps)

    res, live = {}, None
    for mode in ("btvg2", "btvg2_xproj"):
        # the mean term is lgd_mc's by construction at the same seed, so the
        # difference IS this arm's variance step, in x_t space
        S_c = fields[mode][0] - fields["lgd_mc"][0]
        S_f = fields[mode][1] - fields["lgd_mc"][1]
        dmu, dV = directional(S_c, S_f)
        nS = dot(S_c, S_f, S_c, S_f).sqrt()
        # samples whose variance already sits below tau^2 get coef = 0 and so
        # a zero step, correctly; the invariants are vacuous there
        on = nS > 1e-12
        live = on if live is None else (live & on)
        rel = (dmu.abs() / nS.clamp(min=1e-30))
        res[mode] = (rel, dV, on)

    # the threshold is set by CENTRAL-DIFFERENCE truncation at eps = 1e-6, not
    # by the projection: the repaired arm sits at ~1e-7 and the unrepaired one
    # four to seven orders of magnitude above it, which is the whole point
    ok_fix = (bool(live.any())
              and res["btvg2_xproj"][0][live].max().item() < 1e-4
              and res["btvg2_xproj"][1][live].max().item() <= 1e-8)
    ok_broken = (res["btvg2"][0][live].max().item() > 1e-2
                 or res["btvg2"][1][live].max().item() > 1e-8)
    check("X2 x-space: mu_hat unchanged, V_hat not raised (btvg2 fails it)",
          ok_fix and ok_broken,
          "%d/%d live; xproj |dmu|/|S| %.1e, max dV %+.1e | btvg2 %.2f, %+.1e"
          % (int(live.sum()), B,
             res["btvg2_xproj"][0][live].max().item(),
             res["btvg2_xproj"][1][live].max().item(),
             res["btvg2"][0][live].max().item(),
             res["btvg2"][1][live].max().item()))

    # ---------------- X3: variance off => lgd_mc, bit for bit ---------------
    Gx = guidance_field(E["f_net"], E["post_fn"], E["coords"], E["feats"],
                        E["mask"], y, s, mode="btvg2_xproj", n_mc=K,
                        generator=gen(), tau=1e6)[:2]
    check("X3 tau huge => btvg2_xproj == lgd_mc bit for bit",
          torch.equal(Gx[0], fields["lgd_mc"][0])
          and torch.equal(Gx[1], fields["lgd_mc"][1]))

    # ---------------- X4: the cost the repair admits to ---------------------
    costs = {}
    for mode in ("lgd_mc", "btvg2", "btvg2_xproj"):
        c = Cost()
        kw = {} if mode == "lgd_mc" else {"tau": tau}
        guidance_field(E["f_net"], E["post_fn"], E["coords"], E["feats"],
                       E["mask"], y, s, mode=mode, n_mc=K, generator=gen(),
                       cost=c, **kw)
        costs[mode] = c.gen_vjp
    check("X4 cost: 3 VJPs vs lgd_mc's 1, and btvg2's 1",
          costs["btvg2_xproj"] == 3 and costs["lgd_mc"] == 1
          and costs["btvg2"] == 1,
          "lgd_mc %d, btvg2 %d, xproj %d"
          % (costs["lgd_mc"], costs["btvg2"], costs["btvg2_xproj"]))

    # ---------------- X5: the mask is load-bearing --------------------------
    # With padding and a property that does not mask its own input, the
    # pullback leaks gradient into padded rows. The field must not.
    P1 = setup_partial()
    y1 = torch.randn(B)
    G1 = guidance_field(P1["f_net"], P1["post_fn"], P1["coords"], P1["feats"],
                        P1["mask"], y1, s, mode="btvg2_xproj", n_mc=K,
                        generator=gen(), tau=tau)[:2]
    pad = (P1["mask"] < 0.5)
    raw_c, raw_f = _pullback(P1["post_fn"], P1["coords"], P1["feats"],
                             torch.ones_like(P1["coords"]),
                             torch.ones_like(P1["feats"]))
    leaks = max(raw_c[pad].abs().max().item(), raw_f[pad].abs().max().item())
    check("X5 field is zero on padded atoms (and the pullback would leak)",
          G1[0][pad].abs().max().item() == 0.0
          and G1[1][pad].abs().max().item() == 0.0 and leaks > 1e-6,
          "pullback would leak %.2e into padding" % leaks)

    # ---------------- X6: the cap actually binds, and binds exactly ----------
    P2 = setup_partial(curv=400.0)
    y2 = torch.randn(B) * 0.1
    Gb = guidance_field(P2["f_net"], P2["post_fn"], P2["coords"], P2["feats"],
                        P2["mask"], y2, s, mode="lgd_mc", n_mc=K,
                        generator=gen())[:2]
    Gx2, _, dg2 = guidance_field(P2["f_net"], P2["post_fn"], P2["coords"],
                                 P2["feats"], P2["mask"], y2, s,
                                 mode="btvg2_xproj", n_mc=K, generator=gen(),
                                 tau=1e-6)
    # lgd_mc's branch does NOT mask its pullback, so on this padded fixture its
    # field carries the leak X5 is about. Mask both before subtracting or the
    # difference is not the variance step.
    msk2 = P2["mask"].unsqueeze(-1)
    S2_c = Gx2 - Gb[0] * msk2
    S2_f = _ - Gb[1] * msk2
    nS2 = dot(S2_c, S2_f, S2_c, S2_f).sqrt()
    # reference limit, rebuilt from scratch: max(|y-mu|, sqrt V) |a_x| / s^2
    Z2 = frozen_draws(P2)
    mu2, V2 = mu_V_at(P2, P2["coords"], P2["feats"], Z2)
    p2 = P2["post_fn"](P2["coords"], P2["feats"])
    gs = []
    for dc, df in Z2:
        mc = (p2.mean_coords + dc).detach().requires_grad_(True)
        mf = (p2.mean_feats + df).detach().requires_grad_(True)
        gs.append(torch.autograd.grad(P2["f_net"](mc, mf, P2["mask"]).sum(),
                                      (mc, mf)))
    gbc = torch.stack([g[0] for g in gs]).mean(0)
    gbf = torch.stack([g[1] for g in gs]).mean(0)
    ac, af = _pullback(P2["post_fn"], P2["coords"], P2["feats"], gbc, gbf)
    ac, af = ac * msk2, af * msk2
    lim2 = (torch.maximum((y2 - mu2).abs(), V2.sqrt())
            * dot(ac, af, ac, af).sqrt() / s ** 2)
    bound = dg2["btvg2_capped"] > 0.5
    # The INVARIANT uses the arm's own reported step norm. Reconstructing it by
    # subtracting the mean term is hopelessly ill-conditioned here -- |M_x| is
    # ~1.9e7 against a step of ~1e-5, so the difference keeps only ~7 digits --
    # which is why the loose cross-check below is at rtol 1e-4, not 1e-8.
    ratio = dg2["btvg2_step_norm"] / lim2.clamp(min=1e-30)
    live2 = dg2["btvg2_step_norm"] > 1e-12
    ok6 = (torch.allclose(dg2["btvg2_cap_lim"], lim2, rtol=1e-9, atol=0)
           and bool((ratio <= 1 + 1e-8).all())
           and torch.allclose(dg2["btvg2_step_norm"][live2], nS2[live2],
                              rtol=1e-4, atol=0)
           and not bool(bound.any()))
    check("X6 cap limit == max(|y-mu|,sqrtV)|a_x|/s^2, and |S| stays under it",
          ok6, "max |S|/lim %.2e, capped %d/%d (a rail, not a mechanism)"
          % (ratio.max(), int(bound.sum()), B))

    check("X7 mode registered", "btvg2_xproj" in KNOWN_MODES)

    ok = all(R.values())
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
