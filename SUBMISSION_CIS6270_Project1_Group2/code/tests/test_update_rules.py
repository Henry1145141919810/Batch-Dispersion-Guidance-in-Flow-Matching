"""Gates for the inference-time guidance update rules.

The claims these check, in the order they matter:

  1. euler is inert -- the existing sweep cannot move
  2. Newton-Schulz really orthogonalises (singular values -> 1)
  3. Muon and momentum are EXACTLY rotation-equivariant on the coordinate
     block; elementwise Adam is NOT, and the test asserts the failure rather
     than skipping it, so nobody later promotes it to a candidate method
  4. adam_eq repairs that: per-atom scalar preconditioning is equivariant
  5. padded atoms stay exactly zero under every rule
  6. the norm-rescale contract holds: |D| == |G| per sample
  7. bias correction is right at step 1 (m_hat == G for momentum)
  8. the Heun second stage does not advance the optimiser state

Run: python proj1/tests/test_update_rules.py
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
from guidance_update import (EQUIVARIANT, GuidanceUpdater,  # noqa: E402
                             RULES, newton_schulz)

torch.manual_seed(0)
DT = torch.float64
B, N, K = 3, 9, 5
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print("  %-52s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


def fixture():
    mask = torch.ones(B, N, dtype=DT)
    mask[:, -3:] = 0.0                      # three padded atoms
    G_c = torch.randn(B, N, 3, dtype=DT) * mask.unsqueeze(-1)
    G_f = torch.randn(B, N, K, dtype=DT) * mask.unsqueeze(-1)
    return G_c, G_f, mask


def rand_rotation():
    A = torch.randn(3, 3, dtype=DT)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


# 1. euler is inert
G_c, G_f, mask = fixture()
u = GuidanceUpdater("euler")
D_c, D_f = u.apply(G_c, G_f, mask, t_scalar=0.5)
check("euler returns the direction untouched",
      torch.equal(D_c, G_c) and torch.equal(D_f, G_f))

# 2. Newton-Schulz orthogonalises -- to Muon's band, not to exactness.
# The quintic's fixed point is an interval around 1, so asserting sv == 1 would
# be asserting something Muon does not do; what matters is that a badly
# conditioned direction comes out near-isotropic.
X = torch.randn(B, N, 3, dtype=DT)
X = X * torch.tensor([1.0, 30.0, 0.05], dtype=DT)      # condition number ~600
sv_in = torch.linalg.svdvals(X)
sv = torch.linalg.svdvals(newton_schulz(X))
check("newton_schulz singular values land in the quintic's band",
      bool(sv.min() > 0.6 and sv.max() < 1.3),
      "range %.3f-%.3f" % (float(sv.min()), float(sv.max())))
# The measurement behind ns_steps=8: on a badly conditioned block -- exactly
# where orthogonalisation is supposed to earn its keep -- Muon's 5 steps has not
# arrived yet. Stated as a comparison against the settled value rather than a
# threshold on one random draw, which would be a brittle gate.
X_ill = torch.randn(64, N, 3, dtype=DT) * torch.tensor([1., 30., .05], dtype=DT)
sv5 = torch.linalg.svdvals(newton_schulz(X_ill, steps=5))
sv8 = torch.linalg.svdvals(newton_schulz(X_ill, steps=8))
check("ns_steps=5 leaves an ill-conditioned block short of the band",
      bool(sv5.min() < sv8.min() - 0.1),
      "sv_min %.3f at 5 steps vs %.3f at 8" % (float(sv5.min()), float(sv8.min())))
cond_in = float((sv_in.max(dim=-1).values / sv_in.min(dim=-1).values).max())
cond_out = float((sv.max(dim=-1).values / sv.min(dim=-1).values).max())
check("newton_schulz collapses the condition number",
      cond_out < cond_in / 100, "%.1f -> %.3f" % (cond_in, cond_out))

# 2b. the transpose identity the implementation relies on
Xs = torch.randn(B, 4, 7, dtype=DT)
check("p(X)^T == p(X^T) (transpose is exact, not an approximation)",
      bool((newton_schulz(Xs).transpose(-2, -1)
            - newton_schulz(Xs.transpose(-2, -1))).abs().max() < 1e-10))

# 3 + 4. equivariance under rotation, per rule
R = rand_rotation()
for rule in RULES:
    G_c, G_f, mask = fixture()
    ua, ub = GuidanceUpdater(rule), GuidanceUpdater(rule)
    # three steps, so the moment buffers are actually carrying history
    for _ in range(3):
        Da_c, Da_f = ua.apply(G_c, G_f, mask, t_scalar=0.5)
        Db_c, Db_f = ub.apply(G_c @ R.T, G_f, mask, t_scalar=0.5)
    err = float((Da_c @ R.T - Db_c).abs().max())
    is_eq = err < 1e-8
    if EQUIVARIANT[rule]:
        check("%-8s is rotation-equivariant on coords" % rule, is_eq,
              "max err %.2e" % err)
    else:
        check("%-8s is NOT equivariant (asserted, not assumed)" % rule,
              not is_eq, "max err %.2e" % err)

# 5. padded atoms stay zero
for rule in RULES:
    G_c, G_f, mask = fixture()
    u = GuidanceUpdater(rule)
    for _ in range(3):
        D_c, D_f = u.apply(G_c, G_f, mask, t_scalar=0.5)
    pad = (mask == 0)
    ok = (float(D_c[pad].abs().max()) == 0.0
          and float(D_f[pad].abs().max()) == 0.0)
    check("%-8s leaves padded atoms exactly zero" % rule, ok)

# 6. the rescale contract
for rule in RULES:
    G_c, G_f, mask = fixture()
    u = GuidanceUpdater(rule, rescale="norm")
    for _ in range(3):
        D_c, D_f = u.apply(G_c, G_f, mask, t_scalar=0.5)
    gn = torch.sqrt((G_c ** 2).sum((1, 2)) + (G_f ** 2).sum((1, 2)))
    dn = torch.sqrt((D_c ** 2).sum((1, 2)) + (D_f ** 2).sum((1, 2)))
    check("%-8s preserves the per-sample norm" % rule,
          bool((gn - dn).abs().max() < 1e-8),
          "max err %.2e" % float((gn - dn).abs().max()))

# 7. bias correction at the first step
G_c, G_f, mask = fixture()
u = GuidanceUpdater("momentum", rescale="none")
D_c, D_f = u.apply(G_c, G_f, mask, t_scalar=0.5)
check("momentum step 1 equals the raw direction (bias correction)",
      bool((D_c - G_c).abs().max() < 1e-12))

# 7b. with a constant gradient, momentum converges to that gradient
u = GuidanceUpdater("momentum", rescale="none")
for _ in range(50):
    D_c, D_f = u.apply(G_c, G_f, mask, t_scalar=0.5)
check("momentum on a constant signal reproduces it",
      bool((D_c - G_c).abs().max() < 1e-6))

# 8. the Heun second stage does not advance state
u = GuidanceUpdater("momentum")
u.apply(G_c, G_f, mask, t_scalar=0.5, update_state=True)
k_after_primary = u.k
m_snapshot = u.m_c.clone()
u.apply(G_c * 3.0, G_f * 3.0, mask, t_scalar=0.6, update_state=False)
check("Heun stage 2 reads but does not write the buffers",
      u.k == k_after_primary and torch.equal(u.m_c, m_snapshot))

# 9. diagnostics are actually populated
u = GuidanceUpdater("momentum")
for i in range(5):
    u.apply(torch.randn_like(G_c) * mask.unsqueeze(-1),
            torch.randn_like(G_f) * mask.unsqueeze(-1), mask, t_scalar=0.5 + i)
s = u.summary()
check("summary reports staleness and consecutive-direction cosines",
      "cos_mom_g" in s and "cos_prev" in s and "cos_prev_late" in s)

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
if FAIL:
    for f in FAIL:
        print("  FAILED: " + f)
sys.exit(1 if FAIL else 0)
