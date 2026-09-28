"""BLUE-TEAM defence checks for BDG, claims (i) pullback sign survival and (ii) the
unconditional w_eff bound.  Pure numpy, CPU, no repo imports, no generator.

(i)  BDG's dispersion push is c_i * g_i with c_i = -eta*e*(F_i-Fbar)/s^2.  After the
     J^T pullback the realised property move is dF_i = c_i * g_i^T J_i J_i^T g_i
     = c_i * |J_i^T g_i|^2, a SQUARED NORM times c_i, so sign(dF_i)=sign(c_i) for
     every molecule no matter how heterogeneous or non-symmetric J_i is.
     BTVG-2 pushes along h_i != g_i, giving g_i^T J_i J_i^T h_i, an indefinite form.
(ii) e = V_b/tau^2 - 1 >= -1 because V_b >= 0, so w_eff = 1+eta*e >= 1-eta ALWAYS.
     BTVG's b = -0.5(1/tau^2 - 1/V_F) has no such bound as V_F -> 0.
     Also: the field is finite at w_eff = 0 even though y_eff diverges there.
"""
import numpy as np

rng = np.random.default_rng(20260925)
S = lambda x: f"{x:.6g}"
print("=" * 78)
print("[D1] PULLBACK SIGN SURVIVAL  (claim i)")
print("=" * 78)

B, n = 256, 12
# heterogeneous, NON-symmetric Jacobians, gains spanning orders of magnitude
scales = np.exp(rng.normal(0, 1.2, size=B))
J = rng.normal(size=(B, n, n)) * scales[:, None, None]
g = rng.normal(size=(B, n))
F = rng.normal(2.0, 1.5, size=B)
Fbar = F.mean()
d = F - Fbar
eta, s2, tau2 = 4.0, 1.0, 1.0

for e in (+0.75, -0.55):
    c = -eta * e * d / s2                         # dispersion coefficient
    JJt = np.einsum('bij,bkj->bik', J, J)                                 # J J^T
    a = np.einsum('bik,bi,bk->b', JJt, g, g)                              # g^T J J^T g
    a_direct = np.sum(np.einsum('bji,bj->bi', J, g) ** 2, axis=1)         # |J^T g|^2
    dF = c * a
    print(f"  e={e:+.2f}  |J^T g|^2 two ways max rel diff "
          f"{np.max(np.abs(a - a_direct) / np.abs(a)):.3e}   min a {S(a.min())}  "
          f"max a {S(a.max())}  gain ratio {S(a.max()/a.min())}")
    print(f"    sign(dF_i)==sign(c_i) on {100*np.mean(np.sign(dF)==np.sign(c)):.1f}% of molecules"
          f"   (a_i>0 on {100*np.mean(a>0):.1f}%)")
    dV = 2.0 / (B - 1) * np.sum(d * dF)
    closed = -2.0 * eta * e / s2 * np.sum(d ** 2 * a) / (B - 1)
    print(f"    dV_b = {S(dV)}   closed form -2*eta*e/s^2*sum(d^2 a)/(B-1) = {S(closed)}"
          f"   rel err {abs(dV-closed)/abs(dV):.3e}")
    print(f"    sum(d^2 a) = {S(np.sum(d**2*a))} >= 0  =>  sign(dV_b) = -sign(e) unconditionally"
          f"   [got sign(dV)={int(np.sign(dV))}, -sign(e)={int(-np.sign(e))}]")

# finite differences through a genuinely nonlinear f and a nonlinear m(x), one molecule
print("\n  [D1b] finite differences through nonlinear f and nonlinear m(x):")
A = rng.normal(size=(n, n)); A = 0.5 * (A + A.T)
W = rng.normal(size=(n, n))
b_vec = rng.normal(size=n)
fmap = lambda m: np.tanh(m @ b_vec) + 0.3 * float(m @ A @ m) + 0.05 * float(np.sum(np.cos(m)))
mmap = lambda x: np.tanh(W @ x) + 0.4 * x           # nonlinear endpoint map, J = dm/dx


def grad_f(m, h=1e-6):
    out = np.zeros(n)
    for k in range(n):
        ek = np.zeros(n); ek[k] = h
        out[k] = (fmap(m + ek) - fmap(m - ek)) / (2 * h)
    return out


x0 = rng.normal(size=n) * 0.5
m0 = mmap(x0)
gg = grad_f(m0)
Jx = np.zeros((n, n))                                # J[k,j] = dm_k/dx_j
for j in range(n):
    ej = np.zeros(n); ej[j] = 1e-6
    Jx[:, j] = (mmap(x0 + ej) - mmap(x0 - ej)) / (2e-6)
Jtg = Jx.T @ gg
pred = float(Jtg @ Jtg)
for eps in (1e-3, 1e-4, 1e-5):
    for sgn in (+1, -1):
        got = (fmap(mmap(x0 + sgn * eps * Jtg)) - fmap(m0)) / (sgn * eps)
        print(f"    eps={eps:.0e} sgn={sgn:+d}  measured dF/deps {got:+.6f}  "
              f"predicted |J^T g|^2 = {pred:+.6f}  rel err {abs(got-pred)/abs(pred):.2e}  "
              f"sign agrees {np.sign(got)==np.sign(pred)}")

print("\n  [D1c] BTVG-2 contrast: a direction h != g gives the indefinite form g^T J J^T h")
h = rng.normal(size=(B, n))                          # an unrelated variance direction
q = np.einsum('bi,bi->b', np.einsum('bji,bj->bi', J, g), np.einsum('bji,bj->bi', J, h))
print(f"    g^T J J^T h has the OPPOSITE sign to g^T J J^T g on "
      f"{100*np.mean(np.sign(q)!=np.sign(a)):.1f}% of molecules")
# the indefiniteness is generic, not a property of one random draw: repeat over seeds
flips = []
for k in range(200):
    r = np.random.default_rng(1000 + k)
    Jk = r.normal(size=(64, n, n)) * np.exp(r.normal(0, 1.2, size=64))[:, None, None]
    gk, hk = r.normal(size=(64, n)), r.normal(size=(64, n))
    ak = np.sum(np.einsum('bji,bj->bi', Jk, gk) ** 2, axis=1)
    qk = np.einsum('bi,bi->b', np.einsum('bji,bj->bi', Jk, gk),
                   np.einsum('bji,bj->bi', Jk, hk))
    flips.append(np.mean(np.sign(qk) != np.sign(ak)))
flips = np.array(flips)
print(f"    over 200 independent draws the flip fraction for h != g is "
      f"{flips.mean()*100:.1f}% +/- {flips.std()*100:.1f}% (min {flips.min()*100:.1f}, "
      f"max {flips.max()*100:.1f}); for h = g it is 0.0% by algebra")

print()
print("=" * 78)
print("[D2] w_eff >= 1-eta UNCONDITIONALLY  (claim ii)")
print("=" * 78)
for eta in (1.0, 2.0, 4.0, 8.0):
    # drive V_b from a fully collapsed batch to a wildly over-dispersed one
    ratios = np.concatenate([[0.0, 1e-16, 1e-8], np.geomspace(1e-4, 1e4, 40)])
    e_all = ratios - 1.0
    w = 1.0 + eta * e_all
    print(f"  eta={eta:<4g} min e {S(e_all.min())}  min w_eff {S(w.min())}  "
          f"1-eta = {S(1-eta)}  exact match {np.isclose(w.min(), 1-eta, atol=1e-15)}  "
          f"all finite {np.all(np.isfinite(w))}")
# a literally collapsed batch: every F_i identical
F_col = np.full(B, 3.14159)
V_col = F_col.var(ddof=1)
e_col = V_col / tau2 - 1.0
print(f"  collapsed batch (all F_i equal): V_b={S(V_col)}  e={S(e_col)}  "
      f"w_eff={S(1+4.0*e_col)} = 1-eta at eta=4")
print("  BTVG's coefficient on the same excursion (no bound):")
for VF in (1e-2, 1e-4, 1e-8, 1e-12, 1e-16):
    print(f"    V_F={VF:.0e}  b = -0.5(1/tau^2 - 1/V_F) = {-0.5*(1/tau2 - 1/VF):+.3e}")

print("\n  [D2b] the w_eff=0 point is a coordinate singularity, NOT a field singularity")
y = 3.0
eta = 4.0
for ratio in (0.7600, 0.7501, 0.750001, 0.75, 0.749999, 0.7499):
    e = ratio - 1.0
    w_eff = 1 + eta * e
    num = (y - F) - eta * e * d                      # the field, as implemented
    with np.errstate(divide='ignore', invalid='ignore'):
        y_eff = (y + eta * e * Fbar) / w_eff
    ref = y - Fbar
    print(f"    V_b/tau^2={ratio:<9g} w_eff={w_eff:+.3e}  y_eff={y_eff:+.4e}  "
          f"max|num| {np.max(np.abs(num)):.4f}  spread(num) {num.std():.2e}  "
          f"max|num-(y-Fbar)| {np.max(np.abs(num-ref)):.2e}  all finite {np.all(np.isfinite(num))}")
print(f"    at w_eff=0 exactly, num_i = y - Fbar = {S(y-Fbar)} for every i: a pure common")
print("    translation along g_i, finite and analytic in V_b.  y_eff is the ratio of two")
print("    finite things, one of which crosses zero; the field never does anything special.")

print("\n  [D2c] bounded DEPTH of the widening branch")
print("    when e<0 the per-molecule dispersion coefficient is -eta*e*d_i/s^2 with |eta*e| <= eta,")
print("    so the widening push can never exceed eta * |F_i-Fbar|/s^2, i.e. eta times plug's own")
print("    push at one batch-sd of miss.  BTVG's measured widening coefficients were +5.3, +110.8,")
print("    +78.6 (repo guidance.py:1487) against a tau^2 of order 1 -- unbounded by construction.")
