"""BDG items 1, 2, 6, 7: gradient identity, sign table, the §3 reduction,
its degenerate/negative regimes, and the affine-coefficient generalisation."""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
rng = np.random.default_rng(20260925)
out = {}

# ---------------------------------------------------------------- item 1
# V_b = 1/(B-1) sum_i (F_i - Fbar)^2 ; claim dV/dF_i = 2/(B-1)(F_i - Fbar)
B, n = 9, 5


def Vb(F):
    return ((F - F.mean()) ** 2).sum() / (len(F) - 1)


F = torch.tensor(rng.normal(size=B), requires_grad=True)
(gauto,) = torch.autograd.grad(Vb(F), F)
gclaim = 2.0 / (B - 1) * (F - F.mean())
print("[1a] dV/dF autograd vs claim   max|err| = %.3e" % (gauto - gclaim).abs().max())

# finite differences on F
F0 = torch.tensor(rng.normal(size=B))
h = 1e-6
fd = torch.empty(B)
for i in range(B):
    Fp, Fm = F0.clone(), F0.clone()
    Fp[i] += h
    Fm[i] -= h
    fd[i] = (Vb(Fp) - Vb(Fm)) / (2 * h)
cl = 2.0 / (B - 1) * (F0 - F0.mean())
print("[1b] dV/dF central-diff        max|err| = %.3e  (rel %.2e)"
      % ((fd - cl).abs().max(), ((fd - cl).abs() / cl.abs()).max()))

# dV/dm_i through a genuinely nonlinear, non-separable f
A = torch.tensor(rng.normal(size=(n, n)))
A = A + A.T
b = torch.tensor(rng.normal(size=n))


def f_A(M):                       # M: (B, n) -> (B,)
    return torch.tanh(M @ b) + 0.3 * ((M @ A) * M).sum(-1)


M = torch.tensor(rng.normal(size=(B, n)), requires_grad=True)
(dV_auto,) = torch.autograd.grad(Vb(f_A(M)), M)
with torch.no_grad():
    Fv = f_A(M)
Mg = M.detach().clone().requires_grad_(True)
(g_i,) = torch.autograd.grad(f_A(Mg).sum(), Mg)          # rows = grad f(m_i)
dV_claim = (2.0 / (B - 1) * (Fv - Fv.mean())).unsqueeze(-1) * g_i
print("[1c] dV/dm autograd vs claim   max|err| = %.3e" % (dV_auto - dV_claim).abs().max())

# finite differences on m, all B*n entries
M0 = M.detach().clone()
h = 1e-6
fdm = torch.empty(B, n)
for i in range(B):
    for j in range(n):
        Mp, Mm = M0.clone(), M0.clone()
        Mp[i, j] += h
        Mm[i, j] -= h
        fdm[i, j] = (Vb(f_A(Mp)) - Vb(f_A(Mm))) / (2 * h)
print("[1d] dV/dm central-diff        max|err| = %.3e  max|entry| = %.3e"
      % ((fdm - dV_claim).abs().max(), dV_claim.abs().max()))

# does the 2/(B-1) really make the gain batch-size free?
print("[1e] |dV/dF| scale vs B (fixed population sd=1):")
for Bx in (64, 128, 512, 4096):
    Fx = torch.tensor(rng.normal(size=Bx))
    print("     B=%5d  rms(2/(B-1)(F-Fbar)) = %.5f   rms(F-Fbar) = %.5f"
          % (Bx, (2.0 / (Bx - 1) * (Fx - Fx.mean())).pow(2).mean().sqrt(),
             (Fx - Fx.mean()).pow(2).mean().sqrt()))

# ---------------------------------------------------------------- item 2
print("\n[2] REDUCTION  num_i = (y-F_i) - eta*e*(F_i-Fbar) = w_eff*(y_eff-F_i)")
worst = 0.0
for trial in range(2000):
    Bx = int(rng.integers(2, 600))
    Fx = rng.normal(loc=rng.normal() * 3, scale=abs(rng.normal()) + 1e-3, size=Bx)
    y = rng.normal() * 5
    eta = abs(rng.normal()) * 6
    tau2 = (abs(rng.normal()) + 1e-3) ** 2
    Fb = Fx.mean()
    V = ((Fx - Fb) ** 2).sum() / (Bx - 1)
    e = (V - tau2) / tau2
    num = (y - Fx) - eta * e * (Fx - Fb)
    w = 1.0 + eta * e
    if abs(w) < 1e-9:
        continue
    yeff = (y + eta * e * Fb) / w
    err = np.max(np.abs(num - w * (yeff - Fx))) / max(1.0, np.max(np.abs(num)))
    worst = max(worst, err)
print("     max relative factorisation error over 2000 random batches: %.3e" % worst)

# regime map
print("\n[2b] regime of w_eff = 1 + eta*e, with e = V/tau^2 - 1 >= -1")
for eta in (1.0, 2.0, 4.0, 8.0):
    thr = 1.0 - 1.0 / eta          # V/tau^2 below this => w_eff < 0
    print("     eta=%4.1f : w_eff<0 iff V_b < %.4f tau^2 ; w_eff in [%.2f, inf) ; "
          "w_eff=0 at V_b = %.4f tau^2" % (eta, thr, 1 - eta, thr))

# y_eff blow-up near the singularity, and that num stays finite
print("\n[2c] y_eff singularity at w_eff -> 0 (eta=4, y=1, Fbar=0, spread 1)")
Fx = np.array([-1.0, 0.0, 1.0])
y, eta = 1.0, 4.0
for u in (0.80, 0.7600, 0.7501, 0.750001, 0.75, 0.749999, 0.7499, 0.70, 0.0):
    e = u - 1.0
    w = 1 + eta * e
    num = (y - Fx) - eta * e * (Fx - Fx.mean())
    ye = np.inf if w == 0 else (y + eta * e * Fx.mean()) / w
    print("     V/tau^2=%9.6f  w_eff=%+9.6f  y_eff=%+12.4g  num=%s"
          % (u, w, ye, np.array2string(num, precision=4)))

# where does y_eff sit relative to y and Fbar?
print("\n[2d] y_eff as a combination of y and Fbar (weights 1/w_eff, eta*e/w_eff)")
for u, lab in ((2.0, "V>tau^2 tighten"), (0.9, "mild widen"),
               (0.75, "w_eff=0"), (0.5, "hard widen"), (0.0, "V=0")):
    e = u - 1
    w = 1 + 4 * e
    print("     %-16s V/tau^2=%.2f e=%+.2f w_eff=%+.2f  weight_on_y=%s weight_on_Fbar=%s"
          % (lab, u, e, w, "%+.3f" % (1 / w) if w else "inf",
             "%+.3f" % (4 * e / w) if w else "inf"))

# sign table: does the dispersion TERM alone tighten / widen as claimed?
print("\n[2e] sign table check: sign of the DISPERSION term for a high-F molecule")
for u in (1.5, 1.0, 0.5):
    e = u - 1
    disp = -4.0 * e * (1.0 - 0.0)     # F_i - Fbar = +1
    print("     V/tau^2=%.2f  e=%+0.2f  disp coeff on a high-F molecule = %+.3f (%s)"
          % (u, e, disp, "DOWN/tighten" if disp < 0 else
             ("zero" if disp == 0 else "UP/widen")))

# ---------------------------------------------------------------- item 7
print("\n[7] 'per-sample coefficient affine in F_i => plug reweighted'")
Bx = 7
Fx = rng.normal(size=Bx)
y = 0.7
# (a) batch-shared affine slope -> reduces
Aa, Cc = -1.3, 2.0
num = Aa * Fx + Cc
w, ye = -Aa, -Cc / Aa
print("     (a) shared slope    : max|err| = %.2e   -> reduces"
      % np.max(np.abs(num - w * (ye - Fx))))
# (b) per-sample slope (SMG: den = s^2 + v_i) -> does NOT reduce to one (w,y)
s2 = 1.0
v = np.abs(rng.normal(size=Bx))
num_smg = (y - Fx) / (s2 + v)
best = None
for _ in range(20000):                       # brute search for ANY single (w,y)
    ww = rng.normal() * 3
    yy = rng.normal() * 3
    r = np.max(np.abs(num_smg - ww * (yy - Fx)))
    if best is None or r < best[0]:
        best = (r, ww, yy)
# exact least-squares fit of num_smg on [1, F] then residual
Xd = np.stack([np.ones(Bx), Fx], 1)
coef, *_ = np.linalg.lstsq(Xd, num_smg, rcond=None)
resid = num_smg - Xd @ coef
print("     (b) per-sample slope: best single (w,y) max|err| = %.4f ; "
      "LSQ affine residual rms = %.4f vs |num| rms %.4f -> does NOT reduce"
      % (best[0], np.sqrt((resid ** 2).mean()), np.sqrt((num_smg ** 2).mean())))
# (c) degenerate A = 0 : pure common translation, no finite y_eff
num_deg = np.full(Bx, 0.9)
print("     (c) A=0 (BDG's own fixed point): num_i const = 0.9 -> "
      "w_eff=0, y_eff=inf; NOT expressible as plug at any finite (w,y)")
# (d) direction other than g_i
print("     (d) a coefficient affine in F_i but applied to h_i != g_i cannot "
      "reduce: the reduction is a statement about the SCALAR only.")
