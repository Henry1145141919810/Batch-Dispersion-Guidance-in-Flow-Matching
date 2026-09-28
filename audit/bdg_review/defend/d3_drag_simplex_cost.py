"""BLUE-TEAM defence checks for BDG, claims (iii) bounded mean drag, (iv) simplex
transfer needs no matrix inverse, (v) cost.  numpy + torch (CPU), no generator loaded.
"""
import json
import glob
import os
import numpy as np

rng = np.random.default_rng(20260925)
print("=" * 78)
print("[D3] THE MEAN DRAG IS EXACTLY w_eff TIMES PLUG'S  (claim iii)")
print("=" * 78)
B, n = 512, 10
eta, s2, y = 4.0, 1.0, 3.0
F = rng.normal(2.0, 1.5, size=B)
Fbar, d = F.mean(), F - F.mean()

# build gains a_i = |J^T g_i|^2 > 0 with a CONTROLLED correlation to d_i
base = np.abs(rng.normal(0, 1, size=B)) + 0.2
for rho_target in (-0.6, 0.0, +0.5, +0.85):
    z = rho_target * (d / d.std()) + np.sqrt(max(0.0, 1 - rho_target ** 2)) * rng.normal(size=B)
    a = np.exp(0.6 * z) * base                      # strictly positive gains
    rho = float(np.corrcoef(a, d)[0, 1])
    for V_ratio in (1.6, 0.55):
        e = V_ratio - 1.0
        w_eff = 1 + eta * e
        num_plug = (y - F)
        num_disp = -eta * e * d
        num_tot = num_plug + num_disp
        move_plug = float(np.mean(num_plug * a) / s2)
        move_disp = float(np.mean(num_disp * a) / s2)
        move_tot = float(np.mean(num_tot * a) / s2)
        # plug's OWN drag = the part of its mean move caused by corr(a, d)
        drag_plug = float(-np.mean(d * a) / s2)
        identity = float((y - Fbar) * np.mean(a) / s2 - w_eff * np.mean(d * a) / s2)
        print(f"  corr(a,d)={rho:+.3f} V/tau^2={V_ratio:<5g} w_eff={w_eff:+.3f}  "
              f"disp drag {move_disp:+9.4f}  plug drag {drag_plug:+9.4f}  "
              f"ratio {move_disp/drag_plug:+8.4f} (eta*e={eta*e:+.4f})  "
              f"total-vs-identity rel err {abs(move_tot-identity)/abs(move_tot):.2e}")
print("  => disp_drag / plug_drag == eta*e exactly, and total mean move ==")
print("     (y-Fbar)*mean(a)/s^2 - w_eff*mean(d*a)/s^2.  The dispersion term introduces NO new")
print("     drag mechanism: it rescales the one plug already has.  Zero iff corr(a,d)=0.")
z0 = np.array([0.0])
print(f"  at corr(a,d)=0 exactly (a constant): disp drag = "
      f"{float(np.mean(-eta*0.6*d*np.ones(B))/s2):+.3e}")

print()
print("=" * 78)
print("[D4] SIMPLEX: BDG IS A PER-SAMPLE SCALAR ON PLUG'S STEP; NO Sigma, NO INVERSE  (claim iv)")
print("=" * 78)
Bs, L, K = 512, 40, 8
logits = rng.normal(size=(Bs, L, K))
p = np.exp(logits - logits.max(-1, keepdims=True))
p /= p.sum(-1, keepdims=True)
Wf = rng.normal(size=(L, K)) * 0.3
Fs = np.einsum('blk,lk->b', p, Wf)                  # a scalar property of the simplex point


def tangent(v):                                     # sum-zero per position
    return v - v.mean(-1, keepdims=True)


# grad of F wrt logits, through softmax: dF/dlogit = p * (Wf - sum_k p Wf)
gl = p * (Wf[None] - np.einsum('blk,lk->bl', p, Wf)[..., None])
gl = tangent(gl)
Fbar_s, ds = Fs.mean(), Fs - Fs.mean()
e = 0.55 - 1.0
num_plug = (3.0 - Fs)
num_bdg = (3.0 - Fs) - eta * e * ds
step_plug = num_plug[:, None, None] * gl
step_bdg = num_bdg[:, None, None] * gl
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = np.where(np.abs(step_plug) > 1e-12, step_bdg / step_plug, np.nan)
spread = np.nanmax(np.nanmax(ratio, axis=(1, 2)) - np.nanmin(ratio, axis=(1, 2)))
print(f"  BDG step / plug step: max WITHIN-sample spread of the elementwise ratio {spread:.3e}")
print(f"    (so BDG's simplex step is a per-sample SCALAR multiple of plug's; anywhere plug's")
print(f"     pullback is valid, BDG's is too)")
print(f"  tangency preserved: max |row sum| plug {np.abs(step_plug.sum(-1)).max():.2e}  "
      f"bdg {np.abs(step_bdg.sum(-1)).max():.2e}")
print(f"  BDG needs from the batch: Fbar and V_b, two scalar reductions. No Sigma, no solve.")

print("\n  contrast -- what the alternatives need on the simplex:")
Sig = np.einsum('blk,kj->blkj', p, np.eye(K)) - np.einsum('blk,blj->blkj', p, p)
ev = np.linalg.eigvalsh(0.5 * (Sig + np.swapaxes(Sig, -1, -2)))
print(f"    BTVG's V_F = g^T Sigma g with Sigma = diag(p) - p p^T (status doc l.435):")
print(f"      per-position eigenvalues: min |lambda| over all {Bs*L} positions = "
      f"{np.abs(ev).min():.3e}, and the smallest eigenvalue is exactly 0 on "
      f"{100*np.mean(np.abs(ev[..., 0]) < 1e-12):.1f}% of positions (eigenvector = 1)")
print(f"      => Sigma is SINGULAR BY CONSTRUCTION; V_F can be 0 for a non-zero g (any g in")
print(f"         the span of 1), and the repo already measured V_F going negative on QM9 from")
print(f"         a merely non-symmetric Sigma (guidance.py:1464-1470).")
# the xproj repair divides by (a.a)
a_dir = tangent(rng.normal(size=(Bs, L, K)))
aa = np.sum(a_dir ** 2, axis=(1, 2))
print(f"    btvg2_xproj's repair needs a (a.b)/(a.a) projection: min a.a over the batch "
      f"{aa.min():.3e} -- a division that has no lower bound")
print(f"    Three_New sec 5.2's allocation needs M^-1 a_i per sample and a (C R C^T)^-1 2x2")
print(f"      solve per step; BDG needs neither.")

print()
print("=" * 78)
print("[D5] COST, FROM THIS REPO'S OWN v2 CELLS  (claim v)")
print("=" * 78)
root = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1/results/full/v2/n5000/seed20261001"
rows = {}
for f in sorted(glob.glob(os.path.join(root, "*q90*full.json"))):
    d = json.load(open(f))
    arm = os.path.basename(f).split("__")[1]
    rows.setdefault(arm, d.get("cost", {}))
for arm, c in rows.items():
    tot_guide = c.get("guide_fwd", 0) + c.get("guide_bwd", 0) + c.get("guide_hvp", 0)
    tot_gen = c.get("gen_fwd", 0) + c.get("gen_vjp", 0) + c.get("gen_jvp", 0)
    print(f"  {arm:<10} guide passes {tot_guide:>6}   gen passes {tot_gen:>6}   {json.dumps(c)}")
pl = rows.get("plug", {})
for arm in ("lgd_mc", "tfg", "tmpd"):
    r = rows.get(arm, {})
    gp = r.get("guide_fwd", 0) + r.get("guide_bwd", 0)
    pp = pl.get("guide_fwd", 0) + pl.get("guide_bwd", 0)
    print(f"  {arm} / plug guide-pass ratio = {gp/pp:.1f}x")
print("  BDG's own added arithmetic, per step: two reductions over B scalars.")
fa = os.path.getsize(r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1/weights/f_A_mu.pt")
print(f"    f_A_mu.pt is {fa/1e6:.2f} MB on disk (~{fa/4/1e6:.2f}M fp32 params), so one f_A")
print(f"    forward+backward over B=512 molecules is O(1e9) flops against BDG's ~2*512 = 1e3.")
print("    Added FLOPs are ~1e-6 of the pass BDG already shares with plug: free in arithmetic.")
print("    NOT free: the two reductions are a batch-global barrier, so the field is no longer")
print("    per-sample decomposable -- the batch cannot be sharded without an all-reduce, and")
print("    B=1 makes V_b literally 0/0 (undefined), so BDG has no single-molecule mode.")
