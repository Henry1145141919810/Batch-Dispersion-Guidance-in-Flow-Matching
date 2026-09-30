"""Does guidance actually DO anything, and are the arms distinct?
Three gates that must pass before any sweep is worth running."""
import sys, torch, importlib
R='$PROJECT_ROOT'
sys.path.insert(0, R+'/proj1/m2')
import simplex_fm as S, m2_sweep as M
torch.set_num_threads(28)

ck = torch.load(R+'/proj1/m2/blade_bundle/fm_m2_dfb500.pt', map_location='cpu',
                weights_only=False)
net = S.SimplexFM(ck['hidden'], ck['layers']); net.load_state_dict(ck['state_dict']); net.eval()
s = ck['gc_std']; L = ck['crop']
Xa,_,_ = S.load_dfb(crop=L)
gcd = S.gc_hard(Xa)
y50, y90 = float(gcd.median()), float(gcd.quantile(0.90))
print(f"s={s:.4f}  y_q50={y50:.4f}  y_q90={y90:.4f}  (data mean {gcd.mean():.4f})")
N, NFE = 128, 100
real_kmer = M.kmer_freq(Xa[:4096])
delta = 0.16*s

def cell(arm, w, y, variant='-', eta=0.0, tau=None, onesided=False, seed=0,
         n_mc=8, prop='gc', tmin=0.0):
    r,_ = M.run_cell(net, ck, arm, variant, w, y, s, tau, eta, onesided,
                     N, NFE, tmin, 1.0, seed, delta, real_kmer, n_mc=n_mc,
                     prop=prop)
    return r

print("\n=== GATE 1: does guidance move the property toward a q90 target? ===")
print(f"{'arm':>8} {'w':>5} {'gc_mean':>9} {'toward y90':>11} {'gc_sd':>8} {'in_band':>8}")
base = cell('unguided', 0.0, y90)
print(f"{'unguided':>8} {'-':>5} {base['gc_mean']:9.4f} {'-':>11} {base['gc_sd']:8.4f} {base['in_band_fraction']:8.3f}")
for w in (4.0, 16.0, 64.0):
    r = cell('plug', w, y90)
    moved = (r['gc_mean'] - base['gc_mean']) / (y90 - base['gc_mean']) * 100
    print(f"{'plug':>8} {w:5.0f} {r['gc_mean']:9.4f} {moved:10.1f}% {r['gc_sd']:8.4f} {r['in_band_fraction']:8.3f}")

print("\n=== GATE 2: bdg at eta=0 must equal plug EXACTLY ===")
a = cell('plug', 16.0, y90, seed=7)
b = cell('bdg', 16.0, y90, variant='e0t1', eta=0.0, tau=s, seed=7)
for key in ('gc_mean','gc_sd','in_band_fraction','decode_conf'):
    d = abs(a[key]-b[key])
    print(f"  {key:20s} plug {a[key]:.8f}  bdg_e0 {b[key]:.8f}  |d| {d:.2e}  {'OK' if d<1e-12 else 'MISMATCH'}")

print("\n=== GATE 3: are the four published arms distinct at w=16, t_min=0? ===")
print(f"{'arm':>8} {'gc_mean':>9} {'gc_sd':>8} {'in_band':>8} {'vs plug gc_mean':>16}")
p = cell('plug', 16.0, y90, seed=3)
print(f"{'plug':>8} {p['gc_mean']:9.4f} {p['gc_sd']:8.4f} {p['in_band_fraction']:8.3f} {'-':>16}")
for arm in ('tmpd','lgd_mc','tfg_mc'):
    r = cell(arm, 16.0, y90, seed=3)
    print(f"{arm:>8} {r['gc_mean']:9.4f} {r['gc_sd']:8.4f} {r['in_band_fraction']:8.3f} "
          f"{r['gc_mean']-p['gc_mean']:+16.6f}")
    if 'tmpd_vf_over_s2' in r['diag']:
        print(f"         diag v_f/s^2 = {r['diag']['tmpd_vf_over_s2']:.4f}")
    if 'mc_r' in r['diag']:
        print(f"         diag mc_r = {r['diag']['mc_r']:.5f}  obs_sd = {r['diag'].get('mc_obs_sd',0):.5f}")

print("\n=== GATE 4: does the CpG (non-affine) property separate the arms? ===")
import importlib
gcd_c = S.cpg_hard(Xa); s_c = float(gcd_c.std())
y90c = float(gcd_c.quantile(0.90))
print(f"cpg: sd {s_c:.5f}  y90 {y90c:.5f}")
print(f"{'arm':>8} {'cpg_mean':>9} {'cpg_sd':>8} {'in_band':>8} {'vs plug':>12}")
import m2_sweep as MM
def ccell(arm, w=16.0, seed=3):
    r,_ = MM.run_cell(net, ck, arm, '-', w, y90c, s_c, None, 0.0, False,
                      N, NFE, 0.0, 1.0, seed, max(0.16*s_c, 4.4/(L-1)),
                      real_kmer, n_mc=8, prop='cpg')
    return r
pc = ccell('plug')
print(f"{'plug':>8} {pc['gc_mean']:9.5f} {pc['gc_sd']:8.5f} {pc['in_band_fraction']:8.3f} {'-':>12}")
for arm in ('tmpd','lgd_mc','tfg_mc'):
    r = ccell(arm)
    print(f"{arm:>8} {r['gc_mean']:9.5f} {r['gc_sd']:8.5f} {r['in_band_fraction']:8.3f} "
          f"{r['gc_mean']-pc['gc_mean']:+12.6f}")
