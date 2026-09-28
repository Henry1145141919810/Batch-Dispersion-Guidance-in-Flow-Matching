"""Supplement: prop-specific rho for family-wise p of paired z; bdg-only paired/unpaired variance ratios;
cross-draw plug w=1 control counts; design effect of within-batch coupling at eta=1 (analytic, from plug slopes);
power of an n=2048 paired replication."""
import numpy as np, json, math
from scipy import stats
rng = np.random.default_rng(2); M = 13
def pmax(z, rho, N=400000):
    x = math.sqrt(rho)*rng.standard_normal((N, 1)) + math.sqrt(1-rho)*rng.standard_normal((N, M))
    return float((x.max(1) >= z).mean())
P = json.load(open('pairing.json'))
bd = [p for p in P if p['other'].startswith('bdg__') and 'e0t1' not in p['other'] and not p['other'].endswith('o')]
r_all = [p['var_ratio_paired_over_unpaired'] for p in bd]
print('bdg eta=4 vs plug w1, all 6 blocks: n=%d range %.3f-%.3f median %.3f' % (len(r_all), min(r_all), max(r_all), float(np.median(r_all))))
for prop, zu, rho in [('alpha', 1.376, 0.251), ('gap', 1.043, 0.534)]:
    for lab, r in [('r_med', {'alpha': .283, 'gap': .582}[prop]), ('r_max', {'alpha': .377, 'gap': .606}[prop]), ('r_plugw4', {'alpha': .695, 'gap': .679}[prop])]:
        zp = zu/math.sqrt(r)
        print(f'{prop} {lab}={r:.3f}: z_paired {zp:.2f}  FWER13 p (rho={rho}) {pmax(zp, rho):.3f}')
# cross-draw plug w1 control, q50, n=512
for prop, xs, bdg in [('alpha', [23, 32, 35], 33), ('gap', [58, 64, 55], 69)]:
    n = 512; tot = sum(xs); p = tot/(3*n)
    chi2 = sum((x - n*p)**2 for x in xs)/(n*p*(1-p)); pchi = 1-stats.chi2.cdf(chi2, 2)
    se = math.sqrt(p*(1-p)*(1/n + 1/(3*n))); z = (bdg/n - p)/se
    print(f'{prop} plug w1 draws {xs} pooled {p:.4f} homogeneity chi2 {chi2:.2f} p {pchi:.2f}; BDG eta=1 {bdg}/512 vs pooled z {z:.2f}')
# design effect at eta=1: w_eff = V_b/tau^2, sd(w_eff)/w_eff = relse(V_b); var_batch(p) ~ (dp/dw * sd(w_eff))^2
for prop, kurt, p, slope in [('alpha', 3.93, 0.0645, 0.02), ('gap', 2.42, 0.1348, 0.02)]:
    B = 512; relse = math.sqrt(2/(B-1) + (kurt-3)/B)
    for w in [1, 2, 4]:
        vb = (slope*relse*w)**2; vbin = p*(1-p)/B
        print(f'{prop} kurt {kurt} relse(V_b) {relse:.4f}; w_eff {w}: design effect <= {1+vb/vbin:.3f}')
# power of n=2048 paired replication
for prop, r, v in [('alpha', .283, .0645*.9355+.0449*.9551), ('alpha(pess r=.695)', .695, .0645*.9355+.0449*.9551), ('gap', .582, .1348*.8652+.1133*.8867)]:
    se = math.sqrt(r*v/2048)
    print(f'{prop}: n=2048 paired se {se:.4f}; 80% power (one-sided .05) detects d >= {2.487*se:.4f}; power at +0.0196 {1-stats.norm.cdf(1.645-0.0196/se):.3f}; at +0.0098 {1-stats.norm.cdf(1.645-0.0098/se):.3f}')
