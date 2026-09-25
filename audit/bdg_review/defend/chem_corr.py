"""Does chemistry track |w_eff| once the block (prop x target) baseline is removed?"""
import json, glob, os, numpy as np
C = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cells")
rows = []
for f in glob.glob(os.path.join(C, "*__bdg__*.json")):
    j = json.load(open(f))
    if j["variant"].endswith("o"):
        continue
    rows.append((j["prop"] + j["target_name"], j["bdg_tau_mult"], j["bdg_eta"], j["diag"]["bdg_w_eff"],
                 j["mol_stability"]))
b = np.array([r[0] for r in rows]); tm = np.array([r[1] for r in rows]); eta = np.array([r[2] for r in rows])
we = np.array([r[3] for r in rows]); ms = np.array([r[4] for r in rows])
print("n cells", len(rows))
print("raw     corr(|w_eff|, ms) %+.3f   corr(tau_mult, ms) %+.3f" % (np.corrcoef(abs(we), ms)[0, 1], np.corrcoef(tm, ms)[0, 1]))
dm = lambda x: x - np.array([x[b == k].mean() for k in b])
print("within-block corr(|w_eff|, ms) %+.3f   corr(tau_mult, ms) %+.3f" % (
    np.corrcoef(dm(abs(we)), dm(ms))[0, 1], np.corrcoef(dm(tm), dm(ms))[0, 1]))
m = eta > 0
print("eta=4 only, within-block corr(|w_eff|, ms) %+.3f   corr(tau_mult, ms) %+.3f  corr(|w_eff|,tau) %+.3f" % (
    np.corrcoef(dm(abs(we))[m], dm(ms)[m])[0, 1], np.corrcoef(dm(tm)[m], dm(ms)[m])[0, 1],
    np.corrcoef(abs(we)[m], tm[m])[0, 1]))
m2 = m & (tm >= 1)
print("eta=4, widening half (tau>=1): corr(|w_eff|, ms) within-block %+.3f (n=%d)" % (
    np.corrcoef(dm(abs(we))[m2], dm(ms)[m2])[0, 1], m2.sum()))
