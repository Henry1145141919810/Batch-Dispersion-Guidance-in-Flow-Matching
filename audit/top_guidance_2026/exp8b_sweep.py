"""Does DBFG's facet estimator win in the regime it should -- tight proposals, where boundary crossings
are rare and the score-function estimator sees an almost constant reward?  Sweep the proposal scale."""
import numpy as np, warnings
from scipy import stats
warnings.filterwarnings("ignore")
exec(open("dbfg_core.py").read())

def run(scale, NT=3_000_000, R=150):
    global Sig, L, Si
    Sig = (Q * np.linspace(1.0, 0.25, d)) @ Q.T * scale
    L = np.linalg.cholesky(Sig); Si = np.linalg.inv(Sig)
    rg = np.random.default_rng(77)
    Zs = rg.standard_normal((NT, d)) @ L.T + m; lv = ell(Zs)
    Gt = Si @ ((Zs - m) * lv[:, None]).mean(0) / lv.mean()
    res = {}
    for nm, fn, cal in [("SF n=32", lambda r: sf_grad(32, r), 32), ("SF n=128", lambda r: sf_grad(128, r), 128),
                        ("FACET 4x4", lambda r: facet_grad(4, 4, r), 48),
                        ("FACET 12x8", lambda r: facet_grad(12, 8, r), 208)]:
        G = np.array([fn(np.random.default_rng(2000 + r))[0] for r in range(R)])
        if nm.startswith("FACET"):
            Zh = np.array([ell(np.random.default_rng(6000+r).standard_normal((16, d)) @ L.T + m).mean() for r in range(R)])
            G = G / np.maximum(Zh, 1e-9)[:, None]
        res[nm] = (np.sqrt(((G - Gt) ** 2).sum(1).mean()) / np.linalg.norm(Gt), cal)
    return np.linalg.norm(Gt), res

print(f"{'proposal scale':>14s} {'||grad||':>9s} | " + " | ".join(f"{k:>18s}" for k in ["SF n=32 (32)", "SF n=128 (128)", "FACET 4x4 (48)", "FACET 12x8 (208)"]))
for sc in [0.55, 0.25, 0.10, 0.04]:
    nrm, r = run(sc)
    print(f"{sc:14.2f} {nrm:9.4f} | " + " | ".join(f"{r[k][0]:18.3f}" for k in ["SF n=32", "SF n=128", "FACET 4x4", "FACET 12x8"]))
print("\n(relative RMS error of grad_m log Z_D; guide-call budget in parentheses)")
