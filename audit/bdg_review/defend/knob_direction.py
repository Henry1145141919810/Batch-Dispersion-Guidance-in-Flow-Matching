"""Which way does each knob move the batch: centring (bias) or spread (sd)?

plug's knob = strength w (results/sweep, B200 batch 128 n=512 seed 20260921, __cmp cells, w 0.01..4).
bdg's knob  = tau_mult at fixed w=1, eta=4 (local port, 5080, n=256, seed 20260925, 0.5..1.5).
Both scored in delta units with f_B (continuous; sweep JSONs carry no sidecar), sd from
sqrt(rmse^2 - bias^2) (identity validated by the data lens to 2.5e-6).
Statistic: over each knob's ladder, range(sd/delta) / range(bias/delta).  > 1 = spread-dominant knob.
Zero compute: existing cells only.
"""
import json, math, os, glob

REPO = r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
SP = r"C:/Users/MOOOOO~1/AppData/Local/Temp/claude/c--Users-mooooonesy-Downloads-pennstuff-cis-6270-Project-1/7822c1d3-5ed2-4605-bb2c-4e0412afabed/scratchpad/bdg"


def bs(path):
    j = json.load(open(path))
    d = j["delta"]
    b = j["f_B_mean"] - j["target_mean"]
    sd = math.sqrt(max(j["prop_rmse_eval"] ** 2 - b ** 2, 0.0))
    return b / d, sd / d, j["mol_stability"], j["in_band_fraction"]


out = {}
for tgt in ("q50", "q90"):
    for prop in ("mu", "alpha", "gap"):
        pl = []
        for w in ("0.01", "0.05", "0.25", "0.5", "1", "2", "4"):
            p = f"{REPO}/results/sweep/{prop}__plug__{tgt}__w{w}__tmin0.5__cmp.json"
            if os.path.exists(p):
                pl.append((w,) + bs(p))
        bd = []
        for t in ("0.5", "0.75", "1", "1.25", "1.5"):
            p = f"{SP}/cells/{prop}__bdg__{tgt}__w1__tmin0.5__e4t{t}.json"
            if os.path.exists(p):
                bd.append((t,) + bs(p))
        # plug over the w range that stays within the chemistry range bdg spans
        def ratio(rows):
            B = [r[1] for r in rows]; S = [r[2] for r in rows]
            return (max(S) - min(S)) / max(max(B) - min(B), 1e-9), max(S) - min(S), max(B) - min(B)
        rp, dsp, dbp = ratio(pl)
        rb, dsb, dbb = ratio(bd)
        # plug restricted to w<=1 (the chemistry band bdg's ladder mostly lives in)
        rp1, dsp1, dbp1 = ratio([r for r in pl if float(r[0]) <= 1])
        print(f"{prop:5s} {tgt}: plug w0.01-4  d(sd)={dsp:5.2f} d(bias)={dbp:5.2f} ratio={rp:5.2f} | "
              f"plug w<=1 ratio={rp1:5.2f} (d sd {dsp1:4.2f}, d bias {dbp1:4.2f}) | "
              f"bdg tau0.5-1.5 d(sd)={dsb:5.2f} d(bias)={dbb:5.2f} ratio={rb:5.2f}")
        out[f"{prop}_{tgt}"] = dict(plug=pl, bdg=bd, plug_ratio=rp, plug_ratio_w_le1=rp1, bdg_ratio=rb)
json.dump(out, open(SP + "/defend/knob_direction.json", "w"), indent=1)
