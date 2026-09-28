"""btvg_var's strength knob (results/sweep __cmp, B200 n=512 seed 20260921): range(sd/delta)/range(bias/delta)."""
import json, math, os
REPO=r"C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1"
def bs(p):
    j=json.load(open(p)); d=j["delta"]; b=j["f_B_mean"]-j["target_mean"]
    return b/d, math.sqrt(max(j["prop_rmse_eval"]**2-b**2,0))/d, j["mol_stability"]
for tgt in ("q50","q90"):
    for prop in ("mu","alpha","gap"):
        rows=[(w,)+bs(f"{REPO}/results/sweep/{prop}__btvg_var__{tgt}__w{w}__tmin0.5__cmp.json")
              for w in ("0.01","0.05","0.25","0.5","1","2","4")
              if os.path.exists(f"{REPO}/results/sweep/{prop}__btvg_var__{tgt}__w{w}__tmin0.5__cmp.json")]
        B=[r[1] for r in rows]; S=[r[2] for r in rows]
        print(f"{prop:5s} {tgt} btvg_var: d(sd)={max(S)-min(S):5.2f} d(bias)={max(B)-min(B):5.2f} ratio={(max(S)-min(S))/max(max(B)-min(B),1e-9):5.2f}")
