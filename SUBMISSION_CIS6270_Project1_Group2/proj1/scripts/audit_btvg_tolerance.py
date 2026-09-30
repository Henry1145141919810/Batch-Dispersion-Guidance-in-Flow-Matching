"""Log BTVG tolerance activation on its own validation trajectories.

This is a mechanistic diagnostic, not a performance comparison. Changes in
the coefficient for other tau values are evaluated at the same recorded V;
they do not predict the effect of re-running the altered policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"proj1/src"))
sys.path.insert(0,str(ROOT/"proj1/scripts"))
from checkpoint_paths import default_generator,require_predictor
from m1_signed_bias import PhysicalProperty,load_fm
from sampling import FlowSampler,initial_noise,integrate


def coefficient(v,tau):
    # Magnitude after the tau^2/s^2 normalization, relative to 1/(2s^2).
    return torch.where(v>0,(1-tau*tau/v.clamp_min(1e-30)).clamp_min(0),torch.zeros_like(v))


class LoggingSampler(FlowSampler):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        self.records=[]

    def field(self,c,f,t):
        result=super().field(c,f,t)
        if self.t_min_guide<=t<1:
            self.records.append((float(t),self.last_diag["btvg_V_raw"].detach().cpu()))
        return result


def summary(v,tau):
    q=coefficient(v,tau);wide=coefficient(v,2*tau)
    active=v>tau*tau
    change=(q-wide).abs()
    return {"n_sample_steps":v.numel(),"nonpositive_V_fraction":float((v<=0).float().mean()),
            "positive_V_stopped_by_tau_fraction":float(((v>0)&~active).float().mean()),
            "variance_active_fraction":float(active.float().mean()),
            "median_tau2_over_V_when_active":float((tau*tau/v[active]).median()) if active.any() else None,
            "median_active_coefficient":float(q[active].median()) if active.any() else None,
            "doubling_tau_switches_activity_fraction":float(((q>0)!=(wide>0)).float().mean()),
            "doubling_tau_coefficient_change_ge_point1_fraction":float((change>=.1).float().mean()),
            "doubling_tau_median_absolute_coefficient_change":float(change.median())}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n",type=int,default=32)
    ap.add_argument("--seed",type=int,default=20260923)
    ap.add_argument("--w",type=float,default=4.)
    a=ap.parse_args()
    # A positive control: the normalized term responds around its threshold.
    assert coefficient(torch.tensor([2.]),1.).item()==.5
    assert coefficient(torch.tensor([2.]),2.).item()==0
    dev="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)
    data=torch.load(ROOT/"data/qm9.pt",map_location="cpu",weights_only=False)
    ids=data["split"]["val"][:a.n]
    mask=data["mask"][ids].to(dev).float()
    fm=default_generator();net,_=load_fm(fm,5,dev)
    out=ROOT/"results/workshop_claim_audit"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for p in ("mu","alpha","gap"):
        fa=PhysicalProperty(require_predictor(f"f_A_{p}.pt"),5,dev)
        ck=torch.load(require_predictor(f"f_B_{p}.pt"),map_location="cpu",weights_only=False)
        delta=2*float(ck["val_mae"]);tau=delta/1.96
        y=data["y"][ids,data["props"].index(p)].to(dev)
        c,f=initial_noise(mask,5,torch.Generator(device=dev).manual_seed(a.seed))
        sampler=LoggingSampler(net,mask,f_net=fa,y=y,s=fa.y_std,mode="btvg",
                               w=a.w*(tau/fa.y_std)**2,tau=tau,t_min_guide=.5,clip=1.,n_probe=1)
        c,f,_=integrate(sampler,c,f,100,"euler")
        assert torch.isfinite(c).all() and torch.isfinite(f).all()
        times=torch.tensor([t for t,_ in sampler.records]);v=torch.stack([v for _,v in sampler.records])
        row={"property":p,"n":a.n,"w":a.w,"tau":tau,"overall":summary(v,tau),
             "early_t_lt_point8":summary(v[times<.8],tau),"late_t_ge_point9":summary(v[times>=.9],tau),
             "per_time":[{"t":float(t),**summary(vv,tau)} for t,vv in zip(times,v)]}
        torch.save({"times":times,"V":v,"tau":tau,"mol_idx":ids},out/f"tolerance_{p}.pt")
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!="per_time"}),flush=True)
    report={"scope":"Exploratory on-policy BTVG validation diagnostic at w4; no terminal performance claims.",
            "seed":a.seed,"mol_idx":ids.tolist(),"n_steps":100,"t_min":.5,"scoring":"No f_B outcome scoring; checkpoint val_mae only defines existing delta.",
            "note":"Doubling-tau comparison holds states and V fixed; coefficient units are relative to 1/(2s^2), not relative change in applied clipped velocity.",
            "checkpoint_sha256":hashlib.sha256(Path(fm).read_bytes()).hexdigest(),
            "guidance_sha256":hashlib.sha256((ROOT/"proj1/src/guidance.py").read_bytes()).hexdigest(),
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"rows":rows}
    (out/"tolerance_activity.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
