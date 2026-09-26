"""Supplementary claim audit; preserve FR5 and all historical results.

Pair by molecule ID, average each target's differences over the three seeds,
then estimate uncertainty across target clusters. Inference is conditional on
these checkpoints and three seeds; this is not a random-training-seed study.
The 36 comparisons (6 controls x 3 properties x 2 metrics) form one descriptive
Bonferroni family. CIs do not establish equivalence or revise registered FR5.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"proj1/scripts"))
from full_run_table import load_cells

PROPS = ("mu", "alpha", "gap")
CONTROLS = ("unguided", "plug", "tmpd", "lgd_mc", "tfg", "btvg_var")
SEEDS = ("20261001", "20261002", "20261003")


def align(pm):
    ids = pm["mol_idx"].long()
    if ids.unique().numel() != len(ids):
        raise ValueError("Duplicate target IDs in a seed")
    order = ids.argsort()
    return {k: v[order] for k, v in pm.items() if isinstance(v, torch.Tensor)}


def interval(d, family=36):
    # d[seed, target]. Each target remains a single independent cluster.
    by_target = d.mean(axis=0)
    effect = float(by_target.mean())
    se = float(by_target.std(ddof=1)/np.sqrt(len(by_target)))
    z95 = NormalDist().inv_cdf(.975)
    zfam = NormalDist().inv_cdf(1-.05/(2*family))
    return {"difference": effect, "target_cluster_se": se,
            "pointwise_95_ci": [effect-z95*se, effect+z95*se],
            "simultaneous_95_ci": [effect-zfam*se, effect+zfam*se],
            "family_size": family, "per_seed_difference": d.mean(axis=1).tolist(),
            "pooled_pair_se_for_reference": float(d.flatten().std(ddof=1)/np.sqrt(d.size))}


def self_checks():
    # Repeated targets cannot be counted as three independent target samples.
    x=np.array([-1.,0.,1.,1.,-1.,0.])
    duplicated=np.tile(x,(3,1))
    r=interval(duplicated)
    assert np.isclose(r["target_cluster_se"],x.std(ddof=1)/len(x)**.5)
    assert r["target_cluster_se"] > r["pooled_pair_se_for_reference"]*1.7
    p={"mol_idx":torch.tensor([7,2,9]),"y":torch.tensor([70.,20.,90.])}
    assert align(p)["y"].tolist()==[20.,70.,90.]
    try:
        align({"mol_idx":torch.tensor([1,1])})
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate IDs were accepted")


def plot(rows,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(10.5,4.8),sharex=True,sharey=True)
    names={"unguided":"Unguided","plug":"Plug-in","tmpd":"TMPD","lgd_mc":"LGD-MC","tfg":"TFG*","btvg_var":"Variance only"}
    for ax,p in zip(axes,PROPS):
        for i,a in enumerate(CONTROLS):
            r=next(r for r in rows if r["property"]==p and r["control"]==a and r["metric"]=="legacy_in_band")
            e=100*r["difference"];lo,hi=np.array(r["simultaneous_95_ci"])*100;l,h=np.array(r["pointwise_95_ci"])*100
            color="#ab3434" if hi<0 else "#355f86"
            ax.plot([lo,hi],[i,i],color=color,lw=1,alpha=.7)
            ax.plot([l,h],[i,i],color=color,lw=3)
            ax.plot(e,i,"o",color=color,ms=5)
        ax.axvline(0,color="0.5",lw=.8,ls="--")
        ax.set_title(p,fontweight="bold")
        ax.set_yticks(range(len(CONTROLS)),[names[a] for a in CONTROLS])
        ax.grid(axis="x",alpha=.15)
    axes[0].invert_yaxis()
    fig.suptitle("BTVG minus each comparator: target-band coverage",fontweight="bold",y=.97)
    fig.supxlabel("Difference in percentage points (positive favors BTVG)",y=.18)
    fig.legend([Line2D([0],[0],color="0.3",lw=3),Line2D([0],[0],color="0.3",lw=1)],
               ["Pointwise 95% CI","Simultaneous 95% CI, 36 contrasts"],loc="lower center",bbox_to_anchor=(.5,.085),ncol=2,frameon=False)
    fig.text(.07,.035,"5,000 target clusters, each averaged over 3 fixed seeds. Historical continuous-feature scoring.\nSupplementary analysis; does not replace FR5. *TFG was added post hoc.",fontsize=9,color=".3")
    fig.subplots_adjust(left=.13,right=.98,bottom=.30,top=.85,wspace=.15)
    fig.savefig(out/"btvg_cluster_intervals.png",dpi=220)
    fig.savefig(out/"btvg_cluster_intervals.svg")
    plt.close(fig)


def main():
    self_checks()
    source=ROOT/"results/full/n5000"
    out=ROOT/"results/workshop_claim_audit"
    out.mkdir(parents=True,exist_ok=True)
    frozen_path=source/"frozen_q90_tfg.json"
    frozen=json.loads(frozen_path.read_text())["frozen_w"]
    cells=load_cells(str(source))
    rows,manifest=[],[]
    ref_ids=None
    for prop in PROPS:
        arrays={}
        for arm in ("btvg",)+CONTROLS:
            pieces=[]
            ref_y=None
            for seed in SEEDS:
                meta,path=cells[seed][(prop,arm,float(frozen[arm][prop]))]
                p=Path(path)
                pm=align(torch.load(p,map_location="cpu",weights_only=False))
                assert str(meta["seed"])==seed and meta["n"]==5000 and meta["target_name"]=="dist"
                if ref_ids is None: ref_ids=pm["mol_idx"]
                assert torch.equal(ref_ids,pm["mol_idx"])
                if ref_y is None: ref_y=pm["y"]
                assert torch.equal(ref_y,pm["y"])
                hit=((pm["f_B"]-pm["y"]).abs()<=meta["delta"]) & pm["finite"].bool()
                useful=hit & pm["mol_stable"].bool()
                assert abs(float(hit.double().mean())-meta["in_band_fraction"])<1e-7
                pieces.append({"hit":hit.numpy(),"useful":useful.numpy(),"y":pm["y"],"meta":meta,"pm":pm})
                manifest.append({"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
            arrays[arm]=pieces
        for control in CONTROLS:
            for metric,key in (("legacy_in_band","hit"),("legacy_stable_and_in_band","useful")):
                d=[]
                for ours,base in zip(arrays["btvg"],arrays[control]):
                    assert torch.equal(ours["y"],base["y"])
                    assert ours["meta"]["delta"]==base["meta"]["delta"]
                    assert ours["meta"]["prov"]["fm_md5"]==base["meta"]["prov"]["fm_md5"]
                    d.append(ours[key].astype(float)-base[key].astype(float))
                row={"property":prop,"control":control,"metric":metric,
                     "ours_w":frozen["btvg"][prop],"control_w":frozen[control][prop],
                     **interval(np.stack(d))}
                rows.append(row)
    report={"analysis":"Post hoc supplementary paired target-cluster normal intervals; fixed checkpoint and three fixed seeds.",
            "n_target_clusters":5000,"seeds":SEEDS,"family_size":36,
            "scoring":"Historical continuous-feature f_B; most full-run sidecars have no coordinates, so decoded re-scoring cannot be recovered from them.",
            "limits":"No equivalence test; no random-training-seed uncertainty; no change to FR5; TFG post hoc. The 36-contrast Bonferroni family covers comparisons in this supplementary analysis, not all historical exploration or adaptive method development.",
            "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "frozen_sha256":hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
            "rows":rows,"sources":manifest}
    (out/"cluster_intervals.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Supplementary target-cluster analysis", "",report["analysis"],"",report["scoring"],"",report["limits"],"",
           "Intervals below are percentage points for BTVG minus the comparator.","",
           "| Property | Comparator | In-band difference | Pointwise 95% CI | Simultaneous 95% CI |", "|---|---|---:|---|---|"]
    for r in rows:
        if r["metric"]!="legacy_in_band":continue
        f=lambda xs: "["+", ".join(f"{100*x:+.2f}" for x in xs)+"]"
        lines.append(f"| {r['property']} | {r['control']} | {100*r['difference']:+.2f} | {f(r['pointwise_95_ci'])} | {f(r['simultaneous_95_ci'])} |")
    (out/"cluster_intervals.md").write_text("\n".join(lines)+"\n")
    plot(rows,out)
    print("\n".join(lines))
    print("Self-checks passed: target clustering, ID permutation, duplicate rejection; all 63 source scores reproduced.")


if __name__=="__main__":
    main()
