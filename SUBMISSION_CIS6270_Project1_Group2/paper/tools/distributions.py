"""Measured training distributions and frozen evaluation bands for both modalities."""
from pathlib import Path
import hashlib, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'paper/figs'
VERSION = 10

def build():
    qpath=ROOT/'data/qm9.pt'
    dpath=ROOT/'proj1/m2/blade_bundle/dfb500.npz'
    q=torch.load(qpath,map_location='cpu',weights_only=False)
    tok=np.load(dpath)['train']
    dna={'gc':((tok==1)|(tok==2)).mean(1), 'cpg':((tok[:,:-1]==1)&(tok[:,1:]==2)).mean(1)}
    out={'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (qpath,dpath)},'properties':{}}
    for j,p in enumerate(['mu','alpha','gap','gc','cpg']):
        mol=j<3
        vals=q['y'][q['split']['train_a'],j].numpy() if mol else dna[p]
        cell=next(f for f in (ROOT/('results/v3/fm/v3/n2000' if mol else 'results/m2/m2/n2000')).rglob('*.json') if (lambda r:r['prop']==p and r['arm']=='plug')(json.loads(f.read_text())))
        r=json.loads(cell.read_text())
        target=r['target'] if mol else r['y']; delta=r['delta']
        assert abs(float(np.median(vals))-target)<1e-5,(p,np.median(vals),target)
        # Full support; histogram counts sum to the complete declared training pool.
        if mol:
            counts,edges=np.histogram(vals,bins=64)
        else:
            # Align to the count lattice so bins do not contain unequal counts.
            den=500 if p=='gc' else 499
            ints=np.rint(vals*den).astype(int)
            step=3 if p=='gc' else 1
            edges=np.arange(ints.min()-.5,ints.max()+step+.5,step)/den
            counts,edges=np.histogram(vals,bins=edges)
        assert int(counts.sum())==len(vals)
        out['properties'][p]={'n':len(vals),'pool':'QM9 train_a' if mol else 'DeepFlyBrain training',
            'target':target,'delta':delta,'low':target-delta,'high':target+delta,
            'counts':counts.tolist(),'edges':edges.tolist(),'median':float(np.median(vals)),
            's':None if mol else r['s'],'quantum':None if mol else r['quantum'],
            'band_source':str(cell.relative_to(ROOT)),
            'training_in_band':float((np.abs(vals-target)<=delta).mean())}
    OUT.mkdir(exist_ok=True)
    (ROOT/f'paper/property_distributions_v{VERSION}.json').write_text(json.dumps(out,indent=2))
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(7.7,5.35),layout='constrained')
    grid=fig.add_gridspec(2,6)
    axes=[fig.add_subplot(grid[0,2*i:2*i+2]) for i in range(3)]+[fig.add_subplot(grid[1,:3]),fig.add_subplot(grid[1,3:])]
    labels=[r'Dipole $\mu$ (D)',r'Polarizability $\alpha$ (Bohr$^3$)','HOMO–LUMO gap (Ha)','GC fraction','CpG fraction']
    for ax,(p,r),label in zip(axes,out['properties'].items(),labels):
        e=np.array(r['edges']); h=np.array(r['counts'])/r['n']/np.diff(e)
        ax.stairs(h,e,fill=True,color='#215fa9',alpha=.4,lw=.8)
        ax.axvspan(r['low'],r['high'],color='#d98a1e',alpha=.4,label=r'$y\pm\delta$')
        ax.axvline(r['target'],color='#5d3ca6',lw=1.6,label='q50 target')
        ax.set(xlabel=label,ylabel='Density',title=f"q50 = {r['target']:.5g}; δ = {r['delta']:.5g}")
        ax.legend(frameon=False,fontsize=7)
    for ext in ('pdf','png'): fig.savefig(OUT/f'property_distributions_v{VERSION}.{ext}',dpi=220)
    plt.close(fig)
    print(json.dumps({p:{k:r[k] for k in ('n','target','delta','training_in_band')} for p,r in out['properties'].items()},indent=2))
    return out

if __name__=='__main__': build()
