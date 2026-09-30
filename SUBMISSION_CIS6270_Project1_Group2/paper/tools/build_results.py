"""Rebuild v9 figures and inline table blocks from the final n=2000 cells.

Run from the repository root. No pilot, n=5000, or cross-strength pooling.
The manifest stores source hashes, seed measurements, and comparison statistics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import statistics as st
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / 'paper'
sys.path.insert(0, str(ROOT / 'proj1/scripts'))
sys.path.insert(0, str(ROOT / 'proj1/m2'))
import v3_final_summary as V
import m2_v3_results as M

PROPS = ('mu', 'alpha', 'gap')
ARMS = V.HEAD_ARMS
NAMES = {'unguided': 'Unguided', 'plug': 'DPS-style plug-in',
         'tmpd': 'TMPD-inspired', 'lgd_mc': 'LGD-MC', 'tfg': 'TFG',
         'tfg_mc': 'TFG-MC adaptation', 'bdg_e4t0.5': r'BDG, $\tau_m=0.5$',
         'bdg_e4t1': r'BDG, $\tau_m=1$'}
PL = {'mu': r'$\mu$', 'alpha': r'$\alpha$', 'gap': 'gap', 'gc': 'GC', 'cpg': 'CpG'}
KEY = 'in_band_fraction_dec'
# No version suffix: this is the final draft. Figures and manifests are
# written under stable names so a rebuild overwrites in place and the
# paper's \includegraphics never has to be re-pointed.
# The older summary module predates arrival of the VP benchmark. Reuse its
# strict cell validation with the protocol's two VP arms explicitly registered.
V.BACKEND_ARMS['vp'] = ('unguided', 'plug')
heads = {be: V.m1_load(str(ROOT/'results/v3'), be, 'v3') for be in ('fm','vp','equifm','edm')}
abl = {(be,w): V.m1_load(str(ROOT/'results/v3'), be, 'v3abl', w) for be in ('fm','equifm') for w in (1,4)}
m2rows = M.load(str(ROOT/'results/m2'), 2000)
M.duplicate_configs(m2rows)
winrows=[]
for path in sorted((ROOT/'results/m2/m2win/n2000').rglob('*.json')):
    r=json.loads(path.read_text()); r.update(_path=str(path),_stage='m2win',_base=path.name)
    winrows.append(r)

taurows=[]
for path in sorted((ROOT/'results/m2/m2tau/n2000').rglob('*.json')):
    r=json.loads(path.read_text()); r.update(_path=str(path),_stage='m2tau',_base=path.name)
    taurows.append(r)

def tau(a, prop='cpg', w=4., tmin=.3, seeds=3):
    """Setpoint probe cells. Outside the registered grid, so seeds may be 1."""
    rows=[r for r in taurows
          if (r['prop'],M.arm_key(r),r['w'],r['t_min_guide'])==(prop,a,w,tmin)]
    assert len(rows)==seeds, (a,prop,w,tmin,len(rows))
    M.check_pinned(rows,'%s tau probe %s'%(prop.upper(),a))
    return sorted(rows,key=lambda r:r['seed'])

def win(a, tmin=.3, prop='gc', w=4.):
    rows=[r for r in winrows if (r['prop'],M.arm_key(r),r['w'],r['t_min_guide'])==(prop,a,w,tmin)]
    assert sorted(r['seed'] for r in rows)==[20260921,20260922,20260923],(prop,a,w,tmin)
    M.check_pinned(rows,'%s t>=%g %s'%(prop.upper(),tmin,a))
    return rows

def seq(p,a,w,stage='m2',tmin=.5):
    rows = [r for r in m2rows if (r['prop'],M.arm_key(r),r['w'],r['_stage'],r['t_min_guide']) == (p,a,float(w),stage,tmin)]
    assert sorted(r['seed'] for r in rows)==[20260921,20260922,20260923], (p,a,w,stage)
    M.check_pinned(rows, str((p,a,w,stage)))
    return rows

def avg(rs,k): return st.mean(r[k] for r in rs)
def sd(rs,k): return st.stdev(r[k] for r in rs)
def pm(rs,k,scale=100,digits=1):
    return ('$%.*f\\pm%.*f$' % (digits,scale*avg(rs,k),digits,scale*sd(rs,k)))
def spread(rs):
    # Pool second moments, not the mean RMSE squared.
    bias = st.mean(r['f_B_dec_mean']-r['target_mean'] for r in rs)
    return math.sqrt(max(0,st.mean(r['prop_rmse_eval_dec']**2 for r in rs)-bias**2))
def cell_hash(p):
    """sha256 of a result cell, identifying its CONTENT rather than its bytes.

    Any \r\n is normalised away first. A cell is JSON written by our own
    scripts, so its line endings depend on the OS that wrote it and on the git
    checkout it came through (core.autocrlf), not on anything measured. Hashing
    the raw bytes made this manifest checkout-dependent: on a Windows checkout
    1055 of the 1073 hashes differed from the committed ones, every one of them
    for a file whose data was identical, and --check then failed the publication
    gate with "Source files changed; rebuild before publication." The normalised
    hash is what git stores for the same file, so it agrees on every platform.
    """
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def contrast(a,b,key=KEY,bar=2.99):
    assert [r['seed'] for r in a]==[r['seed'] for r in b], 'Paired seed order differs'
    pa,pb=avg(a,key),avg(b,key)
    se=math.sqrt(pa*(1-pa)/6000+pb*(1-pb)/6000)
    return dict(delta_pp=100*(pa-pb),z=(pa-pb)/se,mdd_pp=bar*100*se,
                seed_deltas_pp=[100*(ra[key]-rb[key]) for ra,rb in zip(a,b)])
def duration(rs,m2=False): return avg(rs,'minutes')*60/2000 if m2 else avg(rs,'seconds')/2000
def bias(rs): return avg(rs,'f_B_dec_mean')-avg(rs,'target_mean')
def exchange(be,p,eta=None):
    """Descriptive decoded-IB gain per stability loss from one zero-gain cell.

    eta=None compares w=4 zero gain with w=1 zero gain. Otherwise only eta
    and tau change, at w=1 and tau_m=.5. Use ablation-stage rows throughout.
    """
    base=abl[be,1][p,'bdg_e0t1']
    arm=abl[be,4][p,'bdg_e0t1'] if eta is None else abl[be,1][p,f'bdg_e{eta}t0.5']
    delta=100*(avg(arm,KEY)-avg(base,KEY))
    loss=100*(avg(base,'mol_stability')-avg(arm,'mol_stability'))
    assert loss>0
    return dict(delta_ib_pp=delta,stability_loss_pp=loss,ratio=delta/loss)

def diagnostics():
    out={'exchange':{},'gain_coverage_spearman':{},'spread_ratios':{},'guided_contrasts':{}}
    for be in ('fm','equifm'):
        for p in PROPS:
            out['exchange'][f'{be}/{p}']={str(e):exchange(be,p,e) for e in (4,8,None)}
            u=spread(heads[be][p,'unguided'])
            out['spread_ratios'][f'{be}/{p}']=[spread(abl[be,1][p,f'bdg_e8t{t:g}'])/u for t in (.5,.75,1,1.5)]
            for w in (1,4):
                rs=[abl[be,w][p,a] for a in V.ABL_ARMS if a!='bdg_e0t1']
                out['gain_coverage_spearman'][f'{be}/{p}/{w}']=float(spearmanr([st.mean(r['diag']['bdg_w_eff'] for r in row) for row in rs],[avg(row,KEY) for row in rs]).statistic)
    for be in ('fm','vp'):
        for p in PROPS: out['guided_contrasts'][f'{be}/{p}']=contrast(heads[be][p,'plug'],heads[be][p,'unguided'])
    return out
_NUM = re.compile(r'-?\d+\.?\d*')

def _bare(c):
    """A cell stripped of its row terminator and surrounding blanks."""
    return c.strip().removesuffix(r'\\').strip()


def _num(c):
    """The leading number of one already-split piece, or None."""
    c = _bare(c)
    if c in ('--', '') or '$^' in c:
        return None
    m = _NUM.search(c.replace(r'\pm', ' '))
    return float(m.group()) if m else None


def _cellnum(c):
    """Leading number of a cell, or None when the cell has no single value.

    A cell like '12.3 / 45.6' carries two properties at once, so it has no one
    value. Those columns are ranked component-wise instead, through parts=True.
    """
    c = _bare(c)
    if '/' in c:
        return None
    return _num(c)


def _wrap(c, rank):
    """Bold (rank 0) or underline (rank 1), keeping any surrounding math mode."""
    if c.startswith('$') and c.endswith('$'):
        return '$' + ((r'\mathbf{%s}' if rank == 0 else r'\underline{%s}') % c[1:-1]) + '$'
    return (r'\textbf{%s}' if rank == 0 else r'\underline{%s}') % c

def decorate(rows, cols, better='max', skip=()):
    """Bold the best entry of each given column and underline the runner-up.

    `cols` are 0-based indices into the &-separated cells. `skip` names row
    labels left out of the ranking: the zero-gain control is BDG with its
    feedback switched off, i.e. plug-in under another name, so ranking it as a
    separate method would enter one arm twice.
    """
    for ci in cols:
        vals = {}
        for i, r in enumerate(rows):
            if r.strip() == r'\midrule':
                continue
            cells = r.split('&')
            if ci >= len(cells) or any(k in cells[0] for k in skip):
                continue
            v = _cellnum(cells[ci])
            if v is not None:
                vals[i] = v
        if len(vals) < 2:
            continue
        order = sorted(vals, key=lambda i: vals[i], reverse=(better == 'max'))
        best, second = order[0], order[1]
        if vals[best] == vals[second]:
            continue          # a tie for first: marking one would invent a winner
        marks = [(0, best)]
        if len(order) < 3 or vals[second] != vals[order[2]]:
            marks.append((1, second))   # otherwise the runner-up is a tie too
        for rank, i in marks:
            cells = rows[i].split('&')
            raw = cells[ci]
            end = r'\\' if raw.rstrip().endswith(r'\\') else ''
            cells[ci] = ' ' + _wrap(_bare(raw), rank) + end
            rows[i] = '&'.join(cells)
    return rows


def decorate_parts(rows, cols, better='max', skip=()):
    """Rank a 'a / b / c' column one component at a time.

    Those cells pack several properties into one column to save width, so the
    components are separate comparisons and a single winner would be
    meaningless. Rows whose component count differs are left out.
    """
    for ci in cols:
        live = [i for i, r in enumerate(rows)
                if r.strip() != r'\midrule' and ci < len(r.split('&'))
                and not any(k in r.split('&')[0] for k in skip)]
        parts = {i: _bare(rows[i].split('&')[ci]).split('/') for i in live}
        if not parts:
            continue
        width = max(len(v) for v in parts.values())
        live = [i for i in live if len(parts[i]) == width]
        for k in range(width):
            vals = {i: _num(parts[i][k]) for i in live}
            vals = {i: v for i, v in vals.items() if v is not None}
            if len(vals) < 2:
                continue
            order = sorted(vals, key=lambda i: vals[i], reverse=(better == 'max'))
            if vals[order[0]] == vals[order[1]]:
                continue
            marks = [(0, order[0])]
            if len(order) < 3 or vals[order[1]] != vals[order[2]]:
                marks.append((1, order[1]))
            for rank, i in marks:
                parts[i][k] = ' ' + _wrap(_bare(parts[i][k]), rank) + ' '
        for i in live:
            cells = rows[i].split('&')
            end = r'\\' if cells[ci].rstrip().endswith(r'\\') else ''
            cells[ci] = ' ' + ' / '.join(x.strip() for x in parts[i]) + end
            rows[i] = '&'.join(cells)
    return rows


def decorate_across(rows, cols, better='max'):
    """Rank ACROSS the given columns within each row, not down a column.

    For a table whose columns are the methods and whose rows are separate
    tasks, the comparison runs along the row; ranking down a column would
    compare different properties against each other.
    """
    for i, r in enumerate(rows):
        if r.strip() == r'\midrule':
            continue
        cells = r.split('&')
        vals = {c: _cellnum(cells[c]) for c in cols if c < len(cells)}
        vals = {c: v for c, v in vals.items() if v is not None}
        if len(vals) < 2:
            continue
        order = sorted(vals, key=lambda c: vals[c], reverse=(better == 'max'))
        if vals[order[0]] == vals[order[1]]:
            continue
        marks = [(0, order[0])]
        if len(order) < 3 or vals[order[1]] != vals[order[2]]:
            marks.append((1, order[1]))
        for rank, c in marks:
            end = r'\\' if cells[c].rstrip().endswith(r'\\') else ''
            cells[c] = ' ' + _wrap(_bare(cells[c]), rank) + end
        rows[i] = '&'.join(cells)
    return rows


def decorate_panels(rows, cols, better='max', skip=(), fn=None):
    """decorate() one \\midrule-separated block at a time.

    A table whose panels are different properties must not be ranked as one
    list: GC and CpG coverage sit at different absolute levels for reasons of
    band width alone (Appendix A.9 says so explicitly), so a global best marks
    the property with the looser band instead of the better method.
    """
    out, block = [], []
    for r in rows + [r'\midrule']:
        if r.strip() == r'\midrule':
            out.extend((fn or decorate)(block, cols, better, skip)); out.append(r); block = []
        else:
            block.append(r)
    return out[:-1]


MARK = (' Bold marks the best entry in each marked column and underline the runner-up; a column is marked only where one direction is better, and a tie is left unmarked, never broken arbitrarily.')
MARK_PANEL = (' Bold marks the best entry of each panel in each marked column and underline the runner-up; panels are ranked separately because their rows are not comparable across panels, a column is marked only where one direction is better, and a tie is left unmarked.')

def latex_table(caption,label,cols,header,rows):
    return '\n'.join([r'\begin{table}[!htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r'\\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{table}'])

def tables():
    out={}
    rows=[]
    for be,name,nfe in [('fm','FM, ours',100),('vp','VP diffusion, ours',101),('equifm','EquiFM (flow)',100),('edm','EDMsecond (diffusion)',101)]:
        rs=heads[be][('mu','unguided')]
        time=f'{duration(rs):.3f}'+(r'$^{*}$' if be=='vp' else '')
        rows.append(f'{name} & {pm(rs,"mol_stability")} & {pm(rs,"validity")} & {100*avg(rs,"uniqueness_of_valid"):.1f} & {nfe} & {time}'+r'\\')
        if be=='vp': rows.append(r'\midrule')
    rows=decorate_panels(rows,[1,2,3],'max')  # stability, validity, uniqueness
    out['BASE']=latex_table('Unguided QM9, $3\\times2{,}000$ samples: percentages, mean $\\pm$ seed sd. Lower rows use pretrained external models. Timing includes scoring: RTX A6000 except $^*$B200 MIG; runtimes are not hardware matched.'+MARK_PANEL,'tab:fmvd','lccccc',r'Generator & Mol. stable$\uparrow$ & Valid$\uparrow$ & Unique$\uparrow$ & NFE & s/sample',rows)
    rows=[]
    for w,a in ((1,'bdg_e0t1'),(4,'bdg_e0t1'),(1,'bdg_e4t0.5'),(1,'bdg_e4t1'),(1,'bdg_e8t0.5'),(1,'bdg_e8t1.5')):
        rs=abl['fm',w]['mu',a]
        e,t=a.removeprefix('bdg_e').split('t')
        if e=='0': t='--'
        rows.append(f'{w} & {e} & {t} & '+pm(rs,KEY)+f' & {avg(rs,"prop_mae_eval_dec"):.3f} & {avg(rs,"f_B_dec_mean"):.3f} & {spread(rs):.3f} & {100*avg(rs,"mol_stability"):.1f}'+r'\\')
    rows=decorate(rows,[3],'max'); rows=decorate(rows,[4],'min')
    rows=decorate(rows,[7],'max')  # Stable; Mean and sigma have no better direction
    out['ABLATION']=latex_table('Dipole ablation on our FM: q50 $y=2.4932$ D, band $\\pm0.17541$ D. IB is percent mean $\\pm$ seed sd; stability is percent mean. Decoded MAE, mean and spread $\\sigma$ are in D. Zero gain is plug-in; its $w=4$ row tests stronger guidance. $\\tau_m$ is written -- at $\\eta=0$, where $w_{\\mathrm{eff}}=1$ and the setpoint has no effect. Coverage alone is not a quality ranking, and the quality columns show what each arm spent to reach it.'+MARK,'tab:ablation','cccccccc',r'$w$ & $\eta$ & $\tau_m$ & IB$\uparrow$ & MAE$\downarrow$ & Mean & $\sigma$ & Stable$\uparrow$',rows)
    rows=[]
    for a in ARMS:
        name=r'Plug-in ($\eta=0$)' if a=='plug' else NAMES[a]
        rs=heads['fm']['mu',a]
        rows.append(name+' & '+' & '.join(pm(heads['fm'][(p,a)],KEY) for p in PROPS)+' & '+' / '.join(f'{100*avg(heads["fm"][(p,a)],"mol_stability"):.1f}' for p in PROPS)+f' & {100*avg(rs,"unique_valid_per_sample"):.1f} & {duration(rs):.3f}'+r'\\')
    rows=decorate(rows,[1,2,3],'max'); rows=decorate(rows,[5],'max')
    rows=decorate_parts(rows,[4],'max')   # Stable, one component per property
    rows=decorate(rows,[6],'min')         # time: lower is cheaper
    out['M1']=latex_table('Our FM, $w=1$: decoded IB, percent mean $\\pm$ seed sd; stability means in $\\mu/\\alpha/\\mathrm{gap}$ order. DV is distinct-valid yield (\\%), and time is s/sample, both for $\\mu$. Plug-in is $\\eta=0$; BDG uses $\\eta=4$. Rows are matched-backbone adaptations. Coverage alone is not a quality ranking, and the quality columns show what each arm spent to reach it.'+MARK,'tab:recent','lcccccc',r'Method & IB $\mu\uparrow$ & IB $\alpha\uparrow$ & IB gap$\uparrow$ & Stable$\uparrow$ & DV$\uparrow$ & Time',rows)
    rows=[]
    for a in M.HEADLINE_ARMS:
        # Keep all adaptations visible and report fidelity at the same strength.
        gc,cp=seq('gc',a,4),seq('cpg',a,4)
        rows.append(NAMES[a]+' & '+pm(gc,'in_band_fraction')+' & '+pm(cp,'in_band_fraction')+' & '+f'{1e4*avg(gc,"kmer_js"):.2f} / {1e4*avg(cp,"kmer_js"):.2f} & {avg(gc,"diversity"):.4f} / {avg(cp,"diversity"):.4f} & {duration(cp,True):.3f}'+r'\\')
    rows=decorate(rows,[1,2],'max'); rows=decorate_parts(rows,[3],'min')
    rows=decorate_parts(rows,[5],'min')   # time per property; Hamming has no better direction
    out['M2']=latex_table('DNA, $w=4$, $t\\ge0.5$: decoded IB (percent mean $\\pm$ seed sd); JSD ($\\times10^{-4}$) and Hamming diversity in GC/CpG order. Time is recorded s/sequence for CpG. TFG-MC is a restricted adaptation. Full results: Table~\\ref{tab:full-m2}. Coverage alone is not a quality ranking, and the quality columns show what each arm spent to reach it.'+MARK,'tab:m2','lccccc',r'Method & GC IB$\uparrow$ & CpG IB$\uparrow$ & JSD$\downarrow$ & Hamming & Time',rows)
    out['M2']=out['M2'].replace(r'\centering\small',r'\centering\small\setlength{\tabcolsep}{4pt}')
    # Preserve the original window in the appendix; show the complete new GC
    # comparison beside the still-late CpG comparison, with windows in headers.
    out['M2-LATE']=out['M2'].replace('tab:m2','tab:m2-late')
    rows=[]
    for a in M.HEADLINE_ARMS:
        # Both properties now sit at t >= 0.3. The late window left tmpd and
        # lgd_mc numerically identical to plug-in, so it could not separate the
        # comparators; every arm and both properties have been rerun here.
        gc, cp = win(a), win(a, prop='cpg')
        rows.append(NAMES[a]+' & '+pm(gc,'in_band_fraction')+' & '+pm(cp,'in_band_fraction')
                    +f' & {1e4*avg(gc,"kmer_js"):.2f} / {1e4*avg(cp,"kmer_js"):.2f}'
                    +f' & {avg(gc,"diversity"):.4f} / {avg(cp,"diversity"):.4f}'
                    +f' & {duration(gc,True):.3f} / {duration(cp,True):.3f}'+r'\\')
    rows=decorate(rows,[1,2],'max'); rows=decorate_parts(rows,[3],'min')
    rows=decorate_parts(rows,[5],'min')
    out['M2']=latex_table('DNA, $w=4$, $t\\ge0.3$ for both properties. IB is percent mean $\\pm$ seed sd; JSD ($\\times10^{-4}$), Hamming and s/sequence are GC/CpG. The original $t\\ge0.5$ window, where the three uncertainty-based comparators are numerically indistinguishable from plug-in, is Table~\\ref{tab:m2-late}. Coverage alone is not a quality ranking, and the quality columns show what each arm spent to reach it.'+MARK,'tab:m2','lccccc',r'Method & GC IB$\uparrow$ & CpG IB$\uparrow$ & JSD$\downarrow$ & Hamming & Time',rows).replace(r'\centering\small',r'\centering\small\setlength{\tabcolsep}{3pt}')
    rows=[]
    for prop in ('gc','cpg'):
        if rows: rows.append(r'\midrule')
        for a in ('plug','tmpd','lgd_mc','tfg_mc','bdg_e4t0.5','bdg_e0t1'):
            rs=win(a,prop=prop); c=contrast(rs,win('plug',prop=prop),'in_band_fraction')
            name=r'BDG $\eta=0$' if a=='bdg_e0t1' else NAMES[a]
            rows.append(f'{PL[prop]} & '+name+' & '+pm(rs,'in_band_fraction')+f' & {c["delta_pp"]:+.2f} & {st.stdev(c["seed_deltas_pp"]):.2f} & {1e4*avg(rs,"kmer_js"):.2f} & {avg(rs,"diversity"):.4f} & {100*M.pooled(rs)["clip_frac"]:.2f} & {duration(rs,True):.3f}'+r'\\')
    rows=decorate_panels(rows,[2],'max',skip=(r'\eta=0',))
    rows=decorate_panels(rows,[5],'min',skip=(r'\eta=0',))
    rows=decorate_panels(rows,[8],'min',skip=(r'\eta=0',))
    out['M2-WINDOW']=latex_table('Window follow-up on both properties, $t\\ge0.3$, $w=4$, three seeds of 2,000. IB is percent mean $\\pm$ seed sd; differences and paired SD are percentage points against plug-in. JSD is $\\times10^{-4}$, Clip is percent of guided sample-steps. This exploratory follow-up was added after diagnosing the original late window; the sign of the BDG effect differs between the two properties. Coverage alone is not a quality ranking, and the quality columns show what each arm spent to reach it.'+MARK,'tab:m2-window','llccccccc',r'Prop. & Method & IB$\uparrow$ & $\Delta$IB & SD$(\Delta)$ & JSD$\downarrow$ & Hamming & Clip & s/seq.',rows)
    # The setpoint probe. BDG's sign rule says a setpoint BELOW the baseline's
    # achieved spread should contract and gain; on CpG plug-in already sits at
    # 0.37s, inside the registered tau_m=0.5, which is why the registered rung
    # widens. These cells test the rule at tau_m=0.25 and 0.15. They are outside
    # the pre-registered grid, so they are exploratory, and only tau_m=0.25
    # carries three seeds.
    rows=[]
    pl=win('plug',prop='cpg')
    for lab,rs,c in (('DPS-style plug-in',pl,None),
                     (r'BDG, $\tau_m=0.5$',win('bdg_e4t0.5',prop='cpg'),None),
                     (r'BDG, $\tau_m=0.25$',tau('bdg_e4t0.25'),None),
                     (r'BDG, $\tau_m=0.25$, $\eta=8$',tau('bdg_e8t0.25',seeds=1),None),
                     (r'BDG, $\tau_m=0.15$',tau('bdg_e4t0.15',seeds=1),None)):
        ib=pm(rs,'in_band_fraction') if len(rs)>2 else '$%.2f$'%(100*avg(rs,'in_band_fraction'))
        if len(rs)==len(pl) and [r['seed'] for r in rs]==[r['seed'] for r in pl]:
            c=contrast(rs,pl,'in_band_fraction')
            dd='%+.2f'%c['delta_pp']; sd='%.2f'%st.stdev(c['seed_deltas_pp'])
        else:
            dd='%+.2f'%(100*(avg(rs,'in_band_fraction')-avg(pl,'in_band_fraction'))); sd='--'
        if lab=='DPS-style plug-in': dd,sd='+0.00','0.00'
        rows.append(lab+' & %d & '%len(rs)+ib+f' & {dd} & {sd} & {avg(rs,"gc_sd")/avg(rs,"s"):.3f}'
                    +f' & {1e4*avg(rs,"kmer_js"):.2f} & {avg(rs,"decode_conf"):.3f}'
                    +f' & {100*M.pooled(rs)["clip_frac"]:.2f}'+r'\\')
    rows=decorate(rows,[2],'max',skip=(r'\eta=8',r'\tau_m=0.15'))
    out['SETPOINT']=latex_table('Exploratory setpoint probe on CpG, $t\\ge0.3$, $w=4$, $n=2{,}000$ per seed. BDG contracts only a batch wider than $\\tau$, and plug-in already reaches $0.372s$ here, inside the registered $\\tau_m=0.5$. Lowering the setpoint below the achieved spread restores contraction and coverage, which is the prediction the sign rule makes. Seeds gives the number of seeds; only $\\tau_m=0.25$ has the full three, so the one-seed rows are directional and their paired SD is omitted. These setpoints are outside the pre-registered grid and are reported as exploratory, not as headline results.'+MARK,'tab:setpoint','lcccccccc',r'Method & Seeds & IB$\uparrow$ & $\Delta$IB & SD$(\Delta)$ & $\sigma/s$ & JSD$\downarrow$ & Conf. & Clip',rows).replace(r'\centering\small',r'\centering\small\setlength{\tabcolsep}{4pt}')
    rows=[]
    for be,lab in (('fm','FM, ours'),('vp','VP diffusion, ours')):
        if rows: rows.append(r'\midrule')
        for p in PROPS:
            u,g=heads[be][(p,'unguided')],heads[be][(p,'plug')]
            c=contrast(g,u,KEY)
            rows.append(f'{lab} & '+PL[p]+' & '+pm(u,KEY)+' & '+pm(g,KEY)
                        +f' & {c["delta_pp"]:+.2f} & {100*avg(u,"mol_stability"):.1f} & '
                        +f'{100*avg(g,"mol_stability"):.1f} & {100*avg(u,"validity"):.1f} & '
                        +f'{100*avg(g,"validity"):.1f}'+r'\\')
    rows=decorate(rows,[4],'max')
    out['GUIDANCE']=latex_table('Guided versus unguided under matched settings, $w=1$, $t\\ge0.5$, three seeds of 2,000. Plug-in guidance is compared with the same frozen backbone and sampler. IB is decoded coverage (percent mean $\\pm$ seed sd); $\\Delta$ is percentage points. Stability and validity are reported so guidance is not judged by coverage alone. Bold marks the largest coverage gain and underline the runner-up.','tab:guidance','llccccccc',r'Backend & Prop. & IB un. & IB g. & $\Delta$IB & Stab. un. & Stab. g. & Val. un. & Val. g.',rows)
    # 9 columns overran the text block by 12.4pt at the default tabcolsep.
    out['GUIDANCE']=out['GUIDANCE'].replace(r'\centering\small',r'\centering\small\setlength{\tabcolsep}{4pt}')
    for be in ('fm','equifm','edm','vp'):
        rows=[]
        for p in PROPS:
            if rows: rows.append(r'\midrule')
            for a in V.BACKEND_ARMS[be]:
                rs=heads[be][(p,a)]
                rows.append(PL[p]+' & '+NAMES[a]+' & '+pm(rs,KEY)+' & '+pm(rs,'mol_stability')+' & '+f'{100*avg(rs,"validity"):.1f} & {100*avg(rs,"unique_valid_per_sample"):.1f} & {duration(rs):.3f}'+r'\\')
        rows=decorate_panels(rows,[2,3,4,5],'max')
        rows=decorate_panels(rows,[6],'min')     # s/sample
        be_label={'fm':'our FM','equifm':'borrowed EquiFM','edm':'borrowed EDMsecond','vp':'our VP diffusion'}[be]
        out['FULL-'+be.upper()]=latex_table(f'Complete headline evaluation on {be_label}. IB is decoded; IB and stability show percent mean $\\pm$ seed sd. Validity and distinct-valid yield (DV) are percentages; time is recorded cell wall-clock divided by attempts. $w=1$, $t\\ge0.5$, three seeds, $n=2{{,}}000$ each.'+MARK_PANEL,'tab:full-'+be,'llccccc',r'Property & Method & IB$\uparrow$ & Stable$\uparrow$ & Valid$\uparrow$ & DV$\uparrow$ & s/sample',rows)
    for be in ('fm','equifm'):
        for p in PROPS:
            rows=[]
            for w in (1,4):
                if rows: rows.append(r'\midrule')
                for a in V.ABL_ARMS:
                    rs=abl[be,w][p,a]; d=avg(rs,'delta')
                    e,t=a.removeprefix('bdg_e').split('t')
                    if e=='0': t='--'
                    rows.append(f'{w} & {e} & {t} & '+pm(rs,KEY)+f' & {avg(rs,"prop_mae_eval_dec")/d:.2f} & {bias(rs)/d:+.2f} & {spread(rs)/d:.2f} & '+pm(rs,'mol_stability')+r'\\')
            rows=decorate(rows,[3,7],'max'); rows=decorate(rows,[4],'min')
            rs=abl[be,1][p,'bdg_e0t1']
            unit={'mu':'D','alpha':r'Bohr$^3$','gap':'Ha'}[p]
            name='Our FM' if be=='fm' else 'Pretrained EquiFM (flow matching)'
            cap=f'{name}, {PL[p]}: all gain/setpoint ablations at $w=1,4$, q50 target $y={avg(rs,"target"):.4f}$ {unit}, half-width $\\delta={avg(rs,"delta"):.5f}$ {unit}. Each row uses three seeds of 2,000 samples. IB and molecular stability are percent mean $\\pm$ seed sd. MAE, signed bias of the pooled mean, and pooled standard deviation $\\sigma$ are decoded and normalized by $\\delta$. The $\\eta=0$ row removes the residual; its setpoint is irrelevant.'
            out[f'GRID-{be.upper()}-{p.upper()}']=latex_table(cap+MARK,f'tab:grid-{be}-{p}','cccccccc',r'$w$ & $\eta$ & $\tau_m$ & IB$\uparrow$ & MAE/$\delta\downarrow$ & Bias/$\delta$ & $\sigma/\delta$ & Stable$\uparrow$',rows)
    rows=[]
    for be in ('fm','equifm'):
        for p in PROPS:
            for a in ('bdg_e4t0.5','bdg_e4t1'):
                c=contrast(heads[be][(p,a)],heads[be][(p,'plug')])
                ds=c['seed_deltas_pp']
                rows.append(f'{be} & '+PL[p]+' & '+('0.5' if a.endswith('.5') else '1')+f' & {c["delta_pp"]:+.2f} & {st.stdev(ds):.2f} & {c["z"]:+.2f} & {c["mdd_pp"]:.2f}'+r'\\')
    rows=decorate(rows,[3],'max')
    out['CONTRASTS']=latex_table('BDG minus plug-in on decoded molecular IB. Differences, paired seed sd and the threshold-sized difference $2.99\\,\\mathrm{se}(\\Delta)$ are in percentage points. The registered unpaired statistic is shown separately from descriptive pairing.'+MARK+' The test statistic and its threshold are left unmarked: ranking significance is not a comparison this protocol makes.','tab:contrasts','llccccc',r'Backend & Property & $\tau_m$ & $\Delta$IB & SD$(\Delta)$ & $z$ & Threshold',rows)
    rows=[]
    for p in ('gc','cpg'):
        for w in (1,4):
            if rows: rows.append(r'\midrule')
            for a in ('unguided','plug','tmpd','lgd_mc','tfg_mc','bdg_e4t0.5','bdg_e4t1'):
                rs=seq(p,a,w); pooled=M.pooled(rs)
                rows.append(f'{PL[p]}, {w} & '+NAMES[a]+' & '+pm(rs,'in_band_fraction')+f' & {1e4*avg(rs,"kmer_js"):.2f} & {avg(rs,"decode_conf"):.3f} & {avg(rs,"diversity"):.4f} & {100*pooled["clip_frac"]:.2f} & {duration(rs,True):.3f}'+r'\\')
    rows=decorate_panels(rows,[2],'max'); rows=decorate_panels(rows,[3],'min')
    rows=decorate_panels(rows,[7],'min')
    out['FULL-M2']=latex_table('All DNA headline settings at the pre-registered window $t\\ge0.5$; the same grid at $t\\ge0.3$ is Table~\\ref{tab:full-m2-early}. IB is percent mean $\\pm$ seed sd; JSD is multiplied by $10^4$; Conf. is decoding confidence; Div. is normalized Hamming distance; Clip is percent of guided sample-steps. Unguided rows repeat a common reference and are never pooled twice.'+MARK_PANEL,'tab:full-m2','llcccccc',r'Prop., $w$ & Method & IB$\uparrow$ & JSD$\downarrow$ & Conf. & Div. & Clip & s/seq.',rows)
    # The same grid at the window the headline table uses, so every number in
    # tab:m2 and tab:m2-window can be traced to a cell. Both w rungs are here:
    # w=1 is the pre-registered strength, w=4 the ablation rung.
    rows=[]
    for prop in ('gc','cpg'):
        for w in (1.,4.):
            if rows: rows.append(r'\midrule')
            for a in ('unguided','plug','tmpd','lgd_mc','tfg_mc','bdg_e4t0.5','bdg_e4t1'):
                rs=win(a,prop=prop,w=w); pooled=M.pooled(rs)
                rows.append(f'{PL[prop]}, {w:g} & '+NAMES[a]+' & '+pm(rs,'in_band_fraction')+f' & {1e4*avg(rs,"kmer_js"):.2f} & {avg(rs,"decode_conf"):.3f} & {avg(rs,"diversity"):.4f} & {100*pooled["clip_frac"]:.2f} & {duration(rs,True):.3f}'+r'\\')
    rows=decorate_panels(rows,[2],'max'); rows=decorate_panels(rows,[3],'min')
    rows=decorate_panels(rows,[7],'min')
    out['FULL-M2-EARLY']=latex_table('All DNA headline settings at $t\\ge0.3$, the window the main-text DNA tables use. Columns match Table~\\ref{tab:full-m2}. IB is percent mean $\\pm$ seed sd; JSD is multiplied by $10^4$; Conf. is decoding confidence; Div. is normalized Hamming distance; Clip is percent of guided sample-steps. Unguided rows repeat a common reference and are never pooled twice.'+MARK_PANEL,'tab:full-m2-early','llcccccc',r'Prop., $w$ & Method & IB$\uparrow$ & JSD$\downarrow$ & Conf. & Div. & Clip & s/seq.',rows)
    rows=[]
    for a in ARMS:
        rs=heads['fm']['mu',a]; counts=rs[0]['cost']
        assert all(r['cost']==counts for r in rs)
        rows.append(NAMES[a]+' & '+' & '.join(str(counts[k]) for k in ('gen_fwd','gen_vjp','gen_jvp','guide_fwd','guide_bwd'))+r'\\')
    out['COST']=latex_table('Recorded sampling operations per 2,000-molecule cell, our FM, dipole, $w=1$. Forward calls, vector--Jacobian products (VJP) and Jacobian--vector products (JVP) are separate operations, not equal-cost units. Evaluator calls are outside these counts; Table~\\ref{tab:recent} reports wall-clock including scoring. No entry is marked here: these counts are exact and fixed by construction, plug-in and both BDG rungs are identical by design, and the smallest entry in every column belongs to the unguided reference, which performs no guidance at all.','tab:cost','lccccc',r'Method & Gen. forward & Gen. VJP & Gen. JVP & Guide forward & Guide backward',rows)
    rows=[]
    for be in ('fm','equifm'):
        for p in PROPS:
            vals=[exchange(be,p,e) for e in (4,8,None)]
            rows.append(('Our FM' if be=='fm' else 'EquiFM')+' & '+PL[p]+' & '+' & '.join(f'{v["ratio"]:.3f}' for v in vals)+r'\\')
    rows=decorate_across(rows,[2,3,4],'max')
    out['EXCHANGE']=latex_table('Exploratory coverage gained per stability point lost. All differences use the ablation-stage zero-gain reference at $w=1$. BDG changes $\\eta$ at $\\tau_m=0.5,w=1$; the strength control keeps $\\eta=0$ and increases $w$ to 4. Ratios use decoded IB and molecular stability means, not joint useful yield.'+MARK,'tab:exchange','llccc',r'Backend & Property & BDG $\eta=4$ & BDG $\eta=8$ & Plug-in $w=4$',rows)
    return out

def figures():
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#1965B0','#D55E00','#009E73']
    fig,axs=plt.subplots(1,2,figsize=(7.7,2.55),layout='constrained')
    ax=axs[0]
    for be,ls in [('fm','-'),('equifm','--')]:
        for p,col,mark in zip(PROPS,colors,['o','s','^']):
            u=spread(heads[be][(p,'unguided')])
            xs=[.5,.75,1,1.5]
            ys=[spread(abl[be,1][p,f'bdg_e8t{t:g}'])/u for t in xs]
            ax.plot(xs,ys,color=col,marker=mark,ls=ls,lw=1.2,markersize=4,mfc=col if be=='fm' else 'white')
    ax.axhline(1,color='.5',ls='--',lw=.8)
    ax.set(xlabel=r'Setpoint multiplier $\tau_m$ ($\eta=8$, $w=1$)',ylabel='Decoded spread / unguided',xticks=[.5,.75,1,1.5],ylim=(.55,1.48))
    ax.set_title('(a) QM9: setpoint changes spread',loc='left')
    handles=[Line2D([],[],color=c,marker=m,ls='none',label=PL[p]) for p,c,m in zip(PROPS,colors,['o','s','^'])]
    handles += [Line2D([],[],color='.2',ls=ls,label=name) for ls,name in [('-','Our FM'),('--','EquiFM')]]
    ax.legend(handles=handles,ncol=3,loc='upper left',frameon=False,handlelength=1.4,columnspacing=.8)
    ax=axs[1]
    # Both DNA properties at t>=0.3, the window Table 5 reports; the registered
    # t>=0.5 window stays in Table S18. Both setpoints are drawn on one axis, so
    # the widening tau_m=1 losses are shown at their full size, not clipped.
    # The two setpoints share each tick: their values are far enough apart
    # that no dodge is needed, and one column per condition reads directly.
    cells=[('gc',1),('gc',4),('cpg',1),('cpg',4)]
    lo,hi=0.,0.
    for a,col,off,mark in [('bdg_e4t0.5',colors[0],0.,'o'),('bdg_e4t1',colors[1],0.,'s')]:
        ds=[]; es=[]
        for p,w in cells:
            c=contrast(win(a,prop=p,w=w),win('plug',prop=p,w=w),'in_band_fraction')
            ds.append(c['delta_pp']); es.append(st.stdev(c['seed_deltas_pp']))
        lo=min(lo,min(d-e for d,e in zip(ds,es))); hi=max(hi,max(d+e for d,e in zip(ds,es)))
        ax.errorbar(np.arange(len(cells))+off,ds,yerr=es,ls='none',marker=mark,color=col,capsize=3,label=r'$\tau_m='+('0.5' if a.endswith('.5') else '1')+'$')
    ax.axhline(0,color='.5',ls='--',lw=.8)
    ax.set(xticks=range(len(cells)),xticklabels=[f'{PL[p]}\n$w={w}$' for p,w in cells],ylabel='IB difference vs plug-in (pp)',xlim=(-.5,len(cells)-.5),ylim=(lo-3,hi+3))
    ax.set_title(r'(b) DNA, $t\geq0.3$: sign follows setpoint',loc='left')
    ax.legend(loc='upper right',frameon=False,ncol=2)
    fig.savefig(PAPER/'figs/results.pdf'); fig.savefig(PAPER/'figs/results.png',dpi=220); plt.close(fig)
    for be in ('fm','equifm'):
        fig,axs=plt.subplots(2,3,figsize=(8,5.2),layout='constrained')
        for iw,w in enumerate((1,4)):
            for ip,p in enumerate(PROPS):
                rs=abl[be,w]; ref=rs[p,'bdg_e0t1']
                mat=np.array([[contrast(rs[p,f'bdg_e{e}t{t}'],ref)['delta_pp'] for t in ('0.5','0.75','1','1.5')] for e in (1,2,4,8)])
                ax=axs[iw,ip]; im=ax.imshow(mat,vmin=-8,vmax=8,cmap='RdBu',aspect='auto')
                for i in range(4):
                    for j in range(4): ax.text(j,i,f'{mat[i,j]:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(mat[i,j])>5 else 'black')
                ax.set(xticks=range(4),xticklabels=['.5','.75','1','1.5'],yticks=range(4),yticklabels=['1','2','4','8'],xlabel=r'$\tau_m$',ylabel=r'$\eta$',title=f'{PL[p]}, $w={w}$')
        fig.colorbar(im,ax=axs,label='Decoded IB minus zero-gain control (pp)',shrink=.85)
        fig.savefig(PAPER/f'figs/ablation_{be}.pdf'); fig.savefig(PAPER/f'figs/ablation_{be}.png',dpi=160); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check',action='store_true',help='Read-only verification of all generated table blocks and source hashes.')
    args=ap.parse_args()
    ts=tables()
    path=PAPER/'main.tex'; main=path.read_text(encoding='utf8')
    stale=[]
    for k,v in ts.items():
        pattern=r'% BEGIN DATA '+k+r'\n.*?% END DATA '+k
        found=re.search(pattern,main,re.S)
        wanted=f'% BEGIN DATA {k}\n{v}\n% END DATA {k}'
        if not found: raise SystemExit(f'Missing generated block: {k}')
        if found[0]!=wanted: stale.append(k)
        main=re.sub(pattern,lambda _:wanted,main,flags=re.S)
    source_paths=[p for be in ('fm','vp','equifm','edm') for stage in ('v3','v3abl') for p in (ROOT/f'results/v3/{be}/{stage}/n2000').rglob('*.json')]
    source_paths += [Path(r['_path']) for r in m2rows+winrows]
    # as_posix(), not str(): str() gives OS-dependent separators, so the manifest
    # was written with results\v3\... on Windows and results/v3/... elsewhere. The
    # hashes were identical either way, but all 1073 keys changed, and --check
    # compares them -- so regenerating on a different OS than the one that last
    # wrote the manifest failed the publication gate with "Source files changed"
    # when nothing had. The path is provenance, so it is recorded portably.
    sources={p.relative_to(ROOT).as_posix():cell_hash(p) for p in sorted(source_paths)}
    if args.check:
        saved=json.loads((PAPER/'results_manifest.json').read_text())
        assert not stale, f'Stale table blocks: {stale}'
        assert saved['sources']==sources, 'Source files changed; rebuild before publication.'
        molecular=[contrast(heads[be][p,a],heads[be][p,'plug']) for be in ('fm','equifm') for p in PROPS for a in ('bdg_e4t0.5','bdg_e4t1')]
        assert (sum(c['z']>=2.99 for c in molecular),sum(c['z']<=-2.99 for c in molecular))==(0,3)
        assert all(spread(abl[be,1][p,'bdg_e8t1.5'])>spread(heads[be][p,'unguided']) for be in ('fm','equifm') for p in PROPS)
        assert all(avg(heads[be][p,'bdg_e4t0.5'],'prop_mae_eval_dec')<avg(heads[be][p,'plug'],'prop_mae_eval_dec') for be in ('fm','equifm') for p in PROPS)
        assert all(avg(heads['fm']['mu','unguided'],k)>avg(heads['vp']['mu','unguided'],k) for k in ('mol_stability','validity'))
        diag=diagnostics()
        assert saved['additional_diagnostics']==diag
        assert all(all(x<y for x,y in zip(v,v[1:])) for v in diag['spread_ratios'].values())
        assert sum(d['8']['ratio']>d['None']['ratio'] for d in diag['exchange'].values())==5
        assert sum(d['4']['ratio']>d['None']['ratio'] for d in diag['exchange'].values())==4
        assert .915<min(diag['gain_coverage_spearman'].values())<.925
        assert .985<max(diag['gain_coverage_spearman'].values())<.995
        for be in ('fm','vp'):
            assert all((diag['guided_contrasts'][f'{be}/{p}']['z']>2.99)==(p!='alpha') for p in PROPS)
        for p in ('gc','cpg'):
            for w in (1,4):
                for a in ('tmpd','lgd_mc'):
                    assert all(x['in_band_fraction']==y['in_band_fraction'] for x,y in zip(seq(p,a,w),seq(p,'plug',w)))
        for p,w,expected,above in [('gc',1,.42,False),('gc',4,2.32,True),('cpg',1,4.82,True),('cpg',4,16.83,True)]:
            c=contrast(seq(p,'bdg_e4t0.5',w),seq(p,'plug',w),'in_band_fraction')
            assert abs(c['delta_pp']-expected)<.005
            assert (c['z']>3.16)==above
        assert abs(contrast(win('bdg_e4t0.5'),win('plug'),'in_band_fraction')['delta_pp']-7.083333)<1e-4
        assert avg(win('tmpd'),'in_band_fraction')!=avg(win('plug'),'in_band_fraction')
        print(f'PASS: {len(ts)} inline tables match all final cells; {len(sources)} source hashes unchanged.')
        print('PASS: molecular and DNA claims, own FM/VP guidance, six monotone spread curves, and exploratory exchange/correlation audits.')
        return
    path.write_text(main,encoding='utf8')
    (PAPER/'tmp').mkdir(exist_ok=True)
    (PAPER/'tmp/table_blocks.json').write_text(json.dumps(ts,indent=2),encoding='utf8')
    figures()
    audit={'sources':sources,'m1_contrasts':{f'{be}/{p}/{a}':contrast(heads[be][p,a],heads[be][p,'plug']) for be in ('fm','equifm') for p in PROPS for a in ('bdg_e4t0.5','bdg_e4t1')},'m2_contrasts':{f'{p}/{w}/{a}':contrast(seq(p,a,w),seq(p,'plug',w),'in_band_fraction') for p in ('gc','cpg') for w in (1,4) for a in ('bdg_e4t0.5','bdg_e4t1')}}
    audit['dna_identity']={f'{p}/{w}/{a}':{k:max(abs(x[k]-y[k]) for x,y in zip(seq(p,a,w),seq(p,'plug',w))) for k in ('in_band_fraction','gc_mean','gc_sd','kmer_js','diversity')} for p in ('gc','cpg') for w in (1,4) for a in ('tmpd','lgd_mc','tfg_mc')}
    audit['molecular_target_errors']={f'{be}/{p}/{a}':{'mae':avg(heads[be][p,a],'prop_mae_eval_dec'),'bias':bias(heads[be][p,a])} for be in ('fm','equifm') for p in PROPS for a in ('plug','bdg_e4t0.5')}
    audit['own_model_comparison']={be:{k:avg(heads[be]['mu','unguided'],k) for k in ('mol_stability','validity','uniqueness_of_valid','seconds')} for be in ('fm','vp')}
    audit['additional_diagnostics']=diagnostics()
    audit['m2_window_followup']={a:contrast(win(a),win('plug'),'in_band_fraction') for a in ('tmpd','lgd_mc','tfg_mc','bdg_e4t0.5','bdg_e0t1')}
    # rounded: the unrounded sums differ in the last float digit between runs, which
    # is a diff in a provenance file for no reason at all
    audit['m2_compute']={'new_files':len(winrows),'old_files':len(m2rows),'new_hours':round(sum(r['minutes'] for r in winrows)/60,6),'all_hours':round(sum(r['minutes'] for r in m2rows+winrows)/60,6)}
    (PAPER/'results_manifest.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    print(f'Rebuilt {len(ts)} table blocks and 3 vector figures; {len(sources)} source hashes.')

if __name__=='__main__': main()
