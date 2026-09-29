import importlib.util, json, pathlib, statistics as st, shutil, hashlib
R=pathlib.Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('paper_results',R/'paper/tools/build_results.py')
B=importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
def pack(rs, dna=False):
    key='in_band_fraction' if dna else B.KEY
    d={'ib':100*B.avg(rs,key),'ib_sd':100*B.sd(rs,key),'time':B.duration(rs,dna)}
    if dna: d.update(jsd=1e4*B.avg(rs,'kmer_js'),hamming=B.avg(rs,'diversity'),confidence=B.avg(rs,'decode_conf'),clip=100*B.M.pooled(rs)['clip_frac'])
    else:
        d.update(stable=100*B.avg(rs,'mol_stability'),stable_sd=100*B.sd(rs,'mol_stability'),valid=100*B.avg(rs,'validity'),valid_sd=100*B.sd(rs,'validity'),unique=100*B.avg(rs,'uniqueness_of_valid'),dv=100*B.avg(rs,'unique_valid_per_sample'),mae=B.avg(rs,'prop_mae_eval_dec'),mean=B.avg(rs,'f_B_dec_mean'),spread=B.spread(rs),target=B.avg(rs,'target'),delta=B.avg(rs,'delta'))
    return d
data={'main':{},'abl':{},'dna':{},'contrasts':{},'dnaContrasts':{},'diagnostics':B.diagnostics(),'cost':{}}
for be,cells in B.heads.items():
    for (prop,a),rs in cells.items(): data['main'][f'{be}/{prop}/{a}']=pack(rs)
for (be,w),cells in B.abl.items():
    for (prop,a),rs in cells.items(): data['abl'][f'{be}/{prop}/{w}/{a}']=pack(rs)
for prop in ('gc','cpg'):
    for w in (1,4):
        for a in B.M.HEADLINE_ARMS: data['dna'][f'{prop}/{w}/{a}']=pack(B.seq(prop,a,w),True)
        for a in ('bdg_e4t0.5','bdg_e4t1'):
            c=B.contrast(B.seq(prop,a,w),B.seq(prop,'plug',w),'in_band_fraction');c['sd']=st.stdev(c['seed_deltas_pp']); data['dnaContrasts'][f'{prop}/{w}/{a}']=c
for be in ('fm','equifm'):
    for prop in B.PROPS:
        for a in ('bdg_e4t0.5','bdg_e4t1'): data['contrasts'][f'{be}/{prop}/{a}']=B.contrast(B.heads[be][prop,a],B.heads[be][prop,'plug'])
for a in B.ARMS: data['cost'][a]=B.heads['fm']['mu',a][0]['cost']
source=pathlib.Path('C:/Users/mooooonesy/Downloads/BDG Defense — Group 2 v2.pptx')
version=R/'slides/deck_versions/pptx_v3_2026-09-29'
(version/'source').mkdir(parents=True,exist_ok=True)
(version/'output').mkdir(exist_ok=True)
shutil.copy2(source,version/'source/BDG_Defense_Group_2_v2.pptx')
(R/'slides/.build_v8/deck_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
print('Audited data:',len(data['main']),len(data['abl']),len(data['dna']),'groups; v2 preserved',hashlib.sha256(source.read_bytes()).hexdigest())
