from pathlib import Path
import importlib.util,json,statistics as st,hashlib,shutil
R=Path(__file__).resolve().parents[2]; B=R/'slides/.build_v8'; V=R/'slides/deck_versions/pptx_v4_2026-09-29'
for d in ('source','output'): (V/d).mkdir(parents=True,exist_ok=True)
src=R/'slides/deck_versions/pptx_v3_2026-09-29/source/BDG_Defense_Group_2_v2.pptx'
shutil.copy2(src,V/'source'/src.name)
spec=importlib.util.spec_from_file_location('p',B/'prepare_data.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=m.data; D['dnaWindow']={a:m.pack(m.B.win(a),True) for a in ('plug','tmpd','lgd_mc','tfg_mc','bdg_e4t0.5','bdg_e0t1')}
D['windowContrasts']={a:m.B.contrast(m.B.win(a),m.B.win('plug'),'in_band_fraction') for a in D['dnaWindow']}
for c in D['windowContrasts'].values(): c['sd']=st.stdev(c['seed_deltas_pp'])
D['distributions']=json.loads((R/'paper/property_distributions_v9.json').read_text())['properties']
(B/'deck_data_v4.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf8')
s=(B/'build_v3.mjs').read_text(encoding='utf8')
s=s.replace('pptx_v3_2026-09-29','pptx_v4_2026-09-29').replace('deck_data.json','deck_data_v4.json').replace('candidate_v3_revised','candidate_v4_revised').replace('BDG_Defense_Group_2_v3_revised','BDG_Defense_Group_2_v4_revised').replace('paper v8','paper v9').replace('Paper v8','Paper v9').replace('(v8)','(v9)').replace('manifest_v8','manifest_v9').replace('Deck v3','Deck v4').replace("['A9','DNA at w = 1']","['A9','DNA windows and strengths']")
s=s.replace("Guide for t ≥ 0.5, w ∈ {1, 4}, η = 4 and τₘ ∈ {0.5, 1}. Decode by argmax at the end.","Original t ≥ 0.5, w ∈ {1, 4}. New GC follow-up: t ≥ 0.3, w = 4. η = 4, τₘ = 0.5; CpG has no new early-window run.")
start=s.index('function dnaTable(');end=s.index("s=fresh(25,",start)
s=s[:start]+'''function dnaTable(s,w,y=202,early=false){
 const rows=[['Method',early?'GC ≥.3 IB %':'GC IB % ↑',early?'CpG ≥.5 IB %':'CpG IB % ↑','JSD GC/CpG ↓','Hamming GC/CpG',early?'s/seq GC/CpG':'s/seq']];
 for(let i=0;i<arms.length;i++){
  const a=arms[i]==='tfg'?'tfg_mc':arms[i],cp=DNA('cpg',w,a),gc=early&&a!=='unguided'?D.dnaWindow[a]:DNA('gc',w,a);
  rows.push([a==='tfg_mc'?'TFG-MC [15]':names[i],gc?pm(gc):'—',pm(cp),`${gc?f(gc.jsd,2):'—'} / ${f(cp.jsd,2)}`,`${gc?f(gc.hamming,4):'—'} / ${f(cp.hamming,4)}`,early?`${gc?f(gc.time,3):'—'} / ${f(cp.time,3)}`:f(cp.time,3)]);
 }
 table(s,rows,85,y,1110,43,[260,148,148,174,220,160],16,[6,7]);
}
s=fresh(24,'DNA results: GC now guides earlier','3b','Part 3 · Comparison with recent methods','Original measured results · m2win: cfc5d64 (28 Sep 22:47) · CpG: original m2 cells');
tx(s,'w = 4. GC: t ≥ 0.3 follow-up. CpG: original t ≥ 0.5. Compare arms within each column.',85,157,1110,40,21);
dnaTable(s,4,212,true);
tx(s,'3 seeds × 2,000. IB: mean ± seed SD. JSD ×10⁻⁴. Hamming: first 256 sequences, including self-pairs. GC τₘ = 1 at t ≥ 0.3 was not run (—).',85,571,1110,57,18,C.muted);
tx(s,'BDG τₘ = 0.5: GC +7.08 pp; CpG +16.83 pp vs each window’s plug-in. GC baselines now separate.',85,641,1110,29,20,C.purple,true);

'''+s[end:]
start=s.index('const cats=');end=s.index("s=fresh(26,",start)
s=s[:start]+'''const pairs=[['GC .5',D.dnaContrasts['gc/4/bdg_e4t0.5']],['GC .3',D.windowContrasts['bdg_e4t0.5']],['CpG .5',D.dnaContrasts['cpg/4/bdg_e4t0.5']]];
const transfer=s.charts.add('bar',{position:{left:85,top:180,width:565,height:368},title:'BDG τₘ = 0.5 minus plug-in (pp)',titleTextStyle:{fontSize:21,bold:true},categories:pairs.map(x=>x[0]),series:[{name:'w = 4',values:pairs.map(x=>x[1].delta_pp),fill:C.blue}],hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:150},yAxis:{min:0,max:20,majorUnit:5,textStyle:{fontSize:17},majorGridlines:{fill:C.line,width:.7}},xAxis:{textStyle:{fontSize:18},tickLabelPosition:'low'},chartFill:C.bg,plotAreaFill:C.bg});applyPresentationChartFont(transfer,{fontFamily:'IBM Plex Sans'});chartOwners.push(s.id);
table(s,[['Property / t_min','ΔIB ± paired SD','Status'],...pairs.map(([pr,q])=>[pr,`${sign(q.delta_pp)} ± ${f(q.sd,2)}`,pr==='GC .3'?'Follow-up':'Original'])],690,190,505,52,[150,220,135],18);
tx(s,'Earlier GC guidance increases BDG’s mean gain, with greater seed variation. The GC follow-up was chosen after diagnosing the late-window regime.',690,433,505,143,21);
tx(s,'QM9: +0.45 to +1.52 pp, unresolved by the registered test. DNA: clear mean gains, with property, band and window differences retained in interpretation.',85,602,1110,69,21,C.purple,true);

'''+s[end:]
s=s.replace('At w = 1, starting GC guidance at t = 0 instead of 0.5 raises plug-in’s decoded coverage by 22.82 points. In that early window, BDG τₘ = 0.5 is 2.33 points below plug-in.','At t ≥ 0.3, GC baselines separate and BDG gains +7.08 pp. At t ≥ 0, the earlier w = 1 diagnostic put BDG 2.33 pp below plug-in. Timing changes the result.')
s=s.replace('Winning DNA cells clip 7.4–9.2% of guided sample-steps versus ≤0.21% for plug-in. A matched-force experiment is needed to separate added force from feedback.','GC at t ≥ 0.3: BDG clips 20.8% of guided sample-steps versus 14.9% for plug-in. CpG at t ≥ 0.5: 7.4% versus 0.21%. Match applied force to isolate feedback.')
s=s.replace('The same rule improves CpG coverage by +4.82 and +16.83 points at w = 1 and 4. Hamming diversity stays nearly constant, while 3-mer JSD increases.','GC at t ≥ 0.3 gains +7.08 pp; CpG at t ≥ 0.5 gains +16.83 pp (w = 4). Hamming changes little. GC JSD improves over unguided, while CpG JSD rises.')
s=s.replace('Next: matched-force and replay controls, batch-aware uncertainty, joint chemical yield, and transfer to additional simplex generators.','Next: address confounding with tighter atom-count control and target-range tests that account for the local property distribution. Add matched-force controls and joint useful yield.')
# The native distribution charts and extra evidence are added after the original
# order is constructed, then moved to their appropriate rubric sections.
insert="""
function histChart(s,pr,x,y,w,h){
 const d=D.distributions[pr],xs=[],ys=[];
 for(let i=0;i<d.counts.length;i++){xs.push(d.edges[i],d.edges[i+1]);const density=d.counts[i]/d.n/(d.edges[i+1]-d.edges[i]);ys.push(density,density);}
 const ymax=Math.max(...ys)*1.08;
 const colors=[C.blue,C.purple,C.orange,C.orange];
 const series=[{name:'Training histogram',xValues:xs,values:ys},...[[d.target,'q50'],[d.low,'Lower band edge'],[d.high,'Upper band edge']].map(([v,name])=>({name,xValues:[v,v],values:[0,ymax]}))].map((q,i)=>({...q,line:{fill:colors[i],width:i===0?1.8:2,style:i>1?'dashed':'solid'},marker:{symbol:'none'}}));
 const titles={mu:'Dipole μ (D)',alpha:'Polarizability α (Bohr³)',gap:'HOMO–LUMO gap (Ha)',gc:'GC fraction',cpg:'CpG fraction'};
 const ch=s.charts.add('scatter',{position:{left:x,top:y,width:w,height:h},title:titles[pr],titleTextStyle:{fontSize:22,bold:true},series,scatterOptions:{style:'line'},hasLegend:false,xAxis:{min:{mu:0,alpha:0,gap:0,gc:.2,cpg:0}[pr],max:{mu:30,alpha:150,gap:.5,gc:.8,cpg:.15}[pr],majorUnit:{mu:10,alpha:50,gap:.1,gc:.2,cpg:.05}[pr],textStyle:{fontSize:15},numberFormatCode:pr==='mu'||pr==='alpha'?'0':'0.00'},yAxis:{min:0,max:ymax,title:'Density',textStyle:{fontSize:14},numberFormatCode:pr==='mu'||pr==='alpha'?'0.00':'0',majorGridlines:{fill:C.line,width:.5}},chartFill:C.bg,plotAreaFill:C.bg});applyPresentationChartFont(ch,{fontFamily:'IBM Plex Sans'});chartOwners.push(s.id);
 tx(s,`q50 = ${Number(d.target.toPrecision(5))}   ± δ = ${Number(d.delta.toPrecision(5))}`,x,y+h+8,w,35,19,C.purple,true);
}
s=append('QM9 property distributions and target bands','1e','Part 1 · What the target band measures','Measured QM9 train_a: 51,527 molecules · fixed targets and calibration widths from audited cells');
tx(s,'Blue: empirical training density     Purple: q50     Orange dashed: y − δ and y + δ',85,160,1110,40,21);
for(let i=0;i<3;i++)histChart(s,props[i],85+i*375,218,360,295);
tx(s,'Band half-width: δ = 2 × MAE_cal(f_B) = (2 / n_cal) Σⱼ |f_B(xⱼ) − yⱼ|',85,584,1110,39,25,C.blue,true);
tx(s,'Our predictor pair; f_B is scored on held-out training-half examples. Full bandwidth = 2δ. BDG varies τ, while δ stays fixed.',85,632,1110,44,19);
s.moveTo(9);
s=append('DNA property distributions and target bands','3a','Part 3 · Targets on the DNA simplex','Measured DeepFlyBrain training distribution: 83,722 sequences · exact GC and CpG counts');
tx(s,'Blue: empirical density     Purple: q50     Orange dashed: band edges',85,158,1110,38,21);
histChart(s,'gc',75,220,390,308);histChart(s,'cpg',475,220,390,308);
tx(s,'Band half-width',907,221,285,36,24,C.ink,true);
tx(s,'δ = max(0.16s, 4.4q)',907,277,285,70,24,C.blue,true);
tx(s,'s = training property SD\\nqGC = 1/500\\nqCpG = 1/499\\nFull bandwidth = 2δ',907,362,285,154,21);
tx(s,'CpG’s discrete-count floor binds: δ/s = 0.598, versus 0.160 for GC. Different natural band occupancy helps explain why raw coverage is not comparable across properties.',85,604,1110,67,22,C.purple,true);
s.moveTo(24);
s=append('Why the DNA guidance window changed','A9','Appendix A9 · New GC window follow-up','cfc5d64 / f568f68, 28 Sep 22:47 · results/m2/m2win/n2000 · earlier window diagnosed before follow-up');
tx(s,'GC, w = 4. Original t ≥ 0.5 versus new t ≥ 0.3; 3 seeds × 2,000 per setting.',85,158,1110,40,21);
const winTable=[['Method','IB % at t ≥ .5','IB % at t ≥ .3','JSD ×10⁻⁴ (.3)','Hamming (.3)','Clip % (.3)']];
for(const a of ['plug','tmpd','lgd_mc','tfg_mc','bdg_e4t0.5','bdg_e0t1']){const q=D.dnaWindow[a],late=a==='bdg_e0t1'?DNA('gc',4,'plug'):DNA('gc',4,a);winTable.push([a==='bdg_e0t1'?'BDG η = 0':a==='bdg_e4t0.5'?'BDG τₘ = 0.5':a,pm(late),pm(q),f(q.jsd,2),f(q.hamming,4),f(q.clip,1)]);}
table(s,winTable,85,215,1110,45,[250,170,170,185,170,165],18,[5,6]);
tx(s,'At t ≥ 0.3, all four baseline means differ. BDG gains +7.08 ± '+f(D.windowContrasts['bdg_e4t0.5'].sd,2)+' pp (paired seed SD). η = 0 matches IB; continuous rerun checks use 10⁻⁵ tolerance.',85,555,1110,71,21,C.purple,true);
tx(s,'Only GC received this three-seed follow-up. Four t ≥ 0.2 probes have one seed; no CpG ≥0.3 or GC τₘ = 1 cell is inferred.',85,643,1110,32,18,C.muted);
s=append('DNA: original matched-window comparison','A9','Appendix A9 · Retained original w = 4 comparison','Original m2 cells · both GC and CpG use t ≥ 0.5; separate from m2win follow-up');
tx(s,'w = 4, both properties t ≥ 0.5. Compare with the window-specific main table.',85,156,1110,39,22);dnaTable(s,4,210);
tx(s,'Plug-in / TMPD / LGD-MC coincide numerically here, on both properties. TFG-MC differs. Earlier GC guidance removes this overlap.',85,590,1110,66,22,C.purple,true);
// Refresh all page footers after inserting the two teaching figures.
for(let i=0;i<p.slides.items.length;i++)for(const q of p.slides.items[i].shapes.items){const v=q.text.toString();if(v.startsWith('CIS 6270 · Group 2 ·'))q.text=`CIS 6270 · Group 2 · ${i<31?`${i+1} / 31`:`Appendix ${i-31}`}`;}

"""
s=s.replace('// Audit exact source data',insert+'// Audit exact source data')
s=s.replace("path.join(BUILD,'draft')","path.join(BUILD,'draft_v4')").replace("path.join(BUILD,'draft',","path.join(BUILD,'draft_v4',").replace("'authored_inspect.ndjson'","'authored_inspect_v4.ndjson'")
(B/'build_v4.mjs').write_text(s,encoding='utf8')
print('Prepared v4 with empirical distributions and separate DNA windows')
