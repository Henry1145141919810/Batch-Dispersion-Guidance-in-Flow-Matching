import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const ROOT='C:/Users/mooooonesy/Downloads/pennstuff/cis 6270/Project 1';
const BUILD=path.join(ROOT,'slides/.build_v8');
const SKILL='C:/Users/mooooonesy/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const SRC=path.join(ROOT,'slides/deck_versions/pptx_v4_2026-09-29/source/BDG_Defense_Group_2_v2.pptx');
const OUT=path.join(ROOT,'slides/deck_versions/pptx_v4_2026-09-29/output');
process.env.RUNTIME_NODE_MODULES='C:/Users/mooooonesy/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const D=JSON.parse(await fs.readFile(path.join(BUILD,'deck_data_v4.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(SRC));
const old=[...p.slides.items];
const C={bg:'#F7F6F2',ink:'#16202C',body:'#3A4553',muted:'#5F6A77',blue:'#215FA9',purple:'#5D3CA6',lilac:'#F0EBF9',header:'#EBE9E1',line:'#DAD7CF',white:'#FFFFFF',green:'#27745E',orange:'#AC600D'};
const fontPolicy={basis:'reference',families:['IBM Plex Sans','Source Serif 4'],referencePath:SRC,referenceSha256:'fe365871823b5f765f9acd857705e44e1ffc509979070b7a05b4b218ff3a9dfb'};
const tableOwners=[],chartOwners=[];
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
function tx(s,text,x,y,w,h,size=20,color=C.body,bold=false,extra={}){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=text;q.text.style={typeface:'IBM Plex Sans',fontSize:size,color,bold,autoFit:'none',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0},...extra};return q;
}
function strip(s,text,y=585,color=C.purple){return tx(s,text,85,y,1110,65,22,color,true);}
function topic(s,label,text,x,y,w=520,h=115){tx(s,label,x,y,w,34,23,C.ink,true);tx(s,text,x,y+40,w,h,21);}
function cite(s,text){tx(s,text,415,682,780,20,10.5,C.muted,false,{alignment:'right'});s.speakerNotes.textFrame.setText('Sources: '+text+'\nLocal evidence: paper/main.tex (v9), paper/results_manifest_v9.json; original implementation in proj1/.');}
function footer(s,num){tx(s,`CIS 6270 · Group 2 · ${num<=29?num+' / 29':'Appendix '+(num-30)}`,85,682,325,20,12,C.muted);}
function chrome(s,title,tag,kicker,num,source='Original experiment and analysis, paper v9 (this work)'){
 s.background.fill=C.bg;
 tx(s,kicker.toUpperCase(),85,55,965,26,16,C.blue,true);
 tx(s,title,85,87,1110,55,37,C.ink,true,{typeface:'Source Serif 4'});
 if(tag){const badge=s.shapes.add({geometry:'roundRect',position:{left:1077,top:35,width:117,height:48},fill:tag[0]==='A'?'#495B72':C.blue,line:{fill:'none'},borderRadius:9});badge.text=tag;badge.text.style={typeface:'IBM Plex Sans',fontSize:31,color:'#FFFFFF',bold:true,alignment:'center',verticalAlignment:'middle',insets:{left:0,right:0,top:0,bottom:0}};}
 footer(s,num);cite(s,source);
}
function fresh(n,title,tag,kicker,source){const s=p.slides.add();s.setLayout(p.layouts.items[0]);old[n-1].delete();s.moveTo(n-1);chrome(s,title,tag,kicker,n,source);return s;}
function append(title,tag,kicker,source){const s=p.slides.add();s.setLayout(p.layouts.items[0]);chrome(s,title,tag,kicker,p.slides.items.length,source);return s;}
function table(s,values,x=85,y=180,width=1110,rowH=45,widths=null,fsz=18,highlight=[]){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height:rowH*values.length,values,columnWidths:widths||Array(values[0].length).fill(width/values[0].length)});
 t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({fill:C.line,width:.7,style:'solid'});
 t.cells.block({row:0,column:0,rowCount:values.length,columnCount:values[0].length}).assign({fill:C.bg,textStyle:{typeface:'IBM Plex Sans',fontSize:fsz,color:C.body},margins:{left:9,right:9,top:4,bottom:4},anchor:'center'});
 for(let r=0;r<values.length;r++){
  t.rows[r].height=rowH;
  for(let c=0;c<values[0].length;c++){let q=t.getCell(r,c);q.fill=r===0?C.header:highlight.includes(r)?C.lilac:C.bg;q.text.style={typeface:'IBM Plex Sans',fontSize:fsz,color:r===0?C.ink:C.body,bold:r===0,alignment:c===0?'left':'center',verticalAlignment:'middle',autoFit:'none'};}
 }
 tableOwners.push(s.id);return t;
}
function replace(s,find,txt){let found=false;for(const q of s.shapes.items){const v=q.text.toString();if(v.includes(find)){q.text=txt;found=true;}}if(!found)throw Error('Missing '+find);}
function clearShapes(s){s.shapes.deleteAll();for(const im of [...s.images.items])im.delete();}
const pm=(d,key='ib',digits=1)=>`${d[key].toFixed(digits)} ± ${d[key+'_sd'].toFixed(digits)}`;
const f=(v,n=1)=>Number(v).toFixed(n),sign=(v,n=2)=>(v>=0?'+':'')+f(v,n);
const M=(be,prop,a)=>D.main[`${be}/${prop}/${a}`];
const A=(be,prop,w,a)=>D.abl[`${be}/${prop}/${w}/${a}`];
const DNA=(prop,w,a)=>D.dna[`${prop}/${w}/${a}`];
const arms=['unguided','plug','tmpd','lgd_mc','tfg','bdg_e4t0.5','bdg_e4t1'];
const names=['Unguided','DPS-style plug-in [12]','TMPD-inspired [22]','LGD-MC [23]','TFG [15]','BDG, τₘ = 0.5','BDG, τₘ = 1'];
const props=['mu','alpha','gap'],pl={mu:'μ',alpha:'α',gap:'gap'};

// Cover: keep the supplied title, names, design and defense date.
replace(old[0],'A feedback-servoed','Batch-variance feedback for property-band control\nQM9 3D molecules and DeepFlyBrain DNA');
tx(old[0],'Deck v4 · paper v9',85,551,800,24,17,'#8FB4E5');

let s=fresh(2,'Property-band generation','1a','Part 1 · Set up the study','[1] Ramakrishnan et al. 2014 (QM9) · [4] Stark et al. 2024 (DeepFlyBrain)');
topic(s,'The task','Guide a frozen generator so the decoded property falls inside a target band, y ± δ. Coverage is the fraction of samples in that band.',85,160,700,100);
topic(s,'The question','Can a requested batch spread improve band coverage while retaining sample quality?',85,338,700,110);
tx(s,'QM9 molecules',845,170,350,32,24,C.blue,true);tx(s,'Up to 29 atoms\n3D coordinates + 5 type channels\nEuclidean symmetry',845,218,350,120,21);
tx(s,'DeepFlyBrain DNA',845,375,350,32,24,C.blue,true);tx(s,'500 positions × 4 bases\nOne probability simplex per position',845,423,350,95,21);
strip(s,'Hypothesis: a batch-dispersion setpoint adds useful control beyond guidance strength.',575);

s=fresh(3,'Datasets and independent evaluation','1b','Part 1 · Data and representations','[1] Ramakrishnan et al. 2014 · [2] Hoogeboom et al. 2022 · [4] Stark et al. 2024');
tx(s,'QM9: 133,885 molecules',85,154,530,32,24,C.blue,true);
table(s,[['Split','Role','n'],['train_a','Generator + guide f_A','51,527'],['train_b','Evaluator f_B','51,527'],['Validation','Selection + size masks','17,748'],['Test','Held out','13,083']],85,202,535,44,[114,318,103],18);
tx(s,'Explicit H. Centred Å coordinates, padded to 29. Keep EDM’s 3,054 excluded structures. Split seed 20260917. Targets: dipole magnitude μ, polarizability α and HOMO–LUMO gap.',85,440,535,120,20);
tx(s,'DeepFlyBrain: 500-bp enhancers',660,154,535,32,24,C.blue,true);
table(s,[['Split','Role','n'],['Train','Generator','83,722'],['Validation','Model selection','10,505'],['Test','Held out','10,434']],660,202,535,44,[120,315,100],18);
tx(s,'Remove four ambiguous training sequences. Use A/C/G/T one-hot encoding. The generator receives no cell-type labels.',660,400,535,100,20);
tx(s,'f_A and f_B train on disjoint halves. DNA uses exact decoded counts, with a separate soft guidance score.',660,522,535,90,20);
tx(s,'QM9 normalization uses train_a. Neither random split establishes generalization to unseen molecular or sequence families.',85,609,1110,45,18,C.muted);

s=fresh(4,'Representation and decoded metrics','1b','Part 1 · Data and representations','[2] Hoogeboom et al. 2022 (stability) · [4] Stark et al. 2024 (DNA) · [9] Satorras et al. 2021 (EGNN)');
table(s,[['','QM9 molecules','DeepFlyBrain DNA'],['Sampling state','Real coordinates + 5 continuous type channels','500 × 4 probabilities, each row sums to 1'],['Constraints','Mask padding and remove centre of mass','Clamp to ≥ 0 and renormalize each position'],['Conditioning','Atom mask only. Property enters through guidance.','No class/property input to the generator.'],['Decoding','Type argmax, then distance-based bonds','Base argmax at each position']],85,157,1110,54,[150,480,480],18);
topic(s,'QM9 metrics','In-band (IB): |f_B(decoded x) − y| ≤ δ.\nStability: all atom valences satisfied.\nValidity: RDKit sanitization succeeds.\nUniqueness: distinct / valid molecules.\nDV: distinct valid molecules / attempts.',85,454,535,162);
topic(s,'DNA metrics','GC: fraction of C/G bases. CpG: C–G pair fraction.\nIB: exact decoded property inside the band.\n3-mer JSD: divergence from training composition.\nHamming: mean fraction of differing bases.\nSoft and decoded property scores can disagree.',660,454,535,162);

s=fresh(5,'Flow matching: state, velocity and loss','1c','Part 1 · Flow matching 1 of 2','[6] Lipman et al. 2022 · [7] Liu et al. 2022 · [8] Albergo et al. 2025 · implementation: train_fm.py');
tx(s,'xₜ = (1 − t)x₀ + tx₁        v* = x₁ − x₀        t ∼ Uniform(0, 1)',85,153,1110,43,27,C.blue,true);
tx(s,'At t = 0: masked Gaussian noise in both arrays (coordinates centred).\nAt t = 1: real positions + one-hot types, e.g. carbon [0, 1, 0, 0, 0] in H/C/N/O/F order.',85,207,1110,72,21);
tx(s,'Types stay continuous during training and sampling, so each type channel can be noised and denoised.',85,290,1110,45,21);
table(s,[['Per-molecule tensor','Coordinates c','Type channels f'],['Velocity v = (vᶜ, vᶠ)','How fast each position changes','How fast each type number changes'],['Error z = vθ − v*','zᶜ: 29 × 3','zᶠ: 29 × 5']],85,348,1110,50,[320,395,395],19);
tx(s,'LFM = E [ Σ M(zᶜ)² / (3ΣM)  +  Σ M(zᶠ)² / (5ΣM) ]',85,519,1110,44,27,C.purple,true);
tx(s,'The full error is 29 × 8. M masks padded atoms. Average each part separately, then add with equal weight.',85,568,1110,38,20);
tx(s,'Endpoint estimate for guidance: m = xₜ + (1 − t)vθ(xₜ, t). The generator has no property input.',85,617,1110,37,20);

s=fresh(6,'Flow matching: training and sampling','1c','Part 1 · Flow matching 2 of 2','[6] Lipman et al. 2022 (FM) · [9] Satorras et al. 2021 (EGNN) · recorded training/checkpoint metadata');
topic(s,'Generator architecture','Our EGNNVelocity, trained from scratch.\n256 hidden channels × 8 layers, SiLU.\n3,753,229 parameters, no attention.\nTime joins the input type features.',85,169,535,155);
topic(s,'Training','train_a, batch 256, Adam 2 × 10⁻⁴.\nCosine decay to 5%, gradient clip 5.\nEMA 0.9999, seed 20260918.\n1,500 epochs, 303k updates.',660,169,535,155);
topic(s,'Checkpoint used in this study','Evaluate the final-epoch EMA: epoch 1,500.\nThe trainer also tracks validation atom stability every 25 epochs on 512 samples, with validation loss as the tie-break.',85,394,535,162);
topic(s,'Sampling','100 uniform Euler steps, noise to data.\nOne generator evaluation per step: NFE 100.\nAfter each step: apply the atom mask and remove centre of mass. Decode once at the end.',660,394,535,162);

// Preserve the supplied, accurate VP equation panel.
replace(old[6],'Checked:','Coordinate and type errors use the same separately normalized masked loss as FM (1c).');
replace(old[6],'Integrated from u = 1','100 Euler steps from u = 1 to 0.001, then a final denoising evaluation: NFE 101.');

s=fresh(8,'VP diffusion: training and sampling','1d','Part 1 · VP diffusion 2 of 2','[9] Satorras et al. 2021 · [10] Song et al. 2021 · checkpoint and 18 benchmark cells measured in this work');
topic(s,'Same architecture, different target','Use the same EGNNVelocity class and 3.75M parameters. For VP, interpret its coordinate and type outputs as predicted noise ε, and train against noise.',85,163,535,146);
topic(s,'Matched configured recipe','Same train_a, optimizer, batch size, EMA, clipping, seed and 1,500-epoch budget as FM. A common budget does not imply identical selected checkpoints.',660,163,535,146);
topic(s,'Selected checkpoint','Validation atom stability selects epoch 1,475, with loss as the tie-break. Evaluate its EMA weights (MD5 prefix 8a3390a6). FM uses its final-epoch EMA.',85,386,535,144);
topic(s,'Sampling and completed benchmark','100 reverse probability-flow ODE steps + a final denoising evaluation: NFE 101.\n18 cells: 2 arms × 3 properties × 3 seeds, 2,000 molecules each.',660,386,535,144);
strip(s,'VP results are complete. VP used B200 MIG hardware, so cross-family runtimes are not matched.',603,C.blue);

s=fresh(9,'Evaluation protocol and fixed molecule sizes','1e','Part 1 · Comparison protocol 1 of 2','Registered QM9 protocol · transfer_sweep.py:2382–2426 · verified data/qm9.pt validation masks');
table(s,[['Protocol','QM9 headline comparison'],['Target and band','Training median q50. δ = 2 × evaluator calibration MAE.'],['Band half-widths','μ: 0.17541 D     α: 0.5062 Bohr³     gap: 0.00748 Ha'],['Sampling','3 seeds × 2,000 samples. Four batches of 500 per seed.'],['Guidance','100 Euler steps. Guide in the second half, headline w = 1.'],['Uncertainty','Mean ± seed SD. Registered pooled-binomial |z| ≥ 2.99 (18 contrasts).']],85,152,1110,47,[270,840],19);
tx(s,'The same 2,000 atom counts in every arm, seed and backend',85,456,1110,34,23,C.purple,true);
tx(s,'Molecule i uses validation mask i. Seeds change initial noise only. No histogram draw, target-dependent sizing or size filtering.',85,504,1110,62,21);
tx(s,'Size distribution: min 8   median 18   mean 17.87   max 29 atoms. Every size receives the same q50 target.',85,580,1110,65,22,C.ink,true);

s=fresh(10,'Matched factors and disclosed differences','1e','Part 1 · Comparison protocol 2 of 2','Original protocol and checkpoint metadata (this work) · [9] Satorras et al. 2021');
table(s,[['Factor','Held fixed for our FM and VP','Difference to retain in interpretation'],['Data + architecture','train_a, masks, EGNN 256 × 8, 3.75M','Linear path / velocity vs VP process / noise'],['Training','Optimizer, batch, EMA, clip, seed, budget','FM final EMA 1500 / VP selected EMA 1475'],['Sampling','100 ODE steps, same initial seeds + sizes','VP adds final denoising: NFE 101 vs 100'],['Evaluation','Same f_A, f_B, targets, bands and decoding','FM: RTX A6000 / VP: B200 MIG'],['Borrowed backends','Same evaluation procedure','Different training, scales and predictor pairs']],85,162,1110,60,[185,452,473],18);
topic(s,'What coverage measures','It includes invalid molecules and inherits f_B’s prediction error. Report stability and validity beside coverage.',85,550,535,85);
topic(s,'What the uncertainty measures','Seed SD describes run variation. Batch feedback couples samples, so pooled binomial tests have an independence limitation.',660,550,535,85);

replace(old[10],'Exact GC or CpG steers;','Soft GC or CpG steers; each step projected back (3a)');
replace(old[10],'Everything inside the boxes','All main-path models are trained from scratch. We evaluate borrowed EquiFM and EDM checkpoints separately, and never pool backends.');

s=fresh(12,'Flow matching is the base model we carry forward','2a','Part 2 · Base-model comparison','[2] Hoogeboom et al. 2022 · [24] Song et al. 2023 (EquiFM) · [15] Ye et al. 2024 (EDMsecond release)');
const base=[['Generator','Mol. stable %','Valid %','Unique %','NFE','s/sample']];
for(const [be,label,nfe] of [['fm','Our FM',100],['vp','Our VP diffusion',101],['equifm','EquiFM [24] · flow',100],['edm','EDMsecond [15] · diffusion',101]]){const q=M(be,'mu','unguided');base.push([label,pm(q,'stable'),pm(q,'valid'),f(q.unique),nfe,f(q.time,3)+(be==='vp'?'*':'')]);}
table(s,base,85,176,1110,56,[315,205,205,140,90,155],20,[1]);
tx(s,'Unguided. 3 seeds × 2,000. Mean ± seed SD. Uniqueness is conditional on validity.\n*VP: B200 MIG. Others: RTX A6000. Runtime includes scoring and is not hardware matched.',85,466,1110,67,18,C.muted);
tx(s,'Our FM gains 10.9 stability points and 9.4 validity points over our VP, with one fewer evaluation. This motivates carrying FM forward.',85,552,1110,65,25,C.purple,true);
tx(s,'EquiFM is also flow matching, and EDMsecond is diffusion. Their measured ordering is consistent with this choice, but external training differs.',85,624,1110,43,18);

s=fresh(13,'Plug-in guidance through the frozen generator','2b','Part 2 · Guidance strategy 1 of 2','[12] Chung et al. 2023 (DPS-style adaptation) · original implemented Gaussian-energy guidance');
tx(s,'mᵢ = xₜ,ᵢ + (1 − t)vθ(xₜ,ᵢ, t)       Fᵢ = f_A(mᵢ)       Jᵢ = ∂mᵢ/∂xₜ,ᵢ',85,162,1110,44,27,C.blue,true);
tx(s,'Gᵢ = [(y − Fᵢ) / s²] Jᵢᵀ ∇f_A(mᵢ)',85,237,1110,53,33,C.purple,true);
tx(s,'The guide scores a predicted endpoint. Differentiation passes through both the property network and the frozen generator.',85,310,1110,65,22);
topic(s,'Target, scale and schedule','y is q50. s is the calibration property SD.\nFor FM, add w(1 − t)/t times G for t ≥ 0.5.\nClip each correction to the base velocity norm.\nVP uses its Tweedie endpoint and VP mapping.',85,417,535,142);
topic(s,'Three networks, three roles','Generator: evolves the noisy molecule.\nf_A: provides the property gradient.\nf_B: evaluates decoded samples afterwards.\nAll weights stay frozen during sampling.',660,417,535,142);
strip(s,'BDG retains this gradient computation and modifies the scalar residual.',608);

s=fresh(14,'Guidance improves μ and gap in both families','2b','Part 2 · Guided versus unguided 2 of 2','[12] Chung et al. 2023 · results/v3/{fm,vp}/v3/n2000, audited in paper v9');
const guide=[['Property / model','Unguided IB %','Plug-in IB %','ΔIB (z)','Stable %','Valid %']];
for(const pr of props)for(const be of ['fm','vp']){const u=M(be,pr,'unguided'),q=M(be,pr,'plug'),c=D.diagnostics.guided_contrasts[`${be}/${pr}`];guide.push([`${pl[pr]} / ${be==='fm'?'FM':'VP'}`,pm(u),pm(q),`${sign(c.delta_pp)} (${f(c.z,2)})`,`${f(u.stable)} → ${f(q.stable)}`,`${f(u.valid)} → ${f(q.valid)}`]);}
table(s,guide,85,167,1110,49,[180,180,180,180,195,195],18);
tx(s,'q50, w = 1, second-half guidance. Each arm uses 3 seeds × 2,000 with the same masks and seeds. IB: mean ± seed SD. Quality: mean, unguided → guided.',85,523,1110,61,18,C.muted);
strip(s,'μ and gap clear |z| = 2.99 in both families (z = 3.94–5.22). α remains unresolved.',597);
tx(s,'Guidance trades some chemical quality for coverage. FM retains the stronger starting stability and validity.',85,636,1110,30,18);

s=fresh(15,'BDG adds feedback on batch dispersion','2c','Part 2 · Innovation 1 of 4','[12] Chung 2023 · [15] Ye 2024 · [22] Boys 2024 · [23] Song 2023 · related batch/moment work [16–21]');
tx(s,'Plug-in residual:     y − Fᵢ = (y − F̄) − (Fᵢ − F̄)',85,158,1110,44,29,C.blue,true);
tx(s,'A single strength scales both centring and contraction. BDG keeps the centring coefficient and adjusts the deviation coefficient using the measured batch variance.',85,216,1110,68,22);
table(s,[['Method','Information used for the correction'],['DPS-style plug-in [12]','One sample’s property residual'],['TMPD-inspired [22]','A local surrogate for endpoint uncertainty'],['LGD-MC [23]','Monte Carlo samples around an endpoint'],['TFG [15]','Additional optimization of predicted endpoints'],['BDG','Batch property variance relative to a requested spread τ']],85,315,1110,44,[330,780],20,[5]);
tx(s,'Contribution: this batch-dependent deviation rule within property-gradient guidance, reused on Euclidean molecules and simplex DNA. Prior work already studies batch interactions and moment control [16–21].',85,602,1110,60,20);

// Retain the original paper mechanism figure and its equations.
replace(old[15],'Gain η ≥ 0 and setpoint','Gain η ≥ 0. Requested spread τ = τₘs > 0, where s is the calibration property SD. τₘ controls the setpoint.');

s=fresh(17,'What the added term controls','2c','Part 2 · Innovation 3 of 4','Original formulation and algebra (paper v9) · reduction checks in proj1/tests/test_bdg.py');
tx(s,'e = (Vᵦ − τ²) / τ²         w_eff = 1 + ηe',85,161,1110,44,29,C.purple,true);
tx(s,'nᵢ = (y − F̄) − w_eff(Fᵢ − F̄) = (y − Fᵢ) − ηe(Fᵢ − F̄)',85,221,1110,46,29,C.blue,true);
topic(s,'η = 0 recovers plug-in exactly','This removes the extra residual term.\nτ must stay positive: τ = 0 divides by zero.\nChanging τₘ while η > 0 changes the response to batch spread.',85,313,535,146);
topic(s,'Contraction and expansion','e > 0 increases contraction.\n−1/η < e < 0 weakens contraction.\nw_eff < 0 permits expansion. Since Vᵦ ≥ 0, w_eff ≥ 1 − η. Velocity clipping still applies.',660,313,535,146);
tx(s,'At each step: nᵢ = w_eff(y_eff − Fᵢ), where y_eff = (y + ηeF̄) / w_eff if w_eff ≠ 0.',85,539,1110,49,23,C.ink,true);
tx(s,'The added control is adaptive gain and target adjustment along the same gradient. Finite sampling, nonlinear response and decoding affect the achieved spread.',85,604,1110,61,21);

function spreadChart(s,be,x){
 const colors=[C.blue,C.orange,C.green];const series=props.map((pr,i)=>({name:pl[pr],xValues:[.5,.75,1,1.5],values:D.diagnostics.spread_ratios[`${be}/${pr}`],line:{fill:colors[i],width:2.5},marker:{symbol:['circle','square','triangle'][i],size:7},fill:colors[i]}));
 series.push({name:'Unguided',xValues:[.5,1.5],values:[1,1],line:{fill:'#8B929A',width:1.3,style:'dashed'},marker:{symbol:'none'}});
 const ch=s.charts.add('scatter',{position:{left:x,top:203,width:538,height:335},title:be==='fm'?'Our FM':'EquiFM (flow matching)',titleTextStyle:{fontSize:24,bold:true},series,scatterOptions:{style:'lineWithMarkers'},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},xAxis:{min:.5,max:1.5,majorUnit:.25,title:'Setpoint τₘ',textStyle:{fontSize:16},numberFormatCode:'0.##'},yAxis:{min:.6,max:1.4,majorUnit:.2,title:'Spread / unguided',textStyle:{fontSize:16},numberFormatCode:'0.0',majorGridlines:{fill:'#DAD7CF',width:.8}},chartFill:C.bg,plotAreaFill:C.bg});applyPresentationChartFont(ch,{fontFamily:'IBM Plex Sans'});chartOwners.push(s.id);
}
s=fresh(18,'The setpoint controls spread on both flow backbones','2c','Part 2 · Innovation 4 of 4','Original measurements · results/v3/{fm,equifm}/v3abl/n2000 · pooled decoded property SD');
tx(s,'η = 8, w = 1. Four setpoints, three properties, 3 seeds × 2,000 per point.',85,154,1110,38,21);
spreadChart(s,'fm',85);spreadChart(s,'equifm',657);
strip(s,'All six curves increase monotonically: 0.67–0.77× at τₘ = 0.5, and 1.15–1.25× at τₘ = 1.5.',563);
tx(s,'The tested plug-in strengths w = 1 and 4 only narrow spread. BDG adds a wider operating range with the same network-call budget.',85,618,1110,53,21);

s=fresh(19,'Dipole ablation: removing and adjusting the added term','2d','Part 2 · Innovation ablations 1 of 2','Paper v9 Table 2 · results/v3/fm/v3abl/n2000 · full FM and EquiFM grids in appendix A8');
tx(s,'Our FM, q50 dipole: y = 2.4932 D, band ±0.17541 D. Decoded results, 3 seeds × 2,000.',85,154,1110,40,21);
const selected=[[1,'bdg_e0t1'],[4,'bdg_e0t1'],[1,'bdg_e4t0.5'],[1,'bdg_e4t1'],[1,'bdg_e8t0.5'],[1,'bdg_e8t1.5']];
const abl=[['w','η','τₘ','IB % ↑','MAE D ↓','Mean D','Spread D','Stable % ↑']];
for(const [w,a] of selected){const q=A('fm','mu',w,a);let [e,t]=a.slice(5).split('t');abl.push([w,e,e==='0'?'—':t,pm(q),f(q.mae,3),f(q.mean,3),f(q.spread,3),f(q.stable)]);}
table(s,abl,85,210,1110,46,[70,70,90,225,160,160,165,170],20,[3,5]);
tx(s,'η = 0 removes BDG and returns plug-in. At fixed w = 1, τₘ = 0.5 raises mean coverage and lowers MAE. Raising τₘ widens the batch and changes in-band occupancy.',85,556,1110,80,23,C.purple,true);
tx(s,'The w = 4 zero-gain row isolates stronger plug-in guidance. η = 8 tests a larger feedback gain.',85,638,1110,28,18);

s=fresh(20,'Ablations support an adjustable band-control effect','2d','Part 2 · Innovation ablations 2 of 2','Original analysis · paper v9 Tables 2, S4–S5 and full 612-cell molecular ablation grid');
topic(s,'Better endpoint error at the tighter setting','At η = 4, τₘ = 0.5, both flow backbones improve mean coverage and reduce decoded MAE on all three properties. Coverage gains are +0.45 to +1.52 points.',85,163,535,149);
topic(s,'Variation matters','The three gains on our FM have higher seed variation than plug-in. The registered |z| = 2.99 threshold leaves all six coverage differences unresolved.',660,163,535,149);
topic(s,'A controllable spread changes occupancy','Four setpoints change spread monotonically in all six tasks. τₘ = 0.5 concentrates more samples inside the band. τₘ = 1 gives a wider operating point, with lower coverage in three tasks and unresolved differences in three.',85,388,535,184);
topic(s,'Gain and strength are distinct controls','The grid removes η, changes τₘ and changes w. At η = 8, the coverage gain per stability point lost exceeds stronger plug-in in 5 of 6 tasks (4 of 6 at η = 4). These are descriptive ratios.',660,388,535,184);
tx(s,'The data establish the effect of the added term. Matched-force and replay controls would isolate the contribution of online feedback itself.',85,620,1110,49,19,C.muted);

s=fresh(21,'Recent guidance methods on the same frozen FM','2e','Part 2 · Recent-method comparison 1 of 2','[12] Chung 2023 (DPS) · [22] Boys 2024 (TMPD) · [23] Song 2023 (LGD-MC) · [15] Ye 2024 (TFG)');
const cmp=[['Method','IB μ % ↑','IB α % ↑','IB gap % ↑','Stable μ/α/gap %','DV μ %','Time s']];
for(let i=0;i<arms.length;i++){const a=arms[i],q=M('fm','mu',a);cmp.push([names[i],...props.map(pr=>pm(M('fm',pr,a))),props.map(pr=>f(M('fm',pr,a).stable)).join(' / '),f(q.dv),f(q.time,3)]);}
table(s,cmp,85,160,1110,45,[250,150,150,150,220,95,95],17.5,[6,7]);
tx(s,'q50, w = 1, t ≥ 0.5. Mean ± seed SD for IB, means elsewhere. DV = distinct valid molecules / attempts. Time: μ cells, RTX A6000, including scoring.',85,535,1110,57,18,C.muted);
strip(s,'BDG uses plug-in’s network calls and runtime. Its τₘ = 0.5 setting has the highest mean coverage among the non-TFG arms.',604);

s=fresh(22,'Coverage, quality and comparison scope','2e','Part 2 · Recent-method comparison 2 of 2','[12] Chung et al. 2023 · [22] Boys et al. 2024 · [23] Song et al. 2023 · [15] Ye et al. 2024');
topic(s,'Where BDG improves the operating point','τₘ = 0.5 raises mean coverage and lowers MAE versus plug-in across six molecular tasks. On our FM’s μ task, τₘ = 1 has the highest guided stability (40.5%) and DV (76.3%).',85,163,535,154);
topic(s,'Where TFG obtains more coverage','TFG has higher mean coverage on our FM: μ 28.8% versus BDG’s 11.2%. Its stability is 24.5% versus 35.1%. Across the comparison, coverage comes with larger stability losses.',660,163,535,154);
topic(s,'All scores are our own measurements','Same generator, targets, seeds and evaluator within a backend. All four external methods operate at inference time on frozen models, which makes them relevant comparators.',85,402,535,140);
topic(s,'Adaptations and limits','Plug-in uses Gaussian energy. TMPD uses a local-linear scalar surrogate. TFG uses our deterministic ODE and feature scale. These runs compare implementations at fixed nominal settings, not optimally tuned methods.',660,402,535,160);
tx(s,'EquiFM uses a different released predictor pair and different calibrated bands. Compare arms within that backend.',85,624,1110,43,20,C.muted);

s=fresh(23,'Transfer to DNA keeps the BDG rule','3a','Part 3 · Transfer method','[4] Stark et al. 2024 (DeepFlyBrain / Dirichlet FM) · proj1/m2/simplex_fm.py · original BDG transfer');
tx(s,'Keep the same F̄, Vᵦ, e, τ = τₘs and deviation gain w_eff. Only the generator, property map and state constraints change.',85,157,1110,64,23,C.purple,true);
table(s,[['Component','QM9','DNA'],['State','Coordinates + continuous type channels','500 × 4 probability simplex'],['Generator','EGNN velocity, 3.75M parameters','Dilated 1D CNN, 128 × 10, about 1.02M'],['Guide / evaluator','Learned f_A / independent learned f_B','Soft analytic GC or CpG / exact decoded count'],['Feasibility','Mask and recenter coordinates','Clamp and renormalize after each step']],85,239,1110,48,[190,420,500],18);
topic(s,'Training and checkpoint','83,722 sequences. Dirichlet-to-one-hot linear FM. Adam 2 × 10⁻³, batch 256, cosine to zero, no EMA. 1,500 epochs. Validation selects epoch 1,450 (loss 0.0634).',85,515,535,119);
topic(s,'Sampling','100 Euler steps, batch 500, 3 × 2,000 per arm. Original t ≥ 0.5, w ∈ {1, 4}. New GC follow-up: t ≥ 0.3, w = 4. η = 4, τₘ = 0.5; CpG has no new early-window run.',660,515,535,119);

function dnaTable(s,w,y=202,early=false){
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

s=fresh(25,'Transfer gains depend on the property and setting','3c','Part 3 · Cross-modality analysis','Original decoded contrasts, paper v9 · bars show mean differences · paired seed SD reported alongside');
const pairs=[['GC .5',D.dnaContrasts['gc/4/bdg_e4t0.5']],['GC .3',D.windowContrasts['bdg_e4t0.5']],['CpG .5',D.dnaContrasts['cpg/4/bdg_e4t0.5']]];
const transfer=s.charts.add('bar',{position:{left:85,top:180,width:565,height:368},title:'BDG τₘ = 0.5 minus plug-in (pp)',titleTextStyle:{fontSize:21,bold:true},categories:pairs.map(x=>x[0]),series:[{name:'w = 4',values:pairs.map(x=>x[1].delta_pp),fill:C.blue}],hasLegend:false,barOptions:{direction:'column',grouping:'clustered',gapWidth:150},yAxis:{min:0,max:20,majorUnit:5,textStyle:{fontSize:17},majorGridlines:{fill:C.line,width:.7}},xAxis:{textStyle:{fontSize:18},tickLabelPosition:'low'},chartFill:C.bg,plotAreaFill:C.bg});applyPresentationChartFont(transfer,{fontFamily:'IBM Plex Sans'});chartOwners.push(s.id);
table(s,[['Property / t_min','ΔIB ± paired SD','Status'],...pairs.map(([pr,q])=>[pr,`${sign(q.delta_pp)} ± ${f(q.sd,2)}`,pr==='GC .3'?'Follow-up':'Original'])],690,190,505,52,[150,220,135],18);
tx(s,'Earlier GC guidance increases BDG’s mean gain, with greater seed variation. The GC follow-up was chosen after diagnosing the late-window regime.',690,433,505,143,21);
tx(s,'QM9: +0.45 to +1.52 pp, unresolved by the registered test. DNA: clear mean gains, with property, band and window differences retained in interpretation.',85,602,1110,69,21,C.purple,true);

s=fresh(26,'Measured limitations and failure modes','3d','Part 3 · Limits of the current evidence','Original diagnostics · paper v9 · results/v3 and results/m2');
topic(s,'1. Coverage can cost sample quality','Molecular guidance reduces stability. Strong DNA coverage also shifts 3-mer composition. In-band coverage alone does not measure useful molecular yield.',85,165,535,139);
topic(s,'2. Guidance timing can dominate','At t ≥ 0.3, GC baselines separate and BDG gains +7.08 pp. At t ≥ 0, the earlier w = 1 diagnostic put BDG 2.33 pp below plug-in. Timing changes the result.',660,165,535,145);
topic(s,'3. Nominal strength does not match force','GC at t ≥ 0.3: BDG clips 20.8% of guided sample-steps versus 14.9% for plug-in. CpG at t ≥ 0.5: 7.4% versus 0.21%. Match applied force to isolate feedback.',85,393,535,142);
topic(s,'4. Scores and uncertainty have limits','QM9 inherits evaluator error. DNA soft scores differ from decoded counts. Batch feedback couples samples, while pooled binomial tests assume independence. Three seeds provide limited resolution.',660,393,535,162);
tx(s,'Next measurements: batch-aware paired intervals, in-band-and-stable yield, and open-loop replay of the recorded BDG gains.',85,626,1110,42,21,C.muted);

s=fresh(27,'Conclusion: adjustable spread and positive transfer','3e','Part 3 · Conclusion and next steps','Original conclusions from two flow backbones and two modalities (paper v9)');
tx(s,'BDG adds a batch-dispersion control to plug-in guidance',85,167,1110,49,31,C.purple,true);
topic(s,'Molecules','The setpoint changes decoded spread monotonically on both flow backbones. At τₘ = 0.5, all six tasks have higher mean coverage and lower MAE than plug-in, with unresolved coverage differences.',85,257,535,167);
topic(s,'DNA','GC at t ≥ 0.3 gains +7.08 pp; CpG at t ≥ 0.5 gains +16.83 pp (w = 4). Hamming changes little. GC JSD improves over unguided, while CpG JSD rises.',660,257,535,167);
tx(s,'The control uses plug-in’s network-call budget. η = 0 recovers plug-in, which makes the added term directly testable.',85,488,1110,68,25,C.blue,true);
tx(s,'Next: address confounding with tighter atom-count control and target-range tests that account for the local property distribution. Add matched-force controls and joint useful yield.',85,590,1110,74,23);

// Keep the bibliography supplied with v2. Entry [13] is an implementation choice,
// not an invented external citation; no new claim is attached to it.
replace(old[27],'[13] Velocity-relative trust region:','[13] Velocity-relative correction cap: implementation choice in this study, shared by every guidance arm.');

// Appendix is reordered after all main slides and bibliography.
for(let i=29;i<old.length;i++)old[i].delete();
s=p.slides.add();s.setLayout(p.layouts.items[0]);s.background.fill=C.ink;
tx(s,'APPENDIX · FOR Q&A',85,96,1110,30,18,'#8FB4E5',true);
tx(s,'Implementation and full comparisons',85,169,1110,128,51,C.bg,true,{typeface:'Source Serif 4'});
const index=[['A1','EquiFM and its adapter'],['A2','EGNN, f_A and f_B'],['A3','Data split and fixed sizes'],['A4','Guidance rules and cost'],['A5','BDG algorithm'],['A6','DNA model architecture'],['A7','Point cloud to molecule'],['A8','All molecular ablation grids'],['A9','DNA windows and strengths']];
index.forEach(([a,t],i)=>{const col=Math.floor(i/5),row=i%5;tx(s,a,85+col*580,339+row*59,65,30,22,'#8FB4E5',true);tx(s,t,159+col*580,339+row*59,460,38,23,C.bg);});

s=append('EquiFM is a separate flow-matching backend','A1','Appendix A1 · Borrowed generator','[24] Song et al. 2023 (EquiFM) · [2] Hoogeboom et al. 2022 · released checkpoint audit');
table(s,[['Released checkpoint','Configuration'],['Architecture','Equivariant GNN dynamics, 256 × 9'],['Parameters','5,339,920 (EMA)'],['State per atom','3 coordinates + one-hot types / 4 + charge / 10'],['Native time','τ = 1 at noise, τ = 0 at data'],['Sampling in our study','100 Euler steps, 3 seeds × 2,000']],85,165,1110,47,[310,800],20);
topic(s,'Hybrid probability path','Coordinates use aligned noise (rotation and permutation matching). Type and charge channels use a variance-preserving path. These differ from our straight Gaussian path.',85,480,535,137);
topic(s,'Why include it','It tests the guidance rule on a second flow generator trained outside this project. Its 86.2% molecule stability provides a different starting operating point.',660,480,535,137);

s=append('The EquiFM adapter preserves its native dynamics','A1','Appendix A1 · Backend adapter','[24] Song et al. 2023 · [15] Ye et al. 2024 · transfer_sweep.py EquiFM backend and calibration');
table(s,[['Adapter component','Treatment'],['Predicted endpoint','Use EquiFM’s regression-target algebra for coordinates and features.'],['Plug-in / BDG / TFG','Differentiate the adapted endpoint through its native model.'],['TMPD / LGD-MC','Use a covariance surrogate. It is approximate for aligned coordinates.'],['Feature channels','Keep the model’s normalization. Hold charge fixed during guidance.'],['Guide / evaluator','Use TFG’s released property pair, with backend-specific calibrated bands.']],85,163,1110,59,[295,815],20);
tx(s,'Band widths differ from our predictor pair: δ = 0.157 D (μ), 0.165 Bohr³ (α), 0.0037 Ha (gap). Compare guidance arms within EquiFM.',85,546,1110,76,22,C.purple,true);
tx(s,'Same-seed reruns can shift coverage by up to 0.7 points. Keep this numerical variation in view for small contrasts.',85,628,1110,38,19,C.muted);

s=append('EGNN: the geometry built into the model','A2','Appendix A2 · EGNN 1 of 4','[9] Satorras, Hoogeboom & Welling, E(n)-Equivariant Graph Neural Networks, ICML 2021 · egnn.py');
s.speakerNotes.textFrame.setText('Sources: https://proceedings.mlr.press/v139/satorras21a.html (EGNN original paper). Our implementation: proj1/src/models/egnn.py.');
tx(s,'EGNN = E(n)-Equivariant Graph Neural Network',85,162,1110,44,28,C.blue,true);
topic(s,'Graph','Each real atom is a node. Connect every pair of distinct real atoms (at most 29). The mask excludes padding and self-pairs.',85,245,535,124);
topic(s,'E(n): Euclidean symmetry','Rotations, reflections and translations. In 3D, rotating a molecule rotates its coordinate updates. Scalar properties such as dipole magnitude remain unchanged.',660,245,535,143);
topic(s,'How each layer preserves symmetry','Squared distances do not change under rotation. Difference vectors rotate with the molecule. Learned scalar weights multiply those vectors. Summing neighbours removes dependence on their order.',85,442,535,160);
topic(s,'What we implemented and trained','Our plain-PyTorch EGNN and all “our” generators and predictors train from scratch on QM9. EquiFM and EDM use separate borrowed checkpoints.',660,442,535,139);
tx(s,'A dipole vector rotates with the molecule. Our target μ is its scalar magnitude. Recentring removes translation from coordinate velocities.',85,630,1110,39,18,C.muted);

s=append('One EGNN layer, step by step','A2','Appendix A2 · EGNN 2 of 4','[9] Satorras et al. 2021 · proj1/src/models/egnn.py:58–106 (our masked implementation)');
tx(s,'Atom i carries a feature vector hᵢ and a position xᵢ. Each φ is a small MLP.',85,157,1110,37,22);
tx(s,'1. Messages',85,224,250,38,25,C.blue,true);
tx(s,'mᵢⱼ = φₑ(hᵢ, hⱼ, ‖xᵢ − xⱼ‖²)',355,224,840,43,28,C.ink,true);
tx(s,'What atom j tells atom i, given their features and squared distance.',355,276,840,50,21);
tx(s,'2. Features',85,353,250,38,25,C.blue,true);
tx(s,'hᵢ ← hᵢ + φₕ(hᵢ, Σⱼ mᵢⱼ)',355,353,840,43,28,C.ink,true);
tx(s,'Add the messages, then update atom i’s feature vector.',355,405,840,43,21);
tx(s,'3. Coordinates',85,478,250,38,25,C.blue,true);
tx(s,'xᵢ ← xᵢ + Σⱼ (xᵢ − xⱼ) φₓ(mᵢⱼ) / (dᵢⱼ + 1)',355,478,840,49,27,C.ink,true);
tx(s,'Move along atom-to-atom directions by learned scalar amounts.\ndᵢⱼ = √(‖xᵢ − xⱼ‖² + 10⁻⁸) in code.',355,537,840,67,21);
tx(s,'Mask every update. One dense layer already reaches all atoms. Stacking layers learns more complex interactions.',85,629,1110,39,19,C.muted);

s=append('Two EGNN classes, two different jobs','A2','Appendix A2 · EGNN 3 of 4','[9] Satorras et al. 2021 · egnn.py: EGNNVelocity and EGNNScalar');
tx(s,'EGNNVelocity: the generator',85,164,535,36,26,C.blue,true);
tx(s,'Input: noisy coordinates, continuous type channels, time t and atom mask.',85,222,535,67,22);
tx(s,'Run the EGNN stack with coordinate updates.\nCoordinate output = final − initial positions, then recenter.\nType output = MLP on each final atom feature.',85,306,535,154,22);
tx(s,'FM reads these outputs as velocity.\nVP reads the same architecture’s outputs as noise ε. A 100-step FM sampler calls this generator once per step.',85,485,535,128,22);
tx(s,'EGNNScalar: one property value',660,164,535,36,26,C.purple,true);
tx(s,'Input: a molecule and its atom mask.\nKeep coordinates fixed (update_coords=False).',660,222,535,81,22);
tx(s,'1. Run the feature-message layers.\n2. Average features over real atoms only.\n3. Apply the pooling MLP: this is the embedding.\n4. One linear head returns μ, α or gap.',660,321,535,176,22);
tx(s,'The pooled scalar is unchanged by atom order or rigid transformations. f_A and f_B are separately trained copies of this class.',660,525,535,105,22);

s=append('f_A steers; f_B evaluates after decoding','A2','Appendix A2 · Property predictors 4 of 4','[9] Satorras et al. 2021 · predictor training and calibration audit · [15] Ye et al. 2024');
table(s,[['','f_A: guide','f_B: evaluator'],['Training data','train_a, 51,527 molecules','train_b, 51,527 molecules'],['Architecture','EGNNScalar, width 128 × 4, 429,825 parameters per property','Same architecture, independent weights'],['Training','Standardized-target MSE, Adam 3 × 10⁻⁴ cosine, batch 128, 120 epochs','Best validation MAE, no EMA'],['Sampling role','Differentiate f_A(m) through the predicted endpoint m','Never called inside the guidance update'],['Evaluation role','Diagnostic guide score','Score final coordinates and decoded one-hot atom types']],85,161,1110,56,[180,465,465],18);
tx(s,'Our pair evaluates our FM, VP and borrowed EDMsecond. EquiFM uses TFG’s released predictor pair and its own bands.',85,522,1110,66,21);
tx(s,'Calibration MAE of f_B: μ 0.0877 D, α 0.2531 Bohr³, gap 0.00374 Ha. Band half-width δ = 2 × MAE. Independent training reduces shared bias but does not eliminate evaluator error.',85,603,1110,67,21,C.purple,true);

s=append('Data split and fixed atom-count protocol','A3','Appendix A3 · Data and sample sizes','QM9 split seed 20260917 · transfer_sweep.py:2382–2426 · first 2,000 validation masks in data/qm9.pt');
table(s,[['Split','Used by','n'],['train_a','Our FM, VP and f_A','51,527'],['train_b','Our f_B only','51,527'],['Validation','Checkpoint selection and the 2,000 size masks','17,748'],['Test','Held out from training and checkpoint selection','13,083']],85,161,1110,49,[240,685,185],20);
topic(s,'Exactly which sizes','Molecule i uses validation molecule i’s atom count for i = 0…1999. The same ordered list goes to every arm, seed and backend. Seeds change only initial noise.',85,443,535,144);
topic(s,'Size and target are separate','Every molecule receives the same q50 property target, whatever its size. No histogram sampling, target-based filtering or size filtering. Min 8, median 18, mean 17.87, max 29.',660,443,535,144);
tx(s,'Calibration uses 3,000 molecules from train_a + train_b. Each of our predictors is scored on the half it did not train on.',85,627,1110,39,19,C.muted);

s=append('Guidance rules in a shared endpoint framework','A4','Appendix A4 · Rules 1 of 2','[12] Chung et al. 2023 · [22] Boys et al. 2024 · [23] Song et al. 2023 · [15] Ye et al. 2024');
table(s,[['Arm','What changes relative to plug-in'],['DPS-style plug-in','Gaussian property residual (y − f_A(m)) / s², pulled back through Jᵀ.'],['TMPD-inspired','Divide by s² + gᵀΣg using a local covariance surrogate, g = ∇f_A(m).'],['LGD-MC','Average property gradients over Monte Carlo perturbations near m, with likelihood weights.'],['TFG','Add direct optimization of the predicted endpoint, with four refinement steps.'],['BDG','Replace the residual with (y − F̄) − (1 + ηe)(Fᵢ − F̄). Same gradient and pullback.']],85,171,1110,66,[250,860],20,[5]);
tx(s,'All are adaptations on our frozen checkpoints. Shared atom masks, guidance window and velocity-relative cap. TMPD’s scalar surrogate and our TFG sampler differ from the original full algorithms.',85,591,1110,76,21);

s=append('BDG retains plug-in’s network-call budget','A4','Appendix A4 · Recorded sampling operations 2 of 2','Recorded cost fields · our FM, μ, w = 1, 2,000-molecule cell · paper v9 cost table');
const costs=[['Method','Gen. forward','Gen. VJP','Gen. JVP','Guide forward','Guide backward']];
for(let i=0;i<arms.length;i++){let q=D.cost[arms[i]];costs.push([names[i],...['gen_fwd','gen_vjp','gen_jvp','guide_fwd','guide_bwd'].map(k=>q[k])]);}
table(s,costs,85,175,1110,46,[280,166,166,166,166,166],18,[6,7]);
tx(s,'A VJP is a vector–Jacobian product. A JVP is a Jacobian–vector product. Forward and derivative calls are separate operations, not equal-cost units.',85,565,1110,62,21);
tx(s,'BDG adds scalar batch reductions only. Main-table wall-clock also includes the evaluator.',85,635,1110,34,20,C.purple,true);

s=append('One BDG-guided Euler step','A5','Appendix A5 · Algorithm','Original algorithm · proj1/src/guidance.py and proj1/tests/test_bdg.py · η = 0 verified on our FM');
const algo=[['Step','Computation'],['1','Compute v = vθ(xₜ, t). Before the guidance window, use xₜ₊ₕ = Π(xₜ + hv).'],['2','Predict endpoints mᵢ = xₜ,ᵢ + (1 − t)vᵢ, with differentiation enabled.'],['3','Evaluate Fᵢ = f_A(mᵢ), mean F̄ and unbiased batch variance Vᵦ.'],['4','Set τ = τₘs > 0, e = (Vᵦ − τ²) / τ² and w_eff = 1 + ηe.'],['5','Detach aᵢ = [(y − Fᵢ) − ηe(Fᵢ − F̄)] / s².'],['6','One vector–Jacobian product: G = VJPₓ(F, a).'],['7','Set Cᵢ = w(1 − t)/max(t, ε) Gᵢ and cap ‖Cᵢ‖ at ‖vᵢ‖.'],['8','Return Π{xₜ + h(v + C)}.']];
table(s,algo,85,155,1110,51,[85,1025],20,[4,5]);
tx(s,'Π masks and recentres molecules. For DNA, Π clamps and renormalizes each position. Batch size 500 means four independent controller batches per 2,000-sample cell.',85,632,1110,39,19);

s=append('DNA velocity network and training','A6','Appendix A6 · Simplex FM 1 of 2','[4] Stark et al. 2024 · our implementation: proj1/m2/simplex_fm.py');
table(s,[['Stage','Implementation'],['Input','500 × 4 one-hot or relaxed base channels (A, C, G, T)'],['Stem','Conv1d 4 → 128, kernel 5, padding 2'],['Body','10 residual blocks: dilated Conv1d, GroupNorm(8), SiLU, 1×1 Conv1d'],['Dilations','1, 2, 4, 8 cycling across blocks'],['Time','Sinusoidal embedding + MLP, added within every block'],['Head','1×1 Conv1d 128 → 4, zero-initialized. About 1.02M parameters.']],85,157,1110,44,[230,880],20);
topic(s,'Path and objective','Linear interpolation from Dirichlet(1,1,1,1) noise to one-hot sequences. Regress the pairwise target velocity with mean squared error.',85,501,535,105);
topic(s,'Completed training','1,500 epochs, 492k updates, 539.7 minutes on RTX A6000. Validation selects epoch 1,450 (update 475,600, loss 0.0633644).',660,501,535,105);
tx(s,'GC is affine in relaxed base probabilities. CpG is quadratic. Their bands and clipping also differ, so curvature alone is not an established explanation.',85,631,1110,39,18,C.muted);

s=append('DNA properties, targets and bands','A6','Appendix A6 · Simplex FM 2 of 2','Original DNA protocol · paper v9 Appendix: Sequence Transfer and Failure Diagnostics · results/m2');
tx(s,'GC(p) = Σⱼ(pⱼ,C + pⱼ,G) / L',85,170,1110,44,28,C.blue,true);
tx(s,'CpG(p) = Σⱼ pⱼ,C pⱼ₊₁,G / (L − 1)       L = 500',85,238,1110,47,28,C.purple,true);
tx(s,'The soft score guides an extrapolated endpoint. The final score counts C/G bases or adjacent C–G pairs after argmax. The two scores need not agree.',85,307,1110,69,22);
table(s,[['Property','q50 target','Calibration SD s','Half-width δ','δ / s'],['GC','0.454000','0.055206','0.008833','0.160'],['CpG','0.046092','0.014748','0.008818','0.598']],85,401,1110,49,[180,230,250,230,220],21);
tx(s,'δ = max(0.16s, 4.4q), with count quantum q = 1/500 for GC and 1/499 for CpG. The quantum floor sets the wider relative CpG band.',85,574,1110,72,23);
tx(s,'Compare arms within each property. Composition and diversity metrics do not establish biological enhancer function.',85,649,1110,26,17,C.muted);

s=append('From atom cloud to molecule: types and bonds','A7','Appendix A7 · Molecular decoding 1 of 2','evaluation.py:73–95 · [2] Hoogeboom et al. 2022 (EDM bond-length tables, picometres)');
topic(s,'1. The generator outputs a point cloud','Each real atom slot contains (x, y, z) and five continuous type numbers. There are no bonds yet. Intermediate states are noisy point clouds.',85,161,535,117);
topic(s,'2. Argmax decides the element','H / C / N / O / F channels:\n[0.02, 0.97, 0.01, 0.00, 0.03] becomes C.\nOnly real slots participate. Ignore padding.',660,161,535,117);
tx(s,'3. Infer bond orders from pairwise distances',85,330,1110,36,25,C.blue,true);
tx(s,'Convert Å to pm (×100). Compare each element pair with EDM’s single/double/triple reference lengths, using margins of 10/5/3 pm.',85,381,1110,67,22);
table(s,[['C–C example','Reference length','Threshold in this decoder'],['Single bond','154 pm','139 ≤ d < 164 pm'],['Double bond','134 pm','123 ≤ d < 139 pm'],['Triple bond','120 pm','d < 123 pm'],['No bond','—','d ≥ 164 pm']],85,465,1110,35,[330,330,450],18);
tx(s,'Geometric decoding uses strict upper thresholds. It does not calculate molecular energy.',85,653,1110,23,17,C.muted);

s=append('Chemical validity and property scoring are separate','A7','Appendix A7 · Molecular decoding 2 of 2','evaluation.py: atom stability, to_smiles and evaluate_molecules · [2] Hoogeboom et al. 2022');
topic(s,'4. Check atom and molecule stability','Add bond orders at each atom. The implemented allowed valences are H:1, C:4, N:3, O:2, F:1. An atom is stable if its total matches. A molecule is stable only if every real atom passes.',85,164,535,159);
topic(s,'5. Build and sanitize with RDKit','Create atoms and inferred bonds, sanitize, then form canonical SMILES. Validity is the fraction that succeeds. Uniqueness counts distinct valid SMILES. DV divides that count by all attempts.',660,164,535,159);
topic(s,'6. Evaluate the decoded property','f_B receives the final coordinates and argmax one-hot types. It predicts μ, α or gap. Check whether the value lies in y ± δ. The property network does not read the inferred bond graph.',85,401,535,158);
topic(s,'Why report multiple metrics','The in-band calculation includes invalid molecules. Stability, validity and diversity therefore accompany coverage. Current aggregate results do not give the joint fraction that is both in-band and stable.',660,401,535,158);
tx(s,'f_B is an independent learned proxy for physical properties. Its error remains part of the molecular result.',85,631,1110,39,20,C.purple,true);

function heat(s,be,pr,w,x,y){
 tx(s,`${pl[pr]}  ·  w = ${w}`,x,y,350,30,23,C.blue,true);
 const vals=[['η∖τₘ','0.5','0.75','1','1.5']];const u=A(be,pr,w,'bdg_e0t1');
 for(const e of [1,2,4,8])vals.push([String(e),...[.5,.75,1,1.5].map(t=>sign(A(be,pr,w,`bdg_e${e}t${t}`).ib-u.ib,1))]);
 const t=table(s,vals,x,y+37,350,35,[70,70,70,70,70],17);
 for(let r=1;r<5;r++)for(let c=1;c<5;c++){const v=Number(vals[r][c]);t.getCell(r,c).fill=v>=0?'#E0EDF6':v>-2?'#F9EDE7':v>-5?'#EDBFB1':'#AA3D36';if(v<=-5)t.getCell(r,c).text.style={color:'#FFFFFF'};}
}
for(const be of ['fm','equifm']){
 s=append(`${be==='fm'?'Our FM':'EquiFM'}: complete gain × setpoint coverage grid`,'A8','Appendix A8 · Full molecular ablations',`Original 306-cell ${be} ablation grid · all values decoded · paper v9 appendix tables`);
 tx(s,'ΔIB in percentage points versus η = 0 at the same w. Each cell: mean across 3 seeds × 2,000.',85,154,1110,42,21);
 for(let j=0;j<3;j++)heat(s,be,props[j],1,85+j*380,200);
 for(let j=0;j<3;j++)heat(s,be,props[j],4,85+j*380,431);
}

s=append('DNA transfer at the lower strength','A9','Appendix A9 · Full w = 1 comparison','All four guidance adaptations on our DNA backbone · same evaluator and decoding as the w = 4 main table');
tx(s,'w = 1, t ≥ 0.5. Hamming and JSD shown for both properties.',85,158,1110,39,22);
dnaTable(s,1,213);
tx(s,'GC and CpG IB agree exactly for plug-in, TMPD and LGD-MC at both w = 1 and 4 in these runs. Other decoded metrics differ by at most 4.22 × 10⁻⁶ (GC) and 3.40 × 10⁻⁶ (CpG) in unscaled units.',85,574,1110,83,21,C.purple,true);
tx(s,'IB: mean ± seed SD. JSD ×10⁻⁴. Hamming: first 256 sequences. Time: CpG cells.',85,649,1110,23,16,C.muted);


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
tx(s,'s = training property SD\nqGC = 1/500\nqCpG = 1/499\nFull bandwidth = 2δ',907,362,285,154,21);
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

// Audit exact source data and write previews plus the candidate.
await fs.mkdir(path.join(BUILD,'draft_v4'),{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.items[i];
 const png=await slide.export({format:'png',scale:1});await fs.writeFile(path.join(BUILD,'draft_v4',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 const layout=await slide.export({format:'layout'});await fs.writeFile(path.join(BUILD,'draft_v4',`slide-${String(i+1).padStart(2,'0')}.json`),await layout.text());
}
const owners=ids=>[...new Set(ids.map(id=>p.slides.items.findIndex(s=>s.id===id)+1))].sort((a,b)=>a-b);
await fs.writeFile(path.join(BUILD,'authored_inspect_v4.ndjson'),(await p.inspect({kind:'slide,textbox,table,chart,image',maxChars:2000000})).ndjson);
const candidate=path.join(BUILD,'candidate_v4_revised.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
console.log('AUTHORED',p.slides.items.length,'slides; tables',owners(tableOwners),'charts',owners(chartOwners));
const finalName=process.env.FINAL_NAME||'BDG_Defense_Group_2_v4_revised.pptx';
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:candidate,finalPath:path.join(OUT,finalName),pythonExecutable:'C:/Users/mooooonesy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...owners(tableOwners).flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:owners(tableOwners),requiredNativeChartOwnerSlides:owners(chartOwners),nativeChartTargetApplication:'powerpoint',materializeLiteralChartWorkbooks:true,fontPolicy,verifyArtifactToolImport:true,receiptPath:path.join(BUILD,finalName+'.validation.json')});
console.log(JSON.stringify(result));
