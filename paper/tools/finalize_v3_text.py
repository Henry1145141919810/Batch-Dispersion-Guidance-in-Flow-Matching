from pathlib import Path
root = Path(__file__).resolve().parent.parent
p = root/'main.tex'
s = p.read_text(encoding='utf-8')
s = s.replace('QM9 contains 133{,}885 molecules', r'QM9 \citep{ramakrishnan2014qm9} contains 133{,}885 molecules', 1)
s = s.replace('DeepFlyBrain contains 500-base-pair enhancer sequences', r'DeepFlyBrain \citep{stark2024dirichlet} contains 500-base-pair enhancer sequences', 1)
s = s.replace('FM and VP trained here', 'FM / VP training comparison')
s = s.replace('Blue identifies models trained for this project', 'Blue identifies project model development')
s = s.replace('dilated 1-D CNN\\\\', 'CNN, 128, 10 layers\\\\')
s = s.replace(r'Batch & 256$^*$ & 128 & \pend', r'Batch & 256$^*$ & 128 & 256')
s = s.replace(r'Learning rate & $2\times10^{-4}$, cosine$^*$ & $3\times10^{-4}$, cosine & \pend', r'Learning rate & $2\times10^{-4}$, cosine$^*$ & $3\times10^{-4}$, cosine & $2\times10^{-3}$, cosine')
s = s.replace('Outstanding evidence includes matched FM--VP evaluation, guidance results, BDG ablations, recent-method comparisons in each modality, per-seed values, cost, and checkpoint/software identifiers. This version defines the method and comparison structure. Competitive performance and cross-modality improvement remain unresolved.', '')
s = s.replace('A seeded permutation creates the training halves, validation, and test. Centering removes translation; the generator is E(3)-equivariant and predictors are invariant.', 'A seeded permutation creates the splits. Centering removes translation; predictors are invariant.')
p.write_text(s, encoding='utf-8')
p = root/'refs_extra.bib'
s = p.read_text(encoding='utf-8')
s = s.replace('Equivariant Flow Matching with Hybrid Probability Transport\n             for 3D Molecule Generation', 'Equivariant Flow Matching with Hybrid Probability Transport')
s += '''\n@article{ramakrishnan2014qm9,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O. and Rupp, Matthias and von Lilienfeld, O. Anatole},
  journal={Scientific Data},
  volume={1},
  pages={140022},
  year={2014},
  doi={10.1038/sdata.2014.22}
}\n'''
p.write_text(s, encoding='utf-8')
