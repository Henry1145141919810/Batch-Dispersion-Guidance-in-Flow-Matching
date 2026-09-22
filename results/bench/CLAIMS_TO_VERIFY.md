# Claims to verify independently

Each claim below will appear in the benchmark report. For each: confirm, refute, or mark
unverifiable, with the evidence you used. Sources are local files under `results/bench/papers/`
(PDFs plus `pypdf` text dumps as `.txt`) and the code under `proj1/`.

## A. Our own protocol (code audit)

A1. `proj1/scripts/benchmark_base.py` implements EDM's metric conventions:
    validity = RDKit sanitises the inferred graph, taking the LARGEST CONNECTED FRAGMENT;
    uniqueness = distinct canonical SMILES among valid; novelty = fraction of unique valid
    SMILES not in the training-set SMILES, where training SMILES are inferred from 3D by the
    same bond code; molecule sizes drawn from the train_a size distribution; EMA weights.
A2. Atom stability in `proj1/src/evaluation.py::stability` = bond-order sum EQUALS allowed
    valence {H:1,C:4,N:3,O:2,F:1}, hydrogens included; molecule stable iff every atom is.
A3. The aggregate fields in `results/bench/fm_nfe100_euler_s0.json` (atom_stability,
    mol_stability, validity_edm, uniqueness_edm, valid_x_unique_edm, novelty_vs_train_a,
    novelty_vs_train_ab, connected_of_valid) can be recomputed from its `records` list and
    from `results/bench/dataset_smiles.json`, and they match.
A4. `proj1/scripts/eval_conventions.py` convention `edm_H` reproduces the JSON aggregates
    exactly on `results/bench/fm_nfe100_euler_s0_samples.pt` (0.9352 / 0.3827 / 0.7577).
A5. No evaluation leakage: the generator was trained on `train_a` only (check
    `betty_pull/fm.pt` -> `args.split`); novelty is reported against train_a AND train_a+b.
A6. Seeds: benchmark seed 0/1/2 differ from the trainer's stability-eval seed (4321) and
    inspect_checkpoint's (99991), so the benchmark is a fresh sample.

## B. Metric-convention claims about TFG-Flow

B1. `results/bench/papers/tfg_flow_code/evaluator.py` (fetched from
    github.com/linhaowei1/TFG-Flow, master) defines an atom as stable iff
    `atom_orders[i] <= allowed_bonds_` (less-than-or-equal, NOT equality).
B2. In that file's `compute_stability_one_mol`, the double loop increments ONLY
    `atom_orders[i]`, never `atom_orders[j]`.
B3. TFG-Flow paper, Appendix D.2: "In our implementation, we only model heavy atoms."
B4. TFG-Flow paper, Appendix D.2: EGNN 9 layers, 256 hidden, Adam lr 1e-4, batch size 256,
    1200 epochs for QM9 with early stopping on validation loss; only atom types and
    coordinates modelled (no charges).
B5. TFG-Flow paper, Section 4.1: training set split into two halves; the first half trains
    the unconditional flow model and the guidance predictor; 100 Euler sampling steps.
B6. TFG-Flow Table 8 (TFG-Flow, guided, 3 seeds) rows, in the order
    Validity / Uniqueness / Novelty / Mol. stability / Atom stability / Connectivity:
      alpha  76.31 99.43 93.44 89.89 98.75 66.40
      mu     75.34 99.01 89.89 86.85 98.33 63.64
      Cv     75.56 98.99 90.89 87.78 98.43 68.55
      gap    75.95 97.42 87.26 88.41 98.52 64.86
      HOMO   87.93 95.90 89.14 94.51 99.32 73.72
      LUMO   80.76 98.46 90.57 91.58 99.00 67.42
    and Table 6 (Cond-flow) alpha row: 78.03 98.78 87.75 89.54 98.49 69.55.
B7. TFG-Flow reports NO unconditional (no-guidance) generation-quality row for QM9.
B8. `proj1/scripts/eval_conventions.py` conventions `tfgflow` and `tfgflow_ascoded`
    faithfully reproduce B1 and B1+B2 respectively (heavy atoms only, same lookup table).

## C. EEGSDE (Bao et al., ICLR 2023), arXiv 2209.15408

C1. Section 4 / Appendix: QM9 training set split into two non-overlapping halves Da, Db;
    the noise-prediction network (conditional EDM) and the energy function trained on Db.
C2. Appendix F.3: noise-prediction network trained ~2000 epochs, batch size 64, lr 1e-4,
    Adam, EMA 0.9999 ("the same setting with EDM").
C3. Appendix G.2: novelty / atom stability / molecule stability on 10K generated molecules
    using EDM's official implementation; novelty evaluated against Db.
C4. Table 8, "Conditional EDM" rows (Novelty / AS / MS):
      Cv    83.64+-0.30  98.25+-0.02  80.82+-0.32
      mu    83.93+-0.11  98.17+-0.04  80.25+-0.40
      gap   83.93+-0.45  98.30+-0.04  81.95+-0.27
      HOMO  84.35+-0.31  98.17+-0.07  79.61+-0.32
      alpha 84.56+-0.47  98.13+-0.04  79.33+-0.30
      LUMO  84.62+-0.28  98.26+-0.04  81.34+-0.29
C5. EEGSDE models hydrogens (it uses EDM's QM9 setting with H). Confirm or mark as
    inferred-only if the paper does not state it explicitly.

## D. EDM (Hoogeboom et al., ICML 2022), arXiv 2203.17003

D1. Table 1 (QM9, 3 runs x 10000 samples), Atom stable / Mol stable:
      E-NF 85.0/4.9; G-Schnet 95.7/68.1; GDM 97.0/63.2; GDM-aug 97.6/71.6;
      EDM 98.7+-0.1/82.0+-0.4; Data 99.0/95.2.
D2. Table 10 (feature-scaling ablation):
      [x, 1.00 h_onehot, 1.0 h_charge]  NLL -103.4   atom 95.7   mol 46.9
      [x, 0.25 h_onehot, 0.1 h_charge]  NLL -110.7   atom 98.7   mol 82.0
    with the sentence "it seems that it is easier to learn a denoising process where the
    atom type is decided later, when the atom coordinates are already relatively well
    defined."
D3. Conditional setting: QM9 training partition split into Da, Db of 50K each; classifier
    on Da, conditional EDM on Db.
D4. EDM QM9 training recipe: 3000 epochs (per their README's command; the paper may not
    state it -- report what the PDF says, if anything), batch 64, 1000 diffusion steps.

## E. Other papers

E1. TFG (Ye et al., NeurIPS 2024), arXiv 2409.15761, appendix: second half trains the
    diffusion model (unconditional EDM) and guidance network; DDIM sampler with 100 steps;
    metrics are MAE and RDKit validity only; 4096 molecules per property; NO stability.
E2. PropMolFlow (arXiv 2505.21469): generator trained on 50k of the 100k training set,
    explicit hydrogens, 2000 epochs, batch 128, 8 blocks; stability computed with a REVISED
    formal-charge-aware rule on explicitly generated bonds, which they say differs from
    Hoogeboom's and that Hoogeboom's "inflates" stability; main-text results are box plots.
E3. GeoLDM (Xu et al., ICML 2023), arXiv 2305.01140, Table 1 (QM9, 10000 samples, 3 runs),
    Atom Sta / Mol Sta / Valid / Valid&Unique:
      Data 99.0/95.2/97.7/97.7; ENF 85.0/4.9/40.2/39.4; G-Schnet 95.7/68.1/85.5/80.3;
      GDM 97.0/63.2/-/-; GDM-AUG 97.6/71.6/90.4/89.5; EDM 98.7/82.0/91.9/90.7;
      EDM-Bridge 98.8/84.6/92.0/90.7; GraphLDM 97.2/70.5/83.6/82.7;
      GraphLDM-aug 97.9/78.7/90.5/89.5; GeoLDM 98.9+-0.1/89.4+-0.5/93.8+-0.4/92.7+-0.5.
    Conditional protocol: two 50K halves, predictor on first, conditional model on second.

## F. Facts about our model

F1. `data/qm9.pt` `feats` are plain {0,1} one-hot (row sums exactly 1 on real atoms) and
    `proj1/scripts/train_fm.py` uses them unscaled (no multiplication by 0.25 or similar).
F2. Our training budget: 1500 epochs x ceil(51527/256)=202 steps = ~302k gradient steps.
    EEGSDE's conditional EDM: 2000 epochs x ceil(50000/64)=782 = ~1.56M steps (5.2x ours).
    TFG-Flow: 1200 epochs x ceil(50000/256)=196 = ~235k steps (0.78x ours).
F3. `betty_pull/fm.pt`: family flow, split train_a, epoch 1300, recorded atom_stability
    0.9390, mol_stability 0.3926, hidden 256, layers 8.
