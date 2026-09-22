# Internet search for f_A and f_B

Research date: 18–19 September 2026. Requirements: `REQUIREMENTS_FOR_FA_FB.md`.

**Best candidate: original TFG, NeurIPS 2024.** Public clean-data guide/evaluator weights exist for six properties. I downloaded and loaded its dipole pair and verified basic first/second derivatives, symmetry, padding and nonzero curvature. This is a credible route to reuse pretrained weights, but **not yet a certified replacement for our current pair**: exact checkpoint training IDs, physical-unit normalization, held-out MAE and error correlation still need verification. Its splits also differ from our existing experiment.

The search covered original papers, official repositories, checkpoint listings, source code and public download folders for TFG, TFG-Flow, E(3)-EDM, EEGSDE, OC-Flow, PropMolFlow, MolGuidance, SchNet, DimeNet and SchNetPack/PaiNN. Searches for newer pretrained/denoising approaches did not produce a better documented matched pair. This is a bounded search, not proof that no other suitable models exist.

## 1. Original TFG is the strongest match

The paper's molecule appendix says its evaluator is trained on the first QM9 training half and its guide on the second. It specifically distinguishes its guide from EEGSDE: training uses clean molecules with the time input fixed at zero. The experiment covers mu, alpha, gap, HOMO, LUMO and Cv, with separate models for each property. [Paper, Appendix D.7](https://arxiv.org/html/2409.15761v2#A4.SS7).

The official repository links a public model collection. I inspected the collection itself and confirmed **all six `tf_predict_*` folders contain `model_ema_2000.npy` and matching arguments**, and all six `evaluate_*` folders contain `best_checkpoint.npy` and arguments. These are actual files, not merely example paths in a README. [Official repository](https://github.com/YWolfeee/Training-Free-Guidance), [checkpoint collection](https://drive.google.com/drive/folders/15l3tsw_4FV0GdE_iXG021Wv0hDJgHzHf).

| Property | Clean guide | Evaluator |
|---|---|---|
| Dipole mu | [tf_predict_mu](https://drive.google.com/drive/folders/1ztVromBlJ0Rm8E0guH8nAmvRCUtp39F_) | [evaluate_mu](https://drive.google.com/drive/folders/1zvykGFJtgnA2WH5BMEih0q6KmPQvY7S8) |
| Polarisability alpha | [tf_predict_alpha](https://drive.google.com/drive/folders/1KgC2LhR7omFXzopaCLOpnwEsE_TXxlbM) | [evaluate_alpha](https://drive.google.com/drive/folders/1qGkJuicjoqac2zVnY0jMp1l6jyh5LlY5) |
| HOMO–LUMO gap | [tf_predict_gap](https://drive.google.com/drive/folders/1em7YdcsrhHRC-qcvZ3szU9XSKj9SUoLv) | [evaluate_gap](https://drive.google.com/drive/folders/1QajomT5sqymPC-hdVPGGYn5xOnprLt24) |
| HOMO | [tf_predict_homo](https://drive.google.com/drive/folders/1Uiv5hBgIlbNu8wFEy3iW12BVqwN2kdNw) | [evaluate_homo](https://drive.google.com/drive/folders/17D1qUOfwUbz3O-XEem1c7WMV2dCVYSaE) |
| LUMO | [tf_predict_lumo](https://drive.google.com/drive/folders/11kPU0UCTT9Ft6TrXM8gZJ70QNcA2SvHR) | [evaluate_lumo](https://drive.google.com/drive/folders/1PcVhcqX2TtZiMlz0ZxeLIg89kcQj1YGw) |
| Heat capacity Cv | [tf_predict_Cv](https://drive.google.com/drive/folders/15jD75iORsYwLn_3YSFMbtbdIIJ0kB-gL) | [evaluate_Cv](https://drive.google.com/drive/folders/1FC_MV24TslPJM14X2-KcBAgtzInVkHDe) |

These would form two bundles of property-specific networks, rather than two jointly trained multi-output networks. A wrapper can return `[B,3]` or `[B,6]` without retraining. Both sides are EGNN variants; this does **not** satisfy the preference for genuinely different architecture families.

### What was tested locally

Downloaded the mu guide at epoch 2000, the mu evaluator and their arguments. Loaded checkpoint tensors with `weights_only=True`; inspected metadata using a restricted reader. The research adapters expose raw scalar predictions, bypassing the guidance reward wrapper. Model parameters were frozen while input differentiation remained enabled.

On two real local molecules, using float64 on CPU:

| Check | TFG mu guide | TFG mu evaluator |
|---|---:|---:|
| Coordinate and feature gradients/HVPs | Finite | Finite |
| Maximum tested symmetry error, scaled by max(1, abs(output)) | 7.65e-16 | 1.94e-16 |
| Output change from finite garbage padding | 0 | 0 |
| Gradient change from finite garbage padding | 0 | 0 |
| Padded input gradients | Exactly zero | Exactly zero |
| Coincident / 1e-9-separated atom derivatives | Finite | Finite |
| Coordinate Hessian trace, first molecule | 44.9114 | 2.61475 |
| Atom-feature Hessian trace, first molecule | 18.3294 | 0.270775 |

**Limits:** these are smoke checks, not exhaustive certificates. Traces refer to raw normalized outputs and the adapter's input coordinates; they are not physical-unit curvature estimates. No validation MAE or error-correlation experiment was performed. Other properties' files were located but their weights were not tested. Float32/GPU behavior has not been certified.

Results and the reproducible check are in `audit/fa_fb_search/candidate_checks.json` and `audit/fa_fb_search/check_candidates.py`. Original TFG source revision inspected: `f8d17f3ec2f0e7377dedf7b7bc62fad15f36cb77`.

### Remaining requirements and integration details

1. **Verify training molecule IDs.** The paper reports disjointness and source supplies a deterministic split recipe, but the downloaded weights do not include the exact training-ID manifest required by our file. Reconstruct IDs with the original filtering and file order, or obtain the original split artifacts. Do not infer the training set from the saved argument alone.
2. **Restore physical units.** The mu guide arguments use coordinate/feature normalization factors `[1,8,1]`. The scalar property prediction must also be converted using its original mean and mean absolute deviation. The evaluator training code computes property normalization on its validation data. Those constants must be recovered and checked before claiming Debye outputs. Energy-property conversions also need explicit auditing. [Normalization code](https://github.com/YWolfeee/Training-Free-Guidance/blob/main/tasks/networks/qm9/utils.py).
3. **Test a common untouched holdout.** Establish mu MAE for both models and error correlation. The file's sub-0.09 D preference is not established for this guide by the checks above. Published guided-generation MAE is not the predictor's validation MAE.
4. **Use the raw property function.** `EnergyDiffusion.forward` returns negative squared target error, not the property's scalar value. For our curvature calculation, expose the underlying predictor and unnormalize it. Keep the time feature at its trained constant zero. A `condition_time=True` metadata field by itself does not establish noisy training. [Prediction/reward source](https://github.com/YWolfeee/Training-Free-Guidance/blob/main/tasks/networks/egnn/energy.py).

**Metadata trap:** the evaluator's saved arguments say `qm9_second_half`. The released training script first builds the training loader, then changes `args.dataset` to `qm9_second_half` to construct an auxiliary test loader, and saves those mutated arguments. Its default original training split is `qm9_first_half`. Thus that saved string does not establish which half trained the weights. This is consistent with the paper, but makes ID-level verification especially necessary. [Training script](https://github.com/YWolfeee/Training-Free-Guidance/blob/main/tasks/networks/qm9/property_prediction/main_qm9_prop.py).

## 2. TFG-Flow: real disjoint pair, poorer fit for our inputs

The official code selects `train_flow` for guide training and `train_classifier` for oracle training. Its loader assigns these to the first and second 50,000 rows of a seed-42 shuffle. Both checkpoint ZIPs are public. This **is** a guide/evaluator separation, not merely a generator/evaluator separation. [Training code](https://github.com/linhaowei1/TFG-Flow/blob/master/train_classifier.py), [data loader](https://github.com/linhaowei1/TFG-Flow/blob/master/dataloader/datasets/qm9.py), [weights](https://github.com/linhaowei1/TFG-Flow/tree/master/storage).

However, this released training path uses at most nine heavy atoms and removes hydrogens. Our model uses explicit H and continuously varying atom features. The original forward also accepts integer atom IDs through an embedding lookup. A linear embedding adapter, `h = feats @ embedding_weights`, recovers feature derivatives and preserves outputs at the corresponding one-hot inputs; I tested that adaptation successfully for the mu guide on heavy atoms. It does not solve the untrained hydrogen-input issue. [Predictor](https://github.com/linhaowei1/TFG-Flow/blob/master/diffusion/predictor.py).

Its checkpoint mapping records mu validation MAE about **0.2015 D / 0.1818 D**, versus our file's **0.0897 D / 0.0840 D**. These are different validation sets, but the released figures do not establish an accuracy upgrade. Six properties are supplied as separate models. [Checkpoint mapping](https://github.com/linhaowei1/TFG-Flow/blob/master/utils/env_utils.py).

**Verdict:** useful code and a genuine alternate pretrained pair for a heavy-atom experiment; inferior to original TFG as a route for our current all-atom setup.

## 3. Other useful assets and why they do not solve the full request

| Candidate | What is available | Assessment against our requirements |
|---|---|---|
| OC-Flow | Six clean EGNN property checkpoints and training logs | Useful single-side candidate/code. No separately verified complementary predictor set found in the release. Its mu log reports validation 0.04540 D and test 0.04644 D; these are release figures, not measurements on our split. [Files and logs](https://github.com/WangLuran/Guided-Flow-Matching-with-Optimal-Control/tree/main/molecule/qm9/property_prediction/outputs). |
| EEGSDE | Six guide and evaluator sets | Guide training is explicitly time-dependent/noisy, violating clean-only requirement 7. The clean evaluation networks remain useful single-side assets. [Official release](https://github.com/gracezhao1997/EEGSDE). |
| PropMolFlow GVP | Six predictor checkpoints plus published split-index files | Forward uses learned embeddings of bond features (`e_1_true`), so it is not a coordinate/type-only replacement. Its documented 50k/50k separation is generator versus evaluator, not a released clean guide/evaluator pair. [Predictor code](https://github.com/Liu-Group-UF/PropMolFlow/blob/main/propmolflow/property_regressor/gvp_regressor.py). |
| PropMolFlow's separate EGNN archive | EGNN weights, normalizer and example notebooks | More convenient packaging of clean evaluator assets. The archive explicitly identifies EEGSDE as the checkpoint source; it is not evidence of a new independent partner. [Zenodo archive](https://zenodo.org/records/17726328). |
| MolGuidance | GVP evaluators and trajectory-trained guidance code | Guide samples intermediate flow states and uses time conditioning; evaluator uses bond features. Fails our clean-only/input requirements as released. [Guide training code](https://github.com/Liu-Group-UF/MolGuidance/blob/main/molguidance/classifier/train_classifier.py). |
| PyG pretrained SchNet | Property-specific pretrained models and saved train/validation/test splits | Valuable weights and a different architecture, but no verified disjoint half-pair found. Atomic-number embedding, dipole readout and geometric cutoff behavior require adapters and derivative checks. [Official loader](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/schnet.html). |
| PyG pretrained DimeNet | Checkpoints and split reconstruction | No verified disjoint half-pair found; raw-distance/angle expressions need care at collisions. The pretrained target map inspected omits a dedicated gap model. [Official source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/dimenet.html). |
| SchNetPack / PaiNN | Inspectable training implementations | Practical different-architecture retraining route, not an identified drop-in matched pair. Atom-type differentiation and collision-safe second derivatives still need explicit design/testing. [Official project](https://github.com/atomistic-machine-learning/schnetpack). |

## 4. What this means for our project

Our `prepare_qm9.py` uses a custom PyTorch random split, not the released EDM/Cormorant split. It keeps explicit hydrogens and saves tensors without an original molecule-ID array. Consequently, we cannot assume a public predictor trained on “half QM9” is disjoint from either of our current halves. Exact overlap counts have not been computed.

Using the complete TFG pair together is the most plausible way to avoid retraining the predictors. It would require aligning the experiment with their verified split, including the generator/evaluator separation specified in `PROJECT_GUIDE.md`, and reserving evaluation molecules neither predictor trained on. Fine-tuning a public model only on our half does not erase its prior exposure to the other half.

If preserving our current trained generators and splits is the priority, reuse implementation ideas and train predictors from scratch on our saved halves. The file's stated 23-minute training time makes that a credible alternative to migrating data splits. A SchNet/PaiNN-style second model can address architecture diversity, but its accuracy on half-data and derivative behavior must be measured.

**Recommendation:** original TFG merits a focused acceptance audit before spending more time searching. It is the clearest available pretrained match. Do not mark it fully accepted until the remaining ID, units, accuracy and correlation checks pass. No existing project predictor, data split or training configuration was replaced during this research.
