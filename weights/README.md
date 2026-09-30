# Published weights

Inference-ready checkpoints. Verified byte-identical in the weights that matter
to the full training checkpoints they came from.

Paths below are the repository's. In the submission folder, `proj1/` is named
`code/` and `docs/` is `results_and_docs/`; `ln -s code proj1` at the folder
root makes the `proj1/` paths work there too.

| file | size | what it is |
|---|---|---|
| `fm_ema.pt` | 15.0 MB | the frozen flow-matching generator, **EMA weights only** |
| `diff_ema.pt` | 15.1 MB | the frozen VP-diffusion generator, **EMA weights only** |
| `f_A_{mu,alpha,gap}.pt` | 1.7 MB each | property **guides**, trained on `train_a`, the same half as the generator |
| `f_B_{mu,alpha,gap}.pt` | 1.7 MB each | property **evaluators**, trained on `train_b`, disjoint from everything else |
| `deepflybrain/DeepFlyBrain.json` | 6 KB | the published DeepFlyBrain architecture; its weights are re-fetched (below) |

## fm_ema.pt

Stripped from `fm_last.pt` (57.5 MB) by dropping optimiser and scheduler state,
which are only needed to *resume training*. The EMA weights are what sampling
uses. Verified: `max |w_slim - w_full| = 0.0` across every parameter.

- source md5 (full checkpoint): `a190ac8394902027d4a951f8d30e8c5c`
- EGNN, hidden 256, 8 layers, 3,753,229 parameters; trained on `train_a`
  (51,527 molecules), 1500 epochs, EMA 0.9999, seed 20260918
- `family = "flow"`, epoch 1500

To resume *training* you need the full checkpoint, which is not in this repo.
It is on the cluster at `proj1/checkpoints/fm_last.pt`.

## diff_ema.pt

Stripped from `diff.pt` (30.1 MB) the same way, dropping the non-EMA
`state_dict`. Verified: `max |w_slim - w_full| = 0.0` across every parameter.

- source md5: `4dc2383a7d462132025edaddb770c7ac`
- EGNN, hidden 256, 8 layers; trained on `train_a`, EMA 0.9999, lr 2e-4, batch 256
- `family = "vp_diffusion"`, **epoch 1475**, `tau_min = 0.001`
- val_loss 0.20784, atom_stability 0.9069, mol_stability 0.2969

**Take the epoch number seriously.** Training ran to epoch 1500 and there are
checkpoints named `diff_last.pt` and `diff_ep1500.pt`, but neither is the one
the reported numbers came from. `diff.pt` is the *selected* checkpoint at epoch
1475, and the matched evaluation loaded it (`logs/matched-8670133.out`:
`epoch=1475`). Its val_loss and atom_stability match the `best_loss` and
`best_stab` recorded inside `diff_last.pt`, which is the cross-check that it is
the selected one. Publishing either epoch-1500 file would ship weights that do
not reproduce the table.

Loads through the same `load_fm` as the flow model; `family` distinguishes the
two samplers.

## Why f_A and f_B are separate

`f_A` is what guidance optimises; `f_B` is what we score with. They are trained
on **disjoint halves** of the data, so a good score cannot just mean guidance
found the guide's blind spots. Never score with `f_A`.

| property | f_A val MAE | f_B val MAE | band half-width δ in the paper |
|---|---|---|---|
| `mu` | 0.08971 D | 0.08399 D | 0.17541 D |
| `alpha` | 0.24474 Bohr³ | 0.24068 Bohr³ | 0.50618 Bohr³ |
| `gap` | 0.00390 Ha | 0.00380 Ha | 0.00748 Ha |

The paper's δ is `2 × MAE_cal(f_B)`: twice `f_B`'s error on calibration
examples from `train_a`, which it never trained on (paper Appendix A.1). It is
not twice the validation MAE in the middle columns, which is why the numbers
differ.

All six are EGNN, hidden 128, 4 layers, 429,825 parameters each, 120 epochs. See
`docs/protocol/SPLIT_PROTOCOL.md`.

## Modality 2 (DeepFlyBrain enhancers)

The Modality 2 generator is not in this directory: it lives beside its code, at
`proj1/m2/blade_bundle/fm_m2_dfb500.pt` (4.1 MB, md5 prefix `7f59b60b`), with
the data split it trains and samples on, `proj1/m2/blade_bundle/dfb500.npz`
(12 MB, md5 prefix `6d2df4d0`; 83,722 / 10,505 / 10,434 train / validation /
test). Simplex flow matching, dilated CNN, hidden 128, 10 layers, 500 bp crop,
1,500 epochs, seed 20260921; validation selected epoch 1450 (loss 0.0634).
`proj1/m2/m2_sweep.py` loads both by default. Both ship in the submission folder.

`proj1/m2/enhancer_gate.pt` (0.3 MB, md5 prefix `d2d071cb`) is our
enhancer-versus-Markov discriminator from `proj1/m2/enhancer_gate.py`, a
Modality 2 diagnostic rather than a headline metric.

## Third-party weights, re-fetched rather than redistributed

| weights | fetched by | pinned in |
|---|---|---|
| EDMsecond (the borrowed QM9 diffusion backend) | `proj1/scripts/fetch_tfg_assets.py` | `tfg_manifest.json` (md5) |
| EquiFM (the borrowed QM9 flow backend) | `proj1/scripts/fetch_equifm_assets.py` | the script's `--verify` |
| `DeepFlyBrain.hdf5` | zenodo.org/record/5153337 | `proj1/m2/deepflybrain.py` (md5) |

## Loading

```python
import sys; sys.path.insert(0, "proj1/src"); sys.path.insert(0, "proj1/scripts")
from m1_signed_bias import load_fm, PhysicalProperty

net, ck = load_fm("weights/fm_ema.pt", n_types=5, dev="cuda")
f_A = PhysicalProperty("weights/f_A_mu.pt", n_types=5, dev="cuda")
```

The data file `data/qm9.pt` is **not** in this repo (430 MB). Rebuild it with
`python proj1/scripts/download_qm9.py` and then
`python proj1/scripts/prepare_qm9.py`. The split is seeded (20260917) and
reproducible.
