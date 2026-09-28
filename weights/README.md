# Published weights

Inference-ready checkpoints. Verified byte-identical in the weights that matter
to the full training checkpoints they came from.

| file | size | what it is |
|---|---|---|
| `fm_ema.pt` | 14.3 MB | the frozen flow-matching generator, **EMA weights only** |
| `diff_ema.pt` | 15.1 MB | the frozen VP-diffusion generator, **EMA weights only** |
| `f_A_{mu,alpha,gap}.pt` | 1.7 MB each | property **guides** — trained on `train_a`, the same half as the generator |
| `f_B_{mu,alpha,gap}.pt` | 1.7 MB each | property **evaluators** — trained on `train_b`, disjoint from everything else |

## fm_ema.pt

Stripped from `fm_last.pt` (57.5 MB) by dropping optimiser and scheduler state,
which are only needed to *resume training*. The EMA weights are what sampling
uses. Verified: `max |w_slim - w_full| = 0.0` across every parameter.

- source md5 (full checkpoint): `a190ac8394902027d4a951f8d30e8c5c`
- EGNN, hidden 256, 8 layers; trained on `train_a` (51,527 molecules), 1500 epochs, EMA 0.9999, seed 20260918
- `family = "flow"`, epoch 1500

To resume *training* you need the full checkpoint, which is not in this repo —
it is on the cluster at `proj1/checkpoints/fm_last.pt`.

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

| property | f_A val MAE | f_B val MAE | δ = 2 × f_B MAE |
|---|---|---|---|
| `mu` | 0.08971 D | 0.08399 D | 0.16799 D |
| `alpha` | 0.24474 Bohr³ | 0.24068 Bohr³ | 0.48135 Bohr³ |
| `gap` | 0.00390 Ha | 0.00380 Ha | 0.00760 Ha |

All six are EGNN, hidden 128, 4 layers, 120 epochs. See
[docs/protocol/SPLIT_PROTOCOL.md](../docs/protocol/SPLIT_PROTOCOL.md).

## Loading

```python
import sys; sys.path.insert(0, "proj1/src"); sys.path.insert(0, "proj1/scripts")
from m1_signed_bias import load_fm, PhysicalProperty

net, ck = load_fm("weights/fm_ema.pt", n_types=5, dev="cuda")
f_A = PhysicalProperty("weights/f_A_mu.pt", n_types=5, dev="cuda")
```

The data file `data/qm9.pt` is **not** in this repo (430 MB). Rebuild it with
`python proj1/scripts/prepare_qm9.py` — the split is seeded (20260917) and
reproducible.
