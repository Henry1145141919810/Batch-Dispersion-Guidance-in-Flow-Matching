# Read this first — Modality 2 sweep

The base model is TRAINED and verified. This phase runs the **guidance sweep**.

**Job: run one command, report the table, stop.**

```bash
python proj1/m2/run_sweep.py --device cuda --out-dir results/m2_dfb
```

300 cells, resumable (re-run to continue; finished cells are skipped).
Run from the REPO ROOT, not from this directory.

---

## Do not re-derive these

The code in `proj1/m2/` is canonical. This directory holds **only data and the
checkpoint** — earlier it also held copies of `simplex_fm.py` and `m2_sweep.py`,
which were deleted because a stale copy is exactly how the bugs below survived.

- `dfb500.npz` — DeepFlyBrain official split, uint8 base indices, 12 MB.
  `load_dfb()` falls back to it automatically when the 554 MB PARCC pickle is
  absent. Verified identical: GC 0.45523/0.05521, CpG 0.04708/0.01475.
- `fm_m2_dfb500.pt` — the frozen base. 1500 epochs, 126.0M sample-views,
  val 0.06336. **Never retrain or fine-tune it.**
- `../vf_table.json` — Var(f(x1)|x_t) MEASURED on the held-out split.

## Settings that are load-bearing, with the reason

Each of these was a bug that produced a complete but meaningless results table.
Do not "tidy" them back.

| setting | why |
|---|---|
| `t_min = 0.0` | The property is decided by t~0.5. M1's `t_min=0.5` steers AFTER the decision: 8.7% of the gap closed vs 75% at 0. |
| `NFE = 400` | `gc_soft` (guided) and `gc_hard` (scored) differ 5.3% at NFE 100, 2.7% at 400. Confirmed over 3 runs. |
| `n = 1024`, 3 seeds | in_band is a proportion; se = sqrt(p(1-p)/n) = 0.021 at n=512, but arms separate by ~0.03. n=512 cannot resolve its own metric. |
| `sigma_mc = 0.35` | At 0.02, `tfg_mc` was mathematically identical to plug — a null arm from an unargued default. |
| `v_f` from the table | M1's Tweedie `k=(1-t)^2/t` overstates the conditional variance 30x–20,000x here. It is derived for Gaussian VP diffusion with unit noise; our source is Dirichlet(1), variance 3/80. |
| Dirichlet via `exponential_` | `torch.distributions.Dirichlet.sample()` takes NO generator and uses the GLOBAL RNG, so `--seed` did nothing and every cell started from different noise. |

## The two properties

- **gc** — GC content. EXACTLY AFFINE (Hessian identically zero), so the Tweedie
  second-moment corrections vanish and several M1 ablations coincide here.
- **cpg** — CpG dinucleotide density. QUADRATIC, so they do not.

The contrast is the point: it shows *when* the corrections matter. Report both;
do not drop `gc` for being "degenerate".

## Report

Per (property, arm, w, target): `in_band_fraction` (the headline), `gc_sd`
relative to unguided, `bias_delta`, `decode_conf`, `kmer_js`, `diversity`, and
the `cost` dict. Mean over the 3 seeds with a seed-to-seed standard error.

Two checks that must hold, and are worth stating in the report:
1. **`bdg` variant `e0t1` must equal `plug` at the same w bit-for-bit.** It is
   BDG with the feedback switched off. If it does not, every BDG number is void.
2. Guidance must move `in_band` well above unguided (~0.06 -> ~0.4 for gc at w=16).

## Do not

- Do not retrain, fine-tune, or swap the checkpoint.
- Do not tune any setting to improve a result. The grid is fixed in advance; a
  fidelity floor is set from the UNGUIDED cell, not after seeing arm results.
- Do not add or drop arms. `plug`=DPS, `tmpd`=TMPD/PiGDM, `lgd_mc`=LGD,
  `tfg_mc`=TFG's MC ingredient only (not full TFG), `bdg`=ours.
- Do not run on CPU if a GPU is free; CPU is ~45x slower and this is 300 cells.

## blade hazards

Shared box, no scheduler. Check `nvidia-smi` before launching and pick an idle
GPU; nothing reserves one. `nohup` survives logout but not OOM. The driver is
resumable, so an interrupted run loses at most one cell.
