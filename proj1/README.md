# proj1/ — the code

Everything that runs lives here. The project overview, the reproduction steps
and the map from each paper table to the script that produced it are in the
[root README](../README.md); new teammates should start with
[ONBOARDING.md](../ONBOARDING.md).

```
src/           the library: guidance.py (every guidance arm, including BDG),
               sampling.py, diffusion.py, evaluation.py, dist_metrics.py,
               models/ (EGNN generator, property predictors), external/
               (borrowed backends: TFG, EquiFM, the EDM schedule)
scripts/       entry points: train_fm.py, train_predictor.py, guidance_sweep.py,
               transfer_sweep.py, benchmark_base.py, and one *_table.py per
               results table
tests/         closed-form gates; run them before any sweep, they take seconds
cluster/       SLURM job scripts for PARCC Betty (see docs/protocol/BETTY_RUNBOOK.md)
m2/            Modality 2: DNA enhancer sequences on the probability simplex
checkpoints/   training histories (tracked); the .pt files are ignored, and the
               published weights are in ../weights/
logs/          local training logs (ignored)
```

Most scripts find `data/`, `weights/`, `results/` and `betty_pull/` from their
own location (`ROOT` = two directories above the script), so **those folders
must stay at the repository root**. A few defaults are relative to the working
directory instead (e.g. `src/schnet_ref.py` uses `scratch_schnet/`), so run
everything from the root.
