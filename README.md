# CIS 6270 Project 1 — Group 2

Continuous generative modelling across two modalities: 3D molecules (QM9) and DNA
enhancer sequences on the probability simplex.

**Status:** empty scaffold. Structure lands in Phase 0 (12–14 Sep).

## Team

Bobo Li · Henry Huang · Idea Idehpour · Haimo Fang

## Planning documents

These live one directory up, outside the repo, until we decide to track them here:

- `PROJECT_GUIDE.md` — scope, methods, the innovation, evaluation protocol, timeline, roles
- `FEASIBILITY_TESTS.md` — the go/no-go tests to run before committing (decision due 16 Sep)

## Planned layout

```
configs/     one YAML per experiment
src/
  data/      dataset loaders and splits
  models/    EGNN, 1D CNN, prediction heads
  flows/     flow matching and diffusion
  guidance/  baseline guidance + our method   <- the innovation
  modality/  zero-CoM and simplex projections
  sampling/  Euler / Heun / DDIM with guidance hooks
  eval/      metrics, quantum chemistry, frontier plotting
scripts/     train.py  sample.py  evaluate.py  sweep.py  make_figures.py
results/     raw/  tables/  figures/
paper/
slides/
```

## Setup

TBD — `environment.yml` lands with the scaffold.

## Reproducing the paper

TBD — a table mapping each script and config to the table or figure it produces.
