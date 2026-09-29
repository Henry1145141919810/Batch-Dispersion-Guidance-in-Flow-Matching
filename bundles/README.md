# bundles/ — tarballs shipped to and pulled from Betty

Everything in this folder except this README is gitignored. Until 27 Sep 2026
these files piled up in the project root; they were grouped here then, and
`docs/protocol/BETTY_RUNBOOK.md` now builds and pulls straight into this folder.

| folder | what it holds |
|---|---|
| `code/` | every code tarball shipped to Betty (runbook Step 1): `code_v2` … `code_v19`, plus the earlier `code_update*.tgz` and the one-off `code_v15_basecmp.tgz`. **Names are never reused**, so the highest `N` here is the last one taken. The last one was `code_v19`, so **the next is `code_v20`**. `code_v8.tgz` was deleted on 27 Sep because its md5 (`3bcffbfc…`) matched `code_v7.tgz`. |
| `results/` | tarballs pulled back from Betty (runbook Step 5): `cells_*.tgz`, `v2_cells.tgz`, `v2_screen.tgz`, `new_results.tgz`, `full_v1.tgz`, and `pull_0927.tgz` (27 Sep: Betty's `results/` and `logs/` trees, 5,451 entries; the source of the v3 Betty headline, `tune/`, `basecmp/` and `transfer_equifm/` cells; moved here from the root on 28 Sep). `v2_cells.tgz` is the **only local copy** of the untouched v2 cells, from before `strip_sidecar_coords.py` removed their coordinates, so do not delete it. The other copy is on Betty under `$PROJ/results/`. `cells_v2.tgz` was deleted on 27 Sep because its md5 (`96109f01…`) matched `cells_all.tgz`. |
| `assets/` | third-party model assets built for Betty: `tfg_assets_v1.tgz`, `fm_transfer_assets_v1.tgz` (both rebuildable with `proj1/scripts/fetch_tfg_assets.py` / `fetch_equifm_assets.py`), and the 18–19 Sep `cgm_*.tar.gz` bundles. |
| `packages/` | packages built for teammates: `bobo_v3_benchmark.zip` (26 Sep handoff to Bobo), and the 20 Sep `fm_qm9_training.zip` / `diffusion_qm9_training.zip`, which were tracked at the root by mistake until 27 Sep (they remain in git history). |

Extract pulled results **from the project root** so their `results/...` paths
land in the tree, e.g. `tar xzf bundles/results/cells.tgz --skip-old-files`.
