# DATA INDEX — what the paper uses, and what it does not

**28 September 2026.** Hand-maintained. One page: every number in the paper
traces to something named in §1–§4 here. Everything in §5 is kept for the
record and **must not** be cited.

The rule this page enforces, from
[SCOPE_FM_GUIDANCE_STATUS.md](../status/SCOPE_FM_GUIDANCE_STATUS.md): *anything
that describes a **plan** is superseded; anything that describes a
**measurement** still stands.* A superseded run is not a wrong run — it is a
run answering a question the protocol later changed.

---

## 1. THE CANONICAL RUN — v3-final

Everything the paper reports about molecules comes from here.

| | |
|---|---|
| **cells** | `results/v3/<backend>/v3/n2000/seed{20261001,20261002,20261003}/tr__*.json` |
| **headline** | **162 cells** = 63 `fm` + 63 `equifm` + 18 `edm` + 18 `vp` |
| **ablation** | **612 cells** = `results/v3/{fm,equifm}/v3abl/n2000/seed*/`, w ∈ {1,4}. `edm` and `vp` sit out by design |
| **protocol** | [FULL_RUN_V3_PROTOCOL.md](../protocol/FULL_RUN_V3_PROTOCOL.md) — §1.3 is the 28 Sep `vp` amendment |
| **pinned axes** | n = 2000 in 4 batches of 500 · q50 · w = 1 · t ≥ 0.5 · 100-step Euler · uniform time spacing (recorded as `flow`/`native`/`uniform` per backend) · clip 1.0 · δ = 2 × MAE(f_B) |
| **integrity** | `python proj1/scripts/v3_sanity.py` → **0 fail, 17 warn, 15,243 checks** |

### Modality 2 — the DNA cells

This file listed none of these until an audit noticed, and three of them are read
directly by `paper/tools/build_results.py`.

| tree | cells | what reads it |
|---|---|---|
| `results/m2/m2/n2000/` | 84 | `tab:m2-late`, `tab:full-m2` — the registered t ≥ 0.5 grid |
| `results/m2/m2win/n2000/` | 100 | `tab:m2`, `tab:m2-window`, `tab:full-m2-early`, and panel (b) of `fig:results` — the t ≥ 0.3 window both properties now use |
| `results/m2/m2tau/n2000/` | 5 | `tab:setpoint` — the setpoint sign probe |
| `results/m2/m2abl/n2000/` | 111 | the M2 ablation grid; no paper table |
| `results/m2/m2wsweep/n2000/` | 4 | the strength sweep that chose w; no paper table |
| the loose `results/*.json` | 22 files | `m2_gate_scores.json`, `m2_dfb_activity.json`, `m2_share.json`, `v3_batch_memory.json` and friends. Small, and four documented commands fail without them, so they ship |

### The four base models — which is whose

| backend | generator | ours? | property pair | arms |
|---|---|---|---|---|
| `fm` | flow-matching EGNN | **ours** | ours | all 7 |
| **`vp`** | **VP diffusion EGNN** | **ours** | ours | unguided, plug |
| `edm` | TFG's released EDMsecond | borrowed | ours | unguided, plug |
| `equifm` | EquiFM (Song et al.) | borrowed | **TFG's** | all 7 |

⚠️ **`edm` and `vp` are both diffusion and only `vp` is ours.** This is the
single most repeated mistake in the project's history. If a table, caption or
slide calls `edm` ours, that is a bug.

⚠️ **Never compare `in_band` against `equifm`.** It uses TFG's pair, so its
acceptance band is a different width — ours is wider on all three properties
([V3_PAIR_DELTA.md](V3_PAIR_DELTA.md)). `fm`, `vp` and `edm` share our pair and
are mutually comparable. `v3_table.py` refuses to pool across pairs; do not do
by hand what it refuses.

---

## 2. THE RESULTS PAGES — start here

| page | what it holds |
|---|---|
| **[V3_FINAL_SUMMARY.md](V3_FINAL_SUMMARY.md)** | **the one-stop readout.** §1 `edm` · **§1b `vp`, including the matched `fm`-vs-`vp` table** · §2 `fm` · §3 `equifm` · §4 Modality 2 · §5 across the run · §6 self-consistency |
| **[PAPER_TABLE_FILL_V3.md](PAPER_TABLE_FILL_V3.md)** | **every pending paper cell → the number that fills it, with ready-to-paste LaTeX.** Read §1's census first |
| [V3_RESULTS.md](V3_RESULTS.md) | all four backends side by side |
| [V3_RESULTS_fm.md](V3_RESULTS_fm.md) · [_vp](V3_RESULTS_vp.md) · [_edm](V3_RESULTS_edm.md) · [_equifm](V3_RESULTS_equifm.md) | per backend, full metric block |
| [V3_RESULTS_ABL_w1.md](V3_RESULTS_ABL_w1.md) · [_w4](V3_RESULTS_ABL_w4.md) | the 17-arm BDG grid |
| [V3_PAIR_DELTA.md](V3_PAIR_DELTA.md) | why in-band does not cross pairs |
| [V3_POWER.md](V3_POWER.md) | MDD per n; what the run can and cannot detect |
| [M2_V3_RESULTS.md](M2_V3_RESULTS.md) | Modality 2 (DeepFlyBrain enhancers) |
| **[M2_WINDOW_COMPARISON.md](M2_WINDOW_COMPARISON.md)** | **the same M2 arms at both guidance windows (29 Sep), plus the controller state. BDG's sign against plug FLIPS on `cpg`: +16.83 pp at t>=0.5, -4.88 pp at t>=0.3 — because plug already sits tighter than the setpoint there, so BDG widens** |

All of the above are **script-generated**. Do not hand-edit them — fix the
script and re-run. Commands are in each file's header; §6 below collects them.

---

## 3. THE HEADLINE RESULT ADDED 28 SEPTEMBER

**The matched flow-vs-diffusion comparison, which the project lacked until now.**
`fm` and `vp` share the EGNNVelocity backbone, 3,753,229 parameters, 1500
epochs, batch 256, EMA 0.9999, the `train_a` split and training seed 20260918.
The generator family differs — and so does checkpoint selection: `fm` is the
epoch-1500 export, `vp` the **selected** epoch 1475 (val loss 0.20784), which
`paper/main.tex:255` discloses.

| unguided, 3 seeds | `fm` (flow) | `vp` (diffusion) | Δ |
|---|---|---|---|
| molecule stability | **0.3970 ±0.0061** | 0.2883 ±0.0134 | −0.1087 |
| validity | **0.7562 ±0.0108** | 0.6617 ±0.0070 | −0.0945 |
| atom stability | **0.9356 ±0.0017** | 0.9070 ±0.0013 | −0.0286 |

Every gap is **7–14× the sd of the difference** (√(s₁²+s₂²): 7.4 / 7.4 / 13.5),
and at least 8× the larger single-arm sd. **Flow matching wins on all three**,
which is the evidence the base-model choice previously did not have.

`tab:fmvd` row, ready to paste (from `PAPER_TABLE_FILL_V3.md` §2):

```latex
VP, trained here & 0.288 $\pm$ 0.013 & 0.662 $\pm$ 0.007 & 0.999 $\pm$ 0.001 & 101 & 48.3 $\pm$ 0.1\\
```

⚠️ **Two things to disclose with it.**
1. **`s/sample` is a B200 MIG number**; the `fm`, `equifm` and `edm` rows were
   all timed on an RTX A6000. The paper's caption already footnotes this
   (`paper/main.tex:133`), so no elapsed-seconds claim may be made across the
   two rows.
2. **`plug` on `vp` binds the clip on 6.1–7.5 %** of guided sample-steps
   against M1's under 1 %. Real and mechanistic — the score carries a 1/σ
   factor near the noisy end — and `v3_sanity` reports it as a warning.

---

## 4. BASE DATA, CHECKPOINTS AND FEATURES

| artifact | what | provenance |
|---|---|---|
| `data/qm9.pt` | QM9, split seeded 20260917. **Not in the repo** (142 MB; all of `data/` is 430 MB) | rebuild: `python proj1/scripts/prepare_qm9.py` |
| `weights/fm_ema.pt` | our flow generator, EMA, epoch 1500 | ours. `train_a`, seed 20260918 |
| **`weights/diff_ema.pt`** | **our VP diffusion generator, EMA, selected epoch 1475 of 1500** | **ours. md5 `8a3390a6`. Trained by Bobo, published 28 Sep** |
| `weights/f_{A,B}_{mu,alpha,gap}.pt` | our guide / oracle pair, **disjoint halves** | ours. Never score with `f_A` |
| `weights/EDMsecond/` | TFG's released EDM checkpoint | **borrowed**, md5 `6abbd010`. Fetched, not committed |
| `audit/fa_fb_search/TFG/` | TFG model defs + predictor weights | **borrowed**, unmodified |
| `proj1/m2/blade_bundle/fm_m2_dfb500.pt` | Modality 2 generator | ours |

**Splits** ([SPLIT_PROTOCOL.md](../protocol/SPLIT_PROTOCOL.md)): generators on
`train_a` (51,527); `f_A` on `train_a`, `f_B` on `train_b` — disjoint **by
construction**, which is what makes δ honest.

**Protocols that govern**

| doc | governs |
|---|---|
| [FULL_RUN_V3_PROTOCOL.md](../protocol/FULL_RUN_V3_PROTOCOL.md) | the headline run. §1.3 = the `vp` amendment |
| [ABLATION_V3_PROTOCOL.md](../protocol/ABLATION_V3_PROTOCOL.md) | the 17-arm BDG grid |
| [SPLIT_PROTOCOL.md](../protocol/SPLIT_PROTOCOL.md) | every split, single source of truth |
| [MODALITY2_V3_PROTOCOL.md](../protocol/MODALITY2_V3_PROTOCOL.md) | Modality 2 (supersedes `MODALITY2_V3_PLAN.md`) |
| [BETTY_RUNBOOK.md](../protocol/BETTY_RUNBOOK.md) | how to run anything on the cluster |

---

## 5. DOWN-RANKED — kept for the record, cite none of it

Each row says what replaced it. None of these is a finding for the paper or the
slides.

### 5.1 Superseded runs

| tree | why | replaced by |
|---|---|---|
| `results/v3/{fm,equifm}/n5000/` | the 26–27 Sep Betty run; `fm` cells were scored with **TFG's** pair, which v3 no longer declares for `fm`. ⚠️ **Still load-bearing:** it holds the only per-molecule sidecars for `fm`/`equifm`, so every paired contrast and useful-yield number reads it (`paper_fill_v3.py`, `v3_blade_readout.py`, `V3_BETTY_PAIRED.md`). Superseded for the headline only | the 28 Sep blade run at n = 2000 |
| `results/basecmp/` | **cancelled 26 Sep** part-way through; different target, strength rule and arm set | protocol v3 itself |
| `results/transfer/` | EDMsecond transfer, 216 cells; backend dropped 23 Sep | the `equifm` backend |
| `results/transfer_equifm/` | array **cancelled** mid-run, 26 of 63 cells | the `equifm` backend at n = 2000 × 3 seeds |
| `results/full/`, `results/sweep*/` | v2-era protocol: different δ, target rule and arm set (mixed q50/q90/`dist` targets) | v3 |
| `results/tune/` | q90, n = 1000, one seed; its stage 2 never ran | nothing — the table it fed does not exist |
| `results/bdg_port/`, `results/bdg_local/` | the independent BDG port, n = 256, one seed | the v3 ablation. ⚠️ one **figure** still reads `results/bdg_port/table.txt` — do not orphan it |
| `results/v3/*/v3/n512/` | empty directories, 0 cells | nothing; dead paths |
| `results/pilot_*`, `results/btvg_*`, `results/update_rule*`, `results/xproj/`, `results/qbench/`, `results/force_share/`, `results/_ladder/`, `results/_memprobe/`, `results/variance_decomposition/`, `results/dist_checks/`, `results/workshop_claim_audit/` | exploratory probes, never paper data | — |

### 5.2 Superseded docs

| doc | why |
|---|---|
| [BASE_MODEL_BENCHMARK.md](BASE_MODEL_BENCHMARK.md) | FM-only, built on `fm.pt` **epoch 1300**; superseded by the epoch-1500 EMA and by §3's matched row |
| [BASECMP_PARTIAL_RESULTS.md](BASECMP_PARTIAL_RESULTS.md) | the cancelled chain. Its own first line says nothing in it is a finding |
| [TUNE_N1000_RESULTS.md](TUNE_N1000_RESULTS.md) | q90; the paper is q50 |
| [FULL_RUN_RESULTS.md](FULL_RUN_RESULTS.md), [FULL_RUN_TABLE.md](FULL_RUN_TABLE.md), [FULL_RUN_TABLE_WITH_TFG.md](FULL_RUN_TABLE_WITH_TFG.md), [FULL_RUN_V2_RESULTS.md](FULL_RUN_V2_RESULTS.md), [ARM_RANKINGS.md](ARM_RANKINGS.md), [FULL_METRICS_Q50_Q90_WITH_BDG.md](FULL_METRICS_Q50_Q90_WITH_BDG.md) | v2 / q90 era |
| [V3_BETTY_HEADLINE.md](V3_BETTY_HEADLINE.md), [V3_BETTY_PAIRED.md](V3_BETTY_PAIRED.md) | the n = 5000 Betty run. ⚠️ `V3_BETTY_PAIRED.md` holds the **only** paired / useful-yield reading of v3 anywhere — superseded for the headline, but not yet replaced on that one analysis |
| [BDG_LADDER_MEASURED.md](BDG_LADDER_MEASURED.md) | its closed form `w_eff = 1 + η(1/τ² − 1)` is **withdrawn** |
| [PROJECT_CLAIMS_VERIFICATION.md](PROJECT_CLAIMS_VERIFICATION.md) | a pre-implementation audit; describes a repo that no longer exists |
| `docs/protocol/{BASECMP_PROTOCOL,TUNE_SWEEP_PLAN,FULL_RUN_V2_PROTOCOL,MODALITY2_V3_PLAN,TRANSFER_EXPERIMENT_PLAN}.md` | protocols for the runs above |
| `docs/protocol/BOBO_VP_DIFFUSION_HANDOFF.md` | **executed 28 Sep.** Archival; its commands remain valid |

---

## 6. REGENERATING EVERY GENERATED PAGE

Run in this order — later ones cross-check earlier ones and refuse on a
mismatch.

```bash
python proj1/scripts/v3_table.py --both --stage v3 --n 2000 \
    --seeds 20261001,20261002,20261003 --md-out docs/results/V3_RESULTS.md
for BE in fm equifm edm vp; do
  python proj1/scripts/v3_table.py --backend "$BE" --stage v3 --n 2000 \
      --seeds 20261001,20261002,20261003 --md-out "docs/results/V3_RESULTS_$BE.md"
done
for W in 1 4; do
  python proj1/scripts/v3_table.py --both --stage v3abl --n 2000 --w "$W" \
      --seeds 20261001,20261002,20261003 --md-out "docs/results/V3_RESULTS_ABL_w$W.md"
done
python proj1/scripts/v3_sanity.py
python proj1/scripts/v3_final_summary.py --md-out docs/results/V3_FINAL_SUMMARY.md
python proj1/scripts/paper_fill_v3.py  --md-out docs/results/PAPER_TABLE_FILL_V3.md --latex-check
python proj1/scripts/v3_power.py       --md-out docs/results/V3_POWER.md
python proj1/tests/test_v3.py          # ALL PASS (106 gates)
```

To re-run the `vp` cells themselves: `bash proj1/cluster/submit_vp_bench.sh` on
Betty (gate → 9-task array → table). Its gate pins the checkpoint md5
(`proj1/cluster/vp_bench.slurm`, `VP_MD5`).

---

## 7. STILL OPEN

0. ✅ **RESOLVED 29–30 Sep. `tab:m2` is now built wholly at t ≥ 0.3.** Kept below
   because the reasoning still explains the CpG result. `build_results.py` builds
   `tab:m2` from `results/m2/m2win`, keeps the registered t ≥ 0.5 grid as
   `tab:m2-late`, and adds `tab:m2-window` and `tab:setpoint`. The paper has been
   rebuilt against these cells and reports that BDG loses on CpG. The original
   entry, for the record:

   ⚠️ **`tab:m2` currently mixes windows, and the data to fix it now exists.**
   The table pairs a `gc` column at t >= 0.3 with a `cpg` column at t >= 0.5 —
   its caption says so — and that pairing is what produces CpG's +16.83 pp for
   BDG. Bobo's 29 Sep suite supplies `cpg` at t >= 0.3, where the same contrast
   is **-4.88 pp**. Run at one window, BDG wins on `gc` and loses on `cpg`.
   See [M2_WINDOW_COMPARISON.md](M2_WINDOW_COMPARISON.md); the paper has not
   been rebuilt against these cells.

   **Why, measured (29 Sep, corrected):** on `cpg` BDG is **widening**, not
   failing. With the dispersion term off, plug's own spread sits at **0.37·tau**
   on `cpg` — already tighter than the tightest setpoint the pre-registered grid
   reaches — so the feedback correctly pulls back out, and in-band punishes it.
   On `gc` plug sits at **0.62·tau**, looser, so BDG contracts and gains. Same
   controller, opposite side of the setpoint. A setpoint below 0.37·s reverses
   the sign (`tau_mult = 0.25` gives **+2.28 pp**) — a diagnostic **outside** the
   pre-registration that must never be quoted as the headline. **The grid does
   not reach `cpg`'s regime; the controller does not fail.**


1. **The paper's `tab:fmvd` is already filled** (`paper/main.tex:140`:
   `VP diffusion, ours & $28.8\pm1.3$ & $66.2\pm0.7$ & 99.9 & 101 & 0.048$^{*}$`),
   and `:129` / `:255` carry the prose and the selection disclosure. The
   remaining pending cells and the per-table map stay in
   `PAPER_TABLE_FILL_V3.md`.
2. **`slides/defense.tex` is not corrected.** It carries `\TODO{v3 Job A}` at
   L327, L397 and L451, `\TODO{v3 A}` in the VP row at L438, and further
   `\TODO{v3…}` slots at L366, 530, 714–715, 783–790, 844, 881, 911–914. Its
   prose still says the run is pending. The handoff docs under
   `slides/handoff/` **are** corrected, so a speaker following
   `07_DO_NOT_SAY.md` would contradict the slide behind them — fix the deck
   before rehearsing. `slides/handoff/08_PLACEHOLDERS.md` lists the slots.
3. **`vp`'s `s/sample` is B200, not A6000** (§3) — **already disclosed** in
   `tab:fmvd`'s caption (`paper/main.tex:133`, "RTX A6000 except $^*$B200 MIG;
   runtimes are not hardware matched") and in `:299`. Re-time on an A6000 only
   if an exactly comparable number is wanted.
4. **Useful yield** for `fm`/`equifm` needs the blade sidecars. `vp`'s
   `*.permol.pt` sidecars **are** in the tree, so its yield can be computed now.
5. **`benchmark_base.py` cannot load either published checkpoint** —
   `is_last = "model" in ck and "state_dict" not in ck` misses the slim
   EMA-only layout, so it asks for `ema_state_dict` and raises on both
   `fm_ema.pt` and `diff_ema.pt`. `load_fm` handles both; this does not.
