# Modality 2 under protocol v3: what it would take

**26 September 2026. Nothing here has been run, and nothing here is
pre-registered** — this is a scoping document, written because
`transfer_sweep.V3_BACKENDS` points at it to explain why Modality 2 is *not* a
row in that table.

Henry's 26 Sep instruction lists four base models for the v3 benchmark. Three
are wired in (`fm`, `equifm`, `edm`). The fourth — the Modality 2
flow-matching model — is **a separate stack, not a fourth backend**, and this
document says exactly what separates them.

---

## 1. What Modality 2 already is

Better than the status docs claim. The base model is trained, validated and
**tracked in this repository**.

| | |
|---|---|
| modality | regulatory DNA — DeepFlyBrain enhancers, *Drosophila* accessibility peaks, 500 bp one-hot ACGT |
| state | the probability simplex `(Δ³)^L`, i.e. a `[B, L, 4]` tensor — **not** coords + feats + mask |
| model | dilated 1-D CNN velocity field, linear flow-matching path, Dirichlet(1) source, clamp-renormalise projection. ~1.02 M params |
| checkpoint | `proj1/m2/blade_bundle/fm_m2_dfb500.pt` (3.9 MB, **tracked**) |
| data | `proj1/m2/blade_bundle/dfb500.npz` (11.5 MB, **tracked**); train 83,722 / valid 10,505 / test 10,434 |
| training | 1500 epochs, 492,000 steps, best val 0.06336 |
| driver | `proj1/m2/run_sweep.py` → `proj1/m2/m2_sweep.py` |
| arms | `unguided, plug, tmpd, lgd_mc, tfg_mc, bdg` |

**The base is validated, not pending.** It matches the data on composition, GC
mean, GC spread (97 %) and k-mers; its 3-mer JS divergence is 24× better than
uniform; there is no evidence of memorisation.

**It has a property, a metric and an evaluator already**, which is more than
"pending" suggests:

- **f_A / f_B analogue.** `gc_soft` / `cpg_soft` (differentiable, what
  guidance steers) against `gc_hard` / `cpg_hard` (argmax-decoded, what is
  scored). GC is *exactly affine* (zero Hessian) and CpG is quadratic — the
  contrast is deliberate, because it isolates when a Tweedie second-moment
  correction can matter at all.
- **in-band.** `in_band_fraction` is computed on the **hard, decoded**
  property, so it is the analogue of M1's `in_band_fraction_dec`.
- **a full metric block**: `gc_sd`, `bias_delta`, `decode_conf`, `kmer_js`,
  `diversity`, `cost`.

---

## 2. What is genuinely outstanding

Three things, and only the third is large.

### 2.1 The results

`results/m2*` does not exist. Nothing in `docs/results/` or
`docs/protocol/` reports an M2 run. **The sweep is runnable and has not been
run** — that is the honest statement of where M2 stands.

### 2.2 Four defects in `m2_sweep.py` that must be fixed before any run

These are not v3-specific; they would corrupt any M2 result.

1. **The cell name omits the configuration.** Cells are
   `<prop>__<arm>__<target>__w<w>__<variant>__s<seed>.json` — no `n`, no
   `steps`, no `t_min`, no `clip`, no `delta_ratio`. Resume is
   skip-if-exists, so **an `--n 512` smoke run and an `--n 2000` real run
   produce identical filenames and the small cell is silently kept.** This is
   precisely the failure `transfer_sweep.config_tag` exists to prevent.
2. **There is no `--batch`.** One cell draws **one** batch of `n`, so BDG gets
   a single controller over all `n` samples. v3 runs `n ÷ 500` controllers of
   500 because the batch **is** BDG's estimator. This is a real protocol
   mismatch, not a performance difference: the two runs would be measuring
   differently-estimated controllers.
3. **The default checkpoint is the wrong model.** `m2_sweep.py` defaults
   `--ckpt` to a 200 bp model while `run_sweep.py` defaults to the 500 bp one,
   and the corpus/target reference is chosen by branching on the checkpoint's
   `crop`. A bare `m2_sweep.py` call loads the wrong model *and* the wrong
   k-mer reference.
4. **The fidelity floor is prose, not code.** A k-mer-JS floor is described in
   the file's header and implemented nowhere; `kmer_js`, `decode_conf` and
   `diversity` are recorded and never thresholded.

### 2.3 The two protocol questions that are genuinely open

**δ is a *choice* here, not a measurement.** M1's δ is `2 × f_B's MAE` — it
is the oracle's own error, so the band means "within the evaluator's
resolution". M2's `f_B` is an **exact count** over the decoded sequence, not a
model, so it *has* no error and there is nothing to read a δ off. The code
uses `max(delta_ratio × s, 4.4 × quantum)` with `--delta-ratio 0.16`. That is
defensible but it is a free parameter, and **an M2 in-band number is not
commensurable with an M1 in-band number** because the bands are set by
different kinds of rule. This has to be stated wherever the two appear
together.

**The guidance window is probably wrong at 0.5.** v3 fixes `t ≥ 0.5` for every
arm. `m2_sweep.py` defaults `--t-min 0.0`, and its own notes argue that 0.5 is
wrong *for M2*: the GC batch spread is flat after t ≈ 0.5, so guiding from 0.5
closes ≤ 17 % of the gap where guiding from 0 closes ~75 %. **Do not let M2
inherit `t ≥ 0.5` silently.** Either justify it for this modality or run M2 at
its own window and say so — but the choice must be made deliberately, because
it is the difference between a null and a result.

---

## 3. Why it is not a fourth backend in `V3_BACKENDS`

Adding a row there would mean routing M2 through `transfer_sweep.py`, and that
is a harness rewrite rather than an extension:

| | M1 (v3's three backends) | M2 |
|---|---|---|
| state | coords + feats + mask, EGNN | `[B, L, 4]` simplex, 1-D CNN |
| step | Euler/Heun on ℝ³ⁿ | Euler + clamp-renormalise **projection** each step |
| property | learned networks on disjoint halves | analytic, soft/hard relaxation of the same function |
| δ | `k × MAE(f_B)` | a chosen ratio (§2.3) |
| metrics | atom/molecule stability, RDKit validity, SMILES uniqueness | k-mer JS, decode confidence, GC spread |
| data | `data/qm9.pt` (loaded unconditionally) | `dfb500.npz` |

`transfer_sweep.py` assumes QM9 at the top of `main()`, and `evaluate_samples`
is entirely molecular. Bridging would need a new sampler class, a projection
hook in the integrator, a DNA metric block, and a non-QM9 data path —
touching most of the file for one backend.

`m2_sweep.py` also **re-implements** the BDG guidance field by hand for its
own state shape, so a fix in `proj1/src/guidance.py` does **not** propagate to
it. That is the deeper reason the two stacks are hard to merge, and it is
worth knowing before anyone tries.

---

## 4. The recommended route: extend `m2_sweep.py`, report side by side

Cheaper and more honest than a merge. Five items, in order:

1. Put the configuration into the cell name (§2.2 item 1). **Do this first** —
   without it, every later run risks silently reusing a smoke cell.
2. Add `--batch`, so BDG runs the same estimator it does in v3 (§2.2 item 2).
3. Add a stage label and a stage directory, mirroring v3's
   `results/v3/<backend>/<stage>/n<N>/seed<S>/`.
4. Write an `m2_table.py` over three seeds. Nothing currently reads
   `results/m2*`.
5. Decide the window (§2.3) and record the decision in this file before
   running anything.

Then report M2 **beside** the M1 backends, never pooled into the same table —
different modality, different property semantics, different δ rule. The
comparison that *is* meaningful is the **within-modality arm ranking**: does
the same guidance method win on DNA as on molecules? That question needs no
shared band at all, which is exactly why it is the one to ask.

---

## 5. Status of this document

Scoping only. Nothing here is pre-registered, no n is fixed, and no claim
about M2 may cite this file as evidence of a result. When M2 is actually
planned, it gets its own pre-registration in this directory, written before
the run, like `FULL_RUN_V3_PROTOCOL.md` and `ABLATION_V3_PROTOCOL.md`.
