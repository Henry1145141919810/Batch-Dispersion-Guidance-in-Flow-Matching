# Paper versions

`v1_*` and `v2_*` are the `.tex` sources and the built PDF at each point. The live
files are `paper/body.tex` and `paper/main.tex`.

## v2 — 26 Sep 2026

Three adversarial passes were run against v1: a rubric grader, a hostile reviewer
checking every claim against the repository's own result files, and an audit against
the lab writing guide. A fourth pass verified the five unattributed citations.

### Claims v1 made that its own data contradicted

Each of these was verified against the file named before it was changed.

| v1 said | The data says | Source |
|---|---|---|
| ladder reaches `1.219` / `1.22`, on "six of six curves" | `1.219` is **mu at q90**; this study reports q50 only, where the range is `0.670`–`1.187` over **three** properties | `results/bdg_port/table.txt` |
| "No rung beats plug-in at \|z\|>=3: the largest is 3.07" | the max is **-3.07**, a BDG **loss** (gap q50, e4t1.5); the sentence also refuted itself, since 3.07 >= 3 | `docs/results/BDG_FULL_METRICS.md:549` |
| "more chemistry at matched coverage" as the headline win | `z = 1.73`, `p = 0.083`, one seed; and **both** cells fail the project's own chemistry floor of 0.390 | `results/bdg_port/table.txt` floor column |
| clip counts `884` vs `2,516` | that is a different run (n=128, seed 20261001) from the cells it is said to confound; the matched pair is **701** vs **4,457** | `docs/results/ONEGEN_FULL_METRICS.md` |
| batches sit "one or two" band widths off, spread "five to nine" | `0.55`–`3.03` and `5.95`–`16.37` | `BDG_FULL_METRICS.md` spread-vs-bias lever |
| Real QM9 `0.9940 / 0.9560 / 0.9820` | `0.994 / 0.956 / 0.982`; the fourth digit was introduced by v1's column-alignment pass and is invented | `docs/results/BASE_MODEL_BENCHMARK.md:215` |
| guidance raises in-band `1.9x` on gap, TFG `6.0x` | `1.81x` and `5.94x` | Table 2's own printed values |
| Modality 2 "every signature reproduces" | the widening rung reaches `1.006`–`1.008`, inside the `1.010` ratio the team's own review calls indistinguishable from zero | `docs/methods/BDG_REVIEW.md:210` |
| reduction verified to `6.5e-7` | the cited check reports `6e-7` | `BDG_REVIEW.md:62` |

### Also corrected

- Figure 1 coloured the diffusion model as **external pretrained**. We trained it.
- Three-seed error bars were present in the original Table 1 and were dropped by v1's
  alignment pass. Restored.
- `\pend` pointed at Appendix A.6, which is Algorithms. Now A.7.
- Table 3's caption claimed an ordering by `w_eff`, a column the table does not have,
  and never named its property (mu).
- BDG's own row carried Status "reproduced".
- Eight floats, including four of the five tables and Algorithm 1, were never
  referenced from the text. `tools/check.py` now fails the build if that recurs.
- Five citations had no authors. All five verified against arXiv; **two had wrong or
  truncated titles**. `CFG-Ctrl`'s recorded subtitle does not exist, and
  `ensemblevar2025` is a climate-downscaling paper that sets spread by DDIM step count,
  so it no longer sits in the near-our-mechanism sentence.
- Guide-audit fixes: two rhetorical questions, personified methods, present-tense
  results, "demonstrate" without a test, a claim-colon-slogan pattern, and a citation
  bundle attached to no claim.

### Scope and disclosures added

- The chemistry floor is now defined in the protocol and reported in Table 3.
- Spread ratios are disclosed as scored by the guide `f_A`, in-band by held-out `f_B`.
- Modality 2 inherits the `t >= 0.5` window, where GC batch spread is nearly flat; that
  is now stated rather than silent.
- Table 4's pass counts are defined as network calls at batch granularity, and the text
  no longer implies a cost comparison the generator column contradicts.
- The reduction now carries the answer to "then this is a reparameterisation".

### Cost

Main text went 5.00 -> 5.65 pages under these additions and was brought back to 5.00 by
cutting ~330 words of prose that the appendix or another section already carried. No
rubric line item lost its evidence; `tools/check.py` verifies the structural ones.

## v1 — 26 Sep 2026

First complete draft. Five-page main text, five tables, one figure, appendix A.1–A.7.
