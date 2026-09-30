# Paper v10: 29 September 2026

Title unchanged: **BDG: Batch-Dispersion Guidance for Property-Band Generation**.
Author list unchanged from v9's edit: Bobo Li, Henry Huang.

v10 is a correctness pass over v9, not a new experiment. Two things changed the
headline DNA claim, one LaTeX error would have stopped the paper compiling at
all, and the paper was compiled and checked for the first time.

## The DNA headline result reversed, and that is now the claim

v9's Table 4 was captioned "t >= 0.3 for both properties" while its CpG column
still held the t >= 0.5 numbers. At the window the caption claims, the CpG
result reverses: plug-in 91.51 and BDG 86.63, so BDG is **-4.88 points** rather
than the +16.83 v9 reported.

The reversal is the mechanism working. BDG servos batch spread to tau, so it
contracts only a batch wider than tau. Plug-in leaves GC at 0.62s, outside the
registered tau_m = 0.5, and CpG at 0.37s, already inside it; BDG tightens GC to
0.59s and relaxes CpG to 0.45s. The sign of the coverage effect is therefore
recoverable from the setpoint alone. The abstract, section 4.7 and the
discussion now claim that rule instead of a win on both properties, which is
both honest and a stronger statement.

## The sign rule is now tested rather than asserted

v9 listed "a setpoint below 0.37s would test the sign rule" as future work. The
cells already existed, unreported, under results/m2/m2tau. At tau_m = 0.25 on
CpG, three seeds at n = 2,000: spread contracts to 0.298s and coverage reaches
93.73 percent, **+2.28 points** over plug-in, with per-seed differences +1.90,
+1.65 and +3.30, so every seed agrees in sign. Single-seed probes at tau_m =
0.25 with eta = 8, and at tau_m = 0.15, contract further.

Fidelity tracks contraction monotonically across all five rows: 3-mer JSD rises
from 3.94 to 7.55 and decoding confidence falls from 0.958 to 0.948 as sigma/s
goes from 0.454 to 0.241. That is the intended spread-occupancy exchange, not a
free gain.

These setpoints were chosen after seeing the registered result, so they are
reported as exploratory in new Table S17 and are not folded into any headline
comparison or contrast family.

## New and corrected tables

* **Table S18** (new): the complete t >= 0.3 grid, both properties by both
  strength rungs by seven arms, columns matching Table S15. Every number in the
  main-text DNA tables now traces to a cell.
* **Table 2** (new): guided versus unguided on our FM and VP. Section 4.3 had
  prose and no table.
* **Table S16** gains a CpG panel, so the -4.88 paired contrast is printed
  rather than implied. TFG-MC loses more there, -6.33.
* **Table S16**'s bold and underline are now applied per property panel. Ranked
  as one list, the global best fell in the CpG panel and the GC panel was left
  unmarked, which ranks across properties that Appendix A.9 says cannot be
  ranked.
* **Table S15**'s caption finally states its window, t >= 0.5.
* **Table 3**'s caption explains the dash in the tau_m column at eta = 0, where
  w_eff = 1 and the setpoint has no effect.

## Corrections to claims that had gone stale or were never right

* Section 4.1.2 still described a "GC-only t >= 0.3 follow-up at w = 4". The
  follow-up now covers both properties at w in {1,4}.
* The discussion claimed "GC and CpG gains extend BDG to a second state space".
  CpG is not a gain at the registered setpoint.
* Two appendix sentences said the CpG t >= 0.3 and GC tau_m = 1 follow-ups
  "was not run" and "was not supplied". Both have been run.
* The appendix said the eta = 0 control is checked to 1e-5. It is not: the gate
  is 1e-4 on continuous metrics against a measured 8.5e-6 float-noise floor,
  plus ten sequences on in-band, because that counter is quantised at 1/499 and
  single lattice sequences flip across the band edge.
* DNA file counts: the follow-up adds 105 files and 5.86 recorded hours, for
  11.12 DNA hours, not 22 files and 1.60 hours.
* Result-cell hashes: 1,073, not 995.
* The reproduction map was missing the m2tau stage.

## The paper now compiles

v9 was never compiled in this working tree; pdflatex here fails because every
ls-R database in the TeX Live install is empty and the trees are marked "!!",
which tells kpathsea to trust ls-R and never search disk. Overriding TEXMF
without the "!!" prefixes and building a pdflatex format into a user directory
fixes it without root.

The first compile failed outright: `$\\Delta$IB` in the new Table 2 header, a
double backslash that survived a raw Python string, gave "Extra }, or forgotten
$". Table 2 then overran the text block by 12.4pt and needed a tighter
tabcolsep.

Final build: **0 errors, 0 undefined references or citations, 0 overfull boxes,
0 underfull boxes**, 27 pages, main text 6 pages against a limit of 8.

tools/check.py also failed on v9-plus-edits and now passes: five uses of the
banned phrase "rather than" and one em dash had entered with the v9 edits.

## Verified against the data

Band half-widths 0.008833 and 0.008818; FM/VP stability 39.7/28.8 and validity
75.6/66.2; plug-in gains 2.75/2.35 on FM and 2.13/2.60 on VP with stability
falls 4.08/6.15 and 2.68/5.65; the six BDG-minus-plug molecular differences
spanning 0.45 to 1.52 points; 162 headline and 612 ablation molecular cells;
199 original DNA cells at 5.26 hours; BDG guide calls one fifth of LGD-MC and
TFG. The thresholds are internally consistent: 2.99 is Bonferroni 0.05/18 and
3.16 is 0.05/32.
