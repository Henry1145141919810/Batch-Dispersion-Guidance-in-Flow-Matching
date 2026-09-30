# DNA window follow-up: reporting addendum, 29 September 2026

This records the data added by `cfc5d64` (28 Sep, 22:47:08 -04:00) and
`f568f68` (22:47:59), merged locally by `498dc40`. It supplements the original
[M2 protocol](MODALITY2_V3_PROTOCOL.md); it does not retroactively preregister
the new window. Paper v9 and PowerPoint v4 use this reporting distinction.

| Comparison | Property | Guidance start | Strength | Samples and seeds |
|---|---|---|---|---|
| Original protocol comparison | GC and CpG | t >= 0.5 | w = 1, 4 | 2,000; 20260921/22/23 |
| New window follow-up | GC only | t >= 0.3 | w = 4 | 2,000; 20260921/22/23 |
| Earlier-window probes | GC only | t >= 0.2 | w = 4 | 2,000; seed 20260921 only |

All new cells use 100 Euler steps, batch 500 (four controller batches), q50,
the same checkpoint, delta ratio 0.16, and the velocity-relative cap 1.0.
The t >= 0.3 group contains plug-in, TMPD-inspired, LGD-MC, TFG-MC,
BDG eta=4/tau_m=0.5, and the eta=0 control: 18 cells in total.
Four t >= 0.2 probes bring `results/m2/m2win/n2000` to 22 files.
No three-seed CpG t >= 0.3 or GC tau_m=1 follow-up is present.

The sequence flow's endpoint uncertainty is already very small at t=0.5.
The new window addresses the regime diagnosed by the earlier uncertainty
measurements. At t >= 0.3, mean GC coverage differs among the four baselines:
30.25%, 29.18%, 29.97%, and 29.77%, respectively. BDG reaches 37.33%, a
7.08-point gain over plug-in with paired seed SD 4.53 points. These numerical
differences do not establish an ordering among optimally tuned methods.

Keep the windows separate in every table and contrast. Main Table 4 now shows
GC at 0.3 and CpG at 0.5, with the windows explicit in the caption; the original
both-at-0.5 table remains in the appendix. Missing settings use a dash.
Show mean +/- seed SD, Hamming, 3-mer JSD, and same-cell time. GC's new BDG JSD
is better than unguided but worse than plug-in. The original 32-contrast DNA
threshold belongs to the original comparisons; the new window is exploratory.

Eta=0 gives the plug-in coefficient algebraically. New-window decoded coverage
matches in the supplied controls, while continuous outputs show small rerun
differences. `m2_sanity.py` uses 1e-5 tolerance for the continuous metrics.
The original late-window exact overlap must not be generalized to all windows.

`M2_WINDOW.md` and `M2_STRENGTH.md` report separate n=500 diagnostics.
The latter's w=16 selection is not the w=4 used by the new n=2000 cells.
Single-seed probes and diagnostic settings are not pooled into headline means.

Band half-width remains delta=max(0.16s,4.4q), where s is training-property SD
and q is 1/500 (GC) or 1/499 (CpG). Full bandwidth is 2delta. BDG's requested
spread tau=tau_m*s is a separate control; changing tau does not change delta.
The new empirical training-distribution figures mark q50 and these frozen bands.

Audit: `paper/results_manifest_v9.json` records all 995 result-cell hashes,
including the 22 new files; `paper/property_distributions_v9.json` records the
dataset hashes and histogram bins. New cells add 1.604 recorded hours, giving
6.866 DNA hours overall (not elapsed time or a hardware-matched speed result).
