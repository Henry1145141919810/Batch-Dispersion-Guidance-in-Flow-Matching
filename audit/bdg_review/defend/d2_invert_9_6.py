"""
DEFENCE LINE THE OBJECTION MISSED.

The objection uses u* = 1-1/eta to explain ONE of the three values section 9.6 reports
as an unexplained defect (alpha 0.84-0.87 = sqrt(0.75) at eta=4) and leaves mu
(0.574-0.678) and gap (0.673) as "what a negative natural drift does".

But 9.6 also says "the fixed point is ALSO eta-dependent" -- i.e. the hold array
(bdg_hold.slurm, 6 per-property fixed-point cells) swept eta. So invert the closed form
on each reported value and see whether the implied eta is a round number that a 6-cell
eta sweep would plausibly contain. If it is, the correction explains ALL of 9.6, not 1/3
of it, and 9.6 stops being a defect at all.

  sd*/tau = sqrt(1-1/eta)   =>   eta = 1/(1 - (sd*/tau)^2)
"""
import numpy as np

print("reported BDG_FIXED_POINT measurements (handoff section 9.6) -> implied eta")
print("  value    implied eta   nearest plausible grid eta   sqrt(1-1/eta) at that eta   resid")
rows = [("mu   lo", 0.574), ("mu   hi", 0.678), ("alpha lo", 0.84),
        ("alpha hi", 0.87), ("gap", 0.673)]
grid = [1.0, 1.5, 2.0, 3.0, 4.0, 8.0]
for name, v in rows:
    eta = 1.0 / (1.0 - v ** 2)
    near = min(grid, key=lambda g: abs(np.sqrt(max(1 - 1 / g, 0)) - v))
    pred = np.sqrt(max(1 - 1 / near, 0))
    print(f"  {name:8s} {v:.3f}   {eta:8.3f}        {near:6.1f}                  "
          f"{pred:.4f}              {v-pred:+.4f}")

print()
print("the same table read forward, as the prediction to check against the hold cells:")
for g in grid:
    print(f"  eta={g:4.1f}  ->  measured sd*/tau should be {np.sqrt(max(1-1/g,0)):.4f}")
print()
print("If bdg_hold.slurm's 6 cells used eta in {1.5, 2, 4} (2 per property, which is what")
print("'6 cells, per-property fixed-point' and 'the fixed point is eta-dependent' imply),")
print("then mu 0.574/0.678 ~ eta 1.5/2 (predicted 0.5774/0.7071), gap 0.673 ~ eta 2")
print("(0.7071), alpha 0.84-0.87 ~ eta 4 (0.8660). Residuals -0.003/-0.029/-0.034/")
print("-0.026/+0.004 -- all NEGATIVE and small, i.e. a uniform small shortfall, which is")
print("what a finite 50-step window plus a negative natural drift gives.")
print()
print("COST TO SETTLE: read eta off the 6 hold-cell FILENAMES already on disk in the Betty")
print("tree and compare. Zero compute, ~10 minutes. It converts open item 6 from")
print("'BDG_FIXED_POINT is wrong / unexplained' into a validated closed-form prediction.")
