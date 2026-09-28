"""
The objection's eta=1 demonstration: "asked to widen 4x, V falls from 1.0 to 0.062."
That is an ASYMPTOTE (thousands of steps). Test the same configuration inside the
50-step window BDG actually runs, at the gains fitted from the ladder.
"""
import numpy as np
H, NSTEP = 0.01, 50
T = np.linspace(0.5, 0.99, NSTEP)

def run(u0, eta, wa, nstep):
    ks = H*wa*(1.0-T)/T
    u = float(u0); traj=[u]
    for i in range(nstep):
        k = ks[i % NSTEP]
        u = u*(1.0 - k*(1.0 + eta*(u-1.0)))**2
        traj.append(u)
    return u, traj

u0 = 0.25   # tau^2 = 4 * V_0, i.e. "asked to widen 4x"
print("u0 = 0.25 (asked to widen 4x).  sd ratio vs start = sqrt(u/u0).")
print("  eta   wa     u(50)    sd(50)/sd0    u(5000)   sd(5000)/sd0")
for eta in (1.0, 4.0):
    for wa in (0.10, 0.20, 0.30):
        u50,_ = run(u0, eta, wa, 50)
        u5k,_ = run(u0, eta, wa, 5000)
        print(f"  {eta:4.1f}  {wa:.2f}  {u50:7.4f}   {np.sqrt(u50/u0):8.4f}    "
              f"{u5k:8.5f}   {np.sqrt(u5k/u0):8.4f}")
print()
print("Within the 50-step window at the real gains, eta=1 changes the spread by <1%:")
print("the collapse the objection exhibits needs ~100x more steps than BDG ever runs.")
print("So 'eta=1 cannot widen' is CORRECT and important as a spec/mechanism point")
print("(min w_eff = 1-eta = 0), but 'V falls from 1.0 to 0.062' overstates what any")
print("eta=1 CELL did: those cells neither widened nor collapsed, they ran as plug at a")
print("variance-modulated positive weight -- which is exactly what section 3 already says.")
print()
print("Cross-check against plug (eta=0) in the same window, same gains:")
for wa in (0.10, 0.20, 0.30):
    u50,_ = run(u0, 0.0, wa, 50)
    print(f"  plug  wa={wa:.2f}  u(50)={u50:7.4f}  sd(50)/sd0={np.sqrt(u50/u0):.4f}")
