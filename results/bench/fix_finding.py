"""Repair two escape-mangled passages in the finding document. Run once."""
import io

p = "FINDING_QUADRATIC_CLOSURE_VALIDITY.md"
s = io.open(p, encoding="utf-8").read()

s = s.replace("$\x07lpha$", "polarizability")
s = s.replace("**The effect is larger on polarizability than on $\\mu$**",
              "**The effect is larger on polarizability than on dipole**")

start = s.find("- **One guide, one property, one checkpoint.**")
if start >= 0:
    end = s.find("- **16 molecules, 256 draws.**")
    new = ("- **Three guide/property combinations, one generator.** Measured on `f_A_mu` (EGNN),\n"
           "  `f_A_mu_transformer` and `f_A_alpha_transformer`, all against `fm_v1`. **Not yet\n"
           "  checked:** the EGNN guides for polarizability and gap (trained on the cluster, not\n"
           "  yet pulled locally), any gap guide, SchNet at 0.0210 D, and the diffusion base\n"
           "  model. SchNet is the most valuable remaining test: it is 4x more accurate than\n"
           "  anything here and a different architecture again, so if it shows the same failure\n"
           "  the result is about learned property networks in general, not about ours.\n")
    s = s[:start] + new + s[end:]

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("repaired")
