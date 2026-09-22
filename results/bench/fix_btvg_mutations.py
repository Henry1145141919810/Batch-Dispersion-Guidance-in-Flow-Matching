"""Refresh the BTVG mutation anchors after the D4/D10 fix, and gate the new
guards themselves. Run once.

The D4/D10 fix rewrote the variance-coefficient line, so two mutation anchors
went stale and reported ANCHOR MISSING -- which the harness correctly counts as
a survivor rather than a pass. Both are restored against the new source, and
two more are added so the guards that fix cannot be silently removed.
"""
import io

NL = chr(10)
p = "results/bench/mutation_test.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''    ("BTVG: KL coefficient sign flipped (widens instead of concentrating)",
     "            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V)).view(-1, 1, 1)",
     "            b = (-0.5 * (1.0 / V - 1.0 / tau_t ** 2)).view(-1, 1, 1)"),
    ("BTVG: likelihood coefficient instead of KL (the prior-art baseline)",
     "            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V)).view(-1, 1, 1)",
     "            b = (0.5 * ((fval - y) ** 2 / V ** 2 - 1.0 / V)).view(-1, 1, 1)"),'''

NEW = '''    ("BTVG: KL coefficient sign flipped (widens instead of concentrating)",
     "            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V_safe)).clamp(max=0.0)",
     "            b = (-0.5 * (1.0 / V_safe - 1.0 / tau_t ** 2)).clamp(max=0.0)"),
    ("BTVG: likelihood coefficient instead of KL (the prior-art baseline)",
     "            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V_safe)).clamp(max=0.0)",
     "            b = (0.5 * ((fval - y) ** 2 / V_safe ** 2"
     " - 1.0 / V_safe)).clamp(max=0.0)"),
    # the D10 guard: without the clamp the arm WIDENS over the last ~15% of
    # every trajectory, because V_F -> 0 as t -> 1 by construction
    ("BTVG: <=0 clamp removed (late-trajectory widening returns, D10)",
     "            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V_safe)).clamp(max=0.0)",
     "            b = (-0.5 * (1.0 / tau_t ** 2 - 1.0 / V_safe))"),
    # the D4 guard: a non-positive V_F used to become a +5e11 coefficient
    ("BTVG: non-positive V_F guard removed (+5e11 widening step, D4)",
     "            b = torch.where(V_ok, b, torch.zeros_like(b)).view(-1, 1, 1)",
     "            b = b.view(-1, 1, 1)"),
    # the D5 gate's subject: identically zero in the linear fixture, so before
    # the nonlinear gate existed both of these survived the whole suite
    ("grad V_F: mean-map curvature term deleted (D5)",
     NL.join(["    d_c = torch.zeros_like(c_in) if d_c is None else d_c.detach()",
              "    d_f = torch.zeros_like(f_in) if d_f is None else d_f.detach()"]),
     NL.join(["    d_c = torch.zeros_like(c_in)",
              "    d_f = torch.zeros_like(f_in)"])),
    ("grad V_F: mean-map curvature term scaled by 100 (D5)",
     NL.join(["    d_c = torch.zeros_like(c_in) if d_c is None else d_c.detach()",
              "    d_f = torch.zeros_like(f_in) if d_f is None else d_f.detach()"]),
     NL.join(["    d_c = torch.zeros_like(c_in) if d_c is None else 100.0 * d_c.detach()",
              "    d_f = torch.zeros_like(f_in) if d_f is None else 100.0 * d_f.detach()"])),'''

assert s.count(OLD) == 1, "btvg mutation anchor"
s = s.replace(OLD, NEW)
io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("mutation anchors refreshed; D4/D5/D10 guards now gated")
