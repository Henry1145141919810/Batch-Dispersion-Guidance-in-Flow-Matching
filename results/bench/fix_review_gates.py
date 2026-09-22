"""Gate the D4 / D10 fixes, and de-vacuum the self-limiting check. Run once.

`btvg_var_self_limits_below_tau` short-circuited through `not under.any()`:
in the fixture V_exact ~ 24-30 against tau^2 = 0.137, so `under` was EMPTY and
the check returned 0.0 without evaluating anything. The property the code calls
"the decisive one" had no test at all. Its sibling has an `assert over.any()`
guard; this one had none.

Replaced with three real checks, each with a guard that fails if the fixture
ever stops exercising it:
  * V_F below tau^2  -> the variance term is exactly ZERO (stops, not reverses)
  * V_F non-positive -> the variance term is exactly ZERO (D4: the clamp used
    to turn this into a +5e11 widening coefficient)
  * V_F above tau^2  -> unchanged, still strictly downhill
"""
import io

NL = chr(10)
p = "proj1/tests/test_v2_arms.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''    # below tau^2 the sign must flip: self-limiting, not collapsing
    under = ~over
    R["btvg_var_self_limits_below_tau"] = 0.0 if (not bool(under.any())) or bool(
        (dirn[under] > 1e-6).all()) else 1.0'''

NEW = '''    # BELOW tau^2 THE ARM MUST STOP, NOT REVERSE. The previous version of this
    # check short-circuited on an empty `under` set and so tested nothing. The
    # fixture cannot produce V < tau^2 at a sane tau, so it is produced here by
    # raising tau instead -- and guarded, so it fails if that stops working.
    tau_big = float(V_exact.max().sqrt() * 10.0)
    V_over_big = V_exact / tau_big ** 2
    assert bool((V_over_big < 1.0).all()), "fixture must sit below tau_big^2"
    Gb_c, Gb_f, _ = guidance_field(f_net, post_fn, coords, feats, mask, y, 0.5,
                                   mode="btvg_var", tau=tau_big)
    R["btvg_var_stops_below_tau"] = max(Gb_c.abs().max().item(),
                                        Gb_f.abs().max().item())

    # D4: a NON-POSITIVE V_F means the variance model has broken down. Sigma is
    # only symmetric for an exact score, and on the real generator V came out
    # <= 0 on up to 3/16 samples. Clamping V to +1e-12 turned those into a
    # coefficient of +5e11 -- the largest widening step the clip allows,
    # exactly where the model was least trustworthy. It must now contribute
    # nothing. Forced here by a property whose gradient at m is zero, which
    # makes V identically 0.
    class _Flat(torch.nn.Module):
        def forward(self, c, f, mask):
            return (0.0 * c).sum(dim=(1, 2)) + (0.0 * f).sum(dim=(1, 2))

    Gz_c, Gz_f, dz = guidance_field(_Flat(), post_fn, coords, feats, mask, y,
                                    0.5, mode="btvg_var", tau=tau)
    assert bool((dz["btvg_V_raw"] <= 0).all()), "fixture must force V <= 0"
    R["btvg_var_off_when_V_nonpositive"] = max(Gz_c.abs().max().item(),
                                               Gz_f.abs().max().item())
    R["btvg_reports_V_nonpositive"] = abs(
        float(dz["btvg_V_nonpositive"].mean()) - 1.0)'''

assert s.count(OLD) == 1, "self-limit anchor"
s = s.replace(OLD, NEW)

# the exact-field reference must know about the <= 0 clamp
OLD2 = '''    bcoef = (-0.5 * (1.0 / tau ** 2 - 1.0 / V_exact)).view(-1, 1, 1)'''
NEW2 = '''    # the implementation clamps this coefficient at <= 0 (it may concentrate,
    # never widen); the fixture sits well above tau^2, so the clamp is inactive
    # here and the reference is the raw coefficient. Asserted, not assumed.
    assert bool((V_exact > tau ** 2).all()), "fixture must sit above tau^2"
    bcoef = (-0.5 * (1.0 / tau ** 2 - 1.0 / V_exact)).view(-1, 1, 1)
    assert bool((bcoef <= 0).all()), "clamp must be inactive in this fixture"'''
assert s.count(OLD2) == 1, "bcoef anchor"
s = s.replace(OLD2, NEW2)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("gates: D4 + D10 covered, self-limiting check de-vacuumed")
