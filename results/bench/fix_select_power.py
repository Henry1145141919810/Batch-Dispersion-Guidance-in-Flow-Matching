"""Clause (a) of the drop rule must use MAE, not band coverage. Run once.

WHAT HAPPENED. On the first real stage-v2 screen the rule dropped NOTHING --
14 arms, 0 drops -- which makes the screen useless for choosing what goes to the
full-scale run. The cause is power, not data:

  clause (a) "separated from the best arm" was measured on BAND COVERAGE, a
  binomial with se ~ 0.012 at n=512. On alpha, `spbc` sits 1.5 sigma from the
  best arm on coverage -- and 6.9 sigma from it on MAE.

Coverage is the acceptance criterion, so it belongs in the report. It is a poor
TEST STATISTIC: it throws away every sample's magnitude and keeps one bit.
MAE uses the magnitudes and has three to four times the power here. The
docstring already made this argument for clause (b); clause (a) was simply not
updated with it.

Measured separation of the three arms this changes, best-arm MAE sigma by
property (mu / alpha / gap):

    band  8.6 / 8.1 / 7.1      vs unguided  0.5 /  0.0 / 0.6
    rch   8.0 / 9.4 / 5.9      vs unguided  1.2 / -1.7 / 1.2
    spbc  9.0 / 6.9 / 7.7      vs unguided  0.2 /  1.3 / 0.1

All three are beaten by 6-9 sigma and beat doing nothing by under 1.3 sigma on
either metric. `band`'s best alpha cell is bit-identical to unguided, and `rch`
is WORSE than unguided on alpha.

THIS IS A POST-HOC CHANGE TO A PRE-REGISTERED RULE AND MUST BE REPORTED AS ONE.
Mitigating, and checkable from the numbers above: clause (b) already failed for
these three on BOTH metrics, so only clause (a) moved; the margins are 6-9
sigma against under 1.3, not marginal calls; and no arm that beats unguided
anywhere is dropped -- `btvg_var` is beaten everywhere but wins on gap at 3.4
sigma and is kept.
"""
import io

NL = chr(10)
p = "proj1/scripts/select_arms.py"
s = io.open(p, encoding="utf-8").read()

OLD_D = '''An arm is DROPPED only if, on EVERY property, both of:

  (a) it is separated from the best arm by more than `--sigma` combined
      standard errors on band coverage, AND'''
NEW_D = '''An arm is DROPPED only if, on EVERY property, both of:

  (a) it is separated from the best arm by more than `--sigma` combined
      standard errors ON MAE, AND'''
assert s.count(OLD_D) == 1, "docstring clause a"
s = s.replace(OLD_D, NEW_D)

OLD_N = '''DIVERGENCE IS NOT ALLOWED TO WIN.'''
NEW_N = '''WHY CLAUSE (a) USES MAE AND NOT COVERAGE -- a post-hoc change, recorded.
The first real stage-v2 screen dropped 0 of 14 arms, because clause (a) was
measured on band coverage: a binomial with se ~ 0.012 at n=512, which cannot
separate anything. The same comparisons run at 6-9 sigma on MAE. Coverage is
the acceptance criterion and stays in the report; it is a poor test statistic
because it discards every sample's magnitude. This rule was changed AFTER
seeing that screen, which is why it is written down here. The three arms it
drops (band, rch, spbc) fail clause (b) on BOTH metrics independently, so the
change moved only clause (a), and by margins of 6-9 sigma against under 1.3.

DIVERGENCE IS NOT ALLOWED TO WIN.'''
assert s.count(OLD_N) == 1, "docstring note"
s = s.replace(OLD_N, NEW_N)

# ---- the ceiling becomes a FLOOR on MAE, alongside the coverage ceiling
OLD_C = '''    ceiling = {}
    for p in props:
        for (pp, _a), v in summary.items():
            if pp != p or not v["best"]:
                continue
            key = (p, v["stratum"])
            ceiling[key] = max(ceiling.get(key, 0.0),
                               v["best"]["in_band_fraction"])'''
NEW_C = '''    ceiling = {}          # best band coverage -- reported, not tested on
    floor = {}            # best (lowest) MAE and its se -- what clause (a) tests
    for p in props:
        for (pp, _a), v in summary.items():
            if pp != p or not v["best"]:
                continue
            b = v["best"]
            key = (p, v["stratum"])
            ceiling[key] = max(ceiling.get(key, 0.0), b["in_band_fraction"])
            cand = (b["prop_mae_eval"],
                    se_mae(b["prop_mae_eval"], b["prop_rmse_eval"], b["n"]))
            if key not in floor or cand[0] < floor[key][0]:
                floor[key] = cand'''
assert s.count(OLD_C) == 1, "ceiling anchor"
s = s.replace(OLD_C, NEW_C)

OLD_T = '''            top = ceiling.get((p, v["stratum"]), 0.0)
            s_top = se_prop(top, n)
            comb = math.sqrt(s_ib ** 2 + s_top ** 2)
            if ib + sigma * comb >= top:
                beaten_everywhere = False'''
NEW_T = '''            # clause (a), on MAE: is this arm clearly worse than the best?
            bm, bse = floor.get((p, v["stratum"]), (float("-inf"), 0.0))
            m_here = b["prop_mae_eval"]
            s_here = se_mae(m_here, b["prop_rmse_eval"], n)
            comb = math.sqrt(s_here ** 2 + bse ** 2)
            if m_here - sigma * comb <= bm:
                beaten_everywhere = False'''
assert s.count(OLD_T) == 1, "beaten test anchor"
s = s.replace(OLD_T, NEW_T)

OLD_V = '''            verdicts[a] = ("DROP (beaten everywhere and no better than "
                           "unguided on either metric)", notes)'''
NEW_V = '''            verdicts[a] = ("DROP (beaten on MAE everywhere and no better "
                           "than unguided on either metric)", notes)'''
assert s.count(OLD_V) == 1, "verdict text"
s = s.replace(OLD_V, NEW_V)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("clause (a) now tests MAE; the post-hoc change is recorded in the docstring")
