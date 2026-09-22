"""Three corrections to the selection rule, found by running it on smoke cells.

1. CHEMISTRY ALONE MUST NOT DROP AN ARM. The smoke run dropped `btvg` for
   chemistry while it held the best property MAE on the board by a factor of
   six (2.32 delta vs 14.48 unguided). An arm that dominates the target metric
   and costs stability is a trade-off to report and to re-tune at lower
   strength -- it is not an "obvious non-competitor". It now proceeds, flagged.

2. THE "NO BETTER THAN UNGUIDED" TEST USED BAND COVERAGE ONLY. Coverage is a
   binomial at n=512 and has far less power than the MAE, which uses every
   sample's magnitude. An arm must now fail on BOTH metrics to count as
   useless, which makes dropping strictly harder.

3. THE CHEMISTRY COMPARISON WAS A BARE RATIO. Molecular stability is a
   proportion with its own standard error; at screening n a 0.094-vs-0.125
   difference is inside the noise. It is now a noise-aware test.

Run once.
"""
import io

p = "proj1/scripts/select_arms.py"
s = io.open(p, encoding="utf-8").read()

OLD = '''            ref = unguided.get(p)
            if ref:
                ru = ref["in_band_fraction"]
                s_u = math.sqrt(s_ib ** 2 + se_prop(ru, ref["n"]) ** 2)
                if ib > ru + sigma * s_u:
                    useless_everywhere = False
                if b["mol_stability"] >= chem_floor * ref["mol_stability"]:
                    chem_dead_everywhere = False
            else:
                useless_everywhere = False
                chem_dead_everywhere = False
'''
NEW = '''            ref = unguided.get(p)
            if ref:
                # (b) "no better than doing nothing" must fail on BOTH metrics.
                # Band coverage is a binomial and has little power at screening
                # n; the MAE uses every sample's magnitude and has much more.
                # Requiring both makes a drop strictly harder to justify.
                ru = ref["in_band_fraction"]
                s_u = math.sqrt(s_ib ** 2 + se_prop(ru, ref["n"]) ** 2)
                better_band = ib > ru + sigma * s_u
                m_a = b["prop_mae_eval"]
                m_u = ref["prop_mae_eval"]
                s_ma = se_mae(m_a, b["prop_rmse_eval"], n)
                s_mu = se_mae(m_u, ref["prop_rmse_eval"], ref["n"])
                better_mae = m_a < m_u - sigma * math.sqrt(s_ma ** 2 + s_mu ** 2)
                if better_band or better_mae:
                    useless_everywhere = False
                if better_band or better_mae:
                    wins_property = True

                # chemistry, as a noise-aware proportion rather than a bare
                # ratio: at screening n, 0.094 vs 0.125 is not a difference
                cs, cu = b["mol_stability"], ref["mol_stability"]
                s_c = math.sqrt(se_prop(cs, n) ** 2 + se_prop(cu, ref["n"]) ** 2)
                if cs >= chem_floor * cu - sigma * s_c:
                    chem_dead_everywhere = False
                elif wins_property:
                    tradeoff.append("%s: mol_stab %.3f vs unguided %.3f while "
                                    "winning the property" % (p, cs, cu))
            else:
                useless_everywhere = False
                chem_dead_everywhere = False
'''
assert s.count(OLD) == 1, "ref block anchor"
s = s.replace(OLD, NEW)

OLD2 = '''        beaten_everywhere = True
        useless_everywhere = True
        chem_dead_everywhere = True
        measured_anywhere = False
        notes = []'''
NEW2 = '''        beaten_everywhere = True
        useless_everywhere = True
        chem_dead_everywhere = True
        measured_anywhere = False
        wins_property = False
        tradeoff = []
        notes = []'''
assert s.count(OLD2) == 1, "flags anchor"
s = s.replace(OLD2, NEW2)

OLD3 = '''        if a in ALWAYS_KEEP:
            verdicts[a] = ("KEEP (reference)", notes)
        elif not measured_anywhere:
            verdicts[a] = ("REVIEW (no clean cell anywhere)", notes)
        elif chem_dead_everywhere:
            verdicts[a] = ("DROP (chemistry collapse on every property)", notes)
        elif beaten_everywhere and useless_everywhere:
            verdicts[a] = ("DROP (beaten everywhere and no better than "
                           "unguided)", notes)
        else:
            verdicts[a] = ("PROCEED", notes)'''
NEW3 = '''        notes.extend(tradeoff)
        if a in ALWAYS_KEEP:
            verdicts[a] = ("KEEP (reference)", notes)
        elif not measured_anywhere:
            verdicts[a] = ("REVIEW (no clean cell anywhere)", notes)
        elif chem_dead_everywhere and not wins_property:
            # chemistry collapse is disqualifying only when the arm is not
            # buying anything with it
            verdicts[a] = ("DROP (chemistry collapse, wins nothing)", notes)
        elif beaten_everywhere and useless_everywhere:
            verdicts[a] = ("DROP (beaten everywhere and no better than "
                           "unguided on either metric)", notes)
        elif tradeoff:
            verdicts[a] = ("PROCEED (chemistry trade-off -- re-tune strength "
                           "at full scale)", notes)
        else:
            verdicts[a] = ("PROCEED", notes)'''
assert s.count(OLD3) == 1, "verdict anchor"
s = s.replace(OLD3, NEW3)

# the docstring must state the rule that is actually implemented
OLD4 = '''An arm is DROPPED only if, on EVERY property, both of:

  (a) it is separated from the best arm by more than `--sigma` combined
      standard errors on band coverage, AND
  (b) it is no better than doing nothing -- its band coverage does not beat the
      unguided baseline by more than `--sigma` combined standard errors

...or it collapses chemistry on every property (molecular stability below
`--chem-floor` x unguided). Anything else is kept, including anything the
screening could not measure.'''
NEW4 = '''An arm is DROPPED only if, on EVERY property, both of:

  (a) it is separated from the best arm by more than `--sigma` combined
      standard errors on band coverage, AND
  (b) it is no better than doing nothing on EITHER metric -- neither its band
      coverage nor its MAE beats the unguided baseline by `--sigma` combined
      standard errors

...or it collapses chemistry on every property AND wins nothing. Chemistry
alone never drops an arm: an arm that dominates the property metric and costs
stability is a trade-off to report and re-tune at lower strength, not a
non-competitor. Anything else is kept, including anything the screening could
not measure.'''
assert s.count(OLD4) == 1, "docstring anchor"
s = s.replace(OLD4, NEW4)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("selection rule corrected")
