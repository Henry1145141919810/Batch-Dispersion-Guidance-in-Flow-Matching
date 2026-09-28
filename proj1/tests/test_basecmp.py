"""Gates for the base-model comparison's DECISION logic, on synthetic cells.

No generator, no GPU: every cell is a hand-built JSON whose correct outcome is
fixed by construction, so the rules are tested rather than re-derived.

  names_unchanged         the per-cell window tag reproduces the old global tag
                          byte for byte, so the 216 finished EDMsecond cells and
                          every other pre-basecmp cell are still "done"
  window_mirror           t_start maps to tau_max_guide = 1 - t_start, and the
                          cell records both plus t_min_guide = t_start
  plan_unguided_once      unguided is planned once per property, not once per
                          window (a second copy would double-count the floor)
  plan_grid_complete      every guided arm has the full strength x start-time grid
  pick_maximises_inband   the pick is the in-band argmax, not the MAE argmax
  pick_is_decoded         it follows the DECODED in-band when soft disagrees
  pick_tie_smaller_w      an exact tie goes to the smaller strength
  pick_tie_later_t        a tie at equal strength goes to the LATER start-time
  pick_floor_vs_free      the floor set refuses a cell below the floor; the free
                          set takes it
  pick_floor_limited      an arm that never clears the floor is flagged, and the
                          freeze refuses to write without --allow-floor-limited
  pick_grid_edge          a pick at the top or bottom of the strength grid is
                          flagged
  refine_interior         an interior pick gets the two geometric midpoints
  refine_edge             an edge pick gets the outward extension, capped
  refine_skips_present    a refine cell that already exists is not re-planned
  freeze_refuses_mixed_n  cells at two sample sizes are refused, not averaged
  freeze_refuses_delta    two deltas for one property are refused
  freeze_refuses_backend  a cell from the other base model is refused
  frozen_guard            a frozen file is refused on the wrong backend
  full_plan_dedup         floor and free picking the same cell costs one cell
  delta_is_generator_free the local-delta report is marked generator-independent

Run: python proj1/tests/test_basecmp.py       (CPU, seconds)
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))

import basecmp_freeze as bf                                        # noqa: E402
import transfer_sweep as ts                                        # noqa: E402

R = {}


def gate(name, ok):
    R[name] = bool(ok)


PROPS = ["mu"]
ARMS = ts.backend_arms("equifm")
GUIDED = [a for a in ARMS if a != "unguided"]


def cell(prop, arm, w, t_start, ib_dec, stab, ib_soft=None, mae_dec=1.0,
         n=1000, seed=ts.BASECMP_SEED, delta=0.17, backend="EquiFM",
         nonfinite=0, uniq=1.0, f_b_mean=2.70):
    return {"stage": "basecmp", "study": "basecmp", "prop": prop, "arm": arm,
            "target_name": ts.BASECMP_TARGET, "w": float(w),
            "t_start": float(t_start), "tau_max_guide": 1.0 - float(t_start),
            "t_min_guide": float(t_start), "n": n, "seed": seed,
            "backend": backend, "prov": {"gen_md5": "abc"}, "batch": 128,
            "steps": 100, "solver": "euler", "clip": 1.0, "delta": delta,
            "calibration": {"delta_mode": "local"}, "n_nonfinite": nonfinite,
            "mol_stability": stab, "atom_stability": 0.95, "validity": 0.85,
            "in_band_fraction": ib_dec if ib_soft is None else ib_soft,
            "in_band_fraction_dec": ib_dec,
            "prop_mae_eval": mae_dec, "prop_mae_eval_dec": mae_dec,
            "uniqueness_of_valid": uniq, "unique_valid_per_sample": uniq,
            "guide_eval_gap_mean": 0.1, "clipped_sample_steps": 0,
            "guided_steps": 50, "seconds": 1.0, "f_B_mean": f_b_mean}


def write(d, cells):
    os.makedirs(d, exist_ok=True)
    for i, c in enumerate(cells):
        p = os.path.join(d, "tr__%s__%s__%s__w%g__t%g__%d.json"
                         % (c["prop"], c["arm"], c["target_name"], c["w"],
                            c["t_start"], i))
        with open(p, "w") as fh:
            json.dump(c, fh)


def grid_cells(unguided_stab=0.80):
    """Unguided stability 0.80 -> floor 0.72.

    Built so each arm has a KNOWN right answer:
      plug      in-band rises with w, stability falls through the floor at w=4.
                floor pick -> w=1 (highest in-band still clearing), free -> w=16
      tmpd      clears everywhere, in-band peaks at an interior w=1, t=0.5
      lgd_mc    clears everywhere and in-band still rising at w=16 -> grid_edge
      tfg       best at the BOTTOM edge w=0.05 -> grid_edge
      btvg      never clears the floor -> floor_limited
      btvg_var  flat: every cell identical -> exercises both tie-breaks
    """
    out = [cell("mu", "unguided", 1.0, 0.5, 0.02, unguided_stab)]
    for t in ts.BASECMP_T_STARTS:
        for w in ts.BASECMP_STRENGTHS:
            # a mild window effect peaking at t=0.5, so the joint argmax has to
            # look at both axes
            tb = 1.0 - abs(t - 0.5)
            out.append(cell("mu", "plug", w, t, 0.05 * tb * w ** 0.3,
                            0.79 - 0.02 * w))
            out.append(cell("mu", "tmpd", w, t,
                            0.10 * tb * (1.0 - abs(__import__("math").log(w) ) / 8.0),
                            0.78))
            out.append(cell("mu", "lgd_mc", w, t, 0.04 * tb * w ** 0.25, 0.77))
            out.append(cell("mu", "tfg", w, t, 0.09 * tb / w ** 0.2, 0.76))
            out.append(cell("mu", "btvg", w, t, 0.30, 0.50 - 0.001 * w))
            out.append(cell("mu", "btvg_var", w, t, 0.06, 0.75))
    return out


def fargs(d, **kw):
    base = dict(backend="equifm", screen_dir=d, props="mu", arms=",".join(ARMS),
                json_out="", table_out="", md_out="", emit_refine="",
                refine_t=False, allow_floor_limited=False, allow_partial=False)
    base.update(kw)
    return argparse.Namespace(**base)


def run_freeze(d, **kw):
    """basecmp_freeze.main() with argv patched, returning (rc, stdout)."""
    import contextlib
    import io
    a = fargs(d, **kw)
    buf = io.StringIO()
    real = bf.argparse.ArgumentParser.parse_args
    bf.argparse.ArgumentParser.parse_args = lambda self, *x, **y: a
    try:
        with contextlib.redirect_stdout(buf):
            rc = bf.main()
    finally:
        bf.argparse.ArgumentParser.parse_args = real
    return rc, buf.getvalue()


def main():
    tmp = tempfile.mkdtemp(prefix="bc_")
    try:
        # ---- the non-breaking guarantee: names are unchanged ----------------
        a = argparse.Namespace(n=512, steps=100, solver="euler", grid="uniform",
                               tau_max_guide=0.5, batch=128, seed=20260922,
                               block_start=0)
        old = ts.config_tag(a)
        new = ts.config_tag(a, 1.0 - a.tau_max_guide)
        a2 = argparse.Namespace(**dict(vars(a), tau_max_guide=0.25))
        gate("names_unchanged",
             old == new
             and ts.config_tag(a2) == ts.config_tag(a2, 0.75)
             and ts.config_tag(a, 0.25) != old)

        # ---- the window mirror --------------------------------------------
        c = cell("mu", "plug", 1.0, 0.25, 0.1, 0.8)
        gate("window_mirror", abs(c["tau_max_guide"] - 0.75) < 1e-12
             and abs(c["t_min_guide"] - 0.25) < 1e-12)

        # ---- the plan ------------------------------------------------------
        cells = ts.plan_basecmp_cells(PROPS, ARMS, ts.BASECMP_T_STARTS)
        ung = [x for x in cells if x[1] == "unguided"]
        gate("plan_unguided_once", len(ung) == 1 and ung[0][4] == 0.5)
        want = {(w, t) for w in ts.BASECMP_STRENGTHS for t in ts.BASECMP_T_STARTS}
        gate("plan_grid_complete",
             all({(x[3], x[4]) for x in cells if x[1] == a_} == want
                 for a_ in GUIDED)
             and len(cells) == len(GUIDED) * len(want) + 1
             and len(set(cells)) == len(cells))

        # ---- the picks -----------------------------------------------------
        d = os.path.join(tmp, "grid")
        write(d, grid_cells())
        rc, out = run_freeze(d, allow_floor_limited=True,
                             json_out=os.path.join(tmp, "frozen.json"))
        fz = json.load(open(os.path.join(tmp, "frozen.json")))
        F, G = fz["frozen"]["floor"], fz["frozen"]["free"]
        S = fz["status"]

        # plug: in-band rises with w, stability 0.79-0.02w crosses 0.72 at w>3.5
        gate("pick_maximises_inband",
             rc == 0 and F["plug"]["mu"]["w"] == 1.0
             and G["plug"]["mu"]["w"] == 16.0
             and F["plug"]["mu"]["t_start"] == 0.5)
        gate("pick_floor_vs_free",
             F["plug"]["mu"]["mol_stability"] >= fz["floor"]["mu"] - 1e-12
             and G["plug"]["mu"]["mol_stability"] < fz["floor"]["mu"])
        gate("pick_floor_limited",
             S["floor"]["btvg"]["mu"] == "floor_limited"
             and "btvg/mu" in fz["floor_limited"])
        gate("pick_grid_edge",
             S["free"]["lgd_mc"]["mu"] == "grid_edge"
             and F["tfg"]["mu"]["w"] == 0.05
             and S["floor"]["tfg"]["mu"] == "grid_edge")
        # btvg_var is flat -> tie everywhere: smallest w, then the LATEST t
        gate("pick_tie_smaller_w", F["btvg_var"]["mu"]["w"] == 0.05)
        gate("pick_tie_later_t", F["btvg_var"]["mu"]["t_start"] == 0.75)

        # decoded wins over soft: one cell with a huge SOFT in-band must lose
        d2 = os.path.join(tmp, "dec")
        cs = grid_cells()
        cs.append(cell("mu", "plug", 0.25, 0.5, 0.001, 0.79, ib_soft=0.99))
        write(d2, cs)
        _, _ = run_freeze(d2, allow_floor_limited=True,
                          json_out=os.path.join(tmp, "f2.json"))
        f2 = json.load(open(os.path.join(tmp, "f2.json")))
        gate("pick_is_decoded", f2["frozen"]["floor"]["plug"]["mu"]["w"] == 1.0)

        # ---- the edge flags, both axes -------------------------------------
        # a pick at the first or last START-TIME is at a grid boundary too, on the
        # axis the protocol calls the project's largest measured single effect
        cs_t = [cell("mu", "plug", 1.0, t, 0.1 + (0.05 if t == 0.75 else 0.0), 0.8)
                for t in [0.05, 0.25, 0.5, 0.75]]
        r_t, st_t, fl_t = bf.pick(cs_t, 0.0, constrained=False)
        gate("edge_flags_t_axis",
             r_t["t_start"] == 0.75 and fl_t["t_edge"] and not fl_t["w_edge"]
             and st_t == "grid_edge")

        # THE DISAPPEARING EDGE the review described: the screen picks (16, 0.5),
        # the refine adds w=64 AT t=0.5 ONLY, and the re-freeze's argmax moves to
        # (16, 0.75) -- where 16 is still the top of everything ever run. Pooling
        # all windows would call that interior and drop the warning.
        cs_e = [cell("mu", "plug", w, t, 0.10, 0.8)
                for t in [0.5, 0.75] for w in [4.0, 16.0]]
        cs_e.append(cell("mu", "plug", 64.0, 0.5, 0.05, 0.8))     # refine, t=0.5
        cs_e.append(cell("mu", "plug", 16.0, 0.75, 0.30, 0.8))    # the new argmax
        r_e, st_e, fl_e = bf.pick(cs_e, 0.0, constrained=False)
        gate("edge_survives_refine",
             r_e["w"] == 16.0 and r_e["t_start"] == 0.75
             and fl_e["w_edge"] and fl_e["w_grid_at_t"] == [4.0, 16.0]
             and st_e == "grid_edge")

        # a floor-limited pick used to return before the edge check, so it carried
        # neither warning
        cs_fl = [cell("mu", "btvg", w, 0.5, 0.3, 0.50) for w in [0.05, 16.0]]
        r_f, st_f, fl_f = bf.pick(cs_fl, 0.72, constrained=True)
        gate("edge_on_floor_limited",
             st_f == "floor_limited" and (fl_f["w_edge"] or fl_f["t_edge"]))

        # ---- the refine rule ----------------------------------------------
        gate("refine_interior", bf.neighbours(1.0, ts.BASECMP_STRENGTHS) == [0.5, 2.0])
        top = bf.neighbours(16.0, ts.BASECMP_STRENGTHS)
        bot = bf.neighbours(0.05, ts.BASECMP_STRENGTHS)
        gate("refine_edge",
             8.0 in top and 64.0 in top and 256.0 not in top
             and 0.0125 in bot and all(x >= bf.W_REFINE_MIN for x in bot))
        rp = os.path.join(tmp, "refine.json")
        run_freeze(d, allow_floor_limited=True, emit_refine=rp)
        plan = json.load(open(rp))
        have = {(r["prop"], r["arm"], float(r["w"]), float(r["t_start"]))
                for r in grid_cells()}
        new_cells = {(c["prop"], c["arm"], c["w"], c["t_start"])
                     for c in plan["cells"]}
        gate("refine_skips_present",
             not (new_cells & have) and len(new_cells) > 0
             and plan["backend"] == "EquiFM")

        # ---- the consistency gates ----------------------------------------
        d3 = os.path.join(tmp, "mixed_n")
        write(d3, grid_cells() + [cell("mu", "plug", 1.0, 0.5, 0.9, 0.8, n=512)])
        gate("freeze_refuses_mixed_n", run_freeze(d3)[0] == 2)

        d4 = os.path.join(tmp, "mixed_delta")
        write(d4, grid_cells() + [cell("mu", "plug", 0.5, 0.05, 0.9, 0.8,
                                       delta=0.99)])
        gate("freeze_refuses_delta", run_freeze(d4)[0] == 2)

        # a hole in the base grid is refused, because an argmax cannot show one
        d_hole = os.path.join(tmp, "hole")
        cs = [c for c in grid_cells()
              if not (c["arm"] == "tmpd" and c["w"] == 4.0 and c["t_start"] == 0.25)]
        write(d_hole, cs)
        rc_hole, out_hole = run_freeze(d_hole, allow_floor_limited=True)
        gate("freeze_refuses_hole",
             rc_hole == 2 and "INCOMPLETE" in out_hole
             and run_freeze(d_hole, allow_floor_limited=True,
                            allow_partial=True)[0] == 0)
        # the missing unguided cell is a hole too: without it there is no floor
        d_nu = os.path.join(tmp, "nounguided")
        write(d_nu, [c for c in grid_cells() if c["arm"] != "unguided"])
        gate("freeze_refuses_no_unguided",
             run_freeze(d_nu, allow_floor_limited=True)[0] == 2)

        d5 = os.path.join(tmp, "mixed_backend")
        write(d5, grid_cells() + [cell("mu", "plug", 1.0, 0.05, 0.9, 0.8,
                                       backend="FM (ours)")])
        gate("freeze_refuses_backend", run_freeze(d5)[0] == 2)

        # floor-limited without the override must refuse to write
        gate("pick_floor_limited_refused",
             run_freeze(d, json_out=os.path.join(tmp, "never.json"))[0] == 2
             and not os.path.exists(os.path.join(tmp, "never.json")))

        # ---- the frozen file's guards -------------------------------------
        ok_wrong_backend = False
        try:
            ts.load_basecmp_frozen(os.path.join(tmp, "frozen.json"), PROPS,
                                   ARMS, "fm")
        except SystemExit:
            ok_wrong_backend = True
        ok_right = ts.load_basecmp_frozen(os.path.join(tmp, "frozen.json"),
                                          PROPS, ARMS, "equifm")
        gate("frozen_guard", ok_wrong_backend and ok_right["study"] == "basecmp")

        # ---- the full plan de-duplicates ----------------------------------
        full = ts.plan_basecmp_full_cells(PROPS, ARMS, ok_right)
        # btvg_var's floor and free picks are identical (flat grid) -> one cell
        bv = [c for c in full if c[1] == "btvg_var"]
        gate("full_plan_dedup",
             len(bv) == 1 and len(set(full)) == len(full)
             and len([c for c in full if c[1] == "unguided"]) == 1)

        # ---- the SLURM array tiles the grid exactly ------------------------
        # basecmp_run.slurm splits the screen across 48 tasks by
        # (backend, property, start-time, arm group). The arithmetic there is
        # bash, so it is re-derived here and checked against the whole plan: a
        # task that also planned the reference window would duplicate 180 cells
        # and have several jobs writing one filename at once, and a gap would be
        # invisible in the picks, which are an argmax.
        import collections
        P3, BK = ["mu", "alpha", "gap"], ["fm", "equifm"]
        TS4, TREF = [0.05, 0.25, 0.5, 0.75], 2
        # the four groups basecmp_run.slurm uses: the two costly arms get a task
        # each, because at t_start=0.05 ~95 of 100 steps are guided and a group of
        # three would run past the 225-min guard
        GRP = [["plug", "tmpd"], ["lgd_mc", "tfg"], ["btvg"], ["btvg_var"]]
        NG = len(GRP)
        made = collections.Counter()
        for t in range(2 * NG * 3 * 4):
            b = BK[t % 2]; u = t // 2; g = u % NG; v = u // NG
            p = P3[v % 3]; k = v // 3
            am = (["unguided"] if (k == TREF and g == 0) else []) + GRP[g]
            for c in ts.plan_basecmp_cells([p], am, [TS4[k]]):
                made[(b,) + c] += 1
        whole = set()
        for b in BK:
            for c in ts.plan_basecmp_cells(P3, ts.backend_arms(b), TS4):
                whole.add((b,) + c)
        gate("slurm_screen_tiles_grid",
             set(made) == whole and max(made.values()) == 1
             and len(whole) == 2 * 363)

        # the same for the 18 full-run tasks. A three-property frozen file is
        # synthesised here: the one above was built from a mu-only screen, and
        # the array's property index runs over all three. `floor` and `free`
        # differ for plug and agree for btvg_var, so the de-duplication is
        # exercised rather than assumed.
        fz3 = {"study": "basecmp", "backend": "EquiFM",
               "select_metric": ts.BASECMP_SELECT,
               "frozen": {s: {a: {p: {"w": (1.0 if s == "floor" else 4.0)
                                      if a == "plug" else 0.25,
                                      "t_start": 0.5}
                                  for p in P3} for a in GUIDED}
                          for s in ts.BASECMP_SETS}}
        made_f = collections.Counter()
        for t in range(NG * 3 * 3):
            g = t % NG; u = t // NG
            p = P3[u % 3]; sd = u // 3
            am = (["unguided"] if g == 0 else []) + GRP[g]
            for c in ts.plan_basecmp_full_cells([p], am, fz3):
                made_f[(sd,) + c] += 1
        want_f = set()
        for sd in range(3):
            for c in ts.plan_basecmp_full_cells(P3, ARMS, fz3):
                want_f.add((sd,) + c)
        # plug's two sets differ -> 2 cells; every other guided arm agrees -> 1
        n_plug = len([c for c in made_f if c[2] == "plug" and c[0] == 0])
        n_bv = len([c for c in made_f if c[2] == "btvg_var" and c[0] == 0])
        gate("slurm_full_tiles_plan",
             set(made_f) == want_f and max(made_f.values()) == 1
             and n_plug == 6 and n_bv == 3)

        # ---- the feature-scale check, which no other guard covers ----------
        # build_pair's slope/MAD guard is fitted on REAL molecules, so it cannot
        # see a wrong sampler feature divisor -- that only changes what f_A/f_B
        # are fed at SCORING time. The signature is the oracle's mean on unguided
        # samples drifting off QM9's own mean, which is how the same bug was
        # caught before. QM9 train_a mu mean is 2.7038, MAD 1.1958; the threshold
        # is 1.0 MAD, measured from 80 real unguided cells whose worst legitimate
        # offset is 0.62. 2.70 is a clean generator; 4.30 is ~1.34 MAD out, the
        # size of the historical bug; 5.40 is a clean factor-of-2 error.
        ok_clean = bf.scale_sanity([cell("mu", "unguided", 1.0, 0.5, 0.02, 0.8,
                                         f_b_mean=2.70)], ["mu"])
        ok_bad = bf.scale_sanity([cell("mu", "unguided", 1.0, 0.5, 0.02, 0.8,
                                       f_b_mean=4.30)], ["mu"])
        ok_bad2 = bf.scale_sanity([cell("mu", "unguided", 1.0, 0.5, 0.02, 0.8,
                                        f_b_mean=5.40)], ["mu"])
        ok_edge = bf.scale_sanity([cell("mu", "unguided", 1.0, 0.5, 0.02, 0.8,
                                        f_b_mean=2.70 + 0.62 * 1.1958)], ["mu"])
        skipped = any("skipped" in x for x in ok_clean)
        flagged = lambda o: any("CHECK THE FEATURE SCALE" in x for x in o)
        gate("scale_check_flags_wrong_scale",
             skipped or (not flagged(ok_clean) and not flagged(ok_edge)
                         and flagged(ok_bad) and flagged(ok_bad2)))

        # ---- the refine array's 24 tasks are distinct and complete ---------
        # The refine plan's contents depend on where the picks landed, so what is
        # gated here is the bash arithmetic: every (backend, property, arm group)
        # is covered exactly once. It was one task per (backend, property) until
        # the sizer priced that at 182 of the 225 available minutes.
        seen_r = set()
        for t in range(2 * NG * 3):
            b = BK[t % 2]; u = t // 2
            g = u % NG; p = P3[u // NG]
            seen_r.add((b, p, tuple(GRP[g])))
        gate("slurm_refine_tiles",
             len(seen_r) == 2 * 3 * NG
             and seen_r == {(b, p, tuple(gr)) for b in BK for p in P3
                            for gr in GRP})

        # ---- delta cannot depend on the generator -------------------------
        # The guarantee that makes the two bases comparable is structural, not a
        # promise in a comment: `local_delta` is handed the oracle, the data and
        # a calibration, and there is no parameter through which a generator or
        # a sample could reach it. Assert that, so adding one has to break a gate.
        import inspect
        sig = set(inspect.signature(ts.local_delta).parameters)
        gate("delta_is_generator_free",
             sig == {"prop", "d", "raw_oracle", "slope", "intercept", "dev",
                     "k", "q", "h", "batch"}
             and not (sig & {"net", "generator", "backend", "samples", "coords"}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 46)
    for k, v in R.items():
        print("%-34s %s" % (k, "yes" if v else "NO"))
    print("-" * 46)
    ok = all(R.values())
    print("ALL PASS (%d gates)" % len(R) if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
