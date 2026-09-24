"""Gates for the transfer's DECISION logic, on synthetic cells with known answers.

No generator, no GPU: every cell is a hand-built JSON whose correct outcome is
fixed by construction, so the rules are tested rather than re-derived.

  freeze_refuses_partial      a hole in the base grid stops the freeze
  ext_plan_*                  the edge rule: a top-edge FR3a pick gets w x4, x16;
                              a bottom-edge pick w /4, /16; interior picks nothing
  freeze_refuses_before_ext   freeze stops until the edge cells exist
  freeze_uses_ext             an extension strength can win once present
  select_is_decoded           the pick follows the DECODED MAE when soft disagrees
  frozen_keys                 tfg_post_hoc False, decoded metric, grid recorded
  load_frozen_accepts         guidance_sweep.load_frozen reads the file
  floor_interp_*              in_band_at_floor: interpolated / not reached / below
  eqfreeze_*                  FR3a on the tune block + the frontier value
  eq_frozen_guard             an eqtune file is refused where a compare file is needed
                              and vice versa

Run: python proj1/tests/test_transfer_protocol.py      (CPU, seconds)
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

import transfer_sweep as ts                                        # noqa: E402
from guidance_sweep import load_frozen                             # noqa: E402

R = {}


def gate(name, ok):
    R[name] = bool(ok)


PROPS = ["mu"]
ARMS = ts.backend_arms("equifm")


def cell(prop, arm, tgt, w, mae, stab, mae_dec=None, ib=0.1, stage="compare",
         n=512, seed=20260922, block=None):
    mae_dec = mae if mae_dec is None else mae_dec
    return {"stage": stage, "target_name": tgt, "prop": prop, "arm": arm,
            "w": float(w), "n": n, "seed": seed, "backend": "EquiFM",
            "prov": {"fm_md5": "x"}, "grid": "native", "tau_max_guide": 0.5,
            "batch": 128, "dist_block": block, "n_nonfinite": 0,
            "prop_mae_eval": mae, "prop_rmse_eval": mae * 1.3,
            "prop_mae_eval_dec": mae_dec, "prop_rmse_eval_dec": mae_dec * 1.3,
            "mol_stability": stab, "in_band_fraction": ib,
            "in_band_fraction_dec": ib}


def write(d, cells):
    for i, c in enumerate(cells):
        with open(os.path.join(d, "tr__%s__%s__%s__w%g__%d.json"
                               % (c["prop"], c["arm"], c["target_name"], c["w"], i)),
                  "w") as fh:
            json.dump(c, fh)


def base_cells(skip=None):
    """Unguided stability 0.80 -> floor 0.72. Per arm, MAE falls with w and
    stability falls with w. lgd_mc clears the floor everywhere (pick at the TOP
    edge); btvg_var fails it everywhere (fallback to the most stable = the
    BOTTOM edge); every other arm crosses the floor between 0.5 and 1."""
    out = []
    for tgt in ("q50", "q90"):
        out.append(cell("mu", "unguided", tgt, 1.0, 2.0, 0.80))
        for a in ARMS:
            if a == "unguided":
                continue
            for w in ts.TRANSFER_STRENGTHS:
                mae = 2.0 - 0.2 * w ** 0.5
                if a == "lgd_mc":
                    stab = 0.79 - 0.001 * w
                elif a == "btvg_var":
                    stab = 0.70 - 0.01 * w
                else:
                    stab = 0.79 - 0.08 * w
                c = cell("mu", a, tgt, w, mae, stab)
                if (a, tgt, w) != skip:
                    out.append(c)
    return out


def args_for(d, stage="freeze", json_out=""):
    return argparse.Namespace(out_dir=d, backend="equifm", json_out=json_out,
                              block_start=ts.EQ_TUNE_START if stage.startswith("eq") else 0,
                              n=ts.EQ_TUNE_N if stage.startswith("eq") else 512)


def main():
    tmp = tempfile.mkdtemp(prefix="tp_")
    try:
        # ---- partial grid refused
        d = os.path.join(tmp, "partial")
        os.makedirs(d)
        write(d, base_cells(skip=("plug", "q90", 0.5)))
        gate("freeze_refuses_partial", ts.freeze(args_for(d), PROPS, ARMS) == 2)

        # ---- the edge rule
        d = os.path.join(tmp, "full")
        os.makedirs(d)
        cells = base_cells()
        write(d, cells)
        rows = [c for c in cells if c["target_name"] == "q90"]
        plan = ts.extension_plan(rows, PROPS, ARMS)
        hi, lo = max(ts.TRANSFER_STRENGTHS), min(ts.TRANSFER_STRENGTHS)
        gate("ext_plan_top_edge", ("mu", "lgd_mc", "q90", hi * 4) in plan
             and ("mu", "lgd_mc", "q90", hi * 16) in plan)
        gate("ext_plan_bottom_edge", ("mu", "btvg_var", "q90", lo / 4) in plan
             and ("mu", "btvg_var", "q90", lo / 16) in plan)
        gate("ext_plan_interior_nothing",
             not any(c[1] in ("plug", "tmpd", "tfg", "btvg") for c in plan)
             and len(plan) == 4)
        fz = os.path.join(tmp, "frozen.json")
        gate("freeze_refuses_before_ext",
             ts.freeze(args_for(d, json_out=fz), PROPS, ARMS) == 2
             and not os.path.exists(fz))

        # extension cells: lgd_mc keeps improving and clearing the floor at 16
        ext = []
        for (_, a, tgt, w) in plan:
            if a == "lgd_mc":
                ext.append(cell("mu", a, tgt, w, 2.0 - 0.2 * w ** 0.5, 0.79 - 0.001 * w))
            else:
                ext.append(cell("mu", a, tgt, w, 1.99, 0.70 - 0.01 * w))
        write(d, ext)
        rc = ts.freeze(args_for(d, json_out=fz), PROPS, ARMS)
        F = json.load(open(fz)) if rc == 0 else {}
        gate("freeze_passes_with_ext", rc == 0)
        gate("freeze_uses_ext", F.get("frozen_w", {}).get("lgd_mc", {}).get("mu") == hi * 16)
        # interior arm: the best MAE among floor-clearing strengths
        # stab = 0.79 - 0.08 w >= 0.72  <=>  w <= 0.875 -> pick 0.5
        gate("freeze_interior_pick", F.get("frozen_w", {}).get("plug", {}).get("mu") == 0.5)
        gate("frozen_keys", F.get("tfg_post_hoc") is False
             and F.get("select_metric") == "prop_mae_eval_dec"
             and F.get("grid_strengths") == ts.TRANSFER_STRENGTHS
             and len(F.get("extension", [])) == 4)
        try:
            load_frozen(fz, PROPS, ARMS, ["primary"])
            gate("load_frozen_accepts", True)
        except SystemExit:
            gate("load_frozen_accepts", False)

        # ---- the pick follows the DECODED MAE when soft disagrees
        d = os.path.join(tmp, "dec")
        os.makedirs(d)
        cells = base_cells()
        for c in cells:
            if c["arm"] == "tmpd" and c["target_name"] == "q90":
                # soft says 0.5 is best; decoded says 0.25 is best
                c["prop_mae_eval"] = 1.0 if c["w"] == 0.5 else 2.0
                c["prop_mae_eval_dec"] = 1.0 if c["w"] == 0.25 else 2.0
        write(d, cells)
        write(d, ext)
        fz2 = os.path.join(tmp, "frozen2.json")
        ts.freeze(args_for(d, json_out=fz2), PROPS, ARMS)
        gate("select_is_decoded",
             json.load(open(fz2))["frozen_w"]["tmpd"]["mu"] == 0.25)

        # ---- in_band_at_floor
        pts = [(0.1, 0.80, 0.10), (1.0, 0.76, 0.20), (4.0, 0.68, 0.40)]
        v = ts.in_band_at_floor(pts, 0.72)
        gate("floor_interp_value", v["kind"] == "interpolated"
             and abs(v["value"] - 0.30) < 1e-12 and v["between_w"] == [1.0, 4.0])
        v = ts.in_band_at_floor([(0.1, 0.8, 0.1), (1.0, 0.79, 0.3)], 0.72)
        gate("floor_interp_not_reached", v["kind"] == "not_reached"
             and v["value"] == 0.3)
        v = ts.in_band_at_floor([(0.1, 0.6, 0.1)], 0.72)
        gate("floor_interp_below", v["kind"] == "below_floor" and v["value"] is None)

        # ---- eqfreeze on synthetic tune cells
        d = os.path.join(tmp, "eqtune")
        os.makedirs(d)
        eq = [cell("mu", "unguided", "dist", 1.0, 2.0, 0.80, ib=0.10, stage="eqtune",
                   n=2000, seed=20261001, block=[10000, 12000])]
        for a in ARMS:
            if a == "unguided":
                continue
            for w in ts.EQ_STRENGTHS:
                eq.append(cell("mu", a, "dist", w, 2.0 - 0.1 * w, 0.80 - 0.02 * w,
                               ib=0.10 + 0.01 * w, stage="eqtune", n=2000,
                               seed=20261001, block=[10000, 12000]))
        write(d, eq[:-1])
        gate("eqfreeze_refuses_partial", ts.eqfreeze(args_for(d, "eqfreeze"), PROPS, ARMS) == 2)
        write(d, eq[-1:])
        fz3 = os.path.join(tmp, "frozen_eq.json")
        rc = ts.eqfreeze(args_for(d, "eqfreeze", json_out=fz3), PROPS, ARMS)
        E = json.load(open(fz3)) if rc == 0 else {}
        # stab = 0.80 - 0.02 w >= 0.72 <=> w <= 4 -> best MAE (largest w) = 4
        gate("eqfreeze_pick", E.get("frozen_w", {}).get("plug", {}).get("mu") == 4.0)
        # crossing between w=4 (0.72, clears) and w=8 (0.64): at the floor,
        # interpolation weight 0 -> in_band at w=4 = 0.14
        fv = E.get("frontier", {}).get("mu", {}).get("plug", {}).get("in_band_at_floor", {})
        gate("eqfreeze_frontier_value", fv.get("kind") == "interpolated"
             and abs(fv.get("value", -1) - 0.14) < 1e-9)
        gate("eqfreeze_keys", E.get("source_stage") == "eqtune"
             and E.get("target") == "dist" and E.get("source_block") == [10000, 12000])
        # ---- the edge rule on the equal-chemistry tune grid
        d = os.path.join(tmp, "eqedge")
        os.makedirs(d)
        eq2 = [cell("mu", "unguided", "dist", 1.0, 2.0, 0.80, ib=0.10, stage="eqtune",
                    n=2000, seed=20261001, block=[10000, 12000])]
        for a in ARMS:
            if a == "unguided":
                continue
            for w in ts.EQ_STRENGTHS:
                stab = 0.79 - 0.0001 * w if a == "lgd_mc" else 0.80 - 0.02 * w
                eq2.append(cell("mu", a, "dist", w, 2.0 - 0.1 * w ** 0.5, stab,
                                ib=0.10 + 0.01 * w, stage="eqtune", n=2000,
                                seed=20261001, block=[10000, 12000]))
        write(d, eq2)
        top = max(ts.EQ_STRENGTHS)
        plan2 = ts.extension_plan([c for c in eq2], PROPS, ARMS, ts.EQ_STRENGTHS, "dist")
        gate("eq_ext_plan_top_edge", sorted(plan2) == sorted(
            [("mu", "lgd_mc", "dist", top * 4), ("mu", "lgd_mc", "dist", top * 16)]))
        fz4 = os.path.join(tmp, "frozen_eq2.json")
        gate("eqfreeze_refuses_before_ext",
             ts.eqfreeze(args_for(d, "eqfreeze", json_out=fz4), PROPS, ARMS) == 2)
        write(d, [cell("mu", "lgd_mc", "dist", w, 2.0 - 0.1 * w ** 0.5, 0.79 - 0.0001 * w,
                       ib=0.10 + 0.001 * w, stage="eqtune", n=2000, seed=20261001,
                       block=[10000, 12000]) for (_, _, _, w) in plan2])
        rc = ts.eqfreeze(args_for(d, "eqfreeze", json_out=fz4), PROPS, ARMS)
        E2 = json.load(open(fz4)) if rc == 0 else {}
        gate("eqfreeze_uses_ext", rc == 0
             and E2["frozen_w"]["lgd_mc"]["mu"] == top * 16
             and len(E2.get("extension", [])) == 2)

        # guards: each file only where it belongs
        try:
            ts.load_eq_frozen(fz3, PROPS, ARMS, "equifm")
            ok1 = True
        except SystemExit:
            ok1 = False
        try:
            ts.load_eq_frozen(fz, PROPS, ARMS, "equifm")
            ok2 = False
        except SystemExit:
            ok2 = True
        try:
            load_frozen(fz3, PROPS, ARMS, ["primary"])
            ok3 = False
        except SystemExit:
            ok3 = True
        gate("eq_frozen_guard", ok1 and ok2 and ok3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("%-32s pass" % "check")
    print("-" * 40)
    for k, v in R.items():
        print("%-32s %s" % (k, "yes" if v else "NO"))
    print("-" * 40)
    ok = all(R.values())
    print("ALL PASS (%d gates)" % len(R) if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
