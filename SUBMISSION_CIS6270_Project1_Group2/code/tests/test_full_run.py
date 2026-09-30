"""Gates for the full-scale run: the go/no-go gate, the frozen-strength plan,
the evaluator's large-n path, and the paired-comparison bookkeeping.

Each gate states the failure it exists to catch. Tiny tensors, no model and no
dataset: safe on a login node.

Run: python proj1/tests/test_full_run.py
"""
import io
import json
import os
import shutil
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "proj1", "scripts"), os.path.join(ROOT, "proj1", "src")]
import check_fullrun_go as cg  # noqa: E402
import evaluation as ev  # noqa: E402
import full_run_table as frt  # noqa: E402
import guidance_sweep as gs  # noqa: E402

R = {}


def gate(name, ok, detail=""):
    R[name] = (bool(ok), detail)


# ---------------------------------------------------------------- helpers
def cell(prop, arm, tgt, w, mae, n=512, rmse=None, nonfinite=0, seed=20260921,
         mstab=0.40):
    return {"prop": prop, "arm": arm, "target_name": tgt, "w": w,
            "stage": "compare", "n": n, "prop_mae_eval": mae,
            "prop_rmse_eval": rmse if rmse is not None else mae * 1.25,
            "n_nonfinite": nonfinite, "seed": seed, "mol_stability": mstab}


def write_grid(d, tgt, mae_fn, skip=(), diverge=(), stab_fn=None):
    """The whole compare grid at one target. mae_fn(prop, arm, w) -> MAE;
    stab_fn(prop, arm, w) -> mol_stability (default 0.40 everywhere, so every
    strength clears the 0.9 x unguided floor of 0.36)."""
    for prop in cg.PROPS:
        for (arm, w) in cg.expected_grid(cg.sweep_constants()):
            if (prop, arm, w) in skip:
                continue
            nf = 3 if (prop, arm, w) in diverge else 0
            r = cell(prop, arm, tgt, w, mae_fn(prop, arm, w), nonfinite=nf,
                     mstab=stab_fn(prop, arm, w) if stab_fn else 0.40)
            name = gs.cell_name(prop, arm, tgt, w, gs.V2_WIN, "cmp")
            with open(os.path.join(d, name), "w") as fh:
                json.dump(r, fh)


def run_check(d, tgt, out):
    argv = sys.argv
    sys.argv = ["check_fullrun_go", "--sweep-dir", d, "--target", tgt,
                "--json-out", out]
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            rc = cg.main()
    finally:
        sys.argv = argv
    return rc, buf.getvalue()


def main():
    # ---- 1. the checker reads the SAME grid the sweep plans
    #      (catches: STRENGTHS edited in the sweep, checker still on the old list)
    c = cg.sweep_constants()
    gate("constants_match_sweep",
         c["STRENGTHS"] == gs.STRENGTHS and c["COMPARE_SET"] == gs.COMPARE_SET
         and c["DEFAULT_W"] == gs.DEFAULT_W,
         "%s / %s" % (c["STRENGTHS"], c["COMPARE_SET"]))
    planned = {(a, float(w)) for (p, a, t, w, win, v)
               in gs.plan_compare_cells(["mu"]) if t == "q50"}
    gate("expected_grid_equals_compare_plan", planned == cg.expected_grid(c),
         "%d planned vs %d expected" % (len(planned), len(cg.expected_grid(c))))
    gate("compare_set_is_checker_arms", tuple(gs.COMPARE_SET) == cg.ALL_ARMS)

    tmp = tempfile.mkdtemp(prefix="fullrun_gate_")
    try:
        # plug best at w=0.5, btvg best at w=0.25, everything else worse;
        # btvg 0.01 worse than plug => a tie, FR1 passes
        def mae_ok(prop, arm, w):
            base = {"plug": 1.00, "btvg": 1.01}.get(arm, 1.5)
            best_w = {"plug": 0.5, "btvg": 0.25}.get(arm, 1.0)
            return base + 0.1 * abs(w - best_w)

        # ---- 2. complete grid: pass, and the frozen strengths are the argmins
        d = os.path.join(tmp, "complete")
        os.makedirs(d)
        write_grid(d, "q90", mae_ok)
        out = os.path.join(tmp, "fz.json")
        rc, log = run_check(d, "q90", out)
        fz = json.load(open(out)) if os.path.exists(out) else {}
        gate("complete_grid_passes", rc == 0, "rc=%s" % rc)
        gate("frozen_is_best_mae_strength",
             fz.get("frozen_w_mae", {}).get("plug", {}).get("alpha") == 0.5
             and fz["frozen_w_mae"]["btvg"]["gap"] == 0.25
             and fz["frozen_w_mae"]["unguided"]["mu"] == 1.0
             and fz["frozen_w"] == fz["frozen_w_mae"],   # all clear the floor
             str(fz.get("frozen_w_mae", {}).get("plug")))
        gate("frozen_json_is_atomic", not os.path.exists(out + ".tmp"))

        # ---- 3. ONE missing strength refuses, writes nothing
        #      (catches: the old one-cell-per-arm completeness test, which
        #      would freeze a best-of-partial strength)
        d = os.path.join(tmp, "hole")
        os.makedirs(d)
        write_grid(d, "q90", mae_ok, skip={("alpha", "tmpd", 4.0)})
        out2 = os.path.join(tmp, "fz_hole.json")
        rc, log = run_check(d, "q90", out2)
        gate("one_hole_refuses", rc == 2 and not os.path.exists(out2), "rc=%s" % rc)
        gate("hole_is_named", "alpha/tmpd/w4" in log)

        # ---- 4. a diverged cell is PRESENT (grid complete) but never best,
        #      even when its MAE is the smallest on disk
        d = os.path.join(tmp, "diverged")
        os.makedirs(d)

        def mae_div(prop, arm, w):
            if (prop, arm, w) == ("mu", "plug", 4.0):
                return 0.001                     # tempting, and diverged
            return mae_ok(prop, arm, w)
        write_grid(d, "q90", mae_div, diverge={("mu", "plug", 4.0)})
        out3 = os.path.join(tmp, "fz_div.json")
        rc, log = run_check(d, "q90", out3)
        fz3 = json.load(open(out3)) if os.path.exists(out3) else {}
        gate("diverged_counts_as_present", rc == 0, "rc=%s" % rc)
        gate("diverged_never_best",
             fz3.get("frozen_w", {}).get("plug", {}).get("mu") == 0.5,
             str(fz3.get("frozen_w", {}).get("plug")))

        # ---- 5. FR1 fail => rc 1, and the strengths are STILL written
        #      (the job needs them for q90 even when q90's FR1 fails)
        d = os.path.join(tmp, "fail")
        os.makedirs(d)

        def mae_bad(prop, arm, w):
            if arm == "btvg":
                return 3.0                       # far worse than plug's 1.0
            return mae_ok(prop, arm, w)
        write_grid(d, "q90", mae_bad)
        out4 = os.path.join(tmp, "fz_fail.json")
        rc, log = run_check(d, "q90", out4)
        fz4 = json.load(open(out4)) if os.path.exists(out4) else {}
        gate("fr1_fail_returns_10_not_1", rc == cg.RC_FAIL == 10, "rc=%s" % rc)
        gate("fr1_fail_still_writes_strengths",
             fz4.get("fr1_pass") is False and "btvg" in fz4.get("frozen_w", {}))

        # ---- 6. the full-run plan: one cell per arm, at the frozen w
        cells = gs.plan_full_cells(["mu", "gap"], gs.COMPARE_SET, fz)
        gate("full_plan_one_cell_per_arm",
             len(cells) == 2 * len(gs.COMPARE_SET)
             and len({(p, a) for (p, a, *_r) in cells}) == len(cells))
        gate("full_plan_uses_frozen_w",
             all(w == float(fz["frozen_w"][a][p]) for (p, a, t, w, win, v) in cells))
        gate("full_plan_is_dist_at_v2_window",
             all(t == gs.DIST_TARGET and win == gs.V2_WIN and v == "full"
                 for (p, a, t, w, win, v) in cells))
        names = [gs.cell_name(*c) for c in cells]
        gate("full_names_disjoint_from_screen",
             all(n.endswith("__full.json") for n in names))
        bad = os.path.join(tmp, "bad.json")
        json.dump({"frozen_w": {"plug": {"mu": 1.0}}, "source_stage": "compare",
                   "target": "q90"}, open(bad, "w"))
        try:
            gs.load_frozen(bad, ["mu"], gs.COMPARE_SET)
            refused = False
        except SystemExit as e:
            refused = "btvg/mu" in str(e)
        gate("incomplete_frozen_file_refused", refused)
        # FR3-corrected: a COMPLETE file frozen at q50 is still the wrong one
        q50f = os.path.join(tmp, "q50.json")
        json.dump(dict(fz, target="q50"), open(q50f, "w"))
        try:
            gs.load_frozen(q50f, ["mu"], gs.COMPARE_SET)
            refused = False
        except SystemExit as e:
            refused = "q90" in str(e)
        gate("q50_frozen_file_refused", refused)

        # ---- 6b. FR3a: best MAE AMONG strengths clearing the chemistry floor
        #      (catches: the headline set taking a floor-failing w=4, which
        #      is what FR3 alone picked on the real q90 screen)
        d = os.path.join(tmp, "floor")
        os.makedirs(d)

        def mae_edge(prop, arm, w):               # everyone's MAE falls with w
            return 2.0 - 0.1 * w

        def stab_edge(prop, arm, w):
            if arm == "unguided":
                return 0.40                       # floor = 0.36
            if arm == "plug":
                return 0.30 if w >= 2.0 else 0.39     # w=2,4 fail
            if arm == "tmpd":
                return 0.10                       # nothing passes
            if arm == "lgd_mc":
                return 0.36 if w == 4.0 else 0.40     # exactly AT the floor passes
            return 0.40
        write_grid(d, "q90", mae_edge, stab_fn=stab_edge)
        out7 = os.path.join(tmp, "fz_floor.json")
        rc, log = run_check(d, "q90", out7)
        fz7 = json.load(open(out7)) if os.path.exists(out7) else {}
        W7, M7 = fz7.get("frozen_w", {}), fz7.get("frozen_w_mae", {})
        gate("fr3a_best_mae_among_floor_passing",
             M7.get("plug", {}).get("mu") == 4.0 and W7.get("plug", {}).get("mu") == 1.0,
             "mae %s / fr3a %s" % (M7.get("plug"), W7.get("plug")))
        gate("fr3a_floor_is_inclusive",
             W7.get("lgd_mc", {}).get("gap") == 4.0, str(W7.get("lgd_mc")))
        gate("fr3a_fallback_most_stable_smallest_w",
             W7.get("tmpd", {}).get("alpha") == 0.01
             and "tmpd/alpha" in fz7.get("fell_back", []),
             "%s %s" % (W7.get("tmpd"), fz7.get("fell_back")))

        # ---- 6c. secondary tasks run ONLY cells the primary tasks do not
        #      (catches: two concurrent tasks writing the same cell)
        prim = set(gs.plan_full_cells(["mu", "alpha", "gap"], gs.COMPARE_SET, fz7,
                                      ["primary"]))
        extra = set(gs.plan_full_cells(["mu", "alpha", "gap"], gs.COMPARE_SET, fz7,
                                       ["mae"], ["primary"]))
        both = set(gs.plan_full_cells(["mu", "alpha", "gap"], gs.COMPARE_SET, fz7,
                                      ["primary", "mae"]))
        gate("secondary_disjoint_from_primary", prim and extra and not (prim & extra))
        gate("primary_plus_secondary_is_both_sets", (prim | extra) == both,
             "%d + %d vs %d" % (len(prim), len(extra), len(both)))
        gate("secondary_is_exactly_the_differing_cells",
             {(p, a) for (p, a, *_r) in extra}
             == {(p, a) for a in gs.COMPARE_SET for p in ("mu", "alpha", "gap")
                 if W7[a][p] != M7[a][p]})

        # ---- 5b. a CRASH must never look like an FR1 verdict
        #      (catches: Python's own exit code 1 == the old FR1-fail code)
        d = os.path.join(tmp, "crash")
        shutil.copytree(os.path.join(tmp, "complete"), d)
        json.dump([1, 2, 3], open(os.path.join(d, "zz_stray.json"), "w"))
        argv = sys.argv
        sys.argv = ["check_fullrun_go", "--sweep-dir", d, "--target", "q90"]
        buf = io.StringIO()
        try:
            with redirect_stdout(buf), redirect_stderr(io.StringIO()):
                rc = cg._entry()
        finally:
            sys.argv = argv
        gate("crash_exits_3_not_fail", rc == cg.RC_ERROR == 3, "rc=%s" % rc)

        # ---- 5c. ties break on the smaller w, not on listing order
        d = os.path.join(tmp, "tie")
        os.makedirs(d)

        def mae_tie(prop, arm, w):
            if arm == "btvg" and w in (2.0, 4.0, 0.5):
                return 0.9                        # three bit-identical bests
            return mae_ok(prop, arm, w)
        write_grid(d, "q90", mae_tie)
        out5 = os.path.join(tmp, "fz_tie.json")
        # NTFS lists alphabetically and w0.5 < w2 < w4 alphabetically too, so
        # without this the gate passes even with no tie-break at all. Reverse
        # the listing: order-dependent code now picks w4.
        real_glob = cg.glob

        class _Rev:
            @staticmethod
            def glob(pat):
                return sorted(real_glob.glob(pat), reverse=True)
        cg.glob = _Rev
        try:
            rc, log = run_check(d, "q90", out5)
        finally:
            cg.glob = real_glob
        fz5 = json.load(open(out5)) if os.path.exists(out5) else {}
        gate("tie_breaks_to_smallest_w",
             fz5.get("frozen_w", {}).get("btvg", {}).get("mu") == 0.5,
             str(fz5.get("frozen_w", {}).get("btvg")))

        # ---- 5d. a finite-flagged cell with a NaN MAE is never best
        d = os.path.join(tmp, "nanmae")
        os.makedirs(d)

        def mae_nan(prop, arm, w):
            return float("nan") if (arm, w) == ("tmpd", 0.01) else mae_ok(prop, arm, w)
        write_grid(d, "q90", mae_nan)
        out6 = os.path.join(tmp, "fz_nan.json")
        rc = None
        for order in (False, True):            # BOTH listing orders must hold
            real_glob = cg.glob

            class _Ord:
                @staticmethod
                def glob(pat, _rev=order):
                    return sorted(real_glob.glob(pat), reverse=_rev)
            cg.glob = _Ord
            try:
                rc_o, log = run_check(d, "q90", out6)
            finally:
                cg.glob = real_glob
            fz_o = json.load(open(out6)) if os.path.exists(out6) else {}
            ok_o = rc_o == 0 and fz_o.get("frozen_w_mae", {}).get("tmpd", {}).get("gap") == 1.0
            rc = rc_o if (rc is None or not ok_o) else rc
            if not ok_o:
                break
        fz6 = json.load(open(out6)) if os.path.exists(out6) else {}
        gate("nan_mae_never_best",
             rc == 0 and ok_o and fz6["frozen_w_mae"]["tmpd"]["gap"] == 1.0,
             "rc=%s %s" % (rc, fz6.get("frozen_w", {}).get("tmpd")))

        # ---- 5e. a grid mixing sample sizes is refused, not blended
        d = os.path.join(tmp, "mixed_n")
        shutil.copytree(os.path.join(tmp, "complete"), d)
        one = os.path.join(d, gs.cell_name("mu", "plug", "q90", 2.0, gs.V2_WIN, "cmp"))
        r = json.load(open(one))
        r["n"] = 4096
        json.dump(r, open(one, "w"))
        rc, log = run_check(d, "q90", os.path.join(tmp, "fz_mixed.json"))
        gate("mixed_n_refused", rc == cg.RC_INCOMPLETE and "INCONSISTENT" in log,
             "rc=%s" % rc)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ---- 7. the evaluator's sliced path equals the single call, bit for bit
    #      on this per-molecule stub, and per_mol lines up with the samples
    torch.manual_seed(3)
    B, N = 11, 6
    mask = torch.ones(B, N)
    mask[:, -2:] = 0.0
    mask[0, 3] = 0.0
    coords = torch.randn(B, N, 3) * mask.unsqueeze(-1)
    feats = torch.nn.functional.one_hot(torch.randint(0, 4, (B, N)), 5).float()
    feats = feats * mask.unsqueeze(-1)
    coords[4, 0, 0] = float("nan")               # one exploded sample

    class Stub(torch.nn.Module):
        calls = 0

        def forward(self, c, f, m):
            Stub.calls += 1
            return (c.nan_to_num() ** 2 * m.unsqueeze(-1)).sum((1, 2)) + f.sum((1, 2))

        def embed(self, c, f, m):
            x = torch.cat([c.nan_to_num(), f], -1) * m.unsqueeze(-1)
            return x.sum(1)

    types = ["H", "C", "N", "O", "F"]
    y = torch.linspace(0.5, 3.0, B)
    old = ev.EVAL_CHUNK
    try:
        ev.EVAL_CHUNK = 1024
        r1 = ev.evaluate_samples(coords, feats, mask, types, Stub(), Stub(), y, 0.7,
                                 per_mol=True)
        ev.EVAL_CHUNK = 3                         # 11 = 3+3+3+2: uneven last slice
        Stub.calls = 0
        r2 = ev.evaluate_samples(coords, feats, mask, types, Stub(), Stub(), y, 0.7,
                                 per_mol=True)
        # f_A + f_B, on the soft AND the decoded features, 4 slices each
        sliced_calls = Stub.calls
    finally:
        ev.EVAL_CHUNK = old
    keys = [k for k in r1 if k not in ("_per_mol", "smiles_sample")]
    gate("chunked_path_actually_slices", sliced_calls == 16, "calls=%d" % sliced_calls)
    gate("chunked_eval_equals_single_call",
         all(r1[k] == r2[k] or (r1[k] != r1[k] and r2[k] != r2[k]) for k in keys)
         and torch.equal(r1["_per_mol"]["f_B"], r2["_per_mol"]["f_B"]),
         str([k for k in keys if r1[k] != r2[k]]))
    pm = r1["_per_mol"]
    gate("per_mol_rows_match_samples",
         all(len(pm[k]) == B for k in pm)
         and bool(pm["finite"][4]) is False and int(pm["finite"].sum()) == B - 1
         and pm["n_atoms"].tolist()[0] == 3 and pm["n_atoms"].tolist()[1] == 4)
    ib_pm = (((pm["f_B"] - pm["y"]).abs() <= 0.7) & pm["finite"]).float().mean().item()
    gate("per_mol_reproduces_in_band", abs(ib_pm - r1["in_band_fraction"]) < 1e-12,
         "%.6f vs %.6f" % (ib_pm, r1["in_band_fraction"]))
    r3 = ev.evaluate_samples(coords, feats, mask, types, Stub(), Stub(), y, 0.7)
    gate("per_mol_off_by_default", "_per_mol" not in r3)
    r1.pop("_per_mol")
    try:
        json.dumps(r1)
        ser = True
    except TypeError:
        ser = False
    gate("row_json_serialisable_after_pop", ser)

    # ---- 8. paired z: pairs only on identical molecule order; an arm against
    #      itself is exactly zero difference
    a = {"f_B": torch.tensor([1.0, 2.0, 3.0, 4.0]), "y": torch.tensor([1.1, 2.5, 2.0, 4.0]),
         "finite": torch.tensor([True, True, True, True]),
         "mol_idx": torch.tensor([7, 8, 9, 10])}
    b = dict(a, f_B=torch.tensor([1.5, 2.1, 3.9, 4.3]))
    z = frt.paired_z(a, b, 0.3, "ib")
    gate("paired_z_sign", z > 0, "z=%.3f (a in band on 3, b on 1)" % z)
    gate("paired_self_is_undefined_not_significant",
         frt.paired_z(a, a, 0.3, "ib") != frt.paired_z(a, a, 0.3, "ib"))
    try:
        frt.paired_z(a, dict(b, mol_idx=torch.tensor([7, 8, 10, 9])), 0.3, "ib")
        caught = False
    except SystemExit:
        caught = True
    gate("paired_refuses_misaligned_rows", caught)

    # ---- 9. FR5 verdicts follow the saved rubric
    #      (catches: MAE alone declaring a win; a floor-failing arm winning;
    #      a significant contradiction silently resolved in btvg's favour)
    v = frt.verdict
    gate("verdict_in_band_decides", v(3.5, 0.0, True, True, "plug") == "btvg beats"
         and v(-3.5, 0.0, True, True, "plug") == "plug beats btvg")
    gate("verdict_mae_alone_is_a_tie", v(0.5, -9.0, True, True, "plug") == "tie")
    gate("verdict_contradiction_is_mixed",
         v(3.5, 3.5, True, True, "plug").startswith("mixed")
         and v(-3.5, -3.5, True, True, "plug").startswith("mixed"))
    nan = float("nan")
    gate("zero_se_is_a_tie_not_a_crash",               # both arms at in_band 0
         frt._z(0.0, 0.0) != frt._z(0.0, 0.0) and v(nan, nan, True, True, "plug") == "tie")
    gate("verdict_floor_disqualifies",
         v(9.0, -9.0, False, True, "plug") == "btvg fails floor"
         and v(9.0, -9.0, True, False, "plug") == "plug fails floor")

    # ---- 10. size-conditional reference: each row gets its own size's mean
    na = torch.tensor([9] * 12 + [20] * 12 + [5] * 3, dtype=torch.int32)
    fb = torch.cat([torch.full((12,), 60.0), torch.full((12,), 90.0),
                    torch.full((3,), 1000.0)])
    fin = torch.ones(27, dtype=torch.bool)
    fin[0] = False
    fb[0] = float("nan")                          # excluded from its group
    ref = frt.size_reference({"f_B": fb, "finite": fin, "n_atoms": na})
    glob_m = fb[fin].double().mean().item()
    gate("size_reference_per_group",
         abs(ref[1].item() - 60.0) < 1e-9 and abs(ref[0].item() - 60.0) < 1e-9
         and abs(ref[13].item() - 90.0) < 1e-9
         and abs(ref[25].item() - glob_m) < 1e-9,       # 3 rows < MIN: global
         "%s" % ref[[0, 1, 13, 25]].tolist())

    print("%-42s %s" % ("gate", "pass"))
    print("-" * 60)
    ok = True
    for k, (p, det) in R.items():
        ok = ok and p
        print("%-42s %-4s %s" % (k, "yes" if p else "NO", "" if p else det))
    print("-" * 60)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
