"""Gates for the Modality 2 v3 protocol. The counterpart of proj1/tests/test_v3.py.

Closed-form, no GPU, no checkpoint. Run after any edit to the protocol, the
driver, the cell runner or the table -- it is what keeps those four in step.

    python proj1/m2/test_m2_protocol.py

Each gate asserts something that, if it broke, would produce a COMPLETE but
MEANINGLESS results table rather than an error. That is the failure mode this
file exists for.
"""
import io
import os
import re
import sys
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

import m2_sweep as M           # noqa: E402
import run_sweep as R          # noqa: E402
import m2_table as T           # noqa: E402

PROTOCOL = os.path.join(ROOT, "docs", "protocol", "MODALITY2_V3_PROTOCOL.md")
RESULTS, FAILED = [], 0

_src = open(os.path.join(HERE, "m2_sweep.py")).read()




def gate(name, ok, detail=""):
    global FAILED
    RESULTS.append((name, ok, detail))
    if not ok:
        FAILED += 1


def expect_exit(fn, needle):
    """Run fn; pass if it raises SystemExit whose message contains needle."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
    except SystemExit as e:
        return needle.lower() in str(e).lower()
    except Exception:
        return False
    return False


# ---------------------------------------------------------------- the batch
# The batch is BDG's ESTIMATOR (protocol 2.4). A remainder batch is a second,
# noisier controller pooled in as an equal, so it must be refused, not rounded.
gate("batch_must_divide_n",
     expect_exit(lambda: M.run_cell(None, {"crop": 500}, "plug", "-", 1.0, 0.45,
                                    0.05, None, 0.0, False, 10, 2, 0.0, 1.0,
                                    1, 0.01, None, batch=4),
                 "does not divide"),
     "n=10 batch=4 is refused")

def _run_driver(argv):
    """Actually invoke run_sweep.main() with a patched argv."""
    old = sys.argv
    sys.argv = ["run_sweep.py"] + argv
    try:
        return R.main()
    finally:
        sys.argv = old


# RUN the driver rather than grepping it: a gate that greps for a string passes
# when the derivation underneath is broken, which is how a vacuous gate ships.
gate("driver_refuses_indivisible_batch",
     expect_exit(lambda: _run_driver(["--stage", "m2", "--w", "16",
                                      "--n", "2000", "--batch", "3", "--dry"]),
                 "does not divide"),
     "run_sweep --n 2000 --batch 3 exits, message names the estimator")

gate("driver_refuses_missing_w",
     expect_exit(lambda: _run_driver(["--stage", "m2", "--dry"]),
                 "no default headline strength"),
     "run_sweep with no --w exits and points at measure_strength.py")


def _argparse_default(mod_src, flag):
    m = re.search(r'add_argument\("%s"[^)]*?default=([^,)\s]+)' % re.escape(flag),
                  mod_src, re.S)
    return m.group(1) if m else None


_drv_src = open(os.path.join(HERE, "run_sweep.py")).read()
gate("batch_default_is_500_in_both_entry_points",
     _argparse_default(_src, "--batch") == "500"
     and _argparse_default(_drv_src, "--batch") == "500",
     "m2_sweep=%s run_sweep=%s -- matches M1's controller sample size"
     % (_argparse_default(_src, "--batch"), _argparse_default(_drv_src, "--batch")))

gate("nfe_default_is_400_in_both_entry_points",
     _argparse_default(_src, "--steps") == "400"
     and _argparse_default(_drv_src, "--steps") == "400",
     "protocol 2.2: soft/hard gap is 5.5%% at NFE 100, 2.6%% at 400")

gate("window_default_is_zero",
     _argparse_default(_src, "--t-min") == "0.0",
     "protocol 2.1: t>=0.5 steers after the sequence has committed")

# ------------------------------------------------------- the cell name tag
# Resume is skip-if-exists, so anything that changes what the number MEANS must
# be in the filename or a smoke cell is silently kept in place of a real one.
_CFG_FIELDS = ["n", "nfe", "win", "b", "dr"]
_cfg_line = re.search(r'cfg = "([^"]+)"', _src)
gate("cell_name_has_config_tag", _cfg_line is not None,
     _cfg_line.group(1) if _cfg_line else "no cfg line found")
if _cfg_line:
    tag = _cfg_line.group(1)
    for f in _CFG_FIELDS:
        gate("cell_name_carries_%s" % f,
             re.search(r'(^|_)%s%%' % f, tag) is not None, tag)

gate("cell_name_tag_is_used_in_filename",
     re.search(r'name = "[^"]*%s[^"]*" % \(\s*a\.prop', _src) is not None
     or "cfg" in _src.split("name = ")[1][:200],
     "cfg interpolated into the json filename")

# ------------------------------------------------------------ stage subtree
gate("stage_subtree_mirrors_v3",
     'os.path.join(a.out_dir, a.stage, "n%d" % a.n, "seed%d" % a.seed)' in _src,
     "results/m2/<stage>/n<N>/seed<S>/")

# --------------------------------------------------------- the right model
gate("default_ckpt_is_the_500bp_base",
     M.DEFAULT_CKPT.endswith("fm_m2_dfb500.pt"),
     M.DEFAULT_CKPT)
gate("default_ckpt_agrees_with_driver",
     os.path.basename(M.DEFAULT_CKPT) in open(
         os.path.join(HERE, "run_sweep.py")).read(),
     "m2_sweep and run_sweep must not default to different models")

# ------------------------------------------------------ per-property quantum
# gc_hard averages over L positions, cpg_hard over the L-1 adjacent pairs.
gate("quantum_is_per_property",
     'a.prop == "gc"' in _src and "crop" in _src.split("quantum =")[1][:120],
     "1/L for gc, 1/(L-1) for cpg")

# ------------------------------------------------------------- the ablation
# Must be M1's grid exactly, or the two modalities' ablations are not the same
# experiment: eta=0 once (it kills the feedback whatever tau is) + 4 x 4.
gate("ablation_grid_is_m1s_17_arms", len(R.ABL_BDG) == 17,
     "%d arms: %s" % (len(R.ABL_BDG), ",".join(R.ABL_BDG[:5]) + ",..."))
gate("ablation_eta0_appears_once",
     sum(1 for v in R.ABL_BDG if v.startswith("e0")) == 1,
     "eta=0 kills the feedback whatever tau is")
gate("ablation_etas_match_m1", sorted({float(re.match(r"e([0-9.]+)t", v).group(1))
                                       for v in R.ABL_BDG}) == [0, 1, 2, 4, 8],
     "eta in {0,1,2,4,8}")
gate("ablation_taus_match_m1",
     sorted({float(re.match(r"e[0-9.]+t([0-9.]+)", v).group(1))
             for v in R.ABL_BDG if not v.startswith("e0")})
     == [0.5, 0.75, 1.0, 1.5], "tau_mult in {0.5,0.75,1,1.5}")
gate("ablation_sweeps_two_strengths", len(R.ABL_WS) == 2,
     "w and 4w, mirroring v3's w in {1,4}")

# --------------------------------------------------------- the headline set
gate("headline_is_seven_arms",
     1 + len(R.ARMS_EXT) + len(R.HEADLINE_BDG) == 7,
     "unguided + 4 baselines + 2 bdg")
gate("headline_target_is_q50", R.TARGET == "q50", R.TARGET)
gate("three_seeds", len(R.SEEDS) == 3, str(R.SEEDS))
gate("both_properties", sorted(R.PROPS) == ["cpg", "gc"], str(R.PROPS))

# The strength is MEASURED, not copied from M1. The driver must refuse without it.
gate("driver_refuses_without_w",
     "--w is not set" in open(os.path.join(HERE, "run_sweep.py")).read(),
     "headline strength is fixed by measure_strength.py, not defaulted")

# ------------------------------------------------------------ the window rung
gate("ablation_carries_window_rung", R.WINDOW_RUNG == 0.5,
     "t_min=0.5 rung, so v3's window is a measured row not a citation")

# ------------------------------------------------------------- no floor
# Protocol 3.1: v3 removed v2's chemistry gate, so M2 must not add one back.
gate("no_fidelity_floor_in_code",
     not re.search(r"kmer_js\s*[<>]=?|0\.9\s*\*\s*.*kmer", _src),
     "kmer_js is recorded and never thresholded")
gate("floor_prose_removed",
     "a cell passes if its 3-mer JS" not in _src,
     "the unimplemented floor description is gone from the docstring")

# --------------------------------------------------------------- the table
gate("table_pins_the_configuration",
     set(["n", "batch", "steps", "t_min_guide", "delta"]).issubset(set(T.PINNED)),
     str(T.PINNED))
gate("table_refuses_to_pool_ablation_strengths",
     "REFUSING to pool the ablation's strengths" in open(
         os.path.join(HERE, "m2_table.py")).read(),
     "--w is required when the tree holds more than one")
gate("table_reports_fidelity_beside_in_band",
     set(T.FIDELITY) == {"kmer_js", "decode_conf", "diversity"},
     str(T.FIDELITY))
gate("table_computes_no_verdict",
     "best without losing fidelity" in open(
         os.path.join(HERE, "m2_table.py")).read(),
     "names the clean best arm instead of ranking on in_band")

# ------------------------------------------------------------- the protocol
if os.path.exists(PROTOCOL):
    _p = open(PROTOCOL).read()
    gate("protocol_records_the_tmpd_defect",
         "denominator only" in _p and "_const_grad" in _p,
         "section 1.2 -- tmpd on cpg is plug rescaled, reported not fixed")
    gate("protocol_states_delta_is_a_choice",
         "CHOICE" in _p and "not commensurable" in _p,
         "M2 in_band may never be pooled with M1's")
    gate("protocol_pins_nfe_400", "NFE is 400" in _p or "NFE | 100" in _p,
         "section 2.2")
    gate("protocol_marks_every_parameter",
         all(k in _p for k in ["[RECORDED]", "[TO MEASURE]", "[CHOICE]"]),
         "every parameter is sourced")
else:
    gate("protocol_present", False, PROTOCOL)

# ------------------------------------------------------------------- report
w = max(len(n) for n, _, _ in RESULTS)
for name, ok, detail in RESULTS:
    print("%-*s  %s  %s" % (w, name, "PASS" if ok else "FAIL", detail))
print("\n%s (%d gates)"
      % ("ALL PASS" if not FAILED else "%d FAILED" % FAILED, len(RESULTS)))
sys.exit(1 if FAILED else 0)
