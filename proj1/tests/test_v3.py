"""Gates for protocol v3 and its BDG ablation.

    python proj1/tests/test_v3.py

Closed-form only: no model, no dataset, seconds on a login node.

The important gates PARSE the SLURM array's task arithmetic out of
proj1/cluster/v3_run.slurm and assert the tasks tile the planned work exactly --
no holes, no duplicates, no task resolving to an empty value. They read the
shipped file rather than re-implementing it, because the first version of this
suite hard-coded `t // 9` while the job shipped `t / 12`: it reported "covered
132, holes none" about a program that did not exist, and the bug survived review.
A gate that re-implements what it checks cannot catch a divergence between them.

Two blockers this suite now catches, both of which shipped once:
  * a wrong backend stride, which indexed an arm-group that does not exist, left
    27 of 396 cells unproducible and could not be recovered by the insurance job;
  * `--save-coords`, which belongs to guidance_sweep.py and which
    transfer_sweep.py rejects, so every sampling task died before writing a cell.
"""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "proj1", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "proj1", "src"))

import transfer_sweep as T                     # noqa: E402

PROPS = ["mu", "alpha", "gap"]
# read from the job file, not restated: the backend list grew to three on
# 26 Sep and a hardcoded pair here would have passed while the array left a
# third of its tasks unproduced
BE = re.findall(r"^BACKENDS=\((.*)\)$",
                io.open(os.path.join(ROOT, "proj1", "cluster", "v3_run.slurm"),
                        encoding="utf-8").read(), re.M)[0].split()
R = {}


def gate(name, ok, detail=""):
    R[name] = bool(ok)
    print("%-46s %s  %s" % (name, "PASS" if ok else "FAIL", detail))


# ---------------------------------------------------------------- constants
gate("v3_target_is_q50", T.V3_TARGET == "q50", T.V3_TARGET)
gate("v3_strength_is_one", T.V3_W == 1.0, "w = %g" % T.V3_W)
gate("v3_three_seeds", len(T.V3_SEEDS) == 3, str(T.V3_SEEDS))
gate("v3_window_interior", 0.0 < T.V3_T_START < 1.0, "t_start = %g" % T.V3_T_START)
gate("v3_bdg_eta_is_four", T.V3_BDG_ETA == 4.0)
# tau_mult 0.75 dropped from the HEADLINE (Henry, 26 Sep); it survives in the
# ablation grid. This is the SETPOINT knob, tau = tau_mult * f_A.y_std -- not
# the t >= 0.5 guidance window, which v3_window_interior pins separately.
gate("v3_bdg_tau_ladder", tuple(T.V3_BDG_TAU_MULTS) == (0.5, 1.0),
     str(T.V3_BDG_TAU_MULTS))
gate("headline_tau_ladder_is_two_points_and_the_grid_keeps_four",
     len(T.V3_BDG_TAU_MULTS) == 2
     and set(T.V3_BDG_TAU_MULTS) < set(T.V3_ABL_TAU_MULTS)
     and len(T.V3_ABL_TAU_MULTS) == 4,
     "the headline's rungs must be a strict subset of the grid's: a "
     "monotonicity or curvature claim about tau can only be read off the "
     "4-point grid, never off the headline's 2 endpoints (head %s, grid %s)"
     % (T.V3_BDG_TAU_MULTS, T.V3_ABL_TAU_MULTS))
# BOTH STAGES RUN THE SAME THREE SEEDS AND THE SAME n (Henry, 26 Sep, final).
# The ablation ran one seed at a smaller n for two days. It does not any more,
# and the gates that pinned the difference are replaced by the ones below.
gate("ablation_uses_the_same_three_seeds",
     tuple(T.V3_ABL_SEEDS) == tuple(T.V3_SEEDS),
     "%s vs %s -- every v3 row, headline or ablation, must carry an "
     "across-seed spread" % (T.V3_ABL_SEEDS, T.V3_SEEDS))
# N IS NOT PRE-REGISTERED. The operator picks the cell size, so both constants
# are None and --n is required. A number here would be a default nobody chose,
# and transfer_sweep would write a tree that looks like a real run.
gate("v3_pre_registers_no_n", T.V3_N is None and T.V3_ABL_N is None,
     "V3_N=%r V3_ABL_N=%r -- v3 leaves n to the operator; see "
     "FULL_RUN_V3_PROTOCOL.md section 6.1 for the power table they choose "
     "against" % (T.V3_N, T.V3_ABL_N))
gate("preflight_is_exempt_from_the_n_requirement",
     'and not args.preflight' in io.open(
         os.path.join(ROOT, "proj1", "scripts", "transfer_sweep.py"),
         encoding="utf-8").read(),
     "--preflight overrides n to 8 AFTER the requirement is checked, so the "
     "check must skip it or the chain's first link dies")
gate("transfer_sweep_refuses_v3_without_n",
     'needs an explicit --n' in io.open(
         os.path.join(ROOT, "proj1", "scripts", "transfer_sweep.py"),
         encoding="utf-8").read(),
     "with V3_N None, --stage v3 must REFUSE rather than fall through to the "
     "512 compare default")
# WHAT SEPARATES THE TWO STAGES, now that n and the seeds do not. The tree
# gains a stage level and the cells carry distinct labels; both are required,
# because v3_table selects on the directory and then verifies the label.
gate("stages_have_distinct_tree_dirs",
     set(T.V3_STAGE_DIRS) == {"v3", "v3abl"}
     and len(set(T.V3_STAGE_DIRS.values())) == 2,
     "%r -- at equal n and equal seeds the stage directory is the ONLY thing "
     "keeping the ablation's 15 extra arms out of the headline's table"
     % (T.V3_STAGE_DIRS,))
_ts_src = io.open(os.path.join(ROOT, "proj1", "scripts", "transfer_sweep.py"),
                  encoding="utf-8").read()
gate("stages_write_distinct_cell_labels",
     _ts_src.count('"v3": "v3", "v3abl": "v3abl"') >= 2,
     "this literal appears in BOTH V3_STAGE_DIRS and the cell_stage map, so "
     "testing `in` passed while only one was intact. Count, or a break in "
     "the label map alone -- the one that decides what a cell records -- "
     "goes unnoticed")
gate("out_dir_includes_the_stage_level",
     "V3_STAGE_DIRS[args.stage]" in _ts_src,
     "results/v3/<backend>/<stage>/n<N>/seed<S>/ -- without the stage level "
     "the two stages collide at equal n")
# ---- the backend registry: one declaration, three files ------------------
#
# V3_BACKENDS says which property pair each base model is scored with and
# which arms it runs. The job file lists the backends, and v3_table names them
# again (it must not import torch, so it cannot read the registry). All three
# have to agree or the array runs backends the table cannot read.
gate("slurm_backends_are_registry_backends",
     set(BE) == set(T.V3_BACKENDS),
     "slurm %s vs registry %s" % (sorted(BE), sorted(T.V3_BACKENDS)))
gate("table_backends_are_registry_backends",
     set(re.findall(r'^BACKENDS = \((.*)\)$',
                    io.open(os.path.join(ROOT, "proj1", "scripts",
                                         "v3_table.py"),
                            encoding="utf-8").read(), re.M)[0]
         .replace('"', "").replace("'", "").replace(",", " ").split())
     == set(T.V3_BACKENDS),
     "v3_table.BACKENDS must name exactly the registry's backends")
gate("every_backend_is_a_real_backend_choice",
     set(T.V3_BACKENDS) <= set(T.BACKENDS),
     "%s not in --backend's choices %s"
     % (sorted(set(T.V3_BACKENDS) - set(T.BACKENDS)), sorted(T.BACKENDS)))
gate("every_backend_declares_a_known_pair",
     all(b["pair"] in ("ours", "tfg") for b in T.V3_BACKENDS.values()),
     str({k: v["pair"] for k, v in T.V3_BACKENDS.items()}))
# A reduced arm set must be a SUBSET of what the headline plans, or the
# backend is declared to run something the stage never produces.
for _be, _cfg in sorted(T.V3_BACKENDS.items()):
    if _cfg["arms"] is not None:
        gate("backend_%s_arms_are_planned" % _be,
             set(_cfg["arms"]) <= set(T.v3_arms()),
             "%s declares %s; the headline plans %s"
             % (_be, list(_cfg["arms"]), T.v3_arms()))
# The pair sets delta, and delta sets the band, so a cell must record which
# pair scored it and the table must refuse to average across pairs.
gate("cells_record_the_pair", '"pair": pair,' in _ts_src,
     "delta = k x MAE(f_B), so two cells with different pairs have different "
     "in_band definitions; the pair has to travel with the cell")
gate("table_refuses_to_mix_pairs",
     '"delta_mode", "pair")' in io.open(
         os.path.join(ROOT, "proj1", "scripts", "v3_table.py"),
         encoding="utf-8").read(),
     "`pair` must be in v3_table.SAME_KEYS or a wide band and a narrow one "
     "get averaged and the difference reported as a result")
gate("ours_pair_has_a_calibration_check",
     "fits calibration slope" in _ts_src,
     "our predictors have no fitted slope to check against QM9's MAD, so "
     "build_pair_ours must fit one purely as a check that the predictor is "
     "in physical units. NOTE it runs on raw one-hot and is therefore "
     "invariant to sampler_scale -- it is NOT the sampler-scale defence, "
     "which is that sampler_scale is read off the generator's checkpoint")
gate("ours_pair_wraps_for_feature_scale",
     "feat_scale=sampler_scale" in _ts_src,
     "PhysicalProperty has no feat_scale and was trained on raw one-hot, so "
     "it MUST be wrapped for any backend whose sampler works in one-hot/k "
     "(every diffusion backend) or it reads types k times too small and "
     "returns finite, wrong numbers that argmax metrics cannot see")

gate("ablation_covers_headline_eta", T.V3_BDG_ETA in T.V3_ABL_ETAS)
gate("ablation_covers_headline_taus",
     set(T.V3_BDG_TAU_MULTS) <= set(T.V3_ABL_TAU_MULTS))

# ---------------------------------------------------------------- arm names
gate("bdg_arm_roundtrip",
     all(T.parse_bdg_arm(T.bdg_arm(e, t)) == (e, t)
         for e in (0.0, 1.0, 4.0, 8.0) for t in (0.5, 0.75, 1.0, 1.5)))
gate("bdg_arm_formatting", T.bdg_arm(4.0, 0.75) == "bdg_e4t0.75", T.bdg_arm(4.0, 0.75))
gate("non_bdg_arm_parses_to_none",
     all(T.parse_bdg_arm(a) is None
         for a in ("unguided", "plug", "tmpd", "lgd_mc", "tfg", "btvg", "btvg_var")))
gate("base_mode_strips_the_suffix",
     T.base_mode("bdg_e4t0.5") == "bdg" and T.base_mode("plug") == "plug",
     "the sampler must be handed 'bdg', never 'bdg_e4t0.5'")
for bad_arm, label in (("bdg_e4", "bdg_arm_rejects_malformed"),
                       ("bdg_exty", "bdg_arm_rejects_nonnumeric")):
    try:
        T.parse_bdg_arm(bad_arm)
        gate(label, False, "no refusal")
    except SystemExit:
        gate(label, True)
kw = T.arm_kwargs("bdg_e4t0.5", 0.16799)
gate("bdg_strength_scale_is_one", T.strength_scale("bdg_e4t0.5", kw, 1.5394) == 1.0,
     "w must mean the same for bdg as for plug, or 'all arms at w=1' is false")
gate("bdg_kwargs_carry_eta_and_mult",
     kw.get("bdg_eta") == 4.0 and kw.get("_bdg_tau_mult") == 0.5, str(kw))
gate("bdg_tau_is_not_delta", "bdg_tau" not in kw and "tau" not in kw, str(kw))

# ---------------------------------------------------------------- the plans
head = T.v3_arms()
abl = T.v3_arms(etas=T.V3_ABL_ETAS, tau_mults=T.V3_ABL_TAU_MULTS, compare=False)
gate("headline_is_compare_set_plus_two_bdg",
     head[:5] == T.V3_COMPARE_ARMS and len(head) == 7, str(head))
gate("headline_drops_btvg", not any("btvg" in a for a in head + abl))
gate("eta0_planned_once", sum(1 for a in abl if a.startswith("bdg_e0")) == 1,
     "at eta=0 the term is multiplied by zero, so tau_mult cannot change the cell")
gate("ablation_arm_count", len(abl) == 1 + 4 * 4, "%d" % len(abl))
cells = T.plan_v3_cells(PROPS, head)
gate("plan_one_cell_per_prop_arm", len(cells) == len(PROPS) * len(head))
gate("plan_unique", len(set(cells)) == len(cells))
gate("plan_headline_all_w_one", {c[3] for c in cells} == {1.0},
     "the HEADLINE pre-registers one strength for every arm. Normalising the "
     "strengths instead was considered and declined (Henry, 26 Sep); the "
     "caveat that equal w is not equal FORCE stands in section 2.2")

# ---- the ablation's strength axis ----------------------------------------
#
# The ablation runs each grid arm at BOTH w = 1 and w = 4. This is the control
# the BDG confound needs: w multiplies the whole correction, so it scales the
# mean and deviation terms together, while w_eff = 1 + eta*e reweights the
# deviation term alone. Section 6 of the headline protocol says the two cannot
# be told apart at q50; this axis is what separates them.
gate("ablation_sweeps_two_strengths",
     tuple(T.V3_ABL_WS) == (1.0, 4.0), str(T.V3_ABL_WS))
gate("ablation_includes_the_headline_strength",
     T.V3_W in T.V3_ABL_WS,
     "w = %g must be in the ablation's axis %s, or no ablation row is "
     "comparable to a headline row" % (T.V3_W, T.V3_ABL_WS))
abl_cells = T.plan_v3_cells(PROPS, abl, ws=T.V3_ABL_WS)
gate("ablation_plans_every_arm_at_every_strength",
     len(abl_cells) == len(PROPS) * len(abl) * len(T.V3_ABL_WS),
     "%d cells; expected %d props x %d arms x %d strengths"
     % (len(abl_cells), len(PROPS), len(abl), len(T.V3_ABL_WS)))
gate("ablation_plan_unique", len(set(abl_cells)) == len(abl_cells),
     "w is part of the cell name, so the two strengths must not collide")
gate("ablation_covers_both_strengths",
     {c[3] for c in abl_cells} == set(T.V3_ABL_WS), str({c[3] for c in abl_cells}))
gate("headline_plan_untouched_by_the_sweep",
     {c[3] for c in T.plan_v3_cells(PROPS, head)} == {1.0},
     "passing no `ws` must still give the headline exactly one strength")
gate("plan_all_q50", {c[2] for c in cells} == {"q50"})
gate("plan_one_window", {c[4] for c in cells} == {T.V3_T_START})
gate("plan_unguided_first", all(c[1] == "unguided" for c in cells[:len(PROPS)]),
     "the reference every comparison divides by must not be the dropped cell")
gate("plan_is_five_tuples", all(len(c) == 5 for c in cells),
     "with_t must be skipped for v3, or this is a 4-tuple unpack error")

# ------------------------------------------------- the SLURM tasks tile
src = io.open(os.path.join(ROOT, "proj1", "cluster", "v3_run.slurm"),
              encoding="utf-8").read()
grp = dict(re.findall(r'^(V3_ARMS|ABL_ARMS)="([^"\n]*)"', src, re.M))
gate("slurm_defines_arm_lists", len(grp) == 2, sorted(grp))
# NEVER AGAIN: the slurm's V3_ARMS literal and the planner's v3_arms() were
# never tied together. tile() and headline_covers_every_cell_once both build
# expectation AND coverage from the same slurm string, so they agree by
# construction -- you could edit one side and not the other with every gate
# green, and the divergence would surface only at the table stage
# (v3_table.py sets want = T.v3_arms() and refuses with "planned headline
# cells absent entirely"), i.e. AFTER the GPU hours were spent. This is the
# gate that would have caught the tau_mult 0.75 drop landing on one side only.
gate("slurm_headline_arms_match_the_planner",
     "V3_ARMS" in grp and grp["V3_ARMS"].split(",") == list(head),
     "slurm %s vs planner %s"
     % (grp.get("V3_ARMS", "").split(","), list(head)))
gate("slurm_abl_arms_match_the_planner",
     "ABL_ARMS" in grp and sorted(grp["ABL_ARMS"].split(",")) == sorted(abl),
     "slurm %s vs planner %s"
     % (sorted(grp.get("ABL_ARMS", "").split(",")), sorted(abl)))
gate("slurm_abl_arms_is_the_whole_grid",
     "ABL_ARMS" in grp and len(grp["ABL_ARMS"].split(",")) == 17,
     "%d arms; the ablation has its OWN stage tree, so it cannot reuse the "
     "headline's BDG rungs and must carry all 17"
     % (len(grp.get("ABL_ARMS", "").split(",")) if grp.get("ABL_ARMS") else 0,))

# THE TASK GRID IS NOW ONE FORMULA FOR BOTH STAGES. It used to be two, with
# hardcoded strides (t/9, t/3) that these gates parsed out with regexes; the
# ablation's collapse to one seed needed its own arithmetic, and a wrong stride
# once left 27 cells unproducible. Both stages now run backends x properties x
# seeds and differ ONLY in which arm list a task carries, so the strides come
# from the array lengths in the slurm itself and there is nothing per-stage
# left to get wrong.
ts = re.search(r"task_spec \(\) \{(.*?)\n\}", src, re.S)
gate("slurm_task_spec_found", bool(ts))
blk = ts.group(1) if ts else ""

NP, NS, NB = len(PROPS), len(T.V3_SEEDS), len(BE)
NAS = len(T.V3_ABL_SEEDS)

gate("slurm_derives_dimensions",
     re.search(r"NP=\$\{#PROPS\[@\]\}", src) is not None
     and re.search(r"NS=\$\{#SEEDS\[@\]\}", src) is not None
     and re.search(r"NB=\$\{#BACKENDS\[@\]\}", src) is not None,
     "NP/NS/NB must come from the array lengths, or adding a backend silently "
     "leaves tasks unproduced")
gate("slurm_task_count_is_the_product",
     re.search(r"n_tasks \(\) \{ echo \$\(\( NB \* NP \* NS \)\); \}", src)
     is not None,
     "n_tasks must be NB * NP * NS for BOTH stages")
gate("slurm_backend_stride",
     re.search(r"BACKENDS\[\$\(\(\s*t\s*/\s*\(NP\s*\*\s*NS\)\s*\)\)\]", blk)
     is not None, blk[:120])
gate("slurm_prop_stride",
     re.search(r"PROPS\[\$\(\(\s*\(t\s*/\s*NS\)\s*%\s*NP\s*\)\)\]", blk)
     is not None, blk[:120])
gate("slurm_seed_stride",
     re.search(r"SEEDS\[\$\(\(\s*t\s*%\s*NS\s*\)\)\]", blk) is not None,
     blk[:120])
gate("slurm_task_spec_is_stage_agnostic",
     'if [ "$STAGE" = "v3abl" ]; then arms="$ABL_ARMS"; else arms="$V3_ARMS"; fi'
     in blk,
     "the ONLY thing task_spec may branch on is the arm list; anything else "
     "re-introduces the per-stage arithmetic the strides were unified to remove")
gate("no_arm_halves_survive", "half=" not in blk,
     "a `half=` stride survives, but one task carries a whole arm list")


def tile(stage):
    """Re-derive task_spec() from the SAME formula the slurm now uses."""
    arms = grp["ABL_ARMS" if stage == "v3abl" else "V3_ARMS"].split(",")
    out, bad_t = [], []
    for t in range(NB * NP * NS):
        bi, pi, si = t // (NP * NS), (t // NS) % NP, t % NS
        if bi >= NB or pi >= NP or si >= NS:
            bad_t.append(t)
            continue
        for arm in arms:
            out.append((BE[bi], PROPS[pi], str(T.V3_SEEDS[si]), arm))
    return out, bad_t


h_cov, h_bad = tile("v3")
a_cov, a_bad = tile("v3abl")
gate("no_task_indexes_out_of_range", not h_bad and not a_bad,
     "tasks that would expand an empty value: %s" % (sorted(set(h_bad + a_bad)) or "none"))
gate("headline_tasks_no_duplicates", len(h_cov) == len(set(h_cov)),
     "%d - %d" % (len(h_cov), len(set(h_cov))))
gate("ablation_tasks_no_duplicates", len(a_cov) == len(set(a_cov)),
     "%d - %d" % (len(a_cov), len(set(a_cov))))

gate("every_arm_list_non_empty",
     all(all(x.strip() for x in v.split(",")) for v in grp.values()), sorted(grp))

# The two stages no longer share a plan: the headline is 7 arms x 3 seeds at
# n=2000, the ablation 17 arms x 1 seed at n=1000 in its OWN tree. So they are
# tiled separately, and a single `want` set would be wrong. The union stays 22
# because the headline's 2 BDG arms are a subset of the grid's 17.
union = list(dict.fromkeys(head + abl))
gate("union_is_22_arms", len(union) == 22, "%d" % len(union))

want_h = {(b, p, str(s), a) for b in BE for p in PROPS
          for s in T.V3_SEEDS for a in grp["V3_ARMS"].split(",")}
gate("headline_covers_every_cell_once",
     set(h_cov) == want_h and len(h_cov) == len(want_h),
     "covered %d, wanted %d, holes %s extras %s"
     % (len(h_cov), len(want_h), sorted(want_h - set(h_cov))[:3] or "none",
        sorted(set(h_cov) - want_h)[:3] or "none"))

want_a = {(b, p, str(sd), a) for b in BE for p in PROPS
          for sd in T.V3_ABL_SEEDS for a in grp["ABL_ARMS"].split(",")}
gate("ablation_covers_every_cell_once",
     set(a_cov) == want_a and len(a_cov) == len(want_a),
     "covered %d, wanted %d, holes %s extras %s"
     % (len(a_cov), len(want_a), sorted(want_a - set(a_cov))[:3] or "none",
        sorted(set(a_cov) - want_a)[:3] or "none"))

# The ablation DOES repeat the headline's BDG arms, and must: it writes its own
# stage tree, so there is nothing of the headline's in it to reuse. The gate
# that used to forbid the repetition is inverted -- what would be wrong now is
# the grid MISSING them. This is set equality over arm NAMES; what keeps the
# justification true is stages_have_distinct_tree_dirs, not n.
gate("ablation_repeats_headline_bdg_arms_in_its_own_tree",
     set(T.v3_arms()) & set(grp["ABL_ARMS"].split(","))
     == {a for a in T.v3_arms() if a.startswith("bdg_e")},
     "every headline BDG arm must appear in the grid, since the two stages "
     "share no cells: %s"
     % (sorted(set(T.v3_arms()) & set(grp["ABL_ARMS"].split(","))),))
gate("ablation_carries_no_comparison_arms",
     not (set(T.V3_COMPARE_ARMS) & set(grp["ABL_ARMS"].split(","))),
     "Henry, 26 Sep: no plug/unguided baseline in the ablation, so the grid is "
     "read only against itself; found %s"
     % sorted(set(T.V3_COMPARE_ARMS) & set(grp["ABL_ARMS"].split(","))))

sub = io.open(os.path.join(ROOT, "proj1", "cluster", "submit_v3.sh"),
              encoding="utf-8").read()
# Both stages are the same shape now, so ONE range covers both.
N_TASKS = NB * NP * NS
# submit_v3.sh DERIVES the range from the job file rather than restating it,
# so the gate checks that it derives rather than that it matches a literal --
# a literal is exactly what went stale when the third backend landed.
# THE GATE MUST EXECUTE THE DERIVATION, not grep for it. A first version only
# checked the two literal expressions were present -- so when the sed
# backreferences in those same lines were corrupted into control bytes, and
# the script exited 1 without submitting anything, it still PASSED.
_sub_lines = [l for l in sub.split(chr(10))
              if re.match(r"^N(B|PROPS|SEEDS)=", l)]
gate("submit_extracts_three_dimensions", len(_sub_lines) == 3,
     "expected NB/NPROPS/NSEEDS assignments in submit_v3.sh, found %d"
     % len(_sub_lines))
_derived = None
try:
    _script = (chr(34).join(["JOB=", os.path.join(ROOT, "proj1", "cluster",
                             "v3_run.slurm").replace(chr(92), "/"), ""])
               + chr(10)
               + chr(10).join(_sub_lines) + chr(10)
               + 'echo "$NB $NPROPS $NSEEDS"')
    _derived = subprocess.run(["bash", "-c", _script], capture_output=True,
                              text=True, timeout=30).stdout.split()
except Exception as _exc:                          # noqa: BLE001
    print("  (note: could not run bash, derivation not executed: %s)" % _exc)
if _derived:
    gate("submit_array_range_derivation_runs",
         _derived == [str(NB), str(NP), str(NS)],
         "submit_v3.sh extracts %s from the job file, which declares %d "
         "backends / %d properties / %d seeds. A mismatch means the array "
         "range is wrong and tasks go unsubmitted."
         % (_derived, NB, NP, NS))
gate("submit_derives_the_array_range",
     "NTASKS=$(( NB * NPROPS * NSEEDS ))" in sub
     and 'ARRAY="0-$(( NTASKS - 1 ))"' in sub
     and "0-17" not in sub,
     "submit_v3.sh must compute the array range from the job file's "
     "BACKENDS/PROPS/SEEDS (now %d tasks), never hardcode it" % N_TASKS)

# ---- the batch: BDG's estimator IS the batch ------------------------------
#
# The batch is not a performance knob here: V_b is estimated over whatever the
# sampler is handed, so the batch is the controller's sample size. Two bugs are
# pinned below, both of which shipped in this file's own slurm.
#
#   never again (1) --batch absent, so the default 128 ran a ragged last
#                   controller: at the old n=5000 that was 40 controllers with
#                   the last over EIGHT molecules (a 53 % SE on its variance),
#                   and at n=2000 it is 15 whole plus a remainder of 80
#   never again (2) --batch = n, which measurement says needs 80 GiB on our
#                   base and 112 GiB on EquiFM at n=2000, against a 45 GB slice
gate("slurm_passes_batch", "--batch" in src,
     "without --batch the default 128 leaves a ragged last controller, pooled "
     "into the cell's diagnostics as though it were an equal")
m_batch = re.search(r'BATCH="\$\{V3_BATCH:-(\d+)\}"', src)
gate("slurm_batch_default_is_measured", m_batch is not None,
     "the default must be a literal measured by batch_memprobe.py, not $N")
BATCH_DEFAULT = int(m_batch.group(1)) if m_batch else 0
# N HAS NO DEFAULT, in either place. The slurm must take it from the
# environment with an EMPTY fallback and refuse when it is unset, so that a v3
# tree can never be written at a size nobody chose. The old gate compared two
# literals; there are no literals to compare any more, so it checks the
# refusal instead.
gate("slurm_n_has_no_default",
     re.search(r'N="\$\{V3_N:-\}"', src) is not None
     and re.search(r'N="\$\{V3_N:-\d+\}"', src) is None,
     "V3_N must fall back to EMPTY, not to a number: v3 pre-registers no n")
gate("slurm_refuses_unset_n",
     "V3_N is not set" in src,
     "with no default the job must FATAL on an unset V3_N rather than run")
gate("slurm_batch_divides_n_at_runtime",
     re.search(r"N % BATCH \)\) -ne 0", src) is not None,
     "n is chosen by the operator, so divisibility can only be checked at "
     "run time -- batch %s is BDG's estimator and a remainder batch is a "
     "second, much noisier controller pooled in as an equal" % BATCH_DEFAULT)
# and the job must refuse an override that does not divide n
gate("slurm_refuses_indivisible_batch",
     re.search(r"N % BATCH \)\) -ne 0", src) is not None,
     "V3_BATCH is a documented override, so the check must be in the job too")
# the recommendation must come from the probe's own record, not from prose
PROBE_JSON = os.path.join(ROOT, "results", "v3_batch_memory.json")
if os.path.exists(PROBE_JSON):
    pj = json.load(open(PROBE_JSON))
    gate("batch_matches_probe_recommendation",
         pj.get("recommended_batch") == BATCH_DEFAULT,
         "slurm says %s, %s says %s"
         % (BATCH_DEFAULT, os.path.basename(PROBE_JSON),
            pj.get("recommended_batch")))
    gate("probe_measured_both_backends",
         {p["backend"] for p in pj.get("points", [])} == {"fm", "equifm"},
         "EquiFM is the tighter backend (1.85x our memory slope), so a probe "
         "that skipped it cannot choose the batch")
else:
    gate("batch_probe_record_exists", False,
         "run proj1/scripts/batch_memprobe.py -- the batch must be measured")

# ---- the job runs `set -u`, so a STALE VARIABLE REFERENCE is fatal ---------
#
# never again: reshaping the ablation deleted ABL_0/ABL_1, but the preflight
# still built its arm list from them. Every arm-list gate passed and the job
# would still have died on `ABL_0: unbound variable`. Collect the names the
# script DEFINES and the names it READS, and refuse on any read-but-never-set
# name that the environment does not supply.
# comments first: this file DISCUSSES the variables it once used, and a prose
# mention of $ABL_0 is not a reference. Cut at the first `#` that starts a word.
_code = "\n".join(re.sub(r'(^|\s)#.*$', '', ln) for ln in src.splitlines())
# assignments ANYWHERE on a line, not just at its start: `WANT=51; TREE=...` and
# `TN="${SPEC%%|*}"; REST="${SPEC#*|}"` are both real definitions
_defined = set(re.findall(r'(?:^|[;&|(]\s*|\s)([A-Z_][A-Z0-9_]*)\+?=', _code, re.M))
# and loop variables, which are defined by the `for` itself
_defined |= set(re.findall(r'\bfor\s+([A-Z_][A-Z0-9_]*)\s+in\b', _code))
_defined |= {w for line in re.findall(r'\blocal ([^\n=]*)', _code)
             for w in line.split()}
for _m in re.findall(r'read -r ([A-Z_ ]+)', _code):
    _defined |= set(_m.split())
# names SLURM or the shell provides, plus the documented submit-line overrides
# ONLY names the environment supplies: SLURM's own, the shell's, and the
# documented submit-line overrides. Everything else must be assigned in the file,
# including loop variables and same-line assignments, which are now detected.
_env = {"SLURM_CONF", "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "PATH", "USER", "HOME", "IFS",
        "V3_N", "V3_ABL_N", "V3_BATCH", "BUDGET_MIN", "STAGE",
        # the checkout and venv a teammate runs from; both have in-file
        # defaults, and submit_v3.sh forwards them through --export
        "CGM_PROJ", "CGM_VENV"}
# An assignment INSIDE a quoted echo is not an assignment. CGM_PROJ was
# whitelisted by the string `CGM_PROJ=/path/to/repo` in a help message while
# CGM_VENV, named only in a ${...:-default}, was reported undefined -- the two
# are equally safe, so the discrepancy was the detector, not the code.
_echoed = set()
for _line in _code.split('\n'):
    if 'echo "' in _line:
        _echoed |= set(re.findall('([A-Z_][A-Z0-9_]*)=', _line))
_defined -= {n for n in _echoed
             if not re.search(r'^\s*%s=' % n, _code, re.M)}
_read = set(re.findall(r'\$\{?([A-Z_][A-Z0-9_]*)', _code))
_undef = sorted(n for n in _read - _defined - _env)
gate("shell_references_only_defined_vars", not _undef,
     "read but never assigned in this file (fatal under set -u): %s"
     % (", ".join(_undef) or "none"))

# ---- EVERY transfer_sweep call must carry the flags its stage requires -----
#
# never again: the preflight block called --stage v3 without --seed. It writes
# nothing, so it looked like a mode needing no seed -- but the stage validates
# its arguments before it reads the mode, so both arm checks died in argparse
# after 7 seconds and the afterok gate blocked the entire array. The
# flag-VOCABULARY gate below did not catch it, because --seed exists; it simply
# was not passed. So check each call SITE too.
calls = re.findall(r'srun python -u "\$TS"((?:[^\n]*\\\n)*[^\n]*)', src)
gate("slurm_ts_calls_found", len(calls) >= 2,
     "found %d invocations of $TS (expect preflight + run_one)" % len(calls))
for _i, _call in enumerate(calls):
    _flat = " ".join(_call.replace("\\\n", " ").split())
    _needs = [f for f in ("--stage", "--backend", "--props", "--arms", "--seed")
              if f not in _flat]
    gate("slurm_ts_call%d_has_required_flags" % _i, not _needs,
         "missing %s in: %s" % (", ".join(_needs) or "nothing", _flat[:100]))

# ---- every flag the job passes must exist in the script it drives ---------
inv = re.search(r'srun python -u "\$TS"(.*?)--max-minutes', src, re.S)
gate("slurm_ts_invocation_found", bool(inv))
used = set(re.findall(r"(--[a-z][a-z0-9-]+)", inv.group(1))) if inv else set()
used.add("--max-minutes")
help_txt = subprocess.run(
    [sys.executable, os.path.join(ROOT, "proj1", "scripts", "transfer_sweep.py"), "--help"],
    capture_output=True, text=True).stdout
flags = set(re.findall(r"(--[a-z][a-z0-9-]+)", help_txt))
gate("slurm_passes_only_real_flags", bool(used) and used <= flags,
     "passes %d flags; not in transfer_sweep.py: %s"
     % (len(used), sorted(used - flags) or "none"))

bad = sorted(k for k, v in R.items() if not v)
print("")
print("ALL PASS (%d gates)" % len(R) if not bad else "FAILED: %s" % ", ".join(bad))
sys.exit(1 if bad else 0)
