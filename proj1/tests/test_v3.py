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
BE = ["fm", "equifm"]
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
# The ablation is DELIBERATELY smaller than the headline (Henry, 26 Sep, second
# decision of the day -- the first had them equal and this gate asserted that).
# It is cheap because it is read against ITSELF, not against headline rows.
#
# STRICTLY smaller is now also LOAD-BEARING, not just a budget choice. The two
# stages write one stage label into results/v3/<be>/n<N>/, so n is the ONLY
# thing separating their trees. At equal n, v3_table.py globs the ablation's 15
# single-seed arms into the headline's table and refuses, because it requires
# every (prop, arm) at all three seeds. When the headline dropped 5000 -> 2000
# the ablation had to move 2000 -> 1000 for exactly this reason.
gate("ablation_is_smaller_than_headline", T.V3_ABL_N < T.V3_N,
     "ablation n %d vs headline %d -- at equal n the two stages share a tree "
     "and the table stage refuses" % (T.V3_ABL_N, T.V3_N))
gate("ablation_uses_one_seed", len(T.V3_ABL_SEEDS) == 1,
     "%s -- 1 seed, so ablation rows carry no across-seed spread"
     % (T.V3_ABL_SEEDS,))
gate("ablation_seed_is_a_headline_seed", set(T.V3_ABL_SEEDS) <= set(T.V3_SEEDS),
     "%s not drawn from %s" % (T.V3_ABL_SEEDS, T.V3_SEEDS))
# the batch is BDG's estimator, so it must still divide the SMALLER n or the
# ablation runs a ragged last controller -- the bug the headline already fixed
gate("ablation_n_divisible_by_batch", T.V3_ABL_N % 500 == 0,
     "n %d %% batch 500 = %d; %d whole controllers of 500 is the intent"
     % (T.V3_ABL_N, T.V3_ABL_N % 500, T.V3_ABL_N // 500))
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
gate("plan_all_w_one", {c[3] for c in cells} == {1.0})
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
     "%d arms; the ablation has its OWN tree (n=%d, not the headline's %d), so "
     "it cannot reuse the headline's BDG rungs and must carry all 17"
     % (len(grp.get("ABL_ARMS", "").split(",")) if grp.get("ABL_ARMS") else 0,
        T.V3_ABL_N, T.V3_N))

ts = re.search(r"task_spec \(\) \{(.*?)\n\}", src, re.S)
gate("slurm_task_spec_found", bool(ts))
blk = ts.group(1) if ts else ""
abl_blk = blk[blk.index('if [ "$STAGE" = "v3abl" ]'):blk.index("  else")] if ts else ""
head_blk = blk[blk.index("  else"):] if ts else ""

BE_PAT = r"BACKENDS\[\$\(\(\s*t\s*/\s*(\d+)"
PROP_PAT = r"PROPS\[\$\(\(\s*\(t\s*/\s*(\d+)\)"
HALF_PAT = r"half=\$\(\(\s*\(t\s*/\s*(\d+)\)"


def one(pat, blob):
    m = re.search(pat, blob)
    return int(m.group(1)) if m else None


ABL_PROP_PAT = r"PROPS\[\$\(\(\s*t\s*%\s*(\d+)"   # ablation indexes prop directly
A = {"be": one(BE_PAT, abl_blk), "prop": one(ABL_PROP_PAT, abl_blk)}
H = {"be": one(BE_PAT, head_blk), "prop": one(PROP_PAT, head_blk)}
NP, NS, NB = len(PROPS), len(T.V3_SEEDS), len(BE)
NAS = len(T.V3_ABL_SEEDS)
gate("headline_strides_parsed", None not in H.values(), str(H))
gate("ablation_strides_parsed", None not in A.values(), str(A))
gate("headline_strides_correct", H["be"] == NP * NS and H["prop"] == NS,
     "be %s (want %d), prop %s (want %d)" % (H["be"], NP * NS, H["prop"], NS))
# the ablation collapsed to ONE seed, so there is no seed stride and no arm-half:
# 6 tasks = 2 backends x 3 properties, each carrying all 17 arms
gate("ablation_strides_correct", A["be"] == NP and A["prop"] == NP,
     "be %s (want %d), prop-modulus %s (want %d)" % (A["be"], NP, A["prop"], NP))
gate("ablation_has_no_arm_halves", one(HALF_PAT, abl_blk) is None,
     "a `half=` stride survives, but one task now carries all 17 arms")

nt = re.search(r'v3abl" \]; then echo (\d+); else echo (\d+)', src)
gate("slurm_declares_task_counts", bool(nt), nt.groups() if nt else "")
N_ABL = int(nt.group(1)) if nt else -1
N_HEAD = int(nt.group(2)) if nt else -1
gate("task_counts_match_strides",
     N_HEAD == NB * NP * NS and N_ABL == NB * NP * NAS,
     "headline %d (want %d), ablation %d (want %d)"
     % (N_HEAD, NB * NP * NS, N_ABL, NB * NP * NAS))


def tile(stage):
    """Re-derive task_spec() from the strides PARSED above."""
    out, bad_t = [], []
    if stage == "v3abl":
        for t in range(N_ABL):
            bi, pi = t // A["be"], t % A["prop"]
            if bi >= NB or pi >= NP:
                bad_t.append(t)
                continue
            for arm in grp["ABL_ARMS"].split(","):
                out.append((BE[bi], PROPS[pi], str(T.V3_ABL_SEEDS[0]), arm))
    else:
        for t in range(N_HEAD):
            bi, pi, si = t // H["be"], (t // H["prop"]) % NP, t % NS
            if bi >= NB or pi >= NP or si >= NS:
                bad_t.append(t)
                continue
            for arm in grp["V3_ARMS"].split(","):
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

want_a = {(b, p, str(T.V3_ABL_SEEDS[0]), a) for b in BE for p in PROPS
          for a in grp["ABL_ARMS"].split(",")}
gate("ablation_covers_every_cell_once",
     set(a_cov) == want_a and len(a_cov) == len(want_a),
     "covered %d, wanted %d, holes %s extras %s"
     % (len(a_cov), len(want_a), sorted(want_a - set(a_cov))[:3] or "none",
        sorted(set(a_cov) - want_a)[:3] or "none"))

# The ablation now DOES repeat the headline's BDG arms, and must: it runs at a
# DIFFERENT n (V3_ABL_N vs V3_N), so its tree holds nothing to reuse. The gate
# that used to forbid this is inverted -- what would be wrong now is the grid
# MISSING them. NOTE this gate is set equality over arm NAMES and never reads n;
# ablation_is_smaller_than_headline is what keeps the justification true.
gate("ablation_repeats_headline_bdg_arms_in_its_own_tree",
     set(T.v3_arms()) & set(grp["ABL_ARMS"].split(","))
     == {a for a in T.v3_arms() if a.startswith("bdg_e")},
     "every headline BDG arm must be present at n=%d, since n=%d cells cannot "
     "be reused: %s" % (T.V3_ABL_N, T.V3_N,
                        sorted(set(T.v3_arms()) & set(grp["ABL_ARMS"].split(",")))))
gate("ablation_carries_no_comparison_arms",
     not (set(T.V3_COMPARE_ARMS) & set(grp["ABL_ARMS"].split(","))),
     "Henry, 26 Sep: no plug/unguided baseline in the ablation, so the grid is "
     "read only against itself; found %s"
     % sorted(set(T.V3_COMPARE_ARMS) & set(grp["ABL_ARMS"].split(","))))

sub = io.open(os.path.join(ROOT, "proj1", "cluster", "submit_v3.sh"),
              encoding="utf-8").read()
gate("submit_uses_both_array_ranges",
     ("0-%d" % (N_HEAD - 1)) in sub and ("0-%d" % (N_ABL - 1)) in sub,
     "headline must submit as 0-%d and the ablation as 0-%d" % (N_HEAD - 1, N_ABL - 1))

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
m_n = re.search(r'N="\$\{V3_N:-(\d+)\}"', src)
gate("slurm_n_matches_planner",
     m_n is not None and int(m_n.group(1)) == T.V3_N,
     "slurm n %s vs transfer_sweep V3_N %s"
     % (m_n.group(1) if m_n else None, T.V3_N))
N_CELL = int(m_n.group(1)) if m_n else T.V3_N
gate("slurm_batch_divides_n", BATCH_DEFAULT > 0 and N_CELL % BATCH_DEFAULT == 0,
     "batch %s must divide n %s exactly: a remainder batch is a second, much "
     "noisier controller pooled in as an equal" % (BATCH_DEFAULT, N_CELL))
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
        "V3_N", "V3_ABL_N", "V3_BATCH", "BUDGET_MIN", "STAGE"}
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
