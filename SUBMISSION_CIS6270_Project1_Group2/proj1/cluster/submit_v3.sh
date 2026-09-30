#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"  # default added when this submission folder was built
# Protocol v3 + its BDG ablation, on BOTH base models, in one paste.
#
#   bash proj1/cluster/submit_v3.sh
#
# Prints the job ids and exits. Nothing needs resubmitting by hand unless a
# stage refuses, and the only stage that can refuse is the table.
#
# THE CHAIN
#   A  v3 headline   the comparison set + BDG eta=4, on every base model,
#                    3 properties x 3 seeds. Tree results/v3/*/v3/n<N>
#   B  insurance after A          finishes whatever A's 4-hour window cut off
#   C  v3 ablation   the 17-arm (eta x tau_mult) grid, the SAME n and the
#                    SAME three seeds, its own stage tree
#                    (results/v3/*/v3abl/n<N>). The QM9 diffusion base runs
#                    unguided+plug only, so its ablation tasks exit 0 at once
#
# Both arrays are 0-$((NB*NPROPS*NSEEDS - 1)), derived below from the job file.
#   D  insurance after C
#   E  table                      one page per base model, plus both side by side
#
# The ablation is chained AFTER the headline, but the two no longer share a
# tree or a cell: results/v3/*/v3/ and results/v3/*/v3abl/ are separate, so there is
# nothing to collide over and nothing to reuse. The ordering now buys only one
# thing -- the headline's [timing] lines land before the ablation commits.
#
# `afterany`, not `afterok`: a task that runs out of its 4-hour window exits
# non-zero with its finished cells on disk, and the next link must still run.
# The table is the one link that refuses rather than warns, so if the run is
# incomplete it exits non-zero and says which seed or cell is missing.
set -uo pipefail
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}"

# THE CHECKOUT TO SUBMIT FROM. CGM_PROJ overrides it, and it must -- this
# line was a bare literal while the JOB file already honoured CGM_PROJ, so a
# teammate who followed the handoff's "override it, no editing" got a `cd`
# into a path that does not exist on their cluster.
PROJ="${CGM_PROJ:-$PROJECT_ROOT}"
JOB=proj1/cluster/v3_run.slurm
# NO DEFAULT. v3 pre-registers no cell size: the operator picks it (Henry,
# 26 Sep). Choose it against the power table in section 6.1 of
# docs/protocol/FULL_RUN_V3_PROTOCOL.md, not against the budget alone.
N="${V3_N:-}"
if [ -z "$N" ]; then
  echo "STOP: set V3_N. The v3 protocol does not fix the cell size -- it is" >&2
  echo "yours to choose, and it decides the run's power, not just its cost." >&2
  echo "  docs/protocol/FULL_RUN_V3_PROTOCOL.md section 6.1 has the table." >&2
  echo "Then:  V3_N=<n> bash proj1/cluster/submit_v3.sh [headline|ablation|all]" >&2
  exit 1
fi
# V3_BATCH must travel too. It is BDG's estimator size, not a speed knob: the
# job defaults it to 500 (measured -- one batch of n needs 280 GiB on EquiFM
# against a 45 GB slice) and refuses a value that does not divide N. Leaving it
# out of --export meant an override set on this line silently did nothing.
BATCH="${V3_BATCH:-500}"
# THE ABLATION RUNS AT THE SAME n AND THE SAME THREE SEEDS as the headline
# (Henry, 26 Sep, final). It used to run smaller on both, purely so that n
# could tell the two trees apart; transfer_sweep now writes
# results/v3/<be>/<stage>/n<N>/seed<S>/, so the STAGE directory does that job
# and the two stages are free to match. ABL_N is kept as one name for the
# divisibility loop below.
ABL_N="$N"
# THE ARRAY RANGE IS DERIVED, not typed. It is backends x properties x seeds,
# and it changed the moment a third base model was added; a hardcoded range
# would have silently left the last 9 tasks unsubmitted. The job file is the
# single source of the backend list, so read it from there.
# No sed backreferences here, on purpose: an earlier version used a \1
# and the tool that wrote this file ate the backslash, leaving a literal
# control byte. `bash -n` passed, the guard below fired, and the script
# exited 1 without submitting anything. Strip the prefix and the bracket
# instead -- there is nothing to get wrong.
NB=$(sed -n 's/^BACKENDS=(//p' "$JOB" | tr -d ')' | wc -w)
NPROPS=$(sed -n 's/^PROPS=(//p' "$JOB" | tr -d ')' | wc -w)
NSEEDS=$(sed -n 's/^SEEDS=(//p' "$JOB" | tr -d ')' | wc -w)
if [ "$NB" -lt 1 ] || [ "$NPROPS" -lt 1 ] || [ "$NSEEDS" -lt 1 ]; then
  echo "STOP: could not read BACKENDS/PROPS/SEEDS out of $JOB." >&2
  exit 1
fi
NTASKS=$(( NB * NPROPS * NSEEDS ))
ARRAY="0-$(( NTASKS - 1 ))"
for _n in "$N" "$ABL_N"; do
    if [ $(( _n % BATCH )) -ne 0 ]; then
        echo "STOP: V3_BATCH=$BATCH does not divide n=$_n." >&2
        echo "A remainder batch is a second, far noisier BDG controller: V_b is" >&2
        echo "estimated over the batch, so a short one is a worse estimator" >&2
        echo "pooled in as an equal." >&2
        exit 1
    fi
done
# CGM_PROJ/CGM_VENV travel too, so a teammate running from their own checkout
# does not have to edit the job file. Unset here means the job keeps its
# default (Henry's Betty tree), which is what the existing chain expects.
EXPORTS="ALL,V3_N=$N,V3_ABL_N=$ABL_N,V3_BATCH=$BATCH"
[ -n "${CGM_PROJ:-}" ] && EXPORTS="$EXPORTS,CGM_PROJ=$CGM_PROJ"
[ -n "${CGM_VENV:-}" ] && EXPORTS="$EXPORTS,CGM_VENV=$CGM_VENV"

cd "$PROJ" || {
  echo "STOP: cannot cd to $PROJ." >&2
  echo "Set CGM_PROJ to your checkout:  CGM_PROJ=/path/to/repo V3_N=<n> bash $0" >&2
  exit 1; }
[ -f "$JOB" ] || { echo "STOP: $JOB not found -- ship the code tarball first" >&2; exit 1; }

# LOG DIRECTORY. sbatch parses #SBATCH --output BEFORE the job body runs, so
# the job file cannot place its own logs relative to CGM_PROJ -- the header's
# absolute path would send a teammate's logs to an allocation they cannot
# write, and the failure is invisible because the log is what broke. Pass it
# on the submit line instead, which overrides the header.
LOGDIR="${CGM_LOGDIR:-$PROJ/logs}"
mkdir -p "$LOGDIR" || { echo "STOP: cannot create $LOGDIR" >&2; exit 1; }

sub () {                # sub NAME ARRAY DEPENDENCY STAGE -> echoes job id
    local name="$1" arr="$2" dep="$3" stage="$4"
    local cmd=(sbatch --parsable --job-name="cgm-v3-$name")
    # override the job file's absolute --output/--error (see LOGDIR above)
    cmd+=(--output="$LOGDIR/v3-%j.out" --error="$LOGDIR/v3-%j.err")
    [ -n "${CGM_QOS:-}" ] && cmd+=(--qos="$CGM_QOS")
    [ -n "${CGM_PARTITION:-}" ] && cmd+=(--partition="$CGM_PARTITION")
    [ -n "$arr" ] && cmd+=(--array="$arr")
    [ -n "$dep" ] && cmd+=(--dependency="$dep")
    cmd+=(--export="$EXPORTS,STAGE=$stage" "$JOB")
    local id
    id=$("${cmd[@]}") || { echo "STOP: sbatch failed for $name" >&2; exit 1; }
    echo "$id"
}

WHAT="${1:-headline}"
case "$WHAT" in all|headline|ablation|table) ;; *)
    echo "usage: bash proj1/cluster/submit_v3.sh [headline|ablation|all|table]" >&2
    echo "  headline  the comparison set + BDG eta=4   (the default)" >&2
    echo "  ablation  the eta x tau_mult grid, 17 arms -- the LARGER stage" >&2
    echo "  all       both, chained" >&2
    echo "  cost depends on V3_N: python proj1/scripts/v3_power.py" >&2
    echo "  table     re-read whatever is on disk" >&2
    exit 1 ;;
esac

# WHY THE DEFAULT IS STILL `headline` AND NOT `all`.
# The headline's [timing] lines settle the real per-pass cost of the non-fm
# backends -- the one assumed factor in the whole budget, and the one that
# decides whether the 1.83x or the 2.5x column is true -- before the ABLATION,
# which is the LARGER stage (17 arms against 7), commits. If you already trust
# the ratio, `all` is fine.
#
# Cost depends on V3_N and is not quoted here; hardcoded GPU-hour figures in
# this file went stale twice. Run:  python proj1/scripts/v3_power.py
#
# The two stages do NOT share a tree (v3/ vs v3abl/), so the ablation
# recomputes its own eta = 4 rungs rather than skipping the headline's.
#
# Check your own limit before choosing:
#   sacctmgr show assoc user=$USER format=User,QOS,MaxJobs,GrpTRES%30
#   sinfo -p b200-mig45 -o "%.20P %.6D %.6t %N"

LAST=""
P=""; A=""; B=""; C=""; D=""
dep () {
    # afterok ONLY on the preflight link: a broken arm or a batch that does not
    # fit must stop everything. Every later link is afterany, because a task that
    # runs out of its 4-hour window exits nonzero with good cells on disk and the
    # next link must still run.
    if [ -n "$LAST" ] && [ "$LAST" = "$P" ]; then printf 'afterok:%s' "$LAST"
    elif [ -n "$LAST" ]; then printf 'afterany:%s' "$LAST"
    fi
}

# The preflight link, and the ONE place afterok is used instead of afterany: if a
# single arm cannot run on a backend, or the batch does not fit on this card,
# nothing else should start. Everything after it is afterany, because a task that
# runs out of its 4-hour window exits nonzero with good cells on disk.
if [ "$WHAT" != table ]; then
    P=$(sub pre "" "" preflight); LAST=$P
fi

if [ "$WHAT" = all ] || [ "$WHAT" = headline ]; then
    A=$(sub head   "$ARRAY" "$(dep)" v3);    LAST=$A
    B=$(sub head2  ""   "$(dep)" v3);    LAST=$B
fi
if [ "$WHAT" = all ] || [ "$WHAT" = ablation ]; then
    C=$(sub abl    "$ARRAY" "$(dep)" v3abl); LAST=$C
    D=$(sub abl2   ""   "$(dep)" v3abl); LAST=$D
fi
E=$(sub table  ""   "$(dep)" table)

cat <<EOF

protocol v3 "$WHAT" submitted, batch = $BATCH
  n = $N, 3 seeds, batch $BATCH -> $(( N / BATCH )) BDG controllers per cell
  both stages run the same n and the same seeds; the STAGE directory is what
  separates results/v3/<be>/v3/ from results/v3/<be>/v3abl/

${P:+  $P  preflight               every arm on both bases, and the batch
}${A:+  $A  headline array ($ARRAY)  comparison set + BDG eta=4, every base
}${B:+  $B  headline insurance
}${C:+  $C  ablation array ($ARRAY)  17-arm eta x tau_mult grid, same n, 3 seeds
}${D:+  $D  ablation insurance
}  $E  table                   refuses if anything is missing
$( [ "$WHAT" = headline ] && echo "
the ablation is NOT queued. Read the [timing] lines first, then:
  bash proj1/cluster/submit_v3.sh ablation" )

watch it:
  squeue -u \$USER -o "%.18i %.14j %.9T %.10M %.28E"
  ls results/v3/fm/v3/n$N/seed20261001/ | wc -l       # headline: up to 21 (7 arms x 3 props)
  ls results/v3/fm/v3abl/n$N/seed20261001/ | wc -l    # ablation: up to 51 (17 x 3)
  ls results/v3/edm/v3/n$N/seed20261001/ | wc -l      # QM9 diffusion: up to 6 (2 x 3)
  grep -h '\[timing\]' logs/v3-*.out | tail -20     # real per-task cost

read it:
  docs/results/V3_RESULTS.md              every base, side by side
  docs/results/V3_RESULTS_fm.md           ours
  docs/results/V3_RESULTS_equifm.md       EquiFM
  docs/results/V3_RESULTS_edm.md          QM9 diffusion
  docs/results/V3_RESULTS_ABL*.md         the same, for the ablation

if the table refuses, it names the gap. Finish the cells and re-run just it:
  sbatch --export=$EXPORTS,STAGE=table $JOB
EOF
