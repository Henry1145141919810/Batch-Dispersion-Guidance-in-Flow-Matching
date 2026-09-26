#!/bin/bash
# Protocol v3 + its BDG ablation, on BOTH base models, in one paste.
#
#   bash proj1/cluster/submit_v3.sh
#
# Prints the job ids and exits. Nothing needs resubmitting by hand unless a
# stage refuses, and the only stage that can refuse is the table.
#
# THE CHAIN
#   A  v3 headline,  array 0-17   7 arms x 3 properties x 3 seeds x 2 bases,
#                                 n = 2000, tree results/v3/*/n2000
#   B  insurance after A          finishes whatever A's 4-hour window cut off
#   C  v3 ablation,  array 0-5    the 17-arm (eta x tau_mult) grid at n=1000,
#                                 ONE seed, its OWN tree (results/v3/*/n1000)
#   D  insurance after C
#   E  table                      one page per base model, plus both side by side
#
# The ablation is chained AFTER the headline, but the two no longer share a
# tree or a cell: n = 2000 and n = 1000 are separate directories, so there is
# nothing to collide over and nothing to reuse. The ordering now buys only one
# thing -- the headline's [timing] lines land before the ablation commits.
#
# `afterany`, not `afterok`: a task that runs out of its 4-hour window exits
# non-zero with its finished cells on disk, and the next link must still run.
# The table is the one link that refuses rather than warns, so if the run is
# incomplete it exits non-zero and says which seed or cell is missing.
set -uo pipefail
export SLURM_CONF="${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}"

PROJ=/vast/projects/ajw/wharton/hyhuang/cgm
JOB=proj1/cluster/v3_run.slurm
N="${V3_N:-2000}"
# V3_BATCH must travel too. It is BDG's estimator size, not a speed knob: the
# job defaults it to 500 (measured -- one batch of n needs 280 GiB on EquiFM
# against a 45 GB slice) and refuses a value that does not divide N. Leaving it
# out of --export meant an override set on this line silently did nothing.
BATCH="${V3_BATCH:-500}"
# The ablation runs at its OWN n (1000, one seed -- ABLATION_V3_PROTOCOL.md), and
# the job overrides N for STAGE=v3abl. This script must know that too, or it
# checks and reports the wrong number: it printed "n = 5000, 10 controllers" for
# an ablation submission that the job then correctly ran smaller.
#
# ABL_N MUST STAY BELOW N. They are what separates the two trees; at equal n
# both stages write results/v3/<be>/n<N>/ under one stage label and v3_table.py
# refuses (it needs every (prop, arm) at all three seeds, and the ablation has
# 15 arms at one). test_v3.py's ablation_is_smaller_than_headline gates it.
ABL_N="${V3_ABL_N:-1000}"
for _n in "$N" "$ABL_N"; do
    if [ $(( _n % BATCH )) -ne 0 ]; then
        echo "STOP: V3_BATCH=$BATCH does not divide n=$_n." >&2
        echo "A remainder batch is a second, far noisier BDG controller: V_b is" >&2
        echo "estimated over the batch, so a short one is a worse estimator" >&2
        echo "pooled in as an equal." >&2
        exit 1
    fi
done
EXPORTS="ALL,V3_N=$N,V3_ABL_N=$ABL_N,V3_BATCH=$BATCH"

cd "$PROJ" || exit 1
[ -f "$JOB" ] || { echo "STOP: $JOB not found -- ship the code tarball first" >&2; exit 1; }

sub () {                # sub NAME ARRAY DEPENDENCY STAGE -> echoes job id
    local name="$1" arr="$2" dep="$3" stage="$4"
    local cmd=(sbatch --parsable --job-name="cgm-v3-$name")
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
    echo "  headline  the comparison set + BDG eta=4   ~12.8 GPU-h (the default)" >&2
    echo "  ablation  the eta x tau_mult grid          ~4.2 GPU-h" >&2
    echo "  all       both, chained                    ~17.0 GPU-h" >&2
    echo "  table     re-read whatever is on disk" >&2
    exit 1 ;;
esac

# WHY THE DEFAULT IS STILL `headline` AND NOT `all`.
# It is no longer about the 24-hour window: at n = 2000 the whole run is 17.0
# GPU-h (21.2 at the 2.5x margin) and `all` fits 24 h even SERIALIZED. The
# reason is now only that the headline's [timing] lines settle EquiFM's real
# per-cell cost -- the one assumed factor in the budget, and the one that
# decides whether the 1.83x or the 2.5x column is the true one -- before the
# ablation commits. If you already trust the ratio, `all` is fine.
#
# The two stages NO LONGER share a tree (n2000 vs n1000), so the ablation
# recomputes its own eta = 4 rungs rather than skipping the headline's. That
# duplication is priced in: the ablation is 4.2 GPU-h entire.
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
    A=$(sub head   0-17 "$(dep)" v3);    LAST=$A
    B=$(sub head2  ""   "$(dep)" v3);    LAST=$B
fi
if [ "$WHAT" = all ] || [ "$WHAT" = ablation ]; then
    C=$(sub abl    0-5  "$(dep)" v3abl); LAST=$C
    D=$(sub abl2   ""   "$(dep)" v3abl); LAST=$D
fi
E=$(sub table  ""   "$(dep)" table)

cat <<EOF

protocol v3 "$WHAT" submitted, batch = $BATCH
$( if [ "$WHAT" = ablation ]; then
     echo "  ablation at n = $ABL_N, ONE seed -> $(( ABL_N / BATCH )) BDG controllers per cell"
   elif [ "$WHAT" = all ]; then
     echo "  headline n = $N ($(( N / BATCH )) controllers/cell), ablation n = $ABL_N ($(( ABL_N / BATCH )) controllers/cell, 1 seed)"
   elif [ "$WHAT" = headline ]; then
     echo "  headline at n = $N, 3 seeds -> $(( N / BATCH )) BDG controllers per cell"
   fi )

${P:+  $P  preflight               every arm on both bases, and the batch
}${A:+  $A  headline array (0-17)   comparison set + BDG eta=4, both bases
}${B:+  $B  headline insurance
}${C:+  $C  ablation array (0-5)    17-arm eta x tau_mult grid, n=1000, 1 seed
}${D:+  $D  ablation insurance
}  $E  table                   refuses if anything is missing
$( [ "$WHAT" = headline ] && echo "
the ablation is NOT queued. Read the [timing] lines first, then:
  bash proj1/cluster/submit_v3.sh ablation" )

watch it:
  squeue -u \$USER -o "%.18i %.14j %.9T %.10M %.28E"
  ls results/v3/fm/n$N/seed20261001/ | wc -l        # counts up to 66 (22 arms x 3 props)
  ls results/v3/equifm/n$N/seed20261001/ | wc -l    # counts up to 66
  grep -h '\[timing\]' logs/v3-*.out | tail -20     # real per-task cost

read it:
  docs/results/V3_RESULTS.md          both bases, side by side
  docs/results/V3_RESULTS_fm.md       ours
  docs/results/V3_RESULTS_equifm.md   EquiFM

if the table refuses, it names the gap. Finish the cells and re-run just it:
  sbatch --export=$EXPORTS,STAGE=table $JOB
EOF
