#!/bin/bash
#SBATCH --job-name=cgm-bdg
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdg-%A_%a.out
#SBATCH --error=/vast/projects/pranam/lab/boboli/cis6270-project1-group2/logs/bdg-%A_%a.err
#
# BDG: two-sided setpoint control of the ACROSS-BATCH variance of the predicted
# property. 39 cells = the ablation ladder plus its MATCHED controls.
#
# ONE CELL PER TASK, 4 cores each. Same finding as chem_budget_cpu2: this
# model's tensors are too small for torch intra-op parallelism to engage, so
# threads buy nothing and the parallelism has to be across CELLS. That turns a
# ~8 h serial walk into ~20-30 min of wall clock.
#
# CPU, NOT GPU, ON PURPOSE. b200-mig45 had 46 eligible jobs against 32 slots;
# genoa-std-mem had 27 idle nodes. Speed, not availability. Running every cell
# on one device also keeps the controls device-matched with the arm.
#
# WHY THE CONTROLS ARE IN THIS JOB. results/sweep* was produced with fm_last.pt
# on a B200 GPU at batch=128; this runs weights/fm_ema.pt on CPU at batch=512.
# The WEIGHTS are the same -- fm_ema carries source_md5 = the sweep's fm_md5,
# and load_fm reads the same "ema" tensors from both layouts (README: max
# |w_slim - w_full| = 0.0). Measured drift over 18 matched pairs: in_band
# +0.0012 +/- 0.0129, below independent-draw noise. So mixing IS allowed in a
# comparison table. The in-job controls exist for the FLOOR calls: mol_stability
# drifts -0.008, enough to flip cells that sit near FR3a = 0.362109.
#
# LADDER (w fixed at 1.0 so only the dispersion knobs move):
#   e0t1                eta = 0  -> must reproduce plug BIT-IDENTICALLY
#   e{1,4}t0.5          contract below the natural spread
#   e{1,4}t1.0          hold: e = 0 at the setpoint, the fixed point
#   e{1,4}t1.5          WIDEN -- the branch no arm in this repo can reach
#   e4t0.5o / e4t1.5o   the one-sided (BTVG contract-only) restriction
# tau = tau_mult * s, s = f_A.y_std, which sits within ~7% of the unguided
# population sd, so tau_mult reads as "fraction of the natural spread".
set -uo pipefail
PROJ=/vast/projects/pranam/lab/boboli/cis6270-project1-group2
cd "$PROJ" || exit 1
source /vast/projects/pranam/lab/boboli/diffusion_qm9_training/.venv/bin/activate || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# (prop arm w variant) -- exactly plan_bdg_cells' order, controls first
CELLS=(
 "mu unguided 1 bdgref"    "mu plug 1 bdgref"      "mu bdg 1 e0t1"
 "alpha unguided 1 bdgref" "alpha plug 1 bdgref"   "alpha bdg 1 e0t1"
 "gap unguided 1 bdgref"   "gap plug 1 bdgref"     "gap bdg 1 e0t1"
 "mu bdg 1 e1t0.5"    "mu bdg 1 e4t0.5"    "mu bdg 1 e1t1"
 "mu bdg 1 e4t1"      "mu bdg 1 e1t1.5"    "mu bdg 1 e4t1.5"
 "alpha bdg 1 e1t0.5" "alpha bdg 1 e4t0.5" "alpha bdg 1 e1t1"
 "alpha bdg 1 e4t1"   "alpha bdg 1 e1t1.5" "alpha bdg 1 e4t1.5"
 "gap bdg 1 e1t0.5"   "gap bdg 1 e4t0.5"   "gap bdg 1 e1t1"
 "gap bdg 1 e4t1"     "gap bdg 1 e1t1.5"   "gap bdg 1 e4t1.5"
 "mu bdg 1 e4t0.5o"   "mu bdg 1 e4t1.5o"
 "alpha bdg 1 e4t0.5o" "alpha bdg 1 e4t1.5o"
 "gap bdg 1 e4t0.5o"  "gap bdg 1 e4t1.5o"
 "mu plug 0.25 bdgref"  "mu plug 4 bdgref"
 "alpha plug 0.25 bdgref" "alpha plug 4 bdgref"
 "gap plug 0.25 bdgref" "gap plug 4 bdgref"
 # THE SCIENTIFIC CORE: V_b (across-batch, survives) vs V_F (per-sample, vanishes
 # as k=(1-t)^2/t -> 0). btvg_var is the V_F arm, re-run on THIS checkpoint so
 # the one comparison the paper turns on is not a cross-checkpoint artefact.
 "mu btvg_var 1 tau1"    "mu btvg_var 4 tau1"
 "alpha btvg_var 1 tau1" "alpha btvg_var 4 tau1"
 "gap btvg_var 1 tau1"   "gap btvg_var 4 tau1"
)
T=${SLURM_ARRAY_TASK_ID:-0}
read -r P A W V <<< "${CELLS[$T]}"
echo "task $T -> prop=$P arm=$A w=$W variant=$V  host=$(hostname)"; date -Is

python -u proj1/scripts/guidance_sweep.py --stage bdg \
    --fm weights/fm_ema.pt \
    --props "$P" --only-arm "$A" --only-w "$W" --only-variant "$V" \
    --n 512 --batch 512 --steps 100 --n-probe 1 \
    --seed 20260921 --device cpu --per-mol \
    --out-dir "$PROJ/results/bdg" \
    --max-minutes 220
echo "finished $(date -Is)"
