#!/usr/bin/env bash
# Train (resumable) + evaluate an Isaac Lab hover policy. One run per array task:
# the array index is the random seed, so --array=0-2 trains 3 seeds.
#
#   sbatch --array=0-2 oscar/train_eval_isaac.sbatch
#   RUN_TAG=dd24_long ITERS=6000 sbatch --array=0-2 oscar/train_eval_isaac.sbatch
#
# If training hits the walltime before ITERS, the job resubmits itself (up to
# RESUBMITS times) and continues from runs/<tag>_s<seed>/latest.pt.
#SBATCH --job-name=isaac_hover
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a5000|geforce3090|a6000|a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=oscar/logs/%x_%A_%a.out
#SBATCH --error=oscar/logs/%x_%A_%a.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"

SEED="${SLURM_ARRAY_TASK_ID:-${SEED:-0}}"
AIRFRAME="${AIRFRAME:-dd24}"
RUN_TAG="${RUN_TAG:-${AIRFRAME}_hover}"
RUN_DIR="runs/${RUN_TAG}_s${SEED}"
NUM_ENVS="${NUM_ENVS:-1024}"
ITERS="${ITERS:-3000}"
STEPS="${STEPS:-128}"
EVAL_ENVS="${EVAL_ENVS:-512}"
RESUBMITS="${RESUBMITS:-2}"
# Leave ~25 min of the walltime for shutdown + evaluation.
WALL_MIN="${WALL_MIN:-600}"
TRAIN_MIN=$(( WALL_MIN - 25 ))

cd "$REPO_DIR"
mkdir -p "$RUN_DIR" oscar/logs
echo "== $(date) job ${SLURM_JOB_ID:-local} seed $SEED -> $RUN_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---- train ----------------------------------------------------------------
isaac_exec "cd '$REPO_DIR' && '$ISAACLAB_DIR/isaaclab.sh' -p train_rl_hover.py --headless \
    --airframe $AIRFRAME --num_envs $NUM_ENVS --iters $ITERS --steps $STEPS \
    --seed $SEED --run_dir $RUN_DIR --resume --time_limit_min $TRAIN_MIN"

# ---- not finished? resubmit to continue ----------------------------------
if [ ! -f "$RUN_DIR/DONE" ]; then
    if [ "$RESUBMITS" -gt 0 ]; then
        echo "Training not finished - resubmitting (RESUBMITS left: $((RESUBMITS-1)))"
        sbatch --export=ALL,SEED="$SEED",RUN_TAG="$RUN_TAG",RESUBMITS=$((RESUBMITS-1)) \
            --job-name="${SLURM_JOB_NAME:-isaac_hover}" "$REPO_DIR/oscar/train_eval_isaac.sbatch"
    else
        echo "Out of resubmits; evaluating the best checkpoint so far."
    fi
fi

# ---- test -----------------------------------------------------------------
if [ -f "$RUN_DIR/best.pt" ]; then
    isaac_exec "cd '$REPO_DIR' && '$ISAACLAB_DIR/isaaclab.sh' -p eval_rl_hover.py --headless \
        --checkpoint $RUN_DIR/best.pt --num_envs $EVAL_ENVS --out $RUN_DIR/eval.json"
    # Worst case: the ToFs miss the net 70% of the time.
    isaac_exec "cd '$REPO_DIR' && '$ISAACLAB_DIR/isaaclab.sh' -p eval_rl_hover.py --headless \
        --checkpoint $RUN_DIR/best.pt --num_envs $EVAL_ENVS --net_dropout 0.7 \
        --out $RUN_DIR/eval_net70.json"
fi
echo "== $(date) done"
