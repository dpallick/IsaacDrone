#!/usr/bin/env bash
# Submit an overnight batch: Isaac (GPU) and RotorPy (CPU) training, each with
# 3 seeds, each job training then testing itself. Run from a login node:
#
#   bash oscar/overnight.sh                       # both sims, 3 seeds each
#   SEEDS=0-4 bash oscar/overnight.sh             # 5 seeds
#   ONLY=rotorpy bash oscar/overnight.sh          # just one track
#   MAIL_USER=you@brown.edu bash oscar/overnight.sh   # email when each job ends/fails
#
# In the morning:  python oscar/summarize_runs.py
set -euo pipefail
cd "${REPO_DIR:-$HOME/IsaacDrone}"
mkdir -p oscar/logs

SEEDS="${SEEDS:-0-2}"
ONLY="${ONLY:-all}"
STAMP="$(date +%m%d)"
MAIL=()
[ -n "${MAIL_USER:-}" ] && MAIL=(--mail-type=END,FAIL --mail-user="$MAIL_USER")

# Exploratory accounts get 2 GPUs at once; %2 runs at most 2 Isaac seeds together.
if [ "$ONLY" = "all" ] || [ "$ONLY" = "isaac" ]; then
    RUN_TAG="dd24_hover_${STAMP}" sbatch --array="${SEEDS}%2" "${MAIL[@]}" \
        --export=ALL,RUN_TAG="dd24_hover_${STAMP}" oscar/train_eval_isaac.sbatch
fi
if [ "$ONLY" = "all" ] || [ "$ONLY" = "rotorpy" ]; then
    sbatch --array="${SEEDS}" "${MAIL[@]}" \
        --export=ALL,RUN_TAG="rotorpy_hover_${STAMP}" oscar/train_eval_rotorpy.sbatch
fi
echo
squeue -u "$USER" -o "%.12i %.16j %.8T %.10M %.20R"
echo "Logs: oscar/logs/   Results: runs/*_${STAMP}_s*/eval.json"
