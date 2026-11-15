#!/usr/bin/env bash
# ONE-TIME: clone Isaac Lab (pinned) and install it into the container's Python.
# Run after build_sif.sh:   sbatch oscar/setup_isaaclab.sbatch
#SBATCH --job-name=isaaclab_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=oscar/logs/%x_%j.out
#SBATCH --error=oscar/logs/%x_%j.err
set -euo pipefail
source "${REPO_DIR:-$HOME/IsaacDrone}/oscar/common.sh"

if [ ! -d "$ISAACLAB_DIR/.git" ]; then
    git clone --depth 1 --branch "$ISAACLAB_TAG" \
        https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
fi
ln -sfn /isaac-sim "$ISAACLAB_DIR/_isaac_sim"   # point Isaac Lab at the in-container sim

isaac_exec "
    set -e
    /isaac-sim/python.sh -c 'import site,sys; print(\"user site enabled:\", site.ENABLE_USER_SITE, site.getusersitepackages())'
    cd '$ISAACLAB_DIR'
    # 'none' = core Isaac Lab only (we bring our own PPO); much faster install.
    ./isaaclab.sh --install none
    ./isaaclab.sh -p -c 'import isaaclab, isaaclab_assets; print(\"Isaac Lab OK:\", isaaclab.__file__)'
"
echo "Isaac Lab installed into $PYUSER"
