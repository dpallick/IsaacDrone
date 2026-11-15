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
    PY=/isaac-sim/python.sh
    \$PY -c 'import site,sys; print(\"user site enabled:\", site.ENABLE_USER_SITE, site.getusersitepackages())'
    # isaaclab.sh tries 'sudo apt-get install cmake' unless cmake is already on PATH.
    # No sudo on Oscar, so install cmake from pip into \$PYUSER/bin (on PATH via common.sh).
    \$PY -m pip install --user --quiet cmake
    echo \"cmake: \$(command -v cmake)\"
    echo \"torch seen by pip: \$(\$PY -m pip show torch 2>/dev/null | grep -E '^(Version|Location)' | tr '\\n' ' ')\"
    # isaaclab.sh UNINSTALLS any torch that isn't exactly 2.7.0+cu128 - impossible in
    # the read-only image. Put that exact build in the user site first (it shadows the
    # image's copy), so the installer finds it and leaves things alone.
    TV=\$(\$PY -m pip show torch 2>/dev/null | awk '/^Version:/{print \$2}')
    if [ \"\$TV\" != \"2.7.0+cu128\" ]; then
        echo \"Installing torch 2.7.0+cu128 into the user site (found: \${TV:-none})\"
        \$PY -m pip install --user torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    fi
    cd '$ISAACLAB_DIR'
    # 'none' = core Isaac Lab only (we bring our own PPO); much faster install.
    ./isaaclab.sh --install none
    ./isaaclab.sh -p -c 'import isaaclab, isaaclab_assets, torch; print(\"Isaac Lab OK:\", isaaclab.__file__, \"| torch\", torch.__version__, \"| cuda\", torch.cuda.is_available())'
"
echo "Isaac Lab installed into $PYUSER"
