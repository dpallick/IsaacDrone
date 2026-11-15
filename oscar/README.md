# Running IsaacDrone (Isaac Sim 5.0 / Isaac Lab 2.2) on Oscar

Isaac Sim needs Linux + an RTX NVIDIA GPU, so training runs on Oscar inside an
Apptainer container. Everything large (image, caches, pip installs) lives in
`~/scratch`, not your home quota.

```
build_sif.sh             one-time: pull Isaac Sim 5.0 image -> .sif
setup_isaaclab.sbatch    one-time: clone Isaac Lab v2.2.1, install into container python
convert_dd24.sbatch      DD24 URDF -> USD (re-run whenever mass/inertia/mesh change)
train_hover.sbatch       PPO hover training (headless)
common.sh                shared paths + the apptainer wrapper (sourced by the above)
```

## 0. GPU type: check what your account can actually get
Isaac Sim needs RT cores: A6000 / A40 / A5000 / RTX 3090 work. V100 / A100 / H100
do not. Slurm rejects a job whose `--constraint` names a feature that no node has,
so list the real tags first:
```bash
sinfo -p gpu -o "%f" | tr ',' '\n' | sort -u
```
Edit the `#SBATCH --constraint=` line in `train_hover.sbatch` to match, or override
it: `sbatch --constraint="a5000|geforce3090" oscar/train_hover.sbatch`.
(Exploratory accounts get the `gpu` partition. `gpu-he` needs a High-End priority account.)

## 1. One-time setup
```bash
git clone https://github.com/zyan27/IsaacDrone.git ~/IsaacDrone
cd ~/IsaacDrone && mkdir -p oscar/logs

# Isaac Sim image from NGC (free account + API key):
apptainer remote login --username '$oauthtoken' docker://nvcr.io   # paste NGC key
bash oscar/build_sif.sh                 # -> ~/scratch/isaac/isaac-sim-5.0.0.sif (~10+ GB)

sbatch oscar/setup_isaaclab.sbatch      # clones Isaac Lab v2.2.1 + installs it
tail -f oscar/logs/isaaclab_setup_*.out # wait for "Isaac Lab OK"
```

## 2. Build the DD24 asset (mesh + inertia -> URDF -> USD)
`assets/dd24/dd24.urdf` and `meshes/dd24_body.obj` are already generated from the
DD24 CAD and the bifilar inertia measurement. You only need the USD conversion
(`sbatch oscar/convert_dd24.sbatch`). To regenerate after changing values:
```bash
python assets/dd24/prep_cad_mesh.py DD24.stl assets/dd24/meshes/dd24_body.obj 60000   # only if CAD changed
python assets/dd24/build_dd24_urdf.py --mesh assets/dd24/meshes/dd24_body.obj \
    --prop-z 0.018 --collision-box 0.272 0.272 0.107     # mass/inertia/arm from dd24_params.py
```
Then convert on a GPU node:
```bash
sbatch oscar/convert_dd24.sbatch        # -> assets/dd24/dd24.usd
```
Mass, arm length and inertia all come from `environments/dd24_params.py`, so
Isaac and RotorPy always use the same numbers.

## 3. Train
```bash
sbatch oscar/train_hover.sbatch                                   # DD24, 1024 envs
AIRFRAME=crazyflie NUM_ENVS=512 ITERS=500 sbatch oscar/train_hover.sbatch   # sanity run
squeue -u $USER
tail -f oscar/logs/dd24_hover_*.out    # reward, ep_len (should climb toward 1000)
```
At startup the env prints the mass and body inertia that PhysX read from the
USD. Check them against your measurements.

Checkpoints are saved to `checkpoints/<airframe>_hover_<jobid>.pt`. To pull one back:
```bash
scp YOURID@ssh.ccv.brown.edu:~/IsaacDrone/checkpoints/dd24_hover_*.pt ./checkpoints/
```

## Recommended first run
Start with a 30-minute `AIRFRAME=crazyflie ITERS=200` job. It checks the container,
the Isaac Lab install, and Nucleus asset access (the Crazyflie mesh and ground
plane download on first run) before you spend an 8-hour allocation.

## Troubleshooting
* **Job rejected / "Invalid feature specification"**: the constraint names a GPU
  tag that doesn't exist. See step 0.
* **First run hangs at asset loading**: compute nodes need outbound HTTPS to
  NVIDIA's asset server for the Crazyflie USD and ground plane. Run once
  interactively (`interact -q gpu -g 1 -t 1:00:00`) to warm `~/scratch/isaac/cache`.
* **`No module named isaaclab`** during training: the setup job's user install
  didn't land in `$PYUSER`. Check its log for `user site enabled: False`. If
  you see that, ask me for the overlay-image variant.
* **`apptainer: command not found`**: CCV ships Apptainer natively on compute
  nodes. Make sure you are on a compute node, not a login node.
