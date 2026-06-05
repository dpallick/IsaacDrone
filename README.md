# IsaacDrone

Simulating drones in isaaclab

## Requirements (with these instal instructions)

* Linux (Ubuntu recommended)
* NVIDIA GPU

---

# Installation

## 1. Create an Environment

```bash
conda create -n droneIsaac5.0 python=3.11
conda activate droneIsaac5.0
```

---

## 2. Install CUDA

```bash
conda install cuda
```

---

## 3. Install PyTorch

Install the CUDA 12.8 build used by Isaac Sim 5.0.

```bash
pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

---

## 4. Install cuRobo

Clone the repo then install

```bash
git clone https://github.com/NVlabs/curobo.git

pip install -e ./curobo --no-build-isolation
```

---

## 5. Install Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git

cd IsaacLab

./isaaclab.sh --install
```

---

## 6. Install Isaac Sim

```bash
pip install "isaacsim[all,extscache]==5.0.0" \
    --extra-index-url https://pypi.nvidia.com
```

---

# Running

Activate environment:

```bash
conda activate droneIsaac5.0
```

Run scripts :

```bash
python main.py
```

# File Organization (without curobo / isaaclab)

```text
IsaacDrone/
├── controllers/
│   ├── __init__.py
│   └── pid_controller.py
├── environments/
│   ├── __init__.py
│   └── drone_hover_env.py
├── main.py
└── README.md
```

## Components

### Environment

`environments/drone_hover_env.py`

Contains the drone simulation environment, physics model, observations, rewards, and reset logic.

### Controller

`controllers/pid_controller.py`

Contains PID-based drone control for hovering and waypoint tracking.

### Main

`main.py`

Launches Isaac Lab, loads the environment, initializes the controller, and runs the simulation loop.

---

