INSTALL INSTRUCTIONS


Create and install conda env:

conda create -n droneIsaac5.0 python=3.11
conda activate droneIsaac5.0


Install cuda toolkit:
conda install cuda

pytorch for isaacsim/lab :
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

Curobo: 
git clone https://github.com/NVlabs/curobo.git
pip install -e ./curobo --no-build-isolation

isaaclab : 
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

isaacsim:
pip install 'isaacsim[all,extscache]==5.0.0' --extra-index-url https://pypi.nvidia.com