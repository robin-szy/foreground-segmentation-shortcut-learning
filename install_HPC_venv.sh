#!/bin/bash -l
#SBATCH --job-name=setup_env
#SBATCH --time=01:00:00
#SBATCH --output=setup_env.out
#SBATCH --error=setup_env.err

module --force purge
module load env/development/2025a
module load lang/Python/3.13.1-GCCcore-14.2.0

python --version
which python
module list

python -m venv $HOME/deep_learning/DL_env_latest
source $HOME/deep_learning/DL_env_latest/bin/activate

python -m pip install --upgrade pip setuptools wheel

pip install pandas Pillow numpy matplotlib opencv-python-headless scikit-learn

pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip
pip install git+https://github.com/openai/CLIP.git

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
PY
