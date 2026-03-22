#!/bin/bash -l
#SBATCH --job-name=python_ex
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --time=0-00:10:00 #DD-HH:MM:SS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load ai/PyTorch/2.3.0-foss-2023b-CUDA-12.6.0

#python -c "import torch; print(torch.cuda.is_available())"
srun tar -xzf $HOME/deep_learning/inaturalist_12K.tar.gz -I gzip -C /tmp
srun python "Test_Scripts_HPC/Simple_HPC_Test_no_torch.py"

export DATA_ROOT=/tmp/inaturalist_12K

# Pytorch: GPU
# Start by simple script printing GPU parameters

# Questions to HPC team:
# 1) Why use srun (parallel execution)? Why not a sequential normal call?
